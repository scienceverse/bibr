"""Read the paper's own printed citation ("Citation: ...", "To cite this article: ...").

Many papers print on their first page how to cite them. The DOI in that line
already counts as the paper's own (``bibr.extract.doi_identity``), and the line
is kept out of the title and abstract candidates; this module reads the rest of
it. It works in two steps, because the reference parser and the paper's own DOI
are not known at the same point of the pipeline:

1. :func:`read_printed_citation` (post-parse, after the reference list was
   parsed, while the reference parser is loaded) finds the line on page 1 and
   parses it with the reference parser, through the code that parses one entry
   of the reference list (``ReferenceExtractor.parse_printed_reference``). The
   result rides ``PaperContents.printed_citation``.
2. :func:`apply_printed_citation` (identity stage, once the paper's DOI is
   selected) decides whether the parse is about this paper, fills the
   metadata fields the extraction left empty, and records what it did and
   whether the line agrees with the extraction on the fields both have
   (``PaperContents.printed_citation_receipt``, exported as
   ``extraction.diagnostics.self_citation``).

The line is about this paper when the DOI it prints is the paper's DOI; failing
a DOI comparison, when its title matches the extracted title; failing both,
when its first author's family name is one of the extracted authors' family
names; with nothing to compare, only when its label names this article ("To
cite this article:", "How to cite this paper:", "Full citation:"), not a bare
"Citation:". It never overwrites a value: it fills the journal, volume, issue,
pages and publication year when they are empty, and the title and the authors
only when the extraction found none. ``SELF_CITATION_METADATA`` turns it off.
"""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bibr.paper_contents import CanonicalSection
from bibr.processing_warnings import ProcessingWarning, WarningCode
from bibr.utils.printed_locator import read_prefixed_locator
from bibr.utils.text import collapse_ws, normalize_doi

if TYPE_CHECKING:
    from bibr.models import PaperReference

logger = logging.getLogger(__name__)

# Labels that name this article. The form with "this <work>" is the article's
# own citation block wherever the page prints it; "how to cite this report" (a
# referee report) names another work and is left out.
_STRONG_LABEL = (
    r"(?:please\s+)?(?:to\s+|how\s+to\s+)?cite\s+this\s+"
    r"(?:article|paper|preprint|chapter|version|work|manuscript)(?:\s+as)?"
    r"|full\s+citation"
    r"|pour\s+citer\s+cet\s+(?:article|ouvrage|document)"
    r"|(?:para\s+citar|c[oó]mo\s+citar)\s+est[ea]\s+(?:art[ií]culo|trabajo|documento)"
    r"|como\s+citar\s+este\s+(?:artigo|trabalho)"
    r"|come\s+citare\s+questo\s+articolo"
)
# Labels that introduce a citation without saying whose: on the first page it
# is mostly the article's own, but a dataset or companion paper is printed the
# same way.
_WEAK_LABEL = (
    r"(?:recommended|suggested|preferred)\s+citation"
    r"|citations?"
    r"|(?:please\s+|to\s+|how\s+to\s+)?cite(?:\s+(?:this|as))?"
    r"|para\s+citar|c[oó]mo\s+citar|pour\s+citer|zitierweise|для\s+цитирования|цитування"
)
# A label opens the row (after a bullet or a bar) and is followed by a colon or
# a bar, or ends the row: then the citation is the next row.
_STRONG_CUE_RE = re.compile(rf"^\W*(?:{_STRONG_LABEL})\s*(?:[:|.\-–—]\s*|\s+|$)", re.IGNORECASE)
_WEAK_CUE_RE = re.compile(rf"^\W*(?:{_WEAK_LABEL})\s*(?:[:|]\s*|$)", re.IGNORECASE)
# "Citation Westcot SE, ..." in small capitals, with no colon.
_BARE_CITATION_CUE_RE = re.compile(r"^\W*citation\s+(?=(?-i:[A-ZÀ-Ý]))", re.IGNORECASE)

# Rows that end the citation block: the dates, licence and correspondence lines
# printed beside it, and the cover page's other items.
_STOP_RE = re.compile(
    r"^\s*(?:©|\(c\)\s*\d{4}|[^\w\s]{0,2}\s*(?:received|accepted|revised|published|"
    r"available\s+online|first\s+published|copyright|correspondence|corresponding\s+author|"
    r"e-?mail|edited\s+by|reviewed\s+by|academic\s+editor|handling\s+editor|editor|"
    r"keywords?|key\s+words|abstract|licen[cs]e|open\s+access|funding|competing\s+interests|"
    r"data\s+availability|to\s+link\s+to\s+this|link\s+to\s+this|view\s+the\s+article|"
    r"view\s+related|view\s+crossmark|submit\s+your\s+article|article\s+views|full\s+terms|"
    r"this\s+article\s+(?:is|was|has)|issn)\b)",
    re.IGNORECASE,
)
# A citation ends with its DOI or link.
_DOI_OR_URL_RE = re.compile(r"\b10\.\d{4,9}/\S|https?://\S|\bwww\.\S", re.IGNORECASE)

# Bounds of the joined citation: a few rows of one paragraph, and well inside
# the reference parser's input window.
_MAX_ROWS = 4
_MAX_CHARS = 700
_MIN_CHARS = 20
# Closer than this (on normalised text) is the same title.
_TITLE_RATIO = 0.9
_MAX_FILLED_AUTHORS = 100

_SKIPPED_SECTION_TYPES = frozenset({CanonicalSection.REFERENCES})
_SKIPPED_SYNTHETIC_KINDS = frozenset({"figure", "table"})

# The metadata fields the line may fill, as exported.
LOCATOR_FIELDS = ("volume", "issue", "first_page", "last_page")


@dataclass(frozen=True)
class PrintedCitation:
    """The paper's own citation line as found on its first page, and its parse."""

    text_ids: tuple[int, ...]
    label: str
    strong: bool
    text: str
    # None when the parser read neither a title nor authors in it.
    reference: PaperReference | None = None


# ── Finding the line ────────────────────────────────────────────────────


def _cue(text: str) -> tuple[re.Match[str] | None, bool]:
    strong = _STRONG_CUE_RE.match(text)
    if strong is not None:
        return strong, True
    weak = _WEAK_CUE_RE.match(text) or _BARE_CITATION_CUE_RE.match(text)
    return weak, False


def _first_page_rows(contents) -> list[Any]:
    sections = {section.section_id: section for section in contents.sections}
    rows = []
    for sentence in contents.sentences:
        if sentence.page_number != 1 or sentence.is_display_formula or not sentence.text:
            continue
        section = sections.get(sentence.section_id)
        if section is not None and (
            section.section_type in _SKIPPED_SECTION_TYPES
            or section.synthetic_kind in _SKIPPED_SYNTHETIC_KINDS
        ):
            continue
        rows.append(sentence)
    return rows


def find_printed_citation(contents) -> PrintedCitation | None:
    """The first page's own citation line, joined across the rows it was split into.

    The first row with a strong label is taken, else the first with a weak one.
    A label alone on its row takes the next row. The line then runs on through
    its paragraph until a row prints its DOI or link, a row opens with another
    label (a date, the licence, a link), or the bounds are reached.
    """
    rows = _first_page_rows(contents)
    found: tuple[int, re.Match[str], bool] | None = None
    for index, sentence in enumerate(rows):
        match, strong = _cue(sentence.text)
        if match is None:
            continue
        if strong:
            found = (index, match, strong)
            break
        if found is None:
            found = (index, match, strong)
    if found is None:
        return None
    index, match, strong = found
    cue_row = rows[index]
    remainder = cue_row.text[match.end() :].lstrip(" \t:|.-–—").strip()
    parts = [remainder] if remainder else []
    text_ids = [cue_row.text_id]
    previous = cue_row
    for sentence in rows[index + 1 :]:
        joined = " ".join(parts)
        if len(text_ids) >= _MAX_ROWS or (joined and _DOI_OR_URL_RE.search(joined)):
            break
        if parts and (sentence.section_id, sentence.paragraph_id) != (
            previous.section_id,
            previous.paragraph_id,
        ):
            break
        text = sentence.text.strip()
        if _STOP_RE.match(text) or _cue(text)[0] is not None:
            break
        if len(joined) + len(text) + 1 > _MAX_CHARS:
            break
        parts.append(text)
        text_ids.append(sentence.text_id)
        previous = sentence
    text = " ".join(parts).strip()
    if len(text) < _MIN_CHARS:
        return None
    label = cue_row.text[: match.end()].strip()
    return PrintedCitation(tuple(text_ids), label, strong, text[:_MAX_CHARS])


def read_printed_citation(
    contents,
    *,
    file_hash: str,
    llm_client,
    settings,
    memory_mode: str | None = None,
) -> PrintedCitation | None:
    """Find the paper's own citation line and parse it with the reference parser.

    Runs in post-parse after the reference list was parsed, with the parser
    that parsed it (the ``ner`` strategy). The result is stored on
    ``contents.printed_citation`` and returned. A parse that fails leaves the
    line unparsed; an overridden parser that keeps failing still stops the run.
    """
    from bibr.exceptions import ModelOverrideError
    from bibr.extract.ref_extractor import ReferenceExtractor

    line = find_printed_citation(contents)
    if line is None:
        contents.printed_citation = None
        return None
    extractor = ReferenceExtractor(
        contents,
        file_hash=file_hash,
        llm_client=llm_client,
        parse_strategy="ner",
        settings=settings,
        memory_mode=memory_mode,
    )
    try:
        reference = extractor.parse_printed_reference(line.text)
    except ModelOverrideError:
        raise
    except Exception:  # noqa: BLE001 - a second source must never break extraction
        logger.warning("Could not parse the paper's printed citation", exc_info=True)
        reference = None
    line = PrintedCitation(line.text_ids, line.label, line.strong, line.text, reference)
    contents.printed_citation = line
    return line


# ── Comparing ───────────────────────────────────────────────────────────


def _plain(text: str | None) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[^\W_]+", stripped.casefold()))


def same_title(left: str | None, right: str | None) -> bool:
    """Whether two titles are the same once case, accents and punctuation are set aside."""
    a, b = _plain(left), _plain(right)
    if not a or not b:
        return False
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= _TITLE_RATIO


def _family_keys(family: str | None) -> set[str]:
    key = _plain(family)
    if not key:
        return set()
    return {key, key.split()[-1]}


def _same_family(left: str | None, right: str | None) -> bool:
    return bool(_family_keys(left) & _family_keys(right))


_HONORIFIC_RE = re.compile(r"^(?:dr|prof|mr|mrs|ms)\.?$", re.IGNORECASE)
# "A.", "S. A.", "V.M.", "J.-P.", "SE", "CRL", "MÁ": initials, as a reference
# list prints them after or before a family name.
_INITIALS_RE = re.compile(r"^(?:[^\W\d_]\.?\s*-?\s*){1,4}$")
_PARTICLES = frozenset(
    [
        "van",
        "der",
        "den",
        "von",
        "de",
        "del",
        "della",
        "di",
        "da",
        "das",
        "dos",
        "du",
        "la",
        "le",
        "el",
        "al",
        "ten",
        "ter",
        "bin",
        "ben",
    ]
)
_ET_AL_RE = re.compile(r"[,;]?\s*(?:\.{2,}|…)?\s*\bet\.?\s*al\b.*$", re.IGNORECASE | re.DOTALL)


def _is_initials(token: str) -> bool:
    letters = [char for char in token if char.isalpha()]
    return bool(
        letters
        and _INITIALS_RE.match(token.strip())
        and all(char.isupper() for char in letters)
        and (len(letters) <= 4)
    )


def _name_token_ok(token: str) -> bool:
    if token.casefold() in _PARTICLES:
        return True
    return bool(token[:1].isupper() and not any(char.isdigit() for char in token)) and not any(
        char in token for char in "*@#()[]{}<>=+/"
    )


def _single_name(part: str) -> tuple[str, str] | None:
    """(given, family) of one name printed on its own: "Westcot SE", "Xiao Li", "A. Smith"."""
    tokens = [token for token in part.split() if not _HONORIFIC_RE.match(token)]
    if len(tokens) < 2 or len(tokens) > 6 or not all(_name_token_ok(t) for t in tokens):
        return None
    if _is_initials(tokens[-1]) and not _is_initials(tokens[0]):
        # "Westcot SE", "De la Casa LG"
        given, family = tokens[-1], " ".join(tokens[:-1])
    elif _is_initials(tokens[0]) and not _is_initials(tokens[-1]):
        # "A. Smith", "J.-P. Sartre"
        given, family = tokens[0], " ".join(tokens[1:])
    elif not any(_is_initials(token) for token in tokens):
        # "Hasan Borke Birgin", "Ludwig van Beethoven"
        start = len(tokens) - 1
        while start > 1 and tokens[start - 1].casefold() in _PARTICLES:
            start -= 1
        given, family = " ".join(tokens[:start]), " ".join(tokens[start:])
    else:
        return None
    family = family.strip(" ,.")
    if sum(char.isalpha() for char in family) < 2 or "." in family:
        return None
    return given.strip(" ,"), family


def split_printed_authors(authors: str | None) -> tuple[list[tuple[str, str] | None], bool]:
    """The names of a citation's author string, in order, and whether it ends in "et al.".

    Reads "Family, I., Family, I. & Family, I.", "Family I, Family I and
    Family I", "Family, I.; Family, I." and "Given Family, Given Family". A
    name it cannot read is None, so a caller can refuse a list it read only
    in part.
    """
    if not authors or not authors.strip():
        return [], False
    text = authors.strip()
    et_al = bool(re.search(r"\bet\.?\s*al\b", text, re.IGNORECASE))
    text = _ET_AL_RE.sub("", text).strip(" ,;")
    if not text:
        return [], et_al
    names: list[tuple[str, str] | None] = []
    if ";" in text:
        for part in (p.strip(" ,") for p in text.split(";")):
            if not part:
                continue
            if "," in part:
                family, _, given = part.partition(",")
                family, given = family.strip(), given.strip()
                ok = (
                    family
                    and given
                    and all(_name_token_ok(t) for t in family.split())
                    and "." not in family
                    and sum(char.isalpha() for char in family) >= 2
                )
                names.append((given, family) if ok else None)
            else:
                names.append(_single_name(part))
        return names, et_al
    text = re.sub(r"\s*(?:&|\band\b)\s*", ", ", text)
    parts = [part.strip() for part in text.split(",") if part.strip()]
    paired = (
        len(parts) >= 2
        and len(parts) % 2 == 0
        and all(_is_initials(part) for part in parts[1::2])
        and not any(_is_initials(part) for part in parts[0::2])
    )
    if paired:
        for family, given in zip(parts[0::2], parts[1::2], strict=True):
            ok = (
                all(_name_token_ok(t) for t in family.split())
                and "." not in family
                and sum(char.isalpha() for char in family) >= 2
            )
            names.append((given, family) if ok else None)
        return names, et_al
    return [_single_name(part) for part in parts], et_al


def _first_family(authors: str | None) -> str | None:
    """The first author's family name in a citation's author string."""
    names, _ = split_printed_authors(authors)
    if names and names[0] is not None:
        return names[0][1]
    from bibr.structure.citation_matcher import extract_families

    families = extract_families(_ET_AL_RE.sub("", authors or ""))
    return families[0] if families else None


def _year_of(published: str | None) -> int | None:
    from bibr.export.normalize import iso_date

    iso = iso_date(published)
    return int(iso[:4]) if iso else None


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    # Line breaks inside a printed value ("Brain\r\nSci") are the layout's, not the text's.
    text = collapse_ws(collapse_ws(str(value)).strip(" ,;:."))
    return text or None


def _printed_pages(reference: PaperReference, line_text: str) -> tuple[str | None, str | None]:
    """The first and last page the line prints.

    An abbreviated prefixed range ("S163-9") is expanded ("S163", "S169"); an
    article number ("e00376-18") is a first page alone, as the line prints it.
    """
    first = _clean(reference.first_page)
    last = _clean(reference.last_page)
    pair = read_prefixed_locator(first, last) if first and last else None
    if pair is None:
        return first, last
    if pair.last is None:
        printed = re.search(
            rf"(?<![\w-]){re.escape(first)}\s*[-\u2010-\u2015]\s*{re.escape(last)}(?![\w-])",
            line_text,
        )
        return (re.sub(r"\s+", "", printed.group()) if printed else pair.first), None
    return pair


def _same_locator(left: Any, right: Any) -> bool:
    return _plain(str(left)) == _plain(str(right))


# ── Deciding and filling ────────────────────────────────────────────────


@dataclass(frozen=True)
class _Comparison:
    agreement: dict[str, str]
    check: str
    accepted: bool


def _compare(metadata, line: PrintedCitation, reference: PaperReference) -> _Comparison:
    agreement: dict[str, str] = {}

    def note(name: str, agrees: bool) -> None:
        agreement[name] = "agree" if agrees else "disagree"

    own_doi = normalize_doi(metadata.doi)
    line_doi = normalize_doi(reference.doi)
    doi_agrees = None
    if own_doi and line_doi:
        doi_agrees = own_doi.casefold() == line_doi.casefold()
        note("doi", doi_agrees)

    own_title = metadata.title or ""
    line_title = _clean(reference.title)
    title_agrees = None
    if own_title.strip() and line_title:
        title_agrees = same_title(own_title, line_title)
        note("title", title_agrees)

    first_family = _first_family(reference.authors)
    families = [author.family for author in metadata.authors if author.family]
    author_found = None
    if first_family and families:
        note("first_author", _same_family(first_family, families[0]))
        author_found = any(_same_family(first_family, family) for family in families)

    own_year = _year_of(metadata.published)
    if own_year is not None and reference.year:
        note("year", own_year == reference.year)
    line_volume = _clean(reference.volume)
    if metadata.volume and line_volume:
        note("volume", _same_locator(metadata.volume, line_volume))

    if doi_agrees is not None:
        check, accepted = ("doi_match", True) if doi_agrees else ("doi_mismatch", False)
    elif title_agrees is not None:
        check, accepted = ("title_match", True) if title_agrees else ("title_mismatch", False)
    elif author_found is not None:
        check, accepted = (
            ("first_author_match", True) if author_found else ("first_author_mismatch", False)
        )
    elif line.strong:
        check, accepted = "strong_cue", True
    else:
        check, accepted = "weak_cue", False
    return _Comparison(agreement, check, accepted)


def _abstained(paper) -> bool:
    return any(
        issue.code == "VAL_METADATA_MULTI_ITEM" and issue.blocking
        for issue in paper.validation_issues
    )


def _decided_rule(metadata, name: str) -> str | None:
    from bibr.extract.field_decisions import field_decisions_of

    ledger = field_decisions_of(metadata)
    decision = ledger.get(name) if ledger is not None else None
    return decision.rule if decision is not None else None


def _filled_authors(reference: PaperReference) -> list | None:
    from bibr.models import PaperAuthor

    names, et_al = split_printed_authors(collapse_ws(reference.authors or ""))
    # A list cut short by "et al.", or one with a name the reader could not
    # split, would be a wrong author list, not a partial one.
    if et_al or not names or None in names or len(names) > _MAX_FILLED_AUTHORS:
        return None
    return [
        PaperAuthor(author_id=position, given=given, family=family, affiliation="")
        for position, (given, family) in enumerate(
            (name for name in names if name is not None), start=1
        )
    ]


def _fill(metadata, reference: PaperReference, line: PrintedCitation) -> list[str]:
    """Fill the empty fields from *reference*; the names of the fields filled, as exported."""
    from bibr.extract.field_decisions import FieldCandidate, fill_empty

    evidence = tuple(f"text:{text_id}" for text_id in line.text_ids)
    filled: list[str] = []

    def decided(name: str, value: Any, exported: str) -> None:
        candidate = FieldCandidate(name, "self_citation", value, evidence_ids=evidence)
        if fill_empty(metadata, candidate, "self_citation_fill", "the paper's printed citation"):
            filled.append(exported)

    title = _clean(reference.title)
    if not (metadata.title or "").strip() and title and len(title.split()) >= 2:
        decided("title", title, "title")
    if not metadata.authors and _decided_rule(metadata, "author") != "correction_notice":
        authors = _filled_authors(reference)
        if authors:
            decided("author", authors, "author")
    journal = _clean(reference.container)
    if not (metadata.journal or "").strip() and journal:
        decided("journal", journal, "journal")
    if not (metadata.published or "").strip() and reference.year:
        decided("published", str(reference.year), "published")

    # The volume, issue and pages go together: when the line and the record
    # disagree on one they both carry, it names another version, and none is
    # taken from it.
    first_page, last_page = _printed_pages(reference, line.text)
    printed = {
        "volume": _clean(reference.volume),
        "issue": _clean(reference.issue),
        "first_page": first_page,
        "last_page": last_page,
    }
    current = {name: getattr(metadata, name, None) for name in LOCATOR_FIELDS}
    consistent = all(
        _same_locator(current[name], printed[name])
        for name in LOCATOR_FIELDS
        if current[name] and printed[name]
    )
    if not consistent:
        return filled
    # One producer each, not decided fields: written here directly.
    fills = {name: printed[name] for name in LOCATOR_FIELDS if printed[name] and not current[name]}
    if "volume" in fills:
        metadata.volume = fills["volume"]
    if "issue" in fills:
        metadata.issue = fills["issue"]
    if "first_page" in fills:
        metadata.first_page = fills["first_page"]
    if "last_page" in fills:
        metadata.last_page = fills["last_page"]
    filled.extend(fills)
    return filled


def apply_printed_citation(paper) -> dict | None:
    """Fill the paper's empty metadata from its printed citation, when it is about this paper.

    Runs in the identity stage once ``metadata.doi`` is selected, on what
    :func:`read_printed_citation` stored. Records the receipt on
    ``contents.printed_citation_receipt`` (once: a second call returns it
    unchanged) and returns it; None when no line was found.
    """
    contents = paper.contents
    metadata = paper.metadata
    if contents is None or metadata is None:
        return None
    existing = getattr(contents, "printed_citation_receipt", None)
    if existing is not None:
        return existing
    line: PrintedCitation | None = getattr(contents, "printed_citation", None)
    if line is None:
        return None
    receipt: dict[str, Any] = {
        "text_ids": list(line.text_ids),
        "cue": "strong" if line.strong else "weak",
        "check": "not_parsed",
        "accepted": False,
        "filled": [],
        "agreement": {},
    }
    reference = line.reference
    if reference is not None:
        comparison = _compare(metadata, line, reference)
        receipt["agreement"] = comparison.agreement
        receipt["check"] = comparison.check
        if comparison.accepted and _abstained(paper):
            # The front matter is several papers' and the extraction chose
            # none: neither does the line.
            receipt["check"] = "front_matter_abstained"
        elif comparison.accepted:
            receipt["accepted"] = True
            receipt["filled"] = _fill(metadata, reference, line)
    contents.printed_citation_receipt = receipt
    named = [name for name in ("title", "author") if name in receipt["filled"]]
    if named:
        others = [name for name in receipt["filled"] if name not in named]
        message = (
            f"{' and '.join(named)} read from the paper's printed citation "
            f"({line.label!r}, text {', '.join(map(str, line.text_ids))})"
        )
        if others:
            message += f"; it also gave {', '.join(others)}"
        paper.processing_warnings.append(
            ProcessingWarning(WarningCode.SELF_CITATION_FILLED, message)
        )
    if receipt["filled"]:
        logger.info("Printed citation filled %s", ", ".join(receipt["filled"]))
    return receipt


__all__ = [
    "LOCATOR_FIELDS",
    "PrintedCitation",
    "apply_printed_citation",
    "find_printed_citation",
    "read_printed_citation",
    "same_title",
    "split_printed_authors",
]
