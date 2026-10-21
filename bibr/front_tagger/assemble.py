"""Turn tagged front-matter words into the front-matter LLM calls' answers.

Pure Python: no model, no numpy. :func:`assemble` reads the labelled spans
back as printed text (character offsets into the tagger input, so nothing is
retyped) and builds the :class:`~bibr.schemas.TitleKeywordsLLM` and
:class:`~bibr.schemas.AuthorsLLM` the two LLM calls would have returned. The
rules mirror the prompts' (``bibr/clients/prompts.py``): the title printed
first, the English abstract when parallel ones are printed, given/family as
printed, affiliations resolved through their marks and joined with "; ".

Anything the tags leave undecided is reported in
:attr:`TaggedFrontMatter.problems`; with a problem, or a field below the
confidence bar, the caller asks the LLM as before.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from bibr.export.normalize import license_ids
from bibr.front_tagger.contract import TaggerInput
from bibr.schemas import AuthorLLM, AuthorsLLM, TitleKeywordsLLM

_NAME_FIELDS = frozenset({"GIVEN", "FAMILY"})
_LIST_SEPARATORS = frozenset({",", ";", "，", "；", "、"})
_SYMBOL_MARKS = frozenset("*†‡§¶#‖⁎∗✉")
_RANGE_DASHES = frozenset("-–—")
_TRIM = " \t\n,;:.·|"
_ORCID_RE = re.compile(r"(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})[\s-]?(\d{3}[\dXx])")
_ISO_DATE_RE = re.compile(r"\b((?:19|20)\d\d)-(\d\d)-(\d\d)\b")
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d\d)(?!\d)")
_PAGE_RANGE_RE = re.compile(
    r"^\s*([A-Za-z]{0,2}\d+[A-Za-z]?)\s*[-–—]+\s*([A-Za-z]{0,2}\d+[A-Za-z]?)"
)
_CORRESPONDING_RE = re.compile(
    r"correspond|address correspondence|reprint requests|автор для переписки|"
    r"autor(?:a)? (?:de )?correspond|auteur correspondant|korrespondenz|通讯作者|責任著者",
    re.IGNORECASE,
)
_ABSTRACT_HEADING_RE = re.compile(
    r"^(?:abstract|summary|resumen|resumo|r[ée]sum[ée]|zusammenfassung|riassunto|"
    r"аннотация|резюме|abstrakt|samenvatting|要旨|摘要|초록)$",
    re.IGNORECASE,
)
_ENGLISH_WORDS = frozenset(
    {"the", "of", "and", "in", "to", "a", "is", "was", "we", "for", "with", "that", "this"}
    | {"are", "were", "on", "by", "as", "from"}
)


@dataclass(frozen=True)
class Span:
    """A run of words carrying one field, read back as printed."""

    field: str
    words: tuple[int, ...]  # indices into TaggerInput.words
    text: str
    confidence: float  # lowest word confidence in the span


@dataclass
class TaggedFrontMatter:
    title_keywords: TitleKeywordsLLM  # the title/keywords call's answer
    byline: AuthorsLLM  # the authors call's answer
    #: Field -> confidence in [0, 1] (lowest word posterior; the abstract's is the mean).
    confidence: dict[str, float] = field(default_factory=dict)
    #: Why the tags alone cannot stand in for the LLM (empty when they can).
    problems: list[str] = field(default_factory=list)

    def accepted(self, min_confidence: float) -> bool:
        """Whether the tags can replace the title/keywords and authors calls."""
        if self.problems:
            return False
        needed = ("title", "authors")
        if any(self.confidence.get(name, 0.0) < min_confidence for name in needed):
            return False
        return self.confidence.get("abstract", 1.0) >= min_confidence

    def rejection_reasons(self, min_confidence: float) -> list[str]:
        reasons = list(self.problems)
        for name in ("title", "authors", "abstract"):
            value = self.confidence.get(name)
            if value is not None and value < min_confidence:
                reasons.append(f"low_confidence_{name}")
        return reasons


def _field(tag: str) -> str | None:
    return None if tag == "O" else tag[2:]


def _spans(inp: TaggerInput, tags: Sequence[str], conf: Sequence[float]) -> list[Span]:
    """Maximal B/I runs; an I- tag after another field starts a span, as CRFs allow."""
    spans: list[Span] = []
    current: list[int] = []
    current_field: str | None = None

    def close() -> None:
        if current and current_field is not None:
            first, last = inp.words[current[0]], inp.words[current[-1]]
            text = inp.text[first.start : last.end].replace("\n", " ")
            spans.append(
                Span(
                    field=current_field,
                    words=tuple(current),
                    text=re.sub(r"\s+", " ", text).strip(),
                    confidence=min(conf[i] for i in current),
                )
            )

    for i, tag in enumerate(tags):
        name = _field(tag)
        if name is not None and tag.startswith("I-") and name == current_field:
            current.append(i)
            continue
        close()
        current, current_field = ([i], name) if name is not None else ([], None)
    close()
    return spans


def _clusters(
    spans: list[Span], tags: Sequence[str], inp: TaggerInput, wanted: str
) -> list[list[Span]]:
    """Groups of ``wanted`` spans interrupted only by untagged words or furniture.

    A printed heading (an "Abstract"/"Resumen" label) between two runs also
    starts a new group, which is how parallel-language abstracts separate.
    """
    groups: list[list[Span]] = []
    for span in (s for s in spans if s.field == wanted):
        if groups:
            gap = range(groups[-1][-1].words[-1] + 1, span.words[0])
            interrupted = any(
                _field(tags[i]) not in (None, "FURNITURE")
                or _ABSTRACT_HEADING_RE.match(inp.words[i].text.strip(_TRIM) or "-")
                for i in gap
            )
            if not interrupted:
                groups[-1].append(span)
                continue
        groups.append([span])
    return groups


def _joined(group: list[Span]) -> str:
    return " ".join(span.text for span in group).strip()


def _first(spans: list[Span], name: str) -> Span | None:
    return next((s for s in spans if s.field == name), None)


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.strip(_TRIM)
    return text or None


def _mark_keys(words: list[str]) -> list[str]:
    """Affiliation keys from mark words: "1", ",", "2", "–", "4" -> 1, 2, 3, 4."""
    keys: list[str] = []
    pending_range = False
    for word in words:
        if word in _LIST_SEPARATORS:
            pending_range = False
            continue
        if word in _RANGE_DASHES:
            pending_range = bool(keys) and keys[-1].isdigit()
            continue
        if pending_range and word.isdigit() and int(word) > int(keys[-1]):
            keys.extend(str(n) for n in range(int(keys[-1]) + 1, int(word) + 1))
        else:
            keys.append(word)
        pending_range = False
    return keys


def _license_short(printed: str | None) -> str | None:
    if not printed:
        return None
    _, spdx = license_ids(printed)
    if not spdx:
        return None
    if spdx == "CC0-1.0":
        return "CC0 1.0"
    parts = spdx.split("-")  # CC, BY, [NC, ND|SA], version
    return f"CC {'-'.join(parts[1:-1])} {parts[-1]}"


def _published(printed: str | None) -> str | None:
    if not printed:
        return None
    if m := _ISO_DATE_RE.search(printed):
        return m.group(0)
    if m := _YEAR_RE.search(printed):
        return m.group(1)
    return None


def _englishness(text: str) -> float:
    words = re.findall(r"[a-z]+", text.lower())
    return sum(w in _ENGLISH_WORDS for w in words) / len(words) if words else 0.0


def _title(spans: list[Span], tags, inp: TaggerInput, conf) -> tuple[str | None, float]:
    groups = _clusters(spans, tags, inp, "TITLE")
    if not groups:
        return None, 0.0
    group = groups[0]  # the version printed first
    title = _joined(group)
    first_word, last_word = group[0].words[0], group[-1].words[-1]
    # The words around the title decide where it starts and stops: a running
    # head the model was unsure about counts against the title.
    edge = [i for i in (first_word - 1, last_word + 1) if 0 <= i < len(inp.words)]
    confidence = min([span.confidence for span in group] + [conf[i] for i in edge])
    subtitle = next((s for s in spans if s.field == "SUBTITLE" and s.words[0] > last_word), None)
    if subtitle is not None and subtitle.text:
        from bibr.extract.title_subtitle import _separator

        title = f"{title}{_separator(title, subtitle.text)}{subtitle.text}"
        confidence = min(confidence, subtitle.confidence)
    return title, confidence


def _abstract(spans: list[Span], tags, inp: TaggerInput, conf) -> tuple[str | None, float | None]:
    groups = _clusters(spans, tags, inp, "ABSTRACT")
    if not groups:
        return None, None
    english = [g for g in groups if _englishness(_joined(g)) >= 0.08]
    group = (english or groups)[0]
    words = [i for span in group for i in span.words]
    return _joined(group), sum(conf[i] for i in words) / len(words)


@dataclass
class _Author:
    given: list[Span] = field(default_factory=list)
    family: list[Span] = field(default_factory=list)
    first_word: int = 0
    last_word: int = 0
    marks: list[str] = field(default_factory=list)
    affiliation: str = ""
    email: str | None = None
    orcid: str | None = None
    corresponding: bool = False


def _group_authors(spans: list[Span]) -> list[_Author]:
    """One author per given/family pair, in either printed order.

    An author is complete once it holds one name part of each kind; the next
    name part of a kind it already holds starts the next author. That reads
    "John Smith, Mary Jones" and "Smith J, Jones M" alike, and an organisation
    (a family part alone) followed by another one as two authors.
    """
    authors: list[_Author] = []
    for span in spans:
        if span.field not in _NAME_FIELDS:
            continue
        slot = "given" if span.field == "GIVEN" else "family"
        if not authors or getattr(authors[-1], slot):
            authors.append(_Author(first_word=span.words[0]))
        getattr(authors[-1], slot).append(span)
        authors[-1].last_word = span.words[-1]
    return authors


def _words_between(inp: TaggerInput, tags, lo: int, hi: int, name: str) -> list[str]:
    return [inp.words[i].text for i in range(lo, hi) if _field(tags[i]) == name]


def assemble(
    inp: TaggerInput,
    tags: Sequence[str],
    confidences: Sequence[float],
) -> TaggedFrontMatter:
    """Build the front-matter answers from one paper's word tags.

    ``tags`` holds one BIO tag per ``inp.words`` entry and ``confidences`` the
    posterior of that tag (``1.0`` everywhere when the runtime has none).
    """
    if not len(tags) == len(confidences) == len(inp.words):
        raise ValueError("tags and confidences must have one entry per tagger word")
    spans = _spans(inp, tags, confidences)
    problems: list[str] = []
    confidence: dict[str, float] = {}

    title, title_conf = _title(spans, tags, inp, confidences)
    if title is None:
        problems.append("no_title")
    confidence["title"] = title_conf
    abstract, abstract_conf = _abstract(spans, tags, inp, confidences)
    if abstract_conf is not None:
        confidence["abstract"] = abstract_conf

    pages = _clean(getattr(_first(spans, "PAGES"), "text", None))
    first_page = last_page = None
    if pages and (m := _PAGE_RANGE_RE.match(pages)):
        first_page, last_page = m.group(1), m.group(2)
    license_span = _first(spans, "LICENSE")
    title_keywords = TitleKeywordsLLM(
        title=title,
        abstract=abstract,
        keywords=[k for s in spans if s.field == "KEYWORD" if (k := _clean(s.text))],
        journal=_clean(getattr(_first(spans, "JOURNAL"), "text", None)),
        volume=_clean(getattr(_first(spans, "VOLUME"), "text", None)),
        issue=_clean(getattr(_first(spans, "ISSUE"), "text", None)),
        first_page=first_page,
        last_page=last_page,
        issn=_clean(getattr(_first(spans, "ISSN"), "text", None)),
        publisher=_clean(getattr(_first(spans, "PUBLISHER"), "text", None)),
        published=_published(getattr(_first(spans, "DATE"), "text", None)),
        license=_license_short(license_span.text if license_span else None),
    )

    authors = _group_authors(spans)
    if not authors:
        problems.append("no_authors")
    name_words = [i for a in authors for s in (*a.given, *a.family) for i in s.words]
    confidence["authors"] = min((confidences[i] for i in name_words), default=0.0)
    if any(not a.family and not a.given for a in authors):
        problems.append("empty_author_name")

    # Marks, emails and ORCIDs printed after an author and before the next
    # belong to that author.
    for pos, author in enumerate(authors):
        stop = authors[pos + 1].first_word if pos + 1 < len(authors) else len(inp.words)
        zone_end = stop
        if pos + 1 == len(authors):
            # The last author's zone ends where the next non-mark field starts.
            zone_end = next(
                (
                    i
                    for i in range(author.last_word + 1, len(inp.words))
                    if _field(tags[i]) not in (None, "AUTHOR_MARK", "EMAIL", "ORCID", "FURNITURE")
                ),
                len(inp.words),
            )
        author.marks = _mark_keys(
            _words_between(inp, tags, author.last_word + 1, zone_end, "AUTHOR_MARK")
        )
        for span in spans:
            if not author.last_word < span.words[0] < zone_end:
                continue
            if span.field == "EMAIL" and author.email is None:
                author.email = _clean(span.text)
            elif (
                span.field == "ORCID"
                and author.orcid is None
                and (m := _ORCID_RE.search(span.text))
            ):
                author.orcid = "-".join(m.groups()).upper()

    affiliation_by_mark, unmarked_affiliations = _affiliations(spans, inp, tags)
    _link_affiliations(authors, affiliation_by_mark, unmarked_affiliations, problems)
    _assign_loose_emails(authors, spans)
    _flag_corresponding(authors, inp, tags)

    author_rows = [
        AuthorLLM(
            given=" ".join(s.text for s in a.given),
            family=" ".join(s.text for s in a.family),
            affiliation=a.affiliation,
            email=a.email,
            orcid=a.orcid,
            corresponding=a.corresponding,
        )
        for a in authors
    ]
    return TaggedFrontMatter(
        title_keywords=title_keywords,
        byline=AuthorsLLM(authors=author_rows),
        confidence=confidence,
        problems=problems,
    )


def _affiliations(
    spans: list[Span], inp: TaggerInput, tags
) -> tuple[dict[str, str], list[tuple[int, str]]]:
    """Affiliations keyed by the mark printed in front of them, and unmarked ones.

    Unmarked affiliations keep their first word index, for linking by position.
    """
    by_mark: dict[str, str] = {}
    unmarked: list[tuple[int, str]] = []
    previous_end = 0
    for span in spans:
        if span.field != "AFFILIATION":
            continue
        text = _clean(span.text)
        if not text:
            continue
        keys = _mark_keys(_words_between(inp, tags, previous_end, span.words[0], "AFF_MARK"))
        previous_end = span.words[-1] + 1
        if keys:
            # "1 Dept A" keys the span with "1"; a run "a,b Dept" keys both.
            for key in keys:
                by_mark.setdefault(key, text)
        elif text not in (t for _, t in unmarked):
            unmarked.append((span.words[0], text))
    return by_mark, unmarked


def _link_affiliations(
    authors: list[_Author],
    by_mark: dict[str, str],
    unmarked: list[tuple[int, str]],
    problems: list[str],
) -> None:
    if not authors:
        return
    if by_mark:
        for author in authors:
            texts: list[str] = []
            for key in author.marks:
                text = by_mark.get(key)
                if text is None:
                    if key not in _SYMBOL_MARKS and "affiliation_mark_unresolved" not in problems:
                        # A digit or letter key with no affiliation behind it:
                        # the list was not read, so the LLM resolves it.
                        problems.append("affiliation_mark_unresolved")
                    continue
                if text not in texts:
                    texts.append(text)
            author.affiliation = "; ".join(texts)
        return
    if not unmarked:
        return
    if len(unmarked) == 1:
        for author in authors:
            author.affiliation = unmarked[0][1]
        return
    # Several unmarked affiliations: link each to the authors printed between
    # it and the previous one ("Name / Institution / Name / Institution").
    starts = [start for start, _ in unmarked]
    if all(start > authors[-1].last_word for start in starts):
        problems.append("affiliations_unlinked")
        return
    for author in authors:
        following = [text for start, text in unmarked if start > author.last_word]
        author.affiliation = following[0] if following else ""
    if any(not a.affiliation for a in authors):
        problems.append("affiliations_unlinked")


def _fold(text: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if c.isalnum() and not unicodedata.combining(c)).lower()


def _assign_loose_emails(authors: list[_Author], spans: list[Span]) -> None:
    """Give an email printed away from the byline to the author it names."""
    taken = {a.email for a in authors if a.email}
    for span in spans:
        if span.field != "EMAIL":
            continue
        email = _clean(span.text)
        if not email or email in taken:
            continue
        local = _fold(email.split("@", 1)[0])
        matches = [
            a
            for a in authors
            if a.email is None
            and (family := _fold(" ".join(s.text for s in a.family)))
            and len(family) >= 3
            and family in local
        ]
        if len(matches) == 1:
            matches[0].email = email
            taken.add(email)


def _flag_corresponding(authors: list[_Author], inp: TaggerInput, tags) -> None:
    """Mark the author a "Corresponding author" note points at, by symbol or email."""
    region_words: dict[int, list[int]] = {}
    for i, word in enumerate(inp.words):
        region_words.setdefault(word.region, []).append(i)
    cue_regions = [
        indices
        for indices in region_words.values()
        if _CORRESPONDING_RE.search(" ".join(inp.words[i].text for i in indices))
    ]
    if not cue_regions:
        return
    symbols = {
        inp.words[i].text
        for indices in cue_regions
        for i in indices
        if inp.words[i].text in _SYMBOL_MARKS and _field(tags[i]) != "AUTHOR_MARK"
    }
    cue_text = _fold("".join(inp.words[i].text for indices in cue_regions for i in indices))
    for author in authors:
        by_symbol = bool(symbols.intersection(author.marks))
        by_email = bool(author.email) and _fold(author.email or "") in cue_text
        author.corresponding = by_symbol or by_email


__all__ = ["Span", "TaggedFrontMatter", "assemble"]
