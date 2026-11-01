"""Turn tagged front-matter words into the front-matter LLM calls' answers.

Pure Python: no model, no numpy. :func:`assemble` reads the labelled spans
back as printed text (character offsets into the tagger input, so nothing is
retyped) and builds the :class:`~bibr.schemas.TitleKeywordsLLM` and
:class:`~bibr.schemas.AuthorsLLM` the two LLM calls would have returned. The
rules mirror the prompts' (``bibr/clients/prompts.py``): the title printed
first, the abstract printed first when parallel ones are printed, given/family
as printed, affiliations resolved through their marks and joined with "; ".

Anything the tags leave undecided is reported in
:attr:`TaggedFrontMatter.problems`; with a problem, or a field below the
confidence bar, the caller asks the LLM as before.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace

from bibr.export.normalize import license_ids
from bibr.front_tagger.contract import WORD_RE, TaggerInput
from bibr.schemas import AuthorLLM, AuthorsLLM, TitleKeywordsLLM

_NAME_FIELDS = frozenset({"GIVEN", "FAMILY"})
_LIST_SEPARATORS = frozenset({",", ";", "，", "；", "、"})
_SYMBOL_MARKS = frozenset("*†‡§¶#‖⁎∗✉☯✝")
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
# A keyword label between two keyword runs starts a new printed list, which is
# how parallel-language keyword lists separate ("Palabras clave" ... "Keywords").
_KEYWORD_HEADING_RE = re.compile(
    r"^(?:key\s?words?|index|palabras|palavras(?:-chave)?|mots-cl[ée]s|schl[üu]sselw[öo]rter|kata|"
    r"kunci|parole|trefwoorden|ключові|ключевые|anahtar|kelimeler|słowa|klíčová|kulcsszavak)$",
    re.IGNORECASE,
)
# A family name alone is complete for a group author ("ATLAS Collaboration").
_COLLECTIVE_RE = re.compile(
    r"group|consorti|collaborat|network|team|committee|investigators|study|project|"
    r"initiative|alliance|society|association|council|working|members",
    re.IGNORECASE,
)
# Words a byline prints between names without naming anyone.
_BYLINE_FILLERS = frozenset(
    {"and", "und", "et", "y", "e", "i", "og", "en", "och", "a", "&", "with", "on", "behalf", "of"}
    | {"the", "for", "md", "phd", "msc", "bsc", "mph", "rn", "dphil", "dr", "prof", "jr", "sr"}
)
# A publisher's name printed beside the byline (a logo, an imprint) is no one's
# name: an author whose whole name is one of these is dropped. Folded to letters
# and digits (see _fold). A short list of the ones seen, not a complete one, and
# without the ones that are surnames too ("Wiley", "Sage", "Springer").
_PUBLISHER_NAMES = frozenset(
    {
        "degruyter",
        "degruyteropen",
        "elsevier",
        "springernature",
        "johnwileysons",
        "wileyonlinelibrary",
        "taylorfrancis",
        "taylorfrancisgroup",
        "sagepublications",
        "mdpi",
        "frontiersmedia",
        "oxforduniversitypress",
        "cambridgeuniversitypress",
    }
)
_MIN_ABSTRACT_WORDS = 40
_ROMAN_KEY_RE = re.compile(r"[IVX]{1,4}")
_LETTER_KEY_RE = re.compile(r"[a-h]")
# A letter key glued to the next word of an affiliation line ("aComputational").
_KEYED_WORD_RE = re.compile(r"[a-h][A-Z][a-z]{3,}")
_MAX_GLUED_LETTERS = 3
# A lower-case run followed by a capitalised one: two of them in one word
# ("AnxietyinCollegeStudents") are words a text layer printed without spaces.
# Short pieces are acronyms and brand names ("CoViD", "ConvNeXt").
_CASE_JOIN_RE = re.compile(r"[a-z]{3}(?=[A-Z][a-z]{2})")


@dataclass(frozen=True)
class Span:
    """A run of words carrying one field, read back as printed."""

    field: str
    words: tuple[int, ...]  # indices into TaggerInput.words
    text: str
    confidence: float  # lowest word confidence in the span


# Problems that hold back the title/keywords answer; every other one holds
# back the authors answer. That call also returns the abstract, so a missing
# or clipped one sends it to the LLM.
_TITLE_PROBLEMS = frozenset(
    {
        "no_title",
        "glued_title_words",
        "repeated_title_words",
        "untagged_title_edge",
        "no_abstract",
        "short_abstract",
    }
)


@dataclass
class TaggedFrontMatter:
    """The tagger's two answers. A part set to ``None`` goes to its LLM call."""

    title_keywords: TitleKeywordsLLM | None  # the title/keywords call's answer
    byline: AuthorsLLM | None  # the authors call's answer
    #: Field -> confidence in [0, 1] (lowest word posterior; the abstract's is the mean).
    confidence: dict[str, float] = field(default_factory=dict)
    #: What the tags leave undecided; title problems ("no_title", "no_abstract",
    #: "short_abstract") hold back the title/keywords answer, every other
    #: problem the authors answer.
    problems: list[str] = field(default_factory=list)

    def _low(self, names: tuple[str, ...], min_confidence: float) -> list[str]:
        return [
            f"low_confidence_{name}"
            for name in names
            if (value := self.confidence.get(name)) is not None and value < min_confidence
        ]

    def title_reasons(self, min_confidence: float) -> list[str]:
        """Why the title/keywords answer cannot replace its call (empty when it can)."""
        problems = [p for p in self.problems if p in _TITLE_PROBLEMS]
        return problems + self._low(("title", "abstract"), min_confidence)

    def byline_reasons(self, min_confidence: float) -> list[str]:
        """Why the authors answer cannot replace its call (empty when it can)."""
        problems = [p for p in self.problems if p not in _TITLE_PROBLEMS]
        return problems + self._low(("authors",), min_confidence)

    def rejection_reasons(self, min_confidence: float) -> list[str]:
        return self.title_reasons(min_confidence) + self.byline_reasons(min_confidence)

    def accepted(self, min_confidence: float) -> bool:
        """Whether the tags can replace both the title/keywords and authors calls."""
        return not self.rejection_reasons(min_confidence)


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
    spans: list[Span],
    tags: Sequence[str],
    inp: TaggerInput,
    wanted: str,
    heading: re.Pattern[str] = _ABSTRACT_HEADING_RE,
) -> list[list[Span]]:
    """Groups of ``wanted`` spans interrupted only by untagged words or furniture.

    A printed ``heading`` word (an "Abstract"/"Resumen" label) between two runs
    also starts a new group, which is how parallel-language versions separate.
    """
    groups: list[list[Span]] = []
    for span in (s for s in spans if s.field == wanted):
        if groups:
            gap = range(groups[-1][-1].words[-1] + 1, span.words[0])
            interrupted = any(
                _field(tags[i]) not in (None, "FURNITURE")
                or heading.match(inp.words[i].text.strip(_TRIM) or "-")
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


def _is_symbol(key: str) -> bool:
    return not any(c.isalnum() for c in key)


def _mark_keys(words: list[str]) -> list[str]:
    """Affiliation keys from mark words: "1", ",", "2", "–", "4" -> 1, 2, 3, 4.

    Superscript digits read as plain ones ("²" keys the "2" affiliation), and
    a run of one repeated symbol is one key ("*", "*" -> "**").
    """
    keys: list[str] = []
    pending_range = False
    previous_symbol = False
    for raw in words:
        word = unicodedata.normalize("NFKC", raw)
        if word in _LIST_SEPARATORS:
            pending_range = previous_symbol = False
            continue
        if word in _RANGE_DASHES:
            pending_range = bool(keys) and keys[-1].isdigit()
            previous_symbol = False
            continue
        if previous_symbol and _is_symbol(word) and set(word) == set(keys[-1]):
            keys[-1] += word
        elif pending_range and word.isdigit() and int(word) > int(keys[-1]):
            keys.extend(str(n) for n in range(int(keys[-1]) + 1, int(word) + 1))
        else:
            keys.append(word)
        pending_range = False
        previous_symbol = _is_symbol(word)
    return keys


def _strip_orcid_icon(name: str) -> str:
    """Drop an ORCID icon OCR glued to a name as "ID" ("HeID" -> "He")."""
    if len(name) > 3 and name.endswith("ID") and name[-3].islower():
        return name[:-2]
    return name


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


def _title_text_problems(inp: TaggerInput, tags, first_word: int, last_word: int) -> list[str]:
    """What makes the printed title unsafe to copy as is.

    The LLM respaces a title a text layer glued together, drops a doubled
    text layer's repeat and keeps a drop-capped first word; copied print
    keeps the fault.
    """
    problems: list[str] = []
    words = [inp.words[i].text for i in range(first_word, last_word + 1)]
    if any(len(_CASE_JOIN_RE.findall(word)) >= 2 for word in words):
        problems.append("glued_title_words")
    letters = [w.casefold() for w in words if any(c.isalnum() for c in w)]
    trigrams = [tuple(letters[i : i + 3]) for i in range(len(letters) - 2)]
    if any(trigrams[j] in trigrams[: max(0, j - 2)] for j in range(len(trigrams))):
        problems.append("repeated_title_words")
    # One capitalised word left alone just before the title, in its region
    # (after the region's start or a tagged field) or as a region of its own,
    # is a drop-capped first word; a run of them is a rubric ("Original
    # Article") the tagger rightly left out. A lone capital before a title
    # that starts in lower case is a drop cap split off its word ("W" "ine").
    # A lone word tagged page furniture counts too: the model learned to cut
    # a title's first word from projections that did ("Wine | Quality ...").
    before = first_word - 1
    words = inp.words

    def untagged(i: int, region: int) -> bool:
        return i >= 0 and words[i].region == region and _field(tags[i]) in (None, "FURNITURE")

    if untagged(before, words[before].region if before >= 0 else -1):
        text = words[before].text
        region = words[before].region
        # A furniture word only inside the title's region: one in a region of
        # its own is a journal banner ("Cell").
        alone = not untagged(before - 1, region) and (
            region == words[first_word].region
            or (_field(tags[before]) is None and not untagged(before + 1, region))
        )
        drop_cap = len(text) == 1 and text.isupper() and words[first_word].text[:1].islower()
        if alone and text[:1].isupper() and (not text.isupper() or drop_cap):
            problems.append("untagged_title_edge")
    return problems


def _title(spans: list[Span], tags, inp: TaggerInput, conf) -> tuple[str | None, float, list[str]]:
    groups = _clusters(spans, tags, inp, "TITLE")
    if not groups:
        return None, 0.0, ["no_title"]
    group = groups[0]  # the version printed first
    title = _joined(group)
    first_word, last_word = group[0].words[0], group[-1].words[-1]
    problems = _title_text_problems(inp, tags, first_word, last_word)
    # The words around the title decide where it starts and stops: a running
    # head the model was unsure about counts against the title.
    edge = [i for i in (first_word - 1, last_word + 1) if 0 <= i < len(inp.words)]
    confidence = min([span.confidence for span in group] + [conf[i] for i in edge])
    subtitle = next((s for s in spans if s.field == "SUBTITLE" and s.words[0] > last_word), None)
    if subtitle is not None and subtitle.text:
        from bibr.extract.title_subtitle import _separator

        title = f"{title}{_separator(title, subtitle.text)}{subtitle.text}"
        confidence = min(confidence, subtitle.confidence)
    return title, confidence, problems


def _abstract(spans: list[Span], tags, inp: TaggerInput, conf) -> tuple[str | None, float | None]:
    """The abstract printed first; parallel-language versions are not joined."""
    groups = _clusters(spans, tags, inp, "ABSTRACT")
    if not groups:
        return None, None
    group = groups[0]
    words = [i for span in group for i in span.words]
    return _joined(group), sum(conf[i] for i in words) / len(words)


def _keywords(spans: list[Span], tags, inp: TaggerInput) -> list[str]:
    """The keywords of the list printed first; parallel-language lists are not joined."""
    groups = _clusters(spans, tags, inp, "KEYWORD", heading=_KEYWORD_HEADING_RE)
    return [
        k for span in (groups[0] if groups else []) if (k := _clean(_closed_text(span, tags, inp)))
    ]


def _closed_text(span: Span, tags, inp: TaggerInput) -> str:
    """The span's text, closed over the ")" words the tagger left just after it.

    The model ends a keyword before the parenthesis that closes its
    abbreviation ("non-small cell lung cancer (NSCLC" then ")"). Only a ")" the
    paper prints directly after the span, in its region and in no field, is
    taken, and only for as many "(" as the span leaves open.
    """
    text = span.text
    last = span.words[-1]
    while text.count("(") > text.count(")"):
        last += 1
        if (
            last >= len(inp.words)
            or inp.words[last].text != ")"
            or inp.words[last].region != inp.words[span.words[-1]].region
            or _field(tags[last]) not in (None, "FURNITURE")
        ):
            break
        text += ")"
    return text


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


def _incomplete_name(author: _Author) -> bool:
    """A person with a given or a family name missing; group authors excepted."""
    if author.given and author.family:
        return False
    parts = author.family or author.given
    return not any(_COLLECTIVE_RE.search(s.text) for s in parts)


def _is_publisher(author: _Author) -> bool:
    """Whether the whole name is a publisher's, tagged as a person by its logo."""
    return _fold(" ".join(s.text for s in (*author.given, *author.family))) in _PUBLISHER_NAMES


def _untagged_byline_words(inp: TaggerInput, tags, lo: int, hi: int) -> bool:
    """Whether a capitalised word between the first and last author is in no
    field: a name the tagger dropped from the list."""
    for i in range(lo, hi + 1):
        if _field(tags[i]) is not None:
            continue
        text = inp.words[i].text
        if text[:1].isupper() and text.casefold().strip(".") not in _BYLINE_FILLERS:
            return True
    return False


def _glued_key(name: str, letters: set[str], romans: set[str]) -> str | None:
    """The affiliation key OCR glued to the end of ``name``, if one is."""
    for key in sorted(romans, key=len, reverse=True):
        stem = name[: -len(key)]
        # "LimaII" splits at the case change; all capitals ("LIMAII") only on
        # a key of two or more numerals, so "GANDHI" keeps its "I".
        if (
            name.endswith(key)
            and len(stem) >= 2
            and stem[-1].isalpha()
            and (stem[-1].islower() or len(key) >= 2)
        ):
            return key
    key = name[-1:]
    if key in letters and len(name) >= 3 and name[-2].isalpha():
        return key
    return None


def _printed_letter_keys(inp: TaggerInput, tags) -> set[str]:
    """Letters "a"-"h" the print keys an affiliation with, wherever the model left one.

    Besides the letters it tagged as marks: a lower-case letter alone at the
    start of a line that goes on with an affiliation ("a Department of X"
    with the "a" untagged), and a letter glued to the line's first word
    ("aComputational Science").
    """
    letters: set[str] = set()
    words = inp.words
    for i, word in enumerate(words):
        starts_line = i == 0 or words[i - 1].region != word.region
        if _LETTER_KEY_RE.fullmatch(word.text):
            if _field(tags[i]) == "AFF_MARK" or (
                starts_line and i + 1 < len(words) and tags[i + 1] == "B-AFFILIATION"
            ):
                letters.add(word.text)
        elif _KEYED_WORD_RE.fullmatch(word.text) and (starts_line or tags[i - 1] != "O"):
            letters.add(word.text[0])
    return letters


def _unglue_letter_keys(
    authors: list[_Author], by_mark: dict[str, str], printed_letters: Collection[str] = ()
) -> None:
    """Split a superscript letter or numeral key OCR glued to a family name.

    On a paper whose affiliations are keyed "a", "b", ... (or "I", "II", ...),
    an author with no mark of their own whose family name ends in one of
    those keys printed it as a superscript: "Vegaa" is "Vega" keyed "a",
    "LIMAII" is "LIMA" keyed "II". A list of letter keys ("a,b") glues on as a
    run, taken from the end while the letters rise and the name keeps four
    letters or more, at most three: "Riosab" is "Rios" keyed "a" and "b", and
    "Costaa" only loses its last "a".

    ``printed_letters`` are letter keys the print shows that no affiliation
    was tagged behind (see :func:`_printed_letter_keys`).
    """
    letters = {key for key in by_mark if len(key) == 1 and key.islower()} | set(printed_letters)
    romans = {key for key in by_mark if _ROMAN_KEY_RE.fullmatch(key)}
    if not letters and not romans:
        return
    for author in authors:
        if author.marks or not author.family:
            continue
        last = author.family[-1]
        key = _glued_key(last.text, letters, romans)
        if key is None:
            continue
        keys, stem = [key], last.text[: -len(key)]
        while (
            len(key) == 1
            and len(keys) < _MAX_GLUED_LETTERS
            and len(stem) >= 4
            and stem[-1] in letters
            and stem[-1] < keys[0]
            and stem[-2].isalpha()
        ):
            keys.insert(0, stem[-1])
            stem = stem[:-1]
        author.family[-1] = replace(last, text=stem)
        author.marks = keys


def _given_first(names: list[Span]) -> bool:
    """Whether the byline prints given names first.

    Read from tight pairs: a given and a family part on adjacent words with no
    comma or semicolon between them.
    """
    given_family = family_given = 0
    for a, b in zip(names, names[1:], strict=False):
        if a.field != b.field and b.words[0] == a.words[-1] + 1 and not a.text.endswith((",", ";")):
            given_family += a.field == "GIVEN"
            family_given += a.field == "FAMILY"
    return given_family > family_given


def _group_authors(spans: list[Span]) -> list[_Author]:
    """One author per given/family pair, in either printed order.

    An author is complete once it holds one name part of each kind; the next
    name part of a kind it already holds starts the next author. That reads
    "John Smith, Mary Jones" and "Smith J, Jones M" alike, and an organisation
    (a family part alone) followed by another one as two authors. In a byline
    that prints given names first, a given name also starts the next author
    after a family name, so a one-name author ("Suprihatin, Siti Nuraini") does
    not take the next author's given name.
    """
    names = [span for span in spans if span.field in _NAME_FIELDS]
    given_first = _given_first(names)
    authors: list[_Author] = []
    for span in names:
        slot = "given" if span.field == "GIVEN" else "family"
        current = authors[-1] if authors else None
        if (
            current is None
            or getattr(current, slot)
            or (given_first and slot == "given" and current.family)
        ):
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

    title, title_conf, title_problems = _title(spans, tags, inp, confidences)
    problems.extend(title_problems)
    confidence["title"] = title_conf
    abstract, abstract_conf = _abstract(spans, tags, inp, confidences)
    if abstract_conf is not None:
        confidence["abstract"] = abstract_conf
    if abstract is None:
        problems.append("no_abstract")
    elif len(WORD_RE.findall(abstract)) < _MIN_ABSTRACT_WORDS:
        problems.append("short_abstract")

    pages = _clean(getattr(_first(spans, "PAGES"), "text", None))
    first_page = last_page = None
    if pages and (m := _PAGE_RANGE_RE.match(pages)):
        first_page, last_page = m.group(1), m.group(2)
    license_span = _first(spans, "LICENSE")
    title_keywords = TitleKeywordsLLM(
        title=title,
        abstract=abstract,
        keywords=_keywords(spans, tags, inp),
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

    authors = [a for a in _group_authors(spans) if not _is_publisher(a)]
    if not authors:
        problems.append("no_authors")
    name_words = [i for a in authors for s in (*a.given, *a.family) for i in s.words]
    confidence["authors"] = min((confidences[i] for i in name_words), default=0.0)
    if any(not a.family and not a.given for a in authors):
        problems.append("empty_author_name")
    elif any(_incomplete_name(a) for a in authors):
        problems.append("incomplete_author_name")
    if authors and _untagged_byline_words(inp, tags, authors[0].first_word, authors[-1].last_word):
        problems.append("untagged_byline_word")

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
    _unglue_letter_keys(authors, affiliation_by_mark, _printed_letter_keys(inp, tags))
    _link_affiliations(authors, affiliation_by_mark, unmarked_affiliations, problems)
    _assign_loose_emails(authors, spans)
    _flag_corresponding(authors, inp, tags)

    author_rows = [
        AuthorLLM(
            given=" ".join(s.text for s in a.given),
            family=_strip_orcid_icon(" ".join(s.text for s in a.family)),
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
                    if not _is_symbol(key) and "affiliation_mark_unresolved" not in problems:
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
        key
        for indices in cue_regions
        for key in _mark_keys(
            [
                inp.words[i].text
                for i in indices
                if _is_symbol(inp.words[i].text) and _field(tags[i]) != "AUTHOR_MARK"
            ]
        )
        if key in _SYMBOL_MARKS or len(set(key)) == 1 and key[0] in _SYMBOL_MARKS
    }
    cue_text = _fold("".join(inp.words[i].text for indices in cue_regions for i in indices))
    for author in authors:
        by_symbol = bool(symbols.intersection(author.marks))
        by_email = bool(author.email) and _fold(author.email or "") in cue_text
        author.corresponding = by_symbol or by_email


__all__ = ["Span", "TaggedFrontMatter", "assemble"]
