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
from typing import TYPE_CHECKING

from bibr.export.normalize import license_ids
from bibr.front_tagger.contract import WORD_RE, TaggerInput
from bibr.schemas import AuthorLLM, AuthorsLLM, TitleKeywordsLLM

if TYPE_CHECKING:
    from bibr.models import Affiliation

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
# A word that makes a line of untagged text read as an institution, and the
# openings that make one read as something else (a note, a date line, a heading).
_INSTITUTION_RE = re.compile(
    r"\b(universit\w*|universid\w*|université|universität|uniwersytet|institut\w*|college|"
    r"school of|faculty|department|dept\.?|centre|center|hospital|laborator\w*|academy|"
    r"akademi\w*|polytechnic|hochschule|clinic|ministry|observatory|foundation|council|"
    r"research unit|medical|hemocentro|escola|facultad|facultat|instituto|institution|société|"
    r"company|ltd|inc\.?|gmbh|llc)\b",
    re.IGNORECASE,
)
_NOT_AFFILIATION_RE = re.compile(
    r"^(correspond|received|accepted|published|email|e-mail|address|keywords?|abstract|copyright|©|"
    r"running head|this (article|paper|work|is)|\*|note|we thank|thanks|acknowledg|author note|"
    r"declaration|conflict|"
    # What a page prints around the byline: a stamp, a funder, a citation, a
    # journal's name or an imprint, a reference to the paper's own record.
    r"downloaded|access provided|funded|funding|supported by|presented|cite[sd]?\b|"
    r"send reprint|reprints?\b|journal|proceedings|transactions|annals|bulletin|volume|vol\.|"
    r"issn|doi|https?:|www\.|(?:19|20)\d\d\b|"
    # A line that goes on from a name ("and the ... Group"), or names the paper.
    r"and\b|&|with\b|on behalf|a (?:study|review|survey|trial|case)\b)",
    re.IGNORECASE,
)
# Cues that make a line something else wherever they stand in it.
_NOT_AFFILIATION_ANYWHERE_RE = re.compile(
    r"rights reserved|downloaded (?:from|by)|creative commons|licen[cs]e|grants?\s+(?:no\b|number|from|\d)|funded by|"
    r"supported by|reprint requests|\bpress\b|periodicals|publishing|\bjournal\b|https?://|www\.|\bdoi\b|"
    r"\(\s*(?:19|20)\d\d\s*\)\s*\d+\s*[:(]|^\d+\s+[\w ]{0,30}?"
    r"\b(?:road|rd|street|st|avenue|ave|boulevard|blvd|lane|drive)\b",
    re.IGNORECASE,
)
# A job title alone ("Professor of Medical Education") names no institution; with
# one after a comma it is an affiliation line.
_JOB_TITLE_RE = re.compile(
    r"^(?:(?:associate|assistant|adjunct|emeritus|visiting|senior|full)\s+)*"
    r"(?:professor|lecturer|reader|director|editor|dean|chair|head)\b",
    re.IGNORECASE,
)
# Words of running prose: an affiliation line holds none of them.
_PROSE_RE = re.compile(
    r"\b(?:we|our|were|was|is|are|has|have|had|been|will|would|should|could|that|which|these|"
    r"those|their|they|its|it)\b",
    re.IGNORECASE,
)
# A heading that opens the body (or the abstract the tags did not find): the
# front matter ends there.
_SECTION_HEADING_RE = re.compile(
    r"^\W*(?:(?:\d+|[ivx]+)[.)]?\s+)?(?:abstract|summary|introduction|background|methods?|"
    r"materials? and methods?|methodology|results?|discussion|conclusions?|literature review|"
    r"theoretical (?:background|framework)|related work|overview|resumen|r[ée]sum[ée]|resumo|"
    r"zusammenfassung|einleitung|introducci[oó]n|introdu[cç][aã]o)\W*$",
    re.IGNORECASE,
)
# Where an address line, run on after an affiliation, starts ("..., USA; email: ").
_ADDRESS_CUE_RE = re.compile(r"[;,]?\s*\b(?:e-?mail|correspondence)\b.*$", re.IGNORECASE)
_MIN_AFFILIATION_LETTERS = 3
_MAX_UNTAGGED_RUN_WORDS = 30
_MAX_UNTAGGED_LINE_WORDS = 40
_MIN_ABSTRACT_WORDS = 40
_ROMAN_KEY_RE = re.compile(r"[IVX]{1,4}")
_LETTER_KEY_RE = re.compile(r"[a-h]")
_LONE_LETTER_RE = re.compile(r"[a-z]")
_AFFILIATION_FIELDS = ("AFFILIATION", "AFF_MARK")
# A letter key glued to the next word of an affiliation line ("aComputational").
# The word may be in any script ("aUniversité"); :func:`_keyed_word_letter` reads it.
_KEYED_WORD_RE = re.compile(r"([a-z])([^\W\d_]{4,})")
# A capital key printed alone after a name in the byline ("Eng, A").
_CAPITAL_KEY_RE = re.compile(r"[A-H]")
_MAX_GLUED_LETTERS = 3
# Letters a family name keeps once a run of keys is taken off its end.
_MIN_RUN_NAME_LETTERS = 4
# Lines keyed by a glued letter that stand for each other: two, with distinct letters.
_MIN_KEYED_LINES = 2
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
    #: Affiliations the tags printed that no author carries: their marks did not
    #: resolve, several unmarked ones followed the byline, or an author's turn
    #: came after the list was used up. In print order.
    unattached_affiliations: list[str] = field(default_factory=list)
    #: Institution text the tags left out of every field, read from the print only
    #: where no author carries an affiliation (the rules in
    #: :func:`_untagged_institutions`). In print order. Empty unless ``rules``.
    untagged_affiliations: list[str] = field(default_factory=list)

    def affiliation_rows(self, *, untagged: bool = True) -> list[Affiliation]:
        """The affiliations no author carries, as rows that name no author.

        For a run with no LLM to parse them: the export lists every author's
        affiliation already, and these are the printed ones it would miss.
        ``untagged`` adds the text the tagger left untagged (see
        :attr:`untagged_affiliations`). Rows that differ only in case, accents
        or punctuation are one row.
        """
        from bibr.extract.research_integrity import affiliation_key
        from bibr.models import Affiliation

        texts = [*self.unattached_affiliations, *(self.untagged_affiliations if untagged else [])]
        seen: set[str] = set()
        rows: list[Affiliation] = []
        for text in texts:
            key = affiliation_key(text)
            if key and key not in seen:
                seen.add(key)
                rows.append(Affiliation(text=text))
        return rows

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


# What ends a printed abstract, beside a heading: the next title, a byline, keywords.
_ABSTRACT_ENDS = frozenset({"TITLE", "SUBTITLE", "GIVEN", "FAMILY", "AUTHOR_MARK", "KEYWORD"})


def _clusters(
    spans: list[Span],
    tags: Sequence[str],
    inp: TaggerInput,
    wanted: str,
    heading: re.Pattern[str] = _ABSTRACT_HEADING_RE,
    ends: frozenset[str] | None = None,
) -> list[list[Span]]:
    """Groups of ``wanted`` spans interrupted only by untagged words or furniture.

    A printed ``heading`` word (an "Abstract"/"Resumen" label) between two runs
    also starts a new group, which is how parallel-language versions separate.
    With ``ends`` only those fields interrupt, and any other tag between two
    runs is taken for something set into the text.
    """
    groups: list[list[Span]] = []
    for span in (s for s in spans if s.field == wanted):
        if groups:
            gap = range(groups[-1][-1].words[-1] + 1, span.words[0])
            interrupted = any(
                (
                    _field(tags[i]) not in (None, "FURNITURE")
                    if ends is None
                    else _field(tags[i]) in ends
                )
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


def _abstract(
    spans: list[Span], tags, inp: TaggerInput, conf, *, rules: bool = False
) -> tuple[str | None, float | None]:
    """The abstract printed first; parallel-language versions are not joined.

    With ``rules`` an email, date, volume, affiliation or licence line between
    two runs does not end the abstract: such lines float into its column.
    """
    groups = _clusters(spans, tags, inp, "ABSTRACT", ends=_ABSTRACT_ENDS if rules else None)
    if not groups:
        return None, None
    group = groups[0]
    words = [i for span in group for i in span.words]
    return _joined(group), sum(conf[i] for i in words) / len(words)


def _keywords(spans: list[Span], tags, inp: TaggerInput, *, rules: bool = False) -> list[str]:
    """The keywords of the list printed first; parallel-language lists are not joined.

    With ``rules`` a keyword is closed over the ")" the tagger left outside it.
    """
    groups = _clusters(spans, tags, inp, "KEYWORD", heading=_KEYWORD_HEADING_RE)
    return [
        k
        for span in (groups[0] if groups else [])
        if (k := _clean(_closed_text(span, tags, inp) if rules else span.text))
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
    # A capital key only after a small letter: "TessA" splits, "COSTAA" does not.
    if (
        key in letters
        and len(name) >= 3
        and name[-2].isalpha()
        and (key.islower() or name[-2].islower())
    ):
        return key
    return None


def _printed_letter_keys(inp: TaggerInput, tags) -> set[str]:
    """Letters "a"-"h" the print keys an affiliation with, wherever the model left one.

    Besides the letters it tagged as marks: a lower-case letter alone at the
    start of a line that goes on with an affiliation ("a Department of X"
    with the "a" untagged), and a letter glued to the line's first word
    ("aComputational Science"). A glued word is taken only where it opens a
    line the model did not read as a name or a title, and only when that line
    reads as an affiliation or a second one opens with another letter:
    "eHealth" in a title and the "eLife" of a running head key nothing.

    Also the letters of a run of keys through the affiliation block
    (:func:`_key_chain`) and capitals printed alone in the byline
    (:func:`_capital_keys`).
    """
    letters: set[str] = set()
    keyed: list[tuple[str, bool]] = []
    words = inp.words
    for i, word in enumerate(words):
        starts_line = i == 0 or words[i - 1].region != word.region
        if _LETTER_KEY_RE.fullmatch(word.text):
            if _field(tags[i]) == "AFF_MARK" or (
                starts_line and i + 1 < len(words) and tags[i + 1] == "B-AFFILIATION"
            ):
                letters.add(word.text)
        elif _LETTER_KEY_RE.fullmatch(_keyed_word_letter(word.text) or "") and _opens_keyed_line(
            tags, i, starts_line
        ):
            keyed.append((word.text[0], _keyed_line_reads_as_affiliation(inp, tags, i)))
    if len({letter for letter, _ in keyed}) >= _MIN_KEYED_LINES:
        letters.update(letter for letter, _ in keyed)
    else:
        letters.update(letter for letter, reads in keyed if reads)
    return letters | _key_chain(inp, tags) | _capital_keys(inp, tags)


def _keyed_word_letter(text: str) -> str | None:
    """The key of a word that opens with one glued on: "a" of "aUniversité"."""
    m = _KEYED_WORD_RE.fullmatch(text)
    if m is None or not m[2][0].isupper() or not m[2][1:].islower():
        return None
    return m[1]


def _key_chain(inp: TaggerInput, tags) -> set[str]:
    """Letters the affiliation block prints as a rising run of keys, however tagged.

    The model often tags one affiliation's key, or none, and runs the block
    together: "a Univ A, USA bDept B, USA c Univ C". The letters themselves show
    what they are by keying the block in order, where a lone "a" or an "eHealth"
    shows nothing. Only a region with an affiliation in it counts, or a line that
    opens with the key and reads as one; one letter may be skipped, for a key OCR
    damaged, and the skipped letter is a key too.
    """
    words = inp.words
    block = {w.region for i, w in enumerate(words) if _field(tags[i]) in _AFFILIATION_FIELDS}
    chain: list[str] = []
    for i, word in enumerate(words):
        if _field(tags[i]) not in (None, *_AFFILIATION_FIELDS):
            continue
        if not (
            word.region in block
            or (_opens_region(inp, i) and _keyed_line_reads_as_affiliation(inp, tags, i))
        ):
            continue
        if _LONE_LETTER_RE.fullmatch(word.text):
            follows = i + 1 < len(words) and words[i + 1].region == word.region
            letter = word.text if follows and words[i + 1].text[:1].isupper() else None
        else:
            letter = _keyed_word_letter(word.text)
        if letter is None:
            continue
        expected = ord(chain[-1]) + 1 if chain else ord("a")
        if ord(letter) in (expected, expected + 1):
            chain.append(letter)
    if len(chain) < _MIN_KEYED_LINES:
        return set()
    return {chr(c) for c in range(ord("a"), ord(chain[-1]) + 1)}


def _opens_region(inp: TaggerInput, i: int) -> bool:
    """Whether only symbols come before word ``i`` in its region (a footnote's "*")."""
    words = inp.words
    j = i - 1
    while j >= 0 and words[j].region == words[i].region:
        if any(c.isalnum() for c in words[j].text):
            return False
        j -= 1
    return True


def _capital_keys(inp: TaggerInput, tags) -> set[str]:
    """Capitals the byline prints alone as keys: the "A" of "Cindy Eng, A"."""
    words = inp.words
    return {
        words[i].text
        for i in range(2, len(words))
        if _CAPITAL_KEY_RE.fullmatch(words[i].text)
        and _field(tags[i]) in (None, "AUTHOR_MARK")
        and words[i - 1].text == ","
        and _field(tags[i - 2]) == "FAMILY"
        and words[i].region == words[i - 2].region
    }


def _opens_keyed_line(tags, i: int, starts_line: bool) -> bool:
    """Whether the word at ``i`` can be an affiliation line's first word, key glued on.

    Its own tag leaves it outside every field but an affiliation's start or a
    mark: a word inside a title, a name, a keyword or an affiliation's text is
    not the head of a line. Away from a region's start it is also untagged and
    follows a tagged word (the name or the affiliation before it).
    """
    if tags[i] not in ("O", "B-AFFILIATION", "B-AFF_MARK"):
        return False
    return starts_line or (tags[i] == "O" and tags[i - 1] != "O")


def _keyed_line_reads_as_affiliation(inp: TaggerInput, tags, i: int) -> bool:
    """Whether the line a glued key opens goes on as an affiliation."""
    words = inp.words
    if i + 1 < len(words) and _field(tags[i + 1]) == "AFFILIATION":
        return True
    last = i
    while last + 1 < len(words) and words[last + 1].region == words[i].region:
        last += 1
    # From the word's second letter on: the key is no part of the institution's name.
    return bool(_INSTITUTION_RE.search(inp.text[words[i].start + 1 : words[last].end]))


def _keys_stand_apart(inp: TaggerInput, tags, last_word: int, key: str) -> bool:
    """Whether the keys printed after a name show its last letter is its own.

    "Silva a" and "Silva, a" print the key apart from the name. "Vegaa, b" is the
    run "a, b" with its first letter glued on, so a higher letter says nothing.
    """
    words = inp.words
    following = [
        words[i]
        for i in range(last_word + 1, min(last_word + 3, len(words)))
        if words[i].region == words[last_word].region and _field(tags[i]) != "GIVEN"
    ]
    if following and _LONE_LETTER_RE.fullmatch(following[0].text):
        return True
    return (
        len(following) == 2
        and following[0].text == ","
        and _LONE_LETTER_RE.fullmatch(following[1].text) is not None
        and following[1].text <= key
    )


def _glue_is_ruled_out(author: _Author, key: str, *, runs: bool) -> bool:
    """Whether the author's own marks rule a glued ``key`` out.

    A mark of any kind does, without ``runs``. With them a symbol ("Vegaa*") says
    nothing about a letter, and neither does a higher letter ("Vegaa,b" read as
    "a" glued and "b" tagged); a digit or numeral shows how this paper keys.
    """
    if not runs:
        return bool(author.marks)
    return any(m.isalnum() and not (_LONE_LETTER_RE.fullmatch(m) and m > key) for m in author.marks)


def _letters_shown_before(inp: TaggerInput, tags, authors: list[_Author], pos: int) -> set[str]:
    """Letter keys the byline has already shown when it reaches author ``pos``."""
    shown = {m for a in authors[:pos] for m in a.marks if _LONE_LETTER_RE.fullmatch(m)}
    if pos:
        for i in range(authors[0].first_word, authors[pos].first_word):
            if _field(tags[i]) is None and _LONE_LETTER_RE.fullmatch(inp.words[i].text):
                shown.add(inp.words[i].text)
    return shown


def _unglue_letter_keys(
    authors: list[_Author],
    by_mark: dict[str, str],
    printed_letters: Collection[str] = (),
    *,
    inp: TaggerInput | None = None,
    tags=None,
) -> None:
    """Split a superscript letter or numeral key OCR glued to a family name.

    On a paper whose affiliations are keyed "a", "b", ... (or "I", "II", ...),
    an author with no mark of their own whose family name ends in one of
    those keys printed it as a superscript: "Vegaa" is "Vega" keyed "a",
    "LIMAII" is "LIMA" keyed "II".

    ``printed_letters`` are letter keys the print shows that no affiliation
    was tagged behind (see :func:`_printed_letter_keys`).

    With ``inp`` and ``tags`` (the front-matter rules), a list of letter keys
    ("a,b") glues on as a run, taken from the end while the letters rise and
    the name keeps four letters or more, at most three. Keys are lettered in
    the order the byline first shows them, so a lower letter comes off only if
    the byline has not shown it yet: "Riosab" opening a byline is "Rios" keyed
    "a" and "b", while "Moreirab" after a name keyed "a" only loses its "b".
    "Sarab" is too short to lose more than its "b", and "Costaa" its last "a".
    A name whose keys the print sets apart keeps its letters.
    """
    runs = inp is not None
    letters = {key for key in by_mark if len(key) == 1 and key.islower()} | set(printed_letters)
    romans = {key for key in by_mark if _ROMAN_KEY_RE.fullmatch(key)}
    if not letters and not romans:
        return
    for pos, author in enumerate(authors):
        if not author.family:
            continue
        last = author.family[-1]
        key = _glued_key(last.text, letters, romans)
        if key is None or _glue_is_ruled_out(author, key, runs=runs):
            continue
        if inp is not None and _keys_stand_apart(inp, tags, last.words[-1], key):
            continue
        shown = _letters_shown_before(inp, tags, authors, pos) if inp is not None else set()
        keys, stem = [key], last.text[: -len(key)]
        while (
            runs
            and len(key) == 1
            and len(keys) < _MAX_GLUED_LETTERS
            and len(stem) > _MIN_RUN_NAME_LETTERS
            and stem[-1] in letters
            and stem[-1] < keys[0]
            and stem[-1] not in shown
            and stem[-2].isalpha()
        ):
            keys.insert(0, stem[-1])
            stem = stem[:-1]
        author.family[-1] = replace(last, text=stem)
        author.marks = [*keys, *author.marks]


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


# Byline repair (``rules``). The tagger often finds one part of a name and leaves
# the rest untagged ("Marise/GIVEN Ph. Born", "Muhammad Muddasir/FAMILY"), skips a
# whole name in a list, or reads a byline printed right under the title as page
# furniture. These words are in print, so they can be read back without a model.
_HONORIFICS = frozenset(
    {"dr", "drs", "prof", "professor", "mr", "mrs", "ms", "miss", "sir", "assoc", "assist", "rev"}
)
# Memberships and degrees printed after a name ("Fellow, IEEE", "PG Dip"). "MA" and
# "BA" stay out, being surnames too ("Ma", "Ba").
_CREDENTIALS = frozenset(
    {"fellow", "member", "senior", "student", "graduate", "ieee", "acm"}
    | {"phd", "dphil", "mphil", "msc", "bsc", "mres", "mph", "pg", "dip", "pgdip", "diploma"}
)
_NOT_GIVEN_WORDS = (
    _HONORIFICS
    | _CREDENTIALS
    | frozenset(
        {"author", "authors", "by", "editor", "editors", "corresponding", "correspondence"}
        | {"citation", "contact", "cite", "article", "reviewer", "reviewers"}
        | {"open", "access", "peer", "reviewed", "original", "research", "review", "brief"}
        | {"report", "received", "revised", "accepted", "published", "contributed", "equally"}
    )
)
_NAME_PARTICLES = frozenset(
    {"van", "der", "den", "de", "del", "della", "di", "da", "dos", "das", "du", "la", "le"}
    | {"von", "zu", "ter", "ten", "bin", "ibn", "al", "el"}
)
_NAME_CONNECTIVES = frozenset({"and", "und", "et", "y", "&"})
_NAME_HYPHENS = frozenset("-\u2010\u2011")
_MIN_NAME_UNITS = 2
_MAX_NAME_UNITS = 4
_MAX_GIVEN_UNITS = 3
_MAX_NAME_GAP_WORDS = 12


def _is_connective(text: str) -> bool:
    """A list word, in any case but a lone capital, which is an initial ("B. Y. Bah")."""
    return text in _NAME_CONNECTIVES or (len(text) > 1 and text.casefold() in _NAME_CONNECTIVES)


def _is_name_word(text: str) -> bool:
    """A capitalised word of letters that is no filler, title or institution."""
    letters = text.replace("-", "").replace("'", "").replace("\u2019", "")
    return (
        text[:1].isupper()
        and letters.isalpha()
        and len(letters) >= 2
        and text.casefold() not in _BYLINE_FILLERS
        and text.casefold() not in _NOT_GIVEN_WORDS
        and not _INSTITUTION_RE.search(text)
    )


def _name_unit_end(inp: TaggerInput, tags: list[str], i: int) -> int:
    """Last word of the name word at ``i``, over hyphens printed tight ("Myeong\u2011Ha")."""
    words = inp.words
    while (
        i + 2 < len(words)
        and words[i + 1].text in _NAME_HYPHENS
        and words[i + 1].start == words[i].end
        and words[i + 2].start == words[i + 1].end
        and words[i + 2].text.isalpha()
        and tags[i + 1] == tags[i + 2] == "O"
    ):
        i += 2
    return i


def _surname_after(inp: TaggerInput, tags: list[str], i: int) -> list[int]:
    """The untagged surname printed from word ``i``: particles and one capitalised
    word, or two when a separator closes them ("García López,")."""
    words = inp.words
    region = words[i - 1].region

    def untagged(j: int) -> bool:
        return j < len(words) and words[j].region == region and tags[j] == "O"

    picked: list[int] = []
    while untagged(i) and words[i].text in _NAME_PARTICLES:
        picked.append(i)
        i += 1
    if not untagged(i) or not _is_name_word(words[i].text):
        return []
    end = _name_unit_end(inp, tags, i)
    had_particle = bool(picked)
    picked.extend(range(i, end + 1))
    following = end + 1
    if had_particle or not untagged(following) or not _is_name_word(words[following].text):
        return picked
    second_end = _name_unit_end(inp, tags, following)
    after = second_end + 1
    closed = (
        after >= len(words)
        or words[after].region != region
        or not words[after].text[:1].isalpha()
        or _is_connective(words[after].text)
    )
    if closed:
        picked.extend(range(following, second_end + 1))
    return picked


def _given_before(inp: TaggerInput, tags: list[str], i: int) -> list[int]:
    """The untagged given names printed right before word ``i + 1``, nearest last."""
    words = inp.words
    region = words[i + 1].region
    family_in_capitals = words[i + 1].text.isupper()
    picked: list[int] = []
    units = 0
    while i >= 0 and units < _MAX_GIVEN_UNITS and words[i].region == region and tags[i] == "O":
        text = words[i].text
        if len(text) > 1 and text.isupper() and not family_in_capitals:
            break  # the end of a title in capitals, run on into the byline
        if (
            text == "."
            and i > 0
            and len(words[i - 1].text) == 1
            and words[i - 1].text.isupper()
            and words[i - 1].region == region
            and tags[i - 1] == "O"
        ):
            picked += [i, i - 1]
            i -= 2
        elif len(text) == 1 and text.isupper():
            picked.append(i)
            i -= 1
        elif _is_name_word(text):
            picked.append(i)
            i -= 1
            while (
                i >= 1
                and words[i].text in _NAME_HYPHENS
                and words[i].end == words[i + 1].start
                and words[i - 1].end == words[i].start
                and words[i - 1].text.isalpha()
                and tags[i] == tags[i - 1] == "O"
            ):
                picked += [i, i - 1]
                i -= 2
        else:
            break
        units += 1
    return sorted(picked)


def _name_units(inp: TaggerInput, tags: list[str], i: int, stop: int) -> list[list[int]]:
    """Name units (a word, an initial and its dot, a particle) in a row from word ``i``."""
    words = inp.words
    units: list[list[int]] = []
    while i < stop and (not units or words[i].region == words[units[-1][-1]].region):
        text = words[i].text
        if text in _NAME_PARTICLES:
            units.append([i])
            i += 1
        elif _is_name_word(text):
            end = _name_unit_end(inp, tags, i)
            units.append(list(range(i, end + 1)))
            i = end + 1
        elif len(text) == 1 and text.isupper():
            has_dot = i + 1 < stop and words[i + 1].text == "."
            units.append([i, i + 1] if has_dot else [i])
            i += 2 if has_dot else 1
        else:
            break
    return units


def _whole_name(inp: TaggerInput, units: list[list[int]]) -> bool:
    """Whether the units read as one person: two to four, a word at each end."""
    if not _MIN_NAME_UNITS <= len(units) <= _MAX_NAME_UNITS:
        return False
    return _is_name_word(inp.words[units[0][0]].text) and _is_name_word(
        inp.words[units[-1][0]].text
    )


def _tag_whole_name(inp: TaggerInput, tags: list[str], units: list[list[int]]) -> None:
    """Tag units as given names and a family name, the particles going to the family."""
    family_from = len(units) - 1
    while family_from > 1 and inp.words[units[family_from - 1][0]].text in _NAME_PARTICLES:
        family_from -= 1
    for pos, unit in enumerate(units):
        name = "GIVEN" if pos < family_from else "FAMILY"
        for i in unit:
            starts = pos in (0, family_from) and i == unit[0]
            tags[i] = f"{'B' if starts else 'I'}-{name}"


def _tidy_name_spans(inp: TaggerInput, tags: list[str], spans: list[Span]) -> None:
    """Cut titles, credentials and list words from the ends of name parts; join a part split in two."""
    words = inp.words
    names = [s for s in spans if s.field in _NAME_FIELDS]
    for span in names:
        kept = list(span.words)
        while kept and (
            _is_connective(words[kept[-1]].text)
            or words[kept[-1]].text in _SYMBOL_MARKS
            or words[kept[-1]].text in _LIST_SEPARATORS
            or (len(kept) > 1 and words[kept[-1]].text.casefold() in _CREDENTIALS)
        ):
            tags[kept.pop()] = "O"
        # A family name alone is a surname, whatever it reads as ("Jennifer Senior").
        while (
            kept
            and (len(kept) > 1 or span.field == "GIVEN")
            and words[kept[0]].text.casefold() in _HONORIFICS
        ):
            tags[kept.pop(0)] = "O"
            if kept and words[kept[0]].text == ".":
                tags[kept.pop(0)] = "O"
        if kept:
            tags[kept[0]] = f"B-{span.field}"
    for a, b in zip(names, names[1:], strict=False):
        if (
            a.field == b.field
            and b.words[0] == a.words[-1] + 1
            and words[a.words[-1]].region == words[b.words[0]].region
            and tags[b.words[0]] == f"B-{b.field}"
            and tags[a.words[-1]] != "O"
            and _fold(b.text) not in _PUBLISHER_NAMES
        ):
            tags[b.words[0]] = f"I-{b.field}"


def _complete_names(inp: TaggerInput, tags: list[str], authors: list[_Author]) -> None:
    """Tag the part of a name printed next to the tagged part."""
    for author in authors:
        if author.given and author.family or not _incomplete_name(author) or _is_publisher(author):
            continue
        if author.given:
            picked = _surname_after(inp, tags, author.last_word + 1)
            name = "FAMILY"
        else:
            picked = _given_before(inp, tags, author.first_word - 1) if author.first_word else []
            name = "GIVEN"
        for pos, i in enumerate(picked):
            tags[i] = f"{'I' if pos else 'B'}-{name}"


def _names_between(inp: TaggerInput, tags: list[str], authors: list[_Author]) -> None:
    """Tag whole names the tagger skipped in a list: between two authors, and after
    the last one when a connective leads to it ("..., and Dimitri van der Linden")."""
    words = inp.words
    for a, b in zip(authors, authors[1:], strict=False):
        lo, hi = a.last_word + 1, b.first_word
        gap = range(lo, hi)
        if not 0 < len(gap) <= _MAX_NAME_GAP_WORDS or any(
            _field(tags[i]) not in (None, "AUTHOR_MARK", "AFF_MARK") for i in gap
        ):
            continue
        if any(
            _INSTITUTION_RE.search(words[i].text)
            or words[i].text.casefold() in _CREDENTIALS
            or (
                words[i].text[:1].islower()
                and words[i].text not in _NAME_PARTICLES
                and not _is_connective(words[i].text)
            )
            for i in gap
        ):
            continue
        i = lo
        while i < hi:
            units = _name_units(inp, tags, i, hi)
            if not units:
                i += 1
                continue
            if _whole_name(inp, units) and all(tags[j] == "O" for unit in units for j in unit):
                _tag_whole_name(inp, tags, units)
            i = units[-1][-1] + 1
    if not authors:
        return
    i = authors[-1].last_word + 1
    region = words[i - 1].region
    while (
        i < len(words)
        and _field(tags[i]) in ("AUTHOR_MARK", "AFF_MARK")
        or (i < len(words) and words[i].text in _LIST_SEPARATORS)
    ):
        i += 1
    if i >= len(words) or not _is_connective(words[i].text):
        return
    if words[i].region != region or tags[i] != "O":
        return
    if i + 1 >= len(words) or words[i + 1].region != region:
        return
    units = _name_units(inp, tags, i + 1, len(words))
    for cut, unit in enumerate(units):
        if words[unit[0]].text in _NAME_PARTICLES:
            family = cut
            while family < len(units) and words[units[family][0]].text in _NAME_PARTICLES:
                family += 1
            units = units[: family + 1]
            break
    named = units if all(tags[i] == "O" for unit in units for i in unit) else []
    if _whole_name(inp, named):
        _tag_whole_name(inp, tags, named)


def _byline_under_title(inp: TaggerInput, tags: list[str], spans: list[Span]) -> None:
    """Tag a list of names printed right under the title when no author is tagged.

    The region after the title, tagged nothing or furniture, has to read as at
    least two people split by list separators and nothing else. One name alone
    is no evidence: "Research Article" reads the same.
    """
    words = inp.words
    title = _first(spans, "TITLE")
    if title is None:
        return
    last = next((s for s in reversed(spans) if s.field == "TITLE"), title).words[-1]
    region = words[last].region + 1
    members = [i for i, w in enumerate(words) if w.region == region]
    if not members or any(_field(tags[i]) not in (None, "FURNITURE") for i in members):
        return
    names: list[list[list[int]]] = []
    i, stop = members[0], members[-1] + 1
    while i < stop:
        if words[i].text in _LIST_SEPARATORS or _is_connective(words[i].text):
            i += 1
            continue
        units = _name_units(inp, tags, i, stop)
        if not _whole_name(inp, units):
            return
        names.append(units)
        i = units[-1][-1] + 1
    if len(names) < _MIN_NAME_UNITS:
        return
    for units in names:
        _tag_whole_name(inp, tags, units)


def _repair_byline(inp: TaggerInput, tags: Sequence[str], conf: Sequence[float]) -> list[str]:
    """The tags with the name parts in print that the tagger left out added back."""
    repaired = list(tags)
    _tidy_name_spans(inp, repaired, _spans(inp, repaired, conf))
    spans = _spans(inp, repaired, conf)
    authors = _group_authors(spans)
    if not authors:
        _byline_under_title(inp, repaired, spans)
        return repaired
    _complete_names(inp, repaired, authors)
    _names_between(inp, repaired, _group_authors(_spans(inp, repaired, conf)))
    return repaired


def _words_between(inp: TaggerInput, tags, lo: int, hi: int, name: str) -> list[str]:
    return [inp.words[i].text for i in range(lo, hi) if _field(tags[i]) == name]


def assemble(
    inp: TaggerInput,
    tags: Sequence[str],
    confidences: Sequence[float],
    *,
    rules: bool = False,
) -> TaggedFrontMatter:
    """Build the front-matter answers from one paper's word tags.

    ``tags`` holds one BIO tag per ``inp.words`` entry and ``confidences`` the
    posterior of that tag (``1.0`` everywhere when the runtime has none).

    ``rules`` adds the readings that put right what a run with no LLM behind
    it would otherwise keep: a keyword closed over its parenthesis, a
    publisher's logo dropped from the authors, letter keys glued to family
    names split off (from the print and as a run), the parts of names and
    the whole names in a byline the tags left out, an abstract kept whole
    across what floats into it, and the institution text the tags left out
    (:attr:`TaggedFrontMatter.untagged_affiliations`).
    Without it the answers are the tagger's as before, and the LLM, when
    there is one, covers the rest.
    """
    if not len(tags) == len(confidences) == len(inp.words):
        raise ValueError("tags and confidences must have one entry per tagger word")
    if rules:
        tags = _repair_byline(inp, tags, confidences)
    spans = _spans(inp, tags, confidences)
    problems: list[str] = []
    confidence: dict[str, float] = {}

    title, title_conf, title_problems = _title(spans, tags, inp, confidences)
    problems.extend(title_problems)
    confidence["title"] = title_conf
    abstract, abstract_conf = _abstract(spans, tags, inp, confidences, rules=rules)
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
        keywords=_keywords(spans, tags, inp, rules=rules),
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

    authors = [a for a in _group_authors(spans) if not (rules and _is_publisher(a))]
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

    affiliation_by_mark, unmarked_affiliations = _affiliations(spans, inp, tags, rules=rules)
    _unglue_letter_keys(
        authors,
        affiliation_by_mark,
        _printed_letter_keys(inp, tags) if rules else (),
        inp=inp if rules else None,
        tags=tags if rules else None,
    )
    unattached = _link_affiliations(authors, affiliation_by_mark, unmarked_affiliations, problems)
    _assign_loose_emails(authors, spans, rules=rules)
    _flag_corresponding(authors, inp, tags)
    first_word = {
        text: span.words[0]
        for span in reversed(spans)
        if span.field == "AFFILIATION" and (text := _clean(span.text))
    }
    # Initials in brackets, tagged as an affiliation, are no row of the table.
    unattached = [t for t in unattached if sum(c.isalpha() for c in t) >= _MIN_AFFILIATION_LETTERS]
    unattached.sort(key=lambda text: first_word.get(text, 0))
    carried = any(_fold(a.affiliation) for a in authors)

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
        unattached_affiliations=unattached,
        untagged_affiliations=_untagged_institutions(inp, tags) if rules and not carried else [],
    )


def _affiliations(
    spans: list[Span], inp: TaggerInput, tags, *, rules: bool = False
) -> tuple[dict[str, str], list[tuple[int, str]]]:
    """Affiliations keyed by the mark printed in front of them, and unmarked ones.

    Unmarked affiliations keep their first word index, for linking by position.
    With ``rules``, a mark that is a letter glued to the affiliation's first word
    ("aUniversità") keys it with the letter and gives the word back to its text.
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
        marks = _words_between(inp, tags, previous_end, span.words[0], "AFF_MARK")
        keys = _mark_keys(marks)
        previous_end = span.words[-1] + 1
        if rules and len(marks) == 1 and (letter := _keyed_word_letter(marks[0])):
            keys, text = [letter], f"{marks[0][1:]} {text}"
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
) -> list[str]:
    """Give each author their affiliations; return the tagged ones no author took."""
    tagged = list(dict.fromkeys([*by_mark.values(), *(text for _, text in unmarked)]))
    if not authors:
        return tagged
    taken: set[str] = set()
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
            taken.update(texts)
        return [text for text in tagged if text not in taken]
    if not unmarked:
        return []
    if len(unmarked) == 1:
        for author in authors:
            author.affiliation = unmarked[0][1]
        return []
    # Several unmarked affiliations: link each to the authors printed between
    # it and the previous one ("Name / Institution / Name / Institution").
    starts = [start for start, _ in unmarked]
    if all(start > authors[-1].last_word for start in starts):
        problems.append("affiliations_unlinked")
        return tagged
    for author in authors:
        following = [text for start, text in unmarked if start > author.last_word]
        author.affiliation = following[0] if following else ""
        taken.add(author.affiliation)
    if any(not a.affiliation for a in authors):
        problems.append("affiliations_unlinked")
    return [text for text in tagged if text not in taken]


def _reads_as_affiliation(text: str) -> bool:
    """Whether a line of untagged text names an institution and nothing else.

    An institution word is not enough: a journal's name, a download stamp, a
    funder, an imprint or a sentence of the paper holds one as well.
    """
    return (
        bool(_INSTITUTION_RE.search(text))
        and not _NOT_AFFILIATION_RE.match(text)
        and not _NOT_AFFILIATION_ANYWHERE_RE.search(text)
        and not _PROSE_RE.search(text)
        and not (_JOB_TITLE_RE.match(text) and "," not in text)
    )


def _untagged_institutions(inp: TaggerInput, tags) -> list[str]:
    """Institution text the tagger left out of every field, before the abstract.

    Two rules, for a paper in which no author took an affiliation:

    * a run of untagged words after a family name, in the byline's region
      ("Arnold K. Ho University of Michigan"), when it names an institution;
    * a region with no tagged word at all, of text, heading or header kind,
      that names an institution.

    Both stop at the abstract or at the first heading of the body (the input
    is the tagger's window of pages, and a paper with a cover sheet prints its
    affiliations on the second), and both leave out what reads as something
    else (see :func:`_reads_as_affiliation`).
    """
    words = inp.words
    by_region: dict[int, list[int]] = {}
    for i, word in enumerate(words):
        by_region.setdefault(word.region, []).append(i)

    def printed(indices: list[int]) -> str:
        """The words as printed: each run of adjacent words as one slice of the input."""
        runs: list[list[int]] = []
        for i in indices:
            if runs and i == runs[-1][-1] + 1:
                runs[-1].append(i)
            else:
                runs.append([i])
        text = " ".join(inp.text[words[run[0]].start : words[run[-1]].end] for run in runs)
        return _ADDRESS_CUE_RE.sub("", re.sub(r"\s+", " ", text)).strip(" ,;·•*")

    found: list[str] = []
    for region, indices in by_region.items():
        fields = [_field(tags[i]) for i in indices]
        if "ABSTRACT" in fields or _SECTION_HEADING_RE.match(inp.regions[region].text):
            break
        if any(f in _NAME_FIELDS for f in fields):
            at = 0
            while at < len(indices):
                if fields[at] != "FAMILY":
                    at += 1
                    continue
                end = at + 1
                while end < len(indices) and fields[end] in (None, "AUTHOR_MARK", "AFF_MARK"):
                    end += 1
                run = [indices[n] for n in range(at + 1, end) if fields[n] is None]
                text = printed(run)
                if 2 <= len(run) <= _MAX_UNTAGGED_RUN_WORDS and _reads_as_affiliation(text):
                    found.append(text)
                at = end
        elif all(f is None for f in fields) and inp.regions[region].label in (
            "text",
            "paragraph_title",
            "header",
        ):
            text = printed(indices)
            if 2 <= len(indices) <= _MAX_UNTAGGED_LINE_WORDS and _reads_as_affiliation(text):
                found.append(text)
    return list(dict.fromkeys(found))


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if c.isalnum() and not unicodedata.combining(c)).lower()


def _assign_loose_emails(authors: list[_Author], spans: list[Span], *, rules: bool = False) -> None:
    """Give an email printed away from the byline to the author it names.

    With ``rules``, of two names the address holds, one inside the other ("Berg"
    and "Steinberg" in "omsteinberg@"), it names the longer.
    """
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
        if rules:
            families = [_fold(" ".join(s.text for s in a.family)) for a in matches]
            matches = [
                a
                for a, family in zip(matches, families, strict=True)
                if not any(family != other and family in other for other in families)
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
