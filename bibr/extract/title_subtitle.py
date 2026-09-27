"""Fold a subtitle printed on its own row back into the extracted title.

A title block often prints the main title and then, on the next row, a
subtitle or a numbered series part ("8. Psychotherapy"). The metadata model
sees both rows but regularly returns only the first, and the truncated title
still occurs verbatim on the page, so title grounding accepts it.

The fold is deliberately narrow. It needs the model title to be exactly the
text of one or more consecutive rows of the selected front-matter record, and
it looks only at the single row printed directly after them. That row must be
short and on the same page, and it must not read as anything else a title
block prints nearby: a byline or author name, an affiliation, an abstract, a
DOI, a date or citation line, a label or kicker, a section heading, or a
parallel title in another language or script. Anything uncertain is left
alone.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from bibr.extract.front_matter import (
    _NAME_PARTICLES,
    _ORDINARY_HEADING_TEXT,
    _WORD_RE,
    AFFILIATION_MARKER_RE,
    BYLINE_PROBATION_ROLE,
    CLASSIFIED_BYLINE_TITLE_ROLE,
    _has_person_name_evidence,
    _looks_like_byline,
    _looks_like_separator_name_list,
)
from bibr.input.consolidate_text import strip_affiliation_markers
from bibr.paper_contents import FRONT_MATTER_MASTHEAD_RE, is_exact_front_matter_furniture
from bibr.utils.metadata import is_exact_generic_article_label
from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bibr.extract.front_matter import FrontMatterCandidate, FrontMatterResolution
    from bibr.paper import PaperAuthor

# A title printed across more rows than this is not a title block the fold
# understands.
_MAX_TITLE_ROWS = 3
# "Short" in words and characters: a subtitle, not an abstract or a paragraph.
_MAX_SUBTITLE_WORDS = 20
_MAX_SUBTITLE_CHARS = 200
_NOT_SUBTITLE_ROLES = frozenset(
    {
        "abstract",
        "affiliation",
        "byline",
        "doi",
        BYLINE_PROBATION_ROLE,
        CLASSIFIED_BYLINE_TITLE_ROLE,
    }
)
# The title already ends in printed punctuation that separates it from what
# follows: join with a space, as printed.
_PRINTED_SEPARATOR_END = tuple(":;.?!…–—-")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_LEADING_NUMBERING_RE = re.compile(
    r"^\s*(?:(?:part|teil|partie|parte)\s+)?(?:\d+(?:\.\d+)*|[ivxlc]+)[.):]?\s+", re.IGNORECASE
)
# Opening characters no subtitle row starts with: list bullets and dashes,
# correspondence stars, presenter marks, bracketed notes.
_BAD_OPENING = frozenset("*-–—•·‣○●◦■□[")
# Name-list separators; a subtitle does not use them.
_NAME_SEPARATOR_RE = re.compile(r"[·•‣⁃∙⋅・]")
_CONTACT_RE = re.compile(r"@|https?://|\bwww\.|\bdoi\b|\b10\.\d{4,9}/", re.IGNORECASE)
# History, citation, copyright, correspondence, and classification lines, and
# row labels such as "Running title:" or "Keywords:".
_LABEL_PREFIX_RE = re.compile(
    r"^(?:by\b|received\b|accepted\b|published\b|revised\b|available\s+online\b|"
    r"submitted\b|citation\b|cite\s+(?:this|as)\b|copyright\b|©|\(c\)|correspond\w*|"
    r"e-?mail\b|running\s+(?:title|head)\b|short\s+title\b|key\s*words?\b|abstract\b|"
    r"summary\b|in\s+memoriam\b|in\s+memory\s+of\b|dedicated\s+to\b|edited\s+by\b|"
    r"(?:handling\s+)?editor\b|reviewed\s+by\b|issn\b|isbn\b|vol(?:ume)?\.?\s*\d|"
    r"(?:table|fig(?:ure)?\.?)\s*\d|(?:udc|udk|удк|jel|pacs|msc)\b)",
    re.IGNORECASE,
)
# A journal or proceedings name, not a subtitle.
_VENUE_RE = re.compile(r"\b(?:journal|proceedings)\b", re.IGNORECASE)
_MONTH_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:1[6-9]|20)\d{2}\b")
_DIGIT_GROUP_RE = re.compile(r"\d+")
# "14:1197170", "32(4)": a volume:article or volume(issue) locator.
_LOCATOR_RE = re.compile(r"\d+\s*(?::\s*\d+|\(\s*\d+\s*\))")
# A title footnote marker printed after the subtitle ("... Regions*").
_TRAILING_NOTE_MARK_RE = re.compile(r"\s*[*†‡§¶]+$")
# Rows printed on their own that label something rather than continue a title:
# article-type kickers, author-box and citation-box headings, and the one-word
# section headings of languages the shared alias table does not cover.
_LABEL_ROWS = frozenset(
    {
        "article",
        "author",
        "author information",
        "authors",
        "brief communication",
        "brief report",
        "case report",
        "case study",
        "commentary",
        "communication",
        "editorial",
        "electronic reference",
        "essay",
        "letter",
        "letter to the editor",
        "meta analysis",
        "news",
        "opinion",
        "original paper",
        "perspective",
        "print reference",
        "report",
        "research",
        "research letter",
        "research paper",
        "review",
        "review article",
        "short report",
        "systematic review",
        "to cite this article",
        "viewpoint",
        # French citation boxes
        "pour citer cet article",
        "référence électronique",
        "référence papier",
        # one-word section headings
        "antecedentes",
        "diskussion",
        "einleitung",
        "ergebnisse",
        "hintergrund",
        "inleiding",
        "introducción",
        "introdução",
        "introduzione",
        "methoden",
        "methodik",
        "métodos",
        "résultats",
        "resultados",
        "risultati",
        "введение",
        "вступ",
    }
)
# Function words by language, shared words included in every language that
# uses them. A row whose best-matching languages share none with the title's is
# a parallel title in another language, not a subtitle.
_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    language: frozenset(words.split())
    for language, words in {
        "en": "the of and in for on with to from by an at among between through toward towards "
        "into its their how what why does is are",
        "de": "der die das und in im von zu zur zum für mit bei ein eine einer eines des den dem "
        "auf über als zwischen aus nach",
        "fr": "le la les de des du et en à pour dans sur une un aux au par entre leur leurs d l",
        "es": "el la los las de del y en a para con una un por al sobre entre su sus",
        "pt": "o os a as de da do das dos e em no na nos nas para com uma um ao aos pela pelo "
        "por sobre entre",
        "it": "il lo la gli di del della delle dei degli e in per con nel nella nelle tra fra "
        "sul sulla una un a da",
        "nl": "de het een van en in voor met op bij naar over door tussen uit",
    }.items()
}
_ALL_FUNCTION_WORDS = frozenset().union(*_FUNCTION_WORDS.values())
# A function word of any language keeps a row from being a bare person name;
# surname particles ("da", "van") do not.
_NON_NAME_WORDS = _ALL_FUNCTION_WORDS - _NAME_PARTICLES


def _normalize(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(_PUNCT_RE.sub(" ", folded).split())


def _selected_candidates(
    resolution: FrontMatterResolution,
) -> tuple[FrontMatterCandidate, ...]:
    block = next(
        (block for block in resolution.blocks if block.block_id == resolution.selected_block_id),
        None,
    )
    if block is None:
        return ()
    by_id = {candidate.candidate_id: candidate for candidate in resolution.candidates}
    return tuple(
        by_id[candidate_id] for candidate_id in block.candidate_ids if candidate_id in by_id
    )


def _dominant_script(text: str) -> str | None:
    counts: dict[str, int] = {}
    for char in text:
        if char.isalpha():
            script = unicodedata.name(char, "").split(" ", 1)[0]
            counts[script] = counts.get(script, 0) + 1
    if not counts:
        return None
    script, count = max(counts.items(), key=lambda item: item[1])
    return script if count * 5 >= sum(counts.values()) * 3 else None


def _function_word_languages(text: str) -> frozenset[str]:
    """The languages whose function words the text uses most; empty when it uses none."""

    words = {part.casefold() for word in _WORD_RE.findall(text) for part in re.split(r"['’]", word)}
    counts = {language: len(words & vocabulary) for language, vocabulary in _FUNCTION_WORDS.items()}
    best = max(counts.values())
    return frozenset(language for language, count in counts.items() if best and count == best)


def _is_parallel_title(row: str, title: str) -> bool:
    """Whether *row* is known to be in another script or language than *title*."""

    row_script, title_script = _dominant_script(row), _dominant_script(title)
    if row_script and title_script and row_script != title_script:
        return True
    row_languages = _function_word_languages(row)
    title_languages = _function_word_languages(title)
    return bool(row_languages and title_languages and not row_languages & title_languages)


def _is_bare_name(text: str) -> bool:
    """A lone author name: two to four capitalized words and nothing else.

    Also a small-caps name OCR split into its initial letters ("H ARRIS S HULTZ").
    """

    if any(char.isdigit() for char in text):
        return False
    words = _WORD_RE.findall(text)
    if sum(len(word) == 1 and word.isupper() for word in words[1:]) >= 2:
        return True
    if not 2 <= len(words) <= 4:
        return False
    return all(
        (word[:1].isupper() or word.casefold() in _NAME_PARTICLES)
        and word.casefold() not in _NON_NAME_WORDS
        for word in words
    )


def _names_an_author(text: str, authors: Iterable[PaperAuthor]) -> bool:
    """Whether the row prints a given or family name of an extracted author."""

    words = set(_normalize(text).split())
    return any(
        name in words
        for author in authors
        for name in _normalize(f"{author.given or ''} {author.family or ''}").split()
        if len(name) > 2 and name not in _NON_NAME_WORDS and name not in _NAME_PARTICLES
    )


def _is_label_row(text: str) -> bool:
    if is_exact_front_matter_furniture(text) or is_exact_generic_article_label(text):
        return True
    if FRONT_MATTER_MASTHEAD_RE.match(text) or _VENUE_RE.search(text):
        return True
    unnumbered = _normalize(_LEADING_NUMBERING_RE.sub("", text))
    return unnumbered in _LABEL_ROWS or unnumbered in _ORDINARY_HEADING_TEXT


def _is_date_or_citation(text: str) -> bool:
    if _LOCATOR_RE.search(text):
        return True
    if _YEAR_RE.search(text) is None:
        return False
    return bool(_MONTH_RE.search(text)) or len(_DIGIT_GROUP_RE.findall(text)) >= 2


def _is_subtitle_row(
    row: FrontMatterCandidate,
    title_rows: tuple[FrontMatterCandidate, ...],
    title: str,
    authors: Iterable[PaperAuthor],
) -> bool:
    text = " ".join(row.raw_text.split())
    last_title_row = title_rows[-1]
    if row.roles & _NOT_SUBTITLE_ROLES:
        return False
    if row.page is not None and last_title_row.page is not None and row.page != last_title_row.page:
        return False
    if (
        row.bbox is not None
        and last_title_row.bbox is not None
        and row.bbox[1] < last_title_row.bbox[1]
    ):
        # Reading order put a row printed above the title after it.
        return False
    words = _WORD_RE.findall(text)
    if (
        not any(len(word) >= 3 for word in words)
        or len(words) > _MAX_SUBTITLE_WORDS
        or len(text) > _MAX_SUBTITLE_CHARS
    ):
        return False
    # Already in the title, or the title printed again (a running head repeats it).
    normalized, normalized_title = _normalize(text), _normalize(title)
    if normalized in normalized_title or normalized_title in normalized:
        return False
    # A bullet, a note or a star opens it; a lead-in or a salutation ends it.
    if (
        text[0] in _BAD_OPENING
        or unicodedata.category(text[0]).startswith("S")
        or (text[0] == "(" and text.endswith(")"))
        or text.endswith((":", ","))
    ):
        return False
    if (
        _CONTACT_RE.search(text)
        or _NAME_SEPARATOR_RE.search(text)
        or _LABEL_PREFIX_RE.match(text)
        or _is_date_or_citation(text)
        or _is_label_row(text)
    ):
        return False
    # Series numbering ("V. Psychotherapy") is not a middle initial.
    unnumbered = _LEADING_NUMBERING_RE.sub("", text)
    folded = " ".join(unnumbered.casefold().split())
    if (
        AFFILIATION_MARKER_RE.search(text)
        or _looks_like_byline(unnumbered, folded, source_kind="paragraph")
        or _looks_like_separator_name_list(unnumbered, folded)
        or _has_person_name_evidence(unnumbered)
        or _is_bare_name(unnumbered)
        or _names_an_author(text, authors)
    ):
        return False
    return not _is_parallel_title(text, title)


def _separator(stem: str, subtitle: str) -> str:
    """Return ": " before a subtitle, or a space where the rows read as one phrase."""

    last_words = _WORD_RE.findall(stem)
    continues = subtitle[:1].islower() or (
        bool(last_words)
        and len(last_words[-1]) > 1
        and last_words[-1].casefold() in _ALL_FUNCTION_WORDS
    )
    return " " if continues or stem.endswith(_PRINTED_SEPARATOR_END) else ": "


def _title_runs(
    selected: tuple[FrontMatterCandidate, ...], normalized_title: str
) -> list[tuple[int, int]]:
    """Index ranges of consecutive selected rows whose joined text is the title."""

    runs = []
    for start in range(len(selected)):
        joined = ""
        for end in range(start, min(start + _MAX_TITLE_ROWS, len(selected))):
            joined = f"{joined} {_normalize(selected[end].raw_text)}".strip()
            if joined == normalized_title:
                runs.append((start, end))
                break
            if not normalized_title.startswith(f"{joined} "):
                break
    return runs


def fold_printed_subtitle(
    title: str,
    resolution: FrontMatterResolution | None,
    *,
    authors: Iterable[PaperAuthor] = (),
) -> tuple[str, ValidationIssue | None]:
    """Append the subtitle row printed directly under the extracted title.

    Returns the title to use and a warning when a row was folded in. The fold
    joins with ": ", or with a space when the title already ends in printed
    separating punctuation or the two rows read as one phrase (the title ends
    in a function word, or the row starts in lower case). *authors* are the
    extracted authors: a row naming one of them is a byline, whatever its
    shape.
    """

    normalized_title = _normalize(title)
    if (
        resolution is None
        or not normalized_title
        or is_exact_generic_article_label(title)
        or is_exact_front_matter_furniture(title)
    ):
        return title, None
    selected = _selected_candidates(resolution)
    authors = tuple(authors)
    subtitles: dict[str, FrontMatterCandidate] = {}
    for start, end in _title_runs(selected, normalized_title):
        title_rows = selected[start : end + 1]
        if end + 1 >= len(selected) or not any("title" in row.roles for row in title_rows):
            continue
        row = selected[end + 1]
        if _is_subtitle_row(row, title_rows, title, authors):
            subtitles.setdefault(_normalize(row.raw_text), row)
    # The title printed twice (a cover page and the article page) may repeat its
    # subtitle; two different rows are ambiguous.
    if len(subtitles) != 1:
        return title, None
    row = next(iter(subtitles.values()))
    subtitle = _TRAILING_NOTE_MARK_RE.sub("", strip_affiliation_markers(row.raw_text))
    stem = title.rstrip()
    return f"{stem}{_separator(stem, subtitle)}{subtitle}", ValidationIssue(
        code="VAL_TITLE_REGROUNDED",
        severity=IssueSeverity.WARNING,
        message=(
            "Extracted title stopped before the subtitle printed on the next row of "
            "the selected title block; appended it"
        ),
        origin_stage="extract",
        evidence_ids=(row.candidate_id, "reason:title_subtitle_row_dropped"),
        count=1,
    )


__all__ = ["fold_printed_subtitle"]
