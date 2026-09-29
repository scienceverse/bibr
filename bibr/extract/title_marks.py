"""Drop a note marker or a review-status tag the model copied into the title.

A title row often carries the marker of a note about the article ("... waste
management1" over a note "1 Paper extracted from a doctoral dissertation"),
and a text layer or OCR keeps the raised glyph as a plain or superscript digit
glued to the last word. Open-review platforms append the article's review
status to the printed title ("... [version 1; peer review: 2 approved]").
The metadata model copies either, and grounding accepts the copy because the
title row prints it.

A trailing mark is often title text: a digit ("COVID-19", "BRCA1", "TiO2",
"Study 2", "Web 3.0") or a star ("A*", "IDA*", "CTL*"). A mark is dropped only
when the title row of the selected record prints it where the model put it,
after a word of three letters or more, and never after a word with a capital
inside it (BRCA1, TiO2, IDA*) in a title not set in capitals. A digit is
dropped only when the byline and affiliations do not use that number and the
title page numbers it as a note: the first numbered note on the page starts
with it and is not about an author (a corresponding author, an e-mail address,
an affiliation), or, for a raised digit, the markers of the title page's
byline continue straight after it (title 1, authors 2, 3, ...). A plain digit
also needs a last word of six letters or more. An asterisk, dagger or double
dagger is dropped only when a note on the title page that is not about an
author starts with it and the byline does not use it. Everything else is left
alone.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING, Any

from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from bibr.extract.front_matter import FrontMatterCandidate, FrontMatterResolution

# An open-review platform's status tag: "[version 2; peer review: 2 approved]",
# or "referees:" in older articles. A tag cut at a row end may lack its bracket.
_REVIEW_STATUS_TAG_RE = re.compile(
    r"\s*(?P<tag>\[\s*version\s+\d+\s*;\s*(?:peer\s+review|referees)\s*:)[^\[\]]*\]?\s*$",
    re.IGNORECASE,
)
# A note marker ending the title: superscript digits, or plain digits glued to a
# word of three letters or more, or a run of note symbols. U+2217 is the
# asterisk operator some text layers use.
_SUPERSCRIPT_MARK_RE = re.compile(
    r"(?<=\S)\s?(?P<mark>[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]{1,2})$"
)
_GLUED_DIGIT_MARK_RE = re.compile(r"(?<=[^\W\d_]{3})(?P<mark>\d{1,2})$")
_SYMBOL_MARK_RE = re.compile(r"(?<=\S)\s?(?P<mark>[*\u2217\u2020\u2021]{1,3})$")
_NOTE_SYMBOLS = "*\u2020\u2021"
_SYMBOL_RUN_RE = re.compile(r"[*\u2020\u2021]+")
# The stem left once the marker is gone must still read as a title, and end in
# a word of three letters or more, perhaps closed by a full stop, question mark
# or bracket: a symbol or superscript after a shorter token is part of it ("A*",
# "R²", "mc²").
_MIN_STEM_CHARS = 12
_STEM_END_RE = re.compile(r"[^\W\d_]{3}[.?!)\]]?$")
_LAST_WORD_RE = re.compile(r"[^\W\d_]+(?=[.?!)\]]?$)")
# A plain digit glued to a shorter word is too often part of a name ("Nrf2",
# "Keap1", "Sox2") to be read as a marker.
_MIN_GLUED_WORD_LETTERS = 6
# LaTeX superscripts an OCR engine writes for printed markers: "\(^{2}\)", "^{1,2}".
_LATEX_SUPERSCRIPT_RE = re.compile(r"\\\(\s*\^\{([^{}]*)\}\s*\\\)|\^\{([^{}]*)\}")
_NUMBER_LIST = r"\d{1,2}(?:\s*,\s*\d{1,2})*"
# Byline and affiliation markers: glued to a name ("Veiga2", "Howick1,2") or
# opening a row ("1Universidade ...", "1*").
_NAME_MARKER_RE = re.compile(rf"[^\W\d_]{{3,}}(?P<marks>{_NUMBER_LIST})")
_ROW_MARKER_RE = re.compile(
    rf"^\s*(?P<marks>{_NUMBER_LIST})(?=\s*(?:[^\W\d_]|[*\u2020\u2021]|$))", re.MULTILINE
)
# On the title page, also a marker after any name or note symbol, glued or
# set apart by a space or comma, and closed by a symbol, the row end or the
# next word: "Li1", "Chen 1 | Maitra2", "Doe a,1, Roe b,*", "Silva* 1 and",
# "Visscher 1 ●".
_PAGE_MARKER_RE = re.compile(
    rf"(?:[^\W\d_]|[*\u2020\u2021\u00a7\u00b6#])[\s,]?(?P<marks>{_NUMBER_LIST})"
    r"(?=\s*(?:[^\w\s.()\[\]/:\-]|$)|\s+[^\W\d_])",
    re.MULTILINE,
)
# A note opening with its number: "1 Paper extracted ...", "¹Manuscript ...".
_NOTE_NUMBER_RE = re.compile(r"^\s*(?P<number>\d{1,2})(?=\s*[^\W\d_])")
# A note about an author rather than the article, read near its start: a
# corresponding author, an e-mail or reprint address, an equal contribution, or
# an affiliation.
_AUTHOR_NOTE_RE = re.compile(
    r"@|correspond|e-?mail|reprint|contributed\s+equally|equal(?:ly)?\s+contribut"
    r"|(?:present|current|permanent)\s+address"
    r"|universi|d[eé]part[ae]?ment|facult|faculdade|college|school|institut|cent(?:re|er)\b"
    r"|laborat|hospital",
    re.IGNORECASE,
)
_AUTHOR_NOTE_WINDOW = 60


def _fold(text: str) -> str:
    """NFKC, the asterisk operator as an asterisk, collapsed space, casefolded."""

    folded = unicodedata.normalize("NFKC", text).replace("\u2217", "*")
    return " ".join(folded.split()).casefold()


def _unwrap_latex(text: str) -> str:
    return _LATEX_SUPERSCRIPT_RE.sub(lambda m: m.group(1) or m.group(2) or "", text)


def _numbers(marks: str) -> set[int]:
    return {int(number) for number in re.findall(r"\d+", marks)}


def _selected(resolution: FrontMatterResolution) -> tuple[FrontMatterCandidate, ...]:
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


def _title_page(selected: tuple[FrontMatterCandidate, ...]) -> int | None:
    return next((candidate.page for candidate in selected if "title" in candidate.roles), None)


def _byline_rows(selected: tuple[FrontMatterCandidate, ...]) -> list[FrontMatterCandidate]:
    return [candidate for candidate in selected if candidate.roles & {"byline", "affiliation"}]


def _marker_text(candidate: FrontMatterCandidate) -> str:
    return unicodedata.normalize("NFKC", _unwrap_latex(candidate.raw_text)).replace("\u2217", "*")


def _byline_markers(
    selected: tuple[FrontMatterCandidate, ...], *, title_page_only: bool = False
) -> set[int]:
    """Numbers the selected byline and affiliation rows use as markers.

    Every such row of the record counts with the glued and row-opening
    markers; the rows on the title page count with any name marker too. With
    *title_page_only*, only the title page's rows count.
    """

    page = _title_page(selected)
    markers: set[int] = set()
    for candidate in _byline_rows(selected):
        on_title_page = candidate.page == page
        if title_page_only and not on_title_page:
            continue
        text = _marker_text(candidate)
        patterns = [_NAME_MARKER_RE, _ROW_MARKER_RE]
        if on_title_page:
            patterns.append(_PAGE_MARKER_RE)
        for pattern in patterns:
            for match in pattern.finditer(text):
                markers |= _numbers(match.group("marks"))
    return markers


def _byline_symbols(selected: tuple[FrontMatterCandidate, ...], title_tail: str) -> set[str]:
    """The runs of note symbols the byline and affiliation rows print as markers.

    A run that opens a row is the mark of a note the record took in as a
    byline row, and a row that repeats the title's last word keeps the
    title's own mark: neither is a byline marker.
    """

    runs: set[str] = set()
    for candidate in _byline_rows(selected):
        if "title" in candidate.roles:
            continue
        text = _marker_text(candidate).casefold()
        if title_tail:
            text = text.replace(title_tail, " ")
        for match in _SYMBOL_RUN_RE.finditer(text):
            row_start = text.rfind("\n", 0, match.start()) + 1
            if text[row_start : match.start()].strip():
                runs.add(match.group(0))
    return runs


def _title_page_notes(selected: tuple[FrontMatterCandidate, ...], contents: Any) -> list[str]:
    """The notes printed on the page of the selected title row, NFKC-normalized."""

    page = _title_page(selected)
    regions = getattr(contents, "region_summaries", None)
    if page is None or not isinstance(regions, list):
        return []
    return [
        unicodedata.normalize("NFKC", _unwrap_latex(region.content)).replace("\u2217", "*")
        for region in regions
        if region.page == page and region.label == "footnote" and region.content
    ]


def _about_an_author(note: str) -> bool:
    return _AUTHOR_NOTE_RE.search(note.lstrip()[:_AUTHOR_NOTE_WINDOW]) is not None


def _readings(digits: str) -> list[tuple[int, ...]]:
    """The note numbers a trailing digit run can stand for: "12" is 12, or 1 and 2."""

    readings: list[tuple[int, ...]] = [(int(digits),)]
    if len(digits) == 2 and int(digits[1]) == int(digits[0]) + 1:
        readings.append((int(digits[0]), int(digits[1])))
    return readings


def _numbered_as_note(
    digits: str, selected: tuple[FrontMatterCandidate, ...], contents: Any, *, raised: bool
) -> bool:
    markers = _byline_markers(selected)
    page_markers = _byline_markers(selected, title_page_only=True)
    notes = [
        (int(match.group("number")), note)
        for note in _title_page_notes(selected, contents)
        if (match := _NOTE_NUMBER_RE.match(note)) is not None
    ]
    first = min((number for number, _ in notes), default=None)
    article_notes = {number for number, note in notes if not _about_an_author(note)}
    for reading in _readings(digits):
        if markers & set(reading):
            continue
        if reading[0] == first and first in article_notes:
            return True
        # Only a raised digit is read against the byline's markers: a plain
        # one glued to a word is too often part of a name for that.
        if raised and page_markers and min(page_markers) == reading[-1] + 1:
            return True
    return False


def _symbol_opens_a_note(
    mark: str, selected: tuple[FrontMatterCandidate, ...], contents: Any, title_tail: str
) -> bool:
    if mark in _byline_symbols(selected, title_tail):
        return False
    for note in _title_page_notes(selected, contents):
        stripped = note.lstrip()
        after = stripped[len(mark) : len(mark) + 1]
        if (
            stripped.startswith(mark)
            and not (after and after in _NOTE_SYMBOLS)
            and not _about_an_author(stripped[len(mark) :])
        ):
            return True
    return False


def _issue(reason: str, message: str, evidence: tuple[str, ...]) -> ValidationIssue:
    return ValidationIssue(
        code="VAL_TITLE_REGROUNDED",
        severity=IssueSeverity.WARNING,
        message=message,
        origin_stage="extract",
        evidence_ids=(*evidence, f"reason:{reason}"),
        count=1,
    )


def drop_title_note_marker(
    title: str,
    resolution: FrontMatterResolution | None,
    contents: Any,
) -> tuple[str, ValidationIssue | None]:
    """Return *title* without a trailing note marker or review-status tag.

    *contents* is the paper's ``PaperContents``; its region summaries hold the
    notes printed on the title page. Returns the title unchanged and no issue
    unless the evidence in the module docstring holds.
    """

    if resolution is None or not title.strip():
        return title, None
    selected = _selected(resolution)
    if not selected:
        return title, None
    stripped = title.rstrip()

    tag = _REVIEW_STATUS_TAG_RE.search(stripped)
    if tag is not None:
        stem = stripped[: tag.start()].rstrip()
        printed = _fold(" ".join(candidate.raw_text for candidate in selected))
        if len(stem) >= _MIN_STEM_CHARS and _fold(tag.group("tag")) in printed:
            return stem, _issue(
                "title_review_status_tag",
                "Extracted title ended in the open-review status tag printed after the "
                "title; dropped it",
                (),
            )
        return title, None

    for pattern in (_SUPERSCRIPT_MARK_RE, _GLUED_DIGIT_MARK_RE, _SYMBOL_MARK_RE):
        match = pattern.search(stripped)
        if match is not None:
            break
    else:
        return title, None
    stem = stripped[: match.start()].rstrip()
    if len(stem) < _MIN_STEM_CHARS or not _STEM_END_RE.search(stem):
        return title, None
    # The title row must print the marker where the model put it.
    tail = _fold(stripped[stem.rfind(" ") + 1 :])
    row = next(
        (
            candidate
            for candidate in selected
            if "title" in candidate.roles and _fold(candidate.raw_text).endswith(tail)
        ),
        None,
    )
    if row is None:
        return title, None
    mark = _fold(match.group("mark"))
    word_match = _LAST_WORD_RE.search(stem)
    word = word_match.group(0) if word_match else ""
    if stem != stem.upper() and any(letter.isupper() for letter in word[1:]):
        # "BRCA1", "TiO2", "IDA*", "SPARQL*": the mark belongs to a name.
        return title, None
    if mark.isdigit():
        raised = match.re is _SUPERSCRIPT_MARK_RE
        if not raised and len(word) < _MIN_GLUED_WORD_LETTERS:
            return title, None
        if not _numbered_as_note(mark, selected, contents, raised=raised):
            return title, None
    elif not _symbol_opens_a_note(mark, selected, contents, tail):
        return title, None
    return stem, _issue(
        "title_note_marker",
        "Extracted title ended in the marker of a note printed on the title page; dropped it",
        (row.candidate_id,),
    )


__all__ = ["drop_title_note_marker"]
