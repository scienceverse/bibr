"""Take a title's word breaks from the article, not from an open-archive cover.

An open archive such as HAL puts a cover page in front of the deposited PDF.
The cover retypes the title from the deposit's metadata, and that copy can
lose a line-break hyphen the article prints: "... NEW CADMIUM-" over "AND
THALLIUM-CONTAINING ..." becomes "CADMIUMAND THALLIUM-CONTAINING" on the cover.
The metadata model reads the cover and copies it, and grounding accepts the
copy because the cover prints it.

The repair keeps the letters and changes only the word breaks. It fires when
the grounded title is printed on a cover page that carries the archive's own
notice, the next page's layout title is the same text once spaces and hyphens
are removed, and every word the cover joined is a word the article never
prints. Titles that differ in any letter (a legacy OCR layer misreading the
article's own title, say) keep the cover's text.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from bibr.extract.front_matter import FrontMatterResolution

# The notice HAL prints on every deposit cover, in English and in French.
_ARCHIVE_COVER_NOTICE_RE = re.compile(
    r"\bHAL\s+is\s+a\s+multi-disciplinary\s+open\s+access\s+archive\b"
    r"|\barchive\s+ouverte\s+pluridisciplinaire\s+HAL\b",
    re.IGNORECASE,
)
_WORD_BREAK_RE = re.compile(r"[\s\-\u00ad\u2010\u2011]+")
_WORD_RE = re.compile(r"\w+")


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _compact(text: str) -> str:
    return _WORD_BREAK_RE.sub("", text).casefold()


def _title_page(resolution: FrontMatterResolution) -> int | None:
    """The page of the selected record's first title row."""

    selected = next(
        (block for block in resolution.blocks if block.block_id == resolution.selected_block_id),
        None,
    )
    if selected is None:
        return None
    by_id = {candidate.candidate_id: candidate for candidate in resolution.candidates}
    for candidate_id in selected.title_candidate_ids:
        candidate = by_id.get(candidate_id)
        if candidate is not None and candidate.page is not None:
            return candidate.page
    return None


def reground_title_off_archive_cover(
    title: str,
    resolution: FrontMatterResolution | None,
    contents: Any,
) -> tuple[str, ValidationIssue | None]:
    """Return the article's printing of a title the archive cover retyped.

    *contents* is the paper's ``PaperContents``; its region summaries hold the
    cover's text and the article's own layout title, which the parser keeps
    out of the body as a repeat of the cover's. Returns the title unchanged
    and no issue unless every condition in the module docstring holds.
    """

    if not title.strip() or resolution is None:
        return title, None
    page = _title_page(resolution)
    regions = getattr(contents, "region_summaries", None)
    if page is None or not isinstance(regions, list):
        return title, None
    cover_text = _collapse(
        " ".join(region.content or "" for region in regions if region.page == page)
    )
    if not _ARCHIVE_COVER_NOTICE_RE.search(cover_text):
        return title, None
    cover_title = _collapse(title)
    if cover_title.casefold() not in cover_text.casefold():
        return title, None
    prints = {
        _collapse(region.content)
        for region in regions
        if region.page == page + 1 and region.label == "doc_title" and region.content
    }
    matching = [
        printed
        for printed in prints
        if printed != cover_title and _compact(printed) == _compact(cover_title)
    ]
    if len(matching) != 1:
        return title, None
    printed = matching[0]
    printed_tokens = set(printed.split())
    joined = [token for token in cover_title.split() if token not in printed_tokens]
    joined_words = [token.strip(".,;:!?()[]'\"").casefold() for token in joined]
    # Only a run of letters the article prints apart counts as a joined word;
    # a token with its own hyphen is a different break, not a lost one.
    if not joined_words or not all(word.isalnum() for word in joined_words):
        return title, None
    sentences = getattr(contents, "sentences", None) or []
    article_words = {
        word
        for sentence in sentences
        if sentence.page_number is not None and sentence.page_number != page
        for word in _WORD_RE.findall(sentence.text.casefold())
    }
    if any(word in article_words for word in joined_words):
        return title, None
    return printed, ValidationIssue(
        code="VAL_TITLE_REGROUNDED",
        severity=IssueSeverity.WARNING,
        message=(
            "Extracted title was the open-archive cover's retyping; took the word breaks "
            "the article's own title prints"
        ),
        origin_stage="extract",
        evidence_ids=("reason:title_retyped_on_archive_cover",),
        count=1,
    )


__all__ = ["reground_title_off_archive_cover"]
