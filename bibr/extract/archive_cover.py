"""Take a title's word breaks from the article, not from an open-archive cover.

An open archive such as HAL, or a distributor such as Cairn.info, puts a
cover page in front of the article's PDF. The cover retypes the title from the
deposit's metadata, and that copy can lose a hyphen the article prints: "...
NEW CADMIUM-" over "AND THALLIUM-CONTAINING ..." becomes "CADMIUMAND
THALLIUM-CONTAINING" on the cover, and "transféro-contre-transférentielles"
becomes "transférocontre-transférentielles" in a cover's text layer. The
metadata model reads the cover and copies it, and grounding accepts the copy
because the cover prints it.

The repair keeps the letters and changes only the word breaks. It fires when
the grounded title is printed on a cover page that carries the archive's or
distributor's own notice, one text block of the next page prints the same
text once spaces and hyphens are removed, and every word the cover joined is
a word the article never prints. A cover word that keeps a hyphen of its own
counts only when the article prints it as one word with more hyphens, each
of the cover's pieces a run of the article's. Titles that differ in any
letter (a legacy OCR layer misreading the article's own title, say) keep the
cover's text.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from bibr.extract.front_matter import FrontMatterResolution

# The notice HAL prints on every deposit cover, in English and in French, and
# the one Cairn.info prints at the foot of its distribution cover.
_ARCHIVE_COVER_NOTICE_RE = re.compile(
    r"\bHAL\s+is\s+a\s+multi-disciplinary\s+open\s+access\s+archive\b"
    r"|\barchive\s+ouverte\s+pluridisciplinaire\s+HAL\b"
    r"|\bdistribution\s+[e\u00e9]lectronique\s+cairn\.info\b",
    re.IGNORECASE,
)
_WORD_BREAK_RE = re.compile(r"[\s\-\u00ad\u2010\u2011]+")
_WORD_RE = re.compile(r"\w+")
_HYPHEN_RE = re.compile(r"[\-\u2010\u2011]")


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _compact(text: str) -> str:
    return _WORD_BREAK_RE.sub("", text).casefold()


def _pieces(token: str) -> list[str]:
    return _HYPHEN_RE.split(token)


def _joined_words(token: str, printed_tokens: set[str]) -> list[str]:
    """The words the cover joined in *token*, or none when it is not a lost break.

    A run of letters the article prints apart is one joined word
    ("CADMIUMAND"). A token with a hyphen of its own is a lost break only when
    the article prints it as one token with more hyphens, and each of the
    cover's pieces is a run of the article's ("transférocontre-transférentielles"
    for "transféro-contre-transférentielles"); the pieces the cover joined are
    its joined words. Any other token with its own hyphen is a different break,
    not a lost one.
    """

    word = token.strip(".,;:!?()[]'\"").casefold()
    if word.isalnum():
        return [word]
    cover_pieces = _pieces(word)
    if not all(piece.isalnum() for piece in cover_pieces):
        return []
    for printed in sorted(printed_tokens):
        printed_pieces = _pieces(printed.strip(".,;:!?()[]'\"").casefold())
        if len(printed_pieces) <= len(cover_pieces) or "".join(printed_pieces) != "".join(
            cover_pieces
        ):
            continue
        if not all(piece.isalnum() for piece in printed_pieces):
            continue
        joined: list[str] = []
        rest = printed_pieces
        for piece in cover_pieces:
            run = ""
            count = 0
            while run != piece and count < len(rest) and len(run) < len(piece):
                run += rest[count]
                count += 1
            if run != piece:
                break
            if count > 1:
                joined.append(piece)
            rest = rest[count:]
        else:
            if not rest and joined:
                return joined
    return []


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
    # The article page's own title block: its layout title, or a text block
    # when a kicker ("Recherche") took the title label.
    prints = {
        _collapse(region.content)
        for region in regions
        if region.page == page + 1 and region.content
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
    joined_words: list[str] = []
    for token in cover_title.split():
        if token in printed_tokens:
            continue
        words = _joined_words(token, printed_tokens)
        if not words:
            return title, None
        joined_words.extend(words)
    if not joined_words:
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
