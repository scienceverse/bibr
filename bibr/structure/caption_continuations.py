"""Join caption fragments the layout model split into separate regions.

Every layout region is parsed on its own, so a caption printed in two
columns under a full-width figure, or a bare "FIGURE 1" label above its
title line, reaches the parser as two regions. The labelled fragment becomes
the caption; the other has no label, so it either competes as a caption of
its own and loses (and is replayed into the body), or it is a ``text`` or
``vision_footnote`` region and goes straight to the body or the footnotes.
A lowercase second column even joins the paragraph before the figure.

:func:`find_caption_continuations` runs before dispatch, on geometry and the
cleaned region text, and returns the regions that continue an explicit
caption so :class:`~bibr.structure.pdf_parser.PDFParser` can parse each
caption as one region. Boxes are 0..1000 page space (``bbox_2d``).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from bibr.ocr.types import OcrRegionResult
from bibr.structure.float_labels import LABEL, SUPPLEMENT_WORD
from bibr.structure.text_repair import bbox_to_tuple

RegionKey = tuple[int, int]
Box = tuple[float, float, float, float]

# Labels a caption region can carry, and the ones the content re-router turns
# into a caption when the text reads as one (``PDFParser._FIGURE_CAPTION_RE``).
_CAPTION_LABELS = frozenset({"figure_title", "chart_title"})
_CONTENT_LABELS = frozenset({"text", "content"})
# What a continuation can be: a caption region, or the text and footnote
# regions a caption's second column is often read as. ``text`` only beside
# the caption: below it, a ``text`` region is the next body paragraph.
_SIDE_LABELS = frozenset({"figure_title", "chart_title", "text", "content", "vision_footnote"})
_BELOW_LABELS = frozenset({"figure_title", "chart_title", "vision_footnote"})
_FLOAT_LABELS = frozenset({"image", "chart", "table"})

# A fragment that opens with a float label starts a caption of its own.
_LABEL_START_RE = re.compile(
    rf"^\s*{SUPPLEMENT_WORD}?(?:Figures?|Figs?\.?|Tables?|Tab\.)\s*{LABEL}", re.IGNORECASE
)
_EXPLICIT_CAPTION_RE = re.compile(
    rf"^\s*{SUPPLEMENT_WORD}?(?:Figure|Fig\.?|Table)\s+{LABEL}", re.IGNORECASE
)
_BARE_LABEL_RE = re.compile(
    rf"^\s*{SUPPLEMENT_WORD}?(?:Figure|Fig\.?|Table)\s+{LABEL}\s*[.:]?\s*$", re.IGNORECASE
)
_BARE_TABLE_LABEL_RE = re.compile(rf"^\s*{SUPPLEMENT_WORD}?Table\s+{LABEL}\s*[.:]?\s*$", re.I)
_DOI_OR_URL_ONLY_RE = re.compile(r"^\s*(?:doi:\s*|https?://)\S+\s*$", re.IGNORECASE)
_TERMINAL_RE = re.compile(r"[.!?][\"'”’)\]]*\s*$")

# Side fragment: top-aligned with the caption, starting just right of it,
# ending about where it ends, under (or over) one float spanning both.
_SIDE_TOP_TOLERANCE = 15.0
_SIDE_MAX_GAP = 60.0
_SIDE_BOTTOM_TOLERANCE = 40.0
_FLOAT_SPAN_TOLERANCE = 15.0
_FLOAT_MAX_GAP = 60.0
# Below fragment: directly under the caption and mostly overlapping it.
_BELOW_MIN_GAP = -6.0
_BELOW_MAX_GAP = 20.0
_BELOW_MIN_OVERLAP = 0.6
# A panel marker or a stray glyph is not caption prose.
_MIN_FRAGMENT_CHARS = 15
_MAX_CHAIN = 3


def _horizontal_overlap(left: Box, right: Box) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    return width / max(1.0, min(left[2] - left[0], right[2] - right[0]))


def _starts_lowercase(text: str) -> bool:
    first = next((character for character in text if character.isalnum()), "")
    return first.islower()


def _union(left: Box, right: Box) -> Box:
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def find_caption_continuations(
    pages: list[list[OcrRegionResult]],
    text_of: Callable[[RegionKey], str],
    effective_label: Callable[[OcrRegionResult], str],
    skip: set[RegionKey],
    is_content_caption: Callable[[str], bool],
) -> dict[RegionKey, list[RegionKey]]:
    """Map each explicit caption region to the regions that continue it.

    *text_of* gives a region's cleaned text, *effective_label* its dispatch
    label, *skip* the regions dispatch never parses as content (running
    headers, shadowed references), and *is_content_caption* whether a body
    region's text is re-routed as a caption. A bare table label is left to
    the parser's own label/fragment staging, which confirms it against the
    table that follows.

    A region continues caption X when it does not open with a float label
    and either

    * **beside:** it is top-aligned with X and starts just right of it, ends
      about where X ends, and one float directly above or below spans both;
    * **below:** it is a caption or footnote region directly under X and
      mostly overlapping it, and X is a bare label, or X stops mid-sentence
      and the fragment reads on in lowercase.

    Beside X, a fragment also needs X to stop mid-sentence or itself to start
    in lowercase. Fragments below chain (a title in two lines).
    """
    continuations: dict[RegionKey, list[RegionKey]] = {}
    for page_idx, regions in enumerate(pages):
        page = _Page()
        for region_idx, region in enumerate(regions):
            box = bbox_to_tuple(region.bbox_2d)
            # A float region has no text; it still counts for geometry.
            if box is None or (page_idx, region_idx) in skip:
                continue
            text = text_of((page_idx, region_idx)).strip()
            page.boxes[region_idx] = box
            page.labels[region_idx] = effective_label(region)
            page.texts[region_idx] = text
        page.floats = [
            page.boxes[index] for index, label in page.labels.items() if label in _FLOAT_LABELS
        ]
        page.captions = [
            index
            for index, label in page.labels.items()
            if (
                (label in _CAPTION_LABELS and _EXPLICIT_CAPTION_RE.match(page.texts[index]))
                or (label in _CONTENT_LABELS and is_content_caption(page.texts[index]))
            )
            and not _BARE_TABLE_LABEL_RE.match(page.texts[index])
        ]
        for caption, joined in page.continuations().items():
            continuations[(page_idx, caption)] = [(page_idx, index) for index in joined]
    return continuations


class _Page:
    """One page's usable regions, by region index."""

    def __init__(self) -> None:
        self.boxes: dict[int, Box] = {}
        self.labels: dict[int, str] = {}
        self.texts: dict[int, str] = {}
        self.floats: list[Box] = []
        self.captions: list[int] = []
        self.claimed: set[int] = set()

    def is_fragment(self, index: int) -> bool:
        text = self.texts[index]
        return (
            index not in self.claimed
            and index not in self.captions
            and len(text) >= _MIN_FRAGMENT_CHARS
            and not _LABEL_START_RE.match(text)
            and not _DOI_OR_URL_ONLY_RE.match(text)
        )

    def spanned(self, left: Box, right: Box) -> bool:
        """Does one float directly above or below span both boxes?"""
        low, high = min(left[0], right[0]), max(left[2], right[2])
        top, bottom = min(left[1], right[1]), max(left[3], right[3])
        return any(
            item[0] <= low + _FLOAT_SPAN_TOLERANCE
            and item[2] >= high - _FLOAT_SPAN_TOLERANCE
            and (-8 <= top - item[3] <= _FLOAT_MAX_GAP or -8 <= item[1] - bottom <= _FLOAT_MAX_GAP)
            for item in self.floats
        )

    def beside(self, caption_box: Box, caption_text: str) -> int | None:
        """The caption's second column, if one is printed beside it."""
        for index in sorted(self.boxes, key=lambda item: self.boxes[item][0]):
            box = self.boxes[index]
            if (
                self.labels[index] in _SIDE_LABELS
                and self.is_fragment(index)
                and abs(box[1] - caption_box[1]) <= _SIDE_TOP_TOLERANCE
                and 0 <= box[0] - caption_box[2] <= _SIDE_MAX_GAP
                and abs(box[3] - caption_box[3]) <= _SIDE_BOTTOM_TOLERANCE
                and self.spanned(caption_box, box)
                and (not _TERMINAL_RE.search(caption_text) or _starts_lowercase(self.texts[index]))
            ):
                return index
        return None

    def below(self, caption_box: Box, *, lowercase: bool = False) -> int | None:
        """The fragment printed directly under the caption, if any."""
        below = [
            index
            for index, box in self.boxes.items()
            if self.labels[index] in _BELOW_LABELS
            and self.is_fragment(index)
            and (not lowercase or _starts_lowercase(self.texts[index]))
            and _BELOW_MIN_GAP <= box[1] - caption_box[3] <= _BELOW_MAX_GAP
            and _horizontal_overlap(caption_box, box) >= _BELOW_MIN_OVERLAP
        ]
        return min(below, key=lambda index: self.boxes[index][1]) if below else None

    def continuations(self) -> dict[int, list[int]]:
        found: dict[int, list[int]] = {}
        for caption in self.captions:
            caption_box, caption_text = self.boxes[caption], self.texts[caption]
            index = self.beside(caption_box, caption_text)
            # A bare label ("FIGURE 1", "Figure 3.") is continued by its title
            # even though it may end in a full stop. A titled caption that
            # stops mid-sentence is continued only by a fragment that reads
            # on in lowercase; a capitalised line under it is a heading or
            # the next block ("Appendix E").
            if index is None and _BARE_LABEL_RE.match(caption_text):
                index = self.below(caption_box)
            elif index is None and not _TERMINAL_RE.search(caption_text):
                index = self.below(caption_box, lowercase=True)
            joined: list[int] = []
            while index is not None:
                joined.append(index)
                self.claimed.add(index)
                caption_box = _union(caption_box, self.boxes[index])
                caption_text = self.texts[index]
                if len(joined) >= _MAX_CHAIN or _TERMINAL_RE.search(caption_text):
                    break
                index = self.below(caption_box, lowercase=True)
            if joined:
                found[caption] = joined
        return found
