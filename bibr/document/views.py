"""Read views over a :class:`DocumentLayer`.

Each view reproduces a string or record bibr's pipeline already computes, from
the layer alone: :func:`block_text` the native fill of a region,
:func:`page_lines` the reference line stream's page lines and
:func:`bbox_pdf_pts` a region's ``_bbox_pdf_pts``. The tests hold them to
byte equality with the pipeline's own values.

The views that read a page's glyph columns give empty results ("", [] or
None) for a page without them: a page with no text layer, a page the
harvest failed on (``Page.error`` is set) and every page once the layer's
columns were freed (``DocumentLayer.columns_freed``). Check those two
before reading an empty result as a page without text. The joins of
:class:`StructIndex` are the exception for a freed layer: an empty join
would read as a paper without tagged text, so they raise
:class:`ColumnsFreedError`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np

from bibr.document import destinations, ids
from bibr.document.model import (
    GLYPH_EXCLUDED,
    GLYPH_HYPHEN,
    GLYPH_NO_BOX,
    Block,
    Box,
    DocumentLayer,
    Link,
    Page,
    StructElem,
    as_box,
)

TEXT_SOURCES = ("native", "invisible_layer", "ocr")


class ColumnsFreedError(RuntimeError):
    """The layer's glyph columns were freed, so the text the structure joins to is gone.

    :meth:`StructIndex.span_element`, :meth:`StructIndex.spans_of` and
    :meth:`StructIndex.link_element` raise it once ``DocumentLayer.columns_freed``
    is set, whatever the index was asked before and whether it was built before
    or after the columns were freed. Join before the pipeline frees them (after
    the last stage that requires the layer), or not at all.
    """


def _finite(values) -> Box | None:
    box = as_box(values)
    return box if all(math.isfinite(value) for value in box) else None


def to_layout_bbox(page: Page, box: Box) -> Box | None:
    """*box* (PDF points) in the layout frame: ``bbox_2d``'s 0..1000 image space.

    The frame of the rendered page image, so the page's ``/Rotate`` applies;
    the result is ``(x1, y1, x2, y2)`` with y down. None, never a NaN box,
    when the page has no geometry or *box* is not finite.
    """
    from bibr.ocr.native_text import _pdf_points_to_normalized_bbox

    if page.crop_box is None or page.rotation is None:
        return None
    return _finite(_pdf_points_to_normalized_bbox(box, page.crop_box, page.rotation))


def from_layout_bbox(page: Page, bbox_2d) -> Box | None:
    """A layout ``bbox_2d`` (0..1000 image space) in the page's PDF points.

    None, never a NaN box, when the page has no geometry or the result is not
    finite.
    """
    from bibr.ocr.native_text import _normalized_bbox_to_pdf_points

    if page.crop_box is None or page.rotation is None:
        return None
    return _finite(_normalized_bbox_to_pdf_points(list(bbox_2d), page.crop_box, page.rotation))


def find_block(layer: DocumentLayer, block_id: str) -> tuple[Page, Block] | None:
    """The page and block with *block_id*; None when the layer has no such block."""
    try:
        parsed = ids.parse(block_id)
    except ValueError:
        return None
    # A document id (an outline entry, a structure element) names no page.
    page = None if parsed.page is None else layer.page(parsed.page)
    if page is None:
        return None
    if parsed.kind == ids.BLOCK and parsed.n < len(page.blocks):
        block = page.blocks[parsed.n]
        if block.block_id == block_id:
            return page, block
    for block in page.blocks:
        if block.block_id == block_id:
            return page, block
    return None


def block_for_region(layer: DocumentLayer, *, page_no: int, region_index: int) -> Block | None:
    """The block of the region a region summary names.

    *page_no* is ``RegionSummary.page``: the absolute page, 1-based, unlike
    every other page in the layer. *region_index* is ``RegionSummary.index``,
    the region's position in the page's post-OCR region list.
    """
    page = layer.page(page_no - 1)
    if page is None or not 0 <= region_index < len(page.blocks):
        return None
    return page.blocks[region_index]


def text_in_box(page: Page, box: Box) -> str:
    """Text of the records whose centre lies in *box* (PDF points).

    The rule of ``_reconstruct_text_from_records``: line breaks between kept
    chars become ``"\\r\\n"``, and breaks before the first or after the last
    kept char are dropped.
    """
    cols = page.cols
    if cols is None:
        return ""
    left, bottom, right, top = box
    newline = cols.rec_newline
    inside = (
        ~newline
        & (cols.rec_cx >= left)
        & (cols.rec_cx <= right)
        & (cols.rec_cy >= bottom)
        & (cols.rec_cy <= top)
    )
    kept = np.flatnonzero(inside).tolist()
    if not kept:
        return ""
    breaks = np.cumsum(newline).tolist()
    parts: list[str] = []
    previous: int | None = None
    for index in kept:
        if previous is not None and breaks[index] != breaks[previous]:
            parts.append("\r\n")
        parts.append(cols.record_char(index))
        previous = index
    return "".join(parts)


def block_text(layer: DocumentLayer, block_id: str, source: str = "native") -> str | None:
    """A block's text from *source*.

    ``native`` reads the text layer in the block's box as the native-text
    fill does (centre containment, then the fill's line-end hyphen repair), so
    for a region the fill took it equals the region's content. It reads only
    born-digital text layers: None unless ``Page.text_source`` is ``native``.
    ``invisible_layer`` is the same read, only on pages whose text layer is a
    hidden OCR layer. ``ocr`` is the region's OCR text, which exists only when
    OCR was the region's chosen source. None when the block or the source is
    absent.

    ``Block.text`` keeps only the chosen source's text; this is how the text
    layer under any block is read.
    """
    if source not in TEXT_SOURCES:
        raise ValueError(f"unknown text source {source!r}")
    found = find_block(layer, block_id)
    if found is None:
        return None
    page, block = found
    if source == "ocr":
        return block.text.get("ocr")
    if page.cols is None or page.text_source != source or block.bbox_pdf is None:
        return None
    text = text_in_box(page, block.bbox_pdf)
    if "\x02" in text:
        from bibr.input.consolidate_text import fix_ocr_artifacts

        text = fix_ocr_artifacts(text)
    return text


def bbox_pdf_pts(page: Page, block: Block) -> list[float] | None:
    """The block's ``_bbox_pdf_pts``: its box relative to the CropBox origin, rounded.

    None for a block without a box, as for its region.
    """
    if block.bbox_pdf is None or page.crop_box is None:
        return None
    left, bottom, right, top = block.bbox_pdf
    cx0, cy0, _cx1, _cy1 = page.crop_box
    return [
        round(left - cx0, 2),
        round(bottom - cy0, 2),
        round(right - cx0, 2),
        round(top - cy0, 2),
    ]


def page_chars(page: Page) -> list[tuple[str, Box]]:
    """The ``(char, tight box)`` stream ``ref_geometry._extract_page_chars`` reads.

    Nothing for a char pdfium leaves out of the page text and for an unpaired
    UTF-16 surrogate or a code past U+10FFFF; a char beyond the BMP, held as
    one code or as a surrogate pair (with the union of the pair's boxes);
    U+FFFE for pdfium's line-end hyphen; otherwise the char itself. Raises
    where that read raises (a missing box).

    [] for a page without columns, including a failed page: check
    ``page.error`` and ``DocumentLayer.columns_freed``.
    """
    from bibr.ocr.native_text import compose_spacing_accents

    cols = page.cols
    if cols is None:
        return []
    chars: list[tuple[str, Box]] = []
    boxes = cols.box.tolist()
    codes = cols.cp.tolist()
    gflags = cols.gflags.tolist()
    index = 0
    while index < len(codes):
        code, flags = codes[index], gflags[index]
        rows = [index]
        if 0xD800 <= code <= 0xDBFF and index + 1 < len(codes):
            low = codes[index + 1]
            if 0xDC00 <= low <= 0xDFFF:
                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                rows.append(index + 1)
        index += len(rows)
        if flags & GLYPH_EXCLUDED:
            continue
        if code == 0x2 and flags & GLYPH_HYPHEN:
            ch = "\ufffe"
        elif 0xD800 <= code <= 0xDFFF or code > 0x10FFFF:
            continue
        else:
            ch = chr(code)
        for row in rows:
            if gflags[row] & GLYPH_NO_BOX:
                raise ValueError(f"char {row} has no box")
        box = tuple(boxes[rows[0]])
        if len(rows) == 2:
            other = boxes[rows[1]]
            box = (
                min(box[0], other[0]),
                min(box[1], other[1]),
                max(box[2], other[2]),
                max(box[3], other[3]),
            )
        chars.append((ch, box))
    return compose_spacing_accents(chars)


def page_lines(page: Page) -> list[dict[str, Any]]:
    """The page's text-layer lines as ``PdfInspection.page_lines`` holds them.

    [] for a page without columns, including a failed page: check
    ``page.error`` and ``DocumentLayer.columns_freed``.
    """
    from bibr.ocr.pdf_inspection import _break_wrapped_lines, _page_line_dicts
    from bibr.ocr.ref_geometry import group_chars_into_lines

    if page.cols is None or page.crop_box is None or page.rotation is None:
        return []
    return _page_line_dicts(
        group_chars_into_lines(_break_wrapped_lines(page_chars(page)), page.index),
        page.index + 1,
        page.crop_box,
        page.rotation,
    )


def span_text(page: Page, span: int) -> str:
    cols = page.cols
    if cols is None:
        return ""
    start, end = cols.span_rec[span].tolist()
    return "".join(cols.record_char(index) for index in range(start, end))


def line_text(page: Page, line: int) -> str:
    cols = page.cols
    if cols is None:
        return ""
    first, end = cols.line_span[line].tolist()
    return "".join(span_text(page, span) for span in range(first, end))


def block_at(
    layer: DocumentLayer, page: int, xy: tuple[float | None, float | None] | None
) -> Block | None:
    """The block a destination at *xy* (PDF points, as ``Link.target_xy``) lands in on 0-based *page*.

    The smallest block whose box holds the point; failing that the first block,
    reading from the top down, whose top edge lies in the band under the
    point (``destinations.BAND_ABOVE`` points above it to ``BAND_BELOW`` below)
    and whose box reaches the point's x: a destination sits a little above
    what it points at. An open x matches any block; an open y, which is a
    whole page or its left margin, lands in none. A block without a box (a
    region the layout gave none) is never the one it lands in.
    """
    target = layer.page(page)
    if target is None or xy is None:
        return None
    x, y = xy
    if y is None:
        return None
    above, below = destinations.BAND_ABOVE, destinations.BAND_BELOW

    def reaches(box: Box, slack: float) -> bool:
        return x is None or box[0] - slack <= x <= box[2] + slack

    boxed = [b for b in target.blocks if b.bbox_pdf is not None]
    holding = [b for b in boxed if b.bbox_pdf[1] <= y <= b.bbox_pdf[3] and reaches(b.bbox_pdf, 0.0)]
    if holding:
        return min(
            holding, key=lambda b: (b.bbox_pdf[2] - b.bbox_pdf[0]) * (b.bbox_pdf[3] - b.bbox_pdf[1])
        )
    under = [
        b for b in boxed if y - below <= b.bbox_pdf[3] <= y + above and reaches(b.bbox_pdf, above)
    ]
    return min(under, key=lambda b: (-b.bbox_pdf[3], b.bbox_pdf[0])) if under else None


class StructIndex:
    """A layer's structure elements by id and by the marked content they hold.

    An mcid names marked content on one page only, so an element is found by
    ``(page, mcid)``: the mcid a text object carries in ``PageColumns.obj_mcid``
    with the object's page. The join runs both ways: :meth:`span_element` from
    text to the element that holds it, :meth:`spans_of` from an element to its
    text. A page without columns (see the module docstring) holds no text to
    join: :meth:`span_element` gives None for it, :meth:`spans_of` no spans of
    it, and the elements and their marked content stay.

    A layer whose columns were freed (``DocumentLayer.columns_freed``) has no
    text on any page, so the three joins (:meth:`span_element`,
    :meth:`spans_of` and :meth:`link_element`) raise :class:`ColumnsFreedError`
    on every call, whenever the index was built and whatever it answered
    before. The tree itself (``by_id``, ``by_mcr``, ``children``,
    :meth:`ancestors`) stays.
    """

    def __init__(self, layer: DocumentLayer) -> None:
        self.layer = layer
        self.by_id = {elem.elem_id: elem for elem in layer.struct}
        self.by_mcr: dict[tuple[int, int], StructElem] = {}
        self.children: dict[str, list[StructElem]] = {}
        self._mcr_spans: dict[tuple[int, int], list[int]] | None = None
        for elem in layer.struct:
            if elem.parent is not None:
                self.children.setdefault(elem.parent, []).append(elem)
            for mcr in elem.mcrs:
                self.by_mcr.setdefault(mcr, elem)

    def ancestors(self, elem: StructElem) -> Iterator[StructElem]:
        """The elements above *elem*, nearest first."""
        while elem.parent is not None:
            elem = self.by_id[elem.parent]
            yield elem

    def _need_columns(self) -> None:
        if self.layer.columns_freed:
            raise ColumnsFreedError(
                "the layer's glyph columns were freed, so no text joins to the structure tree"
            )

    def span_element(self, page: Page, span: int) -> StructElem | None:
        """The element that holds the marked content the span's text is in.

        None for text in no marked content and for text inside an artifact.
        Raises :class:`ColumnsFreedError` on a layer whose columns were freed.
        """
        self._need_columns()
        cols = page.cols
        if cols is None:
            return None
        obj = int(cols.span_obj[span])
        if obj < 0 or cols.obj_artifact[obj] or cols.obj_mcid[obj] < 0:
            return None
        return self.by_mcr.get((page.index, int(cols.obj_mcid[obj])))

    def spans_of(self, elem: StructElem) -> list[str]:
        """The ids of the spans of the text *elem* holds, with that of the elements below it.

        The ids come in page order and, on a page, in span order: the order of the
        page's spans, which is not always the order the text was drawn. Raises
        :class:`ColumnsFreedError` on a layer whose columns were freed, even when
        an earlier call found the spans.
        """
        self._need_columns()
        if self._mcr_spans is None:
            self._mcr_spans = {}
            for page in self.layer.pages:
                cols = page.cols
                if cols is None:
                    continue
                for span in range(len(cols.span_rec)):
                    obj = int(cols.span_obj[span])
                    if obj >= 0 and not cols.obj_artifact[obj] and cols.obj_mcid[obj] >= 0:
                        key = (page.index, int(cols.obj_mcid[obj]))
                        self._mcr_spans.setdefault(key, []).append(span)
        found: set[tuple[int, int]] = set()
        pending = [elem]
        while pending:
            member = pending.pop()
            pending.extend(self.children.get(member.elem_id, ()))
            for mcr in member.mcrs:
                found.update((mcr[0], span) for span in self._mcr_spans.get(mcr, ()))
        return [ids.span(page, span) for page, span in sorted(found)]

    def link_element(self, link: Link) -> StructElem | None:
        """The Link element that wraps the text a link annotation covers, or None.

        pdfium names no annotation for an element's object reference, so the
        pair is found through the text: the first covered span that sits in a
        Link element, or under one. Raises :class:`ColumnsFreedError` on a layer
        whose columns were freed.
        """
        self._need_columns()
        page = self.layer.page(link.page)
        if page is None:
            return None
        for span_id in link.source_span_ids:
            elem = self.span_element(page, int(span_id.rsplit(".sp", 1)[1]))
            for candidate in () if elem is None else (elem, *self.ancestors(elem)):
                if candidate.role == "Link":
                    return candidate
        return None
