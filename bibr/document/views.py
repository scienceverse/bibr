"""Read views over a :class:`DocumentLayer`.

Each view reproduces a string or record bibr's pipeline already computes, from
the layer alone: :func:`block_text` the native fill of a region,
:func:`page_lines` the reference line stream's page lines and
:func:`bbox_pdf_pts` a region's ``_bbox_pdf_pts``. The tests hold them to
byte equality with the pipeline's own values.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bibr.document import ids
from bibr.document.model import (
    GLYPH_EXCLUDED,
    GLYPH_HYPHEN,
    GLYPH_NO_BOX,
    Block,
    Box,
    DocumentLayer,
    Page,
    as_box,
)

TEXT_SOURCES = ("native", "invisible_layer", "ocr")


def to_layout_bbox(page: Page, box: Box) -> Box:
    """*box* (PDF points) in the layout frame: ``bbox_2d``'s 0..1000 image space.

    The frame of the rendered page image, so the page's ``/Rotate`` applies;
    the result is ``(x1, y1, x2, y2)`` with y down.
    """
    from bibr.ocr.native_text import _pdf_points_to_normalized_bbox

    return as_box(_pdf_points_to_normalized_bbox(box, page.crop_box, page.rotation))


def from_layout_bbox(page: Page, bbox_2d) -> Box:
    """A layout ``bbox_2d`` (0..1000 image space) in the page's PDF points."""
    from bibr.ocr.native_text import _normalized_bbox_to_pdf_points

    return as_box(_normalized_bbox_to_pdf_points(list(bbox_2d), page.crop_box, page.rotation))


def find_block(layer: DocumentLayer, block_id: str) -> tuple[Page, Block] | None:
    """The page and block with *block_id*; None when the layer has no such block."""
    try:
        parsed = ids.parse(block_id)
    except ValueError:
        return None
    page = layer.page(parsed.page)
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


def page_records(page: Page) -> list[tuple[str, float, float, bool]]:
    """The page's char records as ``_build_page_char_records`` returned them."""
    cols = page.cols
    if cols is None:
        return []
    return [
        (cols.record_char(index), cx, cy, newline)
        for index, (cx, cy, newline) in enumerate(
            zip(cols.rec_cx.tolist(), cols.rec_cy.tolist(), cols.rec_newline.tolist(), strict=True)
        )
    ]


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
    if block.bbox_pdf is None:
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

    ``get_text_range(i, 1)`` per char: empty for a char pdfium leaves out of
    the page text, for a UTF-16 surrogate half and for a char beyond the BMP
    (its one-unit buffer holds only the high surrogate, and pypdfium2 decodes
    with ``errors="ignore"``), U+FFFE for pdfium's line-end hyphen, otherwise
    the char itself. Raises where that read raises (a missing box).
    """
    from bibr.ocr.native_text import compose_spacing_accents

    cols = page.cols
    if cols is None:
        return []
    chars: list[tuple[str, Box]] = []
    boxes = cols.box.tolist()
    for index, (code, flags) in enumerate(zip(cols.cp.tolist(), cols.gflags.tolist(), strict=True)):
        if flags & GLYPH_EXCLUDED:
            continue
        if code == 0x2 and flags & GLYPH_HYPHEN:
            ch = "￾"
        elif 0xD800 <= code <= 0xDFFF or code > 0xFFFF:
            continue
        else:
            ch = chr(code)
        if flags & GLYPH_NO_BOX:
            raise ValueError(f"char {index} has no box")
        chars.append((ch, tuple(boxes[index])))
    return compose_spacing_accents(chars)


def page_lines(page: Page) -> list[dict[str, Any]]:
    """The page's text-layer lines as ``PdfInspection.page_lines`` holds them."""
    from bibr.ocr.pdf_inspection import _break_wrapped_lines, _page_line_dicts
    from bibr.ocr.ref_geometry import group_chars_into_lines

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
