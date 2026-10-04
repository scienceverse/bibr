"""The document layer of a pipeline file: rebuilt when needed, then tied to its regions.

NativeTextStage builds the layer inline (``inspect_pdf(include_doc_layer=True)``).
A complete OCR-bundle hit skips that stage and frees the PDF bytes, so
:func:`ensure_document_layer` rebuilds the PDF-native part from the bytes the
run processed, over the page range the layout stage renders, through the same
``open_text_page`` path. Either way it then attaches the post-OCR regions as
blocks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from bibr.document.harvest import RenderBudget, build_document_layer
from bibr.document.model import COLUMN_DTYPES, Block, DocumentLayer, Page
from bibr.document.views import from_layout_bbox

if TYPE_CHECKING:
    from bibr.ocr.types import OcrRegionResult
    from bibr.pipeline.state import FileState

logger = logging.getLogger(__name__)


def render_budget(settings: Any) -> RenderBudget:
    """The layout stage's render settings."""
    from bibr.ocr.image_utils import MIN_REDUCED_RENDER_DPI

    return RenderBudget(
        dpi=settings.layout.dpi,
        max_pixels=settings.layout.max_render_pixels,
        max_dimension=settings.layout.max_render_dimension,
        min_dpi=MIN_REDUCED_RENDER_DPI,
    )


def layout_page_range(
    n_pages: int, *, start_page: int | None, end_page: int | None, max_pages: int
) -> range:
    """The pages LayoutStage renders (its ``max_pages`` cap, then the iterator's clamp)."""
    if max_pages > 0:
        max_end = (start_page or 0) + max_pages - 1
        if end_page is None or end_page > max_end:
            end_page = max_end
    start = start_page if start_page is not None else 0
    end = min(end_page if end_page is not None else n_pages - 1, n_pages - 1)
    if start < 0 or start > end:
        raise ValueError(f"Invalid page range: start={start}, end={end}, total={n_pages}")
    return range(start, end + 1)


def rebuild_document_layer(
    pdf_bytes: bytes,
    settings: Any,
    *,
    start_page: int | None,
    end_page: int | None,
) -> DocumentLayer:
    """The layer ``inspect_pdf`` builds inline, from the PDF bytes alone."""

    def pages(n_pages: int) -> range:
        return layout_page_range(
            n_pages,
            start_page=start_page,
            end_page=end_page,
            max_pages=settings.pipeline.max_pages,
        )

    return build_document_layer(pdf_bytes, pages, budget=render_budget(settings))


def ensure_document_layer(
    fs: FileState,
    settings: Any,
    *,
    start_page: int | None,
    end_page: int | None,
) -> DocumentLayer | None:
    """*fs*'s layer, rebuilt if NativeTextStage did not try to build it, with blocks attached.

    None for inputs that are not PDFs, when the processed bytes are gone (an
    input file that changed since the run read it) and when a build was
    already tried and failed: a rebuild would parse the same bytes again
    under the lock. Never raises.
    """
    try:
        layer = fs.doc_layer
        if layer is None:
            if fs.doc_layer_attempted:
                return None
            fs.doc_layer_attempted = True
            from bibr.extract.pdf_doi_evidence import is_pdf
            from bibr.pipeline.stages.identity import _processed_bytes

            data = _processed_bytes(fs)
            if data is None or not is_pdf(data):
                return None
            layer = rebuild_document_layer(data, settings, start_page=start_page, end_page=end_page)
            fs.doc_layer = layer
        if fs.ocr_regions is not None:
            attach_blocks(layer, fs.ocr_regions)
        return layer
    except Exception:  # noqa: BLE001 - the layer is optional; the paper goes on without it
        logger.warning("Could not build the document layer for %s", fs.path.name, exc_info=True)
        return None


def _chosen_source(region: OcrRegionResult, page: Page) -> str | None:
    if not region.content:
        return None
    if not region.native_text_used:
        return "ocr"
    # Native fill and the OCR stage's fallback both read the text layer; on a
    # scan that layer is the hidden OCR one.
    return "invisible_layer" if page.text_source == "invisible_layer" else "native"


def attach_blocks(layer: DocumentLayer, ocr_regions: list[list[OcrRegionResult]]) -> None:
    """Place the post-OCR regions on the layer's pages as blocks.

    ``ocr_regions`` is indexed by absolute page (the OCR stage pads the pages
    before a ``start_page`` with empty lists). Replaces earlier blocks, and
    assigns every line to the block holding most of its records' centres.
    """
    for page in layer.pages:
        regions = ocr_regions[page.index] if page.index < len(ocr_regions) else []
        blocks: list[Block] = []
        seen: set[str] = set()
        for position, region in enumerate(regions):
            if not region.bbox_2d:
                continue
            block_id = f"p{page.index}.r{region.index}"
            if block_id in seen:
                block_id = f"{block_id}.{position}"
            seen.add(block_id)
            chosen = _chosen_source(region, page)
            blocks.append(
                Block(
                    block_id=block_id,
                    page=page.index,
                    bbox_pdf=from_layout_bbox(page, region.bbox_2d),
                    label=region.label,
                    native_label=region.native_label,
                    read_order=position,
                    text={chosen: region.content} if chosen is not None else {},
                    chosen=chosen,
                    finish_reason=region.finish_reason,
                    region_key=(page.index + 1, region.index),
                )
            )
        page.blocks = blocks
        _assign_lines(page)


def _assign_lines(page: Page) -> None:
    cols = page.cols
    if cols is None:
        return
    n_lines = len(cols.line_span)
    line_block = np.full(n_lines, -1, dtype=np.dtype(COLUMN_DTYPES["line_block"]))
    if page.blocks and n_lines:
        record_line = np.full(len(cols.rec_cp), -1, dtype=np.int64)
        for line, (first, end) in enumerate(cols.line_span.tolist()):
            for span in range(first, end):
                start, stop = cols.span_rec[span].tolist()
                record_line[start:stop] = line
        valid = (record_line >= 0) & ~cols.rec_newline
        counts = np.zeros((n_lines, len(page.blocks)), dtype=np.int64)
        for position, block in enumerate(page.blocks):
            left, bottom, right, top = block.bbox_pdf
            inside = (
                valid
                & (cols.rec_cx >= left)
                & (cols.rec_cx <= right)
                & (cols.rec_cy >= bottom)
                & (cols.rec_cy <= top)
            )
            counts[:, position] = np.bincount(record_line[inside], minlength=n_lines)
        best = counts.argmax(axis=1)
        line_block = np.where(counts.max(axis=1) > 0, best, -1).astype(line_block.dtype)
    cols.line_block = line_block
    for position, block in enumerate(page.blocks):
        block.lines = tuple(np.flatnonzero(line_block == position).tolist())
