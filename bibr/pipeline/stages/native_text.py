"""NativeTextStage — pre-fill text regions from embedded PDF text.

Runs between ``LayoutStage`` and ``OcrStage``. When a PDF has a reliable
embedded text layer (typical for born-digital academic PDFs), extracting
text directly via pypdfium2 is faster and more accurate than GLM-OCR.
Regions successfully filled here are marked with ``_native_text_used``
so the OCR stage can skip them.

The native-text FILL is guarded by ``Settings.ocr.native_text_enabled``. On
any failure the partial fills for that file are cleared so the subsequent OCR
stage can run over all regions — matching the "fall back to full OCR"
guarantee. The three read-only harvests this stage also performs (doc-info
metadata, PDF outline, geom reference-line geometry) need only ``pdf_bytes``,
not the fill, so they run regardless of that flag — each behind its own
best-effort guard.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from bibr.ocr.pdf_inspection import inspect_pdf

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


def _first_page_native_text(layout_results) -> str:
    """Join the first page's region contents as title-verification context."""
    if not layout_results:
        return ""
    return "\n".join(str(r.get("content") or "") for r in layout_results[0])


def _log_invisible_layer_errors(name: str, component_errors: dict[str, str]) -> None:
    """Report, once per document, pages whose invisible-layer check failed.

    Such a page keeps its text layer, as before the check existed; without the
    log an old PDFium build lacking the needed APIs disables the check unseen.
    """
    errors = [
        error for key, error in component_errors.items() if key.startswith("invisible_text_layer:")
    ]
    if errors:
        logger.info(
            "Invisible OCR-layer detection failed on %d page(s) of %s; they keep their "
            "text layer (%s)",
            len(errors),
            name,
            errors[0],
        )


class NativeTextStage:
    name = "native_text"
    # FileState fields consumed / populated (see validate_stage_contracts).
    requires = ("pdf_bytes", "page_indices", "layout_results")
    produces = (
        "ref_line_geometry",
        "ref_page_lines",
        "pdf_uri_links",
        "native_metadata",
        "pdf_outline",
        "doc_layer",
    )

    async def run(self, ctx: PipelineContext) -> None:
        ctx.progress.stage_start(self.name)
        t0 = time.monotonic()
        settings = ctx.settings

        # Resolve conditional consumers once, then inspect each document under
        # one PDFium lock/open/page walk into detached Python data.
        from bibr.extract.ref_extractor import _resolve_ref_strategies
        from bibr.ocr.native_text import resolve_eligible_labels

        run_config = getattr(ctx, "config", None)
        effective_seg = _resolve_ref_strategies(
            getattr(run_config, "ref_seg_strategy", None),
            settings=settings,
        )[0]
        eligible_labels = resolve_eligible_labels(bool(settings.ocr.native_text_header_footer))
        # The document layer's arguments are passed only when it is on, so the
        # default call is unchanged.
        layer_kwargs: dict[str, Any] = {}
        if settings.pipeline.document_layer is True:
            from bibr.document.rebuild import render_budget

            layer_kwargs = {"include_doc_layer": True, "render_budget": render_budget(settings)}
        native_skip_total = 0
        invisible_layer_pages = 0
        for fs in ctx.alive():
            if not fs.pdf_bytes or fs.layout_results is None:
                continue
            if layer_kwargs:
                fs.doc_layer_attempted = True
            try:
                inspection = await asyncio.to_thread(
                    inspect_pdf,
                    fs.pdf_bytes,
                    fs.layout_results,
                    page_indices=getattr(fs, "page_indices", None),
                    first_page_text=_first_page_native_text(fs.layout_results),
                    fill_native_text=settings.ocr.native_text_enabled,
                    include_outline=settings.pipeline.outline_headings,
                    include_ref_geometry=effective_seg == "geom",
                    min_chars=settings.ocr.native_text_min_chars,
                    min_printable_ratio=settings.ocr.native_text_min_printable_ratio,
                    eligible_labels=eligible_labels,
                    reject_invisible_text_layer=settings.ocr.native_text_reject_invisible_layer,
                    **layer_kwargs,
                )
            except Exception:  # noqa: BLE001 — complete open failure falls back to OCR
                for page in fs.layout_results or []:
                    for region in page:
                        if region.pop("_native_text_used", False):
                            region["content"] = ""
                logger.warning("PDF inspection failed for %s", fs.path.name, exc_info=True)
                continue
            fs.pdf_inspection = inspection
            fs.layout_results = inspection.layout_results
            fs.native_metadata = inspection.metadata or None
            fs.pdf_outline = inspection.outline or None
            fs.ref_line_geometry = inspection.reference_lines or None
            fs.ref_page_lines = inspection.page_lines or None
            fs.pdf_uri_links = inspection.uri_links or None
            if layer_kwargs:
                fs.doc_layer = inspection.document
            native_skip_total += sum(
                1
                for page in inspection.layout_results
                for region in page
                if region.get("_native_text_used")
            )
            invisible_layer_pages += sum(
                1 for page in inspection.pages if page.invisible_text_layer
            )
            watermark_pages = [page for page in inspection.pages if page.watermarks]
            if watermark_pages:
                logger.info(
                    "%s: removed diagonal watermark text on %d pages before reading the "
                    "text layer: %r",
                    fs.path.name,
                    len(watermark_pages),
                    sorted({text for page in watermark_pages for text in page.watermarks})[:5],
                )
            _log_invisible_layer_errors(fs.path.name, inspection.component_errors)

        if native_skip_total:
            logger.info(
                "Native text bypass: %d eligible text regions skipped OCR", native_skip_total
            )
        if invisible_layer_pages:
            logger.info(
                "Scanned pages with an invisible OCR text layer: %d, read with OCR",
                invisible_layer_pages,
            )

        logger.debug("Native-text stage: %.1fs", time.monotonic() - t0)
        ctx.progress.stage_end(self.name)
