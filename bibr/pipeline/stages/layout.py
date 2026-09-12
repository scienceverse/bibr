"""LayoutStage — PDF page render + PP-DocLayoutV3 detection.

Also decides whether to kick off OCR preload in a background thread so
the long SGLang engine startup (~30-90s) overlaps with layout detection.
Stores ``any_needs_ocr`` and ``preloading_ocr`` flags on ``ctx.signals``
(``OcrStage`` self-dispatches via ``await_ocr``; the flag is kept for
tests/observability).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from bibr.ocr.image_utils import iter_pdf_pages_with_index as _iter_pdf_pages

if TYPE_CHECKING:
    from PIL import Image

    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class LayoutStage:
    name = "layout"
    # FileState fields consumed / populated (see validate_stage_contracts).
    requires = ("pdf_bytes",)
    produces = ("page_images", "page_indices", "layout_results")

    async def run(self, ctx: PipelineContext) -> None:
        cfg = ctx.config
        rm = ctx.resources
        settings = ctx.settings

        def _needs_ocr(fs) -> bool:
            return fs.contents is None

        any_needs_ocr = any(_needs_ocr(fs) for fs in ctx.alive())
        ctx.signals.any_needs_ocr = any_needs_ocr

        preloading_ocr = False
        can_preload = (
            any_needs_ocr
            and cfg.memory_mode != "aggressive"
            and cfg.ocr_backend not in ("glm-http", "gemini", "openai", "anthropic")
            and not cfg.ocr_url
            and (rm.ocr is None or not rm.ocr.loaded)
            # Don't re-submit the (slow, doomed) engine constructor once a prior
            # window in this chunk already hit a hard OCR-init failure.
            and not ctx.signals.ocr_init_error
        )
        if can_preload:
            try:
                rm.start_ocr_preload()
                preloading_ocr = True
            except Exception:  # preload is optional; OCR owns required startup
                logger.warning("OCR preload failed; deferring startup to OCR", exc_info=True)
        ctx.signals.preloading_ocr = preloading_ocr

        ctx.progress.stage_start(self.name)
        t0 = time.monotonic()
        if any_needs_ocr:
            try:
                rm.ensure_layout()
            except Exception as exc:  # shared failure applies only to PDF files
                for fs in ctx.alive():
                    if _needs_ocr(fs):
                        fs.set_error(
                            f"Layout initialization failed: {exc}",
                            code="layout_failed",
                            stage=self.name,
                            exc=exc,
                        )
                ctx.signals.any_needs_ocr = False
                logger.warning("Layout initialization failed", exc_info=True)
                ctx.progress.stage_end(self.name)
                return
        loop = asyncio.get_running_loop()

        start_page = cfg.start_page
        end_page = cfg.end_page
        if settings.pipeline.max_pages > 0:
            effective_start = start_page or 0
            max_end = effective_start + settings.pipeline.max_pages - 1
            if end_page is None or end_page > max_end:
                end_page = max_end

        async def _prepare(fs) -> list[Image.Image]:
            """Render pages to PIL images (executor-side CPU work)."""
            if fs.pdf_bytes is None:
                raise RuntimeError("PDF bytes missing — validate stage did not load file")
            pdf_bytes_local: bytes = fs.pdf_bytes

            def _render_all() -> list[tuple[int, Image.Image]]:
                return list(
                    _iter_pdf_pages(
                        pdf_bytes_local,
                        settings.layout.dpi,
                        start_page,
                        end_page,
                        settings.layout.max_render_pixels,
                        settings.layout.max_render_dimension,
                    )
                )

            page_tuples = await loop.run_in_executor(None, _render_all)
            fs.page_images = [img for _, img in page_tuples]
            fs.page_indices = [idx for idx, _ in page_tuples]
            return fs.page_images

        def _start_prepare(fs) -> tuple[asyncio.Task, float]:
            return asyncio.create_task(_prepare(fs)), time.monotonic()

        # Pipeline CPU-side render/encode with GPU-side detection: while file
        # N sits in detect_batch, file N+1's pages render in the executor.
        # One file of lookahead bounds the extra page images held in memory.
        todo = [fs for fs in ctx.alive() if _needs_ocr(fs)]
        pending: tuple[asyncio.Task, float] | None = None
        for i, fs in enumerate(todo):
            prep_task, fs_t0 = pending or _start_prepare(fs)
            pending = _start_prepare(todo[i + 1]) if i + 1 < len(todo) else None
            try:
                page_images = await prep_task
                if not page_images:
                    fs.set_error(
                        "Could not render any pages from this PDF",
                        code="layout_failed",
                        stage=self.name,
                    )
                    continue

                fs.layout_results = await rm.layout.detect_batch(page_images)

                fs.stage_times["layout"] = time.monotonic() - fs_t0
            except Exception as e:  # noqa: BLE001
                fs.set_error(
                    f"Layout analysis failed: {e}",
                    code="layout_failed",
                    stage=self.name,
                    exc=e,
                )
                logger.warning("Layout failed for %s", fs.path.name, exc_info=True)

        if cfg.memory_mode == "aggressive":
            rm.unload_layout()

        logger.debug("Layout stage: %.1fs", time.monotonic() - t0)
        ctx.progress.stage_end(self.name)
