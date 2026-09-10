"""ParseSegmentStage — PDFParser + wtpsplit sentence segmentation."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


def _parse_pdf(ocr_regions, outline=None, *, settings=None, first_page_index=0):
    """Build PDFParser and run parse() — sync, CPU-heavy.

    ``outline`` (PDF bookmarks) is threaded through only when present, so the
    default (outline-off) construction stays ``PDFParser(ocr_regions)``.
    ``first_page_index`` tells the parser which absolute page the slice starts
    at, so front-matter heuristics keep firing under ``--pages``/``start_page``.
    """
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(
        ocr_regions,
        outline=outline,
        settings=settings,
        first_page_index=first_page_index,
    )
    contents = parser.parse()
    return parser, contents


class ParseSegmentStage:
    name = "parse"
    # FileState fields consumed / populated (see validate_stage_contracts).
    requires = ("ocr_regions",)
    produces = ("contents",)

    async def run(self, ctx: PipelineContext) -> None:
        ctx.progress.stage_start(self.name)
        t0 = time.monotonic()
        rm = ctx.resources
        rm.ensure_segmenter()

        async def _process_one(fs):
            try:
                fs_t0 = time.monotonic()
                native_parser = getattr(fs, "_native_parser", None)
                if native_parser is not None:
                    parser = native_parser
                    contents = fs.contents
                else:
                    # Offload sync CPU-heavy parse to a thread so concurrent
                    # files can interleave on the event loop and the thread
                    # pool can fan out across cores. The PDF outline (when
                    # harvested) rides along as an authoritative heading-level
                    # signal; when absent the call stays single-arg so the
                    # default path is unchanged.
                    outline = getattr(fs, "pdf_outline", None)
                    parse_args = (fs.ocr_regions,) if outline is None else (fs.ocr_regions, outline)
                    parser, contents = await asyncio.to_thread(
                        _parse_pdf,
                        *parse_args,
                        settings=ctx.settings,
                        first_page_index=ctx.config.start_page or 0,
                    )
                    fs.contents = contents

                # Hand captured reference-line geometry to the extract stage.
                if contents is not None:
                    contents.ref_line_geometry = getattr(fs, "ref_line_geometry", None)
                    from bibr.ocr.pdf_inspection import inspection_to_dict

                    contents.native_source = inspection_to_dict(getattr(fs, "pdf_inspection", None))

                assembler = parser.assembler
                if len(assembler):
                    texts = assembler.segmentable_texts
                    if texts:
                        # segment_batch serializes access to the shared,
                        # non-thread-safe segmenter internally.
                        all_segments = await rm.segmenter.segment_batch(texts)
                    else:
                        all_segments = []
                    await asyncio.to_thread(parser.apply_segmentation, contents, all_segments)
                await asyncio.to_thread(parser.create_content_sections, contents)

                fs.stage_times["parse_segment"] = time.monotonic() - fs_t0
            except Exception as e:  # noqa: BLE001
                fs.set_error(
                    f"Parse/segment failed: {e}",
                    code="parse_failed",
                    stage=self.name,
                    exc=e,
                )
                logger.warning("Parse failed for %s", fs.path.name, exc_info=True)

        try:
            await asyncio.gather(*(_process_one(fs) for fs in ctx.alive()))
        finally:
            if ctx.config.memory_mode == "aggressive":
                rm.unload_segmenter()

        logger.debug("Parse+segment stage: %.1fs", time.monotonic() - t0)
        ctx.progress.stage_end(self.name)
