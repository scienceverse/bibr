"""DocxHandlingStage — native `.docx` parse via python-docx."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class DocxHandlingStage:
    name = "docx"
    # FileState fields consumed / populated (see validate_stage_contracts).
    requires = ("pdf_bytes",)
    produces = ("contents", "_native_parser")

    async def run(self, ctx: PipelineContext) -> None:
        alive = ctx.alive()
        docx_files = [fs for fs in alive if fs.path.suffix.lower() == ".docx"]
        if not docx_files:
            return

        ctx.progress.stage_start(self.name)
        from bibr.input.docx_native import DocxParser

        for fs in docx_files:
            fs_t0 = time.monotonic()
            try:
                if fs.pdf_bytes is None:
                    raise RuntimeError("DOCX bytes missing — validate stage did not load file")
                content = fs.pdf_bytes

                def _parse(content=content):
                    parser = DocxParser(content)
                    return parser, parser.parse()

                parser, fs.contents = await asyncio.to_thread(_parse)
                fs._native_parser = parser
                fs.stage_times[self.name] = time.monotonic() - fs_t0
            except Exception as e:  # noqa: BLE001
                fs.set_error(
                    f"Native DOCX parse failed: {e}",
                    code="docx_parse_failed",
                    stage=self.name,
                    exc=e,
                )
                fs.free_all()
                logger.warning("Native DOCX parse failed for %s", fs.path.name, exc_info=True)
        ctx.progress.stage_end(self.name)
