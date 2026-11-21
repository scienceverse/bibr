"""JatsHandlingStage — native JATS-XML parse via lxml."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class JatsHandlingStage:
    name = "jats"
    # FileState fields consumed / populated (see validate_stage_contracts).
    requires = ("pdf_bytes",)
    produces = ("contents", "_native_parser")

    async def run(self, ctx: PipelineContext) -> None:
        alive = ctx.alive()
        xml_files = [fs for fs in alive if fs.path.suffix.lower() == ".xml"]
        if not xml_files:
            return

        ctx.progress.stage_start(self.name)
        from bibr.input.jats_native import JatsParser

        for fs in xml_files:
            fs_t0 = time.monotonic()
            try:
                if fs.pdf_bytes is None:
                    raise RuntimeError("XML bytes missing — validate stage did not load file")
                content = fs.pdf_bytes
                artifact = fs.native_validation_artifact

                def _parse(content=content, artifact=artifact):
                    parser = (
                        JatsParser(content, parsed_root=artifact)
                        if artifact is not None
                        else JatsParser(content)
                    )
                    return parser, parser.parse()

                parser, fs.contents = await asyncio.to_thread(_parse)
                fs.native_validation_artifact = None
                fs._native_parser = parser
                fs.stage_times[self.name] = time.monotonic() - fs_t0
            except Exception as e:  # noqa: BLE001
                fs.set_error(
                    f"Native JATS parse failed: {e}",
                    code="parse_failed",
                    stage=self.name,
                    exc=e,
                )
                fs.free_all()
                logger.warning("Native JATS parse failed for %s", fs.path.name, exc_info=True)
        ctx.progress.stage_end(self.name)
