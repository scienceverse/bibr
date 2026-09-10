"""HtmlHandlingStage — native `.html`/`.htm`/`.epub` parse."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class HtmlHandlingStage:
    name = "html"
    requires = ("pdf_bytes",)
    produces = ("contents", "_native_parser")

    async def run(self, ctx: PipelineContext) -> None:
        alive = ctx.alive()
        html_files = [fs for fs in alive if fs.path.suffix.lower() in {".html", ".htm", ".epub"}]
        if not html_files:
            return

        ctx.progress.stage_start(self.name)
        from bibr.input.epub_native import EpubParser
        from bibr.input.html_native import HtmlParser

        for fs in html_files:
            fs_t0 = time.monotonic()
            try:
                if fs.pdf_bytes is None:
                    raise RuntimeError("HTML/ePub bytes missing — validate stage did not load file")
                content = fs.pdf_bytes
                suffix = fs.path.suffix.lower()
                artifact = fs.native_validation_artifact

                def _parse(content=content, suffix=suffix, artifact=artifact):
                    if suffix == ".epub":
                        parser = (
                            EpubParser(content, document=artifact)
                            if artifact
                            else EpubParser(content)
                        )
                    else:
                        parser = (
                            HtmlParser(content, parsed_soup=artifact)
                            if artifact
                            else HtmlParser(content)
                        )
                    return parser, parser.parse()

                parser, fs.contents = await asyncio.to_thread(_parse)
                fs.native_validation_artifact = None
                fs._native_parser = parser
                fs.stage_times[self.name] = time.monotonic() - fs_t0
            except Exception as e:  # noqa: BLE001
                fs.set_error(
                    f"Native HTML/ePub parse failed: {e}",
                    code="parse_failed",
                    stage=self.name,
                    exc=e,
                )
                fs.free_all()
                logger.warning("Native HTML/ePub parse failed for %s", fs.path.name, exc_info=True)
        ctx.progress.stage_end(self.name)
