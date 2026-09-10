"""LlmServerStage — start local LLM server if requested."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class LlmServerStage:
    name = "llm_server"
    # FileState fields consumed / populated (see validate_stage_contracts).
    requires = ()
    produces = ()

    async def run(self, ctx: PipelineContext) -> None:
        # --no-llm never calls the LLM, so never start a managed local server
        # (cheap guard, safe regardless of how llm_backend was resolved).
        if ctx.config.no_llm:
            return
        t_llm = time.monotonic()
        try:
            await ctx.resources.start_llm_server(backend=ctx.config.llm_backend)
        except Exception as e:  # noqa: BLE001
            for fs in ctx.alive():
                fs.set_error(
                    f"LLM server start failed: {e}",
                    code="llm_server_failed",
                    stage=self.name,
                    exc=e,
                )
            logger.warning("LLM server start failed", exc_info=True)
            return
        if ctx.config.llm_backend == "vllm-mlx":
            logger.debug("LLM server started: %.1fs", time.monotonic() - t_llm)
