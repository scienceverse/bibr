"""LlmServerStage — start local LLM server if requested."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


def fail_llm_server_start(ctx: PipelineContext, exc: BaseException) -> None:
    """Fail the chunk's live papers as an LLM outage after a failed server start.

    An outage reaches the batch outcomes as ``upstream_unavailable`` at stage
    ``llm_server``, so the next chunk tries again.
    """
    for fs in ctx.alive():
        fs.set_error(
            f"LLM server start failed: {exc}",
            code="llm_server_failed",
            stage=LlmServerStage.name,
            exc=exc,
            outage=True,
        )
    logger.warning("LLM server start failed", exc_info=exc)


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
            fail_llm_server_start(ctx, e)
            return
        if ctx.config.llm_backend == "vllm-mlx":
            logger.debug("LLM server started: %.1fs", time.monotonic() - t_llm)


class LlmServerPrecheckStage:
    """Fail a chunk before its OCR when its managed LLM server start is sure to be refused.

    The managed vLLM LLM server starts after OCR. Once a start has been
    refused for lack of GPU memory, every later chunk used to start
    PaddleOCR-VL and OCR its papers only to be refused again. The resource
    manager checks the memory free now against the model's need
    (``ResourceManager.llm_start_refusal``); when the start cannot succeed,
    the chunk's papers fail here with the error and outcome that start would
    have given, and OCR never starts. The check runs before every chunk, so a
    chunk goes ahead again once enough memory is free.
    """

    name = "llm_server_precheck"
    requires = ()
    produces = ()

    async def run(self, ctx: PipelineContext) -> None:
        if ctx.config.no_llm or not ctx.alive():
            return
        try:
            refusal = await ctx.resources.llm_start_refusal(ctx.config.llm_backend)
        except Exception:  # noqa: BLE001 - the start itself still reports any real failure
            logger.warning("Checking the managed LLM server start before OCR failed", exc_info=True)
            return
        from bibr.local.vllm_share import ManagedLlmVramError

        if not isinstance(refusal, ManagedLlmVramError):
            return
        logger.warning(
            "The managed vLLM LLM server cannot start with the GPU memory free now: "
            "skipping OCR for this chunk's %d papers",
            len(ctx.alive()),
        )
        fail_llm_server_start(ctx, refusal)
