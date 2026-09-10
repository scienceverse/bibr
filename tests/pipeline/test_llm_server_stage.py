"""LlmServerStage — starts local vllm-mlx LLM server when requested."""

import importlib
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.llm_server import LlmServerStage
from bibr.pipeline.state import FileState


def _ctx(resources, cfg):
    return PipelineContext(
        file_states=[FileState(path=Path("x.pdf"))],
        progress=NullProgress(),
        resources=resources,
        config=cfg,
    )


@pytest.mark.asyncio
async def test_cloud_backend_is_noop():
    rm = MagicMock(start_llm_server=AsyncMock(), start_classifiers=AsyncMock())
    await LlmServerStage().run(_ctx(rm, RunConfig(llm_backend="cloud")))
    rm.start_llm_server.assert_awaited_once_with(backend="cloud")
    rm.start_classifiers.assert_not_awaited()


@pytest.mark.asyncio
async def test_classifier_stage_starts_even_with_no_llm():
    module_name = "bibr.pipeline.stages.classifiers"
    assert importlib.util.find_spec(module_name) is not None, "classifier stage module is missing"
    classifier_stage = importlib.import_module(module_name)
    rm = MagicMock(start_classifiers=AsyncMock())

    await classifier_stage.ClassifierStage().run(_ctx(rm, RunConfig(no_llm=True)))

    rm.start_classifiers.assert_awaited_once()


@pytest.mark.asyncio
async def test_vllm_mlx_starts_server():
    rm = MagicMock(start_llm_server=AsyncMock())
    await LlmServerStage().run(_ctx(rm, RunConfig(llm_backend="vllm-mlx")))
    rm.start_llm_server.assert_awaited_once_with(backend="vllm-mlx")


@pytest.mark.asyncio
async def test_no_llm_skips_server_start():
    """--no-llm must not start a managed local server even if a backend is set."""
    rm = MagicMock(start_llm_server=AsyncMock())
    await LlmServerStage().run(_ctx(rm, RunConfig(llm_backend="vllm", no_llm=True)))
    rm.start_llm_server.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_failure_marks_files_errored_without_raising():
    """LLM server start failure must fail per-file, not crash the chunk."""
    rm = MagicMock(start_llm_server=AsyncMock(side_effect=RuntimeError("no weights")))
    ctx = _ctx(rm, RunConfig(llm_backend="vllm-mlx"))

    await LlmServerStage().run(ctx)

    fs = ctx.file_states[0]
    assert fs.error is not None
    assert fs.error_code == "llm_server_failed"
    assert fs.failed_stage == "llm_server"
    assert isinstance(fs.original_error, RuntimeError)
