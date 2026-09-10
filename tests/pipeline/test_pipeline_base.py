"""Base ``Pipeline`` class — shared orchestration between local and serve."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bibr.config import GlobalSettings
from bibr.pipeline.context import RunConfig
from bibr.pipeline.pipeline import Pipeline
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.state import FileState


class _NoopStage:
    name = "noop"

    async def run(self, ctx):  # noqa: ARG002
        return


class _ErrorStage:
    name = "boom"

    async def run(self, ctx):
        for fs in ctx.file_states:
            fs.set_error("forced", code="forced", stage=self.name)


class _ExportStage:
    name = "export"

    async def run(self, ctx):
        for fs in ctx.file_states:
            fs.result_json = {"ok": True}


def _make_pipeline(stages):
    resources = MagicMock()
    resources.shutdown_llm_server = MagicMock()
    resources.close_llm_client = AsyncMock()
    return Pipeline(
        stages=stages,
        resources=resources,
        config=RunConfig(),
        settings=GlobalSettings(),
    )


@pytest.mark.asyncio
async def test_pipeline_constructor_threads_settings_and_runner_cleans_up():
    class _RecordingParseStage:
        name = "parse"

        def __init__(self):
            self.context = None

        async def run(self, ctx):
            self.context = ctx

    stage = _RecordingParseStage()
    pipeline = _make_pipeline([stage])
    fs = FileState(path=Path("paper.pdf"))
    fs.free_pre_parse = MagicMock()

    await pipeline.process_chunk([fs], progress=NullProgress())

    assert stage.context is not None
    assert stage.context.settings is pipeline.settings
    assert stage.context.scratch["stage_timings"]["parse"] >= 0
    fs.free_pre_parse.assert_called_once_with()


@pytest.mark.asyncio
async def test_pipeline_runner_cleans_up_when_stage_raises():
    class _RaisingParseStage:
        name = "parse"

        async def run(self, ctx):  # noqa: ARG002
            raise RuntimeError("boom")

    pipeline = _make_pipeline([_RaisingParseStage()])
    fs = FileState(path=Path("paper.pdf"))
    fs.free_pre_parse = MagicMock()

    with pytest.raises(RuntimeError, match="boom"):
        await pipeline.process_chunk([fs], progress=NullProgress())

    fs.free_pre_parse.assert_called_once_with()


@pytest.mark.asyncio
async def test_process_chunk_runs_all_stages_in_order():
    calls = []

    class _Recorder:
        def __init__(self, name):
            self.name = name

        async def run(self, ctx):  # noqa: ARG002
            calls.append(self.name)

    pl = _make_pipeline([_Recorder("a"), _Recorder("b"), _Recorder("c")])
    fs = FileState(path=Path("x.pdf"))
    await pl.process_chunk([fs], progress=NullProgress())
    assert calls == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_process_chunk_stops_when_all_errored_before_export():
    pl = _make_pipeline([_ErrorStage(), _NoopStage(), _ExportStage()])
    fs = FileState(path=Path("x.pdf"))
    await pl.process_chunk([fs], progress=NullProgress())
    assert fs.error == "forced"
    assert fs.result_json is None  # export stage never ran


@pytest.mark.asyncio
async def test_process_chunk_does_not_shut_down_llm_server():
    """Per-request teardown is a no-op for long-lived resources.

    Owners call :meth:`Pipeline.aclose` once at end of life instead.
    """
    pl = _make_pipeline([_NoopStage()])
    fs = FileState(path=Path("x.pdf"))
    await pl.process_chunk([fs], progress=NullProgress())
    pl._resources.shutdown_llm_server.assert_not_called()


@pytest.mark.asyncio
async def test_process_chunk_empty_returns_without_error():
    pl = _make_pipeline([_NoopStage()])
    await pl.process_chunk([], progress=NullProgress())
    pl._resources.shutdown_llm_server.assert_not_called()


@pytest.mark.asyncio
async def test_process_file_raises_on_error():
    from bibr.exceptions import ProcessingError

    pl = _make_pipeline([_ErrorStage()])
    with pytest.raises(ProcessingError):
        await pl.process_file(Path("x.pdf"))


class _UpstreamErrorStage:
    name = "ocr"

    async def run(self, ctx):
        from bibr.exceptions import UpstreamServiceError

        for fs in ctx.file_states:
            fs.set_error(
                "ocr down",
                code="ocr_failed",
                stage=self.name,
                exc=UpstreamServiceError("ocr", "server down"),
            )


@pytest.mark.asyncio
async def test_process_file_preserves_upstream_service_error():
    """When the underlying failure was an upstream-service outage (OCR/LLM/
    Crossref), ``process_file`` must raise ``UpstreamServiceError`` (→ HTTP 502),
    not flatten it to ``ProcessingError`` (→ 422)."""
    from bibr.exceptions import UpstreamServiceError

    pl = _make_pipeline([_UpstreamErrorStage()])
    with pytest.raises(UpstreamServiceError):
        await pl.process_file(Path("x.pdf"))


@pytest.mark.asyncio
async def test_process_file_preserves_typed_processing_error_identity():
    from bibr.exceptions import ProcessingError

    error = ProcessingError(
        "LLM returned invalid structured output",
        error_code="llm_invalid_output",
        failed_stage="extract",
    )

    class _TypedErrorStage:
        name = "extract"

        async def run(self, ctx):
            for fs in ctx.file_states:
                fs.set_error(
                    str(error),
                    code=error.error_code,
                    stage=error.failed_stage,
                    exc=error,
                )

    pl = _make_pipeline([_TypedErrorStage()])
    with pytest.raises(ProcessingError) as raised:
        await pl.process_file(Path("x.pdf"))

    assert raised.value is error


@pytest.mark.asyncio
async def test_process_file_returns_result_json_on_success():
    pl = _make_pipeline([_ExportStage()])
    out = await pl.process_file(Path("x.pdf"))
    assert out == {"ok": True}


@pytest.mark.asyncio
async def test_aclose_continues_after_a_teardown_failure():
    """aclose must be best-effort: a raised LLM-server shutdown must NOT skip
    the OCR-engine and LLM-client teardown (otherwise they leak)."""
    pl = _make_pipeline([_NoopStage()])
    pl._resources.shutdown_llm_server = MagicMock(side_effect=RuntimeError("boom"))
    pl._resources.shutdown_ocr = AsyncMock(return_value=None)
    pl._resources.close_llm_client = AsyncMock(return_value=None)

    await pl.aclose()  # must not raise despite the LLM-server shutdown failing

    pl._resources.shutdown_ocr.assert_awaited_once()
    pl._resources.close_llm_client.assert_awaited_once()
