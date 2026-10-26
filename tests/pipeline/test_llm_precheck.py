"""A chunk whose managed vLLM LLM server start is sure to be refused skips OCR.

On an emulated 16 GB card (E25 V2, an 8 GiB hog on an RTX 3090) the 4B
student's first start fitted and every later one was refused by
``ManagedLlmVramError`` about a second after PaddleOCR-VL stopped, yet each
of those chunks still started PaddleOCR-VL and OCRed its 8 papers first: 40
papers OCRed for nothing. ``LlmServerPrecheckStage`` reads the card before a
chunk's OCR and fails the chunk with the very error and outcome the start
would give.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.local import vllm_share
from bibr.local.vllm_share import ManagedLlmVramError
from bibr.ocr.registry import GpuMemory
from bibr.pipeline.resources import ResourceManager

STUDENT_4B = "scienceverse/bibr-extract-qwen35-4b"
CUDA_3090 = 23.56
# E25 V2 readings beside the 8 GiB hog: before chunk 1's LLM start (fits at
# the 12 GB floor), before chunk 2's (refused: 14.00 GiB needed), and a card
# whose other user has gone (memory came back).
FITS = GpuMemory(total_gb=CUDA_3090, free_gb=15.04, own_gb=0.50)
REFUSED = GpuMemory(total_gb=CUDA_3090, free_gb=13.57, own_gb=1.98)
CAME_BACK = GpuMemory(total_gb=CUDA_3090, free_gb=21.58, own_gb=1.98)


@pytest.fixture
def card(monkeypatch):
    """The card bibr reads, as a mutable one-item list; reads are counted."""
    state = {"reading": FITS, "reads": 0}

    def probe():
        state["reads"] += 1
        return state["reading"]

    monkeypatch.setattr("bibr.ocr.registry.managed_gpu_memory", probe)
    monkeypatch.setattr(vllm_share, "_SETTLE_INTERVAL_S", 0.0)
    return state


def _settings():
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.llm.local_model = STUDENT_4B
    settings.llm.vllm_port = 9996
    return settings


def _rm(**kwargs) -> ResourceManager:
    kwargs.setdefault("ocr_backend", "paddle-vllm")
    kwargs.setdefault("managed_vllm_fraction", 0.85)
    kwargs.setdefault("managed_llm_share_fixed", False)
    kwargs.setdefault("settings", _settings())
    return ResourceManager(**kwargs)


class TestLlmStartRefusal:
    async def test_refused_reading_returns_the_typed_error(self, card):
        card["reading"] = REFUSED
        refusal = await _rm().llm_start_refusal("vllm")
        assert isinstance(refusal, ManagedLlmVramError)
        assert refusal.__traceback__ is not None  # raised, so the log shows where
        assert "13.57 of 23.56 GiB free (bibr itself holds 1.98 GiB)" in str(refusal)
        assert "needs 14.00 GiB" in str(refusal)

    async def test_the_message_is_the_starts_own(self, card):
        card["reading"] = REFUSED
        refusal = await _rm().llm_start_refusal("vllm")
        with pytest.raises(ManagedLlmVramError) as start:
            vllm_share.managed_llm_share(
                REFUSED, _settings(), STUDENT_4B, cap=0.85, swaps_with_ocr=True
            )
        assert str(refusal) == str(start.value)

    @pytest.mark.parametrize("reading", [FITS, CAME_BACK, None])
    async def test_a_start_that_can_succeed_goes_ahead(self, card, reading):
        card["reading"] = reading
        assert await _rm().llm_start_refusal("vllm") is None

    async def test_floor_setting_is_used(self, card):
        # The W4A16 build's 7 GB floor fits where the bf16 4B's 12 GB does not.
        card["reading"] = REFUSED
        settings = _settings()
        settings.llm.local_min_vram_gb = 7.0
        assert await _rm(settings=settings).llm_start_refusal("vllm") is None

    @pytest.mark.parametrize("backend", ["cloud", "llama-cpp", "vllm-mlx", "llmster"])
    async def test_only_the_managed_vllm_server_is_checked(self, card, backend):
        card["reading"] = REFUSED
        assert await _rm().llm_start_refusal(backend) is None
        assert card["reads"] == 0

    async def test_a_user_set_share_is_never_refused(self, card):
        card["reading"] = REFUSED
        assert await _rm(managed_llm_share_fixed=True).llm_start_refusal("vllm") is None
        assert card["reads"] == 0

    async def test_a_running_llm_server_is_not_predicted(self, card):
        card["reading"] = REFUSED
        rm = _rm()
        rm._llm_server = object()
        assert await rm.llm_start_refusal("vllm") is None
        assert card["reads"] == 0

    async def test_an_ocr_preload_in_flight_is_not_predicted(self, card):
        card["reading"] = REFUSED
        rm = _rm()
        rm._ocr_future = object()
        assert await rm.llm_start_refusal("vllm") is None

    @pytest.mark.parametrize(
        ("ocr", "predicted"),
        [
            ({"ocr_backend": "paddle-vllm"}, False),  # stopped before the start
            ({"ocr_backend": "glm-llama"}, False),  # a local engine on the card
            ({"ocr_backend": "gemini"}, True),  # holds no GPU memory
            ({"ocr_backend": "paddle-http", "ocr_url": "http://ocr-host:8080"}, True),
        ],
    )
    async def test_a_resident_ocr_engine_on_the_card_is_not_predicted(self, card, ocr, predicted):
        card["reading"] = REFUSED
        rm = _rm(**ocr)
        rm._ocr = object()
        refusal = await rm.llm_start_refusal("vllm")
        assert isinstance(refusal, ManagedLlmVramError) is predicted


def _ctx(resources, *papers, **config):
    from bibr.pipeline.context import PipelineContext, RunConfig
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.state import FileState

    config.setdefault("llm_backend", "vllm")
    return PipelineContext(
        file_states=[FileState(path=Path(p)) for p in papers],
        progress=NullProgress(),
        resources=resources,
        config=RunConfig(**config),
    )


class TestPrecheckStage:
    async def test_refusal_fails_the_chunk_as_the_start_would(self, caplog):
        from bibr.pipeline.stages.llm_server import LlmServerPrecheckStage, LlmServerStage

        error = ManagedLlmVramError(
            "not enough free GPU memory to start the managed vLLM LLM server"
        )
        skipped = _ctx(MagicMock(llm_start_refusal=AsyncMock(return_value=error)), "a.pdf", "b.pdf")
        started = _ctx(MagicMock(start_llm_server=AsyncMock(side_effect=error)), "a.pdf", "b.pdf")
        with caplog.at_level(logging.WARNING, logger="bibr.pipeline.stages.llm_server"):
            await LlmServerPrecheckStage().run(skipped)
        await LlmServerStage().run(started)
        for got, want in zip(skipped.file_states, started.file_states, strict=True):
            assert got.error == want.error == f"LLM server start failed: {error}"
            assert got.error_code == want.error_code == "llm_server_failed"
            assert got.failed_stage == want.failed_stage == "llm_server"
            assert got.error_outage is want.error_outage is True
            assert got.original_error is error
        messages = [r.getMessage() for r in caplog.records]
        assert any("skipping OCR for this chunk's 2 papers" in m for m in messages)
        assert "LLM server start failed" in messages

    async def test_no_refusal_leaves_the_chunk_alone(self):
        from bibr.pipeline.stages.llm_server import LlmServerPrecheckStage

        rm = MagicMock(llm_start_refusal=AsyncMock(return_value=None))
        ctx = _ctx(rm, "a.pdf")
        await LlmServerPrecheckStage().run(ctx)
        rm.llm_start_refusal.assert_awaited_once_with("vllm")
        assert ctx.alive() == ctx.file_states

    async def test_no_llm_and_dead_chunks_are_not_checked(self):
        from bibr.pipeline.stages.llm_server import LlmServerPrecheckStage

        rm = MagicMock(llm_start_refusal=AsyncMock(return_value=None))
        await LlmServerPrecheckStage().run(_ctx(rm, "a.pdf", no_llm=True))
        dead = _ctx(rm, "a.pdf")
        dead.file_states[0].set_error("earlier failure", code="validate_failed", stage="validate")
        await LlmServerPrecheckStage().run(dead)
        rm.llm_start_refusal.assert_not_awaited()

    @pytest.mark.parametrize("answer", [RuntimeError("nvidia-smi hung"), MagicMock()])
    async def test_a_failed_or_odd_check_goes_ahead(self, answer):
        from bibr.pipeline.stages.llm_server import LlmServerPrecheckStage

        if isinstance(answer, BaseException):
            rm = MagicMock(llm_start_refusal=AsyncMock(side_effect=answer))
        else:
            rm = MagicMock(llm_start_refusal=AsyncMock(return_value=answer))
        ctx = _ctx(rm, "a.pdf")
        await LlmServerPrecheckStage().run(ctx)
        assert ctx.alive() == ctx.file_states


def test_the_local_barrier_plan_checks_before_ocr_only():
    from bibr.pipeline.plans import build_stage_plan

    local = [s.name for s in build_stage_plan(mode="local", stream_backhalf=False, enrichers=[])]
    assert local.index("llm_server_precheck") < local.index("render_ocr")
    assert local.index("render_ocr") < local.index("llm_server")
    for mode, stream in (("local", True), ("serve", False)):
        names = [s.name for s in build_stage_plan(mode=mode, stream_backhalf=stream, enrichers=[])]
        assert "llm_server_precheck" not in names


async def test_e25_v2_chunks_after_a_refusal_skip_ocr_until_memory_comes_back(card, monkeypatch):
    """Real ResourceManager and VllmLlmServer (Popen mocked) over three chunks.

    Chunk 1 passes the check, its OCR grows bibr to 1.98 GiB, and the start is
    refused. Chunk 2 is refused before OCR, with the same error text and
    outcome. Chunk 3 finds the memory back: OCR runs and the server starts.
    """
    import bibr.local.vllm_llm as vllm_llm
    from bibr.api import ChewFailure
    from bibr.batch.runner import _local_outcome
    from bibr.config import snapshot_settings
    from bibr.pipeline.context import RunConfig
    from bibr.pipeline.pipeline import Pipeline
    from bibr.pipeline.stages.llm_server import LlmServerPrecheckStage, LlmServerStage
    from bibr.pipeline.state import FileState

    monkeypatch.setattr(vllm_llm, "guard_managed_server_port", lambda *a, **k: False)
    monkeypatch.setattr(vllm_llm.VllmLlmServer, "configure_llm_client", lambda self: None)
    ocred: list[str] = []

    class _RenderOcr:
        name = "render_ocr"
        requires = ()
        produces = ()

        async def run(self, ctx):
            ocred.extend(fs.path.name for fs in ctx.alive())
            if card["reading"] is FITS:
                card["reading"] = REFUSED  # bibr's sessions grew during the chunk

    rm = _rm()
    pipeline = Pipeline(
        stages=[LlmServerPrecheckStage(), _RenderOcr(), LlmServerStage()],
        resources=rm,
        config=RunConfig(llm_backend="vllm"),
        settings=snapshot_settings(),
    )
    chunk1 = [FileState(path=Path(f"c1-{i}.pdf")) for i in range(2)]
    chunk2 = [FileState(path=Path(f"c2-{i}.pdf")) for i in range(2)]
    chunk3 = [FileState(path=Path(f"c3-{i}.pdf")) for i in range(2)]
    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch.object(vllm_llm.subprocess, "Popen") as popen,
        patch.object(vllm_llm.VllmLlmServer, "_wait_until_healthy"),
    ):
        popen.return_value = MagicMock()
        await pipeline.process_chunk(chunk1)
        await pipeline.process_chunk(chunk2)
        assert popen.call_count == 0  # vLLM was never launched to crash
        card["reading"] = CAME_BACK
        await pipeline.process_chunk(chunk3)
        assert popen.call_count == 1

    assert ocred == ["c1-0.pdf", "c1-1.pdf", "c3-0.pdf", "c3-1.pdf"]

    def outcome(fs):
        failure = ChewFailure(
            fs.path, fs.error, fs.error_code, fs.failed_stage, outage=fs.error_outage
        )
        o = _local_outcome(failure, "t0", "t1", 1.0)
        return (o.status, o.error_code, o.failed_stage, o.error)

    assert {outcome(fs) for fs in chunk1} == {outcome(fs) for fs in chunk2}
    status, code, stage, error = outcome(chunk2[0])
    assert (status, code, stage) == ("failed", "upstream_unavailable", "llm_server")
    assert error.startswith("LLM server start failed: Error in llm: not enough free GPU memory")
    assert all(fs.error is None for fs in chunk3)
    rm._llm_server._process = None  # never signal the mocked process
