"""A chunk whose managed vLLM LLM server start is sure to be refused skips OCR.

On an emulated 16 GB card (an 8 GiB hog on an RTX 3090) the 4B
student's first start fitted and every later one was refused by
``ManagedLlmVramError`` about a second after PaddleOCR-VL stopped, yet each
of those chunks still started PaddleOCR-VL and OCRed its 8 papers first: 40
papers OCRed for nothing. ``LlmServerPrecheckStage`` reads the card before a
chunk's OCR and fails the chunk with the very error and outcome the start
would give.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shlex
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.exceptions import UpstreamServiceError
from bibr.local import vllm_share
from bibr.local.vllm_share import ManagedLlmVramError
from bibr.ocr.registry import GpuMemory
from bibr.pipeline.resources import ResourceManager

STUDENT_4B = "scienceverse/bibr-extract-qwen35-4b"
W4A16_LOCAL = "/models/bibr-extract-qwen35-4b-W4A16"
CUDA_3090 = 23.56
# The 16 GB card's reading beside the 8 GiB hog before chunk 1's LLM start (fits at
# the 4B's floor); a later chunk's, 1.07 GiB below that card's chunk 2 reading
# (13.57 GiB free, refused then at the 14.25 GiB vLLM 0.31 needed at 128
# sequences), so the 12.75 GiB needed on vLLM 0.32 at 64 does not fit either;
# and a card whose other user has gone (memory came back).
FITS = GpuMemory(total_gb=CUDA_3090, free_gb=15.04, own_gb=0.50)
REFUSED = GpuMemory(total_gb=CUDA_3090, free_gb=12.50, own_gb=1.98)
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


@pytest.fixture(autouse=True)
def port(monkeypatch):
    """What answers ``/v1/models`` on the LLM server's port (9996): nothing, unless a test says so.

    The start's port check and the precheck both probe through
    ``vllm_llm.request_bytes``. ``models`` lists what a listener serves and
    ``key`` the key it requires (None: no key). Nothing listens on any other
    port. The keys of servers this process started are kept per test.
    """
    from bibr.local import http_runtime, vllm_llm

    state = {"models": None, "key": None, "probes": 0}

    def request_bytes(url, *, headers=None, **_kwargs):
        state["probes"] += 1
        if state["models"] is None or url != "http://localhost:9996/v1/models":
            raise http_runtime.LocalHttpError("connection refused")
        if state["key"] and dict(headers or {}).get("Authorization") != f"Bearer {state['key']}":
            return 401, "Unauthorized", b""
        return 200, "OK", json.dumps({"data": [{"id": m} for m in state["models"]]}).encode()

    monkeypatch.setattr(vllm_llm, "request_bytes", request_bytes)
    monkeypatch.setattr(http_runtime, "_OWN_SERVER_KEYS", {})
    return state


@contextlib.contextmanager
def _no_launch():
    """vLLM's launch and readiness wait mocked out; yields the ``Popen`` mock."""
    import bibr.local.vllm_llm as vllm_llm

    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch.object(vllm_llm.subprocess, "Popen") as popen,
        patch.object(vllm_llm.VllmLlmServer, "_wait_until_healthy"),
        patch.object(vllm_llm.VllmLlmServer, "configure_llm_client", lambda self: None),
    ):
        popen.return_value = MagicMock()
        yield popen


def _settings():
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.llm.local_model = STUDENT_4B
    settings.llm.vllm_port = 9996
    return settings


def _local_w4a16_settings(floor: float | None = None):
    """A local copy of the 4B W4A16 build: not in the registry, so no floor unless set."""
    from bibr.local.llm_models import QWEN35_4B_W4A16_SERVER_ARGS

    settings = _settings()
    settings.llm.local_model = W4A16_LOCAL
    settings.llm.vllm_extra_args = shlex.join(QWEN35_4B_W4A16_SERVER_ARGS)
    settings.llm.local_min_vram_gb = floor
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
        assert "12.50 of 23.56 GiB free (bibr itself holds 1.98 GiB)" in str(refusal)
        assert "needs 12.75 GiB" in str(refusal)

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
        # A local W4A16 copy's 7 GiB floor fits where the bf16 4B's 11.0 does not.
        card["reading"] = REFUSED
        assert await _rm(settings=_local_w4a16_settings(7.0)).llm_start_refusal("vllm") is None

    async def test_a_left_over_floor_setting_keeps_the_registry_floor(self, card, monkeypatch):
        # LLM_LOCAL_MIN_VRAM_GB=7 left set after switching back to the bf16 4B.
        monkeypatch.setattr(vllm_share, "_floor_setting_ignored_for", set())
        card["reading"] = REFUSED
        settings = _settings()
        settings.llm.local_min_vram_gb = 7.0
        refusal = await _rm(settings=settings).llm_start_refusal("vllm")
        assert isinstance(refusal, ManagedLlmVramError)
        assert "the model's 11.00 GiB floor" in str(refusal)

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


class TestReusedServer:
    """A start that would reuse a server already on the port is never predicted.

    ``VllmLlmServer`` asks the port first and reuses a listener that serves
    the model before any share is sized, however little memory is free; that
    server is what holds the card. The precheck asks the same check
    (``vllm_llm.reuses_running_server``) through the same probe.
    """

    async def test_another_processs_server_for_the_model_is_reused(self, card, port):
        card["reading"] = REFUSED
        port["models"] = [STUDENT_4B]
        rm = _rm()
        assert await rm.llm_start_refusal("vllm") is None
        assert card["reads"] == 0  # no share is sized for a reused server
        with _no_launch() as popen:
            await rm.start_llm_server("vllm")
        popen.assert_not_called()
        assert rm._llm_server is not None
        assert rm._llm_server._reused is True

    async def test_a_sibling_pipelines_server_is_reused(self, card, port):
        """Two pipelines in one process: the second finds the first's keyed server on the card."""
        first, second = _rm(), _rm()
        with _no_launch() as popen:
            assert await first.llm_start_refusal("vllm") is None
            await first.start_llm_server("vllm")
            assert popen.call_count == 1
            # The first's server holds the card and answers only to its key.
            port["models"], port["key"] = [STUDENT_4B], first._llm_server.api_key
            card["reading"] = REFUSED
            reads = card["reads"]
            assert await second.llm_start_refusal("vllm") is None
            assert card["reads"] == reads
            await second.start_llm_server("vllm")
            assert popen.call_count == 1
        assert second._llm_server._reused is True
        assert second._llm_server.api_key == first._llm_server.api_key
        first._llm_server._process = None  # never signal the mocked process

    @pytest.mark.parametrize(
        ("models", "key"),
        [
            (["org/another-model"], None),  # a leftover server for another model
            ([STUDENT_4B], "another-process-key"),  # another bibr process's keyed server
        ],
    )
    async def test_a_listener_the_start_cannot_use_is_left_to_the_start(
        self, card, port, models, key
    ):
        card["reading"] = REFUSED
        port["models"], port["key"] = models, key
        rm = _rm()
        assert await rm.llm_start_refusal("vllm") is None
        assert card["reads"] == 0
        with _no_launch() as popen, pytest.raises(UpstreamServiceError) as start:
            await rm.start_llm_server("vllm")
        # The start's own error about the port, not a memory refusal.
        assert not isinstance(start.value, ManagedLlmVramError)
        assert "port 9996" in str(start.value)
        popen.assert_not_called()

    async def test_a_free_port_is_probed_then_predicted(self, card, port):
        card["reading"] = REFUSED
        assert isinstance(await _rm().llm_start_refusal("vllm"), ManagedLlmVramError)
        assert port["probes"] == 1
        assert card["reads"] >= 1


async def _start_refusal(rm: ResourceManager) -> str | None:
    """What the real start does on the card as it reads now: its refusal, or None once launched."""
    with _no_launch() as popen:
        try:
            await rm.start_llm_server("vllm")
        except ManagedLlmVramError as exc:
            popen.assert_not_called()
            return str(exc)
    popen.assert_called_once()
    rm._llm_server._process = None  # never signal the mocked process
    return None


class TestAgreesWithTheStart:
    """The precheck refuses exactly when the start would, with the start's own message.

    Covers what the 0.85 fixtures above cannot tell apart: a model without a
    floor, where bibr's room decides; a planned share below the model's
    floor, which caps it; and a server that stays up for the run (an OCR
    server at an OCR URL, cloud OCR), which is refused where the swap would
    be squeezed to the floor.
    """

    @pytest.mark.parametrize(
        ("settings", "rm", "reading", "refused"),
        [
            pytest.param(_settings, {}, FITS, False, id="bf16-fits"),
            pytest.param(_settings, {}, REFUSED, True, id="bf16-refused"),
            # A local path with LLM_LOCAL_MIN_VRAM_GB=7 starts at its floor,
            # and is refused once bibr holds 2.9 GiB beside a 12 GiB hog
            # (8.75 GiB needed), where without the floor it would start at 0.20.
            pytest.param(lambda: _local_w4a16_settings(7.0), {}, REFUSED, False, id="w4a16-floor"),
            pytest.param(
                lambda: _local_w4a16_settings(7.0),
                {},
                GpuMemory(total_gb=CUDA_3090, free_gb=8.66, own_gb=2.90),
                True,
                id="w4a16-floor-refused",
            ),
            # A local path without it has no floor. A start still leaves bibr
            # 6 GiB less what it holds (4.02 GiB here), so it is refused once
            # that and the 0.75 GiB reserve use up what is free; with only the
            # 1 GiB minimum it would start at 0.11.
            pytest.param(
                _local_w4a16_settings,
                {},
                GpuMemory(total_gb=CUDA_3090, free_gb=4.50, own_gb=1.98),
                True,
                id="floorless-swap-refused",
            ),
            pytest.param(
                _local_w4a16_settings,
                {},
                GpuMemory(total_gb=CUDA_3090, free_gb=6.00, own_gb=1.98),
                False,
                id="floorless-swap-fits",
            ),
            # The same readings with a planned share of 0.40, which runs
            # beside PaddleOCR-VL instead of taking turns with it: the server
            # stays up and leaves bibr the same room.
            pytest.param(
                _local_w4a16_settings,
                {"managed_vllm_fraction": 0.40},
                GpuMemory(total_gb=CUDA_3090, free_gb=4.50, own_gb=1.98),
                True,
                id="floorless-no-swap-refused",
            ),
            pytest.param(
                _local_w4a16_settings,
                {"managed_vllm_fraction": 0.40},
                GpuMemory(total_gb=CUDA_3090, free_gb=6.00, own_gb=1.98),
                False,
                id="floorless-no-swap-fits",
            ),
            # A planned share of 0.40 (9.42 GiB) caps the 4B's 11.0 GiB floor:
            # with 9.75 GiB left after the reserve and bibr's room, the start
            # takes 0.40, where the 11.0 GiB floor would refuse it.
            pytest.param(
                _settings,
                {"managed_vllm_fraction": 0.40},
                GpuMemory(total_gb=CUDA_3090, free_gb=15.50, own_gb=1.00),
                False,
                id="cap-below-floor-fits",
            ),
            pytest.param(
                _settings,
                {"managed_vllm_fraction": 0.40},
                GpuMemory(total_gb=CUDA_3090, free_gb=15.00, own_gb=1.00),
                True,
                id="cap-below-floor-refused",
            ),
            # PaddleOCR-VL at an OCR URL holds about 5.4 GiB of the 3090 and
            # a desktop 0.5 GiB. The bf16 4B would stay up beside it and
            # cannot leave bibr 5.5 GiB beside its floor (17.25 GiB needed);
            # the swap squeezes the same reading to 0.47. Without the desktop
            # (17.66 free) it starts.
            pytest.param(
                _settings,
                {"ocr_backend": "paddle-http", "ocr_url": "http://localhost:8897"},
                GpuMemory(total_gb=CUDA_3090, free_gb=17.16, own_gb=0.50),
                True,
                id="ocr-url-refused",
            ),
            pytest.param(
                _settings,
                {"ocr_backend": "gemini"},
                GpuMemory(total_gb=CUDA_3090, free_gb=17.16, own_gb=0.50),
                True,
                id="cloud-ocr-refused",
            ),
            pytest.param(
                _settings,
                {},
                GpuMemory(total_gb=CUDA_3090, free_gb=17.16, own_gb=0.50),
                False,
                id="swap-squeezed",
            ),
            pytest.param(
                _settings,
                {"ocr_backend": "paddle-http", "ocr_url": "http://localhost:8897"},
                GpuMemory(total_gb=CUDA_3090, free_gb=17.66, own_gb=0.50),
                False,
                id="ocr-url-fits",
            ),
        ],
    )
    async def test_refusal_matches_the_real_start(self, card, settings, rm, reading, refused):
        card["reading"] = reading
        rm = _rm(settings=settings(), **rm)
        predicted = await rm.llm_start_refusal("vllm")
        started = await _start_refusal(rm)
        assert (predicted is not None) is refused
        assert (None if predicted is None else str(predicted)) == started

    def test_the_cases_cover_both_paths_and_a_capped_floor(self):
        assert _rm(settings=_local_w4a16_settings())._llm_swaps_with_ocr() is True
        assert (
            _rm(settings=_local_w4a16_settings(), managed_vllm_fraction=0.40)._llm_swaps_with_ocr()
            is False
        )
        assert vllm_share.managed_llm_floor_gb(W4A16_LOCAL, _local_w4a16_settings()) is None
        # 0.40 + PaddleOCR-VL's 0.21 + the 0.25 ONNX reserve fit a 24 GB card:
        # no swap, and the cap is below the 4B's 11.0 GiB floor.
        capped = _rm(managed_vllm_fraction=0.40)
        assert capped._llm_swaps_with_ocr() is False
        floor = vllm_share.managed_llm_floor_gb(STUDENT_4B)
        assert floor is not None
        assert floor > 0.40 * CUDA_3090


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


async def test_16gb_chunks_after_a_refusal_skip_ocr_until_memory_comes_back(card, port):
    """Real ResourceManager, VllmLlmServer and port check (Popen mocked) over four chunks.

    Chunk 1 passes the check, its OCR grows bibr to 1.98 GiB, and the start is
    refused. Chunk 2 is refused before OCR, with the same error text and
    outcome. Chunk 3 finds the memory back: OCR runs and the server starts.
    Chunk 4 goes to a second pipeline in this process while the first's
    server holds the card: its check finds that server on the port, so its
    OCR runs and its start reuses the server instead of being refused.
    """
    from bibr.api import ChewFailure
    from bibr.batch.runner import _local_outcome
    from bibr.config import snapshot_settings
    from bibr.pipeline.context import RunConfig
    from bibr.pipeline.pipeline import Pipeline
    from bibr.pipeline.stages.llm_server import LlmServerPrecheckStage, LlmServerStage
    from bibr.pipeline.state import FileState

    ocred: list[str] = []

    class _RenderOcr:
        name = "render_ocr"
        requires = ()
        produces = ()

        async def run(self, ctx):
            ocred.extend(fs.path.name for fs in ctx.alive())
            if card["reading"] is FITS:
                card["reading"] = REFUSED  # bibr's sessions grew during the chunk

    def pipeline(rm):
        return Pipeline(
            stages=[LlmServerPrecheckStage(), _RenderOcr(), LlmServerStage()],
            resources=rm,
            config=RunConfig(llm_backend="vllm"),
            settings=snapshot_settings(),
        )

    rm, sibling = _rm(), _rm()
    first, second = pipeline(rm), pipeline(sibling)
    chunk1 = [FileState(path=Path(f"c1-{i}.pdf")) for i in range(2)]
    chunk2 = [FileState(path=Path(f"c2-{i}.pdf")) for i in range(2)]
    chunk3 = [FileState(path=Path(f"c3-{i}.pdf")) for i in range(2)]
    chunk4 = [FileState(path=Path(f"c4-{i}.pdf")) for i in range(2)]
    with _no_launch() as popen:
        await first.process_chunk(chunk1)
        await first.process_chunk(chunk2)
        assert popen.call_count == 0  # vLLM was never launched to crash
        card["reading"] = CAME_BACK
        await first.process_chunk(chunk3)
        assert popen.call_count == 1
        # The server now listens on the port, keyed, and holds the card.
        port["models"], port["key"] = [STUDENT_4B], rm._llm_server.api_key
        card["reading"] = REFUSED
        await second.process_chunk(chunk4)
        assert popen.call_count == 1

    assert ocred == ["c1-0.pdf", "c1-1.pdf", "c3-0.pdf", "c3-1.pdf", "c4-0.pdf", "c4-1.pdf"]

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
    assert all(fs.error is None for fs in chunk3 + chunk4)
    assert sibling._llm_server._reused is True
    rm._llm_server._process = None  # never signal the mocked process


async def test_a_server_that_would_stay_up_without_room_skips_ocr(card):
    """PaddleOCR-VL at an OCR URL on a 3090 (about 5.4 GiB) and the bf16 4B managed.

    The first chunk's check reads 17.16 GiB free with bibr at 0.50 and a
    0.5 GiB desktop: the LLM would stay up beside the OCR server and cannot
    leave bibr 5.5 GiB beside its 11.0 GiB floor. The chunk fails with the
    start's typed outcome before any OCR, and vLLM is never launched.
    """
    from bibr.api import ChewFailure
    from bibr.batch.runner import _local_outcome
    from bibr.config import snapshot_settings
    from bibr.pipeline.context import RunConfig
    from bibr.pipeline.pipeline import Pipeline
    from bibr.pipeline.stages.llm_server import LlmServerPrecheckStage, LlmServerStage
    from bibr.pipeline.state import FileState

    ocred: list[str] = []

    class _RenderOcr:
        name = "render_ocr"
        requires = ()
        produces = ()

        async def run(self, ctx):
            ocred.extend(fs.path.name for fs in ctx.alive())

    card["reading"] = GpuMemory(total_gb=CUDA_3090, free_gb=17.16, own_gb=0.50)
    rm = _rm(memory_mode="balanced", ocr_backend="paddle-http", ocr_url="http://localhost:8897")
    pipeline = Pipeline(
        stages=[LlmServerPrecheckStage(), _RenderOcr(), LlmServerStage()],
        resources=rm,
        config=RunConfig(llm_backend="vllm"),
        settings=snapshot_settings(),
    )
    chunk = [FileState(path=Path(f"p{i}.pdf")) for i in range(3)]
    with _no_launch() as popen:
        await pipeline.process_chunk(chunk)
    popen.assert_not_called()
    assert ocred == []
    for fs in chunk:
        failure = ChewFailure(
            fs.path, fs.error, fs.error_code, fs.failed_stage, outage=fs.error_outage
        )
        o = _local_outcome(failure, "t0", "t1", 1.0)
        assert (o.status, o.error_code, o.failed_stage) == (
            "failed",
            "upstream_unavailable",
            "llm_server",
        )
        assert "since the server stays up for the run" in o.error
        assert "needs 17.25 GiB" in o.error


class _GpuModel:
    """One of bibr's own GPU models: unloading it frees ``frees_gb`` on the card."""

    def __init__(self, card, frees_gb: float) -> None:
        self.loaded = True
        self.unloads = 0
        self._card = card
        self._frees_gb = frees_gb

    def unload(self) -> None:
        self.loaded = False
        self.unloads += 1
        _free(self._card, self._frees_gb)


def _free(card, gb: float) -> None:
    r = card["reading"]
    card["reading"] = GpuMemory(
        total_gb=r.total_gb, free_gb=r.free_gb + gb, own_gb=max(0.0, r.own_gb - gb)
    )


@pytest.fixture
def ner(monkeypatch):
    """The NER parser singleton: not loaded unless a test says so; unloads are counted."""
    from bibr.extract import ref_extractor

    state = {"loaded": False, "unloads": 0}

    def unload():
        state["loaded"] = False
        state["unloads"] += 1

    monkeypatch.setattr(ref_extractor, "ner_parser_loaded", lambda: state["loaded"])
    monkeypatch.setattr(ref_extractor, "unload_ner_parser", unload)
    return state


def _released(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "before the managed vLLM LLM server start" in r.message]


class TestReleaseBeforeRefusing:
    """Balanced mode releases bibr's own GPU models (and a resident OCR server)
    before it refuses a managed vLLM LLM start, then reads the card again.

    REFUSED (12.50 GiB free, bibr holding 1.98) is short of the 4B's 12.75;
    releasing 1.5 GiB of bibr's models brings it to 14.00 with 0.48 held,
    which fits at the floor.
    """

    async def test_the_precheck_releases_and_goes_ahead(self, card, ner, caplog, monkeypatch):
        from bibr.extract import ref_extractor

        card["reading"] = REFUSED
        rm = _rm(memory_mode="balanced")
        rm._layout, rm._segmenter = _GpuModel(card, 1.0), _GpuModel(card, 0.25)
        ner["loaded"] = True

        def unload_ner():
            ner["loaded"] = False
            ner["unloads"] += 1
            _free(card, 0.25)

        monkeypatch.setattr(ref_extractor, "unload_ner_parser", unload_ner)
        with caplog.at_level(logging.DEBUG, logger="bibr.pipeline.resources"):
            assert await rm.llm_start_refusal("vllm") is None
        assert (rm._layout.unloads, rm._segmenter.unloads, ner["unloads"]) == (1, 1, 1)
        assert rm.llm_start_releases == 1
        [record] = _released(caplog)
        assert record.levelno == logging.WARNING
        assert "layout, sentence segmenter, NER reference parser" in record.message
        assert "needs 12.75 GiB" in record.message  # the refusal it avoided

    async def test_still_short_after_the_release_is_refused(self, card, ner):
        card["reading"] = REFUSED
        rm = _rm(memory_mode="balanced")
        rm._layout = _GpuModel(card, 0.1)
        refusal = await rm.llm_start_refusal("vllm")
        assert isinstance(refusal, ManagedLlmVramError)
        assert "12.60 of 23.56 GiB free" in str(refusal)  # read again after the release
        assert rm._layout.unloads == 1
        assert rm.llm_start_releases == 1

    async def test_nothing_to_release_refuses_at_once(self, card, ner, caplog):
        card["reading"] = REFUSED
        rm = _rm(memory_mode="balanced")
        with caplog.at_level(logging.DEBUG, logger="bibr.pipeline.resources"):
            assert isinstance(await rm.llm_start_refusal("vllm"), ManagedLlmVramError)
        assert rm.llm_start_releases == 0
        assert not _released(caplog)

    @pytest.mark.parametrize("mode", ["aggressive", "keep_all"])
    async def test_only_balanced_mode_releases(self, card, ner, mode):
        card["reading"] = REFUSED
        rm = _rm(memory_mode=mode)
        rm._layout = _GpuModel(card, 1.5)
        assert isinstance(await rm.llm_start_refusal("vllm"), ManagedLlmVramError)
        assert rm._layout.unloads == 0
        assert rm.llm_start_releases == 0

    async def test_a_start_that_fits_releases_nothing(self, card, ner):
        card["reading"] = FITS
        rm = _rm(memory_mode="balanced")
        rm._layout = _GpuModel(card, 1.0)
        assert await rm.llm_start_refusal("vllm") is None
        assert await _start_refusal(rm) is None
        assert rm._layout.unloads == 0

    async def test_the_start_releases_a_resident_ocr_server_and_starts(self, card, ner):
        """From the second chunk the start, not the precheck, meets the refusal:
        the precheck saw the LLM server up. A resident OCR server is stopped too.

        GLM-OCR's llama.cpp server is its own process: stopping it frees 5.5
        GiB of the card but none of what bibr holds. The LLM then stays up,
        so it needs 12.25 + 1.0 + 4.02 GiB: 17.27 of 18.5 free.
        """
        card["reading"] = GpuMemory(total_gb=CUDA_3090, free_gb=13.0, own_gb=1.98)
        rm = _rm(memory_mode="balanced", ocr_backend="glm-llama")
        rm._layout = _GpuModel(card, 0.0)
        ocr = MagicMock()

        def stop_ocr_server():
            r = card["reading"]
            card["reading"] = GpuMemory(
                total_gb=r.total_gb, free_gb=r.free_gb + 5.5, own_gb=r.own_gb
            )

        ocr.shutdown.side_effect = stop_ocr_server
        rm._ocr, rm.ocr_backend = ocr, "glm-llama"
        assert await _start_refusal(rm) is None
        ocr.shutdown.assert_called_once()
        assert rm.ocr is None
        assert rm._layout.unloads == 1
        assert rm.llm_start_releases == 1

    async def test_the_start_is_refused_when_still_short(self, card, ner):
        card["reading"] = REFUSED
        rm = _rm(memory_mode="balanced")
        rm._layout = _GpuModel(card, 0.1)
        started = await _start_refusal(rm)
        assert started is not None and "12.60 of 23.56 GiB free" in started
        assert rm.llm_start_releases == 1

    async def test_a_reused_server_releases_nothing(self, card, ner, port):
        card["reading"] = REFUSED
        port["models"] = [STUDENT_4B]
        rm = _rm(memory_mode="balanced")
        rm._layout = _GpuModel(card, 1.5)
        assert await rm.llm_start_refusal("vllm") is None
        assert rm._layout.unloads == 0

    async def test_a_start_that_reuses_a_server_releases_nothing(self, card, ner, port):
        """The start asks the port before it reads the card: a server for the
        model already listening is reused, so nothing of bibr's is unloaded."""
        card["reading"] = REFUSED
        port["models"] = [STUDENT_4B]
        rm = _rm(memory_mode="balanced")
        rm._layout = _GpuModel(card, 1.5)
        with _no_launch() as popen:
            await rm.start_llm_server("vllm")
        popen.assert_not_called()
        assert rm._llm_server._reused is True
        assert rm._layout.unloads == 0
        assert rm.llm_start_releases == 0
        assert card["reads"] == 0

    @pytest.mark.parametrize(
        ("ocr_backend", "ocr_url"),
        [("paddle-http", "http://ocr.test:8080"), ("glm-http", None), ("gemini", None)],
    )
    async def test_a_remote_ocr_client_is_never_released(self, card, ner, ocr_backend, ocr_url):
        """Someone else's OCR server holds none of this card: stopping its client frees nothing."""
        card["reading"] = GpuMemory(total_gb=CUDA_3090, free_gb=10.0, own_gb=1.98)
        rm = _rm(memory_mode="balanced", ocr_backend=ocr_backend, ocr_url=ocr_url)
        rm._layout = _GpuModel(card, 0.1)
        ocr = MagicMock()
        rm._ocr = ocr
        assert isinstance(await rm.llm_start_refusal("vllm"), ManagedLlmVramError)
        assert rm._layout.unloads == 1
        assert rm.llm_start_releases == 1
        ocr.shutdown.assert_not_called()
        assert rm.ocr is ocr

    async def test_a_release_is_logged_once_per_chunk(self, card, ner, caplog):
        card["reading"] = REFUSED
        rm = _rm(memory_mode="balanced")
        rm._layout = _GpuModel(card, 1.5)
        with caplog.at_level(logging.DEBUG, logger="bibr.pipeline.resources"):
            assert await rm.llm_start_refusal("vllm") is None  # chunk 1's precheck
            # The layout reloads for the chunk's OCR and bibr grows again.
            rm._layout.loaded = True
            card["reading"] = REFUSED
            assert await _start_refusal(rm) is None  # chunk 1's start
            # Chunk 2's precheck finds the LLM server up and predicts nothing;
            # PaddleOCR-VL then stops it, and its start meets the refusal.
            assert await rm.llm_start_refusal("vllm") is None
            rm.shutdown_llm_server()
            rm._layout.loaded = True
            card["reading"] = REFUSED
            assert await _start_refusal(rm) is None  # chunk 2's start
        assert rm.llm_start_releases == 3
        levels = [r.levelno for r in _released(caplog)]
        assert levels == [logging.WARNING, logging.DEBUG, logging.WARNING]
