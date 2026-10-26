"""ResourceManager starts the managed vLLM LLM server with a share sized at each start.

Unless the user set it, the configured share is only the cap: the server
works out its share from the VRAM free when it starts (bibr.local.vllm_share).
"""

from __future__ import annotations

import platform

import pytest

from bibr.local.vllm_share import ManagedLlmVramError
from bibr.pipeline.resources import ResourceManager


@pytest.fixture
def launches(monkeypatch):
    """Fake vLLM LLM server recording the keyword arguments of each start."""
    import bibr.local.vllm_llm as vllm_llm

    calls: list[dict] = []

    class FakeVllmLlm:
        def __init__(
            self,
            mem_fraction=None,
            settings=None,
            stop_event=None,
            share_from_free_memory=False,
            swaps_with_ocr=False,
        ):
            calls.append(
                {
                    "mem_fraction": mem_fraction,
                    "share_from_free_memory": share_from_free_memory,
                    "swaps_with_ocr": swaps_with_ocr,
                }
            )

        def configure_llm_client(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(vllm_llm, "VllmLlmServer", FakeVllmLlm)
    return calls


async def test_default_share_is_sized_from_free_memory(launches):
    rm = ResourceManager(ocr_backend="paddle-vllm", managed_vllm_fraction=0.85)
    await rm.start_llm_server(backend="vllm")
    # 0.85 and PaddleOCR-VL's 0.21 cannot share a 24 GB card: they swap.
    assert launches == [
        {"mem_fraction": 0.85, "share_from_free_memory": True, "swaps_with_ocr": True}
    ]


async def test_shares_that_fit_together_are_colocated(launches):
    rm = ResourceManager(ocr_backend="paddle-vllm", managed_vllm_fraction=0.5)
    await rm.start_llm_server(backend="vllm")
    assert launches == [
        {"mem_fraction": 0.5, "share_from_free_memory": True, "swaps_with_ocr": False}
    ]


@pytest.mark.parametrize(
    "ocr",
    [
        {"ocr_backend": "gemini"},
        {"ocr_backend": "glm-llama"},
        {"ocr_backend": "paddle-http", "ocr_url": "http://ocr-host:8080"},
        {"ocr_backend": "paddle-vllm", "ocr_url": "http://ocr-host:8080"},
    ],
)
async def test_llm_only_runs_do_not_leave_room_for_an_ocr_swap(launches, ocr):
    """No managed PaddleOCR-VL server takes turns with the LLM, so no swap headroom.

    The swap headroom was applied whenever 0.85 and PaddleOCR-VL's share did
    not fit together, whatever the OCR, which cut every LLM-only start on a
    24 GB card from 0.85 to 0.70.
    """
    rm = ResourceManager(managed_vllm_fraction=0.85, **ocr)
    await rm.start_llm_server(backend="vllm")
    assert launches == [
        {"mem_fraction": 0.85, "share_from_free_memory": True, "swaps_with_ocr": False}
    ]


@pytest.mark.parametrize(("vram_gb", "swaps"), [(24.0, True), (4.0, False)])
async def test_automatic_paddle_chain_swaps_only_when_it_lists_paddle_vllm(
    monkeypatch, launches, vram_gb, swaps
):
    _pin_linux(monkeypatch, vram_gb=vram_gb)
    rm = ResourceManager(ocr_backend="paddle", managed_vllm_fraction=0.85)
    await rm.start_llm_server(backend="vllm")
    assert launches[0]["swaps_with_ocr"] is swaps


async def test_ocr_that_fell_back_from_paddle_vllm_no_longer_swaps(monkeypatch, launches):
    _pin_linux(monkeypatch)
    rm = ResourceManager(ocr_backend="paddle", managed_vllm_fraction=0.85)
    rm.ocr_backend = "glm-llama"  # what the automatic chain resolved to
    await rm.start_llm_server(backend="vllm")
    assert launches[0]["swaps_with_ocr"] is False


@pytest.mark.parametrize(
    ("ocr_backend", "launched"),
    [("gemini", "0.85"), ("paddle-vllm", "0.7")],
)
async def test_e23_first_start_share_with_and_without_ocr(monkeypatch, ocr_backend, launched):
    """The real server class sized from E23 A's first LLM start (bibr at 504 MiB)."""
    from unittest.mock import MagicMock, patch

    import bibr.local.vllm_llm as vllm_llm
    from bibr.config import GlobalSettings
    from bibr.local import vllm_share
    from bibr.ocr.registry import GpuMemory

    total = 23.56
    reading = GpuMemory(total_gb=total, free_gb=total - 504 / 1024, own_gb=504 / 1024)
    monkeypatch.setattr("bibr.ocr.registry.managed_gpu_memory", lambda: reading)
    monkeypatch.setattr(vllm_share, "_SETTLE_INTERVAL_S", 0.0)
    monkeypatch.setattr(vllm_llm, "guard_managed_server_port", lambda *a, **k: False)
    monkeypatch.setattr(vllm_llm.VllmLlmServer, "configure_llm_client", lambda self: None)
    settings = GlobalSettings()
    settings.llm.local_model = "scienceverse/bibr-extract-qwen35-4b"
    settings.llm.vllm_port = 9997
    rm = ResourceManager(
        ocr_backend=ocr_backend,
        managed_vllm_fraction=0.85,
        managed_llm_share_fixed=False,
        settings=settings,
    )
    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch.object(vllm_llm.subprocess, "Popen") as popen,
        patch.object(vllm_llm.VllmLlmServer, "_wait_until_healthy"),
    ):
        popen.return_value = MagicMock()
        await rm.start_llm_server(backend="vllm")
    cmd = popen.call_args[0][0]
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == launched
    # Never signal the mocked process.
    rm._llm_server._process = None


async def test_user_set_share_is_launched_as_is(launches):
    rm = ResourceManager(
        ocr_backend="paddle-vllm", managed_vllm_fraction=0.6, managed_llm_share_fixed=True
    )
    await rm.start_llm_server(backend="vllm")
    assert launches[0]["share_from_free_memory"] is False
    assert launches[0]["mem_fraction"] == 0.6


async def test_user_set_share_is_read_from_the_settings(monkeypatch, launches):
    from bibr.config import GlobalSettings

    monkeypatch.setenv("LLM_LOCAL_MEM_FRACTION", "0.6")
    settings = GlobalSettings()
    rm = ResourceManager(managed_vllm_fraction=0.6, settings=settings)
    await rm.start_llm_server(backend="vllm")
    assert launches[0]["share_from_free_memory"] is False


async def test_a_start_that_cannot_fit_raises_the_typed_error(monkeypatch):
    import bibr.local.vllm_llm as vllm_llm

    def no_room(**_kwargs):
        raise ManagedLlmVramError("not enough free GPU memory")

    monkeypatch.setattr(vllm_llm, "VllmLlmServer", no_room)
    rm = ResourceManager(managed_vllm_fraction=0.85)
    with pytest.raises(ManagedLlmVramError):
        await rm.start_llm_server(backend="vllm")
    assert rm._llm_server is None


async def test_the_stage_fails_the_chunk_as_an_llm_outage(monkeypatch):
    """LlmServerStage marks the chunk's papers llm_server_failed, as an outage.

    An outage reaches the batch outcomes as ``upstream_unavailable`` at stage
    ``llm_server``.
    """
    from unittest.mock import MagicMock

    from bibr.pipeline.stages.llm_server import LlmServerStage

    paper = MagicMock()
    ctx = MagicMock()
    ctx.config.no_llm = False
    ctx.config.llm_backend = "vllm"
    ctx.alive.return_value = [paper]

    async def no_room(backend):
        raise ManagedLlmVramError("not enough free GPU memory")

    ctx.resources.start_llm_server = no_room
    await LlmServerStage().run(ctx)
    kwargs = paper.set_error.call_args.kwargs
    assert kwargs["code"] == "llm_server_failed"
    assert kwargs["outage"] is True
    assert "not enough free GPU memory" in paper.set_error.call_args.args[0]


def _pin_linux(monkeypatch, vram_gb=24.0):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr("bibr.ocr.registry._cuda_vram_gb", lambda: vram_gb)


def test_local_pipeline_reads_the_users_choice_before_planning(monkeypatch):
    from bibr.config import GlobalSettings
    from bibr.local.pipeline import LocalPipeline

    _pin_linux(monkeypatch)
    pipe = LocalPipeline(
        memory_mode="balanced",
        settings=GlobalSettings(),
        llm_backend="vllm",
        ocr_backend="paddle-vllm",
    )
    assert pipe._resources._llm_share_fixed is False

    monkeypatch.setenv("LLM_LOCAL_MEM_FRACTION", "0.6")
    pipe = LocalPipeline(
        memory_mode="balanced",
        settings=GlobalSettings(),
        llm_backend="vllm",
        ocr_backend="paddle-vllm",
    )
    assert pipe._resources._llm_share_fixed is True
    assert pipe._resources._managed_vllm_fraction == 0.6


def test_llm_only_pipeline_keeps_the_configured_share_as_its_cap(monkeypatch, launches):
    """With OCR that is not managed paddle-vllm there is no swap: 0.85 stays the cap."""
    import asyncio

    from bibr.config import GlobalSettings
    from bibr.local.pipeline import LocalPipeline

    _pin_linux(monkeypatch)
    pipe = LocalPipeline(
        memory_mode="balanced",
        settings=GlobalSettings(),
        llm_backend="vllm",
        ocr_backend="gemini",
    )
    asyncio.run(pipe._resources.start_llm_server(backend="vllm"))
    assert launches == [
        {"mem_fraction": 0.85, "share_from_free_memory": True, "swaps_with_ocr": False}
    ]
