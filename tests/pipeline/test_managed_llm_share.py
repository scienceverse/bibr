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
            colocated=False,
        ):
            calls.append(
                {
                    "mem_fraction": mem_fraction,
                    "share_from_free_memory": share_from_free_memory,
                    "colocated": colocated,
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
    assert launches == [{"mem_fraction": 0.85, "share_from_free_memory": True, "colocated": False}]


async def test_shares_that_fit_together_are_colocated(launches):
    rm = ResourceManager(ocr_backend="paddle-vllm", managed_vllm_fraction=0.5)
    await rm.start_llm_server(backend="vllm")
    assert launches == [{"mem_fraction": 0.5, "share_from_free_memory": True, "colocated": True}]


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
    """LlmServerStage marks the chunk's papers llm_server_failed (an outage)."""
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
