"""Managed vLLM LLM and PaddleOCR-VL vLLM servers never start beside each other.

Each claims a fixed share of total VRAM (defaults 0.85 and 0.92) and will not
start while less is free. The LLM server outlives a chunk, so a later chunk's
``paddle-vllm`` restart found it still up: explicit ``paddle-vllm`` failed
every file after chunk 1 and the automatic chain silently fell back to GLM.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.pipeline.resources import ResourceManager


def _candidate(backend, model="m", profile="paddle"):
    from bibr.ocr.registry import OcrBackendCandidate

    return OcrBackendCandidate(backend=backend, model=model, profile=profile)


@pytest.fixture
def events(monkeypatch):
    """Fake vLLM LLM server recording its starts and stops into ``events``."""
    import bibr.local.vllm_llm as vllm_llm

    log: list[str] = []

    class FakeVllmLlm:
        def __init__(self, mem_fraction=None, settings=None, stop_event=None):
            log.append("llm-start")

        def configure_llm_client(self):
            pass

        def shutdown(self):
            log.append("llm-stop")

    monkeypatch.setattr(vllm_llm, "VllmLlmServer", FakeVllmLlm)
    return log


def _ocr_factory(log: list[str]):
    def create(candidate, stop_event=None):
        log.append(f"ocr-start:{candidate.backend}")
        client = MagicMock(loaded=True)
        client.wait_for_server = AsyncMock()
        client.shutdown = lambda: log.append("ocr-stop")
        return client

    return create


def _candidates(monkeypatch, *backends):
    monkeypatch.setattr(
        "bibr.ocr.registry.resolve_backend_candidates",
        lambda name, settings: tuple(_candidate(b) for b in backends),
    )


async def test_second_chunk_paddle_vllm_stops_the_vllm_llm_server_first(monkeypatch, events):
    _candidates(monkeypatch, "paddle-vllm")
    rm = ResourceManager(ocr_backend="paddle-vllm", managed_vllm_fraction=0.85)
    with patch.object(rm, "_create_ocr_client_for", side_effect=_ocr_factory(events)):
        await rm.await_ocr()  # chunk 1 OCR
        await rm.shutdown_ocr()  # balanced teardown for the local LLM
        await rm.start_llm_server(backend="vllm")  # LlmServerStage
        await rm.await_ocr()  # chunk 2 OCR

    assert events == [
        "ocr-start:paddle-vllm",
        "ocr-stop",
        "llm-start",
        "llm-stop",
        "ocr-start:paddle-vllm",
    ]
    assert rm._llm_server is None

    # LlmServerStage starts it again once chunk 2's OCR has been torn down.
    await rm.shutdown_ocr()
    await rm.start_llm_server(backend="vllm")
    assert events[-2:] == ["ocr-stop", "llm-start"]
    assert rm._llm_server is not None


async def test_automatic_chain_keeps_paddle_instead_of_falling_back(monkeypatch, events):
    _candidates(monkeypatch, "paddle-vllm", "glm-llama")
    rm = ResourceManager(ocr_backend="paddle", managed_vllm_fraction=0.85)
    await rm.start_llm_server(backend="vllm")
    with patch.object(rm, "_create_ocr_client_for", side_effect=_ocr_factory(events)):
        await rm.await_ocr()

    assert events == ["llm-start", "llm-stop", "ocr-start:paddle-vllm"]
    assert rm.ocr_runtime_identity.backend == "paddle-vllm"
    assert rm.ocr_fallback_reason is None


async def test_shares_that_fit_keep_the_llm_server(monkeypatch, events):
    _candidates(monkeypatch, "paddle-vllm")
    rm = ResourceManager(ocr_backend="paddle-vllm", managed_vllm_fraction=0.05)
    await rm.start_llm_server(backend="vllm")
    with patch.object(rm, "_create_ocr_client_for", side_effect=_ocr_factory(events)):
        await rm.await_ocr()

    assert events == ["llm-start", "ocr-start:paddle-vllm"]
    assert rm._llm_server is not None


async def test_non_vllm_llm_server_is_left_running(monkeypatch, events):
    import bibr.local.llama_cpp as llama_cpp

    class FakeLlama:
        def __init__(self, settings=None, stop_event=None):
            events.append("llama-start")

        def configure_llm_client(self):
            pass

        def shutdown(self):
            events.append("llama-stop")

    monkeypatch.setattr(llama_cpp, "LlamaCppLlmServer", FakeLlama)
    _candidates(monkeypatch, "paddle-vllm")
    rm = ResourceManager(ocr_backend="paddle-vllm")
    await rm.start_llm_server(backend="llama-cpp")
    with patch.object(rm, "_create_ocr_client_for", side_effect=_ocr_factory(events)):
        await rm.await_ocr()

    assert events == ["llama-start", "ocr-start:paddle-vllm"]


async def test_preload_does_not_start_paddle_vllm_beside_the_llm(monkeypatch, events):
    _candidates(monkeypatch, "paddle-vllm")
    rm = ResourceManager(ocr_backend="paddle-vllm", managed_vllm_fraction=0.85)
    await rm.start_llm_server(backend="vllm")
    with patch.object(rm, "_create_ocr_client_for", side_effect=_ocr_factory(events)):
        rm.start_ocr_preload()
        assert rm._ocr_future is None
        await rm.await_ocr()

    assert events == ["llm-start", "llm-stop", "ocr-start:paddle-vllm"]


async def test_resident_paddle_vllm_is_stopped_before_the_vllm_llm_starts(monkeypatch, events):
    """keep_all (or OCR_UNLOAD_BETWEEN_CHUNKS=never) keeps OCR loaded into the
    LLM stage, where vLLM would refuse to start beside it."""
    _candidates(monkeypatch, "paddle-vllm")
    rm = ResourceManager(
        ocr_backend="paddle-vllm", managed_vllm_fraction=0.85, memory_mode="keep_all"
    )
    with patch.object(rm, "_create_ocr_client_for", side_effect=_ocr_factory(events)):
        await rm.await_ocr()
        await rm.start_llm_server(backend="vllm")

    assert events == ["ocr-start:paddle-vllm", "ocr-stop", "llm-start"]
    assert rm.ocr is None
