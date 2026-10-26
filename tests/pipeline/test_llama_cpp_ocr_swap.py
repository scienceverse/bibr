"""A managed llama.cpp LLM server stops for a GLM-OCR start that cannot fit beside it.

The LLM server outlives a chunk while OCR is torn down after each one. On an
emulated 8 GB card with a desktop on it (E27 L8), the second chunk's GLM-OCR
llama.cpp server started beside the 4B Q8_0 LLM server (5,088 MiB), could not
allocate its 482 MiB compute buffer, and every page of that chunk failed with
connection errors.
"""

from __future__ import annotations

import pytest

from bibr.local.vllm_share import LLM_ONNX_MIN_HEADROOM_GB
from bibr.ocr.registry import GLM_LLAMA_OCR_VRAM_GB, GpuMemory
from bibr.pipeline.resources import ResourceManager

NEED_GB = GLM_LLAMA_OCR_VRAM_GB + LLM_ONNX_MIN_HEADROOM_GB


def _manager(monkeypatch, free_gb: float | None, *, memory_mode="aggressive", **kwargs):
    reading = None if free_gb is None else GpuMemory(total_gb=7.54, free_gb=free_gb, own_gb=0.3)
    monkeypatch.setattr("bibr.ocr.registry.managed_gpu_memory", lambda: reading)
    rm = ResourceManager(memory_mode=memory_mode, ocr_backend="paddle", **kwargs)
    rm._llm_server_backend = "llama-cpp"
    return rm


def test_glm_ocr_needs_its_hold_and_the_minimum_room():
    assert pytest.approx(3.25) == NEED_GB


def test_llama_cpp_llm_stops_when_glm_ocr_cannot_fit_beside_it(monkeypatch):
    # L8's second chunk: 7.04 GiB to CUDA less the LLM's 4.97 and bibr's 0.27.
    rm = _manager(monkeypatch, free_gb=1.80)
    assert rm._llm_server_blocks_ocr("glm-llama")


@pytest.mark.parametrize("memory_mode", ["aggressive", "balanced"])
def test_llama_cpp_llm_stays_up_when_glm_ocr_fits_beside_it(monkeypatch, memory_mode):
    rm = _manager(monkeypatch, free_gb=NEED_GB + 0.01, memory_mode=memory_mode)
    assert not rm._llm_server_blocks_ocr("glm-llama")


def test_balanced_mode_also_stops_it_when_glm_ocr_cannot_fit(monkeypatch):
    # A 10 GB card in balanced mode, once bibr's own sessions have grown.
    rm = _manager(monkeypatch, free_gb=2.4, memory_mode="balanced")
    assert rm._llm_server_blocks_ocr("glm-llama")


@pytest.mark.parametrize(("memory_mode", "stops"), [("aggressive", True), ("balanced", False)])
def test_without_a_reading_only_aggressive_mode_stops_it(monkeypatch, memory_mode, stops):
    rm = _manager(monkeypatch, free_gb=None, memory_mode=memory_mode)
    assert rm._llm_server_blocks_ocr("glm-llama") is stops


def test_an_ocr_url_is_never_a_reason_to_stop_it(monkeypatch):
    rm = _manager(monkeypatch, free_gb=0.5, ocr_url="http://127.0.0.1:9/v1")
    assert not rm._llm_server_blocks_ocr("glm-llama")


@pytest.mark.parametrize(
    ("llm_backend", "ocr_backend"), [("vllm", "glm-llama"), ("llama-cpp", "glm-http")]
)
def test_other_pairs_are_not_stopped_for_lack_of_memory(monkeypatch, llm_backend, ocr_backend):
    rm = _manager(monkeypatch, free_gb=0.5)
    rm._llm_server_backend = llm_backend
    assert not rm._llm_server_blocks_ocr(ocr_backend)


def test_no_llm_server_running_is_not_stopped(monkeypatch):
    rm = _manager(monkeypatch, free_gb=0.5)
    rm._llm_server_backend = None
    assert not rm._llm_server_blocks_ocr("glm-llama")


async def test_await_ocr_stops_the_llama_cpp_llm_server_first(monkeypatch):
    rm = _manager(monkeypatch, free_gb=1.80)
    events: list[str] = []

    class _Server:
        def shutdown(self):
            events.append("llm stopped")

    class _Client:
        loaded = True

        def wait_for_server(self, *args, **kwargs):
            return True

    class _Candidate:
        backend = "glm-llama"
        model = "ggml-org/GLM-OCR-GGUF:Q8_0"
        profile = "glm"

    rm._llm_server = _Server()
    monkeypatch.setattr(rm, "_resolve_ocr_candidates", lambda: [_Candidate()])

    async def construct(candidate):
        events.append(f"ocr start (llm backend {rm._llm_server_backend})")
        return _Client()

    async def ready(client):
        return None

    monkeypatch.setattr(rm, "_construct_ocr_candidate", construct)
    monkeypatch.setattr(rm, "_await_ocr_client_ready", ready)
    monkeypatch.setattr(rm, "_set_ocr_runtime_identity", lambda candidate: None)

    await rm.await_ocr()

    assert events == ["llm stopped", "ocr start (llm backend None)"]
    assert rm._llm_server is None
