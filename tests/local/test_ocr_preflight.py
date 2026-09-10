"""``bibr chew`` refuses to start a run whose OCR runtime cannot start.

Before this preflight a CPU-only Linux box loaded the layout model and then
spent the OCR startup budget (900 s for a hidden vLLM bootstrap) before
"No OCR startup candidate succeeded" — the transactional chain's verdict is
now delivered before any model loads, with the install hints attached.
"""

from types import SimpleNamespace

import pytest


def _cfg(backend, url=None):
    return SimpleNamespace(ocr_backend=backend, ocr_url=url)


@pytest.fixture
def linux(monkeypatch):
    from bibr.ocr import registry

    monkeypatch.setattr(registry.sys, "platform", "linux")
    monkeypatch.setattr(registry.platform, "machine", lambda: "x86_64")
    return registry


def test_cpu_only_linux_without_llama_cpp_fails_fast(monkeypatch, linux):
    from bibr.local.cli import run_config

    monkeypatch.setattr(linux, "_cuda_vram_gb", lambda: None)
    monkeypatch.setattr("bibr.local.llama_cpp.find_llama_server", lambda: None)

    message = run_config._preflight_ocr_runtime(_cfg("paddle"))

    assert message is not None
    assert "glm-llama" in message and "llama.cpp" in message
    assert "--ocr-url" in message and "gemini" in message
    # paddle-vllm never entered the chain on a GPU-less box, so it is not
    # listed as something the user should go and fix.
    assert "paddle-vllm" not in message


def test_llama_cpp_on_path_is_enough(monkeypatch, linux):
    from bibr.local.cli import run_config

    monkeypatch.setattr(linux, "_cuda_vram_gb", lambda: None)
    monkeypatch.setattr(
        "bibr.local.llama_cpp.find_llama_server", lambda: ["/usr/local/bin/llama-server"]
    )

    assert run_config._preflight_ocr_runtime(_cfg("paddle")) is None


def test_roomy_gpu_passes_on_paddle_vllm_even_without_llama_cpp(monkeypatch, linux):
    from bibr.local.cli import run_config

    monkeypatch.setattr(linux, "_cuda_vram_gb", lambda: 24.0)
    monkeypatch.setattr("bibr.local.llama_cpp.find_llama_server", lambda: None)
    # uv on PATH is enough: the launcher bootstraps vLLM into an isolated
    # tool environment (loudly) when the extra is missing.
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr("shutil.which", lambda name, *a, **k: "/usr/bin/uv")

    assert run_config._preflight_ocr_runtime(_cfg("paddle")) is None


def test_explicit_paddle_vllm_without_a_gpu_names_the_blocker(monkeypatch, linux):
    from bibr.local.cli import run_config

    monkeypatch.setattr(linux, "_cuda_vram_gb", lambda: None)

    message = run_config._preflight_ocr_runtime(_cfg("paddle-vllm"))

    assert message is not None
    assert "paddle-vllm" in message and "no NVIDIA GPU" in message


@pytest.mark.parametrize("backend", ["gemini", "openai", "anthropic", "glm-http"])
def test_cloud_and_http_backends_are_not_vetted_here(backend):
    from bibr.local.cli import run_config

    assert run_config._preflight_ocr_runtime(_cfg(backend)) is None


def test_explicit_ocr_url_skips_the_local_check():
    from bibr.local.cli import run_config

    assert run_config._preflight_ocr_runtime(_cfg("paddle-http", url="http://ocr:8080")) is None
