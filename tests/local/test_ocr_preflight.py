"""``bibr chew`` refuses to start a run whose OCR runtime cannot start.

Before this preflight a CPU-only Linux box loaded the layout model and then
spent the OCR startup budget (900 s for a hidden vLLM bootstrap) before
"No OCR startup candidate succeeded" — the transactional chain's verdict is
now delivered before any model loads, with the install hints attached.
"""

import importlib.util
import sys
from importlib.machinery import ModuleSpec
from types import SimpleNamespace

import pytest


def _cfg(backend, url=None):
    return SimpleNamespace(ocr_backend=backend, ocr_url=url)


def _pin_install(monkeypatch, *, torch, cv2):
    """Pin whether torch is installed and what ``import cv2`` finds (``None``: not installed)."""
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "torch":
            return ModuleSpec("torch", None) if torch else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setitem(sys.modules, "cv2", cv2)


class _NoPipeline:
    def __init__(self, **_kwargs):
        raise AssertionError("the preflight should have stopped the run")


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


async def test_chew_still_checks_the_ocr_runtime_on_a_core_install(tmp_path, monkeypatch, capsys):
    """A core install needs no opencv for PDFs, but the OCR runtime check
    behind the opencv one still runs — as it does for ``bibr batch``."""
    from bibr.local.cli import _build_parser, _run_process

    _pin_install(monkeypatch, torch=False, cv2=None)
    monkeypatch.setattr("bibr.ocr.registry._cuda_vram_gb", lambda: None)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _NoPipeline)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    args = _build_parser().parse_args(["chew", str(pdf), "--ocr", "paddle-vllm"])

    with pytest.raises(SystemExit) as exc:
        await _run_process(args)

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "OCR backend cannot start here — paddle-vllm: no NVIDIA GPU" in err
    assert "Layout/OCR image runtime unavailable" not in err


@pytest.mark.parametrize(
    ("cv2", "reason", "repair"),
    [
        (None, "opencv (cv2) not installed", "uv sync --extra torch"),
        (
            SimpleNamespace(),
            "cv2 module is incomplete",
            "uv pip install --reinstall opencv-python-headless",
        ),
    ],
    ids=["missing", "broken"],
)
async def test_chew_refuses_pdfs_without_a_working_opencv_when_torch_is_installed(
    tmp_path, monkeypatch, capsys, cv2, reason, repair
):
    from bibr.local.cli import _build_parser, _run_process

    _pin_install(monkeypatch, torch=True, cv2=cv2)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _NoPipeline)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    args = _build_parser().parse_args(["chew", str(pdf), "--ocr-url", "http://ocr.invalid:8080"])

    with pytest.raises(SystemExit) as exc:
        await _run_process(args)

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert f"Layout/OCR image runtime unavailable: {reason}" in err
    assert f"Repair with: {repair}" in err
