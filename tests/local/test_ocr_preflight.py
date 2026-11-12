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


def _cfg(backend, url=None, llm_backend="cloud", no_llm=False):
    return SimpleNamespace(ocr_backend=backend, ocr_url=url, llm_backend=llm_backend, no_llm=no_llm)


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


@pytest.mark.parametrize(
    ("llm_backend", "no_llm", "passed"),
    [
        ("llama-cpp", False, "llama-cpp"),
        ("vllm", False, "vllm"),
        ("llama-cpp", True, "cloud"),
    ],
)
def test_the_preflight_resolves_the_chain_with_the_runs_llm_backend(
    monkeypatch, llm_backend, no_llm, passed
):
    """Beside a llama.cpp LLM the chain is GLM-OCR alone, so the preflight
    checks llama.cpp, not paddle-vllm."""
    from bibr.local.cli import run_config

    seen = []
    monkeypatch.setattr(
        run_config, "_ocr_runtime_blocker", lambda backend, **kw: seen.append(kw) or None
    )
    assert (
        run_config._preflight_ocr_runtime(_cfg("paddle", llm_backend=llm_backend, no_llm=no_llm))
        is None
    )
    assert seen == [{"llm_backend": passed}]


@pytest.mark.parametrize(
    ("llm_backend", "blocked"),
    [("llama-cpp", True), ("vllm", False), ("cloud", False)],
)
def test_the_blocker_resolves_its_own_chain_with_the_llm_backend(
    monkeypatch, linux, llm_backend, blocked
):
    """On a 24 GB card without llama.cpp, paddle-vllm could start, but beside a
    llama.cpp LLM the chain is GLM-OCR alone, which cannot: refused."""
    from bibr.local.cli import run_config

    monkeypatch.setattr(linux, "_cuda_vram_gb", lambda: 24.0)
    monkeypatch.setattr("bibr.local.llama_cpp.find_llama_server", lambda: None)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr("shutil.which", lambda name, *a, **k: "/usr/bin/uv")

    blocker = run_config._ocr_runtime_blocker("paddle", llm_backend=llm_backend)

    if blocked:
        assert blocker is not None
        assert "glm-llama" in blocker and "paddle-vllm" not in blocker
    else:
        assert blocker is None


@pytest.mark.parametrize(
    ("llm_backend", "no_llm", "chain"),
    [
        ("llama-cpp", False, ["glm-llama"]),
        ("vllm", False, ["paddle-vllm", "glm-llama"]),
        ("llama-cpp", True, ["paddle-vllm", "glm-llama"]),
    ],
)
def test_the_dry_run_lists_the_chain_the_run_starts(monkeypatch, linux, llm_backend, no_llm, chain):
    from bibr.local.cli import dry_run

    monkeypatch.setattr(linux, "_cuda_vram_gb", lambda: 24.0)

    candidates = dry_run._dry_run_ocr_candidates(
        _cfg("paddle", llm_backend=llm_backend, no_llm=no_llm)
    )

    assert [c.backend for c in candidates] == chain


# --- a fast run reads the text layer unless the user chose an OCR route -------------------


def _fast_cfg(backend="paddle", *, llm_backend="vllm"):
    return SimpleNamespace(
        ocr_backend=backend,
        ocr_url=None,
        llm_backend=llm_backend,
        no_llm=False,
        mode="fast",
    )


@pytest.fixture
def gpu_without_vllm(monkeypatch, linux):
    monkeypatch.setattr(linux, "_cuda_vram_gb", lambda: 24.0)
    monkeypatch.setattr(linux, "_nvidia_compute_capability", lambda: 8.9)
    monkeypatch.setattr("shutil.which", lambda name, *a, **k: "/usr/bin/uv")
    return linux


def test_a_fast_run_with_no_route_chosen_needs_no_ocr_runtime(monkeypatch, gpu_without_vllm):
    """The text layer starts nothing, so a GPU without llama.cpp or vLLM is no obstacle."""
    from bibr.local.cli import run_config

    monkeypatch.setattr("bibr.local.llama_cpp.find_llama_server", lambda: None)

    assert run_config._preflight_ocr_runtime(_fast_cfg("text-layer")) is None


def test_a_fast_run_that_chose_paddle_is_not_stopped_while_vllm_can_be_started(
    monkeypatch, gpu_without_vllm
):
    from bibr.local.cli import run_config

    monkeypatch.setattr("bibr.local.llama_cpp.find_llama_server", lambda: None)

    assert run_config._preflight_ocr_runtime(_fast_cfg()) is None


def test_a_fast_run_that_chose_paddle_with_no_launcher_at_all_is_refused(
    monkeypatch, gpu_without_vllm
):
    import importlib.util

    from bibr.local.cli import run_config

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "vllm" else real_find_spec(name, *a, **k),
    )
    monkeypatch.setattr("bibr.local.llama_cpp.find_llama_server", lambda: None)
    monkeypatch.setattr("shutil.which", lambda name, *a, **k: None)

    message = run_config._preflight_ocr_runtime(_fast_cfg())

    assert message is not None
    assert "glm-llama" in message and "paddle-vllm" in message


def test_an_explicit_paddle_vllm_is_judged_as_it_always_was(monkeypatch, gpu_without_vllm):
    from bibr.local.cli import run_config

    monkeypatch.setattr(gpu_without_vllm, "_cuda_vram_gb", lambda: None)

    message = run_config._preflight_ocr_runtime(_fast_cfg("paddle-vllm"))

    assert message is not None
    assert "no NVIDIA GPU" in message


def test_a_fast_run_is_vetted_against_the_cloud_chain_whatever_llm_is_configured():
    from bibr.local.cli.run_config import _run_llm_backend

    assert _run_llm_backend(_fast_cfg(llm_backend="vllm")) == "cloud"
    assert _run_llm_backend(_cfg("paddle", llm_backend="vllm")) == "vllm"
    assert _run_llm_backend(_cfg("paddle", llm_backend="vllm", no_llm=True)) == "cloud"
