"""A CUDA card nothing in the run can load onto does not decide the memory mode."""

from __future__ import annotations

import pytest

import bibr.local.pipeline as lp
from bibr.config import Settings


@pytest.fixture
def small_card(monkeypatch):
    """A 6 GB NVIDIA card beside 64 GB of RAM, and no pinned memory mode."""
    monkeypatch.setattr(Settings.pipeline, "memory_mode", None)
    monkeypatch.setattr("bibr.local.llm_models.detect_hardware", lambda: ("cuda", 6.0))
    monkeypatch.setattr(lp, "_get_system_memory_gb", lambda: 64.0)


def _install(monkeypatch, *, onnx_cuda: bool, torch_cuda: bool) -> None:
    monkeypatch.setattr("bibr.utils.onnx_providers.cuda_provider_available", lambda: onnx_cuda)
    monkeypatch.setattr("bibr.utils.device.torch_cuda_possible", lambda: torch_cuda)


@pytest.mark.parametrize(
    ("ocr_backend", "ocr_url"),
    [("text-layer", None), ("gemini", None), ("paddle-http", "http://ocr.example:8000")],
)
def test_a_cpu_only_install_plans_as_if_the_card_were_absent(
    small_card, monkeypatch, ocr_backend, ocr_url
):
    _install(monkeypatch, onnx_cuda=False, torch_cuda=False)

    mode = lp._default_memory_mode(llm_backend="cloud", ocr_backend=ocr_backend, ocr_url=ocr_url)

    assert mode == "balanced"


def test_low_system_ram_still_runs_aggressive(small_card, monkeypatch):
    _install(monkeypatch, onnx_cuda=False, torch_cuda=False)
    monkeypatch.setattr(lp, "_get_system_memory_gb", lambda: 8.0)

    mode = lp._default_memory_mode(llm_backend="cloud", ocr_backend="text-layer")

    assert mode == "aggressive"


@pytest.mark.parametrize(
    ("onnx_cuda", "torch_cuda", "llm_backend", "ocr_backend"),
    [
        (True, False, "cloud", "text-layer"),  # bibr's ONNX sessions load onto the card
        (False, True, "cloud", "text-layer"),  # a torch model could
        (False, False, "llama-cpp", "text-layer"),  # bibr's LLM server does
        (False, False, "cloud", "glm-llama"),  # the OCR engine does
        (False, False, "cloud", "paddle"),  # the automatic chain may start one
        (False, False, "cloud", None),  # an unknown route
    ],
)
def test_a_card_the_run_can_load_onto_keeps_the_card_rule(
    small_card, monkeypatch, onnx_cuda, torch_cuda, llm_backend, ocr_backend
):
    _install(monkeypatch, onnx_cuda=onnx_cuda, torch_cuda=torch_cuda)

    mode = lp._default_memory_mode(llm_backend=llm_backend, ocr_backend=ocr_backend)

    assert mode == "aggressive"


def test_a_pinned_mode_is_not_second_guessed(small_card, monkeypatch):
    _install(monkeypatch, onnx_cuda=False, torch_cuda=False)
    monkeypatch.setattr(Settings.pipeline, "memory_mode", "aggressive")

    assert lp._default_memory_mode(llm_backend="cloud", ocr_backend="text-layer") == "aggressive"


def test_torch_that_is_installed_but_not_imported_counts_as_able_to_use_the_card(monkeypatch):
    import sys

    from bibr.utils import device

    monkeypatch.setattr(device, "torch", None)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr("bibr.utils.ml_runtime.torch_available", lambda: True)
    assert device.torch_cuda_possible() is True
    assert "torch" not in sys.modules
    monkeypatch.setattr("bibr.utils.ml_runtime.torch_available", lambda: False)
    assert device.torch_cuda_possible() is False


def test_the_cli_plans_a_default_fast_run_with_its_text_layer_route(small_card, monkeypatch):
    """The fast run's OCR route is chosen after the flags are read: the memory
    mode and the chunk size follow the route the run ends up with."""
    from bibr.local.cli import _build_parser, resolve_run_config

    _install(monkeypatch, onnx_cuda=False, torch_cuda=False)
    args = _build_parser().parse_args(["chew", "paper.pdf", "--fast"])

    config = resolve_run_config(args)

    assert config.ocr_backend == "text-layer"
    assert config.memory_mode == "balanced"
    assert config.chunk_size == 8
