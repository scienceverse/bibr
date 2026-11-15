"""The classifiers' CUDA probe asks the runtime that runs them, and an ONNX run never imports torch."""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from bibr.config import GlobalSettings, MlOptions
from bibr.ocr.registry import GpuMemory
from bibr.pipeline import classifier_resources
from bibr.pipeline.classifier_resources import _cuda_memory_info, _peak_vram_bytes

_GIB = 1024**3


@pytest.fixture
def no_torch_loaded(monkeypatch):
    """torch installed but not imported; importing it fails the test."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("an ONNX run imported torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def _fake_torch(*, available: bool, free: int = 0, total: int = 0):
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        mem_get_info=lambda: (free, total),
        max_memory_allocated=lambda: 123,
    )
    return types.SimpleNamespace(cuda=cuda)


def _settings(**ml):
    settings = GlobalSettings()
    for name in ("paper_classifier_model_id", "section_classifier_model_id"):
        setattr(settings.ml, name, MlOptions.model_fields[name].default)
    for name, value in ml.items():
        setattr(settings.ml, name, value)
    return settings


def _onnx_cuda(monkeypatch, available: bool) -> None:
    monkeypatch.setattr("bibr.utils.onnx_providers.cuda_provider_available", lambda: available)


def test_an_onnx_run_reads_the_card_from_nvidia_smi_without_torch(monkeypatch, no_torch_loaded):
    """The gpu extra without torch: the card is there, and the classifiers may use it."""
    _onnx_cuda(monkeypatch, True)
    monkeypatch.setattr(
        "bibr.ocr.registry.managed_gpu_memory",
        lambda: GpuMemory(total_gb=8.0, free_gb=6.5, own_gb=0.0),
    )

    assert _cuda_memory_info(_settings(runtime="auto")) == (True, int(6.5 * _GIB), 8 * _GIB)


def test_an_unreadable_card_is_reported_without_its_memory(monkeypatch, no_torch_loaded):
    _onnx_cuda(monkeypatch, True)
    monkeypatch.setattr("bibr.ocr.registry.managed_gpu_memory", lambda: None)

    assert _cuda_memory_info(_settings(runtime="auto")) == (True, None, None)


@pytest.mark.parametrize("runtime", ["auto", "onnx"])
def test_a_cpu_onnx_run_of_the_shipped_classifiers_does_not_import_torch(
    monkeypatch, no_torch_loaded, runtime
):
    _onnx_cuda(monkeypatch, False)

    assert _cuda_memory_info(_settings(runtime=runtime)) == (False, None, None)


def test_the_torch_runtime_asks_torch(monkeypatch):
    _onnx_cuda(monkeypatch, True)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True, free=5, total=9))

    assert _cuda_memory_info(_settings(runtime="torch")) == (True, 5, 9)


def test_an_overridden_classifier_without_onnx_cuda_asks_torch(monkeypatch):
    """It may be a torch checkpoint, which torch would put on the card."""
    _onnx_cuda(monkeypatch, False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    fake = _fake_torch(available=True, free=3, total=4)
    real_import = builtins.__import__
    monkeypatch.setattr(
        builtins,
        "__import__",
        lambda name, *a, **k: fake if name == "torch" else real_import(name, *a, **k),
    )
    settings = _settings(runtime="auto", section_classifier_model_id="someone/their-classifier")

    assert _cuda_memory_info(settings) == (True, 3, 4)


def test_the_peak_is_not_asked_of_a_torch_that_is_not_loaded(monkeypatch, no_torch_loaded):
    assert _peak_vram_bytes("cuda") is None
    assert _peak_vram_bytes("cpu") is None


def test_the_peak_of_a_loaded_torch_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True))

    assert _peak_vram_bytes("cuda:0") == 123
    assert classifier_resources._peak_vram_bytes("cpu") is None
