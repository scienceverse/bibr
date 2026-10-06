"""ONNX device requests are honoured."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bibr.utils import onnx_providers

# -- ONNX device selection ---------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_cuda_preload(monkeypatch):
    monkeypatch.setattr(onnx_providers, "_cuda_libraries_preloaded", False)


def _session_providers(available: list[str], device: str | None) -> list:
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = available
    mock_ort.InferenceSession.return_value.get_providers.return_value = ["CPUExecutionProvider"]
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        onnx_providers.create_session("model.onnx", device=device, model_name="layout")
    return mock_ort.InferenceSession.call_args.kwargs["providers"]


def test_cpu_device_keeps_coreml_out_of_the_session():
    """``--device cpu`` asks for the CPU provider; CoreML may compute in FP16."""
    available = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    assert _session_providers(available, "cpu") == ["CPUExecutionProvider"]
    assert _session_providers(available, None) == available
    assert _session_providers(available, "mps") == available


def test_cuda_index_selects_the_cuda_provider_device():
    available = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    name, options = _session_providers(available, "cuda:1")[0]
    assert name == "CUDAExecutionProvider"
    assert options["device_id"] == 1
    _, options = _session_providers(available, "cuda")[0]
    assert "device_id" not in options  # ORT's default GPU


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cuda:2", 2), ("CUDA:0", 0), ("cuda", None), ("cuda:x", None), ("cpu", None), (None, None)],
)
def test_cuda_device_id_for(device, expected):
    assert onnx_providers.cuda_device_id_for(device) == expected
