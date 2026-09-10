"""Device selection for the serve-layer SentenceSegmenter.

The serve segmenter used to hardcode CPU (``use_gpu=False``). On a large
paper that cost ~197s of CPU-bound ONNX inference vs ~0.7s on GPU — the
dominant contributor to the 300s pipeline timeout. It must now mirror the
local segmenter: auto-detect CUDA when ``use_gpu`` is ``None``, while still
honoring an explicit ``True``/``False`` override.
"""

from unittest import mock

import pytest

pytest.importorskip("wtpsplit_lite")
pytest.importorskip("torch")


def _capture_enable_cuda(use_gpu, *, cuda_available):
    """Build a serve SentenceSegmenter and capture the enable_cuda decision
    handed to get_ort_providers, without loading the real ONNX model."""
    from bibr.serve.deployments import segmenter as seg_mod

    captured = {}

    def fake_get_ort_providers(*, enable_cuda, model_name="", gpu_mem_limit=None):
        captured["enable_cuda"] = enable_cuda
        return ["CPUExecutionProvider"]

    with (
        mock.patch("wtpsplit_lite.SaT", return_value=mock.MagicMock()),
        mock.patch(
            "bibr.utils.onnx_providers.get_ort_providers",
            side_effect=fake_get_ort_providers,
        ),
        mock.patch(
            "bibr.utils.onnx_providers.cuda_provider_available",
            return_value=cuda_available,
        ),
    ):
        seg_mod.SentenceSegmenter(use_gpu=use_gpu)
    return captured["enable_cuda"]


def test_auto_detects_gpu_when_available():
    """use_gpu=None + CUDA EP present → enable CUDA."""
    assert _capture_enable_cuda(None, cuda_available=True) is True


def test_auto_detects_cpu_when_no_gpu():
    """use_gpu=None + no CUDA EP → CPU."""
    assert _capture_enable_cuda(None, cuda_available=False) is False


def test_explicit_false_forces_cpu_even_with_gpu():
    """An explicit use_gpu=False keeps the segmenter on CPU (escape hatch for
    VRAM-constrained, co-located deployments)."""
    assert _capture_enable_cuda(False, cuda_available=True) is False


def test_explicit_true_forces_gpu():
    """An explicit use_gpu=True enables CUDA regardless of auto-detection."""
    assert _capture_enable_cuda(True, cuda_available=False) is True


def test_explicit_threshold_reaches_serve_warmup(monkeypatch):
    from bibr.config import Settings
    from bibr.serve.deployments import segmenter as seg_mod

    model = mock.MagicMock()
    monkeypatch.setattr(Settings, "WTPSPLIT_THRESHOLD", None)
    with (
        mock.patch("wtpsplit_lite.SaT", return_value=model),
        mock.patch(
            "bibr.utils.onnx_providers.get_ort_providers",
            return_value=["CPUExecutionProvider"],
        ),
        mock.patch("bibr.utils.device.report_device"),
    ):
        seg_mod.SentenceSegmenter(use_gpu=False, threshold=0.33)

    model.split.assert_called_once_with(
        ["This is a warmup sentence. It has two parts."], threshold=0.33
    )
