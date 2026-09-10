"""Tests for bibr.utils.onnx_providers — provider chain construction."""

from unittest.mock import MagicMock, patch


def test_cpu_only_provider_chain():
    """When no GPU EPs are available, returns CPUExecutionProvider only."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        from bibr.utils.onnx_providers import get_ort_providers

        providers = get_ort_providers(enable_cuda=False)

    assert providers == ["CPUExecutionProvider"]


def test_cuda_provider_chain():
    """When CUDA EP is available and enabled, it appears before CPU with options."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        from bibr.utils.onnx_providers import get_ort_providers

        providers = get_ort_providers(enable_cuda=True)

    assert len(providers) == 2
    assert isinstance(providers[0], tuple)
    assert providers[0][0] == "CUDAExecutionProvider"
    cuda_opts = providers[0][1]
    assert cuda_opts["arena_extend_strategy"] == "kSameAsRequested"
    assert cuda_opts["do_copy_in_default_stream"] is True
    assert providers[1] == "CPUExecutionProvider"


def test_cuda_with_gpu_mem_limit():
    """gpu_mem_limit is forwarded into CUDA EP options when set."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        from bibr.utils.onnx_providers import get_ort_providers

        providers = get_ort_providers(enable_cuda=True, gpu_mem_limit=1024 * 1024 * 1024)

    assert providers[0][1]["gpu_mem_limit"] == 1024 * 1024 * 1024


def test_coreml_included_when_available():
    """CoreMLExecutionProvider is included when available (macOS)."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        from bibr.utils.onnx_providers import get_ort_providers

        providers = get_ort_providers(enable_cuda=False)

    assert providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_cuda_provider_available_true():
    """True when onnxruntime exposes CUDAExecutionProvider in this process."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        from bibr.utils.onnx_providers import cuda_provider_available

        assert cuda_provider_available() is True


def test_cuda_provider_available_false():
    """False when only the CPU EP is available."""
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        from bibr.utils.onnx_providers import cuda_provider_available

        assert cuda_provider_available() is False


def test_cuda_provider_available_handles_missing_onnxruntime():
    """A missing/broken onnxruntime must degrade to False, not raise."""
    with patch.dict("sys.modules", {"onnxruntime": None}):
        from bibr.utils.onnx_providers import cuda_provider_available

        assert cuda_provider_available() is False
