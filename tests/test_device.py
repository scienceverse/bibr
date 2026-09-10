"""Tests for the shared device detection utility."""

import logging
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from bibr.utils.device import cuda_incompatibility, detect_torch_device, report_device

# Arch list of a modern CUDA wheel (sm_75+): no Pascal (sm_61) kernels.
_MODERN_ARCHES = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]


def _mock_cuda(mock_torch, *, arch_list, capability, name="NVIDIA GeForce GTX 1060 6GB"):
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_arch_list.return_value = arch_list
    mock_torch.cuda.get_device_capability.return_value = capability
    mock_torch.cuda.get_device_name.return_value = name


def test_detect_torch_device_cpu():
    """Falls back to CPU when no GPU is available."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        assert detect_torch_device() == "cpu"


def test_detect_torch_device_cuda():
    """Prefers CUDA when available."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        assert detect_torch_device() == "cuda"


def test_detect_torch_device_mps():
    """Falls back to MPS when CUDA is unavailable."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        assert detect_torch_device() == "mps"


def test_detect_torch_device_no_mps_attr():
    """Handles torch builds without the mps backend attribute."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        del mock_torch.backends.mps
        assert detect_torch_device() == "cpu"


def test_cuda_configures_memory_fraction():
    """Selecting CUDA must apply the per-process VRAM cap (once)."""
    import bibr.utils.device as mod

    with mock.patch("bibr.utils.device.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mod._cuda_mem_configured = False
        detect_torch_device()
        detect_torch_device()
        mock_torch.cuda.set_per_process_memory_fraction.assert_called_once()


# --- compute-capability gate ---------------------------------------------
# A GPU can be *present* (is_available() == True) while the installed torch
# wheel ships no kernels for its architecture — e.g. a Pascal GTX 1060
# (sm_61) with a sm_75+ wheel. Every kernel launch then dies with CUDA
# error 209 "no kernel image is available". The ladder must skip such GPUs.


def test_unsupported_arch_falls_back_to_cpu():
    """Pascal card + sm_75+ wheel: ladder must not return cuda."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        _mock_cuda(mock_torch, arch_list=_MODERN_ARCHES, capability=(6, 1))
        mock_torch.backends.mps.is_available.return_value = False
        assert detect_torch_device() == "cpu"
        mock_torch.cuda.set_per_process_memory_fraction.assert_not_called()


def test_unsupported_arch_falls_back_to_mps():
    """Incompatible CUDA still tries the next rung, not straight to cpu."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        _mock_cuda(mock_torch, arch_list=_MODERN_ARCHES, capability=(6, 1))
        mock_torch.backends.mps.is_available.return_value = True
        assert detect_torch_device() == "mps"


def test_supported_arch_returns_cuda():
    """Same-major arch in the wheel: cuda is kept (sm_86 device, sm_80 kernel)."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        _mock_cuda(mock_torch, arch_list=["sm_75", "sm_80"], capability=(8, 6))
        assert detect_torch_device() == "cuda"


def test_newer_device_covered_by_ptx():
    """A device newer than every sm_ entry is fine if compute_ PTX can JIT up."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        _mock_cuda(mock_torch, arch_list=["sm_90", "compute_90"], capability=(12, 0))
        assert detect_torch_device() == "cuda"


def test_empty_arch_list_assumed_compatible():
    """Source builds may report no arch list — don't second-guess them."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        _mock_cuda(mock_torch, arch_list=[], capability=(6, 1))
        assert detect_torch_device() == "cuda"


def test_arch_specific_entries_do_not_disable_gate():
    """Entries like 'sm_90a' (Hopper arch-specific) must be parsed, not abort the probe."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        _mock_cuda(mock_torch, arch_list=["sm_90", "sm_90a", "compute_90"], capability=(6, 1))
        mock_torch.backends.mps.is_available.return_value = False
        assert detect_torch_device() == "cpu"


def test_probe_failure_is_not_fatal():
    """A failing capability probe must not knock out an otherwise-working GPU."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_arch_list.side_effect = RuntimeError("boom")
        assert detect_torch_device() == "cuda"


def test_cuda_incompatibility_reason_names_card_and_archs():
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        _mock_cuda(mock_torch, arch_list=_MODERN_ARCHES, capability=(6, 1))
        reason = cuda_incompatibility()
        assert reason is not None
        assert "NVIDIA GeForce GTX 1060 6GB" in reason
        assert "sm_61" in reason
        assert "sm_75" in reason


def test_ptx_only_wheel_reason_still_lists_archs():
    """PTX-only wheels (no sm_ cubins) must not produce an empty arch list in the message."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        _mock_cuda(mock_torch, arch_list=["compute_90"], capability=(6, 1))
        reason = cuda_incompatibility()
        assert reason is not None
        assert "compute_90" in reason


def test_cuda_incompatibility_none_without_cuda():
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        assert cuda_incompatibility() is None


def test_cuda_incompatibility_none_when_supported():
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        _mock_cuda(mock_torch, arch_list=["sm_75", "sm_80"], capability=(8, 0))
        assert cuda_incompatibility() is None


# --- report_device: the "running on CPU while a GPU sits idle" warning -------
# A GPU-capable model silently defaulting to CPU on a GPU box is the exact
# failure that cost ~197s/paper and was invisible in INFO logs. report_device
# escalates that one case to WARNING; everything else stays INFO.


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def test_report_device_warns_cpu_with_usable_gpu(caplog):
    """GPU-capable component on CPU + a usable CUDA GPU present → WARNING."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        _mock_cuda(mock_torch, arch_list=["sm_80", "sm_86"], capability=(8, 6))
        with caplog.at_level(logging.INFO, logger="bibr.utils.device"):
            report_device("SentenceSegmenter", "cpu", gpu_capable=True)
    warns = _warnings(caplog)
    assert warns, "expected a WARNING when a GPU-capable model runs on CPU with a GPU present"
    assert "CPU" in warns[0].message


def test_report_device_no_warn_on_gpu(caplog):
    """A component already on the GPU logs INFO, never WARNING."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        _mock_cuda(mock_torch, arch_list=["sm_80", "sm_86"], capability=(8, 6))
        with caplog.at_level(logging.INFO, logger="bibr.utils.device"):
            report_device("LayoutDetector", "cuda", gpu_capable=True)
    assert not _warnings(caplog)


def test_report_device_no_warn_when_not_gpu_capable(caplog):
    """CPU-only components (no GPU implementation) must not nag, GPU or not."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        _mock_cuda(mock_torch, arch_list=["sm_80", "sm_86"], capability=(8, 6))
        with caplog.at_level(logging.INFO, logger="bibr.utils.device"):
            report_device("SectionClassifier", "cpu", gpu_capable=False)
    assert not _warnings(caplog)


def test_report_device_no_warn_without_gpu(caplog):
    """No GPU at all → CPU is the only option, so no warning."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        with caplog.at_level(logging.INFO, logger="bibr.utils.device"):
            report_device("SentenceSegmenter", "cpu", gpu_capable=True)
    assert not _warnings(caplog)


def test_report_device_no_warn_when_gpu_unusable(caplog):
    """A present-but-incompatible GPU (kernels would crash) must NOT warn —
    CPU is genuinely the right place to be."""
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        _mock_cuda(mock_torch, arch_list=_MODERN_ARCHES, capability=(6, 1))  # Pascal + modern wheel
        with caplog.at_level(logging.INFO, logger="bibr.utils.device"):
            report_device("SentenceSegmenter", "cpu", gpu_capable=True)
    assert not _warnings(caplog)


# --- configure_cuda_perf: TF32 + cuDNN autotuner, applied once, env-gated ----
# Committing to a CUDA device should enable two free, low-risk knobs: TF32
# matmul (within fp32 error bounds on Ampere+) and the cuDNN autotuner (ideal
# for the layout model's fixed padded batch shape). Both are global and only
# need to be set once per process.


def test_configure_cuda_perf_enables_tf32_and_benchmark():
    import bibr.utils.device as mod

    with mock.patch("bibr.utils.device.torch") as mock_torch:
        mod._cuda_perf_configured = False
        mod.configure_cuda_perf()
    assert mock_torch.backends.cuda.matmul.allow_tf32 is True
    assert mock_torch.backends.cudnn.allow_tf32 is True
    assert mock_torch.backends.cudnn.benchmark is True


def test_configure_cuda_perf_is_idempotent():
    """A second call must not re-touch the flags (so a later override sticks)."""
    import bibr.utils.device as mod

    with mock.patch("bibr.utils.device.torch") as mock_torch:
        mod._cuda_perf_configured = False
        mod.configure_cuda_perf()
        mock_torch.backends.cudnn.benchmark = "untouched"
        mod.configure_cuda_perf()
    assert mock_torch.backends.cudnn.benchmark == "untouched"


def test_configure_cuda_perf_respects_tf32_disable(monkeypatch):
    import bibr.utils.device as mod

    monkeypatch.setenv("BIBR_ALLOW_TF32", "0")
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        mod._cuda_perf_configured = False
        mock_torch.backends.cuda.matmul.allow_tf32 = "untouched"
        mod.configure_cuda_perf()
    assert mock_torch.backends.cuda.matmul.allow_tf32 == "untouched"
    assert mock_torch.backends.cudnn.benchmark is True  # benchmark still enabled


def test_configure_cuda_perf_respects_benchmark_disable(monkeypatch):
    import bibr.utils.device as mod

    monkeypatch.setenv("BIBR_CUDNN_BENCHMARK", "0")
    with mock.patch("bibr.utils.device.torch") as mock_torch:
        mod._cuda_perf_configured = False
        mock_torch.backends.cudnn.benchmark = "untouched"
        mod.configure_cuda_perf()
    assert mock_torch.backends.cudnn.benchmark == "untouched"
    assert mock_torch.backends.cuda.matmul.allow_tf32 is True  # TF32 still enabled


def test_detect_cuda_configures_perf_knobs():
    """Selecting CUDA must enable the perf knobs."""
    with (
        mock.patch("bibr.utils.device.torch") as mock_torch,
        mock.patch("bibr.utils.device.configure_cuda_perf") as cfg,
    ):
        mock_torch.cuda.is_available.return_value = True
        assert detect_torch_device() == "cuda"
        cfg.assert_called_once()


def test_detect_cpu_does_not_configure_perf_knobs():
    with (
        mock.patch("bibr.utils.device.torch") as mock_torch,
        mock.patch("bibr.utils.device.configure_cuda_perf") as cfg,
    ):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        assert detect_torch_device() == "cpu"
        cfg.assert_not_called()
