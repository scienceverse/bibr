"""Device selection for the serve-layer LayoutDetector.

The local LayoutDetector (bibr/local/layout.py) has had an MPS branch from
the start; the serve deployment silently fell back to CPU on Apple Silicon.
Both must resolve devices via the shared bibr.utils.device ladder.
"""

from unittest import mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


def _make_detector(device=None):
    import transformers

    # Materialize both lazy Transformers exports before patching either one.
    # Loading the second export can otherwise repopulate the first attribute on
    # transformers._LazyModule and silently replace its module-level mock.
    from bibr.serve.deployments.layout import LayoutDetector

    image_processor_cls = transformers.AutoImageProcessor
    model_cls = transformers.AutoModelForObjectDetection

    with (
        mock.patch.object(image_processor_cls, "from_pretrained") as mock_proc,
        mock.patch.object(model_cls, "from_pretrained") as mock_model,
    ):
        mock_proc.return_value = mock.MagicMock()
        fake_model = mock.MagicMock()
        fake_model.to.return_value = fake_model
        mock_model.return_value = fake_model
        # Settings.layout.torch_compile defaults to False, so no compile path.
        return LayoutDetector(device=device)


def test_explicit_device_respected():
    det = _make_detector(device="cpu")
    assert det._device.type == "cpu"


def test_auto_device_uses_shared_ladder():
    """When device=None the serve detector must use detect_torch_device —
    on Apple Silicon that means MPS, not a silent CPU fallback."""
    with mock.patch("bibr.utils.device.detect_torch_device", return_value="mps") as ladder:
        det = _make_detector(device=None)
    ladder.assert_called_once()
    assert det._device.type == "mps"


def test_cuda_device_enables_cuda_perf_knobs():
    """A serve layout landing on CUDA must turn on the process-wide perf knobs
    (TF32 + cuDNN autotuner) — even when LitServe passes an explicit 'cuda'
    device that bypasses detect_torch_device()."""
    with mock.patch("bibr.utils.device.configure_cuda_perf") as cfg:
        det = _make_detector(device="cuda")
    assert det._device.type == "cuda"
    cfg.assert_called_once()


def test_cpu_device_does_not_enable_cuda_perf_knobs():
    with mock.patch("bibr.utils.device.configure_cuda_perf") as cfg:
        _make_detector(device="cpu")
    cfg.assert_not_called()


# -- warmup ------------------------------------------------------------------
# The first real forward on a freshly (re)started GPU worker pays cuDNN
# autotuning + CUDA allocator growth + GPU clock ramp, inflating the layout
# time on the first paper(s). A warmup forward at setup() moves that cost into
# the healthcheck start_period. It previously ran only under torch.compile,
# leaving the default (eager) GPU path cold.


def test_warmup_runs_in_eager_cuda_mode():
    """Eager CUDA (torch_compile=False, the default) must still warm up."""
    from bibr.serve.deployments.layout import LayoutDetector

    with mock.patch.object(LayoutDetector, "_run_warmup") as warm:
        _make_detector(device="cuda")
    warm.assert_called_once()


def test_warmup_skipped_on_cpu():
    """CPU has no cuDNN/allocator/clock warmup to pay — don't waste startup on it."""
    from bibr.serve.deployments.layout import LayoutDetector

    with mock.patch.object(LayoutDetector, "_run_warmup") as warm:
        _make_detector(device="cpu")
    warm.assert_not_called()


# -- setup() device resolution -------------------------------------------------
# LitServe's accelerator="auto" only returns a GPU backend when torch is already
# imported in the *master* process (litserve Connector._choose_auto_accelerator).
# bibr lazy-imports torch in the worker, so auto always resolves to "cpu" and the
# worker's setup() receives device="cpu" even on a GPU box — pinning the layout
# model to the ~197s-vs-~0.7s slow path. The serve layer must not trust that
# verdict: a CPU device from LitServe means "auto-detect", not "force CPU".


def test_litserve_cpu_verdict_falls_through_to_autodetect():
    """The core fix: device='cpu' from LitServe → None (auto-detect via the
    shared cuda→mps→cpu ladder), NOT a forced CPU placement."""
    from bibr.serve.deployments.pipeline import _resolve_layout_device

    assert _resolve_layout_device("cpu", use_gpu=None) is None


def test_none_and_auto_autodetect():
    from bibr.serve.deployments.pipeline import _resolve_layout_device

    assert _resolve_layout_device(None, use_gpu=None) is None
    assert _resolve_layout_device("auto", use_gpu=None) is None


def test_explicit_gpu_device_is_honored():
    """An explicit non-CPU device (e.g. a multi-GPU LitServe assigning cuda:1)
    is passed straight through."""
    from bibr.serve.deployments.pipeline import _resolve_layout_device

    assert _resolve_layout_device("cuda:1", use_gpu=None) == "cuda:1"
    assert _resolve_layout_device("mps", use_gpu=None) == "mps"


def test_use_gpu_false_forces_cpu():
    """Escape hatch for VRAM-tight, OCR-co-located boxes: LAYOUT_USE_GPU=false
    keeps layout on CPU even when a usable GPU is present."""
    from bibr.serve.deployments.pipeline import _resolve_layout_device

    assert _resolve_layout_device("cuda:0", use_gpu=False) == "cpu"
    assert _resolve_layout_device(None, use_gpu=False) == "cpu"


def test_use_gpu_true_overrides_cpu_verdict():
    """An explicit LAYOUT_USE_GPU=true also rejects LitServe's CPU verdict,
    deferring to auto-detection."""
    from bibr.serve.deployments.pipeline import _resolve_layout_device

    assert _resolve_layout_device("cpu", use_gpu=True) is None
