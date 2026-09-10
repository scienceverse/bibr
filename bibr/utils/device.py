"""Shared device detection for model inference.

Single source for the CUDA → MPS → CPU ladder. The module itself imports
without torch (a core install reaches ``report_device`` through the sentence
segmenter); only the functions that genuinely need torch import it, and they
raise the 'torch'-extra ImportError when it is absent.
"""

import logging
import os
import sys
import warnings

from bibr.utils.ml_extra import ml_import_error

logger = logging.getLogger(__name__)

# Bound lazily by ``_torch_module()``; kept as a module attribute so tests can
# patch ``bibr.utils.device.torch`` exactly as they did when it was a plain
# import.
torch = None

# Cap bibr's PyTorch VRAM usage so concurrent inference cannot grow unbounded
# and crash the NVIDIA driver.  On a shared single GPU, SGLang's
# --mem-fraction-static is set to 0.60 (in entrypoint.sh); this caps the
# bibr chew to 35% by default, leaving a 5% buffer.
# Override with BIBR_CUDA_MEM_FRACTION env var.
_CUDA_MEM_FRACTION = float(os.environ.get("BIBR_CUDA_MEM_FRACTION", "0.35"))
_cuda_mem_configured = False
_cuda_perf_configured = False


def _torch_module(*, required: bool = True):
    """Return torch, importing it on first use.

    With ``required=False`` only an *already imported* torch is returned: the
    ONNX runtime path must never pull torch into the process just to log a
    device line, and a core install has nothing to import anyway.
    """
    global torch
    if torch is not None:
        return torch
    loaded = sys.modules.get("torch")
    if loaded is None:
        if not required:
            return None
        try:
            import torch as _torch
        except ImportError as e:
            raise ml_import_error("Torch device detection (bibr.utils.device)") from e
        loaded = _torch
    torch = loaded
    return torch


def configure_cuda_perf() -> None:
    """Enable TF32 matmul + the cuDNN autotuner for CUDA inference, once.

    Two free, low-risk knobs that PyTorch leaves off by default:

    - **TF32 matmul** (``cuda.matmul.allow_tf32`` / ``cudnn.allow_tf32``):
      a ~speedup for matmul/conv on Ampere+ (the 3090), within fp32 error
      bounds — object detection is robust to it.
    - **cuDNN autotuner** (``cudnn.benchmark``): picks the fastest conv
      algorithm per input shape. The layout model pads to a fixed batch
      shape, so the tuner runs once at warmup and then every forward hits the
      fast path — no per-call re-tuning.

    Both are process-global, so this only needs to run once after committing
    to a CUDA device; it is idempotent. Each knob is gated by an env var
    (``BIBR_ALLOW_TF32`` / ``BIBR_CUDNN_BENCHMARK``, both default on, set to
    ``0`` to disable for bitwise-reproducibility needs).
    """
    global _cuda_perf_configured
    if _cuda_perf_configured:
        return
    _cuda_perf_configured = True
    torch = _torch_module()
    try:
        if os.environ.get("BIBR_ALLOW_TF32", "1") != "0":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if os.environ.get("BIBR_CUDNN_BENCHMARK", "1") != "0":
            torch.backends.cudnn.benchmark = True
        logger.info("CUDA perf knobs enabled (TF32 matmul, cuDNN autotuner)")
    except Exception as e:  # noqa: BLE001 — perf tuning must never be fatal
        logger.warning("Failed to configure CUDA perf knobs: %s", e)


def _configure_cuda_memory() -> None:
    """Apply the VRAM cap once on first CUDA use."""
    global _cuda_mem_configured
    if _cuda_mem_configured:
        return
    _cuda_mem_configured = True
    torch = _torch_module()
    try:
        torch.cuda.set_per_process_memory_fraction(_CUDA_MEM_FRACTION)
        logger.info(
            "CUDA memory fraction capped at %.0f%% (%.1f GB on this device)",
            _CUDA_MEM_FRACTION * 100,
            torch.cuda.get_device_properties(0).total_memory / 1e9 * _CUDA_MEM_FRACTION,
        )
    except Exception as e:
        logger.warning("Failed to set CUDA memory fraction: %s", e)


def cuda_incompatibility() -> str | None:
    """Reason the present CUDA device cannot run this torch build, or ``None``.

    ``torch.cuda.is_available()`` only proves a device *exists* — it says
    nothing about whether the wheel ships kernels for its architecture. A
    Pascal GTX 1060 (sm_61) with a sm_75+ wheel passes ``is_available()`` and
    then dies on the first kernel launch with CUDA error 209 ("no kernel
    image is available"). This probe compares the device's compute capability
    against the build's arch list, mirroring torch's own ``_check_cubins``
    heuristic: cubins are forward-compatible within a major version, and
    ``compute_XY`` PTX entries can JIT up to any newer architecture.

    Returns ``None`` when CUDA is absent, torch is not loaded, compatible, or
    the probe itself fails (never knock out a GPU on a probe error).
    """
    torch = _torch_module(required=False)
    if torch is None or not torch.cuda.is_available():
        return None

    def _majors(arch_list: list[str], prefix: str) -> set[int]:
        # Tolerate arch-specific suffixes like ``sm_90a`` — strip non-digits
        # per entry rather than letting one odd entry abort the whole probe.
        majors = set()
        for a in arch_list:
            if not a.startswith(prefix):
                continue
            digits = "".join(ch for ch in a.removeprefix(prefix) if ch.isdigit())
            if digits:
                majors.add(int(digits) // 10)
        return majors

    try:
        arch_list = torch.cuda.get_arch_list()
        sm_majors = _majors(arch_list, "sm_")
        ptx_majors = _majors(arch_list, "compute_")
        if not sm_majors and not ptx_majors:
            return None  # source build / unusual wheel — assume it knows best
        # get_device_capability initialises CUDA, which flushes torch's queued
        # arch warnings as raw UserWarning spam. Suppress them — this probe's
        # whole purpose is to report the condition cleanly ourselves.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            major, minor = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
        if major in sm_majors or any(p <= major for p in ptx_majors):
            return None
        # PTX-only wheels have no sm_ entries — fall back to the full list
        # so the message never claims support for "" (nothing).
        supported = ", ".join(a for a in arch_list if a.startswith("sm_")) or ", ".join(arch_list)
        return (
            f"{name} is sm_{major}{minor}, but this PyTorch build only ships "
            f"kernels for {supported}"
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("CUDA capability probe failed: %s", e)
        return None


def detect_torch_device() -> str:
    """Auto-detect the best available torch device: ``cuda`` → ``mps`` → ``cpu``.

    Returns a ``torch.device``-compatible string. Selecting CUDA also applies
    the per-process VRAM cap (once). A CUDA device whose architecture is
    unsupported by the installed wheel (see :func:`cuda_incompatibility`) is
    skipped — kernels would crash at launch — and the ladder falls through to
    the next rung. Raises the 'torch'-extra ImportError without torch.
    """
    torch = _torch_module()
    if torch.cuda.is_available():
        reason = cuda_incompatibility()
        if reason is None:
            _configure_cuda_memory()
            configure_cuda_perf()
            return "cuda"
        logger.warning("Skipping CUDA: %s. Falling back to the next device.", reason)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def report_device(component: str, device: str, *, gpu_capable: bool = True) -> None:
    """Log where a model loaded, escalating the silent slow-path to WARNING.

    A GPU-capable model that quietly defaults to CPU on a GPU box is easy to
    miss in INFO-level logs — and ruinously slow (e.g. ~197s vs ~0.7s for
    sentence segmentation on a large paper). This logs an INFO line for the
    normal case and a WARNING when ``component`` is on CPU while a *usable*
    CUDA GPU is present (``is_available()`` and architecture-compatible — see
    :func:`cuda_incompatibility`). Pass ``gpu_capable=False`` for components
    with no GPU implementation, so their CPU placement is reported as INFO,
    not flagged as a regression. Torch is consulted only when it is already
    loaded, so the ONNX path never imports it here.
    """
    on_cpu = str(device).split(":", 1)[0].strip().lower() == "cpu"
    torch = _torch_module(required=False)
    gpu_usable = torch is not None and torch.cuda.is_available() and cuda_incompatibility() is None
    if gpu_capable and on_cpu and gpu_usable:
        logger.warning(
            "%s is running on CPU while a usable CUDA GPU is available — this is "
            "often dramatically slower. If unintended, enable GPU for this component.",
            component,
        )
    else:
        logger.info("%s running on device: %s", component, device)
