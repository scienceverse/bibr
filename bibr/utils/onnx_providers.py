"""Shared ONNX Runtime execution provider configuration.

Builds the provider chain for ONNX Runtime sessions. All model deployments
call ``get_ort_providers()`` (or ``create_session()``) instead of constructing
providers ad-hoc.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_preload_lock = threading.Lock()
_cuda_libraries_preloaded = False


def _preload_cuda_libraries(ort) -> None:
    """Load the CUDA and cuDNN libraries the CUDA provider needs, once per process.

    ``onnxruntime-gpu[cuda,cudnn]`` installs CUDA and cuDNN as ``nvidia-*``
    wheels, but ORT's CUDA provider library does not look in their
    directories. Unless something loaded them first (a PyTorch built for the
    same CUDA major, imported earlier, or ``LD_LIBRARY_PATH``), the provider
    fails to load and ORT runs the session on CPU. ``onnxruntime.preload_dlls()`` (ORT >= 1.21) loads
    them. It reports failures with ``print``; those go to the log instead of
    stdout, next to the fallback warning in :func:`session_device`.
    """
    global _cuda_libraries_preloaded
    with _preload_lock:
        if _cuda_libraries_preloaded:
            return
        _cuda_libraries_preloaded = True
        preload = getattr(ort, "preload_dlls", None)
        if preload is None:
            return
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                preload()
        except Exception as exc:  # noqa: BLE001 — a failed preload must not break ORT setup
            logger.warning("onnxruntime.preload_dlls() failed: %s", exc)
        lines = [line.strip() for line in printed.getvalue().splitlines() if line.strip()]
        if lines:
            logger.warning("onnxruntime.preload_dlls(): %s", " ".join(lines))


def cuda_provider_available() -> bool:
    """True when onnxruntime exposes ``CUDAExecutionProvider`` in this process.

    The auto-detect signal for ONNX models (sentence segmenter): if the GPU
    execution provider is present, default to using it. Degrades to ``False``
    when onnxruntime is missing or fails to import — never raises.
    """
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001 — absence/breakage just means "no CUDA EP"
        return False


def get_ort_providers(
    *,
    enable_cuda: bool = True,
    model_name: str = "",
    gpu_mem_limit: int | None = None,
) -> list[str | tuple[str, dict]]:
    """Build ONNX Runtime provider list.

    Falls back gracefully: CUDA → CoreML → CPU. Including CUDA first loads
    the CUDA libraries it needs (see :func:`_preload_cuda_libraries`).

    Args:
        enable_cuda: Whether to include CUDAExecutionProvider.
        model_name: Human-readable model name for logging.
        gpu_mem_limit: Optional cap on the CUDA EP arena size in bytes.

    Returns:
        Ordered list of providers suitable for ``ort.InferenceSession(providers=...)``.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:  # pragma: no cover
        from bibr.utils.ml_extra import onnxruntime_import_error

        raise onnxruntime_import_error("ONNX Runtime inference") from e

    available = set(ort.get_available_providers())
    providers: list[str | tuple[str, dict]] = []

    if enable_cuda and "CUDAExecutionProvider" in available:
        _preload_cuda_libraries(ort)
        cuda_opts: dict[str, str | int | bool] = {
            "arena_extend_strategy": "kSameAsRequested",
            "do_copy_in_default_stream": True,
        }
        if gpu_mem_limit is not None:
            cuda_opts["gpu_mem_limit"] = gpu_mem_limit
        providers.append(("CUDAExecutionProvider", cuda_opts))

    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")

    providers.append("CPUExecutionProvider")

    if enable_cuda and "CUDAExecutionProvider" not in available:
        # Only consult a torch that is *already* loaded: the ONNX path must not
        # import torch just to phrase this warning (and a core install has none).
        torch = sys.modules.get("torch")
        try:
            if torch is not None and torch.cuda.is_available():
                logger.warning(
                    "CUDA GPU detected but onnxruntime-gpu is not installed — "
                    "%s will run on CPU. Install with: "
                    "uv pip install 'onnxruntime-gpu[cuda,cudnn]'",
                    model_name or "model",
                )
        except Exception as exc:  # noqa: BLE001 — a broken torch must not break ORT setup
            logger.debug("torch CUDA probe failed while building ORT providers: %s", exc)

    return providers


def selected_device(providers: list[str | tuple[str, dict]]) -> str:
    """``"cuda"`` when the CUDA provider heads the chain, else ``"cpu"``."""
    for provider in providers:
        name = provider[0] if isinstance(provider, tuple) else provider
        if name == "CUDAExecutionProvider":
            return "cuda"
    return "cpu"


def session_device(
    session, requested: list[str | tuple[str, dict]], *, model_name: str = ""
) -> str:
    """The device ``session`` actually runs on: ``"cuda"`` or ``"cpu"``.

    Read from the session's own providers, not the ``requested`` chain: ORT
    drops a provider that fails to start (a CUDA library it cannot load, no
    visible GPU, a driver too old for its CUDA) and runs the session on the
    next one without raising. Losing requested CUDA that way logs a warning.
    """
    device = selected_device(session.get_providers())
    if device != "cuda" and selected_device(requested) == "cuda":
        logger.warning(
            "%s requested CUDA, but onnxruntime could not start its CUDA execution "
            "provider and runs it on CPU. onnxruntime's own error names the cause, "
            "usually a CUDA or cuDNN library it cannot load (install "
            "'onnxruntime-gpu[cuda,cudnn]') or no visible GPU.",
            model_name or "ONNX model",
        )
    return device


def enable_cuda_for(device: str | None) -> bool:
    """Map a torch-style device request onto the CUDA provider switch.

    ``None`` means auto (use CUDA when the provider exists); ``cpu`` forces the
    CPU provider; anything else (``cuda``, ``cuda:1``, ``mps``) allows CUDA and
    otherwise falls through the provider chain.
    """
    if device is None:
        return True
    return str(device).split(":", 1)[0].strip().lower() != "cpu"


def create_session(
    model_path: str | Path,
    *,
    device: str | None = None,
    model_name: str = "",
    gpu_mem_limit: int | None = None,
):
    """Open an ``InferenceSession`` on ``model_path`` and report its device.

    Returns ``(session, device)`` where ``device`` is ``"cuda"`` or ``"cpu"``,
    the one the session got (see :func:`session_device`). Graph
    optimisations are left at ORT's default (all), which is what the
    wtpsplit segmenter already runs with.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:  # pragma: no cover
        from bibr.utils.ml_extra import onnxruntime_import_error

        raise onnxruntime_import_error(model_name or "ONNX Runtime inference") from e

    providers = get_ort_providers(
        enable_cuda=enable_cuda_for(device),
        model_name=model_name,
        gpu_mem_limit=gpu_mem_limit,
    )
    options = ort.SessionOptions()
    options.log_severity_level = 3  # errors only; ORT's warnings are noisy at load
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    return session, session_device(session, providers, model_name=model_name)
