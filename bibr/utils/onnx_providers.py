"""Shared ONNX Runtime execution provider configuration.

Builds the provider chain for ONNX Runtime sessions. All model deployments
call ``get_ort_providers()`` (or ``create_session()``) instead of constructing
providers ad-hoc.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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

    Falls back gracefully: CUDA → CoreML → CPU.

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

    Returns ``(session, device)`` where ``device`` is ``"cuda"`` or ``"cpu"``.
    Graph optimisations are left at ORT's default (all), which is what the
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
    return session, selected_device(providers)
