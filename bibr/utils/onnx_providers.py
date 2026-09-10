"""Shared ONNX Runtime execution provider configuration.

Builds the provider chain for ONNX Runtime sessions. All model deployments
call ``get_ort_providers()`` instead of constructing providers ad-hoc.
"""

import logging

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
        from bibr.utils.ml_extra import ml_import_error

        raise ml_import_error("ONNX Runtime inference") from e

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
        try:
            import torch

            if torch.cuda.is_available():
                logger.warning(
                    "CUDA GPU detected but onnxruntime-gpu is not installed — "
                    "%s will run on CPU. Install with: "
                    "uv pip install 'onnxruntime-gpu[cuda,cudnn]'",
                    model_name or "model",
                )
        except ImportError:
            pass

    return providers
