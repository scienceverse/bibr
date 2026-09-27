"""Registry of OCR backend implementations.

Each concrete backend class applies ``@register`` at import time. Callers
(e.g. ``ResourceManager``) use ``create(name, **kwargs)`` to construct an
instance without knowing the concrete class.

Extra keyword arguments not consumed by the backend are silently ignored
(backends declare ``**_kw: object`` on their constructors).
"""

from __future__ import annotations

import functools
import logging
import platform
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

from bibr.ocr.backend import OcrBackend

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.ocr.profiles import OcrProfileName

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, type[OcrBackend]] = {}


@dataclass(frozen=True)
class OcrBackendCandidate:
    """One concrete OCR runtime that may satisfy an automatic request."""

    backend: str
    model: str
    profile: OcrProfileName


# The managed PaddleOCR-VL vLLM server's ``--gpu-memory-utilization``. vLLM
# claims this fraction of total VRAM and will not start when less is free, so
# the classifier device picker leaves it free (``bibr.local.pipeline``).
PADDLE_VLLM_GPU_MEMORY_UTILIZATION = 0.92

# PaddleOCR-VL through the managed vLLM server needs an NVIDIA GPU with room
# for the ~1.8 GB bf16 weights plus vLLM's KV/activation budget at
# ``PADDLE_VLLM_GPU_MEMORY_UTILIZATION``. Below this floor bibr's documented
# Linux contract ("NVIDIA GPU <= 8 GB uses llama.cpp") routes OCR to
# ``glm-llama``.
PADDLE_VLLM_MIN_VRAM_GB = 8.0


@functools.cache
def _cuda_vram_gb() -> float | None:
    """Total VRAM (GiB) of the first NVIDIA GPU, or ``None`` when none is usable.

    Probed once per process: hardware does not change under a running
    pipeline, and the ``nvidia-smi`` subprocess behind it is not free. Tests
    patch this name to pin a platform.
    """
    from bibr.local.llm_models import detect_hardware

    platform_key, memory_gb = detect_hardware()
    if platform_key == "cuda":
        return memory_gb
    return None


_PROBE = object()


def paddle_vllm_unavailable_reason(*, vram_gb: float | None | object = _PROBE) -> str | None:
    """Why the managed ``paddle-vllm`` runtime cannot start here, or ``None``.

    Consulted by the automatic ``paddle`` chain (so a CPU-only Linux box never
    spends its startup budget bootstrapping vLLM), by the managed launcher
    itself, and by the ``bibr chew`` preflight.
    """
    vram = _cuda_vram_gb() if vram_gb is _PROBE else vram_gb
    if vram is None:
        return "no NVIDIA GPU detected (paddle-vllm needs CUDA)"
    if vram < PADDLE_VLLM_MIN_VRAM_GB:
        return (
            f"the GPU has {vram:.0f} GB of VRAM and paddle-vllm needs at least "
            f"{PADDLE_VLLM_MIN_VRAM_GB:.0f} GB (llama.cpp is the low-VRAM OCR path)"
        )
    return None


def default_backend() -> str:
    """Pick the preferred local OCR runtime supported by the host platform."""
    from bibr.config import snapshot_settings

    # The automatic ``paddle`` chain skips runtimes the host cannot run (no
    # NVIDIA GPU with room → no managed vLLM), so its head — not a fixed
    # platform guess — is the default that can actually start here.
    return resolve_backend_candidates("paddle", snapshot_settings())[0].backend


def default_glm_backend() -> str:
    """Pick a concrete GLM runtime without entering the Paddle selector."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "glm-rapid-mlx"
    return "glm-llama"


def _candidate_model(backend: str, settings: GlobalSettings) -> str:
    from bibr.ocr.profiles import resolve_served_model

    # One table (bibr/ocr/profiles.py) for startup candidates and static
    # identity alike. serve-http resolves its family from the explicit
    # profile: with OCR_PROFILE=paddle the server is a Paddle endpoint and
    # must be asked for the Paddle served alias, not 'glm-ocr'.
    if backend == "serve-http":
        return resolve_served_model(
            requested_backend="serve-http",
            concrete_backend="serve-http",
            settings=settings,
            explicit_profile=settings.ocr.profile,
        )
    return resolve_served_model(
        requested_backend=backend,
        concrete_backend=backend,
        settings=settings,
        explicit_profile=None,
    )


def _candidate(backend: str, settings: GlobalSettings) -> OcrBackendCandidate:
    return OcrBackendCandidate(
        backend=backend,
        model=_candidate_model(backend, settings),
        profile="paddle" if backend.startswith("paddle") else "glm",
    )


def resolve_backend_candidates(
    name: str | None, settings: GlobalSettings
) -> tuple[OcrBackendCandidate, ...]:
    """Resolve an OCR request to concrete, deterministic startup candidates.

    Only the ``paddle`` automatic selector has more than one candidate.
    Explicit concrete backends are deliberately single-candidate requests so a
    user-selected runtime is never silently replaced by another implementation.
    """
    if name is None:
        ocr_opts = settings.ocr
        name = ocr_opts.backend if "backend" in ocr_opts.model_fields_set else "paddle"
    if name in ("http", "glm-http"):
        name = "glm-http"
    elif name in ("glm", "glm-local"):
        name = default_glm_backend()

    if name != "paddle":
        return (_candidate(name, settings),)
    return tuple(_candidate(backend, settings) for backend in automatic_backend_names())


def automatic_backend_names() -> tuple[str, ...]:
    """The concrete runtimes the automatic ``paddle`` selector tries here, in order."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return ("paddle-rapid-mlx", "paddle-mlx-vlm", "glm-rapid-mlx", "glm-llama")
    if sys.platform == "win32":
        return ("glm-llama",)
    if sys.platform.startswith("linux") and platform.machine().lower() in {"x86_64", "amd64"}:
        # The managed vLLM runtime is only a candidate on hardware that can
        # run it. Otherwise a CPU-only or small-GPU box spent the whole 900 s
        # startup budget bootstrapping vLLM before falling through to llama.cpp.
        vllm_blocker = paddle_vllm_unavailable_reason()
        if vllm_blocker is None:
            return ("paddle-vllm", "glm-llama")
        logger.debug("paddle-vllm skipped in the automatic OCR chain: %s", vllm_blocker)
    return ("glm-llama",)


#: The cloud vision-LLM OCR backends. They call their provider's API (set
#: ``OCR_VISION_BASE_URL`` to move it), never an OCR server, and hold no local
#: weights. The one definition every OCR module imports.
CLOUD_VISION_OCR_BACKENDS = frozenset({"gemini", "openai", "anthropic"})

#: Requests an ``ocr_url`` leaves alone: the two HTTP clients already address
#: the URL, and the cloud vision backends ignore it.
_URL_KEEPS_BACKEND = CLOUD_VISION_OCR_BACKENDS | {"paddle-http", "serve-http"}


def url_request(explicit: str | None, configured: str | None) -> str | None:
    """Return the OCR backend request an ``ocr_url`` is resolved from.

    ``explicit`` is the caller's backend (``--ocr``, ``ocr_backend=``,
    ``RunConfig.ocr_backend``) and is the request whenever it is set.
    Without it the configured ``OCR_BACKEND`` is the request, unless it names
    a cloud vision provider: a server URL the caller named outranks a
    configured cloud default, so page images never go to a cloud provider
    because of a ``.env`` setting. ``None`` means no request (the default
    ``paddle`` selector). Every entry point calls this before
    :func:`resolve_url_backend`.
    """
    if explicit:
        return explicit
    if configured in CLOUD_VISION_OCR_BACKENDS:
        return None
    return configured


@overload
def resolve_url_backend(requested: str | None, ocr_url: str) -> str: ...


@overload
def resolve_url_backend(requested: str, ocr_url: str | None) -> str: ...


@overload
def resolve_url_backend(requested: str | None, ocr_url: str | None) -> str | None: ...


def resolve_url_backend(requested: str | None, ocr_url: str | None) -> str | None:
    """Apply an explicit OCR server URL to an OCR backend request.

    The one rule behind ``bibr chew --ocr-url``, ``LocalPipeline(ocr_url=...)``
    (and so ``bibr.chew``), ``ResourceManager`` and the OCR runtime identity,
    so they all start, cache and report the same backend. ``requested`` comes
    from :func:`url_request`. An OCR URL means Paddle, bibr's default OCR:

    - without a URL the request is returned unchanged;
    - ``paddle-http``, ``serve-http`` and the cloud vision backends
      (:data:`CLOUD_VISION_OCR_BACKENDS`) are kept;
    - a GLM request (``glm``, any ``glm-*`` name, or the legacy ``http``
      alias of ``glm-http``) becomes ``glm-http``;
    - anything else, including the ``paddle`` selector and no request at
      all, becomes ``paddle-http``.

    Idempotent: a resolved backend resolves to itself.
    """
    if not ocr_url:
        return requested
    if requested in _URL_KEEPS_BACKEND:
        return requested
    if requested is not None and (requested == "http" or requested.startswith("glm")):
        return "glm-http"
    return "paddle-http"


def resolve_backend_name(name: str | None, settings: GlobalSettings | None = None) -> str:
    """Expand OCR backend aliases to concrete registry names.

    The single source of truth for alias/platform resolution — used by the
    CLI, ``LocalPipeline``, and the demo, so ``bibr.chew(ocr="glm")``
    behaves exactly like ``bibr chew --ocr glm``.

    ``None`` falls back to an explicitly-set ``OCR_BACKEND`` setting, else the
    platform default.
    """
    from bibr.config import snapshot_settings

    effective = settings if settings is not None else snapshot_settings()
    return resolve_backend_candidates(name, effective)[0].backend


def register(cls: type[OcrBackend]) -> type[OcrBackend]:
    """Class decorator that registers an `OcrBackend` under ``cls.name``."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must declare a class-level `name` attribute")
    if name in _BACKENDS and _BACKENDS[name] is not cls:
        raise ValueError(f"OCR backend {name!r} already registered to {_BACKENDS[name].__name__}")
    _BACKENDS[name] = cls
    return cls


def create(name: str, **kwargs: object) -> OcrBackend:
    """Instantiate a registered backend by name."""
    cls = _BACKENDS.get(name)
    if cls is None:
        known = sorted(_BACKENDS)
        raise ValueError(f"Unknown OCR backend: {name!r}. Known: {known}")
    return cls(**kwargs)


def known_backends() -> list[str]:
    """Return a sorted list of registered backend names (for diagnostics)."""
    return sorted(_BACKENDS)
