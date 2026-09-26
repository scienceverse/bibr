"""Model-specific OCR prompts and generation settings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.pipeline.context import RunConfig

OcrTask = Literal["text", "table", "formula"]
OcrProfileName = Literal["paddle", "glm"]
PADDLE_TABLE_RECOVERY_MAX_TOKENS: Final[int] = 8192


@dataclass(frozen=True)
class OcrRequestSettings:
    """Generation settings shared by OCR requests for a model family."""

    temperature: float
    max_tokens: int
    task_max_tokens: tuple[tuple[OcrTask, int], ...] = ()
    top_k: int | None = None
    repetition_penalty: float | None = None

    def max_tokens_for(self, task: OcrTask) -> int:
        """Return the task-specific output cap, falling back to the profile default."""
        return dict(self.task_max_tokens).get(task, self.max_tokens)


@dataclass(frozen=True)
class OcrImageGeometry:
    """Patch-grid resize budget a vision encoder expects for a region crop.

    These are properties of the *model*, not of bibr: every field here has a
    counterpart in the served model's ``preprocessor_config.json``. Carrying
    them on the profile is what keeps one family's constants from silently
    governing another's — see ``PADDLE_IMAGE_GEOMETRY`` for what that cost.
    """

    t_patch_size: int
    max_pixels: int
    min_pixels: int
    patch_expand_factor: int = 1

    def cache_fingerprint(self) -> str:
        """Stable identity for the OCR result cache key.

        Geometry changes what pixels reach the model, so it changes the OCR
        text. Callers must fold this into the cache key or a geometry change
        silently serves results produced under the old budget.
        """
        return (
            f"t{self.t_patch_size}"
            f":max{self.max_pixels}"
            f":min{self.min_pixels}"
            f":x{self.patch_expand_factor}"
        )


#: GLM-OCR / Qwen2-VL-style geometry. ``t_patch_size=2`` is the temporal patch
#: size those encoders use to pair video frames; for a still region crop it acts
#: as a plain multiplier on the pixel-budget test in ``smart_resize``, so the
#: usable budget is ``max_pixels / 2``.
GLM_IMAGE_GEOMETRY = OcrImageGeometry(
    t_patch_size=2,
    max_pixels=14 * 14 * 4 * 1280,  # 1_003_520
    min_pixels=112 * 112,  # 12_544
)

# Paddle uses its own geometry: reusing the GLM temporal factor would halve its accepted pixel
# budget and lower the minimum crop size.
PADDLE_IMAGE_GEOMETRY = OcrImageGeometry(
    t_patch_size=1,
    max_pixels=1_003_520,
    min_pixels=112_896,
)


@dataclass(frozen=True)
class OcrProfile:
    """Prompts, output normalization, and request settings for an OCR model family."""

    name: OcrProfileName
    normalizer_version: str
    prompts: dict[OcrTask, str]
    request: OcrRequestSettings
    image: OcrImageGeometry

    def prompt_for(self, task: OcrTask) -> str:
        """Return the model-specific prompt for an OCR region type."""
        return self.prompts[task]

    def task_for_prompt(self, prompt: str) -> OcrTask:
        """Resolve a known task prompt, treating custom prompts as text for compatibility."""
        for task, task_prompt in self.prompts.items():
            if prompt == task_prompt:
                return task
        return "text"


@dataclass(frozen=True)
class OcrRuntimeIdentity:
    """Concrete OCR implementation identity used for cache and provenance."""

    backend: str
    model: str
    profile: OcrProfileName
    normalizer_version: str


PADDLE_PROFILE = OcrProfile(
    name="paddle",
    normalizer_version="paddle-canonical-v1",
    prompts={
        "text": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
    },
    request=OcrRequestSettings(
        temperature=0.0,
        max_tokens=1024,
        task_max_tokens=(("table", 4096),),
    ),
    image=PADDLE_IMAGE_GEOMETRY,
)

GLM_PROFILE = OcrProfile(
    name="glm",
    normalizer_version="glm-canonical-v1",
    prompts={
        "text": "Text Recognition:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
    },
    request=OcrRequestSettings(
        temperature=0.01,
        max_tokens=16384,
        top_k=1,
        repetition_penalty=1.1,
    ),
    image=GLM_IMAGE_GEOMETRY,
)

_PROFILES: dict[OcrProfileName, OcrProfile] = {"paddle": PADDLE_PROFILE, "glm": GLM_PROFILE}

#: Served-model alias an externally-managed GLM-OCR server advertises on
#: ``/v1/models``. This is what an HTTP backend must ASK for — distinct from
#: ``OCR_LOCAL_MODEL`` (``THUDM/GLM-OCR``), which is the HuggingFace repo id a
#: *local* runtime loads weights from. Requesting the repo id over HTTP makes
#: the readiness gate wait out its whole deadline for a model id the server
#: never advertises. Mirrors ``BaseHttpOcrClient._DEFAULT_MODEL``.
GLM_SERVED_MODEL_ALIAS = "glm-ocr"


def resolve_ocr_profile(
    *,
    explicit: str | None,
    backend: str,
    model: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> OcrProfile:
    """Resolve the OCR profile from an explicit override or model/backend name.

    Custom aliases are intentionally not guessed: callers must supply
    ``OCR_PROFILE`` so their request settings and output normalizer are explicit.
    """
    profile_name = explicit.lower() if explicit else _infer_profile_name(backend, model)
    if profile_name not in _PROFILES:
        if explicit:
            raise ValueError("OCR_PROFILE must be 'paddle' or 'glm'.")
        raise ValueError(
            "Could not infer an OCR profile from backend/model; set OCR_PROFILE to 'paddle' or 'glm'."
        )

    resolved = _PROFILES[profile_name]
    request = replace(
        resolved.request,
        max_tokens=max_tokens if max_tokens is not None else resolved.request.max_tokens,
        task_max_tokens=() if max_tokens is not None else resolved.request.task_max_tokens,
        temperature=temperature if temperature is not None else resolved.request.temperature,
    )
    return replace(resolved, request=request)


def _infer_profile_name(backend: str, model: str) -> OcrProfileName | None:
    """Infer known model families case-insensitively from backend and model names."""
    candidates = (backend, model)
    for name in candidates:
        normalized = name.lower()
        if "paddle" in normalized:
            return "paddle"
        if "glm" in normalized:
            return "glm"
    return None


def resolve_ocr_runtime_identity(cfg: RunConfig, settings: GlobalSettings) -> OcrRuntimeIdentity:
    """Resolve the OCR implementation before cache lookup or startup.

    This is intentionally independent of a constructed backend client: the
    cache must be able to distinguish model families without starting a local
    server merely to discover its identity.
    """
    requested_backend = cfg.ocr_backend or settings.ocr.backend
    from bibr.ocr.registry import resolve_url_backend

    backend = resolve_url_backend(requested_backend, cfg.ocr_url) or requested_backend
    model = (
        settings.ocr.paddle_served_model
        if requested_backend == "paddle-vllm"
        else cfg.ocr_model
        or _default_ocr_model(
            requested_backend=requested_backend,
            concrete_backend=backend,
            settings=settings,
            explicit_profile=cfg.ocr_profile or settings.ocr.profile,
        )
    )
    explicit_profile = cfg.ocr_profile or settings.ocr.profile
    if (backend in {"gemini", "openai", "anthropic"} and explicit_profile is None) or (
        backend == "serve-http"
        and explicit_profile is None
        and _infer_profile_name(backend, model) is None
    ):
        profile = GLM_PROFILE
    else:
        profile = resolve_ocr_profile(
            explicit=explicit_profile,
            backend=backend,
            model=model,
            max_tokens=settings.ocr.generation_max_tokens,
            temperature=settings.ocr.generation_temperature,
        )
    return OcrRuntimeIdentity(
        backend=backend,
        model=model,
        profile=profile.name,
        normalizer_version=profile.normalizer_version,
    )


def resolve_served_family(
    *, requested_backend: str, explicit_profile: str | None
) -> OcrProfileName:
    """Which model family an HTTP OCR endpoint belongs to: paddle or glm.

    A single predicate shared by the served-model table below and the serve
    defaults, so an explicit paddle profile (or a concrete paddle HTTP
    backend) selects the Paddle served alias everywhere. The bare ``paddle``
    automatic selector and the managed-local Paddle runtimes do not count:
    they name no concrete server family.
    """
    if (explicit_profile or "") == "paddle" or requested_backend == "paddle-http":
        return "paddle"
    return "glm"


def resolve_served_model(
    *,
    requested_backend: str,
    concrete_backend: str,
    settings: GlobalSettings,
    explicit_profile: str | None,
) -> str:
    """Return the model string selected by the configured OCR backend.

    The single table behind the registry startup candidates, the static
    runtime identity, the serve defaults and ``--dry-run``: every path that
    asks what an OCR server serves must agree. ``requested_backend`` is the
    configured selector, ``concrete_backend`` the runtime after the
    ``ocr_url`` rewrite.
    """
    ocr = settings.ocr
    if concrete_backend in {"gemini", "openai", "anthropic"}:
        return ocr.model or settings.ocr_vision.model
    if concrete_backend == "serve-http":
        if (
            resolve_served_family(
                requested_backend=requested_backend, explicit_profile=explicit_profile
            )
            == "paddle"
        ):
            return ocr.paddle_served_model
        return ocr.model or GLM_SERVED_MODEL_ALIAS
    if concrete_backend == "glm-http":
        # Keyed on the *concrete* backend so an ``ocr_url`` override — which
        # rewrites any local GLM runtime to glm-http — asks for the served
        # alias rather than the local backend's HuggingFace repo id.
        return ocr.model or GLM_SERVED_MODEL_ALIAS
    if requested_backend in {"paddle-http", "paddle-vllm"}:
        return ocr.paddle_served_model
    if requested_backend == "paddle-mlx-vlm":
        return ocr.paddle_mlx_model
    if requested_backend == "paddle-rapid-mlx":
        return ocr.paddle_rapid_mlx_model
    if requested_backend == "paddle":
        return ocr.paddle_model
    if requested_backend == "glm-rapid-mlx":
        return ocr.rapid_mlx_model
    if requested_backend == "glm-llama":
        return ocr.llama_cpp_model
    return ocr.model or ocr.local_model


def _default_ocr_model(
    *,
    requested_backend: str,
    concrete_backend: str,
    settings: GlobalSettings,
    explicit_profile: str | None,
) -> str:
    """Return the model string selected by the configured OCR backend."""
    return resolve_served_model(
        requested_backend=requested_backend,
        concrete_backend=concrete_backend,
        settings=settings,
        explicit_profile=explicit_profile,
    )
