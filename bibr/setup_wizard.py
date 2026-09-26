"""Interactive setup wizard for bibr.

Run with ``bibr setup`` after installation for the recommended easy flow, or
``bibr setup --advanced`` for the detailed provider/backend picker.
"""

import importlib.metadata
import importlib.resources as resources
import importlib.util
import os
import platform as platform_lib
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rich.console import Console
from rich.prompt import Confirm, Prompt

from bibr.env_utils import _format_env_value, write_env_text
from bibr.env_utils import merge_env as _merge_env  # re-exported for tests
from bibr.local.cli import ui
from bibr.local.llm_models import (
    REGISTRY,
    cuda_llm_backend_for,
    detect_hardware,
    get_model,
    variants_for,
)
from bibr.presets import PresetManager
from bibr.utils.hosts import refuse_plaintext_llm_key
from bibr.utils.onnx_providers import onnxruntime_gpu_reinstall_command

if TYPE_CHECKING:
    from bibr.config import GlobalSettings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXTRAS = {
    "ml": "Local ML models for layout detection and NER references (recommended; Windows uses this with glm-llama)",
    "vllm": "Managed vLLM runtime for PaddleOCR-VL OCR and the local LLM on Linux/CUDA (Python 3.11-3.13)",
    "local": "In-process local OCR/LLM runtime (vllm-mlx) for Apple Silicon",
    "local-mlx": "Local OCR optimized for Apple Silicon Macs",
    "cache": "Response caching for the HTTP API (requires Redis)",
    "gpu": "GPU acceleration for sentence splitting (NVIDIA CUDA)",
    "demo": "Interactive web demo UI",
}

# Prints whether the onnxruntime that imports is the GPU build: only its binary
# registers the CUDA execution provider, with or without a GPU present.
_GPU_BUILD_PROBE = (
    "import onnxruntime; print('CUDAExecutionProvider' in onnxruntime.get_available_providers())"
)

LLM_DEFAULTS: dict[str, dict[str, str]] = {
    "google": {"model": "gemini-3.5-flash-lite", "key_env": "GOOGLE_API_KEY"},
    "openai": {"model": "gpt-5-nano", "key_env": "LLM_API_KEY"},
    "anthropic": {"model": "claude-haiku-4-5-20251001", "key_env": "ANTHROPIC_API_KEY"},
    "groq": {"model": "llama-3.3-70b-versatile", "key_env": "GROQ_API_KEY"},
    "ollama": {"model": "gpt-oss:20b", "key_env": ""},
}

WTPSPLIT_MODELS: dict[str, str] = {
    "sat-6l-sm": "Best speed/quality trade-off, recommended (default)",
    "sat-12l-sm": "Higher quality, slower",
    "sat-3l-sm": "Faster, but 3l and lower produce more frequent text artifacts",
    "sat-1l-sm": "Fastest, but 3l and lower produce more frequent text artifacts",
}

# Honest first-run download note per resolved OCR backend, for the smoke-test
# step. Sizes are approximate. Cloud vision backends and external HTTP servers
# need no local download.
_OCR_BACKEND_DOWNLOAD_NOTES: dict[str, str] = {
    "paddle": "tries the local Paddle OCR candidates on first run; GLM remains a fallback",
    "paddle-vllm": "downloads the PaddleOCR-VL weights (~1.8 GB) on first run",
    "paddle-rapid-mlx": "downloads the PaddleOCR-VL weights (~1 GB quantized) on first run",
    "paddle-mlx-vlm": "downloads the PaddleOCR-VL weights (~1 GB quantized) on first run",
    "paddle-http": "uses your external Paddle OCR server — no local download, but it must "
    "already be running",
    "glm-llama": "downloads the quantized GLM-OCR weights (~1-2 GB GGUF) on first run",
    "glm-http": "uses your external GLM-OCR server — no local download, but it must "
    "already be running",
    "gemini": "cloud vision OCR — no local model download",
    "openai": "cloud vision OCR — no local model download",
    "anthropic": "cloud vision OCR — no local model download",
}

# "OCR server unreachable" hints, per resolved backend. HTTP backends point at
# an externally-managed server the user must launch themselves (commands
# mirror the ones documented in README.md's "external OCR server" section);
# managed backends are started by bibr itself, so an unreachable managed
# server usually means a missing dependency/launcher, not a manual command.
_OCR_HTTP_LAUNCH_HINTS: dict[str, str] = {
    "glm-http": "python -m sglang.launch_server --model-path zai-org/GLM-OCR --port 8080"
    "  (or: vllm serve zai-org/GLM-OCR --port 8080 --dtype auto)",
}

# Substrings that show up in auth/API-key failures across providers'
# underlying SDK exceptions (OpenAI, Google genai, Anthropic, Groq).
_AUTH_ERROR_MARKERS = (
    "api key",
    "apikey",
    "unauthorized",
    "401",
    "invalid_api_key",
    "permission_denied",
    "authentication",
)


def _available_extras() -> dict[str, str]:
    """Return setup extras that make sense on the current platform."""
    extras = dict(EXTRAS)

    if sys.platform == "win32":
        # Native Windows uses llama.cpp for local OCR/LLM. The vLLM and
        # vllm-mlx extras are Linux/macOS only and otherwise steer users into
        # marker-only installs that cannot help the Windows path.
        extras.pop("local", None)
        extras.pop("local-mlx", None)
        extras.pop("vllm", None)
        extras["ml"] = (
            "Local ML models for layout detection and NER references "
            "(recommended with Windows glm-llama OCR)"
        )
    elif sys.platform == "linux":
        # ``local`` / ``local-mlx`` resolve to nothing on Linux (their only
        # member carries an Apple-Silicon marker), so offering them would
        # install nothing and hide that the runtime the plan needs is ``vllm``.
        extras.pop("local", None)
        extras.pop("local-mlx", None)
    elif sys.platform == "darwin":
        extras.pop("gpu", None)
        extras.pop("vllm", None)
        import platform as _platform

        if _platform.machine() != "arm64":
            extras.pop("local", None)
            extras.pop("local-mlx", None)
    else:
        extras.pop("gpu", None)
        extras.pop("local", None)
        extras.pop("local-mlx", None)
        extras.pop("vllm", None)

    return extras


def _ml_extra_available() -> bool:
    """Best-effort check for the heavy deps that make up the ml extra."""
    return all(
        importlib.util.find_spec(module) is not None
        for module in ("torch", "transformers", "cv2", "sklearn", "joblib")
    )


def _looks_like_auth_error(exc: BaseException) -> bool:
    """Best-effort sniff of *exc* for an auth/API-key failure signature."""
    text = str(exc).lower()
    return any(marker in text for marker in _AUTH_ERROR_MARKERS)


def _caused_by_configuration_error(exc: BaseException) -> bool:
    """Whether *exc* is or wraps a :class:`ConfigurationError`.

    chew() reports a bad setting wrapped — the pipeline raises
    ``ProcessingError('Layout initialization failed: ...')`` from the
    ``ConfigurationError`` — so walk ``__cause__``/``__context__``, not just
    the top exception.
    """
    from bibr.exceptions import ConfigurationError

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ConfigurationError):
            return True
        cause = current.__cause__
        current = cause if cause is not None else current.__context__
    return False


SetupPrompt = Literal["install_extras", "cloud_api_key", "private_server_url", "smoke_test"]
SetupTier = Literal["fully_local", "mostly_local", "private_server", "cloud_fallback"]


@dataclass(frozen=True)
class RecommendedSetup:
    tier: SetupTier
    env: dict[str, str]
    extras: set[str]
    preset_name: str
    privacy_summary: str
    download_summary: str
    runtime_summary: str
    required_prompts: tuple[SetupPrompt, ...]


def _local_llm_env(
    platform_key: str,
    memory_gb: float | None,
    sys_platform_name: str,
) -> dict[str, str]:
    """Choose the default managed local LLM env for easy setup."""
    backend = _default_local_llm_backend(platform_key, memory_gb, sys_platform_name)
    model = get_model("nuextract3")
    variants = variants_for(model, platform_key, memory_gb, backend)
    if not variants:
        variants = variants_for(model, platform_key, None, backend)
    variant = variants[0]
    return {"LLM_BACKEND": backend, "LLM_LOCAL_MODEL": variant.hf_id, **model.env}


def _default_local_llm_backend(
    platform_key: str,
    memory_gb: float | None,
    sys_platform_name: str,
) -> str:
    if platform_key == "mlx":
        from bibr.local.rapid_mlx import rapid_mlx_unavailable_reason

        if rapid_mlx_unavailable_reason() is None:
            return "rapid-mlx"
        return "vllm-mlx"
    if sys_platform_name == "win32":
        return "llama-cpp"
    return cuda_llm_backend_for(memory_gb)


def _runtime_for_variant(variant, default_backend: str) -> str:
    return variant.runtime or default_backend


def _setup_variants_for(model, platform_key: str, memory_gb: float | None, backend: str):
    """Show fitting variants, ordered with this machine's default backend first."""
    fitting = variants_for(model, platform_key, memory_gb)
    shown = fitting or variants_for(model, platform_key, None)
    return sorted(
        shown,
        key=lambda variant: (
            _runtime_for_variant(variant, backend) != backend,
            model.variants.index(variant),
        ),
    )


def _platform_local_extra(
    sys_platform_name: str,
    machine: str,
    *,
    ocr_backend: str | None = None,
) -> set[str]:
    """Pip extras needed for the in-process local OCR/LLM runtime.

    ``glm-llama`` uses an external llama.cpp binary (not a Python extra), so
    low-VRAM / Windows paths must not pull a GPU runtime. On Linux/CUDA the
    runtime is the ``vllm`` extra: both ``paddle-vllm`` OCR and the managed
    local LLM launch through it. The ``local`` extra resolves to nothing on
    Linux (its only member is Apple-Silicon-only vllm-mlx); selecting it here
    used to leave the first ``bibr chew`` to bootstrap vLLM through
    ``uv tool run`` in the middle of the run.
    """
    if ocr_backend == "glm-llama":
        return set()
    if sys_platform_name == "linux":
        return {"vllm"}
    if sys_platform_name == "darwin" and machine == "arm64":
        return {"local"}
    return set()


def _registry_short_label(hf_id: str) -> str | None:
    """Friendly display name for a registry model, matched by exact ``hf_id``.

    Returns the leading name of the model's registry label (before the first
    ``" — "`` blurb or ``" ("`` qualifier), e.g. ``"NuExtract 3"``; ``None`` for
    an id not in the curated registry.
    """
    for model in REGISTRY:
        for variant in model.variants:
            if variant.hf_id == hf_id:
                return model.label.split(" — ")[0].split(" (")[0].strip()
    return None


def _build_recommended_setup(
    *,
    platform_key: str | None = None,
    accelerator_memory_gb: float | None = None,
    system_memory_gb: float | None = None,
    sys_platform: str | None = None,
    machine: str | None = None,
    allow_cloud: bool = False,
) -> RecommendedSetup:
    """Build the easy setup recommendation without touching terminal state."""
    if (
        platform_key is None
        and accelerator_memory_gb is None
        and system_memory_gb is None
        and sys_platform is None
        and machine is None
    ):
        platform_key, accelerator_memory_gb = detect_hardware()

    sys_platform_name = sys_platform or sys.platform
    machine_name = machine or platform_lib.machine()
    if system_memory_gb is None:
        from bibr.local.pipeline import _get_system_memory_gb

        system_memory_gb = _get_system_memory_gb()
    from bibr.local.pipeline import _auto_memory_mode

    memory_mode = _auto_memory_mode(platform_key, accelerator_memory_gb, system_memory_gb)
    base_env = {
        "WTPSPLIT_MODEL": "sat-6l-sm",
        "REF_SEG_STRATEGY": "geom",
        "REF_PARSE_STRATEGY": "ner",
        "PIPELINE_MEMORY_MODE": memory_mode,
        "CROSSREF_CONSOLIDATE": "off",
    }

    viable_local = platform_key in {"cuda", "mlx"} and (
        accelerator_memory_gb is None or accelerator_memory_gb >= 5
    )
    if viable_local:
        ocr_backend = "paddle"
        env = {**base_env, "OCR_BACKEND": ocr_backend}
        env.update(_local_llm_env(platform_key, accelerator_memory_gb, sys_platform_name))
        # On Linux each half has its own VRAM gate: the automatic OCR chain
        # runs paddle-vllm from 8 GB (bibr.ocr.registry) and the LLM picks
        # vLLM only when a vLLM variant of the recommended model fits (11 GB
        # for NuExtract 3, bibr.local.llm_models). Below its gate a half runs
        # through llama.cpp, which needs llama-server on PATH — something only
        # the user can install — while the other half may still need the
        # vllm extra.
        llm_on_llama_cpp = sys_platform_name == "linux" and env.get("LLM_BACKEND") == "llama-cpp"
        ocr_on_llama_cpp = ocr_backend == "glm-llama"
        if sys_platform_name == "linux" and accelerator_memory_gb is not None:
            from bibr.ocr.registry import paddle_vllm_unavailable_reason

            ocr_on_llama_cpp = ocr_on_llama_cpp or (
                paddle_vllm_unavailable_reason(vram_gb=accelerator_memory_gb) is not None
            )
        runtime = "Fully local runs can be slow and may download several GB."
        if ocr_on_llama_cpp:
            runtime = (
                "Uses llama.cpp for OCR/LLM: install a CUDA (or Vulkan) build separately "
                "and put llama-server on PATH. Layout and NER use PyTorch; on older GPUs "
                "(Pascal / GTX 10-series) PyTorch falls back to CPU automatically — that "
                "is expected."
            )
        elif llm_on_llama_cpp:
            runtime = (
                "Uses vLLM for OCR and llama.cpp for the LLM (NuExtract 3's vLLM build needs "
                "11 GB of VRAM): install a CUDA (or Vulkan) build of llama.cpp separately "
                "and put llama-server on PATH."
            )
        elif platform_key == "mlx" and env.get("LLM_BACKEND") == "rapid-mlx":
            runtime = (
                "local LLM inference on Apple Silicon (rapid-mlx) typically takes a few "
                "minutes per paper — slower on weaker hardware. OCR-only stages are faster."
            )
        elif platform_key == "mlx":
            runtime = (
                "local LLM inference on Apple Silicon (vllm-mlx) is slow — a full paper "
                "extraction typically takes 15-30+ minutes. This is a known limitation, "
                "not a hang; OCR-only stages are much faster."
            )
        extras = {
            "ml",
            "demo",
            *_platform_local_extra(
                sys_platform_name,
                machine_name,
                ocr_backend="glm-llama" if ocr_on_llama_cpp else ocr_backend,
            ),
        }
        if "vllm" in extras and sys.version_info >= (3, 14):
            runtime += (
                " Python 3.14 has no vLLM wheels yet, so the vllm extra installs nothing; "
                "bibr runs vLLM through uv with a managed Python 3.13 instead (or use a "
                "3.11-3.13 interpreter for this project)."
            )
        return RecommendedSetup(
            tier="fully_local",
            env=env,
            extras=extras,
            preset_name="recommended-local",
            privacy_summary="document contents stay on this machine",
            download_summary="downloads several GB of local OCR, layout, reference, and LLM weights",
            runtime_summary=runtime,
            required_prompts=("install_extras", "smoke_test"),
        )

    if allow_cloud:
        return RecommendedSetup(
            tier="cloud_fallback",
            env={
                **base_env,
                "OCR_BACKEND": "gemini",
                "LLM_PROVIDER": "google",
                "LLM_MODEL": LLM_DEFAULTS["google"]["model"],
            },
            extras={"ml", "demo"},
            preset_name="recommended-cloud",
            privacy_summary="document contents are sent to the approved cloud provider",
            download_summary=(
                "downloads local layout/reference models; cloud OCR/LLM need no local model download"
            ),
            runtime_summary="Usually faster on weak local hardware than fully local processing.",
            required_prompts=("install_extras", "cloud_api_key", "smoke_test"),
        )

    return RecommendedSetup(
        tier="private_server",
        env={**base_env, "OCR_BACKEND": "glm-http"},
        extras={"ml", "demo"},
        preset_name="private-server",
        privacy_summary="document contents stay on your machine or your private server",
        download_summary="downloads local layout/reference models; OCR/LLM weights live on your server",
        runtime_summary="Best when this machine is too weak for fully local models.",
        required_prompts=("install_extras", "private_server_url", "smoke_test"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redact(text: str, api_key: str) -> str:
    """Replace API-key fragments in *text* with ``***`` (see ``utils.redact``)."""
    from bibr.utils.redact import redact_key

    return redact_key(text, api_key)


def _with_llm_routing(env_vars: dict[str, str]) -> dict[str, str]:
    """The wizard's answers plus a value for every LLM routing setting it left unset.

    What gets written to ``.env``, and what the connection test tests. When
    the wizard configures a provider, the settings that decide where LLM
    requests go and with which key must all come from its answers. Otherwise
    a value left by an earlier setup, in the ``.env`` being merged into or in
    ``~/.bibr/.env``, stays in effect: the google, anthropic and groq adapters
    send ``LLM_API_KEY`` in place of their own key whenever it is set, the
    openai adapter sends every request to ``LLM_BASE_URL`` when it is set,
    and a managed ``LLM_BACKEND`` does not use the provider at all. The test
    would then pass on the values just typed while ``bibr chew`` used the old
    ones. So ``LLM_BACKEND`` is written as ``cloud``, and ``LLM_API_KEY`` and
    ``LLM_BASE_URL`` as blank where the answers do not set them.
    """
    provider = env_vars.get("LLM_PROVIDER")
    if not provider:
        return dict(env_vars)
    routing = {"LLM_BACKEND": "cloud"}
    if provider != "ollama":
        routing["LLM_API_KEY"] = ""
    if provider == "openai":
        routing["LLM_BASE_URL"] = ""
    return {**env_vars, **{k: v for k, v in routing.items() if k not in env_vars}}


def _connection_test_settings(env_vars: dict[str, str]) -> "GlobalSettings":
    """Settings for the LLM connection test: the current ones plus the wizard's answers.

    The provider, model, keys and endpoint come from the answers, as
    :func:`_with_llm_routing` writes them to ``.env``. Everything else
    (token caps, thinking budget, Instructor mode) comes from a snapshot of
    the current settings, which a merge into the existing ``.env`` keeps, so
    the test sends what the first ``bibr chew`` will send.
    """
    from bibr.config import snapshot_settings

    answers = _with_llm_routing(env_vars)
    settings = snapshot_settings()
    llm = settings.llm
    llm.provider = answers.get("LLM_PROVIDER", llm.provider)
    llm.model = answers.get("LLM_MODEL", llm.model)
    if "LLM_API_KEY" in answers:
        llm.api_key = answers["LLM_API_KEY"] or None
    if "LLM_BASE_URL" in answers:
        llm.base_url = answers["LLM_BASE_URL"] or None
    llm.ollama_base_url = answers.get("LLM_OLLAMA_BASE_URL") or llm.ollama_base_url
    for key_env in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
        if answers.get(key_env):
            setattr(settings, key_env, answers[key_env])
    if "LLM_ALLOW_INSECURE_HTTP" in answers:
        llm.allow_insecure_http = answers["LLM_ALLOW_INSECURE_HTTP"].strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    return settings


_OPENAI_FILTER_PATTERNS = (
    "embed",
    "tts",
    "whisper",
    "dall-e",
    "moderation",
    "realtime",
    "transcri",
)


def _fetch_models(
    provider: str, api_key: str, base_url: str = "", *, allow_insecure_http: bool = False
) -> list[str]:
    """Fetch available model IDs from a provider's API.

    Returns a sorted list of model ID strings, or an empty list on any error,
    including a public plain-HTTP ``base_url`` the key must not be sent to.
    """
    try:
        if provider == "google":
            return _fetch_google_models(api_key)
        elif provider == "anthropic":
            return _fetch_anthropic_models(api_key)
        elif provider == "groq":
            return _fetch_openai_compat_models(
                api_key,
                base_url="https://api.groq.com/openai/v1",
                filter_non_chat=True,
            )
        elif provider == "ollama":
            from bibr.clients.providers.ollama import ollama_openai_base_url

            return _fetch_openai_compat_models(
                "ollama",
                base_url=ollama_openai_base_url(base_url or "http://localhost:11434"),
                filter_non_chat=False,
            )
        elif provider == "openai":
            if base_url:
                refuse_plaintext_llm_key(base_url, api_key, allow_insecure_http=allow_insecure_http)
                return _fetch_openai_compat_models(
                    api_key, base_url=base_url, filter_non_chat=False
                )
            return _fetch_openai_compat_models(api_key, filter_non_chat=True)
        return []
    except Exception:
        return []


def _fetch_openai_compat_models(
    api_key: str,
    base_url: str = "",
    filter_non_chat: bool = True,
) -> list[str]:
    import openai

    kwargs: dict = {"api_key": api_key, "timeout": 10.0}
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.OpenAI(**kwargs)
    models = client.models.list()
    ids = [m.id for m in models]
    if filter_non_chat:
        ids = [mid for mid in ids if not any(p in mid.lower() for p in _OPENAI_FILTER_PATTERNS)]
    return sorted(ids)


def _fetch_google_models(api_key: str) -> list[str]:
    from google.genai import Client

    client = Client(api_key=api_key, http_options={"timeout": 10_000})
    ids: list[str] = []
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if "generateContent" in actions:
            name = model.name
            if name is None:
                continue
            if name.startswith("models/"):
                name = name[len("models/") :]
            ids.append(name)
    return sorted(ids)


def _fetch_anthropic_models(api_key: str) -> list[str]:
    import anthropic  # type: ignore[import-not-found]

    client = anthropic.Anthropic(api_key=api_key, timeout=10.0)
    response = client.models.list()
    ids = [m.id for m in response.data]
    return sorted(ids)


_MANUAL_ENTRY = "Enter model name manually..."


def _select_model(models: list[str], default: str, console: Console) -> str | None:
    """Present model list for selection. Returns chosen model ID or None for manual entry."""
    if not models:
        return None

    if len(models) == 1:
        console.print(f"  [green]One model available:[/green] {models[0]}")
        return models[0]

    import questionary

    choices = [*models, _MANUAL_ENTRY]
    effective_default = default if default in models else None

    answer = questionary.select(
        "Select a model:",
        choices=choices,
        default=effective_default,
    ).ask()

    if answer is None or answer == _MANUAL_ENTRY:
        return None
    selected: str = answer
    return selected


def _write_env_fresh(path: Path, env_vars: dict[str, str]) -> None:
    """Write a new ``.env`` file with comment section headers."""
    sections: dict[str, list[str]] = {
        "LLM Provider": [
            "LLM_PROVIDER",
            "LLM_MODEL",
            "LLM_BACKEND",
            "LLM_LOCAL_MODEL",
            "LLM_INSTRUCTOR_MODE",
            "GOOGLE_API_KEY",
            "LLM_API_KEY",
            "ANTHROPIC_API_KEY",
            "GROQ_API_KEY",
            "LLM_BASE_URL",
            "LLM_OLLAMA_BASE_URL",
            "LLM_RATE_LIMIT_RPM",
        ],
        "Models": [
            "WTPSPLIT_MODEL",
            "OCR_BACKEND",
            "OCR_BASE_URL",
            "REF_SEG_STRATEGY",
            "REF_PARSE_STRATEGY",
            "PIPELINE_MEMORY_MODE",
        ],
        "External Services": [
            "CROSSREF_API_EMAIL",
            "CROSSREF_CONSOLIDATE",
        ],
        "API (optional)": [
            "REDIS_PASSWORD",
            "REDIS_URL",
        ],
        "General": [],  # catch-all
    }

    # Map keys to their section
    key_to_section: dict[str, str] = {}
    for section, keys in sections.items():
        for k in keys:
            key_to_section[k] = section

    # Group env_vars by section
    grouped: dict[str, list[tuple[str, str]]] = {s: [] for s in sections}
    for k, v in env_vars.items():
        sec = key_to_section.get(k, "General")
        grouped[sec].append((k, v))

    lines: list[str] = [
        "# bibr environment configuration",
        "# Generated by bibr setup",
        "",
    ]

    for section, pairs in grouped.items():
        if not pairs:
            continue
        lines.append(f"# --- {section} ---")
        for k, v in pairs:
            lines.append(f"{k}={_format_env_value(v)}")
        lines.append("")

    write_env_text(path, "\n".join(lines))


def _project_name(cwd: Path) -> str | None:
    """Return the current project's ``[project].name`` when a pyproject exists."""
    pyproject = cwd / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    project = data.get("project")
    if not isinstance(project, dict):
        return None
    name = project.get("name")
    return name if isinstance(name, str) else None


def _runs_in_project_env(cwd: Path) -> bool:
    """Whether this interpreter is the environment of the project at *cwd*.

    ``uv add`` edits the project uv finds from *cwd*; it only lands in the
    interpreter running bibr when that interpreter IS the project
    environment (``cwd/.venv`` or ``UV_PROJECT_ENVIRONMENT``). Otherwise the
    extras must go through ``uv pip install --python sys.executable``.
    """
    candidate = os.environ.get("UV_PROJECT_ENVIRONMENT", "")
    if not candidate:
        candidate = str(cwd / ".venv")
    try:
        return Path(sys.prefix).resolve() == Path(candidate).expanduser().resolve()
    except OSError:
        return False


def _extras_spec(extras: set[str]) -> str:
    return ",".join(sorted(extras))


def _install_command_for_extras(
    extras: set[str],
    *,
    cwd: Path,
    uv_bin: str | None,
) -> tuple[list[str], str]:
    """Pick the right install command for source vs installed-package users."""
    spec = f"bibr[{_extras_spec(extras)}]"
    project_name = _project_name(cwd)

    if project_name == "bibr":
        if not uv_bin:
            raise RuntimeError(
                "uv not found on PATH. Source checkouts install bibr extras with uv sync."
            )
        # ``--inexact``: keep packages the tester installed outside these extras
        # (e.g. a full ``--extra all`` env, or a hand-installed vllm-mlx). A bare
        # ``uv sync --extra=...`` is an EXACT sync and would remove them.
        return (
            [uv_bin, "sync", "--inexact", *(f"--extra={extra}" for extra in sorted(extras))],
            "Installing extras for this bibr source checkout",
        )

    if uv_bin and project_name and _runs_in_project_env(cwd):
        return (
            [uv_bin, "add", spec],
            "Adding bibr extras to this project dependency",
        )
    if uv_bin:
        # ``--python`` pins the install to the interpreter running bibr: uv
        # otherwise picks a venv from VIRTUAL_ENV or cwd, which may belong
        # to an unrelated project (or not exist at all).
        return (
            [uv_bin, "pip", "install", "--python", sys.executable, spec],
            "Installing bibr extras into the current uv environment",
        )
    return (
        [sys.executable, "-m", "pip", "install", spec],
        "Installing bibr extras into the current Python environment",
    )


def _command_label(cmd: list[str]) -> str:
    if len(cmd) >= 4 and cmd[1:3] == ["-m", "pip"]:
        return "python -m pip"
    executable = Path(cmd[0]).name
    return f"{executable} {cmd[1]}" if len(cmd) > 1 else executable


def _reload_settings_in_place() -> None:
    """Refresh the ``bibr.config.Settings`` singleton from the ``.env`` the
    wizard just wrote, WITHOUT changing its identity.

    Many modules on the ``chew()`` path already did ``from bibr.config import
    Settings`` at module level before this function runs (e.g.
    ``bibr.clients.llm``, ``bibr.local.ocr``) — their ``Settings`` name is
    bound to the *same object* ``bibr.config.Settings`` currently points at,
    not a copy. Rebinding ``bibr.config.Settings = GlobalSettings()`` would
    only update the module attribute; those already-imported modules would
    keep pointing at the old object and silently miss everything the wizard
    just configured. Copying every top-level section onto the *existing*
    singleton, instead, mutates the one object all of them share — exactly
    the same trick ``_test_local_server`` uses for a single field
    (``Settings.llm.local_model = model``), generalized here to every
    section since the smoke test can touch any of them.
    """
    from bibr.config import GlobalSettings, Settings

    fresh = GlobalSettings()
    for name in type(fresh).model_fields:
        setattr(Settings, name, getattr(fresh, name))


# ---------------------------------------------------------------------------
# Main wizard class
# ---------------------------------------------------------------------------


class SetupWizard:
    def __init__(self) -> None:
        self.console = Console()
        self.env_vars: dict[str, str] = {}
        self.selected_extras: set[str] = set()
        self._declined_extras: set[str] = set()
        self._extras_installed = False
        self._env_written = False
        self.env_path = Path.cwd() / ".env"

    # ---- public -----------------------------------------------------------

    def run(self) -> None:
        self._print_intro(advanced=False)

        # Detect the hardware ONCE and hand the values to the recommendation
        # builder so nothing probes nvidia-smi / sysctl twice.
        from bibr.local.pipeline import _get_system_memory_gb

        platform_key, accel_gb = detect_hardware()
        system_gb = _get_system_memory_gb()
        self._print_detection(platform_key, accel_gb)

        setup = _build_recommended_setup(
            platform_key=platform_key,
            accelerator_memory_gb=accel_gb,
            system_memory_gb=system_gb,
        )

        if setup.tier == "fully_local":
            self._print_recommendation(setup)
            if not Confirm.ask("Use this configuration?", default=True):
                self._handle_decline()
                return
            self._finish_recommended_setup(setup)
            return

        # Weak-hardware fork: local OCR/LLM models are not viable here.
        self.console.print(
            "\nLocal OCR and LLM models aren't a good fit for this machine's hardware."
        )
        if Confirm.ask("Do you have a private OCR/LLM server bibr can use?", default=False):
            self._print_recommendation(setup)
            self._finish_recommended_setup(setup)
            return

        # No local, no private server: the only remaining path is the cloud.
        self.console.print(
            "\nWithout a local or private server, bibr can process papers with Google "
            "Gemini in the cloud. This sends document-derived content (page images and "
            "extracted text) to Google."
        )
        if not Confirm.ask("Send document-derived content to Google Gemini?", default=True):
            self._handle_decline()
            return

        cloud_setup = _build_recommended_setup(
            platform_key=platform_key,
            accelerator_memory_gb=accel_gb,
            system_memory_gb=system_gb,
            allow_cloud=True,
        )
        self._print_recommendation(cloud_setup)
        self._finish_recommended_setup(cloud_setup)

    def _print_detection(self, platform_key: str | None, accel_gb: float | None) -> None:
        if platform_key == "mlx":
            mem = f"{accel_gb:.0f} GB unified memory" if accel_gb is not None else "unified memory"
            line = f"Detected: Apple Silicon, {mem}"
        elif platform_key == "cuda":
            mem = f"{accel_gb:.0f} GB VRAM" if accel_gb is not None else "VRAM"
            line = f"Detected: NVIDIA GPU, {mem}"
        else:
            line = (
                "Detected: no NVIDIA GPU or Apple Silicon — local OCR/LLM models are "
                "not recommended on this machine."
            )
        self.console.print(f"[dim]{line}[/dim]")

    def _handle_decline(self) -> None:
        """Declining a recommendation is a handoff to --advanced, not a dead end."""
        if Confirm.ask("Switch to the advanced wizard for exact control now?", default=True):
            self.run_advanced()
            return
        self.console.print(
            "[dim]No problem. Run [cyan]bibr setup --advanced[/cyan] when you're ready.[/dim]"
        )

    def _finish_recommended_setup(self, setup: RecommendedSetup) -> None:
        """Collect prompts, persist config, install, validate — in that order.

        Config is written to ``.env`` BEFORE the extras install so a failing
        install never discards the URLs/keys the user just typed.
        """
        self.env_vars.update(setup.env)
        self.selected_extras.update(setup.extras)

        self._collect_required_prompts(setup)
        self._step_write_env(save_preset_name=setup.preset_name, header="Save configuration")
        self._offer_extras_install()
        self._run_validation(setup)
        if "smoke_test" in setup.required_prompts:
            self._step_smoke_test(header="Test extraction", confirm_default=True)
        self._print_done()

    def _collect_required_prompts(self, setup: RecommendedSetup) -> None:
        """Gather the credentials/URLs a tier needs, without installing anything."""
        if "private_server_url" in setup.required_prompts:
            self.env_vars["OCR_BASE_URL"] = Prompt.ask(
                "Private OCR server URL",
                default="http://localhost:8080",
            )
            llm_url = Prompt.ask(
                "Private OpenAI-compatible LLM base URL",
                default="http://localhost:8000/v1",
            )
            self.env_vars["LLM_PROVIDER"] = "openai"
            self.env_vars["LLM_BASE_URL"] = llm_url
            self.env_vars["LLM_API_KEY"] = Prompt.ask(
                "Private LLM API key (press Enter to keep the 'local' placeholder — "
                "for servers that require no key)",
                password=True,
                default="local",
            )
            self.env_vars["LLM_MODEL"] = Prompt.ask(
                "Private LLM model name",
                default="nuextract",
            )

        if "cloud_api_key" in setup.required_prompts:
            self.env_vars["GOOGLE_API_KEY"] = Prompt.ask("Google API key", password=True)

    def _offer_extras_install(self) -> None:
        if not self.selected_extras:
            return
        if not Confirm.ask(
            f"Install recommended extras now ({', '.join(sorted(self.selected_extras))})?",
            default=True,
        ):
            self.console.print(
                "[dim]Skipped — install later with the command shown in `bibr doctor`.[/dim]"
            )
            return
        self._install_selected_extras("Recommended setup", config_saved=True)

    def _run_validation(self, setup: RecommendedSetup) -> None:
        """Offer the tier-appropriate connectivity check after config is saved."""
        backend = self.env_vars.get("LLM_BACKEND")
        if backend in ("vllm", "vllm-mlx", "rapid-mlx", "llama-cpp"):
            self._offer_local_server_test()
        elif setup.tier in ("cloud_fallback", "private_server"):
            # The private-server URL, key and model were just collected; a
            # wrong one otherwise passes setup silently and fails the first chew.
            self._offer_llm_connection_test()

    def run_advanced(self) -> None:
        self._print_intro(advanced=True)

        self._step_extras()
        self._step_llm_provider()
        self._step_test_connection()
        self._step_external_services()
        self._step_memory_mode()
        self._step_write_env()
        self._step_smoke_test()

        self._print_done()

    # ---- steps ------------------------------------------------------------

    def _print_intro(self, advanced: bool = False) -> None:
        subtitle = (
            "full control over providers, backends, and your .env file."
            if advanced
            else "recommended, private-first setup — exact control lives in --advanced."
        )
        ui.brand_header(self.console, "bibr setup", subtitle=subtitle)
        self.console.print("[dim]press Ctrl+C at any time to quit without saving.[/dim]")
        self.console.print()
        self.console.print(
            "[yellow]bibr is experimental, and some papers or machines may need "
            "a little tuning.[/yellow]"
        )
        self.console.print(
            "[dim]Setup will prefer private/local processing when possible, but fully "
            "local runs can be slow and may download several GB.[/dim]"
        )
        self.console.print()

    # Lowercase, human tier names for the plan preview (cloud_fallback -> "cloud").
    _TIER_LABELS = {
        "fully_local": "fully local",
        "mostly_local": "mostly local",
        "private_server": "private server",
        "cloud_fallback": "cloud",
    }

    def _print_recommendation(self, setup: RecommendedSetup) -> None:
        """Print the tier header plus a concrete, itemised plan preview.

        Rows are derived from ``setup.env`` (+ the model registry) so the user
        confirms an explicit plan rather than an opaque label.
        """
        tier_label = self._TIER_LABELS.get(setup.tier, setup.tier.replace("_", " "))
        self.console.print(f"\n[bold]Recommended for this machine:[/bold] {tier_label}")
        for label, value in self._plan_rows(setup):
            self.console.print(ui.kv(label, value))

    def _plan_rows(self, setup: RecommendedSetup) -> list[tuple[str, str]]:
        extras = ", ".join(sorted(setup.extras)) or "(none)"
        return [
            ("OCR", self._plan_ocr_value(setup)),
            ("LLM", self._plan_llm_value(setup)),
            ("Extras", extras),
            ("Memory", self._plan_memory_value(setup)),
            ("Privacy", setup.privacy_summary),
            ("First run", setup.download_summary),
            ("Speed", setup.runtime_summary),
        ]

    def _plan_ocr_value(self, setup: RecommendedSetup) -> str:
        backend = setup.env.get("OCR_BACKEND", "")
        if setup.tier == "private_server":
            return f"your configured OCR server ({backend})"
        if setup.tier == "cloud_fallback":
            return f"{backend} (Google Gemini vision, cloud)"
        if backend == "paddle":
            return "paddle (automatic Paddle-first chain; GLM fallback)"
        if backend == "glm-llama":
            return "glm-llama (GLM-OCR via llama.cpp, local)"
        return f"{backend} (local OCR)"

    def _plan_llm_value(self, setup: RecommendedSetup) -> str:
        if setup.tier == "private_server":
            return "your OpenAI-compatible server"
        if setup.tier == "cloud_fallback":
            provider = setup.env.get("LLM_PROVIDER", "")
            model = setup.env.get("LLM_MODEL", "")
            return f"{provider} / {model} (cloud)"
        backend = setup.env.get("LLM_BACKEND", "")
        hf_id = setup.env.get("LLM_LOCAL_MODEL", "")
        name = _registry_short_label(hf_id) or "local model"
        return f"{name} — {hf_id} via {backend}, local"

    def _plan_memory_value(self, setup: RecommendedSetup) -> str:
        mode = setup.env.get("PIPELINE_MEMORY_MODE", "")
        if mode == "aggressive":
            return "aggressive — models load one at a time to fit limited memory"
        return mode or "balanced"

    def _demo_available(self) -> bool:
        """Whether ``bibr demo`` can run: demo extra actually installed this run
        (selecting it and then declining the install doesn't count), or gradio
        already importable."""
        if "demo" in self.selected_extras and self._extras_installed:
            return True
        return importlib.util.find_spec("gradio") is not None

    def _print_done(self) -> None:
        self.console.print()
        ui.ok(self.console, "[bold]Setup complete![/bold]")
        lines = ["\n[dim]You can now run:[/dim]"]
        if self._demo_available():
            lines.append("  [cyan]bibr demo[/cyan]               — open the interactive demo")
        else:
            lines.append(
                "  [dim]bibr demo needs the demo extra — install it with[/dim] "
                "[cyan]uv sync --extra demo[/cyan] [dim](or pip install 'bibr\\[demo]')[/dim]"
            )
        lines.append("  [cyan]bibr chew paper.pdf[/cyan]     — process a paper directly")
        lines.append("  [cyan]bibr doctor[/cyan]             — validate your setup")
        lines.append("  [cyan]bibr -h[/cyan]                 — see all available commands")
        self.console.print("\n".join(lines))

    def _step_extras(self) -> None:
        ui.step(self.console, 1, 6, "Optional extras")
        self.console.print("These add functionality — skip any you don't need.\n")

        for key, desc in _available_extras().items():
            if Confirm.ask(f"  [cyan]{key}[/cyan] — {desc}", default=False):
                self.selected_extras.add(key)
            else:
                # Remember the decline: step 4 must not install it anyway.
                self._declined_extras.add(key)

        if self.selected_extras:
            self._install_selected_extras()
        else:
            self.console.print("[dim]No extras selected.[/dim]")

    def _install_selected_extras(self, reason: str = "", *, config_saved: bool = False) -> None:
        if not self.selected_extras:
            return

        names = ", ".join(sorted(self.selected_extras))
        if reason:
            self.console.print(f"\n[dim]{reason}.[/dim]")
        self.console.print(f"\nInstalling extras: [cyan]{names}[/cyan]")
        uv_bin = shutil.which("uv")
        try:
            cmd, install_label = _install_command_for_extras(
                self.selected_extras,
                cwd=Path.cwd(),
                uv_bin=uv_bin,
            )
        except RuntimeError as exc:
            self.console.print(
                f"[red]{exc}[/red]\n"
                "  Install uv from: [link=https://docs.astral.sh/uv/]"
                "https://docs.astral.sh/uv/[/link]"
            )
            if config_saved:
                self.console.print(
                    "[dim]Your configuration was already saved to .env — rerun the "
                    "install manually once uv is available.[/dim]"
                )
            raise SystemExit(1) from None
        cmd_label = _command_label(cmd)
        from rich.markup import escape

        self.console.print(f"[dim]{install_label}:[/dim] [cyan]{escape(shlex.join(cmd))}[/cyan]")
        with self.console.status(f"Running {cmd_label} …"):
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
            )
        if result.returncode == 0:
            self._extras_installed = True
            ui.ok(self.console, "Extras installed")
        else:
            detail = result.stderr.strip() or result.stdout.strip() or f"{cmd_label} failed"
            from rich.markup import escape

            self.console.print(f"[red]{cmd_label} failed:[/red]\n  [dim]{escape(detail)}[/dim]")
            if config_saved:
                self.console.print(
                    "[dim]Your configuration was already saved to .env before this "
                    "install — rerun the install manually, then `bibr doctor`.[/dim]"
                )
            raise SystemExit(result.returncode)

        if "gpu" in self.selected_extras:
            self._reinstall_onnxruntime_gpu(uv_bin)

    def _reinstall_onnxruntime_gpu(self, uv_bin: str | None) -> None:
        """Make the GPU build of onnxruntime the one that loads.

        The core ``onnxruntime`` dependency and the ``gpu`` extra's
        ``onnxruntime-gpu`` write the same ``onnxruntime/`` directory. The
        extras install just wrote both wheels at once, and which one's files
        landed last is a race. Reinstalling the ``onnxruntime-gpu`` version that
        install chose rewrites every shared file from the GPU wheel.
        ``onnxruntime`` stays installed: ``uv run`` reinstalls a missing one,
        and its files would replace the GPU build's.
        """
        from rich.markup import escape

        importlib.invalidate_caches()  # the install above ran in another process
        try:
            version = importlib.metadata.version("onnxruntime-gpu")
        except importlib.metadata.PackageNotFoundError:
            ui.warn(
                self.console,
                "onnxruntime-gpu is not installed, so ONNX models will run on CPU",
            )
            return
        cmd = onnxruntime_gpu_reinstall_command(version, uv=uv_bin)
        with self.console.status(f"Reinstalling onnxruntime-gpu {version} over the CPU build …"):
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
            )
            # A fresh interpreter: this one may hold a build it imported earlier.
            probe = (
                subprocess.run(  # noqa: S603
                    [sys.executable, "-c", _GPU_BUILD_PROBE],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0
                else None
            )
        if probe is not None and probe.stdout.strip() == "True":
            ui.ok(self.console, f"onnxruntime-gpu {version} is the onnxruntime build that loads")
            return
        if probe is None:
            reason = (
                result.stderr.strip() or result.stdout.strip() or f"exit status {result.returncode}"
            )
        else:
            lines = probe.stderr.strip().splitlines()
            reason = lines[-1] if lines else "The CPU build still loads."
        ui.warn(
            self.console,
            f"onnxruntime-gpu {version} is not the onnxruntime build that loads, "
            "so ONNX models will run on CPU",
            hint=escape(f"{reason}\nRun: {shlex.join(cmd)}"),
        )

    def _step_llm_provider(self) -> None:
        ui.step(self.console, 2, 6, "LLM provider")
        self.console.print(
            "[dim]tip: bibr's extraction tasks don't need a frontier model — the default\n"
            "gemini-3.5-flash-lite is fast, cheap, and accurate. self-hosting?\n"
            "modern small open-source models (e.g. gemma 4 12b, qwen 3.6) generally\n"
            "work well in early testing.[/dim]"
        )
        provider = Prompt.ask(
            "Provider",
            choices=["google", "openai", "anthropic", "groq", "ollama", "local"],
            default="google",
        )
        if provider == "local":
            self._step_llm_local()
            return
        self.env_vars["LLM_PROVIDER"] = provider
        defaults = LLM_DEFAULTS[provider]

        # --- Collect credentials first (needed for model listing) ---
        api_key = ""
        base_url = ""

        if provider in ("google", "openai", "anthropic", "groq"):
            api_key = Prompt.ask(f"API key for {provider}", password=True)
            self.env_vars[defaults["key_env"]] = api_key

        if provider == "openai":
            base_url = self._ask_llm_base_url(api_key)
            if base_url:
                self.env_vars["LLM_BASE_URL"] = base_url

        if provider == "ollama":
            base_url = Prompt.ask("Ollama base URL", default="http://localhost:11434")
            self.env_vars["LLM_OLLAMA_BASE_URL"] = base_url
            self.env_vars["LLM_RATE_LIMIT_RPM"] = "10"

        # --- Fetch and select model ---
        model = None
        with self.console.status("Fetching available models …"):
            models = _fetch_models(
                provider,
                api_key,
                base_url,
                allow_insecure_http=self._allows_insecure_llm_http(),
            )

        if models:
            model = _select_model(models, defaults["model"], self.console)

        if model is None:
            if not models:
                self.console.print(
                    "[yellow]Couldn't fetch available models — you can type one manually.[/yellow]"
                )
            model = Prompt.ask("Model name", default=defaults["model"])

        self.env_vars["LLM_MODEL"] = model

    def _allows_insecure_llm_http(self) -> bool:
        value = self.env_vars.get("LLM_ALLOW_INSECURE_HTTP") or os.environ.get(
            "LLM_ALLOW_INSECURE_HTTP", ""
        )
        return value.strip().lower() in ("1", "true", "yes", "on")

    def _ask_llm_base_url(self, api_key: str) -> str:
        """Ask for ``LLM_BASE_URL`` before the key is first sent to it.

        Listing models and the connection test both send the key, so a public
        ``http://`` URL is refused here, as the pipeline would refuse it, unless
        the user opts in; the opt-in is saved as ``LLM_ALLOW_INSECURE_HTTP``.
        """
        while True:
            base_url = Prompt.ask("Custom base URL (leave blank for OpenAI default)", default="")
            try:
                refuse_plaintext_llm_key(
                    base_url, api_key, allow_insecure_http=self._allows_insecure_llm_http()
                )
            except ValueError as exc:
                from rich.markup import escape

                ui.error(self.console, escape(str(exc)))
                if Confirm.ask("Send the key over plain HTTP anyway?", default=False):
                    self.env_vars["LLM_ALLOW_INSECURE_HTTP"] = "true"
                    return base_url
                continue
            return base_url

    def _step_llm_local(self) -> None:
        """Managed local LLM: pick a curated model/quant for this machine."""
        platform_key, memory_gb = detect_hardware()
        if platform_key is None:
            self.console.print(
                "[yellow]No NVIDIA GPU or Apple Silicon detected — a local LLM "
                "may be slow or unsupported. Showing all options.[/yellow]"
            )
            platform_key = "cuda"
        elif memory_gb:
            kind = "VRAM" if platform_key == "cuda" else "unified memory"
            self.console.print(f"[dim]Detected {platform_key}: {memory_gb:.0f} GB {kind}[/dim]")

        backend = _default_local_llm_backend(platform_key, memory_gb, sys.platform)
        if backend == "vllm-mlx":
            self.console.print(
                "[yellow]Note: local LLM inference on Apple Silicon (vllm-mlx) is slow — "
                "single-digit tokens/sec is typical, and a full paper extraction can take "
                "15-30+ minutes even with the validated 0.4.x BatchedEngine path. "
                "This is a known limitation, not a hang. For throughput, use an external "
                "hosted model or a cloud API and keep OCR local.[/yellow]"
            )

        # Models with at least one variant for this platform, plus escape hatch.
        keys = [m.key for m in REGISTRY if _setup_variants_for(m, platform_key, None, backend)]
        self.console.print("\n[dim]Available models:[/dim]")
        for m in REGISTRY:
            if m.key not in keys:
                continue
            tag = (
                " [yellow]\\[experimental][/yellow]"
                if platform_key in m.experimental_platforms
                else ""
            )
            self.console.print(f"  [bold]{m.key}[/bold] — {m.label}{tag}")
        self.console.print("  [bold]custom[/bold] — any HF model id (you own quant/flags)")

        choice = Prompt.ask(
            "Model",
            choices=[*keys, "custom"],
            default=keys[0] if keys else "custom",
        )

        if choice == "custom":
            hf_id = Prompt.ask("HF model id (org/name)")
            self.env_vars["LLM_BACKEND"] = backend
            self.env_vars["LLM_LOCAL_MODEL"] = hf_id
            return

        model = get_model(choice)
        fitting = variants_for(model, platform_key, memory_gb)
        shown = _setup_variants_for(model, platform_key, memory_gb, backend)
        if not fitting and shown:
            self.console.print(
                "[yellow]No variant fits the detected memory — showing all; "
                "the server may fail to load.[/yellow]"
            )
        self.console.print("\n[dim]Variants (best quality first):[/dim]")
        for i, v in enumerate(shown, 1):
            runtime = _runtime_for_variant(v, backend)
            self.console.print(
                f"  {i}. {runtime} / {v.quant} — needs ~{v.min_vram_gb:.0f} GB, "
                f"downloads ~{v.approx_download_gb:.1f} GB ({v.hf_id})"
            )
        idx = Prompt.ask(
            "Variant",
            choices=[str(i) for i in range(1, len(shown) + 1)],
            default="1",
        )
        variant = shown[int(idx) - 1]

        self.env_vars["LLM_BACKEND"] = _runtime_for_variant(variant, backend)
        self.env_vars["LLM_LOCAL_MODEL"] = variant.hf_id
        for k, v in model.env.items():
            self.env_vars[k] = v
        if platform_key in model.experimental_platforms:
            self.console.print(
                "[yellow]Note: this platform is experimental for this model — "
                "extraction quality is not yet eval-gated.[/yellow]"
            )

    def _test_local_server(self) -> None:
        """Launch the managed server once, confirm it comes up, shut it down."""
        backend = self.env_vars["LLM_BACKEND"]
        model = self.env_vars["LLM_LOCAL_MODEL"]
        os.environ["LLM_LOCAL_MODEL"] = model  # constructor reads Settings.llm.local_model
        from bibr.config import Settings

        Settings.llm.local_model = model
        try:
            with self.console.status(f"Starting local server ({model}) …"):
                if backend == "vllm":
                    from bibr.local.vllm_llm import VllmLlmServer

                    server = VllmLlmServer()
                elif backend == "vllm-mlx":
                    from bibr.local.llm import VllmMlxLlmServer

                    server = VllmMlxLlmServer()
                elif backend == "rapid-mlx":
                    from bibr.local.rapid_mlx import RapidMlxLlmServer

                    server = RapidMlxLlmServer()
                else:
                    from bibr.local.llama_cpp import LlamaCppLlmServer

                    server = LlamaCppLlmServer()
            ui.ok(self.console, f"Server healthy at {server.base_url}")
            server.shutdown()
        except Exception as e:  # noqa: BLE001
            from rich.markup import escape

            ui.error(self.console, f"Local server test failed: {escape(str(e))}")
            self.console.print(
                "[dim]Config kept — fix and retry with `bibr chew --llm local`.[/dim]"
            )

    def _step_test_connection(self) -> None:
        ui.step(self.console, 3, 6, "Test connection")

        if self.env_vars.get("LLM_BACKEND") in ("vllm", "vllm-mlx", "rapid-mlx", "llama-cpp"):
            self._offer_local_server_test()
            return

        self._offer_llm_connection_test()

    def _offer_local_server_test(self) -> None:
        """Optionally launch the managed local LLM server once to health-check it."""
        self.console.print(
            "[dim]First launch downloads the model (see size above) and can "
            "take several minutes.[/dim]"
        )
        if not Confirm.ask("Launch the local server once to test it now?", default=False):
            self.console.print(
                "[dim]Skipped — `bibr chew --llm local` starts it automatically.[/dim]"
            )
            return
        self._test_local_server()

    def _offer_llm_connection_test(self) -> None:
        """Optionally send one round-trip to the configured cloud LLM, with retry."""
        if not Confirm.ask("Test the LLM connection now?", default=True):
            self.console.print("[dim]Skipped.[/dim]")
            return

        from rich.markup import escape

        from bibr.clients.llm import ping_llm
        from bibr.exceptions import ConfigurationError

        provider = self.env_vars.get("LLM_PROVIDER", "")
        key_env = LLM_DEFAULTS[provider]["key_env"] if provider in LLM_DEFAULTS else ""
        api_key = self.env_vars.get(key_env, "") if key_env else ""
        retried_key = False

        while True:
            try:
                with self.console.status("Connecting to LLM …"):
                    # The provider adapter extraction uses, so this sends what
                    # the first ``bibr chew`` will send.
                    reply = ping_llm(_connection_test_settings(self.env_vars))
                ui.ok(self.console, f"Connected — LLM responded: {escape(reply.strip())}")
                if retried_key and key_env and self._env_written:
                    # The key first saved to .env was rejected; persist the
                    # corrected one so .env matches the key just tested.
                    try:
                        _merge_env(self.env_path, {key_env: api_key})
                    except OSError as exc:
                        self.console.print(
                            f"[dim]Couldn't save the corrected key to "
                            f"{self.env_path}: {escape(str(exc))}[/dim]"
                        )
                break
            except ConfigurationError as exc:
                # An invalid value already in .env, ~/.bibr/.env or the
                # environment, not in the answers: chew stops on it too.
                ui.error(self.console, f"Can't test the connection: {escape(str(exc))}")
                self.console.print(
                    "[dim]That value comes from your existing configuration, not from "
                    "this setup. `bibr chew` stops on it too until it is fixed or "
                    "overwritten; run `bibr doctor` after saving to check again.[/dim]"
                )
                break
            except Exception as exc:
                msg = _redact(str(exc), api_key)
                ui.error(self.console, f"Couldn't connect: {escape(msg)}")
                if provider == "ollama":
                    # Ollama takes no key; what the user can change is the URL.
                    if not Confirm.ask("Retry with a different Ollama base URL?", default=True):
                        self.console.print("[dim]Skipping connection test.[/dim]")
                        break
                    self.env_vars["LLM_OLLAMA_BASE_URL"] = Prompt.ask(
                        "Ollama base URL",
                        default=self.env_vars.get("LLM_OLLAMA_BASE_URL", "http://localhost:11434"),
                    )
                    continue
                if not key_env or not Confirm.ask("Retry with a different API key?", default=True):
                    self.console.print("[dim]Skipping connection test.[/dim]")
                    break
                api_key = Prompt.ask("API key", password=True)
                self.env_vars[key_env] = api_key
                retried_key = True

    def _step_external_services(self) -> None:
        ui.step(self.console, 4, 6, "Models & external services")

        email = Prompt.ask(
            "Crossref API email (for polite access, optional)",
            default="",
        )
        if email:
            self.env_vars["CROSSREF_API_EMAIL"] = email

        self.console.print(
            "\n[dim]Reference enrichment: look each extracted reference up in\n"
            "Crossref (and the optional bibr-resolver) to attach DOIs and a\n"
            "bib_match table. Off by default — it adds network round-trips per\n"
            "paper. Per-run switches: `bibr chew --crossref` / `--no-crossref`.[/dim]"
        )
        if Confirm.ask("  Enable Crossref reference enrichment?", default=False):
            self.env_vars["CROSSREF_ENRICH"] = "true"
            self.console.print(
                "\n[dim]Reference consolidation: fill fields missing from extracted\n"
                "references (DOI, pages, ...) with matched Crossref data. Printed\n"
                "values are never overwritten ('replace' mode exists via\n"
                "CROSSREF_CONSOLIDATE in .env).[/dim]"
            )
            if Confirm.ask("  Enable reference consolidation?", default=False):
                self.env_vars["CROSSREF_CONSOLIDATE"] = "fill"

        if "cache" in self.selected_extras:
            self.console.print("\n[dim]You selected the cache extra — additional config:[/dim]")
            redis_pw = Prompt.ask("Redis password", password=True, default="")
            if redis_pw:
                self.env_vars["REDIS_PASSWORD"] = redis_pw

        # Sentence segmentation model
        self.console.print("\n[dim]Sentence segmentation model (wtpsplit):[/dim]")
        for name, desc in WTPSPLIT_MODELS.items():
            self.console.print(f"  [cyan]{name}[/cyan] — {desc}")
        model = Prompt.ask(
            "Model",
            choices=list(WTPSPLIT_MODELS.keys()),
            default="sat-6l-sm",
        )
        if model != "sat-6l-sm":
            self.env_vars["WTPSPLIT_MODEL"] = model

        # OCR backend selection
        default_ocr = "paddle"

        self.console.print("\n[dim]OCR backend:[/dim]")
        self.console.print(
            "  [cyan]paddle[/cyan]          — automatic Paddle-first local chain (recommended)\n"
            "  [cyan]paddle-vllm[/cyan]     — managed Paddle vLLM server\n"
            "  [cyan]paddle-rapid-mlx[/cyan] — managed Paddle Rapid-MLX (requires OCR smoke)\n"
            "  [cyan]paddle-mlx-vlm[/cyan]  — managed Paddle MLX-VLM fallback\n"
            "  [cyan]paddle-http[/cyan]     — external Paddle OCR server\n"
            "  [cyan]glm-llama[/cyan]      — llama.cpp (native Windows / low VRAM)\n"
            "  [cyan]glm-rapid-mlx[/cyan]  — Rapid-MLX (recommended for Apple Silicon)\n"
            "  [cyan]glm-http[/cyan]       — External GLM-OCR server (vLLM, Ollama, etc.)\n"
            "  [cyan]gemini[/cyan]         — Google Gemini vision (cloud, no local GPU needed)\n"
            "  [cyan]openai[/cyan]         — OpenAI vision (cloud)\n"
            "  [cyan]anthropic[/cyan]      — Anthropic vision (cloud)"
        )
        ocr_backend = Prompt.ask(
            "OCR backend",
            choices=[
                "paddle",
                "paddle-vllm",
                "paddle-rapid-mlx",
                "paddle-mlx-vlm",
                "paddle-http",
                "glm-llama",
                "glm-rapid-mlx",
                "glm-http",
                "gemini",
                "openai",
                "anthropic",
            ],
            default=default_ocr,
        )
        self.env_vars["OCR_BACKEND"] = ocr_backend

        # Reference parsing strategy (mirrors the ``--refs`` chew flag)
        self.console.print("\n[dim]Reference parsing (bibliography → structured fields):[/dim]")
        self.console.print(
            "  [cyan]ner[/cyan] — local ModernBERT-CRF parser (default): cuts most of\n"
            "        bibr's LLM token use (references are the bulk of it); reference\n"
            "        field precision is lower (requires the ml extra)\n"
            "  [cyan]llm[/cyan] — batched LLM parsing: full reference precision, at the\n"
            "        cost of more tokens"
        )
        refs = Prompt.ask(
            "Reference parsing",
            choices=["ner", "llm"],
            default="ner",
        )
        if refs != "ner":
            self.env_vars["REF_PARSE_STRATEGY"] = refs

        # `bibr setup` installs the full local stack: the ONNX runtime in core
        # serves these models too, but the ml (torch) extra is what gives this
        # machine the GPU/MPS path and the CRF segmenter the wizard's defaults
        # can reach.
        if "ml" not in self.selected_extras and not _ml_extra_available():
            if "ml" in self._declined_extras:
                # The step-1 answer stands: a hint only, no install, no re-ask.
                from rich.markup import escape

                try:
                    later_cmd, _label = _install_command_for_extras(
                        {"ml"}, cwd=Path.cwd(), uv_bin=shutil.which("uv")
                    )
                    later = f" Add it later with: {escape(shlex.join(later_cmd))}"
                except RuntimeError:
                    later = ""
                self.console.print(
                    "[dim]Skipped the ml extra (declined in step 1): layout "
                    "detection and reference parsing run on the ONNX runtime "
                    f"instead of torch.{later}[/dim]"
                )
            else:
                self.selected_extras.add("ml")
                reason = "PDF layout detection runs fastest with the ml extra"
                if refs == "ner":
                    reason = (
                        "PDF layout detection and NER reference parsing run fastest "
                        "with the ml extra"
                    )
                self._install_selected_extras(reason)

    def _step_memory_mode(self) -> None:
        """Pin the auto-detected memory mode into .env so it is visible/editable.

        The runtime already auto-detects this each chew, but persisting it here
        makes the choice explicit for a first-time tester — and lets low-memory
        machines (a 6 GB GPU, or an 8 GB Apple Silicon Mac running fully local)
        see why they are in ``aggressive`` mode.
        """
        from bibr.local.pipeline import _auto_memory_mode, _get_system_memory_gb

        platform_key, accel_gb = detect_hardware()
        mode = _auto_memory_mode(platform_key, accel_gb, _get_system_memory_gb())
        self.env_vars["PIPELINE_MEMORY_MODE"] = mode
        if mode == "aggressive":
            self.console.print(
                f"\n[dim]Memory mode: [cyan]{mode}[/cyan] — models are loaded one at a "
                "time to fit limited memory. Edit PIPELINE_MEMORY_MODE in .env to "
                "override.[/dim]"
            )
        else:
            self.console.print(
                f"\n[dim]Memory mode: [cyan]{mode}[/cyan] (set PIPELINE_MEMORY_MODE in "
                ".env to override).[/dim]"
            )

    def _step_write_env(
        self,
        save_preset_name: str | None = None,
        header: str | None = None,
    ) -> None:
        ui.phase(self.console, header or ui.step_label(5, 6, "Save configuration"))

        if not self.env_vars:
            self.console.print("[dim]Nothing to save — no settings were collected.[/dim]")
            return

        if self.env_path.exists():
            action = Prompt.ask(
                f"{self.env_path} already exists — what should I do?",
                choices=["overwrite", "merge", "skip"],
                default="merge",
            )
            if action == "skip":
                self.console.print("[dim]Skipped — .env unchanged[/dim]")
                return
            if action == "merge":
                _merge_env(self.env_path, _with_llm_routing(self.env_vars))
                self._env_written = True
                ui.ok(self.console, f"Merged new settings into {self.env_path}")
                if save_preset_name:
                    self._save_preset(save_preset_name)
                else:
                    self._offer_save_preset()
                return

        _write_env_fresh(self.env_path, _with_llm_routing(self.env_vars))
        self._env_written = True
        ui.ok(self.console, f"Wrote {self.env_path}")
        if save_preset_name:
            self._save_preset(save_preset_name)
        else:
            self._offer_save_preset()

    def _offer_save_preset(self) -> None:
        if not self.env_vars:
            return
        if not Confirm.ask("\nSave this configuration as a named preset?", default=False):
            return

        name = Prompt.ask(
            "Preset name (alphanumeric, ., -, _)",
        )
        self._save_preset(name)

    def _save_preset(self, name: str) -> None:
        try:
            from bibr.presets import is_secret_key

            # Strip secrets — presets are intended to be shareable, so API keys
            # / tokens / passwords stay only in .env.
            shareable = {k: v for k, v in self.env_vars.items() if not is_secret_key(k)}
            manager = PresetManager()
            manager.save(name, shareable)
            ui.ok(
                self.console,
                f"Saved preset [cyan]{name}[/cyan] "
                f"({len(shareable)} settings; secrets stay in .env). "
                f"Switch with [cyan]bibr preset use {name}[/cyan]",
            )
        except Exception as exc:
            from rich.markup import escape

            self.console.print(f"[yellow]! Couldn't save preset:[/yellow] {escape(str(exc))}")

    def _smoke_test_note(self) -> str:
        """Honest first-run download note for the configured OCR backend."""
        from bibr.ocr.registry import resolve_backend_name

        backend = self.env_vars.get("OCR_BACKEND") or None
        resolved = resolve_backend_name(backend)
        note = _OCR_BACKEND_DOWNLOAD_NOTES.get(resolved, "may download model weights on first run")
        return (
            "[dim]Uses page 1 of the sample paper shipped with bibr, skips "
            f"LLM/reference extraction for speed, and {note}. Layout and sentence "
            "models may also download on first run.[/dim]"
        )

    def _ocr_unreachable_hint(self) -> str:
        """Hint text for an unreachable/failed OCR backend."""
        from bibr.ocr.registry import resolve_backend_name

        backend = self.env_vars.get("OCR_BACKEND") or None
        resolved = resolve_backend_name(backend)
        launch_cmd = _OCR_HTTP_LAUNCH_HINTS.get(resolved)
        if launch_cmd:
            return (
                f"Start the external OCR server, e.g.:\n    {launch_cmd}\n"
                "  then confirm OCR_BASE_URL points at it."
            )
        return (
            f"bibr manages the '{resolved}' OCR server automatically — "
            "run `bibr doctor` to check the launcher/dependency is installed."
        )

    def _llm_key_env_hint(self) -> str | None:
        """Env var to check for an auth failure with the configured LLM provider."""
        provider = self.env_vars.get("LLM_PROVIDER")
        if not provider:
            return None
        return LLM_DEFAULTS.get(provider, {}).get("key_env") or None

    def _smoke_failure_hint(self, exc: Exception) -> str:
        """Map a chew() failure to an actionable one-line hint.

        Always ends with a pointer to ``bibr doctor`` — the wizard's config is
        already written to disk by this point, so every branch here just
        informs, it never re-raises.
        """
        from rich.markup import escape

        from bibr.exceptions import UpstreamServiceError

        doctor_line = "[dim]Run `bibr doctor` for a full check.[/dim]"
        key_env = self._llm_key_env_hint()
        api_key = self.env_vars.get(key_env, "") if key_env else ""

        if isinstance(exc, ImportError):
            # ml_import_error() (bibr/utils/ml_extra.py) already bakes the
            # exact `uv sync --extra ...` line into the message.
            return f"[dim]{escape(_redact(str(exc), api_key))}[/dim]\n{doctor_line}"

        if _caused_by_configuration_error(exc):
            # _step_smoke_test already printed the message in full; say once
            # what to do about it instead of labelling it unexpected. chew()
            # reports a bad setting wrapped (a ProcessingError whose cause is
            # the ConfigurationError), so walk the chain, not just the top.
            return (
                "[dim]Fix the configuration value above in your .env, then "
                f"rerun the test.[/dim]\n{doctor_line}"
            )

        if isinstance(exc, UpstreamServiceError):
            service = (exc.service_name or "").lower()
            if service == "ocr":
                return f"[dim]{self._ocr_unreachable_hint()}[/dim]\n{doctor_line}"
            if service == "llm":
                underlying = exc.original_error if exc.original_error is not None else exc
                if _looks_like_auth_error(underlying):
                    if key_env:
                        return (
                            f"[dim]The LLM provider rejected the request — check "
                            f"{key_env} in your .env.[/dim]\n{doctor_line}"
                        )
                    return (
                        "[dim]The LLM provider rejected the request — check your "
                        f"local LLM server logs.[/dim]\n{doctor_line}"
                    )
            return f"[dim]{escape(_redact(str(exc), api_key))}[/dim]\n{doctor_line}"

        return (
            f"[dim]Unexpected error — your configuration is already saved: "
            f"{escape(_redact(str(exc), api_key))}[/dim]\n{doctor_line}"
        )

    def _step_smoke_test(
        self,
        header: str | None = None,
        confirm_default: bool = False,
    ) -> None:
        from rich.markup import escape

        ui.phase(self.console, header or ui.step_label(6, 6, "Test extraction"))
        self.console.print(
            "Run a quick pipeline smoke test on a synthetic sample paper shipped "
            "with bibr, using the configuration just saved."
        )
        self.console.print(self._smoke_test_note())

        if not Confirm.ask("Run a test extraction now?", default=confirm_default):
            self.console.print("[dim]Skipped.[/dim]")
            return

        # Reload the process-global Settings singleton from the .env file we
        # just wrote, in place, so chew() picks up what was just configured
        # instead of whatever was loaded when this process started. Must
        # preserve the Settings object's identity — see
        # _reload_settings_in_place's docstring for why a plain rebind of
        # bibr.config.Settings would leave modules that already imported it
        # (e.g. bibr.clients.llm, bibr.local.ocr) silently stale.
        try:
            _reload_settings_in_place()
        except Exception as exc:  # noqa: BLE001 — best-effort; fall back to current Settings
            self.console.print(f"[dim]Couldn't reload settings from .env: {escape(str(exc))}[/dim]")

        try:
            resource = resources.files("bibr.data").joinpath("sample_paper.pdf")
        except (ModuleNotFoundError, TypeError):
            resource = resources.files("bibr").joinpath("data", "sample_paper.pdf")

        from bibr.api import chew

        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = Path(tmpdir) / "sample_paper.pdf"
            sample_path.write_bytes(resource.read_bytes())

            t0 = time.monotonic()
            try:
                with self.console.status(
                    "Running test extraction (first run may download models) …"
                ):
                    result = chew(sample_path, pages="1", no_llm=True, refs="off")
            except Exception as exc:  # noqa: BLE001 — must never crash the wizard
                key_env = self._llm_key_env_hint()
                api_key = self.env_vars.get(key_env, "") if key_env else ""
                msg = _redact(str(exc), api_key)
                ui.error(self.console, f"Test extraction failed: {escape(msg)}")
                self.console.print(self._smoke_failure_hint(exc))
                return

        elapsed = time.monotonic() - t0
        data = result.data
        metadata = data.get("metadata") or {}
        title = metadata.get("title") or "(no title extracted)"
        n_authors = len(data.get("author") or [])
        n_refs = len(data.get("bib") or [])
        ui.ok(self.console, f"Pipeline smoke test succeeded ({elapsed:.1f}s)")
        self.console.print(
            f"  [dim]Title:[/dim] {escape(title)}\n"
            f"  [dim]Authors:[/dim] {n_authors}\n"
            f"  [dim]References:[/dim] {n_refs}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_HELP_TEXT = """\
usage: bibr setup [-h] [--advanced]

Interactive setup wizard for bibr.

By default this runs the short recommended flow:

  1. Detect your hardware (Apple Silicon / NVIDIA GPU / neither).
  2. Show a concrete plan preview (OCR + LLM backends, extras, memory mode,
     privacy, first-run download, speed) and ask a single confirmation.
  3. If this machine can't run local models, fork: use a private OCR/LLM
     server if you have one, otherwise give informed consent to cloud (Google
     Gemini). Declining any recommendation offers a handoff to --advanced
     rather than dead-ending.
  4. Write .env (and save a preset), then optionally install extras and run
     validation: a connection test for cloud, a local server health check for
     managed local LLMs, and a smoke-test extraction on a bundled sample PDF.

Use --advanced for step-by-step control over extras, LLM provider + API key,
connection test, OCR backend, memory mode, and writing .env.

Re-running is safe: the wizard does not take its answers from your existing
.env and writes .env only at the save step. If one is present then, it asks
whether to overwrite, merge, or skip (merge is the default, so hand-edited
values are preserved). Choosing an LLM provider also writes LLM_BACKEND=cloud,
and a blank LLM_API_KEY or LLM_BASE_URL where you entered none, so that an
older key or server cannot override the one you entered. Press Ctrl+C at any
time to quit without saving.

options:
  -h, --help   show this help message and exit
  --advanced   run the detailed provider/backend picker
"""


def main() -> None:
    args = sys.argv[1:]
    if any(a in {"-h", "--help"} for a in args):
        print(_HELP_TEXT, end="")
        return
    advanced = "--advanced" in args
    unknown = [a for a in args if a != "--advanced"]
    if unknown:
        Console().print(f"[red]Unknown option:[/red] {unknown[0]}")
        raise SystemExit(2)
    try:
        wizard = SetupWizard()
        if advanced:
            wizard.run_advanced()
        else:
            wizard.run()
    except (KeyboardInterrupt, EOFError) as exc:
        Console().print("\n[dim]Aborted.[/dim]")
        raise SystemExit(1) from exc
