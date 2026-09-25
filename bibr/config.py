import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal, cast
from urllib.parse import quote as _url_quote
from urllib.parse import urlparse, urlunparse

from pydantic import (
    AliasChoices,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Safe at import time: bibr.exceptions has no bibr dependencies, so no cycle.
from bibr.exceptions import ConfigurationError


def _compute_code_hash() -> str:
    """Compute a short hash of bibr source for automatic cache versioning.

    Returns a deterministic hash that changes only when Python source files
    change, so container restarts without code changes preserve the cache.

    Memoized after first call — under LitServe each worker imports
    ``bibr.config`` independently, so eager evaluation at import time used
    to walk every ``*.py`` under the package per worker startup. Most
    callers never read ``CacheOptions.version``, so deferring keeps the
    cold-start cost off the critical path entirely.
    """
    global _CODE_HASH_CACHED
    if _CODE_HASH_CACHED is not None:
        return _CODE_HASH_CACHED
    h = hashlib.sha256()
    pkg_dir = Path(__file__).parent
    for f in sorted(pkg_dir.rglob("*.py")):
        try:
            h.update(f.read_bytes())
        except OSError:
            continue
    _CODE_HASH_CACHED = f"auto-{h.hexdigest()[:8]}"
    return _CODE_HASH_CACHED


_CODE_HASH_CACHED: str | None = None


_ENV_FILE = ".env"

#: Replaces the dotenv fallback chain outright. Unset keeps the default chain;
#: empty disables dotenv loading; otherwise an ``os.pathsep``-separated list of
#: paths, merged in the same last-wins order.
ENV_FILE_OVERRIDE_VAR = "BIBR_ENV_FILE"
# Set to 1/true to make every settings model ignore ``.env`` files (the
# process environment still applies). For harnesses and CI: a run that pins its
# variables in the environment must not inherit the twenty-first one from a
# developer's ``.env`` in the checkout or from ``~/.bibr/.env``.
DOTENV_DISABLE_VAR = "BIBR_DISABLE_DOTENV"


def dotenv_disabled() -> bool:
    """True when ``BIBR_DISABLE_DOTENV`` asks bibr to ignore every ``.env`` file."""
    return os.environ.get(DOTENV_DISABLE_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def dotenv_files_present() -> list[Path]:
    """The ``.env`` files bibr would read right now, in load order."""
    return [path.resolve() for path in _default_env_files() if path.is_file()]


def _default_env_files() -> tuple[Path, ...]:
    """Dotenv fallback chain: ``~/.bibr/.env`` first, CWD ``./.env`` last.

    pydantic-settings' ``DotEnvSettingsSource`` reads ``env_file`` entries in
    order and merges them key-by-key, later files overriding earlier ones —
    and silently skips any path that doesn't exist. Listing the CWD file last
    means it wins when both are present, while running ``bibr`` from a
    directory with no local ``.env`` still picks up user-global config from
    ``~/.bibr/.env`` instead of silently losing it. Computed fresh on every
    call (not cached at import time) so it reflects the current CWD/home.

    ``BIBR_ENV_FILE`` replaces the chain: an ``os.pathsep``-separated list of
    paths (same last-wins merge), or an empty value to load no dotenv file at
    all. The empty form is what makes a process hermetic from whatever ``.env``
    happens to sit in its CWD or home — the test suite sets it for exactly that
    reason, and it is equally useful for containers and CI. A per-model
    ``_env_file=`` argument still outranks this.
    """
    override = os.environ.get(ENV_FILE_OVERRIDE_VAR)
    if override is not None:
        return tuple(Path(part) for part in override.split(os.pathsep) if part)
    return (Path.home() / ".bibr" / ".env", Path(".env"))


def _section(prefix: str) -> SettingsConfigDict:
    """Build a SettingsConfigDict for a sub-model reading ``.env`` with ``prefix``."""
    return SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix=prefix,
        extra="ignore",
    )


class _NoInterpolationDotEnvSource(DotEnvSettingsSource):
    """``.env`` source with python-dotenv ``${VAR}`` interpolation disabled.

    Interpolation is on by default in python-dotenv (used under the hood by
    pydantic-settings) and silently shell-expands ``${VAR}`` references inside
    values — e.g. an API key or Redis password literally containing ``${HOME}``
    is mangled on load, and bibr's own ``env_utils.parse_env`` reads the literal
    value, so the loaded config diverges from what ``preset diff``/``show``
    displays. bibr always wants ``.env`` values read verbatim.
    """

    def _read_env_file(self, file_path: Path):
        from pydantic_settings.sources.utils import parse_env_vars

        from bibr.env_utils import read_dotenv

        file_vars = read_dotenv(file_path, encoding=self.env_file_encoding or "utf8")
        return parse_env_vars(
            file_vars,
            self.case_sensitive,
            self.env_ignore_empty,
            self.env_parse_none_str,
        )


# Field-name markers that identify a secret (API key / password / token).
# Shared by the repr/model_dump redaction (audit M3) and the cache-fingerprint
# scrub so the two can't drift.
#
# End-anchored, mirroring ``config_introspect._SECRET_RE``. An unanchored
# substring match classified every ``*_max_tokens`` field as a secret, which
# excluded eleven LLM budget knobs from ``compute_behavior_fingerprint`` — so
# changing LLM_MAX_TOKENS or REF_PARSE_MAX_TOKENS left the serve cache
# namespace byte-identical and results produced under the old budget were
# silently re-served.
_SECRET_NAME_MARKERS = ("api_key", "password", "token", "secret", "api_email")


def _is_secret_name(name: str) -> bool:
    return str(name).lower().endswith(_SECRET_NAME_MARKERS)


def _split_csv_env(value, *, lower: bool = False):
    """Accept a comma-separated env string or a list; normalize to tokens.

    A bare ``list[str]`` field is JSON-decoded by pydantic-settings, so the
    comma-separated form the field descriptions and ``docs/guides/mcp.md``
    prescribe raised ``SettingsError`` on the first attribute access of
    ``Settings`` anywhere — taking ``bibr serve`` down with an unhandled
    traceback. Pair with ``Annotated[list[str], NoDecode]``.
    """
    items = value
    if isinstance(items, str):
        text = items.strip()
        if text.startswith("["):
            # The JSON form is what pydantic-settings decodes natively and what
            # existing deployments already set — keep accepting it alongside
            # the comma-separated form the field descriptions prescribe.
            try:
                items = json.loads(text)
            except ValueError:
                items = text.split(",")
        else:
            items = text.split(",")
    if isinstance(items, (list, tuple)):
        cleaned = [str(item).strip() for item in items]
        return [item.lower() if lower else item for item in cleaned if item]
    return value


class _BibrSettings(BaseSettings):
    """Base for all bibr settings models.

    Swaps the default dotenv source for :class:`_NoInterpolationDotEnvSource` so
    ``${VAR}`` in ``.env`` values is never shell-interpolated on load.

    ``repr()`` and ``model_dump()`` mask secret-named fields (API keys, passwords,
    tokens) so a stray ``logger.debug(settings)`` / ``model_dump()`` can't leak
    plaintext credentials (audit M3). The stored value is untouched — only its
    serialized/printed form is redacted.
    """

    def __repr_args__(self):
        for name, value in super().__repr_args__():
            if value is not None and _is_secret_name(name):
                yield name, "***"
            else:
                yield name, value

    @model_serializer(mode="wrap")
    def _redact_secrets_on_dump(self, handler):
        data = handler(self)
        if isinstance(data, dict):
            for key, value in data.items():
                if value is not None and not isinstance(value, dict) and _is_secret_name(key):
                    data[key] = "***"
        return data

    def __init__(self, **kwargs):
        # Resolve the CWD-then-home fallback chain fresh at instantiation time
        # unless the caller passed an explicit ``_env_file`` override (tests,
        # tooling) — that override must always win. ``BIBR_DISABLE_DOTENV``
        # empties the chain instead, for every section model alike.
        if "_env_file" not in kwargs:
            kwargs["_env_file"] = None if dotenv_disabled() else _default_env_files()
        super().__init__(**kwargs)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        no_interp = _NoInterpolationDotEnvSource(
            settings_cls,
            # Preserve the resolved env_file (incl. any ``_env_file`` override)
            # and encoding; fall back to the section defaults if absent.
            env_file=getattr(dotenv_settings, "env_file", _ENV_FILE),
            env_file_encoding=getattr(dotenv_settings, "env_file_encoding", "utf-8"),
        )
        return (init_settings, env_settings, no_interp, file_secret_settings)


class LlmOptions(_BibrSettings):
    """LLM client settings. Env: ``LLM_PROVIDER``, ``LLM_MODEL``, ``LLM_API_KEY``, ..."""

    model_config = _section("LLM_")

    provider: str = Field(
        "google",
        description='LLM provider: "google" (default), "openai", "anthropic", "groq", or "ollama".',
    )
    model: str = Field("gemini-3.5-flash-lite", description="Model name for the selected provider.")
    model_revision: str | None = Field(
        None,
        description="Exact model revision/commit deployed (surfaced in qualification_provenance; "
        "null falls back to the expected revision only when LLM_MODEL is "
        "numind/NuExtract3-FP8).",
    )
    jinja_sha256: str | None = Field(
        None,
        description="sha256 of the deployed chat template (surfaced in qualification_provenance; "
        "null falls back to the expected template sha only when LLM_MODEL is "
        "numind/NuExtract3-FP8).",
    )
    api_key: str | None = Field(
        None,
        description="Generic API key, used by the openai provider and OpenAI-compatible endpoints. "
        "Google/Anthropic/Groq use their provider-specific keys.",
    )
    base_url: str | None = Field(
        None, description="Base URL override for custom OpenAI-compatible endpoints."
    )
    chat_template_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Chat-template options for custom OpenAI-compatible endpoints, e.g. "
        "LLM_CHAT_TEMPLATE_KWARGS='{\"enable_thinking\": false}'. Ignored for real OpenAI.",
    )
    ollama_base_url: str = Field(
        "http://localhost:11434", description="Ollama server URL (LLM_PROVIDER=ollama)."
    )
    reasoning_effort: str | None = Field(
        "minimal",
        description="Reasoning effort for reasoning models (minimal/low/medium/high; "
        "null disables the parameter for models that reject it).",
    )
    reasoning_effort_authors: str | None = Field(
        "low",
        description="Per-call reasoning effort override for the authors extraction call "
        "(benefits from more reasoning due to spatial/cross-reference logic).",
    )
    reasoning_effort_citations: str | None = Field(
        "low",
        description="Per-call reasoning effort override for the citation-linking call "
        "(benefits from more reasoning due to spatial/cross-reference logic).",
    )
    rate_limit_rpm: int = Field(
        60,
        description="LLM rate limit in requests per minute. Guards a cloud provider's quota, "
        "so it is auto-lowered to 10 for the ollama provider and auto-raised for a managed "
        "local server (which has no external quota) — both unless set explicitly.",
    )
    # Max in-flight LLM requests (0 = unlimited). Local single-device servers
    # share one model/Metal or CUDA context and are often safest serialized;
    # cloud providers handle concurrency fine at the default.
    max_concurrency: int = Field(
        0,
        description="Max in-flight LLM requests (0 = unlimited). Set 1 to serialize against a "
        "local single-device server; cloud providers are fine at the default.",
    )
    temperature: float = Field(
        0.0,
        ge=0.0,
        description="Sampling temperature for OpenAI-compatible LLM responses.",
    )
    max_tokens: int = Field(65536, description="Max completion tokens for LLM responses.")
    title_max_tokens: int | None = Field(
        4096,
        ge=0,
        description="Max completion tokens for title/abstract/keywords extraction; 0 disables the task cap.",
    )
    authors_max_tokens: int | None = Field(
        8192,
        ge=0,
        description="Max completion tokens for author extraction; 0 disables the task cap.",
    )
    paper_classification_max_tokens: int | None = Field(
        1024,
        ge=0,
        description="Max completion tokens for broad paper classification; 0 disables the task cap.",
    )
    paper_type_max_tokens: int | None = Field(
        512,
        ge=0,
        description="Max completion tokens for narrow paper-type labeling; 0 disables the task cap.",
    )
    section_max_tokens: int | None = Field(
        4096,
        ge=0,
        description="Max completion tokens for section classification/detection; 0 disables the task cap.",
    )
    integrity_max_tokens: int | None = Field(
        4096,
        ge=0,
        description="Max completion tokens for research-integrity extraction; 0 disables the task cap.",
    )
    equation_max_tokens: int | None = Field(
        4096,
        ge=0,
        description="Max completion tokens for equation extraction; 0 disables the task cap.",
    )
    citation_max_tokens: int | None = Field(
        8192,
        ge=0,
        description="Max completion tokens for citation resolution; 0 disables the task cap.",
    )
    max_input_chars: int = Field(300_000, description="Max input characters sent to the LLM.")
    # Reference-segmentation OUTPUT bound: the anchor-emit call produces ~1 short
    # anchor per reference, so a long bibliography in one window emits too many
    # anchors to finish within the LLM hard-timeout (the LIGO 118->0 collapse).
    # Windowing seg input to this many chars (~50 refs/window) keeps each call's
    # output bounded. Distinct from max_input_chars (the parse-side input budget).
    ref_seg_window_chars: int = Field(
        16_000,
        description="Max input characters per LLM reference-segmentation window (~50 refs/window).",
    )
    classification_max_chars: int = Field(
        20_000, description="Max input characters sent to the LLM section-classification call."
    )
    core_cutoff_max_sentences: int = Field(
        250,
        description="Upper bound (in sentences) on the core-metadata front-matter slice. Caps "
        "late cutoffs caused by misclassified early section headers; also the fallback slice "
        "when no IMRaD header is found. 0 disables the cap (fallback then stays 250).",
    )
    per_task_context: bool = Field(
        True,
        description="Send task-specific context slices to the core-metadata LLM calls: "
        "page-1 + ORCID/correspondence rows to the authors call and front matter through "
        "the end of the abstract to the classification call, instead of the full "
        "front-matter blob. The title/keywords call always receives the full blob.",
    )
    # Experimental: one merged LLM call for title/abstract/keywords + authors
    # + classification instead of three — saves resending the front-matter
    # text 3x. Off until quality-gated against the three-call path.
    merged_core_metadata: bool = Field(
        False,
        description="Experimental: merge title/abstract/keywords + authors + classification into "
        "one LLM call instead of three, saving repeated front-matter tokens. Off until quality-gated.",
    )
    validation_attempts: int | None = Field(
        None,
        ge=1,
        description="Instructor schema-validation attempts. None auto-selects 1 for deterministic "
        "custom OpenAI-compatible endpoints and 3 for cloud providers.",
    )
    timeout_seconds: int = Field(30, description="Per-request LLM timeout in seconds.")
    track_usage: bool = Field(True, description="Track LLM token usage.")
    capture_trace: bool = Field(
        False,
        description="Opt-in: capture each LLM call's rendered prompt and raw response into "
        "extraction.trace (see LlmTraceExport). Off by default — a default-config export has "
        "no trace key. Prompts/completions are scrubbed of credential-shaped tokens before "
        "capture; callers choose storage and retention for trace-bearing exports. Captures Instructor "
        "backend calls only — the NuExtract-native structured backend is not yet instrumented, "
        "so enabling this on a nuextract-native deployment (a common local/self-hosted setup) "
        "silently produces no trace rows; a warning is logged in that case.",
    )
    thinking_budget: int = Field(
        0, description="Thinking budget for extended-thinking models (0 = disabled)."
    )
    # Default LLM backend for chew when --llm isn't passed: "cloud" or a
    # managed local server ("local", "vllm", "vllm-mlx", "rapid-mlx", "llama-cpp", "llmster"). Written by
    # `bibr setup` when the user picks the local path.
    backend: str = Field(
        "cloud",
        description='Default LLM backend for chew when --llm is not passed: "cloud" or a managed '
        'local server ("local", "vllm", "vllm-mlx", "rapid-mlx", "llama-cpp", "llmster"). '
        "Written by `bibr setup`.",
    )
    local_model: str | None = Field(
        None,
        description="Model id served by the managed local LLM backends. Unset picks the "
        "NuExtract 3 variant the selected backend can load: bf16 for vLLM, GGUF Q4_K_M for "
        "llama.cpp, the 8-bit MLX build for Apple Silicon (8bit runs ~2x faster than nvfp4 "
        "under MLX, which lacks fast nvfp4 kernels). `bibr setup` writes it explicitly.",
    )
    local_mem_fraction: float = Field(
        0.85, description="GPU memory fraction reserved for the managed local LLM server (0.0-1.0)."
    )
    vllm_mlx_port: int = Field(8767, description="Port for the managed vllm-mlx LLM server.")
    # Extra CLI args appended to the managed ``vllm_mlx.server`` command
    # (shlex-split). E.g. disable Qwen thinking and strip reasoning content:
    # '--default-chat-template-kwargs {"enable_thinking":false} --reasoning-parser qwen3'
    vllm_mlx_extra_args: str = Field(
        "",
        description="Extra CLI args (shlex-split) appended to the managed vllm-mlx LLM server command.",
    )
    vllm_mlx_continuous_batching: bool = Field(
        True,
        description="Launch the managed vllm-mlx LLM server with --continuous-batching (BatchedEngine) "
        "instead of the serial SimpleEngine. Default ON: bibr launches the server via "
        "bibr.local._vllm_mlx_server, a stable wrapper around the validated vllm-mlx 0.4.x "
        "entrypoint with conservative hybrid-cache handling for NuExtract3/Qwen3-Next. Managed "
        "local LLM calls are still serialized by default through LLM_MAX_CONCURRENCY=1 unless "
        "the operator explicitly raises that cap after validating their runtime/model combo.",
    )
    # Managed vLLM LLM server (Linux/CUDA twin of vllm-mlx).
    vllm_port: int = Field(
        8769, description="Port for the managed vLLM LLM server (Linux/CUDA; --llm local)."
    )
    vllm_startup_timeout: int = Field(
        600, description="Startup timeout in seconds for the managed vLLM LLM server."
    )
    vllm_extra_args: str = Field(
        "",
        description="Extra CLI args appended to the managed vLLM LLM server command "
        "(overrides registry defaults on repeated flags).",
    )
    # Managed Rapid-MLX LLM server (Apple Silicon experimental fast path).
    rapid_mlx_model: str = Field(
        "qwen3.5-4b-4bit",
        description="Model id or Rapid-MLX alias for the managed Rapid-MLX LLM server. "
        "Default is Qwen3.5 4B 4-bit, served with no-thinking defaults.",
    )
    rapid_mlx_port: int = Field(8773, description="Port for the managed Rapid-MLX LLM server.")
    rapid_mlx_extra_args: str = Field(
        "",
        description="Extra CLI args appended to the managed Rapid-MLX LLM server command.",
    )
    # External LM Studio/llmster runtime. bibr may start already-installed
    # services and load an already-downloaded model, but never installs or
    # downloads either resource automatically.
    llmster_model: str = Field(
        "", description="LM Studio model key already present in `lms ls --json`."
    )
    llmster_model_id: str = Field(
        "bibr-local", description="Stable loaded-model identifier exposed through the local API."
    )
    llmster_port: int = Field(1234, description="LM Studio OpenAI-compatible server port.")
    llmster_context_length: int = Field(
        32768, description="Context length requested when bibr loads the llmster model."
    )
    llmster_load_args: str = Field(
        "", description="Additional expert arguments appended to `lms load`."
    )
    # Native Windows / low-VRAM GGUF runtime (llama.cpp).
    llama_cpp_port: int = Field(
        8770,
        description="Port for the managed llama.cpp LLM server (native Windows / low-VRAM GGUF).",
    )
    llama_cpp_context_size: int = Field(
        16384,
        description="Context size (tokens) for the managed llama.cpp LLM server. NuExtract3's "
        "hybrid attention (8/32 full-attention layers) makes KV cheap, so 16384 fits the "
        "24,000-char input cap plus the 4,096-token completion cap with headroom on 6 GB cards.",
    )
    llama_cpp_startup_timeout: int = Field(
        600, description="Startup timeout in seconds for the managed llama.cpp LLM server."
    )
    llama_cpp_extra_args: str = Field(
        "",
        description=(
            "Extra CLI args appended to the managed llama.cpp LLM server command. "
            "Overrides bibr defaults for the same flags. Role-based LLM defaults include "
            "--flash-attn on, --cache-type-k/v q8_0, --n-gpu-layers 999, plus probe-gated "
            "--parallel 2 --kv-unified, --spec-type ngram-mod, and --no-mmproj when supported "
            "(else --parallel 1)."
        ),
    )
    # Instructor structured-output mode for custom-base_url (local) servers.
    # "" (default) = JSON_SCHEMA (strict guided decoding). Small models under
    # strict grammar satisfy it with required fields only and skip optional
    # ones (e.g. drop authors); "json" (json_object) keeps output valid JSON
    # but lets the prompt drive full field population. Ignored for real OpenAI.
    instructor_mode: str = Field(
        "",
        description='Instructor structured-output mode for custom-base_url (local) servers: "" '
        '(default, strict JSON_SCHEMA) or "json" (json_object, looser but more complete on small '
        "models). Ignored for real OpenAI.",
    )
    # Structured-output backend selector. Native NuExtract templates are
    # experimental and explicit-only pending runtime-specific qualification.
    structured_backend: str = Field(
        "auto",
        description='Structured-output backend: "auto" (default), "instructor", or '
        '"nuextract-native". Auto uses Instructor; native NuExtract templates are '
        "experimental and require explicit runtime-specific qualification.",
    )

    batch_model: str = Field(
        "claude-haiku-4-5", description="Model name for the offline batch LLM layer."
    )
    batch_poll_interval_s: int = Field(
        60,
        description="Poll interval in seconds for the offline batch LLM layer's job status checks.",
    )


class OcrOptions(_BibrSettings):
    """OCR service settings. Env: ``OCR_BACKEND``, ``OCR_BASE_URL``, ..."""

    model_config = _section("OCR_")

    api_key: str | None = Field(
        None, description="OCR server API key, if authentication is required."
    )
    allow_insecure_http: bool = Field(
        False,
        description="Allow plain HTTP to a non-loopback OCR server. Enable only on a trusted "
        "private network; remote OCR defaults to HTTPS-only.",
    )
    request_timeout: int = Field(90, description="OCR request timeout in seconds.")
    backend: str = Field(
        "paddle",
        description='OCR server backend: "paddle" (default), "paddle-vllm" (GPU/vLLM), '
        '"paddle-rapid-mlx" (Apple Silicon/Rapid-MLX), "paddle-mlx-vlm" '
        '(Apple Silicon/MLX-VLM), or "paddle-http" (external Paddle server); '
        '"glm-mlx" (disabled — use glm-rapid-mlx), '
        '"glm-rapid-mlx" (Apple Silicon/Rapid-MLX), "glm-llama" (Windows default; llama.cpp), '
        'or "glm-http" (external GLM server) — plus the cloud vision-LLM providers "gemini", '
        '"openai", and "anthropic".',
    )
    model: str | None = Field(None, description="OCR model name override.")
    profile: Literal["paddle", "glm"] | None = Field(
        None,
        description="OCR model profile override. Null infers Paddle or GLM from the backend/model name.",
    )
    generation_max_tokens: int | None = Field(
        None,
        description=(
            "OCR generation max-token override for every task in the resolved model profile."
        ),
    )
    generation_temperature: float | None = Field(
        None, description="OCR generation temperature override for the resolved model profile."
    )
    paddle_model: str = Field(
        "PaddlePaddle/PaddleOCR-VL-1.6", description="PaddleOCR-VL HuggingFace model id."
    )
    paddle_revision: str = Field(
        "66317acc4c9fc17bd154591ce650735cd2855f3e",
        description="Pinned revision for the PaddleOCR-VL HuggingFace model.",
    )
    paddle_served_model: str = Field(
        "paddle-ocr-vl-1.6", description="Served model name expected from Paddle OCR HTTP servers."
    )
    paddle_vllm_port: int = Field(8774, description="Port for the managed Paddle vLLM OCR server.")
    paddle_vllm_startup_timeout: int = Field(
        900, description="Startup timeout in seconds for the managed Paddle vLLM OCR server."
    )
    paddle_vllm_extra_args: str = Field(
        "", description="Extra CLI args appended to the managed Paddle vLLM OCR server command."
    )
    paddle_mlx_model: str = Field(
        "olragon/PaddleOCR-VL-1.6-8bit",
        description="Model id for the managed Paddle MLX OCR server.",
    )
    paddle_mlx_port: int = Field(8775, description="Port for the managed Paddle MLX OCR server.")
    paddle_mlx_startup_timeout: int = Field(
        600, description="Startup timeout in seconds for the managed Paddle MLX OCR server."
    )
    paddle_mlx_extra_args: str = Field(
        "", description="Extra CLI args appended to the managed Paddle MLX OCR server command."
    )
    paddle_rapid_mlx_model: str = Field(
        "olragon/PaddleOCR-VL-1.6-8bit",
        description="Model id for the managed Paddle Rapid-MLX OCR server.",
    )
    llama_cpp_model: str = Field(
        "ggml-org/GLM-OCR-GGUF:Q8_0", description="Model id for the managed llama.cpp OCR server."
    )
    llama_cpp_port: int = Field(8771, description="Port for the managed llama.cpp OCR server.")
    llama_cpp_context_size: int = Field(
        8192, description="Context size (tokens) for the managed llama.cpp OCR server."
    )
    llama_cpp_startup_timeout: int = Field(
        600, description="Startup timeout in seconds for the managed llama.cpp OCR server."
    )
    llama_cpp_extra_args: str = Field(
        "",
        description=(
            "Extra CLI args appended to the managed llama.cpp OCR server command. "
            "Overrides bibr defaults for the same flags. Role-based OCR defaults include "
            "--flash-attn on, --cache-type-k/v q8_0, --n-gpu-layers 999, --parallel 1 "
            "(OCR image encode serializes across slots, so it stays single-slot)."
        ),
    )
    native_text_enabled: bool = Field(
        True, description="Native text extraction — skip OCR for text-layer PDFs."
    )
    native_text_min_chars: int = Field(
        20, description="Minimum character count for a native-text extraction to be accepted."
    )
    # Minimum fraction of "reasonable" characters (letters/digits/punctuation/
    # whitespace/math symbols, any script) a native-text extraction must have
    # to be trusted. Guards against corrupt CID-to-Unicode maps that produce
    # mojibake/garbage — such pages fall back to OCR instead. See
    # bibr/ocr/native_text.py::_is_native_text_usable.
    native_text_min_printable_ratio: float = Field(
        0.85,
        description="Minimum fraction of printable characters a native-text extraction must have "
        "to be trusted; below this the page falls back to OCR.",
    )
    local_gpus: int = Field(
        1,
        description="Number of GPUs dedicated to the local OCR server (tensor parallelism). Also "
        "sets ocr.max_concurrent_regions = 16 x this value.",
        # OCR_SGLANG_GPUS kept as an alias: the setting predates the removal of
        # the bundled SGLang backend and is not SGLang-specific.
        validation_alias=AliasChoices("OCR_LOCAL_GPUS", "OCR_SGLANG_GPUS"),
    )
    max_concurrent_files: int = Field(
        4, description="Max concurrent files being OCR'd simultaneously."
    )
    max_concurrent_regions: int = Field(
        16,
        description="Max concurrent OCR region requests server-wide "
        "(default 16 x OCR_LOCAL_GPUS; keep <= 16 per server instance to avoid detokenizer "
        "stalls). Binds every OCR path, and is the cap used on its own when files run one at "
        "a time against a managed batching server (paddle-vllm).",
    )
    concurrent_regions_per_file: int = Field(
        6,
        description="Max concurrent OCR region requests per file (prevents one file starving "
        "others when files overlap). Never exceeds OCR_MAX_CONCURRENT_REGIONS.",
    )
    # Which of the two concurrency fields above the USER explicitly provided
    # (env var / .env / init kwarg). Recorded by GlobalSettings'
    # ``compute_ocr_concurrency`` validator before it auto-tunes them.
    _user_set_concurrency: frozenset[str] = PrivateAttr(default=frozenset())

    @property
    def user_set_concurrency(self) -> frozenset[str]:
        """Concurrency fields the user explicitly set, recorded pre-auto-tune.

        ``model_fields_set`` alone cannot answer "did the user set this":
        ``GlobalSettings.compute_ocr_concurrency`` assigns
        ``max_concurrent_regions`` (all platforms) and
        ``concurrent_regions_per_file`` (Apple Silicon) at construction, and
        pydantic v2 adds assigned fields to ``model_fields_set`` — so after
        construction both fields look "set" regardless of origin. Backends
        that want to clamp only non-user-set values (e.g. the single-slot
        llama.cpp OCR client) must consult this snapshot instead.
        """
        return self._user_set_concurrency

    local_model: str = Field(
        "THUDM/GLM-OCR", description="HuggingFace model id for the local GLM OCR backends."
    )
    rapid_mlx_model: str = Field(
        "mlx-community/GLM-OCR-8bit",
        description="Model id for the managed Rapid-MLX OCR server. Defaults to GLM-OCR 8-bit.",
    )
    rapid_mlx_port: int = Field(8772, description="Port for the managed Rapid-MLX OCR server.")
    rapid_mlx_extra_args: str = Field(
        "",
        description="Extra CLI args appended to the managed Rapid-MLX OCR server command.",
    )
    rapid_mlx_recycle_after: int = Field(
        80,
        description="Restart the managed Rapid-MLX OCR subprocess after this many regions. "
        "Mitigates a vendored rapid-mlx bug (vllm_mlx MLLMBatchGenerator's VisionEmbeddingCache "
        "pixel cache is bounded by entry count, not bytes — 100 entries, no eviction by size) "
        "that never gets useful reuse on bibr's unique-crop-per-region workload, so it fills "
        "with ~100 dead ~34-72MB float32 tensors (~3.4GB) and OCR then fails silently while "
        "/health stays green. 80 leaves headroom below the ~100-entry saturation point. 0 "
        "disables recycling.",
    )
    # Files whose OCR success rate falls below this fraction of OCR-needed
    # regions are marked failed with code ``ocr_mostly_failed``. Set to 0
    # to disable. Default 0.5 = "fail if more than half the OCR calls
    # returned no content" — catches silent-engine-crash regressions
    # without false-positives on legitimately empty pages.
    min_success_rate: float = Field(
        0.5,
        description="Mark a file failed (code ocr_mostly_failed) when fewer than this fraction of "
        "OCR-needed regions return content. Set to 0 to disable.",
    )
    # When to tear down a local in-process OCR engine between batch chunks.
    # "auto" (default) keeps it resident in balanced mode unless a local LLM
    # shares the GPU — avoiding per-chunk weight reloads (vllm-mlx 30-60s) on
    # multi-chunk batches; "always" forces the legacy per-chunk teardown;
    # "never" keeps it loaded until the pipeline closes. aggressive always
    # tears down (8 GB boxes); keep_all never does.
    unload_between_chunks: Literal["auto", "always", "never"] = Field(
        "auto",
        description='When to tear down a local in-process OCR engine between batch chunks: "auto" '
        '(default, keep resident in balanced mode unless a local LLM shares the GPU), "always" '
        '(legacy per-chunk teardown), or "never".',
    )

    @model_validator(mode="after")
    def validate_profile(self) -> "OcrOptions":
        """Require an explicit profile when a backend/model has no known OCR family."""
        if self.backend.lower() in {"gemini", "openai", "anthropic"}:
            return self

        from bibr.ocr.profiles import resolve_ocr_profile

        resolve_ocr_profile(
            explicit=self.profile,
            backend=self.backend,
            model=self.model or "",
            max_tokens=self.generation_max_tokens,
            temperature=self.generation_temperature,
        )
        return self


class OcrVisionOptions(_BibrSettings):
    """Vision LLM OCR settings. Env: ``OCR_VISION_PROVIDER``, ``OCR_VISION_MODEL``, ..."""

    model_config = _section("OCR_VISION_")

    provider: str = Field(
        "google",
        description='Vision LLM provider for OCR: "google" (default), "openai", "anthropic".',
    )
    model: str = Field("gemini-3-flash-preview", description="Vision LLM model for OCR.")
    base_url: str | None = Field(None, description="Vision LLM base URL override.")
    rate_limit_rpm: int = Field(30, description="Vision LLM rate limit in requests per minute.")
    max_tokens: int = Field(16384, description="Vision LLM max completion tokens.")
    timeout_seconds: int = Field(60, description="Vision LLM request timeout in seconds.")


class FigureOptions(_BibrSettings):
    """Figure structured-data extraction. Env: ``FIG_EXTRACT``, ``FIG_MODEL``, ...

    Sends each figure crop to a cloud vision LLM and attaches structured
    metadata (chart type, panels, axes, series names) to the export as
    ``figure[].analysis``. Ships dark: ``extract="off"``. The values tier
    ("data") is Phase 2 and not yet accepted.

    NOT YET IMPLEMENTED. No stage reads any field in this section, so setting
    ``FIG_EXTRACT=meta`` produces no ``figure[].analysis`` — the pipeline logs
    a warning saying so rather than failing silently. Keep this section and
    ``RunConfig.emit_figures`` in sync when the analysis stage lands.
    """

    model_config = _section("FIG_")

    extract: Literal["off", "meta"] = Field(
        "off",
        description='Reserved figure-analysis tier: "off" (default) or "meta". Not yet '
        "implemented: meta logs a warning and does not add figure analysis. "
        "All FIG_* settings are inactive until the analysis stage is implemented.",
    )
    provider: str = Field(
        "google",
        description='Vision LLM provider: "google" (default), "openai", "anthropic".',
    )
    model: str = Field("gemini-3-flash-preview", description="Vision LLM model.")
    base_url: str | None = Field(None, description="Vision LLM base URL override.")
    rate_limit_rpm: int = Field(30, description="Rate limit in requests per minute.")
    max_tokens: int = Field(8192, description="Max completion tokens per figure call.")
    timeout_seconds: int = Field(90, description="Per-figure request timeout in seconds.")
    max_figures: int = Field(
        15, description="Max figures analyzed per paper; the rest are skipped with a warning."
    )
    max_concurrency: int = Field(4, description="Concurrent figure calls per paper.")
    min_crop_px: int = Field(
        120, description="Skip crops whose width or height is below this many pixels."
    )


class LayoutOptions(_BibrSettings):
    """PP-DocLayout detection tuning. Env: ``LAYOUT_DPI``, ``LAYOUT_BATCH_SIZE``, ..."""

    model_config = _section("LAYOUT_")

    # The torch checkpoint. PP-DocLayoutV4 (PaddlePaddle/PP-DocLayoutV4_safetensors)
    # loads through the same Auto classes once transformers ships it; switch
    # this, the revision and the ONNX pair below together, behind an eval gate.
    model_id: str = Field(
        "PaddlePaddle/PP-DocLayoutV3_safetensors",
        description="HF Hub repo id (or local directory) of the PP-DocLayout torch checkpoint "
        "(PP-DocLayoutV3 or PP-DocLayoutV4) used when layout runs on torch.",
    )
    model_revision: str = Field(
        "97d101e6db2642e162a1d05392d1b0231c91033e",
        description="HF Hub revision (commit SHA or branch) of LAYOUT_MODEL_ID to load. Pinned "
        "so a hub push cannot change layout output; 'main' tracks the repo head.",
    )
    # ONNX artifact for the layout model. The torch weights live in a
    # third-party repo, so the exported graph is hosted in a bibr-owned repo
    # (or a local bundle directory containing onnx/model.onnx). Resolved under
    # ML_RUNTIME (bibr/utils/ml_runtime.py); a missing repo/revision falls back
    # to torch in "auto" mode. The bundle manifest records which architecture
    # (PP-DocLayoutV3 or V4) it was exported from.
    onnx_model_id: str | None = Field(
        "scienceverse/bibr-layout-onnx",
        description="HF Hub repo id (or local bundle directory) holding the ONNX export of the "
        "layout model under onnx/. Null disables the ONNX runtime for layout.",
    )
    onnx_revision: str = Field(
        "2bcb16a65f5128fd9e61ac6c503204721cf01ce0",
        description="HF Hub revision (commit SHA or branch) of LAYOUT_ONNX_MODEL_ID to load. "
        "Pinned to the export whose regions match the torch model exactly; 'main' tracks the head.",
    )
    dpi: int = Field(200, gt=0, description="Page rasterization DPI for layout detection.")
    max_render_pixels: int = Field(
        25_000_000,
        ge=1,
        description="Maximum rasterized pixels allowed for one PDF page before rendering.",
    )
    max_render_dimension: int = Field(
        10_000,
        ge=1,
        description="Maximum rasterized width or height allowed for one PDF page.",
    )
    detection_threshold: float = Field(
        0.3, description="Minimum confidence score for a layout detection to be kept."
    )
    nms_iou_same: float = Field(
        0.6, description="Non-max-suppression IoU threshold for boxes of the same class."
    )
    nms_iou_diff: float = Field(
        0.98, description="Non-max-suppression IoU threshold for boxes of different classes."
    )
    large_image_area_landscape: float = Field(
        0.82,
        description="Page-area fraction above which a landscape image region is treated as a "
        "large/full-page figure.",
    )
    large_image_area_portrait: float = Field(
        0.93,
        description="Page-area fraction above which a portrait image region is treated as a "
        "large/full-page figure.",
    )
    section_classification_score: float = Field(
        0.85, description="Minimum confidence score for a detected section heading region."
    )
    containment_threshold: float = Field(
        0.5,
        description="Minimum overlap fraction for one detected region to be treated as contained in another.",
    )
    batch_size: int = Field(
        8,
        ge=1,
        description="Page batch size for layout model inference. The configured value is "
        "the CUDA batch; on CPU the local and serve detectors run one page at a time "
        "unless this was set explicitly (a CPU batch of 8 grows the ORT CPU arena to "
        "several GB with no throughput gain).",
    )
    # Coalescing window (ms) for the serve GpuBatcher: how long the layout
    # micro-batcher keeps gathering pages from concurrent requests after the
    # first queued page before flushing a partial batch. A single paper's
    # pages already pack without waiting; this only adds latency on the final
    # partial chunk, so a small value buys cross-request coalescing at
    # negligible cost. 0 = flush immediately (admission-gate only, no wait).
    batch_timeout_ms: int = Field(
        5,
        description="Coalescing window in milliseconds for the serve layout GpuBatcher "
        "(0 = flush immediately, admission-gate only).",
    )
    # Overlap-resolution pass in layout postprocessing. "legacy" = pairwise
    # containment filter with the coincident-pair collapse; "rulebook" = the
    # Docling-derived resolver (union-find grouping, keep-best-absorb-union,
    # IoU-based coincident dedup separated from containment-based nesting).
    # Ships dark: flip after an eval gate.
    overlap_resolver: Literal["legacy", "rulebook"] = Field(
        "legacy",
        description='Overlap-resolution pass for layout regions: "legacy" (pairwise containment '
        'filter) or "rulebook" (union-find keep-best-absorb-union resolver). '
        "Rulebook ships dark pending an eval gate.",
    )
    # Reading-order fallback used only when the layout model emits no order_seq.
    # "xy" = lexsort by (y, x) box centers; "rb" = rule-based horizontal-dilation
    # + up/down adjacency ordering (column-aware, Docling-derived).
    read_order_fallback: Literal["xy", "rb"] = Field(
        "xy",
        description='Reading-order fallback when the layout model emits no order_seq: "xy" '
        '(lexsort by box centers) or "rb" (column-aware horizontal-dilation adjacency ordering).',
    )
    torch_compile: bool = Field(False, description="Enable torch.compile for the layout model.")
    # Run the serve layout model on GPU. None = auto-detect (cuda→mps→cpu in the
    # worker). LitServe's accelerator="auto" can't be trusted here — it resolves
    # to CPU whenever torch isn't pre-imported in the master process, which bibr
    # never does — so the serve layer auto-detects independently. PP-DocLayout
    # on CPU is dramatically slower; force CPU with LAYOUT_USE_GPU=false only on
    # VRAM-tight boxes that co-locate OCR (mirrors SEGMENTER_USE_GPU).
    use_gpu: bool | None = Field(
        None,
        description="Run the serve layout model on GPU. Null = auto-detect (cuda->mps->cpu). "
        "Force CPU only on VRAM-tight boxes that co-locate OCR.",
    )


class CrossrefOptions(_BibrSettings):
    """Crossref enrichment. Env: ``CROSSREF_API_EMAIL``, ``CROSSREF_RATE_LIMIT_RPM``,
    ``CROSSREF_REDIS_CACHE``, ``CROSSREF_CACHE_REDIS_URL``, ``CROSSREF_CACHE_TTL_SECONDS``, ..."""

    model_config = _section("CROSSREF_")

    api_email: str | None = Field(
        None, description="Contact email for the Crossref polite API pool (strongly recommended)."
    )
    api_key: str | None = Field(
        None, description="Crossref API key, for Plus/Metadata Plus access."
    )
    rate_limit_rpm: int = Field(
        200,
        description="Crossref rate limit in requests per minute "
        "(raise to 600 if you set an email or have an API key).",
    )
    # Off by default: enrichment is a network fan-out against Crossref (and the
    # optional resolver) that adds seconds per paper and needs a polite-pool
    # email to run at volume. Per-run switches override this setting either
    # way: ``bibr chew --crossref`` / ``--no-crossref``, ``chew(crossref=...)``,
    # and the serve API's ``crossref`` form field.
    enrich: bool = Field(
        False,
        description="Enable Crossref/resolver reference enrichment (off by default). "
        "Per-run overrides: `bibr chew --crossref`/`--no-crossref`, `chew(crossref=...)`, "
        "or the serve API's `crossref` form field.",
    )
    # Every DOI-bearing reference otherwise costs its own /works/{doi} request
    # against a rate limiter shared by the whole process (and, with Redis, the
    # whole fleet). One /works?filter=doi:... request answers up to
    # _BULK_DOI_CHUNK of them and seeds the same response cache, so the per-ref
    # path below finds them already warm. Only hits are seeded: a DOI absent
    # from the bulk response still takes its own lookup, preserving the 404
    # semantics that stop a bad DOI from falling through to a title search.
    bulk_doi_lookup: bool = Field(
        True,
        description="Prefetch all DOI-bearing references in one Crossref filter query "
        "before the per-reference fan-out. Set false to restore one request per DOI.",
    )
    # Pipeline-fill parallelism for reference lookups. The SHARED rate limiter
    # (rate_limit_rpm), not this, enforces polite-pool compliance, so raising
    # concurrency only fills the pipeline — it never raises the request rate.
    # Sized so the 200 RPM polite pool stays saturated at ~4s/search latency.
    enrich_concurrency: int = Field(
        12, description="Pipeline-fill parallelism for Crossref reference lookups."
    )
    # Per-paper wall-clock budget (seconds) for the whole enrichment stage.
    enrich_timeout: float = Field(
        120.0, description="Per-paper wall-clock budget in seconds for the whole enrichment stage."
    )
    # Per-HTTP-request timeout (seconds), decoupled from enrich_timeout so a
    # single hung Crossref call cannot monopolize a concurrency slot for the
    # entire per-paper budget.
    request_timeout: float = Field(
        15.0, description="Per-HTTP-request timeout in seconds for Crossref calls."
    )
    # In-process LRU cache for works/search responses — duplicate refs across
    # a batch (and repeat serve requests) skip the API. 0 disables.
    cache_size: int = Field(
        1024,
        description="In-process LRU cache size for Crossref works/search responses. 0 disables.",
    )
    # --- Shared (tier-2) Redis response cache (opt-in) -------------------
    # When enabled, works/search responses are cached in Redis so they
    # survive restarts and are shared across bibr-serve and the workers.
    # Off by default → behavior is unchanged (in-process LRU only).
    # Env: CROSSREF_REDIS_CACHE
    redis_cache: bool = Field(
        False,
        description="Enable the shared (tier-2) Redis response cache for Crossref works/search "
        "responses. Off by default (in-process LRU only).",
    )
    # Redis URL for the shared cache. Falls back to REDIS_URL when unset.
    # Use a dedicated instance with `maxmemory-policy allkeys-lru` — cache
    # entries are reconstructible public metadata, unlike privacy-sensitive
    # job data (which needs noeviction). Env: CROSSREF_CACHE_REDIS_URL
    cache_redis_url: str | None = Field(
        None,
        description="Redis URL for the shared Crossref response cache. Falls back to REDIS_URL when unset.",
    )
    # TTL for shared-cache entries, seconds (default 30 days). allkeys-lru
    # handles memory pressure; the TTL bounds staleness from upstream
    # corrections. Env: CROSSREF_CACHE_TTL_SECONDS
    cache_ttl_seconds: int = Field(
        2_592_000,
        description="TTL for shared Crossref cache entries, in seconds (default 30 days).",
    )
    # How long both cache tiers remember a DOI lookup's 404 (no Crossref
    # record: a malformed DOI or another registry's). Short next to the
    # positive TTL, since a newly registered DOI starts resolving within days.
    # 0 = never cached. Env: CROSSREF_NOT_FOUND_TTL_SECONDS
    not_found_ttl_seconds: int = Field(
        86_400,
        ge=0,
        description="How long a Crossref 404 for a DOI (no record) is remembered in the response "
        "caches, in seconds (default 1 day), so repeat lookups skip the request. 0 disables.",
    )
    # Merge accepted bib_match data into bib rows at export:
    # "off" (default) | "fill" (fill empty fields only) | "replace" (also overwrite
    # from a match carrying the reference's printed DOI)
    consolidate: Literal["off", "fill", "replace"] = Field(
        "off",
        description='Merge accepted bib_match data into bib rows at export: "off" (default), '
        '"fill" (fill empty fields only), or "replace" (also overwrite printed values, only '
        "from a match carrying the reference's printed DOI).",
    )

    @field_validator("consolidate", mode="before")
    @classmethod
    def _lower_consolidate(cls, v):
        return v.lower() if isinstance(v, str) else v


class RorOptions(_BibrSettings):
    """ROR organization matching for affiliation strings and funder names.

    Runs as part of enrichment (``--crossref``), so it is off unless enrichment
    is on. Env: ``ROR_ENRICH``, ``ROR_CLIENT_ID``, ``ROR_URL``, ...
    """

    model_config = _section("ROR_")

    enrich: bool = Field(
        True,
        description="Match affiliation strings and funder names to ROR IDs when enrichment runs "
        "(`--crossref`). Set false to skip ROR.",
    )
    client_id: str | None = Field(
        None,
        description="ROR API client ID (free from ror.org), sent as the Client-Id header. It "
        "raises ROR's published rate limit from 50 to 2000 requests per 5 minutes.",
    )
    url: str = Field("https://api.ror.org/v2", description="ROR API base URL.")
    requests_per_5min: int | None = Field(
        None,
        ge=1,
        description="Client-side ROR request budget per 5 minutes. Default: ROR's published "
        "limit, 50 without a client ID and 2000 with one.",
    )
    request_timeout: float = Field(
        10.0, description="Per-HTTP-request timeout in seconds for ROR calls."
    )
    enrich_timeout: float = Field(
        60.0,
        description="Per-paper wall-clock budget in seconds for ROR matching. Strings not "
        "matched in time are left without a match.",
    )
    cache_size: int = Field(
        4096,
        ge=0,
        description="In-process cache size for ROR answers (hits and misses). 0 disables.",
    )


class ResolverOptions(_BibrSettings):
    """Optional external bibr-resolver candidate-search service (tier 3 in the
    enrichment chain). Default-off: when ``url`` is unset, bibr behaves exactly
    as today. Env: ``BIBR_RESOLVER_URL``, ``BIBR_RESOLVER_ENRICH``,
    ``BIBR_RESOLVER_TIMEOUT``, ``BIBR_RESOLVER_LIMIT``,
    ``BIBR_RESOLVER_FALLBACK_SOURCES``."""

    model_config = _section("BIBR_RESOLVER_")

    # Base URL of the resolver service, e.g. http://resolver-host:2010. Unset → disabled.
    url: str | None = Field(
        None,
        description="Base URL of the bibr-resolver service, e.g. http://resolver-host:2010. Unset = disabled.",
    )
    # Master gate: when False, the resolver is skipped even if url is set.
    enrich: bool = Field(
        True,
        description="Master gate for the resolver; when false the resolver is skipped even if url is set.",
    )
    # Per-request timeout (seconds) for single resolver calls (/search, /works, /health).
    timeout: float = Field(
        10.0,
        description="Per-request timeout in seconds for resolver calls (/search, /works, /health).",
    )
    # Max candidates to request from the resolver /search endpoint.
    limit: int = Field(
        20, description="Max candidates to request from the resolver /search endpoint."
    )
    # Max concurrent /search calls when prefetching a reference list's title searches.
    search_concurrency: int = Field(
        8,
        description="Max concurrent /search calls when prefetching a reference list's title searches.",
    )
    # Assert the resolver is backed by the same corpus as CrossRef. When True, a
    # *clean* resolver miss (queried OK, no accepted match) skips the redundant
    # CrossRef fallback — it would only re-query the same data. A resolver *error*
    # (transient transport failure) still falls through to CrossRef.
    authoritative: bool = Field(
        False,
        description="Treat a clean resolver miss (queried OK, no accepted match) as final, skipping "
        "the redundant CrossRef fallback. A resolver error still falls through to CrossRef.",
    )
    # Corpora to query on /search. CrossRef only by default: the resolver serves OpenAlex
    # off separate slow-disk shards (8 indices, ~128 Quickwit splits per query vs 86 for
    # CrossRef — measured ~8x the per-search latency, and worse under concurrency because
    # Quickwit's split-search semaphore is global). Putting it on the primary path costs
    # every reference that penalty to rescue a few; use fallback_sources to pay it only for
    # the references CrossRef missed.
    # Empty → send no `sources`, letting the resolver use its own default tier.
    # NoDecode: keep pydantic-settings from JSON-decoding the env value so the
    # validator below receives the raw comma-separated string (env is "a,b", not JSON).
    sources: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["crossref"],
        description="Resolver corpora to query on /search (comma-separated in env, e.g. "
        '"crossref,openalex"). CrossRef only by default — OpenAlex is ~8x slower per '
        "search, so route it through fallback_sources instead. "
        "Empty = the resolver's own default tier.",
    )
    fallback_sources: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Resolver corpora queried only for references left unmatched by the primary pass.",
    )
    # 4, not more: the fallback pass targets the slow OpenAlex shards, where one query
    # claims ~128 of Quickwit's 300 global split-search permits. Measured, /search starts
    # returning 503 ("resolver backend unavailable") from ~8 concurrent OpenAlex searches,
    # which costs more than the serialization it buys.
    fallback_search_concurrency: int = Field(
        4,
        ge=1,
        description="Max concurrent resolver searches in the unmatched-reference fallback pass.",
    )
    fallback_timeout: float = Field(
        30.0,
        gt=0,
        description="Whole-paper wall-clock deadline in seconds for the resolver fallback pass.",
    )

    @field_validator("sources", "fallback_sources", mode="before")
    @classmethod
    def _split_sources(cls, v):
        """Accept a comma-separated env string or a list; normalize to lowercased tokens."""
        items = v.split(",") if isinstance(v, str) else v
        if isinstance(items, (list, tuple)):
            return [str(s).strip().lower() for s in items if str(s).strip()]
        return v


class CacheOptions(_BibrSettings):
    """Result cache. Env: ``CACHE_ENABLED``, ``CACHE_TTL_SECONDS``, ..."""

    model_config = _section("CACHE_")

    enabled: bool = Field(True, description="Enable the result cache.")
    version: str = Field(
        default_factory=lambda: _compute_code_hash(),
        description="Cache version namespace; invalidates stored entries on change. Defaults to a "
        "hash of the code — override only to pin or force-invalidate manually.",
    )
    ttl_seconds: int = Field(86400, description="Result cache entry TTL in seconds.")
    distributed_singleflight: bool = Field(
        True,
        description="Coalesce identical Redis-backed cache misses across workers. Fail-open.",
    )
    singleflight_wait_seconds: float = Field(
        10.0, ge=0, description="Maximum time a distributed waiter polls for the owner's result."
    )
    singleflight_lease_ttl_seconds: int = Field(
        120, ge=1, description="TTL for a distributed extraction ownership lease."
    )
    singleflight_renew_interval_seconds: float = Field(
        30.0, gt=0, description="Interval between token-safe ownership renewals."
    )
    singleflight_poll_interval_ms: int = Field(
        100, ge=1, description="Base cache polling interval for distributed waiters."
    )
    operation_timeout_seconds: float = Field(
        5.0,
        gt=0,
        description="Upper bound in seconds for one response-cache operation (get, set, lease) "
        "in `bibr serve`. A slower Redis counts as a cache miss instead of stalling the request.",
    )
    # Opt-in disk cache for OCR stage output. When on, re-processing the same
    # PDF (same page range + OCR backend/model) reuses the cached OCR regions
    # from disk instead of re-running the OCR backend — the lever for eval
    # sweeps that re-run identical PDFs under different ref/parse parameters.
    # Off by default: never silently changes results unless opted in.
    # Env: CACHE_OCR.
    ocr: bool = Field(
        False,
        description="Opt-in disk cache for OCR stage output — reuses cached OCR regions for the "
        "same PDF instead of re-running the OCR backend. A complete entry also skips page "
        "rendering, layout detection and native-text analysis, so leave it off when timing "
        "runs. Entries are keyed on the settings, model pins and bibr version, not on source "
        "changes between releases; use a fresh CACHE_OCR_DIR per source revision when "
        "comparing revisions that touch those stages. Off by default.",
    )
    # Directory for the OCR disk cache. None → $XDG_CACHE_HOME/bibr/ocr (else
    # ~/.cache/bibr/ocr). Env: CACHE_OCR_DIR.
    ocr_dir: str | None = Field(
        None,
        description="Directory for the OCR disk cache. Null = $XDG_CACHE_HOME/bibr/ocr "
        "(else ~/.cache/bibr/ocr).",
    )
    # Opt-in disk cache for structured LLM responses, keyed on model + schema +
    # system + user text. A hit costs no tokens and no rate-limit slot. This is
    # also the prefill target for the offline Message Batches path: a batch
    # answers requests at half price and writes them here for a later run to
    # find. Off by default — like the OCR cache, it never silently changes
    # results unless opted in. Env: CACHE_LLM.
    llm: bool = Field(
        False,
        description="Opt-in disk cache for structured LLM responses, keyed on model, schema, "
        "system prompt and user text. Also the prefill target for offline batch runs. Off by "
        "default.",
    )
    # Directory for the LLM response cache. None → $XDG_CACHE_HOME/bibr/llm
    # (else ~/.cache/bibr/llm). Env: CACHE_LLM_DIR.
    llm_dir: str | None = Field(
        None,
        description="Directory for the LLM response cache. Null = $XDG_CACHE_HOME/bibr/llm "
        "(else ~/.cache/bibr/llm).",
    )


class CircuitBreakerOptions(_BibrSettings):
    """Circuit breaker for upstream services. Env: ``CB_FAILURE_THRESHOLD``, ..."""

    model_config = _section("CB_")

    failure_threshold: int = Field(
        5,
        description="Consecutive failures before the circuit breaker opens for an upstream service.",
    )
    reset_timeout_seconds: float = Field(
        30.0, description="Seconds the circuit breaker stays open before allowing a retry probe."
    )
    failure_dedup_window: float = Field(
        2.0,
        description="Seconds within which repeated failures are deduplicated as a single event.",
    )


class AuthOptions(_BibrSettings):
    """HTTP API auth. Env: ``AUTH_API_KEY``."""

    model_config = _section("AUTH_")

    api_key: str | None = Field(
        None,
        description="When set, bibr serve enforces bearer-token auth on every request. "
        "Required in production (ENVIRONMENT=production refuses to start without it).",
    )

    @property
    def required(self) -> bool:
        """True when AUTH_API_KEY is set; server will enforce bearer-token auth."""
        return bool(self.api_key)


class CorsOptions(_BibrSettings):
    """HTTP CORS config. Env: ``CORS_ORIGINS``, ``CORS_ALLOW_CREDENTIALS``, ..."""

    model_config = _section("CORS_")

    origins: Annotated[list[str], NoDecode] = Field(
        [], description="Comma-separated list of allowed CORS origins for production."
    )
    allow_credentials: bool = Field(
        False, description="Allow credentials (cookies/auth headers) in CORS requests."
    )
    allow_methods: Annotated[list[str], NoDecode] = Field(
        ["*"], description="Allowed HTTP methods for CORS requests."
    )
    allow_headers: Annotated[list[str], NoDecode] = Field(
        ["*"], description="Allowed HTTP headers for CORS requests."
    )

    @field_validator("origins", "allow_methods", "allow_headers", mode="before")
    @classmethod
    def _split_lists(cls, v):
        return _split_csv_env(v)


class RedisOptions(_BibrSettings):
    """Redis client. Env: ``REDIS_URL``, ``REDIS_PASSWORD``."""

    model_config = _section("REDIS_")

    password: str | None = Field(
        None,
        description="Redis password. Required in production; injected into redis.url if unset there.",
    )
    url: str | None = Field(
        None,
        description="Redis connection URL. Defaults work with docker-compose; auto-built from password if unset.",
    )
    connect_timeout_seconds: float = Field(
        2.0,
        gt=0,
        description="Seconds to wait for a Redis TCP connection before treating Redis as down.",
    )
    socket_timeout_seconds: float = Field(
        5.0,
        gt=0,
        description="Seconds to wait for a single Redis command reply. A Redis that accepts "
        "connections but never answers fails the command instead of hanging the request.",
    )


class MlOptions(_BibrSettings):
    """Optional local ML models.

    The ``section_classifier_*`` fields are live: they configure the trained
    MiniLM section classifier used by ``structure/section_classifier.py``
    on the default pipeline path (see ``ML_SECTION_CLASSIFIER_MODEL_ID``).
    The ``paper_classifier_*`` fields likewise configure the trained paper-type
    classifier.
    """

    model_config = _section("ML_")

    # Which inference runtime serves bibr's own local models (layout detector,
    # section/paper classifiers, NER parser). "auto" prefers a model's ONNX
    # bundle (core install, onnxruntime) and falls back to the torch classes
    # when the bundle is absent and torch is importable; "onnx"/"torch" force
    # one and fail with a ConfigurationError naming the fix otherwise. See
    # bibr/utils/ml_runtime.py.
    runtime: Literal["auto", "onnx", "torch"] = Field(
        "auto",
        description="Inference runtime for the local models: 'auto' (ONNX bundle when published "
        "or configured, else torch when installed), 'onnx' (require the ONNX bundle), or 'torch' "
        "(require the torch extra).",
    )

    # Trained section classifier (MiniLM two-head, document-context input
    # template). When ``section_classifier_model_id`` is None, classification
    # falls back to the LLM path. Env: ``ML_SECTION_CLASSIFIER_MODEL_ID``,
    # ``ML_SECTION_CLASSIFIER_REVISION``.
    section_classifier_model_id: str | None = Field(
        "scienceverse/bibr-section-classifier",
        description="HF Hub repo id for the trained section classifier. Null falls back to the LLM "
        "classification path.",
    )
    section_classifier_revision: str = Field(
        "ee1a83db01ce947e3a5eb87dfde7221328f4b207",
        description="HF Hub revision (commit SHA or branch) for the section classifier. Pinned "
        "to the audited commit (plus the additive ONNX bundle) so a hub push cannot change "
        "output; 'main' tracks the repo head.",
    )
    # Type-head softmax probability below which a trained-model prediction
    # collapses to UNKNOWN. With 16 classes the uniform baseline is ~0.06, so
    # 0.5 keeps only predictions where the model is meaningfully decisive.
    # Set to 0.0 to disable.
    section_classifier_min_confidence: float = Field(
        0.5,
        description="Type-head softmax probability below which a section-classifier prediction "
        "collapses to UNKNOWN. Set to 0.0 to disable.",
    )
    # When the trained classifier's prediction is UNKNOWN — either collapsed
    # below ``section_classifier_min_confidence`` or confidently predicted as
    # the model's own 'unknown' class — escalate those headers to the batched
    # LLM classifier instead of discarding the header. Fires for every UNKNOWN,
    # so on papers with many topic subheadings most non-alias headings reach
    # the LLM; narrowing that to collapses-only needs a val-set measurement.
    section_classifier_llm_escalation: bool = Field(
        True,
        description="Escalate section headers the trained classifier leaves UNKNOWN "
        "(below-confidence collapses and confident unknown predictions) to the batched LLM "
        "classifier instead of discarding them.",
    )
    # Override the device the classifier runs on. ``None`` picks automatically
    # (CUDA → CPU). MPS is opt-in via ``ML_SECTION_CLASSIFIER_DEVICE=mps`` —
    # benchmarked slower than CPU on this 22M-param model and the chunked
    # workaround for the MPS attention/softmax bug only matches CPU at
    # batch=64 (see ``section_classifier_model._pick_device``).
    section_classifier_device: str | None = Field(
        None,
        description="Device the section classifier runs on. Null = auto (CUDA -> CPU); "
        "MPS is opt-in (benchmarked slower than CPU on this model).",
    )
    # Trained OECD L1/L2 + paper_type multitask classifier (title+abstract
    # input). Mirrors the section_classifier_* fields above. Null falls back
    # to CoreMetadataExtractor._validate_classification's per-paper LLM
    # classification call.
    # Env: ``ML_PAPER_CLASSIFIER_MODEL_ID``, ``ML_PAPER_CLASSIFIER_REVISION``.
    paper_classifier_model_id: str | None = Field(
        "scienceverse/bibr-paper-classifier",
        description="HF Hub repo id for the trained OECD/paper_type multitask classifier. "
        "Set to null to fall back to the existing LLM classification path. Defaults to the "
        "published multitask classifier (all-MiniLM-L6-v2 encoder; OECD L1/L2 + paper_type).",
    )
    paper_classifier_revision: str = Field(
        "6046171b3198a255acb1f07f81a586a32f399ac4",
        description="HF Hub revision (commit SHA or branch) for the paper classifier. Pinned to "
        "the audited commit (plus the additive ONNX bundle) so a hub push cannot change "
        "output; 'main' tracks the repo head.",
    )
    # paper_type-head softmax probability below which a trained-model
    # prediction is escalated to the LLM fallback (see
    # ``paper_classifier_llm_escalation``). OECD L1/L2 predictions are never
    # escalated to an LLM, so this threshold only gates paper_type escalation.
    paper_classifier_min_confidence: float = Field(
        0.5,
        description="paper_type-head softmax probability below which a paper-classifier "
        "prediction is escalated to the LLM fallback (label_paper_type).",
    )
    # When the trained classifier's paper_type prediction falls below
    # ``paper_classifier_min_confidence``, escalate to the batched LLM
    # paper_type labeler instead of keeping the low-confidence prediction.
    paper_classifier_llm_escalation: bool = Field(
        True,
        description="Escalate low-confidence paper_type predictions to the LLM "
        "paper_type labeler instead of keeping them as-is.",
    )
    # OECD L2 (subdomain) is the hardest, most ambiguous head; below this
    # softmax probability the subdomain is emitted as null rather than a
    # low-confidence guess (there is no LLM escalation for OECD). 0.0 disables
    # gating (always emit). L1/paper_type are emitted unconditionally
    # (paper_type has its own LLM escalation).
    paper_classifier_l2_min_confidence: float = Field(
        0.5,
        description="OECD L2 (subdomain) head softmax probability below which the subdomain is "
        "emitted as null instead of a low-confidence label. 0.0 = always emit.",
    )
    paper_classifier_device: str | None = Field(
        None,
        description="Device the paper classifier runs on. Null = auto (CUDA -> CPU).",
    )

    # First-page region-role classifier (bibr/extract/front_role.py): a GBM
    # bundle over the front_role_features contract, trained separately
    # from publisher JATS projected onto OCR regions. Its role scores are
    # evidence for front-matter ownership (title/byline/affiliation/abstract
    # admission, masthead suppression) and for non-English reference headers.
    # Null disables it; front matter then rests on the lexical heuristics alone.
    # Loaded via bibr.ner.checkpoint.resolve_checkpoint (bare repo id gets
    # ":front_role.joblib" appended). scikit-learn and joblib are core deps, so
    # this runs on a torch-free install like every other default-path model.
    front_role_model_id: str | None = Field(
        "scienceverse/bibr-front-role-v1",
        description="HF Hub repo id (or local path) of the first-page region-role classifier "
        "bundle. Null disables the model and rests front matter on the lexical heuristics "
        "alone. TRUST BOUNDARY: deserialized with joblib via the gadget-restricted loader; "
        "only point it at a checkpoint you control.",
    )
    front_role_revision: str = Field(
        "7f01b57e1999d93cb5f17895ed10fdfa27cf6e0b",
        description="Pinned revision for the front-role classifier bundle. Pinned so a hub push "
        "cannot change front-matter output; 'main' tracks the repo head.",
    )
    front_role_enabled: bool = Field(
        True,
        description="Consult the front-role classifier when a model id is configured. False "
        "keeps the model unloaded even when ML_FRONT_ROLE_MODEL_ID is set.",
    )
    # Minimum role probability before a model role counts as evidence in
    # front-matter resolution. 0.5 = the argmax must also be a majority.
    front_role_min_confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum front-role probability for a model role to count as evidence.",
    )
    # A masthead this confident denies the row the right to root a record
    # (journal name / running head / volume line typed as a title by layout).
    front_role_masthead_confidence: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Front-role masthead probability above which a row cannot be a title seed.",
    )
    # A row the heuristics seeded as a title but the classifier confidently
    # types as something else ("Correspondence", "A R T I C L E I N F O",
    # "CITATION", "Key Features" all score heading 1.00) keeps its title role
    # and loses only the right to *root a second record*. See
    # bibr/extract/front_matter.py::_record_title_indices.
    front_role_record_root_confidence: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Front-role probability of a non-title role above which a title seed cannot "
        "root a second front-matter record. 1.0 disables the veto.",
    )
    classifiers_required: bool = Field(
        False,
        description="Fail readiness when a configured local classifier cannot be loaded. "
        "False keeps the existing LLM fallback and reports degraded readiness.",
    )
    paper_classifier_batch_size: int = Field(
        64, ge=1, description="Maximum paper-classifier micro-batch size."
    )
    paper_classifier_batch_timeout_ms: float = Field(
        5.0, ge=0, description="Paper-classifier cross-request coalescing window in milliseconds."
    )
    section_classifier_batch_size: int = Field(
        64, ge=1, description="Maximum section-classifier micro-batch size."
    )
    section_classifier_batch_timeout_ms: float = Field(
        5.0,
        ge=0,
        description="Section-classifier cross-request coalescing window in milliseconds.",
    )
    classifier_vram_safety_reserve_mb: int = Field(
        2048,
        ge=0,
        description="VRAM kept free after managed vLLM and classifier placement.",
    )
    paper_classifier_estimated_peak_mb: int = Field(
        1536,
        ge=0,
        description="Conservative paper-classifier weights plus forward-workspace estimate.",
    )
    section_classifier_estimated_peak_mb: int = Field(
        512,
        ge=0,
        description="Conservative section-classifier weights plus forward-workspace estimate.",
    )


class VllmMlxOptions(_BibrSettings):
    """vLLM-MLX subprocess. Env: ``VLLM_MLX_STARTUP_TIMEOUT``, ``VLLM_MLX_WARMUP_TIMEOUT``."""

    model_config = _section("VLLM_MLX_")

    startup_timeout: int = Field(
        180, description="Startup timeout in seconds for the managed vllm-mlx subprocess."
    )
    # First-inference paging on a 2 GB+ MLX model can take 30-60 s; allow a
    # generous ceiling so the warmup itself doesn't time out before paging
    # finishes. Only paid once per process lifetime.
    warmup_timeout: int = Field(
        300,
        description="Warmup timeout in seconds for the managed vllm-mlx subprocess's first inference "
        "(first-inference paging on a 2 GB+ model can take 30-60s).",
    )


class RapidMlxOptions(_BibrSettings):
    """Rapid-MLX subprocess. Env: ``RAPID_MLX_EXECUTABLE``, ``RAPID_MLX_PREFILL_STEP_SIZE``."""

    model_config = _section("RAPID_MLX_")

    executable: str = Field(
        "rapid-mlx",
        description="rapid-mlx executable path or command name for managed Rapid-MLX servers.",
    )
    startup_timeout: int = Field(
        600, description="Startup timeout in seconds for the managed Rapid-MLX subprocess."
    )
    warmup_timeout: int = Field(
        300, description="Warmup timeout in seconds for the managed Rapid-MLX subprocess."
    )
    prefill_step_size: int = Field(
        8192,
        description="Chunk size passed as --prefill-step-size to managed Rapid-MLX servers.",
    )
    max_num_seqs: int = Field(
        4,
        description="Max concurrent sequences for managed Rapid-MLX servers. Apple Silicon "
        "decode is bandwidth-bound, so batched sequences amortize weight reads and raise "
        "aggregate throughput; Rapid-MLX continuous batching is on by default.",
    )
    max_concurrent_requests: int = Field(
        4, description="Admission cap for managed Rapid-MLX servers."
    )
    spec_decode: str = Field(
        "auto",
        description="Speculative decoding for the managed Rapid-MLX LLM server: 'auto' "
        "(attempt native MTP for Qwen3.5/3.6 models, relaunching without it when the "
        "checkpoint lacks MTP layers), 'mtp' (fail hard if unsupported), or 'none'. The "
        "multimodal OCR server is never launched with speculative decoding.",
    )
    pin_system_prompt: bool = Field(
        True,
        description="Pass --pin-system-prompt so the shared system prompt stays in the "
        "Rapid-MLX prefix cache under memory pressure.",
    )
    force_disk_check: bool = Field(
        True,
        description="Pass --force-disk-check so external HF caches are not rejected by the "
        "current filesystem's free-space check.",
    )
    hf_home: str | None = Field(
        None,
        description="HF_HOME override for managed Rapid-MLX subprocesses, useful for external "
        "model-cache drives.",
    )
    hf_hub_cache: str | None = Field(
        None,
        description="HF_HUB_CACHE override for managed Rapid-MLX subprocesses.",
    )
    home: str | None = Field(
        None,
        description="HOME override for managed Rapid-MLX subprocesses; Rapid-MLX stores some "
        "runtime files under the home directory.",
    )


class PipelineOptions(_BibrSettings):
    """Pipeline orchestration. Env: ``PIPELINE_TIMEOUT``, ``PIPELINE_MEMORY_MODE``, ..."""

    model_config = _section("PIPELINE_")

    timeout: int = Field(300, description="Pipeline timeout in seconds.")
    bibr_sha: str | None = Field(
        None,
        description="Deployed bibr commit SHA surfaced in qualification_provenance "
        "(falls back to BIBR_BUILD_SHA when unset).",
    )
    platform_sha: str | None = Field(
        None,
        description="Deployed serving-platform commit SHA surfaced in qualification_provenance.",
    )
    # Default memory mode when no --memory flag / LocalPipeline arg is given;
    # None uses system RAM and CUDA VRAM to select a memory mode.
    memory_mode: Literal["aggressive", "balanced", "keep_all"] | None = Field(
        None,
        description="Default memory mode when no --memory flag / LocalPipeline arg is given: "
        '"aggressive", "balanced", or "keep_all". Null selects aggressive on systems '
        "with <=8 GB RAM or CUDA GPUs with <=8 GB VRAM; otherwise balanced.",
    )
    integrity_statement_mode: Literal["legacy", "shadow", "active"] = Field(
        "shadow",
        description="Research-integrity statement resolver rollout mode. Shadow preserves "
        "compatibility scalars while emitting typed comparison evidence.",
    )
    # Experimental: proximity alone does not establish the correct title in
    # multilingual front matter. Keep off until validated for the target inputs.
    title_prefer_byline_adjacent: bool = Field(
        False,
        description="Prefer the printed title row directly above the byline over a disagreeing "
        "model title (multilingual front matter). Experimental; validate on representative inputs.",
    )
    # There is deliberately no workers-per-device setting: bibr serve pins one
    # inference worker (see bibr/serve/app.py). Concurrency comes from the
    # LitServe async loop and the layout/segmenter GpuBatcher; a second worker
    # would duplicate the models (~1.5 GB RSS each), split the batcher, and
    # parallelize only GIL-bound Python, since the heavy stages already use
    # every core from one process.
    deployment_ready_timeout: int = Field(120, description="Deployment ready timeout in seconds.")
    max_file_size: int = Field(50 * 1024 * 1024, description="Max upload file size in bytes.")
    multipart_overhead_bytes: int = Field(
        1024 * 1024,
        ge=0,
        description="Bytes reserved above max_file_size for multipart boundaries and headers.",
    )
    restart_workers: bool = Field(
        False,
        description="Opt in to LitServe worker replacement after a critical worker-loop failure. "
        "The safe default is fail-stop because LitServe 0.2.x cannot reliably notify an API "
        "process whose in-flight request belonged to the dead worker.",
    )
    upload_spool_memory_bytes: int = Field(
        1024 * 1024,
        ge=0,
        description="Upload bytes retained in memory before the bounded spool rolls to disk.",
    )
    max_pages: int = Field(
        200,
        ge=1,
        description="Hard maximum pages processed per file. Requests beyond this range are capped "
        "to prevent compact many-page PDFs from exhausting render memory.",
    )
    max_concurrent_post_parse: int = Field(
        4, description="Max concurrent post-parse tasks per request."
    )
    # Cloud-LLM only: with a remote LLM there is no OCR->LLM VRAM handoff, so
    # each OCR window's back half (parse -> extract -> enrich -> export) can
    # start as soon as that window's OCR completes, hiding the LLM/Crossref
    # tail under subsequent windows' OCR. Managed local LLM backends and
    # aggressive memory mode always keep the stage barrier regardless.
    stream_backhalf: bool = Field(
        True,
        description="Overlap each OCR window's back half (parse/extract/enrich/export) with "
        "subsequent windows' OCR. Only applies to the cloud-LLM path (LLM_BACKEND=cloud, memory "
        "mode not aggressive); managed local LLM backends keep the stage barrier. Escape hatch "
        "for debugging.",
    )
    # Max requests running the pipeline concurrently per worker (0 = unlimited).
    # The LitServe async loop dispatches unbounded concurrent predicts; this
    # bounds peak host RAM (page images) under an upload flood. GPU peak VRAM is
    # bounded separately by the layout/segmenter GpuBatcher.
    max_inflight_requests: int = Field(
        8,
        description="Max requests running the pipeline concurrently per worker (0 = unlimited). "
        "Bounds peak host RAM under an upload flood.",
    )
    max_active_uploads: int = Field(
        8,
        ge=1,
        description="Maximum concurrent upload requests admitted by the API server before "
        "multipart parsing (excess requests receive HTTP 429).",
    )
    # Per-paper text-quality score (report-only): rates every region's text for
    # extraction garbage (mojibake, replacement chars, fragmented/spaced-out
    # words), aggregates the 10th percentile per page, and exports a paper-level
    # scalar plus a processing warning below the threshold. Never changes
    # extraction behavior.
    text_quality_report: bool = Field(
        True,
        description="Compute and export a per-paper text-quality score (report-only; "
        "10th-percentile aggregation of per-region garbage/fragmentation ratings).",
    )
    text_quality_warn_threshold: float = Field(
        0.5,
        description="Text-quality score below which a processing warning is attached to the paper.",
    )
    # Use the PDF's embedded outline (bookmarks) as a heading-level signal:
    # outline entries are fuzzy-matched to detected section headers and, on a
    # confident match, override numbering-inferred levels. Ships dark: flip
    # after an eval gate.
    outline_headings: bool = Field(
        False,
        description="Use the PDF outline (bookmarks) as a heading-level signal for section "
        "hierarchy. Ships dark pending an eval gate.",
    )


class JobsOptions(_BibrSettings):
    """Async job API (serve). Env: ``JOBS_ENABLED``, ``JOBS_TTL_SECONDS``, ``JOBS_MAX_ACTIVE``,
    ``JOBS_MAX_RUNNING``, ``JOBS_MAX_RETAINED``, ``JOBS_MAX_RETAINED_BYTES``, ``JOBS_STORE``,
    ``JOBS_REDIS_URL``, ``JOBS_KEY_PREFIX``, ``JOBS_REPLICA_ID``.

    By default jobs are held in an in-process store on the single HTTP API-server
    process (``serve.app.main`` always pins ``num_api_servers=1`` because upload
    ownership, dispatch tracking, and readiness state are process-local).
    ``JOBS_STORE=redis`` moves job status, results, and the active-job cap into Redis
    so several ``bibr serve`` replicas behind one load balancer answer status/result
    polls for each other's jobs; uploads and execution stay on the replica that
    accepted the upload. Nothing here affects extraction output (see
    ``_FINGERPRINT_EXCLUDED_SECTIONS``).
    """

    model_config = _section("JOBS_")

    enabled: bool = Field(True, description="Enable the async job API (serve).")
    store: Literal["memory", "redis"] = Field(
        "memory",
        description="Where job status and results live: 'memory' (one replica; lost on "
        "restart) or 'redis' (shared by every replica pointed at the same Redis, so the "
        "active-job cap is global and any replica can answer status/result polls). "
        "Uploads and execution always stay on the replica that received the upload.",
    )
    redis_url: str | None = Field(
        None,
        description="Redis URL for JOBS_STORE=redis. Falls back to REDIS_URL (the cache's "
        "Redis) when unset; startup fails if neither is set.",
    )
    key_prefix: str = Field(
        "bibr:jobs",
        description="Key prefix for the Redis job store. Every replica sharing one job "
        "namespace must use the same prefix; change it to isolate deployments that share "
        "a Redis.",
    )
    replica_id: str | None = Field(
        None,
        description="Identifier of this bibr serve replica, recorded on each job it "
        "executes and reported as `replica` in job status. Defaults to `<hostname>:<pid>`.",
    )
    ttl_seconds: int = Field(3600, description="TTL in seconds for completed job records.")
    max_active: int = Field(32, description="Max admitted queued plus running jobs.")
    max_running: int = Field(
        2,
        ge=1,
        description="Max job inference descriptors dispatched concurrently.",
    )
    max_retained: int = Field(
        128,
        ge=0,
        description="Max completed job results retained (in the process, or in Redis across "
        "every replica); oldest results are evicted.",
    )
    max_retained_bytes: int = Field(
        256 * 1024 * 1024,
        ge=0,
        description="Byte budget for retained job results (their encoded JSON bodies; the "
        "Redis store charges the same encoded size while holding the body compressed). "
        "Oldest results are evicted until the rest fit; the newest result is always kept so "
        "that an export larger than the budget can still be fetched once. 0 disables the "
        "budget (count-only retention).",
    )


class MeteringOptions(_BibrSettings):
    """Per-request usage metering (serve). Env: ``METER_ENABLED``, ``METER_LOG_PATH``.

    When ``log_path`` is set, metering records (one JSON line per request and per
    extraction) are also written as JSONL to that file. Purely observational —
    does not affect extraction output.
    """

    model_config = _section("METER_")

    enabled: bool = Field(True, description="Enable per-request usage metering (serve).")
    log_path: str | None = Field(
        None,
        description="Path to write metering records as JSONL (one line per request/extraction).",
    )
    log_max_bytes: int = Field(
        100 * 1024 * 1024,
        ge=0,
        description="Rotate the metering JSONL at this size (bytes). The metering middleware "
        "sits outside the auth gate, so unauthenticated request spam would otherwise grow the "
        "log without bound and exhaust disk. 0 disables rotation (unbounded).",
    )
    log_backup_count: int = Field(
        3,
        ge=0,
        description="Number of rotated metering-log backups to keep.",
    )


class McpOptions(_BibrSettings):
    """MCP endpoint on the serve app. Env: ``MCP_ENABLED``.

    When enabled (and the ``mcp`` extra is installed), ``bibr serve`` mounts a
    Model Context Protocol endpoint at ``/mcp`` (streamable HTTP) exposing the
    same chew-then-query tool surface as ``bibr mcp``, with extraction routed
    through the regular serve inference dispatch. Gated by the same bearer
    auth as every other route. Serve-only and purely additive — does not
    affect extraction output (see ``_FINGERPRINT_EXCLUDED_SECTIONS``).
    """

    model_config = _section("MCP_")

    enabled: bool = Field(
        False,
        description="Mount the MCP endpoint at /mcp on bibr serve (requires the mcp extra).",
    )
    max_papers_per_session: int = Field(
        16,
        ge=1,
        description="Chewed papers retained in memory per MCP client session; the oldest "
        "is evicted beyond this.",
    )
    session_idle_timeout_seconds: float = Field(
        1800.0,
        ge=0,
        description="Seconds an MCP client session may sit idle before the server closes it "
        "and drops its papers. A client that disconnects without DELETE would otherwise pin "
        "its session — and up to max_papers_per_session full exports — for the process "
        "lifetime. 0 disables the timeout.",
    )
    chew_url_enabled: bool = Field(
        True,
        description="Expose the chew_url tool on the serve MCP endpoint: a server-side, "
        "SSRF-guarded download of a public https:// URL routed into extraction. Disable "
        "to keep the endpoint free of outbound fetches.",
    )
    url_allowed_hosts: Annotated[list[str], NoDecode] = Field(
        [],
        description="Restrict chew_url downloads to these hosts (subdomains included, "
        "e.g. 'arxiv.org' admits 'export.arxiv.org'). Empty = any public host.",
    )

    @field_validator("url_allowed_hosts", mode="before")
    @classmethod
    def _split_hosts(cls, v):
        # Hostnames are case-insensitive; the SSRF guard compares lowercased.
        return _split_csv_env(v, lower=True)


class GlobalSettings(_BibrSettings):
    """Application settings loaded from ``.env`` / environment.

    Each section is its own ``BaseSettings`` subclass with a short env prefix, so
    env vars read naturally: ``LLM_PROVIDER``, ``OCR_BACKEND``, ``CROSSREF_API_EMAIL``,
    ``REDIS_URL``, etc. No global ``BIBR_`` prefix, no ``__`` delimiters.

    Top-level, un-clustered vars (``ENVIRONMENT``, ``WTPSPLIT_MODEL``,
    ``WTPSPLIT_THRESHOLD``, ``WTPSPLIT_BLOCK_SIZE``, ``WTPSPLIT_STRIDE``, API keys, ...)
    are read bare.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Top-level survivors (not clustered into any sub-model)
    ENVIRONMENT: str = Field(
        "development",
        description='Runtime environment: "development" or "production". '
        "In production mode, redis.password and auth.api_key are required.",
    )
    BIBR_BUILD_SHA: str | None = Field(
        None,
        pattern=r"^[0-9a-f]{40}$",
        description="Exact lowercase Git commit deployed by the serving environment.",
    )
    SERVE_LOG_LEVEL: str = Field(
        "info",
        description="Log level for bibr serve's own loggers (bibr.*), also handed to uvicorn "
        "and LitServe: debug, info, warning or error. Metering records (METER_ENABLED) are "
        "emitted regardless of this level.",
    )

    @field_validator("SERVE_LOG_LEVEL", mode="before")
    @classmethod
    def _normalize_serve_log_level(cls, value):
        level = str(value).strip().lower()
        if level not in ("debug", "info", "warning", "error"):
            raise ValueError("SERVE_LOG_LEVEL must be one of debug, info, warning, error")
        return level

    WTPSPLIT_MODEL: str = Field("sat-6l-sm", description="WtP-split sentence segmentation model.")
    WTPSPLIT_MODEL_REVISION: str | None = Field(
        None,
        description="HF Hub revision for a Hub-hosted wtpsplit model. Unset pins the default "
        "sat-6l-sm to its audited commit and loads other Hub models from main; local bundles "
        "carry their own manifest revision.",
    )
    WTPSPLIT_THRESHOLD: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Optional explicit wtpsplit sentence-boundary threshold.",
    )
    WTPSPLIT_BLOCK_SIZE: int | None = Field(
        None,
        gt=0,
        description="Optional explicit wtpsplit inference block size.",
    )
    WTPSPLIT_STRIDE: int | None = Field(
        None,
        gt=0,
        description="Optional explicit wtpsplit inference stride.",
    )
    EQUATION_EXTRACTION: bool = Field(True, description="Extract equations from OCR output.")
    EQUATION_EXTRACTION_TIMEOUT_SECONDS: int = Field(
        90, description="Timeout in seconds for equation extraction."
    )
    EQUATION_LLM_FALLBACK_MIN_REGEX_STATS: int = Field(
        0,
        description=(
            "Opt-in cost gate for the LLM equation fallback. When > 0, the "
            "fallback runs only on papers whose regex pass already found at "
            "least this many non-LaTeX statistical components — a paper-level "
            "proxy for 'this paper reports statistics'. 0 (default) always "
            "runs the fallback. Measured on a 149-paper run: threshold 1 cut "
            "~20% of fallback calls at ~95% recall of LLM-only stat components."
        ),
    )
    IMPLICIT_SECTION_DETECTION: bool = Field(
        True, description="Infer implicit sections from body text when headers are absent."
    )
    FIGURE_IMAGES: bool = Field(False, description="Extract figure images from PDFs.")
    # Reference extraction strategy — DECOUPLED segmentation + parsing axes
    # (resolved in bibr.extract.ref_extractor._resolve_ref_strategies; per-run
    # overrides ride RunConfig via chew(refs=...) / CLI --refs/--ref-seg).
    # The local ModernBERT-CRF parser ("ner") is the default: reference parsing
    # is the bulk of bibr's LLM tokens, so doing it locally cuts most of the
    # per-paper cost. To use LLM field parsing instead, set parse=llm (CLI
    # `--refs llm`). Segmentation defaults to the local geometry GBM ("geom"),
    # which cascades region anchors → LLM → CRF when geometry is absent or
    # unconfident — val-86 paired judge GEOM 13 / llm 9 / tie 6, aggregate F1
    # >= llm on all domains, ~free. REF_SEG_STRATEGY=region makes region
    # anchors the primary tier (still LLM → CRF fallback); REF_SEG_LLM_FALLBACK
    # =false removes the LLM tier from the cascade entirely. Set
    # REF_SEG_STRATEGY=llm (CLI `--ref-seg llm`) to force the LLM segmenter.
    REF_SEG_STRATEGY: Literal["llm", "crf", "geom", "region"] | None = Field(
        "geom",
        description='Reference-segmentation strategy: "geom" (default; local geometry GBM, cascades '
        'region anchors -> LLM -> CRF), "region" (zero-cost layout-region anchors as primary tier), '
        '"llm" (force the LLM segmenter), or "crf" (ModernBERT+CRF segmenter).',
    )
    REF_PARSE_STRATEGY: Literal["llm", "ner", "llm-chunked", "off"] | None = Field(
        "ner",
        description='Reference-parsing strategy: "ner" (default; local ModernBERT-CRF parser, no '
        'per-reference LLM cost), "llm" (batched LLM), "llm-chunked" (chunk-tolerant '
        'LLM parse over region-aligned chunks), or "off" (disable reference extraction entirely).',
    )
    # "off" disables reference extraction entirely (no segmentation, no
    # parsing, empty bib/bib_match/xref tables) while keeping core metadata,
    # sections, and equations. --no-llm also skips metadata LLM calls and equations.
    # llm-chunked: chunk-tolerant LLM parse over region-aligned chunks (no
    # numbered-list contract) — seg-error-immune on the generative path since
    # the model finds its own reference boundaries. Opt-in pending judge gate.
    # References per LLM parse call. Each call carries ~1.7k fixed input
    # tokens (schema + instructions), so larger batches cut per-paper cost
    # near-proportionally; 15 ≈ -35% total ref tokens vs 5. Sample-validated
    # 2026-06-12 (3 gold-val papers, metrics unchanged); full judge re-gate
    # pending.
    REF_PARSE_BATCH_SIZE: int = Field(
        15,
        description="References per LLM parse call. Larger batches cut per-paper token cost "
        "near-proportionally (each call carries ~1.7k fixed schema/instruction tokens).",
    )
    # Output-token cap for a single ref-parse batch (overrides the global
    # LLM_MAX_TOKENS, which also sizes the much larger metadata calls). A
    # 15-ref batch emits ~1.5-4k JSON tokens, so 8192 is ample headroom — its
    # real job is to bound a *degenerate* generation: a small model under greedy
    # decode can loop on repetitive input (e.g. APA-ellipsis author lists) and
    # run to the cap. Keeping the cap below the per-request hard timeout makes
    # such a batch return as a truncated `IncompleteOutputException` (fast,
    # detectable → NER fallback) instead of burning the whole pipeline budget on
    # a timeout. See `_parse_references_llm`.
    REF_PARSE_MAX_TOKENS: int = Field(
        8192,
        description="Output-token cap for a single LLM ref-parse batch, overriding LLM_MAX_TOKENS "
        "for this call. Bounds degenerate generations to a fast, detectable truncation.",
    )
    # Local geometry reference segmenter (REF_SEG_STRATEGY=geom). Frozen GBM
    # bundle in a public HF repo, loaded via bibr.ner.checkpoint.resolve_checkpoint
    # (the bare repo id gets ":geom_segmenter.joblib" appended). Needs the ml extra.
    REF_GEOM_SEG_MODEL_ID: str = Field(
        "scienceverse/bibr-geom-segmenter-v1",
        description="HF Hub repo id for the local geometry reference segmenter (REF_SEG_STRATEGY=geom). "
        "Needs the ml extra. TRUST BOUNDARY: this is deserialized with joblib (executes code on "
        "load); a local path here is loaded via the gadget-restricted loader but must still be a "
        "checkpoint you control — never point it at an untrusted file.",
    )
    # Keep the pinned bundle compatible with geom_adjacent_features patterns.
    REF_GEOM_SEG_REVISION: str = Field(
        "4d1702e2c766d96c8887bd4b30ef56637aa9b32c",
        description="Pinned commit revision for the geometry reference segmenter bundle.",
    )
    # Per-paper geometry confidence below which segmentation cascades to the LLM.
    REF_GEOM_SEG_CASCADE_THRESHOLD: float = Field(
        0.90,
        description="Per-paper geometry confidence below which reference segmentation cascades to the LLM.",
    )
    # A decisive boundary classifier can still lose anchors during alignment.
    # Decline when too few labeled boundaries survive, regardless of confidence.
    REF_GEOM_MIN_ALIGN_YIELD: float = Field(
        0.5,
        description="Alignment-yield (aligned/labeled boundaries) below which the geometry segmenter "
        "declines regardless of confidence, cascading to the next tier.",
    )
    # Split merged references before parsing, while preserving citations that
    # belong inside a reference title (such as reply/comment titles).
    REF_SPLIT_MERGED_REFS: bool = Field(
        True,
        description="Split a reference string containing a second author-date onset before parsing "
        "(segmenter-agnostic, post-segmentation merged-reference splitter).",
    )
    # Layout-region anchor segmentation, the zero-cost fallback tier between
    # LLM seg and the CRF last resort: reference_content region onsets are
    # filtered to genuine ref starts and snapped onto the ref text. Occupies
    # only the failure slot (LLM timeout/0-spans), where CRF-or-nothing used
    # to be the remaining option; also covers scanned PDFs with no text layer.
    REF_SEG_REGION_ANCHORS: bool = Field(
        True,
        description="Enable layout-region anchor segmentation, the zero-cost fallback tier between "
        "geom/LLM segmentation and the CRF last resort.",
    )
    # Allow the token-costly LLM anchor-emit tier in the segmentation fallback
    # chain (geom decline → region anchors → LLM → CRF). Regions cover nearly
    # all PDFs, so the LLM tier now mostly serves DOCX (no layout regions) and
    # no-`ml` installs. Off = skip straight to the local CRF/marker-split
    # tiers. An explicit REF_SEG_STRATEGY=llm is unaffected.
    REF_SEG_LLM_FALLBACK: bool = Field(
        True,
        description="Allow the token-costly LLM tier in the reference-segmentation fallback chain "
        "(geom decline -> region anchors -> LLM -> CRF). Off skips straight to CRF/marker-split.",
    )
    # Compare region segments with an independent source-record count to catch
    # missing anchors. Keep the region result as a reserve if LLM recovery fails.
    REF_SEG_MIN_SOURCE_RECALL: float = Field(
        0.6,
        description="Fraction of reference-section source records the region tier must recover as "
        "segments; below it, escalate to the LLM tier instead of accepting the segmentation. "
        "0 disables the backstop.",
    )
    REF_TRAINING_DATA_DIR: str | None = Field(
        None,
        description="Directory to save raw bibliography text + LLM extracts as JSON pairs for "
        "reference-extraction model training. Unset disables capture.",
    )
    # NER checkpoints — accept either a local path or an HF Hub repo id
    # (optionally suffixed ``:filename.pt``). Override to pin a local build or
    # a forked repo.
    NER_SEG_CKPT: str = Field(
        "scienceverse/bibr-segmenter-v1",
        description="Local path or HF Hub repo id for the NER reference-segmenter checkpoint "
        "(optionally suffixed :filename.pt).",
    )
    # Parser default is the v4.5 gold-trained feature-gated checkpoint (0.9656
    # strict micro entity F1 on the gold val split vs 0.0544 for giant-v4, whose
    # span conventions are OOD). The parser path (RefParser →
    # FeatureGatedEncoderCRF, add_special_tokens=False) requires a v4-format
    # checkpoint; the legacy v1 parser is no longer loadable here.
    NER_PARSER_CKPT: str = Field(
        "scienceverse/bibr-parser-v4-5-gold",
        description="Local path or HF Hub repo id for the NER reference-parser checkpoint "
        "(optionally suffixed :filename.pt). Requires a v4-format checkpoint.",
    )
    # Pin each checkpoint to an immutable commit SHA. This makes loads
    # reproducible and lets ``hf_hub_download`` serve the cached snapshot with no
    # network round-trip (a SHA revision skips the etag check). Set to ``"main"``
    # to track the latest revision instead. See bibr/ner/checkpoint.py.
    NER_SEG_REVISION: str = Field(
        "344f27d851ffb901b2659629ac879f10b6eba234",
        description='Pinned commit revision for the NER segmenter checkpoint. Set to "main" to '
        "track the latest revision instead.",
    )
    NER_PARSER_REVISION: str = Field(
        "ff50a83e7f5b73dcf6f8f973a1a6e3847ec429e6",
        description='Pinned commit revision for the NER parser checkpoint. Set to "main" to track '
        "the latest revision instead.",
    )
    NER_DEVICE: str | None = Field(
        None, description="Device for NER checkpoints: cpu, cuda, or mps (auto-detected if unset)."
    )
    OCR_BASE_URL: str = Field(
        "http://localhost:8080",
        description="OCR server base URL (Docker Compose overrides this with its private bibr-ocr host).",
    )
    SEGMENTER_SUB_BATCH_SIZE: int = Field(
        32, description="Sub-batch size for sentence segmentation."
    )
    SEGMENTER_GPU_MEM_LIMIT_MB: int = Field(
        0, description="GPU memory limit in MB for the sentence segmenter (0 = unlimited)."
    )
    # Coalescing window (ms) for the serve segmenter's GpuBatcher — gathers
    # texts from concurrent requests into one wtpsplit call. Mirrors
    # LAYOUT_BATCH_TIMEOUT_MS; 0 = no coalescing wait (admission-gate only).
    SEGMENTER_BATCH_TIMEOUT_MS: int = Field(
        5,
        description="Coalescing window in milliseconds for the serve segmenter's GpuBatcher "
        "(0 = no coalescing wait, admission-gate only).",
    )
    # Run the serve sentence segmenter on GPU. None = auto-detect (use CUDA when
    # onnxruntime exposes it). CPU segmentation is NOT cheap on large papers
    # (~197s for a 137k-char doc vs ~0.7s on GPU — the dominant cost of the
    # pipeline timeout), so the accelerator is used by default. Force CPU with
    # SEGMENTER_USE_GPU=false only on VRAM-tight boxes that co-locate OCR.
    SEGMENTER_USE_GPU: bool | None = Field(
        None,
        description="Run the serve sentence segmenter on GPU. Null = auto-detect (CUDA when "
        "onnxruntime exposes it). Force CPU only on VRAM-tight boxes that co-locate OCR.",
    )
    GOOGLE_API_KEY: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY", "LANGEXTRACT_API_KEY"),
        description="Google AI API key, for LLM_PROVIDER=google. Legacy aliases GEMINI_API_KEY "
        "and LANGEXTRACT_API_KEY also accepted.",
    )
    ANTHROPIC_API_KEY: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
        description="Anthropic API key, for LLM_PROVIDER=anthropic. Alias CLAUDE_API_KEY also accepted.",
    )
    GROQ_API_KEY: str | None = Field(
        default=None,
        validation_alias="GROQ_API_KEY",
        description="Groq API key, for LLM_PROVIDER=groq.",
    )

    # Sub-sections — each reads its own env prefix on instantiation.
    llm: LlmOptions = Field(default_factory=LlmOptions)
    ocr: OcrOptions = Field(default_factory=OcrOptions)
    ocr_vision: OcrVisionOptions = Field(default_factory=OcrVisionOptions)
    fig: FigureOptions = Field(default_factory=FigureOptions)
    layout: LayoutOptions = Field(default_factory=LayoutOptions)
    crossref: CrossrefOptions = Field(default_factory=CrossrefOptions)
    resolver: ResolverOptions = Field(default_factory=ResolverOptions)
    ror: RorOptions = Field(default_factory=RorOptions)
    cache: CacheOptions = Field(default_factory=CacheOptions)
    cb: CircuitBreakerOptions = Field(default_factory=CircuitBreakerOptions)
    cors: CorsOptions = Field(default_factory=CorsOptions)
    redis: RedisOptions = Field(default_factory=RedisOptions)
    ml: MlOptions = Field(default_factory=MlOptions)
    vllm_mlx: VllmMlxOptions = Field(default_factory=VllmMlxOptions)
    rapid_mlx: RapidMlxOptions = Field(default_factory=RapidMlxOptions)
    pipeline: PipelineOptions = Field(default_factory=PipelineOptions)
    auth: AuthOptions = Field(default_factory=AuthOptions)
    jobs: JobsOptions = Field(default_factory=JobsOptions)
    metering: MeteringOptions = Field(default_factory=MeteringOptions)
    mcp: McpOptions = Field(default_factory=McpOptions)

    # ------------------------------------------------------------------
    # Validators (rewritten against nested form)
    # ------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def empty_str_to_none(cls, data):
        """Treat empty-string env values as None / defaults for top-level fields."""
        if isinstance(data, dict):
            for k, v in list(data.items()):
                if v == "":
                    data[k] = None
        return data

    @field_validator("REF_SEG_STRATEGY", "REF_PARSE_STRATEGY", mode="before")
    @classmethod
    def lower_ref_strategy(cls, v):
        """Normalize strategy values so the Literal check is case-insensitive."""
        return v.lower() if isinstance(v, str) else v

    @model_validator(mode="after")
    def validate_wtpsplit_windowing(self) -> "GlobalSettings":
        """Require explicit sentence-segmenter windowing to be a valid pair."""
        block_size = self.WTPSPLIT_BLOCK_SIZE
        stride = self.WTPSPLIT_STRIDE
        if (block_size is None) != (stride is None):
            raise ValueError("WTPSPLIT_BLOCK_SIZE and WTPSPLIT_STRIDE must be set together")
        if block_size is not None and stride is not None and stride > block_size:
            raise ValueError("WTPSPLIT_STRIDE cannot exceed WTPSPLIT_BLOCK_SIZE")
        return self

    @model_validator(mode="after")
    def compute_ocr_concurrency(self) -> "GlobalSettings":
        """Auto-compute OCR concurrency knobs from platform/GPU count.

        On Apple Silicon, vision-prefill requests can serialize on the GPU.
        Limit client-side concurrency to avoid extending queued requests
        toward their read timeout. Default both knobs to 1 there:

        - ``max_concurrent_regions`` (used by the remote/serve path)
        - ``concurrent_regions_per_file`` (used by the local-CLI path)

        Override with ``OCR_MAX_CONCURRENT_REGIONS`` /
        ``OCR_CONCURRENT_REGIONS_PER_FILE`` if you have a non-MLX backend
        target (e.g. running ``--ocr-url`` against a remote vLLM server).
        """
        import platform
        import sys

        is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"

        # Snapshot which fields the USER provided before the auto-tune
        # assignments below add them to ``model_fields_set`` (pydantic v2
        # marks assigned fields as set). See ``OcrOptions.user_set_concurrency``.
        self.ocr._user_set_concurrency = frozenset(
            field
            for field in ("max_concurrent_regions", "concurrent_regions_per_file")
            if field in self.ocr.model_fields_set
        )

        if "max_concurrent_regions" not in self.ocr.model_fields_set:
            self.ocr.max_concurrent_regions = 1 if is_apple_silicon else 16 * self.ocr.local_gpus
        if is_apple_silicon and "concurrent_regions_per_file" not in self.ocr.model_fields_set:
            self.ocr.concurrent_regions_per_file = 1
        return self

    @model_validator(mode="after")
    def adjust_ollama_rate_limit(self) -> "GlobalSettings":
        """Auto-lower LLM rate limit for Ollama if not explicitly set."""
        import logging

        logger = logging.getLogger("bibr.config")
        if self.llm.provider == "ollama" and "rate_limit_rpm" not in self.llm.model_fields_set:
            self.llm.rate_limit_rpm = 10
            logger.info("Ollama LLM provider detected — auto-lowering llm.rate_limit_rpm to 10")
        return self

    @model_validator(mode="after")
    def raise_enrich_concurrency_for_resolver(self) -> "GlobalSettings":
        """With the bibr-resolver enabled, most reference lookups resolve against the
        fast local resolver instead of CrossRef. But ``enrich_references`` gates every
        lookup — resolver included — on ``crossref.enrich_concurrency`` (default 3, sized
        for CrossRef politeness), needlessly serializing the resolver path. Raise it when
        the resolver is configured and the user hasn't set it explicitly; the CrossRef
        rate limiter independently paces the now-rare CrossRef fall-through, so politeness
        is unaffected."""
        import logging

        logger = logging.getLogger("bibr.config")
        if self.resolver.url and "enrich_concurrency" not in self.crossref.model_fields_set:
            self.crossref.enrich_concurrency = 16
            logger.info(
                "bibr-resolver enabled — raising crossref.enrich_concurrency 3->16 "
                "(CrossRef rate limiter still paces fall-through calls)"
            )
        return self

    @model_validator(mode="after")
    def set_redis_url(self) -> "GlobalSettings":
        """Set a default redis.url if not provided, using redis.password."""
        import logging

        logger = logging.getLogger("bibr.config")

        if not self.redis.url:
            if self.redis.password:
                self.redis.url = f"redis://:{_url_quote(self.redis.password, safe='')}@redis:6379/0"
            # else: leave self.redis.url as None — cache/limiter callers
            #       must check before initializing.
        elif self.redis.password:
            try:
                parsed = urlparse(self.redis.url)
                if parsed.password is None and "@" not in parsed.netloc:
                    new_netloc = f":{_url_quote(self.redis.password, safe='')}@{parsed.netloc}"
                    self.redis.url = urlunparse(parsed._replace(netloc=new_netloc))
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to inject redis.password into redis.url: {e}")
        return self


# Sections that never affect extraction output — connection plumbing, auth,
# and the cache's own keys. Everything else is included: over-invalidating
# the cache on an irrelevant knob costs one recompute; under-invalidating
# serves a stale result silently.
_FINGERPRINT_EXCLUDED_SECTIONS = frozenset(
    {"redis", "auth", "cors", "cache", "cb", "jobs", "metering", "mcp"}
)


# Excluded so credential rotation doesn't wipe the cache. Covers both nested
# fields (``api_key``/``password``) and top-level un-sectioned secrets whose
# names don't match a section (``GOOGLE_API_KEY``/``ANTHROPIC_API_KEY``).
def _is_secret_fingerprint_key(name: str) -> bool:
    return _is_secret_name(name)


def compute_behavior_fingerprint(settings: "GlobalSettings") -> str:
    """Short stable hash of every behavior-affecting setting.

    Folded into the serve response-cache namespace (see :func:`cache_namespace`)
    so changing e.g. ``LLM_MODEL`` or ``REF_PARSE_STRATEGY`` and restarting can
    never serve JSON produced under the old configuration.
    """
    import json

    def _scrub(value):
        if isinstance(value, dict):
            return {k: _scrub(v) for k, v in value.items() if not _is_secret_fingerprint_key(k)}
        return value

    dump = settings.model_dump(mode="json")
    scrubbed = {
        k: _scrub(v)
        for k, v in dump.items()
        if k not in _FINGERPRINT_EXCLUDED_SECTIONS and not _is_secret_fingerprint_key(k)
    }
    payload = json.dumps(scrubbed, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def cache_namespace(settings: "GlobalSettings") -> str:
    """Redis prefix for the serve response cache.

    Composes the code-hash ``cache.version`` (invalidates on deploys) with the
    behavior fingerprint (invalidates on config changes).
    """
    return f"bibr:{settings.cache.version}:{compute_behavior_fingerprint(settings)}"


def validate_production_settings(settings: "GlobalSettings") -> None:
    """Fail-fast hardening checks for production serve deployments.

    Called from ``bibr.serve.app.build_server()`` — deliberately NOT run at
    settings construction, so ``import bibr.config`` (and with it ``bibr
    --help`` / ``bibr setup``, the tools that would fix the config) still
    works on a production-flagged box with incomplete env.
    """
    if settings.ENVIRONMENT != "production":
        return
    if not settings.redis.password:
        raise ValueError("redis.password must be set in production environment")
    if not settings.auth.api_key:
        raise ValueError(
            "auth.api_key must be set in production environment "
            "(set AUTH_API_KEY) — refusing to start unauthenticated"
        )
    if len(settings.auth.api_key) < 32:
        raise ValueError("auth.api_key must be at least 32 characters in production")
    if "*" in settings.cors.origins:
        raise ValueError(
            "cors.origins must not contain '*' in production. "
            "Set CORS_ORIGINS to an explicit list of allowed origins."
        )


def _env_prefix_for_model(model_name: str) -> str:
    """Env-var prefix for a settings model by class name.

    ``GlobalSettings`` itself (top-level, un-clustered vars) has no prefix; each
    sub-model carries its own via ``model_config["env_prefix"]`` (e.g.
    ``CrossrefOptions`` -> ``CROSSREF_``). Derived from the model structure so it
    stays correct as sections are added.
    """
    if model_name == GlobalSettings.__name__:
        return ""
    for finfo in GlobalSettings.model_fields.values():
        ann = finfo.annotation
        if isinstance(ann, type) and getattr(ann, "__name__", None) == model_name:
            model_config = getattr(ann, "model_config", {})
            return cast(str, model_config.get("env_prefix", "") or "")
    return ""


def _configuration_error(exc: ValidationError) -> ConfigurationError:
    """Translate a pydantic ``ValidationError`` into a ``ConfigurationError``.

    Maps each error ``loc`` back to the actual environment variable the user
    must fix (``('pipeline', 'memory_mode')``-style loc collapses to
    ``PIPELINE_MEMORY_MODE`` via the section's env prefix) and surfaces the
    offending value plus the allowed values pydantic already carries for
    ``Literal``/enum fields.
    """
    import re

    prefix = _env_prefix_for_model(exc.title)
    problems: list[str] = []
    for err in exc.errors():
        loc = err.get("loc") or ()
        field = str(loc[-1]) if loc else exc.title
        # Top-level fields are already the bare env-var name; sub-model fields
        # are prefix + FIELD_NAME.
        env_var = field if not prefix else f"{prefix}{field.upper()}"
        expected = (err.get("ctx") or {}).get("expected")
        if not loc:
            # A model-level validator reports ``loc == ()`` and ``input`` == the
            # whole merged source mapping — every env/dotenv value matching a
            # field on this model, API keys included. The settings models' own
            # redaction cannot help: what pydantic hands back is a plain dict.
            problems.append(f"{env_var} is invalid — {err.get('msg')}")
            continue
        value = "***" if _is_secret_name(field) else err.get("input")
        if expected:
            allowed = ", ".join(re.findall(r"'([^']*)'", expected)) or expected
            problems.append(f"{env_var}={value} is invalid — allowed values: {allowed}")
        else:
            problems.append(f"{env_var}={value} is invalid — {err.get('msg')}")

    if len(problems) == 1:
        message = f"{problems[0]}. Fix it in your .env or environment."
    else:
        body = "\n".join(f"  - {p}" for p in problems)
        message = f"Invalid configuration:\n{body}\nFix these in your .env or environment."
    return ConfigurationError(message, problems=problems)


class _SettingsProxy:
    """Lazy, identity-stable stand-in for the global settings singleton.

    Constructs the real :class:`GlobalSettings` on first *attribute access*
    rather than at import, so a single bad ``.env`` value no longer makes
    ``import bibr.config`` — and with it ``bibr --help`` / ``bibr doctor`` /
    ``bibr setup``, the very tools that would fix the config — die with a raw
    pydantic traceback. On validation failure it raises
    :class:`~bibr.exceptions.ConfigurationError` naming the env var(s) to fix.

    Forwards ``__getattr__`` *and* ``__setattr__`` to the underlying instance so
    identity-preserving mutation keeps working unchanged:
    ``setup_wizard._reload_settings_in_place`` copies every section back onto
    this same object, and ``Settings.llm.local_model = ...`` mutates the nested
    model in place. The proxy keeps a stable identity forever, so the ~85
    modules that did ``from bibr.config import Settings`` at import time all
    share the one object those mutations target.

    A failed construction is not cached: the next access re-attempts, so
    ``doctor`` / ``setup`` can retry after the user fixes ``.env``.
    """

    __slots__ = ("_instance",)

    def __init__(self) -> None:
        object.__setattr__(self, "_instance", None)

    def _get(self) -> "GlobalSettings":
        inst = cast(GlobalSettings | None, object.__getattribute__(self, "_instance"))
        if inst is None:
            try:
                inst = GlobalSettings()
            except ValidationError as exc:
                raise _configuration_error(exc) from exc
            object.__setattr__(self, "_instance", inst)
        return inst

    def __getattr__(self, name: str):
        return getattr(self._get(), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self._get(), name, value)

    def __repr__(self) -> str:
        inst = object.__getattribute__(self, "_instance")
        if inst is None:
            return "<bibr Settings (not yet loaded)>"
        return repr(inst)


# Lazy singleton used throughout the application. Attribute access constructs
# and validates the real settings on first use — see ``_SettingsProxy``.
Settings = _SettingsProxy()


def snapshot_settings(settings: GlobalSettings | None = None) -> GlobalSettings:
    """Return an isolated, concrete settings snapshot for one runtime owner.

    ``Settings`` intentionally remains a mutable process-global proxy for CLI
    and setup boundaries.  Pipelines cross that boundary once, at
    construction, and retain this deep copy so later global mutations cannot
    change a running pipeline or leak between pipeline instances.
    """
    source = settings if settings is not None else Settings
    return cast(GlobalSettings, source.model_copy(deep=True))
