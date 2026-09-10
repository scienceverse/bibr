"""Curated local-LLM registry for onboarding (``bibr setup`` and ``--llm local``).

Every model-specific fact — HF ids, quant variants, VRAM floors, quirk env —
lives in this one static table so the wizard and the managed servers stay
model-agnostic. Update policy: edited per release; the wizard's custom-HF-id
escape hatch covers staleness in between.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalLLMVariant:
    hf_id: str
    platform: str  # "cuda" | "mlx"
    quant: str  # e.g. "bf16", "awq-4bit", "nvfp4"
    min_vram_gb: float
    approx_download_gb: float
    # Extra CLI flags appended to the managed vLLM launch command for this
    # variant (see ``registry_server_args`` / ``VllmLlmServer``). Consumed only
    # for ``platform == "cuda"`` — the vllm-mlx server does not read these.
    server_args: tuple[str, ...] = ()
    runtime: str | None = None


@dataclass(frozen=True)
class LocalLLMModel:
    key: str
    label: str
    variants: tuple[LocalLLMVariant, ...]  # best-quality-first per platform
    env: dict[str, str] = field(default_factory=dict)  # quirk env for .env
    experimental_platforms: frozenset[str] = frozenset()


# Numbers below are pinned from the HF API (Task 1 Step 1; re-verified 2026-09-02 —
# NuExtract 3 bf16 is 4.539B params, every listed artifact is within 0.3 GB of the Hub).
# bf16 variants: approx_download_gb = param_count * 2 / 2**30 (rounded 1 dp);
# the nvfp4 variant uses its on-disk safetensors size. min_vram_gb =
# ceil(approx_download_gb * 1.25).
REGISTRY: tuple[LocalLLMModel, ...] = (
    LocalLLMModel(
        key="nuextract3",
        label="NuExtract 3 (recommended) — extraction-tuned, validated on bibr evals",
        variants=(
            LocalLLMVariant(
                hf_id="numind/NuExtract3-GGUF:Q4_K_M",
                platform="cuda",
                quant="gguf-q4_k_m",
                min_vram_gb=5,
                approx_download_gb=2.7,
                runtime="llama-cpp",
            ),
            LocalLLMVariant(
                hf_id="numind/NuExtract3",
                platform="cuda",
                quant="bf16",
                min_vram_gb=11,  # 8.5 * 1.25 -> ceil 11
                approx_download_gb=8.5,  # 4.539B params * 2 bytes
                # --max-model-len 32768 (not the card's 131072) leaves KV
                # headroom at the 11 GB VRAM floor. MTP deliberately remains
                # opt-in through LLM_VLLM_EXTRA_ARGS: upstream still disables
                # prefix caching with MTP and reports an accuracy interaction.
                server_args=(
                    "--revision",
                    "2e9fca82ee641e6bb6e1f5d905241e994be27a07",
                    "--chat-template-content-format",
                    "openai",
                    "--generation-config",
                    "vllm",
                    "--max-model-len",
                    "32768",
                ),
                runtime="vllm",
            ),
            # MLX quant ladder (sizes = on-disk safetensors, HF API 2026-07-05).
            # 8bits first among MLX variants: best quality per M4-16GB benchmark
            # (21 tok/s single / 50 tok/s batched under continuous batching).
            LocalLLMVariant(
                hf_id="numind/NuExtract3-mlx-8bits",
                platform="mlx",
                quant="8bit",
                min_vram_gb=6,  # 4.8 * 1.25 -> ceil 6
                approx_download_gb=4.8,
            ),
            LocalLLMVariant(
                hf_id="numind/NuExtract3-mlx-6bits",
                platform="mlx",
                quant="6bit",
                min_vram_gb=5,  # 3.8 * 1.25 -> ceil 5
                approx_download_gb=3.8,
            ),
            LocalLLMVariant(
                hf_id="numind/NuExtract3-mlx-5bits",
                platform="mlx",
                quant="5bit",
                min_vram_gb=5,  # 3.3 * 1.25 -> ceil 5
                approx_download_gb=3.3,
            ),
            LocalLLMVariant(
                hf_id="numind/NuExtract3-mlx-4bits",
                platform="mlx",
                quant="4bit",
                min_vram_gb=4,  # 2.8 * 1.25 -> ceil 4
                approx_download_gb=2.8,
            ),
            LocalLLMVariant(
                hf_id="numind/NuExtract3-mlx-nvfp4",
                platform="mlx",
                quant="nvfp4",
                min_vram_gb=4,  # 2.8 * 1.25 -> ceil 4
                approx_download_gb=2.8,  # on-disk safetensors size
            ),
        ),
        env={},  # Auto uses Instructor; native NuExtract templates require explicit opt-in.
        experimental_platforms=frozenset({"mlx"}),  # pending platform validation
    ),
    LocalLLMModel(
        key="gemma-4-e4b",
        label="Gemma 4 E4B — general small model, close to cloud quality on refs",
        variants=(
            LocalLLMVariant(
                hf_id="google/gemma-4-E4B-it",
                platform="cuda",
                quant="bf16",
                min_vram_gb=19,  # 14.9 * 1.25 -> ceil 19
                approx_download_gb=14.9,  # 7.996B params * 2 bytes
            ),
        ),
        # Strict grammar makes small models drop optional fields (authors);
        # json_object mode + prompt-driven population is the validated fix.
        env={"LLM_INSTRUCTOR_MODE": "json"},
        experimental_platforms=frozenset(),
    ),
)


def cuda_llm_backend_for(memory_gb: float | None) -> str:
    """Managed LLM backend for a CUDA card with ``memory_gb`` of VRAM (None = unknown).

    vLLM only when a vLLM variant of the recommended model actually fits;
    otherwise llama.cpp, whose GGUF build needs 5 GB. The old rule ("vLLM
    above 8 GB") handed 9-10 GB cards a plan whose only vLLM variant (bf16,
    11 GB) OOMed after OCR, because the fit filter was dropped silently when
    nothing fit.
    """
    if memory_gb is None:
        return "vllm"
    if variants_for(get_model("nuextract3"), "cuda", memory_gb, "vllm"):
        return "vllm"
    return "llama-cpp"


#: Managed backend → (registry platform, runtime filter) used when
#: ``LLM_LOCAL_MODEL`` is unset.
_BACKEND_VARIANT_FILTER: dict[str, tuple[str, str | None]] = {
    "vllm": ("cuda", "vllm"),
    "llama-cpp": ("cuda", "llama-cpp"),
    "vllm-mlx": ("mlx", None),
    "rapid-mlx": ("mlx", None),
}


def default_local_model(backend: str) -> str:
    """The recommended model's variant for ``backend`` when ``LLM_LOCAL_MODEL`` is unset.

    Every managed server used to read one config default — the 8-bit MLX
    build — so a hand-written ``LLM_BACKEND=vllm`` or ``llama-cpp`` downloaded
    weights the runtime could not load. The answer is the registry's
    best-quality NuExtract 3 entry that the backend's runtime can serve.
    """
    try:
        platform_key, runtime = _BACKEND_VARIANT_FILTER[backend]
    except KeyError:
        raise ValueError(f"no default local model for LLM backend {backend!r}") from None
    variants = variants_for(get_model("nuextract3"), platform_key, None, runtime)
    if not variants:  # pragma: no cover - registry invariant
        raise ValueError(f"the registry has no {platform_key}/{runtime} NuExtract 3 variant")
    return variants[0].hf_id


def registry_server_args(model: str) -> tuple[str, ...]:
    """Extra vLLM launch args for a CUDA registry variant, by exact ``hf_id``.

    Matches ``model`` against every ``platform == "cuda"`` variant's ``hf_id``;
    returns that variant's ``server_args`` or ``()`` for an unknown id. mlx
    variants are intentionally not consulted — the vllm-mlx server does not
    consume ``server_args``.
    """
    for m in REGISTRY:
        for v in m.variants:
            if v.platform == "cuda" and v.hf_id == model:
                return v.server_args
    return ()


def get_model(key: str) -> LocalLLMModel:
    for m in REGISTRY:
        if m.key == key:
            return m
    raise KeyError(key)


def variants_for(
    model: LocalLLMModel,
    platform_key: str,
    memory_gb: float | None,
    runtime: str | None = None,
) -> list[LocalLLMVariant]:
    """Variants for a platform, filtered to fit ``memory_gb`` (None = no filter)."""
    out = [v for v in model.variants if v.platform == platform_key]
    if runtime is not None:
        out = [v for v in out if v.runtime in (None, runtime)]
    if memory_gb is not None:
        out = [v for v in out if v.min_vram_gb <= memory_gb]
    return out


def _nvidia_vram_gb() -> float | None:
    """Total VRAM of GPU 0 via nvidia-smi, or None if unavailable."""
    try:
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        first = result.stdout.strip().splitlines()[0]
        return float(first) / 1024.0  # MiB -> GiB
    except (ValueError, IndexError):
        return None


def _system_memory_gb() -> float:
    from bibr.local.pipeline import _get_system_memory_gb

    return _get_system_memory_gb()


def detect_hardware() -> tuple[str | None, float | None]:
    """Detect the local-LLM platform: ("cuda", vram) | ("mlx", ram) | (None, None)."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return ("mlx", _system_memory_gb())
    vram = _nvidia_vram_gb()
    if vram is not None:
        return ("cuda", vram)
    return (None, None)
