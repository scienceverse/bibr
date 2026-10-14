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


# bibr's fine-tuned extraction students (2026-10-14). Each is a Qwen3.5 base
# with a LoRA trained on licence-filtered Qwen3.8-27B answers to bibr's own
# prompts, merged into a full checkpoint so the base model's MTP layers stay
# usable for speculative decoding (vLLM's spec decoding does not combine well
# with unmerged LoRA). Scores are from the 9 Oct harness (bibr 0.7.0 prompts,
# v5 scorer, gate192 + ci60 = 252 dev papers), paired against Qwen3.8-27B's
# 223/252; see docs/guides/extraction-models.md.
#
# bf16 and GGUF sizes are the built files (2026-10-14; GGUF from llama.cpp 6868a52);
# MLX sizes are estimates until those builds exist. bf16: the merged
# checkpoint (text, vision and MTP weights), with min_vram_gb rounded up from the measured 11.7 GB peak of a
# low-memory 4B vLLM serve on a 24 GB card. GGUF and MLX: the quantized text
# model, with min_vram_gb = ceil(size * 1.25) plus about 2 GB of context and
# compute buffers, the margin the NuExtract 3 Q4_K_M build needed on 6 GB
# cards. The vLLM variants pin the Hub revision they were published at.
_QWEN35_STUDENT_SERVER_ARGS: tuple[str, ...] = (
    # Text only: the vision tower is never used for extraction.
    "--language-model-only",
    # Thinking off, as in training and in the eval harness.
    "--default-chat-template-kwargs",
    '{"enable_thinking": false}',
    "--generation-config",
    "vllm",
    "--max-model-len",
    "32768",
)

#: Hub revisions published 2026-10-14 (the merged safetensors repos).
_QWEN35_4B_REVISION = "6ab47a4e4e8b1b76c1ab234aec66f360683ff9fc"
_QWEN35_2B_REVISION = "81d0442d0ecadc20e1b9587d54bff42242123870"

#: Thinking off per request too, for llama.cpp and MLX servers, which do not
#: read ``server_args``.
_QWEN35_STUDENT_ENV: dict[str, str] = {"LLM_CHAT_TEMPLATE_KWARGS": '{"enable_thinking": false}'}

REGISTRY: tuple[LocalLLMModel, ...] = (
    LocalLLMModel(
        key="bibr-extract-4b",
        label="bibr extract 4B (recommended) — Qwen3.5-4B fine-tuned on bibr's tasks, "
        "2.8 points behind Qwen3.8-27B",
        variants=(
            LocalLLMVariant(
                hf_id="scienceverse/bibr-extract-qwen35-4b",
                platform="cuda",
                quant="bf16",
                min_vram_gb=12,
                approx_download_gb=8.7,
                server_args=("--revision", _QWEN35_4B_REVISION, *_QWEN35_STUDENT_SERVER_ARGS),
                runtime="vllm",
            ),
            # llama.cpp (Windows, cards below 8 GB): Q8_0 first, as the
            # closest to the evaluated bf16 weights.
            LocalLLMVariant(
                hf_id="scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0",
                platform="cuda",
                quant="gguf-q8_0",
                min_vram_gb=7,
                approx_download_gb=4.3,
                runtime="llama-cpp",
            ),
            LocalLLMVariant(
                hf_id="scienceverse/bibr-extract-qwen35-4b-GGUF:Q4_K_M",
                platform="cuda",
                quant="gguf-q4_k_m",
                min_vram_gb=5,
                approx_download_gb=2.6,
                runtime="llama-cpp",
            ),
            # No MLX build: the 4-bit duplicated a funding entry in its smoke
            # test and the 8-bit has not passed one yet, so Macs get the 2B.
        ),
        env=_QWEN35_STUDENT_ENV,
    ),
    LocalLLMModel(
        key="bibr-extract-2b",
        label="bibr extract 2B (low memory) — Qwen3.5-2B fine-tuned on bibr's tasks, "
        "6.3 points behind Qwen3.8-27B",
        variants=(
            LocalLLMVariant(
                hf_id="scienceverse/bibr-extract-qwen35-2b",
                platform="cuda",
                quant="bf16",
                min_vram_gb=8,
                approx_download_gb=4.2,
                server_args=("--revision", _QWEN35_2B_REVISION, *_QWEN35_STUDENT_SERVER_ARGS),
                runtime="vllm",
            ),
            LocalLLMVariant(
                hf_id="scienceverse/bibr-extract-qwen35-2b-GGUF:Q8_0",
                platform="cuda",
                quant="gguf-q8_0",
                min_vram_gb=4,
                approx_download_gb=1.9,
                runtime="llama-cpp",
            ),
            LocalLLMVariant(
                hf_id="scienceverse/bibr-extract-qwen35-2b-GGUF:Q4_K_M",
                platform="cuda",
                quant="gguf-q4_k_m",
                min_vram_gb=3,
                approx_download_gb=1.2,
                runtime="llama-cpp",
            ),
            LocalLLMVariant(
                hf_id="scienceverse/bibr-extract-qwen35-2b-mlx-8bit",
                platform="mlx",
                quant="8bit",
                min_vram_gb=3,
                approx_download_gb=2.0,
            ),
        ),
        env=_QWEN35_STUDENT_ENV,
        experimental_platforms=frozenset({"mlx"}),  # smoke-tested only
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


#: Recommended local models, best first. The first model with a variant that
#: fits the machine and its runtime is the default: the 4B wherever one of its
#: builds fits, else the 2B.
RECOMMENDED_MODEL_KEYS: tuple[str, ...] = ("bibr-extract-4b", "bibr-extract-2b")


def recommended_variant(
    platform_key: str,
    memory_gb: float | None,
    runtime: str | None = None,
) -> tuple[LocalLLMModel, LocalLLMVariant] | None:
    """The recommended model and its best variant for a machine (None = nothing fits).

    ``memory_gb`` None means unknown memory: the best model for the platform
    and runtime wins.
    """
    for key in RECOMMENDED_MODEL_KEYS:
        model = get_model(key)
        fitting = variants_for(model, platform_key, memory_gb, runtime)
        if fitting:
            return model, fitting[0]
    return None


def cuda_llm_backend_for(memory_gb: float | None) -> str:
    """Managed LLM backend for a CUDA card with ``memory_gb`` of VRAM (None = unknown).

    vLLM only when a vLLM variant of a recommended model actually fits (the
    2B student needs 8 GB); otherwise llama.cpp, whose GGUF builds of the
    students go down to 3 GB. The old rule ("vLLM above 8 GB") handed 9-10 GB
    cards a plan whose only vLLM variant (NuExtract 3 bf16, 11 GB) OOMed after
    OCR, because the fit filter was dropped silently when nothing fit.
    """
    if memory_gb is None:
        return "vllm"
    if recommended_variant("cuda", memory_gb, "vllm") is not None:
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
    weights the runtime could not load. The answer is the best recommended
    entry that the backend's runtime can serve: the 4B student, as bf16 for
    vLLM and GGUF Q8_0 for llama.cpp, and the 2B student's 8-bit build for MLX.
    """
    try:
        platform_key, runtime = _BACKEND_VARIANT_FILTER[backend]
    except KeyError:
        raise ValueError(f"no default local model for LLM backend {backend!r}") from None
    picked = recommended_variant(platform_key, None, runtime)
    if picked is None:  # pragma: no cover - registry invariant
        raise ValueError(f"the registry has no recommended {platform_key}/{runtime} variant")
    return picked[1].hf_id


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
