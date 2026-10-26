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
    #: The card-size tier: the smallest card, in GiB of nvidia-smi's
    #: ``memory.total`` (within ``CARD_SIZE_TOLERANCE_GB``), that setup and
    #: ``recommended_variant`` offer this variant on. A size, not a budget.
    min_vram_gb: float
    approx_download_gb: float
    # Extra CLI flags appended to the managed vLLM launch command for this
    # variant (see ``registry_server_args`` / ``VllmLlmServer``). Consumed only
    # for ``platform == "cuda"`` — the vllm-mlx server does not read these.
    server_args: tuple[str, ...] = ()
    runtime: str | None = None
    #: The managed vLLM server's budget floor, in GiB of
    #: ``--gpu-memory-utilization`` x CUDA's total: the smallest budget measured
    #: to start with an empty compile cache and serve one ``--max-model-len``
    #: request at this variant's own ``--max-num-seqs``, rounded up to a
    #: quarter GiB. bibr sizes the managed LLM share against it
    #: (``bibr.local.vllm_share``). None: ``min_vram_gb`` stands in for it.
    min_budget_gb: float | None = None


#: How far under a tier's whole GiB a card's nvidia-smi ``memory.total`` may be
#: and still count as that size. Some cards report a little less than their
#: nominal size: an RTX 4060 8,188 MiB, an RTX 5060 (laptop) 8,151 MiB, an RTX
#: 5070 12,227 MiB and an RTX 5090 32,607 MiB, against 8,192, 12,288 and 32,768;
#: an RTX 3090 reports exactly 24,576. A quarter GiB covers all of them and is
#: far from the next smaller sizes (an 11 GB card reports 11,264 MiB).
CARD_SIZE_TOLERANCE_GB = 0.25

#: The smallest card (GiB of ``memory.total``, within ``CARD_SIZE_TOLERANCE_GB``)
#: whose default LLM runs on vLLM: the 4B student's W4A16 build, taking turns
#: with PaddleOCR-VL on vLLM. Below it the default is the 4B's GGUF build on
#: llama.cpp, with GLM-OCR on llama.cpp too. bibr's memory mode is aggressive
#: on such cards (``bibr.local.pipeline.CUDA_BALANCED_MIN_GB``).
VLLM_LLM_MIN_CARD_GB = 10


def fits_card(min_vram_gb: float, memory_gb: float) -> bool:
    """Whether a card of ``memory_gb`` (nvidia-smi total, GiB) is at least ``min_vram_gb``.

    The one card-size comparison for the registry tiers and the CUDA memory
    mode (``bibr.local.pipeline.CUDA_BALANCED_MIN_GB``), with
    ``CARD_SIZE_TOLERANCE_GB`` of slack. PaddleOCR-VL's 8 GB gate has none, so
    an RTX 4060 keeps GLM-OCR on llama.cpp beside its llama.cpp LLM.
    """
    return memory_gb + CARD_SIZE_TOLERANCE_GB >= min_vram_gb


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
# bf16, W4A16 and GGUF sizes are the built files (2026-10-14 and 2026-10-26;
# GGUF from llama.cpp 6868a52); MLX sizes are estimates until those builds
# exist. bf16: the merged checkpoint (text, vision and MTP weights). The vLLM
# variants pin the Hub revision they were published at, and their min_vram_gb
# is the smallest card whose default finished a batch on an emulated card with
# 0.5 GiB taken by a desktop, at bibr's default memory mode and batch size for
# that card: on the swap path a start needs the budget floor, vLLM's overshoot
# reserve and 1 GiB for bibr free (bibr.local.vllm_share). The 4B bf16 needs
# 14.25 GiB: a card emulated at 16 GB refused its starts once bibr held 2 GiB,
# so 20 (24 GB ran it). The W4A16 build needs 8 GiB: it ran 48 papers on a
# 10 GB card in aggressive mode (bibr's GPU sessions unloaded before each LLM
# start, 0.27 GiB left) and 100 on a 12 GB card in balanced mode. An 8 GB card
# (about 7.0 GiB to CUDA beside a desktop) refuses every start, so it runs the
# GGUF build on llama.cpp. The 2B on vLLM needs more (6.25 GiB at 128
# sequences plus a 1 GiB reserve) and scores less; it is listed from 12 GB for
# advanced setup only. GGUF and MLX: the quantized text model, with
# min_vram_gb = ceil(size * 1.25) plus about 2 GB of context and compute
# buffers, the margin the NuExtract 3 Q4_K_M build needed on 6 GB cards.
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
    # Qwen3.5 is a hybrid model: vLLM gives each running sequence one Mamba
    # cache block, and its default of 256 sequences needs more blocks than the
    # 4B's 0.54 share of a 24 GB card holds (165), so the server failed CUDA
    # graph capture and never started. bibr ran at most 19 sequences at once
    # in 20- and 60-paper batches; vLLM queues any beyond the cap. It also
    # halves the CUDA graph capture sizes (256 instead of 512), which cut the
    # memory capture takes beyond the share from about 1.3 to 0.75 GiB.
    "--max-num-seqs",
    "128",
)

#: The 4B student's W4A16 GPTQ build (int4 weights, the Gated DeltaNet A_log
#: and norm tensors kept in F32), served with at most 64 sequences: at 128,
#: vLLM needs a Mamba cache block for each of them, about 2.1 GiB of KV cache
#: instead of the 1.1 GiB one 32,768-token request needs. The same weights
#: scored 218/252 against the bf16 build's 217 on the same box.
QWEN35_4B_W4A16_SERVER_ARGS: tuple[str, ...] = (
    *_QWEN35_STUDENT_SERVER_ARGS,
    "--max-num-seqs",
    "64",
)

#: Hub revisions published 2026-10-14 (the merged safetensors repos) and
#: 2026-10-26 (the W4A16 build).
_QWEN35_4B_REVISION = "6ab47a4e4e8b1b76c1ab234aec66f360683ff9fc"
_QWEN35_2B_REVISION = "81d0442d0ecadc20e1b9587d54bff42242123870"
_QWEN35_4B_W4A16_REVISION = "a2af8f6e90e6481251af9eb07d02c034c30e3446"

#: Budget floors (``LocalLLMVariant.min_budget_gb``) of the vLLM variants:
#: cold starts (empty vLLM, Inductor and Triton caches) on an RTX 3090 with
#: vLLM 0.31.0, stepping ``--gpu-memory-utilization`` by 0.01 (0.24 GiB) until
#: one budget served a 32,768-token request and the next lower one failed
#: vLLM's KV cache check. The smallest budget that served, rounded up to a
#: quarter GiB. A warm start (the compile cache filled by an earlier start)
#: needs about 1.1 GiB less.
QWEN35_4B_BUDGET_GB = 12.25  # 128 seqs: served at 12.02 GiB; 11.78 had 127 Mamba blocks
QWEN35_4B_W4A16_BUDGET_GB = 6.25  # 64 seqs: served at 6.13 GiB; 5.89 had too little KV cache
QWEN35_2B_BUDGET_GB = 6.25  # 128 seqs: served at 6.13 GiB; 5.89 had 127 Mamba blocks
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
                min_vram_gb=20,
                min_budget_gb=QWEN35_4B_BUDGET_GB,
                approx_download_gb=8.7,
                server_args=("--revision", _QWEN35_4B_REVISION, *_QWEN35_STUDENT_SERVER_ARGS),
                runtime="vllm",
            ),
            # int4 weights: 218/252 against the bf16 build's 217 on the same box, at
            # half its budget.
            LocalLLMVariant(
                hf_id="scienceverse/bibr-extract-qwen35-4b-W4A16",
                platform="cuda",
                quant="w4a16",
                min_vram_gb=VLLM_LLM_MIN_CARD_GB,
                min_budget_gb=QWEN35_4B_W4A16_BUDGET_GB,
                approx_download_gb=3.8,  # 4,050,890,519 bytes at that revision
                server_args=(
                    "--revision",
                    _QWEN35_4B_W4A16_REVISION,
                    *QWEN35_4B_W4A16_SERVER_ARGS,
                ),
                runtime="vllm",
            ),
            # llama.cpp (Windows, cards below 10 GB): Q8_0 first, as the
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
            LocalLLMVariant(
                hf_id="scienceverse/bibr-extract-qwen35-4b-mlx-8bit",
                platform="mlx",
                quant="8bit",
                min_vram_gb=6,
                approx_download_gb=4.5,
            ),
            # No 4-bit MLX build: it duplicated a funding entry in its smoke test.
        ),
        env=_QWEN35_STUDENT_ENV,
        experimental_platforms=frozenset({"mlx"}),  # smoke-tested only
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
                min_vram_gb=12,  # needs 0.25 GiB more than the W4A16 build; not measured at 10
                min_budget_gb=QWEN35_2B_BUDGET_GB,
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
    4B's W4A16 build from ``VLLM_LLM_MIN_CARD_GB``); otherwise llama.cpp,
    whose GGUF builds of the students go down to 3 GB. The old rule ("vLLM
    above 8 GB") handed 9-10 GB cards a plan whose only vLLM variant
    (NuExtract 3 bf16, 11 GB) OOMed after OCR, because the fit filter was
    dropped silently when nothing fit.
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
    vLLM, GGUF Q8_0 for llama.cpp and 8-bit for MLX.
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
        out = [v for v in out if fits_card(v.min_vram_gb, memory_gb)]
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
