"""Curated local-LLM registry for onboarding (``bibr setup`` and ``--llm local``).

Every model-specific fact — HF ids, quant variants, VRAM floors, quirk env —
lives in this one static table so the wizard and the managed servers stay
model-agnostic. Update policy: edited per release; the wizard's custom-HF-id
escape hatch covers staleness in between.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeculativeConfig:
    """A vLLM speculative decoding config measured for one registry variant.

    ``bibr.local.vllm_speculative`` decides at each managed vLLM start whether
    to use it (``LLM_VLLM_SPEC_DECODE``).
    """

    #: vLLM's ``--speculative-config`` method: ``mtp`` (the checkpoint's own
    #: multi-token-prediction layers) or ``dflash`` (a separate drafter model).
    method: str
    num_speculative_tokens: int
    #: The budget floor with this config, measured as
    #: ``LocalLLMVariant.min_budget_gb`` is (the drafted tokens need more KV
    #: cache, and a drafter model its own weights).
    min_budget_gb: float
    #: The drafter's Hub id and pinned revision, for methods that need one.
    drafter: str | None = None
    drafter_revision: str | None = None

    def server_args(self) -> tuple[str, ...]:
        """The vLLM flags for this config."""
        config: dict[str, object] = {"method": self.method}
        if self.drafter is not None:
            config["model"] = self.drafter
            config["revision"] = self.drafter_revision
        config["num_speculative_tokens"] = self.num_speculative_tokens
        return (
            "--speculative-config",
            json.dumps(config, separators=(",", ":")),
            # vLLM 0.32 turns async scheduling on for these methods; every
            # speculative run bibr measured scheduled synchronously.
            "--no-async-scheduling",
        )


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
    #: The Hub commit of a llama.cpp or MLX build, whose servers download a
    #: repository's ``main`` when given its name. bibr downloads ``hub_files``
    #: at this commit into the Hugging Face cache itself and serves the local
    #: copy (``bibr.local.hub_pins``). vLLM variants pin theirs with
    #: ``--revision`` in ``server_args`` instead (``variant_revision``).
    revision: str | None = None
    #: The files of ``revision`` the runtime loads: the one GGUF for
    #: llama.cpp, the model directory's files for MLX.
    hub_files: tuple[str, ...] = ()
    #: The managed vLLM server's budget floor, in GiB of
    #: ``--gpu-memory-utilization`` x CUDA's total: the smallest budget measured
    #: to start with an empty compile cache and serve one ``--max-model-len``
    #: request at this variant's own ``--max-num-seqs``, rounded up to a
    #: quarter GiB. bibr sizes the managed LLM share against it
    #: (``bibr.local.vllm_share``). None: ``min_vram_gb`` stands in for it.
    min_budget_gb: float | None = None
    #: Speculative decoding configs measured for this variant on vLLM, each
    #: with its own budget floor (``SpeculativeConfig``). Empty: none.
    speculative: tuple[SpeculativeConfig, ...] = ()

    def speculative_config(self, method: str) -> SpeculativeConfig | None:
        """This variant's config for ``method`` (``mtp``, ``dflash``), or None."""
        for config in self.speculative:
            if config.method == method:
                return config
        return None


#: How far under a tier's whole GiB a card's nvidia-smi ``memory.total`` may be
#: and still count as that size. Some cards report a little less than their
#: nominal size: an RTX 4060 8,188 MiB, an RTX 5060 (laptop) 8,151 MiB, an RTX
#: 5070 12,227 MiB and an RTX 5090 32,607 MiB, against 8,192, 12,288 and 32,768;
#: an RTX 3090 reports exactly 24,576. A quarter GiB covers all of them and is
#: far from the next smaller sizes (an 11 GB card reports 11,264 MiB).
CARD_SIZE_TOLERANCE_GB = 0.25

#: Default for the arguments probed from this machine when not given.
_PROBE = object()

#: The smallest card (GiB of ``memory.total``, within ``CARD_SIZE_TOLERANCE_GB``)
#: whose default LLM runs on vLLM: the 4B student's W4A16 build, taking turns
#: with PaddleOCR-VL on vLLM. Below it the default is the 4B's GGUF build on
#: llama.cpp, with GLM-OCR on llama.cpp too. bibr's memory mode is aggressive
#: on such cards (``bibr.local.pipeline.CUDA_BALANCED_MIN_GB``). A card older
#: than Ampere gets llama.cpp at any size (``cuda_llm_backend_for``).
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


# bibr's fine-tuned extraction students (2026-10-14; the 4B retrained
# 2026-10-26). Each is a Qwen3.5 base with a LoRA trained on licence-filtered
# Qwen3.8-27B answers to bibr's own prompts, merged into a full checkpoint so
# the base model's MTP layers stay usable for speculative decoding (vLLM's spec
# decoding does not combine well with unmerged LoRA). Scores are from the 9 Oct
# harness (v5 scorer, the 252-paper development set; see
# docs/guides/extraction-models.md): the 2B's paired against Qwen3.8-27B's
# 223/252 at bibr's 0.7.0 prompts, the 4B's (pointer-v1) against its previous
# revision at bibr e491ed64, 218 to 217 on the same box.
#
# bf16, W4A16 and GGUF sizes are the built files (2026-10-14 and 2026-10-26;
# GGUF from llama.cpp 6868a52); MLX sizes are estimates until those builds
# exist. bf16: the merged checkpoint (text, vision and MTP weights). Every 4B
# build and the 2B bf16 are pinned to a Hub commit (the revisions below); bibr
# downloads the pinned llama.cpp and MLX files itself (bibr.local.hub_pins).
#
# min_vram_gb is the card size (``fits_card``) from which setup and
# ``default_local_model`` pick a variant. What backs each one: batches on an
# RTX 3090 whose free memory was cut to an emulated card's, less 0.5 GiB for a
# desktop, at bibr's default memory mode and batch size for that card (the
# card's size was emulated, not its speed). On the swap path a vLLM start
# needs the budget floor, vLLM's overshoot reserve and 1 GiB for bibr free
# (bibr.local.vllm_share); a server that stays up for the run (any other OCR)
# needs 6 GiB less what bibr holds in place of that 1 GiB. Every vLLM run
# below took turns with PaddleOCR-VL.
# - 4B bf16, 20: 48 papers on a 20 GB card (20,475 MiB, an RTX 4000 Ada; the
#   last start found 16.76 GiB free of the 14.25 needed then, at 128 sequences
#   on vLLM 0.31) and 100 on 24 GB. A card emulated at 16 GB refused its starts
#   once bibr held 2 GiB, before bibr released its own models ahead of a
#   refusal; the bf16 4B has not been run on 16 GB since.
# - 4B W4A16, 10: 48 papers on 10 GB in aggressive mode (0.27 GiB left at the
#   LLM starts), 100 on 12 GB and 48 on 16 GB in balanced mode. An 8 GB card
#   (about 7.0 GiB to CUDA beside a desktop) refused every start.
# - 4B GGUF Q8_0, 7: 100 papers on an 8 GB card (8,188 MiB, an RTX 4060), with
#   GLM-OCR taking turns on llama.cpp. No card between 7 and 8 GB was run.
# - 4B GGUF Q4_K_M, 5: 48 papers on a 6 GB card, with GLM-OCR taking turns on
#   llama.cpp and 40 MiB free at the tightest point. No 5 GB card was run.
# - 2B bf16, 12: no run at the default settings beside a desktop backs it. On
#   an emulated 8 GB card with no desktop, bibr refused all 6 of its starts at
#   the default settings (48 papers, none done); with the LLM's share set by
#   hand (about 6.4 GiB) the same 48 papers finished. Its budget (6.25 GiB at
#   128 sequences) needs 0.25 GiB more free than the W4A16 build's did then;
#   it scores less and is offered from 12 GB in advanced setup only.
# - The 2B GGUF builds and every MLX build: a size rule, not a run. GGUF:
#   ceil(1.25 x the file's GiB) + 1; MLX: ceil(1.25 x the estimated size).
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

#: The 2B student, served with at most 128 sequences. Qwen3.5 is a hybrid
#: model: vLLM gives each running sequence one Mamba cache block, and its
#: default of 256 sequences needs more blocks than the 4B's 0.54 share of a
#: 24 GB card holds (165), so the server failed CUDA graph capture and never
#: started. bibr ran at most 19 sequences at once in 20- and 60-paper
#: batches; vLLM queues any beyond the cap. It also halves the CUDA graph
#: capture sizes (256 instead of 512), which cut the memory capture takes
#: beyond the share from about 1.3 to 0.75 GiB (vLLM 0.31). The 2B has not
#: been measured on vLLM 0.32.
_QWEN35_2B_SERVER_ARGS: tuple[str, ...] = (*_QWEN35_STUDENT_SERVER_ARGS, "--max-num-seqs", "128")

#: Both vLLM builds of the 4B student, served with at most 64 sequences, the
#: cap every vLLM 0.32 measurement of them used (128 and vLLM's default were
#: not run on 0.32). At 128, vLLM 0.31 needed a Mamba cache block for each
#: sequence, about 2.1 GiB of KV cache for the W4A16 build instead of the
#: 1.1 GiB one 32,768-token request needs.
QWEN35_4B_SERVER_ARGS: tuple[str, ...] = (*_QWEN35_STUDENT_SERVER_ARGS, "--max-num-seqs", "64")

#: The 4B student's W4A16 GPTQ build (int4 weights, the Gated DeltaNet A_log
#: and norm tensors kept in F32) takes the same args as its bf16 build. The
#: pinned build scored 218/252, level with its bf16 build's 218 on the same box.
QWEN35_4B_W4A16_SERVER_ARGS: tuple[str, ...] = QWEN35_4B_SERVER_ARGS

#: Hub revisions. The 4B's are the commits on each repository's ``pointer-v1``
#: branch (2026-10-26 and 27): the student retrained with pointer answers for
#: the two front-matter calls (``LLM_POINTER_OUTPUTS``, still off by default)
#: and the builds made from it. Each repository's ``main`` still holds the
#: builds of the previous 4B, 6ab47a4e (bf16 6ab47a4e, W4A16 a2af8f6e, GGUF
#: 8f5cb6f8, MLX 16bf58c9). The 2B's was published 2026-10-14; its GGUF and
#: MLX builds are not pinned.
_QWEN35_4B_REVISION = "9199424d4bd200feb247e26e4a860b63da73cbaf"
_QWEN35_4B_GGUF_REVISION = "0aa228e8c908512ad5e5e13fa55e7e7d3180c7d3"
_QWEN35_4B_MLX_REVISION = "432b8cd80ea920919ef26c48401473c6d8c8a3e1"
_QWEN35_2B_REVISION = "81d0442d0ecadc20e1b9587d54bff42242123870"
_QWEN35_4B_W4A16_REVISION = "a32f20485181574d849cfb32f8eac1f266c4143d"
#: What mlx-lm loads from the 4B's MLX build: every file of the revision but
#: its README.md and .gitattributes.
_QWEN35_4B_MLX_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "model.safetensors",
)

#: Budget floors (``LocalLLMVariant.min_budget_gb``,
#: ``SpeculativeConfig.min_budget_gb``): cold starts (empty vLLM, Inductor and
#: Triton caches) on an RTX 3090, stepping ``--gpu-memory-utilization`` by
#: 0.01 of the card until one budget served a 31,000-token prompt and the next
#: lower one failed vLLM's startup check that its KV cache holds one
#: 32,768-token request. The smallest budget that served, rounded up to a
#: quarter GiB. The 4B's: vLLM 0.32.0 at 64 sequences (2026-10-27), on the
#: pinned revisions above (9199424d and its W4A16 build a32f2048). The 2B's:
#: vLLM 0.31.0 at 128 sequences. A warm start (the compile cache filled by an
#: earlier start) needed about 1.1 GiB less on vLLM 0.31.
QWEN35_4B_BUDGET_GB = 11.0  # served at 10.84 GiB; 10.60 had 1.09 of the 1.11 GiB KV cache
QWEN35_4B_W4A16_BUDGET_GB = 6.0  # served at 5.89 GiB; 5.65 had 0.88 of 1.11 GiB
QWEN35_2B_BUDGET_GB = 6.25  # vLLM 0.31: served at 6.13 GiB; 5.89 had 127 Mamba blocks
#: With MTP k=3, which needs 1.42 GiB of KV cache for that request, and the
#: MTP layers' own memory.
QWEN35_4B_MTP_BUDGET_GB = 12.0  # served at 11.78 GiB; 11.54 had 1.29 of 1.42 GiB
QWEN35_4B_W4A16_MTP_BUDGET_GB = 7.25  # served at 7.07 GiB; 6.83 had 1.29 of 1.42 GiB
#: With DFlash k=7, which needs 1.72 GiB of KV cache and the drafter's weights.
QWEN35_4B_DFLASH_BUDGET_GB = 13.5  # served at 13.43 GiB; 13.19 had 1.54 of 1.72 GiB

#: MTP with 3 drafted tokens, from the checkpoint's own MTP layers (the base
#: model's, kept by the merge). On the 252 dev papers (vLLM 0.32, one LLM
#: server up for the whole run, at 0.85 of the card): the bf16 4B (pointer-v1
#: weights) 218/252 against 219 without it on an RTX 3090 (-0.4 points
#: [-1.2, 0]) and 218 against 218 on an RTX 5090, with 3.88 of 4 tokens
#: accepted per step and the LLM stage 0.80x and 0.74x as long. The previous
#: revision's W4A16 build (a2af8f6e) scored 218/252 with it, as without it
#: on vLLM 0.31; the pinned W4A16 build has not been scored with it. In a batch
#: where the LLM server restarts every chunk to take turns with PaddleOCR-VL,
#: each start with it took 51 s instead of 37 s, and papers/h fell 5% and 9%
#: (W4A16 on emulated 12 and 16 GB cards), so ``bibr.local.vllm_speculative``
#: leaves it off there.
_QWEN35_4B_MTP = SpeculativeConfig("mtp", 3, QWEN35_4B_MTP_BUDGET_GB)
_QWEN35_4B_W4A16_MTP = SpeculativeConfig("mtp", 3, QWEN35_4B_W4A16_MTP_BUDGET_GB)
#: DFlash with 7 drafted tokens from z-lab's drafter for Qwen3.5-4B
#: (1.27 GB), pinned to the revision measured. bf16 4B on the 252 papers:
#: 220/252 against 219 without it (RTX 3090) and 217 against 218 (RTX 5090),
#: with the LLM stage 0.72x and 0.64x as long, but about half the
#: KV cache capacity of no speculation (159,251 against 310,108 tokens at 0.85
#: of an RTX 3090). Opt-in only (``LLM_VLLM_SPEC_DECODE=dflash``); not listed
#: for the W4A16 build, whose quality with it was not measured.
QWEN35_4B_DFLASH_DRAFTER = "z-lab/Qwen3.5-4B-DFlash"
QWEN35_4B_DFLASH_DRAFTER_REVISION = "9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf"
_QWEN35_4B_DFLASH = SpeculativeConfig(
    "dflash",
    7,
    QWEN35_4B_DFLASH_BUDGET_GB,
    drafter=QWEN35_4B_DFLASH_DRAFTER,
    drafter_revision=QWEN35_4B_DFLASH_DRAFTER_REVISION,
)
#: Thinking off per request too, for llama.cpp and MLX servers, which do not
#: read ``server_args``.
_QWEN35_STUDENT_ENV: dict[str, str] = {"LLM_CHAT_TEMPLATE_KWARGS": '{"enable_thinking": false}'}

REGISTRY: tuple[LocalLLMModel, ...] = (
    LocalLLMModel(
        key="bibr-extract-4b",
        label="bibr extract 4B (recommended) — Qwen3.5-4B fine-tuned on bibr's tasks",
        variants=(
            LocalLLMVariant(
                hf_id="scienceverse/bibr-extract-qwen35-4b",
                platform="cuda",
                quant="bf16",
                min_vram_gb=20,
                min_budget_gb=QWEN35_4B_BUDGET_GB,
                approx_download_gb=8.7,
                server_args=("--revision", _QWEN35_4B_REVISION, *QWEN35_4B_SERVER_ARGS),
                runtime="vllm",
                speculative=(_QWEN35_4B_MTP, _QWEN35_4B_DFLASH),
            ),
            # int4 weights: 218/252, level with the bf16 build on the same box, at
            # half its budget.
            LocalLLMVariant(
                hf_id="scienceverse/bibr-extract-qwen35-4b-W4A16",
                platform="cuda",
                quant="w4a16",
                min_vram_gb=VLLM_LLM_MIN_CARD_GB,
                min_budget_gb=QWEN35_4B_W4A16_BUDGET_GB,
                approx_download_gb=3.8,  # 4,050,886,734 bytes at that revision
                server_args=(
                    "--revision",
                    _QWEN35_4B_W4A16_REVISION,
                    *QWEN35_4B_W4A16_SERVER_ARGS,
                ),
                runtime="vllm",
                speculative=(_QWEN35_4B_W4A16_MTP,),
            ),
            # llama.cpp (Windows, cards below 10 GB, cards older than Ampere):
            # Q8_0 first, as the closest to the evaluated bf16 weights.
            LocalLLMVariant(
                hf_id="scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0",
                platform="cuda",
                quant="gguf-q8_0",
                min_vram_gb=7,
                approx_download_gb=4.3,
                runtime="llama-cpp",
                revision=_QWEN35_4B_GGUF_REVISION,
                hub_files=("bibr-extract-qwen35-4b-Q8_0.gguf",),
            ),
            LocalLLMVariant(
                hf_id="scienceverse/bibr-extract-qwen35-4b-GGUF:Q4_K_M",
                platform="cuda",
                quant="gguf-q4_k_m",
                min_vram_gb=5,
                approx_download_gb=2.6,
                runtime="llama-cpp",
                revision=_QWEN35_4B_GGUF_REVISION,
                hub_files=("bibr-extract-qwen35-4b-Q4_K_M.gguf",),
            ),
            LocalLLMVariant(
                hf_id="scienceverse/bibr-extract-qwen35-4b-mlx-8bit",
                platform="mlx",
                quant="8bit",
                min_vram_gb=6,
                approx_download_gb=4.5,
                revision=_QWEN35_4B_MLX_REVISION,
                hub_files=_QWEN35_4B_MLX_FILES,
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
                min_vram_gb=12,  # 0.25 GiB more than the W4A16 build; no default run backs it
                min_budget_gb=QWEN35_2B_BUDGET_GB,
                approx_download_gb=4.2,
                server_args=("--revision", _QWEN35_2B_REVISION, *_QWEN35_2B_SERVER_ARGS),
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


def default_variant(
    platform_key: str,
    memory_gb: float | None,
    runtime: str | None = None,
) -> tuple[LocalLLMModel, LocalLLMVariant] | None:
    """The default model and variant for a machine and runtime: setup's and the runtime's.

    The recommended variant that fits ``memory_gb`` (``recommended_variant``).
    When none fits, the runtime's recommended variant with the smallest
    minimum, which comes closest to starting (a start that cannot fit is
    refused by the managed server's own check). ``memory_gb`` None (unknown)
    picks the best variant for the platform and runtime. ``bibr setup``
    writes this as ``LLM_LOCAL_MODEL``, and ``default_local_model`` serves it
    when ``LLM_LOCAL_MODEL`` is unset, so the two agree on every card.
    """
    picked = recommended_variant(platform_key, memory_gb, runtime)
    if picked is not None or memory_gb is None:
        return picked
    candidates = [
        (model, variant)
        for model in (get_model(key) for key in RECOMMENDED_MODEL_KEYS)
        for variant in variants_for(model, platform_key, None, runtime)
    ]
    return min(candidates, key=lambda pair: pair[1].min_vram_gb, default=None)


def cuda_llm_backend_for(
    memory_gb: float | None, *, compute_capability: float | None | object = _PROBE
) -> str:
    """Managed LLM backend for a CUDA card with ``memory_gb`` of VRAM (None = unknown).

    vLLM only when a vLLM variant of a recommended model actually fits (the
    4B's W4A16 build from ``VLLM_LLM_MIN_CARD_GB``); otherwise llama.cpp,
    whose GGUF builds of the students go down to 3 GB. The old rule ("vLLM
    above 8 GB") handed 9-10 GB cards a plan whose only vLLM variant
    (NuExtract 3 bf16, 11 GB) OOMed after OCR, because the fit filter was
    dropped silently when nothing fit. A card whose compute capability is
    below ``bibr.ocr.registry.VLLM_MIN_COMPUTE_CAPABILITY`` (read when
    ``compute_capability`` is not given; unreadable counts as fine) gets
    llama.cpp at any size.
    """
    from bibr.ocr.registry import vllm_compute_capability_blocker

    blocker = (
        vllm_compute_capability_blocker()
        if compute_capability is _PROBE
        else vllm_compute_capability_blocker(capability=compute_capability)
    )
    if blocker is not None:
        return "llama-cpp"
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


def _detected_memory_gb(platform_key: str) -> float | None:
    """This machine's memory for ``platform_key``'s tiers, as setup reads it, or None.

    The first NVIDIA card's total for ``cuda`` (probed once per process), the
    system RAM on Apple Silicon for ``mlx``; None on other hardware.
    """
    if platform_key == "cuda":
        from bibr.ocr.registry import _cuda_vram_gb

        return _cuda_vram_gb()
    if platform_key == "mlx" and platform.system() == "Darwin" and platform.machine() == "arm64":
        return _system_memory_gb()
    return None


def default_local_model(backend: str, *, memory_gb: float | None | object = _PROBE) -> str:
    """The recommended model's variant for ``backend`` when ``LLM_LOCAL_MODEL`` is unset.

    Every managed server used to read one config default — the 8-bit MLX
    build — so a hand-written ``LLM_BACKEND=vllm`` or ``llama-cpp`` downloaded
    weights the runtime could not load. The answer is ``default_variant`` for
    the backend's runtime and this machine's memory (``memory_gb`` when
    given), the variant ``bibr setup`` writes on the same machine: on CUDA the
    4B student as bf16 for vLLM from 20 GB, its W4A16 build below, GGUF Q8_0
    for llama.cpp from 7 GB and Q4_K_M below; 8-bit for MLX. It used to ignore
    the card, so ``--llm local`` without setup on a 10-19 GB card served the
    bf16 build, whose start is refused there.
    """
    try:
        platform_key, runtime = _BACKEND_VARIANT_FILTER[backend]
    except KeyError:
        raise ValueError(f"no default local model for LLM backend {backend!r}") from None
    memory = _detected_memory_gb(platform_key) if memory_gb is _PROBE else memory_gb
    picked = default_variant(
        platform_key, memory if isinstance(memory, int | float) else None, runtime
    )
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


def variant_revision(variant: LocalLLMVariant) -> str | None:
    """The Hub commit a registry variant is pinned to, or None when it follows ``main``.

    ``revision`` for llama.cpp and MLX builds, the ``--revision`` in
    ``server_args`` for vLLM ones.
    """
    if variant.revision is not None:
        return variant.revision
    args = variant.server_args
    if "--revision" in args:
        index = args.index("--revision")
        if index + 1 < len(args):
            return args[index + 1]
    return None


def pinned_download_variant(model: str) -> LocalLLMVariant | None:
    """The registry variant whose pinned files bibr downloads itself, by exact ``hf_id``.

    A llama.cpp or MLX build with a ``revision`` and ``hub_files``; None for
    any other id (vLLM takes ``--revision`` itself; an unpinned build, a local
    path or a custom id is handed to the runtime unchanged).
    """
    for m in REGISTRY:
        for v in m.variants:
            if v.hf_id == model and v.revision is not None and v.hub_files:
                return v
    return None


def vllm_registry_variant(model: str) -> LocalLLMVariant | None:
    """The registry's CUDA vLLM variant whose ``hf_id`` is *model*, or None."""
    for m in REGISTRY:
        for v in m.variants:
            if v.platform == "cuda" and v.runtime == "vllm" and v.hf_id == model:
                return v
    return None


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
