"""The CUDA tier table: what ``bibr setup`` writes equals what the runtime picks.

For each card row, setup's recommended plan (``LLM_BACKEND``, ``LLM_LOCAL_MODEL``,
``PIPELINE_MEMORY_MODE``, the OCR its ``OCR_BACKEND`` resolves to, and the
managed vLLM server's speculative decoding) must equal the runtime's own
resolution with nothing set but ``--llm local``: the ``local`` alias, the
default model, the auto memory mode, the automatic ``paddle`` OCR chain and
``LLM_VLLM_SPEC_DECODE=auto``. The rows are the docs' tier table
(docs/guides/extraction-models.md).
"""

import sys

import pytest

from bibr.config import GlobalSettings
from bibr.local.llm_models import default_local_model
from bibr.local.pipeline import _auto_memory_mode, resolve_llm_backend
from bibr.local.vllm_speculative import choose_spec_decode
from bibr.ocr import registry
from bibr.setup_wizard import _build_recommended_setup

_4B = "scienceverse/bibr-extract-qwen35-4b"
_W4A16 = "scienceverse/bibr-extract-qwen35-4b-W4A16"
_Q8 = "scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0"
_Q4 = "scienceverse/bibr-extract-qwen35-4b-GGUF:Q4_K_M"

_V = "vllm"
_L = "llama-cpp"
_P = "paddle-vllm"
_G = "glm-llama"

#: (card, nvidia-smi memory.total MiB, compute capability) -> (LLM backend,
#: model, memory mode, OCR runtime, speculative decoding of the managed vLLM
#: LLM server; None for llama.cpp). Rows marked "ran" finished a batch with
#: this default on an RTX 3090 cut to that card's memory beside a 0.5 GiB
#: desktop, with no refused start and no failed paper (2026-10-24..26); the
#: others follow the same size rule. Turing and Pascal rows are the
#: compute-capability routing rule, not measured cards. Every vLLM row's LLM
#: server takes turns with PaddleOCR-VL, so ``auto`` leaves speculative
#: decoding off: with MTP, batches on 12 and 16 GB cards ran 5% and 9%
#: slower (on vLLM 0.32).
TIER_TABLE = [
    (("RTX 5090", 32607, 12.0), (_V, _4B, "balanced", _P, "none")),
    (("RTX 3090", 24576, 8.6), (_V, _4B, "balanced", _P, "none")),  # ran: 100 papers
    (("RTX 4000 Ada", 20475, 8.9), (_V, _4B, "balanced", _P, "none")),  # ran: 48
    # 16 GB ran at 16,380 MiB on 2026-10-26, on this change's code, with at
    # least 4.74 GiB to spare at every LLM start.
    (("RTX 4080", 16376, 8.9), (_V, _W4A16, "balanced", _P, "none")),  # ran: 48
    (("RTX 3060 12 GB", 12288, 8.6), (_V, _W4A16, "balanced", _P, "none")),
    (("RTX 5070", 12227, 12.0), (_V, _W4A16, "balanced", _P, "none")),  # ran: 100
    (("RTX 3080 10 GB", 10240, 8.6), (_V, _W4A16, "aggressive", _P, "none")),  # ran: 48
    (("RTX 3070", 8192, 8.6), (_L, _Q8, "aggressive", _G, None)),
    (("RTX 4060", 8188, 8.9), (_L, _Q8, "aggressive", _G, None)),  # ran: 100
    # 6 GB also ran beside a 1.0 GiB desktop: 48 of 48, but 2 MiB free at the
    # minimum and 107 sentence-segmenter CPU fallbacks. No run beside a desktop
    # above 1 GiB.
    (("RTX 3050 6 GB", 6144, 8.6), (_L, _Q4, "aggressive", _G, None)),  # ran: 48
    # The Q4_K_M floor (its min_vram_gb, 5) follows the size rule: no 5 GB card ran.
    (("5 GB card", 5120, 8.6), (_L, _Q4, "aggressive", _G, None)),
    (("TITAN RTX", 24576, 7.5), (_L, _Q8, "balanced", _G, None)),
    (("RTX 2080 Ti", 11264, 7.5), (_L, _Q8, "aggressive", _G, None)),
    (("GTX 1080 Ti", 11264, 6.1), (_L, _Q8, "aggressive", _G, None)),
    (("RTX 2060", 6144, 7.5), (_L, _Q4, "aggressive", _G, None)),
]


@pytest.fixture
def card(monkeypatch):
    def pin(mib: int, capability: float) -> float:
        gb = mib / 1024
        monkeypatch.setattr(registry.sys, "platform", "linux")
        monkeypatch.setattr(registry.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(registry, "_cuda_vram_gb", lambda: gb)
        monkeypatch.setattr(registry, "_nvidia_compute_capability", lambda: capability)
        monkeypatch.setattr("bibr.local.llm_models.detect_hardware", lambda: ("cuda", gb))
        monkeypatch.setattr(sys, "platform", "linux")
        return gb

    return pin


def _spec_decode(backend: str, model: str, ocr: str, gb: float, mode: str) -> str | None:
    """The speculative decoding choice of the plan's managed vLLM LLM starts.

    Its server swaps with a managed PaddleOCR-VL server unless both shares
    fit (``ResourceManager._llm_swaps_with_ocr``); the budget is the
    configured 0.85 of the card, the most a start takes.
    """
    if backend != "vllm":
        return None
    swaps = ocr == "paddle-vllm" and not registry.managed_vllm_shares_fit(
        0.85, registry.paddle_vllm_gpu_memory_utilization(vram_gb=gb), vram_gb=gb
    )
    settings = GlobalSettings()
    settings.llm.vllm_spec_decode = mode
    choice = choose_spec_decode(settings, model, swaps_with_ocr=swaps, budget_gb=0.85 * gb)
    return "none" if choice.config is None else choice.config.method


def _runtime(gb: float) -> tuple[str, str, str, str, str | None]:
    backend = resolve_llm_backend("local")
    model = default_local_model(backend)
    ocr = registry.automatic_backend_names(llm_backend=backend)[0]
    return (
        backend,
        model,
        _auto_memory_mode("cuda", gb, 32.0, llm_backend=backend),
        ocr,
        _spec_decode(backend, model, ocr, gb, GlobalSettings().llm.vllm_spec_decode),
    )


def _setup(gb: float) -> tuple[str, str, str, str, str | None]:
    env = _build_recommended_setup(
        platform_key="cuda",
        accelerator_memory_gb=gb,
        system_memory_gb=32.0,
        sys_platform="linux",
        machine="x86_64",
    ).env
    ocr = env["OCR_BACKEND"]
    if ocr == "paddle":
        ocr = registry.automatic_backend_names(llm_backend=env["LLM_BACKEND"])[0]
    # Setup writes no speculative decoding setting: the runtime's default applies.
    assert "LLM_VLLM_SPEC_DECODE" not in env
    backend, model = env["LLM_BACKEND"], env["LLM_LOCAL_MODEL"]
    return (
        backend,
        model,
        env["PIPELINE_MEMORY_MODE"],
        ocr,
        _spec_decode(backend, model, ocr, gb, "auto"),
    )


@pytest.mark.parametrize(("row", "expected"), TIER_TABLE, ids=[row[0][0] for row in TIER_TABLE])
def test_the_runtime_default_equals_the_setup_default(card, row, expected):
    _name, mib, capability = row
    gb = card(mib, capability)
    assert _runtime(gb) == expected
    assert _setup(gb) == expected


_VLLM_ROWS = [(row, expected) for row, expected in TIER_TABLE if expected[0] == "vllm"]


@pytest.mark.parametrize(("row", "expected"), _VLLM_ROWS, ids=[r[0] for r, _e in _VLLM_ROWS])
def test_every_vllm_default_takes_turns_with_paddleocr(card, monkeypatch, row, expected):
    """A managed vLLM LLM that takes turns with PaddleOCR-VL may be squeezed to
    the model's floor for a chunk; one that stays up for the run must leave
    bibr's sessions their whole reserve (``bibr.local.vllm_share``). Every vLLM
    default takes turns, as did every run that backs one, so that rule changes
    none of them. The whole card counts as CUDA's here, which only makes the
    shares fit together more easily."""
    from bibr.config import GlobalSettings
    from bibr.pipeline.resources import ResourceManager

    _name, mib, capability = row
    gb = card(mib, capability)
    monkeypatch.setattr(registry, "_managed_gpu_vram_gb", lambda: gb)
    env = _build_recommended_setup(
        platform_key="cuda",
        accelerator_memory_gb=gb,
        system_memory_gb=32.0,
        sys_platform="linux",
        machine="x86_64",
    ).env
    settings = GlobalSettings()
    rm = ResourceManager(
        memory_mode=expected[2],
        ocr_backend=env["OCR_BACKEND"],
        managed_vllm_fraction=settings.llm.local_mem_fraction,
        llm_backend=env["LLM_BACKEND"],
        settings=settings,
    )
    assert rm._llm_swaps_with_ocr() is True
