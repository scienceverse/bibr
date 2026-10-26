"""The CUDA tier table: what ``bibr setup`` writes equals what the runtime picks.

For each card row, setup's recommended plan (``LLM_BACKEND``, ``LLM_LOCAL_MODEL``,
``PIPELINE_MEMORY_MODE``, and the OCR its ``OCR_BACKEND`` resolves to) must equal
the runtime's own resolution with nothing set but ``--llm local``: the
``local`` alias, the default model, the auto memory mode and the automatic
``paddle`` OCR chain. The rows are the docs' tier table
(docs/guides/extraction-models.md).
"""

import sys

import pytest

from bibr.local.llm_models import default_local_model
from bibr.local.pipeline import _auto_memory_mode, resolve_llm_backend
from bibr.ocr import registry
from bibr.setup_wizard import _build_recommended_setup

_4B = "scienceverse/bibr-extract-qwen35-4b"
_W4A16 = "scienceverse/bibr-extract-qwen35-4b-W4A16"
_Q8 = "scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0"
_Q4 = "scienceverse/bibr-extract-qwen35-4b-GGUF:Q4_K_M"

#: (card, nvidia-smi memory.total MiB, compute capability) -> (LLM backend,
#: model, memory mode, OCR runtime). Turing and Pascal rows are the
#: compute-capability routing rule, not measured cards.
TIER_TABLE = [
    (("RTX 5090", 32607, 12.0), ("vllm", _4B, "balanced", "paddle-vllm")),
    (("RTX 3090", 24576, 8.6), ("vllm", _4B, "balanced", "paddle-vllm")),
    (("RTX 4000 Ada", 20475, 8.9), ("vllm", _4B, "balanced", "paddle-vllm")),
    (("RTX 4080", 16376, 8.9), ("vllm", _W4A16, "balanced", "paddle-vllm")),
    (("RTX 3060 12 GB", 12288, 8.6), ("vllm", _W4A16, "balanced", "paddle-vllm")),
    (("RTX 5070", 12227, 12.0), ("vllm", _W4A16, "balanced", "paddle-vllm")),
    (("RTX 3080 10 GB", 10240, 8.6), ("vllm", _W4A16, "aggressive", "paddle-vllm")),
    (("RTX 3070", 8192, 8.6), ("llama-cpp", _Q8, "aggressive", "glm-llama")),
    (("RTX 4060", 8188, 8.9), ("llama-cpp", _Q8, "aggressive", "glm-llama")),
    (("RTX 3050 6 GB", 6144, 8.6), ("llama-cpp", _Q4, "aggressive", "glm-llama")),
    (("TITAN RTX", 24576, 7.5), ("llama-cpp", _Q8, "balanced", "glm-llama")),
    (("RTX 2080 Ti", 11264, 7.5), ("llama-cpp", _Q8, "aggressive", "glm-llama")),
    (("GTX 1080 Ti", 11264, 6.1), ("llama-cpp", _Q8, "aggressive", "glm-llama")),
    (("RTX 2060", 6144, 7.5), ("llama-cpp", _Q4, "aggressive", "glm-llama")),
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


def _runtime(gb: float) -> tuple[str, str, str, str]:
    backend = resolve_llm_backend("local")
    return (
        backend,
        default_local_model(backend),
        _auto_memory_mode("cuda", gb, 32.0, llm_backend=backend),
        registry.automatic_backend_names(llm_backend=backend)[0],
    )


def _setup(gb: float) -> tuple[str, str, str, str]:
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
    return env["LLM_BACKEND"], env["LLM_LOCAL_MODEL"], env["PIPELINE_MEMORY_MODE"], ocr


@pytest.mark.parametrize(("row", "expected"), TIER_TABLE, ids=[row[0][0] for row in TIER_TABLE])
def test_the_runtime_default_equals_the_setup_default(card, row, expected):
    _name, mib, capability = row
    gb = card(mib, capability)
    assert _runtime(gb) == expected
    assert _setup(gb) == expected
