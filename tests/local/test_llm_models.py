"""Registry invariants + hardware detection for local LLM onboarding."""

from unittest.mock import patch

import pytest

from bibr.local import llm_models
from bibr.local.llm_models import (
    REGISTRY,
    LocalLLMModel,
    LocalLLMVariant,
    default_local_model,
    detect_hardware,
    get_model,
    registry_server_args,
    variants_for,
)


def test_cuda_llm_backend_follows_the_registry_fit():
    from bibr.local.llm_models import cuda_llm_backend_for

    assert cuda_llm_backend_for(None) == "vllm"  # unknown VRAM keeps the fast path
    assert cuda_llm_backend_for(6.0) == "llama-cpp"
    assert cuda_llm_backend_for(7.7) == "llama-cpp"
    assert cuda_llm_backend_for(8188 / 1024) == "vllm"  # an RTX 4060: the W4A16 build
    assert cuda_llm_backend_for(8.0) == "vllm"
    assert cuda_llm_backend_for(11.0) == "vllm"


@pytest.mark.parametrize(
    ("platform_key", "memory_gb", "runtime", "expected"),
    [
        ("cuda", None, "vllm", "scienceverse/bibr-extract-qwen35-4b"),
        ("cuda", 24.0, "vllm", "scienceverse/bibr-extract-qwen35-4b"),
        ("cuda", 20.0, "vllm", "scienceverse/bibr-extract-qwen35-4b"),
        ("cuda", 16.0, "vllm", "scienceverse/bibr-extract-qwen35-4b-W4A16"),
        ("cuda", 12.0, "vllm", "scienceverse/bibr-extract-qwen35-4b-W4A16"),
        ("cuda", 12227 / 1024, "vllm", "scienceverse/bibr-extract-qwen35-4b-W4A16"),  # RTX 5070
        ("cuda", 11.0, "vllm", "scienceverse/bibr-extract-qwen35-4b-W4A16"),
        ("cuda", 8.0, "vllm", "scienceverse/bibr-extract-qwen35-4b-W4A16"),
        ("cuda", 8188 / 1024, "vllm", "scienceverse/bibr-extract-qwen35-4b-W4A16"),  # RTX 4060
        ("cuda", 24.0, "llama-cpp", "scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0"),
        ("cuda", 7.0, "llama-cpp", "scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0"),
        ("cuda", 6.0, "llama-cpp", "scienceverse/bibr-extract-qwen35-4b-GGUF:Q4_K_M"),
        ("cuda", 4.0, "llama-cpp", "scienceverse/bibr-extract-qwen35-2b-GGUF:Q8_0"),
        ("cuda", 3.0, "llama-cpp", "scienceverse/bibr-extract-qwen35-2b-GGUF:Q4_K_M"),
        ("mlx", 32.0, None, "scienceverse/bibr-extract-qwen35-4b-mlx-8bit"),
        ("mlx", 6.0, None, "scienceverse/bibr-extract-qwen35-4b-mlx-8bit"),
        ("mlx", 5.0, None, "scienceverse/bibr-extract-qwen35-2b-mlx-8bit"),
        ("mlx", 3.0, None, "scienceverse/bibr-extract-qwen35-2b-mlx-8bit"),
    ],
)
def test_recommended_variant_picks_by_scenario(platform_key, memory_gb, runtime, expected):
    from bibr.local.llm_models import recommended_variant

    picked = recommended_variant(platform_key, memory_gb, runtime)
    assert picked is not None
    assert picked[1].hf_id == expected


def test_recommended_variant_none_when_nothing_fits():
    from bibr.local.llm_models import recommended_variant

    assert recommended_variant("cuda", 2.0, "vllm") is None
    assert recommended_variant("cuda", 2.0, "llama-cpp") is None


@pytest.mark.parametrize(
    ("key", "runtimes"),
    [
        ("bibr-extract-4b", ["vllm", "vllm", "llama-cpp", "llama-cpp"]),
        ("bibr-extract-2b", ["vllm", "llama-cpp", "llama-cpp"]),
    ],
)
def test_students_are_merged_vllm_checkpoints_with_thinking_off(key, runtimes):
    model = get_model(key)
    assert [v.runtime for v in model.variants][: len(runtimes)] == runtimes
    assert {v.platform for v in model.variants[len(runtimes) :]} == {"mlx"}
    # llama.cpp and MLX servers do not read server_args: thinking goes off per request.
    assert model.env == {"LLM_CHAT_TEMPLATE_KWARGS": '{"enable_thinking": false}'}
    for variant in model.variants[: runtimes.count("vllm")]:
        args = registry_server_args(variant.hf_id)
        assert "--enable-lora" not in args  # merged, so spec decoding stays usable
        assert "--language-model-only" in args
        revision = args[args.index("--revision") + 1]
        assert len(revision) == 40  # a pinned Hub commit, not a branch
        kwargs_idx = args.index("--default-chat-template-kwargs")
        assert args[kwargs_idx + 1] == '{"enable_thinking": false}'


def _vllm_variants():
    return [v for m in REGISTRY for v in m.variants if v.runtime == "vllm"]


@pytest.mark.parametrize("variant", _vllm_variants(), ids=lambda v: v.hf_id)
def test_students_cap_running_sequences_below_their_mamba_blocks(variant):
    """vLLM 0.31 refused the 4B at 0.54 of a 24 GB card: "max_num_seqs (256)
    exceeds available Mamba cache blocks (165)". bibr ran at most 19 at once."""
    args = registry_server_args(variant.hf_id)
    max_num_seqs = int(args[len(args) - 1 - args[::-1].index("--max-num-seqs") + 1])
    assert 19 < max_num_seqs <= 165


@pytest.mark.parametrize("variant", _vllm_variants(), ids=lambda v: v.hf_id)
def test_vllm_variants_have_a_measured_budget_floor_below_their_tier(variant):
    # The floor is a budget (a share of CUDA's total), the tier a card size.
    assert variant.min_budget_gb is not None
    assert 0 < variant.min_budget_gb < variant.min_vram_gb


def test_budget_floors_and_tiers():
    by_id = {v.hf_id: v for v in _vllm_variants()}
    floors = {h: (v.min_budget_gb, v.min_vram_gb) for h, v in by_id.items()}
    assert floors == {
        "scienceverse/bibr-extract-qwen35-4b": (12.25, 20),
        "scienceverse/bibr-extract-qwen35-4b-W4A16": (6.25, 8),
        "scienceverse/bibr-extract-qwen35-2b": (6.25, 8),
    }


def test_w4a16_build_is_listed_at_its_published_revision():
    variant = get_model("bibr-extract-4b").variants[1]
    assert variant.hf_id == "scienceverse/bibr-extract-qwen35-4b-W4A16"
    assert (variant.platform, variant.quant, variant.runtime) == ("cuda", "w4a16", "vllm")
    assert variant.approx_download_gb == 3.8
    args = registry_server_args(variant.hf_id)
    assert args[:2] == ("--revision", "a2af8f6e90e6481251af9eb07d02c034c30e3446")
    assert args[-2:] == ("--max-num-seqs", "64")


class TestCardSizeTolerance:
    """Cards report a few MiB under their nominal size and still get its tier."""

    @pytest.mark.parametrize(
        ("mib", "tier"),
        [
            (8188, 8),  # RTX 4060
            (8151, 8),  # RTX 5060 laptop
            (12227, 12),  # RTX 5070
            (32607, 32),  # RTX 5090
            (24576, 24),  # RTX 3090
        ],
    )
    def test_real_card_totals_fit_their_size(self, mib, tier):
        assert llm_models.fits_card(tier, mib / 1024)

    @pytest.mark.parametrize(("mib", "tier"), [(11264, 12), (7680, 8), (10240, 11)])
    def test_the_next_smaller_sizes_do_not(self, mib, tier):
        assert not llm_models.fits_card(tier, mib / 1024)

    def test_the_tolerance_is_a_quarter_gib(self):
        assert llm_models.CARD_SIZE_TOLERANCE_GB == 0.25
        assert llm_models.fits_card(8, 7.75)
        assert not llm_models.fits_card(8, 7.74)

    def test_variants_for_uses_it(self):
        model = get_model("bibr-extract-4b")
        hf_ids = [v.hf_id for v in variants_for(model, "cuda", 8188 / 1024, "vllm")]
        assert hf_ids == ["scienceverse/bibr-extract-qwen35-4b-W4A16"]
        assert variants_for(model, "cuda", 7.5, "vllm") == []
        assert variants_for(get_model("bibr-extract-2b"), "cuda", 8151 / 1024, "vllm")


def test_default_local_model_follows_the_backend_runtime():
    """An unset LLM_LOCAL_MODEL must never hand vLLM or llama.cpp the MLX build."""
    assert default_local_model("vllm") == "scienceverse/bibr-extract-qwen35-4b"
    assert default_local_model("llama-cpp") == "scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0"
    assert default_local_model("vllm-mlx") == "scienceverse/bibr-extract-qwen35-4b-mlx-8bit"
    assert default_local_model("rapid-mlx") == "scienceverse/bibr-extract-qwen35-4b-mlx-8bit"
    with pytest.raises(ValueError, match="cloud"):
        default_local_model("cloud")


def test_registry_server_args_unknown_model_empty():
    assert registry_server_args("someone/custom-model") == ()


def test_registry_keys_unique():
    keys = [m.key for m in REGISTRY]
    assert len(keys) == len(set(keys))


def test_registry_invariants():
    for model in REGISTRY:
        assert model.key and model.label
        assert model.variants, f"{model.key} has no variants"
        for v in model.variants:
            assert v.platform in ("cuda", "mlx")
            assert v.min_vram_gb > 0
            assert v.approx_download_gb > 0
            assert v.hf_id.count("/") == 1  # org/name
        assert model.experimental_platforms <= {"cuda", "mlx"}


def test_4b_student_is_first_and_recommended():
    assert REGISTRY[0].key == "bibr-extract-4b"
    assert "recommended" in REGISTRY[0].label.lower()
    assert [m.key for m in REGISTRY[:2]] == list(llm_models.RECOMMENDED_MODEL_KEYS)


def test_nuextract_is_gone_from_the_registry():
    """NuExtract 3 was replaced by the students on every platform."""
    hf_ids = [v.hf_id for m in REGISTRY for v in m.variants]
    assert not [h for h in hf_ids if h.lower().startswith("numind/")]
    with pytest.raises(KeyError):
        get_model("nuextract3")


def test_students_have_low_vram_llama_cpp_quants():
    variants = variants_for(get_model("bibr-extract-2b"), "cuda", 3.0, "llama-cpp")
    assert [v.hf_id for v in variants] == ["scienceverse/bibr-extract-qwen35-2b-GGUF:Q4_K_M"]
    assert variants[0].approx_download_gb < 2


def test_students_mlx_is_experimental():
    for key in ("bibr-extract-4b", "bibr-extract-2b"):
        m = get_model(key)
        assert m.experimental_platforms == frozenset({"mlx"})
        mlx = [v for v in m.variants if v.platform == "mlx"]
        # Only the 8-bit builds passed their smoke tests.
        assert [v.quant for v in mlx] == ["8bit"]


def test_get_model_unknown_key_raises():
    import pytest

    with pytest.raises(KeyError):
        get_model("nope")


def test_variants_for_filters_platform_and_vram():
    m = LocalLLMModel(
        key="x",
        label="x",
        variants=(
            LocalLLMVariant("o/a", "cuda", "bf16", min_vram_gb=16, approx_download_gb=8),
            LocalLLMVariant("o/b", "cuda", "awq-4bit", min_vram_gb=6, approx_download_gb=3),
            LocalLLMVariant("o/c", "mlx", "nvfp4", min_vram_gb=6, approx_download_gb=3),
        ),
        env={},
        experimental_platforms=frozenset(),
    )
    assert [v.hf_id for v in variants_for(m, "cuda", None)] == ["o/a", "o/b"]
    assert [v.hf_id for v in variants_for(m, "cuda", 8.0)] == ["o/b"]
    assert [v.hf_id for v in variants_for(m, "mlx", 4.0)] == []


def test_detect_hardware_cuda():
    with (
        patch.object(llm_models, "_nvidia_vram_gb", return_value=24.0),
        patch("platform.system", return_value="Linux"),
    ):
        assert detect_hardware() == ("cuda", 24.0)


def test_detect_hardware_mac_arm():
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
        patch.object(llm_models, "_system_memory_gb", return_value=32.0),
    ):
        assert detect_hardware() == ("mlx", 32.0)


def test_detect_hardware_none():
    with (
        patch("platform.system", return_value="Linux"),
        patch.object(llm_models, "_nvidia_vram_gb", return_value=None),
    ):
        assert detect_hardware() == (None, None)
