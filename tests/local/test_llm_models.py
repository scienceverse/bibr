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
    assert cuda_llm_backend_for(7.9) == "llama-cpp"
    assert cuda_llm_backend_for(8.0) == "vllm"  # the 2B student fits
    assert cuda_llm_backend_for(11.0) == "vllm"


@pytest.mark.parametrize(
    ("platform_key", "memory_gb", "runtime", "expected"),
    [
        ("cuda", None, "vllm", "scienceverse/bibr-extract-qwen35-4b"),
        ("cuda", 24.0, "vllm", "scienceverse/bibr-extract-qwen35-4b"),
        ("cuda", 12.0, "vllm", "scienceverse/bibr-extract-qwen35-4b"),
        ("cuda", 11.0, "vllm", "scienceverse/bibr-extract-qwen35-2b"),
        ("cuda", 8.0, "vllm", "scienceverse/bibr-extract-qwen35-2b"),
        ("cuda", 24.0, "llama-cpp", "scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0"),
        ("cuda", 7.0, "llama-cpp", "scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0"),
        ("cuda", 6.0, "llama-cpp", "scienceverse/bibr-extract-qwen35-4b-GGUF:Q4_K_M"),
        ("cuda", 4.0, "llama-cpp", "scienceverse/bibr-extract-qwen35-2b-GGUF:Q8_0"),
        ("cuda", 3.0, "llama-cpp", "scienceverse/bibr-extract-qwen35-2b-GGUF:Q4_K_M"),
        ("mlx", 32.0, None, "scienceverse/bibr-extract-qwen35-4b-mlx-8bit"),
        ("mlx", 5.0, None, "scienceverse/bibr-extract-qwen35-4b-mlx-4bit"),
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


def test_students_are_merged_vllm_checkpoints_with_thinking_off():
    for key in ("bibr-extract-4b", "bibr-extract-2b"):
        model = get_model(key)
        assert [v.runtime for v in model.variants][:3] == ["vllm", "llama-cpp", "llama-cpp"]
        assert {v.platform for v in model.variants[3:]} == {"mlx"}
        # llama.cpp and MLX servers do not read server_args: thinking goes off per request.
        assert model.env == {"LLM_CHAT_TEMPLATE_KWARGS": '{"enable_thinking": false}'}
        args = registry_server_args(model.variants[0].hf_id)
        assert "--enable-lora" not in args  # merged, so spec decoding stays usable
        assert "--language-model-only" in args
        revision = args[args.index("--revision") + 1]
        assert len(revision) == 40  # a pinned Hub commit, not a branch
        kwargs_idx = args.index("--default-chat-template-kwargs")
        assert args[kwargs_idx + 1] == '{"enable_thinking": false}'


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
        assert any(v.platform == "mlx" for v in m.variants)


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
