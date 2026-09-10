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
    assert cuda_llm_backend_for(8.0) == "llama-cpp"
    assert cuda_llm_backend_for(10.0) == "llama-cpp"
    assert cuda_llm_backend_for(11.0) == "vllm"


def test_default_local_model_follows_the_backend_runtime():
    """An unset LLM_LOCAL_MODEL must never hand vLLM or llama.cpp the MLX build."""
    assert default_local_model("vllm") == "numind/NuExtract3"
    assert default_local_model("llama-cpp") == "numind/NuExtract3-GGUF:Q4_K_M"
    assert default_local_model("vllm-mlx") == "numind/NuExtract3-mlx-8bits"
    assert default_local_model("rapid-mlx") == "numind/NuExtract3-mlx-8bits"
    with pytest.raises(ValueError, match="cloud"):
        default_local_model("cloud")


def test_registry_server_args_nuextract3_cuda_keeps_mtp_opt_in():
    args = registry_server_args("numind/NuExtract3")
    assert "--speculative-config" not in args
    assert "--trust-remote-code" not in args
    revision_idx = args.index("--revision")
    assert args[revision_idx + 1] == "2e9fca82ee641e6bb6e1f5d905241e994be27a07"
    assert "--max-model-len" in args


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


def test_nuextract3_is_first_and_recommended():
    assert REGISTRY[0].key == "nuextract3"
    assert "recommended" in REGISTRY[0].label.lower()


def test_nuextract3_has_low_vram_llama_cpp_quant():
    variants = variants_for(get_model("nuextract3"), "cuda", 6.0, "llama-cpp")
    assert [v.hf_id for v in variants] == ["numind/NuExtract3-GGUF:Q4_K_M"]
    assert variants[0].approx_download_gb < 3


def test_nuextract3_mlx_is_experimental():
    m = get_model("nuextract3")
    assert "mlx" in m.experimental_platforms
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
