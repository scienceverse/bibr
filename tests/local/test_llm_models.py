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
        ("mlx", 6.0, None, "scienceverse/bibr-extract-qwen35-4b-mlx-8bit"),
        ("mlx", 5.0, None, "scienceverse/bibr-extract-qwen35-2b-mlx-8bit"),
        ("mlx", 3.0, None, "scienceverse/bibr-extract-qwen35-2b-mlx-8bit"),
        # CPU: memory is system RAM, and the fastest build leads at any size.
        ("cpu", None, "llama-cpp", "scienceverse/bibr-extract-qwen35-2b-GGUF:Q4_K_M"),
        ("cpu", 64.0, "llama-cpp", "scienceverse/bibr-extract-qwen35-2b-GGUF:Q4_K_M"),
        ("cpu", 7.6, "llama-cpp", "scienceverse/bibr-extract-qwen35-2b-GGUF:Q4_K_M"),
        ("cpu", 6.0, None, "scienceverse/bibr-extract-qwen35-2b-GGUF:Q4_K_M"),
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
    # 5.9 GB of RAM cannot hold bibr's own models plus the smallest CPU build.
    assert recommended_variant("cpu", 5.9, "llama-cpp") is None


def test_cpu_builds_are_keyed_on_ram_in_speed_order():
    """2B Q4_K_M first, then 2B Q8_0, then 4B Q4_K_M; each floor is system RAM."""
    cpu_builds = [
        (m.key, v.quant, v.min_vram_gb)
        for key in llm_models.RECOMMENDED_MODEL_KEYS_BY_PLATFORM["cpu"]
        for m in [get_model(key)]
        for v in variants_for(m, "cpu", None)
    ]
    assert cpu_builds == [
        ("bibr-extract-2b", "gguf-q4_k_m", 6),
        ("bibr-extract-2b", "gguf-q8_0", 7),
        ("bibr-extract-4b", "gguf-q4_k_m", 8),
    ]
    # Each CPU build needs its GPU floor plus room for bibr's own CPU models.
    for m in REGISTRY:
        for v in variants_for(m, "cpu", None):
            gpu = next(g for g in variants_for(m, "cuda", None) if g.hf_id == v.hf_id)
            assert v.min_vram_gb == gpu.min_vram_gb + 3
            assert v.runtime == "llama-cpp"
    # A 16 GB laptop is offered all three; an "8 GB" one (7.6 GB usable) the 2B only.
    assert len(variants_for(get_model("bibr-extract-2b"), "cpu", 16.0)) == 2
    assert len(variants_for(get_model("bibr-extract-4b"), "cpu", 16.0)) == 1
    assert len(variants_for(get_model("bibr-extract-2b"), "cpu", 7.6)) == 2
    assert variants_for(get_model("bibr-extract-4b"), "cpu", 7.6) == []


def test_cpu_builds_are_experimental():
    """The CPU tier is neither timed nor scored, so setup labels it experimental."""
    for key in ("bibr-extract-4b", "bibr-extract-2b"):
        assert "cpu" in get_model(key).experimental_platforms
    assert "cuda" not in get_model("bibr-extract-4b").experimental_platforms


def test_gguf_builds_are_pinned_to_a_hub_commit():
    from bibr.local.llm_models import PinnedGguf, pinned_gguf

    gguf = [v for m in REGISTRY for v in m.variants if v.runtime == "llama-cpp"]
    assert gguf
    for v in gguf:
        repo, quant = v.hf_id.split(":")
        assert len(v.gguf_revision or "") == 40  # a commit, not a branch
        assert v.gguf_file == f"{repo.split('/')[1].removesuffix('-GGUF')}-{quant}.gguf"
        assert pinned_gguf(v.hf_id) == PinnedGguf(repo, v.gguf_file, v.gguf_revision)
    # The 2B Q4_K_M, the CPU default, at the commit that added it.
    assert pinned_gguf("scienceverse/bibr-extract-qwen35-2b-GGUF:Q4_K_M") == PinnedGguf(
        "scienceverse/bibr-extract-qwen35-2b-GGUF",
        "bibr-extract-qwen35-2b-Q4_K_M.gguf",
        "8e1a5823c4290cd583feb470556967df7259050a",
    )


def test_pinned_gguf_unknown_model():
    from bibr.local.llm_models import pinned_gguf

    assert pinned_gguf("someone/custom-GGUF:Q4_K_M") is None
    assert pinned_gguf("scienceverse/bibr-extract-qwen35-4b") is None  # vLLM build


def test_students_are_merged_vllm_checkpoints_with_thinking_off():
    for key in ("bibr-extract-4b", "bibr-extract-2b"):
        model = get_model(key)
        assert [v.runtime for v in model.variants][:3] == ["vllm", "llama-cpp", "llama-cpp"]
        assert {v.platform for v in model.variants[3:]} == {"mlx", "cpu"}
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


def test_default_local_model_on_a_cpu_build_is_the_fastest_cpu_build():
    assert (
        default_local_model("llama-cpp", cpu=True)
        == "scienceverse/bibr-extract-qwen35-2b-GGUF:Q4_K_M"
    )
    # Only llama.cpp has CPU builds; the flag changes nothing elsewhere.
    assert default_local_model("vllm", cpu=True) == "scienceverse/bibr-extract-qwen35-4b"


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
            assert v.platform in ("cuda", "mlx", "cpu")
            assert v.min_vram_gb > 0
            assert v.approx_download_gb > 0
            assert v.hf_id.count("/") == 1  # org/name
            if v.platform == "cpu":
                assert v.runtime == "llama-cpp"  # the only CPU runtime
        assert model.experimental_platforms <= {"cuda", "mlx", "cpu"}


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
        # MLX: smoke-tested only; CPU: neither timed nor scored.
        assert m.experimental_platforms == frozenset({"mlx", "cpu"})
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
        # Not ("cpu", ram): callers rely on (None, None) for "no GPU".
        assert detect_hardware() == (None, None)
