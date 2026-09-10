"""Backend alias/platform resolution — the library-level single source of truth.

Previously this logic lived only in the CLI, so library calls ignored
``OCR_BACKEND`` and the Apple-Silicon default.
"""

import pytest


class _OcrSettings:
    paddle_model = "paddle/source"
    paddle_mlx_model = "paddle/mlx"
    paddle_rapid_mlx_model = "paddle/rapid"
    paddle_served_model = "paddle-served"
    rapid_mlx_model = "glm/rapid"
    llama_cpp_model = "glm/llama"
    model = "glm/llama"
    local_model = "glm/local"


class _CandidateSettings:
    ocr = _OcrSettings()


class TestResolveBackendCandidates:
    def test_paddle_candidates_are_platform_ordered(self, monkeypatch):
        from bibr.ocr import registry

        monkeypatch.setattr(registry, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(registry.sys, "platform", "linux")
        monkeypatch.setattr(registry.platform, "machine", lambda: "x86_64")
        assert [
            candidate.backend
            for candidate in registry.resolve_backend_candidates("paddle", _CandidateSettings())
        ] == [
            "paddle-vllm",
            "glm-llama",
        ]

        monkeypatch.setattr(registry.sys, "platform", "darwin")
        monkeypatch.setattr(registry.platform, "machine", lambda: "arm64")
        assert [
            candidate.backend
            for candidate in registry.resolve_backend_candidates("paddle", _CandidateSettings())
        ] == [
            "paddle-rapid-mlx",
            "paddle-mlx-vlm",
            "glm-rapid-mlx",
            "glm-llama",
        ]

        monkeypatch.setattr(registry.sys, "platform", "win32")
        assert [
            candidate.backend
            for candidate in registry.resolve_backend_candidates("paddle", _CandidateSettings())
        ] == ["glm-llama"]

    @pytest.mark.parametrize(
        "backend",
        ("glm-llama", "glm-rapid-mlx", "glm-llama", "paddle-vllm", "paddle-http"),
    )
    def test_explicit_backend_has_exactly_one_candidate(self, backend):
        from bibr.ocr import registry

        candidates = registry.resolve_backend_candidates(backend, _CandidateSettings())

        assert [candidate.backend for candidate in candidates] == [backend]


class TestResolveBackendName:
    def test_alias_glm_uses_platform_default(self, monkeypatch):
        from bibr.ocr import registry

        monkeypatch.setattr(registry.sys, "platform", "linux")
        monkeypatch.setattr(registry.platform, "machine", lambda: "x86_64")
        assert registry.resolve_backend_name("glm") == "glm-llama"

    def test_alias_glm_local_uses_platform_default(self, monkeypatch):
        from bibr.ocr import registry

        monkeypatch.setattr(registry.sys, "platform", "darwin")
        monkeypatch.setattr(registry.platform, "machine", lambda: "arm64")
        assert registry.resolve_backend_name("glm-local") == "glm-rapid-mlx"

    def test_alias_glm_uses_llama_on_windows(self, monkeypatch):
        from bibr.ocr import registry

        monkeypatch.setattr(registry.sys, "platform", "win32")
        assert registry.resolve_backend_name("glm") == "glm-llama"

    def test_removed_legacy_name_is_not_registered(self):
        from bibr.ocr import registry

        removed_name = "fal" + "con"
        assert registry.resolve_backend_name(removed_name) == removed_name
        with pytest.raises(ValueError, match=f"Unknown OCR backend: '{removed_name}'"):
            registry.create(removed_name)

    def test_legacy_http_alias(self):
        from bibr.ocr import registry

        assert registry.resolve_backend_name("http") == "glm-http"

    def test_concrete_names_pass_through(self):
        from bibr.ocr import registry

        for name in ("glm-llama", "glm-mlx", "gemini"):
            assert registry.resolve_backend_name(name) == name

    def test_none_falls_back_to_platform_default(self, monkeypatch):
        from bibr.ocr import registry

        class _Opts:
            model_fields_set = frozenset()
            backend = "glm-llama"

        class _Settings:
            ocr = _Opts()

        monkeypatch.setattr(registry, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(registry.sys, "platform", "linux")
        monkeypatch.setattr(registry.platform, "machine", lambda: "x86_64")
        assert registry.resolve_backend_name(None, settings=_Settings()) == "paddle-vllm"

    def test_none_honors_explicit_ocr_backend_setting(self, monkeypatch):
        from bibr.ocr import registry

        class _Opts:
            model_fields_set = frozenset({"backend"})
            backend = "glm-llama"

        class _Settings:
            ocr = _Opts()

        assert registry.resolve_backend_name(None, settings=_Settings()) == "glm-llama"


class TestLocalPipelineResolvesAliases:
    def test_pipeline_ocr_url_forces_glm_http_identity_before_automatic_selection(self):
        from bibr.local.pipeline import LocalPipeline
        from bibr.ocr.profiles import resolve_ocr_runtime_identity

        pipeline = LocalPipeline(ocr_url="http://ocr.example:8000", ocr_model="custom/glm")
        identity = resolve_ocr_runtime_identity(pipeline._config, pipeline._settings)

        assert pipeline._config.ocr_backend == "glm-http"
        assert pipeline._resources.ocr_backend == "glm-http"
        assert identity.backend == "glm-http"
        assert identity.profile == "glm"

    def test_pipeline_paddle_http_url_preserves_explicit_paddle_identity(self):
        from bibr.local.pipeline import LocalPipeline
        from bibr.ocr.profiles import resolve_ocr_runtime_identity

        pipeline = LocalPipeline(
            ocr_backend="paddle-http",
            ocr_url="http://ocr.example:8000",
            ocr_model="paddle/custom",
            ocr_profile="paddle",
        )
        identity = resolve_ocr_runtime_identity(pipeline._config, pipeline._settings)

        assert pipeline._config.ocr_backend == "paddle-http"
        assert pipeline._resources.ocr_backend == "paddle-http"
        assert identity.backend == "paddle-http"
        assert identity.model == "paddle/custom"
        assert identity.profile == "paddle"

    def test_pipeline_keeps_paddle_automatic_selector_for_resource_startup(self):
        from bibr.local.pipeline import LocalPipeline

        pipeline = LocalPipeline(ocr_backend="paddle")

        assert pipeline._config.ocr_backend == "paddle"

    def test_pipeline_expands_glm_alias(self, monkeypatch):
        from bibr.local.pipeline import LocalPipeline
        from bibr.ocr import registry

        monkeypatch.setattr(registry.sys, "platform", "linux")
        monkeypatch.setattr(registry.platform, "machine", lambda: "x86_64")
        pipeline = LocalPipeline(ocr_backend="glm")
        assert pipeline._config.ocr_backend == "glm-llama"

    def test_pipeline_default_uses_resolution(self, monkeypatch):
        import types

        from bibr.config import Settings

        # Shield from the test env's OCR_BACKEND: pretend it was never set.
        monkeypatch.setattr(
            Settings,
            "ocr",
            types.SimpleNamespace(model_fields_set=frozenset(), backend="glm-llama"),
        )
        from bibr.local.pipeline import LocalPipeline

        pipeline = LocalPipeline()
        assert pipeline._config.ocr_backend == "paddle"


def test_default_backend_apple_silicon_prefers_rapid_mlx(monkeypatch):
    """vllm-mlx 0.4 --mllm emitted NUL-riddled garbage on a real page
    (2026-07-09); prefer the rapid-mlx MLLM server whenever it is installed."""
    from bibr.ocr import registry

    monkeypatch.setattr(registry.sys, "platform", "darwin")
    monkeypatch.setattr(registry.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        "bibr.local.rapid_mlx.rapid_mlx_unavailable_reason", lambda executable=None: None
    )
    assert registry.default_backend() == "paddle-rapid-mlx"


def test_default_backend_apple_silicon_defers_availability_to_startup(monkeypatch):
    """The automatic chain handles Rapid-MLX availability transactionally."""
    from bibr.ocr import registry

    monkeypatch.setattr(registry.sys, "platform", "darwin")
    monkeypatch.setattr(registry.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        "bibr.local.rapid_mlx.rapid_mlx_unavailable_reason",
        lambda executable=None: "'rapid-mlx' was not found or is not executable",
    )
    assert registry.default_backend() == "paddle-rapid-mlx"


def test_default_backend_rapid_mlx_path_succeeds(monkeypatch):
    from bibr.ocr import registry

    monkeypatch.setattr(registry.sys, "platform", "darwin")
    monkeypatch.setattr(registry.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        "bibr.local.rapid_mlx.rapid_mlx_unavailable_reason", lambda executable=None: None
    )
    assert registry.default_backend() == "paddle-rapid-mlx"


def test_vllm_mlx_ocr_client_construction_is_disabled():
    """The GLM vllm-mlx --mllm OCR path must refuse to construct at all.

    It emitted NUL-corrupted text on the whole validated 0.4 line and
    independently leaks ~3.4GB through an uncapped vision cache.
    """
    from bibr.exceptions import UpstreamServiceError
    from bibr.local.ocr import raise_vllm_mlx_ocr_disabled

    with pytest.raises(UpstreamServiceError, match="glm-rapid-mlx"):
        raise_vllm_mlx_ocr_disabled("glm-mlx")


def test_default_backend_windows(monkeypatch):
    from bibr.ocr import registry

    monkeypatch.setattr(registry.sys, "platform", "win32")
    assert registry.default_backend() == "glm-llama"


def test_default_backend_linux_x86_64_prefers_managed_paddle_vllm(monkeypatch):
    from bibr.ocr import registry

    monkeypatch.setattr(registry.sys, "platform", "linux")
    monkeypatch.setattr(registry.platform, "machine", lambda: "x86_64")
    assert registry.default_backend() == "paddle-vllm"


def test_explicit_glm_llama_remains_a_concrete_backend():
    from bibr.ocr import registry

    assert registry.resolve_backend_name("glm-llama") == "glm-llama"


class TestLinuxChainIsHardwareAware:
    """A CPU-only or small-GPU Linux box must never spend its OCR startup
    budget bootstrapping vLLM: ``paddle-vllm`` is only a candidate where it can
    run, and the chain goes straight to llama.cpp otherwise."""

    @staticmethod
    def _linux(monkeypatch, vram):
        from bibr.ocr import registry

        monkeypatch.setattr(registry.sys, "platform", "linux")
        monkeypatch.setattr(registry.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(registry, "_cuda_vram_gb", lambda: vram)
        return registry

    def test_cpu_only_linux_skips_paddle_vllm(self, monkeypatch):
        registry = self._linux(monkeypatch, None)
        chain = registry.resolve_backend_candidates("paddle", _CandidateSettings())
        assert [c.backend for c in chain] == ["glm-llama"]

    def test_small_gpu_skips_paddle_vllm(self, monkeypatch):
        registry = self._linux(monkeypatch, 6.0)
        chain = registry.resolve_backend_candidates("paddle", _CandidateSettings())
        assert [c.backend for c in chain] == ["glm-llama"]

    def test_roomy_gpu_keeps_paddle_vllm_first(self, monkeypatch):
        registry = self._linux(monkeypatch, 24.0)
        chain = registry.resolve_backend_candidates("paddle", _CandidateSettings())
        assert [c.backend for c in chain] == ["paddle-vllm", "glm-llama"]

    def test_explicit_paddle_vllm_is_never_rewritten(self, monkeypatch):
        """An explicit choice stays explicit; the launcher reports the blocker."""
        registry = self._linux(monkeypatch, None)
        chain = registry.resolve_backend_candidates("paddle-vllm", _CandidateSettings())
        assert [c.backend for c in chain] == ["paddle-vllm"]

    def test_unavailable_reason_messages(self):
        from bibr.ocr import registry

        assert registry.paddle_vllm_unavailable_reason(vram_gb=24.0) is None
        assert "no NVIDIA GPU" in registry.paddle_vllm_unavailable_reason(vram_gb=None)
        low = registry.paddle_vllm_unavailable_reason(vram_gb=6.0)
        assert "6 GB" in low and "llama.cpp" in low
