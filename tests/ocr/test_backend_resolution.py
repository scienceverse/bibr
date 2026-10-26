"""Backend alias/platform resolution — the library-level single source of truth.

Previously this logic lived only in the CLI, so library calls ignored
``OCR_BACKEND`` and the Apple-Silicon default.
"""

import pytest

from bibr.ocr import registry as _registry_for_probe


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
            paddle_served_model = "paddle-served"
            llama_cpp_model = "glm/llama"

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
            paddle_served_model = "paddle-served"
            llama_cpp_model = "glm/llama"

        class _Settings:
            ocr = _Opts()

        assert registry.resolve_backend_name(None, settings=_Settings()) == "glm-llama"


class TestLocalPipelineResolvesAliases:
    def test_pipeline_bare_ocr_url_resolves_paddle_http_before_automatic_selection(self):
        from bibr.local.pipeline import LocalPipeline
        from bibr.ocr.profiles import resolve_ocr_runtime_identity

        pipeline = LocalPipeline(ocr_url="http://ocr.example:8000", ocr_model="custom/paddle")
        identity = resolve_ocr_runtime_identity(pipeline._config, pipeline._settings)

        assert pipeline._config.ocr_backend == "paddle-http"
        assert pipeline._resources.ocr_backend == "paddle-http"
        assert identity.backend == "paddle-http"
        assert identity.model == "custom/paddle"
        assert identity.profile == "paddle"

    def test_pipeline_glm_ocr_url_resolves_glm_http_before_automatic_selection(self):
        from bibr.local.pipeline import LocalPipeline
        from bibr.ocr.profiles import resolve_ocr_runtime_identity

        pipeline = LocalPipeline(
            ocr_backend="glm", ocr_url="http://ocr.example:8000", ocr_model="custom/glm"
        )
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

    def test_an_rtx_4060_keeps_glm_ocr_on_llama_cpp(self):
        # An RTX 4060 reports 8,188 MiB: no card-size tolerance for OCR, so
        # its automatic chain stays on llama.cpp, the OCR of its default pair.
        from bibr.ocr import registry

        assert registry.paddle_vllm_unavailable_reason(vram_gb=8188 / 1024) is not None
        assert registry.paddle_vllm_unavailable_reason(vram_gb=8.0) is None


# Captured at collection, before the suite's conftest pins it for each test.
_MANAGED_GPU_PROBE = _registry_for_probe._managed_gpu_vram_gb

# nvidia-smi totals (GiB) of common cards, as the VRAM probes report them.
_CARDS = {
    "12 GB": 12.0,
    "16 GB": 15.99,
    "24 GB": 24.0,
    "32 GB (RTX 5090)": 31.84,
    "48 GB": 47.99,
    "64 GB": 64.0,
    "80 GB": 79.65,
}
# vLLM multiplies its share by CUDA's total, which is smaller than
# nvidia-smi's: 31.45 vs 31.84 GiB on an RTX 5090, 23.56 vs 24.0 on an RTX 3090.
_CUDA_OF_SMI = 0.98
# What a server restarting beside bibr's resident ONNX sessions needs outside
# its share on an RTX 5090: bibr held 2.43-2.77 GiB at the later chunks' OCR
# restarts, and the new server's CUDA context takes about 0.49 GiB.
_RESTART_NEED_GIB = 2.77 + 0.49


class TestPaddleVllmShareIsSizedFromItsNeed:
    """The managed PaddleOCR-VL server asks vLLM for about 5 GB, not 92% of the card.

    vLLM will not start unless its whole share is free, and bibr's layout, SaT
    and NER ONNX sessions stay on the GPU between chunks. On an RTX 5090 they
    held 2.43 GiB when the second chunk restarted OCR, so 0.92 (28.93 of 31.45
    GiB) never fitted again and every paper after the first chunk failed.
    """

    @pytest.mark.parametrize(
        ("vram_gb", "share"),
        [
            (12.0, 0.42),
            (15.99, 0.32),
            (24.0, 0.21),
            (23.99, 0.21),  # RTX 4090
            (31.84, 0.16),  # RTX 5090
            (47.99, 0.11),
            (64.0, 0.08),
            (79.65, 0.07),
            (10.0, 0.50),
            (8.0, 0.63),  # paddle-vllm's floor
        ],
    )
    def test_share_by_card(self, vram_gb, share):
        from bibr.ocr import registry

        assert registry.paddle_vllm_gpu_memory_utilization(vram_gb=vram_gb) == share

    @pytest.mark.parametrize("vram_gb", [None, 0.0, 4.0])
    def test_unknown_or_tiny_card_keeps_the_cap(self, vram_gb):
        from bibr.ocr import registry

        assert (
            registry.paddle_vllm_gpu_memory_utilization(vram_gb=vram_gb)
            == registry.PADDLE_VLLM_GPU_MEMORY_UTILIZATION
        )

    def test_probes_the_managed_servers_card_by_default(self, monkeypatch):
        from bibr.ocr import registry

        monkeypatch.setattr(registry, "_managed_gpu_vram_gb", lambda: 31.84)
        assert registry.paddle_vllm_gpu_memory_utilization() == 0.16

    @pytest.mark.parametrize("vram_gb", [8.0, 10.0, *_CARDS.values()])
    def test_share_holds_paddleocr_vl(self, vram_gb):
        """At 0.20 of a 24 GB card (4.71 GiB of CUDA memory) PaddleOCR-VL ran 20/20
        papers with a 2.26 GiB KV cache, at least 10x its peak use."""
        from bibr.ocr import registry

        share = registry.paddle_vllm_gpu_memory_utilization(vram_gb=vram_gb)
        assert share * vram_gb * _CUDA_OF_SMI >= 4.7

    @pytest.mark.parametrize("vram_gb", [10.0, *_CARDS.values()])
    def test_share_leaves_room_to_restart_beside_the_onnx_sessions(self, vram_gb):
        """0.92 left 2.5 GiB of a 32 GB card and 1.9 of 24 GB."""
        from bibr.ocr import registry

        share = registry.paddle_vllm_gpu_memory_utilization(vram_gb=vram_gb)
        assert (1 - share) * vram_gb * _CUDA_OF_SMI >= _RESTART_NEED_GIB


class TestManagedVllmSharesFit:
    """Both managed vLLM servers count as fitting only with bibr's ONNX reserve left over.

    Shares summing to 1 are not enough: at 0.85 + 0.15 on a 48 GB card, a
    later chunk's PaddleOCR-VL restart would find bibr's own sessions inside its
    share and never start, so every later paper would fail.
    """

    @pytest.mark.parametrize(
        ("card", "fits"),
        [
            ("12 GB", False),
            ("16 GB", False),
            ("24 GB", False),
            ("32 GB (RTX 5090)", False),
            ("48 GB", False),  # 0.85 + 0.11 = 0.96 leaves 1.9 GiB
            ("64 GB", False),  # 0.85 + 0.08 = 0.93 leaves 4.4 GiB
            ("80 GB", True),  # 0.85 + 0.07 = 0.92 leaves 6.2 GiB
        ],
    )
    def test_default_llm_share(self, card, fits):
        from bibr.ocr import registry

        vram = _CARDS[card]
        ocr = registry.paddle_vllm_gpu_memory_utilization(vram_gb=vram)
        assert registry.managed_vllm_shares_fit(0.85, ocr, vram_gb=vram) is fits

    @pytest.mark.parametrize(
        ("llm_share", "card", "fits"),
        [
            # LLM_LOCAL_MEM_FRACTION set by the user.
            (0.6, "24 GB", False),  # 0.81 of the card leaves 4.5 GiB
            (0.5, "24 GB", True),
            (0.85, "48 GB", False),
            (0.7, "48 GB", True),
            (0.5, "32 GB (RTX 5090)", True),
            (0.7, "32 GB (RTX 5090)", False),
            (0.05, "12 GB", True),  # 0.05 + 0.42 leaves 6.2 GiB
        ],
    )
    def test_user_set_llm_share(self, llm_share, card, fits):
        from bibr.ocr import registry

        vram = _CARDS[card]
        ocr = registry.paddle_vllm_gpu_memory_utilization(vram_gb=vram)
        assert registry.managed_vllm_shares_fit(llm_share, ocr, vram_gb=vram) is fits

    @pytest.mark.parametrize("card", list(_CARDS))
    @pytest.mark.parametrize("llm_share", [0.3, 0.5, 0.6, 0.7, 0.76, 0.85])
    def test_shares_that_fit_leave_the_reserve(self, card, llm_share):
        """Whenever the servers may be up together, at least ~5.9 GiB of CUDA memory
        stays outside both: what a restart needs, and the room at which layout
        stopped retrying after OOM on a 24 GB card (6.1 GiB)."""
        from bibr.ocr import registry

        vram = _CARDS[card]
        ocr = registry.paddle_vllm_gpu_memory_utilization(vram_gb=vram)
        if registry.managed_vllm_shares_fit(llm_share, ocr, vram_gb=vram):
            room = (1 - llm_share - ocr) * vram * _CUDA_OF_SMI
            assert room >= registry.MANAGED_VLLM_ONNX_RESERVE_GB * _CUDA_OF_SMI - 1e-6
            assert room >= _RESTART_NEED_GIB

    @pytest.mark.parametrize("vram_gb", [None, 0.0])
    def test_unknown_card_never_fits(self, vram_gb):
        from bibr.ocr import registry

        assert registry.managed_vllm_shares_fit(0.05, 0.05, vram_gb=vram_gb) is False

    def test_exact_fit_survives_float_noise(self):
        from bibr.ocr import registry

        # 0.54 + 0.21 + 6/24 is 1.0000000000000002 in floats.
        assert registry.managed_vllm_shares_fit(0.54, 0.21, vram_gb=24.0) is True


class TestManagedServersCard:
    """The PaddleOCR-VL share is sized from the card CUDA_VISIBLE_DEVICES exposes.

    vLLM runs on CUDA's device 0, which need not be nvidia-smi's GPU 0. An
    80 GB GPU 0 beside a visible 16 GB card used to give 0.09 of the 16 GB
    card, 1.4 GiB, less than PaddleOCR-VL's weights.
    """

    A100 = ("GPU-aaaaaaaa-1111-2222-3333-444444444444", 79.65)
    RTX4080 = ("GPU-bbbbbbbb-1111-2222-3333-444444444444", 15.99)
    RTX3090 = ("GPU-cccccccc-1111-2222-3333-444444444444", 24.0)
    RTX3090_2 = ("GPU-dddddddd-1111-2222-3333-444444444444", 24.0)

    @pytest.mark.parametrize(
        ("visible", "order", "expected"),
        [
            # PCI bus order: CUDA numbers the cards as nvidia-smi does.
            (None, "PCI_BUS_ID", 79.65),
            ("1", "PCI_BUS_ID", 15.99),
            ("1,0", "PCI_BUS_ID", 15.99),
            (" 0 ", "pci_bus_id", 79.65),
            # By UUID, full or a unique prefix: no ordering question.
            ("GPU-bbbbbbbb-1111-2222-3333-444444444444", None, 15.99),
            ("GPU-bbbb", None, 15.99),
            ("GPU-bbbb,GPU-aaaa", None, 15.99),
            # CUDA's default order is fastest first: cards of different sizes
            # cannot be told apart by number.
            (None, None, None),
            ("1", None, None),
            ("0", "FASTEST_FIRST", None),
            # No card, a MIG slice, an unknown UUID or an index out of range.
            ("", "PCI_BUS_ID", None),
            ("-1", "PCI_BUS_ID", None),
            ("NoDevFiles", None, None),
            ("MIG-12345678-1111-2222-3333-444444444444", None, None),
            ("GPU-ffff", None, None),
            ("GPU-", None, None),
            ("2", "PCI_BUS_ID", None),
        ],
    )
    def test_mixed_cards(self, visible, order, expected):
        from bibr.ocr import registry

        gpus = (self.A100, self.RTX4080)
        assert registry._visible_gpu_vram_gb(gpus, visible, order) == expected

    @pytest.mark.parametrize("visible", [None, "0", "1", "1,0"])
    @pytest.mark.parametrize("order", [None, "PCI_BUS_ID"])
    def test_cards_of_one_size_need_no_ordering(self, visible, order):
        from bibr.ocr import registry

        gpus = (self.RTX3090, self.RTX3090_2)
        assert registry._visible_gpu_vram_gb(gpus, visible, order) == 24.0

    @pytest.mark.parametrize("visible", [None, "0", "GPU-cccc"])
    def test_single_card(self, visible):
        from bibr.ocr import registry

        assert registry._visible_gpu_vram_gb((self.RTX3090,), visible, None) == 24.0

    @pytest.mark.parametrize("gpus", [None, ()])
    def test_no_inventory(self, gpus):
        from bibr.ocr import registry

        assert registry._visible_gpu_vram_gb(gpus, None, None) is None

    def _use_real_probe(self, monkeypatch, gpus, env):
        from bibr.ocr import registry

        monkeypatch.setattr(registry, "_nvidia_gpu_inventory", lambda: gpus)
        for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
            if name in env:
                monkeypatch.setenv(name, env[name])
            else:
                monkeypatch.delenv(name, raising=False)
        # The suite's conftest pins this probe; put the real one back, uncached.
        monkeypatch.setattr(registry, "_managed_gpu_vram_gb", _MANAGED_GPU_PROBE.__wrapped__)
        return registry

    def test_share_follows_the_visible_card(self, monkeypatch):
        registry = self._use_real_probe(
            monkeypatch,
            (self.A100, self.RTX4080),
            {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
        )
        assert registry.paddle_vllm_gpu_memory_utilization() == 0.32
        # GPU 0's 80 GB would have let 0.85 + 0.07 run together on the 16 GB card.
        assert registry.managed_vllm_shares_fit(0.85, 0.32) is False

    def test_card_that_cannot_be_told_keeps_the_cap_and_swaps(self, monkeypatch):
        registry = self._use_real_probe(
            monkeypatch, (self.A100, self.RTX4080), {"CUDA_VISIBLE_DEVICES": "1"}
        )
        assert registry.paddle_vllm_gpu_memory_utilization() == 0.92
        assert registry.managed_vllm_shares_fit(0.05, 0.05) is False

    def test_nothing_visible(self, monkeypatch):
        registry = self._use_real_probe(monkeypatch, (self.RTX3090,), {"CUDA_VISIBLE_DEVICES": ""})
        assert registry.paddle_vllm_gpu_memory_utilization() == 0.92

    def test_inventory_parses_nvidia_smi(self, monkeypatch):
        import subprocess

        from bibr.ocr import registry

        def fake_run(cmd, **kwargs):
            assert cmd[:2] == ["nvidia-smi", "--query-gpu=uuid,memory.total"]
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"{self.A100[0]}, 81559\n{self.RTX4080[0]}, 16376\n", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        gpus = registry._nvidia_gpu_inventory.__wrapped__()
        assert [uuid for uuid, _ in gpus] == [self.A100[0], self.RTX4080[0]]
        assert [round(total, 2) for _, total in gpus] == [79.65, 15.99]

    @pytest.mark.parametrize("outcome", ["error", "garbage", "missing"])
    def test_inventory_failures(self, monkeypatch, outcome):
        import subprocess

        from bibr.ocr import registry

        def fake_run(cmd, **kwargs):
            if outcome == "missing":
                raise FileNotFoundError("nvidia-smi")
            if outcome == "error":
                return subprocess.CompletedProcess(cmd, 9, stdout="", stderr="NVML init failed")
            return subprocess.CompletedProcess(cmd, 0, stdout="GPU-x, [N/A]\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert registry._nvidia_gpu_inventory.__wrapped__() is None
