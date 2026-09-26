"""Tests for local pipeline chunked processing (FileState, _alive, _auto_batch_size)."""

import concurrent.futures
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bibr.local.pipeline import LocalPipeline, _auto_batch_size
from bibr.pipeline.state import FileState, _alive

# --- FileState cleanup ---


class TestFileStateFreePreOcr:
    def test_nulls_consumed_fields(self):
        fs = FileState(path=Path("test.pdf"))
        fs.pdf_bytes = b"fake-pdf"
        fs.page_images = ["img1", "img2"]
        fs.layout_results = [[], []]
        fs.page_indices = [0, 1]

        fs.free_pre_ocr()

        assert fs.pdf_bytes is None
        assert fs.page_images is None
        assert fs.layout_results is None

    def test_preserves_page_indices(self):
        fs = FileState(path=Path("test.pdf"))
        fs.page_indices = [0, 1, 2]
        fs.page_images = ["a", "b", "c"]

        fs.free_pre_ocr()

        assert fs.page_indices == [0, 1, 2]


class TestFileStateFreePreParse:
    def test_nulls_ocr_regions(self):
        fs = FileState(path=Path("test.pdf"))
        fs.ocr_regions = [[{"content": "text"}]]
        fs.contents = "placeholder"

        fs.free_pre_parse()

        assert fs.ocr_regions is None

    def test_preserves_contents(self):
        fs = FileState(path=Path("test.pdf"))
        fs.ocr_regions = [[]]
        fs.contents = "important"

        fs.free_pre_parse()

        assert fs.contents == "important"


class TestFileStateFreeAll:
    def test_nulls_all_intermediate(self):
        fs = FileState(path=Path("test.pdf"))
        fs.pdf_bytes = b"x"
        fs.page_images = []
        fs.page_indices = [0]
        fs.layout_results = [[]]
        fs.ocr_regions = [[]]
        fs.contents = "c"
        fs.paper = "p"

        fs.free_all()

        assert fs.pdf_bytes is None
        assert fs.page_images is None
        assert fs.page_indices is None
        assert fs.layout_results is None
        assert fs.ocr_regions is None
        assert fs.contents is None
        assert fs.paper is None

    def test_preserves_result_json(self):
        fs = FileState(path=Path("test.pdf"))
        fs.result_json = {"paper_id": "test"}
        fs.contents = "c"

        fs.free_all()

        assert fs.result_json == {"paper_id": "test"}

    def test_preserves_error(self):
        fs = FileState(path=Path("test.pdf"))
        fs.error = "something failed"

        fs.free_all()

        assert fs.error == "something failed"


# --- _alive helper ---


class TestAlive:
    def test_filters_errored_files(self):
        fs1 = FileState(path=Path("a.pdf"))
        fs2 = FileState(path=Path("b.pdf"))
        fs2.error = "failed"
        fs3 = FileState(path=Path("c.pdf"))

        result = _alive([fs1, fs2, fs3])

        assert result == [fs1, fs3]

    def test_all_alive(self):
        fs1 = FileState(path=Path("a.pdf"))
        fs2 = FileState(path=Path("b.pdf"))

        result = _alive([fs1, fs2])

        assert result == [fs1, fs2]

    def test_all_errored(self):
        fs1 = FileState(path=Path("a.pdf"))
        fs1.error = "err1"
        fs2 = FileState(path=Path("b.pdf"))
        fs2.error = "err2"

        result = _alive([fs1, fs2])

        assert result == []

    def test_empty_list(self):
        assert _alive([]) == []


# --- _auto_batch_size ---


class TestAutoBatchSize:
    def test_aggressive(self):
        assert _auto_batch_size("aggressive") == 3

    def test_balanced(self):
        assert _auto_batch_size("balanced") == 8

    def test_keep_all(self):
        assert _auto_batch_size("keep_all") == 20


# --- FileState defaults ---


class TestFileStateDefaults:
    def test_default_fields_are_none(self):
        fs = FileState(path=Path("test.pdf"))
        assert fs.pdf_bytes is None
        assert fs.page_images is None
        assert fs.page_indices is None
        assert fs.layout_results is None
        assert fs.ocr_regions is None
        assert fs.contents is None
        assert fs.paper is None
        assert fs.result_json is None
        assert fs.error is None
        assert fs.paper_id is None

    def test_stage_times_default_empty_dict(self):
        fs = FileState(path=Path("test.pdf"))
        assert fs.stage_times == {}

    def test_stage_times_independent_per_instance(self):
        fs1 = FileState(path=Path("a.pdf"))
        fs2 = FileState(path=Path("b.pdf"))
        fs1.stage_times["layout"] = 1.5
        assert "layout" not in fs2.stage_times

    def test_structured_error_defaults(self):
        fs = FileState(path=Path("test.pdf"))
        assert fs.error_code is None
        assert fs.failed_stage is None
        assert fs.warnings == []

    def test_set_error_populates_all_fields(self):
        fs = FileState(path=Path("test.pdf"))
        fs.set_error("OCR failed: timeout", code="ocr_timeout", stage="ocr")
        assert fs.error == "OCR failed: timeout"
        assert fs.error_code == "ocr_timeout"
        assert fs.failed_stage == "ocr"

    def test_a_later_error_replaces_the_outage_flag(self):
        fs = FileState(path=Path("test.pdf"))
        fs.set_error("OCR backend init failed", code="ocr_failed", stage="ocr", outage=True)
        assert fs.error_outage is True
        fs.set_error("Post-parse failed", code="post_parse_failed", stage="post_parse")
        assert fs.error_outage is False

    def test_warnings_independent_per_instance(self):
        fs1 = FileState(path=Path("a.pdf"))
        fs2 = FileState(path=Path("b.pdf"))
        fs1.warnings.append("crossref timed out")
        assert fs2.warnings == []


# --- pipeline lifetime: aclose tears down LLM server, process_chunk does not ---


@pytest.mark.asyncio
async def test_process_chunk_does_not_shut_down_llm_server_per_request():
    """Per-request finally must NOT teardown the local LLM server.

    The serve worker reuses a single pipeline across requests; tearing
    down vllm-mlx between calls would force a 30-90s reload every time.
    Owners must call ``aclose`` once at end of life instead.
    """
    pipe = LocalPipeline(llm_backend="cloud")
    pipe._resources = MagicMock()
    pipe._resources.shutdown_llm_server = MagicMock()
    pipe._resources.close_llm_client = AsyncMock()

    class KillStage:
        name = "kill"

        async def run(self, ctx):
            for fs in ctx.alive():
                fs.set_error("boom", code="x", stage="kill")

    pipe._stages = [KillStage()]

    fs = FileState(path=Path("x.pdf"))
    await pipe.process_chunk([fs])

    pipe._resources.shutdown_llm_server.assert_not_called()


@pytest.mark.asyncio
async def test_aclose_shuts_down_llm_server_and_ocr():
    """``Pipeline.aclose`` is the explicit teardown the owner calls once."""
    pipe = LocalPipeline(llm_backend="cloud")
    pipe._resources = MagicMock()
    pipe._resources.close_llm_server = AsyncMock()
    pipe._resources.close_llm_client = AsyncMock()

    async def _fake_shutdown_ocr():
        return None

    pipe._resources.shutdown_ocr = MagicMock(side_effect=_fake_shutdown_ocr)

    await pipe.aclose()

    pipe._resources.close_llm_server.assert_awaited_once()
    pipe._resources.shutdown_ocr.assert_called_once()
    pipe._resources.close_llm_client.assert_awaited_once()


# --- ResourceManager OCR executor cleanup ---


@pytest.mark.asyncio
async def test_await_ocr_cleans_up_executor_on_failure():
    from bibr.pipeline.resources import ResourceManager

    rm = ResourceManager()

    def boom():
        raise RuntimeError("init failed")

    rm._create_ocr_client = boom
    # Simulate that preload was started.
    rm._ocr_executor = concurrent.futures.ThreadPoolExecutor(1)
    rm._ocr_future = rm._ocr_executor.submit(boom)

    with pytest.raises(RuntimeError):
        await rm.await_ocr()

    assert rm._ocr_future is None, "future field must be cleared after failure"
    assert rm._ocr_executor is None, "executor must be shut down + cleared after failure"


# --- llm backend alias resolution (library API parity with the CLI) ---


class TestLlmBackendResolution:
    """LocalPipeline must resolve the "local" alias like the CLI does.

    chew(llm="local") used to pass "local" through unresolved:
    start_llm_server matched neither vllm-mlx nor vllm and silently
    started no server at all.
    """

    def test_local_alias_resolves_to_rapid_mlx_on_mac_arm_when_available(self, monkeypatch):
        import platform

        from bibr.local import rapid_mlx

        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(platform, "machine", lambda: "arm64")
        monkeypatch.setattr(rapid_mlx, "rapid_mlx_unavailable_reason", lambda executable=None: None)
        # This test isolates the LLM alias. Pin the unrelated OCR backend so
        # the result does not depend on a checkout-local OCR_BACKEND in .env.
        pipe = LocalPipeline(llm_backend="local", ocr_backend="glm-rapid-mlx")
        assert pipe.llm_backend == "rapid-mlx"
        assert pipe._config.llm_backend == "rapid-mlx"

    def test_local_alias_falls_back_to_vllm_mlx_on_mac_arm(self, monkeypatch):
        import platform

        from bibr.local import rapid_mlx

        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(platform, "machine", lambda: "arm64")
        monkeypatch.setattr(
            rapid_mlx,
            "rapid_mlx_unavailable_reason",
            lambda executable=None: "'rapid-mlx' was not found or is not executable",
        )
        # Isolate the LLM fallback from OCR auto-detection: this test
        # deliberately makes Rapid-MLX unavailable for the LLM decision.
        pipe = LocalPipeline(llm_backend="local", ocr_backend="glm-rapid-mlx")
        assert pipe.llm_backend == "vllm-mlx"
        assert pipe._config.llm_backend == "vllm-mlx"

    def test_local_alias_resolves_to_vllm_elsewhere(self, monkeypatch):
        import platform

        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setattr("bibr.local.llm_models.detect_hardware", lambda: ("cuda", 24.0))
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        pipe = LocalPipeline(llm_backend="local")
        assert pipe.llm_backend == "vllm"
        assert pipe._config.llm_backend == "vllm"

    def test_concrete_backends_pass_through(self):
        for name in ("cloud", "vllm-mlx", "vllm"):
            assert LocalPipeline(llm_backend=name).llm_backend == name

    def test_no_llm_forces_cloud_backend(self):
        # --no-llm must not launch a managed local server that will never be
        # called: the resolved backend is forced to "cloud" (no server start).
        pipe = LocalPipeline(llm_backend="vllm", no_llm=True)
        assert pipe.llm_backend == "cloud"
        assert pipe._config.llm_backend == "cloud"


# --- PIPELINE_MEMORY_MODE wiring ---


class TestMemoryModeSetting:
    """PIPELINE_MEMORY_MODE must actually be consumed.

    Resolution order: explicit arg/flag > PIPELINE_MEMORY_MODE > RAM
    auto-detect. The setting existed but nothing read it.
    """

    def test_setting_overrides_autodetect(self, monkeypatch):
        from bibr.config import Settings
        from bibr.local.pipeline import _default_memory_mode

        monkeypatch.setattr(Settings.pipeline, "memory_mode", "keep_all")
        assert _default_memory_mode() == "keep_all"

    def test_unset_falls_back_to_ram_detect(self, monkeypatch):
        import bibr.local.pipeline as lp
        from bibr.config import Settings

        monkeypatch.setattr(Settings.pipeline, "memory_mode", None)
        # Pin hardware detection so the RAM fallback is exercised deterministically
        # regardless of the machine the test runs on (e.g. an Apple Silicon dev box).
        monkeypatch.setattr("bibr.local.llm_models.detect_hardware", lambda: (None, None))
        monkeypatch.setattr(lp, "_get_system_memory_gb", lambda: 4.0)
        assert lp._default_memory_mode() == "aggressive"
        monkeypatch.setattr(lp, "_get_system_memory_gb", lambda: 32.0)
        assert lp._default_memory_mode() == "balanced"

    def test_auto_memory_mode_rule(self):
        from bibr.local.pipeline import _auto_memory_mode

        # Discrete NVIDIA GPUs: the VRAM ceiling is 8 GB.
        assert _auto_memory_mode("cuda", 6.0, 64.0) == "aggressive"
        assert _auto_memory_mode("cuda", 24.0, 64.0) == "balanced"
        # Apple Silicon hands unified memory from OCR to LLM sequentially and
        # the MLX models are small, so 16 GB runs balanced; only ≤8 GB needs
        # aggressive (via the system-RAM rule — accel_gb is unified memory).
        assert _auto_memory_mode("mlx", 8.0, 8.0) == "aggressive"
        assert _auto_memory_mode("mlx", 16.0, 16.0) == "balanced"
        assert _auto_memory_mode("mlx", 32.0, 32.0) == "balanced"
        # No detected accelerator: fall back to total system RAM.
        assert _auto_memory_mode(None, None, 4.0) == "aggressive"
        assert _auto_memory_mode(None, None, 32.0) == "balanced"

    def test_pipeline_default_uses_setting(self, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.pipeline, "memory_mode", "keep_all")
        assert LocalPipeline(llm_backend="cloud").memory_mode == "keep_all"

    def test_explicit_arg_wins_over_setting(self, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.pipeline, "memory_mode", "keep_all")
        pipe = LocalPipeline(memory_mode="aggressive", llm_backend="cloud")
        assert pipe.memory_mode == "aggressive"
