"""The per-chunk OCR teardown heuristic (_should_unload_ocr_after_chunk).

balanced mode used to tear down the OCR engine after every chunk, reloading
weights (vllm-mlx 30-60s) on each chunk of a batch. The only balanced consumer
of that freed VRAM is a *local* LLM server; with a cloud/remote LLM (the
default) the unload is pure thrash, so OCR now stays resident across chunks.
"""

import pytest

from bibr.config import Settings
from bibr.pipeline.stages.ocr import _should_unload_ocr_after_chunk


@pytest.fixture(autouse=True)
def _override_auto(monkeypatch):
    # Exercise the heuristic itself unless a test overrides the knob.
    monkeypatch.setattr(Settings.ocr, "unload_between_chunks", "auto", raising=False)


def test_balanced_cloud_keeps_ocr_loaded():
    assert _should_unload_ocr_after_chunk("balanced", "cloud") is False


def test_balanced_local_unloads_ocr():
    assert _should_unload_ocr_after_chunk("balanced", "local") is True


def test_balanced_resolved_local_backends_unload_ocr():
    # The CLI resolves --llm local to a concrete backend before RunConfig,
    # so the heuristic must match the resolved names, not just the alias.
    assert _should_unload_ocr_after_chunk("balanced", "vllm") is True
    assert _should_unload_ocr_after_chunk("balanced", "vllm-mlx") is True


def test_aggressive_always_unloads():
    assert _should_unload_ocr_after_chunk("aggressive", "cloud") is True
    assert _should_unload_ocr_after_chunk("aggressive", "local") is True


def test_keep_all_never_unloads():
    assert _should_unload_ocr_after_chunk("keep_all", "cloud") is False
    assert _should_unload_ocr_after_chunk("keep_all", "local") is False


def test_override_always_forces_unload(monkeypatch):
    monkeypatch.setattr(Settings.ocr, "unload_between_chunks", "always", raising=False)
    assert _should_unload_ocr_after_chunk("balanced", "cloud") is True
    assert _should_unload_ocr_after_chunk("keep_all", "local") is True


def test_override_never_keeps_loaded(monkeypatch):
    monkeypatch.setattr(Settings.ocr, "unload_between_chunks", "never", raising=False)
    assert _should_unload_ocr_after_chunk("aggressive", "local") is False
    assert _should_unload_ocr_after_chunk("balanced", "local") is False


# ---------------------------------------------------------------------------
# local-runtimes sweep: cloud vision OCR never unloads between chunks (2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["gemini", "openai", "anthropic"])
def test_cloud_vision_ocr_never_unloads_between_chunks(backend):
    """Cloud vision OCR holds no local weights: teardown only churns (2).

    Even with a local LLM backend following (the balanced unload trigger),
    tearing down a thin HTTPS client reclaims no VRAM.
    """
    assert _should_unload_ocr_after_chunk("balanced", "vllm-mlx", None, backend) is False
    assert _should_unload_ocr_after_chunk("aggressive", "vllm-mlx", None, backend) is False


def test_cloud_vision_skip_yields_to_explicit_always(monkeypatch):
    """OCR_UNLOAD_BETWEEN_CHUNKS=always still forces the teardown (2 guard)."""
    monkeypatch.setattr(Settings.ocr, "unload_between_chunks", "always", raising=False)
    assert _should_unload_ocr_after_chunk("balanced", "cloud", None, "gemini") is True


def test_local_ocr_still_unloads_with_local_llm():
    """Non-cloud backends keep the old balanced/local unload behaviour (2 guard)."""
    assert _should_unload_ocr_after_chunk("balanced", "vllm-mlx", None, "paddle") is True
    assert _should_unload_ocr_after_chunk("balanced", "vllm-mlx", None, None) is True


# ---------------------------------------------------------------------------
# local-runtimes sweep: the per-chunk call sites forward cfg.ocr_backend (2)
# ---------------------------------------------------------------------------


def _stage_ctx(*, ocr_backend, memory_mode="balanced", llm_backend="llama-cpp"):
    """Minimal pipeline context driving the real OCR-stage teardown tail."""
    from unittest.mock import AsyncMock, MagicMock

    from bibr.config import GlobalSettings
    from bibr.pipeline.context import RunConfig

    settings = GlobalSettings()
    cfg = RunConfig(memory_mode=memory_mode, llm_backend=llm_backend, ocr_backend=ocr_backend)
    rm = MagicMock()
    rm.ocr_runtime_identity = None
    rm.ocr = MagicMock(loaded=False)
    rm.shutdown_ocr = AsyncMock()
    ctx = MagicMock()
    ctx.config = cfg
    ctx.settings = settings
    ctx.resources = rm
    ctx.alive.return_value = []
    ctx.file_states = []
    ctx.scratch = {}
    ctx.signals = MagicMock(defer_ocr_teardown=False, ocr_init_error=None)
    ctx.progress = MagicMock()
    return ctx, rm


@pytest.mark.asyncio
async def test_ocr_stage_keeps_cloud_vision_loaded(monkeypatch):
    """OcrStage must pass cfg.ocr_backend to the teardown check (2).

    balanced + llama-cpp with gemini OCR keeps the client: dropping the
    backend argument would unload it (the old per-chunk thrash).
    """
    from unittest.mock import AsyncMock

    from bibr.pipeline.stages.ocr import OcrStage

    monkeypatch.setattr(OcrStage, "_run_local", AsyncMock())
    ctx, rm = _stage_ctx(ocr_backend="gemini")
    await OcrStage()._run(ctx)
    rm.shutdown_ocr.assert_not_awaited()


@pytest.mark.asyncio
async def test_ocr_stage_unloads_local_ocr_with_local_llm(monkeypatch):
    """Same config with paddle OCR still unloads for the local LLM (2 guard)."""
    from unittest.mock import AsyncMock

    from bibr.pipeline.stages.ocr import OcrStage

    monkeypatch.setattr(OcrStage, "_run_local", AsyncMock())
    ctx, rm = _stage_ctx(ocr_backend="paddle")
    await OcrStage()._run(ctx)
    rm.shutdown_ocr.assert_awaited_once()


@pytest.mark.asyncio
async def test_interleaved_stage_keeps_cloud_vision_loaded():
    """InterleavedRenderOcrStage must pass cfg.ocr_backend too (2).

    Its once-per-chunk teardown mirrors OcrStage's; dropping the backend
    would tear down a weightless HTTPS client after every chunk.
    """
    from unittest.mock import AsyncMock

    from bibr.pipeline.stages.render_ocr import InterleavedRenderOcrStage

    stage = InterleavedRenderOcrStage(layout=AsyncMock(), native_text=AsyncMock(), ocr=AsyncMock())
    ctx, rm = _stage_ctx(ocr_backend="gemini")
    await stage.run(ctx)
    rm.shutdown_ocr.assert_not_awaited()
