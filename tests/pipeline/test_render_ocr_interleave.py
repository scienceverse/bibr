"""InterleavedRenderOcrStage — local-only render→OCR interleaving.

Batch mode used to render every file's pages to full-resolution PIL images
up front (LayoutStage) and hold them all resident until the *whole* OCR stage
finished. On a chunk of 8 papers that pins ~2-6 GB of pixels across two
stages; on a 16 GB unified-memory Mac (sharing RAM with the OCR/LLM models)
it spills into swap. This stage runs Layout→NativeText→OCR one window at a
time and frees each window's page images before the next, so page-image RAM
is bounded by the window (1 file for local OCR), not the chunk size.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bibr.config import Settings
from bibr.input.pdf_outline import OutlineItem
from bibr.ocr.types import OcrRegionResult
from bibr.pipeline import ocr_cache
from bibr.pipeline.context import PipelineContext, RunConfig, StageSignals
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.render_ocr import InterleavedRenderOcrStage
from bibr.pipeline.state import FileState


@pytest.fixture(autouse=True)
def fake_page_count(monkeypatch):
    # These orchestration tests inject one-page layout/OCR doubles, not PDFs.
    monkeypatch.setattr("bibr.ocr.utils.get_pdf_page_count", lambda _: 1)


class _CountingLayout:
    """Fake LayoutStage: renders (marks page_images) and records the peak
    number of files holding page images simultaneously across the whole batch."""

    name = "layout"
    requires = ("pdf_bytes",)
    produces = ("page_images", "page_indices", "layout_results")

    def __init__(self, all_files, peak):
        self._all = all_files
        self._peak = peak

    async def run(self, ctx):
        for fs in ctx.alive():
            fs.page_images = [object()]  # stand-in for rendered PIL pages
            fs.page_indices = [0]
            fs.layout_results = [[]]
        live = sum(1 for fs in self._all if fs.page_images is not None)
        self._peak[0] = max(self._peak[0], live)


class _NoopNative:
    name = "native_text"
    requires = ("pdf_bytes", "layout_results")
    produces = ("ref_line_geometry", "native_metadata")

    async def run(self, ctx):
        return


class _CountingOcr:
    """Fake OcrStage: records peak resident page images at OCR time, then
    produces regions. Mirrors the real stage's ``defer_ocr_teardown`` contract
    by only tearing down when the flag is unset."""

    name = "ocr"
    requires = ("layout_results", "page_images", "page_indices")
    produces = ("ocr_regions",)

    def __init__(self, all_files, peak):
        self._all = all_files
        self._peak = peak

    async def run(self, ctx):
        live = sum(1 for fs in self._all if fs.page_images is not None)
        self._peak[0] = max(self._peak[0], live)
        for fs in ctx.alive():
            fs.ocr_regions = [[]]


def _mk_files(n: int) -> list[FileState]:
    files = [FileState(path=Path(f"{i}.pdf")) for i in range(n)]
    for fs in files:
        fs.pdf_bytes = b"%PDF-1.4"
    return files


def _stage(all_files, peak) -> InterleavedRenderOcrStage:
    return InterleavedRenderOcrStage(
        layout=_CountingLayout(all_files, peak),
        native_text=_NoopNative(),
        ocr=_CountingOcr(all_files, peak),
    )


def _ctx(files, config):
    rm = MagicMock()
    rm.shutdown_ocr = AsyncMock(return_value=None)
    return PipelineContext(
        file_states=files,
        progress=NullProgress(),
        resources=rm,
        config=config,
    )


@pytest.mark.asyncio
async def test_complete_cache_hit_skips_render_native_and_ocr(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings.cache, "ocr", True)
    monkeypatch.setattr(Settings.cache, "ocr_dir", str(tmp_path))
    cfg = RunConfig(
        ocr_backend="glm-mlx",
        llm_backend="cloud",
        memory_mode="balanced",
        ref_seg_strategy="geom",
    )
    cached = FileState(path=Path("cached.pdf"), file_hash="same-pdf")
    cached.native_metadata = {"title": "Cached title"}
    cached.ref_line_geometry = [{"text": "Cached reference"}]
    cached.pdf_outline = [OutlineItem(title="Results", level=1, page_no=2, y_top=10.0)]
    regions = [
        [
            OcrRegionResult(
                index=0,
                native_label="text",
                label="text",
                content="Cached text",
                bbox_2d=[0.0, 0.0, 10.0, 10.0],
            )
        ]
    ]
    from bibr.ocr.profiles import resolve_ocr_runtime_identity

    ocr_cache.store(cached, cfg, resolve_ocr_runtime_identity(cfg, Settings), regions)

    incoming = FileState(path=Path("incoming.pdf"), file_hash="same-pdf")
    incoming.pdf_bytes = b"%PDF-1.4"
    layout = MagicMock(run=AsyncMock())
    native = MagicMock(run=AsyncMock())
    ocr = MagicMock(run=AsyncMock())

    await InterleavedRenderOcrStage(layout=layout, native_text=native, ocr=ocr).run(
        _ctx([incoming], cfg)
    )

    layout.run.assert_not_awaited()
    native.run.assert_not_awaited()
    ocr.run.assert_not_awaited()
    assert incoming.pdf_bytes is None
    assert incoming.ocr_regions is not None
    assert incoming.ocr_regions[0][0].content == "Cached text"
    assert incoming.native_metadata == cached.native_metadata
    assert incoming.ref_line_geometry == cached.ref_line_geometry
    assert incoming.pdf_outline == cached.pdf_outline


@pytest.mark.asyncio
async def test_local_ocr_caps_page_images_to_one_file():
    files = _mk_files(6)
    peak = [0]
    ctx = _ctx(files, RunConfig(ocr_backend="glm-mlx", memory_mode="balanced"))

    await _stage(files, peak).run(ctx)

    # Local OCR is sequential → window of 1 → never more than one file's pages
    # resident at once, regardless of the 6-file chunk.
    assert peak[0] == 1
    # Every file was OCR'd and its page images freed afterwards.
    assert all(fs.ocr_regions is not None for fs in files)
    assert all(fs.page_images is None for fs in files)
    assert all(fs.pdf_bytes is None for fs in files)


@pytest.mark.asyncio
async def test_remote_ocr_caps_page_images_to_concurrency_window():
    from bibr.config import Settings

    files = _mk_files(6)
    peak = [0]
    ctx = _ctx(files, RunConfig(ocr_backend="gemini", memory_mode="balanced"))

    await _stage(files, peak).run(ctx)

    # Remote OCR keeps cross-file HTTP concurrency: the window is the OCR
    # file-concurrency cap, so peak resident images == max_concurrent_files.
    assert peak[0] == Settings.ocr.max_concurrent_files
    assert all(fs.page_images is None for fs in files)


@pytest.mark.asyncio
async def test_teardown_runs_once_not_per_window():
    files = _mk_files(4)
    peak = [0]
    # balanced + a managed local LLM backend → OCR must free VRAM for the LLM,
    # so a teardown IS expected — but exactly once for the whole chunk, not
    # once per window (which would thrash-reload the OCR engine per file).
    ctx = _ctx(
        files, RunConfig(ocr_backend="glm-mlx", memory_mode="balanced", llm_backend="vllm-mlx")
    )

    await _stage(files, peak).run(ctx)

    assert ctx.signals.defer_ocr_teardown is True
    ctx.resources.shutdown_ocr.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_teardown_for_cloud_llm():
    files = _mk_files(3)
    peak = [0]
    ctx = _ctx(files, RunConfig(ocr_backend="glm-mlx", memory_mode="balanced", llm_backend="cloud"))

    await _stage(files, peak).run(ctx)

    ctx.resources.shutdown_ocr.assert_not_called()


async def _run_real_ocr_stage(*, defer: bool, llm_backend: str):
    """Drive the REAL OcrStage through one file with mocked OCR I/O, so the
    ``defer_ocr_teardown`` gate is exercised end-to-end (the fused-stage tests
    above use a fake OCR)."""
    from unittest.mock import patch

    from bibr.pipeline.stages.ocr import OcrStage

    fs = FileState(path=Path("x.pdf"))
    fs.page_images = [MagicMock()]
    fs.page_indices = [0]
    fs.layout_results = [[]]
    rm = MagicMock()
    rm.ocr = MagicMock(recognize=AsyncMock(return_value="text"), loaded=True)
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    del rm.ocr.wait_for_server
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=rm,
        config=RunConfig(ocr_backend="glm-mlx", memory_mode="balanced", llm_backend=llm_backend),
        signals=StageSignals(any_needs_ocr=True, defer_ocr_teardown=defer),
    )
    with (
        patch(
            "bibr.pipeline.stages.ocr.ocr_page_regions",
            AsyncMock(return_value=[{"content": "hi"}]),
        ),
        patch(
            "bibr.pipeline.stages.ocr._postprocess_ocr_regions",
            side_effect=lambda x, *_args, **_kwargs: x,
        ),
    ):
        await OcrStage().run(ctx)
    return rm


@pytest.mark.asyncio
async def test_defer_flag_suppresses_ocr_stage_own_teardown():
    # vllm-mlx in balanced mode normally tears OCR down per chunk; the flag
    # must suppress that so the outer driver can do it once.
    rm = await _run_real_ocr_stage(defer=True, llm_backend="vllm-mlx")
    rm.shutdown_ocr.assert_not_called()


@pytest.mark.asyncio
async def test_ocr_stage_tears_down_when_not_deferred():
    # Regression guard: without the flag, the existing per-chunk teardown fires.
    rm = await _run_real_ocr_stage(defer=False, llm_backend="vllm-mlx")
    rm.shutdown_ocr.assert_awaited_once()


def test_stage_declares_contracts_for_static_validation():
    from bibr.pipeline.pipeline import validate_stage_contracts
    from bibr.pipeline.stages.export import ExportStage
    from bibr.pipeline.stages.parse_segment import ParseSegmentStage
    from bibr.pipeline.stages.post_parse import PostParseStage
    from bibr.pipeline.stages.validate import ValidateStage

    # The fused stage must satisfy the requires/produces contract in place of
    # the three stages it replaces: needs pdf_bytes (from validate), and
    # produces what parse/extract downstream require (ocr_regions, native_metadata).
    validate_stage_contracts(
        [
            ValidateStage(),
            InterleavedRenderOcrStage(),
            ParseSegmentStage(),
            PostParseStage(),
            ExportStage(),
        ]
    )
