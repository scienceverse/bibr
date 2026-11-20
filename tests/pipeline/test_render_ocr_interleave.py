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

import asyncio
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


@pytest.mark.parametrize(("mode", "trims"), [("window", 1), ("chunk", 0), ("off", 0)])
@pytest.mark.asyncio
async def test_window_mode_returns_freed_pages_between_windows_only(monkeypatch, mode, trims):
    from bibr.utils import heap

    files = _mk_files(6)
    peak = [0]
    resident_at_trim: list[int] = []

    async def trim():
        resident_at_trim.append(sum(1 for fs in files if fs.page_images is not None))
        return True

    monkeypatch.setattr(heap, "arelease_free_heap", trim)
    monkeypatch.setattr(Settings.ocr, "max_concurrent_files", 4)
    monkeypatch.setattr(Settings.pipeline, "heap_trim", mode)
    ctx = _ctx(files, RunConfig(ocr_backend="gemini", memory_mode="balanced"))

    await _stage(files, peak).run(ctx)

    # Two windows (4 + 2 files): one trim between them, none after the last,
    # which the batch driver's chunk-end trim covers. Each trim runs after its
    # window dropped its page images.
    assert resident_at_trim == [0] * trims


@pytest.mark.parametrize(("n_files", "trims"), [(1, 0), (4, 0), (8, 1), (9, 2)])
@pytest.mark.asyncio
async def test_window_mode_never_trims_after_the_last_window(monkeypatch, n_files, trims):
    from bibr.utils import heap

    files = _mk_files(n_files)
    calls: list[int] = []

    async def trim():
        calls.append(1)
        return True

    monkeypatch.setattr(heap, "arelease_free_heap", trim)
    monkeypatch.setattr(Settings.ocr, "max_concurrent_files", 4)
    monkeypatch.setattr(Settings.pipeline, "heap_trim", "window")
    ctx = _ctx(files, RunConfig(ocr_backend="gemini", memory_mode="balanced"))

    await _stage(files, [0]).run(ctx)

    assert len(calls) == trims


# --- One-file prepare lookahead ---------------------------------------------
#
# With a managed vLLM OCR server the engine is another process: while file N
# waits on it, file N+1 renders, gets its layout and has its text layer read.
# Two files' pages are then resident instead of one. In-process engines and
# aggressive memory mode keep the strict one-file bound.


class _RecordingLayout(_CountingLayout):
    """Fake LayoutStage that also records call order, reports progress like
    the real one, and sets ``any_needs_ocr`` from a per-file table."""

    def __init__(self, all_files, peak, calls, needs_ocr=None):
        super().__init__(all_files, peak)
        self._calls = calls
        self._needs_ocr = needs_ocr or {}

    async def run(self, ctx):
        ctx.progress.stage_start(self.name)
        for fs in ctx.alive():
            self._calls.append(fs.path.name)
            ctx.signals.any_needs_ocr = self._needs_ocr.get(fs.path.name, True)
        await super().run(ctx)
        ctx.progress.stage_end(self.name)


class _YieldingOcr(_CountingOcr):
    """Fake OcrStage that waits like the real one does on the OCR server, then
    records what it saw: whether the following file was prepared meanwhile,
    and the ``any_needs_ocr`` signal."""

    def __init__(self, all_files, peak, seen, *, publish_identity=None, raise_for=()):
        super().__init__(all_files, peak)
        self._seen = seen
        self._publish_identity = publish_identity
        self._raise_for = set(raise_for)

    async def run(self, ctx):
        (fs,) = ctx.file_states
        ctx.progress.stage_start(self.name)
        for _ in range(5):
            await asyncio.sleep(0)
        if self._publish_identity is not None:
            # The real stage records the runtime the automatic chain started.
            ctx.scratch["ocr_runtime_identity"] = self._publish_identity
        idx = self._all.index(fs)
        following = self._all[idx + 1] if idx + 1 < len(self._all) else None
        self._seen.append(
            {
                "file": fs.path.name,
                "next_prepared": following is not None and following.page_images is not None,
                "any_needs_ocr": ctx.signals.any_needs_ocr,
            }
        )
        if fs.path.name in self._raise_for:
            raise RuntimeError(f"OCR crashed on {fs.path.name}")
        await super().run(ctx)
        ctx.progress.stage_end(self.name)


class _RecordingProgress(NullProgress):
    def __init__(self):
        self.events: list[tuple[str, str]] = []

    def stage_start(self, name, detail=""):
        self.events.append(("start", name))

    def stage_end(self, name):
        self.events.append(("end", name))


def _lookahead_stage(files, peak, seen, calls, **ocr_kwargs) -> InterleavedRenderOcrStage:
    return InterleavedRenderOcrStage(
        layout=_RecordingLayout(files, peak, calls),
        native_text=_NoopNative(),
        ocr=_YieldingOcr(files, peak, seen, **ocr_kwargs),
    )


_VLLM_OCR = {"ocr_backend": "paddle-vllm", "memory_mode": "balanced"}


@pytest.mark.asyncio
async def test_managed_vllm_ocr_prepares_the_next_file_while_ocr_runs():
    files = _mk_files(4)
    seen: list[dict] = []
    ctx = _ctx(files, RunConfig(**_VLLM_OCR))

    await _lookahead_stage(files, [0], seen, []).run(ctx)

    # While each file waited on the OCR server, the following one was rendered.
    assert [s["next_prepared"] for s in seen] == [True, True, True, False]


@pytest.mark.asyncio
async def test_lookahead_holds_at_most_two_files_pages_and_frees_them_all():
    files = _mk_files(6)
    peak = [0]
    ctx = _ctx(files, RunConfig(**_VLLM_OCR))

    await _lookahead_stage(files, peak, [], []).run(ctx)

    assert peak[0] == 2
    assert all(fs.ocr_regions is not None for fs in files)
    assert all(fs.page_images is None for fs in files)
    assert all(fs.pdf_bytes is None for fs in files)


@pytest.mark.asyncio
async def test_lookahead_keeps_file_order_and_prepares_each_file_once():
    files = _mk_files(5)
    seen: list[dict] = []
    calls: list[str] = []
    ctx = _ctx(files, RunConfig(**_VLLM_OCR))

    await _lookahead_stage(files, [0], seen, calls).run(ctx)

    names = [fs.path.name for fs in files]
    assert calls == names
    assert [s["file"] for s in seen] == names


@pytest.mark.parametrize(
    "config",
    [
        # An in-process engine shares this process's memory and CPU.
        {"ocr_backend": "glm-mlx", "memory_mode": "balanced"},
        # Aggressive mode is the explicit request for the smallest footprint.
        {"ocr_backend": "paddle-vllm", "memory_mode": "aggressive"},
    ],
)
@pytest.mark.asyncio
async def test_no_lookahead_for_in_process_engines_or_aggressive_memory(config):
    files = _mk_files(4)
    peak = [0]
    seen: list[dict] = []
    ctx = _ctx(files, RunConfig(**config))

    await _lookahead_stage(files, peak, seen, []).run(ctx)

    assert peak[0] == 1
    assert not any(s["next_prepared"] for s in seen)


@pytest.mark.asyncio
async def test_automatic_chain_starts_the_lookahead_once_the_runtime_is_known():
    from bibr.ocr.profiles import resolve_ocr_runtime_identity

    files = _mk_files(4)
    seen: list[dict] = []
    # ``paddle`` is the automatic chain: which engine runs is known only after
    # the first file's OCR started it, so the first file runs without lookahead.
    ctx = _ctx(files, RunConfig(ocr_backend="paddle", memory_mode="balanced"))
    started = resolve_ocr_runtime_identity(RunConfig(ocr_backend="paddle-vllm"), Settings)

    await _lookahead_stage(files, [0], seen, [], publish_identity=started).run(ctx)

    assert [s["next_prepared"] for s in seen] == [False, True, True, False]


@pytest.mark.asyncio
async def test_lookahead_layout_does_not_change_the_signal_of_the_file_in_ocr():
    files = _mk_files(3)
    peak = [0]
    seen: list[dict] = []
    needs = {"0.pdf": True, "1.pdf": False, "2.pdf": True}
    stage = InterleavedRenderOcrStage(
        layout=_RecordingLayout(files, peak, [], needs_ocr=needs),
        native_text=_NoopNative(),
        ocr=_YieldingOcr(files, peak, seen),
    )
    ctx = _ctx(files, RunConfig(**_VLLM_OCR))

    await stage.run(ctx)

    # Each file's OCR reads the signal its own layout pass set, although the
    # following file's layout ran in the meantime.
    assert {s["file"]: s["any_needs_ocr"] for s in seen} == needs
    assert [s["next_prepared"] for s in seen] == [True, True, False]


@pytest.mark.asyncio
async def test_lookahead_reports_no_progress_under_the_running_ocr():
    files = _mk_files(3)
    progress = _RecordingProgress()
    seen: list[dict] = []
    ctx = _ctx(files, RunConfig(**_VLLM_OCR))
    ctx.progress = progress

    await _lookahead_stage(files, [0], seen, []).run(ctx)

    assert [s["next_prepared"] for s in seen] == [True, True, False]
    # The display follows one file at a time: no layout line opens between an
    # OCR start and its end.
    open_stage = None
    for kind, name in progress.events:
        if kind == "start":
            assert open_stage is None, f"{name} started while {open_stage} was open"
            open_stage = name
        else:
            assert open_stage == name
            open_stage = None


@pytest.mark.asyncio
async def test_ocr_crash_cancels_the_lookahead_and_frees_its_pages():
    files = _mk_files(3)
    peak = [0]
    cancelled: list[str] = []

    class _HangingLayout(_CountingLayout):
        async def run(self, ctx):
            (fs,) = ctx.file_states
            await super().run(ctx)
            if fs.path.name == "2.pdf":
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.append(fs.path.name)
                    raise

    stage = InterleavedRenderOcrStage(
        layout=_HangingLayout(files, peak),
        native_text=_NoopNative(),
        ocr=_YieldingOcr(files, peak, [], raise_for={"1.pdf"}),
    )
    ctx = _ctx(files, RunConfig(**_VLLM_OCR))

    with pytest.raises(RuntimeError, match=r"OCR crashed on 1\.pdf"):
        await asyncio.wait_for(stage.run(ctx), timeout=5)

    # The prepare of the following file was running when OCR crashed: it is
    # cancelled, its pages are released and nothing outlives the stage.
    assert cancelled == ["2.pdf"]
    assert files[2].page_images is None
    leftover = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leftover == []
