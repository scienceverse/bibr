"""Bound pixel ownership while preserving whole-document OCR semantics."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from bibr.config import GlobalSettings
from bibr.exceptions import UpstreamServiceError
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.render_ocr import InterleavedRenderOcrStage
from bibr.pipeline.state import FileState


def _region(text="native page text"):
    return {
        "index": 0,
        "label": "text",
        "native_label": "text",
        "task_type": "text",
        "bbox_2d": [0, 0, 1000, 1000],
        "content": text,
        "_native_text_used": True,
    }


class _Native:
    async def run(self, ctx):
        pass


def _context(monkeypatch, *, pages=11, window=3, files=1, config=None, cache_dir=None):
    images = []
    peak = [0]

    def render(_pdf, _dpi, start, end, _max_pixels, _max_dimension):
        for i in range(start, end + 1):
            img = Image.new("RGB", (32, 32), color=(i, 0, 0))
            images.append(img)
            live = 0
            for item in images:
                try:
                    item.getpixel((0, 0))
                    live += 1
                except ValueError:
                    pass
            peak[0] = max(peak[0], live)
            yield i, img

    async def detect(items):
        return [[_region(f"page {img.getpixel((0, 0))[0]}")] for img in items]

    monkeypatch.setattr("bibr.ocr.utils.get_pdf_page_count", lambda _: pages)
    monkeypatch.setattr("bibr.pipeline.stages.layout._iter_pdf_pages", render)
    settings = GlobalSettings(
        pipeline={"page_window_size": window},
        cache={"ocr": cache_dir is not None, "ocr_dir": str(cache_dir) if cache_dir else None},
    )
    resources = SimpleNamespace(
        layout=SimpleNamespace(detect_batch=AsyncMock(side_effect=detect)),
        ocr=SimpleNamespace(loaded=True, recognize=AsyncMock(), wait_for_server=AsyncMock()),
        ocr_runtime_identity=None,
        ensure_layout=lambda: None,
        unload_layout=lambda: None,
        await_ocr=AsyncMock(),
        shutdown_ocr=AsyncMock(),
    )
    states = [
        FileState(path=Path(f"{i}.pdf"), pdf_bytes=b"fake", file_hash=f"hash-{i}")
        for i in range(files)
    ]
    ctx = PipelineContext(
        file_states=states,
        resources=resources,
        config=config or RunConfig(ocr_backend="glm-http"),
        progress=NullProgress(),
        settings=settings,
    )
    return ctx, images, peak


@pytest.mark.parametrize("window", [1, 3, 8])
async def test_long_documents_bound_pixels_and_keep_every_page(monkeypatch, window):
    ctx, images, peak = _context(monkeypatch, pages=19, window=window, files=2)
    await InterleavedRenderOcrStage(native_text=_Native()).run(ctx)

    assert peak[0] <= 2 * window
    for fs in ctx.file_states:
        assert fs.error is None
        assert fs.page_indices == list(range(19))
        assert [page[0].content for page in fs.ocr_regions] == [f"page {i}" for i in range(19)]
        assert fs.ocr_pages_attempted == 19
        assert fs.page_images is None
        assert fs.pdf_bytes is None
        assert fs.pdf_inspection_accumulator is None
    for img in images:
        with pytest.raises(ValueError, match="closed"):
            img.getpixel((0, 0))


async def test_selected_page_range_and_document_cap(monkeypatch):
    ctx, _, peak = _context(
        monkeypatch, config=RunConfig(ocr_backend="glm-http", start_page=2, end_page=9), window=2
    )
    ctx.settings.pipeline.max_pages = 5
    await InterleavedRenderOcrStage(native_text=_Native()).run(ctx)
    fs = ctx.file_states[0]
    assert fs.page_indices == [2, 3, 4, 5, 6]
    assert fs.ocr_regions[:2] == [[], []]
    assert [page[0].content for page in fs.ocr_regions[2:]] == [f"page {i}" for i in range(2, 7)]
    assert peak[0] <= 2


@pytest.mark.parametrize(
    "bad_pages,threshold,expected",
    [
        ({0}, 0.5, None),
        ({0, 1, 2}, 0.5, "ocr_mostly_failed"),
        ({0, 1, 2, 3}, 0.0, "ocr_failed"),
    ],
)
async def test_failure_gate_uses_whole_document(monkeypatch, bad_pages, threshold, expected):
    from bibr.pipeline.stages import ocr

    ctx, _, _ = _context(monkeypatch, pages=4, window=1)
    ctx.settings.ocr.min_success_rate = threshold
    real = ocr.ocr_page_regions

    async def fail_page(image, regions, index, *args, **kwargs):
        if index in bad_pages:
            raise ValueError("page decode failed")
        return await real(image, regions, index, *args, **kwargs)

    monkeypatch.setattr(ocr, "ocr_page_regions", fail_page)
    await InterleavedRenderOcrStage(native_text=_Native()).run(ctx)
    fs = ctx.file_states[0]
    assert fs.error_code == expected
    assert fs.ocr_pages_attempted == 4
    assert fs.ocr_pages_failed == len(bad_pages)
    if expected is None:
        assert fs.ocr_regions[0] == []
        assert fs.ocr_regions[3][0].content == "page 3"


async def test_upstream_failure_stops_later_windows_and_never_caches_partial(monkeypatch, tmp_path):
    ctx, images, _ = _context(monkeypatch, pages=9, window=2, cache_dir=tmp_path)

    async def outage(*args, **kwargs):
        raise UpstreamServiceError("ocr", "OCR unavailable")

    monkeypatch.setattr("bibr.pipeline.stages.ocr.ocr_page_regions", outage)
    await InterleavedRenderOcrStage(native_text=_Native()).run(ctx)
    assert ctx.file_states[0].error_code == "ocr_failed"
    assert isinstance(ctx.file_states[0].original_error, UpstreamServiceError)
    assert len(images) == 2
    assert list(tmp_path.glob("*.json")) == []


async def test_cache_stores_complete_document_and_replay_does_not_render(monkeypatch, tmp_path):
    from bibr.ocr.profiles import resolve_ocr_runtime_identity
    from bibr.pipeline import ocr_cache

    ctx, images, _ = _context(monkeypatch, pages=9, window=2, cache_dir=tmp_path)
    stage = InterleavedRenderOcrStage(native_text=_Native())
    await stage.run(ctx)
    original = ctx.file_states[0]
    incoming = FileState(path=original.path, file_hash=original.file_hash, pdf_bytes=b"fake")
    identity = resolve_ocr_runtime_identity(ctx.config, ctx.settings)
    assert ocr_cache.load_bundle(incoming, ctx.config, identity, ctx.settings)
    assert incoming.ocr_regions == original.ocr_regions
    incoming.ocr_regions = None
    ctx.file_states = [incoming]
    await stage.run(ctx)
    assert len(images) == 9
    assert incoming.ocr_regions == original.ocr_regions


@pytest.mark.parametrize("cancel_twice", [False, True])
async def test_cancellation_waits_for_image_consumer_then_closes_pixels(
    monkeypatch, tmp_path, cancel_twice
):
    ctx, images, _ = _context(monkeypatch, pages=9, window=2, cache_dir=tmp_path)
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def detect(items):
        entered.set()
        await finish.wait()
        # The page owner must not close these while inference still uses them.
        return [[_region(str(item.getpixel((0, 0))))] for item in items]

    ctx.resources.layout.detect_batch = detect
    task = asyncio.create_task(InterleavedRenderOcrStage(native_text=_Native()).run(ctx))
    await entered.wait()
    task.cancel()
    if cancel_twice:
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(images) == 2
    assert list(tmp_path.glob("*.json")) == []
    assert ctx.file_states[0].page_images is None
    for img in images:
        with pytest.raises(ValueError, match="closed"):
            img.getpixel((0, 0))


async def test_real_pdf_matches_previous_single_pass(monkeypatch):
    from dataclasses import replace

    from bibr.ocr.image_utils import iter_pdf_pages_with_index
    from bibr.ocr.utils import get_pdf_page_count
    from bibr.pipeline.context import StageSignals
    from bibr.pipeline.stages.layout import LayoutStage
    from bibr.pipeline.stages.native_text import NativeTextStage
    from bibr.pipeline.stages.ocr import OcrStage

    ctx, _, _ = _context(monkeypatch, window=1)
    monkeypatch.setattr("bibr.ocr.utils.get_pdf_page_count", get_pdf_page_count)
    monkeypatch.setattr("bibr.pipeline.stages.layout._iter_pdf_pages", iter_pdf_pages_with_index)
    pdf = Path("bibr/data/sample_paper.pdf").read_bytes()
    incoming = ctx.file_states[0]
    incoming.pdf_bytes = pdf
    ctx.settings.ocr.native_text_min_chars = 1
    ctx.settings.pipeline.outline_headings = True

    async def layout(items):
        return [
            [
                {
                    "index": 0,
                    "label": "text",
                    "native_label": "text",
                    "task_type": "text",
                    "bbox_2d": [0, 0, 1000, 1000],
                    "content": "",
                }
            ]
            for _ in items
        ]

    ctx.resources.layout.detect_batch = layout
    baseline = FileState(path=incoming.path, pdf_bytes=pdf)
    baseline_ctx = replace(ctx, file_states=[baseline], signals=StageSignals(), scratch={})
    try:
        await LayoutStage().run(baseline_ctx)
        await NativeTextStage().run(baseline_ctx)
        await OcrStage().run(baseline_ctx)
        await InterleavedRenderOcrStage().run(ctx)
        assert incoming.error is None
        assert incoming.ocr_regions == baseline.ocr_regions
        assert incoming.page_indices == baseline.page_indices
        assert incoming.native_metadata == baseline.native_metadata
        assert incoming.ref_line_geometry == baseline.ref_line_geometry
        assert incoming.pdf_outline == baseline.pdf_outline
    finally:
        baseline.free_pre_ocr()


async def test_page_windows_keep_cross_request_gpu_batching(monkeypatch):
    from bibr.serve.batching import GpuBatcher
    from bibr.serve.deployments.layout import LayoutDetector

    a, _, _ = _context(monkeypatch, pages=6, window=2)
    b, _, _ = _context(monkeypatch, pages=6, window=2)
    batches = []

    def forward(items):
        batches.append(len(items))
        return [[_region(f"page {img.getpixel((0, 0))[0]}")] for img in items]

    detector = object.__new__(LayoutDetector)
    detector._batcher = GpuBatcher(forward, max_batch_size=4, batch_timeout=0.01)
    barrier = asyncio.Barrier(2)

    async def detect(items):
        await barrier.wait()
        return await detector.detect_batch(items)

    a.resources.layout.detect_batch = detect
    b.resources.layout.detect_batch = detect
    try:
        await asyncio.gather(
            InterleavedRenderOcrStage(native_text=_Native()).run(a),
            InterleavedRenderOcrStage(native_text=_Native()).run(b),
        )
        assert batches == [4, 4, 4]
        for ctx in (a, b):
            assert ctx.file_states[0].error is None
            assert len(ctx.file_states[0].ocr_regions) == 6
    finally:
        await detector._batcher.close()


@pytest.mark.parametrize("change_model", [False, True])
async def test_aggressive_mode_refreshes_runtime_identity(monkeypatch, tmp_path, change_model):
    from bibr.ocr.profiles import OcrRuntimeIdentity

    ctx, _, _ = _context(
        monkeypatch,
        pages=4,
        window=2,
        cache_dir=tmp_path,
        config=RunConfig(ocr_backend="paddle", memory_mode="aggressive"),
    )
    rm = ctx.resources
    client = rm.ocr
    client.loaded = False
    calls = []

    async def start():
        model = "model-b" if change_model and len(calls) >= 2 else "model-a"
        calls.append(model)
        client.loaded = True
        rm.ocr_runtime_identity = OcrRuntimeIdentity(
            backend="glm-http", model=model, profile="glm", normalizer_version="test"
        )

    async def stop():
        client.loaded = False
        rm.ocr_runtime_identity = None

    real_detect = rm.layout.detect_batch

    async def detect(items):
        assert not client.loaded, "aggressive layout must not coexist with OCR"
        return await real_detect(items)

    rm.await_ocr = start
    rm.shutdown_ocr = stop
    rm.layout.detect_batch = detect
    await InterleavedRenderOcrStage(native_text=_Native()).run(ctx)
    fs = ctx.file_states[0]
    assert len(calls) == 3  # exact cache identity probe, then two page windows
    if change_model:
        assert fs.error_code == "ocr_failed"
        assert isinstance(fs.original_error, UpstreamServiceError)
        assert fs.ocr_regions is None
        assert list(tmp_path.glob("*.json")) == []
    else:
        assert fs.error is None
        assert len(fs.ocr_regions) == 4
        assert ctx.scratch["ocr_runtime_identity"].model == "model-a"
        assert len(list(tmp_path.glob("*.json"))) == 1
