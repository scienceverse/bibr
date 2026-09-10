"""The serve LayoutDetector routes pages through a GpuBatcher.

These tests exercise the wiring (per-page submission, order-preserving
results, batcher construction) without loading the PyTorch model — the
coalescing/serialization guarantees themselves are covered by
test_gpu_batcher.py.
"""

import asyncio

from PIL import Image

from bibr.config import GlobalSettings
from bibr.serve.deployments.layout import _MAX_BATCH_SIZE, LayoutDetector


def _page(width: int) -> Image.Image:
    """A PIL page image tagged by its width (used as an identity marker)."""
    return Image.new("RGB", (width, 100))


class _FakeBatcher:
    """Records submitted items; echoes one region keyed by the page width."""

    def __init__(self):
        self.items: list = []

    async def submit(self, item):
        self.items.append(item)
        return [{"echo": item.width}]

    async def close(self):
        pass


def _detector_with(batcher) -> LayoutDetector:
    det = object.__new__(LayoutDetector)
    det._settings = GlobalSettings()
    det._batcher = batcher
    return det


async def test_detect_batch_empty_short_circuits():
    det = _detector_with(_FakeBatcher())
    assert await det.detect_batch([]) == []


async def test_detect_batch_submits_each_page_and_preserves_order():
    fake = _FakeBatcher()
    det = _detector_with(fake)
    pages = [_page(w) for w in (5, 6, 7)]
    out = await det.detect_batch(pages)
    assert fake.items == pages  # every page submitted individually, in order
    assert out == [[{"echo": 5}], [{"echo": 6}], [{"echo": 7}]]


async def test_build_batcher_wires_detect_fn_and_layout_batch_size():
    det = object.__new__(LayoutDetector)
    det._settings = GlobalSettings()
    det._gpu_executor = None
    det._detect_images = lambda batch: [["region"] for _ in batch]  # type: ignore[method-assign]
    batcher = det._build_batcher()
    try:
        assert batcher._fn is det._detect_images
        assert batcher._max_batch_size == _MAX_BATCH_SIZE
    finally:
        await batcher.close()


async def test_detect_batch_coalesces_concurrent_requests_through_real_batcher():
    """Two concurrent requests' pages may share a single fn batch."""
    batches: list[list] = []

    def fn(batch):
        batches.append(list(batch))
        return [[{"echo": img.width}] for img in batch]

    det = object.__new__(LayoutDetector)
    det._settings = GlobalSettings()
    det._gpu_executor = None
    det._detect_images = fn  # type: ignore[method-assign]
    det._batcher = det._build_batcher()
    try:
        req_a = [_page(i) for i in range(3)]
        req_b = [_page(10 + i) for i in range(3)]
        res_a, res_b = await asyncio.gather(det.detect_batch(req_a), det.detect_batch(req_b))
        assert [r[0]["echo"] for r in res_a] == [0, 1, 2]
        assert [r[0]["echo"] for r in res_b] == [10, 11, 12]
        assert all(len(b) <= _MAX_BATCH_SIZE for b in batches)
        assert sum(len(b) for b in batches) == 6
    finally:
        await det._batcher.close()
