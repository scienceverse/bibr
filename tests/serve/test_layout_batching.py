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


async def test_build_batcher_uses_effective_batch_for_cpu_and_cuda(monkeypatch):
    """The serve micro-batcher runs one page at a time on CPU (no arena
    growth) and keeps the configured batch on CUDA; explicit wins."""
    from types import SimpleNamespace

    from bibr.layout_utils import effective_layout_batch_size

    async def _max_batch(device_type, env_batch=None):
        if env_batch is not None:
            monkeypatch.setenv("LAYOUT_BATCH_SIZE", env_batch)
        else:
            monkeypatch.delenv("LAYOUT_BATCH_SIZE", raising=False)
        det = object.__new__(LayoutDetector)
        det._settings = GlobalSettings()
        det._device = SimpleNamespace(type=device_type)
        det._gpu_executor = None
        det._detect_images = lambda batch: [[{}] for _ in batch]  # type: ignore[method-assign]
        batcher = det._build_batcher()
        try:
            assert batcher._max_batch_size == effective_layout_batch_size(
                det._settings, device_type
            )
            return batcher._max_batch_size
        finally:
            await batcher.close()

    assert await _max_batch("cpu") == 1
    assert await _max_batch("cuda") == 8
    assert await _max_batch("cpu", "4") == 4


def test_onnx_warmup_uses_effective_batch_on_cpu():
    """The ONNX warmup forwards the effective CPU batch (1), not raw 8."""
    from types import SimpleNamespace

    det = object.__new__(LayoutDetector)
    det._settings = GlobalSettings()
    det._device = SimpleNamespace(type="cpu")
    det._runtime = "onnx"
    seen = {}

    class _FakeModel:
        def run(self, batch):
            seen["n"] = len(batch)
            return [[{}] for _ in batch]

    det._model = _FakeModel()
    det._run_warmup()

    assert seen["n"] == 1


def test_torch_warmup_uses_effective_batch_on_cpu(monkeypatch):
    """The torch warmup also forwards the effective CPU batch."""
    import sys
    import types
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    det = object.__new__(LayoutDetector)
    det._settings = GlobalSettings()
    det._device = SimpleNamespace(type="cpu")
    det._runtime = "torch"
    det._model = MagicMock()
    seen = {}

    class _FakeProcessor:
        def __call__(self, images, return_tensors=None):
            seen["n"] = len(images)
            mock = MagicMock()
            mock.to.return_value = mock
            return {"x": mock}

    det._image_processor = _FakeProcessor()
    fake_torch = types.ModuleType("torch")
    fake_torch.inference_mode = lambda: __import__("contextlib").nullcontext()  # type: ignore[attr-defined]
    fake_torch.cuda = SimpleNamespace(empty_cache=lambda: None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    det._run_warmup()

    assert seen["n"] == 1
