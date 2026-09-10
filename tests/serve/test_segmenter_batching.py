"""The serve SentenceSegmenter routes texts through a GpuBatcher.

Wiring tests only (per-text submission, order, lazy batcher construction);
coalescing/serialization guarantees are covered by test_gpu_batcher.py and
the cross-variant test_segmenter_concurrency.py.
"""

import asyncio

from bibr.config import GlobalSettings
from bibr.serve.deployments.segmenter import SentenceSegmenter


class _FakeBatcher:
    def __init__(self):
        self.items: list = []

    async def submit(self, item):
        self.items.append(item)
        return [item.upper()]

    async def close(self):
        pass


async def test_segment_batch_empty_short_circuits():
    seg = object.__new__(SentenceSegmenter)
    seg._settings = GlobalSettings()
    seg._batcher = _FakeBatcher()
    assert await seg.segment_batch([]) == []


async def test_segment_batch_submits_each_text_and_preserves_order():
    seg = object.__new__(SentenceSegmenter)
    seg._settings = GlobalSettings()
    fake = _FakeBatcher()
    seg._batcher = fake
    out = await seg.segment_batch(["a", "b", "c"])
    assert fake.items == ["a", "b", "c"]
    assert out == [["A"], ["B"], ["C"]]


async def test_get_batcher_wires_split_many_and_sub_batch_size():
    seg = object.__new__(SentenceSegmenter)
    seg._settings = GlobalSettings()
    seg._split_many = lambda texts: [[t] for t in texts]  # type: ignore[method-assign]
    batcher = seg._get_batcher()
    try:
        assert batcher._fn is seg._split_many
        assert batcher._max_batch_size == seg._settings.SEGMENTER_SUB_BATCH_SIZE
        # second call returns the same instance (lazy singleton)
        assert seg._get_batcher() is batcher
    finally:
        await batcher.close()


async def test_segment_batch_coalesces_concurrent_requests():
    seen: list[list[str]] = []

    def split_many(texts):
        seen.append(list(texts))
        return [[t] for t in texts]

    seg = object.__new__(SentenceSegmenter)
    seg._settings = GlobalSettings()
    seg._split_many = split_many  # type: ignore[method-assign]
    try:
        res = await asyncio.gather(
            seg.segment_batch(["a"]),
            seg.segment_batch(["b"]),
            seg.segment_batch(["c"]),
        )
        assert res == [[["a"]], [["b"]], [["c"]]]
        # all three texts coalesced into a single wtpsplit call
        assert seen == [["a", "b", "c"]]
    finally:
        await seg._batcher.close()
