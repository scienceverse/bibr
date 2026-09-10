"""Tests for the GpuBatcher async micro-batching broker.

The broker coalesces items submitted by concurrent requests into fixed-size
batches handed to a single sync ``fn`` (the GPU forward pass), serializing
GPU access so peak VRAM is bounded to one in-flight batch regardless of how
many requests are in flight.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from bibr.serve.batching import GpuBatcher


async def test_submit_returns_fn_result():
    batcher = GpuBatcher(lambda items: [x * 2 for x in items], max_batch_size=4, batch_timeout=0.05)
    try:
        assert await batcher.submit(21) == 42
    finally:
        await batcher.close()


async def test_concurrent_submits_coalesce_into_one_call():
    calls: list[list[int]] = []

    def fn(items):
        calls.append(list(items))
        return [x + 1 for x in items]

    batcher = GpuBatcher(fn, max_batch_size=8, batch_timeout=0.1)
    try:
        results = await asyncio.gather(*(batcher.submit(i) for i in range(5)))
        assert results == [1, 2, 3, 4, 5]
        assert len(calls) == 1  # all five coalesced into a single forward pass
        assert sorted(calls[0]) == [0, 1, 2, 3, 4]
    finally:
        await batcher.close()


async def test_batch_respects_max_size():
    calls: list[list[int]] = []

    def fn(items):
        calls.append(list(items))
        return list(items)

    batcher = GpuBatcher(fn, max_batch_size=2, batch_timeout=0.1)
    try:
        results = await asyncio.gather(*(batcher.submit(i) for i in range(5)))
        assert sorted(results) == [0, 1, 2, 3, 4]
        assert all(len(c) <= 2 for c in calls)
        assert sum(len(c) for c in calls) == 5
    finally:
        await batcher.close()


async def test_only_one_fn_runs_at_a_time_even_with_multithread_executor():
    """The collector must serialize fn calls so two GPU forwards never overlap,
    even when handed an executor that *could* run them in parallel."""
    concurrent = 0
    max_concurrent = 0
    lock = threading.Lock()

    def fn(items):
        nonlocal concurrent, max_concurrent
        with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        time.sleep(0.02)
        with lock:
            concurrent -= 1
        return list(items)

    executor = ThreadPoolExecutor(max_workers=4)
    batcher = GpuBatcher(fn, max_batch_size=1, batch_timeout=0.0, executor=executor)
    try:
        await asyncio.gather(*(batcher.submit(i) for i in range(6)))
        assert max_concurrent == 1
    finally:
        await batcher.close()
        executor.shutdown()


async def test_error_propagates_to_each_caller_in_batch():
    def fn(items):
        raise ValueError("boom")

    batcher = GpuBatcher(fn, max_batch_size=4, batch_timeout=0.05)
    try:
        with pytest.raises(ValueError, match="boom"):
            await asyncio.gather(batcher.submit(1), batcher.submit(2))
    finally:
        await batcher.close()


async def test_results_map_back_to_correct_caller():
    """Each caller gets the result for *its* item, not a positional shuffle."""

    def fn(items):
        return [x * 10 for x in items]

    batcher = GpuBatcher(fn, max_batch_size=8, batch_timeout=0.05)
    try:
        results = await asyncio.gather(*(batcher.submit(i) for i in range(8)))
        assert results == [i * 10 for i in range(8)]
    finally:
        await batcher.close()


async def test_cancelled_submission_is_not_sent_to_model():
    calls: list[list[int]] = []

    def fn(items):
        calls.append(list(items))
        return list(items)

    batcher = GpuBatcher(fn, max_batch_size=8, batch_timeout=0.05)
    try:
        task = asyncio.create_task(batcher.submit(7))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.08)
        assert calls == []
    finally:
        await batcher.close()
