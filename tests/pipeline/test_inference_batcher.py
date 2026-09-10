import asyncio
import threading

import pytest


def test_restarts_after_previous_event_loop_closes():
    from bibr.pipeline.inference_batcher import InferenceBatcher

    batcher = InferenceBatcher(lambda items: list(items), 4, 5, "restart")

    async def submit(item):
        return await asyncio.wait_for(batcher.submit(item), timeout=0.2)

    assert asyncio.run(submit("first")) == "first"
    assert asyncio.run(submit("second")) == "second"


def test_start_rejects_live_collector_from_different_loop():
    from bibr.pipeline.inference_batcher import InferenceBatcher

    batcher = InferenceBatcher(lambda items: list(items), 4, 5, "cross-loop")
    owner_loop = asyncio.new_event_loop()
    owner_thread = threading.Thread(target=owner_loop.run_forever)
    owner_thread.start()
    try:
        asyncio.run_coroutine_threadsafe(batcher.start(), owner_loop).result(timeout=1)
        with pytest.raises(RuntimeError, match="different event loop"):
            asyncio.run(batcher.start())
    finally:
        asyncio.run_coroutine_threadsafe(batcher.close(), owner_loop).result(timeout=1)
        owner_loop.call_soon_threadsafe(owner_loop.stop)
        owner_thread.join(timeout=1)
        owner_loop.close()


def test_restart_cancels_stranded_future_bound_to_closed_loop():
    from bibr.pipeline.inference_batcher import InferenceBatcher

    batcher = InferenceBatcher(lambda items: list(items), 4, 5, "closed-loop-pending")
    asyncio.run(batcher.start())

    closed_loop = asyncio.new_event_loop()
    stranded = closed_loop.create_future()
    stranded.add_done_callback(lambda _: None)
    closed_loop.close()
    batcher._pending.add(stranded)

    async def restart_and_close():
        await batcher.start()
        await batcher.close()

    asyncio.run(restart_and_close())
    assert stranded.cancelled()


async def test_restart_cancels_submission_stranded_by_dead_collector():
    from bibr.pipeline.inference_batcher import InferenceBatcher

    entered = threading.Event()
    release = threading.Event()

    def forward(items):
        entered.set()
        release.wait(timeout=2)
        return list(items)

    batcher = InferenceBatcher(forward, 1, 0, "stranded")
    submission = asyncio.create_task(batcher.submit("pending"))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        collector = batcher._collector
        assert collector is not None
        collector.cancel()
        with pytest.raises(asyncio.CancelledError):
            await collector

        await batcher.start()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(submission, timeout=0.1)
    finally:
        release.set()
        if not submission.done():
            submission.cancel()
        await batcher.close()


async def test_concurrent_submitters_share_one_forward_and_preserve_order():
    from bibr.pipeline.inference_batcher import InferenceBatcher

    calls = []

    def forward(items):
        calls.append(items)
        return [item * 2 for item in items]

    batcher = InferenceBatcher(forward, max_batch_size=8, timeout_ms=10, name="numbers")
    try:
        assert await asyncio.gather(*(batcher.submit(i) for i in range(4))) == [0, 2, 4, 6]
        assert calls == [[0, 1, 2, 3]]
    finally:
        await batcher.close()


async def test_max_batch_size_splits_queued_items():
    from bibr.pipeline.inference_batcher import InferenceBatcher

    calls = []

    def forward(items):
        calls.append(items)
        return list(items)

    batcher = InferenceBatcher(forward, max_batch_size=2, timeout_ms=5, name="split")
    try:
        assert await asyncio.gather(*(batcher.submit(i) for i in range(5))) == list(range(5))
        assert calls == [[0, 1], [2, 3], [4]]
    finally:
        await batcher.close()


async def test_cancelled_queued_item_is_not_forwarded():
    from bibr.pipeline.inference_batcher import InferenceBatcher

    entered = threading.Event()
    release = threading.Event()
    calls = []

    def forward(items):
        calls.append(items)
        entered.set()
        release.wait(timeout=2)
        return list(items)

    batcher = InferenceBatcher(forward, max_batch_size=1, timeout_ms=1, name="cancel")
    try:
        first = asyncio.create_task(batcher.submit("first"))
        assert await asyncio.to_thread(entered.wait, 1)
        second = asyncio.create_task(batcher.submit("cancelled"))
        await asyncio.sleep(0)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        release.set()
        assert await first == "first"
        await asyncio.sleep(0.02)
        assert calls == [["first"]]
    finally:
        release.set()
        await batcher.close()


async def test_forward_exception_reaches_every_caller_and_next_batch_still_runs():
    from bibr.pipeline.inference_batcher import InferenceBatcher

    should_fail = True

    def forward(items):
        nonlocal should_fail
        if should_fail:
            should_fail = False
            raise RuntimeError("model failed")
        return list(items)

    batcher = InferenceBatcher(forward, max_batch_size=4, timeout_ms=5, name="errors")
    try:
        results = await asyncio.gather(
            batcher.submit("a"), batcher.submit("b"), return_exceptions=True
        )
        assert len(results) == 2
        assert all(isinstance(result, RuntimeError) for result in results)
        assert await batcher.submit("recovered") == "recovered"
    finally:
        await batcher.close()


async def test_result_count_mismatch_fails_all_callers():
    from bibr.pipeline.inference_batcher import InferenceBatcher

    batcher = InferenceBatcher(lambda items: [], max_batch_size=4, timeout_ms=5, name="bad")
    try:
        results = await asyncio.gather(
            batcher.submit("a"), batcher.submit("b"), return_exceptions=True
        )
        assert all(isinstance(result, RuntimeError) for result in results)
        assert all("returned 0 results for 2 items" in str(result) for result in results)
    finally:
        await batcher.close()


async def test_close_rejects_new_work_and_is_idempotent():
    from bibr.pipeline.inference_batcher import InferenceBatcher

    batcher = InferenceBatcher(lambda items: list(items), 4, 5, "closed")
    await batcher.start()
    await batcher.close()
    await batcher.close()
    with pytest.raises(RuntimeError, match="closed"):
        await batcher.submit("late")
