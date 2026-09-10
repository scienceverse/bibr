"""Tests for AsyncLocalRateLimiter."""

import asyncio
import time

from bibr.utils.rate_limiter import AsyncLocalRateLimiter


class TestLocalRateLimiter:
    async def test_first_acquire_is_immediate(self):
        limiter = AsyncLocalRateLimiter(resource_id="test", max_requests=1, window_seconds=1.0)
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    async def test_second_acquire_waits(self):
        limiter = AsyncLocalRateLimiter(resource_id="test", max_requests=1, window_seconds=0.3)
        await limiter.acquire()
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.2

    async def test_multiple_requests_within_window(self):
        limiter = AsyncLocalRateLimiter(resource_id="test", max_requests=3, window_seconds=1.0)
        for _ in range(3):
            await limiter.acquire()
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed > 0.0

    async def test_close_is_noop(self):
        limiter = AsyncLocalRateLimiter(resource_id="test")
        await limiter.close()


class TestLocalRateLimiterLoopRebinding:
    """M6: the limiter must survive a second ``asyncio.run()`` in one process.

    ``asyncio.Condition`` binds to the loop on first use; reusing the
    singleton-held limiter from a later loop raised ``RuntimeError: ...
    bound to a different event loop``, silently stopping enrichment.
    """

    def test_contended_acquire_works_across_event_loops(self):
        """The uncontended fast path never binds; contention (cond.wait) does."""
        limiter = AsyncLocalRateLimiter(resource_id="test", max_requests=1, window_seconds=0.05)

        async def two_acquires():
            await limiter.acquire()
            await limiter.acquire()  # window full -> waits -> binds the loop

        asyncio.run(two_acquires())
        asyncio.run(two_acquires())  # must not raise

    def test_rate_history_survives_loop_change(self):
        """Rebinding the condition must not discard the sliding window."""
        limiter = AsyncLocalRateLimiter(resource_id="test", max_requests=2, window_seconds=30.0)
        asyncio.run(limiter.acquire())
        asyncio.run(limiter.acquire())
        assert len(limiter._timestamps) == 2


class TestLocalRateLimiterCoordinator:
    async def test_serial_waiters_are_pacing(self):
        """N waiters at 1 req per W must serialize cleanly without burst."""
        limiter = AsyncLocalRateLimiter(resource_id="t", max_requests=1, window_seconds=0.1)

        enter_times: list[float] = []

        async def one():
            await limiter.acquire()
            enter_times.append(time.monotonic())

        await asyncio.gather(*(one() for _ in range(5)))

        deltas = [enter_times[i + 1] - enter_times[i] for i in range(len(enter_times) - 1)]
        # Each acquire should be roughly one window (0.1s) apart.
        assert all(0.08 <= d <= 0.25 for d in deltas), deltas

    async def test_max_requests_gt_one_can_burst(self):
        """When max_requests=3, three waiters acquire immediately."""
        limiter = AsyncLocalRateLimiter(resource_id="b", max_requests=3, window_seconds=1.0)
        start = time.monotonic()
        await asyncio.gather(*(limiter.acquire() for _ in range(3)))
        elapsed = time.monotonic() - start
        assert elapsed < 0.05, elapsed
