"""Resilience fixes in the circuit breaker and the rate limiters.

- A breaker shared by threads that each run their own event loop must not
  spin forever in HALF_OPEN once a loop switch drops the probe's event.
- A provider SDK's 4xx (openai ``BadRequestError``, google-genai
  ``ClientError``) is a rejected request, not an outage: it must not open
  the breaker every paper shares.
- The Redis limiter remembers an outage instead of re-paying the retries on
  every acquire, enforces the budget in-process meanwhile, and reads the
  Redis server's clock rather than each host's.
- The local limiter wakes one waiter per freed slot, not every waiter.
"""

import asyncio
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from bibr.utils import rate_limiter as rate_limiter_mod
from bibr.utils.circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitOpenError,
    CircuitState,
    _is_client_error,
)
from bibr.utils.rate_limiter import AsyncLocalRateLimiter, AsyncRedisRateLimiter


def _run_in_thread(coro_fn, timeout: float = 5.0):
    """Run ``coro_fn()`` under ``asyncio.run`` in a daemon thread.

    Returns ``(finished, outcome)``. A breaker that spins never yields, so not
    even ``wait_for`` could stop it on the test's own loop; a thread with a
    join timeout keeps the test runner alive if the regression comes back.
    """
    outcome: dict[str, object] = {}

    def target():
        try:
            outcome["result"] = asyncio.run(coro_fn())
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    return not thread.is_alive(), outcome


# --- Circuit breaker shared across threads' event loops -------------------


class _Clock:
    """Manually advanced monotonic clock: OPEN → HALF_OPEN without racing wall time."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _open_breaker(**kwargs) -> tuple[AsyncCircuitBreaker, _Clock]:
    """A breaker that is OPEN and past its reset timeout: the next caller probes."""
    clock = _Clock()
    cb = AsyncCircuitBreaker(failure_threshold=1, failure_dedup_window=0.0, clock=clock, **kwargs)
    cb._state = CircuitState.OPEN
    cb._last_failure_time = clock() - 60
    return cb, clock


def _start_thread(coro_fn) -> tuple[threading.Thread, dict[str, object]]:
    """Start ``asyncio.run(coro_fn())`` in a daemon thread; the outcome fills in later."""
    outcome: dict[str, object] = {}

    def target():
        try:
            outcome["result"] = asyncio.run(coro_fn())
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, outcome


class TestBreakerAcrossThreadLoops:
    def test_caller_on_other_loop_waits_for_the_probe(self):
        """Loop B parks while loop A's probe is in flight, then sees its verdict.

        The loop switch used to drop the probe's event but keep HALF_OPEN, and
        ``__aenter__`` then looped without ever awaiting (100% CPU, forever).
        """
        cb, _ = _open_breaker(reset_timeout=0.05)
        probing = threading.Event()
        release = threading.Event()

        async def loop_a():
            with pytest.raises(ConnectionError):
                async with cb:
                    probing.set()
                    while not release.is_set():
                        await asyncio.sleep(0.005)
                    raise ConnectionError("still down")

        thread_a, _ = _start_thread(loop_a)
        assert probing.wait(5)

        async def loop_b():
            async with cb:
                return "admitted"

        thread_b, outcome = _start_thread(loop_b)
        # Several patience timeouts pass: B re-waits instead of probing too.
        thread_b.join(0.3)
        assert thread_b.is_alive(), f"loop B got past A's probe: {outcome}"
        release.set()
        thread_a.join(5)
        thread_b.join(5)

        assert not thread_a.is_alive() and not thread_b.is_alive(), "breaker hung or spun"
        # A's probe failed: the breaker is OPEN again and B fails fast.
        assert isinstance(outcome.get("error"), CircuitOpenError)
        assert cb.state is CircuitState.OPEN

    def test_one_probe_for_all_loops(self):
        """HALF_OPEN admits one caller in total, not one per loop switch.

        Two threads, each with its own loop and many concurrent callers (two
        bibr.chew() calls on one shared Crossref client). Each loop switch used
        to demote the live probe and let the next caller probe too, so every
        caller on every loop got through within a few patience timeouts.
        """
        cb, clock = _open_breaker(reset_timeout=0.05)
        n = 20
        lock = threading.Lock()
        arrived: list[int] = []
        admitted: list[int] = []
        release = threading.Event()
        start = threading.Barrier(2)

        async def caller():
            with lock:
                arrived.append(1)
            try:
                async with cb:
                    with lock:
                        admitted.append(1)
                    while not release.is_set():
                        await asyncio.sleep(0.005)
                    raise ConnectionError("still down")
            except ConnectionError:
                return "failed"
            except CircuitOpenError:
                return "rejected"

        async def main():
            start.wait(5)
            return await asyncio.gather(*(caller() for _ in range(n)))

        threads = [_start_thread(main) for _ in range(2)]
        deadline = time.monotonic() + 5
        while len(arrived) < 2 * n and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.2)  # several patience timeouts while the probe hangs
        with lock:
            admitted_while_probing = len(admitted)
        release.set()
        for thread, _ in threads:
            thread.join(5)

        assert all(not thread.is_alive() for thread, _ in threads), "breaker hung or spun"
        assert admitted_while_probing == 1
        results = sorted(r for _, outcome in threads for r in outcome["result"])
        assert results == ["failed"] + ["rejected"] * (2 * n - 1)
        assert cb.state is CircuitState.OPEN

        # Still usable: once the reset timeout passes, a caller on yet
        # another loop probes and closes it.
        clock.now += 1.0

        async def after():
            async with cb:
                return cb.state

        finished, outcome = _run_in_thread(after)
        assert finished and outcome == {"result": CircuitState.HALF_OPEN}
        assert cb.state is CircuitState.CLOSED

    def test_probe_success_releases_waiters_on_every_loop(self):
        """A probe that succeeds on loop A wakes the waiters parked on A and on B."""
        cb, _ = _open_breaker(reset_timeout=30.0)  # a timeout wake-up would take 30 s
        probing = threading.Event()
        release = threading.Event()

        async def loop_a():
            async def probe():
                async with cb:
                    probing.set()
                    while not release.is_set():
                        await asyncio.sleep(0.005)

            async def waiter():
                await asyncio.sleep(0)  # let the probe enter first
                async with cb:
                    return cb.state

            return await asyncio.gather(probe(), waiter())

        async def loop_b():
            async with cb:
                return cb.state

        thread_a, outcome_a = _start_thread(loop_a)
        assert probing.wait(5)
        thread_b, outcome_b = _start_thread(loop_b)
        thread_b.join(0.2)
        assert thread_b.is_alive(), f"loop B got past A's probe: {outcome_b}"
        release.set()
        thread_a.join(5)
        thread_b.join(5)

        assert not thread_a.is_alive() and not thread_b.is_alive()
        assert outcome_a == {"result": [None, CircuitState.CLOSED]}
        assert outcome_b == {"result": CircuitState.CLOSED}

    def test_probe_on_a_loop_that_stopped_does_not_park_waiters_forever(self):
        """A probe pending on a loop that no longer runs can never report.

        Waiters on other loops must not re-wait for it forever: after one
        patience timeout they fail fast, as for a probe that died.
        """
        cb, _ = _open_breaker(reset_timeout=0.05)
        loop_a = asyncio.new_event_loop()
        try:
            gate = loop_a.create_future()

            async def probe():
                async with cb:
                    await gate

            probe_task = loop_a.create_task(probe())
            loop_a.run_until_complete(asyncio.sleep(0))
            assert cb.state is CircuitState.HALF_OPEN  # loop A stopped mid-probe

            async def loop_b():
                async with cb:
                    return "admitted"

            finished, outcome = _run_in_thread(loop_b)
            assert finished, "waiter re-waited forever for a probe that cannot run"
            assert isinstance(outcome.get("error"), CircuitOpenError)
            assert cb.state is CircuitState.OPEN

            gate.set_result(None)
            loop_a.run_until_complete(probe_task)
        finally:
            loop_a.close()

    def test_half_open_without_a_probe_becomes_the_probe(self):
        """HALF_OPEN with no probe at all: the caller probes instead of spinning."""
        cb = AsyncCircuitBreaker(failure_threshold=1, reset_timeout=30.0, name="orphan")

        async def scenario():
            cb._state = CircuitState.HALF_OPEN
            cb._probe_task = None
            async with cb:
                assert cb._probe_task is asyncio.current_task()
            return cb.state

        finished, outcome = _run_in_thread(scenario)
        assert finished, "HALF_OPEN without a probe spun forever"
        assert outcome == {"result": CircuitState.CLOSED}

    def test_concurrent_threads_share_one_state_machine(self):
        """Many threads hammering one breaker: no errors, no hang, trips once shared."""
        cb = AsyncCircuitBreaker(
            failure_threshold=3, reset_timeout=60.0, name="hammer", failure_dedup_window=0.0
        )
        errors: list[BaseException] = []

        async def hammer():
            for _ in range(50):
                try:
                    async with cb:
                        await asyncio.sleep(0)
                        raise ConnectionError("down")
                except (ConnectionError, CircuitOpenError):
                    pass
                except BaseException as exc:  # noqa: BLE001 — collected for the assert
                    errors.append(exc)

        threads = [threading.Thread(target=lambda: asyncio.run(hammer())) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert not any(t.is_alive() for t in threads)
        assert errors == []
        assert cb.state is CircuitState.OPEN


# --- Provider SDK 4xx are client errors -----------------------------------


def _openai_error(cls_name: str, status: int):
    openai = pytest.importorskip("openai")
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    response = httpx.Response(status, request=request, json={"error": {"message": "x"}})
    return getattr(openai, cls_name)("rejected", response=response, body=None)


class TestProviderClientErrors:
    @pytest.mark.parametrize(
        ("cls_name", "status"),
        [
            ("BadRequestError", 400),
            ("AuthenticationError", 401),
            ("PermissionDeniedError", 403),
            ("NotFoundError", 404),
            ("UnprocessableEntityError", 422),
        ],
    )
    def test_openai_4xx_is_client_error(self, cls_name, status):
        exc = _openai_error(cls_name, status)
        assert _is_client_error(type(exc), exc)

    def test_openai_5xx_is_not_client_error(self):
        exc = _openai_error("InternalServerError", 500)
        assert not _is_client_error(type(exc), exc)

    def test_google_genai_client_error_400(self):
        errors = pytest.importorskip("google.genai.errors")
        exc = errors.ClientError(
            400, {"error": {"code": 400, "message": "too long", "status": "INVALID_ARGUMENT"}}
        )
        assert _is_client_error(type(exc), exc)

    def test_google_genai_server_error_is_not_client_error(self):
        errors = pytest.importorskip("google.genai.errors")
        exc = errors.ServerError(
            503, {"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}}
        )
        assert not _is_client_error(type(exc), exc)

    def test_wrapped_sdk_400_is_client_error(self):
        """Instructor re-raises the provider error as the cause of its own."""
        inner = _openai_error("BadRequestError", 400)
        try:
            try:
                raise inner
            except Exception as e:
                raise RuntimeError("instructor gave up") from e
        except RuntimeError as outer:
            assert _is_client_error(type(outer), outer)

    def test_request_timeout_408_still_counts_as_failure(self):
        """408 reports a slow or overloaded service, not a bad request."""
        exc = _openai_error("APIStatusError", 408)
        assert not _is_client_error(type(exc), exc)
        request = httpx.Request("GET", "https://api.crossref.org/works/x")
        http_408 = httpx.HTTPStatusError(
            "timeout", request=request, response=httpx.Response(408, request=request)
        )
        assert not _is_client_error(type(http_408), http_408)

    async def test_timeout_cutting_short_an_sdk_retry_counts_as_failure(self):
        """openai sleeps before a retry inside ``except HTTPStatusError``.

        A hard timeout there chains TimeoutError <- CancelledError <- the 4xx
        being retried; the call timed out, so it must still count.
        """
        request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
        conflict = httpx.HTTPStatusError(
            "conflict", request=request, response=httpx.Response(409, request=request)
        )

        async def sdk_call():
            try:
                raise conflict
            except httpx.HTTPStatusError:
                await asyncio.sleep(10)  # backoff before the retry

        with pytest.raises(TimeoutError) as excinfo:
            async with asyncio.timeout(0.01):
                await sdk_call()
        exc = excinfo.value
        assert exc.__cause__.__context__ is conflict
        assert not _is_client_error(type(exc), exc)

    def test_httpx_4xx_still_client_error(self):
        request = httpx.Request("GET", "https://api.crossref.org/works/x")
        exc = httpx.HTTPStatusError(
            "missing", request=request, response=httpx.Response(404, request=request)
        )
        assert _is_client_error(type(exc), exc)

    async def test_context_length_400s_do_not_open_the_breaker(self):
        """Five oversized-paper 400s from a local vLLM must not cut every paper off."""
        cb = AsyncCircuitBreaker(failure_threshold=2, reset_timeout=30.0, failure_dedup_window=0.0)
        for _ in range(5):
            with pytest.raises(Exception, match="rejected"):
                async with cb:
                    raise _openai_error("BadRequestError", 400)
        assert cb.state is CircuitState.CLOSED
        async with cb:  # still admits the next paper
            pass


# --- Redis limiter: outage cooldown ---------------------------------------


def _mock_redis_limiter(eval_mock, **kwargs) -> AsyncRedisRateLimiter:
    with patch("redis.asyncio.Redis") as mock_redis_cls:
        mock_redis = AsyncMock()
        mock_redis.eval = eval_mock
        mock_redis_cls.from_url.return_value = mock_redis
        return AsyncRedisRateLimiter("redis://test", "outage", **kwargs)


class TestRedisOutageCooldown:
    @pytest.mark.parametrize("max_requests", [1, 3])
    async def test_outage_is_paid_once_not_per_acquire(self, max_requests):
        eval_mock = AsyncMock(side_effect=ConnectionError("redis down"))
        limiter = _mock_redis_limiter(
            eval_mock, max_requests=max_requests, window_seconds=0.001, max_retries=3
        )
        sleeps: list[float] = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        with patch.object(rate_limiter_mod.asyncio, "sleep", fake_sleep):
            for _ in range(10):
                await limiter.acquire()

        # One outage detection (3 attempts, two 1 s back-offs), then the
        # cooldown keeps the other nine acquires off Redis entirely.
        assert eval_mock.await_count == 3
        assert sleeps.count(1) == 2

    async def test_outage_warning_is_logged_once(self, caplog):
        eval_mock = AsyncMock(side_effect=ConnectionError("redis down"))
        limiter = _mock_redis_limiter(eval_mock, window_seconds=0.001, max_retries=1)
        with caplog.at_level(logging.WARNING, logger="bibr.utils.rate_limiter"):
            for _ in range(5):
                await limiter.acquire()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "in-process" in warnings[0].getMessage()

    async def test_budget_is_enforced_in_process_during_outage(self):
        """A Redis outage used to lift the limit entirely."""
        eval_mock = AsyncMock(side_effect=ConnectionError("redis down"))
        limiter = _mock_redis_limiter(eval_mock, window_seconds=60.0, max_retries=1)
        await limiter.acquire()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(limiter.acquire(), timeout=0.2)

    async def test_redis_is_retried_after_cooldown_and_used_once_back(self):
        eval_mock = AsyncMock(side_effect=ConnectionError("redis down"))
        limiter = _mock_redis_limiter(
            eval_mock, window_seconds=0.001, max_retries=1, fallback_seconds=3600.0
        )
        await limiter.acquire()
        await limiter.acquire()
        assert eval_mock.await_count == 1

        # Cooldown over, still down: one attempt, straight back to the fallback.
        limiter._redis_retry_at = 0.0
        await limiter.acquire()
        assert eval_mock.await_count == 2
        await limiter.acquire()
        assert eval_mock.await_count == 2

        # Cooldown over, Redis back: it limits again from then on.
        eval_mock.side_effect = None
        eval_mock.return_value = 0
        limiter._redis_retry_at = 0.0
        await limiter.acquire()
        await limiter.acquire()
        assert eval_mock.await_count == 4
        assert not limiter._redis_down

    async def test_concurrent_callers_past_cooldown_send_one_probe(self):
        """Only the first caller past the cooldown tries Redis; the rest stay local."""
        gate = asyncio.Event()

        async def hanging_eval(*_args):
            await gate.wait()
            raise ConnectionError("still down")

        eval_mock = AsyncMock(side_effect=hanging_eval)
        limiter = _mock_redis_limiter(eval_mock, max_requests=50, window_seconds=0.001)
        limiter._redis_down = True
        limiter._redis_retry_at = 0.0

        tasks = [asyncio.ensure_future(limiter.acquire()) for _ in range(20)]
        await asyncio.sleep(0.05)
        assert eval_mock.await_count == 1
        assert sum(t.done() for t in tasks) == 19
        gate.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)


# --- Redis limiter: one timeline (Redis TIME), loop-bound connections ------


def _fakeredis_limiter(**kwargs):
    import fakeredis.aioredis

    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    with patch("redis.asyncio.Redis") as mock_redis_cls:
        mock_redis_cls.from_url.return_value = fake
        limiter = AsyncRedisRateLimiter("redis://test", "skew", **kwargs)
    return limiter, fake


def _skewed_clock(offset: float) -> SimpleNamespace:
    """The limiter module's ``time``, on a host whose wall clock is ``offset`` s off."""
    return SimpleNamespace(
        time=lambda: time.time() + offset,
        time_ns=lambda: time.time_ns() + int(offset * 1e9),
        monotonic=time.monotonic,
    )


class TestRedisClockSkew:
    async def test_strict_interval_ignores_a_fast_host_clock(self):
        """A host 60 s ahead used to make every other host wait ~60 s."""
        limiter, _ = _fakeredis_limiter(max_requests=1, window_seconds=0.1)
        sleeps: list[float] = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        with patch.object(rate_limiter_mod.asyncio, "sleep", fake_sleep):
            with patch.object(rate_limiter_mod, "time", _skewed_clock(60.0)):
                await limiter.acquire()  # the fast host
            await limiter.acquire()  # a host with the right time

        assert max(sleeps, default=0.0) <= 0.2

    async def test_sliding_window_ignores_a_fast_host_clock(self):
        """Entries a fast host scored in the future used to block the window."""
        limiter, _ = _fakeredis_limiter(max_requests=2, window_seconds=0.1)
        with patch.object(rate_limiter_mod, "time", _skewed_clock(60.0)):
            await limiter.acquire()
            await limiter.acquire()
        await asyncio.sleep(0.15)  # the window has passed by the server's clock
        await asyncio.wait_for(limiter.acquire(), timeout=1.0)


class _LoopBoundRedis:
    """Stand-in for redis.asyncio: connections belong to the loop that opened them."""

    misuse: list[str] = []

    def __init__(self):
        self.loop = None
        self.calls = 0

    async def eval(self, *_args):
        loop = asyncio.get_running_loop()
        if self.loop is None:
            self.loop = loop
        elif self.loop is not loop:
            _LoopBoundRedis.misuse.append("used from another event loop")
            raise RuntimeError("Event loop is closed")
        self.calls += 1
        return 0

    async def aclose(self):
        pass


class TestRedisLimiterAcrossLoops:
    def test_each_event_loop_gets_its_own_connection(self):
        """One limiter serves successive asyncio.run() calls and threads' loops."""
        _LoopBoundRedis.misuse = []
        with patch("redis.asyncio.Redis") as mock_redis_cls:
            mock_redis_cls.from_url.side_effect = lambda *_a, **_k: _LoopBoundRedis()
            limiter = AsyncRedisRateLimiter(
                "redis://test", "loops", max_requests=1, window_seconds=0.001, max_retries=1
            )

            async def use():
                for _ in range(3):
                    await limiter.acquire()

            asyncio.run(use())
            asyncio.run(use())
            threads = [threading.Thread(target=lambda: asyncio.run(use())) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(10)

        assert _LoopBoundRedis.misuse == []
        assert not limiter._redis_down


# --- Local limiter: one wake per freed slot -------------------------------


class TestLocalLimiterWakeups:
    async def test_freed_slot_wakes_one_waiter_not_all(self):
        """Every waiter used to set a timer for the same instant: all woke per slot."""
        events: list[asyncio.Task | None] = []

        def monotonic():
            events.append(asyncio.current_task())  # one window check
            return time.monotonic()

        clock = SimpleNamespace(monotonic=monotonic, time=time.time, time_ns=time.time_ns)
        limiter = AsyncLocalRateLimiter("herd", max_requests=1, window_seconds=0.01)

        async def acquire():
            await limiter.acquire()
            events.append(None)  # a slot was taken

        with patch.object(rate_limiter_mod, "time", clock):
            await asyncio.wait_for(asyncio.gather(*(acquire() for _ in range(20))), timeout=10)

        # Distinct waiters re-checking the window between two taken slots. A
        # waiter's first check (on arrival) is not a wake-up. Counting tasks,
        # not checks, keeps a coarse clock's early timer wake-ups (Windows)
        # from mattering.
        arrived: set[asyncio.Task] = set()
        woken: set[asyncio.Task] = set()
        most = 0
        for task in events:
            if task is None:
                most, woken = max(most, len(woken)), set()
            elif task not in arrived:
                arrived.add(task)
            else:
                woken.add(task)
        assert most <= 2, f"{most} waiters woke for one slot"

    async def test_cancelled_timer_holder_hands_the_timer_on(self):
        limiter = AsyncLocalRateLimiter("herd", max_requests=1, window_seconds=0.05)
        await limiter.acquire()
        holder = asyncio.ensure_future(limiter.acquire())
        await asyncio.sleep(0)
        parked = [asyncio.ensure_future(limiter.acquire()) for _ in range(3)]
        await asyncio.sleep(0.01)
        holder.cancel()
        await asyncio.wait_for(asyncio.gather(*parked), timeout=2)

    async def test_cancelled_parked_waiter_does_not_swallow_the_hand_off(self):
        limiter = AsyncLocalRateLimiter("herd", max_requests=1, window_seconds=0.05)
        await limiter.acquire()
        waiters = [asyncio.ensure_future(limiter.acquire()) for _ in range(4)]
        await asyncio.sleep(0.01)
        waiters[1].cancel()
        await asyncio.wait_for(asyncio.gather(waiters[0], *waiters[2:]), timeout=2)

    def test_threads_with_their_own_loops_share_one_budget(self):
        """The window is process-wide: two threads' loops cannot double the rate."""
        limiter = AsyncLocalRateLimiter("threads", max_requests=1, window_seconds=0.03)
        stamps: list[float] = []

        async def use():
            for _ in range(4):
                await limiter.acquire()
                stamps.append(time.monotonic())

        threads = [threading.Thread(target=lambda: asyncio.run(use())) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert not any(t.is_alive() for t in threads)
        assert len(stamps) == 8
        stamps.sort()
        # Eight slots, one per 30 ms window: at least seven windows apart
        # (less a coarse monotonic clock's tick per window on Windows).
        tick = time.get_clock_info("monotonic").resolution
        assert stamps[-1] - stamps[0] >= 7 * (0.03 - 2 * tick) - 0.005
