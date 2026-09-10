"""Tests for AsyncCircuitBreaker."""

import asyncio

import pytest

from bibr.utils.circuit_breaker import AsyncCircuitBreaker, CircuitOpenError, CircuitState


async def test_disabled_dedup_counts_failures_at_the_same_clock_tick():
    cb = AsyncCircuitBreaker(failure_threshold=3, failure_dedup_window=0, clock=lambda: 0.0)
    for _ in range(3):
        with pytest.raises(ValueError):
            async with cb:
                raise ValueError("fail")
    assert cb.state == CircuitState.OPEN


class TestCircuitBreakerStateTransitions:
    """Test CLOSED → OPEN → HALF_OPEN → CLOSED transitions."""

    # Use dedup_window=0 so rapid sequential failures each count individually
    _DW = 0.0

    async def test_starts_closed(self):
        cb = AsyncCircuitBreaker(
            failure_threshold=3, reset_timeout=1.0, name="test", failure_dedup_window=self._DW
        )
        assert cb.state == CircuitState.CLOSED

    async def test_stays_closed_on_success(self):
        cb = AsyncCircuitBreaker(
            failure_threshold=3, reset_timeout=1.0, name="test", failure_dedup_window=self._DW
        )
        async with cb:
            pass
        assert cb.state == CircuitState.CLOSED

    async def test_stays_closed_below_threshold(self):
        cb = AsyncCircuitBreaker(
            failure_threshold=3, reset_timeout=1.0, name="test", failure_dedup_window=self._DW
        )
        for _ in range(2):
            with pytest.raises(ValueError):
                async with cb:
                    raise ValueError("fail")
        assert cb.state == CircuitState.CLOSED

    async def test_opens_at_threshold(self):
        cb = AsyncCircuitBreaker(
            failure_threshold=3, reset_timeout=1.0, name="test", failure_dedup_window=self._DW
        )
        for _ in range(3):
            with pytest.raises(ValueError):
                async with cb:
                    raise ValueError("fail")
        assert cb.state == CircuitState.OPEN

    async def test_open_rejects_immediately(self):
        cb = AsyncCircuitBreaker(
            failure_threshold=1, reset_timeout=60.0, name="test", failure_dedup_window=self._DW
        )
        with pytest.raises(ValueError):
            async with cb:
                raise ValueError("fail")
        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitOpenError):
            async with cb:
                pass  # should not reach here

    async def test_half_open_after_timeout(self):
        clock = FakeClock()
        cb = AsyncCircuitBreaker(
            failure_threshold=1,
            reset_timeout=30.0,
            name="test",
            failure_dedup_window=self._DW,
            clock=clock,
        )
        with pytest.raises(ValueError):
            async with cb:
                raise ValueError("fail")
        assert cb.state == CircuitState.OPEN

        clock.advance(31.0)

        # Should transition to HALF_OPEN and allow the probe
        async with cb:
            pass
        assert cb.state == CircuitState.CLOSED

    async def test_half_open_failure_reopens(self):
        clock = FakeClock()
        cb = AsyncCircuitBreaker(
            failure_threshold=1,
            reset_timeout=30.0,
            name="test",
            failure_dedup_window=self._DW,
            clock=clock,
        )
        with pytest.raises(ValueError):
            async with cb:
                raise ValueError("fail")
        assert cb.state == CircuitState.OPEN

        clock.advance(31.0)

        # Probe fails → back to OPEN
        with pytest.raises(ValueError):
            async with cb:
                raise ValueError("probe fail")
        assert cb.state == CircuitState.OPEN

    async def test_success_resets_failure_count(self):
        cb = AsyncCircuitBreaker(
            failure_threshold=3, reset_timeout=1.0, name="test", failure_dedup_window=self._DW
        )
        # 2 failures
        for _ in range(2):
            with pytest.raises(ValueError):
                async with cb:
                    raise ValueError("fail")
        # 1 success resets
        async with cb:
            pass
        assert cb._failure_count == 0
        # 2 more failures — still closed
        for _ in range(2):
            with pytest.raises(ValueError):
                async with cb:
                    raise ValueError("fail")
        assert cb.state == CircuitState.CLOSED


class FakeClock:
    """Manually-advanced monotonic clock for deterministic breaker tests."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


class TestInjectableClock:
    """The breaker accepts a clock so tests need not race wall time."""

    async def test_half_open_via_fake_clock(self):
        clock = FakeClock()
        cb = AsyncCircuitBreaker(
            failure_threshold=1,
            reset_timeout=30.0,
            name="test",
            failure_dedup_window=0.0,
            clock=clock,
        )
        with pytest.raises(ValueError):
            async with cb:
                raise ValueError("fail")
        assert cb.state == CircuitState.OPEN

        # Still rejecting before the window elapses…
        clock.advance(29.0)
        with pytest.raises(CircuitOpenError):
            async with cb:
                pass

        # …probe allowed after, no real sleeping involved.
        clock.advance(2.0)
        async with cb:
            pass
        assert cb.state == CircuitState.CLOSED

    async def test_dedup_window_via_fake_clock(self):
        """Failures inside the dedup window are one incident; outside, two."""
        clock = FakeClock()
        cb = AsyncCircuitBreaker(
            failure_threshold=3,
            reset_timeout=60.0,
            name="test",
            failure_dedup_window=2.0,
            clock=clock,
        )
        for _ in range(2):  # same instant — one incident
            with pytest.raises(ValueError):
                async with cb:
                    raise ValueError("fail")
        assert cb._failure_count == 1

        clock.advance(3.0)  # past the window — a fresh incident
        with pytest.raises(ValueError):
            async with cb:
                raise ValueError("fail")
        assert cb._failure_count == 2


def test_loop_state_tracks_loop_objects_not_reusable_ids():
    cb = AsyncCircuitBreaker()
    seen = []

    async def use_breaker():
        async with cb:
            seen.append(cb._lock_loop)

    asyncio.run(use_breaker())
    asyncio.run(use_breaker())

    assert seen[0] is not seen[1]

    async def test_sustained_failures_are_counted_from_last_counted_failure(self):
        """A rapid outage must eventually trip instead of resetting its window forever."""
        clock = FakeClock()
        cb = AsyncCircuitBreaker(
            failure_threshold=2,
            reset_timeout=60.0,
            name="test",
            failure_dedup_window=2.0,
            clock=clock,
        )

        for _ in range(4):
            with pytest.raises(ValueError):
                async with cb:
                    raise ValueError("fail")
            clock.advance(1.0)

        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerExceptionPropagation:
    """Ensure exceptions are never suppressed."""

    async def test_exception_propagates(self):
        cb = AsyncCircuitBreaker(
            failure_threshold=10, reset_timeout=1.0, name="test", failure_dedup_window=0.0
        )
        with pytest.raises(RuntimeError, match="custom error"):
            async with cb:
                raise RuntimeError("custom error")

    async def test_circuit_open_error_has_retry_after(self):
        cb = AsyncCircuitBreaker(
            failure_threshold=1, reset_timeout=30.0, name="test", failure_dedup_window=0.0
        )
        with pytest.raises(ValueError):
            async with cb:
                raise ValueError("fail")
        with pytest.raises(CircuitOpenError) as exc_info:
            async with cb:
                pass
        assert exc_info.value.retry_after > 0


class TestCircuitBreakerConcurrent:
    """Test concurrent access."""

    async def test_concurrent_successes(self):
        cb = AsyncCircuitBreaker(failure_threshold=3, reset_timeout=1.0, name="test")

        async def success():
            async with cb:
                await asyncio.sleep(0.01)

        await asyncio.gather(*[success() for _ in range(10)])
        assert cb.state == CircuitState.CLOSED

    async def test_concurrent_failures_trip_breaker(self):
        cb = AsyncCircuitBreaker(
            failure_threshold=3, reset_timeout=60.0, name="test", failure_dedup_window=0.0
        )

        async def fail():
            with pytest.raises((ValueError, CircuitOpenError)):
                async with cb:
                    raise ValueError("fail")

        await asyncio.gather(*[fail() for _ in range(5)])
        assert cb.state == CircuitState.OPEN

    async def test_dedup_window_groups_rapid_failures(self):
        """Concurrent failures within the dedup window count as one incident."""
        cb = AsyncCircuitBreaker(
            failure_threshold=3,
            reset_timeout=60.0,
            name="test",
            failure_dedup_window=5.0,  # large window — all failures are one incident
        )

        async def fail():
            with pytest.raises((ValueError, CircuitOpenError)):
                async with cb:
                    raise ValueError("fail")

        # 10 concurrent failures should count as ~1 incident (all within 5s window)
        await asyncio.gather(*[fail() for _ in range(10)])
        assert cb.state == CircuitState.CLOSED  # still closed — only 1 incident < threshold of 3


# ── Rate limit and client error detection ─────────────────────────────

from bibr.utils.circuit_breaker import _is_client_error, _is_rate_limit_error


class TestRateLimitDetection:
    """Test that rate-limit errors are correctly identified."""

    def test_google_resource_exhausted(self):
        exc = type("ResourceExhausted", (Exception,), {})("quota exceeded")
        assert _is_rate_limit_error(exc) is True

    def test_openai_rate_limit_error(self):
        exc = type("RateLimitError", (Exception,), {})("rate limit")
        assert _is_rate_limit_error(exc) is True

    def test_google_genai_client_error_429(self):
        errors = pytest.importorskip("google.genai.errors")
        exc = errors.ClientError(
            429,
            {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}},
        )
        assert _is_rate_limit_error(exc) is True

    @pytest.mark.parametrize("field", ["code", "status_code"])
    def test_provider_neutral_numeric_429(self, field):
        exc = RuntimeError("quota")
        setattr(exc, field, "429")
        assert _is_rate_limit_error(exc) is True

    def test_message_only_429_is_not_rate_limit(self):
        assert _is_rate_limit_error(RuntimeError("server emitted 429 text")) is False

    def test_generic_error_is_not_rate_limit(self):
        assert _is_rate_limit_error(ValueError("fail")) is False

    def test_wrapped_rate_limit_in_cause(self):
        inner = type("ResourceExhausted", (Exception,), {})("quota")
        outer = RuntimeError("wrapped")
        outer.__cause__ = inner
        assert _is_rate_limit_error(outer) is True

    def test_timeout_is_not_rate_limit(self):
        assert _is_rate_limit_error(TimeoutError("timed out")) is False


class TestClientErrorDetection:
    """Test that client errors (4xx) don't trip the circuit breaker."""

    def test_rate_limit_is_client_error(self):
        exc = type("ResourceExhausted", (Exception,), {})("quota")
        assert _is_client_error(type(exc), exc) is True

    def test_http_400_is_client_error(self):
        httpx = pytest.importorskip("httpx")
        response = httpx.Response(400, request=httpx.Request("POST", "http://test"))
        exc = httpx.HTTPStatusError("bad request", request=response.request, response=response)
        assert _is_client_error(type(exc), exc) is True

    def test_http_429_is_client_error(self):
        httpx = pytest.importorskip("httpx")
        response = httpx.Response(429, request=httpx.Request("POST", "http://test"))
        exc = httpx.HTTPStatusError("rate limit", request=response.request, response=response)
        assert _is_client_error(type(exc), exc) is True

    def test_http_500_is_not_client_error(self):
        httpx = pytest.importorskip("httpx")
        response = httpx.Response(500, request=httpx.Request("POST", "http://test"))
        exc = httpx.HTTPStatusError("server error", request=response.request, response=response)
        assert _is_client_error(type(exc), exc) is False

    def test_generic_error_is_not_client_error(self):
        exc = ValueError("fail")
        assert _is_client_error(type(exc), exc) is False

    async def test_rate_limit_does_not_trip_breaker(self):
        cb = AsyncCircuitBreaker(
            failure_threshold=1, reset_timeout=60.0, name="test", failure_dedup_window=0.0
        )
        exc = type("ResourceExhausted", (Exception,), {})("quota")
        with pytest.raises(type(exc)):
            async with cb:
                raise exc
        assert cb.state == CircuitState.CLOSED

    async def test_http_4xx_does_not_trip_breaker(self):
        httpx = pytest.importorskip("httpx")
        cb = AsyncCircuitBreaker(
            failure_threshold=1, reset_timeout=60.0, name="test", failure_dedup_window=0.0
        )
        response = httpx.Response(422, request=httpx.Request("POST", "http://test"))
        exc = httpx.HTTPStatusError("unprocessable", request=response.request, response=response)
        with pytest.raises(httpx.HTTPStatusError):
            async with cb:
                raise exc
        assert cb.state == CircuitState.CLOSED


class TestValidationErrorDetection:
    """M5: pydantic ValidationError means the model returned schema-wrong
    output — the service is responsive, so it must NOT trip the breaker."""

    @staticmethod
    def _validation_error():
        from pydantic import BaseModel, ValidationError

        class _M(BaseModel):
            x: int

        try:
            _M(x="not-an-int")
        except ValidationError as e:
            return e
        raise AssertionError("expected ValidationError")

    def test_validation_error_is_client_error(self):
        exc = self._validation_error()
        assert _is_client_error(type(exc), exc) is True

    def test_wrapped_validation_error_is_client_error(self):
        """Instructor may wrap the ValidationError (InstructorRetryException)."""
        inner = self._validation_error()
        try:
            raise RuntimeError("retries exhausted") from inner
        except RuntimeError as wrapper:
            assert _is_client_error(type(wrapper), wrapper) is True

    async def test_validation_error_does_not_trip_breaker(self):
        from pydantic import ValidationError

        cb = AsyncCircuitBreaker(
            failure_threshold=1, reset_timeout=60.0, name="test", failure_dedup_window=0.0
        )
        exc = self._validation_error()
        for _ in range(3):
            with pytest.raises(ValidationError):
                async with cb:
                    raise exc
        assert cb.state == CircuitState.CLOSED


class TestNeutralCancellationAndNativeInvalidOutput:
    async def test_native_invalid_output_does_not_trip_closed_breaker(self):
        from bibr.clients.nuextract import NuExtractInvalidOutput

        error = NuExtractInvalidOutput(
            category="non_json",
            model="model",
            finish_reason="stop",
            response_chars=4,
            response_sha256="a" * 64,
            input_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            cached_input_tokens=0,
            response_model="Model",
        )
        cb = AsyncCircuitBreaker(
            failure_threshold=1,
            reset_timeout=30.0,
            failure_dedup_window=0.0,
        )

        with pytest.raises(NuExtractInvalidOutput):
            async with cb:
                raise error

        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    async def test_direct_cancellation_keeps_closed_breaker_neutral(self):
        cb = AsyncCircuitBreaker(
            failure_threshold=1,
            reset_timeout=30.0,
            failure_dedup_window=0.0,
        )

        with pytest.raises(asyncio.CancelledError):
            async with cb:
                raise asyncio.CancelledError

        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    async def test_cancelled_half_open_probe_reopens_and_signals_waiters_without_counting(self):
        clock = FakeClock()
        cb = AsyncCircuitBreaker(
            failure_threshold=1,
            reset_timeout=30.0,
            failure_dedup_window=0.0,
            clock=clock,
        )
        clock.advance(0.001)
        with pytest.raises(RuntimeError):
            async with cb:
                raise RuntimeError("trip")
        assert cb.state == CircuitState.OPEN
        clock.advance(31.0)

        with pytest.raises(asyncio.CancelledError):
            async with cb:
                assert cb.state == CircuitState.HALF_OPEN
                assert cb._failure_count == 0
                raise asyncio.CancelledError

        assert cb.state == CircuitState.OPEN
        assert cb._failure_count == 0
        assert cb._probe_event is None
        assert cb._probe_task is None


class TestCircuitBreakerHalfOpenProbeCoordination:
    """Verify that during HALF_OPEN exactly one task probes and the rest wait."""

    async def _trip_to_open(self, cb, clock):
        for _ in range(cb.failure_threshold):
            # A frozen clock would land every failure at the same instant and
            # the dedup window (strict >) would collapse them into one.
            clock.advance(0.001)
            with pytest.raises(ValueError):
                async with cb:
                    raise ValueError("fail")
        assert cb.state == CircuitState.OPEN

    async def test_only_one_probe_runs_concurrently(self):
        clock = FakeClock()
        cb = AsyncCircuitBreaker(
            failure_threshold=2,
            reset_timeout=30.0,
            name="probe-test",
            failure_dedup_window=0.0,
            clock=clock,
        )
        await self._trip_to_open(cb, clock)
        clock.advance(31.0)  # past reset_timeout

        in_flight = 0
        peak = 0

        async def call(idx):
            nonlocal in_flight, peak
            async with cb:
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.05)
                in_flight -= 1

        # Three concurrent callers — only one should be inside the breaker at a time
        # while in HALF_OPEN. After probe success, the others run under CLOSED.
        await asyncio.gather(call(0), call(1), call(2))
        assert peak >= 1
        assert cb.state == CircuitState.CLOSED

    async def test_waiters_resume_after_probe_success(self):
        clock = FakeClock()
        cb = AsyncCircuitBreaker(
            failure_threshold=2,
            reset_timeout=30.0,
            name="resume-success",
            failure_dedup_window=0.0,
            clock=clock,
        )
        await self._trip_to_open(cb, clock)
        clock.advance(31.0)

        results = []

        async def probe():
            async with cb:
                await asyncio.sleep(0.05)  # hold the probe so waiters queue up
                results.append("probe")

        async def waiter():
            await asyncio.sleep(0.005)  # ensure probe enters first
            async with cb:
                results.append("waiter")

        await asyncio.gather(probe(), waiter(), waiter())
        assert results[0] == "probe"
        assert results.count("waiter") == 2
        assert cb.state == CircuitState.CLOSED

    async def test_waiters_get_circuit_open_when_probe_fails(self):
        clock = FakeClock()
        cb = AsyncCircuitBreaker(
            failure_threshold=2,
            reset_timeout=30.0,
            name="resume-fail",
            failure_dedup_window=0.0,
            clock=clock,
        )
        await self._trip_to_open(cb, clock)
        clock.advance(31.0)

        async def probe():
            with pytest.raises(ValueError):
                async with cb:
                    await asyncio.sleep(0.02)
                    raise ValueError("probe failed")

        async def waiter():
            await asyncio.sleep(0.005)
            with pytest.raises(CircuitOpenError):
                async with cb:
                    pass

        await asyncio.gather(probe(), waiter(), waiter())
        assert cb.state == CircuitState.OPEN

    async def test_probe_4xx_treated_as_success(self):
        """A 4xx during a HALF_OPEN probe means the service is up (just our request was bad).
        It should close the breaker, not re-open it."""
        httpx = pytest.importorskip("httpx")
        clock = FakeClock()
        cb = AsyncCircuitBreaker(
            failure_threshold=2,
            reset_timeout=30.0,
            name="probe-4xx",
            failure_dedup_window=0.0,
            clock=clock,
        )
        await self._trip_to_open(cb, clock)
        clock.advance(31.0)

        response = httpx.Response(404, request=httpx.Request("GET", "http://x"))
        exc = httpx.HTTPStatusError("not found", request=response.request, response=response)
        with pytest.raises(httpx.HTTPStatusError):
            async with cb:
                raise exc
        assert cb.state == CircuitState.CLOSED

    async def test_cancelled_probe_force_opens_breaker_for_waiters(self):
        """If the probe task is cancelled before its __aexit__ can signal waiters,
        the waiters' wait_for should time out and force the breaker back to OPEN."""
        clock = FakeClock()
        cb = AsyncCircuitBreaker(
            failure_threshold=2,
            # reset_timeout doubles as the waiters' real wait_for timeout —
            # keep it short so the forced-open path triggers quickly. The
            # OPEN→HALF_OPEN crossing itself is driven by the fake clock.
            reset_timeout=0.1,
            name="cancelled-probe",
            failure_dedup_window=0.0,
            clock=clock,
        )
        await self._trip_to_open(cb, clock)
        clock.advance(0.2)

        async def probe_then_cancel():
            # Enter the breaker (becomes the probe), then never exit normally —
            # we'll cancel from outside.
            async with cb:
                await asyncio.sleep(10)  # would be cancelled

        probe_task = asyncio.create_task(probe_then_cancel())
        await asyncio.sleep(0.005)  # let probe enter HALF_OPEN

        async def waiter():
            with pytest.raises(CircuitOpenError):
                async with cb:
                    pass

        waiter_task = asyncio.create_task(waiter())

        # Cancel the probe — it will never reach __aexit__ to signal waiters
        probe_task.cancel()
        try:
            await probe_task
        except asyncio.CancelledError:
            pass

        await waiter_task
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerCrossLoop:
    """Reusing one breaker instance across event loops (test fixtures, worker
    spawn) must not raise ``RuntimeError: Lock is bound to a different event
    loop``."""

    def test_breaker_reused_across_loops(self):
        cb = AsyncCircuitBreaker(failure_threshold=3, reset_timeout=1.0, name="cross-loop")

        async def probe():
            async with cb:
                pass

        for _ in range(2):
            asyncio.new_event_loop().run_until_complete(probe())
        assert cb.state == CircuitState.CLOSED
