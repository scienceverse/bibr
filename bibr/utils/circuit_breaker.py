"""Lightweight async circuit breaker for external service calls.

States: CLOSED → OPEN (after N failures) → HALF_OPEN (probe) → CLOSED.

In HALF_OPEN, exactly one probe request passes through.  All other
concurrent callers wait for the probe result before proceeding, preventing
the failure cascade that occurs when ``asyncio.gather`` floods the probe
window with simultaneous requests.

Usage::

    breaker = AsyncCircuitBreaker(failure_threshold=5, reset_timeout=30.0)

    async with breaker:
        result = await some_external_call()
"""

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Callable
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


def _exception_status_code(exc_val: BaseException) -> int | None:
    """Return a provider exception's numeric HTTP status when exposed.

    SDK errors carry it on the instance (openai/anthropic ``status_code``,
    google-genai ``code``); httpx's ``HTTPStatusError`` only on its response.
    """
    response = getattr(exc_val, "response", None)
    for value in (
        getattr(exc_val, "status_code", None),
        getattr(exc_val, "code", None),
        getattr(response, "status_code", None),
    ):
        if value is None or isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _is_rate_limit_error(exc_val: BaseException) -> bool:
    """Return True if the exception represents a rate-limit (429) error.

    Checks provider-specific exception types raised by the Instructor
    LLM clients (Google, OpenAI) that don't surface as httpx.HTTPStatusError.
    """
    exc_type_name = type(exc_val).__name__

    # google.api_core.exceptions.ResourceExhausted (Google Gemini 429)
    if exc_type_name == "ResourceExhausted":
        return True

    # openai.RateLimitError (OpenAI 429)
    if exc_type_name == "RateLimitError":
        return True

    # Newer provider SDKs often use one generic client-error class and expose
    # the HTTP status on the instance (e.g. google-genai ClientError.code).
    if _exception_status_code(exc_val) == 429:
        return True

    # Check wrapped cause for rate-limit errors
    cause = getattr(exc_val, "__cause__", None) or getattr(exc_val, "__context__", None)
    if cause is not None and cause is not exc_val:
        return _is_rate_limit_error(cause)

    return False


def _is_validation_error(exc_val: BaseException) -> bool:
    """Return True if the exception is (or wraps) a pydantic ``ValidationError``.

    A ValidationError means the service responded but the output didn't match
    the schema — a client-side concern, not service unhealthiness. Instructor
    may wrap the exhausted ValidationError (``InstructorRetryException``), so
    the ``__cause__``/``__context__`` chain is walked too.
    """
    try:
        from pydantic import ValidationError
    except ImportError:
        return False

    if isinstance(exc_val, ValidationError):
        return True

    cause = getattr(exc_val, "__cause__", None) or getattr(exc_val, "__context__", None)
    if cause is not None and cause is not exc_val:
        return _is_validation_error(cause)

    return False


def _is_http_client_status(exc_val: BaseException) -> bool:
    """Return True if the exception (or the error it wraps) carries a 4xx status.

    408 Request Timeout is excluded: it reports a slow or overloaded service.
    The outermost exception that exposes a status decides, so a 5xx wrapping
    an earlier 4xx still counts as a failure.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc_val
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        # A timeout or cancellation cut the call short; a 4xx it interrupted
        # (an SDK sleeping before a retry) is not this call's answer.
        if isinstance(cur, asyncio.CancelledError | TimeoutError):
            return False
        status = _exception_status_code(cur)
        if status is not None:
            return 400 <= status < 500 and status != 408
        cur = cur.__cause__ or cur.__context__
    return False


def _is_client_error(exc_type: type[BaseException], exc_val: BaseException) -> bool:  # noqa: ARG001
    """Return True if the exception represents an HTTP 4xx client error,
    a rate-limit (429) error, or a schema-validation failure.

    Rate-limit errors from LLM providers (Google, OpenAI) are treated as
    client errors so they don't trip the circuit breaker — the service is
    responsive, just asking us to slow down. Likewise pydantic validation
    failures: the model answered, the answer just didn't fit the schema (M5).
    Any other 4xx — httpx's ``HTTPStatusError`` or a provider SDK's own error
    class (openai ``BadRequestError``, google-genai ``ClientError``) — is a
    rejected request, not an unhealthy service: five context-length 400s from
    one oversized paper must not open the breaker shared by every paper.
    """
    # Native parse/schema rejection means the provider answered. Keep the
    # check dependency-free to avoid importing the LLM client stack here.
    if type(exc_val).__name__ == "NuExtractInvalidOutput":
        return True

    if _is_rate_limit_error(exc_val):
        return True

    if _is_validation_error(exc_val):
        return True

    return _is_http_client_status(exc_val)


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and requests are rejected."""

    def __init__(self, name: str, retry_after: float):
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker '{name}' is OPEN — failing fast (retry after {retry_after:.1f}s)"
        )


class AsyncCircuitBreaker:
    """Async context-manager circuit breaker.

    Only server-side failures (5xx, connection errors, timeouts) should
    trip the breaker.  Client errors (4xx) are NOT counted as failures —
    callers should catch those before exiting the ``async with`` block.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        name: str = "default",
        failure_dedup_window: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.name = name
        self._failure_dedup_window = failure_dedup_window
        # Injectable monotonic clock — lets tests drive OPEN→HALF_OPEN and
        # the dedup window deterministically instead of racing wall time.
        self._clock = clock

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0

        # One breaker can serve several event loops: test fixtures, LitServe
        # worker spawn, or a process-wide client used from threads that each
        # run their own loop (bibr.chew() from a web threadpool). The state
        # machine is shared by all of them, so it is guarded by a thread
        # lock. No critical section awaits, so it is never held across a
        # suspension point and cannot stall a loop.
        self._lock = threading.Lock()

        # HALF_OPEN probe coordination: only one request probes at a time, on
        # whichever loop it runs, and the others wait for its outcome. An
        # asyncio.Event can only be awaited on one loop, so each loop with
        # waiters gets its own; the probe's outcome sets them all.
        self._probe_task: asyncio.Task | None = None
        self._probe_events: dict[asyncio.AbstractEventLoop, asyncio.Event] = {}

    def _start_probe(self) -> None:
        """Enter HALF_OPEN with the current task as the single probe. Call with ``_lock`` held."""
        self._state = CircuitState.HALF_OPEN
        self._probe_task = asyncio.current_task()
        self._failure_count = 0
        logger.info("Circuit breaker '%s': OPEN → HALF_OPEN (probing)", self.name)

    def _probe_is_dead(self) -> bool:
        """True if no probe can still report. Call with ``_lock`` held."""
        probe = self._probe_task
        # A probe pending on a loop that no longer runs (closed, or left by
        # its thread) cannot report back.
        return probe is None or probe.done() or not probe.get_loop().is_running()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def __aenter__(self):
        loop = asyncio.get_running_loop()
        while True:
            with self._lock:
                if self._state == CircuitState.CLOSED:
                    return self

                if self._state == CircuitState.OPEN:
                    elapsed = self._clock() - self._last_failure_time
                    if elapsed >= self.reset_timeout:
                        # Transition to HALF_OPEN — this caller is the probe
                        self._start_probe()
                        return self
                    else:
                        raise CircuitOpenError(self.name, self.reset_timeout - elapsed)

                # HALF_OPEN without a probe: nothing would ever signal a
                # waiter, so this caller probes instead.
                if self._probe_task is None:
                    self._start_probe()
                    return self

                # HALF_OPEN with a probe in flight, on this loop or another —
                # grab this loop's event so we can wait outside the lock.
                event_to_wait = self._probe_events.get(loop)
                if event_to_wait is None:
                    event_to_wait = self._probe_events[loop] = asyncio.Event()

            # Wait (outside the lock) for the probe to finish, then re-check.
            # Every pass that loops around awaits here: the section above never
            # yields, so a pass without this wait would spin the loop.
            # The wait is bounded so a probe task cancelled before its
            # __aexit__ can signal waiters cannot hang them forever — but the
            # timeout is patience, not a verdict: a still-running probe owns
            # the HALF_OPEN → CLOSED/OPEN transition, and a timed-out waiter
            # must not steal it by forcing OPEN underneath a probe that is
            # about to succeed.
            try:
                await asyncio.wait_for(event_to_wait.wait(), timeout=self.reset_timeout)
            except TimeoutError:
                with self._lock:
                    if (
                        self._probe_events.get(loop) is event_to_wait
                        and not event_to_wait.is_set()
                        and self._probe_is_dead()
                    ):
                        # Probe died without signalling — fail fast so
                        # later callers see OPEN instead of parking on
                        # a dead event.
                        self._state = CircuitState.OPEN
                        self._last_failure_time = self._clock()
                        self._signal_waiters()
                    # Else the probe is still mid-flight: loop around
                    # and re-wait. Its __aexit__ will signal us.

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            is_probe = self._probe_task is not None and self._probe_task is asyncio.current_task()

            if exc_type is None:
                # Success — recover immediately
                if self._state == CircuitState.HALF_OPEN:
                    logger.info(
                        "Circuit breaker '%s': HALF_OPEN → CLOSED (recovered)",
                        self.name,
                    )
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._signal_waiters()
            elif exc_type is not None and issubclass(exc_type, asyncio.CancelledError):
                # Cancellation is caller/task control flow, not provider
                # health. A cancelled probe must still release waiters and
                # restore OPEN so no concurrent call mistakes it for success.
                if is_probe:
                    self._state = CircuitState.OPEN
                    self._last_failure_time = self._clock()
                    self._signal_waiters()
            elif exc_type is not None and _is_client_error(exc_type, exc_val):
                # Client errors (4xx) are NOT counted as failures — they
                # indicate a problem with the request, not the service.
                if is_probe:
                    # Treat a 4xx probe as success (service is responsive)
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._signal_waiters()
            else:
                # Failure — deduplicate rapid concurrent failures (e.g. from a
                # single network flap causing 16 OCR calls to fail at once).
                # Only increment the counter if enough time has passed since the
                # last recorded failure; otherwise treat it as part of the same
                # incident.
                now = self._clock()
                if (
                    self._failure_count == 0
                    or self._failure_dedup_window <= 0
                    or now - self._last_failure_time > self._failure_dedup_window
                ):
                    self._failure_count += 1
                    # Anchor deduplication to the last *counted* failure.
                    # Updating this for every rapid failure indefinitely
                    # slides the window during a sustained outage.
                    self._last_failure_time = now

                if is_probe:
                    # Probe failed — back to OPEN
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "Circuit breaker '%s': probe failed → OPEN (failures: %d)",
                        self.name,
                        self._failure_count,
                    )
                    self._signal_waiters()
                elif (
                    self._state != CircuitState.HALF_OPEN
                    and self._failure_count >= self.failure_threshold
                ):
                    # Normal CLOSED→OPEN trip (don't interfere with an active probe)
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "Circuit breaker '%s': → OPEN after %d failures",
                        self.name,
                        self._failure_count,
                    )
        # Never suppress the exception
        return False

    def _signal_waiters(self):
        """Wake every coroutine waiting on a HALF_OPEN probe result, on any loop."""
        current = asyncio.get_running_loop()
        for loop, event in self._probe_events.items():
            if loop is current:
                event.set()
            else:
                # Another thread's loop: asyncio.Event.set is not thread-safe.
                with contextlib.suppress(RuntimeError):  # that loop has closed
                    loop.call_soon_threadsafe(event.set)
        self._probe_events.clear()
        self._probe_task = None
