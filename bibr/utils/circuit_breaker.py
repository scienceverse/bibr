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
import logging
import time
from collections.abc import Callable
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


def _exception_status_code(exc_val: BaseException) -> int | None:
    """Return a provider exception's numeric HTTP status when exposed."""
    for attr in ("status_code", "code"):
        value = getattr(exc_val, attr, None)
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


def _is_client_error(exc_type: type[BaseException], exc_val: BaseException) -> bool:  # noqa: ARG001
    """Return True if the exception represents an HTTP 4xx client error,
    a rate-limit (429) error, or a schema-validation failure.

    Rate-limit errors from LLM providers (Google, OpenAI) are treated as
    client errors so they don't trip the circuit breaker — the service is
    responsive, just asking us to slow down. Likewise pydantic validation
    failures: the model answered, the answer just didn't fit the schema (M5).
    """
    # Native parse/schema rejection means the provider answered. Keep the
    # check dependency-free to avoid importing the LLM client stack here.
    if type(exc_val).__name__ == "NuExtractInvalidOutput":
        return True

    if _is_rate_limit_error(exc_val):
        return True

    if _is_validation_error(exc_val):
        return True

    try:
        import httpx

        if isinstance(exc_val, httpx.HTTPStatusError):
            return 400 <= exc_val.response.status_code < 500
    except ImportError:
        pass
    return False


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

        # ``asyncio.Lock`` and ``asyncio.Event`` lazily bind to the running
        # loop on first use. If a single ``AsyncCircuitBreaker`` instance is
        # reused across loops (e.g. test fixtures, LitServe worker spawn) the
        # second loop hits ``RuntimeError: ... bound to a different event
        # loop``. Recreate the loop-bound primitives whenever the running
        # loop changes; ``_rebind_loop_state`` is sync and runs without
        # yielding so concurrent ``__aenter__`` calls on the same loop never
        # race.
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

        # HALF_OPEN probe coordination: only one request probes at a time,
        # others wait on the event for the probe outcome.
        self._probe_event: asyncio.Event | None = None
        self._probe_task: asyncio.Task | None = None

    def _ensure_loop_state(self) -> None:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
            # Probe state is also loop-bound; reset on loop change so a
            # stale ``_probe_event`` from a dead loop can't deadlock waiters.
            self._probe_event = None
            self._probe_task = None

    @property
    def state(self) -> CircuitState:
        return self._state

    async def __aenter__(self):
        self._ensure_loop_state()
        assert self._lock is not None  # noqa: S101 — _ensure_loop_state guarantees this
        while True:
            event_to_wait = None
            async with self._lock:
                if self._state == CircuitState.CLOSED:
                    return self

                if self._state == CircuitState.OPEN:
                    elapsed = self._clock() - self._last_failure_time
                    if elapsed >= self.reset_timeout:
                        # Transition to HALF_OPEN — this caller is the probe
                        self._state = CircuitState.HALF_OPEN
                        self._probe_event = asyncio.Event()
                        self._probe_task = asyncio.current_task()
                        self._failure_count = 0
                        logger.info(
                            "Circuit breaker '%s': OPEN → HALF_OPEN (probing)",
                            self.name,
                        )
                        return self
                    else:
                        raise CircuitOpenError(self.name, self.reset_timeout - elapsed)

                # HALF_OPEN with a probe already in flight — grab the event
                # so we can wait outside the lock.
                event_to_wait = self._probe_event

            # Wait (outside the lock) for the probe to finish, then re-check.
            # The wait is bounded so a probe task cancelled before its
            # __aexit__ can signal waiters cannot hang them forever — but the
            # timeout is patience, not a verdict: a still-running probe owns
            # the HALF_OPEN → CLOSED/OPEN transition, and a timed-out waiter
            # must not steal it by forcing OPEN underneath a probe that is
            # about to succeed.
            if event_to_wait is not None:
                try:
                    await asyncio.wait_for(event_to_wait.wait(), timeout=self.reset_timeout)
                except TimeoutError:
                    async with self._lock:
                        if self._probe_event is event_to_wait and not event_to_wait.is_set():
                            probe = self._probe_task
                            if probe is None or probe.done():
                                # Probe died without signalling — fail fast so
                                # later callers see OPEN instead of parking on
                                # a dead event.
                                self._state = CircuitState.OPEN
                                self._last_failure_time = self._clock()
                                self._signal_waiters()
                            # Else the probe is still mid-flight: loop around
                            # and re-wait. Its __aexit__ will signal us.

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._ensure_loop_state()
        assert self._lock is not None  # noqa: S101 — _ensure_loop_state guarantees this
        async with self._lock:
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
        """Wake any coroutines waiting on a HALF_OPEN probe result."""
        if self._probe_event is not None:
            self._probe_event.set()
            self._probe_event = None
            self._probe_task = None
