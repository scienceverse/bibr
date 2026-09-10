"""Pre-parse admission control for memory-heavy upload routes."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

#: Where the middleware parks this request's early spool release, for
#: :func:`release_spool_slot` to find. Lives on ``request.state``, which is
#: backed by the ASGI ``scope`` and so is shared with the route handler.
_SPOOL_RELEASE_ATTR = "upload_spool_release"


class _SlotCounter:
    """A non-blocking counting semaphore. ``limit`` 0 disables the bound."""

    __slots__ = ("active", "limit", "name")

    def __init__(self, limit: int, name: str) -> None:
        self.limit = max(0, limit)
        self.active = 0
        self.name = name

    def try_acquire(self) -> bool:
        if self.limit == 0:
            return True
        if self.active >= self.limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        if self.limit == 0:
            return
        if self.active <= 0:
            raise RuntimeError(f"upload admission {self.name} slot released without an acquisition")
        self.active -= 1


class UploadAdmissionError(RuntimeError):
    """The shared upload or extraction capacity is exhausted."""


class UploadAdmission:
    """Own acquired slots, optionally handing inference ownership to dispatch."""

    def __init__(self, spool: _SlotCounter, inflight: _SlotCounter | None) -> None:
        self._spool: _SlotCounter | None = spool
        self._inflight = inflight

    def release_spool(self) -> None:
        if self._spool is not None:
            self._spool.release()
            self._spool = None

    def detach_inflight(self) -> Callable[[], None] | None:
        """Transfer the slot to the dispatch task, which outlives cancellation."""
        counter, self._inflight = self._inflight, None
        return counter.release if counter is not None else None

    def release(self) -> None:
        self.release_spool()
        release_inflight = self.detach_inflight()
        if release_inflight is not None:
            release_inflight()


class UploadAdmissionGate:
    """Reject excess uploads before Starlette parses or buffers their bodies.

    Two independent counters separate upload buffering from extraction:

    ``spool`` bounds multipart/JSON parsing, MCP downloads, and buffering a
    body to disk. Every admitted upload holds a spool slot and
    releases it the moment its upload is persisted, via
    :func:`release_spool_slot`.

    ``inflight`` bounds synchronous REST and MCP extraction. Each keeps an
    inflight slot until dispatch finishes, even if the caller cancels:
    without that, an upload flood would queue
    unboundedly on the worker's in-flight semaphore, each waiter pinning a
    connection and a spooled file rather than being told to back off.

    ``POST /papers/jobs`` returns 202 as soon as the upload is queued, so it
    takes no inflight slot — and that is the point of the split. Previously one
    counter was held across the whole request, so N concurrent extracts
    rejected async job submissions with 429 while the upload path itself sat
    completely idle. Spooling and running are now bounded separately.

    FastAPI middleware executes on one event-loop thread.  ``try_acquire`` has
    no await point, so the check-and-increment is atomic with respect to other
    requests on that loop.
    """

    def __init__(self, limit: int, *, inflight_limit: int | None = None) -> None:
        self.spool = _SlotCounter(limit, "spool")
        self.inflight = _SlotCounter(
            limit if inflight_limit is None else inflight_limit, "inflight"
        )

    @property
    def limit(self) -> int:
        return self.spool.limit

    @property
    def active(self) -> int:
        return self.spool.active

    def try_acquire(self) -> bool:
        return self.spool.try_acquire()

    def release(self) -> None:
        self.spool.release()

    @contextmanager
    def admit(self, *, inflight: bool = True) -> Iterator[UploadAdmission]:
        if not self.spool.try_acquire():
            raise UploadAdmissionError("Too many active uploads")
        if inflight and not self.inflight.try_acquire():
            self.spool.release()
            raise UploadAdmissionError("Too many requests in flight")
        admission = UploadAdmission(self.spool, self.inflight if inflight else None)
        try:
            yield admission
        finally:
            admission.release()


def release_spool_slot(request) -> None:
    """Release this request's spool slot early — call once the upload is on disk.

    Idempotent, and a no-op for requests the gate never admitted (every other
    route, and any caller invoking a handler outside the middleware stack). The
    middleware still releases on exit, so a route that forgets to call this
    degrades to the old hold-it-all-the-way behaviour rather than leaking a slot.
    """
    release = getattr(request.state, _SPOOL_RELEASE_ATTR, None)
    if release is not None:
        release()


def add_upload_admission(
    app, limit: int, *, inflight_limit: int | None = None
) -> UploadAdmissionGate:
    """Install fail-fast admission middleware on bibr's upload endpoints."""
    from fastapi.responses import JSONResponse

    gate = UploadAdmissionGate(limit, inflight_limit=inflight_limit)
    upload_paths = frozenset({"/papers/extract", "/papers/jobs"})
    # Only the synchronous route holds a slot past the upload itself.
    inflight_paths = frozenset({"/papers/extract"})

    def _rejected(detail: str) -> JSONResponse:
        return JSONResponse({"detail": detail}, status_code=429, headers={"Retry-After": "1"})

    @app.middleware("http")
    async def _upload_admission(request, call_next):
        path = request.url.path
        # MCP parses whole JSON bodies (including base64 uploads). Bound that
        # allocation before the SDK reads the body, including mounted subpaths.
        is_mcp = path == "/mcp" or path.startswith("/mcp/")
        if request.method != "POST" or (path not in upload_paths and not is_mcp):
            return await call_next(request)
        try:
            with gate.admit(inflight=path in inflight_paths) as admission:
                setattr(request.state, _SPOOL_RELEASE_ATTR, admission.release_spool)
                request.state.upload_admission = admission
                return await call_next(request)
        except UploadAdmissionError as exc:
            return _rejected(str(exc))

    return gate
