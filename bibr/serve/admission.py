"""Pre-parse admission control for memory-heavy upload routes."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
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

    Two independent counters, because the two upload routes have different
    shapes and only one of them is long-running:

    ``spool`` bounds the phase this gate exists for — multipart parsing and
    buffering a body to disk. Every admitted request holds a spool slot and
    releases it the moment its upload is persisted, via
    :func:`release_spool_slot`.

    ``inflight`` bounds whole *synchronous* requests. ``POST /papers/extract``
    runs the pipeline before it responds (30-120s), so it keeps an inflight
    slot until dispatch finishes, even if the caller cancels: without that, an upload flood would queue
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


def base64_envelope(size: int) -> int:
    """Bytes a ``size``-byte file occupies once base64-encoded (4 per 3, padded)."""
    return 4 * ((max(0, size) + 2) // 3)


def _declared_content_length(scope) -> int | None:
    for name, value in scope.get("headers") or ():
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _send_json(send, status: int, payload: dict, extra_headers: Iterable = ()) -> None:
    body = json.dumps(payload).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        *extra_headers,
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class BodyAdmission:
    """Raw ASGI middleware: hold an upload slot while a large request body is received.

    The MCP transport buffers every JSON-RPC body whole before a tool sees it, and
    ``chew_paper`` carries its file inside that body — so on ``/mcp`` the memory-heavy
    phase is the *receive*, not a multipart parse. A POST to one of ``paths`` whose
    declared body exceeds ``threshold`` (or declares no length) takes a slot from the
    shared gate for exactly as long as its body is being received, and is refused with
    429 when none is free. This slot is released once the body is in, before the
    transport dispatches the tool — the tool then takes its own spool slot for the
    persist and an inflight slot for the extraction (see :mod:`bibr.serve.mcp`), so
    a single call never holds two spool slots at once.

    A body declared larger than ``max_body`` is refused with a 413 that explains the
    base64 arithmetic before a byte of it is read; LitServe's payload middleware still
    enforces the same cap on bodies that arrive without a length.
    """

    def __init__(
        self,
        app,
        *,
        gate: UploadAdmissionGate,
        paths: Iterable[str],
        threshold: int,
        max_body: int | None = None,
        max_file_size: int | None = None,
    ) -> None:
        self._app = app
        self._gate = gate
        self._paths = frozenset(paths)
        self._threshold = max(0, threshold)
        self._max_body = max_body
        self._max_file_size = max_file_size

    async def __call__(self, scope, receive, send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in self._paths
        ):
            await self._app(scope, receive, send)
            return
        declared = _declared_content_length(scope)
        if declared is not None and declared <= self._threshold:
            await self._app(scope, receive, send)
            return
        if self._max_body is not None and declared is not None and declared > self._max_body:
            await _send_json(send, 413, {"detail": self._too_large_detail(declared)})
            return
        if not self._gate.try_acquire():
            await _send_json(
                send,
                429,
                {"detail": "Too many active uploads"},
                extra_headers=[(b"retry-after", b"1")],
            )
            return

        held = True

        def _release() -> None:
            nonlocal held
            if held:
                held = False
                self._gate.release()

        async def _gated_receive():
            message = await receive()
            kind = message["type"]
            if kind == "http.disconnect" or (
                kind == "http.request" and not message.get("more_body", False)
            ):
                _release()
            return message

        try:
            await self._app(scope, _gated_receive, send)
        finally:
            _release()

    def _too_large_detail(self, declared: int) -> str:
        mib = 1024 * 1024
        detail = f"Request body of {declared / mib:.1f} MiB exceeds the {self._max_body / mib:.1f} MiB limit."  # type: ignore[operator]
        if self._max_file_size is not None:
            detail += (
                f" chew_paper accepts files up to {self._max_file_size / mib:.0f} MiB"
                " (base64 adds a third); larger files go through chew_url or"
                " POST /papers/extract."
            )
        return detail


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
    app,
    limit: int,
    *,
    inflight_limit: int | None = None,
    body_gated_paths: Iterable[str] = (),
    body_threshold: int = 0,
    max_body: int | None = None,
    max_file_size: int | None = None,
) -> UploadAdmissionGate:
    """Install fail-fast admission middleware on bibr's upload endpoints.

    ``/papers/extract`` and ``/papers/jobs`` take a spool slot for the upload;
    only the synchronous route also takes an inflight slot, held for the whole
    request. ``body_gated_paths`` (the MCP mount) take a spool slot only while
    a body larger than ``body_threshold`` is received — see
    :class:`BodyAdmission`.
    """
    from fastapi.responses import JSONResponse

    gate = UploadAdmissionGate(limit, inflight_limit=inflight_limit)
    upload_paths = frozenset({"/papers/extract", "/papers/jobs"})
    # Only the synchronous route holds a slot past the upload itself.
    inflight_paths = frozenset({"/papers/extract"})

    def _rejected(detail: str) -> JSONResponse:
        return JSONResponse({"detail": detail}, status_code=429, headers={"Retry-After": "1"})

    @app.middleware("http")
    async def _upload_admission(request, call_next):
        if request.method != "POST" or request.url.path not in upload_paths:
            return await call_next(request)
        try:
            with gate.admit(inflight=request.url.path in inflight_paths) as admission:
                setattr(request.state, _SPOOL_RELEASE_ATTR, admission.release_spool)
                request.state.upload_admission = admission
                return await call_next(request)
        except UploadAdmissionError as exc:
            return _rejected(str(exc))

    gated = tuple(body_gated_paths)
    if gated:
        app.add_middleware(
            BodyAdmission,
            gate=gate,
            paths=gated,
            threshold=body_threshold,
            max_body=max_body,
            max_file_size=max_file_size,
        )

    return gate
