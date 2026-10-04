"""Async job API for the bibr serve layer (D1).

Submitting a large PDF and holding an HTTP connection open for the full
extraction (tens of seconds) is fragile behind proxies/load balancers. These
routes offer a fire-and-poll alternative:

  - ``POST /papers/jobs``            → 202 ``{job_id, status, status_url}``
  - ``GET  /papers/jobs/{id}``       → job status (no result body)
  - ``GET  /papers/jobs/{id}/result`` → the paper_json once succeeded
  - ``DELETE /papers/jobs/{id}``     → cancel a job that is still queued

**Cancelling.** A queued job is failed at once with ``error_code``
``job_cancelled``: it stops counting against ``JOBS_MAX_ACTIVE``, its upload is
deleted (on the replica that holds it; another replica's worker skips it when it
dequeues it), and its result answers ``410``. A cancelled job reports
``status: failed``, so a client that polls for ``succeeded``/``failed`` stops
like it does for any failure. A running job cannot be stopped between pipeline
stages yet, so ``DELETE`` answers ``409`` for it.

**Duplicate uploads.** With ``JOBS_DEDUPE_INFLIGHT=true`` a ``POST`` whose file
(sha256), filename and options match a job this replica still has queued or
running returns that job instead of queueing the paper again. The active-job cap
is checked first, so at ``JOBS_MAX_ACTIVE`` a duplicate is refused with 429 too.
bibr serve has one principal (the shared API key), so this suits a single-tenant deployment; a
multi-user front end must deduplicate per user itself.

**Architecture.** Jobs persist their uploads through the same disk-backed
ingress as ``/papers/extract`` and queue only its opaque descriptor. The job
runner submits that descriptor through LitServe's private inference adapter,
which owns worker dispatch and upload cleanup without rebuilding a multipart
request or forwarding caller credentials.

**Job state** lives behind the :class:`JobStore` protocol. The default
:class:`MemoryJobStore` is a process-local dict: every custom route runs in the
*one* API-server process that ``serve.app.main`` pins (LitServe 0.2.17 would
otherwise default ``num_api_servers`` to the inference-worker count), so it is
consistent without Redis — but only for a single ``bibr serve`` instance.
``JOBS_STORE=redis`` swaps in :class:`bibr.serve.jobs_redis.RedisJobStore`,
which shares job status, results, and the active-job cap between replicas
behind a load balancer: any replica answers status/result polls for a job that
another replica accepted. Upload bytes and execution still belong to the
replica that received the upload (its ``UploadStore`` and ``JobDispatcher``
are process-local), and the job record carries that replica's id. Handing
queued work to a different replica (a shared queue) is a separate design.

The store protocol reports backend outages as :class:`JobStoreUnavailableError`;
the routes translate it into ``503 {"detail": "job store unavailable"}`` and
the runner logs (never raises) when it cannot record a state transition.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

from bibr.config import Settings
from bibr.serve.admission import release_spool_slot
from bibr.serve.ingress import (
    MULTIPART_OPENAPI_EXTRA,
    EmptyUploadError,
    InferenceDispatchTracker,
    InvalidMultipartError,
    InvalidUploadOptionError,
    StoredUpload,
    UploadStorageError,
    UploadStore,
    UploadTooLargeError,
    bounded_upload_filename,
    parse_multipart_request,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger("bibr.serve.jobs")

# Job statuses that count against the active cap (not yet terminal).
_ACTIVE_STATUSES = frozenset({"queued", "running"})

_STORE_UNAVAILABLE_DETAIL = "job store unavailable"
# Recorded on jobs the owning replica abandons at shutdown (queued ones lose their
# upload; running ones lose their inference task). 503 tells the client to resubmit.
_SHUTDOWN_ERROR = {"detail": "replica shut down before the job finished"}
# Recorded on a job cancelled while it waited in the queue. It is an ordinary failure,
# so every client that polls for succeeded/failed stops; its result answers 410 Gone.
CANCELLED_ERROR = {"detail": "job cancelled before it started", "error_code": "job_cancelled"}
CANCELLED_HTTP_STATUS = 410


class JobCapacityError(Exception):
    """Raised by :meth:`JobStore.create` when the active-job cap is reached."""


class JobStoreUnavailableError(Exception):
    """The backing job store could not be reached (or answer) within its timeout.

    Only shared stores raise it; the memory store never does. Routes map it to
    ``503``; the runner logs it and lets the record's TTL clean up.
    """


def default_replica_id() -> str:
    """``<hostname>:<pid>`` — distinct per ``bibr serve`` process by construction."""
    return f"{socket.gethostname()}:{os.getpid()}"


def _iso(wall: float | None) -> str | None:
    """UTC ISO-8601 for a wall-clock epoch seconds value (None passes through)."""
    if wall is None:
        return None
    return datetime.fromtimestamp(wall, tz=UTC).isoformat()


@dataclass
class Job:
    """A single async extraction job.

    The memory store records monotonic timestamps next to the wall ones and
    derives durations and TTL from them (immune to wall-clock jumps). A shared
    store can only exchange wall timestamps between replicas, so when the
    monotonic fields are ``None`` durations fall back to the wall clock. The
    ISO fields surfaced to clients always come from the wall timestamps.
    """

    job_id: str
    filename: str
    status: str = "queued"  # queued | running | succeeded | failed
    created_mono: float | None = None
    created_wall: float = 0.0
    started_mono: float | None = None
    started_wall: float | None = None
    finished_mono: float | None = None
    finished_wall: float | None = None
    http_status: int | None = None
    error: dict | None = None  # {kind?, detail}
    # The replica (``JOBS_REPLICA_ID``) whose dispatcher executes the job.
    replica: str | None = None
    # The paper_json, rendered once at completion exactly as ``JSONResponse`` would
    # render it. Keeping the encoded body (not the dict) makes the result's memory
    # cost a plain byte count, which is what the retention budget is charged in.
    result_json: bytes | None = field(default=None, repr=False)
    # Encoded-result byte count for a status-only read that left the body behind
    # (shared stores); the memory store derives it from ``result_json``.
    result_size_hint: int | None = field(default=None, repr=False)
    task: object | None = field(default=None, repr=False)  # asyncio.Task ref

    @property
    def result(self) -> dict | None:
        """The paper JSON, decoded on demand (tests and introspection only)."""
        if self.result_json is None:
            return None
        return json.loads(self.result_json)

    @property
    def result_size(self) -> int:
        """Bytes this job's result holds in the store (0 for failed/unfinished jobs)."""
        if self.result_json is not None:
            return len(self.result_json)
        return self.result_size_hint or 0

    @property
    def duration_ms(self) -> int | None:
        if self.finished_mono is not None and self.created_mono is not None:
            start = self.started_mono if self.started_mono is not None else self.created_mono
            return int((self.finished_mono - start) * 1000)
        if self.finished_wall is not None:
            start = self.started_wall if self.started_wall is not None else self.created_wall
            return max(0, int((self.finished_wall - start) * 1000))
        return None

    def status_dict(self) -> dict:
        """Status view — everything except the (potentially large) result body."""
        out: dict = {
            "job_id": self.job_id,
            "status": self.status,
            "filename": self.filename,
            "replica": self.replica,
            "created_at": _iso(self.created_wall),
            "started_at": _iso(self.started_wall),
            "finished_at": _iso(self.finished_wall),
        }
        duration_ms = self.duration_ms
        if duration_ms is not None:
            out["duration_ms"] = duration_ms
        if self.status == "failed" and self.error is not None:
            out["error"] = self.error
        if self.status == "succeeded":
            out["result_url"] = f"/papers/jobs/{self.job_id}/result"
        return out


def is_cancelled(job: Job) -> bool:
    """Whether ``job`` is a queued job that ``DELETE /papers/jobs/{id}`` cancelled."""
    return (
        job.status == "failed"
        and isinstance(job.error, dict)
        and job.error.get("error_code") == CANCELLED_ERROR["error_code"]
    )


def encode_result(result: dict) -> bytes:
    """Render a job result the way ``JSONResponse`` does, once, at completion."""
    return json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )


class JobStore(Protocol):
    """Job state shared by the submit/status/result routes and the job runner.

    Implementations: :class:`MemoryJobStore` (one process) and
    :class:`bibr.serve.jobs_redis.RedisJobStore` (shared between replicas).
    Every method reads the retention knobs from ``Settings.jobs`` at call time.
    ``create`` raises :class:`JobCapacityError` past ``max_active``; shared
    stores raise :class:`JobStoreUnavailableError` from ``create``/``get`` when
    the backend is unreachable and only log it from the ``set_*``/``discard``
    transitions, so a backend outage never escapes the runner.
    """

    async def create(self, *, filename: str) -> Job: ...

    async def get(self, job_id: str, *, include_result: bool = True) -> Job | None:
        """Return the job, or ``None`` if unknown/expired.

        ``include_result=False`` lets a store leave the result body behind for a
        status poll; ``Job.result_size`` still reports its size.
        """
        ...

    async def discard(self, job_id: str) -> None:
        """Remove a job that failed before it could be admitted to dispatch."""
        ...

    async def set_running(self, job_id: str) -> bool:
        """Claim a dequeued job for execution.

        ``False`` means the runner must skip the job: it is no longer queued (it was
        cancelled while it waited) or its record is gone (a finished record, such as
        a cancelled one, that retention already evicted). A store that cannot be
        reached returns ``True``, so an outage does not stop the work.
        """
        ...

    async def cancel(self, job_id: str) -> Job | None:
        """Fail a *queued* job as cancelled and return the job as it now stands.

        The cancelled job stops counting against the active cap at once. A running or
        finished job is returned unchanged; ``None`` if unknown/expired. Shared stores
        raise :class:`JobStoreUnavailableError` when the backend cannot be reached.
        """
        ...

    async def set_succeeded(self, job_id: str, result: dict) -> None: ...

    async def set_failed(
        self, job_id: str, *, http_status: int | None, error: dict, required: str = ""
    ) -> None:
        """Record a failure; with ``required`` set, only a job in that status changes."""
        ...

    async def close(self) -> None: ...


class MemoryJobStore:
    """In-process job store: a dict guarded by an ``asyncio.Lock``.

    Finished jobs are purged lazily (on every access) once older than
    ``Settings.jobs.ttl_seconds``, and oldest-first beyond ``Settings.jobs.max_retained``
    results or ``Settings.jobs.max_retained_bytes`` of encoded result bodies (the newest
    result always survives the byte budget, so even an export larger than the budget can
    be fetched once). The active-job cap (``Settings.jobs.max_active``) bounds
    queued+running jobs so an upload flood can't grow the store without limit.
    ``clock`` / ``wall_clock`` are injectable for deterministic tests.
    """

    def __init__(
        self,
        *,
        clock=time.monotonic,
        wall_clock=time.time,
        replica_id: str | None = None,
    ) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._clock = clock
        self._wall_clock = wall_clock
        self._replica_id = replica_id or default_replica_id()

    @property
    def replica_id(self) -> str:
        return self._replica_id

    def _purge_locked(self) -> None:
        ttl = Settings.jobs.ttl_seconds
        now = self._clock()
        expired = [
            jid
            for jid, job in self._jobs.items()
            if job.finished_mono is not None and (now - job.finished_mono) > ttl
        ]
        for jid in expired:
            del self._jobs[jid]

        max_retained = Settings.jobs.max_retained
        max_bytes = Settings.jobs.max_retained_bytes
        finished = sorted(
            (job for job in self._jobs.values() if job.finished_mono is not None),
            key=lambda job: job.finished_mono or 0.0,
        )
        retained_bytes = sum(job.result_size for job in finished)
        evict = 0
        while evict < len(finished):
            remaining = len(finished) - evict
            over_count = remaining > max_retained
            # Results are evicted oldest-first until the rest fit the byte budget,
            # but the most recent result is never evicted for size alone.
            over_bytes = max_bytes > 0 and retained_bytes > max_bytes and remaining > 1
            if not (over_count or over_bytes):
                break
            retained_bytes -= finished[evict].result_size
            evict += 1
        for job in finished[:evict]:
            del self._jobs[job.job_id]

    async def create(self, *, filename: str) -> Job:
        async with self._lock:
            self._purge_locked()
            active = sum(1 for j in self._jobs.values() if j.status in _ACTIVE_STATUSES)
            if active >= Settings.jobs.max_active:
                raise JobCapacityError(
                    f"active job cap reached ({active}/{Settings.jobs.max_active})"
                )
            job = Job(
                job_id=uuid.uuid4().hex,
                filename=filename,
                created_mono=self._clock(),
                created_wall=self._wall_clock(),
                replica=self._replica_id,
            )
            self._jobs[job.job_id] = job
            return job

    async def get(self, job_id: str, *, include_result: bool = True) -> Job | None:  # noqa: ARG002
        # The body is already in memory, so a status-only read costs nothing extra.
        async with self._lock:
            self._purge_locked()
            return self._jobs.get(job_id)

    async def discard(self, job_id: str) -> None:
        """Remove a job that failed before it could be admitted to dispatch."""
        async with self._lock:
            self._jobs.pop(job_id, None)

    async def set_running(self, job_id: str) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != "queued":
                return False
            job.status = "running"
            job.started_mono = self._clock()
            job.started_wall = self._wall_clock()
            return True

    async def cancel(self, job_id: str) -> Job | None:
        async with self._lock:
            self._purge_locked()
            job = self._jobs.get(job_id)
            if job is None or job.status != "queued":
                return job
            job.status = "failed"
            job.http_status = CANCELLED_HTTP_STATUS
            job.error = dict(CANCELLED_ERROR)
            job.finished_mono = self._clock()
            job.finished_wall = self._wall_clock()
            self._purge_locked()
            return job

    async def set_succeeded(self, job_id: str, result: dict) -> None:
        # Encode outside the lock: a large export takes real CPU time to render.
        encoded = encode_result(result)
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "succeeded"
            job.result_json = encoded
            job.finished_mono = self._clock()
            job.finished_wall = self._wall_clock()
            self._purge_locked()

    async def set_failed(
        self, job_id: str, *, http_status: int | None, error: dict, required: str = ""
    ) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or (required and job.status != required):
                return
            job.status = "failed"
            job.http_status = http_status
            job.error = error
            job.finished_mono = self._clock()
            job.finished_wall = self._wall_clock()
            self._purge_locked()

    async def close(self) -> None:
        """Nothing to release; the dict dies with the process."""


def build_job_store(settings) -> JobStore:
    """Construct the store ``JOBS_STORE`` selects, failing fast on a bad setup.

    ``JOBS_STORE=redis`` needs a URL (``JOBS_REDIS_URL``, else ``REDIS_URL``) and
    the optional ``redis`` dependency; both are checked here, before any model
    loads, and reported as :class:`bibr.exceptions.ConfigurationError`.
    """
    from bibr.exceptions import ConfigurationError

    jobs = settings.jobs
    replica_id = jobs.replica_id or default_replica_id()
    if jobs.store == "memory":
        return MemoryJobStore(replica_id=replica_id)

    url = jobs.redis_url or settings.redis.url
    if not url:
        raise ConfigurationError(
            "JOBS_STORE=redis requires a Redis URL: set JOBS_REDIS_URL (or REDIS_URL / "
            "REDIS_PASSWORD for the compose default).",
            problems=["JOBS_REDIS_URL: unset, and REDIS_URL is unset too"],
        )
    try:
        from bibr.serve.jobs_redis import RedisJobStore
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] == "redis":
            raise ConfigurationError(
                "JOBS_STORE=redis requires the optional 'redis' dependency — install it "
                "with 'uv sync --extra cache' (source checkout) or pip install 'bibr[cache]'."
            ) from e
        raise
    return RedisJobStore(
        url,
        key_prefix=jobs.key_prefix,
        replica_id=replica_id,
        connect_timeout=settings.redis.connect_timeout_seconds,
        socket_timeout=settings.redis.socket_timeout_seconds,
    )


_REQUEST_ID_STRIP = re.compile(r"[^-a-zA-Z0-9_]")


def _sanitize_request_id(raw: str | None) -> str | None:
    """Keep only ``[-a-zA-Z0-9_]``, cap at 64 chars. Empty → None."""
    if not raw:
        return None
    cleaned = _REQUEST_ID_STRIP.sub("", raw)[:64]
    return cleaned or None


_BOOLEAN_OPTIONS = frozenset({"include_figures", "include_regions", "crossref"})
_INTEGER_OPTIONS = frozenset({"start_page", "end_page"})


def _canonical_option(name: str, value: object) -> str:
    text = str(value).strip()
    if name in _BOOLEAN_OPTIONS:
        return "true" if text.lower() in ("true", "1", "yes") else "false"
    if name in _INTEGER_OPTIONS:
        try:
            return str(int(text))
        except ValueError:
            return text
    return text.lower()


def upload_fingerprint(sha256_hex: str, filename: str, options: Mapping[str, object]) -> str:
    """What makes two uploads the same request: the file's sha256, name and options.

    The name counts because the export carries it (``source.file_name``): a renamed
    copy of a file runs again rather than receiving a result named after the first
    upload. Options compare by meaning (``1``/``yes``/``true``, ``03``/``3``); an
    option left out is not the same as its default spelled out, so such a pair just
    runs twice.
    """
    canonical = {name: _canonical_option(name, value) for name, value in options.items()}
    material = {"filename": filename, "options": canonical}
    return sha256_hex + ":" + json.dumps(material, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class JobPayload:
    """Small queue descriptor for one disk-backed job upload."""

    descriptor: dict[str, object]


class JobDispatcher:
    """Process-local FIFO dispatcher with bounded descriptor dispatch concurrency.

    It also knows which of its jobs still wait in the queue (so a cancelled one's
    upload can be deleted at once) and, for ``JOBS_DEDUPE_INFLIGHT``, the upload
    fingerprint of every job it has queued or running.
    """

    def __init__(
        self,
        *,
        store: JobStore,
        tracker: InferenceDispatchTracker,
        max_running: int,
    ) -> None:
        self._store = store
        self._tracker = tracker
        self._max_running = max(1, max_running)
        self._queue: asyncio.Queue[tuple[str, JobPayload]] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._closed = False
        # Jobs in the queue that no worker has taken yet.
        self._waiting: dict[str, JobPayload] = {}
        # Upload fingerprint -> job id (and back) for jobs queued or running here.
        self._by_fingerprint: dict[str, str] = {}
        self._fingerprint_of: dict[str, str] = {}

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("job dispatcher is closed")
        if not self._workers:
            self._workers = [
                asyncio.create_task(self._worker(), name=f"bibr-job-worker-{index}")
                for index in range(self._max_running)
            ]

    async def submit(
        self, job_id: str, payload: JobPayload, *, fingerprint: str | None = None
    ) -> None:
        await self.start()
        self._waiting[job_id] = payload
        if fingerprint is not None:
            self._by_fingerprint[fingerprint] = job_id
            self._fingerprint_of[job_id] = fingerprint
        self._queue.put_nowait((job_id, payload))

    def job_for_fingerprint(self, fingerprint: str) -> str | None:
        """The job this dispatcher holds (queued or running) for an upload fingerprint."""
        return self._by_fingerprint.get(fingerprint)

    async def release_cancelled(self, job_id: str) -> None:
        """Delete the upload of a job cancelled while it waited here, without waiting
        for a worker to dequeue it (the worker then skips it)."""
        self._forget_fingerprint(job_id)
        payload = self._waiting.pop(job_id, None)
        if payload is not None:
            await self._tracker.discard(payload.descriptor)

    def _forget_fingerprint(self, job_id: str) -> None:
        fingerprint = self._fingerprint_of.pop(job_id, None)
        if fingerprint is not None and self._by_fingerprint.get(fingerprint) == job_id:
            del self._by_fingerprint[fingerprint]

    async def join(self) -> None:
        await self._queue.join()

    async def _worker(self) -> None:
        # A dependency may consume cancellation while finishing a request.
        # Closing must still stop this worker before it waits for another job.
        while not self._closed:
            job_id, payload = await self._queue.get()
            self._waiting.pop(job_id, None)
            try:
                await _run_job(
                    store=self._store,
                    job_id=job_id,
                    descriptor=payload.descriptor,
                    tracker=self._tracker,
                )
            except Exception:
                # _run_job records every expected failure itself; this fence keeps
                # an unexpected one (a store bug, say) from killing the worker loop
                # and silently stranding every job queued behind it.
                logger.exception("job %s: runner failed outside the job's own handling", job_id)
            finally:
                self._forget_fingerprint(job_id)
                self._queue.task_done()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        while True:
            try:
                job_id, payload = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            # A job cancelled here already left _waiting and has its final record.
            cancelled = self._waiting.pop(job_id, None) is None
            self._forget_fingerprint(job_id)
            try:
                await self._tracker.discard(payload.descriptor)
                # The upload is gone, so the job can never run: say so instead of
                # leaving a "queued" record (in a shared store, one that would hold
                # a cap slot until its safety TTL).
                # Only a job still queued: another replica may have cancelled it.
                if not cancelled:
                    await self._store.set_failed(
                        job_id, http_status=503, error=_SHUTDOWN_ERROR, required="queued"
                    )
            finally:
                self._queue.task_done()


async def _run_job(
    *,
    store: JobStore,
    job_id: str,
    descriptor: dict[str, object],
    tracker: InferenceDispatchTracker,
) -> None:
    """Execute one job through LitServe's in-process descriptor adapter."""

    if await store.set_running(job_id) is False:
        # Cancelled while it waited (its record is final, or already evicted as a
        # finished one): only the upload is left, and that is already gone when the
        # cancel reached the replica that holds it.
        logger.info("job %s: cancelled or expired before it started; skipped", job_id)
        await tracker.discard(descriptor)
        return
    try:
        result = await tracker.submit(descriptor)
    except asyncio.CancelledError:
        # Shutdown cancelled the worker mid-flight; the tracker discards the
        # upload, so record the loss (bounded, in a shared store) and keep
        # unwinding rather than leave a "running" record behind.
        await store.set_failed(job_id, http_status=503, error=_SHUTDOWN_ERROR)
        raise
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {"detail": str(exc.detail)}
        await store.set_failed(
            job_id,
            http_status=exc.status_code,
            error=detail,
        )
    except Exception:
        logger.exception("job %s crashed during inference dispatch", job_id)
        await store.set_failed(
            job_id,
            http_status=500,
            error={"detail": "internal job error"},
        )
    else:
        try:
            await store.set_succeeded(job_id, result)
        except (TypeError, ValueError):
            logger.exception("job %s produced a result that cannot be rendered as JSON", job_id)
            await store.set_failed(
                job_id,
                http_status=500,
                error={"detail": "internal job error"},
            )


def register_job_routes(
    app: FastAPI,
    *,
    store: JobStore,
    upload_store: UploadStore,
    tracker: InferenceDispatchTracker,
) -> None:
    """Mount the async-job routes on ``app``.

    Routes are NOT added to ``PUBLIC_PATHS`` — the app-wide ``_auth_gate``
    middleware covers them like any other route. Jobs share the public
    extraction route's disk-backed upload store and LitServe dispatch tracker.
    The caller owns ``store`` (and closes it); this only closes the dispatcher.
    """

    async def _close_job_resources() -> None:
        dispatcher = getattr(app.state, "job_dispatcher", None)
        if isinstance(dispatcher, JobDispatcher):
            await dispatcher.close()

    app.router.add_event_handler("shutdown", _close_job_resources)

    def _job_dispatcher() -> JobDispatcher:
        dispatcher = getattr(app.state, "job_dispatcher", None)
        if not isinstance(dispatcher, JobDispatcher) or dispatcher._closed:
            dispatcher = JobDispatcher(
                store=store,
                tracker=tracker,
                max_running=Settings.jobs.max_running,
            )
            app.state.job_dispatcher = dispatcher
        return dispatcher

    async def _active_duplicate(dispatcher: JobDispatcher, fingerprint: str) -> Job | None:
        existing_id = dispatcher.job_for_fingerprint(fingerprint)
        if existing_id is None:
            return None
        existing = await store.get(existing_id, include_result=False)
        if existing is None or existing.status not in _ACTIVE_STATUSES:
            return None  # finished or cancelled since: run this upload afresh
        return existing

    def _store_unavailable(job_id: str | None, exc: JobStoreUnavailableError) -> JSONResponse:
        logger.error("job store unavailable (job %s): %s", job_id or "-", exc)
        return JSONResponse({"detail": _STORE_UNAVAILABLE_DETAIL}, status_code=503)

    @app.post("/papers/jobs", status_code=202, openapi_extra=MULTIPART_OPENAPI_EXTRA)
    async def submit_job(  # pyright: ignore[reportUnusedFunction]
        request: Request,
    ):
        descriptor: dict[str, object] | None = None
        job: Job | None = None
        stored: StoredUpload | None = None
        duplicate_of: Job | None = None

        async def _discard_submission() -> None:
            if descriptor is not None:
                await tracker.discard(descriptor)
            elif stored is not None:
                await upload_store.remove(stored.upload_id)
            if job is not None:
                await store.discard(job.job_id)

        try:
            async with parse_multipart_request(request) as (upload, form_values):
                # Preserve the previous capacity ordering: Starlette has parsed
                # the admitted body, but do not make a second private copy when
                # the process-local job queue is already full.
                job = await store.create(filename=bounded_upload_filename(upload.filename))
                stored = await upload_store.persist(upload)
            # Upload is on disk; free the admission slot before queueing.
            release_spool_slot(request)
            descriptor = stored.to_descriptor(form_values)
            request_id = getattr(request.state, "request_id", None)
            if request_id is not None:
                # Link the worker-side extract record back to this request's
                # per-request metering record (serve-8).
                descriptor["request_id"] = request_id
            # The job id links the worker-side extract record to the job the
            # client polls; the handoff ignores unknown keys.
            descriptor["job_id"] = job.job_id
            dispatcher = _job_dispatcher()
            fingerprint: str | None = None
            if Settings.jobs.dedupe_inflight:
                fingerprint = upload_fingerprint(stored.sha256_hex, job.filename, form_values)
                duplicate_of = await _active_duplicate(dispatcher, fingerprint)
            if duplicate_of is not None:
                # The same file with the same options is already queued or running
                # here: hand back that job and drop this copy.
                logger.info("job %s: duplicate upload, answered with that job", duplicate_of.job_id)
                await _discard_submission()
            else:
                await dispatcher.submit(
                    job.job_id,
                    JobPayload(descriptor=descriptor),
                    fingerprint=fingerprint,
                )
        except EmptyUploadError:
            await _discard_submission()
            return JSONResponse({"detail": "Empty or missing file"}, status_code=400)
        except InvalidMultipartError as exc:
            await _discard_submission()
            return JSONResponse({"detail": str(exc)}, status_code=400)
        except InvalidUploadOptionError as exc:
            await _discard_submission()
            return JSONResponse({"detail": str(exc)}, status_code=400)
        except JobCapacityError as exc:
            await _discard_submission()
            return JSONResponse({"detail": str(exc)}, status_code=429)
        except JobStoreUnavailableError as exc:
            # The multipart spool is released by the context manager and nothing
            # was persisted (create runs before persist), so the upload is gone.
            await _discard_submission()
            return _store_unavailable(None, exc)
        except UploadTooLargeError:
            await _discard_submission()
            max_mib = upload_store.max_size / 1024 / 1024
            return JSONResponse({"detail": f"File too large (>{max_mib:.0f}MB)"}, status_code=413)
        except UploadStorageError:
            await _discard_submission()
            return JSONResponse({"detail": "Insufficient temporary storage"}, status_code=507)
        except BaseException:
            await _discard_submission()
            raise

        assert job is not None and descriptor is not None
        if duplicate_of is not None:
            return JSONResponse(
                {
                    "job_id": duplicate_of.job_id,
                    "status": duplicate_of.status,
                    "status_url": f"/papers/jobs/{duplicate_of.job_id}",
                    "duplicate": True,
                },
                status_code=202,
            )
        return JSONResponse(
            {
                "job_id": job.job_id,
                "status": "queued",
                "status_url": f"/papers/jobs/{job.job_id}",
            },
            status_code=202,
        )

    @app.get("/papers/jobs/{job_id}")
    async def job_status(job_id: str):  # pyright: ignore[reportUnusedFunction]
        try:
            job = await store.get(job_id, include_result=False)
        except JobStoreUnavailableError as exc:
            return _store_unavailable(job_id, exc)
        if job is None:
            return JSONResponse({"detail": "job not found"}, status_code=404)
        return JSONResponse(job.status_dict())

    @app.get("/papers/jobs/{job_id}/result")
    async def job_result(job_id: str):  # pyright: ignore[reportUnusedFunction]
        # No per-job ownership check (L2): bibr authenticates with a single shared
        # API key, so every authenticated caller is the same principal — there is no
        # "other user" to isolate against. The 128-bit random job_id is the boundary
        # that keeps results unguessable by unauthenticated callers.
        try:
            job = await store.get(job_id)
        except JobStoreUnavailableError as exc:
            return _store_unavailable(job_id, exc)
        if job is None:
            return JSONResponse({"detail": "job not found"}, status_code=404)
        if job.status in _ACTIVE_STATUSES:
            return JSONResponse(
                {"detail": "job not finished", "status": job.status},
                status_code=409,
            )
        if job.status == "failed":
            status = job.http_status or 500
            return JSONResponse(job.error or {"detail": "job failed"}, status_code=status)
        if job.result_json is None:
            # A shared store keeps the body under its own key: it can expire (or be
            # pruned) a moment before the status record does.
            return JSONResponse({"detail": "job result no longer available"}, status_code=404)
        # succeeded — the body was rendered once at completion.
        return Response(content=job.result_json, media_type="application/json")

    @app.delete("/papers/jobs/{job_id}")
    async def cancel_job(job_id: str):  # pyright: ignore[reportUnusedFunction]
        """Cancel a queued job: 200 with its (failed, ``job_cancelled``) status, also
        when it was already cancelled; 409 for a running or finished job."""
        try:
            job = await store.cancel(job_id)
        except JobStoreUnavailableError as exc:
            return _store_unavailable(job_id, exc)
        if job is None:
            return JSONResponse({"detail": "job not found"}, status_code=404)
        if is_cancelled(job):
            dispatcher = getattr(app.state, "job_dispatcher", None)
            if isinstance(dispatcher, JobDispatcher):
                await dispatcher.release_cancelled(job.job_id)
            return JSONResponse(job.status_dict())
        if job.status == "running":
            detail = "job is already running; only a queued job can be cancelled"
        else:
            detail = "job already finished"
        return JSONResponse({"detail": detail, "status": job.status}, status_code=409)
