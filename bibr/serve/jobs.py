"""Async job API for the bibr serve layer (D1).

Submitting a large PDF and holding an HTTP connection open for the full
extraction (tens of seconds) is fragile behind proxies/load balancers. These
routes offer a fire-and-poll alternative:

  - ``POST /papers/jobs``            → 202 ``{job_id, status, status_url}``
  - ``GET  /papers/jobs/{id}``       → job status (no result body)
  - ``GET  /papers/jobs/{id}/result`` → the paper_json once succeeded

**Architecture.** Jobs persist their uploads through the same disk-backed
ingress as ``/papers/extract`` and queue only its opaque descriptor. The job
runner submits that descriptor through LitServe's private inference adapter,
which owns worker dispatch and upload cleanup without rebuilding a multipart
request or forwarding caller credentials.

Because every custom route runs in the *one* API-server process, the in-process
:class:`JobStore` is consistent without Redis. This holds ONLY with a single API
server. LitServe 0.2.17 defaults ``num_api_servers`` to the inference-worker
count, so ``serve.app.main`` unconditionally pins one API server for the shared
upload root and process-local request state. If you deploy multiple API servers,
jobs and upload ownership must first move to shared external stores.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

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


class JobCapacityError(Exception):
    """Raised by :meth:`JobStore.create` when the active-job cap is reached."""


def _iso(wall: float | None) -> str | None:
    """UTC ISO-8601 for a wall-clock epoch seconds value (None passes through)."""
    if wall is None:
        return None
    return datetime.fromtimestamp(wall, tz=UTC).isoformat()


@dataclass
class Job:
    """A single async extraction job.

    Monotonic timestamps drive durations and TTL (immune to wall-clock jumps);
    wall timestamps drive the ISO fields surfaced to clients.
    """

    job_id: str
    filename: str
    status: str = "queued"  # queued | running | succeeded | failed
    created_mono: float = 0.0
    created_wall: float = 0.0
    started_mono: float | None = None
    started_wall: float | None = None
    finished_mono: float | None = None
    finished_wall: float | None = None
    http_status: int | None = None
    error: dict | None = None  # {kind?, detail}
    result: dict | None = None  # the paper_json
    task: object | None = field(default=None, repr=False)  # asyncio.Task ref

    @property
    def duration_ms(self) -> int | None:
        if self.finished_mono is None:
            return None
        start = self.started_mono if self.started_mono is not None else self.created_mono
        return int((self.finished_mono - start) * 1000)

    def status_dict(self) -> dict:
        """Status view — everything except the (potentially large) result body."""
        out: dict = {
            "job_id": self.job_id,
            "status": self.status,
            "filename": self.filename,
            "created_at": _iso(self.created_wall),
            "started_at": _iso(self.started_wall),
            "finished_at": _iso(self.finished_wall),
        }
        if self.finished_mono is not None:
            out["duration_ms"] = self.duration_ms
        if self.status == "failed" and self.error is not None:
            out["error"] = self.error
        if self.status == "succeeded":
            out["result_url"] = f"/papers/jobs/{self.job_id}/result"
        return out


class JobStore:
    """In-process job store: a dict guarded by an ``asyncio.Lock``.

    Finished jobs are purged lazily (on every access) once older than
    ``Settings.jobs.ttl_seconds``. The active-job cap (``Settings.jobs.max_active``)
    bounds queued+running jobs so an upload flood can't grow the store without
    limit. ``clock`` / ``wall_clock`` are injectable for deterministic tests.
    """

    def __init__(
        self,
        *,
        clock=time.monotonic,
        wall_clock=time.time,
    ) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._clock = clock
        self._wall_clock = wall_clock

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
        finished = sorted(
            (job for job in self._jobs.values() if job.finished_mono is not None),
            key=lambda job: job.finished_mono or 0.0,
        )
        for job in finished[: max(0, len(finished) - max_retained)]:
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
            )
            self._jobs[job.job_id] = job
            return job

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            self._purge_locked()
            return self._jobs.get(job_id)

    async def discard(self, job_id: str) -> None:
        """Remove a job that failed before it could be admitted to dispatch."""
        async with self._lock:
            self._jobs.pop(job_id, None)

    async def set_running(self, job_id: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "running"
            job.started_mono = self._clock()
            job.started_wall = self._wall_clock()

    async def set_succeeded(self, job_id: str, result: dict) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "succeeded"
            job.result = result
            job.finished_mono = self._clock()
            job.finished_wall = self._wall_clock()
            self._purge_locked()

    async def set_failed(self, job_id: str, *, http_status: int | None, error: dict) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "failed"
            job.http_status = http_status
            job.error = error
            job.finished_mono = self._clock()
            job.finished_wall = self._wall_clock()
            self._purge_locked()


_REQUEST_ID_STRIP = re.compile(r"[^-a-zA-Z0-9_]")


def _sanitize_request_id(raw: str | None) -> str | None:
    """Keep only ``[-a-zA-Z0-9_]``, cap at 64 chars. Empty → None."""
    if not raw:
        return None
    cleaned = _REQUEST_ID_STRIP.sub("", raw)[:64]
    return cleaned or None


@dataclass(frozen=True)
class JobPayload:
    """Small queue descriptor for one disk-backed job upload."""

    descriptor: dict[str, object]


class JobDispatcher:
    """Process-local FIFO dispatcher with bounded descriptor dispatch concurrency."""

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

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("job dispatcher is closed")
        if not self._workers:
            self._workers = [
                asyncio.create_task(self._worker(), name=f"bibr-job-worker-{index}")
                for index in range(self._max_running)
            ]

    async def submit(self, job_id: str, payload: JobPayload) -> None:
        await self.start()
        self._queue.put_nowait((job_id, payload))

    async def join(self) -> None:
        await self._queue.join()

    async def _worker(self) -> None:
        while True:
            job_id, payload = await self._queue.get()
            try:
                await _run_job(
                    store=self._store,
                    job_id=job_id,
                    descriptor=payload.descriptor,
                    tracker=self._tracker,
                )
            finally:
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
                _, payload = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                await self._tracker.discard(payload.descriptor)
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

    await store.set_running(job_id)
    try:
        result = await tracker.submit(descriptor)
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
        await store.set_succeeded(job_id, result)


def register_job_routes(
    app: FastAPI,
    *,
    store: JobStore,
    upload_store: UploadStore,
    tracker: InferenceDispatchTracker,
) -> None:
    """Mount the three async-job routes on ``app``.

    Routes are NOT added to ``PUBLIC_PATHS`` — the app-wide ``_auth_gate``
    middleware covers them like any other route. Jobs share the public
    extraction route's disk-backed upload store and LitServe dispatch tracker.
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

    @app.post("/papers/jobs", status_code=202, openapi_extra=MULTIPART_OPENAPI_EXTRA)
    async def submit_job(  # pyright: ignore[reportUnusedFunction]
        request: Request,
    ):
        descriptor: dict[str, object] | None = None
        job: Job | None = None
        stored: StoredUpload | None = None

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
            await _job_dispatcher().submit(
                job.job_id,
                JobPayload(descriptor=descriptor),
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
        job = await store.get(job_id)
        if job is None:
            return JSONResponse({"detail": "job not found"}, status_code=404)
        return JSONResponse(job.status_dict())

    @app.get("/papers/jobs/{job_id}/result")
    async def job_result(job_id: str):  # pyright: ignore[reportUnusedFunction]
        # No per-job ownership check (L2): bibr authenticates with a single shared
        # API key, so every authenticated caller is the same principal — there is no
        # "other user" to isolate against. The 128-bit random job_id is the boundary
        # that keeps results unguessable by unauthenticated callers.
        job = await store.get(job_id)
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
        # succeeded
        return JSONResponse(job.result)
