"""Remote executor: drive papers through a ``bibr serve`` async job API.

Protocol (``bibr/serve/jobs.py``): ``POST /papers/jobs`` (multipart ``file``
plus optional form options) → 202 ``{job_id, status_url}``; poll
``GET /papers/jobs/{id}`` until ``status`` is ``succeeded`` / ``failed``;
``GET /papers/jobs/{id}/result`` returns the export (or the failure with its
HTTP status). ``GET /ready`` gates the run and, when authenticated, reveals
the deployment's ``build_sha``.

Backpressure and failure policy (mirrors the bibr-training campaign script):

* **429** on submit is the serve's queue cap (``JOBS_MAX_ACTIVE``) — normal
  under load. In-flight drops to ``--min-concurrency`` and the submit waits
  ``Retry-After`` (or a capped backoff); it never counts as a failure.
* **5xx / connection errors / an upstream OCR-LLM outage** reported by a
  failed job are *transient*: in-flight shrinks by one, the paper is retried
  with backoff up to ``--retries`` times, and the last transient code is
  recorded if it never recovers.
* **4xx** is a property of the request (rejected upload, bad option): failed
  at once, no retry. 401/403 stops the whole run — every paper would fail.
* Every success grows in-flight by one, back toward ``--max-concurrency``.
* Ctrl-C stops submitting, waits ``grace`` seconds for in-flight jobs, then
  records the rest as ``failed`` / ``interrupted`` (re-run by default).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from bibr.batch.ledger import INTERRUPTED, Outcome, bounded_text, utc_now_iso
from bibr.batch.manifest import BatchItem, sha256_file

logger = logging.getLogger(__name__)

TRANSIENT_HTTP = frozenset({502, 503, 504})
AUTH_HTTP = frozenset({401, 403})
TERMINAL = frozenset({"succeeded", "failed"})
# Text of a failed job that describes the serve's dependencies, not the paper.
TRANSIENT_MARKERS = (
    "circuit breaker",
    "error in ocr",
    "ocr server",
    "ocr service",
    "llm server",
    "unreachable",
    "upstream",
    "timed out",
    "connection",
    "temporarily unavailable",
)
MAX_POLL_ERRORS = 5
MAX_POLL_INTERVAL = 15.0
MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xml": "application/xml",
    ".html": "text/html",
    ".htm": "text/html",
    ".epub": "application/epub+zip",
}
TOKEN_ENV_VARS = ("AUTH_API_KEY", "BIBR_SERVE_TOKEN")


class TransientError(Exception):
    """A failure that says nothing about the paper — retry after a pause."""

    def __init__(self, code: str, detail: str, *, shrink: bool = True, job_id: str | None = None):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.shrink = shrink
        self.job_id = job_id


class PermanentError(Exception):
    """A failure that retrying would only repeat."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        http_status: int | None = None,
        failed_stage: str | None = None,
        job_id: str | None = None,
    ):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.http_status = http_status
        self.failed_stage = failed_stage
        self.job_id = job_id


class RemoteAuthError(Exception):
    """The serve rejected the bearer token — nothing else can succeed."""


def configured_value(env_name: str) -> str | None:
    """A setting's raw effective value: process env, then bibr's ``.env`` chain.

    The same precedence ``bibr config show`` reports (env > ``./.env`` >
    ``~/.bibr/.env``), honouring ``BIBR_DISABLE_DOTENV``, without touching the
    process-global ``Settings`` — the batch package is a library, not a
    process boundary.
    """
    from bibr.config import dotenv_disabled
    from bibr.config_cli import resolve_provenance
    from bibr.config_introspect import iter_setting_docs

    if dotenv_disabled():
        value = os.environ.get(env_name, "")
        return value.strip() or None
    for doc in iter_setting_docs():
        if doc.env_name == env_name:
            value = resolve_provenance(doc).value
            return (value.strip() or None) if value else None
    return None


def resolve_token(explicit: str | None = None) -> str | None:
    """``--token``, else ``AUTH_API_KEY`` / ``BIBR_SERVE_TOKEN``, else the ``.env`` setting."""
    if explicit:
        return explicit
    for name in TOKEN_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        return configured_value("AUTH_API_KEY")
    except Exception:  # noqa: BLE001 — a broken .env is reported elsewhere; no token here
        return None


def looks_transient(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in TRANSIENT_MARKERS)


@dataclass
class RemoteOptions:
    serve_url: str
    token: str | None = None
    concurrency: int = 2
    min_concurrency: int = 1
    max_concurrency: int = 4
    poll_timeout: float = 2400.0
    poll_interval: float = 4.0
    retries: int = 3
    ready_timeout: float = 900.0
    ready_interval: float = 15.0
    submit_wait_budget: float = 3600.0
    grace: float = 30.0
    form: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.serve_url = self.serve_url.rstrip("/")
        self.min_concurrency = max(1, self.min_concurrency)
        self.max_concurrency = max(self.min_concurrency, self.max_concurrency)
        self.concurrency = min(max(self.concurrency, self.min_concurrency), self.max_concurrency)
        self.retries = max(0, self.retries)


class AdaptiveGate:
    """In-flight budget that shrinks on backpressure and grows on success."""

    def __init__(self, initial: int, lo: int, hi: int):
        self.lo = max(1, lo)
        self.hi = max(self.lo, hi)
        self._size = min(max(initial, self.lo), self.hi)

    @property
    def size(self) -> int:
        return self._size

    def shrink(self) -> None:
        self._size = max(self.lo, self._size - 1)

    def shrink_to_min(self) -> None:
        self._size = self.lo

    def grow(self) -> None:
        self._size = min(self.hi, self._size + 1)


OutcomeCallback = Callable[[BatchItem, Outcome], None]
SleepFn = Callable[[float], Awaitable[None]]


class RemoteExecutor:
    """Submit/poll/fetch papers against one serve with adaptive concurrency.

    ``transport``, ``sleep``, ``clock`` and ``wall`` are injectable so tests
    run against an ``httpx.MockTransport`` with a virtual clock.
    """

    def __init__(
        self,
        options: RemoteOptions,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: SleepFn | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        rng: random.Random | None = None,
    ):
        self.options = options
        self.gate = AdaptiveGate(
            options.concurrency, options.min_concurrency, options.max_concurrency
        )
        self._transport = transport
        self._sleep: SleepFn = sleep or asyncio.sleep
        self._clock = clock
        self._wall = wall
        self._rng = rng or random.Random()  # noqa: S311 — jitter only, not security
        self.ready: dict[str, Any] | None = None
        self.stats: dict[str, int] = {"submit_429": 0, "transient_retries": 0}
        self.fatal: str | None = None

    # -- HTTP plumbing ---------------------------------------------------

    def client(self) -> httpx.AsyncClient:
        headers = {}
        if self.options.token:
            headers["Authorization"] = f"Bearer {self.options.token}"
        return httpx.AsyncClient(
            base_url=self.options.serve_url,
            headers=headers,
            transport=self._transport,
            timeout=httpx.Timeout(60.0, read=600.0),
            follow_redirects=True,
        )

    async def wait_ready(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """Block until ``GET /ready`` says ``ready`` (or ``ready_timeout`` passes)."""
        waited = 0.0
        first = True
        while True:
            body = await self._probe_ready(client)
            if body.get("status") == "ready":
                if not first:
                    # The serve's OCR breaker also counts failed readiness
                    # probes; give it a moment before the first submit.
                    await self._sleep(2.0)
                self.ready = body
                return body
            if body.get("http_status") in AUTH_HTTP:
                raise RemoteAuthError(f"serve rejected the token on /ready ({body})")
            if waited >= self.options.ready_timeout:
                raise TimeoutError(f"serve never became ready: {body}")
            logger.warning("serve not ready (%s); waiting", json.dumps(body, default=str))
            first = False
            await self._sleep(self.options.ready_interval)
            waited += self.options.ready_interval

    async def _probe_ready(self, client: httpx.AsyncClient) -> dict[str, Any]:
        try:
            response = await client.get("/ready", timeout=30.0)
        except httpx.HTTPError as exc:
            return {"status": "unreachable", "error": f"{type(exc).__name__}: {exc}"[:200]}
        try:
            body = response.json()
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        body.setdefault("status", "unknown")
        body["http_status"] = response.status_code
        return body

    # -- one paper -------------------------------------------------------

    async def process(self, client: httpx.AsyncClient, item: BatchItem) -> Outcome:
        """Run one paper to a terminal :class:`Outcome`; never raises for it."""
        started_at = utc_now_iso()
        t0 = self._clock()
        retries_used = 0
        last_transient: TransientError | None = None
        try:
            sha256 = sha256_file(item.path)
            data = item.path.read_bytes()
        except OSError as exc:
            return self._finish(
                Outcome("failed", error_code="unreadable_input", error=str(exc)),
                started_at,
                t0,
                sha256=None,
                size=None,
                retries=0,
            )
        size = len(data)

        for attempt in range(self.options.retries + 1):
            try:
                outcome = await self._attempt(client, item, data)
            except TransientError as exc:
                last_transient = exc
                self.stats["transient_retries"] += 1
                if exc.shrink:
                    self.gate.shrink()
                if attempt >= self.options.retries:
                    break
                retries_used += 1
                logger.warning(
                    "%s: %s (%s); retry %d/%d",
                    item.paper_id,
                    exc.code,
                    exc.detail[:120],
                    retries_used,
                    self.options.retries,
                )
                await self._sleep(self._backoff(attempt))
                continue
            except PermanentError as exc:
                outcome = Outcome(
                    "failed",
                    error_code=exc.code,
                    failed_stage=exc.failed_stage,
                    error=exc.detail,
                    extra={"http_status": exc.http_status, "job_id": exc.job_id},
                )
                return self._finish(
                    outcome, started_at, t0, sha256=sha256, size=size, retries=retries_used
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — one paper must never take the run down
                logger.exception("%s: unexpected client error", item.paper_id)
                outcome = Outcome(
                    "failed", error_code="client_error", error=f"{type(exc).__name__}: {exc}"
                )
                return self._finish(
                    outcome, started_at, t0, sha256=sha256, size=size, retries=retries_used
                )
            else:
                if outcome.ok:
                    self.gate.grow()
                return self._finish(
                    outcome, started_at, t0, sha256=sha256, size=size, retries=retries_used
                )

        assert last_transient is not None  # noqa: S101 — loop only exits via a transient
        outcome = Outcome(
            "failed",
            error_code=last_transient.code,
            error=last_transient.detail,
            extra={"job_id": last_transient.job_id, "transient_exhausted": True},
        )
        return self._finish(outcome, started_at, t0, sha256=sha256, size=size, retries=retries_used)

    def _finish(
        self,
        outcome: Outcome,
        started_at: str,
        t0: float,
        *,
        sha256: str | None,
        size: int | None,
        retries: int,
    ) -> Outcome:
        outcome.started_at = started_at
        outcome.finished_at = utc_now_iso()
        outcome.duration_s = max(self._clock() - t0, 0.0)
        outcome.sha256 = sha256
        outcome.size = size
        outcome.extra.setdefault("retries", retries)
        return outcome

    def _backoff(self, attempt: int) -> float:
        base = min(60.0, 5.0 * (2**attempt))
        return base + self._rng.uniform(0.0, 2.0)

    async def _attempt(self, client: httpx.AsyncClient, item: BatchItem, data: bytes) -> Outcome:
        job_id = await self._submit(client, item, data)
        body = await self._poll(client, job_id)
        if body.get("status") == "succeeded":
            export = await self._fetch_result(client, job_id)
            return Outcome("ok", export=export, extra={"job_id": job_id})

        error = body.get("error")
        error = error if isinstance(error, Mapping) else {"detail": str(error or "job failed")}
        message = str(error.get("message") or error.get("detail") or json.dumps(error))
        code = error.get("error_code")
        http_status: int | None = None
        text = ""
        try:
            result = await client.get(f"/papers/jobs/{job_id}/result")
            http_status = result.status_code
            text = result.text
        except httpx.HTTPError:
            pass
        if http_status in TRANSIENT_HTTP or looks_transient(f"{message} {text}"):
            raise TransientError("upstream_unavailable", message, job_id=job_id)
        raise PermanentError(
            str(code) if code else f"job_failed_{http_status or 'unknown'}",
            message,
            http_status=http_status,
            failed_stage=error.get("failed_stage"),
            job_id=job_id,
        )

    async def _submit(self, client: httpx.AsyncClient, item: BatchItem, data: bytes) -> str:
        budget_end = self._clock() + self.options.submit_wait_budget
        waits = 0
        mime = MIME_TYPES.get(item.path.suffix.lower(), "application/octet-stream")
        while True:
            try:
                response = await client.post(
                    "/papers/jobs",
                    files={"file": (item.path.name, data, mime)},
                    data=self.options.form,
                    timeout=httpx.Timeout(180.0),
                )
            except httpx.TransportError as exc:
                raise TransientError("connection_error", f"{type(exc).__name__}: {exc}") from exc
            status = response.status_code
            if status == 202:
                try:
                    job_id = response.json().get("job_id")
                except ValueError:
                    job_id = None
                if not job_id:
                    raise PermanentError(
                        "bad_submit_response",
                        bounded_text(response.text, 300) or "",
                        http_status=202,
                    )
                return str(job_id)
            if status == 429:
                self.stats["submit_429"] += 1
                self.gate.shrink_to_min()
                if self._clock() >= budget_end:
                    raise TransientError(
                        "submit_wait_exhausted",
                        f"queue full (429) for {self.options.submit_wait_budget:.0f}s",
                        shrink=False,
                    )
                wait = _retry_after(response) or min(60.0, 5.0 * (2 ** min(waits, 4)))
                waits += 1
                await self._sleep(wait + self._rng.uniform(0.0, 2.0))
                continue
            if status >= 500:
                raise TransientError(f"http_{status}", bounded_text(response.text, 300) or "")
            raise PermanentError(
                f"http_{status}", bounded_text(response.text, 500) or "", http_status=status
            )

    async def _poll(self, client: httpx.AsyncClient, job_id: str) -> dict[str, Any]:
        deadline = self._clock() + self.options.poll_timeout
        interval = self.options.poll_interval
        errors = 0
        while self._clock() < deadline:
            await self._sleep(interval)
            try:
                response = await client.get(f"/papers/jobs/{job_id}", timeout=60.0)
            except httpx.TransportError as exc:
                errors += 1
                if errors > MAX_POLL_ERRORS:
                    raise TransientError(
                        "connection_error", f"{type(exc).__name__}: {exc}", job_id=job_id
                    ) from exc
                continue
            status = response.status_code
            if status == 404:
                raise TransientError("job_lost", "job vanished from the serve", shrink=False)
            if status >= 500:
                errors += 1
                if errors > MAX_POLL_ERRORS:
                    raise TransientError(
                        f"http_{status}", bounded_text(response.text, 300) or "", job_id=job_id
                    )
                continue
            if status >= 400:
                raise PermanentError(
                    f"http_{status}",
                    bounded_text(response.text, 500) or "",
                    http_status=status,
                    job_id=job_id,
                )
            errors = 0
            try:
                body = response.json()
            except ValueError:
                continue
            if isinstance(body, dict) and body.get("status") in TERMINAL:
                return body
            interval = min(interval * 1.5, max(self.options.poll_interval, MAX_POLL_INTERVAL))
        raise PermanentError(
            "poll_timeout",
            f"job not finished after {self.options.poll_timeout:.0f}s",
            job_id=job_id,
        )

    async def _fetch_result(self, client: httpx.AsyncClient, job_id: str) -> dict[str, Any]:
        try:
            response = await client.get(f"/papers/jobs/{job_id}/result", timeout=600.0)
        except httpx.TransportError as exc:
            raise TransientError(
                "connection_error", f"{type(exc).__name__}: {exc}", job_id=job_id
            ) from exc
        status = response.status_code
        if status >= 500 or status == 409:
            raise TransientError(
                f"http_{status}", bounded_text(response.text, 300) or "", job_id=job_id
            )
        if status != 200:
            raise PermanentError(
                f"http_{status}",
                bounded_text(response.text, 500) or "",
                http_status=status,
                job_id=job_id,
            )
        try:
            export = response.json()
        except ValueError:
            raise PermanentError("bad_result_json", "result is not JSON", job_id=job_id) from None
        if not isinstance(export, dict):
            raise PermanentError("bad_result_json", "result is not a JSON object", job_id=job_id)
        return export

    # -- the run ---------------------------------------------------------

    async def run(
        self,
        items: Sequence[BatchItem],
        *,
        on_outcome: OutcomeCallback,
        stop: asyncio.Event | None = None,
        deadline: float | None = None,
        on_ready: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        """Process *items*; returns ``completed`` | ``deadline`` | ``stopped``.

        ``stop`` halts submission (Ctrl-C); in-flight jobs get ``grace``
        seconds, then are recorded as interrupted. ``deadline`` (epoch
        seconds) only stops *submission* — in-flight jobs finish normally.
        """
        stop = stop or asyncio.Event()
        async with self.client() as client:
            ready = await self.wait_ready(client)
            if on_ready is not None:
                on_ready(ready)
            pending = list(items)
            tasks: dict[asyncio.Task[Outcome], BatchItem] = {}
            stop_waiter = asyncio.ensure_future(stop.wait())
            reason = "completed"
            try:
                while pending or tasks:
                    if pending and deadline is not None and self._wall() >= deadline:
                        reason = "deadline"
                        pending.clear()
                    while pending and not stop.is_set() and len(tasks) < self.gate.size:
                        item = pending.pop(0)
                        task = asyncio.create_task(
                            self.process(client, item), name=f"bibr-batch-{item.paper_id}"
                        )
                        tasks[task] = item
                    if not tasks:
                        if stop.is_set() and pending:
                            reason = "stopped"
                        break
                    done, _ = await asyncio.wait(
                        {*tasks, stop_waiter}, return_when=asyncio.FIRST_COMPLETED
                    )
                    for finished in done:
                        if finished is stop_waiter:
                            continue
                        item = tasks.pop(finished)  # type: ignore[arg-type]
                        outcome = self._outcome_of(finished)  # type: ignore[arg-type]
                        on_outcome(item, outcome)
                        if outcome.extra.get("http_status") in AUTH_HTTP:
                            self.fatal = f"serve rejected the token ({outcome.error_code})"
                            stop.set()
                    if stop.is_set() and tasks:
                        await self._drain(tasks, on_outcome)
                    if stop.is_set():
                        reason = "stopped"
                        break
            except asyncio.CancelledError:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                for item in tasks.values():
                    on_outcome(item, _interrupted())
                raise
            finally:
                stop_waiter.cancel()
            return reason

    async def _drain(
        self, tasks: dict[asyncio.Task[Outcome], BatchItem], on_outcome: OutcomeCallback
    ) -> None:
        """Give in-flight jobs ``grace`` seconds, then cancel and record the rest."""
        done, still = await asyncio.wait(set(tasks), timeout=self.options.grace)
        for finished in done:
            on_outcome(tasks.pop(finished), self._outcome_of(finished))
        for task in still:
            task.cancel()
        if still:
            await asyncio.gather(*still, return_exceptions=True)
        for task in still:
            on_outcome(tasks.pop(task), _interrupted())

    @staticmethod
    def _outcome_of(task: asyncio.Task[Outcome]) -> Outcome:
        if task.cancelled():
            return _interrupted()
        exc = task.exception()
        if exc is not None:
            return Outcome(
                "failed",
                error_code="client_error",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=utc_now_iso(),
            )
        return task.result()


def _interrupted() -> Outcome:
    return Outcome(
        "failed",
        error_code=INTERRUPTED,
        error="interrupted before the job finished",
        finished_at=utc_now_iso(),
    )


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return min(60.0, max(1.0, float(raw)))
    except ValueError:
        return None
