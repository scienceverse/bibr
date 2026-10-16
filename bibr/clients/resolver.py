"""Async client for the optional bibr-resolver candidate-search service.

Thin wrapper over the resolver's ``/search`` and ``/works/{doi}`` endpoints. Every
call degrades gracefully — any HTTP/parse error returns an empty result so the
caller falls through to CrossRef; the resolver never fails enrichment.

A busy resolver is retried, not given up on: a 429/502/503/504 answer, a timeout
or a dropped connection is tried again after an exponential backoff. Every client
in the process that talks to the same resolver also shares one cap on requests in
flight, so a batch with many papers open at once can't flood it.
"""

import asyncio
import contextlib
import logging
import random
import threading
from urllib.parse import quote

import httpx

from bibr.utils.text import is_url_safe_doi

logger = logging.getLogger(__name__)

# What a busy or briefly unreachable resolver answers. bibr-resolver says 503
# ("resolver backend unavailable") when its search backend runs out of capacity;
# a proxy in front of it may say 502/504 or 429.
_RETRY_STATUSES = frozenset({429, 502, 503, 504})

# A request the resolver did not finish: a timeout (including waiting for a
# pooled connection) or a connection it dropped. A refused connection is left
# out on purpose: that resolver is down, not busy, and the health probe should
# say so at once rather than after the whole backoff.
_RETRY_ERRORS: tuple[type[Exception], ...] = (
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)

# No single backoff wait is longer than this, whatever Retry-After says.
_MAX_RETRY_DELAY = 30.0

# A timed-out request is retried once at most, within the retry budget. The
# resolver may still be running the search it gave up waiting for, so every
# retry of a timeout adds load to a resolver that is already behind, and four
# 30 s attempts would outlast the paper's whole enrichment budget.
_TIMEOUT_RETRIES = 1

# The health probe decides whether a paper uses the resolver at all, and it runs
# inside the paper's enrichment budget: a resolver that accepts the connection
# but never answers must cost seconds here, not the full search timeout.
_HEALTH_TIMEOUT = 10.0

# One in-flight cap per (event loop, resolver URL, limit). Papers each build
# their own ResolverClient, so a per-client cap would multiply by the papers in
# flight; this one is shared. An asyncio.Semaphore belongs to one loop, hence
# the loop in the key; gates of closed loops are dropped as new ones are made.
_GATES: dict[tuple[asyncio.AbstractEventLoop, str, int], asyncio.Semaphore] = {}
_GATES_LOCK = threading.Lock()


def _shared_gate(base_url: str, limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    key = (loop, base_url, limit)
    with _GATES_LOCK:
        gate = _GATES.get(key)
        if gate is None:
            for stale in [k for k in _GATES if k[0].is_closed()]:
                del _GATES[stale]
            gate = _GATES[key] = asyncio.Semaphore(limit)
    return gate


def _retry_after(headers: httpx.Headers) -> float | None:
    """A numeric ``Retry-After`` in seconds; the HTTP-date form and garbage are None."""
    try:
        secs = float(headers.get("retry-after", ""))
    except ValueError:
        return None
    return secs if secs >= 0 else None


class ResolverClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
        retries: int = 3,
        retry_backoff: float = 1.0,
        max_concurrency: int = 8,
    ) -> None:
        """``retries`` is how many times a busy answer, a timeout (once at most) or
        a dropped connection is tried again, after ``retry_backoff`` seconds
        doubled per attempt (with jitter, and never less than the resolver's
        Retry-After). ``max_concurrency`` caps requests in
        flight across every client in this process that uses the same resolver
        URL; 0 means no cap."""
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._retries = max(0, retries)
        self._retry_backoff = max(0.0, retry_backoff)
        self._max_concurrency = max(0, max_concurrency)
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            http2=True,
        )
        self._healthy: bool | None = None

    async def close(self) -> None:
        if not self._client.is_closed:
            await self._client.aclose()

    def _slot(self) -> "asyncio.Semaphore | contextlib.nullcontext[None]":
        if not self._max_concurrency:
            return contextlib.nullcontext()
        return _shared_gate(self.base_url, self._max_concurrency)

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        # Jittered so the requests a busy spell turned away don't all come back
        # in the same instant and make it busy again.
        delay = self._retry_backoff * (2**attempt) * random.uniform(0.5, 1.0)  # noqa: S311
        if retry_after is not None:
            delay = max(delay, retry_after)
        return min(delay, _MAX_RETRY_DELAY)

    async def _request(
        self, method: str, path: str, *, timeout_retries: int = _TIMEOUT_RETRIES, **kwargs
    ) -> httpx.Response:
        """Send one request, retrying a busy answer or an unfinished request.

        Returns the last response, whatever its status, for the caller to judge;
        raises the last transport error once the retries are spent. At most
        ``timeout_retries`` of the retries may follow a timeout. The backoff
        wait happens outside the in-flight cap, so a request that is waiting
        frees its slot for one that can run."""
        attempt = 0
        timeouts = 0
        while True:
            try:
                async with self._slot():
                    resp = await self._client.request(method, path, **kwargs)
            except _RETRY_ERRORS as e:
                if isinstance(e, httpx.TimeoutException):
                    timeouts += 1
                    if timeouts > timeout_retries:
                        raise
                if attempt >= self._retries:
                    raise
                reason: object = e
                delay = self._backoff(attempt, None)
            else:
                if resp.status_code not in _RETRY_STATUSES or attempt >= self._retries:
                    return resp
                reason = f"HTTP {resp.status_code}"
                delay = self._backoff(attempt, _retry_after(resp.headers))
            attempt += 1
            logger.debug(
                "resolver %s %s: %s; retry %d/%d in %.1fs",
                method,
                path[:80],
                reason,
                attempt,
                self._retries,
                delay,
            )
            await asyncio.sleep(delay)

    async def healthy(self) -> bool:
        """Probe GET /health once and cache the verdict for this client's lifetime.

        Returns True only when the service responds with status ``"ok"``. A
        degraded index, an HTTP/parse error, or an unreachable service all yield
        False so the caller can skip the resolver tier instead of paying a failed
        round-trip per reference.
        """
        if self._healthy is None:
            self._healthy = await self._probe_health()
        return self._healthy

    async def _probe_health(self) -> bool:
        try:
            resp = await self._request(
                "GET", "/health", timeout_retries=0, timeout=min(self._timeout, _HEALTH_TIMEOUT)
            )
            resp.raise_for_status()
            payload = resp.json()
            # A proxy or another service answering /health may return any JSON
            # (``["ok"]``, ``"ok"``); only an object with status "ok" is healthy.
            return isinstance(payload, dict) and payload.get("status") == "ok"
        except Exception as e:  # noqa: BLE001 — the probe's contract is a verdict, never an error
            logger.debug("resolver health probe failed: %s", e)
            return False

    async def search(
        self,
        title: str,
        year: int | None,
        limit: int,
        *,
        sources: list[str] | None = None,
        raise_on_error: bool = False,
    ) -> list[dict]:
        """POST /search → list of candidate dicts. Returns [] on any error.

        ``sources`` selects which resolver corpora to query (e.g.
        ``["crossref", "openalex"]``); omit it (or pass an empty list) to let the resolver
        use its own default tier. The resolver ranks CrossRef first when it has the record.

        With ``raise_on_error=True`` a transport/parse error propagates instead of
        degrading to [], so an authoritative caller can tell a real error apart from a
        genuinely empty result (and still fall through to CrossRef on the former)."""
        body: dict = {"title": title, "year": year, "limit": limit}
        if sources:
            body["sources"] = list(sources)
        try:
            resp = await self._request("POST", "/search", json=body)
            resp.raise_for_status()
            payload = resp.json()
            # A body of ``{"candidates": null}`` makes ``.get(..., [])`` return
            # None, not the default — and this method is documented to return a
            # list. One such response used to poison the whole fallback batch
            # downstream. Anything that isn't a list is treated as no results,
            # and an entry that isn't an object is no candidate.
            candidates = payload.get("candidates") if isinstance(payload, dict) else None
            if not isinstance(candidates, list):
                return []
            return [c for c in candidates if isinstance(c, dict)]
        except (httpx.HTTPError, ValueError) as e:
            if raise_on_error:
                raise
            logger.debug("resolver search failed for %r: %s", title[:50], e)
            return []

    async def search_many(
        self,
        queries: list[dict],
        *,
        concurrency: int = 4,
        sources: list[str] | None = None,
        raise_on_error: bool = False,
    ) -> list[list[dict] | Exception]:
        """Resolve many ``{"title", "year", "limit"}`` queries concurrently, results
        aligned 1:1 with ``queries``. ``concurrency`` bounds how many /search calls run at
        once so a large reference list can't flood the resolver; each /search already
        degrades to [] on error, so one bad query never sinks the rest. ``sources`` is
        applied to every query (see :meth:`search`).

        With ``raise_on_error=True`` a per-query error is captured as an ``Exception``
        object in that slot (never raised out of the batch), so the caller can treat one
        slot as an error while the other slots keep their real results."""
        if not queries:
            return []
        sem = asyncio.Semaphore(concurrency)

        async def one(q: dict) -> list[dict]:
            async with sem:
                return await self.search(
                    q.get("title", ""),
                    q.get("year"),
                    q.get("limit", 20),
                    sources=sources,
                    raise_on_error=raise_on_error,
                )

        return list(await asyncio.gather(*(one(q) for q in queries), return_exceptions=True))

    async def lookup_doi(self, doi: str, *, raise_on_error: bool = False) -> dict | None:
        """GET /works/{doi} → candidate dict, or None on 404 / error.

        A 404 is a genuine not-found and always returns None. With
        ``raise_on_error=True`` any *other* transport/parse error propagates instead of
        degrading to None, so an authoritative caller can tell a clean miss (404) apart
        from a transient error.

        A DOI that is not URL-path-safe (path traversal / SSRF; see
        :func:`is_url_safe_doi`) is a bad input, not a transient error, so it is a
        clean miss (None) even under ``raise_on_error`` — the caller falls through
        to CrossRef."""
        if not is_url_safe_doi(doi):
            logger.debug("resolver lookup_doi rejected unsafe DOI: %r", doi)
            return None
        path = quote(doi, safe="/")
        try:
            resp = await self._request("GET", f"/works/{path}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                raise ValueError(f"resolver /works answered {type(payload).__name__}, not a work")
            return payload
        except (httpx.HTTPError, ValueError) as e:
            if raise_on_error:
                raise
            logger.debug("resolver lookup_doi failed for %s: %s", doi, e)
            return None
