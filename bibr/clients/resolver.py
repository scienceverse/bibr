"""Async client for the optional bibr-resolver candidate-search service.

Thin wrapper over the resolver's ``/search`` and ``/works/{doi}`` endpoints. Every
call degrades gracefully — any HTTP/parse error returns an empty result so the
caller falls through to CrossRef; the resolver never fails enrichment.
"""

import asyncio
import logging
from urllib.parse import quote

import httpx

from bibr.utils.text import is_url_safe_doi

logger = logging.getLogger(__name__)


class ResolverClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            http2=True,
        )
        self._healthy: bool | None = None

    async def close(self) -> None:
        if not self._client.is_closed:
            await self._client.aclose()

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
            resp = await self._client.get("/health")
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
            resp = await self._client.post("/search", json=body)
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
        concurrency: int = 8,
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
            resp = await self._client.get(f"/works/{path}")
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
