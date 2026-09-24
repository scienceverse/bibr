"""Async client for ROR affiliation matching.

One ``GET {url}/organizations?affiliation=<text>&single_search`` per string.
ROR marks at most one result ``chosen`` when it is confident in the match and
advises against selecting by score, so only the chosen result is taken; a
string without one stays unmatched.

ROR publishes a rate limit of 50 requests per 5 minutes without a client ID
and 2000 with one (sent as the ``Client-Id`` header). The client keeps to it
with a sliding window, remembers answers (misses too) in a process-wide
cache, and backs off entirely after a 429 for the ``Retry-After`` interval.
Every failure degrades to "no match": ROR never fails enrichment.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import httpx

from bibr.enrich.schemas import canonical_ror
from bibr.models import OrganizationMatch
from bibr.utils.rate_limiter import AsyncLocalRateLimiter

if TYPE_CHECKING:
    from bibr.config import GlobalSettings

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 300.0
_ANONYMOUS_REQUESTS_PER_WINDOW = 50
_CLIENT_ID_REQUESTS_PER_WINDOW = 2000
# Strings outside these bounds are not affiliations worth a request.
_MIN_CHARS = 3
_MAX_CHARS = 500


def _cache_key(text: str) -> str:
    return " ".join(text.split()).casefold()


class RorClient:
    """ROR affiliation matching with rate limiting and an answer cache."""

    def __init__(
        self,
        *,
        settings: GlobalSettings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        ror = settings.ror
        self._base_url = ror.url.rstrip("/")
        budget = ror.requests_per_5min or (
            _CLIENT_ID_REQUESTS_PER_WINDOW if ror.client_id else _ANONYMOUS_REQUESTS_PER_WINDOW
        )
        self._limiter = AsyncLocalRateLimiter("ror", budget, _WINDOW_SECONDS)
        mailto = settings.crossref.api_email
        headers = {"User-Agent": f"bibr (mailto:{mailto})" if mailto else "bibr"}
        if ror.client_id:
            headers["Client-Id"] = ror.client_id
        self._headers = headers
        self._timeout = ror.request_timeout
        self._client = client
        self._owns_client = client is None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._cache_size = ror.cache_size
        self._cache: OrderedDict[str, OrganizationMatch | None] = OrderedDict()
        self._blocked_until = 0.0

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        # The shared client outlives one ``asyncio.run``; an httpx pool is
        # bound to the loop that opened it, so an owned client is rebuilt when
        # the loop changes.
        loop = asyncio.get_running_loop()
        if self._owns_client and (self._client is None or self._client_loop is not loop):
            self._client = httpx.AsyncClient(
                headers=self._headers, timeout=httpx.Timeout(self._timeout)
            )
            self._client_loop = loop
        assert self._client is not None
        return self._client

    def _remember(self, key: str, value: OrganizationMatch | None) -> None:
        if self._cache_size <= 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    @property
    def blocked(self) -> bool:
        """True while backing off after a 429."""
        return time.monotonic() < self._blocked_until

    async def match(self, text: str) -> OrganizationMatch | None:
        """ROR's chosen organization for *text*, or ``None``."""
        text = " ".join(text.split())
        if not _MIN_CHARS <= len(text) <= _MAX_CHARS:
            return None
        key = _cache_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if self.blocked:
            return None
        await self._limiter.acquire()
        try:
            response = await self._http().get(
                f"{self._base_url}/organizations",
                params={"affiliation": text, "single_search": ""},
                headers=self._headers,
            )
        except httpx.HTTPError as exc:
            logger.debug("ROR request failed for %r: %s", text[:80], exc)
            return None
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.isdigit() else _WINDOW_SECONDS
            self._blocked_until = time.monotonic() + delay
            logger.warning("ROR rate limit reached; skipping ROR lookups for %.0fs", delay)
            return None
        if response.status_code != 200:
            logger.debug("ROR returned HTTP %d for %r", response.status_code, text[:80])
            return None
        try:
            result = chosen_organization(response.json())
        except ValueError:
            logger.debug("ROR returned unparseable JSON for %r", text[:80])
            return None
        self._remember(key, result)
        return result


def chosen_organization(payload: Any) -> OrganizationMatch | None:
    """The ``chosen`` result of a ROR v2 affiliation response, or ``None``."""
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict) or item.get("chosen") is not True:
            continue
        org = item.get("organization") or {}
        ror_id = canonical_ror(org.get("id"))
        if ror_id is None:
            return None
        score = item.get("score")
        return OrganizationMatch(
            service_id=ror_id,
            score=float(score) if isinstance(score, (int, float)) else None,
            name=_display_name(org),
            country_code=_country_code(org),
            funder_doi=_funder_doi(org),
        )
    return None


def _display_name(org: dict) -> str | None:
    names = org.get("names")
    if not isinstance(names, list):
        names = []
    for wanted in ("ror_display", "label"):
        for name in names:
            if isinstance(name, dict) and wanted in (name.get("types") or []):
                return name.get("value") or None
    return org.get("name") or None  # v1 records


def _country_code(org: dict) -> str | None:
    for location in org.get("locations") or []:
        details = location.get("geonames_details") if isinstance(location, dict) else None
        code = details.get("country_code") if isinstance(details, dict) else None
        if isinstance(code, str) and len(code) == 2:
            return code.upper()
    return None


def _funder_doi(org: dict) -> str | None:
    for ext in org.get("external_ids") or []:
        if not isinstance(ext, dict) or str(ext.get("type", "")).lower() != "fundref":
            continue
        fundref = ext.get("preferred") or next(iter(ext.get("all") or []), None)
        if isinstance(fundref, str) and fundref.strip().isdigit():
            return f"10.13039/{fundref.strip()}"
    return None


_clients: dict[tuple, RorClient] = {}


def get_client(settings: GlobalSettings) -> RorClient:
    """The shared client for one ROR configuration, so the cache and rate
    window span the whole process rather than one paper."""
    ror = settings.ror
    key = (ror.url, ror.client_id, ror.requests_per_5min, ror.request_timeout, ror.cache_size)
    client = _clients.get(key)
    if client is None:
        client = _clients[key] = RorClient(settings=settings)
    return client
