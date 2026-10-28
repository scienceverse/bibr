"""Async client for ROR affiliation matching.

One ``GET {url}/organizations?affiliation=<text>&single_search`` per string.
ROR marks at most one result ``chosen`` when it is confident in the match and
advises against selecting by score, so only the chosen result is taken; a
string without one stays unmatched.

ROR limits each IP address to 2000 requests per 5 minutes. Its lower limit
for requests without a client ID (50 per 5 minutes) is not enforced, and
client ID registration is paused, so the client's own budget is 1000 requests
per 5 minutes per process without a client ID: half the per-address limit,
so two processes on one address stay under it. With a client ID (sent as the
``Client-Id`` header) the budget is 2000. ``ROR_REQUESTS_PER_5MIN`` sets it.
The client keeps to the budget with a sliding window, remembers answers
(misses too) in a process-wide cache, and backs off entirely after a 429 for
the ``Retry-After`` interval (at most one window). api.ror.org answers
over-limit requests with a 429 and no ``Retry-After``, so the pause is one
window, 5 minutes. The first 429 of a pause also halves the process's budget,
not below 50 (or the configured budget, if lower), so processes that share an
address settle under ROR's limit; each hour without a 429 doubles it back, up
to the configured budget. A 403 means ROR has blocked the address (or
``ROR_URL`` names a wrong path): lookups pause for one window, without
halving. A lookup given a deadline does not wait for a slot that would free
after it: it reports "request budget spent" at once (a 429 backoff reports
"rate limited", a 403 pause "blocked (HTTP 403)"), so a paper does not sit out
its whole ROR time budget in the queue.
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
# Per process. Two processes on one address at 1000 each stay under ROR's
# 2000 per address; one 429 stops ROR lookups for up to a window.
_ANONYMOUS_REQUESTS_PER_WINDOW = 1000
_CLIENT_ID_REQUESTS_PER_WINDOW = 2000
# A 429 halves the budget, never below ROR's published figure for clients
# without an ID (or the configured budget, if lower); an hour without a 429
# doubles it back toward the configured budget.
_MIN_BUDGET_AFTER_429 = 50
_RECOVERY_QUIET_SECONDS = 3600.0
_RATE_LIMITED = "rate limited"
_BLOCKED = "blocked (HTTP 403)"
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
        self._budget = budget
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
        self._block_reason = _RATE_LIMITED
        self._last_429: float | None = None

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
        """True while backing off after a 429 or a 403."""
        return time.monotonic() < self._blocked_until

    def _lower_budget(self) -> None:
        floor = min(_MIN_BUDGET_AFTER_429, self._budget)
        self._limiter.max_requests = max(floor, self._limiter.max_requests // 2)

    def _recover_budget(self) -> None:
        """Double a lowered budget back after an hour without a 429."""
        if self._last_429 is None or self._limiter.max_requests >= self._budget:
            return
        now = time.monotonic()
        if now - self._last_429 < _RECOVERY_QUIET_SECONDS:
            return
        self._limiter.max_requests = min(self._budget, self._limiter.max_requests * 2)
        self._last_429 = now  # one doubling per quiet hour
        logger.info(
            "No ROR 429 for an hour; ROR budget back up to %d requests per 5 minutes",
            self._limiter.max_requests,
        )

    async def match(self, text: str) -> OrganizationMatch | None:
        """ROR's chosen organization for *text*, or ``None``."""
        result, _failure = await self.lookup(text)
        return result

    async def lookup(
        self, text: str, *, deadline: float | None = None
    ) -> tuple[OrganizationMatch | None, str | None]:
        """ROR's chosen organization for *text*, and why the lookup failed if it did.

        The failure is ``None`` for an answered lookup, matched or not, and a
        short reason when ROR could not be asked or did not answer usefully:
        "HTTP 503", "rate limited" (backing off after a 429), "blocked (HTTP
        403)" (pausing after a 403), "request budget spent", "invalid JSON",
        or the type name of the error raised ("ConnectError"). It never
        raises an ``Exception``.

        *deadline* is a ``time.monotonic()`` value: when the request budget
        has no slot free before it, the lookup fails with "request budget
        spent" without waiting.
        """
        try:
            return await self._lookup(text, deadline)
        except Exception as exc:  # noqa: BLE001 - one bad answer must not fail the paper's lookups
            logger.debug("ROR lookup failed for %r", text[:80], exc_info=True)
            return None, type(exc).__name__

    async def _lookup(
        self, text: str, deadline: float | None = None
    ) -> tuple[OrganizationMatch | None, str | None]:
        text = " ".join(text.split())
        if not _MIN_CHARS <= len(text) <= _MAX_CHARS:
            return None, None
        key = _cache_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key], None
        if self.blocked:
            return None, self._block_reason
        self._recover_budget()
        if deadline is not None and (
            time.monotonic() + self._limiter.seconds_until_slot() >= deadline
        ):
            return None, "request budget spent"
        await self._limiter.acquire()
        try:
            response = await self._http().get(
                f"{self._base_url}/organizations",
                params={"affiliation": text, "single_search": ""},
                headers=self._headers,
            )
        except httpx.HTTPError as exc:
            logger.debug("ROR request failed for %r: %s", text[:80], exc)
            return None, type(exc).__name__
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "")
            # isdecimal, not isdigit: float() rejects a superscript digit. The
            # back-off is process-wide, so a huge value (float("9" * 400) is
            # inf) is capped at one rate-limit window rather than ending ROR
            # matching for the life of the process.
            delay = float(retry_after) if retry_after.isdecimal() else _WINDOW_SECONDS
            delay = min(delay, _WINDOW_SECONDS)
            now = time.monotonic()
            self._last_429 = now
            # Lookups already in flight when ROR starts refusing answer 429
            # too: only the first 429 of a pause lowers the budget and warns.
            if now >= self._blocked_until:
                self._lower_budget()
                self._block_reason = _RATE_LIMITED
                logger.warning(
                    "ROR rate limit reached; skipping ROR lookups for %.0fs, then at most %d "
                    "requests per 5 minutes",
                    delay,
                    self._limiter.max_requests,
                )
            self._blocked_until = max(self._blocked_until, now + delay)
            return None, _RATE_LIMITED
        if response.status_code == 403:
            # ROR answers 403 only for addresses it has blocked (bad queries
            # get 400), and its gateway answers 403 for a wrong ROR_URL path:
            # every later request would fail too, so pause without halving.
            if not (self.blocked and self._block_reason == _BLOCKED):
                logger.warning(
                    "ROR refused this address (HTTP 403): %r; skipping ROR lookups for %.0fs",
                    response.text[:200],
                    _WINDOW_SECONDS,
                )
            self._block_reason = _BLOCKED
            self._blocked_until = max(self._blocked_until, time.monotonic() + _WINDOW_SECONDS)
            return None, _BLOCKED
        if response.status_code != 200:
            logger.debug("ROR returned HTTP %d for %r", response.status_code, text[:80])
            return None, f"HTTP {response.status_code}"
        try:
            result = chosen_organization(response.json())
        except ValueError:
            logger.debug("ROR returned unparseable JSON for %r", text[:80])
            return None, "invalid JSON"
        self._remember(key, result)
        return result, None


def chosen_organization(payload: Any) -> OrganizationMatch | None:
    """The ``chosen`` result of a ROR v2 affiliation response, or ``None``.

    A field of the wrong JSON type counts as absent, so a malformed record is
    no match rather than an error.
    """
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict) or item.get("chosen") is not True:
            continue
        org = item.get("organization")
        if not isinstance(org, dict):
            return None
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


def _list(record: dict, key: str) -> list:
    value = record.get(key)
    return value if isinstance(value, list) else []


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _display_name(org: dict) -> str | None:
    names = [name for name in _list(org, "names") if isinstance(name, dict)]
    for wanted in ("ror_display", "label"):
        for name in names:
            if wanted in _list(name, "types"):
                return _text(name.get("value"))
    return _text(org.get("name"))  # v1 records


def _country_code(org: dict) -> str | None:
    for location in _list(org, "locations"):
        details = location.get("geonames_details") if isinstance(location, dict) else None
        code = details.get("country_code") if isinstance(details, dict) else None
        if isinstance(code, str) and len(code) == 2:
            return code.upper()
    return None


def _funder_doi(org: dict) -> str | None:
    for ext in _list(org, "external_ids"):
        if not isinstance(ext, dict) or str(ext.get("type", "")).lower() != "fundref":
            continue
        fundref = ext.get("preferred") or next(iter(_list(ext, "all")), None)
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
