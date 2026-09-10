import asyncio
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import httpx

from bibr.config import GlobalSettings, snapshot_settings
from bibr.utils.transient import is_transient_network_error

if TYPE_CHECKING:
    from bibr.clients.response_cache import RedisResponseCache
    from bibr.utils.rate_limiter import AsyncLocalRateLimiter, AsyncRedisRateLimiter

logger = logging.getLogger(__name__)

_CROSSREF_API_BASE = "https://api.crossref.org"

# Transient errors worth retrying
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds, doubles each attempt

# DOIs per bulk /works?filter=doi:... request. At ~35 chars per encoded
# `doi:<id>,` term this keeps the query string well inside any reverse proxy's
# URL limit, while collapsing a typical paper's DOI-bearing references into a
# single request.
_BULK_DOI_CHUNK = 50

# Crossref's anonymous (no-mailto) pool ceiling. The polite pool needs a
# contact address in the User-Agent; without one the configured RPM cannot be
# honored no matter how high it is set.
_ANONYMOUS_RATE_LIMIT_RPM = 60

# Ceiling for the blind multiplicative 429 throttle (seconds between requests):
# 2.0s ≈ 30 RPM, slow enough that Crossref stops throttling yet fast enough
# that a large reference list still finishes within the enrichment budget.
_MAX_THROTTLE_WINDOW_SECONDS = 2.0

# A 429-widened window starts narrowing back only after this long without a
# blind 429: long enough to outlive a Crossref throttling episode, short
# enough that a long batch run doesn't spend hours at the widened rate.
_THROTTLE_RECOVERY_QUIET_SECONDS = 60.0

# Module-level singleton (double-checked locking, same pattern as SectionClassifier etc.)
_client: "CrossrefClient | None" = None
_clients_by_settings: dict[tuple[object, ...], "CrossrefClient"] = {}
_client_lock = threading.Lock()


def _settings_key(settings: GlobalSettings) -> tuple[object, ...]:
    return (
        settings.crossref.api_email,
        settings.crossref.api_key,
        settings.crossref.rate_limit_rpm,
        settings.crossref.enrich_concurrency,
        settings.crossref.request_timeout,
        settings.crossref.cache_size,
        settings.crossref.redis_cache,
        settings.crossref.cache_redis_url,
        settings.crossref.cache_ttl_seconds,
        settings.redis.url,
        settings.cb.failure_threshold,
        settings.cb.reset_timeout_seconds,
    )


def get_client(settings: GlobalSettings | None = None) -> "CrossrefClient":
    """Get or create a Crossref client for one concrete settings snapshot."""
    global _client
    if settings is None:
        if _client is None:
            with _client_lock:
                if _client is None:
                    _client = CrossrefClient()
        return _client
    effective = settings
    key = _settings_key(effective)
    client = _clients_by_settings.get(key)
    if client is None:
        with _client_lock:
            client = _clients_by_settings.get(key)
            if client is None:
                client = CrossrefClient(settings=effective)
                _clients_by_settings[key] = client
    return client


class CrossrefClient:
    """Async client for CrossRef with rate limiting and dynamic header adaptation.

    Polite-pool compliance:
    1. Sends ``mailto`` in User-Agent header.
    2. Enforces a minimum interval between requests (via Redis or local limiter).
    3. Reads ``X-Rate-Limit-Limit`` and ``X-Rate-Limit-Interval`` response
       headers to dynamically tighten the rate if CrossRef signals a lower
       allowance than configured.
    """

    def __init__(
        self,
        mailto: str | None = None,
        *,
        settings: GlobalSettings | None = None,
    ):
        self._settings = settings if settings is not None else snapshot_settings()
        self.mailto = mailto or self._settings.crossref.api_email
        self.api_key = self._settings.crossref.api_key

        # Derive interval from configured RPM.
        # Without mailto, cap at 60 RPM (anonymous limit).
        rpm = self._settings.crossref.rate_limit_rpm
        # Silently running at a third of the configured rate is the difference
        # between an 80-reference paper enriching in ~25s and in ~81s, against
        # a 120s per-paper budget — worth naming the numbers in the warning
        # below rather than leaving the operator to infer it from timings.
        self._anonymous_clamped_from = (
            rpm if not self.mailto and rpm > _ANONYMOUS_RATE_LIMIT_RPM else None
        )
        if not self.mailto:
            rpm = min(rpm, _ANONYMOUS_RATE_LIMIT_RPM)
        self.interval = 60.0 / rpm

        ua = (
            f"bibr/1.0 (https://github.com/bibr; mailto:{self.mailto})"
            if self.mailto
            else "bibr/1.0"
        )
        headers: dict[str, str] = {"User-Agent": ua}
        if self.api_key:
            headers["Crossref-Plus-API-Token"] = f"Bearer {self.api_key}"

        self._http_headers = headers
        self._http: httpx.AsyncClient | None = None
        self._http_loop: asyncio.AbstractEventLoop | None = None

        # Floor for 429-throttle recovery: never narrow below the configured
        # rate, or below a server-advertised interval (_adapt_rate_limit
        # raises it when Crossref mandates a wider window).
        self._recovery_floor = self.interval
        self._last_blind_429_monotonic: float | None = None

        # Limiter built lazily on first request to avoid a sync Redis ping
        # during async construction. See _ensure_limiter().
        self.limiter: AsyncRedisRateLimiter | AsyncLocalRateLimiter | None = None
        # Guards lazy-init across concurrent first callers; otherwise both
        # coroutines pass the ``self.limiter is None`` check, both ping
        # Redis, and one of the two limiters is orphaned. Created
        # synchronously on first use so its construction itself is race-free.
        self._limiter_init_lock: asyncio.Lock | None = None
        self._limiter_init_loop: asyncio.AbstractEventLoop | None = None

        # Lazily created in the running event loop to avoid binding to the
        # wrong loop when the singleton is constructed at import time.
        self._enrich_semaphore: asyncio.Semaphore | None = None
        self._enrich_loop: asyncio.AbstractEventLoop | None = None
        self._enrich_concurrency = self._settings.crossref.enrich_concurrency
        self._breaker: object | None = None
        self._cb_failure_threshold = self._settings.cb.failure_threshold
        self._cb_reset_timeout = self._settings.cb.reset_timeout_seconds

        # Tier-2 shared cache (created lazily on first request when enabled).
        self._response_cache_backend: object | None = None
        self._response_cache_init = False

        if self.mailto:
            logger.info(f"CrossrefClient initialized with mailto: {self.mailto}")
        elif self._anonymous_clamped_from is not None:
            logger.warning(
                "CrossrefClient initialized WITHOUT mailto: Crossref's anonymous pool caps "
                "this client at %d RPM, not the configured %d. Set CROSSREF_API_EMAIL to "
                "enrich at the configured rate.",
                _ANONYMOUS_RATE_LIMIT_RPM,
                self._anonymous_clamped_from,
            )
        else:
            logger.warning(
                "CrossrefClient initialized WITHOUT mailto. "
                "You may be subject to stricter rate limits. "
                "Set CROSSREF_API_EMAIL environment variable."
            )

    async def _ensure_limiter(self) -> None:
        """Create the rate limiter on first request (Redis probe runs off-loop)."""
        if self.limiter is not None:
            return
        loop = asyncio.get_running_loop()
        if self._limiter_init_lock is None or self._limiter_init_loop is not loop:
            self._limiter_init_lock = asyncio.Lock()
            self._limiter_init_loop = loop
        async with self._limiter_init_lock:
            if self.limiter is not None:
                return
            try:
                if not self._settings.redis.url:
                    raise RuntimeError("Redis URL not configured")
                from redis.asyncio import Redis as AsyncRedis

                from bibr.utils.rate_limiter import AsyncRedisRateLimiter

                r = AsyncRedis.from_url(self._settings.redis.url, socket_connect_timeout=1)
                try:
                    await r.ping()
                finally:
                    await r.aclose()

                self.limiter = AsyncRedisRateLimiter(
                    redis_url=self._settings.redis.url,
                    resource_id="crossref",
                    max_requests=1,
                    window_seconds=self.interval,
                )
            except Exception as exc:
                from bibr.utils.rate_limiter import AsyncLocalRateLimiter

                logger.info("Redis unavailable, using local Crossref rate limiter: %s", exc)
                self.limiter = AsyncLocalRateLimiter(
                    resource_id="crossref",
                    max_requests=1,
                    window_seconds=self.interval,
                )

    def _make_http_client(self) -> httpx.AsyncClient:
        """Construct a Crossref HTTP client with the configured transport policy."""
        return httpx.AsyncClient(
            base_url=_CROSSREF_API_BASE,
            headers=self._http_headers,
            timeout=self._settings.crossref.request_timeout,
            # Crossref 301-redirects /works/{doi} for non-canonical DOI forms
            # (casing, encoding); without this those refs fail permanently.
            follow_redirects=True,
        )

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Return an HTTP client bound to the current running event loop."""
        loop = asyncio.get_running_loop()
        if self._http is not None and self._http_loop is loop and not self._http.is_closed:
            return self._http

        old_http = self._http
        old_limiter = self.limiter
        self._http = self._make_http_client()
        self._http_loop = loop
        self.limiter = None
        self._limiter_init_lock = None
        self._limiter_init_loop = None

        if old_http is not None and not old_http.is_closed:
            try:
                await old_http.aclose()
            except RuntimeError:
                logger.debug("Discarding Crossref HTTP client bound to a closed event loop")
        if old_limiter is not None:
            close = getattr(old_limiter, "close", None)
            if close is not None:
                try:
                    await close()
                except RuntimeError:
                    logger.debug("Discarding Crossref limiter bound to a closed event loop")
        return self._http

    @property
    def enrich_semaphore(self) -> asyncio.Semaphore:
        """Lazily create the semaphore in the current event loop.

        Recreated whenever the running loop changes — a singleton-held
        semaphore bound to a finished ``asyncio.run()`` loop raises
        ``RuntimeError`` on the next acquire (M6). Waiters from the old
        loop are gone with that loop, so dropping the instance is safe.
        """
        loop = asyncio.get_running_loop()
        if self._enrich_semaphore is None or self._enrich_loop is not loop:
            self._enrich_semaphore = asyncio.Semaphore(self._enrich_concurrency)
            self._enrich_loop = loop
        return self._enrich_semaphore

    def _get_breaker(self):
        """Lazily create the circuit breaker in the current event loop."""
        if self._breaker is None:
            from bibr.utils.circuit_breaker import AsyncCircuitBreaker

            self._breaker = AsyncCircuitBreaker(
                failure_threshold=self._cb_failure_threshold,
                reset_timeout=self._cb_reset_timeout,
                name="crossref",
            )
        return self._breaker

    # ------------------------------------------------------------------
    # Public API (unchanged signatures for enrich_references callers)
    # ------------------------------------------------------------------

    async def works(self, ids: str | list[str], **kwargs) -> dict[str, Any]:  # noqa: ARG002
        """Fetch a work by DOI(s).

        Args:
            ids: DOI or list of DOIs.

        Returns:
            The JSON response from CrossRef.
        """
        from urllib.parse import quote

        from bibr.utils.text import is_url_safe_doi

        if isinstance(ids, list):
            ids = ",".join(ids)
        # Refuse DOIs that would traverse out of `/works/*` (path traversal within
        # the hardcoded api.crossref.org host; audit L11). `is_url_safe_doi` rejects
        # `.`/`..`/empty path segments; a comma-joined list is caught the same way
        # since any unsafe member surfaces its bad segment in the split.
        if not is_url_safe_doi(ids):
            logger.debug("crossref works rejected unsafe DOI id(s): %r", ids)
            return {}
        # Preserve `/` (DOI internal slash), `,` (DOI list separator), and
        # `();:` which legacy DOIs (e.g. `10.1002/(SICI)...;2-#`) require
        # literal in the Crossref REST path. Encode genuinely unsafe chars
        # (space, `#`, `<`, `>`, etc.).
        return await self._cached(
            # DOIs are case-insensitive, so key on the folded form: otherwise
            # "10.1037/ABC" and "10.1037/abc" occupied two cache entries and
            # each paid its own network round trip. The request path keeps the
            # original casing.
            f"works:{ids.casefold()}",
            lambda: self._request(f"/works/{quote(ids, safe='/,();:')}"),
        )

    async def prefetch_works_by_doi(self, dois: list[str]) -> int:
        """Warm the response cache for many DOIs with one request each chunk.

        ``/works/{doi}`` answers exactly one DOI, so an 80-reference paper spent
        one rate-limited slot per DOI-bearing reference. ``/works?filter=doi:``
        answers a whole chunk in one slot, and the entries are stored under the
        same ``works:{doi}`` keys the per-reference path reads — so that path is
        untouched and simply stops missing.

        Only hits are seeded. A DOI the bulk query does not return keeps its own
        lookup, so its 404 still surfaces as a 404 rather than a silent miss
        that would fall through to a bibliographic search.

        Returns the number of cached entries seeded. Never raises: enrichment
        must degrade to the per-reference path, not fail.
        """
        from bibr.utils.text import is_url_safe_doi

        if not dois:
            return 0
        seen: set[str] = set()
        wanted: list[str] = []
        for doi in dois:
            if not doi:
                continue
            folded = doi.casefold()
            if folded in seen or not is_url_safe_doi(doi):
                continue
            seen.add(folded)
            if self._cache_peek(f"works:{folded}") is None:
                wanted.append(doi)
        if not wanted:
            return 0

        seeded = 0
        for start in range(0, len(wanted), _BULK_DOI_CHUNK):
            chunk = wanted[start : start + _BULK_DOI_CHUNK]
            try:
                data = await self._request(
                    "/works",
                    params={
                        "filter": ",".join(f"doi:{d}" for d in chunk),
                        # Without an explicit rows the default (20) would
                        # silently truncate a full chunk's results.
                        "rows": len(chunk),
                    },
                )
            except Exception as e:  # noqa: BLE001 — degrade to the per-ref path
                logger.debug("Crossref bulk DOI prefetch failed for %d DOIs: %s", len(chunk), e)
                continue
            for item in (data.get("message") or {}).get("items") or []:
                item_doi = item.get("DOI")
                if not item_doi:
                    continue
                # Store the single-work shape so the cached value is
                # indistinguishable from what /works/{doi} would have returned.
                await self._cache_store(f"works:{item_doi.casefold()}", {"message": item})
                seeded += 1
        if seeded:
            logger.debug(
                "Crossref bulk DOI prefetch seeded %d/%d works in %d request(s)",
                seeded,
                len(wanted),
                (len(wanted) + _BULK_DOI_CHUNK - 1) // _BULK_DOI_CHUNK,
            )
        return seeded

    async def search(self, query: str, limit: int = 3, **kwargs) -> dict[str, Any]:  # noqa: ARG002
        """Search works by bibliographic query.

        Args:
            query: Bibliographic search string (title + author + year).
            limit: Max results to return.

        Returns:
            The JSON response from CrossRef.
        """
        # select= is only supported on the list/search route, not on /works/{doi}
        _SELECT = "DOI,title,author,editor,issued,container-title,volume,issue,page,publisher,type,URL,score"
        return await self._cached(
            f"search:{limit}:{query}",
            lambda: self._request(
                "/works",
                params={"query.bibliographic": query, "rows": limit, "select": _SELECT},
            ),
        )

    async def _ensure_response_cache(self) -> "RedisResponseCache | None":
        """Lazily build the tier-2 Redis cache on first use (None if disabled)."""
        if not self._settings.crossref.redis_cache:
            return None
        if not self._response_cache_init:
            self._response_cache_init = True
            url = self._settings.crossref.cache_redis_url or self._settings.redis.url
            if not url:
                logger.warning(
                    "CROSSREF_REDIS_CACHE enabled but no Redis URL configured "
                    "(set CROSSREF_CACHE_REDIS_URL or REDIS_URL); shared cache disabled"
                )
                self._response_cache_backend = None
            else:
                from bibr.clients.response_cache import RedisResponseCache

                self._response_cache_backend = RedisResponseCache(
                    redis_url=url,
                    ttl_seconds=self._settings.crossref.cache_ttl_seconds,
                    connect_timeout=self._settings.redis.connect_timeout_seconds,
                    socket_timeout=self._settings.redis.socket_timeout_seconds,
                )
        return self._response_cache_backend

    async def _cached(
        self, key: str, fetch: Callable[[], Awaitable[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Two-tier cache: in-process LRU → shared Redis → upstream fetch.

        Tier 1 is the per-process LRU (``CROSSREF_CACHE_SIZE``, 0 = off). Tier 2
        is the optional shared Redis cache (``CROSSREF_REDIS_CACHE``), which
        survives restarts and is shared across serve + workers. Failures are
        never cached; a Redis outage degrades to the fetch. With both disabled
        this is a plain fetch (today's behavior).
        """
        size = self._settings.crossref.cache_size
        cache: OrderedDict[str, dict[str, Any]] | None = None
        if size > 0:
            cache = self.__dict__.get("_response_cache")
            if cache is None:
                cache = self.__dict__["_response_cache"] = OrderedDict()
            if key in cache:
                cache.move_to_end(key)
                return cache[key]

        redis_cache = await self._ensure_response_cache()
        if redis_cache is not None:
            hit = await redis_cache.get(key)
            if hit is not None:
                if cache is not None:
                    cache[key] = hit
                    while len(cache) > size:
                        cache.popitem(last=False)
                return hit

        data = await fetch()

        await self._cache_store(key, data)
        return data

    def _cache_peek(self, key: str) -> dict[str, Any] | None:
        """Tier-1 (in-process) lookup only — no await, no Redis round trip.

        Used by the bulk prefetch to skip DOIs already in hand. A tier-2 hit
        that this misses costs nothing worse than including that DOI in a
        request that was being sent anyway.
        """
        if self._settings.crossref.cache_size <= 0:
            return None
        cache = self.__dict__.get("_response_cache")
        return None if cache is None else cache.get(key)

    async def _cache_store(self, key: str, data: dict[str, Any]) -> None:
        """Write one response into both cache tiers."""
        redis_cache = await self._ensure_response_cache()
        if redis_cache is not None:
            await redis_cache.set(key, data)
        size = self._settings.crossref.cache_size
        if size > 0:
            cache = self.__dict__.get("_response_cache")
            if cache is None:
                cache = self.__dict__["_response_cache"] = OrderedDict()
            cache[key] = data
            while len(cache) > size:
                cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _request(self, path: str, params: dict | None = None) -> dict[str, Any]:
        """Rate-limited GET with retry, circuit breaker, and dynamic header adaptation."""
        async with self._get_breaker():
            return await self._request_inner(path, params)

    async def _request_inner(self, path: str, params: dict | None = None) -> dict[str, Any]:
        """Inner retry loop for Crossref requests (wrapped by circuit breaker)."""
        http = await self._get_http_client()
        await self._ensure_limiter()
        assert self.limiter is not None  # noqa: S101 — _ensure_limiter sets it
        last_exc: Exception | None = None
        retry_after: float | None = None
        for attempt in range(_MAX_RETRIES):
            await self.limiter.acquire()
            try:
                logger.debug("CrossRef GET %s (attempt %d)", path, attempt + 1)
                resp = await http.get(path, params=params)
                self._adapt_rate_limit(resp.headers)
                resp.raise_for_status()
                self._recover_rate_limit()
                try:
                    data: dict[str, Any] = resp.json()
                    return data
                except ValueError as e:
                    raise httpx.DecodingError(str(e), request=resp.request) from e
            except httpx.HTTPStatusError as e:
                last_exc = e
                # Crossref also tells us the limit on 429 — adapt before
                # giving up or retrying so the next request has the new
                # window width.
                if e.response is not None:
                    self._adapt_rate_limit(e.response.headers)
                    if e.response.status_code == 429:
                        self._throttle_on_429(e.response.headers)
                    retry_after = self._parse_retry_after(e.response.headers)
                if not self._is_retryable(e) or attempt == _MAX_RETRIES - 1:
                    # 404 just means "no Crossref record for that DOI" —
                    # extremely common when extracted DOIs are slightly
                    # malformed or point to non-Crossref registries (OSF,
                    # bioRxiv data deposits). Log quietly so a normal run
                    # doesn't look full of red ERRORs.
                    if e.response is not None and e.response.status_code == 404:
                        logger.debug("CrossRef 404 (no record): %s", path)
                    else:
                        logger.error("CrossRef request failed: %s", e)
                    raise
            except Exception as e:  # noqa: BLE001 — re-raised unless transient
                # Naming only ReadTimeout/ConnectTimeout/ConnectError left whole
                # classes of retryable failure — RemoteProtocolError, ReadError,
                # PoolTimeout, and the DecodingError raised just above for an
                # unparseable body — escaping with zero retries, which
                # permanently loses enrichment for that reference.
                if not is_transient_network_error(e):
                    raise
                last_exc = e
                if attempt == _MAX_RETRIES - 1:
                    logger.error("CrossRef request failed: %s", e)
                    raise

            delay = min(_RETRY_BASE_DELAY * (2**attempt), 30.0)
            if retry_after is not None:
                # CrossRef told us exactly when to come back (429 Retry-After);
                # retrying on the shorter backoff just earns another 429.
                delay = min(max(delay, retry_after), 60.0)
                retry_after = None
            logger.warning(
                "CrossRef request failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1,
                _MAX_RETRIES,
                delay,
                last_exc,
            )
            await asyncio.sleep(delay)

        assert last_exc is not None  # noqa: S101 — loop sets it on every retry path
        raise last_exc

    def _adapt_rate_limit(self, headers: httpx.Headers) -> None:
        """Tighten the local rate limiter if CrossRef headers advertise a
        lower allowance than currently configured.

        Reads:
        - ``X-Rate-Limit-Limit``: requests allowed per interval
        - ``X-Rate-Limit-Interval``: time window (e.g. ``1s``)
        """
        limit_str = headers.get("x-rate-limit-limit")
        interval_str = headers.get("x-rate-limit-interval")
        if not limit_str or not interval_str or self.limiter is None:
            return

        try:
            limit = int(limit_str)
            # Interval is like "1s" — parse the numeric part
            match = re.match(r"(\d+)", interval_str)
            if not match:
                return
            interval_secs = int(match.group(1))

            server_interval = interval_secs / limit
            if server_interval > self.limiter.window_seconds:
                configured_floor = getattr(self, "interval", self.limiter.window_seconds)
                logger.info(
                    "CrossRef headers indicate tighter rate limit: "
                    "%d req / %ds → %.3fs interval (was %.3fs)",
                    limit,
                    interval_secs,
                    server_interval,
                    self.limiter.window_seconds,
                )
                # Single attribute write; atomic in CPython, no lock needed.
                self.limiter.window_seconds = server_interval
                # A server-mandated interval is authoritative — 429 recovery
                # must never narrow the window back below it.
                self._recovery_floor = max(
                    getattr(self, "_recovery_floor", configured_floor), server_interval
                )
        except (ValueError, ZeroDivisionError):
            pass

    def _throttle_on_429(self, headers: httpx.Headers) -> None:
        """Tighten the local limiter on a 429 that carries NO rate-limit headers.

        Crossref removed the ``X-Rate-Limit-*`` response headers years ago, so
        in practice every 429 arrives blind and ``_adapt_rate_limit`` never
        fires — without this, a storm of resolver-miss bibliographic searches
        keeps hammering at the configured RPM and earns hundreds of 429s per
        run. Multiplicative widening (capped) self-heals the rate instead;
        header-carrying responses stay owned by ``_adapt_rate_limit``.
        """
        if headers.get("x-rate-limit-limit") and headers.get("x-rate-limit-interval"):
            return
        if self.limiter is None:
            return
        # Refresh the recovery quiet-period timer on EVERY blind 429 — also
        # when the window already sits at the ceiling — so recovery cannot
        # start in the middle of a sustained throttling episode.
        self._last_blind_429_monotonic = time.monotonic()
        new_window = min(self.limiter.window_seconds * 2, _MAX_THROTTLE_WINDOW_SECONDS)
        if new_window > self.limiter.window_seconds:
            logger.info(
                "CrossRef 429 without rate-limit headers — throttling to %.2fs/request",
                new_window,
            )
            # Single attribute write; atomic in CPython, no lock needed.
            self.limiter.window_seconds = new_window

    def _recover_rate_limit(self) -> None:
        """Narrow a 429-widened window back once Crossref has gone quiet.

        ``_throttle_on_429`` only ever widens the window, so without this a
        single early 429 burst leaves the client at up to 2s/request for the
        life of the process — slow enough that DOI-heavy papers blow the
        per-paper enrichment budget hours after Crossref stopped complaining.
        Halving per success after a quiet minute (multiplicative increase,
        multiplicative decrease) restores the configured rate within a few
        requests, while a still-throttling Crossref immediately re-widens the
        window and resets the quiet timer.
        """
        lim = self.limiter
        if lim is None or lim.window_seconds <= self._recovery_floor:
            return
        last = self._last_blind_429_monotonic
        if last is not None and time.monotonic() - last < _THROTTLE_RECOVERY_QUIET_SECONDS:
            return
        new_window = max(self._recovery_floor, lim.window_seconds / 2)
        logger.info(
            "CrossRef quiet since last 429 — recovering throttle to %.2fs/request",
            new_window,
        )
        # Single attribute write; atomic in CPython, no lock needed.
        lim.window_seconds = new_window

    @staticmethod
    def _parse_retry_after(headers: httpx.Headers) -> float | None:
        """Parse a numeric ``Retry-After`` header (seconds). HTTP-date form and
        garbage both yield None so the caller keeps its default backoff."""
        value = headers.get("retry-after")
        if not value:
            return None
        try:
            secs = float(value)
        except ValueError:
            return None
        return secs if secs >= 0 else None

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Check if an exception is transient and worth retrying."""
        if isinstance(exc, (httpx.ReadTimeout, httpx.ConnectTimeout)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status >= 500 or status == 429
        return False

    async def close(self):
        """Close the HTTP client and rate limiter."""
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
        self._http = None
        self._http_loop = None
        if self.limiter is not None:
            await self.limiter.close()
        if self._response_cache_backend is not None:
            await self._response_cache_backend.close()
