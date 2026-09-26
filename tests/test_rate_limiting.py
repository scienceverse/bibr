import time

import pytest

from bibr.utils.rate_limiter import AsyncLocalRateLimiter

# --- AsyncLocalRateLimiter tests (no Redis needed) ---


@pytest.mark.asyncio
async def test_local_limiter_acquire_single():
    """Single acquire should return immediately."""
    limiter = AsyncLocalRateLimiter(resource_id="test", max_requests=1, window_seconds=1.0)
    t0 = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_local_limiter_rate_limiting():
    """Second acquire within the window should block."""
    limiter = AsyncLocalRateLimiter(resource_id="test", max_requests=1, window_seconds=0.5)
    await limiter.acquire()

    t0 = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - t0
    # Should have waited ~0.5s (generous threshold for slow CI runners)
    assert elapsed >= 0.3


@pytest.mark.asyncio
async def test_local_limiter_multiple_requests():
    """With max_requests=3, the first 3 should be instant."""
    limiter = AsyncLocalRateLimiter(resource_id="test", max_requests=3, window_seconds=1.0)
    t0 = time.monotonic()
    for _ in range(3):
        await limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_local_limiter_close_is_noop():
    """close() should not raise."""
    limiter = AsyncLocalRateLimiter(resource_id="test")
    await limiter.close()


# --- AsyncRedisRateLimiter tests (Redis mocked) ---

redis = pytest.importorskip("redis", reason="redis not installed (optional [api] dependency)")

from unittest.mock import AsyncMock, patch

import httpx

from bibr.clients.crossref import CrossrefClient
from bibr.utils.rate_limiter import AsyncRedisRateLimiter


@pytest.mark.asyncio
async def test_crossref_client_init_and_call():
    # Limiter is built lazily on first request; only async Redis is used.
    with patch("redis.asyncio.Redis") as mock_redis_cls:
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.eval = AsyncMock(return_value=0)
        mock_redis_cls.from_url.return_value = mock_redis

        client = CrossrefClient(mailto="test@example.com")

        # Interval derives from CROSSREF_RATE_LIMIT_RPM (env-configurable).
        from bibr.config import Settings

        expected = 60.0 / Settings.crossref.rate_limit_rpm
        assert client.interval == pytest.approx(expected)
        # Limiter is unset until first request triggers _ensure_limiter
        assert client.limiter is None

        # Mock the httpx response
        mock_response = httpx.Response(
            200,
            json={"message": "success"},
            headers={"content-type": "application/json"},
            request=httpx.Request("GET", "https://api.crossref.org/works/10.1000/1"),
        )

        http = await client._get_http_client()
        with patch.object(http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            res = await client.works("10.1000/1")

            assert res == {"message": "success"}
            assert isinstance(client.limiter, AsyncRedisRateLimiter)
            assert mock_redis.eval.called

        await client.close()
        assert mock_redis.aclose.called


@pytest.mark.asyncio
async def test_crossref_dynamic_rate_adaptation():
    """CrossRef response headers should tighten the rate limiter."""
    with patch("redis.asyncio.Redis") as mock_redis_cls:
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.eval = AsyncMock(return_value=0)
        mock_redis_cls.from_url.return_value = mock_redis

        client = CrossrefClient(mailto="test@example.com")

        # Simulate CrossRef signalling a tighter limit (e.g. 2 req/1s = 0.5s)
        mock_response = httpx.Response(
            200,
            json={"message": "success"},
            headers={
                "content-type": "application/json",
                "x-rate-limit-limit": "2",
                "x-rate-limit-interval": "1s",
            },
            request=httpx.Request("GET", "https://api.crossref.org/works/10.1000/1"),
        )

        http = await client._get_http_client()
        with patch.object(http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            await client.works("10.1000/1")

            # 2 req / 1s → 0.5s interval, which is tighter than the 0.3s default
            assert client.limiter.window_seconds == 0.5

        await client.close()


@pytest.mark.asyncio
async def test_crossref_429_honors_retry_after():
    """A 429's Retry-After header must set the retry delay — CrossRef tells us exactly
    when to come back; retrying on the fixed 1s/2s backoff just earns another 429."""
    client = CrossrefClient(mailto="test@example.com")
    responses = [
        httpx.Response(
            429,
            headers={"retry-after": "7"},
            request=httpx.Request("GET", "https://api.crossref.org/works/10.1000/1"),
        ),
        httpx.Response(
            200,
            json={"message": "ok"},
            headers={"content-type": "application/json"},
            request=httpx.Request("GET", "https://api.crossref.org/works/10.1000/1"),
        ),
    ]
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    http = await client._get_http_client()
    with (
        patch.object(http, "get", new_callable=AsyncMock, side_effect=responses),
        patch("bibr.clients.crossref.asyncio.sleep", side_effect=fake_sleep),
        patch("bibr.config.Settings.redis") as mock_redis_settings,
    ):
        mock_redis_settings.url = None
        res = await client.works("10.1000/1")

    assert res == {"message": "ok"}
    assert 7.0 in sleeps  # Retry-After respected, not the 1s base backoff
    await client.close()


@pytest.mark.asyncio
async def test_crossref_429_retry_after_capped():
    """An absurd Retry-After must not stall enrichment for minutes — cap it."""
    client = CrossrefClient(mailto="test@example.com")
    responses = [
        httpx.Response(
            429,
            headers={"retry-after": "3600"},
            request=httpx.Request("GET", "https://api.crossref.org/works/10.1000/2"),
        ),
        httpx.Response(
            200,
            json={"message": "ok"},
            headers={"content-type": "application/json"},
            request=httpx.Request("GET", "https://api.crossref.org/works/10.1000/2"),
        ),
    ]
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    http = await client._get_http_client()
    with (
        patch.object(http, "get", new_callable=AsyncMock, side_effect=responses),
        patch("bibr.clients.crossref.asyncio.sleep", side_effect=fake_sleep),
        patch("bibr.config.Settings.redis") as mock_redis_settings,
    ):
        mock_redis_settings.url = None
        await client.works("10.1000/2")

    assert sleeps and max(sleeps) <= 60.0
    await client.close()


@pytest.mark.asyncio
class TestAdaptRateLimitEdgeCases:
    """Edge cases for CrossrefClient._adapt_rate_limit."""

    async def test_missing_headers_noop(self):
        """Missing rate-limit headers → no change."""
        with patch("redis.asyncio.Redis") as mock_redis_cls:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis_cls.from_url.return_value = mock_redis
            client = CrossrefClient(mailto="test@example.com")
            await client._ensure_limiter()
            original = client.limiter.window_seconds

            headers = httpx.Headers({"content-type": "application/json"})
            client._adapt_rate_limit(headers)

            assert client.limiter.window_seconds == original
            await client.close()

    async def test_looser_limit_not_applied(self):
        """Server advertising a looser limit should NOT relax the local limiter."""
        with patch("redis.asyncio.Redis") as mock_redis_cls:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis_cls.from_url.return_value = mock_redis
            client = CrossrefClient(mailto="test@example.com")
            await client._ensure_limiter()
            # Set a tight window
            client.limiter.window_seconds = 1.0

            headers = httpx.Headers(
                {
                    "x-rate-limit-limit": "1000",
                    "x-rate-limit-interval": "1s",
                }
            )
            client._adapt_rate_limit(headers)

            # 1/1000 = 0.001 < 1.0 → should NOT relax
            assert client.limiter.window_seconds == 1.0
            await client.close()

    async def test_zero_limit_no_crash(self):
        """limit=0 causes ZeroDivisionError → silently caught."""
        with patch("redis.asyncio.Redis") as mock_redis_cls:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis_cls.from_url.return_value = mock_redis
            client = CrossrefClient(mailto="test@example.com")
            await client._ensure_limiter()
            original = client.limiter.window_seconds

            headers = httpx.Headers(
                {
                    "x-rate-limit-limit": "0",
                    "x-rate-limit-interval": "1s",
                }
            )
            client._adapt_rate_limit(headers)

            assert client.limiter.window_seconds == original
            await client.close()

    async def test_non_numeric_limit_no_crash(self):
        """Non-integer limit string → ValueError silently caught."""
        with patch("redis.asyncio.Redis") as mock_redis_cls:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis_cls.from_url.return_value = mock_redis
            client = CrossrefClient(mailto="test@example.com")
            await client._ensure_limiter()
            original = client.limiter.window_seconds

            headers = httpx.Headers(
                {
                    "x-rate-limit-limit": "abc",
                    "x-rate-limit-interval": "1s",
                }
            )
            client._adapt_rate_limit(headers)

            assert client.limiter.window_seconds == original
            await client.close()

    async def test_unparseable_interval_noop(self):
        """Interval without leading digits → early return."""
        with patch("redis.asyncio.Redis") as mock_redis_cls:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis_cls.from_url.return_value = mock_redis
            client = CrossrefClient(mailto="test@example.com")
            await client._ensure_limiter()
            original = client.limiter.window_seconds

            headers = httpx.Headers(
                {
                    "x-rate-limit-limit": "10",
                    "x-rate-limit-interval": "unknown",
                }
            )
            client._adapt_rate_limit(headers)

            assert client.limiter.window_seconds == original
            await client.close()


@pytest.mark.asyncio
async def test_redis_rate_limiter_acquire():
    with patch("redis.asyncio.Redis") as mock_redis_cls:
        mock_redis = AsyncMock()
        mock_redis_cls.from_url.return_value = mock_redis

        limiter = AsyncRedisRateLimiter(
            "redis://test", "resource", max_requests=1, window_seconds=1.0
        )

        # Mock eval to return 100 ms wait (the Lua script reports milliseconds)
        mock_redis.eval.return_value = 100

        # Mock sleep
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await limiter.acquire()

            # Should have called sleep once with ~0.1
            assert mock_sleep.called
            args, _ = mock_sleep.call_args
            assert abs(args[0] - 0.1) < 0.001

            # redis.eval called once (reservation made)
            assert mock_redis.eval.call_count == 1


# --- AsyncRedisRateLimiter strict-interval retry path -----------------


@pytest.mark.asyncio
async def test_strict_interval_retries_on_redis_error_then_gives_up():
    """When Redis fails, the limiter retries and finally proceeds without rate
    limiting after exhausting max_retries — the request must not be blocked."""
    with patch("redis.asyncio.Redis") as mock_redis_cls:
        mock_redis = AsyncMock()
        mock_redis.eval = AsyncMock(side_effect=ConnectionError("redis down"))
        mock_redis_cls.from_url.return_value = mock_redis

        limiter = AsyncRedisRateLimiter(
            "redis://test", "resource", max_requests=1, window_seconds=0.1, max_retries=2
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await limiter.acquire()

        # Should have retried max_retries times, all failures
        assert mock_redis.eval.call_count == 2


# --- AsyncRedisRateLimiter sliding-window path ------------------------


@pytest.mark.asyncio
async def test_sliding_window_acquired_immediately():
    """When the window is empty, eval returns 1 (acquired) and acquire returns."""
    with patch("redis.asyncio.Redis") as mock_redis_cls:
        mock_redis = AsyncMock()
        mock_redis.eval = AsyncMock(return_value=1)
        mock_redis_cls.from_url.return_value = mock_redis

        limiter = AsyncRedisRateLimiter(
            "redis://test", "resource", max_requests=5, window_seconds=1.0
        )

        await limiter.acquire()
        assert mock_redis.eval.call_count == 1


@pytest.mark.asyncio
async def test_sliding_window_blocks_then_acquires():
    """First eval returns 0 (full), second returns 1 (slot freed). Acquire polls."""
    with patch("redis.asyncio.Redis") as mock_redis_cls:
        mock_redis = AsyncMock()
        # Window full once, then slot free
        mock_redis.eval = AsyncMock(side_effect=[0, 1])
        mock_redis_cls.from_url.return_value = mock_redis

        limiter = AsyncRedisRateLimiter(
            "redis://test", "resource", max_requests=2, window_seconds=1.0
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await limiter.acquire()

        assert mock_redis.eval.call_count == 2
        # The poll between attempts should have slept ~0.1s
        mock_sleep.assert_awaited()


@pytest.mark.asyncio
async def test_sliding_window_retries_on_redis_error():
    """Redis errors during eval trigger retry, then give up after max_retries."""
    with patch("redis.asyncio.Redis") as mock_redis_cls:
        mock_redis = AsyncMock()
        mock_redis.eval = AsyncMock(side_effect=ConnectionError("redis down"))
        mock_redis_cls.from_url.return_value = mock_redis

        limiter = AsyncRedisRateLimiter(
            "redis://test", "resource", max_requests=2, window_seconds=1.0, max_retries=3
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await limiter.acquire()

        assert mock_redis.eval.call_count == 3


@pytest.mark.asyncio
async def test_redis_limiter_close_calls_aclose():
    with patch("redis.asyncio.Redis") as mock_redis_cls:
        mock_redis = AsyncMock()
        mock_redis_cls.from_url.return_value = mock_redis

        limiter = AsyncRedisRateLimiter("redis://test", "resource")
        await limiter.close()
        mock_redis.aclose.assert_awaited_once()


# --- AsyncRedisRateLimiter against fakeredis (ml-runtime-utils-2) ---------


def _fakeredis_limiter(**kwargs):
    """AsyncRedisRateLimiter backed by fakeredis instead of a real server."""
    import fakeredis.aioredis

    from bibr.utils.rate_limiter import AsyncRedisRateLimiter

    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    with patch("redis.asyncio.Redis") as mock_redis_cls:
        mock_redis_cls.from_url.return_value = fake
        limiter = AsyncRedisRateLimiter("redis://test", "fakeredis-resource", **kwargs)
    return limiter, fake


@pytest.mark.asyncio
async def test_strict_interval_stores_whole_milliseconds():
    """The next-allowed stamp is an int-ms string, not a float (ml-2).

    Redis converts a Lua number reply to an integer, so float-second waits
    came back truncated (a 0.3 s wait became 0) and callers never slept.
    """
    limiter, fake = _fakeredis_limiter(max_requests=1, window_seconds=0.2)
    await limiter.acquire()
    stored = await fake.get("rate_limit:fakeredis-resource:next_allowed_ms")
    assert stored is not None and "." not in stored
    int(stored)  # whole milliseconds since the epoch
    # The pre-ms key must stay untouched: old releases stored epoch seconds
    # there, and sharing it across versions hangs old workers for ~1.8e12 s.
    assert await fake.get("rate_limit:fakeredis-resource:next_allowed") is None


@pytest.mark.asyncio
async def test_strict_interval_enforces_spacing():
    """Back-to-back acquires are spaced by the window (ml-2)."""
    import time

    limiter, _ = _fakeredis_limiter(max_requests=1, window_seconds=0.2)
    t0 = time.monotonic()
    await limiter.acquire()
    await limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.19


@pytest.mark.asyncio
async def test_strict_interval_ttl_covers_queued_horizon():
    """The key TTL spans the queued horizon, not just 2x the interval (ml-2).

    Concurrent callers each push the next-allowed stamp one interval out,
    so a fixed 2x-interval TTL expired the key while sleepers were still
    queued — a late caller then saw no key and fired immediately. The TTL
    now runs from the queued horizon plus a margin. Sleeps are stubbed so
    all N acquires queue from the same instant: the stamp ends N intervals
    out, and the TTL must still cover it. No wall-clock gap assertions, so
    a loaded CI runner cannot fail this test.
    """
    import asyncio
    import time

    interval = 0.2
    queued = 5
    limiter, fake = _fakeredis_limiter(max_requests=1, window_seconds=interval)
    key = "rate_limit:fakeredis-resource:next_allowed_ms"

    async def _no_sleep(_delay: float) -> None:
        return None

    with patch.object(asyncio, "sleep", _no_sleep):
        for _ in range(queued):
            await limiter.acquire()
        stored = int(await fake.get(key))
        pttl = await fake.pttl(key)
        now_ms = int(time.time() * 1000)

    horizon_ms = stored - now_ms
    assert horizon_ms >= int(queued * interval * 1000) - 50  # N intervals queued
    assert pttl >= horizon_ms, f"pttl {pttl} ms does not cover queued horizon {horizon_ms} ms"


@pytest.mark.asyncio
async def test_sliding_window_burst_then_blocks():
    """Two slots admit two immediate acquires; the third polls (ml-2 guard)."""
    import asyncio

    limiter, fake = _fakeredis_limiter(max_requests=2, window_seconds=60.0)
    await limiter.acquire()
    await limiter.acquire()
    assert await fake.zcard("rate_limit:fakeredis-resource:window") == 2
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(limiter.acquire(), timeout=0.3)


# --- 429 throttle recovery (CrossrefClient._recover_rate_limit) ---


def _resp_429(doi: str) -> httpx.Response:
    return httpx.Response(
        429, request=httpx.Request("GET", f"https://api.crossref.org/works/{doi}")
    )


def _resp_ok(doi: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"message": "ok"},
        headers={"content-type": "application/json"},
        request=httpx.Request("GET", f"https://api.crossref.org/works/{doi}"),
    )


@pytest.mark.asyncio
async def test_throttle_recovery_waits_for_quiet_period():
    """A success right after a blind 429 must NOT narrow the widened window —
    Crossref may still be throttling; recovery starts only after the quiet period."""
    client = CrossrefClient(mailto="test@example.com")
    base = client.interval

    http = await client._get_http_client()
    with (
        patch.object(
            http,
            "get",
            new_callable=AsyncMock,
            side_effect=[_resp_429("10.1000/1"), _resp_ok("10.1000/1")],
        ),
        patch("bibr.clients.crossref.asyncio.sleep", new_callable=AsyncMock),
        patch("bibr.config.Settings.redis") as mock_redis_settings,
    ):
        mock_redis_settings.url = None
        await client.works("10.1000/1")

    # 429 doubled the window; the immediate success left it alone.
    assert client.limiter.window_seconds == pytest.approx(base * 2)
    await client.close()


@pytest.mark.asyncio
async def test_throttle_recovery_halves_after_quiet_period():
    """Once the quiet period has passed, each success halves the window back
    down to the configured interval — a single early burst must not leave the
    client throttled for the life of the process."""
    client = CrossrefClient(mailto="test@example.com")
    base = client.interval

    http = await client._get_http_client()
    with (
        patch.object(
            http,
            "get",
            new_callable=AsyncMock,
            side_effect=[
                _resp_429("10.1000/1"),
                _resp_ok("10.1000/1"),
                _resp_ok("10.1000/2"),
                _resp_ok("10.1000/3"),
            ],
        ),
        patch("bibr.clients.crossref.asyncio.sleep", new_callable=AsyncMock),
        patch("bibr.config.Settings.redis") as mock_redis_settings,
    ):
        mock_redis_settings.url = None
        await client.works("10.1000/1")
        # Fast-forward past the quiet period without real sleeping.
        client._last_blind_429_monotonic -= 61.0

        await client.works("10.1000/2")
        after_first = client.limiter.window_seconds
        await client.works("10.1000/3")

    assert after_first == pytest.approx(base)  # 2x halved back to the floor
    # Further successes never narrow below the configured interval.
    assert client.limiter.window_seconds == pytest.approx(base)
    await client.close()


@pytest.mark.asyncio
async def test_throttle_recovery_never_undercuts_server_mandated_interval():
    """A server-advertised rate limit is authoritative: recovery narrows a
    429-widened window back to it, never below it."""
    client = CrossrefClient(mailto="test@example.com")

    mandated = httpx.Response(
        200,
        json={"message": "ok"},
        headers={
            "content-type": "application/json",
            "x-rate-limit-limit": "2",
            "x-rate-limit-interval": "1s",
        },
        request=httpx.Request("GET", "https://api.crossref.org/works/10.1000/1"),
    )

    http = await client._get_http_client()
    with (
        patch.object(
            http,
            "get",
            new_callable=AsyncMock,
            side_effect=[
                mandated,  # server mandates 0.5s -> floor rises
                _resp_429("10.1000/2"),  # widens to 1.0s
                _resp_ok("10.1000/2"),
                _resp_ok("10.1000/3"),
            ],
        ),
        patch("bibr.clients.crossref.asyncio.sleep", new_callable=AsyncMock),
        patch("bibr.config.Settings.redis") as mock_redis_settings,
    ):
        mock_redis_settings.url = None
        await client.works("10.1000/1")
        assert client.limiter.window_seconds == pytest.approx(0.5)

        await client.works("10.1000/2")
        assert client.limiter.window_seconds == pytest.approx(1.0)

        client._last_blind_429_monotonic -= 61.0
        await client.works("10.1000/3")

    # Recovered to the server-mandated 0.5s, not the (looser) configured rate.
    assert client.limiter.window_seconds == pytest.approx(0.5)
    await client.close()


def test_crossref_http_client_follows_redirects():
    """Crossref 301-redirects /works/{doi} for non-canonical DOI forms; the
    client must follow them instead of failing the lookup permanently."""
    client = CrossrefClient(mailto="test@example.com")
    http = client._make_http_client()
    assert http.follow_redirects is True
