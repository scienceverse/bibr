"""Non-positive Redis TTLs mean "no expiry", not EX 0 (core-api-15).

Redis rejects ``SET ... EX 0`` with "invalid expire time"; the error was
swallowed per request, so ``CACHE_TTL_SECONDS=0`` silently disabled the
whole result cache. A fake Redis records the ``ex`` kwarg — no server.
"""

from unittest.mock import AsyncMock

from bibr.cache import ResponseCache


def _make_cache(mock_redis, ttl):
    cache = ResponseCache.__new__(ResponseCache)
    cache._redis = mock_redis
    cache._ttl = ttl
    cache._prefix = "test"
    return cache


async def test_zero_cache_ttl_sends_no_expiry():
    redis = AsyncMock()
    await _make_cache(redis, ttl=0).set("k", b"v")
    redis.set.assert_awaited_once_with("test:k", b"v", ex=None)


async def test_negative_ttl_sends_no_expiry():
    redis = AsyncMock()
    await _make_cache(redis, ttl=-5).set("k", b"v")
    redis.set.assert_awaited_once_with("test:k", b"v", ex=None)


async def test_positive_ttl_unchanged():
    redis = AsyncMock()
    await _make_cache(redis, ttl=300).set("k", b"v")
    redis.set.assert_awaited_once_with("test:k", b"v", ex=300)


async def test_explicit_per_call_ttl_wins():
    redis = AsyncMock()
    cache = _make_cache(redis, ttl=300)
    await cache._set_raw("k", b"v", ttl_seconds=60)
    redis.set.assert_awaited_once_with("test:k", b"v", ex=60)
    await cache._set_raw("k", b"v", ttl_seconds=0)
    redis.set.assert_awaited_with("test:k", b"v", ex=None)
