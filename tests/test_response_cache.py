"""Tests for bibr.clients.response_cache.RedisResponseCache."""

import json
from unittest.mock import AsyncMock, patch

import pytest

redis = pytest.importorskip("redis", reason="redis not installed (optional [cache] dependency)")

from bibr.clients.response_cache import RedisResponseCache


def _make_cache(mock_redis, **kwargs):
    with patch("redis.asyncio.Redis") as cls:
        cls.from_url.return_value = mock_redis
        return RedisResponseCache("redis://x:6379/0", **kwargs)


async def test_get_miss_returns_none_and_counts():
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    cache = _make_cache(mock_redis)

    assert await cache.get("works:10.1/x") is None
    assert cache.misses == 1
    assert cache.hits == 0
    mock_redis.get.assert_awaited_once_with("crossref:cache:works:10.1/x")


async def test_get_hit_decodes_json_and_counts():
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=json.dumps({"message": "ok"}))
    cache = _make_cache(mock_redis)

    assert await cache.get("works:10.1/x") == {"message": "ok"}
    assert cache.hits == 1
    assert cache.misses == 0


async def test_set_writes_prefixed_key_with_ttl():
    mock_redis = AsyncMock()
    cache = _make_cache(mock_redis, ttl_seconds=99)

    await cache.set("works:10.1/x", {"message": "ok"})
    mock_redis.set.assert_awaited_once_with(
        "crossref:cache:works:10.1/x", json.dumps({"message": "ok"}), ex=99
    )


async def test_get_degrades_to_none_on_redis_error():
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(side_effect=ConnectionError("redis down"))
    cache = _make_cache(mock_redis)

    assert await cache.get("works:10.1/x") is None
    assert cache.misses == 0  # connection errors are a bypass, not a miss
    assert cache.hits == 0


async def test_get_decode_failure_counts_as_miss():
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value="not-json{")
    cache = _make_cache(mock_redis)

    assert await cache.get("works:10.1/x") is None
    assert cache.misses == 1
    assert cache.hits == 0


async def test_set_swallows_redis_error():
    mock_redis = AsyncMock()
    mock_redis.set = AsyncMock(side_effect=ConnectionError("redis down"))
    cache = _make_cache(mock_redis)

    # Must not raise.
    await cache.set("works:10.1/x", {"message": "ok"})
