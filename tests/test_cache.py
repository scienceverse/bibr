"""Tests for bibr/cache.py — ResponseCache async Redis wrapper.

The class is a thin wrapper that swallows exceptions on get/set so cache
failures never break the request path. We bypass __init__ to avoid the
``redis.asyncio.from_url`` call and inject an AsyncMock directly — this
also dodges the sys.modules MagicMock that another test installs.
"""

from unittest.mock import AsyncMock

import pytest

from bibr.cache import ResponseCache


def _make_cache(mock_redis, ttl=60, prefix="test"):
    cache = ResponseCache.__new__(ResponseCache)
    cache._redis = mock_redis
    cache._ttl = ttl
    cache._prefix = prefix
    return cache


@pytest.fixture
def mock_redis():
    return AsyncMock()


class TestResponseCacheGet:
    async def test_returns_cached_bytes(self, mock_redis):
        mock_redis.get.return_value = b"cached"
        cache = _make_cache(mock_redis)
        result = await cache.get("key1")
        assert result == b"cached"
        mock_redis.get.assert_awaited_once_with("test:key1")

    async def test_returns_none_on_miss(self, mock_redis):
        mock_redis.get.return_value = None
        cache = _make_cache(mock_redis)
        assert await cache.get("missing") is None

    async def test_returns_none_on_redis_error(self, mock_redis, caplog):
        mock_redis.get.side_effect = ConnectionError("redis down")
        cache = _make_cache(mock_redis)
        assert await cache.get("key") is None
        assert any("Cache get error" in r.message for r in caplog.records)


class TestResponseCacheSet:
    async def test_set_with_ttl(self, mock_redis):
        cache = _make_cache(mock_redis, ttl=300, prefix="api")
        await cache.set("foo", b"bar")
        mock_redis.set.assert_awaited_once_with("api:foo", b"bar", ex=300)

    async def test_swallows_redis_error(self, mock_redis, caplog):
        mock_redis.set.side_effect = ConnectionError("redis down")
        cache = _make_cache(mock_redis)
        await cache.set("foo", b"bar")  # must not raise
        assert any("Cache set error" in r.message for r in caplog.records)


class TestResponseCacheDelete:
    async def test_delete_uses_namespaced_key(self, mock_redis):
        cache = _make_cache(mock_redis, prefix="api")

        await cache.delete("foo")

        mock_redis.delete.assert_awaited_once_with("api:foo")

    async def test_swallows_redis_error(self, mock_redis, caplog):
        mock_redis.delete.side_effect = ConnectionError("redis down")
        cache = _make_cache(mock_redis)

        await cache.delete("foo")  # must not raise

        assert any("Cache delete error" in r.message for r in caplog.records)


class TestResponseCacheClose:
    async def test_close_calls_aclose(self, mock_redis):
        cache = _make_cache(mock_redis)
        await cache.close()
        mock_redis.aclose.assert_awaited_once()

    async def test_close_swallows_errors(self, mock_redis):
        mock_redis.aclose.side_effect = RuntimeError("already closed")
        cache = _make_cache(mock_redis)
        await cache.close()  # must not raise


class TestResponseCachePrefixIsolation:
    async def test_different_prefixes_dont_collide(self, mock_redis):
        cache_a = _make_cache(mock_redis, prefix="A")
        cache_b = _make_cache(mock_redis, prefix="B")
        await cache_a.set("k", b"v_a")
        await cache_b.set("k", b"v_b")
        calls = mock_redis.set.await_args_list
        assert calls[0].args[0] == "A:k"
        assert calls[1].args[0] == "B:k"


class TestResponseCacheInit:
    """Cover the actual __init__ path that calls redis.asyncio.from_url."""

    def test_init_calls_from_url_and_stores_settings(self, monkeypatch):
        import sys
        import types

        # redis is an optional dep (cache extra); skip when absent.
        redis_mod = pytest.importorskip("redis")

        captured = {}

        def fake_from_url(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return "STUB_REDIS"

        # ``import redis.asyncio as aioredis`` resolves to
        # ``sys.modules["redis"].asyncio``; the importorskip above ensures
        # the parent is in sys.modules so we can patch the attribute even
        # when this test runs in isolation.
        fake_mod = types.ModuleType("redis.asyncio")
        fake_mod.from_url = fake_from_url
        monkeypatch.setitem(sys.modules, "redis.asyncio", fake_mod)
        monkeypatch.setattr(redis_mod, "asyncio", fake_mod, raising=False)

        cache = ResponseCache(redis_url="redis://x:6379", ttl_seconds=42, prefix="P")
        assert captured["url"] == "redis://x:6379"
        assert captured["kwargs"] == {
            "decode_responses": False,
            "socket_connect_timeout": 2.0,
            "socket_timeout": 5.0,
            "socket_keepalive": True,
            "health_check_interval": 30,
        }
        assert cache._ttl == 42
        assert cache._prefix == "P"
        assert cache._redis == "STUB_REDIS"
