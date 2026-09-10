from unittest.mock import AsyncMock

import pytest

from bibr.cache import ResponseCache


def _cache(redis):
    cache = ResponseCache.__new__(ResponseCache)
    cache._redis = redis
    cache._ttl = 60
    cache._prefix = "test"
    return cache


async def test_try_acquire_lease_is_atomic_and_namespaced():
    redis = AsyncMock()
    redis.set.return_value = True
    lease = await _cache(redis).try_acquire_lease("paper", ttl_seconds=30, ownership_id="owner")
    assert lease is not None
    assert lease.token == "owner"
    redis.set.assert_awaited_once_with("test:flight:paper", "owner", nx=True, ex=30)


async def test_try_acquire_returns_none_when_owned():
    redis = AsyncMock()
    redis.set.return_value = None
    assert await _cache(redis).try_acquire_lease("paper", ttl_seconds=30) is None


async def test_renew_and_release_are_token_safe():
    redis = AsyncMock()
    redis.set.return_value = True
    redis.eval.side_effect = [1, 1]
    lease = await _cache(redis).try_acquire_lease("paper", ttl_seconds=30, ownership_id="owner")
    assert lease is not None
    assert await lease.renew() is True
    assert await lease.release() is True
    renew, release = redis.eval.await_args_list
    assert renew.args[1:] == (1, "test:flight:paper", "owner", 30)
    assert release.args[1:] == (1, "test:flight:paper", "owner")
    assert "get" in renew.args[0].lower()
    assert "expire" in renew.args[0].lower()
    assert "del" in release.args[0].lower()


async def test_non_owner_cannot_renew_or_release():
    redis = AsyncMock()
    redis.set.return_value = True
    redis.eval.side_effect = [0, 0]
    lease = await _cache(redis).try_acquire_lease("paper", ttl_seconds=30, ownership_id="stale")
    assert lease is not None
    assert await lease.renew() is False
    assert await lease.release() is False


async def test_redis_errors_propagate_to_fail_open_caller():
    redis = AsyncMock()
    redis.set.side_effect = ConnectionError("down")
    with pytest.raises(ConnectionError, match="down"):
        await _cache(redis).try_acquire_lease("paper", ttl_seconds=30)
