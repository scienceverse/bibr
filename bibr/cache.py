"""
Redis Response Cache

Optional caching layer for API responses. Gracefully skips when Redis is
unavailable.

``ResponseCache`` also serves as the shared core for
``bibr.clients.response_cache.RedisResponseCache`` (the JSON-level cache
used by the CrossRef client's tier-2 cache). Connection setup, the TTL'd
Redis ``set``, error-swallowing ``get``/``set``, and ``close`` all live here
in exactly one place; subclasses override ``_connect``, ``_full_key``, the
``_log_*`` hooks, and/or the public ``get``/``set``/``close`` methods to
layer serialization or bookkeeping on top without duplicating the
Redis-handling logic.
"""

import logging
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Sentinel distinguishing "Redis raised (already logged + swallowed)" from a
# genuine cache miss (``None``), so subclasses that track hit/miss stats can
# tell the two apart without re-implementing the error handling below.
_REDIS_ERROR = object()

_RENEW_LEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('expire', KEYS[1], ARGV[2])
end
return 0
"""

_RELEASE_LEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""


@dataclass(frozen=True)
class RedisLease:
    redis: object
    key: str
    token: str
    ttl_seconds: int

    async def renew(self) -> bool:
        result = await self.redis.eval(
            _RENEW_LEASE_SCRIPT, 1, self.key, self.token, self.ttl_seconds
        )
        return bool(result)

    async def release(self) -> bool:
        result = await self.redis.eval(_RELEASE_LEASE_SCRIPT, 1, self.key, self.token)
        return bool(result)


class ResponseCache:
    """Async Redis wrapper for caching serialized API responses (bytes-level)."""

    _decode_responses: bool = False

    #: Connection-establishment and per-command socket budgets. Without them
    #: redis-py waits forever on a Redis that accepts the TCP connection but
    #: never answers (fsync stall, blackholed route), and every serve request
    #: that touches the cache hangs while holding its admission slot.
    DEFAULT_CONNECT_TIMEOUT: float = 2.0
    DEFAULT_SOCKET_TIMEOUT: float = 5.0

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        prefix: str,
        *,
        connect_timeout: float | None = None,
        socket_timeout: float | None = None,
    ):
        self._connect_timeout = (
            self.DEFAULT_CONNECT_TIMEOUT if connect_timeout is None else float(connect_timeout)
        )
        self._socket_timeout = (
            self.DEFAULT_SOCKET_TIMEOUT if socket_timeout is None else float(socket_timeout)
        )
        self._redis = self._connect(redis_url)
        self._ttl = ttl_seconds
        self._prefix = prefix

    def _redis_client_kwargs(self) -> dict:
        """Keyword arguments every Redis client built by this cache must carry."""
        return {
            "decode_responses": self._decode_responses,
            "socket_connect_timeout": self._connect_timeout,
            "socket_timeout": self._socket_timeout,
            "socket_keepalive": True,
            # Ping idle pooled connections before reuse so a half-open socket
            # left behind by a Redis restart fails fast, not on the first command.
            "health_check_interval": 30,
        }

    def _connect(self, redis_url: str):
        import redis.asyncio as aioredis

        return aioredis.from_url(redis_url, **self._redis_client_kwargs())

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    def _log_get_error(self, key: str, exc: Exception) -> None:  # noqa: ARG002
        logger.warning("Cache get error: %s", exc)

    def _log_set_error(self, key: str, exc: Exception) -> None:  # noqa: ARG002
        logger.warning("Cache set error: %s", exc)

    def _log_close_error(self, exc: Exception) -> None:
        pass  # ResponseCache silently swallows close errors.

    async def _get_raw(self, key: str):
        """Fetch the raw value stored for ``key``.

        Returns the stored value, ``None`` on a genuine cache miss, or the
        ``_REDIS_ERROR`` sentinel if Redis raised (already logged via
        ``_log_get_error``).
        """
        try:
            return await self._redis.get(self._full_key(key))
        except Exception as e:
            self._log_get_error(key, e)
            return _REDIS_ERROR

    async def _set_raw(self, key: str, raw, *, ttl_seconds: int | None = None) -> None:
        """Store ``raw`` for ``key`` with TTL (the cache's own unless *ttl_seconds*).

        Errors are logged and swallowed.
        """
        try:
            await self._redis.set(self._full_key(key), raw, ex=ttl_seconds or self._ttl)
        except Exception as e:
            self._log_set_error(key, e)

    async def get(self, key: str) -> bytes | None:
        """Return cached bytes, or None on miss / error."""
        raw = await self._get_raw(key)
        return None if raw is _REDIS_ERROR else raw

    async def set(self, key: str, value: bytes) -> None:
        """Store bytes with TTL. Silently ignores errors."""
        await self._set_raw(key, value)

    async def delete(self, key: str) -> None:
        """Delete cached bytes. Silently ignores errors."""
        try:
            await self._redis.delete(self._full_key(key))
        except Exception as e:
            logger.warning("Cache delete error: %s", e)

    async def try_acquire_lease(
        self, key: str, *, ttl_seconds: int, ownership_id: str | None = None
    ) -> RedisLease | None:
        """Atomically acquire a token-owned flight lease, or return ``None``."""
        token = ownership_id or uuid.uuid4().hex
        lease_key = self._flight_key(key)
        acquired = await self._redis.set(lease_key, token, nx=True, ex=ttl_seconds)
        if not acquired:
            return None
        return RedisLease(self._redis, lease_key, token, ttl_seconds)

    def _flight_key(self, key: str) -> str:
        return f"{self._prefix}:flight:{key}"

    async def lease_alive(self, key: str) -> bool:
        """Whether another replica still holds the flight lease for ``key``.

        Raises on Redis errors — waiters treat that as fail-open (extract
        normally) rather than as an owner death.
        """
        return bool(await self._redis.exists(self._flight_key(key)))

    async def close(self) -> None:
        """Shut down the Redis connection."""
        try:
            await self._redis.aclose()
        except Exception as e:
            self._log_close_error(e)
