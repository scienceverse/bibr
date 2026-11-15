"""Shared, cross-process response cache for client metadata lookups.

Backs the CrossRef client's tier-2 cache: the in-process LRU is tier 1, this
Redis-backed store is tier 2 (shared across bibr-serve and the workers,
survives restarts). Entries are reconstructible public metadata, so every
operation degrades gracefully — a Redis outage silently falls back to the
upstream fetch rather than failing enrichment.

Subclasses ``bibr.cache.ResponseCache`` (the shared core: connection setup,
TTL'd set, error-swallowing get/set, close) to add JSON (de)serialization
and hit/miss bookkeeping on top.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from bibr.cache import _REDIS_ERROR, ResponseCache

logger = logging.getLogger(__name__)


class RedisResponseCache(ResponseCache):
    """JSON response cache over Redis. Best-effort: never raises on Redis errors."""

    _decode_responses = True

    def __init__(
        self,
        redis_url: str,
        key_prefix: str = "crossref:cache:",
        ttl_seconds: int = 2_592_000,
        *,
        connect_timeout: float | None = None,
        socket_timeout: float | None = None,
    ):
        super().__init__(
            redis_url,
            ttl_seconds,
            key_prefix,
            connect_timeout=connect_timeout,
            socket_timeout=socket_timeout,
        )
        self.hits = 0
        self.misses = 0
        # The CrossRef client that holds this cache is process-wide, so it
        # outlives one ``asyncio.run`` and serves threads that run their own
        # loops, and a redis.asyncio connection belongs to the loop that opened
        # it: on any other loop its first command fails with "Event loop is
        # closed". The client built above serves the first loop; every other
        # loop gets its own.
        self._redis_url = redis_url
        self._loop_clients: dict[asyncio.AbstractEventLoop, Any] = {}
        self._loop_clients_lock = threading.Lock()
        self._redis_claimed = False

    def _connect(self, redis_url: str):
        from redis.asyncio import Redis

        return Redis.from_url(redis_url, **self._redis_client_kwargs())

    def _client(self):
        loop = asyncio.get_running_loop()
        client = self._loop_clients.get(loop)
        if client is not None:
            return client
        with self._loop_clients_lock:
            client = self._loop_clients.get(loop)
            if client is None:
                for other in list(self._loop_clients):
                    if other.is_closed():
                        self._loop_clients.pop(other, None)
                client = self._connect(self._redis_url) if self._redis_claimed else self._redis
                self._redis_claimed = True
                self._loop_clients[loop] = client
        return client

    def _full_key(self, key: str) -> str:
        return self._prefix + key

    def _log_get_error(self, key: str, exc: Exception) -> None:
        logger.debug("Redis cache get failed for %s: %s", key, exc)

    def _log_set_error(self, key: str, exc: Exception) -> None:
        logger.debug("Redis cache set failed for %s: %s", key, exc)

    def _log_close_error(self, exc: Exception) -> None:
        logger.debug("Redis cache close failed: %s", exc)

    async def get(self, key: str) -> dict[str, Any] | None:
        raw = await self._get_raw(key)
        if raw is _REDIS_ERROR:
            return None
        if raw is None:
            self.misses += 1
            return None
        try:
            value = json.loads(raw)
        except (ValueError, TypeError) as exc:
            logger.debug("Redis cache decode failed for %s: %s", key, exc)
            self.misses += 1
            return None
        self.hits += 1
        return value

    async def set(self, key: str, value: dict[str, Any], *, ttl_seconds: int | None = None) -> None:
        await self._set_raw(key, json.dumps(value), ttl_seconds=ttl_seconds)

    async def close(self) -> None:
        if self.hits or self.misses:
            total = self.hits + self.misses
            logger.info(
                "Redis response cache: %d/%d hits (%.0f%%)",
                self.hits,
                total,
                100.0 * self.hits / total,
            )
        await super().close()
