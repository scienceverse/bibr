import asyncio
import collections
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class _Waitroom:
    """One event loop's waiters: their condition, and whether one holds the timer."""

    __slots__ = ("cond", "timer_armed")

    def __init__(self) -> None:
        self.cond = asyncio.Condition()
        self.timer_armed = False


class AsyncLocalRateLimiter:
    """In-process rate limiter using an asyncio.Condition wait queue.

    Sliding-window: at most ``max_requests`` slots within any ``window_seconds``.
    Only one waiter per event loop sleeps until the oldest slot ages out; the
    others park on the condition and are woken one at a time, as each taken
    slot hands the turn on — so a window boundary wakes one coroutine, not
    every waiter.
    """

    def __init__(
        self,
        resource_id: str,
        max_requests: int = 1,
        window_seconds: float = 1.0,
    ):
        self.resource_id = resource_id
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: collections.deque[float] = collections.deque()
        # ``asyncio.Condition`` binds to the running loop on first wait, but
        # one limiter can serve several loops: a singleton-held limiter reused
        # from a second ``asyncio.run()``, or a process-wide client used from
        # threads that each run their own loop. Each loop gets its own
        # condition; the timestamp window is shared (rate history isn't
        # loop-bound), so its bookkeeping is guarded by a thread lock that is
        # never held across an await.
        self._waitrooms: dict[asyncio.AbstractEventLoop, _Waitroom] = {}
        self._window_lock = threading.Lock()

    def _waitroom(self) -> _Waitroom:
        loop = asyncio.get_running_loop()
        room = self._waitrooms.get(loop)
        if room is None:
            # A closed loop's waiters are gone with it.
            for other in list(self._waitrooms):
                if other.is_closed():
                    self._waitrooms.pop(other, None)
            room = self._waitrooms[loop] = _Waitroom()
        return room

    def _take_slot(self) -> float | None:
        """Take a slot if the window has room, else return seconds until one frees."""
        with self._window_lock:
            now = time.monotonic()
            cutoff = now - self.window_seconds
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()

            if len(self._timestamps) < self.max_requests:
                self._timestamps.append(now)
                return None
            return self._timestamps[0] + self.window_seconds - now

    async def acquire(self) -> None:
        """Acquire a slot. Blocks if the window is full; resumes when a slot ages out."""
        room = self._waitroom()
        cond = room.cond
        async with cond:
            while True:
                wait = self._take_slot()
                if wait is None:
                    cond.notify(1)  # Wake the next waiter so they can re-check.
                    return

                if room.timer_armed:
                    # Another waiter already sleeps until the next slot frees;
                    # wait for the turn it hands on rather than setting a
                    # timer for the same instant.
                    try:
                        await cond.wait()
                    except asyncio.CancelledError:
                        cond.notify(1)  # Don't swallow a hand-off meant for us.
                        raise
                    continue

                logger.debug(f"Rate limiting {self.resource_id}: sleeping for {wait:.3f}s")
                room.timer_armed = True
                try:
                    await asyncio.wait_for(cond.wait(), timeout=max(wait, 0.001))
                except TimeoutError:
                    pass  # Timer fired; loop and re-check.
                except asyncio.CancelledError:
                    cond.notify(1)  # Hand the timer on to the next waiter.
                    raise
                finally:
                    room.timer_armed = False

    async def close(self) -> None:
        """No-op for interface compatibility with AsyncRedisRateLimiter."""
        return None


class AsyncRedisRateLimiter:
    """
    Distributed rate limiter using Redis.
    Supports two modes:
    1. Strict Interval (max_requests=1): Ensures a minimum time interval between requests.
    2. Sliding Window (max_requests>1): Ensures max N requests in W seconds.

    Both scripts read the Redis server's clock (``TIME``), so hosts whose wall
    clocks disagree still share one timeline. While Redis is unreachable, each
    process enforces the same budget in-process and tries Redis again every
    ``fallback_seconds``.
    """

    def __init__(
        self,
        redis_url: str,
        resource_id: str,
        max_requests: int = 1,
        window_seconds: float = 1.0,
        max_retries: int = 3,
        connect_timeout: float = 2.0,
        socket_timeout: float = 5.0,
        fallback_seconds: float = 30.0,
    ):
        from redis.asyncio import Redis

        def connect() -> "Redis":
            # Bounded like bibr.cache.ResponseCache: a Redis that accepts the
            # connection but never answers must fail the limiter, not hang it.
            return Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=connect_timeout,
                socket_timeout=socket_timeout,
                socket_keepalive=True,
                health_check_interval=30,
            )

        self._connect = connect
        self.redis = connect()
        # redis.asyncio connections belong to the loop that opened them, while
        # a process-wide client's limiter serves every loop (successive
        # asyncio.run() calls, threads running their own loop). The client
        # built above goes to the first loop that uses it; others get their own.
        self._loop_clients: dict[asyncio.AbstractEventLoop, Redis] = {}
        self._unclaimed: Redis | None = self.redis
        self._clients_lock = threading.Lock()
        self.resource_id = resource_id
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_retries = max_retries
        self.fallback_seconds = fallback_seconds
        # Outage memory: without it every acquire re-paid the retries (2 s,
        # or ~17 s against a Redis that accepts but never answers) and then
        # ran unlimited.
        self._fallback = AsyncLocalRateLimiter(resource_id, max_requests, window_seconds)
        self._redis_down = False
        self._redis_retry_at = 0.0

    def _client(self) -> "Redis":
        """The Redis client for the running event loop."""
        loop = asyncio.get_running_loop()
        client = self._loop_clients.get(loop)
        if client is None:
            with self._clients_lock:
                client = self._loop_clients.get(loop)
                if client is None:
                    # A closed loop's connections died with it.
                    for other in list(self._loop_clients):
                        if other.is_closed():
                            self._loop_clients.pop(other, None)
                    client = self._unclaimed if self._unclaimed is not None else self._connect()
                    self._unclaimed = None
                    self._loop_clients[loop] = client
        return client

    def _use_fallback(self) -> bool:
        """True while a Redis outage's cooldown runs.

        The first caller past the cooldown claims the next Redis attempt by
        pushing the deadline out, so concurrent callers stay in-process
        instead of all paying a timeout against a Redis that may still be down.
        """
        if not self._redis_down:
            return False
        now = time.monotonic()
        if now < self._redis_retry_at:
            return True
        self._redis_retry_at = now + self.fallback_seconds
        return False

    def _mark_redis_down(self, exc: Exception) -> None:
        if not self._redis_down:
            logger.warning(
                "Redis unavailable for %s rate limiting (%s); enforcing the limit "
                "in-process and retrying Redis every %.0fs.",
                self.resource_id,
                exc,
                self.fallback_seconds,
            )
        self._redis_down = True
        self._redis_retry_at = time.monotonic() + self.fallback_seconds

    def _mark_redis_up(self) -> None:
        if self._redis_down:
            self._redis_down = False
            logger.info("Redis rate limiting for %s restored.", self.resource_id)

    async def _redis_failed(self, exc: Exception, retries: int) -> bool:
        """Handle a failed Redis call; True once the caller has fallen back."""
        # A known outage is not retried: the cooldown already said when to.
        if self._redis_down or retries >= self.max_retries:
            self._mark_redis_down(exc)
            await self._acquire_fallback()
            return True
        logger.error(f"Redis rate limit error: {exc}. Sleeping 1s and retrying.")
        await asyncio.sleep(1)
        return False

    async def _acquire_fallback(self) -> None:
        # Follow the live window: callers retune it (Crossref's 429 throttle).
        self._fallback.window_seconds = self.window_seconds
        await self._fallback.acquire()

    async def acquire(self):
        """
        Acquire a token from the bucket. Blocks if necessary.
        """
        if self._use_fallback():
            await self._acquire_fallback()
        elif self.max_requests == 1:
            await self._acquire_strict_interval()
        else:
            await self._acquire_sliding_window()

    # Redis < 5 replicates a script verbatim and refuses writes after a
    # non-deterministic TIME unless effects replication is switched on; later
    # versions always replicate effects (and fakeredis lacks the call).
    _REDIS_NOW_MS = """
    if redis.replicate_commands then redis.replicate_commands() end
    local t = redis.call('TIME')
    local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
    """

    # Returns the whole milliseconds the caller must wait before its slot.
    _STRICT_INTERVAL_SCRIPT = (
        """
    local key = KEYS[1]
    local interval_ms = tonumber(ARGV[1])
    """
        + _REDIS_NOW_MS
        + """
    local next_allowed = tonumber(redis.call('GET', key) or 0)

    local wait_ms = 0
    if next_allowed > now_ms then
        wait_ms = next_allowed - now_ms
    end

    local new_next = math.max(now_ms, next_allowed) + interval_ms
    redis.call('SET', key, new_next)
    redis.call('PEXPIRE', key, math.ceil(new_next - now_ms + interval_ms))

    return wait_ms
    """
    )

    async def _acquire_strict_interval(self):
        """
        Enforce strict time interval between requests using Lua script for atomicity.

        All times cross into Lua as whole MILLISECONDS (ints). Passing float
        seconds would break spacing: Redis converts a Lua number reply to an
        integer, so a fractional-second wait (e.g. 0.3 s) came back as 0 and
        the caller never slept.
        """
        # ``_ms`` suffix: the stamp is epoch milliseconds, while older
        # releases stored epoch seconds under ``next_allowed``. A distinct
        # key keeps the two from reading each other's values when old and
        # new processes share one Redis (rolling deploy).
        key = f"rate_limit:{self.resource_id}:next_allowed_ms"
        retries = 0
        while True:
            try:
                interval_ms = int(round(self.window_seconds * 1000))
                wait_ms = await self._client().eval(
                    self._STRICT_INTERVAL_SCRIPT, 1, key, interval_ms
                )
            except Exception as e:  # noqa: BLE001 — any Redis failure falls back
                retries += 1
                if await self._redis_failed(e, retries):
                    return
                continue
            self._mark_redis_up()

            # Lua returns whole milliseconds; back to seconds for asyncio.
            wait_time = float(wait_ms) / 1000.0
            if wait_time > 0:
                logger.debug(f"Rate limiting {self.resource_id}: sleeping for {wait_time:.3f}s")
                await asyncio.sleep(wait_time)
            return

    # Lua script that atomically prunes, checks, and conditionally inserts.
    # Returns 1 if a slot was acquired, 0 if the window is full. Scores stay
    # in seconds (whole milliseconds), the unit older releases wrote.
    _SLIDING_WINDOW_SCRIPT = (
        """
    local key = KEYS[1]
    local max_requests = tonumber(ARGV[1])
    local window_seconds = tonumber(ARGV[2])
    local member = ARGV[3]
    local expire_seconds = tonumber(ARGV[4])
    """
        + _REDIS_NOW_MS
        + """
    local now = now_ms / 1000

    redis.call('ZREMRANGEBYSCORE', key, 0, now - window_seconds)
    local current = redis.call('ZCARD', key)

    if current < max_requests then
        redis.call('ZADD', key, now, member)
        redis.call('EXPIRE', key, expire_seconds)
        return 1
    end
    return 0
    """
    )

    async def _acquire_sliding_window(self):
        """
        Sliding window log using an atomic Lua script to avoid TOCTOU races.
        """
        key = f"rate_limit:{self.resource_id}:window"
        retries = 0
        while True:
            try:
                acquired = await self._client().eval(
                    self._SLIDING_WINDOW_SCRIPT,
                    1,
                    key,
                    self.max_requests,
                    self.window_seconds,
                    uuid.uuid4().hex,
                    int(self.window_seconds) + 10,
                )
            except Exception as e:  # noqa: BLE001 — any Redis failure falls back
                retries += 1
                if await self._redis_failed(e, retries):
                    return
                continue
            self._mark_redis_up()

            if acquired:
                return
            await asyncio.sleep(0.1)

    async def close(self):
        """Close this loop's Redis client, and the initial one if no loop used it.

        Clients other running loops opened stay with those loops.
        """
        loop = asyncio.get_running_loop()
        with self._clients_lock:
            clients = [self._loop_clients.pop(loop, None), self._unclaimed]
            self._unclaimed = None
        for client in clients:
            if client is not None:
                await client.aclose()
