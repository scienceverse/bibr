import asyncio
import collections
import logging
import time

logger = logging.getLogger(__name__)


class AsyncLocalRateLimiter:
    """In-process rate limiter using an asyncio.Condition wait queue.

    Sliding-window: at most ``max_requests`` slots within any ``window_seconds``.
    A single coordinator notifies exactly one waiter when the oldest slot ages
    out — no per-waiter polling, no thundering-herd wake on each window
    boundary.
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
        # ``asyncio.Condition`` binds to the running loop on first wait. A
        # singleton-held limiter reused from a second ``asyncio.run()`` would
        # raise ``RuntimeError: ... bound to a different event loop``, so the
        # condition is recreated whenever the loop changes (same pattern as
        # AsyncCircuitBreaker). The timestamp window deliberately survives the
        # rebind — rate history isn't loop-bound.
        self._cond: asyncio.Condition | None = None
        self._cond_loop_id: int | None = None

    def _ensure_loop_state(self) -> asyncio.Condition:
        loop_id = id(asyncio.get_running_loop())
        if self._cond is None or self._cond_loop_id != loop_id:
            self._cond = asyncio.Condition()
            self._cond_loop_id = loop_id
        return self._cond

    async def acquire(self) -> None:
        """Acquire a slot. Blocks if the window is full; resumes when a slot ages out."""
        cond = self._ensure_loop_state()
        async with cond:
            while True:
                now = time.monotonic()
                cutoff = now - self.window_seconds
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    cond.notify(1)  # Wake the next waiter so they can re-check.
                    return

                wait = self._timestamps[0] + self.window_seconds - now
                logger.debug(f"Rate limiting {self.resource_id}: sleeping for {wait:.3f}s")
                try:
                    await asyncio.wait_for(cond.wait(), timeout=max(wait, 0.001))
                except TimeoutError:
                    pass  # Timer fired; loop and re-check.

    async def close(self) -> None:
        """No-op for interface compatibility with AsyncRedisRateLimiter."""
        return None


class AsyncRedisRateLimiter:
    """
    Distributed rate limiter using Redis.
    Supports two modes:
    1. Strict Interval (max_requests=1): Ensures a minimum time interval between requests.
    2. Sliding Window (max_requests>1): Ensures max N requests in W seconds.
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
    ):
        from redis.asyncio import Redis

        # Bounded like bibr.cache.ResponseCache: a Redis that accepts the
        # connection but never answers must fail the limiter, not hang it.
        self.redis = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=connect_timeout,
            socket_timeout=socket_timeout,
            socket_keepalive=True,
            health_check_interval=30,
        )
        self.resource_id = resource_id
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_retries = max_retries

    async def acquire(self):
        """
        Acquire a token from the bucket. Blocks if necessary.
        """
        if self.max_requests == 1:
            await self._acquire_strict_interval()
        else:
            await self._acquire_sliding_window()

    async def _acquire_strict_interval(self):
        """
        Enforce strict time interval between requests using Lua script for atomicity.

        All times cross into Lua as whole MILLISECONDS (ints). Passing float
        seconds would break spacing: Redis converts a Lua number reply to an
        integer, so a fractional-second wait (e.g. 0.3 s) came back as 0 and
        the caller never slept.
        """
        script = """
        local key = KEYS[1]
        local interval_ms = tonumber(ARGV[1])
        local now_ms = tonumber(ARGV[2])

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
        retries = 0
        while True:
            try:
                now_ms = int(time.time() * 1000)
                interval_ms = int(round(self.window_seconds * 1000))
                # ``_ms`` suffix: the stamp is epoch milliseconds, while older
                # releases stored epoch seconds under ``next_allowed``. A
                # distinct key keeps the two from reading each other's values
                # when old and new processes share one Redis (rolling deploy).
                key = f"rate_limit:{self.resource_id}:next_allowed_ms"
                # Using eval directly on the connection
                wait_ms = await self.redis.eval(script, 1, key, interval_ms, now_ms)

                # Lua returns whole milliseconds; back to seconds for asyncio.
                wait_time = float(wait_ms) / 1000.0

                if wait_time > 0:
                    logger.debug(f"Rate limiting {self.resource_id}: sleeping for {wait_time:.3f}s")
                    await asyncio.sleep(wait_time)
                return
            except Exception as e:
                retries += 1
                if retries >= self.max_retries:
                    logger.warning(
                        f"Redis unavailable after {retries} retries for {self.resource_id}. "
                        f"Proceeding without rate limiting."
                    )
                    return
                logger.error(f"Redis rate limit error: {e}. Sleeping 1s and retrying.")
                await asyncio.sleep(1)

    # Lua script that atomically prunes, checks, and conditionally inserts.
    # Returns 1 if a slot was acquired, 0 if the window is full.
    _SLIDING_WINDOW_SCRIPT = """
    local key = KEYS[1]
    local max_requests = tonumber(ARGV[1])
    local window_start = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local member = ARGV[4]
    local expire_seconds = tonumber(ARGV[5])

    redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
    local current = redis.call('ZCARD', key)

    if current < max_requests then
        redis.call('ZADD', key, now, member)
        redis.call('EXPIRE', key, expire_seconds)
        return 1
    end
    return 0
    """

    async def _acquire_sliding_window(self):
        """
        Sliding window log using an atomic Lua script to avoid TOCTOU races.
        """
        key = f"rate_limit:{self.resource_id}:window"
        retries = 0
        while True:
            try:
                # Whole milliseconds: the strict-interval path reports waits in
                # integer ms (Redis truncates Lua number replies), so quantize
                # scores here for stable pruning and comparison.
                now = int(time.time() * 1000) / 1000.0
                window_start = now - self.window_seconds
                member = f"{now}:{time.time_ns()}"
                expire_seconds = int(self.window_seconds) + 10

                acquired = await self.redis.eval(
                    self._SLIDING_WINDOW_SCRIPT,
                    1,
                    key,
                    self.max_requests,
                    window_start,
                    now,
                    member,
                    expire_seconds,
                )

                if acquired:
                    return
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                retries += 1
                if retries >= self.max_retries:
                    logger.warning(
                        f"Redis unavailable after {retries} retries for {self.resource_id}. "
                        f"Proceeding without rate limiting."
                    )
                    return
                logger.error(f"Redis rate limit error: {e}. Sleeping 1s and retrying.")
                await asyncio.sleep(1)

    async def close(self):
        await self.redis.aclose()
