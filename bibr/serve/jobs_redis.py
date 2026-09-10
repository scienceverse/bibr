"""Redis-backed :class:`~bibr.serve.jobs.JobStore` for multi-replica ``bibr serve``.

Several ``bibr serve`` replicas behind one load balancer point at the same
Redis (``JOBS_STORE=redis``); each keeps executing the jobs it accepted, but
status, results, and the active-job cap become global, so a poll that lands on
a different replica than the upload still answers.

**Keys** (under ``JOBS_KEY_PREFIX``, default ``bibr:jobs``):

- ``{prefix}:job:{id}`` — a hash: ``status``, ``filename``, ``created_at`` /
  ``started_at`` / ``finished_at`` (wall-clock epoch seconds), ``http_status``,
  ``error`` (JSON), ``replica``, ``result_size`` (encoded JSON bytes).
- ``{prefix}:result:{id}`` — the paper JSON, rendered once at completion and
  zlib-compressed. Kept apart from the hash so a status poll never drags the
  body across the wire.
- ``{prefix}:active`` — SET of queued/running ids, the global cap.
- ``{prefix}:finished`` — ZSET of finished ids scored by ``finished_at``, the
  global ``JOBS_MAX_RETAINED`` / ``JOBS_MAX_RETAINED_BYTES`` eviction order.

**Atomicity.** ``create`` is one Lua script: it reconciles the active set
against the job hashes (dropping ids whose record is gone), counts, refuses
past ``JOBS_MAX_ACTIVE`` or admits and writes the record — no window between
the check and the insert, whichever replica runs it. Lua was chosen over
WATCH/MULTI because the reconcile-then-count step is a read-modify-write
over a whole set, which WATCH cannot express without retries. The finish
transition (status, TTLs, result, cap release, retention pruning) is a second
script, so pruning runs under the same global order every replica sees.

**TTLs.** Finished job + result keys expire after ``JOBS_TTL_SECONDS``, the
same window the memory store uses, and ``get`` treats a record past that
window as gone even before Redis reaps it. Active records carry a generous
safety TTL (24 h by default): if a replica dies mid-job its record lingers
as ``queued``/``running`` until then, after which the next ``create`` drops
the id from the active set and the cap slot returns. A job owned by another
replica is reported as-is; no replica ever executes another's job.

**Failure semantics.** Every Redis touch runs under ``asyncio.timeout`` with
the connect+socket budget from ``REDIS_CONNECT_TIMEOUT_SECONDS`` /
``REDIS_SOCKET_TIMEOUT_SECONDS`` (the client carries the same socket timeouts
as a first fence). ``create``/``get`` raise
:class:`~bibr.serve.jobs.JobStoreUnavailableError`, which the routes turn into
``503``; ``set_*``/``discard`` log the job id and swallow, so a Redis outage
never unwinds the dispatcher worker loop — the TTLs clean up whatever was
left half-written.

**Clocks.** The memory store measures durations and TTL on the monotonic
clock. Replicas cannot share a monotonic clock, so this store records wall
timestamps (``time.time`` by default, injectable) and derives ``duration_ms``
from them; a wall-clock jump on the executing replica shows up in that one
job's duration, nothing else.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import zlib
from collections.abc import Callable
from typing import Any

import redis.asyncio as aioredis

from bibr.config import Settings
from bibr.serve.jobs import (
    Job,
    JobCapacityError,
    JobStoreUnavailableError,
    default_replica_id,
    encode_result,
)

logger = logging.getLogger("bibr.serve.jobs.redis")

DEFAULT_KEY_PREFIX = "bibr:jobs"
#: Safety TTL on queued/running records: bounds how long a crashed replica's
#: jobs can hold cap slots (and report a stale status).
ACTIVE_SAFETY_TTL_SECONDS = 24 * 3600

#: Timeouts matching ``bibr.cache.ResponseCache`` when the settings pass none.
DEFAULT_CONNECT_TIMEOUT = 2.0
DEFAULT_SOCKET_TIMEOUT = 5.0

# KEYS[1] = active set, KEYS[2] = job hash
# ARGV[1] = max_active, ARGV[2] = job id, ARGV[3] = active safety TTL (s),
# ARGV[4] = key prefix, ARGV[5..] = hash field/value pairs
# Returns {admitted (0/1), active count after the call}.
_CREATE_SCRIPT = """
local active_set = KEYS[1]
local prefix = ARGV[4]
local active = 0
for _, id in ipairs(redis.call('SMEMBERS', active_set)) do
  if redis.call('EXISTS', prefix .. ':job:' .. id) == 1 then
    active = active + 1
  else
    redis.call('SREM', active_set, id)
  end
end
if active >= tonumber(ARGV[1]) then
  return {0, active}
end
redis.call('SADD', active_set, ARGV[2])
for i = 5, #ARGV, 2 do
  redis.call('HSET', KEYS[2], ARGV[i], ARGV[i + 1])
end
redis.call('EXPIRE', KEYS[2], ARGV[3])
return {1, active + 1}
"""

# KEYS[1] = job hash; ARGV[1] = started_at. Never resurrects a missing record.
_RUNNING_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 0 then
  return 0
end
redis.call('HSET', KEYS[1], 'status', 'running', 'started_at', ARGV[1])
return 1
"""

# KEYS[1] = job hash, KEYS[2] = result key, KEYS[3] = active set, KEYS[4] = finished zset
# ARGV[1] = job id, ARGV[2] = status, ARGV[3] = finished_at, ARGV[4] = ttl (s),
# ARGV[5] = http_status ('' = none), ARGV[6] = error JSON ('' = none),
# ARGV[7] = result_size, ARGV[8] = compressed result body ('' for failures),
# ARGV[9] = key prefix, ARGV[10] = max_retained, ARGV[11] = max_retained_bytes
_FINISH_SCRIPT = """
local job_key, result_key, active_set, finished = KEYS[1], KEYS[2], KEYS[3], KEYS[4]
local job_id, status, finished_at, ttl = ARGV[1], ARGV[2], ARGV[3], ARGV[4]
local prefix = ARGV[9]

redis.call('SREM', active_set, job_id)
if redis.call('EXISTS', job_key) == 1 then
  redis.call('HSET', job_key,
    'status', status, 'finished_at', finished_at,
    'http_status', ARGV[5], 'error', ARGV[6], 'result_size', ARGV[7])
  redis.call('EXPIRE', job_key, ttl)
  if status == 'succeeded' then
    redis.call('SET', result_key, ARGV[8], 'EX', ttl)
  end
  redis.call('ZADD', finished, finished_at, job_id)
end

local function drop(id)
  redis.call('DEL', prefix .. ':job:' .. id, prefix .. ':result:' .. id)
  redis.call('ZREM', finished, id)
end

-- Past the TTL window (the keys carry EXPIRE too; this keeps the index honest).
local cutoff = tonumber(finished_at) - tonumber(ttl)
for _, id in ipairs(redis.call('ZRANGEBYSCORE', finished, '-inf', cutoff)) do
  drop(id)
end

-- Oldest-first beyond the count or byte budget; the newest result always stays.
local live, sizes, total = {}, {}, 0
for _, id in ipairs(redis.call('ZRANGE', finished, 0, -1)) do
  local size = redis.call('HGET', prefix .. ':job:' .. id, 'result_size')
  if size == false then
    drop(id)
  else
    live[#live + 1] = id
    sizes[#live] = tonumber(size) or 0
    total = total + sizes[#live]
  end
end
local max_retained, max_bytes = tonumber(ARGV[10]), tonumber(ARGV[11])
local count, evict = #live, 0
while evict < count do
  local remaining = count - evict
  local over_count = remaining > max_retained
  local over_bytes = max_bytes > 0 and total > max_bytes and remaining > 1
  if not (over_count or over_bytes) then
    break
  end
  total = total - sizes[evict + 1]
  evict = evict + 1
end
for i = 1, evict do
  drop(live[i])
end
return 1
"""


def _text(value: bytes | str | None) -> str | None:
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _float_or_none(value: bytes | str | None) -> float | None:
    text = _text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


class RedisJobStore:
    """Shared job store on ``redis.asyncio`` — see the module docstring for the design.

    ``client_factory`` (``url -> redis.asyncio.Redis``) lets tests inject a fake
    server; ``wall_clock`` is injectable the same way as the memory store's.
    """

    def __init__(
        self,
        url: str,
        *,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        replica_id: str | None = None,
        connect_timeout: float | None = None,
        socket_timeout: float | None = None,
        wall_clock: Callable[[], float] = time.time,
        active_ttl_seconds: int = ACTIVE_SAFETY_TTL_SECONDS,
        client_factory: Callable[[str], Any] | None = None,
        compress_level: int = 6,
    ) -> None:
        self._prefix = key_prefix.rstrip(":") or DEFAULT_KEY_PREFIX
        self._replica_id = replica_id or default_replica_id()
        self._wall_clock = wall_clock
        self._active_ttl = max(1, int(active_ttl_seconds))
        self._compress_level = compress_level
        self._connect_timeout = (
            DEFAULT_CONNECT_TIMEOUT if connect_timeout is None else float(connect_timeout)
        )
        self._socket_timeout = (
            DEFAULT_SOCKET_TIMEOUT if socket_timeout is None else float(socket_timeout)
        )
        # A command may have to (re)connect before it can wait for its reply.
        self._op_timeout = self._connect_timeout + self._socket_timeout
        factory = client_factory or self._default_client
        self._redis = factory(url)
        self._create_script = self._redis.register_script(_CREATE_SCRIPT)
        self._running_script = self._redis.register_script(_RUNNING_SCRIPT)
        self._finish_script = self._redis.register_script(_FINISH_SCRIPT)
        self._closed = False

    def _default_client(self, url: str):
        return aioredis.Redis.from_url(
            url,
            decode_responses=False,  # result bodies are binary
            socket_connect_timeout=self._connect_timeout,
            socket_timeout=self._socket_timeout,
            socket_keepalive=True,
            health_check_interval=30,
        )

    # -- keys ----------------------------------------------------------------

    @property
    def replica_id(self) -> str:
        return self._replica_id

    @property
    def key_prefix(self) -> str:
        return self._prefix

    @property
    def operation_timeout(self) -> float:
        return self._op_timeout

    @property
    def active_key(self) -> str:
        return f"{self._prefix}:active"

    @property
    def finished_key(self) -> str:
        return f"{self._prefix}:finished"

    def job_key(self, job_id: str) -> str:
        return f"{self._prefix}:job:{job_id}"

    def result_key(self, job_id: str) -> str:
        return f"{self._prefix}:result:{job_id}"

    # -- plumbing ------------------------------------------------------------

    async def _bounded(self, what: str, awaitable):
        """Run one Redis call under the operation budget; any failure → unavailable."""
        try:
            # Keep external cancellation even if Redis finishes in the same loop
            # turn. Python 3.11's wait_for can swallow it and strand shutdown.
            async with asyncio.timeout(self._op_timeout):
                return await awaitable
        except Exception as exc:  # RedisError, TimeoutError, OSError, ...
            raise JobStoreUnavailableError(
                f"redis job store {what} failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _job_from_fields(self, job_id: str, fields: dict) -> Job:
        decoded = {_text(key): value for key, value in fields.items()}
        error_raw = _text(decoded.get("error"))
        error: dict | None = None
        if error_raw:
            try:
                parsed = json.loads(error_raw)
                error = parsed if isinstance(parsed, dict) else {"detail": str(parsed)}
            except ValueError:
                error = {"detail": "job failed"}
        http_status_raw = _text(decoded.get("http_status"))
        size_raw = _text(decoded.get("result_size"))
        return Job(
            job_id=job_id,
            filename=_text(decoded.get("filename")) or "",
            status=_text(decoded.get("status")) or "queued",
            created_wall=_float_or_none(decoded.get("created_at")) or 0.0,
            started_wall=_float_or_none(decoded.get("started_at")),
            finished_wall=_float_or_none(decoded.get("finished_at")),
            http_status=int(http_status_raw) if http_status_raw else None,
            error=error,
            replica=_text(decoded.get("replica")),
            result_size_hint=int(size_raw) if size_raw and size_raw.isdigit() else None,
        )

    def _expired(self, job: Job) -> bool:
        if job.finished_wall is None:
            return False
        return (self._wall_clock() - job.finished_wall) > Settings.jobs.ttl_seconds

    # -- JobStore protocol ---------------------------------------------------

    async def create(self, *, filename: str) -> Job:
        job_id = uuid.uuid4().hex
        now = self._wall_clock()
        fields = {
            "status": "queued",
            "filename": filename,
            "created_at": repr(now),
            "replica": self._replica_id,
        }
        max_active = Settings.jobs.max_active
        args: list[Any] = [max_active, job_id, self._active_ttl, self._prefix]
        for key, value in fields.items():
            args.extend((key, value))
        admitted, active = await self._bounded(
            "create",
            self._create_script(keys=[self.active_key, self.job_key(job_id)], args=args),
        )
        if not int(admitted):
            raise JobCapacityError(f"active job cap reached ({int(active)}/{max_active})")
        return Job(
            job_id=job_id,
            filename=filename,
            created_wall=now,
            replica=self._replica_id,
        )

    async def get(self, job_id: str, *, include_result: bool = True) -> Job | None:
        raw: bytes | None = None
        if include_result:
            pipe = self._redis.pipeline(transaction=False)
            pipe.hgetall(self.job_key(job_id))
            pipe.get(self.result_key(job_id))
            fields, raw = await self._bounded("get", pipe.execute())
        else:
            fields = await self._bounded("get", self._redis.hgetall(self.job_key(job_id)))
        if not fields:
            return None
        job = self._job_from_fields(job_id, fields)
        if self._expired(job):
            return None
        if raw is not None:
            try:
                job.result_json = await asyncio.to_thread(zlib.decompress, raw)
            except zlib.error:
                logger.error("job %s: stored result is not valid zlib data; dropping it", job_id)
        return job

    async def discard(self, job_id: str) -> None:
        pipe = self._redis.pipeline(transaction=True)
        pipe.srem(self.active_key, job_id)
        pipe.delete(self.job_key(job_id), self.result_key(job_id))
        try:
            await self._bounded("discard", pipe.execute())
        except JobStoreUnavailableError as exc:
            logger.error("job %s: could not discard the job record (%s)", job_id, exc)

    async def set_running(self, job_id: str) -> None:
        try:
            await self._bounded(
                "set_running",
                self._running_script(keys=[self.job_key(job_id)], args=[repr(self._wall_clock())]),
            )
        except JobStoreUnavailableError as exc:
            logger.error("job %s: could not record the running state (%s)", job_id, exc)

    async def set_succeeded(self, job_id: str, result: dict) -> None:
        # Encode first so an unrenderable result raises exactly like the memory
        # store's; compress off the loop, a large export takes real CPU time.
        encoded = encode_result(result)
        compressed = await asyncio.to_thread(zlib.compress, encoded, self._compress_level)
        await self._finish(
            job_id,
            status="succeeded",
            http_status=None,
            error=None,
            result=compressed,
            result_size=len(encoded),
        )

    async def set_failed(self, job_id: str, *, http_status: int | None, error: dict) -> None:
        await self._finish(
            job_id,
            status="failed",
            http_status=http_status,
            error=error,
            result=b"",
            result_size=0,
        )

    async def _finish(
        self,
        job_id: str,
        *,
        status: str,
        http_status: int | None,
        error: dict | None,
        result: bytes,
        result_size: int,
    ) -> None:
        ttl = max(1, int(Settings.jobs.ttl_seconds))
        args: list[Any] = [
            job_id,
            status,
            repr(self._wall_clock()),
            ttl,
            "" if http_status is None else str(http_status),
            "" if error is None else json.dumps(error),
            result_size,
            result,
            self._prefix,
            Settings.jobs.max_retained,
            Settings.jobs.max_retained_bytes,
        ]
        keys = [
            self.job_key(job_id),
            self.result_key(job_id),
            self.active_key,
            self.finished_key,
        ]
        try:
            await self._bounded(f"set_{status}", self._finish_script(keys=keys, args=args))
        except JobStoreUnavailableError as exc:
            logger.error("job %s: could not record %s (%s)", job_id, status, exc)

    async def ping(self) -> None:
        """Bounded liveness probe for ``/ready``; raises when Redis does not answer."""
        await self._bounded("ping", self._redis.ping())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._redis.aclose()
        except Exception as exc:
            logger.debug("redis job store close failed: %s", exc)
