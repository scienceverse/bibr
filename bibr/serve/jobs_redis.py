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
- ``{prefix}:lease:{id}`` — the owner lease of a queued/running job: a short-TTL
  key the owning replica renews while it holds the job.
- ``{prefix}:active`` — SET of queued/running ids, the global cap.
- ``{prefix}:finished`` — ZSET of finished ids scored by ``finished_at``, the
  global ``JOBS_MAX_RETAINED`` / ``JOBS_MAX_RETAINED_BYTES`` eviction order.

**Atomicity.** ``create`` is one Lua script: it reconciles the active set
against the job hashes (dropping ids whose record is gone, failing ids whose
owner lease lapsed), counts, refuses past ``JOBS_MAX_ACTIVE`` or admits and
writes the record and its lease — no window between the check and the
insert, whichever replica runs it. Lua was chosen over
WATCH/MULTI because the reconcile-then-count step is a read-modify-write
over a whole set, which WATCH cannot express without retries. The finish
transition (status, TTLs, result, cap release, retention pruning) is a second
script, so pruning runs under the same global order every replica sees.

**TTLs and leases.** Finished job + result keys expire after
``JOBS_TTL_SECONDS``, the same window the memory store uses, and ``get`` treats
a record past that window as gone even before Redis reaps it. A queued/running
record carries an owner lease (``LEASE_TTL_SECONDS``, 60 s by default) that the
owning replica renews from a background heartbeat while it holds the job. When
the lease lapses — the replica died, or gave up recording the job's outcome —
the next ``create`` on any replica fails the record (``503``, ``error_code``
``job_lost``) and its cap slot returns. Lease keys live in Redis, so no replica
compares its clock with another's. Records also keep a generous safety TTL
(24 h): records written by a replica without leases (an older release during a
rolling upgrade) still rely on it. A job owned by another replica is reported
as-is; no replica ever executes another's job.

**Failure semantics.** Every Redis touch runs under ``asyncio.timeout`` with
the connect+socket budget from ``REDIS_CONNECT_TIMEOUT_SECONDS`` /
``REDIS_SOCKET_TIMEOUT_SECONDS`` (the client carries the same socket timeouts
as a first fence). ``create``/``get``/``cancel`` raise
:class:`~bibr.serve.jobs.JobStoreUnavailableError`, which the routes turn into
``503``. The runner's transitions never raise: ``set_succeeded``/``set_failed``
retry with backoff for up to ``TRANSITION_RETRY_SECONDS`` (a blip as a job
finishes must not lose its result), then log and drop the job, whose lease
lapses so a later ``create`` fails it; ``set_running``/``discard`` log and
swallow after one attempt (the lease covers what they leave behind).

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
    CANCELLED_ERROR,
    CANCELLED_HTTP_STATUS,
    Job,
    JobCapacityError,
    JobStoreUnavailableError,
    default_replica_id,
    encode_result,
)

logger = logging.getLogger("bibr.serve.jobs.redis")

DEFAULT_KEY_PREFIX = "bibr:jobs"
#: Safety TTL on queued/running records. Leased records are failed long before it;
#: it still bounds records from replicas that write no lease.
ACTIVE_SAFETY_TTL_SECONDS = 24 * 3600
#: Owner lease on queued/running records, renewed every quarter of it: bounds how
#: long a dead replica's jobs hold cap slots (and report a stale status).
LEASE_TTL_SECONDS = 60
#: How long ``set_succeeded``/``set_failed`` keep retrying before giving the job up.
TRANSITION_RETRY_SECONDS = 60.0
_RETRY_FIRST_DELAY = 0.1
_RETRY_MAX_DELAY = 5.0
#: Recorded on a leased job whose lease lapsed: 503 tells the client to resubmit.
LOST_HTTP_STATUS = 503
LOST_ERROR = {"detail": "replica lost the job before it finished", "error_code": "job_lost"}

#: Timeouts matching ``bibr.cache.ResponseCache`` when the settings pass none.
DEFAULT_CONNECT_TIMEOUT = 2.0
DEFAULT_SOCKET_TIMEOUT = 5.0

# KEYS[1] = active set, KEYS[2] = job hash, KEYS[3] = job lease, KEYS[4] = finished zset
# ARGV[1] = max_active, ARGV[2] = job id, ARGV[3] = active safety TTL (s),
# ARGV[4] = key prefix, ARGV[5] = lease TTL (s), ARGV[6] = now (wall clock),
# ARGV[7] = finished-record TTL (s), ARGV[8] = lost-job http_status,
# ARGV[9] = lost-job error JSON, ARGV[10..] = hash field/value pairs
# Returns {admitted (0/1), active count after the call}.
_CREATE_SCRIPT = """
local active_set, finished = KEYS[1], KEYS[4]
local prefix, now = ARGV[4], ARGV[6]
local active = 0
for _, id in ipairs(redis.call('SMEMBERS', active_set)) do
  local job_key = prefix .. ':job:' .. id
  if redis.call('EXISTS', job_key) == 0 then
    redis.call('SREM', active_set, id)
  elseif redis.call('HEXISTS', job_key, 'leased') == 1
      and redis.call('EXISTS', prefix .. ':lease:' .. id) == 0 then
    -- The owner stopped renewing the lease: it died, or gave up recording the
    -- outcome. Fail the job so it stops holding a slot. (Unleased records come
    -- from replicas that predate leases and wait for the safety TTL.)
    redis.call('SREM', active_set, id)
    redis.call('HSET', job_key,
      'status', 'failed', 'finished_at', now,
      'http_status', ARGV[8], 'error', ARGV[9], 'result_size', 0)
    redis.call('EXPIRE', job_key, ARGV[7])
    redis.call('ZADD', finished, now, id)
  else
    active = active + 1
  end
end
if active >= tonumber(ARGV[1]) then
  return {0, active}
end
redis.call('SADD', active_set, ARGV[2])
for i = 10, #ARGV, 2 do
  redis.call('HSET', KEYS[2], ARGV[i], ARGV[i + 1])
end
redis.call('EXPIRE', KEYS[2], ARGV[3])
redis.call('SET', KEYS[3], '1', 'EX', ARGV[5])
return {1, active + 1}
"""

# ARGV[1] = key prefix, ARGV[2] = lease TTL (s), ARGV[3..] = ids the replica holds.
# Renews (or restores, after a blip) the lease of each job that is still active and
# returns the ids that are not, which the replica stops renewing.
_RENEW_SCRIPT = """
local prefix, ttl = ARGV[1], ARGV[2]
local gone = {}
for i = 3, #ARGV do
  local id = ARGV[i]
  if redis.call('SISMEMBER', prefix .. ':active', id) == 1 then
    redis.call('SET', prefix .. ':lease:' .. id, '1', 'EX', ttl)
  else
    gone[#gone + 1] = id
  end
end
return gone
"""

# KEYS[1] = job hash; ARGV[1] = started_at. Never resurrects a missing record.
# Returns 1 = claimed, 0 = no record, -1 = no longer queued (cancelled: skip it).
_RUNNING_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 0 then
  return 0
end
local status = redis.call('HGET', KEYS[1], 'status')
if status and status ~= 'queued' then
  return -1
end
redis.call('HSET', KEYS[1], 'status', 'running', 'started_at', ARGV[1])
return 1
"""

# KEYS[1] = job hash, KEYS[2] = result key, KEYS[3] = active set, KEYS[4] = finished zset
# ARGV[1] = job id, ARGV[2] = status, ARGV[3] = finished_at, ARGV[4] = ttl (s),
# ARGV[5] = http_status ('' = none), ARGV[6] = error JSON ('' = none),
# ARGV[7] = result_size, ARGV[8] = compressed result body ('' for failures),
# ARGV[9] = key prefix, ARGV[10] = max_retained, ARGV[11] = max_retained_bytes,
# ARGV[12] = the status the record must have now ('' = any; a cancel passes 'queued')
# Returns 1, or 0 when ARGV[12] is set and the record is missing or in another status
# (then nothing changes). A transition that applies also drops the job's lease.
_FINISH_SCRIPT = """
local job_key, result_key, active_set, finished = KEYS[1], KEYS[2], KEYS[3], KEYS[4]
local job_id, status, finished_at, ttl = ARGV[1], ARGV[2], ARGV[3], ARGV[4]
local prefix = ARGV[9]
local required = ARGV[12]
if required and required ~= '' and redis.call('HGET', job_key, 'status') ~= required then
  return 0
end

redis.call('SREM', active_set, job_id)
redis.call('DEL', prefix .. ':lease:' .. job_id)
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
        lease_ttl_seconds: int = LEASE_TTL_SECONDS,
        transition_retry_seconds: float = TRANSITION_RETRY_SECONDS,
        client_factory: Callable[[str], Any] | None = None,
        compress_level: int = 6,
    ) -> None:
        self._prefix = key_prefix.rstrip(":") or DEFAULT_KEY_PREFIX
        self._replica_id = replica_id or default_replica_id()
        self._wall_clock = wall_clock
        self._active_ttl = max(1, int(active_ttl_seconds))
        self._lease_ttl = max(1, int(lease_ttl_seconds))
        self._retry_seconds = max(0.0, float(transition_retry_seconds))
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
        self._renew_script = self._redis.register_script(_RENEW_SCRIPT)
        # Jobs this replica created and has not let go of: the heartbeat renews
        # their leases.
        self._owned: set[str] = set()
        self._heartbeat: asyncio.Task | None = None
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

    def lease_key(self, job_id: str) -> str:
        return f"{self._prefix}:lease:{job_id}"

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

    async def _retrying(self, what: str, call: Callable[[], Any]):
        """``_bounded`` with backoff until Redis answers or the retry budget runs out.

        ``call`` makes a fresh awaitable per attempt. Only for idempotent scripts: an
        attempt whose reply was lost may have run.
        """
        deadline = time.monotonic() + self._retry_seconds
        delay = _RETRY_FIRST_DELAY
        while True:
            try:
                return await self._bounded(what, call())
            except JobStoreUnavailableError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                if delay == _RETRY_FIRST_DELAY:
                    logger.warning("%s; retrying for up to %.1fs", exc, remaining)
                await asyncio.sleep(min(delay, remaining))
                delay = min(delay * 2, _RETRY_MAX_DELAY)

    def _start_heartbeat(self) -> None:
        if self._closed:
            return
        loop = asyncio.get_running_loop()
        task = self._heartbeat
        # A task left on another (finished) loop no longer runs.
        if task is None or task.done() or task.get_loop() is not loop:
            self._heartbeat = loop.create_task(self._heartbeat_loop(), name="bibr-job-leases")

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._lease_ttl / 4)
            try:
                await self._renew_leases()
            except Exception:
                # A dead heartbeat would let every lease lapse under live jobs.
                logger.exception("job lease heartbeat failed")

    async def _renew_leases(self) -> None:
        """Renew the lease of every job this replica holds; forget the ones that ended."""
        if not self._owned:
            return
        held = sorted(self._owned)
        try:
            gone = await self._bounded(
                "renew leases",
                self._renew_script(args=[self._prefix, self._lease_ttl, *held]),
            )
        except JobStoreUnavailableError as exc:
            logger.warning("could not renew %d job lease(s) (%s)", len(held), exc)
            return
        for job_id in gone:
            self._owned.discard(_text(job_id))

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
            # Marks a record whose lease create may enforce (see _CREATE_SCRIPT).
            "leased": "1",
        }
        max_active = Settings.jobs.max_active
        args: list[Any] = [
            max_active,
            job_id,
            self._active_ttl,
            self._prefix,
            self._lease_ttl,
            repr(now),
            max(1, int(Settings.jobs.ttl_seconds)),
            LOST_HTTP_STATUS,
            json.dumps(LOST_ERROR),
        ]
        for key, value in fields.items():
            args.extend((key, value))
        keys = [self.active_key, self.job_key(job_id), self.lease_key(job_id), self.finished_key]
        admitted, active = await self._bounded("create", self._create_script(keys=keys, args=args))
        if not int(admitted):
            raise JobCapacityError(f"active job cap reached ({int(active)}/{max_active})")
        self._owned.add(job_id)
        self._start_heartbeat()
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
        # Let go first: a record left behind is failed once its lease lapses.
        self._owned.discard(job_id)
        pipe = self._redis.pipeline(transaction=True)
        pipe.srem(self.active_key, job_id)
        pipe.delete(self.job_key(job_id), self.result_key(job_id), self.lease_key(job_id))
        try:
            await self._bounded("discard", pipe.execute())
        except JobStoreUnavailableError as exc:
            logger.error("job %s: could not discard the job record (%s)", job_id, exc)

    async def set_running(self, job_id: str) -> bool:
        try:
            claimed = await self._bounded(
                "set_running",
                self._running_script(keys=[self.job_key(job_id)], args=[repr(self._wall_clock())]),
            )
        except JobStoreUnavailableError as exc:
            # Run it anyway; its lease (still renewed) keeps the record from going stale.
            logger.error("job %s: could not record the running state (%s)", job_id, exc)
            return True
        # 1 = claimed; -1 = no longer queued; 0 = the record is gone (evicted).
        if int(claimed) != 1:
            self._owned.discard(job_id)
            return False
        return True

    async def cancel(self, job_id: str) -> Job | None:
        keys, args = self._finish_call(
            job_id,
            status="failed",
            http_status=CANCELLED_HTTP_STATUS,
            error=CANCELLED_ERROR,
            result=b"",
            result_size=0,
            required="queued",
        )
        if int(await self._bounded("cancel", self._finish_script(keys=keys, args=args))):
            self._owned.discard(job_id)
        return await self.get(job_id, include_result=False)

    def _render(self, result: dict) -> tuple[bytes, bytes]:
        encoded = encode_result(result)
        return encoded, zlib.compress(encoded, self._compress_level)

    async def set_succeeded(self, job_id: str, result: dict) -> None:
        # Encode first so an unrenderable result raises exactly like the memory
        # store's; both steps run off the loop, a large export takes real CPU time.
        encoded, compressed = await asyncio.to_thread(self._render, result)
        await self._finish(
            job_id,
            status="succeeded",
            http_status=None,
            error=None,
            result=compressed,
            result_size=len(encoded),
        )

    async def set_failed(
        self, job_id: str, *, http_status: int | None, error: dict, required: str = ""
    ) -> None:
        await self._finish(
            job_id,
            status="failed",
            http_status=http_status,
            error=error,
            result=b"",
            result_size=0,
            required=required,
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
        required: str = "",
    ) -> None:
        keys, args = self._finish_call(
            job_id,
            status=status,
            http_status=http_status,
            error=error,
            result=result,
            result_size=result_size,
            required=required,
        )
        try:
            # The script is idempotent (the same finished_at on every attempt), so a
            # retry after a lost reply rewrites the same outcome.
            await self._retrying(
                f"set_{status} (job {job_id})",
                lambda: self._finish_script(keys=keys, args=args),
            )
        except JobStoreUnavailableError as exc:
            logger.error(
                "job %s: could not record %s (%s); its lease lapses and a later create fails it",
                job_id,
                status,
                exc,
            )
        finally:
            # The runner is done with the job either way: stop renewing its lease.
            self._owned.discard(job_id)

    def _finish_call(
        self,
        job_id: str,
        *,
        status: str,
        http_status: int | None,
        error: dict | None,
        result: bytes,
        result_size: int,
        required: str = "",
    ) -> tuple[list[str], list[Any]]:
        """KEYS and ARGV for ``_FINISH_SCRIPT``."""
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
            required,
        ]
        keys = [
            self.job_key(job_id),
            self.result_key(job_id),
            self.active_key,
            self.finished_key,
        ]
        return keys, args

    async def ping(self) -> None:
        """Bounded liveness probe for ``/ready``; raises when Redis does not answer."""
        await self._bounded("ping", self._redis.ping())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        heartbeat, self._heartbeat = self._heartbeat, None
        if heartbeat is not None and not heartbeat.done():
            try:
                heartbeat.cancel()
            except RuntimeError:  # its loop is already closed
                pass
            else:
                if heartbeat.get_loop() is asyncio.get_running_loop():
                    await asyncio.gather(heartbeat, return_exceptions=True)
        try:
            await self._redis.aclose()
        except Exception as exc:
            logger.debug("redis job store close failed: %s", exc)
