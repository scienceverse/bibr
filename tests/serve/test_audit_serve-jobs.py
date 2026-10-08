"""Audit regressions for the serve job stores and dispatcher.

- Redis store: a terminal transition that meets a Redis blip is retried until it
  lands, and a queued/running record whose owner stops renewing its lease is
  failed by the next ``create`` instead of pinning a global cap slot for 24 h.
- Shutdown records every job a cancelled worker held, wherever the cancellation
  landed, within a bounded time.
- A job cancelled while queued leaves the dispatcher's queue at once (or, when
  cancelled through another replica, once the queue reaches the active cap).
- Results are rendered and compressed off the event loop, in memory close to
  ``json.dumps``' and at any depth ``json.dumps`` reaches.
"""

import asyncio
import gzip
import json
import logging
import tracemalloc

import pytest

pytest.importorskip("fastapi")

from starlette.middleware.gzip import GZipMiddleware

from bibr.config import Settings
from bibr.serve import jobs as jobs_mod
from bibr.serve.jobs import JobCapacityError, MemoryJobStore
from tests.serve.test_jobs import (
    _client,
    _FakeTracker,
    _MemoryHarness,
    _poll_until,
    _post,
    _RedisHarness,
)

_SHUTDOWN_ERROR = {"detail": "replica shut down before the job finished"}


@pytest.fixture(params=["memory", "redis"])
async def harness(request):
    harness = _MemoryHarness() if request.param == "memory" else _RedisHarness()
    try:
        yield harness
    finally:
        await harness.close()


@pytest.fixture
async def redis_harness():
    harness = _RedisHarness()
    try:
        yield harness
    finally:
        await harness.close()


class _BlockingTracker(_FakeTracker):
    """Holds every submitted job until ``release`` is set."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def submit(self, descriptor, request_state=None):
        self.descriptors.append(descriptor)
        self.entered.set()
        await self.release.wait()
        return self.result


def _payload(upload_id) -> jobs_mod.JobPayload:
    return jobs_mod.JobPayload(descriptor={"upload_id": upload_id})


# --------------------------------------------------------------------------- #
# Redis store: lost transitions and owner leases
# --------------------------------------------------------------------------- #


async def test_a_finish_that_meets_a_redis_blip_lands_once_redis_answers(
    redis_harness, monkeypatch
):
    monkeypatch.setattr(Settings.jobs, "max_active", 2)
    store = redis_harness.make()
    loop = asyncio.get_running_loop()
    jobs = []
    for index in range(2):
        job = await store.create(filename=f"{index}.pdf")
        assert await store.set_running(job.job_id)
        # Redis drops out just as the job finishes and comes back a moment later.
        redis_harness.server.connected = False
        loop.call_later(0.2, setattr, redis_harness.server, "connected", True)
        await asyncio.wait_for(store.set_succeeded(job.job_id, {"index": index}), timeout=10)
        jobs.append(job)

    for index, job in enumerate(jobs):
        got = await store.get(job.job_id)
        assert got.status == "succeeded"
        assert got.result == {"index": index}
        assert not await store._redis.sismember(store.active_key, job.job_id)
        assert await store._redis.exists(store.lease_key(job.job_id)) == 0
        assert await store._redis.ttl(store.job_key(job.job_id)) <= Settings.jobs.ttl_seconds
    # Both cap slots came back.
    assert (await store.create(filename="next.pdf")).status == "queued"


async def test_a_job_whose_outcome_could_not_be_recorded_is_failed_once_its_lease_lapses(
    redis_harness, monkeypatch
):
    from bibr.serve.jobs_redis import LOST_ERROR

    monkeypatch.setattr(Settings.jobs, "max_active", 2)
    store = redis_harness.make(transition_retry_seconds=0.05)
    other = redis_harness.make(replica_id="replica-b")
    jobs = []
    for index in range(2):
        job = await store.create(filename=f"{index}.pdf")
        assert await store.set_running(job.job_id)
        redis_harness.server.connected = False  # longer than the retry budget
        await asyncio.wait_for(store.set_succeeded(job.job_id, {"index": index}), timeout=10)
        redis_harness.server.connected = True
        jobs.append(job)
    # The replica gave the jobs up: its heartbeat no longer renews their leases.
    await store._renew_leases()
    for job in jobs:
        # Stand-in for the lease TTL running out.
        await store._redis.delete(store.lease_key(job.job_id))
    await store._renew_leases()

    # The next create, on any replica, fails the stale records and admits.
    assert (await other.create(filename="next.pdf")).status == "queued"
    for job in jobs:
        got = await other.get(job.job_id)
        assert got.status == "failed"
        assert got.http_status == 503
        assert got.error == LOST_ERROR
        assert got.status_dict()["error"]["error_code"] == "job_lost"
        assert got.finished_wall is not None
        assert not await store._redis.sismember(store.active_key, job.job_id)
        assert await store._redis.zscore(store.finished_key, job.job_id) is not None
        assert 0 < await store._redis.ttl(store.job_key(job.job_id)) <= Settings.jobs.ttl_seconds


async def test_a_dead_replicas_jobs_free_their_slots_once_their_leases_lapse(
    redis_harness, monkeypatch
):
    monkeypatch.setattr(Settings.jobs, "max_active", 2)
    dead = redis_harness.make(replica_id="replica-a")
    live = redis_harness.make(replica_id="replica-b")
    running = await dead.create(filename="running.pdf")
    await dead.set_running(running.job_id)
    queued = await dead.create(filename="queued.pdf")
    # While the owner renews its leases, its jobs hold their slots.
    with pytest.raises(JobCapacityError):
        await live.create(filename="b.pdf")

    # replica-a dies: nothing renews its leases, and they run out.
    for job in (running, queued):
        await dead._redis.delete(dead.lease_key(job.job_id))

    assert (await live.create(filename="b.pdf")).status == "queued"
    for job in (running, queued):
        got = await live.get(job.job_id)
        assert got.status == "failed"
        assert got.error["error_code"] == "job_lost"
        assert got.replica == "replica-a"


async def test_a_record_without_a_lease_keeps_its_slot_until_the_safety_ttl(
    redis_harness, monkeypatch
):
    """A record written by a replica that predates leases is never failed early."""
    monkeypatch.setattr(Settings.jobs, "max_active", 1)
    store = redis_harness.make()
    old_id = "0" * 32
    await store._redis.hset(
        store.job_key(old_id),
        mapping={"status": "running", "filename": "old.pdf", "created_at": "1.0"},
    )
    await store._redis.expire(store.job_key(old_id), 3600)
    await store._redis.sadd(store.active_key, old_id)

    with pytest.raises(JobCapacityError):
        await store.create(filename="new.pdf")
    assert (await store.get(old_id)).status == "running"


async def test_leases_are_written_with_the_record_and_dropped_with_it(redis_harness):
    store = redis_harness.make(lease_ttl_seconds=30)
    finished = await store.create(filename="finished.pdf")
    assert 0 < await store._redis.ttl(store.lease_key(finished.job_id)) <= 30
    assert await store._redis.hget(store.job_key(finished.job_id), "leased") == b"1"
    await store.set_failed(finished.job_id, http_status=422, error={"detail": "x"})
    discarded = await store.create(filename="discarded.pdf")
    await store.discard(discarded.job_id)
    cancelled = await store.create(filename="cancelled.pdf")
    await store.cancel(cancelled.job_id)
    for job in (finished, discarded, cancelled):
        assert await store._redis.exists(store.lease_key(job.job_id)) == 0
    assert store._owned == set()


async def test_the_heartbeat_renews_held_leases_and_forgets_ended_jobs(redis_harness):
    store = redis_harness.make(lease_ttl_seconds=30)
    other = redis_harness.make(replica_id="replica-b")
    held = await store.create(filename="held.pdf")
    lapsed = await store.create(filename="lapsed.pdf")
    cancelled = await store.create(filename="cancelled.pdf")
    await store._redis.expire(store.lease_key(held.job_id), 2)
    # A blip outlived this lease, but no create has failed the job yet.
    await store._redis.delete(store.lease_key(lapsed.job_id))
    # Cancelled through another replica: this one is not told.
    await other.cancel(cancelled.job_id)

    await store._renew_leases()

    assert await store._redis.ttl(store.lease_key(held.job_id)) > 2
    assert await store._redis.exists(store.lease_key(lapsed.job_id)) == 1
    assert await store._redis.exists(store.lease_key(cancelled.job_id)) == 0
    assert store._owned == {held.job_id, lapsed.job_id}


async def test_the_heartbeat_runs_in_the_background_until_close(redis_harness, monkeypatch):
    store = redis_harness.make(lease_ttl_seconds=1)  # renews every 0.25 s
    renewed = asyncio.Event()
    real_renew = store._renew_leases

    async def spy():
        await real_renew()
        renewed.set()

    monkeypatch.setattr(store, "_renew_leases", spy)
    await store.create(filename="a.pdf")
    await asyncio.wait_for(renewed.wait(), timeout=5)
    heartbeat = store._heartbeat
    assert heartbeat is not None and not heartbeat.done()
    await store.close()
    assert heartbeat.done()


async def test_the_heartbeat_retries_a_failed_renew_soon_and_warns_once(
    redis_harness, monkeypatch, caplog
):
    from types import SimpleNamespace

    from bibr.serve import jobs_redis

    real_sleep = asyncio.sleep
    delays = []
    # The heartbeat starts inside create(), and on Python 3.11 redis-py's
    # wait_for lets it run before the test's next command is sent.
    lapsed = asyncio.Event()

    async def fake_sleep(delay):
        delays.append(delay)
        if len(delays) == 1:
            await lapsed.wait()
            redis_harness.server.connected = False  # two renews fail
        elif len(delays) == 3:
            redis_harness.server.connected = True
        elif len(delays) == 4:
            raise asyncio.CancelledError
        await real_sleep(0)

    # Only this module's sleeps: the heartbeat's interval is 15 s here.
    monkeypatch.setattr(
        jobs_redis, "asyncio", SimpleNamespace(**{**vars(asyncio), "sleep": fake_sleep})
    )
    store = redis_harness.make(lease_ttl_seconds=60)
    job = await store.create(filename="a.pdf")
    await store._redis.delete(store.lease_key(job.job_id))  # lapsed during the outage
    lapsed.set()
    with caplog.at_level(logging.DEBUG, logger="bibr.serve.jobs.redis"):
        await asyncio.gather(store._heartbeat, return_exceptions=True)

    assert delays == [15.0, 1.0, 1.0, 15.0]
    assert await store._redis.ttl(store.lease_key(job.job_id)) > 15  # restored
    warnings = [r for r in caplog.records if "could not renew" in r.getMessage()]
    assert [r.levelno for r in warnings] == [logging.WARNING, logging.DEBUG]


async def test_a_transition_in_a_task_whose_cancellation_was_consumed_is_not_retried(
    redis_harness, monkeypatch
):
    store = redis_harness.make(transition_retry_seconds=1.0)
    job = await store.create(filename="a.pdf")
    attempts = []
    real_bounded = store._bounded

    async def counting_bounded(what, awaitable):
        attempts.append(what)
        return await real_bounded(what, awaitable)

    monkeypatch.setattr(store, "_bounded", counting_bounded)

    async def finish_after_a_swallowed_cancel():
        task = asyncio.current_task()
        task.cancel()
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass  # a dependency swallows it without uncancel()
        assert task.cancelling() == 1
        redis_harness.server.connected = False
        await store.set_failed(job.job_id, http_status=503, error={"detail": "x"})

    await asyncio.wait_for(asyncio.create_task(finish_after_a_swallowed_cancel()), timeout=5)
    # One attempt, not one per backoff step: shutdown is waiting for this task.
    assert len(attempts) == 1
    assert store._owned == set()


# --------------------------------------------------------------------------- #
# Shutdown records the jobs that cancelled workers held
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("claimed", [False, True], ids=["before-the-claim", "after-the-claim"])
async def test_shutdown_fails_a_job_whose_worker_was_cancelled_in_set_running(
    harness, monkeypatch, claimed
):
    monkeypatch.setattr(Settings.jobs, "max_active", 1)
    store = harness.make()
    entered = asyncio.Event()
    real_set_running = store.set_running

    async def stuck_set_running(job_id):
        if claimed:
            await real_set_running(job_id)
        entered.set()
        await asyncio.Event().wait()  # the reply never comes

    monkeypatch.setattr(store, "set_running", stuck_set_running)
    tracker = _FakeTracker(result={"paper_id": "p"})
    dispatcher = jobs_mod.JobDispatcher(store=store, tracker=tracker, max_running=1)
    job = await store.create(filename="a.pdf")
    await dispatcher.submit(job.job_id, _payload("a"))
    await asyncio.wait_for(entered.wait(), timeout=5)
    assert (await store.get(job.job_id)).status == ("running" if claimed else "queued")

    await asyncio.wait_for(dispatcher.close(), timeout=5)

    got = await store.get(job.job_id)
    assert got.status == "failed"
    assert got.http_status == 503
    assert got.error == _SHUTDOWN_ERROR
    assert tracker.descriptors == []
    assert (await store.create(filename="b.pdf")).status == "queued"  # the slot is free


@pytest.mark.parametrize("landed", [False, True], ids=["before-it-lands", "after-it-lands"])
async def test_shutdown_during_set_succeeded_keeps_a_landed_result_and_fails_the_rest(
    harness, monkeypatch, landed
):
    monkeypatch.setattr(Settings.jobs, "max_active", 1)
    store = harness.make()
    entered = asyncio.Event()
    real_set_succeeded = store.set_succeeded

    async def stuck_set_succeeded(job_id, result):
        if landed:
            await real_set_succeeded(job_id, result)
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(store, "set_succeeded", stuck_set_succeeded)
    dispatcher = jobs_mod.JobDispatcher(
        store=store, tracker=_FakeTracker(result={"paper_id": "p"}), max_running=1
    )
    job = await store.create(filename="a.pdf")
    await dispatcher.submit(job.job_id, _payload("a"))
    await asyncio.wait_for(entered.wait(), timeout=5)

    await asyncio.wait_for(dispatcher.close(), timeout=5)

    got = await store.get(job.job_id)
    if landed:
        assert got.status == "succeeded"
        assert got.result == {"paper_id": "p"}
    else:
        assert got.status == "failed"
        assert got.error == _SHUTDOWN_ERROR
    assert (await store.create(filename="b.pdf")).status == "queued"


async def test_shutdown_gives_up_on_a_store_that_never_answers(monkeypatch):
    monkeypatch.setattr(jobs_mod, "_SHUTDOWN_RECORD_SECONDS", 0.1)
    store = MemoryJobStore()
    tracker = _BlockingTracker()
    dispatcher = jobs_mod.JobDispatcher(store=store, tracker=tracker, max_running=1)
    running = await store.create(filename="running.pdf")
    queued = await store.create(filename="queued.pdf")
    for job in (running, queued):
        await dispatcher.submit(job.job_id, _payload(job.filename))
    await asyncio.wait_for(tracker.entered.wait(), timeout=5)

    async def hang(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(store, "set_failed", hang)
    await asyncio.wait_for(dispatcher.close(), timeout=5)
    # The waiting job's upload is still let go.
    assert tracker.discarded == [{"upload_id": "queued.pdf"}]


async def test_a_dispatch_cancelled_under_its_worker_fails_the_job_and_the_queue_goes_on(
    harness,
):
    class _CancelledOnce(_FakeTracker):
        async def submit(self, descriptor, request_state=None):
            self.descriptors.append(descriptor)
            if len(self.descriptors) == 1:
                # The inference task was cancelled; the worker itself was not.
                raise asyncio.CancelledError
            return {"paper_id": descriptor["upload_id"]}

    store = harness.make()
    dispatcher = jobs_mod.JobDispatcher(store=store, tracker=_CancelledOnce(), max_running=1)
    first = await store.create(filename="first.pdf")
    second = await store.create(filename="second.pdf")
    for job in (first, second):
        await dispatcher.submit(job.job_id, _payload(job.filename))
    await asyncio.wait_for(dispatcher.join(), timeout=5)
    await dispatcher.close()

    lost = await store.get(first.job_id)
    assert lost.status == "failed"
    assert lost.http_status == 503
    assert (await store.get(second.job_id)).result == {"paper_id": "second.pdf"}


async def test_a_job_whose_runner_failed_outside_its_own_handling_is_failed(harness, monkeypatch):
    monkeypatch.setattr(Settings.jobs, "max_active", 1)
    store = harness.make()

    async def broken_set_succeeded(job_id, result):
        raise RuntimeError("store bug")

    monkeypatch.setattr(store, "set_succeeded", broken_set_succeeded)
    dispatcher = jobs_mod.JobDispatcher(
        store=store, tracker=_FakeTracker(result={"paper_id": "p"}), max_running=1
    )
    job = await store.create(filename="a.pdf")
    await dispatcher.submit(job.job_id, _payload("a"))
    await asyncio.wait_for(dispatcher.join(), timeout=5)
    await dispatcher.close()

    got = await store.get(job.job_id)
    assert got.status == "failed"
    assert got.http_status == 500
    assert got.error == {"detail": "internal job error"}
    if harness.backend == "redis":
        assert store._owned == set()  # nobody renews a lease for a job no one runs
    assert (await store.create(filename="b.pdf")).status == "queued"  # the slot is free


async def test_a_dispatcher_closed_during_the_stale_check_refuses_the_job(monkeypatch):
    monkeypatch.setattr(Settings.jobs, "max_active", 2)
    store = MemoryJobStore()
    tracker = _BlockingTracker(result={"paper_id": "held"})
    dispatcher = jobs_mod.JobDispatcher(store=store, tracker=tracker, max_running=1)
    held = await store.create(filename="held.pdf")
    await dispatcher.submit(held.job_id, _payload("held"))
    await asyncio.wait_for(tracker.entered.wait(), timeout=5)
    waiting = await store.create(filename="waiting.pdf")
    await dispatcher.submit(waiting.job_id, _payload("waiting"))
    monkeypatch.setattr(Settings.jobs, "max_active", 1)  # the queue is at the cap

    checking = asyncio.Event()
    answer = asyncio.Event()
    real_get = store.get

    async def slow_get(job_id, **kwargs):
        checking.set()
        await answer.wait()
        return await real_get(job_id, **kwargs)

    monkeypatch.setattr(store, "get", slow_get)
    late = asyncio.create_task(dispatcher.submit("late", _payload("late")))
    await asyncio.wait_for(checking.wait(), timeout=5)
    await asyncio.wait_for(dispatcher.close(), timeout=5)
    answer.set()

    with pytest.raises(RuntimeError, match="closed"):
        await asyncio.wait_for(late, timeout=5)
    # Refused, so the route discards it, instead of stranded in a closed queue.
    assert dispatcher._waiting == {}


# --------------------------------------------------------------------------- #
# Cancelled jobs leave the queue
# --------------------------------------------------------------------------- #


async def test_jobs_cancelled_while_queued_leave_the_queue_at_once(harness, monkeypatch):
    store = harness.make()
    runs = []
    real_run_job = jobs_mod._run_job

    async def counting_run_job(**kwargs):
        runs.append(kwargs["job_id"])
        await real_run_job(**kwargs)

    monkeypatch.setattr(jobs_mod, "_run_job", counting_run_job)
    tracker = _BlockingTracker(result={"paper_id": "held"})
    dispatcher = jobs_mod.JobDispatcher(store=store, tracker=tracker, max_running=1)
    held = await store.create(filename="held.pdf")
    await dispatcher.submit(held.job_id, _payload("held"))
    await asyncio.wait_for(tracker.entered.wait(), timeout=5)

    # Submit-and-cancel while the only worker is busy, as DELETE /papers/jobs/{id} does.
    for index in range(50):
        job = await store.create(filename=f"{index}.pdf")
        await dispatcher.submit(job.job_id, _payload(index))
        assert jobs_mod.is_cancelled(await store.cancel(job.job_id))
        await dispatcher.release_cancelled(job.job_id)
    assert dispatcher._waiting == {}

    tracker.release.set()
    await asyncio.wait_for(dispatcher.join(), timeout=5)
    await dispatcher.close()
    assert runs == [held.job_id]  # no worker ever dequeued a cancelled job
    assert [d["upload_id"] for d in tracker.discarded] == list(range(50))  # once each


async def test_a_queue_of_jobs_cancelled_through_another_replica_stays_bounded(
    harness, monkeypatch
):
    monkeypatch.setattr(Settings.jobs, "max_active", 3)
    store = harness.make()
    # The memory store is per process; there the "other replica" is the same store.
    other = harness.make(replica_id="replica-b") if harness.backend == "redis" else store
    tracker = _BlockingTracker(result={"paper_id": "held"})
    dispatcher = jobs_mod.JobDispatcher(store=store, tracker=tracker, max_running=1)
    held = await store.create(filename="held.pdf")
    await dispatcher.submit(held.job_id, _payload("held"))
    await asyncio.wait_for(tracker.entered.wait(), timeout=5)

    for index in range(20):
        job = await store.create(filename=f"{index}.pdf")
        await dispatcher.submit(job.job_id, _payload(index))
        # Cancelled through another replica: this dispatcher is not told.
        assert jobs_mod.is_cancelled(await other.cancel(job.job_id))
        assert len(dispatcher._waiting) <= Settings.jobs.max_active
    # The entries let go of their uploads as they left the queue.
    assert len(tracker.discarded) >= 20 - Settings.jobs.max_active

    tracker.release.set()
    await asyncio.wait_for(dispatcher.join(), timeout=5)
    await dispatcher.close()
    discarded = [d["upload_id"] for d in tracker.discarded]
    assert sorted(discarded) == list(range(20))  # every upload exactly once
    assert (await store.get(held.job_id)).status == "succeeded"


# --------------------------------------------------------------------------- #
# Rendering and compression stay off the event loop
# --------------------------------------------------------------------------- #


def _on_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def test_encode_result_matches_json_dumps_without_the_gil_holding_c_encoder(monkeypatch):
    payload = {
        "text": [
            {"id": i, "t": 'naïve – ü \U0001f642 "q"\n', "x": 1.5, "n": None} for i in range(3)
        ],
        1: "an int key",
        "nested": {"a": [1, 2, (3, 4)], "b": True, "big": 10**30},
    }
    expected = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    def c_encoder(*args, **kwargs):
        raise AssertionError("the C encoder holds the GIL, stalling the loop from a thread")

    monkeypatch.setattr(json.encoder, "c_make_encoder", c_encoder)
    assert jobs_mod.encode_result(payload) == expected.encode("utf-8")
    with pytest.raises(ValueError):
        jobs_mod.encode_result({"x": float("nan")})
    with pytest.raises(TypeError):
        jobs_mod.encode_result({"x": object()})


def test_encode_result_peak_memory_stays_near_the_body():
    # Token-heavy, like an export's text rows: numbers, bboxes and flags. The
    # pure-Python encoder yields a small str per token; holding all of them at once
    # peaked at about 12x the body.
    rows = [
        {"id": i, "page": i % 30, "bbox": [1.25, 2.5, 3.75, 4.0], "flags": [True, False, None]}
        for i in range(6000)
    ]
    payload = {"text": rows}
    tracemalloc.start()
    try:
        body = jobs_mod.encode_result(payload)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert body == json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    assert peak < 3 * len(body)  # json.dumps itself needs 2x (one str, one bytes)


def _nested(depth: int) -> dict:
    out: dict = {"leaf": 1}
    for _ in range(depth):
        out = {"a": [out]}
    return out


def test_encode_result_renders_what_json_dumps_renders_however_deep():
    # Past the pure-Python encoder's frames (about 500 levels) and, on Python 3.12+,
    # within the C encoder's reach (about 5000).
    payload = _nested(1500)
    try:
        expected = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except RecursionError:
        with pytest.raises(RecursionError):
            jobs_mod.encode_result(payload)
    else:
        assert jobs_mod.encode_result(payload) == expected.encode("utf-8")


async def test_a_result_too_deep_to_render_fails_the_job(harness):
    store = harness.make()
    job = await store.create(filename="a.pdf")
    await jobs_mod._run_job(
        store=store,
        job_id=job.job_id,
        descriptor={"upload_id": "a"},
        tracker=_FakeTracker(result=_nested(200_000)),  # beyond any encoder
    )
    got = await store.get(job.job_id)
    assert got.status == "failed"
    assert got.http_status == 500
    assert got.error == {"detail": "internal job error"}
    if harness.backend == "redis":
        assert store._owned == set()


async def test_set_succeeded_renders_the_result_off_the_event_loop(harness, monkeypatch):
    calls = []
    real_encode = jobs_mod.encode_result

    def spy(result):
        calls.append(_on_loop())
        return real_encode(result)

    monkeypatch.setattr(jobs_mod, "encode_result", spy)
    if harness.backend == "redis":
        from bibr.serve import jobs_redis

        monkeypatch.setattr(jobs_redis, "encode_result", spy)
    store = harness.make()
    job = await store.create(filename="a.pdf")
    await store.set_succeeded(job.job_id, {"paper_id": "p"})
    assert calls == [False]
    assert (await store.get(job.job_id)).result == {"paper_id": "p"}


def _large_result() -> dict:
    return {"text": [{"id": i, "t": f"The quick brown fox {i}."} for i in range(4000)]}


def test_the_result_route_gzips_a_large_result_off_the_loop(monkeypatch):
    payload = _large_result()
    assert len(jobs_mod.encode_result(payload)) >= jobs_mod._RESULT_GZIP_MIN_BYTES
    compress_calls = []
    real_compress = gzip.compress

    def spy(data, *args, **kwargs):
        compress_calls.append(_on_loop())
        return real_compress(data, *args, **kwargs)

    monkeypatch.setattr(jobs_mod.gzip, "compress", spy)
    client = _client(MemoryJobStore(), tracker=_FakeTracker(result=payload))
    try:
        with client:
            job_id = _post(client).json()["job_id"]
            _poll_until(client, job_id, "succeeded")
            response = client.get(f"/papers/jobs/{job_id}/result")  # accepts gzip
            assert response.status_code == 200
            assert response.headers["content-encoding"] == "gzip"
            assert response.headers["vary"] == "Accept-Encoding"
            assert response.headers["content-type"] == "application/json"
            assert response.json() == payload
            assert compress_calls == [False]

            plain = client.get(
                f"/papers/jobs/{job_id}/result", headers={"Accept-Encoding": "identity"}
            )
            assert "content-encoding" not in plain.headers
            assert plain.content == jobs_mod.encode_result(payload)
    finally:
        asyncio.run(client.app.state.upload_store.close())


def test_a_small_result_is_left_to_the_middleware():
    client = _client(MemoryJobStore(), tracker=_FakeTracker(result={"paper_id": "abc"}))
    try:
        with client:
            job_id = _post(client).json()["job_id"]
            _poll_until(client, job_id, "succeeded")
            response = client.get(f"/papers/jobs/{job_id}/result")
            assert "content-encoding" not in response.headers
            assert response.json() == {"paper_id": "abc"}
    finally:
        asyncio.run(client.app.state.upload_store.close())


def test_litserves_gzip_middleware_passes_the_compressed_result_through():
    payload = _large_result()
    client = _client(MemoryJobStore(), tracker=_FakeTracker(result=payload))
    client.app.add_middleware(GZipMiddleware, minimum_size=1000)  # what LitServe installs
    try:
        with client:
            job_id = _post(client).json()["job_id"]
            _poll_until(client, job_id, "succeeded")
            response = client.get(f"/papers/jobs/{job_id}/result")
            assert response.headers["content-encoding"] == "gzip"  # once, not twice
            assert response.json() == payload
    finally:
        asyncio.run(client.app.state.upload_store.close())
