"""``RedisJobStore``: job state shared between bibr-serve replicas (fakeredis-backed).

The backend-agnostic store contract is covered by the parametrized
``TestJobStore`` in ``test_jobs.py``; this module pins what only the shared
store has to get right — key layout, TTLs, the global cap and retention across
replica instances, cross-replica visibility, bounded Redis calls, and the
503 / logged-and-swallowed outage semantics.
"""

import asyncio
import logging
import sys
import time
import zlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("redis")
fakeredis = pytest.importorskip("fakeredis")
pytest.importorskip("lupa")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bibr.config import Settings
from bibr.exceptions import ConfigurationError
from bibr.serve import jobs as jobs_mod
from bibr.serve.jobs import (
    JobCapacityError,
    JobStoreUnavailableError,
    MemoryJobStore,
    build_job_store,
    default_replica_id,
)
from bibr.serve.jobs_redis import RedisJobStore
from tests.serve.test_jobs import _client, _Clock, _FakeTracker, _poll_until, _RedisHarness


@pytest.fixture
async def harness():
    harness = _RedisHarness()
    try:
        yield harness
    finally:
        await harness.close()


# --------------------------------------------------------------------------- #
# Key layout, TTLs
# --------------------------------------------------------------------------- #


async def test_result_is_stored_zlib_compressed_under_its_own_key(harness):
    store = harness.make()
    job = await store.create(filename="a.pdf")
    payload = {"text": "x" * 10_000}
    encoded = jobs_mod.encode_result(payload)

    await store.set_succeeded(job.job_id, payload)

    raw = await store._redis.get(store.result_key(job.job_id))
    assert raw != encoded
    assert len(raw) < len(encoded)
    assert zlib.decompress(raw) == encoded
    fields = await store._redis.hgetall(store.job_key(job.job_id))
    assert fields[b"status"] == b"succeeded"
    assert fields[b"replica"] == b"replica-a"
    # The byte budget is charged at the encoded size, like the memory store.
    assert int(fields[b"result_size"]) == len(encoded)
    assert b"result" not in fields  # the body never rides along with a status poll
    got = await store.get(job.job_id)
    assert got.result_json == encoded
    assert got.result_size == len(encoded)
    summary = await store.get(job.job_id, include_result=False)
    assert summary.result_json is None
    assert summary.result_size == len(encoded)


async def test_finished_keys_carry_the_job_ttl_and_active_records_a_safety_ttl(
    harness, monkeypatch
):
    monkeypatch.setattr(Settings.jobs, "ttl_seconds", 120)
    store = harness.make(active_ttl_seconds=600)
    job = await store.create(filename="a.pdf")
    assert 120 < await store._redis.ttl(store.job_key(job.job_id)) <= 600
    assert await store._redis.sismember(store.active_key, job.job_id)

    await store.set_succeeded(job.job_id, {"ok": True})

    assert 0 < await store._redis.ttl(store.job_key(job.job_id)) <= 120
    assert 0 < await store._redis.ttl(store.result_key(job.job_id)) <= 120
    assert not await store._redis.sismember(store.active_key, job.job_id)
    assert await store._redis.zscore(store.finished_key, job.job_id) is not None


async def test_get_treats_a_record_past_the_ttl_as_gone_before_redis_reaps_it(harness, monkeypatch):
    monkeypatch.setattr(Settings.jobs, "ttl_seconds", 100)
    clock = _Clock()
    store = harness.make(clock)
    job = await store.create(filename="a.pdf")
    await store.set_succeeded(job.job_id, {"ok": True})
    clock.advance(101)
    # The fake server's clock has not moved, so the keys still exist...
    assert await store._redis.exists(store.job_key(job.job_id)) == 1
    # ...yet the store's own TTL window already says the job is gone.
    assert await store.get(job.job_id) is None
    # The next finish prunes the stale index entry and keys (global tidy-up).
    other = await store.create(filename="b.pdf")
    await store.set_failed(other.job_id, http_status=422, error={"detail": "x"})
    assert await store._redis.exists(store.job_key(job.job_id)) == 0
    assert await store._redis.zscore(store.finished_key, job.job_id) is None


# --------------------------------------------------------------------------- #
# Global cap and retention across replica instances sharing one server
# --------------------------------------------------------------------------- #


async def test_active_cap_is_global_across_replicas(harness, monkeypatch):
    monkeypatch.setattr(Settings.jobs, "max_active", 2)
    a = harness.make(replica_id="replica-a")
    b = harness.make(replica_id="replica-b")
    first = await a.create(filename="1.pdf")
    await b.create(filename="2.pdf")

    with pytest.raises(JobCapacityError, match=r"\(2/2\)"):
        await a.create(filename="3.pdf")
    with pytest.raises(JobCapacityError):
        await b.create(filename="3.pdf")

    await a.set_failed(first.job_id, http_status=422, error={"detail": "x"})
    assert (await b.create(filename="3.pdf")).status == "queued"


async def test_cap_slot_returns_once_a_dead_replicas_record_expires(harness, monkeypatch):
    monkeypatch.setattr(Settings.jobs, "max_active", 1)
    a = harness.make(replica_id="replica-a")
    b = harness.make(replica_id="replica-b")
    job = await a.create(filename="a.pdf")
    with pytest.raises(JobCapacityError):
        await b.create(filename="b.pdf")

    # replica-a died mid-job; its record reaches the safety TTL (simulated by
    # dropping the hash) while the id still sits in the active set.
    await a._redis.delete(a.job_key(job.job_id))
    assert await b._redis.sismember(b.active_key, job.job_id)

    assert (await b.create(filename="b.pdf")).status == "queued"
    assert not await b._redis.sismember(b.active_key, job.job_id)


async def test_retention_prunes_across_replicas(harness, monkeypatch):
    monkeypatch.setattr(Settings.jobs, "max_retained", 2)
    clock = _Clock()
    a = harness.make(clock, replica_id="replica-a")
    b = harness.make(clock, replica_id="replica-b")
    jobs = []
    for index, store in enumerate((a, a, b)):
        job = await store.create(filename=f"{index}.pdf")
        await store.set_succeeded(job.job_id, {"index": index})
        jobs.append(job)
        clock.advance(1)

    # b's finish evicted a's oldest result: the order is global, not per replica.
    assert await a.get(jobs[0].job_id) is None
    assert await b.get(jobs[0].job_id) is None
    assert await a._redis.exists(a.result_key(jobs[0].job_id)) == 0
    assert await a._redis.zcard(a.finished_key) == 2
    assert (await a.get(jobs[2].job_id)).result == {"index": 2}


async def test_byte_budget_prunes_across_replicas(harness, monkeypatch):
    payload = {"text": "x" * 100}
    one = len(jobs_mod.encode_result(payload))
    monkeypatch.setattr(Settings.jobs, "max_retained_bytes", 2 * one)
    clock = _Clock()
    a = harness.make(clock, replica_id="replica-a")
    b = harness.make(clock, replica_id="replica-b")
    jobs = []
    for index, store in enumerate((a, b, a)):
        job = await store.create(filename=f"{index}.pdf")
        await store.set_succeeded(job.job_id, payload)
        jobs.append(job)
        clock.advance(1)

    assert await b.get(jobs[0].job_id) is None
    assert (await b.get(jobs[1].job_id)).result == payload
    assert (await a.get(jobs[2].job_id)).result == payload


# --------------------------------------------------------------------------- #
# Cross-replica visibility
# --------------------------------------------------------------------------- #


async def test_status_and_result_are_visible_from_another_replica(harness):
    clock = _Clock()
    a = harness.make(clock, replica_id="replica-a")
    b = harness.make(clock, replica_id="replica-b")
    job = await a.create(filename="a.pdf")

    seen = await b.get(job.job_id)
    assert seen.status == "queued"
    assert seen.replica == "replica-a"  # status only: b never executes a's job

    await a.set_running(job.job_id)
    assert (await b.get(job.job_id, include_result=False)).status == "running"

    clock.advance(3)
    await a.set_succeeded(job.job_id, {"paper_id": "p"})
    seen = await b.get(job.job_id)
    assert seen.status == "succeeded"
    assert seen.replica == "replica-a"
    assert seen.result == {"paper_id": "p"}
    status = seen.status_dict()
    assert status["replica"] == "replica-a"
    assert status["duration_ms"] == 3000
    assert status["result_url"] == f"/papers/jobs/{job.job_id}/result"


async def test_failure_details_round_trip_between_replicas(harness):
    a = harness.make(replica_id="replica-a")
    b = harness.make(replica_id="replica-b")
    job = await a.create(filename="a.pdf")
    detail = {"message": "LLM returned invalid structured output", "error_code": "llm"}
    await a.set_failed(job.job_id, http_status=422, error=detail)

    seen = await b.get(job.job_id)
    assert seen.status == "failed"
    assert seen.http_status == 422
    assert seen.error == detail
    assert seen.status_dict()["error"] == detail


async def test_default_replica_id_is_hostname_and_pid():
    import os
    import socket

    assert default_replica_id() == f"{socket.gethostname()}:{os.getpid()}"
    store = RedisJobStore(
        "redis://fake",
        client_factory=lambda _url: fakeredis.aioredis.FakeRedis(),
    )
    try:
        assert store.replica_id == default_replica_id()
        assert store.key_prefix == "bibr:jobs"
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# Outage semantics: bounded calls, 503 on the routes, logged in the runner
# --------------------------------------------------------------------------- #


class _HangingPipeline:
    """Queues anything, and ``execute`` never answers."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: self

    async def execute(self, *args, **kwargs):  # noqa: ARG002
        await asyncio.Event().wait()


class _HangingRedis(fakeredis.aioredis.FakeRedis):
    """A Redis that accepts every command and never answers."""

    async def evalsha(self, *args, **kwargs):  # noqa: ARG002
        await asyncio.Event().wait()

    async def eval(self, *args, **kwargs):  # noqa: ARG002
        await asyncio.Event().wait()

    async def hgetall(self, *args, **kwargs):  # noqa: ARG002
        await asyncio.Event().wait()

    async def ping(self, *args, **kwargs):  # noqa: ARG002
        await asyncio.Event().wait()

    def pipeline(self, *args, **kwargs):  # noqa: ARG002
        return _HangingPipeline()


async def test_every_call_is_bounded_by_the_redis_timeouts():
    server = fakeredis.FakeServer()
    store = RedisJobStore(
        "redis://fake",
        key_prefix="test:jobs",
        connect_timeout=0.02,
        socket_timeout=0.03,
        client_factory=lambda _url: _HangingRedis(server=server),
    )
    assert store.operation_timeout == pytest.approx(0.05)
    try:
        raising = (
            lambda: store.create(filename="a.pdf"),
            lambda: store.get("x"),
            lambda: store.get("x", include_result=False),
            lambda: store.ping(),
        )
        for call in raising:
            started = time.monotonic()
            with pytest.raises(JobStoreUnavailableError, match="TimeoutError"):
                await asyncio.wait_for(call(), timeout=5)
            assert time.monotonic() - started < 1.0
        # Transitions are bounded the same way but never raise.
        swallowing = (
            lambda: store.set_running("x"),
            lambda: store.set_succeeded("x", {"ok": True}),
            lambda: store.set_failed("x", http_status=500, error={"detail": "x"}),
            lambda: store.discard("x"),
        )
        for call in swallowing:
            started = time.monotonic()
            await asyncio.wait_for(call(), timeout=5)
            assert time.monotonic() - started < 1.0
    finally:
        await store.close()


async def test_completed_redis_call_does_not_swallow_shutdown_cancellation(harness):
    store = harness.make()
    reply = asyncio.get_running_loop().create_future()
    operation = asyncio.create_task(store._bounded("ping", reply))
    await asyncio.sleep(0)  # suspend the operation while Redis is still pending

    # Redis completes just as the dispatcher cancels its worker on shutdown.
    # Python 3.11's wait_for can return the reply and lose this cancellation,
    # leaving the worker waiting forever for another job during close().
    reply.set_result(True)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation


async def test_runner_logs_and_survives_a_redis_outage(harness, caplog):
    store = harness.make()
    job = await store.create(filename="a.pdf")
    harness.server.connected = False
    with caplog.at_level(logging.ERROR, logger="bibr.serve.jobs.redis"):
        await jobs_mod._run_job(
            store=store,
            job_id=job.job_id,
            descriptor={"upload_id": "x"},
            tracker=_FakeTracker(result={"paper_id": "p"}),
        )
    messages = [record.getMessage() for record in caplog.records]
    assert any(job.job_id in message and "running" in message for message in messages)
    assert any(job.job_id in message and "succeeded" in message for message in messages)

    harness.server.connected = True
    # Nothing was recorded; the TTL will reap the stale record. The runner is intact.
    assert (await store.get(job.job_id)).status == "queued"
    nxt = await store.create(filename="b.pdf")
    await jobs_mod._run_job(
        store=store,
        job_id=nxt.job_id,
        descriptor={"upload_id": "y"},
        tracker=_FakeTracker(result={"paper_id": "q"}),
    )
    assert (await store.get(nxt.job_id)).result == {"paper_id": "q"}


async def test_dispatcher_worker_survives_a_store_that_raises(harness, caplog):
    """A store bug must not kill the worker loop and strand the queue behind it."""

    class _Broken(RedisJobStore):
        async def set_succeeded(self, job_id, result):
            raise RuntimeError("store bug")

    store = _Broken("redis://fake", key_prefix="test:jobs", client_factory=harness.client)
    harness.stores.append(store)
    tracker = _FakeTracker(result={"paper_id": "p"})
    dispatcher = jobs_mod.JobDispatcher(store=store, tracker=tracker, max_running=1)
    try:
        for index in range(2):
            await dispatcher.submit(
                str(index), jobs_mod.JobPayload(descriptor={"upload_id": index})
            )
        await asyncio.wait_for(dispatcher.join(), timeout=5)
    finally:
        await dispatcher.close()
    # Both jobs were dispatched: the first crash did not take the worker down.
    assert [d["upload_id"] for d in tracker.descriptors] == [0, 1]
    assert [r.getMessage() for r in caplog.records if "runner failed" in r.getMessage()] == [
        "job 0: runner failed outside the job's own handling",
        "job 1: runner failed outside the job's own handling",
    ]


def test_routes_round_trip_on_the_redis_store():
    harness = _RedisHarness()
    store = harness.make()
    tracker = _FakeTracker(result={"paper_id": "abc"})
    client = _client(store, tracker=tracker)
    upload_store = client.app.state.upload_store
    try:
        with client:
            response = client.post(
                "/papers/jobs",
                files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
            )
            assert response.status_code == 202
            job_id = response.json()["job_id"]
            status = _poll_until(client, job_id, "succeeded")
            assert status["replica"] == "replica-a"
            assert client.get(f"/papers/jobs/{job_id}/result").json() == {"paper_id": "abc"}

            # The body lives under its own key: if it goes before the record does,
            # the result route answers 404 instead of an empty 200.
            client.portal.call(store._redis.delete, store.result_key(job_id))
            missing = client.get(f"/papers/jobs/{job_id}/result")
            assert missing.status_code == 404
            assert missing.json() == {"detail": "job result no longer available"}
            assert client.get(f"/papers/jobs/{job_id}").json()["status"] == "succeeded"
    finally:
        asyncio.run(upload_store.close())
        asyncio.run(harness.close())


def test_submit_returns_503_and_keeps_no_upload_when_redis_is_down():
    harness = _RedisHarness()
    store = harness.make()
    tracker = _FakeTracker(result={"paper_id": "abc"})
    client = _client(store, tracker=tracker)
    upload_store = client.app.state.upload_store
    try:
        with client:
            harness.server.connected = False
            response = client.post(
                "/papers/jobs",
                files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
            )
            assert response.status_code == 503
            assert response.json() == {"detail": "job store unavailable"}
            # Nothing was persisted or queued: create runs before persist, and the
            # multipart spool is released with the request.
            assert list(upload_store.root.iterdir()) == []
            assert tracker.descriptors == []
            assert client.get("/papers/jobs/deadbeef").status_code == 503
            assert client.get("/papers/jobs/deadbeef/result").status_code == 503
            assert client.get("/papers/jobs/deadbeef").json() == {"detail": "job store unavailable"}

            harness.server.connected = True
            assert client.get("/papers/jobs/deadbeef").status_code == 404
            assert (
                client.post(
                    "/papers/jobs",
                    files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
                ).status_code
                == 202
            )
    finally:
        asyncio.run(upload_store.close())
        asyncio.run(harness.close())


# --------------------------------------------------------------------------- #
# Readiness and configuration
# --------------------------------------------------------------------------- #


def test_readiness_reports_the_job_store(monkeypatch):
    from bibr.serve.app import _register_readiness_route

    harness = _RedisHarness()
    store = harness.make()
    monkeypatch.setattr(Settings.jobs, "enabled", True)
    monkeypatch.setattr(Settings.jobs, "store", "redis")
    monkeypatch.setattr(Settings.auth, "api_key", None)
    # A closed port: the OCR check fails fast instead of waiting on a real server.
    monkeypatch.setattr(Settings, "OCR_BASE_URL", "http://127.0.0.1:9")

    class _Server:
        app = FastAPI()

    server = _Server()
    server.app.state.job_store = store
    _register_readiness_route(server, Settings)
    try:
        with TestClient(server.app) as client:
            checks = client.get("/ready").json()["checks"]
            assert checks["jobs_store"] == "ok"

            harness.server.connected = False
            response = client.get("/ready")
            assert response.status_code == 503
            assert response.json()["status"] == "not_ready"
            assert response.json()["checks"]["jobs_store"] == "error"
    finally:
        asyncio.run(harness.close())


def test_readiness_skips_the_job_store_check_for_the_memory_store(monkeypatch):
    from bibr.serve.app import _register_readiness_route

    monkeypatch.setattr(Settings.jobs, "store", "memory")
    monkeypatch.setattr(Settings.auth, "api_key", None)
    monkeypatch.setattr(Settings, "OCR_BASE_URL", "http://127.0.0.1:9")

    class _Server:
        app = FastAPI()

    server = _Server()
    server.app.state.job_store = MemoryJobStore()
    _register_readiness_route(server, Settings)
    with TestClient(server.app) as client:
        assert "jobs_store" not in client.get("/ready").json()["checks"]


def test_build_job_store_defaults_to_memory(monkeypatch):
    monkeypatch.setattr(Settings.jobs, "store", "memory")
    monkeypatch.setattr(Settings.jobs, "replica_id", None)
    store = build_job_store(Settings)
    assert isinstance(store, MemoryJobStore)
    assert store.replica_id == default_replica_id()


def test_build_job_store_requires_a_url_for_redis(monkeypatch):
    monkeypatch.setattr(Settings.jobs, "store", "redis")
    monkeypatch.setattr(Settings.jobs, "redis_url", None)
    monkeypatch.setattr(Settings.redis, "url", None)
    with pytest.raises(ConfigurationError, match="JOBS_REDIS_URL"):
        build_job_store(Settings)


def test_build_job_store_falls_back_to_the_cache_redis_url(monkeypatch):
    monkeypatch.setattr(Settings.jobs, "store", "redis")
    monkeypatch.setattr(Settings.jobs, "redis_url", None)
    monkeypatch.setattr(Settings.jobs, "replica_id", "api-1")
    monkeypatch.setattr(Settings.jobs, "key_prefix", "staging:jobs")
    monkeypatch.setattr(Settings.redis, "url", "redis://cache-host:6379/0")
    monkeypatch.setattr(Settings.redis, "connect_timeout_seconds", 1.5)
    monkeypatch.setattr(Settings.redis, "socket_timeout_seconds", 2.5)
    store = build_job_store(Settings)
    try:
        assert isinstance(store, RedisJobStore)
        assert store.replica_id == "api-1"
        assert store.key_prefix == "staging:jobs"
        assert store.operation_timeout == pytest.approx(4.0)
        kwargs = store._redis.connection_pool.connection_kwargs
        assert kwargs["host"] == "cache-host"
        assert kwargs["socket_connect_timeout"] == 1.5
        assert kwargs["socket_timeout"] == 2.5
    finally:
        asyncio.run(store.close())


def test_build_job_store_reports_a_missing_redis_package(monkeypatch):
    monkeypatch.setattr(Settings.jobs, "store", "redis")
    monkeypatch.setattr(Settings.jobs, "redis_url", "redis://localhost:6379/0")
    monkeypatch.delitem(sys.modules, "bibr.serve.jobs_redis", raising=False)
    monkeypatch.setitem(sys.modules, "redis", None)
    with pytest.raises(ConfigurationError, match=r"bibr\[cache\]"):
        build_job_store(Settings)


def test_job_store_settings_stay_out_of_the_behavior_fingerprint(monkeypatch):
    from bibr.config import compute_behavior_fingerprint

    before = compute_behavior_fingerprint(Settings)
    monkeypatch.setattr(Settings.jobs, "store", "redis")
    monkeypatch.setattr(Settings.jobs, "redis_url", "redis://elsewhere:6379/1")
    monkeypatch.setattr(Settings.jobs, "key_prefix", "other:jobs")
    monkeypatch.setattr(Settings.jobs, "replica_id", "api-9")
    assert compute_behavior_fingerprint(Settings) == before
