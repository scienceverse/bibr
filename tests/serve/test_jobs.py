"""Tests for the async job API (D1) — JobStore lifecycle + HTTP routes."""

import asyncio

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from bibr.config import Settings
from bibr.serve import jobs as jobs_mod
from bibr.serve.ingress import UploadStore
from bibr.serve.jobs import JobCapacityError, JobStore, register_job_routes


class _Clock:
    """Controllable monotonic clock."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeTracker:
    """Descriptor dispatch double; the real tracker is covered in ingress tests."""

    def __init__(self, *, result=None, error=None):
        self.result = result
        self.error = error
        self.descriptors = []
        self.discarded = []

    async def submit(self, descriptor, request_state=None):
        self.descriptors.append(descriptor)
        if self.error is not None:
            raise self.error
        return self.result

    async def discard(self, descriptor):
        self.discarded.append(descriptor)


# --------------------------------------------------------------------------- #
# JobStore unit tests
# --------------------------------------------------------------------------- #


class TestJobStore:
    async def test_create_and_get(self):
        store = JobStore()
        job = await store.create(filename="a.pdf")
        assert job.status == "queued"
        assert job.filename == "a.pdf"
        got = await store.get(job.job_id)
        assert got is job

    async def test_lifecycle_transitions(self):
        store = JobStore()
        job = await store.create(filename="a.pdf")
        await store.set_running(job.job_id)
        assert (await store.get(job.job_id)).status == "running"
        await store.set_succeeded(job.job_id, {"paper_id": "x"})
        done = await store.get(job.job_id)
        assert done.status == "succeeded"
        assert done.result == {"paper_id": "x"}
        assert done.duration_ms is not None
        assert done.status_dict()["result_url"] == f"/papers/jobs/{job.job_id}/result"

    async def test_failed_records_status_and_error(self):
        store = JobStore()
        job = await store.create(filename="a.pdf")
        await store.set_failed(job.job_id, http_status=422, error={"detail": "bad parse"})
        got = await store.get(job.job_id)
        assert got.status == "failed"
        assert got.http_status == 422
        assert got.status_dict()["error"] == {"detail": "bad parse"}

    async def test_ttl_purge_on_access(self, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.jobs, "ttl_seconds", 100)
        clock = _Clock()
        store = JobStore(clock=clock)
        job = await store.create(filename="a.pdf")
        await store.set_succeeded(job.job_id, {"ok": True})
        # Still within TTL.
        clock.advance(50)
        assert await store.get(job.job_id) is not None
        # Past TTL → purged lazily on access.
        clock.advance(60)
        assert await store.get(job.job_id) is None

    async def test_unfinished_jobs_never_purged(self, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.jobs, "ttl_seconds", 10)
        clock = _Clock()
        store = JobStore(clock=clock)
        job = await store.create(filename="a.pdf")
        clock.advance(10_000)
        # Queued job has no finished_mono → never expires.
        assert await store.get(job.job_id) is not None

    async def test_max_active_cap_raises(self, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.jobs, "max_active", 2)
        store = JobStore()
        await store.create(filename="1.pdf")
        await store.create(filename="2.pdf")
        with pytest.raises(JobCapacityError):
            await store.create(filename="3.pdf")

    async def test_finished_jobs_free_capacity(self, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.jobs, "max_active", 1)
        store = JobStore()
        j1 = await store.create(filename="1.pdf")
        with pytest.raises(JobCapacityError):
            await store.create(filename="2.pdf")
        await store.set_succeeded(j1.job_id, {"ok": True})
        # j1 no longer active → slot freed.
        j2 = await store.create(filename="2.pdf")
        assert j2.status == "queued"

    async def test_completed_result_retention_is_bounded(self, monkeypatch):
        monkeypatch.setattr(Settings.jobs, "max_retained", 2)
        clock = _Clock()
        store = JobStore(clock=clock)
        jobs = []
        for i in range(3):
            job = await store.create(filename=f"{i}.pdf")
            await store.set_succeeded(job.job_id, {"index": i})
            jobs.append(job)
            clock.advance(1)

        assert len(store._jobs) == 2
        assert await store.get(jobs[0].job_id) is None
        assert await store.get(jobs[1].job_id) is not None
        assert await store.get(jobs[2].job_id) is not None


class TestJobDispatcher:
    async def test_queues_fifo_and_bounds_running(self, monkeypatch):
        dispatcher_cls = getattr(jobs_mod, "JobDispatcher", None)
        payload_cls = getattr(jobs_mod, "JobPayload", None)
        assert dispatcher_cls is not None, "JobDispatcher is missing"
        assert payload_cls is not None, "JobPayload is missing"

        running = 0
        peak = 0
        started = []
        release = asyncio.Event()

        async def fake_run_job(*, job_id, **kwargs):
            nonlocal running, peak
            started.append(job_id)
            running += 1
            peak = max(peak, running)
            await release.wait()
            running -= 1

        monkeypatch.setattr(jobs_mod, "_run_job", fake_run_job)
        tracker = _FakeTracker()
        dispatcher = dispatcher_cls(store=JobStore(), tracker=tracker, max_running=2)
        payloads = []
        for index in range(4):
            payload = payload_cls(descriptor={"upload_id": str(index)})
            payloads.append(payload)
            await dispatcher.submit(str(index), payload)

        for _ in range(100):
            if len(started) == 2:
                break
            await asyncio.sleep(0)
        assert started == ["0", "1"]
        assert peak == 2

        release.set()
        await dispatcher.join()
        await dispatcher.close()
        assert started == ["0", "1", "2", "3"]

    async def test_close_discards_queued_descriptors(self, monkeypatch):
        entered = asyncio.Event()

        async def blocked_run_job(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(jobs_mod, "_run_job", blocked_run_job)
        tracker = _FakeTracker()
        dispatcher = jobs_mod.JobDispatcher(store=JobStore(), tracker=tracker, max_running=1)
        queued_descriptor = {"upload_id": "queued"}
        for index in range(2):
            descriptor = {"upload_id": "running"} if index == 0 else queued_descriptor
            payload = jobs_mod.JobPayload(descriptor=descriptor)
            await dispatcher.submit(str(index), payload)

        await entered.wait()
        await dispatcher.close()

        assert tracker.discarded == [queued_descriptor]

    async def test_stale_sweep_preserves_upload_while_job_is_queued(self):
        """Catches a later upload deleting an old descriptor before queue dispatch."""
        import io
        import os
        import time

        from starlette.datastructures import UploadFile

        from bibr.serve.ingress import InferenceDispatchTracker

        entered = asyncio.Event()
        release = asyncio.Event()

        async def dispatch(_descriptor):
            entered.set()
            await release.wait()
            return {"ok": True}

        upload_store = UploadStore.create(
            max_size=100,
            spool_memory_bytes=4,
            stale_after_seconds=10,
        )
        tracker = InferenceDispatchTracker(dispatch=dispatch, store=upload_store)
        job_store = JobStore()
        dispatcher = jobs_mod.JobDispatcher(store=job_store, tracker=tracker, max_running=1)
        try:
            first = await upload_store.persist(
                UploadFile(file=io.BytesIO(b"first"), filename="first.pdf")
            )
            second = await upload_store.persist(
                UploadFile(file=io.BytesIO(b"second"), filename="second.pdf")
            )
            first_job = await job_store.create(filename=first.filename)
            second_job = await job_store.create(filename=second.filename)
            await dispatcher.submit(
                first_job.job_id,
                jobs_mod.JobPayload(descriptor=first.to_descriptor({})),
            )
            await dispatcher.submit(
                second_job.job_id,
                jobs_mod.JobPayload(descriptor=second.to_descriptor({})),
            )
            await entered.wait()

            queued_path = upload_store.root / second.upload_id
            old = time.time() - 11
            os.utime(queued_path, (old, old))
            trigger = await upload_store.persist(
                UploadFile(file=io.BytesIO(b"trigger"), filename="trigger.pdf")
            )

            assert queued_path.read_bytes() == b"second"
            await upload_store.remove(trigger.upload_id)
        finally:
            release.set()
            await dispatcher.join()
            await dispatcher.close()
            await tracker.close()
            await upload_store.close()


# --------------------------------------------------------------------------- #
# Route tests (minimal app + monkeypatched _run_job)
# --------------------------------------------------------------------------- #


def _client(store: JobStore, *, tracker: _FakeTracker | None = None) -> TestClient:
    app = FastAPI()
    upload_store = UploadStore.create(
        max_size=Settings.pipeline.max_file_size,
        spool_memory_bytes=1,
        stale_after_seconds=120,
    )
    app.state.upload_store = upload_store
    app.state.inference_tracker = tracker or _FakeTracker(result={"paper_id": "default"})
    register_job_routes(
        app,
        store=store,
        upload_store=upload_store,
        tracker=app.state.inference_tracker,
    )
    return TestClient(app)


def _poll_until(client: TestClient, job_id: str, target: str, tries: int = 100) -> dict:
    for _ in range(tries):
        resp = client.get(f"/papers/jobs/{job_id}")
        body = resp.json()
        if body.get("status") == target:
            return body
    raise AssertionError(f"job never reached {target}: last={body}")


class TestJobRoutes:
    def test_submission_openapi_documents_202_multipart_contract(self):
        """Catches explicit parsing losing the async route's documented status/body."""
        store = JobStore()
        client = _client(store)
        try:
            operation = client.app.openapi()["paths"]["/papers/jobs"]["post"]
            assert "202" in operation["responses"]
            assert "200" not in operation["responses"]
            schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
            assert schema["additionalProperties"] is False
        finally:
            asyncio.run(client.app.state.upload_store.close())

    def test_submission_persists_shared_upload_and_queues_descriptor(self):
        store = JobStore()
        tracker = _FakeTracker(result={"paper_id": "abc"})
        client = _client(store, tracker=tracker)
        try:
            with client:
                response = client.post(
                    "/papers/jobs",
                    files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
                    data={"refs": "off", "start_page": "2"},
                    headers={"Authorization": "Bearer sk_test"},
                )
                assert response.status_code == 202
                job_id = response.json()["job_id"]
                assert response.json()["status_url"] == f"/papers/jobs/{job_id}"
                status = _poll_until(client, job_id, "succeeded")
                assert status["result_url"] == f"/papers/jobs/{job_id}/result"
                assert client.get(f"/papers/jobs/{job_id}/result").json() == {"paper_id": "abc"}

            descriptor = tracker.descriptors[0]
            assert descriptor["filename"] == "a.pdf"
            assert descriptor["start_page"] == "2"
            assert descriptor["refs"] == "off"
            assert "upload_id" in descriptor
            assert not {"path", "content", "authorization", "base_url"} & descriptor.keys()
            persisted = client.app.state.upload_store.root / descriptor["upload_id"]
            assert persisted.read_bytes() == b"%PDF-1.4"
        finally:
            asyncio.run(client.app.state.upload_store.close())

    def test_submission_caps_filename_in_descriptor_and_job_status(self):
        """Catches the job store retaining attacker-sized raw filename metadata."""
        store = JobStore()
        tracker = _FakeTracker(result={"paper_id": "abc"})
        client = _client(store, tracker=tracker)
        try:
            response = client.post(
                "/papers/jobs",
                files={"file": ("a" * 3_000, b"%PDF-1.4", "application/pdf")},
            )

            assert response.status_code == 202
            job_id = response.json()["job_id"]
            status = _poll_until(client, job_id, "succeeded")
            assert status["filename"] == "a" * 255
            assert tracker.descriptors[0]["filename"] == "a" * 255
        finally:
            asyncio.run(client.app.state.upload_store.close())

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("start_page", "9" * 100_000),
            ("refs", "bogus"),
        ],
    )
    def test_submission_rejects_invalid_options_before_job_or_descriptor(self, field, value):
        """Catches raw form metadata amplifying the queue or bypassing validation."""
        store = JobStore()
        tracker = _FakeTracker(result={"paper_id": "abc"})
        client = _client(store, tracker=tracker)
        try:
            response = client.post(
                "/papers/jobs",
                files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
                data={field: value},
            )

            assert response.status_code == 400
            assert store._jobs == {}
            assert tracker.descriptors == []
        finally:
            asyncio.run(client.app.state.upload_store.close())

    def test_empty_file_rejected(self):
        store = JobStore()
        client = _client(store)
        resp = client.post("/papers/jobs", files={"file": ("a.pdf", b"", "application/pdf")})
        assert resp.status_code == 400

    def test_oversized_file_rejected_413(self, monkeypatch):
        store = JobStore()
        monkeypatch.setattr(Settings.pipeline, "max_file_size", 10)
        client = _client(store)
        resp = client.post("/papers/jobs", files={"file": ("a.pdf", b"x" * 11, "application/pdf")})
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()
        # No job was created for the rejected upload.
        assert not store._jobs

    def test_file_at_exact_limit_is_accepted(self, monkeypatch):
        store = JobStore()
        monkeypatch.setattr(Settings.pipeline, "max_file_size", 10)
        client = _client(store)

        response = client.post(
            "/papers/jobs",
            files={"file": ("at-limit.pdf", b"x" * 10, "application/pdf")},
        )

        assert response.status_code == 202

    def test_failed_job_result_uses_recorded_status(self):
        store = JobStore()
        client = _client(store, tracker=_FakeTracker(error=HTTPException(422, "bad parse")))

        job_id = client.post(
            "/papers/jobs", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")}
        ).json()["job_id"]
        status = _poll_until(client, job_id, "failed")
        assert status["error"] == {"detail": "bad parse"}

        result = client.get(f"/papers/jobs/{job_id}/result")
        assert result.status_code == 422
        assert result.json()["detail"] == "bad parse"

    def test_result_409_while_running(self, monkeypatch):
        store = JobStore()

        # Never advances past running.
        async def fake_run_job(*, store, job_id, **kw):
            await store.set_running(job_id)

        monkeypatch.setattr(jobs_mod, "_run_job", fake_run_job)
        client = _client(store)

        job_id = client.post(
            "/papers/jobs", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")}
        ).json()["job_id"]
        _poll_until(client, job_id, "running")
        resp = client.get(f"/papers/jobs/{job_id}/result")
        assert resp.status_code == 409
        assert resp.json()["status"] == "running"

    def test_unknown_job_404(self):
        store = JobStore()
        client = _client(store)
        assert client.get("/papers/jobs/nope").status_code == 404
        assert client.get("/papers/jobs/nope/result").status_code == 404

    def test_max_active_returns_429(self, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.jobs, "max_active", 1)
        store = JobStore()

        # Keep jobs in "running" so they stay active and consume capacity.
        async def fake_run_job(*, store, job_id, **kw):
            await store.set_running(job_id)

        monkeypatch.setattr(jobs_mod, "_run_job", fake_run_job)
        client = _client(store)

        first = client.post(
            "/papers/jobs", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")}
        )
        assert first.status_code == 202
        second = client.post(
            "/papers/jobs", files={"file": ("b.pdf", b"%PDF-1.4", "application/pdf")}
        )
        assert second.status_code == 429

    def test_max_active_rejects_before_private_upload_persistence(self, monkeypatch):
        """Catches copying a full admitted body after the process-local queue is full."""
        monkeypatch.setattr(Settings.jobs, "max_active", 1)
        store = JobStore()
        asyncio.run(store.create(filename="active.pdf"))
        client = _client(store)
        upload_store = client.app.state.upload_store
        real_persist = upload_store.persist
        persist_calls = 0

        async def spy_persist(upload):
            nonlocal persist_calls
            persist_calls += 1
            return await real_persist(upload)

        monkeypatch.setattr(upload_store, "persist", spy_persist)
        try:
            response = client.post(
                "/papers/jobs",
                files={"file": ("rejected.pdf", b"%PDF-1.4", "application/pdf")},
            )

            assert response.status_code == 429
            assert persist_calls == 0
        finally:
            asyncio.run(upload_store.close())

    def test_parser_disk_exhaustion_returns_sanitized_507(self, monkeypatch):
        """Catches job multipart spool ENOSPC escaping as an internal error."""
        import errno

        from starlette.datastructures import UploadFile

        async def fail_write(_self, _data):
            raise OSError(errno.ENOSPC, "private job spool path")

        monkeypatch.setattr(UploadFile, "write", fail_write)
        store = JobStore()
        base_client = _client(store)
        client = TestClient(base_client.app, raise_server_exceptions=False)
        try:
            response = client.post(
                "/papers/jobs",
                files={"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")},
            )

            assert response.status_code == 507
            assert response.json() == {"detail": "Insufficient temporary storage"}
            assert "private job spool path" not in response.text
            assert store._jobs == {}
        finally:
            asyncio.run(base_client.app.state.upload_store.close())

    def test_repeated_file_part_is_rejected_before_job_creation(self):
        """Catches job ingress admitting multiple independent file spools."""
        store = JobStore()
        client = _client(store)
        try:
            response = client.post(
                "/papers/jobs",
                files=[
                    ("file", ("first.pdf", b"first", "application/pdf")),
                    ("file", ("second.pdf", b"second", "application/pdf")),
                ],
            )

            assert response.status_code == 400
            assert store._jobs == {}
        finally:
            asyncio.run(client.app.state.upload_store.close())

    def test_enqueue_failure_discards_descriptor_and_reservation(self, monkeypatch):
        store = JobStore()
        tracker = _FakeTracker()

        async def fail_submit(*args, **kwargs):
            raise RuntimeError("dispatcher stopped")

        monkeypatch.setattr(jobs_mod.JobDispatcher, "submit", fail_submit)
        client = _client(store, tracker=tracker)
        try:
            with client:
                with pytest.raises(RuntimeError, match="dispatcher stopped"):
                    client.post("/papers/jobs", files={"file": ("a.pdf", b"%PDF")})

            assert len(tracker.discarded) == 1
            assert store._jobs == {}
        finally:
            asyncio.run(client.app.state.upload_store.close())


# --------------------------------------------------------------------------- #
# _run_job descriptor dispatch mapping
# --------------------------------------------------------------------------- #


class TestRunJobDispatch:
    async def test_success_records_result(self):
        store = JobStore()
        job = await store.create(filename="a.pdf")
        tracker = _FakeTracker(result={"paper_id": "xyz"})
        descriptor = {"upload_id": "opaque"}
        await jobs_mod._run_job(
            store=store,
            job_id=job.job_id,
            descriptor=descriptor,
            tracker=tracker,
        )
        got = await store.get(job.job_id)
        assert got.status == "succeeded"
        assert got.result == {"paper_id": "xyz"}
        assert tracker.descriptors == [descriptor]

    async def test_http_exception_records_compatible_failure(self):
        store = JobStore()
        job = await store.create(filename="a.pdf")
        tracker = _FakeTracker(error=HTTPException(status_code=422, detail="bad parse"))
        await jobs_mod._run_job(
            store=store,
            job_id=job.job_id,
            descriptor={"upload_id": "opaque"},
            tracker=tracker,
        )
        got = await store.get(job.job_id)
        assert got.status == "failed"
        assert got.http_status == 422
        assert got.error == {"detail": "bad parse"}

    async def test_structured_http_detail_is_preserved(self):
        store = JobStore()
        job = await store.create(filename="a.pdf")
        detail = {
            "message": "LLM returned invalid structured output",
            "error_code": "llm_invalid_output",
        }
        await jobs_mod._run_job(
            store=store,
            job_id=job.job_id,
            descriptor={"upload_id": "opaque"},
            tracker=_FakeTracker(error=HTTPException(status_code=422, detail=detail)),
        )

        got = await store.get(job.job_id)
        assert got.http_status == 422
        assert got.error == detail

    async def test_unexpected_exception_is_sanitized(self):
        store = JobStore()
        job = await store.create(filename="a.pdf")
        await jobs_mod._run_job(
            store=store,
            job_id=job.job_id,
            descriptor={"upload_id": "opaque"},
            tracker=_FakeTracker(error=RuntimeError("secret upstream failure")),
        )
        got = await store.get(job.job_id)
        assert got.status == "failed"
        assert got.http_status == 500
        assert got.error == {"detail": "internal job error"}


# --------------------------------------------------------------------------- #
# Auth gate applies to job routes (full server)
# --------------------------------------------------------------------------- #


class TestJobRoutesAuth:
    @pytest.fixture(autouse=True)
    def _restore_api_key(self):
        from bibr.config import Settings

        original = Settings.auth.api_key
        yield
        Settings.auth.api_key = original

    def test_jobs_route_requires_auth_when_configured(self, monkeypatch):
        pytest.importorskip("litserve")
        from bibr.config import Settings

        monkeypatch.setattr(Settings.auth, "api_key", "sk_int")
        from bibr.serve.app import build_server

        client = TestClient(build_server().app, raise_server_exceptions=False)
        resp = client.post(
            "/papers/jobs", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")}
        )
        assert resp.status_code == 401
