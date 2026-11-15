"""What a client sees of the inference worker boundary, pinned on a real process.

Each test posts to the public ``/papers/extract`` route of a server whose
inference worker is a real spawned process running ``_worker_stub``. The
stub's outcome is chosen by the uploaded file's name. These are the
behaviours bibr relies on from the process boundary: a worker's
``HTTPException`` keeps its status and detail, any other worker exception
is a generic 500, a request that waited in the queue longer than the
timeout is a 504, ``/health`` says ``ok`` once the worker is ready, and only
the descriptor crosses (the upload is consumed in the worker).
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os
import time
from dataclasses import dataclass

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

PDF = b"%PDF-1.7\nworker contract\n%%EOF"


@dataclass
class Harness:
    client: TestClient
    store: object
    worker_pids: list[int]
    worker: object = None


@contextlib.contextmanager
def _bibr_harness(*, timeout: float, restart: bool = False):
    from fastapi import FastAPI

    from bibr.serve.app import _register_health_route
    from bibr.serve.ingress import InferenceDispatchTracker, UploadStore, register_extract_route
    from bibr.serve.worker import InferenceWorker
    from tests.serve._worker_stub import STUB_MAX_SIZE, StubBehaviour

    store = UploadStore.create(
        max_size=STUB_MAX_SIZE, spool_memory_bytes=1 << 20, stale_after_seconds=600
    )
    worker = InferenceWorker(StubBehaviour(str(store.root)), timeout=timeout, restart=restart)
    try:
        worker.start()
        app = FastAPI()
        tracker = InferenceDispatchTracker(dispatch=worker.dispatch, store=store)
        register_extract_route(app, store, tracker)
        _register_health_route(app, worker)
        with TestClient(app) as client:
            yield Harness(client, store, [worker.pid], worker)
    finally:
        worker.stop()
        store.close_sync()


@contextlib.contextmanager
def _litserve_harness(*, timeout: float):
    ls = pytest.importorskip("litserve")
    import litserve.server as ls_server

    from bibr.serve.ingress import InferenceDispatchTracker, UploadStore, register_extract_route
    from tests.serve._worker_stub import STUB_MAX_SIZE, LitServeStubAPI

    def resolve_litserve_dispatch(app):
        (route,) = [r for r in app.routes if getattr(r, "path", None) == "/_bibr/inference"]
        return route.endpoint

    ls_server._MCP_AVAILABLE = False
    store = UploadStore.create(
        max_size=STUB_MAX_SIZE, spool_memory_bytes=1 << 20, stale_after_seconds=600
    )
    api = LitServeStubAPI(str(store.root))
    server = ls.LitServer(api, accelerator="cpu", devices=1, timeout=timeout)
    server.inference_workers = []
    manager = server._init_manager(1)
    try:
        server.inference_workers.extend(server.launch_inference_worker(api))
        deadline = time.monotonic() + 30
        while not all(s == "ready" for s in server.workers_setup_status.values()):
            assert time.monotonic() < deadline, "worker did not start"
            time.sleep(0.05)
        server.app.response_queue_id = 0
        tracker = InferenceDispatchTracker(
            dispatch=resolve_litserve_dispatch(server.app), store=store
        )
        register_extract_route(server.app, store, tracker)
        with TestClient(server.app) as client:
            yield Harness(client, store, [w.pid for w in server.inference_workers])
    finally:
        server._perform_graceful_shutdown(manager, {}, "test")
        store.close_sync()


HARNESSES = {"litserve": _litserve_harness, "bibr": _bibr_harness}


@pytest.fixture(params=sorted(HARNESSES))
def impl(request):
    return request.param


@pytest.fixture
def harness(impl):
    with HARNESSES[impl](timeout=30) as h:
        yield h


def _post(client: TestClient, name: str, **fields):
    return client.post(
        "/papers/extract",
        files={"file": (name, PDF, "application/pdf")},
        data=fields or None,
    )


def test_success_crosses_the_process_and_consumes_the_upload(harness):
    response = _post(harness.client, "ok.pdf", refs="off")
    assert response.status_code == 200
    body = response.json()
    assert body["size"] == len(PDF)
    assert body["pid"] in harness.worker_pids
    assert body["pid"] != os.getpid()
    assert body["options"] == {"refs": "off"}
    assert list(harness.store.root.iterdir()) == []


def test_worker_http_exception_keeps_status_and_dict_detail(harness):
    response = _post(harness.client, "http_dict.pdf")
    assert response.status_code == 422
    assert response.json() == {"detail": {"message": "bad paper", "error_code": "stub_code"}}


def test_worker_http_exception_keeps_status_and_string_detail(harness):
    response = _post(harness.client, "http_str.pdf")
    assert response.status_code == 400
    assert response.json() == {"detail": "Filename required"}


def test_other_worker_exception_is_a_generic_500(harness):
    response = _post(harness.client, "crash.pdf")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "internal detail" not in response.text


def test_health_says_ok_once_the_worker_is_ready(harness):
    response = harness.client.get("/health")
    assert response.status_code == 200
    assert response.text == "ok"


def test_concurrent_requests_overlap_in_the_one_worker(harness):
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        responses = list(pool.map(lambda _: _post(harness.client, "slow.pdf"), range(3)))
    assert [r.status_code for r in responses] == [200, 200, 200]
    assert len({r.json()["pid"] for r in responses}) == 1
    # Three 3-second predicts in one async worker finish together.
    assert time.monotonic() - started < 8.0


def test_a_request_that_waited_past_the_timeout_is_a_504(impl):
    with HARNESSES[impl](timeout=1) as h:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            blocked = pool.submit(_post, h.client, "block.pdf")
            time.sleep(0.5)
            late = pool.submit(_post, h.client, "ok.pdf")
            assert blocked.result(timeout=30).status_code == 200
            response = late.result(timeout=30)
    assert response.status_code == 504
    assert response.json() == {"detail": "Request timed out"}


# ---------------------------------------------------------------- bibr's own worker
#
# LitServe 0.2.x leaves a request stranded when its worker dies (the waiter
# never hears back) and replaces workers only as an unsupported opt-in; these
# pin what bibr's worker does instead.


def test_worker_death_answers_the_waiter_and_calls_on_fatal():
    from bibr.serve.worker import WORKER_STOPPED_DETAIL, WORKER_UNAVAILABLE_DETAIL

    fatal = []
    with _bibr_harness(timeout=30) as h:
        h.worker.on_fatal = lambda: fatal.append(True)
        started = time.monotonic()
        response = _post(h.client, "die.pdf")
        assert response.status_code == 500
        assert response.json() == {"detail": WORKER_STOPPED_DETAIL}
        assert time.monotonic() - started < 10
        deadline = time.monotonic() + 10
        while not fatal and time.monotonic() < deadline:
            time.sleep(0.05)
        assert fatal == [True]
        assert h.worker.died
        assert h.client.get("/health").status_code == 503
        later = _post(h.client, "ok.pdf")
        assert later.status_code == 503
        assert later.json() == {"detail": WORKER_UNAVAILABLE_DETAIL}


def test_worker_death_answers_every_waiter_in_flight():
    with _bibr_harness(timeout=30) as h:
        h.worker.on_fatal = lambda: None
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            slow = [pool.submit(_post, h.client, "slow.pdf") for _ in range(2)]
            time.sleep(0.5)
            dying = pool.submit(_post, h.client, "die.pdf")
            statuses = [f.result(timeout=20).status_code for f in (*slow, dying)]
    assert statuses == [500, 500, 500]


def test_restart_replaces_the_dead_worker():
    with _bibr_harness(timeout=30, restart=True) as h:
        first_pid = h.worker.pid
        assert _post(h.client, "die.pdf").status_code == 500
        deadline = time.monotonic() + 30
        while not (h.worker.ready and h.worker.pid != first_pid):
            assert time.monotonic() < deadline, "no replacement worker"
            time.sleep(0.05)
        response = _post(h.client, "ok.pdf")
        assert response.status_code == 200
        assert response.json()["pid"] == h.worker.pid != first_pid
        assert not h.worker.died


def test_a_failed_setup_fails_start(tmp_path):
    from bibr.serve.worker import InferenceWorker, WorkerStartError
    from tests.serve._worker_stub import FailingSetupStub

    worker = InferenceWorker(FailingSetupStub(str(tmp_path)), timeout=30)
    with pytest.raises(WorkerStartError, match="setup failed: RuntimeError"):
        worker.start()
    assert not worker.ready


def test_stop_answers_a_request_still_in_flight():
    from bibr.serve.worker import WORKER_STOPPED_DETAIL

    with _bibr_harness(timeout=30) as h:
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            pending = pool.submit(_post, h.client, "slow.pdf")
            time.sleep(0.5)
            h.worker.stop()
            response = pending.result(timeout=20)
    assert response.status_code == 500
    assert response.json() == {"detail": WORKER_STOPPED_DETAIL}
