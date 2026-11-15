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


# The LitServe flavour that pinned these behaviours before the swap ran the same
# assertions; it went with the dependency.
HARNESSES = {"bibr": _bibr_harness}


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


# ---------------------------------------------------------------- death and restart
#
# LitServe 0.2.x left a request stranded when its worker died (the waiter never
# heard back) and replaced workers only as an unsupported opt-in; these pin what
# bibr's worker does instead.


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


@contextlib.contextmanager
def _stub_server_process():
    """``BibrServer.run`` over the stub worker in its own process group."""
    import socket
    import subprocess
    import sys
    from pathlib import Path

    import httpx

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    root = Path(__file__).resolve().parents[2]
    process = subprocess.Popen(  # noqa: S603 - fixed argv
        [sys.executable, "-m", "tests.serve._stub_server", str(port)],
        cwd=root,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            assert process.poll() is None, process.stdout.read()
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert time.monotonic() < deadline, "stub server did not start"
            time.sleep(0.1)
        yield process, base
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=30)


def _post_url(base: str, name: str):
    import httpx

    return httpx.post(
        f"{base}/papers/extract",
        files={"file": (name, PDF, "application/pdf")},
        timeout=60,
    )


@pytest.mark.skipif(os.name == "nt", reason="sends SIGINT to a POSIX process group")
def test_ctrl_c_lets_the_request_in_flight_finish():
    import signal

    with _stub_server_process() as (process, base):
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            pending = pool.submit(_post_url, base, "slow.pdf")
            time.sleep(0.5)
            # What a terminal's Ctrl+C does: SIGINT to the whole group.
            os.killpg(process.pid, signal.SIGINT)
            response = pending.result(timeout=30)
        # uvicorn re-raises the signal once it has shut down: Ctrl+C's status.
        assert process.wait(timeout=30) == -signal.SIGINT
        output = process.stdout.read()
    assert "Inference worker" not in output or "exited with code" not in output
    assert response.status_code == 200
    assert response.json()["name"] == "slow.pdf"


@pytest.mark.skipif(os.name == "nt", reason="kills a POSIX process")
def test_the_server_stops_with_status_1_when_its_worker_dies():
    from bibr.serve.worker import WORKER_STOPPED_DETAIL

    with _stub_server_process() as (process, base):
        response = _post_url(base, "die.pdf")
        assert process.wait(timeout=30) == 1
    assert response.status_code == 500
    assert response.json() == {"detail": WORKER_STOPPED_DETAIL}


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
