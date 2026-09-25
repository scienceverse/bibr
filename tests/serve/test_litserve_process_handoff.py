import hashlib
import multiprocessing
import os
import time
import uuid

import litserve as ls
import pytest
from fastapi.testclient import TestClient


class _DescriptorHandoffAPI(ls.LitAPI):
    """Pickleable API used to exercise LitServe's spawned worker process."""

    def __init__(self, root, parent_pid):
        super().__init__(api_path="/handoff")
        self.root = root
        self.parent_pid = parent_pid

    def setup(self, device):
        return None

    def decode_request(self, request: dict):
        from bibr.serve.ingress import consume_upload_descriptor

        if os.getpid() == self.parent_pid:
            raise RuntimeError("decode_request ran in API process")
        content, digest = consume_upload_descriptor(self.root, request, max_size=1024)
        return {"size": len(content), "digest": digest}

    def predict(self, inputs):
        return inputs


def _wait_for_worker_ready(server, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        statuses = tuple(server.workers_setup_status.values())
        if statuses and all(status == "ready" for status in statuses):
            return
        exited = [
            worker.exitcode for worker in server.inference_workers if worker.exitcode is not None
        ]
        if exited:
            pytest.fail(f"LitServe inference worker exited during startup: {exited}")
        time.sleep(0.05)
    pytest.fail(
        f"LitServe inference worker did not become ready: {dict(server.workers_setup_status)}"
    )


def _cleanup_handoff_resources(server, manager, stored_path) -> None:
    cleanup_error = None
    active_manager = manager
    if active_manager is None:
        transport_config = getattr(server, "transport_config", None)
        active_manager = getattr(transport_config, "manager", None)
    if active_manager is not None:
        try:
            if getattr(server, "_transport", None) is None:
                active_manager.shutdown()
            else:
                server._perform_graceful_shutdown(active_manager, {}, "test")
        except BaseException as exc:
            cleanup_error = exc
    try:
        stored_path.unlink(missing_ok=True)
    except BaseException as exc:
        if cleanup_error is None:
            cleanup_error = exc
    if cleanup_error is not None:
        raise cleanup_error


def _wait_for_condition(predicate, *, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_descriptor_handoff_rejects_inline_decode_in_api_process(tmp_path):
    """Catches LitServe bypassing its worker queue and decoding in the API process."""
    content = b"%PDF-1.7\ninline decode regression\n%%EOF"
    upload_id = uuid.uuid4().hex
    stored_path = tmp_path / upload_id
    stored_path.write_bytes(content)
    descriptor = {
        "upload_id": upload_id,
        "filename": "paper.pdf",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    api = _DescriptorHandoffAPI(str(tmp_path), parent_pid=os.getpid())

    with pytest.raises(RuntimeError, match="decode_request ran in API process"):
        api.decode_request(descriptor)

    assert stored_path.exists()


def test_descriptor_crosses_spawned_litserve_worker_and_file_is_consumed(tmp_path):
    """Catches queueing upload bytes/paths or reading outside LitServe's real worker."""
    content = b"%PDF-1.7\nreal LitServe process handoff\n%%EOF"
    upload_id = uuid.uuid4().hex
    stored_path = tmp_path / upload_id
    descriptor = {
        "upload_id": upload_id,
        "filename": "paper.pdf",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "refs": "off",
    }

    assert "content" not in descriptor
    assert "path" not in descriptor

    api = _DescriptorHandoffAPI(str(tmp_path), parent_pid=os.getpid())
    server = ls.LitServer(api, accelerator="cpu", devices=1)
    server.inference_workers = []
    manager = None
    body_failed = False
    try:
        stored_path.write_bytes(content)
        manager = server._init_manager(1)
        server.inference_workers.extend(server.launch_inference_worker(api))
        assert len(server.inference_workers) == 1
        assert server.inference_workers[0].pid not in (None, os.getpid())
        _wait_for_worker_ready(server)
        server.app.response_queue_id = 0

        with TestClient(server.app) as client:
            response = client.post("/handoff", json=descriptor)

        assert response.status_code == 200
        assert response.json() == {
            "size": len(content),
            "digest": hashlib.sha256(content).hexdigest(),
        }
        assert not stored_path.exists()
    except BaseException:
        body_failed = True
        raise
    finally:
        try:
            _cleanup_handoff_resources(server, manager, stored_path)
        except BaseException:
            if not body_failed:
                raise


def test_default_worker_death_fail_stops_without_replacement(tmp_path, monkeypatch):
    """Catches replacing a dead worker while its API request remains stranded."""
    from bibr.config import GlobalSettings

    monkeypatch.delenv("PIPELINE_RESTART_WORKERS", raising=False)
    restart_workers = GlobalSettings(_env_file=None).pipeline.restart_workers
    api = _DescriptorHandoffAPI(str(tmp_path), parent_pid=os.getpid())
    server = ls.LitServer(
        api,
        accelerator="cpu",
        devices=1,
        restart_workers=restart_workers,
    )
    server.inference_workers = []
    manager = None
    api_process = multiprocessing.get_context("spawn").Process(
        target=time.sleep,
        args=(60,),
        name="litserve-death-path-api",
    )
    all_workers = []
    try:
        manager = server._init_manager(1)
        server.inference_workers.extend(server.launch_inference_worker(api))
        original_worker = server.inference_workers[0]
        all_workers.append(original_worker)
        _wait_for_worker_ready(server)

        api_process.start()
        assert api_process.is_alive()
        server.monitor_internal = 0.05
        server._start_worker_monitoring(manager, {0: api_process})

        original_worker.terminate()
        original_worker.join(timeout=5)
        assert not original_worker.is_alive()
        assert _wait_for_condition(
            lambda: (
                not api_process.is_alive() or server.inference_workers[0] is not original_worker
            ),
        )

        assert not api_process.is_alive()
        assert server.inference_workers[0] is original_worker
        assert not any(worker.is_alive() for worker in server.inference_workers)
    finally:
        try:
            server._shutdown_event.set()
        except (BrokenPipeError, EOFError):
            pass
        if server.inference_workers:
            all_workers.extend(
                worker for worker in server.inference_workers if worker not in all_workers
            )
        for worker in all_workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
        if api_process.is_alive():
            api_process.terminate()
        if api_process.pid is not None:
            api_process.join(timeout=5)
        if manager is not None:
            try:
                manager.shutdown()
            except (BrokenPipeError, EOFError):
                pass
