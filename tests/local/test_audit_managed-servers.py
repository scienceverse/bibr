"""Managed local model servers: ownership, binding, isolation and cleanup.

Regression tests for the managed-servers audit findings: Rapid-MLX restarts
that a cancellation or shutdown() could orphan, recycling a server bibr does
not own, ``python -m`` children importing from the working directory, the
unbound vllm-mlx host, keys sent to (or missing from) loopback servers,
process groups left after a startup failure, leaked/unbounded logs, and an
`lms load` timeout that was never unloaded.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import select
import signal
import subprocess
import sys
import threading
from unittest.mock import MagicMock

import pytest

from bibr.config import GlobalSettings
from bibr.exceptions import UpstreamServiceError

_POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
# The launch tests below stub the module-wide Popen; the planted-module test
# still needs a real child.
_REAL_POPEN = subprocess.Popen


# ---------------------------------------------------------------------------
# 1. Rapid-MLX restarts cannot orphan a server
# ---------------------------------------------------------------------------


class _FakeServer:
    base_url = "http://127.0.0.1:1"

    def __init__(self, gen: int) -> None:
        self.gen = gen
        self.shut = False

    @property
    def loaded(self) -> bool:
        return not self.shut

    def shutdown(self) -> None:
        self.shut = True


class _FakeHttp:
    async def recognize(self, image, prompt):
        return "text"

    async def shutdown(self):
        return None


def _blocking_restart_client(monkeypatch):
    """A Rapid-MLX OCR client whose restarts block until the test releases them."""
    from bibr.local import rapid_mlx as mod

    started: list[_FakeServer] = []
    stop_events: list[threading.Event | None] = []
    restarting = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def fake_start_generation(self, stop_event=None):
        server = _FakeServer(len(started) + 1)
        if started:  # a restart: the model is still loading
            stop_events.append(stop_event)
            restarting.set()
            release.wait(5)
        started.append(server)
        if len(started) > 1:
            finished.set()
        return server, _FakeHttp()

    monkeypatch.setattr(mod._ManagedRapidMlxOcrClient, "_start_generation", fake_start_generation)
    settings = GlobalSettings()
    settings.ocr.rapid_mlx_recycle_after = 1
    client = mod.RapidMlxOcrClient(settings=settings)
    return client, started, stop_events, restarting, release, finished


async def _wait_for(event: threading.Event) -> None:
    assert await asyncio.get_running_loop().run_in_executor(None, event.wait, 5)


async def test_cancelled_recycle_stops_the_server_it_started(monkeypatch):
    client, started, _stops, restarting, release, finished = _blocking_restart_client(monkeypatch)
    task = asyncio.create_task(client.recognize(None, "OCR:"))
    await _wait_for(restarting)  # the post-request recycle is starting gen 2

    task.cancel()
    release.set()
    await asyncio.gather(task, return_exceptions=True)
    await client.shutdown()
    await _wait_for(finished)
    await asyncio.sleep(0.05)  # let any executor-side shutdown land

    assert [server.shut for server in started] == [True, True]
    assert client._server is None


async def test_cancelled_dead_generation_restart_stops_the_server_it_started(monkeypatch):
    client, started, _stops, restarting, release, finished = _blocking_restart_client(monkeypatch)
    started[0].shut = True  # the first server died: the next request restarts it
    task = asyncio.create_task(client.recognize(None, "OCR:"))
    await _wait_for(restarting)

    task.cancel()
    release.set()
    await asyncio.gather(task, return_exceptions=True)
    await client.shutdown()
    await _wait_for(finished)
    await asyncio.sleep(0.05)

    assert started[1].shut is True


async def test_shutdown_during_recycle_stops_the_restarting_server(monkeypatch):
    client, started, stop_events, restarting, release, _finished = _blocking_restart_client(
        monkeypatch
    )
    task = asyncio.create_task(client.recognize(None, "OCR:"))
    await _wait_for(restarting)

    await client.shutdown()  # finds no installed generation: the recycle detached it
    # shutdown() cut the in-flight startup wait short.
    assert stop_events[-1] is not None and stop_events[-1].is_set()
    release.set()

    assert await task == "text"  # the region was transcribed before the recycle
    assert started[1].shut is True
    assert client._server is None and client._http_client is None


async def test_closed_client_does_not_start_a_new_server(monkeypatch):
    client, started, _stops, _restarting, release, _finished = _blocking_restart_client(monkeypatch)
    release.set()
    await client.shutdown()

    with pytest.raises(UpstreamServiceError, match="shut down"):
        await client.recognize(None, "OCR:")
    assert len(started) == 1


# ---------------------------------------------------------------------------
# 2. A recycling client refuses a Rapid-MLX server it does not own
# ---------------------------------------------------------------------------


def _reused_port(monkeypatch):
    from bibr.local import rapid_mlx as mod

    requests = MagicMock(side_effect=AssertionError("no request to a refused listener"))
    monkeypatch.setattr(mod, "guard_managed_server_port", lambda *_a, **_k: True)
    monkeypatch.setattr(mod.subprocess, "Popen", MagicMock(side_effect=AssertionError("spawn")))
    monkeypatch.setattr(mod, "request_bytes", requests)
    return mod, requests


@pytest.mark.parametrize(
    ("client_name", "port_option"),
    [("RapidMlxOcrClient", "rapid_mlx_port"), ("PaddleRapidMlxOcrClient", "paddle_mlx_port")],
)
def test_recycling_client_refuses_a_reused_server(monkeypatch, client_name, port_option):
    mod, requests = _reused_port(monkeypatch)
    settings = GlobalSettings()
    setattr(settings.ocr, port_option, 8799)

    with pytest.raises(UpstreamServiceError) as excinfo:
        getattr(mod, client_name)(settings=settings)

    message = str(excinfo.value)
    assert "8799" in message
    assert "did not start" in message
    assert "OCR_RAPID_MLX_RECYCLE_AFTER=0" in message
    requests.assert_not_called()  # not even the Paddle smoke image


async def test_reused_server_is_still_accepted_with_recycling_off(monkeypatch):
    mod, _requests = _reused_port(monkeypatch)
    settings = GlobalSettings()
    settings.ocr.rapid_mlx_recycle_after = 0

    client = mod.RapidMlxOcrClient(settings=settings)
    try:
        assert client._server._reused is True
        assert client.loaded is True
    finally:
        await client.shutdown()


# ---------------------------------------------------------------------------
# 3/4. `python -m` children never import from the cwd; vllm-mlx binds loopback
# ---------------------------------------------------------------------------


def _vllm_mlx_cmd(monkeypatch) -> list[str]:
    from bibr.local import ocr as mod

    captured: dict[str, list[str]] = {}

    def fake_popen(cmd, **_kwargs):
        captured["cmd"] = cmd
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 1
        return proc

    monkeypatch.setattr(
        "bibr.local.vllm_mlx_runtime.vllm_mlx_unavailable_reason", lambda: None, raising=False
    )
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    with pytest.raises(RuntimeError, match="exited during startup"):
        mod.VllmMlxServer(model="org/model", port=8766, multimodal=False)
    return captured["cmd"]


def test_vllm_mlx_launch_is_safe_path_and_loopback_only(monkeypatch):
    cmd = _vllm_mlx_cmd(monkeypatch)

    assert cmd[:4] == [sys.executable, "-P", "-m", "bibr.local._vllm_mlx_server"]
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"


@pytest.mark.skipif(
    importlib.util.find_spec("vllm_mlx") is not None,
    reason="a real vllm_mlx would start loading here",
)
def test_vllm_mlx_launch_ignores_a_module_planted_in_the_cwd(monkeypatch, tmp_path):
    cmd = _vllm_mlx_cmd(monkeypatch)
    marker = tmp_path / "planted-ran"
    (tmp_path / "vllm_mlx.py").write_text(f"open({str(marker)!r}, 'w').close()\n")

    with (
        _REAL_POPEN(  # noqa: S603 - the launcher argv under test
            [*cmd[:4], "--help"],
            cwd=tmp_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        ) as child
    ):
        _out, stderr = child.communicate(timeout=60)

    assert b"vllm_mlx" in stderr  # the shim ran and looked for the runtime
    assert not marker.exists()


@pytest.mark.parametrize("module", ["bibr.local.vllm_llm", "bibr.local.vllm_ocr"])
def test_vllm_module_launch_is_safe_path(monkeypatch, module):
    mod = importlib.import_module(module)
    server_cls = mod.VllmLlmServer if module.endswith("llm") else mod.VllmOcrServer
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name, *a, **k: object())

    cmd = server_cls._resolve_launch_cmd("org/model")

    assert cmd[:4] == [sys.executable, "-P", "-m", "vllm.entrypoints.openai.api_server"]


def test_mlx_vlm_module_launch_is_safe_path(monkeypatch):
    from bibr.local import mlx_vlm_ocr as mod

    monkeypatch.setattr(mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda _name: object())

    cmd = mod.MlxVlmOcrServer._resolve_launch_cmd("org/model")

    assert cmd[:4] == [sys.executable, "-P", "-m", "mlx_vlm.server"]


# ---------------------------------------------------------------------------
# 5. Keys: per-launch keys for vLLM/llama.cpp, OCR_API_KEY never to loopback
# ---------------------------------------------------------------------------


def _user_key_settings() -> GlobalSettings:
    settings = GlobalSettings()
    settings.ocr.api_key = "user-ocr-secret"
    return settings


def _auth_header(http_client) -> str | None:
    return http_client._client.headers.get("authorization")


async def test_rapid_mlx_clients_never_send_ocr_api_key(monkeypatch):
    from bibr.local import rapid_mlx as mod

    server = MagicMock(base_url="http://localhost:8772", loaded=True)
    monkeypatch.setattr(mod, "RapidMlxServer", MagicMock(return_value=server))
    for client_cls in (mod.RapidMlxOcrClient, mod.PaddleRapidMlxOcrClient):
        client = client_cls(settings=_user_key_settings())
        try:
            assert _auth_header(client._http_client) is None
        finally:
            await client.shutdown()


async def test_mlx_vlm_client_never_sends_ocr_api_key(monkeypatch):
    from bibr.local import mlx_vlm_ocr as mod

    server = MagicMock(base_url="http://localhost:8775", loaded=True)
    monkeypatch.setattr(mod, "MlxVlmOcrServer", MagicMock(return_value=server))
    client = mod.PaddleMlxVlmOcrClient(settings=_user_key_settings())
    try:
        assert _auth_header(client._http_client) is None
    finally:
        await client.shutdown()


def test_mlx_vlm_smoke_never_sends_ocr_api_key(monkeypatch):
    from bibr.local import mlx_vlm_ocr as mod

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def recognize(self, _image, _prompt):
            return "OCR OK"

        async def shutdown(self):
            return None

    monkeypatch.setattr(mod, "PaddleHttpOcrClient", FakeClient)
    server = mod.MlxVlmOcrServer.__new__(mod.MlxVlmOcrServer)
    server._settings = _user_key_settings()
    server._model = "org/model"
    server._port = 8775

    server._run_smoke()

    assert captured["api_key"] == ""


@pytest.mark.parametrize("module", ["vllm_ocr", "llama_cpp"])
async def test_keyed_ocr_clients_send_the_server_key_not_ocr_api_key(monkeypatch, module):
    from bibr.local import ocr as ocr_mod
    from bibr.local import vllm_ocr

    server = MagicMock(base_url="http://127.0.0.1:8771", loaded=True, api_key="per-launch")
    if module == "vllm_ocr":
        monkeypatch.setattr(vllm_ocr, "VllmOcrServer", MagicMock(return_value=server))
        client = vllm_ocr.PaddleVllmOcrClient(settings=_user_key_settings())
    else:
        from bibr.local import llama_cpp

        monkeypatch.setattr(llama_cpp, "LlamaCppServer", MagicMock(return_value=server))
        client = ocr_mod.LlamaCppOcrClient(settings=_user_key_settings())
    try:
        assert _auth_header(client._http_client) == "Bearer per-launch"
    finally:
        await client.shutdown()


async def test_configured_ocr_endpoint_still_receives_ocr_api_key():
    from bibr.local.ocr import HttpOcrClient

    client = HttpOcrClient(base_url="https://ocr.example.org", settings=_user_key_settings())
    try:
        assert _auth_header(client) == "Bearer user-ocr-secret"
    finally:
        await client.shutdown()


def _launch_vllm_llm(monkeypatch):
    from bibr.local import vllm_llm

    popen = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name, *a, **k: object())
    monkeypatch.setattr(vllm_llm.subprocess, "Popen", popen)
    monkeypatch.setattr(vllm_llm.VllmLlmServer, "_wait_until_healthy", lambda self: None)
    server = vllm_llm.VllmLlmServer(model="org/m", port=9999, settings=GlobalSettings())
    return server, popen


def test_vllm_llm_gets_a_per_launch_key_through_the_environment(monkeypatch):
    first, popen = _launch_vllm_llm(monkeypatch)
    second, _popen = _launch_vllm_llm(monkeypatch)

    env = popen.call_args.kwargs["env"]
    assert len(first.api_key) >= 32
    assert first.api_key != second.api_key
    assert env["VLLM_API_KEY"] == first.api_key
    assert first.api_key not in popen.call_args.args[0]  # never visible in `ps`

    first.configure_llm_client()
    assert first._settings.llm.api_key == first.api_key


def test_vllm_ocr_gets_a_per_launch_key_and_polls_with_it(monkeypatch):
    from bibr.local import vllm_ocr

    real_wait_until_ready = vllm_ocr.VllmOcrServer._wait_until_ready
    popen = MagicMock(return_value=MagicMock())
    monkeypatch.setattr("bibr.ocr.registry.paddle_vllm_unavailable_reason", lambda **_k: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name, *a, **k: object())
    monkeypatch.setattr(vllm_ocr.subprocess, "Popen", popen)
    monkeypatch.setattr(vllm_ocr.VllmOcrServer, "_wait_until_ready", lambda self: None)
    server = vllm_ocr.VllmOcrServer(model="org/m", port=9123, settings=GlobalSettings())

    assert popen.call_args.kwargs["env"]["VLLM_API_KEY"] == server.api_key
    assert server.api_key not in popen.call_args.args[0]

    # The readiness poll of /v1/models carries the key vLLM now requires.
    process = MagicMock()
    process.poll.return_value = None
    server._process = process
    seen = []

    def models(url, *, headers=None, timeout):
        seen.append(headers)
        alias = server._served_model
        return 200, "OK", json.dumps({"data": [{"id": alias}]}).encode()

    monkeypatch.setattr(vllm_ocr, "request_bytes", models)
    real_wait_until_ready(server)
    assert seen == [{"Authorization": f"Bearer {server.api_key}"}]


def test_llama_cpp_gets_a_per_launch_key_through_the_environment(monkeypatch):
    from bibr.local import llama_cpp

    popen = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(llama_cpp, "find_llama_server", lambda: ["llama-server"])
    monkeypatch.setattr(llama_cpp, "supported_flags", lambda _prefix: frozenset())
    monkeypatch.setattr(llama_cpp, "_warn_cuda_steering_once", lambda _prefix: None)
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", popen)
    monkeypatch.setattr(llama_cpp.LlamaCppServer, "_wait_until_healthy", lambda self, _t: None)
    monkeypatch.setattr(llama_cpp.os, "killpg", MagicMock(), raising=False)

    server = llama_cpp.LlamaCppServer(model="org/m:Q4", port=8770, role="llm")
    try:
        assert len(server.api_key) >= 32
        assert popen.call_args.kwargs["env"]["LLAMA_API_KEY"] == server.api_key
        assert server.api_key not in popen.call_args.args[0]

        llm = llama_cpp.LlamaCppLlmServer.__new__(llama_cpp.LlamaCppLlmServer)
        llm._settings = GlobalSettings()
        llm._server = server
        llm.configure_llm_client()
        assert llm._settings.llm.api_key == server.api_key
    finally:
        server._process = None
        server.shutdown()


def test_reused_server_has_no_key_and_keeps_the_placeholder(monkeypatch):
    from bibr.local import vllm_llm

    monkeypatch.setattr(vllm_llm, "guard_managed_server_port", lambda *_a, **_k: True)
    server = vllm_llm.VllmLlmServer(model="org/m", port=9999, settings=GlobalSettings())

    assert server.api_key == ""
    server.configure_llm_client()
    assert server._settings.llm.api_key == "not-needed"


def _models_response(ids, status=200):
    def request(url, **_kwargs):
        return status, "OK", json.dumps({"data": [{"id": i} for i in ids]}).encode()

    return request


def test_guard_warns_that_a_reused_listener_is_not_ours(caplog):
    from bibr.local.http_runtime import guard_managed_server_port

    with caplog.at_level(logging.WARNING, logger="bibr.local.http_runtime"):
        reused = guard_managed_server_port(
            "llm",
            base_url="http://localhost:8769",
            model="org/m",
            server_label="vLLM",
            request_fn=_models_response(["org/m"]),
        )

    assert reused is True
    assert any("did not start it" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("status", [401, 403])
def test_guard_refuses_a_listener_that_requires_a_key(status):
    from bibr.local.http_runtime import guard_managed_server_port

    with pytest.raises(UpstreamServiceError) as excinfo:
        guard_managed_server_port(
            "llm",
            base_url="http://localhost:8769",
            model="org/m",
            server_label="vLLM",
            request_fn=_models_response([], status=status),
        )

    message = str(excinfo.value)
    assert "8769" in message
    assert "API key" in message


# ---------------------------------------------------------------------------
# 6. A server that exits during startup leaves no process group behind
# ---------------------------------------------------------------------------


def _exited(pid=424242):
    process = MagicMock(pid=pid, returncode=1)
    process.poll.return_value = 1
    return process


def _bare(cls, **attrs):
    server = cls.__new__(cls)
    server._settings = GlobalSettings()
    server._port = 9123
    server._model = "org/m"
    server._stderr_fh = None
    server._stderr_log = None
    server._reused = False
    for name, value in attrs.items():
        setattr(server, name, value)
    return server


def _startup_waiters():
    from bibr.local import llama_cpp, mlx_vlm_ocr, rapid_mlx, vllm_llm, vllm_ocr

    return {
        "vllm_llm": lambda p: _bare(vllm_llm.VllmLlmServer, _process=p)._wait_until_healthy(),
        "vllm_ocr": lambda p: _bare(
            vllm_ocr.VllmOcrServer, _process=p, _served_model="alias"
        )._wait_until_ready(),
        "mlx_vlm": lambda p: _bare(mlx_vlm_ocr.MlxVlmOcrServer, _process=p)._wait_until_ready(),
        "rapid_mlx": lambda p: _bare(
            rapid_mlx.RapidMlxServer, _process=p, _served_model_name="org/m"
        )._wait_until_healthy(),
        "llama_cpp": lambda p: _bare(
            llama_cpp.LlamaCppServer, _process=p, _role="ocr"
        )._wait_until_healthy(10),
    }


@_POSIX_ONLY
@pytest.mark.parametrize("runtime", ["vllm_llm", "vllm_ocr", "mlx_vlm", "rapid_mlx", "llama_cpp"])
def test_startup_exit_kills_the_process_group(monkeypatch, runtime):
    killpg = MagicMock()
    monkeypatch.setattr(os, "killpg", killpg)

    with pytest.raises(RuntimeError, match="exited during startup"):
        _startup_waiters()[runtime](_exited(pid=424242))

    killpg.assert_any_call(424242, signal.SIGKILL)


@_POSIX_ONLY
def test_startup_exit_kills_a_surviving_worker():
    """The real thing: a leader that dies during startup leaves a worker behind.

    The worker holds the only write end of a pipe, so EOF on the read end
    proves it is gone (a zombie would still answer ``kill(pid, 0)``).
    """
    from bibr.local import vllm_llm

    read_fd, write_fd = os.pipe()
    leader = subprocess.Popen(  # noqa: S603 - fixed test double
        [
            sys.executable,
            "-c",
            "import subprocess, sys\n"
            "w = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
            f"pass_fds=({write_fd},))\n"
            "print(w.pid, flush=True)\n"
            "sys.exit(3)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
        pass_fds=(write_fd,),
    )
    os.close(write_fd)
    worker_pid = int(leader.stdout.readline())
    leader.wait(timeout=30)
    worker_gone = False
    try:
        server = _bare(vllm_llm.VllmLlmServer, _process=leader)
        with pytest.raises(RuntimeError, match="exited during startup"):
            server._wait_until_healthy()

        readable, _, _ = select.select([read_fd], [], [], 10)
        worker_gone = bool(readable) and os.read(read_fd, 1) == b""
        assert worker_gone, "the startup worker outlived its failed server"
    finally:
        leader.stdout.close()
        os.close(read_fd)
        if not worker_gone:
            os.kill(worker_pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# 7. Log handles never leak; Rapid-MLX keeps one log across restarts
# ---------------------------------------------------------------------------


def _tracking_log(monkeypatch):
    from bibr.utils import secure_temp

    handles = []
    real = secure_temp.open_subprocess_log

    def tracking(*args, **kwargs):
        path, handle = real(*args, **kwargs)
        handles.append((path, handle))
        return path, handle

    monkeypatch.setattr(secure_temp, "open_subprocess_log", tracking)
    return handles


def test_vllm_mlx_closes_its_log_when_popen_fails(monkeypatch):
    from bibr.local import ocr as mod

    handles = _tracking_log(monkeypatch)
    monkeypatch.setattr(
        "bibr.local.vllm_mlx_runtime.vllm_mlx_unavailable_reason", lambda: None, raising=False
    )
    monkeypatch.setattr(mod.subprocess, "Popen", MagicMock(side_effect=OSError("no exec")))

    with pytest.raises(OSError, match="no exec"):
        mod.VllmMlxServer(model="org/model", port=8766)

    assert handles and all(handle.closed for _path, handle in handles)
    for path, _handle in handles:
        path.unlink(missing_ok=True)


def test_rapid_mlx_closes_its_log_when_popen_fails(monkeypatch):
    from bibr.local import rapid_mlx as mod

    handles = _tracking_log(monkeypatch)
    monkeypatch.setattr(mod.subprocess, "Popen", MagicMock(side_effect=OSError("no exec")))
    server = _bare(mod.RapidMlxServer, _process=None, _served_model_name="org/m", _multimodal=False)

    with pytest.raises(OSError, match="no exec"):
        server._spawn_and_wait(["rapid-mlx", "serve"])

    assert handles and all(handle.closed for _path, handle in handles)
    for path, _handle in handles:
        path.unlink(missing_ok=True)


async def test_rapid_mlx_recycles_keep_one_log(monkeypatch):
    from bibr.local import rapid_mlx as mod

    proc = MagicMock(pid=424242)
    proc.poll.return_value = None
    monkeypatch.setattr(mod.os, "killpg", MagicMock(), raising=False)
    monkeypatch.setattr(mod, "_resolve_executable", lambda *_a, **_k: "/opt/rapid-mlx")
    monkeypatch.setattr(mod.subprocess, "Popen", MagicMock(return_value=proc))
    monkeypatch.setattr(mod, "request_bytes", lambda *_a, **_k: (200, "OK", b"{}"))
    monkeypatch.setattr(mod.RapidMlxServer, "_warmup_model", lambda self: None)
    monkeypatch.setattr(mod, "HttpOcrClient", lambda **_k: _FakeHttp())
    settings = GlobalSettings()
    settings.ocr.rapid_mlx_recycle_after = 1
    settings.ocr.rapid_mlx_port = 18772  # a log key no other test shares

    client = mod.RapidMlxOcrClient(settings=settings)
    logs = {client._server._stderr_log}
    for _ in range(4):
        await client.recognize(None, "OCR:")
        logs.add(client._server._stderr_log)
    await client.shutdown()

    assert len(logs) == 1
    logs.pop().unlink(missing_ok=True)


def test_shared_log_appends_instead_of_truncating():
    from bibr.utils.secure_temp import open_subprocess_log

    first_path, first = open_subprocess_log("audit-append", 9001, shared=True)
    first.write(b"first\n")
    first.close()
    second_path, second = open_subprocess_log("audit-append", 9001, shared=True)
    second.write(b"second\n")
    second.close()
    try:
        assert second_path == first_path
        assert first_path.read_bytes() == b"first\nsecond\n"
    finally:
        first_path.unlink(missing_ok=True)


@_POSIX_ONLY
def test_shared_log_never_follows_a_replaced_path(tmp_path):
    from bibr.utils.secure_temp import open_subprocess_log

    path, handle = open_subprocess_log("audit-link", 9002, shared=True)
    handle.close()
    target = tmp_path / "elsewhere"
    target.write_bytes(b"")
    path.unlink()
    path.symlink_to(target)
    try:
        new_path, new_handle = open_subprocess_log("audit-link", 9002, shared=True)
        new_handle.write(b"log line")
        new_handle.close()

        assert new_path != path
        assert target.read_bytes() == b""
        new_path.unlink(missing_ok=True)
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 8. A timed-out `lms load` is still unloaded
# ---------------------------------------------------------------------------


def test_timed_out_llmster_load_is_unloaded(monkeypatch):
    from bibr.local import llmster

    calls: list[list[str]] = []

    def run(args, *, json_output=False):
        calls.append(args)
        if args[:2] == ["daemon", "status"]:
            return {"status": "running"}
        if args[:2] == ["server", "status"]:
            return {"running": True, "port": 1234}
        if args == ["ls", "--json"]:
            return [{"modelKey": "org/model"}]
        if args == ["ps", "--json"]:
            return []
        if args[0] == "load":
            raise UpstreamServiceError("llmster", "`lms load` timed out after 120s")
        if args[0] == "unload":
            return None
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(llmster, "find_lms", lambda: "/usr/local/bin/lms")

    with pytest.raises(UpstreamServiceError, match="timed out"):
        llmster.LlmsterLlmServer(model="org/model", identifier="bibr-model", runner=run)

    assert calls[-1] == ["unload", "bibr-model"]
