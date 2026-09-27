"""Cancelling a managed server's startup stops it within seconds (local-runtimes-13).

Every managed runtime spawns a real child that never becomes healthy (this
directory's conftest refuses every health probe). Cancelling the owning task,
or discarding an unclaimed OCR preload, must return well before the startup
timeout and leave no child behind.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time

import pytest

from bibr.config import GlobalSettings
from bibr.pipeline.resources import ResourceManager
from bibr.utils.async_tasks import await_owned

# Well above the bound below, so a startup that ignores the stop signal fails
# the timing assertion instead of hanging the suite for the 600 s default.
_STARTUP_TIMEOUT_S = 10
_CANCEL_BOUND_S = 3.0
_NEVER_HEALTHY = [sys.executable, "-c", "import time; time.sleep(600)"]
# Launchers the ``settings`` fixture below makes every managed runtime use.
_FAKE_LAUNCHERS = {"fake", "llama-server", "/fake/rapid-mlx", "bibr.local._vllm_mlx_server"}


@pytest.fixture
def spawned(monkeypatch):
    """Route every managed launch to a real child that never becomes healthy."""
    real_popen = subprocess.Popen
    children: list[subprocess.Popen] = []

    def fake_popen(cmd, *args, **kwargs):
        if not _FAKE_LAUNCHERS.intersection(cmd[:3]):
            return real_popen(cmd, *args, **kwargs)
        proc = real_popen(  # noqa: S603 — fixed test double, not a model server
            _NEVER_HEALTHY,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        children.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    yield children
    for proc in children:
        if proc.poll() is None:
            proc.kill()
            proc.wait(5)


@pytest.fixture
def settings(monkeypatch):
    """Settings whose launch preflights pass on any host."""
    import bibr.local.llama_cpp as llama_cpp
    import bibr.local.mlx_vlm_ocr as mlx_vlm_ocr
    import bibr.local.rapid_mlx as rapid_mlx
    import bibr.local.vllm_llm as vllm_llm
    import bibr.local.vllm_mlx_runtime as vllm_mlx_runtime
    import bibr.local.vllm_ocr as vllm_ocr
    import bibr.ocr.registry as registry

    monkeypatch.setattr(llama_cpp, "find_llama_server", lambda: ["llama-server"])
    monkeypatch.setattr(llama_cpp, "supported_flags", lambda prefix: frozenset())
    monkeypatch.setattr(llama_cpp, "probe_backend_kind", lambda *a, **k: "cpu")
    monkeypatch.setattr(rapid_mlx, "_resolve_executable", lambda *a, **k: "/fake/rapid-mlx")
    monkeypatch.setattr(vllm_mlx_runtime, "vllm_mlx_unavailable_reason", lambda: None)
    monkeypatch.setattr(registry, "paddle_vllm_unavailable_reason", lambda **k: None)
    for server in (vllm_llm.VllmLlmServer, vllm_ocr.VllmOcrServer, mlx_vlm_ocr.MlxVlmOcrServer):
        monkeypatch.setattr(server, "_resolve_launch_cmd", staticmethod(lambda model: ["fake"]))

    effective = GlobalSettings()
    effective.llm.vllm_startup_timeout = _STARTUP_TIMEOUT_S
    effective.llm.llama_cpp_startup_timeout = _STARTUP_TIMEOUT_S
    effective.ocr.llama_cpp_startup_timeout = _STARTUP_TIMEOUT_S
    effective.ocr.paddle_vllm_startup_timeout = _STARTUP_TIMEOUT_S
    effective.ocr.paddle_mlx_startup_timeout = _STARTUP_TIMEOUT_S
    effective.rapid_mlx.startup_timeout = _STARTUP_TIMEOUT_S
    effective.vllm_mlx.startup_timeout = _STARTUP_TIMEOUT_S
    return effective


async def _wait_for_child(children: list[subprocess.Popen]) -> subprocess.Popen:
    deadline = time.monotonic() + 10
    while not children:
        assert time.monotonic() < deadline, "the managed server was never launched"
        await asyncio.sleep(0.02)
    # Let the worker enter its health wait before the stop arrives.
    await asyncio.sleep(0.2)
    assert len(children) == 1
    return children[0]


async def _cancel_and_time(task: asyncio.Task) -> tuple[BaseException | None, float]:
    started = time.monotonic()
    task.cancel()
    (result,) = await asyncio.gather(task, return_exceptions=True)
    return result, time.monotonic() - started


@pytest.mark.parametrize("backend", ["vllm", "llama-cpp", "rapid-mlx", "vllm-mlx"])
async def test_cancelled_llm_startup_stops_and_kills_server(backend, settings, spawned):
    rm = ResourceManager(settings=settings)
    task = asyncio.create_task(rm.start_llm_server(backend))
    child = await _wait_for_child(spawned)

    result, elapsed = await _cancel_and_time(task)

    assert isinstance(result, asyncio.CancelledError)
    assert elapsed < _CANCEL_BOUND_S, f"{backend} startup ignored cancellation for {elapsed:.1f}s"
    assert child.poll() is not None, "the half-started server outlived the cancellation"
    assert rm._llm_server is None


@pytest.mark.parametrize(
    "backend", ["paddle-vllm", "paddle-mlx-vlm", "paddle-rapid-mlx", "glm-rapid-mlx", "glm-llama"]
)
@pytest.mark.parametrize("preload", [False, True])
async def test_cancelled_ocr_startup_stops_and_kills_server(backend, preload, settings, spawned):
    rm = ResourceManager(ocr_backend=backend, settings=settings)
    if preload:
        rm.start_ocr_preload()
    task = asyncio.create_task(rm.await_ocr())
    child = await _wait_for_child(spawned)

    result, elapsed = await _cancel_and_time(task)

    assert isinstance(result, asyncio.CancelledError)
    assert elapsed < _CANCEL_BOUND_S, f"{backend} startup ignored cancellation for {elapsed:.1f}s"
    assert child.poll() is not None, "the half-started server outlived the cancellation"
    assert rm.ocr is None and rm._ocr_future is None


async def test_shutdown_stops_an_unclaimed_ocr_preload_startup(settings, spawned):
    """A preload nobody collected is discarded, so its startup is stopped too."""
    rm = ResourceManager(ocr_backend="glm-llama", settings=settings)
    rm.start_ocr_preload()
    child = await _wait_for_child(spawned)

    started = time.monotonic()
    await rm.shutdown_ocr()
    elapsed = time.monotonic() - started

    assert elapsed < _CANCEL_BOUND_S, f"shutdown waited {elapsed:.1f}s for the preload"
    assert child.poll() is not None
    assert rm._ocr_future is None and rm._ocr_preload_stop is None


async def test_uncancelled_startup_still_publishes_the_server(monkeypatch, settings, spawned):
    """The stop signal changes nothing when nobody cancels: startup completes."""
    import bibr.local.vllm_llm as vllm_llm

    monkeypatch.setattr(vllm_llm, "request_bytes", lambda url, **kw: (200, "OK", b"{}"))
    monkeypatch.setattr(vllm_llm, "_HEALTH_GRACE_S", 0)
    rm = ResourceManager(settings=settings)

    await asyncio.wait_for(rm.start_llm_server("vllm"), 10)

    assert rm._llm_server is not None
    assert rm._llm_server._stop_event is not None
    assert not rm._llm_server._stop_event.is_set()
    assert spawned[0].poll() is None, "a healthy server must keep running"
    assert settings.llm.base_url == rm._llm_server.base_url + "/v1"
    await rm.close_llm_server()
    assert spawned[0].poll() is not None


async def test_await_owned_sets_stop_event_on_cancel_and_still_settles():
    loop = asyncio.get_running_loop()
    stop = threading.Event()
    entered = threading.Event()

    def work() -> str:
        entered.set()
        assert stop.wait(5), "cancellation never set the stop event"
        return "settled"

    task = asyncio.create_task(await_owned(loop.run_in_executor(None, work), stop_event=stop))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    (result,) = await asyncio.gather(task, return_exceptions=True)

    assert isinstance(result, asyncio.CancelledError)
    assert stop.is_set()


async def test_await_owned_leaves_stop_event_unset_without_cancel():
    stop = threading.Event()

    async def work() -> str:
        return "done"

    assert await await_owned(work(), stop_event=stop) == "done"
    assert not stop.is_set()
