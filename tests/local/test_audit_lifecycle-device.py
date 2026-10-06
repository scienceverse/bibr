"""SIGTERM and SIGHUP take Ctrl-C's path, so managed servers are shut down.

Managed inference servers run in their own session: a closed terminal never
sends them SIGHUP, and Python's default SIGTERM exits without running
``finally`` blocks, so ``kill``/``docker stop``/``timeout`` or a closed SSH
session left vLLM or llama.cpp holding VRAM and the port.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
import types

import pytest

from bibr.local import cli

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX signals")

_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


@pytest.fixture(autouse=True)
def _restore_dispositions():
    saved = {signum: signal.getsignal(signum) for signum in _SIGNALS}
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGHUP, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    for signum, handler in saved.items():
        signal.signal(signum, handler)


def _send(signum: int) -> None:
    # Never deliver a default-disposition SIGTERM/SIGHUP: it would kill pytest.
    assert signal.getsignal(signum) not in (signal.SIG_DFL, signal.SIG_IGN, None)
    os.kill(os.getpid(), signum)
    time.sleep(2)  # the interrupt lands here; reaching the end is a failure


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_signal_runs_cleanup_and_exits_128_plus_signum(signum):
    cleaned = []
    with pytest.raises(SystemExit) as exc_info:
        with cli._interrupt_on_termination():
            try:
                _send(signum)
            finally:
                cleaned.append(True)

    assert exc_info.value.code == 128 + signum
    assert cleaned == [True]
    assert signal.getsignal(signum) is signal.SIG_DFL  # restored on exit


def test_signal_interrupts_even_when_sigint_is_ignored():
    """A background job inherits SIGINT ignored; SIGTERM must still interrupt."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with pytest.raises(SystemExit) as exc_info:
        with cli._interrupt_on_termination():
            _send(signal.SIGTERM)
    assert exc_info.value.code == 128 + signal.SIGTERM


def test_inherited_ignored_sighup_is_kept():
    """``nohup`` ignores SIGHUP so the run survives a closed terminal; keep that."""
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with cli._interrupt_on_termination():
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
    assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN


def test_plain_ctrl_c_still_raises_keyboard_interrupt():
    with pytest.raises(KeyboardInterrupt):
        with cli._interrupt_on_termination():
            raise KeyboardInterrupt


def test_sigterm_cancels_asyncio_main_task_so_cleanup_awaits_run():
    """Under ``asyncio.run`` the signal cancels the main task the way Ctrl-C
    does, so awaiting teardown (``Pipeline.aclose``) completes."""
    cleaned = []

    async def work():
        asyncio.get_running_loop().call_later(0.05, _send_async, signal.SIGTERM)
        try:
            await asyncio.sleep(5)
        finally:
            await asyncio.sleep(0)
            cleaned.append(True)

    with pytest.raises(SystemExit) as exc_info:
        with cli._interrupt_on_termination():
            asyncio.run(work())

    assert exc_info.value.code == 128 + signal.SIGTERM
    assert cleaned == [True]


def _send_async(signum: int) -> None:
    assert signal.getsignal(signum) not in (signal.SIG_DFL, signal.SIG_IGN, None)
    os.kill(os.getpid(), signum)


def test_sigterm_is_the_batch_runners_graceful_first_ctrl_c():
    """The remote batch runner's SIGINT handler stops submitting on the first
    interrupt; SIGTERM reaches the same handler instead of killing the run."""
    from bibr.batch.runner import _run_remote

    class Executor:
        async def run(self, items, *, on_outcome, stop, deadline, on_ready):
            asyncio.get_running_loop().call_later(0.05, _send_async, signal.SIGTERM)
            await asyncio.wait_for(stop.wait(), 5)
            return "stopped"

    with cli._interrupt_on_termination():
        reason = _run_remote(
            Executor(), [], on_outcome=print, on_ready=print, deadline=None, stop=None
        )

    assert reason == "stopped"


def _fake_module(monkeypatch, name: str, **attrs) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_main_installs_the_handler_for_commands_that_own_servers(monkeypatch):
    seen: dict[str, object] = {}

    def record(name):
        return lambda *_a, **_k: seen.setdefault(name, signal.getsignal(signal.SIGTERM))

    _fake_module(monkeypatch, "bibr.serve.app", main=record("serve"))
    _fake_module(monkeypatch, "bibr.demo.server", main=record("demo"))
    _fake_module(monkeypatch, "bibr.setup_wizard", main=record("setup"))

    async def fake_process(args):
        record("chew")()

    def fake_batch(args):
        record("batch")()
        return 0

    monkeypatch.setattr(cli, "_run_process", fake_process)
    import bibr.local.cli.batch as cli_batch

    monkeypatch.setattr(cli_batch, "_run_batch", fake_batch)

    for argv in (["serve"], ["demo"], ["setup"], ["chew", "paper.pdf"], ["batch", "papers"]):
        monkeypatch.setattr("sys.argv", ["bibr", *argv])
        try:
            cli.main()
        except SystemExit as exc:
            assert exc.code in (0, None)
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

    # serve's LitServe/uvicorn own their signal handling.
    assert seen.pop("serve") is signal.SIG_DFL
    assert set(seen) == {"demo", "setup", "chew", "batch"}
    for name, handler in seen.items():
        assert callable(handler), name
