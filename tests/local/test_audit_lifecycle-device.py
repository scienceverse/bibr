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

# Skip before the module body runs: Windows has no SIGHUP, so the constants
# below would fail at collection, not just in the tests.
if os.name != "posix":
    pytest.skip("POSIX signals", allow_module_level=True)

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

    def fake_mcp(args):
        record("mcp")()
        return 0

    async def fake_process(args):
        record("chew")()

    def fake_batch(args):
        record("batch")()
        return 0

    _fake_module(monkeypatch, "bibr.mcp_server", run_mcp=fake_mcp)
    monkeypatch.setattr(cli, "_run_process", fake_process)
    import bibr.local.cli.batch as cli_batch

    monkeypatch.setattr(cli_batch, "_run_batch", fake_batch)

    commands = (["serve"], ["demo"], ["setup"], ["mcp"], ["chew", "paper.pdf"], ["batch", "papers"])
    for argv in commands:
        monkeypatch.setattr("sys.argv", ["bibr", *argv])
        try:
            cli.main()
        except SystemExit as exc:
            assert exc.code in (0, None)
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

    # serve's LitServe/uvicorn own their signal handling.
    assert seen.pop("serve") is signal.SIG_DFL
    assert set(seen) == {"demo", "setup", "mcp", "chew", "batch"}
    for name, handler in seen.items():
        assert callable(handler), name


def test_the_terminating_signal_is_known_until_the_command_ends():
    """``bibr mcp`` reads it to exit once its cleanup has run."""
    seen = []
    with pytest.raises(SystemExit):
        with cli._interrupt_on_termination():
            try:
                _send(signal.SIGHUP)
            finally:
                seen.append(cli._terminating_signal())
    assert seen == [signal.SIGHUP]
    assert cli._terminating_signal() is None

    with pytest.raises(KeyboardInterrupt):
        with cli._interrupt_on_termination():
            try:
                raise KeyboardInterrupt
            finally:
                seen.append(cli._terminating_signal())
    assert seen[-1] is None  # a plain Ctrl-C


class _Exported:
    ok = True

    def __init__(self, path):
        self.data = {"info": {"title": path.stem}, "bib": [], "text": []}


@pytest.mark.parametrize("stage", ["discover_inputs", "write_batch_tables"])
@pytest.mark.parametrize("signum", _SIGNALS, ids=lambda signum: signum.name)
def test_batch_exits_130_when_interrupted_outside_the_executors(
    monkeypatch, tmp_path, signum, stage
):
    """The executors map an interrupt mid-run to 130; one during input
    discovery or the closing table rebuild exited 128 + the signal (or died
    with a Ctrl-C traceback), against the documented 130."""
    import bibr.batch.runner as runner

    real = getattr(runner, stage)

    def interrupted(*args, **kwargs):
        _send(signum)
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, stage, interrupted)
    monkeypatch.setattr(
        runner, "open_chew_many", lambda local: lambda paths, size: [_Exported(p) for p in paths]
    )
    monkeypatch.setattr(runner, "local_build_sha", lambda: "sha")
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "a.html").write_text("<html><body><h1>A</h1><p>Text.</p></body></html>")
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["bibr", "batch", str(papers), "--out", str(out), "--no-llm"])

    try:
        cli.main()
    except SystemExit as exc:
        code = exc.code
    except KeyboardInterrupt:
        code = "KeyboardInterrupt"
    assert code == 130
    if stage == "write_batch_tables":
        assert (out / "a.json").is_file()  # the run itself had finished
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
