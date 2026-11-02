"""``bibr batch`` / ``bibr chew`` end the process without interpreter teardown."""

import subprocess
import sys
import textwrap

import pytest

from bibr.local import cli


@pytest.mark.parametrize(
    ("code", "expected"),
    [(None, 0), (0, 0), (2, 2), (130, 130), ("boom", 1), (object(), 1)],
)
def test_exit_status_mirrors_system_exit(code, expected):
    assert cli._exit_status(code) == expected


@pytest.mark.parametrize(
    ("argv", "env", "expected"),
    [
        (["bibr", "batch", "papers/"], None, True),
        (["bibr", "chew", "paper.pdf"], None, True),
        (["bibr", "mcp"], None, False),
        (["bibr", "serve"], None, False),
        (["bibr", "doctor"], None, False),
        (["bibr"], None, False),
        (["bibr", "batch", "papers/"], "0", False),
        (["bibr", "batch", "papers/"], "off", False),
        (["bibr", "batch", "papers/"], "1", True),
    ],
)
def test_hard_exit_wanted(monkeypatch, argv, env, expected):
    monkeypatch.setattr(sys, "argv", argv)
    if env is None:
        monkeypatch.delenv("BIBR_FAST_EXIT", raising=False)
    else:
        monkeypatch.setenv("BIBR_FAST_EXIT", env)
    assert cli._hard_exit_wanted() is expected


def test_hook_leaves_an_unfinished_run_to_python(monkeypatch):
    exits = []
    monkeypatch.setattr(cli, "_exit_now", exits.append)
    cli._hard_exit_hook([None])
    assert exits == []


def test_hook_exits_with_the_recorded_status(monkeypatch):
    exits = []
    monkeypatch.setattr(cli, "_exit_now", exits.append)
    cli._hard_exit_hook([3])
    assert exits == [3]


def test_entrypoint_does_not_register_a_hook_for_other_commands(monkeypatch):
    registered = []
    ran = []
    monkeypatch.setattr(sys, "argv", ["bibr", "doctor"])
    monkeypatch.delenv("BIBR_FAST_EXIT", raising=False)
    monkeypatch.setattr(cli.atexit, "register", lambda *a, **k: registered.append(a))
    monkeypatch.setattr(cli, "main", lambda: ran.append(True))
    cli.entrypoint()
    assert ran == [True]
    assert registered == []


def test_entrypoint_records_system_exit_for_batch(monkeypatch):
    registered = []
    monkeypatch.setattr(sys, "argv", ["bibr", "batch", "papers/"])
    monkeypatch.delenv("BIBR_FAST_EXIT", raising=False)
    monkeypatch.setattr(cli.atexit, "register", lambda fn, *a: registered.append((fn, a)))

    def fake_main():
        raise SystemExit(4)

    monkeypatch.setattr(cli, "main", fake_main)
    with pytest.raises(SystemExit) as excinfo:
        cli.entrypoint()
    assert excinfo.value.code == 4
    (hook, (status,)) = registered[0]
    assert hook is cli._hard_exit_hook
    assert status == [4]


def test_entrypoint_records_a_normal_return_as_zero(monkeypatch):
    registered = []
    monkeypatch.setattr(sys, "argv", ["bibr", "chew", "paper.pdf"])
    monkeypatch.delenv("BIBR_FAST_EXIT", raising=False)
    monkeypatch.setattr(cli.atexit, "register", lambda fn, *a: registered.append((fn, a)))
    monkeypatch.setattr(cli, "main", lambda: None)
    cli.entrypoint()
    assert registered[0][1][0] == [0]


def test_an_unhandled_exception_keeps_the_unfinished_marker(monkeypatch):
    registered = []
    monkeypatch.setattr(sys, "argv", ["bibr", "batch", "papers/"])
    monkeypatch.delenv("BIBR_FAST_EXIT", raising=False)
    monkeypatch.setattr(cli.atexit, "register", lambda fn, *a: registered.append((fn, a)))

    def fake_main():
        raise RuntimeError("bug")

    monkeypatch.setattr(cli, "main", fake_main)
    with pytest.raises(RuntimeError):
        cli.entrypoint()
    assert registered[0][1][0] == [None]


_CHILD = textwrap.dedent(
    """
    import atexit, os, sys
    from bibr.local import cli

    marker = sys.argv[-1]

    class Teardown:
        # Finalised only when the interpreter frees its objects.
        # The builtin is bound now: it may be gone by the time this runs.
        def __del__(self, _open=open, _path=marker + ".teardown"):
            with _open(_path, "w") as f:
                f.write("ran")

    keep = Teardown()
    sys.modules["__main__"].keep = keep

    def fake_main():
        # Registered after entrypoint's hook, so it runs before it.
        atexit.register(lambda: open(marker + ".atexit", "w").write("ran"))
        print("result line")
        raise SystemExit(3)

    cli.main = fake_main
    cli.entrypoint()
    """
)


def _run_child(tmp_path, *, fast_exit):
    marker = str(tmp_path / "m")
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": ":".join(sys.path)}
    if not fast_exit:
        env["BIBR_FAST_EXIT"] = "0"
    result = subprocess.run(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-c", _CHILD, "batch", marker],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    return result, tmp_path


def test_process_skips_teardown_but_runs_other_exit_handlers(tmp_path):
    result, root = _run_child(tmp_path, fast_exit=True)
    assert result.returncode == 3, result.stderr
    assert "result line" in result.stdout
    assert (root / "m.atexit").exists()
    assert not (root / "m.teardown").exists()


def test_opt_out_restores_the_normal_teardown(tmp_path):
    result, root = _run_child(tmp_path, fast_exit=False)
    assert result.returncode == 3, result.stderr
    assert (root / "m.atexit").exists()
    assert (root / "m.teardown").exists()


class _FullStream:
    def flush(self):
        raise OSError(28, "No space left on device")


def _exit_status(monkeypatch, stdout, code):
    statuses = []

    def fake_exit(status):
        statuses.append(status)

    monkeypatch.setattr(cli.os, "_exit", fake_exit)
    monkeypatch.setattr(cli.sys, "stdout", stdout)
    cli._exit_now(code)
    return statuses[-1]


def test_output_lost_at_exit_turns_a_success_into_status_120(monkeypatch):
    assert _exit_status(monkeypatch, _FullStream(), 0) == 120


def test_output_lost_at_exit_keeps_a_failure_status(monkeypatch):
    assert _exit_status(monkeypatch, _FullStream(), 3) == 3
