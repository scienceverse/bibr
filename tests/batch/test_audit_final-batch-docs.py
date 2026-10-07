"""Final review, batch: a Windows out dir that cannot be locked runs unguarded
instead of reporting a phantom concurrent run."""

from __future__ import annotations

import errno
import logging
import sys
import types

import pytest

from bibr.batch.runner import LOCK_FILENAME, OutDirBusy, out_dir_lock


def _fake_msvcrt(monkeypatch, error: OSError | None) -> list[int]:
    """Pretend to be Windows, with ``msvcrt.locking`` raising *error*; returns the modes used."""
    calls: list[int] = []

    def locking(fd: int, mode: int, nbytes: int) -> None:
        calls.append(mode)
        if error is not None and mode == fake.LK_NBLCK:
            raise error

    fake = types.ModuleType("msvcrt")
    fake.LK_UNLCK, fake.LK_NBLCK = 0, 2
    fake.locking = locking
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(sys, "platform", "win32")
    return calls


# --- the out-dir lock on Windows ---------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.EINVAL, "Invalid argument"),  # a share without byte-range locks
        OSError(errno.ENOLCK, "No locks available"),
    ],
)
def test_a_windows_out_dir_without_locks_runs_unguarded(tmp_path, monkeypatch, caplog, error):
    calls = _fake_msvcrt(monkeypatch, error)
    out = tmp_path / "out"

    with caplog.at_level(logging.WARNING, logger="bibr.batch.runner"), out_dir_lock(out):
        ran = True

    assert ran
    assert f"cannot lock {out / LOCK_FILENAME}" in caplog.text
    assert "a concurrent run is not detected" in caplog.text
    assert calls == [2]  # nothing was locked, so nothing is unlocked


@pytest.mark.parametrize(
    "error",
    [
        PermissionError(errno.EACCES, "Permission denied"),  # ERROR_LOCK_VIOLATION
        OSError(errno.EDEADLK, "Resource deadlock avoided"),
    ],
)
def test_a_windows_out_dir_held_by_another_run_is_busy(tmp_path, monkeypatch, error):
    _fake_msvcrt(monkeypatch, error)
    out = tmp_path / "out"

    with pytest.raises(OutDirBusy, match="another bibr batch run is using"), out_dir_lock(out):
        pytest.fail("the body must not run")


def test_a_windows_lock_is_released_at_the_end(tmp_path, monkeypatch):
    calls = _fake_msvcrt(monkeypatch, None)

    with out_dir_lock(tmp_path / "out"):
        assert calls == [2]

    assert calls == [2, 0]
