"""Secure temporary-file helpers for local managed subprocesses."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import BinaryIO

# Per-process logs opened with ``shared=True``, by (label, port).
_SHARED_LOGS: dict[tuple[str, int], Path] = {}
_SHARED_LOGS_LOCK = threading.Lock()


def open_subprocess_log(label: str, port: int, *, shared: bool = False) -> tuple[Path, BinaryIO]:
    """Create a private, unpredictable log file and return its path + handle.

    *shared* appends to the log this process already opened for the same
    label and port, so a server restarted many times in one run (Rapid-MLX
    recycles its OCR server every few dozen regions) keeps one log instead of
    leaving one behind per restart.
    """
    if not shared:
        return _new_log(label, port)
    with _SHARED_LOGS_LOCK:
        path = _SHARED_LOGS.get((label, port))
        handle = _reopen_own_log(path) if path is not None else None
        if path is None or handle is None:
            path, handle = _new_log(label, port)
            _SHARED_LOGS[(label, port)] = path
        return path, handle


def _new_log(label: str, port: int) -> tuple[Path, BinaryIO]:
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - owner closes with subprocess
        mode="w+b",
        prefix=f"bibr-{label}-{port}-",
        suffix=".log",
        delete=False,
    )
    return Path(handle.name), handle


def _reopen_own_log(path: Path) -> BinaryIO | None:
    """Append to a log this process created, or ``None`` if it is gone or not ours.

    Never creates the file or follows a link: a temp cleaner may have removed
    it, and in a shared /tmp another user may since have taken its name.
    """
    flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    if hasattr(os, "getuid") and os.fstat(fd).st_uid != os.getuid():
        os.close(fd)
        return None
    return os.fdopen(fd, "ab")
