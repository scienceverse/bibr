"""Secure temporary-file helpers for local managed subprocesses."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import BinaryIO

# Per-process logs opened with ``shared=True``, by (label, port). The handle
# from creation stays open and each caller gets a duplicate of it: the log is
# never reopened by path, where a temp cleaner may have removed it and, in a
# shared /tmp, another user may since have put a link or FIFO in its place.
_SHARED_LOGS: dict[tuple[str, int], tuple[Path, BinaryIO]] = {}
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
        entry = _SHARED_LOGS.get((label, port))
        if entry is None or not _still_linked(entry[1]):
            if entry is not None:
                entry[1].close()
            entry = _SHARED_LOGS[(label, port)] = _new_log(label, port)
        path, original = entry
        # "ab" positions the duplicate at the end: restarts append.
        return path, os.fdopen(os.dup(original.fileno()), "ab")


def _new_log(label: str, port: int) -> tuple[Path, BinaryIO]:
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - owner closes with subprocess
        mode="w+b",
        prefix=f"bibr-{label}-{port}-",
        suffix=".log",
        delete=False,
    )
    return Path(handle.name), handle


def _still_linked(handle: BinaryIO) -> bool:
    """Whether the log still has a name; a removed one would swallow new output."""
    try:
        return os.fstat(handle.fileno()).st_nlink > 0
    except (OSError, ValueError):
        return False
