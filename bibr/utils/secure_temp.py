"""Secure temporary-file helpers for local managed subprocesses."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import BinaryIO


def open_subprocess_log(label: str, port: int) -> tuple[Path, BinaryIO]:
    """Create a private, unpredictable log file and return its path + handle."""
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - owner closes with subprocess
        mode="w+b",
        prefix=f"bibr-{label}-{port}-",
        suffix=".log",
        delete=False,
    )
    return Path(handle.name), handle
