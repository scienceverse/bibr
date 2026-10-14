"""Memory readings for this process, from the operating system (stdlib only).

Linux reads ``/proc/self/status``: ``VmRSS`` (resident now), ``VmHWM`` (the
peak resident size, the "high water mark") and ``VmSwap``. The peak can be
reset to the current size by writing ``5`` to ``/proc/self/clear_refs``
(Linux 4.0+), which is what makes a per-stage peak measurable.

Elsewhere (macOS) only the lifetime peak is known, from
``resource.getrusage``'s ``ru_maxrss``; it can never be reset, and the current
size and swap are unknown. Every reading degrades to ``None`` rather than
raising, so callers never fail on a platform that hides a number.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MIB = 1024 * 1024
_KIB = 1024

_STATUS_PATH = Path("/proc/self/status")
_CLEAR_REFS_PATH = Path("/proc/self/clear_refs")
# clear_refs value 5: reset the peak resident set size to the current one. It
# touches nothing else (1-4 clear page reference / soft-dirty bits).
_RESET_PEAK = b"5"


@dataclass(frozen=True)
class ProcessMemory:
    """One reading, in bytes. ``None`` where the platform does not say."""

    rss: int | None
    peak: int | None
    swap: int | None
    # True when ``peak`` is Linux's resettable VmHWM; False for ru_maxrss,
    # the lifetime peak.
    peak_resettable: bool = False


def parse_status(text: str) -> ProcessMemory:
    """Parse ``/proc/<pid>/status`` text (``VmRSS:  1234 kB`` lines)."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if not sep or key not in ("VmRSS", "VmHWM", "VmSwap"):
            continue
        fields = rest.split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0]) * _KIB
    return ProcessMemory(
        rss=values.get("VmRSS"),
        peak=values.get("VmHWM"),
        swap=values.get("VmSwap"),
        peak_resettable="VmHWM" in values,
    )


def _rusage_peak() -> int | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and KiB on Linux and the BSDs.
    return int(peak) if sys.platform == "darwin" else int(peak) * _KIB


def read_process_memory() -> ProcessMemory:
    """Current RSS, peak RSS and swap of this process."""
    try:
        reading = parse_status(_STATUS_PATH.read_text(encoding="ascii", errors="replace"))
    except OSError:
        reading = None
    if reading is not None and reading.rss is not None:
        return reading
    return ProcessMemory(rss=None, peak=_rusage_peak(), swap=None)


def reset_peak_rss() -> bool:
    """Reset this process's peak RSS to its current RSS; True when it worked.

    Linux only. A kernel or sandbox that refuses the write returns False, and
    the peak then keeps covering the whole process lifetime.
    """
    try:
        with _CLEAR_REFS_PATH.open("wb", buffering=0) as handle:
            handle.write(_RESET_PEAK)
    except OSError:
        return False
    return True


def to_mib(value: int | None) -> float | None:
    """Bytes to MiB, one decimal; ``None`` stays ``None``."""
    return None if value is None else round(value / MIB, 1)


@contextmanager
def log_rss_delta(label: str, *, verbose: bool = False) -> Iterator[None]:
    """Log how much this process's RSS grew while the block ran (a model load).

    Logged at INFO when ``verbose`` (``PIPELINE_MEMORY_PROFILE`` on), else at
    DEBUG. The RSS is the whole process's, so work running beside the load
    in other threads counts toward it. A block that raises logs nothing.
    """
    before = read_process_memory().rss
    started = time.monotonic()
    yield
    after = read_process_memory().rss
    if before is None or after is None:
        return
    logger.log(
        logging.INFO if verbose else logging.DEBUG,
        "%s loaded: RSS %+.0f MiB (now %.0f MiB, %.1f s)",
        label,
        (after - before) / MIB,
        after / MIB,
        time.monotonic() - started,
    )
