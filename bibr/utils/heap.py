"""Give memory the C heap holds but no longer uses back to the operating system.

A local batch renders every page of every paper to a full-resolution image,
crops and encodes OCR regions, and parses the text, in worker threads. glibc
gives each thread its own malloc arena, and a buffer freed into an arena stays
mapped there: only the free space at the very top of a heap is ever returned
on its own. The page buffers of the next window rarely fit the holes the last
window left, so a long ``bibr batch`` grows from chunk to chunk while its live
objects do not (an 80-paper CPU batch ended holding 3.8 GiB of free heap next
to 1.6 GiB in use). ``malloc_trim(0)`` walks every arena and returns whole
free pages from anywhere in them, in about 0.1 s for a few GiB.

The pipeline calls :func:`release_free_heap` where large buffers have just
been dropped: after each render/OCR window frees its page images and after
each chunk. It is a no-op off glibc (macOS, Windows, musl), where the function
does not exist.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import sys
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Trims closer together than this are skipped: a window or a chunk always
# frees far more than one trim's cost, but concurrent serve requests ending
# together need only one.
MIN_INTERVAL_SECONDS = 1.0

_UNRESOLVED = object()
_malloc_trim: object = _UNRESOLVED
_last_trim = float("-inf")
_lock = threading.Lock()


def _resolve() -> Callable[[int], int] | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        # The process's own symbols: glibc exports malloc_trim, musl does not.
        fn = ctypes.CDLL(None).malloc_trim
    except (OSError, AttributeError):
        return None
    fn.argtypes = [ctypes.c_size_t]
    fn.restype = ctypes.c_int
    return fn


def release_free_heap(*, min_interval: float = MIN_INTERVAL_SECONDS) -> bool:
    """Return the C heap's free pages to the OS; ``True`` when a trim ran.

    Thread-safe. Skips (``False``) when there is no ``malloc_trim`` or the
    last trim was less than *min_interval* seconds ago. Never raises.
    """
    global _malloc_trim, _last_trim
    with _lock:
        if _malloc_trim is _UNRESOLVED:
            _malloc_trim = _resolve()
        trim = _malloc_trim
        now = time.monotonic()
        if trim is None or now - _last_trim < min_interval:
            return False
        _last_trim = now
    try:
        trim(0)  # type: ignore[operator]
    except Exception:  # noqa: BLE001 — memory housekeeping must never fail a paper
        logger.debug("malloc_trim failed", exc_info=True)
        return False
    return True


async def arelease_free_heap(*, min_interval: float = MIN_INTERVAL_SECONDS) -> bool:
    """:func:`release_free_heap` off the event loop (a trim can take 0.1 s or more)."""
    return await asyncio.to_thread(release_free_heap, min_interval=min_interval)
