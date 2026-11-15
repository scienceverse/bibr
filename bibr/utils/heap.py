"""Give memory the C heap holds but no longer uses back to the operating system.

A local batch renders every page of every paper to a full-resolution image,
crops and encodes OCR regions, and parses the text, in worker threads. glibc
gives each thread its own malloc arena, and a buffer freed into an arena stays
mapped there: only the free space at the very top of a heap is ever returned
on its own. The page buffers of the next window rarely fit the holes the last
window left, so a long ``bibr batch`` grows from chunk to chunk while its live
objects do not (an 80-paper CPU batch ended holding 3.8 GiB of free heap next
to 1.6 GiB in use). ``malloc_trim(0)`` walks every arena and returns whole
free pages from anywhere in them, in 0.1-0.6 s for a few GiB.

``Settings.pipeline.heap_trim`` (``PIPELINE_HEAP_TRIM``) says when the batch
drivers trim:

* ``"chunk"`` (default): once after each chunk of a batch (``bibr batch``,
  ``bibr chew`` on several files, ``chew()``/``Chewer`` on a list);
* ``"window"``: also between the render/OCR windows inside a chunk, which
  keeps the footprint down inside large chunks (``--batch-size`` well above
  ``OCR_MAX_CONCURRENT_FILES``). The next window then faults the released
  pages back in while it renders, the step the OCR server waits on, and a
  24-paper chunk ran 20-30% slower;
* ``"off"``: never.

Single-paper calls (a warm ``Chewer`` given one path at a time included) and
``bibr serve`` requests never trim: neither their growth over many calls nor
the per-call cost of a trim has been measured. It is a no-op off glibc
(macOS, Windows, musl), where the function does not exist.
"""

from __future__ import annotations

import asyncio
import ctypes
import functools
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from bibr.config import GlobalSettings

logger = logging.getLogger(__name__)

HeapTrimMode = Literal["off", "chunk", "window"]

# Trims closer together than this are skipped: a window or a chunk always
# frees far more than one trim's cost, but two trim points that land together
# need only one.
MIN_INTERVAL_SECONDS = 1.0

_UNRESOLVED = object()
_malloc_trim: object = _UNRESOLVED
_last_trim = float("-inf")
_lock = threading.Lock()
# A thread of its own: the shared default executor also runs page rendering
# and back-half work, and a trim queued behind those would land late.
_executor: ThreadPoolExecutor | None = None


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


def _resolved_trim() -> object:
    """``malloc_trim``, or ``None`` where there is none. Call with ``_lock`` held."""
    global _malloc_trim
    if _malloc_trim is _UNRESOLVED:
        _malloc_trim = _resolve()
    return _malloc_trim


def release_free_heap(*, min_interval: float = MIN_INTERVAL_SECONDS) -> bool:
    """Return the C heap's free pages to the OS; ``True`` when a trim ran.

    Thread-safe. Skips (``False``) when there is no ``malloc_trim`` or the
    last trim was less than *min_interval* seconds ago. Never raises.
    """
    global _last_trim
    with _lock:
        trim = _resolved_trim()
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


def _trim_executor() -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bibr-heap-trim")
        return _executor


def _reset_after_fork() -> None:
    # A forked child inherits the executor object but not its thread, and
    # possibly a lock some other thread held at the fork.
    global _executor, _lock
    _executor = None
    _lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


async def arelease_free_heap(*, min_interval: float = MIN_INTERVAL_SECONDS) -> bool:
    """:func:`release_free_heap` off the event loop, on a thread of its own.

    Without ``malloc_trim`` (macOS, Windows, musl) it returns ``False`` and
    starts no thread. A trim that cannot be handed to its thread (no new
    thread can start, or the interpreter is shutting down) is logged and
    skipped (``False``). Never raises, cancellation aside.
    """
    with _lock:
        available = _resolved_trim() is not None
    if not available:
        return False
    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(
            _trim_executor(), functools.partial(release_free_heap, min_interval=min_interval)
        )
    except Exception:  # noqa: BLE001 — no trim thread (shutdown, thread limit) must not fail a batch
        logger.debug("heap trim could not be scheduled", exc_info=True)
        return False
    return await future


def heap_trim_mode(settings: GlobalSettings | None) -> HeapTrimMode:
    """The configured :data:`HeapTrimMode` (``"chunk"`` when unset)."""
    mode = getattr(getattr(settings, "pipeline", None), "heap_trim", "chunk")
    return cast(HeapTrimMode, mode) if mode in ("off", "chunk", "window") else "chunk"


async def trim_after_chunk(settings: GlobalSettings | None) -> bool:
    """Trim after a batch chunk, unless ``heap_trim`` is ``"off"``."""
    if heap_trim_mode(settings) == "off":
        return False
    return await arelease_free_heap()


async def trim_between_windows(settings: GlobalSettings | None) -> bool:
    """Trim between two render/OCR windows of a chunk, in ``"window"`` mode only."""
    if heap_trim_mode(settings) != "window":
        return False
    return await arelease_free_heap()


#: ``M_ARENA_MAX`` in glibc's ``malloc.h``.
_M_ARENA_MAX = -8
_arena_limit_applied = False


def limit_malloc_arenas(count: int) -> bool:
    """Cap glibc's malloc arenas for this process at *count*; ``True`` when the cap was set.

    glibc gives a thread that finds every arena locked a new one, up to eight
    per CPU core, and each arena keeps the memory freed into it. A batch's
    model and worker threads spread their buffers over dozens of arenas, so
    free memory piles up in each of them. Two arenas lowered the peak of a
    60-paper fast batch by 0.6 to 0.8 GB.

    Skipped (``False``) when *count* is 0, when ``MALLOC_ARENA_MAX`` is set in
    the environment (glibc read it at start-up), off Linux, and where the C
    library has no ``mallopt`` or refuses the parameter (musl). Arenas made
    before the call stay; the cap only stops new ones, so call it before the
    batch starts its threads. Never raises.
    """
    if count <= 0 or "MALLOC_ARENA_MAX" in os.environ or not sys.platform.startswith("linux"):
        return False
    try:
        mallopt = ctypes.CDLL(None).mallopt
        mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
        mallopt.restype = ctypes.c_int
        return bool(mallopt(_M_ARENA_MAX, int(count)))
    except Exception:  # noqa: BLE001 — memory housekeeping must never fail a batch
        logger.debug("mallopt(M_ARENA_MAX) failed", exc_info=True)
        return False


def _malloc_arenas_setting(settings: GlobalSettings | None) -> int:
    """``pipeline.malloc_arenas`` from *settings*, else from the environment and ``.env`` chain."""
    pipeline = getattr(settings, "pipeline", None)
    if pipeline is None:
        try:
            from bibr.config import PipelineOptions

            # pydantic-settings fills every field from the environment; mypy cannot see that.
            pipeline = PipelineOptions()  # type: ignore[call-arg]
        except Exception:  # noqa: BLE001 — a bad .env is reported by the CLI itself
            return 2
    count = getattr(pipeline, "malloc_arenas", 2)
    return count if isinstance(count, int) and not isinstance(count, bool) else 2


def limit_arenas_for_batch(settings: GlobalSettings | None = None) -> bool:
    """Apply ``pipeline.malloc_arenas`` (default 2) once per process, for a batch driver.

    The batch commands call it before they build a pipeline, so the cap is in
    place before model sessions and worker pools start their threads; the
    library's batch path calls it again (a no-op by then) for callers that
    did not come through the command line.
    """
    global _arena_limit_applied
    with _lock:
        if _arena_limit_applied:
            return False
        _arena_limit_applied = True
    count = _malloc_arenas_setting(settings)
    applied = limit_malloc_arenas(count)
    if applied:
        logger.debug("malloc arenas capped at %s for this batch process", count)
    return applied
