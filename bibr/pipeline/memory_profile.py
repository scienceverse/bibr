"""Opt-in per-stage memory telemetry (``PIPELINE_MEMORY_PROFILE``).

With the setting on, every pipeline stage records what the process's memory
did while it ran: the peak resident size, the resident size and swap when it
finished, how much the resident size changed, and how many files it held.
The records land on each file (``FileState.stage_memory``), in the export's
``extraction.timings.memory``, and from there in ``bibr batch``'s ledger and
report. With the setting off (the default) nothing is read or recorded.

Memory belongs to the process, not to a file, so each record says how far it
can be attributed:

- ``files``: a stage runs over its whole chunk (or OCR window) at once, so its
  peak belongs to all of those files together.
- ``overlapped``: another stage ran at the same time — back-half streaming,
  serve's concurrent requests, several pipelines in one process — and the
  peak covers both. ``PIPELINE_STREAM_BACKHALF=false`` makes a local run's
  stages sequential, so each peak is that stage's own.

How a peak is per stage: Linux's peak RSS (``VmHWM``) can be reset. Each stage
resets it when it starts, after folding the peak so far into every stage
still running, and reads it when it ends. A stage's peak is the highest of the
segments inside it, so nesting (the fused render/OCR stage around its layout
and OCR passes) and overlap never lose a peak. Where the peak cannot be reset
(macOS, or a sandbox that refuses the write), ``peak_reset`` is false and the
peak is the process's peak since it started.

``PIPELINE_MEMORY_PROFILE_TRACEMALLOC`` adds the peak of Python's own
allocations (``tracemalloc``), NumPy arrays included; ONNX Runtime arenas and
image buffers allocated in C are invisible to it. It slows allocation-heavy
code, so it is a separate switch.
"""

from __future__ import annotations

import contextvars
import itertools
import threading
import tracemalloc
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from bibr.utils.process_memory import read_process_memory, reset_peak_rss, to_mib

SCRATCH_KEY = "stage_memory"


@dataclass(frozen=True)
class StageMemory:
    """Process memory over one stage, in bytes (``None``: not reported here)."""

    peak_rss: int | None
    rss: int | None
    rss_delta: int | None
    swap: int | None
    files: int
    overlapped: bool
    peak_reset: bool
    python_peak: int | None = None

    def merge(self, newer: StageMemory) -> StageMemory:
        """Fold a later record of the same stage into this one."""
        return StageMemory(
            peak_rss=_max(self.peak_rss, newer.peak_rss),
            rss=newer.rss,
            rss_delta=(
                None
                if self.rss_delta is None or newer.rss_delta is None
                else self.rss_delta + newer.rss_delta
            ),
            swap=newer.swap,
            files=max(self.files, newer.files),
            overlapped=self.overlapped or newer.overlapped,
            peak_reset=self.peak_reset and newer.peak_reset,
            python_peak=_max(self.python_peak, newer.python_peak),
        )

    def to_export(self) -> dict[str, Any]:
        """The ``extraction.timings.memory.stages`` row, in MiB."""
        row: dict[str, Any] = {
            "peak_rss_mib": to_mib(self.peak_rss),
            "rss_mib": to_mib(self.rss),
            "rss_delta_mib": to_mib(self.rss_delta),
            "swap_mib": to_mib(self.swap),
            "files": self.files,
            "overlapped": self.overlapped,
            "peak_reset": self.peak_reset,
        }
        if self.python_peak is not None:
            row["python_peak_mib"] = to_mib(self.python_peak)
        return row


def _max(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


@dataclass
class _Measurement:
    ancestors: frozenset[int]
    start_rss: int | None
    peak: int | None = None
    python_peak: int | None = None
    overlapped: bool = False
    peak_reset: bool = False


# Process-wide: VmHWM and tracemalloc's peak are per process, so every open
# measurement in the process shares one reset. The lock keeps a fold and its
# reset together when stages start on different threads' event loops.
_lock = threading.Lock()
_active: dict[int, _Measurement] = {}
_ids = itertools.count(1)
# The measurements open in the current task: nested stages (and tasks spawned
# inside a stage, which copy the context) are not overlap, they are part of it.
_open: contextvars.ContextVar[frozenset[int]] = contextvars.ContextVar(
    "bibr_open_stage_measurements", default=frozenset()
)


def profiling_enabled(settings: Any) -> bool:
    """Whether ``PIPELINE_MEMORY_PROFILE`` is on in ``settings``."""
    pipeline = getattr(settings, "pipeline", None)
    return getattr(pipeline, "memory_profile", False) is True


def _tracing_python(settings: Any) -> bool:
    pipeline = getattr(settings, "pipeline", None)
    return getattr(pipeline, "memory_profile_tracemalloc", False) is True


def _begin(trace_python: bool) -> tuple[int, _Measurement]:
    ancestors = _open.get()
    with _lock:
        reading = read_process_memory()
        measurement = _Measurement(ancestors=ancestors, start_rss=reading.rss)
        python_peak = None
        if trace_python:
            if tracemalloc.is_tracing():
                python_peak = tracemalloc.get_traced_memory()[1]
            else:
                tracemalloc.start()
        for other_id, other in _active.items():
            if other_id not in ancestors:
                other.overlapped = True
                measurement.overlapped = True
            # Fold the peak so far into every open stage before resetting it.
            other.peak = _max(other.peak, reading.peak)
            other.python_peak = _max(other.python_peak, python_peak)
        if trace_python:
            tracemalloc.reset_peak()
        measurement.peak_reset = reading.peak_resettable and reset_peak_rss()
        measurement_id = next(_ids)
        _active[measurement_id] = measurement
    return measurement_id, measurement


def _end(
    measurement_id: int, measurement: _Measurement, *, files: int, trace_python: bool
) -> StageMemory:
    with _lock:
        reading = read_process_memory()
        _active.pop(measurement_id, None)
        python_peak = None
        if trace_python and tracemalloc.is_tracing():
            python_peak = _max(measurement.python_peak, tracemalloc.get_traced_memory()[1])
    rss_delta = (
        reading.rss - measurement.start_rss
        if reading.rss is not None and measurement.start_rss is not None
        else None
    )
    return StageMemory(
        peak_rss=_max(measurement.peak, reading.peak),
        rss=reading.rss,
        rss_delta=rss_delta,
        swap=reading.swap,
        files=files,
        overlapped=measurement.overlapped,
        peak_reset=measurement.peak_reset,
        python_peak=python_peak,
    )


def _files_of(ctx: Any) -> list[Any]:
    alive = getattr(ctx, "alive", None)
    if callable(alive):
        return list(alive())
    return list(getattr(ctx, "file_states", None) or ())


def _store(target: dict, name: str, record: StageMemory) -> None:
    previous = target.get(name)
    target[name] = record if previous is None else previous.merge(record)


@contextmanager
def measure_stage(ctx: Any, name: str) -> Iterator[None]:
    """Record the process memory of the block as stage ``name`` of ``ctx``.

    A no-op unless ``ctx.settings`` has ``PIPELINE_MEMORY_PROFILE`` on. The
    record goes to every file alive when the block starts and to the chunk's
    ``ctx.scratch["stage_memory"]``.
    """
    settings = getattr(ctx, "settings", None)
    if not profiling_enabled(settings):
        yield
        return
    files = _files_of(ctx)
    trace_python = _tracing_python(settings)
    measurement_id, measurement = _begin(trace_python)
    token = _open.set(measurement.ancestors | {measurement_id})
    try:
        yield
    finally:
        _open.reset(token)
        record = _end(measurement_id, measurement, files=len(files), trace_python=trace_python)
        for fs in files:
            per_file = getattr(fs, "stage_memory", None)
            if isinstance(per_file, dict):
                _store(per_file, name, record)
        scratch = getattr(ctx, "scratch", None)
        if isinstance(scratch, dict):
            _store(scratch.setdefault(SCRATCH_KEY, {}), name, record)


def memory_export(records: Mapping[str, StageMemory] | None) -> dict[str, Any] | None:
    """``extraction.timings.memory`` for one file's records; ``None`` when none."""
    if not records:
        return None
    peaks = [record.peak_rss for record in records.values() if record.peak_rss is not None]
    return {
        "peak_rss_mib": to_mib(max(peaks)) if peaks else None,
        "stages": {name: record.to_export() for name, record in records.items()},
    }
