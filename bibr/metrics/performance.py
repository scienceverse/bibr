"""Request-local performance observations for pipeline benchmarks.

The recorder is deliberately payload-blind: it stores durations and counters,
never document content.  Disabled recorders keep the same call surface while
performing no resource probes or allocations beyond short-lived context objects.
"""

from __future__ import annotations

import contextvars
import resource
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StageObservation:
    calls: int = 0
    total_seconds: float = 0.0


@dataclass(frozen=True)
class ModelObservation:
    calls: int = 0
    items: int = 0


@dataclass(frozen=True)
class PerformanceSnapshot:
    stages: dict[str, StageObservation] = field(default_factory=dict)
    model_calls: dict[str, ModelObservation] = field(default_factory=dict)
    peak_rss_bytes: int | None = None
    peak_vram_bytes: int | None = None


@dataclass
class _MutableRequestMetrics:
    stages: dict[str, list[float | int]] = field(default_factory=lambda: defaultdict(list))
    model_calls: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    peak_rss_bytes: int | None = None
    peak_vram_bytes: int | None = None


class PerformanceRecorder:
    """Collect bounded timing and model-call data keyed by request id."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self._requests: dict[str, _MutableRequestMetrics] = {}
        self._current_request: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"performance_request_{id(self)}", default=None
        )

    @asynccontextmanager
    async def request(self, request_id: str):
        if not self.enabled:
            yield
            return
        metrics = _MutableRequestMetrics()
        metrics.peak_rss_bytes = _peak_rss_bytes()
        metrics.peak_vram_bytes = _peak_vram_bytes()
        self._requests[request_id] = metrics
        token = self._current_request.set(request_id)
        try:
            yield
        finally:
            metrics.peak_rss_bytes = _max_optional(metrics.peak_rss_bytes, _peak_rss_bytes())
            metrics.peak_vram_bytes = _max_optional(metrics.peak_vram_bytes, _peak_vram_bytes())
            self._current_request.reset(token)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        request = self._active()
        if request is None:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            current = request.stages.get(name)
            if current is None:
                request.stages[name] = [1, elapsed]
            else:
                current[0] += 1
                current[1] += elapsed

    def record_model_call(self, name: str, batch_size: int) -> None:
        request = self._active()
        if request is None:
            return
        current = request.model_calls.get(name)
        if current is None:
            request.model_calls[name] = [1, int(batch_size)]
        else:
            current[0] += 1
            current[1] += int(batch_size)

    def snapshot(self, request_id: str) -> PerformanceSnapshot:
        if not self.enabled:
            return PerformanceSnapshot()
        metrics = self._requests.get(request_id)
        if metrics is None:
            return PerformanceSnapshot()
        return PerformanceSnapshot(
            stages={
                name: StageObservation(calls=int(values[0]), total_seconds=float(values[1]))
                for name, values in metrics.stages.items()
            },
            model_calls={
                name: ModelObservation(calls=values[0], items=values[1])
                for name, values in metrics.model_calls.items()
            },
            peak_rss_bytes=metrics.peak_rss_bytes,
            peak_vram_bytes=metrics.peak_vram_bytes,
        )

    def _active(self) -> _MutableRequestMetrics | None:
        if not self.enabled:
            return None
        request_id = self._current_request.get()
        return self._requests.get(request_id) if request_id is not None else None


def _max_optional(a: int | None, b: int | None) -> int | None:
    values = [value for value in (a, b) if value is not None]
    return max(values) if values else None


def _peak_rss_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return None
    # macOS reports bytes; Linux and the BSDs exposed by Python report KiB.
    return value if sys.platform == "darwin" else value * 1024


def _peak_vram_bytes() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        return None
