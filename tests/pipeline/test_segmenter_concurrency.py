"""SentenceSegmenter must serialize concurrent ``segment_batch`` calls.

Both segmenter variants wrap a non-thread-safe wtpsplit/ONNX model and run
inference in a thread executor. Concurrent callers sharing one instance
(e.g. multiple serve requests on one worker, or many files in one parse
chunk) must not enter the model concurrently — the stage used to rely on a
per-invocation ``asyncio.Semaphore`` that did not serialize across requests.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from bibr.config import GlobalSettings
from bibr.local.segmenter import SentenceSegmenter as LocalSegmenter
from bibr.serve.deployments.segmenter import SentenceSegmenter as ServeSegmenter


class _ConcurrencyProbe:
    """Stands in for the sync split; records peak concurrent entries."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.inflight = 0
        self.max_inflight = 0

    def __call__(self, texts: list[str]) -> list[list[str]]:
        with self._lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        time.sleep(0.05)  # hold the "model" busy so overlap is observable
        with self._lock:
            self.inflight -= 1
        return [[t] for t in texts]


@pytest.mark.parametrize(
    ("cls", "split_attr"),
    [(ServeSegmenter, "_split_many"), (LocalSegmenter, "_split_many")],
)
async def test_segment_batch_serializes_concurrent_calls(cls, split_attr):
    # Bypass __init__ (it loads the heavy ONNX model); we only exercise the
    # async serialization wrapper around the sync split function.
    seg = object.__new__(cls)
    seg._settings = GlobalSettings()
    probe = _ConcurrencyProbe()
    setattr(seg, split_attr, probe)

    # Force a multi-worker executor so unserialized calls truly run in
    # parallel threads (otherwise a single-worker pool would mask the race).
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=4))

    await asyncio.gather(
        seg.segment_batch(["a"]),
        seg.segment_batch(["b"]),
        seg.segment_batch(["c"]),
    )

    assert probe.max_inflight == 1, (
        f"{cls.__module__}.{cls.__name__}.segment_batch ran {probe.max_inflight} "
        "model calls concurrently — the wtpsplit/ONNX model is shared and not "
        "thread-safe, so segment_batch must serialize concurrent callers."
    )
