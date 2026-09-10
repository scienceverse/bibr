"""GpuBatcher — async micro-batching broker for shared GPU models.

A single worker process holds one copy of each GPU model (layout, segmenter)
and serves many concurrent requests via the LitServe async loop. Without
coordination, those concurrent requests would each fire their own forward
pass into the GPU at once — unbounded peak VRAM and an OOM on small cards.

``GpuBatcher`` fixes both problems with one mechanism. Callers
``await submit(item)``; a single collector coroutine drains the queue up to
``max_batch_size`` items (or until ``batch_timeout`` elapses), runs **one**
sync ``fn`` on the collected batch in a dedicated executor thread, then
scatters results back to each caller's future. Because there is exactly one
collector awaiting exactly one ``fn`` call at a time:

  - GPU access is serialized → peak VRAM is bounded to a single batch, and
  - concurrent requests' work is coalesced → fuller batches, better GPU use.

``fn`` is the existing batched model call (e.g.
``LayoutDetector._detect_images`` or ``SentenceSegmenter._split_many``);
it must return a list of results positionally aligned with its input list.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class _Entry(Generic[T, R]):
    item: T
    future: asyncio.Future[R]


class GpuBatcher(Generic[T, R]):
    """Coalesce concurrent ``submit`` calls into serialized, fixed-size batches.

    Args:
        fn: Sync callable mapping ``list[item] -> list[result]`` (the GPU
            forward pass). Runs in ``executor``; its output must be the same
            length as, and positionally aligned with, its input.
        max_batch_size: Hard cap on items handed to ``fn`` per call. Should
            match the model's batch size so each call is exactly one batch.
        batch_timeout: Seconds to keep gathering after the first queued item
            before flushing a partial batch. ``0`` flushes whatever is
            already queued immediately (no coalescing wait).
        executor: Executor to run ``fn`` in. Defaults to the event loop's
            default thread pool; pass a single-thread executor to pin all GPU
            work to one thread.
        name: Label for logging.
    """

    def __init__(
        self,
        fn: Callable[[list[T]], list[R]],
        *,
        max_batch_size: int,
        batch_timeout: float,
        executor: Executor | None = None,
        name: str = "gpu",
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self._fn = fn
        self._max_batch_size = max_batch_size
        self._batch_timeout = batch_timeout
        self._executor = executor
        self._name = name

        self._queue: asyncio.Queue[_Entry[T, R]] | None = None
        self._collector: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _ensure_collector(self) -> asyncio.Queue[_Entry[T, R]]:
        """Lazily bind the queue + collector to the running loop.

        Rebinds if the loop changed (e.g. across tests) — an ``asyncio.Queue``
        and its consumer task are tied to the loop they were created on. No
        ``await`` here, so this runs atomically on the single-threaded loop.
        """
        loop = asyncio.get_running_loop()
        if self._collector is None or self._loop is not loop or self._queue is None:
            self._loop = loop
            self._queue = asyncio.Queue()
            self._collector = loop.create_task(self._run(self._queue), name=f"batcher:{self._name}")
        return self._queue

    async def submit(self, item: T) -> R:
        """Enqueue ``item`` for the next batch and await its result."""
        queue = self._ensure_collector()
        future: asyncio.Future[R] = asyncio.get_running_loop().create_future()
        queue.put_nowait(_Entry(item, future))
        return await future

    async def _run(self, queue: asyncio.Queue[_Entry[T, R]]) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await queue.get()
            batch = [first]
            self._gather_more_nowait(queue, batch)
            if self._batch_timeout > 0 and len(batch) < self._max_batch_size:
                await self._gather_more_timed(queue, batch, loop)
            await self._run_batch(loop, batch)

    def _gather_more_nowait(
        self, queue: asyncio.Queue[_Entry[T, R]], batch: list[_Entry[T, R]]
    ) -> None:
        """Pull already-queued entries without yielding, up to the size cap."""
        while len(batch) < self._max_batch_size and not queue.empty():
            batch.append(queue.get_nowait())

    async def _gather_more_timed(
        self,
        queue: asyncio.Queue[_Entry[T, R]],
        batch: list[_Entry[T, R]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Keep gathering until the batch fills or the timeout window closes."""
        deadline = loop.time() + self._batch_timeout
        while len(batch) < self._max_batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), remaining))
            except asyncio.TimeoutError:  # noqa: UP041
                break

    async def _run_batch(self, loop: asyncio.AbstractEventLoop, batch: list[_Entry[T, R]]) -> None:
        """Run ``fn`` on one batch and resolve every caller's future."""
        batch = [entry for entry in batch if not entry.future.done()]
        if not batch:
            return
        items = [e.item for e in batch]
        try:
            results = await loop.run_in_executor(self._executor, self._fn, items)
            if len(results) != len(batch):
                raise RuntimeError(  # noqa: TRY301
                    f"GpuBatcher fn returned {len(results)} results for {len(batch)} items"
                )
        except Exception as exc:  # noqa: BLE001 — fan the failure out to callers
            for e in batch:
                if not e.future.done():
                    e.future.set_exception(exc)
            return
        for e, result in zip(batch, results, strict=True):
            if not e.future.done():
                e.future.set_result(result)

    async def close(self) -> None:
        """Cancel the collector and fail any still-pending submissions."""
        collector = self._collector
        self._collector = None
        if collector is not None:
            collector.cancel()
            try:
                await collector
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                logger.debug("batcher %s collector raised during close", self._name, exc_info=True)
        if self._queue is not None:
            while not self._queue.empty():
                entry = self._queue.get_nowait()
                if not entry.future.done():
                    entry.future.cancel()
        self._queue = None
