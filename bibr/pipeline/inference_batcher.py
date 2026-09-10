"""Async micro-batching for synchronous model forward functions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class _Entry(Generic[T, R]):
    item: T
    future: asyncio.Future[R]


class InferenceBatcher(Generic[T, R]):
    """Coalesce concurrent submissions and run one synchronous batch forward."""

    _STOP = object()

    def __init__(
        self,
        fn,
        max_batch_size: int,
        timeout_ms: float,
        name: str,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if timeout_ms < 0:
            raise ValueError("timeout_ms must be >= 0")
        self._fn = fn
        self._max_batch_size = max_batch_size
        self._timeout_seconds = timeout_ms / 1000.0
        self._name = name
        self._queue: asyncio.Queue[_Entry[T, R] | object] = asyncio.Queue()
        self._collector: asyncio.Task[None] | None = None
        self._pending: set[asyncio.Future[R]] = set()
        self._closed = False

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError(f"inference batcher {self._name!r} is closed")
        if self._collector is not None and self._collector.done():
            self._collector = None
            self._queue = asyncio.Queue()
            for future in self._pending:
                if not future.done():
                    try:
                        future.cancel(f"inference batcher {self._name!r} collector stopped")
                    except RuntimeError:
                        # A closed owner loop raises while scheduling callbacks,
                        # after Future.cancel() has already changed the state.
                        if not future.cancelled():
                            raise
            self._pending.clear()
        if self._collector is not None:
            if self._collector.get_loop() is not asyncio.get_running_loop():
                raise RuntimeError(
                    f"inference batcher {self._name!r} collector is running "
                    "on a different event loop"
                )
            return
        self._collector = asyncio.create_task(
            self._collect(), name=f"inference-batcher-{self._name}"
        )

    async def submit(self, item: T) -> R:
        if self._closed:
            raise RuntimeError(f"inference batcher {self._name!r} is closed")
        await self.start()
        future = asyncio.get_running_loop().create_future()
        self._pending.add(future)
        try:
            await self._queue.put(_Entry(item=item, future=future))
            return await future
        finally:
            self._pending.discard(future)

    async def close(self) -> None:
        if self._closed:
            if self._collector is not None:
                await self._collector
            return
        self._closed = True
        collector = self._collector
        if collector is None:
            return
        await self._queue.put(self._STOP)
        await collector

    async def _collect(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self._queue.get()
            if first is self._STOP:
                return
            batch = [first]
            stop_after_batch = False
            deadline = loop.time() + self._timeout_seconds
            while len(batch) < self._max_batch_size:
                try:
                    next_item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        next_item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    except TimeoutError:
                        break
                if next_item is self._STOP:
                    stop_after_batch = True
                    break
                batch.append(next_item)
            await self._run_batch(loop, batch)
            if stop_after_batch:
                return

    async def _run_batch(
        self, loop: asyncio.AbstractEventLoop, batch: list[_Entry[T, R] | object]
    ) -> None:
        active = [entry for entry in batch if isinstance(entry, _Entry) and not entry.future.done()]
        if not active:
            return
        try:
            results = await loop.run_in_executor(None, self._fn, [entry.item for entry in active])
            if len(results) != len(active):
                raise RuntimeError(
                    f"inference batcher {self._name!r} returned {len(results)} results "
                    f"for {len(active)} items"
                )
        except Exception as exc:  # noqa: BLE001 - every caller receives model failures
            for entry in active:
                if not entry.future.done():
                    entry.future.set_exception(exc)
            return
        for entry, result in zip(active, results, strict=True):
            if not entry.future.done():
                entry.future.set_result(result)
