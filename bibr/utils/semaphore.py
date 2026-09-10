"""Dual async semaphore — enforces both a global and a per-caller cap."""

from __future__ import annotations

import asyncio


class DualSemaphore:
    """Async context manager acquiring a global then a per-caller semaphore.

    Ensures a single caller cannot monopolise all server-wide slots while
    still allowing overall concurrency up to the global cap.
    """

    def __init__(self, global_sem: asyncio.Semaphore, per_file_limit: int):
        self._global = global_sem
        self._local = asyncio.Semaphore(per_file_limit)

    async def __aenter__(self):
        await self._local.acquire()
        try:
            await self._global.acquire()
        except BaseException:
            self._local.release()
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._global.release()
        self._local.release()
        return False
