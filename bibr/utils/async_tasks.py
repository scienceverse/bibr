"""Cancellation boundaries for work that cannot stop with its awaiting task."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


async def await_owned(
    work: Awaitable[T],
    *,
    on_cancel: Callable[[T], Awaitable[None]] | None = None,
    stop_event: threading.Event | None = None,
) -> T:
    """Settle owned work before propagating cancellation, optionally disposing its result.

    Executor calls keep running when their awaiter is cancelled. Keep the
    future alive until it finishes so a model lock cannot be released early
    and a late-created server cannot lose its owner. Repeated cancellation
    must not interrupt settlement or disposal. The underlying operation must
    have its own timeout; this cannot terminate a stuck worker thread.
    *stop_event* is set as soon as cancellation arrives, so work that checks
    it (a managed server's startup wait) can stop early instead of running
    to that timeout.
    """
    future = asyncio.ensure_future(work)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        if stop_event is not None:
            stop_event.set()

        async def finish() -> None:
            result = await future
            if on_cancel is not None:
                await on_cancel(result)

        settlement = asyncio.create_task(finish())
        while not settlement.done():
            try:
                await asyncio.shield(settlement)
            except asyncio.CancelledError:
                continue
            except Exception:  # the result below retrieves/logs the exception
                break
        try:
            settlement.result()
        except (Exception, asyncio.CancelledError):
            # The original cancellation remains authoritative, even when
            # startup itself failed or best-effort disposal raised.
            logger.debug("Owned work failed during cancellation cleanup", exc_info=True)
        raise
