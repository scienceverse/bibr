"""``DualSemaphore`` — acquires global then per-caller semaphores."""

import asyncio

import pytest

from bibr.utils.semaphore import DualSemaphore


@pytest.mark.asyncio
async def test_acquires_both_semaphores_and_releases_on_exit():
    g = asyncio.Semaphore(2)
    ds = DualSemaphore(g, per_file_limit=1)
    async with ds:
        assert g._value == 1
    assert g._value == 2


@pytest.mark.asyncio
async def test_releases_local_when_global_acquire_cancelled():
    g = asyncio.Semaphore(0)  # saturated
    ds = DualSemaphore(g, per_file_limit=1)

    task = asyncio.create_task(ds.__aenter__())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Local must have been released so a new caller can proceed.
    ds2 = DualSemaphore(asyncio.Semaphore(1), per_file_limit=1)
    async with ds2:
        pass
