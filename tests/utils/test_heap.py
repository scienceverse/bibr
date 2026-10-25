"""bibr.utils.heap — returning freed C-heap pages to the OS between windows/chunks."""

from __future__ import annotations

import asyncio
import ctypes
import platform
import threading

import pytest

from bibr.utils import heap


@pytest.fixture
def fake_trim(monkeypatch):
    calls: list[int] = []

    def trim(pad: int) -> int:
        calls.append(pad)
        return 1

    monkeypatch.setattr(heap, "_malloc_trim", trim)
    monkeypatch.setattr(heap, "_last_trim", float("-inf"))
    return calls


def test_trims_with_zero_padding(fake_trim):
    assert heap.release_free_heap() is True
    assert fake_trim == [0]


def test_calls_closer_than_min_interval_are_skipped(fake_trim, monkeypatch):
    clock = iter([100.0, 100.5, 101.6])
    monkeypatch.setattr(heap.time, "monotonic", lambda: next(clock))

    assert heap.release_free_heap(min_interval=1.0) is True
    assert heap.release_free_heap(min_interval=1.0) is False
    assert heap.release_free_heap(min_interval=1.0) is True
    assert len(fake_trim) == 2


def test_no_malloc_trim_is_a_noop(monkeypatch):
    monkeypatch.setattr(heap, "_malloc_trim", None)
    assert heap.release_free_heap(min_interval=0) is False


def test_resolves_once_and_caches_absence(monkeypatch):
    resolved = []

    def resolve():
        resolved.append(1)

    monkeypatch.setattr(heap, "_malloc_trim", heap._UNRESOLVED)
    monkeypatch.setattr(heap, "_resolve", resolve)
    assert heap.release_free_heap(min_interval=0) is False
    assert heap.release_free_heap(min_interval=0) is False
    assert resolved == [1]


def test_a_failing_trim_never_raises(monkeypatch):
    def trim(pad: int) -> int:
        raise OSError("no")

    monkeypatch.setattr(heap, "_malloc_trim", trim)
    monkeypatch.setattr(heap, "_last_trim", float("-inf"))
    assert heap.release_free_heap(min_interval=0) is False


def test_off_linux_there_is_nothing_to_resolve(monkeypatch):
    monkeypatch.setattr(heap.sys, "platform", "darwin")
    assert heap._resolve() is None


def test_libc_without_malloc_trim_resolves_to_none(monkeypatch):
    class _NoTrim:
        def __init__(self, name):
            pass

        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(heap.sys, "platform", "linux")
    monkeypatch.setattr(heap.ctypes, "CDLL", _NoTrim)
    assert heap._resolve() is None


@pytest.mark.skipif(
    platform.system() != "Linux" or platform.libc_ver()[0] != "glibc", reason="glibc only"
)
def test_real_malloc_trim_runs_on_glibc(monkeypatch):
    monkeypatch.setattr(heap, "_malloc_trim", heap._UNRESOLVED)
    monkeypatch.setattr(heap, "_last_trim", float("-inf"))
    assert heap.release_free_heap(min_interval=0) is True
    assert isinstance(heap._malloc_trim, ctypes._CFuncPtr)


@pytest.mark.asyncio
async def test_async_form_trims_off_the_event_loop(monkeypatch):
    threads: list[int] = []

    def trim(pad: int) -> int:
        threads.append(threading.get_ident())
        return 1

    monkeypatch.setattr(heap, "_malloc_trim", trim)
    monkeypatch.setattr(heap, "_last_trim", float("-inf"))
    assert await heap.arelease_free_heap() is True
    assert threads and threads[0] != threading.get_ident()
    assert asyncio.get_running_loop().is_running()
