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
async def test_async_form_trims_on_its_own_thread_not_the_shared_executor(monkeypatch):
    threads: list[str] = []

    def trim(pad: int) -> int:
        threads.append(threading.current_thread().name)
        return 1

    monkeypatch.setattr(heap, "_malloc_trim", trim)
    monkeypatch.setattr(heap, "_last_trim", float("-inf"))
    # Saturate the loop's default executor: a trim queued behind its jobs
    # (page rendering, back-half work) would wait for them.
    gate = threading.Event()
    loop = asyncio.get_running_loop()
    busy = [loop.run_in_executor(None, gate.wait) for _ in range(64)]
    try:
        assert await asyncio.wait_for(heap.arelease_free_heap(), timeout=5) is True
    finally:
        gate.set()
        await asyncio.gather(*busy)
    assert threads == [threads[0]]
    assert threads[0].startswith("bibr-heap-trim")
    assert threads[0] != threading.current_thread().name


def test_a_forked_child_gets_a_fresh_executor(monkeypatch):
    monkeypatch.setattr(heap, "_executor", object())
    heap._reset_after_fork()
    assert heap._executor is None
    assert heap._trim_executor() is heap._trim_executor()


def _settings(mode):
    class _Pipeline:
        heap_trim = mode

    class _Settings:
        pipeline = _Pipeline()

    return _Settings()


@pytest.fixture
def async_trims(monkeypatch):
    calls: list[str] = []

    async def trim():
        calls.append("trim")
        return True

    monkeypatch.setattr(heap, "arelease_free_heap", trim)
    return calls


@pytest.mark.parametrize(
    ("mode", "after_chunk", "between_windows"),
    [("off", 0, 0), ("chunk", 1, 0), ("window", 1, 1)],
)
@pytest.mark.asyncio
async def test_modes_say_where_the_batch_trims(async_trims, mode, after_chunk, between_windows):
    settings = _settings(mode)

    await heap.trim_after_chunk(settings)
    assert len(async_trims) == after_chunk
    await heap.trim_between_windows(settings)
    assert len(async_trims) == after_chunk + between_windows


@pytest.mark.asyncio
async def test_settings_without_the_field_trim_after_chunks_only(async_trims):
    assert heap.heap_trim_mode(None) == "chunk"
    assert heap.heap_trim_mode(_settings("bogus")) == "chunk"
    await heap.trim_between_windows(None)
    assert async_trims == []
    await heap.trim_after_chunk(None)
    assert async_trims == ["trim"]


def test_the_setting_defaults_to_chunk_and_reads_the_env(monkeypatch):
    from pydantic import ValidationError

    from bibr.config import PipelineOptions

    monkeypatch.delenv("PIPELINE_HEAP_TRIM", raising=False)
    assert PipelineOptions().heap_trim == "chunk"
    monkeypatch.setenv("PIPELINE_HEAP_TRIM", " Window ")
    assert PipelineOptions().heap_trim == "window"
    monkeypatch.setenv("PIPELINE_HEAP_TRIM", "always")
    with pytest.raises(ValidationError):
        PipelineOptions()
