"""Real executor cancellation must not orphan servers or release models early."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bibr.config import GlobalSettings
from bibr.local.segmenter import SentenceSegmenter
from bibr.pipeline.resources import ResourceManager


@pytest.mark.parametrize("backend", ["ocr", "preload", "llm"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_close_waits_for_startup_and_cleans_late_server(monkeypatch, backend, cancel):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    server = SimpleNamespace(loaded=True, shutdown=MagicMock(), configure_llm_client=MagicMock())

    def construct(*args, **kwargs):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5), "test did not release the constructor"
        return server

    rm = ResourceManager(ocr_backend="glm-http", settings=GlobalSettings())
    if backend == "llm":
        monkeypatch.setattr("bibr.local.vllm_llm.VllmLlmServer", construct)
        start = rm.start_llm_server("vllm")
        close = rm.close_llm_server
    else:
        monkeypatch.setattr(rm, "_create_ocr_client_for", construct)
        if backend == "preload":
            rm.start_ocr_preload()
        start = rm.await_ocr()
        close = rm.shutdown_ocr
    task = asyncio.create_task(start)
    closing = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        # Preload can enter its constructor before await_ocr runs.
        await asyncio.sleep(0)
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()  # a second interrupt must not interrupt disposal
        closing = asyncio.create_task(close())
        await asyncio.sleep(0)
        assert not closing.done(), "close returned before startup settled"
        if cancel:
            assert not task.done(), "cancelled startup lost ownership of its worker"
    finally:
        release.set()
        results = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
        if closing is not None:
            await asyncio.wait_for(closing, 2)
    assert isinstance(results[0], asyncio.CancelledError) if cancel else results == [None]
    server.shutdown.assert_called_once()
    assert rm.ocr is None and rm._llm_server is None
    assert rm._ocr_future is None and rm._ocr_executor is None
    await close()
    server.shutdown.assert_called_once()


@pytest.mark.parametrize("preload", [False, True])
async def test_cancellation_during_ocr_readiness_disposes_unpublished_client(monkeypatch, preload):
    ready_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def ready():
        ready_started.set()
        await asyncio.Event().wait()

    async def shutdown():
        cleanup_started.set()
        await release_cleanup.wait()

    client = SimpleNamespace(
        loaded=True, wait_for_server=ready, shutdown=AsyncMock(side_effect=shutdown)
    )
    rm = ResourceManager(ocr_backend="glm-http")
    monkeypatch.setattr(rm, "_create_ocr_client_for", lambda *args: client)
    if preload:
        rm.start_ocr_preload()
    task = asyncio.create_task(rm.await_ocr())
    try:
        await asyncio.wait_for(ready_started.wait(), 2)
        assert rm.ocr is None, "an unready client must not be published"
        task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release_cleanup.set()
        result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
    assert isinstance(result[0], asyncio.CancelledError)
    client.shutdown.assert_awaited_once()
    assert rm.ocr is None and rm.ocr_runtime_identity is None


async def test_cancelled_failed_constructor_preserves_cancellation(monkeypatch):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    def construct(*args):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        raise RuntimeError("constructor failed after cancellation")

    rm = ResourceManager(ocr_backend="glm-http")
    monkeypatch.setattr(rm, "_create_ocr_client_for", construct)
    task = asyncio.create_task(rm.await_ocr())
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
    finally:
        release.set()
        result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
    assert isinstance(result[0], asyncio.CancelledError)
    assert rm.ocr is None


async def test_segmenter_cancellation_holds_lock_until_worker_finishes():
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    entries = []
    segmenter = object.__new__(SentenceSegmenter)
    segmenter._loaded = True
    segmenter.model = object()

    def split(texts):
        entries.append(texts[0])
        if texts == ["first"]:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
        assert segmenter.model is not None, "model unloaded while inference was running"
        return [[text] for text in texts]

    segmenter._split_many = split
    first = asyncio.create_task(segmenter.segment_batch(["first"]))
    second = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        second = asyncio.create_task(segmenter.segment_batch(["second"]))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not first.done()
        assert entries == ["first"]
    finally:
        release.set()
        result = await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), 2)
        if second is not None:
            assert await asyncio.wait_for(second, 2) == [["second"]]
        segmenter.unload()
    assert isinstance(result[0], asyncio.CancelledError)
    assert entries == ["first", "second"]
    assert segmenter.model is None


async def test_pipeline_close_cancellation_still_releases_every_resource():
    from bibr.pipeline.context import RunConfig
    from bibr.pipeline.pipeline import Pipeline

    started, release = asyncio.Event(), asyncio.Event()

    async def close_classifiers():
        started.set()
        await release.wait()

    resources = SimpleNamespace(
        close_classifiers=close_classifiers,
        close_llm_server=AsyncMock(),
        shutdown_ocr=AsyncMock(),
        close_llm_client=AsyncMock(),
        close_models=AsyncMock(),
    )
    pipeline = Pipeline(
        stages=[], resources=resources, config=RunConfig(), settings=GlobalSettings()
    )
    task = asyncio.create_task(pipeline.aclose())
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
    assert isinstance(result[0], asyncio.CancelledError)
    resources.close_llm_server.assert_awaited_once()
    resources.shutdown_ocr.assert_awaited_once()
    resources.close_llm_client.assert_awaited_once()
    resources.close_models.assert_awaited_once()


async def test_readiness_cancellation_survives_failed_disposal(monkeypatch):
    started = asyncio.Event()

    async def ready():
        started.set()
        await asyncio.Event().wait()

    client = SimpleNamespace(
        loaded=True,
        wait_for_server=ready,
        shutdown=MagicMock(side_effect=RuntimeError("shutdown failed")),
    )
    rm = ResourceManager(ocr_backend="glm-http")
    monkeypatch.setattr(rm, "_create_ocr_client_for", lambda *args: client)
    task = asyncio.create_task(rm.await_ocr())
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
    assert isinstance(result[0], asyncio.CancelledError)
    client.shutdown.assert_called_once()


async def test_aggressive_parse_cancellation_waits_before_unloading_segmenter():
    from pathlib import Path

    from bibr.pipeline.context import PipelineContext, RunConfig
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.stages.parse_segment import ParseSegmentStage
    from bibr.pipeline.state import FileState

    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    segmenter = object.__new__(SentenceSegmenter)
    segmenter._loaded = True
    segmenter.model = object()

    def split(texts):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        assert segmenter.model is not None
        return [texts]

    segmenter._split_many = split
    parser = MagicMock()
    parser.assembler.__len__.return_value = 1
    parser.assembler.segmentable_texts = ["test sentence"]
    fs = FileState(path=Path("native.xml"), contents=SimpleNamespace(), _native_parser=parser)
    rm = MagicMock()
    rm.segmenter = segmenter
    rm.unload_segmenter.side_effect = segmenter.unload
    context = PipelineContext([fs], NullProgress(), rm, RunConfig(memory_mode="aggressive"))
    task = asyncio.create_task(ParseSegmentStage().run(context))
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        rm.unload_segmenter.assert_not_called()
    finally:
        release.set()
        result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
    assert isinstance(result[0], asyncio.CancelledError)
    rm.unload_segmenter.assert_called_once()
    assert segmenter.model is None
