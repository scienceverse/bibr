"""Cleanup on the error and cancellation paths of the pipeline stages."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.config import GlobalSettings
from bibr.pipeline.context import PipelineContext, RunConfig, StageSignals
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.ocr import OcrStage
from bibr.pipeline.state import FileState


async def test_ocr_end_runs_when_consensus_startup_is_cancelled():
    fs = FileState(path=Path("a.pdf"))
    fs.layout_results = [[{"label": "text", "task_type": "text"}]]
    rm = MagicMock()
    rm.await_ocr = AsyncMock(return_value=None)
    rm.ocr.wait_for_server = AsyncMock(return_value=None)
    rm.shutdown_ocr = AsyncMock(return_value=None)
    progress = NullProgress()
    progress.ocr_start = MagicMock()
    progress.ocr_end = MagicMock()
    settings = GlobalSettings()
    settings.ocr.consensus_backend = "glm-http"
    ctx = PipelineContext(
        file_states=[fs],
        progress=progress,
        resources=rm,
        config=RunConfig(ocr_backend="glm-http", memory_mode="aggressive"),
        settings=settings,
        signals=StageSignals(any_needs_ocr=True, preloading_ocr=False),
    )
    with (
        patch(
            "bibr.scan.consensus.start_recognizers",
            AsyncMock(side_effect=asyncio.CancelledError),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await OcrStage().run(ctx)

    progress.ocr_start.assert_called_once()
    progress.ocr_end.assert_called_once()
    rm.shutdown_ocr.assert_awaited_once()


async def test_classifier_close_during_start_leaves_nothing_loaded():
    from bibr.pipeline.classifier_resources import ClassifierResources, ClassifierState
    from tests.pipeline.test_classifier_resources import _Model, _settings

    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()

    def slow_loader(model_id, revision, device):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return _Model("paper")

    resources = ClassifierResources(
        _settings(section_id=None),
        memory_mode="balanced",
        managed_vllm_fraction=0.0,
        cuda_available=False,
        paper_loader=slow_loader,
    )
    start = asyncio.create_task(resources.start())
    await asyncio.wait_for(entered.wait(), 2)
    close = asyncio.create_task(resources.close())
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(asyncio.gather(start, close, return_exceptions=True), 5)

    assert resources._paper.model is None
    assert resources._paper.batcher is None
    assert resources.status()["paper"].state is ClassifierState.CLOSED
    with pytest.raises(RuntimeError):
        await resources.classify_paper("x")


async def test_front_role_load_runs_off_the_event_loop():
    from bibr.pipeline.stages.parse_segment import _predict_front_roles

    loop = asyncio.get_running_loop()
    loop_ran = threading.Event()
    waited: list[bool] = []

    def ensure():
        loop.call_soon_threadsafe(loop_ran.set)
        # The event is set only if the loop keeps running during the load.
        waited.append(loop_ran.wait(1))
        return None

    rm = MagicMock()
    rm.ensure_front_role = ensure
    assert await _predict_front_roles(rm, [[]], first_page_index=0) is None
    assert waited == [True], "front-role load blocked the event loop"


async def test_concurrent_front_role_loads_share_one_load(monkeypatch):
    from bibr.pipeline.resources import ResourceManager

    loads = []
    release = threading.Event()

    def load(settings):
        loads.append(settings)
        assert release.wait(5)
        return "model"

    monkeypatch.setattr("bibr.extract.front_role.load_front_role_classifier", load)
    monkeypatch.setattr("bibr.extract.front_role._note_use", lambda _settings, clf: clf)
    rm = ResourceManager(settings=GlobalSettings())
    calls = [asyncio.to_thread(rm.ensure_front_role) for _ in range(4)]
    gathered = asyncio.gather(*calls)
    await asyncio.sleep(0.05)
    release.set()
    assert await asyncio.wait_for(gathered, 5) == ["model"] * 4
    assert len(loads) == 1
