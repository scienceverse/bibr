"""ParseSegmentStage scores front roles while OCR regions are still resident."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibr.extract.front_role import FrontRolePredictions, RoleScores
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.parse_segment import ParseSegmentStage
from bibr.pipeline.state import FileState


def _ctx(file_states, rm, **config):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=rm,
        config=RunConfig(**config),
    )


class _Classifier:
    def __init__(self):
        self.calls = []

    def predict_pages(self, pages, *, first_page_index=0):
        self.calls.append((pages, first_page_index))
        return FrontRolePredictions(
            {(1, 0): RoleScores(probs={"title": 1.0}, top="title", confidence=1.0)},
            model_version="test",
        )


def _run(rm, *, start_page=None):
    fs = FileState(path=Path("x.pdf"))
    fs.ocr_regions = [[{"content": "hello world.", "task_type": "text"}]]
    contents = MagicMock()
    parser = MagicMock()
    parser.parse.return_value = contents
    parser._deferred_texts = []
    parser.assembler = MagicMock(__len__=lambda self: 0)
    with patch("bibr.structure.pdf_parser.PDFParser", return_value=parser):
        import asyncio

        asyncio.run(ParseSegmentStage().run(_ctx([fs], rm, start_page=start_page)))
    return fs, contents


def test_predictions_are_attached_when_a_classifier_is_available():
    clf = _Classifier()
    rm = MagicMock()
    rm.segmenter = MagicMock(segment_batch=MagicMock(return_value=[]))
    rm.ensure_front_role = MagicMock(return_value=clf)
    fs, contents = _run(rm, start_page=2)
    assert isinstance(contents.front_role_predictions, FrontRolePredictions)
    assert clf.calls[0][1] == 2
    assert fs.error is None


def test_missing_classifier_leaves_contents_untouched():
    rm = MagicMock()
    rm.segmenter = MagicMock(segment_batch=MagicMock(return_value=[]))
    rm.ensure_front_role = MagicMock(return_value=None)
    _fs, contents = _run(rm)
    assert not isinstance(contents.front_role_predictions, FrontRolePredictions)


def test_classifier_failure_never_fails_the_stage(caplog):
    rm = MagicMock()
    rm.segmenter = MagicMock(segment_batch=MagicMock(return_value=[]))
    broken = MagicMock()
    broken.predict_pages.side_effect = RuntimeError("boom")
    rm.ensure_front_role = MagicMock(return_value=broken)
    with caplog.at_level("WARNING"):
        fs, _contents = _run(rm)
    assert fs.error is None
    assert "Front-role classification failed" in caplog.text


@pytest.mark.asyncio
async def test_resource_manager_loads_the_classifier_once(monkeypatch):
    from bibr.pipeline import resources as resources_module

    calls = []

    def _load(settings):
        calls.append(settings)
        return "clf"

    monkeypatch.setattr("bibr.extract.front_role.load_front_role_classifier", _load)
    rm = resources_module.ResourceManager(memory_mode="keep_all", ocr_backend="glm-http")
    assert rm.front_role is None
    assert rm.ensure_front_role() == "clf"
    assert rm.ensure_front_role() == "clf"
    assert rm.front_role == "clf"
    assert len(calls) == 1
