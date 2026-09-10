import types
from unittest.mock import MagicMock, patch

import pytest

import bibr.config
from bibr.pipeline.stages.native_text import NativeTextStage


class _FS:
    def __init__(self):
        self.pdf_bytes = b"%PDF-1.4 fake"
        self.layout_results = [[{"label": "text", "content": "x"}]]
        self.ref_line_geometry = None
        self.path = types.SimpleNamespace(name="paper.pdf")


class _Ctx:
    def __init__(self, fs):
        from bibr.config import snapshot_settings

        self._fs = [fs]
        self.settings = snapshot_settings()
        self.progress = types.SimpleNamespace(
            stage_start=lambda *_: None, stage_end=lambda *_: None
        )

    def alive(self):
        return self._fs


@pytest.fixture
def _restore_strategy():
    prev = bibr.config.Settings.REF_SEG_STRATEGY
    yield
    bibr.config.Settings.REF_SEG_STRATEGY = prev


async def test_capture_populates_geometry_when_geom_selected(_restore_strategy):
    bibr.config.Settings.REF_SEG_STRATEGY = "geom"
    fs = _FS()
    inspection = MagicMock(
        layout_results=fs.layout_results,
        metadata={},
        outline=[],
        reference_lines=[{"text": "Aknin, L. (2013)."}],
    )
    with patch("bibr.pipeline.stages.native_text.inspect_pdf", return_value=inspection):
        await NativeTextStage().run(_Ctx(fs))
    assert fs.ref_line_geometry is not None
    assert fs.ref_line_geometry[0]["text"] == "Aknin, L. (2013)."


async def test_capture_skipped_when_strategy_not_geom(_restore_strategy):
    bibr.config.Settings.REF_SEG_STRATEGY = "llm"
    fs = _FS()
    inspection = MagicMock(
        layout_results=fs.layout_results,
        metadata={},
        outline=[],
        reference_lines=[],
    )
    with patch("bibr.pipeline.stages.native_text.inspect_pdf", return_value=inspection) as inspect:
        await NativeTextStage().run(_Ctx(fs))
    assert inspect.call_args.kwargs["include_ref_geometry"] is False
    assert fs.ref_line_geometry is None
