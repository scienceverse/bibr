"""PipelineContext helpers."""

from pathlib import Path
from unittest.mock import MagicMock

from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.state import FileState


def test_alive_filters_errored():
    ok = FileState(path=Path("a.pdf"))
    err = FileState(path=Path("b.pdf"))
    err.set_error("boom", code="x", stage="y")

    ctx = PipelineContext(
        file_states=[ok, err],
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(),
    )

    assert ctx.alive() == [ok]


def test_alive_empty_when_all_errored():
    err = FileState(path=Path("a.pdf"))
    err.set_error("boom", code="x", stage="y")

    ctx = PipelineContext(
        file_states=[err],
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(),
    )

    assert ctx.alive() == []


def test_run_config_has_include_figures_default_none():
    """Default is None → OCR stage falls back to Settings.FIGURE_IMAGES."""
    from bibr.pipeline.context import RunConfig

    cfg = RunConfig()
    assert cfg.include_figures is None


def test_run_config_include_figures_settable():
    from bibr.pipeline.context import RunConfig

    cfg = RunConfig(include_figures=True)
    assert cfg.include_figures is True

    cfg_off = RunConfig(include_figures=False)
    assert cfg_off.include_figures is False


def test_run_config_defaults_to_paddle_and_preserves_optional_ocr_profile():
    assert RunConfig().ocr_backend == "paddle"
    assert RunConfig().ocr_profile is None
    assert RunConfig(ocr_profile="glm").ocr_profile == "glm"
