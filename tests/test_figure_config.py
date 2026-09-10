"""FIG_ settings namespace + RunConfig figure-extract resolution."""

from bibr.config import FigureOptions, GlobalSettings
from bibr.pipeline.context import RunConfig


def test_figure_options_defaults():
    fig = FigureOptions()
    assert fig.extract == "off"
    assert fig.provider == "google"
    assert fig.max_figures == 15
    assert fig.max_concurrency == 4
    assert fig.min_crop_px == 120


def test_figure_options_env_override(monkeypatch):
    monkeypatch.setenv("FIG_EXTRACT", "meta")
    monkeypatch.setenv("FIG_MAX_FIGURES", "3")
    fig = FigureOptions()
    assert fig.extract == "meta"
    assert fig.max_figures == 3


def test_runconfig_figure_tier_defers_to_settings():
    settings = GlobalSettings()
    settings.fig.extract = "meta"
    assert RunConfig().figure_extract_tier(settings) == "meta"
    assert RunConfig(figure_extract="off").figure_extract_tier(settings) == "off"


def test_emit_figures_widened_by_extraction():
    settings = GlobalSettings()
    settings.FIGURE_IMAGES = False
    settings.fig.extract = "off"
    assert RunConfig().emit_figures(settings) is False
    assert RunConfig(figure_extract="meta").emit_figures(settings) is True
    assert RunConfig(include_figures=True).emit_figures(settings) is True
    # user images explicitly off, extraction on -> still emit crops
    assert RunConfig(include_figures=False, figure_extract="meta").emit_figures(settings) is True
