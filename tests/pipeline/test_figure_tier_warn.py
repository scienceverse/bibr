"""FIG_EXTRACT=meta warns at pipeline construction (pipeline-core-12).

No figure-analysis stage consumes the tier yet; constructing a pipeline with
a non-"off" tier must surface the documented warning instead of passing
silently.
"""

import logging

import bibr.pipeline.context as pipeline_context
from bibr.config import GlobalSettings
from bibr.pipeline.context import RunConfig
from bibr.pipeline.pipeline import Pipeline
from bibr.pipeline.resources import ResourceManager


def _build(settings):
    return Pipeline(
        stages=[],
        resources=ResourceManager(settings=settings),
        config=RunConfig(),
        settings=settings,
    )


def test_meta_tier_warns_on_construction(caplog):
    pipeline_context._figure_tier_warned = False
    settings = GlobalSettings(fig={"extract": "meta"})
    with caplog.at_level(logging.WARNING, logger="bibr.pipeline.context"):
        _build(settings)
    assert any("Figure extraction tier" in r.message for r in caplog.records)


def test_off_tier_stays_quiet(caplog):
    pipeline_context._figure_tier_warned = False
    settings = GlobalSettings(fig={"extract": "off"})
    with caplog.at_level(logging.WARNING, logger="bibr.pipeline.context"):
        _build(settings)
    assert not [r for r in caplog.records if "Figure extraction tier" in r.message]
