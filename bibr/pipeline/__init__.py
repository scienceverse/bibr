"""Composable stage-based pipeline orchestration."""

from __future__ import annotations


def __getattr__(name: str):  # lazy re-exports
    if name == "Stage":
        from bibr.pipeline.stage import Stage

        return Stage
    if name == "PipelineContext":
        from bibr.pipeline.context import PipelineContext

        return PipelineContext
    if name == "RunConfig":
        from bibr.pipeline.context import RunConfig

        return RunConfig
    if name == "ResourceManager":
        from bibr.pipeline.resources import ResourceManager

        return ResourceManager
    if name == "Enricher":
        from bibr.pipeline.enricher import Enricher

        return Enricher
    if name == "CrossrefEnricher":
        from bibr.pipeline.enricher import CrossrefEnricher

        return CrossrefEnricher
    raise AttributeError(f"module 'bibr.pipeline' has no attribute {name!r}")
