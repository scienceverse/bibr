"""Canonical ordered stage plans for local and served pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from bibr.pipeline.enricher import Enricher
    from bibr.pipeline.stage import Stage


def build_stage_plan(
    *,
    mode: Literal["local", "serve"],
    stream_backhalf: bool,
    enrichers: Sequence[Enricher],
) -> tuple[Stage, ...]:
    """Build a fresh, validated-by-owner ordered stage plan.

    Render/OCR runs in bounded page windows in both modes. Serve's shared
    GPU batchers still coalesce windows across concurrent requests.
    """
    from bibr.pipeline.stages.classifiers import ClassifierStage
    from bibr.pipeline.stages.core_checkpoint import CoreCheckpointStage
    from bibr.pipeline.stages.docx import DocxHandlingStage
    from bibr.pipeline.stages.enrich import EnrichmentStage
    from bibr.pipeline.stages.export import ExportStage
    from bibr.pipeline.stages.html import HtmlHandlingStage
    from bibr.pipeline.stages.identity import IdentityValidationStage
    from bibr.pipeline.stages.jats import JatsHandlingStage
    from bibr.pipeline.stages.parse_segment import ParseSegmentStage
    from bibr.pipeline.stages.post_parse import PostParseStage
    from bibr.pipeline.stages.validate import ValidateStage

    common_front = (
        ValidateStage(),
        DocxHandlingStage(),
        JatsHandlingStage(),
        HtmlHandlingStage(),
    )

    if mode == "local":
        from bibr.pipeline.stages.llm_server import LlmServerStage
        from bibr.pipeline.stages.render_ocr import (
            InterleavedRenderOcrStage,
            StreamingRenderOcrStage,
        )

        if stream_backhalf:
            return common_front + (
                ClassifierStage(),
                StreamingRenderOcrStage(
                    parse=ParseSegmentStage(),
                    post_parse=PostParseStage(),
                    identity=IdentityValidationStage(),
                    checkpoint=CoreCheckpointStage(enrichment_requested=bool(enrichers)),
                    enrich=EnrichmentStage(enrichers=list(enrichers)),
                    export=ExportStage(),
                ),
            )
        return common_front + (
            InterleavedRenderOcrStage(),
            ClassifierStage(),
            LlmServerStage(),
            ParseSegmentStage(),
            PostParseStage(),
            IdentityValidationStage(),
            CoreCheckpointStage(enrichment_requested=bool(enrichers)),
            EnrichmentStage(enrichers=list(enrichers)),
            ExportStage(),
        )

    if mode == "serve":
        from bibr.pipeline.stages.render_ocr import InterleavedRenderOcrStage

        return common_front + (
            InterleavedRenderOcrStage(),
            ClassifierStage(),
            ParseSegmentStage(),
            PostParseStage(),
            IdentityValidationStage(),
            CoreCheckpointStage(enrichment_requested=bool(enrichers)),
            EnrichmentStage(enrichers=list(enrichers)),
            ExportStage(),
        )

    raise ValueError(f"Unsupported pipeline mode: {mode!r}")
