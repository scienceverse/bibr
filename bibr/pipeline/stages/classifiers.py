"""ClassifierStage — install managed deterministic classifier resources."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class ClassifierStage:
    """Start classifiers independently from optional managed LLM servers."""

    name = "classifiers"
    requires = ()
    produces = ()

    async def run(self, ctx: PipelineContext) -> None:
        try:
            start = ctx.resources.start_classifiers()
            if inspect.isawaitable(start):
                await start
        except Exception:  # noqa: BLE001 - the trained classifiers are an optional tier
            # No file is at fault, and each paper's classification already
            # degrades (with its *_CLASSIFIER_DEGRADED warning) when the
            # trained tier does not answer; failing here killed the chunk.
            logger.warning(
                "Classifier startup failed; papers use the fallback classification",
                exc_info=True,
            )
