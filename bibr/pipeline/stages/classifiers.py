"""ClassifierStage — install managed deterministic classifier resources."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext


class ClassifierStage:
    """Start classifiers independently from optional managed LLM servers."""

    name = "classifiers"
    requires = ()
    produces = ()

    async def run(self, ctx: PipelineContext) -> None:
        start = ctx.resources.start_classifiers()
        if inspect.isawaitable(start):
            await start
