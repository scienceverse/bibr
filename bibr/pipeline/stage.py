"""Stage protocol — a unit of work in the bibr pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext


@runtime_checkable
class Stage(Protocol):
    """A pipeline stage.

    Stages mutate ``ctx.file_states`` in-place. They must:
    - Emit ``ctx.progress.stage_start(self.name)`` / ``stage_end(self.name)``.
    - Set ``fs.error`` via ``fs.set_error(...)`` on per-file failure and
      continue with the remaining files; never raise out of ``run()``.
    - Respect ``ctx.alive()`` — files with ``fs.error`` already set must
      not be processed further.
    """

    name: str

    async def run(self, ctx: PipelineContext) -> None: ...
