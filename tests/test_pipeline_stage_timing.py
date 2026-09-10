"""The pipeline stage loop records per-stage wall-clock into ``ctx.scratch``.

This feeds the v10.3 ``extraction.timings`` provenance block.
"""

from pathlib import Path
from unittest.mock import MagicMock

from bibr.config import GlobalSettings
from bibr.pipeline.context import RunConfig
from bibr.pipeline.pipeline import Pipeline
from bibr.pipeline.state import FileState


class _Stage:
    def __init__(self, name: str):
        self.name = name
        self.ctx = None

    async def run(self, ctx):
        # Stash the shared context so the test can inspect the final timings
        # dict after the loop has timed every stage.
        self.ctx = ctx


def _pipeline(stages: list) -> Pipeline:
    return Pipeline(
        stages=stages,
        resources=MagicMock(),
        config=RunConfig(),
        settings=GlobalSettings(),
    )


async def test_loop_records_timing_for_every_stage():
    a, b = _Stage("alpha"), _Stage("beta")
    fs = FileState(path=Path("paper.pdf"))

    await _pipeline([a, b]).process_chunk([fs])

    timings = b.ctx.scratch["stage_timings"]
    assert set(timings) == {"alpha", "beta"}
    assert all(v >= 0.0 for v in timings.values())
