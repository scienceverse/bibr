"""End-to-end smoke: a stage list runs in order and respects ctx.alive()."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.state import FileState


class RecordingStage:
    def __init__(self, name: str, error_file: str | None = None):
        self.name = name
        self.error_file = error_file
        self.seen: list[str] = []

    async def run(self, ctx: PipelineContext) -> None:
        for fs in ctx.alive():
            self.seen.append(fs.path.name)
            if self.error_file and fs.path.name == self.error_file:
                fs.set_error("induced", code="x", stage=self.name)


@pytest.mark.asyncio
async def test_list_runs_in_order_and_skips_errored():
    a = FileState(path=Path("a.pdf"))
    b = FileState(path=Path("b.pdf"))

    s1 = RecordingStage("s1", error_file="a.pdf")
    s2 = RecordingStage("s2")
    s3 = RecordingStage("s3")

    ctx = PipelineContext(
        file_states=[a, b],
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(),
    )

    for stage in (s1, s2, s3):
        await stage.run(ctx)

    # s1 sees both; s2 and s3 skip the errored one.
    assert s1.seen == ["a.pdf", "b.pdf"]
    assert s2.seen == ["b.pdf"]
    assert s3.seen == ["b.pdf"]
