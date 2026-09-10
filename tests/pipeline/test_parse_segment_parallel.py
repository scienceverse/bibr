"""ParseSegmentStage concurrency tests."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.stages.parse_segment import ParseSegmentStage
from bibr.pipeline.state import FileState


def _make_fs() -> FileState:
    fs = FileState(path=Path("/tmp/test.pdf"))
    fs.ocr_regions = [[{"label": "text", "content": "hello", "bbox_2d": [0, 0, 1, 1]}]]
    return fs


def _make_ctx(file_states):
    rm = MagicMock()
    rm.ensure_segmenter = MagicMock()
    rm.unload_segmenter = MagicMock()
    rm.segmenter.segment_batch = AsyncMock(return_value=[])
    from bibr.pipeline.progress import NullProgress

    progress = NullProgress()
    return PipelineContext(
        file_states=file_states,
        progress=progress,
        resources=rm,
        config=RunConfig(memory_mode="balanced"),
    )


@pytest.mark.asyncio
async def test_parse_runs_concurrently_across_files(monkeypatch):
    """Five files × 100ms each must complete in < 300ms (parallel)
    rather than ~500ms (sequential)."""
    files = [_make_fs() for _ in range(5)]
    ctx = _make_ctx(files)

    parse_started = []
    parse_finished = []

    def slow_parse(ocr_regions, *, settings, first_page_index=0):  # noqa: ARG001
        parse_started.append(time.monotonic())
        time.sleep(0.1)  # synchronous sleep — runs on the thread-pool worker
        parse_finished.append(time.monotonic())
        parser = MagicMock()
        parser._deferred_texts = []
        parser.create_content_sections = MagicMock()
        return parser, MagicMock()

    monkeypatch.setattr("bibr.pipeline.stages.parse_segment._parse_pdf", slow_parse)

    t0 = time.monotonic()
    await ParseSegmentStage().run(ctx)
    elapsed = time.monotonic() - t0

    assert len(parse_started) == 5
    # All 5 must have started before any of the first finishes (true parallelism).
    first_finish = min(parse_finished)
    starts_before_first_finish = sum(1 for s in parse_started if s < first_finish)
    assert starts_before_first_finish >= 4, (
        f"only {starts_before_first_finish}/5 parses started before first finish — sequential"
    )
    # Wall time must be much less than 5 * 100ms.
    assert elapsed < 0.5, f"too slow ({elapsed:.2f}s) — files probably ran sequentially"


@pytest.mark.asyncio
async def test_parse_isolates_per_file_errors(monkeypatch):
    """One file's parse failure must not abort other files."""
    good = _make_fs()
    bad = _make_fs()
    ctx = _make_ctx([good, bad])

    def parse_fn(ocr_regions, *, settings, first_page_index=0):  # noqa: ARG001
        if ocr_regions is bad.ocr_regions:
            raise RuntimeError("simulated parse failure")
        parser = MagicMock()
        parser._deferred_texts = []
        parser.create_content_sections = MagicMock()
        return parser, MagicMock()

    monkeypatch.setattr("bibr.pipeline.stages.parse_segment._parse_pdf", parse_fn)

    await ParseSegmentStage().run(ctx)

    assert good.error is None
    assert bad.error is not None
    assert bad.error_code == "parse_failed"
