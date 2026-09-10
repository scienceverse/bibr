"""Characterization + new-behavior tests for OcrStage partial-result handling.

The pre-refactor pattern (`gather(return_exceptions=True)` then
`raise file_errors[0]`) fails the entire file when any single page
errors. These tests pin both that behavior and the post-refactor target
(accumulate warnings, drop failed pages, only fail when all pages fail)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bibr.pipeline.context import PipelineContext, RunConfig, StageSignals
from bibr.pipeline.stages.ocr import OcrStage
from bibr.pipeline.state import FileState


def _make_fs(pages: int = 3) -> FileState:
    fs = FileState(path=Path("/tmp/test.pdf"))
    fs.page_indices = list(range(pages))
    fs.page_images = [MagicMock() for _ in range(pages)]
    fs.layout_results = [
        [{"task_type": "text", "label": "text", "bbox_2d": [0, 0, 1, 1]}] for _ in range(pages)
    ]
    return fs


def _make_ctx(file_states, *, ocr_backend="glm-llama"):
    rm = MagicMock()
    rm.ocr.recognize = AsyncMock(return_value="text")
    rm.ocr.wait_for_server = AsyncMock()
    rm.await_ocr = AsyncMock()
    rm.shutdown_ocr = AsyncMock()
    progress = MagicMock()
    cfg = RunConfig(ocr_backend=ocr_backend)
    return PipelineContext(
        file_states=file_states,
        progress=progress,
        resources=rm,
        config=cfg,
        signals=StageSignals(any_needs_ocr=True, preloading_ocr=False),
    )


@pytest.mark.asyncio
async def test_local_path_one_page_error_does_not_fail_other_pages(monkeypatch):
    """Post-refactor: one page error → other pages still produce regions."""
    fs = _make_fs(pages=3)
    ctx = _make_ctx([fs], ocr_backend="glm-llama")  # local

    async def fake_ocr_page(
        page_img, regions, idx, name, fn, sem, include_figures, settings=None, **kwargs
    ):
        if idx == 1:
            raise RuntimeError("simulated page failure")
        return [{"index": idx, "label": "text", "content": f"page-{idx}", "bbox_2d": [0, 0, 1, 1]}]

    monkeypatch.setattr("bibr.pipeline.stages.ocr.ocr_page_regions", fake_ocr_page)
    await OcrStage().run(ctx)

    # Post-refactor expectation: file is NOT errored; bad page is empty list.
    assert fs.error is None, f"file should not error on partial failure: {fs.error}"
    assert fs.ocr_regions is not None
    # Page 0 and 2 succeeded, page 1 is empty.
    assert any("page-0" in str(r) for r in fs.ocr_regions[0])
    assert fs.ocr_regions[1] == []
    assert any("page-2" in str(r) for r in fs.ocr_regions[2])
    # Warning recorded.
    assert fs.warnings, "expected at least one warning for the failed page"


@pytest.mark.asyncio
async def test_local_path_all_pages_fail_errors_the_file(monkeypatch):
    """Post-refactor: when every page fails, the file IS errored."""
    fs = _make_fs(pages=2)
    ctx = _make_ctx([fs], ocr_backend="glm-llama")

    async def fake_ocr_page(*args, **kwargs):
        raise RuntimeError("all pages broken")

    monkeypatch.setattr("bibr.pipeline.stages.ocr.ocr_page_regions", fake_ocr_page)
    await OcrStage().run(ctx)

    assert fs.error is not None
    assert "ocr_failed" in (fs.error_code or "")


@pytest.mark.asyncio
async def test_local_path_cancelled_error_propagates(monkeypatch):
    """CancelledError must NOT be swallowed — it should crash the chunk so
    the caller's shutdown logic runs."""
    fs = _make_fs(pages=2)
    ctx = _make_ctx([fs], ocr_backend="glm-llama")

    async def fake_ocr_page(
        page_img, regions, idx, name, fn, sem, include_figures, settings=None, **kwargs
    ):
        if idx == 0:
            raise asyncio.CancelledError()
        return []

    monkeypatch.setattr("bibr.pipeline.stages.ocr.ocr_page_regions", fake_ocr_page)
    with pytest.raises(asyncio.CancelledError):
        await OcrStage().run(ctx)


@pytest.mark.asyncio
async def test_remote_path_one_page_error_does_not_fail_other_pages(monkeypatch):
    """Mirror: same partial-result behavior in the remote path."""
    fs = _make_fs(pages=3)
    ctx = _make_ctx([fs], ocr_backend="glm-http")  # triggers _run_remote

    call_count = {"n": 0}

    async def fake_ocr_page(
        page_img, regions, idx, name, fn, sem, include_figures, settings=None, **kwargs
    ):
        call_count["n"] += 1
        if idx == 0:
            raise RuntimeError("simulated remote page failure")
        return [{"index": idx, "label": "text", "content": f"page-{idx}"}]

    monkeypatch.setattr("bibr.pipeline.stages.ocr.ocr_page_regions", fake_ocr_page)
    await OcrStage().run(ctx)

    assert fs.error is None
    assert fs.warnings  # at least one warning
    assert fs.ocr_regions is not None
    assert fs.ocr_regions[0] == []  # failed page is empty
    # Page 1 and 2 succeeded.
    assert any("page-1" in str(r) for r in fs.ocr_regions[1])
    assert any("page-2" in str(r) for r in fs.ocr_regions[2])
