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
from bibr.processing_warnings import ProcessingWarning, WarningCode


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
    # Warning recorded, with the page numbered from 1 like the region warnings.
    assert fs.warnings == [
        ProcessingWarning(
            WarningCode.OCR_PAGE_FAILED,
            "OCR failed for a page; its text is missing (page 2): "
            "RuntimeError: simulated page failure",
        )
    ]


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


def _real_pages(fs: FileState) -> FileState:
    from PIL import Image

    fs.page_images = [Image.new("RGB", (64, 64), "white") for _ in fs.page_indices]
    return fs


@pytest.mark.asyncio
async def test_an_ocr_server_that_dies_mid_file_fails_the_file_as_an_outage():
    """The OCR server goes away after page 1: the transport's retries end in a
    refused connection for every later region. Shipped blank, those regions
    failed the file as ocr_mostly_failed (a verdict on the paper, which a
    resumed batch skips) or let it pass with text missing."""
    import httpx

    fs = _real_pages(_make_fs(pages=4))
    ctx = _make_ctx([fs], ocr_backend="glm-llama")
    calls = {"n": 0}

    async def recognize(image, prompt):
        calls["n"] += 1
        if calls["n"] > 1:
            raise httpx.ConnectError("[Errno 111] Connection refused")
        return "text"

    ctx.resources.ocr.recognize = recognize
    await OcrStage().run(ctx)

    assert fs.error_code == "ocr_failed"
    assert fs.error == "OCR upstream service failed: [Errno 111] Connection refused"
    assert fs.error_outage is True


@pytest.mark.asyncio
async def test_a_region_that_times_out_still_ships_blank_with_a_warning():
    """A timeout can be the region's own (a dense table): not an outage, so the
    file keeps the partial-result behaviour."""
    import httpx

    fs = _real_pages(_make_fs(pages=3))
    ctx = _make_ctx([fs], ocr_backend="glm-llama")
    calls = {"n": 0}

    async def recognize(image, prompt):
        calls["n"] += 1
        if calls["n"] == 2:
            raise httpx.ReadTimeout("timed out")
        return "text"

    ctx.resources.ocr.recognize = recognize
    await OcrStage().run(ctx)

    assert fs.error is None
    assert [w.code for w in fs.warnings] == [WarningCode.OCR_REGION_FAILED]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 502, 503])
async def test_a_busy_answer_for_one_region_still_ships_it_blank(status):
    """A server that answers 429/502/503 after the transport's retries is up,
    if busy. Failing the whole file for one region would turn a usable export
    into a failed paper (a hard failure for `bibr chew` and the serve)."""
    import httpx

    fs = _real_pages(_make_fs(pages=3))
    ctx = _make_ctx([fs], ocr_backend="paddle-http")
    calls = {"n": 0}
    request = httpx.Request("POST", "http://ocr:8080/v1/chat/completions")

    async def recognize(image, prompt):
        calls["n"] += 1
        if calls["n"] == 2:
            response = httpx.Response(status, request=request)
            raise httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)
        return "text"

    ctx.resources.ocr.recognize = recognize
    await OcrStage().run(ctx)

    assert fs.error is None
    assert [w.code for w in fs.warnings] == [WarningCode.OCR_REGION_FAILED]


@pytest.mark.asyncio
async def test_an_open_breaker_keeps_the_upstream_error_after_a_refused_page(monkeypatch):
    """The serve's breaker opens after page 1's refused connection. The file
    must carry the UpstreamServiceError, which the serve answers with 502;
    the raw transport error would come out as a 422 processing error."""
    import httpx

    from bibr.exceptions import UpstreamServiceError

    fs = _make_fs(pages=3)
    ctx = _make_ctx([fs], ocr_backend="serve-http")
    breaker_open = UpstreamServiceError("ocr", "Circuit breaker 'ocr' is OPEN")

    async def fake_ocr_page(
        page_img, regions, idx, name, fn, sem, include_figures, settings=None, **kwargs
    ):
        if idx == 0:
            raise httpx.ConnectError("[Errno 111] Connection refused")
        raise breaker_open

    monkeypatch.setattr("bibr.pipeline.stages.ocr.ocr_page_regions", fake_ocr_page)
    await OcrStage().run(ctx)

    assert fs.error_code == "ocr_failed"
    assert fs.original_error is breaker_open
    assert fs.error_outage is True
