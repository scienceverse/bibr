"""Upstream errors reach exports and HTTP bodies without the endpoint behind them.

An httpx status error quotes the full request URL, user-info included, so an OCR
server configured as ``https://user:pass@ocr.internal`` used to be written into
``extraction.warnings`` and the serve 422 body verbatim (x-security-1).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from PIL import Image

from bibr.pipeline.state import FileState
from bibr.processing_warnings import ProcessingWarning, WarningCode

_OCR_URL = "https://ocruser:ocr-s3cret@sglang.internal.corp:30000/v1/chat/completions"


def _status_error(url: str = _OCR_URL) -> httpx.HTTPStatusError:
    response = httpx.Response(400, request=httpx.Request("POST", url))
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return exc
    raise AssertionError("unreachable")


async def test_region_warning_names_the_status_not_the_ocr_url():
    from bibr.pipeline.stages.ocr import ocr_page_regions

    async def ocr_fn(_img, _prompt):
        raise _status_error()

    warnings: list[ProcessingWarning] = []
    regions = [{"label": "text", "task_type": "text", "bbox_2d": [0, 0, 100, 100]}]
    page = Image.new("RGB", (200, 200), "white")
    await ocr_page_regions(page, regions, 0, "x.pdf", ocr_fn, warning_sink=warnings.append)

    assert warnings == [
        ProcessingWarning(
            WarningCode.OCR_REGION_FAILED,
            "OCR failed for a region; its text is missing (page 1, region 0, task text): "
            "HTTPStatusError: HTTP 400 Bad Request",
        )
    ]


def _ocr_ctx(fs):
    from unittest.mock import AsyncMock

    from bibr.pipeline.context import PipelineContext, RunConfig, StageSignals

    rm = MagicMock()
    rm.ocr.recognize = AsyncMock(return_value="text")
    rm.ocr.wait_for_server = AsyncMock()
    rm.await_ocr = AsyncMock()
    rm.shutdown_ocr = AsyncMock()
    return PipelineContext(
        file_states=[fs],
        progress=MagicMock(),
        resources=rm,
        config=RunConfig(ocr_backend="glm-llama"),
        signals=StageSignals(any_needs_ocr=True, preloading_ocr=False),
    )


def _ocr_fs(pages: int) -> FileState:
    fs = FileState(path=Path("/tmp/test.pdf"))
    fs.page_indices = list(range(pages))
    fs.page_images = [MagicMock() for _ in range(pages)]
    fs.layout_results = [
        [{"task_type": "text", "label": "text", "bbox_2d": [0, 0, 1, 1]}] for _ in range(pages)
    ]
    return fs


async def test_page_warning_and_file_error_name_the_status_not_the_ocr_url(monkeypatch):
    from bibr.pipeline.stages.ocr import OcrStage

    async def failing_page(*_args, **_kwargs):
        raise _status_error()

    monkeypatch.setattr("bibr.pipeline.stages.ocr.ocr_page_regions", failing_page)
    fs = _ocr_fs(pages=2)
    await OcrStage().run(_ocr_ctx(fs))

    assert [w.message for w in fs.warnings] == [
        "OCR failed for a page; its text is missing (page 1): HTTPStatusError: HTTP 400 Bad Request",
        "OCR failed for a page; its text is missing (page 2): HTTPStatusError: HTTP 400 Bad Request",
    ]
    assert fs.error == "OCR failed for all pages: HTTPStatusError: HTTP 400 Bad Request"


async def test_an_upstream_outage_fails_the_file_without_the_ocr_url(monkeypatch):
    from bibr.exceptions import UpstreamServiceError
    from bibr.pipeline.stages.ocr import OcrStage

    async def outage(*_args, **_kwargs):
        raise UpstreamServiceError("ocr", f"Client error '503' for url '{_OCR_URL}'")

    monkeypatch.setattr("bibr.pipeline.stages.ocr.ocr_page_regions", outage)
    fs = _ocr_fs(pages=1)
    await OcrStage().run(_ocr_ctx(fs))

    assert fs.error == (
        "OCR upstream service failed: Error in ocr: Client error '503' for url '<url>'"
    )


async def test_an_unexpected_ocr_failure_names_the_status_not_the_ocr_url(monkeypatch):
    from bibr.pipeline.stages.ocr import OcrStage

    def fails_before_any_page(*_args, **_kwargs):
        raise _status_error()

    monkeypatch.setattr("bibr.pipeline.stages.ocr.ocr_page_regions", fails_before_any_page)
    fs = _ocr_fs(pages=1)
    await OcrStage().run(_ocr_ctx(fs))

    assert fs.error == "OCR failed: HTTPStatusError: HTTP 400 Bad Request"


async def test_an_enricher_crash_warning_omits_the_request_url():
    from bibr.config import GlobalSettings
    from bibr.pipeline.context import PipelineContext, RunConfig
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.stages.enrich import EnrichmentStage

    class _Crashes:
        name = "crashes"

        async def enrich(self, fs):  # noqa: ARG002
            raise _status_error("https://api.crossref.org/works?mailto=me@example.org&query=x")

    fs = FileState(path=Path("p.pdf"), paper=MagicMock())
    settings = GlobalSettings()
    settings.crossref.enrich = True
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(crossref=True),
        settings=settings,
    )
    await EnrichmentStage([_Crashes()]).run(ctx)

    assert [w.message for w in fs.warnings] == [
        "_Crashes failed: HTTPStatusError: HTTP 400 Bad Request"
    ]


async def test_crossref_failure_warning_and_detail_omit_the_request_url():
    from bibr.pipeline.enricher import CrossrefEnricher

    fs = FileState(path=Path("p.pdf"))
    fs.paper = MagicMock()
    fs.paper.metadata = MagicMock(references=[MagicMock()], doi="")

    async def boom(_refs, **_kwargs):
        raise _status_error("https://api.crossref.org/works?mailto=me@example.org&query=x")

    with patch("bibr.enrich.references.enrich_references", boom):
        outcome = await CrossrefEnricher().enrich(fs)

    expected = "Crossref enrichment failed: HTTPStatusError: HTTP 400 Bad Request"
    assert [w.message for w in fs.warnings] == [expected]
    assert outcome.detail == expected


@pytest.mark.parametrize(
    ("exc_type", "kind", "prefix"),
    [
        ("ProcessingError", "processing", ""),
        ("UpstreamServiceError", "upstream_service", "Error in ocr: "),
    ],
)
def test_serve_error_body_names_no_url(exc_type, kind, prefix):
    from bibr import exceptions
    from bibr.serve.deployments.pipeline import BibrPipelineAPI

    message = f"OCR failed: Client error '400 Bad Request' for url '{_OCR_URL}'"
    if exc_type == "ProcessingError":
        exc = exceptions.ProcessingError(message)
    else:
        exc = exceptions.UpstreamServiceError("ocr", message)
    failure = BibrPipelineAPI._translate_error("x.pdf", exc)

    assert failure["error_kind"] == kind
    assert failure["error"] == f"{prefix}OCR failed: Client error '400 Bad Request' for url '<url>'"
