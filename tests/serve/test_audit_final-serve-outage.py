"""Final-review regression: a serve outage stays transient for a remote batch.

Serve answers a stage failure caused by an untyped exception with only its
stage and code, so the OCR-server text ``bibr batch --serve-url`` used to read
as an outage was gone and the paper failed for good. The outage flag now
crosses the wire: the failure is still a 422 that hides the exception text,
but it says ``outage: true`` and ends in bibr's own words that every remote
client release reads as transient.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

pytest.importorskip("litserve")

from fastapi import HTTPException  # noqa: E402

from bibr.batch.ledger import UPSTREAM_UNAVAILABLE  # noqa: E402
from bibr.batch.remote import looks_transient  # noqa: E402
from bibr.config import GlobalSettings  # noqa: E402
from bibr.exceptions import ProcessingError  # noqa: E402
from bibr.pipeline.context import RunConfig  # noqa: E402
from bibr.pipeline.pipeline import Pipeline  # noqa: E402
from bibr.pipeline.stages.llm_server import LlmServerStage  # noqa: E402
from bibr.pipeline.stages.ocr import OcrStage, fail_ocr_targets  # noqa: E402
from bibr.serve import jobs as jobs_mod  # noqa: E402
from bibr.serve.deployments.pipeline import BibrPipelineAPI  # noqa: E402
from bibr.serve.jobs import MemoryJobStore  # noqa: E402
from tests.batch.test_remote import FakeServe, _executor, _items, _run  # noqa: E402
from tests.serve.test_jobs import _FakeTracker  # noqa: E402

_REFUSED = "[Errno 111] Connection refused by ocr.internal:30000"
_CACHE_PATH = "/app/.hf_cache/models--PaddlePaddle--PP-DocLayoutV3/model.safetensors"


class _OcrServerGone:
    """The real OCR stage on a file whose second page finds the server gone."""

    name = "ocr"

    async def run(self, ctx):
        for fs in ctx.file_states:
            fs.page_indices = [0, 1]
            fs.page_images = [MagicMock(), MagicMock()]
            fs.layout_results = [[{"label": "text", "bbox_2d": [0, 0, 1, 1]}]] * 2
        ctx.signals.any_needs_ocr = True
        await OcrStage().run(ctx)


async def _ocr_page(page_img, regions, idx, *args, **kwargs):
    if idx:
        raise httpx.ConnectError(_REFUSED)
    return [{"index": idx, "label": "text", "content": "text", "bbox_2d": [0, 0, 1, 1]}]


class _OcrInitFailure:
    name = "ocr"

    async def run(self, ctx):
        fail_ocr_targets(ctx.file_states, RuntimeError(_REFUSED), stage=self.name, log=False)


class _LayoutCacheMiss:
    """A failure of the file's own: an untyped cause, no outage."""

    name = "layout"

    async def run(self, ctx):
        exc = OSError(f"[Errno 2] No such file or directory: '{_CACHE_PATH}'")
        for fs in ctx.file_states:
            fs.set_error(
                f"Layout initialization failed: {exc}",
                code="layout_failed",
                stage=self.name,
                exc=exc,
            )


def _resources() -> MagicMock:
    resources = MagicMock()
    resources.await_ocr = AsyncMock()
    resources.ocr.wait_for_server = AsyncMock()
    resources.start_llm_server = AsyncMock(side_effect=RuntimeError(f"vllm-mlx died: {_REFUSED}"))
    return resources


async def _processing_error(stage) -> ProcessingError:
    pipeline = Pipeline(
        stages=[stage],
        resources=_resources(),
        config=RunConfig(ocr_backend="glm-llama"),
        settings=GlobalSettings(),
    )
    with pytest.raises(ProcessingError) as raised:
        await pipeline.process_file("paper.pdf", content=b"%PDF-1.4")
    return raised.value


async def _failed_job(exc: ProcessingError, tmp_path: Path) -> tuple[dict, int]:
    """The error and status a failed job reports, as the serve records them."""
    failure = BibrPipelineAPI._translate_error("paper.pdf", exc)
    with pytest.raises(HTTPException) as raised:
        await BibrPipelineAPI(upload_root=tmp_path).encode_response(failure)
    store = MemoryJobStore()
    job = await store.create(filename="paper.pdf")
    await jobs_mod._run_job(
        store=store,
        job_id=job.job_id,
        descriptor={"upload_id": "opaque"},
        tracker=_FakeTracker(error=raised.value),
    )
    got = await store.get(job.job_id)
    assert got is not None
    return got.status_dict()["error"], got.http_status


@pytest.fixture
def ocr_server_gone(monkeypatch):
    monkeypatch.setattr("bibr.pipeline.stages.ocr.ocr_page_regions", _ocr_page)
    return _OcrServerGone()


async def test_ocr_server_gone_is_a_422_outage_without_its_exception_text(
    ocr_server_gone, tmp_path
):
    exc = await _processing_error(ocr_server_gone)
    assert exc.outage is True
    assert _REFUSED in str(exc)  # the log keeps it

    error, status = await _failed_job(exc, tmp_path)

    assert status == 422
    assert error == {
        "message": "Processing failed in ocr (ocr_failed): service temporarily unavailable",
        "error_code": "ocr_failed",
        "outage": True,
    }


@pytest.mark.parametrize("client", ["current", "marker-only"])
async def test_remote_batch_retries_an_ocr_outage_and_records_it_as_unavailable(
    ocr_server_gone, tmp_path, client
):
    error, status = await _failed_job(await _processing_error(ocr_server_gone), tmp_path)
    if client == "marker-only":
        # A client that predates the field reads only the message.
        error = {key: value for key, value in error.items() if key != "outage"}
        assert looks_transient(error["message"])

    serve = FakeServe(fail_once={"flaky": (error, status)})
    executor, _ = _executor(serve, retries=2)
    _, [(_, recovered)] = await _run(executor, _items(tmp_path, "flaky"))

    assert recovered.ok
    assert executor.stats["transient_retries"] == 1
    assert len(serve.submits) == 2

    serve = FakeServe(fail={"down": (error, status)})
    executor, _ = _executor(serve, retries=1)
    _, [(_, outcome)] = await _run(executor, _items(tmp_path, "down"))

    assert outcome.error_code == UPSTREAM_UNAVAILABLE
    assert outcome.extra["transient_exhausted"] is True


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        (_OcrInitFailure(), "Processing failed in ocr (ocr_failed)"),
        (LlmServerStage(), "Processing failed in llm_server (llm_server_failed)"),
    ],
)
async def test_every_untyped_outage_site_reaches_the_client_as_an_outage(stage, message, tmp_path):
    error, status = await _failed_job(await _processing_error(stage), tmp_path)

    assert status == 422
    assert error["outage"] is True
    assert error["message"] == f"{message}: service temporarily unavailable"
    assert "refused" not in error["message"]


async def test_remote_batch_trusts_the_outage_field_over_the_message(tmp_path):
    error = {"message": "Processing failed in layout (layout_failed)", "outage": True}
    assert not looks_transient(error["message"])
    serve = FakeServe(fail_once={"flaky": (error, 422)})
    executor, _ = _executor(serve, retries=2)

    _, [(_, outcome)] = await _run(executor, _items(tmp_path, "flaky"))

    assert outcome.ok
    assert executor.stats["transient_retries"] == 1


async def test_a_failure_of_the_paper_stays_permanent_and_hides_its_text(tmp_path, caplog):
    exc = await _processing_error(_LayoutCacheMiss())
    assert exc.outage is False

    error, status = await _failed_job(exc, tmp_path)

    assert (status, error) == (
        422,
        {"message": "Processing failed in layout (layout_failed)", "error_code": "layout_failed"},
    )
    assert _CACHE_PATH in caplog.text

    serve = FakeServe(fail={"broken": (error, status)})
    executor, _ = _executor(serve, retries=2)
    _, [(_, outcome)] = await _run(executor, _items(tmp_path, "broken"))

    assert outcome.error_code == "layout_failed"
    assert len(serve.submits) == 1
    assert executor.stats["transient_retries"] == 0


async def test_typed_processing_error_recorded_as_an_outage_keeps_the_flag():
    """A stage may record its own ``ProcessingError`` as an outage."""

    class _Stage:
        name = "post_parse"

        async def run(self, ctx):
            for fs in ctx.file_states:
                fs.set_error(
                    "post-parse failed",
                    code="post_parse_failed",
                    stage=self.name,
                    exc=ProcessingError("post-parse failed", error_code="post_parse_failed"),
                    outage=True,
                )

    exc = await _processing_error(_Stage())

    assert (exc.outage, exc.failed_stage) == (True, "post_parse")
