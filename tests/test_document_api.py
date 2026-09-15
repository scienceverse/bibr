"""Public document API preserves failures and owns pipeline cleanup."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

import bibr
from bibr.document_api import DocumentResult
from tests.test_api import _export_fixture


def _payload():
    paper = _export_fixture()
    return {
        "document_schema_version": "1.0",
        "document_id": "sha256:abcdef",
        "source": paper["source"],
        "status": "partial",
        "diagnostics": {"detected_record_ids": ["block-1", "block-2"]},
        "records": [
            {"record_id": "block-1", "status": "extracted", "paper": paper},
            {
                "record_id": "block-2",
                "status": "unresolved",
                "paper": None,
                "reason_flags": ["ambiguous_boundary"],
            },
        ],
    }


def test_document_result_exposes_all_outcomes_and_saves(tmp_path):
    result = bibr.DocumentResult(_payload())

    assert not result.ok
    assert result.status == "partial"
    assert len(result.records) == 2
    assert len(result.papers) == 1
    assert result.records[0].paper.title == "A Paper"
    assert result.records[1].paper is None
    assert result.records[1].reason_flags == ["ambiguous_boundary"]
    path = result.save(tmp_path / "results" / "document.json")
    assert json.loads(path.read_text()) == result.data
    assert DocumentResult(result.model).data == result.data


async def test_async_document_api_forwards_options_and_closes(monkeypatch):
    instances = []

    class Pipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.process_document = AsyncMock(return_value=_payload())
            self.aclose = AsyncMock()
            instances.append(self)

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", Pipeline)

    result = await bibr.achew_document("bundle.pdf", no_llm=True, refs=False, ocr="paddle")

    assert isinstance(result, DocumentResult)
    instances[0].process_document.assert_awaited_once_with("bundle.pdf")
    instances[0].aclose.assert_awaited_once()
    assert instances[0].kwargs["ref_parse_strategy"] == "off"
    assert instances[0].kwargs["ocr_backend"] == "paddle"


@pytest.mark.parametrize("failure", [RuntimeError("front stage failed"), asyncio.CancelledError()])
async def test_document_api_closes_after_failure_or_cancellation(monkeypatch, failure):
    pipeline = AsyncMock()
    pipeline.process_document.side_effect = failure
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", lambda **kwargs: pipeline)

    with pytest.raises(type(failure)):
        await bibr.achew_document("bundle.pdf", no_llm=True)

    pipeline.aclose.assert_awaited_once()


async def test_sync_document_api_rejects_active_loop():
    with pytest.raises(RuntimeError, match="await bibr.achew_document"):
        bibr.chew_document("bundle.pdf", no_llm=True)


async def test_document_api_requires_single_file(tmp_path):
    with pytest.raises(IsADirectoryError, match="requires a file"):
        await bibr.achew_document(tmp_path, no_llm=True)


def test_sync_document_api(monkeypatch):
    async_api = AsyncMock(return_value=DocumentResult(_payload()))
    monkeypatch.setattr("bibr.document_api.achew_document", async_api)
    assert bibr.chew_document("bundle.pdf", no_llm=True).status == "partial"
    async_api.assert_awaited_once()


def test_partial_paper_is_accessible_without_entering_successful_papers():
    from tests.export.test_document_schema import _blocked_paper

    payload = _payload()
    payload["records"][1]["partial_paper"] = _blocked_paper(payload["records"][0]["paper"])
    result = DocumentResult(payload)
    assert len(result.papers) == 1
    assert result.records[0].partial_paper is None
    assert result.records[1].paper is None
    assert result.records[1].partial_paper.title == "A Paper"
    assert result.records[1].partial_paper.data["validation"]["promotable"] is False
    assert DocumentResult(result.data).records[1].partial_paper is not None
