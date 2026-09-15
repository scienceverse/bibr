"""Partial serialization obeys record boundaries and cancellation semantics."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bibr.config import GlobalSettings
from bibr.extract.document_scope import scope_document_records
from bibr.models import PaperMetadata
from bibr.pipeline import document
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.export import build_result_payload
from bibr.pipeline.stages.post_parse import _build_paper
from bibr.pipeline.state import FileState
from bibr.validation import IssueSeverity, ValidationIssue
from tests.export.test_document_schema import _blocked_paper
from tests.extract.test_document_scope import _document


def _context():
    contents, resolution = _document(same_page=True)
    scope = scope_document_records(contents, resolution)[0]
    original = FileState(Path("collected.pdf"), file_hash="a" * 16, content_sha256="a" * 64)
    ctx = PipelineContext(
        file_states=[original],
        progress=NullProgress(),
        resources=SimpleNamespace(),
        config=RunConfig(no_llm=True, crossref=False, consolidate="off"),
        settings=GlobalSettings(),
    )
    return scope, original, ctx


def _paper(state, *, blocking=True):
    paper = _build_paper(
        state.contents,
        PaperMetadata(doi="10.9999/study-1", title="A printed study"),
        str(state.path),
        state.file_hash,
        "first-record",
    )
    if blocking:
        paper.validation_issues.append(
            ValidationIssue(
                code="VAL_METADATA_FIELD_FAILED",
                severity=IssueSeverity.ERROR,
                message="A metadata field failed",
                origin_stage="extract",
                blocking=True,
            )
        )
    return paper


@pytest.mark.parametrize("fault", ["serialization", "source", "record-ownership", "cancellation"])
async def test_partial_serialization_fault_is_contained_except_cancellation(monkeypatch, fault):
    scope, original, ctx = _context()
    visited = []

    class BlockedStage:
        name = "extract"

        async def run(self, record_ctx):
            state = record_ctx.file_states[0]
            visited.append(state)
            state.paper = _paper(state)

    async def serialize(record_ctx, state):
        if fault == "cancellation":
            raise asyncio.CancelledError()
        if fault == "serialization":
            raise ValueError("private source and secret-token")
        payload = await build_result_payload(record_ctx, state)
        if fault == "source":
            payload["source"]["file_hash"] = "foreign-source"
        else:
            payload["metadata_variant"] = [
                {
                    "variant_id": "foreign",
                    "record_id": "foreign-record",
                    "field": "title",
                    "text": "private source",
                    "is_primary": False,
                }
            ]
        return payload

    monkeypatch.setattr(document, "build_result_payload", serialize)
    if fault == "cancellation":
        with pytest.raises(asyncio.CancelledError):
            await document._extract_record(
                scope, original, ctx, [BlockedStage()], single_record=False
            )
    else:
        result = await document._extract_record(
            scope, original, ctx, [BlockedStage()], single_record=False
        )
        assert (
            result.status == "unresolved" and result.paper is None and result.partial_paper is None
        )
        assert result.reason_flags == [
            *scope.reason_flags,
            "VAL_METADATA_FIELD_FAILED",
            "partial_paper_serialization_failed",
        ]
        assert "secret-token" not in result.model_dump_json()
        assert "private source" not in result.model_dump_json()
        assert "foreign-source" not in result.model_dump_json()
    assert visited[0].paper is None and visited[0].contents is None


async def test_existing_export_survives_paper_cleanup_and_is_not_serialized_again(monkeypatch):
    scope, original, ctx = _context()
    serialize_again = AsyncMock(side_effect=AssertionError("Paper was already exported"))
    monkeypatch.setattr(document, "build_result_payload", serialize_again)

    class ExportedStage:
        name = "export"

        async def run(self, record_ctx):
            state = record_ctx.file_states[0]
            state.paper = _paper(state, blocking=False)
            state.result_json = _blocked_paper(await build_result_payload(record_ctx, state))
            state.free_all()

    result = await document._extract_record(
        scope, original, ctx, [ExportedStage()], single_record=False
    )
    assert result.status == "unresolved" and result.paper is None
    assert result.partial_paper.metadata.title == "A printed study"
    assert result.partial_paper.validation.promotable is False
    serialize_again.assert_not_awaited()


async def test_failed_stage_can_retain_known_blocked_export_with_sanitized_error():
    scope, original, ctx = _context()

    class FailedStage:
        name = "extract"

        async def run(self, record_ctx):
            state = record_ctx.file_states[0]
            state.paper = _paper(state)
            state.result_json = await build_result_payload(record_ctx, state)
            state.set_error("private source and secret-token", code="metadata_incomplete")

    result = await document._extract_record(
        scope, original, ctx, [FailedStage()], single_record=False
    )
    assert result.status == "failed" and result.paper is None
    assert result.partial_paper.metadata.title == "A printed study"
    assert result.error == "Record extraction failed during extract."
    assert "secret-token" not in result.model_dump_json()


async def test_unsafe_scope_has_no_partial_export_or_serializer_call(monkeypatch):
    from dataclasses import replace

    scope, original, ctx = _context()
    scope = replace(scope, contents=None, reason_flags=("ambiguous_record_boundary",))
    serializer = AsyncMock(side_effect=AssertionError("Unsafe scope cannot be serialized"))
    monkeypatch.setattr(document, "build_result_payload", serializer)
    result = await document._extract_record(scope, original, ctx, [], single_record=False)
    assert result.partial_paper is None and result.paper is None
    serializer.assert_not_awaited()
