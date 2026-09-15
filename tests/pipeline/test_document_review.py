"""Independent document orchestration accounting and ownership-boundary review."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bibr.config import snapshot_settings
from bibr.extract.document_scope import scope_document_records
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.document import _extract_record, process_document
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.state import FileState
from tests.extract.test_document_scope_review import _two_articles
from tests.pipeline.test_document_pipeline import _pipeline


async def test_scope_failure_after_detection_retains_all_failed_candidates(monkeypatch):
    pipeline, observations = _pipeline(monkeypatch)

    def broken_scope(*_args, **_kwargs):
        raise ValueError("private source text and secret-token")

    monkeypatch.setattr("bibr.pipeline.document.scope_document_records", broken_scope)

    payload = await process_document(pipeline, "collected.pdf")

    assert payload["status"] == "unresolved"
    assert len(payload["records"]) == 2
    assert {row["record_id"] for row in payload["records"]} == set(
        payload["diagnostics"]["detected_record_ids"]
    )
    assert all(row["status"] == "failed" and row["paper"] is None for row in payload["records"])
    assert all("record_scoping_failed" in row["reason_flags"] for row in payload["records"])
    assert observations["records"] == []
    assert "secret-token" not in str(payload)
    assert "private source text" not in str(payload)


async def test_document_diagnostics_retain_record_detection_reasons(monkeypatch):
    from bibr.pipeline import document

    pipeline, _ = _pipeline(monkeypatch)
    original_resolve = document.resolve_front_matter

    def resolve(*args, **kwargs):
        resolution, issues = original_resolve(*args, **kwargs)
        return replace(
            resolution,
            reason_flags=resolution.reason_flags + ("source_record_detection_evidence",),
        ), issues

    monkeypatch.setattr(document, "resolve_front_matter", resolve)

    payload = await process_document(pipeline, "collected.pdf")

    assert "source_record_detection_evidence" in payload["diagnostics"]["reason_flags"]


@pytest.mark.parametrize("foreign_source", ["text_id", "section_id"])
async def test_prepared_candidates_cannot_supply_evidence_outside_scoped_contents(
    foreign_source, monkeypatch
):
    from bibr.pipeline.stages import post_parse as stage

    source, resolution = _two_articles()
    scope = scope_document_records(source, resolution)[0]
    contents = scope.contents
    assert contents is not None
    prepared = contents.front_matter_resolution
    candidates = list(prepared.candidates)
    byline = candidates[1]
    candidates[1] = replace(
        byline,
        text_ids=(999,) if foreign_source == "text_id" else byline.text_ids,
        section_id=999 if foreign_source == "section_id" else byline.section_id,
        raw_text="Unowned foreign paper evidence",
    )
    prepared = replace(prepared, candidates=tuple(candidates))
    normalize = AsyncMock(side_effect=AssertionError("unowned evidence reached normalization"))
    monkeypatch.setattr(stage, "_normalize_section_structure", normalize)

    with pytest.raises(ValueError, match="Prepared record ownership"):
        await stage.post_parse(
            contents,
            "collected.pdf",
            "source-hash",
            no_llm=True,
            extract_equations=False,
            ref_parse_strategy="off",
            settings=snapshot_settings(),
            prepared_front_matter=prepared,
        )

    normalize.assert_not_awaited()


async def test_record_cancellation_propagates_and_cleans_state_without_closing_shared_resources(
    monkeypatch,
):
    source, resolution = _two_articles()
    scope = scope_document_records(source, resolution)[0]
    original = FileState(Path("collected.pdf"), file_hash="source-hash", content_sha256="a" * 64)
    resources = SimpleNamespace(aclose=AsyncMock())
    ctx = PipelineContext(
        file_states=[original],
        progress=NullProgress(),
        resources=resources,
        config=RunConfig(no_llm=True, crossref=False),
        settings=snapshot_settings(),
    )
    visited = []

    class CancelStage:
        name = "extract"

        async def run(self, record_ctx):
            assert record_ctx.resources is resources
            visited.append(record_ctx.file_states[0])
            raise asyncio.CancelledError

    discard = AsyncMock()
    monkeypatch.setattr("bibr.pipeline.document.discard_prefetches", discard)

    with pytest.raises(asyncio.CancelledError):
        await _extract_record(scope, original, ctx, (CancelStage(),), single_record=False)

    assert len(visited) == 1
    assert visited[0].contents is None
    assert visited[0].prepared_front_matter is None
    discard.assert_awaited_once_with(visited)
    resources.aclose.assert_not_awaited()
