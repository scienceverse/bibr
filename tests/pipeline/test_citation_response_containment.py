"""Real post-parse/export retains independent native fields after optional linking fails."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bibr.clients.structured import StructuredResponseError
from bibr.config import GlobalSettings
from bibr.exceptions import ProcessingError
from bibr.export.models import PaperExport
from bibr.models import PaperAuthor, PaperMetadata
from bibr.pipeline.context import RunConfig
from bibr.pipeline.stages.export import build_result_payload
from bibr.pipeline.stages.post_parse import _link_citations, post_parse
from bibr.pipeline.state import FileState
from tests.structure.test_citation_linker_response_containment import citation_fixture


async def test_link_warning_is_nonblocking_and_has_only_affected_source_evidence():
    contents, refs = citation_fixture()
    issues = []
    client = SimpleNamespace(
        resolve_citations=AsyncMock(side_effect=StructuredResponseError("schema_invalid"))
    )

    await _link_citations(
        contents, SimpleNamespace(references=refs), "hash", client, validation_issue_sink=issues
    )

    issue = next(row for row in issues if row.code == "VAL_XREF_LLM_RESPONSE_INVALID")
    assert issue.severity == "warning"
    assert issue.blocking is False
    assert issue.origin_stage == "post_parse"
    assert issue.count == 4
    assert set(issue.evidence_ids) == {
        "reason:llm_response_invalid:schema_invalid",
        "text:5",
        "text:6",
        "text:7",
        "text:8",
    }
    assert contents.citation_receipt is not None
    assert len(contents.xrefs) == 3


async def run_native_post_parse(monkeypatch, error):
    contents, refs = citation_fixture()
    metadata = PaperMetadata(
        doi="10.9999/printed",
        title="A printed study of plant growth",
        abstract="Plant growth was measured.",
        authors=[PaperAuthor(author_id=1, given="Mara", family="Quill", affiliation="")],
        references=refs,
    )
    contents.preparsed_metadata = metadata
    contents.native_references = refs
    client = SimpleNamespace(resolve_citations=AsyncMock(side_effect=error))
    monkeypatch.setattr(
        "bibr.pipeline.stages.post_parse._classify_sections", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "bibr.structure.implicit_sections.detect_implicit_sections", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "bibr.extract.research_integrity.extract_structured_integrity", AsyncMock(return_value=None)
    )
    paper = await post_parse(
        contents=contents,
        file_name="printed.xml",
        file_hash="a" * 64,
        llm_client=client,
        extract_equations=False,
        settings=GlobalSettings(),
    )
    return paper, metadata, refs


async def test_real_post_parse_export_keeps_metadata_references_and_confirmed_links(monkeypatch):
    paper, metadata, refs = await run_native_post_parse(
        monkeypatch, StructuredResponseError("non_json", last_completion="RAW-PROVIDER-SECRET")
    )
    assert paper.metadata is metadata
    assert paper.metadata.references == refs
    state = FileState(path=Path("printed.xml"))
    state.paper = paper
    context = SimpleNamespace(
        config=RunConfig(crossref=False, consolidate="off"),
        settings=GlobalSettings(),
        scratch={},
    )
    exported = PaperExport.model_validate(await build_result_payload(context, state))

    assert exported.metadata.title == "A printed study of plant growth"
    assert exported.metadata.doi == "10.9999/printed"
    assert [(row.given, row.family) for row in exported.author] == [("Mara", "Quill")]
    assert [row.title for row in exported.bib] == [f"Printed study {i}" for i in range(1, 5)]
    assert exported.validation is not None
    assert exported.validation.promotable is True
    issue = next(
        row for row in exported.validation.issues if row.code == "VAL_XREF_LLM_RESPONSE_INVALID"
    )
    assert issue.blocking is False
    assert issue.severity == "warning"
    assert exported.extraction is not None
    assert exported.extraction.diagnostics is not None
    diagnostic = exported.extraction.diagnostics.citation_linking
    assert diagnostic is not None
    assert {row.text_id for row in diagnostic.candidates if row.accepted} == {1, 2, 8}
    assert any(
        "llm_response_invalid:non_json" in row.rejection_reasons for row in diagnostic.candidates
    )
    assert "RAW-PROVIDER-SECRET" not in exported.model_dump_json()


@pytest.mark.parametrize(
    "error",
    [ProcessingError("typed failure", error_code="llm_invalid_output"), asyncio.CancelledError()],
)
async def test_real_post_parse_keeps_generic_typed_failure_and_cancellation_policy(
    monkeypatch, error
):
    with pytest.raises(type(error)) as raised:
        await run_native_post_parse(monkeypatch, error)
    assert raised.value is error
