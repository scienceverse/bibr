"""Late optional integrity parsing cannot discard a completed paper."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bibr.clients.structured import StructuredResponseError
from bibr.exceptions import ProcessingError
from bibr.export.models import PaperExport
from bibr.models import FundingEntry
from bibr.paper_contents import CanonicalSection, PaperSection, PaperSentence
from bibr.pipeline.context import RunConfig
from bibr.pipeline.stages.export import build_result_payload
from bibr.pipeline.stages.post_parse import post_parse
from bibr.pipeline.state import FileState
from tests.pipeline.test_equation_response_containment import equation_fixture


def _fixture(monkeypatch, error):
    contents, metadata, settings = equation_fixture()
    settings.EQUATION_EXTRACTION = False
    metadata.authors[0].affiliation = "Department of Botany, Example University"
    metadata.authors[0].role = ["Conceptualization"]
    metadata.funding_statement = "Supported by the Garden Council, award G42."
    metadata.funding = [FundingEntry(funder="Garden Council", award_ids=["G42"])]
    contents.sections.append(PaperSection(4, "Funding", 1, None, CanonicalSection.FUNDING))
    contents.sentences.append(PaperSentence(200, metadata.funding_statement, 4, 200))
    client = SimpleNamespace(
        resolve_citations=AsyncMock(return_value=[]),
        extract_research_integrity=AsyncMock(side_effect=error),
    )
    monkeypatch.setattr(
        "bibr.pipeline.stages.post_parse._classify_sections", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "bibr.structure.implicit_sections.detect_implicit_sections", AsyncMock(return_value=None)
    )
    return contents, metadata, settings, client


@pytest.mark.parametrize(
    "category",
    ["empty", "non_json", "non_object", "truncated", "trailing_content", "schema_invalid"],
)
async def test_invalid_integrity_response_retains_paper_and_source_metadata(monkeypatch, category):
    error = StructuredResponseError(category, last_completion="PRIVATE-COMPLETION")
    error.category = "MUTATED-UNTRUSTED-CATEGORY"
    contents, metadata, settings, client = _fixture(monkeypatch, error)
    paper = await post_parse(
        contents=contents,
        file_name="printed.xml",
        file_hash="a" * 64,
        llm_client=client,
        settings=settings,
    )
    client.extract_research_integrity.assert_awaited_once()
    state = FileState(path=Path("printed.xml"))
    state.paper = paper
    context = SimpleNamespace(
        config=RunConfig(crossref=False, consolidate="off"), settings=settings, scratch={}
    )
    exported = PaperExport.model_validate(await build_result_payload(context, state))
    assert exported.metadata.title == metadata.title
    assert len(exported.bib) == 2
    assert exported.author[0].affiliation == "Department of Botany, Example University"
    assert exported.author[0].role == ["Conceptualization"]
    assert exported.metadata.funding_statement == "Supported by the Garden Council, award G42."
    assert exported.funding[0].funder == "Garden Council"
    assert exported.validation.promotable
    issue = next(
        row
        for row in exported.validation.issues
        if row.code == "VAL_INTEGRITY_LLM_RESPONSE_INVALID"
    )
    assert not issue.blocking and issue.severity == "warning"
    assert f"reason:llm_response_invalid:{category}" in issue.evidence_ids
    rendered = exported.model_dump_json()
    assert "PRIVATE-COMPLETION" not in rendered and "MUTATED-UNTRUSTED-CATEGORY" not in rendered


@pytest.mark.parametrize(
    "error", [ProcessingError("operational failure"), asyncio.CancelledError()]
)
async def test_non_response_failures_still_propagate(monkeypatch, error):
    contents, _, settings, client = _fixture(monkeypatch, error)
    with pytest.raises(type(error)) as raised:
        await post_parse(
            contents=contents,
            file_name="printed.xml",
            file_hash="a" * 64,
            llm_client=client,
            settings=settings,
        )
    assert raised.value is error
    assert not any("INTEGRITY_LLM_RESPONSE_INVALID" in row for row in contents.processing_warnings)
