"""A field protocol failure retains independent extraction with blocking export."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from bibr.clients.llm import LLMClient
from bibr.clients.structured import StructuredResponseError
from bibr.config import GlobalSettings
from bibr.export.json_export import build_paper_export
from bibr.extract.extractor import MetadataExtractor
from bibr.extract.front_matter import FrontMatterBlock, FrontMatterCandidate, FrontMatterResolution
from bibr.models import PaperReference
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence
from bibr.pipeline.stages.post_parse import (
    _build_paper,
    _finalize_abstract_and_keywords,
    _resolve_selected_title,
)
from bibr.schemas import AuthorLLM, AuthorsLLM, PaperClassificationLLM


def _contents(abstract_heading="Abstract"):
    contents = PaperContents(
        sentences=[
            PaperSentence(1, "Mara Quill", 1, 1, page_number=1),
            PaperSentence(2, "doi:10.9999/printed", 1, 2, page_number=1),
            PaperSentence(
                3, "Shade improved seedling growth in a controlled experiment.", 2, 3, page_number=1
            ),
            PaperSentence(4, "We measured growth over six weeks.", 3, 4, page_number=1),
            PaperSentence(
                5, "Brook E. (2020). Earlier study. Journal of Plants, 3, 1-9.", 4, 5, page_number=2
            ),
        ],
        sections=[
            PaperSection(1, "SHADE AND SEEDLING GROWTH", 1, None, CanonicalSection.TITLE),
            PaperSection(2, abstract_heading, 2, 1, CanonicalSection.ABSTRACT),
            PaperSection(3, "Introduction", 2, 1, CanonicalSection.INTRODUCTION),
            PaperSection(4, "References", 2, 1, CanonicalSection.REFERENCES),
        ],
        tables=[],
        links=[],
        sections_text={},
    )
    candidates = []
    for index, (kind, text, section_id, text_ids, roles) in enumerate(
        [
            ("heading", contents.sections[0].header, 1, (), {"title", "heading"}),
            ("paragraph", contents.sentences[0].text, 1, (1,), {"byline"}),
            ("paragraph", contents.sentences[1].text, 1, (2,), {"doi"}),
            ("heading", abstract_heading, 2, (), {"abstract", "heading"}),
            ("paragraph", contents.sentences[2].text, 2, (3,), {"abstract"}),
        ]
    ):
        candidates.append(
            FrontMatterCandidate(
                candidate_id=f"c{index}",
                source_kind=kind,
                reading_order=index,
                page=1,
                bbox=None,
                region_label=None,
                font_size=None,
                font_bold=None,
                section_id=section_id,
                text_ids=text_ids,
                paragraph_id=text_ids[0] if text_ids else None,
                raw_text=text,
                normalized_text=text.casefold(),
                roles=frozenset(roles),
            )
        )
    resolution = FrontMatterResolution(
        candidates=tuple(candidates),
        blocks=(FrontMatterBlock("record-1", tuple(c.candidate_id for c in candidates), ("c0",)),),
        selected_block_id="record-1",
        selection_method="single_record",
        reason_flags=(),
        allowed_text_ids=frozenset({1, 2, 3}),
        allowed_section_ids=frozenset({1, 2}),
    )
    contents.front_matter_resolution = resolution
    return contents, resolution


@pytest.mark.parametrize("category", ["non_json", "truncated", "schema_invalid"])
@pytest.mark.parametrize("abstract_heading", ["Abstract", "SUMMARY", "Resumo"])
async def test_field_failure_keeps_authors_and_concurrent_references_but_blocks_export(
    monkeypatch, category, abstract_heading
):
    contents, resolution = _contents(abstract_heading)
    settings = GlobalSettings()
    client = LLMClient(settings=settings)
    forbidden_service = AsyncMock(side_effect=AssertionError("Unexpected service call"))
    monkeypatch.setattr(client._backend, "create", forbidden_service)
    refs_started = asyncio.Event()
    refs_completed = asyncio.Event()

    async def invalid_title(*args, **kwargs):
        await refs_started.wait()
        raise StructuredResponseError(category)

    monkeypatch.setattr(client, "extract_title_keywords", AsyncMock(side_effect=invalid_title))
    monkeypatch.setattr(
        client,
        "extract_authors",
        AsyncMock(
            return_value=AuthorsLLM(
                authors=[AuthorLLM(given="Mara", family="Quill")],
            )
        ),
    )
    monkeypatch.setattr(
        client,
        "extract_paper_classification",
        AsyncMock(return_value=PaperClassificationLLM(paper_type="empirical")),
    )
    extractor = MetadataExtractor(
        contents, llm_client=client, settings=settings, front_matter_resolution=resolution
    )
    monkeypatch.setattr(
        extractor.core,
        "_classify_paper",
        AsyncMock(
            return_value=("empirical", "Natural Sciences", "Biological Sciences", 0.9, 0.9),
        ),
    )

    async def references(ref_df):
        assert ref_df["text_id"].tolist() == [5]
        refs_started.set()
        await asyncio.sleep(0.01)
        refs_completed.set()
        return [
            PaperReference(
                bib_id=1,
                title="Earlier study",
                first_page="1",
                volume="3",
                authors="Brook E.",
                year=2020,
                container="Journal of Plants",
                text_id=5,
            )
        ]

    monkeypatch.setattr(extractor, "_extract_references", references)

    metadata = await asyncio.wait_for(extractor.extract_all_metadata(), timeout=2)

    assert refs_completed.is_set()
    assert [(author.given, author.family) for author in metadata.authors] == [("Mara", "Quill")]
    assert [reference.title for reference in metadata.references] == ["Earlier study"]
    assert metadata.doi == "10.9999/printed"
    issue = next(
        issue for issue in extractor.validation_issues if issue.code == "VAL_METADATA_FIELD_FAILED"
    )
    assert issue.blocking
    assert f"reason:{category}" in issue.evidence_ids
    assert {"field:title", "field:abstract", "field:keywords"}.issubset(issue.evidence_ids)
    client.extract_title_keywords.assert_awaited_once()
    client.extract_authors.assert_awaited_once()
    forbidden_service.assert_not_awaited()

    _resolve_selected_title(contents, metadata, validation_issue_sink=extractor.validation_issues)
    _finalize_abstract_and_keywords(
        contents,
        metadata,
        resolution=resolution,
        validation_issue_sink=extractor.validation_issues,
    )
    assert metadata.title == "SHADE AND SEEDLING GROWTH"
    assert metadata.abstract == "Shade improved seedling growth in a controlled experiment."

    paper = _build_paper(contents, metadata, "synthetic.pdf", "a" * 64, "record-1")
    paper.validation_issues = extractor.validation_issues
    exported = build_paper_export(paper)
    assert exported.author[0].family == "Quill"
    assert exported.bib[0].title == "Earlier study"
    assert exported.validation is not None
    assert exported.validation.promotable is False
    assert exported.validation.blocking >= 1
    assert any(
        issue.code == "VAL_METADATA_FIELD_FAILED" and issue.blocking
        for issue in exported.validation.issues
    )
