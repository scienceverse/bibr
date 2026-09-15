"""Owned printed variants survive extraction and section normalization."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from bibr.config import snapshot_settings
from bibr.extract.core_metadata import CoreMetadataExtractor
from bibr.extract.metadata_variants import collect_metadata_variants
from bibr.paper_contents import CanonicalSection, PaperSentence
from bibr.schemas import AuthorLLM, CoreMetadataLLM
from tests.extract.test_metadata_variants import _fixture

ORIGINAL_TITLE = "SOMBRA E CRESCIMENTO DE MUDAS"
ORIGINAL_ABSTRACT = "A sombra melhorou o crescimento das mudas."
ENGLISH_TITLE = "SHADE AND SEEDLING GROWTH"
ENGLISH_ABSTRACT = "Shade improved seedling growth."


def _printed_pair():
    contents, resolution = _fixture(
        second_title=ENGLISH_TITLE,
        second_abstract=ENGLISH_ABSTRACT,
    )
    replacements = {
        0: ORIGINAL_TITLE,
        3: "Resumo",
        4: ORIGINAL_ABSTRACT,
        8: "Abstract",
    }
    candidates = list(resolution.candidates)
    for index, text in replacements.items():
        candidates[index] = replace(
            candidates[index], raw_text=text, normalized_text=text.casefold()
        )
    contents.sections[0].header = ORIGINAL_TITLE
    contents.sections[1].header = "Resumo"
    contents.sections[3].header = "Abstract"
    next(
        sentence for sentence in contents.sentences if sentence.text_id == 5
    ).text = ORIGINAL_ABSTRACT
    resolution = replace(resolution, candidates=tuple(candidates))
    contents.front_matter_resolution = resolution
    contents.metadata_variants = collect_metadata_variants(contents, resolution)
    return contents, resolution


class _FakeLlm:
    def __init__(self, *, abstract=ENGLISH_ABSTRACT):
        self.result = CoreMetadataLLM(
            title=ENGLISH_TITLE,
            abstract=abstract,
            authors=[
                AuthorLLM(given="Mara", family="Quill"),
                AuthorLLM(given="Elian", family="Brook"),
            ],
            keywords=["shade"],
        )
        self.extract_core_metadata = AsyncMock(return_value=self.result)


def _extractor(contents, resolution, monkeypatch, *, llm=None):
    llm = llm or _FakeLlm()
    extractor = CoreMetadataExtractor(
        contents,
        llm_client=llm,
        settings=snapshot_settings(),
        front_matter_resolution=resolution,
    )
    classifier = AsyncMock(
        return_value=("empirical", "Natural Sciences", "Biological Sciences", 0.9, 0.9)
    )
    monkeypatch.setattr(extractor, "_classify_paper", classifier)
    return extractor, llm, classifier


async def test_first_printed_primary_overrides_later_english_model_choice_before_classification(
    monkeypatch,
):
    contents, resolution = _printed_pair()
    extractor, llm, classifier = _extractor(contents, resolution, monkeypatch)

    metadata = await extractor.extract()

    llm.extract_core_metadata.assert_awaited_once()
    assert metadata.title == ORIGINAL_TITLE
    assert metadata.abstract == ORIGINAL_ABSTRACT
    assert classifier.await_args.args[:2] == (ORIGINAL_TITLE, ORIGINAL_ABSTRACT)
    assert llm.result.title == ENGLISH_TITLE
    assert llm.result.abstract == ENGLISH_ABSTRACT


def _with_original_label(contents, resolution, field):
    candidates = tuple(
        replace(candidate, reading_order=candidate.reading_order * 10)
        for candidate in resolution.candidates
    )
    label = f"Original {field}:"
    marker = replace(
        candidates[7],
        candidate_id="original-label",
        reading_order=55 if field == "title" else 85,
        text_ids=(11,),
        paragraph_id=11,
        raw_text=label,
        normalized_text=label.casefold(),
        roles=frozenset({"metadata"}),
    )
    contents.sentences.append(PaperSentence(11, label, 3, 11, page_number=2))
    block = replace(
        resolution.blocks[0],
        candidate_ids=resolution.blocks[0].candidate_ids + (marker.candidate_id,),
    )
    resolution = replace(
        resolution,
        candidates=candidates + (marker,),
        blocks=(block,),
        allowed_text_ids=resolution.allowed_text_ids | {11},
    )
    contents.front_matter_resolution = resolution
    contents.metadata_variants = collect_metadata_variants(contents, resolution)
    assert sum(variant.field == field for variant in contents.metadata_variants) == 2
    return contents, resolution


@pytest.mark.parametrize("field", ["title", "abstract"])
async def test_explicit_original_label_selects_the_entire_presentation(field, monkeypatch):
    contents, resolution = _with_original_label(*_printed_pair(), field)
    extractor, _, classifier = _extractor(contents, resolution, monkeypatch)

    metadata = await extractor.extract()

    expected_title = ENGLISH_TITLE
    expected_abstract = ENGLISH_ABSTRACT
    assert metadata.title == expected_title
    assert metadata.abstract == expected_abstract
    assert classifier.await_args.args[:2] == (expected_title, expected_abstract)


async def test_byline_preference_preserves_explicit_original_title_selection(monkeypatch):
    from bibr.pipeline.stages.post_parse import _prefer_byline_adjacent_title

    contents, resolution = _with_original_label(*_printed_pair(), "title")
    extractor, _, _ = _extractor(contents, resolution, monkeypatch)
    metadata = await extractor.extract()
    settings = snapshot_settings()
    settings.pipeline.title_prefer_byline_adjacent = True

    changed = _prefer_byline_adjacent_title(
        contents,
        metadata,
        validation_issue_sink=[],
        settings=settings,
    )

    assert changed is False
    assert metadata.title == ENGLISH_TITLE


@pytest.mark.parametrize(
    "gate", ["single", "unmarked", "two-primaries", "other-record", "split-records"]
)
async def test_complete_source_pair_is_authoritative_over_individual_primary_flags(
    gate, monkeypatch
):
    contents, resolution = _printed_pair()
    variants = contents.metadata_variants
    if gate == "single":
        variants = [variant for variant in variants if variant.is_primary]
    elif gate == "unmarked":
        variants = [replace(variant, is_primary=False) for variant in variants]
    elif gate == "two-primaries":
        variants = [replace(variant, is_primary=True) for variant in variants]
    elif gate == "other-record":
        variants = [replace(variant, record_id="another-paper") for variant in variants]
    else:
        variants = [
            replace(variant, record_id="another-paper") if not variant.is_primary else variant
            for variant in variants
        ]
    contents.metadata_variants = variants
    extractor, _, classifier = _extractor(contents, resolution, monkeypatch)

    metadata = await extractor.extract()

    expected = (
        (ENGLISH_TITLE, ENGLISH_ABSTRACT)
        if gate == "other-record"
        else (ORIGINAL_TITLE, ORIGINAL_ABSTRACT)
    )
    assert (metadata.title, metadata.abstract) == expected
    assert classifier.await_args.args[:2] == expected


async def test_other_record_primary_cannot_replace_selected_record_primary(monkeypatch):
    contents, resolution = _printed_pair()
    other = [
        replace(variant, record_id="another-paper", text=f"Unowned {variant.field}")
        for variant in contents.metadata_variants
    ]
    contents.metadata_variants = other + contents.metadata_variants
    extractor, _, _ = _extractor(contents, resolution, monkeypatch)

    metadata = await extractor.extract()

    assert metadata.title == ORIGINAL_TITLE
    assert metadata.abstract == ORIGINAL_ABSTRACT


async def test_positive_printed_abstract_clears_explicit_model_absence(monkeypatch):
    contents, resolution = _printed_pair()
    llm = _FakeLlm(abstract=None)
    assert llm.result._abstract_explicitly_absent
    extractor, _, _ = _extractor(contents, resolution, monkeypatch, llm=llm)

    metadata = await extractor.extract()

    assert metadata.abstract == ORIGINAL_ABSTRACT
    assert metadata._abstract_explicitly_absent is False
    assert llm.result._abstract_explicitly_absent is True


async def test_notice_guard_still_suppresses_printed_abstract_after_variant_selection(monkeypatch):
    contents, resolution = _printed_pair()
    notice_title = "Correction: Shade and seedling growth"
    contents.metadata_variants = [
        replace(variant, text=notice_title)
        if variant.field == "title" and variant.is_primary
        else variant
        for variant in contents.metadata_variants
    ]
    extractor, _, _ = _extractor(contents, resolution, monkeypatch)

    metadata = await extractor.extract()

    assert metadata.title == notice_title
    assert metadata.paper_type == "corrigendum"
    assert metadata.abstract == ""
    assert metadata.authors == []
    assert metadata.keywords == []


async def test_post_parse_captures_variants_before_section_normalization(monkeypatch):
    from bibr.extract import front_matter
    from bibr.pipeline.stages import post_parse as stage

    contents, resolution = _printed_pair()
    contents.metadata_variants = []
    expected = collect_metadata_variants(contents, resolution)
    events = []
    llm = _FakeLlm()
    settings = snapshot_settings()
    settings.EQUATION_EXTRACTION = False

    async def no_op(*_args, **_kwargs):
        return None

    async def classify(*_args, **_kwargs):
        events.append("classify")

    async def normalize(actual_contents, *_args, **_kwargs):
        assert actual_contents.metadata_variants == expected
        events.append("normalize")
        for section in actual_contents.sections:
            if section.section_type == CanonicalSection.ABSTRACT:
                section.header_is_synthetic = True
                section.header = "Inferred abstract"
                section.section_type = CanonicalSection.UNKNOWN
        assert not any(
            variant.field == "abstract"
            for variant in collect_metadata_variants(actual_contents, resolution)
        )

    def resolve(actual_contents, **_kwargs):
        assert actual_contents is contents
        events.append("resolve")
        return resolution, ()

    monkeypatch.setattr(stage, "_classify_sections", classify)
    monkeypatch.setattr(front_matter, "resolve_front_matter", resolve)
    monkeypatch.setattr(stage, "_normalize_section_structure", normalize)
    monkeypatch.setattr(stage, "_link_citations", no_op)
    monkeypatch.setattr("bibr.extract.research_integrity.extract_structured_integrity", no_op)
    monkeypatch.setattr(
        CoreMetadataExtractor,
        "_classify_paper",
        AsyncMock(return_value=("empirical", "Natural Sciences", "Biological Sciences", 0.9, 0.9)),
    )

    paper = await stage.post_parse(
        contents,
        "translated-source.pdf",
        "synthetic-source-hash",
        llm_client=llm,
        ref_parse_strategy="off",
        extract_equations=False,
        settings=settings,
    )

    assert events == ["classify", "resolve", "normalize"]
    assert paper.contents.metadata_variants == expected
    assert paper.metadata.title == ORIGINAL_TITLE
    assert paper.metadata.abstract == ORIGINAL_ABSTRACT
