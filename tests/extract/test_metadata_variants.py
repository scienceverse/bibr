"""Printed-language preservation without changing scalar extraction contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from bibr.extract.front_matter import FrontMatterBlock, FrontMatterCandidate, FrontMatterResolution
from bibr.extract.metadata_variants import collect_metadata_variants
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence


def _fixture(
    *, second_title="SOMBRA E CRESCIMENTO", second_abstract="A sombra melhorou o crescimento."
):
    candidates = []
    sections = []
    sentences = []
    for ordinal, (title, abstract, label) in enumerate(
        [
            ("SHADE AND SEEDLING GROWTH", "Shade improved growth.", "Abstract"),
            (second_title, second_abstract, "Resumo"),
        ]
    ):
        title_sid, abstract_sid = ordinal * 2 + 1, ordinal * 2 + 2
        sections.extend(
            [
                PaperSection(title_sid, title, 1, None, CanonicalSection.TITLE),
                PaperSection(abstract_sid, label, 2, title_sid, CanonicalSection.ABSTRACT),
            ]
        )
        for kind, text, roles, section_id, region_label in [
            ("heading", title, {"title", "heading"}, title_sid, "doc_title"),
            ("paragraph", "Mara Quill and Elian Brook", {"byline"}, title_sid, "text"),
            ("paragraph", "https://doi.org/10.9999/shade", {"doi"}, title_sid, "text"),
            ("heading", label, {"abstract", "heading"}, abstract_sid, "paragraph_title"),
            ("paragraph", abstract, {"abstract"}, abstract_sid, "abstract"),
        ]:
            index = len(candidates) + 1
            candidates.append(
                FrontMatterCandidate(
                    candidate_id=f"candidate-{index}",
                    source_kind=kind,
                    reading_order=index,
                    page=ordinal + 1,
                    bbox=None,
                    region_label=region_label,
                    font_size=None,
                    font_bold=None,
                    section_id=section_id,
                    text_ids=(index,) if kind == "paragraph" else (),
                    paragraph_id=index,
                    raw_text=text,
                    normalized_text=text.casefold(),
                    roles=frozenset(roles),
                )
            )
            if kind == "paragraph":
                sentences.append(
                    PaperSentence(index, text, section_id, index, page_number=ordinal + 1)
                )
    contents = PaperContents(sentences, sections, [], [], {})
    block = FrontMatterBlock(
        "record-1",
        tuple(candidate.candidate_id for candidate in candidates),
        ("candidate-1", "candidate-6"),
    )
    resolution = FrontMatterResolution(
        tuple(candidates),
        (block,),
        "record-1",
        "shared_identity",
        (),
        frozenset(sentence.text_id for sentence in sentences),
        frozenset(range(1, 5)),
    )
    return contents, resolution


def _field(variants, field):
    return [variant for variant in variants if variant.field == field]


def _append_third_variant(contents, resolution):
    new_candidates = []
    for candidate in resolution.candidates[5:10]:
        index = candidate.reading_order + 5
        text = candidate.raw_text
        if candidate.reading_order == 6:
            text = "OMBRA E CRESCITA DELLE PIANTE"
        elif candidate.reading_order == 10:
            text = "L'ombra ha migliorato la crescita."
        new_candidates.append(
            replace(
                candidate,
                candidate_id=f"candidate-{index}",
                reading_order=index,
                section_id=candidate.section_id + 2,
                text_ids=tuple(value + 5 for value in candidate.text_ids),
                raw_text=text,
                normalized_text=text.casefold(),
                page=3,
            )
        )
        if candidate.source_kind == "paragraph":
            contents.sentences.append(
                PaperSentence(index, text, candidate.section_id + 2, index, page_number=3)
            )
    contents.sections.extend(
        [
            PaperSection(5, "OMBRA E CRESCITA DELLE PIANTE", 1, None, CanonicalSection.TITLE),
            PaperSection(6, "Abstract", 2, 5, CanonicalSection.ABSTRACT),
        ]
    )
    block = replace(
        resolution.blocks[0],
        candidate_ids=resolution.blocks[0].candidate_ids
        + tuple(candidate.candidate_id for candidate in new_candidates),
        title_candidate_ids=resolution.blocks[0].title_candidate_ids + ("candidate-11",),
    )
    return replace(
        resolution,
        blocks=(block,),
        candidates=resolution.candidates + tuple(new_candidates),
        allowed_text_ids=resolution.allowed_text_ids | {12, 13, 15},
        allowed_section_ids=resolution.allowed_section_ids | {5, 6},
    )


def test_complete_printed_variants_preserve_first_printed_primary_and_provenance():
    contents, resolution = _fixture()

    variants = collect_metadata_variants(contents, resolution)

    titles = _field(variants, "title")
    abstracts = _field(variants, "abstract")
    assert [title.text for title in titles] == ["SHADE AND SEEDLING GROWTH", "SOMBRA E CRESCIMENTO"]
    assert [abstract.text for abstract in abstracts] == [
        "Shade improved growth.",
        "A sombra melhorou o crescimento.",
    ]
    assert [title.is_primary for title in titles] == [True, False]
    assert [abstract.is_primary for abstract in abstracts] == [True, False]
    assert abstracts[0].source_text_ids == (5,)
    assert abstracts[0].source_section_ids == (2,)
    assert abstracts[0].pages == (1,)
    assert all(variant.language is None and variant.record_id == "record-1" for variant in variants)
    assert collect_metadata_variants(contents, resolution) == variants
    with pytest.raises(FrozenInstanceError):
        variants[0].text = "changed"


def test_repeated_text_deduplicates_but_retains_all_source_provenance():
    contents, resolution = _fixture(
        second_title="SHADE AND SEEDLING GROWTH", second_abstract="Shade improved growth."
    )

    variants = collect_metadata_variants(contents, resolution)

    assert len(_field(variants, "title")) == 1
    abstract = _field(variants, "abstract")[0]
    assert abstract.source_text_ids == (5, 10)
    assert abstract.source_section_ids == (2, 4)
    assert abstract.pages == (1, 2)


def test_printed_structured_abstract_children_stay_inside_one_variant():
    contents, resolution = _fixture()
    resolution = replace(
        resolution,
        candidates=tuple(
            replace(candidate, reading_order=candidate.reading_order * 10)
            for candidate in resolution.candidates
        ),
    )
    contents.sections.append(PaperSection(5, "Results:", 3, 2, CanonicalSection.ABSTRACT))
    contents.sentences.append(
        PaperSentence(11, "The effect persisted over two seasons.", 5, 11, page_number=2)
    )
    base = resolution.candidates[4]
    heading = replace(
        base,
        candidate_id="structured-heading",
        source_kind="heading",
        reading_order=51,
        section_id=5,
        text_ids=(),
        raw_text="Results:",
        roles=frozenset({"abstract", "heading"}),
        region_label="paragraph_title",
        page=2,
    )
    paragraph = replace(
        base,
        candidate_id="structured-paragraph",
        reading_order=52,
        section_id=5,
        text_ids=(11,),
        raw_text="The effect persisted over two seasons.",
        page=2,
    )
    block = replace(
        resolution.blocks[0],
        candidate_ids=resolution.blocks[0].candidate_ids
        + (heading.candidate_id, paragraph.candidate_id),
    )
    resolution = replace(
        resolution,
        candidates=resolution.candidates + (heading, paragraph),
        blocks=(block,),
        allowed_text_ids=resolution.allowed_text_ids | {11},
        allowed_section_ids=resolution.allowed_section_ids | {5},
    )

    abstracts = _field(collect_metadata_variants(contents, resolution), "abstract")

    assert len(abstracts) == 2
    assert (
        abstracts[0].text
        == "Shade improved growth.\nResults:\nThe effect persisted over two seasons."
    )
    assert abstracts[0].source_text_ids == (5, 11)
    assert abstracts[0].pages == (1, 2)


@pytest.mark.parametrize(
    "problem", ["synthetic-heading", "body-label", "unowned-row", "unsafe-role"]
)
def test_ambiguous_abstract_evidence_does_not_become_a_variant(problem):
    contents, resolution = _fixture()
    if problem == "synthetic-heading":
        contents.sections[1].header_is_synthetic = True
    elif problem == "body-label":
        candidates = list(resolution.candidates)
        candidates[4] = replace(candidates[4], region_label="text")
        resolution = replace(resolution, candidates=tuple(candidates))
    elif problem == "unowned-row":
        contents.sentences.append(
            PaperSentence(99, "Another record's abstract.", 2, 99, page_number=1)
        )
    else:
        candidates = list(resolution.candidates)
        candidates[4] = replace(candidates[4], roles=frozenset({"abstract", "byline"}))
        resolution = replace(resolution, candidates=tuple(candidates))

    abstracts = _field(collect_metadata_variants(contents, resolution), "abstract")

    # Skipping an unsupported or inferred earlier variant cannot establish
    # that a later supported language should become the primary abstract.
    assert abstracts == []


def test_unselected_record_is_never_captured():
    contents, resolution = _fixture()
    first = replace(
        resolution.blocks[0],
        candidate_ids=tuple(candidate.candidate_id for candidate in resolution.candidates[:5]),
    )
    second = FrontMatterBlock(
        "record-2",
        tuple(candidate.candidate_id for candidate in resolution.candidates[5:]),
        ("candidate-6",),
    )
    resolution = replace(
        resolution,
        blocks=(first, second),
        allowed_text_ids=frozenset({2, 3, 5}),
        allowed_section_ids=frozenset({1, 2}),
    )

    variants = collect_metadata_variants(contents, resolution)

    assert len(variants) == 2
    assert all(variant.pages == (1,) for variant in variants)
    assert all(
        "SOMBRA" not in variant.text and "sombra" not in variant.text for variant in variants
    )


def test_abstention_or_missing_resolution_captures_nothing():
    contents, resolution = _fixture()

    assert collect_metadata_variants(contents, None) == []
    assert collect_metadata_variants(contents, replace(resolution, selected_block_id=None)) == []


def test_adjacent_title_fragments_without_independent_anatomy_are_not_two_variants():
    contents, resolution = _fixture()
    resolution = replace(
        resolution,
        candidates=tuple(
            replace(candidate, reading_order=candidate.reading_order * 10)
            for candidate in resolution.candidates
        ),
    )
    title = resolution.candidates[0]
    fragment = replace(
        title, candidate_id="fragment", reading_order=15, raw_text="ACROSS DIFFERENT CLIMATES"
    )
    block = replace(
        resolution.blocks[0],
        candidate_ids=resolution.blocks[0].candidate_ids + (fragment.candidate_id,),
    )
    resolution = replace(
        resolution, candidates=resolution.candidates + (fragment,), blocks=(block,)
    )

    titles = _field(collect_metadata_variants(contents, resolution), "title")

    # Neither incomplete part may override the complete model title, even if
    # a later complete translated title supplies another potential variant.
    assert titles == []


def test_unsupported_original_title_cannot_promote_two_later_translations():
    contents, resolution = _fixture()
    resolution = _append_third_variant(contents, resolution)
    candidates = list(resolution.candidates)
    candidates[0] = replace(candidates[0], roles=frozenset({"title", "byline"}))
    resolution = replace(resolution, candidates=tuple(candidates))

    variants = collect_metadata_variants(contents, resolution)

    assert _field(variants, "title") == []
    assert len(_field(variants, "abstract")) == 3


def test_unsupported_original_abstract_cannot_promote_two_later_translations():
    contents, resolution = _fixture()
    resolution = _append_third_variant(contents, resolution)
    candidates = list(resolution.candidates)
    candidates[4] = replace(candidates[4], region_label="text")
    resolution = replace(resolution, candidates=tuple(candidates))

    variants = collect_metadata_variants(contents, resolution)

    assert _field(variants, "abstract") == []
    assert len(_field(variants, "title")) == 3


def test_unrecognized_original_abstract_heading_cannot_promote_later_languages():
    contents, resolution = _fixture()
    resolution = _append_third_variant(contents, resolution)
    contents.sections[1].header = "Summary"
    candidates = list(resolution.candidates)
    candidates[3] = replace(candidates[3], raw_text="Summary")
    resolution = replace(resolution, candidates=tuple(candidates))

    variants = collect_metadata_variants(contents, resolution)

    assert _field(variants, "abstract") == []
