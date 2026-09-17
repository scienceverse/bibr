"""Printed abstract inventories survive coarse sections and page continuations."""

from dataclasses import replace

import pytest

from bibr.extract.front_matter import resolve_front_matter
from bibr.extract.metadata_variants import collect_metadata_variants
from bibr.structure.pdf_parser import PDFParser


def _parse(pages):
    regions = [
        [
            {
                "index": index,
                "label": label,
                "content": text,
                "bbox_2d": [40, 40 + index * 70, 550, 90 + index * 70],
            }
            for index, (label, text) in enumerate(page)
        ]
        for page in pages
    ]
    parser = PDFParser(regions)
    contents = parser.parse()
    parser.apply_segmentation(
        contents, [[text] for text, _, _, needs, _ in parser._deferred_texts if needs]
    )
    parser.create_content_sections(contents)
    resolution, _ = resolve_front_matter(contents, target_required=False)
    return contents, resolution


def _abstracts(contents, resolution):
    return [
        row for row in collect_metadata_variants(contents, resolution) if row.field == "abstract"
    ]


def _parallel():
    return _parse(
        [
            [
                ("doc_title", "Shade and seedling growth"),
                ("text", "Mara Quill and Elian Brook"),
                ("paragraph_title", "Abstract"),
                ("abstract", "Shade improved seedling growth."),
                ("text", "Keywords: shade; seedlings"),
                ("paragraph_title", "Resumo"),
                ("abstract", "A sombra melhorou o crescimento das mudas."),
                ("text", "Palavras-chave: sombra; mudas"),
                ("paragraph_title", "Introduction"),
                ("text", "This body paragraph must not enter either abstract."),
            ]
        ]
    )


def _continued():
    return _parse(
        [
            [
                ("doc_title", "Shade and seedling growth"),
                ("text", "Mara Quill and Elian Brook"),
                ("abstract", "Resumo: A sombra melhorou o crescimento das mudas."),
                ("text", "Palavras-chave: sombra; mudas"),
                ("text", "Abstract: " + "We measured growth over two seasons. " * 8 + "Shade"),
                ("footnote", "1 Department of Plant Studies."),
                ("footer", "Journal of Seedlings 1"),
            ],
            [
                ("header", "Journal of Seedlings"),
                ("abstract", "improved seedling growth at the university nursery."),
                ("text", "Keywords: shade; seedlings"),
                ("paragraph_title", "Introduction"),
                ("text", "The body discusses a different problem."),
            ],
        ]
    )


def test_separate_printed_headings_bound_abstracts_even_after_section_reassignment():
    contents, resolution = _parallel()
    # Reproduce semantic normalization coalescing both languages and keywords.
    abstract_section = next(row.section_id for row in contents.sections if row.header == "Abstract")
    ids = {key for row in resolution.candidates if row.reading_order >= 2 for key in row.text_ids}
    for sentence in contents.sentences:
        if sentence.text_id in ids:
            sentence.section_id = abstract_section
    resolution = replace(
        resolution,
        candidates=tuple(
            replace(row, section_id=abstract_section)
            if row.text_ids and set(row.text_ids) <= ids
            else row
            for row in resolution.candidates
        ),
    )

    variants = _abstracts(contents, resolution)

    assert [row.text for row in variants] == [
        "Shade improved seedling growth.",
        "A sombra melhorou o crescimento das mudas.",
    ]
    assert all(not row.presentation_ids for row in variants)
    assert set(variants[0].source_text_ids).isdisjoint(variants[1].source_text_ids)


def test_inline_text_label_and_next_page_abstract_region_form_one_complete_variant():
    contents, resolution = _continued()

    variants = _abstracts(contents, resolution)

    assert len(variants) == 2
    assert variants[1].text == (
        "We measured growth over two seasons. " * 8
        + "Shade\nimproved seedling growth at the university nursery."
    )
    assert variants[1].pages == (1, 2)
    assert len(variants[1].source_text_ids) == 2
    assert all(not row.presentation_ids for row in variants)
    assert "Keywords" not in variants[1].text
    assert "Department" not in variants[1].text
    inline = next(
        row
        for row in contents.region_summaries
        if (row.canonical_ocr_content or "").startswith("Abstract:")
    )
    assert len(inline.content) == 200
    assert len(inline.canonical_ocr_content) > 200


@pytest.mark.parametrize("citation_first", [True, False])
def test_citation_boxes_are_boundaries_and_never_abstract_alternatives(citation_first):
    citation = (
        [
            ("paragraph_title", "CITATION"),
            ("abstract", "Quill M. Shade and seedling growth. Plant Journal 2025;1:2."),
        ]
        if citation_first
        else [
            ("text", "Citation: Quill M. Shade and seedling growth. Plant Journal 2025;1:2."),
            ("text", "Received: 1 January 2025"),
        ]
    )
    abstract = [("abstract", "Abstract: Shade improved growth.")]
    contents, resolution = _parse(
        [
            [
                ("doc_title", "Shade and seedling growth"),
                ("text", "Mara Quill and Elian Brook"),
                *(citation + abstract if citation_first else abstract + citation),
                ("text", "Keywords: shade; seedlings"),
            ]
        ]
    )

    variants = _abstracts(contents, resolution)

    assert [row.text for row in variants] == ["Shade improved growth."]


@pytest.mark.parametrize(
    "fault",
    [
        "foreign-owner",
        "partial-text",
        "duplicate-owner",
        "missing-provenance",
        "missing-first-provenance",
        "unexplained-gap",
        "unlabelled-first",
        "duplicate-region",
        "unlabelled-continuation",
        "embedded-keywords",
    ],
)
def test_unsafe_continuation_cannot_leave_a_truncated_or_shifted_inventory(fault):
    contents, resolution = _continued()
    continuation = next(
        row for row in contents.sentences if row.text.startswith("improved seedling")
    )
    if fault == "foreign-owner":
        resolution = replace(
            resolution, allowed_text_ids=resolution.allowed_text_ids - {continuation.text_id}
        )
    elif fault == "partial-text":
        continuation.text += " A sentence from another source."
    elif fault == "missing-provenance":
        continuation.provenance = []
    elif fault == "missing-first-provenance":
        next(row for row in contents.sentences if row.text.startswith("Resumo:")).provenance = []
    elif fault == "duplicate-region":
        contents.region_summaries.append(contents.region_summaries[2])
    elif fault == "unlabelled-continuation":
        next(
            row for row in contents.region_summaries if row.page == 2 and row.index == 1
        ).label = "text"
    elif fault == "embedded-keywords":
        region = next(row for row in contents.region_summaries if row.page == 2 and row.index == 1)
        region.canonical_ocr_content += "\nKeywords: shade"
        continuation.text += "\nKeywords: shade"
        resolution = replace(
            resolution,
            candidates=tuple(
                replace(row, raw_text=row.raw_text + "\nKeywords: shade")
                if continuation.text_id in row.text_ids
                else row
                for row in resolution.candidates
            ),
        )
    elif fault == "duplicate-owner":
        original = next(
            row for row in resolution.candidates if continuation.text_id in row.text_ids
        )
        duplicate = replace(original, candidate_id="duplicate")
        block = resolution.blocks[0]
        resolution = replace(
            resolution,
            candidates=(*resolution.candidates, duplicate),
            blocks=(replace(block, candidate_ids=(*block.candidate_ids, duplicate.candidate_id)),),
        )
    elif fault == "unexplained-gap":
        next(row for row in contents.region_summaries if row.label == "footnote").label = "text"
    else:
        first = next(row for row in contents.region_summaries if row.label == "abstract")
        first.canonical_ocr_content = first.canonical_ocr_content.removeprefix("Resumo: ")
        first.content = first.canonical_ocr_content
        sentence = next(row for row in contents.sentences if row.text.startswith("Resumo: "))
        sentence.text = sentence.text.removeprefix("Resumo: ")
        resolution = replace(
            resolution,
            candidates=tuple(
                replace(row, raw_text=row.raw_text.removeprefix("Resumo: "))
                if sentence.text_id in row.text_ids
                else row
                for row in resolution.candidates
            ),
        )

    assert _abstracts(contents, resolution) == []
