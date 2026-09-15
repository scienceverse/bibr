"""Complete presentation choice uses source ownership rather than scalar ordering."""

from dataclasses import replace

from bibr.extract.metadata_variants import collect_metadata_variants
from bibr.extract.primary_presentation import (
    presentation_author_context,
    select_printed_presentation,
)
from tests.extract.test_metadata_variants import _fixture
from tests.extract.test_metadata_variants_integration import (
    ENGLISH_ABSTRACT,
    ENGLISH_TITLE,
    ORIGINAL_ABSTRACT,
    ORIGINAL_TITLE,
    _extractor,
    _printed_pair,
)


def test_repeated_field_text_preserves_the_primary_physical_byline():
    contents, resolution = _fixture(second_title="SHADE AND SEEDLING GROWTH")
    variants = collect_metadata_variants(contents, resolution)
    choice = select_printed_presentation(variants, resolution)
    first, second = choice.presentations
    assert first.title_variant_id == second.title_variant_id
    assert first.byline_source_text_ids == (2,)
    assert second.byline_source_text_ids == (7,)
    assert choice.selected_presentation_id == first.presentation_id
    assert presentation_author_context(first, resolution).count("Mara Quill") == 1


async def test_missing_first_byline_selects_the_first_complete_pair_without_shifting(monkeypatch):
    contents, resolution = _printed_pair()
    resolution = replace(
        resolution,
        candidates=tuple(
            replace(row, roles=frozenset({"metadata"})) if row.reading_order == 2 else row
            for row in resolution.candidates
        ),
    )
    contents.metadata_variants = collect_metadata_variants(contents, resolution)
    extractor, _, _ = _extractor(contents, resolution, monkeypatch)
    result = await extractor.extract()
    assert (result.title, result.abstract) == (ENGLISH_TITLE, ENGLISH_ABSTRACT)
    assert contents.presentation_selection.selected_presentation_id == "record-1-presentation-2"


async def test_unpaired_variants_never_independently_override_a_model_pair(monkeypatch):
    contents, resolution = _printed_pair()
    contents.metadata_variants = [
        replace(row, presentation_ids=()) for row in contents.metadata_variants
    ]
    extractor, _, _ = _extractor(contents, resolution, monkeypatch)
    result = await extractor.extract()
    assert (result.title, result.abstract) == (ENGLISH_TITLE, ENGLISH_ABSTRACT)
    assert contents.presentation_selection.selected_presentation_id is None
    assert any(
        issue.code == "VAL_PRIMARY_PRESENTATION_UNRESOLVED" for issue in extractor.validation_issues
    )


async def test_author_call_receives_only_selected_presentation_byline(monkeypatch):
    contents, resolution = _printed_pair()
    extractor, client, _ = _extractor(contents, resolution, monkeypatch)
    result = await extractor.extract()
    context = client.extract_core_metadata.await_args.kwargs["authors_text"]
    assert context.count("Mara Quill") == 1
    assert ORIGINAL_ABSTRACT not in context and ENGLISH_ABSTRACT not in context
    assert (result.title, result.abstract) == (ORIGINAL_TITLE, ORIGINAL_ABSTRACT)


def test_foreign_byline_and_duplicate_variant_identity_cannot_establish_presentation():
    contents, resolution = _printed_pair()
    variants = [replace(row, byline_source_text_ids=(999,)) for row in contents.metadata_variants]
    assert not select_printed_presentation(variants, resolution).presentations
    variants = [replace(row, variant_id="same") for row in contents.metadata_variants]
    assert select_printed_presentation(variants, resolution).reason == "variant_identity_ambiguous"


def test_printed_inline_abstract_survives_synthetic_section_and_affiliation_vocabulary():
    from bibr.extract.front_matter import resolve_front_matter
    from tests.extract.test_title_source import _source

    contents = _source()
    text = "Resumo: O estudo investigou infecções em animais no hospital universitário."
    contents.sentences[2].text = contents.region_summaries[3].content = text
    resolution, _ = resolve_front_matter(contents, target_required=False)
    variants = collect_metadata_variants(contents, resolution)
    abstracts = [row for row in variants if row.field == "abstract"]
    assert len(abstracts) == 1
    assert abstracts[0].text == text.removeprefix("Resumo: ")
    assert abstracts[0].source_text_ids == (3,)
    # A stacked title's language relationship remains unproven.
    assert not abstracts[0].presentation_ids


def test_partial_region_provenance_cannot_create_an_abstract_variant():
    from bibr.extract.front_matter import resolve_front_matter
    from tests.extract.test_title_source import _source

    contents = _source()
    contents.region_summaries[3].content += " An unassigned continuation."
    resolution, _ = resolve_front_matter(contents, target_required=False)
    assert not any(
        row.field == "abstract" for row in collect_metadata_variants(contents, resolution)
    )


def test_parser_retains_full_labelled_abstract_beyond_diagnostic_preview():
    from bibr.extract.front_matter import resolve_front_matter
    from bibr.structure.pdf_parser import PDFParser
    from tests.extract.test_title_source import _source

    source = _source()
    abstract = "Resumo: " + "O estudo investigou animais no hospital universitário. " * 12
    regions = [
        {
            "index": region.index,
            "label": region.label,
            "content": abstract if region.label == "abstract" else region.content,
            "bbox_2d": list(region.bbox),
        }
        for region in source.region_summaries
    ]
    parser = PDFParser([regions])
    contents = parser.parse()
    parser.apply_segmentation(
        contents, [[text] for text, _, _, needs, _ in parser._deferred_texts if needs]
    )
    parser.create_content_sections(contents)
    resolution, _ = resolve_front_matter(contents, target_required=False)
    region = next(row for row in contents.region_summaries if row.label == "abstract")
    assert len(region.content) == 200
    versions = collect_metadata_variants(contents, resolution)
    captured = next(row for row in versions if row.field == "abstract")
    assert captured.text == abstract.removeprefix("Resumo: ").strip()

    # Retained canonical text cannot authorize a sentence borrowed from another region.
    contents.sentences[-1].text += " Neighboring content."
    assert not any(
        row.field == "abstract" for row in collect_metadata_variants(contents, resolution)
    )
