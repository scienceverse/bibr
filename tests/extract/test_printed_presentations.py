"""Presentation links need local byline ownership, never parallel list positions."""

from dataclasses import replace

from bibr.extract.metadata_variants import collect_metadata_variants
from tests.extract.test_metadata_variants import _fixture


def test_title_and_abstract_share_only_their_own_presentation():
    contents, resolution = _fixture()
    variants = collect_metadata_variants(contents, resolution)
    first_title, first_abstract, second_title, second_abstract = variants
    assert (
        first_title.presentation_ids
        == first_abstract.presentation_ids
        == ("record-1-presentation-1",)
    )
    assert (
        second_title.presentation_ids
        == second_abstract.presentation_ids
        == ("record-1-presentation-2",)
    )
    assert first_abstract.byline_source_text_ids == (2,)
    assert second_abstract.byline_source_text_ids == (7,)


def test_repeated_title_retains_links_to_both_printed_presentations():
    contents, resolution = _fixture(second_title="SHADE AND SEEDLING GROWTH")
    variants = collect_metadata_variants(contents, resolution)
    title = next(row for row in variants if row.field == "title")
    assert title.presentation_ids == ("record-1-presentation-1", "record-1-presentation-2")
    assert title.byline_source_text_ids == (2, 7)


def test_missing_first_byline_does_not_shift_second_abstract_to_first_title():
    contents, resolution = _fixture()
    resolution = replace(
        resolution,
        candidates=tuple(
            replace(row, roles=frozenset({"metadata"})) if row.reading_order == 2 else row
            for row in resolution.candidates
        ),
    )
    variants = collect_metadata_variants(contents, resolution)
    first_abstract = next(row for row in variants if row.field == "abstract")
    assert not first_abstract.presentation_ids
    for row in variants:
        if row.presentation_ids:
            assert row.presentation_ids == ("record-1-presentation-2",)


def test_abstract_before_byline_is_preserved_but_unpaired():
    contents, resolution = _fixture()
    resolution = replace(
        resolution,
        candidates=tuple(
            replace(row, reading_order=4.5) if row.reading_order == 2 else row
            for row in resolution.candidates
        ),
    )
    variants = collect_metadata_variants(contents, resolution)
    assert not next(row for row in variants if row.field == "abstract").presentation_ids


def test_conflicting_byline_is_not_dropped_to_make_the_inventory_look_unique():
    contents, resolution = _fixture()
    resolution = replace(
        resolution,
        candidates=tuple(
            replace(row, roles=frozenset({"byline", "affiliation"}))
            if row.reading_order == 3
            else row
            for row in resolution.candidates
        ),
    )
    variants = collect_metadata_variants(contents, resolution)
    assert not next(row for row in variants if row.field == "abstract").presentation_ids


def test_two_abstracts_under_one_title_are_not_paired_by_variant_ordinal():
    from bibr.extract.printed_presentations import link_printed_presentations

    contents, resolution = _fixture()
    variants = collect_metadata_variants(contents, resolution)
    unlinked = [replace(row, presentation_ids=()) for row in variants]
    first_title, first_abstract, second_title, second_abstract = unlinked
    inputs = [(1, first_title), (4, first_abstract), (4.5, second_abstract), (6, second_title)]
    result = link_printed_presentations(inputs, resolution.candidates, "record-1")
    assert all(not row.presentation_ids for _, row in result)
