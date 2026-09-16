"""Title grounding preserves an invented printed grammatical error instead of silently correcting it."""

from dataclasses import replace

import pytest

from bibr.extract.core_metadata import ground_title_to_printed_text

from .test_core_metadata_author_guards import _candidate, _resolution

_PRINTED = "ВИВЧЕННЯ СЕЗОННИХ ЗМІН У МАЛЕНЬКУ САДУ БІЛЯ ШКОЛИ"
_REWRITTEN = "ВИВЧЕННЯ СЕЗОННИХ ЗМІН У МАЛЕНЬКОМУ САДУ БІЛЯ ШКОЛИ"


def _titles(*texts):
    return _resolution(
        *(
            _candidate(f"c{index}", text, roles=frozenset({"title"}), source_kind="heading")
            for index, text in enumerate(texts, start=1)
        )
    )


def test_verbatim_title_passes_through_untouched():
    title, issue = ground_title_to_printed_text(_PRINTED, _titles(_PRINTED), _PRINTED)

    assert title == _PRINTED
    assert issue is None


def test_grounding_ignores_case_and_punctuation_differences():
    printed = "Plant Growth, Seasonal Colors, and Garden Design"
    extracted = "Plant Growth Seasonal Colors and Garden Design"
    title, issue = ground_title_to_printed_text(extracted, _titles(printed), printed)

    assert title == extracted
    assert issue is None


def test_rewritten_title_is_replaced_with_the_printed_row():
    title, issue = ground_title_to_printed_text(_REWRITTEN, _titles(_PRINTED), _PRINTED)

    assert title == _PRINTED
    assert issue is not None
    assert issue.code == "VAL_TITLE_REGROUNDED"
    assert not issue.blocking


def test_ungrounded_title_is_recovered_from_unique_printed_title():
    invented = "An Entirely Different Paper About Something Else"
    title, issue = ground_title_to_printed_text(invented, _titles(_PRINTED), _PRINTED)

    assert title == _PRINTED
    assert issue is not None
    assert issue.code == "VAL_TITLE_RECOVERED"
    assert "reason:title_not_printed_verbatim" in issue.evidence_ids


def test_title_assembled_from_two_regions_is_not_truncated():
    # The model legitimately joins a title split across regions. The title
    # region alone scores well against the join, so a loose rule would replace
    # the complete title with its first half.
    main = "Association of Daily Sunlight With Garden Plant Growth"
    subtitle = "A Classroom Study Across Several Different Garden Plots"
    joined = f"{main}: {subtitle}"

    title, issue = ground_title_to_printed_text(
        joined,
        _titles(main, subtitle),
        f"{main}\n{subtitle}",
    )

    assert title == joined
    assert issue is None


def test_short_ungrounded_notice_title_cannot_override_verified_research_title():
    # Short strings are not fuzzy-matched, but a unique source title still
    # outranks an invented notice label that would suppress the real authors.
    title, issue = ground_title_to_printed_text("Erratum", _titles(_PRINTED), _PRINTED)

    assert title == _PRINTED
    assert issue.code == "VAL_TITLE_RECOVERED"


def test_grounding_is_inert_without_a_resolution():
    title, issue = ground_title_to_printed_text(_PRINTED, None, _PRINTED)

    assert title == _PRINTED
    assert issue is None


def test_title_merged_into_an_affiliation_paragraph_is_still_repaired():
    # Printed rows recover line granularity when a title shares its candidate
    # paragraph with a preceding affiliation.
    affiliation = "дослідник кафедри природничих наук прикладного університету"
    merged = _candidate(
        "c1",
        f"{affiliation} {_PRINTED}",
        roles=frozenset({"title"}),
        text_ids=(3, 4),
    )
    resolution = _resolution(merged)

    title, issue = ground_title_to_printed_text(
        _REWRITTEN,
        resolution,
        f"{affiliation} {_PRINTED}",
        printed_rows={3: affiliation, 4: _PRINTED},
    )

    assert title == _PRINTED
    assert issue is not None
    assert issue.code == "VAL_TITLE_REGROUNDED"


def test_printed_rows_outside_the_front_matter_cannot_supply_a_title():
    # Ownership: only rows belonging to a candidate are eligible. A reference
    # entry or a "how to cite" line that happens to carry the title must never
    # become the substitution source.
    resolution = _resolution(_candidate("c1", _PRINTED, roles=frozenset({"title"}), text_ids=(4,)))
    unrelated = "An Entirely Different Paper About Something Else Entirely Here"

    title, issue = ground_title_to_printed_text(
        unrelated,
        resolution,
        _PRINTED,
        printed_rows={4: _PRINTED, 99: unrelated},
    )

    assert title == _PRINTED
    assert issue is not None
    assert issue.code == "VAL_TITLE_RECOVERED"


def test_ambiguous_printed_titles_abstain_instead_of_keeping_an_invention():
    title, issue = ground_title_to_printed_text(
        "A Fabricated Paper About Ocean Temperatures",
        _titles("The Seasonal Behavior of Forest Birds", "Le comportement saisonnier des oiseaux"),
        "",
    )

    assert title == ""
    assert issue.code == "VAL_TITLE_UNGROUNDED"
    assert "reason:title_recovery_ambiguous" in issue.evidence_ids


def test_unselected_record_title_cannot_validate_or_replace_the_selected_title():
    selected, foreign = _titles(
        "The Seasonal Behavior of Forest Birds", "Ocean Temperatures"
    ).candidates
    resolution = _resolution(selected, foreign, selected_ids=(selected.candidate_id,))

    title, issue = ground_title_to_printed_text(foreign.raw_text, resolution, foreign.raw_text)

    assert title == selected.raw_text
    assert issue.code == "VAL_TITLE_RECOVERED"
    assert selected.candidate_id in issue.evidence_ids


@pytest.mark.parametrize("source_kind", ["heading", "paragraph"])
def test_candidate_must_match_the_owned_printed_source_before_recovery(source_kind):
    candidate = replace(_titles(_PRINTED).candidates[0], source_kind=source_kind, text_ids=(1,))
    title, issue = ground_title_to_printed_text(
        "A Fabricated Paper About Ocean Temperatures",
        _resolution(candidate),
        _PRINTED,
        printed_rows={1: "Another source paragraph"},
        printed_sections={1: "Another source heading"},
    )

    assert title == ""
    assert "reason:title_recovery_unverified" in issue.evidence_ids


def test_unsafe_first_title_does_not_make_a_later_language_the_unique_original():
    first, second = _titles(_PRINTED, "The Seasonal Behavior of Forest Birds").candidates
    first = replace(first, roles=first.roles | {"byline"})

    title, issue = ground_title_to_printed_text(
        "A Fabricated Paper About Ocean Temperatures", _resolution(first, second), ""
    )

    assert title == ""
    assert "reason:title_recovery_unverified" in issue.evidence_ids


def test_citation_sidebar_is_not_title_grounding_evidence():
    printed = "The Seasonal Behavior of Forest Birds"
    cited = "Ocean Temperatures and Marine Ecosystems"
    title_candidate = _titles(printed).candidates[0]
    citation = _candidate("c2", cited, roles=frozenset({"metadata"}))

    title, issue = ground_title_to_printed_text(
        "Ocean Temperature and Marine Ecosystems", _resolution(title_candidate, citation), cited
    )

    assert title == printed
    assert issue.code == "VAL_TITLE_RECOVERED"


@pytest.mark.parametrize(
    "label",
    ["Introduction", "Original Research", "CITATION", "APRESENTAÇÃO E ANÁLISE DOS RESULTADOS"],
)
def test_body_and_furniture_labels_cannot_recover_a_title(label):
    title, issue = ground_title_to_printed_text(
        "A Fabricated Paper About Ocean Temperatures", _titles(label), label
    )

    assert title == ""
    assert "reason:title_recovery_unverified" in issue.evidence_ids
