"""Title grounding preserves an invented printed grammatical error instead of silently correcting it."""

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


def test_ungrounded_title_is_reported_but_not_guessed_at():
    # Nothing on the page resembles it, so there is no printed row to prefer.
    # Report it and leave the extraction alone rather than substituting a
    # different title.
    invented = "An Entirely Different Paper About Something Else"
    title, issue = ground_title_to_printed_text(invented, _titles(_PRINTED), _PRINTED)

    assert title == invented
    assert issue is not None
    assert issue.code == "VAL_TITLE_UNGROUNDED"


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


def test_short_titles_are_left_alone():
    # Short strings hit high similarity ratios by accident.
    title, issue = ground_title_to_printed_text("Erratum", _titles(_PRINTED), _PRINTED)

    assert title == "Erratum"
    assert issue is None


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

    assert title == unrelated
    assert issue is not None
    assert issue.code == "VAL_TITLE_UNGROUNDED"
