"""Title grounding keeps the printed title when a model silently corrects or truncates it."""

import pytest

from bibr.extract.core_metadata import ground_title_to_printed_text
from bibr.schemas import AuthorLLM, CoreMetadataLLM

from .test_core_metadata_author_guards import _candidate, _extractor, _resolution

_PRINTED = "ВИВЧЕННЯ СЕЗОННИХ ЗМІН У МАЛЕНЬКУ САДУ БІЛЯ ШКОЛИ"
_REWRITTEN = "ВИВЧЕННЯ СЕЗОННИХ ЗМІН У МАЛЕНЬКОМУ САДУ БІЛЯ ШКОЛИ"
_PARENTHESIZED = "(Rural) Clinics as layered civic organizations"
_UNPARENTHESIZED = "Clinics as layered civic organizations"


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


def test_dropped_leading_parenthetical_is_restored():
    # A model reads "(Rural)" as an annotation and returns the rest of the
    # title. The rest is still printed verbatim, so the verbatim test alone
    # would accept the truncated title.
    title, issue = ground_title_to_printed_text(
        _UNPARENTHESIZED, _titles(_PARENTHESIZED), _PARENTHESIZED
    )

    assert title == _PARENTHESIZED
    assert issue is not None
    assert issue.code == "VAL_TITLE_REGROUNDED"
    assert issue.evidence_ids == ("c1", "reason:title_leading_parenthetical_dropped")
    assert not issue.blocking


def test_printed_leading_parenthetical_passes_through_untouched():
    title, issue = ground_title_to_printed_text(
        _PARENTHESIZED, _titles(_PARENTHESIZED), _PARENTHESIZED
    )

    assert title == _PARENTHESIZED
    assert issue is None


def test_restored_parenthetical_keeps_a_subtitle_printed_on_another_row():
    subtitle = "A Survey of Twelve Districts"

    title, issue = ground_title_to_printed_text(
        f"{_UNPARENTHESIZED}: {subtitle}",
        _titles(_PARENTHESIZED, subtitle),
        f"{_PARENTHESIZED}\n{subtitle}",
    )

    assert title == f"{_PARENTHESIZED}: {subtitle}"
    assert issue is not None
    assert issue.code == "VAL_TITLE_REGROUNDED"


def test_fused_leading_parenthetical_is_restored_without_a_space():
    printed = "(Re)thinking Garden Plots as Classrooms"

    title, _ = ground_title_to_printed_text(
        "thinking Garden Plots as Classrooms", _titles(printed), printed
    )

    assert title == printed


@pytest.mark.parametrize(
    "label",
    ["(1)", "(2.1)", "(b)", "(iv)", "(2020)", "(Review)", "(Original Article)", "(Open Access)"],
)
def test_numbering_and_article_type_labels_stay_dropped(label):
    # These label the row rather than belong to the title, so the model was
    # right to leave them out. "(2020)" is a citation line's year, wrapped to
    # the start of a row after the author list.
    printed = f"{label} {_UNPARENTHESIZED}"

    title, issue = ground_title_to_printed_text(_UNPARENTHESIZED, _titles(printed), printed)

    assert title == _UNPARENTHESIZED
    assert issue is None


def test_parenthetical_is_restored_only_from_the_selected_title_rows():
    # An abstract sentence, or another record's title, that opens with the same
    # words is not this paper's printed title row.
    abstract = _candidate("c1", _PARENTHESIZED, roles=frozenset({"abstract"}))
    other_record = _candidate("c2", _PARENTHESIZED, roles=frozenset({"title"}))
    own_title = _candidate("c3", _UNPARENTHESIZED, roles=frozenset({"title"}))
    resolution = _resolution(abstract, other_record, own_title, selected_ids=("c1", "c3"))

    title, issue = ground_title_to_printed_text(
        _UNPARENTHESIZED, resolution, f"{_PARENTHESIZED}\n{_UNPARENTHESIZED}"
    )

    assert title == _UNPARENTHESIZED
    assert issue is None


def test_title_rows_that_disagree_on_the_parenthetical_abstain():
    printed = (_PARENTHESIZED, f"(Urban) {_UNPARENTHESIZED}")

    title, issue = ground_title_to_printed_text(
        _UNPARENTHESIZED, _titles(*printed), "\n".join(printed)
    )

    assert title == _UNPARENTHESIZED
    assert issue is None


async def test_extracted_title_keeps_the_printed_leading_parenthetical():
    # With a selected front-matter record, the model title is final: the
    # layout-title preference in post_parse does not run, so a truncated model
    # title used to reach the export unchanged.
    resolution = _resolution(
        _candidate("c1", _PARENTHESIZED, roles=frozenset({"title"}), source_kind="heading"),
        _candidate("c2", "Mira Ellison", roles=frozenset({"byline"}), text_ids=(2,)),
    )
    ext = _extractor(
        resolution,
        CoreMetadataLLM(
            title=_UNPARENTHESIZED,
            authors=[AuthorLLM(given="Mira", family="Ellison")],
            keywords=[],
        ),
    )

    metadata = await ext.extract()

    assert metadata.title == _PARENTHESIZED
    assert "VAL_TITLE_REGROUNDED" in [issue.code for issue in ext.validation_issues]


@pytest.mark.parametrize("one_title_region", [False, True])
async def test_bilingual_record_keeps_the_title_and_byline_printed_first(one_title_region):
    # The original title and byline are printed first, then their translation;
    # layout may read both titles as one region. The model's pick is final once
    # a record is selected, so nothing after it may swap in the later version.
    original = "Краткий обзор школьных садов"
    translation = "A brief review of school gardens"
    titles = (
        [_candidate("c1", f"{original} {translation}", roles=frozenset({"title"}))]
        if one_title_region
        else [
            _candidate("c1", original, roles=frozenset({"title"})),
            _candidate("c3", translation, roles=frozenset({"title"})),
        ]
    )
    bylines = [
        _candidate("c2", "Мира Эллисон", roles=frozenset({"byline"}), text_ids=(2,)),
        _candidate("c4", "Mira Ellison", roles=frozenset({"byline"}), text_ids=(4,)),
    ]
    ext = _extractor(
        _resolution(*sorted(titles + bylines, key=lambda candidate: candidate.reading_order)),
        CoreMetadataLLM(
            title=original,
            authors=[AuthorLLM(given="Мира", family="Эллисон")],
            keywords=[],
        ),
    )

    metadata = await ext.extract()

    assert metadata.title == original
    assert [(author.given, author.family) for author in metadata.authors] == [("Мира", "Эллисон")]
    codes = {issue.code for issue in ext.validation_issues}
    assert codes.isdisjoint(
        {"VAL_TITLE_REGROUNDED", "VAL_TITLE_UNGROUNDED", "VAL_AUTHOR_FABRICATED"}
    )
