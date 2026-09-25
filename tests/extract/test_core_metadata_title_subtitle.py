"""A subtitle printed on the row under the title is folded back into the title.

All titles, names, and rows below are invented.
"""

from dataclasses import replace

import pytest

from bibr.extract.title_subtitle import fold_printed_subtitle
from bibr.paper import PaperAuthor
from bibr.schemas import AuthorLLM, CoreMetadataLLM

from .test_core_metadata_author_guards import _candidate, _extractor, _resolution

_TITLE = "Seasonal colour change in small allotment gardens"
_SUBTITLE = "Lessons from a two-year classroom project"
_BYLINE = "Mira Ellison and Tomas Reyes"
_BODY = (
    "Allotment gardens are used in many towns to teach plant biology, and "
    "teachers often ask how the colour of leaves changes across the seasons."
)


def _title(text=_TITLE, candidate_id="c1", *, heading=True):
    return _candidate(
        candidate_id,
        text,
        roles=frozenset({"title", "heading"} if heading else {"title"}),
        source_kind="heading" if heading else "paragraph",
    )


def _row(candidate_id, text, roles=frozenset()):
    return _candidate(candidate_id, text, roles=roles)


def _byline(candidate_id="c3", text=_BYLINE):
    return _candidate(candidate_id, text, roles=frozenset({"byline"}), text_ids=(3,))


def test_subtitle_row_between_title_and_byline_is_folded():
    resolution = _resolution(_title(), _row("c2", _SUBTITLE), _byline())

    title, issue = fold_printed_subtitle(_TITLE, resolution)

    assert title == f"{_TITLE}: {_SUBTITLE}"
    assert issue is not None
    assert issue.code == "VAL_TITLE_REGROUNDED"
    assert issue.evidence_ids == ("c2", "reason:title_subtitle_row_dropped")
    assert not issue.blocking


def test_numbered_series_part_heading_is_folded():
    # The parser can make the series part its own heading, typed as a title.
    resolution = _resolution(
        _title("Careers in garden design"),
        _title("4. Planting schemes", "c2"),
        _byline(),
    )

    title, _ = fold_printed_subtitle("Careers in garden design", resolution)

    assert title == "Careers in garden design: 4. Planting schemes"


def test_subtitle_is_folded_when_the_byline_is_printed_above_the_title():
    # Byline, title, subtitle, then body text: nothing but the subtitle sits
    # between the title and the body.
    german_title = "Saisonale Farbwechsel und kleine Schulgärten"
    resolution = _resolution(
        _byline("c1"),
        _title(german_title, "c2"),
        _row("c3", "Ein Überblick über den Schwerpunkt"),
        _row("c4", _BODY),
    )

    title, _ = fold_printed_subtitle(german_title, resolution)

    assert title == f"{german_title}: Ein Überblick über den Schwerpunkt"


def test_title_printed_on_two_rows_still_finds_the_row_under_it():
    resolution = _resolution(
        _title("Seasonal colour change in small", "c1"),
        _title("allotment gardens", "c2", heading=False),
        _row("c3", _SUBTITLE),
        _byline("c4"),
    )

    title, _ = fold_printed_subtitle(_TITLE, resolution)

    assert title == f"{_TITLE}: {_SUBTITLE}"


@pytest.mark.parametrize(
    ("model_title", "row", "expected"),
    [
        # Printed punctuation already separates them.
        (
            "Are allotment gardens worth it?",
            _SUBTITLE,
            f"Are allotment gardens worth it? {_SUBTITLE}",
        ),
        ("Allotment gardens:", _SUBTITLE, f"Allotment gardens: {_SUBTITLE}"),
        # The rows are one phrase that the model cut short.
        ("Seasonal colour change in", "small allotment gardens", _TITLE),
    ],
)
def test_join_follows_the_printed_punctuation(model_title, row, expected):
    resolution = _resolution(_title(model_title), _row("c2", row), _byline())

    title, _ = fold_printed_subtitle(model_title, resolution)

    assert title == expected


@pytest.mark.parametrize(
    "row",
    [
        _row("c2", _BYLINE, frozenset({"byline"})),
        # Author lines the resolver did not mark as a byline.
        _row("c2", "Mira Ellison"),
        _row("c2", "MIRA ELLISON"),
        _row("c2", "M. A. Ellison"),
        _row("c2", "Mira Ellison1, Tomas Reyes2"),
        _row("c2", "Mira Ellison · Tomas Reyes"),
        _row("c2", "By Mira Ellison"),
        _row("c2", "Department of Plant Sciences, Northfield University"),
        _row("c2", "Northfield Horticultural Society", frozenset({"affiliation"})),
        _row("c2", "Received 3 March 2021; accepted 9 June 2021"),
        _row("c2", "Garden Studies Quarterly 12(3): 45-67"),
        _row("c2", "https://doi.org/10.1234/garden.5678"),
        _row("c2", "Research Article"),
        _row("c2", "Editorial"),
        _row("c2", "Introduction"),
        _row("c2", "1. Introduction"),
        _row("c2", "Keywords: gardens; seasons"),
        _row("c2", "*Correspondence"),
        _row("c2", "Journal of Classroom Gardening"),
        _row("c2", "(Continued from the previous issue)"),
        _row("c2", "Dear Editor,"),
        _row("c2", _BODY),
        # A parallel title in another language or script.
        _row("c2", "Cambio de color estacional en los huertos escolares"),
        _row("c2", "Сезонная смена цвета в школьных садах"),
    ],
    ids=lambda row: row.raw_text[:30],
)
def test_rows_that_are_not_a_subtitle_are_never_folded(row):
    resolution = _resolution(_title(), row, _byline())

    title, issue = fold_printed_subtitle(_TITLE, resolution)

    assert title == _TITLE
    assert issue is None


def test_a_row_naming_an_extracted_author_is_a_byline():
    row_text = "Notes from Ellison"
    resolution = _resolution(_title(), _row("c2", row_text), _byline())
    author = PaperAuthor(author_id=1, given="Mira", family="Ellison", affiliation="", role=[])

    assert fold_printed_subtitle(_TITLE, resolution)[0] == f"{_TITLE}: {row_text}"
    assert fold_printed_subtitle(_TITLE, resolution, authors=[author])[0] == _TITLE


def test_parallel_title_in_the_same_script_is_left_out():
    # The original title first, then its translation directly under it: the
    # model keeps the version printed first, and the fold must not undo that.
    original = "Cambio de color estacional en los huertos escolares"
    resolution = _resolution(_title(original), _row("c2", _TITLE), _byline())

    title, issue = fold_printed_subtitle(original, resolution)

    assert title == original
    assert issue is None


def test_title_the_model_already_joined_is_left_alone():
    resolution = _resolution(_title(), _row("c2", _SUBTITLE), _byline())
    joined = f"{_TITLE}: {_SUBTITLE}"

    assert fold_printed_subtitle(joined, resolution) == (joined, None)


def test_title_row_ending_the_record_has_nothing_to_fold():
    resolution = _resolution(_byline("c1"), _title(candidate_id="c2"))

    assert fold_printed_subtitle(_TITLE, resolution) == (_TITLE, None)


def test_row_on_the_next_page_is_not_under_the_title():
    resolution = _resolution(_title(), replace(_row("c2", _SUBTITLE), page=2))

    assert fold_printed_subtitle(_TITLE, resolution) == (_TITLE, None)


def test_row_printed_above_the_title_is_not_under_it():
    title_row = replace(_title(), bbox=(100.0, 200.0, 900.0, 240.0))
    row = replace(_row("c2", _SUBTITLE), bbox=(100.0, 120.0, 900.0, 150.0))

    assert fold_printed_subtitle(_TITLE, _resolution(title_row, row)) == (_TITLE, None)


def test_title_rows_outside_the_selected_record_are_ignored():
    resolution = _resolution(
        _title(),
        _row("c2", _SUBTITLE),
        _byline(),
        selected_ids=("c1", "c3"),
    )

    title, issue = fold_printed_subtitle(_TITLE, resolution)

    assert title == _TITLE
    assert issue is None


def test_two_different_rows_under_repeated_titles_abstain():
    resolution = _resolution(
        _title(),
        _row("c2", _SUBTITLE),
        _title(candidate_id="c3"),
        _row("c4", "A report for young gardeners"),
    )

    title, issue = fold_printed_subtitle(_TITLE, resolution)

    assert title == _TITLE
    assert issue is None


async def test_extracted_title_gets_the_subtitle_printed_under_it():
    resolution = _resolution(_title(), _row("c2", _SUBTITLE), _byline())
    ext = _extractor(
        resolution,
        CoreMetadataLLM(
            title=_TITLE,
            authors=[
                AuthorLLM(given="Mira", family="Ellison"),
                AuthorLLM(given="Tomas", family="Reyes"),
            ],
            keywords=[],
        ),
    )

    metadata = await ext.extract()

    assert metadata.title == f"{_TITLE}: {_SUBTITLE}"
    assert "reason:title_subtitle_row_dropped" in [
        evidence for issue in ext.validation_issues for evidence in issue.evidence_ids
    ]


async def test_extracted_title_keeps_out_an_author_line_printed_under_it():
    # The only byline row is unmarked, so the extracted authors are the evidence.
    resolution = _resolution(_title(), _row("c2", "Mira Ellison with Tomas Reyes"))
    ext = _extractor(
        resolution,
        CoreMetadataLLM(
            title=_TITLE,
            authors=[
                AuthorLLM(given="Mira", family="Ellison"),
                AuthorLLM(given="Tomas", family="Reyes"),
            ],
            keywords=[],
        ),
    )

    metadata = await ext.extract()

    assert metadata.title == _TITLE
