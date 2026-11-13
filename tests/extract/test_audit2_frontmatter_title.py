"""Title repairs: the subtitle fold, its trailing mark, and the masthead guard.

All titles, names, and rows below are invented.
"""

import pytest

from bibr.extract.title_candidates import _detected_title_is_masthead, selected_title_candidate
from bibr.extract.title_subtitle import fold_printed_subtitle

from .test_core_metadata_author_guards import _candidate, _resolution

_TITLE = "Deep learning for the reading of chest radiographs"


def _title_row(text=_TITLE):
    return _candidate("c1", text, roles=frozenset({"title", "heading"}), source_kind="heading")


def _row(text):
    return _candidate("c2", text, roles=frozenset())


def _byline():
    return _candidate("c3", "Mira Ellison and Tomas Reyes", roles=frozenset({"byline"}))


def _fold(row_text, title=_TITLE):
    return fold_printed_subtitle(title, _resolution(_title_row(title), _row(row_text), _byline()))[
        0
    ]


@pytest.mark.parametrize(
    "subtitle",
    ["A systematic review", "A pilot study in two hospitals", "A case series"],
)
def test_an_english_subtitle_opening_with_a_is_not_a_parallel_title(subtitle):
    assert _fold(subtitle) == f"{_TITLE}: {subtitle}"


@pytest.mark.parametrize(
    ("subtitle", "expected"),
    [
        ("Optimal route planning with IDA*", "Optimal route planning with IDA*"),
        ("Route planning with A*", "Route planning with A*"),
        # A note mark after an ordinary word, or after a number, still goes.
        ("Lessons from two regional hospitals*", "Lessons from two regional hospitals"),
        ("Lessons from the COVID-19 pandemic†", "Lessons from the COVID-19 pandemic"),
        ("Lessons from two years of COVID-19*", "Lessons from two years of COVID-19"),
    ],
)
def test_a_star_that_ends_a_name_in_the_subtitle_is_kept(subtitle, expected):
    assert _fold(subtitle) == f"{_TITLE}: {expected}"


@pytest.mark.parametrize(
    "row_text", ["Li Wei", "Xi Chen", "Lili Zhang", "Vic Matthews", "Li Wei and Xi Chen"]
)
def test_a_name_that_spells_a_roman_numeral_is_not_series_numbering(row_text):
    assert _fold(row_text) == _TITLE


@pytest.mark.parametrize(
    "row_text", ["IV Psychotherapy", "IV. Psychotherapy", "Part II Planting schemes"]
)
def test_series_numbering_is_still_read_as_numbering(row_text):
    assert _fold(row_text) == f"{_TITLE}: {row_text}"


@pytest.mark.parametrize(
    ("detected", "journal"),
    [
        ("The signature of selection in wild mouse populations", "Nature"),
        ("Dental implants in older adults", "Plants"),
    ],
)
def test_a_journal_name_inside_a_word_of_the_title_is_no_masthead(detected, journal):
    assert not _detected_title_is_masthead(detected, journal, None)


@pytest.mark.parametrize(
    ("detected", "journal"),
    [
        ("NATURE", "Nature"),
        ("Nature | Vol 581 | 2020", "Nature"),
        ("Plants 2021, 10, 345", "Plants"),
    ],
)
def test_a_journal_banner_is_still_a_masthead(detected, journal):
    assert _detected_title_is_masthead(detected, journal, None)


def test_the_selected_title_row_is_recovered_when_a_word_contains_the_journal_name():
    title = "The signature of selection in wild mouse populations"
    resolution = _resolution(_title_row(title), _byline())

    candidate, _reason = selected_title_candidate(resolution, journal="Nature", publisher=None)

    assert candidate is not None
    assert candidate.value == title
