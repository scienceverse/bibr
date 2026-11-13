"""List-number stripping on a one-entry list, and references that open with a number."""

import pytest

from bibr.extract.ref_extractor import _strip_enum_markers


@pytest.mark.parametrize(
    "reference",
    [
        "24 Hours in A&E (2011). Channel 4 documentary series.",
        "100 Women in Science (2019). BBC Radio 4.",
        "3 Rivers Trust (2012). Annual report. Pittsburgh.",
    ],
)
def test_a_lone_reference_keeps_the_number_its_title_opens_with(reference):
    assert _strip_enum_markers([reference]) == [reference]


@pytest.mark.parametrize(
    ("reference", "parsed"),
    [
        ("1 Smith J. A study. Nature 2020;1:1.", "Smith J. A study. Nature 2020;1:1."),
        ("[12] Smith J. A study. 2020.", "Smith J. A study. 2020."),
        ("12. Smith J. A study. 2020.", "Smith J. A study. 2020."),
        ("(3) Smith J. A study. 2020.", "Smith J. A study. 2020."),
    ],
)
def test_a_lone_numbered_reference_still_loses_its_list_number(reference, parsed):
    assert _strip_enum_markers([reference]) == [parsed]


def test_titles_opening_with_a_number_or_year_inside_a_numbered_list():
    assert _strip_enum_markers(
        [
            "[1] Smith J. A study. 2020.",
            "[2] 2001: A Space Odyssey. MGM; 1968.",
            "[3] 3GPP TS 36.211. Physical channels and modulation; 2010.",
        ]
    ) == [
        "Smith J. A study. 2020.",
        "2001: A Space Odyssey. MGM; 1968.",
        "3GPP TS 36.211. Physical channels and modulation; 2010.",
    ]


def test_an_unnumbered_list_keeps_references_that_open_with_a_year():
    refs = [
        "2001: A Space Odyssey. Kubrick S. 1968.",
        "3GPP TS 36.211. 2010.",
        "Smith J (2003) Title. Journal 1:2.",
    ]
    assert _strip_enum_markers(refs) == refs
