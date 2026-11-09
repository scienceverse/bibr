"""The parts of a printed affiliation string a fast run reads without an LLM."""

from __future__ import annotations

import pytest

from bibr.extract.affiliation_rules import parse_affiliation


@pytest.mark.parametrize(
    ("text", "institution", "department", "city", "country"),
    [
        (
            "Department of Psychology, University of Example, Utrecht, The Netherlands",
            "University of Example",
            "Department of Psychology",
            "Utrecht",
            "The Netherlands",
        ),
        # A region after the city is not the city.
        (
            "School of Nursing, Example State University, Muncie, Indiana, United States",
            "Example State University",
            "School of Nursing",
            "Muncie",
            "United States",
        ),
        # City, state code and country in one segment.
        (
            "Department of Psychology, Example University, Cambridge MA USA",
            "Example University",
            "Department of Psychology",
            "Cambridge",
            "USA",
        ),
        # A street and a postcode are not the city; "75000 Paris" is Paris.
        (
            "Institute of Example Studies, 12 Example Street, 75000 Paris, France",
            "Institute of Example Studies",
            None,
            "Paris",
            "France",
        ),
        # Without a university, the last place is the city.
        ("Example Agency, London UK", "Example Agency", None, "London", "UK"),
        ("Example University (España)", "Example University", None, None, "España"),
        # A role is not a department, and a company keeps its "Inc.".
        (
            "Assistant Professor, Dept. of Physics, Example Institute of Technology, Pune, India",
            "Example Institute of Technology",
            "Dept. of Physics",
            "Pune",
            "India",
        ),
        (
            "Example Pharmaceuticals, Inc., Chapel Hill, NC, USA",
            "Example Pharmaceuticals, Inc.",
            None,
            "Chapel Hill",
            "USA",
        ),
    ],
)
def test_reads_the_printed_parts(text, institution, department, city, country):
    assert parse_affiliation(text) == {
        "institution": institution,
        "department": department,
        "city": city,
        "country": country,
    }


@pytest.mark.parametrize("text", ["", "Corresponding author: Jane Doe"])
def test_a_string_that_is_not_an_affiliation_gives_no_parts(text):
    assert set(parse_affiliation(text).values()) == {None}


def test_a_single_name_gives_no_place():
    assert parse_affiliation("Example University") == {
        "institution": "Example University",
        "department": None,
        "city": None,
        "country": None,
    }
