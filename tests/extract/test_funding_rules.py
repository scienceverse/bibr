"""The funders and award numbers a fast run reads from a funding statement without an LLM."""

from __future__ import annotations

import pytest

from bibr.extract.funding_rules import parse_funding_statement


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        (
            "Funding: This work was supported by the National Science Foundation (grant no. "
            "BCS-1734304) and the Deutsche Forschungsgemeinschaft (DFG) under grant 431549029.",
            [
                ("National Science Foundation", ["BCS-1734304"]),
                ("Deutsche Forschungsgemeinschaft (DFG)", ["431549029"]),
            ],
        ),
        # The award is printed before its funder, joined by "from".
        (
            "This study was funded by grant R01 MH075750 from the National Institute of Mental "
            "Health.",
            [("National Institute of Mental Health", ["R01 MH075750"])],
        ),
        # An acronym named again later is the same funder; its award joins the first mention.
        (
            "This work was funded by the Medical Research Council (MRC) and the Wellcome Trust. "
            "MRC grant MR/N0137941/1.",
            [("Medical Research Council (MRC)", ["MR/N0137941/1"]), ("Wellcome Trust", [])],
        ),
        # A line-break hyphen inside an identifier is closed up.
        (
            "Supported by the Research Council of Norway (grant CMS- CSST-2021-A01).",
            [("Research Council of Norway", ["CMS-CSST-2021-A01"])],
        ),
    ],
)
def test_reads_the_funders_and_their_awards(statement, expected):
    assert parse_funding_statement(statement) == expected


@pytest.mark.parametrize(
    "statement",
    [
        "",
        "This research received no external funding.",
        "The authors received no specific funding for this work. The funders had no role in "
        "study design, data collection and analysis.",
        "This study was approved by the ethics committee of the University of Example.",
    ],
)
def test_a_statement_that_names_no_funding_gives_none(statement):
    assert parse_funding_statement(statement) == []


def test_an_authors_initials_are_not_a_funder():
    statement = "JB was supported by a Wellcome Trust fellowship (204841/Z/16/Z)."
    assert parse_funding_statement(statement, [("Jane", "Brown")]) == [
        ("Wellcome Trust", ["204841/Z/16/Z"])
    ]


def test_a_person_the_grant_was_awarded_to_is_not_a_funder():
    statement = (
        "This work was supported by the Swiss National Science Foundation (grant 100014_179134 "
        "awarded to Anna Keller)."
    )
    names = [name for name, _ in parse_funding_statement(statement, [("Anna", "Keller")])]
    assert names == ["Swiss National Science Foundation"]


def test_years_and_call_identifiers_are_not_awards():
    statement = (
        "Funded by the European Commission under call FP7-PEOPLE-2013-ITN in 2014 "
        "(grant agreement No. 607139)."
    )
    assert parse_funding_statement(statement) == [("European Commission", ["607139"])]
