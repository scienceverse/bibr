"""Funding-rule cases a fast run got wrong: lost acronym funders and repeated awards."""

from __future__ import annotations

from bibr.extract.funding_rules import parse_funding_statement


def test_two_acronym_funders_joined_by_and_are_both_read():
    statement = "This work was supported by NSF and NIH."
    assert parse_funding_statement(statement, [("Jane", "Brown")]) == [("NSF", []), ("NIH", [])]


def test_awards_after_two_acronym_funders_go_to_the_nearest():
    statement = "Funded by NSF and NIH grants 1234567 and R01 AG012345."
    assert parse_funding_statement(statement) == [
        ("NSF", []),
        ("NIH", ["1234567", "R01 AG012345"]),
    ]


def test_author_initials_joined_by_and_are_still_not_a_funder():
    statement = "JB and NM were supported by the Wellcome Trust (204841/Z/16/Z)."
    assert parse_funding_statement(statement, [("Jane", "Brown"), ("Nina", "Moss")]) == [
        ("Wellcome Trust", ["204841/Z/16/Z"])
    ]
    assert parse_funding_statement(statement) == [("Wellcome Trust", ["204841/Z/16/Z"])]


def test_an_award_repeated_in_a_later_sentence_is_listed_once():
    statement = (
        "This work was supported by the National Science Foundation under grant 1234567. "
        "Grant 1234567 also funded the data collection."
    )
    assert parse_funding_statement(statement) == [("National Science Foundation", ["1234567"])]
