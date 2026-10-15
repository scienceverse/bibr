"""The reference check as the reference-list writers run it on an export
without ``bib_check`` rows: no lookups, only ``bib`` and ``bib_match``."""

from __future__ import annotations

from bibr.enrich.check import check_bibs


def _bib(bib_id: int, **fields):
    row = {
        "bib_id": bib_id,
        "bib_type": "journal_article",
        "title": "A Prior Study of Things",
        "authors": "Smith, J., & Doe, A.",
        "year": 2020,
        "container": "Journal of Things",
    }
    row.update(fields)
    return row


def _match(bib_id: int, **fields):
    row = {
        "bib_id": bib_id,
        "service": "crossref",
        "score": 1.0,
        "bib_type": "journal_article",
        "doi": "10.1234/prior",
        "title": "A Prior Study of Things",
        "author": [{"family": "Smith", "given": "Jane"}, {"family": "Doe", "given": "Anna"}],
        "year": 2020,
        "container": "Journal of Things",
    }
    row.update(fields)
    return row


def _statuses(data):
    check_bibs(data)
    return [row["status"] for row in data["bib_check"]]


def test_agreeing_record_is_matched():
    data = {"bib": [_bib(1)], "bib_match": [_match(1)]}
    assert _statuses(data) == ["matched"]
    assert data["bib_check"][0]["service"] == "crossref"
    assert data["bib_check"][0]["retraction"] is None


def test_another_title_disagrees_and_says_which_field():
    data = {
        "bib": [_bib(1, title="Completely Different Title About Fish")],
        "bib_match": [_match(1, title="Neural Correlates of Something Else Entirely")],
    }
    assert _statuses(data) == ["fields_disagree"]
    assert {note["field"] for note in data["bib_check"][0]["notes"]} >= {"title"}


def test_without_a_record_or_lookup_nothing_is_compared():
    data = {"bib": [_bib(1)], "bib_match": []}
    assert _statuses(data) == ["unchecked"]
    assert data["bib_check"][0]["notes"] == []


def test_a_year_two_off_on_the_later_side_still_agrees_but_three_does_not():
    later = {"bib": [_bib(1, year=2022)], "bib_match": [_match(1, year=2020)]}
    assert _statuses(later) == ["matched"]
    far = {"bib": [_bib(1, year=2023)], "bib_match": [_match(1, year=2020)]}
    assert _statuses(far) == ["fields_disagree"]


def test_rows_skip_entries_without_an_id():
    data = {"bib": [_bib(1), {"title": "no id"}], "bib_match": [_match(1)]}
    assert _statuses(data) == ["matched"]
