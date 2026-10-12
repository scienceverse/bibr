"""Verification of parsed reference locators against the matched record."""

from __future__ import annotations

import pytest

from bibr.enrich.verify import verify_bibs


def _payload(bib: dict, match: dict, raw: str | None) -> dict:
    text = [{"text_id": 7, "text": raw}] if raw is not None else []
    return {
        "text": text,
        "bib": [{"bib_id": 1, "text_id": 7 if raw is not None else None, **bib}],
        "bib_match": [{"bib_id": 1, "service": "crossref", "score": 1.0, **match}],
        "extraction": {"diagnostics": {}},
    }


def _row(payload: dict) -> dict:
    rows = payload["extraction"]["diagnostics"]["verification"]
    assert len(rows) == 1
    return rows[0]


VANCOUVER = "4. Kopf M, Baumann H. Impaired responses in mice. Nature. 1994;368(6469):339-342."


def test_fills_a_dropped_volume_and_pages_the_reference_prints():
    payload = _payload(
        {"volume": None, "issue": "368(6469):339-342"},
        {"volume": "368", "issue": "6469", "first_page": "339", "last_page": "342"},
        VANCOUVER,
    )
    verify_bibs(payload)
    bib = payload["bib"][0]
    assert (bib["volume"], bib["first_page"], bib["last_page"]) == ("368", "339", "342")
    # The parsed issue swallowed the rest of the locator; the record's issue is
    # printed, so it replaces it.
    assert bib["issue"] == "6469"
    assert _row(payload) == {
        "bib_id": 1,
        "service": "crossref",
        "volume": "filled",
        "issue": "corrected",
        "first_page": "filled",
        "last_page": "filled",
    }
    assert payload["extraction"]["diagnostics"]["consolidation"] == [
        {"bib_id": 1, "fields": ["volume", "issue", "first_page", "last_page"]}
    ]


def test_never_takes_a_value_the_reference_does_not_print():
    # An article-number journal: the record's pagination is not on the page.
    raw = "Boele S. (2022). Testing processes. J Youth Adolesc, 51(8), 1-15."
    payload = _payload(
        {"volume": "51", "first_page": None},
        {"volume": "51", "first_page": "1656", "last_page": "1670"},
        raw,
    )
    verify_bibs(payload)
    assert payload["bib"][0]["first_page"] is None
    assert payload["bib"][0].get("last_page") is None
    assert _row(payload) == {"bib_id": 1, "service": "crossref", "volume": "agree"}


def test_a_printed_value_that_differs_from_the_record_is_kept_and_flagged():
    raw = "Smith J. A title here. J Phil. 2001;12:100-110."
    payload = _payload(
        {"volume": "12", "first_page": "100", "last_page": "110"},
        {"volume": "13", "first_page": "100", "last_page": "110"},
        raw,
    )
    verify_bibs(payload)
    assert payload["bib"][0]["volume"] == "12"
    assert _row(payload)["volume"] == "disagree"
    assert "consolidation" not in payload["extraction"]["diagnostics"]


def test_corrects_a_span_that_ran_over():
    raw = "Wang L. Targeting txnip. J Cell Physiol. 2021;236(6): 4625-4639."
    payload = _payload(
        {"volume": "2021;236(6", "first_page": "4625", "last_page": "4639"},
        {"volume": "236", "issue": "6", "first_page": "4625", "last_page": "4639"},
        raw,
    )
    verify_bibs(payload)
    assert payload["bib"][0]["volume"] == "236"
    assert _row(payload)["volume"] == "corrected"


def test_does_not_move_a_token_another_field_already_holds():
    # The record (wrongly) says volume 20; the reference prints 20 as its first
    # page, and the parser already read it so.
    raw = "Balter M. 2011. Was North Africa the launch pad? Science 331:20-23."
    payload = _payload(
        {"volume": None, "first_page": "20", "last_page": "23"},
        {"volume": "20", "first_page": "20", "last_page": "23"},
        raw,
    )
    verify_bibs(payload)
    assert payload["bib"][0]["volume"] is None
    assert "volume" not in _row(payload)


def test_a_number_inside_a_longer_token_is_not_printed():
    raw = "Doe J. Some title words. Journal X. 2012;e124."
    payload = _payload({"volume": None}, {"volume": "12"}, raw)
    verify_bibs(payload)
    assert payload["bib"][0]["volume"] is None


def test_keeps_an_abbreviated_last_page_as_printed():
    raw = "Connors JP. Citizen science. Ann Assoc Am Geogr. 2012;102(6): 1267-89."
    payload = _payload(
        {"volume": "102", "first_page": "1267", "last_page": None},
        {"volume": "102", "first_page": "1267", "last_page": "1289"},
        raw,
    )
    verify_bibs(payload)
    assert payload["bib"][0]["last_page"] == "89"
    assert _row(payload)["last_page"] == "filled"


@pytest.mark.parametrize(
    ("parsed", "record"),
    [("50", "50 Suppl"), ("10", "10(1)")],
)
def test_a_record_that_only_adds_to_the_parsed_volume_agrees(parsed, record):
    raw = f"Roy-Byrne PP. Valproate in anxiety. J Clin Psychiatry. 1989; {record}:44-8."
    payload = _payload({"volume": parsed}, {"volume": record}, raw)
    verify_bibs(payload)
    assert payload["bib"][0]["volume"] == parsed
    assert _row(payload)["volume"] == "agree"


def test_a_parsed_range_in_last_page_agrees_with_either_end():
    raw = "Raymond J. Biofeedback and dance. Appl Psychophysiol Biofeedback 2005;30:64-73"
    payload = _payload({"last_page": "64-73"}, {"last_page": "73"}, raw)
    verify_bibs(payload)
    assert payload["bib"][0]["last_page"] == "64-73"
    assert _row(payload)["last_page"] == "agree"


def test_an_unconfident_match_is_not_used():
    payload = _payload({"volume": None}, {"volume": "368", "score": 0.85}, VANCOUVER)
    verify_bibs(payload)
    assert payload["bib"][0]["volume"] is None
    assert payload["extraction"]["diagnostics"]["verification"] == []


def test_reads_a_0_to_100_score():
    payload = _payload({"volume": None}, {"volume": "368", "score": 98.0}, VANCOUVER)
    verify_bibs(payload)
    assert payload["bib"][0]["volume"] == "368"


def test_without_the_reference_string_it_only_compares():
    payload = _payload({"volume": None}, {"volume": "368"}, None)
    verify_bibs(payload)
    assert payload["bib"][0]["volume"] is None
    assert payload["extraction"]["diagnostics"]["verification"] == []


def test_apply_false_reports_without_writing():
    payload = _payload({"volume": None}, {"volume": "368"}, VANCOUVER)
    report = verify_bibs(payload, apply=False)
    assert payload["bib"][0]["volume"] is None
    assert report.outcomes == {"volume": {"filled": 1}}
    assert "consolidation" not in payload["extraction"]["diagnostics"]


def test_the_receipt_validates_against_the_export_model():
    from bibr.export.models import DiagnosticsExport

    payload = _payload(
        {"volume": None},
        {"volume": "368", "first_page": "339", "last_page": "342"},
        VANCOUVER,
    )
    verify_bibs(payload)
    model = DiagnosticsExport.model_validate(payload["extraction"]["diagnostics"])
    dumped = model.model_dump()
    assert dumped["verification"] == [
        {
            "bib_id": 1,
            "service": "crossref",
            "volume": "filled",
            "first_page": "filled",
            "last_page": "filled",
        }
    ]
