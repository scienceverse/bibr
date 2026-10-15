"""The paper's title and authors compared with its own registry record."""

from __future__ import annotations

import copy

import pytest

from bibr.enrich.verify_identity import authors_agree, titles_agree, verify_identity


def _person(given: str | None, family: str | None) -> dict:
    return {"given": given, "family": family, "literal": None}


def _payload(
    title: str | None = "Trust in science across forty countries",
    authors: list[dict] | None = None,
    match: dict | None = None,
    *,
    doi: str | None = "10.1234/self",
) -> dict:
    authors = [_person("Jane", "Smith"), _person("Ali", "Khan")] if authors is None else authors
    record = {
        "service": "crossref",
        "score": 1.0,
        "doi": "10.1234/self",
        "title": "Trust in science across forty countries",
        "author": [{"given": "Jane", "family": "Smith"}, {"given": "Ali", "family": "Khan"}],
    }
    record.update(match or {})
    return {
        "metadata": {"title": title, "doi": doi},
        "author": authors,
        "metadata_match": [record],
        "extraction": {
            "fields": {
                "title": {"state": "extracted", "source": "llm", "issues": []},
                "author": {"state": "extracted", "source": "llm", "issues": []},
                "doi": {"state": "extracted", "source": "doi_regex", "issues": []},
            }
        },
    }


def _registry(payload: dict, field: str) -> dict | None:
    return payload["extraction"]["fields"][field].get("registry")


# --- the payload --------------------------------------------------------------


def test_agreeing_title_and_authors_are_marked_agree():
    payload = _payload()

    assert verify_identity(payload) == {"title": "agree", "author": "agree"}
    assert _registry(payload, "title") == {"service": "crossref", "outcome": "agree"}
    assert _registry(payload, "author") == {"service": "crossref", "outcome": "agree"}
    # Only title and author are compared.
    assert "registry" not in payload["extraction"]["fields"]["doi"]


def test_disagreeing_fields_are_marked_and_never_changed():
    payload = _payload(
        title="Supplementary materials",
        authors=[_person("Jane", "Smith"), _person("Bo", "Lee")],
    )
    before = copy.deepcopy(payload)

    assert verify_identity(payload) == {"title": "disagree", "author": "disagree"}
    # The printed values stay as they are: the record only reports.
    assert payload["metadata"] == before["metadata"]
    assert payload["author"] == before["author"]
    assert payload["metadata_match"] == before["metadata_match"]


def test_no_record_leaves_no_registry_and_clears_a_stale_one():
    payload = _payload()
    payload["metadata_match"] = []
    payload["extraction"]["fields"]["title"]["registry"] = {
        "service": "crossref",
        "outcome": "agree",
    }

    assert verify_identity(payload) == {}
    assert _registry(payload, "title") is None
    assert _registry(payload, "author") is None


@pytest.mark.parametrize("score", [0.85, None, 80])
def test_a_record_below_the_confidence_floor_is_not_compared(score):
    payload = _payload(match={"score": score})

    assert verify_identity(payload) == {}
    assert _registry(payload, "title") is None


def test_a_record_scored_on_the_old_0_to_100_scale_is_compared():
    assert verify_identity(_payload(match={"score": 100})) == {
        "title": "agree",
        "author": "agree",
    }


def test_a_record_of_another_doi_is_not_compared():
    payload = _payload(match={"doi": "10.1234/other"})

    assert verify_identity(payload) == {}


def test_doi_case_does_not_matter():
    payload = _payload(doi="10.1234/SELF")

    assert verify_identity(payload) == {"title": "agree", "author": "agree"}


def test_a_field_the_record_lacks_gets_no_outcome():
    payload = _payload(match={"title": None, "author": None})
    payload["extraction"]["fields"]["author"]["registry"] = {
        "service": "crossref",
        "outcome": "disagree",
    }

    assert verify_identity(payload) == {}
    assert _registry(payload, "title") is None
    assert _registry(payload, "author") is None


def test_a_missing_printed_title_disagrees_with_the_record():
    payload = _payload(title=None, authors=[])

    assert verify_identity(payload) == {"title": "disagree", "author": "disagree"}


def test_crossref_wins_over_other_services():
    payload = _payload()
    payload["metadata_match"].insert(
        0,
        {
            "service": "openalex",
            "score": 1.0,
            "doi": "10.1234/self",
            "title": "Something else entirely",
        },
    )

    assert verify_identity(payload)["title"] == "agree"
    assert _registry(payload, "title")["service"] == "crossref"


def test_running_twice_gives_the_same_payload():
    payload = _payload(title="Supplementary materials")
    verify_identity(payload)
    once = copy.deepcopy(payload)

    verify_identity(payload)

    assert payload == once


def test_apply_false_only_reports():
    payload = _payload()
    before = copy.deepcopy(payload)

    assert verify_identity(payload, apply=False) == {"title": "agree", "author": "agree"}
    assert payload == before


def test_a_payload_without_field_states_is_left_alone():
    payload = _payload()
    del payload["extraction"]["fields"]

    assert verify_identity(payload) == {"title": "agree", "author": "agree"}
    assert "fields" not in payload["extraction"]


# --- titles -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("printed", "record"),
    [
        ("Trust in Science", "Trust in science"),
        ("Trust in science: a survey", "Trust in science - a survey"),
        ("Effects of CO2 on growth", "Effects of CO<sub>2</sub> on growth"),
        ("Genes & development in mice", "Genes &amp; development in mice"),
        # A word-aligned prefix that keeps at least 70% of the longer title.
        ("Trust in science across forty countries", "Trust in science across forty"),
        # Scripts without spaces: a prefix of six or more characters.
        ("深層学習による画像認識の研究：新しい手法", "深層学習による画像認識の研究"),
    ],
)
def test_titles_that_agree(printed, record):
    payload = _payload(title=printed, match={"title": record})

    assert verify_identity(payload)["title"] == "agree"


@pytest.mark.parametrize(
    ("printed", "record"),
    [
        ("Trust in science", "Trust in science across forty countries"),
        ("Trust in science across forty countries", "Trust in sciences across forty countries"),
        ("Introduction", "Introductions"),
        ("Supplementary materials", "Trust in science"),
        ("", "Trust in science"),
    ],
)
def test_titles_that_disagree(printed, record):
    payload = _payload(title=printed, match={"title": record})

    assert verify_identity(payload)["title"] == "disagree"


def test_the_record_subtitle_is_tried_after_the_title():
    printed = "Trust in science: Evidence from a survey of forty countries"
    with_subtitle = _payload(
        title=printed,
        match={
            "title": "Trust in science",
            "subtitle": "Evidence from a survey of forty countries",
        },
    )
    without_subtitle = _payload(title=printed, match={"title": "Trust in science"})

    assert verify_identity(with_subtitle)["title"] == "agree"
    assert verify_identity(without_subtitle)["title"] == "disagree"


def test_a_subtitle_marked_up_in_the_record_is_read_as_text():
    payload = _payload(
        title="Plants under stress: The role of CO2",
        match={"title": "Plants under stress", "subtitle": "The role of CO<sub>2</sub>"},
    )

    assert verify_identity(payload)["title"] == "agree"


# --- authors ------------------------------------------------------------------


def test_a_different_given_family_split_of_the_same_name_agrees():
    printed = [_person("Leila", "Ben Ammar Elgaaied"), _person("Ali", "Khan")]
    record = [
        {"given": "Leila Ben Ammar", "family": "Elgaaied"},
        {"given": "Ali", "family": "Khan"},
    ]

    assert authors_agree(printed, record)


def test_name_order_and_accents_do_not_matter():
    printed = [_person("Felix", "Oloda Oluwatayo"), _person("José", "Müller")]
    record = [
        {"given": "Oluwatayo Felix", "family": "Oloda"},
        {"given": "Jose", "family": "Muller"},
    ]

    assert authors_agree(printed, record)


def test_author_order_does_not_matter():
    printed = [_person("Ali", "Khan"), _person("Jane", "Smith")]
    record = [{"given": "Jane", "family": "Smith"}, {"given": "Ali", "family": "Khan"}]

    assert authors_agree(printed, record)


def test_initials_agree_with_full_given_names():
    printed = [_person("J. A.", "Smith"), _person("A.", "Khan")]
    record = [{"given": "Jane A.", "family": "Smith"}, {"given": "Ali", "family": "Khan"}]

    assert authors_agree(printed, record)


def test_a_dropped_surname_disagrees():
    # Jaro-Winkler alone scores "kaisin" against "kaisin yee" above the threshold.
    printed = [_person("Kaisin", None)]
    record = [{"given": "Kaisin", "family": "Yee"}]

    assert not authors_agree(printed, record)


def test_one_author_missing_from_four_disagrees():
    record = [
        {"given": g, "family": f}
        for g, f in [("Jane", "Smith"), ("Ali", "Khan"), ("Bo", "Lee"), ("Ana", "Silva")]
    ]
    printed = [_person(r["given"], r["family"]) for r in record[:3]]

    assert not authors_agree(printed, record)


def test_one_author_missing_from_ten_still_agrees():
    # F1 over the pairs is 0.947: the threshold tolerates one slip in a long list.
    record = [{"given": f"Given{i}", "family": f"Family{chr(65 + i)}"} for i in range(10)]
    printed = [_person(r["given"], r["family"]) for r in record[:9]]

    assert authors_agree(printed, record)


def test_printed_group_authors_are_left_out_when_the_record_lists_none():
    # Crossref records keep only people, so a consortium byline is not a miss.
    printed = [_person("Jane", "Smith"), {"given": None, "family": None, "literal": "ABC Group"}]
    record = [{"given": "Jane", "family": "Smith"}]

    assert authors_agree(printed, record)


def test_a_group_only_byline_disagrees_with_a_list_of_people():
    printed = [{"given": None, "family": None, "literal": "ABC Group"}]
    record = [{"given": "Jane", "family": "Smith"}]

    assert not authors_agree(printed, record)


def test_group_authors_are_compared_when_the_record_lists_them():
    printed = [_person("Jane", "Smith"), {"given": None, "family": None, "literal": "ABC Group"}]
    record = [{"given": "Jane", "family": "Smith"}, {"literal": "ABC Group"}]

    assert authors_agree(printed, record)
    assert not authors_agree(printed[:1], record)


# --- parity with the validation scorer -----------------------------------------

TITLE_PAIRS = [
    ("Trust in Science", "Trust in science"),
    ("Trust in science: a survey", "Trust in science - a survey"),
    ("Trust in science across forty countries", "Trust in science across forty"),
    ("Trust in science", "Trust in science across forty countries"),
    ("Introduction", "Introductions"),
    ("A B", "A B C"),
    ("深層学習による画像認識の研究：新しい手法", "深層学習による画像認識の研究"),
    ("画像認識：新しい", "画像認識"),
    ("Mental health, the pandemic and young people", "Mental health the pandemic and young"),
]

AUTHOR_PAIRS = [
    ([("Leila", "Ben Ammar Elgaaied")], [("Leila Ben Ammar", "Elgaaied")]),
    ([("Felix", "Oloda Oluwatayo")], [("Oluwatayo Felix", "Oloda")]),
    ([("Kaisin", "")], [("Kaisin", "Yee")]),
    ([("J. A.", "Smith"), ("A.", "Khan")], [("Jane A.", "Smith"), ("Ali", "Khan")]),
    ([("Jane", "Smith"), ("Ali", "Khan"), ("Bo", "Lee")], [("Jane", "Smith"), ("Ali", "Khan")]),
    ([("Jane", "Smith")], [("John", "Smyth")]),
    ([("Maria", "Garcia-Lopez")], [("María", "García López")]),
]


@pytest.mark.parametrize(("printed", "record"), TITLE_PAIRS)
def test_titles_match_the_scorer(printed, record):
    from evaluation.validation_metrics import title_soft_match

    assert titles_agree(printed, record) is (title_soft_match(printed, record) == 1.0)


@pytest.mark.parametrize(("printed", "record"), AUTHOR_PAIRS)
def test_author_lists_match_the_scorer(printed, record):
    from evaluation.validation_metrics import authors_fullname_f1

    printed_rows = [{"given": g, "family": f} for g, f in printed]
    record_rows = [{"given": g, "family": f} for g, f in record]
    scored = authors_fullname_f1(printed_rows, record_rows) >= 0.9

    assert authors_agree(printed_rows, record_rows) is scored
