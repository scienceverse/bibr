"""CSL-JSON, BibTeX and RIS reference lists (bibr.export.citations)."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, get_args

import pytest

import bibr
from bibr.api import Result
from bibr.export.citations import (
    CONTENT_TYPES,
    FORMATS,
    expand_page_range,
    format_for_path,
    normalize_format,
    render_references,
    to_bibtex,
    to_csl_json,
    to_ris,
    write_references,
)
from bibr.export.models import BibTypeLiteral

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
# An enriched export as a bibr with the reference check (export schema 12.3)
# writes it, with bib_check rows the writers read when present.
ENRICHED = FIXTURES / "references" / "enriched_check_12_3.json"
INSPECT = FIXTURES / "inspect_full_export.json"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _export(
    bibs: list[dict[str, Any]],
    matches: list[dict[str, Any]] | None = None,
    checks: list[dict[str, Any]] | None = None,
    paper_id: str = "paper-1",
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "paper_id": paper_id,
        "schema_version": "12.3",
        "bib": bibs,
        "bib_match": matches or [],
    }
    if checks is not None:
        data["bib_check"] = checks
    return data


def _bib(bib_id: int = 1, **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "bib_id": bib_id,
        "bib_type": "journal_article",
        "title": "A Prior Study",
        "authors": "Smith, J., & Doe, A.",
        "year": 2020,
        "container": "Journal of Things",
    }
    row.update(fields)
    return row


def _match(bib_id: int = 1, **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "bib_id": bib_id,
        "service": "crossref",
        "score": 1.0,
        "bib_type": "journal_article",
        "doi": "10.1234/prior",
        "title": "A Prior Study",
        "author": [{"family": "Smith", "given": "Jane"}, {"family": "Doe", "given": "Anna"}],
        "year": 2020,
        "container": "Journal of Things",
        "volume": "8",
        "issue": "2",
        "first_page": "101",
        "last_page": "120",
        "publisher": "Things Press",
    }
    row.update(fields)
    return row


def _check(bib_id: int = 1, status: str = "matched", **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "bib_id": bib_id,
        "status": status,
        "service": "crossref" if status in ("matched", "fields_disagree") else None,
        "notes": [],
        "retraction": None,
    }
    row.update(fields)
    return row


def _one_csl(data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    items = to_csl_json(data, **kwargs)
    assert len(items) == 1
    return items[0]


# ---------------------------------------------------------------------------
# A tiny BibTeX reader (BibTeX's own brace counting) and format checks
# ---------------------------------------------------------------------------

_ENTRY_START = re.compile(r"@(\w+)\{([^,\s{}]+),?")
_FIELD_NAME = re.compile(r"\s*(\w+)\s*=\s*")
_UNESCAPE = [
    (r"\textbackslash{}", "\\"),
    (r"\textbraceleft{}", "{"),
    (r"\textbraceright{}", "}"),
    (r"\textasciitilde{}", "~"),
    (r"\textasciicircum{}", "^"),
    (r"\&", "&"),
    (r"\%", "%"),
    (r"\$", "$"),
    (r"\#", "#"),
    (r"\_", "_"),
]


def _braced(text: str, pos: int) -> tuple[str, int]:
    """The value of the brace group at *pos*, counting every brace as BibTeX does."""
    assert text[pos] == "{"
    depth, start = 0, pos
    while True:
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : pos], pos + 1
        pos += 1
        assert pos < len(text), "unbalanced braces"


def parse_bibtex(text: str) -> list[tuple[str, str, dict[str, str]]]:
    entries = []
    pos = 0
    while (at := text.find("@", pos)) >= 0:
        match = _ENTRY_START.match(text, at)
        assert match, text[at : at + 40]
        kind, key = match.groups()
        pos = match.end()
        fields: dict[str, str] = {}
        while True:
            while text[pos] in " \n\t,":
                pos += 1
            if text[pos] == "}":
                pos += 1
                break
            name_match = _FIELD_NAME.match(text, pos)
            assert name_match, text[pos : pos + 40]
            name = name_match.group(1)
            pos = name_match.end()
            if text[pos] == "{":
                value, pos = _braced(text, pos)
            else:
                bare = re.match(r"[a-z0-9]+", text[pos:])
                assert bare, text[pos : pos + 20]
                value = bare.group(0)
                pos += len(value)
            assert name not in fields, f"duplicate field {name}"
            fields[name] = value
        entries.append((kind, key, fields))
    return entries


def unescape_bibtex(value: str) -> str:
    for escaped, plain in _UNESCAPE:
        value = value.replace(escaped, plain)
    return value


def bibtex_names(value: str) -> list[str]:
    """Split an author field on top-level " and " (as BibTeX does)."""
    names, depth, current = [], 0, ""
    tokens = re.split(r"(\s+and\s+|[{}])", value)
    for token in tokens:
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
        if depth == 0 and re.fullmatch(r"\s+and\s+", token or ""):
            names.append(current)
            current = ""
        else:
            current += token
    names.append(current)
    return names


_RIS_LINE = re.compile(r"^[A-Z][A-Z0-9]  - .*$")


def assert_ris_grammar(text: str) -> list[list[tuple[str, str]]]:
    """Every line a "TAG  - value" CRLF line; each record TY ... ER; returns the
    records as (tag, value) lists."""
    assert "\n" not in text.replace("\r\n", ""), "a bare LF or CR"
    assert text.endswith("\r\n")
    records, current = [], None
    for line in text.split("\r\n"):
        if line == "":
            assert current is None, "blank line inside a record"
            continue
        assert _RIS_LINE.match(line) or line == "ER  - ", line
        tag, value = line[:2], line[6:]
        if tag == "TY":
            assert current is None, "TY inside a record"
            current = []
        assert current is not None, f"{tag} outside a record"
        if tag == "ER":
            assert value == ""
            records.append(current)
            current = None
            continue
        assert value and value == value.strip(), line
        current.append((tag, value))
    assert current is None, "record without ER"
    return records


# CSL 1.0.2 item types and the variables these writers may use.
_CSL_TYPES = {
    "article",
    "article-journal",
    "book",
    "chapter",
    "dataset",
    "document",
    "paper-conference",
    "report",
    "software",
    "thesis",
}
_CSL_STRINGS = {
    "id",
    "citation-key",
    "type",
    "title",
    "container-title",
    "collection-title",
    "publisher",
    "edition",
    "version",
    "volume",
    "issue",
    "page",
    "status",
    "DOI",
    "URL",
    "PMID",
    "note",
}
_CSL_NAMES = {"author", "editor"}
_CSL_DATES = {"issued", "accessed"}


def assert_csl_items(items: list[dict[str, Any]]) -> None:
    json.dumps(items)  # serializable
    ids = [item["id"] for item in items]
    assert len(set(ids)) == len(ids), "ids must be unique"
    for item in items:
        assert item["type"] in _CSL_TYPES
        for key, value in item.items():
            if key in _CSL_STRINGS:
                assert isinstance(value, str) and value, (key, value)
            elif key in _CSL_NAMES:
                assert isinstance(value, list) and value
                for name in value:
                    assert set(name) <= {"family", "given", "suffix", "literal"}, name
                    assert ("literal" in name) != ("family" in name or "given" in name), name
                    assert all(isinstance(v, str) and v for v in name.values())
            elif key in _CSL_DATES:
                if "raw" in value:
                    assert set(value) == {"raw"}
                    continue
                assert set(value) == {"date-parts"}
                (parts,) = value["date-parts"]
                assert 1 <= len(parts) <= 3 and all(isinstance(p, int) for p in parts)
            else:
                raise AssertionError(f"unexpected CSL variable {key!r}")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

_TYPE_TABLE = {
    "journal_article": ("article-journal", "article", "JOUR"),
    "book": ("book", "book", "BOOK"),
    "book_chapter": ("chapter", "incollection", "CHAP"),
    "conference_paper": ("paper-conference", "inproceedings", "CPAPER"),
    "report": ("report", "techreport", "RPRT"),
    "thesis": ("thesis", "phdthesis", "THES"),
    "dataset": ("dataset", "misc", "DATA"),
    "software": ("software", "misc", "COMP"),
    "preprint": ("article", "misc", "UNPB"),
    "other": ("document", "misc", "GEN"),
}


def test_type_table_covers_every_bib_type():
    assert set(_TYPE_TABLE) == set(get_args(BibTypeLiteral))


@pytest.mark.parametrize(("bib_type", "expected"), sorted(_TYPE_TABLE.items()))
def test_every_bib_type_maps_to_each_format(bib_type, expected):
    data = _export([_bib(bib_type=bib_type, publisher="Some Press", series="A Series")])
    csl_type, bibtex_type, ris_type = expected
    item = _one_csl(data)
    assert item["type"] == csl_type
    assert_csl_items([item])
    ((kind, _key, fields),) = parse_bibtex(to_bibtex(data))
    assert kind == bibtex_type
    assert unescape_bibtex(fields["title"]) == "A Prior Study"
    ((first, *_rest),) = assert_ris_grammar(to_ris(data))
    assert first == ("TY", ris_type)


def test_container_field_follows_the_bibtex_type():
    rows = [
        _bib(1, bib_type="journal_article"),
        _bib(2, bib_type="book_chapter", container="Handbook of Things"),
        _bib(3, bib_type="conference_paper", container="Proc. Things"),
        _bib(4, bib_type="report", container="Report Series", publisher="Institute"),
        _bib(5, bib_type="thesis", container=None, publisher="University of Things"),
        _bib(6, bib_type="dataset", container="Zenodo"),
        _bib(7, bib_type="book", container="Lecture Notes"),
    ]
    entries = {key: fields for _, key, fields in parse_bibtex(to_bibtex(_export(rows)))}
    by_order = list(entries.values())
    assert by_order[0]["journal"] == "Journal of Things"
    assert by_order[1]["booktitle"] == "Handbook of Things"
    assert by_order[2]["booktitle"] == "Proc. Things"
    assert by_order[3]["institution"] == "Institute"
    assert by_order[3]["series"] == "Report Series"
    assert by_order[4]["school"] == "University of Things"
    assert by_order[5]["howpublished"] == "Zenodo"
    assert by_order[6]["series"] == "Lecture Notes"


def test_matched_record_type_wins_but_not_its_catch_all():
    data = _export(
        [_bib(1, bib_type="journal_article"), _bib(2, bib_type="book")],
        [_match(1, bib_type="book_chapter"), _match(2, bib_type="other")],
        [_check(1), _check(2)],
    )
    types = [item["type"] for item in to_csl_json(data)]
    assert types == ["chapter", "book"]


def test_untyped_reference_with_journal_locators_is_an_article():
    data = _export(
        [
            _bib(1, bib_type=None, volume="3"),
            _bib(2, bib_type=None, container=None),
        ]
    )
    assert [item["type"] for item in to_csl_json(data)] == ["article-journal", "document"]


def test_records_catch_all_type_does_not_hide_a_printed_journal():
    data = _export(
        [
            _bib(1, bib_type=None, container="Journal X", volume="3"),
            _bib(2, bib_type=None, container=None),
        ],
        [_match(1, bib_type="other"), _match(2, bib_type="other")],
        [_check(1), _check(2)],
    )
    assert [item["type"] for item in to_csl_json(data)] == ["article-journal", "document"]
    kinds = [kind for kind, _, _ in parse_bibtex(to_bibtex(data))]
    assert kinds == ["article", "misc"]
    assert [record[0] for record in assert_ris_grammar(to_ris(data))] == [
        ("TY", "JOUR"),
        ("TY", "GEN"),
    ]


# ---------------------------------------------------------------------------
# Pages, dates, identifiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "last", "expected"),
    [
        ("1267", "89", ("1267", "1289")),
        ("1267", "1289", ("1267", "1289")),
        ("123", "8", ("123", "128")),
        ("e0123456", None, ("e0123456", None)),
        ("S12", "S19", ("S12", "S19")),
        ("S12", "19", ("S12", "S19")),
        ("12", "12", ("12", None)),
        ("xii", "xv", ("xii", "xv")),
        ("120", "11", ("120", "11")),
        ("1299", "03", ("1299", "03")),
        (None, "89", (None, None)),
        ("S12", "e19", ("S12", "e19")),
    ],
)
def test_expand_page_range(first, last, expected):
    assert expand_page_range(first, last) == expected


def test_pages_in_each_format():
    data = _export([_bib(1, first_page="1267", last_page="89"), _bib(2, first_page="e0123456")])
    items = to_csl_json(data)
    assert [i.get("page") for i in items] == ["1267-1289", "e0123456"]
    entries = parse_bibtex(to_bibtex(data))
    assert [fields["pages"] for _, _, fields in entries] == ["1267--1289", "e0123456"]
    records = assert_ris_grammar(to_ris(data))
    assert [(dict(r).get("SP"), dict(r).get("EP")) for r in records] == [
        ("1267", "1289"),
        ("e0123456", None),
    ]


def test_dates_in_each_format():
    data = _export([_bib(published_date="2020-05-03", year=2020)])
    assert _one_csl(data)["issued"] == {"date-parts": [[2020, 5, 3]]}
    ((_, _, fields),) = parse_bibtex(to_bibtex(data))
    assert (fields["year"], fields["month"]) == ("2020", "may")
    ((record),) = assert_ris_grammar(to_ris(data))
    assert ("PY", "2020") in record
    assert ("DA", "2020/05/03") in record


def test_in_press():
    data = _export([_bib(year=None, is_in_press=True)])
    item = _one_csl(data)
    assert item["status"] == "in press"
    assert "issued" not in item
    assert item["id"] == "Smithinpress"
    ((_, _, fields),) = parse_bibtex(to_bibtex(data))
    assert fields["pubstate"] == "inpress"
    assert "year" not in fields
    ((record),) = assert_ris_grammar(to_ris(data))
    assert ("N1", "In press") in record


def test_in_press_reference_dated_by_its_matched_record_is_published():
    data = _export(
        [_bib(year=None, is_in_press=True)], [_match(year=2021)], [_check(status="matched")]
    )
    item = _one_csl(data)
    assert item["issued"] == {"date-parts": [[2021]]}
    assert "status" not in item


def test_identifiers():
    data = _export(
        [
            _bib(
                doi="https://doi.org/10.1000/ABC_1",
                url="https://example.org/a_b?x=1%20y#frag",
                pmid="12345678",
                arxiv="2101.00001",
                access_date="2021-04-05",
                edition="2nd",
                editors="Roe, R. (Ed.)",
            )
        ]
    )
    item = _one_csl(data)
    assert item["DOI"] == "10.1000/abc_1"
    assert item["URL"] == "https://example.org/a_b?x=1%20y#frag"
    assert item["PMID"] == "12345678"
    assert item["accessed"] == {"date-parts": [[2021, 4, 5]]}
    assert item["edition"] == "2nd"
    assert item["editor"] == [{"family": "Roe", "given": "R."}]
    assert "arXiv: 2101.00001" in item["note"]
    ((_, _, fields),) = parse_bibtex(to_bibtex(data))
    assert fields["doi"] == "10.1000/abc_1"  # verbatim: no \_
    assert fields["url"] == "https://example.org/a_b?x=1%20y#frag"
    assert fields["urldate"] == "2021-04-05"
    assert (fields["eprint"], fields["eprinttype"]) == ("2101.00001", "arxiv")
    assert fields["pmid"] == "12345678"
    assert fields["editor"] == "Roe, R."
    ((record),) = assert_ris_grammar(to_ris(data))
    assert ("DO", "10.1000/abc_1") in record
    assert ("ED", "Roe, R.") in record
    assert ("N1", "PMID: 12345678") in record


def test_printed_access_date_that_is_not_iso_stays_raw():
    data = _export([_bib(access_date="May 3, 2021")])
    assert _one_csl(data)["accessed"] == {"raw": "May 3, 2021"}
    ((_, _, fields),) = parse_bibtex(to_bibtex(data))
    assert "urldate" not in fields


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def test_et_al_becomes_and_others_in_bibtex_only():
    data = _export([_bib(authors="Smith, J., Doe, A., et al.")])
    assert _one_csl(data)["author"] == [
        {"family": "Smith", "given": "J."},
        {"family": "Doe", "given": "A."},
    ]
    ((_, _, fields),) = parse_bibtex(to_bibtex(data))
    assert bibtex_names(fields["author"]) == ["Smith, J.", "Doe, A.", "others"]
    ((record),) = assert_ris_grammar(to_ris(data))
    assert [v for t, v in record if t == "AU"] == ["Smith, J.", "Doe, A."]


def test_organization_is_one_name_everywhere():
    data = _export(
        [_bib(authors="World Health Organization & Department of Health and Human Services")]
    )
    item = _one_csl(data)
    assert item["author"] == [
        {"literal": "World Health Organization"},
        {"literal": "Department of Health and Human Services"},
    ]
    ((_, key, fields),) = parse_bibtex(to_bibtex(data))
    assert bibtex_names(fields["author"]) == [
        "{World Health Organization}",
        "{Department of Health and Human Services}",
    ]
    assert key == "WorldHealthOrganization2020"
    ((record),) = assert_ris_grammar(to_ris(data))
    assert [v for t, v in record if t == "AU"] == [
        "World Health Organization",
        "Department of Health and Human Services",
    ]


def test_unsplittable_list_stays_one_literal():
    data = _export([_bib(authors="张三, 李四")])
    assert _one_csl(data)["author"] == [{"literal": "张三, 李四"}]
    ((_, key, fields),) = parse_bibtex(to_bibtex(data))
    assert fields["author"] == "{张三, 李四}"
    assert key == "ref2020"


def test_given_name_first_list_is_one_name_per_person():
    data = _export([_bib(authors="Sepp Hochreiter and Jürgen Schmidhuber", year=1997)])
    assert _one_csl(data)["author"] == [
        {"family": "Hochreiter", "given": "Sepp"},
        {"family": "Schmidhuber", "given": "Jürgen"},
    ]
    ((_, key, fields),) = parse_bibtex(to_bibtex(data))
    assert key == "Hochreiter1997"
    assert bibtex_names(fields["author"]) == ["Hochreiter, Sepp", "Schmidhuber, Jürgen"]
    ((record),) = assert_ris_grammar(to_ris(data))
    assert [v for t, v in record if t == "AU"] == ["Hochreiter, Sepp", "Schmidhuber, Jürgen"]


def test_particles_and_suffixes_in_bibtex_and_ris():
    data = _export([_bib(authors="van der Berg, C., & King, M. L., Jr.")])
    ((_, _, fields),) = parse_bibtex(to_bibtex(data))
    assert bibtex_names(fields["author"]) == ["van der Berg, C.", "King, Jr., M. L."]
    ((record),) = assert_ris_grammar(to_ris(data))
    assert [v for t, v in record if t == "AU"] == ["van der Berg, C.", "King, M. L., Jr."]
    assert _one_csl(data)["author"][1] == {"family": "King", "given": "M. L.", "suffix": "Jr."}


def test_rule_repeats_the_previous_references_authors():
    data = _export(
        [
            _bib(1, authors="Smith, J., & Doe, A."),
            _bib(2, authors="——", year=2021),
            _bib(3, authors="———, & Roe, R.", year=2022),
        ]
    )
    authors = [item["author"] for item in to_csl_json(data)]
    assert authors[1] == authors[0]
    assert authors[2] == [{"family": "Smith", "given": "J."}, {"family": "Roe", "given": "R."}]


def test_matched_reference_takes_the_records_names():
    data = _export([_bib(authors="Smith, J., et al.")], [_match()], [_check()])
    item = _one_csl(data)
    assert item["author"] == [
        {"family": "Smith", "given": "Jane"},
        {"family": "Doe", "given": "Anna"},
    ]
    ((_, _, fields),) = parse_bibtex(to_bibtex(data))
    assert "others" not in fields["author"]


def test_record_organization_author_and_stray_and_are_handled():
    match = _match(
        author=[
            {"family": "Smith", "given": "Jane"},
            {"family": "And"},
            {"literal": "Open Science Collaboration"},
        ]
    )
    data = _export([_bib()], [match], [_check()])
    assert _one_csl(data)["author"] == [
        {"family": "Smith", "given": "Jane"},
        {"literal": "Open Science Collaboration"},
    ]


# ---------------------------------------------------------------------------
# Which values: printed first, a record only when the check matched
# ---------------------------------------------------------------------------


def test_printed_values_win_and_gaps_come_from_a_matched_record():
    data = _export(
        [_bib(volume="7", container=None, first_page=None)], [_match()], [_check(status="matched")]
    )
    item = _one_csl(data)
    assert item["volume"] == "7"  # printed
    assert item["container-title"] == "Journal of Things"  # gap filled
    assert item["page"] == "101-120"  # gap filled, both ends from the record
    assert item["DOI"] == "10.1234/prior"
    assert item["publisher"] == "Things Press"


def test_prefer_record_takes_a_matched_records_values():
    data = _export([_bib(volume="7", title="A prior study")], [_match()], [_check()])
    item = _one_csl(data, prefer="record")
    assert item["volume"] == "8"
    assert item["title"] == "A Prior Study"
    assert item["issue"] == "2"


def test_pages_are_never_mixed_from_both_sides():
    data = _export([_bib(first_page="100", last_page=None)], [_match()], [_check()])
    assert _one_csl(data)["page"] == "100"
    assert _one_csl(data, prefer="record")["page"] == "101-120"


@pytest.mark.parametrize(
    "status", ["fields_disagree", "doi_unregistered", "not_found", "unchecked"]
)
@pytest.mark.parametrize("prefer", ["printed", "record"])
def test_a_record_that_did_not_match_is_never_used(status, prefer):
    data = _export(
        [_bib(volume="7", container=None, doi="10.1234/printed")],
        [_match(title="Another Work", doi="10.9999/other")],
        [_check(status=status, service="crossref" if status == "fields_disagree" else None)],
    )
    item = _one_csl(data, prefer=prefer)
    assert item["volume"] == "7"
    assert "container-title" not in item
    assert "page" not in item
    assert item["DOI"] == "10.1234/printed"
    assert item["author"] == [{"family": "Smith", "given": "J."}, {"family": "Doe", "given": "A."}]


def test_record_placeholders_markup_and_doi_urls_are_not_taken():
    match = _match(
        volume="0",
        title="<i>Drosophila</i> genes when p < .05 and q > 3",
        url="http://dx.doi.org/10.1234/prior",
        first_page="0",
    )
    data = _export([_bib(volume=None, first_page=None)], [match], [_check()])
    item = _one_csl(data, prefer="record")
    assert "volume" not in item
    assert "page" not in item
    assert "URL" not in item
    assert item["title"] == "Drosophila genes when p < .05 and q > 3"


def test_export_without_bib_check_is_checked_first():
    agreeing = _export([_bib(container=None, volume=None)], [_match()])
    item = _one_csl(agreeing)
    assert item["container-title"] == "Journal of Things"
    assert "bib_check" not in agreeing
    other_work = _export(
        [_bib(container=None, title="Completely Different Title About Fish")],
        [_match(title="Neural Correlates of Something Else Entirely")],
    )
    assert "container-title" not in _one_csl(other_work)


def test_writers_never_change_the_export():
    data = json.loads(ENRICHED.read_text())
    before = copy.deepcopy(data)
    for prefer in ("printed", "record"):
        to_csl_json(data, prefer=prefer)
        to_bibtex(data, prefer=prefer)
        to_ris(data, prefer=prefer)
    no_check = copy.deepcopy(data)
    del no_check["bib_check"]
    snapshot = copy.deepcopy(no_check)
    to_csl_json(no_check)
    assert data == before
    assert no_check == snapshot


# ---------------------------------------------------------------------------
# Check notes
# ---------------------------------------------------------------------------


def test_check_notes_for_each_status():
    data = json.loads(ENRICHED.read_text())
    notes = [item.get("note") for item in to_csl_json(data)]
    assert notes[0].startswith("bibr reference check: retracted (the Crossref record lists")
    assert "2021-03-01" in notes[0]
    assert "https://doi.org/10.1234/prior.retraction" in notes[0]
    assert notes[1] == "bibr reference check: doi.org says the printed DOI does not exist"
    assert notes[2] == "bibr reference check: differs from the record its DOI names (year)"
    assert notes[3] == "bibr reference check: not found in Crossref"
    assert all("note" not in item for item in to_csl_json(data, check_note=False))


def test_fields_disagree_note_names_fields_and_retraction_wording():
    retraction = {"type": "withdrawal", "date": "2022", "doi": "10.1/w"}
    check = _check(
        status="fields_disagree",
        notes=[
            {"field": "title", "printed": "a", "record": "b", "severity": "major"},
            {"field": "first_page", "printed": "1", "record": "2", "severity": "minor"},
            {"field": "last_page", "printed": "3", "record": "4", "severity": "minor"},
        ],
        retraction=retraction,
    )
    data = _export([_bib(note="Original work published 1890")], [_match()], [check])
    assert _one_csl(data)["note"].split("\n") == [
        "Original work published 1890",
        "bibr reference check: differs from the Crossref record (title, pages)",
        "bibr reference check: the Crossref record matched to this reference lists a "
        "withdrawal notice dated 2022, https://doi.org/10.1/w",
    ]


def test_matched_and_unchecked_references_get_no_check_note():
    data = _export([_bib(1), _bib(2)], [_match(1)], [_check(1), _check(2, status="unchecked")])
    assert all("note" not in item for item in to_csl_json(data))


# ---------------------------------------------------------------------------
# Escaping and keys
# ---------------------------------------------------------------------------

_NASTY = r"Costs & 50% of $10 #1 a_b {braced} back\slash ~tilde ^caret"


def test_bibtex_escapes_round_trip():
    data = _export([_bib(title=_NASTY, container="J. {Odd}", note="line one\nline two")])
    text = to_bibtex(data, check_note=False)
    ((_, _, fields),) = parse_bibtex(text)
    assert unescape_bibtex(fields["title"]) == _NASTY
    assert unescape_bibtex(fields["journal"]) == "J. {Odd}"
    assert fields["note"] == "line one line two"


def test_bibtex_name_with_and_or_comma_is_braced():
    match = _match(author=[{"family": "Smith, Jr", "given": "J."}, {"family": "Anderson"}])
    data = _export([_bib()], [match], [_check()])
    ((_, _, fields),) = parse_bibtex(to_bibtex(data))
    assert bibtex_names(fields["author"]) == ["{Smith, Jr}, J.", "Anderson"]


def test_ris_folds_line_breaks_into_one_line():
    data = _export([_bib(title="A title\nwith a break\r\nand more", note="a\nb")])
    ((record),) = assert_ris_grammar(to_ris(data))
    assert ("TI", "A title with a break and more") in record


def test_cite_keys_are_deterministic_and_unique():
    rows = [
        _bib(1),
        _bib(2),
        _bib(3),
        _bib(4, authors="Müller, H.", year=2019),
        _bib(5, authors="Øster, K.", year=None),
        _bib(6, authors=None),
    ]
    keys = [item["id"] for item in to_csl_json(_export(rows))]
    assert keys == ["Smith2020", "Smith2020a", "Smith2020b", "Muller2019", "Osternd", "ref2020"]
    assert keys == [item["id"] for item in to_csl_json(_export(rows))]


def test_cite_keys_differing_only_in_case_do_not_collide():
    # BibTeX compares keys ignoring case: "BRASILnd" and "Brasilnd" are one key.
    rows = [
        _bib(1, authors="BRASIL", year=None, title="a"),
        _bib(2, authors="Brasil", year=None, title="b"),
        _bib(3, authors="team TACS", year=2018, title="c"),
        _bib(4, authors="Team TACS", year=2018, title="d"),
    ]
    text = to_bibtex(_export(rows))
    keys = [key for _, key, _ in parse_bibtex(text)]
    assert keys == ["BRASILnd", "Brasilnda", "teamTACS2018", "TeamTACS2018a"]
    assert len({key.casefold() for key in keys}) == len(keys)
    # The same across papers written to one file.
    one = _export([_bib(1, authors="BRASIL", year=None)], paper_id="a")
    two = _export([_bib(1, authors="Brasil", year=None)], paper_id="b")
    combined, _ = render_references([one, two], "bibtex")
    assert [key for _, key, _ in parse_bibtex(combined)] == ["BRASILnd", "Brasilnda"]


def test_cite_keys_are_unique_across_papers():
    one = _export([_bib(1)], paper_id="a")
    two = _export([_bib(1)], paper_id="b")
    text, count = render_references([one, two], "bibtex")
    assert count == 2
    assert [key for _, key, _ in parse_bibtex(text)] == ["Smith2020", "Smith2020a"]


# ---------------------------------------------------------------------------
# Whole exports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", [ENRICHED, INSPECT])
@pytest.mark.parametrize("prefer", ["printed", "record"])
def test_fixture_exports_are_well_formed(fixture, prefer):
    data = json.loads(fixture.read_text())
    items = to_csl_json(data, prefer=prefer)
    assert len(items) == len(data["bib"])
    assert_csl_items(items)
    entries = parse_bibtex(to_bibtex(data, prefer=prefer))
    assert len(entries) == len(data["bib"])
    assert [key for _, key, _ in entries] == [item["id"] for item in items]
    assert len(assert_ris_grammar(to_ris(data, prefer=prefer))) == len(data["bib"])


def test_export_without_references_gives_empty_files():
    data = _export([])
    assert to_csl_json(data) == []
    assert to_bibtex(data) == ""
    assert to_ris(data) == ""


def test_bad_options_raise():
    data = _export([_bib()])
    with pytest.raises(ValueError, match="prefer"):
        to_csl_json(data, prefer="registry")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown reference format"):
        render_references([data], "endnote")
    with pytest.raises(TypeError):
        to_bibtex(42)


def test_format_names():
    assert FORMATS == ("csl-json", "bibtex", "ris")
    assert set(CONTENT_TYPES) == set(FORMATS)
    assert normalize_format("BIB") == "bibtex"
    assert normalize_format("csl") == "csl-json"
    assert format_for_path("refs.BIB") == "bibtex"
    assert format_for_path("refs.ris") == "ris"
    assert format_for_path("refs.json") == "csl-json"
    assert format_for_path("refs.txt") is None


# ---------------------------------------------------------------------------
# Result methods and write_references
# ---------------------------------------------------------------------------


def test_result_methods():
    result = Result(json.loads(INSPECT.read_text()))
    items = result.to_csl_json()
    assert [i["id"] for i in items] == ["Smith2020", "Lee2021"]
    assert items[0]["DOI"] == "10.1000/aaa"
    assert result.to_bibtex().startswith("@article{Smith2020,")
    assert result.to_ris().startswith("TY  - JOUR\r\n")
    assert result.to_csl_json(prefer="record", check_note=False)[0]["volume"] == "1"


@pytest.mark.parametrize(
    ("suffix", "fmt"), [(".bib", "bibtex"), (".ris", "ris"), (".json", "csl-json")]
)
def test_write_references_infers_the_format(tmp_path, suffix, fmt):
    out = tmp_path / "sub" / f"refs{suffix}"
    report = bibr.write_references(Result(json.loads(ENRICHED.read_text())), out)
    assert (report.format, report.references, report.papers) == (fmt, 4, 1)
    raw = out.read_bytes()
    if fmt == "ris":
        assert raw.count(b"\r\n") > 0 and b"\r\r\n" not in raw
        assert_ris_grammar(raw.decode())
    elif fmt == "csl-json":
        assert_csl_items(json.loads(raw))
    else:
        assert len(parse_bibtex(raw.decode())) == 4


def test_write_references_reads_paths_and_skips_what_is_no_export(tmp_path):
    good = tmp_path / "a.json"
    good.write_text(ENRICHED.read_text())
    other = tmp_path / "run_info.json"
    other.write_text(json.dumps({"started": "now"}))
    out = tmp_path / "refs.txt"
    report = write_references([good, other], out, format="bib")
    assert report.format == "bibtex"
    assert report.papers == 1
    assert report.skipped == ((str(other), "not a bibr export (no root schema_version)"),)
    assert len(parse_bibtex(out.read_text())) == 4


def test_write_references_needs_a_format(tmp_path):
    with pytest.raises(ValueError, match="cannot tell the format"):
        write_references(_export([_bib()]), tmp_path / "refs.txt")


def test_write_references_file_mode_follows_the_umask(tmp_path):
    mask = os.umask(0o022)
    try:
        out = write_references(_export([_bib()]), tmp_path / "refs.bib").path
    finally:
        os.umask(mask)
    assert out.stat().st_mode & 0o777 == 0o644


def test_failed_write_keeps_the_old_file(tmp_path, monkeypatch):
    out = tmp_path / "refs.bib"
    out.write_text("old")

    def broken_replace(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        write_references(_export([_bib()]), out)
    assert out.read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["refs.bib"]
