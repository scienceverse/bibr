"""Comparing exports by content: run stamps and score noise aside."""

from __future__ import annotations

import copy
import json

import pytest

from bibr.export.compare import (
    DEFAULT_FLOAT_TOLERANCE,
    Difference,
    canonical_export,
    compare_export_dirs,
    diff_exports,
    main,
)


def _export(**changes) -> dict:
    doc = {
        "schema_version": "12.0",
        "paper_id": "paper",
        "metadata": {"title": "A title", "abstract": "An abstract."},
        "author": [
            {"author_id": 1, "given": "Ada", "family": "Lovelace"},
            {"author_id": 2, "given": "Alan", "family": "Turing"},
        ],
        "bib": [{"bib_id": 1, "title": "One", "year": 1936}],
        "extraction": {
            "completed_at": "2026-11-10T21:02:40Z",
            "producer": {"name": "bibr", "version": "0.9.0", "build_sha": "abc"},
            "timings": {"stages": {"layout": 2.7}, "total_seconds": 13.0},
            "settings": {"mode": "fast"},
            "warnings": [{"code": "REF_SEG_GEOM_CASCADE", "message": "10 < 12"}],
            "diagnostics": {
                "paper_classification": {"oecd_confidence": 0.8699201941490173},
                "section_classification": [{"section_id": 1, "score": 0.5729027986526489}],
            },
        },
    }
    doc.update(changes)
    return doc


def _rerun(doc: dict) -> dict:
    """The same paper read again: later, slower, by another build stamp."""
    again = copy.deepcopy(doc)
    again["extraction"]["completed_at"] = "2026-11-11T00:36:58Z"
    again["extraction"]["timings"] = {"stages": {"layout": 4.1}, "total_seconds": 46.7}
    again["extraction"]["producer"]["build_sha"] = None
    return again


def test_a_rerun_is_the_same_export():
    first = _export()
    assert diff_exports(first, _rerun(first)) == []


def test_canonical_export_drops_the_run_stamps_and_copies():
    first = _export()
    canonical = canonical_export(first)

    assert set(first["extraction"]) - set(canonical["extraction"]) == {
        "completed_at",
        "producer",
        "timings",
    }
    assert canonical["extraction"]["settings"] == {"mode": "fast"}
    canonical["author"].clear()
    assert len(first["author"]) == 2  # the input is not touched


def test_canonical_export_takes_an_export_with_no_extraction_record():
    assert canonical_export({"schema_version": "12.0", "extraction": None}) == {
        "schema_version": "12.0",
        "extraction": None,
    }


def test_scores_that_differ_in_the_last_digits_are_one_score():
    first = _export()
    other_cpu = _rerun(first)
    scores = other_cpu["extraction"]["diagnostics"]
    scores["paper_classification"]["oecd_confidence"] += 1.6e-6
    scores["section_classification"][0]["score"] -= 1.6e-6

    assert diff_exports(first, other_cpu) == []
    assert diff_exports(first, other_cpu, float_tolerance=0) == [
        Difference(".extraction.diagnostics.paper_classification.oecd_confidence", "number"),
        Difference(".extraction.diagnostics.section_classification[].score", "number"),
    ]


def test_a_score_that_moves_is_reported():
    first = _export()
    moved = _rerun(first)
    moved["extraction"]["diagnostics"]["section_classification"][0]["score"] += (
        2 * DEFAULT_FLOAT_TOLERANCE
    )

    assert diff_exports(first, moved) == [
        Difference(".extraction.diagnostics.section_classification[].score", "number")
    ]


def test_whole_numbers_are_never_within_tolerance():
    first = _export()
    other = _rerun(first)
    other["bib"][0]["year"] = 1937

    assert diff_exports(first, other, float_tolerance=5) == [Difference(".bib[].year", "number")]


def test_a_whole_number_and_its_float_are_equal():
    assert diff_exports({"width": 612}, {"width": 612.0}) == []


def test_a_boolean_is_not_a_number():
    assert diff_exports({"flag": True}, {"flag": 1}) == [Difference(".flag", "type")]


def test_a_changed_value_names_its_place_without_row_numbers():
    first = _export()
    other = _rerun(first)
    other["author"][1]["family"] = "Turing "
    other["metadata"]["title"] = None

    assert diff_exports(first, other) == [
        Difference(".author[].family", "value"),
        Difference(".metadata.title", "type"),
    ]


def test_an_added_row_is_one_difference():
    first = _export()
    other = _rerun(first)
    other["author"].insert(0, {"author_id": 0, "given": "Grace", "family": "Hopper"})

    assert diff_exports(first, other) == [Difference(".author[]", "length")]


def test_a_key_one_side_lacks_is_missing():
    first = _export()
    other = _rerun(first)
    del other["metadata"]["abstract"]
    other["extraction"]["warnings"][0]["page"] = 3

    assert diff_exports(first, other) == [
        Difference(".extraction.warnings[].page", "missing"),
        Difference(".metadata.abstract", "missing"),
    ]


def _write(folder, name: str, doc) -> None:
    folder.mkdir(exist_ok=True)
    (folder / name).write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture
def two_runs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for name in ("one.json", "two.json", "three.json"):
        doc = _export(paper_id=name)
        _write(a, name, doc)
        _write(b, name, _rerun(doc))
    # What a batch writes beside the exports, and a file that is no JSON at all.
    _write(a, "run_info.json", {"run_id": "1", "started_at": "then"})
    _write(b, "run_info.json", {"run_id": "2", "started_at": "now"})
    (a / "broken.json").write_text("{", encoding="utf-8")
    return a, b


def test_two_runs_of_the_same_papers_compare_the_same(two_runs):
    a, b = two_runs
    result = compare_export_dirs(a, b)

    assert (result.papers, result.identical, result.same) == (3, 3, True)
    assert result.by_path() == []


def test_differences_are_counted_per_place(two_runs):
    a, b = two_runs
    for name in ("one.json", "two.json"):
        doc = json.loads((b / name).read_text(encoding="utf-8"))
        doc["author"][0]["given"] = "A."
        doc["author"][1]["given"] = "A."
        if name == "one.json":
            doc["extraction"]["warnings"] = []
        _write(b, name, doc)
    only_warnings = json.loads((b / "three.json").read_text(encoding="utf-8"))
    only_warnings["extraction"]["warnings"][0]["message"] = "11 < 12"
    _write(b, "three.json", only_warnings)

    result = compare_export_dirs(a, b)

    assert (result.papers, result.identical, result.same) == (3, 0, False)
    assert result.changed_outside_extraction == ["one.json", "two.json"]
    assert result.by_path() == [
        (".author[].given", "value", 2, 4),
        (".extraction.warnings[]", "length", 1, 1),
        (".extraction.warnings[].message", "value", 1, 1),
    ]
    summary = result.to_dict()
    assert summary["changed"] == 3
    assert summary["changed_outside_extraction"] == 2
    assert summary["changed_papers"]["three.json"] == [".extraction.warnings[].message"]


def test_a_paper_in_one_folder_only_is_not_the_same(two_runs):
    a, b = two_runs
    (b / "three.json").unlink()
    _write(b, "four.json", _export(paper_id="four"))

    result = compare_export_dirs(a, b)

    assert (result.papers, result.identical) == (2, 2)
    assert (result.only_a, result.only_b) == (["three.json"], ["four.json"])
    assert not result.same


def test_the_command_exits_zero_for_the_same_content(two_runs, capsys):
    a, b = two_runs

    assert main([str(a), str(b)]) == 0
    assert "3 papers in both folders: 3 the same, 0 differ" in capsys.readouterr().out


def test_the_command_exits_one_and_lists_what_differs(two_runs, capsys):
    a, b = two_runs
    doc = json.loads((b / "two.json").read_text(encoding="utf-8"))
    doc["bib"][0]["title"] = "Won"
    _write(b, "two.json", doc)

    assert main([str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "2 the same, 1 differ (1 outside the extraction record)" in out
    assert ".bib[].title" in out

    assert main([str(a), str(b), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["by_path"] == [
        {"path": ".bib[].title", "kind": "value", "papers": 1, "places": 1}
    ]


def test_the_command_refuses_a_folder_with_no_exports(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(SystemExit) as stop:
        main([str(empty), str(empty)])
    assert stop.value.code == 2
    assert "no bibr exports" in capsys.readouterr().err

    with pytest.raises(SystemExit) as stop:
        main([str(empty), str(tmp_path / "absent")])
    assert stop.value.code == 2
