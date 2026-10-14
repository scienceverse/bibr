"""``bibr references``: reference lists of saved exports as CSL-JSON, BibTeX or RIS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from bibr.local.cli.parser import _build_parser
from bibr.local.cli.references import run_references

ENRICHED = (
    Path(__file__).parent / "fixtures" / "schema_conformance" / "valid" / "enriched_check.json"
)
INSPECT = Path(__file__).parent / "fixtures" / "inspect_full_export.json"


def _run(*argv: str) -> int:
    return run_references(_build_parser().parse_args(["references", *argv]))


@pytest.fixture
def results(tmp_path: Path) -> Path:
    folder = tmp_path / "results"
    folder.mkdir()
    (folder / "a.json").write_text(ENRICHED.read_text())
    (folder / "b.json").write_text(INSPECT.read_text())
    # What 'bibr chew -o' writes beside an export is not read.
    (folder / "b.json.receipt.json").write_text(json.dumps({"paper_id": "x"}))
    return folder


def test_writes_the_format_its_extension_names(results, tmp_path, capsys):
    out = tmp_path / "refs.bib"
    assert _run(str(results), "-o", str(out)) == 0
    text = out.read_text()
    assert text.count("\n@") + text.startswith("@") == 6
    assert "6 references from 2 papers" in capsys.readouterr().err


def test_csl_json_written_beside_the_exports_is_not_read_back(results, capsys):
    out = results / "refs.json"
    assert _run(str(results), "-o", str(out)) == 0
    assert _run(str(results), "-o", str(out)) == 0
    err = capsys.readouterr().err
    assert "skipped" not in err
    assert len(json.loads(out.read_text())) == 6


def test_standard_output_needs_and_takes_a_format(results, capsysbinary):
    assert _run(str(results)) == 2
    assert _run(str(results), "--format", "ris") == 0
    out = capsysbinary.readouterr().out
    assert out.startswith(b"TY  - JOUR\r\n")
    assert b"\r\r\n" not in out
    assert out.count(b"ER  - \r\n") == 6


def test_options_reach_the_writer(results, tmp_path):
    out = tmp_path / "refs.txt"
    assert (
        _run(
            str(results / "a.json"),
            "--format",
            "csl-json",
            "-o",
            str(out),
            "--prefer",
            "record",
            "--no-check-note",
        )
        == 0
    )
    items = json.loads(out.read_text())
    assert items[0]["volume"] == "8"  # the matched record's, printed "7"
    assert all("note" not in item for item in items)


def test_unknown_extension_and_no_exports_fail(results, tmp_path, capsys):
    assert _run(str(results), "-o", str(tmp_path / "refs.txt")) == 2
    assert "Cannot tell the format" in capsys.readouterr().err
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _run(str(empty), "-o", str(tmp_path / "refs.bib")) == 2
    assert "No JSON exports found" in capsys.readouterr().err


def test_parser_defaults_and_unknown_formats():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["references", "x.json", "--format", "endnote"])
    args = parser.parse_args(["references", "x.json"])
    assert (args.format, args.out, args.prefer, args.no_check_note) == (
        None,
        None,
        "printed",
        False,
    )


def test_mcp_help_names_the_reference_tools():
    parser = _build_parser()
    commands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    mcp = commands.choices["mcp"]
    text = " ".join(mcp.format_help().split())
    assert "export_references" in text
    assert "save_references" in text
    allow_dir = next(a for a in mcp._actions if "--allow-dir" in a.option_strings)
    assert "save_references" in allow_dir.help
