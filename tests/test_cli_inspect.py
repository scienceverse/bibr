"""Tests for ``bibr inspect`` — a human-readable summary of a bibr JSON export.

Fixtures:
  - ``fixtures/inspect_full_export.json``: a hand-built, schema-faithful v10.6
    export (verified against ``bibr.export.json_export.PaperExport`` in this
    test module's own setup) covering every block ``bibr inspect`` reports on.
  - ``fixtures/inspect_degraded_export.json``: only ``info`` + ``author`` —
    every other block is entirely absent, exercising the degrade contract
    (missing block -> placeholder, exit 0).

Two more contracts are covered without dedicated fixture files: a file that
isn't JSON, and JSON that is clearly not a bibr export (no top-level ``info``
object) — both must exit 1 with a clear one-line message, never a traceback.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
FULL = FIXTURES / "inspect_full_export.json"
DEGRADED = FIXTURES / "inspect_degraded_export.json"

# Local-only real-world corpus export (736 KB of a copyrighted paper's full
# extracted text) — not committed, and slated for removal by the SCI-120
# history purge. Only present on machines that still have the local `data/`
# corpus checked out; see test_synthetic_pre_v10_3_export_without_schema_
# version_exits_zero below for the always-running, fixture-free regression
# coverage of the same code path.
_REAL_PRE_V10_3_EXPORT = (
    Path(__file__).parent.parent
    / "data"
    / "test_results"
    / "arxiv_platform"
    / "attention_is_all_you_need.json"
)


def test_full_fixture_validates_against_export_schema():
    """Guard against the fixture silently drifting from the real schema."""
    import json

    from bibr.export.json_export import PaperExport

    data = json.loads(FULL.read_text())
    PaperExport.model_validate(data)  # raises on drift


# --- full export: report contents -------------------------------------------


def test_full_export_exits_zero(capsys):
    from bibr.local.inspect import run_inspect

    code = run_inspect(str(FULL))

    assert code == 0
    capsys.readouterr()


def test_full_export_title_doi_paper_type(capsys):
    from bibr.local.inspect import run_inspect

    run_inspect(str(FULL))
    out = capsys.readouterr().out

    assert "Deep Learning for Something Great" in out
    assert "10.1234/example.5678" in out
    assert "empirical" in out


def test_full_export_authors_count_and_first_three(capsys):
    from bibr.local.inspect import run_inspect

    run_inspect(str(FULL))
    out = capsys.readouterr().out

    assert "Authors: 4" in out
    assert "Jane Doe" in out
    assert "John Smith" in out
    assert "Alice Wong" in out
    # Only the first 3 are named; the 4th is folded into a "+N more" marker.
    assert "Bob Lee" not in out
    assert "+1 more" in out


def test_full_export_structure_counts(capsys):
    from bibr.local.inspect import run_inspect

    run_inspect(str(FULL))
    out = capsys.readouterr().out

    assert "Sections: 3" in out
    assert "Sentences: 5" in out
    assert "Tables: 1" in out
    assert "Figures: 1" in out
    assert "Equations: 1" in out


def test_full_export_references_block(capsys):
    """Bib count from ``bib``; in-text citation coverage from ``xref``
    (xref_type == "bib": N = count of such entries, M = count of distinct
    non-null xref_id values among them, K = ``len(bib)``) — this is the
    truthful replacement for the old "matched/total over the xref list"
    metric, which read ~100% on every real export because the production
    citation linker never stores an xref with a null xref_id (unresolved
    candidates are dropped before an xref entry is ever created, see
    bibr/structure/citation_linker.py). Enrichment state from the dedicated
    ``enrichment`` block — see bibr/export/json_export.py:467-474 (xref),
    :223-257 (bib), :426-435 (enrichment)."""
    from bibr.local.inspect import run_inspect

    run_inspect(str(FULL))
    out = capsys.readouterr().out

    assert "Bibliography entries: 2" in out
    # Fixture: 2 xref entries of type "bib", both non-null xref_id=1 (bib_id 2
    # is never cited in text) -> N=2 linked, M=1 of K=2 references cited.
    assert "In-text citations: 2 linked → 1 of 2 references cited (50%)" in out
    assert "complete" in out
    assert "1/2 refs enriched" in out


def test_full_export_validation_block(capsys):
    from bibr.local.inspect import run_inspect

    run_inspect(str(FULL))
    out = capsys.readouterr().out

    assert "VAL_XREF_ZERO" in out
    assert "low xref count" in out
    # non-VALIDATION-prefixed processing_warnings entry folded in by Task 5's
    # _format_validation_line.
    assert "processing warnings" in out


def test_full_export_llm_usage_table(capsys):
    from bibr.local.inspect import run_inspect

    run_inspect(str(FULL))
    out = capsys.readouterr().out

    assert "gemini-2.5-flash" in out
    assert "in=12,345" in out
    assert "out=678" in out
    assert "total=13,023" in out
    # No price table exists in the codebase — tokens only, never a $ figure.
    assert "$" not in out


# --- degraded export: placeholders, still exit 0 -----------------------------


def test_degraded_export_exits_zero(capsys):
    from bibr.local.inspect import run_inspect

    code = run_inspect(str(DEGRADED))

    assert code == 0
    capsys.readouterr()


def test_degraded_export_renders_info_and_author(capsys):
    from bibr.local.inspect import run_inspect

    run_inspect(str(DEGRADED))
    out = capsys.readouterr().out

    assert "Authors: 1" in out
    assert "Only Author" in out


def test_degraded_export_missing_blocks_are_placeholders(capsys):
    """Every block absent from the degraded fixture (text, section, table,
    figure, eq, bib, xref, bib_match, enrichment, validation, llm_usage) must
    render as a placeholder, never a crash and never a bare 0 that would be
    indistinguishable from "present but empty"."""
    from bibr.local.inspect import run_inspect

    run_inspect(str(DEGRADED))
    out = capsys.readouterr().out

    assert "—" in out or "not present" in out
    assert "Title: —" in out
    assert "DOI: —" in out
    assert "Paper type: —" in out
    assert "Sections: —" in out
    assert "Sentences: —" in out
    assert "Tables: —" in out
    assert "Figures: —" in out
    assert "Equations: —" in out
    assert "Bibliography entries: —" in out


# --- error paths: exit 1, no traceback ---------------------------------------


def test_missing_file_exits_one(tmp_path, capsys):
    from bibr.local.inspect import run_inspect

    missing = tmp_path / "nope.json"
    code = run_inspect(str(missing))

    assert code == 1
    err = capsys.readouterr().err
    assert "nope.json" in err


def test_invalid_json_exits_one(tmp_path, capsys):
    from bibr.local.inspect import run_inspect

    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not valid json ][")
    code = run_inspect(str(bad))

    assert code == 1
    err = capsys.readouterr().err
    assert "not valid JSON" in err or "JSON" in err


def test_non_bibr_json_object_exits_one(tmp_path, capsys):
    """Valid JSON, but clearly not a bibr export (no top-level ``info``)."""
    import json

    from bibr.local.inspect import run_inspect

    other = tmp_path / "package.json"
    other.write_text(json.dumps({"name": "not-bibr", "version": "1.0.0"}))
    code = run_inspect(str(other))

    assert code == 1
    err = capsys.readouterr().err
    assert "bibr export" in err.lower()


def test_non_bibr_json_array_exits_one(tmp_path, capsys):
    import json

    from bibr.local.inspect import run_inspect

    other = tmp_path / "list.json"
    other.write_text(json.dumps([1, 2, 3]))
    code = run_inspect(str(other))

    assert code == 1
    capsys.readouterr()


def test_binary_file_exits_one_no_traceback(tmp_path, capsys):
    """A non-UTF-8/binary file (e.g. the user pointed ``bibr inspect`` at the
    wrong file entirely) must degrade the same as any other invalid input:
    exit 1, a single clear stderr line, never a raw ``UnicodeDecodeError``
    traceback."""
    from bibr.local.inspect import run_inspect

    binary = tmp_path / "binary_garbage.bin"
    binary.write_bytes(b"\x80\x81\xfe")

    code = run_inspect(str(binary))

    assert code == 1
    out, err = capsys.readouterr()
    assert out == ""
    err_lines = [line for line in err.splitlines() if line.strip()]
    assert len(err_lines) == 1
    assert "binary_garbage.bin" in err_lines[0]
    assert "valid" in err_lines[0].lower()
    assert "Traceback" not in err


def test_openapi_shaped_json_exits_one(tmp_path, capsys):
    """A JSON object with a top-level ``info`` dict is not automatically a
    bibr export — an OpenAPI spec is the canonical example of unrelated JSON
    that happens to use the same top-level key."""
    import json

    from bibr.local.inspect import run_inspect

    openapi = tmp_path / "openapi.json"
    openapi.write_text(
        json.dumps({"info": {"title": "x", "version": "1"}, "openapi": "3.0", "paths": {}})
    )

    code = run_inspect(str(openapi))

    assert code == 1
    err = capsys.readouterr().err
    assert "bibr export" in err.lower()


@pytest.mark.skipif(
    not _REAL_PRE_V10_3_EXPORT.exists(),
    reason="local corpus export not present (purged)",
)
def test_real_pre_v10_3_export_still_exits_zero():
    """``attention_is_all_you_need.json`` predates ``schema_version`` but its
    ``info`` block still carries ``bibr_version``/``input_format``/
    ``file_hash`` — the tightened export-detection heuristic must still
    accept it (guards against I1's fix being overly strict).

    This file is a local-only real-world corpus export, not committed to the
    repo, and is expected to disappear once the SCI-120 history purge runs —
    see test_synthetic_pre_v10_3_export_without_schema_version_exits_zero for
    fixture-free coverage of the same code path that keeps running after
    that."""
    from bibr.local.inspect import run_inspect

    real_export = _REAL_PRE_V10_3_EXPORT
    assert real_export.exists(), "expected real pre-v10.3 fixture export to exist in data/"

    code = run_inspect(str(real_export))

    assert code == 0


def test_synthetic_pre_v10_3_export_without_schema_version_exits_zero(tmp_path):
    """Fixture-free, always-running regression coverage for the same code
    path ``test_real_pre_v10_3_export_still_exits_zero`` exercises: a v10.x
    export whose ``info`` block predates ``schema_version`` (but still
    carries ``bibr_version``/``input_format``, like
    ``attention_is_all_you_need.json``) must still be recognized as a bibr
    export by ``_looks_like_bibr_export`` and exit 0.

    Built by deep-copying the redistributable, schema-validated
    ``inspect_full_export.json`` fixture and deleting ``schema_version`` —
    no copyrighted paper text involved, so this survives the SCI-120 history
    purge that may remove the real corpus export above."""
    import copy
    import json

    from bibr.local.inspect import run_inspect

    data = copy.deepcopy(json.loads(FULL.read_text()))
    assert "schema_version" in data["info"]  # sanity: fixture actually has it to remove
    del data["info"]["schema_version"]
    # Sanity: the other version/format fields the real pre-v10.3 file still
    # carries must remain, so the gate is exercised via the *same* tolerance
    # path (bibr_version/input_format), not accidentally via schema_version.
    assert "bibr_version" in data["info"]
    assert "input_format" in data["info"]

    synthetic = tmp_path / "pre_v10_3_export.json"
    synthetic.write_text(json.dumps(data))

    code = run_inspect(str(synthetic))

    assert code == 0


# --- crash-safety: malformed-but-parseable payloads must not traceback ------


def test_malformed_validation_and_llm_usage_do_not_crash(tmp_path, capsys):
    """Non-dict issues, non-int counts, wrong-typed llm_usage — the report
    must degrade gracefully (inherits Task 5's hardened validation helpers),
    never raise, and still exit 0 for an otherwise-parseable export."""
    import json

    from bibr.local.inspect import run_inspect

    payload = {
        "info": {"title": "x"},
        "author": "not-a-list",
        "validation": {"errors": "bad", "warnings": None, "issues": "nope"},
        "llm_usage": {"model-a": "not-a-dict", "model-b": {"input_tokens": "x"}},
    }
    path = tmp_path / "weird.json"
    path.write_text(json.dumps(payload))

    code = run_inspect(str(path))

    assert code == 0
    capsys.readouterr()


# --- CLI wiring ---------------------------------------------------------------


def test_cli_parses_inspect_subcommand_with_positional():
    from bibr.local.cli import _build_parser

    args = _build_parser().parse_args(["inspect", str(FULL)])

    assert args.command == "inspect"
    assert args.json_file == str(FULL)


def test_cli_dispatch_calls_run_inspect(monkeypatch):
    """``bibr inspect <file>`` dispatches to ``run_inspect`` and exits with
    its return code."""
    import sys

    from bibr.local import cli

    called = {}

    def _fake_run_inspect(path):
        called["path"] = path
        return 0

    monkeypatch.setattr("bibr.local.inspect.run_inspect", _fake_run_inspect)
    monkeypatch.setattr(sys, "argv", ["bibr", "inspect", str(FULL)])

    try:
        cli.main()
    except SystemExit as e:
        assert e.code == 0
    else:
        raise AssertionError("main() should have called sys.exit(0)")

    assert called["path"] == str(FULL)
