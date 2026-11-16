"""``bibr tables`` / ``bibr references`` on batch output, on papers and on non-exports.

bibr's own files beside the exports (run_info.json, validation_report.json) are
skipped without a warning; a PDF given in place of an export is named as such;
inputs that hold no export at all exit 1 and write nothing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from bibr.export.citations import write_references
from bibr.export.json_export import _export_paper_payload
from bibr.export.tables import NoExportsError, write_tables
from bibr.local.cli.references import run_references
from bibr.local.cli.tables import run_tables


@pytest.fixture
def payload(demo_paper) -> dict:
    return _export_paper_payload(demo_paper)


@pytest.fixture
def batch_out(tmp_path, payload):
    out = tmp_path / "results"
    out.mkdir()
    (out / "a.json").write_text(json.dumps(payload))
    (out / "run_info.json").write_text(json.dumps({"run_id": "abc", "n_ok": 1}))
    (out / "validation_report.json").write_text(json.dumps({"noise": True}))
    return out


def _refs_args(inputs, out, fmt=None):
    return SimpleNamespace(
        inputs=[str(i) for i in inputs],
        out=str(out),
        format=fmt,
        prefer="printed",
        no_check_note=False,
    )


def test_tables_skip_bibrs_own_files_silently(batch_out, tmp_path, capsys):
    assert run_tables(SimpleNamespace(inputs=[str(batch_out)], out=str(tmp_path / "t"))) == 0
    err = capsys.readouterr().err
    assert "run_info.json" not in err and "validation_report.json" not in err
    assert "1 paper →" in err


def test_references_skip_bibrs_own_files_silently(batch_out, tmp_path, capsys):
    assert run_references(_refs_args([batch_out], tmp_path / "refs.bib")) == 0
    err = capsys.readouterr().err
    assert "run_info.json" not in err
    assert "from 1 paper →" in err


def test_another_non_export_is_still_reported(batch_out, tmp_path, capsys):
    (batch_out / "notes.json").write_text(json.dumps({"hello": 1}))
    assert run_tables(SimpleNamespace(inputs=[str(batch_out)], out=str(tmp_path / "t"))) == 0
    assert "skipped" in capsys.readouterr().err


def test_a_pdf_is_named_as_a_paper_and_nothing_is_written(tmp_path, capsys):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF")
    tables_out = tmp_path / "tables"
    refs_out = tmp_path / "refs.bib"

    assert run_tables(SimpleNamespace(inputs=[str(pdf)], out=str(tables_out))) == 1
    assert run_references(_refs_args([pdf], refs_out)) == 1

    err = capsys.readouterr().err
    assert "paper.pdf is a paper, not a bibr export" in err
    assert "bibr chew" in err and "-o paper.json" in err
    assert not tables_out.exists() and not refs_out.exists()


def test_json_that_holds_no_export_exits_1_and_writes_nothing(tmp_path, capsys):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"not": "an export"}))
    tables_out = tmp_path / "tables"
    refs_out = tmp_path / "refs.ris"

    assert run_tables(SimpleNamespace(inputs=[str(other)], out=str(tables_out))) == 1
    assert run_references(_refs_args([other], refs_out)) == 1
    assert run_references(_refs_args([other], "-", fmt="bibtex")) == 1
    assert not tables_out.exists() and not refs_out.exists()
    assert "No bibr exports among the inputs" in capsys.readouterr().err


def test_library_writers_raise_only_when_asked(tmp_path):
    # Default: an empty corpus still writes empty tables (library behaviour unchanged).
    assert write_tables([], tmp_path / "empty").papers == 0
    assert (tmp_path / "empty" / "paper.parquet").is_file()

    from bibr.api import ChewFailure

    failed = ChewFailure("x.pdf", "OCR failed")
    with pytest.raises(NoExportsError) as raised:
        write_tables([failed], tmp_path / "strict", require_exports=True)
    assert raised.value.skipped and isinstance(raised.value, ValueError)
    assert not (tmp_path / "strict").exists()

    with pytest.raises(NoExportsError):
        write_references([], tmp_path / "r.bib", require_exports=True)
    assert not (tmp_path / "r.bib").exists()
