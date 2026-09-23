"""Parquet corpus tables: one file per table, rows keyed by paper_id."""

from __future__ import annotations

import copy
import json

import pyarrow.parquet as pq
import pytest

from bibr.export.json_export import _export_paper_payload
from bibr.export.tables import PAPER_TABLE, TABLES, write_tables


@pytest.fixture
def payload(demo_paper) -> dict:
    """A complete export (``completed_at`` included, which the reader requires)."""
    return _export_paper_payload(demo_paper)


def _second_paper(payload: dict) -> dict:
    other = copy.deepcopy(payload)
    other["paper_id"] = "second"
    other["bib"] = []
    other["bib_match"] = []
    return other


def test_every_table_is_written_keyed_by_paper_id(payload, tmp_path):
    report = write_tables([payload, _second_paper(payload)], tmp_path)

    assert report.papers == 2
    assert set(report.files) == {PAPER_TABLE, *TABLES}
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        f"{name}.parquet" for name in report.files
    )
    papers = pq.read_table(tmp_path / "paper.parquet").to_pylist()
    assert [p["paper_id"] for p in papers] == [payload["paper_id"], "second"]
    assert papers[0]["title"] == payload["metadata"]["title"]
    assert papers[0]["sha256"] == payload["source"]["sha256"]

    bib = pq.read_table(tmp_path / "bib.parquet")
    assert bib.column_names[:2] == ["paper_id", "bib_id"]
    assert bib.num_rows == len(payload["bib"])  # the second paper has none
    text = pq.read_table(tmp_path / "text.parquet")
    assert text.num_rows == 2 * len(payload["text"])


def test_column_types_come_from_the_models_not_the_data(payload, tmp_path):
    """An empty table still has its columns, and nested fields stay nested."""
    import pyarrow as pa

    payload = copy.deepcopy(payload)
    payload["eq"] = []
    write_tables([payload], tmp_path)

    eq = pq.read_schema(tmp_path / "eq.parquet")
    assert eq.names[:3] == ["paper_id", "eq_id", "text_id"]
    assert eq.field("start").type == pa.int64()
    author = pq.read_schema(tmp_path / "author.parquet")
    assert author.field("credit_roles").type == pa.list_(pa.string())
    table = pq.read_schema(tmp_path / "table.parquet")
    assert table.field("contents").type == pa.list_(pa.list_(pa.string()))
    match = pq.read_table(tmp_path / "metadata_match.parquet").to_pylist()[0]
    assert match["author"][0]["orcid"] == "https://orcid.org/0000-0002-1825-0097"
    assert match["funder"][0]["funder_doi"] == "10.13039/100000001"


def test_constrained_columns_keep_their_type(payload, tmp_path):
    """A 1-based id or a 4-number box is still an integer or a float list, not
    the JSON-string fallback of a type the writer does not recognize."""
    import pyarrow as pa

    write_tables([payload], tmp_path)

    xref = pq.read_schema(tmp_path / "xref.parquet")
    assert xref.field("xref_id").type == pa.int64()
    assert xref.field("target_id").type == pa.int64()
    affiliation = pq.read_schema(tmp_path / "affiliation.parquet")
    assert affiliation.field("author_ids").type == pa.list_(pa.int64())
    parts = pq.read_schema(tmp_path / "extraction_float_parts.parquet")
    assert parts.field("bbox").type == pa.list_(pa.float64())
    for name, (_, model) in TABLES.items():
        schema = pq.read_schema(tmp_path / f"{name}.parquet")
        for field in model.model_fields:
            if field.endswith("_id") and field != "service_id":
                assert schema.field(field).type == pa.int64(), (name, field)


def test_same_schema_whatever_the_papers(payload, tmp_path):
    empty = copy.deepcopy(payload)
    for key in TABLES:
        if not key.startswith("extraction_"):
            empty[key] = []
    write_tables([payload], tmp_path / "a")
    write_tables([empty], tmp_path / "b")
    for name in (PAPER_TABLE, *TABLES):
        assert pq.read_schema(tmp_path / "a" / f"{name}.parquet") == pq.read_schema(
            tmp_path / "b" / f"{name}.parquet"
        ), name


def test_processing_lists_get_their_own_tables(payload, tmp_path):
    payload = copy.deepcopy(payload)
    warning = {"code": "OCR_PAGE_FAILED", "message": "OCR failed for a page (page 2)"}
    payload["extraction"]["warnings"] = [warning]
    write_tables([payload], tmp_path)
    issues = pq.read_table(tmp_path / "extraction_validation_issues.parquet").to_pylist()
    expected = payload["extraction"]["validation"]["issues"]
    assert [i["code"] for i in issues] == [i["code"] for i in expected]
    sections = pq.read_table(tmp_path / "extraction_section_classification.parquet")
    assert sections.num_rows == len(payload["extraction"]["diagnostics"]["section_classification"])
    warnings = pq.read_table(tmp_path / "extraction_warnings.parquet").to_pylist()
    assert warnings == [{"paper_id": payload["paper_id"], **warning}]


def test_reads_json_files_and_skips_other_json(payload, tmp_path):
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "a.json").write_text(json.dumps(payload))
    (exports / "run_info.json").write_text(json.dumps({"run_id": "x"}))
    (exports / "broken.json").write_text("{not json")

    report = write_tables(sorted(exports.glob("*.json")), tmp_path / "tables")

    assert report.papers == 1
    assert {source.rsplit("/", 1)[-1] for source, _ in report.skipped} == {
        "run_info.json",
        "broken.json",
    }


def test_duplicate_paper_id_fails_and_leaves_no_partial_output(payload, tmp_path):
    out = tmp_path / "tables"
    out.mkdir()
    (out / "keep.txt").write_text("untouched")
    with pytest.raises(ValueError, match="must be unique"):
        write_tables([payload, copy.deepcopy(payload)], out)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_other_major_versions_are_refused(payload, tmp_path):
    from pydantic import ValidationError

    old = copy.deepcopy(payload)
    old["schema_version"] = "11.0"
    with pytest.raises(ValidationError):
        write_tables([old], tmp_path)


def test_results_are_accepted(payload, tmp_path):
    from bibr.api import Result

    report = write_tables([Result(payload)], tmp_path)
    assert report.papers == 1


def test_cli_writes_tables(payload, tmp_path, capsys):
    from types import SimpleNamespace

    from bibr.local.cli.tables import run_tables

    exports = tmp_path / "results"
    exports.mkdir()
    (exports / "a.json").write_text(json.dumps(payload))
    out = exports / "tables"  # inside the input directory: never read back

    assert run_tables(SimpleNamespace(inputs=[str(exports)], out=str(out))) == 0
    assert (out / "paper.parquet").is_file()
    assert run_tables(SimpleNamespace(inputs=[str(exports)], out=str(out))) == 0
    assert run_tables(SimpleNamespace(inputs=[str(tmp_path / "nope")], out=str(out))) == 2
