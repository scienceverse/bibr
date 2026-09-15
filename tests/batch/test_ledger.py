"""Ledger append/read, resume planning, and per-line export summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bibr.batch.ledger import (
    ERROR_TEXT_LIMIT,
    INTERRUPTED,
    Ledger,
    LedgerContext,
    Outcome,
    bounded_text,
    summarize_export,
    warning_key,
)
from bibr.batch.manifest import BatchItem, sha256_file

CTX = LedgerContext(run_id="run1", executor="local", bibr_version="9.9.9", build_sha="abc123")


def _item(tmp_path: Path, name: str = "paper", body: bytes = b"%PDF-1.4\nx") -> BatchItem:
    path = tmp_path / f"{name}.pdf"
    path.write_bytes(body)
    return BatchItem(path=path, paper_id=name, stem=name)


def _export(**overrides) -> dict:
    data = {
        "info": {"title": "T"},
        "text": [{"id": 1}, {"id": 2}, {"id": 3}],
        "bib": [{"id": "b1"}, {"id": "b2"}],
        "bib_match": [{"bib_id": "b1"}],
        "llm_usage": {
            "gemini": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            "local": {"input_tokens": 10, "output_tokens": 5},
        },
        "extraction": {
            "bibr_version": "9.9.9",
            "timings": {"ocr": 10.0, "extract": 4.0, "enrich": 1.0},
            "total_seconds": 15.5,
        },
        "processing_warnings": [
            "VALIDATION:warning:REF_YEAR_MISSING: 3 refs without a year",
            "VALIDATION:warning:REF_YEAR_MISSING: again",
            "STATEMENT_LEXICAL_FALLBACK: used lexical",
        ],
        "validation": {"errors": 0, "warnings": 2, "issues": []},
    }
    data.update(overrides)
    return data


# --- append / read -------------------------------------------------------------


@pytest.mark.parametrize("schema_version", ["11.0", "11.1", "11.2"])
def test_v11_summary_preserves_timings_usage_and_warnings(schema_version):
    legacy = _export()
    current = {
        **legacy,
        "schema_version": schema_version,
        "metadata": legacy["info"],
        "extraction": {
            "timings": {
                "stages": legacy["extraction"]["timings"],
                "total_seconds": legacy["extraction"]["total_seconds"],
            },
            "usage": {
                "totals": {"input_tokens": 110, "output_tokens": 25, "total_tokens": 135},
                "breakdown": [{"input_tokens": 110, "output_tokens": 25, "total_tokens": 135}],
            },
            "warnings": legacy["processing_warnings"],
        },
    }
    for key in ("info", "llm_usage", "processing_warnings"):
        del current[key]
    assert summarize_export(current) == summarize_export(legacy)


def test_append_and_read_roundtrip_skips_malformed_lines(tmp_path):
    ledger = Ledger(tmp_path / "out" / "outcomes.jsonl")
    ledger.append({"paper_id": "a", "status": "ok"})
    ledger.append({"paper_id": "b", "status": "failed"})
    with ledger.path.open("a") as fh:
        fh.write("not json\n{}\n")
    rows = ledger.read()
    assert [r["paper_id"] for r in rows] == ["a", "b"]
    assert ledger.latest()["b"]["status"] == "failed"
    assert ledger.attempts() == {"a": 1, "b": 1}


def test_read_missing_ledger_is_empty(tmp_path):
    assert Ledger(tmp_path / "none.jsonl").read() == []


# --- resume planning -----------------------------------------------------------


def _items(tmp_path: Path, names: list[str]) -> list[BatchItem]:
    return [_item(tmp_path, n) for n in names]


def test_plan_skips_ok_and_failed_by_default_but_reruns_interrupted(tmp_path):
    ledger = Ledger(tmp_path / "outcomes.jsonl")
    ledger.append({"paper_id": "ok1", "status": "ok"})
    ledger.append({"paper_id": "bad1", "status": "failed", "error_code": "http_422"})
    ledger.append({"paper_id": "int1", "status": "failed", "error_code": INTERRUPTED})
    items = _items(tmp_path, ["ok1", "bad1", "int1", "new1"])
    plan = ledger.plan(items)
    assert [i.paper_id for i in plan.to_run] == ["int1", "new1"]
    assert [i.paper_id for i in plan.skipped_ok] == ["ok1"]
    assert [i.paper_id for i in plan.skipped_failed] == ["bad1"]


def test_plan_retry_failed_reruns_failed(tmp_path):
    ledger = Ledger(tmp_path / "outcomes.jsonl")
    ledger.append({"paper_id": "ok1", "status": "ok"})
    ledger.append({"paper_id": "bad1", "status": "failed", "error_code": "x"})
    plan = ledger.plan(_items(tmp_path, ["ok1", "bad1"]), retry_failed=True)
    assert [i.paper_id for i in plan.to_run] == ["bad1"]
    assert [i.paper_id for i in plan.skipped_ok] == ["ok1"]
    assert plan.skipped_failed == []


def test_plan_force_reruns_everything(tmp_path):
    ledger = Ledger(tmp_path / "outcomes.jsonl")
    ledger.append({"paper_id": "ok1", "status": "ok"})
    ledger.append({"paper_id": "bad1", "status": "failed"})
    plan = ledger.plan(_items(tmp_path, ["ok1", "bad1"]), force=True)
    assert [i.paper_id for i in plan.to_run] == ["ok1", "bad1"]


def test_plan_uses_the_latest_line_per_paper(tmp_path):
    ledger = Ledger(tmp_path / "outcomes.jsonl")
    ledger.append({"paper_id": "p", "status": "ok"})
    ledger.append({"paper_id": "p", "status": "failed", "error_code": "later"})
    plan = ledger.plan(_items(tmp_path, ["p"]))
    assert plan.skipped_failed and not plan.skipped_ok
    ledger.append({"paper_id": "p", "status": "ok"})
    plan = ledger.plan(_items(tmp_path, ["p"]))
    assert plan.skipped_ok and not plan.skipped_failed


# --- record ----------------------------------------------------------------------


def test_record_ok_line_carries_identity_summary_and_context(tmp_path):
    ledger = Ledger(tmp_path / "outcomes.jsonl")
    item = _item(tmp_path)
    outcome = Outcome(
        "ok",
        export=_export(),
        started_at="2026-09-02T10:00:00+00:00",
        finished_at="2026-09-02T10:00:20+00:00",
        duration_s=20.0,
    )
    entry = ledger.record(item, outcome, context=CTX)
    on_disk = json.loads(ledger.path.read_text().splitlines()[-1])
    assert on_disk == entry
    assert entry["paper_id"] == "paper"
    assert entry["stem"] == "paper"
    assert entry["path"] == str(item.path)
    assert entry["sha256"] == sha256_file(item.path)
    assert entry["bytes"] == item.path.stat().st_size
    assert entry["status"] == "ok"
    assert entry["error_code"] is None and entry["failed_stage"] is None
    assert entry["duration_s"] == 20.0
    assert entry["pipeline_seconds"] == 15.5
    assert entry["stage_times"] == {"ocr": 10.0, "extract": 4.0, "enrich": 1.0}
    assert entry["llm_tokens"] == 135  # 120 (total) + 15 (in+out fallback)
    assert entry["llm_input_tokens"] == 110
    assert entry["llm_output_tokens"] == 25
    assert entry["n_refs"] == 2
    assert entry["n_matched"] == 1
    assert entry["n_sentences"] == 3
    assert entry["warnings"]["count"] == 3
    assert entry["warnings"]["codes"] == {
        "VALIDATION:warning:REF_YEAR_MISSING": 2,
        "STATEMENT_LEXICAL_FALLBACK": 1,
    }
    assert len(entry["warnings"]["first"]) == 3
    assert entry["bibr_version"] == "9.9.9"
    assert entry["build_sha"] == "abc123"
    assert entry["executor"] == "local"
    assert entry["run_id"] == "run1"
    assert entry["attempt"] == 1


def test_record_failed_line_bounds_error_text_and_increments_attempt(tmp_path):
    ledger = Ledger(tmp_path / "outcomes.jsonl")
    item = _item(tmp_path)
    ledger.record(item, Outcome("ok", export=_export()), context=CTX)
    long_error = "boom " * 1000
    entry = ledger.record(
        item,
        Outcome(
            "failed",
            error_code="REF_PARSE",
            failed_stage="extract",
            error=long_error,
            extra={"job_id": "j1", "retries": 2},
        ),
        context=CTX,
    )
    assert entry["attempt"] == 2
    assert entry["status"] == "failed"
    assert entry["error_code"] == "REF_PARSE"
    assert entry["failed_stage"] == "extract"
    assert len(entry["error"]) == ERROR_TEXT_LIMIT
    assert entry["error"].endswith("…")
    assert entry["stage_times"] is None and entry["n_refs"] is None
    assert entry["job_id"] == "j1" and entry["retries"] == 2


def test_record_uses_outcome_hash_when_provided(tmp_path):
    ledger = Ledger(tmp_path / "outcomes.jsonl")
    item = _item(tmp_path)
    entry = ledger.record(
        item, Outcome("failed", error_code="x", sha256="deadbeef", size=7), context=CTX
    )
    assert entry["sha256"] == "deadbeef" and entry["bytes"] == 7


def test_record_duration_falls_back_to_export_total_seconds(tmp_path):
    ledger = Ledger(tmp_path / "outcomes.jsonl")
    entry = ledger.record(_item(tmp_path), Outcome("ok", export=_export()), context=CTX)
    assert entry["duration_s"] == 15.5


# --- summaries -----------------------------------------------------------------


def test_summarize_export_enrichment_fallback_and_missing_blocks():
    summary = summarize_export(
        {"bib": [{}, {}, {}], "enrichment": {"refs_enriched": 2, "refs_total": 3}}
    )
    assert summary["n_refs"] == 3
    assert summary["n_matched"] == 2
    assert summary["stage_times"] is None
    assert summary["llm_tokens"] == 0
    assert summary["warnings"] == {"count": 0, "first": [], "codes": {}}
    assert summarize_export(None) == {}
    assert summarize_export("nope") == {}


def test_warning_key_shapes():
    assert warning_key("VALIDATION:warning:REF_YEAR: 3 refs") == "VALIDATION:warning:REF_YEAR"
    assert warning_key("STATEMENT_LEXICAL_FALLBACK: details") == "STATEMENT_LEXICAL_FALLBACK"
    assert warning_key("x" * 100) == "x" * 60


def test_bounded_text():
    assert bounded_text(None) is None
    assert bounded_text("short") == "short"
    assert len(bounded_text("y" * 2000, 10)) == 10
