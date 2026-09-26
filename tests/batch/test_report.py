"""Report computations over a synthetic ledger."""

from __future__ import annotations

import json

from bibr.batch.report import compute_report, render_report


def _row(paper_id, status="ok", **fields):
    base = {
        "paper_id": paper_id,
        "status": status,
        "started_at": "2026-09-02T10:00:00+00:00",
        "finished_at": "2026-09-02T10:01:00+00:00",
        "run_id": "r1",
    }
    base.update(fields)
    return base


def _ledger():
    return [
        _row(
            "a",
            duration_s=10.0,
            stage_times={"ocr": 6.0, "extract": 3.0, "enrich": 1.0},
            llm_tokens=1000,
            n_refs=10,
            n_matched=8,
            warnings={"count": 2, "codes": {"VALIDATION:warning:X": 2}},
        ),
        _row(
            "b",
            duration_s=20.0,
            stage_times={"ocr": 12.0, "extract": 6.0, "enrich": 2.0},
            llm_tokens=3000,
            n_refs=30,
            n_matched=22,
            warnings={"count": 1, "codes": {"VALIDATION:warning:X": 1, "OTHER": 1}},
            finished_at="2026-09-02T12:00:00+00:00",
        ),
        # c failed first, then succeeded in a later attempt: counts as ok.
        _row("c", status="failed", error_code="http_503", failed_stage=None, duration_s=5.0),
        _row(
            "c",
            duration_s=30.0,
            stage_times={"ocr": 30.0},
            llm_tokens=500,
            n_refs=0,
            n_matched=0,
            warnings={"count": 0, "codes": {}},
            run_id="r2",
        ),
        # d failed and stayed failed.
        _row("d", status="failed", error_code="poll_timeout", failed_stage="ocr"),
        _row("e", status="failed", error_code="REF_PARSE", failed_stage="extract", run_id="r2"),
    ]


def test_compute_report_counts_latest_attempt_per_paper():
    report = compute_report(_ledger())
    assert report["papers"] == {"total": 5, "ok": 3, "failed": 2, "attempts": 6}


def test_compute_report_window_and_throughput():
    report = compute_report(_ledger())
    assert report["window"]["started_at"] == "2026-09-02T10:00:00+00:00"
    assert report["window"]["finished_at"] == "2026-09-02T12:00:00+00:00"
    assert report["window"]["hours"] == 2.0
    assert report["throughput_per_hour"] == 1.5  # 3 ok / 2 h


def test_compute_report_latency_percentiles():
    latency = compute_report(_ledger())["latency_s"]
    assert latency["n"] == 3
    assert latency["p50"] == 20.0
    assert latency["p90"] == 28.0
    assert latency["max"] == 30.0
    assert latency["mean"] == 20.0


def test_compute_report_stage_shares_sum_to_100_and_sort_desc():
    shares = compute_report(_ledger())["stage_shares_pct"]
    assert list(shares) == ["ocr", "extract", "enrich"]
    assert shares == {"ocr": 80.0, "extract": 15.0, "enrich": 5.0}


def test_compute_report_excludes_overlapped_prefetch_from_stage_shares():
    # enrich_prefetch runs under extract and is already inside its wall
    # clock (like extraction.timings.total_seconds excludes it): it must
    # not appear as a phantom stage or deflate the real shares.
    rows = [
        _row(
            "p",
            duration_s=42.0,
            stage_times={"ocr": 30.0, "extract": 10.0, "enrich": 2.0, "enrich_prefetch": 8.0},
        ),
    ]
    shares = compute_report(rows)["stage_shares_pct"]
    assert "enrich_prefetch" not in shares
    assert shares == {"ocr": 71.4, "extract": 23.8, "enrich": 4.8}


def test_compute_report_stage_shares_all_overlapped_is_empty():
    rows = [_row("p", duration_s=8.0, stage_times={"enrich_prefetch": 8.0})]
    assert compute_report(rows)["stage_shares_pct"] == {}


def test_compute_report_tokens_and_references():
    report = compute_report(_ledger())
    assert report["llm_tokens"] == {"total": 4500, "papers": 3, "mean_per_paper": 1500.0}
    refs = report["references"]
    assert refs["total"] == 40 and refs["matched"] == 30
    assert refs["match_rate"] == 0.75
    assert refs["mean_per_paper"] == round(40 / 3, 3)


def test_compute_report_failure_breakdown_and_warning_top():
    report = compute_report(_ledger())
    assert report["failures"]["by_error_code"] == {"REF_PARSE": 1, "poll_timeout": 1}
    assert report["failures"]["by_failed_stage"] == {"extract": 1, "ocr": 1}
    assert report["warnings"]["papers_with_warnings"] == 2
    assert report["warnings"]["top"] == [
        {"warning": "VALIDATION:warning:X", "count": 3},
        {"warning": "OTHER", "count": 1},
    ]


def test_compute_report_filters_by_run_id():
    report = compute_report(_ledger(), run_id="r2")
    assert report["run_id"] == "r2"
    assert report["papers"] == {"total": 2, "ok": 1, "failed": 1, "attempts": 2}


def test_compute_report_on_empty_ledger():
    report = compute_report([])
    assert report["papers"]["total"] == 0
    assert report["throughput_per_hour"] is None
    assert report["latency_s"]["p50"] is None
    assert report["stage_shares_pct"] == {}
    text = render_report(report)
    assert "0 ok · 0 failed" in text


def test_render_report_is_compact_and_json_roundtrips():
    report = compute_report(_ledger())
    text = render_report(report, title="bibr batch report · x")
    lines = text.splitlines()
    assert lines[0] == "bibr batch report · x"
    assert any(line.strip().startswith("papers") and "3 ok · 2 failed" in line for line in lines)
    assert any("1.5 papers/h" in line for line in lines)
    assert any("p50 20.0s · p90 28.0s · max 30.0s" in line for line in lines)
    assert any("ocr 80% · extract 15% · enrich 5%" in line for line in lines)
    assert any("4,500 total · 1,500 / paper" in line for line in lines)
    assert any("40 refs · 30 matched (75%)" in line for line in lines)
    assert any(
        "REF_PARSE ×1 · poll_timeout ×1 | stage: extract ×1 · ocr ×1" in line for line in lines
    )
    assert any("VALIDATION:warning:X ×3" in line for line in lines)
    assert json.loads(json.dumps(report)) == report
