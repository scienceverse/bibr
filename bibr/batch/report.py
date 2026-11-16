"""Ledger statistics for ``bibr batch report`` and the end-of-run summary.

Everything here is computed from ``outcomes.jsonl`` lines alone — no export
is re-read. Paper-level counts use the *latest* attempt per ``paper_id``
(a resumed run's earlier failures are history, not state); latency,
stage shares, tokens and reference counts use successful attempts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from bibr.batch.ledger import (
    INTERRUPTED,
    MAX_UNSETTLED_FAILURES,
    MODEL_OVERRIDE_FAILED,
    fell_back_on_override,
)
from bibr.pipeline.stages.export import _OVERLAPPED_TIMINGS

TOP_WARNINGS = 10

# Warning codes that describe the run, not a problem with the paper: a fast run
# records LLM_SKIPPED on every paper (what an LLM would have added). They stay in
# the report JSON under warnings.expected and out of the warnings row and count.
EXPECTED_WARNINGS = frozenset({"LLM_SKIPPED"})

# What the warnings row says for the codes a batch meets most; any other code is
# shown as it is. ``bibr batch --json`` and ``bibr batch report --json`` keep the codes.
WARNING_PHRASES = {
    "OCR_REGION_FAILED": "OCR region failed",
    "OCR_PAGE_FAILED": "OCR page failed",
    "OCR_OUTPUT_TRUNCATED": "OCR output cut off",
    "OCR_TABLE_DROPPED": "table dropped",
    "OCR_TEXT_LAYER_FALLBACK": "OCR fell back to the text layer",
    "OCR_BACKEND_UNAVAILABLE": "no OCR backend, text layer only",
    "TEXT_LAYER_PAGE_UNUSABLE": "page without a usable text layer",
    "TEXT_LAYER_REGION_UNREAD": "regions not in the text layer",
    "TEXT_LAYER_TABLE_REBUILT": "table rebuilt from the text layer",
    "TEXT_LAYER_FORMULA_TEXT": "formula read as plain text",
    "TEXT_LAYER_GLYPHS_UNMAPPED": "unmapped glyphs",
    "TEXT_LAYER_PRIVATE_USE_DROPPED": "private-use characters dropped",
    "LOW_TEXT_QUALITY": "low text quality",
    "SECTION_CLASSIFIER_FAILED": "section classifier failed",
    "PAPER_CLASSIFICATION_FAILED": "paper classification failed",
    "AUTHORS_EMPTY": "no authors",
    "FRONT_TAGGER_UNAVAILABLE": "front matter not read",
    "FRONT_TAGGER_FLAGGED": "front matter flagged",
    "FRONT_MATTER_NOT_READ": "front matter not read (no layout)",
    "SELF_CITATION_FILLED": "title/authors from the paper's own citation",
    "REF_SECTION_NOT_FOUND": "no reference section",
    "REF_SECTION_INFERRED": "reference section inferred",
    "REF_FOOTNOTE_CITATIONS": "references read from footnotes",
    "REF_SEG_GEOM_CASCADE": "reference split: geometry declined",
    "REF_SEG_REGION_CASCADE": "reference split: regions declined",
    "REF_SEG_REGION_RECOVERY": "reference split: recovered from regions",
    "REF_SEG_CRF_FALLBACK": "reference split without the LLM",
    "REF_SEG_MARKER_SPLIT_RECOVERY": "reference split: on printed markers",
    "REF_SEG_FAILED": "no references split",
    "REF_SEG_NOT_SPLIT": "reference list not split",
    "REF_SEG_MERGE_SPLIT": "merged references split",
    "REF_PARSE_TRUNCATED": "long reference parsed in part",
    "REF_PARSE_LOST": "references lost",
    "REF_UNDER_EXTRACTION_SUSPECTED": "fewer references than cited",
    "CROSSREF_ENRICHMENT_TIMEOUT": "Crossref timed out",
    "CROSSREF_ENRICHMENT_FAILED": "Crossref failed",
    "ROR_MATCHING_TIMEOUT": "ROR timed out",
    "ROR_MATCHING_FAILED": "ROR failed",
    "MODEL_OVERRIDE_INFERENCE_FAILED": "overridden model failed",
}


def _percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def compute_report(
    entries: Iterable[Mapping[str, Any]],
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate ledger *entries* (optionally only those from *run_id*)."""
    entries = list(entries)
    rows = [dict(e) for e in entries if run_id is None or e.get("run_id") == run_id]
    # Fallbacks in a row per paper over the whole ledger: a resume runs a paper
    # whose export fell back again until it has fallen back the cap's times.
    in_a_row: dict[str, int] = {}
    for entry in entries:
        if entry.get("status") == "ok":
            pid = str(entry.get("paper_id"))
            in_a_row[pid] = in_a_row.get(pid, 0) + 1 if fell_back_on_override(entry) else 0

    latest: dict[str, dict] = {}
    for row in rows:
        latest[str(row.get("paper_id"))] = row
    ok_latest = [r for r in latest.values() if r.get("status") == "ok"]
    # A paper the run stopped on a model override before it finished has no
    # verdict of its own (bibr.utils.model_overrides): not a failed paper.
    not_finished = [
        r
        for r in latest.values()
        if r.get("status") != "ok" and r.get("error_code") == MODEL_OVERRIDE_FAILED
    ]
    # Interrupted (Ctrl-C, SIGTERM) before it finished: no verdict either, and a
    # resumed run processes it again.
    interrupted = [
        r for r in latest.values() if r.get("status") != "ok" and r.get("error_code") == INTERRUPTED
    ]
    failed_latest = [
        r
        for r in latest.values()
        if r.get("status") != "ok"
        and r.get("error_code") not in (MODEL_OVERRIDE_FAILED, INTERRUPTED)
    ]
    ok_attempts = [r for r in rows if r.get("status") == "ok"]

    # Run window: earliest start to latest finish across every attempt.
    starts = [d for d in (_parse_iso(r.get("started_at")) for r in rows) if d is not None]
    ends = [d for d in (_parse_iso(r.get("finished_at")) for r in rows) if d is not None]
    window_start = min(starts) if starts else None
    window_end = max(ends) if ends else None
    hours = (
        max((window_end - window_start).total_seconds(), 0.0) / 3600.0
        if window_start and window_end
        else 0.0
    )
    throughput = (len(ok_latest) / hours) if hours > 0 else None

    durations = [d for d in (_number(r.get("duration_s")) for r in ok_attempts) if d is not None]

    stage_totals: dict[str, float] = {}
    for row in ok_attempts:
        times = row.get("stage_times")
        if not isinstance(times, Mapping):
            continue
        for stage, seconds in times.items():
            if str(stage) in _OVERLAPPED_TIMINGS:
                # Timers that overlap another stage's wall clock (e.g. the
                # prefetch runs under extract) are already inside that stage's
                # share; counting them again deflates the real stages.
                continue
            value = _number(seconds)
            if value is None:
                continue
            stage_totals[str(stage)] = stage_totals.get(str(stage), 0.0) + value
    stage_sum = sum(stage_totals.values())
    stage_shares = (
        {
            stage: round(seconds / stage_sum * 100.0, 1)
            for stage, seconds in sorted(stage_totals.items(), key=lambda kv: -kv[1])
        }
        if stage_sum > 0
        else {}
    )

    token_rows = [t for t in (_number(r.get("llm_tokens")) for r in ok_attempts) if t is not None]
    total_tokens = int(sum(token_rows))

    refs = [n for n in (_number(r.get("n_refs")) for r in ok_latest) if n is not None]
    matched = [n for n in (_number(r.get("n_matched")) for r in ok_latest) if n is not None]
    total_refs = int(sum(refs))
    total_matched = int(sum(matched))

    by_code: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    for row in failed_latest:
        code = str(row.get("error_code") or "unknown")
        by_code[code] = by_code.get(code, 0) + 1
        stage = row.get("failed_stage")
        if stage:
            by_stage[str(stage)] = by_stage.get(str(stage), 0) + 1

    warning_counts: dict[str, int] = {}
    expected_counts: dict[str, int] = {}
    papers_with_warnings = 0
    for row in ok_latest:
        warnings = row.get("warnings")
        if not isinstance(warnings, Mapping):
            continue
        count = _number(warnings.get("count")) or 0
        codes = warnings.get("codes")
        expected = 0
        if isinstance(codes, Mapping):
            for key, value in codes.items():
                n = _number(value)
                if n is None:
                    continue
                target = expected_counts if str(key) in EXPECTED_WARNINGS else warning_counts
                target[str(key)] = target.get(str(key), 0) + int(n)
                if str(key) in EXPECTED_WARNINGS:
                    expected += int(n)
        if count - expected > 0:
            papers_with_warnings += 1
    top_warnings = sorted(warning_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_WARNINGS]

    return {
        "run_id": run_id,
        "papers": {
            "total": len(latest),
            "ok": len(ok_latest),
            "failed": len(failed_latest),
            # Interrupted before it finished; a resume runs them.
            "interrupted": len(interrupted),
            # Not finished because the run stopped on a model override; a resume runs them.
            "stopped_on_model_override": len(not_finished),
            # ok, but an overridden model raised on them and they fell back; a resume runs
            # them, until they have fallen back MAX_UNSETTLED_FAILURES times in a row.
            "fell_back_on_model_override": sum(1 for r in ok_latest if fell_back_on_override(r)),
            # Of those, the papers that have reached that cap: they wait for --retry-failed.
            "fell_back_capped": sum(
                1
                for r in ok_latest
                if fell_back_on_override(r)
                and in_a_row.get(str(r.get("paper_id")), 0) >= MAX_UNSETTLED_FAILURES
            ),
            "attempts": len(rows),
        },
        "window": {
            "started_at": window_start.isoformat(timespec="seconds") if window_start else None,
            "finished_at": window_end.isoformat(timespec="seconds") if window_end else None,
            "hours": round(hours, 3),
        },
        "throughput_per_hour": round(throughput, 2) if throughput is not None else None,
        "latency_s": {
            "n": len(durations),
            "p50": _round(_percentile(durations, 0.5)),
            "p90": _round(_percentile(durations, 0.9)),
            "max": _round(max(durations)) if durations else None,
            "mean": _round(sum(durations) / len(durations)) if durations else None,
        },
        "stage_shares_pct": stage_shares,
        "llm_tokens": {
            "total": total_tokens,
            "papers": len(token_rows),
            "mean_per_paper": _round(total_tokens / len(token_rows)) if token_rows else None,
        },
        "references": {
            "total": total_refs,
            "matched": total_matched,
            "match_rate": _round(total_matched / total_refs) if total_refs else None,
            "mean_per_paper": _round(total_refs / len(refs)) if refs else None,
        },
        "failures": {
            "by_error_code": dict(sorted(by_code.items(), key=lambda kv: (-kv[1], kv[0]))),
            "by_failed_stage": dict(sorted(by_stage.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "warnings": {
            # Papers with a warning other than an EXPECTED_WARNINGS one.
            "papers_with_warnings": papers_with_warnings,
            "top": [{"warning": key, "count": count} for key, count in top_warnings],
            # EXPECTED_WARNINGS codes (a fast run's LLM_SKIPPED): not counted above.
            "expected": [
                {"warning": key, "count": count}
                for key, count in sorted(expected_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
        },
    }


def _papers(n: int) -> str:
    return "paper" if n == 1 else "papers"


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def _fmt_seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}s"


def _fmt_counts(counts: Mapping[str, int], *, limit: int | None = None) -> str:
    items = list(counts.items())
    if limit is not None:
        items = items[:limit]
    return " · ".join(f"{key} ×{count}" for key, count in items) if items else "—"


def render_report(report: Mapping[str, Any], *, title: str = "bibr batch report") -> str:
    """Compact, plain-text table of :func:`compute_report`'s output."""
    papers = report["papers"]
    window = report["window"]
    latency = report["latency_s"]
    tokens = report["llm_tokens"]
    refs = report["references"]
    failures = report["failures"]
    warnings = report["warnings"]

    rows: list[tuple[str, str]] = []
    stopped = papers.get("stopped_on_model_override") or 0
    capped = papers.get("fell_back_capped") or 0
    fell_back = (papers.get("fell_back_on_model_override") or 0) - capped
    interrupted = papers.get("interrupted") or 0
    attempts = papers["attempts"]
    rows.append(
        (
            "papers",
            f"{papers['ok']} ok · {papers['failed']} failed · "
            + (f"{interrupted} interrupted · " if interrupted else "")
            + (f"{stopped} not finished · " if stopped else "")
            + f"{papers['total']} total ({attempts} {'attempt' if attempts == 1 else 'attempts'})",
        )
    )
    if stopped or fell_back:
        parts = []
        if stopped:
            parts.append(
                f"the run stopped on a model override: {stopped} "
                f"{_papers(stopped)} not finished, through no fault of the PDF"
            )
        if fell_back:
            parts.append(
                f"{fell_back} {_papers(fell_back)} fell back from an overridden model "
                "(MODEL_OVERRIDE_INFERENCE_FAILED)"
            )
        tail = "a resumed run processes them again"
        if fell_back:
            tail += f", until each has fallen back {MAX_UNSETTLED_FAILURES} times in a row"
        rows.append(("override", "; ".join(parts) + "; " + tail))
    if capped:
        rows.append(
            (
                "override",
                f"{capped} {_papers(capped)} fell back on a model override "
                f"{MAX_UNSETTLED_FAILURES} times in a row; a resumed run skips them: rerun with "
                "--retry-failed after fixing the model",
            )
        )
    if window["started_at"]:
        rows.append(
            (
                "window",
                f"{window['started_at']} → {window['finished_at']} ({window['hours']:.2f} h)",
            )
        )
    tp = report["throughput_per_hour"]
    rows.append(("throughput", "—" if tp is None else f"{tp:.1f} papers/h"))
    rows.append(
        (
            "latency",
            f"p50 {_fmt_seconds(latency['p50'])} · p90 {_fmt_seconds(latency['p90'])} · "
            f"max {_fmt_seconds(latency['max'])} (n={latency['n']})",
        )
    )
    # Stages under half a percent (a text-layer run's ocr) would print as 0%.
    shares = {stage: pct for stage, pct in report["stage_shares_pct"].items() if pct >= 0.5}
    rows.append(
        (
            "stage share",
            " · ".join(f"{stage} {pct:.0f}%" for stage, pct in shares.items()) if shares else "—",
        )
    )
    mean_tokens = tokens["mean_per_paper"]
    rows.append(
        (
            "llm tokens",
            f"{tokens['total']:,} total"
            + (f" · {mean_tokens:,.0f} / paper" if mean_tokens is not None else ""),
        )
    )
    rate = refs["match_rate"]
    mean_refs = refs["mean_per_paper"]
    rows.append(
        (
            "references",
            f"{refs['total']:,} refs · {refs['matched']:,} matched"
            + (f" ({rate * 100:.0f}%)" if rate is not None else "")
            + (f" · {mean_refs:.1f} / paper" if mean_refs is not None else ""),
        )
    )
    failure_text = _fmt_counts(failures["by_error_code"])
    if failures["by_failed_stage"]:
        failure_text += f" | stage: {_fmt_counts(failures['by_failed_stage'])}"
    rows.append(("failures", failure_text))
    n_warned = warnings["papers_with_warnings"]
    rows.append(
        (
            "warnings",
            f"{n_warned} {_papers(n_warned)}"
            + (
                " · "
                + _fmt_counts(
                    {WARNING_PHRASES.get(w["warning"], w["warning"]): w["count"] for w in top}
                )
                if (top := warnings["top"])
                else ""
            ),
        )
    )

    width = max(len(label) for label, _ in rows)
    lines = [title]
    lines.extend(f"  {label:<{width}}  {value}" for label, value in rows)
    return "\n".join(lines)
