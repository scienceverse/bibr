"""Section-text recall metrics (Phase 2 of the section-text benchmark).

Pure functions — no file I/O. Scores prediction vs reference section text:
token recall (the layout-drop alarm), 5-gram shingle recall, precision,
presence, and ROUGE-L/NED; plus per-type aggregation and a drop report.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

from evaluation.validation_metrics import (
    _normalize_unicode,
    abstract_ned,
    abstract_rouge_l,
)

# ROUGE-L (pure-Python O(m*n) LCS) and NED (Levenshtein, O(m*n) even via
# rapidfuzz's C core) blow up on large sections — a 2.5k-token section is already
# seconds of pure-Python LCS, so a corpus run takes minutes. Cap both. Beyond the
# cost, both are order-sensitive: on multi-column papers the text-layer reading
# order is scrambled, so on large body sections they read low from reordering, not
# content loss — misleading. So restrict them to short, contiguous sections
# (abstracts, brief intros) where order is stable; unigram and shingle recall
# (O(n), order-invariant) are the primary signals at every size. Threshold is in
# tokens (NED runs on chars, but chars >= tokens so a token cap also bounds chars).
_EDIT_DISTANCE_MAX_TOKENS = 1000

# Body-typed sections concatenated for the type-agnostic total-body metric.
BODY_SECTION_TYPES: frozenset[str] = frozenset(
    {
        "abstract",
        "intro",
        "method",
        "results",
        "discussion",
        "acknowledgment",
        "funding",
        "ethics",
        "author_contributions",
        "coi",
        # Exports since 12.0 say data_availability; older gold and predictions
        # say open_data.
        "open_data",
        "data_availability",
        "endnote",
    }
)

# Per-type metrics reported for these canonical IMRaD types + abstract.
SCORED_TYPES: tuple[str, ...] = ("abstract", "intro", "method", "results", "discussion")

_TOKEN_SPLIT_RE = re.compile(r"[^0-9a-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercase -> unicode-normalize -> split on non-alphanumeric -> drop empties.

    Numbers are kept (content); no stopword removal (we measure true coverage).
    Prediction and reference go through the SAME tokenizer, so any exotic
    characters dropped here are dropped on both sides (recall stays consistent).
    """
    if not text:
        return []
    norm = _normalize_unicode(text.lower())
    return [t for t in _TOKEN_SPLIT_RE.split(norm) if t]


def unigram_recall(pred_tokens: list[str], ref_tokens: list[str]) -> float | None:
    """Multiset (min-count) recall of reference tokens covered by prediction.

    Order-invariant — the primary alarm, robust to multi-column reading-order
    scrambling. None when the reference is empty (nothing to recall)."""
    if not ref_tokens:
        return None
    if not pred_tokens:
        return 0.0
    pred_c = Counter(pred_tokens)
    ref_c = Counter(ref_tokens)
    covered = sum(min(count, pred_c.get(tok, 0)) for tok, count in ref_c.items())
    return covered / len(ref_tokens)


def unigram_precision(pred_tokens: list[str], ref_tokens: list[str]) -> float | None:
    """Multiset precision: fraction of predicted tokens present in the reference.

    Guards against padding / misfiling. None when the prediction is empty."""
    if not pred_tokens:
        return None
    pred_c = Counter(pred_tokens)
    ref_c = Counter(ref_tokens)
    covered = sum(min(count, ref_c.get(tok, 0)) for tok, count in pred_c.items())
    return covered / len(pred_tokens)


def shingle_recall(pred_tokens: list[str], ref_tokens: list[str], n: int = 5) -> float | None:
    """Set recall of reference n-gram shingles present in the prediction.

    Detects contiguous drops: a dropped paragraph removes its 5-grams. None
    (caller falls back to unigram) when the reference has fewer than n tokens."""
    if len(ref_tokens) < n:
        return None
    ref_sh = {tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)}
    if len(pred_tokens) < n:
        return 0.0
    pred_sh = {tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)}
    return len(ref_sh & pred_sh) / len(ref_sh)


def section_present(
    unigram_recall_value: float | None,
    ref_n_tokens: int,
    tau: float = 0.5,
    min_tokens: int = 20,
) -> bool | None:
    """Whether a section counts as 'present' in the prediction.

    None for trivially short references (< min_tokens) — excluded from the
    presence rate. Otherwise True iff unigram recall >= tau."""
    if ref_n_tokens < min_tokens:
        return None
    if unigram_recall_value is None:
        return False
    return unigram_recall_value >= tau


def score_section(pred_text: str, ref_text: str) -> dict:
    """Bundle all section metrics for one (prediction, reference) text pair.

    A metric is None when undefined for the inputs (empty side / ref too short
    for shingles) — callers must exclude None from aggregates, never average
    it as 0.

    rouge_l and ned are both None when the larger side exceeds
    _EDIT_DISTANCE_MAX_TOKENS tokens: their O(m*n) LCS / edit distance become
    unacceptably slow/OOM on _total_body sections. Unigram recall and shingle
    recall (both O(n)) are the primary signals for large sections."""
    pred_tokens = tokenize(pred_text)
    ref_tokens = tokenize(ref_text)
    u_rec = unigram_recall(pred_tokens, ref_tokens)
    edit_distance_ok = max(len(pred_tokens), len(ref_tokens)) <= _EDIT_DISTANCE_MAX_TOKENS
    rouge_l = abstract_rouge_l(pred_text, ref_text) if edit_distance_ok else None
    ned = abstract_ned(pred_text, ref_text) if edit_distance_ok else None
    return {
        "ref_tokens": len(ref_tokens),
        "pred_tokens": len(pred_tokens),
        "unigram_recall": u_rec,
        "shingle_recall": shingle_recall(pred_tokens, ref_tokens),
        "unigram_precision": unigram_precision(pred_tokens, ref_tokens),
        "rouge_l": rouge_l,
        "ned": ned,
        "present": section_present(u_rec, len(ref_tokens)),
    }


# layout_recall is present only on the _total_body row (recall of the reference
# body vs ALL of bibr's text, ignoring section labels): the classifier-independent
# layout-drop measure. aggregate_by_type's .get() yields None for per-type rows.
_AGG_METRICS = (
    "unigram_recall",
    "shingle_recall",
    "unigram_precision",
    "rouge_l",
    "ned",
    "layout_recall",
)


def aggregate_by_type(per_paper: list[dict]) -> dict[str, dict]:
    """Aggregate per-(paper, type) score rows into per-type summaries.

    Returns {section_type: {metric: {mean, median, n}, presence_rate,
    n_present_eval, n_papers}}. None values are excluded per metric (never
    averaged as 0)."""
    by_type: dict[str, list[dict]] = {}
    for row in per_paper:
        by_type.setdefault(row["section_type"], []).append(row)

    out: dict[str, dict] = {}
    for stype, rows in by_type.items():
        summary: dict = {}
        for metric in _AGG_METRICS:
            vals = [r[metric] for r in rows if r.get(metric) is not None]
            summary[metric] = {
                "mean": round(statistics.fmean(vals), 4) if vals else None,
                "median": round(statistics.median(vals), 4) if vals else None,
                "n": len(vals),
            }
        present_vals = [r["present"] for r in rows if r.get("present") is not None]
        summary["presence_rate"] = (
            round(sum(1 for p in present_vals if p) / len(present_vals), 4)
            if present_vals
            else None
        )
        summary["n_present_eval"] = len(present_vals)
        summary["n_papers"] = len(rows)
        out[stype] = summary
    return out


def build_drop_report_rows(per_paper: list[dict]) -> list[dict]:
    """Severity-sorted drop list (worst recall first) — the actionable
    'where we lose text' view. Includes only rows with a computable
    unigram_recall over a non-trivial reference (ref_tokens >= 20)."""
    rows = [
        {
            "paper_id": r["paper_id"],
            "section_type": r["section_type"],
            "ref_tokens": r["ref_tokens"],
            "unigram_recall": r["unigram_recall"],
            "shingle_recall": r.get("shingle_recall"),
            "layout_recall": r.get("layout_recall"),
            "pages": r.get("pages", []),
        }
        for r in per_paper
        if r.get("unigram_recall") is not None and r.get("ref_tokens", 0) >= 20
    ]
    rows.sort(key=lambda r: r["unigram_recall"])
    return rows
