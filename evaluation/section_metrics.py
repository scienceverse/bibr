"""Section-text recall metrics (Phase 2 of the section-text benchmark).

Pure functions — no file I/O. Scores prediction vs reference section text:
token recall (the layout-drop alarm), 5-gram shingle recall, precision,
presence, and ROUGE-L/NED; plus per-type aggregation and a drop report.
"""

from __future__ import annotations

import functools
import re
import statistics
import unicodedata
from collections import Counter

from evaluation.validation_metrics import (
    _SCRIPTIO_CONTINUA_RE,
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

# One token per character of a script written without inter-word spaces (the
# same scripts title_soft special-cases), or one per run of other word characters.
_CONTINUA_CLASS = _SCRIPTIO_CONTINUA_RE.pattern
_CONTINUA_PIECE_RE = re.compile(rf"{_CONTINUA_CLASS}|(?:(?!{_CONTINUA_CLASS}).)+")


@functools.lru_cache(maxsize=4096)
def _is_latin(char: str) -> bool:
    return unicodedata.name(char, "").startswith("LATIN ")


def _word_text(text: str) -> str:
    r"""Casefolded NFKD text with every non-word character replaced by a space.

    Word characters are those ``[\W_]`` does not match: letters and digits in
    every script. Combining marks on Latin letters (and on anything that is not
    a letter) are dropped, so "Méthode" reads "methode"; marks on letters of
    other scripts stay in their word, since ``\W`` would otherwise split a
    Devanagari word at every vowel sign. NFC then recomposes what is left.
    """
    decomposed = unicodedata.normalize("NFKD", unicodedata.normalize("NFKD", text).casefold())
    out: list[str] = []
    keep_marks = False
    for char in decomposed:
        if char < "\x80":
            keep_marks = False
            out.append(char if char.isalnum() else " ")
        elif unicodedata.category(char)[0] == "M":
            if keep_marks:
                out.append(char)
        elif char.isalnum():
            keep_marks = char.isalpha() and not _is_latin(char)
            out.append(char)
        else:
            keep_marks = False
            out.append(" ")
    return unicodedata.normalize("NFC", "".join(out))


def tokenize(text: str) -> list[str]:
    r"""Casefold -> NFKD -> split on ``[\W_]+`` -> drop empties.

    Letters survive in every script ("Straße" is one token, "strasse"), and
    diacritics on Latin letters are stripped. Scripts written without spaces
    between words (Han, kana, Thai) give one token per character, so recall on
    them measures their text rather than the digits around it. Numbers are kept
    (content); no stopword removal (we measure true coverage). Prediction and
    reference go through the SAME tokenizer.
    """
    if not text:
        return []
    tokens: list[str] = []
    for word in _word_text(text).split():
        if _SCRIPTIO_CONTINUA_RE.search(word):
            tokens.extend(_CONTINUA_PIECE_RE.findall(word))
        else:
            tokens.append(word)
    return tokens


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
