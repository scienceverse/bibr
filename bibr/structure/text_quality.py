"""Parse-quality ("garbage text") scoring — report-only.

Ported from Docling's ``rate_text_quality`` (``docling/models/stages/
page_preprocessing/page_preprocessing_model.py``) and adapted to bibr's
failure modes. bibr text comes from two sources with different garbage
signatures:

* a native PDF text-layer pass (``OcrRegionResult.native_text_used``) — where
  classic mojibake / broken-ToUnicode output (U+FFFD replacement chars) shows
  up, and
* VLM OCR — where the failure modes are wide letter spacing ("F á b o s"),
  repeated-token loops, and empty content.

Docling's ``GLYPH<hex>`` / ``/G\\d+`` patterns are docling-parse artifacts and
are deliberately NOT ported. Each text unit is scored in ``[0, 1]``: a hard
fail returns ``0.0``; otherwise the score is ``1.0`` minus a per-run
fragmentation penalty (mirroring Docling). Page and paper scores take the 10th
percentile of the per-unit scores to emphasise problems, exactly as the donor
does with ``np.nanquantile(scores, q=0.10)``. At the paper level an EMPTY
scoreable region (layout found text, OCR produced none) is a coverage failure
scored ``0.0`` by :func:`paper_text_quality`; the pure-string
:func:`rate_text_quality` still returns ``1.0`` for empty input, so that
coverage semantics stays out of the per-string contract.

This module never rewrites text — it only rates it. The spaced-letter detector
mirrors (does not reuse) the collapse heuristic in
``bibr.utils.text.normalize_text``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

# --- Hard-fail thresholds -------------------------------------------------

# Unicode replacement char — the canonical broken-decode marker (native-text
# mojibake). Any occurrence hard-fails the unit, matching Docling's blacklist.
_REPLACEMENT_CHAR = "�"

# VLM decode loop: the same whitespace-delimited token repeated this many times
# in a row is degenerate output, never natural prose.
_MAX_TOKEN_RUN = 5

# Symbol-soup guard: a unit at least this long whose characters are at least
# this fraction non-alphanumeric and non-space is a broken glyph decode rather
# than prose. The length floor keeps punctuation-only fragments ("(a=b)") safe.
_SYMBOL_SOUP_MIN_LEN = 20
_SYMBOL_SOUP_RATIO = 0.5

# --- Fragmentation penalty ------------------------------------------------

# Spaced-letter guard (the "F á b o s" case): a run of this many consecutive
# single-alphanumeric-char tokens is a tracked/kerned word the OCR split into
# letters. A penalty applies only when at least ``_MIN_SPACED_RUNS_FOR_PENALTY``
# such runs occur in the unit — matching Docling, which penalises fragmentation
# only when it is pervasive (>= 3 matches). Author initials ("J. R. R.") carry
# a trailing period, so their tokens are length 2 and never count.
_MIN_SPACED_RUN = 4
_MIN_SPACED_RUNS_FOR_PENALTY = 3
_SPACED_RUN_PENALTY = 0.1

# Region treatments (per ``PDFParser.LABEL_TREATMENT``) whose text is scoreable
# prose. Formulas / images / tables / structural / abandoned regions carry no
# prose to rate. ``section_hint`` (abstract/reference regions) IS scored:
# reference text is where the spaced-letter OCR failure has actually been
# observed in production.
SCOREABLE_TREATMENTS = frozenset({"content", "heading", "footnote", "section_hint"})


@dataclass
class TextQualityReport:
    """Aggregated parse-quality for a paper.

    ``score`` is the paper-level scalar (10th percentile across every rated
    region). ``page_scores`` maps source page -> its own 10th-percentile score,
    kept for warning messages / diagnostics.
    """

    score: float
    n_regions: int
    page_scores: dict[int, float] = field(default_factory=dict)


def _max_consecutive_token_run(tokens: list[str]) -> int:
    """Length of the longest run of identical consecutive tokens."""
    best = 0
    run = 0
    prev: str | None = None
    for tok in tokens:
        if tok == prev:
            run += 1
        else:
            run = 1
            prev = tok
        if run > best:
            best = run
    return best


def _count_spaced_letter_runs(tokens: list[str]) -> int:
    """Count maximal runs of >= ``_MIN_SPACED_RUN`` single-alphanumeric tokens."""
    runs = 0
    run_len = 0
    for tok in tokens:
        if len(tok) == 1 and tok.isalnum():
            run_len += 1
        else:
            if run_len >= _MIN_SPACED_RUN:
                runs += 1
            run_len = 0
    if run_len >= _MIN_SPACED_RUN:
        runs += 1
    return runs


def rate_text_quality(text: str) -> float:
    """Rate one text unit in ``[0, 1]``: ``0.0`` on garbage, else ``1.0`` minus
    a spaced-letter fragmentation penalty. Empty text is neutral (``1.0``)."""
    if not text:
        return 1.0

    # Hard fails -> 0.0.
    if _REPLACEMENT_CHAR in text:
        return 0.0
    tokens = text.split()
    if _max_consecutive_token_run(tokens) >= _MAX_TOKEN_RUN:
        return 0.0
    if len(text) >= _SYMBOL_SOUP_MIN_LEN:
        noise = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if noise / len(text) >= _SYMBOL_SOUP_RATIO:
            return 0.0

    # Fragmentation penalty (only when pervasive), like Docling's frag rule.
    runs = _count_spaced_letter_runs(tokens)
    if runs >= _MIN_SPACED_RUNS_FOR_PENALTY:
        return max(1.0 - _SPACED_RUN_PENALTY * runs, 0.0)
    return 1.0


def paper_text_quality(
    regions: Iterable[tuple[int, str, str]],
    *,
    label_treatment: Mapping[str, str] | None = None,
) -> TextQualityReport | None:
    """Aggregate per-region text-quality into a paper-level report.

    ``regions`` yields ``(page, label, text)`` triples. Labels are mapped to a
    treatment via ``label_treatment`` (defaults to ``PDFParser.LABEL_TREATMENT``,
    imported lazily to keep this module free of heavy deps); only treatments in
    ``SCOREABLE_TREATMENTS`` are rated. A scoreable region whose OCR text is
    empty is a coverage failure, scored ``0.0`` (not skipped) — a page of empty
    text regions is OCR that produced nothing, not clean prose. Per-page scores
    are the 10th percentile of that page's units; the paper scalar is the 10th
    percentile across every rated unit. Returns ``None`` only when NO region is
    scoreable (e.g. DOCX-native input, which carries no OCR regions, or input
    with only non-scoreable labels) — never merely because the scoreable text
    was empty.
    """
    if label_treatment is None:  # lazy: pdf_parser pulls heavy structure deps
        from bibr.structure.pdf_parser import LABEL_TREATMENT

        label_treatment = LABEL_TREATMENT

    per_page: dict[int, list[float]] = {}
    all_scores: list[float] = []
    for page, label, text in regions:
        if label_treatment.get(label) not in SCOREABLE_TREATMENTS:
            continue
        # An EMPTY scoreable region is an OCR-coverage failure, not neutral:
        # layout detected prose here but OCR produced nothing. Score it a hard
        # 0.0 so it drags the (problem-emphasising) 10th-percentile score down.
        # The pure-string ``rate_text_quality("")`` deliberately keeps returning
        # 1.0 for its own API contract — the coverage semantics live only here.
        score = 0.0 if not text or not text.strip() else rate_text_quality(text)
        per_page.setdefault(page, []).append(score)
        all_scores.append(score)

    if not all_scores:
        return None

    page_scores = {page: float(np.nanquantile(scores, 0.10)) for page, scores in per_page.items()}
    return TextQualityReport(
        score=float(np.nanquantile(all_scores, 0.10)),
        n_regions=len(all_scores),
        page_scores=page_scores,
    )
