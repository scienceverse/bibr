"""
Paper Type and OECD Domain constants and validation.

Classification is handled by the LLM during metadata extraction unless the
trained multitask classifier is configured (``ML_PAPER_CLASSIFIER_MODEL_ID``),
in which case :func:`classify_paper_async` predicts OECD L1/L2 + paper_type
locally. This module retains the taxonomy constants and validation functions
used by the extractor and export layers, plus the lazy, thread-safe loader for
the trained classifier.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from bibr.config import GlobalSettings, snapshot_settings
from bibr.utils.locks import LOCAL_INFERENCE_LOCK

if TYPE_CHECKING:
    from bibr.pipeline.classifier_resources import ClassifierResources
    from bibr.structure.paper_classifier_model import PaperClassifierModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paper Type
# ---------------------------------------------------------------------------


class PaperType(StrEnum):
    EMPIRICAL = "empirical"
    REVIEW = "review"
    META_ANALYSIS = "meta-analysis"
    CASE_STUDY = "case-study"
    COMMENTARY = "commentary"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# OECD Domain Taxonomy
# ---------------------------------------------------------------------------

OECD_L1_LABELS = [
    "Natural Sciences",
    "Engineering and Technology",
    "Medical and Health Sciences",
    "Agricultural and Veterinary Sciences",
    "Social Sciences",
    "Humanities and the Arts",
]

OECD_L2_MAP: dict[str, list[str]] = {
    "Natural Sciences": [
        "Mathematics",
        "Computer and Information Sciences",
        "Physical Sciences",
        "Chemical Sciences",
        "Earth and Related Environmental Sciences",
        "Biological Sciences",
    ],
    "Engineering and Technology": [
        "Civil Engineering",
        "Electrical Engineering, Electronic Engineering, Information Engineering",
        "Mechanical Engineering",
        "Chemical Engineering",
        "Materials Engineering",
        "Medical Engineering",
        "Environmental Engineering",
        "Environmental Biotechnology",
        "Industrial Biotechnology",
        "Nano-technology",
    ],
    "Medical and Health Sciences": [
        "Basic Medicine",
        "Clinical Medicine",
        "Health Sciences",
        "Medical Biotechnology",
    ],
    "Agricultural and Veterinary Sciences": [
        "Agriculture, Forestry, and Fisheries",
        "Animal and Dairy Science",
        "Veterinary Science",
        "Agricultural Biotechnology",
    ],
    "Social Sciences": [
        "Psychology and Cognitive Sciences",
        "Economics and Business",
        "Education",
        "Sociology",
        "Law",
        "Political Science",
        "Social and Economic Geography",
        "Media and Communications",
    ],
    "Humanities and the Arts": [
        "History and Archaeology",
        "Languages and Literature",
        "Philosophy, Ethics and Religion",
        "Arts (arts, history of arts, performing arts, music)",
    ],
}

# Flattened list of every L2 label, in L1 order — the candidate set for
# L1-agnostic canonicalization and the single source for OECDSubdomainLiteral.
ALL_OECD_L2_LABELS: list[str] = [label for labels in OECD_L2_MAP.values() for label in labels]

# Reverse map L2 → parent L1. Every L2 label belongs to exactly one L1
# (asserted by tests); the cross-L1 rescue uses this to flip the domain.
OECD_L2_TO_L1: dict[str, str] = {
    label: l1 for l1, labels in OECD_L2_MAP.items() for label in labels
}

# Canonical paper_type vocabulary. The last three are notices amending or
# withdrawing a previously published article.
PAPER_TYPE_LABELS: list[str] = [
    "empirical",
    "review",
    "meta-analysis",
    "case-study",
    "commentary",
    "corrigendum",
    "erratum",
    "retraction",
]

# Literal aliases consumed by bibr.schemas as guided-decoding grammars. They
# must be hand-written (typing.Literal can't unpack a runtime list) but are
# kept in lock-step with the lists above by drift-guard tests.
OECDDomainLiteral = Literal[
    "Natural Sciences",
    "Engineering and Technology",
    "Medical and Health Sciences",
    "Agricultural and Veterinary Sciences",
    "Social Sciences",
    "Humanities and the Arts",
]

OECDSubdomainLiteral = Literal[
    "Mathematics",
    "Computer and Information Sciences",
    "Physical Sciences",
    "Chemical Sciences",
    "Earth and Related Environmental Sciences",
    "Biological Sciences",
    "Civil Engineering",
    "Electrical Engineering, Electronic Engineering, Information Engineering",
    "Mechanical Engineering",
    "Chemical Engineering",
    "Materials Engineering",
    "Medical Engineering",
    "Environmental Engineering",
    "Environmental Biotechnology",
    "Industrial Biotechnology",
    "Nano-technology",
    "Basic Medicine",
    "Clinical Medicine",
    "Health Sciences",
    "Medical Biotechnology",
    "Agriculture, Forestry, and Fisheries",
    "Animal and Dairy Science",
    "Veterinary Science",
    "Agricultural Biotechnology",
    "Psychology and Cognitive Sciences",
    "Economics and Business",
    "Education",
    "Sociology",
    "Law",
    "Political Science",
    "Social and Economic Geography",
    "Media and Communications",
    "History and Archaeology",
    "Languages and Literature",
    "Philosophy, Ethics and Religion",
    "Arts (arts, history of arts, performing arts, music)",
]

PaperTypeLiteral = Literal[
    "empirical",
    "review",
    "meta-analysis",
    "case-study",
    "commentary",
    "corrigendum",
    "erratum",
    "retraction",
]


def validate_oecd_l1(raw: str | None) -> str:
    """Validate and canonicalize an LLM-returned OECD L1 label.

    Tries exact match first, then fuzzy matching via rapidfuzz.
    Returns the canonical label or "" if no match.
    """
    if not raw or not raw.strip():
        return ""

    raw_stripped = raw.strip()

    # Exact match (case-sensitive — LLM should return exact strings)
    if raw_stripped in OECD_L1_LABELS:
        return raw_stripped

    # Case-insensitive exact match
    raw_lower = raw_stripped.lower()
    for label in OECD_L1_LABELS:
        if label.lower() == raw_lower:
            return label

    # Fuzzy fallback
    from rapidfuzz import fuzz

    best_score = 0.0
    best_label = ""
    for label in OECD_L1_LABELS:
        score = fuzz.ratio(raw_stripped, label)
        if score > best_score:
            best_score = score
            best_label = label

    if best_score >= 80:
        logger.debug(
            "Fuzzy-matched OECD L1 '%s' → '%s' (score=%.1f)", raw_stripped, best_label, best_score
        )
        return best_label

    logger.warning("Could not match OECD L1 label '%s' to any known domain", raw_stripped)
    return ""


# Trailing parenthetical examples (e.g. "Arts (arts, history of arts, ...)")
# otherwise create spurious token overlap with unrelated siblings under
# fuzzy matching — "History" would tie 100 against both "History and
# Archaeology" and "Arts (..., history of arts, ...)". Dropped before
# scoring only; the untouched canonical label is still what gets returned.
_L2_PAREN_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")
# Several canonical L2 labels end in the plural "Sciences" (e.g. "Psychology
# and Cognitive Sciences"), while a near-miss LLM answer is often singular
# ("Cognitive Science"). Collapsing the plural to singular before scoring
# lets token_set_ratio treat "science"/"sciences" as the same token instead
# of penalizing the mismatch, without touching unrelated plurals.
_L2_SCIENCES_RE = re.compile(r"\bsciences\b")


def _normalize_l2_label(label: str) -> str:
    """Normalize an OECD L2 label for fuzzy-match scoring (not for display)."""
    normalized = _L2_PAREN_SUFFIX_RE.sub("", label).lower()
    return _L2_SCIENCES_RE.sub("science", normalized)


def _match_l2_label(raw: str | None, candidates: list[str]) -> str:
    """Shared scoring core: exact → case-insensitive → fuzzy over ``candidates``.

    Returns the canonical label or "". Callers supply the candidate set —
    siblings of one L1 (:func:`validate_oecd_l2`) or every L2 label
    (:func:`canonicalize_oecd_l2_any`).
    """
    if not raw or not raw.strip() or not candidates:
        return ""

    raw_stripped = raw.strip()

    # Exact match (case-sensitive — LLM should return exact strings)
    if raw_stripped in candidates:
        return raw_stripped

    # Case-insensitive exact match
    raw_lower = raw_stripped.lower()
    for label in candidates:
        if label.lower() == raw_lower:
            return label

    # Fuzzy fallback. token_set_ratio handles subset phrases (e.g. the LLM
    # returning a bare "Psychology" for canonical "Psychology and Cognitive
    # Sciences"), which plain fuzz.ratio scores far too low to catch (~47).
    from rapidfuzz import fuzz

    raw_norm = _normalize_l2_label(raw_stripped)
    best_score = 0.0
    best_label = ""
    second_score = 0.0
    for label in candidates:
        score = fuzz.token_set_ratio(raw_norm, _normalize_l2_label(label))
        if score > best_score:
            best_score, best_label, second_score = score, label, best_score
        elif score > second_score:
            second_score = score

    # Require both a minimum absolute score AND a margin over the runner-up.
    # Several sibling L2 labels share a generic head word — every
    # "Engineering and Technology" subfield contains "Engineering", and the
    # Biotechnology entries tie at 100 for a bare "Biotechnology" query —
    # so a bare score threshold alone would pick an arbitrary winner by list
    # order. Discard genuinely ambiguous near-ties instead of guessing.
    if best_score >= 85 and (best_score - second_score) >= 5:
        return best_label

    return ""


def validate_oecd_l2(l1: str, raw: str | None) -> str:
    """Validate and canonicalize an LLM-returned OECD L2 (subdomain) label.

    Tries exact match first, then case-insensitive match, then fuzzy matching
    via rapidfuzz against the candidates for ``l1`` in ``OECD_L2_MAP``.
    Returns the canonical label or "" if no match, ``l1`` is unknown/empty,
    or ``raw`` is empty.
    """
    if not raw or not raw.strip() or not l1:
        return ""

    result = _match_l2_label(raw, OECD_L2_MAP.get(l1, []))
    if result:
        logger.debug("Matched OECD L2 '%s' → '%s' (L1=%s)", raw.strip(), result, l1)
    return result


def canonicalize_oecd_l2_any(raw: str | None) -> str:
    """Canonicalize an L2 label against ALL OECD subdomains, ignoring L1.

    Same normalization, score threshold, and runner-up margin as
    :func:`validate_oecd_l2`, but scored over the union of every L1's
    candidates. Used by the schema BeforeValidator (which has no L1 context)
    and by the extractor's cross-L1 rescue. Returns the canonical label or "".
    """
    return _match_l2_label(raw, ALL_OECD_L2_LABELS)


# ---------------------------------------------------------------------------
# Trained multitask classifier — lazy, thread-safe loader + orchestration.
# Mirrors bibr.structure.section_classifier._get_trained_model(_async).
# ---------------------------------------------------------------------------

# Process-wide cache of the loaded trained classifier. Sentinel ``False`` means
# we've checked and there's no configured model id (skip the lookup again).
_paper_model_cache: PaperClassifierModel | None | bool = None
# Single-flight guard for the first load: without it, concurrent first callers
# (asyncio.gather across files → to_thread) each download and load the model.
# A threading.Lock (not asyncio.Lock) because callers reach here from worker
# threads.
_paper_model_load_lock = threading.Lock()


def _get_paper_model(settings: GlobalSettings | None = None) -> PaperClassifierModel | None:
    """Lazy-load the trained multitask paper classifier (thread-safe).

    Returns None when ``Settings.ml.paper_classifier_model_id`` is unset (the
    dark default) or the ``ml`` extra is absent. Loaded once per process.
    Blocking: downloads + loads weights on first call — async callers must use
    :func:`_get_paper_model_async` to keep the event loop free.
    """
    global _paper_model_cache
    effective = settings if settings is not None else snapshot_settings()
    cached = _paper_model_cache
    if cached is not None:
        return None if cached is False else cached  # type: ignore[return-value]
    with _paper_model_load_lock:
        if _paper_model_cache is not None:
            return None if _paper_model_cache is False else _paper_model_cache  # type: ignore[return-value]
        model_id = effective.ml.paper_classifier_model_id
        if not model_id:
            _paper_model_cache = False
            return None
        try:
            from bibr.structure.paper_classifier_model import PaperClassifierModel
        except ImportError as e:
            # Core (non-'ml') install: torch/transformers are absent. Fall back
            # to the LLM classifier rather than aborting extraction.
            logger.info("Trained paper classifier unavailable (%s); using LLM fallback", e)
            _paper_model_cache = False
            return None

        revision = effective.ml.paper_classifier_revision
        device = effective.ml.paper_classifier_device
        logger.info(
            "Loading trained paper classifier %s@%s (device=%s)", model_id, revision, device
        )
        _paper_model_cache = PaperClassifierModel.from_pretrained(
            model_id, revision=revision, device=device
        )
        return _paper_model_cache  # type: ignore[return-value]


async def _get_paper_model_async(
    settings: GlobalSettings | None = None,
) -> PaperClassifierModel | None:
    """Async wrapper for :func:`_get_paper_model` — the first load (download +
    weights) runs off-loop under the shared inference lock; cached calls only
    pay a thread hop. Lock order mirrors the section classifier (inference lock
    OUTSIDE the load lock)."""

    def _locked_load() -> PaperClassifierModel | None:
        with LOCAL_INFERENCE_LOCK:
            if settings is None:
                return _get_paper_model()
            return _get_paper_model(settings)

    return await asyncio.to_thread(_locked_load)


def _paper_prediction_tuple(prediction) -> tuple[str, float, str, float, str, float]:
    return (
        prediction.oecd_l1,
        float(prediction.oecd_l1_score),
        prediction.oecd_l2,
        float(prediction.oecd_l2_score),
        prediction.paper_type,
        float(prediction.paper_type_score),
    )


def _classify_paper_locked(model, title: str, abstract: str):
    with LOCAL_INFERENCE_LOCK:
        return model.classify_batch([(title, abstract)])


async def classify_paper_async(
    title: str,
    abstract: str,
    *,
    classifier_resources: ClassifierResources | None = None,
    settings: GlobalSettings | None = None,
) -> tuple[str, float, str, float, str, float] | None:
    """Classify a paper's OECD L1/L2 + paper_type from title + abstract.

    Returns ``(oecd_l1, l1_score, oecd_l2, l2_score, paper_type, pt_score)``
    when the trained classifier is configured, else ``None`` — signalling the
    caller to fall back to the existing LLM classification path. The forward
    pass is a synchronous PyTorch call, offloaded to a thread (under the shared
    inference lock) so the async caller stays non-blocking.
    """
    if classifier_resources is not None:
        prediction = await classifier_resources.classify_paper((title, abstract))
        if prediction is None:
            return None
        return _paper_prediction_tuple(prediction)

    model = (
        await _get_paper_model_async()
        if settings is None
        else await _get_paper_model_async(settings)
    )
    if model is None:
        return None

    predictions = await asyncio.to_thread(_classify_paper_locked, model, title, abstract)
    if not predictions:
        return None
    return _paper_prediction_tuple(predictions[0])
