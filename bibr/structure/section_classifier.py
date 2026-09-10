"""
Section Classifier for scientific paper sections.

Provides semantic classification of paper section headers into canonical types.
Uses a combination of exact/fuzzy lookup and LLM-based classification for
headers that don't match known aliases.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field, model_validator

from bibr.clients.nuextract_schema import NuExtractSchemaPolicy
from bibr.config import GlobalSettings, snapshot_settings
from bibr.exceptions import ProcessingError
from bibr.paper_contents import (
    CANONICAL_SECTION_ALIASES,
    CanonicalSection,
    is_exact_front_matter_furniture,
)

# Pure-Python bucket helper, shared with both classifier runtimes (torch-free
# module, so building the composite dedup key never forces torch).
from bibr.structure.section_classifier_common import _position_bucket
from bibr.utils.locks import LOCAL_INFERENCE_LOCK
from bibr.utils.text import normalize_text

if TYPE_CHECKING:
    from bibr.clients.llm_protocol import LlmClient
    from bibr.pipeline.classifier_resources import ClassifierResources
    from bibr.structure.section_classifier_model import SectionClassifierModel

logger = logging.getLogger(__name__)

SECTION_CLASSIFIER_DEGRADED_WARNING = "section_classifier_degraded"


def _note_section_classifier_degraded(degradation_warnings: list[str] | None) -> None:
    """Record once per paper that the configured trained tier did not answer.

    Whatever the cause — a core install without torch, weights that failed
    to load, a serve-side classifier resource in its degraded state, or an
    inference error — the LLM tier decides instead, and the export must say
    so rather than look like a healthy run.
    """
    if (
        degradation_warnings is not None
        and SECTION_CLASSIFIER_DEGRADED_WARNING not in degradation_warnings
    ):
        degradation_warnings.append(SECTION_CLASSIFIER_DEGRADED_WARNING)


def _normalize_header_echo(header: str | None) -> str:
    """Fold a header for echo comparison.

    The LLM echoes each header back; models routinely differ from the input in
    case, surrounding whitespace, or a trailing colon / numbering dot. Compare
    on that folded form so cosmetic differences don't force a real match into
    the unmatched bucket.
    """
    return re.sub(r"\s+", " ", (header or "").strip().rstrip(".:").casefold()).strip()


@dataclass
class HeaderContext:
    """Lightweight context row for the trained section-classifier template."""

    heading: str
    body: str
    relative_position: float = 0.5
    prev_heading: str = ""
    next_heading: str = ""


# Process-wide cache of the loaded trained classifier. Sentinel ``False`` means
# we've checked and there's no configured model id (skip the lookup again).
_model_cache: SectionClassifierModel | None | bool = None
# Single-flight guard for the first load: without it, concurrent first
# callers (asyncio.gather across files → to_thread) each download and load
# the model — duplicate work and 2× weights in memory. A threading.Lock
# (not asyncio.Lock) because callers reach this from worker threads.
_model_load_lock = threading.Lock()


def _get_trained_model(settings: GlobalSettings | None = None) -> SectionClassifierModel | None:
    """Lazy-load the trained MiniLM section classifier (thread-safe).

    Returns None when ``Settings.ml.section_classifier_model_id`` is unset.
    The model is loaded once per process; subsequent calls return the cache.
    Blocking: downloads + loads weights on first call — async callers must
    use :func:`_get_trained_model_async` to keep the event loop free.
    """
    global _model_cache
    effective = settings if settings is not None else snapshot_settings()
    cached = _model_cache
    if cached is not None:
        return None if cached is False else cached  # type: ignore[return-value]
    with _model_load_lock:
        if _model_cache is not None:
            return None if _model_cache is False else _model_cache  # type: ignore[return-value]
        model_id = effective.ml.section_classifier_model_id
        if not model_id:
            _model_cache = False
            return None
        from bibr.exceptions import ConfigurationError
        from bibr.structure.section_classifier_common import load_section_classifier

        revision = effective.ml.section_classifier_revision
        device = effective.ml.section_classifier_device
        logger.info(
            "Loading trained section classifier %s@%s (device=%s)", model_id, revision, device
        )
        try:
            _model_cache = load_section_classifier(
                model_id, revision=revision, device=device, settings=effective
            )
        except (ImportError, ConfigurationError) as e:
            # No usable runtime (core install without a published ONNX bundle,
            # or the torch extra missing). Fall back to the LLM classifier
            # rather than aborting the structure stage.
            logger.warning("Trained section classifier unavailable (%s); using LLM fallback", e)
            _model_cache = False
            return None
        return _model_cache  # type: ignore[return-value]


async def _get_trained_model_async(
    settings: GlobalSettings | None = None,
) -> SectionClassifierModel | None:
    """Async wrapper for :func:`_get_trained_model` — the first load
    (download + weights) runs off-loop; cached calls only pay a thread hop.

    The load takes the shared inference lock (OUTSIDE ``_model_load_lock`` —
    lock order matters): loading weights onto MPS while a sibling thread runs
    another local model's forward intermittently segfaulted the process (see
    bibr.utils.locks).
    """

    def _locked_load() -> SectionClassifierModel | None:
        with LOCAL_INFERENCE_LOCK:
            if settings is None:
                return _get_trained_model()
            return _get_trained_model(settings)

    return await asyncio.to_thread(_locked_load)


async def _classify_trained_batch(
    items: list[HeaderContext],
    *,
    classifier_resources: ClassifierResources | None = None,
    settings: GlobalSettings | None = None,
) -> list[tuple[CanonicalSection, float, bool | None]]:
    """Run the trained MiniLM classifier on document-context items.

    Returns ``(canonical, score, is_top_level)`` per item. When the type-head
    softmax probability falls below ``Settings.ml.section_classifier_min_confidence``
    the prediction collapses to ``UNKNOWN`` and ``is_top_level`` is set to
    ``None`` — failures in the tail (rare classes) typically show up as
    low-confidence spreads, and acting on them adds noise downstream.
    ``classify_batch`` is a synchronous PyTorch forward pass; we offload to a
    thread so the async caller stays non-blocking.
    """
    if not items:
        return []
    effective = settings if settings is not None else snapshot_settings()
    if classifier_resources is not None:
        preds = await classifier_resources.classify_sections(items)
        if preds is None:
            return []
    else:
        model = (
            await _get_trained_model_async()
            if settings is None
            else await _get_trained_model_async(effective)
        )
        if model is None:
            return []

        def _locked_classify():
            with LOCAL_INFERENCE_LOCK:
                return model.classify_batch(items)

        preds = await asyncio.to_thread(_locked_classify)
    threshold = float(effective.ml.section_classifier_min_confidence)
    out: list[tuple[CanonicalSection, float, bool | None]] = []
    for p in preds:
        score = float(p.score)
        if score < threshold:
            out.append((CanonicalSection.UNKNOWN, score, None))
        else:
            out.append((p.canonical_type, score, bool(p.is_top_level)))
    return out


# Valid canonical section values for LLM response validation
_VALID_SECTION_VALUES = {s.value for s in CanonicalSection}

# Single source of truth for the section types the LLM classifier may return,
# with the prompt-facing description of each. Drives both the pydantic field
# description and the prompt's "Valid section types" block so the two can't
# drift apart. Order is the order shown in the prompt.
_LLM_SECTION_TYPE_DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("title", "Paper title or running head (the heading that IS the paper's title)"),
    ("abstract", "Paper abstract or summary"),
    ("intro", "Introduction, background, related work, literature review"),
    ("method", "Methods, materials, experimental setup, procedures, participants"),
    ("results", "Results, findings, evaluation, experiments"),
    ("discussion", "Discussion, interpretation, limitations"),
    ("references", "References, bibliography"),
    (
        "acknowledgment",
        "General acknowledgments, transparency statements, action editor, "
        "corresponding-author / ORCID author-metadata blocks",
    ),
    ("funding", "Funding, financial support, grants"),
    ("keywords", "Keywords, key words"),
    (
        "open_data",
        "Data availability, code availability, materials availability, open practices, "
        "open science statements",
    ),
    ("author_contributions", "Author contributions, CRediT authorship contribution statement"),
    ("coi", "Conflict of interest, competing interests, declaration of conflicting interests"),
    ("ethics", "Ethics statement, ethical approval, IRB approval, informed consent"),
    ("endnote", "Conclusion, future work, extended data"),
    ("appendix", "Appendix, appendices, supplementary/supporting material"),
    ("footnote", "Footnotes"),
    ("unknown", "Cannot determine the section type"),
)


def _llm_section_type_field_description() -> str:
    """Pydantic field description enumerating the allowed section types."""
    values = ", ".join(v for v, _ in _LLM_SECTION_TYPE_DESCRIPTIONS)
    return f"The canonical section type. Must be one of: {values}"


def _llm_section_type_prompt_block() -> str:
    """The prompt's ``- value: description`` lines, one per allowed type."""
    return "\n".join(f"- {v}: {d}" for v, d in _LLM_SECTION_TYPE_DESCRIPTIONS)


# Precomputed reverse lookup: alias → CanonicalSection (for O(1) exact match)
_ALIAS_EXACT_LOOKUP: dict[str, CanonicalSection] = {
    alias: section for section, aliases in CANONICAL_SECTION_ALIASES.items() for alias in aliases
}


def _build_alias_substring_re() -> re.Pattern[str]:
    pairs: list[tuple[str, CanonicalSection]] = []
    for section, aliases in CANONICAL_SECTION_ALIASES.items():
        for alias in aliases:
            pairs.append((alias, section))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    pattern = "|".join(rf"\b{re.escape(alias)}\b" for alias, _ in pairs)
    return re.compile(pattern)


_ALIAS_SUBSTRING_RE = _build_alias_substring_re()
_ALIAS_TO_SECTION: dict[str, CanonicalSection] = {
    alias: section for section, aliases in CANONICAL_SECTION_ALIASES.items() for alias in aliases
}


_LLM_KEY_DECORATION_RE = re.compile(r"^[^A-Za-z_]+|[^A-Za-z0-9_]+$")


def _sanitize_llm_keys(data: object) -> object:
    """Strip stray non-word decoration from dict keys emitted by reasoning models.

    GPT-5-class models occasionally produce keys like ``<header`` or ``header>``
    (Harmony channel-tag leakage or prompt-mirroring). Without sanitization the
    pydantic field lookup misses and the whole LLM response is discarded.
    """
    if not isinstance(data, dict):
        return data
    cleaned: dict = {}
    for k, v in data.items():
        if isinstance(k, str):
            new_k = _LLM_KEY_DECORATION_RE.sub("", k)
            cleaned[new_k or k] = v
        else:
            cleaned[k] = v
    return cleaned


class SectionClassification(BaseModel):
    """One source header paired with its canonical section classification."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy(
        semantics={"header": "verbatim-string"},
        choices={"section_type": tuple(value for value, _ in _LLM_SECTION_TYPE_DESCRIPTIONS)},
    )

    header: str = Field(description="The original header text")
    section_type: str = Field(description=_llm_section_type_field_description())

    _sanitize = model_validator(mode="before")(_sanitize_llm_keys)


class SectionClassificationResult(BaseModel):
    """Structured result for a batch of section headers."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy()

    classifications: list[SectionClassification] = Field(
        description="List of classified section headers"
    )

    _sanitize = model_validator(mode="before")(_sanitize_llm_keys)


# A word-boundary substring alias hit is trusted outright only when the alias
# accounts for at least this fraction of the (normalized) header — e.g.
# "materials and methods section" (0.72) is trusted, "limitations of existing
# theories" (0.34) is not and must be confirmed by the model/LLM.
_SUBSTRING_TRUST_COVERAGE = 0.6
# Score assigned when an untrusted alias hit is used as the last-resort
# fallback after the model/LLM returned UNKNOWN.
_ALIAS_PRIOR_SCORE = 0.6


def _classify_lookup_full(header_text: str) -> tuple[CanonicalSection, float, bool]:
    """Alias lookup returning ``(section, score, trusted)``.

    Exact matches are trusted at 1.0. Word-boundary substring matches score
    0.95 but are only *trusted* when the alias covers most of the header
    (``_SUBSTRING_TRUST_COVERAGE``); generic one-word aliases ("model",
    "limitations") inside longer headers are returned untrusted so the
    caller can confirm them with the trained model / LLM.
    """
    header = header_text.lower()  # defensive: never trust the caller to normalize

    # Direct Lookup (O(1) via precomputed reverse dict)
    exact = _ALIAS_EXACT_LOOKUP.get(header)
    if exact is not None:
        return exact, 1.0, True

    # Word-boundary Substring Lookup — longest match wins to avoid short
    # aliases (e.g. "model") stealing headers from longer, more
    # specific aliases (e.g. "statistical methods").
    # Uses \b word boundaries to prevent partial-word matches
    # (e.g. "preferences" must not match the "references" alias).
    best: tuple[CanonicalSection, int] | None = None
    for m in _ALIAS_SUBSTRING_RE.finditer(header):
        alias = m.group(0)
        section = _ALIAS_TO_SECTION.get(alias)
        if section is None:
            continue
        if best is None or len(alias) > best[1]:
            best = (section, len(alias))

    if best is not None:
        coverage = best[1] / max(len(header), 1)
        return best[0], 0.95, coverage >= _SUBSTRING_TRUST_COVERAGE

    return CanonicalSection.UNKNOWN, 0.0, False


def _classify_lookup(header_text: str) -> tuple[CanonicalSection, float]:
    """Exact or substring alias match; substring hits score 0.95.

    Lookup-only contract used by the sync/no_llm path — trust semantics are
    intentionally ignored here (no model to confirm against).
    """
    section, score, _trusted = _classify_lookup_full(header_text)
    return section, score


async def _classify_llm_batch(
    header_texts: list[str],
    body_snippets: list[str] | None = None,
    llm_client: LlmClient | None = None,
    scope_context: list[str | None] | None = None,
    *,
    settings: GlobalSettings | None = None,
) -> list[tuple[CanonicalSection, float]]:
    """Use LLM to classify unknown section headers in a single batch call.

    Args:
        header_texts: Section headers to classify.
        body_snippets: Optional body text snippets (one per header) to provide
            additional context for ambiguous headers. Truncated to first 200 chars.
        llm_client: Optional pre-existing LlmClient to reuse. If None, creates
            a temporary one (legacy behaviour).
        scope_context: Optional per-header study-marker labels (parallel list,
            ``None`` outside any scope). Adds one terse context line per scope
            just above the section list; with no labels the prompt is
            byte-identical to the scopeless one. The marker regex — not the
            LLM — stays the sole authority on scope boundaries.
    """
    if not header_texts:
        return []

    from bibr.clients.llm import LLMClient, _task_max_tokens

    effective = settings if settings is not None else snapshot_settings()
    owns_client = llm_client is None
    if owns_client:
        llm_client = LLMClient(settings=effective)
    assert llm_client is not None  # noqa: S101 — owns_client branch sets it

    try:
        # Build the section block with optional body snippets for context.
        # Plain numbered format (no brackets/colons) avoids tempting reasoning
        # models into echoing decorated JSON keys in the response.
        lines = []
        for i, text in enumerate(header_texts):
            line = f"{i + 1}. {text}"
            if body_snippets and i < len(body_snippets) and body_snippets[i]:
                snippet = body_snippets[i][:500].replace("\n", " ").strip()
                if snippet:
                    line += f" — opening text: {snippet}..."
            lines.append(line)
        sections_block = "\n".join(lines)

        body_instruction = ""
        if body_snippets:
            body_instruction = (
                "\nUse the opening-text snippet to help disambiguate headers "
                "that could belong to multiple section types.\n"
            )

        # One terse line per study scope ("Headers 3..6 are inside 'Study 2'
        # of a multi-study paper."). Empty when no headers carry a scope label
        # — keeps the prompt byte-identical to the scopeless one. Placed just
        # above the section list (not at the front) so the static instruction
        # stays a byte-identical leading prefix across calls — local MLX
        # runtimes reuse cached prefill for it (radix prefix cache).
        scope_block = ""
        if scope_context:
            spans: dict[str, list[int]] = {}
            for i, label in enumerate(scope_context[: len(header_texts)]):
                if label:
                    spans.setdefault(label, [i, i])[1] = i
            if spans:
                scope_lines = [
                    f"Headers {lo + 1}..{hi + 1} are inside '{label}' of a multi-study paper."
                    for label, (lo, hi) in spans.items()
                ]
                scope_block = "\n".join(scope_lines) + "\n\n"

        instructions = f"""Classify each section header into a canonical section type for a scientific paper.

Valid section types:
{_llm_section_type_prompt_block()}
{body_instruction}
"""
        document = f"""{scope_block}Sections to classify:
{sections_block}
"""

        from bibr.clients.prompts import part

        result = await llm_client.invoke_structured(
            SectionClassificationResult,
            [
                {
                    "role": "user",
                    "content": [
                        part(instructions, nuextract_role="instructions"),
                        part(document, nuextract_role="document"),
                    ],
                }
            ],
            "You are a scientific paper section classifier.",
            label="section_classifier",
            max_tokens=_task_max_tokens(effective, effective.llm.section_max_tokens),
        )

        classifications = result.classifications
        results = []
        # Map LLM results back to their inputs. Position alone was trusted
        # here, so a short or reordered response shifted every classification
        # after the first omission onto the wrong header — silently, and for
        # the rest of the document. The response echoes each header, so verify
        # the echo before trusting the position.
        normalized = [_normalize_header_echo(text) for text in header_texts]
        # Models that don't echo (empty header field) keep the old positional
        # mapping, but only when the count also matches exactly.
        positional_ok = len(classifications) == len(header_texts) and all(
            not _normalize_header_echo(cls.header)
            or _normalize_header_echo(cls.header) == normalized[idx]
            for idx, cls in enumerate(classifications)
        )
        llm_map = {}
        if positional_ok:
            for idx, cls in enumerate(classifications):
                llm_map[idx] = cls.section_type
        else:
            logger.warning(
                "LLM section classification did not line up with its input "
                "(%d classifications for %d headers); matching on echoed header text",
                len(classifications),
                len(header_texts),
            )
            by_text: dict[str, list[int]] = {}
            for i, text in enumerate(normalized):
                by_text.setdefault(text, []).append(i)
            for cls in classifications:
                echoed = _normalize_header_echo(cls.header)
                # Unmatched classifications are dropped rather than assigned to
                # the next free slot: an unplaceable label is no evidence about
                # any particular header, and those headers fall through to
                # ``unknown`` below.
                for i in by_text.get(echoed, ()):
                    if i not in llm_map:
                        llm_map[i] = cls.section_type
                        break

        for i, _text in enumerate(header_texts):
            section_val = llm_map.get(i, "unknown")
            if section_val in _VALID_SECTION_VALUES:
                canon = CanonicalSection(section_val)
                # LLM-returned "unknown" is no more confident than a lookup miss.
                score = (
                    effective.layout.section_classification_score
                    if canon != CanonicalSection.UNKNOWN
                    else 0.0
                )
                results.append((canon, score))
            else:
                results.append((CanonicalSection.UNKNOWN, 0.0))

        return results

    except ProcessingError:
        raise
    except Exception as e:
        logger.warning("LLM section classification failed: %s", e)
        return [(CanonicalSection.UNKNOWN, 0.0)] * len(header_texts)
    finally:
        if owns_client and llm_client is not None:
            await llm_client.close()


def classify_header(header_text: str) -> tuple[CanonicalSection, float]:
    """Classify a single header using lookup only (sync convenience function).

    For headers that don't match lookup, returns UNKNOWN.
    Use classify_headers_batch for LLM fallback on unknowns.
    """
    clean_text = normalize_text(header_text)
    return _classify_lookup(clean_text)


async def classify_headers_batch_async(
    header_texts: list[str],
    body_snippets: list[str] | None = None,
    llm_client: LlmClient | None = None,
    scope_context: list[str | None] | None = None,
    *,
    classifier_resources: ClassifierResources | None = None,
    settings: GlobalSettings | None = None,
    degradation_warnings: list[str] | None = None,
) -> list[tuple[CanonicalSection, float, bool | None, str | None]]:
    """Classify multiple headers efficiently (async version with model fallback).

    Pipeline (per header):
      1. Fast alias lookup — trusted exact/word-boundary aliases bypass both
         the trained model and the LLM.
      2. For lookup misses: if ``Settings.ml.section_classifier_model_id`` is
         set, route through the trained MiniLM classifier (which also predicts
         ``is_top_level``). Otherwise fall through to the LLM.

    Returns a list of ``(canonical, score, is_top_level, source)`` tuples,
    parallel to ``header_texts``. ``is_top_level`` is ``True``/``False`` only
    when the trained model produced it; ``None`` for alias-lookup hits and the
    LLM path. ``source`` names the tier that produced the decision
    ("exact_alias", "substring_alias", "model", "llm", "alias_prior") or
    ``None`` when the result is UNKNOWN.

    Args:
        header_texts: Section headers to classify, in document order — the
            order is used to derive each header's document-context (relative
            position, previous/next heading) for the trained-model path.
        body_snippets: Optional body text snippets (one per header, parallel
            list) passed to the trained model / LLM for ambiguous headers.
        llm_client: Optional pre-existing LlmClient to reuse for the LLM path.
        scope_context: Optional study-marker labels (one per header, parallel
            list; ``None`` outside any scope). Forwarded to the LLM fallback
            as terse per-scope prompt context — the trained-model path
            ignores it.
    """
    effective = settings if settings is not None else snapshot_settings()
    explicit_runtime = classifier_resources is not None or settings is not None
    n = len(header_texts)
    results: list[tuple[CanonicalSection, float, bool | None, str | None] | None] = [None] * n
    unknown_indices: list[int] = []
    unknown_texts: list[str] = []
    unknown_bodies: list[str] = []
    unknown_scopes: list[str | None] = []
    # Raw (un-normalized) neighbor headers and relative position, one per
    # unknown — only consumed by the trained-model path's composite dedup key.
    unknown_prev: list[str] = []
    unknown_next: list[str] = []
    unknown_relpos: list[float] = []
    # Untrusted alias hits (generic alias inside a longer header): fallback
    # prior per normalized header text, applied only when model+LLM say UNKNOWN.
    alias_priors: dict[str, CanonicalSection] = {}

    for i, text in enumerate(header_texts):
        clean_text = normalize_text(text)
        if is_exact_front_matter_furniture(clean_text):
            # Printed page furniture ("OPEN ACCESS", "SHORT COMMUNICATION",
            # "Check for updates"). There is no vocabulary member for it, so
            # the model/LLM tiers are forced to guess and confidently answer
            # TITLE — which then seeds a false front-matter record. UNKNOWN is
            # the honest answer and still reaches front matter, where
            # ``_is_false_title_seed`` vetoes it by the same list.
            results[i] = (CanonicalSection.UNKNOWN, 0.0, None, None)
            continue
        section, score, trusted = _classify_lookup_full(clean_text)
        if section != CanonicalSection.UNKNOWN and trusted:
            # Trusted alias — bypass both trained model and LLM. Score 1.0 is an
            # exact table hit; 0.95 is a trusted word-boundary substring match.
            source = "exact_alias" if score >= 1.0 else "substring_alias"
            results[i] = (section, float(score), None, source)
        else:
            if clean_text:
                if section != CanonicalSection.UNKNOWN:
                    alias_priors[clean_text] = section
                unknown_indices.append(i)
                unknown_texts.append(clean_text)
                unknown_bodies.append(
                    body_snippets[i] if body_snippets and i < len(body_snippets) else ""
                )
                unknown_scopes.append(
                    scope_context[i] if scope_context and i < len(scope_context) else None
                )
                unknown_prev.append(header_texts[i - 1] if i > 0 else "")
                unknown_next.append(header_texts[i + 1] if i < n - 1 else "")
                unknown_relpos.append(i / max(n - 1, 1))
            else:
                results[i] = (CanonicalSection.UNKNOWN, 0.0, None, None)

    if unknown_texts:
        trained_available = classifier_resources is not None
        if not trained_available:
            model = (
                await _get_trained_model_async(effective)
                if settings is not None
                else await _get_trained_model_async()
            )
            trained_available = model is not None
            if not trained_available and effective.ml.section_classifier_model_id:
                _note_section_classifier_degraded(degradation_warnings)
        if trained_available:
            # Composite dedup key: header text alone collapses two headers
            # with the same wording but different document context (e.g. two
            # "Participants" sections in different studies) onto one
            # prediction. Folding in prev/next heading + position bucket
            # keeps such pairs distinct for the trained model.
            keys = [
                f"{t}\x1f{p}\x1f{nx}\x1f{_position_bucket(rp)}"
                for t, p, nx, rp in zip(
                    unknown_texts, unknown_prev, unknown_next, unknown_relpos, strict=True
                )
            ]
            first_idx_for_key: dict[str, int] = {}
            for local_idx, key in enumerate(keys):
                first_idx_for_key.setdefault(key, local_idx)
            unique_keys = list(first_idx_for_key.keys())

            items = [
                HeaderContext(
                    heading=unknown_texts[li],
                    body=unknown_bodies[li],
                    relative_position=unknown_relpos[li],
                    prev_heading=unknown_prev[li],
                    next_heading=unknown_next[li],
                )
                for li in (first_idx_for_key[k] for k in unique_keys)
            ]
            try:
                if explicit_runtime:
                    trained_results = await _classify_trained_batch(
                        items,
                        classifier_resources=classifier_resources,
                        settings=effective,
                    )
                else:
                    trained_results = await _classify_trained_batch(items)
            except Exception as exc:  # noqa: BLE001 - trained inference is an optional tier
                logger.warning(
                    "Trained section classifier inference failed (%s); "
                    "falling back to alias/LLM classification",
                    type(exc).__name__,
                )
                _note_section_classifier_degraded(degradation_warnings)
                trained_results = []
            if len(trained_results) != len(unique_keys):
                logger.warning(
                    "Trained section classifier returned %d predictions for %d headers; "
                    "falling back to the LLM tier",
                    len(trained_results),
                    len(unique_keys),
                )
                if effective.ml.section_classifier_model_id:
                    _note_section_classifier_degraded(degradation_warnings)
                trained_results = [(CanonicalSection.UNKNOWN, 0.0, None)] * len(unique_keys)
            unique_map: dict[str, tuple[CanonicalSection, float, bool | None, str | None]] = {
                key: (
                    *trained_results[j],
                    "model" if trained_results[j][0] != CanonicalSection.UNKNOWN else None,
                )
                for j, key in enumerate(unique_keys)
            }
            # Low-confidence predictions collapsed to UNKNOWN by
            # _classify_trained_batch. Discarding them loses information the
            # LLM tier can recover (rare classes like ethics are exactly
            # where the trained model is weakest), so escalate the misses.
            if effective.ml.section_classifier_llm_escalation:
                esc_keys = [k for k in unique_keys if unique_map[k][0] == CanonicalSection.UNKNOWN]
                if esc_keys:
                    # Dedupe by plain header text within the LLM sub-call —
                    # the LLM prompt has no use for position/neighbor context,
                    # so distinct keys sharing text would otherwise bloat it.
                    esc_text_for_key = {k: unknown_texts[first_idx_for_key[k]] for k in esc_keys}
                    first_idx_for_text: dict[str, int] = {}
                    for k in esc_keys:
                        first_idx_for_text.setdefault(esc_text_for_key[k], first_idx_for_key[k])
                    esc_texts = list(first_idx_for_text.keys())
                    esc_bodies = [unknown_bodies[first_idx_for_text[t]] for t in esc_texts]
                    esc_scopes = [unknown_scopes[first_idx_for_text[t]] for t in esc_texts]
                    scope_kwargs: dict = {"scope_context": esc_scopes} if any(esc_scopes) else {}
                    llm_kwargs = {
                        "body_snippets": esc_bodies if any(esc_bodies) else None,
                        "llm_client": llm_client,
                        **scope_kwargs,
                    }
                    if explicit_runtime:
                        llm_kwargs["settings"] = effective
                    llm_results = await _classify_llm_batch(esc_texts, **llm_kwargs)
                    text_result_map = dict(zip(esc_texts, llm_results, strict=True))
                    for k in esc_keys:
                        canon, score = text_result_map[esc_text_for_key[k]]
                        if canon != CanonicalSection.UNKNOWN:
                            unique_map[k] = (canon, score, None, "llm")

            # Last-resort fallback: model and LLM both said UNKNOWN, but the
            # header did contain a known alias — better a weak alias signal
            # than UNKNOWN (folded into the previous section downstream).
            for k in unique_keys:
                text = unknown_texts[first_idx_for_key[k]]
                prior = alias_priors.get(text)
                if prior is not None and unique_map[k][0] == CanonicalSection.UNKNOWN:
                    unique_map[k] = (prior, _ALIAS_PRIOR_SCORE, None, "alias_prior")

            for idx, key in zip(unknown_indices, keys, strict=True):
                results[idx] = unique_map[key]
        else:
            # Alias-context-free path (no trained model configured): dedupe
            # purely on header text, as before — the LLM path has no signal
            # that depends on document position.
            unique_texts = list(dict.fromkeys(unknown_texts))
            unique_body_map: dict[str, str] = {}
            unique_scope_map: dict[str, str | None] = {}
            for text, body, scope in zip(
                unknown_texts, unknown_bodies, unknown_scopes, strict=True
            ):
                if text not in unique_body_map:
                    unique_body_map[text] = body
                    unique_scope_map[text] = scope
            unique_bodies = [unique_body_map[t] for t in unique_texts]
            unique_scopes = [unique_scope_map[t] for t in unique_texts]

            # scope_context is only passed when at least one header carries a
            # study-marker label — the no-marker call stays identical to the
            # scopeless one.
            scope_kwargs = {"scope_context": unique_scopes} if any(unique_scopes) else {}
            llm_kwargs = {
                "body_snippets": unique_bodies if any(unique_bodies) else None,
                "llm_client": llm_client,
                **scope_kwargs,
            }
            if explicit_runtime:
                llm_kwargs["settings"] = effective
            llm_results_unique = await _classify_llm_batch(unique_texts, **llm_kwargs)
            unique_map_by_text = {
                text: (canon, score, None, "llm" if canon != CanonicalSection.UNKNOWN else None)
                for text, (canon, score) in zip(unique_texts, llm_results_unique, strict=True)
            }

            for text, prior in alias_priors.items():
                if (
                    text in unique_map_by_text
                    and unique_map_by_text[text][0] == CanonicalSection.UNKNOWN
                ):
                    unique_map_by_text[text] = (prior, _ALIAS_PRIOR_SCORE, None, "alias_prior")

            for idx, text in zip(unknown_indices, unknown_texts, strict=True):
                results[idx] = unique_map_by_text[text]

    # All slots are filled by this point — either by lookup, the trained model,
    # the LLM, or the empty-string short-circuit. mypy can't see that, so the
    # cast keeps the typed return clean.
    return [r if r is not None else (CanonicalSection.UNKNOWN, 0.0, None, None) for r in results]


def classify_headers_batch(
    header_texts: list[str],
) -> list[tuple[CanonicalSection, float]]:
    """Classify multiple headers (sync wrapper).

    Uses lookup only — no LLM fallback. For LLM fallback, use
    classify_headers_batch_async.
    """
    results = []
    for text in header_texts:
        clean_text = normalize_text(text)
        results.append(_classify_lookup(clean_text))
    return results
