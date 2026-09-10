"""Inline citation detection and resolution — links sentences to bib entries.

3-tier hybrid approach:
  Tier 1: Numeric citations — bracket [1], [3-5] and superscript ^{3}, ^{15,16}
  Tier 2: Author-year citations (Smith, 2020) and Smith (2020)
  Tier 3: LLM fallback for ambiguous/unresolved candidates
"""

import logging
import re
import unicodedata
from dataclasses import replace

from bibr.exceptions import ProcessingError
from bibr.paper_contents import (
    CanonicalSection,
    CitationCandidate,
    CitationLinkingReceipt,
    PaperSentence,
    PaperXref,
)
from bibr.structure.xref_utils import expand_int_range
from bibr.utils.text import collapse_ws

logger = logging.getLogger(__name__)


def _normalize_citation_text(text: str) -> str:
    """Collapse internal whitespace (line breaks, NBSPs) inside citation text."""
    return collapse_ws(text)


def _citation_span_is_resolved(
    text_id: int,
    cite_text: str,
    resolved_pairs: set[tuple[int, str]],
) -> bool:
    """Return whether every component of a citation span is already linked."""
    normalized = _normalize_citation_text(cite_text)
    if (text_id, normalized) in resolved_pairs:
        return True
    inner = normalized.strip("()[]")
    components = [
        _normalize_citation_text(component).strip("()[]")
        for component in re.split(r"\s*;\s*", inner)
        if component.strip()
    ]
    return bool(components) and all(
        (text_id, component) in resolved_pairs for component in components
    )


# Bracketed content that LOOKS like a citation but isn't.  Used as a Tier 3
# pre-filter so we don't ask the LLM (and risk it inventing matches) for:
#   [.59, .72]  [-.12, .04]  [0.000, 0.114]  — confidence intervals
#   [Item 5]    [Item 7]                     — survey-item references
#   [IRR]       [OR]         [missed opportunity]  — uppercase abbreviations
_NON_CITATION_BRACKET_RE = re.compile(
    r"^\s*(?:"
    r"[-+]?\s*\.?\d+(?:\.\d+)?\s*[,;–\-]\s*[-+]?\s*\.?\d+(?:\.\d+)?\s*"  # CI / range
    r"|"
    r"item\s+\d+"  # [Item 5]
    r"|"
    r"(?-i:[A-Z]{2,5})"  # [IRR], [OR], [SE]
    r"|"
    r"(?-i:[a-z])[A-Za-z\s\-']*"  # [sic], [missed opportunity]
    r"|"
    r"[A-Za-z][A-Za-z\-']*\s[A-Za-z\s\-']+"  # [Emphasis Added] — but not [Smith]
    r")\s*$",
    re.IGNORECASE,
)


def _looks_like_non_citation_bracket(content: str) -> bool:
    """Return True if bracketed content is clearly not an inline citation.

    Used to filter Tier-3 LLM candidates so confidence intervals, item refs,
    and uppercase abbreviations don't get assigned spurious bib_ids.
    """
    return bool(_NON_CITATION_BRACKET_RE.match(content))


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Tier 1a: Numeric bracket citations [1], [3-5], [1,4,7]
NUMERIC_CITE_RE = re.compile(
    r"\[(\d+(?:\s*[-\u2013]\s*\d+)?(?:\s*[,;]\s*\d+(?:\s*[-\u2013]\s*\d+)?)*)\]"
)

# Bare parenthetical-numeric citations (3), (7, 8), (6-8).
# Accepted when recurring markers and printed bibliography evidence agree.
PAREN_NUMERIC_CITE_RE = re.compile(r"\((\d{1,3}(?:\s*[,\u2013-]\s*\d{1,3})*)\)")

_PROCEDURAL_LABEL_BEFORE_RE = re.compile(
    r"(?:^|\b)(?:step|phase|stage|criterion|criteria|option|item)\s*$",
    re.IGNORECASE,
)
_AGGREGATE_COUNT_CUE_RE = re.compile(
    r"\b(?:many|majority|minority|half|total|number|count)\b|\bn\s*=\s*$",
    re.IGNORECASE,
)
_AGGREGATE_COUNT_NOUN_RE = re.compile(
    r"\b(?:participants?|respondents?|indicators?|cases?|samples?|observations?|records?)\b",
    re.IGNORECASE,
)
_COUNT_COPULA_AFTER_RE = re.compile(r"^\s*(?:was|were|is|are|had|have|has)\b", re.IGNORECASE)
_COUNT_CONTAINER_BEFORE_RE = re.compile(
    r"\b(?:domain|subdomain|category|group|class|sector)\s*$",
    re.IGNORECASE,
)

# Tier 1d: OCR-flattened superscript citations \u2014 digits glued to the END of a
# word or after punctuation where GLM-OCR dropped the superscript markup:
# "traits1", "characteristics2,3", "rehabilitation,1", "effectiveness.7,10,11".
# group(1) is the trailing word (with optional trailing punctuation), group(2)
# is the citation digit run.  The lookahead requires the run to be followed by
# whitespace/punctuation/end so mid-word or measurement digits are skipped.
FLATTENED_SUP_CITE_RE = re.compile(
    r"([a-zA-Z][a-zA-Z\)\]]*[.,;:]?)(\d{1,3}(?:[,\u2013-]\d{1,3})*)(?=[\s.,;:)\]]|$)"
)

_MIN_FLATTENED_ANCHOR_HITS = 2
_MIN_FLATTENED_AMBIGUOUS_PROOF_HITS = _MIN_FLATTENED_ANCHOR_HITS


def _flattened_marker_is_high_specificity(word: str, digits: str) -> bool:
    """Return whether a flattened marker is intrinsically citation-shaped.

    Citation punctuation and multi-ID/range runs are strong anchors. A bare
    word-plus-single-ID carrier remains ambiguous regardless of letter case;
    model names, instruments, disease labels, and real citations are locally
    indistinguishable in that surface form.
    """

    carrier = word.rstrip(".,;:")
    punctuation_attached = len(carrier) < len(word)
    multiple_ids = len(_expand_numeric_range(digits)) > 1
    return punctuation_attached or multiple_ids


_FLATTENED_MODEL_CONTEXT_BEFORE_RE = re.compile(
    r"\b(?:assessed(?:\s+\w+){0,3}\s+using|trained|tuned|fine[- ]?tuned|using)\s+$",
    re.IGNORECASE,
)
_FLATTENED_MODEL_CONTEXT_AFTER_RE = re.compile(
    r"^\s*(?:model|network|architecture|instrument|scale|questionnaire|score)\b",
    re.IGNORECASE,
)
_FLATTENED_ENTITY_CONTEXT_BEFORE_RE = re.compile(
    r"\b(?:disease|entity|class|category|variant|strain)\s+label"
    r"(?:\s+(?:is|was|named|called))?\s+$",
    re.IGNORECASE,
)


def _flattened_carrier(word: str) -> str:
    """Return the lexical carrier immediately preceding a flattened marker."""

    return word.rstrip(".,;:)]").lstrip("([")


def _citation_lexical_tokens(value: str | None) -> set[str]:
    """Build accent-insensitive tokens for local carrier-to-reference proof."""

    if not value:
        return set()
    normalized = "".join(
        char
        for char in unicodedata.normalize("NFKD", value).casefold()
        if not unicodedata.combining(char)
    )
    return set(re.findall(r"[a-z0-9]+", normalized))


def _flattened_carrier_requires_grounding(carrier: str) -> bool:
    """Return whether a bare single-ID carrier is lexically ambiguous.

    Lowercase prose carriers (``alone1``/``productivity2``) are the ordinary
    shape produced when OCR flattens a superscript. Acronyms, CamelCase model
    names, and title-cased entity names are also common non-citation tokens,
    so those shapes need proof from their exact target reference.
    """

    letters = "".join(char for char in carrier if char.isalpha())
    return bool(letters) and not letters.islower()


_ACRONYM_DEFINITION_RE = re.compile(
    r"\b((?:[A-Za-z][A-Za-z'-]*\s+){1,7}[A-Za-z][A-Za-z'-]*)"
    r"\s*\(([A-Z][A-Z0-9-]{1,9})\)"
)
_ACRONYM_ALIAS_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "before",
        "between",
        "from",
        "global",
        "health",
        "into",
        "plan",
        "regional",
        "report",
        "study",
        "system",
        "with",
        "within",
        "without",
        "world",
    }
)


def _body_acronym_alias_tokens(body_sents) -> dict[str, set[str]]:
    """Return meaningful expansion tokens for explicitly defined acronyms."""

    aliases: dict[str, set[str]] = {}
    for sentence in body_sents:
        for match in _ACRONYM_DEFINITION_RE.finditer(sentence.text):
            acronym_tokens = _citation_lexical_tokens(match.group(2))
            if len(acronym_tokens) != 1:
                continue
            expansion = {
                token
                for token in _citation_lexical_tokens(match.group(1))
                if len(token) >= 5 and token not in _ACRONYM_ALIAS_STOPWORDS
            }
            if expansion:
                aliases.setdefault(next(iter(acronym_tokens)), set()).update(expansion)
    return aliases


def _reference_supports_flattened_carrier(
    reference,
    carrier: str,
    *,
    source_text: str | None,
    acronym_alias_tokens: dict[str, set[str]],
) -> bool:
    """Ground *carrier* in bibliographic text from its exact target only."""

    carrier_tokens = _citation_lexical_tokens(carrier)
    if len(carrier_tokens) != 1:
        return False
    target_tokens: set[str] = set()
    for field in ("title", "container", "publisher"):
        target_tokens.update(_citation_lexical_tokens(getattr(reference, field, None)))
    target_tokens.update(_citation_lexical_tokens(source_text))
    if carrier_tokens & target_tokens:
        return True
    carrier_key = next(iter(carrier_tokens))
    return bool(acronym_alias_tokens.get(carrier_key, set()) & target_tokens)


def _flattened_local_context_reasons(text: str, start: int, end: int) -> tuple[str, ...]:
    """Return local model/instrument/entity evidence for an ungrounded token."""

    before = text[max(0, start - 64) : start]
    after = text[end : min(len(text), end + 48)]
    reasons: list[str] = []
    if _FLATTENED_MODEL_CONTEXT_BEFORE_RE.search(
        before
    ) or _FLATTENED_MODEL_CONTEXT_AFTER_RE.search(after):
        reasons.append("model_or_instrument_context")
    if _FLATTENED_ENTITY_CONTEXT_BEFORE_RE.search(before):
        reasons.append("entity_label_context")
    return tuple(reasons)


# Tier 3: Parenthetical author-year (Smith, 2020), (Smith & Jones, 2020; Brown, 2019)
PAREN_AUTHOR_YEAR_RE = re.compile(
    r"\("
    r"([A-Z][a-zA-Z\u00C0-\u024F'\-]+"
    r"(?:\s+(?:&|and)\s+[A-Z][a-zA-Z\u00C0-\u024F'\-]+)?"
    r"(?:\s+et\s+al\.?)?)"
    r",?\s*(\d{4}[a-z]?)"
    r"((?:\s*;\s*[A-Z][a-zA-Z\u00C0-\u024F'\-]+"
    r"(?:\s+(?:&|and)\s+[A-Z][a-zA-Z\u00C0-\u024F'\-]+)?"
    r"(?:\s+et\s+al\.?)?,?\s*\d{4}[a-z]?)*)"
    r"\)"
)

# Tier 3: Narrative author-year — Smith (2020), Smith et al. (2020)
NARRATIVE_AUTHOR_YEAR_RE = re.compile(
    r"([A-Z][a-zA-Z\u00C0-\u024F'\-]+"
    r"(?:\s+(?:&|and)\s+[A-Z][a-zA-Z\u00C0-\u024F'\-]+)?"
    r"(?:\s+et\s+al\.?)?)"
    r"\s*\((\d{4}[a-z]?)\)"
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _expand_numeric_range(text: str) -> list[int]:
    """Expand a numeric citation group like '1-3,5' into [1, 2, 3, 5].

    Reversed ranges (e.g. '5-3' from OCR errors) are interpreted in
    ascending order \u2014 citations should be recoverable even from typos.
    """
    result: list[int] = []
    # Split on comma or semicolon
    parts = re.split(r"[,;]", text)
    for part in parts:
        part = part.strip()
        # Check for range (hyphen or en-dash)
        range_match = re.match(r"(\d+)\s*[-\u2013]\s*(\d+)", part)
        if range_match:
            a, b = int(range_match.group(1)), int(range_match.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            result.extend(expand_int_range(lo, hi))
        elif part.isdigit():
            result.append(int(part))
    return result


# A large gap in a two-number bracket is treated as interval-like evidence rather than a
# citation pair.
_CI_SPREAD_THRESHOLD = 15


def _is_likely_citation_bracket(group_text: str, nums: list[int], valid_bib_ids: set[int]) -> bool:
    """Filter out numeric brackets that look more like CIs than citations.

    Two heuristics, both targeting cases where ``[A, B]`` happens to have one
    number that coincidentally matches a bib_id:

    1. **Out-of-range:** if any number exceeds the largest valid bib_id, the
       whole bracket is suspect (real citation lists don't reference refs
       that don't exist).
    2. **Wide-gap 2-tuple:** comma-separated brackets with exactly two
       numbers and a spread above ``_CI_SPREAD_THRESHOLD`` are almost always
       confidence intervals or percentile ranges.

    Range brackets like ``[3-5]`` and longer lists like ``[1, 7, 12]`` are
    not affected.
    """
    if not nums or not valid_bib_ids:
        return bool(nums)

    max_bib = max(valid_bib_ids)
    if any(n > max_bib for n in nums):
        return False

    # bib_ids are 1-indexed: a bracket containing 0 (or a negative) is never a
    # citation.  Rejects maths notation like "[0, 1]" / "[0,1]" (the unit
    # interval) whose only in-range member coincidentally matches a bib_id.
    if any(n < 1 for n in nums):
        return False

    # Only apply the wide-gap rule to comma/semicolon-separated 2-tuples
    # where the source group has no range marker.  ``[3-5]`` should still
    # expand to [3, 4, 5] regardless of spread.
    return not (
        len(nums) == 2
        and "-" not in group_text
        and "\u2013" not in group_text
        and abs(nums[0] - nums[1]) >= _CI_SPREAD_THRESHOLD
    )


def _get_body_sentences(sentences, sections) -> list[PaperSentence]:
    """Return sentences from body sections only.

    Excludes references, every title section (which can contain author bylines
    with affiliation superscripts), figure/table captions, and footnotes.
    Abstract sentences remain eligible because scientific abstracts can cite
    bibliography entries. Display-formula sentences are excluded: they export
    as "[equation]" placeholders, and in a numbered bibliography their equation
    tags ("(2)") pass every numeric-citation filter.
    """
    _EXCLUDE_TYPES = {
        CanonicalSection.REFERENCES,
        CanonicalSection.FOOTNOTE,
        CanonicalSection.FIGURE,
        CanonicalSection.TABLE,
    }
    exclude_ids = set()
    for s in sections:
        # Every TITLE-typed section is front matter, including genuine title
        # sections at level > 0 and the synthetic level-0 root/title placeholder.
        # Level-0 UNKNOWN sections remain eligible body-continuation text.
        if s.section_type in _EXCLUDE_TYPES or s.section_type == CanonicalSection.TITLE:
            exclude_ids.add(s.section_id)
    body = [s for s in sentences if not s.is_display_formula]
    if not exclude_ids:
        return body
    return [s for s in body if s.section_id not in exclude_ids]


# ---------------------------------------------------------------------------
# Numeric citation style evidence
# ---------------------------------------------------------------------------

# Minimum number of distinct sentences supporting a parenthetical citation style.
_MIN_FALLBACK_STYLE_HITS = 3


def _has_numeric_context(text: str, start: int, end: int) -> bool:
    """Return True when a parenthetical group sits in a maths/CI/decimal context.

    Rejects ``5(3)``, ``.05(3)``, ``(3)=``, ``(3)%``, ``(3)7`` — adjacency to a
    digit, ``=`` or ``%`` marks the parentheses as arithmetic rather than a
    citation.  Trailing punctuation (``(3).``) and whitespace are fine.
    """
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    if before.isdigit() or before in ("=", "%"):
        return True
    if before == "." and start - 2 >= 0 and text[start - 2].isdigit():
        return True
    return after.isdigit() or after in ("=", "%")


# Equation tags on display-formula lines: a trailing "(2)" (optionally followed
# by punctuation) or an explicit LaTeX "\tag{2}".
_EQ_TRAILING_TAG_RE = re.compile(r"\((\d{1,3})\)\s*[.,;]?\s*$")
_EQ_LATEX_TAG_RE = re.compile(r"\\tag\{(\d{1,3})\}")


def _collect_equation_tags(sentences) -> set[int]:
    """Return equation numbers tagged on the paper's display formulas.

    Prose mentions of these numbers ("Plugging (3) into (1)") look exactly
    like parenthetical-numeric citations, so the fallback tier cross-checks
    candidates against this set.
    """
    tags: set[int] = set()
    for s in sentences:
        if not s.is_display_formula or not s.text:
            continue
        m = _EQ_TRAILING_TAG_RE.search(s.text)
        if m:
            tags.add(int(m.group(1)))
        tags.update(int(t) for t in _EQ_LATEX_TAG_RE.findall(s.text))
    return tags


def _is_non_citation_digit_run(text: str, digits: str, end: int) -> bool:
    """Return True when a flattened digit run is not a citation marker.

    Rejects thousands-separated numbers (``60,000``), 4+ digit runs, standalone
    years, decimals (``3.5``) and measurements (``28x28``, ``5%``, ``3×``).
    """
    # thousands separator: comma then exactly three digits, e.g. "60,000"
    if re.search(r"\d,\d{3}(?!\d)", digits):
        return True
    # 4+ digit run (ids/measurements) — the pattern caps each number at 3
    # digits, so this is defensive.
    if any(len(p) >= 4 for p in re.split(r"[,–-]", digits)):
        return True
    if any(1900 <= n <= 2035 for n in _expand_numeric_range(digits)):
        return True
    # decimal continuation: "rate3.5"
    if end < len(text) and text[end] == "." and end + 1 < len(text) and text[end + 1].isdigit():
        return True
    # measurement units glued after the digits: "28x28", "3×", "5%"
    return end < len(text) and text[end] in ("x", "×", "%")


# ---------------------------------------------------------------------------
# Tier 2: Author-year citations — reference-anchored matcher in
# bibr.structure.citation_matcher (detect_bib_xrefs calls
# match_with_ambiguous directly).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tier 3: LLM fallback
# ---------------------------------------------------------------------------


async def _resolve_with_llm(ambiguous, references, llm_client, file_hash) -> list[PaperXref]:
    """Tier 3: batch unresolved citation candidates into a single LLM call.

    Args:
        ambiguous: list of (text_id, citation_text) tuples that couldn't be resolved
        references: list of PaperReference objects
        llm_client: LlmClient instance
        file_hash: file hash for rate-limit tracking

    Returns gracefully with [] on any failure.
    """
    if not ambiguous or not llm_client:
        return []

    try:
        result = await llm_client.resolve_citations(
            ambiguous_citations=ambiguous,
            reference_summary=[
                {
                    "bib_id": r.bib_id,
                    "author": r.authors or "",
                    "year": r.year,
                    "title": r.title,
                }
                for r in references
            ],
            file_hash=file_hash,
        )
        # Gate on real bib_ids, mirroring Tier 1's valid_bib_ids check — a
        # hallucinated bib_id would otherwise flow unvalidated into the
        # exported xrefs. text_id is deliberately NOT validated: the caller's
        # expansion keys on citation_text and re-derives text_id from the
        # original candidates, so a mangled text_id is harmless while a
        # correct match would be lost.
        valid_bib_ids = {r.bib_id for r in references}
        xrefs = []
        for match in result:
            if match.bib_id is None:
                continue
            if match.bib_id not in valid_bib_ids:
                logger.debug("Tier 3 dropped match with unknown bib_id=%s", match.bib_id)
                continue
            xrefs.append(
                PaperXref(
                    xref_id=match.bib_id,
                    xref_type="bib",
                    contents=_normalize_citation_text(match.citation_text),
                    text_id=match.text_id,
                )
            )
        return xrefs
    except ProcessingError:
        raise
    except Exception as e:
        logger.warning("Tier 3 LLM citation resolution failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Post-linking cleanup
# ---------------------------------------------------------------------------


_STRIP_CITE_SUP_RE = re.compile(
    r"\$\s+\^\{(\d+(?:\s*[-–,;]\s*\d+)*)\}\s+\$"
    r"|(?<![\d$])\^\{(\d+(?:\s*[-–,;]\s*\d+)*)\}"
)


def _is_citation_superscript(m: re.Match) -> str:
    # A whitespace-padded math wrapper (``$ ^{3} $``) is how GLM-OCR
    # commonly emits citation superscripts. Consume the wrapper atomically;
    # otherwise late inline-math cleanup pairs the orphaned dollars across
    # ordinary prose and leaves spurious ``$ ... $`` spans behind.
    if m.group(1) is not None:
        return ""
    pos = m.start()
    text = m.string
    if pos > 0 and text[pos - 1].isalpha() and (pos < 2 or not text[pos - 2].isalpha()):
        return str(m.group(0))
    return ""


def strip_citation_superscripts(
    sentences,
    sections,
    receipt: CitationLinkingReceipt | None = None,
) -> None:
    """Remove superscript citation markers from body text in-place.

    Must run **after** citation linking (which needs the ``^{N}`` patterns)
    and **before** ``finalize_text()`` (which would flatten ``^{3}`` to
    bare ``3`` instead of removing it).

    Preserves single-letter-variable exponents (``x^{2}``, ``p^{2}``)
    while stripping multi-letter-word superscripts (``effective^{9}``).
    """
    body_sents = _get_body_sentences(sentences, sections)
    if receipt is None:
        # Legacy direct-call behavior. The pipeline always supplies a receipt,
        # so production cleanup is constrained to accepted detector spans.
        for sent in body_sents:
            sent.text = _STRIP_CITE_SUP_RE.sub(_is_citation_superscript, sent.text)
        return

    accepted_by_text: dict[int, list[CitationCandidate]] = {}
    for candidate in receipt.candidates:
        if (
            candidate.accepted
            and candidate.style == "numeric"
            and "superscript_marker" in candidate.evidence
        ):
            accepted_by_text.setdefault(candidate.text_id, []).append(candidate)
    for sent in body_sents:
        spans = accepted_by_text.get(sent.text_id, [])
        for candidate in sorted(spans, key=lambda item: (item.start, item.end), reverse=True):
            if sent.text[candidate.start : candidate.end] == candidate.raw:
                sent.text = sent.text[: candidate.start] + sent.text[candidate.end :]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Receipt-bearing citation linker
# ---------------------------------------------------------------------------

_PRINTED_REFERENCE_PREFIX_RE = re.compile(
    r"^\s*(?:\[(\d{1,4})\]|\((\d{1,4})\)|(\d{1,4})[.)]|\^\{(\d{1,4})\})\s*"
)


def _printed_reference_sources(sentences, sections, references) -> dict[int, str]:
    """Return raw reference rows keyed by their matching printed marker.

    Native/JATS references can lack ``PaperReference.text_id`` even though the
    assembled REFERENCES sentences preserve their ordered printed prefixes.
    Restricting the fallback scan to REFERENCES-typed sections keeps those raw
    source strings eligible without treating dense internal IDs or body lists
    as bibliography-style evidence.
    """

    text_by_id = {sent.text_id: sent.text for sent in sentences if sent.text}
    valid_bib_ids = {reference.bib_id for reference in references}
    reference_section_ids = {
        section.section_id
        for section in sections
        if section.section_type == CanonicalSection.REFERENCES
    }
    sources: dict[int, str] = {}
    for reference in references:
        if reference.text_id is None:
            continue
        source = text_by_id.get(reference.text_id, "")
        match = _PRINTED_REFERENCE_PREFIX_RE.match(source)
        if match is None:
            continue
        marker = next((value for value in match.groups() if value is not None), None)
        if marker is not None and int(marker) == reference.bib_id:
            sources[reference.bib_id] = source
    for sentence in sentences:
        if sentence.section_id not in reference_section_ids or not sentence.text:
            continue
        match = _PRINTED_REFERENCE_PREFIX_RE.match(sentence.text)
        if match is None:
            continue
        marker = next((value for value in match.groups() if value is not None), None)
        if marker is not None and int(marker) in valid_bib_ids:
            sources.setdefault(int(marker), sentence.text)
    return sources


def _numeric_candidates(body_sents, valid_bib_ids: set[int]) -> list[CitationCandidate]:
    candidates: list[CitationCandidate] = []
    for sent in body_sents:
        handled_bracket_spans: set[tuple[int, int]] = set()
        for match in NUMERIC_CITE_RE.finditer(sent.text):
            handled_bracket_spans.add((match.start(), match.end()))
            nums = _expand_numeric_range(match.group(1))
            reasons: list[str] = []
            if not nums:
                reasons.append("empty_numeric_marker")
            elif not _is_likely_citation_bracket(match.group(1), nums, valid_bib_ids):
                reasons.append(
                    "unknown_bib_id"
                    if any(num not in valid_bib_ids for num in nums)
                    else "numeric_interval_guard"
                )
            bib_ids = tuple(num for num in nums if num in valid_bib_ids)
            if not bib_ids and "unknown_bib_id" not in reasons:
                reasons.append("unknown_bib_id")
            candidates.append(
                CitationCandidate(
                    text_id=sent.text_id,
                    start=match.start(),
                    end=match.end(),
                    raw=match.group(0),
                    style="numeric",
                    bib_ids=bib_ids,
                    evidence=("bracket_marker",),
                    confidence=1.0 if not reasons else 0.0,
                    accepted=not reasons,
                    rejection_reasons=tuple(reasons),
                )
            )

        for match in _STRIP_CITE_SUP_RE.finditer(sent.text):
            group_text = match.group(1) or match.group(2) or ""
            nums = _expand_numeric_range(group_text)
            reasons = []
            if _is_citation_superscript(match):
                reasons.append("math_superscript")
            if not nums or any(num not in valid_bib_ids for num in nums):
                reasons.append("unknown_bib_id")
            bib_ids = tuple(num for num in nums if num in valid_bib_ids)
            candidates.append(
                CitationCandidate(
                    text_id=sent.text_id,
                    start=match.start(),
                    end=match.end(),
                    raw=match.group(0),
                    style="numeric",
                    bib_ids=bib_ids,
                    evidence=("superscript_marker",),
                    confidence=1.0 if not reasons else 0.0,
                    accepted=not reasons,
                    rejection_reasons=tuple(dict.fromkeys(reasons)),
                )
            )

        # Retain locally rejected bracket-shaped negatives in the receipt.
        # They are deliberately excluded from LLM fallback, but silently
        # dropping them would make candidate-resolution diagnostics dishonest.
        for match in re.finditer(r"\[([^\]]+)\]", sent.text):
            if (match.start(), match.end()) in handled_bracket_spans:
                continue
            if not _looks_like_non_citation_bracket(match.group(1)):
                continue
            candidates.append(
                CitationCandidate(
                    text_id=sent.text_id,
                    start=match.start(),
                    end=match.end(),
                    raw=match.group(0),
                    style="numeric",
                    bib_ids=(),
                    evidence=("bracket_marker", "non_citation_shape"),
                    confidence=0.0,
                    accepted=False,
                    rejection_reasons=("non_citation_bracket",),
                )
            )
    return candidates


def _fallback_style_reasons(
    local_candidates: list[CitationCandidate],
    printed_numeric_ids: set[int],
) -> tuple[tuple[str, ...], float]:
    eligible_sentences = {
        candidate.text_id for candidate in local_candidates if not candidate.rejection_reasons
    }
    # Keep headroom above the minimum acceptance floor so independently
    # recurring fallback styles can still be compared by evidence strength.
    # A detector with exactly three eligible sentences clears the gate but
    # does not receive the same score as one recurring throughout the paper.
    recurrence_score = min(len(eligible_sentences) / (2 * _MIN_FALLBACK_STYLE_HITS), 1.0)
    printed_floor = 3
    printed_score = min(len(printed_numeric_ids) / printed_floor, 1.0)
    reasons: list[str] = []
    if len(eligible_sentences) < _MIN_FALLBACK_STYLE_HITS:
        reasons.append("insufficient_recurring_marker_evidence")
    if len(printed_numeric_ids) < printed_floor:
        reasons.append("no_printed_numeric_bibliography_evidence")
    return tuple(reasons), recurrence_score * printed_score


def _parenthetical_local_reasons(
    text: str,
    match: re.Match,
    sentence_matches: list[re.Match],
) -> tuple[str, ...]:
    """Reject locally evidenced list, count, and label parentheses.

    Bare ``(N)`` syntax is shared by citations, procedure numbering, sample
    counts, and entity/version labels. These guards use only the candidate's
    sentence; no stronger detector is allowed to veto a genuine independent
    parenthetical citation style elsewhere in the document.
    """

    before = text[: match.start()]
    after = text[match.end() :]
    window = before[-160:]
    reasons: list[str] = []

    sentence_numbers = [_expand_numeric_range(candidate.group(1)) for candidate in sentence_matches]
    flat_sequence = [numbers[0] for numbers in sentence_numbers if len(numbers) == 1]
    is_small_contiguous_sequence = (
        len(flat_sequence) >= 2
        and len(flat_sequence) == len(sentence_matches)
        and flat_sequence == list(range(flat_sequence[0], flat_sequence[0] + len(flat_sequence)))
        and flat_sequence[0] <= 3
    )
    if (
        not before.strip()
        or _PROCEDURAL_LABEL_BEFORE_RE.search(before)
        or is_small_contiguous_sequence
    ):
        reasons.append("list_enumeration")

    if (
        _AGGREGATE_COUNT_CUE_RE.search(window)
        and _AGGREGATE_COUNT_NOUN_RE.search(window)
        and (_COUNT_COPULA_AFTER_RE.match(after) or _COUNT_CONTAINER_BEFORE_RE.search(before))
    ):
        reasons.append("parenthetical_count")

    # ``GLASS63 (207)`` and ``IPCAT 213 (111)`` are entity/version labels,
    # not a second citation grammar. The immediately preceding digit is
    # candidate-local evidence and does not suppress parenthetical citations
    # elsewhere in a mixed-style document.
    if re.search(r"(?:[A-Za-z]{2,}[)\]]?\d+|[A-Z][A-Z0-9-]{2,}\s+\d+)\s*$", before):
        reasons.append("adjacent_numeric_label")

    return tuple(reasons)


def _parenthetical_candidates(
    body_sents,
    valid_bib_ids: set[int],
    equation_tags: frozenset[int],
    printed_numeric_ids: set[int],
) -> tuple[list[CitationCandidate], float]:
    candidates: list[CitationCandidate] = []
    for sent in body_sents:
        sentence_matches = list(PAREN_NUMERIC_CITE_RE.finditer(sent.text))
        for match in sentence_matches:
            nums = _expand_numeric_range(match.group(1))
            reasons = list(_parenthetical_local_reasons(sent.text, match, sentence_matches))
            if not nums:
                reasons.append("empty_numeric_marker")
            if any(num >= 1900 for num in nums):
                reasons.append("year")
            if any(num in equation_tags for num in nums):
                reasons.append("equation_tag")
            if _has_numeric_context(sent.text, match.start(), match.end()):
                reasons.append("math_or_measurement_context")
            if any(num not in valid_bib_ids for num in nums):
                reasons.append("unknown_bib_id")
            candidates.append(
                CitationCandidate(
                    text_id=sent.text_id,
                    start=match.start(),
                    end=match.end(),
                    raw=match.group(0),
                    style="paren-numeric",
                    bib_ids=tuple(num for num in nums if num in valid_bib_ids),
                    evidence=("parenthetical_marker",),
                    confidence=0.75,
                    accepted=False,
                    rejection_reasons=tuple(dict.fromkeys(reasons)),
                )
            )
    style_reasons, score = _fallback_style_reasons(
        candidates,
        printed_numeric_ids,
    )
    resolved = [
        replace(
            candidate,
            accepted=not candidate.rejection_reasons and not style_reasons,
            confidence=score if not candidate.rejection_reasons else 0.0,
            rejection_reasons=(
                candidate.rejection_reasons if candidate.rejection_reasons else style_reasons
            ),
            evidence=candidate.evidence
            + (
                f"recurring_sentences:{len({c.text_id for c in candidates if not c.rejection_reasons})}",
                f"printed_reference_prefixes:{len(printed_numeric_ids)}",
            ),
        )
        for candidate in candidates
    ]
    return resolved, score


def _flattened_candidates(
    body_sents,
    valid_bib_ids: set[int],
    printed_numeric_ids: set[int],
    references,
    reference_sources: dict[int, str],
) -> tuple[list[CitationCandidate], float]:
    references_by_id = {reference.bib_id: reference for reference in references}
    acronym_alias_tokens = _body_acronym_alias_tokens(body_sents)
    candidates: list[CitationCandidate] = []
    for sent in body_sents:
        for match in FLATTENED_SUP_CITE_RE.finditer(sent.text):
            word, digits = match.group(1), match.group(2)
            reasons: list[str] = []
            if sum(char.isalpha() for char in word) < 2:
                reasons.append("single_letter_carrier")
            if _is_non_citation_digit_run(sent.text, digits, match.end()):
                reasons.append("math_or_measurement_context")
            nums = _expand_numeric_range(digits)
            if not nums or any(num not in valid_bib_ids for num in nums):
                reasons.append("unknown_bib_id")
            high_specificity = _flattened_marker_is_high_specificity(word, digits)
            carrier = _flattened_carrier(word)
            grounding_required = (
                not high_specificity
                and len(nums) == 1
                and _flattened_carrier_requires_grounding(carrier)
            )
            target = references_by_id.get(nums[0]) if len(nums) == 1 else None
            grounded = bool(
                grounding_required
                and target is not None
                and _reference_supports_flattened_carrier(
                    target,
                    carrier,
                    source_text=reference_sources.get(nums[0]),
                    acronym_alias_tokens=acronym_alias_tokens,
                )
            )
            # Model/instrument/entity syntax is candidate-local negative
            # evidence even when the target reference happens to repeat the
            # token. Lexical overlap alone cannot turn ``BERT12 model`` or
            # ``COVID19`` into a superscript citation.
            reasons.extend(
                _flattened_local_context_reasons(
                    sent.text,
                    match.start(1),
                    match.end(2),
                )
            )
            if grounding_required and not grounded:
                reasons.append("reference_carrier_mismatch")
            evidence = [
                "word_attached_digit_run",
                (
                    "high_specificity_marker"
                    if high_specificity
                    else "ambiguous_alphanumeric_carrier"
                ),
            ]
            if grounding_required and nums:
                evidence.append(
                    f"reference_carrier_{'grounded' if grounded else 'ungrounded'}:{nums[0]}"
                )
            candidates.append(
                CitationCandidate(
                    text_id=sent.text_id,
                    start=match.start(2),
                    end=match.end(2),
                    raw=digits,
                    style="flattened-superscript",
                    bib_ids=tuple(num for num in nums if num in valid_bib_ids),
                    evidence=tuple(evidence),
                    confidence=0.7,
                    accepted=False,
                    rejection_reasons=tuple(dict.fromkeys(reasons)),
                )
            )
    anchors = [
        candidate
        for candidate in candidates
        if not candidate.rejection_reasons and "high_specificity_marker" in candidate.evidence
    ]
    anchor_sentence_count = len({candidate.text_id for candidate in anchors})
    printed_floor = 3
    recurrence_score = min(
        anchor_sentence_count / (2 * _MIN_FLATTENED_ANCHOR_HITS),
        1.0,
    )
    printed_score = min(len(printed_numeric_ids) / printed_floor, 1.0)
    style_reasons: list[str] = []
    if anchor_sentence_count < _MIN_FLATTENED_ANCHOR_HITS:
        style_reasons.append("insufficient_recurring_marker_evidence")
    if len(printed_numeric_ids) < printed_floor:
        style_reasons.append("no_printed_numeric_bibliography_evidence")
    score = recurrence_score * printed_score
    resolved: list[CitationCandidate] = []
    for candidate in candidates:
        is_ambiguous = "high_specificity_marker" not in candidate.evidence
        has_document_proof = (
            not is_ambiguous or anchor_sentence_count >= _MIN_FLATTENED_AMBIGUOUS_PROOF_HITS
        )
        accepted = not candidate.rejection_reasons and not style_reasons and has_document_proof
        rejection_reasons = list(candidate.rejection_reasons)
        if not accepted:
            rejection_reasons.extend(style_reasons)
            if is_ambiguous:
                rejection_reasons.append("ambiguous_alphanumeric_carrier")
        resolved.append(
            replace(
                candidate,
                accepted=accepted,
                confidence=score if accepted else 0.0,
                rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
                evidence=candidate.evidence
                + (
                    f"recurring_anchor_sentences:{anchor_sentence_count}",
                    f"printed_reference_prefixes:{len(printed_numeric_ids)}",
                ),
            )
        )
    return resolved, score


def _resolve_competing_fallback_styles(
    parenthetical: list[CitationCandidate],
    flattened: list[CitationCandidate],
) -> tuple[list[CitationCandidate], list[CitationCandidate]]:
    """Preserve independently proven fallback styles.

    Relative frequency is useful diagnostic evidence, not a candidate-local
    rejection reason. Mixed source documents can legitimately contain both
    parenthetical and flattened numeric citations, so each detector keeps its
    own accepted/rejected result.
    """

    return parenthetical, flattened


def _dedupe_candidates(candidates: list[CitationCandidate]) -> list[CitationCandidate]:
    """Deduplicate accepted spans while retaining distinct rejected evidence."""

    out: list[CitationCandidate] = []
    accepted_positions: dict[tuple[int, int, int, tuple[int, ...]], int] = {}
    rejected_seen: set[tuple] = set()
    for candidate in candidates:
        if candidate.accepted:
            key = (candidate.text_id, candidate.start, candidate.end, candidate.bib_ids)
            previous = accepted_positions.get(key)
            if previous is None:
                accepted_positions[key] = len(out)
                out.append(candidate)
            elif candidate.confidence > out[previous].confidence:
                out[previous] = candidate
            continue
        key = (
            candidate.text_id,
            candidate.start,
            candidate.end,
            candidate.style,
            candidate.bib_ids,
            candidate.rejection_reasons,
        )
        if key not in rejected_seen:
            rejected_seen.add(key)
            out.append(candidate)
    return out


async def detect_bib_xrefs_with_receipt(
    sentences,
    sections,
    references,
    llm_client=None,
    file_hash="unknown",
) -> tuple[list[PaperXref], CitationLinkingReceipt]:
    """Detect bib xrefs and return the full evidence/rejection receipt."""

    body_sents = _get_body_sentences(sentences, sections)
    valid_bib_ids = {reference.bib_id for reference in references}
    reference_sources = _printed_reference_sources(sentences, sections, references)
    printed_numeric_ids = set(reference_sources)

    numeric = _numeric_candidates(body_sents, valid_bib_ids)
    numeric_sentence_count = len({candidate.text_id for candidate in numeric if candidate.accepted})
    numeric_score = min(numeric_sentence_count / _MIN_FALLBACK_STYLE_HITS, 1.0)
    equation_tags = frozenset(_collect_equation_tags(sentences))
    parenthetical, parenthetical_score = _parenthetical_candidates(
        body_sents,
        valid_bib_ids,
        equation_tags,
        printed_numeric_ids,
    )
    flattened, flattened_score = _flattened_candidates(
        body_sents,
        valid_bib_ids,
        printed_numeric_ids,
        references,
        reference_sources,
    )
    parenthetical, flattened = _resolve_competing_fallback_styles(
        parenthetical,
        flattened,
    )

    from bibr.structure.citation_matcher import match_with_candidates

    _tier2, _tier2_ambiguous, author_year = match_with_candidates(body_sents, references)
    candidates = _dedupe_candidates(numeric + parenthetical + flattened + author_year)

    all_xrefs: list[PaperXref] = []
    seen: set[tuple[int, int]] = set()

    def _add_xrefs(xrefs: list[PaperXref]) -> None:
        for xref in xrefs:
            key = (xref.text_id, xref.xref_id)
            if key not in seen:
                seen.add(key)
                all_xrefs.append(xref)

    _add_xrefs(
        [
            PaperXref(
                xref_id=bib_id,
                xref_type="bib",
                contents=_normalize_citation_text(candidate.raw),
                text_id=candidate.text_id,
                tier=candidate.style,
            )
            for candidate in candidates
            if candidate.accepted
            for bib_id in candidate.bib_ids
        ]
    )

    if llm_client:
        resolved_pairs = {
            (xref.text_id, _normalize_citation_text(xref.contents)) for xref in all_xrefs
        }
        ambiguous: list[tuple[int, str, int, int]] = []
        for candidate in candidates:
            if (
                candidate.style == "author-year"
                and not candidate.accepted
                and not _citation_span_is_resolved(
                    candidate.text_id,
                    candidate.raw,
                    resolved_pairs,
                )
            ):
                ambiguous.append((candidate.text_id, candidate.raw, candidate.start, candidate.end))
        for sent in body_sents:
            for match in re.finditer(r"\[([^\]]+)\]", sent.text):
                match_text = match.group(1)
                if NUMERIC_CITE_RE.fullmatch(f"[{match_text.strip()}]"):
                    continue
                if _looks_like_non_citation_bracket(match_text):
                    continue
                cite_text = match.group(0)
                if not _citation_span_is_resolved(sent.text_id, cite_text, resolved_pairs):
                    ambiguous.append((sent.text_id, cite_text, match.start(), match.end()))
            for pattern in (PAREN_AUTHOR_YEAR_RE, NARRATIVE_AUTHOR_YEAR_RE):
                for match in pattern.finditer(sent.text):
                    cite_text = match.group(0)
                    if not _citation_span_is_resolved(sent.text_id, cite_text, resolved_pairs):
                        ambiguous.append((sent.text_id, cite_text, match.start(), match.end()))

        if ambiguous:
            ambiguous = list(dict.fromkeys(ambiguous))
            ambiguous = [
                occurrence
                for occurrence in ambiguous
                if not any(
                    other[0] == occurrence[0]
                    and other[2] <= occurrence[2]
                    and other[3] >= occurrence[3]
                    and (other[2], other[3]) != (occurrence[2], occurrence[3])
                    for other in ambiguous
                )
            ]
            source_by_id = {sent.text_id: sent.text for sent in body_sents}
            for text_id, cite_text, start, end in ambiguous:
                if any(
                    candidate.text_id == text_id
                    and candidate.start == start
                    and candidate.end == end
                    for candidate in candidates
                ):
                    continue
                source = source_by_id.get(text_id, "")
                candidates.append(
                    CitationCandidate(
                        text_id=text_id,
                        start=start,
                        end=end,
                        raw=source[start:end] if source else cite_text,
                        style="llm",
                        bib_ids=(),
                        evidence=("llm_candidate",),
                        confidence=0.0,
                        accepted=False,
                        rejection_reasons=("unresolved_llm_candidate",),
                    )
                )

            unique_cites: dict[str, int] = {}
            for text_id, cite_text, _start, _end in ambiguous:
                unique_cites.setdefault(_normalize_citation_text(cite_text), text_id)
            tier3 = await _resolve_with_llm(
                [(text_id, cite_text) for cite_text, text_id in unique_cites.items()],
                references,
                llm_client,
                file_hash,
            )
            resolved_map = {_normalize_citation_text(xref.contents): xref.xref_id for xref in tier3}
            resolved_occurrences = [
                (
                    PaperXref(
                        xref_id=bib_id,
                        xref_type="bib",
                        contents=_normalize_citation_text(cite_text),
                        text_id=text_id,
                        tier="llm",
                    ),
                    start,
                    end,
                )
                for text_id, cite_text, start, end in ambiguous
                if (bib_id := resolved_map.get(_normalize_citation_text(cite_text))) is not None
            ]
            llm_xrefs = [xref for xref, _start, _end in resolved_occurrences]
            _add_xrefs(llm_xrefs)

            for xref, start, end in resolved_occurrences:
                normalized_contents = _normalize_citation_text(xref.contents)
                matched_candidate = False
                for index, candidate in enumerate(candidates):
                    if (
                        candidate.text_id == xref.text_id
                        and candidate.start == start
                        and candidate.end == end
                        and not candidate.accepted
                        and _normalize_citation_text(candidate.raw) == normalized_contents
                    ):
                        candidates[index] = replace(
                            candidate,
                            bib_ids=(xref.xref_id,),
                            evidence=candidate.evidence
                            + tuple(
                                f"llm_overrode:{reason}" for reason in candidate.rejection_reasons
                            )
                            + ("llm_resolution",),
                            confidence=max(candidate.confidence, 0.85),
                            accepted=True,
                            rejection_reasons=(),
                        )
                        matched_candidate = True
                if matched_candidate:
                    continue
                source = source_by_id.get(xref.text_id, "")
                candidates.append(
                    CitationCandidate(
                        text_id=xref.text_id,
                        start=start,
                        end=end,
                        raw=source[start:end] if source else xref.contents,
                        style="llm",
                        bib_ids=(xref.xref_id,),
                        evidence=("llm_resolution",),
                        confidence=0.85,
                        accepted=True,
                        rejection_reasons=(),
                    )
                )

            candidates = _dedupe_candidates(candidates)

    accepted_count = sum(1 for candidate in candidates if candidate.accepted)
    resolved_candidate_fraction = accepted_count / len(candidates) if candidates else None
    unique_linked = {
        xref.xref_id
        for xref in all_xrefs
        if xref.xref_type == "bib" and xref.xref_id in valid_bib_ids
    }
    unique_linked_bib_fraction = len(unique_linked) / len(valid_bib_ids) if valid_bib_ids else None
    author_year_candidates = [
        candidate for candidate in candidates if candidate.style == "author-year"
    ]
    author_year_score = (
        sum(1 for candidate in author_year_candidates if candidate.accepted)
        / len(author_year_candidates)
        if author_year_candidates
        else 0.0
    )
    receipt = CitationLinkingReceipt(
        style_scores={
            "numeric": numeric_score,
            "paren-numeric": parenthetical_score,
            "flattened-superscript": flattened_score,
            "author-year": author_year_score,
        },
        candidates=tuple(candidates),
        resolved_candidate_fraction=resolved_candidate_fraction,
        unique_linked_bib_fraction=unique_linked_bib_fraction,
    )
    logger.info("Bib xref detection complete: %d total xrefs", len(all_xrefs))
    return all_xrefs, receipt


async def detect_bib_xrefs(
    sentences,
    sections,
    references,
    llm_client=None,
    file_hash="unknown",
    *,
    receipt_sink: list[CitationLinkingReceipt] | None = None,
) -> list[PaperXref]:
    """Compatibility wrapper preserving the public list-return contract."""

    xrefs, receipt = await detect_bib_xrefs_with_receipt(
        sentences,
        sections,
        references,
        llm_client=llm_client,
        file_hash=file_hash,
    )
    if receipt_sink is not None:
        receipt_sink.append(receipt)
    return xrefs
