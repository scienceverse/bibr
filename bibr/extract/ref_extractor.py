"""Reference segmentation and parsing.

``ReferenceExtractor`` turns a located references DataFrame into
``PaperReference`` objects: segmentation (geom GBM → region anchors → LLM
anchor-emit → CRF cascade, plus marker-split last resort) followed by parsing
(batched LLM or local ModernBERT-CRF NER, dispatched through
``REF_PARSE_STRATEGIES``).
Module-level helpers cover the strategy-independent finalize steps (DOI/author
rescue, vol/issue normalization, bib_type inference, bib_id sequencing).

Locating the reference rows lives in ``bibr.extract.ref_locator``; training
capture in ``bibr.extract.training_capture``.
"""

import asyncio
import dataclasses
import logging
import re
import threading
import unicodedata
from typing import Any
from urllib.parse import unquote

import pandas as pd
from pydantic import ValidationError

from bibr.clients.llm import LLMClient, incomplete_output_text
from bibr.clients.llm_protocol import LlmClient
from bibr.config import GlobalSettings, snapshot_settings
from bibr.exceptions import ProcessingError, UpstreamServiceError
from bibr.extract.anchor_snap import find_anchor_starts, segment_by_anchors
from bibr.extract.merge_split import _onset_finder_for_bibliography, split_merged_refs
from bibr.extract.ref_line_stream import (
    StreamSegmentation,
    _match_key,
    _roman_value,
    build_line_stream,
    link_dois_for_segments,
    segment_line_stream,
    segmentation_quality,
    stream_probabilities,
    typical_entry_length,
)
from bibr.extract.ref_locator import _ENTRY_NUMBERING_RE
from bibr.extract.region_seg import (
    _CHUNK_TARGET_CHARS,
    region_anchor_texts,
    region_chunks,
    segment_by_region_anchors,
)
from bibr.extract.segment_filter import drop_non_reference_segments, is_non_reference_segment
from bibr.extract.training_capture import save_ref_training_data, save_seg_training_data
from bibr.paper import BibType, PaperReference, migrate_bib_type
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    ReferenceSegmentationAttempt,
    ReferenceYieldReceipt,
)
from bibr.processing_warnings import ProcessingWarning, WarningCode
from bibr.schemas import PaperReferenceLLM
from bibr.utils.json_salvage import salvage_array_objects
from bibr.utils.locks import LOCAL_INFERENCE_LOCK
from bibr.utils.text import collapse_ws, normalize_doi
from bibr.utils.transient import is_transient_network_error
from bibr.validation import IssueSeverity, ValidationIssue

try:
    # Raised by instructor when the model hits its output-token cap mid-object.
    from instructor.core.exceptions import IncompleteOutputException
except ImportError:  # pragma: no cover — instructor is always present on LLM paths

    class IncompleteOutputException(Exception):  # type: ignore[no-redef]
        """Fallback stand-in so isinstance() checks degrade to False if
        instructor's exception module path changes."""


logger = logging.getLogger(__name__)

# Every fallback, recovery and loss in reference segmentation and parsing is
# recorded as a coded warning (``_record_warning``) that reaches the exported
# ``extraction.warnings``, not just a log line, so eval tooling can count each
# tier's events across a corpus. ``--refs llm`` promises full-precision LLM
# parsing, so a surviving fallback to the NER parser must be visible; and a
# populated references region that ends with 0 references (``REF_SEG_FAILED``)
# or references lost to a failed parse (``REF_PARSE_LOST``) must never look like
# a paper that printed fewer.

# Geom segment-count sanity gate: reject a confident geom result whose span
# count is below this fraction of the independent layout reference-onset count.
# A GBM that confidently under-segments an OOD reference style (arXiv/ACL
# bare-year) clears the confidence and align-yield gates, so this ratio is the
# backstop. Module constant on purpose — not a tunable in config.py.
GEOM_MIN_REGION_RATIO = 0.6
# When nearly every filtered layout onset aligns back to the exact reference
# text, the layout denominator is much stronger than a raw label count.  In
# that case, reject a geom result that loses more than 10% of those starts.
GEOM_STRONG_REGION_RATIO = 0.9
GEOM_STRONG_REGION_ALIGN_FRACTION = 0.9

# A references region shorter than this is too small for "0 segments" to count
# as a hard failure (avoids crying wolf on tiny/empty regions).
_REF_REGION_NONEMPTY_MIN_CHARS = 200

# Line-start reference numbering markers ("[1] ", "12. ", "3) ") for the
# zero-LLM last-resort splitter used when the whole cascade yields nothing.
_MARKER_LINE_RE = re.compile(r"(?m)^\s*(?:\[\d{1,3}\]|\d{1,3}[.)])\s+")

# Reference line stream selection (``bibr.extract.ref_line_stream``). Its
# segmentation replaces the cascade's when the cascade fell back to region
# recovery, the CRF or the marker split, found nothing, took most of its
# segments from the merged-reference splitter, or under-yielded against its
# credible starts, provided the stream's quality is at least the cascade's;
# it replaces a selected geom or LLM-anchor result only when its quality is
# higher by the override margin. It never replaces a non-empty result with
# fewer entries than that result has distinct ones, nor on a paper with a
# rotated reference page. Quality is ``segmentation_quality``: section
# coverage times the share of entries that look like one complete reference,
# scored alike for both. Module constants on purpose, like the geom gates
# above.
_STREAM_FALLBACK_TIERS = frozenset({"region", "crf", "marker_split"})
_STREAM_OVERRIDE_MARGIN = 0.15
# Pre-parse form of the receipt's credible-start check: segments below this
# share of the section's credible entry starts.
_STREAM_CREDIBLE_START_YIELD = 0.6
_STREAM_MIN_EVIDENCE = 5
# The cascade's segments outnumber its selected tier's spans by this factor
# when the merged-reference splitter made most of them.
_STREAM_SPLIT_REPAIR_RATIO = 1.5


def _build_ref_text(row_texts: list[str]) -> str:
    """Join reference rows into segmenter input, collapsing intra-row whitespace.

    OCR of wide-letter-spaced reference sections emits pathological inter-token whitespace — tabs, carriage
    returns, and non-breaking spaces between every word
    (``"1.\\t\\r \\xa0Example,\\t\\r \\xa0B."``). The whitespace-normalizing
    cleanup (``PaperContents.finalize_text``) runs only AFTER extraction, so the
    segmenter would otherwise receive the raw spacing. Anchor snapping then
    fails its exact match and the rapidfuzz fallback mis-locates the boundary,
    dropping or mangling the leading reference (Example → ``ample``).

    Collapsing each row to single spaces and joining with newlines yields clean
    anchor-snappable text while preserving the one-reference-per-line structure
    that boundary detection relies on. A no-op on already-clean rows.
    """
    cleaned = (collapse_ws(t) for t in row_texts)
    return "\n".join(t for t in cleaned if t)


def _marker_split_refs(ref_text: str) -> list[str]:
    """Zero-LLM last-resort split of a numbered references block.

    Splits on a run of ≥3 line-start numbering markers; declines otherwise so a
    stray "1." in prose never fabricates segments. Recovers e.g. a Vancouver
    "[1]…[118]" list when geom, LLM, and CRF all failed (the LIGO 0-ref case).
    """
    starts = [m.start() for m in _MARKER_LINE_RE.finditer(ref_text)]
    if len(starts) < 3:
        return []
    spans: list[str] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(ref_text)
        seg = ref_text[start:end].strip()
        if seg:
            spans.append(seg)
    return spans


def _strip_enum_markers(ref_strings: list[str]) -> list[str]:
    """Strip printed list-numbering ("12. ", "[3] ") for NER parser input.

    The v4 GIANT checkpoint never learned bare dot-markers as O — it absorbs
    them into the adjacent field span ("1." → authors, "42." → title). Gated
    at the bibliography level: a numbered style numbers every entry, so only
    a marker on the majority of segments triggers stripping; a lone
    number-leading segment in an unnumbered bibliography stays verbatim.
    The LLM parse path keeps the markers — its prompt is built around a
    numbered-list contract (see ``ReferenceExtractor._parse_references_llm``).
    """
    marked = sum(1 for s in ref_strings if _ENTRY_NUMBERING_RE.match(s))
    if marked * 2 <= len(ref_strings):
        return _strip_roman_markers(ref_strings)
    return [_ENTRY_NUMBERING_RE.sub("", s, count=1) for s in ref_strings]


# A roman list number opening a reference ("IV. Bergasa, L.M., ...").
_ROMAN_ENTRY_RE = re.compile(r"^\s*([IVXLC]{1,7}|[ivxlc]{1,7})[.)]\s+")


def _strip_roman_markers(ref_strings: list[str]) -> list[str]:
    """Strip roman list numbering ("I.", "II.", ...) for NER parser input.

    Only for a bibliography numbered that way: a majority of segments open
    with a roman numeral and those numerals count up from I, so a lone author
    initial ("V. Lal") in an unnumbered list stays.
    """
    values = []
    for segment in ref_strings:
        match = _ROMAN_ENTRY_RE.match(segment)
        values.append(_roman_value(match.group(1)) if match else None)
    numbered = [value for value in values if value is not None]
    if len(numbered) * 2 <= len(ref_strings) or 1 not in numbered:
        return ref_strings
    steps = sum(1 for a, b in zip(numbered, numbered[1:], strict=False) if b == a + 1)
    if steps * 5 < (len(numbered) - 1) * 4:
        return ref_strings
    return [
        _ROMAN_ENTRY_RE.sub("", segment, count=1) if value is not None else segment
        for segment, value in zip(ref_strings, values, strict=True)
    ]


def _normalized_reference_identity(text: str) -> str:
    """Conservative exact-identity form; never used for title/fuzzy matching."""
    normalized = collapse_ws(unicodedata.normalize("NFKC", text)).strip()
    return _ENTRY_NUMBERING_RE.sub("", normalized, count=1).casefold()


def _locate_segment_spans(ref_text: str, segments: list[str]) -> tuple[tuple[int, int] | None, ...]:
    """Recover each exact source span independently, preserving partial evidence."""
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    for segment in segments:
        start = ref_text.find(segment, cursor)
        if start < 0:
            # A duplicated tier can replay a span already passed. Preserve that
            # provenance when it is exactly locatable; otherwise abstain.
            start = ref_text.find(segment)
        if start < 0:
            spans.append(None)
            continue
        end = start + len(segment)
        spans.append((start, end))
        cursor = max(cursor, end)
    return tuple(spans)


def _segments_to_spans(ref_text: str, segments: list[str]) -> tuple[tuple[int, int], ...]:
    """Return every exact source offset available for *segments*, in order."""
    return tuple(span for span in _locate_segment_spans(ref_text, segments) if span is not None)


def _prepare_segment_candidates(
    ref_text: str,
    segments: list[str],
    located_spans: tuple[tuple[int, int] | None, ...],
) -> tuple[list[str], tuple[tuple[int, int], ...], float, tuple[str, ...]]:
    """Keep unlocated segments, but deduplicate proven identical source spans."""
    if len(segments) != len(located_spans):
        raise ValueError("segments and located_spans must stay positionally aligned")
    kept_segments: list[str] = []
    kept_spans: list[tuple[int, int]] = []
    reasons: set[str] = set()
    seen_spans: set[tuple[tuple[int, int], str]] = set()
    duplicate_count = 0
    located_count = 0
    for segment, span in zip(segments, located_spans, strict=True):
        if span is None:
            kept_segments.append(segment)
            continue
        start, end = span
        if start < 0 or end <= start or end > len(ref_text):
            kept_segments.append(segment)
            continue
        located_count += 1
        identity = _normalized_reference_identity(ref_text[start:end])
        key = (span, identity)
        if key in seen_spans:
            duplicate_count += 1
            reasons.add("duplicate_source_span")
            continue
        seen_spans.add(key)
        kept_segments.append(segment)
        kept_spans.append(span)

    if located_count == 0 and segments:
        reasons.add("source_offsets_unavailable")
    elif located_count < len(segments):
        reasons.add("source_offsets_partially_unavailable")
    duplicate_rate = duplicate_count / len(segments) if segments else 0.0
    return kept_segments, tuple(kept_spans), duplicate_rate, tuple(sorted(reasons))


def _prepare_segmented_references(
    ref_text: str, spans: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[list[str], tuple[tuple[int, int], ...], float, tuple[str, ...]]:
    """Deduplicate only exact source spans.

    A repeated singleton at another offset remains: exact surface equality is
    not enough to establish duplication. The same holds for a whole repeated
    run at distinct offsets. A replay is suppressed only when a segmentation
    path selects the same physical source spans again (the ord227 evidence).
    """
    segments: list[str] = []
    valid_spans: list[tuple[int, int]] = []
    for start, end in spans:
        if start < 0 or end <= start or end > len(ref_text):
            continue
        segments.append(ref_text[start:end])
        valid_spans.append((start, end))
    return _prepare_segment_candidates(ref_text, segments, tuple(valid_spans))


def _source_character_coverage(ref_text: str, spans: tuple[tuple[int, int], ...]) -> float | None:
    if not ref_text:
        return None
    source_chars = sum(not char.isspace() for char in ref_text)
    if not source_chars:
        return None
    if not spans:
        return 0.0
    covered: set[int] = set()
    for start, end in spans:
        covered.update(i for i in range(start, end) if not ref_text[i].isspace())
    return min(1.0, len(covered) / source_chars)


# A printed DOI marker followed by its identifier token. Used to deterministically
# recover a DOI the batched LLM parser under-emitted (residual #4) from the
# reference's OWN segment. The captured ``\S+`` stops at the first whitespace, so
# an OCR-damaged DOI (space inside the identifier, e.g. "10.1126/ science...")
# yields an invalid token that ``normalize_doi`` rejects. Line-wrapped DOIs are
# recovered by ``_rescue_wrapped_doi`` below, but only after this strict form
# has already failed.
_DOI_RESCUE_MARKER_RE = re.compile(
    r"(?:doi:|doi\.org/|https?://(?:dx\.)?doi\.org/)\s*(\S+)",
    re.IGNORECASE,
)

# Characters a printed DOI is built from. Anything else terminates the scan.
_DOI_BODY_CHAR_RE = re.compile(r"[-._;()/:A-Za-z0-9]")
# A DOI broken over two printed lines yields at most one gap; three tolerates a
# narrow column plus an OCR word-split without letting the scan run into the
# rest of the entry.
_DOI_WRAP_MAX_GAPS = 3


def _rescue_wrapped_doi(tail: str) -> str | None:
    """Rejoin a DOI broken across an OCR line wrap.

    Consulted only after the whitespace-free token fails normalize_doi, this rescue cannot change an already successful normalization; it can only recover a previously declined token.

    A wrap continues an identifier mid-token, so a gap is bridged only when the
    next character could continue the DOI (a digit or a lowercase letter). A
    capital, bracket, or punctuation starts a new field — author initials, a
    year, "In:" — and ends the scan.
    """

    pieces: list[str] = []
    gaps = 0
    index = 0
    length = len(tail)
    while index < length:
        char = tail[index]
        if _DOI_BODY_CHAR_RE.fullmatch(char):
            pieces.append(char)
            index += 1
            continue
        if not char.isspace():
            break
        gap_end = index
        while gap_end < length and tail[gap_end].isspace():
            gap_end += 1
        if gaps >= _DOI_WRAP_MAX_GAPS or gap_end >= length:
            break
        following = tail[gap_end]
        if not (following.isdigit() or (following.isalpha() and following.islower())):
            break
        gaps += 1
        index = gap_end
    if not gaps:
        return None
    return _clean_url_doi_token("".join(pieces))


# "Available online:" entries (MDPI/DG house style) print the DOI only inside a
# publisher URL path — degruyter.com/document/doi/<doi>/pdf,
# tandfonline.com/doi/full/<doi>, legacy Wiley /doi/<doi>/abstract. Requires an
# explicit /doi/ path marker; a bare 10.x token in an arbitrary URL (JSTOR
# stable ids) is never harvested.
_DOI_RESCUE_URL_PATH_RE = re.compile(
    r"/doi/(?:(?:abs|full|pdf|epdf|epub|html|document)/)?(10\.\d{4,9}/\S+)",
    re.IGNORECASE,
)

# Trailing render-format path segments after the DOI in publisher URLs.
_DOI_URL_PATH_SUFFIX_RE = re.compile(
    r"(?:/(?:abstract|full|pdf|epdf|epub|html|summary))+$",
    re.IGNORECASE,
)


def _clean_url_doi_token(token: str) -> str | None:
    """Strip URL fragment/query, render suffixes, and %-encoding from *token*."""
    token = token.split("#")[0].split("?")[0]
    token = _DOI_URL_PATH_SUFFIX_RE.sub("", token)
    return normalize_doi(unquote(token))


def _rescue_doi_from_segment(segment: str | None) -> str | None:
    """Recover a DOI from a clean printed token in *segment*.

    Fires on a ``doi:``/doi.org marker or an explicit ``/doi/`` publisher URL
    path, for a syntactically clean identifier (validated through
    :func:`normalize_doi`), falling back to :func:`_rescue_wrapped_doi` when the
    printed DOI is broken across a line; never invents or repairs a damaged DOI
    otherwise. Intended as a last-resort fill when the LLM returned ``doi=None``
    despite a DOI being printed — scoped to the single reference's own text so
    it cannot borrow a neighbour's DOI.
    """
    if not segment:
        return None
    m = _DOI_RESCUE_MARKER_RE.search(segment)
    if m:
        doi = _clean_url_doi_token(m.group(1))
        if doi:
            return doi
        wrapped = _rescue_wrapped_doi(segment[m.start(1) :])
        if wrapped:
            return wrapped
    m = _DOI_RESCUE_URL_PATH_RE.search(segment)
    if m:
        return _clean_url_doi_token(m.group(1))
    return None


# An author-date publication year in parentheses ("(2014)", "(2012a)",
# "(2014, March)"). It marks the end of the leading author block, so the text
# before it is the printed byline. Vancouver / numbered styles (bare year at the
# end, "(3)" issue markers) never match, so the author rescue declines on them
# rather than over-capturing the title/container.
_AUTHOR_YEAR_CUT_RE = re.compile(r"\(\s*(?:18|19|20)\d{2}[a-z]?\b")

# Guard: a rescued byline longer than this almost certainly captured a title or
# sentence rather than an author list.
_MAX_RESCUED_AUTHORS_LEN = 300


def _rescue_authors_from_segment(segment: str | None) -> str | None:
    """Recover the author byline from the head of a reference's own segment.

    Scoped to author-date references: cuts at the first parenthesized
    publication year and returns the verbatim text before it, normalized the
    same way an LLM-parsed author is (trailing ``.,`` stripped) so a rescued
    byline is indistinguishable from a parsed one. Declines — returns
    ``None`` — when there is no leading ``(YEAR)`` (Vancouver/numbered styles),
    when the prefix is empty (no printed byline), when it is implausibly long
    (a captured title), or when it carries no letters. Never fabricates; only
    consulted when the LLM emitted ``authors=None``.
    """
    if not segment:
        return None
    m = _AUTHOR_YEAR_CUT_RE.search(segment)
    if not m or m.start() == 0:
        return None
    candidate = segment[: m.start()].strip().rstrip(".,").strip()
    if not candidate or len(candidate) > _MAX_RESCUED_AUTHORS_LEN:
        return None
    if not any(c.isalpha() for c in candidate):
        return None
    return candidate


_NER_SEGMENTER = None
_NER_PARSER = None
_NER_SEGMENTER_KEY: tuple[str, str, str] | None = None
_NER_PARSER_KEY: tuple[str, str, str] | None = None
_NER_LOCK = threading.Lock()


def _get_ner_segmenter(settings: GlobalSettings | None = None):
    """Lazy-load the CRF segmenter singleton (no parser)."""
    global _NER_SEGMENTER, _NER_SEGMENTER_KEY
    settings = settings if settings is not None else snapshot_settings()
    key = (settings.NER_SEG_CKPT, settings.NER_DEVICE, settings.NER_SEG_REVISION)
    if _NER_SEGMENTER is None or key != _NER_SEGMENTER_KEY:
        with _NER_LOCK:
            if _NER_SEGMENTER is None or key != _NER_SEGMENTER_KEY:
                from bibr.ner.segmenter import RefSegmenter

                logger.info("loading NER segmenter: %s", settings.NER_SEG_CKPT)
                _NER_SEGMENTER = RefSegmenter(
                    settings.NER_SEG_CKPT,
                    device=settings.NER_DEVICE,
                    revision=settings.NER_SEG_REVISION,
                )
                _NER_SEGMENTER_KEY = key
    return _NER_SEGMENTER


_GEOM_SEGMENTER = None
_GEOM_LOAD_FAILED = False
_GEOM_SEGMENTER_KEY: tuple[str, str] | None = None
_GEOM_LOAD_FAILED_KEY: tuple[str, str] | None = None


def _get_geom_segmenter(settings: GlobalSettings | None = None):
    """Lazy-load the geometry segmenter singleton.

    Returns ``None`` (failure cached) when the artifact or its deps (the ``ml``
    extra) are unavailable, so callers cascade to the LLM once rather than
    retrying the load on every paper — geom is the default segmenter, so this
    path is hit by every core install without the ``ml`` extra.
    """
    global _GEOM_LOAD_FAILED, _GEOM_LOAD_FAILED_KEY, _GEOM_SEGMENTER, _GEOM_SEGMENTER_KEY
    settings = settings if settings is not None else snapshot_settings()
    key = (settings.REF_GEOM_SEG_MODEL_ID, settings.REF_GEOM_SEG_REVISION)
    failed_for_key = _GEOM_LOAD_FAILED and key == _GEOM_LOAD_FAILED_KEY
    if (_GEOM_SEGMENTER is None or key != _GEOM_SEGMENTER_KEY) and not failed_for_key:
        with _NER_LOCK:
            if (_GEOM_SEGMENTER is None or key != _GEOM_SEGMENTER_KEY) and not (
                _GEOM_LOAD_FAILED and key == _GEOM_LOAD_FAILED_KEY
            ):
                from bibr.extract.geom_segmenter import GeomSegmenter

                logger.info("loading geom segmenter: %s", settings.REF_GEOM_SEG_MODEL_ID)
                try:
                    _GEOM_SEGMENTER = GeomSegmenter(
                        settings.REF_GEOM_SEG_MODEL_ID,
                        revision=settings.REF_GEOM_SEG_REVISION,
                    )
                    _GEOM_SEGMENTER_KEY = key
                except Exception as e:  # noqa: BLE001 — unavailable deps/artifact → cascade
                    # Only a *definitive* failure is cached. A missing extra or
                    # absent artifact will fail identically forever, so the
                    # sticky flag saves repeated load attempts. A network-shaped
                    # failure will not: caching it disabled the default
                    # segmentation strategy for the whole process lifetime over
                    # one blip, silently cascading every later paper to the LLM
                    # tier.
                    if not is_transient_network_error(e):
                        _GEOM_LOAD_FAILED = True
                        _GEOM_LOAD_FAILED_KEY = key
                    logger.warning(
                        "geom segmenter unavailable (%r); cascading to LLM segmentation", e
                    )
    return _GEOM_SEGMENTER


def _get_ner_parser(settings: GlobalSettings | None = None):
    """Lazy-load the CRF parser singleton (no segmenter)."""
    global _NER_PARSER, _NER_PARSER_KEY
    settings = settings if settings is not None else snapshot_settings()
    key = (settings.NER_PARSER_CKPT, settings.NER_DEVICE, settings.NER_PARSER_REVISION)
    if _NER_PARSER is None or key != _NER_PARSER_KEY:
        with _NER_LOCK:
            if _NER_PARSER is None or key != _NER_PARSER_KEY:
                from bibr.ner.runtime import load_ref_parser

                logger.info("loading NER parser: %s", settings.NER_PARSER_CKPT)
                _NER_PARSER = load_ref_parser(
                    settings.NER_PARSER_CKPT,
                    device=settings.NER_DEVICE,
                    revision=settings.NER_PARSER_REVISION,
                    settings=settings,
                )
                _NER_PARSER_KEY = key
    return _NER_PARSER


def _page_range(ref_df: pd.DataFrame) -> tuple[int, int] | None:
    """First and last page of the located reference rows, or None without pages."""
    if "page_number" not in ref_df.columns:
        return None
    pages = pd.to_numeric(ref_df["page_number"], errors="coerce").dropna()
    if pages.empty:
        return None
    return int(pages.min()), int(pages.max())


def _chunk(items: list, n: int):
    """Yield successive ``n``-sized chunks of *items*."""
    for i in range(0, len(items), max(1, n)):
        yield items[i : i + n]


def _is_degenerate_ref_failure(exc: BaseException) -> bool:
    """True when a ref-parse batch failed *degenerately* — the model timed out
    or ran to its output-token cap (``IncompleteOutputException``).

    Both are signatures of greedy decoding looping on the batch input (e.g. a
    repetitive APA-ellipsis author list), so the failure is deterministic and
    re-asking the identical prompt is futile — the caller should fall straight
    back to the NER parser instead of burning a retry (and the pipeline budget)
    on a re-loop. ``extract_references`` wraps the original error in
    ``UpstreamServiceError``, so unwrap ``original_error``/``__cause__`` layers
    before classifying.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (TimeoutError, IncompleteOutputException)):
            return True
        # The NuExtract native backend reports its own output-cap truncation as
        # a ``NuExtractInvalidOutput`` with category "truncated" rather than an
        # Instructor exception. Same deterministic failure, so it earns the same
        # treatment: skip the futile identical re-ask and salvage / split / fall
        # back to NER. Duck-typed to keep the native client's heavy import out
        # of this module.
        if getattr(cur, "category", None) == "truncated":
            return True
        nxt = getattr(cur, "original_error", None)
        cur = nxt if isinstance(nxt, BaseException) else cur.__cause__
    return False


def _salvage_truncated_batch(exc: BaseException, offset: int) -> list[PaperReferenceLLM]:
    """Recover the complete leading references from a truncated batch completion.

    When a batch fails degenerately by running to its output-token cap, its raw
    completion usually holds many complete leading JSON objects before the cut.
    This validates those against ``PaperReferenceLLM`` and returns the maximal
    prefix that is *positionally aligned* with the input batch — object ``j`` is
    kept only while its model-emitted ``index`` equals ``offset + j`` (the same
    contiguous-numbering contract :meth:`LLMClient.extract_references` trusts).
    The walk stops at the first object whose index breaks that run or fails
    validation, so the caller can treat the remaining ``batch[k:]`` as un-parsed
    and route only it to retry / NER fallback. Returns ``[]`` (no salvage) when
    the raw text is unavailable or the first object is already misaligned, in
    which case the caller's existing degenerate handling is unchanged.
    """
    raw = incomplete_output_text(exc)
    if not raw:
        return []
    salvaged: list[PaperReferenceLLM] = []
    for j, obj in enumerate(salvage_array_objects(raw, "references")):
        if not isinstance(obj, dict) or obj.get("index") != offset + j:
            break
        try:
            salvaged.append(PaperReferenceLLM.model_validate(obj))
        except ValidationError:
            break
    return salvaged


def _resolve_ref_strategies(
    seg_override: str | None = None,
    parse_override: str | None = None,
    *,
    settings: GlobalSettings | None = None,
) -> tuple[str, str]:
    """Resolve effective ``(seg_strategy, parse_strategy)``.

    Per-run overrides (carried on ``RunConfig`` from ``chew(refs=...)`` / CLI
    ``--refs``/``--ref-seg``) win over every Settings knob, so callers never
    need to mutate the process-global ``Settings``.

    Below the overrides sit the decoupled Settings knobs
    (``REF_SEG_STRATEGY`` / ``REF_PARSE_STRATEGY``); an explicit ``None``
    means the built-in default — ``("geom", "ner")``, the local geometry
    segmenter + local ModernBERT-CRF parser. Set parse=llm (CLI ``--refs
    llm``) for full reference precision.

    Segmentation defaults to the local geometry GBM (``geom``), which cascades
    region anchors → LLM → CRF when geometry is absent or unconfident.
    ``REF_SEG_STRATEGY=region`` makes region anchors the primary tier (still
    falling back to LLM → CRF); ``REF_SEG_LLM_FALLBACK=false`` removes the LLM
    tier from the cascade entirely (region → CRF only). Force the LLM
    segmenter with ``REF_SEG_STRATEGY=llm``; the CRF segmenter is opt-in via
    ``=crf``.
    """
    effective = settings if settings is not None else snapshot_settings()
    seg = seg_override if seg_override is not None else effective.REF_SEG_STRATEGY
    parse = parse_override if parse_override is not None else effective.REF_PARSE_STRATEGY
    if seg is None:
        seg = "geom"
    if parse is None:
        parse = "ner"
    return seg.lower(), parse.lower()


_IN_PRESS_RE = re.compile(
    r"in\s*press|forthcoming|advance\s*online|manuscript\s*submitted|epub\s*ahead",
    re.IGNORECASE,
)


def _is_in_press(year_str: str | None) -> bool:
    """True if the year text looks like 'in press' / 'forthcoming' / etc."""
    return bool(year_str and _IN_PRESS_RE.search(year_str))


def _parse_year(year_str: str | None) -> int | None:
    """Parse a year string to int, returning None on failure or 'in press'."""
    if not year_str:
        return None
    if _IN_PRESS_RE.search(year_str):
        return None
    # Strip letter suffixes like "2020a"
    cleaned = re.sub(r"[a-zA-Z]+$", "", year_str.strip())
    try:
        return int(cleaned)
    except (ValueError, TypeError):
        return None


def _expand_compact_last_page(first_page: str | None, last_page: str | None) -> str | None:
    """Expand a bibliographic compact page range ("782-92" → last_page "792").

    MDPI (and Vancouver-style) references drop the shared prefix from the end
    page; the LLM parses the printed digits verbatim. Prefix-fill from
    ``first_page`` when ``last_page`` is numeric, shorter, and smaller — and
    only when the expanded value actually moves the range forward.
    """
    if not first_page or not last_page:
        return last_page
    fp, lp = str(first_page).strip(), str(last_page).strip()
    if not (fp.isdigit() and lp.isdigit()):
        return last_page
    if len(lp) >= len(fp) or int(lp) >= int(fp):
        return last_page
    expanded = fp[: len(fp) - len(lp)] + lp
    return expanded if int(expanded) > int(fp) else last_page


# Combined "N(M)" volume/issue form, e.g. "21(4)", "55(7)", "12(4-5)".  The
# inner group allows digit ranges and alphanumerics (e.g. "S1") but the outer
# volume must be purely numeric so we never mis-split a year or page span.
_VOL_ISSUE_RE = re.compile(r"^(\d+)\s*\(\s*([0-9A-Za-z/\-–—]+)\s*\)$")
# Same form with the closing paren cut off — the parser regularly ends the span
# mid-issue on Vancouver tails ("2021;236(6" from the printed "236(6):…").
_VOL_ISSUE_TRUNC_RE = re.compile(r"^(\d+)\s*\(\s*([0-9A-Za-z/\-–—]+)$")
# Vancouver "<YEAR>;" delimiter swallowed into the start of a volume/issue span.
_YEAR_PREFIX_RE = re.compile(r"^\s*(?:19|20)\d{2}\s*[;:]\s*")
# Bib-punctuation noise to trim from short numeric spans (volume/issue/pages).
# Parentheses are excluded here so the "N(M)" split can run first.
_BIB_FIELD_TRIM = " \t,;:."


def _clean_bib_field(value: str | None, *, strip_parens: bool = True) -> str | None:
    """Trim trailing/leading bib-punctuation noise from a short field span.

    Also drops a leading Vancouver ``<YEAR>;`` the parser swallowed into the
    span ("2021;236(6" → "236(6"); a bare year-like volume has no delimiter
    after it and is left alone.
    """
    if not isinstance(value, str):
        return value
    chars = _BIB_FIELD_TRIM + "()[]" if strip_parens else _BIB_FIELD_TRIM
    return _YEAR_PREFIX_RE.sub("", value).strip(chars) or None


def _split_vol_issue(volume: str | None, issue: str | None) -> tuple[str | None, str | None]:
    """Normalize the (volume, issue) pair from a NER parse.

    Strips trailing punctuation the parser left attached ("25," → "25") and
    splits a combined ``N(M)`` that landed whole in either span ("21(4)," →
    volume "21", issue "4"). Pure string logic; no segment lookup.
    """
    # First test for a combined form, trimming only outer separators (NOT the
    # parens) so the "N(M)" pattern is still intact.
    candidates = (
        _clean_bib_field(volume, strip_parens=False),
        _clean_bib_field(issue, strip_parens=False),
    )
    for pattern in (_VOL_ISSUE_RE, _VOL_ISSUE_TRUNC_RE):
        for cand in candidates:
            if cand:
                m = pattern.match(cand)
                if m:
                    return m.group(1), (m.group(2).strip() or None)
    # No combined form: just clean each field (parens included now).
    return _clean_bib_field(volume), _clean_bib_field(issue)


def _backfill_issue(issue: str | None, volume: str | None, segment: str) -> str | None:
    """Recover an issue number the LLM dropped from the printed "vol(issue)".

    The reference segment retains the printed form ("Psychophysiology, 55(7),")
    even when the parsed ``issue`` came back null. Only fills a confirmed
    ``{volume}(<digits>)`` match anchored on the known volume — supplements,
    bare volumes, and parenthesized years never match.
    """
    if issue is not None or not volume or not segment:
        return issue
    vol = re.escape(str(volume).strip())
    m = re.search(rf"\b{vol}\s*\(\s*(\d+(?:\s*[–—-]\s*\d+)?)\s*\)", segment)
    return m.group(1).strip() if m else None


def _normalize_vol_issue(
    volume: str | None, issue: str | None, segment: str
) -> tuple[str | None, str | None]:
    """Full (volume, issue) normalization for the NER parse path: clean +
    split a combined ``N(M)`` first, then segment-anchored issue backfill."""
    vol, iss = _split_vol_issue(volume, issue)
    iss = _backfill_issue(iss, vol, segment)
    return vol, iss


def _infer_bibtype(
    container: str | None,
    isbn: str | None,
    bib_text: str,
    crossref_type: str | None = None,
) -> str | None:
    """Infer reference type from available fields.

    Returns a ``BibType`` value string (v6.0 taxonomy).
    Uses ``migrate_bib_type`` for Crossref type mapping.
    """
    if crossref_type:
        return migrate_bib_type(crossref_type)

    if container:
        return BibType.JOURNAL_ARTICLE.value
    if isbn:
        return BibType.BOOK.value
    text_lower = bib_text.lower()
    if any(kw in text_lower for kw in ("proceedings", "conference", "workshop", "symposium")):
        return BibType.CONFERENCE_PAPER.value
    if any(kw in text_lower for kw in ("thesis", "dissertation")):
        return BibType.THESIS.value
    if any(kw in text_lower for kw in ("preprint", "arxiv", "biorxiv", "medrxiv", "ssrn")):
        return BibType.PREPRINT.value
    if any(kw in text_lower for kw in ("technical report", "working paper", "tech. rep.")):
        return BibType.REPORT.value
    if "chapter" in text_lower or (
        "in:" in text_lower and not any(kw in text_lower for kw in ("proceedings", "conference"))
    ):
        return BibType.BOOK_CHAPTER.value
    return None


# Vancouver/medical journal tail: "<Container> <YEAR>;<VOL>[(<ISS>)][:<PAGES>].".
# The year sits immediately before a ';' or ':' — a delimiter APA (parenthesized
# year) and other styles never place right after the year — so these anchors are
# inherently Vancouver-scoped and cannot fire on a parenthesized/comma-delimited
# year. The trailing "[eE]?\d" requires a volume/article-id token after the
# delimiter, rejecting a mid-title "…2020: a review." false positive.
_VANCOUVER_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*[;:]\s*[eE]?\d")
# Container = the capitalized journal-abbreviation run immediately before the year,
# anchored on the title-ending "[.?!] " so a capitalized title fragment is never grabbed.
# Tokens are capitalized words, optionally joined by " & " ("Work & Stress").
_VANCOUVER_CONTAINER_RE = re.compile(
    r"[.?!]\s+([A-Z][A-Za-z]*(?:\s+(?:&\s+)?[A-Z][A-Za-z]*)*)\s+(?:19|20)\d{2}\s*[;:]\s*[eE]?\d"
)
# Full numeric tail: "<YEAR>;<VOL>[(<ISS>)][:<FIRST>[-<LAST>]]". Page tokens allow
# a letter prefix ("S163", "e015981") and an article-id dot ("1222.e1221"), which
# is how these journals print them. The volume stays purely numeric.
_VANCOUVER_TAIL_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*;\s*"  # <YEAR>;
    r"(\d+)"  # volume
    r"(?:\s*\(\s*([^()]{1,24}?)\s*\))?"  # optional (issue)
    r"(?:\s*:\s*"  # optional :pages
    r"([A-Za-z]?\d+(?:\.[A-Za-z]?\d+)?)"
    r"(?:\s*[-–—]\s*([A-Za-z]?\d+))?"
    r")?"
)


def _match_vancouver_tail(fields: dict[str, Any], segment: str) -> re.Match[str] | None:
    """Pick the printed numeric tail this reference's fields belong to.

    Requires an issue or a page after the volume: a bare ``<YEAR>; <N>`` also
    matches book-chapter page spans ("Springer; 2019; 45-52"), where ``N`` is a
    page and not a volume. When the parser already emitted a volume or a year,
    the tail must agree with it — a disagreement means the anchor belongs to
    some other number in the segment, so we fill nothing.
    """
    matches = [m for m in _VANCOUVER_TAIL_RE.finditer(segment) if m.group(3) or m.group(4)]
    if not matches:
        return None
    volume = str(fields.get("volume") or "").strip()
    if volume:
        return next((m for m in matches if m.group(2) == volume), None)
    year = fields.get("year")
    if year:
        return next((m for m in matches if m.group(1) == str(year)), None)
    return matches[0]


def _is_lumped_span(value: Any, segment: str, match: re.Match[str], *groups: int) -> bool:
    """True when ``value`` is unset, or is an undecomposed printed run itself.

    The parser sometimes returns a whole tail run as one span (issue
    ``"368(6469):339-342"``, last_page ``"118–25"``). Replacing a value that is
    byte-identical (modulo whitespace) to the printed run starting at one of
    ``groups`` cannot destroy a correct field — no real issue or page number
    contains the delimiters that follow it.
    """
    if not value:
        return True
    flat = "".join(str(value).split())
    return any(
        match.start(g) >= 0 and flat == "".join(segment[match.start(g) : match.end()].split())
        for g in groups
    )


def _backfill_vancouver_numbers(fields: dict[str, Any], segment: str) -> None:
    """Decompose the Vancouver numeric tail into volume/issue/first/last page.

    Fill-only: a field the parser already populated survives untouched, except
    when it holds the undecomposed span (see ``_is_lumped_span``). Interim
    measure for a ref-parse training-distribution gap — the v4.5-gold parser is
    APA-heavy and tags this tail ``O``.
    """
    match = _match_vancouver_tail(fields, segment)
    if match is None:
        return
    _, volume, issue, first_page, last_page = match.groups()
    if not fields.get("volume"):
        fields["volume"] = volume
    if issue and _is_lumped_span(fields.get("issue"), segment, match, 2, 3):
        fields["issue"] = issue
    if not first_page:
        return
    if _is_lumped_span(fields.get("first_page"), segment, match, 2, 3, 4):
        fields["first_page"] = first_page
    if not last_page or str(fields["first_page"]).strip() != first_page:
        # A first page the parser read differently means the tail is not the
        # range this reference printed — don't pair a last page onto it.
        return
    if _is_lumped_span(fields.get("last_page"), segment, match, 2, 3, 4, 5):
        fields["last_page"] = last_page


def _backfill_vancouver_tail(fields: dict[str, Any], segment: str | None) -> None:
    """Recover dropped fields from a Vancouver journal-tail segment.

    Fills only what the parser left null (never overrides), verbatim from the printed
    ``<Container> <YEAR>;<VOL>(<ISS>):<PAGES>`` tail. The ``<YEAR>[;:]`` anchor is
    Vancouver-specific, so this is a no-op on APA/other styles. Year prefers the value
    immediately before the already-parsed volume (highest precision), else the first
    journal anchor; the numeric tail is then decomposed into volume/issue/pages.
    """
    if not segment:
        return
    if not fields.get("year") and not _IN_PRESS_RE.search(segment):
        year: int | None = None
        volume = fields.get("volume")
        if volume:
            m = re.search(
                rf"(?<!\d)((?:19|20)\d{{2}})\s*[;:]\s*{re.escape(str(volume))}\b", segment
            )
            if m:
                year = int(m.group(1))
        if year is None:
            m = _VANCOUVER_YEAR_RE.search(segment)
            if m:
                year = int(m.group(1))
        if year is not None:
            fields["year"] = year
    if not fields.get("container"):
        m = _VANCOUVER_CONTAINER_RE.search(segment)
        if m:
            fields["container"] = m.group(1)
    _backfill_vancouver_numbers(fields, segment)


# One end of a printed page range. A letter prefix ("S163", "e0123") and an
# article-id dot ("1222.e1221") are how journals print them; the volume/issue
# span never carries either.
_PAGE_NUMBER = r"[A-Za-z]?\d+(?:\.[A-Za-z]?\d+)?"
# A whole "<FIRST>-<LAST>" run that landed in one page field.
_PAGE_RANGE_SPLIT_RE = re.compile(rf"^\s*({_PAGE_NUMBER})\s*[-–—]\s*({_PAGE_NUMBER})\s*$")
# The anchor may not be the tail of a longer token (a DOI's "021-02446", an
# ISSN, a year range "2010-2015" glued to a word) nor sit inside a volume
# "12(3-4)" span, and the partner must end the run — so a hyphenated
# identifier ("s12872-021-02446-z") never yields a page.
_PAGE_RANGE_BOUNDARY_BEFORE = r"(?<![\w./(-])"
_PAGE_RANGE_BOUNDARY_AFTER = r"(?![\w/-])"


def _page_range_partner(anchor: str, segment: str, *, anchor_is_first: bool) -> str | None:
    """The other end of the one printed page range that starts (or ends) with ``anchor``.

    ``None`` when the segment prints no such range, or more than one distinct
    candidate (a volume span and a page span sharing a number): an ambiguous
    anchor must fill nothing, because a wrong page is worse than a missing one.
    """
    token = re.escape(anchor)
    if anchor_is_first:
        pattern = rf"{_PAGE_RANGE_BOUNDARY_BEFORE}{token}\s*[-–—]\s*({_PAGE_NUMBER}){_PAGE_RANGE_BOUNDARY_AFTER}"
    else:
        pattern = rf"{_PAGE_RANGE_BOUNDARY_BEFORE}({_PAGE_NUMBER})\s*[-–—]\s*{token}{_PAGE_RANGE_BOUNDARY_AFTER}"
    candidates = set(re.findall(pattern, segment))
    return candidates.pop() if len(candidates) == 1 else None


def _backfill_page_range(fields: dict[str, Any], segment: str | None) -> None:
    """Complete a half-emitted page range, fill-only.

    Both parsers drop one end of a range: the CRF tags ``PAGE_RANGE_END``
    without a start (audit M9: 421 of 5,071 val120 references), and the LLM
    returns ``first_page`` alone on a compact "339-42". Three repairs, none of
    which overwrites a populated field with a different value:

    * a whole range lumped into one page field ("41–49") is split, and a
      ``last_page`` that repeats the emitted ``first_page`` ("118–25") keeps
      only its own end;
    * a missing ``last_page`` is read from the segment anchored on the emitted
      ``first_page``, and a missing ``first_page`` from the emitted
      ``last_page`` — only when the segment prints exactly one such range.

    Runs before the compact-range expansion, which then turns "339"/"42" into
    "339"/"342" as it always did.
    """
    first = str(fields.get("first_page") or "").strip()
    last = str(fields.get("last_page") or "").strip()
    if last and (not first or last.startswith(first)):
        m = _PAGE_RANGE_SPLIT_RE.match(last)
        if m and (not first or m.group(1) == first):
            fields["first_page"], fields["last_page"] = m.group(1), m.group(2)
            return
    if first and not last:
        m = _PAGE_RANGE_SPLIT_RE.match(first)
        if m:
            fields["first_page"], fields["last_page"] = m.group(1), m.group(2)
            return
    if not segment or bool(first) == bool(last):
        return
    if first:
        partner = _page_range_partner(first, segment, anchor_is_first=True)
        if partner:
            fields["last_page"] = partner
    else:
        partner = _page_range_partner(last, segment, anchor_is_first=False)
        if partner:
            fields["first_page"] = partner


def _finalize_reference_fields(fields: dict[str, Any], segment: str | None) -> dict[str, Any]:
    """Strategy-independent reference finalize shared by every parse path.

    DOI rescue: a clean printed doi:/doi.org token in the ref's own segment
    covers a DOI the parser under-emitted, but never overrides an emitted one.
    Also completes a half-emitted page range from the segment, expands compact
    last pages ("782-92" → "792"), backfills a dropped Vancouver year/container
    from the segment, and migrates/infers ``bib_type``.
    Parser-specific steps (issue handling, author rescue) stay in each parse path.
    """
    _backfill_vancouver_tail(fields, segment)
    _backfill_page_range(fields, segment)
    fields["doi"] = normalize_doi(fields.get("doi")) or _rescue_doi_from_segment(segment)
    fields["last_page"] = _expand_compact_last_page(
        fields.get("first_page"), fields.get("last_page")
    )
    fields["bib_type"] = migrate_bib_type(
        fields.get("bib_type")
        or _infer_bibtype(fields.get("container"), None, segment or fields.get("title") or "")
    )
    return fields


def _is_stub_reference(ref: PaperReference) -> bool:
    """A reference carrying neither a title nor authors — nothing to match on."""
    return not (ref.title or "").strip() and not (ref.authors or "").strip()


def _sequence_references(refs: list[PaperReference]) -> list[PaperReference]:
    """Drop stub refs (no title AND no authors), then assign contiguous
    1-based ``bib_id``s — the single filter/sequence point for all paths."""
    kept = [ref for ref in refs if not _is_stub_reference(ref)]
    for i, ref in enumerate(kept, start=1):
        ref.bib_id = i
    return kept


# Leading run of ≥3 dash-like characters (em/en dash, horizontal bar, figure
# dash, hyphen-minus), each optionally trailed by whitespace, anchored at the
# start of the author byline. Matches the printed "same author as the entry
# above" convention (———, — — —, ---) while leaving a stray one/two-char hyphen
# in a real name alone. OCR often renders an em-dash run as ASCII hyphens, so
# the class includes U+002D.
_REPEAT_AUTHOR_RE = re.compile(r"^\s*(?:[‒–—―-]\s*){3,}")


def _resolve_repeated_authors(refs: list[PaperReference]) -> None:
    """Expand the em-dash "same author as previous entry" placeholder in place.

    Chicago / older-APA / many economics & humanities bibliographies replace a
    repeated author byline with a run of dashes ("———.", "— — —", "——— and
    Fischer") meaning "same author(s) as the entry above". Both parse paths (NER
    and LLM) extract the author span verbatim, so the placeholder survives into
    ``authors`` — useless for xref/Crossref matching.

    Replaces the leading dash-run with the *previous* reference's resolved
    authors, preserving any trailing coauthors ("——— and Fischer" → "Chami, R.
    and Fischer"). Chained dittos follow the resolved chain. Resolution uses only
    the preceding in-document reference (never external enrichment), so it keeps
    the ground-truth contract. A ditto with no resolvable predecessor — the first
    reference, or one whose immediate predecessor has no author — is left
    verbatim rather than borrowing further up the list.
    """
    prev: str | None = None
    for ref in refs:
        authors = (ref.authors or "").strip()
        m = _REPEAT_AUTHOR_RE.match(authors) if authors else None
        if m is None:
            # A real byline (or an author-less entry) sets / breaks the chain.
            prev = authors or None
            continue
        if not prev:
            # No predecessor to copy from — leave the placeholder untouched and
            # keep the chain broken.
            prev = None
            continue
        suffix = authors[m.end() :].strip()
        # A punctuation-only remainder is just the byline terminator ("———.").
        if suffix and not any(c.isalnum() for c in suffix):
            suffix = ""
        # Collapse the space the join introduces before trailing punctuation
        # ("Chami, R. , & Smith" → "Chami, R., & Smith").
        resolved = re.sub(r"\s+([,;.)])", r"\1", f"{prev} {suffix}") if suffix else prev
        ref.authors = resolved
        prev = resolved


def _map_bib_text_ids(refs: list[PaperReference], contents: PaperContents) -> None:
    """Map each reference to its source sentence in the references section.

    Uses title-based substring matching (case-insensitive) to find the
    sentence containing each reference's title, then assigns
    ``ref.text_id`` to that sentence's ``text_id``.

    Falls back to fuzzy matching via ``rapidfuzz.fuzz.partial_ratio``
    when exact substring match fails.
    """
    if not refs:
        return

    # Collect sentences in REFERENCES sections
    ref_section_ids = {
        s.section_id for s in contents.sections if s.section_type == CanonicalSection.REFERENCES
    }
    if not ref_section_ids:
        return

    ref_sentences = [s for s in contents.sentences if s.section_id in ref_section_ids]
    if not ref_sentences:
        return

    from rapidfuzz import fuzz, process

    sent_lower_texts = [s.text.lower() for s in ref_sentences]
    sent_text_ids = [s.text_id for s in ref_sentences]

    for ref in refs:
        if not ref.title or ref.text_id is not None:
            continue

        title_lower = ref.title.lower()

        # Exact substring fast path
        matched_id = next(
            (sent_text_ids[i] for i, t in enumerate(sent_lower_texts) if title_lower in t),
            None,
        )
        if matched_id is not None:
            ref.text_id = matched_id
            continue

        # Vectorized fuzzy fallback (rapidfuzz process.extractOne runs in C)
        result = process.extractOne(
            title_lower,
            sent_lower_texts,
            scorer=fuzz.partial_ratio,
            score_cutoff=75,
        )
        if result is not None:
            ref.text_id = sent_text_ids[result[2]]


def _group_segments_into_chunks(ref_strings: list[str], target_chars: int) -> list[str]:
    """Fallback chunking when regions decline: greedy-join segmented refs."""
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for ref in ref_strings:
        if cur and size + len(ref) > target_chars:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(ref)
        size += len(ref) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _chunk_ner_members(chunk: str, ref_strings: list[str]) -> list[str]:
    """NER-fallback members for one region chunk.

    The already-segmented ``ref_strings`` that fall wholly inside *chunk* are
    the members; any run of chunk text they leave uncovered — a reference
    straddling a region-chunk boundary (``region_chunks`` cuts on region
    onsets, which need not align with the seg strategy's boundaries), or a
    whitespace-normalisation mismatch — is recovered by marker-splitting the
    gap, else kept whole, so a boundary-straddled reference is still parsed
    rather than silently dropped. When no segmented ref matches at all, the
    whole chunk is marker-split (else returned as a single entry).
    """
    covered: list[tuple[int, int]] = []
    for r in ref_strings:
        pos = chunk.find(r)
        if pos != -1:
            covered.append((pos, pos + len(r)))
    if not covered:
        return _marker_split_refs(chunk) or [chunk]
    covered.sort()
    members: list[str] = []
    cursor = 0
    for start, end in covered:
        if start < cursor:
            # Overlapping match (one segment a substring of another) — its
            # text is already accounted for by the earlier, outer span.
            continue
        gap = chunk[cursor:start]
        if gap.strip():
            members.extend(_marker_split_refs(gap) or [gap.strip()])
        members.append(chunk[start:end])
        cursor = end
    trailing = chunk[cursor:]
    if trailing.strip():
        members.extend(_marker_split_refs(trailing) or [trailing.strip()])
    return members


async def _parse_refs_via_ner(
    ext: "ReferenceExtractor",
    ref_text: str,  # noqa: ARG001 — uniform strategy signature
    ref_strings: list[str],
) -> list[PaperReference]:
    # Offload the synchronous ModernBERT-CRF forward pass to a worker thread so
    # it does not block the shared serve event loop (BibrPipelineAPI runs
    # enable_async=True, one loop for many concurrent requests).
    return await asyncio.to_thread(ext._parse_references_ner, ref_strings)


async def _parse_refs_via_llm(
    ext: "ReferenceExtractor", ref_text: str, ref_strings: list[str]
) -> list[PaperReference]:
    return await ext._parse_references_llm(ref_text, ref_strings)


async def _parse_refs_via_llm_chunked(
    ext: "ReferenceExtractor", ref_text: str, ref_strings: list[str]
) -> list[PaperReference]:
    return await ext._parse_references_llm_chunked(ref_text, ref_strings)


# Parse-strategy registry: adding a strategy (e.g. NuExtract) is one adapter +
# one entry; unknown names fall back to "llm" at the dispatch site.
REF_PARSE_STRATEGIES: dict[str, Any] = {
    "ner": _parse_refs_via_ner,
    "llm": _parse_refs_via_llm,
    "llm-chunked": _parse_refs_via_llm_chunked,
}


# Two segments this alike (fuzzy ratio of their alphanumerics) and printing
# the same years are one entry read twice, as when the layout returns an
# aggregate box and the entry boxes inside it and one of the two reads went
# through OCR. Two editions of one work differ in their year.
_REPEAT_MIN_RATIO = 97
_YEAR_TOKEN = re.compile(r"(?<!\d)(?:1[6-9]|20)\d\d(?!\d)")


def _distinct_count(segments: list[str]) -> int:
    """Segments that do not repeat an earlier one, exactly or nearly."""
    from rapidfuzz import fuzz, process

    keys: list[str] = []
    years: list[frozenset[str]] = []
    distinct = 0
    for segment in segments:
        key = _match_key(segment)
        if not key:
            continue
        segment_years = frozenset(_YEAR_TOKEN.findall(segment))
        hits = process.extract(
            key, keys, scorer=fuzz.ratio, score_cutoff=_REPEAT_MIN_RATIO, limit=None
        )
        if not any(years[index] == segment_years for _choice, _score, index in hits):
            distinct += 1
        keys.append(key)
        years.append(segment_years)
    return distinct


@dataclasses.dataclass(frozen=True)
class _StreamChoice:
    """The reference line stream's entries and whether they replace the cascade's."""

    selected: bool
    ref_text: str
    ref_strings: list[str]
    spans: tuple[tuple[int, int] | None, ...]
    # One DOI from a link annotation per final segment (the stream's entries
    # when selected, else the cascade's), None where none applies.
    link_dois: list[str | None]


class ReferenceExtractor:
    """Segments and parses a paper's reference list.

    Consumes the references DataFrame located by
    :class:`bibr.extract.ref_locator.RefLocator`; produces finalized
    ``PaperReference`` objects. Warnings are recorded on
    ``contents.processing_warnings`` so they reach the export.
    """

    def __init__(
        self,
        contents: PaperContents,
        file_hash: str = "unknown",
        llm_client: LlmClient | None = None,
        seg_strategy: str | None = None,
        parse_strategy: str | None = None,
        settings: GlobalSettings | None = None,
    ):
        self.contents = contents
        self.file_hash = file_hash
        self._settings = settings if settings is not None else snapshot_settings()
        self.llm_client = llm_client or LLMClient(settings=self._settings)
        self._ref_seg_strategy = seg_strategy
        self._ref_parse_strategy = parse_strategy
        # Diagnostics the reference line stream adds to the yield receipt.
        self._stream_reason_flags: set[str] = set()
        self.validation_issues: list[ValidationIssue] = []
        self._segmentation_attempts: list[ReferenceSegmentationAttempt] = []
        self._selected_segmentation_spans: tuple[tuple[int, int], ...] = ()
        self._credible_source_starts: int | None = None
        self._source_record_count: int | None = None
        self._reference_pages: tuple[int, int] | None = None

    # Class-attribute seams so tests can patch capture without touching the
    # module functions other callers share.
    _save_ref_training_data = staticmethod(save_ref_training_data)
    _save_seg_training_data = staticmethod(save_seg_training_data)

    async def extract(self, ref_df: pd.DataFrame) -> list[PaperReference]:
        """Segment the references block, then parse each reference.

        Segmentation and parsing strategies are resolved independently
        (:func:`_resolve_ref_strategies`). Segmentation: LLM anchor-emit with
        automatic CRF fallback, or CRF directly. Parsing: batched LLM
        (≤``REF_PARSE_BATCH_SIZE`` refs/call) or per-ref NER.
        """
        self.contents.reference_yield_receipt = None
        self.validation_issues.clear()
        self._stream_reason_flags = set()
        self._segmentation_attempts.clear()
        self._selected_segmentation_spans = ()
        self._credible_source_starts = None
        self._source_record_count = None
        self._reference_pages = _page_range(ref_df)

        seg_strategy, parse_strategy = _resolve_ref_strategies(
            self._ref_seg_strategy,
            self._ref_parse_strategy,
            settings=self._settings,
        )
        source_rows = ref_df["text"].astype(str).tolist()
        source_record_count = sum(bool(collapse_ws(row)) for row in source_rows)
        self._source_record_count = source_record_count
        ref_text = _build_ref_text(source_rows)
        logger.info(
            f"Reference extraction: {len(ref_text)} chars "
            f"(seg={seg_strategy}, parse={parse_strategy})"
        )

        raw_ref_strings = await self._segment_references(ref_text, seg_strategy)
        raw_spans = self._selected_segmentation_spans
        ref_strings = raw_ref_strings
        if not self._authoritative_native_selected():
            ref_strings = drop_non_reference_segments(ref_strings)
            ref_strings = self._maybe_split_merged(ref_strings)
        if ref_strings == raw_ref_strings and len(raw_spans) == len(ref_strings):
            located_spans: tuple[tuple[int, int] | None, ...] = raw_spans
        else:
            located_spans = _locate_segment_spans(ref_text, ref_strings)
        ref_strings, selected_spans, duplicate_rate, duplicate_reasons = (
            _prepare_segment_candidates(ref_text, ref_strings, located_spans)
        )
        stream_choice = await asyncio.to_thread(
            self._consider_line_stream, ref_df, ref_text, ref_strings, seg_strategy
        )
        link_dois: list[str | None] | None = None
        if stream_choice is not None:
            link_dois = stream_choice.link_dois
            if stream_choice.selected:
                ref_text = stream_choice.ref_text
                ref_strings, selected_spans, duplicate_rate, duplicate_reasons = (
                    _prepare_segment_candidates(
                        ref_text, stream_choice.ref_strings, stream_choice.spans
                    )
                )
        logger.info(f"Segmented {len(ref_strings)} references")

        parser = REF_PARSE_STRATEGIES.get(parse_strategy, REF_PARSE_STRATEGIES["llm"])
        if parse_strategy == "ner" and link_dois and any(link_dois):
            all_refs = await asyncio.to_thread(
                self._parse_references_ner_with_links, ref_strings, link_dois
            )
        else:
            all_refs = await parser(self, ref_text, ref_strings)

        # Resolve the em-dash "same author as above" convention on the fully
        # ordered list — a cross-reference step every parse strategy shares.
        _resolve_repeated_authors(all_refs)
        _map_bib_text_ids(all_refs, self.contents)
        self._record_yield_receipt(
            ref_text,
            selected_spans,
            parsed_refs=all_refs,
            duplicate_rate=duplicate_rate,
            duplicate_reasons=duplicate_reasons,
            source_record_count=source_record_count,
        )
        logger.info(f"Extracted {len(all_refs)} references")
        return all_refs

    def _record_segmentation_attempt(
        self,
        strategy: str,
        ref_text: str,
        *,
        segments: list[str] | None = None,
        spans: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
        credible_starts: int | None = None,
        selected: bool,
        reason_flags: tuple[str, ...] = (),
    ) -> None:
        resolved_spans = tuple(spans or ())
        if not resolved_spans and segments:
            resolved_spans = _segments_to_spans(ref_text, segments)
        attempt = ReferenceSegmentationAttempt(
            strategy=strategy,
            spans=resolved_spans,
            credible_starts=credible_starts,
            selected=selected,
            reason_flags=reason_flags,
        )
        self._segmentation_attempts.append(attempt)
        if credible_starts is not None:
            current = self._credible_source_starts or 0
            self._credible_source_starts = max(current, credible_starts)
        if selected:
            self._selected_segmentation_spans = resolved_spans

    def _estimate_credible_source_starts(self, ref_text: str) -> int | None:
        estimates: list[int] = []
        native = self.contents.native_ref_strings
        if isinstance(native, (list, tuple)) and native:
            estimates.append(len(native))
        region_onsets = self._aligned_region_onset_count(ref_text)
        if region_onsets:
            estimates.append(region_onsets)
        marker_starts = len(list(_MARKER_LINE_RE.finditer(ref_text)))
        if marker_starts >= 3:
            estimates.append(marker_starts)
        if self._credible_source_starts is not None:
            estimates.append(self._credible_source_starts)
        return max(estimates) if estimates else None

    def _aligned_region_onset_count(self, ref_text: str) -> int:
        """Count unique layout anchors aligned to physical source offsets."""
        summaries = self._reference_region_summaries()
        if not summaries:
            return 0
        try:
            return len(set(find_anchor_starts(ref_text, region_anchor_texts(summaries))))
        except Exception:  # noqa: BLE001 — receipt evidence is best-effort
            return 0

    def _record_yield_receipt(
        self,
        ref_text: str,
        selected_spans: tuple[tuple[int, int], ...],
        *,
        parsed_refs: list[PaperReference],
        duplicate_rate: float,
        duplicate_reasons: tuple[str, ...],
        source_record_count: int | None = None,
    ) -> None:
        credible_starts = self._estimate_credible_source_starts(ref_text)
        parsed_count = len(parsed_refs)
        valid_count = sum(
            bool((ref.title or "").strip() or (ref.authors or "").strip()) for ref in parsed_refs
        )
        reasons = set(duplicate_reasons)
        reasons.update(getattr(self.contents, "reference_boundary_reason_flags", None) or ())
        reasons.update(self._stream_reason_flags)
        unavailable_offsets = {
            "source_offsets_unavailable",
            "source_offsets_partially_unavailable",
        }.intersection(reasons)
        coverage = (
            None if unavailable_offsets else _source_character_coverage(ref_text, selected_spans)
        )

        expected_starts = len(selected_spans)
        if credible_starts is not None:
            expected_starts = max(expected_starts, credible_starts)
            if credible_starts >= 5 and valid_count < 0.6 * expected_starts:
                reasons.add("credible_start_under_yield")
        if (
            coverage is not None
            and coverage >= 0.5
            and len(selected_spans) >= 5
            and valid_count < 0.6 * len(selected_spans)
        ):
            reasons.add("parse_under_yield")
        if (
            source_record_count is not None
            and source_record_count >= 5
            and valid_count < source_record_count * 0.85
        ):
            reasons.add("source_record_under_yield")
            expected_starts = max(expected_starts, source_record_count)

        ordered_reasons = tuple(sorted(reasons))
        receipt = ReferenceYieldReceipt(
            credible_source_starts=credible_starts,
            attempts=tuple(self._segmentation_attempts),
            selected_spans=selected_spans,
            source_character_coverage=coverage,
            parsed_count=parsed_count,
            valid_count=valid_count,
            duplicate_rate=duplicate_rate,
            reason_flags=ordered_reasons,
        )
        self.contents.reference_yield_receipt = receipt
        if {
            "credible_start_under_yield",
            "parse_under_yield",
            "source_record_under_yield",
        }.intersection(reasons):
            denominator = (
                f" of {expected_starts} credible/selected starts" if expected_starts else ""
            )
            self.validation_issues.append(
                ValidationIssue(
                    "VAL_REF_LOW_YIELD",
                    IssueSeverity.WARNING,
                    f"{valid_count} valid parsed references{denominator}",
                    origin_stage="extract",
                    count=max(1, expected_starts - valid_count),
                    blocking=False,
                )
            )

    async def _segment_references(self, ref_text: str, seg_strategy: str) -> list[str]:
        """Return per-reference strings.

        ``geom`` runs the local geometry GBM and cascades region→LLM→CRF when
        geometry is absent or the model is unconfident; ``region`` segments by
        layout-region anchors with LLM→CRF fallback; ``crf`` runs the CRF
        directly; ``llm`` is anchor-emit with region→CRF fallback.

        Native pre-segmentation wins over any configured strategy: when the
        input carried already-delimited reference strings (JATS mixed-citations
        on ``contents.native_ref_strings``), segmentation is a solved problem —
        return them verbatim and skip the LLM/geom cascade entirely.
        """
        native = self.contents.native_ref_strings
        if isinstance(native, (list, tuple)) and native:
            logger.debug("Using %d pre-segmented native reference strings", len(native))
            segments = list(native)
            self._record_segmentation_attempt(
                "native",
                ref_text,
                segments=segments,
                credible_starts=len(segments),
                selected=True,
            )
            return segments

        if seg_strategy == "crf":
            return await asyncio.to_thread(self._crf_segment_or_recover, ref_text)

        if seg_strategy == "geom":
            # Offload the synchronous GBM predict to a worker thread so it does
            # not block the shared serve event loop (see _parse_refs_via_ner).
            geom_strings = await asyncio.to_thread(self._segment_geom, ref_text)
            if geom_strings is not None:
                return geom_strings
            return await self._segment_fallback_chain(ref_text)

        if seg_strategy == "region":
            region_strings = self._segment_region_anchors(ref_text, as_fallback=False)
            if region_strings:
                return region_strings
            return await self._segment_fallback_chain(ref_text, try_region=False)

        return await self._segment_llm_then_crf(ref_text)

    def _source_recall_shortfall(self, segment_count: int) -> int | None:
        """Source-record count when *segment_count* falls short of it, else None.

        The region tier's own alignment threshold is computed over surviving
        anchors, so it cannot see references the onset filter discarded. This is
        the missing recall term: the reference section's source record count is
        an independent lower bound on how many references exist.
        """
        min_recall = self._settings.REF_SEG_MIN_SOURCE_RECALL
        records = self._source_record_count
        if not min_recall or not records or records < 5:
            return None
        return records if segment_count < min_recall * records else None

    async def _segment_fallback_chain(self, ref_text: str, try_region: bool = True) -> list[str]:
        """Geom-decline cascade: region anchors (zero cost) → LLM → CRF."""
        region_reserve: list[str] | None = None
        if try_region:
            region_strings = self._segment_region_anchors(ref_text)
            if region_strings:
                records = self._source_recall_shortfall(len(region_strings))
                if records is None:
                    return region_strings
                # Keep the segmentation as a reserve rather than discarding it:
                # it is still better than the CRF last resort if the LLM fails.
                region_reserve = region_strings
                self._record_warning(
                    WarningCode.REF_SEG_REGION_CASCADE,
                    f"region segment count {len(region_strings)} < "
                    f"{self._settings.REF_SEG_MIN_SOURCE_RECALL:.2f} × {records} "
                    "reference-section source records",
                )
        if self._settings.REF_SEG_LLM_FALLBACK:
            return await self._segment_llm_then_crf(
                ref_text, try_region=False, reserve=region_reserve
            )
        self._record_segmentation_attempt(
            "llm_anchor", ref_text, selected=False, reason_flags=("tier_disabled",)
        )
        self._record_warning(
            WarningCode.REF_SEG_CRF_FALLBACK, "LLM seg tier disabled (REF_SEG_LLM_FALLBACK=false)"
        )
        if region_reserve:
            return self._select_region_reserve(ref_text, region_reserve)
        return await asyncio.to_thread(self._crf_segment_or_recover, ref_text)

    def _select_region_reserve(self, ref_text: str, reserve: list[str]) -> list[str]:
        """Re-select a held-back region segmentation after the LLM tier failed."""
        self._record_segmentation_attempt(
            "region",
            ref_text,
            segments=reserve,
            credible_starts=self._aligned_region_onset_count(ref_text) or None,
            selected=True,
        )
        self._record_warning(
            WarningCode.REF_SEG_REGION_RECOVERY,
            f"kept {len(reserve)} region segment(s) after the LLM tier failed",
        )
        return reserve

    def _segment_geom(self, ref_text: str) -> list[str] | None:
        """Local geometry segmentation. Returns ref strings, or None to cascade."""
        lines = self.contents.ref_line_geometry
        if not lines:
            self._record_segmentation_attempt(
                "geom",
                ref_text,
                selected=False,
                reason_flags=("source_geometry_unavailable",),
            )
            self._record_warning(
                WarningCode.REF_SEG_GEOM_CASCADE, "no ref-line geometry (DOCX/non-native)"
            )
            return None
        with LOCAL_INFERENCE_LOCK:
            segmenter = _get_geom_segmenter(self._settings)
        if segmenter is None:
            self._record_segmentation_attempt(
                "geom", ref_text, selected=False, reason_flags=("segmenter_unavailable",)
            )
            self._record_warning(
                WarningCode.REF_SEG_GEOM_CASCADE, "geom segmenter unavailable (deps/artifact)"
            )
            return None
        try:
            with LOCAL_INFERENCE_LOCK:
                spans, confidence, labeled, aligned = segmenter.segment_spans(ref_text, lines)
        except Exception as e:  # noqa: BLE001 — any geom failure cascades
            self._record_segmentation_attempt(
                "geom", ref_text, selected=False, reason_flags=("segmentation_error",)
            )
            self._record_warning(WarningCode.REF_SEG_GEOM_CASCADE, f"geom error: {e!r}")
            return None
        # Alignment-yield gate: see REF_GEOM_MIN_ALIGN_YIELD in config.py for
        # the OOD rationale. Rides the same cascade path as a sub-threshold
        # confidence result.
        if labeled == 0:
            self._record_segmentation_attempt(
                "geom",
                ref_text,
                spans=spans,
                credible_starts=None,
                selected=False,
                reason_flags=("zero_labeled_starts",),
            )
            self._record_warning(
                WarningCode.REF_SEG_GEOM_CASCADE,
                f"geom alignment yield undefined (labeled 0, aligned {aligned})",
            )
            return None
        yield_ratio = aligned / labeled
        aligned_starts = aligned or None
        if yield_ratio < self._settings.REF_GEOM_MIN_ALIGN_YIELD:
            self._record_segmentation_attempt(
                "geom",
                ref_text,
                spans=spans,
                credible_starts=aligned_starts,
                selected=False,
                reason_flags=("low_alignment_yield",),
            )
            self._record_warning(
                WarningCode.REF_SEG_GEOM_CASCADE,
                f"geom alignment yield {yield_ratio:.2f} < "
                f"{self._settings.REF_GEOM_MIN_ALIGN_YIELD} "
                f"(labeled {labeled}, aligned {aligned})",
            )
            return None
        # Segment-count sanity gate: a confidently *under*-segmenting geom result
        # (arXiv/ACL bare-year styles the GBM's text features miss) clears both
        # the confidence and align-yield gates yet emits far too few spans. The
        # layout reference-onset count is an independent lower bound — when geom
        # emits fewer than GEOM_MIN_REGION_RATIO of it, decline and cascade like a
        # low-confidence result rather than trust the merged spans.
        region_onsets = self._region_onset_count()
        receipt_region_onsets = self._aligned_region_onset_count(ref_text) if region_onsets else 0
        min_region_ratio = GEOM_MIN_REGION_RATIO
        if (
            region_onsets
            and receipt_region_onsets >= GEOM_STRONG_REGION_ALIGN_FRACTION * region_onsets
        ):
            min_region_ratio = GEOM_STRONG_REGION_RATIO
        if region_onsets and len(spans) < min_region_ratio * region_onsets:
            self._record_segmentation_attempt(
                "geom",
                ref_text,
                spans=spans,
                credible_starts=max(aligned_starts or 0, receipt_region_onsets) or None,
                selected=False,
                reason_flags=("low_segment_count",),
            )
            self._record_warning(
                WarningCode.REF_SEG_GEOM_CASCADE,
                f"geom segment count {len(spans)} < "
                f"{min_region_ratio:.2f} × {region_onsets} region onsets",
            )
            return None
        ref_strings = [ref_text[s:e] for s, e in spans]
        if ref_strings and confidence >= self._settings.REF_GEOM_SEG_CASCADE_THRESHOLD:
            self._record_segmentation_attempt(
                "geom", ref_text, spans=spans, credible_starts=aligned_starts, selected=True
            )
            # Not captured as seg training data: a model prediction, not an LLM
            # label (see save_seg_training_data).
            return ref_strings
        self._record_segmentation_attempt(
            "geom",
            ref_text,
            spans=spans,
            credible_starts=aligned_starts,
            selected=False,
            reason_flags=("low_confidence_or_empty",),
        )
        self._record_warning(
            WarningCode.REF_SEG_GEOM_CASCADE,
            f"geom low-confidence ({confidence:.3f} < "
            f"{self._settings.REF_GEOM_SEG_CASCADE_THRESHOLD}) or empty",
        )
        return None

    def _region_onset_count(self) -> int:
        """Count of layout reference-onset anchors, or 0 when unavailable.

        The same filtered ``reference_content`` region stream that
        ``_segment_region_anchors`` snaps onto ``ref_text`` — used only as an
        independent lower bound for the geom segment-count sanity gate. Never
        raises: an absent or malformed summary stream just disables the gate.
        """
        summaries = self._reference_region_summaries()
        if not summaries:
            return 0
        try:
            return len(region_anchor_texts(summaries))
        except Exception:  # noqa: BLE001 — the gate must never break extraction
            return 0

    def _reference_region_summaries(self) -> list:
        """Layout regions on the pages of the located reference rows.

        ``contents.region_summaries`` covers the whole document. Reference
        onsets from another list (a multi-article PDF, supplementary references
        the locator did not select, a tail trimmed off the section) would
        otherwise inflate the geom segment-count gate's lower bound and pull
        the region tier's alignment fraction under its threshold, declining
        both free tiers for a correct segmentation. Every summary is kept when
        the reference rows carry no page numbers, and so is any summary without
        a page.
        """
        summaries = list(getattr(self.contents, "region_summaries", None) or [])
        if self._reference_pages is None:
            return summaries
        first, last = self._reference_pages
        return [
            summary
            for summary in summaries
            if not isinstance(getattr(summary, "page", None), int) or first <= summary.page <= last
        ]

    async def _segment_llm_then_crf(
        self, ref_text: str, try_region: bool = True, reserve: list[str] | None = None
    ) -> list[str]:
        """LLM anchor-emit segmentation with region-anchor → CRF fallback.

        *reserve* is a region segmentation the caller held back on a recall
        shortfall; it is preferred over the CRF last resort if the LLM fails.
        """
        try:
            anchors = await self.llm_client.segment_references(ref_text, file_hash=self.file_hash)
            ref_strings = segment_by_anchors(ref_text, anchors)
            spans = _segments_to_spans(ref_text, ref_strings)
            if ref_strings:
                self._record_segmentation_attempt(
                    "llm_anchor", ref_text, spans=spans, selected=True
                )
                self._save_seg_training_data(ref_text, ref_strings, settings=self._settings)
                return ref_strings
            self._record_warning(
                WarningCode.REF_SEG_CRF_FALLBACK, "LLM segmentation produced 0 usable spans"
            )
            self._record_segmentation_attempt(
                "llm_anchor", ref_text, selected=False, reason_flags=("no_usable_spans",)
            )
        except ProcessingError:
            raise
        except Exception as e:  # noqa: BLE001 — any seg failure triggers fallback
            self._record_segmentation_attempt(
                "llm_anchor", ref_text, selected=False, reason_flags=("segmentation_error",)
            )
            self._record_warning(WarningCode.REF_SEG_CRF_FALLBACK, f"LLM segmentation error: {e!r}")
        if try_region:
            region_strings = self._segment_region_anchors(ref_text)
            if region_strings:
                return region_strings
        if reserve:
            return self._select_region_reserve(ref_text, reserve)
        return await asyncio.to_thread(self._crf_segment_or_recover, ref_text)

    def _segment_region_anchors(
        self, ref_text: str, *, as_fallback: bool = True
    ) -> list[str] | None:
        """Zero-cost layout-region anchor segmentation.

        The first fallback tier on the geom-decline path (and the primary tier
        for ``REF_SEG_STRATEGY=region``); also the fallback after an explicit
        ``llm`` strategy fails. Declines (``None``) when the region stream is
        absent, too sparse, or does not align with the text.

        ``as_fallback`` distinguishes the two call contexts: when ``False``
        (an explicit ``region`` primary selection), ``REF_SEG_REGION_ANCHORS``
        is bypassed — that flag gates the FALLBACK TIER only, mirroring how
        ``REF_SEG_LLM_FALLBACK`` does not affect an explicit ``llm``
        strategy — every decline is recorded (never silent) so the cascade
        is observable, and a successful segmentation does NOT emit the
        fallback-tier recovery warning (that warning means "recovered after
        another tier declined", which is not true of the normal primary
        path).
        """
        if as_fallback and not self._settings.REF_SEG_REGION_ANCHORS:
            self._record_segmentation_attempt(
                "region", ref_text, selected=False, reason_flags=("tier_disabled",)
            )
            return None
        summaries = self._reference_region_summaries()
        if not summaries:
            self._record_segmentation_attempt(
                "region", ref_text, selected=False, reason_flags=("no_summaries",)
            )
            if not as_fallback:
                self._record_warning(
                    WarningCode.REF_SEG_REGION_CASCADE, "no layout regions available"
                )
            return None
        try:
            segments = segment_by_region_anchors(ref_text, summaries)
        except Exception as e:  # noqa: BLE001 — a fallback tier must never raise
            self._record_segmentation_attempt(
                "region", ref_text, selected=False, reason_flags=("segmentation_error",)
            )
            self._record_warning(WarningCode.REF_SEG_REGION_ERROR, f"region-anchor error: {e!r}")
            return None
        if segments:
            credible_starts = self._aligned_region_onset_count(ref_text)
            self._record_segmentation_attempt(
                "region",
                ref_text,
                segments=segments,
                credible_starts=credible_starts or None,
                selected=True,
            )
            if as_fallback:
                self._record_warning(
                    WarningCode.REF_SEG_REGION_RECOVERY,
                    f"recovered {len(segments)} segment(s) from layout regions",
                )
            return segments
        if not as_fallback:
            self._record_warning(
                WarningCode.REF_SEG_REGION_CASCADE, "too few anchors or misaligned with ref_text"
            )
        self._record_segmentation_attempt(
            "region",
            ref_text,
            credible_starts=self._aligned_region_onset_count(ref_text) or None,
            selected=False,
            reason_flags=("too_few_or_misaligned_anchors",),
        )
        return segments

    def _crf_segment_or_recover(self, ref_text: str) -> list[str]:
        """CRF last-resort segmentation, guarded so a populated references region
        never silently yields 0.

        Synchronous: a ModernBERT-CRF load plus N sliding-window forward passes
        holding the process-wide inference lock. Every caller offloads it with
        ``asyncio.to_thread``, matching ``_segment_geom`` and
        ``_parse_refs_via_ner`` — ``bibr serve`` runs one async worker, so this
        on the loop is head-of-line blocking for every co-resident request. On CRF empty OR exception (e.g. a missing NER
        checkpoint), try a zero-LLM marker split; if that also fails on a
        substantial region, surface a HIGH-severity warning instead of 0 refs.
        """
        failure_reason = "no_segments"
        try:
            with LOCAL_INFERENCE_LOCK:
                ref_strings = _get_ner_segmenter(self._settings).segment(ref_text)
        except Exception as e:  # noqa: BLE001 — a CRF load/inference failure must not 0 the refs
            failure_reason = "segmentation_error"
            self._record_warning(WarningCode.REF_SEG_CRF_ERROR, f"CRF segmenter error: {e!r}")
            ref_strings = []
        if ref_strings:
            self._record_segmentation_attempt("crf", ref_text, segments=ref_strings, selected=True)
            return ref_strings
        self._record_segmentation_attempt(
            "crf", ref_text, selected=False, reason_flags=(failure_reason,)
        )
        if len(ref_text.strip()) >= _REF_REGION_NONEMPTY_MIN_CHARS:
            recovered = _marker_split_refs(ref_text)
            if recovered:
                self._record_segmentation_attempt(
                    "marker_split",
                    ref_text,
                    segments=recovered,
                    credible_starts=len(recovered),
                    selected=True,
                )
                self._record_warning(
                    WarningCode.REF_SEG_MARKER_SPLIT_RECOVERY,
                    f"recovered {len(recovered)} segment(s) by splitting on reference markers",
                )
                return recovered
            self._record_warning(
                WarningCode.REF_SEG_FAILED,
                f"references region present ({len(ref_text.strip())} chars) but 0 segments",
            )
        return ref_strings

    def _authoritative_native_selected(self) -> bool:
        """Whether the selected segmentation is a JATS ref-list taken verbatim.

        Each ``<ref>`` is one reference by construction, so the junk filter
        (which drops short entries without a year) and the merge splitter
        (which cuts at in-title citations and bare years) can only lose or
        split real entries, shifting every later bib_id.
        """
        if getattr(self.contents, "native_ref_strings_authoritative", False) is not True:
            return False
        return any(
            attempt.strategy == "native" and attempt.selected
            for attempt in self._segmentation_attempts
        )

    def _maybe_split_merged(self, ref_strings: list[str]) -> list[str]:
        """Split merged reference strings when REF_SPLIT_MERGED_REFS is on.

        Segmenter-agnostic: runs on the post-segment strings regardless of which
        segmentation strategy produced them. One-sided-safe and best-effort.
        """
        if not self._settings.REF_SPLIT_MERGED_REFS:
            return ref_strings
        split, n_new = split_merged_refs(ref_strings)
        if n_new:
            self._record_warning(
                WarningCode.REF_SEG_MERGE_SPLIT,
                f"split merged reference strings into {n_new} more segment(s)",
            )
        return split

    def _record_warning(self, code: WarningCode, message: str) -> None:
        """Log and persist a segmentation/parse fallback, recovery or loss so
        eval tooling can count it."""
        logger.warning("Reference extraction warning %s: %s", code, message)
        # Mock(spec=PaperContents) doubles don't expose default_factory
        # dataclass fields — create the sink on first write if missing.
        if not hasattr(self.contents, "processing_warnings"):
            self.contents.processing_warnings = []
        self.contents.processing_warnings.append(ProcessingWarning(code, message))

    async def _reparse_split(
        self, batch: list[str], offset: int, depth: int
    ) -> tuple[list[Any], list[tuple[int, list[str], BaseException]]]:
        """Re-parse a degenerately-failed batch slice, halving on repeat failure.

        Returns ``(parsed_llm_refs, failed_spans)`` where each failed span is
        ``(document_offset, segments, cause)`` for the caller's NER fallback.
        ``depth`` bounds the recursion: with the default entry depth of 1 a
        15-ref batch retries as 8+7, then 4+4+4+3 — small enough for an 8k
        context window, and at most 6 extra LLM calls per failed batch.
        """
        numbered = "\n".join(f"{offset + i}. {r}" for i, r in enumerate(batch))
        try:
            refs = await self.llm_client.extract_references(
                numbered,
                file_hash=self.file_hash,
                start_index=offset,
                expected_count=len(batch),
            )
            return refs, []
        except asyncio.CancelledError:
            raise
        except ProcessingError:
            raise
        except BaseException as exc:  # noqa: BLE001 — every failure lands in a NER span
            if depth > 0 and len(batch) > 1 and _is_degenerate_ref_failure(exc):
                mid = (len(batch) + 1) // 2
                left, right = await asyncio.gather(
                    self._reparse_split(batch[:mid], offset, depth - 1),
                    self._reparse_split(batch[mid:], offset + mid, depth - 1),
                )
                return left[0] + right[0], left[1] + right[1]
            return [], [(offset, batch, exc)]

    async def _parse_references_llm(
        self, ref_text: str, ref_strings: list[str]
    ) -> list[PaperReference]:
        """Parse pre-segmented references in concurrent batches of ≤N."""
        batch_size = max(1, self._settings.REF_PARSE_BATCH_SIZE)
        batches = list(_chunk(ref_strings, batch_size))
        numbered_batches: list[str] = []
        batch_offsets: list[int] = []
        tasks = []
        offset = 1
        for batch in batches:
            numbered = "\n".join(f"{offset + i}. {r}" for i, r in enumerate(batch))
            numbered_batches.append(numbered)
            batch_offsets.append(offset)
            tasks.append(
                self.llm_client.extract_references(
                    numbered,
                    file_hash=self.file_hash,
                    start_index=offset,
                    expected_count=len(batch),
                )
            )
            offset += len(batch)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, ProcessingError):
                raise result
        llm_refs = []
        failures: list[BaseException] = []
        # Indices of refs in a batch the LLM returned with EVERY author null —
        # the observed batch-coherence failure (qual-grade30 M2). Authors are
        # rescued from each ref's own segment only for these; a lone null-author
        # ref in an otherwise-populated batch is left as the LLM emitted it.
        rescue_author_indices: set[int] = set()
        # Refs recovered from their own segments when an LLM batch hard-fails,
        # tagged with the batch's starting document index so they merge back in
        # order. A failed batch is never silently dropped: a *degenerate* failure
        # (timeout / output-cap loop) falls straight back to the NER parser,
        # while a *transient* failure (5xx/429 under concurrent-batch load) is
        # retried once and only then falls back.
        ner_recovered: list[tuple[int, PaperReference]] = []

        async def _fallback_to_ner(offset: int, segs: list[str], cause: BaseException) -> None:
            """Recover a failed span's refs via the NER parser, tagged with
            their document position so they merge back in order. Logs and gives
            up on just this span if NER itself fails."""
            if isinstance(cause, ProcessingError):
                raise cause
            failures.append(cause)
            last_index = offset + len(segs) - 1
            try:
                # Same offload as _parse_refs_via_ner: keep the CRF forward
                # pass off the shared serve event loop. The *aligned* variant
                # keeps one slot per input segment, so a survivor's tag is its
                # true document position rather than its position among the
                # survivors — the filtered variant silently pulled every ref
                # after a dropped slot one position earlier.
                aligned = await asyncio.to_thread(self._parse_references_ner_aligned, segs)
            except ProcessingError:
                raise
            except Exception as ner_exc:  # noqa: BLE001
                logger.error(
                    "Reference parse batch (indices %d-%d) hard-failed and "
                    "NER fallback also failed: %s",
                    offset,
                    last_index,
                    ner_exc,
                )
                # The span's references are gone. "A failed batch is never
                # silently dropped" is a contract this path was breaking: only
                # a log line marked the loss, so the export was
                # indistinguishable from a paper that simply printed fewer
                # references.
                self._record_warning(
                    WarningCode.REF_PARSE_LOST,
                    f"indices {offset}-{last_index}, 0/{len(segs)} refs recovered "
                    f"(NER fallback also failed: {ner_exc.__class__.__name__})",
                )
                return
            recovered = [(slot, ref) for slot, ref in enumerate(aligned) if ref is not None]
            for slot, rec in recovered:
                ner_recovered.append((offset + slot, rec))
            self._record_warning(
                WarningCode.REF_PARSE_NER_FALLBACK,
                f"indices {offset}-{last_index}, {len(recovered)}/{len(segs)} refs "
                f"recovered ({cause.__class__.__name__})",
            )

        for bi, res in enumerate(results):
            if isinstance(res, BaseException):
                # A degenerate failure (the model timed out or looped to its
                # output-token cap) is deterministic under greedy decoding —
                # re-asking the IDENTICAL prompt just reproduces the loop. But
                # on small-context servers the same signature is usually
                # size-induced (the batch overflowed the window), and a SMALLER
                # prompt is a different prompt: split the batch and retry the
                # halves (then quarters) before surrendering the span to NER.
                if _is_degenerate_ref_failure(res):
                    b, off = batches[bi], batch_offsets[bi]
                    # An output-cap truncation still delivered the complete
                    # leading refs in its raw completion; keep those and shrink
                    # the batch to the un-parsed tail before splitting / NER.
                    salvaged = _salvage_truncated_batch(res, off)
                    if salvaged:
                        llm_refs.extend(salvaged)
                        self._record_warning(
                            WarningCode.REF_PARSE_SALVAGE_RECOVERY,
                            f"indices {off}-{off + len(salvaged) - 1}, "
                            f"{len(salvaged)}/{len(b)} refs salvaged from truncated "
                            f"completion ({res.__class__.__name__})",
                        )
                        b, off = b[len(salvaged) :], off + len(salvaged)
                        if not b:
                            continue
                    logger.warning(
                        "Reference parse batch (indices %d-%d) failed degenerately (%s); %s",
                        off,
                        off + len(b) - 1,
                        res,
                        "splitting batch before NER fallback"
                        if len(b) > 1
                        else "falling back to NER",
                    )
                    if len(b) > 1:
                        mid = (len(b) + 1) // 2
                        (l_refs, l_failed), (r_refs, r_failed) = await asyncio.gather(
                            self._reparse_split(b[:mid], off, depth=1),
                            self._reparse_split(b[mid:], off + mid, depth=1),
                        )
                        split_refs = l_refs + r_refs
                        if split_refs:
                            self._record_warning(
                                WarningCode.REF_PARSE_SPLIT_RECOVERY,
                                f"indices {off}-{off + len(b) - 1}, "
                                f"{len(split_refs)}/{len(b)} refs re-parsed in "
                                f"smaller batches ({res.__class__.__name__})",
                            )
                            llm_refs.extend(split_refs)
                        for span_off, span_segs, span_exc in l_failed + r_failed:
                            await _fallback_to_ner(span_off, span_segs, span_exc)
                    else:
                        await _fallback_to_ner(off, b, res)
                    continue
                logger.warning("Reference parse batch failed: %s; retrying once", res)
                try:
                    res = await self.llm_client.extract_references(
                        numbered_batches[bi],
                        file_hash=self.file_hash,
                        start_index=batch_offsets[bi],
                        expected_count=len(batches[bi]),
                    )
                except asyncio.CancelledError:
                    # Cancellation is not a parse failure — propagate it so the
                    # cancelled coroutine actually stops, never divert to NER.
                    raise
                except ProcessingError:
                    raise
                except BaseException as retry_exc:  # noqa: BLE001
                    await _fallback_to_ner(batch_offsets[bi], batches[bi], retry_exc)
                    continue
            # ``res`` non-empty, not ``len(res) > 1``: with a batch size of 15,
            # a paper whose reference count is 1 mod 15 ends in a single-entry
            # remainder batch, and requiring two refs left that last reference
            # permanently unrescuable. A one-ref batch that came back
            # author-less IS an all-null batch — there is no
            # "otherwise-populated batch" for it to be the odd one out of.
            if res and all(not r.authors for r in res):
                rescue_author_indices.update(r.index for r in res)
                logger.warning(
                    "Reference parse batch returned all-null authors for %d refs "
                    "(indices %s-%s); rescuing from segments",
                    len(res),
                    res[0].index,
                    res[-1].index,
                )
            llm_refs.extend(res)

        if tasks and not llm_refs and not ner_recovered:
            # Nothing recovered at all (every batch failed the LLM *and* its NER
            # fallback) — an empty result here would be indistinguishable from a
            # paper without references, so fail loud. If NER recovered any batch,
            # we keep those refs rather than discarding a partial-but-valid list.
            # ``failures`` can be empty here: the prompt invites skipping
            # entries it cannot parse, so every batch can "succeed" while
            # returning nothing. Indexing it unguarded turned that into an
            # IndexError — reported as a crash rather than the upstream
            # failure it is. Mirrors the chunked path's guard.
            cause = failures[0] if failures else None
            raise UpstreamServiceError(
                "LLM",
                f"All {len(tasks)} reference parse batches failed",
                cause,
            ) from cause

        self._save_ref_training_data(ref_text, llm_refs, settings=self._settings)

        all_refs = []
        for ref in llm_refs:
            # An untrusted index is a positional guess, so the segment it points
            # at may belong to a neighbouring reference. Withhold it entirely:
            # every segment-anchored backfill downstream — issue recovery here,
            # author rescue below, DOI rescue inside
            # ``_finalize_reference_fields`` — would otherwise copy another
            # reference's printed values onto this one verbatim.
            segment = (
                ref_strings[ref.index - 1]
                if ref.index_trusted and 0 < ref.index <= len(ref_strings)
                else None
            )
            # The LLM sometimes returns the printed "55(7)" whole in one field.
            # The NER path splits that before the segment-anchored backfill;
            # this path only backfilled, so the combined form survived into
            # ``volume`` *and* defeated the backfill, which anchors its regex on
            # a clean volume.
            volume, issue = _normalize_vol_issue(ref.volume, ref.issue, segment or "")
            fields = _finalize_reference_fields(
                {
                    **ref.model_dump(),
                    "bib_id": ref.index,
                    "year": ref.year if ref.year else None,
                    "volume": volume,
                    "issue": issue,
                    "authors": ref.authors
                    or (
                        _rescue_authors_from_segment(segment)
                        if ref.index in rescue_author_indices
                        else None
                    ),
                    "editors": ref.editors or None,
                },
                segment,
            )
            all_refs.append(PaperReference.model_validate(fields))

        # Merge the LLM-parsed refs with any NER-recovered refs from hard-failed
        # batches, ordered by document position, then filter + sequence bib_ids.
        positioned: list[tuple[int, PaperReference]] = [
            (lref.index, built) for lref, built in zip(llm_refs, all_refs, strict=True)
        ]
        positioned.extend(ner_recovered)
        await self._recover_skipped_segments(positioned, ref_strings)
        ordered = [ref for _, ref in sorted(positioned, key=lambda item: item[0])]
        return _sequence_references(ordered)

    async def _recover_skipped_segments(
        self,
        positioned: list[tuple[int, PaperReference]],
        ref_strings: list[str],
    ) -> None:
        """Re-parse segments no batch ever returned a reference for, in place.

        A batch that raises is already routed to NER. A batch that *succeeds*
        while quietly returning 12 refs for 15 numbered entries was not: the
        three skipped entries were dropped with only a log warning, which reads
        downstream as a paper that simply printed fewer references and silently
        breaks every numbered inline citation past the gap. The prompt
        explicitly invites skipping entries it cannot parse, so this is the
        expected failure, not an exotic one.

        Recovery is the same NER parser the hard-failure path uses, applied
        only to the missing slots and tagged with their true document position.
        """
        # A ref the LLM returned with neither a title nor authors does not
        # count as covered: _sequence_references drops it further down, and it
        # would take the entry's slot with it — deleting one printed reference
        # and shifting every later bib_id, so [8]..[15] all resolve one row
        # off. Treat those slots as missing and re-parse them like any other.
        stubs = sorted({index for index, ref in positioned if _is_stub_reference(ref)})
        if not ref_strings:
            if stubs:
                self._record_warning(
                    WarningCode.REF_PARSE_LOST,
                    f"{len(stubs)} ref(s) parsed with no title and no authors were dropped; "
                    "no reference strings were available to re-parse them",
                )
            return
        covered = {index for index, ref in positioned if not _is_stub_reference(ref)}
        missing = [i for i in range(1, len(ref_strings) + 1) if i not in covered]
        if not missing:
            return
        try:
            aligned = await asyncio.to_thread(
                self._parse_references_ner_aligned, [ref_strings[i - 1] for i in missing]
            )
        except ProcessingError:
            raise
        except Exception as ner_exc:  # noqa: BLE001 — recovery is best-effort
            logger.error("NER recovery of %d skipped refs failed: %s", len(missing), ner_exc)
            self._record_warning(
                WarningCode.REF_PARSE_LOST,
                f"{len(missing)} ref(s) skipped by the LLM, 0 recovered "
                f"(NER failed: {ner_exc.__class__.__name__})",
            )
            return
        if len(aligned) != len(missing):
            # ``_parse_references_ner_aligned`` guarantees one slot per input,
            # so this is unreachable in production — but this whole path is
            # best-effort recovery, and raising here would turn a partial
            # result into a failed extraction.
            logger.error(
                "NER recovery returned %d slots for %d skipped refs; skipping recovery",
                len(aligned),
                len(missing),
            )
            return
        recovered = [
            (index, ref)
            for index, ref in zip(missing, aligned, strict=True)
            if ref and not _is_stub_reference(ref)
        ]
        # Drop the stub that occupied a slot NER has now filled, so the entry
        # is not represented twice on the way into _sequence_references.
        replaced = {index for index, _ in recovered}
        if replaced:
            positioned[:] = [
                item
                for item in positioned
                if not (item[0] in replaced and _is_stub_reference(item[1]))
            ]
        positioned.extend(recovered)
        self._record_warning(
            WarningCode.REF_PARSE_NER_FALLBACK,
            f"{len(recovered)}/{len(missing)} ref(s) skipped by the LLM recovered via NER",
        )

    async def _parse_references_llm_chunked(
        self, ref_text: str, ref_strings: list[str]
    ) -> list[PaperReference]:
        """Parse the reference list in region-aligned, chunk-tolerant LLM calls.

        Each chunk is a contiguous slice of ``ref_text`` (:func:`region_chunks`
        when the layout-region stream aligns, else a greedy grouping of the
        already-segmented ``ref_strings`` via :func:`_group_segments_into_chunks`).
        The chunk prompt finds its own reference boundaries rather than trusting
        a numbered contract, so upstream under/over-segmentation is non-critical
        on this path — unlike :meth:`_parse_references_llm`.

        Failure discipline mirrors the batched-LLM path: a degenerate failure
        (timeout / output-token-cap loop) is deterministic under greedy
        decoding, so it skips the retry and falls straight back to the NER
        parser on that chunk's members; a transient failure is retried once
        before falling back.
        """
        # Parse-chunk sourcing is a distinct axis from the seg-cascade tier:
        # region_chunks() here only shapes parse batches, so it runs regardless
        # of REF_SEG_REGION_ANCHORS (which gates the region *segmentation* tier).
        summaries = self._reference_region_summaries()
        chunks = region_chunks(ref_text, summaries)
        if chunks is None:
            chunks = _group_segments_into_chunks(ref_strings, _CHUNK_TARGET_CHARS)
        if not chunks:
            return []

        tasks = [
            self.llm_client.extract_references_chunk(chunk, file_hash=self.file_hash)
            for chunk in chunks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, ProcessingError):
                raise result

        ordered_refs: list[PaperReference] = []
        failures: list[BaseException] = []

        async def _fallback_to_ner(ci: int, cause: BaseException) -> list[PaperReference]:
            """Recover chunk ``ci``'s refs via the NER parser. Members are the
            already-segmented refs that fall inside the chunk, or — when the
            chunk was built from raw region text with no matching segment
            (rare) — a zero-LLM marker split, or the whole chunk as a single
            entry as the last resort."""
            if isinstance(cause, ProcessingError):
                raise cause
            failures.append(cause)
            chunk = chunks[ci]
            members = _chunk_ner_members(chunk, ref_strings)
            try:
                # Same offload as _parse_refs_via_ner: keep the CRF forward
                # pass off the shared serve event loop.
                recovered = await asyncio.to_thread(self._parse_references_ner, members)
            except ProcessingError:
                raise
            except Exception as ner_exc:  # noqa: BLE001
                logger.error(
                    "Reference parse chunk %d hard-failed and NER fallback also failed: %s",
                    ci,
                    ner_exc,
                )
                # Same contract as the batched path: a dropped chunk must be
                # visible in the export, not only in the log.
                self._record_warning(
                    WarningCode.REF_PARSE_LOST,
                    f"chunk {ci}, 0/{len(members)} refs recovered "
                    f"(NER fallback also failed: {ner_exc.__class__.__name__})",
                )
                return []
            self._record_warning(
                WarningCode.REF_PARSE_NER_FALLBACK,
                f"chunk {ci}, {len(recovered)}/{len(members)} refs recovered",
            )
            return recovered

        for ci, res in enumerate(results):
            if isinstance(res, BaseException):
                if _is_degenerate_ref_failure(res):
                    logger.warning(
                        "Reference parse chunk %d failed degenerately (%s); "
                        "skipping retry, falling back to NER",
                        ci,
                        res,
                    )
                    ordered_refs.extend(await _fallback_to_ner(ci, res))
                    continue
                logger.warning("Reference parse chunk %d failed: %s; retrying once", ci, res)
                try:
                    res = await self.llm_client.extract_references_chunk(
                        chunks[ci], file_hash=self.file_hash
                    )
                except asyncio.CancelledError:
                    # Cancellation is not a parse failure — propagate it so the
                    # cancelled coroutine actually stops, never divert to NER.
                    raise
                except ProcessingError:
                    raise
                except BaseException as retry_exc:  # noqa: BLE001
                    ordered_refs.extend(await _fallback_to_ner(ci, retry_exc))
                    continue

            chunk_refs = []
            for ref in res:
                fields = _finalize_reference_fields(
                    {
                        **ref.model_dump(),
                        "bib_id": ref.index,
                        "year": ref.year if ref.year else None,
                        "editors": ref.editors or None,
                    },
                    None,
                )
                chunk_refs.append(PaperReference.model_validate(fields))
            ordered_refs.extend(chunk_refs)

        if tasks and not ordered_refs:
            # Every chunk failed the LLM *and* its NER fallback — an empty
            # result here would be indistinguishable from a paper without
            # references, so fail loud rather than silently exporting 0 refs.
            cause = failures[0] if failures else None
            raise UpstreamServiceError(
                "LLM", f"All {len(tasks)} reference parse chunks failed", cause
            ) from cause

        return _sequence_references(ordered_refs)

    def _parse_references_ner(self, ref_strings: list[str]) -> list[PaperReference]:
        """Parse pre-segmented reference strings with the NER parser (batched).

        List-numbering markers are stripped from the parser input only; the
        verbatim segments are kept for issue backfill.
        """
        aligned = self._parse_references_ner_aligned(ref_strings)
        return _sequence_references([ref for ref in aligned if ref is not None])

    def _parse_references_ner_aligned(self, ref_strings: list[str]) -> list[PaperReference | None]:
        """Parse a batch while retaining one output slot per input citation.

        Production extraction filters empty slots through
        :meth:`_parse_references_ner`; evaluation uses the alignment to keep
        invalid outputs in metric denominators.
        """
        # Load + forward under the shared inference lock: concurrent native
        # model work from sibling post-parse threads intermittently
        # segfaulted the process on MPS (see bibr.utils.locks).
        with LOCAL_INFERENCE_LOCK:
            ref_parser = _get_ner_parser(self._settings)
            parsed = ref_parser.parse_batch(_strip_enum_markers(ref_strings))
        aligned: list[PaperReference | None] = []
        parsed_count = 0
        for ref_text, fields in zip(ref_strings, parsed, strict=True):
            title = fields.get("title") or ""
            authors = fields.get("authors")
            if not title and not authors:
                aligned.append(None)
                continue
            # Strip punctuation noise + split a combined "N(M)" before the
            # segment-anchored backfill (which needs a clean volume).
            volume, issue = _normalize_vol_issue(
                fields.get("volume"), fields.get("issue"), ref_text
            )
            ref_fields = _finalize_reference_fields(
                {
                    "bib_id": parsed_count + 1,
                    "title": title,
                    "authors": authors,
                    "container": fields.get("container"),
                    "year": fields.get("year"),
                    "volume": volume,
                    "issue": issue,
                    "first_page": fields.get("first_page"),
                    "last_page": fields.get("last_page"),
                    "doi": fields.get("doi"),
                    "url": fields.get("url"),
                    "publisher": fields.get("publisher"),
                    "editors": fields.get("editors"),
                    "edition": fields.get("edition"),
                    # Every field the decoder maps must be passed on here, or
                    # it is dropped before export (TestNerPathKeepsNerOnlyFields
                    # guards it). Verbatim: finalize leaves these five alone.
                    "arxiv": fields.get("arxiv"),
                    "pmid": fields.get("pmid"),
                    "series": fields.get("series"),
                    "access_date": fields.get("access_date"),
                    "note": fields.get("note"),
                    # The NER tag set has no in-press concept, so the tagger
                    # drops the year *and* leaves the flag false — an
                    # "(in press)" reference exported year: null,
                    # is_in_press: false under the default parse strategy and
                    # its in-text citation could never match. Read it off the
                    # segment, the way _backfill_vancouver_tail already does.
                    "is_in_press": _is_in_press(ref_text),
                },
                ref_text,
            )
            aligned.append(PaperReference.model_validate(ref_fields))
            parsed_count += 1
        return aligned

    # ------------------------------------------------------------------
    # Reference line stream (bibr.extract.ref_line_stream)
    # ------------------------------------------------------------------

    def _consider_line_stream(
        self,
        ref_df: pd.DataFrame,
        ref_text: str,
        ref_strings: list[str],
        seg_strategy: str,
    ) -> _StreamChoice | None:
        """Segment the section as one line stream; decide whether it replaces the cascade.

        Runs for the ``geom`` strategy on inputs with layout regions (PDFs)
        unless native reference strings were taken. The stream's attempt is
        always recorded. When it is not selected the cascade's segments stand
        exactly as they were; a DOI link annotation over a segment's own lines
        can still fill a segment that prints no DOI. Any error keeps the
        cascade's result.
        """
        if seg_strategy != "geom" or ref_df is None:
            return None
        if any(a.strategy == "native" and a.selected for a in self._segmentation_attempts):
            return None
        if not getattr(self.contents, "region_summaries", None):
            # No layout regions: a DOCX, JATS or HTML input, whose paragraphs
            # are the entries already.
            return None
        try:
            return self._line_stream_choice(ref_df, ref_text, ref_strings)
        except Exception as e:  # noqa: BLE001 — an alternative must never break extraction
            logger.warning("Reference line stream failed: %r", e)
            return None

    def _line_stream_choice(
        self, ref_df: pd.DataFrame, ref_text: str, ref_strings: list[str]
    ) -> _StreamChoice | None:
        stream = build_line_stream(self.contents, ref_df)
        if stream is None:
            return None
        probabilities = stream_probabilities(stream, self._geom_line_predictor())
        segmentation = segment_line_stream(stream, probabilities)
        if segmentation is None:
            return None
        entries, _numbered, entry_dois, split_count = self._post_process_stream_entries(
            segmentation
        )
        if not entries:
            return None

        typical = typical_entry_length(ref_strings, entries)
        section_text = segmentation.section_key
        cascade_quality = segmentation_quality(ref_strings, section_text, typical_length=typical)
        stream_quality = segmentation_quality(entries, section_text, typical_length=typical)
        tier, tier_spans = self._cascade_selected_tier()
        if ref_strings and len(entries) < _distinct_count(ref_strings):
            trigger, selected = "fewer_entries", False
        elif ref_strings and stream.rotated:
            trigger, selected = "rotated_page", False
        elif tier is None or tier in _STREAM_FALLBACK_TIERS or not ref_strings:
            trigger = "cascade_fallback"
            selected = stream_quality >= cascade_quality and stream_quality > 0
        elif len(ref_strings) > _STREAM_SPLIT_REPAIR_RATIO * tier_spans:
            # Most of the cascade's segments came from the merged-reference
            # splitter, not from the tier that was selected.
            trigger = "cascade_split_repaired"
            selected = stream_quality >= cascade_quality
        elif self._cascade_under_yield(ref_text, len(ref_strings)):
            trigger = "cascade_under_yield"
            selected = stream_quality >= cascade_quality
        else:
            trigger = "quality_margin"
            selected = stream_quality >= cascade_quality + _STREAM_OVERRIDE_MARGIN
        logger.info(
            "Reference line stream: %d entries, quality %.2f vs cascade %s %.2f (%s) -> %s",
            len(entries),
            stream_quality,
            tier,
            cascade_quality,
            trigger,
            "selected" if selected else "kept cascade",
        )
        spans = _locate_segment_spans(segmentation.text, entries)
        flags = (
            trigger,
            f"stream_quality_{stream_quality:.2f}",
            f"cascade_quality_{cascade_quality:.2f}",
            "stream_text_offsets",
            *segmentation.reason_flags,
        )
        if not selected:
            link_dois = link_dois_for_segments(ref_strings, stream.lines)
            self._record_segmentation_attempt(
                "line_stream",
                segmentation.text,
                spans=tuple(span for span in spans if span is not None),
                selected=False,
                reason_flags=flags,
            )
            return _StreamChoice(
                selected=False,
                ref_text=ref_text,
                ref_strings=ref_strings,
                spans=(),
                link_dois=link_dois,
            )
        superseded = [
            dataclasses.replace(
                attempt,
                selected=False,
                reason_flags=(*attempt.reason_flags, "superseded_by_line_stream"),
            )
            if attempt.selected
            else attempt
            for attempt in self._segmentation_attempts
        ]
        self._segmentation_attempts[:] = superseded
        if split_count:
            self._record_warning(
                WarningCode.REF_SEG_MERGE_SPLIT,
                f"split merged line-stream entries into {split_count} more segment(s)",
            )
        self._record_segmentation_attempt(
            "line_stream",
            segmentation.text,
            spans=tuple(span for span in spans if span is not None),
            credible_starts=len(entries),
            selected=True,
            reason_flags=flags,
        )
        return _StreamChoice(
            selected=True,
            ref_text=segmentation.text,
            ref_strings=entries,
            spans=spans,
            link_dois=entry_dois,
        )

    def _post_process_stream_entries(
        self, segmentation: StreamSegmentation
    ) -> tuple[list[str], list[bool], list[str | None], int]:
        """The junk filter and merge splitter the cascade's segments get.

        Entries opened by a verified numbering sequence skip the junk filter:
        a short numbered entry without a year ("[3] Inwent: Internationale
        Weiterbildung und Entwicklung") is still an entry. A split entry's
        link DOI is dropped, as no piece can claim it.
        """
        entries: list[str] = []
        protected: list[bool] = []
        dois: list[str | None] = []
        for entry, numbered, doi in zip(
            segmentation.entries, segmentation.numbered, segmentation.link_dois, strict=True
        ):
            if not numbered and is_non_reference_segment(entry):
                continue
            entries.append(entry)
            protected.append(numbered)
            dois.append(doi)
        if not self._settings.REF_SPLIT_MERGED_REFS or not entries:
            return entries, protected, dois, 0
        find_onsets = _onset_finder_for_bibliography(entries)
        split_entries: list[str] = []
        split_protected: list[bool] = []
        split_dois: list[str | None] = []
        added = 0
        for entry, numbered, doi in zip(entries, protected, dois, strict=True):
            try:
                offsets = find_onsets(entry)
            except Exception:  # noqa: BLE001 — never fail extraction over a split
                offsets = []
            bounds = [0, *offsets, len(entry)]
            pieces = [entry[a:b].strip() for a, b in zip(bounds, bounds[1:], strict=False)]
            if len(pieces) > 1 and all(pieces):
                split_entries.extend(pieces)
                split_protected.extend([numbered] + [False] * (len(pieces) - 1))
                split_dois.extend([None] * len(pieces))
                added += len(pieces) - 1
            else:
                split_entries.append(entry)
                split_protected.append(numbered)
                split_dois.append(doi)
        return split_entries, split_protected, split_dois, added

    def _cascade_selected_tier(self) -> tuple[str | None, int]:
        """Strategy and span count of the cascade's selected attempt (the last one)."""
        selected = [attempt for attempt in self._segmentation_attempts if attempt.selected]
        if not selected:
            return None, 0
        return selected[-1].strategy, len(selected[-1].spans)

    def _cascade_under_yield(self, ref_text: str, segment_count: int) -> bool:
        """Pre-parse form of the receipt's credible-start check.

        Counts entry starts (aligned layout onsets, printed markers, the
        tiers' own credible starts), not rows: a reference split into several
        sentence rows is still one entry.
        """
        credible = self._estimate_credible_source_starts(ref_text)
        return bool(
            credible is not None
            and credible >= _STREAM_MIN_EVIDENCE
            and segment_count < _STREAM_CREDIBLE_START_YIELD * credible
        )

    def _geom_line_predictor(self):
        """Lazy per-line start probabilities from the geom segmenter."""

        def predict(records: list[dict]) -> list[float]:
            with LOCAL_INFERENCE_LOCK:
                segmenter = _get_geom_segmenter(self._settings)
                if segmenter is None:
                    return []
                return list(segmenter.line_start_probabilities(records))

        return predict

    def _parse_references_ner_with_links(
        self, ref_strings: list[str], link_dois: list[str | None]
    ) -> list[PaperReference]:
        """NER parse, then fill a DOI from a link annotation where none was parsed."""
        aligned = self._parse_references_ner_aligned(ref_strings)
        if len(link_dois) == len(aligned):
            filled = 0
            for ref, doi in zip(aligned, link_dois, strict=True):
                if ref is not None and doi and not ref.doi:
                    ref.doi = doi
                    filled += 1
            if filled:
                self._stream_reason_flags.add("doi_from_link_annotation")
                logger.info("Filled %d reference DOI(s) from link annotations", filled)
        return _sequence_references([ref for ref in aligned if ref is not None])
