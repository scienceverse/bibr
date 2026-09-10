"""Source-provenance DOI candidate collection and deterministic selection."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import replace
from difflib import SequenceMatcher

from bibr.input.consolidate_text import fix_ocr_artifacts
from bibr.paper_contents import CanonicalSection
from bibr.pipeline.identity import DoiCandidate, DoiSelection, ExpectedIdentity
from bibr.utils.text import normalize_doi
from bibr.validation import IssueSeverity, ValidationIssue

EXPECTED_VISIBLE = 4
EXPLICIT_SELF_ID = 3
FRONT_MATTER_OR_REPEATED_FURNITURE = 2
UNCONTESTED_UNTYPED = 1

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
_REFERENCE_PREFIX_RE = re.compile(r"^\s*(?:\[\d+[A-Za-z]?\]|\d+[.)]\s)")
_COMPONENT_SUFFIX_RE = re.compile(r"\.(?:g|f|fig|t|table)\d+[A-Za-z]*$", re.IGNORECASE)
_NON_SELF_MARKERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"reference\s+doi\s*[:.]?\s*$", re.IGNORECASE), "reference_doi"),
    (re.compile(r"parent(?:\s+article)?\s+doi\s*[:.]?\s*$", re.IGNORECASE), "parent_doi"),
    (
        re.compile(
            r"(?:component|supplement(?:ary)?|figure|table)\s+doi\s*[:.]?\s*$",
            re.IGNORECASE,
        ),
        "component_doi",
    ),
    (
        re.compile(
            r"(?:data(?:set)?|code|software|repository|archive|materials?)\s+doi\s*[:.]?\s*$",
            re.IGNORECASE,
        ),
        "data_doi",
    ),
)
_SELF_MARKERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"article\s+doi\s*[:.]?\s*$", re.IGNORECASE), "article_doi"),
    (re.compile(r"please\s+cite\s+as\s*[:.]?\s*$", re.IGNORECASE), "please_cite_as"),
    (
        re.compile(r"abstract\s+citation\s+(?:id|doi)\s*[:.]?\s*$", re.IGNORECASE),
        "abstract_citation_id",
    ),
    (re.compile(r"\bcitation\s*[:.]?\s*$", re.IGNORECASE), "citation"),
    (
        re.compile(r"(?<!journal\s)\b(?:paper\s+)?doi\s*[:.]?\s*$", re.IGNORECASE),
        "explicit_doi",
    ),
)
_REPOSITORY_NAME_RE = re.compile(
    r"\b(?:osf|zenodo|dryad|mendeley(?:\s+data)?|figshare)\b",
    re.IGNORECASE,
)
_FUNDER_REGISTRY_PREFIX = "10.13039/"
_REPOSITORY_CONTEXT_RE = re.compile(
    r"\b(?:data|datasets?|code|software|materials?)\s+(?:availability|repository)\b"
    r"|\b(?:data|datasets?|code|software|materials?)\s+(?:are|is)\s+"
    r"(?:available|deposited|archived|hosted)\b"
    r"|\brepository\s+(?:doi|record|link|url)\s*[:.]?",
    re.IGNORECASE,
)


def _canonical_doi(value: str | None) -> str | None:
    normalized = normalize_doi(value)
    return normalized.casefold() if normalized else None


def doi_sha256(value: str) -> str:
    return hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()


# Publishers print the resolver host with or without ``www.``; without it the
# label in front ("Journal DOI: www.doi.org/…") is left stranded on the prefix
# and every marker pattern below misses.
_DOI_HOST = r"(?:https?://)?(?:www\.)?(?:dx\.)?doi\.org/"


def _marker_kind(text: str, start: int) -> str:
    prefix = text[max(0, start - 100) : start]
    if re.search(rf"journal\s+doi\s*[:.]?\s*{_DOI_HOST}\s*$", prefix, re.IGNORECASE):
        return "journal_doi"
    marker_prefix = re.sub(rf"{_DOI_HOST}\s*$", "", prefix)
    for pattern, kind in _NON_SELF_MARKERS:
        if pattern.search(marker_prefix):
            return kind
    for pattern, kind in _SELF_MARKERS:
        if pattern.search(marker_prefix):
            return kind
    if re.search(rf"{_DOI_HOST}\s*$", prefix, re.IGNORECASE):
        return "doi_url"
    return "bare"


def _candidate_from_match(
    text: str,
    match: re.Match[str],
    *,
    raw: str,
    source_kind: str,
    page: int | None,
    section_id: int | None,
    section_type: str | None,
    region_index: int | None,
    region_type: str | None,
    text_id: int | None,
    repeated_count: int = 0,
) -> DoiCandidate | None:
    normalized = _canonical_doi(match.group(0))
    if normalized is None:
        return None
    marker_kind = _marker_kind(text, match.start())
    lowered = text.casefold()
    section_value = str(section_type or "").casefold()

    rejection_reason = None
    if (
        section_value == CanonicalSection.REFERENCES.value
        or _REFERENCE_PREFIX_RE.match(text)
        or marker_kind == "reference_doi"
    ):
        semantic_context = "reference"
        rejection_reason = "reference_candidate"
        tier = 0
    elif marker_kind in {"parent_doi", "component_doi"}:
        semantic_context = "parent_or_component"
        rejection_reason = "component_candidate"
        tier = 0
    elif marker_kind == "data_doi":
        semantic_context = "data_or_code"
        rejection_reason = "data_or_code_candidate"
        tier = 0
    elif normalized.startswith(_FUNDER_REGISTRY_PREFIX):
        # 10.13039 is Crossref's Funder Registry registrant: these identify a
        # funding body, never an article, and are printed bare inside funding
        # statements where they otherwise land in the front-matter tier.
        semantic_context = "funder_registry"
        rejection_reason = "funder_registry_candidate"
        tier = 0
    elif marker_kind in {"article_doi", "please_cite_as", "abstract_citation_id"}:
        semantic_context = "article_self"
        tier = EXPLICIT_SELF_ID
    elif (
        section_value == CanonicalSection.OPEN_DATA.value
        or _REPOSITORY_NAME_RE.search(text)
        or _REPOSITORY_CONTEXT_RE.search(text)
        or normalized.casefold().startswith("10.5281/zenodo.")
    ):
        semantic_context = "data_or_code"
        rejection_reason = "data_or_code_candidate"
        tier = 0
    elif (
        section_value in {CanonicalSection.FIGURE.value, CanonicalSection.TABLE.value}
        or _COMPONENT_SUFFIX_RE.search(normalized)
        or re.search(r"\b(?:fig(?:ure)?|table|component|supplement)\s*\d+", lowered)
    ):
        semantic_context = "parent_or_component"
        rejection_reason = "component_candidate"
        tier = 0
    elif marker_kind == "explicit_doi":
        semantic_context = "article_self"
        tier = EXPLICIT_SELF_ID
    elif marker_kind == "citation":
        semantic_context = "article_self"
        tier = FRONT_MATTER_OR_REPEATED_FURNITURE
    elif source_kind in {"header", "footer"}:
        semantic_context = "repeated_furniture" if repeated_count > 1 else "structural_furniture"
        tier = FRONT_MATTER_OR_REPEATED_FURNITURE
    elif section_value in {
        CanonicalSection.TITLE.value,
        CanonicalSection.ABSTRACT.value,
        CanonicalSection.KEYWORDS.value,
    } or (page is not None and page <= 2):
        semantic_context = "front_matter"
        tier = FRONT_MATTER_OR_REPEATED_FURNITURE
    else:
        semantic_context = "untyped"
        tier = UNCONTESTED_UNTYPED

    if marker_kind == "journal_doi" and rejection_reason is None:
        semantic_context = "journal_identity"
        tier = UNCONTESTED_UNTYPED

    return DoiCandidate(
        raw=raw,
        normalized=normalized,
        source_kind=source_kind,
        page=page,
        section_id=section_id,
        section_type=section_type,
        region_index=region_index,
        region_type=region_type,
        text_id=text_id,
        marker_kind=marker_kind,
        repeated_header_footer_count=repeated_count,
        semantic_context=semantic_context,
        selection_tier=tier,
        rejection_reason=rejection_reason,
    )


# OCR sometimes wraps a DOI between the ``10.`` and its registrant digits
# ("https://doi.org/10. 1016/j.lanwpc.2023. 100933"), which every bridge in
# ``consolidate_text`` misses because they all require an intact ``10.\d{4,9}/``.
# Close that one gap only when an explicit DOI marker sits immediately in front
# of it and real registrant digits plus the path slash follow, so ordinary prose
# containing "10." can never be joined into a fabricated identifier.
_DOI_REGISTRANT_WRAP_RE = re.compile(
    rf"((?:{_DOI_HOST}|\bdoi\s*[:.]?[ \t]*)10\.)[ \t]*\r?\n?[ \t]*(?=\d{{4,9}}/)",
    re.IGNORECASE,
)
_BARE_DOI_REGISTRANT_WRAP_RE = re.compile(r"(?<!\S)(10\.)[ \t]+(?=\d{4,9}/[-._;()/:A-Za-z0-9]+)")


def _repair_doi_text(text: str, *, furniture: bool = False) -> str:
    # Bridge the registrant wrap first: the downstream ``consolidate_text``
    # bridges (dot wrap, slash-space) only engage on an intact registrant.
    bridged = _DOI_REGISTRANT_WRAP_RE.sub(r"\1", text) if "10." in text else text
    # Layout furniture supplies the missing DOI-label evidence for a bare
    # header identifier. Never apply this unmarked repair to ordinary prose.
    if furniture:
        bridged = _BARE_DOI_REGISTRANT_WRAP_RE.sub(r"\1", bridged)
    cleaned = fix_ocr_artifacts(bridged)
    return re.sub(
        r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]*-)\s+"
        r"([-._;()/:A-Za-z0-9]*[\d.][-._;()/:A-Za-z0-9]*)",
        r"\1\2",
        cleaned,
    )


def normalize_candidate_doi(raw: str) -> str | None:
    """Normalize repaired DOI spelling while retaining source case for legacy callers."""

    repaired = _repair_doi_text(raw)
    match = _DOI_RE.search(repaired)
    return normalize_doi(match.group(0)) if match is not None else None


def _cleaned_char_source_ranges(source: str, cleaned: str) -> list[tuple[int, int]]:
    """Map each cleaned character to its source interval for receipt provenance."""

    ranges: list[tuple[int, int]] = [(0, 0)] * len(cleaned)
    for tag, source_start, source_end, clean_start, clean_end in SequenceMatcher(
        None, source, cleaned, autojunk=False
    ).get_opcodes():
        clean_length = clean_end - clean_start
        source_length = source_end - source_start
        if tag == "equal":
            for offset in range(clean_length):
                source_index = source_start + offset
                ranges[clean_start + offset] = (source_index, source_index + 1)
        elif clean_length:
            for offset in range(clean_length):
                interval_start = source_start + (offset * source_length) // clean_length
                interval_end = (
                    source_start + ((offset + 1) * source_length + clean_length - 1) // clean_length
                )
                ranges[clean_start + offset] = (interval_start, interval_end)
    return ranges


def _raw_source_match(
    source: str,
    match: re.Match[str],
    source_ranges: list[tuple[int, int]],
) -> str:
    if match.start() >= len(source_ranges) or match.end() <= match.start():
        return match.group(0)
    start = source_ranges[match.start()][0]
    end = source_ranges[match.end() - 1][1]
    if end <= start:
        return match.group(0)
    return source[start:end]


def _candidates_from_text(text: str, **provenance) -> list[DoiCandidate]:
    source = text or ""
    furniture = provenance.get("source_kind") in {"header", "footer"} or provenance.get(
        "region_type"
    ) in {"header", "footer"}
    cleaned = _repair_doi_text(source, furniture=furniture)
    matches = tuple(_DOI_RE.finditer(cleaned))
    if not matches:
        return []
    source_ranges = _cleaned_char_source_ranges(source, cleaned)
    return [
        candidate
        for match in matches
        if (
            candidate := _candidate_from_match(
                cleaned,
                match,
                raw=_raw_source_match(source, match, source_ranges),
                **provenance,
            )
        )
        is not None
    ]


def collect_doi_candidates(contents) -> tuple[DoiCandidate, ...]:
    """Collect every source-visible DOI with sentence or furniture provenance."""

    section_map = {section.section_id: section for section in contents.sections}
    candidates: list[DoiCandidate] = []
    for sentence in contents.sentences:
        section = section_map.get(sentence.section_id)
        region_meta = sentence.region_meta or {}
        candidates.extend(
            _candidates_from_text(
                sentence.text,
                source_kind="sentence",
                page=sentence.page_number,
                section_id=sentence.section_id,
                section_type=section.section_type.value
                if section and section.section_type
                else None,
                region_index=region_meta.get("region_index"),
                region_type=region_meta.get("region_type"),
                text_id=sentence.text_id,
            )
        )

    furniture = [("header", line) for line in contents.detected_headers] + [
        ("footer", line) for line in contents.detected_footers
    ]
    counts = Counter(line.strip().casefold() for _, line in furniture)
    for source_kind, line in furniture:
        candidates.extend(
            _candidates_from_text(
                line,
                source_kind=source_kind,
                page=None,
                section_id=None,
                section_type=None,
                region_index=None,
                region_type=source_kind,
                text_id=None,
                repeated_count=counts[line.strip().casefold()],
            )
        )

    preparsed = getattr(contents, "preparsed_metadata", None)
    structured_doi = _canonical_doi(getattr(preparsed, "doi", None))
    if structured_doi:
        candidates.append(
            DoiCandidate(
                raw=str(preparsed.doi),
                normalized=structured_doi,
                source_kind="structured_metadata",
                page=None,
                section_id=None,
                section_type=None,
                region_index=None,
                region_type=None,
                text_id=None,
                marker_kind="structured_doi",
                repeated_header_footer_count=0,
                semantic_context="article_self",
                selection_tier=EXPLICIT_SELF_ID,
            )
        )
    return tuple(candidates)


_FURNITURE_SOURCES = frozenset({"header", "footer"})
_NUMERIC_EXTENSION_RE = re.compile(r"^[/.]\d")


def _distinct_dois(candidates: list[DoiCandidate]) -> set[str]:
    return {candidate.normalized.casefold() for candidate in candidates}


def _drop_truncated_prefixes(candidates: list[DoiCandidate]) -> list[DoiCandidate]:
    """Drop a DOI another candidate strictly extends at a ``/`` or ``.`` boundary.

    Only a numeric extension counts: mastheads print journal-level stems that
    the article DOI extends with an issue or article number
    (``10.30574/wjarr`` → ``10.30574/wjarr.2022.14.3.0574``), while supplement
    and component spellings extend with a letter (``/s1``, ``.s001``, ``.g001``)
    and must never displace the article they belong to.
    """

    values = _distinct_dois(candidates)
    truncated = {
        value
        for value in values
        for other in values
        if other != value
        and other.startswith(value)
        and _NUMERIC_EXTENSION_RE.match(other[len(value) :])
    }
    kept = [c for c in candidates if c.normalized.casefold() not in truncated]
    return kept or candidates


def _prefer_marked(candidates: list[DoiCandidate]) -> list[DoiCandidate]:
    """Prefer a labelled or resolver-URL DOI over one printed with no marker."""

    marked = [c for c in candidates if c.marker_kind != "bare"]
    return marked or candidates


def _prefer_body_sources(candidates: list[DoiCandidate]) -> list[DoiCandidate]:
    """Prefer a DOI read from the page body over running header/footer furniture."""

    body = [c for c in candidates if c.source_kind not in _FURNITURE_SOURCES]
    return body or candidates


def _prefer_lowest_page(candidates: list[DoiCandidate]) -> list[DoiCandidate]:
    """Prefer the earliest page; the paper's own DOI is printed in front matter."""

    if any(c.page is None for c in candidates):
        return candidates
    lowest = min(c.page for c in candidates)  # type: ignore[type-var]
    return [c for c in candidates if c.page == lowest]


_TIE_BREAK_LADDER = (
    _drop_truncated_prefixes,
    _prefer_marked,
    _prefer_body_sources,
    _prefer_lowest_page,
)


def _select_without_expected(candidates: tuple[DoiCandidate, ...]) -> DoiSelection:
    eligible = [candidate for candidate in candidates if candidate.rejection_reason is None]
    if not eligible:
        return DoiSelection(None, candidates, ())
    highest_tier = max(candidate.selection_tier for candidate in eligible)
    highest = [candidate for candidate in eligible if candidate.selection_tier == highest_tier]
    if len(_distinct_dois(highest)) > 1:
        evidence = tuple(
            f"text:{candidate.text_id}" if candidate.text_id is not None else candidate.source_kind
            for candidate in highest
        )
        # A tier tie is real ambiguity and is always reported, but one spurious
        # co-tier candidate must not suppress a correctly read DOI: walk a
        # deterministic provenance ladder and take the winner only if the tie
        # collapses to a single DOI. Otherwise abstain, as before.
        resolved = highest
        for rule in _TIE_BREAK_LADDER:
            if len(_distinct_dois(resolved)) == 1:
                break
            resolved = rule(resolved)
        selected = resolved[0] if len(_distinct_dois(resolved)) == 1 else None
        message = f"Conflicting source-visible DOI candidates at tier {highest_tier}"
        if selected is not None:
            message += f", resolved by source provenance to {selected.normalized}"
        issue = ValidationIssue(
            code="VAL_DOI_AMBIGUOUS",
            severity=IssueSeverity.WARNING,
            message=message,
            origin_stage="identity",
            evidence_ids=evidence,
        )
        return DoiSelection(selected, candidates, (issue,))
    return DoiSelection(highest[0], candidates, ())


def select_doi_candidates(
    candidates, expected_identity: ExpectedIdentity | None = None
) -> DoiSelection:
    """Select by explicit evidence tiers, then by source provenance within a tier.

    An equal-tier conflict always raises ``VAL_DOI_AMBIGUOUS``; it resolves to a
    candidate only when the provenance ladder in ``_select_without_expected``
    collapses it to a single DOI, and abstains otherwise.
    """

    candidate_tuple = tuple(candidates)
    if expected_identity is None or not (
        expected_identity.expected_doi or expected_identity.expected_doi_sha256
    ):
        selection = _select_without_expected(candidate_tuple)
        if expected_identity and expected_identity.doi_required and selection.selected is None:
            issue = ValidationIssue(
                code="VAL_EXPECTED_ID_MISSING",
                severity=IssueSeverity.ERROR,
                message="A DOI is required but no eligible source-visible DOI candidate was found",
                origin_stage="identity",
                blocking=True,
                evidence_ids=(expected_identity.queue_record_id,),
            )
            return replace(selection, issues=(*selection.issues, issue))
        return selection

    expected_doi = _canonical_doi(expected_identity.expected_doi)
    expected_hash = (
        expected_identity.expected_doi_sha256.casefold()
        if expected_identity.expected_doi_sha256
        else None
    )
    matching_indexes = [
        index
        for index, candidate in enumerate(candidate_tuple)
        if candidate.rejection_reason is None
        and (
            (
                expected_doi is not None
                and candidate.normalized.casefold() == expected_doi.casefold()
            )
            or (expected_hash is not None and doi_sha256(candidate.normalized) == expected_hash)
        )
    ]
    if matching_indexes:
        boosted = tuple(
            replace(candidate, selection_tier=EXPECTED_VISIBLE)
            if index in matching_indexes
            else candidate
            for index, candidate in enumerate(candidate_tuple)
        )
        return DoiSelection(boosted[matching_indexes[0]], boosted, ())

    fallback = _select_without_expected(candidate_tuple)
    if fallback.selected is None:
        issue = ValidationIssue(
            code="VAL_EXPECTED_ID_MISSING",
            severity=IssueSeverity.ERROR,
            message="Expected DOI is not present in eligible source-visible evidence",
            origin_stage="identity",
            evidence_ids=(expected_identity.queue_record_id,),
            blocking=True,
        )
    else:
        issue = ValidationIssue(
            code="VAL_EXPECTED_ID_MISMATCH",
            severity=IssueSeverity.ERROR,
            message=(
                f"Source-selected DOI {fallback.selected.normalized!r} does not match expected identity"
            ),
            origin_stage="identity",
            evidence_ids=(expected_identity.queue_record_id,),
            blocking=True,
        )
    return replace(fallback, issues=(*fallback.issues, issue))


def select_doi_from_text(text: str) -> DoiSelection:
    candidates: list[DoiCandidate] = []
    # OCR cleanup can legitimately join a line-ending DOI with the next
    # doi.org URL. Restore only that unmistakable identifier boundary so the
    # following article-self candidate keeps its own context.
    prepared = re.sub(
        r"(?<=[-._;()/:A-Za-z0-9])(?=https?://(?:www\.)?(?:dx\.)?doi\.org/)",
        "\n",
        text or "",
        flags=re.IGNORECASE,
    )
    for line_number, line in enumerate(prepared.splitlines(), start=1):
        candidates.extend(
            _candidates_from_text(
                line,
                source_kind="text",
                page=None,
                section_id=None,
                section_type=None,
                region_index=None,
                region_type=None,
                text_id=line_number,
            )
        )
    return _select_without_expected(tuple(candidates))
