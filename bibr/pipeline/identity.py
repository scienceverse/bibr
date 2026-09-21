"""Caller-provided and source-extracted paper identity records."""

from __future__ import annotations

from dataclasses import dataclass

from bibr.validation import ValidationIssue


@dataclass(frozen=True)
class ExpectedIdentity:
    queue_record_id: str
    expected_doi: str | None = None
    expected_doi_sha256: str | None = None
    expected_title: str | None = None
    target_block_hint: dict[str, object] | None = None
    source_sha256: str | None = None
    doi_required: bool = False


@dataclass(frozen=True)
class DoiCandidate:
    raw: str
    normalized: str
    source_kind: str
    page: int | None
    section_id: int | None
    section_type: str | None
    # With ``page``, the RegionSummary ``(page, index)`` of the layout region
    # the DOI was read from; the contract is documented on DoiCandidateExport.
    region_index: int | None
    region_type: str | None
    text_id: int | None
    marker_kind: str
    repeated_header_footer_count: int
    semantic_context: str
    selection_tier: int
    rejection_reason: str | None = None


@dataclass(frozen=True)
class DoiSelection:
    selected: DoiCandidate | None
    candidates: tuple[DoiCandidate, ...]
    issues: tuple[ValidationIssue, ...]
