"""
Paper

Thin facade — data models live in ``bibr.models``, export logic in
``bibr.export``.  This module re-exports everything for backward
compatibility and keeps the ``Paper`` class with its processing methods.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from bibr.input.file import InputFile
from bibr.models import (
    BibAuthor,
    BibType,
    ErrorCode,
    ExternalMatch,
    MatchSource,
    PaperAuthor,
    PaperMetadata,
    PaperReference,
    ProcessingStatus,
    format_bib_authors,
    migrate_bib_type,
)
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection
from bibr.processing_warnings import ProcessingWarning
from bibr.validation import ValidationIssue, references_incomplete_issue

if TYPE_CHECKING:
    from bibr.pipeline.identity import DoiSelection, ExpectedIdentity

# Re-export data classes so that ``from bibr.paper import PaperAuthor`` still works.
__all__ = [
    "BibAuthor",
    "BibType",
    "ErrorCode",
    "ExternalMatch",
    "MatchSource",
    "OcrFallbackMetadata",
    "Paper",
    "PaperAuthor",
    "PaperMetadata",
    "PaperReference",
    "ProcessingStatus",
    "enforce_imrad_order",
    "enforce_section_sanity",
    "format_bib_authors",
    "migrate_bib_type",
    "_merge_ocr_metadata",
]

logger = logging.getLogger(__name__)


class OcrFallbackMetadata(BaseModel):
    """Typed schema for OCR-extracted metadata used as fallback.

    Use ``OcrFallbackMetadata.from_raw(dict)`` to parse raw OCR output.
    """

    title: str | None = None
    doi: str | None = None
    keywords: list[str] = []
    authors: list[str] = []

    @classmethod
    def from_raw(cls, raw: dict) -> "OcrFallbackMetadata":
        """Parse a raw OCR metadata dict, coercing invalid types."""
        title = raw.get("title")
        if title is not None:
            title = str(title).strip() or None

        doi = raw.get("doi")
        if doi is not None:
            doi = str(doi).strip() or None

        keywords = raw.get("keywords", [])
        if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
            keywords = []

        authors = raw.get("authors", [])
        if not isinstance(authors, list):
            authors = []

        return cls(title=title, doi=doi, keywords=keywords, authors=authors)


# Canonical section types that should appear at most once in a paper.
_UNIQUE_SECTION_TYPES: set[CanonicalSection] = {
    CanonicalSection.ABSTRACT,
    CanonicalSection.REFERENCES,
}

# Trust ranking of classification_source tiers, used to pick the duplicate
# that keeps its type. Unlisted sources (and None) rank lowest.
_SOURCE_TRUST: dict[str, int] = {
    "title": 7,
    "exact_alias": 6,
    "substring_alias": 5,
    "model": 4,
    "llm": 3,
    "alias_prior": 2,
    "parent_context": 1,
}


def enforce_imrad_order(sections: list[PaperSection]) -> None:
    """Reset duplicate section classifications to UNKNOWN.

    Resets duplicates of types that should appear at most once (Abstract,
    References). Among duplicates the section with the highest-trust
    ``classification_source`` keeps its type (document order breaks ties) —
    a corrupted-header section that classified first must not outrank a
    later clean exact-alias heading. Sections like Methods, Results, and
    Discussion are allowed to repeat (subsections with those labels are
    common) and sections are allowed in any order — many real papers deviate
    from strict IMRaD.

    Dedup is deliberately global, not per study scope: the unique types
    (ABSTRACT, REFERENCES) have no legitimate per-study repeats, and scoping
    them lets a mid-study "Summary" header survive as a second ABSTRACT that
    pollutes the exported abstract.

    Demoted sections get ``classification_source="imrad_dedup"`` so the
    export records why the type was dropped. Mutates sections in-place.
    """
    duplicates: dict[CanonicalSection, list[int]] = {}
    for i, section in enumerate(sections):
        if section.section_type in _UNIQUE_SECTION_TYPES:
            duplicates.setdefault(section.section_type, []).append(i)

    for indices in duplicates.values():
        if len(indices) < 2:
            continue
        winner = max(
            indices,
            key=lambda i: (_SOURCE_TRUST.get(sections[i].classification_source or "", 0), -i),
        )
        for i in indices:
            if i == winner:
                continue
            section = sections[i]
            logger.debug(
                f"IMRaD dedup: resetting '{section.header}' from "
                f"{section.section_type.value} to UNKNOWN (duplicate)"
            )
            section.section_type = CanonicalSection.UNKNOWN
            section.classification_score = 0.0
            section.classification_source = "imrad_dedup"


# Core IMRaD body types used by the positional sanity pass.
_CORE_BODY_TYPES: set[CanonicalSection] = {
    CanonicalSection.INTRODUCTION,
    CanonicalSection.METHODS,
    CanonicalSection.RESULTS,
    CanonicalSection.DISCUSSION,
}


def enforce_section_sanity(sections: list[PaperSection]) -> None:
    """Demote positionally implausible unique-type classifications to UNKNOWN.

    Deliberately conservative — only two flagrant cases:
    - ABSTRACT appearing after the paper's first METHODS/RESULTS section
      (abstracts are front matter; a late "Summary" is discussion-flavored
      and would pollute the exported abstract fallback).
    - REFERENCES in the first half of the section list with core IMRaD body
      sections still to come (reference lists end the body; appendices and
      floats after a terminal reference list are fine and stay untouched).

    Positions are read from the list order, which must be document order
    (``implicit_sections`` re-sorts the list after it synthesizes sections
    from the LLM's boundaries). The root and the synthetic
    figure/table/footnote sections that ``create_content_sections`` appends at
    the tail are not body sections: they take no position, so a paper's float
    count cannot move the "first half".

    Runs after ``enforce_imrad_order``. Mutates sections in-place.
    """
    sections = [s for s in sections if s.level > 0 and not s.synthetic_kind]
    if not sections:
        return

    def _demote(section: PaperSection, reason: str) -> None:
        logger.debug(
            f"Section sanity: resetting '{section.header}' from "
            f"{section.section_type.value} to UNKNOWN ({reason})"
        )
        section.section_type = CanonicalSection.UNKNOWN
        section.classification_score = 0.0

    first_body_idx = next(
        (
            i
            for i, s in enumerate(sections)
            if s.section_type in (CanonicalSection.METHODS, CanonicalSection.RESULTS)
        ),
        None,
    )
    n = len(sections)
    for i, section in enumerate(sections):
        if (
            section.section_type == CanonicalSection.ABSTRACT
            and first_body_idx is not None
            and i > first_body_idx
        ):
            _demote(section, "abstract after body start")
        elif (
            section.section_type == CanonicalSection.REFERENCES
            and i < n / 2
            and any(s.section_type in _CORE_BODY_TYPES for s in sections[i + 1 :])
        ):
            _demote(section, "early references with body after")


def _merge_ocr_metadata(metadata: PaperMetadata, ocr: dict) -> None:
    """Merge OCR-extracted metadata as fallback into PaperMetadata.

    Only fills fields that are empty/missing in the LLM-extracted metadata.
    Mutates *metadata* in-place.
    """
    parsed = OcrFallbackMetadata.from_raw(ocr)

    if not metadata.title and parsed.title:
        metadata.title = parsed.title
        logger.debug("OCR fallback: filled title")

    if not metadata.doi and parsed.doi and parsed.doi.startswith("10."):
        metadata.doi = parsed.doi
        logger.debug("OCR fallback: filled DOI")

    if not metadata.keywords and parsed.keywords:
        metadata.keywords = parsed.keywords
        logger.debug("OCR fallback: filled keywords")

    if not metadata.authors and parsed.authors:
        author_id = 0
        for name in parsed.authors:
            if isinstance(name, str) and name.strip():
                stripped = name.strip()
                if "," in stripped:
                    # "Family, Given" — PDF docinfo and BibTeX both use it.
                    # Splitting on the last space instead produced
                    # given="Smith," / family="John", inverting every name.
                    family_part, _, given_part = stripped.partition(",")
                    family = family_part.strip()
                    given = given_part.strip()
                else:
                    parts = stripped.rsplit(" ", 1)
                    given = parts[0] if len(parts) > 1 else ""
                    family = parts[-1]
                if not family.strip():
                    continue
                author_id += 1
                metadata.authors.append(
                    PaperAuthor(
                        author_id=author_id,
                        given=given,
                        family=family,
                        affiliation="",
                    )
                )
        if metadata.authors:
            logger.debug("OCR fallback: filled %d authors", len(metadata.authors))


@dataclass
class Paper:
    # raw content
    input_file: InputFile
    # parsed content
    metadata: PaperMetadata | None = None  # we have to generate them
    contents: PaperContents | None = None
    processing_status: ProcessingStatus = field(default_factory=ProcessingStatus)
    paper_id: str | None = None  # user-supplied ID
    # Non-fatal warnings collected during processing (Crossref timeouts,
    # per-page OCR failures, etc.) — surfaced in the JSON export so consumers
    # can detect partial failures programmatically.
    processing_warnings: list[ProcessingWarning] = field(default_factory=list)
    # Per-paper LLM token usage keyed by the ``(label, provider, model)``
    # triple (e.g. ``("extract_authors", "google", "gemini-flash-lite")``),
    # attached by post_parse. One row per engine that ran a label — the source
    # ``extraction.usage`` aggregates. Empty when no LLM ran or usage tracking
    # is disabled. (A by-model-only ``llm_usage`` sibling existed until v11;
    # its sole reader was the removed root ``llm_usage`` export key.)
    llm_usage_labels: dict[tuple[str, str | None, str | None], dict[str, int]] = field(
        default_factory=dict
    )
    # Opt-in LLM trace rows (LLM_CAPTURE_TRACE): each call's rendered prompt
    # and raw response, attached by post_parse alongside llm_usage_labels.
    # Empty when capture is off (default) or no LLM ran — export renders an
    # empty list to `None` so `extraction.trace` is omitted entirely, per the
    # absence rule.
    llm_trace: list[dict] = field(default_factory=list)
    # Deployment-qualification provenance (identity SHAs, protocol hashes,
    # native-validity + fallback outcome, request counts). One object per paper,
    # attached by post_parse. None when no LLM ran or usage tracking is disabled.
    qualification_provenance: dict | None = None
    # Report-only parse-quality score in [0, 1] (10th-percentile aggregation of
    # per-region garbage/fragmentation ratings), attached by post_parse. None
    # when scoring is disabled or nothing was scoreable (e.g. DOCX-native).
    text_quality: float | None = None
    # Extraction provenance (bibr version, resolved reference strategies,
    # seg-fallback, enrichment flags, stage timings). Built by ``ExportStage``
    # from the pipeline context; ``None`` when a Paper is exported outside the
    # pipeline (e.g. hand-built in tests).
    extraction: dict | None = None
    # Typed semantic findings emitted while source provenance still exists.
    # Export merges these with payload-replay validation issues.
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    # Manifest caller evidence remains distinct from ``paper_id`` and the
    # source-selected scalar DOI. Both are serialized only in the additive
    # extraction receipt.
    expected_identity: "ExpectedIdentity | None" = None
    doi_selection: "DoiSelection | None" = None
    # Runtime-only: the in-flight enrichment prefetch task started by
    # post_parse as soon as references were parsed
    # (``bibr.pipeline.enrich_prefetch.EnrichmentPrefetchHandle``). Consumed by
    # ``CrossrefEnricher``, cancelled by every path that skips enrichment.
    # Never serialized — the export layer does not read it.
    enrichment_prefetch: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        internal_issues = (
            self.contents.structure_validation_issues if self.contents is not None else []
        )
        deduplicated: list[ValidationIssue] = []
        issue_keys: set[tuple[str, str, tuple[str, ...]]] = set()
        for issue in [*self.validation_issues, *internal_issues]:
            key = (issue.code, issue.message, issue.evidence_ids)
            if key in issue_keys:
                continue
            issue_keys.add(key)
            deduplicated.append(issue)
        self.validation_issues = deduplicated
        if (
            self.metadata is not None
            and self.metadata.references_incomplete
            and not any(i.code == "VAL_REFERENCES_INCOMPLETE" for i in self.validation_issues)
        ):
            self.validation_issues.append(references_incomplete_issue(self.metadata))

    def _compute_paper_id(self) -> str | None:
        """Compute the paper ID: user-supplied, else the input file's stem.

        The stem is what ``bibr batch`` and metacheck use too, and unlike the
        DOI it does not change when a later bibr reads the DOI differently.
        """
        if self.paper_id:
            return self.paper_id
        return Path(self.input_file.file_name or "").stem or None

    def export_to_json(
        self, *, include_regions: bool = False, include_region_meta: bool = False
    ) -> dict:
        """Export to JSON. Delegates to ``bibr.export.json_export``."""
        from bibr.export.json_export import export_paper_to_json

        return export_paper_to_json(
            self,
            include_regions=include_regions,
            include_region_meta=include_region_meta,
        )
