"""JSON export for Paper objects — v11.0 schema."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from pydantic import ValidationError

# ``bibr.export.models`` is the canonical home for every export model, and
# ``bibr.export`` re-exports the public ones straight from there. Every name
# below is used inside this module except the two marked ``noqa: F401``, which
# are kept only because importers still reach them through this module path.
from bibr.export.models import (
    _SCHEMA_VERSION,
    AffiliationExport,
    AuthorExport,
    BibExport,
    BibMatchExport,
    CaptionAssignmentExport,
    CaptionAssignmentReceiptExport,
    CaptionCandidateExport,
    CitationCandidateExport,
    CitationLinkingExport,
    EnrichmentExport,
    EqExport,
    FigureExport,
    FigurePartExport,
    FundingExport,
    LlmEngineExport,  # noqa: F401 - re-exported for existing importers
    MetadataExport,
    MetadataMatchExport,
    OcrEngineExport,  # noqa: F401 - re-exported for existing importers
    PaperExport,
    ProvenanceExport,
    ReferenceSegmentationAttemptExport,
    ReferenceYieldExport,
    RegionExport,
    SectionExport,
    SourceExport,
    TableExport,
    TablePartExport,
    TextExport,
    UrlExport,
    ValidationExport,
    ValidationIssueExport,
    XrefExport,
)
from bibr.export.name_split import split_person_names
from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from bibr.paper import Paper


from bibr.models import canonicalize_orcid

logger = logging.getLogger(__name__)

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def normalized_input_format(input_file) -> str | None:
    """The declared input format, lowercased ("pdf", "docx", "xml", ...).

    The pipeline stores enum member names (``"PDF"``) on
    ``InputFile.input_format`` while some callers keep a bare string; the schema
    contract is lowercase, so both shapes normalize through here — one
    implementation shared by the exporter and by ``ExportStage``, which uses it
    to tell the native-parse inputs from the OCR'd ones.
    """
    raw_fmt = getattr(input_file, "input_format", None)
    if raw_fmt is None:
        return None
    fmt = raw_fmt.file_type if hasattr(raw_fmt, "file_type") else str(raw_fmt)
    return fmt.lower() if fmt is not None else None


def _is_sane_url(url: str) -> bool:
    """Reject URLs an upstream line-wrap join left truncated (e.g. "https://blog",
    "http://scikit-learn" — a host-continuation wrap that failed to join).

    Only http/https links are checked, and only for a netloc containing at
    least one dot; everything else (doi.org links, query strings, trailing
    slashes, other schemes) passes through unchanged.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return True
    return "." in parts.netloc


def _normalize_export_url(url: str) -> str:
    """Strip PDF line-wrap artifacts from an extracted URL.

    A URL broken across a line in the source picks up the wrap whitespace when
    the text is re-joined, and a sentence-final period gets absorbed into the
    href. metacheck patched both downstream; fix them here instead.
    """
    cleaned = "".join(url.split())
    return cleaned.rstrip(".")


def _sane_export_links(links: list) -> list:
    """Filter *links* to those passing :func:`_is_sane_url`, logging drops.

    Normalizes before the sanity check so a wrapped URL is judged on its
    repaired form, not its raw (possibly truncated-looking) one.
    """
    kept = []
    for link in links:
        if _is_sane_url(_normalize_export_url(link.url)):
            kept.append(link)
        else:
            logger.debug("Dropping malformed URL from export: %r", link.url)
    return kept


def validate_export(data: dict) -> list[str]:
    """Validate export dict against the current v11.0 Pydantic schema.

    Returns a list of validation error messages (empty if valid).
    """
    try:
        PaperExport.model_validate(data)
        return []
    except ValidationError as exc:
        return [e["msg"] for e in exc.errors()]


def _issue_key(issue: ValidationIssue) -> tuple[str, str, str, tuple[str, ...], str]:
    return (
        issue.code,
        str(issue.severity),
        issue.origin_stage,
        tuple(issue.evidence_ids),
        issue.message,
    )


def _merge_validation_issues(*groups: list[ValidationIssue]) -> list[ValidationIssue]:
    """Deduplicate stage/replay findings and return them in stable key order."""

    merged: dict[tuple[str, str, str, tuple[str, ...], str], ValidationIssue] = {}
    for issue in (issue for group in groups for issue in group):
        key = _issue_key(issue)
        current = merged.get(key)
        if current is None:
            merged[key] = issue
            continue
        merged[key] = ValidationIssue(
            code=issue.code,
            severity=issue.severity,
            message=issue.message,
            origin_stage=issue.origin_stage,
            evidence_ids=tuple(issue.evidence_ids),
            count=max(current.count, issue.count),
            blocking=current.blocking or issue.blocking,
        )
    return [merged[key] for key in sorted(merged)]


def _apply_output_validation(
    payload: dict, stage_issues: list[ValidationIssue] | None = None
) -> dict:
    """Run the output validation gate and fold its findings into *payload*.

    Adds a structured top-level ``validation`` block. Gate findings are NOT
    mirrored into ``extraction.warnings``: they carry structure (code, severity,
    evidence) that a prose line throws away, and duplicating them made every
    consumer reason about two shapes of the same finding.

    Never raises: a gate failure degrades to a single ``VAL_INTERNAL`` issue.
    """
    from bibr.export.validation import ValidationIssue
    from bibr.export.validation import validate_export as _run_output_validation

    try:
        replay_issues = _run_output_validation(payload)
    except Exception:  # noqa: BLE001 - the gate must never break export
        logger.exception("output validation gate raised")
        replay_issues = [
            ValidationIssue("VAL_INTERNAL", IssueSeverity.WARNING, "output validation gate raised")
        ]

    source_issues = list(stage_issues or [])
    source_codes = {issue.code for issue in source_issues}
    source_owned_replay_codes = {"VAL_ABSTRACT_SUSPECT", "VAL_XREF_LOW_COVERAGE"}
    replay_issues = [
        issue
        for issue in replay_issues
        if not (issue.code in source_owned_replay_codes and issue.code in source_codes)
    ]
    issues = _merge_validation_issues(source_issues, replay_issues)

    # Gate findings live only in ``validation.issues`` — mirroring them into
    # warnings duplicated the same finding in two shapes for every consumer.
    blocking = sum(1 for issue in issues if issue.blocking)
    payload["validation"] = ValidationExport(
        errors=sum(1 for issue in issues if issue.severity == IssueSeverity.ERROR),
        warnings=sum(1 for issue in issues if issue.severity == IssueSeverity.WARNING),
        blocking=blocking,
        promotable=blocking == 0,
        issues=[
            ValidationIssueExport(
                code=issue.code,
                severity=str(issue.severity),
                message=issue.message,
                origin_stage=issue.origin_stage,
                evidence_ids=list(issue.evidence_ids),
                count=issue.count,
                blocking=issue.blocking,
            )
            for issue in issues
        ],
    ).model_dump(mode="json")
    return payload


def append_payload_warning(payload: dict, message: str) -> dict:
    """Append a non-fatal warning to an already-serialized payload.

    Warnings live at ``extraction.warnings`` in v11. Post-export mutators
    (consolidation, checkpoint enrichment replay) reach the payload after the
    export models have run, so they go through this helper instead of touching a
    root key that no longer exists. A no-op when there is no ``extraction``
    block — a Paper exported outside the pipeline has nowhere to record
    processing provenance.
    """
    extraction = payload.get("extraction")
    if not isinstance(extraction, dict):
        return payload
    existing = extraction.get("warnings")
    warnings = list(existing) if isinstance(existing, list) else []
    if message not in warnings:
        warnings.append(message)
    extraction["warnings"] = warnings
    return payload


def _sanitize_json_strings(value):
    """Replace lone surrogate code points so JSON responses are UTF-8 encodable."""
    if isinstance(value, str):
        return _SURROGATE_RE.sub("\ufffd", value)
    if isinstance(value, list):
        return [_sanitize_json_strings(item) for item in value]
    if isinstance(value, dict):
        return {
            _sanitize_json_strings(key) if isinstance(key, str) else key: _sanitize_json_strings(
                item
            )
            for key, item in value.items()
        }
    return value


def _export_paper_payload(
    paper: Paper,
    *,
    include_regions: bool = False,
    include_region_meta: bool = False,
    validate: bool = True,
) -> dict:
    """Export Paper as a JSON-serializable dict matching the bibr v11.0 schema.

    Returns a dictionary conforming to the scienceverse paper.json schema,
    suitable for serialization with json.dumps().

    The dict is CONSTRUCTED through the ``PaperExport`` models — the schema
    is the builder, not a post-hoc checker — so any divergence raises
    ``pydantic.ValidationError`` at build time instead of shipping invalid
    output with a logged warning.

    ``include_regions`` controls whether the ``extraction.regions`` debug
    payload (per-region layout: bbox, font, content, etc.) is emitted. Off by
    default — the field is large (≈25% of typical output) and not consumed
    by Metacheck or the standard paper schema. It rides ``extraction``, so it
    is dropped when a Paper is exported outside the pipeline.

    ``include_region_meta`` controls the per-text underscore fields
    (``_bbox_2d``, ``_font_size``, ``_region_type``, …) originally added as
    v4 training features. Off by default — they are ≈19% of typical output
    and, like ``extraction.regions``, not part of the Metacheck-facing API.
    When off
    the keys are omitted entirely; when on, the pre-v10.5 shape is emitted
    (keys always present, ``null`` when unavailable).

    ``validate`` (default on) runs the output validation gate
    (:mod:`bibr.export.validation`) and adds a top-level ``validation`` block.
    Gate findings live there only — they are never mirrored into
    ``extraction.warnings``. The gate never raises out of export.
    """
    if not paper.contents:
        raise ValueError("Paper has no contents")

    input_fmt = normalized_input_format(paper.input_file)

    # text (built from sentences; section_id=0 remapped to null, display math → formatted)
    text_data = []
    for sent in paper.contents.sentences:
        if sent.is_display_formula:
            text = "[equation]"
            formatted = sent.text
        else:
            text = sent.text
            formatted = None
        rm = (sent.region_meta if include_region_meta else None) or {}
        text_data.append(
            TextExport(
                text=text,
                text_id=sent.text_id,
                paragraph_id=sent.paragraph_id,
                section_id=sent.section_id if sent.section_id != 0 else None,
                page_number=sent.page_number,
                formatted=formatted,
                # v4 training region metadata (underscore-prefixed in JSON; not
                # part of the Metacheck-facing API).  None when unavailable
                # (DOCX input, scanned PDF without a native text layer, or
                # pre-v4 pipeline runs).
                region_font_size=rm.get("font_size"),
                region_font_bold=rm.get("font_bold"),
                region_is_italic=rm.get("is_italic"),
                region_bbox_2d=rm.get("bbox_2d"),
                region_type=rm.get("region_type"),
                region_page_w=rm.get("page_w"),
                region_page_h=rm.get("page_h"),
            )
        )

    def _authors_to_dicts(authors: list | None) -> list[dict] | None:
        if not authors:
            return None
        return [{"given": a.given, "family": a.family} for a in authors]

    # bib (flat, no nested match) and bib_match (top-level)
    bib_data = []
    bib_match_data = []
    for r in paper.metadata.references if paper.metadata else []:
        # ``is_in_press`` is a flag, not a year-replacement: a paper can be
        # accepted-but-not-yet-issued AND have a known copyright year. Only
        # collapse the export year to ``None`` when the year really is the
        # legacy ``0`` sentinel; preserve real numeric years otherwise.
        is_in_press = bool(getattr(r, "is_in_press", False)) or r.year == 0
        export_year = None if r.year == 0 else r.year
        # `bib[]` is the printed reference verbatim — never backfilled from
        # external matches. Crossref/OpenAlex hits live in `bib_match[]` for
        # consumers that want enrichment.
        bib_data.append(
            BibExport(
                bib_id=r.bib_id,
                text_id=r.text_id,
                bib_type=r.bib_type,
                doi=r.doi,
                title=r.title or None,
                authors=r.authors,
                # ``or None``: an unsplittable/empty verbatim string yields
                # ``null``, matching the sibling match tables' encoding rather
                # than emitting an empty list. See BibExport.author.
                author=split_person_names(r.authors) or None,
                editors=r.editors,
                editor=split_person_names(r.editors) or None,
                publisher=r.publisher,
                year=export_year,
                year_suffix=r.year_suffix,
                date=r.date,
                container=r.container,
                volume=r.volume,
                issue=r.issue,
                first_page=r.first_page,
                last_page=r.last_page,
                edition=r.edition,
                version=r.version,
                # Reference URLs are parsed from line-joined reference text, so
                # they carry the same wrap artifacts as ``url[].href`` and get
                # the same repair. The downstream R consumer drops its own
                # whitespace/trailing-dot patch on the strength of this release.
                url=_normalize_export_url(r.url) if r.url else r.url,
                is_in_press=is_in_press,
                arxiv=r.arxiv,
                pmid=r.pmid,
                series=r.series,
                access_date=r.access_date,
                note=r.note,
            )
        )
        # Flatten matches into bib_match table
        for src, m in r.match.items():
            try:
                d = m.model_dump()
                bib_match_data.append(
                    BibMatchExport(
                        bib_id=r.bib_id,
                        service=src.value,
                        service_id=d.get("id"),
                        score=d.get("score"),
                        bib_type=d.get("bib_type"),
                        doi=d.get("doi"),
                        title=d.get("title"),
                        author=_authors_to_dicts(m.authors),
                        editor=_authors_to_dicts(m.editors),
                        publisher=d.get("publisher"),
                        year=d.get("year"),
                        date=d.get("date"),
                        container=d.get("container"),
                        volume=d.get("volume"),
                        issue=d.get("issue"),
                        first_page=d.get("first_page"),
                        last_page=d.get("last_page"),
                        edition=d.get("edition"),
                        version=d.get("version"),
                        url=d.get("url"),
                    )
                )
            except Exception:
                logger.debug("Failed to serialize match for bib_id=%s src=%s", r.bib_id, src)

    # metadata_match: the paper's OWN identity enrichment, flattened the same
    # way bib_match flattens reference matches (minus bib_id).
    metadata_match_data = []
    for src, m in paper.metadata.match.items() if paper.metadata else []:
        try:
            d = m.model_dump()
            metadata_match_data.append(
                MetadataMatchExport(
                    service=src.value,
                    service_id=d.get("id"),
                    score=d.get("score"),
                    bib_type=d.get("bib_type"),
                    doi=d.get("doi"),
                    title=d.get("title"),
                    author=_authors_to_dicts(m.authors),
                    editor=_authors_to_dicts(m.editors),
                    publisher=d.get("publisher"),
                    year=d.get("year"),
                    date=d.get("date"),
                    container=d.get("container"),
                    volume=d.get("volume"),
                    issue=d.get("issue"),
                    first_page=d.get("first_page"),
                    last_page=d.get("last_page"),
                    edition=d.get("edition"),
                    version=d.get("version"),
                    url=d.get("url"),
                )
            )
        except Exception:
            logger.debug("Failed to serialize metadata match for src=%s", src)

    # Abstract/keywords policy (section fallback, keyword recovery, commentary
    # guard) is finalized in post-parse (``_finalize_abstract_and_keywords`` in
    # bibr/pipeline/stages/post_parse.py) — this layer serializes metadata
    # verbatim.
    abstract_text = ((paper.metadata.abstract or "").strip() or None) if paper.metadata else None
    exported_keywords = paper.metadata.keywords if paper.metadata else []

    # Structured enrichment completeness — emitted only when enrichment actually
    # ran (metadata.enrichment_complete is not None) and there are refs to enrich.
    enrichment_export: EnrichmentExport | None = None
    _refs = paper.metadata.references if paper.metadata else []
    if paper.metadata is not None and paper.metadata.enrichment_complete is not None and _refs:
        enrichment_export = EnrichmentExport(
            complete=paper.metadata.enrichment_complete,
            refs_enriched=sum(1 for r in _refs if r.match),
            refs_total=len(_refs),
        )

    regions_data: list[RegionExport] | None = None
    if include_regions and paper.contents.region_summaries:
        regions_data = [
            RegionExport(
                page=r.page,
                index=r.index,
                label=r.label,
                bbox=list(r.bbox) if r.bbox else None,
                font_size=r.font_size,
                font_weight=r.font_weight,
                font_bold=r.font_bold,
                section_id=r.section_id,
                content=r.content,
                raw_ocr_content=r.raw_ocr_content,
                bbox_height=r.bbox_height,
                bbox_width=r.bbox_width,
                char_density=r.char_density,
                estimated_line_height=r.estimated_line_height,
            )
            for r in paper.contents.region_summaries
        ]

    citation_linking: CitationLinkingExport | None = None
    if paper.contents.citation_receipt is not None:
        receipt = paper.contents.citation_receipt
        citation_linking = CitationLinkingExport(
            style_scores=dict(receipt.style_scores),
            candidates=[
                CitationCandidateExport(
                    text_id=candidate.text_id,
                    start=candidate.start,
                    end=candidate.end,
                    raw=candidate.raw,
                    style=candidate.style,
                    bib_ids=list(candidate.bib_ids),
                    evidence=list(candidate.evidence),
                    confidence=candidate.confidence,
                    accepted=candidate.accepted,
                    rejection_reasons=list(candidate.rejection_reasons),
                )
                for candidate in receipt.candidates
            ],
            resolved_candidate_fraction=receipt.resolved_candidate_fraction,
            unique_linked_bib_fraction=receipt.unique_linked_bib_fraction,
        )

    caption_assignment: CaptionAssignmentReceiptExport | None = None
    if paper.contents.caption_assignment_receipt is not None:
        receipt = paper.contents.caption_assignment_receipt
        # Every non-null object_id must name a float that survived to the
        # export. The receipt is frozen before floats_normalize renumbers, so
        # a stale id here means its old→new remap did not reach this
        # assignment. Logged, not raised: the receipt is a diagnostic, and
        # dropping it (or the export) would hide the very inconsistency it is
        # meant to expose.
        live_object_ids = {f"figure:{item.figure_id}" for item in paper.contents.figures} | {
            f"table:{item.table_id}" for item in paper.contents.tables
        }
        dangling = sorted(
            {
                assignment.object_id
                for assignment in receipt.assignments
                if assignment.object_id is not None and assignment.object_id not in live_object_ids
            }
        )
        if dangling:
            logger.debug(
                "caption_assignment receipt names %d object(s) with no live float: %s",
                len(dangling),
                ", ".join(dangling[:10]),
            )
        caption_assignment = CaptionAssignmentReceiptExport(
            candidates=[
                CaptionCandidateExport(
                    caption_id=candidate.caption_id,
                    text=candidate.text,
                    object_type=candidate.object_type,
                    page_number=candidate.page_number,
                    bbox=list(candidate.bbox) if candidate.bbox else None,
                    source_index=candidate.source_index,
                )
                for candidate in receipt.candidates
            ],
            assignments=[
                CaptionAssignmentExport(
                    caption_id=assignment.caption_id,
                    object_id=assignment.object_id,
                    score=assignment.score,
                    reasons=list(assignment.reasons),
                    ambiguous=assignment.ambiguous,
                )
                for assignment in receipt.assignments
            ],
        )

    reference_yield: ReferenceYieldExport | None = None
    if paper.contents.reference_yield_receipt is not None:
        receipt = paper.contents.reference_yield_receipt
        reference_yield = ReferenceYieldExport(
            credible_source_starts=receipt.credible_source_starts,
            attempts=[
                ReferenceSegmentationAttemptExport(
                    strategy=attempt.strategy,
                    spans=[list(span) for span in attempt.spans],
                    credible_starts=attempt.credible_starts,
                    selected=attempt.selected,
                    reason_flags=list(attempt.reason_flags),
                )
                for attempt in receipt.attempts
            ],
            selected_spans=[list(span) for span in receipt.selected_spans],
            source_character_coverage=receipt.source_character_coverage,
            parsed_count=receipt.parsed_count,
            valid_count=receipt.valid_count,
            duplicate_rate=receipt.duplicate_rate,
            reason_flags=list(receipt.reason_flags),
        )

    # v11: every telemetry surface hangs off ``extraction``. The stage builds
    # the provenance skeleton; the receipts/enrichment/regions that only exist
    # at serialization time are folded in here. ``extraction`` is absent
    # entirely when a Paper is exported outside the pipeline.
    extraction_data = dict(paper.extraction) if paper.extraction is not None else None
    if extraction_data is not None:
        diagnostics = dict(extraction_data.get("diagnostics") or {})
        diagnostics["references_complete"] = not bool(
            paper.metadata and paper.metadata.references_incomplete
        )
        # ``setdefault``: the pipeline's ``_build_extraction`` already stamps the
        # score, and its value wins. This only covers callers that hand-build an
        # ``extraction`` block without one, so the score is never silently lost.
        diagnostics.setdefault("text_quality", paper.text_quality)
        if citation_linking is not None:
            diagnostics["citation_linking"] = citation_linking
        if caption_assignment is not None:
            diagnostics["caption_assignment"] = caption_assignment
        if reference_yield is not None:
            diagnostics["reference_yield"] = reference_yield
        extraction_data["diagnostics"] = diagnostics
        if enrichment_export is not None:
            extraction_data["enrichment"] = enrichment_export
        if regions_data is not None:
            extraction_data["regions"] = regions_data
        # Warnings are unioned rather than overwritten: the stage snapshots
        # ``paper.processing_warnings`` when it builds the block, but callers
        # (and the stage itself) may append after that point.
        extraction_data["warnings"] = list(
            dict.fromkeys([*(extraction_data.get("warnings") or []), *paper.processing_warnings])
        )

    export = PaperExport(
        paper_id=paper._compute_paper_id(),
        schema_version=_SCHEMA_VERSION,
        source=SourceExport(
            file_name=paper.input_file.file_name,
            file_hash=paper.input_file.file_hash,
            input_format=input_fmt or "unknown",
        ),
        metadata=MetadataExport(
            title=(paper.metadata.title or None) if paper.metadata else None,
            abstract=abstract_text,
            keywords=exported_keywords,
            doi=(paper.metadata.doi or None) if paper.metadata else None,
            paper_type=(paper.metadata.paper_type or None) if paper.metadata else None,
            paper_type_confidence=(
                paper.metadata.paper_type_confidence
                if paper.metadata and paper.metadata.paper_type
                else None
            ),
            oecd_l1=(paper.metadata.oecd_l1 or None) if paper.metadata else None,
            oecd_l2=(paper.metadata.oecd_l2 or None) if paper.metadata else None,
            oecd_confidence=(
                paper.metadata.oecd_confidence
                if paper.metadata and paper.metadata.oecd_l1
                else None
            ),
            journal=(paper.metadata.journal or None) if paper.metadata else None,
            volume=(paper.metadata.volume or None) if paper.metadata else None,
            issue=(paper.metadata.issue or None) if paper.metadata else None,
            first_page=(paper.metadata.first_page or None) if paper.metadata else None,
            last_page=(paper.metadata.last_page or None) if paper.metadata else None,
            issn=(paper.metadata.issn or None) if paper.metadata else None,
            publisher=(paper.metadata.publisher or None) if paper.metadata else None,
            published=(paper.metadata.published or None) if paper.metadata else None,
            license=(paper.metadata.license or None) if paper.metadata else None,
            funding_statement=(paper.metadata.funding_statement or None)
            if paper.metadata
            else None,
            coi_statement=(paper.metadata.coi_statement or None) if paper.metadata else None,
            ethics_statement=(paper.metadata.ethics_statement or None) if paper.metadata else None,
            data_availability=(
                (paper.metadata.data_availability or None) if paper.metadata else None
            ),
        ),
        author=[
            AuthorExport(
                author_id=a.author_id,
                given=a.given,
                family=a.family,
                suffix=getattr(a, "suffix", None),
                affiliation=a.affiliation or None,
                email=a.email,
                corresponding=a.corresponding,
                orcid=canonicalize_orcid(a.orcid),
                role=a.role,
            )
            for a in (paper.metadata.authors if paper.metadata else [])
        ],
        text=text_data,
        section=[
            SectionExport(
                section_id=s.section_id,
                header=s.header,
                level=s.level,
                parent_section_id=(s.parent_section_id if s.parent_section_id != 0 else None),
                section_type=s.section_type.value if s.section_type else None,
                classification_score=s.classification_score,
                classification_source=s.classification_source,
            )
            for s in paper.contents.sections
            if s.section_id != 0
        ],
        url=[
            UrlExport(
                href=_normalize_export_url(link.url),
                link_text=link.link_text,
                text_id=link.text_id,
            )
            for link in _sane_export_links(paper.contents.links)
        ],
        bib=bib_data,
        xref=[
            XrefExport(
                target_id=x.xref_id,
                xref_type=x.xref_type,
                contents=x.contents,
                text_id=x.text_id,
                tier=x.tier or None,
            )
            for x in paper.contents.xrefs
        ],
        figure=[
            FigureExport(
                figure_id=f.figure_id,
                section_id=f.section_id if f.section_id != 0 else None,
                image=f.image_b64,
                caption=f.caption,
                page_number=f.page_number,
                parts=[
                    FigurePartExport(
                        part_index=index,
                        image=part.image_b64,
                        page_number=part.page_number,
                        bbox=list(part.bbox) if part.bbox else None,
                        provenance=[
                            ProvenanceExport(
                                page=provenance.page_no,
                                bbox=list(provenance.bbox) if provenance.bbox else None,
                            )
                            for provenance in part.provenance
                        ],
                    )
                    for index, part in enumerate(f.parts, 1)
                ],
            )
            for f in paper.contents.figures
        ],
        table=[
            TableExport(
                table_id=t.table_id,
                section_id=t.section_id if t.section_id != 0 else None,
                html=t.tbl_html or None,
                contents=t.contents,
                caption=t.caption,
                page_number=t.page_number,
                parts=[
                    TablePartExport(
                        part_index=index,
                        html=part.tbl_html,
                        contents=part.contents,
                        page_number=part.page_number,
                        bbox=list(part.bbox) if part.bbox else None,
                        provenance=[
                            ProvenanceExport(
                                page=provenance.page_no,
                                bbox=list(provenance.bbox) if provenance.bbox else None,
                            )
                            for provenance in part.provenance
                        ],
                    )
                    for index, part in enumerate(t.parts, 1)
                ],
            )
            for t in paper.contents.tables
        ],
        eq=[
            EqExport(
                eq_id=position,
                text_id=eq.text_id,
                grp_id=eq.grp_id,
                verbatim=getattr(eq, "verbatim", None),
                lhs=eq.lhs,
                df=eq.df,
                comp=eq.comp,
                rhs=eq.rhs,
            )
            for position, eq in enumerate(paper.contents.equations, start=1)
        ],
        bib_match=bib_match_data,
        metadata_match=metadata_match_data,
        funding=[
            FundingExport(funding_id=position, funder=f.funder, award_ids=f.award_ids)
            for position, f in enumerate(paper.metadata.funding if paper.metadata else [], start=1)
        ],
        affiliation=[
            AffiliationExport(
                affiliation_id=position,
                text=a.text,
                institution=a.institution,
                department=a.department,
                city=a.city,
                country=a.country,
                author_ids=a.author_ids,
            )
            for position, a in enumerate(
                paper.metadata.affiliations if paper.metadata else [], start=1
            )
        ],
        qualification_provenance=paper.qualification_provenance or None,
        extraction=extraction_data,
    )

    # Without the opt-in, the underscore keys are omitted entirely (not null):
    # they are debug/training payload, and null-emitting them costs bytes on
    # every sentence.
    exclude = (
        None
        if include_region_meta
        else {
            "text": {
                "__all__": {
                    "region_font_size",
                    "region_font_bold",
                    "region_is_italic",
                    "region_bbox_2d",
                    "region_type",
                    "region_page_w",
                    "region_page_h",
                }
            }
        }
    )
    payload = export.model_dump(by_alias=True, exclude=exclude)
    if validate:
        payload = _apply_output_validation(payload, paper.validation_issues)
    return cast(dict[str, Any], _sanitize_json_strings(payload))


def build_paper_export(
    paper: Paper,
    *,
    include_regions: bool = False,
    include_region_meta: bool = False,
    validate: bool = True,
) -> PaperExport:
    """Build and validate the typed v11.0 export model for *paper*."""
    payload = _export_paper_payload(
        paper,
        include_regions=include_regions,
        include_region_meta=include_region_meta,
        validate=validate,
    )
    return cast(PaperExport, PaperExport.model_validate(payload))


def export_paper_to_json(
    paper: Paper,
    *,
    include_regions: bool = False,
    include_region_meta: bool = False,
    validate: bool = True,
) -> dict[str, Any]:
    """Export *paper* as the v11.0 dictionary payload."""
    export = build_paper_export(
        paper,
        include_regions=include_regions,
        include_region_meta=include_region_meta,
        validate=validate,
    )
    return cast(dict[str, Any], export.model_dump(by_alias=True, exclude_unset=True))
