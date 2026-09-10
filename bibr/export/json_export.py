"""JSON export for Paper objects — v10.7 schema."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_serializer

from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from bibr.paper import Paper


from bibr.models import canonicalize_orcid

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "10.7"
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


# ---------------------------------------------------------------------------
# Pydantic v10.7 export schema — single source of truth for validation
#
# v10.7 (vs 10.6) — additive:
#   - Figure/table ``parts`` preserve every physical payload and provenance.
#   - Optional ``caption_assignment`` and ``reference_yield`` diagnostic receipts.
#   - Typed readers continue to accept v10.6 payloads without these fields.
#
# v10.6 (vs 10.5) — additive:
#   - New top-level ``llm_usage_by_label``: per-call-site LLM token usage
#     (e.g. ``extract_authors``), sibling of ``llm_usage`` (which is
#     per-model); null when no LLM ran or usage tracking is disabled.
#   - ``xref`` gains optional ``tier`` field: bib-link detection provenance
#     ("numeric", "paren-numeric", "flattened-superscript", "author-year",
#     "llm"); null for non-bib xrefs.
#
# v10.5 (vs 10.4) — all additive:
#   - ``info.bibr_version`` restored, now carrying the producing *package*
#     version (pre-10.3 it held the schema version). ``info.schema_version``
#     is unchanged; ``extraction.bibr_version`` keeps a copy for provenance.
#   - ``info.input_format`` is now contractually lowercase ("pdf", "docx",
#     "xml", "unknown") — previously leaked enum member names ("PDF").
#   - Per-text underscore region metadata (``_bbox_2d``, ``_font_size``, …)
#     is now opt-in (``include_region_meta=True``; CLI ``--region-meta``),
#     mirroring ``_regions``. Off by default: extra detail from
#     v4-training payload not consumed by Metacheck.
#   - New top-level ``validation`` block ``{errors, warnings, issues[]}`` and
#     ``VALIDATION:<severity>:<code>: <message>`` lines appended to
#     ``processing_warnings`` — the output validation gate (bibr/export/
#     validation.py) makes catastrophic output defects loudly visible.
#
# v10.4 (vs 10.3) — all additive:
#   - ``figure[].caption`` / ``table[].caption``: the matched caption text
#     (previously captured internally but dropped at export).
#   - ``info`` gains the paper's OWN bibliographic self-identity, verbatim
#     from the front matter: ``journal``, ``volume``, ``issue``,
#     ``first_page``, ``last_page``, ``issn``, ``publisher``, ``published``,
#     ``license`` (all nullable).
#   - New top-level ``info_match``: enrichment matches for the paper's own
#     identity (self-DOI lookup), shaped like ``bib_match`` minus ``bib_id``.
#   - ``info`` gains four verbatim research-integrity statements, copied from
#     the classified section bodies (nullable scalars): ``funding_statement``,
#     ``coi_statement``, ``ethics_statement``, ``data_availability``.
#   - New top-level ``funding``: structured funding parsed from the funding
#     statement — a list of ``{funder, award_ids}`` (empty by default).
#   - New top-level ``affiliations``: structured affiliations parsed from the
#     author byline — a list of ``{affiliation_id, text, institution,
#     department, city, country, author_ids}`` (empty by default). ``text`` is
#     our verbatim byline string; the parsed components are best-effort LLM.
#   - ``author[].role``: contribution phrases mapped from the
#     author-contributions statement (already in the schema, now populated).
#
# v10.3 (vs 10.2):
#   - ``info.bibr_version`` renamed to ``info.schema_version``. The old name
#     conflated the schema shape with the producing software; it was always
#     the *schema* version. The bibr package version now lives in the new
#     top-level ``extraction.bibr_version``. (Breaking change for consumers
#     that read ``info.bibr_version`` — e.g. metacheck's ``read_bibr``.)
#   - New top-level ``extraction`` object: extraction provenance — bibr
#     package version, resolved reference seg/parse strategies, whether the
#     CRF seg-fallback fired, effective Crossref-enrich/consolidate flags, and
#     per-stage timings. Additive optional object; ``null`` when a Paper is
#     exported outside the pipeline.
#
# v10.2 (vs 10.1) — tracks scienceverse/schema "Updated text order and eq.df":
#   - ``eq`` entries carry a ``df`` field: degrees of freedom shown
#     parenthetically on the LHS are split out (``t(28)`` → ``lhs="t",
#     df="28"``) instead of being folded into ``lhs``.
#   - ``text`` keys follow the upstream order (text, text_id, paragraph_id,
#     section_id, page_number, formatted); ``formatted`` may also carry XML
#     for grobid imports.
#   - ``xref.xref_id`` is nullable — an xref need not resolve to a
#     bib/table/figure/footnote target.
#
# v10.1 (vs 10.0):
#   - ``info.ocr_config`` moved to top-level ``ocr_config``
#   - ``info.processing_warnings`` moved to top-level ``processing_warnings``
#     Both moves keep ``info`` scalar-only so R consumers (metacheck) can
#     ``as.data.frame(info)`` without nested-object row-count errors.
#   - ``_regions`` debug payload is now opt-in (``include_regions=True``).
# ---------------------------------------------------------------------------


# Every export model rejects unknown keys so a divergence between the schema
# and the hand-built dicts in ``export_paper_to_json`` surfaces as a
# validation error instead of being silently ignored. ``populate_by_name``
# lets the underscore-prefixed JSON keys (pydantic forbids them as field
# names) be declared via aliases.
_STRICT = ConfigDict(extra="forbid", populate_by_name=True)


class BibAuthorExport(BaseModel):
    """Structured author for bib_match entries (from external services)."""

    model_config = _STRICT

    given: str | None
    family: str | None


class OcrConfigExport(BaseModel):
    # Reject unknown keys so a divergence between this schema and the dict
    # built in ``ExportStage._build_ocr_config`` raises ``ValidationError``
    # instead of being silently dropped.
    model_config = _STRICT

    ocr_backend: str | None = None
    ocr_model: str | None = None
    ocr_profile: Literal["paddle", "glm"] | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    no_llm: bool = False


class InfoExport(BaseModel):
    model_config = _STRICT

    # Scalars only: nested objects/lists here break R consumers that call
    # ``as.data.frame(info)`` (e.g. metacheck's ``read_bibr``). Processing
    # metadata lives at the top level of ``PaperExport`` instead.
    title: str | None
    abstract: str | None = None
    keywords: list[str]
    doi: str | None
    file_hash: str
    input_format: str
    file_name: str
    schema_version: Literal["10.6", "10.7"]
    # The producing bibr package version (e.g. "0.3.0"). Restored alongside
    # schema_version: pre-10.3 consumers required ``info.bibr_version`` (it
    # then held the schema version); it now carries the package version, which
    # also lives in ``extraction.bibr_version`` (kept for provenance grouping).
    bibr_version: str
    paper_type: str | None = None
    paper_type_confidence: float | None = None
    oecd_l1: str | None = None
    oecd_l2: str | None = None
    oecd_confidence: float | None = None
    # The paper's OWN bibliographic self-identity, verbatim from the front
    # matter. Scalars only; the enrichment match lives in ``info_match``.
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    first_page: str | None = None
    last_page: str | None = None
    issn: str | None = None
    publisher: str | None = None
    published: str | None = None
    license: str | None = None
    # Research-integrity statements, copied verbatim from the classified
    # section bodies. Scalars only; structured funding lives in top-level
    # ``funding``.
    funding_statement: str | None = None
    coi_statement: str | None = None
    ethics_statement: str | None = None
    data_availability: str | None = None
    # Report-only parse-quality score in [0, 1] (10th-percentile aggregation of
    # per-region garbage/fragmentation ratings). None when scoring is disabled
    # or nothing was scoreable (e.g. DOCX-native input).
    text_quality: float | None = None


class AuthorExport(BaseModel):
    model_config = _STRICT

    author_id: int
    given: str
    family: str
    affiliation: str | None = None
    email: str | None = None
    corresponding: bool
    orcid: str | None = None
    role: list[str] = []


class TextExport(BaseModel):
    model_config = _STRICT

    # Key order mirrors the upstream schema (text, text_id, paragraph_id,
    # section_id, page_number, formatted) so metacheck's R data.frame columns
    # line up.
    text: str
    text_id: int
    paragraph_id: int
    section_id: int | None
    page_number: int | None
    formatted: str | None = None
    # v4 training region metadata — bibr-internal, underscore-prefixed in the
    # JSON (not part of the Metacheck-facing API). Pydantic forbids leading
    # underscores in field names, hence the aliases.
    region_font_size: float | None = Field(default=None, alias="_font_size")
    region_font_bold: bool | None = Field(default=None, alias="_font_bold")
    region_is_italic: bool | None = Field(default=None, alias="_is_italic")
    region_bbox_2d: list[float] | None = Field(default=None, alias="_bbox_2d")
    region_type: str | None = Field(default=None, alias="_region_type")
    region_page_w: float | None = Field(default=None, alias="_page_w")
    region_page_h: float | None = Field(default=None, alias="_page_h")


class SectionExport(BaseModel):
    model_config = _STRICT

    section_id: int
    header: str
    level: int
    parent_section_id: int | None
    section_type: str | None
    classification_score: float
    classification_source: str | None = None


class UrlExport(BaseModel):
    model_config = _STRICT

    href: str
    link_text: str | None
    text_id: int


class BibExport(BaseModel):
    model_config = _STRICT

    bib_id: int
    text_id: int | None = None
    bib_type: str | None = None
    doi: str | None = None
    title: str | None = None
    authors: str | None = None
    editors: str | None = None
    publisher: str | None = None
    year: int | None = None
    year_suffix: str | None = None
    date: str | None = None
    container: str | None = None
    volume: str | None = None
    issue: str | None = None
    first_page: str | None = None
    last_page: str | None = None
    edition: str | None = None
    version: str | None = None
    url: str | None = None
    is_in_press: bool = False
    # Set by consolidation (bibr/enrich/consolidate.py) after export: a
    # comma-joined list of the field names filled/replaced from bib_match.
    # Absent (None → excluded by the serializer) on unconsolidated output,
    # which therefore stays byte-identical.
    consolidated_fields: str | None = None

    @model_serializer(mode="wrap")
    def _omit_unset_consolidated(self, handler):
        data = handler(self)
        if data.get("consolidated_fields") is None:
            data.pop("consolidated_fields", None)
        return data


class BibMatchExport(BaseModel):
    """Flat enrichment match from an external service."""

    model_config = _STRICT

    bib_id: int
    service: str
    service_id: str | None = None
    score: float | None = None
    bib_type: str | None = None
    doi: str | None = None
    title: str | None = None
    authors: list[BibAuthorExport] | None = None
    editors: list[BibAuthorExport] | None = None
    publisher: str | None = None
    year: int | None = None
    date: str | None = None
    container: str | None = None
    volume: str | None = None
    issue: str | None = None
    first_page: str | None = None
    last_page: str | None = None
    edition: str | None = None
    version: str | None = None
    url: str | None = None


class InfoMatchExport(BaseModel):
    """Flat enrichment match for the paper's OWN bibliographic identity.

    Mirrors :class:`BibMatchExport` (same external-service shape) but keyed to
    the paper rather than a reference, so it carries no ``bib_id``.
    """

    model_config = _STRICT

    service: str
    service_id: str | None = None
    score: float | None = None
    bib_type: str | None = None
    doi: str | None = None
    title: str | None = None
    authors: list[BibAuthorExport] | None = None
    editors: list[BibAuthorExport] | None = None
    publisher: str | None = None
    year: int | None = None
    date: str | None = None
    container: str | None = None
    volume: str | None = None
    issue: str | None = None
    first_page: str | None = None
    last_page: str | None = None
    edition: str | None = None
    version: str | None = None
    url: str | None = None


class FundingExport(BaseModel):
    """Structured funding entry parsed from the funding statement."""

    model_config = _STRICT

    funder: str
    award_ids: list[str] = []


class AffiliationExport(BaseModel):
    """Structured affiliation parsed from the author byline. ``text`` is our own
    verbatim byline string; the parsed components are best-effort (LLM)."""

    model_config = _STRICT

    affiliation_id: int  # 1-based position
    text: str
    institution: str | None = None
    department: str | None = None
    city: str | None = None
    country: str | None = None
    author_ids: list[int] = []


class XrefExport(BaseModel):
    model_config = _STRICT

    # Nullable: an xref need not resolve to a bib/table/figure/footnote target.
    xref_id: int | None
    xref_type: Literal["bib", "table", "figure", "foot", "supplementary", "equation", "section"]
    contents: str | None
    text_id: int
    # Detection tier for bib xrefs ("numeric", "paren-numeric",
    # "flattened-superscript", "author-year", "llm"); null for non-bib types.
    tier: (
        Literal["numeric", "paren-numeric", "flattened-superscript", "author-year", "llm"] | None
    ) = None


class FigureExport(BaseModel):
    model_config = _STRICT

    figure_id: int
    section_id: int | None = None
    image: str | None = None
    caption: str | None = None
    page_number: int | None
    parts: list[FigurePartExport] = []


class ProvenanceExport(BaseModel):
    model_config = _STRICT

    page: int
    bbox: list[float] | None = None


class FigurePartExport(BaseModel):
    model_config = _STRICT

    part_index: int
    image: str | None = None
    page_number: int | None = None
    bbox: list[float] | None = None
    provenance: list[ProvenanceExport] = []


class TableExport(BaseModel):
    model_config = _STRICT

    table_id: int
    section_id: int | None = None
    html: str | None = None
    contents: list
    caption: str | None = None
    page_number: int | None
    parts: list[TablePartExport] = []


class TablePartExport(BaseModel):
    model_config = _STRICT

    part_index: int
    html: str | None = None
    contents: list
    page_number: int | None = None
    bbox: list[float] | None = None
    provenance: list[ProvenanceExport] = []


class EqExport(BaseModel):
    model_config = _STRICT

    text_id: int
    grp_id: int
    lhs: str
    df: str
    comp: str
    rhs: str


class RegionExport(BaseModel):
    """Per-region layout debug payload (``include_regions=True``)."""

    model_config = _STRICT

    page: int
    index: int
    label: str | None
    bbox: list[float] | None
    font_size: float | None = None
    font_weight: float | None = None
    font_bold: bool | None = None
    section_id: int | None = None
    content: str | None = None
    raw_ocr_content: str | None = None
    source_region_ids: list[str] | None = None
    native_spans: list[dict[str, Any]] | None = None
    formula_proposals: list[dict[str, Any]] | None = None
    bbox_height: float | None = None
    bbox_width: float | None = None
    char_density: float | None = None
    estimated_line_height: float | None = None


class ExpectedIdentityExport(BaseModel):
    model_config = _STRICT

    queue_record_id: str
    expected_doi: str | None = None
    expected_doi_sha256: str | None = None
    expected_title: str | None = None
    target_block_hint: dict[str, object] | None = None
    source_sha256: str | None = None
    doi_required: bool = False


class DoiCandidateExport(BaseModel):
    model_config = _STRICT

    raw: str
    normalized: str
    source_kind: str
    page: int | None
    section_id: int | None
    section_type: str | None
    region_index: int | None
    region_type: str | None
    text_id: int | None
    marker_kind: str
    repeated_header_footer_count: int
    semantic_context: str
    selection_tier: int
    rejection_reason: str | None = None


class IdentityReceiptExport(BaseModel):
    model_config = _STRICT

    selected: DoiCandidateExport | None = None
    candidates: list[DoiCandidateExport] = Field(default_factory=list)
    issue_codes: list[str] = Field(default_factory=list)


class ExtractionExport(BaseModel):
    """Extraction provenance — how this output was produced.

    Built by ``ExportStage._build_extraction`` from the pipeline context; the
    export function passes it through verbatim. ``null`` when a Paper is
    exported outside the pipeline.
    """

    model_config = _STRICT

    # bibr *package* version (e.g. "0.3.0"), distinct from ``info.schema_version``.
    bibr_version: str
    build_sha: str | None = None
    ref_seg_strategy: str
    ref_parse_strategy: str
    references_complete: bool = True
    ref_seg_fallback_used: bool
    crossref_enrich: bool
    consolidate: str
    # Per-stage wall-clock seconds (export stage's own time excluded — it is
    # still running when this is built); ``null`` when timing was not recorded.
    timings: dict[str, float] | None = None
    total_seconds: float | None = None
    expected_identity: ExpectedIdentityExport | None = None
    identity_receipt: IdentityReceiptExport | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_identity(self, handler):
        data = handler(self)
        if self.expected_identity is None:
            data.pop("expected_identity", None)
        if self.identity_receipt is None:
            data.pop("identity_receipt", None)
        return data


class EnrichmentExport(BaseModel):
    """Reference-enrichment completeness — lets a consumer distinguish a
    partial (timed-out) Crossref enrichment from a complete one without
    grepping ``processing_warnings``. ``null`` when enrichment never ran."""

    model_config = _STRICT

    complete: bool
    refs_enriched: int
    refs_total: int


class ValidationIssueExport(BaseModel):
    """JSON-safe representation of a shared pipeline validation issue."""

    model_config = _STRICT

    code: str
    severity: str
    message: str
    origin_stage: str = "export"
    evidence_ids: list[str] = Field(default_factory=list)
    count: int
    blocking: bool = False


class ValidationExport(BaseModel):
    """Promotion disposition derived from typed validation issues."""

    model_config = _STRICT

    errors: int
    warnings: int
    blocking: int = 0
    promotable: bool = True
    issues: list[ValidationIssueExport]


class CitationCandidateExport(BaseModel):
    """Lossless JSON representation of one citation detector candidate."""

    model_config = _STRICT

    text_id: int
    start: int
    end: int
    raw: str
    style: str
    bib_ids: list[int]
    evidence: list[str]
    confidence: float
    accepted: bool
    rejection_reasons: list[str]


class CitationLinkingExport(BaseModel):
    """Evidence-bearing diagnostic receipt for inline citation linking."""

    model_config = _STRICT

    style_scores: dict[str, float]
    candidates: list[CitationCandidateExport]
    resolved_candidate_fraction: float | None
    unique_linked_bib_fraction: float | None


class CaptionCandidateExport(BaseModel):
    model_config = _STRICT

    caption_id: str
    text: str
    object_type: str
    page_number: int | None
    bbox: list[float] | None
    source_index: int


class CaptionAssignmentExport(BaseModel):
    model_config = _STRICT

    caption_id: str
    object_id: str | None
    score: float
    reasons: list[str]
    ambiguous: bool = False


class CaptionAssignmentReceiptExport(BaseModel):
    model_config = _STRICT

    candidates: list[CaptionCandidateExport]
    assignments: list[CaptionAssignmentExport]


class ReferenceSegmentationAttemptExport(BaseModel):
    model_config = _STRICT

    strategy: str
    spans: list[list[int]]
    credible_starts: int | None
    selected: bool
    reason_flags: list[str]


class ReferenceYieldExport(BaseModel):
    model_config = _STRICT

    credible_source_starts: int | None
    attempts: list[ReferenceSegmentationAttemptExport]
    selected_spans: list[list[int]]
    source_character_coverage: float | None
    parsed_count: int
    valid_count: int
    duplicate_rate: float
    reason_flags: list[str]


class PaperExport(BaseModel):
    """Pydantic model for the bibr v10.7 JSON export schema (reads v10.6 too)."""

    model_config = _STRICT

    paper_id: str | None = Field(
        description="Paper identifier: user-supplied --paper-id, else the DOI, else the source "
        "file name."
    )
    info: InfoExport = Field(
        description="Scalar paper-level metadata: title, abstract, keywords, DOI, file identity, "
        "schema/package versions, paper-type and OECD classification, the paper's own "
        "journal/venue identity, and research-integrity statement text."
    )
    author: list[AuthorExport] = Field(
        description="Extracted authors, with name, affiliation, email, corresponding-author flag, "
        "ORCID, and contribution roles."
    )
    text: list[TextExport] = Field(
        description="Full body text as an ordered array of sentence-level spans, each linked to "
        "its paragraph, section, and page."
    )
    section: list[SectionExport] = Field(
        description="Document section headers with hierarchy (level, parent) and IMRaD "
        "classification."
    )
    url: list[UrlExport] = Field(
        description="Hyperlinks found in the body text, with their link text and source sentence."
    )
    bib: list[BibExport] = Field(
        description="Parsed bibliography entries from the printed reference list. External "
        "enrichment stays separate unless fill/replace consolidation is explicitly enabled."
    )
    xref: list[XrefExport] = Field(
        description="In-text citation/cross-reference markers linking a sentence to a "
        "bibliography entry, table, figure, footnote, equation, or section."
    )
    citation_linking: CitationLinkingExport | None = Field(
        None,
        description="Citation detector scores plus every accepted/rejected source span; omitted "
        "when citation linking did not run.",
    )
    caption_assignment: CaptionAssignmentReceiptExport | None = Field(
        None,
        description="Document-wide caption candidates and deterministic ownership decisions.",
    )
    reference_yield: ReferenceYieldExport | None = Field(
        None,
        description="Reference segmentation attempts, selected spans, and parse-yield receipt.",
    )
    figure: list[FigureExport] = Field(
        description="Extracted figures with caption, page number, and (optionally) "
        "base64-encoded image data."
    )
    table: list[TableExport] = Field(
        description="Extracted tables with HTML markup, structured cell contents, caption, and "
        "page number."
    )
    eq: list[EqExport] = Field(
        description="Parsed statistical/mathematical expressions split into left-hand side, "
        "degrees of freedom, comparator, and right-hand side."
    )
    bib_match: list[BibMatchExport] = Field(
        [],
        description="Flattened enrichment matches (e.g. Crossref) for bibliography entries, one "
        "row per external-service hit.",
    )
    # Enrichment match for the paper's OWN identity (self-DOI lookup).
    info_match: list[InfoMatchExport] = Field(
        [],
        description="Enrichment matches for the paper's own bibliographic identity (self-DOI "
        "lookup), shaped like bib_match minus bib_id.",
    )
    # Structured funding parsed from the funding statement; empty by default.
    funding: list[FundingExport] = Field(
        [], description="Structured funder names and award IDs parsed from the funding statement."
    )
    # Structured affiliations parsed from the author byline; empty by default.
    affiliations: list[AffiliationExport] = Field(
        [],
        description="Structured author affiliations parsed from the byline: verbatim text plus "
        "best-effort institution/department/city/country and linked author IDs.",
    )
    ocr_config: OcrConfigExport | None = Field(
        None,
        description="OCR/LLM backend and model configuration used to produce this extraction; "
        "null when unavailable.",
    )
    # Reference-enrichment completeness; null when enrichment never ran.
    enrichment: EnrichmentExport | None = Field(
        None,
        description="Reference-enrichment completeness — how many references were enriched vs "
        "total; null when enrichment never ran.",
    )
    processing_warnings: list[str] = Field(
        [],
        description="Non-fatal warnings emitted during processing, including output-validation-"
        "gate findings.",
    )
    # Per-paper LLM token usage by model; null when no LLM ran or usage
    # tracking is disabled. Additive optional field.
    llm_usage: dict[str, dict[str, int]] | None = Field(
        None,
        description="Per-model LLM token usage counts for this extraction; null when no LLM ran "
        "or usage tracking is disabled.",
    )
    # Per-paper LLM token usage by call-site label; null when no LLM ran or
    # usage tracking is disabled. Additive optional field, sibling of llm_usage.
    llm_usage_by_label: dict[str, dict[str, int]] | None = Field(
        None,
        description="Per-call-site-label LLM token usage counts for this extraction (e.g. "
        "'extract_authors'); null when no LLM ran or usage tracking is disabled.",
    )
    # Deployment-qualification provenance surface (one object per paper); null
    # when no LLM ran or usage tracking is disabled. Read by an external gate.
    qualification_provenance: dict | None = Field(
        None,
        description="Deployment-qualification provenance: identity SHAs, per-task protocol "
        "hashes, native-validity + fallback outcome, and request counts; null when no LLM ran "
        "or usage tracking is disabled.",
    )
    # Extraction provenance; null when exported outside the pipeline.
    extraction: ExtractionExport | None = Field(
        None,
        description="Extraction provenance: producing bibr version, resolved reference seg/parse "
        "strategies, Crossref enrich/consolidate settings, and per-stage timings; null when "
        "exported outside the pipeline.",
    )
    native_source: dict[str, Any] | None = Field(
        default=None,
        alias="_native_source",
        description="Opt-in detached PDF character evidence, geometric ownership and raster "
        "coverage diagnostics; exported only with include_regions=True. Not a fidelity score.",
    )
    # Opt-in debug payload; underscore-prefixed in JSON, aliased here.
    regions: list[RegionExport] | None = Field(
        default=None,
        alias="_regions",
        description="Opt-in per-region layout debug payload (bbox, font, section linkage, raw "
        "content); omitted unless include_regions=True.",
    )
    # Output validation gate result ``{errors, warnings, issues[]}``; injected
    # after model_dump by the exporter (never set on the instance) and absent
    # from the output entirely when the gate is skipped (``validate=False``).
    validation: ValidationExport | None = Field(
        None,
        description="Output validation gate result: counts of errors/warnings and the structured "
        "issue list; absent when the gate was skipped.",
    )

    @model_serializer(mode="wrap")
    def _omit_absent_regions(self, handler):
        # ``_regions`` is opt-in: absent from the output entirely (not null)
        # unless the exporter populated it. ``validation`` is likewise injected
        # post-dump, so it is dropped here when unset.
        data = handler(self)
        if self.native_source is None:
            data.pop("_native_source", None)
            data.pop("native_source", None)
        if self.regions is None:
            data.pop("_regions", None)
            data.pop("regions", None)
        if self.validation is None:
            data.pop("validation", None)
        if self.citation_linking is None:
            data.pop("citation_linking", None)
        if self.caption_assignment is None:
            data.pop("caption_assignment", None)
        if self.reference_yield is None:
            data.pop("reference_yield", None)
        return data


def _bibr_version() -> str:
    """The producing bibr package version (import deferred: bibr/__init__ is
    lazy and importing it at module scope would be circular)."""
    import bibr

    return cast(str, bibr.__version__)


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


def _sane_export_links(links: list) -> list:
    """Filter *links* to those passing :func:`_is_sane_url`, logging drops."""
    kept = []
    for link in links:
        if _is_sane_url(link.url):
            kept.append(link)
        else:
            logger.debug("Dropping malformed URL from export: %r", link.url)
    return kept


def validate_export(data: dict) -> list[str]:
    """Validate export dict against the current v10.6/v10.7 Pydantic schema.

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

    Appends one ``VALIDATION:<severity>:<code>: <message>`` line per issue to
    ``processing_warnings`` and adds a structured top-level ``validation`` block.
    Never raises: a gate failure degrades to a single ``VAL_INTERNAL`` warning.
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

    warnings = list(payload.get("processing_warnings") or [])
    warnings.extend(f"VALIDATION:{i.severity}:{i.code}: {i.message}" for i in issues)
    payload["processing_warnings"] = warnings
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
    """Export Paper as a JSON-serializable dict matching the bibr v10.7 schema.

    Returns a dictionary conforming to the scienceverse paper.json schema,
    suitable for serialization with json.dumps().

    The dict is CONSTRUCTED through the ``PaperExport`` models — the schema
    is the builder, not a post-hoc checker — so any divergence raises
    ``pydantic.ValidationError`` at build time instead of shipping invalid
    output with a logged warning.

    ``include_regions`` controls whether the ``_regions`` debug payload
    (per-region layout: bbox, font, content, etc.) is emitted. Off by
    default — the field is large and not consumed
    by Metacheck or the standard paper schema.

    ``include_region_meta`` controls the per-text underscore fields
    (``_bbox_2d``, ``_font_size``, ``_region_type``, …) originally added as
    v4 training features. Off by default — they add substantial detail to the output
    and, like ``_regions``, not part of the Metacheck-facing API. When off
    the keys are omitted entirely; when on, the pre-v10.5 shape is emitted
    (keys always present, ``null`` when unavailable).

    ``validate`` (default on) runs the output validation gate
    (:mod:`bibr.export.validation`): it appends ``VALIDATION:*`` lines to
    ``processing_warnings`` and adds a top-level ``validation`` block. The gate
    never raises out of export.
    """
    if not paper.contents:
        raise ValueError("Paper has no contents")

    raw_fmt = paper.input_file.input_format
    input_fmt: str | None
    if raw_fmt is not None and hasattr(raw_fmt, "file_type"):
        input_fmt = raw_fmt.file_type
    elif raw_fmt is not None:
        input_fmt = str(raw_fmt)
    else:
        input_fmt = None
    # The pipeline stores enum member names ("PDF"); the schema contract is
    # lowercase ("pdf", "docx", "xml", "unknown").
    if input_fmt is not None:
        input_fmt = input_fmt.lower()

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
                editors=r.editors,
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
                url=r.url,
                is_in_press=is_in_press,
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
                        authors=_authors_to_dicts(m.authors),
                        editors=_authors_to_dicts(m.editors),
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

    # info_match: the paper's OWN identity enrichment, flattened the same way
    # bib_match flattens reference matches (minus bib_id).
    info_match_data = []
    for src, m in paper.metadata.match.items() if paper.metadata else []:
        try:
            d = m.model_dump()
            info_match_data.append(
                InfoMatchExport(
                    service=src.value,
                    service_id=d.get("id"),
                    score=d.get("score"),
                    bib_type=d.get("bib_type"),
                    doi=d.get("doi"),
                    title=d.get("title"),
                    authors=_authors_to_dicts(m.authors),
                    editors=_authors_to_dicts(m.editors),
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
            logger.debug("Failed to serialize info match for src=%s", src)

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
                source_region_ids=r.source_region_ids,
                native_spans=r.native_spans,
                formula_proposals=r.formula_proposals,
                bbox_height=r.bbox_height,
                bbox_width=r.bbox_width,
                char_density=r.char_density,
                estimated_line_height=r.estimated_line_height,
            )
            for r in paper.contents.region_summaries
        ]

    extraction_data = dict(paper.extraction) if paper.extraction is not None else None
    if extraction_data is not None:
        extraction_data["references_complete"] = not bool(
            paper.metadata and paper.metadata.references_incomplete
        )

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

    export = PaperExport(
        paper_id=paper._compute_paper_id(),
        info=InfoExport(
            title=(paper.metadata.title or None) if paper.metadata else None,
            abstract=abstract_text,
            keywords=exported_keywords,
            doi=(paper.metadata.doi or None) if paper.metadata else None,
            file_hash=paper.input_file.file_hash,
            input_format=input_fmt or "unknown",
            file_name=paper.input_file.file_name,
            schema_version=_SCHEMA_VERSION,
            bibr_version=_bibr_version(),
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
            text_quality=paper.text_quality,
        ),
        author=[
            AuthorExport(
                author_id=a.author_id,
                given=a.given,
                family=a.family,
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
            UrlExport(href=link.url, link_text=link.link_text, text_id=link.text_id)
            for link in _sane_export_links(paper.contents.links)
        ],
        bib=bib_data,
        xref=[
            XrefExport(
                xref_id=x.xref_id,
                xref_type=x.xref_type,
                contents=x.contents,
                text_id=x.text_id,
                tier=x.tier or None,
            )
            for x in paper.contents.xrefs
        ],
        citation_linking=citation_linking,
        caption_assignment=caption_assignment,
        reference_yield=reference_yield,
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
                text_id=eq.text_id,
                grp_id=eq.grp_id,
                lhs=eq.lhs,
                df=eq.df,
                comp=eq.comp,
                rhs=eq.rhs,
            )
            for eq in paper.contents.equations
        ],
        bib_match=bib_match_data,
        info_match=info_match_data,
        funding=[
            FundingExport(funder=f.funder, award_ids=f.award_ids)
            for f in (paper.metadata.funding if paper.metadata else [])
        ],
        affiliations=[
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
        ocr_config=paper.ocr_config,
        enrichment=enrichment_export,
        processing_warnings=list(paper.processing_warnings),
        llm_usage=paper.llm_usage or None,
        llm_usage_by_label=paper.llm_usage_by_label or None,
        qualification_provenance=paper.qualification_provenance or None,
        extraction=extraction_data,
        regions=regions_data,
        native_source=paper.contents.native_source if include_regions else None,
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
    """Build and validate the typed v10.7 export model for *paper*."""
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
    """Export *paper* as the additive v10.7 dictionary payload."""
    export = build_paper_export(
        paper,
        include_regions=include_regions,
        include_region_meta=include_region_meta,
        validate=validate,
    )
    return cast(dict[str, Any], export.model_dump(by_alias=True, exclude_unset=True))
