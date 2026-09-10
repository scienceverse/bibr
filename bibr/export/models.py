"""Pydantic models for the bibr JSON export schema — the single source of truth.

Every model rejects unknown keys so a divergence between the schema and the
hand-built dicts in ``bibr.export.json_export`` surfaces as a validation error
instead of being silently ignored. ``populate_by_name`` lets the
underscore-prefixed JSON keys (pydantic forbids them as field names) be
declared via aliases.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

_SCHEMA_VERSION = "11.0"


# ---------------------------------------------------------------------------
# Pydantic v11.0 export schema — single source of truth for validation
#
# v11.0 (vs 10.9) — BREAKING:
#   - ``schema_version`` moved to the ROOT. Its presence there is how readers
#     dispatch v11 vs every earlier version — never by parsing a value.
#   - New root ``source`` {file_name, file_hash, input_format}: the identity of
#     the input artifact, which was never paper-level metadata.
#   - ``info`` renamed to ``metadata`` (scalars only, minus the file identity,
#     the versions and ``text_quality``); ``info_match`` -> ``metadata_match``.
#   - ``info.text_quality`` moved to ``extraction.diagnostics.text_quality``.
#   - ``info.bibr_version`` removed: the producing package version lives in
#     ``extraction.bibr_version`` only. The pre-10.3 compat copy is gone.
#   - All telemetry consolidated under ``extraction``, replacing the root
#     ``ocr_config``, ``llm_usage``, ``llm_usage_by_label``, ``enrichment``,
#     ``processing_warnings``, ``citation_linking``, ``caption_assignment``,
#     ``reference_yield``, and ``_regions`` keys. ``extraction.ocr`` and
#     ``extraction.llm`` sit directly on ``extraction`` — there is no
#     ``models`` wrapper. Both are ``null`` (never omitted) when that engine
#     did not run this request; ``extraction.ocr`` is always ``null`` for
#     natively-parsed formats (docx, xml, html, htm, epub).
#   - New ``extraction.completed_at`` (UTC ISO-8601 export timestamp) and
#     ``extraction.usage`` {totals, breakdown}, where each ``breakdown`` row
#     is keyed by the ``(label, provider, model)`` triple and is authoritative
#     for every engine that actually ran — ``extraction.llm`` names only the
#     CORE extraction engine. New opt-in ``extraction.trace`` (behind
#     ``LLM_CAPTURE_TRACE``, default off): captured only for
#     ``InstructorBackend`` calls — the NuExtract-native structured backend
#     is not yet instrumented and logs a warning instead of emitting rows.
#   - Validation-gate findings (``VALIDATION:<severity>:<code>: <message>``)
#     are no longer mirrored into ``extraction.warnings``; they live only in
#     ``validation.issues``.
#   - ``xref[].xref_id`` renamed to ``xref[].target_id``: it is a foreign key
#     to the target table per ``xref_type`` (nullable when unresolved), never
#     this row's own primary key — the old name read as one.
#   - ``eq[]`` gains ``eq_id`` (1-based position; the only record table that
#     previously had no primary key) and ``verbatim`` (the printed expression
#     as matched; null until the equation parser populates it).
#   - ``funding[]`` gains ``funding_id`` (1-based position; same no-PK gap).
#   - Root ``affiliations`` renamed to ``affiliation`` — singular-table rule,
#     matching every other root record-array key.
#   - Naming rule (CSL-borrowed): plural = verbatim printed string, singular =
#     derived structured array. ``bib[].authors``/``editors`` stay exactly as
#     printed (unchanged, gold-aligned); new ``bib[].author``/``editor`` carry
#     the best-effort split into ``{family, given, suffix}`` or a
#     ``{literal}`` fallback (see ``bibr/export/name_split.py``).
#     ``bib_match[]``/``metadata_match[].authors``/``editors`` renamed to
#     ``author``/``editor`` for the same reason (they were always structured).
#     ``author[].suffix`` added to paper authors for parity. All four
#     structured name fields (``bib[]`` and both match tables) share ONE
#     absence encoding: ``null`` when there was nothing to split, never ``[]``.
#   - ``table[].contents`` / ``table[].parts[].contents`` typed as
#     ``list[list[str]]`` instead of bare ``list`` — the builder
#     (``PaperTable``/``PaperTablePart``) stringifies every cell.
#   - Forward policy: 11.x is additive-only — new fields may appear in any
#     11.x release and readers must ignore keys they don't recognize. Any
#     rename, move, removal, or type change of an existing field requires a
#     new major version (12.0) and a coordinated release across bibr,
#     sv/schema, and metacheck, the same way 11.0 was coordinated. Readers
#     dispatch on the *presence* of a root ``schema_version`` key, never on
#     parsing its value — pre-v11 payloads (and metacheck's fixture corpus)
#     have no such key at all, so a value-parsing rule would strand them.
#
# v10.7 (vs 10.6) — additive:
#   - Figure/table ``parts`` preserve every physical payload and provenance.
#   - Optional ``caption_assignment`` and ``reference_yield`` diagnostic receipts.
#   - Typed readers accepted v10.6 payloads without these fields (until v11.0,
#     which is a clean break and reads 11.0 only).
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
#     mirroring ``_regions``. Off by default: ≈19% of output bytes of
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


_STRICT = ConfigDict(extra="forbid", populate_by_name=True)


class PersonNameExport(BaseModel):
    """One structured person name.

    ``literal`` holds corporate/unsplittable names whole (the JATS
    ``string-name`` analog); when it is set, ``family``/``given`` are absent.
    """

    model_config = _STRICT

    family: str | None = None
    given: str | None = None
    suffix: str | None = None
    literal: str | None = None

    @model_validator(mode="after")
    def _requires_an_identity(self) -> PersonNameExport:
        # A person record identifying nobody (all fields None/absent) is not
        # a best-effort split, it's an empty row — reject it rather than
        # letting it validate clean and serialize to `{}`. Reachable today
        # only via externally-sourced enrichment-sidecar replay rows
        # (bibr/pipeline/artifacts.py), never from bibr's own producers.
        if self.family is None and self.given is None and self.literal is None:
            raise ValueError("PersonNameExport requires at least one of family/given/literal")
        return self

    @model_serializer(mode="wrap")
    def _omit_absent_parts(self, handler):
        data = handler(self)
        return {k: v for k, v in data.items() if v is not None}


class SourceExport(BaseModel):
    """Identity of the input artifact — not of the paper."""

    model_config = _STRICT

    file_name: str
    file_hash: str
    # Contractually lowercase: "pdf", "docx", "xml", "unknown".
    input_format: str


class MetadataExport(BaseModel):
    """Scalar paper-level metadata.

    Scalars only: nested objects/lists here break R consumers that call
    ``as.data.frame(metadata)`` (metacheck's ``.read_bibr``). Processing
    metadata lives under ``extraction`` instead; file identity under ``source``.
    """

    model_config = _STRICT

    title: str | None
    abstract: str | None = None
    keywords: list[str]
    doi: str | None
    paper_type: str | None = None
    paper_type_confidence: float | None = None
    oecd_l1: str | None = None
    oecd_l2: str | None = None
    oecd_confidence: float | None = None
    # The paper's OWN bibliographic self-identity, verbatim from the front
    # matter. Scalars only; the enrichment match lives in ``metadata_match``.
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


class AuthorExport(BaseModel):
    model_config = _STRICT

    author_id: int
    given: str
    family: str
    suffix: str | None = None
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
    # Naming rule (CSL-borrowed): plural = the verbatim printed string
    # (canonical, gold-aligned); singular = the derived structured list.
    #
    # The structured lists are nullable, not empty-list-defaulted, so they use
    # the SAME absence encoding as the identically-named ``bib_match[]`` /
    # ``metadata_match[]`` fields and as their own verbatim sibling: nothing to
    # split yields ``null``, never ``[]``. In R the two encodings are different
    # column types, and the versioning policy would make fixing a mismatch a
    # 12.0-only change once shipped.
    authors: str | None = None
    author: list[PersonNameExport] | None = None
    editors: str | None = None
    editor: list[PersonNameExport] | None = None
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
    arxiv: str | None = None
    pmid: str | None = None
    series: str | None = None
    access_date: str | None = None
    note: str | None = None
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
    author: list[PersonNameExport] | None = None
    editor: list[PersonNameExport] | None = None
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


class MetadataMatchExport(BaseModel):
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
    author: list[PersonNameExport] | None = None
    editor: list[PersonNameExport] | None = None
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

    funding_id: int  # 1-based position; this table has no other primary key
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

    # Foreign key to the target table (bib_id/table_id/figure_id/... per
    # xref_type). Nullable: an xref need not resolve to a target. Named
    # target_id, not xref_id — it was never this row's primary key (this
    # table has no primary key of its own).
    target_id: int | None
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
    # ``PaperTable.contents`` (bibr/paper_contents.py:422) stringifies every
    # cell — headers via ``str(c)``, data cells via ``str(value)`` — so the
    # runtime shape is always a list of string rows, never numeric/None
    # cells. Matches the property's own ``-> list[list[str]]`` annotation.
    contents: list[list[str]]
    caption: str | None = None
    page_number: int | None
    parts: list[TablePartExport] = []


class TablePartExport(BaseModel):
    model_config = _STRICT

    part_index: int
    html: str | None = None
    # Same stringify-every-cell guarantee as ``TableExport.contents`` — see
    # ``PaperTablePart.contents`` (bibr/paper_contents.py:398).
    contents: list[list[str]]
    page_number: int | None = None
    bbox: list[float] | None = None
    provenance: list[ProvenanceExport] = []


class EqExport(BaseModel):
    model_config = _STRICT

    # This table has no other primary key; position within the export order
    # (1-based) is a stable enough surrogate since the equation list order is
    # itself deterministic per paper.
    eq_id: int
    text_id: int
    grp_id: int
    # The printed expression as matched, verbatim. lhs/df/comp/rhs are lossy
    # parsed pieces; this is the ground-truth string. Named verbatim rather
    # than text to avoid confusion with the adjacent text_id. Null until the
    # equation parser populates it (not yet wired as of v11.0 — the slot
    # ships now because a later addition would need another major bump).
    verbatim: str | None = None
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


class EnrichmentExport(BaseModel):
    """Reference-enrichment completeness — lets a consumer distinguish a
    partial (timed-out) Crossref enrichment from a complete one without
    grepping ``extraction.warnings``. ``null`` when enrichment never ran."""

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


class OcrEngineExport(BaseModel):
    """Which OCR engine produced this output. ``null`` on the DOCX-native path."""

    model_config = _STRICT

    backend: str | None = None
    model: str | None = None
    profile: Literal["paddle", "glm"] | None = None


class LlmEngineExport(BaseModel):
    """The CORE extraction LLM. ``null`` when the run deliberately had none.

    ``usage.breakdown`` is authoritative for every engine that actually ran;
    when the two disagree (multi-engine run), breakdown wins.
    """

    model_config = _STRICT

    provider: str | None = None
    model: str | None = None
    backend: str | None = None


class ExtractionSettingsExport(BaseModel):
    """Per-run resolutions of the reference/enrichment knobs."""

    model_config = _STRICT

    ref_seg: str
    ref_parse: str
    crossref_enrich: bool
    consolidate: str


class TimingsExport(BaseModel):
    """Per-stage wall-clock seconds (the export stage's own time is excluded —
    it is still running when this is built).

    Both fields are always populated when this object exists: an untimed run
    omits the whole ``extraction.timings`` object rather than emitting one with
    null fields (``ExportStage._build_extraction`` passes ``None``, which
    ``ExtractionExport._omit_absent`` then drops). The ``| None`` types are
    structural only.
    """

    model_config = _STRICT

    stages: dict[str, float] | None = None
    total_seconds: float | None = None


class UsageRowExport(BaseModel):
    """One ``(label, provider, model)`` LLM usage row."""

    model_config = _STRICT

    label: str
    provider: str | None = None
    model: str | None = None
    calls: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int


class UsageTotalsExport(BaseModel):
    """Aggregate of every ``breakdown`` row — denormalized deliberately."""

    model_config = _STRICT

    calls: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int


class UsageExport(BaseModel):
    model_config = _STRICT

    totals: UsageTotalsExport
    breakdown: list[UsageRowExport] = Field(default_factory=list)


class IdentityExport(BaseModel):
    """Caller-supplied expectations and the DOI-selection receipt."""

    model_config = _STRICT

    expected: ExpectedIdentityExport | None = None
    receipt: IdentityReceiptExport | None = None

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler):
        data = handler(self)
        for key in ("expected", "receipt"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class LlmTraceExport(BaseModel):
    """One LLM call's captured prompt and raw response.

    Capture is opt-in via ``LLM_CAPTURE_TRACE`` (default off). Traces are joined
    to the extraction by paper ID and to usage breakdowns by label, provider,
    and model. Credential-shaped tokens are scrubbed before export; document
    text is still present, so callers control where trace-bearing exports go.

    Only the Instructor backend currently captures traces. The NuExtract-native
    backend logs a warning when tracing is requested and emits no trace rows.
    Only successful physical attempts are captured: validation re-asks inside
    Instructor and transient failures in the outer retry loop are absent.
    Consequently ``attempt`` is currently always 1; failure fields do not
    constitute a complete retry history.
    """

    model_config = _STRICT

    label: str
    provider: str | None = None
    model: str | None = None
    messages: list[dict] = Field(default_factory=list)
    raw_completion: str | None = None
    parsed_ok: bool
    finish_reason: str | None = None
    # Resolved sampling parameters (temperature/top_p/reasoning_effort/...,
    # provider-shaped — e.g. the Google adapter nests temperature under
    # generation_config). Mandatory in spirit: a completion recorded without
    # them cannot be told apart from a lucky generation, so the call site
    # always populates this from ``_build_call_kwargs``'s resolved output.
    # Untyped ``dict`` (pydantic does not validate its values), so
    # ``_record_trace`` recursively coerces every leaf to a JSON-safe
    # primitive and scrubs strings before this is ever constructed — see
    # ``bibr.clients.llm._sanitize_trace_value``.
    params: dict = Field(default_factory=dict)
    attempt: int = 1
    error: str | None = None


class DiagnosticsExport(BaseModel):
    """Outcome flags and stage receipts — never raw payloads.

    ``regions`` and ``trace`` are deliberately NOT here: they are megabyte-scale
    intermediate data, and bundling them would force every diagnostics consumer
    to reason about size.
    """

    model_config = _STRICT

    text_quality: float | None = None
    references_complete: bool = True
    ref_seg_fallback_used: bool = False
    citation_linking: CitationLinkingExport | None = None
    caption_assignment: CaptionAssignmentReceiptExport | None = None
    reference_yield: ReferenceYieldExport | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_receipts(self, handler):
        data = handler(self)
        for key in ("citation_linking", "caption_assignment", "reference_yield"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class ExtractionExport(BaseModel):
    """Extraction provenance — how this output was produced.

    Built by ``ExportStage._build_extraction`` from the pipeline context; the
    export function passes it through verbatim. Omitted entirely when a Paper is
    exported outside the pipeline.
    """

    model_config = _STRICT

    # bibr *package* version (e.g. "0.3.0"), distinct from the schema version.
    bibr_version: str
    build_sha: str | None = None
    # UTC ISO-8601 export timestamp — the export's only wall-clock provenance.
    # Excluded from fixture/replay diffs (see tests) so it stays deterministic.
    completed_at: str
    ocr: OcrEngineExport | None = None
    llm: LlmEngineExport | None = None
    settings: ExtractionSettingsExport
    timings: TimingsExport | None = None
    usage: UsageExport | None = None
    enrichment: EnrichmentExport | None = None
    identity: IdentityExport | None = None
    diagnostics: DiagnosticsExport | None = None
    warnings: list[str] = Field(default_factory=list)
    # Opt-in heavy payloads — siblings, never nested under a hot key.
    regions: list[RegionExport] | None = None
    trace: list[LlmTraceExport] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler):
        # Absence rule: omitted = the subsystem did not run / was not requested.
        # ``ocr``/``llm`` are NOT in this list — ``null`` there is meaningful
        # (the run happened, deliberately without that engine).
        data = handler(self)
        for key in (
            "enrichment",
            "identity",
            "diagnostics",
            "timings",
            "usage",
            "regions",
            "trace",
        ):
            if data.get(key) is None:
                data.pop(key, None)
        return data


# The ONLY root keys the exporter may omit (see ``PaperExport._omit_absent``).
# Every other root key is always emitted — record arrays because the
# uniform-tables contract requires them (empty allowed), the scalar/object
# blocks because they are unconditional. This tuple is the single authority:
# ``bibr.export.schema_artifact`` derives the published schema's ``required``
# list from it, so the artifact can never claim a key is optional that the
# exporter in fact always emits.
OMITTABLE_ROOT_KEYS: tuple[str, ...] = ("validation", "extraction")


class PaperExport(BaseModel):
    """Pydantic model for the bibr v11.0 JSON export schema.

    v11 is a clean break: no v10 payload validates against this model, and none
    is meant to. Readers dispatch on the presence of the root ``schema_version``.
    """

    model_config = _STRICT

    paper_id: str | None = Field(
        description="Paper identifier: user-supplied --paper-id, else the DOI, else the source "
        "file name."
    )
    schema_version: Literal["11.0"] = Field(
        description="Export schema version. Its presence at the root is how readers "
        "distinguish v11 from all earlier versions."
    )
    source: SourceExport = Field(
        description="Identity of the input artifact: file name, content hash, and format."
    )
    metadata: MetadataExport = Field(
        description="Scalar paper-level metadata: title, abstract, keywords, DOI, paper-type "
        "and OECD classification, the paper's own journal/venue identity, and "
        "research-integrity statement text."
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
        description="Parsed bibliography entries, verbatim from the printed reference list — "
        "never backfilled from external enrichment."
    )
    xref: list[XrefExport] = Field(
        description="In-text citation/cross-reference markers linking a sentence to a "
        "bibliography entry, table, figure, footnote, equation, or section."
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
    metadata_match: list[MetadataMatchExport] = Field(
        [],
        description="Enrichment matches for the paper's own bibliographic identity (self-DOI "
        "lookup), shaped like bib_match minus bib_id.",
    )
    # Structured funding parsed from the funding statement; empty by default.
    funding: list[FundingExport] = Field(
        [], description="Structured funder names and award IDs parsed from the funding statement."
    )
    # Structured affiliations parsed from the author byline; empty by default.
    # Root key is singular ("affiliation", not "affiliations"), matching every
    # other root record-array key (author, bib, figure, table, ...) —
    # singular-table rule; the model class stays ``AffiliationExport``.
    affiliation: list[AffiliationExport] = Field(
        [],
        description="Structured author affiliations parsed from the byline: verbatim text plus "
        "best-effort institution/department/city/country and linked author IDs.",
    )
    # Deployment-qualification provenance surface (one object per paper). Always
    # present: ``null`` (never omitted) when no LLM ran or usage tracking is
    # disabled, because an external gate reads the key by subscript.
    qualification_provenance: dict | None = Field(
        None,
        description="Deployment-qualification provenance: identity SHAs, per-task protocol "
        "hashes, native-validity + fallback outcome, and request counts. Always present; null "
        "(never omitted) when no LLM ran or usage tracking is disabled.",
    )
    # Extraction provenance; omitted when exported outside the pipeline.
    extraction: ExtractionExport | None = Field(
        None,
        description="Extraction provenance: how this output was produced — engines, per-run "
        "settings, timings, LLM usage, enrichment, identity receipts, diagnostics and warnings; "
        "omitted when exported outside the pipeline.",
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
    def _omit_absent(self, handler):
        # Root record arrays are ALWAYS present (empty allowed) — metacheck's
        # uniform-tables contract needs every table to exist. Only the optional
        # objects are omitted.
        #
        # ``qualification_provenance`` is deliberately NOT in this list even
        # though the absence rule would put it there: the external qualification
        # runner reads it by subscript (see bibr/export/qualification_provenance
        # .py), so omitting it would be a KeyError on every payload where no LLM
        # ran. It stays ``null``. Changing that is its own announced change,
        # agreed with the gate owner.
        data = handler(self)
        for key in OMITTABLE_ROOT_KEYS:
            if data.get(key) is None:
                data.pop(key, None)
        return data
