"""JSON export for Paper objects — v12.0 schema."""

from __future__ import annotations

import base64
import logging
import re
from typing import TYPE_CHECKING, Any, cast, get_args
from urllib.parse import urlsplit

from pydantic import ValidationError

from bibr.export.geometry import PageGeometry

# ``bibr.export.models`` is the canonical home for every export model, and
# ``bibr.export`` re-exports the public ones straight from there. Every name
# below is used inside this module except the two marked ``noqa: F401``, which
# are kept only because importers still reach them through this module path.
from bibr.export.models import (
    _SCHEMA_VERSION,
    COUNTRY_CODE_PATTERN,
    DOI_PATTERN,
    ISO_DATE_PATTERN,
    ORCID_PATTERN,
    ROR_PATTERN,
    AffiliationExport,
    AffiliationMatchExport,
    AuthorExport,
    BibExport,
    BibMatchExport,
    BibTypeLiteral,
    CaptionAssignmentExport,
    CaptionAssignmentReceiptExport,
    CaptionCandidateExport,
    CitationCandidateExport,
    CitationLinkingExport,
    EnrichmentExport,
    EqCompLiteral,
    EqExport,
    FieldStatesExport,
    FigureExport,
    FloatPartExport,
    FundingExport,
    FundingMatchExport,
    InputFormatLiteral,
    LlmEngineExport,  # noqa: F401 - re-exported for existing importers
    MetadataExport,
    MetadataMatchExport,
    OcrEngineExport,  # noqa: F401 - re-exported for existing importers
    OecdL1Literal,
    OecdL2Literal,
    PaperClassificationExport,
    PaperExport,
    PaperTypeLiteral,
    ReferenceSegmentationAttemptExport,
    ReferenceYieldExport,
    RegionExport,
    SectionClassificationExport,
    SectionExport,
    SectionTypeLiteral,
    SeverityLiteral,
    SourceExport,
    TableExport,
    TextExport,
    TextRegionExport,
    UrlExport,
    ValidationExport,
    ValidationIssueExport,
    XrefExport,
    XrefTierExport,
    XrefTierLiteral,
)
from bibr.export.normalize import arxiv_id, credit_roles, iso_date, license_ids
from bibr.export.spans import SpanLocator, equation_span, url_span, xref_span
from bibr.export.structure_ids import ExportIds, export_ids
from bibr.extract.research_integrity import collect_affiliations
from bibr.models import ORGANIZATION_ROLE, BibType, canonicalize_orcid, migrate_bib_type
from bibr.processing_warnings import ProcessingWarning
from bibr.utils.text import normalize_doi
from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from bibr.paper import Paper


logger = logging.getLogger(__name__)

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

_BIB_TYPES = frozenset(BibType)
_PAPER_TYPES = frozenset(get_args(PaperTypeLiteral))
_OECD_L1 = frozenset(get_args(OecdL1Literal))
_OECD_L2 = frozenset(get_args(OecdL2Literal))
_INPUT_FORMATS = frozenset(get_args(InputFormatLiteral))
_EQ_COMPS = frozenset(get_args(EqCompLiteral))
_XREF_TIERS = frozenset(get_args(XrefTierLiteral))

# Runtime labels spelled differently in the published vocabularies. The
# vocabulary names the format, not the file extension; bibr reads XML only as
# JATS (``SupportedFileType.XML``).
_EXPORT_INPUT_FORMATS = {"xml": "jats", "htm": "html"}
_EXPORT_SECTION_TYPES = {"open_data": "data_availability"}

# Spellings the equation extractor's LLM path may return for a comparator.
_COMP_SPELLINGS = {
    "<=": "≤",
    ">=": "≥",
    "=<": "≤",
    "=>": "≥",
    "⩽": "≤",
    "⩾": "≥",
    "<<": "≪",
    ">>": "≫",
    "!=": "≠",
    "==": "=",
    "∼": "~",
    "≃": "≈",
    "≅": "≈",
}

_DOI_RE = re.compile(DOI_PATTERN)
_ORCID_RE = re.compile(ORCID_PATTERN)
_ROR_RE = re.compile(ROR_PATTERN)
_COUNTRY_CODE_RE = re.compile(COUNTRY_CODE_PATTERN)
_ISO_DATE_RE = re.compile(ISO_DATE_PATTERN)


def _export_bib_type(value: str | None) -> BibTypeLiteral | None:
    """A ``BibType`` value for the schema enum, or ``None`` when absent.

    Canonical values pass through; a legacy or foreign type string (BibTeX
    ``article``, Crossref ``journal-article``) maps through
    :func:`migrate_bib_type`, which sends anything unrecognized to ``other``.
    """
    if not value:
        return None
    return cast(BibTypeLiteral, value if value in _BIB_TYPES else migrate_bib_type(value))


def _in_vocabulary(value: str | None, vocabulary: frozenset[str], field: str) -> str | None:
    """*value* when it belongs to *vocabulary*; ``None`` (logged) otherwise.

    A classifier label outside the published enum would fail the strict export
    model and with it the whole paper, so it is dropped instead.
    """
    if not value:
        return None
    if value in vocabulary:
        return value
    logger.warning("Dropping off-vocabulary %s %r from export", field, value)
    return None


def _or_none(value: str | None) -> str | None:
    """``None`` for an empty or whitespace-only string: absence is ``null``, never ``""``."""
    return value if value and value.strip() else None


def _snake(label: str | None) -> str | None:
    """A runtime label in the published snake_case spelling (``meta-analysis``
    -> ``meta_analysis``)."""
    return label.replace("-", "_") if label else label


def _conformed(value: str | None, pattern: re.Pattern[str], field: str) -> str | None:
    """*value* when it has the published format; ``None`` (logged) otherwise.

    For identifiers that reach the export from outside bibr (a registry record,
    a publisher's metadata): one malformed value must not fail the export.
    """
    if value is None:
        return None
    # ``fullmatch``: Python's ``$`` also matches before a final newline, which
    # the models' patterns reject.
    if pattern.fullmatch(value):
        return value
    logger.warning("Dropping malformed %s %r from export", field, value)
    return None


def _export_doi(value: str | None, field: str) -> str | None:
    """*value* as a bare, lowercase DOI, or ``None``.

    DOIs are case-insensitive; one spelling lets the paper's own DOI, its
    references' DOIs and the registries' records join.
    """
    if not value or not value.strip():
        return None
    bare = normalize_doi(value) or value.strip()
    return _conformed(bare.lower(), _DOI_RE, field)


def _export_service_id(value: str | None) -> str | None:
    """A match row's ``service_id``, with a DOI identifier lowercased like every DOI."""
    if value and normalize_doi(value) == value.strip():
        return value.strip().lower()
    return value


def _unit_interval(value: float | None, field: str) -> float | None:
    """*value* when it lies in [0, 1]; ``None`` (logged) otherwise."""
    if value is None or 0 <= value <= 1:
        return value
    logger.warning("Dropping out-of-range %s %r from export", field, value)
    return None


def _match_score(value: float | None) -> float | None:
    """A match score on the published 0–1 scale.

    Enrichment scores a match 0–100 (100 for a DOI lookup, a fuzzy title
    similarity for a search hit); every score in the export is 0–1.
    """
    if value is None:
        return None
    return round(min(max(float(value) / 100.0, 0.0), 1.0), 4)


def _export_comp(value: str) -> EqCompLiteral | None:
    """The comparator in the published spelling, or ``None`` when it is none of them."""
    comp = _COMP_SPELLINGS.get(value.strip(), value.strip())
    return cast(EqCompLiteral, comp) if comp in _EQ_COMPS else None


def _data_uri(image_b64: str | None) -> str | None:
    """A base64 image as a ``data:`` URI naming its media type.

    Figure images are JPEG crops, PNG or JPEG composites, or the image a DOCX
    embeds (any format Word takes), so the type is read from the bytes.
    """
    if not image_b64:
        return None
    if image_b64.startswith("data:"):
        return image_b64
    try:
        head = base64.b64decode(image_b64[:64] + "=" * (-len(image_b64[:64]) % 4))
    except (ValueError, TypeError):
        head = b""
    return f"data:{_image_media_type(head)};base64,{image_b64}"


def _image_media_type(head: bytes) -> str:
    """The media type of an image from its first bytes."""
    signatures = (
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"GIF87a", "image/gif"),
        (b"GIF89a", "image/gif"),
        (b"II*\x00", "image/tiff"),
        (b"MM\x00*", "image/tiff"),
        (b"BM", "image/bmp"),
        (b"\xd7\xcd\xc6\x9a", "image/wmf"),
    )
    for signature, media_type in signatures:
        if head.startswith(signature):
            return media_type
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:4] == b"\x01\x00\x00\x00" and head[40:44] == b" EMF":
        return "image/emf"
    if head.lstrip().startswith((b"<svg", b"<?xml")):
        return "image/svg+xml"
    return "application/octet-stream"


def _export_author(author) -> AuthorExport:
    """One ``author[]`` row: a person's split name, or a group author's ``literal``."""
    roles = list(author.role or [])
    given, family, literal = _or_none(author.given), _or_none(author.family), None
    if ORGANIZATION_ROLE in roles:
        roles = [role for role in roles if role != ORGANIZATION_ROLE]
        literal = " ".join(part for part in (given, family) if part) or None
        given = family = None
    return AuthorExport(
        author_id=author.author_id,
        given=given,
        family=family,
        suffix=getattr(author, "suffix", None),
        literal=literal,
        email=author.email,
        corresponding=author.corresponding,
        orcid=_conformed(canonicalize_orcid(author.orcid), _ORCID_RE, "orcid"),
        role=roles,
        credit_roles=credit_roles(roles),
    )


def _organization_fields(org) -> dict[str, Any]:
    """The ROR organization fields shared by ``affiliation_match`` and ``funding_match``."""
    code = org.country_code.upper() if org.country_code else None
    return {
        "service_id": org.service_id,
        "score": _unit_interval(org.score, "ROR score"),
        "name": org.name,
        "country_code": _conformed(code, _COUNTRY_CODE_RE, "country_code"),
    }


def export_section_type(value: str | None) -> str | None:
    """A runtime section type in the published spelling (``open_data`` ->
    ``data_availability``), wherever the export names one."""
    return _EXPORT_SECTION_TYPES.get(value, value) if value else value


def _export_input_format(fmt: str | None) -> InputFormatLiteral:
    """The published ``input_format`` for a runtime file type."""
    exported = _EXPORT_INPUT_FORMATS.get(fmt or "", fmt)
    return cast(InputFormatLiteral, exported if exported in _INPUT_FORMATS else "unknown")


def _bib_published_date(printed_date: str | None, year: int | None) -> str | None:
    """ISO 8601 date of a reference: its printed date when that agrees with its
    year ("2020, May 3"), else the year alone."""
    iso = iso_date(printed_date)
    if year and 0 < year <= 9999 and (iso is None or not iso.startswith(f"{year:04d}")):
        return f"{year:04d}"
    return iso


def _export_affiliations(paper: Paper) -> list[AffiliationExport]:
    """The ``affiliation[]`` table, built from the author byline strings.

    Every distinct verbatim component of ``PaperAuthor.affiliation`` becomes a
    row, whether or not the LLM parse ran; the parse (``metadata.affiliations``)
    only fills the structured components of the row with the same text. Parsed
    rows with no matching byline string (natively parsed inputs) are kept after
    the byline rows.
    """
    metadata = paper.metadata
    if metadata is None:
        return []
    texts, author_ids = collect_affiliations(metadata.authors)
    parsed = {a.text: a for a in metadata.affiliations}
    rows: list[tuple[str, list[int]]] = list(zip(texts, author_ids, strict=True))
    seen = set(texts)
    rows.extend((a.text, list(a.author_ids)) for a in metadata.affiliations if a.text not in seen)
    exported = []
    for position, (text, ids) in enumerate(rows, start=1):
        parse = parsed.get(text)
        exported.append(
            AffiliationExport(
                affiliation_id=position,
                text=text,
                institution=parse.institution if parse else None,
                department=parse.department if parse else None,
                city=parse.city if parse else None,
                country=parse.country if parse else None,
                author_ids=ids,
            )
        )
    return exported


def bibr_producer(build_sha: str | None = None) -> dict[str, Any]:
    """``extraction.producer`` for an export this bibr writes."""
    import bibr

    return {"name": "bibr", "version": bibr.__version__, "build_sha": build_sha}


def _minimal_extraction() -> dict:
    """The ``extraction`` skeleton for a Paper exported outside the pipeline:
    no engines or settings are known, only the producer and the export time."""
    import datetime as _dt

    return {
        "producer": bibr_producer(),
        "completed_at": _dt.datetime.now(_dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    }


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


# Schemes a browser runs as script (or as an inline document) when the link is
# followed. Link targets come from untrusted markup — HTML ``<a href>``, JATS
# ``ext-link``/``uri``, DOCX hyperlinks — and readers render ``url[].href`` and
# ``bib[].url`` as anchors. Everything else passes: the real corpora carry
# ``info:``, ``ncbi-n:``, ``pdb:``, ``arxiv:``, ``mailto:`` and ``tel:`` links.
_ACTIVE_URL_SCHEMES = frozenset({"javascript", "vbscript", "data"})
_URL_SCHEME_RE = re.compile(r"([a-z][a-z0-9+.\-]*):", re.IGNORECASE)
# What a browser's URL parser ignores before reading the scheme.
_URL_LEADING_JUNK = "".join(map(chr, range(0x21)))


def _has_active_scheme(url: str) -> bool:
    """Whether following *url* would run script.

    Reads the scheme as a browser does — leading control characters and spaces
    stripped, tabs and newlines dropped anywhere, case ignored — and never
    raises, so a URL malformed elsewhere is judged by its scheme alone.
    """
    cleaned = url.lstrip(_URL_LEADING_JUNK).translate({9: None, 10: None, 13: None})
    match = _URL_SCHEME_RE.match(cleaned)
    return match is not None and match.group(1).lower() in _ACTIVE_URL_SCHEMES


def _without_active_scheme(url: str | None) -> str | None:
    """*url*, or ``None`` when its scheme would run script in a browser."""
    if url and _has_active_scheme(url):
        logger.debug("Dropping script-capable URL from export: %r", url)
        return None
    return url


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
    repaired form, not its raw (possibly truncated-looking) one. A link whose
    scheme would run script is dropped too.
    """
    kept = []
    for link in links:
        href = _normalize_export_url(link.url)
        if not _is_sane_url(href):
            logger.debug("Dropping malformed URL from export: %r", link.url)
        elif _without_active_scheme(href) is not None:
            kept.append(link)
    return kept


def validate_export(data: dict) -> list[str]:
    """Validate export dict against the current v12.0 Pydantic schema.

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

    Adds a structured ``extraction.validation`` block (creating ``extraction``
    for a bare payload). Gate findings are NOT
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
    extraction = payload.get("extraction")
    if not isinstance(extraction, dict):
        extraction = payload["extraction"] = {}
    extraction["validation"] = ValidationExport(
        errors=sum(1 for issue in issues if issue.severity == IssueSeverity.ERROR),
        warnings=sum(1 for issue in issues if issue.severity == IssueSeverity.WARNING),
        blocking=blocking,
        promotable=blocking == 0,
        issues=[
            ValidationIssueExport(
                code=issue.code,
                severity=cast(SeverityLiteral, str(issue.severity)),
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


def append_payload_warning(payload: dict, warning: ProcessingWarning) -> dict:
    """Append a non-fatal warning to an already-serialized payload.

    Warnings live at ``extraction.warnings``. Post-export mutators
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
    row = warning.to_dict()
    if row not in warnings:
        warnings.append(row)
    extraction["warnings"] = warnings
    return payload


def _field_states(paper: Paper, *, present: dict[str, bool], warnings: list) -> FieldStatesExport:
    """``extraction.fields``: the state of each tracked field of *paper*.

    *present* says which fields have an exported value; *warnings* are the
    export's merged warning rows.
    """
    from bibr.field_states import FieldScope, build_field_states

    decisions = getattr(paper, "field_decisions", None)
    selection = paper.doi_selection
    records = build_field_states(
        present=present,
        sources=decisions.sources() if decisions is not None else {},
        scope=paper.field_scope or FieldScope(),
        issues=paper.validation_issues,
        warnings=[ProcessingWarning.from_dict(row) for row in warnings],
        doi_selected=selection is not None and selection.selected is not None,
        rules=decisions.rules() if decisions is not None else None,
    )
    return FieldStatesExport.model_validate(
        {field: record.to_dict() for field, record in records.items()}
    )


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


def _export_float_ref(object_id: str | None, ids: ExportIds) -> str | None:
    """A caption receipt's ``figure:<id>``/``table:<id>``, with the float's export id.

    None when that float did not survive to the export: the receipt is frozen
    before floats are merged, so an id can name a float that no longer exists.
    """
    kind, _, internal = (object_id or "").partition(":")
    if not internal.isdigit():
        return None
    position = {"figure": ids.figure_id, "table": ids.table_id}.get(kind, lambda _: None)(
        int(internal)
    )
    return f"{kind}:{position}" if position is not None else None


def _export_identity_sections(identity: Any, ids: ExportIds) -> Any:
    """The identity block with its DOI candidates' ``section_id`` in export ids.

    A DOI read from a caption or footnote keeps its ``section_type``
    (``figure``, ``table``, ``footnote``) but has no section to point at.
    """
    receipt = identity.get("receipt") if isinstance(identity, dict) else None
    if not isinstance(receipt, dict):
        return identity

    def candidate(row: Any) -> Any:
        if not isinstance(row, dict):
            return row
        return {**row, "section_id": ids.section_id(row.get("section_id"))}

    remapped = dict(receipt)
    if "selected" in remapped:
        remapped["selected"] = candidate(remapped["selected"])
    if "candidates" in remapped:
        remapped["candidates"] = [candidate(row) for row in remapped["candidates"] or []]
    return {**identity, "receipt": remapped}


def _export_paper_payload(
    paper: Paper,
    *,
    include_regions: bool = False,
    include_region_meta: bool = False,
    validate: bool = True,
) -> dict:
    """Export Paper as a JSON-serializable dict matching the bibr v12.0 schema.

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

    ``include_region_meta`` controls ``extraction.text_regions``: per-sentence
    layout features (bbox, font, region type) originally added as v4 training
    features. Off by default — they are ≈19% of typical output and, like
    ``extraction.regions``, processing payload rather than paper content.

    ``validate`` (default on) runs the output validation gate
    (:mod:`bibr.export.validation`) and adds ``extraction.validation``.
    Gate findings live there only — they are never mirrored into
    ``extraction.warnings``. The gate never raises out of export.
    """
    if not paper.contents:
        raise ValueError("Paper has no contents")

    input_fmt = normalized_input_format(paper.input_file)
    # Every exported bounding box is converted from the 0..1000 layout space to
    # points on the displayed page; ``extraction.pages`` lists the page sizes.
    geometry = PageGeometry(paper.contents.page_sizes)
    # Captions and footnotes leave their synthetic sections, and every
    # structural id becomes a position in document order.
    ids = export_ids(paper.contents)

    # text (built from sentences; captions and footnotes have no section,
    # display math → formatted)
    text_data = []
    # v4 training layout features, opt-in: processing payload, so they ride
    # ``extraction.text_regions`` rather than the text rows. A sentence without
    # them (DOCX input, scanned PDF without a native text layer) has no row.
    text_regions: list[TextRegionExport] = []
    for sent in paper.contents.sentences:
        if sent.is_display_formula:
            text = "[equation]"
            formatted = sent.text
        else:
            text = sent.text
            formatted = None
        text_data.append(
            TextExport(
                text=text,
                text_id=sent.text_id,
                paragraph_id=sent.paragraph_id,
                section_id=ids.section_id(sent.section_id),
                page_number=sent.page_number,
                formatted=formatted,
            )
        )
        rm = sent.region_meta if include_region_meta else None
        if rm:
            region_page = rm.get("region_page")
            text_regions.append(
                TextRegionExport(
                    text_id=sent.text_id,
                    font_size=rm.get("font_size"),
                    font_bold=rm.get("font_bold"),
                    is_italic=rm.get("is_italic"),
                    page_number=region_page,
                    bbox=geometry.box(region_page, rm.get("bbox")),
                    region_type=rm.get("region_type"),
                )
            )

    def _authors_to_dicts(authors: list | None) -> list[dict] | None:
        if not authors:
            return None
        rows = []
        for a in authors:
            row: dict[str, Any] = {"given": a.given, "family": a.family}
            if orcid := _conformed(getattr(a, "orcid", None), _ORCID_RE, "orcid"):
                row["orcid"] = orcid
            if getattr(a, "affiliation", None):
                row["affiliation"] = [
                    {"name": org.name, "ror": _conformed(org.ror, _ROR_RE, "ror")}
                    for org in a.affiliation
                ]
            rows.append(row)
        return rows

    def _record_ids(m) -> dict[str, Any]:
        """License and funder identifiers of one external record."""
        # Taken from the external record as deposited, like its ``url``.
        license_url = _without_active_scheme(getattr(m, "license_url", None))
        funders = getattr(m, "funders", None)
        return {
            "license_url": license_url,
            "license_spdx": license_ids(license_url)[1] if license_url else None,
            "funder": [
                {
                    "name": f.name,
                    "funder_doi": _export_doi(f.funder_doi, "funder_doi"),
                    "ror": _conformed(f.ror, _ROR_RE, "ror"),
                    "award_ids": list(f.award_ids),
                }
                for f in funders
            ]
            if funders
            else None,
        }

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
                bib_type=_export_bib_type(r.bib_type),
                doi=_export_doi(r.doi, "bib.doi"),
                title=r.title or None,
                authors=r.authors,
                editors=r.editors,
                publisher=r.publisher,
                year=export_year,
                year_suffix=r.year_suffix,
                date=r.date,
                published_date=_bib_published_date(r.date, export_year),
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
                url=_without_active_scheme(_normalize_export_url(r.url)) if r.url else r.url,
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
                        service_id=_export_service_id(d.get("id")),
                        score=_match_score(d.get("score")),
                        bib_type=_export_bib_type(d.get("bib_type")),
                        doi=_export_doi(d.get("doi"), "bib_match.doi"),
                        title=d.get("title"),
                        author=_authors_to_dicts(m.authors),
                        editor=_authors_to_dicts(m.editors),
                        publisher=d.get("publisher"),
                        year=d.get("year"),
                        published_date=_conformed(
                            d.get("date"), _ISO_DATE_RE, "bib_match.published_date"
                        ),
                        container=d.get("container"),
                        volume=d.get("volume"),
                        issue=d.get("issue"),
                        first_page=d.get("first_page"),
                        last_page=d.get("last_page"),
                        edition=d.get("edition"),
                        version=d.get("version"),
                        url=_without_active_scheme(d.get("url")),
                        **_record_ids(m),
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
                    service_id=_export_service_id(d.get("id")),
                    score=_match_score(d.get("score")),
                    bib_type=_export_bib_type(d.get("bib_type")),
                    doi=_export_doi(d.get("doi"), "metadata_match.doi"),
                    title=d.get("title"),
                    author=_authors_to_dicts(m.authors),
                    editor=_authors_to_dicts(m.editors),
                    publisher=d.get("publisher"),
                    year=d.get("year"),
                    published_date=_conformed(
                        d.get("date"), _ISO_DATE_RE, "metadata_match.published_date"
                    ),
                    container=d.get("container"),
                    volume=d.get("volume"),
                    issue=d.get("issue"),
                    first_page=d.get("first_page"),
                    last_page=d.get("last_page"),
                    edition=d.get("edition"),
                    version=d.get("version"),
                    url=_without_active_scheme(d.get("url")),
                    **_record_ids(m),
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
        regions_data = []
        for region in paper.contents.region_summaries:
            # The summary measures in layout units; the export in points.
            scale = geometry.scale(region.page)
            char_density = line_height = None
            if scale is not None:
                sx, sy = scale
                if region.char_density is not None:
                    char_density = round(region.char_density / (sx * sy), 6)
                if region.estimated_line_height is not None:
                    line_height = round(region.estimated_line_height * sy, 2)
            regions_data.append(
                RegionExport(
                    page=region.page,
                    index=region.index,
                    label=region.label,
                    bbox=geometry.box(region.page, region.bbox),
                    font_size=region.font_size,
                    font_weight=region.font_weight,
                    font_bold=region.font_bold,
                    section_id=ids.section_id(region.section_id),
                    content=region.content,
                    raw_ocr_content=region.raw_ocr_content,
                    char_density=char_density,
                    estimated_line_height=line_height,
                )
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
                    bbox=geometry.box(candidate.page_number, candidate.bbox),
                    source_index=candidate.source_index,
                )
                for candidate in receipt.candidates
            ],
            assignments=[
                CaptionAssignmentExport(
                    caption_id=assignment.caption_id,
                    object_id=_export_float_ref(assignment.object_id, ids),
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

    meta = paper.metadata
    # Spans of xrefs, links and expressions are offsets into the exported text.
    texts = {row.text_id: row.text for row in text_data}
    xref_locator = SpanLocator(texts, shared=True)
    url_locator = SpanLocator(texts, shared=False)
    eq_locator = SpanLocator(texts, shared=False)
    exported_links = _sane_export_links(paper.contents.links)
    exported_xrefs = list(enumerate(paper.contents.xrefs, start=1))

    # Processing facts about content rows, keyed by the rows' primary keys.
    section_classification = [
        SectionClassificationExport(
            section_id=position,
            # ``PaperSection`` defaults an unclassified section's score to 0.0
            # with no source; that is "not scored", not a zero-confidence score.
            score=_unit_interval(
                (
                    s.classification_score
                    if s.classification_source is not None or s.classification_score
                    else None
                ),
                "section classification score",
            ),
            source=s.classification_source,
        )
        for position, s in enumerate(ids.sections, start=1)
    ]
    xref_tier = [
        XrefTierExport(xref_id=xref_id, tier=cast(XrefTierLiteral, tier))
        for xref_id, x in exported_xrefs
        if (tier := _in_vocabulary(_snake(x.tier), _XREF_TIERS, "xref tier"))
    ]
    license_url, license_spdx = license_ids(meta.license if meta else None)
    first_page_texts = [row.text for row in text_data if row.page_number == 1]
    paper_type = _in_vocabulary(
        _snake(meta.paper_type if meta else None), _PAPER_TYPES, "paper_type"
    )
    oecd_l1 = _in_vocabulary(meta.oecd_l1 if meta else None, _OECD_L1, "oecd_l1")
    oecd_l2 = _in_vocabulary(meta.oecd_l2 if meta else None, _OECD_L2, "oecd_l2")
    paper_classification: PaperClassificationExport | None = None
    if meta is not None and (paper_type or oecd_l1):
        paper_classification = PaperClassificationExport(
            paper_type_confidence=_unit_interval(
                meta.paper_type_confidence if paper_type else None, "paper_type_confidence"
            ),
            oecd_confidence=_unit_interval(
                meta.oecd_confidence if oecd_l1 else None, "oecd_confidence"
            ),
        )

    # Where each printed piece of a figure or table sits: layout provenance,
    # so it rides ``extraction`` rather than the figure/table rows.
    float_parts = [
        FloatPartExport(
            object_type=object_type,
            object_id=object_id,
            part_index=index,
            page_number=part.page_number,
            bbox=geometry.box(part.page_number, part.bbox),
        )
        for object_type, object_id, parts in (
            *(("figure", position, f.parts) for position, f in enumerate(ids.figures, start=1)),
            *(("table", position, t.parts) for position, t in enumerate(ids.tables, start=1)),
        )
        for index, part in enumerate(parts, start=1)
        if part.page_number is not None or part.bbox
    ]

    # Every processing surface hangs off ``extraction``. The stage builds the
    # provenance skeleton; the receipts/enrichment/regions that only exist at
    # serialization time are folded in here. A Paper exported outside the
    # pipeline gets a minimal skeleton, so ``extraction`` is always present
    # and nothing processing-shaped has to fall back onto the content rows.
    extraction_data = (
        dict(paper.extraction) if paper.extraction is not None else _minimal_extraction()
    )
    if extraction_data is not None:
        diagnostics = dict(extraction_data.get("diagnostics") or {})
        diagnostics["references_complete"] = not bool(meta and meta.references_incomplete)
        # ``setdefault``: the pipeline's ``_build_extraction`` already stamps the
        # score, and its value wins. This only covers callers that hand-build an
        # ``extraction`` block without one, so the score is never silently lost.
        diagnostics.setdefault("text_quality", paper.text_quality)
        diagnostics["text_quality"] = _unit_interval(diagnostics["text_quality"], "text_quality")
        if section_classification:
            diagnostics["section_classification"] = section_classification
        if paper_classification is not None:
            diagnostics["paper_classification"] = paper_classification
        if xref_tier:
            diagnostics["xref_tier"] = xref_tier
        if citation_linking is not None:
            diagnostics["citation_linking"] = citation_linking
        if caption_assignment is not None:
            diagnostics["caption_assignment"] = caption_assignment
        if reference_yield is not None:
            diagnostics["reference_yield"] = reference_yield
        extraction_data["diagnostics"] = diagnostics
        if identity := extraction_data.get("identity"):
            extraction_data["identity"] = _export_identity_sections(identity, ids)
        if enrichment_export is not None:
            extraction_data["enrichment"] = enrichment_export
        if paper.qualification_provenance:
            extraction_data["qualification"] = paper.qualification_provenance
        if regions_data is not None:
            extraction_data["regions"] = regions_data
        if text_regions:
            extraction_data["text_regions"] = text_regions
        if float_parts:
            extraction_data["float_parts"] = float_parts
        pages = geometry.pages()
        if pages is not None:
            extraction_data["pages"] = pages
        # Warnings are unioned rather than overwritten: the stage snapshots
        # ``paper.processing_warnings`` when it builds the block, but callers
        # (and the stage itself) may append after that point.
        warnings = map(
            ProcessingWarning.from_dict,
            [*(extraction_data.get("warnings") or []), *paper.processing_warnings],
        )
        extraction_data["warnings"] = [w.to_dict() for w in dict.fromkeys(warnings)]

    affiliation_data = _export_affiliations(paper)
    funding_data = [
        FundingExport(funding_id=position, funder=f.funder, award_ids=f.award_ids)
        for position, f in enumerate(meta.funding if meta else [], start=1)
    ]
    exported_doi = _export_doi(meta.doi if meta else None, "metadata.doi")
    from bibr.field_states import FieldScope

    if extraction_data is not None and isinstance(paper.field_scope, FieldScope):
        extraction_data["fields"] = _field_states(
            paper,
            present={
                "title": bool(meta and meta.title),
                "author": bool(meta and meta.authors),
                "abstract": bool(abstract_text),
                "keywords": bool(exported_keywords),
                "doi": bool(exported_doi),
                "published": bool(meta and meta.published),
                "journal": bool(meta and meta.journal),
                "funding_statement": bool(meta and meta.funding_statement),
                "funding": bool(funding_data),
                "paper_type": bool(paper_type),
                "bib": bool(bib_data),
            },
            warnings=extraction_data.get("warnings") or [],
        )
    # ROR matches are keyed by the printed string, so a string that occurs once
    # in the table gets its match whatever position it ended up at.
    affiliation_match_data = [
        AffiliationMatchExport(
            affiliation_id=a.affiliation_id, service="ror", **_organization_fields(org)
        )
        for a in affiliation_data
        if meta
        and (org := meta.affiliation_match.get(a.text)) is not None
        and _conformed(org.service_id, _ROR_RE, "affiliation_match.service_id")
    ]
    funding_match_data = [
        FundingMatchExport(
            funding_id=f.funding_id,
            service="ror",
            **_organization_fields(org),
            funder_doi=_export_doi(org.funder_doi, "funding_match.funder_doi"),
        )
        for f in funding_data
        if meta
        and (org := meta.funder_match.get(f.funder)) is not None
        and _conformed(org.service_id, _ROR_RE, "funding_match.service_id")
    ]
    equations: list[tuple[Any, EqCompLiteral]] = []
    for eq in paper.contents.equations:
        comp = _export_comp(eq.comp)
        if comp is None:
            logger.warning("Dropping expression with unknown comparator %r from export", eq.comp)
            continue
        equations.append((eq, comp))

    sha256 = paper.input_file.sha256
    export = PaperExport(
        paper_id=(
            paper._compute_paper_id()
            or (sha256 or paper.input_file.file_hash or "")[:16]
            or "paper"
        ),
        schema_version=_SCHEMA_VERSION,
        source=SourceExport(
            file_name=paper.input_file.file_name,
            sha256=sha256,
            input_format=_export_input_format(input_fmt),
        ),
        metadata=MetadataExport(
            title=(meta.title or None) if meta else None,
            abstract=abstract_text,
            keywords=exported_keywords,
            doi=exported_doi,
            pmid=(meta.pmid or None) if meta else None,
            pmcid=(meta.pmcid or None) if meta else None,
            arxiv=(
                (meta.arxiv if meta else None)
                or arxiv_id(meta.doi if meta else None, first_page_texts)
            ),
            language=(meta.language or None) if meta else None,
            paper_type=paper_type,
            oecd_l1=oecd_l1,
            oecd_l2=oecd_l2,
            journal=(meta.journal or None) if meta else None,
            volume=(meta.volume or None) if meta else None,
            issue=(meta.issue or None) if meta else None,
            first_page=(meta.first_page or None) if meta else None,
            last_page=(meta.last_page or None) if meta else None,
            issn=(meta.issn or None) if meta else None,
            publisher=(meta.publisher or None) if meta else None,
            published=(meta.published or None) if meta else None,
            published_date=iso_date(meta.published if meta else None),
            license=(meta.license or None) if meta else None,
            license_url=license_url,
            license_spdx=license_spdx,
            funding_statement=(meta.funding_statement or None) if meta else None,
            coi_statement=(meta.coi_statement or None) if meta else None,
            ethics_statement=(meta.ethics_statement or None) if meta else None,
            data_availability=(meta.data_availability or None) if meta else None,
        ),
        author=[_export_author(a) for a in (meta.authors if meta else [])],
        affiliation=affiliation_data,
        funding=funding_data,
        text=text_data,
        section=[
            SectionExport(
                section_id=position,
                header=_or_none(s.header),
                level=s.level,
                parent_section_id=ids.section_id(s.parent_section_id),
                section_type=(
                    cast(SectionTypeLiteral, export_section_type(s.section_type.value))
                    if s.section_type
                    else None
                ),
            )
            for position, s in enumerate(ids.sections, start=1)
        ],
        url=[
            UrlExport(
                url_id=position,
                href=href,
                link_text=link.link_text,
                text_id=link.text_id,
                start=span[0] if span else None,
                end=span[1] if span else None,
            )
            for position, link in enumerate(exported_links, start=1)
            for href in (_normalize_export_url(link.url),)
            for span in (url_span(url_locator, link, href),)
        ],
        bib=bib_data,
        xref=[
            XrefExport(
                xref_id=xref_id,
                target_id=ids.target_id(x),
                xref_type=x.xref_type,
                contents=x.contents,
                text_id=x.text_id,
                start=span[0] if span else None,
                end=span[1] if span else None,
            )
            for xref_id, x in exported_xrefs
            for span in (xref_span(xref_locator, texts, x),)
        ],
        figure=[
            FigureExport(
                figure_id=position,
                label=_or_none(f.label),
                section_id=ids.float_section_id(f),
                text_id=ids.caption_text_id(f),
                image=_data_uri(f.image_b64),
                caption=f.caption,
                page_number=f.page_number,
            )
            for position, f in enumerate(ids.figures, start=1)
        ],
        table=[
            TableExport(
                table_id=position,
                label=_or_none(t.label),
                section_id=ids.float_section_id(t),
                text_id=ids.caption_text_id(t),
                html=t.tbl_html or None,
                contents=t.contents,
                caption=t.caption,
                page_number=t.page_number,
            )
            for position, t in enumerate(ids.tables, start=1)
        ],
        footnote=ids.footnotes,
        eq=[
            EqExport(
                eq_id=position,
                text_id=eq.text_id,
                start=span[0] if span else None,
                end=span[1] if span else None,
                grp_id=eq.grp_id,
                verbatim=texts[eq.text_id][span[0] : span[1]] if span else None,
                lhs=eq.lhs,
                df=_or_none(eq.df),
                comp=comp,
                rhs=eq.rhs,
            )
            for position, (eq, comp) in enumerate(equations, start=1)
            for span in (equation_span(eq_locator, eq),)
        ],
        metadata_match=metadata_match_data,
        affiliation_match=affiliation_match_data,
        funding_match=funding_match_data,
        bib_match=bib_match_data,
        extraction=extraction_data,
    )

    payload = export.model_dump(by_alias=True)
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
    """Build and validate the typed v12.0 export model for *paper*."""
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
    """Export *paper* as the v12.0 dictionary payload."""
    export = build_paper_export(
        paper,
        include_regions=include_regions,
        include_region_meta=include_region_meta,
        validate=validate,
    )
    return cast(dict[str, Any], export.model_dump(by_alias=True, exclude_unset=True))
