"""Pydantic models for the bibr JSON export schema — the single source of truth.

Every model rejects unknown keys so a divergence between the schema and the
hand-built dicts in ``bibr.export.json_export`` surfaces as a validation error
instead of being silently ignored.

Those strict models are the *producer* contract. :data:`PaperExportReader` is
the matching *reader*: a generated lenient mirror that accepts any export of
the current major version, including fields added by a later minor writer.

Every field carries a ``description``; ``tests/export/test_schema_artifact.py``
fails on one that does not, because the generated JSON Schema is the only
contract downstream readers (metacheck, scienceverse/schema) get.
"""

from __future__ import annotations

import copy
import functools
import operator
from types import UnionType
from typing import Annotated, Any, ClassVar, Literal, Union, cast, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from pydantic.fields import FieldInfo

_SCHEMA_VERSION = "12.0"


# ---------------------------------------------------------------------------
# Pydantic v12.0 export schema — single source of truth for validation
#
# v12.0 (vs 11.0) — BREAKING. The payload separates what the paper says from
# how bibr produced it: ``extraction`` holds all processing data, and the
# content tables carry none.
#   - Root keys in order: ``paper_id``, ``schema_version``, ``source``;
#     ``metadata``, ``author``, ``affiliation``, ``funding``, ``text``,
#     ``section``, ``url``, ``bib``, ``xref``, ``figure``, ``table``,
#     ``footnote``, ``eq``; ``metadata_match``, ``affiliation_match``,
#     ``funding_match``, ``bib_match``; ``extraction``.
#   - Processing fields moved out of the content rows:
#       ``section[].classification_score``/``classification_source``
#         -> ``extraction.diagnostics.section_classification[]``
#       ``xref[].tier`` -> ``extraction.diagnostics.xref_tier[]``
#       ``metadata.paper_type_confidence``/``oecd_confidence``
#         -> ``extraction.diagnostics.paper_classification``
#       ``bib[].consolidated_fields`` (comma-joined string)
#         -> ``extraction.diagnostics.consolidation[]`` (list of names)
#       ``text[]._font_size``/``_font_bold``/``_is_italic``/``_bbox_2d``/
#         ``_region_type``/``_page_w``/``_page_h`` (opt-in)
#         -> ``extraction.text_regions[]`` (same opt-in)
#       root ``qualification_provenance`` -> ``extraction.qualification``
#         (omitted, not ``null``, when no LLM task ran)
#       root ``validation`` -> ``extraction.validation``
#   - ``extraction`` is always present, so everything that is not the paper
#     lives in one place: a Paper exported outside the pipeline gets a minimal
#     block, and ``extraction.settings`` is omitted there.
#   - Figure/table ``parts`` removed from the content rows. The whole-object
#     fields are now truly whole: a multi-panel figure's ``image`` is composited
#     from its panel crops, and a continued table's ``html`` keeps each printed
#     piece's markup instead of re-rendering the merged cells. Each piece's page
#     and bbox moved to ``extraction.float_parts``; ``ProvenanceExport`` is gone.
#   - Duplicates removed: ``bib[].author``/``editor`` (a split derived from
#     the printed ``authors``/``editors`` strings, which stay) and
#     ``author[].affiliation`` (the ``affiliation[]`` table, linked through
#     ``author_ids``, is the one source and is now built on every run, not only
#     when the LLM parse runs).
#   - Every record table has a primary key: new ``xref[].xref_id`` and
#     ``url[].url_id`` (1-based position). ``xref_id`` meant the *target* up
#     to v10.x; that meaning lives in ``target_id``.
#   - ``xref[]``, ``url[]`` and ``eq[]`` gain ``start``/``end``: the character
#     span of the item in its sentence's ``text``; ``eq[].verbatim`` is filled
#     from it.
#   - Normalized companions next to printed values: ``metadata.published_date``
#     (ISO 8601), ``license_url``/``license_spdx``, ``language``, ``pmid``,
#     ``pmcid``, ``arxiv``; ``author[].credit_roles`` (CRediT URIs).
#   - Absent values are ``null``, never a sentinel: ``author[].given``/
#     ``family``, ``section[].header`` and ``eq[].df`` were ``""`` when absent.
#   - Closed vocabularies are enums in the schema: ``section[].section_type``,
#     ``bib[].bib_type`` (and the match tables'), ``metadata.paper_type``,
#     ``metadata.oecd_l1``/``oecd_l2``, ``*_match[].service``,
#     ``source.input_format``, ``eq[].comp``, ``validation.issues[].severity``.
#     Every token is snake_case: ``paper_type`` ``meta_analysis``/
#     ``case_study``; ``section_type`` ``open_data`` is ``data_availability``;
#     ``input_format`` names the format, not the file extension (``jats`` and
#     ``tei`` for XML, no ``htm``).
#   - ``section[]`` holds only the paper's sections. Captions and footnotes
#     are no longer sections of their own (``section_type`` ``figure``,
#     ``table``, ``footnote``, header "Figure 2"); their sentences stay in
#     ``text[]`` with a ``null`` ``section_id``. ``figure[]``/``table[]`` gain
#     ``text_id`` (the caption's row) and their ``section_id`` is the section
#     they are printed in; the new ``footnote[]`` table has one row per
#     footnote or endnote with its printed ``label`` and ``text_id``.
#   - Every id is a 1-based position: ``section_id`` in document order with no
#     gaps, and ``figure_id``/``table_id`` in document order (PDF used the
#     printed number, leaving gaps).
#   - ``xref[].target_id`` is a key of a real row or ``null``: ``foot`` points at
#     ``footnote[].footnote_id`` (it held the footnote's ordinal), and
#     ``equation``, ``section`` and ``supplementary`` references are ``null``
#     (they held the printed number, or ``0``).
#   - One scale and one name per concept: every ``score`` and confidence is 0–1
#     (``*_match[].score`` was 0–100); ``published_date`` is ISO 8601
#     everywhere (``*_match[].date`` is renamed, ``bib[]`` gains it and
#     consolidation fills it instead of the printed ``bib[].date``); every DOI
#     is bare and lowercase.
#   - ``author[].literal`` holds a group author's name (it was in ``family``).
#   - ``source.file_hash`` (16 hex of the SHA-256) -> ``source.sha256`` (all
#     64); ``extraction.bibr_version``/``build_sha`` ->
#     ``extraction.producer`` {name, version, build_sha}, so other producers of
#     the format can identify themselves.
#   - ``paper_id`` defaults to the input file's stem, as ``bibr batch`` and
#     metacheck already do, not to the DOI.
#   - ``figure[].image`` is a ``data:`` URI that names its media type.
#   - ``extraction.warnings`` items are ``{code, message}`` objects, not prose:
#     ``code`` is a stable UPPER_SNAKE code, ``message`` the details.
#   - The published schema has a stable ``$id`` under https://bibr.org/schema/.
#     In the strict schema every key the exporter always writes is
#     ``required``; nullable scalars are written ``"type": [T, "null"]``; and
#     identifiers, dates, ids and scores carry patterns and bounds.
#   - Forward policy: 12.x is additive-only — new optional fields and new enum
#     values may appear in any 12.x release, and readers must ignore keys they
#     don't recognize and accept enum values they don't know
#     (:data:`PaperExportReader` does both). Any rename, move, removal, type
#     change, new required key or dropped enum value requires 13.0 and a coordinated
#     release across bibr, scienceverse/schema, and metacheck. Readers dispatch
#     on the *presence* of a root ``schema_version`` key; pre-v11 payloads (and
#     metacheck's fixture corpus) have no such key at all.
#
# v11.1 was drafted in open PRs and never released; it is retired, and 12.0
# follows 11.0 directly.
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
#   - ``eq[]`` gains ``eq_id`` (1-based position) and ``verbatim`` (the printed
#     expression as matched; null until the equation parser populates it).
#   - ``funding[]`` gains ``funding_id`` (1-based position; same no-PK gap).
#   - Root ``affiliations`` renamed to ``affiliation`` — singular-table rule,
#     matching every other root record-array key.
#   - Naming rule (CSL-borrowed): plural = verbatim printed string, singular =
#     derived structured array. ``bib[].authors``/``editors`` stay exactly as
#     printed; new ``bib[].author``/``editor`` carry the best-effort split.
#     ``bib_match[]``/``metadata_match[].authors``/``editors`` renamed to
#     ``author``/``editor`` (they were always structured). ``author[].suffix``
#     added to paper authors.
#   - ``table[].contents`` / ``table[].parts[].contents`` typed as
#     ``list[list[str]]`` instead of bare ``list``.
#
# v10.9 (vs 10.8) — additive: structured ``bib[].author``/``editor`` lists and
#   ``author[].suffix``; first generated JSON Schema artifact.
#
# v10.8 (vs 10.7) — additive: ``bib[].arxiv``, ``pmid``, ``series``,
#   ``access_date``, ``note``.
#
# v10.7 (vs 10.6) — additive:
#   - Figure/table ``parts`` preserve every physical payload and provenance.
#   - Optional ``caption_assignment`` and ``reference_yield`` diagnostic receipts.
#
# v10.6 (vs 10.5) — additive:
#   - New top-level ``llm_usage_by_label``: per-call-site LLM token usage.
#   - ``xref`` gains optional ``tier`` field: bib-link detection provenance.
#
# v10.5 (vs 10.4) — all additive:
#   - ``info.bibr_version`` restored, now carrying the producing *package*
#     version (pre-10.3 it held the schema version).
#   - ``info.input_format`` is now contractually lowercase.
#   - Per-text underscore region metadata (``_bbox_2d``, ``_font_size``, …)
#     is now opt-in (``include_region_meta=True``; CLI ``--region-meta``).
#   - New top-level ``validation`` block ``{errors, warnings, issues[]}``.
#
# v10.4 (vs 10.3) — all additive:
#   - ``figure[].caption`` / ``table[].caption``.
#   - ``info`` gains the paper's OWN bibliographic self-identity (``journal``,
#     ``volume``, ``issue``, ``first_page``, ``last_page``, ``issn``,
#     ``publisher``, ``published``, ``license``) and four research-integrity
#     statements.
#   - New top-level ``info_match``, ``funding`` and ``affiliations``.
#   - ``author[].role`` populated from the author-contributions statement.
#
# v10.3 (vs 10.2):
#   - ``info.bibr_version`` renamed to ``info.schema_version``; the package
#     version moved to the new top-level ``extraction.bibr_version``.
#
# v10.2 (vs 10.1) — tracks scienceverse/schema "Updated text order and eq.df":
#   - ``eq`` entries carry a ``df`` field split out of the LHS.
#   - ``text`` keys follow the upstream order.
#   - ``xref.xref_id`` is nullable.
#
# v10.1 (vs 10.0):
#   - ``info.ocr_config`` and ``info.processing_warnings`` moved to the top
#     level, keeping ``info`` scalar-only for R consumers (metacheck).
#   - ``_regions`` debug payload is now opt-in (``include_regions=True``).
# ---------------------------------------------------------------------------


_STRICT = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------
# Closed vocabularies
#
# Hand-written because ``typing.Literal`` cannot unpack a runtime list, and
# kept here (not imported) so this module stays dependency-free for readers.
# ``tests/export/test_vocabularies.py`` pins each one to its runtime source:
# ``CanonicalSection``, ``BibType``, ``PAPER_TYPE_LABELS``, the OECD label
# lists, ``MatchSource``, ``SupportedFileType``, the equation extractor's
# comparators and ``IssueSeverity``. Where a runtime label is spelled
# differently (a classifier's ``meta-analysis``, the ``open_data`` section, a
# ``.htm`` file) the exporter maps it, and it normalizes an off-list value
# (``bibr.export.json_export``) instead of letting one stray label fail a
# paper's whole export.
# ---------------------------------------------------------------------------

SectionTypeLiteral = Literal[
    "title",
    "abstract",
    "intro",
    "method",
    "results",
    "discussion",
    "references",
    "acknowledgment",
    "funding",
    "keywords",
    "endnote",
    "appendix",
    "data_availability",
    "author_contributions",
    "coi",
    "ethics",
    "footnote",
    "table",
    "figure",
    "unknown",
]

BibTypeLiteral = Literal[
    "journal_article",
    "book",
    "book_chapter",
    "dataset",
    "software",
    "preprint",
    "conference_paper",
    "report",
    "thesis",
    "other",
]

PaperTypeLiteral = Literal[
    "empirical",
    "review",
    "meta_analysis",
    "case_study",
    "commentary",
    "corrigendum",
    "erratum",
    "retraction",
    "unknown",
]

OecdL1Literal = Literal[
    "Natural Sciences",
    "Engineering and Technology",
    "Medical and Health Sciences",
    "Agricultural and Veterinary Sciences",
    "Social Sciences",
    "Humanities and the Arts",
]

OecdL2Literal = Literal[
    "Mathematics",
    "Computer and Information Sciences",
    "Physical Sciences",
    "Chemical Sciences",
    "Earth and Related Environmental Sciences",
    "Biological Sciences",
    "Civil Engineering",
    "Electrical Engineering, Electronic Engineering, Information Engineering",
    "Mechanical Engineering",
    "Chemical Engineering",
    "Materials Engineering",
    "Medical Engineering",
    "Environmental Engineering",
    "Environmental Biotechnology",
    "Industrial Biotechnology",
    "Nano-technology",
    "Basic Medicine",
    "Clinical Medicine",
    "Health Sciences",
    "Medical Biotechnology",
    "Agriculture, Forestry, and Fisheries",
    "Animal and Dairy Science",
    "Veterinary Science",
    "Agricultural Biotechnology",
    "Psychology and Cognitive Sciences",
    "Economics and Business",
    "Education",
    "Sociology",
    "Law",
    "Political Science",
    "Social and Economic Geography",
    "Media and Communications",
    "History and Archaeology",
    "Languages and Literature",
    "Philosophy, Ethics and Religion",
    "Arts (arts, history of arts, performing arts, music)",
]

MatchServiceLiteral = Literal[
    "crossref", "openalex", "datacite", "doi.org", "openlibrary", "ror", "manual", "other"
]

# The format of the input, not its file extension: XML is JATS (what bibr
# reads) or TEI (what GROBID writes, for converters into this format).
InputFormatLiteral = Literal["pdf", "docx", "jats", "tei", "html", "epub", "unknown"]

# ``bibr.extract.equation_extractor._COMP_PATTERN`` after ``_normalize_comp``.
EqCompLiteral = Literal["=", "<", ">", "≤", "≥", "≈", "≠", "≪", "≫", "~"]

SeverityLiteral = Literal["error", "warning"]

XrefTypeLiteral = Literal["bib", "table", "figure", "foot", "supplementary", "equation", "section"]

# The citation linker's tiers with ``-`` spelled ``_`` (see ``json_export``),
# then how ``detect_xrefs`` resolved a figure or table xref.
XrefTierLiteral = Literal[
    "numeric", "paren_numeric", "flattened_superscript", "author_year", "llm", "label", "position"
]


# ---------------------------------------------------------------------------
# Value formats
#
# Patterns of the values the exporter normalizes, enforced by the models and
# published in the schema. ``bibr.export.json_export`` conforms a value that
# comes from outside bibr (a registry record, a publisher's metadata) before
# building the models, and drops one that still does not fit instead of failing
# the export. The regexes use only syntax that JSON Schema (ECMA-262) and
# pydantic's regex engine share.
# ---------------------------------------------------------------------------

# Bare and lowercase: DOIs are case-insensitive, so one spelling joins.
DOI_PATTERN = r"^10\.[0-9]{4,9}/[^\sA-Z]+$"
ORCID_PATTERN = r"^https://orcid\.org/[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]$"
ROR_PATTERN = r"^https://ror\.org/0[a-z0-9]{6}[0-9]{2}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
# ISO 8601, as precise as the source: YYYY, YYYY-MM or YYYY-MM-DD.
ISO_DATE_PATTERN = r"^[0-9]{4}(-[0-9]{2}(-[0-9]{2})?)?$"
UTC_TIMESTAMP_PATTERN = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?Z$"
COUNTRY_CODE_PATTERN = r"^[A-Z]{2}$"
CREDIT_ROLE_PATTERN = r"^https://credit\.niso\.org/contributor-roles/[a-z-]+/$"
DATA_URI_PATTERN = r"^data:[a-z]+/[a-z0-9.+-]+;base64,"

# A 1-based id, and a bounding box of exactly four numbers.
Id = Annotated[int, Field(ge=1)]
Box = Annotated[list[float], Field(min_length=4, max_length=4)]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class SourceExport(BaseModel):
    """Identity of the input file — not of the paper."""

    model_config = _STRICT

    file_name: str = Field(description="Name of the input file, without directories.")
    sha256: str | None = Field(
        pattern=SHA256_PATTERN,
        description="SHA-256 digest of the input file's bytes, as 64 lowercase hexadecimal "
        "characters; identical files give the same digest. Null only when the export was "
        "made without the input bytes.",
    )
    input_format: InputFormatLiteral = Field(
        description="The input's format: 'jats' for JATS XML, 'tei' for TEI XML (GROBID), "
        "otherwise 'pdf', 'docx', 'html' or 'epub'; 'unknown' when it could not be determined."
    )


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------


class MetadataExport(BaseModel):
    """Paper-level metadata, as printed in the paper.

    Scalars only (``keywords`` is the one list), so R consumers can call
    ``as.data.frame(metadata)`` (metacheck's ``.read_bibr``). How the values
    were produced lives under ``extraction``; the input file under ``source``.
    """

    model_config = _STRICT

    title: str | None = Field(description="Paper title; null when none was found.")
    abstract: str | None = Field(
        default=None, description="Abstract text; null when the paper has none or it was not found."
    )
    keywords: list[str] = Field(
        description="Author keywords in printed order; empty when none were found."
    )
    doi: str | None = Field(
        pattern=DOI_PATTERN,
        description="The paper's own DOI, bare (10.xxxx/..., no resolver prefix) and lowercase; "
        "null when none was found.",
    )
    pmid: str | None = Field(default=None, description="PubMed ID, when the input declares one.")
    pmcid: str | None = Field(
        default=None, description="PubMed Central ID (PMC…), when the input declares one."
    )
    arxiv: str | None = Field(
        default=None,
        description="arXiv identifier (e.g. 2101.12345), declared by the input, taken from an "
        "arXiv DOI, or read from the arXiv stamp on page 1.",
    )
    language: str | None = Field(
        default=None,
        description="Language tag the input declares (BCP 47, e.g. 'en'); null for PDFs.",
    )
    paper_type: PaperTypeLiteral | None = Field(
        default=None,
        description="Paper type from bibr's classifier; null when not classified. Its "
        "confidence is extraction.diagnostics.paper_classification.paper_type_confidence.",
    )
    oecd_l1: OecdL1Literal | None = Field(
        default=None, description="OECD Fields of Science domain; null when not classified."
    )
    oecd_l2: OecdL2Literal | None = Field(
        default=None,
        description="OECD Fields of Science subdomain, within oecd_l1; null when not classified.",
    )
    journal: str | None = Field(
        default=None, description="Journal or venue name from the paper's own front matter."
    )
    volume: str | None = Field(default=None, description="The paper's own volume, as printed.")
    issue: str | None = Field(default=None, description="The paper's own issue, as printed.")
    first_page: str | None = Field(
        default=None, description="The paper's own first page or article number, as printed."
    )
    last_page: str | None = Field(
        default=None, description="The paper's own last page, as printed."
    )
    issn: str | None = Field(default=None, description="The journal ISSN, as printed.")
    publisher: str | None = Field(default=None, description="The publisher, as printed.")
    published: str | None = Field(
        default=None, description="Publication date as printed (free text, not normalized)."
    )
    published_date: str | None = Field(
        default=None,
        pattern=ISO_DATE_PATTERN,
        description="The publication date in published as ISO 8601: YYYY, YYYY-MM or "
        "YYYY-MM-DD, as precise as printed; null when absent or ambiguous.",
    )
    license: str | None = Field(default=None, description="License statement or name, as printed.")
    license_url: str | None = Field(
        default=None,
        description="License URL: the one printed in license, else the canonical Creative "
        "Commons URL when the license is identified.",
    )
    license_spdx: str | None = Field(
        default=None,
        description="SPDX identifier of the license (e.g. 'CC-BY-4.0'), for Creative Commons "
        "licenses with a known version and CC0.",
    )
    funding_statement: str | None = Field(
        default=None,
        description="Funding statement text from the paper's funding section. The funders and "
        "award numbers parsed from it are in funding[].",
    )
    coi_statement: str | None = Field(
        default=None, description="Conflict-of-interest statement text, as printed."
    )
    ethics_statement: str | None = Field(
        default=None, description="Ethics approval statement text, as printed."
    )
    data_availability: str | None = Field(
        default=None, description="Data-availability statement text, as printed."
    )


class AuthorExport(BaseModel):
    """One author of the paper, in byline order.

    Affiliations are in ``affiliation[]``, linked by ``author_ids``.
    """

    model_config = _STRICT

    author_id: Id = Field(description="Primary key; 1-based position in the byline.")
    given: str | None = Field(
        description="Given name(s) or initials; null for mononyms and group authors."
    )
    family: str | None = Field(
        description="Family name; null for group authors and when not found."
    )
    suffix: str | None = Field(default=None, description="Name suffix such as 'Jr.' or 'III'.")
    literal: str | None = Field(
        default=None,
        description="The whole name of a group or organization author (a consortium, a "
        "working group), as printed; given and family are null then. Null for people.",
    )
    email: str | None = Field(default=None, description="Email address, as printed.")
    corresponding: bool = Field(
        description="True when the paper marks this author as a corresponding author."
    )
    orcid: str | None = Field(
        default=None,
        pattern=ORCID_PATTERN,
        description="ORCID iD as a canonical URI (https://orcid.org/0000-0000-0000-0000).",
    )
    role: list[str] = Field(
        default=[],
        description="Contribution roles from the author-contributions statement, as printed; "
        "empty when the paper has none.",
    )
    credit_roles: list[Annotated[str, Field(pattern=CREDIT_ROLE_PATTERN)]] = Field(
        default=[],
        description="CRediT roles matched from role, as NISO term URIs "
        "(https://credit.niso.org/contributor-roles/...); empty when none matched.",
    )


class AffiliationExport(BaseModel):
    """One distinct affiliation from the author byline.

    Built on every run from the byline strings. The parsed components come
    from an LLM and stay null when no LLM ran or it returned no parse.
    """

    model_config = _STRICT

    affiliation_id: Id = Field(description="Primary key; 1-based position in first-seen order.")
    text: str = Field(description="The affiliation string, verbatim from the byline.")
    institution: str | None = Field(default=None, description="Parsed institution name.")
    department: str | None = Field(default=None, description="Parsed department or unit.")
    city: str | None = Field(default=None, description="Parsed city.")
    country: str | None = Field(default=None, description="Parsed country.")
    author_ids: list[Id] = Field(
        default=[],
        description="author[].author_id of every author with this affiliation, in byline order.",
    )


class FundingExport(BaseModel):
    """One funder parsed from the funding statement."""

    model_config = _STRICT

    funding_id: Id = Field(description="Primary key; 1-based position.")
    funder: str = Field(description="Funder name, as printed.")
    award_ids: list[str] = Field(
        default=[], description="Grant or award numbers attributed to this funder, as printed."
    )


class TextExport(BaseModel):
    """One sentence-level span of the document text, in reading order."""

    model_config = _STRICT

    # Key order mirrors the upstream schema (text, text_id, paragraph_id,
    # section_id, page_number, formatted) so metacheck's R data.frame columns
    # line up.
    text: str = Field(
        description="The sentence as plain text. A display equation is the placeholder "
        "'[equation]', with the expression in formatted."
    )
    text_id: Id = Field(
        description="Primary key; 1-based reading-order position. Caption and footnote rows "
        "come after the body."
    )
    paragraph_id: Id = Field(description="Paragraph this sentence belongs to (1-based).")
    section_id: Id | None = Field(
        description="section[].section_id of the section the sentence belongs to; null for a "
        "caption or footnote row (figure[], table[] and footnote[] point at those by text_id) "
        "and before the first section."
    )
    page_number: Id | None = Field(
        description="1-based page the sentence starts on; null for inputs without pages."
    )
    formatted: str | None = Field(
        default=None,
        description="Source form of the span when it differs from text: a display equation's "
        "expression as the input gives it (LaTeX from OCR); null otherwise.",
    )


class SectionExport(BaseModel):
    """One section of the paper, with its place in the section tree.

    The sections the paper has: its printed headings, the title when it is
    printed as one, and sections inferred where the paper prints no heading,
    such as an unheaded abstract. Captions and footnotes are not sections;
    ``figure[]``, ``table[]`` and ``footnote[]`` point at their text rows.
    """

    model_config = _STRICT

    section_id: Id = Field(description="Primary key; 1-based position in document order.")
    header: str | None = Field(
        description="Heading text as printed; null when the section has no heading."
    )
    level: Id = Field(
        description="Heading depth: 1 for top-level sections, 2 for their subsections, and so on."
    )
    parent_section_id: Id | None = Field(
        description="section_id of the parent section; null for top-level sections."
    )
    section_type: SectionTypeLiteral | None = Field(
        description="Section role (IMRaD plus front and back matter). How it was decided is in "
        "extraction.diagnostics.section_classification."
    )


class UrlExport(BaseModel):
    """One hyperlink found in the text."""

    model_config = _STRICT

    url_id: Id = Field(description="Primary key; 1-based position.")
    href: str = Field(
        description="Link target, repaired of line-wrap whitespace and a trailing sentence period."
    )
    link_text: str | None = Field(
        description="Visible link text when it differs from the URL; null when the URL itself "
        "is the text."
    )
    text_id: Id = Field(description="text[].text_id of the sentence containing the link.")
    start: int | None = Field(
        default=None,
        ge=0,
        description="Where the link's visible text (or the URL) starts in text[].text of "
        "text_id: a 0-based offset in Unicode code points; null when it could not be located.",
    )
    end: int | None = Field(
        default=None,
        ge=0,
        description="Where the link's visible text (or the URL) ends (exclusive), in the same "
        "units as start.",
    )


class BibExport(BaseModel):
    """One entry of the paper's reference list, parsed from the printed entry.

    Values are as printed unless consolidation filled or replaced them from
    ``bib_match``; the field names it took are listed per ``bib_id`` in
    ``extraction.diagnostics.consolidation``.
    """

    model_config = _STRICT

    bib_id: Id = Field(description="Primary key; 1-based position in the reference list.")
    text_id: Id | None = Field(
        default=None,
        description="text[].text_id of the reference-section sentence holding this entry; null "
        "when it could not be matched.",
    )
    bib_type: BibTypeLiteral | None = Field(
        default=None, description="Reference type; null when not determined."
    )
    doi: str | None = Field(
        default=None,
        pattern=DOI_PATTERN,
        description="DOI printed in the entry, bare and lowercase.",
    )
    title: str | None = Field(default=None, description="Title of the cited work.")
    authors: str | None = Field(
        default=None,
        description="Author list exactly as printed, including punctuation and 'et al.'.",
    )
    editors: str | None = Field(default=None, description="Editor list exactly as printed.")
    publisher: str | None = Field(default=None, description="Publisher.")
    year: int | None = Field(
        default=None, description="Publication year; null when absent or in press."
    )
    year_suffix: str | None = Field(
        default=None, description="Disambiguation letter after the year, e.g. 'a' in '2020a'."
    )
    date: str | None = Field(default=None, description="Fuller publication date, as printed.")
    published_date: str | None = Field(
        default=None,
        pattern=ISO_DATE_PATTERN,
        description="The publication date as ISO 8601 (YYYY, YYYY-MM or YYYY-MM-DD), as "
        "precise as the entry prints it in date or year; consolidation fills it from "
        "bib_match[].published_date. Null when neither is known.",
    )
    container: str | None = Field(
        default=None, description="Journal, book or proceedings the work appeared in."
    )
    volume: str | None = Field(default=None, description="Volume.")
    issue: str | None = Field(default=None, description="Issue.")
    first_page: str | None = Field(default=None, description="First page or article number.")
    last_page: str | None = Field(default=None, description="Last page.")
    edition: str | None = Field(default=None, description="Edition.")
    version: str | None = Field(default=None, description="Version (software, datasets).")
    url: str | None = Field(
        default=None, description="URL as printed, repaired of line-wrap artifacts."
    )
    is_in_press: bool = Field(
        default=False, description="True when the entry is marked in press or forthcoming."
    )
    arxiv: str | None = Field(default=None, description="arXiv identifier.")
    pmid: str | None = Field(default=None, description="PubMed identifier.")
    series: str | None = Field(default=None, description="Book or report series.")
    access_date: str | None = Field(
        default=None, description="Accessed/retrieved date of an online source, as printed."
    )
    note: str | None = Field(default=None, description="Trailing free-text note.")


class XrefExport(BaseModel):
    """One in-text reference from a sentence to a reference-list entry,
    table, figure, footnote, equation, supplement or section.

    A reference naming several targets (``[1, 2]``, ``Tables 2-4``) is one row
    per target; the rows share ``contents`` and the span.
    """

    model_config = _STRICT

    xref_id: Id = Field(description="Primary key; 1-based position.")
    target_id: Id | None = Field(
        description="Primary key of the referenced row, by xref_type: bib → bib_id, figure → "
        "figure_id, table → table_id, foot → footnote_id. A figure or table reference resolves "
        "by the label it prints; extraction.diagnostics.xref_tier says when a paper with no "
        "labels fell back to position. Null for equation, section and supplementary "
        "references, which name no exported row (the printed label is in contents), and "
        "whenever the reference did not resolve."
    )
    xref_type: XrefTypeLiteral = Field(description="What kind of item is referenced.")
    contents: str | None = Field(
        description="The citation or reference as printed, e.g. '(Smith, 2020)' or 'Table 2'."
    )
    text_id: Id = Field(description="text[].text_id of the sentence containing the reference.")
    start: int | None = Field(
        default=None,
        ge=0,
        description="Where the printed reference starts in text[].text of text_id: a 0-based "
        "offset in Unicode code points; null when it could not be located.",
    )
    end: int | None = Field(
        default=None,
        ge=0,
        description="Where the printed reference ends (exclusive), in the same units as start.",
    )


class FigureExport(BaseModel):
    """One figure."""

    model_config = _STRICT

    figure_id: Id = Field(description="Primary key; 1-based position in document order.")
    label: str | None = Field(
        default=None,
        description="The printed label without the word, as printed with whitespace removed: "
        "'3', '3.1', 'S2', 'A1', 'IV'; a 'Supplementary Figure 4' caption gives 'S4'. In-text "
        "references resolve by it. Not unique: the unmerged pieces of a "
        "figure split across pages repeat it ('Figure 3. Cont.'), and references target "
        "the piece not marked as continued. Null when none was printed or detected.",
    )
    section_id: Id | None = Field(
        default=None,
        description="section[].section_id of the section the figure is printed in (for PDF, "
        "the section being read where it appears); null before the first section.",
    )
    text_id: Id | None = Field(
        default=None,
        description="text[].text_id of the caption's row; null when no caption was found.",
    )
    image: str | None = Field(
        default=None,
        pattern=DATA_URI_PATTERN,
        description="The whole figure as a data URI naming its media type "
        "(data:image/jpeg;base64,...); a figure detected as several panel crops is composited "
        "from them, with white between panels. Null unless images are requested.",
    )
    caption: str | None = Field(default=None, description="Caption text, as printed.")
    page_number: Id | None = Field(description="1-based page the figure starts on.")


class TableExport(BaseModel):
    """One table."""

    model_config = _STRICT

    table_id: Id = Field(description="Primary key; 1-based position in document order.")
    label: str | None = Field(
        default=None,
        description="The printed label without the word, as printed with whitespace removed: "
        "'3', '3.1', 'S2', 'A1', 'IV'; a 'Supplementary Table 4' caption gives 'S4'. In-text "
        "references resolve by it. Not unique: the unmerged pieces of a "
        "table split across pages repeat it ('Table 3 (continued)'), and references target "
        "the piece not marked as continued. Null when none was printed or detected.",
    )
    section_id: Id | None = Field(
        default=None,
        description="section[].section_id of the section the table is printed in (for PDF, "
        "the section being read where it appears); null before the first section.",
    )
    text_id: Id | None = Field(
        default=None,
        description="text[].text_id of the caption's row; null when no caption was found.",
    )
    html: str | None = Field(
        default=None,
        description="HTML markup as printed; a table continued across pages has one <table> "
        "element per printed piece, joined by newlines.",
    )
    # ``PaperTable.contents`` (bibr/paper_contents.py) stringifies every cell —
    # headers via ``str(c)``, data cells via ``str(value)`` — so the runtime
    # shape is always a list of string rows, never numeric/None cells.
    contents: list[list[str]] = Field(
        description="Cell text as rows of strings; the first row is the header. A table "
        "continued across pages has all its pieces' rows merged."
    )
    caption: str | None = Field(default=None, description="Caption text, as printed.")
    page_number: Id | None = Field(description="1-based page the table starts on.")


class FootnoteExport(BaseModel):
    """One footnote or endnote. Its text is a row of ``text[]``, after the body."""

    model_config = _STRICT

    footnote_id: Id = Field(description="Primary key; 1-based position in document order.")
    label: str | None = Field(
        description="The marker the note is printed with, e.g. '1', '*' or '†'; null when none "
        "is printed or detected."
    )
    text_id: Id = Field(description="text[].text_id of the note's text row.")


class EqExport(BaseModel):
    """One statistical or mathematical expression found in a sentence,
    e.g. ``t(28) = 2.10``."""

    model_config = _STRICT

    eq_id: Id = Field(description="Primary key; 1-based position.")
    text_id: Id = Field(description="text[].text_id of the sentence containing the expression.")
    start: int | None = Field(
        default=None,
        ge=0,
        description="Where the printed expression starts in text[].text of text_id: a 0-based "
        "offset in Unicode code points; null when it could not be located.",
    )
    end: int | None = Field(
        default=None,
        ge=0,
        description="Where the printed expression ends (exclusive), in the same units as start.",
    )
    grp_id: Id = Field(
        description="Group of expressions reported together, e.g. the t, p and d of one test."
    )
    verbatim: str | None = Field(
        default=None,
        description="The expression exactly as printed (text[].text sliced by start and end); "
        "null when it could not be located.",
    )
    lhs: str = Field(description="Left-hand side, e.g. 't', 'p' or 'F'.")
    df: str | None = Field(
        description="Degrees of freedom printed with the left-hand side, e.g. '28' for t(28); "
        "null when none."
    )
    comp: EqCompLiteral = Field(
        description="Comparator, normalized: '<=' is '≤', '>=' is '≥', '<<' is '≪' and '>>' is '≫'."
    )
    rhs: str = Field(description="Right-hand side, e.g. '2.10' or '.003'.")


# ---------------------------------------------------------------------------
# Lookup (external services)
# ---------------------------------------------------------------------------


class MatchOrganizationExport(BaseModel):
    """An organization named on an external-service record."""

    model_config = _STRICT

    name: str | None = Field(
        default=None, description="Organization name, as the service gives it."
    )
    ror: str | None = Field(
        default=None,
        pattern=ROR_PATTERN,
        description="ROR ID as a URI (https://ror.org/...), when the record has one.",
    )


class MatchFunderExport(BaseModel):
    """A funder named on an external-service record."""

    model_config = _STRICT

    name: str | None = Field(default=None, description="Funder name, as the service gives it.")
    funder_doi: str | None = Field(
        default=None,
        pattern=DOI_PATTERN,
        description="Open Funder Registry DOI, bare and lowercase (10.13039/...), when recorded.",
    )
    ror: str | None = Field(
        default=None,
        pattern=ROR_PATTERN,
        description="ROR ID as a URI (https://ror.org/...), when recorded.",
    )
    award_ids: list[str] = Field(default_factory=list, description="Award numbers, as recorded.")


class PersonNameExport(BaseModel):
    """One structured person name from an external service.

    ``literal`` holds corporate/unsplittable names whole; when it is set,
    ``family``/``given`` are absent. Absent parts are omitted.
    """

    model_config = _STRICT

    family: str | None = Field(default=None, description="Family name.")
    given: str | None = Field(default=None, description="Given name(s) or initials.")
    suffix: str | None = Field(default=None, description="Name suffix such as 'Jr.'.")
    literal: str | None = Field(
        default=None, description="Whole name of an organization or an unsplittable name."
    )
    orcid: str | None = Field(
        default=None,
        pattern=ORCID_PATTERN,
        description="ORCID iD as a canonical URI (https://orcid.org/0000-0000-0000-0000), as "
        "the service records it.",
    )
    affiliation: list[MatchOrganizationExport] | None = Field(
        default=None,
        description="The person's affiliations on this record, as the service lists them.",
    )

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

    # Every part is written only when it has a value (CSL-JSON style).
    OMITTED_WHEN_ABSENT: ClassVar[tuple[str, ...]] = (
        "family",
        "given",
        "suffix",
        "literal",
        "orcid",
        "affiliation",
    )

    @model_serializer(mode="wrap")
    def _omit_absent_parts(self, handler):
        data = handler(self)
        return {k: v for k, v in data.items() if v is not None}


class BibMatchExport(BaseModel):
    """One external-service record matched to a reference-list entry."""

    model_config = _STRICT

    bib_id: Id = Field(description="bib[].bib_id of the entry this record matches.")
    service: MatchServiceLiteral = Field(description="External service that returned the record.")
    service_id: str | None = Field(
        default=None, description="The record's identifier at that service."
    )
    score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Match score, 0–1: 1 for a DOI lookup; a bibliographic search hit scores "
        "its title similarity after author and year checks.",
    )
    bib_type: BibTypeLiteral | None = Field(default=None, description="Work type at the service.")
    doi: str | None = Field(
        default=None, pattern=DOI_PATTERN, description="DOI of the matched record, lowercase."
    )
    title: str | None = Field(default=None, description="Title of the matched record.")
    author: list[PersonNameExport] | None = Field(
        default=None, description="Structured authors of the matched record; null when none."
    )
    editor: list[PersonNameExport] | None = Field(
        default=None, description="Structured editors of the matched record; null when none."
    )
    publisher: str | None = Field(default=None, description="Publisher.")
    year: int | None = Field(default=None, description="Publication year.")
    published_date: str | None = Field(
        default=None,
        pattern=ISO_DATE_PATTERN,
        description="Publication date as ISO 8601 (YYYY-MM or YYYY-MM-DD, as precise as the "
        "record); null when the record gives only a year (see year).",
    )
    container: str | None = Field(default=None, description="Journal, book or proceedings.")
    volume: str | None = Field(default=None, description="Volume.")
    issue: str | None = Field(default=None, description="Issue.")
    first_page: str | None = Field(default=None, description="First page or article number.")
    last_page: str | None = Field(default=None, description="Last page.")
    edition: str | None = Field(default=None, description="Edition.")
    version: str | None = Field(default=None, description="Version.")
    url: str | None = Field(default=None, description="URL of the record.")
    license_url: str | None = Field(
        default=None,
        description="License URL of the version of record (else of an unspecified version); "
        "text-mining licenses are skipped.",
    )
    license_spdx: str | None = Field(
        default=None,
        description="SPDX identifier of license_url, for Creative Commons licenses and CC0.",
    )
    funder: list[MatchFunderExport] | None = Field(
        default=None, description="Funders and awards the record lists."
    )


class MetadataMatchExport(BaseModel):
    """One external-service record matched to the paper itself (self-DOI lookup)."""

    model_config = _STRICT

    service: MatchServiceLiteral = Field(description="External service that returned the record.")
    service_id: str | None = Field(
        default=None, description="The record's identifier at that service."
    )
    score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Match score, 0–1: 1 for a DOI lookup; a bibliographic search hit scores "
        "its title similarity after author and year checks.",
    )
    bib_type: BibTypeLiteral | None = Field(default=None, description="Work type at the service.")
    doi: str | None = Field(
        default=None, pattern=DOI_PATTERN, description="DOI of the matched record, lowercase."
    )
    title: str | None = Field(default=None, description="Title of the matched record.")
    author: list[PersonNameExport] | None = Field(
        default=None, description="Structured authors of the matched record; null when none."
    )
    editor: list[PersonNameExport] | None = Field(
        default=None, description="Structured editors of the matched record; null when none."
    )
    publisher: str | None = Field(default=None, description="Publisher.")
    year: int | None = Field(default=None, description="Publication year.")
    published_date: str | None = Field(
        default=None,
        pattern=ISO_DATE_PATTERN,
        description="Publication date as ISO 8601 (YYYY-MM or YYYY-MM-DD, as precise as the "
        "record); null when the record gives only a year (see year).",
    )
    container: str | None = Field(default=None, description="Journal, book or proceedings.")
    volume: str | None = Field(default=None, description="Volume.")
    issue: str | None = Field(default=None, description="Issue.")
    first_page: str | None = Field(default=None, description="First page or article number.")
    last_page: str | None = Field(default=None, description="Last page.")
    edition: str | None = Field(default=None, description="Edition.")
    version: str | None = Field(default=None, description="Version.")
    url: str | None = Field(default=None, description="URL of the record.")
    license_url: str | None = Field(
        default=None,
        description="License URL of the version of record (else of an unspecified version); "
        "text-mining licenses are skipped.",
    )
    license_spdx: str | None = Field(
        default=None,
        description="SPDX identifier of license_url, for Creative Commons licenses and CC0.",
    )
    funder: list[MatchFunderExport] | None = Field(
        default=None, description="Funders and awards the record lists."
    )


class AffiliationMatchExport(BaseModel):
    """The ROR organization matched to one affiliation string.

    Only ROR's own recommended match (``chosen``) is kept; an affiliation
    without one has no row.
    """

    model_config = _STRICT

    affiliation_id: Id = Field(description="affiliation[].affiliation_id this record matches.")
    service: MatchServiceLiteral = Field(description="External service that returned the record.")
    service_id: str = Field(
        pattern=ROR_PATTERN,
        description="The organization's ROR ID as a URI (https://ror.org/...).",
    )
    score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="The service's match score, 0–1 (for ranking, not selection).",
    )
    name: str | None = Field(default=None, description="The organization's display name at ROR.")
    country_code: str | None = Field(
        default=None,
        pattern=COUNTRY_CODE_PATTERN,
        description="ISO 3166-1 alpha-2 country code of the organization.",
    )


class FundingMatchExport(BaseModel):
    """The ROR organization matched to one printed funder name.

    Only ROR's own recommended match (``chosen``) is kept; a funder without
    one has no row.
    """

    model_config = _STRICT

    funding_id: Id = Field(description="funding[].funding_id this record matches.")
    service: MatchServiceLiteral = Field(description="External service that returned the record.")
    service_id: str = Field(
        pattern=ROR_PATTERN,
        description="The organization's ROR ID as a URI (https://ror.org/...).",
    )
    score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="The service's match score, 0–1 (for ranking, not selection).",
    )
    name: str | None = Field(default=None, description="The organization's display name at ROR.")
    country_code: str | None = Field(
        default=None,
        pattern=COUNTRY_CODE_PATTERN,
        description="ISO 3166-1 alpha-2 country code of the organization.",
    )
    funder_doi: str | None = Field(
        default=None,
        pattern=DOI_PATTERN,
        description="The organization's Open Funder Registry DOI, bare and lowercase "
        "(10.13039/...), when ROR records one.",
    )


# ---------------------------------------------------------------------------
# Processing: extraction provenance, diagnostics, validation
# ---------------------------------------------------------------------------


class RegionExport(BaseModel):
    """One layout region of a page (opt-in ``include_regions`` debug payload)."""

    model_config = _STRICT

    page: Id = Field(description="1-based page number.")
    index: int = Field(
        description="0-based position of the region among its page's regions after OCR "
        "post-processing."
    )
    label: str | None = Field(description="Layout class, e.g. 'text', 'title', 'table'.")
    bbox: Box | None = Field(
        description="Bounding box [x0, y0, x1, y1] in PDF points from the top-left corner of the "
        "displayed page (see extraction.pages)."
    )
    font_size: float | None = Field(default=None, description="Dominant font size, in points.")
    font_weight: float | None = Field(default=None, description="Dominant font weight.")
    font_bold: bool | None = Field(default=None, description="Whether the dominant font is bold.")
    section_id: int | None = Field(
        default=None, description="section[].section_id the region was assigned to."
    )
    content: str | None = Field(default=None, description="Region text after post-processing.")
    raw_ocr_content: str | None = Field(default=None, description="Region text as OCR returned it.")
    char_density: float | None = Field(
        default=None, description="Characters per square point of bounding-box area."
    )
    estimated_line_height: float | None = Field(
        default=None,
        description="Bounding-box height divided by the region's line count, in points.",
    )


class TextRegionExport(BaseModel):
    """Layout features of one text[] row (opt-in ``include_region_meta``;
    training and debug payload)."""

    model_config = _STRICT

    text_id: Id = Field(description="text[].text_id these features describe.")
    font_size: float | None = Field(default=None, description="Font size, in points.")
    font_bold: bool | None = Field(default=None, description="Whether the font is bold.")
    is_italic: bool | None = Field(default=None, description="Whether the font is italic.")
    page_number: Id | None = Field(default=None, description="1-based page of the source region.")
    bbox: Box | None = Field(
        default=None,
        description="Bounding box of the source region: [x0, y0, x1, y1] in PDF points from the "
        "top-left corner of the displayed page (see extraction.pages).",
    )
    region_type: str | None = Field(default=None, description="Layout class of the source region.")


class PageExport(BaseModel):
    """Size of one page of the input document, the frame of every bounding box."""

    model_config = _STRICT

    page_number: Id = Field(description="1-based page number.")
    width: float = Field(description="Width of the page as displayed, in PDF points (1/72 inch).")
    height: float = Field(description="Height of the page as displayed, in PDF points.")


class FloatPartExport(BaseModel):
    """Where one physical piece of a figure or table was printed: a panel crop,
    or one page of a table continued across pages."""

    model_config = _STRICT

    object_type: Literal["figure", "table"] = Field(description="'figure' or 'table'.")
    object_id: Id = Field(description="figure[].figure_id or table[].table_id, by object_type.")
    part_index: Id = Field(description="1-based position of the piece within its object.")
    page_number: Id | None = Field(description="1-based page the piece is printed on.")
    bbox: Box | None = Field(
        description="Bounding box [x0, y0, x1, y1] in PDF points from the top-left corner of the "
        "displayed page (see extraction.pages)."
    )


class ExpectedIdentityExport(BaseModel):
    """What the caller said the input should be (queue-driven runs)."""

    model_config = _STRICT

    queue_record_id: str = Field(description="Caller's record identifier.")
    expected_doi: str | None = Field(default=None, description="DOI the caller expected.")
    expected_doi_sha256: str | None = Field(
        default=None, description="sha256 of the case-folded expected DOI."
    )
    expected_title: str | None = Field(default=None, description="Title the caller expected.")
    target_block_hint: dict[str, object] | None = Field(
        default=None, description="Caller's hint for locating the target article in the file."
    )
    source_sha256: str | None = Field(
        default=None, description="sha256 the caller expected for the input file."
    )
    doi_required: bool = Field(
        default=False, description="Whether a missing or mismatched DOI fails the run."
    )


class DoiCandidateExport(BaseModel):
    """One source-visible DOI and where it was read.

    ``page`` and ``region_index`` name the layout region the DOI was read
    from: the ``extraction.regions`` row with the same ``page`` and ``index``.
    That index is the region's 0-based position among its page's regions after
    OCR post-processing. Post-processing renumbers a page when it drops
    duplicate regions or merges split ones (a formula and its equation number,
    a word hyphenated across two blocks), so the index can differ from the
    layout detector's original slot.

    A ``sentence`` candidate names the region that began the sentence's
    paragraph (``text[].paragraph_id``), which ``region_type`` labels. If the
    parser joined following regions into that paragraph, the DOI may be
    printed in one of those instead.

    ``region_index`` is null when no layout region on ``page`` is recorded:
    header and footer furniture, structured metadata, captions, footnotes,
    input parsed without layout analysis, and a sentence printed on a later
    page than the region that began its paragraph.
    """

    model_config = _STRICT

    raw: str = Field(description="The DOI as printed.")
    normalized: str = Field(description="The DOI normalized for comparison.")
    source_kind: str = Field(
        description="Where it was read, e.g. 'sentence', 'text', 'structured_metadata' or "
        "'publication_region'."
    )
    page: int | None = Field(description="1-based page, when known.")
    section_id: int | None = Field(description="section[].section_id, when known.")
    section_type: str | None = Field(
        description="section_type of that section, when known; 'figure', 'table' or 'footnote' "
        "for a caption or footnote row, whose section_id is null."
    )
    region_index: int | None = Field(
        description="0-based region position on the page; see the model description."
    )
    region_type: str | None = Field(description="Layout class of that region.")
    text_id: int | None = Field(description="text[].text_id of the sentence, when from text.")
    marker_kind: str = Field(
        description="Label printed before the DOI, e.g. 'article_doi', 'data_doi' or "
        "'structured_doi'."
    )
    repeated_header_footer_count: int = Field(
        description="On how many pages the same DOI appears as header or footer furniture."
    )
    semantic_context: str = Field(description="Surrounding-context class used in selection.")
    selection_tier: int = Field(
        description="Selection strength: 4 matches the caller's expected DOI, 3 is an explicit "
        "self-identification, 2 is front matter or repeated furniture, 1 is uncontested and "
        "untyped."
    )
    rejection_reason: str | None = Field(
        default=None, description="Why the candidate was not selected; null when eligible."
    )


class IdentityReceiptExport(BaseModel):
    """Which DOI was chosen as the paper's own, and from which candidates."""

    model_config = _STRICT

    selected: DoiCandidateExport | None = Field(
        default=None, description="The chosen candidate; null when none qualified."
    )
    candidates: list[DoiCandidateExport] = Field(
        default_factory=list, description="Every DOI candidate considered."
    )
    issue_codes: list[str] = Field(
        default_factory=list, description="Identity issue codes raised during selection."
    )


class IdentityExport(BaseModel):
    """Caller-supplied expectations and the DOI-selection receipt."""

    model_config = _STRICT

    expected: ExpectedIdentityExport | None = Field(
        default=None, description="What the caller expected; omitted when nothing was supplied."
    )
    receipt: IdentityReceiptExport | None = Field(
        default=None, description="The DOI-selection receipt; omitted when selection did not run."
    )

    OMITTED_WHEN_ABSENT: ClassVar[tuple[str, ...]] = ("expected", "receipt")

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler):
        data = handler(self)
        for key in self.OMITTED_WHEN_ABSENT:
            if data.get(key) is None:
                data.pop(key, None)
        return data


class EnrichmentExport(BaseModel):
    """Reference-enrichment completeness — lets a consumer distinguish a
    partial (timed-out) enrichment from a complete one without scanning
    ``extraction.warnings``."""

    model_config = _STRICT

    complete: bool = Field(
        description="False when enrichment stopped before every reference was looked up, "
        "e.g. on a timeout."
    )
    refs_enriched: int = Field(description="References with at least one bib_match row.")
    refs_total: int = Field(description="References in bib[].")


class CitationCandidateExport(BaseModel):
    """One in-text citation the detector considered."""

    model_config = _STRICT

    text_id: int = Field(description="text[].text_id of the sentence.")
    start: int = Field(description="Start character offset within the sentence.")
    end: int = Field(description="End character offset (exclusive).")
    raw: str = Field(description="The citation text as printed.")
    style: str = Field(description="Citation style the candidate was read as.")
    bib_ids: list[int] = Field(description="bib[].bib_id values the candidate links to.")
    evidence: list[str] = Field(description="Signals supporting the link.")
    confidence: float = Field(description="Detector confidence.")
    accepted: bool = Field(description="Whether the candidate became an xref row.")
    rejection_reasons: list[str] = Field(description="Why a rejected candidate was dropped.")


class CitationLinkingExport(BaseModel):
    """Evidence-bearing receipt for inline citation linking."""

    model_config = _STRICT

    style_scores: dict[str, float] = Field(description="Score per citation style for this paper.")
    candidates: list[CitationCandidateExport] = Field(description="Every candidate considered.")
    resolved_candidate_fraction: float | None = Field(
        description="Share of candidates linked to at least one reference."
    )
    unique_linked_bib_fraction: float | None = Field(
        description="Share of references cited at least once."
    )


class CaptionCandidateExport(BaseModel):
    """One caption found in the source."""

    model_config = _STRICT

    caption_id: str = Field(description="Receipt-local caption identifier.")
    text: str = Field(description="Caption text.")
    object_type: str = Field(description="'figure' or 'table'.")
    page_number: int | None = Field(description="1-based page.")
    bbox: Box | None = Field(
        description="Bounding box [x0, y0, x1, y1] in PDF points from the top-left corner of the "
        "displayed page (see extraction.pages)."
    )
    source_index: int = Field(description="Position of the caption in the source order.")


class CaptionAssignmentExport(BaseModel):
    """One caption-to-float decision."""

    model_config = _STRICT

    caption_id: str = Field(description="caption_id of the caption.")
    object_id: str | None = Field(
        description="'figure:<figure_id>' or 'table:<table_id>' it was assigned to; null when "
        "unassigned, or when that float did not survive to the export."
    )
    score: float = Field(description="Assignment score.")
    reasons: list[str] = Field(description="Signals behind the decision.")
    ambiguous: bool = Field(default=False, description="Whether another float scored as well.")


class CaptionAssignmentReceiptExport(BaseModel):
    """How captions were matched to figures and tables."""

    model_config = _STRICT

    candidates: list[CaptionCandidateExport] = Field(description="Every caption found.")
    assignments: list[CaptionAssignmentExport] = Field(description="Every assignment decision.")


class ReferenceSegmentationAttemptExport(BaseModel):
    """One attempt at splitting the reference section into entries."""

    model_config = _STRICT

    strategy: str = Field(description="Segmentation strategy tried.")
    spans: list[list[int]] = Field(description="Character spans [start, end] of the entries found.")
    credible_starts: int | None = Field(description="Entries that start like a reference.")
    selected: bool = Field(description="Whether this attempt was used.")
    reason_flags: list[str] = Field(description="Why the attempt was or was not selected.")


class ReferenceYieldExport(BaseModel):
    """How many references were found, and how the reference section was split."""

    model_config = _STRICT

    credible_source_starts: int | None = Field(
        description="Entry starts visible in the source reference section."
    )
    attempts: list[ReferenceSegmentationAttemptExport] = Field(
        description="Every segmentation attempt."
    )
    selected_spans: list[list[int]] = Field(description="Spans of the selected attempt.")
    source_character_coverage: float | None = Field(
        description="Share of the reference section's characters covered by entries."
    )
    parsed_count: int = Field(description="Entries parsed.")
    valid_count: int = Field(description="Parsed entries that passed validity checks.")
    duplicate_rate: float = Field(description="Share of entries that duplicate another.")
    reason_flags: list[str] = Field(description="Yield warnings, e.g. suspected under-extraction.")


class SectionClassificationExport(BaseModel):
    """How one section's ``section_type`` was decided."""

    model_config = _STRICT

    section_id: Id = Field(description="section[].section_id.")
    score: float | None = Field(
        ge=0, le=1, description="Classifier confidence, 0–1; null when not scored."
    )
    source: str | None = Field(
        description="Which tier decided the type: 'exact_alias', 'substring_alias' or "
        "'alias_prior' (heading lookup), 'model' (trained classifier), 'llm', "
        "'parent_context', 'title', 'implicit', 'positional', 'appendix_repair' or "
        "'imrad_dedup'; null when unclassified."
    )


class XrefTierExport(BaseModel):
    """How one bibliography, figure or table xref was linked."""

    model_config = _STRICT

    xref_id: Id = Field(description="xref[].xref_id.")
    tier: XrefTierLiteral = Field(
        description="How the link was made: for a bib reference, the citation detector that "
        "found it; for a figure or table reference, 'label' (matched against the printed "
        "labels) or 'position' (no float of that kind has a label, so the printed number was "
        "taken as a position)."
    )


class PaperClassificationExport(BaseModel):
    """Confidence of the paper-level classifications in ``metadata``."""

    model_config = _STRICT

    paper_type_confidence: float | None = Field(
        default=None, ge=0, le=1, description="Confidence of metadata.paper_type, 0–1."
    )
    oecd_confidence: float | None = Field(
        default=None, ge=0, le=1, description="Confidence of metadata.oecd_l1 and oecd_l2, 0–1."
    )


class ConsolidationExport(BaseModel):
    """Fields of one reference that consolidation took from ``bib_match``."""

    model_config = _STRICT

    bib_id: Id = Field(description="bib[].bib_id of the modified entry.")
    fields: list[str] = Field(description="bib[] field names filled or replaced, in field order.")


class DiagnosticsExport(BaseModel):
    """Outcome flags and stage receipts — never raw payloads.

    ``regions``, ``text_regions`` and ``trace`` are deliberately NOT here: they
    are megabyte-scale intermediate data, and bundling them would force every
    diagnostics consumer to reason about size. Receipts are omitted when their
    stage did not run.
    """

    model_config = _STRICT

    text_quality: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Text-layer quality score, 0–1; null when not measured.",
    )
    references_complete: bool = Field(
        default=True, description="False when reference extraction is known to be incomplete."
    )
    ref_seg_fallback_used: bool = Field(
        default=False, description="Whether reference segmentation fell back to the CRF model."
    )
    section_classification: list[SectionClassificationExport] | None = Field(
        default=None, description="One row per section[] row: how its section_type was decided."
    )
    paper_classification: PaperClassificationExport | None = Field(
        default=None, description="Confidence of metadata.paper_type and metadata.oecd_*."
    )
    xref_tier: list[XrefTierExport] | None = Field(
        default=None,
        description="How each bibliography, figure and table xref that recorded it was linked.",
    )
    citation_linking: CitationLinkingExport | None = Field(
        default=None, description="Receipt for inline citation linking."
    )
    caption_assignment: CaptionAssignmentReceiptExport | None = Field(
        default=None, description="Receipt for caption-to-float assignment."
    )
    reference_yield: ReferenceYieldExport | None = Field(
        default=None, description="Receipt for reference segmentation and yield."
    )
    consolidation: list[ConsolidationExport] | None = Field(
        default=None,
        description="Per modified reference, the bib[] fields consolidation took from bib_match; "
        "omitted when consolidation did not run.",
    )

    OMITTED_WHEN_ABSENT: ClassVar[tuple[str, ...]] = (
        "section_classification",
        "paper_classification",
        "xref_tier",
        "citation_linking",
        "caption_assignment",
        "reference_yield",
        "consolidation",
    )

    @model_serializer(mode="wrap")
    def _omit_absent_receipts(self, handler):
        data = handler(self)
        for key in self.OMITTED_WHEN_ABSENT:
            if data.get(key) is None:
                data.pop(key, None)
        return data


class OcrEngineExport(BaseModel):
    """Which OCR engine produced this output. ``null`` on the native-parse path."""

    model_config = _STRICT

    backend: str | None = Field(default=None, description="OCR backend.")
    model: str | None = Field(default=None, description="OCR model identifier.")
    profile: Literal["paddle", "glm"] | None = Field(default=None, description="OCR profile.")


class LlmEngineExport(BaseModel):
    """The CORE extraction LLM. ``null`` when the run deliberately had none.

    ``usage.breakdown`` is authoritative for every engine that actually ran;
    when the two disagree (multi-engine run), breakdown wins.
    """

    model_config = _STRICT

    provider: str | None = Field(default=None, description="LLM provider.")
    model: str | None = Field(default=None, description="Model identifier.")
    backend: str | None = Field(default=None, description="Structured-output backend.")


class ExtractionSettingsExport(BaseModel):
    """Per-run resolutions of the reference/enrichment knobs."""

    model_config = _STRICT

    ref_seg: str = Field(description="Reference segmentation strategy used.")
    ref_parse: str = Field(description="Reference parsing strategy used.")
    crossref_enrich: bool = Field(description="Whether Crossref enrichment was enabled.")
    consolidate: str = Field(description="Consolidation mode: 'off', 'fill' or 'replace'.")


class TimingsExport(BaseModel):
    """Per-stage wall-clock seconds (the export stage's own time is excluded —
    it is still running when this is built).

    Both fields are always populated when this object exists: an untimed run
    omits the whole ``extraction.timings`` object rather than emitting one with
    null fields. The ``| None`` types are structural only.
    """

    model_config = _STRICT

    stages: dict[str, float] | None = Field(default=None, description="Seconds per stage.")
    total_seconds: float | None = Field(default=None, description="Seconds for the whole run.")


class UsageRowExport(BaseModel):
    """One ``(label, provider, model)`` LLM usage row."""

    model_config = _STRICT

    label: str = Field(description="Call site, e.g. 'extract_authors'.")
    provider: str | None = Field(default=None, description="LLM provider.")
    model: str | None = Field(default=None, description="Model identifier.")
    calls: int = Field(description="Number of calls.")
    input_tokens: int = Field(description="Input tokens, including cached ones.")
    cached_input_tokens: int = Field(description="Input tokens served from the provider cache.")
    output_tokens: int = Field(description="Output tokens.")
    total_tokens: int = Field(description="Input plus output tokens.")


class UsageTotalsExport(BaseModel):
    """Aggregate of every ``breakdown`` row — denormalized deliberately."""

    model_config = _STRICT

    calls: int = Field(description="Number of calls.")
    input_tokens: int = Field(description="Input tokens, including cached ones.")
    cached_input_tokens: int = Field(description="Input tokens served from the provider cache.")
    output_tokens: int = Field(description="Output tokens.")
    total_tokens: int = Field(description="Input plus output tokens.")


class UsageExport(BaseModel):
    """LLM token usage for the run."""

    model_config = _STRICT

    totals: UsageTotalsExport = Field(description="Sum over all breakdown rows.")
    breakdown: list[UsageRowExport] = Field(
        default_factory=list, description="One row per (label, provider, model)."
    )


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

    label: str = Field(description="Call site.")
    provider: str | None = Field(default=None, description="LLM provider.")
    model: str | None = Field(default=None, description="Model identifier.")
    messages: list[dict] = Field(default_factory=list, description="Chat messages sent.")
    raw_completion: str | None = Field(default=None, description="Raw model output.")
    parsed_ok: bool = Field(description="Whether the output parsed into the expected structure.")
    finish_reason: str | None = Field(default=None, description="Provider finish reason.")
    # Resolved sampling parameters (temperature/top_p/reasoning_effort/...,
    # provider-shaped — e.g. the Google adapter nests temperature under
    # generation_config). Mandatory in spirit: a completion recorded without
    # them cannot be told apart from a lucky generation, so the call site
    # always populates this from ``_build_call_kwargs``'s resolved output.
    # Untyped ``dict`` (pydantic does not validate its values), so
    # ``_record_trace`` recursively coerces every leaf to a JSON-safe
    # primitive and scrubs strings before this is ever constructed — see
    # ``bibr.clients.llm._sanitize_trace_value``.
    params: dict = Field(
        default_factory=dict, description="Resolved sampling parameters, provider-shaped."
    )
    attempt: int = Field(default=1, description="Attempt number; currently always 1.")
    error: str | None = Field(default=None, description="Error message of a failed call.")


class ValidationIssueExport(BaseModel):
    """One finding of the output validation gate."""

    model_config = _STRICT

    code: str = Field(description="Stable issue code, e.g. 'VAL_AUTHOR_BLANK'.")
    severity: SeverityLiteral = Field(description="'error' or 'warning'.")
    message: str = Field(description="Human-readable description.")
    origin_stage: str = Field(default="export", description="Pipeline stage that raised it.")
    evidence_ids: list[str] = Field(
        default_factory=list, description="Identifiers of the rows involved."
    )
    count: int = Field(description="Number of occurrences.")
    blocking: bool = Field(default=False, description="Whether it prevents promotion.")


class ValidationExport(BaseModel):
    """Output validation gate result and promotion disposition."""

    model_config = _STRICT

    errors: int = Field(description="Issues with severity 'error'.")
    warnings: int = Field(description="Issues with severity 'warning'.")
    blocking: int = Field(default=0, description="Issues that block promotion.")
    promotable: bool = Field(default=True, description="True when no issue blocks promotion.")
    issues: list[ValidationIssueExport] = Field(description="Every finding.")


class ProducerExport(BaseModel):
    """The software that wrote the export."""

    model_config = _STRICT

    name: str = Field(
        min_length=1,
        description="Name of the producing software: 'bibr', or another tool that writes this "
        "format (e.g. a converter from GROBID TEI).",
    )
    version: str = Field(
        min_length=1,
        description="Version of the producing software (not the schema version).",
    )
    build_sha: str | None = Field(
        default=None, description="Commit SHA of the producer's build, when known."
    )


# UPPER_SNAKE, like the validation issue codes. A pattern, not an enum: another
# producer of the format may add codes of its own.
WARNING_CODE_PATTERN = r"^[A-Z][A-Z0-9_]*$"


class WarningExport(BaseModel):
    """One non-fatal processing warning."""

    model_config = _STRICT

    code: str = Field(
        pattern=WARNING_CODE_PATTERN,
        description="Stable, machine-readable warning code in UPPER_SNAKE case, e.g. "
        "'OCR_PAGE_FAILED'. bibr's codes are listed in its JSON schema reference; other "
        "producers may add their own, so readers must accept a code they do not know.",
    )
    message: str = Field(
        description="Human-readable details, such as the page, counts or exception type."
    )


class ExtractionExport(BaseModel):
    """How this output was produced — everything that is not the paper itself.

    Built by ``ExportStage._build_extraction`` from the pipeline context; the
    export function folds in the receipts that exist only at serialization
    time, and the output validation gate adds ``validation``. Always present:
    a Paper exported outside the pipeline gets a minimal block (producer,
    export time, the diagnostics the Paper carries, validation).
    """

    model_config = _STRICT

    producer: ProducerExport = Field(description="The software that wrote this export.")
    # UTC ISO-8601 export timestamp — the export's only wall-clock provenance.
    # Excluded from fixture/replay diffs (see tests) so it stays deterministic.
    completed_at: str = Field(
        pattern=UTC_TIMESTAMP_PATTERN,
        json_schema_extra={"format": "date-time"},
        description="Time the export was written, as a UTC ISO 8601 timestamp "
        "(2026-01-15T09:30:00Z).",
    )
    ocr: OcrEngineExport | None = Field(
        default=None, description="OCR engine; null when the run used none."
    )
    llm: LlmEngineExport | None = Field(
        default=None, description="Core extraction LLM; null when the run used none."
    )
    settings: ExtractionSettingsExport | None = Field(
        default=None,
        description="Resolved per-run settings; omitted when the export was made outside the "
        "pipeline.",
    )
    timings: TimingsExport | None = Field(
        default=None, description="Wall-clock timings; omitted when untimed."
    )
    usage: UsageExport | None = Field(
        default=None, description="LLM token usage; omitted when usage tracking is off."
    )
    enrichment: EnrichmentExport | None = Field(
        default=None, description="Enrichment completeness; omitted when enrichment did not run."
    )
    identity: IdentityExport | None = Field(
        default=None, description="DOI identity expectations and receipt."
    )
    diagnostics: DiagnosticsExport | None = Field(
        default=None, description="Outcome flags and stage receipts."
    )
    validation: ValidationExport | None = Field(
        default=None,
        description="Output validation gate result: error/warning counts, promotion disposition "
        "and the issue list. Omitted when the gate was skipped.",
    )
    qualification: dict[str, Any] | None = Field(
        default=None,
        description="Deployment-qualification provenance read by the external qualification "
        "runner: identity SHAs, per-task protocol hashes, native-validity and fallback outcome, "
        "request counts. Hashes and counts only, never document text. Omitted when no LLM task "
        "ran or usage tracking is off.",
    )
    warnings: list[WarningExport] = Field(
        default_factory=list,
        description="Non-fatal processing warnings, each a stable code and a message.",
    )
    pages: list[PageExport] | None = Field(
        default=None,
        description="Size of every page with layout analysis: the frame of the bounding boxes "
        "in extraction. A box is [x0, y0, x1, y1] in PDF points (1/72 inch) on the page as "
        "displayed (after its /Rotate, within its CropBox), measured from the top-left corner "
        "with y increasing downward. Omitted for inputs without pages (DOCX, XML, HTML, ePub).",
    )
    # Opt-in heavy payloads — siblings, never nested under a hot key.
    regions: list[RegionExport] | None = Field(
        default=None, description="Per-region layout debug payload; opt-in (include_regions)."
    )
    text_regions: list[TextRegionExport] | None = Field(
        default=None,
        description="Per-sentence layout features, one row per text[] row that has them; opt-in "
        "(include_region_meta).",
    )
    float_parts: list[FloatPartExport] | None = Field(
        default=None,
        description="Page and bounding box of every printed piece of the figures and tables; "
        "omitted when no piece has a location (natively parsed inputs).",
    )
    trace: list[LlmTraceExport] | None = Field(
        default=None, description="Captured LLM calls; opt-in (LLM_CAPTURE_TRACE)."
    )

    # Absence rule: omitted = the subsystem did not run / was not requested.
    # ``ocr``/``llm`` are NOT in this list — ``null`` there is meaningful (the
    # run happened, deliberately without that engine).
    OMITTED_WHEN_ABSENT: ClassVar[tuple[str, ...]] = (
        "settings",
        "enrichment",
        "identity",
        "diagnostics",
        "validation",
        "qualification",
        "timings",
        "usage",
        "pages",
        "regions",
        "text_regions",
        "float_parts",
        "trace",
    )

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler):
        data = handler(self)
        for key in self.OMITTED_WHEN_ABSENT:
            if data.get(key) is None:
                data.pop(key, None)
        return data


# The ONLY root keys the exporter may omit (see ``PaperExport._omit_absent``);
# since v12 there are none. Every root key is always emitted — record arrays
# because the uniform-tables contract requires them (empty allowed), the
# object blocks because they are unconditional.
#
# This tuple, and the ``OMITTED_WHEN_ABSENT`` of the few nested models that
# drop an absent key, are the single authority on which keys may be missing:
# ``bibr.export.schema_artifact`` makes every other key ``required`` in the
# published schema, so the artifact can never claim a key is optional that the
# exporter in fact always emits.
OMITTABLE_ROOT_KEYS: tuple[str, ...] = ()


class PaperExport(BaseModel):
    """A research paper as extracted by bibr (export schema v12).

    ``extraction`` holds everything about how the output was produced.
    Everything else is the paper: ``paper_id``, ``schema_version`` and
    ``source`` identify the paper and its input file; ``metadata`` and the
    record tables ``author``, ``affiliation``, ``funding``, ``text``,
    ``section``, ``url``, ``bib``, ``xref``, ``figure``, ``table`` and ``eq``
    hold what the paper says; the ``*_match`` tables hold what external
    registries returned for the paper, its affiliations, funders and references.

    Every record table has an integer primary key named ``<table>_id``,
    1-based; the other ``*_id`` columns are foreign keys to the table they name.
    Absent values are ``null``, never an empty-string or zero sentinel. Every
    root key is always present; record tables may be empty. Readers dispatch on
    the presence of the root ``schema_version``. Character offsets count
    Unicode code points of the exported text. Every score and confidence is
    0–1. Identifiers use each registry's canonical form: DOIs bare and
    lowercase, ORCID iDs and ROR IDs as https URIs. Every bounding box is
    ``[x0, y0, x1, y1]`` in PDF points (1/72 inch) on the page as displayed,
    measured from its top-left corner with y increasing downward;
    ``extraction.pages`` gives each page's size in the same units.
    """

    model_config = _STRICT

    paper_id: str = Field(
        min_length=1,
        description="Paper identifier and the key that joins this paper's rows across files: "
        "the user-supplied --paper-id, else the input file's name without its extension "
        "(bibr batch writes its corpus-unique id, the name of the JSON file).",
    )
    schema_version: Literal["12.0"] = Field(
        description="Export schema version. Its presence at the root is how readers "
        "distinguish v11 and later from all earlier versions.",
    )
    source: SourceExport = Field(
        description="Identity of the input file: name, SHA-256 digest, and format.",
    )
    metadata: MetadataExport = Field(
        description="Paper-level metadata: title, abstract, keywords, DOI, paper type and OECD "
        "classification, the paper's own journal/venue identity, and research-integrity "
        "statement text.",
    )
    author: list[AuthorExport] = Field(
        description="Authors in byline order, with name, email, corresponding-author flag, "
        "ORCID, and contribution roles.",
    )
    affiliation: list[AffiliationExport] = Field(
        [],
        description="Distinct byline affiliations: verbatim text, best-effort parsed components, "
        "and the authors they belong to.",
    )
    funding: list[FundingExport] = Field(
        [],
        description="Funders and award numbers parsed from the funding statement.",
    )
    text: list[TextExport] = Field(
        description="Document text as ordered sentence-level spans, each linked to its "
        "paragraph, section and page; caption and footnote rows follow the body.",
    )
    section: list[SectionExport] = Field(
        description="Sections with their headings, hierarchy and section type.",
    )
    url: list[UrlExport] = Field(
        description="Hyperlinks in the text, with link text and source sentence.",
    )
    bib: list[BibExport] = Field(
        description="Reference-list entries, parsed from the printed entries.",
    )
    xref: list[XrefExport] = Field(
        description="In-text references linking a sentence to a reference-list entry, table, "
        "figure, footnote, equation, supplement or section.",
    )
    figure: list[FigureExport] = Field(
        description="Figures with caption, page and optional image data.",
    )
    table: list[TableExport] = Field(
        description="Tables with HTML markup, cell contents, caption and page.",
    )
    footnote: list[FootnoteExport] = Field(
        [],
        description="Footnotes and endnotes: the printed marker and the note's text row.",
    )
    eq: list[EqExport] = Field(
        description="Statistical and mathematical expressions, split into left-hand side, "
        "degrees of freedom, comparator and right-hand side.",
    )
    metadata_match: list[MetadataMatchExport] = Field(
        [],
        description="External-service records matched to the paper itself (self-DOI lookup).",
    )
    affiliation_match: list[AffiliationMatchExport] = Field(
        [],
        description="ROR organizations matched to affiliation strings; enrichment runs only.",
    )
    funding_match: list[FundingMatchExport] = Field(
        [],
        description="ROR organizations matched to printed funder names; enrichment runs only.",
    )
    bib_match: list[BibMatchExport] = Field(
        [],
        description="External-service records matched to reference-list entries, one row per hit.",
    )
    extraction: ExtractionExport = Field(
        description="How this output was produced — everything that is not the paper itself: "
        "engines, settings, timings, LLM usage, enrichment, identity receipts, diagnostics, "
        "qualification provenance, validation and warnings.",
    )

    OMITTED_WHEN_ABSENT: ClassVar[tuple[str, ...]] = OMITTABLE_ROOT_KEYS

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler):
        # Root record arrays are ALWAYS present (empty allowed) — metacheck's
        # uniform-tables contract needs every table to exist. Only keys listed
        # in OMITTABLE_ROOT_KEYS (none since v12) may be omitted.
        data = handler(self)
        for key in self.OMITTED_WHEN_ABSENT:
            if data.get(key) is None:
                data.pop(key, None)
        return data


# ---------------------------------------------------------------------------
# Lenient reader
#
# The models above pin what THIS bibr writes: unknown keys are forbidden and
# ``schema_version`` is exactly ``_SCHEMA_VERSION``. A reader needs the
# opposite, because the forward policy lets any later minor writer of the same
# major add fields at any nesting level and bump the minor version.
#
# ``PaperExportReader`` is generated rather than hand-written: every model
# reachable from ``PaperExport`` gets a subclass with ``extra="allow"``, each
# field that nests a model is re-pointed at that model's reader, and each
# closed vocabulary (``Literal``) is opened to ``str``, because a later minor
# may add a value (a new ``section_type``); the known values stay in the
# reader schema as ``examples``. The
# readers inherit validators, serializers, defaults, aliases and descriptions,
# stay in lockstep with the producer fields, and are instances of their
# producer classes, so code typed against ``PaperExport`` keeps working.
# pydantic's per-call ``extra=`` override also reaches nested models, but it
# is missing from the older pydantic 2.x releases bibr supports and cannot
# relax the root ``schema_version``.
# ---------------------------------------------------------------------------

_LENIENT = ConfigDict(extra="allow", populate_by_name=True)

_SCHEMA_MAJOR = _SCHEMA_VERSION.split(".")[0]

# Any minor of the major this bibr writes. A different major is a breaking
# change, so the reader refuses it. ``[0-9]`` rather than ``\d``, which
# pydantic's regex engine matches against any Unicode digit.
_READER_SCHEMA_VERSION_PATTERN = rf"^{_SCHEMA_MAJOR}\.[0-9]+$"

_ModelReaders = dict[type[BaseModel], type[BaseModel]]


def _reader_annotation(annotation: Any, readers: _ModelReaders) -> Any:
    """Return *annotation* with every nested export model swapped for its reader
    and every ``Literal`` vocabulary opened to ``str``."""
    origin = get_origin(annotation)
    if origin is Literal:
        return str
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return _reader_model(annotation, readers)
        return annotation
    args = get_args(annotation)
    reader_args = tuple(_reader_annotation(arg, readers) for arg in args)
    if reader_args == args:
        return annotation
    if origin in (Union, UnionType):
        return functools.reduce(operator.or_, reader_args)
    return origin[reader_args]


def _literal_values(annotation: Any) -> list[Any]:
    """The values of the first ``Literal`` inside *annotation* (none: ``[]``)."""
    if get_origin(annotation) is Literal:
        return list(get_args(annotation))
    for arg in get_args(annotation):
        values = _literal_values(arg)
        if values:
            return values
    return []


def _reader_model(
    model: type[BaseModel],
    readers: _ModelReaders,
    overrides: dict[str, tuple[Any, FieldInfo]] | None = None,
    doc: str | None = None,
) -> type[BaseModel]:
    """Return the lenient reader subclass of *model*, building it on first use."""
    if model in readers:
        return readers[model]
    # Models declared before the models they nest hold unresolved forward
    # references until rebuilt.
    model.model_rebuild()
    annotations: dict[str, Any] = {}
    fields: dict[str, FieldInfo] = {}
    for field_name, field in model.model_fields.items():
        annotation = _reader_annotation(field.annotation, readers)
        if annotation is not field.annotation:
            annotations[field_name] = annotation
            fields[field_name] = copy.copy(field)
            known = _literal_values(field.annotation)
            if known:
                fields[field_name].examples = known
    for field_name, (annotation, field) in (overrides or {}).items():
        annotations[field_name] = annotation
        fields[field_name] = field
    name = f"{model.__name__}Reader"
    metaclass: Any = type(model)  # pydantic's model metaclass, as in a class statement
    reader: type[BaseModel] = metaclass(
        name,
        (model,),
        {
            "__module__": __name__,
            "__qualname__": name,
            # Reused so the reader's JSON Schema keeps the producer's descriptions.
            "__doc__": doc or model.__doc__,
            "__annotations__": annotations,
            "model_config": _LENIENT,
            **fields,
        },
    )
    readers[model] = reader
    return reader


_PAPER_EXPORT_READER_DOC = f"""Lenient reader for any bibr {_SCHEMA_MAJOR}.x JSON export.

Accepts every export the strict v{_SCHEMA_MAJOR} schema accepts, plus what the
{_SCHEMA_MAJOR}.x forward policy allows: unknown keys at any nesting level,
enum values it does not know yet, and any ``schema_version`` of the form
``{_SCHEMA_MAJOR}.<minor>``. A different
major version, a missing root ``schema_version``, and a known field with the
wrong type are still rejected.

In Python, unknown keys are kept, not dropped: each model exposes them through
``model_extra`` and includes them in ``model_dump()``. Every nested reader
model subclasses its strict counterpart, so ``isinstance`` checks against
``PaperExport`` and the other strict models still hold.
"""

PaperExportReader = cast(
    "type[PaperExport]",
    _reader_model(
        PaperExport,
        {},
        overrides={
            "schema_version": (
                str,
                Field(
                    pattern=_READER_SCHEMA_VERSION_PATTERN,
                    description=f"Export schema version, any {_SCHEMA_MAJOR}.x minor. Its "
                    "presence at the root is how readers distinguish v11 and later from all "
                    "earlier versions.",
                ),
            )
        },
        doc=_PAPER_EXPORT_READER_DOC,
    ),
)
