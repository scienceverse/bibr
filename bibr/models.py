"""Data models for papers, authors, references, and metadata.

Pydantic v2 BaseModel subclasses; previously dataclasses. The migration
preserves keyword-argument constructors and field defaults exactly so
callers keep working. ``model_dump()`` replaces dataclass ``asdict()``
at serialization sites.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from bibr.extract.field_decisions import FieldDecisions


class _Base(BaseModel):
    """Common config — match previous dataclass behaviour: assignment is
    allowed without re-validation, and arbitrary types (e.g. ``MatchSource``
    enum keys in dicts) are tolerated.
    """

    model_config = ConfigDict(
        validate_assignment=False,
        arbitrary_types_allowed=True,
    )


class MatchOrganization(_Base):
    """An organization named on an external-service record."""

    name: str | None = None
    ror: str | None = None  # https://ror.org/... URI


class MatchFunder(_Base):
    """A funder named on an external-service record."""

    name: str | None = None
    funder_doi: str | None = None  # bare Open Funder Registry DOI, 10.13039/...
    ror: str | None = None  # https://ror.org/... URI
    award_ids: list[str] = Field(default_factory=list)


class OrganizationMatch(_Base):
    """The registry organization matched to a printed affiliation or funder name."""

    service_id: str  # ROR ID URI
    score: float | None = None
    name: str | None = None
    country_code: str | None = None
    funder_doi: str | None = None


class BibAuthor(_Base):
    """Lightweight author representation for bibliography entries and external matches."""

    given: str
    family: str
    # External matches only: identifiers the service records for the person.
    orcid: str | None = None
    affiliation: list[MatchOrganization] | None = None


def format_bib_authors(authors: list[BibAuthor]) -> str:
    """Format a list of BibAuthor as 'Family, Given; Family, Given'."""
    parts = []
    for a in authors:
        if a.family:
            parts.append(f"{a.family}, {a.given}" if a.given else a.family)
    return "; ".join(parts) if parts else ""


class BibType(StrEnum):
    """Reference type taxonomy."""

    JOURNAL_ARTICLE = "journal_article"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    DATASET = "dataset"
    SOFTWARE = "software"
    PREPRINT = "preprint"
    CONFERENCE_PAPER = "conference_paper"
    REPORT = "report"
    THESIS = "thesis"
    OTHER = "other"


# Legacy BibTeX types and the Crossref ``type`` vocabulary, to ``BibType``.
_BIB_TYPE_ALIASES: dict[str, str] = {
    "article": BibType.JOURNAL_ARTICLE.value,
    "journal_article": BibType.JOURNAL_ARTICLE.value,
    "journal-article": BibType.JOURNAL_ARTICLE.value,
    "book": BibType.BOOK.value,
    # Crossref files monographs, edited volumes and reference works as their
    # own types; left out, a matched book came back as "other".
    "monograph": BibType.BOOK.value,
    "edited-book": BibType.BOOK.value,
    "reference-book": BibType.BOOK.value,
    "book-set": BibType.BOOK.value,
    "inbook": BibType.BOOK_CHAPTER.value,
    "incollection": BibType.BOOK_CHAPTER.value,
    "book_chapter": BibType.BOOK_CHAPTER.value,
    "book-chapter": BibType.BOOK_CHAPTER.value,
    "book-section": BibType.BOOK_CHAPTER.value,
    "book-part": BibType.BOOK_CHAPTER.value,
    "book-track": BibType.BOOK_CHAPTER.value,
    "conference": BibType.CONFERENCE_PAPER.value,
    "inproceedings": BibType.CONFERENCE_PAPER.value,
    "proceedings": BibType.CONFERENCE_PAPER.value,
    "proceedings-article": BibType.CONFERENCE_PAPER.value,
    "conference_paper": BibType.CONFERENCE_PAPER.value,
    "techreport": BibType.REPORT.value,
    "report": BibType.REPORT.value,
    "report-component": BibType.REPORT.value,
    "report-series": BibType.REPORT.value,
    "unpublished": BibType.PREPRINT.value,
    "preprint": BibType.PREPRINT.value,
    "posted-content": BibType.PREPRINT.value,
    "dataset": BibType.DATASET.value,
    "database": BibType.DATASET.value,
    "software": BibType.SOFTWARE.value,
    "thesis": BibType.THESIS.value,
    "phdthesis": BibType.THESIS.value,
    "mastersthesis": BibType.THESIS.value,
    "masterthesis": BibType.THESIS.value,
    "dissertation": BibType.THESIS.value,
}


def migrate_bib_type(old: str | None) -> str:
    """Map legacy BibTeX and Crossref type strings to ``BibType`` values.

    Returns a ``BibType`` value string.  Unknown inputs map to ``"other"``.
    """
    if not old or not isinstance(old, str):
        return BibType.OTHER.value
    return _BIB_TYPE_ALIASES.get(old.lower().strip(), BibType.OTHER.value)


# ``PaperAuthor.role`` entry that marks a group or organization author (a
# consortium, a working group) rather than a person: the LLM author parse emits
# it, and the JATS reader gives a ``<collab>`` byline the same mark. The export
# writes such an author's name to ``author[].literal`` and drops the marker
# from ``role``.
ORGANIZATION_ROLE = "organization"


class PaperAuthor(_Base):
    """CrossRef-like author representation"""

    author_id: int  # positional index, 1-based
    given: str
    family: str
    affiliation: str
    email: str | None = None
    corresponding: bool = False
    orcid: str | None = None
    role: list[str] = Field(default_factory=list)


# ASCII digits only: ``\d`` also matches other scripts' digits, which no ORCID has.
_ORCID_BARE_RE = re.compile(r"^[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]$")
# Same identifier with no separators — some JATS deposits carry this form.
_ORCID_DIGITS_RE = re.compile(r"^[0-9]{15}[0-9X]$")
# Optional scheme/host wrapper: "https://orcid.org/", "orcid.org/", "www."
_ORCID_HOST_RE = re.compile(r"^(?:https?://)?(?:www\.)?orcid\.org/", re.IGNORECASE)


def canonicalize_orcid(orcid: str | None) -> str | None:
    """Ensure ORCID is in canonical URI form: ``https://orcid.org/XXXX-XXXX-XXXX-XXXX``.

    Returns ``None`` for anything that is not a recognizable ORCID. Byline
    superscripts ("1", "a", "*") reach this slot from the LLM, and asserting
    them as an ``orcid`` is worse than asserting nothing.
    """
    if not orcid or not isinstance(orcid, str):
        return None
    bare = _ORCID_HOST_RE.sub("", orcid.strip()).rstrip("/").strip().upper()
    if _ORCID_DIGITS_RE.match(bare):
        bare = "-".join(bare[i : i + 4] for i in range(0, 16, 4))
    if _ORCID_BARE_RE.match(bare):
        return f"https://orcid.org/{bare}"
    return None


class MatchSource(StrEnum):
    """Supported external match services."""

    CROSSREF = "crossref"
    OPENALEX = "openalex"
    DATACITE = "datacite"
    DOI_ORG = "doi.org"
    OPENLIBRARY = "openlibrary"
    ROR = "ror"
    MANUAL = "manual"
    OTHER = "other"


class ExternalMatch(_Base):
    """External service match for a bibliography entry."""

    id: str | None = None  # DOI, OpenAlex ID, etc.
    score: float | None = None
    title: str | None = None
    authors: list[BibAuthor] | None = None
    year: int | None = None
    container: str | None = None  # journal or book title
    volume: str | None = None
    issue: str | None = None
    first_page: str | None = None
    last_page: str | None = None
    publisher: str | None = None
    editors: list[BibAuthor] | None = None
    doi: str | None = None
    bib_type: str | None = None  # BibType value
    url: str | None = None
    date: str | None = None  # ISO date string if available
    edition: str | None = None
    version: str | None = None
    license_url: str | None = None
    funders: list[MatchFunder] | None = None


class PaperReference(_Base):
    """Parsed reference entry."""

    bib_id: int  # position in the paper (ordered)
    title: str  # full title
    first_page: str | None
    volume: str | None
    authors: str | None  # author string as written in the reference
    year: int | None  # numeric year (None for unknown OR in-press)
    container: str | None  # journal or book title
    year_suffix: str | None = None  # disambiguator, e.g. "a" in "2005a"
    doi: str | None = None
    bib_type: str | None = None  # BibType value (journal_article, book, …)
    text_id: int | None = None  # sentence text_id in the references section
    last_page: str | None = None
    issue: str | None = None  # issue number
    editors: str | None = None  # editor string as written in the reference
    publisher: str | None = None
    url: str | None = None
    date: str | None = None  # ISO date string if available
    edition: str | None = None
    version: str | None = None
    is_in_press: bool = False  # True for "in press"/"forthcoming"/"advance online" refs
    # Identifiers and trailing matter the NER parser has always tagged and that
    # had nowhere to go: the 39-tag BIO scheme carries ARXIV, PMID, SERIES,
    # ACCESS_DATE and NOTE, and every value was dropped at decode because
    # ``_FIELD_TO_PAPER_REF`` had no target for them. Verbatim as printed, like
    # the fields above -- ``date`` is the only normalised one.
    arxiv: str | None = None  # arXiv id as printed ("1803.04219", "arXiv:1803.04219")
    pmid: str | None = None  # PubMed id as printed
    series: str | None = None  # book or report series title
    access_date: str | None = None  # "Accessed 12 March 2020", verbatim
    note: str | None = None  # trailing free text no other field claims
    match: dict[MatchSource, ExternalMatch] = Field(default_factory=dict)


class FundingEntry(_Base):
    """A funding body and the award/grant numbers printed for it, verbatim."""

    funder: str
    award_ids: list[str] = Field(default_factory=list)


class Affiliation(_Base):
    """One printed affiliation string with its structured parse."""

    text: str  # verbatim component, from the author byline
    institution: str | None = None
    department: str | None = None
    city: str | None = None
    country: str | None = None
    author_ids: list[int] = Field(default_factory=list)  # PaperAuthor.author_id refs


class PaperMetadata(_Base):
    """Metadata about the paper - actual bibr content of importance"""

    doi: str
    title: str
    abstract: str = ""
    # An explicit model refusal must not be undone by a layout-only abstract
    # guess. Missing fields / no-LLM runs still permit the ordinary fallback.
    _abstract_explicitly_absent: bool = PrivateAttr(default=False)
    paper_type: str = ""
    paper_type_confidence: float | None = None
    oecd_l1: str = ""
    oecd_l2: str = ""
    oecd_confidence: float | None = None
    keywords: list[str] = Field(default_factory=list)
    authors: list[PaperAuthor] = Field(default_factory=list)
    references: list[PaperReference] = Field(default_factory=list)
    # Distinct from Crossref ``enrichment_complete``: False means reference
    # extraction was intentionally skipped or completed (including no refs);
    # True means the reference task failed after durable core metadata existed.
    references_incomplete: bool = False
    # Bounded, non-schema diagnostic carried until Paper assembly turns it into
    # a typed validation issue. It is never serialized as metadata directly.
    _references_incomplete_diagnostic: str = PrivateAttr(default="")
    # Each decided field's proposals and receipt
    # (``bibr.extract.field_decisions``); feeds ``extraction.fields``.
    _field_decisions: FieldDecisions = PrivateAttr(default_factory=FieldDecisions)
    # The paper's OWN bibliographic self-identity, verbatim from the front
    # matter (journal-issue line, footers, copyright/license lines). Null when
    # not printed — never inferred or backfilled from enrichment.
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    first_page: str | None = None
    last_page: str | None = None
    issn: str | None = None
    publisher: str | None = None
    published: str | None = None
    license: str | None = None
    # Identifiers and language the input itself declares (JATS article-id and
    # xml:lang, HTML citation meta tags / <html lang>); null for PDFs, where
    # the exporter derives what it safely can (e.g. arXiv from the DOI).
    language: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    arxiv: str | None = None
    # Research-integrity statements, copied verbatim from the classified
    # section body (no LLM); null when the paper prints no such section.
    funding_statement: str | None = None
    coi_statement: str | None = None
    ethics_statement: str | None = None
    data_availability: str | None = None
    # Structured funding parsed from the funding statement by the LLM
    # research-integrity call; empty when no funding section exists or the run
    # is no-LLM.
    funding: list[FundingEntry] = Field(default_factory=list)
    # Structured affiliations parsed from the author byline by the LLM
    # research-integrity call; the verbatim ``text`` is ours (from the author
    # affiliation strings), the parsed sub-fields are the LLM's best effort.
    affiliations: list[Affiliation] = Field(default_factory=list)
    # Enrichment of the paper's OWN identity (self-DOI lookup); mirrors
    # PaperReference.match. Printed fields above are never overwritten by it.
    match: dict[MatchSource, ExternalMatch] = Field(default_factory=dict)
    # ROR organizations matched to printed affiliation strings and funder
    # names, keyed by the exact string (``Affiliation.text`` / ``FundingEntry.funder``).
    affiliation_match: dict[str, OrganizationMatch] = Field(default_factory=dict)
    funder_match: dict[str, OrganizationMatch] = Field(default_factory=dict)
    # Set by CrossrefEnricher: True if enrichment ran to completion, False if it
    # timed out / failed (so bib_match is a partial prefix), None if it never ran.
    enrichment_complete: bool | None = None

    # A copy gets its own receipts, so a decision on it never rewrites the
    # original's; a field an ``update`` rewrites loses its now stale receipt.
    def __copy__(self) -> Self:
        copied = super().__copy__()
        copied._field_decisions = self._field_decisions.copy()
        return copied

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        if update:
            copied._field_decisions.forget_attributes(update)
        return copied


class ErrorCode(StrEnum):
    """Structured error codes for pipeline failures."""

    # Input validation
    UNSUPPORTED_FORMAT = "unsupported_format"
    CORRUPTED_FILE = "corrupted_file"
    ENCRYPTED_FILE = "encrypted_file"

    # Processing
    DOC_CONVERT_FAILED = "doc_convert_failed"
    LAYOUT_FAILED = "layout_failed"
    OCR_FAILED = "ocr_failed"
    PARSE_FAILED = "parse_failed"
    EXTRACTION_FAILED = "extraction_failed"
    LLM_INVALID_OUTPUT = "llm_invalid_output"
    LLM_TRUNCATED = "llm_truncated"
    EXPORT_FAILED = "export_failed"

    # Upstream
    LLM_TIMEOUT = "llm_timeout"
    LLM_FAILED = "llm_failed"
    OCR_TIMEOUT = "ocr_timeout"
    CROSSREF_TIMEOUT = "crossref_timeout"

    # Generic
    UNKNOWN = "unknown"


class ProcessingStatus(_Base):
    """Structured processing status with error reporting and timing."""

    parsed: bool = False
    error_code: ErrorCode | None = None
    error_message: str | None = None
    failed_stage: str | None = None
    stage_times: dict[str, float] = Field(default_factory=dict)
