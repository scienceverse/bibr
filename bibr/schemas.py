"""Contains Pydantic models that help us enforce type safety for requests and responses."""

import logging
from typing import ClassVar, get_args

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from bibr.clients.nuextract_schema import NuExtractSchemaPolicy
from bibr.models import BibType, PaperAuthor, PaperReference, migrate_bib_type
from bibr.structure.paper_classifier import (
    PAPER_TYPE_LABELS,
    OECDDomainLiteral,
    OECDSubdomainLiteral,
    PaperTypeLiteral,
    canonicalize_oecd_l2_any,
    validate_oecd_l1,
)

logger = logging.getLogger(__name__)

_PAPER_TYPE_VALUES = frozenset(PAPER_TYPE_LABELS)
_NUEXTRACT_CLASSIFICATION_CHOICES = {
    "oecd_domain": get_args(OECDDomainLiteral),
    "oecd_subdomain": get_args(OECDSubdomainLiteral),
    "paper_type": tuple(PAPER_TYPE_LABELS),
}


def _canonicalize_paper_type(v: object) -> str | None:
    """Lenient BeforeValidator for ``paper_type``.

    Keeps the cloud (Gemini) path from hard-failing on near-miss/garbage
    labels: strip + lowercase, accept exact membership, else None. None
    passes through untouched; never raises.
    """
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in _PAPER_TYPE_VALUES:
        return s
    if not s:
        return None  # blank string = null-emission quirk, not garbage worth warning about
    logger.warning("Coercing unrecognized paper_type %r to None", v)
    return None


def _canonicalize_oecd_domain(v: object) -> str | None:
    """Lenient BeforeValidator for ``oecd_domain`` (canonicalize or None)."""
    if v is None:
        return None
    result = validate_oecd_l1(v if isinstance(v, str) else str(v))
    if result:
        return result
    if not str(v).strip():
        return None  # blank string = null-emission quirk, not garbage worth warning about
    logger.warning("Coercing unrecognized oecd_domain %r to None", v)
    return None


def _canonicalize_oecd_subdomain(v: object) -> str | None:
    """Lenient BeforeValidator for ``oecd_subdomain`` (canonicalize or None).

    Matches across ALL L2 labels (no L1 context available here); the
    extractor's cross-L1 rescue reconciles the parent L1 afterwards.
    """
    if v is None:
        return None
    result = canonicalize_oecd_l2_any(v if isinstance(v, str) else str(v))
    if result:
        return result
    if not str(v).strip():
        return None  # blank string = null-emission quirk, not garbage worth warning about
    logger.warning("Coercing unrecognized oecd_subdomain %r to None", v)
    return None


def _coerce_none_to_empty(v: object) -> object:
    """Coerce None → "" so LLM nulls don't crash Pydantic validation.

    Returns ``object`` so Pydantic's later type validators can apply per-field
    coercion (e.g. ``int | None`` fields aren't forced through a string round-trip).
    """
    return v if v is not None else ""


# NuExtract3's template DSL type tokens, occasionally echoed back as field
# values instead of real content (notably under guided decoding when the
# model gets confused about which grammar branch it's filling in).
_PLACEHOLDER_TOKENS = frozenset(
    {"verbatim-string", "string", "integer", "number", "boolean", "date-time", "date"}
)


def _is_placeholder_token(v: object) -> bool:
    """True when ``v`` is a NuExtract3 template-DSL placeholder token.

    Exact whole-value match only, after strip + casefold — never substring —
    so legitimate values like "string theory" or "A number of things" are
    left untouched.
    """
    return isinstance(v, str) and v.strip().casefold() in _PLACEHOLDER_TOKENS


def _scrub_str_placeholder(v: object) -> object:
    """Scrub NuExtract3 placeholder tokens (and None) to "" on the
    non-Optional string fields.

    Used for ``title`` and the author ``given``/``family``/``affiliation``
    parts, which the codebase guarantees are never None post-validation
    (downstream ``strip_affiliation_markers`` and ``PaperAuthor``'s non-Optional
    str fields both assume that). Both None and a template-DSL placeholder token
    collapse to "" so a leaked "verbatim-string" degrades to an empty value
    rather than either surviving as literal noise or crashing str-typed callers.
    """
    v = _coerce_none_to_empty(v)
    return "" if _is_placeholder_token(v) else v


def _stringify_numeric_locator(v: object) -> object:
    """Render a JSON number in a short-numeric field as its string form.

    Page/volume/issue are typed ``str`` because they are printed locators,
    not quantities — "S13", "e12345", "iii" and "Article 13" are all real
    values. But a reference whose page range really is ``100-115`` invites
    the LLM to emit ``100`` unquoted, and rejecting that discards every
    other field of the reference over a formatting detail. ``bool`` is an
    ``int`` subclass and is deliberately left alone: a stray ``true`` is
    not a page number.
    """
    if isinstance(v, bool) or not isinstance(v, int | float):
        return v
    if isinstance(v, float):
        # 100.0 is the JSON spelling of page 100; 100.5 is not a page at all.
        if not v.is_integer():
            return v
        v = int(v)
    return str(v)


def _strip_citation_punctuation(v: str) -> str:
    """Strip trailing citation punctuation (period, comma) from LLM-extracted text."""
    return v.rstrip(".,") if v else v


# Fields carrying the paper's OWN bibliographic self-identity, shared by
# TitleKeywordsLLM (extraction) and CoreMetadataLLM (assembly carrier).
_BIBLIO_STR_FIELDS = (
    "journal",
    "volume",
    "issue",
    "first_page",
    "last_page",
    "issn",
    "publisher",
    "published",
    "license",
)
# Short-numeric fields where ``:`` is never legitimate — a colon signals an
# LLM serialization leak from another field (mirrors PaperReferenceLLM).
_BIBLIO_COLON_GUARD_FIELDS = ("volume", "issue", "first_page", "last_page")


def _strip_schema_name_envelope(data: object, schema_name: str) -> object:
    """Unwrap a single-key ``{"<SchemaName>": {...}}`` envelope, else return
    ``data`` unchanged. Idempotent — safe to call more than once.

    Small local models (vllm-mlx, where ``json_schema`` mode is advisory) wrap
    the payload in the schema name. :class:`LLMResponse` unwraps this via a
    ``mode="before"`` model validator, but a subclass's own ``mode="before"``
    validators can run BEFORE the inherited one — so any subclass validator
    that needs the flat payload must unwrap first itself.
    """
    if isinstance(data, dict) and len(data) == 1:
        value = next(iter(data.values()))
        if next(iter(data)) == schema_name and isinstance(value, dict):
            return value
    return data


def _coerce_biblio_none_strings(data: object) -> object:
    """Sanitise LLM null-emission failures on the self-identity fields.

    Coerces the literal strings ``"None"`` / ``"null"`` and NuExtract3
    template-DSL placeholder tokens (e.g. ``"verbatim-string"``) to ``None``
    on every field, and drops colon-carrying values from the short-numeric
    fields where a colon can only be a serialization-fragment leak.
    """
    if isinstance(data, dict):
        for key in _BIBLIO_STR_FIELDS:
            val = data.get(key)
            if val in ("None", "null") or _is_placeholder_token(val):
                data[key] = None
        for key in _BIBLIO_COLON_GUARD_FIELDS:
            val = data.get(key)
            if isinstance(val, str) and ":" in val:
                data[key] = None
            elif key in data:
                data[key] = _stringify_numeric_locator(val)
    return data


class LLMResponse(BaseModel):
    """Base for top-level LLM structured-output models.

    Small local models often wrap the payload in the schema name
    (``{"RefAnchors": {"anchors": [...]}}``) — servers whose ``json_schema``
    mode is advisory (e.g. vllm-mlx simple engine) return it as-is and
    validation fails. Unwrap that envelope instead of burning a retry.
    """

    @model_validator(mode="before")
    @classmethod
    def _unwrap_schema_name_envelope(cls, data: object) -> object:
        return _strip_schema_name_envelope(data, cls.__name__)


class AbstractResponse(LLMResponse):
    """Retain an explicit JSON null separately from sanitized blank output."""

    _abstract_explicitly_absent: bool = PrivateAttr(default=False)

    @model_validator(mode="wrap")
    @classmethod
    def _record_abstract_absence(cls, data, handler):
        raw = _strip_schema_name_envelope(data, cls.__name__)
        explicit_null = (
            "abstract" in raw and raw["abstract"] is None
            if isinstance(raw, dict)
            else getattr(raw, "_abstract_explicitly_absent", False)
        )
        result = handler(data)
        result._abstract_explicitly_absent = explicit_null
        return result


class AuthorLLM(PaperAuthor):
    """A printed author with their affiliation, contact details and ORCID."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy(
        semantics={
            "given": "string",
            "family": "string",
            "affiliation": "string",
            "email": "email-address",
            "orcid": "verbatim-string",
        },
    )

    # Exclude downstream fields from every generation schema, not just NuExtract.
    # A permissive role array can trap local guided decoders in whitespace loops.
    # Keep the fields on the model for downstream assembly and cached responses.
    # LLM doesn't emit author_id; assigned post-extraction by enumeration.
    author_id: SkipJsonSchema[int | None] = None
    # role is filled by downstream affiliation/email harvesting, not the LLM.
    role: SkipJsonSchema[list[str]] = Field(default_factory=list)

    # Re-declare nullable variants of the parent's required str fields so the
    # JSON schema given to Instructor signals that None is acceptable. The
    # ``_coerce_strs`` validator (mode=before) turns both None and NuExtract3
    # template-DSL placeholder tokens (e.g. "verbatim-string") into "" before
    # type validation, keeping downstream code that assumes non-None str safe.
    given: str | None = Field(
        description="First name and any middle names or initials, e.g. 'Saskia M' or 'Jean-Pierre'"
    )
    family: str | None = Field(
        description=(
            "Family name (surname) only, including particles like 'van der', "
            "e.g. 'Kelders' or 'van der Berg'. For organisation or consortium "
            "authors put the full name here and leave 'given' empty"
        )
    )
    affiliation: str | None = Field(
        default="",
        description="Full institution name(s). Resolve numbered/superscript references to actual names; join multiple with '; '",
    )
    email: str | None = Field(
        default=None,
        description=(
            "This author's email address if printed near their name "
            "(byline or correspondence section)"
        ),
    )
    corresponding: bool = Field(
        default=False,
        description=(
            "True only when an explicit anchor ('Corresponding author' / "
            "'Address correspondence to' line, envelope icon, or a footnote "
            "the paper labels as corresponding) singles out this author; a "
            "bylined email alone is not sufficient"
        ),
    )
    orcid: str | None = Field(
        default=None, description="The ORCID identifier of the author (URL or bare ID)"
    )

    # Also scrubs NuExtract3 placeholder tokens (e.g. "verbatim-string")
    # echoed into these name/affiliation fields to "" (these fields are
    # non-Optional on PaperAuthor, so None is not an option).
    _coerce_strs = field_validator("given", "family", "affiliation", mode="before")(
        _scrub_str_placeholder
    )

    @model_validator(mode="before")
    @classmethod
    def _null_strings_to_none(cls, data):
        """Coerce literal ``"None"`` / ``"null"`` strings to ``None`` for
        nullable fields. Mirrors :class:`PaperReferenceLLM` — LLMs occasionally
        emit Python's ``str(None)`` instead of JSON null, which would otherwise
        flow into corresponding-author email harvesting downstream."""
        if isinstance(data, dict):
            for key in ("email", "orcid"):
                if data.get(key) in ("None", "null"):
                    data[key] = None
        return data


class TitleKeywordsLLM(AbstractResponse):
    """Title, abstract, and keywords extracted by LLM.

    Bundling abstract with title/keywords keeps front-matter extraction in a
    single LLM call (same input context, no extra latency/cost).
    """

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy(
        semantics={
            "title": "verbatim-string",
            "abstract": "verbatim-string",
            "keywords[]": "verbatim-string",
            "journal": "verbatim-string",
            "volume": "string",
            "issue": "string",
            "first_page": "string",
            "last_page": "string",
            "issn": "verbatim-string",
            "publisher": "verbatim-string",
            "published": "date",
            "license": "string",
        },
    )

    # Layout detection is the authoritative title source downstream, so a
    # title-keywords completion that omits this duplicate field is still useful.
    # The explicit default also keeps it out of JSON Schema's ``required`` list;
    # ``str | None`` alone remains required in Pydantic v2.
    title: str | None = Field(default=None, description="The title of the paper")
    abstract: str | None = Field(
        default=None,
        description=(
            "The paper's abstract, copied verbatim from the abstract section. "
            "Exclude running headers, journal-issue lines, copyright notices, "
            "DOI URLs, 'Reprints and permissions' / 'Article reuse guidelines' "
            "banners, 'www.<journal>.org' or publisher-logo fragments, "
            "affiliation blocks, 'Statement of Relevance' boxes, Author Note "
            "blocks, funding / disclosure statements, and correspondence "
            "preambles — only the abstract prose. Return null if no abstract is "
            "present (e.g. commentaries)."
        ),
    )
    keywords: list[str] = Field(default_factory=list, description="The keywords of the paper")
    journal: str | None = Field(
        default=None,
        description=(
            "The journal / venue name the paper was published in, verbatim as "
            "printed in the front matter — the journal-issue line "
            "('Psychological Science 2020, Vol. 31(1)') or a running header that "
            "repeats the journal name is a valid source. Null when no journal "
            "name is printed (e.g. a preprint or working paper)"
        ),
    )
    volume: str | None = Field(
        default=None,
        description=(
            "The volume number printed for THIS paper, e.g. 'Vol. 31(1)' → '31'; "
            "null when none is printed — never guess"
        ),
    )
    issue: str | None = Field(
        default=None,
        description=(
            "The issue number printed for THIS paper, e.g. 'Vol. 31(1)' → '1'; "
            "null when none is printed"
        ),
    )
    first_page: str | None = Field(
        default=None,
        description=(
            "The first page of THIS paper when a page range is printed "
            "(e.g. '65-74' → '65', 'pp. 65-74' → '65'); null if no range is printed"
        ),
    )
    last_page: str | None = Field(
        default=None,
        description=(
            "The last page of THIS paper when a page range is printed "
            "(e.g. '65-74' → '74'); null if no range is printed"
        ),
    )
    issn: str | None = Field(
        default=None,
        description=(
            "The ISSN of the journal, verbatim as printed (e.g. '0956-7976'); "
            "null when none is printed"
        ),
    )
    publisher: str | None = Field(
        default=None,
        description=(
            "The publishing house or organization printed for THIS paper "
            "(e.g. 'SAGE Publications', 'Springer'); null when none is printed"
        ),
    )
    published: str | None = Field(
        default=None,
        description=(
            "The publication date of THIS paper as printed — ISO-normalized "
            "(YYYY-MM-DD) when a full date is printed, otherwise the bare year "
            "(YYYY). Use the publication / issue date, NOT 'Received', "
            "'Accepted', or 'Revised' dates. Null when no publication date is printed"
        ),
    )
    license: str | None = Field(
        default=None,
        description=(
            "The reuse license when an explicit license / Creative Commons line "
            "is printed, normalized to its short form (e.g. 'This article is "
            "distributed under the terms of the Creative Commons Attribution 4.0 "
            "License' → 'CC BY 4.0'). Null when no explicit license line is printed"
        ),
    )

    _coerce_title = field_validator("title", mode="before")(_scrub_str_placeholder)

    @field_validator("abstract", mode="before")
    @classmethod
    def abstract_blank_to_none(cls, v: str | None) -> str | None:
        if v is None or _is_placeholder_token(v):
            return None
        s = v.strip()
        return s or None

    @field_validator("keywords", mode="before")
    @classmethod
    def keywords_none_to_empty(cls, v: list[str] | None) -> list[str]:
        """None → [] ; drop placeholder tokens (e.g. "verbatim-string") and
        non-str items defensively, preserving the order of the rest."""
        if v is None:
            return []
        return [kw for kw in v if isinstance(kw, str) and not _is_placeholder_token(kw)]

    @model_validator(mode="before")
    @classmethod
    def _coerce_biblio(cls, data):
        # Unwrap the schema-name envelope first: this ``mode="before"``
        # validator can run before the inherited ``_unwrap_schema_name_envelope``,
        # so on enveloped input the biblio sanitiser would otherwise see only the
        # outer wrapper and never touch the fields (idempotent — the inherited
        # unwrap then no-ops).
        data = _strip_schema_name_envelope(data, cls.__name__)
        return _coerce_biblio_none_strings(data)


class CompactTitleKeywordsLLM(TitleKeywordsLLM):
    # Preserve the evaluated wire schema identity and all validation behavior.
    # This opt-in schema removes duplicated publication instructions only.
    __doc__ = TitleKeywordsLLM.__doc__
    model_config = {"title": "TitleKeywordsLLM"}

    @model_validator(mode="wrap")
    @classmethod
    def _record_abstract_absence(cls, data, handler):
        # The wire title remains TitleKeywordsLLM. Unwrap that alias before
        # inherited sanitation/absence tracking, while still accepting the
        # Python class name used by some native backends.
        return super()._record_abstract_absence(
            _strip_schema_name_envelope(data, "TitleKeywordsLLM"), handler
        )

    # Detailed source/normalization rules live in the task prompt. Keep field
    # descriptions concise so Instructor does not resend those examples twice.
    journal: str | None = Field(default=None, description="This paper's printed journal/venue")
    volume: str | None = Field(default=None, description="Printed volume; null if absent")
    issue: str | None = Field(default=None, description="Printed issue; null if absent")
    first_page: str | None = Field(default=None, description="Start of the printed page range")
    last_page: str | None = Field(default=None, description="End of the printed page range")
    issn: str | None = Field(default=None, description="Printed journal ISSN")
    publisher: str | None = Field(
        default=None, description="Publishing house verbatim, including suffixes; never infer"
    )
    published: str | None = Field(
        default=None,
        description="Printed publication/online date: YYYY-MM-DD when full, otherwise YYYY; "
        "not Received/Accepted/Revised",
    )
    license: str | None = Field(
        default=None,
        description="Explicit license, short form with printed version; Open Access alone is null",
    )


class AuthorsLLM(LLMResponse):
    """Authors extracted by LLM with affiliations, emails, and ORCID."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy()

    authors: list[AuthorLLM] = Field(description="The authors of the paper")


class PaperClassificationLLM(LLMResponse):
    """OECD domain/subdomain and paper type classification extracted by LLM.

    The classification fields are ``Literal`` enums so that under guided
    decoding (vLLM / NuExtract3 JSON_SCHEMA mode) invalid strings — e.g. the
    template-DSL token "verbatim-string" — are unrepresentable. The lenient
    BeforeValidators keep the cloud path from hard-failing on near-miss labels
    (canonicalize where possible, else None); None always passes through.
    """

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy(
        choices=_NUEXTRACT_CLASSIFICATION_CHOICES,
    )

    oecd_domain: OECDDomainLiteral | None = Field(
        default=None,
        description=(
            "The broad OECD research domain of the paper. Must be exactly one of: "
            "'Natural Sciences', 'Engineering and Technology', "
            "'Medical and Health Sciences', 'Agricultural and Veterinary Sciences', "
            "'Social Sciences', 'Humanities and the Arts'. "
            "Return null if uncertain."
        ),
    )
    oecd_subdomain: OECDSubdomainLiteral | None = Field(
        default=None,
        description=(
            "The specific OECD research subdomain within the broad domain. "
            "For Natural Sciences: 'Mathematics', 'Computer and Information Sciences', 'Physical Sciences', 'Chemical Sciences', 'Earth and Related Environmental Sciences', 'Biological Sciences'. "
            "For Engineering and Technology: 'Civil Engineering', 'Electrical Engineering, Electronic Engineering, Information Engineering', 'Mechanical Engineering', 'Chemical Engineering', 'Materials Engineering', 'Medical Engineering', 'Environmental Engineering', 'Environmental Biotechnology', 'Industrial Biotechnology', 'Nano-technology'. "
            "For Medical and Health Sciences: 'Basic Medicine', 'Clinical Medicine', 'Health Sciences', 'Medical Biotechnology'. "
            "For Agricultural and Veterinary Sciences: 'Agriculture, Forestry, and Fisheries', 'Animal and Dairy Science', 'Veterinary Science', 'Agricultural Biotechnology'. "
            "For Social Sciences: 'Psychology and Cognitive Sciences', 'Economics and Business', 'Education', 'Sociology', 'Law', 'Political Science', 'Social and Economic Geography', 'Media and Communications'. "
            "For Humanities and the Arts: 'History and Archaeology', 'Languages and Literature', 'Philosophy, Ethics and Religion', 'Arts (arts, history of arts, performing arts, music)'. "
            "Return null if uncertain."
        ),
    )
    paper_type: PaperTypeLiteral | None = Field(
        default=None,
        description=(
            "The type of paper. Must be exactly one of: "
            "'empirical', 'review', 'meta-analysis', 'case-study', 'commentary', "
            "'corrigendum', 'erratum', 'retraction'. The last three apply to "
            "notices amending or withdrawing a previously published article. "
            "Return null if uncertain."
        ),
    )

    _canon_oecd_domain = field_validator("oecd_domain", mode="before")(_canonicalize_oecd_domain)
    _canon_oecd_subdomain = field_validator("oecd_subdomain", mode="before")(
        _canonicalize_oecd_subdomain
    )
    _canon_paper_type = field_validator("paper_type", mode="before")(_canonicalize_paper_type)


class PaperTypeLabel(LLMResponse):
    """Paper-type-only classification for the offline labeler (title+abstract input).

    A focused subset of PaperClassificationLLM: the enum-constrained paper_type
    plus a self-reported confidence. Used by evaluation/label_paper_type_llm.py
    via the batch layer; its two fields map onto that script's
    {llm_paper_type, llm_confidence} output.
    """

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy(
        choices={"paper_type": tuple(PAPER_TYPE_LABELS)}
    )

    paper_type: PaperTypeLiteral | None = Field(
        default=None,
        description=(
            "The type of paper. Must be exactly one of: 'empirical', 'review', "
            "'meta-analysis', 'case-study', 'commentary', 'corrigendum', 'erratum', "
            "'retraction'. Return null if uncertain."
        ),
    )
    confidence: float | None = Field(
        default=None,
        description="Self-reported confidence in the paper_type label, 0.0-1.0.",
    )

    _canon_paper_type = field_validator("paper_type", mode="before")(_canonicalize_paper_type)


class CoreMetadataLLM(AbstractResponse):
    """Combined core metadata — assembled from the three focused extraction calls."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy(
        semantics={
            "title": "verbatim-string",
            "abstract": "verbatim-string",
            "keywords[]": "verbatim-string",
            "journal": "verbatim-string",
            "volume": "string",
            "issue": "string",
            "first_page": "string",
            "last_page": "string",
            "issn": "verbatim-string",
            "publisher": "verbatim-string",
            "published": "date",
            "license": "string",
        },
        choices=_NUEXTRACT_CLASSIFICATION_CHOICES,
    )

    title: str | None = Field(default=None, description="The title of the paper")
    abstract: str | None = Field(default=None, description="The paper's abstract")
    authors: list[AuthorLLM] = Field(description="The authors of the paper")
    keywords: list[str] = Field(default_factory=list, description="The keywords of the paper")
    # Enum-constrained like PaperClassificationLLM: this model is also used
    # directly as a response_model on the merged-core-metadata path
    # (LLM_MERGED_CORE_METADATA), so it needs the same guided-decoding grammar
    # and lenient coercion. On the default (three-call) path it merely re-runs
    # the validators idempotently over already-canonical values.
    oecd_domain: OECDDomainLiteral | None = Field(default=None)
    oecd_subdomain: OECDSubdomainLiteral | None = Field(default=None)
    paper_type: PaperTypeLiteral | None = Field(default=None)
    journal: str | None = Field(default=None)
    volume: str | None = Field(default=None)
    issue: str | None = Field(default=None)
    first_page: str | None = Field(default=None)
    last_page: str | None = Field(default=None)
    issn: str | None = Field(default=None)
    publisher: str | None = Field(default=None)
    published: str | None = Field(default=None)
    license: str | None = Field(default=None)

    _coerce_title = field_validator("title", mode="before")(_scrub_str_placeholder)
    _canon_oecd_domain = field_validator("oecd_domain", mode="before")(_canonicalize_oecd_domain)
    _canon_oecd_subdomain = field_validator("oecd_subdomain", mode="before")(
        _canonicalize_oecd_subdomain
    )
    _canon_paper_type = field_validator("paper_type", mode="before")(_canonicalize_paper_type)

    @field_validator("abstract", mode="before")
    @classmethod
    def abstract_blank_to_none(cls, v: str | None) -> str | None:
        """Mirror TitleKeywordsLLM: blank / NuExtract3 placeholder → None."""
        if v is None or _is_placeholder_token(v):
            return None
        s = v.strip()
        return s or None

    @field_validator("keywords", mode="before")
    @classmethod
    def keywords_none_to_empty(cls, v: list[str] | None) -> list[str]:
        """None → [] ; drop placeholder tokens (e.g. "verbatim-string") and
        non-str items defensively, preserving the order of the rest."""
        if v is None:
            return []
        return [kw for kw in v if isinstance(kw, str) and not _is_placeholder_token(kw)]

    @model_validator(mode="before")
    @classmethod
    def _coerce_biblio(cls, data):
        # Unwrap the schema-name envelope first: this ``mode="before"``
        # validator can run before the inherited ``_unwrap_schema_name_envelope``,
        # so on enveloped input the biblio sanitiser would otherwise see only the
        # outer wrapper and never touch the fields (idempotent — the inherited
        # unwrap then no-ops).
        data = _strip_schema_name_envelope(data, cls.__name__)
        return _coerce_biblio_none_strings(data)


class PaperReferenceLLM(PaperReference):
    """LLM-extracted reference. Subclasses :class:`PaperReference` so the
    field set is declared once. ``bib_id`` (positional, post-extraction) and
    ``text_id`` (linked later) are relaxed to optional; ``match`` is empty
    until enrichment runs. ``index`` is added as the LLM-emitted 1-based
    ordinal that downstream code maps to ``bib_id``."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy(
        exclude=frozenset({"bib_id", "text_id", "match"}),
        semantics={
            "title": "verbatim-string",
            "first_page": "string",
            "volume": "string",
            "authors": "verbatim-string",
            "container": "verbatim-string",
            "year_suffix": "string",
            "doi": "verbatim-string",
            "last_page": "string",
            "issue": "string",
            "editors": "verbatim-string",
            "publisher": "verbatim-string",
            "url": "url",
            "date": "date",
            "edition": "verbatim-string",
            "version": "verbatim-string",
        },
        # Derived from BibType rather than restated: a hand-copied list drifted
        # out of sync once already, and the model was then told a narrower
        # taxonomy than the one it would be validated against.
        choices={"bib_type": tuple(member.value for member in BibType)},
    )

    # LLM doesn't emit these; assigned post-extraction.
    bib_id: int | None = None
    text_id: int | None = None
    # match is populated by Crossref enrichment, not the LLM.
    match: dict = Field(default_factory=dict)

    # New field on the LLM model only — the 1-based position the LLM used in
    # its output (mapped to ``bib_id`` after re-indexing).
    index: int = Field(description="The index/order of the reference")

    # Whether ``index`` can be trusted to name the input segment this ref was
    # parsed from. False when the client had to re-index positionally after the
    # LLM's own numbering failed validation *and* the returned count did not
    # match the batch — the two together mean position N in the output is not
    # necessarily input N. Segment-anchored backfills (issue recovery, DOI
    # rescue) must be suppressed for those, or ref N inherits ref N−1's printed
    # DOI verbatim. A PrivateAttr so it stays out of the Instructor/NuExtract
    # schema and out of ``model_dump()``; the LLM never sees it.
    _index_trusted: bool = PrivateAttr(default=True)

    @property
    def index_trusted(self) -> bool:
        """Is :attr:`index` a real segment pointer rather than a positional guess?"""
        return self._index_trusted

    def mark_index_untrusted(self) -> None:
        self._index_trusted = False

    # Re-declare to attach Instructor schema descriptions and (where parent is
    # required) relax to optional so the LLM can emit nulls without raising.
    # NOTE (residual #2): the LLM sometimes siphons a printed "(Version X)" out
    # of a software title into the inherited ``version`` field, truncating the
    # title. We steer it via the title rule + prompt below rather than excluding
    # the field, because ``edition`` (shared mechanism) is legitimately used by
    # books ("2nd ed.") and ``version`` by software — dropping either from the
    # schema would silently lose real data.
    title: str | None = Field(
        description=(
            "The title of the reference, verbatim as printed. For software / "
            "dataset references, KEEP any trailing parenthetical version "
            "('(Version 2021.09.2+382)', '(R Package Version 1.0-1)') and any "
            "bracketed medium tag ('[Computer software]', '[Data set]') inside "
            "the title — never move the version into a separate field or drop "
            "the medium tag"
        )
    )
    first_page: str | None = Field(
        description=(
            "The first page number when a page range is printed "
            "(e.g. 'pp. 100-115' → '100'); null if no range is printed — "
            "never carry over from another reference"
        )
    )
    last_page: str | None = Field(
        default=None,
        description=(
            "The last page number when a page range is printed "
            "(e.g. 'pp. 100-115' → '115'); null if no range is printed — "
            "never carry over from another reference"
        ),
    )
    volume: str | None = Field(
        description=(
            "The volume number printed in the reference; null when none is "
            "printed (books usually have no volume — do not guess)"
        )
    )
    issue: str | None = Field(
        default=None,
        description="The issue number printed in the reference; null when none is printed",
    )
    authors: str | None = Field(
        default=None,
        description=(
            "The full author string verbatim as written in the reference, "
            "including any APA-style ellipsis ('. . .') — never collapse a "
            "long author list to only its leading or trailing author"
        ),
    )
    year: int | None = Field(
        description=(
            "The publication year as an integer; null when none can be "
            "determined ('n.d.', 'in press', 'forthcoming', "
            "'advance online publication')"
        )
    )
    year_suffix: str | None = Field(
        default=None,
        description="The year suffix letter, e.g. 'a' from '2005a'",
    )
    container: str | None = Field(
        description=(
            "The journal name for articles, proceedings name for conference "
            "papers, or book title for chapters"
        )
    )
    doi: str | None = Field(
        default=None,
        description=(
            "The DOI as a bare identifier starting with '10.' — strip URL "
            "prefixes (https://doi.org/, dx.doi.org/, doi:). Capture it "
            "whenever a 'doi:'/doi.org token appears anywhere in the entry, "
            "including on the final line after the publisher; suffixes may end "
            "in letters/dots (e.g. '.n83', '_18') — keep the full suffix. Null "
            "only if no DOI is printed"
        ),
    )
    publisher: str | None = Field(
        default=None,
        description="The publishing house or organization (e.g. 'Springer', 'IEEE')",
    )
    editors: str | None = Field(
        default=None,
        description=(
            "The full editor string as written in the reference "
            "(e.g. 'In R. Smith & J. Doe (Eds.)'); null for journal articles"
        ),
    )
    bib_type: str | None = Field(
        default=None,
        # Keep value order in sync with bibr.models.BibType.
        description="The reference type: journal_article, book, book_chapter, dataset, software, preprint, conference_paper, report, thesis, other",
    )
    is_in_press: bool = Field(
        default=False,
        description=(
            "True when the reference is marked 'in press', 'forthcoming', or "
            "'advance online publication'"
        ),
    )

    _coerce_title = field_validator("title", mode="before")(_coerce_none_to_empty)
    _stringify_locators = field_validator(
        "first_page", "last_page", "volume", "issue", mode="before"
    )(_stringify_numeric_locator)

    @field_validator("bib_type", mode="before")
    @classmethod
    def normalize_bib_type(cls, v: str | None) -> str | None:
        """Coerce legacy/free-form bib_type strings to a canonical
        :class:`bibr.models.BibType` value (or None if unset)."""
        if v is None:
            return None
        return migrate_bib_type(v)

    @model_validator(mode="before")
    @classmethod
    def coerce_none_strings(cls, data):
        """Sanitise common LLM null-emission failures.

        Two patterns are coerced to ``None``:

        1. The literal string ``"None"`` (LLM emitting Python's repr instead
           of JSON null).
        2. Strings in short-numeric fields (page/volume/issue) that contain a
           colon. Legitimate values are short alphanumerics like ``"100"``,
           ``"S13"``, or ``"Article 13"``; a colon is a clear sign that
           serialization fragments from another reference's field have leaked
           into the value (e.g. ``"None,index:6,is_in_press:false,last_page:"``).
        """
        if isinstance(data, dict):
            for key in (
                "first_page",
                "last_page",
                "volume",
                "issue",
                "container",
                "doi",
                "publisher",
                "editors",
                "authors",
            ):
                if data.get(key) == "None":
                    data[key] = None

            # Defence-in-depth against LLM serialization leaks. Only applied
            # to short-numeric fields where ``:`` is never a legitimate
            # character; container/publisher/editors/authors can legitimately
            # contain colons (e.g. "Nature: Reviews ...").
            for key in ("first_page", "last_page", "volume", "issue"):
                val = data.get(key)
                if isinstance(val, str) and ":" in val:
                    data[key] = None
        return data

    @model_validator(mode="after")
    def strip_citation_punctuation(self):
        """Strip trailing citation punctuation from all text fields."""
        for field in ("title", "container", "authors", "editors", "publisher"):
            val = getattr(self, field)
            if val:
                setattr(self, field, _strip_citation_punctuation(val))
        return self


class PaperReferenceList(LLMResponse):
    """Wrapper for a list of references"""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy()

    references: list[PaperReferenceLLM] = Field(description="The list of extracted references")


class RefAnchors(LLMResponse):
    """LLM segmentation output: one verbatim opening anchor per reference."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy(
        semantics={"anchors[]": "verbatim-string"}
    )

    anchors: list[str] = Field(
        description=(
            "Opening text of each distinct reference (roughly its first 5-10 "
            "words, copied verbatim), in order — one anchor per reference, "
            "none merged or skipped."
        )
    )


class EquationComponentLLM(BaseModel):
    """A single decomposed equation component extracted by LLM."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy(
        semantics={
            "lhs": "verbatim-string",
            "df": "verbatim-string",
            "comp": "verbatim-string",
            "rhs": "verbatim-string",
        }
    )

    sentence_index: int | None = Field(
        default=None,
        description="The [index] of the sentence this equation was found in",
    )
    lhs: str | None = Field(
        description="Left-hand side statistic name WITHOUT degrees of freedom (e.g., 't', 'p', 'α')"
    )
    df: str | None = Field(
        default="",
        description=(
            "Degrees of freedom shown parenthetically on the LHS (e.g., '28' for t(28), "
            "'2, 47' for F(2, 47)); empty string if there are none"
        ),
    )
    comp: str | None = Field(description="Comparison operator ('=', '<', '>', '≤', '≥', '≈')")
    rhs: str | None = Field(description="Right-hand side (e.g., '3.42', '.003', '[2.0, 4.7]')")

    _coerce_strs = field_validator("lhs", "df", "comp", "rhs", mode="before")(_coerce_none_to_empty)


class EquationExtractionResult(LLMResponse):
    """Batch result from LLM equation extraction."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy()

    equations: list[EquationComponentLLM] = Field(
        description="List of decomposed equation components found in the sentences"
    )


class CitationMatch(BaseModel):
    """A single resolved inline citation → bib entry match."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy(
        semantics={"citation_text": "verbatim-string"}
    )

    text_id: int = Field(description="The sentence text_id containing this citation")
    citation_text: str = Field(description="The citation text as it appears in the sentence")
    bib_id: int | None = Field(
        description="The bib_id of the matched reference, or null if unresolvable"
    )


class CitationResolutionResult(LLMResponse):
    """Batch result from LLM citation resolution."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy()

    matches: list[CitationMatch] = Field(description="List of resolved citation matches")


class FundingEntryLLM(BaseModel):
    """A funding body and its award/grant numbers, as printed in the funding
    statement."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy(
        semantics={"funder": "verbatim-string", "award_ids[]": "verbatim-string"}
    )

    funder: str = Field(
        description=(
            "The funding body/organisation exactly as printed (e.g. 'National "
            "Science Foundation', 'Wellcome Trust'); never abbreviate or expand"
        )
    )
    award_ids: list[str] = Field(
        default_factory=list,
        description=(
            "The grant/award numbers printed for THIS funder, verbatim "
            "(e.g. 'R01-MH12345', '834861'); empty list when none is printed"
        ),
    )

    # Scrubs None AND NuExtract3 placeholder tokens (e.g. "verbatim-string") to
    # "" — funder is non-Optional, so a leaked placeholder degrades to empty and
    # the consuming extractor drops the entry rather than emitting noise.
    _coerce_funder = field_validator("funder", mode="before")(_scrub_str_placeholder)

    @field_validator("award_ids", mode="before")
    @classmethod
    def _award_ids_none_to_empty(cls, v: list[str] | None) -> list[str]:
        """None → [] ; drop NuExtract3 placeholder tokens (e.g. leaked
        "verbatim-string" award ids), preserving the order of the rest."""
        if v is None:
            return []
        return [a for a in v if not _is_placeholder_token(a)]


class AffiliationLLM(BaseModel):
    """One affiliation string's structured parse. Every component is copied
    verbatim from the printed affiliation — never translated, expanded, or
    normalised."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy(
        semantics={
            "institution": "verbatim-string",
            "department": "verbatim-string",
            "city": "verbatim-string",
            "country": "verbatim-string",
        }
    )

    index: int = Field(
        description="The [n] number of this affiliation in the provided affiliation list"
    )
    institution: str | None = Field(
        default=None,
        description=(
            "The institution/organisation name, copied as the exact words printed "
            "in the affiliation string (e.g. 'Univ. of Cambridge' stays 'Univ. of "
            "Cambridge'); null when no institution is printed. Never translate, "
            "expand abbreviations, or normalise."
        ),
    )
    department: str | None = Field(
        default=None,
        description=(
            "The department/faculty/school, copied verbatim from the affiliation "
            "string; null when none is printed. Never translate, expand "
            "abbreviations, or normalise."
        ),
    )
    city: str | None = Field(
        default=None,
        description=(
            "The city/locality, copied verbatim from the affiliation string; null "
            "when none is printed. Never translate, expand abbreviations, or "
            "normalise."
        ),
    )
    country: str | None = Field(
        default=None,
        description=(
            "The country, copied verbatim from the affiliation string; null when "
            "none is printed. Never translate, expand abbreviations, or normalise."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_none_strings(cls, data: object) -> object:
        """Coerce literal ``"None"`` / ``"null"`` strings and NuExtract3
        template-DSL placeholder tokens (e.g. ``"verbatim-string"``) to ``None``
        on every parsed component (all Optional)."""
        if isinstance(data, dict):
            for key in ("institution", "department", "city", "country"):
                val = data.get(key)
                if val in ("None", "null") or _is_placeholder_token(val):
                    data[key] = None
        return data


class AuthorContributionLLM(BaseModel):
    """One author's contribution roles, as printed in the contributions
    statement."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy(
        semantics={"author": "verbatim-string", "roles[]": "verbatim-string"}
    )

    author: str = Field(
        description=(
            "The author name or initials exactly as printed in the contributions "
            "statement (e.g. 'J.W.', 'Jakub Werner', 'Werner')"
        )
    )
    roles: list[str] = Field(
        default_factory=list,
        description=(
            "The contribution phrases printed for this author, verbatim "
            "(e.g. 'Conceptualization', 'wrote the manuscript'); never invent a "
            "role or map it to a taxonomy — copy exactly what is written"
        ),
    )

    _coerce_author = field_validator("author", mode="before")(_coerce_none_to_empty)

    @field_validator("roles", mode="before")
    @classmethod
    def _roles_none_to_empty(cls, v: list[str] | None) -> list[str]:
        return v if v is not None else []


class ResearchIntegrityLLM(LLMResponse):
    """Structured funding + author contributions parsed from the funding and
    author-contributions section text."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy()

    funding: list[FundingEntryLLM] = Field(
        default_factory=list,
        description="One entry per distinct funding body printed in the funding statement",
    )
    contributions: list[AuthorContributionLLM] = Field(
        default_factory=list,
        description="One entry per author named in the contributions statement",
    )
    affiliations: list[AffiliationLLM] = Field(
        default_factory=list,
        description="One entry per numbered affiliation in the provided affiliation list",
    )

    @field_validator("funding", "contributions", "affiliations", mode="before")
    @classmethod
    def _list_none_to_empty(cls, v: list | None) -> list:
        return v if v is not None else []


class FrontMatterSegment(BaseModel):
    """A detected segment boundary in unlabeled front matter."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy(
        choices={"section_type": ("abstract", "intro", "keywords", "metadata")}
    )

    first_text_id: int = Field(description="text_id of the first sentence in this segment")
    section_type: str = Field(
        description=("The detected section type: 'abstract', 'intro', 'keywords', or 'metadata'")
    )


class FrontMatterResult(LLMResponse):
    """LLM result for implicit section boundary detection."""

    nuextract_policy: ClassVar[NuExtractSchemaPolicy] = NuExtractSchemaPolicy()

    segments: list[FrontMatterSegment] = Field(description="Detected segments in document order")
