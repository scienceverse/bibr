"""Non-fatal processing warnings and the registry of their codes.

Every warning a run records is a :class:`ProcessingWarning`: a stable,
machine-readable ``code`` from :class:`WarningCode` and a human-readable
``message`` carrying the details (page, counts, exception type). The export
writes them to ``extraction.warnings`` as ``{code, message}`` objects.

Codes are UPPER_SNAKE, like the validation issue codes. The export schema pins
that pattern, not this list: another producer of the format may add codes of
its own, so a reader must accept a code it does not know.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class WarningCode(StrEnum):
    """Every code bibr writes to ``extraction.warnings[].code``.

    Each member's value is its name; :data:`DESCRIPTIONS` says what it means.
    """

    # OCR and layout parsing
    OCR_REGION_FAILED = "OCR_REGION_FAILED"
    OCR_PAGE_FAILED = "OCR_PAGE_FAILED"
    OCR_OUTPUT_TRUNCATED = "OCR_OUTPUT_TRUNCATED"
    OCR_TABLE_INCOMPLETE = "OCR_TABLE_INCOMPLETE"
    OCR_TABLE_MALFORMED = "OCR_TABLE_MALFORMED"
    OCR_TABLE_DROPPED = "OCR_TABLE_DROPPED"
    OCR_CONTROL_CHARS = "OCR_CONTROL_CHARS"
    OCR_NATIVE_TEXT_PUA_FALLBACK = "OCR_NATIVE_TEXT_PUA_FALLBACK"
    LOW_TEXT_QUALITY = "LOW_TEXT_QUALITY"
    # Classifiers
    SECTION_CLASSIFIER_DEGRADED = "SECTION_CLASSIFIER_DEGRADED"
    PAPER_CLASSIFIER_DEGRADED = "PAPER_CLASSIFIER_DEGRADED"
    # Metadata, statements and equations
    AUTHORS_EMPTY = "AUTHORS_EMPTY"
    AUTHORS_FABRICATED = "AUTHORS_FABRICATED"
    AUTHORS_ANOMALY = "AUTHORS_ANOMALY"
    STATEMENT_LEXICAL_FALLBACK = "STATEMENT_LEXICAL_FALLBACK"
    EQUATION_LLM_FALLBACK_TIMEOUT = "EQUATION_LLM_FALLBACK_TIMEOUT"
    EQUATION_LLM_FALLBACK_FAILED = "EQUATION_LLM_FALLBACK_FAILED"
    # Reference segmentation
    REF_SEG_GEOM_CASCADE = "REF_SEG_GEOM_CASCADE"
    REF_SEG_REGION_CASCADE = "REF_SEG_REGION_CASCADE"
    REF_SEG_REGION_ERROR = "REF_SEG_REGION_ERROR"
    REF_SEG_REGION_RECOVERY = "REF_SEG_REGION_RECOVERY"
    REF_SEG_CRF_FALLBACK = "REF_SEG_CRF_FALLBACK"
    REF_SEG_CRF_ERROR = "REF_SEG_CRF_ERROR"
    REF_SEG_MARKER_SPLIT_RECOVERY = "REF_SEG_MARKER_SPLIT_RECOVERY"
    REF_SEG_FAILED = "REF_SEG_FAILED"
    REF_SEG_MERGE_SPLIT = "REF_SEG_MERGE_SPLIT"
    # Reference parsing
    REF_PARSE_NER_FALLBACK = "REF_PARSE_NER_FALLBACK"
    REF_PARSE_LOST = "REF_PARSE_LOST"
    REF_PARSE_SALVAGE_RECOVERY = "REF_PARSE_SALVAGE_RECOVERY"
    REF_PARSE_SPLIT_RECOVERY = "REF_PARSE_SPLIT_RECOVERY"
    REF_EXTRACTION_ERROR = "REF_EXTRACTION_ERROR"
    REF_UNDER_EXTRACTION_SUSPECTED = "REF_UNDER_EXTRACTION_SUSPECTED"
    # Enrichment
    CROSSREF_ENRICHMENT_TIMEOUT = "CROSSREF_ENRICHMENT_TIMEOUT"
    CROSSREF_ENRICHMENT_FAILED = "CROSSREF_ENRICHMENT_FAILED"
    ENRICHMENT_LOOKUP_FAILED = "ENRICHMENT_LOOKUP_FAILED"
    RESOLVER_FALLBACK_TIMEOUT = "RESOLVER_FALLBACK_TIMEOUT"
    RESOLVER_FALLBACK_FAILED = "RESOLVER_FALLBACK_FAILED"
    ROR_MATCHING_TIMEOUT = "ROR_MATCHING_TIMEOUT"
    ENRICHER_FAILED = "ENRICHER_FAILED"
    ENRICHMENT_INCOMPLETE = "ENRICHMENT_INCOMPLETE"
    CONSOLIDATE_WITHOUT_ENRICHMENT = "CONSOLIDATE_WITHOUT_ENRICHMENT"
    # Export content
    URL_MALFORMED_DROPPED = "URL_MALFORMED_DROPPED"
    EPUB_SPINE_MEMBER_SKIPPED = "EPUB_SPINE_MEMBER_SKIPPED"


DESCRIPTIONS: dict[WarningCode, str] = {
    WarningCode.OCR_REGION_FAILED: "OCR of a layout region failed; its text is missing.",
    WarningCode.OCR_PAGE_FAILED: "OCR of a whole page failed; its text is missing.",
    WarningCode.OCR_OUTPUT_TRUNCATED: "OCR output for a region stopped at the generation "
    "limit; its text may be cut off.",
    WarningCode.OCR_TABLE_INCOMPLETE: "OCR output for a table stopped early or is not a closed "
    "grid; the table may be incomplete.",
    WarningCode.OCR_TABLE_MALFORMED: "The spans of an OCR table could not be resolved; the "
    "table was kept without merged cells.",
    WarningCode.OCR_TABLE_DROPPED: "Table regions could not be parsed and were dropped.",
    WarningCode.OCR_CONTROL_CHARS: "OCR output contained control characters; section headers "
    "and references may be unreliable.",
    WarningCode.OCR_NATIVE_TEXT_PUA_FALLBACK: "Embedded PDF text used private-use characters; "
    "those regions were read with OCR instead.",
    WarningCode.LOW_TEXT_QUALITY: "The text-quality score is below the warning threshold.",
    WarningCode.SECTION_CLASSIFIER_DEGRADED: "The trained section classifier did not answer; "
    "the LLM classified the section headers.",
    WarningCode.PAPER_CLASSIFIER_DEGRADED: "The trained paper classifier did not answer; the "
    "LLM classified the paper.",
    WarningCode.AUTHORS_EMPTY: "No authors were extracted from a paper that is not a "
    "correction notice.",
    WarningCode.AUTHORS_FABRICATED: "Extracted authors were discarded because none appears in "
    "the text the extraction was given.",
    WarningCode.AUTHORS_ANOMALY: "The extracted author list looked degenerate and was trimmed "
    "or emptied.",
    WarningCode.STATEMENT_LEXICAL_FALLBACK: "A research-integrity statement was filled by "
    "lexical anchor matching.",
    WarningCode.EQUATION_LLM_FALLBACK_TIMEOUT: "The equation LLM fallback timed out; only "
    "regex-extracted equations are kept.",
    WarningCode.EQUATION_LLM_FALLBACK_FAILED: "The equation LLM fallback failed; only "
    "regex-extracted equations are kept.",
    WarningCode.REF_SEG_GEOM_CASCADE: "Geometry reference segmentation declined; the next tier "
    "segmented the references.",
    WarningCode.REF_SEG_REGION_CASCADE: "Region-anchor reference segmentation declined or found "
    "too few references; the next tier segmented them.",
    WarningCode.REF_SEG_REGION_ERROR: "Region-anchor reference segmentation raised an error.",
    WarningCode.REF_SEG_REGION_RECOVERY: "References were segmented from layout regions after "
    "another tier declined.",
    WarningCode.REF_SEG_CRF_FALLBACK: "LLM reference segmentation failed or is disabled; "
    "segmentation fell back to layout regions or the CRF.",
    WarningCode.REF_SEG_CRF_ERROR: "The CRF reference segmenter raised an error.",
    WarningCode.REF_SEG_MARKER_SPLIT_RECOVERY: "Every segmenter failed; the references were "
    "split on their printed markers.",
    WarningCode.REF_SEG_FAILED: "A non-empty reference section produced no references.",
    WarningCode.REF_SEG_MERGE_SPLIT: "Reference strings holding several references were split.",
    WarningCode.REF_PARSE_NER_FALLBACK: "LLM reference parsing failed or skipped entries; the "
    "local NER parser parsed them.",
    WarningCode.REF_PARSE_LOST: "References were dropped: LLM parsing failed on them or returned "
    "them empty, and no fallback recovered them.",
    WarningCode.REF_PARSE_SALVAGE_RECOVERY: "Complete references were salvaged from a truncated "
    "LLM completion.",
    WarningCode.REF_PARSE_SPLIT_RECOVERY: "References were re-parsed in smaller LLM batches "
    "after a batch failed.",
    WarningCode.REF_EXTRACTION_ERROR: "Reference extraction raised an unexpected error; the "
    "references are missing.",
    WarningCode.REF_UNDER_EXTRACTION_SUSPECTED: "Far fewer references were parsed than the body "
    "cites; reference entries may have been dropped.",
    WarningCode.CROSSREF_ENRICHMENT_TIMEOUT: "Crossref enrichment stopped at its per-paper time "
    "budget.",
    WarningCode.CROSSREF_ENRICHMENT_FAILED: "Crossref enrichment failed.",
    WarningCode.ENRICHMENT_LOOKUP_FAILED: "A registry lookup for a reference or the paper's DOI "
    "failed.",
    WarningCode.RESOLVER_FALLBACK_TIMEOUT: "The resolver fallback stopped at its time budget.",
    WarningCode.RESOLVER_FALLBACK_FAILED: "The resolver fallback failed.",
    WarningCode.ROR_MATCHING_TIMEOUT: "ROR matching stopped at its time budget; some "
    "affiliation and funder strings are unmatched.",
    WarningCode.ENRICHER_FAILED: "An enricher raised an unexpected error.",
    WarningCode.ENRICHMENT_INCOMPLETE: "Enrichment ended incomplete; the message is the reason "
    "recorded with the checkpoint.",
    WarningCode.CONSOLIDATE_WITHOUT_ENRICHMENT: "Consolidation was requested but enrichment is "
    "off, so there were no matches to merge.",
    WarningCode.URL_MALFORMED_DROPPED: "A link was dropped from the export as malformed; "
    "the URL is lost from the payload.",
    WarningCode.EPUB_SPINE_MEMBER_SKIPPED: "An ePub spine member was missing from the "
    "archive; its text is absent from the export.",
}


@dataclass(frozen=True)
class ProcessingWarning:
    """One non-fatal processing warning: a stable code and a readable message.

    Frozen and hashable, so the pipeline de-duplicates warnings with
    ``dict.fromkeys`` wherever it merges them.
    """

    code: str
    message: str

    def __post_init__(self) -> None:
        # Store a WarningCode member as the plain string it equals.
        object.__setattr__(self, "code", str(self.code))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}

    @classmethod
    def from_dict(cls, value: object) -> ProcessingWarning:
        """Decode an exported ``{code, message}`` object; a warning passes through.

        Raises ``ValueError`` for anything else, such as an older export's prose string.
        """
        if isinstance(value, ProcessingWarning):
            return value
        if (
            not isinstance(value, Mapping)
            or set(value) != {"code", "message"}
            or not isinstance(value["code"], str)
            or not isinstance(value["message"], str)
        ):
            raise ValueError(f"not a {{code, message}} processing warning: {value!r:.80}")
        return cls(value["code"], value["message"])
