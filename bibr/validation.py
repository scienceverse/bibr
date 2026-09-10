"""Shared typed validation issues emitted across pipeline stages."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.models import PaperMetadata

_MAX_DIAGNOSTIC_LENGTH = 256
ABSTRACT_SUSPICION_REASON_ORDER = (
    "ungrounded",
    "cross_boundary",
    "length_gt_2500",
    "non_reference_share_gt_20pct",
)


class IssueSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    """A semantic or replay-validation finding attached to an extraction."""

    code: str
    severity: IssueSeverity | str
    message: str
    origin_stage: str = "export"
    evidence_ids: tuple[str, ...] = ()
    count: int = 1
    blocking: bool = False


def _normalize_validation_text(value: str) -> str:
    return " ".join(value.casefold().split())


def abstract_suspicion_reasons(
    abstract_text: str,
    *,
    source_text: str | None,
    outside_texts: Iterable[str] = (),
    non_reference_prose: str = "",
) -> tuple[str, ...]:
    """Return canonical, strictly-thresholded abstract suspicion reasons."""

    abstract = abstract_text.strip()
    normalized_abstract = _normalize_validation_text(abstract)
    reasons: set[str] = set()
    if source_text is not None:
        normalized_source = _normalize_validation_text(source_text)
        if not normalized_source or normalized_abstract not in normalized_source:
            reasons.add("ungrounded")
            if any(
                len(normalized_outside) >= 16 and normalized_outside in normalized_abstract
                for outside_text in outside_texts
                if (normalized_outside := _normalize_validation_text(outside_text))
            ):
                reasons.add("cross_boundary")
    if len(abstract) > 2500:
        reasons.add("length_gt_2500")
    prose = non_reference_prose.strip()
    if prose and len(abstract) / len(prose) > 0.2:
        reasons.add("non_reference_share_gt_20pct")
    return tuple(reason for reason in ABSTRACT_SUSPICION_REASON_ORDER if reason in reasons)


def _bounded_exception_diagnostic(exc: BaseException) -> str:
    """Keep the exception type/message useful without exporting unbounded text."""

    detail = " ".join(str(exc).split())
    diagnostic = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return diagnostic[:_MAX_DIAGNOSTIC_LENGTH]


def mark_references_incomplete(metadata: PaperMetadata, exc: BaseException) -> None:
    """Mark a core extraction whose reference task failed in full."""

    metadata.references = []
    metadata.references_incomplete = True
    metadata._references_incomplete_diagnostic = _bounded_exception_diagnostic(exc)


def clear_reference_state(metadata: PaperMetadata) -> None:
    """Reset references when extraction is intentionally disabled."""

    metadata.references = []
    metadata.references_incomplete = False
    metadata._references_incomplete_diagnostic = ""


def references_incomplete_issue(metadata: PaperMetadata) -> ValidationIssue:
    diagnostic = metadata._references_incomplete_diagnostic
    message = "Reference extraction failed after core metadata succeeded; references are incomplete"
    if diagnostic:
        message = f"{message} ({diagnostic})"
    return ValidationIssue(
        code="VAL_REFERENCES_INCOMPLETE",
        severity=IssueSeverity.ERROR,
        message=message,
        origin_stage="extract",
        blocking=True,
    )
