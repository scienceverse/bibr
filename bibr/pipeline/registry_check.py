"""Look the paper's own DOI up while its front matter is still being read.

The paper's registry record (``metadata_match``) normally arrives with
enrichment, after the title and authors are decided. Two front-matter rules
need it before then: the registry title rule (``FRONT_MATTER_REGISTRY_TITLE``),
which picks between printed title rows, and the second-opinion review
(``LLM_REVIEW_TRIGGER=registry_disagree``), which re-asks a disagreeing title
or author call. When either is on and the run looks the paper's DOI up anyway
(``CROSSREF_IDENTITY`` or ``CROSSREF_ENRICH``), ``post_parse`` reads a
preliminary DOI from the parsed text and starts the lookup as a task before
the core metadata call, so the round-trip hides under that call.

The preliminary DOI is used only when the selection is unambiguous. The
extract stage reads the PDF's own DOI evidence (text layer, links, document
info) first and hands it to both selections, so the identity stage selects
the final DOI from the same evidence; the two differ only when the parsed
text changes in between. When they differ, the identity stage drops what
the early record decided (:func:`reconcile_with_final_doi`) and reports
``VAL_REGISTRY_DOI_CHANGED``. When they agree, the record found here is the
paper's ``metadata_match`` and the enrichment stage does not look the DOI up
again. A failed or timed-out lookup leaves no record and the later lookup
runs as usual. The handle rides the ``Paper`` as ``paper.registry_check``
(runtime-only, never exported).

The record never replaces a printed value: it only chooses between texts the
paper prints.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bibr.config import GlobalSettings
    from bibr.extract.field_decisions import FieldCandidate, FieldDecision
    from bibr.extract.pdf_doi_evidence import PdfDoiEvidence
    from bibr.models import ExternalMatch, MatchSource, PaperAuthor
    from bibr.pipeline.identity import ExpectedIdentity
    from bibr.validation import ValidationIssue

logger = logging.getLogger(__name__)

RegistryCheckStatus = Literal[
    "pending", "found", "not_found", "failed", "cancelled", "year_mismatch"
]

# Statuses after which a later lookup of the same DOI would add nothing.
_COMPLETE = frozenset({"found", "not_found"})

# A record dated more than this many years from the printed publication year
# is another work (the enrichment's own DOI-lookup rule).
_MAX_YEAR_GAP = 1

# asyncio keeps only weak references to tasks: hold ours until they finish.
_INFLIGHT: set[asyncio.Task[Any]] = set()


def registry_check_wanted(settings: GlobalSettings) -> bool:
    """Whether a front-matter rule reads the registry record before enrichment."""
    title_rule = getattr(settings, "FRONT_MATTER_REGISTRY_TITLE", False) is True
    llm = getattr(settings, "llm", None)
    return title_rule or getattr(llm, "review_trigger", "off") == "registry_disagree"


def preliminary_doi(
    contents: Any,
    expected_identity: ExpectedIdentity | None = None,
    pdf_evidence: PdfDoiEvidence | None = None,
) -> str | None:
    """The DOI the paper names unambiguously, or None.

    The selection the identity stage makes, from the same evidence when the
    caller read the PDF's own DOI evidence (*pdf_evidence*), and used only
    when it raised no issue (no tie, no expected-identity conflict).
    """
    from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates

    try:
        selection = select_doi_candidates(
            collect_doi_candidates(contents, pdf_evidence), expected_identity
        )
    except Exception:  # noqa: BLE001 - the early record is optional
        logger.debug("Preliminary DOI selection failed", exc_info=True)
        return None
    if selection.selected is None or selection.issues:
        return None
    return selection.selected.normalized or None


def _year(value: Any) -> int | None:
    from bibr.enrich.references import _year_from_published

    return _year_from_published(value) if isinstance(value, str) else None


def author_rows(authors: Iterable[PaperAuthor]) -> list[dict]:
    """``{given, family, literal}`` rows, as ``author[]`` exports them."""
    from bibr.models import ORGANIZATION_ROLE

    rows = []
    for author in authors:
        given = (author.given or "").strip() or None
        family = (author.family or "").strip() or None
        if ORGANIZATION_ROLE in (author.role or []):
            literal = " ".join(part for part in (given, family) if part) or None
            rows.append({"given": None, "family": None, "literal": literal})
        else:
            rows.append({"given": given, "family": family, "literal": None})
    return rows


def match_row(source: MatchSource | str, match: ExternalMatch) -> dict:
    """One ``metadata_match`` row as the export writes it (score on 0-1)."""
    score = match.score
    return {
        "service": getattr(source, "value", source),
        "score": None if score is None else min(max(float(score) / 100.0, 0.0), 1.0),
        "doi": match.doi,
        "title": match.title,
        "subtitle": match.subtitle,
        "author": [{"given": a.given, "family": a.family} for a in match.authors or []],
    }


class RegistryCheck:
    """The early lookup of the paper's own DOI, and what it decided."""

    __slots__ = (
        "early_doi",
        "finished_at",
        "matches",
        "primary_title",
        "record_backed",
        "review_receipt",
        "started_at",
        "status",
        "task",
        "title_fallback",
    )

    def __init__(self, task: asyncio.Task, *, doi: str, started_at: float) -> None:
        self.task = task
        self.early_doi = doi
        self.started_at = started_at
        self.finished_at: float | None = None
        self.status: RegistryCheckStatus = "pending"
        self.matches: dict[MatchSource, ExternalMatch] = {}
        # Fields whose value was chosen because it agrees with the record.
        self.record_backed: set[str] = set()
        # The title decision without the record, restored when the final DOI
        # turns out to be another one, and the first extracted title when the
        # review replaced it because the record agreed with its answer.
        self.title_fallback: FieldDecision | None = None
        self.primary_title: FieldCandidate | None = None
        # The front-matter review's receipt (``extraction.diagnostics``).
        self.review_receipt: dict | None = None
        task.add_done_callback(self._on_done)

    def _on_done(self, task: asyncio.Task) -> None:
        if self.finished_at is None:
            self.finished_at = time.monotonic()
        _INFLIGHT.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.debug("Early registry lookup failed: %s", task.exception())

    @property
    def seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return max(0.0, self.finished_at - self.started_at)

    @property
    def complete(self) -> bool:
        """Whether a later lookup of :attr:`early_doi` would add nothing."""
        return self.status in _COMPLETE

    def covers(self, doi: str | None) -> bool:
        """Whether this lookup already answered for *doi*."""
        return self.complete and bool(doi) and str(doi).casefold() == self.early_doi.casefold()

    async def settle(self, published: str | None = None) -> RegistryCheckStatus:
        """Wait for the lookup (bounded by its own timeout) and keep its record.

        *published* is the printed publication date; a record more than a year
        away from it is dropped (``year_mismatch``), as the enrichment's own
        DOI lookup does. Idempotent: only the first call waits. A failed,
        timed-out or cancelled lookup leaves no record. The caller's own
        cancellation propagates.
        """
        if self.status != "pending":
            return self.status
        try:
            report, matches = await self.task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            self.status = "cancelled"
            return self.status
        except TimeoutError:
            logger.info("Early registry lookup of %s timed out", self.early_doi)
            self.status = "failed"
            return self.status
        except Exception as exc:  # noqa: BLE001 - the early record is optional
            logger.info("Early registry lookup of %s failed: %s", self.early_doi, exc)
            self.status = "failed"
            return self.status
        finally:
            if self.task.done() and self.finished_at is None:
                self.finished_at = time.monotonic()
        if getattr(report, "matched", 0):
            printed_year = _year(published)
            kept = {
                source: match
                for source, match in matches.items()
                if not (
                    printed_year
                    and match.year
                    and abs(int(match.year) - printed_year) > _MAX_YEAR_GAP
                )
            }
            if kept:
                self.matches = kept
                self.status = "found"
            else:
                self.status = "year_mismatch"
        elif getattr(report, "failed", 0):
            self.status = "failed"
        else:
            self.status = "not_found"
        return self.status

    def record(self) -> dict | None:
        """The confident record of :attr:`early_doi`, export-shaped, or None."""
        if self.status != "found":
            return None
        from bibr.enrich.verify import _best_match

        rows = [match_row(source, match) for source, match in self.matches.items()]
        row = _best_match(rows)
        if row is None:
            return None
        if row.get("doi") and str(row["doi"]).strip().casefold() != self.early_doi.casefold():
            return None
        return row

    def cancel(self) -> None:
        if not self.task.done():
            self.task.cancel()


async def _lookup(doi: str, settings: GlobalSettings) -> tuple[Any, dict]:
    from bibr.enrich.references import enrich_paper_identity

    # The printed date is not read yet: the year check runs in ``settle``.
    probe = SimpleNamespace(doi=doi, published=None, match={})
    report = await enrich_paper_identity(probe, settings=settings)
    return report, dict(probe.match)


async def _bounded_lookup(doi: str, settings: GlobalSettings, timeout: float):
    return await asyncio.wait_for(_lookup(doi, settings), timeout=timeout)


def start_registry_check(
    contents: Any,
    *,
    settings: GlobalSettings,
    expected_identity: ExpectedIdentity | None = None,
    pdf_evidence: PdfDoiEvidence | None = None,
) -> RegistryCheck | None:
    """Start the early lookup; None when the paper names no single DOI."""
    doi = preliminary_doi(contents, expected_identity, pdf_evidence)
    if not doi:
        return None
    timeout = float(settings.crossref.request_timeout)
    started = time.monotonic()
    task = asyncio.create_task(
        _bounded_lookup(doi, settings, timeout), name=f"bibr-registry-check[{doi}]"
    )
    _INFLIGHT.add(task)
    return RegistryCheck(task, doi=doi, started_at=started)


def registry_check_of(paper: Any) -> RegistryCheck | None:
    """The early lookup attached to *paper*, if any (tolerates test doubles)."""
    check = getattr(paper, "registry_check", None)
    return check if isinstance(check, RegistryCheck) else None


def covered_by_registry_check(paper: Any) -> bool:
    """Whether the early lookup already answered for the paper's final DOI."""
    check = registry_check_of(paper)
    metadata = getattr(paper, "metadata", None)
    return check is not None and check.covers(getattr(metadata, "doi", None))


def reconcile_with_final_doi(paper: Any) -> ValidationIssue | None:
    """Drop the early record's effects when the final DOI is another one.

    Called by the identity stage right after it writes ``metadata.doi``. The
    record leaves ``metadata.match``, and the check is detached so the
    enrichment stage looks the final DOI up. A title chosen because it agreed
    with the record goes back to the decision made without it, with that
    decision's issues in place of the dropped one's. An author list the
    review chose because it agreed with the record is kept: every name in it
    is printed, and the affiliations, contacts and contribution roles are
    already built on it. Its source stays ``llm_review``, but the issue
    returned here (which ``extraction.fields`` lists for the title and the
    author list) keeps the export from reporting it ``corrected``.

    Returns ``VAL_REGISTRY_DOI_CHANGED`` when the front-matter rules could
    read the early record; an early lookup that found no confident record is
    dropped silently.
    """
    from bibr.extract.field_decisions import apply_decision, field_decisions_of
    from bibr.validation import IssueSeverity, ValidationIssue

    check = registry_check_of(paper)
    metadata = getattr(paper, "metadata", None)
    if check is None or metadata is None:
        return None
    final = str(getattr(metadata, "doi", "") or "")
    if final.casefold() == check.early_doi.casefold():
        return None
    used = check.record() is not None
    paper.registry_check = None
    check.cancel()
    for source in check.matches:
        metadata.match.pop(source, None)
    if not used:
        return None
    restored = []
    if "title" in check.record_backed and check.title_fallback is not None:
        ledger = field_decisions_of(metadata)
        dropped = ledger.get("title") if ledger is not None else None
        apply_decision(metadata, check.title_fallback, supersede=True)
        issues = getattr(paper, "validation_issues", None)
        if isinstance(issues, list):
            stale = set(dropped.issues) if dropped is not None else set()
            issues[:] = [issue for issue in issues if issue not in stale]
            for issue in check.title_fallback.issues:
                if issue not in issues:
                    issues.append(issue)
        restored.append("the title was decided again without it")
    if "author" in check.record_backed:
        restored.append(
            "the author list the review chose with it is kept (every name is printed) but no "
            "longer counts as corrected"
        )
    message = (
        f"The final DOI ({final or 'none'}) differs from the one read early ({check.early_doi}); "
        "the early registry record was dropped"
        + (f" and {', and '.join(restored)}" if restored else "")
    )
    return ValidationIssue(
        code="VAL_REGISTRY_DOI_CHANGED",
        severity=IssueSeverity.WARNING,
        message=message,
        origin_stage="identity",
        evidence_ids=(f"doi:{check.early_doi}", f"doi:{final}" if final else "doi:none"),
    )


__all__ = [
    "RegistryCheck",
    "author_rows",
    "covered_by_registry_check",
    "match_row",
    "preliminary_doi",
    "reconcile_with_final_doi",
    "registry_check_of",
    "registry_check_wanted",
    "start_registry_check",
]
