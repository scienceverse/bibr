"""Enricher protocol — post-extraction transforms on a single ``Paper``.

An ``Enricher`` runs after ``PostParseStage`` has produced ``fs.paper``.
It enriches the paper's metadata (e.g. adds DOIs from Crossref, runs NER
on references). Failures set ``fs.warnings``, never ``fs.error``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.pipeline.state import FileState

logger = logging.getLogger(__name__)


@runtime_checkable
class Enricher(Protocol):
    """A post-extraction enricher running on one paper at a time."""

    name: str

    async def enrich(self, fs: FileState) -> EnrichmentOutcome | None: ...


class EnrichmentStatus(StrEnum):
    NO_WORK = "no_work"
    COMPLETE = "complete"
    PARTIAL = "partial"


@dataclass(frozen=True)
class EnrichmentOutcome:
    status: EnrichmentStatus
    warnings: tuple[str, ...] = ()
    detail: str | None = None


class CrossrefEnricher:
    """Enriches references with Crossref metadata (DOI + bibliographic search)."""

    name = "crossref"

    def __init__(
        self,
        timeout: float | None = None,
        *,
        settings: GlobalSettings | None = None,
    ) -> None:
        from bibr.config import snapshot_settings

        self._timeout = timeout
        self._settings = settings if settings is not None else snapshot_settings()

    async def enrich(self, fs: FileState) -> EnrichmentOutcome:
        paper = fs.paper
        if paper is None or paper.metadata is None:
            return EnrichmentOutcome(EnrichmentStatus.NO_WORK)
        meta = paper.metadata
        if not meta.references and not meta.doi:
            return EnrichmentOutcome(EnrichmentStatus.NO_WORK)
        from bibr.enrich.references import (
            EnrichmentReport,
            enrich_paper_identity,
            enrich_references,
        )

        timeout = (
            self._timeout if self._timeout is not None else self._settings.crossref.enrich_timeout
        )

        async def _run() -> list[EnrichmentReport]:
            reports: list[EnrichmentReport] = []
            # Self-DOI lookup first (one call), then the reference fan-out.
            if meta.doi:
                report = await enrich_paper_identity(meta, settings=self._settings)
                reports.append(
                    report
                    if isinstance(report, EnrichmentReport)
                    else EnrichmentReport(attempted=1)
                )
            if meta.references:
                report = await enrich_references(meta.references, settings=self._settings)
                reports.append(
                    report
                    if isinstance(report, EnrichmentReport)
                    else EnrichmentReport(attempted=len(meta.references))
                )
            return reports

        try:
            reports = await asyncio.wait_for(_run(), timeout=timeout)
            details = tuple(
                dict.fromkeys(detail for report in reports for detail in report.details)
            )
            partial = any(report.failed for report in reports)
            if meta.references:
                meta.enrichment_complete = not partial
            return EnrichmentOutcome(
                EnrichmentStatus.PARTIAL if partial else EnrichmentStatus.COMPLETE,
                warnings=details,
                detail=(
                    f"{sum(report.failed for report in reports)} terminal enrichment failures"
                    if partial
                    else None
                ),
            )
        except TimeoutError:
            if meta.references:
                meta.enrichment_complete = False
            fs.warnings.append("Crossref enrichment timed out")
            logger.warning("[%s] Crossref enrichment timed out", fs.path.name)
            return EnrichmentOutcome(
                EnrichmentStatus.PARTIAL,
                warnings=("Crossref enrichment timed out",),
                detail="Crossref enrichment timed out",
            )
        except Exception as e:  # noqa: BLE001
            if meta.references:
                meta.enrichment_complete = False
            fs.warnings.append(f"Crossref enrichment failed: {e}")
            logger.warning("[%s] Crossref enrichment failed: %s", fs.path.name, e)
            warning = f"Crossref enrichment failed: {e}"
            return EnrichmentOutcome(
                EnrichmentStatus.PARTIAL,
                warnings=(warning,),
                detail=warning,
            )
