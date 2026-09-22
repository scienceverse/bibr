"""Enricher protocol — post-extraction transforms on a single ``Paper``.

An ``Enricher`` runs after ``PostParseStage`` has produced ``fs.paper``.
It enriches the paper's metadata (e.g. adds DOIs from Crossref, runs NER
on references). Failures set ``fs.warnings``, never ``fs.error``.
"""

from __future__ import annotations

import asyncio
import logging
import time
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

    async def _await_prefetch(self, fs: FileState, handle, enrich_started_at: float):
        """Consume the prefetch started under extraction; ``None`` → inline path.

        Records the prefetch wall time as ``fs.stage_times["enrich_prefetch"]``
        and logs how much of it was hidden under the extract stage. A failed
        or cancelled prefetch is a debug-level event, never an enrichment
        failure: the inline path simply redoes the round-trips.
        """
        from bibr.pipeline.enrich_prefetch import PrefetchUnavailable

        try:
            prefetch = await handle.result()
        except PrefetchUnavailable as exc:
            logger.debug(
                "[%s] enrichment prefetch unavailable (%s); prefetching inline", fs.path.name, exc
            )
            prefetch = None
        seconds = handle.seconds
        if seconds is not None:
            fs.stage_times["enrich_prefetch"] = seconds
            hidden, exposed = handle.overlap(enrich_started_at)
            logger.debug(
                "[%s] enrichment prefetch for %d refs took %.2fs: %.2fs overlapped "
                "extraction, %.2fs on the enrich stage's critical path%s",
                fs.path.name,
                handle.n_references,
                seconds,
                hidden,
                exposed,
                "" if exposed > 0 else " (fully hidden)",
            )
        return prefetch

    async def enrich(self, fs: FileState) -> EnrichmentOutcome:
        paper = fs.paper
        if paper is None or paper.metadata is None:
            return EnrichmentOutcome(EnrichmentStatus.NO_WORK)
        meta = paper.metadata
        from bibr.pipeline.enrich_prefetch import take_prefetch_handle

        # Own the prefetch from here on: whether it is consumed below or the
        # paper turns out to have nothing to enrich, nobody else must await it.
        prefetch_handle = take_prefetch_handle(paper)
        if not meta.references and not meta.doi:
            if prefetch_handle is not None:
                await prefetch_handle.discard()
            return EnrichmentOutcome(EnrichmentStatus.NO_WORK)
        from bibr.enrich.references import (
            EnrichmentReport,
            enrich_paper_identity,
            enrich_references,
        )

        timeout = (
            self._timeout if self._timeout is not None else self._settings.crossref.enrich_timeout
        )
        enrich_started_at = time.monotonic()

        async def _identity() -> EnrichmentReport | None:
            if not meta.doi:
                return None
            report = await enrich_paper_identity(meta, settings=self._settings)
            return report if isinstance(report, EnrichmentReport) else EnrichmentReport(attempted=1)

        async def _references() -> EnrichmentReport | None:
            if not meta.references:
                if prefetch_handle is not None:
                    await prefetch_handle.discard()
                return None
            kwargs: dict = {"settings": self._settings}
            if prefetch_handle is not None:
                # Inside the timeout budget: a timed-out wait cancels the
                # prefetch along with the rest of the enrichment.
                prefetch = await self._await_prefetch(fs, prefetch_handle, enrich_started_at)
                if prefetch is not None:
                    kwargs["prefetch"] = prefetch
            report = await enrich_references(meta.references, **kwargs)
            return (
                report
                if isinstance(report, EnrichmentReport)
                else EnrichmentReport(attempted=len(meta.references))
            )

        async def _run() -> list[EnrichmentReport]:
            # The self-DOI lookup writes only ``meta.match`` and the reference
            # fan-out only each ``ref.match``; they share nothing but the
            # Crossref rate limiter, so the paper's own lookup no longer delays
            # the references by a round-trip. Both finish before a failure is
            # raised, so neither is left running unowned; a timeout cancels both.
            results = await asyncio.gather(_identity(), _references(), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            return [result for result in results if result is not None]

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
