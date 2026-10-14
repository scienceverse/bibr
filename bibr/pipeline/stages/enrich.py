"""EnrichmentStage — drives a list of Enricher objects over alive papers.

Each enricher runs against every alive file that has a populated
``fs.paper``; failures are collected into ``fs.warnings`` by the enricher
itself, so this stage never sets ``fs.error``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from bibr.pipeline.enrich_prefetch import cancel_leftover_prefetches, discard_prefetches
from bibr.processing_warnings import ProcessingWarning, WarningCode
from bibr.utils.redact import describe_error

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext
    from bibr.pipeline.enricher import Enricher

logger = logging.getLogger(__name__)


class EnrichmentStage:
    name = "enrich"
    # FileState fields consumed / populated (see validate_stage_contracts).
    requires = ("paper",)
    produces = ()

    def __init__(
        self,
        enrichers: Sequence[Enricher],
        identity_enrichers: Sequence[Enricher] = (),
    ) -> None:
        # ``enrichers`` run when reference enrichment is on; ``identity_enrichers``
        # (the paper's own DOI lookup) when only that was asked for.
        self._enrichers = enrichers
        self._identity_enrichers = identity_enrichers

    @staticmethod
    async def _record_cutoff(papers, detail: str) -> None:
        """Best-effort durable interruption receipt without masking cancellation."""

        async def _record(fs):
            sink = fs.artifact_sink
            if sink is None or fs.core_sha256 is None:
                return
            from bibr.pipeline.artifacts import RunState

            try:
                await asyncio.shield(
                    asyncio.to_thread(
                        sink.record,
                        fs,
                        RunState.CUTOFF_INTERRUPTED,
                        detail=detail,
                    )
                )
            except Exception:  # noqa: BLE001 - interruption remains authoritative
                logger.warning(
                    "Failed to persist cutoff receipt for %s", fs.path.name, exc_info=True
                )

        await asyncio.gather(*[_record(fs) for fs in papers])

    async def run(self, ctx: PipelineContext) -> None:
        # The serve pipeline is shared across requests, so request-scoped
        # switches (refs=off, the tri-state ``crossref`` and
        # ``crossref_identity`` form fields) cannot be represented by removing
        # the stage at construction: gate per run.
        mode = ctx.config.enrichment_mode(ctx.settings)
        if mode == "full" and self._enrichers:
            enrichers = self._enrichers
            papers = [fs for fs in ctx.alive() if fs.paper is not None]
        elif mode == "identity" and self._identity_enrichers:
            from bibr.pipeline.enricher import needs_identity_lookup

            enrichers = self._identity_enrichers
            # Only papers with a DOI: the checkpoint marked just these pending,
            # and the rest keep their core as the final export.
            papers = [fs for fs in ctx.alive() if needs_identity_lookup(fs.paper)]
        else:
            # Nothing will consume an enrichment prefetch on this path (e.g. a
            # serve request that switched refs/enrichment off against a
            # pipeline that started one): settle it before leaving.
            await discard_prefetches(ctx.alive())
            return
        ctx.progress.stage_start(self.name)
        t0 = time.monotonic()
        if papers:
            unexpected_failure_ids: set[int] = set()
            explicit_partial_ids: set[int] = set()
            enrichment_warnings: dict[int, list[ProcessingWarning]] = {id(fs): [] for fs in papers}
            enrichment_details: dict[int, list[str]] = {id(fs): [] for fs in papers}

            async def _timed(enricher, fs):
                # Per-file wall clock, accumulated across enrichers, so batch
                # exports can report each paper's own enrichment time.
                fs_t0 = time.monotonic()
                try:
                    return await enricher.enrich(fs)
                finally:
                    fs.stage_times[self.name] = fs.stage_times.get(self.name, 0.0) + (
                        time.monotonic() - fs_t0
                    )

            for enricher in enrichers:
                # return_exceptions: one file's enricher blowing up must not
                # cancel its siblings or abort the chunk — enrichment is
                # best-effort, so unexpected failures demote to warnings.
                try:
                    results = await asyncio.gather(
                        *[_timed(enricher, fs) for fs in papers], return_exceptions=True
                    )
                except (asyncio.CancelledError, KeyboardInterrupt, SystemExit) as exc:
                    cancel_leftover_prefetches(papers)
                    await self._record_cutoff(papers, type(exc).__name__)
                    raise
                enricher_name = type(enricher).__name__
                for fs, res in zip(papers, results, strict=True):
                    if isinstance(res, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                        cancel_leftover_prefetches(papers)
                        await self._record_cutoff(papers, type(res).__name__)
                        raise res
                    if isinstance(res, BaseException):
                        unexpected_failure_ids.add(id(fs))
                        logger.warning(
                            "Enricher %s failed for %s: %s",
                            enricher_name,
                            fs.path.name,
                            res,
                            exc_info=res,
                        )
                        warning = ProcessingWarning(
                            WarningCode.ENRICHER_FAILED,
                            f"{enricher_name} failed: {describe_error(res)}",
                        )
                        fs.warnings.append(warning)
                        enrichment_warnings[id(fs)].append(warning)
                    else:
                        from bibr.pipeline.enricher import EnrichmentOutcome, EnrichmentStatus

                        if isinstance(res, EnrichmentOutcome):
                            if res.status is EnrichmentStatus.PARTIAL:
                                explicit_partial_ids.add(id(fs))
                            enrichment_warnings[id(fs)].extend(res.warnings)
                            if res.detail:
                                enrichment_details[id(fs)].append(res.detail)
                            fs.warnings = list(dict.fromkeys([*fs.warnings, *res.warnings]))
            from bibr.pipeline.artifacts import RunState

            for fs in papers:
                metadata = getattr(fs.paper, "metadata", None)
                explicitly_partial = getattr(metadata, "enrichment_complete", None) is False
                fs.enrichment_warnings = list(dict.fromkeys(enrichment_warnings[id(fs)]))
                fs.enrichment_detail = (
                    "; ".join(dict.fromkeys(enrichment_details[id(fs)]))[:512]
                    if enrichment_details[id(fs)]
                    else None
                )
                # Deliberately not "did fs.warnings grow?": CrossrefEnricher
                # returns COMPLETE with a non-empty warnings tuple whenever any
                # detail is recorded — including a reference whose DOI lookup
                # errored but which a later title search resolved. That demoted
                # a fully-enriched paper to PARTIAL and put a blocking
                # VAL_ENRICHMENT_PENDING gate on the export with failed == 0.
                fs.enrichment_state = (
                    RunState.ENRICHMENT_PARTIAL
                    if id(fs) in unexpected_failure_ids
                    or id(fs) in explicit_partial_ids
                    or explicitly_partial
                    else RunState.ENRICHMENT_COMPLETE
                )
            # An enricher that never reached the prefetch (it raised first, or
            # a custom one ignores it) leaves the handle on the paper: settle
            # it so the stage never leaks a task.
            await discard_prefetches(papers)
        if mode == "identity":
            # The identity lookup consumes no reference prefetch, and papers
            # without a DOI were never in ``papers``.
            await discard_prefetches(ctx.alive())
        logger.debug("Enrichment stage: %.1fs", time.monotonic() - t0)
        ctx.progress.stage_end(self.name)
