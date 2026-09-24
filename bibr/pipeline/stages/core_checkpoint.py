"""Persist a schema-valid unenriched core before optional enrichment."""

from __future__ import annotations

import logging

from bibr.pipeline.artifacts import RunState, disposition_for_issues, mark_enrichment_pending
from bibr.pipeline.stages.export import build_result_payload
from bibr.validation import payload_validation

logger = logging.getLogger(__name__)


class CoreCheckpointStage:
    name = "core_checkpoint"
    requires = ("paper",)
    produces = ("artifact_disposition", "core_sha256", "result_json")

    def __init__(self, *, enrichment_requested: bool = False) -> None:
        self._enrichment_requested = enrichment_requested

    async def run(self, ctx) -> None:
        ctx.progress.stage_start(self.name)
        for fs in ctx.alive():
            paper = fs.paper
            if paper is None:
                continue
            fs.artifact_disposition = disposition_for_issues(paper.validation_issues)
            sink = fs.artifact_sink
            if sink is None:
                continue
            try:
                if not fs.artifact_started:
                    sink.record(fs, RunState.STARTED)
                    fs.artifact_started = True
                payload = await build_result_payload(ctx, fs)
                # The serialized validation block is the public authority: it
                # may contain output-only blockers absent from Paper's in-memory
                # issue list. Resolve routing before adding our temporary gate.
                serialized_issues = (payload_validation(payload) or {}).get("issues") or []
                fs.artifact_disposition = disposition_for_issues(serialized_issues)
                enrichment_requested = (
                    self._enrichment_requested
                    and ctx.config.enrichment_enabled(ctx.settings)
                    and (ctx.config.ref_parse_strategy or ctx.settings.REF_PARSE_STRATEGY) != "off"
                )
                if enrichment_requested:
                    payload = mark_enrichment_pending(payload)
                fs.result_json = payload
                fs.core_sha256 = sink.write_core(fs, payload)
                try:
                    sink.materialize(fs, payload)
                except Exception as materialize_exc:
                    # The immutable core is already durable: record that fact
                    # before surfacing the independently failed public write.
                    sink.record(
                        fs,
                        RunState.CORE_WRITTEN,
                        detail=f"public_materialization_failed: {materialize_exc}",
                    )
                    raise
                sink.record(
                    fs,
                    RunState.CORE_WRITTEN,
                    detail=None if enrichment_requested else "enrichment_not_requested",
                )
            except Exception as exc:  # noqa: BLE001 - per-file durable failure
                try:
                    sink.record(fs, RunState.FAILED, detail=str(exc))
                except Exception:  # noqa: BLE001 - retain the originating failure
                    logger.warning("Failed to persist checkpoint failure receipt", exc_info=True)
                fs.set_error(
                    f"Core checkpoint failed: {exc}",
                    code="core_checkpoint_failed",
                    stage=self.name,
                    exc=exc,
                )
                logger.warning("Core checkpoint failed for %s", fs.path.name, exc_info=True)
        ctx.progress.stage_end(self.name)
