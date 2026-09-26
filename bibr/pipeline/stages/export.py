"""ExportStage — serialize Paper to JSON, free state."""

from __future__ import annotations

import copy
import gc
import logging
import time
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING

from bibr.processing_warnings import ProcessingWarning, WarningCode

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

# gc.collect() holds the GIL for the whole pass, stalling the event loop —
# in serve mode every request is one chunk, so an unconditional collect per
# chunk stacks loop stalls under load. Throttle to one full collect per
# window; refcounting already frees the big DataFrames immediately, the
# collect only mops up cycles.
_GC_MIN_INTERVAL_SECONDS = 30.0

# Per-file timings that overlap another stage's wall clock and therefore must
# not be added into ``extraction.timings.total_seconds``.
_OVERLAPPED_TIMINGS = frozenset({"enrich_prefetch"})

# Warnings meaning the configured reference segmenter did NOT produce the final
# segmentation: an LLM->CRF fall-back, a geom->LLM cascade, or a region or CRF
# segmenter that raised. (Merge-split is a correction layered on top of the
# segmenter, not a fall-back from it, so it does not count.)
_REF_SEG_FALLBACK_CODES = frozenset(
    {
        WarningCode.REF_SEG_CRF_FALLBACK,
        WarningCode.REF_SEG_GEOM_CASCADE,
        WarningCode.REF_SEG_REGION_ERROR,
        WarningCode.REF_SEG_CRF_ERROR,
    }
)


# Input formats parsed natively, without ever reaching an OCR engine: DocxHandling
# Stage (bibr/pipeline/stages/docx.py) and HtmlHandlingStage (.html/.htm/.epub)
# populate ``contents`` directly, and the OCR stage skips every file that already
# has it. ``extraction.ocr`` must be ``null`` for these: reporting the
# *configured* backend would fabricate an engine for a file it never saw.
# Mirrors SupportedFileType (bibr/input/supported_files.py) minus PDF; both
# ``file_type`` spellings for .htm are listed since the two producers disagree.
_NATIVE_PARSE_FORMATS = frozenset({"docx", "xml", "html", "htm", "epub"})


def _build_engines(ctx: PipelineContext, paper=None) -> tuple[dict | None, dict | None]:
    """Resolve the OCR and core-LLM engine identities for ``extraction``.

    ``None`` is meaningful for both: no OCR engine ran (the DOCX/XML native
    paths), or the run deliberately had no LLM (``--no-llm``).

    ``resolve_ocr_runtime_identity`` is pure config resolution — it answers
    "which backend *would* run", not "which one did" — so the native-parse
    formats are excluded here from *paper*'s declared input format. Callers
    without a paper (identity-probe tests) get the resolved configuration.
    """
    from bibr.export.json_export import normalized_input_format
    from bibr.ocr.profiles import resolve_ocr_runtime_identity

    ocr = None
    if normalized_input_format(getattr(paper, "input_file", None)) not in _NATIVE_PARSE_FORMATS:
        scratch = getattr(ctx, "scratch", None) or {}
        identity = scratch.get("ocr_runtime_identity")
        if identity is None:
            identity = resolve_ocr_runtime_identity(ctx.config, ctx.settings)
        if identity is not None and identity.backend:
            ocr = {
                "backend": identity.backend,
                "model": identity.model,
                "profile": identity.profile,
            }

    llm = None
    if not ctx.config.no_llm:
        llm = {
            "provider": ctx.settings.llm.provider,
            "model": ctx.settings.llm.model,
            "backend": getattr(ctx.settings.llm, "backend", None),
        }
    return ocr, llm


def _build_extraction(ctx: PipelineContext, paper, fs=None) -> dict:
    """Extraction provenance for the v11 ``extraction`` block.

    Records the producing engines, the bibr package version, the resolved
    reference seg/parse strategies, whether the CRF seg-fallback fired, the
    *effective* Crossref-enrich/consolidate modes, per-stage wall-clock
    timings, LLM token usage, and the non-fatal processing warnings.

    Timings come from the file's own ``fs.stage_times`` — in a multi-file
    chunk the chunk wall clock must not be attributed to every paper. The
    chunk-level ``ctx.scratch["stage_timings"]`` is only a fallback for
    callers without per-file times.
    """
    import datetime as _dt

    from bibr.export.json_export import bibr_producer, export_section_type
    from bibr.export.usage import build_usage_export
    from bibr.extract.extractor import _resolve_ref_strategies

    seg_strategy, parse_strategy = _resolve_ref_strategies(
        getattr(ctx.config, "ref_seg_strategy", None),
        getattr(ctx.config, "ref_parse_strategy", None),
        settings=ctx.settings,
    )
    fallback_used = any(
        w.code in _REF_SEG_FALLBACK_CODES for w in (paper.processing_warnings or [])
    )
    raw_timings = (getattr(fs, "stage_times", None) or {}) or (
        (getattr(ctx, "scratch", None) or {}).get("stage_timings") or {}
    )
    # Absent, not a zeroed row: an untimed run did not record timings at all.
    timings = (
        {
            "stages": {k: round(float(v), 3) for k, v in raw_timings.items()},
            "total_seconds": round(
                sum(v for k, v in raw_timings.items() if k not in _OVERLAPPED_TIMINGS), 3
            ),
        }
        if raw_timings
        else None
    )
    ocr, llm = _build_engines(ctx, paper)

    extraction = {
        "producer": bibr_producer(ctx.settings.BIBR_BUILD_SHA),
        "completed_at": _dt.datetime.now(_dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "ocr": ocr,
        "llm": llm,
        "settings": {
            "ref_seg": seg_strategy,
            "ref_parse": parse_strategy,
            "crossref_enrich": ctx.config.enrichment_enabled(ctx.settings),
            "consolidate": ctx.config.consolidate or ctx.settings.crossref.consolidate,
        },
        "timings": timings,
        "usage": build_usage_export(paper.llm_usage_labels or None),
        "diagnostics": {
            "text_quality": paper.text_quality,
            "references_complete": not bool(
                getattr(getattr(paper, "metadata", None), "references_incomplete", False)
            ),
            "ref_seg_fallback_used": fallback_used,
        },
        "warnings": [w.to_dict() for w in paper.processing_warnings or []],
        # Opt-in (LLM_CAPTURE_TRACE); a sibling of "regions", not nested under
        # diagnostics. Empty list -> None so the key is omitted entirely when
        # nothing was captured (absence rule).
        "trace": list(paper.llm_trace or []) or None,
    }

    identity_block = {}
    expected_identity = getattr(paper, "expected_identity", None)
    if is_dataclass(expected_identity):
        identity_block["expected"] = asdict(expected_identity)
    doi_selection = getattr(paper, "doi_selection", None)
    if is_dataclass(doi_selection):

        def _candidate(candidate) -> dict:
            row = asdict(candidate)
            row["section_type"] = export_section_type(row.get("section_type"))
            return row

        identity_block["receipt"] = {
            "selected": _candidate(doi_selection.selected)
            if doi_selection.selected is not None
            else None,
            "candidates": [_candidate(candidate) for candidate in doi_selection.candidates],
            "issue_codes": [issue.code for issue in doi_selection.issues],
        }
    if identity_block:
        extraction["identity"] = identity_block
    return extraction


async def build_result_payload(ctx: PipelineContext, fs) -> dict:
    """Prepare and serialize one paper identically for checkpoint and export."""

    import asyncio

    if fs.paper is None:
        raise RuntimeError("Paper not built before export stage")
    # Merge stage warnings BEFORE building the block: ``extraction.warnings``
    # snapshots ``paper.processing_warnings``, so appending afterwards would
    # drop them.
    if fs.warnings:
        fs.paper.processing_warnings = list(
            dict.fromkeys([*fs.paper.processing_warnings, *fs.warnings])
        )
    fs.paper.extraction = _build_extraction(ctx, fs.paper, fs)
    return await asyncio.to_thread(
        fs.paper.export_to_json,
        include_regions=ctx.config.include_regions,
        include_region_meta=ctx.config.include_region_meta,
    )


async def _consolidate_payload(ctx: PipelineContext, payload: dict) -> bool:
    import asyncio

    mode = ctx.config.consolidate or ctx.settings.crossref.consolidate
    if mode == "off":
        return False
    from bibr.enrich.consolidate import consolidate_bibs
    from bibr.export.json_export import append_payload_warning

    await asyncio.to_thread(consolidate_bibs, payload, mode=mode)
    crossref_on = ctx.config.enrichment_enabled(ctx.settings)
    if not crossref_on and not payload.get("bib_match"):
        append_payload_warning(
            payload,
            ProcessingWarning(
                WarningCode.CONSOLIDATE_WITHOUT_ENRICHMENT,
                "consolidate enabled but Crossref enrichment is off — no matches to merge",
            ),
        )
    return True


def _load_checkpoint_core(sink, fs) -> dict:
    """Read the immutable local core, with protocol-only sink compatibility."""

    read_core = getattr(sink, "read_core", None)
    if callable(read_core):
        return read_core(fs)
    if isinstance(fs.result_json, dict):
        return fs.result_json
    raise RuntimeError("Checkpoint core is unavailable for enrichment replay")


class ExportStage:
    name = "export"
    # FileState fields consumed / populated (see validate_stage_contracts).
    requires = ("paper",)
    produces = ("result_json",)

    def __init__(self) -> None:
        self._last_gc_time = 0.0

    async def _maybe_gc_collect(self) -> None:
        now = time.monotonic()
        if now - self._last_gc_time < _GC_MIN_INTERVAL_SECONDS:
            return
        self._last_gc_time = now
        # gc.collect() holds the GIL for the whole sweep; run it in a thread so
        # the event loop keeps servicing other in-flight requests meanwhile.
        import asyncio

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, gc.collect)

    async def run(self, ctx: PipelineContext) -> None:
        ctx.progress.stage_start(self.name)
        for fs in ctx.alive():
            try:
                sink = getattr(fs, "artifact_sink", None)
                core_hash = getattr(fs, "core_sha256", None)
                enrichment_state = getattr(fs, "enrichment_state", None)
                if sink is not None and core_hash is not None and enrichment_state is None:
                    # Enrichment was not requested (including refs=off). The
                    # checkpoint stage already serialized and materialized the
                    # authoritative core, so do not serialize Paper a second
                    # time merely to discard that result. Consolidation still
                    # runs here so the -o file matches the unsinked export
                    # (which always consolidates, e.g. the
                    # CONSOLIDATE_WITHOUT_ENRICHMENT warning); the change is
                    # re-materialized only when consolidation ran, so the
                    # default consolidate-off run does not rewrite identical
                    # bytes. The sibling core stays immutable. A
                    # consolidate failure keeps the verified core (audit
                    # pipeline-stages-12).
                    if not isinstance(fs.result_json, dict):
                        raise RuntimeError("Verified in-memory core checkpoint is unavailable")
                    try:
                        if await _consolidate_payload(ctx, fs.result_json):
                            sink.materialize(fs, fs.result_json)
                    except Exception as exc:  # noqa: BLE001 - consolidation is optional
                        logger.warning(
                            "Consolidation failed for %s; keeping verified core: %s",
                            fs.path.name,
                            exc,
                            exc_info=True,
                        )
                    continue

                enriched_payload = await build_result_payload(ctx, fs)
                if sink is not None and core_hash is not None and enrichment_state is not None:
                    from bibr.pipeline.artifacts import (
                        EnrichmentSidecar,
                        RunState,
                        enrichment_settings_digest,
                        make_enrichment_sidecar,
                        replay_enrichment_sidecar,
                    )

                    settings_digest = enrichment_settings_digest(ctx)
                    completeness = (
                        "complete"
                        if enrichment_state == RunState.ENRICHMENT_COMPLETE
                        else "partial"
                    )
                    sidecar = make_enrichment_sidecar(
                        enriched_payload,
                        core_sha256=core_hash,
                        settings_digest=settings_digest,
                        completeness=completeness,
                        warnings=tuple(getattr(fs, "enrichment_warnings", ())),
                        detail=getattr(fs, "enrichment_detail", None),
                    )
                    checkpoint = (
                        copy.deepcopy(fs.result_json) if isinstance(fs.result_json, dict) else None
                    )
                    if checkpoint is None:
                        raise RuntimeError("Verified in-memory core checkpoint is unavailable")
                    materialized = False
                    try:
                        sink.write_enrichment(fs, sidecar)
                        durable_sidecar = sink.read_enrichment(fs)
                        if isinstance(durable_sidecar, dict):
                            durable_sidecar = EnrichmentSidecar.from_dict(durable_sidecar)
                        core_payload = _load_checkpoint_core(sink, fs)
                        replayed_payload = replay_enrichment_sidecar(
                            core_payload,
                            durable_sidecar,
                            expected_settings_digest=settings_digest,
                        )
                        # The replayed core's extraction block predates
                        # enrichment, so its timings omit the enrich stage even
                        # though enrichment ran. Carry the enriched run's
                        # timings over so sinked and unsinked exports report the
                        # same provenance (audit pipeline-stages-12).
                        enriched_timings = (enriched_payload.get("extraction") or {}).get("timings")
                        if isinstance(enriched_timings, dict):
                            replayed_payload.setdefault("extraction", {})["timings"] = (
                                copy.deepcopy(enriched_timings)
                            )
                        await _consolidate_payload(ctx, replayed_payload)
                        sink.materialize(fs, replayed_payload)
                        materialized = True
                        fs.result_json = replayed_payload
                        sink.record(fs, enrichment_state)
                    except Exception as exc:  # noqa: BLE001 - enrichment is optional
                        logger.warning(
                            "Enrichment sidecar/replay failed for %s: %s",
                            fs.path.name,
                            exc,
                            exc_info=True,
                        )
                        # Never replace a verified checkpoint with bytes that
                        # just failed replay validation. Atomic materialization
                        # also leaves the prior public checkpoint untouched.
                        fs.result_json = checkpoint
                        if materialized:
                            try:
                                sink.materialize(fs, checkpoint)
                            except Exception:  # noqa: BLE001 - retain original replay failure
                                logger.warning(
                                    "Failed to restore public checkpoint for %s",
                                    fs.path.name,
                                    exc_info=True,
                                )
                        try:
                            sink.record(fs, RunState.ENRICHMENT_PARTIAL, detail=str(exc))
                        except Exception:  # noqa: BLE001 - immutable core remains authoritative
                            logger.warning(
                                "Failed to persist enrichment failure receipt for %s",
                                fs.path.name,
                                exc_info=True,
                            )
                else:
                    fs.result_json = enriched_payload
                    await _consolidate_payload(ctx, fs.result_json)
            except Exception as e:  # noqa: BLE001
                fs.set_error(f"Export failed: {e}", code="export_failed", stage=self.name, exc=e)
                logger.warning("Export failed for %s", fs.path.name, exc_info=True)
                artifact_sink = getattr(fs, "artifact_sink", None)
                if artifact_sink is not None and getattr(fs, "core_sha256", None) is not None:
                    from bibr.pipeline.artifacts import RunState

                    try:
                        artifact_sink.record(fs, RunState.FAILED, detail=str(e))
                    except Exception:  # noqa: BLE001 - preserve originating export failure
                        logger.warning("Failed to persist export failure receipt", exc_info=True)
            finally:
                fs.free_all()
        ctx.progress.stage_end(self.name)

        await self._maybe_gc_collect()
