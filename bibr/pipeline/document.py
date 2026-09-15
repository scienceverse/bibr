"""Parse a document once, then extract each independently scoped article.

This opt-in local path keeps document identity distinct from article identity.
The existing single-paper stage plan and wire format remain separate contracts.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from bibr.exceptions import ProcessingError
from bibr.export.document_models import (
    DocumentDiagnosticsExport,
    DocumentExport,
    DocumentRecordExport,
)
from bibr.export.models import SourceExport
from bibr.extract.document_records import refine_document_records
from bibr.extract.document_scope import DocumentRecordScope, scope_document_records
from bibr.extract.front_matter import resolve_front_matter
from bibr.pipeline.context import PipelineContext
from bibr.pipeline.enrich_prefetch import discard_prefetches
from bibr.pipeline.pipeline import run_stage
from bibr.pipeline.plans import build_stage_plan
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.export import build_result_payload
from bibr.pipeline.stages.post_parse import _classify_sections
from bibr.pipeline.state import FileState

if TYPE_CHECKING:
    from bibr.pipeline.context import RunConfig
    from bibr.pipeline.pipeline import Pipeline
    from bibr.pipeline.progress import ProgressTracker

logger = logging.getLogger(__name__)


def _source(state: FileState) -> SourceExport:
    suffix = state.path.suffix.lower().lstrip(".")
    return SourceExport(
        file_name=state.path.name,
        file_hash=state.file_hash or "",
        input_format="html" if suffix == "htm" else suffix,
    )


def _record(scope: DocumentRecordScope, **outcome) -> DocumentRecordExport:
    return DocumentRecordExport(
        record_id=scope.record_id,
        source_text_ids=list(scope.source_text_ids),
        source_section_ids=list(scope.source_section_ids),
        pages=list(scope.pages),
        **outcome,
    )


async def _blocked_record(scope, original, ctx, state, *, status, reason_flags, error=None):
    """Retain already extracted owned fields without continuing failed work."""
    partial = None
    reasons = list(reason_flags)
    if state.result_json is not None or state.paper is not None:
        try:
            # Reference enrichment may have been prefetched during extraction.
            # Stop it before serializing the current partial extraction.
            await discard_prefetches([state])
            # Export may already have freed the Paper. Its validated scoped
            # payload is still the retained source of partial fields.
            payload = (
                state.result_json
                if state.result_json is not None
                else await build_result_payload(ctx, state)
            )
            candidate = _record(scope, status=status, paper=None, partial_paper=payload)
            if candidate.partial_paper is None or candidate.partial_paper.source != _source(
                original
            ):
                raise ValueError("Partial record has inconsistent source identity")
            partial = candidate.partial_paper
        except Exception:  # noqa: BLE001 - partial serialization must not lose the inventory
            logger.warning(
                "Partial record %s could not be serialized safely", scope.record_id, exc_info=True
            )
            reasons.append("partial_paper_serialization_failed")
    return _record(
        scope,
        status=status,
        paper=None,
        partial_paper=partial,
        reason_flags=list(dict.fromkeys(reasons)),
        error=error,
    )


async def _extract_record(scope, original, ctx, stages, *, single_record):
    if scope.contents is None:
        scoping_failed = "record_scoping_failed" in scope.reason_flags
        return _record(
            scope,
            status="failed" if scoping_failed else "unresolved",
            paper=None,
            reason_flags=list(scope.reason_flags),
            error="Document record scoping failed." if scoping_failed else None,
        )

    state = FileState(
        path=original.path,
        file_hash=original.file_hash,
        content_sha256=original.content_sha256,
        contents=scope.contents,
        prepared_front_matter=scope.contents.front_matter_resolution,
        native_metadata=deepcopy(original.native_metadata) if single_record else None,
        warnings=list(original.warnings),
    )
    record_ctx = PipelineContext(
        file_states=[state],
        progress=ctx.progress,
        resources=ctx.resources,
        config=ctx.config,
        settings=ctx.settings,
        # Shared OCR identity is factual; shared document timings must not be
        # charged to every article. Other scratch entries are document-scoped.
        scratch={key: ctx.scratch[key] for key in ("ocr_runtime_identity",) if key in ctx.scratch},
    )
    stage_name = "extract"
    try:
        for stage in stages:
            stage_name = stage.name
            await run_stage(record_ctx, stage)
            blockers = [
                issue.code
                for issue in (state.paper.validation_issues if state.paper is not None else [])
                if issue.blocking
            ]
            if state.result_json is not None:
                blockers.extend(
                    issue["code"]
                    for issue in (state.result_json.get("validation") or {}).get("issues", [])
                    if issue.get("blocking")
                )
            if state.error is not None:
                reasons = [
                    *scope.reason_flags,
                    state.error_code or "record_extraction_failed",
                    *blockers,
                ]
                # Stage exceptions may contain prompts or source text.
                error = f"Record extraction failed during {stage_name}."
                if blockers:
                    return await _blocked_record(
                        scope,
                        original,
                        record_ctx,
                        state,
                        status="failed",
                        reason_flags=reasons,
                        error=error,
                    )
                return _record(
                    scope, status="failed", paper=None, reason_flags=reasons, error=error
                )
            if blockers:
                return await _blocked_record(
                    scope,
                    original,
                    record_ctx,
                    state,
                    status="unresolved",
                    reason_flags=list(dict.fromkeys([*scope.reason_flags, *blockers])),
                )
        if state.result_json is None:
            raise ProcessingError("Record produced no paper export")
        result = _record(
            scope,
            status="extracted",
            paper=state.result_json,
            reason_flags=list(scope.reason_flags),
        )
        if result.paper is None or result.paper.source != _source(original):
            raise ValueError("Extracted record has inconsistent source identity")
        return result
    except Exception:  # noqa: BLE001 - retain this candidate and continue other records
        logger.warning(
            "Document record %s failed during %s", scope.record_id, stage_name, exc_info=True
        )
        return _record(
            scope,
            status="failed",
            paper=None,
            reason_flags=[*scope.reason_flags, "record_extraction_failed"],
            error=f"Record extraction failed during {stage_name}.",
        )
    finally:
        await discard_prefetches([state])
        state.free_all()


async def process_document(
    pipeline: Pipeline,
    path: str | Path,
    *,
    progress: ProgressTracker | None = None,
    content: bytes | None = None,
    content_hash: str | None = None,
    config: RunConfig | None = None,
) -> dict:
    """Return all detected article outcomes from one local input document.

    Input/OCR/detection failures before an article inventory exists raise the
    usual processing exception. Once detected, every candidate survives in the
    output, even when its boundary or extraction cannot be resolved.
    """
    from bibr.pipeline.enricher import CrossrefEnricher

    effective = config if config is not None else pipeline._config
    settings = pipeline.settings
    refs_off = (effective.ref_parse_strategy or settings.REF_PARSE_STRATEGY or "").lower() == "off"
    enrichers = []
    if effective.enrichment_enabled(settings) and not refs_off:
        enrichers.append(CrossrefEnricher(settings=settings))
    stages = build_stage_plan(mode="local", stream_backhalf=False, enrichers=enrichers)
    split = next(index for index, stage in enumerate(stages) if stage.name == "extract")
    original = FileState(path=Path(path), pdf_bytes=content, content_sha256=content_hash)
    ctx = PipelineContext(
        file_states=[original],
        progress=progress if progress is not None else NullProgress(),
        resources=pipeline._resources,
        config=effective,
        settings=settings,
    )
    try:
        for stage in stages[:split]:
            await run_stage(ctx, stage)
            if original.error is not None:
                if isinstance(original.original_error, Exception):
                    raise original.original_error
                raise ProcessingError(
                    original.error,
                    error_code=original.error_code,
                    failed_stage=original.failed_stage,
                )
        contents = original.contents
        if contents is None or not original.content_sha256 or not original.file_hash:
            raise ProcessingError("Document parsing produced no contents or source identity")
        await _classify_sections(
            contents,
            contents.layout_hints or None,
            effective.no_llm,
            None if effective.no_llm else ctx.resources.llm_client,
            classifier_resources=ctx.resources.classifiers,
            settings=settings,
        )
        resolution, _issues = resolve_front_matter(
            contents, target_required=True, settings=settings
        )
        resolution = refine_document_records(resolution, contents=contents)
        try:
            scopes = scope_document_records(contents, resolution)
            if [scope.record_id for scope in scopes] != [
                block.block_id for block in resolution.blocks
            ]:
                raise ValueError("Scoped record inventory differs from detected inventory")
        except Exception:  # noqa: BLE001 - detection already established the required inventory
            logger.warning("Document record scoping failed", exc_info=True)
            scopes = tuple(
                DocumentRecordScope(
                    record_id=block.block_id,
                    contents=None,
                    source_text_ids=(),
                    source_section_ids=(),
                    pages=block.pages,
                    reason_flags=("record_scoping_failed",),
                )
                for block in resolution.blocks
            )
        records = [
            await _extract_record(
                scope, original, ctx, stages[split:], single_record=len(scopes) == 1
            )
            for scope in scopes
        ]
        extracted = sum(record.status == "extracted" for record in records)
        status: Literal["complete", "partial", "unresolved"] = (
            "complete"
            if records and extracted == len(records)
            else "partial"
            if extracted
            else "unresolved"
        )
        reasons = list(dict.fromkeys(["document_record_detection", *resolution.reason_flags]))
        assigned_text_ids = {text_id for scope in scopes for text_id in scope.source_text_ids}
        unassigned_text_ids = [
            row.text_id for row in contents.sentences if row.text_id not in assigned_text_ids
        ]
        if unassigned_text_ids:
            reasons.append("unassigned_document_text")
        if not records:
            reasons.append("no_article_records_detected")
        if effective.start_page is not None or effective.end_page is not None:
            reasons.append("page_range_requested")
        return DocumentExport(
            document_schema_version="1.0",
            document_id=f"sha256:{original.content_sha256}",
            source=_source(original),
            records=records,
            status=status,
            diagnostics=DocumentDiagnosticsExport(
                detected_record_ids=[block.block_id for block in resolution.blocks],
                reason_flags=reasons,
                unassigned_source_text_ids=unassigned_text_ids,
            ),
        ).model_dump(mode="json", by_alias=True)
    finally:
        original.free_all()
