"""InterleavedRenderOcrStage — local-only render→OCR interleaving.

The three page-image stages — ``LayoutStage`` (renders PDF pages to
full-resolution PIL images), ``NativeTextStage``, and ``OcrStage`` — run
stage-at-a-time across a whole chunk in the shared driver. That means every
file's page images (~11.6 MB/page at 200 DPI) stay resident from the start of
layout through the end of OCR: peak page-image RAM scales with the *chunk
size*, not with one document. On a 16 GB unified-memory Mac (sharing RAM with
the OCR/LLM models) an 8-file chunk of long PDFs spills into swap.

This stage fuses those three into one and drives them per **window** of files,
freeing each window's page images before the next window renders. Page-image
RAM is then bounded by the window, independent of chunk size:

- **local OCR** (GLM and generic local engines): sequential engine → window of 1,
  so at most one file's pages are ever resident.
- **remote OCR** (glm-http / cloud vision): network-bound → window of
  ``OCR_MAX_CONCURRENT_FILES`` so cross-file HTTP concurrency is preserved
  while still capping resident pages.

It is used only by ``LocalPipeline``. The serve pipeline keeps the three
stages separate on purpose: its ``GpuBatcher`` deliberately coalesces pages
across concurrent requests, which per-file windowing would defeat.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from bibr.pipeline.pipeline import run_stage
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.ocr import (
    OcrStage,
    _is_remote_ocr,
    _should_unload_ocr_after_chunk,
)

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class InterleavedRenderOcrStage:
    """Runs Layout → NativeText → OCR per file-window, freeing pages between."""

    name = "render_ocr"
    # Consumes ``pdf_bytes`` (from validate); the three inner stages resolve
    # ``layout_results``/``page_images``/``page_indices`` among themselves.
    requires = ("pdf_bytes",)
    # The union of the inner stages' outputs, so downstream parse/extract still
    # see their required fields under static contract validation. ``page_images``
    # etc. are produced then freed per window; no later stage consumes them.
    produces = (
        "page_images",
        "page_indices",
        "layout_results",
        "ref_line_geometry",
        "ref_page_lines",
        "pdf_uri_links",
        "native_metadata",
        "pdf_outline",
        "ocr_regions",
    )

    def __init__(self, layout=None, native_text=None, ocr=None):
        # Injectable for testing; default to the real stages.
        if layout is None or native_text is None or ocr is None:
            from bibr.pipeline.stages.layout import LayoutStage
            from bibr.pipeline.stages.native_text import NativeTextStage

            layout = layout or LayoutStage()
            native_text = native_text or NativeTextStage()
            ocr = ocr or OcrStage()
        self._layout = layout
        self._native = native_text
        self._ocr = ocr

    async def _probe_cache(self, ctx: PipelineContext) -> list:
        """Complete-bundle cache probe; returns the files still pending OCR.

        Probes the complete artifact bundle before rendering. OcrStage also
        has a later regions-only probe for standalone use, but only this
        fused local stage can avoid layout/native work as well.
        """
        from bibr.ocr.profiles import OcrRuntimeIdentity, resolve_ocr_runtime_identity
        from bibr.pipeline import ocr_cache

        alive = ctx.alive()
        identity = ctx.scratch.get("ocr_runtime_identity")
        requested_backend = ctx.config.ocr_backend or ctx.settings.ocr.backend
        # Files that could actually reach the OCR backend. Native parses
        # (DOCX/JATS/HTML) already carry ``contents`` and cache hits carry
        # ``ocr_regions``; a chunk with neither must not pay — or fail on —
        # OCR startup, mirroring ``LayoutStage``'s ``_needs_ocr`` guard.
        pending = [fs for fs in alive if fs.contents is None and fs.ocr_regions is None]
        runtime_identity = getattr(ctx.resources, "ocr_runtime_identity", None)
        runtime_client = getattr(ctx.resources, "ocr", None)
        runtime_loaded = bool(getattr(runtime_client, "loaded", False))
        if (
            requested_backend == "paddle"
            and isinstance(runtime_identity, OcrRuntimeIdentity)
            and not runtime_loaded
        ):
            ctx.resources.ocr_runtime_identity = None
            runtime_identity = None
            identity = None
            ctx.scratch.pop("ocr_runtime_identity", None)
        if (
            requested_backend == "paddle"
            and identity is None
            and isinstance(runtime_identity, OcrRuntimeIdentity)
            and runtime_loaded
        ):
            identity = runtime_identity
            ctx.scratch["ocr_runtime_identity"] = identity
        if requested_backend == "paddle" and identity is None and pending:
            if not ocr_cache.is_enabled(ctx.settings):
                # With no bundle to look up nothing here needs the concrete
                # runtime identity. Leave the automatic chain unresolved so
                # OcrStage starts it after native text is known — and only if
                # a region still needs the backend.
                return pending
            # A hard init failure in an earlier window of this chunk must not
            # re-run the slow, doomed engine constructor here.
            prior_error = ctx.signals.ocr_init_error
            if prior_error is not None:
                self._fail_ocr_init(pending, prior_error, log=False)
                return []
            # The automatic chain must choose a concrete runtime before this
            # complete-bundle lookup; its selected identity is cache evidence.
            #
            # Shared-resource init failures (missing model, unreachable server)
            # must fail per-file rather than crash the whole chunk: ``run_stage``
            # wraps stages in try/*finally* only and ``process_chunk`` has no
            # handler, so an escaping exception would abort every file in the
            # chunk — including ones that never needed OCR. ``OcrStage`` guards
            # the identical call for exactly this reason.
            try:
                await ctx.resources.await_ocr()
                identity = ctx.resources.ocr_runtime_identity
                if identity is None:
                    injected = getattr(ctx.resources, "_ocr", None)
                    if not getattr(injected, "loaded", False):
                        raise RuntimeError(
                            "OCR startup completed without a concrete runtime identity"
                        )
                    # Tests and library integrations can inject an already-ready
                    # client before the managed startup path runs. Such a client
                    # has no selected candidate to record, so retain a stable
                    # static identity for this compatibility seam only. A real
                    # automatic startup still must provide its concrete identity.
                    identity = resolve_ocr_runtime_identity(ctx.config, ctx.settings)
            except Exception as e:  # noqa: BLE001 - shared-resource boundary
                ctx.signals.ocr_init_error = e
                self._fail_ocr_init(pending, e)
                return []
            ctx.scratch["ocr_runtime_identity"] = identity
        if identity is None:
            identity = resolve_ocr_runtime_identity(ctx.config, ctx.settings)
            ctx.scratch["ocr_runtime_identity"] = identity
        if ocr_cache.is_enabled(ctx.settings):
            for fs in pending:
                if ocr_cache.load_bundle(fs, ctx.config, identity, ctx.settings):
                    fs.free_pre_ocr()
            # Cached output must satisfy this run's quality policy too. In
            # particular, failed pages cannot become invisible on a cache hit.
            OcrStage._check_ocr_success(ctx)
        return [fs for fs in pending if fs.contents is None and fs.ocr_regions is None]

    def _fail_ocr_init(self, targets: list, exc: BaseException, *, log: bool = True) -> None:
        """Mark the OCR-needing files failed instead of aborting the chunk."""
        for fs in targets:
            if fs.error is None:
                fs.set_error(
                    f"OCR backend init failed: {exc}",
                    code="ocr_failed",
                    stage=self.name,
                    exc=exc,
                )
        if log:
            logger.warning("OCR backend init failed", exc_info=exc)

    async def run(self, ctx: PipelineContext) -> None:
        cfg = ctx.config

        pending = await self._probe_cache(ctx)

        # Window size = how many files' page images may be resident at once.
        window = ctx.settings.ocr.max_concurrent_files if _is_remote_ocr(cfg) else 1
        window = max(1, window)

        # Let each window's OcrStage skip the per-chunk engine teardown; we do
        # it once below so a local LLM backend still reclaims VRAM without
        # reloading the OCR engine per window.
        ctx.signals.defer_ocr_teardown = True

        # Snapshot alive files up front: files that error inside a window drop
        # out of later inner stages via ``ctx.alive()`` but must still have
        # their page buffers freed, so we iterate the original grouping.
        for start in range(0, len(pending), window):
            group = pending[start : start + window]
            sub = replace(ctx, file_states=group)
            await self._layout.run(sub)
            await self._native.run(sub)
            await self._ocr.run(sub)
            # Reclaim this window's page images (and pdf_bytes / layout_results)
            # before the next window renders — the memory cap.
            for fs in group:
                fs.free_pre_ocr()

        # Single per-chunk teardown, matching what OcrStage would have done once
        # had it processed the whole chunk in one pass.
        if _should_unload_ocr_after_chunk(cfg.memory_mode, cfg.llm_backend, ctx.settings):
            await ctx.resources.shutdown_ocr()


class StreamingRenderOcrStage(InterleavedRenderOcrStage):
    """Streams each OCR window's back half under subsequent windows' OCR.

    Used by ``LocalPipeline`` only when the LLM is remote (``llm_backend ==
    "cloud"``), memory mode is not ``aggressive``, and
    ``Settings.pipeline.stream_backhalf`` is on. With a cloud LLM there is no
    OCR→LLM VRAM handoff, so a window's back half (parse → extract → enrich →
    export) can start the moment its OCR completes — the LLM/Crossref tail
    then hides under the next windows' OCR instead of serializing after it.

    The composite owns the back-half stage instances; the driver sees one
    stage whose ``produces`` is the union of everything the fused stages
    populate, so ``validate_stage_contracts`` still passes with this stage
    terminal. When more than one back-half group runs, each gets a
    ``NullProgress`` so overlapping per-window stage transitions never
    interleave in the terminal. A run with a single back-half group has
    nothing to overlap, so that group keeps the real ``ctx.progress`` — the
    dominant single-file ``bibr chew`` case then shows live per-stage
    feedback through the LLM+Crossref tail instead of going silent, and the
    CLI's advertised ``parse/extract/enrich/export`` stages actually fire.
    (Per-file timings still land in ``fs.stage_times`` via the inner stages
    either way.)
    """

    name = "render_ocr_stream"
    requires = ("pdf_bytes",)
    produces = InterleavedRenderOcrStage.produces + (
        "contents",
        "paper",
        "doi_selection",
        "result_json",
    )

    # Max windows running the back half concurrently. OCR remains the
    # throughput governor — the point is overlap, not fan-out — and each
    # concurrent PostParseStage.run multiplies its internal
    # ``max_concurrent_post_parse`` semaphore, so keep this small and fixed.
    _MAX_CONCURRENT_BACKHALF_WINDOWS = 2

    def __init__(
        self,
        *,
        parse,
        post_parse,
        enrich,
        export,
        identity=None,
        checkpoint=None,
        layout=None,
        native_text=None,
        ocr=None,
    ):
        super().__init__(layout=layout, native_text=native_text, ocr=ocr)
        self._parse = parse
        self._post_parse = post_parse
        self._identity = identity
        self._checkpoint = checkpoint
        self._enrich = enrich
        self._export = export

    async def run(self, ctx: PipelineContext) -> None:
        cfg = ctx.config

        pending = await self._probe_cache(ctx)
        # Native parses (DOCX/JATS/HTML: ``fs.contents``) and cache hits
        # (``fs.ocr_regions``) need no rendering or OCR — their back half
        # starts immediately, before any window renders.
        bypass = [fs for fs in ctx.alive() if fs.contents is not None or fs.ocr_regions is not None]

        window = ctx.settings.ocr.max_concurrent_files if _is_remote_ocr(cfg) else 1
        window = max(1, window)

        # A run with exactly one back-half group (one pending window and no
        # bypass, or bypass-only) has nothing to interleave with, so that group
        # reports real per-stage progress; more than one group overlaps and
        # falls back to NullProgress. Computed up front so the bypass spawn (the
        # first group) already knows whether it is the sole group.
        pending_windows = (len(pending) + window - 1) // window
        single_group = pending_windows + (1 if bypass else 0) <= 1

        sem = asyncio.Semaphore(self._MAX_CONCURRENT_BACKHALF_WINDOWS)
        tasks: list[tuple[asyncio.Task, list]] = []

        def _spawn(group: list) -> None:
            if group:
                task = asyncio.create_task(self._run_backhalf(ctx, group, sem, single_group))
                tasks.append((task, group))

        ctx.signals.defer_ocr_teardown = True

        try:
            # Spawn the bypass back half inside the try so its task is settled
            # by the finally below even if a later front-half stage raises —
            # the "no spawned task outlives run()" invariant stays structural.
            _spawn(bypass)
            for start in range(0, len(pending), window):
                group = pending[start : start + window]
                sub = replace(ctx, file_states=group)
                await self._layout.run(sub)
                await self._native.run(sub)
                await self._ocr.run(sub)
                # Reclaim this window's page images before the next renders —
                # same memory cap as the non-streaming path.
                for fs in group:
                    fs.free_pre_ocr()
                _spawn(group)
        finally:
            # Settle every spawned back half before the stage returns — no
            # window task may outlive run(), even when a front-half stage
            # raises mid-loop. Inner stages already map expected per-file
            # failures to fs.set_error; anything surfacing here is unexpected
            # and is pinned on the window's files rather than lost.
            active_exception = sys.exception()
            if active_exception is not None or asyncio.current_task().cancelling():
                for task, _ in tasks:
                    if not task.done():
                        task.cancel()
            settlement = asyncio.gather(*(t for t, _ in tasks), return_exceptions=True)
            try:
                results = await asyncio.shield(settlement)
            except asyncio.CancelledError:
                # Cancellation arriving while the front half is otherwise
                # healthy must also tear down every child. Shield the bounded
                # settlement so no back-half task leaks beyond run().
                for task, _ in tasks:
                    if not task.done():
                        task.cancel()
                try:
                    await asyncio.shield(settlement)
                except asyncio.CancelledError:
                    pass
                raise
            for (_, group), result in zip(tasks, results, strict=True):
                # A CancelledError here is this task being torn down, not a
                # per-file failure — re-raise it rather than recording it.
                if isinstance(result, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    if active_exception is None:
                        raise result
                    continue
                if isinstance(result, BaseException):
                    for fs in group:
                        if fs.error is None:
                            fs.set_error(
                                f"Streaming back-half failed: {result}",
                                code="backhalf_failed",
                                stage=self.name,
                                exc=result,
                            )
                    for fs in group:
                        if fs.error is not None and fs.result_json is None:
                            fs.free_all()

        # No-op for the gated config (cloud LLM + balanced/keep_all never
        # unloads) but kept for parity with the non-streaming path.
        if _should_unload_ocr_after_chunk(cfg.memory_mode, cfg.llm_backend, ctx.settings):
            await ctx.resources.shutdown_ocr()

    async def _run_backhalf(
        self, ctx: PipelineContext, group: list, sem, live_progress: bool = False
    ) -> None:
        """Run parse → extract → enrich → export over one window's files.

        ``live_progress`` is True only for a run's single back-half group: with
        nothing to interleave, it reports real per-stage progress; otherwise it
        gets a ``NullProgress`` so concurrent windows' stage lines never garble
        the terminal.
        """
        async with sem:
            progress = ctx.progress if live_progress else NullProgress()
            sub = replace(ctx, file_states=group, progress=progress)
            stages = [self._parse, self._post_parse]
            if self._identity is not None:
                stages.append(self._identity)
            if self._checkpoint is not None:
                stages.append(self._checkpoint)
            stages.extend((self._enrich, self._export))
            for stage in stages:
                await run_stage(sub, stage)
                if not sub.alive():
                    break
