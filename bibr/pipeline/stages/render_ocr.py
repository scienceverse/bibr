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
- **managed vLLM OCR** (``paddle-vllm``): window of 1 with a one-window
  lookahead. The engine is another process, so while one file waits on it the
  next file is rendered, laid out and has its text layer read. At most two
  files' pages are resident. ``--memory aggressive`` turns the lookahead off.
- **remote OCR** (glm-http / cloud vision): network-bound → window of
  ``OCR_MAX_CONCURRENT_FILES`` so cross-file HTTP concurrency is preserved
  while still capping resident pages.

It is used only by ``LocalPipeline``. The serve pipeline keeps the three
stages separate on purpose: its ``GpuBatcher`` deliberately coalesces pages
across concurrent requests, which per-file windowing would defeat.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import sys
import weakref
from dataclasses import replace
from typing import TYPE_CHECKING

from bibr.ocr.profiles import OcrRuntimeIdentity
from bibr.ocr.registry import TEXT_LAYER_BACKEND
from bibr.pipeline.pipeline import run_stage
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.ocr import (
    CONCURRENT_MANAGED_OCR_BACKENDS,
    OcrStage,
    _is_remote_ocr,
    _should_unload_ocr_after_chunk,
    fail_ocr_targets,
    resolve_ocr_identity,
)
from bibr.utils.heap import trim_between_windows

if TYPE_CHECKING:
    from collections.abc import Callable

    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

#: An event that a caller ready to start its next chunk early sets before running
#: this one (``bibr batch``'s overlapped chunks, :mod:`bibr.batch.runner`). The
#: streaming stage sets it when the chunk's last window has been read, before it
#: waits for the back halves: from then on the chunk renders no more pages, so the
#: next chunk's front half can run while this chunk's back halves finish. Unset
#: (the default), nothing is signalled and chunks run one after another.
FRONT_HALF_DONE: contextvars.ContextVar[asyncio.Event | None] = contextvars.ContextVar(
    "bibr_front_half_done", default=None
)


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
        "doc_layer",
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
        from bibr.pipeline import ocr_cache

        alive = ctx.alive()
        # Files that could actually reach the OCR backend. Native parses
        # (DOCX/JATS/HTML) already carry ``contents`` and cache hits carry
        # ``ocr_regions``; a chunk with neither must not pay — or fail on —
        # OCR startup, mirroring ``LayoutStage``'s ``_needs_ocr`` guard.
        pending = [fs for fs in alive if fs.contents is None and fs.ocr_regions is None]
        cache_on = ocr_cache.is_enabled(ctx.settings)
        # Shared-resource init failures (missing model, unreachable server)
        # must fail per-file rather than crash the whole chunk: ``run_stage``
        # wraps stages in try/*finally* only and ``process_chunk`` has no
        # handler, so an escaping exception would abort every file in the
        # chunk — including ones that never needed OCR. ``OcrStage`` guards
        # the identical call for exactly this reason.
        try:
            resolution = await resolve_ocr_identity(
                ctx,
                needs_backend=bool(pending) and cache_on,
                fail_targets=pending,
                stage=self.name,
                strict=True,
            )
        except Exception as e:  # noqa: BLE001 - shared-resource boundary
            ctx.signals.ocr_init_error = e
            self._fail_ocr_init(pending, e)
            return []
        identity = resolution.identity
        if identity is None:
            # resolve_ocr_identity returns None only after a startup failure
            # (the targets were already failed and the chunk marker recorded),
            # so there is no bundle to look up here.
            return []
        if cache_on:
            from bibr.ocr.registry import TEXT_LAYER_BACKEND
            from bibr.ocr.text_layer import note_text_layer_fallback, text_layer_fallback_reason

            for fs in pending:
                if ocr_cache.load_bundle(fs, ctx.config, identity, ctx.settings):
                    fs.free_pre_ocr()
                    if identity.backend == TEXT_LAYER_BACKEND:
                        # Read through OCR_FALLBACK_TO_TEXT_LAYER=auto: say so,
                        # as a fresh read does (the cache entry does not).
                        note_text_layer_fallback(fs, text_layer_fallback_reason(ctx.resources))
            # Cached output must satisfy this run's quality policy too. In
            # particular, failed pages cannot become invisible on a cache hit.
            OcrStage._check_ocr_success(ctx)
        return [fs for fs in pending if fs.contents is None and fs.ocr_regions is None]

    def _fail_ocr_init(self, targets: list, exc: BaseException, *, log: bool = True) -> None:
        """Mark the OCR-needing files failed instead of aborting the chunk."""
        fail_ocr_targets(targets, exc, stage=self.name, log=log)

    @staticmethod
    def _window_size(ctx: PipelineContext) -> int:
        """How many files' page images one window holds."""
        if _is_remote_ocr(ctx.config):
            return max(1, ctx.settings.ocr.max_concurrent_files)
        return 1

    @staticmethod
    def _may_prepare_ahead(ctx: PipelineContext) -> bool:
        """Whether the next window may be prepared while this one is in OCR.

        Only when the OCR engine is a managed vLLM server: it is another
        process, so this process is idle while a file waits on it. An
        in-process engine shares this process's CPU and memory, and remote OCR
        already overlaps files within a window. ``aggressive`` memory mode
        keeps the strict one-window bound. The automatic chain publishes its
        runtime only once an engine has started, so its first window runs
        without lookahead.
        """
        cfg = ctx.config
        if _is_remote_ocr(cfg) or cfg.memory_mode == "aggressive":
            return False
        if ctx.signals.ocr_init_error is not None:
            return False
        identity = ctx.scratch.get("ocr_runtime_identity")
        return (
            isinstance(identity, OcrRuntimeIdentity)
            and identity.backend in CONCURRENT_MANAGED_OCR_BACKENDS
        )

    async def _prepare(self, ctx: PipelineContext) -> None:
        """Everything a window needs before OCR: render, layout, text layer."""
        await self._layout.run(ctx)
        await self._native.run(ctx)

    async def _settle_prepare_ahead(self, task: asyncio.Task, ahead_ctx: PipelineContext) -> None:
        """Stop an unconsumed lookahead and release its files' pages.

        Reached only when the window loop is unwinding (an inner stage raised
        or the stage was cancelled): no prepare task may outlive ``run()``.
        """
        if not task.done():
            task.cancel()
        settlement = asyncio.gather(task, return_exceptions=True)
        try:
            try:
                (result,) = await asyncio.shield(settlement)
            except asyncio.CancelledError:
                task.cancel()
                try:
                    await asyncio.shield(settlement)
                except asyncio.CancelledError:
                    pass
                raise
            if isinstance(result, Exception):
                # The loop is already unwinding with its own error, which is
                # the one reported; keep this one findable.
                logger.debug("Discarded prepare-ahead failed: %r", result)
        finally:
            for fs in ahead_ctx.file_states:
                fs.free_pre_ocr()

    async def _run_windows(
        self,
        ctx: PipelineContext,
        pending: list,
        on_window_done: Callable[[list], None] | None = None,
    ) -> None:
        """Drive Layout → NativeText → OCR over ``pending``, one window at a time.

        ``on_window_done`` is called with each window's files once their OCR is
        finished and their pages are freed.
        """
        window = self._window_size(ctx)
        # Snapshot alive files up front: files that error inside a window drop
        # out of later inner stages via ``ctx.alive()`` but must still have
        # their page buffers freed, so we iterate the original grouping.
        groups = [pending[start : start + window] for start in range(0, len(pending), window)]
        # The next window's prepare, running under this window's OCR.
        ahead: tuple[asyncio.Task, PipelineContext] | None = None
        try:
            for i, group in enumerate(groups):
                sub = replace(ctx, file_states=group)
                if ahead is None:
                    await self._prepare(sub)
                else:
                    task, ahead_ctx = ahead
                    await task
                    ahead = None
                    # The lookahead ran on its own copy of the signals, so the
                    # previous window's OCR kept reading its own. They become
                    # this window's now.
                    ctx.signals.any_needs_ocr = ahead_ctx.signals.any_needs_ocr
                    ctx.signals.preloading_ocr = ahead_ctx.signals.preloading_ocr
                if i + 1 < len(groups) and self._may_prepare_ahead(ctx):
                    # Silent: its stage lines would open under the OCR line of
                    # the window on display. Per-file timings still land in
                    # ``fs.stage_times``.
                    ahead_ctx = replace(
                        ctx,
                        file_states=groups[i + 1],
                        progress=NullProgress(),
                        signals=replace(ctx.signals),
                    )
                    ahead = (asyncio.create_task(self._prepare(ahead_ctx)), ahead_ctx)
                await self._ocr.run(sub)
                # Reclaim this window's page images (and pdf_bytes / layout_results)
                # before the next window's OCR — the memory cap. With a
                # lookahead, the next window's pages are the only ones left.
                for fs in group:
                    fs.free_pre_ocr()
                if on_window_done is not None:
                    on_window_done(group)
                # ...and, with PIPELINE_HEAP_TRIM=window, hand the freed pages back
                # to the OS before the next window (the last window's go at the
                # chunk-end trim).
                if i + 1 < len(groups):
                    await trim_between_windows(ctx.settings)
        finally:
            if ahead is not None:
                await self._settle_prepare_ahead(*ahead)

    async def run(self, ctx: PipelineContext) -> None:
        cfg = ctx.config

        pending = await self._probe_cache(ctx)

        # Let each window's OcrStage skip the per-chunk engine teardown; we do
        # it once below so a local LLM backend still reclaims VRAM without
        # reloading the OCR engine per window.
        ctx.signals.defer_ocr_teardown = True

        await self._run_windows(ctx, pending)

        # Single per-chunk teardown, matching what OcrStage would have done once
        # had it processed the whole chunk in one pass.
        if _should_unload_ocr_after_chunk(
            cfg.memory_mode, cfg.llm_backend, ctx.settings, cfg.ocr_backend
        ):
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

    # Max windows running the back half concurrently, on one event loop: the
    # chunks bibr batch overlaps share it. OCR remains the throughput
    # governor — the point is overlap, not fan-out — and each concurrent
    # PostParseStage.run multiplies its internal ``max_concurrent_post_parse``
    # semaphore, so keep this small and fixed.
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
        # One back-half semaphore per event loop, shared by every chunk this stage
        # runs on it: chunks that overlap (FRONT_HALF_DONE) share the cap rather
        # than each bringing its own.
        self._backhalf_sems: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = weakref.WeakKeyDictionary()
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

        window = self._window_size(ctx)

        # A run with exactly one back-half group (one pending window and no
        # bypass, or bypass-only) has nothing to interleave with, so that group
        # reports real per-stage progress; more than one group overlaps and
        # falls back to NullProgress. Computed up front so the bypass spawn (the
        # first group) already knows whether it is the sole group.
        pending_windows = (len(pending) + window - 1) // window
        single_group = pending_windows + (1 if bypass else 0) <= 1

        sem = self._backhalf_semaphore(ctx)
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
            # Same windows, memory cap and lookahead as the non-streaming
            # path; each window's back half starts once its pages are freed.
            await self._run_windows(ctx, pending, on_window_done=_spawn)
            self._signal_front_half_done(ctx)
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
        if _should_unload_ocr_after_chunk(
            cfg.memory_mode, cfg.llm_backend, ctx.settings, cfg.ocr_backend
        ):
            await ctx.resources.shutdown_ocr()

    def _backhalf_semaphore(self, ctx: PipelineContext | None = None) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        sem = self._backhalf_sems.get(loop)
        if sem is None:
            cap = self._MAX_CONCURRENT_BACKHALF_WINDOWS
            # Fast / text_layer mode frees page images immediately after layout +
            # text layer and has no GPU/remote OCR bottleneck, so let back-half
            # windows scale up to max_concurrent_post_parse instead of capping at 2.
            if ctx is not None:
                identity = ctx.scratch.get("ocr_runtime_identity")
                is_text_layer = ctx.config.ocr_backend == TEXT_LAYER_BACKEND or (
                    isinstance(identity, OcrRuntimeIdentity)
                    and identity.backend == TEXT_LAYER_BACKEND
                )
                if is_text_layer:
                    cap = max(cap, int(ctx.settings.pipeline.max_concurrent_post_parse))
            sem = asyncio.Semaphore(cap)
            self._backhalf_sems[loop] = sem
        return sem

    @staticmethod
    def _signal_front_half_done(ctx: PipelineContext) -> None:
        """Tell a waiting caller that this chunk renders no more pages (FRONT_HALF_DONE)."""
        event = FRONT_HALF_DONE.get()
        if event is None:
            return
        cfg = ctx.config
        if _should_unload_ocr_after_chunk(
            cfg.memory_mode, cfg.llm_backend, ctx.settings, cfg.ocr_backend
        ):
            # This chunk shuts the OCR engine down when it ends, which would pull
            # it from under the next chunk's windows: no early start.
            return
        event.set()

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
