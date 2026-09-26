"""OcrStage — runs per-region OCR on layout regions."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bibr.exceptions import UpstreamServiceError
from bibr.layout_utils import LABEL_TASK_MAPPING
from bibr.ocr.image_utils import pil_to_bytes
from bibr.ocr.normalization import normalize_ocr_output
from bibr.ocr.otsl import check_otsl_completeness
from bibr.ocr.profiles import (
    GLM_PROFILE,
    OcrProfile,
    OcrRuntimeIdentity,
    OcrTask,
    resolve_ocr_profile,
    resolve_ocr_runtime_identity,
)
from bibr.ocr.ref_patterns import alnum_key, alnum_text_covered
from bibr.ocr.types import OcrRegionResult
from bibr.processing_warnings import ProcessingWarning, WarningCode
from bibr.utils.redact import describe_error, redact_urls
from bibr.utils.semaphore import DualSemaphore as _DualSemaphore
from bibr.utils.text import OCR_CORRUPTION_MIN_CHARS, ocr_corruption_count
from bibr.utils.transient import is_service_outage

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

# OCR backends whose inference runs off-process (HTTP server or cloud vision
# API). For these the local render still happens here, but OCR itself is
# network-bound, so files can be processed concurrently. Local engines
# (GLM and generic local engines) run sequentially. ``cfg.ocr_url`` also forces the
# remote path regardless of backend name.
REMOTE_OCR_BACKENDS = frozenset(
    {"glm-http", "paddle-http", "serve-http", "gemini", "openai", "anthropic"}
)

# Shared GPU execution is serialized by the runtime; keep this concurrency default aligned with
# the settings model.
CONCURRENT_MANAGED_OCR_BACKENDS = frozenset({"paddle-vllm"})


def _ocr_server_gone(exc: BaseException) -> bool:
    """Did an OCR request fail because the server is gone, after the
    transport's retries: a refused or dropped connection, an open breaker?

    A 429/502/503 answer is left out. The server answered, so it is up but
    busy, and after about 1.5 s of retries one busy answer must not cost the
    whole file: the region ships blank with a warning, as it did before, and
    ``ocr_mostly_failed`` still catches a server that answers nothing else.
    """
    return is_service_outage(exc, http_status=False)


def _local_region_limit(settings: GlobalSettings, backend: str) -> int:
    """Concurrent OCR regions allowed on the sequential-file path.

    ``concurrent_regions_per_file`` exists to stop one file starving others,
    which only means something when files overlap. On this path they don't, so
    the real ceiling is the server-wide ``max_concurrent_regions`` — which this
    path previously ignored entirely, leaving a managed continuous-batching
    server under-subscribed (PaddleOCR-VL's vLLM is launched with
    ``--max-num-seqs 12`` against a client cap of 6).
    """
    ocr = settings.ocr
    if backend in CONCURRENT_MANAGED_OCR_BACKENDS:
        return max(1, ocr.max_concurrent_regions)
    return max(1, min(ocr.concurrent_regions_per_file, ocr.max_concurrent_regions))


def _backend_region_count(file_states) -> int:
    """Regions that will actually invoke the OCR backend.

    Regions pre-filled by NativeTextStage (``_native_text_used``) and the
    ``abandon``/``skip`` task types bypass OCR, and a file that already carries
    ``ocr_regions`` (a cache hit) contributes nothing. Zero means no engine is
    needed for these files at all.
    """
    total = 0
    for fs in file_states:
        if fs.ocr_regions is not None or not fs.layout_results:
            continue
        for page_regions in fs.layout_results:
            for region in page_regions:
                if region.get("task_type", "text") in ("abandon", "skip"):
                    continue
                if region.get("_native_text_used"):
                    continue
                total += 1
    return total


async def _no_engine_recognize(_image, _prompt: str) -> str:
    raise RuntimeError("OCR backend invoked although no region needed it; no engine was started")


@dataclass(frozen=True)
class OcrIdentityResolution:
    """Outcome of the shared automatic-chain identity state machine.

    ``identity`` is None only when startup failed — the targets were already
    failed and the caller must return early. ``engine_ready`` means the
    concrete runtime was adopted via ``await_ocr`` in this call, so the
    caller skips its later plain engine start.
    """

    identity: OcrRuntimeIdentity | None
    engine_ready: bool


def fail_ocr_targets(targets: list, exc: BaseException, *, stage: str, log: bool = True) -> None:
    """Mark OCR-needing files failed instead of aborting the chunk.

    Only files still needing OCR are targeted: files already served from the
    regions cache, or native parses that never needed the engine, keep their
    output. Callers pass the pending set, not ``ctx.alive()``.
    """
    for fs in targets:
        if fs.error is None:
            fs.set_error(
                f"OCR backend init failed: {exc}",
                code="ocr_failed",
                stage=stage,
                exc=exc,
                outage=True,
            )
    if log:
        logger.warning("OCR backend init failed", exc_info=exc)


async def resolve_ocr_identity(
    ctx,
    *,
    needs_backend: bool,
    fail_targets: list,
    stage: str = "ocr",
    strict: bool = False,
) -> OcrIdentityResolution:
    """Resolve the OCR runtime identity for one window (both OCR stages).

    Owns the automatic-chain (``ocr_backend == \"paddle\"``) state machine in
    one place: a stale runtime whose process died unseen is cleared, a
    retained loaded client is reused, otherwise the chain starts the engine
    and adopts the selected concrete runtime. A window that needs no backend
    (native text covers every region, or — pre-layout — nothing to look up)
    keeps a local static fallback without persisting it: persisting it used
    to make the next window reuse Paddle prompts, profile and provenance
    while a GLM engine did the work.

    On startup failure the targets are failed, the chunk-scoped marker is
    recorded so later windows fast-fail, and the resolution carries None.
    When startup reports success without a concrete identity, the bundle
    probe (``strict=True``) raises while OcrStage keeps the compatibility
    seam for injected, already-ready clients.
    """
    cfg = ctx.config
    settings = ctx.settings
    rm = ctx.resources
    requested_backend = cfg.ocr_backend or settings.ocr.backend
    automatic_backend = requested_backend == "paddle"
    identity = ctx.scratch.get("ocr_runtime_identity")

    runtime_identity = getattr(rm, "ocr_runtime_identity", None)
    runtime_client = getattr(rm, "ocr", None)
    runtime_loaded = bool(getattr(runtime_client, "loaded", False))
    if (
        automatic_backend
        and isinstance(runtime_identity, OcrRuntimeIdentity)
        and not runtime_loaded
    ):
        # A managed process can die without going through shutdown_ocr().
        # Its identity is no longer evidence for the next client: clear it
        # so the original selector runs and records the replacement.
        rm.ocr_runtime_identity = None
        runtime_identity = None
        identity = None
        ctx.scratch.pop("ocr_runtime_identity", None)
    if (
        automatic_backend
        and identity is None
        and isinstance(runtime_identity, OcrRuntimeIdentity)
        and runtime_loaded
    ):
        # A retained automatic client selected this concrete runtime in an
        # earlier process_chunk context. Reuse it for this context's
        # profile, cache key, and provenance.
        identity = runtime_identity
        ctx.scratch["ocr_runtime_identity"] = identity
    if automatic_backend and identity is None and needs_backend:
        # A hard init failure in an earlier window of this chunk must not
        # re-run the slow, doomed engine constructor here.
        prior_init_error = ctx.signals.ocr_init_error
        if prior_init_error is not None:
            fail_ocr_targets(fail_targets, prior_init_error, stage=stage, log=False)
            return OcrIdentityResolution(None, False)
        # An automatic chain has no exact cache identity until startup
        # chooses a concrete candidate. This trades a cold-cache startup
        # for correct cache/provenance separation between fallback models.
        try:
            await rm.await_ocr()
        except Exception as e:  # noqa: BLE001
            ctx.signals.ocr_init_error = e
            fail_ocr_targets(fail_targets, e, stage=stage)
            return OcrIdentityResolution(None, False)
        runtime_identity = getattr(rm, "ocr_runtime_identity", None)
        if isinstance(runtime_identity, OcrRuntimeIdentity):
            identity = runtime_identity
            ctx.scratch["ocr_runtime_identity"] = identity
            return OcrIdentityResolution(identity, True)
        injected = getattr(rm, "_ocr", None)
        if strict and not getattr(injected, "loaded", False):
            raise RuntimeError("OCR startup completed without a concrete runtime identity")
        # Compatibility seam for injected, already-ready clients that
        # do not participate in managed candidate identity selection.
        identity = resolve_ocr_runtime_identity(cfg, settings)
        ctx.scratch["ocr_runtime_identity"] = identity
        return OcrIdentityResolution(identity, False)
    if identity is None:
        identity = resolve_ocr_runtime_identity(cfg, settings)
        if not automatic_backend:
            ctx.scratch["ocr_runtime_identity"] = identity
    return OcrIdentityResolution(identity, False)


def _effective_settings(settings: GlobalSettings | None) -> GlobalSettings:
    if settings is not None:
        return settings
    from bibr.config import snapshot_settings

    return snapshot_settings()


def _is_remote_ocr(cfg) -> bool:
    """Whether OCR inference runs off-process (concurrent) vs a local engine."""
    return cfg.ocr_backend in REMOTE_OCR_BACKENDS or bool(cfg.ocr_url)


# Labels excluded from the OCR success-rate denominator: regions that are
# never OCR'd and legitimately have empty content. Derived from the layout
# task mapping (skip → image/chart, abandon → number/aside_text/...) so it
# can't drift; the task-type names themselves are kept defensively.
_NON_TEXT_LABELS = (
    frozenset(LABEL_TASK_MAPPING["skip"])
    | frozenset(LABEL_TASK_MAPPING["abandon"])
    | {"figure", "abandon", "skip"}
)


def _bbox_containment(inner: list, outer: list) -> float:
    """Fraction of *inner* bbox's area that overlaps with *outer*.

    Returns intersection_area / inner_area.  A value of 1.0 means the inner
    bbox is fully contained within the outer bbox.  Unlike IoU this correctly
    detects small regions inside large ones (e.g., an inline formula inside
    a text block).
    """
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    inner_area = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])
    return inter / inner_area if inner_area > 0 else 0.0


def _deduplicate_formula_text_regions(
    pages: list[list[dict]], settings: GlobalSettings | None = None
) -> list[list[dict]]:
    """Remove formula regions that are contained within text regions.

    When the layout detector returns both an ``inline_formula`` region and a
    ``text`` region covering the same area, both get OCR'd independently,
    producing duplicate content (formula wrapped in ``$$`` plus the same
    content inline in the text block).  This function drops the formula
    region when >=50% of its area is contained within a text region.
    """
    _FORMULA_LABELS = {"formula", "display_formula", "inline_formula"}
    _TEXT_LABELS = {"text", "content"}
    containment_threshold = _effective_settings(settings).layout.containment_threshold

    for page_regions in pages:
        text_regions = [
            r
            for r in page_regions
            if r and r.get("native_label") in _TEXT_LABELS and r.get("bbox_2d")
        ]
        if not text_regions:
            continue

        to_remove = set()
        for i, region in enumerate(page_regions):
            if not region or region.get("label") not in _FORMULA_LABELS:
                continue
            bbox = region.get("bbox_2d")
            if not bbox:
                continue
            for text_r in text_regions:
                if _bbox_containment(bbox, text_r["bbox_2d"]) > containment_threshold:
                    to_remove.add(i)
                    break

        if to_remove:
            page_regions[:] = [r for i, r in enumerate(page_regions) if i not in to_remove]

    return pages


def _deduplicate_reference_regions(
    pages: list[list[dict]], settings: GlobalSettings | None = None
) -> list[list[dict]]:
    """Suppress ``reference`` regions whose text is already covered elsewhere.

    PP-DocLayoutV3 sometimes detects both a ``reference`` region (label 18)
    and a ``reference_content`` region (label 19) covering the same text area.
    Cross-class NMS (IoU threshold 0.98) is too strict to suppress them, and
    containment filtering processes each label class independently.  When both
    survive, the same text is OCR'd twice, producing duplicate reference entries
    that confuse downstream LLM extraction.

    Pass 1 drops a ``reference`` region when >=50% of its area overlaps
    with ``reference_content`` regions (or vice versa) whose text already
    holds its own. Geometry alone is not enough: an aggregate box over four
    entries with entry boxes for only two would take the other two with it.
    A kept box's duplicated entry boxes are shadowed by the parser.

    Pass 2 handles the same double-detection against per-entry ``text``
    regions (data/prereg.pdf pages 22-23): a full-column ``reference`` region
    plus per-entry ``text`` rows over the same column.  Geometry is useless
    here — the per-entry provenance bboxes are coarse (several rows share one
    bbox) — so the duplicate is detected by content: when the region's
    normalized text is already contained in the page's text regions, its
    content is blanked.  The region itself is KEPT, because it still keys the
    References section (``_handle_section_hint``) and emits the layout hint
    used by section-classification fallbacks.

    Both passes tolerate OCR noise between the two reads, but not a run of
    text the other regions lack (``alnum_text_covered``): a region holding a
    line or a DOI that no text region has, such as the end of a reference
    continued from the previous page, keeps its content.
    """
    containment_threshold = _effective_settings(settings).layout.containment_threshold

    for page_regions in pages:
        ref_content_regions = [
            r
            for r in page_regions
            if r and r.get("native_label") == "reference_content" and r.get("bbox_2d")
        ]
        if not ref_content_regions:
            continue

        to_remove = set()
        for i, region in enumerate(page_regions):
            if not region or region.get("native_label") != "reference":
                continue
            # A rejected native candidate is an alternate source diagnostic;
            # never delete its carrier during overlap cleanup.
            if region.get("_native_text_candidate") is not None:
                continue
            bbox = region.get("bbox_2d")
            if not bbox:
                continue
            overlapping = [
                rc_r
                for rc_r in ref_content_regions
                if _bbox_containment(bbox, rc_r["bbox_2d"]) > containment_threshold
                or _bbox_containment(rc_r["bbox_2d"], bbox) > containment_threshold
            ]
            if overlapping and alnum_text_covered(
                alnum_key(region.get("content") or ""),
                alnum_key("".join(rc_r.get("content") or "" for rc_r in overlapping)),
            ):
                to_remove.add(i)

        if to_remove:
            page_regions[:] = [r for i, r in enumerate(page_regions) if i not in to_remove]

    # Pass 2: content-based suppression against plain text regions.
    _TEXT_BLOB_LABELS = {"text", "content", "reference_content"}
    for page_regions in pages:
        blob = alnum_key(
            "".join(
                r.get("content") or ""
                for r in page_regions
                if r and r.get("native_label") in _TEXT_BLOB_LABELS
            )
        )
        if not blob:
            continue
        for region in page_regions:
            if not region or region.get("native_label") != "reference":
                continue
            if region.get("_native_text_candidate") is not None:
                continue
            ref_norm = alnum_key(region.get("content") or "")
            if ref_norm and alnum_text_covered(ref_norm, blob):
                logger.info(
                    "Blanked duplicate reference region (%d chars already "
                    "covered by text regions on the page)",
                    len(ref_norm),
                )
                region["content"] = ""

    return pages


def _to_typed_regions(pages: list[list[dict]]) -> list[list[OcrRegionResult]]:
    """Lift postprocessed wire-format dicts into the typed Region IR.

    Everything before this point (OCR fan-out, dedupe, postprocess merges)
    works on the legacy dict shape stage-internally; ``FileState.ocr_regions``
    carries only ``OcrRegionResult`` objects.
    """
    return [[OcrRegionResult.from_dict(r) for r in page] for page in pages]


def _postprocess_ocr_regions(
    pages: list[list[dict]], settings: GlobalSettings | None = None
) -> list[list[dict]]:
    """Apply post-processing to OCR region results.

    Runs formula-number merging, hyphenated-word merging, and bullet-point
    inference using standalone functions from ``bibr.ocr.postprocess``.
    Regions filled from the PDF text layer (``_native_text_used``) skip the
    OCR-artifact cleanup; they are only trimmed.
    """
    from bibr.ocr.normalization import strip_one_balanced_formula_wrapper
    from bibr.ocr.postprocess import (
        clean_ocr_content,
        format_bullet_points,
        merge_formula_numbers,
        merge_text_blocks,
    )

    # Remove formula regions that overlap with text regions before merging
    effective = _effective_settings(settings)
    pages = _deduplicate_formula_text_regions(pages, effective)

    # Remove duplicate reference/reference_content overlaps
    pages = _deduplicate_reference_regions(pages, effective)

    for page_regions in pages:
        # 0. Clean raw OCR output: strip \t, collapse repeated punctuation,
        #    remove hallucinated repetitions, normalise numbered lists. Text
        #    from the PDF text layer has none of these artifacts, and the
        #    repairs damage it ("U.S." -> "U. S.", a dot-leader table of
        #    contents cut after its first entry), so it is only trimmed.
        for region in page_regions:
            content = region.get("content")
            if not content:
                continue
            if region.get("_native_text_used"):
                region["content"] = content.strip()
            else:
                region["content"] = clean_ocr_content(
                    content, formula=region.get("label") == "formula"
                )

        # 1. Pre-wrap formula content in $$\n...\n$$ so that
        #    merge_formula_numbers can detect endswith("\n$$") for \tag{}.
        #    PDFParser._handle_formula won't double-wrap: it checks
        #    startswith("$") first. Only a wrapper whose opening delimiter
        #    the final one closes is removed ("\\(a\\) + \\(b\\)" is two
        #    formulas), and a single "$...$" pair too, which would otherwise
        #    end up nested inside the "$$".
        for region in page_regions:
            if region.get("label") == "formula":
                content = region.get("content", "")
                if content:
                    inner = strip_one_balanced_formula_wrapper(content, single_dollar=True)
                    region["content"] = "$$\n" + inner + "\n$$"

        # 2. Normalise bullet markers so format_bullet_points can detect
        #    existing bullet context for gap-filling. "* " is the Markdown
        #    bullet OCR emits; in the text layer it is a printed asterisk
        #    ("* p < .05"), so only bullet glyphs are rewritten there.
        for region in page_regions:
            if region.get("native_label") == "text":
                content = region.get("content", "")
                if content and (
                    content.startswith("·")
                    or content.startswith("•")
                    or (content.startswith("* ") and not region.get("_native_text_used"))
                ):
                    region["content"] = "- " + content[1:].lstrip()

        # 3. Run the three self-contained merge/format functions.
        page_regions[:] = merge_formula_numbers(page_regions)
        # Dehyphenation consumes the second region.  Treat alternate-source
        # diagnostics as merge boundaries so no native candidate/raw OCR pair
        # can disappear into a neighboring block. Ordinary regions retain the
        # existing merge behavior within each contiguous run.
        merged_text_regions: list[dict] = []
        merge_run: list[dict] = []
        for region in page_regions:
            if region.get("_native_text_candidate") is not None:
                if merge_run:
                    merged_text_regions.extend(merge_text_blocks(merge_run))
                    merge_run = []
                merged_text_regions.append(region)
            else:
                merge_run.append(region)
        if merge_run:
            merged_text_regions.extend(merge_text_blocks(merge_run))
        for index, region in enumerate(merged_text_regions):
            region["index"] = index
        page_regions[:] = merged_text_regions
        page_regions[:] = format_bullet_points(page_regions)

    return pages


def ocr_page_regions(
    page_img,
    regions: list[dict],
    page_idx: int,
    filename: str,
    ocr_fn,
    ocr_sem: asyncio.Semaphore | _DualSemaphore | None = None,
    *,
    include_figures: bool | None = None,
    settings: GlobalSettings | None = None,
    profile: OcrProfile = GLM_PROFILE,
    warning_sink: Callable[[ProcessingWarning], None] | None = None,
) -> Coroutine[None, None, list[dict]]:
    """OCR all regions on a single page (shared between LitServe and local pipelines).

    ``ocr_fn`` is an async callable ``(cropped_image, prompt) -> str`` that runs
    OCR on a single cropped region image.

    Preserves the reading order from the layout detector by pre-allocating
    slots and filling them in after OCR completes.
    Regions pre-filled by ``fill_regions_from_native_text`` (flagged with
    ``_native_text_used=True``) are emitted directly without an OCR call.

    ``include_figures`` overrides ``Settings.FIGURE_IMAGES`` for this request
    when not None.
    """
    return _ocr_page_regions_impl(
        page_img,
        regions,
        page_idx,
        filename,
        ocr_fn,
        ocr_sem,
        include_figures=include_figures,
        settings=settings,
        profile=profile,
        warning_sink=warning_sink,
    )


async def _ocr_page_regions_impl(
    page_img,
    regions: list[dict],
    page_idx: int,
    filename: str,  # noqa: ARG001 — part of public interface, used by callers for context
    ocr_fn,
    ocr_sem,
    *,
    include_figures: bool | None = None,
    settings: GlobalSettings | None = None,
    profile: OcrProfile = GLM_PROFILE,
    warning_sink: Callable[[ProcessingWarning], None] | None = None,
) -> list[dict]:
    """Implementation of ocr_page_regions; see public wrapper for documentation.

    Preserves the reading order from the layout detector by pre-allocating
    slots and filling them in after OCR completes.
    Regions pre-filled by ``fill_regions_from_native_text`` (flagged with
    ``_native_text_used=True``) are emitted directly without an OCR call.
    """
    # ``crop_image_region`` needs opencv (the ``ml`` extra). Imported at each
    # call site (not top-of-function) so pages with no crop-requiring regions
    # (native-text bypass, no figures) work in a core-only install; both call
    # sites already fall back gracefully (PIL crop / page_img) if it's absent.

    # Per-request override falls back to global setting
    effective = _effective_settings(settings)
    emit_figures = include_figures if include_figures is not None else effective.FIGURE_IMAGES

    # Pre-allocate one slot per non-abandoned region to preserve reading order.
    region_list: list[dict | None] = []
    slot_map: dict[int, int] = {}
    ocr_tasks = []

    for i, region in enumerate(regions):
        task_type = region.get("task_type", "text")

        if task_type == "abandon":
            continue

        slot_idx = len(region_list)
        slot_map[i] = slot_idx

        if task_type == "skip":
            # Image/chart regions: optionally crop and base64-encode the figure image
            image_b64 = None
            if emit_figures:
                fig_bbox = region.get("bbox_2d", [0, 0, 1000, 1000])
                try:
                    from bibr.ocr.image_processing import crop_image_region

                    fig_crop = crop_image_region(page_img, fig_bbox)
                except Exception:
                    fig_crop = page_img
                image_b64 = base64.b64encode(pil_to_bytes(fig_crop)).decode()
            region_list.append(
                OcrRegionResult.from_layout_region(
                    region,
                    slot_idx=slot_idx,
                    content="",
                    image_b64=image_b64,
                ).to_dict()
            )
            continue

        # --- Native text bypass: region was pre-filled by
        # fill_regions_from_native_text; emit its content directly
        # without an OCR call.
        if region.get("_native_text_used"):
            region_list.append(
                OcrRegionResult.from_layout_region(
                    region,
                    slot_idx=slot_idx,
                    content=region["content"],
                ).to_dict()
            )
            continue

        # Reserve the slot (filled after OCR)
        region_list.append(None)

        bbox = region.get("bbox_2d", [0, 0, 1000, 1000])
        ocr_task: OcrTask = task_type if task_type in {"text", "table", "formula"} else "text"
        ocr_tasks.append((i, region, bbox, ocr_task, profile.prompt_for(ocr_task)))

    # Send OCR requests concurrently (throttled by semaphore)
    if ocr_tasks:

        def _crop(bbox):
            """Crop one region. ``Image.crop`` is an eager copy, so this is
            deferred until the region semaphore has been acquired: cropping
            every region of every page up front held all of them alongside
            ``fs.page_images``, which still holds every page image."""
            try:
                from bibr.ocr.image_processing import crop_image_region

                return crop_image_region(page_img, bbox)
            except Exception as e:  # noqa: BLE001
                logger.warning("crop_image_region failed, using PIL fallback: %s", e)
                w, h = page_img.size
                return page_img.crop(
                    (
                        int(bbox[0] * w / 1000),
                        int(bbox[1] * h / 1000),
                        int(bbox[2] * w / 1000),
                        int(bbox[3] * h / 1000),
                    )
                )

        async def _ocr_with_corruption_retry(cropped, prompt):
            # Control-char-riddled output (the vllm-mlx --mllm NUL failure
            # mode) gets one re-OCR; the cleaner of the two results wins.
            # Downstream strips the controls, but stripping can't restore
            # characters the first pass destroyed — a retry sometimes can.
            result = await ocr_fn(cropped, prompt)
            corruption = ocr_corruption_count(result)
            if corruption < OCR_CORRUPTION_MIN_CHARS:
                return result
            logger.warning(
                "[page %d] OCR region text contains %d control chars (corrupted output); "
                "retrying region once",
                page_idx,
                corruption,
            )
            try:
                retried = await ocr_fn(cropped, prompt)
            except Exception as e:
                logger.warning("[page %d] corrupted-region re-OCR failed: %s", page_idx, e)
                return result
            return retried if ocr_corruption_count(retried) < corruption else result

        async def _run_ocr(bbox, prompt):
            if ocr_sem is not None:
                async with ocr_sem:
                    return await _ocr_with_corruption_retry(_crop(bbox), prompt)
            return await _ocr_with_corruption_retry(_crop(bbox), prompt)

        ocr_coros = [_run_ocr(bbox, prompt) for _, _, bbox, _, prompt in ocr_tasks]
        ocr_results = await asyncio.gather(*ocr_coros, return_exceptions=True)

        # Fill pre-allocated slots in their original positions
        from bibr.exceptions import BibrError

        for (orig_idx, region, _, task_type, _), result in zip(ocr_tasks, ocr_results, strict=True):
            content: str = ""
            raw_content: str | None = None
            finish_reason: str | None = None
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException) and (
                isinstance(result, BibrError) or _ocr_server_gone(result)
            ):
                # Systemic backend failures (auth, config, an OCR server that
                # went down: connection refused or dropped once the transport's
                # retries ran out) must not be silently converted to blank
                # content — propagate to fail the page. A busy answer
                # (429/502/503) is not one of them: see _ocr_server_gone.
                raise result
            if isinstance(result, BaseException):
                # The region ships blank. Without an export-visible warning
                # three failed regions in a 40-page paper look like missing
                # paragraphs behind a clean receipt. The export names the
                # error but not the OCR endpoint; the log keeps the raw text.
                where = f"page {page_idx + 1}, region {orig_idx}, task {task_type}"
                warning = ProcessingWarning(
                    WarningCode.OCR_REGION_FAILED,
                    f"OCR failed for a region; its text is missing ({where}): "
                    f"{describe_error(result)}",
                )
                logger.warning(
                    "OCR failed for a region (%s): %s: %s", where, type(result).__name__, result
                )
                if warning_sink is not None:
                    warning_sink(warning)
            else:
                provider_finish_reason = getattr(result, "finish_reason", None)
                finish_reason = provider_finish_reason
                if profile.name == "paddle" and task_type == "table":
                    completeness = check_otsl_completeness(result)
                    incomplete_reasons = list(completeness.reasons)
                    if provider_finish_reason == "length":
                        incomplete_reasons.insert(0, "finish_reason_length")
                    if incomplete_reasons:
                        reasons = ", ".join(dict.fromkeys(incomplete_reasons))
                        warning = ProcessingWarning(
                            WarningCode.OCR_TABLE_INCOMPLETE,
                            "OCR table output incomplete "
                            f"(page {page_idx + 1}, region {orig_idx}, "
                            f"finish_reason={provider_finish_reason or 'unknown'}, "
                            f"reasons: {reasons})",
                        )
                        logger.warning(warning.message)
                        if warning_sink is not None:
                            warning_sink(warning)
                elif provider_finish_reason == "length":
                    warning = ProcessingWarning(
                        WarningCode.OCR_OUTPUT_TRUNCATED,
                        "OCR output truncated at the generation limit "
                        f"(page {page_idx + 1}, region {orig_idx}, task {task_type})",
                    )
                    logger.warning(warning.message)
                    if warning_sink is not None:
                        warning_sink(warning)
                normalized = normalize_ocr_output(profile, task_type, result)
                content = normalized.content
                raw_content = (
                    result
                    if region.get("_native_text_rejection_reason") is not None
                    else normalized.raw_content
                )
                if warning_sink is not None:
                    for warning in normalized.warnings:
                        warning_sink(
                            ProcessingWarning(
                                warning.code,
                                f"{warning.message} (page {page_idx + 1}, region {orig_idx})",
                            )
                        )

            slot_idx = slot_map[orig_idx]
            region_list[slot_idx] = OcrRegionResult.from_layout_region(
                region,
                slot_idx=slot_idx,
                content=content,
                raw_content=raw_content,
                finish_reason=finish_reason,
            ).to_dict()

    # All slots are filled by this point — `None` placeholders only exist
    # mid-loop while OCR is in flight.
    return [r for r in region_list if r is not None]


def _should_unload_ocr_after_chunk(
    memory_mode: str,
    llm_backend: str,
    settings: GlobalSettings | None = None,
) -> bool:
    """Whether to tear down the OCR engine at the end of an OCR stage.

    The freed VRAM only has a consumer when a *local* LLM server shares the GPU
    during the extract stage. With a cloud/remote LLM (the default) a per-chunk
    teardown just reloads OCR weights (vllm-mlx 30-60s) on the next chunk of a
    batch — pure thrash. So:

    - ``keep_all`` never unloads (``Pipeline.aclose`` tears down once);
    - ``aggressive`` always unloads (8 GB boxes reclaim every byte);
    - ``balanced`` keeps OCR resident across chunks unless a local LLM server
      backend follows (``vllm`` / ``vllm-mlx``, or the unresolved ``local``
      alias).

    ``OCR_UNLOAD_BETWEEN_CHUNKS`` overrides the heuristic: ``always`` forces the
    per-chunk teardown, ``never`` keeps OCR loaded, ``auto`` (default) applies
    the rules above.
    """
    override = _effective_settings(settings).ocr.unload_between_chunks
    if override == "always":
        return True
    if override == "never":
        return False
    if memory_mode == "keep_all":
        return False
    if memory_mode == "aggressive":
        return True
    # balanced: unload only when a local LLM server will use the freed VRAM.
    # "local" is the CLI alias; it resolves to a concrete managed backend
    # (vllm / vllm-mlx) before reaching RunConfig, so match both.
    from bibr.local.pipeline import LOCAL_LLM_BACKENDS

    return llm_backend == "local" or llm_backend in LOCAL_LLM_BACKENDS


class OcrStage:
    name = "ocr"
    # FileState fields consumed / populated (see validate_stage_contracts).
    requires = ("layout_results", "page_images", "page_indices")
    produces = ("ocr_regions",)

    async def run(self, ctx: PipelineContext) -> None:
        ctx.progress.stage_start(self.name)
        t0 = time.monotonic()
        if not ctx.signals.any_needs_ocr:
            logger.debug("OCR stage: skipped (no files need OCR)")
        else:
            await self._run(ctx)
        logger.debug("OCR stage: %.1fs", time.monotonic() - t0)
        ctx.progress.stage_end(self.name)

    async def _run(self, ctx: PipelineContext) -> None:
        rm = ctx.resources
        cfg = ctx.config

        # Opt-in disk cache: serve OCR regions for already-seen PDFs from disk
        # so re-runs under different ref/parse params skip OCR entirely. Only
        # engages on the PDF/OCR path — DOCX files are pre-parsed (contents set)
        # and never enter OCR.
        from bibr.pipeline import ocr_cache

        settings = ctx.settings
        requested_backend = cfg.ocr_backend or settings.ocr.backend
        automatic_backend = requested_backend == "paddle"
        # Shared automatic-chain identity state machine (see
        # ``resolve_ocr_identity``): stale/retained runtimes, engine start,
        # and the static fallback. ``eligible`` — not ``ctx.alive()`` — is
        # the failure set, so native parses and cache hits are never failed
        # for an engine they never needed.
        resolution = await resolve_ocr_identity(
            ctx,
            needs_backend=_backend_region_count(ctx.alive()) > 0,
            fail_targets=self._eligible_files(ctx),
            stage=self.name,
        )
        identity = resolution.identity
        if identity is None:
            return
        cache_on = ocr_cache.is_enabled(settings)
        alive = ctx.alive()
        if cache_on:
            for fs in alive:
                if fs.contents is not None or fs.ocr_regions is not None:
                    continue
                cached = ocr_cache.load(fs, cfg, identity, settings)
                if cached is not None:
                    fs.ocr_regions = cached
            self._check_ocr_success(ctx)
            alive = ctx.alive()
            if not alive:
                return
        pending = [fs for fs in alive if fs.contents is None and fs.ocr_regions is None]
        if alive and not pending:
            logger.debug("OCR stage: all files served from disk cache")
            return

        # Shared-resource init failures (missing model, unreachable server)
        # must fail per-file rather than crash the whole chunk — subsequent
        # stages iterate ``ctx.alive()`` and will no-op.
        #
        # The interleaved render/OCR stage drives this per file-window (window
        # of 1 for local OCR), so a hard init failure would otherwise re-run the
        # slow, doomed engine constructor once per file. Record it on the
        # chunk-scoped signals and fast-fail the rest of the chunk; ``signals``
        # is fresh per ``process_chunk``, so a later chunk (or serve request)
        # still retries — unlike a ResourceManager-lifetime cache that would
        # poison a long-lived server after one transient failure.
        # Count only regions that will actually invoke the OCR backend.
        # Regions pre-filled by NativeTextStage (``_native_text_used``) and
        # ``abandon``/``skip`` task types bypass OCR, so excluding them here
        # makes the progress bar reflect real work — otherwise users see
        # ``0/127`` even when the document was fully extracted via native text.
        # It also decides whether an engine is started at all: a window whose
        # regions were all filled natively never pays for OCR startup.
        total_regions = _backend_region_count(pending)
        engine_needed = total_regions > 0
        if engine_needed and not resolution.engine_ready and ctx.signals.ocr_init_error is not None:
            # Chunk-scoped fast-fail for every backend. resolve_ocr_identity
            # covers the automatic chain before its startup; explicit backends
            # (and an automatic window arriving with a retained or static
            # identity) reach the engine start below, so check here — after
            # the regions-cache probe — that only pending files fail while
            # cache hits and native parses survive.
            fail_ocr_targets(pending, ctx.signals.ocr_init_error, stage=self.name, log=False)
            return
        if engine_needed:
            try:
                if not resolution.engine_ready:
                    await rm.await_ocr()  # always async, regardless of whether preload ran
                if not resolution.engine_ready and hasattr(rm.ocr, "wait_for_server"):
                    await rm.ocr.wait_for_server()
            except Exception as e:  # noqa: BLE001
                ctx.signals.ocr_init_error = e
                fail_ocr_targets(pending, e, stage=self.name)
                return
            if automatic_backend:
                # The engine that actually started selects the concrete
                # runtime: adopt it when this window arrived with a stale
                # static fallback. Defensive — current code no longer persists
                # one, but the injected-client compatibility seam (or a future
                # caller) could still hand one to the engine start. Prompts,
                # profile, region limit, cache key and provenance below must
                # describe this runtime.
                started = getattr(rm, "ocr_runtime_identity", None)
                if isinstance(started, OcrRuntimeIdentity) and started != identity:
                    identity = started
                    ctx.scratch["ocr_runtime_identity"] = identity
        else:
            logger.info(
                "OCR engine not started: native text covers every region of %d file(s)",
                len(pending),
            )
        ctx.progress.ocr_start(total_regions)

        raw_ocr_fn = rm.ocr.recognize if engine_needed else _no_engine_recognize
        profile = resolve_ocr_profile(
            explicit=identity.profile,
            backend=identity.backend,
            model=identity.model,
            max_tokens=settings.ocr.generation_max_tokens,
            temperature=settings.ocr.generation_temperature,
        )
        ctx.scratch["ocr_profile"] = profile

        async def tracked_ocr_fn(image, prompt: str) -> str:
            result: str = await raw_ocr_fn(image, prompt)
            ctx.progress.ocr_region_done()
            return result

        ocr_fn = tracked_ocr_fn
        is_remote = _is_remote_ocr(cfg)

        try:
            if is_remote:
                await self._run_remote(ctx, ocr_fn)
            else:
                # Size against the runtime that actually started, not the
                # requested backend: the automatic ``paddle`` selector resolves
                # to a concrete candidate whose concurrency ceiling differs.
                ocr_sem = asyncio.Semaphore(_local_region_limit(settings, identity.backend))
                await self._run_local(ctx, ocr_fn, ocr_sem)
            self._check_ocr_success(ctx)
            # A native-only window under the automatic chain ran with a static
            # identity that no later probe looks up; storing it would be waste.
            if cache_on and (engine_needed or not automatic_backend):
                for fs in pending:
                    if fs.error is None and fs.ocr_regions is not None:
                        ocr_cache.store(fs, cfg, identity, fs.ocr_regions, settings)
        finally:
            # Always close the Rich Live progress so a CancelledError or
            # mid-stage exception doesn't leave the terminal in alt-screen mode.
            ctx.progress.ocr_end()
            # Tear down the engine between chunks only when something will use
            # the freed VRAM (see ``_should_unload_ocr_after_chunk``); otherwise
            # keep it resident so local engines do not
            # re-load weights every chunk. ``Pipeline.aclose()`` always tears it
            # down at end of life. ``defer_ocr_teardown`` lets an outer driver
            # (InterleavedRenderOcrStage) run OCR per file-window yet tear the
            # engine down just once for the whole chunk, not once per window.
            if not ctx.signals.defer_ocr_teardown and _should_unload_ocr_after_chunk(
                cfg.memory_mode, cfg.llm_backend, settings
            ):
                await rm.shutdown_ocr()

    @staticmethod
    def _check_ocr_success(ctx) -> None:
        """Fail files where OCR delivered too little usable content.

        Catches the silent-success failure mode where every OCR call errors
        (e.g. the local engine crashed) but per-region failures fall through
        as empty content. Without this, the pipeline emits a 'success' JSON
        with no extracted text and the user only finds out later.

        Native-text-bypass regions (which never went to OCR) and
        ``abandon``/``skip`` regions are excluded from the denominator.
        """
        threshold = ctx.settings.ocr.min_success_rate
        if threshold <= 0:
            return
        for fs in ctx.alive():
            if fs.ocr_regions is None:
                continue
            # A page that failed wholesale yields no regions at all, so it
            # lands in neither the numerator nor the denominator below: a file
            # whose pages nearly all died still scored 100% on the handful that
            # survived. Gate on the page ratio first.
            if fs.ocr_pages_attempted:
                page_rate = 1.0 - (fs.ocr_pages_failed / fs.ocr_pages_attempted)
                if page_rate < threshold:
                    msg = (
                        f"OCR failed outright on {fs.ocr_pages_failed}/"
                        f"{fs.ocr_pages_attempted} pages "
                        f"({page_rate:.0%} < {threshold:.0%} threshold) — "
                        "the OCR backend likely failed mid-run"
                    )
                    fs.set_error(msg, code="ocr_mostly_failed", stage="ocr")
                    logger.warning("%s: %s", fs.path.name, msg)
                    continue
            ocr_needed = 0
            ocr_filled = 0
            for page in fs.ocr_regions:
                for region in page:
                    label = region.native_label or region.label or "text"
                    # Skip non-text regions (figures, abandoned).
                    if label in _NON_TEXT_LABELS:
                        continue
                    # Native-text-bypassed regions never went through OCR — they
                    # were pre-filled from the embedded text layer. Counting
                    # them (as either needed or filled) defeats the gate: a
                    # mostly-native doc whose few OCR calls all crashed would
                    # still show a high "success" rate. Exclude them so the
                    # denominator/numerator cover only regions OCR processed.
                    if region.native_text_used:
                        continue
                    ocr_needed += 1
                    if region.content.strip():
                        ocr_filled += 1
            if ocr_needed == 0:
                continue
            success_rate = ocr_filled / ocr_needed
            if success_rate < threshold:
                msg = (
                    f"OCR yielded content for only {ocr_filled}/{ocr_needed} "
                    f"text regions ({success_rate:.0%} < {threshold:.0%} threshold) — "
                    "the OCR backend likely failed mid-run"
                )
                fs.set_error(msg, code="ocr_mostly_failed", stage="ocr")
                logger.warning("%s: %s", fs.path.name, msg)

    @staticmethod
    def _eligible_files(ctx) -> list:
        """Files still needing OCR: no native contents and no OCR regions yet.

        DOCX files carry ``contents`` (native parse) and cache hits carry
        ``ocr_regions`` — both bypass OCR. Identical filter for both the remote
        and local paths.
        """
        return [fs for fs in ctx.alive() if fs.contents is None and fs.ocr_regions is None]

    async def _ocr_one_file(self, fs, ctx, ocr_fn, region_sem) -> None:
        """OCR all pages of one file; sets fs.ocr_regions or fs.error.

        Shared per-file body for both the concurrent remote path and the
        sequential local path — they differ only in file-level concurrency and
        how ``region_sem`` is sized (see ``_run_remote`` / ``_run_local``).
        """
        cfg = ctx.config
        profile: OcrProfile = ctx.scratch["ocr_profile"]
        try:
            fs_t0 = time.monotonic()
            page_coros = [
                ocr_page_regions(
                    page_img,
                    regions,
                    orig_idx,
                    fs.path.name,
                    ocr_fn,
                    region_sem,
                    include_figures=cfg.include_figures,
                    settings=ctx.settings,
                    profile=profile,
                    warning_sink=fs.warnings.append,
                )
                for orig_idx, page_img, regions in zip(
                    fs.page_indices, fs.page_images, fs.layout_results, strict=True
                )
            ]
            page_results = await asyncio.gather(*page_coros, return_exceptions=True)

            # CancelledError must escape so the caller's shutdown logic runs.
            for r in page_results:
                if isinstance(r, asyncio.CancelledError):
                    raise r

            # Accumulate per-page failures as warnings; substitute empty region
            # lists for failed pages so post-processing can continue.
            clean_pages: list[list[dict]] = []
            errors: list[BaseException] = []
            for page_idx, r in zip(fs.page_indices, page_results, strict=True):
                if isinstance(r, BaseException):
                    errors.append(r)
                    fs.warnings.append(
                        ProcessingWarning(
                            WarningCode.OCR_PAGE_FAILED,
                            "OCR failed for a page; its text is missing "
                            f"(page {page_idx + 1}): {describe_error(r)}",
                        )
                    )
                    logger.warning(
                        "OCR failed for page %d of %s: %s: %s",
                        page_idx + 1,
                        fs.path.name,
                        type(r).__name__,
                        r,
                    )
                    clean_pages.append([])
                else:
                    clean_pages.append(r)
            fs.ocr_pages_attempted = len(page_results)
            fs.ocr_pages_failed = len(errors)

            # A systemic upstream OCR outage (e.g. circuit breaker open, an
            # OCR server that died mid-file) on ANY page fails the whole file
            # — never emit a partial result with silently blank pages. It is
            # an outage, so a resumed ``bibr batch`` runs the file again. An
            # UpstreamServiceError goes first: the serve answers 502 for it but
            # 422 for a raw transport error, so a breaker that opened after an
            # earlier page's refused connection must still give the 502.
            upstream = next((e for e in errors if isinstance(e, UpstreamServiceError)), None)
            if upstream is None:
                upstream = next((e for e in errors if _ocr_server_gone(e)), None)
            if upstream is not None:
                fs.set_error(
                    f"OCR upstream service failed: {redact_urls(str(upstream))}",
                    code="ocr_failed",
                    stage=self.name,
                    exc=upstream,
                    outage=True,
                )
                return

            # Only fail the file if ALL pages failed.
            if errors and len(errors) == len(page_results):
                fs.set_error(
                    f"OCR failed for all pages: {describe_error(errors[0])}",
                    code="ocr_failed",
                    stage=self.name,
                    exc=errors[0],
                )
                return

            first_idx = fs.page_indices[0] if fs.page_indices else 0
            # Pure synchronous regex/text work over every region of the file.
            # ``bibr serve`` runs one async worker, so leaving it on the loop
            # made it head-of-line blocking for every co-resident request —
            # and the repeated-content detector's tail is measured in seconds.
            fs.ocr_regions = _to_typed_regions(
                await asyncio.to_thread(
                    _postprocess_ocr_regions,
                    [[] for _ in range(first_idx)] + clean_pages,
                    ctx.settings,
                )
            )
            # Record OCR wall-time so remote/cloud-vision OCR shows up in
            # exported timings alongside the local path.
            fs.stage_times["ocr"] = time.monotonic() - fs_t0
        except Exception as e:  # noqa: BLE001
            fs.set_error(
                f"OCR failed: {describe_error(e)}", code="ocr_failed", stage=self.name, exc=e
            )
            logger.warning("OCR failed for %s", fs.path.name, exc_info=True)

    async def _run_remote(self, ctx, ocr_fn) -> None:
        files = self._eligible_files(ctx)
        if not files:
            return

        # Cross-file region semaphore — caps total regions in flight at
        # max_concurrent_files * concurrent_regions_per_file regardless of
        # how many files are in flight. Without this, N parallel files
        # multiply the effective region concurrency by N.
        #
        # ``max_concurrent_regions`` is the server-wide ceiling and must also
        # bind: this path ignored it entirely, so ``OCR_MAX_CONCURRENT_REGIONS``
        # did nothing on the local remote-OCR route, and the Apple-Silicon
        # auto-tune that sets it to 1 (vision prefill serializes on the GPU, so
        # client concurrency only grows queue latency toward the read timeout)
        # was defeated — 4 x 1 still admitted four concurrent regions.
        region_sem = asyncio.Semaphore(
            max(
                1,
                min(
                    ctx.settings.ocr.max_concurrent_files
                    * ctx.settings.ocr.concurrent_regions_per_file,
                    ctx.settings.ocr.max_concurrent_regions,
                ),
            )
        )
        file_sem = asyncio.Semaphore(ctx.settings.ocr.max_concurrent_files)

        async def _process(fs) -> None:
            async with file_sem:
                await self._ocr_one_file(fs, ctx, ocr_fn, region_sem)

        await asyncio.gather(*(_process(fs) for fs in files))

    async def _run_local(self, ctx, ocr_fn, ocr_sem) -> None:
        # Sequential across files; the single ``ocr_sem`` (built once in
        # ``_run`` and passed down) caps concurrent regions within each file and
        # is reused across files.
        for fs in self._eligible_files(ctx):
            await self._ocr_one_file(fs, ctx, ocr_fn, ocr_sem)
