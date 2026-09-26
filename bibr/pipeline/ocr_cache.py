"""Opt-in disk cache for OCR stage output (``bibr.pipeline.stages.ocr``).

Keyed on ``file_hash`` + page range + OCR backend/model + every setting that
shapes the cached artifacts + the bibr version + a format-version constant.
The key cannot see code changes between releases: when comparing source
revisions that touch rendering, layout, native text or OCR, use a fresh
``CACHE_OCR_DIR`` per revision. A complete entry
contains OCR regions plus the native-PDF artifacts needed by parsing, so the
local pipeline can skip render, layout, native analysis, OCR model load, and
inference entirely. Off by default (``CACHE_OCR``). Corrupt/unreadable entries
are treated as a miss and deleted, never an error.

Values are JSON — ``OcrRegionResult`` is a flat dataclass whose only
image-derived field (``image_b64``) is already a base64 string, so the region
list round-trips through ``to_dict``/``from_dict`` with no binary payload.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from bibr.ocr.profiles import (
    PADDLE_TABLE_RECOVERY_MAX_TOKENS,
    OcrRuntimeIdentity,
    resolve_ocr_profile,
)
from bibr.ocr.types import OcrRegionResult
from bibr.processing_warnings import ProcessingWarning

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.pipeline.context import RunConfig
    from bibr.pipeline.state import FileState

logger = logging.getLogger(__name__)

# Version 9 preserves OCR completion evidence and warnings. Earlier bundles
# cannot distinguish failed pages from empty pages, so invalidate them.
# Version 10 stores the warnings as ``{code, message}`` objects.
_CACHE_FORMAT_VERSION = 10


def _effective_settings(settings: GlobalSettings | None) -> GlobalSettings:
    if settings is not None:
        return settings
    from bibr.config import snapshot_settings

    return snapshot_settings()


def is_enabled(settings: GlobalSettings | None = None) -> bool:
    """Whether the OCR disk cache is turned on (``CACHE_OCR``)."""
    return _effective_settings(settings).cache.ocr


def cache_dir(settings: GlobalSettings | None = None) -> Path:
    """Resolve the cache directory: ``CACHE_OCR_DIR`` else ``$XDG_CACHE_HOME``/~."""
    configured = _effective_settings(settings).cache.ocr_dir
    if configured:
        return Path(configured).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "bibr" / "ocr"


def _include_figures(cfg: RunConfig, settings: GlobalSettings) -> bool:
    return cfg.include_figures if cfg.include_figures is not None else settings.FIGURE_IMAGES


def _effective_ref_seg(cfg: RunConfig, settings: GlobalSettings) -> str:
    from bibr.extract.ref_extractor import _resolve_ref_strategies

    return _resolve_ref_strategies(cfg.ref_seg_strategy, settings=settings)[0]


def _key(
    fs: FileState,
    cfg: RunConfig,
    identity: OcrRuntimeIdentity,
    settings: GlobalSettings | None = None,
) -> str:
    effective = _effective_settings(settings)
    layout = effective.layout
    profile = resolve_ocr_profile(
        explicit=identity.profile,
        backend=identity.backend,
        model=identity.model,
        max_tokens=effective.ocr.generation_max_tokens,
        temperature=effective.ocr.generation_temperature,
    )
    request = profile.request
    table_recovery_limit = (
        PADDLE_TABLE_RECOVERY_MAX_TOKENS
        if identity.backend == "serve-http" and identity.profile == "paddle"
        else 0
    )
    from bibr import __version__

    parts = [
        str(_CACHE_FORMAT_VERSION),
        # A release can change how the cached artifacts are produced without
        # anyone bumping _CACHE_FORMAT_VERSION; never reuse another release's.
        f"bibr={__version__}",
        fs.file_hash or "",
        "" if cfg.start_page is None else str(cfg.start_page),
        "" if cfg.end_page is None else str(cfg.end_page),
        identity.backend,
        cfg.ocr_url or "",
        identity.model,
        identity.profile,
        identity.normalizer_version,
        f"generation_max_tokens_text={request.max_tokens_for('text')}",
        f"generation_max_tokens_table={request.max_tokens_for('table')}",
        f"generation_max_tokens_formula={request.max_tokens_for('formula')}",
        f"paddle_table_recovery_max_tokens={table_recovery_limit}",
        f"generation_temperature={request.temperature}",
        # Region-crop resize budget. Sits beside the layout render knobs below
        # for the same reason: it decides which pixels reach the model, so a
        # change to it invalidates every cached OCR result.
        f"image_geometry={profile.image.cache_fingerprint()}",
        "figures=1" if _include_figures(cfg, effective) else "figures=0",
        f"dpi={layout.dpi}",
        f"max_render_pixels={layout.max_render_pixels}",
        f"max_render_dimension={layout.max_render_dimension}",
        f"detection_threshold={layout.detection_threshold}",
        f"nms_iou_same={layout.nms_iou_same}",
        f"nms_iou_diff={layout.nms_iou_diff}",
        f"large_image_area_landscape={layout.large_image_area_landscape}",
        f"large_image_area_portrait={layout.large_image_area_portrait}",
        f"section_classification_score={layout.section_classification_score}",
        f"containment_threshold={layout.containment_threshold}",
        f"overlap_resolver={layout.overlap_resolver}",
        f"read_order_fallback={layout.read_order_fallback}",
        f"max_pages={effective.pipeline.max_pages}",
        f"native_text={int(effective.ocr.native_text_enabled)}",
        f"native_text_min_chars={effective.ocr.native_text_min_chars}",
        f"native_text_min_printable_ratio={effective.ocr.native_text_min_printable_ratio}",
        f"native_text_header_footer={int(effective.ocr.native_text_header_footer)}",
        f"outline_headings={int(effective.pipeline.outline_headings)}",
        f"ref_seg={_effective_ref_seg(cfg, effective)}",
        # Which *weights* produced the cached artifacts. A complete entry lets
        # the pipeline skip layout detection and OCR inference outright, so the
        # pins that select those models decide its contents as surely as the
        # tuning knobs above do — and unlike the knobs, a model swap changes
        # every region in the bundle.
        #
        # ``identity.model`` does not cover this. For every served backend it
        # is the *alias* ("paddle-ocr-vl-1.6"), which is what the vLLM server
        # is launched with under `--served-model-name` while `--revision` takes
        # the pin below; the alias is unchanged by a re-pin.
        f"layout_model_id={layout.model_id}",
        f"layout_model_revision={layout.model_revision}",
        # The ONNX bundle is a separate artifact: a V3 -> V4 switch can move
        # only this pair while the torch pin stays put. ML_RUNTIME picks which
        # of the two pairs runs, so the two runtimes never share an entry.
        f"layout_onnx_model_id={layout.onnx_model_id}",
        f"layout_onnx_revision={layout.onnx_revision}",
        f"ml_runtime={effective.ml.runtime}",
        f"ocr_paddle_model={effective.ocr.paddle_model}",
        f"ocr_paddle_revision={effective.ocr.paddle_revision}",
    ]
    raw = "\x1f".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _path(
    fs: FileState,
    cfg: RunConfig,
    identity: OcrRuntimeIdentity,
    settings: GlobalSettings | None = None,
) -> Path:
    effective = _effective_settings(settings)
    return cache_dir(effective) / f"{_key(fs, cfg, identity, effective)}.json"


def _read_payload(
    fs: FileState,
    cfg: RunConfig,
    identity: OcrRuntimeIdentity,
    settings: GlobalSettings | None = None,
) -> dict | None:
    if not fs.file_hash:
        return None
    path = _path(fs, cfg, identity, settings)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != _CACHE_FORMAT_VERSION:
            raise ValueError("cache format version mismatch")
        return payload
    except Exception as e:  # noqa: BLE001 — any bad entry is a miss, not an error
        logger.info("OCR cache entry unreadable (%s); treating as miss", e)
        path.unlink(missing_ok=True)
        return None


def _decode_regions(payload: dict) -> list[list[OcrRegionResult]]:
    return [[OcrRegionResult.from_dict(d) for d in page] for page in payload["regions"]]


def _decode_quality(payload: dict) -> tuple[int, int, list[ProcessingWarning]]:
    evidence = payload["ocr_quality"]
    attempted, failed = evidence["pages_attempted"], evidence["pages_failed"]
    warnings = evidence["warnings"]
    if type(attempted) is not int or type(failed) is not int or not 0 <= failed <= attempted:
        raise ValueError("invalid OCR page completion evidence")
    if not isinstance(warnings, list):
        raise ValueError("invalid OCR warnings")
    return attempted, failed, [ProcessingWarning.from_dict(w) for w in warnings]


def _restore_quality(fs: FileState, quality: tuple[int, int, list[ProcessingWarning]]) -> None:
    fs.ocr_pages_attempted, fs.ocr_pages_failed, warnings = quality
    fs.warnings = list(dict.fromkeys([*fs.warnings, *warnings]))


def load(
    fs: FileState,
    cfg: RunConfig,
    identity: OcrRuntimeIdentity,
    settings: GlobalSettings | None = None,
) -> list[list[OcrRegionResult]] | None:
    """Return cached regions and restore completion evidence, or miss with None."""
    effective = _effective_settings(settings)
    payload = _read_payload(fs, cfg, identity, effective)
    if payload is None:
        return None
    try:
        regions = _decode_regions(payload)
        quality = _decode_quality(payload)
    except Exception as e:  # noqa: BLE001 — any bad entry is a miss, not an error
        logger.info("OCR cache entry unreadable (%s); treating as miss", e)
        _path(fs, cfg, identity, effective).unlink(missing_ok=True)
        return None
    _restore_quality(fs, quality)
    return regions


def load_bundle(
    fs: FileState,
    cfg: RunConfig,
    identity: OcrRuntimeIdentity,
    settings: GlobalSettings | None = None,
) -> bool:
    """Restore every pre-parse artifact into *fs* on a complete cache hit."""
    effective = _effective_settings(settings)
    payload = _read_payload(fs, cfg, identity, effective)
    if payload is None:
        return False
    try:
        artifacts = payload["artifacts"]
        native_metadata = artifacts["native_metadata"]
        ref_line_geometry = artifacts["ref_line_geometry"]
        outline_data = artifacts["pdf_outline"]
        inspection_data = artifacts.get("pdf_inspection")
        if outline_data is None:
            pdf_outline = None
        else:
            from bibr.input.pdf_outline import OutlineItem

            pdf_outline = [OutlineItem(**item) for item in outline_data]
        regions = _decode_regions(payload)
        quality = _decode_quality(payload)
        from bibr.ocr.pdf_inspection import inspection_from_dict

        pdf_inspection = inspection_from_dict(
            inspection_data,
            metadata=native_metadata,
            outline=pdf_outline,
            reference_lines=ref_line_geometry,
        )
    except Exception as e:  # noqa: BLE001 — any bad entry is a miss, not an error
        logger.info("OCR cache bundle unreadable (%s); treating as miss", e)
        _path(fs, cfg, identity, effective).unlink(missing_ok=True)
        return False

    # Publish only after the entire payload validates, so a corrupt entry can
    # never leave partially-restored state behind.
    fs.ocr_regions = regions
    fs.native_metadata = native_metadata
    fs.ref_line_geometry = ref_line_geometry
    fs.pdf_outline = pdf_outline
    fs.pdf_inspection = pdf_inspection
    _restore_quality(fs, quality)
    return True


def store(
    fs: FileState,
    cfg: RunConfig,
    identity: OcrRuntimeIdentity,
    regions: list[list[OcrRegionResult]],
    settings: GlobalSettings | None = None,
) -> None:
    """Atomically write OCR and native-PDF artifacts (best-effort)."""
    if not fs.file_hash:
        return
    path = _path(fs, cfg, identity, settings)
    payload = {
        "version": _CACHE_FORMAT_VERSION,
        "regions": [[r.to_dict() for r in page] for page in regions],
        "ocr_quality": {
            "pages_attempted": fs.ocr_pages_attempted,
            "pages_failed": fs.ocr_pages_failed,
            "warnings": [w.to_dict() for w in fs.warnings],
        },
        "artifacts": {
            "native_metadata": fs.native_metadata,
            "ref_line_geometry": fs.ref_line_geometry,
            "pdf_outline": (
                [asdict(item) for item in fs.pdf_outline] if fs.pdf_outline is not None else None
            ),
            "pdf_inspection": _inspection_payload(fs),
        },
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError) as e:
        logger.info("OCR cache write failed (%s); continuing uncached", e)
        if "tmp" in locals():
            tmp.unlink(missing_ok=True)


def _inspection_payload(fs: FileState) -> dict | None:
    from bibr.ocr.pdf_inspection import inspection_to_dict

    return inspection_to_dict(fs.pdf_inspection)
