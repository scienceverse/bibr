"""Pipeline context carrying per-run state shared across stages."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.pipeline.progress import ProgressTracker
    from bibr.pipeline.resources import ResourceManager
    from bibr.pipeline.state import FileState

logger = logging.getLogger(__name__)

ConsolidationMode = Literal["off", "fill", "replace"]
FigureExtractTier = Literal["off", "meta"]
MemoryMode = Literal["aggressive", "balanced", "keep_all"]
RefParseStrategy = Literal["ner", "llm", "llm-chunked", "off"]
RefSegStrategy = Literal["geom", "region", "llm", "crf"]


_figure_tier_warned = False


def _warn_figure_tier_inert(tier: FigureExtractTier) -> None:
    """Warn once per process that a non-``off`` figure tier does nothing.

    ``FigureOptions`` documents a vision-LLM stage that attaches
    ``figure[].analysis``, and the whole config surface for it exists
    (``FIG_MODEL``, ``FIG_MAX_FIGURES``, ...) — but no stage reads any of it.
    Accepting ``FIG_EXTRACT=meta`` and producing no ``analysis`` looked like a
    silent extraction failure; say so instead.
    """
    global _figure_tier_warned
    if _figure_tier_warned:
        return
    _figure_tier_warned = True
    logger.warning(
        "Figure extraction tier %r is configured but not implemented — no "
        "figure analysis is produced. Unset FIG_EXTRACT / figure_extract to "
        "silence this.",
        tier,
    )


def _default_settings_snapshot() -> GlobalSettings:
    """Keep direct context construction isolated without eagerly loading config."""
    from bibr.config import snapshot_settings

    return snapshot_settings()


@dataclass
class RunConfig:
    """Per-chunk run parameters (immutable within a chunk)."""

    memory_mode: MemoryMode = "balanced"
    ocr_backend: str = "paddle"
    ocr_url: str | None = None
    ocr_model: str | None = None
    ocr_profile: str | None = None
    device: str | None = None
    crossref: bool = True
    equations: bool = True
    start_page: int | None = None
    end_page: int | None = None
    llm_backend: str = "cloud"
    include_figures: bool | None = None
    """Emit base64 figure images. ``None`` means fall back to ``Settings.FIGURE_IMAGES``."""
    figure_extract: FigureExtractTier | None = None
    """Figure-analysis tier. ``None`` defers to ``Settings.fig.extract``."""
    include_regions: bool = False
    """Emit the ``_regions`` debug payload (per-region layout: bbox, font,
    content, etc.). Off by default — the field is large and not consumed by
    standard downstream consumers like Metacheck."""
    include_region_meta: bool = False
    """Emit the per-text underscore region metadata (``_bbox_2d``,
    ``_font_size``, ``_region_type``, …; v4 training features). Off by
    default — like ``_regions``, not part of the
    Metacheck-facing API."""
    no_llm: bool = False
    """When True, skip all LLM-driven steps (section classification fallback,
    implicit section detection, metadata extraction, equation extraction,
    citation linking) and Crossref enrichment. Produces structural output
    only — intended for training-data preprocessing."""
    consolidate: ConsolidationMode | None = None
    """Per-run consolidation mode ("off"/"fill"/"replace"). ``None`` defers
    to ``Settings.crossref.consolidate``."""
    ref_seg_strategy: RefSegStrategy | None = None
    """Per-run reference segmentation strategy ("geom"/"llm"/"crf").
    ``None`` defers to ``Settings.REF_SEG_STRATEGY``."""
    ref_parse_strategy: RefParseStrategy | None = None
    """Per-run reference parse strategy ("ner"/"llm"). ``None`` defers to
    ``Settings.REF_PARSE_STRATEGY`` / the legacy alias."""

    def figure_extract_tier(self, settings: GlobalSettings) -> FigureExtractTier:
        """Resolve the per-run figure-analysis tier.

        NOTE: no figure-analysis stage consumes this yet — see
        :meth:`emit_figures`. Anything other than ``"off"`` is accepted and
        then does nothing, so warn rather than let it pass silently.
        """
        tier = self.figure_extract if self.figure_extract is not None else settings.fig.extract
        if tier != "off":
            _warn_figure_tier_inert(cast("FigureExtractTier", tier))
        return cast("FigureExtractTier", tier)

    def emit_figures(self, settings: GlobalSettings) -> bool:
        """Whether OCR must retain figure crops for output or analysis.

        Currently unused. It exists for the figure-analysis tier described in
        :class:`~bibr.config.FigureOptions`, whose stage is not built: the two
        sites that decide crop retention (``OcrStage`` and the OCR cache key)
        deliberately keep asking ``include_figures``/``FIGURE_IMAGES`` alone,
        because retaining crops for a consumer that does not exist would cost
        memory and cache-key churn for nothing. Wire this in together with the
        stage that reads the crops, not before.
        """
        emit_images = self.include_figures
        if emit_images is None:
            emit_images = settings.FIGURE_IMAGES
        return emit_images or self.figure_extract_tier(settings) != "off"


@dataclass
class StageSignals:
    """Cross-stage coordination signals, shared by reference across sub-contexts.

    ``InterleavedRenderOcrStage`` creates per-window sub-contexts via
    ``dataclasses.replace(ctx, file_states=group)``; plain dataclass fields on
    ``PipelineContext`` would NOT propagate across those copies (each
    ``replace`` gets independent field slots). This object is created once per
    chunk and threaded through every ``replace`` copy by reference, so a
    signal written in one window is visible in later windows and the outer
    context.
    """

    any_needs_ocr: bool = True
    preloading_ocr: bool = False
    ocr_init_error: BaseException | None = None
    defer_ocr_teardown: bool = False
    ocr_page_window: bool = False
    """A partial document: defer OCR cache writes and success gates until all pages finish."""


@dataclass
class PipelineContext:
    """Mutable per-chunk context threaded through stages."""

    file_states: list[FileState]
    progress: ProgressTracker
    resources: ResourceManager
    config: RunConfig
    settings: GlobalSettings = field(default_factory=_default_settings_snapshot)
    """Concrete runtime settings owned by the enclosing pipeline."""
    scratch: dict[str, Any] = field(default_factory=dict)
    """Free-form scratch for inter-stage signals (e.g. ``"stage_timings"``)."""
    signals: StageSignals = field(default_factory=StageSignals)
    """Typed cross-stage coordination signals; see ``StageSignals``."""

    def alive(self) -> list[FileState]:
        """Return file states that have not errored."""
        return [fs for fs in self.file_states if fs.error is None]

    def free_after_stage(self, stage_name: str) -> None:
        """Reclaim per-file intermediate memory at known stage boundaries.

        Centralizes freeing policy so that a partial-stage failure still
        runs the relevant ``free_*`` calls on every file (errored or alive).
        Stages no longer need to call ``fs.free_*`` themselves. A file that
        has errored (and has no exported result to preserve) is fully freed
        at the next boundary — it will never be exported, so holding its
        buffers until GC just pins memory.
        """
        for fs in self.file_states:
            if stage_name == "ocr":
                fs.free_pre_ocr()
            elif stage_name == "parse":
                fs.free_pre_parse()
            if fs.error is not None and fs.result_json is None:
                fs.free_all()
