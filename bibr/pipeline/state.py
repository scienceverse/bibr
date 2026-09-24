"""Per-file pipeline state.

``FileState`` is the carrier every stage reads from and writes to. It lives
in the generic ``bibr.pipeline`` layer — stages, the ``Pipeline`` base class,
and ``PipelineContext`` all depend on it, and none of them may import from
``bibr.local`` (the single-machine orchestrator is a *consumer* of this
layer, not a dependency of it). ``bibr.local.pipeline`` re-exports it for
backward compatibility.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from bibr.input.docx_native import DocxParser
    from bibr.input.epub_native import EpubParser
    from bibr.input.html_native import HtmlParser
    from bibr.input.jats_native import JatsParser
    from bibr.input.pdf_outline import OutlineItem
    from bibr.ocr.pdf_inspection import PdfInspection
    from bibr.ocr.types import OcrRegionResult
    from bibr.paper import Paper
    from bibr.paper_contents import PaperContents
    from bibr.pipeline.artifacts import ArtifactDisposition, ArtifactSink, RunState
    from bibr.pipeline.identity import DoiSelection, ExpectedIdentity
    from bibr.processing_warnings import ProcessingWarning


@dataclass
class FileState:
    """Tracks per-file intermediate state through the pipeline.

    Fields are populated progressively and nulled after each stage to
    bound memory.  ``error`` gates participation in subsequent stages.
    """

    path: Path
    paper_id: str | None = None
    # sha256[:16] of the input bytes; set by ValidateStage. Used for LLM usage
    # attribution, training-data keys, and cache identity.
    file_hash: str | None = None
    # Full SHA-256 supplied by streaming upload readers. ValidateStage derives
    # ``file_hash`` from it and hashes bytes only when this is absent.
    content_sha256: str | None = None
    expected_identity: "ExpectedIdentity | None" = None
    manifest_output_path: Path | None = None
    doi_selection: "DoiSelection | None" = None
    artifact_sink: "ArtifactSink | None" = None
    artifact_started: bool = False
    core_sha256: str | None = None
    artifact_disposition: "ArtifactDisposition | None" = None
    enrichment_state: "RunState | None" = None
    enrichment_warnings: "list[ProcessingWarning]" = field(default_factory=list)
    enrichment_detail: str | None = None

    # Populated during processing — nulled progressively
    pdf_bytes: bytes | None = None
    page_images: "list[PILImage] | None" = None
    page_indices: list[int] | None = None
    layout_results: list[list[dict[str, Any]]] | None = None
    pdf_inspection: "PdfInspection | None" = None
    # Typed Region IR: OcrStage emits OcrRegionResult objects (wire-format
    # dicts stay stage-internal); ParseSegmentStage hands them to PDFParser.
    ocr_regions: "list[list[OcrRegionResult]] | None" = None
    # Pages OCR attempted, and pages that failed wholesale. A page that raises
    # substitutes an empty region list, contributing to neither side of the
    # region-level success ratio — so a run where most pages died outright
    # scored 100% and sailed through ``_check_ocr_success``. Recorded per file
    # by ``OcrStage._ocr_one_file`` so the gate can see them.
    ocr_pages_attempted: int = 0
    ocr_pages_failed: int = 0
    ref_line_geometry: list[dict[str, Any]] | None = None
    # PDF outline (bookmarks) harvested by NativeTextStage when
    # ``Settings.pipeline.outline_headings`` is on; handed to PDFParser as an
    # authoritative heading-hierarchy signal. Internal-only, never exported.
    pdf_outline: "list[OutlineItem] | None" = None
    # Guarded PDF doc-info metadata (title/doi/keywords) harvested by
    # NativeTextStage; consumed as post_parse's fill-empty backstop.
    native_metadata: dict[str, Any] | None = None
    # Reusable HTML DOM or parsed ePub package produced during validation.
    native_validation_artifact: object | None = None
    contents: "PaperContents | None" = None
    paper: "Paper | None" = None
    result_json: dict[str, Any] | None = None
    # Native parser (DOCX, JATS-XML, HTML, or ePub) held across the handling stage →
    # ParseSegmentStage so the parser instance can be reused for segmentation.
    _native_parser: "DocxParser | JatsParser | HtmlParser | EpubParser | None" = None

    # Error tracking
    error: str | None = None
    error_code: str | None = None  # ErrorCode value for structured reporting
    failed_stage: str | None = None
    original_error: BaseException | None = None
    stage_times: dict = field(default_factory=dict)
    warnings: "list[ProcessingWarning]" = field(default_factory=list)

    def free_pre_ocr(self):
        """Free data consumed by OCR stage."""
        self.pdf_bytes = None
        self.page_images = None
        self.layout_results = None
        self.pdf_inspection = None

    def free_pre_parse(self):
        """Free data consumed by parse stage."""
        self.ocr_regions = None
        self.ref_line_geometry = None
        self.pdf_outline = None
        # page_indices is consumed in OCR; drop it now to reclaim memory
        # across large-chunk runs.
        self.page_indices = None
        # Native parser is held across its handling stage →
        # ParseSegmentStage so the parser can be reused; release after parse.
        self._native_parser = None

    def set_error(
        self,
        message: str,
        *,
        code: str | None = None,
        stage: str | None = None,
        exc: BaseException | None = None,
    ):
        """Set structured error info.

        ``exc`` captures the originating exception so it can be chained via
        ``raise ... from`` in the outer pipeline boundary, preserving the
        traceback for structured loggers and debuggers.
        """
        self.error = message
        self.error_code = code
        self.failed_stage = stage
        if exc is not None:
            self.original_error = exc

    def free_all(self):
        """Free all intermediate state after export."""
        # A paper freed before its enrich stage ran (it errored) still owns an
        # in-flight enrichment prefetch; cancel it rather than orphan it.
        if self.paper is not None:
            from bibr.pipeline.enrich_prefetch import cancel_prefetch

            cancel_prefetch(self.paper)
        self.pdf_bytes = None
        self.page_images = None
        self.page_indices = None
        self.layout_results = None
        self.pdf_inspection = None
        self.ocr_regions = None
        self.ref_line_geometry = None
        self.pdf_outline = None
        self.native_validation_artifact = None
        self.contents = None
        self.paper = None


def _alive(file_states: list[FileState]) -> list[FileState]:
    """Return file states that have not errored."""
    return [fs for fs in file_states if fs.error is None]
