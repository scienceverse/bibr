"""The text-layer OCR backend: read born-digital PDFs with no OCR model.

``--ocr text-layer`` (``OCR_BACKEND=text-layer``) reads every layout region
from the PDF's embedded text layer instead of sending page crops to an OCR
model, so a machine that cannot run one (CPU only, no llama.cpp) can still
process born-digital PDFs. ``OCR_FALLBACK_TO_TEXT_LAYER=auto`` switches to it
when no runtime of the automatic OCR backend can start on the machine.

What changes against the native-text pass that runs before OCR:

- every region that OCR would read is filled, not only running text:
  headers and footers, captions (snapped to the caption's own lines, so axis
  labels above "Figure 2" stay out), display formulas and their numbers as
  plain text (no LaTeX), and tables rebuilt from glyph positions
  (:mod:`bibr.ocr.text_layer_tables`);
- short regions need three characters, as headings do in the native pass;
- each page passes the gates in :mod:`bibr.ocr.text_layer_quality` or loses
  all its text, since there is no OCR to correct a wrong layer: a scan, a
  broken font encoding. Such a page is recorded as
  ``PdfPageInspection.text_layer_ok = False`` with a reason, and page triage
  (:mod:`bibr.scan.page_kind`) counts it as a scan or a broken text layer;
- regions left without text are marked ``_text_layer_unread``; the OCR stage
  emits them empty and fails a file whose text regions are mostly unread
  ("scanned PDF needs OCR").

The client registered here never starts a process and never recognizes an
image: the OCR stage does not call it.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from bibr.ocr.registry import TEXT_LAYER_BACKEND, TEXT_LAYER_MODEL, register
from bibr.processing_warnings import ProcessingWarning, WarningCode

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.ocr.text_layer_quality import PageGateResult

__all__ = [
    "TEXT_LAYER_BACKEND",
    "TEXT_LAYER_MODEL",
    "TextLayerOcrClient",
    "TextLayerPage",
    "TextLayerRead",
    "apply_text_layer_read",
    "is_text_layer_mode",
    "note_text_layer_fallback",
    "read_file_text_layer",
    "read_text_layer",
    "text_layer_fallback_reason",
    "text_layer_requested",
]

logger = logging.getLogger(__name__)

# A text-layer region may carry one unmapped glyph, or one per this many characters.
_CHARS_PER_UNMAPPED_GLYPH = 100
_UNMAPPED_GLYPH = "\ufffd"


class TextLayerOcrClient:
    """The registry entry for ``text-layer``: always ready, never asked to OCR."""

    name: ClassVar[str] = TEXT_LAYER_BACKEND

    def __init__(self, **_kw: object) -> None:
        pass

    @property
    def loaded(self) -> bool:
        return True

    async def recognize(self, image: Any, prompt: str) -> str:  # noqa: ARG002
        raise RuntimeError(
            "the text-layer OCR backend reads the PDF text layer and has no OCR model; "
            "no region should reach recognize()"
        )

    async def wait_for_server(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


register(TextLayerOcrClient)


# --- Mode --------------------------------------------------------------------


def text_layer_requested(config: Any, settings: GlobalSettings) -> bool:
    """Whether the run asked for the text-layer backend (``--ocr text-layer``).

    Mirrors the runtime identity's request rule: with an ``ocr_url`` a
    configured ``OCR_BACKEND=text-layer`` gives way to the named server, an
    explicit request does not.
    """
    from bibr.ocr.registry import url_request

    explicit = getattr(config, "ocr_backend", None)
    configured = getattr(getattr(settings, "ocr", None), "backend", None)
    if getattr(config, "ocr_url", None):
        return url_request(explicit, configured) == TEXT_LAYER_BACKEND
    return (explicit or configured) == TEXT_LAYER_BACKEND


def text_layer_fallback_reason(resources: Any) -> str | None:
    """Why the automatic OCR backend reads the text layer here, or None.

    ``OCR_FALLBACK_TO_TEXT_LAYER=auto`` on a machine where no OCR runtime can
    start (``ResourceManager.check_text_layer_fallback``).
    """
    reason = getattr(resources, "ocr_text_layer_fallback", None)
    if isinstance(reason, str):
        return reason
    # Looked up on the class: a stand-in for the resources (a test double)
    # answers only if it defines the check.
    check = getattr(type(resources), "check_text_layer_fallback", None)
    if not callable(check):
        return None
    reason = check(resources)
    return reason if isinstance(reason, str) else None


def is_text_layer_mode(ctx: Any) -> bool:
    """Whether this pipeline context reads PDFs from the text layer alone."""
    scratch = getattr(ctx, "scratch", None)
    identity = scratch.get("ocr_runtime_identity") if isinstance(scratch, dict) else None
    if getattr(identity, "backend", None) == TEXT_LAYER_BACKEND:
        return True
    if text_layer_fallback_reason(getattr(ctx, "resources", None)) is not None:
        return True
    return text_layer_requested(getattr(ctx, "config", None), ctx.settings)


# --- Reading -----------------------------------------------------------------

# Region task types OCR never reads (figures, page furniture).
_UNREAD_TASKS = frozenset({"skip", "abandon"})
_FORMULA_LABELS = frozenset({"formula", "display_formula", "inline_formula"})
_CAPTION_LABELS = frozenset({"figure_title", "table_title", "chart_title"})
# Formula numbers ("(3)") and one-symbol formulas are shorter than the
# three-character allowance every other region gets.
_FORMULA_MIN_CHARS = 1
# Captions lines above the caption's first line ("Figure 2") smaller than
# this share of its font size are figure text (axis labels, legends).
_CAPTION_SMALL_FONT_RATIO = 0.92
_CAPTION_START_RE = re.compile(
    r"^\s*(?:(?i:supplementary|extended\s+data|supporting)\s+)?"
    r"(?i:fig(?:ure)?s?|tab(?:le)?s?|scheme|chart|plate|exhibit|abb(?:ildung)?|tabelle|"
    r"tableau|tabla|tabela|graph)\.?\s*[\dA-Z]"
)

#: Page verdict reasons. ``no_text_layer``, ``invisible_ocr_layer`` and
#: ``low_coverage`` describe a scanned page; the others a broken text layer.
SCAN_REASONS = frozenset({"no_text_layer", "invisible_ocr_layer", "low_coverage"})
_REASON_TEXT = {
    "no_text_layer": "no text layer (a scanned page)",
    "invisible_ocr_layer": "the text layer is invisible (an OCR layer over a scanned image)",
    "low_coverage": "the text layer covers too little of the page's text (a scanned page)",
    "corrupt_text": "most of its text is corrupt (a broken font encoding)",
    "implausible_text": "the text layer does not read as words (a broken font encoding)",
}


@dataclass(frozen=True)
class TextLayerPage:
    """One page's text-layer verdict. ``reason`` is None when the page was read."""

    index: int
    ok: bool
    reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict, compare=False)


@dataclass
class TextLayerRead:
    """What :func:`read_text_layer` returns: the filled layout and its verdicts."""

    layout_results: list[list[dict[str, Any]]]
    pages: list[TextLayerPage]
    warnings: list[ProcessingWarning]
    filled: Counter = field(default_factory=Counter)
    unread: Counter = field(default_factory=Counter)
    tables_rebuilt: int = 0
    formulas_read: int = 0
    private_use_dropped: int = 0
    private_use_inside: int = 0
    unmapped_glyphs: int = 0


@dataclass
class _PageText:
    """One page's text layer, detached from PDFium."""

    index: int
    crop_box: tuple[float, float, float, float]
    rotation: int
    char_count: int
    invisible: bool
    records: list[tuple[str, float, float, bool]]
    # (char, layout-space box) per char in text-layer order, whitespace included.
    chars: list[tuple[str, tuple[float, float, float, float]]]
    # (text, layout-space box, font size) per printed line.
    lines: list[tuple[str, tuple[float, float, float, float], float]]

    @property
    def glyphs(self) -> list[tuple[str, tuple[float, float, float, float]]]:
        """The visible glyphs: no whitespace, no empty boxes."""
        return [
            (ch, box)
            for ch, box in self.chars
            if ch and not ch.isspace() and box[2] > box[0] and box[3] > box[1]
        ]


def _region_task(region: dict[str, Any]) -> str:
    task = region.get("task_type")
    if task:
        return str(task)
    from bibr.layout_utils import _LABEL_TO_TASK

    return _LABEL_TO_TASK.get(str(region.get("label") or ""), "text")


def _centre_in(box: tuple[float, float, float, float], bbox: Sequence[float]) -> bool:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def _load_page(doc: Any, page_index: int) -> _PageText:
    """Read one page's text layer under the (caller-held) PDFium lock."""
    from bibr.ocr.native_text import (
        _build_page_char_records,
        _invisible_text_share,
        _page_crop_box,
        _page_rotation,
        _pdf_points_to_normalized_bbox,
        _share_qualifies,
        open_text_page,
    )
    from bibr.ocr.pdf_inspection import _break_wrapped_lines
    from bibr.ocr.ref_geometry import _extract_page_chars, group_chars_into_lines

    page = doc[page_index]
    try:
        crop_box = _page_crop_box(page)
        rotation = _page_rotation(page)
        textpage = open_text_page(page)
        try:
            char_count = textpage.count_chars()
            if not char_count:
                return _PageText(page_index, crop_box, rotation, 0, False, [], [], [])
            try:
                # Mostly invisible text is a hidden OCR layer, not printed text.
                # The native pass also asks for a page-sized image; scans
                # cropped to their text block have smaller ones (dev scans:
                # 0.55 and 0.67 of the page), and no OCR reads them here.
                invisible = _share_qualifies(_invisible_text_share(textpage))
            except Exception:  # noqa: BLE001 - an unreadable check keeps the layer
                logger.debug("Invisible-layer check failed on page %d", page_index, exc_info=True)
                invisible = False
            if invisible:
                return _PageText(page_index, crop_box, rotation, char_count, True, [], [], [])
            records = _build_page_char_records(textpage)
            chars = _extract_page_chars(textpage)
        finally:
            textpage.close()
    finally:
        page.close()

    def to_layout(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return _pdf_points_to_normalized_bbox(box, crop_box, rotation)

    layout_chars = [(ch, to_layout(box)) for ch, box in chars if ch]
    lines = [
        (line.text, to_layout((line.x0, line.y_bottom, line.x1, line.y_top)), line.font_size)
        for line in group_chars_into_lines(_break_wrapped_lines(chars), page_index)
    ]
    return _PageText(
        page_index, crop_box, rotation, char_count, False, records, layout_chars, lines
    )


def _snapped_caption_box(page: _PageText, bbox: Sequence[float]) -> list[float] | None:
    """The caption region's box shrunk to its caption lines, or None to keep it.

    A caption box often reaches up into the figure. When the region's lines
    include a caption start ("Figure 2", "Table 1"), the smaller-font lines
    above it are figure text and are left out.
    """
    inside = sorted(
        (line for line in page.lines if _centre_in(line[1], bbox)), key=lambda line: line[1][1]
    )
    start = next((k for k, line in enumerate(inside) if _CAPTION_START_RE.match(line[0])), None)
    if not start:
        return None
    caption_size = inside[start][2]
    kept = [
        line
        for k, line in enumerate(inside)
        if k >= start or line[2] >= _CAPTION_SMALL_FONT_RATIO * caption_size
    ]
    if len(kept) == len(inside):
        return None
    return [
        min(line[1][0] for line in kept) - 1.0,
        min(line[1][1] for line in kept) - 1.0,
        max(line[1][2] for line in kept) + 1.0,
        max(line[1][3] for line in kept) + 1.0,
    ]


def _read_text(page: _PageText, bbox: Sequence[float]) -> str:
    from bibr.ocr.native_text import _normalized_bbox_to_pdf_points, _reconstruct_text_from_records

    left, bottom, right, top = _normalized_bbox_to_pdf_points(
        list(bbox), page.crop_box, page.rotation
    )
    return _reconstruct_text_from_records(page.records, left, bottom, right, top)


def _drop_private_use(text: str) -> tuple[str, int, int]:
    """Remove private-use code points: (text, dropped, dropped inside a word).

    Publisher fonts put icons (an ORCID badge, an e-mail or affiliation
    mark) and unmapped glyphs in the private-use area. They carry no text.
    One followed directly by a letter or digit may stand for a letter of that
    word (a ligature) or for a sign before a number (a minus), so it is counted
    separately as *inside a word*: the text may have changed there. Rejecting
    the region instead loses whole paragraphs and reference lists for a few glyphs.
    """
    if not any(unicodedata.category(ch) == "Co" for ch in text):
        return text, 0, 0
    kept: list[str] = []
    dropped = inside = 0
    for index, ch in enumerate(text):
        if unicodedata.category(ch) != "Co":
            kept.append(ch)
            continue
        dropped += 1
        if index + 1 < len(text) and text[index + 1].isalnum():
            inside += 1
    return "".join(kept), dropped, inside


def _mark_unmapped_glyphs(text: str) -> tuple[str, int]:
    """Show rare unmapped glyphs as U+FFFD: (text, how many), or (text, 0) if none or too many.

    A font with no character map for a glyph hands out its raw code, which reads
    as a control character. That is the mark of a broken font when there are many,
    but one or two in a paragraph are single symbols: a ligature, a superscript
    minus, a degree sign. What they stood for depends on the font and cannot be
    recovered, and dropping them changes the text without a trace ("W m-1" reads
    as "W m1"), so the mark stays in their place. No OCR follows on this route, so
    rejecting the region would lose the whole paragraph for one symbol.
    """
    from bibr.ocr.native_text import _is_disallowed_control_char

    count = sum(1 for ch in text if _is_disallowed_control_char(ch))
    if not count or count > max(1, len(text) // _CHARS_PER_UNMAPPED_GLYPH):
        return text, 0
    return "".join(_UNMAPPED_GLYPH if _is_disallowed_control_char(ch) else ch for ch in text), count


def _unmapped_glyph_warning(count: int) -> ProcessingWarning | None:
    """The document's warning for *count* glyphs shown as U+FFFD, if any."""
    if not count:
        return None
    return ProcessingWarning(
        WarningCode.TEXT_LAYER_GLYPHS_UNMAPPED,
        f"{count} glyph(s) in the PDF text layer have no character mapping and are shown as "
        "U+FFFD; each may have stood for a letter or a symbol (a ligature, a minus sign, a "
        "degree sign)",
    )


def _accept(region: dict[str, Any]) -> None:
    """Mark *region* as read from the text layer, clearing the native pass's rejection."""
    region["_native_text_used"] = True
    # The native-text pass may have set the region aside (private-use glyphs)
    # for OCR to read; with no OCR the text layer's own reading stands, so
    # the export must not say OCR read it instead.
    for key in ("_native_gate", "_native_text_candidate", "_native_text_rejection_reason"):
        region.pop(key, None)


def _page_aspect(page: _PageText) -> float | None:
    """The rendered page's height over its width, or None for an empty box."""
    left, bottom, right, top = page.crop_box
    width, height = right - left, top - bottom
    if page.rotation % 180:
        width, height = height, width
    if width <= 0 or height <= 0:
        return None
    return height / width


_TableChar = tuple[str, float, float, float, float]


def _clean_table_chars(chars: list[_TableChar], region: dict[str, Any]) -> list[_TableChar]:
    """A table region's chars with private-use glyphs dropped and unmapped ones marked.

    As for text (:func:`_drop_private_use`), a private-use glyph is an icon or
    an unmapped glyph and is dropped, counted as *inside a word* when a letter
    or digit follows it. A control character is a glyph with no character map
    (a bracket or a sign in a math font) and shows as U+FFFD in its cell
    however many there are: a table is numbers and labels, and rejecting it for
    its symbols would lose every cell.
    """
    from bibr.ocr.native_text import _is_disallowed_control_char

    region.pop("_private_use_dropped", None)
    region.pop("_private_use_inside", None)
    region.pop("_unmapped_glyphs", None)
    kept: list[_TableChar] = []
    dropped = inside = unmapped = 0
    for index, glyph in enumerate(chars):
        ch = glyph[0]
        if ch and unicodedata.category(ch) == "Co":
            dropped += 1
            following = chars[index + 1][0] if index + 1 < len(chars) else ""
            inside += bool(following) and following.isalnum()
            continue
        # The separator controls (U+001C..U+001F) count as whitespace and only end a word.
        if ch and not ch.isspace() and _is_disallowed_control_char(ch):
            unmapped += 1
            glyph = (_UNMAPPED_GLYPH, *glyph[1:])
        kept.append(glyph)
    if dropped:
        region["_private_use_dropped"] = dropped
        region["_private_use_inside"] = inside
    if unmapped:
        region["_unmapped_glyphs"] = unmapped
    return kept


def _fill_region(
    page: _PageText,
    region: dict[str, Any],
    *,
    min_printable_ratio: float,
) -> str:
    """Fill one region from the text layer: "filled", "rejected" or "empty"."""
    from bibr.input.consolidate_text import fix_ocr_artifacts
    from bibr.ocr.native_text import _SHORT_NATIVE_TEXT_MIN_CHARS, _is_native_text_usable
    from bibr.ocr.text_layer_tables import build_table_html

    label = str(region.get("label") or "")
    bbox = region.get("bbox_2d")
    if not bbox or len(bbox) < 4:
        return "empty"
    if _region_task(region) == "table":
        # Whitespace stays in, wherever it sits: it only ever ends a word.
        chars = _clean_table_chars(
            [(ch, *box) for ch, box in page.chars if ch.isspace() or _centre_in(box, bbox)], region
        )
        text = "".join(" " if g[0].isspace() else g[0] for g in chars)
        if region.get("_unmapped_glyphs"):
            # The other gates judge the text around the marks, which are not its own fault.
            text = text.replace(_UNMAPPED_GLYPH, "")
        if not _is_native_text_usable(text, min_printable_ratio):
            region["_native_gate"] = "usability"
            return "rejected"
        table = build_table_html(chars, aspect=_page_aspect(page))
        if table is None:
            return "empty"
        region["content"] = table
        _accept(region)
        region["_text_layer_table"] = True
        region["_text_layer_kind"] = "table_rebuilt"
        return "filled"
    read_box = bbox
    if label in _CAPTION_LABELS:
        read_box = _snapped_caption_box(page, bbox) or bbox
    text = _read_text(page, read_box)
    if "\x02" in text:
        text = fix_ocr_artifacts(text)
    if not text.strip():
        return "empty"
    text, dropped, inside = _drop_private_use(text)
    region.pop("_private_use_dropped", None)
    region.pop("_private_use_inside", None)
    if dropped:
        region["_private_use_dropped"] = dropped
        region["_private_use_inside"] = inside
    if not text.strip():
        return "empty"
    is_formula = label in _FORMULA_LABELS or _region_task(region) == "formula"
    min_chars = (
        _FORMULA_MIN_CHARS
        if is_formula or label == "formula_number"
        else _SHORT_NATIVE_TEXT_MIN_CHARS
    )
    if len(text.strip()) < min_chars:
        return "empty"
    region.pop("_unmapped_glyphs", None)
    if not _is_native_text_usable(text, min_printable_ratio):
        # A formula stays unread: its control characters are usually math symbols.
        marked, unmapped = (text, 0) if is_formula else _mark_unmapped_glyphs(text)
        # The other gates judge the text around the marks, which are not its own fault.
        if not unmapped or not _is_native_text_usable(
            marked.replace(_UNMAPPED_GLYPH, ""), min_printable_ratio
        ):
            region["_native_gate"] = "usability"
            return "rejected"
        text = marked
        region["_unmapped_glyphs"] = unmapped
    _accept(region)
    if is_formula:
        # Plain text: the parser keeps it as printed, without LaTeX.
        text = text.strip()
        region["_formula_source"] = "text_layer"
        region["_text_layer_kind"] = "formula_plain"
    region["content"] = text
    return "filled"


def _judge(
    page: _PageText,
    regions: list[dict[str, Any]],
    outcomes: dict[int, str],
) -> PageGateResult:
    """Gate one page whose readable regions *outcomes* records."""
    from bibr.ocr.text_layer_quality import (
        PageGateResult,
        covered_regions,
        judge_page,
        lexical_profile,
        text_coverage,
    )
    from bibr.scan.page_kind import _TEXT_LABELS

    prose = [r for r in regions if r.get("label") in _TEXT_LABELS and r.get("bbox_2d")]
    if not page.char_count:
        # A page with nothing to read (a full-page figure) loses nothing.
        readable = any(_region_task(r) not in _UNREAD_TASKS for r in regions)
        return PageGateResult("no_text_layer" if readable else None)
    if page.invisible:
        return PageGateResult("invisible_ocr_layer")
    prose_boxes = [tuple(float(v) for v in r["bbox_2d"][:4]) for r in prose]
    glyph_boxes = [box for _, box in page.glyphs if 0 <= box[0] <= 1000 and 0 <= box[1] <= 1000]
    coverage = text_coverage(prose_boxes, glyph_boxes)
    covered = covered_regions(prose_boxes, glyph_boxes)
    region_coverage = sum(covered) / len(covered) if covered else None
    read = [id(r) for r in prose if outcomes.get(id(r)) in ("filled", "rejected")]
    rejected = sum(1 for key in read if outcomes.get(key) == "rejected")
    lexical = lexical_profile(
        str(r.get("content") or "") for r in prose if outcomes.get(id(r)) == "filled"
    )
    return judge_page(
        coverage=coverage,
        rejected_regions=rejected,
        read_regions=len(read),
        lexical=lexical,
        region_coverage=region_coverage,
    )


def _blank(region: dict[str, Any]) -> None:
    region["content"] = ""
    region.pop("_native_text_used", None)
    region.pop("_formula_source", None)
    region.pop("_text_layer_table", None)
    region["_text_layer_kind"] = "unread"
    region["_text_layer_unread"] = True


def read_text_layer(
    pdf_bytes: bytes,
    layout_results: list[list[dict[str, Any]]],
    *,
    page_indices: Sequence[int] | None = None,
    min_printable_ratio: float = 0.85,
) -> TextLayerRead:
    """Fill every region OCR would read from the PDF text layer, page-gated.

    *layout_results* is not modified: the filled copy is returned with each
    page's verdict and the file's warnings. Regions the native-text pass
    already filled keep their text.
    """
    import pypdfium2

    from bibr.ocr.utils import pdfium_lock

    layouts = deepcopy(layout_results)
    indices = list(page_indices) if page_indices is not None else list(range(len(layouts)))
    result = TextLayerRead(layout_results=layouts, pages=[], warnings=[])

    with pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
    try:
        for slot, page_index in enumerate(indices):
            if slot >= len(layouts):
                break
            with pdfium_lock:
                page = _load_page(doc, page_index)
            regions = layouts[slot]
            result.pages.append(
                _read_page(page, regions, result, min_printable_ratio=min_printable_ratio)
            )
    finally:
        with pdfium_lock:
            doc.close()

    if result.tables_rebuilt:
        result.warnings.append(
            ProcessingWarning(
                WarningCode.TEXT_LAYER_TABLE_REBUILT,
                f"{result.tables_rebuilt} table(s) were rebuilt from text-layer glyph "
                "positions, not read by an OCR model",
            )
        )
    if result.formulas_read:
        result.warnings.append(
            ProcessingWarning(
                WarningCode.TEXT_LAYER_FORMULA_TEXT,
                f"{result.formulas_read} formula region(s) were read from the PDF text layer "
                "as plain text, not LaTeX",
            )
        )
    if result.private_use_dropped:
        result.warnings.append(
            ProcessingWarning(
                WarningCode.TEXT_LAYER_PRIVATE_USE_DROPPED,
                f"{result.private_use_dropped} private-use code point(s) (publisher icons or "
                "unmapped glyphs) were dropped from the PDF text layer"
                + (
                    f"; {result.private_use_inside} sat next to a letter or digit, where "
                    "the glyph may have stood for one (a ligature, a minus sign, a degree sign)"
                    if result.private_use_inside
                    else ""
                ),
            )
        )
    unmapped_warning = _unmapped_glyph_warning(result.unmapped_glyphs)
    if unmapped_warning is not None:
        result.warnings.append(unmapped_warning)
    return result


def _read_page(
    page: _PageText,
    regions: list[dict[str, Any]],
    result: TextLayerRead,
    *,
    min_printable_ratio: float,
) -> TextLayerPage:
    readable = [r for r in regions if _region_task(r) not in _UNREAD_TASKS]
    outcomes: dict[int, str] = {}
    if page.char_count and not page.invisible:
        for region in readable:
            if region.get("_native_text_used") and region.get("content"):
                outcomes[id(region)] = "filled"
                continue
            outcomes[id(region)] = _fill_region(
                page, region, min_printable_ratio=min_printable_ratio
            )
    verdict = _judge(page, regions, outcomes)
    number = page.index + 1
    if not verdict.ok:
        for region in readable:
            _blank(region)
            result.unread[str(region.get("label") or "")] += 1
        if readable:
            result.warnings.append(
                ProcessingWarning(
                    WarningCode.TEXT_LAYER_PAGE_UNUSABLE,
                    f"page {number}: {_REASON_TEXT.get(verdict.reason or '', verdict.reason)}; "
                    f"its {len(readable)} text region(s) are empty without OCR",
                )
            )
        return TextLayerPage(page.index, False, verdict.reason, verdict.metrics)

    unread_labels: Counter = Counter()
    for region in readable:
        label = str(region.get("label") or "")
        if region.get("_native_text_used") and region.get("content"):
            result.filled[label] += 1
            region.setdefault("_text_layer_kind", "text")
            if region.get("_text_layer_table"):
                result.tables_rebuilt += 1
            if region.get("_formula_source") == "text_layer":
                result.formulas_read += 1
            result.private_use_dropped += int(region.get("_private_use_dropped") or 0)
            result.private_use_inside += int(region.get("_private_use_inside") or 0)
            result.unmapped_glyphs += int(region.get("_unmapped_glyphs") or 0)
        else:
            _blank(region)
            unread_labels[label] += 1
    result.unread.update(unread_labels)
    if unread_labels:
        listed = ", ".join(f"{label} x{count}" for label, count in sorted(unread_labels.items()))
        result.warnings.append(
            ProcessingWarning(
                WarningCode.TEXT_LAYER_REGION_UNREAD,
                f"page {number}: {sum(unread_labels.values())} region(s) have no usable text "
                f"in the PDF text layer and are empty without OCR ({listed})",
            )
        )
    return TextLayerPage(page.index, True, None, verdict.metrics)


# --- Pipeline glue -------------------------------------------------------------


def apply_text_layer_read(fs: Any, read: TextLayerRead) -> None:
    """Put a read's layout, page verdicts and warnings on a pipeline FileState."""
    from dataclasses import replace

    fs.layout_results = read.layout_results
    verdicts = {page.index: page for page in read.pages}
    inspection = getattr(fs, "pdf_inspection", None)
    if inspection is not None:
        pages = tuple(
            replace(page, text_layer_ok=verdict.ok, text_layer_reason=verdict.reason)
            if (verdict := verdicts.get(page.index)) is not None
            else page
            for page in inspection.pages
        )
        fs.pdf_inspection = replace(inspection, pages=pages, layout_results=read.layout_results)
    # Page classes follow the verdicts; recompute them on next use.
    fs.page_kinds = None
    fs.warnings.extend(read.warnings)
    fs.text_layer_read = True


def _mark_all_unread(fs: Any) -> int:
    count = 0
    for page in fs.layout_results or []:
        for region in page:
            if _region_task(region) in _UNREAD_TASKS:
                continue
            _blank(region)
            count += 1
    return count


def page_verdict_summary(fs: Any) -> str:
    """The page verdicts of a file read from its text layer, in a phrase, or "".

    ``7 of 8 pages: most of its text is corrupt (a broken font encoding)``: how many
    of the judged pages failed, for each reason, so a refusal says whether the file
    is a scan or has a broken font.
    """
    inspection = getattr(fs, "pdf_inspection", None)
    judged = [page for page in getattr(inspection, "pages", ()) if page.text_layer_ok is not None]
    failed = Counter(page.text_layer_reason for page in judged if not page.text_layer_ok)
    return "; ".join(
        f"{count} of {len(judged)} pages: {_REASON_TEXT.get(reason or '', reason)}"
        for reason, count in failed.most_common()
    )


def note_text_layer_fallback(fs: Any, reason: str | None) -> None:
    """Record on *fs* that OCR could not read it here, once (``OCR_BACKEND_UNAVAILABLE``).

    *reason* is :func:`text_layer_fallback_reason`; None (an explicit
    ``--ocr text-layer`` run) records nothing. Called for every file the
    automatic backend's fallback reads, from the text layer or from the OCR
    cache, so the export says why no OCR model read it.
    """
    if reason is None:
        return
    if any(warning.code == WarningCode.OCR_BACKEND_UNAVAILABLE for warning in fs.warnings):
        return
    fs.warnings.append(
        ProcessingWarning(
            WarningCode.OCR_BACKEND_UNAVAILABLE,
            f"no OCR backend can start on this machine ({reason}); this PDF was read from "
            "its embedded text layer alone (OCR_FALLBACK_TO_TEXT_LAYER=auto)",
        )
    )


async def read_file_text_layer(
    fs: Any, settings: GlobalSettings, *, fallback_reason: str | None = None
) -> None:
    """Read one file's layout regions from its text layer (no OCR), in place.

    Used by the native-text stage when the run reads the text layer from the
    start, and by the OCR stage for files the automatic backend's fallback
    reaches after their native-text pass. A file read once is not read again.
    """
    import asyncio

    if getattr(fs, "text_layer_read", False) or not fs.pdf_bytes or fs.layout_results is None:
        return
    name = getattr(getattr(fs, "path", None), "name", "?")
    note_text_layer_fallback(fs, fallback_reason)
    try:
        read = await asyncio.to_thread(
            read_text_layer,
            fs.pdf_bytes,
            fs.layout_results,
            page_indices=fs.page_indices,
            min_printable_ratio=settings.ocr.native_text_min_printable_ratio,
        )
    except Exception:  # noqa: BLE001 - an unreadable layer leaves the regions empty
        logger.warning("Could not read the text layer of %s", name, exc_info=True)
        unread = _mark_all_unread(fs)
        fs.text_layer_read = True
        if unread:
            fs.warnings.append(
                ProcessingWarning(
                    WarningCode.TEXT_LAYER_REGION_UNREAD,
                    f"the PDF text layer could not be read; {unread} region(s) are empty "
                    "without OCR",
                )
            )
        return
    apply_text_layer_read(fs, read)
    failed = [page for page in read.pages if not page.ok]
    logger.info(
        "%s: read from the text layer: %d region(s) filled, %d unread, %d of %d page(s) unusable%s",
        name,
        sum(read.filled.values()),
        sum(read.unread.values()),
        len(failed),
        len(read.pages),
        f" ({', '.join(sorted({page.reason or '' for page in failed}))})" if failed else "",
    )
