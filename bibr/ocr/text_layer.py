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
    "TextLayerStream",
    "apply_text_layer_read",
    "is_text_layer_mode",
    "note_text_layer_fallback",
    "read_file_text_layer",
    "read_text_layer",
    "text_layer_fallback_reason",
    "text_layer_requested",
    "text_layer_stream",
]

logger = logging.getLogger(__name__)

# A text-layer region may carry one unmapped glyph, or one per this many characters ...
_CHARS_PER_UNMAPPED_GLYPH = 100
# ... or, in a region of at most this many characters (a byline or a heading)
# where no two of them stand side by side, one per this many.
_MAX_SPARSE_UNMAPPED_GLYPH_CHARS = 200
_CHARS_PER_SPARSE_UNMAPPED_GLYPH = 20
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
# Layout labels of reference entries and lists (see ``fix_ocr_artifacts``).
_REFERENCE_LABELS = frozenset({"reference", "reference_content"})
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
    # 1-based numbers of the scanned pages read from the OCR layer the PDF
    # carries (``read_ocr_layer``).
    ocr_layer_pages: list[int] = field(default_factory=list)
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
    # The characters are a scan's embedded OCR layer (invisible text over a
    # page-sized image), read because the caller asked for it.
    ocr_layer: bool = False
    # Raw PDF-point chars deferred until after pdfium_lock is released.
    _raw_chars: list[tuple[str, tuple[float, float, float, float]]] | None = field(
        default=None, repr=False, compare=False
    )

    def _materialize(self, broken_lines: list[Any] | None = None) -> None:
        raw = self._raw_chars
        if raw is None:
            return
        self._raw_chars = None
        self.chars, self.lines = _build_layout_chars_and_lines(
            raw, self.index, self.crop_box, self.rotation, broken_lines=broken_lines
        )

    @property
    def glyphs(self) -> list[tuple[str, tuple[float, float, float, float]]]:
        """The visible glyphs: no whitespace, no empty boxes."""
        return [
            (ch, box)
            for ch, box in self.chars
            if ch and not ch.isspace() and box[2] > box[0] and box[3] > box[1]
        ]


def _build_layout_chars_and_lines(
    chars: list[tuple[str, tuple[float, float, float, float]]],
    page_index: int,
    crop_box: tuple[float, float, float, float],
    rotation: int,
    *,
    broken_lines: list[Any] | None = None,
) -> tuple[
    list[tuple[str, tuple[float, float, float, float]]],
    list[tuple[str, tuple[float, float, float, float], float]],
]:
    """Convert raw PDF-point chars and lines to 0..1000 layout coordinates."""
    from bibr.ocr.native_text import _pdf_points_to_normalized_bbox
    from bibr.ocr.pdf_inspection import _break_wrapped_lines
    from bibr.ocr.ref_geometry import group_chars_into_lines

    cx0, cy0, cx1, cy1 = crop_box
    crop_w = (cx1 - cx0) or 1.0
    crop_h = (cy1 - cy0) or 1.0

    if rotation == 0:
        layout_chars: list[tuple[str, tuple[float, float, float, float]]] = []
        append_char = layout_chars.append
        for ch, box in chars:
            if not ch:
                continue
            left, bottom, right, top = box
            u1 = (left - cx0) / crop_w * 1000.0
            u2 = (right - cx0) / crop_w * 1000.0
            v1 = (cy1 - top) / crop_h * 1000.0
            v2 = (cy1 - bottom) / crop_h * 1000.0
            append_char(
                (
                    ch,
                    (
                        u1 if u1 <= u2 else u2,
                        v1 if v1 <= v2 else v2,
                        u2 if u2 >= u1 else u1,
                        v2 if v2 >= v1 else v1,
                    ),
                )
            )
        src_lines = (
            broken_lines
            if broken_lines is not None
            else group_chars_into_lines(_break_wrapped_lines(chars), page_index)
        )
        lines: list[tuple[str, tuple[float, float, float, float], float]] = []
        append_line = lines.append
        for line in src_lines:
            u1 = (line.x0 - cx0) / crop_w * 1000.0
            u2 = (line.x1 - cx0) / crop_w * 1000.0
            v1 = (cy1 - line.y_top) / crop_h * 1000.0
            v2 = (cy1 - line.y_bottom) / crop_h * 1000.0
            append_line(
                (
                    line.text,
                    (
                        u1 if u1 <= u2 else u2,
                        v1 if v1 <= v2 else v2,
                        u2 if u2 >= u1 else u1,
                        v2 if v2 >= v1 else v1,
                    ),
                    line.font_size,
                )
            )
        return layout_chars, lines

    def to_layout(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return _pdf_points_to_normalized_bbox(box, crop_box, rotation)

    layout_chars = [(ch, to_layout(box)) for ch, box in chars if ch]
    src_lines = (
        broken_lines
        if broken_lines is not None
        else group_chars_into_lines(_break_wrapped_lines(chars), page_index)
    )
    lines = [
        (line.text, to_layout((line.x0, line.y_bottom, line.x1, line.y_top)), line.font_size)
        for line in src_lines
    ]
    return layout_chars, lines


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


def _load_page(
    doc: Any, page_index: int, *, read_ocr_layer: bool = False, defer_lines: bool = False
) -> _PageText:
    """Read one page's text layer under the (caller-held) PDFium lock.

    With *read_ocr_layer*, a scanned page's embedded OCR layer is read like
    printed text: mostly invisible text over images that cover the page, the
    test the native pass uses to send a page to OCR. Invisible text on any
    other page stays unread: it was hidden on purpose, or the scan is cropped
    and the test cannot tell it from that.
    """
    from bibr.ocr.native_text import _page_crop_box, _page_rotation, open_text_page

    page = doc[page_index]
    try:
        crop_box = _page_crop_box(page)
        rotation = _page_rotation(page)
        textpage = open_text_page(page)
        try:
            return _page_text(
                page,
                textpage,
                page_index,
                crop_box,
                rotation,
                read_ocr_layer=read_ocr_layer,
                defer_lines=defer_lines,
            )
        finally:
            textpage.close()
    finally:
        page.close()


def _page_text(
    page: Any,
    textpage: Any,
    page_index: int,
    crop_box: tuple[float, float, float, float],
    rotation: int,
    *,
    read_ocr_layer: bool,
    records: list[tuple[str, float, float, bool]] | None = None,
    chars: list[tuple[str, tuple[float, float, float, float]]] | None = None,
    page_cache: dict[str, object] | None = None,
    defer_lines: bool = False,
) -> _PageText:
    """One page's text layer from its open text page (see :func:`_load_page`).

    *records* and *chars*, when the caller already read them from *textpage*,
    are used as they are.
    """
    from bibr.ocr.native_text import (
        _build_page_char_records,
        _coverage_qualifies,
        _invisible_text_share,
        _page_image_coverage,
        _share_qualifies,
    )
    from bibr.ocr.ref_geometry import _extract_page_chars

    char_count = textpage.count_chars()
    if not char_count:
        return _PageText(page_index, crop_box, rotation, 0, False, [], [], [])
    if page_cache is not None and "invisible_share" in page_cache:
        invisible = _share_qualifies(page_cache["invisible_share"])  # type: ignore[arg-type]
    else:
        try:
            # Mostly invisible text is a hidden OCR layer, not printed text.
            # The native pass also asks for a page-sized image; scans
            # cropped to their text block have smaller ones (dev scans:
            # 0.55 and 0.67 of the page), and no OCR reads them here.
            codes = page_cache.get("codes") if page_cache is not None else None
            share = _invisible_text_share(textpage, codes=codes)  # type: ignore[arg-type]
            if page_cache is not None:
                page_cache["invisible_share"] = share
            invisible = _share_qualifies(share)
        except Exception:  # noqa: BLE001 - an unreadable check keeps the layer
            logger.debug("Invisible-layer check failed on page %d", page_index, exc_info=True)
            invisible = False
    ocr_layer = False
    if invisible and read_ocr_layer:
        if page_cache is not None and "image_coverage" in page_cache:
            ocr_layer = _coverage_qualifies(page_cache["image_coverage"])  # type: ignore[arg-type]
        else:
            try:
                coverage = _page_image_coverage(page, crop_box)
                if page_cache is not None:
                    page_cache["image_coverage"] = coverage
                ocr_layer = _coverage_qualifies(coverage)
            except Exception:  # noqa: BLE001 - an unreadable check leaves the page unread
                logger.debug("Image-coverage check failed on page %d", page_index, exc_info=True)
    if invisible and not ocr_layer:
        return _PageText(page_index, crop_box, rotation, char_count, True, [], [], [])
    raw_out: dict[str, object] | None = None
    if records is None:
        if chars is None:
            raw_out = {}
        records = _build_page_char_records(textpage, raw_out=raw_out)
    if chars is None:
        codes = (
            raw_out.get("codes")
            if raw_out is not None
            else (page_cache.get("codes") if page_cache is not None else None)
        )
        boxes = (
            raw_out.get("boxes")
            if raw_out is not None
            else (page_cache.get("boxes") if page_cache is not None else None)
        )
        chars = _extract_page_chars(textpage, codes=codes, char_boxes=boxes)  # type: ignore[arg-type]

    if defer_lines:
        return _PageText(
            page_index,
            crop_box,
            rotation,
            char_count,
            False,
            records,
            [],
            [],
            ocr_layer,
            _raw_chars=chars,
        )

    layout_chars, lines = _build_layout_chars_and_lines(chars, page_index, crop_box, rotation)
    return _PageText(
        page_index, crop_box, rotation, char_count, False, records, layout_chars, lines, ocr_layer
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

    A short line takes a few more, see :func:`_sparse_unmapped_glyphs`.
    """
    from bibr.ocr.native_text import _is_disallowed_control_char

    unmapped = [_is_disallowed_control_char(ch) for ch in text]
    count = sum(unmapped)
    if not count:
        return text, 0
    if count > max(1, len(text) // _CHARS_PER_UNMAPPED_GLYPH) and not _sparse_unmapped_glyphs(
        unmapped
    ):
        return text, 0
    marked = (_UNMAPPED_GLYPH if bad else ch for ch, bad in zip(text, unmapped, strict=True))
    return "".join(marked), count


def _sparse_unmapped_glyphs(unmapped: Sequence[bool]) -> bool:
    """Whether the unmapped glyphs (*unmapped* flags each char) are a few single letters.

    A byline or a heading set in a font that maps every letter but its
    accented ones reads "Ana Mar\x03a Lo\x04pez" among its names: two marks in a
    line of fifty characters, more than one per hundred. They are single letters of
    otherwise whole words, never two side by side, and at most one per
    :data:`_CHARS_PER_SPARSE_UNMAPPED_GLYPH` characters. A broken font
    encoding turns runs of letters into control characters, which this
    leaves rejected. Only a short line (:data:`_MAX_SPARSE_UNMAPPED_GLYPH_CHARS`
    characters at most) takes them: a paragraph that shows a mark every few
    words ("e\x0ecient") is body text in a font with a broken ligature, which
    the one-per-hundred limit rejects.
    """
    count = sum(unmapped)
    if len(unmapped) > _MAX_SPARSE_UNMAPPED_GLYPH_CHARS:
        return False
    if count > len(unmapped) // _CHARS_PER_SPARSE_UNMAPPED_GLYPH:
        return False
    return not any(a and b for a, b in zip(unmapped, unmapped[1:], strict=False))


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
        # A glyph is one code point, or a letter and the accent composed onto it
        # when Unicode has no single code point for the pair ("x" + U+0302).
        ch = glyph[0]
        if ch and all(unicodedata.category(c) == "Co" for c in ch):
            dropped += 1
            following = chars[index + 1][0] if index + 1 < len(chars) else ""
            inside += following[:1].isalnum()
            continue
        # The separator controls (U+001C..U+001F) count as whitespace and only end a word.
        if ch and not ch.isspace() and any(_is_disallowed_control_char(c) for c in ch):
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
            # The other gates judge the text around the marks, which are not its own fault;
            # a U+FFFD the text layer itself carries still counts.
            text = text.replace(_UNMAPPED_GLYPH, "", region["_unmapped_glyphs"])
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
        text = fix_ocr_artifacts(text, reference=label in _REFERENCE_LABELS)
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
            marked.replace(_UNMAPPED_GLYPH, "", unmapped), min_printable_ratio
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
        _area,
        covered_regions,
        judge_page,
        lexical_profile,
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
    covered = covered_regions(prose_boxes, glyph_boxes)
    total_area = sum(_area(box) for box in prose_boxes)
    coverage = (
        sum(_area(box) for box, hit in zip(prose_boxes, covered, strict=True) if hit) / total_area
        if total_area > 0
        else None
    )
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


def _page_ranges(numbers: Sequence[int]) -> str:
    """``pages 1-4, 7``: sorted page numbers with runs folded."""
    ordered = sorted(set(numbers))
    runs: list[tuple[int, int]] = []
    for number in ordered:
        if runs and number == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], number)
        else:
            runs.append((number, number))
    listed = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)
    return f"page{'s' if len(ordered) != 1 else ''} {listed}"


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
    read_ocr_layer: bool = False,
) -> TextLayerRead:
    """Fill every region OCR would read from the PDF text layer, page-gated.

    *layout_results* is not modified: the filled copy is returned with each
    page's verdict and the file's warnings. Regions the native-text pass
    already filled keep their text. *read_ocr_layer* also reads the embedded
    OCR layer of a scanned page (see :func:`_load_page`); such a page passes
    the same gate as printed text and is named in a
    ``TEXT_LAYER_OCR_LAYER_READ`` warning.
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
                page = _load_page(doc, page_index, read_ocr_layer=read_ocr_layer, defer_lines=True)
            page._materialize()
            regions = layouts[slot]
            verdict = _read_page(page, regions, result, min_printable_ratio=min_printable_ratio)
            result.pages.append(verdict)
            if page.ocr_layer and verdict.ok:
                result.ocr_layer_pages.append(page_index + 1)
    finally:
        with pdfium_lock:
            doc.close()
    return _finish_read(result)


class TextLayerStream:
    """:func:`read_text_layer` done page by page inside ``inspect_pdf``'s own walk.

    ``inspect_pdf`` hands over each page's open text page, with the char
    records and chars it read from it, once the native pass is done with the
    page's regions, so the PDF is not opened, stripped and read a second time.
    :meth:`finish` returns what :func:`read_text_layer` would return for the
    inspection's layout, or None when any page could not be read here; the
    caller then reads the text layer the usual way, which fails or succeeds
    as it always did.
    """

    def __init__(
        self,
        page_indices: Sequence[int] | None,
        *,
        min_printable_ratio: float = 0.85,
        read_ocr_layer: bool = False,
    ) -> None:
        self._indices = list(page_indices) if page_indices is not None else None
        self._min_printable_ratio = min_printable_ratio
        self._read_ocr_layer = read_ocr_layer
        self._read: dict[int, list[dict[str, Any]]] = {}
        self._pending: list[tuple[int, int, _PageText, list[dict[str, Any]], bool]] = []
        self._result = TextLayerRead(layout_results=[], pages=[], warnings=[])
        self._failed = False

    def add_page(
        self,
        slot: int,
        page: Any,
        textpage: Any,
        page_index: int,
        crop_box: tuple[float, float, float, float],
        rotation: int,
        regions: list[dict[str, Any]],
        *,
        records: list[tuple[str, float, float, bool]] | None = None,
        chars: list[tuple[str, tuple[float, float, float, float]]] | None = None,
        page_cache: dict[str, object] | None = None,
        copy_regions: bool = True,
    ) -> None:
        """Capture detached PDFium data for one page under the caller's PDFium lock.

        Pure-Python layout coordinate projection, region filling, table
        rebuilding, and page gating are deferred to :meth:`process_pending`
        (called by ``inspect_pdf`` after releasing ``pdfium_lock``, or lazily
        by :meth:`finish`). *regions* are never modified.
        """
        if self._failed:
            return
        expected = len(self._read) + len(self._pending)
        if slot != expected or (self._indices is not None and self._indices[slot] != page_index):
            self._failed = True
            return
        try:
            text = _page_text(
                page,
                textpage,
                page_index,
                crop_box,
                rotation,
                read_ocr_layer=self._read_ocr_layer,
                records=records,
                chars=chars,
                page_cache=page_cache,
                defer_lines=True,
            )
            stored_regions = deepcopy(regions) if copy_regions else regions
        except Exception:  # noqa: BLE001 - the caller reads the text layer again
            logger.debug("Streamed text-layer read failed on page %d", page_index, exc_info=True)
            self._failed = True
            return
        self._pending.append((slot, page_index, text, stored_regions, copy_regions))

    def process_pending(self, broken_lines_by_slot: dict[int, list[Any]] | None = None) -> None:
        """Run the deferred pure-Python text-layer reading & gating outside ``pdfium_lock``."""
        if self._failed or not self._pending:
            self._pending.clear()
            return
        pending = self._pending
        self._pending = []
        for slot, page_index, text, regions, already_copied in pending:
            try:
                broken_lines = (
                    broken_lines_by_slot.get(slot) if broken_lines_by_slot is not None else None
                )
                text._materialize(broken_lines=broken_lines)
                copied = regions if already_copied else deepcopy(regions)
                verdict = _read_page(
                    text, copied, self._result, min_printable_ratio=self._min_printable_ratio
                )
            except Exception:  # noqa: BLE001 - the caller reads the text layer again
                logger.debug(
                    "Streamed text-layer read failed on page %d", page_index, exc_info=True
                )
                self._failed = True
                return
            self._read[slot] = copied
            self._result.pages.append(verdict)
            if text.ocr_layer and verdict.ok:
                self._result.ocr_layer_pages.append(page_index + 1)

    def finish(self, layout_results: list[list[dict[str, Any]]]) -> TextLayerRead | None:
        """The read of *layout_results* (the inspection's), or None to read it again."""
        self.process_pending()
        indices = self._indices if self._indices is not None else range(len(layout_results))
        if self._failed or len(self._read) != min(len(indices), len(layout_results)):
            return None
        self._result.layout_results = [
            self._read[slot] if slot in self._read else deepcopy(regions)
            for slot, regions in enumerate(layout_results)
        ]
        self._failed = True  # a stream is finished once
        return _finish_read(self._result)


def _finish_read(result: TextLayerRead) -> TextLayerRead:
    """Add the file's warnings to a read whose pages are all done."""
    if result.ocr_layer_pages:
        result.warnings.append(
            ProcessingWarning(
                WarningCode.TEXT_LAYER_OCR_LAYER_READ,
                f"{len(result.ocr_layer_pages)} scanned page(s) were read from the OCR text "
                f"layer the PDF carries ({_page_ranges(result.ocr_layer_pages)}), not by an "
                "OCR model",
            )
        )
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
    failed: Counter = Counter()
    failed_ids: set[int] = set()
    if page.char_count and not page.invisible:
        for region in readable:
            if region.get("_native_text_used") and region.get("content"):
                outcomes[id(region)] = "filled"
                continue
            try:
                outcomes[id(region)] = _fill_region(
                    page, region, min_printable_ratio=min_printable_ratio
                )
            except Exception:  # noqa: BLE001 - a fault in one region must not cost the paper its text
                label = str(region.get("label") or "")
                logger.warning(
                    "Reading a %s region on page %d from the text layer failed",
                    label or "layout",
                    page.index + 1,
                    exc_info=True,
                )
                outcomes[id(region)] = "rejected"
                failed[label] += 1
                failed_ids.add(id(region))
    verdict = _judge(page, regions, outcomes)
    number = page.index + 1
    if failed:
        listed = ", ".join(f"{label} x{count}" for label, count in sorted(failed.items()))
        result.warnings.append(
            ProcessingWarning(
                WarningCode.TEXT_LAYER_READ_ERROR,
                f"page {number}: reading {sum(failed.values())} region(s) from the PDF text "
                f"layer failed inside bibr and they are empty ({listed})",
            )
        )
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
        elif id(region) in failed_ids:  # reported above, as bibr's fault and not the PDF's
            _blank(region)
            result.unread[label] += 1
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


def has_unread_ocr_layer(fs: Any) -> bool:
    """Whether ``OCR_TEXT_LAYER_READ_OCR_LAYER`` would read a page this read left out.

    A page the gate turned down for its invisible text that the native pass
    found to be a scan under a page-sized image. With the option on such a page
    is read, so this is true only when it is unset or off: unset, the layer was
    left unread or turned down in place of a refusal (see
    :func:`_read_ocr_layer_instead`).
    """
    inspection = getattr(fs, "pdf_inspection", None)
    return any(
        page.text_layer_reason == "invisible_ocr_layer" and page.invisible_text_layer
        for page in getattr(inspection, "pages", ())
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


def text_layer_stream(fs: Any, settings: GlobalSettings) -> TextLayerStream | None:
    """A stream for ``inspect_pdf`` to read *fs*'s text layer in its own walk, or None.

    None when :func:`read_file_text_layer` would not read the file.
    """
    if getattr(fs, "text_layer_read", False) or not fs.pdf_bytes or fs.layout_results is None:
        return None
    return TextLayerStream(
        getattr(fs, "page_indices", None),
        min_printable_ratio=settings.ocr.native_text_min_printable_ratio,
        read_ocr_layer=settings.ocr.text_layer_read_ocr_layer is True,
    )


# --- A scan's own OCR layer, read rather than refused --------------------------

# Page verdicts that make a page a scan, not a broken font. A file whose failed
# pages all carry one, at least one for an invisible OCR layer, is read from
# that layer instead of being refused (see _read_ocr_layer_instead).
_SCAN_REASONS = frozenset({"invisible_ocr_layer", "no_text_layer", "low_coverage"})


def _text_region_share(layout_results: list[list[dict[str, Any]]]) -> float | None:
    """Share of the running-text regions that hold text, or None when there are none.

    The share ``OcrStage._check_text_layer_success`` refuses a file under, on
    the layout regions before they become OCR regions.
    """
    from bibr.scan.page_kind import _TEXT_LABELS

    needed = filled = 0
    for page in layout_results:
        for region in page:
            # The OCR stage emits no region for an abandoned one.
            if region.get("task_type") == "abandon":
                continue
            if str(region.get("label") or "text") not in _TEXT_LABELS:
                continue
            needed += 1
            filled += bool(str(region.get("content") or "").strip())
    return filled / needed if needed else None


def _refused_for_its_ocr_layer(read: TextLayerRead, threshold: float) -> bool:
    """Whether *read* leaves the file to be refused as a scan, for its OCR layer only."""
    share = _text_region_share(read.layout_results)
    if share is None or share >= threshold:
        return False
    failed = {page.reason for page in read.pages if not page.ok}
    return "invisible_ocr_layer" in failed and failed <= _SCAN_REASONS


def _ocr_layer_texts(read: TextLayerRead) -> list[str]:
    """The region texts of the pages *read* took from the PDF's OCR layer."""
    layer_pages = set(read.ocr_layer_pages)
    return [
        str(region.get("content") or "")
        for page, regions in zip(read.pages, read.layout_results, strict=False)
        if page.index + 1 in layer_pages
        for region in regions
    ]


def _read_ocr_layer_instead(
    fs: Any, settings: GlobalSettings, read: TextLayerRead
) -> tuple[TextLayerRead, str | None]:
    """*read*, or the file read from its scans' OCR layer when *read* would refuse it.

    A scan with the scanner's OCR text under its image (invisible text over a
    page-sized image) has no text this route can read, and the OCR stage
    refuses it (``scan_needs_ocr``) unless ``OCR_TEXT_LAYER_READ_OCR_LAYER``
    is on. Refusing loses the whole paper for want of an OCR model, while the
    layer under the image usually reads well, so a file that would be refused
    for that reason alone, and never for a font with no usable text, is read
    again with the layer. The layer read is kept when it lifts the file over
    the refusal threshold and its words are words
    (:func:`~bibr.ocr.text_layer_quality.lexicon_word_share`); its
    ``TEXT_LAYER_OCR_LAYER_READ`` warning says the text is the PDF's own OCR.
    Called only with the option unset: off keeps the refusal.

    Returns the read to apply and, when the layer was tried and turned down,
    why, for the refusal to say.
    """
    from bibr.ocr.text_layer_quality import MIN_LEXICON_WORD_SHARE, lexicon_word_share

    threshold = settings.ocr.min_success_rate
    if threshold <= 0 or not _refused_for_its_ocr_layer(read, threshold):
        return read, None
    try:
        layer = read_text_layer(
            fs.pdf_bytes,
            fs.layout_results,
            page_indices=fs.page_indices,
            min_printable_ratio=settings.ocr.native_text_min_printable_ratio,
            read_ocr_layer=True,
        )
        words = lexicon_word_share(_ocr_layer_texts(layer), enough=MIN_LEXICON_WORD_SHARE)
    except Exception:  # noqa: BLE001 - the first read stands, and the file is refused
        logger.warning("Reading the OCR layer of a scan failed", exc_info=True)
        return read, None
    if not layer.ocr_layer_pages:
        return read, "no page of it could be read"
    share = _text_region_share(layer.layout_results)
    if share is None or share < threshold:
        return read, f"still leaves too many text regions unread ({share or 0.0:.0%} read)"
    if words is not None and words < MIN_LEXICON_WORD_SHARE:
        return read, f"does not read as words (only {words:.0%} of its words are known words)"
    layer.warnings = [
        ProcessingWarning(
            warning.code,
            f"{warning.message}, because the file would otherwise be refused as a scan "
            "(scan_needs_ocr)",
        )
        if warning.code == WarningCode.TEXT_LAYER_OCR_LAYER_READ
        else warning
        for warning in layer.warnings
    ]
    return layer, None


async def read_file_text_layer(
    fs: Any,
    settings: GlobalSettings,
    *,
    fallback_reason: str | None = None,
    read: TextLayerRead | None = None,
) -> None:
    """Read one file's layout regions from its text layer (no OCR), in place.

    Used by the native-text stage when the run reads the text layer from the
    start, and by the OCR stage for files the automatic backend's fallback
    reaches after their native-text pass. A file read once is not read again.
    *read*, when given, is the file's read already done (a
    :class:`TextLayerStream`'s) and is applied as it is. A file the read would
    leave to be refused for its scans' OCR layer is read from that layer
    instead (:func:`_read_ocr_layer_instead`).
    """
    import asyncio

    if getattr(fs, "text_layer_read", False) or not fs.pdf_bytes or fs.layout_results is None:
        return
    name = getattr(getattr(fs, "path", None), "name", "?")
    note_text_layer_fallback(fs, fallback_reason)
    try:
        if read is None:
            read = await asyncio.to_thread(
                read_text_layer,
                fs.pdf_bytes,
                fs.layout_results,
                page_indices=fs.page_indices,
                min_printable_ratio=settings.ocr.native_text_min_printable_ratio,
                read_ocr_layer=settings.ocr.text_layer_read_ocr_layer is True,
            )
    except Exception:  # noqa: BLE001 - an unreadable layer leaves the regions empty
        logger.warning("Could not read the text layer of %s", name, exc_info=True)
        unread = _mark_all_unread(fs)
        fs.text_layer_read = True
        if unread:
            fs.warnings.append(
                ProcessingWarning(
                    WarningCode.TEXT_LAYER_READ_ERROR,
                    f"reading the PDF text layer failed inside bibr; {unread} region(s) are "
                    "empty without OCR",
                )
            )
        return
    # Unset, not off: OCR_TEXT_LAYER_READ_OCR_LAYER=false keeps the refusal.
    if settings.ocr.text_layer_read_ocr_layer is None:
        read, fs.text_layer_ocr_layer_refusal = await asyncio.to_thread(
            _read_ocr_layer_instead, fs, settings, read
        )
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
