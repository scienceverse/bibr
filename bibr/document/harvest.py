"""Harvest of the PDF-native document layer.

The harvest reads a page's text layer once more, after bibr's own reads, on the
same text page and under the same ``pdfium_lock`` hold: the glyph stream with
origins, loose boxes and flags, the text objects (font, effective size,
colour, render mode, marked content) once each, and the reading records that
``_build_page_char_records`` already builds for the native-text fill.

Two callers share :class:`LayerBuilder`, so their glyph indexes match:
``inspect_pdf`` (with ``include_doc_layer``) builds the layer inline, and
:func:`build_document_layer` rebuilds it from the PDF bytes on a cache hit.
Nothing here writes to the page, the text page or the regions; every failure
lands in ``DocumentLayer.component_errors`` and never fails the paper.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from bibr.document import ids
from bibr.document.model import (
    COLUMN_DTYPES,
    GLYPH_EXCLUDED,
    GLYPH_GENERATED,
    GLYPH_HYPHEN,
    GLYPH_INVISIBLE_RENDER,
    GLYPH_MAP_ERROR,
    GLYPH_NO_BOX,
    GLYPH_NO_LOOSE_BOX,
    GLYPH_NO_ORIGIN,
    INDEX_FRAME,
    INVISIBLE_LAYER_RULE,
    LAYER_VERSION,
    LINE_NUMBER_RULE,
    OBJ_NO_FILL,
    OBJ_NO_FONT_SIZE,
    OBJ_NO_MATRIX,
    OBJ_NO_STROKE,
    OBJ_NOT_IN_WALK,
    WATERMARK_RULE,
    Box,
    Decided,
    DocumentLayer,
    Font,
    Furniture,
    Page,
    PageColumns,
    Presence,
    RenderRecipe,
    RoleTag,
    as_box,
)
from bibr.ocr import native_text as nt

_NAN = float("nan")
_NAN_BOX = (_NAN, _NAN, _NAN, _NAN)
_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

# pdfium functions the harvest reads with; a build without one leaves its
# fields unread and names it in Presence.missing_apis.
_HARVEST_APIS = (
    "FPDFText_GetTextObject",
    "FPDFText_GetCharOrigin",
    "FPDFText_GetLooseCharBox",
    "FPDFText_IsGenerated",
    "FPDFText_IsHyphen",
    "FPDFText_HasUnicodeMapError",
    "FPDFText_GetTextIndexFromCharIndex",
    "FPDFTextObj_GetFont",
    "FPDFTextObj_GetFontSize",
    "FPDFTextObj_GetTextRenderMode",
    "FPDFPageObj_GetMatrix",
    "FPDFPageObj_GetFillColor",
    "FPDFPageObj_GetStrokeColor",
    "FPDFPageObj_GetMarkedContentID",
    "FPDFPageObj_CountMarks",
    "FPDFPageObj_GetMark",
    "FPDFPageObjMark_GetName",
    "FPDFFont_GetBaseFontName",
    "FPDFFont_GetFamilyName",
    "FPDFFont_GetFlags",
    "FPDFFont_GetWeight",
    "FPDFFont_GetItalicAngle",
    "FPDFFont_GetIsEmbedded",
)

# How the layout stage renders a page (bibr.ocr.image_utils.iter_pdf_pages_with_index):
# pypdfium2's PdfPage.render(scale=dpi/72) defaults, then to_pil().
_RENDER_FLAGS = ("pypdfium2.render", "FPDF_ANNOT", "fill=255,255,255,255", "no_forms")

_FURNITURE_DECIDED = {
    "watermark": Decided("furniture.watermark", WATERMARK_RULE),
    "line_number": Decided("furniture.line_number", LINE_NUMBER_RULE),
}

# The Page.text_source values of a page with a text layer.
_TEXT_LAYER_SOURCES = ("native", "invisible_layer")

# Spans: a glyph continues the open span while its baseline stays within
# this share of the size and the size within _SPAN_SIZE_TOLERANCE of it.
_SPAN_BASELINE_TOLERANCE = 0.1
_SPAN_SIZE_TOLERANCE = 0.01

# Superscript rule superscript/1: a span at most _SCRIPT_MAX_SIZE_RATIO of the
# line's dominant size whose baseline sits at least _SUPERSCRIPT_MIN_RAISE of
# that size above the dominant baseline is a superscript; one at least
# _SUBSCRIPT_MIN_DROP below it is a subscript.
SCRIPT_RULE = Decided("span_rules", "superscript/1", calibrated=False)
_SCRIPT_MAX_SIZE_RATIO = 0.9
_SUPERSCRIPT_MIN_RAISE = 0.2
_SUBSCRIPT_MIN_DROP = 0.1


@dataclass(frozen=True, slots=True)
class RenderBudget:
    """The layout stage's render settings, for the per-page render recipe."""

    dpi: int
    max_pixels: int
    max_dimension: int
    min_dpi: int | None = None


def missing_apis() -> tuple[str, ...]:
    import pypdfium2.raw as pdfium_c

    return tuple(name for name in _HARVEST_APIS if not hasattr(pdfium_c, name))


def pdfium_version() -> str:
    import pypdfium2

    return f"pypdfium2 {pypdfium2.PYPDFIUM_INFO} / pdfium {pypdfium2.PDFIUM_INFO}"


def layout_render_dpi(width: float, height: float, budget: RenderBudget) -> int | None:
    """The DPI the layout stage renders a *width* x *height* pt page at.

    Mirrors ``iter_pdf_pages_with_index``: a page over the budget renders at
    the largest DPI that fits when that is at least ``min_dpi``. None when
    the layout stage would refuse the page.
    """
    from bibr.ocr.image_utils import _fits, _render_size, fitting_render_dpi

    dpi = budget.dpi
    size = _render_size(width, height, dpi)
    if budget.min_dpi is not None and not _fits(*size, budget.max_pixels, budget.max_dimension):
        reduced = fitting_render_dpi(width, height, dpi, budget.max_pixels, budget.max_dimension)
        if reduced >= budget.min_dpi:
            dpi = reduced
            size = _render_size(width, height, dpi)
    if not _fits(*size, budget.max_pixels, budget.max_dimension):
        return None
    return dpi


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


def _address(pointer) -> int:
    return ctypes.addressof(pointer.contents)


class _Api:
    """The pdfium functions, resolved once; a missing one is None."""

    def __init__(self) -> None:
        import pypdfium2.raw as pdfium_c

        self.c = pdfium_c
        for name in _HARVEST_APIS:
            setattr(self, name, getattr(pdfium_c, name, None))

    if TYPE_CHECKING:
        # The functions are set by name in __init__.
        def __getattr__(self, name: str) -> Any: ...


class FontTable:
    """The document's fonts, deduplicated by their properties."""

    def __init__(self) -> None:
        self.fonts: list[Font] = []
        self._ids: dict[tuple, int] = {}

    def font_id(self, api: _Api, handle, memo: dict[int, int]) -> int:
        """The id of the font behind an FPDF_FONT *handle* (-1 for none)."""
        if not handle:
            return -1
        address = _address(handle)
        font_id = memo.get(address)
        if font_id is None:
            props = _font_props(api, handle)
            font_id = self._ids.get(props)
            if font_id is None:
                font_id = len(self.fonts)
                self._ids[props] = font_id
                self.fonts.append(Font(font_id, *props))
            memo[address] = font_id
        return font_id


def _font_name(function, handle) -> str:
    if function is None:
        return ""
    size = function(handle, None, 0)
    if size <= 0:
        return ""
    buffer = ctypes.create_string_buffer(size)
    if function(handle, buffer, size) <= 0:
        return ""
    return buffer.raw[:size].split(b"\x00", 1)[0].decode("utf-8", "replace")


def _font_props(api: _Api, handle) -> tuple[str, str, int, int, int, bool]:
    base_name = _font_name(api.FPDFFont_GetBaseFontName, handle)
    family = _font_name(api.FPDFFont_GetFamilyName, handle)
    flags = api.FPDFFont_GetFlags(handle) if api.FPDFFont_GetFlags else -1
    weight = api.FPDFFont_GetWeight(handle) if api.FPDFFont_GetWeight else -1
    angle = ctypes.c_int(0)
    italic_angle = 0
    if api.FPDFFont_GetItalicAngle and api.FPDFFont_GetItalicAngle(handle, ctypes.byref(angle)):
        italic_angle = angle.value
    embedded = (
        bool(api.FPDFFont_GetIsEmbedded(handle) == 1) if api.FPDFFont_GetIsEmbedded else False
    )
    return base_name, family, int(flags), int(weight), int(italic_angle), embedded


def _mark_name(api: _Api, mark) -> str:
    size = ctypes.c_ulong(0)
    if not api.FPDFPageObjMark_GetName(mark, None, 0, ctypes.byref(size)) or size.value < 2:
        return ""
    buffer = (ctypes.c_ushort * (size.value // 2))()
    if not api.FPDFPageObjMark_GetName(mark, buffer, size.value, ctypes.byref(size)):
        return ""
    return bytes(buffer)[: max(size.value - 2, 0)].decode("utf-16-le", "replace")


def _in_artifact(api: _Api, obj) -> bool:
    """Whether the object sits inside /Artifact marked content."""
    if not (api.FPDFPageObj_CountMarks and api.FPDFPageObj_GetMark and api.FPDFPageObjMark_GetName):
        return False
    for index in range(max(api.FPDFPageObj_CountMarks(obj), 0)):
        mark = api.FPDFPageObj_GetMark(obj, index)
        if mark and _mark_name(api, mark) == "Artifact":
            return True
    return False


def _color(function, obj) -> int | None:
    if function is None:
        return None
    r, g, b, a = (ctypes.c_uint(0) for _ in range(4))
    if not function(obj, ctypes.byref(r), ctypes.byref(g), ctypes.byref(b), ctypes.byref(a)):
        return None
    return (r.value & 0xFF) << 24 | (g.value & 0xFF) << 16 | (b.value & 0xFF) << 8 | a.value & 0xFF


def _union(boxes: list[Box]) -> Box:
    if not boxes:
        return _NAN_BOX
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _weighted_median(values: list[tuple[float, int]]) -> float:
    ordered = sorted(values)
    half = sum(weight for _value, weight in ordered) / 2.0
    seen = 0
    for value, weight in ordered:
        seen += weight
        if seen >= half:
            return value
    return ordered[-1][0]


def _objects_in_page(api: _Api, page) -> dict[int, tuple[float, ...] | None]:
    """Address of every text object on *page* -> the matrix of the forms holding it."""
    pdfium_c = api.c
    top_level = [
        pdfium_c.FPDFPage_GetObject(page.raw, index)
        for index in range(pdfium_c.FPDFPage_CountObjects(page.raw))
    ]
    found: list = []
    nt._collect_text_objects(pdfium_c, top_level, None, None, 0, found)
    return {_address(obj): matrix for _parent, obj, matrix in found}


@dataclass(slots=True)
class _Objects:
    """Per-object columns of one page, filled once per object."""

    font: list[int]
    tf: list[float]
    size_eff: list[float]
    matrix: list[tuple[float, float, float, float]]
    fill: list[int]
    stroke: list[int]
    mode: list[int]
    mcid: list[int]
    artifact: list[bool]
    flags: list[int]


def _read_objects(
    api: _Api, page, handles: list, fonts: FontTable, found: list | None = None
) -> _Objects:
    """The columns of the text objects in *handles*.

    *found* is the furniture strip's walk of the page (``open_text_page``'s
    *walk*); without it the page is walked again.
    """
    if not handles:
        walk = {}
    elif found:
        walk = {_address(obj): matrix for _parent, obj, matrix in found}
    else:
        walk = _objects_in_page(api, page)
    out = _Objects([], [], [], [], [], [], [], [], [], [])
    font_memo: dict[int, int] = {}
    size = ctypes.c_float(0.0)
    for handle in handles:
        flags = 0
        parent = walk.get(_address(handle), ...)
        if parent is ...:
            flags |= OBJ_NOT_IN_WALK
            parent = None
        own = nt._object_matrix(api.c, handle) if api.FPDFPageObj_GetMatrix else None
        if own is None:
            flags |= OBJ_NO_MATRIX
            own = _IDENTITY
        a, b, c, d, _e, _f = nt._compose(own, parent)
        tf = _NAN
        if api.FPDFTextObj_GetFontSize and api.FPDFTextObj_GetFontSize(handle, ctypes.byref(size)):
            tf = size.value
        else:
            flags |= OBJ_NO_FONT_SIZE
        fill = _color(api.FPDFPageObj_GetFillColor, handle)
        if fill is None:
            flags |= OBJ_NO_FILL
        stroke = _color(api.FPDFPageObj_GetStrokeColor, handle)
        if stroke is None:
            flags |= OBJ_NO_STROKE
        out.font.append(
            fonts.font_id(api, api.FPDFTextObj_GetFont(handle), font_memo)
            if api.FPDFTextObj_GetFont
            else -1
        )
        out.tf.append(tf)
        out.size_eff.append(tf * math.sqrt(abs(a * d - b * c)))
        out.matrix.append((a, b, c, d))
        out.fill.append(fill or 0)
        out.stroke.append(stroke or 0)
        out.mode.append(
            api.FPDFTextObj_GetTextRenderMode(handle) if api.FPDFTextObj_GetTextRenderMode else -1
        )
        out.mcid.append(
            api.FPDFPageObj_GetMarkedContentID(handle) if api.FPDFPageObj_GetMarkedContentID else -1
        )
        out.artifact.append(_in_artifact(api, handle))
        out.flags.append(flags)
    return out


def _read_glyphs(api: _Api, textpage, n_chars: int, boxes: list) -> tuple[list, ...]:
    """The second pass over the chars: object, origin, loose box and flags of each."""
    handle = textpage.raw
    get_object = api.FPDFText_GetTextObject
    get_origin = api.FPDFText_GetCharOrigin
    get_loose = api.FPDFText_GetLooseCharBox
    is_generated = api.FPDFText_IsGenerated
    is_hyphen = api.FPDFText_IsHyphen
    map_error = api.FPDFText_HasUnicodeMapError
    text_index = api.FPDFText_GetTextIndexFromCharIndex
    x, y = ctypes.c_double(0.0), ctypes.c_double(0.0)
    rect = api.c.FS_RECTF()
    rows: list[int] = []
    handles: list = []
    index_of: dict[int, int] = {}
    origins: list[float] = []
    loose: list[float] = []
    gflags: list[int] = []
    for i in range(n_chars):
        row = -1
        if get_object is not None:
            obj = get_object(handle, i)
            if obj:
                address = _address(obj)
                row = index_of.get(address, -1)
                if row < 0:
                    row = index_of[address] = len(handles)
                    handles.append(obj)
        rows.append(row)
        flags = 0
        if is_generated is not None and is_generated(handle, i) == 1:
            flags |= GLYPH_GENERATED
        if is_hyphen is not None and is_hyphen(handle, i) == 1:
            flags |= GLYPH_HYPHEN
        if map_error is not None and map_error(handle, i) == 1:
            flags |= GLYPH_MAP_ERROR
        if text_index is not None and text_index(handle, i) < 0:
            flags |= GLYPH_EXCLUDED
        if boxes[i] is None:
            flags |= GLYPH_NO_BOX
        if get_origin is not None and get_origin(handle, i, x, y):
            origins += (x.value, y.value)
        else:
            origins += (_NAN, _NAN)
            flags |= GLYPH_NO_ORIGIN
        if get_loose is not None and get_loose(handle, i, rect):
            loose += (rect.left, rect.bottom, rect.right, rect.top)
        else:
            loose += _NAN_BOX
            flags |= GLYPH_NO_LOOSE_BOX
        gflags.append(flags)
    return rows, handles, origins, loose, gflags


def _text_source(
    page, crop_box: Box, codes: list[int], rows: list[int], gflags: list[int], modes: list[int]
) -> tuple[str, Decided, float | None]:
    """``native`` or ``invisible_layer``, its decision, and the image coverage if read.

    The inspection's own invisible-layer rule (the ``bibr.ocr.native_text``
    helpers ``_is_invisible_text_layer_page`` uses), on the harvested
    columns. The decision's score is the invisible share.
    """
    share = nt._invisible_share(
        len(codes),
        codes.__getitem__,
        lambda index: bool(gflags[index] & GLYPH_GENERATED),
        lambda index: modes[rows[index]] if rows[index] >= 0 else None,
    )
    decided = Decided("text_source", INVISIBLE_LAYER_RULE, score=share)
    if not nt._share_qualifies(share):
        return "native", decided, None
    coverage = nt._page_image_coverage(page, crop_box)
    if nt._coverage_qualifies(coverage):
        return "invisible_layer", decided, coverage
    return "native", decided, coverage


@dataclass(slots=True)
class _OpenSpan:
    start: int
    obj: int = -1
    key: tuple | None = None
    size: float = 0.0
    baseline: float = 0.0


def _spans_and_lines(
    records: list[tuple[str, float, float, bool]],
    record_src: list[int],
    rows: list[int],
    boxes: list,
    origins: list[float],
    objects: _Objects,
) -> tuple[list[tuple], list[tuple[int, int]]]:
    """Group the records into spans and lines.

    A span is a run of records whose glyphs share font, effective size, fill,
    render mode and marked-content id, on one baseline; whitespace and
    inserted spaces join the open span. A line ends at a newline record and
    where the text wraps to a new printed line without one: a glyph wholly
    below the previous glyph and to its left (the rule of
    ``bibr.ocr.pdf_inspection._break_wrapped_lines``).

    Returns spans as ``(start, end, object row, bbox, baseline)`` and lines as
    ``(first span, end span)``.
    """
    spans: list[tuple] = []
    lines: list[tuple[int, int]] = []
    line_start = 0
    current: _OpenSpan | None = None
    span_boxes: list[Box] = []
    previous_box: Box | None = None

    def close_span(end: int) -> None:
        nonlocal current
        if current is not None and end > current.start:
            spans.append((current.start, end, current.obj, _union(span_boxes), current.baseline))
        current = None
        span_boxes.clear()

    def close_line(end: int) -> None:
        nonlocal line_start
        close_span(end)
        if len(spans) > line_start:
            lines.append((line_start, len(spans)))
        line_start = len(spans)

    for index, (ch, _cx, _cy, is_newline) in enumerate(records):
        if is_newline:
            close_line(index)
            previous_box = None
            continue
        src = record_src[index]
        row = rows[src] if src >= 0 else -1
        if ch.isspace() or row < 0:
            if current is None:
                current = _OpenSpan(index)
            continue
        box = boxes[src]
        if (
            box is not None
            and previous_box is not None
            and box[3] <= previous_box[1]
            and box[0] < previous_box[0]
        ):
            close_line(index)
        if box is not None:
            previous_box = box
        size = objects.size_eff[row]
        a, b, _c, _d = objects.matrix[row]
        norm = math.hypot(a, b)
        ox, oy = origins[2 * src], origins[2 * src + 1]
        baseline = (-b * ox + a * oy) / norm if norm > 0 else oy
        key = (objects.font[row], objects.fill[row], objects.mode[row], objects.mcid[row])
        if current is not None and current.key is not None:
            same_size = abs(size - current.size) <= _SPAN_SIZE_TOLERANCE * max(
                abs(size), abs(current.size)
            )
            same_baseline = abs(baseline - current.baseline) <= _SPAN_BASELINE_TOLERANCE * max(
                abs(current.size), 1e-6
            )
            if key != current.key or not same_size or not same_baseline:
                close_span(index)
        if current is None:
            current = _OpenSpan(index)
        if current.key is None:
            current.key = key
            current.obj = row
            current.size = size
            current.baseline = baseline
        if box is not None:
            span_boxes.append(box)
    close_line(len(records))
    return spans, lines


def _script_tags(
    page_index: int,
    spans: list[tuple],
    lines: list[tuple[int, int]],
    records: list[tuple[str, float, float, bool]],
    objects: _Objects,
) -> list[RoleTag]:
    """Superscript and subscript tags of the spans (rule ``superscript/1``)."""
    tags: list[RoleTag] = []
    for line_index, (first, end) in enumerate(lines):
        weighted: list[tuple[int, float, float, int]] = []
        for span_index in range(first, end):
            start, stop, row, _bbox, baseline = spans[span_index]
            if row < 0 or not math.isfinite(baseline):
                continue
            glyphs = sum(1 for k in range(start, stop) if not records[k][0].isspace())
            size = objects.size_eff[row]
            if glyphs and math.isfinite(size) and size > 0:
                weighted.append((span_index, size, baseline, glyphs))
        if len(weighted) < 2:
            continue
        sizes: Counter[float] = Counter()
        for _span, size, _baseline, glyphs in weighted:
            sizes[round(size, 2)] += glyphs
        dominant = max(sizes.items(), key=lambda item: (item[1], item[0]))[0]
        base = _weighted_median(
            [
                (baseline, glyphs)
                for _span, size, baseline, glyphs in weighted
                if abs(round(size, 2) - dominant) <= _SPAN_SIZE_TOLERANCE * dominant
            ]
        )
        for span_index, size, baseline, _glyphs in weighted:
            if size > _SCRIPT_MAX_SIZE_RATIO * dominant:
                continue
            shift = (baseline - base) / dominant
            if shift >= _SUPERSCRIPT_MIN_RAISE:
                role = "superscript"
            elif shift <= -_SUBSCRIPT_MIN_DROP:
                role = "subscript"
            else:
                continue
            tags.append(
                RoleTag(
                    ids.span(page_index, span_index),
                    role,
                    Decided(
                        SCRIPT_RULE.component,
                        SCRIPT_RULE.version,
                        score=round(shift, 4),
                        calibrated=False,
                        evidence=(ids.line(page_index, line_index),),
                    ),
                )
            )
    return tags


def _array(values, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.dtype(COLUMN_DTYPES[name]))
    if shape is not None:
        array = array.reshape(shape)
    return array


@dataclass(slots=True)
class _PageDraft:
    """What the locked pass read from one page; :func:`_complete` builds the rest."""

    page: Page
    codes: list[int]
    boxes: list
    records: list[tuple[str, float, float, bool]]
    record_src: list[int]
    record_flags: list[int]
    rows: list[int]
    origins: list[float]
    loose: list[float]
    gflags: list[int]
    objects: _Objects


def page_header(
    page,
    *,
    page_index: int,
    crop_box: Box,
    rotation: int,
    furniture: list[tuple],
    budget: RenderBudget | None,
    version: str,
) -> Page:
    """The page as known before its text layer is read: size, furniture and render recipe.

    *furniture* comes from ``open_text_page(page, ..., furniture)``. The
    ``text_source`` stays ``unread`` until :func:`read_page` decides it, so a
    page whose read fails keeps the rest. Called under ``pdfium_lock``.
    """
    width, height = page.get_size()
    render = None
    if budget is not None:
        render = RenderRecipe(
            version,
            layout_render_dpi(float(width), float(height), budget),
            crop_box,
            rotation,
            _RENDER_FLAGS,
        )
    removed = [
        Furniture(
            furniture_id=ids.furniture(page_index, n),
            page=page_index,
            kind=kind,
            bbox_pdf=None if box is None else as_box(box),
            text=text,
            decided=_FURNITURE_DECIDED[kind],
        )
        for n, (kind, box, text) in enumerate(furniture)
    ]
    return Page(
        index=page_index,
        label=None,
        width=float(width),
        height=float(height),
        crop_box=crop_box,
        rotation=rotation,
        text_source="unread",
        cols=None,
        furniture=removed,
        render=render,
    )


def read_page(
    built: Page,
    page,
    textpage,
    *,
    trace: nt.PageCharTrace | None,
    records: list[tuple[str, float, float, bool]] | None,
    fonts: FontTable,
    api: _Api | None = None,
    walk: list | None = None,
) -> _PageDraft:
    """Read the text layer of *built*, the :func:`page_header` of *page*: the rest that needs pdfium.

    Decides ``built.text_source``. *trace* and *records* come from
    ``_build_page_char_records(textpage, trace)``, and *walk* from
    ``open_text_page(page, ..., walk)``. Called under ``pdfium_lock``.
    """
    api = api or _Api()
    crop_box = built.crop_box
    if crop_box is None:
        raise ValueError("the page has no size")
    n_chars = textpage.count_chars()
    empty = _Objects([], [], [], [], [], [], [], [], [], [])
    if n_chars == 0:
        built.text_source = "ocr"
        return _PageDraft(built, [], [], [], [], [], [], [], [], [], empty)
    if trace is None or records is None or not trace.complete or len(trace.codes) != n_chars:
        raise RuntimeError("the page's char records are unavailable")
    if len(trace.record_src) != len(records):
        raise RuntimeError("the char trace does not match the records")
    rows, handles, origins, loose, gflags = _read_glyphs(api, textpage, n_chars, trace.boxes)
    objects = _read_objects(api, page, handles, fonts, walk)
    for i, row in enumerate(rows):
        if row >= 0 and objects.mode[row] in nt._INVISIBLE_TEXT_RENDER_MODES:
            gflags[i] |= GLYPH_INVISIBLE_RENDER
    built.text_source, built.text_source_decided, built.image_coverage = _text_source(
        page, crop_box, trace.codes, rows, gflags, objects.mode
    )
    return _PageDraft(
        built,
        trace.codes,
        trace.boxes,
        records,
        trace.record_src,
        trace.record_flags,
        rows,
        origins,
        loose,
        gflags,
        objects,
    )


@dataclass(slots=True)
class _PackedDraft:
    """A :class:`_PageDraft` held as numpy arrays from the locked read until ``finish``.

    The draft's per-glyph Python lists cost about 760 bytes a glyph, and
    every page of the document waited in them for ``finish``. Packed, a
    glyph costs about 110 bytes; :func:`_unpack` gives back the exact lists.
    A char with no tight box is the one flagged ``GLYPH_NO_BOX``, and a
    record whose char is not one code point keeps it in ``rec_text``.
    """

    page: Page
    codes: np.ndarray
    boxes: np.ndarray
    rec_cp: np.ndarray
    rec_text: dict[int, str]
    rec_cx: np.ndarray
    rec_cy: np.ndarray
    rec_newline: np.ndarray
    record_src: np.ndarray
    record_flags: np.ndarray
    rows: np.ndarray
    origins: np.ndarray
    loose: np.ndarray
    gflags: np.ndarray
    objects: dict[str, np.ndarray]


# The packed dtype of each _Objects list: float64 where the value feeds a
# computation in _complete, the column's own dtype elsewhere.
_OBJECT_DTYPES = {
    "font": np.int32,
    "tf": np.float64,
    "size_eff": np.float64,
    "matrix": np.float64,
    "fill": np.uint32,
    "stroke": np.uint32,
    "mode": np.int8,
    "mcid": np.int32,
    "artifact": np.bool_,
    "flags": np.uint8,
}


def _pack(draft: _PageDraft) -> _PackedDraft:
    rec_cp: list[int] = []
    rec_text: dict[int, str] = {}
    for index, record in enumerate(draft.records):
        if len(record[0]) == 1:
            rec_cp.append(ord(record[0]))
        else:
            rec_cp.append(0)
            rec_text[index] = record[0]
    objects = draft.objects
    return _PackedDraft(
        page=draft.page,
        codes=np.asarray(draft.codes, dtype=np.uint32),
        boxes=np.asarray(
            [_NAN_BOX if box is None else box for box in draft.boxes], dtype=np.float64
        ).reshape(-1, 4),
        rec_cp=np.asarray(rec_cp, dtype=np.uint32),
        rec_text=rec_text,
        rec_cx=np.asarray([record[1] for record in draft.records], dtype=np.float64),
        rec_cy=np.asarray([record[2] for record in draft.records], dtype=np.float64),
        rec_newline=np.asarray([record[3] for record in draft.records], dtype=np.bool_),
        record_src=np.asarray(draft.record_src, dtype=np.int32),
        record_flags=np.asarray(draft.record_flags, dtype=np.uint8),
        rows=np.asarray(draft.rows, dtype=np.int32),
        origins=np.asarray(draft.origins, dtype=np.float64),
        # Only ever stored as float32.
        loose=np.asarray(draft.loose, dtype=np.float32),
        gflags=np.asarray(draft.gflags, dtype=np.uint16),
        objects={
            name: np.asarray(getattr(objects, name), dtype=dtype)
            for name, dtype in _OBJECT_DTYPES.items()
        },
    )


def _unpack(packed: _PackedDraft) -> _PageDraft:
    gflags = packed.gflags.tolist()
    rec_text = packed.rec_text
    records = [
        (rec_text[index] if index in rec_text else chr(code), cx, cy, newline)
        for index, (code, cx, cy, newline) in enumerate(
            zip(
                packed.rec_cp.tolist(),
                packed.rec_cx.tolist(),
                packed.rec_cy.tolist(),
                packed.rec_newline.tolist(),
                strict=True,
            )
        )
    ]
    columns = {name: array.tolist() for name, array in packed.objects.items()}
    columns["matrix"] = [tuple(matrix) for matrix in columns["matrix"]]
    return _PageDraft(
        packed.page,
        packed.codes.tolist(),
        [
            None if flags & GLYPH_NO_BOX else tuple(box)
            for box, flags in zip(packed.boxes.tolist(), gflags, strict=True)
        ],
        records,
        packed.record_src.tolist(),
        packed.record_flags.tolist(),
        packed.rows.tolist(),
        packed.origins.tolist(),
        packed.loose.tolist(),
        gflags,
        _Objects(**columns),
    )


def _complete(draft: _PageDraft) -> tuple[Page, list[RoleTag]]:
    """Columns, spans, lines and script tags of a drafted page (no pdfium calls)."""
    built = draft.page
    if not draft.codes:
        return built, []
    records = draft.records
    rec_cp: list[int] = []
    rec_text: dict[int, str] = {}
    for index, (ch, _cx, _cy, _newline) in enumerate(records):
        if len(ch) == 1:
            rec_cp.append(ord(ch))
        else:
            rec_cp.append(ord(ch[0]) if ch else 0)
            rec_text[index] = ch
    objects = draft.objects
    spans, lines = _spans_and_lines(
        records, draft.record_src, draft.rows, draft.boxes, draft.origins, objects
    )
    tags = _script_tags(built.index, spans, lines, records, objects)
    n_chars = len(draft.codes)
    n_obj = len(objects.font)
    line_boxes = [
        _union([spans[k][3] for k in range(first, end) if not math.isnan(spans[k][3][0])])
        for first, end in lines
    ]
    built.cols = PageColumns(
        cp=_array(draft.codes, "cp"),
        box=_array(
            [box if box is not None else _NAN_BOX for box in draft.boxes], "box", (n_chars, 4)
        ),
        loose=_array(draft.loose, "loose", (n_chars, 4)),
        origin=_array(draft.origins, "origin", (n_chars, 2)),
        obj=_array(draft.rows, "obj"),
        gflags=_array(draft.gflags, "gflags"),
        obj_font=_array(objects.font, "obj_font"),
        obj_tf=_array(objects.tf, "obj_tf"),
        obj_size_eff=_array(objects.size_eff, "obj_size_eff"),
        obj_matrix=_array(objects.matrix, "obj_matrix", (n_obj, 4)),
        obj_fill=_array(objects.fill, "obj_fill"),
        obj_stroke=_array(objects.stroke, "obj_stroke"),
        obj_render_mode=_array(objects.mode, "obj_render_mode"),
        obj_mcid=_array(objects.mcid, "obj_mcid"),
        obj_artifact=_array(objects.artifact, "obj_artifact"),
        obj_flags=_array(objects.flags, "obj_flags"),
        rec_cp=_array(rec_cp, "rec_cp"),
        rec_cx=_array([record[1] for record in records], "rec_cx"),
        rec_cy=_array([record[2] for record in records], "rec_cy"),
        rec_newline=_array([record[3] for record in records], "rec_newline"),
        rec_src=_array(draft.record_src, "rec_src"),
        rec_flags=_array(draft.record_flags, "rec_flags"),
        span_rec=_array([(span[0], span[1]) for span in spans], "span_rec", (len(spans), 2)),
        span_obj=_array([span[2] for span in spans], "span_obj"),
        span_bbox=_array([span[3] for span in spans], "span_bbox", (len(spans), 4)),
        span_baseline=_array([span[4] for span in spans], "span_baseline"),
        line_span=_array(lines, "line_span", (len(lines), 2)),
        line_bbox=_array(line_boxes, "line_bbox", (len(lines), 4)),
        line_block=np.full(len(lines), -1, dtype=np.dtype(COLUMN_DTYPES["line_block"])),
        rec_text=rec_text,
    )
    return built, tags


class LayerBuilder:
    """Collects a document's pages into a :class:`DocumentLayer`.

    Use :meth:`records` and :meth:`add_page` per page, on the text page
    ``open_text_page(page, ..., furniture)`` returned, under ``pdfium_lock``;
    then :meth:`finish`, which needs no pdfium and so can run after the lock
    is released.
    """

    def __init__(self, pdf_bytes: bytes, budget: RenderBudget | None) -> None:
        self.version = pdfium_version()
        self.source_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
        self.budget = budget
        self.fonts = FontTable()
        # Drafted pages, and the pages that failed before a draft, in page order.
        self.drafts: deque[_PackedDraft | Page] = deque()
        self._added: set[int] = set()
        self.errors: dict[str, str] = {}
        self.missing = missing_apis()
        self._api = _Api()

    def records(
        self, textpage, page_index: int
    ) -> tuple[nt.PageCharTrace | None, list[tuple[str, float, float, bool]] | None]:
        """The page's char records with their trace, or ``(None, None)`` on failure."""
        trace = nt.PageCharTrace()
        try:
            return trace, nt._build_page_char_records(textpage, trace)
        except Exception as exc:  # noqa: BLE001 - the fill rebuilds them and reports its own error
            self.errors[f"records:{page_index}"] = _error_text(exc)
            return None, None

    def add_page(
        self,
        page,
        textpage,
        *,
        page_index: int,
        crop_box: Box,
        rotation: int,
        trace: nt.PageCharTrace | None,
        records: list[tuple[str, float, float, bool]] | None,
        furniture: list[tuple],
        walk: list | None = None,
    ) -> None:
        failed = [text for kind, _box, text in furniture if kind == "error"]
        if failed:
            self.errors[f"furniture:{page_index}"] = failed[0]
            furniture = [item for item in furniture if item[0] != "error"]
        key = f"harvest:{page_index}"
        try:
            built = page_header(
                page,
                page_index=page_index,
                crop_box=crop_box,
                rotation=rotation,
                furniture=furniture,
                budget=self.budget,
                version=self.version,
            )
        except Exception as exc:  # noqa: BLE001 - a layer page never fails the paper
            self.page_failed(page_index, key, exc)
            return
        try:
            packed = _pack(
                read_page(
                    built,
                    page,
                    textpage,
                    trace=trace,
                    records=records,
                    fonts=self.fonts,
                    api=self._api,
                    walk=walk,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a layer page never fails the paper
            self.page_failed(page_index, key, exc, built)
            return
        self.drafts.append(packed)
        self._added.add(page_index)

    def page_failed(
        self, page_index: int, key: str, exc: BaseException, page: Page | None = None
    ) -> None:
        """Record *exc* under *key*, and keep the page with the error unless it was added.

        *page* is what was read of it before the failure (a :func:`page_header`
        and, once decided, its text source); without it the page could not be
        opened or sized and is kept unread, with None geometry.
        """
        self.errors[key] = _error_text(exc)
        if page_index in self._added:
            return
        self._added.add(page_index)
        if page is None:
            page = _unread_page(page_index)
        page.cols = None
        page.error = self.errors[key]
        self.drafts.append(page)

    def finish(self) -> DocumentLayer:
        """Complete the drafted pages; needs no pdfium, so call it after the lock."""
        pages: list[Page] = []
        roles: list[RoleTag] = []
        # Each draft is released once its page is complete.
        while self.drafts:
            packed = self.drafts.popleft()
            if isinstance(packed, Page):
                pages.append(packed)
                continue
            try:
                built, tags = _complete(_unpack(packed))
            except Exception as exc:  # noqa: BLE001 - a layer page never fails the paper
                key = f"harvest:{packed.page.index}"
                self.errors[key] = _error_text(exc)
                packed.page.cols = None
                packed.page.error = self.errors[key]
                pages.append(packed.page)
                continue
            pages.append(built)
            roles.extend(tags)
        # A fact no examined page shows is unknown, not absent, while another
        # page was not examined. A page's text source counts once decided, even
        # if the page failed later; marked content needs the page's columns.
        decided = [page for page in pages if page.text_source != "unread"]
        read = [page for page in pages if page.error is None]

        def seen(found: bool, examined: list[Page]) -> bool | None:
            if found:
                return True
            return False if pages and len(examined) == len(pages) else None

        native = seen(any(page.text_source == "native" for page in decided), decided)
        mcids = any(
            page.cols is not None and bool((page.cols.obj_mcid >= 0).any()) for page in read
        )
        presence = Presence(
            has_text_layer=seen(
                any(page.text_source in _TEXT_LAYER_SOURCES for page in decided), decided
            ),
            is_scan=None if native is None else not native,
            has_invisible_layer=seen(
                any(page.text_source == "invisible_layer" for page in decided), decided
            ),
            has_mcids=(
                None if "FPDFPageObj_GetMarkedContentID" in self.missing else seen(mcids, read)
            ),
            missing_apis=self.missing,
        )
        return DocumentLayer(
            version=LAYER_VERSION,
            pdfium=self.version,
            source_sha256=self.source_sha256,
            index_frame=INDEX_FRAME,
            pages=pages,
            fonts=list(self.fonts.fonts),
            roles=roles,
            presence=presence,
            component_errors=dict(self.errors),
        )


def _unread_page(page_index: int) -> Page:
    """A page that could not be opened or sized: unread, with None geometry."""
    return Page(
        index=page_index,
        label=None,
        width=None,
        height=None,
        crop_box=None,
        rotation=None,
        text_source="unread",
        cols=None,
    )


def build_document_layer(
    pdf_bytes: bytes,
    page_indices: Iterable[int] | Callable[[int], Iterable[int]],
    *,
    budget: RenderBudget | None,
) -> DocumentLayer:
    """Build the layer for *page_indices* of a PDF, as ``inspect_pdf`` does inline.

    *page_indices* may be a function of the document's page count, so the
    PDF is parsed once. Opens the document under ``pdfium_lock`` and reads
    each page through ``open_text_page``, in page order, so the glyph
    indexes match the inline build.
    """
    import pypdfium2

    from bibr.ocr.utils import pdfium_lock

    builder = LayerBuilder(pdf_bytes, budget)
    with pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            if callable(page_indices):
                page_indices = page_indices(len(doc))
            for page_index in page_indices:
                try:
                    page = doc[page_index]
                except Exception as exc:  # noqa: BLE001
                    builder.page_failed(page_index, f"page:{page_index}", exc)
                    continue
                try:
                    crop_box = nt._page_crop_box(page)
                    rotation = nt._page_rotation(page)
                    furniture: list[tuple] = []
                    walk: list[tuple] = []
                    textpage = nt.open_text_page(page, [], furniture, walk)
                    try:
                        trace, records = builder.records(textpage, page_index)
                        builder.add_page(
                            page,
                            textpage,
                            page_index=page_index,
                            crop_box=crop_box,
                            rotation=rotation,
                            trace=trace,
                            records=records,
                            furniture=furniture,
                            walk=walk,
                        )
                    finally:
                        textpage.close()
                except Exception as exc:  # noqa: BLE001
                    builder.page_failed(page_index, f"page:{page_index}", exc)
                finally:
                    page.close()
        finally:
            doc.close()
    return builder.finish()
