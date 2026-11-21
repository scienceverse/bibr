"""Recover ordered per-line records (text + bbox + font) for a paper's
references section from the PDF text layer.

Scoped by the text-layer "References" header (NOT by reference-labeled layout
regions), so layout-dropped or text-mislabeled references are still captured.
Coordinates are PDF points, origin bottom-left (y increases upward).

The capture (:func:`recover_reference_lines`) runs in the OCR-stage native-text
pass while ``pdf_bytes`` is alive; results serialize to plain dicts
(:func:`record_to_dict`) so they can ride on ``PaperContents.ref_line_geometry``
into the extract stage, where the geometry segmenter consumes them.
"""

from __future__ import annotations

import ctypes
import re
from collections import defaultdict
from dataclasses import dataclass

from bibr.ocr.ref_patterns import _REF_HEADER_RE
from bibr.ocr.utils import pdfium_lock

_WS = re.compile(r"\s+")
_PAGE_NUMBER = re.compile(r"\b\d+\b")
# Header/footer furniture belongs at a page edge. Two lines covers a running
# title paired with a page number without scanning bibliography body lines for
# coincidental repeated text.
_EDGE_LINES = 2


@dataclass(frozen=True)
class LineRecord:
    text: str
    page: int
    x0: float
    y_top: float
    x1: float
    y_bottom: float
    font_size: float


def group_chars_into_lines(
    chars: list[tuple[str, tuple[float, float, float, float]]], page: int
) -> list[LineRecord]:
    """Group a page's chars (in reading order) into non-empty text lines.

    A new line begins at each ``\\n``/``\\r``. Line bbox is the union of its
    non-whitespace char boxes; ``x0`` includes leading-space indent;
    ``font_size`` is the median glyph height.
    """
    lines: list[LineRecord] = []
    buf: list[tuple[str, tuple[float, float, float, float]]] = []

    def flush() -> None:
        if not buf:
            return
        text = "".join(c for c, _ in buf).strip()
        if text:
            x0 = float("inf")
            x1 = float("-inf")
            y_bottom = float("inf")
            y_top = float("-inf")
            heights: list[float] = []
            for c, b in buf:
                if b[0] < x0:
                    x0 = b[0]
                if not c.isspace():
                    if b[2] > x1:
                        x1 = b[2]
                    if b[1] < y_bottom:
                        y_bottom = b[1]
                    if b[3] > y_top:
                        y_top = b[3]
                    heights.append(b[3] - b[1])
            if heights:
                heights.sort()
                fs = heights[len(heights) // 2]
                lines.append(LineRecord(text, page, x0, y_top, x1, y_bottom, fs))
        buf.clear()

    for ch, box in chars:
        if ch in ("\n", "\r"):
            flush()
        elif ch:
            buf.append((ch, box))
    flush()
    return lines


def _extract_page_chars(
    textpage,
    *,
    codes: list[int] | None = None,
    char_boxes: list[tuple[float, float, float, float] | None] | None = None,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    import pypdfium2 as pdfium

    from bibr.ocr.native_text import compose_spacing_accents

    n_chars = textpage.count_chars()
    if not n_chars:
        return []

    pdfium_c = pdfium.raw
    raw = textpage.raw
    get_unicode = pdfium_c.FPDFText_GetUnicode
    get_text_idx = pdfium_c.FPDFText_GetTextIndexFromCharIndex
    use_codes = codes is not None and len(codes) == n_chars
    use_boxes = char_boxes is not None and len(char_boxes) == n_chars

    # Bulk-read the page's UCS-2 text buffer once instead of calling
    # textpage.get_text_range(i, 1) (which allocates a ctypes buffer and decodes
    # UTF-16LE) for every single character on the page.
    bulk_buf = None
    bulk_t_start = 0
    bulk_len = 0
    is_real_tp = isinstance(textpage, pdfium.PdfTextPage)
    if is_real_tp:
        active = textpage._get_active_text_range(0, n_chars - 1)
        if active is None:
            # No active BMP text chars; still check non-BMP if any exist.
            bulk_buf = ()
            bulk_len = 0
        else:
            t_start, t_end, l_passive, r_passive = active
            in_count = t_end + 2 - t_start
            buf = (ctypes.c_ushort * in_count)()
            out_count = pdfium_c.FPDFText_GetText(
                raw, l_passive, n_chars - l_passive - r_passive, buf
            )
            if out_count - 1 == t_end - t_start + 1:
                bulk_buf = buf
                bulk_t_start = t_start
                bulk_len = out_count - 1

    if is_real_tp and not use_boxes:
        _c_l = ctypes.c_double()
        _c_r = ctypes.c_double()
        _c_b = ctypes.c_double()
        _c_t = ctypes.c_double()
        _b_l = ctypes.byref(_c_l)
        _b_r = ctypes.byref(_c_r)
        _b_b = ctypes.byref(_c_b)
        _b_t = ctypes.byref(_c_t)
        _get_box = pdfium_c.FPDFText_GetCharBox

        def _box(idx: int) -> tuple[float, float, float, float]:
            if _get_box(raw, idx, _b_l, _b_r, _b_b, _b_t):
                return (_c_l.value, _c_b.value, _c_r.value, _c_t.value)
            return textpage.get_charbox(idx)
    elif use_boxes:

        def _box(idx: int) -> tuple[float, float, float, float]:
            cached = char_boxes[idx]
            if cached is not None:
                return cached
            return textpage.get_charbox(idx)
    else:

        def _box(idx: int) -> tuple[float, float, float, float]:
            return textpage.get_charbox(idx)

    out: list[tuple[str, tuple[float, float, float, float]]] = []
    out_append = out.append
    i = 0
    while i < n_chars:
        # get_text_range is UCS-2: it drops a char beyond the BMP, whether
        # pdfium holds it as one code or as a UTF-16 surrogate pair. Read
        # those from the codes, as the native char records do.
        code = codes[i] if use_codes else get_unicode(raw, i)
        width = 1
        if 0xD800 <= code <= 0xDBFF and i + 1 < n_chars:
            low = codes[i + 1] if use_codes else get_unicode(raw, i + 1)
            if 0xDC00 <= low <= 0xDFFF:
                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                width = 2
        if 0xFFFF < code <= 0x10FFFF:
            # A char left out of the page text stays out.
            if get_text_idx(raw, i) >= 0:
                b0 = _box(i)
                if width == 2:
                    b1 = _box(i + 1)
                    box = (
                        b0[0] if b0[0] <= b1[0] else b1[0],
                        b0[1] if b0[1] <= b1[1] else b1[1],
                        b0[2] if b0[2] >= b1[2] else b1[2],
                        b0[3] if b0[3] >= b1[3] else b1[3],
                    )
                else:
                    box = b0
                out_append((chr(code), box))
            i += width
            continue
        if bulk_buf is not None:
            tidx = get_text_idx(raw, i)
            if tidx >= 0:
                pos = tidx - bulk_t_start
                if 0 <= pos < bulk_len:
                    u = bulk_buf[pos]
                    if not (0xD800 <= u <= 0xDFFF):
                        out_append((chr(u), _box(i)))
        else:
            ch = textpage.get_text_range(i, 1)
            if ch:
                out_append((ch, _box(i)))
        i += 1
    return compose_spacing_accents(out)


def recover_reference_lines(pdf_bytes: bytes) -> list[LineRecord]:
    """Return ordered reference-section lines, or ``[]`` when there is no text
    layer or no References header."""
    import pypdfium2

    from bibr.ocr.native_text import open_text_page

    page_lines: dict[int, list[LineRecord]] = {}
    header_page: int | None = None
    with pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            for pi in range(len(doc)):
                page = doc[pi]
                try:
                    tp = open_text_page(page)
                    try:
                        full = tp.get_text_range() or ""
                        if header_page is None and _REF_HEADER_RE.search(full):
                            header_page = pi
                        if header_page is not None:
                            page_lines[pi] = group_chars_into_lines(_extract_page_chars(tp), pi)
                    finally:
                        tp.close()
                finally:
                    page.close()
        finally:
            doc.close()

    return reference_lines_from_pages(page_lines, header_page)


def reference_lines_from_pages(
    page_lines: dict[int, list[LineRecord]], header_page: int | None
) -> list[LineRecord]:
    """Flatten captured page lines starting immediately after References.

    Running heads and footers are part of PDF text layers, so a bibliography
    crossing pages can otherwise inherit them as reference continuations. Only
    repeated first/last page lines are removed: a genuine continuation at the
    top of one page is retained unless it repeats as page furniture elsewhere.
    A line inside the page is never furniture, even when its digits-masked text
    matches an edge line's ("2." against a page-top "4.", "2015" against a
    page number).
    """
    if header_page is None:
        return []

    furniture_pages: dict[str, set[int]] = defaultdict(set)
    for pi, plines in page_lines.items():
        for line in [*plines[:_EDGE_LINES], *plines[-_EDGE_LINES:]]:
            key = _furniture_key(line)
            if key:
                furniture_pages[key].add(pi)
    furniture = {key for key, pages in furniture_pages.items() if len(pages) >= 2}

    lines: list[LineRecord] = []
    for pi in sorted(page_lines):
        plines = page_lines[pi]
        start = 0
        if pi == header_page:
            cut = next(
                (k for k, ln in enumerate(plines) if _REF_HEADER_RE.match(ln.text.strip())),
                None,
            )
            if cut is not None:
                start = cut + 1
        last_body = len(plines) - _EDGE_LINES
        lines.extend(
            line
            for k, line in enumerate(plines[start:], start=start)
            if not ((k < _EDGE_LINES or k >= last_body) and _furniture_key(line) in furniture)
        )
    return lines


def _furniture_key(line: LineRecord) -> str:
    """Case- and digit-insensitive form under which page furniture repeats."""
    return _PAGE_NUMBER.sub("#", _WS.sub(" ", line.text.casefold()).strip())


def record_to_dict(r: LineRecord) -> dict:
    """Serialize a LineRecord to a plain dict for PaperContents transport."""
    return {
        "text": r.text,
        "page": r.page,
        "x0": r.x0,
        "y_top": r.y_top,
        "x1": r.x1,
        "y_bottom": r.y_bottom,
        "font_size": r.font_size,
    }


def records_from_dicts(ds: list[dict]) -> list[LineRecord]:
    """Reconstitute LineRecords from :func:`record_to_dict` output."""
    return [
        LineRecord(
            text=d["text"],
            page=d["page"],
            x0=d["x0"],
            y_top=d["y_top"],
            x1=d["x1"],
            y_bottom=d["y_bottom"],
            font_size=d["font_size"],
        )
        for d in ds
    ]
