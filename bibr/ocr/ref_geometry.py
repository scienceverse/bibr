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
        text = "".join(c for c, _ in buf).strip()
        all_boxes = [b for _, b in buf]
        glyph_boxes = [b for c, b in buf if not c.isspace()]
        if text and glyph_boxes:
            x0 = min(b[0] for b in all_boxes)
            x1 = max(b[2] for b in glyph_boxes)
            y_bottom = min(b[1] for b in glyph_boxes)
            y_top = max(b[3] for b in glyph_boxes)
            heights = sorted(b[3] - b[1] for b in glyph_boxes)
            fs = heights[len(heights) // 2] if heights else 0.0
            lines.append(LineRecord(text, page, x0, y_top, x1, y_bottom, fs))
        buf.clear()

    for ch, box in chars:
        if ch in ("\n", "\r"):
            flush()
        elif ch:
            buf.append((ch, box))
    flush()
    return lines


def _extract_page_chars(textpage) -> list[tuple[str, tuple[float, float, float, float]]]:
    out = []
    for i in range(textpage.count_chars()):
        ch = textpage.get_text_range(i, 1)
        if not ch:
            continue
        out.append((ch, textpage.get_charbox(i)))
    return out


def recover_reference_lines(pdf_bytes: bytes) -> list[LineRecord]:
    """Return ordered reference-section lines, or ``[]`` when there is no text
    layer or no References header."""
    import pypdfium2

    page_lines: dict[int, list[LineRecord]] = {}
    header_page: int | None = None
    with pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            for pi in range(len(doc)):
                page = doc[pi]
                try:
                    tp = page.get_textpage()
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
