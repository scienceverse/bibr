"""Native PDF text extraction for OCR bypass.

When a PDF has an embedded text layer (typical for modern academic PDFs),
extracting text directly via pypdfium2 is faster and more accurate than
sending a rendered page crop to GLM-OCR. This module exposes the primitive
that lets the OCR stage opt out of GLM-OCR on a per-region basis.

Coordinate systems:
    - PP-DocLayoutV3 bboxes: normalized (0..1000) image-space, y-down,
      origin top-left, relative to the rendered CropBox.
    - pypdfium2 text coords: PDF points (1/72 inch), origin bottom-left, y-up,
      in the page's native (MediaBox) space.
    The CropBox origin offsets the two systems. pypdfium2 renders the CropBox
    to the image the layout indexes into, but ``get_text_bounded`` queries the
    native page space. On PDFs whose CropBox origin is non-zero (common for
    publisher typesetting, e.g. Elsevier), that offset must be added back or
    every box is shifted and text columns get sliced.
"""

from __future__ import annotations

import ctypes
import logging
import math
import re
import unicodedata

from bibr.input.consolidate_text import fix_ocr_artifacts
from bibr.ocr.utils import pdfium_lock

logger = logging.getLogger(__name__)

# A private-use glyph in prose or references is especially dangerous: many
# publisher PDFs encode old-style digits in the PUA, so accepting the native
# string silently changes years and citation identity.  Keep this narrower
# than the general native-text usability gate; headings and other labels retain
# their existing behavior.
_CITATION_CRITICAL_NATIVE_LABELS = frozenset(
    {"text", "content", "vertical_text", "reference", "reference_content"}
)
_PRIVATE_USE_REJECTION_REASON = "private_use"

# Backward-compat alias — callers that imported _pdfium_lock from this module
# continue to work and share the same canonical lock object.
_pdfium_lock = pdfium_lock


def _page_crop_box(page) -> tuple[float, float, float, float]:
    """Return the page CropBox ``(x0, y0, x1, y1)`` in PDF points.

    This is the region pypdfium2 renders to the image PP-DocLayoutV3 indexes
    into: the CropBox clipped to the MediaBox, either inherited from the page
    tree. ``get_cropbox()`` reads the page dictionary alone and falls back to
    a US Letter MediaBox, so an A4 page whose MediaBox sits on the page tree
    put the byline's text under the title region.
    Falls back to ``(0, 0, width, height)`` when the box is missing or
    degenerate so callers always get a usable box.
    """
    try:
        x0, y0, x1, y1 = page.get_bbox()
        if x1 - x0 > 0 and y1 - y0 > 0:
            return (float(x0), float(y0), float(x1), float(y1))
    except Exception:  # noqa: BLE001, S110 - fall back to mediabox-origin page size
        pass
    w, h = page.get_size()
    return (0.0, 0.0, float(w), float(h))


def _page_rotation(page) -> int:
    """Page ``/Rotate`` in degrees clockwise, normalised to 0/90/180/270."""
    try:
        return int(page.get_rotation()) % 360
    except Exception:  # noqa: BLE001, S110 - treat an unreadable rotation as none
        return 0


def _unrotate_normalized_bbox(
    bbox_normalized: list[float], rotation: int
) -> tuple[float, float, float, float]:
    """Map a rendered-image bbox back into the unrotated CropBox frame.

    ``page.render()`` applies the page's ``/Rotate``, so on a 90°/270° page the
    image PP-DocLayoutV3 indexes is transposed while the CropBox and the char
    boxes stay in unrotated user space. Without this the bbox emitted for a
    text region queried no characters at all.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox_normalized[:4])
    if rotation == 90:
        corners = ((y1, 1000.0 - x1), (y2, 1000.0 - x2))
    elif rotation == 180:
        corners = ((1000.0 - x1, 1000.0 - y1), (1000.0 - x2, 1000.0 - y2))
    elif rotation == 270:
        corners = ((1000.0 - y1, x1), (1000.0 - y2, x2))
    else:
        return (x1, y1, x2, y2)
    us = [corner[0] for corner in corners]
    vs = [corner[1] for corner in corners]
    return (min(us), min(vs), max(us), max(vs))


def _normalized_bbox_to_pdf_points(
    bbox_normalized: list[float],
    crop_box: tuple[float, float, float, float],
    rotation: int = 0,
) -> tuple[float, float, float, float]:
    """Convert image-space (0..1000) bbox to PDF-point coords.

    ``crop_box`` is the page CropBox ``(x0, y0, x1, y1)`` in PDF points (see
    :func:`_page_crop_box`). The crop origin (``x0``/``y0``) is added back so
    the result lands in the page's native text-coordinate space; it is zero
    for the common case where MediaBox == CropBox == ``[0, 0, w, h]``.

    ``rotation`` is the page's ``/Rotate`` in degrees clockwise, which
    ``page.render()`` applies and the text layer does not.
    """
    cx0, cy0, cx1, cy1 = crop_box
    crop_w = cx1 - cx0
    crop_h = cy1 - cy0
    bx1, by1, bx2, by2 = _unrotate_normalized_bbox(bbox_normalized, rotation)
    left = cx0 + bx1 / 1000.0 * crop_w
    right = cx0 + bx2 / 1000.0 * crop_w
    top_pts = cy1 - (by1 / 1000.0 * crop_h)
    bottom_pts = cy1 - (by2 / 1000.0 * crop_h)
    return (left, bottom_pts, right, top_pts)


def _pdf_points_to_normalized_bbox(
    box_pts: tuple[float, float, float, float],
    crop_box: tuple[float, float, float, float],
    rotation: int = 0,
) -> tuple[float, float, float, float]:
    """Inverse of :func:`_normalized_bbox_to_pdf_points`.

    Maps a ``(left, bottom, right, top)`` box in PDF points (the text layer's
    frame) into the 0..1000 rendered-image space layout boxes use, as
    ``(x1, y1, x2, y2)``, applying the page's ``/Rotate`` the way
    ``page.render()`` does.
    """
    cx0, cy0, cx1, cy1 = crop_box
    crop_w = (cx1 - cx0) or 1.0
    crop_h = (cy1 - cy0) or 1.0
    left, bottom, right, top = box_pts
    u1 = (left - cx0) / crop_w * 1000.0
    u2 = (right - cx0) / crop_w * 1000.0
    v1 = (cy1 - top) / crop_h * 1000.0
    v2 = (cy1 - bottom) / crop_h * 1000.0
    if rotation == 90:
        corners = ((1000.0 - v1, u1), (1000.0 - v2, u2))
    elif rotation == 180:
        corners = ((1000.0 - u1, 1000.0 - v1), (1000.0 - u2, 1000.0 - v2))
    elif rotation == 270:
        corners = ((v1, 1000.0 - u1), (v2, 1000.0 - u2))
    else:
        return (min(u1, u2), min(v1, v2), max(u1, u2), max(v1, v2))
    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    return (min(xs), min(ys), max(xs), max(ys))


# Whitespace placement in _build_page_char_records. A whitespace box flatter
# than this fraction of the preceding glyph's height is flat (no ink).
_FLAT_WHITESPACE_HEIGHT_RATIO = 0.25
# The preceding glyph stands on the flat box's baseline when its own box
# starts at most this fraction of its height above it.
_BASELINE_TOLERANCE_RATIO = 0.1
# A flat box sits on a raised baseline (a superscript's) when it lies more
# than this fraction of the next glyph's height above that glyph's bottom,
# and no higher than its top (higher up, the next glyph opens the line
# below). A descender reaches about a third of its glyph's height below the
# line.
_RAISED_BASELINE_RATIO = 0.5
# A space lies inside the preceding ligature when its box starts within the
# ligature's box and ends at most this fraction of its own width past it...
_INNER_WHITESPACE_OVERSHOOT = 0.25
# ...and it adds no gap when the next glyph's advance box then starts less
# than this fraction of the space's width after the ligature's.
_INNER_WHITESPACE_MAX_GAP = 0.25
# Latin ligature presentation forms (ff, fi, fl, ffi, ffl, long st, st).
_LIGATURE_CHARS = frozenset("\ufb00\ufb01\ufb02\ufb03\ufb04\ufb05\ufb06")


def _loose_charbox(textpage, index: int) -> tuple[float, float, float, float] | None:
    try:
        return textpage.get_charbox(index, loose=True)
    except Exception:  # noqa: BLE001 - pdfium per-char failures are noisy and benign here
        return None


# Word-boundary repair. Two glyphs are on one line when their loose
# (font ascent/descent) boxes start and end within this fraction of the line
# height of each other: same font size, same baseline. Superscript markers and
# subscripts never qualify.
_SAME_LINE_TOLERANCE = 0.15
# Between two such glyphs, a gap of at least this fraction of the line height
# is a word gap when pdfium put nothing there...
_WORD_GAP_MIN = 0.12
# ...and it is this many times the in-word gaps next to it (a letter-spaced
# heading has no in-word gap smaller than its spacing, so it is never split).
_WORD_GAP_RATIO = 3.0
_WORD_GAP_FLOOR = 0.02
# Gaps outside this range (in line heights) are not neighbours on one line.
_LINE_GAP_RANGE = (-0.3, 1.5)


def _printable_glyph(ch: str) -> bool:
    return ch != "\ufffd" and unicodedata.category(ch)[0] in "LNPS"


def _same_line_gap(
    a: tuple[float, float, float, float] | None, b: tuple[float, float, float, float] | None
) -> float | None:
    """The gap from glyph *a* to glyph *b* in line heights, or None when not on one line."""
    if a is None or b is None:
        return None
    height = max(a[3] - a[1], b[3] - b[1])
    if height <= 0:
        return None
    tolerance = _SAME_LINE_TOLERANCE * height
    if abs(a[1] - b[1]) > tolerance or abs(a[3] - b[3]) > tolerance:
        return None
    gap = (b[0] - a[2]) / height
    return gap if _LINE_GAP_RANGE[0] <= gap <= _LINE_GAP_RANGE[1] else None


def _repair_word_boundaries(
    records: list[tuple[str, float, float, bool]],
    glyphs: list[tuple[int, str, tuple[float, float, float, float] | None]],
    generated_breaks: set[int],
) -> list[tuple[str, float, float, bool]]:
    """Fix pdfium's whitespace between consecutive glyphs of one line.

    pdfium decides spaces and line breaks per text object. A PDF that draws
    each glyph (or each punctuation mark) as its own object can come out with
    no space at a word gap ("arterialand") or with a generated line break in
    the middle of a line ("projects\\r\\n, are"). Between two glyphs of the
    same size on the same baseline, a generated break is dropped (replaced by
    a space when the gap is word-sized), and a missing space is inserted when
    the gap is at least ``_WORD_GAP_MIN`` of the line height and
    ``_WORD_GAP_RATIO`` times an in-word gap next to it. Printed whitespace
    and printed line breaks are never touched. *glyphs* holds the record
    index, char and loose box of every non-whitespace glyph, in order.
    """
    pairs: list[tuple[str | None, float]] = []
    for (ra, ca, la), (rb, cb, lb) in zip(glyphs, glyphs[1:], strict=False):
        gap = _same_line_gap(la, lb) if _printable_glyph(ca) and _printable_glyph(cb) else None
        kind = None
        if gap is not None:
            if rb == ra + 1:
                kind = "adjacent"
            elif all(index in generated_breaks for index in range(ra + 1, rb)):
                kind = "break"
        pairs.append((kind, gap if gap is not None else 0.0))
    drop: set[int] = set()
    insert: dict[int, tuple[float, float]] = {}
    for k, (kind, gap) in enumerate(pairs):
        if kind is None:
            continue
        (ra, _ca, la), (rb, _cb, lb) = glyphs[k], glyphs[k + 1]
        if la is None or lb is None:  # kind is only set for two boxed glyphs
            continue
        at = ((la[2] + lb[0]) / 2.0, (max(la[1], lb[1]) + min(la[3], lb[3])) / 2.0)
        if kind == "break":
            drop.update(range(ra + 1, rb))
            if gap >= _WORD_GAP_MIN:
                insert[ra] = at
        elif gap >= _WORD_GAP_MIN:
            in_word = [
                pairs[j][1]
                for j in (k - 1, k + 1)
                if 0 <= j < len(pairs) and pairs[j][0] == "adjacent" and pairs[j][1] < _WORD_GAP_MIN
            ]
            if in_word and gap >= _WORD_GAP_RATIO * max(max(in_word), _WORD_GAP_FLOOR):
                insert[ra] = at
    if not drop and not insert:
        return records
    repaired: list[tuple[str, float, float, bool]] = []
    for index, record in enumerate(records):
        if index in drop:
            continue
        repaired.append(record)
        if index in insert:
            x, y = insert[index]
            repaired.append((" ", x, y, False))
    return repaired


def _is_inner_whitespace(
    textpage,
    n_chars: int,
    ligature_index: int,
    ligature_box: tuple[float, float, float, float],
    space_box: tuple[float, float, float, float],
    next_index: int,
) -> bool:
    """True for a space drawn inside the ligature before it that adds no gap.

    Some fonts follow a ligature ("fi", "fl", "ff") with a space drawn inside
    the ligature's box. Inside a word the next glyph then starts where the
    ligature's advance ends ("Traffi cking" is one word); after a word-final
    ligature the same space is a real word space and pushes the next glyph a
    space-width further ("Sofi Oksanen"). Only the advance (loose) boxes tell
    the two apart. Plain glyphs are never passed here: an italic "f" hangs
    over the real space after it ("of pervasive").
    """
    import pypdfium2 as pdfium

    gl, _gb, gr, _gt = ligature_box
    sl, _sb, sr, _st = space_box
    width = sr - sl
    if not (width > 0 and gl <= sl and sr <= gr + _INNER_WHITESPACE_OVERSHOOT * width):
        return False
    if next_index >= n_chars:
        return False
    next_code = pdfium.raw.FPDFText_GetUnicode(textpage.raw, next_index)
    # Past U+10FFFF chr() raises; the page records keep U+FFFD there, a glyph.
    if next_code <= 0x10FFFF and chr(next_code).isspace():
        return False
    ligature_loose = _loose_charbox(textpage, ligature_index)
    next_loose = _loose_charbox(textpage, next_index)
    if ligature_loose is None or next_loose is None:
        return False
    return next_loose[0] - ligature_loose[2] < _INNER_WHITESPACE_MAX_GAP * width


def _build_page_char_records(textpage) -> list[tuple[str, float, float, bool]]:
    """Precompute ``(char, center_x, center_y, is_newline)`` for every char on the page.

    ``get_text_bounded`` assigns a glyph to a region whenever its char box merely
    INTERSECTS the query rect, so a region bbox edge that slices through an
    adjacent line bleeds in that neighbor's partial glyphs as garbage. Using the
    glyph's CENTER instead assigns each char to exactly one region.

    Whitespace is placed against the glyph before it on the same line:

    - A space's tight box is flat on the baseline, below the centre of the
      letters around it. A region whose bottom edge crosses the line between
      the baseline and the letters' centres kept the words and dropped the
      spaces between them ("MODEL ANALYSIS" became "MODELANALYSIS"), so a flat
      whitespace box is raised to the vertical centre of that glyph when the
      glyph stands on the same baseline. A raised glyph (closing quote,
      superscript) is skipped, since a region's top edge can cut it off its
      line, and the space is never lowered to a comma hanging below the
      baseline, which a region's bottom edge can cut off the same way. pdfium
      sets the space after a LaTeX superscript on the superscript's own
      baseline, which the baseline test cannot tell from the line's, so a
      flat box that sits more than half the next glyph's height above that
      glyph's bottom, and not above its top, is raised no higher than the
      next glyph's centre: a top edge that cuts the superscript off would
      take the space too.
    - A space drawn inside a ligature that adds no gap (some fonts follow
      "fi" with one) is dropped; keeping it split the word ("Traffi cking").
      See :func:`_is_inner_whitespace`. A ligature is a presentation-form
      char or a glyph whose box pdfium shares across the chars it maps to.

    Computing this once per page (rather than once per region) avoids
    O(regions × chars) pdfium calls in :func:`_fill_page_regions_from_textpage`,
    which queries one region at a time over the same page.
    """
    import pypdfium2 as pdfium

    def _charbox(index: int) -> tuple[float, float, float, float] | None:
        try:
            return textpage.get_charbox(index)
        except Exception:  # noqa: BLE001 - pdfium per-char failures are noisy and benign here
            return None

    n_chars = textpage.count_chars()
    records: list[tuple[str, float, float, bool]] = []
    # Index, box and ligature flag of the last non-whitespace glyph on the line.
    line_glyph: tuple[int, tuple[float, float, float, float], bool] | None = None
    # Raised spaces waiting for the glyph after them: (record index, the
    # space's own centre, the centre of the glyph before it).
    raised: list[tuple[int, float, float]] = []
    # Every non-whitespace glyph (record index, char, loose box) and the
    # record indexes of pdfium's generated line breaks, for the word-boundary
    # repair after the loop.
    glyphs: list[tuple[int, str, tuple[float, float, float, float] | None]] = []
    generated_breaks: set[int] = set()

    def _settle(next_box: tuple[float, float, float, float] | None) -> None:
        for index, own_y, glyph_y in raised:
            ch, x, _y, is_newline = records[index]
            limit = glyph_y
            if next_box is not None:
                _nl, nb, _nr, nt = next_box
                if nb + _RAISED_BASELINE_RATIO * (nt - nb) < own_y <= nt:
                    limit = min(glyph_y, (nb + nt) / 2.0)
            records[index] = (ch, x, max(own_y, limit), is_newline)
        raised.clear()

    i = 0
    while i < n_chars:
        code_unit = pdfium.raw.FPDFText_GetUnicode(textpage.raw, i)
        consumed = 1
        boxes = [_charbox(i)]

        # PDFium exposes non-BMP text as UTF-16 code units. Joining each unit
        # independently creates lone-surrogate Python strings that cannot be
        # UTF-8 encoded and crash Hugging Face tokenizers downstream.
        if 0xD800 <= code_unit <= 0xDBFF and i + 1 < n_chars:
            low = pdfium.raw.FPDFText_GetUnicode(textpage.raw, i + 1)
            if 0xDC00 <= low <= 0xDFFF:
                scalar = 0x10000 + ((code_unit - 0xD800) << 10) + (low - 0xDC00)
                ch = chr(scalar)
                consumed = 2
                boxes.append(_charbox(i + 1))
            else:
                ch = "\ufffd"
        elif 0xD800 <= code_unit <= 0xDFFF or code_unit > 0x10FFFF:
            ch = "\ufffd"
        else:
            ch = chr(code_unit)

        if ch in ("\n", "\r"):
            _settle(None)
            if pdfium.raw.FPDFText_IsGenerated(textpage.raw, i) == 1:
                generated_breaks.add(len(records))
            records.append((ch, 0.0, 0.0, True))
            line_glyph = None
            i += consumed
            continue

        valid_boxes = [box for box in boxes if box is not None]
        if not valid_boxes:
            i += consumed
            continue
        cl = min(box[0] for box in valid_boxes)
        cb = min(box[1] for box in valid_boxes)
        cr = max(box[2] for box in valid_boxes)
        ct = max(box[3] for box in valid_boxes)
        box = (cl, cb, cr, ct)
        center_y = (cb + ct) / 2.0
        if not ch.isspace():
            _settle(box)
            is_ligature = ch in _LIGATURE_CHARS or (line_glyph is not None and line_glyph[1] == box)
            line_glyph = (i, box, is_ligature)
        elif line_glyph is not None:
            glyph_index, glyph_box, glyph_is_ligature = line_glyph
            if glyph_is_ligature and _is_inner_whitespace(
                textpage, n_chars, glyph_index, glyph_box, box, i + consumed
            ):
                i += consumed
                continue
            gl, gb, _gr, gt = glyph_box
            if (
                gl <= cl
                and ct - cb < _FLAT_WHITESPACE_HEIGHT_RATIO * (gt - gb)
                and gb <= cb + _BASELINE_TOLERANCE_RATIO * (gt - gb)
            ):
                raised.append((len(records), center_y, (gb + gt) / 2.0))
        if not ch.isspace():
            glyphs.append((len(records), ch, _loose_charbox(textpage, i)))
        records.append((ch, (cl + cr) / 2.0, center_y, False))
        i += consumed
    _settle(None)
    return _repair_word_boundaries(records, glyphs, generated_breaks)


def _reconstruct_text_from_records(
    records: list[tuple[str, float, float, bool]],
    left: float,
    bottom: float,
    right: float,
    top: float,
) -> str:
    """Reconstruct region text from precomputed char records via center-containment.

    A glyph belongs to the region iff its center falls within
    ``[left, right] x [bottom, top]``. Orphan newlines belonging to excluded
    neighboring lines are swallowed: a line break is only emitted once content
    has already been emitted AND another in-region char follows it, joined as
    ``"\\r\\n"`` (matching ``get_text_bounded``'s line separator).
    """
    parts: list[str] = []
    pending_break = False
    for ch, cx, cy, is_newline in records:
        if is_newline:
            if parts:
                pending_break = True
            continue
        if left <= cx <= right and bottom <= cy <= top:
            if pending_break:
                parts.append("\r\n")
                pending_break = False
            parts.append(ch)
    return "".join(parts)


def _text_from_bbox_on_textpage(
    textpage,
    crop_box: tuple[float, float, float, float],
    bbox_normalized: list[float],
    *,
    rotation: int = 0,
    records: list[tuple[str, float, float, bool]] | None = None,
) -> str:
    """Extract text inside *bbox_normalized* from an already-open textpage.

    Parameters
    ----------
    textpage :
        An open pypdfium2 ``PdfTextPage`` object.
    crop_box : tuple[float, float, float, float]
        Page CropBox ``(x0, y0, x1, y1)`` in PDF points (see
        :func:`_page_crop_box`).
    bbox_normalized : list[float]
        Bounding box ``[x1, y1, x2, y2]`` in image-space normalised
        coordinates (range 0..1000), y-down, origin top-left.
    records : list[tuple[str, float, float, bool]] | None
        Precomputed per-page char records from :func:`_build_page_char_records`
        (char, center_x, center_y, is_newline). When callers process many
        regions on the same page (see :func:`_fill_page_regions_from_textpage`),
        passing shared records avoids re-walking every page char per region.
        When ``None`` (the single-shot public path via
        :func:`get_native_text_in_bbox`), records are built from *textpage*.

    Returns
    -------
    str
        Extracted text, or ``""`` if no characters fall inside the bbox.
    """
    left, bottom, right, top = _normalized_bbox_to_pdf_points(bbox_normalized, crop_box, rotation)
    if records is None:
        records = _build_page_char_records(textpage)
    return _reconstruct_text_from_records(records, left, bottom, right, top)


def get_native_text_in_bbox(
    pdf_bytes: bytes,
    page_idx: int,
    bbox_normalized: list[float],
) -> str:
    """Extract native PDF text whose characters fall inside a normalised bbox.

    Queries the PDF text layer directly via pypdfium2 without rendering the page
    to an image.  When a region is covered by native text this is faster and
    more accurate than sending a rendered crop to the VLM OCR engine.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file contents.
    page_idx : int
        0-indexed page number.
    bbox_normalized : list[float]
        Bounding box ``[x1, y1, x2, y2]`` in image-space normalised coordinates
        (range 0..1000), y-down, origin top-left.  This matches the coordinate
        system used by PP-DocLayoutV3 bounding-box outputs.

    Returns
    -------
    str
        Concatenated native text whose characters fall inside *bbox_normalized*
        in reading order.  Returns an empty string when: the PDF has no embedded
        text layer, *page_idx* is out of range, or no characters are found
        within the specified bbox.
    """
    import pypdfium2

    with _pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            if page_idx < 0 or page_idx >= len(doc):
                return ""
            page = doc[page_idx]
            try:
                crop_box = _page_crop_box(page)
                rotation = _page_rotation(page)
                textpage = open_text_page(page)
                try:
                    if textpage.count_chars() == 0:
                        return ""
                    return _text_from_bbox_on_textpage(
                        textpage, crop_box, bbox_normalized, rotation=rotation
                    )
                finally:
                    textpage.close()
            finally:
                page.close()
        finally:
            doc.close()


# Labels in LABEL_TREATMENT that receive text treatment but are intentionally
# excluded from DEFAULT_ELIGIBLE_LABELS:
#   - figure_title / table_title / chart_title: captions are often adjacent to
#     figures/tables with fuzzy bbox boundaries — OCR is safer.
#   - formula_number: short inline text (typically 1-3 chars) that is better
#     handled by OCR than by the min_chars threshold.
#   - header / footer: only eligible when OCR_NATIVE_TEXT_HEADER_FOOTER is on
#     (see HEADER_FOOTER_LABELS). Their only consumers (detected_headers /
#     detected_footers feeding DOI furniture candidates and front-matter
#     masthead checks) need plain text, which the native layer returns exactly
#     on born-digital PDFs — about half of all OCR calls there. The
#     printable-ratio corruption gate still applies.
DEFAULT_ELIGIBLE_LABELS: frozenset[str] = frozenset(
    {
        "text",
        "content",
        "vertical_text",
        "paragraph_title",
        "doc_title",
        "abstract",
        "reference",
        "reference_content",
        "footnote",
        "vision_footnote",
        "algorithm",
        "seal",
    }
)

# Header/footer labels, eligible only when the header/footer native-text
# setting is on (off by default, so default output is unchanged).
HEADER_FOOTER_LABELS: frozenset[str] = frozenset({"header", "footer"})


def resolve_eligible_labels(include_header_footer: bool) -> frozenset[str]:
    """Eligible native-text labels for the header/footer setting.

    Takes a plain bool (not Settings) to keep this module free of the
    config import cycle; callers pass
    ``settings.ocr.native_text_header_footer``.
    """
    if include_header_footer:
        return DEFAULT_ELIGIBLE_LABELS | HEADER_FOOTER_LABELS
    return DEFAULT_ELIGIBLE_LABELS


# Short running heads ("Cell Biology", "Benartzi et al.") are shorter than
# the body min_chars: "header"/"footer" share the short-text allowance, but
# only take effect when the header/footer setting makes them eligible —
# otherwise they are skipped as ineligible before this check.
_SHORT_NATIVE_TEXT_LABELS: frozenset[str] = frozenset(
    {"doc_title", "paragraph_title", "header", "footer"}
)
_SHORT_NATIVE_TEXT_MIN_CHARS = 3

_FONT_SAMPLE_SIZE = 15
_FONT_BBOX_TOLERANCE = 5.0  # PDF points

# PDF font-descriptor "Italic" flag (spec Table 121: bit position 7 → 1 << 6).
# pypdfium2's ``FPDFText_GetFontInfo`` returns these descriptor flags. Note that
# the standard-14 fonts (Helvetica-Oblique, Times-Italic, ...) carry no embedded
# descriptor, so pdfium leaves this bit clear for them — hence italic detection
# ALSO falls back to the font name (see ``_font_name_looks_italic``).
_FONT_FLAG_ITALIC = 0x40


def _char_font_info(textpage, idx: int) -> tuple[str, int]:
    """Return ``(font_name, descriptor_flags)`` for the character at *idx*.

    Wraps ``FPDFText_GetFontInfo``; returns ``("", 0)`` when the font name
    cannot be read.
    """
    import pypdfium2 as pdfium

    flags = ctypes.c_int(0)
    buf = ctypes.create_string_buffer(256)
    n = pdfium.raw.FPDFText_GetFontInfo(textpage.raw, idx, buf, 256, ctypes.byref(flags))
    if n <= 0:
        return "", flags.value
    name = buf.raw[:n].split(b"\x00", 1)[0].decode("utf-8", "replace")
    return name, flags.value


def _font_name_looks_italic(font_name: str) -> bool:
    """True when a font name signals an italic/oblique style.

    Covers the standard-14 fonts (Helvetica-Oblique, Times-Italic) and the
    common ``...-It`` / ``...ItalicMT`` embedded-font naming conventions where
    the descriptor Italic flag is unreliable.
    """
    lowered = font_name.lower()
    return "italic" in lowered or "oblique" in lowered


def _char_is_italic(textpage, idx: int) -> bool:
    """Whether the character at *idx* is rendered italic/oblique.

    Uses the descriptor Italic flag when present (embedded fonts) and falls
    back to the font name (standard-14 obliques leave the flag clear).
    """
    name, flags = _char_font_info(textpage, idx)
    return bool(flags & _FONT_FLAG_ITALIC) or _font_name_looks_italic(name)


def _sample_font_metadata_in_bbox(
    textpage,
    n_chars: int,
    left: float,
    bottom: float,
    right: float,
    top: float,
) -> tuple[float | None, int | None, bool | None]:
    """Sample character font metrics within a PDF-coordinate bbox.

    Returns ``(median_charbox_height, dominant_font_weight, is_italic)`` or
    ``(None, None, None)`` when no characters are found. ``is_italic`` is a
    majority vote over the sampled characters.
    """
    import pypdfium2 as pdfium

    tol = _FONT_BBOX_TOLERANCE
    start_idx = pdfium.raw.FPDFText_GetCharIndexAtPos(
        textpage.raw,
        (left + right) / 2,
        (top + bottom) / 2,
        (right - left) / 2 + tol,
        (top - bottom) / 2 + tol,
    )
    if start_idx < 0:
        return None, None, None

    char_heights: list[float] = []
    char_weights: list[int] = []
    char_italics: list[bool] = []

    scan_start = max(0, start_idx - 10)
    scan_end = min(n_chars, start_idx + 40)
    for i in range(scan_start, scan_end):
        ch = chr(pdfium.raw.FPDFText_GetUnicode(textpage.raw, i))
        if ch in ("\n", "\r", " ", "\t"):
            continue
        try:
            cl, cb, cr, ct = textpage.get_charbox(i)
        except Exception:  # noqa: S112, BLE001 - pdfium per-char failures are noisy and benign here
            continue
        if cl < left - tol or cr > right + tol or cb < bottom - tol or ct > top + tol:
            continue
        char_h = abs(ct - cb)
        if char_h > 0:
            char_heights.append(char_h)
        weight = pdfium.raw.FPDFText_GetFontWeight(textpage.raw, i)
        if weight > 0:
            char_weights.append(weight)
        char_italics.append(_char_is_italic(textpage, i))
        if len(char_heights) >= _FONT_SAMPLE_SIZE:
            break

    if not char_heights:
        return None, None, None

    from statistics import median

    med_h = median(char_heights)

    from collections import Counter

    dominant_weight: int | None = None
    if char_weights:
        dominant_weight = Counter(char_weights).most_common(1)[0][0]

    is_italic: bool | None = None
    if char_italics:
        # Majority vote: italic when at least half the sampled glyphs are italic.
        is_italic = sum(char_italics) * 2 >= len(char_italics)

    return med_h, dominant_weight, is_italic


def _attach_page_dimensions(page, regions: list[dict]) -> None:
    """Attach ``_page_w`` / ``_page_h`` (PDF points) to every region.

    Page geometry is a property of the page, not a character measurement, so it
    is available with or without a text layer and is attached before any
    char-sampling gate.
    """
    page_w, page_h = page.get_size()
    for region in regions:
        region["_page_w"] = round(float(page_w), 2)
        region["_page_h"] = round(float(page_h), 2)


def _attach_bbox_pdf_pts(
    crop_box: tuple[float, float, float, float],
    regions: list[dict],
    rotation: int = 0,
) -> None:
    """Attach ``_bbox_pdf_pts`` (PDF points) to every region.

    Converts each region's ``bbox_2d`` from PP-DocLayoutV3 image space
    (0..1000, y-down, top-left origin, relative to the rendered CropBox) into
    PDF points via :func:`_normalized_bbox_to_pdf_points`. The result lives in
    the page's native text-coordinate space and follows that convention:
    **PDF points (1/72 inch), bottom-left origin, y-up**, as ``(x1, y1, x2,
    y2)`` with ``x1 <= x2`` and ``y1 <= y2``. This matches the ``_page_w`` /
    ``_page_h`` frame (see :func:`_attach_page_dimensions`), so region geometry
    can be checked for page containment against them.

    This is a SEPARATE field from ``bbox_2d``: internal consumers (region
    cropping, containment dedup, caption matching) stay tuned to the 0..1000
    image space, so ``bbox_2d`` is left untouched. Regions without a
    ``bbox_2d`` get ``None``. Like page geometry, this is a coordinate
    transform of the layout bbox, so it is attached with or without a text
    layer (scanned pages that fall back to OCR still carry it).
    """
    for region in regions:
        bbox = region.get("bbox_2d")
        if not bbox:
            region["_bbox_pdf_pts"] = None
            continue
        left, bottom, right, top = _normalized_bbox_to_pdf_points(bbox, crop_box, rotation)
        # Crop-relative, matching the ``_page_w`` / ``_page_h`` frame this
        # docstring promises: ``page.get_size()`` returns the CropBox extent,
        # so leaving the crop origin in broke the ``0 <= x1 <= x2 <= page_w``
        # invariant on any page whose CropBox does not start at (0, 0).
        cx0, cy0, _cx1, _cy1 = crop_box
        region["_bbox_pdf_pts"] = [
            round(left - cx0, 2),
            round(bottom - cy0, 2),
            round(right - cx0, 2),
            round(top - cy0, 2),
        ]


def _sample_page_font_metadata(
    textpage,
    crop_box: tuple[float, float, float, float],
    n_chars: int,
    regions: list[dict],
    rotation: int = 0,
) -> None:
    """Attach per-region font metadata sampled from an already-open textpage.

    Sets ``_font_size`` / ``_font_weight`` / ``_font_bold`` / ``_is_italic`` on
    regions with a ``bbox_2d`` and at least one measurable character. Operates
    on a caller-provided (lock-held) textpage; does not open or lock anything.
    """
    for region in regions:
        bbox = region.get("bbox_2d")
        if not bbox:
            continue
        left, bottom, right, top = _normalized_bbox_to_pdf_points(bbox, crop_box, rotation)
        med_h, weight, is_italic = _sample_font_metadata_in_bbox(
            textpage, n_chars, left, bottom, right, top
        )
        if med_h is not None:
            region["_font_size"] = round(med_h, 2)
        if weight is not None:
            region["_font_weight"] = weight
            region["_font_bold"] = weight >= 700
        if is_italic is not None:
            region["_is_italic"] = is_italic


def _fill_page_regions_from_textpage(
    textpage,
    crop_box: tuple[float, float, float, float],
    regions: list[dict],
    *,
    min_chars: int,
    eligible_labels: frozenset[str],
    min_printable_ratio: float,
    page_idx: int,
    rotation: int = 0,
) -> None:
    """Pre-fill eligible regions' ``content`` from an already-open textpage.

    Operates on a caller-provided (lock-held) textpage; does not open or lock
    anything. See :func:`fill_regions_from_native_text` for eligibility rules.

    Per-page char records (center coordinates for center-containment) are
    computed ONCE and reused across every region on the page, avoiding
    O(regions × chars) pdfium calls.
    """
    records = _build_page_char_records(textpage)
    for region in regions:
        label = region.get("label")
        if label not in eligible_labels:
            continue
        bbox = region.get("bbox_2d")
        if not bbox:
            continue
        native_text = _text_from_bbox_on_textpage(
            textpage, crop_box, bbox, rotation=rotation, records=records
        )
        if not native_text:
            continue
        # Some born-digital PDFs expose a line-end hyphen as STX (U+0002),
        # the same marker GLM-OCR uses.  It is not a corrupt CMap byte: the
        # artifact normalizer can distinguish a wrapped word (``con\x02ducted``)
        # from a literal compound (``self\x02proclaimed``).  Resolve it before
        # the control-character corruption gate, otherwise one benign marker
        # sends an entire, otherwise clean region through the vision model.
        if "\x02" in native_text:
            native_text = fix_ocr_artifacts(native_text)
        if label in _CITATION_CRITICAL_NATIVE_LABELS and any(
            unicodedata.category(ch) == "Co" for ch in native_text
        ):
            region["_native_text_candidate"] = native_text
            region["_native_text_rejection_reason"] = _PRIVATE_USE_REJECTION_REASON
            logger.debug(
                "Native text contained a private-use character (page %d), "
                "falling back to OCR for this region",
                page_idx,
            )
            continue
        effective_min_chars = min_chars
        if label in _SHORT_NATIVE_TEXT_LABELS:
            effective_min_chars = min(min_chars, _SHORT_NATIVE_TEXT_MIN_CHARS)
        if len(native_text.strip()) < effective_min_chars:
            continue
        if not _is_native_text_usable(native_text, min_printable_ratio):
            logger.debug(
                "Native text failed corruption gate (page %d), falling back to OCR for this region",
                page_idx,
            )
            continue
        region.pop("_native_text_candidate", None)
        region.pop("_native_text_rejection_reason", None)
        region["content"] = native_text
        region["_native_text_used"] = True


def fill_font_metadata(
    pdf_bytes: bytes,
    pages_regions: list[list[dict]],
) -> list[list[dict]]:
    """Extract font metadata for layout-detected regions.

    Samples characters within each region's bbox via pypdfium2 and attaches:

    - ``_font_size``: median charbox height in PDF points (reliable even when
      the PDF encodes font_size as 1.0 with transform matrices).
    - ``_font_weight``: dominant font weight (400 = regular, 700+ = bold).
    - ``_font_bold``: ``True`` when weight >= 700.
    - ``_is_italic``: majority-vote italic/oblique flag (descriptor Italic bit,
      falling back to the font name for standard-14 obliques).
    - ``_page_w`` / ``_page_h``: page dimensions in PDF points. These are a
      page property, not a character measurement, so they are attached to every
      region — including regions with no measurable text and pages with no text
      layer at all (which fall back to OCR but still carry page geometry).

    Font/italic keys require a ``bbox_2d`` and at least one measurable
    character; page dimensions do not.

    Mutates *pages_regions* in place and returns the same list.
    """
    import pypdfium2

    with _pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            for page_idx, regions in enumerate(pages_regions):
                if page_idx >= len(doc):
                    break
                page = doc[page_idx]
                try:
                    # Page geometry is available with or without a text layer;
                    # attach it to every region before any char-sampling gate.
                    _attach_page_dimensions(page, regions)
                    crop_box = _page_crop_box(page)
                    rotation = _page_rotation(page)
                    _attach_bbox_pdf_pts(crop_box, regions, rotation)
                    textpage = open_text_page(page)
                    try:
                        n_chars = textpage.count_chars()
                        if n_chars == 0:
                            continue
                        _sample_page_font_metadata(textpage, crop_box, n_chars, regions, rotation)
                    finally:
                        textpage.close()
                finally:
                    page.close()
        finally:
            doc.close()
    return pages_regions


# --- Corruption gate ---------------------------------------------------
#
# Some PDFs carry a broken CID-to-Unicode map: pypdfium2 still returns
# "text" for such pages, but it is mojibake/garbage rather than the
# printed content. Trusting it verbatim is worse than falling back to
# GLM-OCR. The checks below are deliberately Unicode-wide (not ASCII-only)
# because scientific papers routinely contain diacritics (Müller, café),
# Greek letters (α, β, Ω — used as variables/coefficients), math symbols
# (±, ∑, ∫, χ²), and typographic dashes/quotes ('-', em dash, curly quotes).

# Unicode general-category first letters treated as "reasonable" content:
#   L* - letters (any script/case: Latin, Greek, Cyrillic, CJK, ...)
#   M* - combining marks (precomposed diacritics decompose into these)
#   N* - digits/numerals (any script)
#   P* - punctuation (hyphens, dashes, quotes, brackets, ...)
#   Z* - separators (spaces, line/paragraph separators)
# "Sm" (math symbol, e.g. ±, ∑, ∫, <, =) is included explicitly so formula-
# heavy prose isn't penalized. Deliberately EXCLUDED: "Cc" (control chars,
# except the common whitespace controls below), "Cf"/"Co"/"Cs"/"Cn"
# (format/private-use/surrogate/unassigned — exactly where broken font
# maps tend to dump undecodable glyphs), and "So"/"Sk" (misc symbols,
# which is also where U+FFFD REPLACEMENT CHARACTER lives).
_REASONABLE_CATEGORY_PREFIXES = ("L", "M", "N", "P", "Z")
_REASONABLE_EXTRA_CATEGORIES = frozenset({"Sm"})
_REASONABLE_WHITESPACE_CONTROLS = frozenset({"\t", "\n", "\r", "\f", "\v"})

# A run of "(cid:123)" tokens is an unambiguous artifact of pypdfium2 falling
# back to raw glyph IDs because the font's CID-to-Unicode map is broken or
# missing. It never occurs in legitimately extracted text, regardless of
# printable-ratio (the tokens themselves are plain ASCII).
_CID_ARTIFACT_RE = re.compile(r"\(cid:\d+\)")
_CID_ARTIFACT_MIN_COUNT = 2

# A high density of U+FFFD (used by decoders to stand in for bytes/glyphs
# they couldn't map) is a second, independent corruption signal.
_FFFD_MAX_DENSITY = 0.05


def _has_disallowed_control_char(text: str) -> bool:
    """Return True if *text* contains any Unicode category-Cc char other than
    the common whitespace controls (tab/newline/CR/FF/VT).

    Broken ToUnicode CMaps in LaTeX math fonts sometimes leak raw CIDs as C0
    control chars (e.g. 0x0F). A single such glyph never trips the overall
    printable-ratio gate in an otherwise clean paragraph, but genuine text
    never contains raw control bytes, so any occurrence is disqualifying.
    Known STX soft-hyphen markers are resolved before this function is called.
    """
    return any(
        ch not in _REASONABLE_WHITESPACE_CONTROLS and unicodedata.category(ch) == "Cc"
        for ch in text
    )


def _printable_ratio(text: str) -> float:
    """Return the fraction of *text* made up of "reasonable" characters.

    Empty text returns ``1.0`` (vacuously fine — callers gate on length
    separately via ``min_chars``).
    """
    if not text:
        return 1.0
    reasonable = 0
    for ch in text:
        if ch in _REASONABLE_WHITESPACE_CONTROLS:
            reasonable += 1
            continue
        category = unicodedata.category(ch)
        if category[0] in _REASONABLE_CATEGORY_PREFIXES or category in _REASONABLE_EXTRA_CATEGORIES:
            reasonable += 1
    return reasonable / len(text)


def _is_native_text_usable(text: str, min_printable_ratio: float) -> bool:
    """Decide whether *text* looks like a genuine extraction vs. mojibake.

    Rejects text that:
      - contains repeated ``(cid:N)`` artifact tokens (broken glyph map), or
      - has a high density of U+FFFD replacement characters, or
      - contains any C0 control char other than tab/newline/CR/FF/VT (a
        broken font CMap leaking a raw CID), or
      - falls below *min_printable_ratio* fraction of "reasonable" chars
        (letters/digits/punctuation/whitespace/math symbols, any script).
    """
    if len(_CID_ARTIFACT_RE.findall(text)) >= _CID_ARTIFACT_MIN_COUNT:
        return False
    if text and (text.count("�") / len(text)) > _FFFD_MAX_DENSITY:
        return False
    if _has_disallowed_control_char(text):
        return False
    return _printable_ratio(text) >= min_printable_ratio


# A scanned page run through an OCR engine (Acrobat Paper Capture, ABBYY,
# Tesseract's PDF renderer) keeps the scan as a page-sized image and lays the
# recognised text over it in an invisible text render mode, so the page is
# searchable. That layer is usually printable, so the corruption gate above
# accepts it, but it is the legacy engine's reading of the scan: misread titles
# and merged reference lines. Such a page counts as a scan and is read by OCR.
# Only the render mode is checked, not paint order: visible text painted under
# an opaque image is hidden too, but PDFium cannot tell whether an image has a
# soft mask, so a watermark over real text would look the same.
# On the dev PDFs (2026-09-27), every scanned page of the targets is >= 0.93
# covered by images with >= 0.96 of its text invisible; no born-digital page
# has more than 2% of its text invisible.
_INVISIBLE_TEXT_RENDER_MODES = frozenset({3, 7})  # FPDF_TEXTRENDERMODE_INVISIBLE, _CLIP
_SCAN_PAGE_MIN_IMAGE_COVERAGE = 0.85
_SCAN_PAGE_MIN_INVISIBLE_SHARE = 0.5
# Coverage is the union of the largest image boxes, so a cap can only
# under-count it (and keep a page on the text layer).
_SCAN_PAGE_MAX_IMAGE_BOXES = 64
_SCAN_PAGE_MAX_FORM_DEPTH = 8
_UNCOUNTED_CHAR_CODES = frozenset({0x00, 0x02, 0xFFFE})


def _object_bounds(pdfium_c, obj) -> tuple[float, float, float, float] | None:
    left, bottom, right, top = (ctypes.c_float() for _ in range(4))
    if not pdfium_c.FPDFPageObj_GetBounds(
        obj, ctypes.byref(left), ctypes.byref(bottom), ctypes.byref(right), ctypes.byref(top)
    ):
        return None
    return (left.value, bottom.value, right.value, top.value)


def _compose(
    inner: tuple[float, ...], outer: tuple[float, ...] | None
) -> tuple[float, float, float, float, float, float]:
    """The affine matrix applying *inner*, then *outer* (``None`` = identity)."""
    a1, b1, c1, d1, e1, f1 = inner
    if outer is None:
        return (a1, b1, c1, d1, e1, f1)
    a2, b2, c2, d2, e2, f2 = outer
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )


def _transform_box(
    matrix: tuple[float, ...], box: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    a, b, c, d, e, f = matrix
    left, bottom, right, top = box
    xs, ys = [], []
    for x, y in ((left, bottom), (right, bottom), (left, top), (right, top)):
        xs.append(a * x + c * y + e)
        ys.append(b * x + d * y + f)
    return (min(xs), min(ys), max(xs), max(ys))


def _collect_image_boxes(pdfium_c, objects, matrix, depth: int, boxes: list) -> None:
    """Append the page-space box of every image object, descending into forms.

    PDFium reports an object inside a Form XObject in the form's own space, so
    each level's box goes through the matrices of the forms that contain it.
    """
    for obj in objects:
        kind = pdfium_c.FPDFPageObj_GetType(obj)
        if kind == pdfium_c.FPDF_PAGEOBJ_IMAGE:
            box = _object_bounds(pdfium_c, obj)
            if box is not None:
                boxes.append(box if matrix is None else _transform_box(matrix, box))
        elif kind == pdfium_c.FPDF_PAGEOBJ_FORM and depth < _SCAN_PAGE_MAX_FORM_DEPTH:
            form_matrix = pdfium_c.FS_MATRIX()
            if not pdfium_c.FPDFPageObj_GetMatrix(obj, ctypes.byref(form_matrix)):
                continue
            inner = _compose(
                (
                    form_matrix.a,
                    form_matrix.b,
                    form_matrix.c,
                    form_matrix.d,
                    form_matrix.e,
                    form_matrix.f,
                ),
                matrix,
            )
            children = (
                pdfium_c.FPDFFormObj_GetObject(obj, index)
                for index in range(pdfium_c.FPDFFormObj_CountObjects(obj))
            )
            _collect_image_boxes(pdfium_c, children, inner, depth + 1, boxes)


# --- Diagonal watermarks ------------------------------------------------
#
# Review copies and stamped PDFs draw a large diagonal string over every page
# ("For Review Only", a review disclaimer, "RETRACTED", "ARTICLE IN PRESS").
# PP-DocLayoutV3 has no watermark class, so its glyphs land in whatever region
# box their centres fall in: a few stray letters at the start (drawn first) or
# end (drawn last) of abstracts, headings and statements, and skewed region font
# sizes. Drawn first, the object also skews pdfium's line breaking, which then
# breaks the line before every one-glyph text object ("Buyer -Supplier").
# Dropping the chars afterwards cannot undo that, so such text objects are
# removed from the in-memory page before its text page is built.
#
# A watermark is a text object whose baseline, composed through the forms that
# hold it, is more than _WATERMARK_MIN_SKEW_DEG off a multiple of 90 degrees,
# at an effective size (Tf size times the matrix scale) of at least
# _WATERMARK_MIN_SIZE_PT. Colour, alpha and /Artifact tags do not separate
# them (red, grey or light blue; opaque or not; mostly untagged). On the dev
# PDFs (2026-10-02) watermarks are 24-100 pt and the only other off-axis text,
# rotated chart tick labels, is 5-9 pt; 90/270 degree text (side stamps,
# landscape tables) is on-axis and kept. FPDFText_GetCharAngle is not used:
# it reports the shear of synthetic italics as rotation.
_WATERMARK_MIN_SKEW_DEG = 3.0
_WATERMARK_MIN_SIZE_PT = 16.0


def _object_matrix(pdfium_c, obj) -> tuple[float, float, float, float, float, float] | None:
    matrix = pdfium_c.FS_MATRIX()
    if not pdfium_c.FPDFPageObj_GetMatrix(obj, ctypes.byref(matrix)):
        return None
    return (matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f)


def _is_watermark_text(pdfium_c, obj, matrix: tuple[float, ...]) -> bool:
    """True for a large text object set on an off-axis baseline (see above)."""
    a, b, c, d, _e, _f = matrix
    skew = math.degrees(math.atan2(b, a)) % 90.0
    if min(skew, 90.0 - skew) <= _WATERMARK_MIN_SKEW_DEG:
        return False
    size = ctypes.c_float(0.0)
    if not pdfium_c.FPDFTextObj_GetFontSize(obj, ctypes.byref(size)):
        return False
    return size.value * math.sqrt(abs(a * d - b * c)) >= _WATERMARK_MIN_SIZE_PT


def _collect_watermark_objects(pdfium_c, objects, parent, matrix, depth: int, found: list) -> None:
    """Append ``(parent form or None, text object)`` for every watermark text object.

    Like :func:`_collect_image_boxes`, an object inside a Form XObject is
    judged in page space, through the matrices of the forms that contain it:
    a stamp can be upright text inside a rotated form.
    """
    for obj in objects:
        kind = pdfium_c.FPDFPageObj_GetType(obj)
        if kind == pdfium_c.FPDF_PAGEOBJ_TEXT:
            own = _object_matrix(pdfium_c, obj)
            if own is not None and _is_watermark_text(pdfium_c, obj, _compose(own, matrix)):
                found.append((parent, obj))
        elif kind == pdfium_c.FPDF_PAGEOBJ_FORM and depth < _SCAN_PAGE_MAX_FORM_DEPTH:
            own = _object_matrix(pdfium_c, obj)
            if own is None:
                continue
            children = [
                pdfium_c.FPDFFormObj_GetObject(obj, index)
                for index in range(pdfium_c.FPDFFormObj_CountObjects(obj))
            ]
            _collect_watermark_objects(
                pdfium_c, children, obj, _compose(own, matrix), depth + 1, found
            )


def _text_object_text(pdfium_c, obj, textpage) -> str:
    length = pdfium_c.FPDFTextObj_GetText(obj, textpage.raw, None, 0)
    if length <= 2:
        return ""
    buffer = ctypes.create_string_buffer(length)
    pdfium_c.FPDFTextObj_GetText(
        obj, textpage.raw, ctypes.cast(buffer, ctypes.POINTER(pdfium_c.FPDF_WCHAR)), length
    )
    return buffer.raw[: length - 2].decode("utf-16-le", "replace")


def strip_watermark_objects(page) -> list[str]:
    """Remove diagonal watermark text objects from *page* in memory.

    Returns the removed strings (whitespace-normalised, one per object that
    had text). Only the loaded page changes: nothing is written back, and the
    callers open the document from bytes for their own pass. Operates on a
    caller-provided (lock-held) page.
    """
    import pypdfium2.raw as pdfium_c

    found: list = []
    top_level = [
        pdfium_c.FPDFPage_GetObject(page.raw, index)
        for index in range(pdfium_c.FPDFPage_CountObjects(page.raw))
    ]
    _collect_watermark_objects(pdfium_c, top_level, None, None, 0, found)
    if not found:
        return []
    texts: list[str] = []
    # The strings need a text page of the unstripped page; only pages that
    # carry a watermark pay for it.
    textpage = page.get_textpage()
    try:
        for _parent, obj in found:
            text = " ".join(_text_object_text(pdfium_c, obj, textpage).split())
            if text:
                texts.append(text)
    finally:
        textpage.close()
    remove_from_form = getattr(pdfium_c, "FPDFFormObj_RemoveObject", None)
    for parent, obj in found:
        if parent is None:
            removed = pdfium_c.FPDFPage_RemoveObject(page.raw, obj)
        elif remove_from_form is not None:
            removed = remove_from_form(parent, obj)
        else:
            removed = False
        if removed:
            # Removal hands the object to the caller.
            pdfium_c.FPDFPageObj_Destroy(obj)
    return texts


def open_text_page(page, watermarks: list[str] | None = None):
    """Build *page*'s pdfium text page without its diagonal watermark text.

    Every native-text reader goes through this, so region text, font
    metadata, reference geometry and DOI evidence see the same characters.
    The removed strings are appended to *watermarks* when it is given.
    """
    removed = strip_watermark_objects(page)
    if watermarks is not None:
        watermarks.extend(removed)
    return page.get_textpage()


def _union_area(boxes: list[tuple[float, float, float, float]]) -> float:
    """Exact area of the union of axis-aligned boxes, by vertical slabs."""
    xs = sorted({x for left, _, right, _ in boxes for x in (left, right)})
    total = 0.0
    for x0, x1 in zip(xs, xs[1:], strict=False):
        spans = sorted(
            (bottom, top) for left, bottom, right, top in boxes if left <= x0 and right >= x1
        )
        covered = 0.0
        run_bottom = run_top = None
        for bottom, top in spans:
            if run_top is None or bottom > run_top:
                if run_top is not None:
                    covered += run_top - run_bottom
                run_bottom, run_top = bottom, top
            elif top > run_top:
                run_top = top
        if run_top is not None:
            covered += run_top - run_bottom
        total += covered * (x1 - x0)
    return total


def _page_image_coverage(page, crop_box: tuple[float, float, float, float]) -> float:
    """Fraction of the CropBox covered by the page's image objects (0..1)."""
    import pypdfium2.raw as pdfium_c

    cx0, cy0, cx1, cy1 = crop_box
    area = (cx1 - cx0) * (cy1 - cy0)
    if area <= 0:
        return 0.0
    boxes: list[tuple[float, float, float, float]] = []
    top_level = (
        pdfium_c.FPDFPage_GetObject(page.raw, index)
        for index in range(pdfium_c.FPDFPage_CountObjects(page.raw))
    )
    _collect_image_boxes(pdfium_c, top_level, None, 0, boxes)
    clipped = []
    for left, bottom, right, top in boxes:
        left, bottom = max(left, cx0), max(bottom, cy0)
        right, top = min(right, cx1), min(top, cy1)
        if right > left and top > bottom:
            clipped.append((left, bottom, right, top))
    if not clipped:
        return 0.0
    clipped.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    return min(1.0, _union_area(clipped[:_SCAN_PAGE_MAX_IMAGE_BOXES]) / area)


def _invisible_text_share(textpage) -> float | None:
    """Share of the page's characters drawn in an invisible text render mode.

    Whitespace and the characters PDFium generates (inferred spaces and line
    breaks) are not counted. ``None`` when nothing is countable. Raises when
    this PDFium build cannot map a character to its text object, so the caller
    keeps the text layer and reports the rule as unavailable.
    """
    import pypdfium2.raw as pdfium_c

    text_object_of = getattr(pdfium_c, "FPDFText_GetTextObject", None)
    if text_object_of is None:
        raise RuntimeError("this PDFium build has no FPDFText_GetTextObject")
    is_generated = getattr(pdfium_c, "FPDFText_IsGenerated", None)
    handle = textpage.raw
    counted = invisible = 0
    for index in range(textpage.count_chars()):
        code = pdfium_c.FPDFText_GetUnicode(handle, index)
        if code in _UNCOUNTED_CHAR_CODES or (code <= 0x10FFFF and chr(code).isspace()):
            continue
        if is_generated is not None and is_generated(handle, index) == 1:
            continue
        obj = text_object_of(handle, index)
        if not obj:
            continue
        counted += 1
        if pdfium_c.FPDFTextObj_GetTextRenderMode(obj) in _INVISIBLE_TEXT_RENDER_MODES:
            invisible += 1
    return invisible / counted if counted else None


def _is_invisible_text_layer_page(
    page, textpage, crop_box: tuple[float, float, float, float]
) -> bool:
    """True when the page is a scan whose text layer is a hidden OCR layer.

    Images cover at least :data:`_SCAN_PAGE_MIN_IMAGE_COVERAGE` of the CropBox
    and at least :data:`_SCAN_PAGE_MIN_INVISIBLE_SHARE` of the characters are
    invisible (render mode 3 or 7). A born-digital page with a background image
    keeps its visible text, and a figure-sized image never qualifies. Operates
    on a caller-provided (lock-held) page and textpage.
    """
    if _page_image_coverage(page, crop_box) < _SCAN_PAGE_MIN_IMAGE_COVERAGE:
        return False
    share = _invisible_text_share(textpage)
    return share is not None and share >= _SCAN_PAGE_MIN_INVISIBLE_SHARE


def fill_regions_from_native_text(
    pdf_bytes: bytes,
    pages_regions: list[list[dict]],
    *,
    min_chars: int,
    eligible_labels: frozenset[str] = DEFAULT_ELIGIBLE_LABELS,
    min_printable_ratio: float = 0.85,
) -> list[list[dict]]:
    """Pre-fill region ``content`` from the PDF's native text layer where possible.

    Mutates regions in-place and also returns the same list for chaining.

    A region is eligible iff:
      - ``region["label"]`` ∈ *eligible_labels*,
      - the PDF has a text layer covering this region with
        ``len(native_text.strip()) >= min_chars`` (except short heading-like
        labels such as ``doc_title`` / ``paragraph_title``, which use a lower
        native-text threshold so headings like ``Method`` do not fall through
        to OCR), and
      - the extracted text passes the corruption gate (see
        :func:`_is_native_text_usable`) — this rejects pages with a broken
        CID-to-Unicode map (mojibake, ``(cid:N)`` artifacts, U+FFFD-heavy
        text) so they fall back to OCR instead of ingesting garbage.

    Filled regions gain ``region["_native_text_used"] = True`` and have their
    ``content`` set to the extracted native text (unstripped). The downstream
    OCR stage detects this flag and skips the GLM-OCR call for such regions.

    The document is opened once and all pages are processed under a single
    acquisition of :data:`pdfium_lock`, avoiding the overhead of 300+ open/
    close cycles for a typical 15-page paper with ~20 regions per page.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file bytes.
    pages_regions : list[list[dict]]
        Per-page list of region dicts as produced by the layout detector.
        Each region is expected to carry ``label`` and ``bbox_2d`` keys.
    min_chars : int
        Minimum stripped character count for a region to be pre-filled.
    eligible_labels : frozenset[str]
        Labels eligible for native-text pre-fill. Defaults to text-flavored
        labels — tables, formulas, and figures always go to OCR.
    min_printable_ratio : float
        Minimum fraction of "reasonable" characters (letters/digits/
        punctuation/whitespace/math symbols, any script) required for the
        extracted text to be trusted. See :data:`bibr.config.OcrOptions.
        native_text_min_printable_ratio`.

    Returns
    -------
    list[list[dict]]
        The same *pages_regions* list, mutated in place.
    """
    import pypdfium2

    with _pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            for page_idx, regions in enumerate(pages_regions):
                if page_idx >= len(doc):
                    break
                page = doc[page_idx]
                try:
                    crop_box = _page_crop_box(page)
                    rotation = _page_rotation(page)
                    _attach_bbox_pdf_pts(crop_box, regions, rotation)
                    textpage = open_text_page(page)
                    try:
                        if textpage.count_chars() == 0:
                            continue
                        _fill_page_regions_from_textpage(
                            textpage,
                            crop_box,
                            regions,
                            min_chars=min_chars,
                            eligible_labels=eligible_labels,
                            min_printable_ratio=min_printable_ratio,
                            page_idx=page_idx,
                            rotation=rotation,
                        )
                    finally:
                        textpage.close()
                finally:
                    page.close()
        finally:
            doc.close()
    return pages_regions


def fill_native_text_and_fonts(
    pdf_bytes: bytes,
    pages_regions: list[list[dict]],
    *,
    min_chars: int,
    eligible_labels: frozenset[str] = DEFAULT_ELIGIBLE_LABELS,
    min_printable_ratio: float = 0.85,
) -> list[list[dict]]:
    """Native-text fill and font-metadata sampling from a single PDF open.

    Passes 1 (:func:`fill_regions_from_native_text`) and 2
    (:func:`fill_font_metadata`) each opened the document and derived a textpage
    per page over the *same* pages. This combined pass opens the document once
    and derives one textpage per page that serves BOTH — halving the pdfium
    open/textpage work for native PDFs — under a single acquisition of
    :data:`pdfium_lock`.

    Output is identical to calling the two functions in sequence, with the
    per-pass fallback contract preserved: the font-metadata pass is best-effort
    *per page* — a failure there is logged and skipped so it never discards the
    native-text fill already applied on that page. A failure in the native-text
    fill propagates (as in :func:`fill_regions_from_native_text`) so the caller
    can fall back to full OCR.

    Mutates *pages_regions* in place and returns the same list.
    """
    import pypdfium2

    with _pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            for page_idx, regions in enumerate(pages_regions):
                if page_idx >= len(doc):
                    break
                page = doc[page_idx]
                try:
                    crop_box = _page_crop_box(page)
                    rotation = _page_rotation(page)
                    # Page geometry (font pass) and the PDF-point bbox attach to
                    # every region with or without a text layer — set them before
                    # the char-count gate.
                    _attach_page_dimensions(page, regions)
                    _attach_bbox_pdf_pts(crop_box, regions, rotation)
                    textpage = open_text_page(page)
                    try:
                        n_chars = textpage.count_chars()
                        if n_chars == 0:
                            continue
                        # Pass 1: native-text fill (propagates on failure).
                        _fill_page_regions_from_textpage(
                            textpage,
                            crop_box,
                            regions,
                            min_chars=min_chars,
                            eligible_labels=eligible_labels,
                            min_printable_ratio=min_printable_ratio,
                            page_idx=page_idx,
                            rotation=rotation,
                        )
                        # Pass 2: font metadata — best-effort. A failure here
                        # must NOT discard the native-text fill above.
                        try:
                            _sample_page_font_metadata(
                                textpage, crop_box, n_chars, regions, rotation
                            )
                        except Exception:  # noqa: BLE001
                            logger.warning(
                                "Font metadata sampling failed on page %d; "
                                "keeping native-text fill",
                                page_idx,
                                exc_info=True,
                            )
                    finally:
                        textpage.close()
                finally:
                    page.close()
        finally:
            doc.close()
    return pages_regions
