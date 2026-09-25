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
    into. Falls back to ``(0, 0, width, height)`` when the CropBox is missing
    or degenerate so callers always get a usable box.
    """
    try:
        x0, y0, x1, y1 = page.get_cropbox()
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


def _build_page_char_records(textpage) -> list[tuple[str, float, float, bool]]:
    """Precompute ``(char, center_x, center_y, is_newline)`` for every char on the page.

    ``get_text_bounded`` assigns a glyph to a region whenever its char box merely
    INTERSECTS the query rect, so a region bbox edge that slices through an
    adjacent line bleeds in that neighbor's partial glyphs as garbage. Using the
    glyph's CENTER instead assigns each char to exactly one region.

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
            records.append((ch, 0.0, 0.0, True))
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
        records.append((ch, (cl + cr) / 2.0, (cb + ct) / 2.0, False))
        i += consumed
    return records


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
                textpage = page.get_textpage()
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
# Header and footer ARE eligible: their only consumers (detected_headers /
# detected_footers feeding DOI furniture candidates and front-matter
# masthead checks) need plain text, which the native layer returns exactly
# on born-digital PDFs — they are about half of all OCR calls there. The
# printable-ratio corruption gate still applies.
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
        "header",
        "footer",
    }
)


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
                    textpage = page.get_textpage()
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
                    textpage = page.get_textpage()
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
                    textpage = page.get_textpage()
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
