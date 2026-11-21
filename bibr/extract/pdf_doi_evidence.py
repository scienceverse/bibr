"""DOI evidence a PDF carries outside its parsed text.

The identity stage reads the paper's DOI from the parsed text: sentences, page
furniture and the input's structured metadata. A PDF carries it in places the
parse never reads. Text-layer lines the layout did not turn into a region (a
repository banner rotated into the page margin, a masthead) never reach the
text. Link annotations and the document-information dictionary are not text
at all. Text the furniture strip removes as hidden (white on white, off the
page, invisible on a born-digital page; rule hidden_text in
``bibr.ocr.native_text``) never prints. :func:`read_pdf_doi_evidence` reads
those, best effort, from the input bytes. ``doi_identity`` decides what each is
worth: a DOI the pages print can name the paper, one found only in metadata, in
a link target or in hidden text can only agree with a printed one.

Positions are in the 0..1000 frame the layout regions use (``bbox_2d``: y
down, origin top-left, the page's ``/Rotate`` applied), so a text-layer line
can be matched to the region it was printed in.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from bibr.utils.text import DOI_CANDIDATE_RE

# Document-information keys publishers put a DOI in: Elsevier and Springer a
# citation line in Subject ("… doi:10.1016/…"), Elsevier also a ``/doi`` key,
# Wiley ``/WPS-ARTICLEDOI``.
_INFO_KEYS = ("Subject", "Keywords", "doi", "DOI", "WPS-ARTICLEDOI")
# Hyphen and dash look-alikes some typesetters use inside DOIs, mapped one to
# one so character positions stay aligned with the text layer.
_HYPHEN_LOOKALIKES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2212": "-"})
# Zero-width characters that split a DOI in the text layer without printing
# anything (a zero-width space after each hyphen in BMC DOIs).
_ZERO_WIDTH = frozenset({"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"})

# Text render modes that paint nothing: invisible text (3), such as a scan's
# hidden OCR layer, and text that only clips (7). Such text is not printed.
_UNPAINTED_RENDER_MODES = frozenset({3, 7})

# How far around a link rectangle, in points, the printed DOI may sit: a link on
# the first line of a wrapped resolver URL covers only that line.
_LINK_MARGIN_X = 2.0
_LINK_MARGIN_LINES = 1.0
# DOI links read per page; a page of thousands of links is a reference index.
_MAX_DOI_LINKS_PER_PAGE = 200


@dataclass(frozen=True)
class TextLayerLine:
    """One text-layer line that holds a DOI-shaped string."""

    page: int  # 1-based
    text: str
    # Centre of each character of ``text`` in the layout frame; None for a
    # character the text layer gives no box.
    centers: tuple[tuple[float, float] | None, ...]
    # The next line on the page ("" after the last), to tell a DOI wrapped
    # onto it from one the parse ran into it.
    next_text: str


@dataclass(frozen=True)
class LinkDoi:
    """A link annotation whose target is a DOI."""

    page: int  # 1-based
    doi: str
    uri: str
    rect: tuple[float, float, float, float]  # layout frame (x1, y1, x2, y2)
    # Text-layer text under the rectangle and one line around it.
    printed_text: str


@dataclass(frozen=True)
class MetadataDoi:
    """A DOI-bearing value of the document metadata."""

    source: str  # "pdf_info"
    key: str  # Info key
    value: str


@dataclass(frozen=True)
class HiddenDoi:
    """A DOI-bearing text object the furniture strip removed as hidden text."""

    page: int  # 1-based
    text: str  # the object's text, whitespace-normalised


@dataclass(frozen=True)
class PdfDoiEvidence:
    pages: tuple[int, ...]  # 1-based pages that were read
    lines: tuple[TextLayerLine, ...]
    links: tuple[LinkDoi, ...]
    metadata: tuple[MetadataDoi, ...]
    hidden: tuple[HiddenDoi, ...] = ()


def is_pdf(data: bytes | None) -> bool:
    """Whether *data* looks like a PDF (a ``%PDF-`` header near the start)."""
    return data is not None and b"%PDF-" in data[:1024]


def _to_layout_point(
    x: float,
    y: float,
    crop_box: tuple[float, float, float, float],
    rotation: int,
) -> tuple[float, float]:
    """Map a text-layer point (PDF points, y up) into the layout frame.

    Inverse of ``native_text._normalized_bbox_to_pdf_points`` for one point.
    """
    cx0, cy0, cx1, cy1 = crop_box
    u = (x - cx0) / ((cx1 - cx0) or 1.0) * 1000.0
    v = (cy1 - y) / ((cy1 - cy0) or 1.0) * 1000.0
    if rotation == 90:
        return (1000.0 - v, u)
    if rotation == 180:
        return (1000.0 - u, 1000.0 - v)
    if rotation == 270:
        return (v, 1000.0 - u)
    return (u, v)


def _printed_char_records(
    textpage,
    *,
    codes: list[int] | None = None,
    char_boxes: list[tuple[float, float, float, float] | None] | None = None,
) -> list[tuple[str, float, float, bool]]:
    """``(char, centre x, centre y, is_newline)`` for each printed character.

    Characters in an unpainted text object (render mode 3 or 7) are left out,
    and so is a character the text layer gives no box.
    """
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    records: list[tuple[str, float, float, bool]] = []
    count = textpage.count_chars()
    if not count:
        return records
    raw = textpage.raw
    get_unicode = pdfium_c.FPDFText_GetUnicode
    get_text_obj = pdfium_c.FPDFText_GetTextObject
    get_render_mode = pdfium_c.FPDFTextObj_GetTextRenderMode
    unpainted = _UNPAINTED_RENDER_MODES
    use_codes = codes is not None and len(codes) == count
    use_boxes = char_boxes is not None and len(char_boxes) == count

    if use_boxes:

        def _box(idx: int) -> tuple[float, float, float, float] | None:
            return char_boxes[idx]
    elif isinstance(textpage, pdfium.PdfTextPage):
        _c_l = ctypes.c_double()
        _c_r = ctypes.c_double()
        _c_b = ctypes.c_double()
        _c_t = ctypes.c_double()
        _b_l = ctypes.byref(_c_l)
        _b_r = ctypes.byref(_c_r)
        _b_b = ctypes.byref(_c_b)
        _b_t = ctypes.byref(_c_t)
        _get_box = pdfium_c.FPDFText_GetCharBox

        def _box(idx: int) -> tuple[float, float, float, float] | None:
            if _get_box(raw, idx, _b_l, _b_r, _b_b, _b_t):
                return (_c_l.value, _c_b.value, _c_r.value, _c_t.value)
            return None
    else:

        def _box(idx: int) -> tuple[float, float, float, float] | None:
            try:
                return textpage.get_charbox(idx)
            except Exception:  # noqa: BLE001 - a char without a box is not placed
                return None

    append = records.append
    index = 0
    while index < count:
        code = codes[index] if use_codes else get_unicode(raw, index)
        width = 1
        # Non-BMP text arrives as a UTF-16 surrogate pair.
        if 0xD800 <= code <= 0xDBFF and index + 1 < count:
            low = codes[index + 1] if use_codes else get_unicode(raw, index + 1)
            if 0xDC00 <= low <= 0xDFFF:
                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                width = 2
        ch = "\ufffd" if 0xD800 <= code <= 0xDFFF or code > 0x10FFFF else chr(code)
        if ch in ("\n", "\r"):
            append((ch, 0.0, 0.0, True))
            index += width
            continue
        text_object = get_text_obj(raw, index)
        if text_object and get_render_mode(text_object) in unpainted:
            index += width
            continue
        box = _box(index)
        if box is None:
            index += width
            continue
        left, bottom, right, top = box
        append((ch, (left + right) / 2.0, (bottom + top) / 2.0, False))
        index += width
    return records


def _page_lines(records, to_layout) -> list[tuple[str, tuple[tuple[float, float] | None, ...]]]:
    """Split the page's character records into lines, as the text layer breaks them."""
    lines: list[tuple[str, tuple[tuple[float, float] | None, ...]]] = []
    chars: list[str] = []
    centers: list[tuple[float, float] | None] = []
    for ch, cx, cy, is_newline in records:
        if is_newline:
            if chars:
                lines.append(("".join(chars), tuple(centers)))
                chars, centers = [], []
            continue
        if ch in _ZERO_WIDTH:
            continue
        chars.append(ch.translate(_HYPHEN_LOOKALIKES))
        centers.append(to_layout(cx, cy))
    if chars:
        lines.append(("".join(chars), tuple(centers)))
    return lines


def _text_near(records, rect_pts: tuple[float, float, float, float]) -> str:
    """Text-layer characters whose centre lies in or just around a link rectangle."""
    left, bottom, right, top = rect_pts
    reach = (top - bottom) * _LINK_MARGIN_LINES
    left, right = left - _LINK_MARGIN_X, right + _LINK_MARGIN_X
    bottom, top = bottom - reach, top + reach
    parts: list[str] = []
    for ch, cx, cy, is_newline in records:
        if is_newline:
            if parts and parts[-1] != "\n":
                parts.append("\n")
            continue
        if ch not in _ZERO_WIDTH and left <= cx <= right and bottom <= cy <= top:
            parts.append(ch.translate(_HYPHEN_LOOKALIKES))
    return "".join(parts).strip()


def build_pdf_doi_evidence(
    info_values: Mapping[str, str],
    raw_pages: list[tuple[int, tuple[float, float, float, float], int, list, list, list[tuple]]],
) -> PdfDoiEvidence:
    """Assemble :class:`PdfDoiEvidence` from detached per-page PDFium data (lock-free)."""
    from bibr.ocr.pdf_links import doi_from_uri

    metadata: list[MetadataDoi] = []
    for key in _INFO_KEYS:
        value = str(info_values.get(key) or "")
        if DOI_CANDIDATE_RE.search(value):
            metadata.append(MetadataDoi("pdf_info", key, " ".join(value.split())[:300]))

    ordered = sorted(raw_pages, key=lambda item: item[0])
    hidden: list[HiddenDoi] = []
    lines: list[TextLayerLine] = []
    links: list[LinkDoi] = []
    for page_number, crop_box, rotation, records, page_links, furniture in ordered:
        hidden.extend(
            HiddenDoi(page_number, text[:300])
            for kind, _box, text in furniture
            if kind == "hidden_text" and isinstance(text, str) and DOI_CANDIDATE_RE.search(text)
        )

        def to_layout(x, y, crop_box=crop_box, rotation=rotation):
            return _to_layout_point(x, y, crop_box, rotation)

        page_lines = _page_lines(records, to_layout)
        for index, (text, centers) in enumerate(page_lines):
            if "10." in text and DOI_CANDIDATE_RE.search(text):
                following = page_lines[index + 1][0] if index + 1 < len(page_lines) else ""
                lines.append(TextLayerLine(page_number, text, centers, following))
        doi_links = 0
        for link in page_links:
            doi = doi_from_uri(link.uri)
            if doi is None:
                continue
            doi_links += 1
            if doi_links > _MAX_DOI_LINKS_PER_PAGE:
                break
            left, bottom, right, top = link.rect
            corners = (to_layout(left, top), to_layout(right, bottom))
            rect = (
                min(c[0] for c in corners),
                min(c[1] for c in corners),
                max(c[0] for c in corners),
                max(c[1] for c in corners),
            )
            links.append(LinkDoi(page_number, doi, link.uri, rect, _text_near(records, link.rect)))
    pages_read = tuple(page_number for page_number, *_rest in ordered)
    return PdfDoiEvidence(pages_read, tuple(lines), tuple(links), tuple(metadata), tuple(hidden))


def read_pdf_doi_evidence(pdf_bytes: bytes, pages: Iterable[int]) -> PdfDoiEvidence:
    """Read the text-layer DOI lines, DOI links and hidden-text DOIs of *pages*, and the Info DOIs.

    *pages* are 1-based; pages the PDF does not have are skipped. Only the
    character and link records are read under the process-wide PDFium lock;
    lines and link texts are built after it is released. Raises on an
    unreadable PDF; the caller treats that as no evidence.
    """
    import pypdfium2

    from bibr.ocr.native_text import _page_crop_box, _page_rotation, open_text_page
    from bibr.ocr.pdf_links import page_uri_links
    from bibr.ocr.utils import pdfium_lock

    info_values: dict[str, str] = {}
    read: list[tuple[int, tuple[float, float, float, float], int, list, list, list[tuple]]] = []
    with pdfium_lock:
        document = pypdfium2.PdfDocument(pdf_bytes)
        try:
            for key in _INFO_KEYS:
                info_values[key] = str(document.get_metadata_value(key) or "")
            count = len(document)
            for page_number in sorted(set(pages)):
                if not 1 <= page_number <= count:
                    continue
                page = document[page_number - 1]
                try:
                    furniture: list[tuple] = []
                    textpage = open_text_page(page, None, furniture)
                    try:
                        records = _printed_char_records(textpage)
                    finally:
                        textpage.close()
                    page_links = page_uri_links(document, page, page_number - 1)
                    read.append(
                        (
                            page_number,
                            _page_crop_box(page),
                            _page_rotation(page),
                            records,
                            page_links,
                            furniture,
                        )
                    )
                finally:
                    page.close()
        finally:
            document.close()

    return build_pdf_doi_evidence(info_values, read)
