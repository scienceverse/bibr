"""DOI evidence a PDF carries outside its parsed text.

The identity stage reads the paper's DOI from the parsed text: sentences, page
furniture and the input's structured metadata. A PDF carries it in places the
parse never reads. Text-layer lines the layout did not turn into a region (a
repository banner rotated into the page margin, a masthead) never reach the
text. Link annotations and the document-information dictionary are not text
at all. :func:`read_pdf_doi_evidence` reads those, best effort, from the input
bytes. ``doi_identity`` decides what each is worth: a
DOI the pages print can name the paper, one found only in metadata or in a
link target can only agree with a printed one.

Positions are in the 0..1000 frame the layout regions use (``bbox_2d``: y
down, origin top-left, the page's ``/Rotate`` applied), so a text-layer line
can be matched to the region it was printed in.
"""

from __future__ import annotations

from collections.abc import Iterable
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
class PdfDoiEvidence:
    pages: tuple[int, ...]  # 1-based pages that were read
    lines: tuple[TextLayerLine, ...]
    links: tuple[LinkDoi, ...]
    metadata: tuple[MetadataDoi, ...]


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


def _printed_char_records(textpage) -> list[tuple[str, float, float, bool]]:
    """``(char, centre x, centre y, is_newline)`` for each printed character.

    Characters in an unpainted text object (render mode 3 or 7) are left out,
    and so is a character the text layer gives no box.
    """
    import pypdfium2.raw as pdfium_c

    records: list[tuple[str, float, float, bool]] = []
    count = textpage.count_chars()
    index = 0
    while index < count:
        code = pdfium_c.FPDFText_GetUnicode(textpage.raw, index)
        width = 1
        # Non-BMP text arrives as a UTF-16 surrogate pair.
        if 0xD800 <= code <= 0xDBFF and index + 1 < count:
            low = pdfium_c.FPDFText_GetUnicode(textpage.raw, index + 1)
            if 0xDC00 <= low <= 0xDFFF:
                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                width = 2
        ch = "\ufffd" if 0xD800 <= code <= 0xDFFF or code > 0x10FFFF else chr(code)
        if ch in ("\n", "\r"):
            records.append((ch, 0.0, 0.0, True))
            index += width
            continue
        text_object = pdfium_c.FPDFText_GetTextObject(textpage.raw, index)
        if (
            text_object
            and pdfium_c.FPDFTextObj_GetTextRenderMode(text_object) in _UNPAINTED_RENDER_MODES
        ):
            index += width
            continue
        try:
            left, bottom, right, top = textpage.get_charbox(index)
        except Exception:  # noqa: BLE001 - a char without a box is not placed
            index += width
            continue
        records.append((ch, (left + right) / 2.0, (bottom + top) / 2.0, False))
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


def read_pdf_doi_evidence(pdf_bytes: bytes, pages: Iterable[int]) -> PdfDoiEvidence:
    """Read the text-layer DOI lines and DOI links of *pages*, and the Info DOIs.

    *pages* are 1-based; pages the PDF does not have are skipped. Only the
    character and link records are read under the process-wide PDFium lock;
    lines and link texts are built after it is released. Raises on an
    unreadable PDF; the caller treats that as no evidence.
    """
    import pypdfium2

    from bibr.ocr.native_text import _page_crop_box, _page_rotation
    from bibr.ocr.pdf_links import doi_from_uri, page_uri_links
    from bibr.ocr.utils import pdfium_lock

    metadata: list[MetadataDoi] = []
    read: list[tuple[int, tuple, int, list, list]] = []
    with pdfium_lock:
        document = pypdfium2.PdfDocument(pdf_bytes)
        try:
            for key in _INFO_KEYS:
                value = str(document.get_metadata_value(key) or "")
                if DOI_CANDIDATE_RE.search(value):
                    metadata.append(MetadataDoi("pdf_info", key, " ".join(value.split())[:300]))
            count = len(document)
            for page_number in sorted(set(pages)):
                if not 1 <= page_number <= count:
                    continue
                page = document[page_number - 1]
                try:
                    textpage = page.get_textpage()
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
                        )
                    )
                finally:
                    page.close()
        finally:
            document.close()

    lines: list[TextLayerLine] = []
    links: list[LinkDoi] = []
    for page_number, crop_box, rotation, records, page_links in read:

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
    pages_read = tuple(page_number for page_number, *_rest in read)
    return PdfDoiEvidence(pages_read, tuple(lines), tuple(links), tuple(metadata))
