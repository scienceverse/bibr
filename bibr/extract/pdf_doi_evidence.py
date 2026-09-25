"""DOI evidence a PDF carries outside its parsed text.

The identity stage reads the paper's DOI from the parsed text: sentences, page
furniture and the input's structured metadata. A PDF carries it in places the
parse never reads. Text-layer lines the layout did not turn into a region (a
repository banner rotated into the page margin, a masthead) never reach the
text. Link annotations, the document-information dictionary and the XMP
packet are not text at all. :func:`read_pdf_doi_evidence` reads those, best
effort, from the input bytes. ``doi_identity`` decides what each is worth: a
DOI the pages print can name the paper, one found only in metadata or in a
link target can only agree with a printed one.

Positions are in the 0..1000 frame the layout regions use (``bbox_2d``: y
down, origin top-left, the page's ``/Rotate`` applied), so a text-layer line
can be matched to the region it was printed in.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

from bibr.utils.text import DOI_CANDIDATE_RE

logger = logging.getLogger(__name__)

# Document-information keys publishers put a DOI in: Elsevier and Springer a
# citation line in Subject ("… doi:10.1016/…"), Elsevier also a ``/doi`` key,
# Wiley ``/WPS-ARTICLEDOI``.
_INFO_KEYS = ("Subject", "Keywords", "doi", "DOI", "WPS-ARTICLEDOI")
# XMP properties that name the document's DOI. ``prism:url`` and
# ``dc:identifier`` also carry other identifiers; only DOI-shaped values count.
_XMP_PROPERTIES = (
    "prism:doi",
    "dc:identifier",
    "crossmark:DOI",
    "pdfx:doi",
    "pdfx:DOI",
    "pdfx:WPS-ARTICLEDOI",
    "prism:url",
)
_XMP_PROPERTY_RE = re.compile(
    r"<(?P<tag>" + "|".join(re.escape(p) for p in _XMP_PROPERTIES) + r")\b[^>]*>"
    r"(?P<value>.{0,2000}?)</(?P=tag)>"
    r"|\b(?P<attr>" + "|".join(re.escape(p) for p in _XMP_PROPERTIES) + r")\s*=\s*"
    r"\"(?P<attr_value>[^\"]{0,500})\"",
    re.DOTALL,
)
_XMP_START = b"<x:xmpmeta"
_XMP_END = b"</x:xmpmeta>"
# A document-level packet is a few kilobytes; bound the read and the count so a
# file full of per-image packets cannot make this slow.
_MAX_XMP_PACKET_BYTES = 256 * 1024
_MAX_XMP_PACKETS = 4

# Hyphen and dash look-alikes some typesetters use inside DOIs, mapped one to
# one so character positions stay aligned with the text layer.
_HYPHEN_LOOKALIKES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2212": "-"})
# Invisible characters that split a DOI in the text layer without printing
# anything (a zero-width space after each hyphen in BMC DOIs).
_INVISIBLE = frozenset({"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"})

# How far around a link rectangle, in points, the printed DOI may sit: a link on
# the first line of a wrapped resolver URL covers only that line.
_LINK_MARGIN_X = 2.0
_LINK_MARGIN_LINES = 1.0


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

    source: str  # "pdf_info" or "pdf_xmp"
    key: str  # Info key or XMP property
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
        if ch in _INVISIBLE:
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
        if ch not in _INVISIBLE and left <= cx <= right and bottom <= cy <= top:
            parts.append(ch.translate(_HYPHEN_LOOKALIKES))
    return "".join(parts).strip()


def _xmp_values(pdf_bytes: bytes) -> list[MetadataDoi]:
    found: list[MetadataDoi] = []
    start = 0
    for _ in range(_MAX_XMP_PACKETS):
        begin = pdf_bytes.find(_XMP_START, start)
        if begin < 0:
            break
        end = pdf_bytes.find(_XMP_END, begin, begin + _MAX_XMP_PACKET_BYTES)
        if end < 0:
            break
        packet = pdf_bytes[begin:end].decode("utf-8", "replace")
        start = end + len(_XMP_END)
        for match in _XMP_PROPERTY_RE.finditer(packet):
            key = match.group("tag") or match.group("attr")
            value = match.group("value") if match.group("tag") else match.group("attr_value")
            if value and DOI_CANDIDATE_RE.search(value):
                found.append(MetadataDoi("pdf_xmp", key, " ".join(value.split())[:300]))
    return found


def read_pdf_doi_evidence(pdf_bytes: bytes, pages: Iterable[int]) -> PdfDoiEvidence:
    """Read the text-layer DOI lines and DOI links of *pages*, and the metadata DOIs.

    *pages* are 1-based; pages the PDF does not have are skipped. Raises on an
    unreadable PDF; the caller treats that as no evidence.
    """
    import pypdfium2

    from bibr.ocr.native_text import _build_page_char_records, _page_crop_box, _page_rotation
    from bibr.ocr.pdf_links import doi_from_uri, page_uri_links
    from bibr.ocr.utils import pdfium_lock

    lines: list[TextLayerLine] = []
    links: list[LinkDoi] = []
    metadata: list[MetadataDoi] = []
    read_pages: list[int] = []
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
                    crop_box = _page_crop_box(page)
                    rotation = _page_rotation(page)

                    def to_layout(x, y, crop_box=crop_box, rotation=rotation):
                        return _to_layout_point(x, y, crop_box, rotation)

                    textpage = page.get_textpage()
                    try:
                        records = _build_page_char_records(textpage)
                    finally:
                        textpage.close()
                    read_pages.append(page_number)
                    page_lines = _page_lines(records, to_layout)
                    for index, (text, centers) in enumerate(page_lines):
                        if "10." in text and DOI_CANDIDATE_RE.search(text):
                            following = (
                                page_lines[index + 1][0] if index + 1 < len(page_lines) else ""
                            )
                            lines.append(TextLayerLine(page_number, text, centers, following))
                    for link in page_uri_links(document, page, page_number - 1):
                        doi = doi_from_uri(link.uri)
                        if doi is None:
                            continue
                        left, bottom, right, top = link.rect
                        corners = (to_layout(left, top), to_layout(right, bottom))
                        rect = (
                            min(c[0] for c in corners),
                            min(c[1] for c in corners),
                            max(c[0] for c in corners),
                            max(c[1] for c in corners),
                        )
                        links.append(
                            LinkDoi(
                                page_number, doi, link.uri, rect, _text_near(records, link.rect)
                            )
                        )
                finally:
                    page.close()
        finally:
            document.close()
    try:
        metadata.extend(_xmp_values(pdf_bytes))
    except Exception:  # noqa: BLE001 - a malformed packet is no evidence, not a failure
        logger.debug("XMP DOI read failed", exc_info=True)
    return PdfDoiEvidence(tuple(read_pages), tuple(lines), tuple(links), tuple(metadata))
