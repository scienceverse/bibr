"""URI link annotations of a PDF: page, rectangle and target URI.

Some publishers print a "[CrossRef]", "[PubMed]" or "Google Scholar" label
after each reference and put the reference's DOI only in that label's link
annotation (MDPI, BMJ, IOP): the text layer never prints it. The reader here
returns every URI link with its rectangle, so a caller can map a link onto the
text beneath it. :func:`doi_from_uri` recognises a DOI link target.

Rectangles are PDF points in the page's user space (origin bottom-left, y up),
the frame the text layer's character boxes use.
"""

from __future__ import annotations

import ctypes
import html
from dataclasses import dataclass
from urllib.parse import unquote

from bibr.ocr.utils import pdfium_lock
from bibr.utils.text import normalize_doi


@dataclass(frozen=True)
class PdfUriLink:
    """One URI link annotation on a page."""

    page_index: int
    rect: tuple[float, float, float, float]
    uri: str


def page_uri_links(document, page, page_index: int) -> list[PdfUriLink]:
    """Every URI link annotation on an open *page* of an open *document*.

    Operates on caller-provided pypdfium2 objects; the caller holds
    :data:`bibr.ocr.utils.pdfium_lock`. Links whose action is not a URI (a
    jump inside the document, a launch action) are skipped.
    """
    import pypdfium2.raw as pdfium_c

    links: list[PdfUriLink] = []
    position = ctypes.c_int(0)
    link = pdfium_c.FPDF_LINK()
    while pdfium_c.FPDFLink_Enumerate(page.raw, ctypes.byref(position), ctypes.byref(link)):
        action = pdfium_c.FPDFLink_GetAction(link)
        if not action or pdfium_c.FPDFAction_GetType(action) != pdfium_c.PDFACTION_URI:
            continue
        rect = pdfium_c.FS_RECTF()
        if not pdfium_c.FPDFLink_GetAnnotRect(link, ctypes.byref(rect)):
            continue
        # The URI is 7-bit ASCII with a terminating NUL; the first call sizes it.
        size = pdfium_c.FPDFAction_GetURIPath(document.raw, action, None, 0)
        if size <= 1:
            continue
        buffer = ctypes.create_string_buffer(size)
        pdfium_c.FPDFAction_GetURIPath(document.raw, action, buffer, size)
        uri = buffer.raw[: size - 1].decode("utf-8", "replace").strip()
        if not uri:
            continue
        left, right = sorted((float(rect.left), float(rect.right)))
        bottom, top = sorted((float(rect.bottom), float(rect.top)))
        links.append(PdfUriLink(page_index=page_index, rect=(left, bottom, right, top), uri=uri))
    return links


def read_uri_links(
    pdf_bytes: bytes, page_indices: list[int] | tuple[int, ...] | None = None
) -> list[PdfUriLink]:
    """Every URI link annotation in *pdf_bytes*, in page order.

    ``page_indices`` limits the read to those 0-based pages; out-of-range
    indices are ignored.
    """
    import pypdfium2

    links: list[PdfUriLink] = []
    with pdfium_lock:
        document = pypdfium2.PdfDocument(pdf_bytes)
        try:
            count = len(document)
            indices = range(count) if page_indices is None else page_indices
            for page_index in indices:
                if not 0 <= page_index < count:
                    continue
                page = document[page_index]
                try:
                    links.extend(page_uri_links(document, page, page_index))
                finally:
                    page.close()
        finally:
            document.close()
    return links


# HTML entities a publisher left in the URI, sometimes escaped twice
# ("&amp;lt;" for "<"), are undone this many times at most.
_MAX_UNESCAPES = 3


def doi_from_uri(uri: str | None) -> str | None:
    """The DOI a link targets, or None.

    Accepts doi.org / dx.doi.org URLs (http or https) and ``doi:`` URIs, with
    HTML entities and percent-encoding undone. Any other URL, including a
    publisher page whose path merely contains a DOI, returns None, and so
    does a target holding a NUL or a replacement character or that is not
    DOI-shaped once decoded.
    """
    if not uri:
        return None
    text = uri.strip()
    for _ in range(_MAX_UNESCAPES):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    lower = text.lower()
    if lower.startswith("doi:"):
        candidate = text[4:]
    else:
        for prefix in ("https://", "http://"):
            if lower.startswith(prefix):
                lower = lower[len(prefix) :]
                text = text[len(prefix) :]
                break
        else:
            return None
        for host in ("doi.org/", "dx.doi.org/", "www.doi.org/"):
            if lower.startswith(host):
                candidate = text[len(host) :]
                break
        else:
            return None
    candidate = unquote(candidate.split("#", 1)[0].split("?", 1)[0]).strip()
    if "\x00" in candidate or "\ufffd" in candidate:
        return None
    return normalize_doi(candidate)
