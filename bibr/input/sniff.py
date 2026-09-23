"""Content-based file-type detection for input validation.

bibr accepts a handful of formats (PDF, DOCX, JATS XML, HTML and ePub), each
with a cheap, unambiguous signature, so the type that validation acts on is
determined here in pure Python and needs no system library.

libmagic, through the ``python-magic`` binding, is an optional hint. It is
imported on first use, only for content none of the rules below recognize, and
only to give such content a more specific name in logs (an image, a legacy
``.doc``). A missing or broken libmagic is never an error, and its guess can
never claim a type these rules decide, so a file is accepted or rejected the
same way whether or not libmagic is installed.
"""

from __future__ import annotations

import functools
import io
import logging
import re
import zipfile
import zlib
from types import ModuleType

logger = logging.getLogger(__name__)

PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
EPUB_MIME = "application/epub+zip"
ZIP_MIME = "application/zip"
XML_MIME = "text/xml"
HTML_MIME = "text/html"
XHTML_MIME = "application/xhtml+xml"
WINDOWS_EXECUTABLE_MIME = "application/x-msdownload"
ELF_EXECUTABLE_MIME = "application/x-executable"
TEXT_MIME = "text/plain"
EMPTY_MIME = "application/x-empty"
UNKNOWN_MIME = "application/octet-stream"

EXECUTABLE_MIMES: frozenset[str] = frozenset({WINDOWS_EXECUTABLE_MIME, ELF_EXECUTABLE_MIME})

# Every type the rules below can report. For these the rules are authoritative:
# a libmagic guess naming one of them is discarded rather than trusted.
_SNIFFED_MIMES: frozenset[str] = frozenset(
    {
        PDF_MIME,
        DOCX_MIME,
        EPUB_MIME,
        ZIP_MIME,
        XML_MIME,
        HTML_MIME,
        XHTML_MIME,
        *EXECUTABLE_MIMES,
    }
)

_PDF_SIGNATURE = b"%PDF-"
# PDF readers accept the header anywhere in the first KiB, after junk or a
# byte-order mark. Past offset 0 a version number is also required, so text
# that merely mentions the header is not taken for a PDF.
_PDF_VERSIONED_HEADER = re.compile(rb"%PDF-\d\.\d")
_PDF_HEADER_SEARCH_END = 1024 + len(b"%PDF-1.7")

_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06")
_ZIP_LOCAL_HEADER_BYTES = 30
_EPUB_MIMETYPE = b"application/epub+zip"
_DOCX_MAIN_PART = "word/document.xml"

_ELF_SIGNATURE = b"\x7fELF"

_MARKUP_SNIFF_BYTES = 8192
_MAX_PROLOG_TOKENS = 64
# Comments, processing instructions and a DOCTYPE may precede the first element.
# Every alternative is anchored and cannot backtrack across alternatives, so a
# hostile head cannot make the scan super-linear.
_PROLOG_TOKEN = re.compile(
    r"""\s*(?:
        <!--.*?-->
      | <\?(?P<pi>[^\s?>]*).*?\?>
      | <!DOCTYPE\s+(?P<doctype>[^\s>\[]+)(?:[^>\[]|\[[^\]]*\])*>
      | <(?P<element>[A-Za-z_][\w.:-]*)
    )""",
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)
# First elements that only occur in HTML (a JATS document starts at <article>).
_HTML_LEADING_TAGS = frozenset(
    {
        "html",
        "head",
        "body",
        "title",
        "meta",
        "link",
        "base",
        "script",
        "style",
        "main",
        "div",
        "p",
        "table",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }
)

_UTF16_BOMS = ((b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"))
_UTF8_BOM = b"\xef\xbb\xbf"
# Control bytes that do not occur in text files (tab, newlines, form feed,
# backspace and escape do).
_BINARY_BYTES = bytes(b for b in range(0x20) if b not in b"\t\n\r\f\b\x1b")


def sniff_mime_type(content: bytes) -> str | None:
    """Return the MIME type of *content* when a built-in rule recognizes it.

    Recognizes PDF, DOCX, ePub and other zip archives, XML and HTML, and
    Windows/ELF executables. Returns ``None`` for anything else; what that
    content is does not affect validation.
    """
    if content.startswith(_PDF_SIGNATURE):
        return PDF_MIME
    if content.startswith(_ZIP_SIGNATURES):
        return _sniff_zip(content)
    if _is_windows_executable(content):
        return WINDOWS_EXECUTABLE_MIME
    if content.startswith(_ELF_SIGNATURE):
        return ELF_EXECUTABLE_MIME
    markup = _sniff_markup(content)
    if markup is not None:
        return markup
    if _PDF_VERSIONED_HEADER.search(content, 0, _PDF_HEADER_SEARCH_END):
        return PDF_MIME
    return None


def detect_mime_type(file_content: bytes) -> str:
    """Return a MIME type for *file_content*.

    Built-in rules decide the supported formats. Other content gets libmagic's
    name for it when libmagic is usable, else ``text/plain``,
    ``application/x-empty`` or ``application/octet-stream``. A missing or
    failing libmagic never raises.
    """
    return (
        sniff_mime_type(file_content)
        or _libmagic_hint(file_content)
        or _fallback_mime_type(file_content)
    )


def libmagic_available() -> bool:
    """Whether the optional libmagic hint is usable on this host."""
    return _load_libmagic() is not None


@functools.cache
def _load_libmagic() -> ModuleType | None:
    """Import ``python-magic`` and probe libmagic once; ``None`` if either fails.

    The binding raises ``ImportError`` at import time when the system library
    is missing, and a present library can still fail on first use (no magic
    database), so both are tried here and every failure means "no hint".
    """
    try:
        import magic

        magic.from_buffer(_PDF_SIGNATURE, mime=True)
    except Exception as exc:  # noqa: BLE001 - libmagic is optional; any failure disables it
        logger.debug("libmagic unavailable, using built-in file detection only: %s", exc)
        return None
    return magic


def _libmagic_hint(content: bytes) -> str | None:
    magic = _load_libmagic()
    if magic is None:
        return None
    try:
        hint = magic.from_buffer(content, mime=True)
    except Exception:  # noqa: BLE001 - a failed hint falls back to the built-in name
        logger.debug("libmagic could not identify the input", exc_info=True)
        return None
    if not isinstance(hint, str) or not hint or hint in _SNIFFED_MIMES:
        return None
    return hint


def _fallback_mime_type(content: bytes) -> str:
    if not content:
        return EMPTY_MIME
    head = content[:_MARKUP_SNIFF_BYTES]
    if head.startswith(tuple(bom for bom, _ in _UTF16_BOMS)):
        return TEXT_MIME
    if len(head.translate(None, _BINARY_BYTES)) == len(head):
        return TEXT_MIME
    return UNKNOWN_MIME


def _sniff_zip(content: bytes) -> str:
    """Classify a zip archive as ePub, DOCX or a generic zip."""
    if _leading_member_is_epub_mimetype(content):
        return EPUB_MIME
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = set(zf.namelist())
            if "mimetype" in names:
                with zf.open("mimetype") as member:
                    if member.read(len(_EPUB_MIMETYPE) + 16).strip() == _EPUB_MIMETYPE:
                        return EPUB_MIME
            if _DOCX_MAIN_PART in names:
                return DOCX_MIME
    except (zipfile.BadZipFile, RuntimeError, OSError, EOFError, ValueError, zlib.error):
        # A damaged archive is still a zip; the per-format corruption checks
        # decide what the damage means for the file.
        logger.debug("Zip input could not be listed while sniffing", exc_info=True)
    return ZIP_MIME


def _leading_member_is_epub_mimetype(content: bytes) -> bool:
    """Check the ePub container's required first member from its local header.

    The OCF spec puts an uncompressed ``mimetype`` entry first, so this holds
    even when the rest of the archive is truncated.
    """
    if len(content) < _ZIP_LOCAL_HEADER_BYTES or not content.startswith(b"PK\x03\x04"):
        return False
    method = int.from_bytes(content[8:10], "little")
    name_len = int.from_bytes(content[26:28], "little")
    extra_len = int.from_bytes(content[28:30], "little")
    name_end = _ZIP_LOCAL_HEADER_BYTES + name_len
    if method != zipfile.ZIP_STORED or content[_ZIP_LOCAL_HEADER_BYTES:name_end] != b"mimetype":
        return False
    data_start = name_end + extra_len
    return content[data_start : data_start + len(_EPUB_MIMETYPE)] == _EPUB_MIMETYPE


def _is_windows_executable(content: bytes) -> bool:
    """An MZ header whose ``e_lfanew`` field points at a PE signature."""
    if not content.startswith(b"MZ") or len(content) < 0x40:
        return False
    pe_offset = int.from_bytes(content[0x3C:0x40], "little")
    return content[pe_offset : pe_offset + 4] == b"PE\x00\x00"


def _decode_markup_head(content: bytes) -> str:
    head = content[:_MARKUP_SNIFF_BYTES]
    if head.startswith(_UTF8_BOM):
        return head[len(_UTF8_BOM) :].decode("utf-8", errors="ignore")
    for bom, codec in _UTF16_BOMS:
        if head.startswith(bom):
            return head[len(bom) :].decode(codec, errors="ignore")
    # BOM-less UTF-16 markup starts with "<" and a NUL byte in either order.
    if head.startswith(b"<\x00"):
        return head.decode("utf-16-le", errors="ignore")
    if head.startswith(b"\x00<"):
        return head.decode("utf-16-be", errors="ignore")
    # Markup is ASCII in every other encoding a document declares; latin-1
    # maps each byte to one character without failing.
    return head.decode("latin-1")


def _sniff_markup(content: bytes) -> str | None:
    """Classify an XML or HTML document from its prolog and first element."""
    text = _decode_markup_head(content)
    xml_declared = False
    doctype: str | None = None
    root: str | None = None
    pos = 0
    for _ in range(_MAX_PROLOG_TOKENS):
        match = _PROLOG_TOKEN.match(text, pos)
        if match is None:
            break
        if match["element"]:
            root = match["element"]
            break
        if match["pi"] and match["pi"].lower().startswith("xml"):
            xml_declared = True
        if match["doctype"]:
            doctype = match["doctype"]
        pos = match.end()

    names = {name.rsplit(":", 1)[-1].lower() for name in (doctype, root) if name}
    if "html" in names:
        return XHTML_MIME if xml_declared else HTML_MIME
    if xml_declared or doctype is not None or "article" in names:
        # "article" is the JATS root, which bibr accepts without a prolog.
        return XML_MIME
    if root is not None and root.lower() in _HTML_LEADING_TAGS:
        return HTML_MIME
    return None
