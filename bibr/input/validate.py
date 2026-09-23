"""Input file validation module.

Validates input files by detecting MIME types, checking file type support,
cross-referencing extensions against detected content types, and populating
InputFile validation flags. Content types come from the pure-Python rules in
:mod:`bibr.input.sniff`, so validation needs no system library.
"""

import hashlib
import logging
import zipfile
import zlib
from pathlib import Path

from bibr.exceptions import InputValidationError
from bibr.input.file import InputFile, InputFormat
from bibr.input.sniff import EXECUTABLE_MIMES, detect_mime_type
from bibr.input.supported_files import (
    SUPPORTED_EXTENSIONS,
    UNSUPPORTED_EXTENSIONS,
    SupportedFileType,
)

logger = logging.getLogger(__name__)

# Content MIME types distinctive enough (magic bytes) that an extension mismatch
# means the file is mislabeled/spoofed, not a fuzzy guess. html<->xml overlap
# (XHTML is both), so they are intentionally excluded (audit L8).
_STRICT_MISMATCH_MIMES: frozenset[str] = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/epub+zip",
    }
)

# Mapping from MIME type to expected extensions for cross-validation
_MIME_TO_EXTENSIONS: dict[str, set[str]] = {
    "application/pdf": {".pdf"},
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {".docx"},
    "application/zip": {".docx", ".epub", ".zip"},
    "application/x-zip": {".docx", ".epub", ".zip"},
    "application/x-zip-compressed": {".docx", ".epub", ".zip"},
    "application/xml": {".xml"},
    "text/xml": {".xml"},
    "text/html": {".html", ".htm"},
    "application/xhtml+xml": {".html", ".htm"},
    "application/epub+zip": {".epub"},
}


def _check_extension_mime_consistency(extension: str, detected_mime: str) -> bool:
    """Check whether the file extension is consistent with the detected MIME type.

    Args:
        extension: File extension including dot, e.g. ".pdf".
        detected_mime: MIME type detected from file content.

    Returns:
        True if consistent (or if no mapping exists), False if mismatch detected.
    """
    expected_extensions = _MIME_TO_EXTENSIONS.get(detected_mime)
    if expected_extensions is None:
        # No mapping for this MIME type, assume consistent
        return True
    return extension.lower() in expected_extensions


def _check_pdf_corruption(file_content: bytes) -> bool:
    """Check if a PDF file appears corrupted.

    Checks for PDF magic bytes (%PDF-) at start and %%EOF marker near the end.

    Args:
        file_content: Raw PDF bytes.

    Returns:
        True if the file appears corrupted.
    """
    if not file_content.startswith(b"%PDF-"):
        return True
    # %%EOF should appear near the end.  Spec allows trailing whitespace or
    # comments after it, so search the last 8 KiB rather than the last 1 KiB.
    tail = file_content[-8192:] if len(file_content) > 8192 else file_content
    return tail.rfind(b"%%EOF") < 0


# OLE Compound File magic. A .docx that starts with this is not a zip —
# it is either a password-protected Office file (encrypted OOXML is stored
# inside a CFB container) or a legacy binary .doc renamed to .docx.
_CFB_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# Declared-uncompressed-size ceilings. A tiny stored zip entry can claim
# gigabytes of uncompressed payload (zip bomb) that python-docx would
# happily inflate downstream; reject such archives at validation.
_DOCX_DOCUMENT_XML_MAX_BYTES = 64 * 1024 * 1024
_DOCX_TOTAL_UNCOMPRESSED_MAX_BYTES = 128 * 1024 * 1024
_DOCX_MAX_ENTRIES = 10_000
_DOCX_MAX_COMPRESSION_RATIO = 100


def _check_html_corruption(file_content: bytes) -> tuple[bool, object | None]:
    """Check HTML content and return a reusable parsed DOM when valid."""
    try:
        from bibr.input.html_native import inspect_html
    except Exception:  # noqa: BLE001
        logger.warning("HTML validation helper unavailable", exc_info=True)
        return True, None
    has_content, soup = inspect_html(file_content)
    return not has_content, soup


def _check_epub_corruption(file_content: bytes) -> tuple[bool, object | None]:
    """Check an ePub and return its reusable parsed package when valid."""
    try:
        from bibr.input.epub_native import read_epub_document
    except Exception:  # noqa: BLE001
        logger.warning("ePub validation helper unavailable", exc_info=True)
        return True, None
    try:
        document = read_epub_document(file_content)
    except Exception:
        return True, None
    return False, document


def _check_docx_corruption(file_content: bytes) -> bool:
    """Check if a DOCX file appears corrupted.

    A DOCX is a zip archive containing ``word/document.xml``. Truncated
    archives, non-zip bytes, zips without the wordprocessing part
    (e.g. an xlsx renamed to .docx), and archives whose declared
    uncompressed sizes exceed sane ceilings (zip bombs) are all flagged
    so they fail at validation rather than deep in the parse stage.

    Args:
        file_content: Raw DOCX bytes.

    Returns:
        True if the file appears corrupted.
    """
    import io

    if file_content.startswith(_CFB_MAGIC):
        # Encrypted container is handled by the encryption check; a CFB
        # without it is a legacy binary .doc renamed to .docx — unparseable.
        return b"EncryptedPackage" not in file_content
    try:
        with zipfile.ZipFile(io.BytesIO(file_content)) as zf:
            if "word/document.xml" not in zf.namelist():
                return True
            entries = zf.infolist()
            if any(info.flag_bits & 0x1 for info in entries):
                # Zip-level member encryption. The archive is intact and
                # readable-with-a-password, so it is not corrupt — return
                # False and let _check_docx_encryption own the classification
                # (the pipeline reports is_corrupted before is_encrypted).
                return False
            if len(entries) > _DOCX_MAX_ENTRIES:
                logger.warning("DOCX contains too many archive entries: %d", len(entries))
                return True
            document_size = zf.getinfo("word/document.xml").file_size
            total_size = sum(info.file_size for info in entries)
            total_compressed = sum(info.compress_size for info in entries)
            compression_ratio = total_size / max(total_compressed, 1)
            if (
                document_size > _DOCX_DOCUMENT_XML_MAX_BYTES
                or total_size > _DOCX_TOTAL_UNCOMPRESSED_MAX_BYTES
                or compression_ratio > _DOCX_MAX_COMPRESSION_RATIO
            ):
                logger.warning(
                    "DOCX archive exceeds expansion limits: document.xml=%d B, "
                    "total=%d B, compressed=%d B, ratio=%.1f",
                    document_size,
                    total_size,
                    total_compressed,
                    compression_ratio,
                )
                return True
            # Declared sizes above are attacker-controlled central-directory
            # metadata; verify document.xml's *real* decompressed size too so the
            # check can't be bypassed by a lying header (audit M7).
            from bibr.input.zip_limits import uncompressed_size_within

            if not uncompressed_size_within(
                zf, "word/document.xml", max_bytes=_DOCX_DOCUMENT_XML_MAX_BYTES
            ):
                logger.warning("DOCX document.xml exceeds real-byte expansion limit")
                return True
            return False
    except zipfile.BadZipFile:
        return True
    except (RuntimeError, OSError, EOFError, ValueError, zlib.error):
        # Reading a member — not just opening the archive — can fail in ways
        # BadZipFile does not cover: zipfile raises RuntimeError for a
        # password-protected entry and NotImplementedError (a RuntimeError)
        # for a compression method it cannot inflate, and a truncated or
        # damaged deflate stream surfaces as zlib.error or EOFError from the
        # streaming size check. Every one of them means the archive is
        # unreadable, which is exactly what this function reports; letting
        # them escape crashed validation instead of classifying the file.
        logger.warning("DOCX archive could not be read", exc_info=True)
        return True


def _inspect_xml_root(file_content: bytes) -> tuple[bool, str]:
    """Return ``(is_corrupted, root_localname)`` for an XML document."""
    from lxml import etree

    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    try:
        root = etree.fromstring(file_content, parser=parser)
    except Exception:
        return True, ""
    tag = root.tag
    localname = tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""
    return False, localname


def _check_jats_article(file_content: bytes) -> tuple[bool, bool]:
    """Inspect XML input for JATS suitability.

    Returns ``(is_corrupted, is_jats_article)``:

    * The bytes are parsed with entity resolution disabled and networking off
      (the blob is user-supplied, so a crafted DOCTYPE must not leak local
      files via XXE or detonate an entity-expansion bomb).
    * Unparseable XML → ``(True, False)`` (corrupted).
    * Parseable XML whose root element localname is ``article`` → ``(False,
      True)`` (a JATS document; namespaced roots are handled by matching the
      localname). Any other root → ``(False, False)`` — well-formed XML that is
      not a JATS article.
    """
    corrupted, root_localname = _inspect_xml_root(file_content)
    return corrupted, root_localname == "article"


def _check_docx_encryption(file_content: bytes) -> bool:
    """Check if a DOCX file is password-protected.

    Encrypted OOXML documents are stored as an OLE Compound File with an
    ``EncryptedPackage`` stream rather than as a plain zip archive. Third-party
    tools instead produce an ordinary zip whose *members* are encrypted, which
    the CFB check cannot see — detected here from the entry's general-purpose
    flag so such a file is reported as password-protected rather than as
    generic corruption.

    Args:
        file_content: Raw DOCX bytes.

    Returns:
        True if the file is password-protected.
    """
    if file_content.startswith(_CFB_MAGIC):
        return b"EncryptedPackage" in file_content
    import io

    try:
        with zipfile.ZipFile(io.BytesIO(file_content)) as zf:
            return any(info.flag_bits & 0x1 for info in zf.infolist())
    except Exception:  # noqa: BLE001 — unreadable archives are the corruption check's job
        return False


def _check_pdf_encryption(file_content: bytes) -> bool:
    """Check if a PDF is encrypted by attempting to open it.

    Uses pypdfium2 as the ground truth — only PDFs that actually require
    a password to open are flagged as encrypted.  PDFs with restrictive
    permissions (e.g. "no printing") that open without a password are
    correctly treated as readable, and PDFs that merely mention the
    literal ``/Encrypt`` in body text or comments no longer trigger a
    false positive.

    The :data:`bibr.ocr.utils.pdfium_lock` is held for the document
    lifetime — PDFium's global C state is not thread-safe.

    Args:
        file_content: Raw PDF bytes.

    Returns:
        True if the file is password-protected.
    """
    try:
        import pypdfium2

        from bibr.ocr.utils import pdfium_lock

        try:
            import pypdfium2_raw

            password_code: int | None = pypdfium2_raw.FPDF_ERR_PASSWORD
        except ImportError:
            password_code = None

        with pdfium_lock:
            try:
                doc = pypdfium2.PdfDocument(file_content)
            except pypdfium2.PdfiumError as exc:
                # Primary detection: PDFium's numeric error code (FPDF_ERR_PASSWORD = 4).
                # pypdfium2 exposes this as `PdfiumError.err_code` since the helper API
                # was introduced; treat any other code as non-encryption (corruption,
                # format error, etc.) and let the corruption check surface those.
                err_code = getattr(exc, "err_code", None)
                if err_code is not None:
                    return bool(
                        err_code == password_code if password_code is not None else err_code == 4
                    )
                # Fallback for older pypdfium2 builds that don't annotate the code.
                msg = str(exc).lower()
                return "password" in msg or "encrypt" in msg
            try:
                doc.close()
            except Exception:  # noqa: S110
                pass
            return False
    except ImportError:
        # Fallback to the legacy tail scan only when pypdfium2 isn't installed.
        tail = file_content[-16384:] if len(file_content) > 16384 else file_content
        return b"/Encrypt" in tail


def validate_input_file(
    file_path: Path | str,
    file_content: bytes,
    file_hash: str | None = None,
) -> InputFile:
    """Validate an input file and return a fully populated InputFile.

    Performs the following checks in order:
    1. Extension-based support check (is_supported)
    2. MIME type detection from content (populates InputFormat.detected_mime_type)
    3. Cross-validation of extension vs MIME type
    4. Format-specific checks (PDF corruption/encryption)
    5. Sets is_valid = is_supported and not is_corrupted and not is_encrypted

    Args:
        file_path: Path to the file (used for extension and name).
        file_content: Raw bytes of the file.

    Returns:
        Fully populated InputFile with all validation flags set.

    Raises:
        InputValidationError: If the file is in a known-unsupported format
            (e.g., .exe, .zip) to fail fast with a clear message.
    """
    input_file = InputFile(path=file_path)
    input_file.file_hash = file_hash or hashlib.sha256(file_content).hexdigest()[:16]

    extension = input_file.file_extension.lower()

    # Fail fast for known-unsupported types
    if extension in UNSUPPORTED_EXTENSIONS:
        raise InputValidationError(f"File type '{extension}' is explicitly unsupported")

    # Check extension support
    input_file.is_supported = extension in SUPPORTED_EXTENSIONS

    # Detect MIME type from content
    detected_mime = detect_mime_type(file_content)

    # Resolve file type from extension
    file_type = "UNKNOWN"
    for ft in SupportedFileType:
        if ft.value == extension:
            file_type = ft.name
            break

    input_file.input_format = InputFormat(
        file_extension=extension,
        detected_mime_type=detected_mime,
        file_type=file_type,
    )

    # Cross-check extension vs MIME type
    if not _check_extension_mime_consistency(extension, detected_mime):
        logger.warning(
            f"Extension/MIME mismatch for {input_file.file_name}: "
            f"extension={extension}, detected_mime={detected_mime}"
        )
        # Reject when the content is an unambiguous document type the extension
        # misrepresents, rather than dispatching by the spoofable extension (L8).
        # Scoped to distinctive magic-byte formats — html<->xml overlap, so
        # those stay warn-only to avoid false rejects.
        if detected_mime in _STRICT_MISMATCH_MIMES:
            input_file.is_supported = False

    # If MIME indicates dangerous or unsupported content regardless of extension
    if detected_mime in EXECUTABLE_MIMES:
        logger.warning(
            f"MIME type {detected_mime} indicates executable content for {input_file.file_name}"
        )
        input_file.is_supported = False

    # Format-specific checks
    if extension == ".pdf":
        input_file.is_corrupted = _check_pdf_corruption(file_content)
        input_file.is_encrypted = _check_pdf_encryption(file_content)
    elif extension == ".docx":
        input_file.is_corrupted = _check_docx_corruption(file_content)
        input_file.is_encrypted = _check_docx_encryption(file_content)
    elif extension == ".xml":
        corrupted, root_localname = _inspect_xml_root(file_content)
        is_article = root_localname == "article"
        input_file.is_corrupted = corrupted
        # Well-formed XML whose root is not <article> is not something bibr can
        # process; fail fast with a clear message (mirrors the known-unsupported
        # extension guard above) rather than a generic "unsupported format".
        if not corrupted and not is_article:
            if root_localname == "TEI":
                message = "GROBID XML is not supported; only JATS XML is supported"
                logger.warning(message)
                raise InputValidationError(message)
            raise InputValidationError("XML input must be a JATS <article> document")
    elif extension in (".html", ".htm"):
        input_file.is_corrupted, input_file.native_artifact = _check_html_corruption(file_content)
    elif extension == ".epub":
        input_file.is_corrupted, input_file.native_artifact = _check_epub_corruption(file_content)

    # Derive is_valid
    input_file.is_valid = (
        input_file.is_supported and not input_file.is_corrupted and not input_file.is_encrypted
    )

    logger.debug(
        f"Validated {input_file.file_name}: "
        f"supported={input_file.is_supported}, valid={input_file.is_valid}, "
        f"mime={detected_mime}, type={file_type}"
    )

    return input_file
