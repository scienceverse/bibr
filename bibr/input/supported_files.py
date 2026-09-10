from enum import StrEnum


class SupportedFileType(StrEnum):
    """File types that bibr can process."""

    PDF = ".pdf"
    DOCX = ".docx"
    XML = ".xml"  # JATS XML (root <article>) — parsed natively, skips OCR
    HTML = ".html"  # Publisher/article HTML — parsed natively, skips OCR
    HTM = ".htm"
    EPUB = ".epub"  # ePub archive — XHTML spine parsed natively, skips OCR


class UnsupportedFileType(StrEnum):
    """Known file types that bibr explicitly refuses."""

    DOC = ".doc"  # legacy Word — convert to .docx
    EXE = ".exe"
    ZIP = ".zip"  # will eventually be needed if we want to process LaTeX, but it's not ideal
    TEX = ".tex"  # just .tex isn't enough for us to infer anything, we need dependencies


class SupportedMimeType(StrEnum):
    """Mime types that bibr can process."""

    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    XML = "application/xml"
    XML_TEXT = "text/xml"
    HTML = "text/html"
    XHTML = "application/xhtml+xml"
    EPUB = "application/epub+zip"


class UnsupportedMimeType(StrEnum):
    """Mime types that bibr explicitly refuses."""

    DOC = "application/msword"
    EXE = "application/x-msdownload"
    ZIP = "application/zip"  # Allowed for .docx/.epub by extension-specific validators.
    TEX = "text/x-tex"


# Convenience sets for quick lookups
SUPPORTED_EXTENSIONS: set[str] = {ft.value for ft in SupportedFileType}
UNSUPPORTED_EXTENSIONS: set[str] = {ft.value for ft in UnsupportedFileType}
SUPPORTED_MIME_TYPES: set[str] = {mt.value for mt in SupportedMimeType}
UNSUPPORTED_MIME_TYPES: set[str] = {mt.value for mt in UnsupportedMimeType}
