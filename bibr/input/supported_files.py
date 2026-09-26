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


# Convenience sets for quick lookups
SUPPORTED_EXTENSIONS: set[str] = {ft.value for ft in SupportedFileType}
UNSUPPORTED_EXTENSIONS: set[str] = {ft.value for ft in UnsupportedFileType}
