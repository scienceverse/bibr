"""Concurrency and zip-bomb hardening for input validation."""

import io
import struct
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from bibr.input.validate import _check_docx_corruption, _check_pdf_encryption, validate_input_file


class TestPdfEncryptionHoldsPdfiumLock:
    def test_pdfium_lock_held_across_open_and_close(self, monkeypatch):
        pypdfium2 = pytest.importorskip("pypdfium2")
        from bibr.ocr.utils import pdfium_lock

        seen: dict[str, bool] = {}

        class FakeDoc:
            def __init__(self, _data):
                seen["locked_on_open"] = pdfium_lock.locked()

            def close(self):
                seen["locked_on_close"] = pdfium_lock.locked()

        monkeypatch.setattr(pypdfium2, "PdfDocument", FakeDoc)
        assert _check_pdf_encryption(b"%PDF-1.4 fake") is False
        assert seen == {"locked_on_open": True, "locked_on_close": True}

    def test_lock_released_after_check(self, monkeypatch):
        pypdfium2 = pytest.importorskip("pypdfium2")
        from bibr.ocr.utils import pdfium_lock

        class FakeDoc:
            def __init__(self, _data):
                pass

            def close(self):
                pass

        monkeypatch.setattr(pypdfium2, "PdfDocument", FakeDoc)
        _check_pdf_encryption(b"%PDF-1.4 fake")
        assert pdfium_lock.locked() is False

    def test_lock_released_on_pdfium_error(self, monkeypatch):
        pypdfium2 = pytest.importorskip("pypdfium2")
        from bibr.ocr.utils import pdfium_lock

        def boom(_data):
            raise pypdfium2.PdfiumError("Failed to load document (PDFium: Incorrect password).")

        monkeypatch.setattr(pypdfium2, "PdfDocument", boom)
        _check_pdf_encryption(b"%PDF-1.4 fake")
        assert pdfium_lock.locked() is False


def _docx_with_forged_sizes(
    entries: dict[str, bytes],
    forge: dict[int, int],
) -> bytes:
    """Build a zip and rewrite size fields so entries claim huge uncompressed sizes.

    ``forge`` maps a real payload length to the size the headers should claim.
    Entries are stored (no compression), so the 4-byte little-endian length
    appears verbatim in the local and central-directory headers.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    raw = buf.getvalue()
    for real, fake in forge.items():
        raw = raw.replace(struct.pack("<I", real), struct.pack("<I", fake))
    return raw


class TestDocxZipBomb:
    _DOC_PAYLOAD = b"<document/>" + b"a" * 79214  # 79225 bytes, distinctive packed form
    _MEDIA_PAYLOAD = b"b" * 79333

    def test_document_xml_over_ceiling_is_corrupted(self):
        raw = _docx_with_forged_sizes(
            {"[Content_Types].xml": b"<Types/>", "word/document.xml": self._DOC_PAYLOAD},
            {len(self._DOC_PAYLOAD): 600 * 1024 * 1024},
        )
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            assert zf.getinfo("word/document.xml").file_size == 600 * 1024 * 1024
        assert _check_docx_corruption(raw) is True

    def test_total_uncompressed_over_ceiling_is_corrupted(self):
        raw = _docx_with_forged_sizes(
            {
                "[Content_Types].xml": b"<Types/>",
                "word/document.xml": b"<document/>",
                "word/media/img1.bin": self._MEDIA_PAYLOAD,
                "word/media/img2.bin": self._MEDIA_PAYLOAD,
                "word/media/img3.bin": self._MEDIA_PAYLOAD,
            },
            {len(self._MEDIA_PAYLOAD): 400 * 1024 * 1024},
        )
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            total = sum(info.file_size for info in zf.infolist())
        assert total > 1024 * 1024 * 1024
        assert _check_docx_corruption(raw) is True

    def test_normal_docx_under_ceilings_passes(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<document/>")
        assert _check_docx_corruption(buf.getvalue()) is False

    def test_extreme_compression_ratio_is_corrupted(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", b"a" * (2 * 1024 * 1024))
        assert _check_docx_corruption(buf.getvalue()) is True

    def test_excessive_entry_count_is_corrupted(self, monkeypatch):
        import bibr.input.validate as mod

        monkeypatch.setattr(mod, "_DOCX_MAX_ENTRIES", 3)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<document/>")
            zf.writestr("word/styles.xml", "<styles/>")
            zf.writestr("word/settings.xml", "<settings/>")
        assert _check_docx_corruption(buf.getvalue()) is True

    @patch("bibr.input.validate.detect_mime_type", return_value="application/octet-stream")
    def test_zip_bomb_docx_is_invalid_input(self, _mock_mime):
        raw = _docx_with_forged_sizes(
            {"[Content_Types].xml": b"<Types/>", "word/document.xml": self._DOC_PAYLOAD},
            {len(self._DOC_PAYLOAD): 600 * 1024 * 1024},
        )
        result = validate_input_file(Path("/tmp/bomb.docx"), raw)
        assert result.is_corrupted is True
        assert result.is_valid is False
