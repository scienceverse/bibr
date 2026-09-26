"""Tests for bibr.input.validate module."""

import io
from pathlib import Path
from unittest.mock import patch

import pytest

from bibr.exceptions import InputValidationError
from bibr.input.validate import (
    _check_extension_mime_consistency,
    _check_pdf_corruption,
    _check_pdf_encryption,
    validate_input_file,
)

# Minimal valid PDF bytes for testing
MINIMAL_PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 1\ntrailer\n<<>>\nstartxref\n0\n%%EOF"
CORRUPTED_PDF = b"This is not a PDF file at all"
PLAIN_TEXT = b"Hello, this is a plain text file."


class TestCheckPdfCorruption:
    def test_valid_pdf(self):
        assert _check_pdf_corruption(MINIMAL_PDF) is False

    def test_missing_header(self):
        assert _check_pdf_corruption(b"Not a PDF") is True

    def test_missing_eof(self):
        assert _check_pdf_corruption(b"%PDF-1.4\nsome content without eof marker") is True

    def test_empty_content(self):
        assert _check_pdf_corruption(b"") is True


def _build_pdf_with_encrypt_string_in_text() -> bytes:
    """Construct a valid (unencrypted) PDF whose body text contains `/Encrypt`."""
    pdfium = pytest.importorskip("pypdfium2")

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(72, 72)
    buf = io.BytesIO()
    pdf.save(buf)
    raw = buf.getvalue()
    needle = b"%%EOF"
    idx = raw.rfind(needle)
    return raw[:idx] + b"% /Encrypt mention in comment\n" + raw[idx:]


class TestCheckPdfEncryption:
    def test_plain_text(self):
        assert _check_pdf_encryption(PLAIN_TEXT) is False

    def test_open_access_pdf_with_encrypt_string_is_not_flagged(self):
        """PDFs that merely mention /Encrypt in a comment must not be flagged.

        The legacy tail byte-scan would false-positive on any PDF whose tail
        bytes contain the literal `/Encrypt` (e.g. open-access PDFs that
        embed the word in metadata or a comment).  The pypdfium2 ground-truth
        check only flags PDFs that actually require a password.
        """
        pytest.importorskip("pypdfium2")
        content = _build_pdf_with_encrypt_string_in_text()
        # Sanity check: the synthesized PDF really does carry the offending
        # bytes in its tail, so this would have tripped the old scan.
        assert b"/Encrypt" in content[-16384:]
        assert _check_pdf_encryption(content) is False

    def test_valid_unencrypted_pdf_not_flagged(self):
        """A freshly-built unencrypted PDF must not be flagged as encrypted."""
        pdfium = pytest.importorskip("pypdfium2")
        pdf = pdfium.PdfDocument.new()
        pdf.new_page(72, 72)
        buf = io.BytesIO()
        pdf.save(buf)
        assert _check_pdf_encryption(buf.getvalue()) is False

    def test_password_required_detected_via_err_code(self, monkeypatch):
        """Password-protected PDFs must be flagged via PdfiumError.err_code.

        Pins the contract that we read the numeric code (FPDF_ERR_PASSWORD = 4)
        rather than substring-matching the English error message — so a future
        wording change in pypdfium2 cannot silently regress this branch.
        """
        pdfium = pytest.importorskip("pypdfium2")
        raw = pytest.importorskip("pypdfium2_raw")

        def _raise_password(*_args, **_kwargs):
            raise pdfium.PdfiumError(
                "Failed to load document (PDFium: Incorrect password error).",
                err_code=raw.FPDF_ERR_PASSWORD,
            )

        monkeypatch.setattr(pdfium, "PdfDocument", _raise_password)
        assert _check_pdf_encryption(MINIMAL_PDF) is True

    def test_non_password_pdfium_error_not_flagged_as_encrypted(self, monkeypatch):
        """Non-password PdfiumError codes (corruption, format) must not be
        misreported as encryption — corruption is surfaced by its own check."""
        pdfium = pytest.importorskip("pypdfium2")
        raw = pytest.importorskip("pypdfium2_raw")

        def _raise_format(*_args, **_kwargs):
            raise pdfium.PdfiumError(
                "Failed to load document (PDFium: Format error).",
                err_code=raw.FPDF_ERR_FORMAT,
            )

        monkeypatch.setattr(pdfium, "PdfDocument", _raise_format)
        assert _check_pdf_encryption(MINIMAL_PDF) is False


def _minimal_encrypted_pdf_bytes() -> bytes:
    """A genuinely password-protected PDF: a bare trailer carrying a V1/R2
    ``/Encrypt`` dictionary, which PDFium refuses without a password
    (FPDF_ERR_PASSWORD) even though the bytes are otherwise well-formed."""
    owner = b"A" * 32
    user = b"B" * 32
    return (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
        b"trailer\n<< /Size 3 /Root 1 0 R /Encrypt << /Filter /Standard /V 1 /R 2 /O ("
        + owner
        + b") /U ("
        + user
        + b") /P -4 >> >>\n%%EOF"
    )


class TestPasswordProtectedPdfVerdicts:
    """A password error means encrypted, not corrupt — even with reader-tolerated
    framing bytes around the file. Both flags derive from the one shared open,
    so this pins the password branch of the single verdict."""

    def test_password_error_is_not_corruption(self):
        pytest.importorskip("pypdfium2")
        assert _check_pdf_corruption(_minimal_encrypted_pdf_bytes()) is False

    def test_password_error_is_encrypted(self):
        pytest.importorskip("pypdfium2")
        assert _check_pdf_encryption(_minimal_encrypted_pdf_bytes()) is True

    @patch("bibr.input.validate.detect_mime_type", return_value="application/pdf")
    def test_encrypted_pdf_with_bom_and_trailing_junk_validates_as_encrypted_only(self, mock_mime):
        pytest.importorskip("pypdfium2")
        blob = b"\xef\xbb\xbf" + _minimal_encrypted_pdf_bytes() + b" " * 12000
        result = validate_input_file(Path("/tmp/enc.pdf"), blob)
        assert result.is_corrupted is False
        assert result.is_encrypted is True
        assert result.is_valid is False


class TestExtensionMimeConsistency:
    def test_pdf_consistent(self):
        assert _check_extension_mime_consistency(".pdf", "application/pdf") is True

    def test_pdf_inconsistent(self):
        """A .pdf with DOCX MIME should be flagged as inconsistent."""
        assert (
            _check_extension_mime_consistency(
                ".pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            is False
        )

    def test_docx_consistent(self):
        assert (
            _check_extension_mime_consistency(
                ".docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            is True
        )

    def test_unknown_mime_passes(self):
        """Unknown MIME types are treated as consistent (permissive)."""
        assert _check_extension_mime_consistency(".pdf", "application/x-unknown") is True

    def test_removed_formats_no_mapping(self):
        """Removed formats have no MIME mapping, so consistency check passes (permissive)."""
        for ext in [".txt", ".md", ".html"]:
            assert _check_extension_mime_consistency(ext, "text/plain") is True


def _minimal_docx_bytes(*, include_document=True) -> bytes:
    """Build a minimal structurally-valid DOCX (zip with the two key parts)."""
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        if include_document:
            zf.writestr("word/document.xml", "<document/>")
    return buf.getvalue()


# OLE Compound File magic — what an encrypted (password-protected) Office
# file starts with. MS-CFB directory entry names are UTF-16LE on disk, so the
# fixture carries the EncryptedPackage stream name in that encoding (a plain
# ASCII embedding would pass a byte search no real file can satisfy).
_ENCRYPTED_DOCX = (
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    + b"\x00" * 56
    + "EncryptedPackage".encode("utf-16-le")
    + b"\x00" * 100
)


class TestCheckDocx:
    @patch("bibr.input.validate.detect_mime_type", return_value="application/octet-stream")
    def test_valid_docx_passes(self, mock_mime):
        result = validate_input_file(Path("/tmp/paper.docx"), _minimal_docx_bytes())
        assert result.is_corrupted is False
        assert result.is_encrypted is False
        assert result.is_valid is True

    @patch("bibr.input.validate.detect_mime_type", return_value="application/octet-stream")
    def test_truncated_docx_is_corrupted(self, mock_mime):
        truncated = _minimal_docx_bytes()[: len(_minimal_docx_bytes()) // 2]
        result = validate_input_file(Path("/tmp/paper.docx"), truncated)
        assert result.is_corrupted is True
        assert result.is_valid is False

    @patch("bibr.input.validate.detect_mime_type", return_value="text/plain")
    def test_non_zip_docx_is_corrupted(self, mock_mime):
        result = validate_input_file(Path("/tmp/paper.docx"), b"Not a zip archive at all")
        assert result.is_corrupted is True
        assert result.is_valid is False

    @patch("bibr.input.validate.detect_mime_type", return_value="application/octet-stream")
    def test_zip_without_document_xml_is_corrupted(self, mock_mime):
        """A zip that isn't a wordprocessing document (e.g. xlsx renamed)."""
        result = validate_input_file(
            Path("/tmp/paper.docx"), _minimal_docx_bytes(include_document=False)
        )
        assert result.is_corrupted is True
        assert result.is_valid is False

    @patch("bibr.input.validate.detect_mime_type", return_value="application/octet-stream")
    def test_encrypted_docx_detected(self, mock_mime):
        result = validate_input_file(Path("/tmp/paper.docx"), _ENCRYPTED_DOCX)
        assert result.is_encrypted is True
        assert result.is_valid is False

    @patch("bibr.input.validate.detect_mime_type", return_value="application/msword")
    def test_legacy_doc_renamed_to_docx_is_corrupted(self, mock_mime):
        """A binary .doc (CFB without EncryptedPackage) renamed to .docx."""
        legacy_doc = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 200
        result = validate_input_file(Path("/tmp/paper.docx"), legacy_doc)
        assert result.is_corrupted is True
        assert result.is_valid is False


def _deflated_docx_bytes() -> bytes:
    """A structurally-valid DOCX whose members are *deflated*, not stored.

    Every other DOCX fixture here is ZIP_STORED, so no test ever exercised the
    decompression path where a damaged archive actually fails.
    """
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", "<document><body>Some prose.</body></document>")
    return buf.getvalue()


def _break_deflate_stream(docx_bytes: bytes, name: str) -> bytes:
    """Make *name*'s deflate stream start with the reserved block type.

    A deflate block header of ``0b111`` (BFINAL set, BTYPE=11) is invalid by
    definition, so zlib rejects it deterministically instead of the test
    depending on random bytes happening to be undecodable.
    """
    import zipfile

    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
        info = zf.getinfo(name)
        assert info.compress_type == zipfile.ZIP_DEFLATED
    buf = bytearray(docx_bytes)
    off = info.header_offset
    name_len = int.from_bytes(buf[off + 26 : off + 28], "little")
    extra_len = int.from_bytes(buf[off + 28 : off + 30], "little")
    buf[off + 30 + name_len + extra_len] = 0x07
    return bytes(buf)


def _flag_members_encrypted(docx_bytes: bytes) -> bytes:
    """Set the zip encryption flag on every central-directory entry.

    Python cannot *write* an encrypted zip, and third-party tools that produce
    one leave the archive otherwise intact — which is the case this covers.
    Only the central directory is patched (located via the end-of-central-
    directory record) so a ``PK\x01\x02`` byte pair inside compressed data
    cannot be mistaken for a header.
    """
    buf = bytearray(docx_bytes)
    eocd = buf.rfind(b"PK\x05\x06")
    assert eocd != -1
    cd_start = int.from_bytes(buf[eocd + 16 : eocd + 20], "little")
    i = cd_start
    while (i := buf.find(b"PK\x01\x02", i, eocd)) != -1:
        buf[i + 8] |= 0x01
        i += 4
    return bytes(buf)


class TestDocxArchiveReadFailures:
    """A DOCX that cannot be *read* must be classified, not crash validation."""

    @patch("bibr.input.validate.detect_mime_type", return_value="application/octet-stream")
    def test_damaged_deflate_stream_is_corrupted(self, mock_mime):
        damaged = _break_deflate_stream(_deflated_docx_bytes(), "word/document.xml")
        result = validate_input_file(Path("/tmp/paper.docx"), damaged)
        assert result.is_corrupted is True
        assert result.is_valid is False

    @patch("bibr.input.validate.detect_mime_type", return_value="application/octet-stream")
    def test_zip_encrypted_docx_is_encrypted_not_corrupted(self, mock_mime):
        locked = _flag_members_encrypted(_deflated_docx_bytes())
        result = validate_input_file(Path("/tmp/paper.docx"), locked)
        assert result.is_encrypted is True
        assert result.is_corrupted is False
        assert result.is_valid is False

    @patch("bibr.input.validate.detect_mime_type", return_value="application/octet-stream")
    def test_a_readable_deflated_docx_still_passes(self, mock_mime):
        result = validate_input_file(Path("/tmp/paper.docx"), _deflated_docx_bytes())
        assert result.is_corrupted is False
        assert result.is_encrypted is False
        assert result.is_valid is True


class TestValidateInputFile:
    @patch("bibr.input.validate.detect_mime_type", return_value="application/pdf")
    def test_valid_pdf(self, mock_mime):
        result = validate_input_file(Path("/tmp/test.pdf"), MINIMAL_PDF)
        assert result.is_supported is True
        assert result.is_valid is True
        assert result.is_corrupted is False
        assert result.is_encrypted is False
        assert result.input_format is not None
        assert result.input_format.file_extension == ".pdf"
        assert result.input_format.file_type == "PDF"
        assert result.file_hash is not None

    def test_unsupported_extension_raises(self):
        with pytest.raises(InputValidationError, match="explicitly unsupported"):
            validate_input_file(Path("/tmp/malware.exe"), b"MZ\x90\x00")

    def test_zip_raises(self):
        with pytest.raises(InputValidationError, match="explicitly unsupported"):
            validate_input_file(Path("/tmp/archive.zip"), b"PK\x03\x04")

    @patch("bibr.input.validate.detect_mime_type", return_value="text/plain")
    def test_corrupted_pdf(self, mock_mime):
        result = validate_input_file(Path("/tmp/test.pdf"), CORRUPTED_PDF)
        assert result.is_corrupted is True
        assert result.is_valid is False

    @patch("bibr.input.validate._check_pdf_encryption", return_value=True)
    @patch("bibr.input.validate.detect_mime_type", return_value="application/pdf")
    def test_encrypted_pdf(self, mock_mime, mock_encryption):
        result = validate_input_file(Path("/tmp/test.pdf"), MINIMAL_PDF)
        assert result.is_encrypted is True
        assert result.is_valid is False

    @patch("bibr.input.validate.detect_mime_type", return_value="text/plain")
    def test_removed_formats_unsupported(self, mock_mime):
        """Markdown and plaintext remain unsupported generic text formats."""
        for path in ["/tmp/readme.md", "/tmp/notes.txt"]:
            result = validate_input_file(Path(path), PLAIN_TEXT)
            assert result.is_supported is False
            assert result.is_valid is False

    @patch("bibr.input.validate.detect_mime_type", return_value="text/plain")
    def test_plain_text_renamed_html_is_corrupted(self, mock_mime):
        result = validate_input_file(Path("/tmp/page.html"), PLAIN_TEXT)
        assert result.is_supported is True
        assert result.is_corrupted is True
        assert result.is_valid is False

    @patch("bibr.input.validate.detect_mime_type", return_value="application/octet-stream")
    def test_unknown_extension(self, mock_mime):
        result = validate_input_file(Path("/tmp/data.xyz"), b"some data")
        assert result.is_supported is False
        assert result.is_valid is False
        assert result.input_format.file_type == "UNKNOWN"

    @patch("bibr.input.validate.detect_mime_type", return_value="text/plain")
    def test_extension_mime_mismatch_logged(self, mock_mime):
        """A .pdf file with text/plain MIME should still be marked supported
        (extension takes precedence) but the mismatch is logged."""
        result = validate_input_file(Path("/tmp/test.pdf"), MINIMAL_PDF)
        assert result.is_supported is True
        # PDF corruption check will still run based on extension,
        # and the minimal PDF is valid
        assert result.is_corrupted is False

    @patch("bibr.input.validate.detect_mime_type", return_value="application/x-msdownload")
    def test_exe_mime_overrides_support(self, mock_mime):
        """If MIME says executable but extension looks supported, mark unsupported."""
        result = validate_input_file(Path("/tmp/test.pdf"), b"MZ\x90\x00")
        assert result.is_supported is False


# JATS XML input
_JATS_XML = b"""<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front><article-meta>
    <title-group><article-title>A JATS Paper</article-title></title-group>
  </article-meta></front>
  <body><sec><title>Intro</title><p>Text.</p></sec></body>
</article>"""
_NAMESPACED_JATS_XML = b"""<?xml version="1.0"?>
<article xmlns="https://jats.nlm.nih.gov/ns"><front/></article>"""
_GROBID_XML = b"""<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/></TEI>"""
_NON_ARTICLE_XML = b"""<?xml version="1.0"?><root><child>hello</child></root>"""
_MALFORMED_XML = b"""<?xml version="1.0"?><article><unclosed></article>"""


class TestCheckJatsArticle:
    def test_valid_jats_article(self):
        from bibr.input.validate import _check_jats_article

        corrupted, is_article = _check_jats_article(_JATS_XML)
        assert corrupted is False
        assert is_article is True

    def test_namespaced_root_recognized(self):
        from bibr.input.validate import _check_jats_article

        corrupted, is_article = _check_jats_article(_NAMESPACED_JATS_XML)
        assert corrupted is False
        assert is_article is True

    def test_non_article_root(self):
        from bibr.input.validate import _check_jats_article

        corrupted, is_article = _check_jats_article(_NON_ARTICLE_XML)
        assert corrupted is False
        assert is_article is False

    def test_malformed_is_corrupted(self):
        from bibr.input.validate import _check_jats_article

        corrupted, is_article = _check_jats_article(_MALFORMED_XML)
        assert corrupted is True
        assert is_article is False


class TestXmlValidation:
    @patch("bibr.input.validate.detect_mime_type", return_value="application/xml")
    def test_jats_xml_is_valid(self, mock_mime):
        result = validate_input_file(Path("/tmp/paper.xml"), _JATS_XML)
        assert result.is_supported is True
        assert result.is_corrupted is False
        assert result.is_valid is True
        assert result.input_format.file_type == "XML"

    @patch("bibr.input.validate.detect_mime_type", return_value="text/xml")
    def test_jats_text_xml_mime_is_valid(self, mock_mime):
        result = validate_input_file(Path("/tmp/paper.xml"), _JATS_XML)
        assert result.is_valid is True

    @patch("bibr.input.validate.detect_mime_type", return_value="application/xml")
    def test_malformed_xml_is_corrupted(self, mock_mime):
        result = validate_input_file(Path("/tmp/paper.xml"), _MALFORMED_XML)
        assert result.is_corrupted is True
        assert result.is_valid is False

    @patch("bibr.input.validate.detect_mime_type", return_value="application/xml")
    def test_non_article_xml_rejected_with_clear_message(self, mock_mime):
        with pytest.raises(InputValidationError, match="JATS <article>"):
            validate_input_file(Path("/tmp/other.xml"), _NON_ARTICLE_XML)

    @patch("bibr.input.validate.detect_mime_type", return_value="application/xml")
    def test_grobid_xml_warns_that_only_jats_is_supported(self, mock_mime, caplog):
        with (
            caplog.at_level("WARNING", logger="bibr.input.validate"),
            pytest.raises(InputValidationError, match="GROBID XML.*only JATS XML"),
        ):
            validate_input_file(Path("/tmp/grobid.xml"), _GROBID_XML)

        assert "GROBID XML is not supported; only JATS XML is supported" in caplog.messages


def test_pdf_content_under_xml_extension_is_rejected(tmp_path):
    """A file whose content is unambiguously a PDF but whose extension says .xml
    must not be dispatched by the (spoofable) extension — reject it (audit L8)."""
    from bibr.input.validate import validate_input_file

    pdf = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
    f = tmp_path / "spoof.xml"
    f.write_bytes(pdf)
    result = validate_input_file(f, pdf)
    assert result.is_supported is False
    assert result.is_valid is False


# Audit input-parsers-22: the EncryptedPackage stream name is UTF-16LE on disk.
def _cfb_bytes(name_payload: bytes) -> bytes:
    return b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 56 + name_payload + b"\x00" * 100


class TestEncryptedDocxNameEncoding:
    @patch("bibr.input.validate.detect_mime_type", return_value="application/octet-stream")
    def test_ascii_name_is_not_an_encrypted_package(self, mock_mime):
        """The pre-fix fixture embedded the name in ASCII, which no real CFB
        file contains — it must read as a legacy .doc (corrupted), not as an
        encrypted file. Fails on base, where ASCII matched."""
        result = validate_input_file(Path("/tmp/paper.docx"), _cfb_bytes(b"EncryptedPackage"))
        assert result.is_encrypted is False
        assert result.is_corrupted is True
        assert result.is_valid is False

    @patch("bibr.input.validate.detect_mime_type", return_value="application/octet-stream")
    def test_utf16le_name_is_encrypted(self, mock_mime):
        result = validate_input_file(
            Path("/tmp/paper.docx"),
            _cfb_bytes("EncryptedPackage".encode("utf-16-le")),
        )
        assert result.is_encrypted is True
        assert result.is_corrupted is False
        assert result.is_valid is False


# Audit input-parsers-23: PDFium opens files the byte heuristics reject.
def _real_pdf_bytes() -> bytes:
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(72, 72)
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


class TestPdfReaderTolerantCorruption:
    def test_leading_bom_before_header_is_not_corrupted(self):
        pytest.importorskip("pypdfium2")
        assert _check_pdf_corruption(b"\xef\xbb\xbf" + _real_pdf_bytes()) is False

    def test_trailing_junk_after_eof_is_not_corrupted(self):
        pytest.importorskip("pypdfium2")
        assert _check_pdf_corruption(_real_pdf_bytes() + b" " * 12000) is False

    def test_header_past_first_kibibyte_is_still_corrupted(self):
        pytest.importorskip("pypdfium2")
        blob = b"\x00" * 2000 + _real_pdf_bytes()
        assert _check_pdf_corruption(blob) is True


# Audit input-parsers-29: NCBI efetch <pmc-articleset> wrappers.
_PMC_SINGLE = (
    b'<pmc-articleset><article xmlns:xlink="http://www.w3.org/1999/xlink">'
    b"<front><article-meta><title-group><article-title>T</article-title></title-group>"
    b"</article-meta></front><body><p>Hi.</p></body></article></pmc-articleset>"
)
_PMC_MULTI = (
    b"<pmc-articleset><article><front/></article><article><front/></article></pmc-articleset>"
)


class TestPmcArticleset:
    def test_single_article_wrapper_is_jats(self):
        from bibr.input.validate import _check_jats_article

        corrupted, is_article = _check_jats_article(_PMC_SINGLE)
        assert (corrupted, is_article) == (False, True)

    @patch("bibr.input.validate.detect_mime_type", return_value="application/xml")
    def test_single_article_wrapper_validates(self, mock_mime):
        result = validate_input_file(Path("/tmp/PMC123.xml"), _PMC_SINGLE)
        assert result.is_corrupted is False
        assert result.is_valid is True

    def test_single_article_wrapper_parses(self):
        from bibr.input.jats_native import JatsParser

        contents = JatsParser(_PMC_SINGLE).parse()
        assert contents.preparsed_metadata.title == "T"

    @patch("bibr.input.validate.detect_mime_type", return_value="application/xml")
    def test_multi_article_set_rejected_with_count(self, mock_mime):
        with pytest.raises(InputValidationError, match="contains 2 <article>"):
            validate_input_file(Path("/tmp/PMC123.xml"), _PMC_MULTI)

    def test_multi_article_set_does_not_parse(self):
        from bibr.exceptions import ProcessingError
        from bibr.input.jats_native import JatsParser

        with pytest.raises(ProcessingError, match="contains 2 <article>"):
            JatsParser(_PMC_MULTI).parse()


# Audit input-parsers-30: a lying central-directory size must still be rejected
# (kept as defense in depth; CPython itself refuses the read with BadZipFile).
class TestDocxLyingDeclaredSize:
    def test_patched_central_directory_size_is_corrupted(self):
        import zipfile

        from bibr.input.validate import _check_docx_corruption

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<document>" + "x" * 5000 + "</document>")
        raw = bytearray(buf.getvalue())
        # Locate the central-directory header for word/document.xml and lie
        # about its uncompressed size (1000 instead of ~5021).
        lying = None
        blob = bytes(raw)
        pos = 0
        while (at := blob.find(b"\x50\x4b\x01\x02", pos)) >= 0:
            name_len = int.from_bytes(blob[at + 28 : at + 30], "little")
            name = blob[at + 46 : at + 46 + name_len]
            if name == b"word/document.xml":
                raw[at + 24 : at + 28] = (1000).to_bytes(4, "little")
                lying = bytes(raw)
                break
            pos = at + 1
        assert lying is not None
        with zipfile.ZipFile(io.BytesIO(lying)) as zf:
            with pytest.raises(zipfile.BadZipFile):
                zf.open("word/document.xml").read()
        assert _check_docx_corruption(lying) is True
