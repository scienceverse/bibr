"""libmagic is a system library, not something pip can supply.

``python-magic`` raises at import time when the shared library is missing, so a
fresh macOS/Linux install used to take down ``import bibr`` and every extraction
with a bare "failed to find libmagic" (issue #64). The binding is imported
defensively instead: the failure is deferred to first use and reported as a
named, fixable ``bibr doctor`` check.
"""

import pytest

from bibr.input import validate


@pytest.fixture
def missing_libmagic(monkeypatch):
    """Simulate a host whose system libmagic could not be loaded."""
    monkeypatch.setattr(validate, "magic", None)
    monkeypatch.setattr(validate, "_MAGIC_IMPORT_ERROR", ImportError("failed to find libmagic"))


def test_reason_is_none_when_libmagic_loads():
    assert validate.libmagic_unavailable_reason() is None


def test_reason_names_the_macos_install_command(monkeypatch, missing_libmagic):
    monkeypatch.setattr(validate.sys, "platform", "darwin")
    assert "brew install libmagic" in validate.libmagic_unavailable_reason()


def test_reason_names_the_linux_install_commands(monkeypatch, missing_libmagic):
    monkeypatch.setattr(validate.sys, "platform", "linux")
    reason = validate.libmagic_unavailable_reason()
    assert "libmagic1" in reason
    assert "file-libs" in reason


def test_reason_falls_back_on_an_unknown_platform(monkeypatch, missing_libmagic):
    monkeypatch.setattr(validate.sys, "platform", "sunos5")
    reason = validate.libmagic_unavailable_reason()
    assert "libmagic is not installed" in reason


def test_detect_mime_type_raises_the_actionable_message(monkeypatch, missing_libmagic):
    monkeypatch.setattr(validate.sys, "platform", "darwin")
    with pytest.raises(ImportError, match="brew install libmagic"):
        validate.detect_mime_type(b"%PDF-1.7")


def test_detect_mime_type_works_when_libmagic_is_present():
    assert validate.detect_mime_type(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3") == "application/pdf"


# --- the built-in sniff: validate keeps working without libmagic ---------------------


def _zip_bytes(entries: dict[str, bytes], *, stored: tuple[str, ...] = ()) -> bytes:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            kind = zipfile.ZIP_STORED if name in stored else zipfile.ZIP_DEFLATED
            archive.writestr(zipfile.ZipInfo(name), data, compress_type=kind)
    return buffer.getvalue()


_DOCX = _zip_bytes({"[Content_Types].xml": b"<Types/>", "word/document.xml": b"<w:document/>"})
_EPUB = _zip_bytes(
    {"mimetype": b"application/epub+zip", "META-INF/container.xml": b"<container/>"},
    stored=("mimetype",),
)
_SNIFF_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj", "application/pdf"),
        (b"\xef\xbb\xbf%PDF-1.4\n", "application/pdf"),  # a BOM before the header
        (_DOCX, _SNIFF_DOCX_MIME),
        (_EPUB, "application/epub+zip"),
        (_zip_bytes({"notes.txt": b"hello"}), "application/zip"),
        (b"<?xml version='1.0'?>\n<article/>", "text/xml"),
        (b"  \n<article><front/></article>", "text/xml"),
        (b"<!DOCTYPE html>\n<html><body>x</body></html>", "text/html"),
        (b"<HTML><BODY>x</BODY></HTML>", "text/html"),
        (b"\xef\xbb\xbf<html lang='en'></html>", "text/html"),
        (
            b"<?xml version='1.0'?><html xmlns='http://www.w3.org/1999/xhtml'/>",
            "application/xhtml+xml",
        ),
        (b"MZ\x90\x00\x03\x00", "application/x-msdownload"),
        (b"\x7fELF\x02\x01\x01", "application/x-executable"),
        (b"just some text", "application/octet-stream"),
        (b"", "application/octet-stream"),
        (b"PK\x03\x04 not really a zip", "application/zip"),
    ],
)
def test_sniff_names_the_formats_bibr_reads(content, expected):
    assert validate.sniff_mime_type(content) == expected


@pytest.mark.parametrize(
    ("content", "extension"),
    [(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3", ".pdf"), (_DOCX, ".docx"), (_EPUB, ".epub")],
)
def test_sniff_and_libmagic_accept_the_same_extension(content, extension):
    """Whatever libmagic names these, the sniff's answer passes the same extension cross-check."""
    if validate.libmagic_unavailable_reason() is not None:
        pytest.skip("libmagic is not installed here")
    check = validate._check_extension_mime_consistency
    assert check(extension, validate.detect_mime_type(content))
    assert check(extension, validate.sniff_mime_type(content))


@pytest.fixture
def no_libmagic_notice(monkeypatch, missing_libmagic):
    """Libmagic missing, with the once-per-process hint armed again."""
    monkeypatch.setattr(validate.sys, "platform", "linux")
    monkeypatch.setattr(validate, "_sniff_notice_shown", False)


def _sample_pdf() -> bytes:
    import importlib.resources as resources

    return resources.files("bibr.data").joinpath("sample_paper.pdf").read_bytes()


def test_validate_accepts_a_pdf_without_libmagic(no_libmagic_notice):
    result = validate.validate_input_file("paper.pdf", _sample_pdf())

    assert result.is_valid
    assert result.input_format.detected_mime_type == "application/pdf"


def test_validate_accepts_a_docx_without_libmagic(no_libmagic_notice):
    result = validate.validate_input_file("paper.docx", _DOCX)

    assert result.input_format.detected_mime_type == _SNIFF_DOCX_MIME
    assert result.is_supported


def test_validate_accepts_html_and_jats_xml_without_libmagic(no_libmagic_notice):
    page = validate.validate_input_file("paper.html", b"<!DOCTYPE html><html><body>x</body></html>")
    jats = validate.validate_input_file("paper.xml", b"<?xml version='1.0'?><article/>")

    assert page.input_format.detected_mime_type == "text/html"
    assert jats.input_format.detected_mime_type == "text/xml"
    assert page.is_supported
    assert jats.is_supported


def test_a_file_that_fails_the_sniff_is_rejected_as_it_is_with_libmagic(no_libmagic_notice):
    """A PDF-named file that is not a PDF is corrupt; a PDF under a DOCX name is refused."""
    not_a_pdf = validate.validate_input_file("paper.pdf", b"this is not a pdf at all")
    pdf_as_docx = validate.validate_input_file("paper.docx", _sample_pdf())
    exe_as_pdf = validate.validate_input_file("paper.pdf", b"MZ\x90\x00" + b"\x00" * 64)

    assert not not_a_pdf.is_valid
    assert not pdf_as_docx.is_supported
    assert not exe_as_pdf.is_valid
    assert not exe_as_pdf.is_supported


def test_validate_prints_the_libmagic_hint_once(no_libmagic_notice, caplog):
    with caplog.at_level("WARNING", logger=validate.logger.name):
        validate.validate_input_file("a.pdf", _sample_pdf())
        validate.validate_input_file("b.pdf", _sample_pdf())

    hints = [r for r in caplog.records if "libmagic1" in r.getMessage()]
    assert len(hints) == 1
    assert "sudo apt install libmagic1" in hints[0].getMessage()


def test_validate_says_nothing_about_libmagic_when_it_is_present(caplog):
    with caplog.at_level("WARNING", logger=validate.logger.name):
        validate.validate_input_file("a.pdf", _sample_pdf())

    assert not [r for r in caplog.records if "libmagic" in r.getMessage()]
