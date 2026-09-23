"""Built-in content sniffing decides input types without libmagic.

Every fixture is built in memory; none of these tests need the system libmagic
library, and the validation tests run with the ``magic`` import disabled.
"""

import io
import struct
import zipfile
from pathlib import Path

import pytest

from bibr.input.sniff import (
    DOCX_MIME,
    ELF_EXECUTABLE_MIME,
    EPUB_MIME,
    HTML_MIME,
    PDF_MIME,
    WINDOWS_EXECUTABLE_MIME,
    XHTML_MIME,
    XML_MIME,
    ZIP_MIME,
    detect_mime_type,
    libmagic_available,
    sniff_mime_type,
)
from bibr.input.validate import validate_input_file

# --- Synthetic fixtures -------------------------------------------------------


def _zip(members, *, compression=zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compression) as zf:
        for name, data, *member_compression in members:
            zf.writestr(name, data, compress_type=(member_compression or [None])[0])
    return buf.getvalue()


def _pdf() -> bytes:
    return (
        b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
        b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
    )


def _docx() -> bytes:
    return _zip(
        [
            ("[Content_Types].xml", "<Types/>"),
            ("_rels/.rels", "<Relationships/>"),
            ("word/document.xml", "<w:document><w:body>Text.</w:body></w:document>"),
        ]
    )


def _epub(*, mimetype_first: bool = True) -> bytes:
    container = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
    package = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest><item id="c1" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="c1"/></spine>
</package>"""
    mimetype = ("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
    rest = [
        ("META-INF/container.xml", container),
        ("OEBPS/content.opf", package),
        ("OEBPS/chapter.xhtml", "<html><body><p>Text.</p></body></html>"),
    ]
    return _zip([mimetype, *rest] if mimetype_first else [*rest, mimetype])


def _plain_zip() -> bytes:
    return _zip([("notes.txt", "hello")])


_JATS = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Publishing DTD v1.3 20210610//EN"
  "JATS-journalpublishing1-3.dtd">
<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="research-article">
  <front><article-meta>
    <title-group><article-title>A Synthetic Paper</article-title></title-group>
  </article-meta></front>
  <body><sec><title>Intro</title><p>Text.</p></sec></body>
</article>"""
_HTML = b"""<!DOCTYPE html>
<html><head><title>A Synthetic Paper</title></head>
<body><article><h1>A Synthetic Paper</h1><p>Text.</p></article></body></html>"""
_TEXT = b"Plain notes about a paper, not a document format bibr reads.\n"


def _windows_executable() -> bytes:
    header = bytearray(b"MZ" + b"\x00" * 0x3E)
    header[0x3C:0x40] = struct.pack("<I", 0x40)
    return bytes(header) + b"PE\x00\x00" + b"\x4c\x01" + b"\x00" * 64


_ELF = b"\x7fELF\x02\x01\x01" + b"\x00" * 57


# --- Sniffing rules -----------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(_pdf(), PDF_MIME, id="pdf"),
        pytest.param(b"%PDF-", PDF_MIME, id="pdf-bare-header"),
        pytest.param(b"\xef\xbb\xbf%PDF-1.4\n%%EOF", PDF_MIME, id="pdf-after-bom"),
        pytest.param(b"\r\n%PDF-1.4\n%%EOF", PDF_MIME, id="pdf-after-newline"),
        pytest.param(b"HTTP junk\n" * 50 + b"%PDF-1.4\n%%EOF", PDF_MIME, id="pdf-after-junk"),
        pytest.param(_docx(), DOCX_MIME, id="docx"),
        pytest.param(
            _zip([("word/document.xml", "<w:document/>"), ("[Content_Types].xml", "<Types/>")]),
            DOCX_MIME,
            id="docx-members-reordered",
        ),
        pytest.param(_epub(), EPUB_MIME, id="epub"),
        pytest.param(_epub(mimetype_first=False), EPUB_MIME, id="epub-mimetype-not-first"),
        pytest.param(_plain_zip(), ZIP_MIME, id="zip"),
        pytest.param(
            _zip([("[Content_Types].xml", "<Types/>"), ("xl/workbook.xml", "<workbook/>")]),
            ZIP_MIME,
            id="spreadsheet-zip",
        ),
        pytest.param(_zip([]), ZIP_MIME, id="empty-zip"),
        pytest.param(_JATS, XML_MIME, id="jats"),
        pytest.param(b"<article><front/></article>", XML_MIME, id="jats-without-prolog"),
        pytest.param(b'<!DOCTYPE article SYSTEM "x.dtd">\n<article/>', XML_MIME, id="doctype"),
        pytest.param(b'\n\n<?xml version="1.0"?><TEI/>', XML_MIME, id="xml-leading-space"),
        pytest.param(b'<?xml-stylesheet href="a.xsl"?><article/>', XML_MIME, id="xml-pi"),
        pytest.param(b"<!-- saved -->\n<j:article xmlns:j='x'/>", XML_MIME, id="prefixed-root"),
        pytest.param(b"\xef\xbb\xbf" + _JATS, XML_MIME, id="xml-utf8-bom"),
        pytest.param('<?xml version="1.0"?><article/>'.encode("utf-16"), XML_MIME, id="utf16-bom"),
        pytest.param('<?xml version="1.0"?><article/>'.encode("utf-16-le"), XML_MIME, id="utf16"),
        pytest.param(_HTML, HTML_MIME, id="html5"),
        pytest.param(b"<!doctype html><html><body>x</body></html>", HTML_MIME, id="html-lower"),
        pytest.param(
            b'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">\n<html></html>',
            HTML_MIME,
            id="html4",
        ),
        pytest.param(b"<!-- saved page -->\n<html><body>x</body></html>", HTML_MIME, id="comment"),
        pytest.param(b"<body><p>x</p></body>", HTML_MIME, id="html-body-first"),
        pytest.param(b"<p>A paragraph.</p>", HTML_MIME, id="html-fragment"),
        pytest.param(
            b'<?xml version="1.0"?>\n<html xmlns="http://www.w3.org/1999/xhtml"><body/></html>',
            XHTML_MIME,
            id="xhtml",
        ),
        pytest.param(_windows_executable(), WINDOWS_EXECUTABLE_MIME, id="windows-executable"),
        pytest.param(_ELF, ELF_EXECUTABLE_MIME, id="elf-executable"),
        pytest.param(_TEXT, None, id="text"),
        pytest.param(b"", None, id="empty"),
        pytest.param(b"MZ\x90\x00", None, id="mz-without-pe-header"),
        pytest.param(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64, None, id="ole-container"),
        pytest.param(b"<widget/>", None, id="unknown-markup"),
        pytest.param(b"Version headers look like %PDF- in the spec.", None, id="pdf-mention"),
        pytest.param(b"x" * 2048 + b"%PDF-1.4\n%%EOF", None, id="pdf-header-past-first-kib"),
    ],
)
def test_sniff_mime_type(content, expected):
    assert sniff_mime_type(content) == expected


def test_markup_mentioning_the_pdf_header_stays_markup():
    assert sniff_mime_type(b"<html><body><p>Starts with %PDF-1.7.</p></body></html>") == HTML_MIME


def test_truncated_epub_is_recognized_from_its_leading_mimetype_member():
    epub = _epub()
    assert sniff_mime_type(epub[:80]) == EPUB_MIME


def test_truncated_docx_is_a_damaged_zip():
    docx = _docx()
    assert sniff_mime_type(docx[: len(docx) // 2]) == ZIP_MIME


def test_unterminated_prolog_is_scanned_in_linear_time():
    hostile = b"<!DOCTYPE x " + b"[]" * 4000 + b"<!--" + b"-" * 4000
    assert sniff_mime_type(hostile) is None


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (_TEXT, "text/plain"),
        (b"", "application/x-empty"),
        (b"\x00\x01\x02binary", "application/octet-stream"),
    ],
)
def test_detect_mime_type_names_unrecognized_content_without_libmagic(
    no_libmagic, content, expected
):
    assert libmagic_available() is False
    assert detect_mime_type(content) == expected


# --- Validation with python-magic unavailable ---------------------------------


@pytest.mark.parametrize(
    ("name", "content", "file_type"),
    [
        ("paper.pdf", _pdf(), "PDF"),
        ("paper.docx", _docx(), "DOCX"),
        ("paper.xml", _JATS, "XML"),
        ("paper.html", _HTML, "HTML"),
        ("paper.htm", _HTML, "HTM"),
        ("paper.epub", _epub(), "EPUB"),
    ],
)
def test_supported_formats_validate_without_libmagic(no_libmagic, name, content, file_type):
    result = validate_input_file(Path(name), content)

    assert result.is_valid is True
    assert result.is_supported is True
    assert result.input_format.file_type == file_type


# (name, content, is_supported, is_corrupted)
_MISNAMED = [
    pytest.param("paper.docx", _pdf(), False, True, id="pdf-named-docx"),
    pytest.param("paper.epub", _pdf(), False, True, id="pdf-named-epub"),
    pytest.param("paper.html", _pdf(), False, True, id="pdf-named-html"),
    pytest.param("paper.pdf", _docx(), False, True, id="docx-named-pdf"),
    pytest.param("paper.epub", _docx(), False, True, id="docx-named-epub"),
    pytest.param("paper.docx", _epub(), False, True, id="epub-named-docx"),
    pytest.param("paper.pdf", _epub(), False, True, id="epub-named-pdf"),
    pytest.param("paper.docx", _plain_zip(), True, True, id="plain-zip-named-docx"),
    pytest.param("paper.epub", _plain_zip(), True, True, id="plain-zip-named-epub"),
    pytest.param("paper.pdf", _TEXT, True, True, id="text-named-pdf"),
    pytest.param("paper.docx", _TEXT, True, True, id="text-named-docx"),
    pytest.param("paper.pdf", _windows_executable(), False, True, id="executable-named-pdf"),
    pytest.param("paper.html", _windows_executable(), False, True, id="executable-named-html"),
    pytest.param("paper.docx", _ELF, False, True, id="elf-named-docx"),
]


@pytest.mark.parametrize(("name", "content", "is_supported", "is_corrupted"), _MISNAMED)
def test_misnamed_files_are_rejected_without_libmagic(
    no_libmagic, name, content, is_supported, is_corrupted
):
    result = validate_input_file(Path(name), content)

    assert result.is_supported is is_supported
    assert result.is_corrupted is is_corrupted
    assert result.is_valid is False


def test_html_mentioning_the_pdf_header_is_accepted(no_libmagic):
    page = (
        b"<html><body><p>Every PDF file starts with %PDF-1.7 and ends with %%EOF.</p></body></html>"
    )

    result = validate_input_file(Path("page.html"), page)

    assert result.input_format.detected_mime_type == HTML_MIME
    assert result.is_valid is True


def test_pdf_with_a_misleading_extension_reports_its_real_type(no_libmagic):
    result = validate_input_file(Path("paper.docx"), _pdf())

    assert result.input_format.detected_mime_type == PDF_MIME
    assert result.input_format.file_type == "DOCX"


_OUTCOME_CASES = [
    ("paper.pdf", _pdf()),
    ("paper.docx", _docx()),
    ("paper.xml", _JATS),
    ("paper.html", _HTML),
    ("paper.epub", _epub()),
    *[(param.values[0], param.values[1]) for param in _MISNAMED],
    ("paper.html", b"<p>%PDF-1.4 mention</p>"),
    ("paper.pdf", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64),
    ("paper.html", b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"),
]


def _outcome(name, content):
    result = validate_input_file(Path(name), content)
    return result.is_supported, result.is_corrupted, result.is_encrypted, result.is_valid


def test_libmagic_never_changes_a_validation_outcome(monkeypatch):
    """With libmagic installed, every accept/reject matches a host without it."""
    import sys

    from bibr.input import sniff

    sniff._load_libmagic.cache_clear()
    if not libmagic_available():
        pytest.skip("libmagic is not installed on this host")
    with_libmagic = [_outcome(name, content) for name, content in _OUTCOME_CASES]

    monkeypatch.setitem(sys.modules, "magic", None)
    sniff._load_libmagic.cache_clear()
    try:
        without_libmagic = [_outcome(name, content) for name, content in _OUTCOME_CASES]
    finally:
        sniff._load_libmagic.cache_clear()

    assert with_libmagic == without_libmagic
