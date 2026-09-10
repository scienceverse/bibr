"""Validation coverage for native HTML/ePub inputs."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from bibr.input.validate import validate_input_file


def _make_epub_bytes() -> bytes:
    container_xml = b"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
    opf_xml = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest><item id="c1" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="c1"/></spine>
</package>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf_xml)
        zf.writestr("OEBPS/chapter.xhtml", "<html><body><p>Text.</p></body></html>")
    return buf.getvalue()


@pytest.mark.parametrize("name", ["paper.html", "paper.htm"])
def test_html_is_supported(name: str):
    result = validate_input_file(Path(name), b"<html><body><p>x</p></body></html>")

    assert result.is_valid
    assert result.input_format.file_type in {"HTML", "HTM"}


def test_empty_html_is_corrupted():
    result = validate_input_file(Path("paper.html"), b"<html><body></body></html>")

    assert result.is_supported
    assert result.is_corrupted
    assert not result.is_valid


def test_epub_is_supported():
    result = validate_input_file(Path("paper.epub"), _make_epub_bytes())

    assert result.is_valid
    assert result.input_format.file_type == "EPUB"


def test_epub_without_container_is_corrupted():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")

    result = validate_input_file(Path("paper.epub"), buf.getvalue())

    assert result.is_supported
    assert result.is_corrupted
    assert not result.is_valid


# --- Real-byte zip-member caps (audit M7/L10) --------------------------------


def _make_epub_with_chapter(chapter_body: str) -> bytes:
    container_xml = b"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
    opf_xml = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest><item id="c1" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="c1"/></spine>
</package>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf_xml)
        zf.writestr("OEBPS/chapter.xhtml", chapter_body)
    return buf.getvalue()


def test_epub_rejects_oversized_spine_member(monkeypatch):
    """A single huge spine document must not be decompressed fully into memory —
    a per-member real-byte cap rejects it (audit L10)."""
    from bibr.input import epub_native

    monkeypatch.setattr(epub_native, "_EPUB_MAX_MEMBER_BYTES", 2048)
    epub_bytes = _make_epub_with_chapter("<html><body>" + "x" * 8192 + "</body></html>")
    with pytest.raises(ValueError):
        epub_native.read_epub_document(epub_bytes)


def test_epub_allows_normal_spine_member(monkeypatch):
    from bibr.input import epub_native

    monkeypatch.setattr(epub_native, "_EPUB_MAX_MEMBER_BYTES", 1 << 20)
    epub_bytes = _make_epub_with_chapter("<html><body><p>Small chapter.</p></body></html>")
    doc = epub_native.read_epub_document(epub_bytes)
    assert b"Small chapter." in doc.html_bytes


def test_docx_rejects_real_oversized_document_xml(monkeypatch):
    """document.xml is enforced on its real decompressed size, not the declared
    (attacker-controlled) central-directory value (audit M7)."""
    from bibr.input import validate

    monkeypatch.setattr(validate, "_DOCX_DOCUMENT_XML_MAX_BYTES", 1024)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", b"<w:document>" + b"A" * 8192 + b"</w:document>")
    assert validate._check_docx_corruption(buf.getvalue()) is True
