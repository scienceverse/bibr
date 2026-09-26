"""Validation coverage for native HTML/ePub inputs."""

from __future__ import annotations

import io
import random
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


# --- Spine repetition and percent-encoded hrefs ------------------------------


def _make_epub_spine(*, repeats: int, href: str, member: str, body: str = "<p>Hi.</p>") -> bytes:
    container_xml = b"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
    itemrefs = "".join('<itemref idref="c1"/>' for _ in range(repeats))
    opf_xml = f"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest><item id="c1" href="{href}" media-type="application/xhtml+xml"/></manifest>
  <spine>{itemrefs}</spine>
</package>""".encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf_xml)
        zf.writestr(member, f"<html><body>{body}</body></html>")
    return buf.getvalue()


def test_epub_spine_repetition_does_not_multiply_member_caps():
    """Every zip limit is per member, so a spine repeating one 1 MiB chapter
    N times used to reach N MiB of resident text from a ~1 MB upload."""
    from bibr.input import epub_native

    # Poorly-compressible so the archive-wide ratio guard is not what rejects it.
    rng = random.Random(1234)  # noqa: S311 - deterministic test fixture, not crypto
    body = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789 ") for _ in range(1 << 20))
    epub_bytes = _make_epub_spine(
        repeats=300, href="chapter.xhtml", member="OEBPS/chapter.xhtml", body=body
    )
    assert len(epub_bytes) < 2 << 20
    doc = epub_native.read_epub_document(epub_bytes)
    # The repeated member is read once, not 300 times.
    assert len(doc.html_bytes) < 4 << 20


def test_epub_rejects_spine_longer_than_cap(monkeypatch):
    from bibr.input import epub_native

    monkeypatch.setattr(epub_native, "_EPUB_MAX_SPINE_DOCUMENTS", 4)
    epub_bytes = _make_epub_spine(repeats=5, href="chapter.xhtml", member="OEBPS/chapter.xhtml")
    with pytest.raises(ValueError):
        epub_native.read_epub_document(epub_bytes)


def test_epub_accepts_percent_encoded_manifest_href():
    """OPF hrefs are URL references, so a member name with a space is
    percent-encoded in the manifest but not in the zip directory."""
    from bibr.input import epub_native

    epub_bytes = _make_epub_spine(
        repeats=1,
        href="chapter%201.xhtml",
        member="OEBPS/chapter 1.xhtml",
        body="<p>Encoded chapter.</p>",
    )
    doc = epub_native.read_epub_document(epub_bytes)
    assert b"Encoded chapter." in doc.html_bytes


def test_docx_rejects_real_oversized_document_xml(monkeypatch):
    """document.xml is enforced on its real decompressed size, not the declared
    (attacker-controlled) central-directory value (audit M7)."""
    from bibr.input import validate

    monkeypatch.setattr(validate, "_DOCX_DOCUMENT_XML_MAX_BYTES", 1024)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", b"<w:document>" + b"A" * 8192 + b"</w:document>")
    assert validate._check_docx_corruption(buf.getvalue()) is True


# --- Missing spine members and prefixed DOI identifiers (input-parsers-28) ---


def _make_epub_two_chapters(*, with_ch2: bool, identifier: str) -> bytes:
    container_xml = b"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
    opf_xml = f"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Sample</dc:title>
    <dc:identifier>{identifier}</dc:identifier>
  </metadata>
  <manifest>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="ch1"/><itemref idref="ch2"/></spine>
</package>""".encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf_xml)
        zf.writestr("OEBPS/ch1.xhtml", "<html><body><p>Chapter one.</p></body></html>")
        if with_ch2:
            zf.writestr("OEBPS/ch2.xhtml", "<html><body><p>Chapter two.</p></body></html>")
    return buf.getvalue()


def test_epub_skips_missing_spine_member():
    """One manifest item absent from the zip must not reject the whole book —
    the readable chapters still validate. Fails on base (corrupted=True)."""
    from bibr.input.epub_native import EpubParser

    epub_bytes = _make_epub_two_chapters(with_ch2=False, identifier="10.1234/abc")
    result = validate_input_file(Path("paper.epub"), epub_bytes)
    assert result.is_corrupted is False
    assert result.is_valid is True
    # The lost chapter must be visible, not just a log line: the parse carries
    # a coded warning that reaches extraction.warnings in the export.
    contents = EpubParser(epub_bytes).parse()
    codes = [w.code for w in contents.processing_warnings]
    assert "EPUB_SPINE_MEMBER_SKIPPED" in codes
    assert any("OEBPS/ch2.xhtml" in w.message for w in contents.processing_warnings)


def _make_epub_two_chapters_with_bodies(*, ch1_body: str, ch2_body: str) -> bytes:
    container_xml = b"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
    opf_xml = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="ch1"/><itemref idref="ch2"/></spine>
</package>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf_xml)
        zf.writestr("OEBPS/ch1.xhtml", ch1_body)
        zf.writestr("OEBPS/ch2.xhtml", ch2_body)
    return buf.getvalue()


def test_epub_rejects_oversized_spine_member_alongside_readable_chapter(monkeypatch):
    """An over-cap spine member rejects the book even when another chapter is
    readable — only a *missing* member is skipped. Fails while the spine loop
    swallows the expansion-limit rejection."""
    from bibr.input import epub_native

    monkeypatch.setattr(epub_native, "_EPUB_MAX_MEMBER_BYTES", 2048)
    epub_bytes = _make_epub_two_chapters_with_bodies(
        ch1_body="<html><body><p>Small chapter.</p></body></html>",
        ch2_body="<html><body>" + "x" * 8192 + "</body></html>",
    )
    with pytest.raises(ValueError, match="exceeds 2048 bytes"):
        epub_native.read_epub_document(epub_bytes)
    result = validate_input_file(Path("paper.epub"), epub_bytes)
    assert result.is_corrupted is True
    assert result.is_valid is False


def test_epub_all_spine_members_missing_is_still_corrupted():
    """Guard: when nothing is readable the book is still rejected."""
    from bibr.input import epub_native

    buf = io.BytesIO()
    container_xml = b"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
    opf_xml = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest><item id="c1" href="gone.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="c1"/></spine>
</package>"""
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf_xml)
    with pytest.raises(ValueError, match="no readable spine"):
        epub_native.read_epub_document(buf.getvalue())
    result = validate_input_file(Path("paper.epub"), buf.getvalue())
    assert result.is_corrupted is True
    assert result.is_valid is False


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("10.1234/abc", "10.1234/abc"),
        ("doi:10.1234/abc", "10.1234/abc"),
        ("https://doi.org/10.1234/abc", "10.1234/abc"),
        ("urn:doi:10.1234/abc", "10.1234/abc"),
    ],
)
def test_epub_identifier_doi_forms(identifier: str, expected: str):
    """Packagers spell the DOI with urn:/doi:/URL prefixes — every form must
    land on the bare DOI. The urn: form fails on base (no doi key)."""
    from bibr.input import epub_native

    doc = epub_native.read_epub_document(
        _make_epub_two_chapters(with_ch2=True, identifier=identifier)
    )
    assert doc.metadata.get("doi") == expected
