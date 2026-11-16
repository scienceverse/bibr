"""Tests for real-byte (not declared-metadata) zip member limits (audit M7/L10)."""

import io
import zipfile

import pytest

from bibr.input.zip_limits import (
    ZipExpansionLimitError,
    only_standard_methods,
    read_zip_member_capped,
    uncompressed_size_within,
)


def _zip_with(name: str, data: bytes) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, data)
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_reads_member_under_cap():
    zf = _zip_with("a.txt", b"hello world")
    assert read_zip_member_capped(zf, "a.txt", max_bytes=1024) == b"hello world"


def test_rejects_member_over_cap_on_real_bytes():
    # 1 MB of a single byte compresses to almost nothing (a zip bomb's trick),
    # but the *real* decompressed size must be what the cap sees, not the tiny
    # declared compress_size.
    payload = b"A" * (1024 * 1024)
    zf = _zip_with("big.xml", payload)
    with pytest.raises(ZipExpansionLimitError):
        read_zip_member_capped(zf, "big.xml", max_bytes=64 * 1024)


def test_does_not_decompress_beyond_cap():
    # Reading must stop at cap+1 bytes — never materialize the whole member.
    payload = b"B" * (5 * 1024 * 1024)
    zf = _zip_with("big.xml", payload)
    with pytest.raises(ZipExpansionLimitError):
        read_zip_member_capped(zf, "big.xml", max_bytes=1024)


def _bzip2_zip(name: str = "a.txt", data: bytes = b"hello") -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_BZIP2) as zf:
        zf.writestr(name, data)
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_only_standard_methods_accepts_stored_and_deflated():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("s", b"x", compress_type=zipfile.ZIP_STORED)
        zf.writestr("d", b"x", compress_type=zipfile.ZIP_DEFLATED)
    assert only_standard_methods(zipfile.ZipFile(buf))


def test_bzip2_member_is_refused_before_it_is_read():
    zf = _bzip2_zip()
    assert not only_standard_methods(zf)
    with pytest.raises(ZipExpansionLimitError):
        read_zip_member_capped(zf, "a.txt", max_bytes=1024)
    assert uncompressed_size_within(zf, "a.txt", max_bytes=1024) is False


def test_bzip2_docx_is_corrupted_and_not_sniffed():
    from bibr.input.sniff import ZIP_MIME, detect_mime_type
    from bibr.input.validate import _check_docx_corruption

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_BZIP2) as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", "<document/>")
    assert _check_docx_corruption(buf.getvalue()) is True
    assert detect_mime_type(buf.getvalue()) == ZIP_MIME


def test_bzip2_epub_is_not_sniffed_as_epub():
    from bibr.input.sniff import ZIP_MIME, detect_mime_type

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_BZIP2) as zf:
        zf.writestr("mimetype", "application/epub+zip")
    assert detect_mime_type(buf.getvalue()) == ZIP_MIME


def test_bzip2_epub_is_rejected():
    from bibr.input.epub_native import read_epub_document

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_BZIP2) as zf:
        zf.writestr("mimetype", "application/epub+zip")
    with pytest.raises(ValueError, match="expansion limits"):
        read_epub_document(buf.getvalue())
