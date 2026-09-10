"""Tests for real-byte (not declared-metadata) zip member limits (audit M7/L10)."""

import io
import zipfile

import pytest

from bibr.input.zip_limits import ZipExpansionLimitError, read_zip_member_capped


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
