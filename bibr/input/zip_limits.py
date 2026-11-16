"""Real-byte limits for zip-member decompression.

CPython's ZipExtFile already enforces a member's declared ``file_size`` when
it is read — truncating the output and raising BadZipFile on the CRC mismatch
a lying header causes — so a bomb cannot slip through on declared sizes
alone for stored and deflated members. These helpers are defense in depth on
top of that: they decompress incrementally and enforce the limit on the
*actual* bytes produced, stopping as soon as the cap is crossed so the
rejection stays at validation, however the archive is read downstream.

That bound on ``read(n)`` is only dependable for stored and deflated members;
on some CPython builds a bzip2 or LZMA member with a lying declared size
inflates fully in memory. DOCX (OPC) and ePub (OCF) permit only those two
methods, so :func:`only_standard_methods` lets callers refuse any other
archive before reading a member, and the helpers here refuse a member
compressed any other way.
"""

from __future__ import annotations

import zipfile

_CHUNK = 1 << 20  # 1 MiB


class ZipExpansionLimitError(Exception):
    """Raised when a zip member's real decompressed size exceeds the cap."""


def only_standard_methods(zf: zipfile.ZipFile) -> bool:
    """True when every member is stored or deflated, the only methods OPC and OCF permit."""
    return all(
        info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) for info in zf.infolist()
    )


def _member_has_standard_method(zf: zipfile.ZipFile, name: str) -> bool:
    return zf.getinfo(name).compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)


def read_zip_member_capped(zf: zipfile.ZipFile, name: str, *, max_bytes: int) -> bytes:
    """Return ``name``'s bytes, decompressing at most ``max_bytes`` of real output.

    Raises :class:`ZipExpansionLimitError` if the member decompresses to more than
    ``max_bytes`` — measured on produced bytes, never on the declared metadata —
    or if the member is compressed with anything but stored or deflate, whose
    output ``read(n)`` cannot be relied on to bound.
    """
    if not _member_has_standard_method(zf, name):
        raise ZipExpansionLimitError(f"zip member {name!r} uses an unsupported compression method")
    with zf.open(name) as member:
        # Read one byte past the cap: if we get it, the member is over the limit.
        data = member.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ZipExpansionLimitError(
            f"zip member {name!r} exceeds {max_bytes} bytes when decompressed"
        )
    return data


def uncompressed_size_within(zf: zipfile.ZipFile, name: str, *, max_bytes: int) -> bool:
    """True if ``name`` decompresses to at most ``max_bytes`` real bytes.

    Streams the member and bails out early once the cap is crossed, so a bomb is
    rejected after producing only ``max_bytes`` + one chunk, not its full size.
    A member compressed with anything but stored or deflate is never opened: False.
    """
    if not _member_has_standard_method(zf, name):
        return False
    total = 0
    with zf.open(name) as member:
        while chunk := member.read(_CHUNK):
            total += len(chunk)
            if total > max_bytes:
                return False
    return True
