"""Real-byte limits for zip-member decompression.

A zip central directory's declared ``file_size``/``compress_size`` are fully
attacker-controlled, so zip-bomb defenses that only inspect declared sizes trust
the attacker (audit M7/L10). These helpers decompress incrementally and enforce
the limit on the *actual* bytes produced, stopping as soon as the cap is crossed
so a bomb never fully materializes in memory.
"""

from __future__ import annotations

import zipfile

_CHUNK = 1 << 20  # 1 MiB


class ZipExpansionLimitError(Exception):
    """Raised when a zip member's real decompressed size exceeds the cap."""


def read_zip_member_capped(zf: zipfile.ZipFile, name: str, *, max_bytes: int) -> bytes:
    """Return ``name``'s bytes, decompressing at most ``max_bytes`` of real output.

    Raises :class:`ZipExpansionLimitError` if the member decompresses to more than
    ``max_bytes`` — measured on produced bytes, never on the declared metadata.
    """
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
    """
    total = 0
    with zf.open(name) as member:
        while chunk := member.read(_CHUNK):
            total += len(chunk)
            if total > max_bytes:
                return False
    return True
