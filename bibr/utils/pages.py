"""Page-range parsing shared by the CLI and the library API."""

from __future__ import annotations


def parse_pages(pages_str: str) -> tuple[int | None, int | None]:
    """Parse a page range string like '1-5' or '3' into (start, end) 0-indexed.

    Pages are 1-based on the user surface. Raises ``ValueError`` for
    0/negative pages (a 0 would silently wrap to the *last* page via negative
    indexing downstream) and for reversed ranges.
    """
    if "-" in pages_str:
        parts = pages_str.split("-", 1)
        start = int(parts[0])
        end = int(parts[1])
    else:
        start = end = int(pages_str)

    if start < 1 or end < 1:
        raise ValueError(f"pages are 1-based, got '{pages_str}'")
    if start > end:
        raise ValueError(f"range start exceeds end: '{pages_str}'")
    return start - 1, end - 1
