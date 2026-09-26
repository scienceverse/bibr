"""BIO tag definitions for the gold-trained ref parser.

Mirrors `bibr_training.tags` exactly so the trained checkpoint loads cleanly.
"""

from __future__ import annotations

FIELD_TYPES: list[str] = [
    "AUTHOR",
    "TITLE",
    "CONTAINER",
    "YEAR",
    "VOLUME",
    "ISSUE",
    "PAGES",
    "DOI",
    "URL",
    "ARXIV",
    "PMID",
    "PUBLISHER",
    "EDITOR",
    "EDITION",
    "SERIES",
    "ACCESS_DATE",
    "NOTE",
    "PAGE_RANGE_START",
    "PAGE_RANGE_END",
]

BIO_TAGS: list[str] = ["O"] + [
    f"{prefix}-{field}" for field in FIELD_TYPES for prefix in ("B", "I")
]

SEG_TAGS: list[str] = ["O", "B-REF", "I-REF"]
