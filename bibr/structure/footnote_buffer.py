"""Pending-footnote buffer.

Footnotes appear inline in OCR reading order but must be relocated to a
dedicated 'Footnote N' section after the main body. This buffer
accumulates them as they arrive and surfaces them in document order
during ``create_content_sections``.

Previously a single instance var on PDFParser; extracted here so the
record format is colocated with the buffering logic.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

# The marker a footnote is printed with: a number, or one of the conventional
# footnote symbols (optionally doubled, as in "††"). Anchored at the start and
# required to be followed by whitespace or a bracket so an opening word is
# never mistaken for a marker.
_PRINTED_MARKER_RE = re.compile(r"^\s*([0-9]{1,3}|[*†‡§¶#]{1,3})\s*[).\]]?\s+")


def printed_marker(text: str) -> str | None:
    """Extract the marker a footnote was printed with, if it carries one.

    Footnote xrefs used the buffer ordinal, which only coincides with the
    printed marker when every footnote in the paper was captured, in order,
    exactly once. Papers that restart numbering per page, use symbols, or lose
    one footnote to OCR then exported xref contents that match nothing a reader
    can see on the page.
    """
    match = _PRINTED_MARKER_RE.match(text)
    return match.group(1) if match else None


class FootnoteBuffer:
    """Buffer for footnote records pending relocation to a Footnotes section.

    Each record is a 4-tuple ``(text, page_number, body_section_id,
    deferred_text_index)`` — the index lets ``create_content_sections``
    find the nearest preceding sentence after segmentation populates real
    text_ids.
    """

    def __init__(self) -> None:
        self._records: list[tuple[str, int, int, int]] = []
        self._seen: set[tuple[str, int]] = set()

    def record(
        self,
        *,
        text: str,
        page_number: int,
        body_section_id: int,
        deferred_text_index: int,
    ) -> None:
        """Append a footnote record (deferred to create_content_sections).

        Identical text on the same page is dropped. Layout routinely emits the
        same page-bottom note twice — once per column, or once per overlapping
        region the resolver did not collapse — and each copy became its own
        "Footnote N" section and its own xref, inflating both and pushing every
        later footnote's number off by one. The same text on a *different*
        page is kept: repeating a note per page is a real convention.
        """
        key = (" ".join(text.split()), page_number)
        if key in self._seen:
            return
        self._seen.add(key)
        self._records.append((text, page_number, body_section_id, deferred_text_index))

    def is_empty(self) -> bool:
        return not self._records

    def __iter__(self) -> Iterator[tuple[str, int, int, int]]:
        """Iterate records in insertion order without consuming the buffer."""
        return iter(self._records)
