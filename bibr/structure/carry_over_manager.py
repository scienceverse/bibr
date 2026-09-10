"""Cross-page carry-over text manager.

When a paragraph spans the bottom of one page and the top of the next,
OCR emits two separate text regions. The carry-over manager buffers the
trailing fragment from page N until the head of page N+1 arrives, then
flushes them as a single sentence list.

C6 invariant: a heading region can arrive between the capture and the
flush. The captured ``section_id`` snapshot survives across that, so
flushed sentences are attributed to the section that was active when the
fragment was first appended — NOT the new section.

Previously four instance vars on PDFParser; bundled here so the C6
invariant is localised to one object. The flush logic stays on PDFParser
because it depends on ``_emit_sentences`` and ``_current_section_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.paper_contents import Provenance


class CarryOverState:
    """State holder for cross-page paragraph carry-over."""

    def __init__(self) -> None:
        self.text: str = ""
        # Page the buffered paragraph *started* on. Stamps the flushed entry
        # and seeds ``page_spans[0]``, so it must not advance on a join.
        self.page: int | None = None
        # Page of the most recently appended fragment. Distinct from
        # :attr:`page` once a paragraph has been joined across a break: it is
        # what a further region must be compared against to decide whether the
        # join is same-page (lenient) or cross-page (strict).
        self.last_page: int | None = None
        # Snapshot of the section_id that was active when the carry-over
        # was first captured. Preserved across heading transitions so the
        # flush attributes sentences to the original section (C6).
        self.section_id: int | None = None
        self.provenance: list[Provenance] = []
        # Region metadata from the first contributing layout region.
        # Preserved for v4 training feature export.
        self.region_meta: dict | None = None
        # ``(char_offset, page_no)`` per contributing page, ascending. Only
        # grows when a paragraph is joined across a page break, so each
        # flushed sentence can be attributed to the page it was printed on.
        self.page_spans: list[tuple[int, int]] = []

    def has_pending(self) -> bool:
        """True iff there is non-empty buffered text awaiting flush."""
        return bool(self.text.strip())

    def reset(self) -> None:
        """Clear all buffered state."""
        self.text = ""
        self.page = None
        self.last_page = None
        self.section_id = None
        self.provenance = []
        self.region_meta = None
        self.page_spans = []
