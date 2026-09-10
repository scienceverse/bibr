"""Shared deferred-text buffer + sentence-emission driver.

Both :class:`bibr.structure.pdf_parser.PDFParser` and
:class:`bibr.input.docx_native.DocxParser` collect body text during parse and
defer sentence segmentation to a later stage (``apply_segmentation``), so the
segmenter can run once over a whole batch. Historically each parser hand-rolled
this machinery: a list of anonymous ``(text, page, section_id, needs_seg,
is_formula)`` 5-tuples plus, on the PDF side, two *parallel* arrays
(``_deferred_provenance`` / ``_deferred_region_meta``) that had to be padded to
length defensively in case an append site forgot one of them.

:class:`DocumentAssembler` replaces that with a single typed
:class:`DeferredText` entry that carries its own optional side-channels
(provenance, region_meta) inline — so misalignment is structurally impossible
and the padding logic is gone. It also owns the shared emission loop:
segmentable counting, iterating entries, splitting segmented vs pass-through
text, ``text_id``/``paragraph_id`` bookkeeping, and the per-entry
``last_text_id`` trail used for footnote xref linking.

Genuinely format-specific behaviour stays in each parser and is injected:

* the ``sentence_factory`` builds the :class:`PaperSentence` (PDF attaches
  provenance/region_meta and marks display formulas; DOCX does neither),
* the optional ``on_sentence`` hook runs per emitted sentence (PDF detects URLs
  inline; DOCX defers URL detection to a second pass so hyperlink captures can
  guard against duplicates),
* footnote/hyperlink resolution keyed by deferred index reads
  :attr:`DocumentAssembler.last_text_id` but stays in each parser because the
  index conventions differ.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bibr.paper_contents import PaperSentence, Provenance

from bibr.utils.text import normalize_unicode


@dataclass
class DeferredText:
    """One deferred paragraph awaiting sentence segmentation.

    ``provenance`` and ``region_meta`` are optional side-channels used by the
    PDF path (source bboxes + training region metadata); the DOCX path leaves
    them at their empty defaults. Carrying them on the entry keeps them aligned
    with the text by construction — there is no separate parallel array to pad.
    """

    text: str
    page_number: int | None
    section_id: int
    needs_segmentation: bool
    is_formula: bool = False
    provenance: list[Provenance] = field(default_factory=list)
    region_meta: dict | None = None
    # ``(char_offset, page_no)`` marks where each contributing page's text
    # begins within :attr:`text`, ascending by offset. Only populated when a
    # paragraph was joined across a page break; a single-page entry leaves it
    # empty and is described entirely by :attr:`page_number`.
    page_spans: list[tuple[int, int]] = field(default_factory=list)

    def page_for_offset(self, offset: int) -> int | None:
        """Page on which the text at ``offset`` was printed.

        Falls back to :attr:`page_number` when the entry carries no spans,
        which is every single-page entry and every page-less input (DOCX,
        JATS, HTML).
        """
        page = self.page_number
        for span_offset, span_page in self.page_spans:
            if span_offset > offset:
                break
            page = span_page
        return page


class DocumentAssembler:
    """Owns the deferred-text buffer and drives sentence emission.

    Parsers append entries during :meth:`parse`, hand the segmentable texts to
    an external segmenter, then call :meth:`emit` with the resulting segment
    lists to produce :class:`PaperSentence` rows.
    """

    def __init__(self) -> None:
        self.entries: list[DeferredText] = []
        # Last emitted text_id per deferred entry (``None`` when an entry
        # produced no sentence). Rebuilt by :meth:`emit`; initialised empty so
        # footnote resolution that runs without a prior :meth:`emit` (nothing
        # was deferred) still has a list to consult.
        self.last_text_id: list[int | None] = []

    # ------------------------------------------------------------------
    # Buffer building
    # ------------------------------------------------------------------

    def append(
        self,
        text: str,
        page_number: int | None,
        section_id: int,
        needs_segmentation: bool,
        is_formula: bool = False,
        *,
        provenance: list[Provenance] | None = None,
        region_meta: dict | None = None,
        page_spans: list[tuple[int, int]] | None = None,
    ) -> int:
        """Append a deferred entry and return its index in the buffer."""
        normalized = normalize_unicode(text)
        if page_spans and normalized != text:
            # Offsets were measured on the raw text; NFC can change length
            # (``e`` + combining acute -> ``é``), so re-measure each boundary
            # through the same normalization to keep the spans aligned.
            page_spans = [
                (len(normalize_unicode(text[:offset])), page) for offset, page in page_spans
            ]
        self.entries.append(
            DeferredText(
                text=normalized,
                page_number=page_number,
                section_id=section_id,
                needs_segmentation=needs_segmentation,
                is_formula=is_formula,
                provenance=list(provenance) if provenance else [],
                region_meta=region_meta,
                page_spans=list(page_spans) if page_spans else [],
            )
        )
        return len(self.entries) - 1

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[DeferredText]:
        return iter(self.entries)

    @property
    def segmentable_texts(self) -> list[str]:
        """Texts (in order) for entries that need sentence segmentation."""
        return [e.text for e in self.entries if e.needs_segmentation]

    @property
    def segmentable_count(self) -> int:
        """Number of entries that need sentence segmentation."""
        return sum(1 for e in self.entries if e.needs_segmentation)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit(
        self,
        all_segments: list[list[str]],
        *,
        sentence_factory: Callable[[DeferredText, str, int, int], Any],
        sentence_counter: int,
        paragraph_counter: int,
        on_sentence: Callable[[Any], None] | None = None,
    ) -> tuple[list[PaperSentence], int, int]:
        """Turn deferred entries into sentences via *sentence_factory*.

        *all_segments* must have one list per segmentable entry (validated).
        Each segmentable entry consumes the next segment list; non-segmentable
        entries (formulas, reference lines) pass through as a single sentence.

        ``sentence_factory(entry, text, text_id, paragraph_id)`` builds each
        :class:`PaperSentence` (parser-specific fields live there). Every entry
        advances ``paragraph_counter`` — even one that yields no sentence — to
        preserve historical paragraph numbering. ``on_sentence`` runs once per
        emitted sentence when provided.

        Populates :attr:`last_text_id` (one slot per entry). Returns
        ``(sentences, next_sentence_counter, next_paragraph_counter)`` so the
        caller can resume its counters for later content sections.
        """
        if len(all_segments) != self.segmentable_count:
            raise ValueError(
                f"Expected {self.segmentable_count} segment lists, got {len(all_segments)}"
            )

        seg_iter = iter(all_segments)
        self.last_text_id = []
        sentences: list[PaperSentence] = []

        for entry in self.entries:
            paragraph_counter += 1
            paragraph_id = paragraph_counter
            last_tid: int | None = None

            if entry.needs_segmentation:
                # Walk the joined text so each segment is attributed to the page
                # it was printed on, not the page the paragraph started on.
                cursor = 0
                for segment in next(seg_iter):
                    segment = segment.strip()
                    if not segment:
                        continue
                    segment_entry = entry
                    if entry.page_spans:
                        found = entry.text.find(segment, cursor)
                        if found >= 0:
                            cursor = found + len(segment)
                        page = entry.page_for_offset(found if found >= 0 else cursor)
                        if page != entry.page_number:
                            segment_entry = replace(entry, page_number=page)
                    sent = sentence_factory(segment_entry, segment, sentence_counter, paragraph_id)
                    sentences.append(sent)
                    if on_sentence is not None:
                        on_sentence(sent)
                    last_tid = sentence_counter
                    sentence_counter += 1
            else:
                sent = sentence_factory(entry, entry.text, sentence_counter, paragraph_id)
                sentences.append(sent)
                if on_sentence is not None:
                    on_sentence(sent)
                last_tid = sentence_counter
                sentence_counter += 1

            self.last_text_id.append(last_tid)

        return sentences, sentence_counter, paragraph_counter

    @staticmethod
    def build_sections_text(sentences: list[PaperSentence]) -> dict[int, str]:
        """Join sentence texts per ``section_id`` into a section-text map."""
        grouped: dict[int, list[str]] = {}
        for sent in sentences:
            grouped.setdefault(sent.section_id, []).append(sent.text)
        return {k: " ".join(v) for k, v in grouped.items()}
