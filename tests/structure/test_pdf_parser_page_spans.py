"""Sentences of a cross-page paragraph retain their physical page.

A deferred paragraph can span several pages; assigning its starting page to every sentence loses occurrence provenance for later pages."""

import unicodedata

from bibr.structure.assembler import DeferredText
from bibr.structure.pdf_parser import PDFParser


def test_cross_page_join_records_a_page_span_for_the_continuation():
    parser = PDFParser(json_result=[])
    parser._current_section_id = 1

    # Bottom of page 2: no terminal punctuation -> armed as carry-over.
    parser._handle_content(
        content="The effect persisted across every cohort we examined and",
        page_number=2,
        bbox=[0, 0, 1, 1],
    )
    # Top of page 3: lowercase continuation -> joined into the same paragraph.
    parser._handle_content(
        content="remained stable throughout. A second finding appeared here.",
        page_number=3,
        bbox=[0, 0, 1, 1],
    )
    parser._flush_carry_over()

    entry = next(e for e in parser.assembler.entries if "effect persisted" in e.text)
    # The entry still starts on page 2 ...
    assert entry.page_number == 2
    # ... but it must record where page 3's contribution begins.
    assert entry.page_spans, "cross-page join must record page spans"
    assert entry.page_spans[0] == (0, 2)
    offset, page = entry.page_spans[1]
    assert page == 3
    assert entry.text[offset:].startswith("remained stable")


def test_page_for_offset_resolves_each_side_of_the_boundary():
    entry = DeferredText(
        text="alpha beta gamma",
        page_number=2,
        section_id=1,
        needs_segmentation=True,
        page_spans=[(0, 2), (11, 3)],
    )
    assert entry.page_for_offset(0) == 2
    assert entry.page_for_offset(10) == 2
    assert entry.page_for_offset(11) == 3
    assert entry.page_for_offset(15) == 3


def test_page_for_offset_falls_back_to_page_number_without_spans():
    entry = DeferredText(
        text="alpha beta",
        page_number=7,
        section_id=1,
        needs_segmentation=True,
    )
    assert entry.page_for_offset(0) == 7
    assert entry.page_for_offset(9) == 7


def test_emit_stamps_each_sentence_with_the_page_it_was_printed_on():
    """The end-to-end contract: segments after the break get the later page."""
    parser = PDFParser(json_result=[])
    parser._current_section_id = 1
    parser._handle_content(
        content="The effect persisted across every cohort we examined and",
        page_number=2,
        bbox=[0, 0, 1, 1],
    )
    parser._handle_content(
        content="remained stable throughout. A second finding appeared here.",
        page_number=3,
        bbox=[0, 0, 1, 1],
    )
    parser._flush_carry_over()

    entry = next(e for e in parser.assembler.entries if "effect persisted" in e.text)
    index = [e for e in parser.assembler.entries if e.needs_segmentation].index(entry)
    segments = [[] for _ in range(parser.assembler.segmentable_count)]
    # Split so the first sentence spans the page break and the second is wholly
    # on page 3 — mirroring how wtpsplit would segment the joined paragraph.
    first = "The effect persisted across every cohort we examined and remained stable throughout."
    segments[index] = [first, "A second finding appeared here."]

    sentences, _, _ = parser.assembler.emit(
        segments,
        sentence_factory=parser._make_sentence,
        sentence_counter=0,
        paragraph_counter=0,
    )
    by_text = {s.text: s.page_number for s in sentences}
    # The sentence that begins on page 2 keeps page 2 ...
    assert by_text[first] == 2
    # ... and the one printed wholly on page 3 is attributed to page 3.
    assert by_text["A second finding appeared here."] == 3


def test_page_spans_survive_nfc_normalization_length_change():
    """NFC can shorten text; span offsets must be re-measured, not carried raw."""
    from bibr.structure.assembler import DocumentAssembler

    assembler = DocumentAssembler()
    # "e" + U+0301 (combining acute) composes to a single "é" under NFC, so the
    # raw offset of the continuation is one greater than its normalized offset.
    raw = "cafe\u0301 start " + "continues here"  # decomposed: e + U+0301
    # Guard: this must actually shrink under NFC, else the test is vacuous.
    assert len(unicodedata.normalize("NFC", raw)) == len(raw) - 1
    boundary = raw.index("continues here")
    assembler.append(
        raw,
        2,
        1,
        needs_segmentation=True,
        page_spans=[(0, 2), (boundary, 3)],
    )
    entry = assembler.entries[0]
    assert entry.text[entry.page_spans[1][0] :].startswith("continues here")
    assert entry.page_for_offset(entry.page_spans[1][0]) == 3
    assert entry.page_for_offset(0) == 2
