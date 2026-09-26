"""FootnoteBuffer unit tests."""

from __future__ import annotations


def test_buffer_starts_empty():
    from bibr.structure.footnote_buffer import FootnoteBuffer

    fb = FootnoteBuffer()
    assert list(fb) == []


def test_record_adds_entry():
    from bibr.structure.footnote_buffer import FootnoteBuffer

    fb = FootnoteBuffer()
    fb.record(text="A footnote", page_number=2, body_section_id=5, deferred_text_index=10)
    assert list(fb) == [("A footnote", 2, 5, 10, True)]


def test_iteration_preserves_order():
    from bibr.structure.footnote_buffer import FootnoteBuffer

    fb = FootnoteBuffer()
    fb.record(text="A", page_number=1, body_section_id=1, deferred_text_index=10)
    fb.record(text="B", page_number=2, body_section_id=1, deferred_text_index=20)
    out = list(fb)
    assert out == [
        ("A", 1, 1, 10, True),
        ("B", 2, 1, 20, True),
    ]
    # Iteration does not consume — buffer still holds the records.
    assert list(fb) == out


def test_record_keeps_whether_the_note_was_ocr_output():
    """A note read from the PDF text layer is not OCR text; the late cleanup
    must leave it alone, so the record carries that to its sentence."""
    from bibr.structure.footnote_buffer import FootnoteBuffer

    fb = FootnoteBuffer()
    fb.record(
        text="1 Contact john_smith@uni.edu",
        page_number=1,
        body_section_id=1,
        deferred_text_index=0,
        from_ocr=False,
    )
    assert list(fb) == [("1 Contact john_smith@uni.edu", 1, 1, 0, False)]


def test_shift_moves_a_tie_only_when_captured_after_the_insert():
    """At the insert position, notes captured before the inserted entry stay put."""
    from bibr.structure.footnote_buffer import FootnoteBuffer

    fb = FootnoteBuffer()
    fb.record(text="A", page_number=1, body_section_id=1, deferred_text_index=4)
    fb.record(text="B", page_number=1, body_section_id=1, deferred_text_index=5)
    fb.record(text="C", page_number=2, body_section_id=1, deferred_text_index=5)
    fb.record(text="D", page_number=2, body_section_id=1, deferred_text_index=6)
    assert len(fb) == 4

    fb.shift_deferred_indices(5, ties_from=2)

    assert [record[3] for record in fb] == [4, 5, 6, 7]


def test_shift_without_capture_order_moves_every_tie():
    from bibr.structure.footnote_buffer import FootnoteBuffer

    fb = FootnoteBuffer()
    fb.record(text="A", page_number=1, body_section_id=1, deferred_text_index=5)

    fb.shift_deferred_indices(5)

    assert [record[3] for record in fb] == [6]
