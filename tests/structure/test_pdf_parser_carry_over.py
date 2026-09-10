"""Carry-over text must be attributed to its originating section.

Regression test for C6 (Plan 5, structure-correctness): when a section
change occurs between a carry-over text being captured and its later flush,
the flushed sentences must keep the section_id of the section they were
emitted under, not the new section that happens to be current at flush time.
"""

from bibr.structure.pdf_parser import PDFParser


def test_carry_over_attributed_to_originating_section():
    """A carry-over sentence flushed AFTER a section change must keep the
    section_id of the section it was emitted under, not the new section."""
    parser = PDFParser(json_result=[])

    # Section A (id=1)
    parser._current_section_id = 1
    # Emit a no-terminal-punct text -> enters carry-over capture.
    parser._handle_content(
        content="incomplete sentence without period",
        page_number=1,
        bbox=[0, 0, 1, 1],
    )
    # Sanity: carry-over is now armed under section 1.
    assert parser._carry_over.text == "incomplete sentence without period"

    # Now a section change happens (e.g. a heading region was processed).
    parser._current_section_id = 2

    # A new content arrives on a different page -> never joins across page
    # boundaries, so the carry-over is flushed and the new text is emitted.
    parser._handle_content(
        content="New sentence in section 2.",
        page_number=2,
        bbox=[0, 0, 1, 1],
    )

    # Inspect the deferred-text queue: each entry is
    # (text, page_number, section_id, needs_segmentation, is_display_formula).
    flushed = [entry for entry in parser._deferred_texts if "incomplete sentence" in entry[0]]
    assert flushed, "carry-over should have been flushed into _deferred_texts"
    assert flushed[0][2] == 1, f"expected section_id=1 (origin), got {flushed[0][2]}"

    # And the second emission is correctly attributed to section 2.
    new_entry = [entry for entry in parser._deferred_texts if "New sentence" in entry[0]]
    assert new_entry, "new sentence should have been emitted"
    assert new_entry[0][2] == 2, f"expected section_id=2 for new content, got {new_entry[0][2]}"


def test_carry_over_join_keeps_origin_section():
    """When a carry-over is *extended* (continuation join on the same page) and
    only later flushed after a section change, the merged text must still be
    attributed to the originating section."""
    parser = PDFParser(json_result=[])

    parser._current_section_id = 1
    parser._handle_content(
        content="incomplete sentence without period",
        page_number=1,
        bbox=[0, 0, 1, 1],
    )
    # Continuation: lowercase start + still no terminal punct -> joins carry-over.
    parser._handle_content(
        content="continues here without period",
        page_number=1,
        bbox=[0, 0, 1, 1],
    )
    assert "continues here" in parser._carry_over.text

    # Section change, then a cross-page text triggers flush (cross-page never joins).
    parser._current_section_id = 2
    parser._handle_content(
        content="New sentence in section 2.",
        page_number=2,
        bbox=[0, 0, 1, 1],
    )

    flushed = [entry for entry in parser._deferred_texts if "incomplete sentence" in entry[0]]
    assert flushed, "carry-over should have been flushed"
    assert flushed[0][2] == 1, (
        f"expected section_id=1 (origin) for joined carry-over, got {flushed[0][2]}"
    )


def test_carry_over_without_section_change_unaffected():
    """Sanity: when no section change occurs, the carry-over is still emitted
    under the (unchanged) current section."""
    parser = PDFParser(json_result=[])

    parser._current_section_id = 5
    parser._handle_content(
        content="incomplete sentence without period",
        page_number=1,
        bbox=[0, 0, 1, 1],
    )
    # Cross-page never joins -> the carry-over is flushed under section 5.
    parser._handle_content(
        content="New sentence stays in section 5.",
        page_number=2,
        bbox=[0, 0, 1, 1],
    )

    flushed = [entry for entry in parser._deferred_texts if "incomplete sentence" in entry[0]]
    assert flushed
    assert flushed[0][2] == 5


def test_carry_over_page_cursor_advances_after_a_cross_page_join():
    """After joining onto page N+1 the cursor must sit on N+1, not on N.

    Regression: ``CarryOverState.page`` was written only when a *fresh*
    carry-over was armed, never on the join branch. A third region printed on
    the new page therefore compared against the page the paragraph started on,
    saw ``same_page=False``, and fell under the stricter cross-page rule — so
    a capitalised continuation of the very same sentence was flushed apart.
    """
    parser = PDFParser(json_result=[])
    parser._current_section_id = 1

    # Bottom of page 1 -> armed as carry-over.
    parser._handle_content("The effect persisted across cohorts and", 1, bbox=[0, 0, 1, 1])
    # Top of page 2, lowercase -> cross-page join.
    parser._handle_content("remained stable under every model we", 2, bbox=[0, 0, 1, 1])

    assert parser._carry_over.page == 1, "the entry still starts on page 1"
    assert parser._carry_over.last_page == 2, "but the cursor has moved to page 2"

    # Third region, also page 2, capitalised (a cross-column break). With a
    # stale cursor this is judged cross-page and rejected by the lowercase
    # guard, splitting the sentence.
    parser._handle_content("Fit indices confirmed the result.", 2, bbox=[0, 0, 1, 1])
    parser._flush_carry_over()

    joined = [e.text for e in parser.assembler.entries if "effect persisted" in e.text]
    assert len(joined) == 1
    assert joined[0].endswith("Fit indices confirmed the result.")
    assert not [e for e in parser.assembler.entries if e.text.startswith("Fit indices")]


def test_cross_page_join_stamps_the_start_page_and_spans_both():
    """The cursor split must not disturb page attribution of the flushed entry."""
    parser = PDFParser(json_result=[])
    parser._current_section_id = 1

    parser._handle_content("The effect persisted across cohorts and", 1, bbox=[0, 0, 1, 1])
    parser._handle_content("remained stable throughout the trial.", 2, bbox=[0, 0, 1, 1])
    parser._flush_carry_over()

    entry = next(e for e in parser.assembler.entries if "effect persisted" in e.text)
    assert entry.page_number == 1
    assert entry.page_spans[0] == (0, 1)
    assert entry.page_for_offset(len(entry.text) - 1) == 2
