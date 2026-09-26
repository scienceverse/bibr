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


def _cross_page_pages(extra_page1=(), extra_page2=()):
    first = "Participants were recruited from the local university and they"
    return [
        [
            {"label": "paragraph_title", "content": "Method", "bbox_2d": [100, 100, 900, 130]},
            {"label": "text", "content": first, "bbox_2d": [100, 400, 900, 450]},
            *extra_page1,
        ],
        [
            *extra_page2,
            {
                "label": "text",
                "content": "completed the survey online.",
                "bbox_2d": [100, 200, 900, 250],
            },
        ],
    ]


def test_footnote_between_halves_does_not_split_paragraph():
    """A footnote at the bottom of page N must not flush the carry-over."""
    pages = _cross_page_pages(
        extra_page1=[
            {
                "label": "footnote",
                "content": "1 We thank the reviewers.",
                "bbox_2d": [100, 850, 900, 900],
            }
        ]
    )
    parser = PDFParser(pages)
    parser.parse()

    assert [e.text for e in parser.assembler.entries] == [
        "Participants were recruited from the local university and they "
        "completed the survey online."
    ]


def test_figure_between_halves_does_not_split_paragraph():
    """A figure at the top of page N+1 must not flush the carry-over."""
    pages = _cross_page_pages(
        extra_page2=[{"label": "image", "content": "", "bbox_2d": [100, 60, 900, 120]}]
    )
    parser = PDFParser(pages)
    parser.parse()

    assert [e.text for e in parser.assembler.entries] == [
        "Participants were recruited from the local university and they "
        "completed the survey online."
    ]


def test_publisher_noise_heading_between_halves_does_not_split_paragraph():
    """A heading later dropped as publisher noise must not flush the carry-over."""
    pages = _cross_page_pages(
        extra_page2=[
            {"label": "paragraph_title", "content": "Wiley", "bbox_2d": [100, 60, 900, 90]}
        ]
    )
    parser = PDFParser(pages)
    parser.parse()

    assert [e.text for e in parser.assembler.entries] == [
        "Participants were recruited from the local university and they "
        "completed the survey online."
    ]


def test_real_heading_between_halves_still_splits_paragraph():
    """Guard: a real section heading still ends the in-flight paragraph."""
    pages = _cross_page_pages(
        extra_page2=[
            {"label": "paragraph_title", "content": "Results", "bbox_2d": [100, 60, 900, 90]}
        ]
    )
    parser = PDFParser(pages)
    parser.parse()

    assert [e.text for e in parser.assembler.entries] == [
        "Participants were recruited from the local university and they",
        "completed the survey online.",
    ]


def test_footnote_xref_anchors_to_joined_paragraph():
    """The footnote xref must land on the joined paragraph's sentence."""
    pages = _cross_page_pages(
        extra_page1=[
            {
                "label": "footnote",
                "content": "1 We thank the reviewers.",
                "bbox_2d": [100, 850, 900, 900],
            }
        ]
    )
    parser = PDFParser(pages)
    contents = parser.parse()
    parser.apply_segmentation(
        contents, [[entry.text] for entry in parser.assembler.entries if entry.needs_segmentation]
    )
    parser.create_content_sections(contents)

    assert [record[3] for record in parser._footnotes] == [1]
    foot_xrefs = [x for x in contents.xrefs if x.xref_type == "foot"]
    assert len(foot_xrefs) == 1
    body_sentence = next(s for s in contents.sentences if "completed the survey" in s.text)
    assert foot_xrefs[0].text_id == body_sentence.text_id


def _two_region_pages(first):
    return [
        [
            {"label": "paragraph_title", "content": "Method", "bbox_2d": [100, 100, 900, 130]},
            {"label": "text", "content": first, "bbox_2d": [100, 200, 900, 250]},
            {
                "label": "text",
                "content": "A second, separate paragraph begins here.",
                "bbox_2d": [100, 300, 900, 350],
            },
        ]
    ]


def test_quoted_ending_starts_new_paragraph():
    """A paragraph ending in a closing quote is terminal for both quote styles."""
    for first in [
        'As one participant put it, "I never trusted the study."',
        "As one participant put it, \u201cI never trusted the study.\u201d",
    ]:
        parser = PDFParser(_two_region_pages(first))
        parser.parse()
        assert [e.text for e in parser.assembler.entries] == [
            first,
            "A second, separate paragraph begins here.",
        ]


def test_footnote_marker_ending_starts_new_paragraph():
    """A period plus footnote marker ($^{1}$ or ¹) ends the paragraph."""
    for first in [
        "Sleep supports consolidation.$^{1}$",
        "Sleep supports consolidation.\u00b9",
    ]:
        parser = PDFParser(_two_region_pages(first))
        parser.parse()
        assert [e.text for e in parser.assembler.entries] == [
            first,
            "A second, separate paragraph begins here.",
        ]


def test_trailing_url_starts_new_paragraph():
    """A region ending in a bare URL is a complete stop, not a wrap."""
    first = "Data are available at https://osf.io/abc12"
    parser = PDFParser(_two_region_pages(first))
    parser.parse()
    assert [e.text for e in parser.assembler.entries] == [
        first,
        "A second, separate paragraph begins here.",
    ]


def test_url_wrap_hyphen_still_joins():
    """Guard: a region boundary inside a URL at a wrap hyphen still joins."""
    parser = PDFParser(
        [
            [
                {
                    "label": "paragraph_title",
                    "content": "Method",
                    "bbox_2d": [100, 100, 900, 130],
                },
                {
                    "label": "text",
                    "content": "Data are available at https://osf.io/Lak-",
                    "bbox_2d": [100, 200, 900, 250],
                },
                {
                    "label": "text",
                    "content": "ens/data for review.",
                    "bbox_2d": [100, 300, 900, 350],
                },
            ]
        ]
    )
    parser.parse()
    assert [e.text for e in parser.assembler.entries] == [
        "Data are available at https://osf.io/Lak-ens/data for review."
    ]


def test_unfinished_sentence_still_joins():
    """Guard: ordinary unterminated text still joins the next region."""
    parser = PDFParser(_two_region_pages("The effect persisted across cohorts and"))
    parser.parse()
    assert [e.text for e in parser.assembler.entries] == [
        "The effect persisted across cohorts and A second, separate paragraph begins here."
    ]
