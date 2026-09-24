"""PDF sentences record whether OCR produced any of their text.

Late cleanup (``PaperContents.finalize_text``) repairs OCR artifacts, and on
text read from the PDF text layer those repairs only corrupt prose: "a 2 x 2
design" became "a2x2 design" and ``age_group`` became ``agegroup``. A
paragraph is OCR text when any region joined into it was OCR output.
"""

from __future__ import annotations

import re

import pytest

from bibr.pipeline.stages.ocr import _postprocess_ocr_regions, _to_typed_regions
from bibr.structure.pdf_parser import PDFParser

NATIVE_PROSE = "Participants completed a 2 x 2 x 3 design; age_group was coded 1 2 3."


def _region(label: str, content: str, top: int, *, native: bool) -> dict:
    region = {"native_label": label, "label": label, "content": content}
    region["bbox_2d"] = [100, top, 900, top + 40]
    if native:
        region["_native_text_used"] = True
    return region


def _parse(pages: list[list[dict]]):
    for page in pages:
        for slot, region in enumerate(page):
            region["index"] = slot
    parser = PDFParser(_to_typed_regions(_postprocess_ocr_regions(pages)))
    contents = parser.parse()
    parser.apply_segmentation(
        contents,
        [re.split(r"(?<=[.!?])\s+", text) for text in parser.assembler.segmentable_texts],
    )
    parser.create_content_sections(contents)
    return contents


def _sentence(contents, fragment: str):
    return next(s for s in contents.sentences if fragment in s.text)


def test_text_layer_sentences_keep_their_prose_through_finalize():
    contents = _parse(
        [
            [
                _region("paragraph_title", "Methods", 100, native=True),
                _region("text", NATIVE_PROSE, 150, native=True),
                _region("text", "OCR saw the \\alpha level with N 1 2 0 here.", 250, native=False),
                _region(
                    "footnote", "1 Contact john_smith@uni.edu or see run_all.R.", 900, native=True
                ),
            ]
        ]
    )
    contents.finalize_text()

    assert _sentence(contents, "Participants").text == NATIVE_PROSE
    assert _sentence(contents, "OCR saw").text == "OCR saw the α level with N120 here."
    assert _sentence(contents, "Contact").text == "1 Contact john_smith@uni.edu or see run_all.R."
    assert _sentence(contents, "Participants").from_ocr is False
    assert _sentence(contents, "OCR saw").from_ocr is True
    assert _sentence(contents, "Contact").from_ocr is False


@pytest.mark.parametrize("first_native", [True, False], ids=["text-layer-first", "ocr-first"])
def test_paragraph_joined_from_text_layer_and_ocr_regions_is_ocr_text(first_native):
    contents = _parse(
        [
            [
                _region("text", "The first half of this paragraph and", 100, native=first_native),
                _region("text", "the second half of it.", 150, native=not first_native),
            ]
        ]
    )
    assert _sentence(contents, "first half").from_ocr is True


def test_paragraph_joined_across_pages_from_the_text_layer_stays_text_layer():
    contents = _parse(
        [
            [_region("text", "A paragraph that runs over the page and", 900, native=True)],
            [_region("text", "ends on the next one.", 100, native=True)],
        ]
    )
    assert _sentence(contents, "runs over").from_ocr is False


def test_section_hint_regions_carry_their_source():
    contents = _parse(
        [
            [
                _region("abstract", "We study a 2 x 2 design.", 100, native=True),
                _region(
                    "reference",
                    "Smith, J. (2020). A_B testing. J. Stat., 1, 1-2.",
                    800,
                    native=True,
                ),
            ]
        ]
    )
    assert _sentence(contents, "We study").from_ocr is False
    assert _sentence(contents, "Smith").from_ocr is False


def test_regions_without_a_source_are_treated_as_ocr():
    """Frozen region dicts predate the flag; unknown text keeps the OCR repairs."""
    contents = _parse([[{"native_label": "text", "label": "text", "content": "Plain body."}]])
    assert _sentence(contents, "Plain body").from_ocr is True


TABLE_HTML = "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>"


def test_text_layer_captions_are_not_ocr_text():
    """A text region that reads as a caption is routed to the caption
    handlers; the caption sentence keeps the region's source."""
    contents = _parse(
        [
            [
                _region("paragraph_title", "Results", 50, native=True),
                _region("text", "Table 2. Scores for items 1 2 3 by age_group.", 100, native=True),
                _region("table", TABLE_HTML, 160, native=False),
                _region("image", "", 400, native=False),
                _region("text", "Figure 1. Items 1 2 3 by age_group.", 600, native=True),
            ]
        ]
    )
    contents.finalize_text()

    table_caption = _sentence(contents, "Scores for items")
    figure_caption = _sentence(contents, "Figure 1.")
    assert table_caption.text == "Table 2. Scores for items 1 2 3 by age_group."
    assert figure_caption.text == "Figure 1. Items 1 2 3 by age_group."
    assert table_caption.from_ocr is False
    assert figure_caption.from_ocr is False


def test_text_layer_caption_labelled_as_a_heading_is_not_ocr_text():
    contents = _parse(
        [
            [
                _region("paragraph_title", "Results", 50, native=True),
                _region("paragraph_title", "Table 3 Items 4 5 6 by age_group", 100, native=True),
                _region("table", TABLE_HTML, 160, native=False),
            ]
        ]
    )
    contents.finalize_text()

    caption = _sentence(contents, "Items 4 5 6")
    assert (caption.text, caption.from_ocr) == ("Table 3 Items 4 5 6 by age_group", False)


def test_unowned_text_layer_caption_replayed_as_body_is_not_ocr_text():
    contents = _parse(
        [
            [
                _region("paragraph_title", "Results", 50, native=True),
                _region("text", "Table 2. Scores for items 1 2 3 by age_group.", 100, native=True),
                _region("text", "Body text follows.", 400, native=True),
            ]
        ]
    )
    assert _sentence(contents, "Scores for items").from_ocr is False


def test_replayed_table_label_fragment_keeps_its_source():
    """A fragment staged after a bare "Table N" label and never confirmed by a
    table is replayed as body text; it pulled the next text-layer paragraph
    into OCR cleanup when the replay dropped its source."""
    contents = _parse(
        [
            [
                _region("paragraph_title", "Results", 50, native=True),
                _region("table_title", "Table 3", 100, native=False),
                _region("text", "Scores for items 1 2 3 by age_group", 145, native=True),
                _region("text", "Body: items 1 2 3 by age_group were scored.", 400, native=True),
            ]
        ]
    )
    contents.finalize_text()

    replayed = _sentence(contents, "Scores for items")
    assert replayed.from_ocr is False
    assert "items 1 2 3 by age_group were scored." in replayed.text


def test_demoted_text_layer_heading_is_not_ocr_text():
    """The heading gate demotes a sentence-like paragraph_title to body text."""
    prose = "Participants completed a 2 x 2 x 3 design and age_group was coded as 1 2 3 here."
    contents = _parse(
        [
            [
                _region("paragraph_title", "Method", 50, native=True),
                _region("paragraph_title", prose, 100, native=True),
            ]
        ]
    )
    contents.finalize_text()

    assert _sentence(contents, "Participants").text == prose
    assert _sentence(contents, "Participants").from_ocr is False


def test_reference_rows_after_the_reference_list_keep_their_source():
    """Publisher boilerplate ends References; it and the rows after it are
    emitted through the terminal-tail paths, which must keep the source."""
    note = (
        "Publisher's Note Springer Nature remains neutral with regard to jurisdictional "
        "claims in published maps and institutional affiliations."
    )
    contents = _parse(
        [
            [
                _region("paragraph_title", "References", 100, native=True),
                _region("reference_content", "Smith, J. 2020. First.", 150, native=True),
                _region("reference_content", note, 800, native=True),
            ],
            [_region("reference_content", "Chair of Data_Science, Saarland.", 100, native=True)],
        ]
    )

    assert _sentence(contents, "Springer Nature").from_ocr is False
    assert _sentence(contents, "Chair of").from_ocr is False


def test_reference_row_under_a_footnotes_heading_keeps_its_source():
    contents = _parse(
        [
            [
                _region("paragraph_title", "Footnotes", 100, native=True),
                _region("reference_content", "1 See run_all.R for items 1 2 3.", 150, native=True),
            ]
        ]
    )
    assert _sentence(contents, "run_all.R").from_ocr is False


def test_carry_over_flushed_after_a_section_change_keeps_its_source():
    parser = PDFParser(json_result=[])
    parser._current_section_id = 1
    parser._handle_content("a text-layer paragraph without an end", 1, [0, 0, 1, 1], from_ocr=False)
    parser._current_section_id = 2
    parser._handle_content("New sentence in section 2.", 2, [0, 0, 1, 1], from_ocr=False)

    (flushed,) = [e for e in parser.assembler.entries if "without an end" in e.text]
    assert (flushed.section_id, flushed.from_ocr) == (1, False)
