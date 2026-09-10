"""A decimal point must not double as the caption separator.

``_TABLE_CAPTION_RE`` / ``_FIGURE_CAPTION_RE`` require a separator after the
id precisely so prose ("Table 7 shows the results.") is not misrouted into
caption ownership. But a chapter-numbered document writes "Table 7.5 shows
…", and the decimal satisfied that separator — ``_handle_content`` then
handed the whole sentence to ``_handle_table_caption`` and returned, so the
body text never reached the assembler at all.
"""

import pytest

from bibr.structure.pdf_parser import PDFParser

PROSE = [
    "Table 7.5 shows the results of the regression.",
    "Table 7.5.1 reports the residuals for each cohort.",
    "Figure 3.2 illustrates the effect of dose on latency.",
    "Fig. 3.2 illustrates the effect of dose on latency.",
    "Table 7 shows the results.",
]

CAPTIONS = [
    "Table 1: Descriptive statistics",
    "Table 7. Descriptive statistics",
    "Table 7.5: Descriptive statistics",
    "Table 7.5. Descriptive statistics",
    "Table 2 - Results by condition",
    "Table 1 | Overview of measures",
]

FIGURE_CAPTIONS = [
    "Figure 3: Effect of dose",
    "Fig. 3. Effect of dose",
    "Figure 3.2: Effect of dose",
]


@pytest.mark.parametrize("text", PROSE)
def test_prose_with_a_dotted_number_is_not_a_caption(text):
    assert not PDFParser._TABLE_CAPTION_RE.match(text)
    assert not PDFParser._FIGURE_CAPTION_RE.match(text)


@pytest.mark.parametrize("text", CAPTIONS)
def test_real_table_captions_still_match(text):
    assert PDFParser._TABLE_CAPTION_RE.match(text)


@pytest.mark.parametrize("text", FIGURE_CAPTIONS)
def test_real_figure_captions_still_match(text):
    assert PDFParser._FIGURE_CAPTION_RE.match(text)


@pytest.mark.parametrize("text", PROSE)
def test_prose_reaches_the_assembler_instead_of_caption_ownership(text):
    """The end-to-end consequence: the sentence must survive as body text."""
    parser = PDFParser(json_result=[])
    parser._current_section_id = 1
    parser._handle_content(text, 4, bbox=[0, 0, 1, 1])
    parser._flush_carry_over()

    assert [e.text for e in parser.assembler.entries] == [text]


def test_a_real_caption_in_a_content_region_is_still_re_routed():
    """The re-router must keep working for genuinely mislabelled captions."""
    parser = PDFParser(json_result=[])
    parser._current_section_id = 1
    parser._handle_content("Table 7.5: Descriptive statistics", 4, bbox=[0, 0, 1, 1])
    parser._flush_carry_over()

    assert parser.assembler.entries == []
