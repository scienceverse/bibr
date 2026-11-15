"""Paragraph joins across region breaks, and footnote markers."""

import pytest

from bibr.structure.footnote_buffer import printed_marker
from bibr.structure.pdf_parser import PDFParser


def _region(label, content, bbox):
    return {"index": 0, "label": label, "native_label": label, "content": content, "bbox_2d": bbox}


def _body_sentences(pages):
    parser = PDFParser(pages)
    contents = parser.parse()
    texts = [text for text, _, _, needs_seg, _ in parser._deferred_texts if needs_seg]
    parser.apply_segmentation(contents, [[text] for text in texts])
    return [sentence.text for sentence in contents.sentences]


def _across_page_break(first, second):
    return _body_sentences(
        [
            [
                _region("text", "An earlier paragraph.", [100, 100, 900, 300]),
                _region("text", first, [100, 800, 900, 950]),
            ],
            [_region("text", second, [100, 50, 900, 200])],
        ]
    )


@pytest.mark.parametrize(
    ("first", "second", "joined"),
    [
        (
            "the results of the inter-",
            "national study were clear.",
            "the results of the international study were clear.",
        ),
        (
            "this is a well-",
            "known effect in the field.",
            "this is a well-known effect in the field.",
        ),
        (
            "the results of the inter‐",
            "national study were clear.",
            "the results of the international study were clear.",
        ),
    ],
)
def test_a_word_hyphenated_at_a_page_break_is_mended_like_a_line_wrap(first, second, joined):
    assert _across_page_break(first, second)[-1] == joined


def test_a_doi_broken_at_a_page_break_keeps_its_hyphen():
    sentences = _across_page_break("see https://doi.org/10.1000/abc-", "def for details.")
    assert sentences[-1] == "see https://doi.org/10.1000/abc-def for details."


def test_a_word_hyphenated_at_a_column_break_past_a_figure_is_mended():
    sentences = _body_sentences(
        [
            [
                _region("text", "the results of the inter-", [100, 600, 480, 950]),
                _region("image", "", [520, 50, 900, 300]),
                _region("text", "national study were clear.", [520, 320, 900, 600]),
            ]
        ]
    )
    assert sentences == ["the results of the international study were clear."]


def test_a_page_span_after_a_mended_wrap_starts_at_the_second_page_text():
    parser = PDFParser(
        [
            [_region("text", "the results of the inter-", [100, 800, 900, 950])],
            [_region("text", "national study were clear.", [100, 50, 900, 200])],
        ]
    )
    parser.parse()
    entry = parser.assembler.entries[-1]
    assert entry.page_spans == [(0, 1), (len("the results of the inter"), 2)]
    assert entry.text[entry.page_spans[1][0] :].startswith("national")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("10Beginning of the preaching through the movie.", "10"),
        ("2Department of Psychology, University of Example.", "2"),
        ("∗ Corresponding author at: Example Hall.", "∗"),
        ("3D printing was used for the models.", None),
        ("5HT2A receptors were blocked.", None),
        ("19th-century sources are cited.", None),
        ("2024Annual report of the society.", None),
    ],
)
def test_printed_marker_reads_glued_numbers_and_the_asterisk_operator(text, expected):
    assert printed_marker(text) == expected
