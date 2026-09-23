"""Printed figure/table labels and resolving in-text mentions by them.

A float's label is what its caption prints after the word ("3.1", "S2",
"A1", "IV"). A mention resolves by label when any float of its kind has one,
else by position; every mention yields a row, with no target when it names
no float or two.
"""

import pandas as pd
import pytest

from bibr.paper_contents import PaperFigure, PaperSentence, PaperTable
from bibr.structure.float_labels import (
    caption_label,
    is_supplementary_label,
    label_element_label,
    normalize_label,
)
from bibr.structure.xref_utils import detect_xrefs


@pytest.mark.parametrize(
    ("caption", "kind", "label"),
    [
        ("Table 3.1. Descriptive statistics", "table", "3.1"),
        ("Table S2: Robustness checks", "table", "S2"),
        ("Figure A1. Sensitivity", "figure", "A1"),
        ("TABLE IV", "table", "IV"),
        ("Table C Appendix items", "table", "C"),
        ("Supplementary Table 4. Items", "table", "S4"),
        ("Suppl. Fig. 4 Traces", "figure", "S4"),
        ("Supplementary Table S4. Items", "table", "S4"),
        ("Table S 1. Items", "table", "S1"),
        ("Fig. 2a Panel", "figure", "2a"),
        ("Table 3 (Continued)", "table", "3"),
        ("Table 2 - Results by condition", "table", "2"),
        # No label: none printed, the other kind's word, a year, prose.
        ("Table showing the results", "table", None),
        ("Table 2. Values", "figure", None),
        ("Table 2019 statistics", "table", None),
        ("table civil", "table", None),
        (None, "table", None),
        # An eLife supplement is not its parent: no duplicate "1".
        ("Figure 1—figure supplement 1. Controls", "figure", None),
        ("Table 2—source data 1", "table", None),
    ],
)
def test_caption_label(caption, kind, label):
    assert caption_label(caption, kind) == label


@pytest.mark.parametrize(
    ("text", "kind", "label"),
    [
        ("Table 2", "table", "2"),
        ("Fig. 3.", "figure", "3"),
        ("S1", "table", "S1"),
        ("S1 Table", "table", "S1"),
        ("Supplementary Table 1", "table", "S1"),
        ("2", "figure", "2"),
        ("Scheme 1", "figure", None),
        ("Figure 1—figure supplement 1", "figure", None),
    ],
)
def test_label_element_label(text, kind, label):
    assert label_element_label(text, kind) == label


def test_labels_compare_case_insensitively_without_whitespace():
    assert normalize_label("S 2") == normalize_label("s2")
    assert is_supplementary_label("S2") and is_supplementary_label("s1.3")
    assert not is_supplementary_label("SA") and not is_supplementary_label("3")


def _sent(text: str) -> list[PaperSentence]:
    return [PaperSentence(text_id=1, text=text, section_id=1, paragraph_id=1)]


def _table(table_id: int, label: str | None = None, page: int | None = None) -> PaperTable:
    return PaperTable(
        table_id=table_id,
        df=pd.DataFrame(),
        tbl_html="",
        section_id=1,
        page_number=page,
        label=label,
    )


def _figure(figure_id: int, label: str | None = None, page: int | None = None) -> PaperFigure:
    return PaperFigure(
        figure_id=figure_id,
        section_id=1,
        image_b64=None,
        caption=None,
        page_number=page,
        label=label,
    )


def _links(text, tables=(), figures=()):
    return [
        (xref.xref_type, xref.xref_id, xref.tier)
        for xref in detect_xrefs(_sent(text), list(tables), list(figures))
    ]


LABELLED = [
    _table(10, "1"),
    _table(11, "3.1"),
    _table(12, "3.2"),
    _table(13, "S2"),
    _table(14, "A1"),
    _table(15, "IV"),
    _table(16, "C"),
]


class TestResolveByLabel:
    @pytest.mark.parametrize(
        ("text", "target"),
        [
            ("Table 3.1 lists", 11),
            ("Table S2 lists", 13),
            ("Table s2 lists", 13),
            ("Table A1 lists", 14),
            ("Table IV lists", 15),
            ("Table C lists", 16),
            ("Supplementary Table 2 lists", 13),
        ],
    )
    def test_a_printed_label_finds_its_table(self, text, target):
        assert _links(text, LABELLED) == [("table", target, "label")]

    def test_a_dotted_list_and_range(self):
        assert _links("Tables 3.1 and 3.2 list", LABELLED) == [
            ("table", 11, "label"),
            ("table", 12, "label"),
        ]
        tables = [_table(1, "3.1"), _table(2, "3.2"), _table(3, "3.3")]
        assert [x[1] for x in _links("Tables 3.1–3.3 list", tables)] == [1, 2, 3]

    def test_the_number_is_not_the_id(self):
        # "Table 1" is table 10 here; a missed float shifts no later link.
        assert _links("Table 1 lists", LABELLED) == [("table", 10, "label")]

    def test_an_unmatched_label_keeps_its_row_without_target(self):
        assert _links("Table 9 lists", LABELLED) == [("table", 0, "label")]

    def test_a_label_two_floats_print_is_ambiguous(self):
        tables = [_table(1, "2"), _table(2, "2")]
        assert _links("Table 2 lists", tables) == [("table", 0, "label")]

    def test_roman_and_arabic_labels_are_different_strings(self):
        assert _links("Table 4 lists", LABELLED) == [("table", 0, "label")]

    def test_a_lettered_list(self):
        tables = [_table(1, "II"), _table(2, "III")]
        assert [x[1] for x in _links("Tables II and III list", tables)] == [1, 2]

    def test_panel_letters_are_not_figures(self):
        figures = [_figure(1, "1"), _figure(2, "2")]
        assert _links("Fig. 2A, B and Figure 1b show", figures=figures) == [
            ("figure", 2, "label"),
            ("figure", 1, "label"),
        ]
        # A float printed as "1a" is found before its "1" would be.
        figures = [_figure(1, "1a"), _figure(2, "1b")]
        assert _links("Figure 1b shows", figures=figures) == [("figure", 2, "label")]

    def test_lettered_labels_after_a_lowercase_word_are_prose(self):
        assert _links("the table I made", LABELLED) == []

    def test_kinds_resolve_separately(self):
        assert _links("Figure S2 shows", LABELLED, [_figure(1, "1")]) == [
            ("supplementary", 2, None)
        ]


class TestSupplements:
    def test_an_s_label_no_float_carries_is_a_supplement(self):
        assert _links("Table S5 lists", LABELLED) == [("supplementary", 5, None)]
        assert _links("Table S5 lists", [_table(1)]) == [("supplementary", 5, None)]

    @pytest.mark.parametrize("tables", [LABELLED, [_table(1), _table(2), _table(3), _table(4)]])
    def test_supplementary_table_4_is_one_supplementary_row(self, tables):
        # Not also a "table" row pointing at the main Table 4.
        assert _links("Supplementary Table 4 lists", tables) == [("supplementary", 4, None)]

    def test_s_prefixed_range_takes_the_prefix(self):
        tables = [_table(1, "S1"), _table(2, "S2"), _table(3, "S3")]
        assert [x[1] for x in _links("Tables S1–3 list", tables)] == [1, 2, 3]

    def test_named_supplements_without_a_label_still_link_nothing(self):
        assert _links("See Supplementary Materials and Supplementary Tables.", LABELLED) == [
            ("supplementary", 0, None),
            ("supplementary", 0, None),
        ]


class TestPositionFallback:
    def test_no_labels_means_the_number_is_a_position(self):
        # Page order, a float without a page first, then list order.
        figures = [_figure(20, page=2), _figure(21, page=1), _figure(22), _figure(23, page=2)]
        assert _links("Figures 1-5 show", figures=figures) == [
            ("figure", 22, "position"),
            ("figure", 21, "position"),
            ("figure", 20, "position"),
            ("figure", 23, "position"),
            ("figure", 0, "position"),
        ]

    def test_a_panel_mention_takes_its_figure_position(self):
        figures = [_figure(5), _figure(6)]
        assert _links("Fig. 2b shows", figures=figures) == [("figure", 6, "position")]

    def test_a_label_that_is_not_a_number_has_no_position(self):
        assert _links("Table 3.1 lists", [_table(1), _table(2), _table(3)]) == [
            ("table", 0, "position")
        ]

    def test_one_labelled_float_switches_the_kind_to_labels(self):
        tables = [_table(1), _table(2, "2")]
        assert _links("Tables 1 and 2 list", tables) == [
            ("table", 0, "label"),
            ("table", 2, "label"),
        ]
