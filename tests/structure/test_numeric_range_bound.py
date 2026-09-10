"""Compound numeric references must not be expandable into a memory bomb.

Both xref detection ("Tables 1-3") and numeric citation linking ("[4-7]")
fill a range so every id gets its own link. The input is document text, which
a crafted DOCX/HTML/JATS — or plain bad OCR — fully controls, so an unbounded
fill turns one sentence into a billion ints (tens of GB). ``MAX_INT_RANGE_SPAN``
caps the fill; wider spans keep only their endpoints.
"""

import time

import pytest

from bibr.paper_contents import PaperSentence, PaperTable
from bibr.structure.citation_linker import _expand_numeric_range
from bibr.structure.xref_utils import (
    MAX_INT_RANGE_SPAN,
    _expand_nums,
    detect_xrefs,
    expand_int_range,
)


def test_plausible_ranges_are_still_filled():
    assert _expand_nums("1-3") == [1, 2, 3]
    assert _expand_nums("2, 5-7") == [2, 5, 6, 7]
    assert _expand_numeric_range("4-7") == [4, 5, 6, 7]
    # Reversed ranges from OCR typos keep their ascending interpretation.
    assert _expand_numeric_range("5-3,9") == [3, 4, 5, 9]


def test_range_at_the_bound_is_filled_and_one_past_it_is_not():
    assert len(expand_int_range(1, 1 + MAX_INT_RANGE_SPAN)) == MAX_INT_RANGE_SPAN + 1
    assert expand_int_range(1, 2 + MAX_INT_RANGE_SPAN) == [1, 2 + MAX_INT_RANGE_SPAN]


@pytest.mark.parametrize("expand", [_expand_nums, _expand_numeric_range])
def test_absurd_range_is_bounded_not_materialised(expand):
    started = time.perf_counter()
    result = expand("1-999999999")
    elapsed = time.perf_counter() - started

    assert result == [1, 999999999]
    # Unbounded this allocates ~40 GB and takes minutes; bounded it is instant.
    assert elapsed < 1.0


def test_detect_xrefs_survives_a_crafted_range_in_document_text():
    sentences = [
        PaperSentence(
            text_id=1,
            text="As shown in Tables 1-999999999 the effect holds.",
            section_id=1,
            paragraph_id=1,
            page_number=1,
        )
    ]
    tables = [
        PaperTable(
            table_id=1,
            df=None,
            tbl_html="<table></table>",
            section_id=1,
            caption="Descriptives",
        )
    ]

    started = time.perf_counter()
    xrefs = detect_xrefs(sentences, tables, [])
    assert time.perf_counter() - started < 1.0
    # Table 1 exists and still links; the absurd upper bound matches nothing.
    assert [x.xref_id for x in xrefs] == [1]
