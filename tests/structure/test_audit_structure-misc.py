"""Audit regressions: malformed region boxes."""

from __future__ import annotations

import pytest

_TABLE = "| Group | Mean |\n|---|---|\n| A | 1.0 |"


def _region(index, label, content, bbox):
    return {"index": index, "label": label, "content": content, "bbox_2d": bbox}


def _parse(pages):
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(pages)
    contents = parser.parse()
    return (
        [section.header for section in parser.sections],
        [table.caption for table in contents.tables],
        parser._detected_title,
    )


# --- malformed bbox_2d ------------------------------------------------------

_MALFORMED_LAYOUTS = {
    "fragment_after_bare_table_label": [
        _region(0, "figure_title", "Table 2", [100, 100, 900, 120]),
        _region(1, "text", "Descriptive statistics by group", "BOX"),
        _region(2, "table", _TABLE, [100, 200, 900, 500]),
    ],
    "bare_table_label": [
        _region(0, "figure_title", "Table 2", "BOX"),
        _region(1, "text", "Descriptive statistics by group", [100, 140, 900, 170]),
        _region(2, "table", _TABLE, [100, 200, 900, 500]),
    ],
    "table_confirming_the_fragment": [
        _region(0, "figure_title", "Table 2", [100, 100, 900, 120]),
        _region(1, "text", "Descriptive statistics by group", [100, 140, 900, 170]),
        _region(2, "table", _TABLE, "BOX"),
    ],
    "split_title_continuation": [
        _region(0, "doc_title", "A Study of", [100, 100, 900, 120]),
        _region(1, "doc_title", "Many Things", "BOX"),
        _region(2, "text", "Body text.", [100, 300, 900, 400]),
    ],
    "title_after_kicker": [
        _region(0, "doc_title", "Research Article", [100, 100, 900, 120]),
        _region(1, "doc_title", "Many Things", "BOX"),
        _region(2, "text", "Body text.", [100, 300, 900, 400]),
    ],
}


def _with_box(layout, box):
    return [
        [
            {**region, "bbox_2d": box if region["bbox_2d"] == "BOX" else region["bbox_2d"]}
            for region in layout
        ]
    ]


@pytest.mark.parametrize("box", [[], [100], [100, 140, 900]])
@pytest.mark.parametrize("layout", sorted(_MALFORMED_LAYOUTS))
def test_malformed_bbox_parses_like_a_missing_one(layout, box):
    """A short ``bbox_2d`` (bad OCR JSON, corrupted cache) used to raise
    IndexError in ``_is_bbox_nearby`` and abort the whole parse; it now
    counts as missing, as it already did for the region's provenance."""
    malformed = _parse(_with_box(_MALFORMED_LAYOUTS[layout], box))

    assert malformed == _parse(_with_box(_MALFORMED_LAYOUTS[layout], None))


def test_short_boxes_keep_the_composed_caption_and_title():
    _headers, captions, _title = _parse(_with_box(_MALFORMED_LAYOUTS["bare_table_label"], [1, 2]))
    assert captions == ["Table 2 Descriptive statistics by group"]

    layout = _MALFORMED_LAYOUTS["split_title_continuation"]
    _headers, _captions, title = _parse(_with_box(layout, [1, 2]))
    assert title == "A Study of Many Things"


def test_is_bbox_nearby_treats_malformed_boxes_as_missing():
    from bibr.structure.parse_text import TextHandlersMixin

    nearby = TextHandlersMixin._is_bbox_nearby
    far = [100, 900, 900, 950]
    for malformed in ([], [100], [100, 100, 900]):
        assert nearby(malformed, 1, far, 1) is True
        assert nearby(far, 1, malformed, 1) is True
        assert nearby(malformed, 1, far, 2) is False
    # Well-formed boxes still measure the vertical gap; extra values are ignored.
    assert nearby([100, 100, 900, 120], 1, far, 1) is False
    assert nearby([100, 100, 900, 120, 7], 1, [100, 130, 900, 150], 1) is True


def test_merge_bboxes_ignores_a_malformed_box():
    from bibr.structure.parse_text import TextHandlersMixin

    merge = TextHandlersMixin._merge_bboxes
    assert merge([100, 100, 900], [10, 20, 30, 40]) == [10.0, 20.0, 30.0, 40.0]
    assert merge([10, 20, 30, 40], []) == [10.0, 20.0, 30.0, 40.0]
    assert merge([], None) is None
    assert merge([0, 50, 10, 60], [5, 0, 20, 55]) == [0.0, 0.0, 20.0, 60.0]
