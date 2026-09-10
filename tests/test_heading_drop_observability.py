"""Silent heading-loss points must leave a trace in the logs.

Three places lose a heading with no downstream artifact: layout overlap
resolution dropping a heading box, the publisher-noise allowlist, and
running-header demotion. Each should be observable when debugging a paper
with missing sections.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest


def _region(index, label, content):
    return {
        "index": index,
        "label": label,
        "content": content,
        "bbox_2d": [0, 0, 100, 100],
    }


def test_containment_drop_of_heading_box_is_logged(caplog):
    from bibr.layout_base import _log_dropped_heading_boxes
    from bibr.layout_utils import _filter_containment

    id2label = {0: "paragraph_title"}
    # Inner heading box fully contained in a larger same-class box → dropped.
    boxes = np.array(
        [
            [0, 0.9, 100, 100, 500, 200],
            [0, 0.8, 120, 110, 300, 150],
        ],
        dtype=float,
    )
    keep = _filter_containment(boxes, id2label)
    assert not keep.all()  # precondition: the filter drops one

    with caplog.at_level(logging.INFO, logger="bibr.layout_base"):
        _log_dropped_heading_boxes(boxes, keep, id2label)
    messages = [r.message for r in caplog.records]
    assert any("paragraph_title" in m for m in messages)


def test_containment_drop_of_body_text_is_not_logged(caplog):
    from bibr.layout_base import _log_dropped_heading_boxes

    id2label = {0: "text"}
    boxes = np.array(
        [
            [0, 0.9, 100, 100, 500, 200],
            [0, 0.8, 120, 110, 300, 150],
        ],
        dtype=float,
    )
    keep = np.array([True, False])
    with caplog.at_level(logging.INFO, logger="bibr.layout_base"):
        _log_dropped_heading_boxes(boxes, keep, id2label)
    assert not caplog.records


def test_publisher_noise_heading_drop_is_logged(caplog):
    from bibr.structure.pdf_parser import PDFParser

    json_result = [
        [
            _region(0, "doc_title", "Paper Title"),
            _region(1, "paragraph_title", "SAGE"),
            _region(2, "text", "Body sentence."),
        ]
    ]
    with caplog.at_level(logging.INFO, logger="bibr.structure.parse_headings"):
        PDFParser(json_result).parse()
    assert any("publisher noise" in r.message.lower() for r in caplog.records)


@pytest.mark.parametrize("level_visible", [True])
def test_running_header_demotion_logs_texts(caplog, level_visible):
    from bibr.structure.pdf_parser import PDFParser

    # The same paragraph_title on two pages is demoted as a running header;
    # the demoted text must be visible at DEBUG for traceability.
    json_result = [
        [
            _region(0, "doc_title", "Paper Title"),
            _region(1, "paragraph_title", "Journal of Testing 12(3)"),
            _region(2, "text", "Body sentence one."),
        ],
        [
            _region(0, "paragraph_title", "Journal of Testing 12(3)"),
            _region(1, "text", "Body sentence two."),
        ],
    ]
    with caplog.at_level(logging.DEBUG, logger="bibr.structure.pdf_parser"):
        PDFParser(json_result).parse()
    assert any(
        "journal of testing 12(3)" in r.message.lower() and "running header" in r.message.lower()
        for r in caplog.records
    )
