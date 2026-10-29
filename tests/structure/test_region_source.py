"""Each region summary names the route its text took (text layer or OCR)."""

from __future__ import annotations


def _region(index, label, content, **wire):
    return {"index": index, "label": label, "content": content, "bbox_2d": [0, 0, 100, 100], **wire}


def _summaries(regions):
    from bibr.structure.pdf_parser import PDFParser

    return PDFParser([regions]).parse().region_summaries


def test_region_summaries_record_the_text_route():
    summaries = _summaries(
        [
            _region(0, "header", "J Things 12 (2024)", _native_text_used=True),
            _region(1, "doc_title", "Paper Title"),
            _region(
                2,
                "text",
                "Recovered body text.",
                _ocr_consensus={"chosen": "second", "chosen_backend": "glm-http"},
            ),
            _region(3, "image", ""),
        ]
    )

    assert [(s.source, s.ocr_backend) for s in summaries] == [
        ("text_layer", None),
        ("ocr", None),
        ("ocr", "glm-http"),
        (None, None),
    ]


def test_edge_cut_is_kept_only_for_text_layer_regions():
    summaries = _summaries(
        [
            _region(
                0,
                "header",
                "https://doi.org/10.1177/117793222311603",
                _native_text_used=True,
                _native_text_edge_cut=True,
            ),
            _region(1, "footer", "Page 1 of 9", _native_text_edge_cut=True),
        ]
    )

    assert [s.native_text_edge_cut for s in summaries] == [True, False]
