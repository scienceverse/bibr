from __future__ import annotations

import builtins

import pytest

from bibr.evaluation.segmenter_capture import capture_segmenter_inputs
from bibr.structure.pdf_parser import PDFParser


def _region(index: int, content: str, bbox: list[int], *, label: str = "text") -> dict:
    return {
        "index": index,
        "label": label,
        "native_label": label,
        "content": content,
        "bbox_2d": bbox,
    }


def test_capture_matches_pdf_parser_segmentable_texts_and_provenance(monkeypatch) -> None:
    ocr_regions = [
        [
            _region(0, "First paragraph.", [0, 0, 100, 20]),
            _region(1, "E = mc^2", [0, 30, 100, 50], label="display_formula"),
            _region(
                2,
                "This cross page paragraph deliberately contains enough words to remain "
                "ordinary body text and continues",
                [0, 60, 100, 80],
            ),
        ],
        [_region(0, "on the next page.", [0, 0, 100, 20])],
    ]
    original_import = builtins.__import__

    def forbid_wtpsplit(name, *args, **kwargs):
        if name.startswith("wtpsplit"):
            raise AssertionError("capture must not load wtpsplit")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_wtpsplit)

    captured = capture_segmenter_inputs(ocr_regions)
    parser = PDFParser(ocr_regions)
    parser.parse()

    assert [item.text for item in captured] == parser.assembler.segmentable_texts
    assert [item.source_region_indices for item in captured] == [
        ((1, 0),),
        ((1, 2), (2, 0)),
    ]
    assert captured[1].page_numbers == (1, 2)
    assert all(item.source_region_indices for item in captured)


def test_capture_rejects_ambiguous_duplicate_region_locations() -> None:
    ocr_regions = [
        [
            _region(0, "Repeated body text.", [0, 0, 100, 20]),
            _region(1, "Repeated body text.", [0, 0, 100, 20]),
        ]
    ]

    with pytest.raises(ValueError, match="Ambiguous segmenter capture provenance"):
        capture_segmenter_inputs(ocr_regions)
