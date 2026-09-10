"""Tests for OcrRegionResult — typed construction and dict serialization."""

from __future__ import annotations

import pytest

from bibr.ocr.types import OcrRegionResult


def test_minimal_region_to_dict():
    r = OcrRegionResult(
        index=0,
        native_label="text",
        label="text",
        content="hello",
        bbox_2d=[0, 0, 100, 100],
    )
    assert r.to_dict() == {
        "index": 0,
        "native_label": "text",
        "label": "text",
        "content": "hello",
        "bbox_2d": [0, 0, 100, 100],
    }


def test_optional_fields_present_only_when_set():
    """Wire format must omit ``image_b64`` and font keys when None — PDFParser
    treats absence and explicit None differently in some places."""
    r = OcrRegionResult(
        index=1,
        native_label="figure",
        label="figure",
        content="",
        bbox_2d=None,
        image_b64="abc",
    )
    d = r.to_dict()
    assert d["image_b64"] == "abc"
    assert "_font_size" not in d
    assert "_font_weight" not in d
    assert "_font_bold" not in d


def test_font_metadata_in_to_dict():
    r = OcrRegionResult(
        index=2,
        native_label="text",
        label="text",
        content="x",
        bbox_2d=[0, 0, 1, 1],
        font_size=12.5,
        font_weight="bold",
        font_bold=True,
    )
    d = r.to_dict()
    assert d["_font_size"] == 12.5
    assert d["_font_weight"] == "bold"
    assert d["_font_bold"] is True


def test_from_layout_region_copies_font_metadata():
    """from_layout_region constructs a result by carrying font keys from the
    source layout region dict (the producers' current pattern)."""
    layout = {
        "label": "text",
        "bbox_2d": [10, 20, 30, 40],
        "_font_size": 11.0,
        "_font_weight": "normal",
        "_font_bold": False,
    }
    r = OcrRegionResult.from_layout_region(
        layout,
        slot_idx=5,
        content="ocr text",
    )
    assert r.index == 5
    assert r.native_label == "text"
    assert r.label == "text"  # _map_native_label is identity for "text"
    assert r.content == "ocr text"
    assert r.bbox_2d == [10, 20, 30, 40]
    assert r.font_size == 11.0
    assert r.font_weight == "normal"
    assert r.font_bold is False


def test_from_layout_region_maps_formula_native_labels():
    """display_formula / inline_formula → label="formula"."""
    for native in ("display_formula", "inline_formula"):
        layout = {"label": native, "bbox_2d": [0, 0, 1, 1]}
        r = OcrRegionResult.from_layout_region(layout, slot_idx=0, content="x")
        assert r.native_label == native
        assert r.label == "formula"


def test_region_dict_roundtrip_matches_legacy_shape():
    """Critical: the dict shape must equal what current producers emit so
    downstream PDFParser consumers see no diff."""
    layout = {
        "label": "doc_title",
        "bbox_2d": [0, 0, 100, 50],
        "_font_size": 18.0,
    }
    r = OcrRegionResult.from_layout_region(layout, slot_idx=0, content="My Paper")
    d = r.to_dict()
    legacy = {
        "index": 0,
        "native_label": "doc_title",
        "label": "doc_title",
        "content": "My Paper",
        "bbox_2d": [0, 0, 100, 50],
        "_font_size": 18.0,
    }
    assert d == legacy


def test_from_layout_region_copies_region_meta_and_native_flag():
    """is_italic / page_w / page_h / native_text_used must survive the
    OcrRegionResult seam — PDFParser reads them off the post-OCR region dicts
    to build sentence region_meta (exported as _is_italic/_page_w/_page_h) and
    the OCR success gate uses _native_text_used to exclude bypassed regions."""
    layout = {
        "label": "text",
        "bbox_2d": [0, 0, 1, 1],
        "_is_italic": True,
        "_page_w": 595.0,
        "_page_h": 842.0,
        "_native_text_used": True,
    }
    r = OcrRegionResult.from_layout_region(layout, slot_idx=0, content="x")
    assert r.is_italic is True
    assert r.page_w == 595.0
    assert r.page_h == 842.0
    assert r.native_text_used is True
    d = r.to_dict()
    assert d["_is_italic"] is True
    assert d["_page_w"] == 595.0
    assert d["_page_h"] == 842.0
    assert d["_native_text_used"] is True


def test_region_meta_fields_omitted_when_none():
    """A region with no font/geometry metadata must not emit the new keys —
    absence vs explicit None matters to PDFParser (mirrors font-key behaviour)."""
    r = OcrRegionResult(index=0, native_label="text", label="text", content="x", bbox_2d=None)
    d = r.to_dict()
    for key in ("_is_italic", "_page_w", "_page_h", "_native_text_used", "_bbox_pdf_pts"):
        assert key not in d


def test_bbox_pdf_pts_flows_through_seam():
    """The containment-correct PDF-point bbox (bottom-left origin, y-up) must
    survive the OcrRegionResult seam so PDFParser can populate region_meta's
    exported ``_bbox_2d`` from it instead of the 0..1000 layout-space bbox."""
    layout = {
        "label": "text",
        "bbox_2d": [0, 0, 1000, 1000],
        "_bbox_pdf_pts": [0.0, 0.0, 612.0, 792.0],
    }
    r = OcrRegionResult.from_layout_region(layout, slot_idx=0, content="x")
    assert r.bbox_pdf_pts == [0.0, 0.0, 612.0, 792.0]
    d = r.to_dict()
    assert d["_bbox_pdf_pts"] == [0.0, 0.0, 612.0, 792.0]
    # from_dict must recover it too.
    assert OcrRegionResult.from_dict(d).bbox_pdf_pts == [0.0, 0.0, 612.0, 792.0]


def test_is_italic_false_is_emitted():
    """is_italic=False is a real determination (upright text), not 'unknown', so
    it must be emitted just like font_bold=False."""
    r = OcrRegionResult(
        index=0, native_label="text", label="text", content="x", bbox_2d=None, is_italic=False
    )
    assert r.to_dict()["_is_italic"] is False


def test_invalid_construction_rejected():
    """Required fields without defaults raise TypeError."""
    with pytest.raises(TypeError):
        OcrRegionResult(index=0)  # missing required fields


# --- from_dict: the wire-format adapter (typed Region IR seam) ---


def test_from_dict_roundtrips_to_dict():
    """from_dict must invert to_dict exactly for a fully-populated region."""
    r = OcrRegionResult(
        index=3,
        native_label="text",
        label="text",
        content="hello",
        bbox_2d=[1.0, 2.0, 3.0, 4.0],
        image_b64="abc",
        font_size=11.5,
        font_weight="bold",
        font_bold=True,
        is_italic=False,
        page_w=595.0,
        page_h=842.0,
        native_text_used=True,
        bbox_pdf_pts=[0.61, 1.68, 1.83, 3.35],
    )
    assert OcrRegionResult.from_dict(r.to_dict()) == r


def test_from_dict_minimal_legacy_dict_gets_consumer_defaults():
    """A sparse legacy dict (frozen eval JSON, hand-built test fixture) must
    load with the same defaults PDFParser's ``.get()`` reads used: label/
    native_label/content default to '', index to 0, everything else None."""
    r = OcrRegionResult.from_dict({"label": "text", "content": "x"})
    assert r.label == "text"
    assert r.content == "x"
    assert r.native_label == ""
    assert r.index == 0
    assert r.bbox_2d is None
    assert r.image_b64 is None
    assert r.font_size is None
    assert r.native_text_used is None


def test_from_dict_ignores_unknown_keys():
    """Frozen JSONs may carry extra keys from older producers — ignore them
    like the dict consumers' ``.get()`` calls did."""
    r = OcrRegionResult.from_dict({"label": "text", "content": "x", "some_old_key": 1})
    assert r.label == "text"


def test_from_dict_none_content_normalized_to_empty():
    """Consumers guarded with ``region.get('content', '') or ''`` — explicit
    None in a frozen dict must not surface as a None attribute."""
    r = OcrRegionResult.from_dict({"label": "text", "content": None})
    assert r.content == ""


def test_raw_ocr_content_roundtrips_only_when_present():
    region = OcrRegionResult(
        index=0,
        native_label="table",
        label="table",
        content="<table><tr><td>canonical</td></tr></table>",
        raw_content="<fcel>canonical<nl>",
        bbox_2d=[0, 0, 1, 1],
    )

    serialized = region.to_dict()

    assert serialized["_raw_ocr_content"] == "<fcel>canonical<nl>"
    assert OcrRegionResult.from_dict(serialized) == region
    assert (
        "_raw_ocr_content"
        not in OcrRegionResult(
            index=0,
            native_label="text",
            label="text",
            content="canonical",
            bbox_2d=None,
        ).to_dict()
    )


def test_finish_reason_roundtrips_only_when_present():
    region = OcrRegionResult(
        index=0,
        native_label="table",
        label="table",
        content="partial",
        bbox_2d=[0, 0, 1, 1],
        finish_reason="length",
    )

    serialized = region.to_dict()

    assert serialized["_ocr_finish_reason"] == "length"
    assert OcrRegionResult.from_dict(serialized) == region


def test_native_rejection_provenance_roundtrips_losslessly():
    native = "Ernst Lau’s survey article ()."
    raw_ocr = "Ernst Lau’s survey article (1927)."
    region = OcrRegionResult.from_layout_region(
        {
            "label": "text",
            "bbox_2d": [109, 118, 833, 896],
            "_native_text_candidate": native,
            "_native_text_rejection_reason": "private_use",
        },
        slot_idx=2,
        content=raw_ocr,
        raw_content=raw_ocr,
    )

    serialized = region.to_dict()

    assert serialized["_native_text_candidate"] == native
    assert serialized["_native_text_rejection_reason"] == "private_use"
    assert serialized["_raw_ocr_content"] == raw_ocr
    assert OcrRegionResult.from_dict(serialized) == region
