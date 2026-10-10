"""Page triage from the native-text pass's decisions."""

from types import SimpleNamespace

from bibr.ocr.pdf_inspection import PdfPageInspection
from bibr.scan.page_kind import PageKind, classify_page, classify_pages, ensure_page_kinds


def _page(index=0, *, chars=500, invisible=False):
    return PdfPageInspection(
        index=index,
        width=612.0,
        height=792.0,
        crop_box=(0.0, 0.0, 612.0, 792.0),
        char_count=chars,
        invisible_text_layer=invisible,
    )


def _native(label="text"):
    return {"label": label, "_native_text_used": True}


def _gated(gate="usability", label="text"):
    return {"label": label, "_native_gate": gate}


def test_no_inspection_has_no_class():
    assert classify_page(None, [_native()]) is None


def test_invisible_layer_is_a_scan():
    assert classify_page(_page(invisible=True), [_native()]) is PageKind.SCAN


def test_no_text_layer_with_text_regions_is_a_scan():
    assert classify_page(_page(chars=0), [{"label": "text"}]) is PageKind.SCAN


def test_no_text_layer_and_no_text_regions_is_not_a_scan():
    assert classify_page(_page(chars=0), [{"label": "image"}]) is PageKind.BORN_DIGITAL


def test_unknown_layout_rests_on_the_inspection():
    assert classify_page(_page(chars=0), None) is PageKind.SCAN
    assert classify_page(_page(chars=10), None) is PageKind.BORN_DIGITAL


def test_mostly_rejected_text_layer_is_broken():
    regions = [_gated(), _gated("private_use"), _native(), {"label": "image"}]
    assert classify_page(_page(), regions) is PageKind.BROKEN_TEXT_LAYER


def test_one_rejected_region_is_not_a_broken_page():
    assert classify_page(_page(), [_gated(), _native()]) is PageKind.BORN_DIGITAL


def test_rejections_in_the_minority_are_not_a_broken_page():
    regions = [_gated(), _gated(), _native(), _native(), _native()]
    assert classify_page(_page(), regions) is PageKind.BORN_DIGITAL


def test_classify_pages_maps_window_to_absolute_pages():
    inspection = SimpleNamespace(pages=(_page(4, invisible=True), _page(5)))
    kinds = classify_pages(inspection, [[_native()], [_native()]], page_indices=[4, 5])
    assert kinds == {4: PageKind.SCAN, 5: PageKind.BORN_DIGITAL}


def test_ensure_page_kinds_computes_once():
    fs = SimpleNamespace(
        page_kinds=None,
        pdf_inspection=SimpleNamespace(pages=(_page(0, chars=0),)),
        layout_results=[[{"label": "text"}]],
        page_indices=[0],
    )
    assert ensure_page_kinds(fs) == {0: PageKind.SCAN}
    assert fs.page_kinds == {0: "scan"}
    fs.pdf_inspection = None
    assert ensure_page_kinds(fs) == {0: PageKind.SCAN}
