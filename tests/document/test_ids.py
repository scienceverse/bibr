"""Layer ids and the public frame conversion."""

from __future__ import annotations

import pytest

from bibr.document import ids, views
from bibr.document.model import Page
from bibr.document.rebuild import attach_blocks
from bibr.ocr.types import OcrRegionResult


@pytest.mark.parametrize(
    ("made", "expected"),
    [
        (ids.span(3, 210), "p3.sp210"),
        (ids.line(3, 40), "p3.l40"),
        (ids.block(0, 12), "p0.r12"),
        (ids.furniture(7, 0), "p7.f0"),
        (ids.make(2, "lk", 17), "p2.lk17"),
    ],
)
def test_ids_are_made_and_parsed_back(made, expected):
    parsed = ids.parse(made)

    assert made == expected
    assert str(parsed) == made
    assert ids.make(parsed.page, parsed.kind, parsed.n) == made


def test_parse_reads_page_kind_and_index():
    assert ids.parse("p12.sp3") == ids.LayerId(12, "sp", 3)
    assert ids.parse("p0.lk17") == ids.LayerId(0, "lk", 17)


@pytest.mark.parametrize(
    "text", ["", "p3", "p3.r", "3.r1", "p-1.r1", "p03.r1", "p3.r01", "p3.R1", "p3.r1x", " p3.r1"]
)
def test_parse_rejects_what_is_not_an_id(text):
    with pytest.raises(ValueError):
        ids.parse(text)


@pytest.mark.parametrize(
    ("page", "kind", "n"), [(-1, "r", 0), (0, "r", -1), (0, "R", 0), (0, "", 0)]
)
def test_make_rejects_what_parse_could_not_read(page, kind, n):
    with pytest.raises(ValueError):
        ids.make(page, kind, n)


def test_a_document_id_names_no_page():
    parsed = ids.parse("ol5")

    assert parsed == ids.LayerId(None, "ol", 5)
    assert parsed.path == (5,)
    assert str(parsed) == ids.outline_entry(5) == ids.make_document("ol", 5) == "ol5"


def test_an_outline_entry_id_is_one_number_and_a_structure_id_a_path():
    assert ids.parse("ol0") == ids.LayerId(None, "ol", 0)
    assert ids.parse("ol12").path == (12,)
    # A path of one level is a structure id too, and a deeper one is not an outline id.
    assert ids.parse("st0") == ids.LayerId(None, "st", 0)
    assert ids.parse("st1.2").path == (1, 2)
    assert ids.make_document("st", 1, 2) == "st1.2"
    with pytest.raises(ValueError):
        ids.make_document("ol", 1, 2)


def test_a_structure_element_id_is_its_path_in_the_tree():
    made = ids.struct_element((0, 3, 2))
    parsed = ids.parse(made)

    assert made == ids.make_document("st", 0, 3, 2) == "st0.3.2"
    assert parsed == ids.LayerId(None, "st", 2, (0, 3))
    assert parsed.path == (0, 3, 2)
    assert str(parsed) == made


def test_the_path_of_a_page_id_is_its_index():
    assert ids.parse("p2.lk17").path == (17,)


@pytest.mark.parametrize(
    "text",
    [
        *["ol", "ol05", "ol-1", "ol5.", "ol.5", "ol5x", "OL5", " ol5", "xx5", "r5", "p3", "p3.ol5"],
        *["st", "st0.", "st0..1", "st.1", "st0.01", "st0.3x", "st0 .3", "p3.st5"],
        # An outline entry is one number: the tree is in OutlineEntry.parent.
        *["ol1.2", "ol0.0", "ol1.2.3", "ol5.0"],
    ],
)
def test_parse_rejects_what_is_not_a_document_id_either(text):
    with pytest.raises(ValueError):
        ids.parse(text)


@pytest.mark.parametrize(
    "args",
    [("r", 5), ("ol",), ("ol", -1), ("OL", 1), ("", 1), ("st",), ("st", 0, -1)]
    + [("ol", 1, 2), ("ol", 0, 0), ("ol", 1, 2, 3)],
)
def test_make_document_rejects_what_parse_could_not_read(args):
    with pytest.raises(ValueError):
        ids.make_document(*args)


@pytest.mark.parametrize("kind", ["ol", "st"])
def test_a_kind_is_a_page_kind_or_a_document_kind_never_both(kind):
    with pytest.raises(ValueError):
        ids.make(3, kind, 5)


def test_a_document_id_names_no_block():
    layer = _layer_of(_page(0))

    assert views.find_block(layer, "ol0") is None


def _page(rotation: int) -> Page:
    return Page(
        index=0,
        label=None,
        width=612.0,
        height=792.0,
        crop_box=(18.0, 36.0, 594.0, 756.0),
        rotation=rotation,
        text_source="native",
        cols=None,
    )


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_layout_frame_conversion_round_trips(rotation):
    page = _page(rotation)
    box = (90.0, 400.5, 300.25, 512.0)

    layout = views.to_layout_bbox(page, box)
    back = views.from_layout_bbox(page, layout)

    assert all(0.0 <= value <= 1000.0 for value in layout)
    assert back == pytest.approx(box)


@pytest.mark.parametrize("rotation", [0, 90])
def test_blocks_sit_where_their_regions_are(rotation):
    page = _page(rotation)
    bbox_2d = [100.0, 250.0, 600.0, 400.0]
    region = OcrRegionResult.from_layout_region(
        {"label": "text", "bbox_2d": bbox_2d, "content": ""}, slot_idx=0, content=""
    )
    layer = _layer_of(page)

    attach_blocks(layer, [[region]])

    block = page.blocks[0]
    assert block.bbox_pdf == views.from_layout_bbox(page, bbox_2d)
    assert views.to_layout_bbox(page, block.bbox_pdf) == pytest.approx(bbox_2d)


def _layer_of(page: Page):
    from bibr.document.model import LAYER_VERSION, DocumentLayer

    return DocumentLayer(
        version=LAYER_VERSION,
        pdfium="test",
        source_sha256="0" * 64,
        index_frame="post_strip",
        pages=[page],
        fonts=[],
    )
