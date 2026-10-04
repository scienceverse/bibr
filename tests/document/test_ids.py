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
    from bibr.document.model import DocumentLayer

    return DocumentLayer(
        version="doclayer/1",
        pdfium="test",
        source_sha256="0" * 64,
        index_frame="post_strip",
        pages=[page],
        fonts=[],
    )
