"""The document layer: built beside the inspection, which it never changes.

``inspect_pdf(include_doc_layer=True)`` must return the same inspection as
without the layer, byte for byte, and the layer must reproduce what the
pipeline reads (region text, page lines, region boxes) from its own columns.
"""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

from bibr.config import GlobalSettings
from bibr.document import harvest, serialize, views
from bibr.document.harvest import build_document_layer, layout_render_dpi
from bibr.document.model import (
    COLUMN_DTYPES,
    INDEX_FRAME,
    LAYER_VERSION,
    Decided,
    Furniture,
)
from bibr.document.rebuild import _page_count, attach_blocks, render_budget
from bibr.ocr.image_utils import iter_pdf_pages_with_index
from bibr.ocr.pdf_inspection import inspect_pdf, inspection_to_dict
from bibr.ocr.types import OcrRegionResult
from tests.document import _pdfs

_FIXTURES = _pdfs.fixture_pdfs()
_SETTINGS = GlobalSettings()
_BUDGET = render_budget(_SETTINGS)


def _inspect(pdf_bytes: bytes, *, layer: bool, layout=None, **overrides):
    layout = layout if layout is not None else _pdfs.band_layout(_page_count(pdf_bytes))
    kwargs = {
        "fill_native_text": True,
        "include_outline": True,
        "include_ref_geometry": True,
        "min_chars": 3,
        "min_printable_ratio": 0.85,
        "reject_invisible_text_layer": True,
    }
    kwargs.update(overrides)
    if layer:
        kwargs.update(include_doc_layer=True, render_budget=_BUDGET)
    return inspect_pdf(pdf_bytes, deepcopy(layout), **kwargs)


def _regions(layout_results: list[list[dict]]) -> list[list[OcrRegionResult]]:
    return [
        [
            OcrRegionResult.from_layout_region(region, slot_idx=slot, content=region["content"])
            for slot, region in enumerate(page)
        ]
        for page in layout_results
    ]


def _synthetic_layer():
    inspection = _inspect(_pdfs.synthetic_paper(), layer=True)
    return inspection, inspection.document


def _line_index(page, text: str) -> int:
    for index in range(len(page.cols.line_span)):
        if views.line_text(page, index) == text:
            return index
    raise AssertionError(f"no line {text!r}")


# --- Byte identity ------------------------------------------------------------


@pytest.mark.parametrize("reject", [False, True])
@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_inspection_is_byte_identical_with_the_layer_on(name, reject):
    off = _inspect(_FIXTURES[name], layer=False, reject_invisible_text_layer=reject)
    on = _inspect(_FIXTURES[name], layer=True, reject_invisible_text_layer=reject)

    assert off.document is None
    assert on.document is not None
    assert on.document.component_errors == {}
    assert json.dumps(inspection_to_dict(on)) == json.dumps(inspection_to_dict(off))
    assert json.dumps(on.layout_results) == json.dumps(off.layout_results)
    assert on.page_lines == off.page_lines
    assert on.reference_lines == off.reference_lines
    assert on.uri_links == off.uri_links
    assert on.metadata == off.metadata
    assert on.outline == off.outline
    assert on.component_errors == off.component_errors
    assert on == off


def test_a_harvest_failure_stays_in_the_layer(monkeypatch):
    pdf_bytes = _pdfs.synthetic_paper()
    off = _inspect(pdf_bytes, layer=False)

    def broken(*_args, **_kwargs):
        raise RuntimeError("objects unreadable")

    monkeypatch.setattr(harvest, "_read_objects", broken)
    on = _inspect(pdf_bytes, layer=True)

    assert json.dumps(inspection_to_dict(on)) == json.dumps(inspection_to_dict(off))
    assert on == off
    errors = on.document.component_errors
    text_pages = [i for i, source in enumerate(_pdfs.SYNTHETIC_TEXT_SOURCES) if source != "ocr"]
    assert sorted(errors) == sorted(f"harvest:{i}" for i in text_pages)
    assert all(value == "RuntimeError: objects unreadable" for value in errors.values())
    assert [page.index for page in on.document.pages] == [5]


def test_a_failure_to_finish_the_layer_leaves_the_inspection(monkeypatch):
    pdf_bytes = _pdfs.synthetic_paper()
    off = _inspect(pdf_bytes, layer=False)

    def broken(_self):
        raise RuntimeError("finish failed")

    monkeypatch.setattr(harvest.LayerBuilder, "finish", broken)
    on = _inspect(pdf_bytes, layer=True)

    assert on.document is None
    assert json.dumps(inspection_to_dict(on)) == json.dumps(inspection_to_dict(off))


# --- Parity with the pipeline's own reads ---------------------------------------


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_block_text_and_boxes_reproduce_the_native_fill(name):
    inspection = _inspect(_FIXTURES[name], layer=True)
    layer = inspection.document
    attach_blocks(layer, _regions(inspection.layout_results))

    filled = 0
    for page_index, regions in enumerate(inspection.layout_results):
        page = layer.page(page_index)
        for slot, region in enumerate(regions):
            block = views.block_for_region(layer, page_index + 1, slot)
            assert block is not None and block.block_id == f"p{page_index}.r{slot}"
            assert views.bbox_pdf_pts(page, block) == region["_bbox_pdf_pts"]
            if region.get("_native_text_used"):
                filled += 1
                assert views.block_text(layer, block.block_id) == region["content"]
                assert block.text == {block.chosen: region["content"]}
                assert block.chosen == "native"
    if name in {"synthetic_paper.pdf", "sample_paper.pdf", "native_text_sample.pdf"}:
        assert filled > 0


@pytest.mark.parametrize("reject", [False, True])
@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_page_lines_reproduce_the_reference_line_stream(name, reject):
    inspection = _inspect(_FIXTURES[name], layer=True, reject_invisible_text_layer=reject)
    by_page: dict[int, list[dict]] = {}
    for line in inspection.page_lines:
        by_page.setdefault(line["page"], []).append(line)

    for page in inspection.document.pages:
        if reject and page.text_source == "invisible_layer":
            # The inspection reads no lines from a rejected hidden layer.
            assert page.index + 1 not in by_page
            continue
        assert views.page_lines(page) == by_page.get(page.index + 1, [])


def test_invisible_layer_text_is_read_only_on_scanned_pages():
    inspection = _inspect(_pdfs.synthetic_paper(), layer=True, reject_invisible_text_layer=False)
    layer = inspection.document
    attach_blocks(layer, _regions(inspection.layout_results))

    scan = layer.page(3)
    born_digital = layer.page(0)
    assert scan.text_source == "invisible_layer"
    for block in scan.blocks:
        if block.chosen is not None:
            assert block.chosen == "invisible_layer"
            assert (
                views.block_text(layer, block.block_id, "invisible_layer")
                == block.text["invisible_layer"]
            )
    for block in born_digital.blocks:
        assert views.block_text(layer, block.block_id, "invisible_layer") is None
    with pytest.raises(ValueError):
        views.block_text(layer, born_digital.blocks[0].block_id, "layout")


# --- Rebuild, determinism, serialisation ------------------------------------------


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_rebuild_from_the_bytes_matches_the_inline_layer(name):
    pdf_bytes = _FIXTURES[name]
    inline = _inspect(pdf_bytes, layer=True).document
    rebuilt = build_document_layer(pdf_bytes, range(_page_count(pdf_bytes)), budget=_BUDGET)

    assert serialize.canonical_bytes(rebuilt) == serialize.canonical_bytes(inline)


def test_rebuild_of_a_page_range_matches_the_inline_layer():
    pdf_bytes = _pdfs.synthetic_paper()
    pages = [2, 3, 4]
    inline = _inspect(
        pdf_bytes, layer=True, layout=_pdfs.band_layout(len(pages)), page_indices=pages
    ).document
    rebuilt = build_document_layer(pdf_bytes, pages, budget=_BUDGET)

    assert [page.index for page in inline.pages] == pages
    assert serialize.digest(rebuilt) == serialize.digest(inline)


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_layer_is_deterministic_and_round_trips(name):
    first = _inspect(_FIXTURES[name], layer=True).document
    second = _inspect(_FIXTURES[name], layer=True).document
    text = serialize.canonical_bytes(first)
    restored = serialize.from_dict(json.loads(text))

    assert serialize.canonical_bytes(second) == text
    assert serialize.canonical_bytes(restored) == text
    assert first.version == LAYER_VERSION
    assert first.index_frame == INDEX_FRAME
    for page in first.pages:
        if page.cols is None:
            continue
        for column, dtype in COLUMN_DTYPES.items():
            assert getattr(page.cols, column).dtype == np.dtype(dtype), column


# --- What the layer reads ---------------------------------------------------------


def test_text_sources_and_presence():
    _inspection, layer = _synthetic_layer()

    assert [page.text_source for page in layer.pages] == _pdfs.SYNTHETIC_TEXT_SOURCES
    assert layer.page(5).cols is None
    presence = layer.presence
    assert presence.has_text_layer is True
    assert presence.is_scan is False
    assert presence.has_invisible_layer is True
    assert presence.has_mcids is True
    assert presence.missing_apis == ()
    # D2 and D3 facts are not read yet.
    assert presence.has_outline is None and presence.is_tagged is None
    assert layer.links == [] and layer.struct == [] and layer.outline == []


def test_a_scan_without_a_text_layer_is_flagged_as_one():
    layer = _inspect(_FIXTURES["scanned_sample.pdf"], layer=True).document

    assert all(page.text_source == "ocr" and page.cols is None for page in layer.pages)
    assert layer.presence.has_text_layer is False
    assert layer.presence.is_scan is True
    assert layer.presence.has_invisible_layer is False


def test_effective_size_is_the_nominal_size_times_the_matrix_scale():
    _inspection, layer = _synthetic_layer()
    first = layer.page(0)
    scaled = int(
        first.cols.span_obj[first.cols.line_span[_line_index(first, _pdfs.SCALED_LINE)][0]]
    )
    form_page = layer.page(4)

    assert first.cols.obj_tf[scaled] == 1.0
    assert first.cols.obj_size_eff[scaled] == pytest.approx(_pdfs.SCALED_MATRIX)
    assert form_page.cols.obj_tf.tolist() == [_pdfs.FORM_SIZE]
    assert form_page.cols.obj_size_eff.tolist() == [
        pytest.approx(_pdfs.FORM_SIZE * _pdfs.FORM_SCALE)
    ]


def test_superscripts_and_subscripts_are_tagged():
    _inspection, layer = _synthetic_layer()
    page = layer.page(0)
    tagged = []
    for tag in layer.roles:
        page_index, span = tag.target.removeprefix("p").split(".sp")
        assert int(page_index) == 0
        line = int(tag.decided.evidence[0].split(".l")[1])
        tagged.append((tag.role, views.span_text(page, int(span)), views.line_text(page, line)))
        assert tag.decided.component == "span_rules"
        assert tag.decided.version == "superscript/1"
        assert tag.decided.calibrated is False

    superscript_line = "".join(_pdfs.SUPERSCRIPT_LINE)
    subscript_line = "".join(_pdfs.SUBSCRIPT_LINE)
    assert tagged == [
        ("superscript", "12", superscript_line),
        ("subscript", "2", subscript_line),
    ]
    scores = [tag.decided.score for tag in layer.roles]
    assert scores == [
        pytest.approx(_pdfs.SUPERSCRIPT_RAISE / _pdfs.BODY_SIZE),
        pytest.approx(-_pdfs.SUBSCRIPT_DROP / _pdfs.BODY_SIZE),
    ]


def test_fonts_colours_and_marked_content():
    _inspection, layer = _synthetic_layer()
    page = layer.page(0)
    cols = page.cols

    def obj_of(text: str) -> int:
        return int(cols.span_obj[cols.line_span[_line_index(page, text)][0]])

    names = {font.base_name for font in layer.fonts}
    assert names == {"Helvetica", "Helvetica-Bold", "Times-Roman"}
    assert len(layer.fonts) == len({font.font_id for font in layer.fonts})
    heading = obj_of(_pdfs.HEADING)
    assert layer.fonts[int(cols.obj_font[heading])].base_name == "Helvetica-Bold"

    red, tagged, header = obj_of(_pdfs.RED_LINE), obj_of(_pdfs.TAGGED_LINE), obj_of(_pdfs.HEADER)
    assert cols.obj_fill[red] == 0xFF0000FF
    assert {int(cols.obj_fill[i]) for i in range(len(cols.obj_fill)) if i != red} == {0x000000FF}
    assert cols.obj_mcid[tagged] == _pdfs.MCID
    assert {int(cols.obj_mcid[i]) for i in range(len(cols.obj_mcid)) if i != tagged} == {-1}
    assert cols.obj_artifact.tolist() == [i == header for i in range(len(cols.obj_artifact))]
    assert set(cols.obj_flags.tolist()) == {0}


def test_the_removed_watermark_is_kept_as_furniture():
    _inspection, layer = _synthetic_layer()
    page = layer.page(6)

    assert len(page.furniture) == 1
    found = page.furniture[0]
    assert found == Furniture(
        6,
        "watermark",
        found.bbox_pdf,
        "For Review Only",
        Decided("furniture.watermark", "watermark/1"),
    )
    left, bottom, right, top = found.bbox_pdf
    assert 0 <= left < right <= _pdfs.PAGE_W and 0 <= bottom < top <= _pdfs.PAGE_H
    assert not any(
        "Review" in views.line_text(page, line) for line in range(len(page.cols.line_span))
    )
    assert all(not other.furniture for other in layer.pages if other.index != 6)


def test_render_recipe_mirrors_the_layout_render():
    # A small budget, so a letter page renders at a reduced DPI and a stamp-sized
    # rotated page at the full one.
    budget = harvest.RenderBudget(dpi=200, max_pixels=1_000_000, max_dimension=2_000, min_dpi=24)
    pdf_bytes = _pdfs.build_pdf(
        [
            _pdfs.PageSpec(_pdfs.text("Letter page", 72.0, 700.0)),
            _pdfs.PageSpec(b"", media=(0.0, 0.0, 144.0, 216.0), rotate=90),
        ]
    )
    layer = build_document_layer(pdf_bytes, range(2), budget=budget)
    reduced: dict[int, int] = {}
    sizes = {
        index: image.size
        for index, image in iter_pdf_pages_with_index(
            pdf_bytes,
            budget.dpi,
            None,
            None,
            budget.max_pixels,
            budget.max_dimension,
            min_dpi=budget.min_dpi,
            on_reduced_dpi=lambda index, dpi: reduced.__setitem__(index, dpi),
        )
    }

    assert sorted(reduced) == [0]
    assert [page.render.dpi for page in layer.pages] == [reduced[0], budget.dpi]
    for page in layer.pages:
        recipe = page.render
        assert recipe.crop_box == page.crop_box and recipe.rotation == page.rotation
        assert recipe.pdfium == layer.pdfium
        assert recipe.rgb_sha256 is None
        scale = recipe.dpi / 72
        assert sizes[page.index] == (
            int(np.ceil(page.width * scale)),
            int(np.ceil(page.height * scale)),
        )
    assert layout_render_dpi(1e6, 1e6, _BUDGET) is None


def test_lines_go_to_the_blocks_that_hold_them():
    inspection, layer = _synthetic_layer()
    attach_blocks(layer, _regions(inspection.layout_results))

    for page in layer.pages:
        if page.cols is None:
            continue
        line_block = page.cols.line_block.tolist()
        # The bands cover the page, so every line lands in one of them.
        assert all(position >= 0 for position in line_block)
        for position, block in enumerate(page.blocks):
            assert list(block.lines) == [
                line for line, owner in enumerate(line_block) if owner == position
            ]
    first = layer.page(0)
    heading_block = first.blocks[first.cols.line_block[_line_index(first, _pdfs.HEADING)]]
    assert _pdfs.HEADING in heading_block.text["native"]
