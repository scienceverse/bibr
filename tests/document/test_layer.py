"""The document layer: built beside the inspection, which it never changes.

``inspect_pdf(include_doc_layer=True)`` must return the same inspection as
without the layer, byte for byte, and the layer must reproduce what the
pipeline reads (region text, page lines, region boxes) from its own columns.
"""

from __future__ import annotations

import dataclasses
import json
import math
from copy import deepcopy

import numpy as np
import pytest

from bibr.config import GlobalSettings
from bibr.document import harvest, ids, serialize, views
from bibr.document.harvest import build_document_layer, layout_render_dpi
from bibr.document.model import (
    COLUMN_DTYPES,
    GLYPH_GENERATED,
    GLYPH_NO_BOX,
    INDEX_FRAME,
    LAYER_VERSION,
    Decided,
    Furniture,
    Page,
)
from bibr.document.rebuild import attach_blocks, render_budget
from bibr.ocr.image_utils import iter_pdf_pages_with_index
from bibr.ocr.pdf_inspection import inspect_pdf, inspection_to_dict
from bibr.ocr.types import OcrRegionResult
from tests.document import _pdfs

_FIXTURES = _pdfs.fixture_pdfs()
_SETTINGS = GlobalSettings()
_BUDGET = render_budget(_SETTINGS)


def _inspect(pdf_bytes: bytes, *, layer: bool, layout=None, **overrides):
    layout = layout if layout is not None else _pdfs.band_layout(_pdfs.page_count(pdf_bytes))
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


def test_pages_the_inspection_does_not_read_are_harvested_alike():
    # With neither the native fill nor reference geometry, the inspection opens
    # no text page; the layer reads each page through the same furniture strip.
    pdf_bytes = _pdfs.synthetic_paper()
    off = _inspect(pdf_bytes, layer=False, fill_native_text=False, include_ref_geometry=False)
    on = _inspect(pdf_bytes, layer=True, fill_native_text=False, include_ref_geometry=False)
    rebuilt = build_document_layer(pdf_bytes, range(_pdfs.page_count(pdf_bytes)), budget=_BUDGET)

    assert json.dumps(inspection_to_dict(on)) == json.dumps(inspection_to_dict(off))
    assert on == off
    assert on.document.component_errors == {}
    assert serialize.digest(on.document) == serialize.digest(rebuilt)
    assert [page.text_source for page in on.document.pages] == _pdfs.SYNTHETIC_TEXT_SOURCES


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
    # Every page stays; the failed ones are unread and say why.
    pages = on.document.pages
    assert [page.index for page in pages] == list(range(len(_pdfs.SYNTHETIC_TEXT_SOURCES)))
    for page in pages:
        if page.index in text_pages:
            assert page.text_source == "unread"
            assert page.cols is None
            assert page.error == "RuntimeError: objects unreadable"
            assert math.isnan(page.width)
        else:
            assert page.text_source == "ocr"
            assert page.error is None
    # What no read page shows is unknown, not absent.
    presence = on.document.presence
    assert presence.has_text_layer is None
    assert presence.is_scan is None
    assert presence.has_invisible_layer is None
    assert presence.has_mcids is None


def test_a_page_that_fails_after_its_read_keeps_what_was_read(monkeypatch):
    pdf_bytes = _pdfs.synthetic_paper()
    whole = _inspect(pdf_bytes, layer=True).document

    def broken(*_args, **_kwargs):
        raise RuntimeError("no spans")

    monkeypatch.setattr(harvest, "_spans_and_lines", broken)
    layer = _inspect(pdf_bytes, layer=True).document

    assert [page.index for page in layer.pages] == [page.index for page in whole.pages]
    for page, read in zip(layer.pages, whole.pages, strict=True):
        assert page.text_source == read.text_source
        assert page.text_source_decided == read.text_source_decided
        assert (page.width, page.crop_box, page.furniture) == (
            read.width,
            read.crop_box,
            read.furniture,
        )
        assert page.cols is None
        failed = read.cols is not None
        assert page.error == ("RuntimeError: no spans" if failed else None)
        assert (f"harvest:{page.index}" in layer.component_errors) is failed


def test_a_page_the_rebuild_cannot_open_is_kept_once():
    pdf_bytes = _pdfs.synthetic_paper()
    builder = harvest.LayerBuilder(pdf_bytes, _BUDGET)
    builder.page_failed(7, "page:7", RuntimeError("first"))
    builder.page_failed(7, "other:7", RuntimeError("second"))
    layer = builder.finish()

    assert [(page.index, page.text_source, page.error) for page in layer.pages] == [
        (7, "unread", "RuntimeError: first")
    ]
    assert layer.component_errors == {
        "page:7": "RuntimeError: first",
        "other:7": "RuntimeError: second",
    }

    rebuilt = build_document_layer(pdf_bytes, [0, 99], budget=_BUDGET)
    assert [page.index for page in rebuilt.pages] == [0, 99]
    assert rebuilt.page(99).text_source == "unread"
    assert rebuilt.page(99).error == rebuilt.component_errors["page:99"]
    assert rebuilt.presence.has_text_layer is True
    assert rebuilt.presence.is_scan is False


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
            block = views.block_for_region(layer, page_no=page_index + 1, region_index=slot)
            assert block is not None and block.block_id == f"p{page_index}.r{slot}"
            assert views.bbox_pdf_pts(page, block) == region["_bbox_pdf_pts"]
            if region.get("_native_text_used"):
                filled += 1
                assert views.block_text(layer, block.block_id) == region["content"]
                assert block.text == {block.chosen: region["content"]}
                assert block.chosen == "native"
    if name in {"synthetic_paper.pdf", "sample_paper.pdf", "native_text_sample.pdf"}:
        assert filled > 0


def test_every_region_gets_a_block_at_its_position():
    _inspection, layer = _synthetic_layer()
    page = layer.page(0)
    box = [0.0, 0.0, 1000.0, 500.0]
    regions = [
        OcrRegionResult.from_layout_region(
            {"label": "text", "bbox_2d": box}, slot_idx=0, content="first"
        ),
        # No layout box.
        OcrRegionResult.from_layout_region({"label": "text"}, slot_idx=1, content="second"),
        # A region index seen before.
        OcrRegionResult.from_layout_region(
            {"label": "text", "bbox_2d": box}, slot_idx=0, content="third"
        ),
    ]

    attach_blocks(layer, [regions])

    assert [block.block_id for block in page.blocks] == ["p0.r0", "p0.r1", "p0.r2"]
    assert [block.region_index for block in page.blocks] == [0, 1, 2]
    for position, block in enumerate(page.blocks):
        assert views.block_for_region(layer, page_no=1, region_index=position) is block
        assert views.find_block(layer, block.block_id) == (page, block)
    unboxed = page.blocks[1]
    assert unboxed.bbox_pdf is None
    assert unboxed.text == {"ocr": "second"}
    assert unboxed.lines == ()
    assert views.bbox_pdf_pts(page, unboxed) is None
    assert views.block_text(layer, unboxed.block_id) is None
    assert views.block_text(layer, unboxed.block_id, "ocr") == "second"
    assert views.block_for_region(layer, page_no=1, region_index=3) is None
    assert views.block_for_region(layer, page_no=0, region_index=0) is None
    assert views.find_block(layer, "p0.r3") is None
    assert views.find_block(layer, "p99.r0") is None
    assert views.find_block(layer, "r0") is None


def test_the_region_lookup_takes_its_page_by_keyword_only():
    _inspection, layer = _synthetic_layer()

    with pytest.raises(TypeError):
        views.block_for_region(layer, 1, 0)


def test_pages_are_found_in_a_layer_with_gaps():
    pdf_bytes = _pdfs.synthetic_paper()
    layer = build_document_layer(pdf_bytes, [1, 3, 4], budget=_BUDGET)

    assert [page.index for page in layer.pages] == [1, 3, 4]
    assert [layer.page(index) for index in (1, 3, 4)] == layer.pages
    assert layer.page(0) is None
    assert layer.page(2) is None
    assert layer.page(5) is None


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


def test_page_lines_drop_chars_beyond_the_bmp_as_the_line_stream_does():
    # pdfium reads such a char one UTF-16 unit at a time: the high surrogate
    # alone, which decodes to nothing, so the line stream leaves it out.
    line = f"Let {_pdfs.MATH_ALPHA_CODE} be the angle, and {_pdfs.MATH_ALPHA_CODE} is small"
    pdf = _pdfs.build_pdf([_pdfs.PageSpec(_pdfs.text(line, 72.0, 700.0, size=12.0, font="F4"))])
    inspection = _inspect(pdf, layer=True)
    page = inspection.document.page(0)

    assert page.cols.cp.tolist().count(ord(_pdfs.MATH_ALPHA)) == 2
    lines = [found for found in inspection.page_lines if found["page"] == 1]
    assert lines
    assert views.page_lines(page) == lines
    assert not any(_pdfs.MATH_ALPHA in json.dumps(found, ensure_ascii=False) for found in lines)
    assert views.line_text(page, 0) == line.replace(_pdfs.MATH_ALPHA_CODE, _pdfs.MATH_ALPHA)


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
    assert scan.blocks
    for block in scan.blocks:
        # The hidden layer is third-party OCR, never born-digital text.
        assert views.block_text(layer, block.block_id) is None
        assert views.block_text(layer, block.block_id, "native") is None
    for block in born_digital.blocks:
        assert views.block_text(layer, block.block_id, "invisible_layer") is None
    with pytest.raises(ValueError):
        views.block_text(layer, born_digital.blocks[0].block_id, "layout")


# --- Rebuild, determinism, serialisation ------------------------------------------


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_rebuild_from_the_bytes_matches_the_inline_layer(name):
    pdf_bytes = _FIXTURES[name]
    inline = _inspect(pdf_bytes, layer=True).document
    rebuilt = build_document_layer(pdf_bytes, range(_pdfs.page_count(pdf_bytes)), budget=_BUDGET)

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
def test_a_packed_draft_unpacks_to_the_lists_read(name, monkeypatch):
    # The builder keeps each page's draft as arrays until finish().
    drafts = []
    read = harvest.read_page

    def keep(*args, **kwargs):
        drafts.append(read(*args, **kwargs))
        return drafts[-1]

    monkeypatch.setattr(harvest, "read_page", keep)
    build_document_layer(_FIXTURES[name], range(_pdfs.page_count(_FIXTURES[name])), budget=_BUDGET)

    assert drafts
    for draft in drafts:
        unpacked = harvest._unpack(harvest._pack(draft))
        for item in dataclasses.fields(draft):
            if item.name == "loose":
                # Kept as the float32 column it becomes.
                expected = np.asarray(draft.loose, dtype=np.float32).tolist()
                assert unpacked.loose == expected
            elif item.name == "objects":
                assert json.dumps(dataclasses.asdict(unpacked.objects)) == json.dumps(
                    dataclasses.asdict(draft.objects)
                )
            else:
                value = getattr(draft, item.name)
                assert json.dumps(getattr(unpacked, item.name), default=repr) == json.dumps(
                    value, default=repr
                ), item.name


def test_packing_keeps_missing_boxes_and_records_of_other_lengths():
    page = Page(0, None, 10.0, 10.0, (0.0, 0.0, 10.0, 10.0), 0, "native", None)
    objects = harvest._Objects(
        [0], [1.0], [2.0], [(2.0, 0.0, 0.0, 2.0)], [0xFF0000FF], [0], [3], [-1], [True], [0]
    )
    nan = float("nan")
    draft = harvest._PageDraft(
        page,
        [0x41, 0xD835, 0xDEFC],
        [(1.0, 2.0, 3.0, 4.0), None, None],
        [("A", 2.0, 3.0, False), ("\U0001d6fc", 5.0, 3.0, False), ("", 0.0, 0.0, False)],
        [0, 1, -1],
        [0, 1, 0],
        [0, 0, -1],
        [1.0, 2.0, nan, nan, 0.5, 0.5],
        [1.0, 2.0, 3.0, 4.0] * 3,
        [0, GLYPH_NO_BOX, GLYPH_NO_BOX | GLYPH_GENERATED],
        objects,
    )

    unpacked = harvest._unpack(harvest._pack(draft))

    assert repr(unpacked) == repr(draft)


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_layer_is_deterministic_and_round_trips(name):
    first = _inspect(_FIXTURES[name], layer=True).document
    second = _inspect(_FIXTURES[name], layer=True).document
    text = serialize.canonical_bytes(first)
    restored = serialize.from_dict(json.loads(text))

    assert serialize.canonical_bytes(second) == text
    assert serialize.canonical_bytes(restored) == text
    assert first.version == LAYER_VERSION
    assert first.index_frame == INDEX_FRAME == "post_strip:watermark/1+line_number/1"
    for page in first.pages:
        if page.cols is None:
            continue
        for column, dtype in COLUMN_DTYPES.items():
            assert getattr(page.cols, column).dtype == np.dtype(dtype), column


@pytest.mark.parametrize(
    ("key", "value"),
    [("version", "doclayer/0"), ("index_frame", "post_strip"), ("@", "Page"), (None, None)],
)
def test_a_layer_of_another_version_or_frame_is_not_loaded(key, value):
    data = json.loads(
        serialize.canonical_bytes(_inspect(_pdfs.synthetic_paper(), layer=True).document)
    )
    if key is None:
        data = [data]
    else:
        data[key] = value

    with pytest.raises(ValueError):
        serialize.from_dict(data)


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


@pytest.mark.parametrize("name", ["synthetic_paper.pdf", "scanned_sample.pdf"])
def test_the_text_source_is_the_inspections_invisible_layer_rule(name):
    import pypdfium2

    from bibr.ocr import native_text as nt
    from bibr.ocr.utils import pdfium_lock

    pdf_bytes = _FIXTURES[name]
    layer = _inspect(pdf_bytes, layer=True).document
    found = []
    with pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            for page in layer.pages:
                pdf_page = doc[page.index]
                textpage = nt.open_text_page(pdf_page)
                try:
                    if textpage.count_chars():
                        share = nt._invisible_text_share(textpage)
                        flagged = nt._is_invisible_text_layer_page(
                            pdf_page, textpage, page.crop_box
                        )
                        found.append((page.index, share, flagged))
                finally:
                    textpage.close()
                    pdf_page.close()
        finally:
            doc.close()

    with_text = {index for index, _share, _flagged in found}
    for page in layer.pages:
        if page.index not in with_text:
            assert page.text_source == "ocr"
            assert page.text_source_decided is None
    for index, share, flagged in found:
        page = layer.page(index)
        assert page.text_source_decided == Decided("text_source", "invisible_layer/1", score=share)
        assert (page.text_source == "invisible_layer") is flagged
    if name == "synthetic_paper.pdf":
        assert layer.page(3).text_source_decided.score >= 0.5
        assert layer.page(0).text_source_decided.score == 0.0


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
        furniture_id="p6.f0",
        page=6,
        kind="watermark",
        bbox_pdf=found.bbox_pdf,
        text="For Review Only",
        decided=Decided("furniture.watermark", "watermark/1"),
    )
    # The rule that removed it is part of the key of the page's glyph indexes.
    assert found.decided.version in layer.index_frame
    left, bottom, right, top = found.bbox_pdf
    assert 0 <= left < right <= _pdfs.PAGE_W and 0 <= bottom < top <= _pdfs.PAGE_H
    assert not any(
        "Review" in views.line_text(page, line) for line in range(len(page.cols.line_span))
    )
    assert all(not other.furniture for other in layer.pages if other.index != 6)


def test_removed_line_numbers_get_page_scoped_ids():
    from tests.ocr.test_line_number_column import _reference_page

    layer = _inspect(_reference_page(), layer=True).document

    assert all(page.furniture for page in layer.pages)
    for page in layer.pages:
        assert {item.kind for item in page.furniture} == {"line_number"}
        assert {item.decided.version for item in page.furniture} == {"line_number/1"}
        assert [item.furniture_id for item in page.furniture] == [
            ids.furniture(page.index, n) for n in range(len(page.furniture))
        ]


def test_rule_decisions_do_not_claim_calibration():
    _inspection, layer = _synthetic_layer()
    decided = [item.decided for page in layer.pages for item in page.furniture]
    decided += [tag.decided for tag in layer.roles]

    assert Decided("any_rule", "any_rule/1").calibrated is False
    assert decided
    assert not any(item.calibrated for item in decided)


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
