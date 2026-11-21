"""Regression and exact-equivalence tests for Wave 3C performance optimizations."""

from __future__ import annotations

import base64
import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from docx import Document
from PIL import Image

from bibr.document import harvest as doc_harvest
from bibr.document import links as doc_links
from bibr.document.model import Page, PageColumns
from bibr.input.docx_native import DocxParser
from bibr.input.html_native import HtmlParser, inspect_html
from bibr.input.jats_native import JatsParser
from bibr.input.validate import validate_input_file
from bibr.paper_contents import CaptionCandidate, PaperFigurePart, PaperTable, PaperTablePart
from bibr.structure.caption_matcher import (
    CaptionTarget,
    _maximum_weight_assignment,
    assign_captions,
)
from bibr.structure.float_images import composite_panel_image
from bibr.structure.floats_normalize import merge_table_continuations_with_remap
from bibr.structure.html_table import html_table_frame


def test_html_table_frame_is_readonly_and_handles_hidden_and_br():
    html = (
        "<div><table>"
        "<style>.x{color:red}</style>"
        "<thead><tr><th>H1</th><th style='display:none'>HiddenH</th><th>H2</th></tr></thead>"
        "<tbody>"
        "<tr><td>A<br/>B<span style='display:none'>secret</span></td><td>C</td></tr>"
        "<tr style='display: none'><td>SkipRow1</td><td>SkipRow2</td></tr>"
        "<tr><td colspan='2'>Span<br>Cell</td></tr>"
        "</tbody>"
        "</table></div>"
    )
    soup = BeautifulSoup(html, "html5lib")
    table_tag = soup.find("table")
    assert table_tag is not None
    before_html = str(table_tag)

    df = html_table_frame(table_tag)
    assert df is not None
    # Original BeautifulSoup Tag must not be mutated (no decompose / replace_with).
    assert str(table_tag) == before_html
    assert list(df.columns) == ["H1", "H2"]
    assert df.values.tolist() == [["A B", "C"], ["Span Cell", "Span Cell"]]

    # Parsing from string yields the exact same DataFrame.
    df_str = html_table_frame(html)
    assert df_str is not None
    assert df_str.equals(df)


def test_html_native_inline_runs_do_not_mutate_dom_and_preserve_bold_and_links():
    html = (
        b"<html><body><article>"
        b"<div><b>Bold Heading</b><div>Body text with "
        b'<a href="https://example.org/a">first link</a> and '
        b'<em>emphasis</em> and <a href="https://example.org/b">second</a>.</div>'
        b"</div></article></body></html>"
    )
    has_content, soup = inspect_html(html)
    assert has_content and soup is not None
    before_dom = str(soup)

    parser = HtmlParser(html, parsed_soup=soup)
    contents = parser.parse()
    segs = [[e.text] for e in parser.assembler.entries]
    parser.apply_segmentation(contents, segs)

    # Read-only _InlineRun leaves the input soup untouched.
    assert str(soup) == before_dom
    assert any(sec.header == "Bold Heading" for sec in contents.sections)
    urls = [lk.url for lk in contents.links]
    assert urls == ["https://example.org/a", "https://example.org/b"]


def test_jats_validate_reuses_parsed_root_with_exact_parity():
    jats_path = Path("tests/fixtures/jats/PMC4383902.xml")
    xml_bytes = jats_path.read_bytes()
    vf = validate_input_file(jats_path, xml_bytes)
    assert vf.is_valid
    assert vf.native_artifact is not None

    p_reused = JatsParser(xml_bytes, parsed_root=vf.native_artifact)
    c_reused = p_reused.parse()

    p_fresh = JatsParser(xml_bytes)
    c_fresh = p_fresh.parse()

    assert c_reused.detected_title == c_fresh.detected_title
    assert [s.header for s in c_reused.sections] == [s.header for s in c_fresh.sections]
    assert [e.text for e in p_reused.assembler.entries] == [
        e.text for e in p_fresh.assembler.entries
    ]
    assert c_reused.preparsed_metadata == c_fresh.preparsed_metadata


def test_docx_validate_single_open_and_cached_plain_parity():
    buf = io.BytesIO()
    d = Document()
    d.add_heading("Study Title", level=1)
    d.add_heading("Methods", level=2)
    d.add_paragraph("Sample paragraph one.")
    d.add_paragraph("Sample paragraph two.")
    d.save(buf)
    raw = buf.getvalue()

    vf = validate_input_file(Path("sample.docx"), raw)
    assert vf.is_valid and not vf.is_corrupted and not vf.is_encrypted

    p = DocxParser(raw)
    c = p.parse()
    assert c.detected_title == "Study Title"
    assert [e.text for e in p.assembler.entries] == [
        "Sample paragraph one.",
        "Sample paragraph two.",
    ]


def test_document_layer_complete_packed_matches_complete_unpack_bit_for_bit():
    n_g = 120
    codes = [ord("A") + (k % 26) if k % 7 else 32 for k in range(n_g)]
    boxes = [
        None
        if k == 15
        else (
            50.0 + (k % 30) * 8.0,
            700.0 - (k // 30) * 16.0,
            57.0 + (k % 30) * 8.0,
            710.0 - (k // 30) * 16.0,
        )
        for k in range(n_g)
    ]
    records = [
        (
            "fi" if k == 5 else ("\n" if k % 30 == 29 else chr(codes[k])),
            53.0 + (k % 30) * 8.0,
            705.0 - (k // 30) * 16.0,
            k % 30 == 29,
        )
        for k in range(n_g)
    ]
    rec_src = list(range(n_g))
    rec_flags = [0] * n_g
    rows = [k // 10 for k in range(n_g)]
    origins = []
    loose = []
    gflags = []
    for b in boxes:
        if b is None:
            origins.extend((float("nan"), float("nan")))
            loose.extend((float("nan"),) * 4)
            gflags.append(doc_harvest.GLYPH_NO_BOX)
        else:
            origins.extend((b[0], b[1] + (3.0 if b[0] > 200 else 0.0)))
            loose.extend(b)
            gflags.append(0)
    n_obj = n_g // 10
    objs = doc_harvest._Objects(
        font=[0] * n_obj,
        tf=[10.0] * n_obj,
        size_eff=[7.0 if j % 4 == 3 else 10.0 for j in range(n_obj)],
        matrix=[(1.0, 0.0, 0.0, 1.0)] * n_obj,
        fill=[0xFF] * n_obj,
        stroke=[0] * n_obj,
        mode=[0] * n_obj,
        mcid=[-1] * n_obj,
        artifact=[False] * n_obj,
        flags=[0] * n_obj,
    )

    def _mk_page():
        return Page(
            index=0,
            label="1",
            width=612.0,
            height=792.0,
            crop_box=(0.0, 0.0, 612.0, 792.0),
            rotation=0,
            text_source="native",
            cols=None,
        )

    packed_a = doc_harvest._pack(
        doc_harvest._PageDraft(
            _mk_page(), codes, boxes, records, rec_src, rec_flags, rows, origins, loose, gflags, objs
        )
    )
    packed_b = doc_harvest._pack(
        doc_harvest._PageDraft(
            _mk_page(), codes, boxes, records, rec_src, rec_flags, rows, origins, loose, gflags, objs
        )
    )

    page_ref, tags_ref = doc_harvest._complete(doc_harvest._unpack(packed_a))
    page_fast, tags_fast = doc_harvest._complete_packed(packed_b)

    assert tags_fast == tags_ref
    assert page_fast.cols is not None and page_ref.cols is not None
    assert page_fast.cols.rec_text == page_ref.cols.rec_text
    for field_name in doc_harvest.COLUMN_DTYPES:
        a = getattr(page_fast.cols, field_name)
        b = getattr(page_ref.cols, field_name)
        assert a.dtype == b.dtype, field_name
        assert a.shape == b.shape, field_name
        np.testing.assert_array_equal(a, b, err_msg=field_name)


def test_links_spatial_index_matches_covered_spans_with_nans_and_quads():
    n_rec = 200
    rec_cp = np.full(n_rec, ord("x"), dtype=np.uint32)
    rec_cp[10] = 32
    rec_cx = np.linspace(10.0, 500.0, n_rec, dtype=np.float32)
    rec_cy = np.linspace(700.0, 100.0, n_rec, dtype=np.float32)
    rec_cx[25] = np.nan
    rec_cy[26] = np.nan
    span_rec = np.stack(
        [np.arange(0, n_rec, 10, dtype=np.int32), np.arange(10, n_rec + 1, 10, dtype=np.int32)],
        axis=1,
    )
    cols = PageColumns(
        cp=rec_cp,
        box=np.zeros((n_rec, 4), dtype=np.float32),
        loose=np.zeros((n_rec, 4), dtype=np.float32),
        origin=np.zeros((n_rec, 2), dtype=np.float32),
        obj=np.zeros(n_rec, dtype=np.int32),
        gflags=np.zeros(n_rec, dtype=np.uint16),
        obj_font=np.zeros(1, dtype=np.int32),
        obj_tf=np.array([10.0], dtype=np.float32),
        obj_size_eff=np.array([10.0], dtype=np.float32),
        obj_matrix=np.array([[1, 0, 0, 1]], dtype=np.float32),
        obj_fill=np.zeros(1, dtype=np.uint32),
        obj_stroke=np.zeros(1, dtype=np.uint32),
        obj_render_mode=np.zeros(1, dtype=np.int8),
        obj_mcid=np.array([-1], dtype=np.int32),
        obj_artifact=np.zeros(1, dtype=np.bool_),
        obj_flags=np.zeros(1, dtype=np.uint8),
        rec_cp=rec_cp,
        rec_cx=rec_cx,
        rec_cy=rec_cy,
        rec_newline=np.zeros(n_rec, dtype=np.bool_),
        rec_src=np.arange(n_rec, dtype=np.int32),
        rec_flags=np.zeros(n_rec, dtype=np.uint8),
        span_rec=span_rec,
        span_obj=np.zeros(len(span_rec), dtype=np.int32),
        span_bbox=np.zeros((len(span_rec), 4), dtype=np.float32),
        span_baseline=np.zeros(len(span_rec), dtype=np.float32),
        line_span=np.array([[0, len(span_rec)]], dtype=np.int32),
        line_bbox=np.zeros((1, 4), dtype=np.float32),
        line_block=np.array([-1], dtype=np.int32),
        rec_text={},
    )
    page = Page(
        index=0,
        label="1",
        width=612.0,
        height=792.0,
        crop_box=(0.0, 0.0, 612.0, 792.0),
        rotation=0,
        text_source="native",
        cols=cols,
    )
    raws = [
        doc_links.RawLink(
            page=0,
            number=0,
            rect=(20.0, 500.0, 200.0, 690.0),
            quads=(),
            action="uri",
            uri="https://a.example",
            dest_name=None,
            name_source=None,
            target_page=None,
            target_xy=None,
        ),
        doc_links.RawLink(
            page=0,
            number=1,
            rect=(0.0, 0.0, 1.0, 1.0),
            quads=(
                (50.0, 600.0, 120.0, 600.0, 50.0, 660.0, 120.0, 660.0),
                (300.0, 200.0, 450.0, 200.0, 300.0, 350.0, 450.0, 350.0),
            ),
            action="uri",
            uri="https://b.example",
            dest_name=None,
            name_source=None,
            target_page=None,
            target_xy=None,
        ),
    ]
    built = doc_links.build_links(raws, [page])
    for raw, link in zip(raws, built, strict=True):
        assert link.source_span_ids == doc_links.covered_spans(page, raw)


def test_caption_matcher_uncontested_and_contested_exact_optimality():
    # Uncontested fast path matches optimal assignment on random instances.
    rng = random.Random(42)  # noqa: S311
    for _ in range(25):
        n = 12
        rows = [{i: rng.uniform(1.0, 5.0), (i + 1) % n: rng.uniform(0.1, 0.9)} for i in range(n)]
        selected = _maximum_weight_assignment(rows, 2 * n)
        assert selected == list(range(n))

    # Contested case (two rows sharing the same best column) exercises full Hungarian.
    contested_rows = [
        {0: 5.0, 1: 4.5},
        {0: 5.0, 1: 1.0},
    ]
    assert _maximum_weight_assignment(contested_rows, 4) == [1, 0]

    # End-to-end assign_captions with two figures on one page.
    caps = [
        CaptionCandidate("c1", "Figure 1. Top", "figure", 1, (50.0, 250.0, 550.0, 280.0), 1),
        CaptionCandidate("c2", "Figure 2. Mid", "figure", 1, (50.0, 520.0, 550.0, 550.0), 3),
    ]
    targs = [
        CaptionTarget("figure:1", "figure", 1, (50.0, 50.0, 550.0, 240.0), 0),
        CaptionTarget("figure:2", "figure", 1, (50.0, 320.0, 550.0, 500.0), 2),
    ]
    assigned = assign_captions(caps, targs, same_page_only=True)
    assert {a.caption_id: a.object_id for a in assigned} == {
        "c1": "figure:1",
        "c2": "figure:2",
    }


def test_merge_table_continuations_batched_concat_and_headerless_restore():
    t1 = PaperTable(
        table_id=1,
        df=pd.DataFrame({"ColA": ["1"], "ColB": ["2"]}),
        tbl_html="<table>1</table>",
        section_id=1,
        caption="Table 1. Summary",
        page_number=1,
        parts=[PaperTablePart(page_number=1, bbox=None, tbl_html="<table>1</table>", df=pd.DataFrame({"ColA": ["1"], "ColB": ["2"]}))],
    )
    t2 = PaperTable(
        table_id=2,
        df=pd.DataFrame({"Col A": ["3"], "Col B": ["4"]}),
        tbl_html="<table>2</table>",
        section_id=1,
        caption="Table 1 (Continued)",
        page_number=2,
        parts=[PaperTablePart(page_number=2, bbox=None, tbl_html="<table>2</table>", df=pd.DataFrame({"Col A": ["3"], "Col B": ["4"]}))],
    )
    t3 = PaperTable(
        table_id=3,
        df=pd.DataFrame({"5": ["7"], "6": ["8"]}),
        tbl_html="<table>3</table>",
        section_id=1,
        caption="Table 1 (Continued)",
        page_number=3,
        parts=[PaperTablePart(page_number=3, bbox=None, tbl_html="<table>3</table>", df=pd.DataFrame({"5": ["7"], "6": ["8"]}))],
    )
    merged, remap = merge_table_continuations_with_remap([t1, t2, t3])
    assert len(merged) == 1
    assert merged[0].df.values.tolist() == [["1", "2"], ["3", "4"], ["5", "6"], ["7", "8"]]
    assert merged[0].tbl_html == "<table>1</table>\n<table>2</table>\n<table>3</table>"
    assert remap == {"table:1": "table:1", "table:2": "table:1", "table:3": "table:1"}


def test_composite_panel_image_handles_rgb_and_rgba():
    parts = []
    for idx, mode in enumerate(("RGB", "RGBA")):
        color = (100, 150, 200) if mode == "RGB" else (50, 80, 120, 255)
        im = Image.new(mode, (40, 30), color)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        parts.append(
            PaperFigurePart(
                page_number=1,
                bbox=(idx * 50.0, 0.0, idx * 50.0 + 40.0, 30.0),
                image_b64=b64,
            )
        )
    out = composite_panel_image(parts)
    assert out is not None
    img = Image.open(io.BytesIO(base64.b64decode(out)))
    assert img.mode == "RGB"
    assert img.size == (90, 30)
