"""Text-layer reads of hand-built PDFs, checked against the same content laid out another way."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bibr.ocr import text_layer as _text_layer
from bibr.ocr.pdf_inspection import inspect_pdf
from bibr.ocr.text_layer import TextLayerStream, read_text_layer
from tests.document._pdfs import (
    band_layout,
    fixture_pdfs,
    page_count,
)

# --- The text-layer read inside the inspection's page walk ---------------------

_FIXTURES = fixture_pdfs()


def _layout(n_pages: int) -> list[list[dict]]:
    """Text bands with a header, a table and a formula on every page."""
    pages = band_layout(n_pages)
    for regions in pages:
        regions.insert(0, {"label": "header", "bbox_2d": [0.0, 0.0, 1000.0, 80.0], "content": ""})
        regions.append(
            {"label": "table", "task_type": "table", "bbox_2d": [0.0, 300.0, 1000.0, 700.0]}
        )
        regions.append({"label": "display_formula", "bbox_2d": [100.0, 700.0, 900.0, 800.0]})
    return pages


def _inspect(pdf: bytes, layout, page_indices, stream=None, **kwargs):
    return inspect_pdf(
        pdf,
        layout,
        page_indices=page_indices,
        fill_native_text=True,
        include_outline=False,
        include_ref_geometry=kwargs.pop("include_ref_geometry", True),
        min_chars=20,
        min_printable_ratio=0.85,
        reject_invisible_text_layer=True,
        **({"text_layer": stream} if stream is not None else {}),
        **kwargs,
    )


@pytest.mark.parametrize("name", sorted(_FIXTURES))
@pytest.mark.parametrize(
    ("read_ocr_layer", "ref_geometry", "doc_layer", "reverse"),
    [
        (False, True, False, False),
        (True, False, False, True),
        (False, True, True, False),
    ],
    ids=["default", "ocr-layer-no-geometry-reversed", "document-layer"],
)
def test_the_streamed_read_is_the_read_of_the_inspected_layout(
    name, read_ocr_layer, ref_geometry, doc_layer, reverse
):
    pdf = _FIXTURES[name]
    n_pages = page_count(pdf)
    indices = list(range(n_pages))[::-1] if reverse else list(range(n_pages))
    layout = _layout(n_pages)
    extra = {"include_ref_geometry": ref_geometry}
    if doc_layer:
        extra["include_doc_layer"] = True
    stream = TextLayerStream(indices, read_ocr_layer=read_ocr_layer)

    streamed = _inspect(pdf, layout, indices, stream, **extra)
    plain = _inspect(pdf, layout, indices, **extra)
    read = stream.finish(streamed.layout_results)

    assert streamed.layout_results == plain.layout_results
    assert read is not None
    expected = read_text_layer(
        pdf, plain.layout_results, page_indices=indices, read_ocr_layer=read_ocr_layer
    )
    assert read == expected
    assert [page.metrics for page in read.pages] == [page.metrics for page in expected.pages]


def test_a_page_the_stream_cannot_read_leaves_the_read_to_read_text_layer(monkeypatch):
    pdf = _FIXTURES["synthetic_paper.pdf"]
    n_pages = page_count(pdf)
    real = _text_layer._page_text

    def page_text(page, textpage, page_index, *args, **kwargs):
        if page_index == 2:
            raise RuntimeError("unreadable page")
        return real(page, textpage, page_index, *args, **kwargs)

    monkeypatch.setattr(_text_layer, "_page_text", page_text)
    stream = TextLayerStream(list(range(n_pages)))

    inspection = _inspect(pdf, _layout(n_pages), list(range(n_pages)), stream)

    assert stream.finish(inspection.layout_results) is None


def test_a_stream_that_saw_no_page_reads_nothing():
    pdf = _FIXTURES["synthetic_paper.pdf"]
    n_pages = page_count(pdf)
    stream = TextLayerStream(list(range(n_pages)))

    # Neither the fill nor the geometry reads the text layer: no page reaches the stream.
    inspection = inspect_pdf(
        pdf,
        _layout(n_pages),
        page_indices=list(range(n_pages)),
        fill_native_text=False,
        include_outline=False,
        include_ref_geometry=False,
        min_chars=20,
        min_printable_ratio=0.85,
        text_layer=stream,
    )

    assert stream.finish(inspection.layout_results) is None


async def test_the_native_text_stage_reads_the_text_layer_from_its_one_open(monkeypatch):
    import pypdfium2

    from bibr.pipeline.context import PipelineContext, RunConfig
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.stages.native_text import NativeTextStage
    from bibr.pipeline.state import FileState

    pdf = _FIXTURES["sample_paper.pdf"]
    n_pages = page_count(pdf)
    layout = _layout(n_pages)
    expected = read_text_layer(
        pdf, _inspect(pdf, layout, list(range(n_pages))).layout_results
    ).layout_results
    real_document = pypdfium2.PdfDocument
    opens = []

    def document(*args, **kwargs):
        opens.append(1)
        return real_document(*args, **kwargs)

    monkeypatch.setattr(pypdfium2, "PdfDocument", document)
    fs = FileState(path=Path("sample_paper.pdf"))
    fs.pdf_bytes = pdf
    fs.layout_results = layout
    fs.page_indices = list(range(n_pages))
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=MagicMock(spec=[]),
        config=RunConfig(ocr_backend="text-layer"),
    )

    await NativeTextStage().run(ctx)

    assert fs.text_layer_read is True
    assert len(opens) == 1
    # The stage's settings may set a different printable ratio than the default.
    assert [[r.get("content") for r in page] for page in fs.layout_results] == [
        [r.get("content") for r in page] for page in expected
    ]
    assert any(r.get("content") for page in fs.layout_results for r in page)


# --- The usability gate and the U+FFFD the text layer itself carries -------------

# A paragraph whose font maps a tenth of its glyphs to U+FFFD: a broken font,
# which the U+FFFD density gate turns down.
_GARBLED = "The res�lts sh�w th�t the m�an sc�re r�se in �ach gr�up."


def _fill_prose(monkeypatch, text: str):
    monkeypatch.setattr(_text_layer, "_read_text", lambda page, bbox: text)
    region = {"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}
    return _text_layer._fill_region(object(), region, min_printable_ratio=0.85), region


def test_one_unmapped_glyph_does_not_excuse_the_replacement_characters_of_a_broken_font(
    monkeypatch,
):
    without_mark, _ = _fill_prose(monkeypatch, _GARBLED)
    with_mark, region = _fill_prose(monkeypatch, _GARBLED + " Plain\x01")

    assert without_mark == "rejected"
    assert with_mark == "rejected"
    assert region["_native_gate"] == "usability"


def _table_page(glyphs: list) -> _text_layer._PageText:
    return _text_layer._PageText(
        0, (0, 0, 595, 842), 0, len(glyphs), False, [], [(ch, box) for ch, *box in glyphs], []
    )


def _table_glyphs(rows: list[str]) -> list:
    glyphs: list = []
    for k, row in enumerate(rows):
        x = 100.0
        for cell in row.split("|"):
            glyphs += [
                (ch, x + 6 * i, 10 + 16 * k, x + 6 * i + 4, 20 + 16 * k)
                for i, ch in enumerate(cell)
            ]
            glyphs.append((" ", 0, 0, 0, 0))
            x += 100.0
    return glyphs


def test_a_table_mark_does_not_excuse_the_replacement_characters_of_a_broken_font():
    rows = ["Gr�up|N|M�an"] + [f"G�{k}|1�|0.�{k}" for k in range(6)]
    region = {"label": "table", "task_type": "table", "bbox_2d": [0, 0, 1000, 1000]}
    marked = {"label": "table", "task_type": "table", "bbox_2d": [0, 0, 1000, 1000]}

    plain = _text_layer._fill_region(
        _table_page(_table_glyphs(rows)), region, min_printable_ratio=0.85
    )
    with_mark = _text_layer._fill_region(
        _table_page(_table_glyphs([*rows, "Total|\x0512|0.9"])), marked, min_printable_ratio=0.85
    )

    assert plain == "rejected"
    assert with_mark == "rejected"
    assert marked["_native_gate"] == "usability"
