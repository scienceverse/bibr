from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest


def _layout():
    return [
        [
            {
                "label": "text",
                "bbox_2d": [0, 0, 1000, 1000],
                "content": "",
            }
        ]
    ]


def test_windowed_inspection_preserves_reference_continuations_and_furniture():
    import io

    import pypdfium2

    from bibr.ocr.pdf_inspection import PdfInspectionAccumulator, inspect_pdf
    from tests.test_pipeline_smoke import _build_pdf

    doc = pypdfium2.PdfDocument.new()
    try:
        for index in range(3):
            name = ("Ada", "Boris", "Cora")[index]
            lines = [(60, 750, 10, "Repeated running header")]
            if index == 0:
                lines += [(60, 700, 14, "References")]
            lines += [
                (60, 650, 10, f"{name}. Unique publication number {index}."),
                (60, 620, 10, f"Journal of {name}, volume 2, pp. 10-20."),
                (60, 40, 10, f"Page {index + 1}"),
            ]
            source = pypdfium2.PdfDocument(_build_pdf(lines))
            try:
                doc.import_pages(source)
            finally:
                source.close()
        buffer = io.BytesIO()
        doc.save(buffer)
        pdf = buffer.getvalue()
    finally:
        doc.close()

    options = {
        "fill_native_text": True,
        "include_outline": True,
        "include_ref_geometry": True,
        "min_chars": 1,
        "min_printable_ratio": 0.5,
    }
    full = inspect_pdf(pdf, [_layout()[0] for _ in range(3)], **options)
    state = PdfInspectionAccumulator()
    window_layouts = []
    for i in range(3):
        window = inspect_pdf(pdf, _layout(), page_indices=[i], accumulator=state, **options)
        window_layouts.extend(window.layout_results)

    assert window_layouts == full.layout_results
    assert window.pages == full.pages
    assert window.metadata == full.metadata
    assert window.outline == full.outline
    assert state.reference_lines() == full.reference_lines
    assert {line["page"] for line in state.reference_lines()} == {0, 1, 2}
    assert all("running header" not in line["text"] for line in state.reference_lines())
    assert not _contains_native_handle(state)


def _contains_native_handle(value) -> bool:
    if type(value).__module__.startswith("pypdfium2"):
        return True
    if is_dataclass(value):
        return any(_contains_native_handle(getattr(value, field.name)) for field in fields(value))
    if isinstance(value, dict):
        return any(_contains_native_handle(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_native_handle(item) for item in value)
    return False


def test_inspection_opens_document_once_and_returns_detached_data(monkeypatch):
    import pypdfium2

    from bibr.ocr.pdf_inspection import inspect_pdf

    pdf_bytes = Path("tests/fixtures/native_text_sample.pdf").read_bytes()
    real_factory = pypdfium2.PdfDocument
    opened = []

    def factory(*args, **kwargs):
        opened.append(1)
        return real_factory(*args, **kwargs)

    monkeypatch.setattr(pypdfium2, "PdfDocument", factory)
    result = inspect_pdf(
        pdf_bytes,
        _layout(),
        first_page_text="",
        fill_native_text=True,
        include_outline=True,
        include_ref_geometry=True,
        min_chars=1,
        min_printable_ratio=0.5,
    )

    assert len(opened) == 1
    assert result.pages[0].width > 0
    assert result.pages[0].height > 0
    assert result.layout_results[0][0]["_native_text_used"] is True
    assert not _contains_native_handle(result)


def test_inspection_matches_existing_native_fill_and_components():
    from copy import deepcopy

    from bibr.input.pdf_metadata import harvest_pdf_metadata
    from bibr.input.pdf_outline import extract_pdf_outline
    from bibr.ocr.native_text import fill_native_text_and_fonts
    from bibr.ocr.pdf_inspection import inspect_pdf
    from bibr.ocr.ref_geometry import record_to_dict, recover_reference_lines

    pdf_bytes = Path("tests/fixtures/native_text_sample.pdf").read_bytes()
    expected_layout = fill_native_text_and_fonts(
        pdf_bytes,
        deepcopy(_layout()),
        min_chars=1,
        min_printable_ratio=0.5,
    )
    first_page_text = expected_layout[0][0]["content"]
    result = inspect_pdf(
        pdf_bytes,
        _layout(),
        first_page_text=first_page_text,
        fill_native_text=True,
        include_outline=True,
        include_ref_geometry=True,
        min_chars=1,
        min_printable_ratio=0.5,
    )

    assert result.layout_results[0][0]["_source_region_id"] == "p0:r0"
    expected_layout[0][0]["_source_region_id"] = "p0:r0"
    assert result.layout_results == expected_layout
    assert result.metadata == harvest_pdf_metadata(pdf_bytes, first_page_text)
    assert result.outline == extract_pdf_outline(pdf_bytes)
    assert result.reference_lines == [record_to_dict(r) for r in recover_reference_lines(pdf_bytes)]


def test_component_failure_is_recorded_without_discarding_other_results(monkeypatch):
    from pathlib import Path

    from bibr.ocr import pdf_inspection

    monkeypatch.setattr(
        pdf_inspection,
        "_walk_pdfium_outline",
        lambda doc: (_ for _ in ()).throw(RuntimeError("bad outline")),
    )
    result = pdf_inspection.inspect_pdf(
        Path("tests/fixtures/native_text_sample.pdf").read_bytes(),
        _layout(),
        first_page_text="",
        fill_native_text=True,
        include_outline=True,
        include_ref_geometry=False,
        min_chars=1,
        min_printable_ratio=0.5,
    )
    assert result.layout_results[0][0]["content"]
    assert result.outline == []
    assert "outline" in result.component_errors


def test_selected_page_uses_physical_pdf_index_for_text_and_geometry():
    from bibr.ocr.pdf_inspection import inspect_pdf

    pdf_bytes = Path("bibr/data/sample_paper.pdf").read_bytes()
    selected_layout = _layout()

    result = inspect_pdf(
        pdf_bytes,
        selected_layout,
        page_indices=[1],
        first_page_text="",
        fill_native_text=True,
        include_outline=False,
        include_ref_geometry=True,
        min_chars=1,
        min_printable_ratio=0.5,
    )

    assert [page.index for page in result.pages] == [1]
    assert "structural elements" in result.layout_results[0][0]["content"]
    assert "Coefficient of Rodential" not in result.layout_results[0][0]["content"]
    assert all(line["page"] == 1 for line in result.reference_lines)


@pytest.mark.parametrize(
    ("page_indices", "layout_page_count"),
    [
        pytest.param([0, 1], 1, id="length-mismatch"),
        pytest.param([0, 0], 2, id="duplicates"),
        pytest.param([-1], 1, id="negative"),
        pytest.param([2], 1, id="out-of-range"),
    ],
)
def test_selected_page_map_must_be_valid(page_indices, layout_page_count):
    from bibr.ocr.pdf_inspection import inspect_pdf

    pdf_bytes = Path("bibr/data/sample_paper.pdf").read_bytes()
    layout_results = [_layout()[0] for _ in range(layout_page_count)]

    with pytest.raises(ValueError):
        inspect_pdf(
            pdf_bytes,
            layout_results,
            page_indices=page_indices,
            first_page_text="",
            fill_native_text=True,
            include_outline=False,
            include_ref_geometry=True,
            min_chars=1,
            min_printable_ratio=0.5,
        )
