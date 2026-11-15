"""Tests for formula_number merging in OCR postprocessing."""

from __future__ import annotations

from bibr.ocr.postprocess import merge_formula_numbers


def test_formula_number_then_formula_merges_tag():
    page = [
        {"label": "text", "native_label": "formula_number", "content": "(1)"},
        {"label": "formula", "native_label": "display_formula", "content": "$$E = mc^2\n$$"},
    ]
    out = merge_formula_numbers(page)
    assert len(out) == 1
    assert "\\tag{1}" in out[0]["content"]


def test_formula_then_formula_number_merges_tag():
    page = [
        {"label": "formula", "native_label": "display_formula", "content": "$$E = mc^2\n$$"},
        {"label": "text", "native_label": "formula_number", "content": "(2)"},
    ]
    out = merge_formula_numbers(page)
    assert len(out) == 1
    assert "\\tag{2}" in out[0]["content"]


def test_orphan_formula_number_last_on_page_is_preserved():
    """A formula_number with no adjacent formula (page break) must pass through,
    not vanish from the output."""
    page = [
        {"label": "text", "native_label": "text", "content": "Body text."},
        {"label": "text", "native_label": "formula_number", "content": "(3)"},
    ]
    out = merge_formula_numbers(page)
    assert [b["content"] for b in out] == ["Body text.", "(3)"]


def test_orphan_formula_number_before_non_formula_is_preserved():
    page = [
        {"label": "text", "native_label": "formula_number", "content": "(7)"},
        {"label": "text", "native_label": "text", "content": "Next paragraph."},
    ]
    out = merge_formula_numbers(page)
    assert [b["content"] for b in out] == ["(7)", "Next paragraph."]


def test_output_reindexed_sequentially():
    page = [
        {"label": "text", "native_label": "text", "content": "A.", "index": 5},
        {"label": "text", "native_label": "formula_number", "content": "(4)", "index": 9},
    ]
    out = merge_formula_numbers(page)
    assert [b["index"] for b in out] == list(range(len(out)))


# --- Formulas read from the PDF text layer (plain text, no LaTeX) ---------------


def _text_layer_formula(content: str = "y = a + b x") -> dict:
    return {
        "label": "formula",
        "native_label": "display_formula",
        "content": content,
        "_formula_source": "text_layer",
    }


def _number(content: str = "(1)") -> dict:
    return {"label": "text", "native_label": "formula_number", "content": content}


def test_a_text_layer_formula_takes_its_number_as_printed():
    after = merge_formula_numbers([_text_layer_formula(), _number("(1)")])
    before = merge_formula_numbers([_number("(2.1)"), _text_layer_formula("z = 2 ")])

    assert [block["content"] for block in after] == ["y = a + b x (1)"]
    assert [block["content"] for block in before] == ["z = 2 (2.1)"]
    assert "\\tag" not in after[0]["content"]


def test_a_formula_of_unknown_source_without_delimiters_is_left_alone():
    page = [_text_layer_formula(), _number()]
    del page[0]["_formula_source"]

    out = merge_formula_numbers(page)

    assert [block["content"] for block in out] == ["y = a + b x"]


def test_postprocessing_does_not_wrap_a_text_layer_formula_in_latex():
    from bibr.pipeline.stages.ocr import _postprocess_ocr_regions

    def page(formula: dict) -> list[dict]:
        regions = [formula, _number()]
        for index, region in enumerate(regions):
            region.update(index=index, bbox_2d=[100 + 400 * index, 100, 300 + 400 * index, 120])
        regions[0]["_native_text_used"] = True
        return regions

    ocr = _text_layer_formula()
    del ocr["_formula_source"]
    [[layer_region]] = _postprocess_ocr_regions([page(_text_layer_formula())])
    [[ocr_region]] = _postprocess_ocr_regions([page(ocr)])

    assert layer_region["content"] == "y = a + b x (1)"
    assert ocr_region["content"] == "$$\ny = a + b x \\tag{1}\n$$"


def test_the_parser_keeps_a_text_layer_formula_as_printed():
    from bibr.structure.pdf_parser import PDFParser

    def formula_text(source: str | None) -> str:
        region = {
            "index": 0,
            "label": "formula",
            "native_label": "display_formula",
            "content": "y = a +\n   b x (1)",
            "bbox_2d": [100, 100, 300, 140],
        }
        if source is not None:
            region["_formula_source"] = source
        parser = PDFParser([[region]])
        parser.parse()
        [entry] = [entry for entry in parser._deferred_texts if entry[4]]  # is_formula
        return entry[0]

    assert formula_text("text_layer") == "y = a + b x (1)"
    # OCR output goes through the LaTeX cleanup, which keeps its line breaks.
    assert formula_text(None) == "y = a +\n   b x (1)"
