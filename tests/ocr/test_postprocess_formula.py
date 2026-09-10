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
