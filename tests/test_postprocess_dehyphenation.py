"""Tests for hyphenated text-block merging in OCR postprocessing."""

from __future__ import annotations

from bibr.ocr.postprocess import merge_text_blocks


def test_dehyphenation_does_not_cross_figure_boundary():
    page = [
        {"label": "text", "content": "the metho-"},
        {"label": "figure", "content": ""},
        {"label": "text", "content": "dology used"},
    ]
    out = merge_text_blocks(page)
    contents = [b.get("content") for b in out]
    assert "methodology" not in " ".join(contents), (
        f"hyphenated word merged across figure boundary; got {contents}"
    )


def test_dehyphenation_merges_adjacent_text_blocks():
    """Sanity: the normal adjacent-text merge still works."""
    page = [
        {"label": "text", "content": "the metho-"},
        {"label": "text", "content": "dology used"},
    ]
    out = merge_text_blocks(page)
    contents = " ".join(b.get("content", "") for b in out)
    assert "methodology" in contents, f"adjacent dehyphenation broke; got {contents}"
