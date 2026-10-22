"""The tagger reads born-digital front pages only."""

from __future__ import annotations

from types import SimpleNamespace

from bibr.pipeline.stages.parse_segment import _front_page_was_read


def test_ocr_read_front_pages_keep_the_llm():
    assert _front_page_was_read(SimpleNamespace(page_kinds={0: "scan"}), 0)
    assert _front_page_was_read(SimpleNamespace(page_kinds={2: "broken_text_layer"}), 2)
    assert not _front_page_was_read(SimpleNamespace(page_kinds={0: "born_digital"}), 0)
    # Only the first page decides; an unclassified page is no scan.
    assert not _front_page_was_read(SimpleNamespace(page_kinds={1: "scan"}), 0)
    assert not _front_page_was_read(SimpleNamespace(), 0)
