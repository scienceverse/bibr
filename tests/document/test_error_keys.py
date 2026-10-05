"""The keys of DocumentLayer.component_errors: one table says them, and the table is the truth."""

from __future__ import annotations

import re
from collections.abc import Callable

import pypdfium2.raw as pdfium_raw
import pytest

from bibr.document import destinations, harvest, links, outline, outline_guard, structure
from bibr.document.harvest import build_document_layer
from bibr.document.model import COMPONENT_ERROR_KEYS, DocumentLayer
from bibr.document.rebuild import attach_blocks
from bibr.ocr import native_text as nt
from bibr.ocr.types import OcrRegionResult
from tests.document import _linked, _pdfs
from tests.document.test_destinations import _paper_with_destinations
from tests.document.test_layer import _BUDGET, _inspect


def _boom(*_args, **_kwargs):
    raise RuntimeError("boom")


def _linked_layer() -> DocumentLayer:
    return build_document_layer(_linked.linked_paper(), range(_linked.N_PAGES), budget=_BUDGET)


# One scenario for each key of the table: a layer that has the key, made with the seam that
# gives it (a limit lowered, a reader that raises).


def _start(patch: pytest.MonkeyPatch) -> DocumentLayer:
    patch.setattr(harvest.LayerBuilder, "start", _boom)
    return _inspect(_linked.linked_paper(), layer=True).document


def _unreadable_page(patch: pytest.MonkeyPatch) -> DocumentLayer:
    return build_document_layer(_pdfs.synthetic_paper(), [0, 99], budget=_BUDGET)


def _harvest(patch: pytest.MonkeyPatch) -> DocumentLayer:
    patch.setattr(harvest, "page_header", _boom)
    return _linked_layer()


def _records(patch: pytest.MonkeyPatch) -> DocumentLayer:
    patch.setattr(nt, "_build_page_char_records", _boom)
    return _linked_layer()


def _furniture(patch: pytest.MonkeyPatch) -> DocumentLayer:
    patch.setattr(nt, "strip_furniture_objects", _boom)
    return _linked_layer()


def _blocks(patch: pytest.MonkeyPatch) -> DocumentLayer:
    layer = _linked_layer()
    regions = [
        OcrRegionResult.from_layout_region({"label": "text"}, slot_idx=0, content=text)
        for text in ("first", "second")
    ]
    # Both regions have the index 0: the second is not at its own position.
    attach_blocks(layer, [regions])
    return layer


def _label(patch: pytest.MonkeyPatch) -> DocumentLayer:
    patch.setattr(pdfium_raw, "FPDF_GetPageLabel", _boom)
    return _linked_layer()


def _tagged(patch: pytest.MonkeyPatch) -> DocumentLayer:
    patch.setattr(pdfium_raw, "FPDFCatalog_IsTagged", _boom)
    return _linked_layer()


def _named_dests(patch: pytest.MonkeyPatch) -> DocumentLayer:
    patch.setattr(destinations, "MAX_NAMED_DESTS", 3)
    return _linked_layer()


def _outline(patch: pytest.MonkeyPatch) -> DocumentLayer:
    patch.setattr(outline, "MAX_ENTRIES", 2)
    return _linked_layer()


def _dead_ends(patch: pytest.MonkeyPatch) -> DocumentLayer:
    # An allowance of one walk: the second dead end of the outline and of the links is skipped.
    patch.setattr(destinations, "MAX_PAGE_CHECKS", 0)
    patch.setattr(destinations, "MIN_UNRESOLVED", 1)
    return build_document_layer(_paper_with_destinations("ddd", "ddd"), [0], budget=_BUDGET)


def _outline_guard(patch: pytest.MonkeyPatch) -> DocumentLayer:
    patch.setattr(outline_guard, "judge", _boom)
    return _linked_layer()


def _links(patch: pytest.MonkeyPatch) -> DocumentLayer:
    patch.setattr(links, "MAX_LINKS", 5)
    return _linked_layer()


def _links_of_a_page(patch: pytest.MonkeyPatch) -> DocumentLayer:
    patch.setattr(links, "MAX_QUADS", 1)
    return _linked_layer()


def _links_build(patch: pytest.MonkeyPatch) -> DocumentLayer:
    patch.setattr(links, "build_links", _boom)
    return _linked_layer()


def _struct(patch: pytest.MonkeyPatch) -> DocumentLayer:
    patch.setattr(structure, "MAX_ELEMENTS", 4)
    return _linked_layer()


def _struct_of_a_page(patch: pytest.MonkeyPatch) -> DocumentLayer:
    patch.setattr(structure, "read_page_tree", _boom)
    return _linked_layer()


def _struct_merge(patch: pytest.MonkeyPatch) -> DocumentLayer:
    patch.setattr(structure, "merge", _boom)
    return _linked_layer()


_SCENARIOS: dict[str, Callable[[pytest.MonkeyPatch], DocumentLayer]] = {
    "start": _start,
    "page:{page}": _unreadable_page,
    "harvest:{page}": _harvest,
    "records:{page}": _records,
    "furniture:{page}": _furniture,
    "blocks:{page}": _blocks,
    "label:{page}": _label,
    "tagged": _tagged,
    "named_dests": _named_dests,
    "outline": _outline,
    "outline_unresolved": _dead_ends,
    "outline_guard": _outline_guard,
    "links": _links,
    "links:{page}": _links_of_a_page,
    "links_build": _links_build,
    "links_unresolved": _dead_ends,
    "struct": _struct,
    "struct:{page}": _struct_of_a_page,
    "struct_merge": _struct_merge,
}


def _pattern(key: str) -> re.Pattern[str]:
    """The regular expression a key of the table stands for: {page} is a page number."""
    return re.compile("[0-9]+".join(re.escape(part) for part in key.split("{page}")))


_PATTERNS = {key: _pattern(key) for key in COMPONENT_ERROR_KEYS}


def test_every_key_of_the_table_has_a_scenario_and_every_scenario_a_key():
    assert set(_SCENARIOS) == set(COMPONENT_ERROR_KEYS)


@pytest.mark.parametrize("key", sorted(COMPONENT_ERROR_KEYS))
def test_a_scenario_gives_its_key_and_no_key_the_table_does_not_say(key, monkeypatch):
    errors = _SCENARIOS[key](monkeypatch).component_errors

    assert any(_PATTERNS[key].fullmatch(found) for found in errors), errors
    for found in errors:
        assert any(pattern.fullmatch(found) for pattern in _PATTERNS.values()), found


def test_a_page_key_names_a_number_and_nothing_else():
    assert _PATTERNS["links:{page}"].fullmatch("links:12")
    assert not _PATTERNS["links:{page}"].fullmatch("links")
    assert not _PATTERNS["links:{page}"].fullmatch("links:x")
    assert not _PATTERNS["links"].fullmatch("links:1")
    assert all(text for text in COMPONENT_ERROR_KEYS.values())
