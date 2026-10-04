"""The outline the layer reads: every bookmark as the PDF declares it, nothing filtered."""

from __future__ import annotations

import pypdfium2

from bibr.document import harvest, outline
from bibr.document.harvest import build_document_layer
from bibr.ocr.utils import pdfium_lock
from tests.document import _linked, _pdfs
from tests.document.test_layer import _BUDGET, _inspect


def _layer(**kwargs):
    return build_document_layer(
        _linked.linked_paper(**kwargs), range(_linked.N_PAGES), budget=_BUDGET
    )


def test_every_bookmark_is_read_with_its_depth_parent_and_target():
    layer = _layer()

    assert [entry.idx for entry in layer.outline] == list(range(len(_linked.OUTLINE)))
    for entry, bookmark, read in zip(
        layer.outline, _linked.OUTLINE, _linked.OUTLINE_READ, strict=True
    ):
        parent, page, x, y, name = read
        assert (entry.title, entry.level) == (bookmark.title, bookmark.level)
        assert (entry.parent, entry.page, entry.x, entry.y, entry.dest_name) == (
            parent,
            page,
            x,
            y,
            name,
        )
    assert layer.component_errors == {}


def test_an_entry_is_named_by_its_position_in_the_outline():
    layer = _layer()

    assert [entry.entry_id for entry in layer.outline] == [
        f"ol{n}" for n in range(len(_linked.OUTLINE))
    ]
    # The ids the guard cites are these.
    assert set(layer.outline_guard.decided.evidence) <= {entry.entry_id for entry in layer.outline}


def test_blank_titles_and_entries_the_guard_drops_stay_in_the_outline():
    titles = [entry.title for entry in _layer().outline]

    assert "" in titles and "Contents" in titles and "Figure 1" in titles and "12" in titles


def test_a_jump_into_another_file_and_a_page_past_the_end_name_no_page():
    layer = _layer()
    by_title = {entry.title: entry for entry in layer.outline}

    for title in ("Supplement", "Appendix"):
        entry = by_title[title]
        assert (entry.page, entry.x, entry.y, entry.dest_name) == (None, None, None, None)


def test_a_destination_reached_by_name_keeps_the_name_whether_it_is_a_dest_or_a_goto_action():
    by_title = {entry.title: entry for entry in _layer().outline}

    assert by_title["1 Introduction"].dest_name == "section.1"
    assert by_title["2 Methods"].dest_name == "section.2"
    assert by_title["Abstract"].dest_name is None


def test_a_title_keeps_its_characters_beyond_latin_1_and_the_basic_plane():
    title = "Résumé – 結果 \U0001d6fc"
    layer = _layer(outline=[_linked.Bookmark(title, 0, ("array", 0, "/Fit"))])

    assert [entry.title for entry in layer.outline] == [title]


def test_every_depth_is_read():
    # pypdfium2's get_toc stops at depth 15; the layer's walk does not.
    deep = [_linked.Bookmark(f"Level {n}", n, ("array", 1, "/Fit")) for n in range(20)]
    layer = _layer(outline=deep)

    assert [entry.level for entry in layer.outline] == list(range(20))
    assert [entry.parent for entry in layer.outline] == [None, *range(19)]


def test_a_bookmark_chain_that_loops_ends_at_the_loop_and_says_so():
    layer = _layer(loop=True)

    assert [entry.title for entry in layer.outline] == [b.title for b in _linked.OUTLINE]
    assert layer.component_errors == {"outline": "circular bookmark reference"}


def test_an_outline_over_the_limit_is_read_to_the_limit(monkeypatch):
    monkeypatch.setattr(outline, "MAX_ENTRIES", 4)
    layer = _layer()

    assert [entry.idx for entry in layer.outline] == [0, 1, 2, 3]
    assert "more than 4 bookmarks" in layer.component_errors["outline"]


def test_the_layer_says_whether_the_document_has_an_outline():
    assert _layer().presence.has_outline is True
    assert _layer(outline=None).presence.has_outline is False
    assert (
        build_document_layer(_pdfs.synthetic_paper(), range(7), budget=_BUDGET).presence.has_outline
        is False
    )


def test_a_missing_pdfium_function_leaves_the_outline_unread(monkeypatch):
    monkeypatch.setattr(harvest, "missing_apis", lambda: ("FPDFBookmark_GetTitle",))
    layer = _layer()

    assert layer.outline == [] and layer.outline_guard is None
    assert layer.presence.has_outline is None
    assert layer.presence.outline_guard_pass is None
    assert layer.component_errors == {}


def test_a_failing_walk_keeps_what_it_read_and_says_why(monkeypatch):
    real = outline._entry
    calls = []

    def fail_on_the_third(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("bad bookmark")
        return real(*args, **kwargs)

    monkeypatch.setattr(outline, "_entry", fail_on_the_third)
    layer = _layer()

    assert [entry.idx for entry in layer.outline] == [0, 1]
    assert layer.component_errors == {"outline": "RuntimeError: bad bookmark"}
    assert layer.presence.has_outline is True


def test_the_pdf_title_is_read_from_the_information_dictionary():
    with pdfium_lock:
        doc = pypdfium2.PdfDocument(_linked.linked_paper())
        try:
            assert outline.meta_title(harvest._Api(), doc) == _linked.TITLE
        finally:
            doc.close()
        doc = pypdfium2.PdfDocument(_pdfs.synthetic_paper())
        try:
            assert outline.meta_title(harvest._Api(), doc) is None
        finally:
            doc.close()


def test_the_inline_build_reads_the_same_outline_and_verdict():
    pdf_bytes = _linked.linked_paper()
    inline = _inspect(pdf_bytes, layer=True).document
    rebuilt = build_document_layer(pdf_bytes, range(_linked.N_PAGES), budget=_BUDGET)

    assert inline.outline == rebuilt.outline == _layer().outline
    assert inline.outline_guard == rebuilt.outline_guard
    assert inline.presence.has_outline and inline.presence.outline_guard_pass


def test_the_outline_does_not_change_what_the_inspection_reads():
    # The inspection's own outline stays the pypdfium2 walk the heading matcher uses.
    pdf_bytes = _linked.linked_paper()
    inspection = _inspect(pdf_bytes, layer=True)

    assert [item.title for item in inspection.outline][:3] == [
        _linked.TITLE,
        "Abstract",
        "1 Introduction",
    ]
    assert len(inspection.outline) == len(_linked.OUTLINE) - 1
