"""Page labels and destinations: what the layer reads of how a PDF numbers and names its pages."""

from __future__ import annotations

import ctypes

import pypdfium2
import pytest

from bibr.document import destinations, harvest
from bibr.document.harvest import build_document_layer
from bibr.ocr.pdf_inspection import inspect_pdf
from bibr.ocr.utils import pdfium_lock
from tests.document import _linked, _pdfs
from tests.document.test_layer import _BUDGET, _inspect


def _layer(pdf_bytes: bytes):
    return build_document_layer(pdf_bytes, range(_linked.N_PAGES), budget=_BUDGET)


class _Doc:
    """An open document with the destination table, under the pdfium lock."""

    def __init__(self, pdf_bytes: bytes) -> None:
        self.pdf_bytes = pdf_bytes

    def __enter__(self):
        pdfium_lock.acquire()
        self.doc = pypdfium2.PdfDocument(self.pdf_bytes)
        self.api = harvest._Api()
        return self

    def __exit__(self, *exc) -> None:
        self.doc.close()
        pdfium_lock.release()

    def named(self, name: str):
        """The destination pdfium resolves *name* to, as a link or bookmark gets it."""
        return self.api.c.FPDF_GetNamedDestByName(self.doc.raw, name.encode())


# --- Page labels ---------------------------------------------------------------


def test_page_labels_come_from_the_page_labels_tree():
    layer = _layer(_linked.linked_paper())

    assert [page.label for page in layer.pages] == _linked.PAGE_LABELS
    # An empty label is a label: the page's range sets a prefix and no number.
    assert layer.page(5).label == ""


def test_a_document_without_page_labels_leaves_them_unset():
    layer = _layer(_pdfs.synthetic_paper())

    assert [page.label for page in layer.pages] == [None] * len(layer.pages)


def test_labels_are_the_same_inline_and_rebuilt():
    pdf_bytes = _linked.linked_paper()
    inline = _inspect(pdf_bytes, layer=True).document
    rebuilt = _layer(pdf_bytes)

    assert [page.label for page in inline.pages] == _linked.PAGE_LABELS
    assert [page.label for page in rebuilt.pages] == _linked.PAGE_LABELS


def test_a_label_is_read_for_the_pages_of_a_range_only():
    layer = build_document_layer(_linked.linked_paper(), [4, 5], budget=_BUDGET)

    assert [(page.index, page.label) for page in layer.pages] == [(4, "A-1"), (5, "")]


def test_utf16_text_tells_an_absent_string_from_an_empty_one():
    def getter(text: str | None):
        def read(*args):
            *_head, buffer, size = args
            if text is None:
                return 0
            raw = text.encode("utf-16-le") + b"\x00\x00"
            if buffer is not None:
                ctypes.memmove(buffer, raw, min(size, len(raw)))
            return len(raw)

        return read

    assert destinations.utf16_text(getter(None), "doc") is None
    assert destinations.utf16_text(getter(""), "doc") == ""
    assert destinations.utf16_text(getter("Anhang é"), "doc") == "Anhang é"


# --- Named destinations -----------------------------------------------------------


def test_the_named_destination_table_finds_a_name_by_the_destination_it_holds():
    with _Doc(_linked.linked_paper()) as opened:
        names = destinations.NamedDests(opened.api, opened.doc)

        assert names.count == len(_linked.DESTS) + len(_linked.UNSORTED)
        for name in _linked.DESTS:
            assert names.name_of(opened.named(name)) == name
        assert names.error is None


def test_the_table_finds_a_destination_pdfiums_lookup_by_name_misses():
    with _Doc(_linked.linked_paper()) as opened:
        names = destinations.NamedDests(opened.api, opened.doc)
        n_pages = len(opened.doc)

        for name, (page, _view) in _linked.UNSORTED.items():
            # pdfium's search stops at the first name that sorts after the one it wants.
            assert not opened.named(name)
            dest = names.dest_of(name)
            assert destinations.dest_page(opened.api, opened.doc, dest, n_pages) == page
            assert destinations.dest_position(opened.api, dest) == _linked.DEST_XY[name]
            assert names.name_of(dest) == name
        assert names.dest_of("figure.1") is not None
        assert names.dest_of("no.such.name") is None


def test_a_destination_the_table_does_not_hold_has_no_name():
    with _Doc(_linked.linked_paper()) as opened:
        names = destinations.NamedDests(opened.api, opened.doc)

        assert names.name_of(None) is None
        assert names.name_of(opened.named("no.such.name")) is None


def test_destinations_give_their_page_and_position():
    with _Doc(_linked.linked_paper()) as opened:
        n_pages = len(opened.doc)
        for name, (page, _view) in _linked.DESTS.items():
            dest = opened.named(name)
            assert destinations.dest_page(opened.api, opened.doc, dest, n_pages) == page, name
            assert destinations.dest_position(opened.api, dest) == _linked.DEST_XY[name], name


def test_a_destination_past_the_last_page_names_no_page():
    with _Doc(_linked.linked_paper()) as opened:
        dest = opened.named("figure.1")

        assert destinations.dest_page(opened.api, opened.doc, dest, 3) is None
        assert destinations.dest_page(opened.api, opened.doc, dest, 4) == 3


def test_the_layer_says_whether_the_document_has_named_destinations():
    assert _layer(_linked.linked_paper()).presence.has_named_dests is True
    assert _layer(_pdfs.synthetic_paper()).presence.has_named_dests is False
    scanned = _inspect(_pdfs.fixture_pdfs()["scanned_sample.pdf"], layer=True).document
    assert scanned.presence.has_named_dests is False


def test_named_destinations_over_the_limit_stay_unread(monkeypatch):
    monkeypatch.setattr(destinations, "MAX_NAMED_DESTS", 3)
    with _Doc(_linked.linked_paper()) as opened:
        names = destinations.NamedDests(opened.api, opened.doc)

        assert names.name_of(opened.named("figure.1")) is None
        assert names.error is not None and "over the limit" in names.error


def test_a_missing_pdfium_function_leaves_its_fact_unread(monkeypatch):
    monkeypatch.setattr(
        harvest, "missing_apis", lambda: ("FPDF_GetPageLabel", "FPDF_CountNamedDests")
    )
    layer = _layer(_linked.linked_paper())

    assert [page.label for page in layer.pages] == [None] * _linked.N_PAGES
    assert layer.presence.has_named_dests is None
    assert layer.presence.missing_apis == ("FPDF_GetPageLabel", "FPDF_CountNamedDests")
    assert layer.component_errors == {}


def test_the_new_pdfium_functions_are_in_the_harvest_list():
    for name in ("FPDF_GetPageLabel", *destinations.APIS):
        assert name in harvest._HARVEST_APIS
    assert harvest.missing_apis() == ()


def test_the_inline_build_reads_the_destinations_too():
    inspection = inspect_pdf(
        _linked.linked_paper(),
        _pdfs.band_layout(_linked.N_PAGES),
        fill_native_text=True,
        include_outline=True,
        include_ref_geometry=True,
        min_chars=3,
        min_printable_ratio=0.85,
        include_doc_layer=True,
        render_budget=_BUDGET,
    )

    assert inspection.document.presence.has_named_dests is True
    assert [page.label for page in inspection.document.pages] == _linked.PAGE_LABELS


@pytest.mark.parametrize("name", sorted(_pdfs.fixture_pdfs()))
def test_every_fixtures_labels_are_the_ones_pypdfium2_reads(name):
    pdf_bytes = _pdfs.fixture_pdfs()[name]
    layer = _inspect(pdf_bytes, layer=True).document
    with pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            # pypdfium2 returns "" for a page without a label.
            oracle = [doc.get_page_label(index) for index in range(len(doc))]
        finally:
            doc.close()

    for page in layer.pages:
        assert (page.label or "") == oracle[page.index]
