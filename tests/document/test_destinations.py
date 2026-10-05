"""Page labels and destinations: what the layer reads of how a PDF numbers and names its pages."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pypdfium2
import pypdfium2.raw as pdfium_raw
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


def _named_paper(count: int) -> bytes:
    """A one-page PDF with *count* named destinations, in one flat /Names array."""
    pairs = b" ".join(b"(n%05d) [3 0 R /Fit]" % index for index in range(count))
    return _pdfs.serialize_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R /Names << /Dests << /Names [" + pairs + b"] >> >> >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
        ]
    )


def test_a_table_of_exactly_the_limit_is_read_and_one_name_more_is_left_unread():
    limit = destinations.MAX_NAMED_DESTS
    with _Doc(_named_paper(limit)) as opened:
        names = destinations.NamedDests(opened.api, opened.doc)

        assert names.count == limit
        assert names.dest_of("n00000") is not None
        assert names.dest_of(f"n{limit - 1:05d}") is not None
        assert names.error is None
    with _Doc(_named_paper(limit + 1)) as opened:
        names = destinations.NamedDests(opened.api, opened.doc)

        assert names.count == limit + 1
        assert names.dest_of("n00000") is None
        assert names.error == f"{limit + 1} named destinations, over the limit of {limit}"


def test_each_name_is_read_with_one_call_into_a_buffer():
    with _Doc(_linked.linked_paper()) as opened:
        real = opened.api.FPDF_GetNamedDest
        calls = []

        def counted(doc, index, buffer, size):
            calls.append((index, buffer is not None))
            return real(doc, index, buffer, size)

        opened.api.FPDF_GetNamedDest = counted
        names = destinations.NamedDests(opened.api, opened.doc)

        assert names.name_of(opened.named("figure.1")) == "figure.1"
        # pdfium walks the name tree for every call, so a second call for the size would
        # double the time the table takes.
        assert calls == [(index, True) for index in range(names.count)]


def test_a_name_too_long_for_the_buffer_is_skipped_and_the_others_are_read(monkeypatch):
    # A name of 11 characters and the NUL fill 24 bytes.
    limit = 24
    monkeypatch.setattr(destinations, "MAX_TEXT", limit)
    with _Doc(_linked.linked_paper()) as opened:
        # What the read relies on: pdfium gives the destination of a name that does not
        # fit the buffer all the same, with a size of -1.
        buffer = ctypes.create_string_buffer(limit)
        sizes = []
        for index in range(opened.api.FPDF_CountNamedDests(opened.doc.raw)):
            size = ctypes.c_long(limit)
            assert opened.api.FPDF_GetNamedDest(opened.doc.raw, index, buffer, ctypes.byref(size))
            sizes.append(size.value)
        assert -1 in sizes

        names = destinations.NamedDests(opened.api, opened.doc)
        every = [*_linked.DESTS, *_linked.UNSORTED]
        read = {name for name in every if names.dest_of(name) is not None}

        assert read == {name for name in every if 2 * len(name) + 2 <= limit}
        assert 0 < len(read) < len(every)
        assert all(names.name_of(names.dest_of(name)) == name for name in read)
        assert names.error is None


# --- Destinations that point at no page --------------------------------------------


def _paper_with_destinations(links: str = "", marks: str = "", *, pages: int = 1) -> bytes:
    """A PDF of *pages* blank pages, with a link on the first for each letter of *links* and a
    bookmark for each letter of *marks*.

    "d" is a dead end: its destination is an object that is no page (a font). "v" is valid: it
    points at the first page.
    """
    font = 3 + pages
    first_link = font + 1
    root = first_link + len(links)
    outlines = b" /Outlines %d 0 R" % root if marks else b""
    annots = b" ".join(b"%d 0 R" % number for number in range(first_link, root))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R%s >>" % outlines,
        b"<< /Type /Pages /Kids [%s] /Count %d >>"
        % (b" ".join(b"%d 0 R" % number for number in range(3, 3 + pages)), pages),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Annots [%s] >>" % annots,
        *(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>" for _ in range(pages - 1)),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for number, kind in enumerate(links):
        objects.append(
            b"<< /Type /Annot /Subtype /Link /Rect [10 %d 100 %d] /Border [0 0 0] "
            b"/Dest [%d 0 R /XYZ 0 700 0] >>"
            % (10 + number, 19 + number, font if kind == "d" else 3)
        )
    if marks:
        first = root + 1
        objects.append(
            b"<< /Type /Outlines /First %d 0 R /Last %d 0 R /Count %d >>"
            % (first, first + len(marks) - 1, len(marks))
        )
        for number, kind in enumerate(marks):
            links_to = b"/Prev %d 0 R " % (first + number - 1) if number else b""
            if number < len(marks) - 1:
                links_to += b"/Next %d 0 R " % (first + number + 1)
            objects.append(
                b"<< /Title (Mark %d) /Parent %d 0 R %s/Dest [%d 0 R /Fit] >>"
                % (number, root, links_to, font if kind == "d" else 3)
            )
    return _pdfs.serialize_pdf(objects)


def _page_lookups(monkeypatch) -> list[int]:
    """What each call of FPDFDest_GetDestPageIndex gave, the call that costs a walk of the pages."""
    real = pdfium_raw.FPDFDest_GetDestPageIndex
    results: list[int] = []

    def recorded(*args):
        index = real(*args)
        results.append(index)
        return index

    monkeypatch.setattr(pdfium_raw, "FPDFDest_GetDestPageIndex", recorded)
    return results


def test_the_dead_end_fixture_points_at_no_page_and_the_last_link_at_the_page():
    paper = _paper_with_destinations("ddv", "dd")
    layer = build_document_layer(paper, [0], budget=_BUDGET)

    assert [link.target_page for link in layer.links] == [None, None, 0]
    assert [link.target_class for link in layer.links] == ["unresolved", "unresolved", "other"]
    assert [entry.page for entry in layer.outline] == [None, None]
    assert layer.component_errors == {}


def test_the_allowance_for_destinations_that_point_at_no_page_falls_as_the_page_count_grows():
    allowance = destinations.unresolved_allowance

    assert (destinations.MAX_PAGE_CHECKS, destinations.MIN_UNRESOLVED) == (5_000_000, 256)
    # A paper never reaches it: 5,000,000 page checks are 250,000 walks of 20 pages.
    assert allowance(20) == 250_000
    assert allowance(1) == allowance(0) == 5_000_000
    assert allowance(1_000) == 5_000
    assert allowance(19_000) == 263
    # From about 19,500 pages the floor holds.
    assert allowance(20_000) == 256
    assert allowance(1_000_000) == 256
    for pages in (1, 2, 20, 500, 5_000, 10_000, 19_000):
        assert allowance(pages) * pages <= destinations.MAX_PAGE_CHECKS


def _stub_resolver(n_pages: int):
    """A resolver whose destinations all point at no page, and the calls pdfium would get."""
    calls: list[object] = []
    api = SimpleNamespace(FPDFDest_GetDestPageIndex=lambda _raw, dest: calls.append(dest) or -1)
    return destinations.Resolver(api, SimpleNamespace(raw=None), n_pages), calls


def test_a_resolver_in_a_large_document_stops_at_the_floor_and_in_a_small_one_goes_on():
    large, large_calls = _stub_resolver(20_000)
    small, small_calls = _stub_resolver(20)

    assert (large.allowance, small.allowance) == (256, 250_000)
    for number in range(1_000):
        assert large.page(number) is None and small.page(number) is None
    # 256 destinations were looked up; the other 744 were not asked for.
    assert len(large_calls) == 256 and (large.unresolved, large.skipped) == (256, 744)
    assert len(small_calls) == 1_000 and (small.unresolved, small.skipped) == (1_000, 0)
    assert large.note is not None and small.note is None


def test_a_resolver_counts_the_destinations_that_point_at_no_page_and_stops_at_the_limit(
    monkeypatch,
):
    monkeypatch.setattr(destinations, "MIN_UNRESOLVED", 2)
    monkeypatch.setattr(destinations, "MAX_PAGE_CHECKS", 0)
    with _Doc(_linked.linked_paper()) as opened:
        calls = []
        real = opened.api.FPDFDest_GetDestPageIndex
        opened.api.FPDFDest_GetDestPageIndex = lambda *args: calls.append(1) or real(*args)
        resolver = destinations.Resolver(opened.api, opened.doc, 3)
        on_page, past_the_end = opened.named("section.2"), opened.named("figure.1")

        # figure.1 is on page 3, past the end of a document of three pages.
        assert resolver.page(on_page) == 2
        assert resolver.page(past_the_end) is None and resolver.unresolved == 1
        assert resolver.page(on_page) == 2 and resolver.unresolved == 1
        assert resolver.note is None and not resolver.skipped
        assert resolver.page(past_the_end) is None and resolver.unresolved == 2
        assert len(calls) == 4 and resolver.note is None

        # The allowance is spent: nothing more is resolved, one that has a page included.
        assert resolver.page(on_page) is None and resolver.page(past_the_end) is None
        assert len(calls) == 4 and resolver.skipped == 2
        assert resolver.note == (
            "after 2 destinations that point at no page, the rest are left unresolved"
        )


def test_a_document_stops_resolving_after_256_destinations_that_point_at_no_page(monkeypatch):
    calls = _page_lookups(monkeypatch)
    paper = _paper_with_destinations("d" * 300 + "v", pages=20_000)
    layer = build_document_layer(paper, [0], budget=_BUDGET)

    # 256 were resolved, with a walk of the 20,000 pages each; the other 44 and the one that
    # has a page were not asked for, so they point at none.
    assert len(calls) == 256
    assert len(layer.links) == 301
    assert [link.target_page for link in layer.links] == [None] * 301
    assert {link.target_class for link in layer.links} == {"unresolved"}
    assert layer.component_errors == {
        "unresolved_dests": "after 256 destinations that point at no page, the rest are left unresolved"
    }
    # With room for them all the last link has its page.
    with monkeypatch.context() as patch:
        patch.setattr(destinations, "MAX_PAGE_CHECKS", 10**12)
        unlimited = build_document_layer(paper, [0], budget=_BUDGET)
    assert unlimited.links[-1].target_page == 0 and unlimited.component_errors == {}


def test_exactly_256_destinations_that_point_at_no_page_are_all_resolved_and_say_nothing(
    monkeypatch,
):
    calls = _page_lookups(monkeypatch)
    paper = _paper_with_destinations("d" * 256, pages=20_000)
    layer = build_document_layer(paper, [0], budget=_BUDGET)

    assert len(calls) == 256 and len(layer.links) == 256
    assert layer.component_errors == {}


def test_a_paper_never_reaches_the_allowance(monkeypatch):
    # In a document of 20 pages 257 destinations that point at no page are far from the 250,000.
    calls = _page_lookups(monkeypatch)
    layer = build_document_layer(_paper_with_destinations("d" * 257, pages=20), [0], budget=_BUDGET)

    assert len(calls) == 257
    assert layer.component_errors == {}


def test_the_outline_and_the_links_share_the_limit(monkeypatch):
    monkeypatch.setattr(destinations, "MIN_UNRESOLVED", 3)
    monkeypatch.setattr(destinations, "MAX_PAGE_CHECKS", 0)
    calls = _page_lookups(monkeypatch)
    layer = build_document_layer(_paper_with_destinations("dddv", "dd"), [0], budget=_BUDGET)

    # The outline is read first and spends two of the three; the links get the third.
    assert len(calls) == 3
    assert [entry.page for entry in layer.outline] == [None, None]
    assert [link.target_page for link in layer.links] == [None] * 4
    assert list(layer.component_errors) == ["unresolved_dests"]


def test_destinations_that_have_a_page_do_not_count_against_the_limit(monkeypatch):
    results = _page_lookups(monkeypatch)
    whole = _layer(_linked.linked_paper())
    dead_ends = sum(1 for index in results if not 0 <= index < _linked.N_PAGES)
    # The paper has a few destinations that point at no page and many that have one.
    assert 0 < dead_ends < len(results) - 8

    # A limit one over those few, which the many do not use up, leaves everything as it was
    # read, with no note: the limit stops the destinations after the last of its count.
    monkeypatch.setattr(destinations, "MIN_UNRESOLVED", dead_ends + 1)
    monkeypatch.setattr(destinations, "MAX_PAGE_CHECKS", 0)
    limited = _layer(_linked.linked_paper())

    assert limited.links == whole.links and limited.outline == whole.outline
    assert limited.component_errors == {}


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
