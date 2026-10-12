"""The native text reader's document-layer hooks only append to side lists.

``_build_page_char_records(textpage, trace)`` and ``open_text_page(page,
watermarks, furniture, walk)`` feed the document layer. With or without the
hook arguments they return the same records and leave the same text page, and
a failure while recording never changes what the furniture strip removes.
"""

from __future__ import annotations

import pypdfium2
import pytest

from bibr.document import harvest, serialize
from bibr.document.harvest import build_document_layer
from bibr.ocr import native_text
from bibr.ocr.native_text import (
    DEFAULT_ELIGIBLE_LABELS,
    PageCharTrace,
    _build_page_char_records,
    _fill_page_regions_from_textpage,
    _page_crop_box,
    _page_rotation,
    open_text_page,
)
from bibr.ocr.utils import pdfium_lock
from tests.document import _pdfs
from tests.document._pdfs import band_layout, fixture_pdfs
from tests.ocr.test_line_number_column import _reference_page
from tests.ocr.test_watermark_text import _REVIEW_WATERMARK, _body, _pdf

_FIXTURES = fixture_pdfs()


def _pages(pdf_bytes: bytes):
    """``(page_index, page)`` of a freshly opened document, under the lock."""
    with pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            for index in range(len(doc)):
                page = doc[index]
                try:
                    yield index, page
                finally:
                    page.close()
        finally:
            doc.close()


def _text_page_state(textpage) -> tuple[int, str, list]:
    n_chars = textpage.count_chars()
    return n_chars, textpage.get_text_range(), [textpage.get_charbox(i) for i in range(n_chars)]


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_trace_leaves_the_records_unchanged(name):
    for _index, page in _pages(_FIXTURES[name]):
        textpage = open_text_page(page)
        try:
            plain = _build_page_char_records(textpage)
            trace = PageCharTrace()
            traced = _build_page_char_records(textpage, trace)
            n_chars = textpage.count_chars()
        finally:
            textpage.close()
        assert traced == plain
        assert trace.complete
        assert len(trace.codes) == len(trace.boxes) == n_chars
        assert len(trace.record_src) == len(trace.record_flags) == len(traced)
        assert all(-1 <= src < n_chars for src in trace.record_src)
        # A record without a pdfium char is an inserted space.
        assert all(
            traced[k][0] == " "
            for k, src in enumerate(trace.record_src)
            if src < 0 and not traced[k][3]
        )


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_shared_records_leave_the_fill_unchanged(name):
    for index, page in _pages(_FIXTURES[name]):
        crop_box = _page_crop_box(page)
        rotation = _page_rotation(page)
        fills = []
        for share in (False, True):
            regions = band_layout(1)[0]
            textpage = open_text_page(page)
            try:
                records = _build_page_char_records(textpage, PageCharTrace()) if share else None
                _fill_page_regions_from_textpage(
                    textpage,
                    crop_box,
                    regions,
                    min_chars=3,
                    eligible_labels=DEFAULT_ELIGIBLE_LABELS,
                    min_printable_ratio=0.85,
                    page_idx=index,
                    rotation=rotation,
                    records=records,
                )
            finally:
                textpage.close()
            fills.append(regions)
        assert fills[0] == fills[1]


def _furniture_cases() -> dict[str, bytes]:
    return {
        "watermark": _pdf(_REVIEW_WATERMARK + _body()),
        "line_numbers": _reference_page(),
        "clean": _pdf(_body()),
    }


@pytest.mark.parametrize("case", sorted(_furniture_cases()))
def test_furniture_list_leaves_the_text_page_unchanged(case):
    pdf_bytes = _furniture_cases()[case]
    plain_runs = []
    for _index, page in _pages(pdf_bytes):
        watermarks: list[str] = []
        textpage = open_text_page(page, watermarks)
        try:
            plain_runs.append((watermarks, _text_page_state(textpage)))
        finally:
            textpage.close()
    hooked_runs = []
    furniture_by_page = []
    for _index, page in _pages(pdf_bytes):
        watermarks = []
        furniture: list[tuple] = []
        textpage = open_text_page(page, watermarks, furniture, [])
        try:
            hooked_runs.append((watermarks, _text_page_state(textpage)))
        finally:
            textpage.close()
        furniture_by_page.append(furniture)

    assert hooked_runs == plain_runs
    kinds = [[kind for kind, _box, _text in found] for found in furniture_by_page]
    if case == "watermark":
        assert kinds == [["watermark"]]
        _kind, box, text = furniture_by_page[0][0]
        assert text == "For Review Only"
        left, bottom, right, top = box
        assert 0 <= left < right <= 612 and 0 <= bottom < top <= 792
    elif case == "line_numbers":
        assert all(
            kinds_on_page and set(kinds_on_page) == {"line_number"} for kinds_on_page in kinds
        )
        texts = [text for _kind, _box, text in furniture_by_page[0]]
        assert texts == [str(668 + offset) for offset in range(len(texts))]
    else:
        assert kinds == [[]]


def test_a_bookkeeping_failure_still_removes_the_watermark(monkeypatch):
    pdf_bytes = _pdf(_REVIEW_WATERMARK + _body())
    expected = []
    for _index, page in _pages(pdf_bytes):
        textpage = open_text_page(page, [])
        try:
            expected.append(_text_page_state(textpage))
        finally:
            textpage.close()

    def broken(*_args, **_kwargs):
        raise RuntimeError("no bounds")

    monkeypatch.setattr(native_text, "_watermark_details", broken)
    for _index, page in _pages(pdf_bytes):
        watermarks: list[str] = []
        furniture: list[tuple] = []
        textpage = open_text_page(page, watermarks, furniture)
        try:
            state = _text_page_state(textpage)
        finally:
            textpage.close()

        assert state == expected[0]
        assert watermarks == ["For Review Only"]
        assert furniture == [("error", None, "RuntimeError: no bounds")]


def test_a_failed_furniture_pass_reads_the_page_as_it_is_and_says_so(monkeypatch):
    pdf_bytes = _pdf(_REVIEW_WATERMARK + _body())

    def broken(*_args, **_kwargs):
        raise RuntimeError("scan failed")

    monkeypatch.setattr(native_text, "_find_furniture", broken)
    states = []
    for _index, page in _pages(pdf_bytes):
        furniture: list[tuple] = []
        for hooked in (False, True):
            textpage = open_text_page(page, [], furniture if hooked else None)
            try:
                states.append(_text_page_state(textpage))
            finally:
                textpage.close()

        assert states[0] == states[1]
        assert "Review" in states[0][1]
        assert furniture == [("error", None, "RuntimeError: scan failed")]


@pytest.mark.parametrize("case", sorted(_furniture_cases()))
def test_the_walk_holds_the_text_objects_left_after_the_strip(case):
    api = harvest._Api()
    for _index, page in _pages(_furniture_cases()[case]):
        before: list = []
        native_text._find_furniture(page, walk=before)
        walk: list = []
        furniture: list[tuple] = []
        textpage = open_text_page(page, [], furniture, walk)
        textpage.close()

        reused = {harvest._address(obj): matrix for _parent, obj, matrix in walk}
        assert reused == harvest._objects_in_page(api, page)
        assert len(walk) == len(before) - len(furniture)
        assert walk if case == "clean" else len(walk) < len(before)


def test_the_layer_reads_its_objects_from_the_strip_walk(monkeypatch):
    pdf_bytes = _pdfs.synthetic_paper()
    pages = range(len(_pdfs.SYNTHETIC_TEXT_SOURCES))
    reused = build_document_layer(pdf_bytes, pages, budget=None)
    read = harvest._read_objects

    def walk_again(api, page, handles, fonts, found=None):
        return read(api, page, handles, fonts, None)

    monkeypatch.setattr(harvest, "_read_objects", walk_again)
    walked_again = build_document_layer(pdf_bytes, pages, budget=None)
    monkeypatch.setattr(harvest, "_read_objects", read)

    def no_second_walk(*_args):
        raise AssertionError("the page was walked again")

    monkeypatch.setattr(harvest, "_objects_in_page", no_second_walk)
    without_fallback = build_document_layer(pdf_bytes, pages, budget=None)

    assert serialize.digest(walked_again) == serialize.digest(reused)
    assert without_fallback.component_errors == {}
    assert serialize.digest(without_fallback) == serialize.digest(reused)


def test_a_failed_strip_leaves_the_walk_empty(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("removal failed")

    monkeypatch.setattr(native_text, "_remove_objects", broken)
    for _index, page in _pages(_pdf(_REVIEW_WATERMARK + _body())):
        walk: list = []
        furniture: list[tuple] = []
        textpage = open_text_page(page, [], furniture, walk)
        textpage.close()

        assert walk == []
        assert furniture == [("error", None, "RuntimeError: removal failed")]
