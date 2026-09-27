"""A scanned page with an invisible OCR text layer is read like a scan.

Old scans often carry the searchable layer a legacy OCR engine added: the
scan is a page-sized image and the recognised text lies over it in render
mode 3. The fixtures are minimal hand-built PDFs that copy the structure of
the dev scans (page size, paint order, form nesting, crop offset) with one or
two lines of their front matter; born-digital shapes must keep their text.
"""

from __future__ import annotations

import pytest

# Front matter of the dev targets (gold titles, a stamp, an author line).
_TITLE = "Ultrahigh Carbon Steels, Damascus Steels and Ancient Blacksmiths"  # W1968665195
_AUTHORS = "Oleg D. Sherby and Jeffrey Wadsworth"
_STAMP = "Downloaded by [Northwestern University] at 02:34 05 June 2016"  # W2061615921
_BODY = [
    "The history of ultrahigh carbon steels is traced from ancient times.",
    "Damascus swords were forged from cakes of steel made in India.",
    "Their surface shows a pattern of carbide bands in a pearlite matrix.",
]


def _num(value: float) -> bytes:
    return (f"{value:.2f}").encode()


def _text(lines, *, mode: int = 0, x: float = 72, y: float = 700, size: float = 12) -> bytes:
    ops = [b"BT /F1 " + _num(size) + b" Tf " + str(mode).encode() + b" Tr"]
    for index, line in enumerate(lines):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(
            b"1 0 0 1 "
            + _num(x)
            + b" "
            + _num(y - 1.5 * size * index)
            + b" Tm ("
            + escaped.encode("latin-1")
            + b") Tj"
        )
    ops.append(b"ET")
    return b"\n".join(ops) + b"\n"


def _image(width: float, height: float, x: float = 0, y: float = 0) -> bytes:
    return b"q " + b" ".join(_num(v) for v in (width, 0, 0, height, x, y)) + b" cm /Im1 Do Q\n"


def _form(scale: float) -> bytes:
    return b"q " + b" ".join(_num(v) for v in (scale, 0, 0, scale, 0, 0)) + b" cm /Fm1 Do Q\n"


def _pdf(pages: list[dict]) -> bytes:
    """Build a PDF; each page is ``{"content", "media", "form"?}``.

    ``media`` is the MediaBox ``(x0, y0, x1, y1)``; ``form`` is
    ``(content, (w, h))`` for a Form XObject ``/Fm1`` that can draw the image.
    """
    objects: list[bytes] = [b"", b""]  # catalog and page tree, filled last

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    image = add(
        b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray "
        b"/BitsPerComponent 8 /Length 4 >>\nstream\n\xff\x80\x80\xff\nendstream"
    )
    kids = []
    for page in pages:
        xobjects = b"/Im1 %d 0 R" % image
        if "form" in page:
            content, (width, height) = page["form"]
            form = add(
                b"<< /Type /XObject /Subtype /Form /BBox [0 0 "
                + _num(width)
                + b" "
                + _num(height)
                + b"] /Resources << /XObject << /Im1 %d 0 R >> /Font << /F1 %d 0 R >> >> "
                % (image, font)
                + b"/Length %d >>\nstream\n" % len(content)
                + content
                + b"\nendstream"
            )
            xobjects += b" /Fm1 %d 0 R" % form
        stream = add(
            b"<< /Length %d >>\nstream\n" % len(page["content"]) + page["content"] + b"\nendstream"
        )
        media = b" ".join(_num(v) for v in page.get("media", (0, 0, 612, 792)))
        kids.append(
            add(
                b"<< /Type /Page /Parent 2 0 R /MediaBox [" + media + b"] "
                b"/Resources << /Font << /F1 %d 0 R >> /XObject << "
                % font
                + xobjects
                + b" >> >> /Contents %d 0 R >>" % stream
            )
        )
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % kid for kid in kids),
        len(kids),
    )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def _page_verdicts(pdf_bytes: bytes) -> list[bool]:
    import pypdfium2

    from bibr.ocr.native_text import _is_invisible_text_layer_page, _page_crop_box
    from bibr.ocr.utils import pdfium_lock

    verdicts = []
    with pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            for index in range(len(doc)):
                page = doc[index]
                textpage = page.get_textpage()
                try:
                    verdicts.append(
                        _is_invisible_text_layer_page(page, textpage, _page_crop_box(page))
                    )
                finally:
                    textpage.close()
                    page.close()
        finally:
            doc.close()
    return verdicts


def _coverage(pdf_bytes: bytes) -> float:
    import pypdfium2

    from bibr.ocr.native_text import _page_crop_box, _page_image_coverage
    from bibr.ocr.utils import pdfium_lock

    with pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            page = doc[0]
            try:
                return _page_image_coverage(page, _page_crop_box(page))
            finally:
                page.close()
        finally:
            doc.close()


_LAYER = [_TITLE, _AUTHORS, *_BODY]


# --- the page predicate --------------------------------------------------------


def test_page_image_under_invisible_text_is_a_scan():
    """W1968665195: the scan is painted first, then 795 mode-3 text objects."""
    pdf = _pdf([{"media": (0, 0, 592, 836), "content": _image(592, 836) + _text(_LAYER, mode=3)}])
    assert _page_verdicts(pdf) == [True]


def test_invisible_text_painted_before_the_image_is_a_scan():
    """W2020826466 / W2056096589 paint the hidden layer first, the scan last."""
    pdf = _pdf([{"media": (0, 0, 516, 728), "content": _text(_LAYER, mode=3) + _image(516, 728)}])
    assert _page_verdicts(pdf) == [True]


def test_scan_slightly_smaller_than_the_page_is_a_scan():
    """W2928210665: the scan covers 0.94 of the page."""
    pdf = _pdf(
        [
            {
                "media": (0, 0, 595.2, 841.8),
                "content": _image(578, 818, 8, 12) + _text(_LAYER, mode=3),
            }
        ]
    )
    assert _page_verdicts(pdf) == [True]


def test_scan_inside_a_scaled_form_covers_the_page():
    """W2094242498 draws the scan inside a Form XObject: the form's matrix
    maps the image box, reported in form space, onto the page."""
    pdf = _pdf(
        [
            {
                "media": (0, 0, 478.8, 669.1),
                "form": (_image(239.4, 334.55), (239.4, 334.55)),
                "content": _form(2) + _text(_LAYER, mode=3),
            }
        ]
    )
    assert _coverage(pdf) == pytest.approx(1.0, abs=0.01)
    assert _page_verdicts(pdf) == [True]


def test_scan_on_an_offset_page_box_is_a_scan():
    """W2042044793: MediaBox and CropBox start at y = 6.72."""
    pdf = _pdf(
        [
            {
                "media": (0, 6.72, 612, 798.72),
                "content": _image(612, 792, 0, 6.72) + _text(_LAYER, mode=3),
            }
        ]
    )
    assert _page_verdicts(pdf) == [True]


def test_visible_stamp_over_the_hidden_layer_is_still_a_scan():
    """W2061615921: every scanned page carries one visible download stamp."""
    pdf = _pdf(
        [
            {
                "media": (0, 0, 418.3, 672.7),
                "content": _image(418.3, 672.7)
                + _text(_LAYER * 2, mode=3, x=40, y=600, size=9)
                + _text([_STAMP], x=40, y=20, size=7),
            }
        ]
    )
    assert _page_verdicts(pdf) == [True]


def test_clip_only_text_counts_as_invisible():
    pdf = _pdf([{"content": _image(612, 792) + _text(_LAYER, mode=7)}])
    assert _page_verdicts(pdf) == [True]


def test_born_digital_page_on_a_background_image_keeps_its_text():
    """W4308442706: a page-sized template image under real visible text."""
    pdf = _pdf([{"content": _image(612, 792) + _text(_LAYER)}])
    assert _page_verdicts(pdf) == [False]


def test_visible_text_page_is_not_a_scan():
    pdf = _pdf([{"content": _text(_LAYER)}])
    assert _page_verdicts(pdf) == [False]


def test_figure_sized_image_with_invisible_text_is_not_a_scan():
    """Invisible text over a figure is not a scanned page."""
    pdf = _pdf([{"content": _image(400, 300, 100, 400) + _text(_LAYER, mode=3)}])
    assert _page_verdicts(pdf) == [False]


def test_a_little_invisible_text_does_not_make_a_scan():
    """10.17951_kw p4: 29 invisible of 1613 characters on a born-digital page."""
    pdf = _pdf(
        [
            {
                "content": _image(612, 792)
                + _text(_LAYER * 3)
                + _text(["hidden alt text"], mode=3, y=100),
            }
        ]
    )
    assert _page_verdicts(pdf) == [False]


def test_visible_text_under_an_opaque_scan_is_not_flagged():
    """Paint order is deliberately not used: visible text hidden under a later
    image stays on the text layer (PDFium cannot see an image's soft mask)."""
    pdf = _pdf([{"content": _text(_LAYER) + _image(612, 792)}])
    assert _page_verdicts(pdf) == [False]


def test_image_only_page_has_no_layer_to_reject():
    pdf = _pdf([{"content": _image(612, 792)}])
    assert _page_verdicts(pdf) == [False]


# --- inspect_pdf ---------------------------------------------------------------

_REFERENCES = ["References", "1. Wadsworth J, Sherby OD. Progress in Materials Science 1980."]


def _cover_and_scan() -> bytes:
    """W2061615921 / W2094242498: a born-digital cover page, then scans."""
    return _pdf(
        [
            {"content": _text(["Full Terms & Conditions of access and use", _TITLE])},
            {"content": _image(612, 792) + _text([*_LAYER, *_REFERENCES], mode=3)},
        ]
    )


def _full_page_layout(pages: int) -> list[list[dict]]:
    return [[{"label": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}] for _ in range(pages)]


def _inspect(pdf_bytes: bytes, *, reject: bool):
    from bibr.ocr.pdf_inspection import inspect_pdf

    return inspect_pdf(
        pdf_bytes,
        _full_page_layout(2),
        fill_native_text=True,
        include_outline=False,
        include_ref_geometry=True,
        min_chars=20,
        min_printable_ratio=0.85,
        reject_invisible_text_layer=reject,
    )


def test_inspection_leaves_the_scanned_page_to_ocr():
    inspection = _inspect(_cover_and_scan(), reject=True)

    assert [page.invisible_text_layer for page in inspection.pages] == [False, True]
    cover, scan = (page[0] for page in inspection.layout_results)
    assert cover["_native_text_used"] is True
    assert _TITLE in cover["content"]
    assert not scan.get("_native_text_used")
    assert scan["content"] == ""
    assert "_font_size" not in scan
    # No text-layer lines from the scan: neither the line stream nor the
    # geometry segmenter's reference lines.
    assert {line["page"] for line in inspection.page_lines} == {1}
    assert inspection.reference_lines == []
    assert inspection.component_errors == {}


def test_inspection_trusts_the_layer_when_disabled():
    inspection = _inspect(_cover_and_scan(), reject=False)

    assert [page.invisible_text_layer for page in inspection.pages] == [False, False]
    scan = inspection.layout_results[1][0]
    assert scan["_native_text_used"] is True
    assert _TITLE in scan["content"]
    assert {line["page"] for line in inspection.page_lines} == {1, 2}
    assert inspection.reference_lines


def test_detection_failure_keeps_the_text_layer(monkeypatch):
    import bibr.ocr.pdf_inspection as inspection_mod

    def boom(page, textpage, crop_box):
        raise RuntimeError("pdfium")

    monkeypatch.setattr(inspection_mod, "_is_invisible_text_layer_page", boom)
    inspection = _inspect(_cover_and_scan(), reject=True)

    assert inspection.layout_results[1][0]["_native_text_used"] is True
    assert set(inspection.component_errors) == {
        "invisible_text_layer:0",
        "invisible_text_layer:1",
    }


def test_page_flag_survives_the_ocr_cache_round_trip():
    from bibr.ocr.pdf_inspection import inspection_from_dict, inspection_to_dict

    data = inspection_to_dict(_inspect(_cover_and_scan(), reject=True))
    restored = inspection_from_dict(data, metadata={}, outline=[], reference_lines=[])
    assert [page.invisible_text_layer for page in restored.pages] == [False, True]

    for page in data["pages"]:
        del page["invisible_text_layer"]  # a bundle written before the flag existed
    restored = inspection_from_dict(data, metadata={}, outline=[], reference_lines=[])
    assert [page.invisible_text_layer for page in restored.pages] == [False, False]


# --- the pipeline stage --------------------------------------------------------


@pytest.mark.parametrize("reject", [True, False])
async def test_native_text_stage_leaves_the_scanned_page_to_ocr(monkeypatch, reject):
    from pathlib import Path
    from unittest.mock import MagicMock

    from bibr.config import Settings
    from bibr.pipeline.context import PipelineContext, RunConfig
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.stages.native_text import NativeTextStage
    from bibr.pipeline.state import FileState

    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)
    monkeypatch.setattr(Settings.ocr, "native_text_reject_invisible_layer", reject)
    fs = FileState(path=Path("scan.pdf"))
    fs.pdf_bytes = _cover_and_scan()
    fs.layout_results = _full_page_layout(2)
    ctx = PipelineContext(
        file_states=[fs], progress=NullProgress(), resources=MagicMock(), config=RunConfig()
    )

    await NativeTextStage().run(ctx)

    # OcrStage skips exactly the regions marked _native_text_used.
    cover, scan = (page[0] for page in fs.layout_results)
    assert cover["_native_text_used"] is True
    assert bool(scan.get("_native_text_used")) is not reject
    assert (scan["content"] == "") is reject
    assert [page.invisible_text_layer for page in fs.pdf_inspection.pages] == [False, reject]
    assert fs.error is None


# --- the setting ---------------------------------------------------------------


def test_setting_defaults_on_and_can_be_disabled(monkeypatch):
    from bibr.config import GlobalSettings

    assert GlobalSettings().ocr.native_text_reject_invisible_layer is True
    monkeypatch.setenv("OCR_NATIVE_TEXT_REJECT_INVISIBLE_LAYER", "false")
    assert GlobalSettings().ocr.native_text_reject_invisible_layer is False
