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
_TITLE = "Ultrahigh Carbon Steels, Damascus Steels and Ancient Blacksmiths"  # a metallurgy scan
_AUTHORS = "Oleg D. Sherby and Jeffrey Wadsworth"
_STAMP = "Downloaded by [Northwestern University] at 02:34 05 June 2016"  # a publisher-archive scan
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
    """The metallurgy scan: the image is painted first, then 795 mode-3 text objects."""
    pdf = _pdf([{"media": (0, 0, 592, 836), "content": _image(592, 836) + _text(_LAYER, mode=3)}])
    assert _page_verdicts(pdf) == [True]


def test_invisible_text_painted_before_the_image_is_a_scan():
    """Two scanned papers paint the hidden layer first, the scan last."""
    pdf = _pdf([{"media": (0, 0, 516, 728), "content": _text(_LAYER, mode=3) + _image(516, 728)}])
    assert _page_verdicts(pdf) == [True]


def test_scan_slightly_smaller_than_the_page_is_a_scan():
    """In one scanned paper the scan covers 0.94 of the page."""
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
    """A publisher-archive PDF draws the scan inside a Form XObject: the form's matrix
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
    """A scanned paper whose MediaBox and CropBox start at y = 6.72."""
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
    """A publisher-archive scan: every scanned page carries one visible download stamp."""
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
    """A born-digital paper: a page-sized template image under real visible text."""
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
    """Page 4 of a born-digital paper: 29 invisible of 1613 characters."""
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
    """Publisher-archive scans: a born-digital cover page, then scans."""
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


# --- OCR returns nothing for a scanned page ------------------------------------


def test_inspection_keeps_the_layer_text_as_an_ocr_fallback():
    """The flagged page's regions carry the text the fill would have taken."""
    inspection = _inspect(_cover_and_scan(), reject=True)

    cover, scan = (page[0] for page in inspection.layout_results)
    assert _TITLE in scan["_invisible_layer_text"]
    # Only a fallback: OCR still reads the region.
    assert scan["content"] == ""
    assert not scan.get("_native_text_used")
    assert "_invisible_layer_text" not in cover
    kept = _inspect(_cover_and_scan(), reject=False).layout_results[1][0]
    assert "_invisible_layer_text" not in kept


def _scan_regions() -> list[dict]:
    return [
        {
            "label": "doc_title",
            "bbox_2d": [100, 80, 900, 140],
            "content": "",
            "_invisible_layer_text": _TITLE,
        },
        {
            "label": "text",
            "bbox_2d": [100, 200, 900, 700],
            "content": "",
            "_invisible_layer_text": "\n".join(_BODY),
        },
        # No layer text inside it: nothing to fall back to.
        {"label": "table", "task_type": "table", "bbox_2d": [100, 720, 900, 900], "content": ""},
    ]


async def _ocr_scan_page(ocr_fn, warnings: list | None = None):
    from PIL import Image

    from bibr.pipeline.stages.ocr import ocr_page_regions

    fallback_pages: list[int] = []
    result = await ocr_page_regions(
        Image.new("RGB", (612, 792), "white"),
        _scan_regions(),
        1,
        "scan.pdf",
        ocr_fn,
        warning_sink=None if warnings is None else warnings.append,
        layer_fallback_sink=fallback_pages.append,
    )
    return result, fallback_pages


_BUSY = "RuntimeError: 503 Service Unavailable"


def _warning(code: str, message: str):
    from bibr.processing_warnings import ProcessingWarning

    return ProcessingWarning(code, message)


def _layer_warning(regions: str, error: str | None = None, *, page: int = 2):
    message = (
        "OCR returned no text for regions of a scanned page; read them from its "
        f"invisible text layer (page {page}, regions {regions})"
    )
    return _warning("OCR_TEXT_LAYER_FALLBACK", message + (f": {error}" if error else ""))


def _region_failed(region: int, task: str):
    return _warning(
        "OCR_REGION_FAILED",
        f"OCR failed for a region; its text is missing (page 2, region {region}, task {task}): "
        + _BUSY,
    )


@pytest.mark.parametrize("failure", ["empty", "raises"])
async def test_scanned_page_falls_back_to_its_layer_when_ocr_returns_nothing(failure):
    """An OCR outage (blank answers, or busy errors after the retries) must not
    blank a page whose text layer the fill used to read."""

    async def broken_ocr(image, prompt):
        if failure == "raises":
            raise RuntimeError("503 Service Unavailable")
        return ""

    warnings: list = []
    result, fallback_pages = await _ocr_scan_page(broken_ocr, warnings)

    assert [region["content"] for region in result] == [_TITLE, "\n".join(_BODY), ""]
    # Restored exactly as the fill would have filled them, so the OCR success
    # gate and the cleanup treat them as text-layer regions.
    assert [region.get("_native_text_used") for region in result] == [True, True, None]
    assert fallback_pages == [1]
    assert not any("_invisible_layer_text" in region for region in result)
    # The export says where the text came from, and only the table, which had
    # no layer text, is reported missing.
    if failure == "raises":
        assert warnings == [_region_failed(2, "table"), _layer_warning("0, 1", _BUSY)]
    else:
        assert warnings == [_layer_warning("0, 1")]


async def test_ocr_text_wins_over_the_layer():
    async def ocr(image, prompt):
        return "Ultrahigh Carbon Steels"

    result, fallback_pages = await _ocr_scan_page(ocr)

    assert [region["content"] for region in result] == ["Ultrahigh Carbon Steels"] * 3
    assert not any(region.get("_native_text_used") for region in result)
    assert fallback_pages == []


async def test_one_region_read_by_ocr_keeps_the_layer_out():
    """OCR that reads part of the page is working: an empty region stays empty."""
    answers = iter(["Ultrahigh Carbon Steels", "", ""])

    async def ocr(image, prompt):
        return next(answers)

    result, fallback_pages = await _ocr_scan_page(ocr)

    assert sorted(region["content"] for region in result) == ["", "", "Ultrahigh Carbon Steels"]
    assert not any(region.get("_native_text_used") for region in result)
    assert fallback_pages == []


def _partly_read_regions() -> list[dict]:
    body = [
        {"label": "text", "bbox_2d": [100, 200 + 150 * i, 900, 330 + 150 * i], "content": ""}
        for i in range(3)
    ]
    for region, line in zip(body, _BODY, strict=True):
        region["_invisible_layer_text"] = line
    return [_scan_regions()[0], *body, _scan_regions()[2]]


async def test_failed_regions_fall_back_on_a_page_ocr_read_elsewhere():
    """OCR reads the title, two body regions fail after the retries, one body
    region and the table (no layer text) fail or answer blank: each failed
    region with layer text takes it, a blank answer stays blank."""
    from PIL import Image

    from bibr.pipeline.stages.ocr import ocr_page_regions

    answers = iter(
        [
            "Ultrahigh Carbon Steels",
            RuntimeError("503 Service Unavailable"),
            RuntimeError("503 Service Unavailable"),
            "",
            RuntimeError("503 Service Unavailable"),
        ]
    )

    async def ocr(image, prompt):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    fallback_pages: list[int] = []
    warnings: list = []
    result = await ocr_page_regions(
        Image.new("RGB", (612, 792), "white"),
        _partly_read_regions(),
        1,
        "scan.pdf",
        ocr,
        warning_sink=warnings.append,
        layer_fallback_sink=fallback_pages.append,
    )

    assert [region["content"] for region in result] == [
        "Ultrahigh Carbon Steels",
        _BODY[0],
        _BODY[1],
        "",
        "",
    ]
    assert [region.get("_native_text_used") for region in result] == [
        None,
        True,
        True,
        None,
        None,
    ]
    assert fallback_pages == [1]
    # Refilled regions are not reported missing; the table is.
    assert warnings == [_region_failed(4, "table"), _layer_warning("1, 2", _BUSY)]


async def test_healthy_ocr_on_a_scanned_page_ignores_the_layer_entirely():
    """With OCR answering, a flagged page's output and warnings are those of
    the same page without layer text: the fallback leaves no trace."""
    from PIL import Image

    from bibr.pipeline.stages.ocr import ocr_page_regions

    async def ocr(image, prompt):
        return f"OCR text {image.size}"

    async def run(regions):
        warnings: list = []
        fallback_pages: list[int] = []
        result = await ocr_page_regions(
            Image.new("RGB", (612, 792), "white"),
            regions,
            1,
            "scan.pdf",
            ocr,
            warning_sink=warnings.append,
            layer_fallback_sink=fallback_pages.append,
        )
        return result, warnings, fallback_pages

    flagged = await run(_partly_read_regions())
    plain = [
        {key: value for key, value in region.items() if key != "_invisible_layer_text"}
        for region in _partly_read_regions()
    ]
    assert flagged == await run(plain)
    assert flagged[1:] == ([], [])


def _cover_and_two_scans() -> bytes:
    scan = {"content": _image(612, 792) + _text([*_LAYER, *_REFERENCES], mode=3)}
    return _pdf(
        [{"content": _text(["Full Terms & Conditions of access and use", _TITLE])}, scan, scan]
    )


@pytest.mark.parametrize(
    "answer", ["", RuntimeError("503 Service Unavailable"), "An OCR reading of the page"]
)
async def test_ocr_stage_reads_the_layer_only_when_ocr_returns_nothing(
    monkeypatch, caplog, tmp_path, answer
):
    """NativeTextStage and OcrStage on a cover and two scanned pages: blank OCR
    answers or failed requests fall back to each scan's layer, logged once for
    the document, flagged in the export and kept out of the OCR cache."""
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock

    from PIL import Image

    import bibr.pipeline.stages.ocr as ocr_mod
    from bibr.config import Settings
    from bibr.pipeline.context import PipelineContext, RunConfig, StageSignals
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.stages.native_text import NativeTextStage
    from bibr.pipeline.state import FileState

    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)
    monkeypatch.setattr(Settings.ocr, "native_text_reject_invisible_layer", True)
    monkeypatch.setattr(Settings.cache, "ocr", True)
    monkeypatch.setattr(Settings.cache, "ocr_dir", str(tmp_path))
    fs = FileState(path=Path("scan.pdf"))
    fs.pdf_bytes = _cover_and_two_scans()
    fs.file_hash = "0" * 64
    fs.layout_results = _full_page_layout(3)
    fs.page_indices = [0, 1, 2]
    fs.page_images = [Image.new("RGB", (612, 792), "white") for _ in range(3)]
    rm = MagicMock()
    recognize = (
        AsyncMock(side_effect=answer)
        if isinstance(answer, Exception)
        else AsyncMock(return_value=answer)
    )
    rm.ocr = MagicMock(recognize=recognize, loaded=True)
    rm.await_ocr = AsyncMock(return_value=None)
    rm.shutdown_ocr = AsyncMock(return_value=None)
    del rm.ocr.wait_for_server
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=rm,
        config=RunConfig(ocr_backend="glm-llama"),
        signals=StageSignals(any_needs_ocr=True, preloading_ocr=False),
    )

    await NativeTextStage().run(ctx)
    with caplog.at_level("WARNING", logger=ocr_mod.__name__):
        await ocr_mod.OcrStage().run(ctx)

    assert fs.error is None
    assert rm.ocr.recognize.await_count == 2  # the cover page stays native
    cover, *scans = (page[0].content for page in fs.ocr_regions)
    assert _TITLE in cover
    fallback_logs = [r for r in caplog.records if "invisible text layer" in r.getMessage()]
    cached = list(tmp_path.glob("*.json"))
    if isinstance(answer, str) and answer:
        assert scans == [answer, answer]
        assert fallback_logs == []
        assert fs.warnings == []
        assert len(cached) == 1
    else:
        assert all(_TITLE in scan and "Wadsworth J, Sherby OD" in scan for scan in scans)
        error = None if isinstance(answer, str) else _BUSY
        assert fs.warnings == [
            _layer_warning("0", error, page=2),
            _layer_warning("0", error, page=3),
        ]
        # Not a final answer: serve does not cache the export, and the OCR
        # cache does not keep the layer text for the next, healthy run.
        from bibr.serve.deployments.pipeline import _is_final_result

        payload = {"extraction": {"warnings": [w.to_dict() for w in fs.warnings]}}
        assert _is_final_result(payload) is False
        assert cached == []
        assert [r.getMessage() for r in fallback_logs] == [
            "OCR returned no text for regions of 2 scanned page(s) of scan.pdf (pages 2, 3); "
            "read those regions from their invisible text layer instead"
        ]


# --- detection failures are reported -------------------------------------------


async def _run_native_text_stage(monkeypatch, caplog, pdf_bytes: bytes, pages: int):
    from pathlib import Path
    from unittest.mock import MagicMock

    import bibr.pipeline.stages.native_text as stage_mod
    from bibr.config import Settings
    from bibr.pipeline.context import PipelineContext, RunConfig
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.state import FileState

    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)
    monkeypatch.setattr(Settings.ocr, "native_text_reject_invisible_layer", True)
    fs = FileState(path=Path("scan.pdf"))
    fs.pdf_bytes = pdf_bytes
    fs.layout_results = _full_page_layout(pages)
    ctx = PipelineContext(
        file_states=[fs], progress=NullProgress(), resources=MagicMock(), config=RunConfig()
    )
    with caplog.at_level("INFO", logger=stage_mod.__name__):
        await stage_mod.NativeTextStage().run(ctx)
    detection_logs = [
        record
        for record in caplog.records
        if record.name == stage_mod.__name__ and "layer detection" in record.getMessage()
    ]
    return fs, detection_logs


async def test_detection_errors_are_logged_once_per_document(monkeypatch, caplog):
    """A failing detector keeps every page on its text layer, silently before."""
    import bibr.ocr.pdf_inspection as inspection_mod

    def boom(page, textpage, crop_box):
        raise RuntimeError("pdfium")

    monkeypatch.setattr(inspection_mod, "_is_invisible_text_layer_page", boom)
    fs, logs = await _run_native_text_stage(monkeypatch, caplog, _cover_and_two_scans(), 3)

    assert all(page[0]["_native_text_used"] for page in fs.layout_results)
    assert [(record.levelname, record.getMessage()) for record in logs] == [
        (
            "INFO",
            "Invisible OCR-layer detection failed on 3 page(s) of scan.pdf; they keep "
            "their text layer (RuntimeError: pdfium)",
        )
    ]


async def test_missing_pdfium_api_is_reported_not_skipped(monkeypatch, caplog):
    """A PDFium build without FPDFText_GetTextObject cannot tell invisible text
    apart: the scan keeps its layer, and the run says why."""
    import pypdfium2.raw as pdfium_c

    monkeypatch.delattr(pdfium_c, "FPDFText_GetTextObject")
    fs, logs = await _run_native_text_stage(monkeypatch, caplog, _cover_and_scan(), 2)

    cover, scan = (page[0] for page in fs.layout_results)
    assert cover["_native_text_used"] is True
    assert scan["_native_text_used"] is True
    # Only the scan got far enough to need the API: the cover's images do not
    # cover the page.
    assert set(fs.pdf_inspection.component_errors) == {"invisible_text_layer:1"}
    assert len(logs) == 1
    assert "failed on 1 page(s) of scan.pdf" in logs[0].getMessage()
    assert "FPDFText_GetTextObject" in logs[0].getMessage()


async def test_clean_detection_logs_nothing(monkeypatch, caplog):
    _, logs = await _run_native_text_stage(monkeypatch, caplog, _cover_and_scan(), 2)
    assert logs == []
