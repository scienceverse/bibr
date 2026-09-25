"""A page over the render budget renders at a lower DPI instead of failing.

The renderer used to refuse any page above ``LAYOUT_MAX_RENDER_PIXELS`` or
``LAYOUT_MAX_RENDER_DIMENSION`` at the one configured DPI, and the paper failed
as ``layout_failed``: a 2420x3205 pt poster page is ~59.9 MP at 200 DPI. Every
box downstream is normalized by the rendered image's own size, so a reduced
page must place the same content at the same page coordinates.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.export.geometry import PageGeometry
from bibr.ocr.image_processing import crop_image_region
from bibr.ocr.image_utils import fitting_render_dpi, iter_pdf_pages_with_index


def _pdf(pages: list[tuple[float, float, str]]) -> bytes:
    """A PDF with one page per ``(width_pt, height_pt, content stream)``."""
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    kids = []
    for width, height, stream in pages:
        content = stream.encode("latin-1")
        page_obj = len(objects) + 1
        kids.append(f"{page_obj} 0 R")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:g} {height:g}] "
            f"/Contents {page_obj + 1} 0 R >>".encode()
        )
        objects.append(
            b"<< /Length "
            + str(len(content)).encode()
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        )
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode()
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


# A black box at x 100..200 pt, 100..150 pt below the top of a Letter page.
_BOX = "0 0 0 rg 100 642 100 50 re f"
_BOX_POINTS = (100.0, 100.0, 200.0, 150.0)


def _dark_bbox_normalized(image) -> tuple[float, float, float, float]:
    """The dark pixels' box in 0..1000 layout units of *image*."""
    mask = image.convert("L").point(lambda value: 255 if value < 128 else 0)
    x0, y0, x1, y1 = mask.getbbox()
    width, height = image.size
    return (x0 * 1000 / width, y0 * 1000 / height, x1 * 1000 / width, y1 * 1000 / height)


def test_fitting_dpi_for_a_poster_page():
    dpi = fitting_render_dpi(2420, 3205, 200, 25_000_000, 10_000)

    assert dpi == 129
    width, height = -(-2420 * dpi // 72), -(-3205 * dpi // 72)
    assert width * height <= 25_000_000
    assert -(-2420 * (dpi + 1) // 72) * -(-3205 * (dpi + 1) // 72) > 25_000_000


def test_fitting_dpi_respects_the_dimension_limit_and_the_configured_dpi():
    assert fitting_render_dpi(612, 7200, 200, 10**9, 10_000) == 100
    assert fitting_render_dpi(612, 792, 200, 25_000_000, 10_000) == 200


def test_oversized_page_renders_at_a_reduced_dpi_and_reports_it():
    reduced = []
    pdf = _pdf([(612, 792, _BOX), (612, 792, _BOX)])

    pages = list(
        iter_pdf_pages_with_index(
            pdf,
            dpi=200,
            max_pixels=100_000,
            min_dpi=20,
            on_reduced_dpi=lambda page, dpi: reduced.append((page, dpi)),
        )
    )

    assert reduced == [(0, 32), (1, 32)]
    assert [image.size for _, image in pages] == [(272, 352), (272, 352)]


def test_page_below_the_floor_is_still_refused():
    with pytest.raises(ValueError, match="pixel limit"):
        list(iter_pdf_pages_with_index(_pdf([(612, 792, _BOX)]), max_pixels=1_000, min_dpi=72))


def test_without_a_floor_the_renderer_keeps_refusing():
    with pytest.raises(ValueError, match="pixel limit"):
        list(iter_pdf_pages_with_index(_pdf([(612, 792, _BOX)]), max_pixels=100_000))


def test_only_the_oversized_page_is_reduced():
    reduced = []
    pdf = _pdf([(612, 792, _BOX), (2420, 3205, "")])

    pages = list(
        iter_pdf_pages_with_index(
            pdf,
            dpi=200,
            max_pixels=5_000_000,
            min_dpi=40,
            on_reduced_dpi=lambda page, dpi: reduced.append((page, dpi)),
        )
    )

    assert reduced == [(1, 57)]
    assert pages[0][1].size == (1700, 2200)  # page 1 keeps the configured DPI
    assert pages[1][1].size == (1916, 2538)


def test_reduced_page_places_boxes_at_the_same_page_coordinates():
    pdf = _pdf([(612, 792, _BOX)])
    [(_, full)] = iter_pdf_pages_with_index(pdf, dpi=144)
    [(_, reduced)] = iter_pdf_pages_with_index(pdf, dpi=200, max_pixels=100_000, min_dpi=20)
    assert reduced.size[0] < full.size[0] / 4

    full_box = _dark_bbox_normalized(full)
    reduced_box = _dark_bbox_normalized(reduced)
    # One pixel of the reduced render is ~3.7 layout units wide.
    assert reduced_box == pytest.approx(full_box, abs=5)

    # Layout box -> exported PDF points, as the export writes it.
    points = PageGeometry({1: (612, 792)}).box(1, reduced_box)
    assert points == pytest.approx(_BOX_POINTS, abs=3)

    # An OCR or figure crop of that layout box covers the box on the reduced page.
    crop = crop_image_region(reduced, [round(value) for value in reduced_box])
    assert crop.size == pytest.approx(
        (100 * reduced.size[0] / 612, 50 * reduced.size[1] / 792), abs=3
    )
    assert crop.convert("L").getextrema()[1] < 128  # all dark: nothing of the page around it


async def test_layout_stage_warns_for_each_reduced_page():
    from bibr.pipeline.context import PipelineContext, RunConfig
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.stages.layout import LayoutStage
    from bibr.pipeline.state import FileState

    def fake_render(pdf_bytes, dpi, start_page, end_page, max_pixels, max_dimension, **kwargs):
        assert kwargs["min_dpi"] == 72
        kwargs["on_reduced_dpi"](1, 129)
        return iter([(0, MagicMock()), (1, MagicMock())])

    fs = FileState(path=Path("poster.pdf"))
    fs.pdf_bytes = b"%PDF"
    resources = MagicMock()
    resources.layout = MagicMock(detect_batch=AsyncMock(return_value=[[], []]))
    ctx = PipelineContext(
        file_states=[fs], progress=NullProgress(), resources=resources, config=RunConfig()
    )

    with patch("bibr.pipeline.stages.layout._iter_pdf_pages", side_effect=fake_render):
        await LayoutStage().run(ctx)

    assert fs.error is None
    assert [(w.code, w.message) for w in fs.warnings] == [
        ("PAGE_DPI_REDUCED", "page 2 rendered at 129 DPI instead of 200 to fit the render budget")
    ]
