"""Whole-figure images composited from panel crops."""

from __future__ import annotations

import base64
import io

from PIL import Image

from bibr.paper_contents import PaperFigurePart
from bibr.structure.float_images import composite_panel_image


def _crop(color: str, size: tuple[int, int], fmt: str = "JPEG") -> str:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format=fmt)
    return base64.b64encode(out.getvalue()).decode("ascii")


def _decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def test_panels_are_placed_by_their_boxes_on_one_canvas():
    # Two side-by-side panels, 2 px per box unit on both axes.
    parts = [
        PaperFigurePart(
            page_number=3, bbox=(100, 50, 150, 100), image_b64=_crop("red", (100, 100), "PNG")
        ),
        PaperFigurePart(
            page_number=3, bbox=(200, 50, 250, 100), image_b64=_crop("blue", (100, 100), "PNG")
        ),
    ]

    image = _decode(composite_panel_image(parts) or "")

    assert image.size == (300, 100)
    assert image.getpixel((50, 50)) == (255, 0, 0)
    assert image.getpixel((250, 50)) == (0, 0, 255)
    assert image.getpixel((150, 50)) == (255, 255, 255)  # the gap between panels


def test_axes_scale_independently_for_normalized_page_boxes():
    parts = [
        PaperFigurePart(page_number=1, bbox=(0, 0, 100, 50), image_b64=_crop("red", (200, 50))),
        PaperFigurePart(page_number=1, bbox=(0, 50, 100, 100), image_b64=_crop("blue", (200, 50))),
    ]
    image = _decode(composite_panel_image(parts) or "")
    assert image.size == (200, 100)


def test_no_composite_without_two_placed_panels_on_one_page():
    img = _crop("red", (10, 10))
    one = [PaperFigurePart(page_number=1, bbox=(0, 0, 10, 10), image_b64=img)]
    two_pages = [
        PaperFigurePart(page_number=1, bbox=(0, 0, 10, 10), image_b64=img),
        PaperFigurePart(page_number=2, bbox=(0, 0, 10, 10), image_b64=img),
    ]
    no_images = [
        PaperFigurePart(page_number=1, bbox=(0, 0, 10, 10), image_b64=None),
        PaperFigurePart(page_number=1, bbox=(20, 0, 30, 10), image_b64=None),
    ]
    no_boxes = [
        PaperFigurePart(page_number=1, bbox=None, image_b64=img),
        PaperFigurePart(page_number=1, bbox=None, image_b64=img),
    ]
    for parts in (one, two_pages, no_images, no_boxes):
        assert composite_panel_image(parts) is None


def test_undecodable_or_degenerate_panels_keep_the_existing_image():
    img = _crop("red", (10, 10))
    garbage = [
        PaperFigurePart(page_number=1, bbox=(0, 0, 10, 10), image_b64=img),
        PaperFigurePart(page_number=1, bbox=(20, 0, 30, 10), image_b64="not-an-image"),
    ]
    zero_width = [
        PaperFigurePart(page_number=1, bbox=(0, 0, 0, 10), image_b64=img),
        PaperFigurePart(page_number=1, bbox=(20, 0, 30, 10), image_b64=img),
    ]
    assert composite_panel_image(garbage) is None
    assert composite_panel_image(zero_width) is None
