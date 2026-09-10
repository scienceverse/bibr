"""Pure image preprocessing coverage; no model weights or GPU required."""

import base64
import io

import pytest
from PIL import Image

pytest.importorskip("cv2")

from bibr.ocr.image_processing import crop_image_region, load_image_to_base64, smart_resize


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _decode_image(encoded: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


@pytest.mark.parametrize(
    ("height", "width", "min_pixels", "max_pixels", "expected"),
    [
        (28, 56, 1, 1_000_000, (28, 56)),
        (280, 280, 1, 28 * 28, (28, 28)),
        (10, 10, 28 * 28, 1_000_000, (28, 28)),
    ],
)
def test_smart_resize_bounds(height, width, min_pixels, max_pixels, expected):
    assert smart_resize(1, height, width, min_pixels=min_pixels, max_pixels=max_pixels) == expected


def test_smart_resize_rejects_too_small_temporal_dimension():
    with pytest.raises(AssertionError):
        smart_resize(0, 28, 28)


def test_load_image_accepts_supported_sources(tmp_path):
    image = Image.new("RGBA", (28, 28), color=(10, 20, 30, 255))
    png = _png_bytes(image)
    path = tmp_path / "input.png"
    path.write_bytes(png)
    raw_b64 = base64.b64encode(png).decode()
    sources = [
        image,
        png,
        str(path),
        f"file://{path}",
        f"data:image/png;base64,{raw_b64}",
        raw_b64,
        f"<|base64|>{raw_b64}",
    ]

    for source in sources:
        encoded = load_image_to_base64(
            source,
            t_patch_size=1,
            max_pixels=1_000_000,
            min_pixels=1,
            image_format="PNG",
        )
        decoded = _decode_image(encoded)
        assert decoded.size == (28, 28)
        assert decoded.mode == "RGB"


@pytest.mark.parametrize("source", ["short", "definitely.not_an_image", object()])
def test_load_image_rejects_invalid_sources(source):
    error = TypeError if not isinstance(source, str) else ValueError
    with pytest.raises(error):
        load_image_to_base64(source, 1, 1_000_000, "PNG", min_pixels=1)


def test_crop_image_region_bbox_and_invalid_bbox(caplog):
    image = Image.new("RGB", (100, 100), color="red")
    cropped = crop_image_region(image, [100, 200, 900, 800])
    assert cropped.size == (80, 60)

    unchanged = crop_image_region(image, [900, 0, 100, 1000])
    assert unchanged is image
    assert "Invalid bbox" in caplog.text


def test_crop_image_region_applies_polygon_mask():
    image = Image.new("RGB", (100, 100), color=(200, 0, 0))
    cropped = crop_image_region(
        image,
        [100, 100, 900, 900],
        polygon=[[100, 100], [900, 100], [100, 900]],
    )
    assert cropped.size == (80, 80)
    assert cropped.getpixel((5, 5)) == (200, 0, 0)
    assert cropped.getpixel((79, 79)) == (255, 255, 255)


def test_crop_image_region_handles_grayscale_polygon():
    image = Image.new("L", (100, 100), color=20)
    cropped = crop_image_region(
        image,
        [0, 0, 1000, 1000],
        polygon=[[0, 0], [1000, 0], [0, 1000]],
        fill_color=240,
    )
    assert cropped.mode == "L"
    assert cropped.getpixel((99, 99)) == 240
