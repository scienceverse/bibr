"""Image processing utilities for OCR pipeline.

Extracted from bibr._vendor.glmocr.utils.image_utils — only the functions
actually used by the LitServe pipeline.
"""

import base64
import io
import math

import numpy as np
from PIL import Image, ImageDraw


def smart_resize(
    t: int,
    h: int,
    w: int,
    t_factor: int = 1,
    h_factor: int = 28,
    w_factor: int = 28,
    min_pixels: int = 112 * 112,
    max_pixels: int = 14 * 14 * 4 * 15000,
) -> tuple[int, int]:
    """Smart resize ensuring dimensions are divisible by patch factors.

    Keeps aspect ratio while constraining total pixel count to
    [min_pixels, max_pixels].
    """
    assert t >= t_factor, "Temporal dimension must be greater than the factor."

    h_bar = max(round(h / h_factor), 1) * h_factor
    w_bar = max(round(w / w_factor), 1) * w_factor
    t_bar = max(round(t / t_factor), 1) * t_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((t * h * w) / max_pixels)
        h_bar = max(math.floor(h / beta / h_factor), 1) * h_factor
        w_bar = max(math.floor(w / beta / w_factor), 1) * w_factor
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (t * h * w))
        h_bar = max(math.ceil(h * beta / h_factor), 1) * h_factor
        w_bar = max(math.ceil(w * beta / w_factor), 1) * w_factor

    return h_bar, w_bar


def load_image_to_base64(
    image_source,
    t_patch_size: int,
    max_pixels: int,
    image_format: str,
    patch_expand_factor: int = 1,
    min_pixels: int = 112 * 112,
) -> str:
    """Load an image from various sources and convert to base64.

    Applies ``smart_resize`` to ensure dimensions are multiples of the
    vision-transformer patch grid so GLM-OCR produces accurate output.

    Supported inputs: PIL.Image, bytes, file path, data URL, base64 blob.
    """
    import os

    def _try_decode_base64_to_image_bytes(s: str) -> bytes | None:
        candidate = "".join(str(s).split())
        if len(candidate) < 32:
            return None
        if candidate.startswith("<|base64|>"):
            candidate = candidate[len("<|base64|>") :]
        if "." in candidate and len(candidate.rsplit(".", 1)[-1]) <= 5:
            return None
        pad = (-len(candidate)) % 4
        if pad:
            candidate = candidate + ("=" * pad)
        try:
            return base64.b64decode(candidate, validate=True)
        except Exception:
            return None

    if isinstance(image_source, Image.Image):
        image = image_source
    elif isinstance(image_source, bytes):
        image = Image.open(io.BytesIO(image_source))
    elif isinstance(image_source, str):
        if image_source.startswith("file://"):
            image_source = image_source[7:]
        if os.path.isfile(image_source):
            with open(image_source, "rb") as f:
                image_data = f.read()
            image = Image.open(io.BytesIO(image_data))
        elif image_source.startswith("data:image/"):
            image_data = base64.b64decode(image_source.split(",")[1])
            image = Image.open(io.BytesIO(image_data))
        else:
            decoded = _try_decode_base64_to_image_bytes(image_source)
            if decoded is None:
                raise ValueError(f"Invalid image source: {image_source}")
            image = Image.open(io.BytesIO(decoded))
    else:
        raise TypeError(f"Unsupported image source type: {type(image_source)}")

    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    h_bar, w_bar = smart_resize(
        t=t_patch_size,
        h=h,
        w=w,
        t_factor=t_patch_size,
        h_factor=14 * 2 * patch_expand_factor,
        w_factor=14 * 2 * patch_expand_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # Skip the BICUBIC resample when the input already aligns to the patch
    # grid — for region crops that come straight from the layout pass this
    # is the common case, and ``Image.resize`` even on a no-op still copies
    # the buffer.
    if (w_bar, h_bar) != (w, h):
        image = image.resize((w_bar, h_bar), Image.Resampling.BICUBIC)

    buffered = io.BytesIO()
    try:
        image.save(buffered, format=image_format)
        buffered.seek(0)
        image_data = buffered.getvalue()
    finally:
        buffered.close()

    return base64.b64encode(image_data).decode("utf-8")


def crop_image_region(
    image: Image.Image,
    bbox_2d: list,
    polygon: list | None = None,
    fill_color: int = 255,
) -> Image.Image:
    """Crop an image region using bbox and optionally mask outside polygon.

    Args:
        image: PIL Image.
        bbox_2d: [x1_norm, y1_norm, x2_norm, y2_norm] normalised to 0-1000.
        polygon: List of [x, y] coordinates normalised to 0-1000 (optional).
        fill_color: Fill colour outside polygon (default 255 = white).
    """
    image_width, image_height = image.size

    x1_norm, y1_norm, x2_norm, y2_norm = bbox_2d
    x1 = max(0, min(int(x1_norm * image_width / 1000), image_width))
    y1 = max(0, min(int(y1_norm * image_height / 1000), image_height))
    x2 = max(0, min(int(x2_norm * image_width / 1000), image_width))
    y2 = max(0, min(int(y2_norm * image_height / 1000), image_height))

    if x1 >= x2 or y1 >= y2:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "Invalid bbox (x1=%d >= x2=%d or y1=%d >= y2=%d), returning full image",
            x1,
            x2,
            y1,
            y2,
        )
        return image

    if not polygon or len(polygon) < 3:
        return image.crop((x1, y1, x2, y2))

    img_array = np.asarray(image)
    img_crop = img_array[y1:y2, x1:x2]

    scale_x = image_width / 1000
    scale_y = image_height / 1000
    crop_width = x2 - x1
    crop_height = y2 - y1
    polygon_pixels = np.empty((len(polygon), 2), dtype=np.int32)
    for i, point in enumerate(polygon):
        polygon_pixels[i, 0] = max(0, min(int(point[0] * scale_x) - x1, crop_width - 1))
        polygon_pixels[i, 1] = max(0, min(int(point[1] * scale_y) - y1, crop_height - 1))

    # Rasterise the polygon with Pillow (boundary pixels included, as
    # cv2.fillPoly did) and composite: inside → source pixels, outside → fill.
    mask_image = Image.new("L", (crop_width, crop_height), 0)
    ImageDraw.Draw(mask_image).polygon(
        [(int(x), int(y)) for x, y in polygon_pixels], fill=1, outline=1
    )
    mask = np.asarray(mask_image, dtype=bool)
    if mask.shape != img_crop.shape[:2]:  # defensive: PIL sizes are (w, h)
        mask = mask[: img_crop.shape[0], : img_crop.shape[1]]

    if img_crop.ndim == 3:
        output = np.where(mask[..., None], img_crop, np.uint8(fill_color))
    else:
        output = np.where(mask, img_crop, np.uint8(fill_color))

    return Image.fromarray(output.astype(np.uint8))
