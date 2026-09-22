"""Whole-figure images for figures assembled from several panel crops.

Layout analysis crops each image region separately, so a figure whose panels
were detected as separate regions arrives as several crops. Once grouping has
decided they are one figure, :func:`composite_panel_image` pastes the crops
onto one canvas at their relative positions on the page, so the exported
``figure.image`` shows the whole figure rather than its first panel. Only the
pixels of the detected panels are available here (the page image is not), so
gaps between panels come out white.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.paper_contents import PaperFigurePart

logger = logging.getLogger(__name__)

# Refuse to build a canvas larger than this on either side: a malformed bbox
# (or a crop far off its box's scale) must not allocate gigapixel images.
_MAX_CANVAS_SIDE = 8000


def composite_panel_image(parts: Sequence[PaperFigurePart]) -> str | None:
    """Base64 image of all panel crops placed by their boxes, or ``None``.

    Needs at least two parts that each carry an image and a bounding box, all
    on the same page; box coordinates only need to share one top-left-origin
    space. Returns ``None`` (keep the existing image) whenever that does not
    hold or an image cannot be decoded.
    """
    placed = [p for p in parts if p.image_b64 and p.bbox and len(p.bbox) == 4]
    if len(placed) < 2 or len({p.page_number for p in placed}) != 1:
        return None
    try:
        from PIL import Image

        images = [Image.open(io.BytesIO(base64.b64decode(p.image_b64 or ""))) for p in placed]
    except (binascii.Error, OSError, ValueError):
        logger.debug("Panel composite skipped: a panel image could not be decoded")
        return None

    boxes = [tuple(float(v) for v in p.bbox or ()) for p in placed]
    if any(x2 <= x1 or y2 <= y1 for x1, y1, x2, y2 in boxes):
        return None
    # Pixels per box unit, per axis: layout boxes are often in a normalized
    # page space whose x and y units differ in length.
    scale_x = sum(img.width / (b[2] - b[0]) for img, b in zip(images, boxes, strict=True)) / len(
        images
    )
    scale_y = sum(img.height / (b[3] - b[1]) for img, b in zip(images, boxes, strict=True)) / len(
        images
    )
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    width = round((max(b[2] for b in boxes) - left) * scale_x)
    height = round((max(b[3] for b in boxes) - top) * scale_y)
    if not (0 < width <= _MAX_CANVAS_SIDE and 0 < height <= _MAX_CANVAS_SIDE):
        return None

    canvas = Image.new("RGB", (width, height), "white")
    for img, (x1, y1, _x2, _y2) in zip(images, boxes, strict=True):
        canvas.paste(
            img.convert("RGB"), (round((x1 - left) * scale_x), round((y1 - top) * scale_y))
        )
    fmt = "PNG" if images[0].format == "PNG" else "JPEG"
    out = io.BytesIO()
    canvas.save(out, format=fmt, **({"quality": 90} if fmt == "JPEG" else {}))
    return base64.b64encode(out.getvalue()).decode("ascii")
