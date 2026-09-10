"""Detached PDF character evidence and layout-independent source accounting."""

from __future__ import annotations

import ctypes
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class NativeChar:
    start: int
    end: int  # exclusive PDFium code-unit interval, including surrogate pairs
    text: str
    bbox: tuple[float, float, float, float] | None
    font_id: int | None = None
    angle: float | None = None


def extract_characters(textpage, *, include_fonts=True):
    """Read each PDFium code unit once, retaining failed geometry and raw intervals."""
    import pypdfium2 as pdfium

    chars, fonts, font_ids = [], [], {}
    i = 0
    count = textpage.count_chars()
    while i < count:
        start = i
        code = pdfium.raw.FPDFText_GetUnicode(textpage.raw, i)
        i += 1
        if 0xD800 <= code <= 0xDBFF and i < count:
            low = pdfium.raw.FPDFText_GetUnicode(textpage.raw, i)
            if 0xDC00 <= low <= 0xDFFF:
                code = 0x10000 + ((code - 0xD800) << 10) + low - 0xDC00
                i += 1
        text = "\ufffd" if 0xD800 <= code <= 0xDFFF or code > 0x10FFFF else chr(code)
        boxes = []
        for index in range(start, i):
            try:
                box = tuple(float(v) for v in textpage.get_charbox(index))
                if all(math.isfinite(v) for v in box):
                    boxes.append(box)
            except Exception:  # noqa: BLE001, S110 - retained as unlocated source evidence
                pass
        bbox = (
            (
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            )
            if boxes
            else None
        )
        font_id, angle = None, None
        if include_fonts and not text.isspace():
            try:
                flags = ctypes.c_int()
                buf = ctypes.create_string_buffer(256)
                n = pdfium.raw.FPDFText_GetFontInfo(
                    textpage.raw, start, buf, 256, ctypes.byref(flags)
                )
                name = buf.value.decode("utf-8", errors="replace") if 0 < n <= 256 else ""
                size = float(pdfium.raw.FPDFText_GetFontSize(textpage.raw, start))
                weight = int(pdfium.raw.FPDFText_GetFontWeight(textpage.raw, start))
                key = (name, flags.value, size if math.isfinite(size) else None, weight)
                if key not in font_ids:
                    font_ids[key] = len(fonts)
                    fonts.append(dict(zip(("name", "flags", "size", "weight"), key, strict=True)))
                font_id = font_ids[key]
                value = float(pdfium.raw.FPDFText_GetCharAngle(textpage.raw, start))
                angle = value if math.isfinite(value) else None
            except Exception:  # noqa: BLE001, S110 - missing font metadata is not missing text
                pass
        chars.append(NativeChar(start, i, text, bbox, font_id, angle))
    return chars, fonts


def center_records(chars):
    """Compatibility view for the existing region acceptance policy."""
    return [
        (
            c.text,
            (c.bbox[0] + c.bbox[2]) / 2 if c.bbox else 0.0,
            (c.bbox[1] + c.bbox[3]) / 2 if c.bbox else 0.0,
            c.text in ("\r", "\n"),
        )
        for c in chars
        if c.bbox is not None or c.text in ("\r", "\n")
    ]


def normalized_box(bbox, crop, rotation=0):
    x0, y0, x1, y1 = crop
    points = []
    for x, y in ((bbox[0], bbox[1]), (bbox[2], bbox[3])):
        u, v = (x - x0) / (x1 - x0), (y1 - y) / (y1 - y0)
        u, v = {0: (u, v), 90: (1 - v, u), 180: (1 - u, 1 - v), 270: (v, 1 - u)}[rotation % 360]
        points.append((1000 * u, 1000 * v))
    return [
        min(p[0] for p in points),
        min(p[1] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points),
    ]


def overlap(inner, outer):
    area = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])
    intersection = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0])) * max(
        0, min(inner[3], outer[3]) - max(inner[1], outer[1])
    )
    return intersection / area if area else 0.0


def account_characters(chars, fonts, regions, crop, page_index, rotation=0):
    """Inventory source independently of layout; ambiguous claims remain explicit.

    These are geometric owners, not a claim of successful output transcription.
    An inline formula's parent prose is a container, not a second leaf owner.
    """
    for index, region in enumerate(regions):
        region["_source_region_id"] = f"p{page_index}:r{index}"
    evidence = []
    seen_glyphs = {}
    duplicate_count = 0
    counts = dict.fromkeys(("assigned", "unassigned", "ambiguous", "excluded", "unlocated"), 0)
    for char in chars:
        item = asdict(char)
        item["bbox"] = list(char.bbox) if char.bbox is not None else None
        item["source_id"] = f"p{page_index}:c{char.start}-{char.end}"
        item["owners"] = []
        if char.bbox is not None:
            item["bbox_2d"] = normalized_box(char.bbox, crop, rotation)
        if char.text.isspace() or char.text == "\x00":
            item["status"] = "separator"
        elif char.bbox is None:
            item["status"] = "unlocated"
        else:
            box = normalized_box(char.bbox, crop, rotation)
            item["bbox_2d"] = box
            candidates = [
                r for r in regions if r.get("bbox_2d") and overlap(box, r["bbox_2d"]) > 0.5
            ]
            leaves = [
                r
                for r in candidates
                if r.get("label") in {"inline_formula", "display_formula", "formula"}
            ]
            if len(leaves) == 1:
                candidates = [r for r in candidates if r.get("label") not in {"text", "content"}]
            item["owners"] = [r["_source_region_id"] for r in candidates]
            if not candidates:
                item["status"] = "unassigned"
            elif len(candidates) > 1:
                item["status"] = "ambiguous"
            elif candidates[0].get("task_type") == "abandon":
                item["status"] = "excluded"
            else:
                item["status"] = "assigned"
        if item["status"] in counts:
            counts[item["status"]] += 1
        if char.bbox is not None and not char.text.isspace() and char.text != "\x00":
            key = (char.text, *(round(v, 2) for v in char.bbox))
            if key in seen_glyphs:
                item["duplicate_of"] = seen_glyphs[key]
                duplicate_count += 1
            else:
                seen_glyphs[key] = item["source_id"]
        evidence.append(item)
    return {
        "characters": evidence,
        "fonts": fonts,
        "counts": counts,
        "page_rotation": rotation,
        "duplicate_glyph_candidates": duplicate_count,
        "visual_coverage": "not_checked",
    }


def visual_coverage(image, regions, source, *, grid_size=32):
    """Independent raster-ink audit; dark pixels are evidence, not text detections.

    Retains suspicious tiles for review even on scans with no native characters.
    It cannot prove completeness or distinguish photographs from omitted text.
    """
    import numpy as np
    from PIL import Image

    with image.convert("L") as gray:
        gray.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        ink = np.asarray(gray) < 180
    height, width = ink.shape

    def mask(boxes):
        covered = np.zeros(ink.shape, dtype=bool)
        for box in boxes:
            x0, y0 = max(0, int(box[0] * width / 1000)), max(0, int(box[1] * height / 1000))
            x1, y1 = (
                min(width, math.ceil(box[2] * width / 1000)),
                min(height, math.ceil(box[3] * height / 1000)),
            )
            covered[y0:y1, x0:x1] = True
        return covered

    layout_mask = mask([r["bbox_2d"] for r in regions if r.get("bbox_2d")])
    native_mask = mask([c["bbox_2d"] for c in source["characters"] if c.get("bbox_2d")])
    outside = ink & ~layout_mask
    tiles = []
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            patch = outside[y : y + grid_size, x : x + grid_size]
            if int(patch.sum()) >= max(3, patch.size * 0.02):
                tiles.append(
                    [
                        1000 * x / width,
                        1000 * y / height,
                        1000 * min(x + grid_size, width) / width,
                        1000 * min(y + grid_size, height) / height,
                    ]
                )
    return {
        "method": "raster_ink_v1",
        "status": "checked",
        "ink_pixels": int(ink.sum()),
        "outside_layout_pixels": int(outside.sum()),
        "outside_native_pixels": int((ink & ~native_mask).sum()),
        "outside_layout_tiles": tiles,
        "resolution": [width, height],
        "interpretation": "Diagnostic only: ink includes images, rules and rasterization fringes.",
    }


def attach_visual_coverage(inspection, images, page_indices=None):
    """Attach raster evidence to detached physical-page records, including windowed runs."""
    pages = {page.index: page for page in inspection.pages}
    indices = page_indices if page_indices is not None else range(len(images))
    for index, image, regions in zip(indices, images, inspection.layout_results, strict=True):
        page = pages.get(index)
        if page is not None and page.source is not None:
            page.source["visual_coverage"] = visual_coverage(image, regions, page.source)
