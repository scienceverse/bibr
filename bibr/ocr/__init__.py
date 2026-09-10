"""
OCR Module for bibr

Configuration and utilities for OCR processing.
The legacy ``bibr.ocr.layout.TASK_PROMPTS`` compatibility mapping remains for
older consumers; active pipeline runtime uses resolved ``OcrProfile`` objects.
"""

from typing import TYPE_CHECKING

from bibr.ocr.postprocess import (
    clean_formula_number,
    format_bullet_points,
    merge_formula_numbers,
    merge_text_blocks,
)

if TYPE_CHECKING:
    from bibr.ocr.image_processing import (
        crop_image_region,
        load_image_to_base64,
        smart_resize,
    )

# cv2-gated symbols are imported lazily so a core (non-'ml') install can import
# bibr.ocr (and its cloud-OCR submodules) without opencv present.
_LAZY = {"crop_image_region", "load_image_to_base64", "smart_resize"}


def __getattr__(name: str):
    if name in _LAZY:
        from bibr.ocr import image_processing

        return getattr(image_processing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "crop_image_region",
    "load_image_to_base64",
    "smart_resize",
    "clean_formula_number",
    "format_bullet_points",
    "merge_formula_numbers",
    "merge_text_blocks",
]
