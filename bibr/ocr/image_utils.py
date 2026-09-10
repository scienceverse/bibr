"""PIL / PDF rendering helpers for OCR pipeline stages.

Relocated from ``bibr.pipeline_helpers`` during P4 serve migration.
"""

from __future__ import annotations

import io
import logging
import math
from typing import TYPE_CHECKING

from bibr.ocr.utils import pdfium_lock

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


def pil_to_bytes(img, fmt: str = "JPEG") -> bytes:
    """Convert a PIL Image to in-memory bytes."""
    with io.BytesIO() as buf:
        img.save(buf, format=fmt)
        return buf.getvalue()


def encode_region_for_ocr(img, geometry, fmt: str = "JPEG") -> str:
    """Convert a region crop to base64, resized to one model family's patch grid.

    Applies ``smart_resize`` so dimensions are multiples of the encoder's patch
    factor and the pixel count stays inside the budget that encoder was trained
    for. Feeding a vision transformer a misaligned or over-large grid degrades
    output into hallucinated text rather than an explicit error.

    ``geometry`` must be the geometry of the model actually being called — see
    :class:`bibr.ocr.profiles.OcrImageGeometry`. Passing another family's
    constants silently changes resolution rather than failing.
    """
    from bibr.ocr.image_processing import load_image_to_base64

    return load_image_to_base64(
        img,
        t_patch_size=geometry.t_patch_size,
        max_pixels=geometry.max_pixels,
        image_format=fmt,
        patch_expand_factor=geometry.patch_expand_factor,
        min_pixels=geometry.min_pixels,
    )


def pil_to_base64_glmocr(img, fmt: str = "JPEG") -> str:
    """Encode a region crop using GLM-OCR geometry.

    For callers with no OCR profile to consult: the cloud vision-LLM client and
    the MLX benchmarks. Anything holding an :class:`~bibr.ocr.profiles.OcrProfile`
    must use :func:`encode_region_for_ocr` with ``profile.image`` instead — this
    helper is GLM-specific, and it was applied to every backend until 2026-08-03.
    """
    from bibr.ocr.profiles import GLM_IMAGE_GEOMETRY

    return encode_region_for_ocr(img, GLM_IMAGE_GEOMETRY, fmt=fmt)


def iter_pdf_pages_with_index(
    pdf_bytes: bytes,
    dpi: int = 200,
    start_page: int | None = None,
    end_page: int | None = None,
    max_pixels: int = 25_000_000,
    max_dimension: int = 10_000,
):
    """Yield ``(page_index, PIL.Image)`` one page at a time.

    Holds :data:`bibr.ocr.utils.pdfium_lock` for the full document lifetime.
    PDFium maintains global mutable C state, so concurrent operations on
    different documents (or interleaved page renders on one) corrupt that
    state and cause "Data format error" or segfault. Lazy consumers hold
    the lock for the duration of the iteration, matching :func:`iter_pdf_pages`.
    """
    import pypdfium2

    if dpi <= 0:
        raise ValueError(f"DPI must be positive, got {dpi}")

    with pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            total = len(doc)
            start = start_page if start_page is not None else 0
            end = end_page if end_page is not None else total - 1
            end = min(end, total - 1)
            # A negative start silently indexed from the end of the document
            # (``doc[-3]``), so a library caller passing ``start_page=-1``
            # rendered arbitrary trailing pages and surfaced later as an opaque
            # ``layout_failed``. Serve validates this at the HTTP boundary; the
            # library API had no such gate.
            # Same guard the sibling renderer in ``bibr.ocr.utils`` already has.
            if start < 0 or start > end:
                raise ValueError(f"Invalid page range: start={start}, end={end}, total={total}")
            scale = dpi / 72

            for page_idx in range(start, end + 1):
                page = doc[page_idx]
                try:
                    width_points, height_points = page.get_size()
                    width = math.ceil(width_points * scale)
                    height = math.ceil(height_points * scale)
                    if width > max_dimension or height > max_dimension:
                        raise ValueError(
                            f"PDF page {page_idx + 1} renders to {width}x{height} pixels, "
                            f"exceeding the {max_dimension}-pixel dimension limit"
                        )
                    if width * height > max_pixels:
                        raise ValueError(
                            f"PDF page {page_idx + 1} renders to {width * height:,} pixels, "
                            f"exceeding the {max_pixels:,}-pixel limit"
                        )

                    bitmap = page.render(scale=scale)
                    try:
                        pil_img = bitmap.to_pil()
                    finally:
                        try:
                            bitmap.close()
                        except Exception:  # noqa: S110
                            pass
                finally:
                    page.close()
                yield page_idx, pil_img
        finally:
            doc.close()


def render_pdf_pages_with_index(
    pdf_bytes: bytes,
    dpi: int = 200,
    start_page: int | None = None,
    end_page: int | None = None,
    max_pixels: int = 25_000_000,
    max_dimension: int = 10_000,
) -> list[tuple[int, Image.Image]]:
    """Eager list form for callers that want the full result up-front."""
    return list(
        iter_pdf_pages_with_index(
            pdf_bytes,
            dpi,
            start_page,
            end_page,
            max_pixels,
            max_dimension,
        )
    )
