"""PIL / PDF rendering helpers for OCR pipeline stages.

Relocated from ``bibr.pipeline_helpers`` during P4 serve migration.
"""

from __future__ import annotations

import io
import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from bibr.ocr.utils import pdfium_lock

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# Lowest DPI a page is rendered at to fit the render budget: one pixel per PDF
# point. A page that does not fit even there is refused as before.
MIN_REDUCED_RENDER_DPI = 72


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


def _render_size(width_points: float, height_points: float, dpi: int) -> tuple[int, int]:
    scale = dpi / 72
    return math.ceil(width_points * scale), math.ceil(height_points * scale)


def _fits(width: int, height: int, max_pixels: int, max_dimension: int) -> bool:
    return width <= max_dimension and height <= max_dimension and width * height <= max_pixels


def fitting_render_dpi(
    width_points: float,
    height_points: float,
    dpi: int,
    max_pixels: int,
    max_dimension: int,
) -> int:
    """The largest whole DPI, at most *dpi*, at which a page fits the render budget.

    ``0`` when the page does not fit even at 1 DPI.
    """
    if width_points <= 0 or height_points <= 0:
        return dpi
    ceiling = 72 * min(
        max_dimension / width_points,
        max_dimension / height_points,
        math.sqrt(max_pixels / (width_points * height_points)),
    )
    candidate = min(dpi, math.floor(ceiling))
    # The render size is rounded up, so the float ceiling can overshoot by a pixel.
    while candidate > 0 and not _fits(
        *_render_size(width_points, height_points, candidate), max_pixels, max_dimension
    ):
        candidate -= 1
    return max(candidate, 0)


def iter_pdf_pages_with_index(
    pdf_bytes: bytes,
    dpi: int = 200,
    start_page: int | None = None,
    end_page: int | None = None,
    max_pixels: int = 25_000_000,
    max_dimension: int = 10_000,
    *,
    min_dpi: int | None = None,
    on_reduced_dpi: Callable[[int, int], None] | None = None,
):
    """Yield ``(page_index, PIL.Image)`` one page at a time.

    A page too large for the render budget at *dpi* raises ``ValueError``,
    unless *min_dpi* is set: then it renders at the largest DPI that fits, if
    that is at least *min_dpi*, and ``on_reduced_dpi(page_index, dpi)`` is
    called. Only the pixel density changes; every box downstream is in page
    coordinates normalized by the image's own size, so it stays in place.

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

            for page_idx in range(start, end + 1):
                page = doc[page_idx]
                try:
                    width_points, height_points = page.get_size()
                    page_dpi = dpi
                    width, height = _render_size(width_points, height_points, dpi)
                    if min_dpi is not None and not _fits(width, height, max_pixels, max_dimension):
                        reduced = fitting_render_dpi(
                            width_points, height_points, dpi, max_pixels, max_dimension
                        )
                        if reduced >= min_dpi:
                            page_dpi = reduced
                            width, height = _render_size(width_points, height_points, page_dpi)
                            if on_reduced_dpi is not None:
                                on_reduced_dpi(page_idx, page_dpi)
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

                    bitmap = page.render(scale=page_dpi / 72)
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
