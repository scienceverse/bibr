"""
PDF and image utilities for OCR processing.
"""

from __future__ import annotations

import gc
import io
import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# pypdfium2 wraps the PDFium C library which maintains global mutable state.
# Concurrent calls from separate threads corrupt that state and cause subsequent
# operations to raise "Data format error" or segfault.  Every caller that
# touches pypdfium2 MUST acquire this lock before opening a PdfDocument and
# hold it for the duration of all PDFium operations on that document.
pdfium_lock = threading.Lock()


def release_gpu_cache(device: str | None) -> None:
    """Release cached GPU memory back to the driver.

    PyTorch's CUDA caching allocator holds freed memory blocks for reuse.
    When multiple processes share a GPU this reserved-but-unused memory
    starves other processes.  Calling ``empty_cache`` returns it to the
    CUDA/MPS driver so other processes (or later allocations) can use it.

    Parameters
    ----------
    device : str or None
        Device string (``"cuda"``, ``"mps"``, ``"cpu"``, or ``None``).
        Only CUDA and MPS trigger a cache flush; CPU and None are no-ops.
    """
    gc.collect()
    if device == "cuda":
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            logger.debug("Could not clear CUDA cache", exc_info=True)
    elif device == "mps":
        try:
            import torch

            torch.mps.empty_cache()
        except Exception:
            logger.debug("Could not clear MPS cache", exc_info=True)


def render_pdf_pages(
    pdf_bytes: bytes,
    dpi: int = 200,
    start_page: int | None = None,
    end_page: int | None = None,
) -> list[tuple[int, Image.Image]]:
    """Render PDF pages to PIL Images.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file bytes.
    dpi : int
        Render resolution (200 DPI recommended for LightOnOCR).
    start_page : int, optional
        Start page (0-indexed, inclusive).
    end_page : int, optional
        End page (0-indexed, inclusive).

    Returns
    -------
    List[Tuple[int, Image.Image]]
        List of (page_number, PIL Image) tuples.
    """
    return list(iter_pdf_pages(pdf_bytes, dpi=dpi, start_page=start_page, end_page=end_page))


def iter_pdf_pages(
    pdf_bytes: bytes,
    dpi: int = 200,
    start_page: int | None = None,
    end_page: int | None = None,
):
    """Lazily render PDF pages to PIL Images one at a time.

    Yields pages individually to avoid holding all rendered images in memory
    simultaneously. Preferred over ``render_pdf_pages`` when processing large
    documents on memory-constrained devices.

    The :data:`pdfium_lock` is held for the entire document lifetime (from
    ``PdfDocument(bytes)`` through ``doc.close()``).  Callers that consume
    this generator lazily — e.g. ``for page_num, img in iter_pdf_pages(...)``
    — hold the lock for the full iteration.  That is intentional: PDFium's
    global state must not be touched by another thread while the document is
    open.  If interleaving with other PDFium work is required, fully consume
    the generator (e.g. via :func:`render_pdf_pages`) before releasing the
    GIL to another thread.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file bytes.
    dpi : int
        Render resolution (200 DPI recommended for LightOnOCR).
    start_page : int, optional
        Start page (0-indexed, inclusive).
    end_page : int, optional
        End page (0-indexed, inclusive).

    Yields
    ------
    Tuple[int, Image.Image]
        (page_number, PIL Image) for each page.
    """
    import pypdfium2 as pdfium

    if dpi <= 0:
        raise ValueError(f"DPI must be positive, got {dpi}")

    with pdfium_lock:
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            total_pages = len(pdf)

            # Handle page range
            start = start_page if start_page is not None else 0
            end = end_page if end_page is not None else total_pages - 1
            end = min(end, total_pages - 1)

            if start < 0 or start > end:
                raise ValueError(
                    f"Invalid page range: start={start}, end={end}, total={total_pages}"
                )

            # Scale factor: DPI / 72 (PDF points per inch)
            scale = dpi / 72.0

            logger.info(f"Rendering {end - start + 1} pages from PDF ({start + 1} to {end + 1})")
            for page_num in range(start, end + 1):
                page = pdf[page_num]
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()
                try:
                    bitmap.close()
                except Exception:  # noqa: S110
                    pass
                page.close()
                logger.debug(f"Rendered page {page_num + 1}/{total_pages} at {dpi} DPI")
                yield (page_num, pil_image)
        finally:
            pdf.close()


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string.

    Parameters
    ----------
    image : Image.Image
        PIL Image to convert.
    format : str
        Image format (PNG recommended).

    Returns
    -------
    str
        Base64-encoded image string.
    """
    import base64

    buffer = io.BytesIO()
    try:
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    finally:
        buffer.close()


def crop_region(pil_image, bbox: tuple[float, float, float, float]):
    """Crop a region from a PIL image using a bounding box.

    Parameters
    ----------
    pil_image : Image.Image
        Source image.
    bbox : tuple[float, float, float, float]
        Bounding box as (x0, y0, x1, y1) in pixel coordinates.

    Returns
    -------
    Image.Image
        Cropped image region.
    """
    return pil_image.crop(bbox)


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Get total page count from PDF bytes.

    The :data:`pdfium_lock` is held for the entire document lifetime (from
    ``PdfDocument(bytes)`` through ``doc.close()``).  This prevents concurrent
    PDFium operations from corrupting global state.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file bytes.

    Returns
    -------
    int
        Total number of pages.
    """
    import pypdfium2 as pdfium

    with pdfium_lock:
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            return len(pdf)
        finally:
            pdf.close()
