"""
Tests for the OCR module.

Tests cover configuration, utilities, and OcrClient (glmocr SDK wrapper).
"""

import importlib.util
from pathlib import Path

import pytest

# Get sample PDF for testing
SAMPLE_PDF_PATH = Path(__file__).parent.parent / "bibr" / "data" / "sample_paper.pdf"


# ============================================================================
# Utils Tests
# ============================================================================

HAS_PYPDFIUM2 = importlib.util.find_spec("pypdfium2") is not None
HAS_PIL = importlib.util.find_spec("PIL") is not None


class TestUtils:
    """Tests for OCR utility functions."""

    @pytest.fixture
    def sample_pdf_bytes(self) -> bytes:
        """Load sample PDF for testing."""
        if not SAMPLE_PDF_PATH.exists():
            pytest.skip("Sample PDF not found")
        return SAMPLE_PDF_PATH.read_bytes()

    @pytest.mark.skipif(not HAS_PYPDFIUM2, reason="pypdfium2 not installed")
    def test_get_pdf_page_count(self, sample_pdf_bytes):
        """Test PDF page count extraction."""
        from bibr.ocr.utils import get_pdf_page_count

        count = get_pdf_page_count(sample_pdf_bytes)
        assert count > 0
        assert isinstance(count, int)

    @pytest.mark.skipif(not HAS_PYPDFIUM2, reason="pypdfium2 not installed")
    def test_render_pdf_pages_all(self, sample_pdf_bytes):
        """Test rendering all PDF pages."""
        from bibr.ocr.utils import render_pdf_pages

        pages = render_pdf_pages(sample_pdf_bytes, dpi=100)  # Lower DPI for speed
        assert len(pages) > 0
        for page_num, image in pages:
            assert isinstance(page_num, int)
            assert page_num >= 0
            # Check it's a PIL Image
            assert hasattr(image, "size")
            assert hasattr(image, "mode")

    @pytest.mark.skipif(not HAS_PYPDFIUM2, reason="pypdfium2 not installed")
    def test_render_pdf_pages_range(self, sample_pdf_bytes):
        """Test rendering a specific page range."""
        from bibr.ocr.utils import get_pdf_page_count, render_pdf_pages

        total = get_pdf_page_count(sample_pdf_bytes)
        if total < 2:
            pytest.skip("PDF has less than 2 pages")

        # Get only first page
        pages = render_pdf_pages(sample_pdf_bytes, dpi=100, start_page=0, end_page=0)
        assert len(pages) == 1
        assert pages[0][0] == 0

    @pytest.mark.skipif(not HAS_PYPDFIUM2, reason="pypdfium2 not installed")
    def test_render_pdf_pages_invalid_range(self, sample_pdf_bytes):
        """Test invalid page range raises error."""
        from bibr.ocr.utils import render_pdf_pages

        with pytest.raises(ValueError, match="Invalid page range"):
            render_pdf_pages(sample_pdf_bytes, start_page=5, end_page=2)

    @pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
    def test_image_to_base64(self):
        """Test image to base64 conversion."""
        import base64

        from PIL import Image

        from bibr.ocr.utils import image_to_base64

        image = Image.new("RGB", (100, 100), color="red")
        b64 = image_to_base64(image, format="PNG")
        assert isinstance(b64, str)
        assert len(b64) > 0

        # Verify it's valid base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    @pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
    def test_crop_region_returns_subimage(self):
        from PIL import Image

        from bibr.ocr.utils import crop_region

        img = Image.new("RGB", (100, 100), color="blue")
        cropped = crop_region(img, (10, 20, 60, 80))
        assert cropped.size == (50, 60)

    def test_iter_pdf_pages_rejects_zero_dpi(self, sample_pdf_bytes):
        from bibr.ocr.utils import iter_pdf_pages

        with pytest.raises(ValueError, match="DPI must be positive"):
            # Generator only validates when iteration starts
            list(iter_pdf_pages(sample_pdf_bytes, dpi=0))

    def test_iter_pdf_pages_rejects_negative_dpi(self, sample_pdf_bytes):
        from bibr.ocr.utils import iter_pdf_pages

        with pytest.raises(ValueError, match="DPI must be positive"):
            list(iter_pdf_pages(sample_pdf_bytes, dpi=-100))

    def test_iter_pdf_pages_clamps_end_to_total(self, sample_pdf_bytes):
        """end_page beyond total should be clamped, not raise."""
        from bibr.ocr.utils import get_pdf_page_count, iter_pdf_pages

        total = get_pdf_page_count(sample_pdf_bytes)
        pages = list(iter_pdf_pages(sample_pdf_bytes, dpi=72, end_page=total + 50))
        assert len(pages) == total

    def test_release_gpu_cache_none_is_noop(self):
        from bibr.ocr.utils import release_gpu_cache

        release_gpu_cache(None)  # must not raise

    def test_release_gpu_cache_cpu_is_noop(self):
        from bibr.ocr.utils import release_gpu_cache

        release_gpu_cache("cpu")  # must not raise

    def test_release_gpu_cache_unknown_device_is_noop(self):
        from bibr.ocr.utils import release_gpu_cache

        release_gpu_cache("xpu")  # unknown device — silently ignored
