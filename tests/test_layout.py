"""Tests for bibr.ocr.layout module — label classification constants."""

import pytest

from bibr.ocr.layout import (
    SECTION_HINT_LABELS,
    STRUCTURAL_LABELS,
)

# ============================================================================
# Label Classification Tests
# ============================================================================


class TestLabelClassification:
    """Tests for label classification helpers."""

    def test_structural_labels(self):
        """Structural labels should include header, footer, number."""
        assert "header" in STRUCTURAL_LABELS
        assert "footer" in STRUCTURAL_LABELS
        assert "number" in STRUCTURAL_LABELS

    def test_structural_excludes_content(self):
        """Content labels should not be structural."""
        assert "text" not in STRUCTURAL_LABELS
        assert "abstract" not in STRUCTURAL_LABELS
        assert "table" not in STRUCTURAL_LABELS

    def test_section_hint_labels(self):
        """Section hint labels should include abstract, reference, footnote."""
        assert "abstract" in SECTION_HINT_LABELS
        assert "reference" in SECTION_HINT_LABELS
        assert "reference_content" in SECTION_HINT_LABELS
        assert "footnote" in SECTION_HINT_LABELS
        assert "vision_footnote" in SECTION_HINT_LABELS

    def test_section_hint_excludes_body(self):
        """Body content labels should not be section hints."""
        assert "text" not in SECTION_HINT_LABELS
        assert "header" not in SECTION_HINT_LABELS
        assert "table" not in SECTION_HINT_LABELS


# ============================================================================
# crop_region Utility Tests
# ============================================================================


class TestCropRegion:
    """Tests for the crop_region utility."""

    @pytest.mark.skipif(
        not pytest.importorskip("PIL", reason="PIL not installed"),
        reason="PIL not installed",
    )
    def test_crop_region_basic(self):
        from PIL import Image

        from bibr.ocr.utils import crop_region

        image = Image.new("RGB", (600, 800), color="white")
        cropped = crop_region(image, (100.0, 200.0, 400.0, 500.0))
        assert cropped.width == 300
        assert cropped.height == 300
