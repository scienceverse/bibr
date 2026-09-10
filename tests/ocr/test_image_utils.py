"""Coverage for relocated PIL/PDF helpers."""

from pathlib import Path

import pytest
from PIL import Image

from bibr.ocr.image_utils import pil_to_base64_glmocr, pil_to_bytes, render_pdf_pages_with_index


def test_pil_to_bytes_roundtrips():
    img = Image.new("RGB", (10, 10), color="red")
    data = pil_to_bytes(img)
    assert isinstance(data, bytes) and len(data) > 0


def test_pil_to_base64_glmocr_returns_str():
    pytest.importorskip("cv2")
    img = Image.new("RGB", (224, 224), color="white")
    out = pil_to_base64_glmocr(img)
    assert isinstance(out, str) and len(out) > 0


@pytest.mark.integration
def test_render_pdf_pages_with_index_respects_range():
    fixture = Path(__file__).parent.parent / "fixtures" / "native_text_sample.pdf"
    if not fixture.exists():
        pytest.skip("no native_text_sample.pdf fixture available")
    pages = render_pdf_pages_with_index(fixture.read_bytes(), dpi=72, start_page=0, end_page=0)
    assert len(pages) == 1
    idx, img = pages[0]
    assert idx == 0
    assert hasattr(img, "size")


class TestNegativeStartPage:
    """L14: a negative start indexed from the end and failed opaquely later."""

    def test_a_negative_start_page_is_rejected(self):
        from bibr.ocr.image_utils import iter_pdf_pages_with_index

        pdf = _one_page_pdf()
        with pytest.raises(ValueError, match="Invalid page range"):
            list(iter_pdf_pages_with_index(pdf, start_page=-1))

    def test_a_valid_start_page_still_renders(self):
        from bibr.ocr.image_utils import iter_pdf_pages_with_index

        assert len(list(iter_pdf_pages_with_index(_one_page_pdf(), start_page=0))) == 1


def _one_page_pdf() -> bytes:
    pypdfium2 = pytest.importorskip("pypdfium2")
    doc = pypdfium2.PdfDocument.new()
    doc.new_page(200, 200)
    buf = __import__("io").BytesIO()
    doc.save(buf)
    return buf.getvalue()
