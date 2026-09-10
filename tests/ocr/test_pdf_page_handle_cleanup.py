"""iter_pdf_pages must close native PDFium handles (bitmap + page) per page."""

import pytest

from bibr.ocr.image_utils import iter_pdf_pages_with_index
from bibr.ocr.utils import iter_pdf_pages


class FakeBitmap:
    def __init__(self):
        self.closed = False

    def to_pil(self):
        return object()

    def close(self):
        self.closed = True


class FakePage:
    def __init__(self):
        self.bitmap = FakeBitmap()
        self.closed = False
        self.rendered = False

    def get_size(self):
        return 612, 792

    def render(self, scale):
        self.rendered = True
        return self.bitmap

    def close(self):
        self.closed = True


class FakeDoc:
    def __init__(self, _data):
        self.pages = [FakePage(), FakePage(), FakePage()]
        self.closed = False

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, idx):
        return self.pages[idx]

    def close(self):
        self.closed = True


@pytest.fixture
def fake_doc(monkeypatch):
    pypdfium2 = pytest.importorskip("pypdfium2")
    docs: list[FakeDoc] = []

    def factory(data):
        doc = FakeDoc(data)
        docs.append(doc)
        return doc

    monkeypatch.setattr(pypdfium2, "PdfDocument", factory)
    return docs


def test_iter_pdf_pages_closes_bitmap_and_page(fake_doc):
    pages = list(iter_pdf_pages(b"%PDF-fake", dpi=72))
    assert len(pages) == 3
    doc = fake_doc[0]
    assert all(p.closed for p in doc.pages)
    assert all(p.bitmap.closed for p in doc.pages)
    assert doc.closed


def test_iter_pdf_pages_closes_handles_before_yield(fake_doc):
    gen = iter_pdf_pages(b"%PDF-fake", dpi=72)
    next(gen)
    doc = fake_doc[0]
    assert doc.pages[0].closed
    assert doc.pages[0].bitmap.closed
    gen.close()
    assert doc.closed


def test_indexed_renderer_rejects_oversized_page_before_allocation(fake_doc):
    with pytest.raises(ValueError, match="pixel limit"):
        list(
            iter_pdf_pages_with_index(
                b"%PDF-fake",
                dpi=72,
                max_pixels=100_000,
                max_dimension=10_000,
            )
        )

    doc = fake_doc[0]
    assert not doc.pages[0].rendered
    assert doc.pages[0].closed
    assert doc.closed
