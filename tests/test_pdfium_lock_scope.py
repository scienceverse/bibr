import io

import pytest

pytest.importorskip("pypdfium2")


def _build_minimal_pdf_bytes(num_pages: int) -> bytes:
    """Build a minimal multi-page PDF (one blank page each) for tests."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    for _ in range(num_pages):
        pdf.new_page(72, 72)
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def test_pdfium_lock_held_for_full_doc_lifetime(monkeypatch):
    """PDFium has global mutable C state — the lock must wrap the entire
    document lifetime, not be released between page renders. Otherwise a
    concurrent caller can open a second PdfDocument and corrupt the state
    of the first."""
    from bibr.ocr import image_utils
    from bibr.ocr import utils as ocr_utils

    enters = []

    real_lock = ocr_utils.pdfium_lock

    class _SpyLock:
        def __enter__(self):
            enters.append(True)
            return real_lock.__enter__()

        def __exit__(self, *a):
            return real_lock.__exit__(*a)

    monkeypatch.setattr(ocr_utils, "pdfium_lock", _SpyLock())
    monkeypatch.setattr(image_utils, "pdfium_lock", _SpyLock())

    pdf_bytes = _build_minimal_pdf_bytes(num_pages=3)
    pages = list(image_utils.iter_pdf_pages_with_index(pdf_bytes, dpi=72))
    assert len(pages) == 3
    # Lock should be acquired once for the whole doc, not per page.
    assert len(enters) == 1, enters
