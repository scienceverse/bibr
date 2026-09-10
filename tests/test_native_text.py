import ctypes
import io
import json
import unicodedata
from pathlib import Path

import pypdfium2
import pypdfium2.raw as pdfium_raw
import pytest

from bibr.ocr.native_text import (
    _build_page_char_records,
    _is_native_text_usable,
    _normalized_bbox_to_pdf_points,
    _printable_ratio,
    fill_font_metadata,
    fill_regions_from_native_text,
    get_native_text_in_bbox,
)

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "native_text_sample.pdf"
PUA_SOURCE_FIXTURE = Path(__file__).parent / "fixtures" / "native_text_pua.json"


class _FakeTextPage:
    raw = object()

    def __init__(self, boxes):
        self._boxes = boxes

    def count_chars(self):
        return len(self._boxes)

    def get_charbox(self, index):
        return self._boxes[index]


def test_page_char_records_combine_utf16_surrogate_pair(monkeypatch):
    code_units = [0xD835, 0xDC46]
    monkeypatch.setattr(
        pypdfium2.raw,
        "FPDFText_GetUnicode",
        lambda _raw, index: code_units[index],
    )

    records = _build_page_char_records(_FakeTextPage([(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 4.0, 2.0)]))

    assert records == [("𝑆", 2.0, 1.0, False)]
    assert all(unicodedata.category(ch) != "Cs" for ch, *_rest in records)
    assert records[0][0].encode("utf-8") == b"\xf0\x9d\x91\x86"


@pytest.mark.parametrize("code_unit", [0xD835, 0xDC46])
def test_page_char_records_replace_unpaired_surrogate(monkeypatch, code_unit):
    monkeypatch.setattr(
        pypdfium2.raw,
        "FPDFText_GetUnicode",
        lambda _raw, _index: code_unit,
    )

    records = _build_page_char_records(_FakeTextPage([(1.0, 2.0, 3.0, 4.0)]))

    assert records == [("\ufffd", 2.0, 3.0, False)]


def _encode_wide(text: str):
    """Encode a Python str as the ctypes c_ushort array FPDFText_SetText wants."""
    return (ctypes.c_ushort * (len(text) + 1))(*[ord(c) for c in text], 0)


def _make_single_text_pdf(
    text: str,
    *,
    font_name: bytes = b"Helvetica",
    page_width: float = 595.0,
    page_height: float = 842.0,
    with_text: bool = True,
) -> bytes:
    """Build a one-page PDF with *text* set in *font_name* (a standard-14 font).

    ``with_text=False`` produces a blank page (no text layer) with the given
    dimensions — used to prove page-size metadata is attached even when a page
    would fall back to OCR.
    """
    doc = pypdfium2.PdfDocument.new()
    page = doc.new_page(page_width, page_height)
    if with_text:
        font_raw = pdfium_raw.FPDFText_LoadStandardFont(doc.raw, font_name)
        text_obj = pdfium_raw.FPDFPageObj_CreateTextObj(doc.raw, font_raw, ctypes.c_float(12))
        pdfium_raw.FPDFText_SetText(text_obj, _encode_wide(text))
        # Place near page centre so a full-page bbox's centre-probe
        # (FPDFText_GetCharIndexAtPos) lands on the text.
        pdfium_raw.FPDFPageObj_Transform(text_obj, 1, 0, 0, 1, page_width * 0.2, page_height * 0.5)
        pdfium_raw.FPDFPage_InsertObject(page.raw, text_obj)
    pdfium_raw.FPDFPage_GenerateContent(page.raw)
    page.close()
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def test_fill_font_metadata_sets_is_italic_true_for_oblique_font():
    """Standard-14 oblique/italic fonts expose italic only via the font name
    (pdfium leaves the descriptor Italic flag unset for them), so name-based
    detection must flag the region as italic."""
    pdf_bytes = _make_single_text_pdf("Italic words here now", font_name=b"Helvetica-Oblique")
    regions = [[{"label": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]]
    fill_font_metadata(pdf_bytes, regions)
    assert regions[0][0]["_is_italic"] is True


def test_fill_font_metadata_sets_is_italic_false_for_regular_font():
    pdf_bytes = _make_single_text_pdf("Regular upright words here", font_name=b"Helvetica")
    regions = [[{"label": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]]
    fill_font_metadata(pdf_bytes, regions)
    assert regions[0][0]["_is_italic"] is False


def test_fill_font_metadata_sets_page_dimensions():
    pdf_bytes = _make_single_text_pdf("Some page text here", page_width=612.0, page_height=792.0)
    regions = [[{"label": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]]
    fill_font_metadata(pdf_bytes, regions)
    assert regions[0][0]["_page_w"] == pytest.approx(612.0)
    assert regions[0][0]["_page_h"] == pytest.approx(792.0)


def test_fill_font_metadata_sets_page_dimensions_even_on_textless_page():
    """Page dimensions are available regardless of a text layer; a scanned /
    image-only page that will fall back to OCR must still carry page_w/page_h."""
    pdf_bytes = _make_single_text_pdf("", page_width=500.0, page_height=700.0, with_text=False)
    regions = [[{"label": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]]
    fill_font_metadata(pdf_bytes, regions)
    assert regions[0][0]["_page_w"] == pytest.approx(500.0)
    assert regions[0][0]["_page_h"] == pytest.approx(700.0)


def test_extracts_text_from_region_bbox():
    """A bbox covering the full first page should yield the page's native text."""
    pdf_bytes = FIXTURE_PDF.read_bytes()
    bbox = [0, 0, 1000, 1000]
    text = get_native_text_in_bbox(pdf_bytes, page_idx=0, bbox_normalized=bbox)
    assert "The quick brown fox" in text


def test_empty_bbox_returns_empty_string():
    pdf_bytes = FIXTURE_PDF.read_bytes()
    text = get_native_text_in_bbox(pdf_bytes, page_idx=0, bbox_normalized=[0, 0, 1, 1])
    assert text == ""


def test_scanned_pdf_returns_empty_string():
    """A scanned/image-only PDF has no text layer."""
    scanned = Path(__file__).parent / "fixtures" / "scanned_sample.pdf"
    text = get_native_text_in_bbox(
        scanned.read_bytes(), page_idx=0, bbox_normalized=[0, 0, 1000, 1000]
    )
    assert text == ""


def test_coordinate_conversion_matches_expected_values():
    """Sanity-check the y-flip math on a known-size page with origin (0,0).

    CropBox (0, 0, 612, 792) (US Letter), bbox [0, 0, 500, 250] in image-space
    (top-left quadrant half-height).
    Expected PDF-point coords: left=0, top=792, right=306, bottom=594.
    """
    left, bottom, right, top = _normalized_bbox_to_pdf_points(
        [0, 0, 500, 250], crop_box=(0, 0, 612, 792)
    )
    assert left == pytest.approx(0.0)
    assert right == pytest.approx(306.0)
    assert top == pytest.approx(792.0)  # y-flip: image y=0 (top) -> PDF y=crop top
    assert bottom == pytest.approx(594.0)  # 792 - 250/1000*792 = 594


def test_coordinate_conversion_full_page():
    """Full-page normalized bbox produces full-page PDF-point bbox."""
    left, bottom, right, top = _normalized_bbox_to_pdf_points(
        [0, 0, 1000, 1000], crop_box=(0, 0, 612, 792)
    )
    assert (left, bottom, right, top) == pytest.approx((0.0, 0.0, 612.0, 792.0))


def test_coordinate_conversion_offsets_by_crop_origin():
    """Pages with a non-zero CropBox origin must offset bboxes by that origin.

    pypdfium2 renders the CropBox to the image the layout bboxes index into,
    but text coords live in the page's native (MediaBox) space. Assuming a
    (0,0) origin shifts every box, slicing text columns. Regression test for
    native-text garbling on publisher PDFs (e.g. Elsevier) whose CropBox
    origin is non-zero.
    """
    # CropBox origin (40, 50), 600 wide x 800 tall. Right-half top quadrant.
    left, bottom, right, top = _normalized_bbox_to_pdf_points(
        [500, 0, 1000, 250], crop_box=(40, 50, 640, 850)
    )
    assert left == pytest.approx(340.0)  # 40 + 500/1000*600
    assert right == pytest.approx(640.0)  # 40 + 1000/1000*600
    assert top == pytest.approx(850.0)  # crop top (y1); image y=0
    assert bottom == pytest.approx(650.0)  # 850 - 250/1000*800


def test_nonzero_crop_origin_does_not_slice_columns():
    """End-to-end: a PDF whose MediaBox origin is (100,100) must not have its
    text columns sliced. The fixture places ALPHA near the left and OMEGA near
    the right edge; a right-half bbox must return OMEGA, not empty.
    """
    fixture = Path(__file__).parent / "fixtures" / "cropbox_offset_sample.pdf"
    pdf_bytes = fixture.read_bytes()
    right_half = get_native_text_in_bbox(
        pdf_bytes, page_idx=0, bbox_normalized=[600, 0, 1000, 1000]
    )
    assert "OMEGA" in right_half
    assert "ALPHA" not in right_half
    left_half = get_native_text_in_bbox(pdf_bytes, page_idx=0, bbox_normalized=[0, 0, 400, 1000])
    assert "ALPHA" in left_half
    assert "OMEGA" not in left_half


_HRV_PDF = Path("data/hrv.pdf")


@pytest.mark.skipif(not _HRV_PDF.exists(), reason="data/hrv.pdf sample not present")
def test_bbox_intersection_does_not_leak_clipped_neighboring_line():
    """A region bbox whose top edge slices through the line above it must not
    pull in the partial glyphs of that neighboring line as garbage.

    Regression test: ``get_text_bounded`` treats a char as "inside" whenever
    its bounding box merely INTERSECTS the query rect, so a bbox edge that
    slices through an affiliation line above the region bleeds in stray
    partial-glyph garbage (e.g. ``pf ygyppypp``) before the real text.
    """
    pdf_bytes = _HRV_PDF.read_bytes()
    text = get_native_text_in_bbox(pdf_bytes, page_idx=0, bbox_normalized=[68, 317, 550, 328])
    assert "pf ygyppypp" not in text
    assert not text.startswith("pf ")
    assert "Department of Psychological Medicine" in text


def test_fill_populates_text_regions_from_native_pdf():
    """Text regions get their `content` pre-filled with native text."""
    pdf_bytes = FIXTURE_PDF.read_bytes()
    regions_per_page = [
        [  # page 0
            {"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
        ],
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes,
        regions_per_page,
        min_chars=5,
        eligible_labels=frozenset({"text"}),
    )
    assert filled[0][0]["_native_text_used"] is True
    assert "The quick brown fox" in filled[0][0]["content"]


def test_fill_skips_table_regions():
    pdf_bytes = FIXTURE_PDF.read_bytes()
    regions = [
        [{"label": "table", "task_type": "table", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes, regions, min_chars=5, eligible_labels=frozenset({"text"})
    )
    assert "_native_text_used" not in filled[0][0]
    assert filled[0][0]["content"] == ""


def test_fill_respects_min_chars_threshold():
    """fill_regions_from_native_text skips / fills based on the min_chars threshold."""
    pdf_bytes = FIXTURE_PDF.read_bytes()
    full_page_bbox = [0, 0, 1000, 1000]
    total_chars = len(
        get_native_text_in_bbox(pdf_bytes, page_idx=0, bbox_normalized=full_page_bbox)
    )
    assert total_chars > 0, "fixture PDF must have native text"

    def _make_regions():
        return [[{"label": "text", "task_type": "text", "bbox_2d": full_page_bbox, "content": ""}]]

    # min_chars one above total: should NOT fill
    filled = fill_regions_from_native_text(
        pdf_bytes, _make_regions(), min_chars=total_chars + 1, eligible_labels=frozenset({"text"})
    )
    assert filled[0][0].get("_native_text_used", False) is False

    # min_chars ten below total: should fill
    filled = fill_regions_from_native_text(
        pdf_bytes, _make_regions(), min_chars=total_chars - 10, eligible_labels=frozenset({"text"})
    )
    assert filled[0][0]["_native_text_used"] is True


def test_fill_accepts_short_heading_native_text_below_body_min_chars(monkeypatch):
    """Short layout headings should use the native PDF text instead of OCR fallback."""
    import bibr.ocr.native_text as native_mod

    pdf_bytes = FIXTURE_PDF.read_bytes()
    monkeypatch.setattr(native_mod, "_text_from_bbox_on_textpage", lambda *a, **k: "Method")

    regions = [
        [
            {
                "label": "paragraph_title",
                "task_type": "text",
                "bbox_2d": [0, 0, 1000, 1000],
                "content": "",
            },
            {"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
        ]
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes,
        regions,
        min_chars=20,
        eligible_labels=frozenset({"paragraph_title", "text"}),
    )

    assert filled[0][0]["_native_text_used"] is True
    assert filled[0][0]["content"] == "Method"
    assert filled[0][1].get("_native_text_used", False) is False
    assert filled[0][1]["content"] == ""


def test_fill_is_idempotent_on_scanned_pdf():
    scanned = (FIXTURE_PDF.parent / "scanned_sample.pdf").read_bytes()
    regions = [
        [{"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]
    ]
    filled = fill_regions_from_native_text(
        scanned, regions, min_chars=5, eligible_labels=frozenset({"text"})
    )
    assert filled[0][0].get("_native_text_used", False) is False
    assert filled[0][0]["content"] == ""


def test_printable_ratio_clean_ascii_text_is_high():
    text = "The quick brown fox jumps over the lazy dog. It was a bright cold day in April."
    assert _printable_ratio(text) == pytest.approx(1.0)


def test_printable_ratio_clean_multilingual_scientific_text_is_high():
    """Diacritics, Greek letters, math symbols, dashes, curly quotes are all 'reasonable'."""
    text = (
        "Müller, J., & Garcia-López, M. (2023). Föhn effects on café résumé—a "
        "pilot study of α, β, and Ω coefficients (p < .001, χ² = 4.2, r = .85 ± .03). "
        "The effect size (Cohen’s d = 0.62) was “significant” across ∑ conditions "
        "and ∫ f(x)dx converged for n = 30 participants."
    )
    assert _printable_ratio(text) > 0.9


def test_printable_ratio_control_char_soup_is_low():
    text = "".join(chr(c) for c in range(1, 20)) * 5
    assert _printable_ratio(text) < 0.3


def test_printable_ratio_cid_artifacts_counted_as_printable_chars():
    """(cid:N) runs are ASCII letters/digits/punctuation — caught by a separate
    artifact check, not by the printable-ratio calculation itself."""
    text = "(cid:72)(cid:88)(cid:94)" * 10
    assert _printable_ratio(text) == pytest.approx(1.0)


def test_printable_ratio_empty_text_is_treated_as_usable():
    assert _printable_ratio("") == 1.0


def test_is_native_text_usable_accepts_clean_scientific_text():
    text = (
        "Müller, J., & Garcia-López, M. (2023). Föhn effects on café résumé—a "
        "pilot study of α, β, and Ω coefficients (p < .001, χ² = 4.2, r = .85 ± .03). "
        "The effect size (Cohen’s d = 0.62) was “significant” across ∑ conditions."
    )
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is True


def test_is_native_text_usable_rejects_cid_artifact_runs():
    text = "(cid:72)(cid:88)(cid:94)(cid:98)(cid:100)(cid:103)" * 5
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is False


def test_is_native_text_usable_rejects_fffd_heavy_text():
    text = "abc���def���ghi���" * 5
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is False


def test_is_native_text_usable_rejects_control_char_soup():
    text = "".join(chr(c) for c in range(1, 20)) * 5
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is False


def test_is_native_text_usable_rejects_symbol_gibberish():
    text = "".join(chr(c) for c in range(0xE000, 0xE030)) * 5  # Private Use Area
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is False


def test_is_native_text_usable_threshold_is_configurable():
    """A noisy-but-not-garbage text is rejected at a strict threshold but
    accepted once the caller relaxes it via config."""
    text = "clean words here" + "★☆♦♣" * 5
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is False
    assert _is_native_text_usable(text, min_printable_ratio=0.2) is True


def test_is_native_text_usable_accepts_normal_whitespace_controls():
    """Tab/newline/CR/FF/VT are legitimate whitespace, not corruption."""
    text = "Line one.\nLine two.\tTabbed.\r\nCRLF too.\x0cFF.\x0bVT." * 3
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is True


def test_is_native_text_usable_rejects_single_c0_control_char():
    """A single non-whitespace C0 control char (e.g. a broken CID leak, like
    U+000F from a LaTeX math font's broken ToUnicode CMap) must reject the
    whole native-text region, even though it barely dents the printable
    ratio in an otherwise clean paragraph."""
    text = (
        "This is a clean paragraph of scientific prose with plenty of words "
        "so that a single stray control character would never trip the "
        "overall printable-ratio gate on its \x0fown, yet it must still be "
        "rejected because genuine text never contains raw control bytes."
    )
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is False


def test_is_native_text_usable_accepts_clean_paragraph_without_control_chars():
    text = (
        "This is a clean paragraph of scientific prose with plenty of words "
        "and no control characters anywhere, so it should be accepted as "
        "genuine native text extracted from the PDF's text layer."
    )
    assert _is_native_text_usable(text, min_printable_ratio=0.85) is True


def test_fill_skips_region_with_corrupt_cid_native_text(monkeypatch):
    """A region whose native text is corrupt (cid artifacts) must be left
    unfilled so the OCR stage falls back to GLM-OCR for it, even though the
    text clears the min_chars threshold."""
    import bibr.ocr.native_text as native_mod

    pdf_bytes = FIXTURE_PDF.read_bytes()
    corrupt_text = "(cid:72)(cid:88)(cid:94)(cid:98)(cid:100)" * 10

    monkeypatch.setattr(native_mod, "_text_from_bbox_on_textpage", lambda *a, **k: corrupt_text)

    regions = [
        [{"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes, regions, min_chars=5, eligible_labels=frozenset({"text"})
    )
    assert filled[0][0].get("_native_text_used", False) is False
    assert filled[0][0]["content"] == ""


def test_fill_resolves_stx_soft_hyphens_before_corruption_gate(monkeypatch):
    """PDFium uses STX for benign line-end hyphens in some native text layers.

    Resolve the marker with the shared OCR-artifact logic instead of rejecting
    the whole region as a corrupt CMap. The resolver joins wrapped words while
    preserving genuine compound hyphens.
    """
    import bibr.ocr.native_text as native_mod

    pdf_bytes = FIXTURE_PDF.read_bytes()
    native_text = "Interviews were con\x02ducted with self\x02proclaimed users."
    monkeypatch.setattr(native_mod, "_text_from_bbox_on_textpage", lambda *a, **k: native_text)

    regions = [
        [{"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes, regions, min_chars=5, eligible_labels=frozenset({"text"})
    )

    assert filled[0][0]["_native_text_used"] is True
    assert filled[0][0]["content"] == "Interviews were conducted with self-proclaimed users."


@pytest.mark.parametrize(
    "critical_label",
    ["text", "content", "vertical_text", "reference", "reference_content"],
)
def test_fill_rejects_source_backed_private_use_critical_region_only(monkeypatch, critical_label):
    """A single PUA glyph in a citation-critical region forces OCR, while a
    clean sibling remains on the native path with its exact source geometry."""
    import bibr.ocr.native_text as native_mod

    source = json.loads(PUA_SOURCE_FIXTURE.read_text(encoding="utf-8"))
    clean, corrupt = source["regions"]
    by_bbox = {
        tuple(clean["bbox_2d"]): clean["native_text"],
        tuple(corrupt["bbox_2d"]): corrupt["native_text"],
    }
    monkeypatch.setattr(
        native_mod,
        "_text_from_bbox_on_textpage",
        lambda _textpage, _crop_box, bbox, **_kwargs: by_bbox[tuple(bbox)],
    )
    regions = [
        [
            {"label": clean["label"], "task_type": "text", "bbox_2d": clean["bbox_2d"]},
            {"label": critical_label, "task_type": "text", "bbox_2d": corrupt["bbox_2d"]},
        ]
    ]

    fill_regions_from_native_text(FIXTURE_PDF.read_bytes(), regions, min_chars=5)

    assert source["source_page"] == 3
    assert regions[0][0]["_native_text_used"] is True
    assert regions[0][0]["content"] == clean["native_text"]
    assert "_native_text_candidate" not in regions[0][0]
    assert regions[0][1].get("_native_text_used") is None
    assert regions[0][1]["_native_text_candidate"] == corrupt["native_text"]
    assert regions[0][1]["_native_text_rejection_reason"] == "private_use"


def test_private_use_fallback_is_label_scoped_not_global(monkeypatch):
    """PUA fallback must not alter the general usability predicate or broaden
    beyond citation-critical labels."""
    import bibr.ocr.native_text as native_mod

    text = "A sufficiently long document title with one private glyph  in context"
    monkeypatch.setattr(native_mod, "_text_from_bbox_on_textpage", lambda *_a, **_k: text)
    regions = [[{"label": "doc_title", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000]}]]

    fill_regions_from_native_text(FIXTURE_PDF.read_bytes(), regions, min_chars=5)

    assert _is_native_text_usable(text, min_printable_ratio=0.85) is True
    assert regions[0][0]["_native_text_used"] is True
    assert regions[0][0]["content"] == text
    assert "_native_text_rejection_reason" not in regions[0][0]


def test_fill_skips_region_with_fffd_heavy_native_text(monkeypatch):
    import bibr.ocr.native_text as native_mod

    pdf_bytes = FIXTURE_PDF.read_bytes()
    corrupt_text = "�" * 40

    monkeypatch.setattr(native_mod, "_text_from_bbox_on_textpage", lambda *a, **k: corrupt_text)

    regions = [
        [{"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes, regions, min_chars=5, eligible_labels=frozenset({"text"})
    )
    assert filled[0][0].get("_native_text_used", False) is False


def test_fill_respects_custom_min_printable_ratio(monkeypatch):
    """Callers can relax the printable-ratio gate (e.g. via config) to accept
    noisier-but-not-garbage text."""
    import bibr.ocr.native_text as native_mod

    pdf_bytes = FIXTURE_PDF.read_bytes()
    noisy_text = "clean words here" + "★☆♦♣" * 5

    monkeypatch.setattr(native_mod, "_text_from_bbox_on_textpage", lambda *a, **k: noisy_text)

    regions = [
        [{"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]
    ]
    filled = fill_regions_from_native_text(
        pdf_bytes,
        regions,
        min_chars=5,
        eligible_labels=frozenset({"text"}),
        min_printable_ratio=0.85,
    )
    assert filled[0][0].get("_native_text_used", False) is False

    filled = fill_regions_from_native_text(
        pdf_bytes,
        regions,
        min_chars=5,
        eligible_labels=frozenset({"text"}),
        min_printable_ratio=0.2,
    )
    assert filled[0][0]["_native_text_used"] is True


def test_fill_partial_mutation_can_be_cleaned_up():
    """If fill is interrupted mid-iteration, the caller can restore regions
    to an OCR-eligible state by clearing the _native_text_used flag and content."""
    # Simulate a partial mutation: first region flagged, second not.
    pages_regions = [
        [
            {
                "label": "text",
                "bbox_2d": [0, 0, 1000, 500],
                "content": "pre-filled",
                "_native_text_used": True,
            },
            {"label": "text", "bbox_2d": [0, 500, 1000, 1000], "content": ""},
        ]
    ]
    # Caller cleanup (what the pipeline hooks' except blocks do):
    for page in pages_regions:
        for r in page:
            if r.pop("_native_text_used", False):
                r["content"] = ""

    assert pages_regions[0][0].get("_native_text_used") is None
    assert pages_regions[0][0]["content"] == ""
    assert pages_regions[0][1].get("_native_text_used") is None
    assert pages_regions[0][1]["content"] == ""


# --- Page /Rotate (audit [16]) ----------------------------------------------


def _rotated_pdf(rotate: int) -> bytes:
    """A 600x800 page with ALPHA near the top-left and OMEGA near the
    bottom-right in *unrotated* user space, carrying ``/Rotate rotate``."""
    content = b"BT /F1 24 Tf 50 750 Td (ALPHA) Tj ET\nBT /F1 24 Tf 460 50 Td (OMEGA) Tj ET\n"
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        (
            b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 600 800]/Rotate "
            + str(rotate).encode()
            + b"/Resources<</Font<</F1 5 0 R>>>>/Contents 4 0 R>>"
        ),
        b"<</Length " + str(len(content)).encode() + b">>stream\n" + content + b"endstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(index).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer<</Size "
        + str(len(objects) + 1).encode()
        + b"/Root 1 0 R>>\nstartxref\n"
        + str(xref_at).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


_ROTATION_QUADRANTS = {
    # (rotation) -> (quadrant holding ALPHA, quadrant holding OMEGA) in the
    # rendered image the layout detector indexes. Verified against
    # page.render(): rotation moves the ink, and the text layer does not move
    # with it.
    0: ("top-left", "bottom-right"),
    90: ("top-right", "bottom-left"),
    180: ("bottom-right", "top-left"),
    270: ("bottom-left", "top-right"),
}

_QUADRANT_BBOX = {
    "top-left": [0, 0, 300, 300],
    "top-right": [700, 0, 1000, 300],
    "bottom-left": [0, 700, 300, 1000],
    "bottom-right": [700, 700, 1000, 1000],
}


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_native_text_follows_page_rotation(rotation):
    """``page.render()`` applies ``/Rotate``, so on a 90/270 page the image is
    transposed while the CropBox and char boxes stay unrotated. Mapping the
    layout bbox through the CropBox alone returned '' for every text region."""
    pdf_bytes = _rotated_pdf(rotation)
    alpha_quadrant, omega_quadrant = _ROTATION_QUADRANTS[rotation]

    alpha = get_native_text_in_bbox(
        pdf_bytes, page_idx=0, bbox_normalized=_QUADRANT_BBOX[alpha_quadrant]
    )
    omega = get_native_text_in_bbox(
        pdf_bytes, page_idx=0, bbox_normalized=_QUADRANT_BBOX[omega_quadrant]
    )

    assert "ALPHA" in alpha
    assert "OMEGA" not in alpha
    assert "OMEGA" in omega
    assert "ALPHA" not in omega


# --- CropBox origin (audit [33]) --------------------------------------------


def test_bbox_pdf_pts_shares_the_frame_page_dimensions_describe():
    """``page.get_size()`` reports the CropBox extent while ``_bbox_pdf_pts``
    included the CropBox origin, so ``0 <= x1 <= x2 <= page_w`` was false on
    any page whose CropBox does not start at (0, 0) — and
    ``front_role_features.from_pdf_bbox`` divides by exactly those dimensions.
    """
    import pypdfium2

    from bibr.ocr.native_text import (
        _attach_bbox_pdf_pts,
        _attach_page_dimensions,
        _page_crop_box,
        _page_rotation,
    )

    fixture = Path(__file__).parent / "fixtures" / "cropbox_offset_sample.pdf"
    doc = pypdfium2.PdfDocument(fixture.read_bytes())
    try:
        page = doc[0]
        regions = [{"label": "text", "bbox_2d": [0, 0, 1000, 1000]}]
        _attach_page_dimensions(page, regions)
        _attach_bbox_pdf_pts(_page_crop_box(page), regions, _page_rotation(page))
        page.close()
    finally:
        doc.close()

    region = regions[0]
    x1, y1, x2, y2 = region["_bbox_pdf_pts"]
    assert 0 <= x1 <= x2 <= region["_page_w"] + 0.01
    assert 0 <= y1 <= y2 <= region["_page_h"] + 0.01
