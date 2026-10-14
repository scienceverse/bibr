"""The text-layer OCR backend fills regions from the PDF text layer, page-gated.

Every PDF here is a fixture or hand-built (``tests/document/_pdfs.py``), and
the layouts are synthetic region boxes: no layout model, no OCR model.
"""

from __future__ import annotations

import pytest

from bibr.ocr import registry
from bibr.ocr.text_layer import read_text_layer
from bibr.ocr.text_layer_quality import judge_page, lexical_profile
from bibr.ocr.text_layer_tables import build_table_html
from bibr.ocr.utils import pdfium_lock
from bibr.processing_warnings import WarningCode
from tests.document._pdfs import FIXTURES, PAGE_H, PAGE_W, SAMPLE_PAPER, PageSpec, build_pdf, text


@pytest.fixture(autouse=True)
def _no_ocr_model(monkeypatch):
    """Reading the text layer never creates an OCR backend."""

    def create(*_args, **_kwargs):
        raise AssertionError("the text-layer read must not create an OCR backend")

    monkeypatch.setattr(registry, "create", create)


def _box(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    """PDF points (y up) to the 0..1000 layout space (y down)."""
    return [
        x0 / PAGE_W * 1000,
        (PAGE_H - y1) / PAGE_H * 1000,
        x1 / PAGE_W * 1000,
        (PAGE_H - y0) / PAGE_H * 1000,
    ]


def _region(label: str, box: list[float], task: str = "text") -> dict:
    return {"label": label, "task_type": task, "bbox_2d": box, "content": ""}


def _render(pdf: bytes, scale: float = 2.0):
    import pypdfium2

    with pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf)
        try:
            return [doc[i].render(scale=scale).to_pil() for i in range(len(doc))]
        finally:
            doc.close()


def _lines(lines: list[str], *, x: float = 72.0, top: float = 700.0, step: float = 14.0) -> bytes:
    return b"".join(text(line, x, top - step * k) for k, line in enumerate(lines))


def _paragraph_box(count: int, *, top: float = 700.0, step: float = 14.0) -> list[float]:
    return _box(60, top - step * (count - 1) - 6, 560, top + 14)


_PROSE = [
    "We measured the reading speed of the participants in the two groups and",
    "found that the effect of the training was larger than in the earlier work.",
    "The results show that the method is robust to the choice of the stimuli,",
    "and that it can be used with the samples that are common in the field.",
    "This is in line with the theory of the authors, but it is not a proof of",
    "the model, which was tested on a larger set of data from the same lab.",
]


def _caesar(line: str, shift: int = 3) -> str:
    def rot(ch: str) -> str:
        if "a" <= ch <= "z":
            return chr((ord(ch) - 97 + shift) % 26 + 97)
        if "A" <= ch <= "Z":
            return chr((ord(ch) - 65 + shift) % 26 + 65)
        return ch

    return "".join(rot(ch) for ch in line)


# --- Filling regions ---------------------------------------------------------


def _sample_paper_layout() -> list[list[dict]]:
    return [
        [
            _region("doc_title", [170, 70, 830, 118]),
            _region("text", [370, 128, 630, 148]),
            _region("paragraph_title", [98, 185, 185, 200]),
            _region("abstract", [98, 206, 902, 316]),
            _region("figure_title", [98, 573, 450, 590]),
            _region("table", [240, 595, 815, 698], task="table"),
            _region("image", [100, 700, 200, 710], task="skip"),
        ],
        [
            _region("paragraph_title", [98, 118, 210, 133]),
            _region("reference", [98, 140, 902, 455]),
        ],
    ]


def test_sample_paper_fills_every_readable_region():
    pdf = SAMPLE_PAPER.read_bytes()
    layout = _sample_paper_layout()

    read = read_text_layer(pdf, layout, page_images=_render(pdf))

    assert [(page.index, page.ok) for page in read.pages] == [(0, True), (1, True)]
    first, second = read.layout_results
    by_label = {region["label"]: region for region in first}
    assert by_label["doc_title"]["content"].startswith("The Coefficient of Rodential Efficiency")
    assert by_label["paragraph_title"]["content"] == "Abstract"
    assert "Coefficient of Rodential Efficiency (CRE)" in by_label["abstract"]["content"]
    assert (
        by_label["figure_title"]["content"] == "Table 1. Simulated sample characteristics by group."
    )
    assert by_label["table"]["content"].startswith(
        "<table><tr><td>Group</td><td>N</td><td>Mean CRE</td><td>SD</td></tr>"
        "<tr><td>A</td><td>12</td><td>0.91</td><td>0.14</td></tr>"
    )
    assert by_label["table"]["_text_layer_table"] is True
    # A figure is not read, and not counted unread.
    assert by_label["image"]["content"] == ""
    assert "_text_layer_unread" not in by_label["image"]
    assert second[1]["content"].startswith("[1] Vole, A., & Capybarov, B. (2023).")
    for region in (*first, *second):
        if region["task_type"] != "skip":
            assert region["_native_text_used"] is True, region["label"]
    assert read.tables_rebuilt == 1
    assert read.unread == {}
    assert [w.code for w in read.warnings] == [WarningCode.TEXT_LAYER_TABLE_REBUILT]
    # The input layout is left as it was.
    assert layout[0][0]["content"] == ""


def test_native_text_fixture_is_read():
    pdf = (FIXTURES / "native_text_sample.pdf").read_bytes()

    read = read_text_layer(pdf, [[_region("text", [50, 50, 950, 950])]], page_images=_render(pdf))

    assert read.pages[0].ok
    assert read.layout_results[0][0]["content"] == (
        "The quick brown fox jumps over the lazy dog.\r\nAnother line of text for robustness."
    )
    assert read.warnings == []


def test_page_furniture_formulas_and_short_regions():
    pdf = build_pdf(
        [
            PageSpec(
                text("Journal of Tests 12 (2020) 1-10", 72, 760)
                + _lines(_PROSE[:3])
                + text("x = a + b", 200, 600)
                + text("(1)", 500, 600)
                + text("2.1", 72, 560, size=12)
                + text("7", 300, 40)
            )
        ]
    )
    layout = [
        [
            _region("header", _box(60, 754, 300, 772)),
            _region("text", _paragraph_box(3)),
            _region("formula", _box(190, 594, 260, 612), task="formula"),
            _region("formula_number", _box(495, 594, 520, 612)),
            _region("paragraph_title", _box(60, 554, 100, 574)),
            _region("footer", _box(290, 34, 320, 52)),
        ]
    ]

    read = read_text_layer(pdf, layout)

    header, body, formula, number, heading, footer = read.layout_results[0]
    assert header["content"] == "Journal of Tests 12 (2020) 1-10"
    assert body["content"].startswith("We measured the reading speed")
    assert formula["content"] == "x = a + b"
    assert formula["_formula_source"] == "text_layer"
    assert number["content"] == "(1)"
    assert heading["content"] == "2.1"
    # One digit is below the three-character allowance of a footer.
    assert footer["content"] == ""
    assert footer["_text_layer_unread"] is True
    assert read.formulas_read == 1
    assert read.unread == {"footer": 1}
    codes = [w.code for w in read.warnings]
    assert codes == [WarningCode.TEXT_LAYER_REGION_UNREAD, WarningCode.TEXT_LAYER_FORMULA_TEXT]
    assert "footer x1" in read.warnings[0].message


def test_native_filled_region_is_kept():
    pdf = (FIXTURES / "native_text_sample.pdf").read_bytes()
    region = _region("text", [50, 50, 950, 950])
    region.update(content="already read", _native_text_used=True)

    read = read_text_layer(pdf, [[region]])

    assert read.layout_results[0][0]["content"] == "already read"
    assert read.filled == {"text": 1}


def test_page_indices_map_layout_to_absolute_pages():
    pdf = build_pdf([PageSpec(text("First page text", 72, 700)), PageSpec(_lines(_PROSE[:2]))])

    read = read_text_layer(pdf, [[_region("text", _paragraph_box(2))]], page_indices=[1])

    assert read.pages[0].index == 1
    assert read.layout_results[0][0]["content"].startswith("We measured the reading speed")


def test_caption_box_drops_figure_text_above_the_caption():
    pdf = build_pdf(
        [
            PageSpec(
                text("0", 100, 440, size=7)
                + text("10", 200, 440, size=7)
                + text("20", 300, 440, size=7)
                + text("Time (s)", 190, 428, size=7)
                + text("Figure 2. Response time by condition in the two", 72, 410)
                + text("experiments of the study.", 72, 397)
            )
        ]
    )
    caption = _region("figure_title", _box(60, 390, 400, 452))

    read = read_text_layer(pdf, [[caption]])

    assert read.layout_results[0][0]["content"] == (
        "Figure 2. Response time by condition in the two\r\nexperiments of the study."
    )
    # The region keeps the layout model's box.
    assert read.layout_results[0][0]["bbox_2d"] == caption["bbox_2d"]


def test_caption_without_a_caption_start_reads_the_whole_box():
    pdf = build_pdf(
        [PageSpec(text("Mean scores", 72, 440, size=7) + text("Notes on the data.", 72, 420))]
    )

    read = read_text_layer(pdf, [[_region("figure_title", _box(60, 414, 400, 452))]])

    assert read.layout_results[0][0]["content"] == "Mean scores\r\nNotes on the data."


# --- Tables ------------------------------------------------------------------

_TABLE_ROWS = [("Name", "Age", "Score"), ("Ann", "31", "0.82"), ("Bob", "27", "0.75")]


def _table_pdf() -> bytes:
    content = b"".join(
        text(cell, 100 + 100 * col, 600 - 15 * row)
        for row, cells in enumerate(_TABLE_ROWS)
        for col, cell in enumerate(cells)
    )
    return build_pdf([PageSpec(content)])


def test_three_by_three_table_becomes_html_the_table_parser_reads():
    from bibr.structure.parse_media import MediaHandlersMixin

    pdf = _table_pdf()

    read = read_text_layer(
        pdf, [[_region("table", _box(90, 560, 360, 615), task="table")]], page_images=_render(pdf)
    )

    html = read.layout_results[0][0]["content"]
    assert html == (
        "<table>"
        "<tr><td>Name</td><td>Age</td><td>Score</td></tr>"
        "<tr><td>Ann</td><td>31</td><td>0.82</td></tr>"
        "<tr><td>Bob</td><td>27</td><td>0.75</td></tr>"
        "</table>"
    )
    frame = MediaHandlersMixin._parse_html_table(html)
    assert list(frame.columns) == ["Name", "Age", "Score"]
    assert frame.values.tolist() == [["Ann", "31", "0.82"], ["Bob", "27", "0.75"]]
    assert [w.code for w in read.warnings] == [WarningCode.TEXT_LAYER_TABLE_REBUILT]


def _glyphs(word: str, x: float, y: float, *, width: float = 6.0, height: float = 10.0):
    return [
        (ch, x + width * k, y, x + width * (k + 1) - 2.0, y + height) for k, ch in enumerate(word)
    ]


def test_table_builder_lets_a_spanning_header_cross_a_column_gap():
    chars = _glyphs("Reaction time condition", 100, 10)
    for row in range(1, 8):
        y = 10 + 16 * row
        chars += _glyphs(f"r{row}", 100, y) + [(" ", 0, 0, 0, 0)]
        chars += _glyphs(f"{row}0", 200, y) + [(" ", 0, 0, 0, 0)]
        chars += _glyphs(f"{row}.5", 300, y)

    html = build_table_html(chars)

    assert html is not None
    # The header crosses the first gap; each of its words goes to the column
    # holding its centre, and the columns stay those of the data rows.
    assert html.startswith("<table><tr><td>Reaction</td><td>time condition</td><td></td></tr>")
    assert "<tr><td>r1</td><td>10</td><td>1.5</td></tr>" in html
    assert html.count("<tr>") == 8


def test_table_builder_joins_tight_digit_boxes_and_escapes_cells():
    # Narrow digits leave a gap of a third of a height between their boxes.
    chars = [("1", 0, 0, 4, 10), ("2", 7, 0, 11, 10), (" ", 0, 0, 0, 0), ("<", 40, 0, 46, 10)]
    chars += [(".", 46, 0, 48, 10), ("0", 48, 0, 54, 10), ("5", 54, 0, 60, 10)]

    assert build_table_html(chars) == "<table><tr><td>12</td><td>&lt;.05</td></tr></table>"
    assert build_table_html([(" ", 0, 0, 0, 0)]) is None


# --- Page gates ----------------------------------------------------------------


def _prose_pdf(lines: list[str], extra: bytes = b"") -> bytes:
    return build_pdf([PageSpec(_lines(lines) + extra)])


def test_good_prose_page_passes_every_gate():
    pdf = _prose_pdf(_PROSE)

    read = read_text_layer(pdf, [[_region("text", _paragraph_box(6))]], page_images=_render(pdf))

    page = read.pages[0]
    assert page.ok, page.metrics
    assert page.metrics["coverage"] == 1.0
    assert page.metrics["glyphs_on_ink"] > 0.9
    assert page.metrics["stopword_share"] > 0.2


def test_text_layer_off_its_ink_fails():
    from PIL import Image

    pdf = _prose_pdf(_PROSE)
    image = _render(pdf)[0]
    shifted = Image.new("RGB", image.size, "white")
    shifted.paste(image, (0, 400))

    read = read_text_layer(pdf, [[_region("text", _paragraph_box(6))]], page_images=[shifted])

    page = read.pages[0]
    assert (page.ok, page.reason) == (False, "text_off_ink")
    assert read.layout_results[0][0]["content"] == ""
    assert read.layout_results[0][0]["_text_layer_unread"] is True
    assert [w.code for w in read.warnings] == [WarningCode.TEXT_LAYER_PAGE_UNUSABLE]


def test_ink_the_text_layer_does_not_hold_fails():
    # A block of printed matter drawn as paths, with a heading in the text layer.
    pdf = _prose_pdf(["Results"], b"q 0 g 72 450 300 230 re f Q\n")

    read = read_text_layer(
        pdf, [[_region("text", _box(60, 440, 400, 714))]], page_images=_render(pdf)
    )

    assert read.pages[0].reason == "ink_without_text"


def test_caesar_shifted_text_layer_fails():
    pdf = _prose_pdf([_caesar(line) for line in _PROSE])

    read = read_text_layer(pdf, [[_region("text", _paragraph_box(6))]], page_images=_render(pdf))

    page = read.pages[0]
    assert (page.ok, page.reason) == (False, "implausible_text")
    assert page.metrics["stopword_share"] < 0.04


def test_text_layer_without_word_spaces_fails():
    pdf = _prose_pdf([line.replace(" ", "") for line in _PROSE])

    read = read_text_layer(pdf, [[_region("text", _paragraph_box(6))]])

    assert read.pages[0].reason == "implausible_text"


def test_scanned_fixture_has_no_text_layer():
    pdf = (FIXTURES / "scanned_sample.pdf").read_bytes()

    read = read_text_layer(pdf, [[_region("text", [50, 50, 950, 950])]], page_images=_render(pdf))

    assert (read.pages[0].ok, read.pages[0].reason) == (False, "no_text_layer")
    assert read.layout_results[0][0]["_text_layer_unread"] is True
    assert read.unread == {"text": 1}
    assert read.warnings[0].code == WarningCode.TEXT_LAYER_PAGE_UNUSABLE


def test_page_with_nothing_to_read_and_no_text_layer_passes():
    pdf = (FIXTURES / "scanned_sample.pdf").read_bytes()

    read = read_text_layer(pdf, [[_region("image", [50, 50, 950, 950], task="skip")]])

    assert read.pages[0].ok
    assert read.warnings == []


def test_a_stray_stamp_over_a_scan_is_low_coverage():
    pdf = build_pdf([PageSpec(text("Downloaded from example.org on 1 May 2020", 72, 770, size=7))])
    layout = [
        [
            _region("header", _box(60, 764, 300, 780)),
            _region("text", _box(60, 100, 550, 700)),
        ]
    ]

    read = read_text_layer(pdf, layout)

    assert read.pages[0].reason == "low_coverage"
    assert [region["content"] for region in read.layout_results[0]] == ["", ""]


def test_cid_codes_in_most_regions_are_a_corrupt_text_layer():
    cid = "(cid:12)(cid:15)(cid:3) (cid:7)(cid:9)"
    pdf = build_pdf([PageSpec(text(cid, 72, 700) + text(cid, 72, 600))])
    layout = [[_region("text", _box(60, 690, 550, 716)), _region("text", _box(60, 590, 550, 616))]]

    read = read_text_layer(pdf, layout)

    assert read.pages[0].reason == "corrupt_text"
    assert all(region["_native_gate"] == "usability" for region in read.layout_results[0])


@pytest.mark.parametrize(
    "scan",
    [(PAGE_W, PAGE_H, 0.0, 0.0), (400.0, 500.0, 100.0, 200.0)],
    ids=["page-sized", "cropped-to-the-text-block"],
)
def test_invisible_ocr_layer_over_a_scan_is_not_read(scan):
    from tests.document._pdfs import image

    width, height, x, y = scan
    pdf = build_pdf(
        [PageSpec(image(width, height, x, y) + text("Scanned words", 120, 600, mode=3))]
    )

    read = read_text_layer(pdf, [[_region("text", _box(110, 590, 550, 616))]])

    assert read.pages[0].reason == "invisible_ocr_layer"
    assert read.layout_results[0][0]["content"] == ""


def test_a_little_invisible_text_on_a_born_digital_page_is_read():
    lines = _lines(_PROSE)
    pdf = build_pdf([PageSpec(lines + text("hidden", 72, 760, mode=3))])

    read = read_text_layer(pdf, [[_region("text", _paragraph_box(6))]])

    assert read.pages[0].ok


# --- Gate units and page triage --------------------------------------------------


def test_lexical_profile_leaves_other_scripts_and_short_text_alone():
    assert lexical_profile(["Αυτό είναι ένα ελληνικό κείμενο για τη δοκιμή. " * 10]) == {}
    assert lexical_profile(["Too short to judge."]) == {}
    english = lexical_profile([" ".join(_PROSE)])
    assert english["glued_letter_share"] == 0.0
    assert english["stopword_share"] > 0.2


def test_judge_page_names_the_first_failing_gate():
    assert judge_page(coverage=None, rejected_regions=0, read_regions=0, ink={}, lexical={}).ok
    low = judge_page(
        coverage=0.1, rejected_regions=3, read_regions=3, ink={"glyphs_on_ink": 0}, lexical={}
    )
    assert low.reason == "low_coverage"
    one_bad = judge_page(coverage=1.0, rejected_regions=1, read_regions=1, ink={}, lexical={})
    assert one_bad.ok


@pytest.mark.parametrize(
    ("reason", "kind"),
    [
        ("no_text_layer", "scan"),
        ("low_coverage", "scan"),
        ("invisible_ocr_layer", "scan"),
        ("text_off_ink", "broken_text_layer"),
        ("implausible_text", "broken_text_layer"),
        ("corrupt_text", "broken_text_layer"),
    ],
)
def test_page_triage_follows_the_text_layer_verdict(reason, kind):
    from bibr.ocr.pdf_inspection import PdfPageInspection
    from bibr.scan.page_kind import classify_page

    page = PdfPageInspection(
        index=0,
        width=PAGE_W,
        height=PAGE_H,
        crop_box=(0.0, 0.0, PAGE_W, PAGE_H),
        char_count=500,
        text_layer_ok=False,
        text_layer_reason=reason,
    )

    assert classify_page(page, [{"label": "text", "_native_text_used": True}]).value == kind
