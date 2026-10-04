"""Hand-built PDFs for the document layer tests.

One synthetic paper covers the shapes the layer has to read right: a running
header in /Artifact marked content, a bold heading, body lines with a
superscript citation and a subscript, coloured text, tagged text with an
MCID, a unit font size scaled by the text matrix, a line-end hyphen, a
rotated page, a page with an offset CropBox, a scanned page with an
invisible OCR layer, text inside a scaled form, an empty page and a page
under a diagonal review watermark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.ocr.test_watermark_text import _REVIEW_WATERMARK, _body, _width

PAGE_W, PAGE_H = 612.0, 792.0
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SAMPLE_PAPER = Path(__file__).resolve().parents[2] / "bibr" / "data" / "sample_paper.pdf"


def _num(value: float) -> bytes:
    return f"{value:.4f}".encode()


def text(
    string: str,
    x: float,
    y: float,
    *,
    size: float = 10.0,
    matrix: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    font: str = "F1",
    mode: int = 0,
    rgb: tuple[float, float, float] | None = None,
) -> bytes:
    """One text object drawing *string* with its origin at (x, y).

    A fill colour is set inside ``q``/``Q``: it is graphics state, which
    ``ET`` does not restore.
    """
    escaped = string.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    ops = b"BT /" + font.encode() + b" " + _num(size) + b" Tf "
    if mode:
        ops += b"%d Tr " % mode
    if rgb is not None:
        ops += b" ".join(_num(value) for value in rgb) + b" rg "
    ops += b" ".join(_num(value) for value in (*matrix, x, y))
    ops += b" Tm (" + escaped.encode("latin-1") + b") Tj ET\n"
    return ops if rgb is None else b"q " + ops + b"Q\n"


def marked(tag: str, body: bytes, mcid: int | None = None) -> bytes:
    if mcid is None:
        return b"/" + tag.encode() + b" BMC\n" + body + b"EMC\n"
    return b"/" + tag.encode() + b" <</MCID %d>> BDC\n" % mcid + body + b"EMC\n"


def image(width: float, height: float, x: float = 0.0, y: float = 0.0) -> bytes:
    matrix = (width, 0.0, 0.0, height, x, y)
    return b"q " + b" ".join(_num(value) for value in matrix) + b" cm /Im1 Do Q\n"


def draw_form(scale: float) -> bytes:
    matrix = (scale, 0.0, 0.0, scale, 0.0, 0.0)
    return b"q " + b" ".join(_num(value) for value in matrix) + b" cm /Fm1 Do Q\n"


@dataclass(frozen=True)
class PageSpec:
    content: bytes
    media: tuple[float, float, float, float] = (0.0, 0.0, PAGE_W, PAGE_H)
    crop: tuple[float, float, float, float] | None = None
    rotate: int = 0
    # Content of the Form XObject /Fm1 and its BBox size.
    form: tuple[bytes, tuple[float, float]] | None = None


def build_pdf(pages: list[PageSpec]) -> bytes:
    """A PDF with fonts /F1 Helvetica, /F2 Times-Roman, /F3 Helvetica-Bold and image /Im1."""
    objects: list[bytes] = [b"", b""]

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    fonts = b" ".join(
        b"/F%d %d 0 R" % (number, add(b"<< /Type /Font /Subtype /Type1 /BaseFont /%s >>" % name))
        for number, name in ((1, b"Helvetica"), (2, b"Times-Roman"), (3, b"Helvetica-Bold"))
    )
    picture = add(
        b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray "
        b"/BitsPerComponent 8 /Length 4 >>\nstream\n\xff\x80\x80\xff\nendstream"
    )
    kids = []
    for page in pages:
        xobjects = b"/Im1 %d 0 R" % picture
        if page.form is not None:
            body, (width, height) = page.form
            form_id = add(
                b"<< /Type /XObject /Subtype /Form /BBox [0 0 "
                + _num(width)
                + b" "
                + _num(height)
                + b"] /Resources << /Font << "
                + fonts
                + b" >> >> /Length %d >>\nstream\n" % len(body)
                + body
                + b"\nendstream"
            )
            xobjects += b" /Fm1 %d 0 R" % form_id
        stream = add(
            b"<< /Length %d >>\nstream\n" % len(page.content) + page.content + b"\nendstream"
        )
        boxes = b"/MediaBox [" + b" ".join(_num(value) for value in page.media) + b"]"
        if page.crop is not None:
            boxes += b" /CropBox [" + b" ".join(_num(value) for value in page.crop) + b"]"
        if page.rotate:
            boxes += b" /Rotate %d" % page.rotate
        kids.append(
            add(
                b"<< /Type /Page /Parent 2 0 R "
                + boxes
                + b" /Resources << /Font << "
                + fonts
                + b" >> /XObject << "
                + xobjects
                + b" >> >> /Contents %d 0 R >>" % stream
            )
        )
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % kid for kid in kids),
        len(kids),
    )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


# --- The synthetic paper ------------------------------------------------------

BODY_SIZE = 10.0
SCRIPT_SIZE = 6.0
SUPERSCRIPT_RAISE = 4.0
SUBSCRIPT_DROP = 2.5
SCALED_MATRIX = 10.0
FORM_SCALE = 0.5
FORM_SIZE = 20.0
RED = (1.0, 0.0, 0.0)
MCID = 3

HEADER = "Journal of Examples 12 (2024) 1-9"
HEADING = "1 Introduction"
SUPERSCRIPT_LINE = ("of born-digital PDFs and runs OCR on scans", "12", ", but")
SUBSCRIPT_LINE = ("mark a footnote or a reference number. Water is H", "2", "O.")
RED_LINE = "Coloured text keeps its fill colour in the layer."
TAGGED_LINE = "Tagged text carries a marked-content id."
SCALED_LINE = "A unit font size scaled by the text matrix."
FORM_LINE = "Text inside a scaled form keeps its size."


def _scripted(parts: tuple[str, str, str], y: float, shift: float) -> bytes:
    before, script, after = parts
    x = 72.0
    out = text(before, x, y)
    x += _width(before, BODY_SIZE)
    out += text(script, x, y + shift, size=SCRIPT_SIZE)
    x += _width(script, SCRIPT_SIZE)
    return out + text(after, x, y)


def _first_page() -> bytes:
    y = 690.0
    lead = 14.0
    out = marked("Artifact", text(HEADER, 72.0, 760.0, size=8.0))
    out += text(HEADING, 72.0, 720.0, size=14.0, font="F3")
    out += text("Prior work on citation parsing reads the text layer", 72.0, y)
    out += _scripted(SUPERSCRIPT_LINE, y - lead, SUPERSCRIPT_RAISE)
    out += text("these tools drop the font sizes and baselines that", 72.0, y - 2 * lead)
    out += _scripted(SUBSCRIPT_LINE, y - 3 * lead, -SUBSCRIPT_DROP)
    out += text(RED_LINE, 72.0, y - 4 * lead, rgb=RED)
    out += marked("P", text(TAGGED_LINE, 72.0, y - 5 * lead), mcid=MCID)
    out += text(
        SCALED_LINE, 72.0, y - 6 * lead, size=1.0, matrix=(SCALED_MATRIX, 0, 0, SCALED_MATRIX)
    )
    out += text("A hyphenated pre-", 72.0, y - 7 * lead, font="F2")
    out += text("existing word ends the paragraph.", 72.0, y - 8 * lead, font="F2")
    return out


def _lines(lines: list[str], *, x: float = 72.0, top: float = 700.0, mode: int = 0) -> bytes:
    return b"".join(
        text(line, x, top - 18.0 * index, size=12.0, mode=mode) for index, line in enumerate(lines)
    )


_PROSE = [
    "The layer keeps every glyph the text layer holds.",
    "Rotated pages keep their glyphs in user space.",
    "Each line ends where the reader breaks it.",
    "References follow the last section of the paper.",
]


def synthetic_paper() -> bytes:
    return build_pdf(
        [
            PageSpec(_first_page()),
            PageSpec(_lines(_PROSE), rotate=90),
            PageSpec(_lines(_PROSE, x=90.0, top=680.0), crop=(36.0, 36.0, 576.0, 756.0)),
            PageSpec(image(PAGE_W, PAGE_H) + _lines(_PROSE, mode=3)),
            PageSpec(
                draw_form(FORM_SCALE),
                form=(
                    text(FORM_LINE, 144.0, 1200.0, size=FORM_SIZE),
                    (PAGE_W / FORM_SCALE, PAGE_H / FORM_SCALE),
                ),
            ),
            PageSpec(b""),
            PageSpec(_REVIEW_WATERMARK + _body()),
        ]
    )


SYNTHETIC_TEXT_SOURCES = [
    "native",
    "native",
    "native",
    "invisible_layer",
    "native",
    "ocr",
    "native",
]


def band_layout(n_pages: int, bands: int = 6) -> list[list[dict]]:
    """Full-width horizontal bands on every page, in the layout frame (0..1000)."""
    step = 1000.0 / bands
    return [
        [
            {
                "label": "text",
                "bbox_2d": [0.0, step * index, 1000.0, step * (index + 1)],
                "content": "",
            }
            for index in range(bands)
        ]
        for _ in range(n_pages)
    ]


def fixture_pdfs() -> dict[str, bytes]:
    """The repository's small PDF fixtures and the synthetic paper."""
    found = {path.name: path.read_bytes() for path in sorted(FIXTURES.glob("*.pdf"))}
    found[SAMPLE_PAPER.name] = SAMPLE_PAPER.read_bytes()
    found["synthetic_paper.pdf"] = synthetic_paper()
    return found
