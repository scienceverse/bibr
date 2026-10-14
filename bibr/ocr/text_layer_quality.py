"""Page gates for reading a PDF from its text layer alone.

With no OCR model to fall back on, the text-layer backend
(:mod:`bibr.ocr.text_layer`) must decide per page whether the embedded text
can stand in for what is printed. Each gate measures one way a text layer
lies:

- **coverage**: the share of the text-labelled layout area that holds any
  text-layer glyph. A scan with a stray text stamp ("Downloaded from ...")
  has text, but not where the page's text is.
- **ink**: on the page image the layout model already saw, glyph boxes
  should sit on ink, and the ink inside text regions should be under glyph
  boxes. A text layer shifted off its page, or one drawn over a different
  image, fails the first; text drawn as an image or as paths fails the
  second.
- **lexical**: Latin-script text should have word lengths and function
  words like prose. A broken font encoding that maps every letter to another
  one (a Caesar shift) passes the character gates but has no function words;
  a layer with no word spaces glues whole lines into one token.

The thresholds are deliberately loose: a page that fails loses all its text,
so a gate only fires on a layer that is wrong, not on one that is unusual.
They were checked on development PDFs; see the module constants.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

#: A glyph box ``(x0, y0, x1, y1)`` in the 0..1000 layout space, y down.
Box = tuple[float, float, float, float]

# --- Thresholds ---------------------------------------------------------------

#: A page fails when less than this share of its text-labelled layout area
#: holds text-layer glyphs.
MIN_TEXT_COVERAGE = 0.5
#: A text region counts as covered when it holds at least this many glyphs.
MIN_REGION_GLYPHS = 2
#: A page fails when less than this share of its glyph boxes touch ink.
MIN_GLYPHS_ON_INK = 0.6
#: ... or when less than this share of the ink in its text regions lies
#: under glyph boxes.
MIN_INK_UNDER_GLYPHS = 0.35
#: The ink gates need at least this many glyphs / ink pixels to say anything.
MIN_INK_GLYPHS = 40
MIN_INK_PIXELS = 400
#: Pixels darker than this (0..255 grey) are ink.
INK_THRESHOLD = 160
#: The page image is reduced to about this width before the ink gates.
INK_WORK_WIDTH = 1000
#: The lexical gate reads pages with at least this many letters, at least
#: this share of them Latin script ...
MIN_LEXICAL_LETTERS = 200
MIN_LATIN_LETTER_SHARE = 0.6
#: ... and counts function words on pages with at least this many tokens.
MIN_LEXICAL_TOKENS = 30
#: A page fails when more than this share of its letters sit in tokens of
#: at least GLUED_TOKEN_LETTERS letters (a layer without word spaces) ...
MAX_GLUED_LETTER_SHARE = 0.5
GLUED_TOKEN_LETTERS = 25
#: ... or when fewer than this share of its tokens are common function words.
MIN_STOPWORD_SHARE = 0.04

# Common function words of the Latin-script languages bibr's corpora carry.
# A text layer in any of them clears MIN_STOPWORD_SHARE many times over; a
# substituted alphabet almost never forms these words.
_STOPWORD_LINES = (
    "a an and are as at be by for from has have in is it its not of on or that the this to",
    "was were which with we our their these those also can may than been but",
    "der die das und ist im in den von zu mit sich des auf für nicht ein eine als auch es an",
    "werden aus er hat dass sie nach wird bei einer um am sind noch wie einem über so zum",
    "le la les de des du un une et est en que qui dans pour par sur au aux pas plus ce il",
    "elle sont avec ou se son sa ses leur nous vous",
    "el los las del y es en que por con para una al lo como más se su sus o sobre entre",
    "il lo la gli le di da del della che per con non una sono nel nella al",
    "o os as do da dos das em no na que com para por um uma não mais se ao à",
    "de het een en van in is op te dat die met voor zijn niet aan er ook als bij",
    "och att det som en är av för med till den på om har inte og er til af ikke paa",
    "w i z na się do nie że jest to o od po dla oraz jak",
    "a se na je že v z o do to jako pro by",
    "și în de la cu pe un o este care din pentru nu",
    "i je u na se da za od su ili kao",
    "ve bir bu da de için ile olarak olan",
    "dan yang di dari ini itu untuk dengan pada adalah tidak",
    "ja on ei että se kuin mutta tai myös",
    "és a az hogy nem egy is meg van",
    "và của các là có được trong cho một những với không này để",
    "ir kad yra su iš į tai bet kaip un ar kas par uz ja et ei see oli",
    "na ya wa kwa la katika ni za ang ng sa mga at ay",
    "dhe të në një për që me og að er í á sem",
    "i el les amb eta da ez du yn ac agus an ar die van het nie",
    "və bu ilə üçün va bilan uchun et in est ad cum non quod ut",
)
_STOPWORDS = frozenset(word for line in _STOPWORD_LINES for word in line.split())

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True)
class PageGateResult:
    """One page's gate verdict: ``reason`` is None when the page passes."""

    reason: str | None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.reason is None


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def text_coverage(region_boxes: Sequence[Box], glyph_boxes: Sequence[Box]) -> float | None:
    """Area-weighted share of *region_boxes* holding text-layer glyph centres.

    None when there is no text-labelled area to cover.
    """
    total = sum(_area(box) for box in region_boxes)
    if total <= 0:
        return None
    import numpy as np

    if glyph_boxes:
        glyphs = np.asarray(glyph_boxes, dtype=np.float32)
        cx = (glyphs[:, 0] + glyphs[:, 2]) / 2.0
        cy = (glyphs[:, 1] + glyphs[:, 3]) / 2.0
    else:
        cx = cy = np.zeros(0, dtype=np.float32)
    covered = 0.0
    for box in region_boxes:
        inside = (cx >= box[0]) & (cx <= box[2]) & (cy >= box[1]) & (cy <= box[3])
        if int(inside.sum()) >= MIN_REGION_GLYPHS:
            covered += _area(box)
    return covered / total


def _pixel_box(box: Box, width: int, height: int, pad: int) -> tuple[int, int, int, int]:
    x0 = int(box[0] * width / 1000.0) - pad
    y0 = int(box[1] * height / 1000.0) - pad
    x1 = int(round(box[2] * width / 1000.0)) + pad
    y1 = int(round(box[3] * height / 1000.0)) + pad
    return max(0, x0), max(0, y0), min(width, max(x1, x0 + 1)), min(height, max(y1, y0 + 1))


def ink_agreement(
    page_image: Image.Image,
    glyph_boxes: Sequence[Box],
    region_boxes: Sequence[Box],
) -> dict[str, float]:
    """How well the glyph boxes and the page's ink agree.

    Returns ``glyphs_on_ink`` (share of glyph boxes touching ink) and
    ``ink_under_glyphs`` (share of the ink inside *region_boxes* under a
    glyph box). A measure is left out when there is too little to measure.
    """
    import numpy as np

    image = page_image.convert("L")
    factor = max(1, image.width // INK_WORK_WIDTH)
    if factor > 1:
        image = image.reduce(factor)
    ink = np.asarray(image, dtype=np.uint8) < INK_THRESHOLD
    height, width = ink.shape
    out: dict[str, float] = {}

    if len(glyph_boxes) >= MIN_INK_GLYPHS:
        integral = np.zeros((height + 1, width + 1), dtype=np.int32)
        integral[1:, 1:] = ink.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
        on_ink = 0
        for box in glyph_boxes:
            x0, y0, x1, y1 = _pixel_box(box, width, height, pad=1)
            total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
            if total > 0:
                on_ink += 1
        out["glyphs_on_ink"] = on_ink / len(glyph_boxes)

    region_mask = np.zeros_like(ink)
    for box in region_boxes:
        x0, y0, x1, y1 = _pixel_box(box, width, height, pad=0)
        region_mask[y0:y1, x0:x1] = True
    region_ink = ink & region_mask
    ink_pixels = int(region_ink.sum())
    if ink_pixels >= MIN_INK_PIXELS:
        glyph_mask = np.zeros_like(ink)
        for box in glyph_boxes:
            x0, y0, x1, y1 = _pixel_box(box, width, height, pad=2)
            glyph_mask[y0:y1, x0:x1] = True
        out["ink_under_glyphs"] = int((region_ink & glyph_mask).sum()) / ink_pixels
    return out


def _is_latin_letter(ch: str) -> bool:
    try:
        return unicodedata.name(ch).startswith("LATIN")
    except ValueError:
        return False


def lexical_profile(texts: Iterable[str]) -> dict[str, float]:
    """Token statistics of Latin-script text; empty when there is too little.

    ``glued_letter_share``: share of letters in tokens of at least
    :data:`GLUED_TOKEN_LETTERS` letters. ``stopword_share``: share of tokens
    of two or more letters that are common function words.
    """
    tokens = [token for text in texts for token in _TOKEN_RE.findall(text)]
    letters = sum(len(token) for token in tokens)
    if letters < MIN_LEXICAL_LETTERS:
        return {}
    latin = sum(1 for token in tokens for ch in token if _is_latin_letter(ch))
    if latin / letters < MIN_LATIN_LETTER_SHARE:
        return {}
    glued = sum(len(token) for token in tokens if len(token) >= GLUED_TOKEN_LETTERS)
    profile = {"tokens": float(len(tokens)), "glued_letter_share": glued / letters}
    words = [token.lower() for token in tokens if len(token) >= 2]
    if len(words) >= MIN_LEXICAL_TOKENS:
        profile["stopword_share"] = sum(1 for word in words if word in _STOPWORDS) / len(words)
    return profile


def judge_page(
    *,
    coverage: float | None,
    rejected_regions: int,
    read_regions: int,
    ink: dict[str, float],
    lexical: dict[str, float],
) -> PageGateResult:
    """Combine the gate measures into one verdict, the first failing gate named.

    ``rejected_regions`` of ``read_regions`` text regions with text-layer
    text failed the character gates (CID codes, U+FFFD, control or
    private-use characters); most of them failing is a broken font encoding,
    the rule :mod:`bibr.scan.page_kind` uses.
    """
    metrics: dict[str, float] = {**ink, **lexical}
    if coverage is not None:
        metrics["coverage"] = coverage
    if read_regions:
        metrics["rejected_share"] = rejected_regions / read_regions
    if coverage is not None and coverage < MIN_TEXT_COVERAGE:
        return PageGateResult("low_coverage", metrics)
    if rejected_regions >= 2 and rejected_regions >= 0.5 * read_regions:
        return PageGateResult("corrupt_text", metrics)
    if ink.get("glyphs_on_ink", 1.0) < MIN_GLYPHS_ON_INK:
        return PageGateResult("text_off_ink", metrics)
    if ink.get("ink_under_glyphs", 1.0) < MIN_INK_UNDER_GLYPHS:
        return PageGateResult("ink_without_text", metrics)
    if (
        lexical.get("glued_letter_share", 0.0) > MAX_GLUED_LETTER_SHARE
        or lexical.get("stopword_share", 1.0) < MIN_STOPWORD_SHARE
    ):
        return PageGateResult("implausible_text", metrics)
    return PageGateResult(None, metrics)
