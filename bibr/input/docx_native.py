"""Native DOCX parser: python-docx Document → PaperContents.

Heading levels come directly from Word styles (``Title``,
``Heading 1``-``Heading 6``), giving a faithful multi-level hierarchy at
parse time. This depth is intentionally **not** preserved in the final
output: the shared post-parse stage runs ``assign_hierarchy_from_top_level``
(``pipeline/stages/post_parse.py``), which canonicalizes every *unnumbered*
heading to the IMRaD 1-2 level shape — the same structure the PDF path
produces — so DOCX and PDF outputs stay uniform. Only numbered headings
("2.3.1 …") keep deeper levels. See
docs/superpowers/specs/2026-06-15-docx-hierarchy-flattening-issue.md.

Mirrors :class:`bibr.structure.pdf_parser.PDFParser`'s public surface
(``parse``, ``_deferred_texts``, ``apply_segmentation``,
``create_content_sections``) so the existing pipeline orchestrator can
dispatch on file type without further branching.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from bibr.input.docx_footnotes import load_endnotes as _load_endnotes
from bibr.input.docx_footnotes import load_footnotes as _load_footnotes
from bibr.input.docx_omml import omml_to_text as _omml_to_text
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperFigure,
    PaperFigurePart,
    PaperSection,
    PaperSentence,
    PaperTable,
    PaperTablePart,
    PaperURLLink,
    PaperXref,
)
from bibr.structure.assembler import DeferredText, DocumentAssembler
from bibr.structure.float_labels import FIGURE_WORD, SUPPLEMENT_WORD, TABLE_WORD, caption_label
from bibr.structure.section_tree import infer_level_from_numbering
from bibr.structure.xref_utils import URL_RE, detect_xrefs
from bibr.utils.text import clean_extracted_url

logger = logging.getLogger(__name__)

# A caption paragraph that opens with a figure or table word, labelled or not.
_FIGURE_CAPTION_START_RE = re.compile(
    rf"{SUPPLEMENT_WORD}?{FIGURE_WORD}(?![A-Za-z])", re.IGNORECASE
)
_TABLE_CAPTION_START_RE = re.compile(rf"{SUPPLEMENT_WORD}?{TABLE_WORD}(?![A-Za-z])", re.IGNORECASE)

# OOXML namespaces — used for XPath into paragraph trees.
_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
}

# Run-level children that end the run of text they sit in without carrying any
# text of their own: an explicit line/page/column break, a tab, and the legacy
# carriage return. Concatenating around them fuses the words on either side.
_RUN_SEPARATORS: frozenset[str] = frozenset(
    {
        f"{{{_NS['w']}}}br",
        f"{{{_NS['w']}}}tab",
        f"{{{_NS['w']}}}cr",
    }
)

_HEADING_STYLE_LEVELS: dict[str, int] = {
    "Title": 1,
    "Heading 1": 1,
    "Heading 2": 2,
    "Heading 3": 3,
    "Heading 4": 4,
    "Heading 5": 5,
    "Heading 6": 6,
    # python-docx normalises some style names — accept both spellings
    "heading 1": 1,
    "heading 2": 2,
    "heading 3": 3,
    "heading 4": 4,
    "heading 5": 5,
    "heading 6": 6,
}

# Inline subtrees that carry no document text: deletions and tracked-move
# sources (their text is not the paper as read), field codes and markers
# (the result runs are ordinary w:r and are walked), content-control
# properties, the ruby pronunciation (w:rt duplicates the w:rubyBase it
# annotates), and the VML fallback twin of an AlternateContent drawing
# (walking both would duplicate the text box).
_INLINE_SKIP_TAGS: frozenset[str] = frozenset(
    {
        f"{{{_NS['w']}}}del",
        f"{{{_NS['w']}}}moveFrom",
        f"{{{_NS['w']}}}moveFromRangeStart",
        f"{{{_NS['w']}}}moveFromRangeEnd",
        f"{{{_NS['w']}}}delText",
        f"{{{_NS['w']}}}instrText",
        f"{{{_NS['w']}}}fldChar",
        f"{{{_NS['w']}}}sdtPr",
        f"{{{_NS['w']}}}sdtEndPr",
        f"{{{_NS['w']}}}rt",
        f"{{{_NS['mc']}}}Fallback",
    }
)

# Word stores an Insert > Symbol character from the Symbol font as PUA
# ``F0xx`` (``w:font="Symbol"``, ``w:char="F061"``), where ``xx`` is the
# Adobe Symbol byte. Letters read as Greek and the 0xA0-0xFE block as the
# mathematical and technical symbols below (per the Adobe Symbol encoding:
# F0B1 ±, F0A3 ≤, F0B3 ≥, F0B4 ×, F0B9 ≠, F0BB ≈, F0B0 °, F0AE → — the
# characters statistics text uses most). Extensible delimiter pieces have
# no readable single character and stay skipped, as before.
_SYMBOL_TEXT: dict[int, str] = {
    0x22: "∀",
    0x24: "∃",
    0x27: "∋",
    0x2A: "∗",
    0x2D: "−",
    0x40: "≅",
    0x41: "Α",
    0x42: "Β",
    0x43: "Χ",
    0x44: "Δ",
    0x45: "Ε",
    0x46: "Φ",
    0x47: "Γ",
    0x48: "Η",
    0x49: "Ι",
    0x4A: "ϑ",
    0x4B: "Κ",
    0x4C: "Λ",
    0x4D: "Μ",
    0x4E: "Ν",
    0x4F: "Ο",
    0x50: "Π",
    0x51: "Θ",
    0x52: "Ρ",
    0x53: "Σ",
    0x54: "Τ",
    0x55: "Υ",
    0x56: "ς",
    0x57: "Ω",
    0x58: "Ξ",
    0x59: "Ψ",
    0x5A: "Ζ",
    0x61: "α",
    0x62: "β",
    0x63: "χ",
    0x64: "δ",
    0x65: "ε",
    0x66: "φ",
    0x67: "γ",
    0x68: "η",
    0x69: "ι",
    0x6A: "ϕ",
    0x6B: "κ",
    0x6C: "λ",
    0x6D: "μ",
    0x6E: "ν",
    0x6F: "ο",
    0x70: "π",
    0x71: "θ",
    0x72: "ρ",
    0x73: "σ",
    0x74: "τ",
    0x75: "υ",
    0x76: "ϖ",
    0x77: "ω",
    0x78: "ξ",
    0x79: "ψ",
    0x7A: "ζ",
    0xA0: "€",
    0xA1: "ϒ",
    0xA2: "′",
    0xA3: "≤",
    0xA4: "⁄",
    0xA5: "∞",
    0xA6: "ƒ",
    0xA7: "♣",
    0xA8: "♦",
    0xA9: "♥",
    0xAA: "♠",
    0xAB: "↔",
    0xAC: "←",
    0xAD: "↑",
    0xAE: "→",
    0xAF: "↓",
    0xB0: "°",
    0xB1: "±",
    0xB2: "″",
    0xB3: "≥",
    0xB4: "×",
    0xB5: "∝",
    0xB6: "∂",
    0xB7: "•",
    0xB8: "÷",
    0xB9: "≠",
    0xBA: "≡",
    0xBB: "≈",
    0xBC: "…",
    0xBF: "↵",
    0xC0: "ℵ",
    0xC1: "ℑ",
    0xC2: "ℜ",
    0xC3: "℘",
    0xC4: "⊗",
    0xC5: "⊕",
    0xC6: "∅",
    0xC7: "∩",
    0xC8: "∪",
    0xC9: "⊃",
    0xCA: "⊇",
    0xCB: "⊄",
    0xCC: "⊂",
    0xCD: "⊆",
    0xCE: "∈",
    0xCF: "∉",
    0xD0: "∠",
    0xD1: "∇",
    0xD2: "®",
    0xD3: "©",
    0xD4: "™",
    0xD5: "∏",
    0xD6: "√",
    0xD7: "⋅",
    0xD8: "¬",
    0xD9: "∧",
    0xDA: "∨",
    0xDB: "⇔",
    0xDC: "⇐",
    0xDD: "⇑",
    0xDE: "⇒",
    0xDF: "⇓",
    0xE0: "◊",
    0xE1: "⟨",
    0xE2: "®",
    0xE3: "©",
    0xE4: "™",
    0xE5: "∑",
    0xF9: "⟩",
    0xFA: "∫",
}


def _symbol_text(sym_el) -> str:
    """Readable text for a ``w:sym`` element, or ``""`` when it has none."""
    if sym_el.get(f"{{{_NS['w']}}}font") != "Symbol":
        return ""
    try:
        code = int(sym_el.get(f"{{{_NS['w']}}}char") or "", 16)
    except ValueError:
        return ""
    if code & 0xFF00 != 0xF000:
        return ""
    return _SYMBOL_TEXT.get(code & 0xFF, "")


@dataclass
class _Block:
    """A document block — either a paragraph or a table."""

    kind: str  # "paragraph" | "table"
    obj: Any  # Paragraph or Table


@dataclass
class _InlineAccum:
    """Text accumulated while walking one paragraph's inline XML."""

    text: list[str] = field(default_factory=list)
    # The ``$…$`` each inline equation became, for the late clean-up.
    math: list[str] = field(default_factory=list)
    # (kind, id) pairs — "footnote" and "endnote" ids collide otherwise.
    note_refs: list[tuple[str, str]] = field(default_factory=list)
    # Display-text buffers of enclosing hyperlinks, innermost last.
    link_stack: list[list[str]] = field(default_factory=list)
    had_image: bool = False


def _is_placeholder_sdt(sdt_el) -> bool:
    """True when a content control shows Word's placeholder, not content.

    Journal title-page templates contain empty controls whose
    ``w:sdtPr`` carries ``w:showingPlcHdr``; the ``w:sdtContent`` then holds
    the prompt ("Click or tap here to enter text."), which is Word's chrome
    rather than the paper.
    """
    w = _NS["w"]
    sdt_pr = sdt_el.find(f"{{{w}}}sdtPr")
    return sdt_pr is not None and sdt_pr.find(f"{{{w}}}showingPlcHdr") is not None


def _iter_blocks(doc) -> list[_Block]:
    """Yield body paragraphs, tables, and block-level OMML in document order."""
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    out: list[_Block] = []

    def walk(parent) -> None:
        for child in parent.iterchildren():
            if isinstance(child, CT_P):
                out.append(_Block(kind="paragraph", obj=Paragraph(child, doc)))
            elif isinstance(child, CT_Tbl):
                out.append(_Block(kind="table", obj=Table(child, doc)))
            elif child.tag == f"{{{_NS['m']}}}oMathPara":
                out.append(_Block(kind="math_para", obj=child))
            elif child.tag == f"{{{_NS['w']}}}sdt":
                # Structured document tag (content control). Word wraps whole
                # blocks in these — cover pages, abstract boxes, citation
                # fields, entire bibliographies generated by its reference
                # manager. Only the body's direct children were walked, so
                # every paragraph inside one was invisible to the parse. The
                # payload lives under ``w:sdtContent``; the surrounding
                # ``w:sdtPr`` properties carry no document text. An empty
                # control still showing Word's placeholder prompt contributes
                # no paper text.
                if _is_placeholder_sdt(child):
                    continue
                for sdt_child in child.iterchildren():
                    if sdt_child.tag == f"{{{_NS['w']}}}sdtContent":
                        walk(sdt_child)

    walk(doc.element.body)
    return out


def _heading_level_from_style(style_name: str | None) -> int | None:
    """Return depth from Word style name, or ``None`` if not a heading."""
    if not style_name:
        return None
    return _HEADING_STYLE_LEVELS.get(style_name)


def _outline_level(paragraph_or_style_el) -> int | None:
    """Level from a ``w:outlineLvl`` element (0-based), or ``None``."""
    w = _NS["w"]
    ppr = paragraph_or_style_el.find(f"{{{w}}}pPr")
    if ppr is None:
        return None
    lvl = ppr.find(f"{{{w}}}outlineLvl")
    if lvl is None:
        return None
    try:
        value = int(lvl.get(f"{{{w}}}val") or 0)
    except ValueError:
        return None
    # ECMA-376: 9 is body text, not an outline level — a paragraph carrying
    # it (Word writes it for "Outline level: Body Text") stays body text.
    if value < 0 or value > 8:
        return None
    return value + 1


# Heading styles proper: the base-style chain below resolves custom styles
# derived from these, but never from "Title" — pandoc/Quarto reference docs
# base their Author/Date/Subtitle styles on Title, and those are front
# matter, not sections.
_BASE_HEADING_STYLE_LEVELS: dict[str, int] = {
    name: level for name, level in _HEADING_STYLE_LEVELS.items() if name != "Title"
}


def _heading_level_for_paragraph(paragraph) -> int | None:
    """Heading depth for a paragraph: style name, base styles, outline level.

    Journal and APA templates define their own heading styles based on
    ``Heading N`` — the exact-name lookup missed them, so whole IMRaD
    structures parsed as body text. The base-style chain is walked bounded,
    like :func:`_is_caption_style`. A ``w:outlineLvl`` (direct paragraph
    formatting, then the style's) covers templates that set the level
    without renaming the style.
    """
    style = getattr(paragraph, "style", None)
    style_name = getattr(style, "name", None)
    if isinstance(style_name, str):
        level = _HEADING_STYLE_LEVELS.get(style_name)
        if level is not None:
            return level
    probe = getattr(style, "base_style", None)
    for _ in range(8):  # base-style chains are short; a malformed cycle must end
        if probe is None:
            break
        name = getattr(probe, "name", None)
        level = _BASE_HEADING_STYLE_LEVELS.get(name) if isinstance(name, str) else None
        if level is not None:
            return level
        probe = getattr(probe, "base_style", None)
    level = _outline_level(paragraph._element)
    if level is not None:
        return level
    element = getattr(style, "element", None)
    if element is not None:
        return _outline_level(element)
    return None


def _looks_like_section_header(text: str) -> bool:
    """True when lookup classification names *text* a section, not a title."""
    try:
        from bibr.structure.section_classifier import classify_headers_batch
    except Exception:  # noqa: BLE001 — without the classifier, keep the old read
        return False
    try:
        [(canon, score)] = classify_headers_batch([text])
    except Exception:  # noqa: BLE001 — same fallback
        return False
    # Only an exact-alias hit counts: a word-boundary substring hit (score
    # 0.95) fires on any title containing a section word ("Methods for
    # measuring sleep in older adults"), which rejected real titles.
    return canon != CanonicalSection.UNKNOWN and score == 1.0


def _extract_image_blobs(paragraph, doc) -> list[tuple[bytes, str]]:
    """Find inline ``<w:drawing>`` images in a paragraph; return ``(blob, ext)`` list."""
    out: list[tuple[bytes, str]] = []
    blips = paragraph._element.findall(f".//{{{_NS['w']}}}drawing//{{{_NS['a']}}}blip")
    related = doc.part.related_parts
    for blip in blips:
        rid = blip.get(f"{{{_NS['r']}}}embed")
        if rid is None:
            continue
        image_part = related.get(rid)
        if image_part is None:
            continue
        blob = getattr(image_part, "blob", None)
        if not blob:
            continue
        # python-docx ImagePart exposes .partname like "/word/media/image1.png"
        partname = str(getattr(image_part, "partname", "image"))
        ext = partname.rsplit(".", 1)[-1].lower() if "." in partname else "bin"
        out.append((blob, ext))
    return out


def _is_caption_style(style) -> bool:
    """True for Word's Caption style and styles based on it (pandoc's "Table
    Caption" and "Image Caption")."""
    for _ in range(8):  # base-style chains are short; a malformed cycle must end
        if style is None:
            return False
        if getattr(style, "name", None) == "Caption":
            return True
        style = getattr(style, "base_style", None)
    return False


def _has_picture(paragraph) -> bool:
    return bool(paragraph._element.findall(f".//{{{_NS['w']}}}drawing//{{{_NS['a']}}}blip"))


def _has_embedded_picture(paragraph) -> bool:
    """True when a picture paragraph holds an embedded (not linked) image.

    A linked picture (``r:link``) never yields a figure — there is no blob
    to extract — so its caption must stay body text rather than pair with
    nothing and vanish.
    """
    for blip in paragraph._element.findall(f".//{{{_NS['w']}}}drawing//{{{_NS['a']}}}blip"):
        if blip.get(f"{{{_NS['r']}}}embed") is not None:
            return True
    return False


def _nearest_block(
    blocks: list[_Block], index: int, step: int, *, skip_tables: bool = False
) -> int | None:
    """Index of the nearest block before (``step=-1``) or after (``step=1``)
    *index* that is not an empty paragraph."""
    index += step
    while 0 <= index < len(blocks):
        block = blocks[index]
        if skip_tables and block.kind == "table":
            # A float may sit between a picture and its caption ("under a
            # table, but it names a figure") — look past it. Any other block
            # breaks the adjacency.
            index += step
            continue
        if not (
            block.kind == "paragraph"
            and not (getattr(block.obj, "text", None) or "").strip()
            and not _has_picture(block.obj)
        ):
            return index
        index += step
    return None


def _is_table_caption(blocks: list[_Block], index: int) -> bool:
    """Can the block at *index*, found next to a table, be that table's caption?

    It must be a Caption-styled paragraph, without a picture of its own, that
    does not name a figure. An unlabelled one directly under a picture stays
    the picture's caption.
    """
    block = blocks[index]
    if (
        block.kind != "paragraph"
        or not _is_caption_style(getattr(block.obj, "style", None))
        or _has_picture(block.obj)
    ):
        return False
    text = (getattr(block.obj, "text", None) or "").strip()
    if not text or _FIGURE_CAPTION_START_RE.match(text):
        return False
    if _TABLE_CAPTION_START_RE.match(text):
        return True
    previous = _nearest_block(blocks, index, -1)
    return previous is None or not (
        blocks[previous].kind == "paragraph" and _has_picture(blocks[previous].obj)
    )


def _table_caption_blocks(blocks: list[_Block]) -> dict[int, int]:
    """Map each table block's index to that of its caption paragraph.

    A table's caption is the Caption-styled paragraph directly above or below
    it (empty paragraphs between are skipped). A caption between two tables
    could be either one's, so the side that the unambiguous captions of the
    document sit on is tried first — above, on a tie.
    """
    above: dict[int, int] = {}
    below: dict[int, int] = {}
    for index, block in enumerate(blocks):
        # A table without cells is dropped, and must not take its caption along.
        if block.kind != "table" or not any(row.cells for row in block.obj.rows):
            continue
        for side, step in ((above, -1), (below, 1)):
            neighbour = _nearest_block(blocks, index, step)
            if neighbour is not None and _is_table_caption(blocks, neighbour):
                side[index] = neighbour
    shared = set(above.values()) & set(below.values())
    votes_above = sum(caption not in shared for caption in above.values())
    votes_below = sum(caption not in shared for caption in below.values())
    captions: dict[int, int] = {}
    for side in (above, below) if votes_above >= votes_below else (below, above):
        for table_index, caption_index in side.items():
            if table_index not in captions and caption_index not in captions.values():
                captions[table_index] = caption_index
    return captions


def _is_figure_caption(blocks: list[_Block], index: int) -> bool:
    """Can the block at *index*, found next to a picture, be its caption?

    It must be a Caption-styled paragraph, without a picture of its own,
    that does not name a table.
    """
    block = blocks[index]
    if (
        block.kind != "paragraph"
        or not _is_caption_style(getattr(block.obj, "style", None))
        or _has_picture(block.obj)
    ):
        return False
    text = (getattr(block.obj, "text", None) or "").strip()
    # A labelled figure caption always qualifies; an unlabelled one next to a
    # picture is its caption. Precedence over a neighbouring table's claim is
    # settled by the table pairing, which runs first and marks its captions
    # claimed.
    return bool(text) and not _TABLE_CAPTION_START_RE.match(text)


def _figure_caption_blocks(blocks: list[_Block], claimed: set[int]) -> dict[int, int]:
    """Map each picture paragraph's index to that of its caption paragraph.

    Captions pair by adjacency — the Caption-styled paragraph directly above
    or below the picture (empty paragraphs between are skipped) — instead of
    FIFO across the document, so an uncaptioned image (logo, icon, an extra
    panel) no longer shifts every later caption. A caption between two
    pictures could be either one's, so the side that the unambiguous
    captions of the document sit on is tried first — above, on a tie.
    Captions already claimed by tables are never re-paired.
    """
    above: dict[int, int] = {}
    below: dict[int, int] = {}
    for index, block in enumerate(blocks):
        if block.kind != "paragraph" or not _has_picture(block.obj):
            continue
        if (
            not _has_embedded_picture(block.obj)
            or _heading_level_for_paragraph(block.obj) is not None
        ):
            # A linked picture yields no figure (no blob to extract), and a
            # picture inside a heading paragraph never becomes one either —
            # claiming the adjacent caption would attach it to nothing and
            # delete its text.
            continue
        for side, step in ((above, -1), (below, 1)):
            neighbour = _nearest_block(blocks, index, step, skip_tables=True)
            if (
                neighbour is not None
                and neighbour not in claimed
                and _is_figure_caption(blocks, neighbour)
            ):
                side[index] = neighbour
    shared = set(above.values()) & set(below.values())
    votes_above = sum(caption not in shared for caption in above.values())
    votes_below = sum(caption not in shared for caption in below.values())
    captions: dict[int, int] = {}
    for side in (above, below) if votes_above >= votes_below else (below, above):
        for pic_index, caption_index in side.items():
            if (
                pic_index not in captions
                and caption_index not in claimed
                and caption_index not in captions.values()
            ):
                captions[pic_index] = caption_index
    return captions


class DocxParser:
    """Parses DOCX bytes directly into a :class:`PaperContents`.

    Sentence segmentation is deferred — call :meth:`apply_segmentation`
    after :meth:`parse` with externally-produced segments, identical to
    :class:`PDFParser`.
    """

    def __init__(self, docx_bytes: bytes) -> None:
        self.docx_bytes = docx_bytes
        # Deferred-text buffer + sentence-emission driver shared with PDFParser.
        # page_number is None for DOCX — pages are a render-time concept.
        self.assembler = DocumentAssembler()

        self.sections: list[PaperSection] = []
        self.sentences: list[PaperSentence] = []
        self.links: list[PaperURLLink] = []
        self.tables: list[PaperTable] = []
        self.figures: list[PaperFigure] = []
        self.detected_headers: list[str] = []
        self.detected_footers: list[str] = []
        self.layout_hints: list[tuple[str, int]] = []

        self._section_counter = 0
        self._sentence_counter = 1
        self._paragraph_counter = 0
        self._table_counter = 1
        self._figure_counter = 1
        self._current_section_id = 0
        self._detected_title: str | None = None

        # Notes: text maps from footnotes.xml / endnotes.xml + deferred
        # references for xref linking. The two id spaces are independent (both
        # start at 1), so the maps stay separate and every pending entry
        # records which kind it came from.
        self._footnotes_map: dict[str, str] = {}
        self._endnotes_map: dict[str, str] = {}
        # Each entry: (note_text, body_section_id, deferred_text_index, kind)
        self._pending_footnotes: list[tuple[str, int, int, str]] = []
        # Hyperlink captures awaiting text_id resolution at segmentation time.
        # Each entry: (url, link_text, section_id, deferred_text_index)
        self._pending_url_links: list[tuple[str, str, int, int]] = []
        # Whether any non-empty block has been seen: a Heading 1 reads as the
        # paper title only while it is still the first content block.
        self._seen_content_block = False
        # Whether the last _flush_inline_acc call appended an entry.
        self._last_flush_appended = False

    # ------------------------------------------------------------------
    # Public API (mirrors PDFParser)
    # ------------------------------------------------------------------

    @property
    def _deferred_texts(self) -> list[tuple[str, int | None, int, bool, bool]]:
        """Legacy 5-tuple view of the deferred-text buffer (read-only).

        The source of truth is :attr:`assembler`; this projection keeps the
        historical ``(text, page, section_id, needs_seg, is_formula)`` contract
        that tests and analysis scripts unpack.
        """
        return [
            (e.text, e.page_number, e.section_id, e.needs_segmentation, e.is_formula)
            for e in self.assembler.entries
        ]

    def parse(self) -> PaperContents:
        """Walk the DOCX body and populate sections/tables/deferred texts."""
        from docx import Document  # lazy import — keeps load cost off bibr import

        # Root section (section_id=0)
        self.sections.append(
            PaperSection(
                section_id=0,
                header="Root",
                level=0,
                parent_section_id=None,
            )
        )

        try:
            doc = Document(io.BytesIO(self.docx_bytes))
        except Exception as exc:
            from bibr.exceptions import ProcessingError

            raise ProcessingError(f"Failed to open DOCX: {exc}") from exc

        self._doc = doc  # needed by image/footnote helpers
        self._footnotes_map = _load_footnotes(doc)
        self._endnotes_map = _load_endnotes(doc)

        blocks = _iter_blocks(doc)
        # A table's caption paragraph leaves the body text, as a figure's does.
        table_captions = _table_caption_blocks(blocks)
        figure_captions = _figure_caption_blocks(blocks, set(table_captions.values()))
        caption_blocks = set(table_captions.values()) | set(figure_captions.values())
        # Which block each emitted figure came from, for caption pairing.
        figure_blocks: dict[int, int] = {}
        for index, block in enumerate(blocks):
            if block.kind == "paragraph":
                if index not in caption_blocks:
                    before = len(self.figures)
                    self._handle_paragraph(block.obj)
                    for fig in self.figures[before:]:
                        figure_blocks[id(fig)] = index
            elif block.kind == "table":
                caption_index = table_captions.get(index)
                caption_text = ""
                if caption_index is not None:
                    caption_acc = self._paragraph_inline_text(blocks[caption_index].obj)
                    caption_text = "".join(caption_acc.text)
                    # A note anchored on the caption (a sourced table) queues
                    # like a heading note — its text is walked, not emitted.
                    self._enqueue_note_refs(caption_acc.note_refs)
                self._handle_table(block.obj, caption=caption_text.strip() or None)
            elif block.kind == "math_para":
                self._handle_math_para(block.obj)
        # Pair each picture paragraph with its adjacent caption: every figure
        # from one paragraph shares it (a multi-panel figure's panels), and a
        # picture with no adjacent caption stays uncaptioned instead of
        # shifting every later caption.
        for pic_index, caption_index in figure_captions.items():
            caption_acc = self._paragraph_inline_text(blocks[caption_index].obj)
            caption_text = "".join(caption_acc.text).strip()
            self._enqueue_note_refs(caption_acc.note_refs)
            if not caption_text:
                continue
            for fig in self.figures:
                if figure_blocks.get(id(fig)) == pic_index and not fig.caption:
                    fig.caption = caption_text
                    fig.label = caption_label(caption_text, "figure")

        return PaperContents(
            sentences=[],
            sections=self.sections,
            tables=self.tables,
            links=self.links,
            sections_text={},
            figures=self.figures,
            xrefs=[],
            detected_title=self._detected_title,
            detected_headers=self.detected_headers,
            detected_footers=self.detected_footers,
            layout_hints=self.layout_hints,
        )

    def _make_sentence(
        self,
        entry: DeferredText,
        text: str,
        text_id: int,
        paragraph_id: int,
    ) -> PaperSentence:
        """Build a DOCX sentence — no provenance/region_meta side-channels, and
        never OCR text. It keeps the paragraph's inline equations whose ``$…$``
        text it holds, so the late clean-up unwraps them even where a word
        touches them ("the $n$th"). Held means the text occurs in the sentence:
        a literal "$n$" elsewhere in a paragraph with an equation ``n`` would
        be unwrapped too, and an equation the segmenter splits is not held.

        Formula entries are intentionally left with ``is_display_formula`` at
        its default of ``False`` (the historical DOCX behaviour).
        """
        return PaperSentence(
            text_id=text_id,
            text=text,
            section_id=entry.section_id,
            paragraph_id=paragraph_id,
            page_number=entry.page_number,
            from_ocr=False,
            inline_math=tuple(span for span in entry.inline_math if span in text),
        )

    def apply_segmentation(
        self,
        contents: PaperContents,
        all_segments: list[list[str]],
    ) -> None:
        """Populate sentences from externally-segmented results.

        The shared assembler drives sentence emission (also tracking the last
        text_id per deferred entry for footnote xref linking). URL detection is
        deferred to a second pass here so hyperlink-resolved entries can guard
        against duplicate regex matches on the same text.
        """
        self.sentences, self._sentence_counter, self._paragraph_counter = self.assembler.emit(
            all_segments,
            sentence_factory=self._make_sentence,
            sentence_counter=self._sentence_counter,
            paragraph_counter=self._paragraph_counter,
        )

        # Resolve pending hyperlink captures: the deferred index points to the
        # paragraph entry; use its last emitted sentence's text_id (or the next
        # one if the entry itself produced no sentences).
        for url, link_text, sec_id, deferred_idx in self._pending_url_links:
            text_id = self._find_nearest_text_id(deferred_idx)
            link_paragraph_id: int | None = None
            for sent in self.sentences:
                if sent.text_id == text_id:
                    link_paragraph_id = sent.paragraph_id
                    break
            self.links.append(
                PaperURLLink(
                    url=url,
                    section_id=sec_id,
                    paragraph_id=link_paragraph_id if link_paragraph_id is not None else 0,
                    text_id=text_id,
                    link_text=link_text,
                )
            )

        # Second pass: regex-detect URLs in sentence text, skipping any
        # ``(url, text_id)`` pair already covered by a resolved hyperlink so
        # ``https://...`` displayed verbatim inside a ``w:hyperlink`` doesn't
        # produce a duplicate ``PaperURLLink``.
        # Limitation: ``_find_nearest_text_id`` resolves to the *last* sentence
        # of a multi-sentence paragraph, so a URL-as-display-text hyperlink in
        # an earlier sentence of the same paragraph can still produce a regex
        # duplicate. In practice the fix targets URL-only paragraphs (the
        # common case for verbatim DOI/repository links).
        hyperlink_covered = {(link.url, link.text_id) for link in self.links}
        for sent in self.sentences:
            self._detect_urls(sent, hyperlink_covered)

        contents.sentences = self.sentences
        contents.links = self.links
        contents.sections_text = DocumentAssembler.build_sections_text(self.sentences)
        contents.xrefs = detect_xrefs(self.sentences, self.tables, self.figures)

    def create_content_sections(self, contents: PaperContents) -> None:
        """Create dedicated sections for figures, tables, and footnotes.

        Mirrors :meth:`PDFParser.create_content_sections` so downstream
        post-parse logic sees the same shape.
        """
        # --- Figure sections ---
        for fig in self.figures:
            fig._body_section_id = fig.section_id
            self._section_counter += 1
            section = PaperSection(
                section_id=self._section_counter,
                header=f"Figure {fig.figure_id}",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.FIGURE,
                synthetic_kind="figure",
            )
            contents.sections.append(section)
            fig.section_id = self._section_counter

            if fig.caption:
                self._paragraph_counter += 1
                contents.sentences.append(
                    PaperSentence(
                        text_id=self._sentence_counter,
                        text=fig.caption,
                        section_id=self._section_counter,
                        paragraph_id=self._paragraph_counter,
                        page_number=None,
                        from_ocr=False,
                    )
                )
                self._sentence_counter += 1

        # --- Table sections ---
        for tbl in self.tables:
            tbl._body_section_id = tbl.section_id
            self._section_counter += 1
            section = PaperSection(
                section_id=self._section_counter,
                header=f"Table {tbl.table_id}",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.TABLE,
                synthetic_kind="table",
            )
            contents.sections.append(section)
            tbl.section_id = self._section_counter

            if tbl.caption:
                self._paragraph_counter += 1
                contents.sentences.append(
                    PaperSentence(
                        text_id=self._sentence_counter,
                        text=tbl.caption,
                        section_id=self._section_counter,
                        paragraph_id=self._paragraph_counter,
                        page_number=None,
                        from_ocr=False,
                    )
                )
                self._sentence_counter += 1

        # --- Footnote / endnote sections + xrefs ---
        # Word numbers footnotes and endnotes independently, so the *printed*
        # marker is each note's position within its own kind. ``xref_id`` is the
        # note's own text row (``text_id``), unique across both kinds.
        printed_counts: dict[str, int] = {}
        for fn_text, _orig_section, deferred_idx, kind in self._pending_footnotes:
            printed_counts[kind] = printed_counts.get(kind, 0) + 1
            printed_num = printed_counts[kind]
            label = "Endnote" if kind == "endnote" else "Footnote"
            self._section_counter += 1
            footnote_section_id = self._section_counter
            contents.sections.append(
                PaperSection(
                    section_id=footnote_section_id,
                    header=f"{label} {printed_num}",
                    level=1,
                    parent_section_id=0,
                    section_type=CanonicalSection.FOOTNOTE,
                    synthetic_kind="footnote",
                    footnote_label=str(printed_num),
                )
            )

            self._paragraph_counter += 1
            footnote_text_id = self._sentence_counter
            contents.sentences.append(
                PaperSentence(
                    text_id=footnote_text_id,
                    text=fn_text,
                    section_id=footnote_section_id,
                    paragraph_id=self._paragraph_counter,
                    page_number=None,
                    from_ocr=False,
                )
            )
            self._sentence_counter += 1

            nearest_text_id = self._find_nearest_text_id(deferred_idx)
            contents.xrefs.append(
                PaperXref(
                    xref_id=footnote_text_id,
                    xref_type="foot",
                    contents=str(printed_num),
                    text_id=nearest_text_id,
                )
            )

    def _find_nearest_text_id(self, deferred_idx: int) -> int:
        """Last text_id at or before ``deferred_idx`` in the deferred-texts stream."""
        last_text_id = self.assembler.last_text_id
        if not last_text_id:
            return self.sentences[0].text_id if self.sentences else 1
        clamped = min(max(0, deferred_idx), len(last_text_id) - 1)
        for i in range(clamped, -1, -1):
            tid = last_text_id[i]
            if tid is not None:
                return tid
        return self.sentences[0].text_id if self.sentences else 1

    # ------------------------------------------------------------------
    # Block handlers
    # ------------------------------------------------------------------

    def _handle_paragraph(self, paragraph) -> None:
        style = getattr(paragraph, "style", None)
        style_name = getattr(style, "name", None)

        level = _heading_level_for_paragraph(paragraph)
        if level is not None:
            # Headings walk their runs with the same inline walker: a Title
            # or Heading paragraph can anchor footnote/endnote references
            # (author notes, funding, preregistration) and hide text inside
            # content controls, like any other paragraph.
            inline = self._paragraph_inline_text(paragraph)
            plain = "".join(inline.text).strip()
            if plain:
                self._handle_heading(plain, level, style_name)
            self._enqueue_note_refs(inline.note_refs)
            self._seen_content_block = True
            return

        # Walk the paragraph XML in document order with one recursive inline
        # walker. It descends into any wrapper (w:sdt/w:sdtContent, w:smartTag,
        # w:customXml, w:fldSimple, w:moveTo, w:bdo, w:dir, and
        # mc:AlternateContent taking only mc:Choice), skips deletions,
        # move sources, field codes and the VML fallback twin, and maps
        # w:sym and w:noBreakHyphen to readable text.
        acc = _InlineAccum()
        for child in paragraph._element.iterchildren():
            self._walk_inline(child, acc, collect_only=False)

        # Images: extract after the text walk so we know section context
        if acc.had_image:
            for blob, _ext in _extract_image_blobs(paragraph, self._doc):
                image_b64 = base64.b64encode(blob).decode("ascii")
                fig = PaperFigure(
                    figure_id=self._figure_counter,
                    section_id=self._current_section_id,
                    image_b64=image_b64,
                    caption=None,
                    page_number=None,
                    parts=[
                        PaperFigurePart(
                            page_number=None,
                            bbox=None,
                            image_b64=image_b64,
                        )
                    ],
                )
                self.figures.append(fig)
                self._figure_counter += 1

        self._flush_inline_acc(acc)
        flushed = self._last_flush_appended
        self._last_flush_appended = False
        if acc.had_image or flushed or acc.note_refs:
            self._seen_content_block = True

    def _paragraph_inline_text(self, paragraph) -> _InlineAccum:
        """Walk a paragraph's runs without emitting: heading/caption/cell text."""
        acc = _InlineAccum()
        for child in paragraph._element.iterchildren():
            self._walk_inline(child, acc, collect_only=True)
        return acc

    def _acc_boxed_text(self, txbx_el, acc: _InlineAccum) -> None:
        """Append one legacy/VML text box's paragraphs with word boundaries.

        The same per-paragraph join the ``w:drawing`` branch uses: runs
        concatenate untouched inside a paragraph (Word splits runs mid-word)
        and paragraphs separate, wherever the ``w:txbxContent`` sits.
        """
        w = _NS["w"]
        for para in txbx_el.iter(f"{{{w}}}p"):
            part = "".join(t.text or "" for t in para.iter(f"{{{w}}}t")).strip()
            if part:
                # A boxed paragraph is its own paragraph visually — keep
                # word boundaries on both sides.
                self._acc_separator(acc)
                self._acc_text(acc, part)
                self._acc_separator(acc)

    def _acc_text(self, acc: _InlineAccum, text: str) -> None:
        """Append inline text, mirroring it into any enclosing hyperlink."""
        acc.text.append(text)
        if acc.link_stack:
            acc.link_stack[-1].append(text)

    def _acc_separator(self, acc: _InlineAccum) -> None:
        """Space for markup that ends a run of text, if not already spaced."""
        if acc.text and acc.text[-1] and not acc.text[-1][-1].isspace():
            acc.text.append(" ")
            if acc.link_stack:
                acc.link_stack[-1].append(" ")

    def _flush_inline_acc(self, acc: _InlineAccum) -> None:
        txt = "".join(acc.text).strip()
        acc.text.clear()
        inline_math = tuple(acc.math)
        acc.math.clear()
        if not txt:
            return
        # Footnote refs collected so far attach to this just-flushed entry.
        deferred_idx = self.assembler.append(
            txt, None, self._current_section_id, True, False, inline_math=inline_math
        )
        for kind, note_id in acc.note_refs:
            source = self._footnotes_map if kind == "footnote" else self._endnotes_map
            note_text = source.get(note_id, "").strip()
            if note_text:
                self._pending_footnotes.append(
                    (note_text, self._current_section_id, deferred_idx, kind)
                )
        acc.note_refs.clear()
        self._last_flush_appended = True

    def _enqueue_note_refs(self, refs: list[tuple[str, str]]) -> None:
        """Queue heading/cell note references against the nearest body text."""
        deferred_idx = len(self.assembler)
        for kind, note_id in refs:
            source = self._footnotes_map if kind == "footnote" else self._endnotes_map
            note_text = source.get(note_id, "").strip()
            if note_text:
                self._pending_footnotes.append(
                    (note_text, self._current_section_id, deferred_idx, kind)
                )

    def _walk_inline(self, el, acc: _InlineAccum, *, collect_only: bool) -> None:
        """Pull text/footnotes/math/images from one inline element, recursively."""
        w = _NS["w"]
        m = _NS["m"]
        mc = _NS["mc"]
        tag = el.tag
        if tag in _INLINE_SKIP_TAGS:
            return
        if tag == f"{{{w}}}r":
            for child in el.iterchildren():
                self._walk_inline(child, acc, collect_only=collect_only)
        elif tag == f"{{{w}}}hyperlink":
            # Capture the URL from the relationship target, then walk nested
            # content so its text contributes to paragraph flow.
            rel_id = el.get(f"{{{_NS['r']}}}id")
            target_url: str | None = None
            if rel_id is not None:
                rel = self._doc.part.rels.get(rel_id)
                if rel is not None:
                    target_url = getattr(rel, "target_ref", None) or getattr(rel, "target", None)
            acc.link_stack.append([])
            for child in el.iterchildren():
                self._walk_inline(child, acc, collect_only=collect_only)
            link_text_buf = acc.link_stack.pop()
            if target_url and not collect_only:
                link_text = "".join(link_text_buf).strip() or target_url
                # text_id is unknown until segmentation; record the deferred
                # entry index that *will* hold this paragraph's text. Since
                # we have not yet flushed, the next deferred index is
                # ``len(self.assembler)`` — apply_segmentation resolves
                # it once sentences exist.
                self._pending_url_links.append(
                    (
                        target_url,
                        link_text,
                        self._current_section_id,
                        len(self.assembler),
                    )
                )
        elif tag in (
            f"{{{w}}}ins",
            f"{{{w}}}moveTo",
            f"{{{w}}}smartTag",
            f"{{{w}}}customXml",
            f"{{{w}}}fldSimple",
            f"{{{w}}}bdo",
            f"{{{w}}}dir",
        ):
            # Tracked insertions and moves, smart tags, simple fields and
            # bidirectional wrappers: all real document text, walked inline.
            for child in el.iterchildren():
                self._walk_inline(child, acc, collect_only=collect_only)
        elif tag == f"{{{w}}}sdt":
            # Inline content control (Word citations, Mendeley Cite): the
            # payload lives under w:sdtContent; w:sdtPr carries no text. An
            # empty control still showing the placeholder prompt is skipped.
            if _is_placeholder_sdt(el):
                return
            for child in el.iterchildren():
                if child.tag == f"{{{w}}}sdtContent":
                    for grandchild in child.iterchildren():
                        self._walk_inline(grandchild, acc, collect_only=collect_only)
        elif tag == f"{{{mc}}}AlternateContent":
            # Real Word drawings/text boxes wrap in AlternateContent with a
            # VML Fallback twin — descend into the Choice only, or the text
            # box (and picture) is walked twice.
            for child in el.iterchildren():
                if child.tag == f"{{{mc}}}Choice":
                    for grandchild in child.iterchildren():
                        self._walk_inline(grandchild, acc, collect_only=collect_only)
                    break
        elif tag == f"{{{m}}}oMathPara":
            math_text = " ".join(_omml_to_text(om) for om in el.iter(f"{{{m}}}oMath"))
            math_text = math_text.strip()
            if not math_text:
                return
            if collect_only:
                acc.math.append(f"$${math_text}$$")
                self._acc_text(acc, f"$${math_text}$$")
                return
            # Display math — flush any text so far, then emit math as own entry
            self._flush_inline_acc(acc)
            self._seen_content_block = True
            self.assembler.append(
                f"$${math_text}$$",
                None,
                self._current_section_id,
                needs_segmentation=False,
                is_formula=True,
            )
        elif tag == f"{{{m}}}oMath":
            inline = _omml_to_text(el)
            if inline:
                acc.math.append(f"${inline}$")
                self._acc_text(acc, acc.math[-1])
        elif tag == f"{{{w}}}t":
            if el.text:
                self._acc_text(acc, el.text)
        elif tag in _RUN_SEPARATORS:
            # A line break, tab or carriage return inside a run carries
            # no text of its own, so dropping it fused the words on
            # either side: a Shift+Enter title page collapsed to
            # "Cognitive load and recallJane SmithDepartment of...".
            # Word splits runs mid-word for formatting, so only these
            # explicit separators may contribute whitespace — adjacent
            # <w:t> must still concatenate untouched.
            self._acc_separator(acc)
        elif tag == f"{{{w}}}footnoteReference":
            fn_id = el.get(f"{{{w}}}id")
            if fn_id is not None:
                acc.note_refs.append(("footnote", fn_id))
        elif tag == f"{{{w}}}endnoteReference":
            # Endnote references were dropped on the floor, so a
            # document using endnotes (the humanities convention for
            # bibliographies) exported none of them.
            en_id = el.get(f"{{{w}}}id")
            if en_id is not None:
                acc.note_refs.append(("endnote", en_id))
        elif tag == f"{{{w}}}drawing":
            acc.had_image = True
            # A drawing can be a text box rather than a picture, and
            # its paragraphs are real document text — pull quotes,
            # boxed methods notes, poster-style layouts. Nothing walked
            # into it, so that text was lost entirely.
            for txbx in el.iter(f"{{{w}}}txbxContent"):
                self._acc_boxed_text(txbx, acc)
        elif tag == f"{{{w}}}txbxContent":
            # A legacy VML text box (w:pict/v:shape/v:textbox) reaches here
            # through the generic descend instead — join it per paragraph
            # the same way, or its paragraphs fuse into the running text.
            self._acc_boxed_text(el, acc)
        elif tag == f"{{{w}}}sym":
            sym_text = _symbol_text(el)
            if sym_text:
                self._acc_text(acc, sym_text)
        elif tag == f"{{{w}}}noBreakHyphen":
            self._acc_text(acc, "-")
        else:
            # Any other wrapper (w:ins already handled, bookmarks,
            # proofing markers, VML shapes, unknown tags): descend — the
            # skip-list above names what carries no text.
            for child in el.iterchildren():
                self._walk_inline(child, acc, collect_only=collect_only)

    def _handle_heading(self, text: str, level: int, style_name: str | None) -> None:
        # The Title style is the paper title. A first Heading 1 used to be
        # taken too — but manuscripts often format the title by hand and use
        # Heading 1 for section names, so an 'Introduction' heading became
        # the detected title and post_parse typed that section TITLE. A
        # Heading 1 still reads as the title when it is the first non-empty
        # block and does not name a section ('Paper About X', not
        # 'Introduction' or 'Abstract').
        if self._detected_title is None and (
            style_name == "Title"
            or (
                style_name in ("Heading 1", "heading 1")
                and not self._seen_content_block
                and not _looks_like_section_header(text)
            )
        ):
            self._detected_title = text

        # Deepen via numbering pattern when present (e.g. "2.3.1 Subsection" → 3)
        inferred = infer_level_from_numbering(text)
        if inferred is not None and inferred > level:
            level = inferred

        self._section_counter += 1
        parent_id = 0
        for sec in reversed(self.sections):
            if 0 < sec.level < level:
                parent_id = sec.section_id
                break

        self.sections.append(
            PaperSection(
                section_id=self._section_counter,
                header=text,
                level=level,
                parent_section_id=parent_id,
            )
        )
        self._current_section_id = self._section_counter

    def _handle_math_para(self, omath_para_el) -> None:
        """Block-level OMML — emit as a non-segmented formula deferred entry."""
        m = _NS["m"]
        math_text = " ".join(_omml_to_text(om) for om in omath_para_el.iter(f"{{{m}}}oMath"))
        math_text = math_text.strip()
        if not math_text:
            return
        self._seen_content_block = True
        self.assembler.append(
            f"$${math_text}$$",
            None,
            self._current_section_id,
            needs_segmentation=False,
            is_formula=True,
        )

    def _handle_table(self, table, *, caption: str | None = None) -> None:
        rows = list(table.rows)
        if not rows:
            return

        # Footnote/endnote references inside table cells were lost with
        # cell.text — collect them against the table's section. python-docx
        # repeats the same merged cell for each spanned grid cell, so dedupe
        # by element: otherwise one note in a merged cell queues once per
        # span and every later footnote prints the wrong number.
        seen_cells: set[int] = set()
        for row in rows:
            for cell in row.cells:
                if id(cell._tc) in seen_cells:
                    continue
                seen_cells.add(id(cell._tc))
                for para in cell.paragraphs:
                    self._enqueue_note_refs(self._paragraph_inline_text(para).note_refs)
        self._seen_content_block = True

        # Cell text comes from the same inline walker as headings and
        # captions — cell.text skips content controls, smart tags and simple
        # fields, so a citation in a cell read as "". Paragraphs join with
        # newlines, exactly as cell.text joins them.
        cell_rows = [
            [
                "\n".join(
                    "".join(self._paragraph_inline_text(para).text) for para in cell.paragraphs
                ).strip()
                for cell in row.cells
            ]
            for row in rows
        ]
        # Pad ragged rows so DataFrame construction is uniform
        width = max(len(r) for r in cell_rows)
        cell_rows = [r + [""] * (width - len(r)) for r in cell_rows]

        header_row = cell_rows[0]
        data_rows = cell_rows[1:] if len(cell_rows) > 1 else []
        df = pd.DataFrame(data_rows, columns=header_row)
        if df.empty and not header_row:
            return

        html = df.to_html(index=False)
        self.tables.append(
            PaperTable(
                table_id=self._table_counter,
                df=df,
                tbl_html=html,
                section_id=self._current_section_id,
                caption=caption or None,
                page_number=None,
                parts=[
                    PaperTablePart(
                        page_number=None,
                        bbox=None,
                        tbl_html=html,
                        df=df,
                    )
                ],
                label=caption_label(caption, "table"),
            )
        )
        self._table_counter += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_urls(
        self,
        sent: PaperSentence,
        already_covered: set[tuple[str, int]] | None = None,
    ) -> None:
        """Match URL_RE on the sentence and append PaperURLLink entries.

        ``already_covered`` is an optional ``{(url, text_id)}`` guard set —
        regex hits whose ``(url, text_id)`` is in the set are skipped so a
        ``w:hyperlink`` whose display text equals its URL doesn't produce a
        second link entry.
        """
        for m in URL_RE.finditer(sent.text):
            url = clean_extracted_url(m.group(0))
            if already_covered is not None and (url, sent.text_id) in already_covered:
                continue
            self.links.append(
                PaperURLLink(
                    url=url,
                    section_id=sent.section_id,
                    paragraph_id=sent.paragraph_id,
                    text_id=sent.text_id,
                    link_text=None,
                )
            )
