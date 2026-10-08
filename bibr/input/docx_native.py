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
from dataclasses import dataclass
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
from bibr.processing_warnings import ProcessingWarning, WarningCode
from bibr.structure.assembler import DeferredText, DocumentAssembler
from bibr.structure.float_labels import FIGURE_WORD, SUPPLEMENT_WORD, TABLE_WORD, caption_label
from bibr.structure.html_table import rendered_size as _rendered_size
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

# Inline containers whose content is paragraph text: hyperlinks, tracked
# insertions and moves, content controls (Word's citation tool and Mendeley
# Cite put each citation in one; its runs sit under w:sdtContent), simple
# fields, smart tags, custom XML and bidi embeddings. Their text was dropped,
# so "(Smith, 2020)" vanished from the sentence citing it. Deleted text
# (w:del, w:moveFrom) stays out.
_INLINE_WRAPPERS: frozenset[str] = frozenset(
    f"{{{_NS['w']}}}{tag}"
    for tag in (
        "hyperlink",
        "ins",
        "moveTo",
        "sdt",
        "sdtContent",
        "fldSimple",
        "smartTag",
        "customXml",
        "dir",
        "bdo",
    )
)
_W_R = f"{{{_NS['w']}}}r"
_W_DRAWING = f"{{{_NS['w']}}}drawing"
_A_BLIP = f"{{{_NS['a']}}}blip"
_M_OMATH = f"{{{_NS['m']}}}oMath"
_W_TXBX_CONTENT = f"{{{_NS['w']}}}txbxContent"
_W_STYLE = f"{{{_NS['w']}}}style"
_W_STYLE_ID = f"{{{_NS['w']}}}styleId"
_W_TYPE = f"{{{_NS['w']}}}type"
_W_DEFAULT = f"{{{_NS['w']}}}default"
_W_NAME = f"{{{_NS['w']}}}name"
_W_BASED_ON = f"{{{_NS['w']}}}basedOn"
_ON = frozenset({"1", "true", "on"})  # true ST_OnOff values

# Table grid limits. gridSpan is a free integer that python-docx's row.cells
# repeated a cell for, one copy per column, so a 36 KB file spanning 2e9
# columns ran for days; it is read as at most this wide, as the HTML table
# reader caps colspan.
_MAX_GRID_SPAN = 1000
# A table over the per-table limit, or one taking the document past a
# per-document one, is dropped with a warning. Clamped spans still multiply: a
# 50-byte cell fills 1,000 columns, and a few-byte vMerge continuation repeats
# the merged cell, text included, across its full width. A table's grid cells
# are rows x widest row plus _COLUMN_CELLS per column: pandas pays per column
# what it pays for some 20 cells, and a 36 KB one-row table 1,000,000 columns
# wide took three minutes. The document limit bounds table work to a minute or
# two. The table HTML is limited by its rendered size: every grid cell that
# repeats a merged cell's text writes it again, escaped (pandas turns "&" into
# "&amp;"), with some _CELL_MARKUP_CHARS of markup, and one character past
# U+00FF or U+FFFF makes the whole string 2 or 4 bytes a character (PEP 393).
# Counting raw characters let ten kilobytes of "&" and emoji merged across a
# table render to over a gigabyte.
_MAX_TABLE_CELLS = 1_000_000
_MAX_DOCUMENT_TABLE_CELLS = 4_000_000
_COLUMN_CELLS = 100
_CELL_MARKUP_CHARS = 16
_MAX_DOCUMENT_TABLE_BYTES = 64 * 1024 * 1024
_W_VAL = f"{{{_NS['w']}}}val"
_GRID_BEFORE = f"{{{_NS['w']}}}trPr/{{{_NS['w']}}}gridBefore"
_GRID_SPAN = f"{{{_NS['w']}}}tcPr/{{{_NS['w']}}}gridSpan"
_V_MERGE = f"{{{_NS['w']}}}tcPr/{{{_NS['w']}}}vMerge"

# Figure limits. Every picture showing one image part got its own base64 copy,
# though the package stores the image once: a 1 MB file gave 300 figures and
# 412 MiB of image data. The copy is now shared, but the export still writes it
# once per figure, so image data is counted per figure. Past the validator's
# 128 MiB ceiling on the whole package, which distinct images cannot pass,
# figures keep their caption without an image. Pictures past the first
# _MAX_FIGURES are dropped.
_MAX_FIGURES = 1000
_MAX_FIGURE_IMAGE_BYTES = 128 * 1024 * 1024

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


@dataclass
class _Block:
    """A document block — either a paragraph or a table."""

    kind: str  # "paragraph" | "table"
    obj: object  # Paragraph or Table


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
                # ``w:sdtPr`` properties carry no document text.
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


def _image_parts(paragraph, doc) -> list:
    """The image parts shown by a paragraph's inline ``<w:drawing>`` pictures,
    in order; a part shown twice is listed twice."""
    out = []
    related = doc.part.related_parts
    for blip in _drawing_blips(paragraph._element):
        rid = blip.get(f"{{{_NS['r']}}}embed")
        if rid is None:
            continue
        image_part = related.get(rid)
        if image_part is None or not getattr(image_part, "blob", None):
            continue
        out.append(image_part)
    return out


def _decimal(el, default: int) -> int:
    """The ``w:val`` number of a decimal-number element, or *default* when it
    is absent or not a number."""
    if el is None:
        return default
    try:
        return int(el.get(_W_VAL))
    except (TypeError, ValueError):
        return default


def _table_rows(tbl) -> list[list[tuple[object, int]]]:
    """Each ``w:tr`` of a table as ``(w:tc, columns)`` pairs: the cell whose
    text fills that many grid columns of the row.

    Mirrors python-docx's ``row.cells`` (a cell repeated across its gridSpan, a
    vMerge continuation showing the cell its column continues) without its
    costs: that resolved each continuation by recursing up the column, so tall
    merged columns took quadratic time and then raised RecursionError. Here a
    row resolves against the one above it. A continuation with no cell above
    at its offset reads as a cell of its own, where python-docx raised.
    """
    rows: list[list[tuple[object, int]]] = []
    above: dict[int, tuple[object, int]] = {}
    for tr in tbl.tr_lst:
        row: list[tuple[object, int]] = []
        here: dict[int, tuple[object, int]] = {}
        offset = _decimal(tr.find(_GRID_BEFORE), 0)
        for tc in tr.tc_lst:
            span = min(max(_decimal(tc.find(_GRID_SPAN), 1), 1), _MAX_GRID_SPAN)
            cell = (tc, span)
            v_merge = tc.find(_V_MERGE)
            if v_merge is not None and v_merge.get(_W_VAL, "continue") == "continue":
                cell = above.get(offset, cell)
            here[offset] = cell
            row.append(cell)
            offset += span
        rows.append(row)
        above = here
    return rows


class _Styles:
    """A document's paragraph styles, each looked up once.

    python-docx searches the whole styles part on every ``paragraph.style`` and
    every ``base_style`` step, so the parse cost paragraphs x styles: a 56 KB
    file of 10,000 paragraphs and 10,000 styles took 52 s. A paragraph's style
    is found as python-docx finds it: the first style with the id its
    ``w:pStyle`` names if that is a paragraph style, else the last default
    paragraph style.
    """

    def __init__(self, doc) -> None:
        self._by_id: dict[str, Any] = {}  # w:style elements
        self._default: Any = None
        for style in doc.styles.element.iterchildren(_W_STYLE):
            style_id = style.get(_W_STYLE_ID)
            if style_id is not None:
                self._by_id.setdefault(style_id, style)
            if style.get(_W_TYPE) == "paragraph" and style.get(_W_DEFAULT) in _ON:
                self._default = style
        self._resolved: dict[str | None, tuple[str | None, bool]] = {}

    def of(self, paragraph) -> tuple[str | None, bool]:
        """A paragraph's style name, and whether that is Word's Caption style
        or one based on it (pandoc's "Table Caption" and "Image Caption")."""
        style_id = paragraph._p.style
        resolved = self._resolved.get(style_id)
        if resolved is None:
            style = self._by_id.get(style_id) if style_id else None
            if style is None or style.get(_W_TYPE) != "paragraph":
                style = self._default
            resolved = self._resolved[style_id] = (_style_name(style), self._is_caption(style))
        return resolved

    def _is_caption(self, style) -> bool:
        for _ in range(8):  # base-style chains are short; a malformed cycle must end
            if style is None:
                return False
            if _style_name(style) == "Caption":
                return True
            based_on = style.find(_W_BASED_ON)
            style = None if based_on is None else self._by_id.get(based_on.get(_W_VAL))
        return False


def _style_name(style) -> str | None:
    """A ``w:style``'s name as python-docx shows it ("heading 1" as "Heading 1")."""
    from docx.styles import BabelFish

    name = None if style is None else style.find(_W_NAME)
    value = None if name is None else name.get(_W_VAL)
    return None if value is None else BabelFish.internal2ui(value)


def _drawing_blips(p):
    """The ``a:blip`` pictures inside a body paragraph's ``w:drawing`` elements,
    each once. A ``.//w:drawing//a:blip`` search lists a picture once per
    drawing around it: a picture in a text box's drawing became two figures,
    and 250 nested drawings made 25 million matches of 100,000 pictures."""
    for blip in p.iter(_A_BLIP):
        if next(blip.iterancestors(_W_DRAWING), None) is not None:
            yield blip


def _has_picture(paragraph) -> bool:
    return next(_drawing_blips(paragraph._element), None) is not None


def _display_math_text(omath_para_el) -> str:
    """A display equation's text: each ``m:oMath`` in the ``m:oMathPara``, read
    once. An equation nested in another is read as part of the outer one: read
    again on its own, a nest 250 deep turned 1 MB of math into 250 MB."""
    return " ".join(
        _omml_to_text(om)
        for om in omath_para_el.iter(_M_OMATH)
        if next(om.iterancestors(_M_OMATH), None) is None
    ).strip()


def _paragraph_text(p) -> str:
    """A ``w:p``'s text as python-docx's ``CT_P.text`` reads it, but with the
    runs inside inline wrappers: that read only direct runs and hyperlinks, so
    a title in a content control lost the heading, and a caption numbered by a
    simple field lost its number."""
    return "".join(run.text for run in _wrapped_runs(p))


def _wrapped_runs(el):
    """The ``w:r`` children of *el* and of the inline wrappers in it, in order."""
    for child in el.iterchildren():
        if child.tag == _W_R:
            yield child
        elif child.tag in _INLINE_WRAPPERS:
            yield from _wrapped_runs(child)


def _nearest_block(blocks: list[_Block], index: int, step: int) -> int | None:
    """Index of the nearest block before (``step=-1``) or after (``step=1``)
    *index* that is not an empty paragraph."""
    index += step
    while 0 <= index < len(blocks):
        block = blocks[index]
        if not (
            block.kind == "paragraph"
            and not _paragraph_text(block.obj._p).strip()
            and not _has_picture(block.obj)
        ):
            return index
        index += step
    return None


def _is_table_caption(blocks: list[_Block], index: int, styles: _Styles) -> bool:
    """Can the block at *index*, found next to a table, be that table's caption?

    It must be a Caption-styled paragraph, without a picture of its own, that
    does not name a figure. An unlabelled one directly under a picture stays
    the picture's caption.
    """
    block = blocks[index]
    if block.kind != "paragraph" or not styles.of(block.obj)[1] or _has_picture(block.obj):
        return False
    text = _paragraph_text(block.obj._p).strip()
    if not text or _FIGURE_CAPTION_START_RE.match(text):
        return False
    if _TABLE_CAPTION_START_RE.match(text):
        return True
    previous = _nearest_block(blocks, index, -1)
    return previous is None or not (
        blocks[previous].kind == "paragraph" and _has_picture(blocks[previous].obj)
    )


def _table_caption_blocks(
    blocks: list[_Block], kept_tables: set[int], styles: _Styles
) -> dict[int, int]:
    """Map each kept table block's index to that of its caption paragraph.

    A table's caption is the Caption-styled paragraph directly above or below
    it (empty paragraphs between are skipped). A caption between two tables
    could be either one's, so the side that the unambiguous captions of the
    document sit on is tried first — above, on a tie.
    """
    above: dict[int, int] = {}
    below: dict[int, int] = {}
    # A dropped table (no cells, or too large) must not take its caption along.
    for index in sorted(kept_tables):
        for side, step in ((above, -1), (below, 1)):
            neighbour = _nearest_block(blocks, index, step)
            if neighbour is not None and _is_table_caption(blocks, neighbour, styles):
                side[index] = neighbour
    shared = set(above.values()) & set(below.values())
    votes_above = sum(caption not in shared for caption in above.values())
    votes_below = sum(caption not in shared for caption in below.values())
    captions: dict[int, int] = {}
    taken: set[int] = set()  # a scan of captions.values() per table was quadratic
    for side in (above, below) if votes_above >= votes_below else (below, above):
        for table_index, caption_index in side.items():
            if table_index not in captions and caption_index not in taken:
                captions[table_index] = caption_index
                taken.add(caption_index)
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
        # Headings that can still parent the next one, levels increasing.
        self._open_headings: list[PaperSection] = []

        # Notes: text maps from footnotes.xml / endnotes.xml + deferred
        # references for xref linking. The two id spaces are independent (both
        # start at 1), so the maps stay separate and every pending entry
        # records which kind it came from.
        self._footnotes_map: dict[str, str] = {}
        self._endnotes_map: dict[str, str] = {}
        # Each referenced note once, in first-reference order: (note_text, kind).
        # A note referenced again was queued again, text and all, so one note
        # referenced 20,000 times in a 34 KB file made 1 GB of sentences.
        self._pending_footnotes: list[tuple[str, str]] = []
        self._note_index: dict[tuple[str, str], int] = {}  # (kind, id) -> queue index
        # One xref per (queue index, deferred_text_index) pair: each deferred
        # entry referencing a note links to it once, however many marks it holds.
        self._note_refs: dict[tuple[int, int], None] = {}
        # Hyperlink captures awaiting text_id resolution at segmentation time.
        # Each entry: (url, link_text, section_id, deferred_text_index)
        self._pending_url_links: list[tuple[str, str, int, int]] = []
        # The latest text_id at or before each deferred entry, set at segmentation.
        self._nearest_text_ids: list[int | None] = []
        # Figures awaiting a Caption-styled paragraph immediately after
        self._unfilled_caption_figures: list[PaperFigure] = []

        # Resource limits: each image part is encoded once and shared by the
        # figures showing it; the counters feed the warnings parse() records.
        self.processing_warnings: list[ProcessingWarning] = []
        self._image_b64: dict[object, str] = {}
        self._figure_image_bytes = 0
        self._figures_dropped = 0
        self._figure_images_omitted = 0
        self._table_cells = 0
        self._table_bytes = 0
        self._tables_dropped = 0

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
        self._styles = _Styles(doc)
        self._footnotes_map = _load_footnotes(doc)
        self._endnotes_map = _load_endnotes(doc)

        blocks = _iter_blocks(doc)
        # Read every table's cells first: which tables are kept decides which
        # paragraphs are table captions.
        table_cells = {
            index: self._table_cells_of(block.obj)
            for index, block in enumerate(blocks)
            if block.kind == "table"
        }
        # A table's caption paragraph leaves the body text, as a figure's does.
        table_captions = _table_caption_blocks(
            blocks,
            {index for index, cells in table_cells.items() if cells is not None},
            self._styles,
        )
        caption_blocks = set(table_captions.values())
        for index, block in enumerate(blocks):
            if block.kind == "paragraph":
                if index not in caption_blocks:
                    self._handle_paragraph(block.obj)
            elif block.kind == "table":
                cells = table_cells[index]
                if cells is None:
                    continue
                caption_index = table_captions.get(index)
                caption = (
                    _paragraph_text(blocks[caption_index].obj._p).strip()
                    if caption_index is not None
                    else None
                )
                self._handle_table(cells, caption=caption)
            elif block.kind == "math_para":
                self._handle_math_para(block.obj)
        self._record_limit_warnings()

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
            processing_warnings=self.processing_warnings,
        )

    def _record_limit_warnings(self) -> None:
        """Record what the table and figure limits left out of the parse."""
        if self._tables_dropped:
            self.processing_warnings.append(
                ProcessingWarning(
                    WarningCode.DOCX_TABLE_DROPPED,
                    f"Dropped {self._tables_dropped} table(s) over the size limits "
                    f"({_MAX_TABLE_CELLS:,} grid cells per table; {_MAX_DOCUMENT_TABLE_CELLS:,} "
                    f"grid cells and {_MAX_DOCUMENT_TABLE_BYTES // 2**20} MiB of "
                    "table HTML per document)",
                )
            )
        if self._figures_dropped:
            self.processing_warnings.append(
                ProcessingWarning(
                    WarningCode.DOCX_FIGURES_DROPPED,
                    f"Dropped {self._figures_dropped} picture(s) past the first "
                    f"{_MAX_FIGURES:,} figures",
                )
            )
        if self._figure_images_omitted:
            self.processing_warnings.append(
                ProcessingWarning(
                    WarningCode.DOCX_FIGURE_IMAGES_OMITTED,
                    f"Kept {self._figure_images_omitted} figure(s) without their image: the "
                    f"figures' image data would pass {_MAX_FIGURE_IMAGE_BYTES // 2**20} MiB",
                )
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
        # Each link and note reference walked back over the entries that made
        # no sentence to find its text_id; carrying it forward once is linear.
        latest: int | None = None
        self._nearest_text_ids = []
        for text_id in self.assembler.last_text_id:
            latest = latest if text_id is None else text_id
            self._nearest_text_ids.append(latest)

        # Resolve pending hyperlink captures: the deferred index points to the
        # paragraph entry; use its last emitted sentence's text_id (or the next
        # one if the entry itself produced no sentences). Each link scanned the
        # sentences for its paragraph, so 100,000 links after 20,000
        # paragraphs took 53 s; the paragraphs are now indexed once.
        paragraph_of: dict[int, int] = {}
        for sent in self.sentences:
            paragraph_of.setdefault(sent.text_id, sent.paragraph_id)
        for url, link_text, sec_id, deferred_idx in self._pending_url_links:
            text_id = self._find_nearest_text_id(deferred_idx)
            link_paragraph_id = paragraph_of.get(text_id)
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
        note_rows: list[tuple[int, int]] = []  # (text_id, printed number) per note
        for fn_text, kind in self._pending_footnotes:
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
            note_rows.append((footnote_text_id, printed_num))

        for index, deferred_idx in self._note_refs:
            footnote_text_id, printed_num = note_rows[index]
            contents.xrefs.append(
                PaperXref(
                    xref_id=footnote_text_id,
                    xref_type="foot",
                    contents=str(printed_num),
                    text_id=self._find_nearest_text_id(deferred_idx),
                )
            )

    def _find_nearest_text_id(self, deferred_idx: int) -> int:
        """Last text_id at or before ``deferred_idx`` in the deferred-texts stream."""
        nearest = self._nearest_text_ids
        if nearest:
            text_id = nearest[min(max(0, deferred_idx), len(nearest) - 1)]
            if text_id is not None:
                return text_id
        return self.sentences[0].text_id if self.sentences else 1

    # ------------------------------------------------------------------
    # Block handlers
    # ------------------------------------------------------------------

    def _handle_paragraph(self, paragraph) -> None:
        style_name, caption_style = self._styles.of(paragraph)

        # Quick path: heading paragraphs cannot host inline math/images/refs we care about.
        plain = _paragraph_text(paragraph._p).strip()
        level = _heading_level_from_style(style_name)
        if level is not None:
            if plain:
                self._handle_heading(plain, level, style_name)
            return

        # Caption-styled paragraph: drain pending images, attach text as caption.
        if self._unfilled_caption_figures and plain and caption_style:
            fig = self._unfilled_caption_figures.pop(0)
            fig.caption = plain
            fig.label = caption_label(plain, "figure")
            return

        # Walk the paragraph XML in document order to interleave:
        #   - w:t (text)
        #   - w:footnoteReference / w:endnoteReference (note IDs)
        #   - m:oMath (inline math → wrap in $...$)
        #   - m:oMathPara (display math → emit as separate deferred entry)
        #   - w:drawing (inline image → register as PaperFigure)
        text_buf: list[str] = []
        # The ``$…$`` each inline equation became, for the late clean-up.
        math_buf: list[str] = []
        # (kind, id) pairs — "footnote" and "endnote" ids collide otherwise.
        pending_note_refs: list[tuple[str, str]] = []
        had_image = False
        w = _NS["w"]
        m = _NS["m"]

        def flush_body() -> None:
            txt = "".join(text_buf).strip()
            text_buf.clear()
            inline_math = tuple(math_buf)
            math_buf.clear()
            if not txt:
                return
            # Footnote refs collected so far attach to this just-flushed entry.
            deferred_idx = self.assembler.append(
                txt, None, self._current_section_id, True, False, inline_math=inline_math
            )
            for kind, note_id in pending_note_refs:
                index = self._note_index.get((kind, note_id))
                if index is None:
                    source = self._footnotes_map if kind == "footnote" else self._endnotes_map
                    note_text = source.get(note_id, "").strip()
                    if not note_text:
                        continue
                    index = self._note_index[kind, note_id] = len(self._pending_footnotes)
                    self._pending_footnotes.append((note_text, kind))
                self._note_refs[index, deferred_idx] = None
            pending_note_refs.clear()

        def inline_math(math_el, extra_buf: list[str] | None) -> None:
            """Emit an inline equation as ``$…$`` text."""
            inline = _omml_to_text(math_el)
            if inline:
                math_buf.append(f"${inline}$")
                text_buf.append(math_buf[-1])
                if extra_buf is not None:
                    extra_buf.append(math_buf[-1])

        def walk_run(run_el, extra_buf: list[str] | None = None) -> None:
            """Pull text/footnotes/math from a single ``w:r`` run.

            If ``extra_buf`` is provided, text emitted into ``text_buf`` is
            mirrored into it — used so a ``w:hyperlink`` wrapper can capture
            its display text without re-implementing this walk.
            """
            nonlocal had_image

            def emit_separator() -> None:
                """Space for markup that ends a run of text, if not already spaced."""
                if text_buf and text_buf[-1] and not text_buf[-1][-1].isspace():
                    text_buf.append(" ")
                    if extra_buf is not None:
                        extra_buf.append(" ")

            for run_child in run_el.iterchildren():
                rt = run_child.tag
                if rt == f"{{{w}}}t" and run_child.text:
                    text_buf.append(run_child.text)
                    if extra_buf is not None:
                        extra_buf.append(run_child.text)
                elif rt in _RUN_SEPARATORS:
                    # A line break, tab or carriage return inside a run carries
                    # no text of its own, so dropping it fused the words on
                    # either side: a Shift+Enter title page collapsed to
                    # "Cognitive load and recallJane SmithDepartment of...".
                    # Word splits runs mid-word for formatting, so only these
                    # explicit separators may contribute whitespace — adjacent
                    # <w:t> must still concatenate untouched.
                    emit_separator()
                elif rt == f"{{{w}}}footnoteReference":
                    fn_id = run_child.get(f"{{{w}}}id")
                    if fn_id is not None:
                        pending_note_refs.append(("footnote", fn_id))
                elif rt == f"{{{w}}}endnoteReference":
                    # Endnote references were dropped on the floor, so a
                    # document using endnotes (the humanities convention for
                    # bibliographies) exported none of them.
                    en_id = run_child.get(f"{{{w}}}id")
                    if en_id is not None:
                        pending_note_refs.append(("endnote", en_id))
                elif rt == f"{{{m}}}oMath":
                    inline_math(run_child, extra_buf)
                elif rt == f"{{{w}}}drawing":
                    had_image = True
                    # A drawing can be a text box rather than a picture, and
                    # its paragraphs are real document text — pull quotes,
                    # boxed methods notes, poster-style layouts. Nothing walked
                    # into it, so that text was lost entirely. A text box in a
                    # text box is read with it: read again on its own, a nest
                    # 240 deep made 240 MB of text from 1 MB.
                    for txbx in run_child.iter(_W_TXBX_CONTENT):
                        if next(txbx.iterancestors(_W_TXBX_CONTENT), None) is not None:
                            continue
                        boxed = " ".join(
                            t.text.strip() for t in txbx.iter(f"{{{w}}}t") if t.text
                        ).strip()
                        if boxed:
                            text_buf.append(boxed)
                            if extra_buf is not None:
                                extra_buf.append(boxed)

        def walk_content(parent_el, into_buf: list[str] | None = None) -> None:
            """Walk a paragraph's inline content in document order: runs, math,
            hyperlinks, and the inline wrappers (``_INLINE_WRAPPERS``) at any
            depth, which hold the same content as the paragraph itself.

            ``into_buf``, if provided, mirrors the text emitted — used to
            capture hyperlink display text for ``PaperURLLink.link_text`` while
            still keeping the text inline in the paragraph stream.
            """
            for child in parent_el.iterchildren():
                tag = child.tag
                if tag == f"{{{m}}}oMathPara":
                    # Display math — flush any text so far, then emit math as own entry
                    flush_body()
                    math_text = _display_math_text(child)
                    if math_text:
                        self.assembler.append(
                            f"$${math_text}$$",
                            None,
                            self._current_section_id,
                            needs_segmentation=False,
                            is_formula=True,
                        )
                elif tag == f"{{{m}}}oMath":
                    inline_math(child, into_buf)
                elif tag == f"{{{w}}}r":
                    # Run: walk for w:t, note references, m:oMath, w:drawing
                    walk_run(child, into_buf)
                elif tag == f"{{{w}}}hyperlink":
                    walk_hyperlink(child, into_buf)
                elif tag in _INLINE_WRAPPERS:
                    # Tracked insertion, content control, field, smart tag...:
                    # walk its content as paragraph-level text.
                    walk_content(child, into_buf)

        def walk_hyperlink(link_el, into_buf: list[str] | None) -> None:
            """Capture a hyperlink's URL from its relationship target, then walk
            its content so the text contributes to paragraph flow."""
            rel_id = link_el.get(f"{{{_NS['r']}}}id")
            target_url: str | None = None
            if rel_id is not None:
                rel = self._doc.part.rels.get(rel_id)
                if rel is not None:
                    target_url = getattr(rel, "target_ref", None) or getattr(rel, "target", None)
            if not target_url or into_buf is not None:
                # A link without a URL is paragraph text, and a link inside a
                # link (Word never nests them) more of the outer link's text:
                # each level of a nest kept a copy of all the text under it,
                # 250 MB of link text from a 1 MB paragraph 250 deep.
                walk_content(link_el, into_buf)
                return
            link_text_buf: list[str] = []
            walk_content(link_el, link_text_buf)
            link_text = "".join(link_text_buf).strip() or target_url
            # text_id is unknown until segmentation; record the deferred
            # entry index that *will* hold this paragraph's text. Since
            # we have not yet flushed, the next deferred index is
            # ``len(self._deferred_texts)`` — apply_segmentation resolves
            # it once sentences exist.
            self._pending_url_links.append(
                (
                    target_url,
                    link_text,
                    self._current_section_id,
                    len(self.assembler),
                )
            )

        walk_content(paragraph._element)

        # Images: extract after the text walk so we know section context
        if had_image:
            for image_part in _image_parts(paragraph, self._doc):
                if len(self.figures) >= _MAX_FIGURES:
                    self._figures_dropped += 1
                    continue
                image_b64 = self._figure_image(image_part)
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
                self._unfilled_caption_figures.append(fig)
                self._figure_counter += 1

        flush_body()

    def _handle_heading(self, text: str, level: int, style_name: str | None) -> None:
        # Capture the document Title style (or first Heading 1) as the paper title
        if self._detected_title is None and style_name in (
            "Title",
            "Heading 1",
            "heading 1",
        ):
            self._detected_title = text

        # Deepen via numbering pattern when present (e.g. "2.3.1 Subsection" → 3)
        inferred = infer_level_from_numbering(text)
        if inferred is not None and inferred > level:
            level = inferred

        # The parent is the latest heading shallower than this one. Searching
        # back through every section for it was quadratic in a run of
        # same-level headings (20,000 Heading 1 paragraphs: 6 s), so the open
        # headings are kept as a stack, shallowest first.
        open_headings = self._open_headings
        while open_headings and open_headings[-1].level >= level:
            open_headings.pop()
        self._section_counter += 1
        section = PaperSection(
            section_id=self._section_counter,
            header=text,
            level=level,
            parent_section_id=open_headings[-1].section_id if open_headings else 0,
        )
        self.sections.append(section)
        open_headings.append(section)
        self._current_section_id = self._section_counter

    def _handle_math_para(self, omath_para_el) -> None:
        """Block-level OMML — emit as a non-segmented formula deferred entry."""
        math_text = _display_math_text(omath_para_el)
        if not math_text:
            return
        self.assembler.append(
            f"$${math_text}$$",
            None,
            self._current_section_id,
            needs_segmentation=False,
            is_formula=True,
        )

    def _table_cells_of(self, table) -> list[list[str]] | None:
        """A table's cell texts, one list per row padded to the widest row, or
        ``None`` when the table has no cells or is over the size limits."""
        rows = _table_rows(table._tbl)
        width = max((sum(columns for _, columns in row) for row in rows), default=0)
        if not width:
            return None
        size = (len(rows) + _COLUMN_CELLS) * width
        if size > _MAX_TABLE_CELLS or self._table_cells + size > _MAX_DOCUMENT_TABLE_CELLS:
            self._tables_dropped += 1
            return None

        # A merged cell's text is read once, however many grid cells it fills,
        # and charged at its rendered size in each of them.
        texts: dict[object, str] = {}
        costs: dict[object, int] = {}
        chars = 0
        char_width = 1
        for row in rows:
            for tc, columns in row:
                cost = costs.get(tc)
                if cost is None:
                    text = texts[tc] = "\n".join(_paragraph_text(p) for p in tc.p_lst).strip()
                    cost, char_bytes = _rendered_size(text)
                    costs[tc] = cost
                    char_width = max(char_width, char_bytes)
                chars += (cost + _CELL_MARKUP_CHARS) * columns
        html_bytes = chars * char_width
        if self._table_bytes + html_bytes > _MAX_DOCUMENT_TABLE_BYTES:
            self._tables_dropped += 1
            return None
        self._table_cells += size
        self._table_bytes += html_bytes

        cell_rows: list[list[str]] = []
        for row in rows:
            cells: list[str] = []
            for tc, columns in row:
                cells.extend([texts[tc]] * columns)
            # Pad ragged rows so DataFrame construction is uniform
            cells.extend([""] * (width - len(cells)))
            cell_rows.append(cells)
        return cell_rows

    def _handle_table(self, cell_rows: list[list[str]], *, caption: str | None = None) -> None:
        header_row = cell_rows[0]
        data_rows = cell_rows[1:] if len(cell_rows) > 1 else []
        df = pd.DataFrame(data_rows, columns=header_row)
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

    def _figure_image(self, image_part) -> str | None:
        """The base64 image for a new figure showing *image_part*, or ``None``
        once the figures' image data would pass the limit."""
        size = len(image_part.blob)
        if self._figure_image_bytes + size > _MAX_FIGURE_IMAGE_BYTES:
            self._figure_images_omitted += 1
            return None
        self._figure_image_bytes += size
        image_b64 = self._image_b64.get(image_part)
        if image_b64 is None:
            image_b64 = self._image_b64[image_part] = base64.b64encode(image_part.blob).decode(
                "ascii"
            )
        return image_b64

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
