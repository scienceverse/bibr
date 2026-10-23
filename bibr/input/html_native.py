"""Native HTML parser: article HTML/XHTML -> PaperContents.

This mirrors the native DOCX/JATS parser surface so the pipeline can parse
publisher HTML without OCR. It is intentionally structural rather than
browser-like: scripts/styles/navigation are discarded, no remote resources are
fetched, and body text is mapped into the same deferred sentence-segmentation
contract used by the other native inputs.
"""

from __future__ import annotations

import codecs
import itertools
import logging
import re
import unicodedata
from collections.abc import Iterator
from typing import Any

import pandas as pd
import webencodings  # html5lib's encoding table (a dependency of html5lib)
from bs4 import BeautifulSoup, CData, NavigableString, Tag

from bibr.input.heading_cues import (
    REFERENCES_HEADER,
    names_reference_list,
    reads_as_bold_heading,
    reads_as_reference_heading,
)
from bibr.input.mathml_whitespace import FlatText, mspace_separates
from bibr.models import PaperAuthor, PaperMetadata
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
)
from bibr.structure.assembler import DeferredText, DocumentAssembler
from bibr.structure.float_labels import caption_label
from bibr.structure.html_table import TableBudget, html_table_frame, is_hidden_table
from bibr.structure.xref_utils import URL_RE, detect_xrefs
from bibr.utils.text import YEARISH_RE, clean_extracted_url, collapse_ws

logger = logging.getLogger(__name__)

_DROP_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "nav",
    "aside",
    "form",
    "iframe",
    "svg",
}
_BLOCK_TEXT_TAGS = {"p", "blockquote", "pre"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Elements a browser lays out inline, MathML presentation markup included:
# their text continues the word around them. Every other element is a word
# boundary.
_INLINE_TAGS = frozenset(
    {
        "a",
        "abbr",
        "acronym",
        "b",
        "bdi",
        "bdo",
        "big",
        "cite",
        "code",
        "data",
        "del",
        "dfn",
        "em",
        "font",
        "i",
        "ins",
        "kbd",
        "label",
        "mark",
        "q",
        "s",
        "samp",
        "small",
        "span",
        "strike",
        "strong",
        "sub",
        "sup",
        "time",
        "tt",
        "u",
        "var",
        "wbr",
        "math",
        "menclose",
        "mfenced",
        "mfrac",
        "mi",
        "mmultiscripts",
        "mn",
        "mo",
        "mover",
        "mpadded",
        "mphantom",
        "mprescripts",
        "mroot",
        "mrow",
        "ms",
        "mspace",
        "msqrt",
        "mstyle",
        "msub",
        "msubsup",
        "msup",
        "mtext",
        "munder",
        "munderover",
        "semantics",
    }
)
# What a run of inline content between blocks can hold: the inline elements, a
# line break and an image. A browser lays such a run out as a paragraph of its
# own, whatever element it sits in.
_RUN_TAGS = _INLINE_TAGS | {"br", "img"}
# Document metadata, never page text: an ePub's chapters each bring a <head>,
# and html5lib keeps a <title> met inside <body> where it stands.
_NON_CONTENT_TAGS = frozenset({"head", "title", "meta", "link", "base"})
# Elements that set their text bold, and a style declaration that does.
_BOLD_TAGS = frozenset({"b", "strong"})
_BOLD_STYLE_RE = re.compile(r"font-weight\s*:\s*(?:bold(?:er)?|[6-9]00)\b", re.IGNORECASE)
# Class names of a reference list's entries in HTML that Word ("Save as Web
# Page") writes for its Bibliography and EndNote's EndNote Bibliography style.
_BIBLIOGRAPHY_CLASSES = frozenset({"msobibliography", "endnotebibliography"})
# Class and id names of a container holding a reference list and nothing
# else: pandoc and Quarto's citeproc output (<div id="refs" class="references
# csl-bib-body">), MediaWiki's and others. Whole names only: a substring such
# as "citation" also names a page's "Cite this article" tools.
_REFERENCE_CONTAINER_NAMES = frozenset(
    {"refs", "references", "bibliography", "csl-bib-body", "ref-list", "reflist"}
)
# Upper bound on HTML fed to the pure-Python html5lib parser (audit L9). Well
# above any real article/JATS/EPUB spine document, below what makes parsing a
# DoS. Kept below the serve upload cap so it fails fast on the parse path.
_MAX_HTML_BYTES = 48 * 1024 * 1024
# How much of a file inspect_html reads for markup before parsing it: far
# more than a doctype, a licence comment or inline styles ahead of the first
# element.
_SNIFF_BYTES = 1024 * 1024
_ELEMENT_RE = re.compile(
    r"<\s*(?:html|body|article|main|section|p|h[1-6]|table|figure|div|li|blockquote|ol|ul)\b",
    re.ASCII | re.IGNORECASE,
)
# Where html5lib looks for an encoding declaration, and what it finds there.
_PRESCAN_BYTES = 1024
_META_CHARSET_RE = re.compile(rb"<meta[^>]*charset", re.IGNORECASE)
_BOMS = (codecs.BOM_UTF8, codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)
# collapse_ws's whitespace rule: every run of whitespace becomes one space.
_WS_RUN_RE = re.compile(r"\s+")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+\b", re.IGNORECASE)
_DATE_RE = re.compile(r"(\d{4})(?:[-/](\d{1,2})(?:[-/](\d{1,2}))?)?")


def _tag_name(tag: Any) -> str:
    return (getattr(tag, "name", "") or "").lower()


def _flatten(tag: Tag) -> str:
    """Concatenate *tag*'s text the way a browser lays it out.

    ``get_text(" ")`` put a space around every element, so ``H<sub>2</sub>O``
    read "H 2 O", ``m<sup>6</sup>A`` "m 6 A" and a linked citation
    "( Figure 1 )". Inline elements now join their neighbours, as they do in
    the JATS parser; any other element still separates words. The strings
    kept are the ones ``get_text`` keeps (no comments).

    The walk keeps its own stack: html5lib does not bound nesting depth, and
    legacy markup such as unclosed ``<font>`` or ``<span>`` tags nests every
    later element one level deeper, past Python's recursion limit.

    Whitespace between MathML elements is dropped as a renderer drops it,
    except where it keeps two words apart (:mod:`bibr.input.mathml_whitespace`).
    """
    flat = FlatText()
    name = _tag_name(tag)
    serials = itertools.count()

    # One frame per open element: its remaining children, whether the element
    # separates words (a boundary goes in on entry and on exit), its name,
    # whether it sits inside ``<math>``, and the serial numbers of the element
    # and of its parent.
    stack: list[tuple[Iterator[Any], bool, str, bool, int, int]] = [
        (iter(tag.children), False, name, name == "math", next(serials), next(serials))
    ]
    while stack:
        children, separates, name, in_math, serial, parent = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            if separates:
                flat.separate()
        elif isinstance(child, Tag):
            child_name = _tag_name(child)
            separates = child_name not in _INLINE_TAGS
            if separates or (child_name == "mspace" and mspace_separates(child.attrs)):
                flat.separate()
            in_child_math = in_math or child_name == "math"
            stack.append(
                (iter(child.children), separates, child_name, in_child_math, next(serials), serial)
            )
        elif type(child) in (NavigableString, CData):
            if in_math:
                flat.add_math(str(child), name, parent)
            else:
                flat.add(str(child))
    return flat.join()


def _text(tag: Any) -> str:
    if tag is None:
        return ""
    text = collapse_ws(_flatten(tag)).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def _attr_tokens(tag: Tag, *names: str) -> str:
    values: list[str] = []
    for name in names:
        value = tag.get(name)
        if isinstance(value, list):
            values.extend(str(v) for v in value)
        elif value is not None:
            values.append(str(value))
    return " ".join(values).lower()


def _normal_meta_key(value: str | None) -> str:
    return (value or "").lower().replace(":", ".").replace("_", ".").strip()


def _split_keywords(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for item in re.split(r"[;,]", value):
            item = item.strip()
            if item:
                out.append(item)
    return out


def _parse_date(value: str | None) -> str | None:
    if not value:
        return None
    match = _DATE_RE.search(value.strip())
    if not match:
        return None
    year, month, day = match.groups()
    if month and day:
        return f"{year}-{int(month):02d}-{int(day):02d}"
    if month:
        return f"{year}-{int(month):02d}"
    return year


def _split_person_name(value: str) -> tuple[str, str]:
    value = collapse_ws(value).strip()
    if not value:
        return "", ""
    if "," in value:
        family, given = [part.strip() for part in value.split(",", 1)]
        return given, family
    parts = value.split()
    if len(parts) == 1:
        return "", parts[0]
    return " ".join(parts[:-1]), parts[-1]


def _map_heading(header: str) -> CanonicalSection:
    if reads_as_reference_heading(header):
        # "Literature Cited", "Reference list", "Literaturverzeichnis", ...
        return CanonicalSection.REFERENCES
    norm = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", header.lower()).strip()
    for canon, aliases in {
        CanonicalSection.ABSTRACT: {"abstract", "summary"},
        CanonicalSection.INTRODUCTION: {"introduction", "background", "intro"},
        CanonicalSection.METHODS: {"methods", "method", "materials and methods"},
        CanonicalSection.RESULTS: {"results", "findings"},
        CanonicalSection.DISCUSSION: {"discussion", "conclusion", "conclusions"},
        CanonicalSection.REFERENCES: {"references", "bibliography", "works cited"},
        CanonicalSection.ACKNOWLEDGMENT: {"acknowledgments", "acknowledgements"},
        CanonicalSection.FUNDING: {"funding"},
        CanonicalSection.KEYWORDS: {"keywords", "key words"},
        CanonicalSection.OPEN_DATA: {"data availability", "code availability"},
        CanonicalSection.COI: {"conflict of interest", "competing interests"},
        CanonicalSection.ETHICS: {"ethics", "ethical approval"},
        CanonicalSection.AUTHOR_CONTRIBUTIONS: {"author contributions"},
        CanonicalSection.APPENDIX: {"appendix", "supplementary material"},
    }.items():
        if norm in aliases:
            return canon
    return CanonicalSection.UNKNOWN


def _resolve_link_sentence(
    candidates,
    url: str,
    link_text: str,
    fallback_id: int | None,
    entry_text: str = "",
    anchor_offset: int | None = None,
):
    """Pick the sentence of one deferred entry that holds an anchor (as JATS)."""
    if anchor_offset is not None and entry_text:
        picked = _sentence_at_offset(candidates, entry_text, anchor_offset)
        if picked is not None:
            needle = (link_text or "").strip()
            if not needle or needle in picked.text or url in picked.text:
                return picked
    if link_text:
        needle = link_text.strip()
        if needle:
            for sent in candidates:
                if needle in sent.text:
                    return sent
    for sent in candidates:
        if url in sent.text:
            return sent
    if fallback_id is not None:
        for sent in candidates:
            if sent.text_id == fallback_id:
                return sent
    return candidates[-1] if candidates else None


def _sentence_at_offset(candidates, entry_text: str, offset: int):
    """Return the candidate sentence covering *offset* in *entry_text* (as JATS)."""
    if not candidates:
        return None
    chosen = candidates[0]
    cursor = 0
    for sent in candidates:
        found = entry_text.find(sent.text, cursor)
        start = found if found >= 0 else cursor
        if start <= offset:
            chosen = sent
        else:
            break
        cursor = start + len(sent.text)
    return chosen


def _nfc_cut(text: str, start: int) -> int:
    """Index of the last character of *text*, from *start* on, that NFC cannot
    join to what precedes it, or 0 when there is none.

    Normalising each side of that point apart gives the text normalised
    whole: the character starts with a starter, so canonical reordering stops
    there and what follows composes with it or later, and it does not compose
    with what precedes it. Every ASCII character qualifies.
    """
    nfc = unicodedata.normalize
    for at in range(len(text) - 1, max(start, 1) - 1, -1):
        char = text[at]
        if char < "\x80":
            return at
        if unicodedata.combining(nfc("NFD", char)[0]):
            continue
        head = text[:at]
        if nfc("NFC", head + char) == nfc("NFC", head) + nfc("NFC", char):
            return at
    return 0


def _without_inert_marks(text: str) -> tuple[str, int]:
    """*text* in NFD without the marks that cannot change how NFC joins it,
    and how many were left out.

    NFC composes at most three marks onto one character, so after a starter
    the fifth and later marks of a combining class never compose, and one of
    the first four stays to block whatever they would block: each adds one
    character to the NFC length wherever it stands.
    """
    kept: list[str] = []
    per_class: dict[int, int] = {}
    decomposed = unicodedata.normalize("NFD", text)
    for char in decomposed:
        ccc = unicodedata.combining(char)
        if not ccc:
            per_class.clear()
        elif per_class.get(ccc, 0) < 4:
            per_class[ccc] = per_class.get(ccc, 0) + 1
        else:
            continue
        kept.append(char)
    return "".join(kept), len(decomposed) - len(kept)


def _with_collapsed(state: tuple[int, bool, bool], text: str) -> tuple[int, bool, bool]:
    """*state* with *text* appended: the length of the text read so far with
    collapse_ws applied but not its strip, and whether it starts and ends
    with a space. The text must start where NFC cannot reach back."""
    piece = _WS_RUN_RE.sub(" ", unicodedata.normalize("NFC", text))
    if not piece:
        return state
    length, lead, trail = state
    if not length:
        lead = piece[0] == " "
    return length + len(piece) - (trail and piece[0] == " "), lead, piece[-1] == " "


class _CollapsedLength:
    """``len(collapse_ws(text))`` of text that grows piece by piece.

    The text before the last point NFC cannot reach across (:func:`_nfc_cut`)
    is collapsed once and kept as its length, so each measurement reads only
    the text after that point instead of the whole text again. Only a run of
    combining marks has no such point; past a few hundred characters its
    marks that NFC cannot join are counted and left out
    (:func:`_without_inert_marks`), so that text stays short too.
    """

    def __init__(self) -> None:
        self._done = (0, False, False)
        self._tail: list[str] = []
        # Tail positions below this were searched for a cut and hold none.
        self._searched = 0
        # Marks left out of the tail, and the tail length that prompts the
        # next leaving out.
        self._left_out = 0
        self._shorten_at = 256

    def add(self, text: str) -> None:
        self._tail.append(text)

    def measure(self) -> int:
        tail = "".join(self._tail)
        cut = _nfc_cut(tail, self._searched)
        if cut:
            length, lead, trail = _with_collapsed(self._done, tail[:cut])
            self._done = (length + self._left_out, lead, trail)
            tail = tail[cut:]
            self._left_out, self._shorten_at = 0, 256
        if len(tail) > self._shorten_at:
            tail, left_out = _without_inert_marks(tail)
            self._left_out += left_out
            self._shorten_at = 2 * len(tail) + 256
        self._tail = [tail]
        self._searched = len(tail)
        length, lead, trail = _with_collapsed(self._done, tail)
        length += self._left_out - lead
        return length - 1 if trail and length else length


def _anchor_offsets(block: Tag, anchors: list[tuple[Tag, str]]) -> list[int]:
    """Character offsets where each anchor's display text starts in the block
    text, for ``(anchor, display text)`` pairs.

    The strings before an anchor in document order are the anchor's prefix;
    block text is that accumulation collapsed, so the collapsed prefix
    length is the anchor's start, plus one separator space when the source
    spells whitespace on either side of it. One walk measures every prefix:
    collapsing each from the block start took time links x block size.
    """
    if not anchors:
        return []
    order = {id(anchor): i for i, (anchor, _) in enumerate(anchors)}
    offsets = [0] * len(anchors)
    prefix = _CollapsedLength()
    last = ""
    for node in block.descendants:
        i = order.get(id(node))
        if i is not None:
            anchor, display = anchors[i]
            offset = prefix.measure()
            if offset and display:
                first_inside = next(
                    (str(s) for s in anchor.descendants if isinstance(s, (NavigableString, CData))),
                    "",
                )
                if last.isspace() or first_inside[:1].isspace():
                    offset += 1
            offsets[i] = offset
        elif isinstance(node, (NavigableString, CData)) and (text := str(node)):
            prefix.add(text)
            last = text[-1]
    return offsets


def _block_holders(root: Tag) -> set[int]:
    """The ids of the inline elements under *root* that hold a block element,
    at any depth, found in one pass from the leaves up."""
    holders: set[int] = set()
    for tag in reversed(root.find_all(True)):
        for child in tag.children:
            if isinstance(child, Tag) and (
                _tag_name(child) not in _RUN_TAGS or id(child) in holders
            ):
                holders.add(id(tag))
                break
    return holders


def _bold_throughout(block: Tag) -> bool:
    """Whether every piece of text in *block* is bold: inside ``<b>`` or
    ``<strong>``, or under an element whose style sets a bold font weight
    (the block's own included)."""
    seen = False
    for string in block.find_all(string=True):
        if type(string) not in (NavigableString, CData) or not string.strip():
            continue
        seen = True
        element = string.parent
        while element is not None:
            if _tag_name(element) in _BOLD_TAGS or _BOLD_STYLE_RE.search(
                str(element.get("style") or "")
            ):
                break
            if element is block:
                return False
            element = element.parent
        else:
            return False
    return seen


class HtmlParser:
    """Parses article HTML/XHTML bytes into :class:`PaperContents`."""

    def __init__(
        self,
        html_bytes: bytes,
        *,
        metadata_overrides: dict[str, Any] | None = None,
        parsed_soup: BeautifulSoup | None = None,
    ):
        self.html_bytes = html_bytes
        self.metadata_overrides = metadata_overrides or {}
        self._parsed_soup = parsed_soup
        self.assembler = DocumentAssembler()

        self.sections: list[PaperSection] = []
        self.sentences: list[PaperSentence] = []
        self.links: list[PaperURLLink] = []
        self.tables: list[PaperTable] = []
        self.figures: list[PaperFigure] = []

        self._section_counter = 0
        self._sentence_counter = 1
        self._paragraph_counter = 0
        self._table_counter = 1
        self._figure_counter = 1
        self._current_section_id = 0
        # Each section's type by id, so _in_references is a lookup per block.
        self._section_types: dict[int, CanonicalSection] = {}
        self._heading_stack: list[tuple[int, int]] = [(0, 0)]
        self._detected_title: str | None = None
        self._metadata: PaperMetadata = PaperMetadata(doi="", title="")
        self._native_ref_strings: list[str] | None = None
        self._pending_url_links: list[tuple[str, str, int, int, int]] = []
        # Spans repeat a cell's text in every slot they cover: the tables of
        # one document (an ePub's chapters together) share one limit.
        self._table_budget = TableBudget()
        # The parsed document, the inline elements holding a block (read as
        # containers), and whether bold paragraphs are headings: all set when
        # parse() reads the document.
        self._soup: BeautifulSoup | None = None
        self._block_holders: set[int] = set()
        self._bold_headings = False
        # The section open before the references heading, and whether a
        # reference list (a list element, or paragraphs Word or EndNote marked
        # as entries) has been read under it.
        self._section_before_references = 0
        self._reference_list = False

    @property
    def _deferred_texts(self) -> list[tuple[str, int | None, int, bool, bool]]:
        """Legacy 5-tuple view of deferred text, matching DOCX/JATS."""
        return [
            (e.text, e.page_number, e.section_id, e.needs_segmentation, e.is_formula)
            for e in self.assembler.entries
        ]

    def parse(self) -> PaperContents:
        """Parse HTML structure and leave sentence segmentation deferred."""
        from bibr.exceptions import ProcessingError

        # html5lib is pure-Python and slow on large/pathological markup; bound the
        # input so a huge document can't tie up the parser (audit L9). Byte size
        # bounds node count, so this caps parse work proportionally. Only applies
        # when we parse here: a _parsed_soup comes from inspect_html, which
        # applies the same cap before parsing.
        if self._parsed_soup is None and len(self.html_bytes) > _MAX_HTML_BYTES:
            raise ProcessingError(
                f"HTML input exceeds the {_MAX_HTML_BYTES}-byte parse limit "
                f"({len(self.html_bytes)} bytes)"
            )

        try:
            soup = (
                self._parsed_soup if self._parsed_soup is not None else _parse_soup(self.html_bytes)
            )
        except Exception as exc:
            raise ProcessingError(f"Failed to parse HTML: {exc}") from exc

        self._remove_noise(soup)
        self._metadata = self._parse_metadata(soup)
        self._detected_title = self._metadata.title or self._first_heading_text(soup)
        if self._detected_title and not self._metadata.title:
            self._metadata.title = self._detected_title

        self.sections.append(
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None)
        )

        root = soup.find("article") or soup.find("main") or soup.body or soup
        self._soup = soup
        self._block_holders = _block_holders(root)
        # A page without heading elements (HTML a word processor wrote from a
        # manuscript whose headings were bolded by hand) has no other sign of
        # its sections.
        self._bold_headings = root.find(_HEADING_TAGS) is None
        self._process_children(root)

        if not self.assembler.entries and not self.tables and not self.figures:
            raise ProcessingError("HTML input did not contain parseable article content")

        return PaperContents(
            sentences=[],
            sections=self.sections,
            tables=self.tables,
            links=self.links,
            sections_text={},
            figures=self.figures,
            xrefs=[],
            detected_title=self._detected_title,
            preparsed_metadata=self._metadata,
            native_ref_strings=self._native_ref_strings,
            processing_warnings=self._table_budget.warnings(),
        )

    def _make_sentence(
        self,
        entry: DeferredText,
        text: str,
        text_id: int,
        paragraph_id: int,
    ) -> PaperSentence:
        return PaperSentence(
            text_id=text_id,
            text=text,
            section_id=entry.section_id,
            paragraph_id=paragraph_id,
            page_number=entry.page_number,
            from_ocr=False,
        )

    def apply_segmentation(self, contents: PaperContents, all_segments: list[list[str]]) -> None:
        """Populate sentences from externally-produced segment lists."""
        start_paragraph = self._paragraph_counter
        self.sentences, self._sentence_counter, self._paragraph_counter = self.assembler.emit(
            all_segments,
            sentence_factory=self._make_sentence,
            sentence_counter=self._sentence_counter,
            paragraph_counter=self._paragraph_counter,
        )

        by_paragraph: dict[int, list] = {}
        for sent in self.sentences:
            by_paragraph.setdefault(sent.paragraph_id, []).append(sent)
        covered: set[tuple[str, int]] = set()
        for url, link_text, section_id, deferred_index, anchor_offset in self._pending_url_links:
            if deferred_index >= len(self.assembler.last_text_id):
                continue
            fallback_id = self.assembler.last_text_id[deferred_index]
            if fallback_id is None:
                continue
            entry_para = start_paragraph + deferred_index + 1
            candidates = by_paragraph.get(entry_para, [])
            if not candidates:
                continue
            entry_text = ""
            if 0 <= deferred_index < len(self.assembler.entries):
                entry_text = self.assembler.entries[deferred_index].text
            sentence = _resolve_link_sentence(
                candidates, url, link_text or "", fallback_id, entry_text, anchor_offset
            )
            if sentence is None:
                continue
            text_id = sentence.text_id
            paragraph_id = sentence.paragraph_id
            self.links.append(
                PaperURLLink(
                    url=url,
                    section_id=section_id,
                    paragraph_id=paragraph_id,
                    text_id=text_id,
                    link_text=link_text or None,
                )
            )
            covered.add((url, text_id))

        for sent in self.sentences:
            self._detect_urls(sent, covered)

        contents.sentences = self.sentences
        contents.links = self.links
        contents.sections_text = DocumentAssembler.build_sections_text(self.sentences)
        contents.xrefs = detect_xrefs(self.sentences, self.tables, self.figures)

    def create_content_sections(self, contents: PaperContents) -> None:
        """Create synthetic figure/table sections like DOCX/JATS."""
        for fig in self.figures:
            fig._body_section_id = fig.section_id
            self._section_counter += 1
            contents.sections.append(
                PaperSection(
                    section_id=self._section_counter,
                    header=f"Figure {fig.figure_id}",
                    level=1,
                    parent_section_id=0,
                    section_type=CanonicalSection.FIGURE,
                    synthetic_kind="figure",
                )
            )
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

        for tbl in self.tables:
            tbl._body_section_id = tbl.section_id
            self._section_counter += 1
            contents.sections.append(
                PaperSection(
                    section_id=self._section_counter,
                    header=f"Table {tbl.table_id}",
                    level=1,
                    parent_section_id=0,
                    section_type=CanonicalSection.TABLE,
                    synthetic_kind="table",
                )
            )
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

    @staticmethod
    def _remove_noise(soup: BeautifulSoup) -> None:
        for tag in soup.find_all(_DROP_TAGS):
            tag.decompose()
        for tag in soup.find_all(attrs={"role": True}):
            role = str(tag.get("role") or "").lower()
            if role in {"navigation", "complementary", "banner", "contentinfo", "search"}:
                tag.decompose()

    def _parse_metadata(self, soup: BeautifulSoup) -> PaperMetadata:
        meta = PaperMetadata(doi="", title="")
        lookup: dict[str, list[str]] = {}
        for tag in soup.find_all("meta"):
            key = _normal_meta_key(tag.get("name") or tag.get("property") or tag.get("itemprop"))
            content = str(tag.get("content") or "").strip()
            if key and content:
                lookup.setdefault(key, []).append(content)

        def values(*keys: str) -> list[str]:
            out: list[str] = []
            for key in keys:
                out.extend(lookup.get(_normal_meta_key(key), []))
            return out

        def first(*keys: str) -> str:
            vals = values(*keys)
            return vals[0] if vals else ""

        title = first("citation_title", "dc.title", "og.title") or _text(soup.find("title"))
        meta.title = title

        doi = first("citation_doi", "dc.identifier", "dc.identifier.doi")
        match = _DOI_RE.search(doi)
        meta.doi = (match.group(0) if match else doi).removeprefix("doi:").strip()

        meta.abstract = first("citation_abstract", "dc.description", "description")
        meta.keywords = _split_keywords(values("citation_keywords", "keywords", "dc.subject"))
        meta.journal = first("citation_journal_title", "citation_journal_abbrev") or None
        meta.volume = first("citation_volume") or None
        meta.issue = first("citation_issue") or None
        meta.first_page = first("citation_firstpage", "citation_first_page") or None
        meta.last_page = first("citation_lastpage", "citation_last_page") or None
        meta.published = _parse_date(
            first("citation_publication_date", "article.published_time", "dc.date", "date")
        )
        meta.publisher = first("dc.publisher", "citation_publisher", "publisher") or None
        meta.license = first("dc.rights", "rights") or None
        meta.pmid = first("citation_pmid") or None
        meta.arxiv = first("citation_arxiv_id") or None
        html_tag = soup.find("html")
        meta.language = (
            first("citation_language", "dc.language")
            or (str(html_tag.get("lang") or "") if html_tag else "")
            or None
        )
        license_link = soup.find("link", rel=lambda rel: rel and "license" in rel)
        if not meta.license and license_link is not None:
            meta.license = str(license_link.get("href") or "").strip() or None

        authors = values("citation_author", "dc.creator", "author")
        meta.authors = []
        for idx, author in enumerate(authors, start=1):
            given, family = _split_person_name(author)
            if family or given:
                meta.authors.append(
                    PaperAuthor(
                        author_id=idx,
                        given=given,
                        family=family,
                        affiliation="",
                    )
                )

        overrides = self.metadata_overrides
        for field, value in overrides.items():
            if value not in (None, "", []):
                setattr(meta, field, value)
        return meta

    @staticmethod
    def _first_heading_text(soup: BeautifulSoup) -> str | None:
        heading = soup.find(_HEADING_TAGS)
        text = _text(heading)
        return text or None

    def _process_children(self, parent: Tag) -> None:
        # The walk keeps its own stack, as _flatten does: html5lib does not
        # bound nesting depth, and recursing per container hit Python's
        # recursion limit on deeply nested <div>s or unclosed <span>s.
        stack: list[Iterator[Any]] = [iter(self._blocks(parent))]
        while stack:
            child = next(stack[-1], None)
            if child is None:
                stack.pop()
                continue
            if not isinstance(child, Tag):
                continue
            name = _tag_name(child)
            if name in _NON_CONTENT_TAGS:
                continue
            if name in _HEADING_TAGS:
                self._handle_heading(child)
            elif name in _BLOCK_TEXT_TAGS:
                self._handle_text_block(child)
            elif name in {"ol", "ul"}:
                if self._in_references() or self._looks_like_reference_list(child):
                    self._handle_reference_items(child)
                else:
                    self._process_list(child)
            elif name == "li":
                if self._in_references():
                    self._append_reference(_text(child))
                else:
                    self._handle_text_block(child)
            elif name == "dl" and (self._in_references() or self._looks_like_reference_list(child)):
                self._handle_reference_definitions(child)
            elif name == "table":
                self._handle_table(child)
            elif name == "figure":
                self._handle_figure(child)
            elif self._is_reference_container(child):
                self._handle_reference_container(child)
            else:
                # Every other element (known containers and unknown tags alike)
                # may hold text deeper down — descend rather than drop it.
                stack.append(iter(self._blocks(child)))

    def _blocks(self, container: Tag) -> list[Any]:
        """*container*'s children, each run of inline content between them
        (text, inline elements, line breaks) wrapped in a ``<p>`` of its own.

        A browser lays out such a run as a paragraph whatever element holds
        it, but the walk only read text from blocks: the paragraphs of a page
        written as ``<div>text</div>`` (ScienceDirect's are), a definition
        list's entries and citeproc's ``<div class="csl-entry">`` references
        were all dropped. An inline element holding a block (``<a>`` around a
        ``<div>``) stays a container.
        """
        out: list[Any] = []
        run: list[Any] = []

        def flush() -> None:
            if any(
                (node.get_text() if isinstance(node, Tag) else str(node)).strip() for node in run
            ):
                assert self._soup is not None
                wrapper = self._soup.new_tag("p")
                run[0].insert_before(wrapper)
                for node in run:
                    wrapper.append(node.extract())
                out.append(wrapper)
            run.clear()

        for child in list(container.children):
            if isinstance(child, Tag):
                if _tag_name(child) in _RUN_TAGS and id(child) not in self._block_holders:
                    run.append(child)
                    continue
                flush()
                out.append(child)
            elif type(child) in (NavigableString, CData):
                run.append(child)
        flush()
        return out

    def _handle_heading(self, tag: Tag) -> None:
        header = _text(tag)
        if not header:
            return
        self._open_section(header, int(_tag_name(tag)[1]))

    def _open_section(self, header: str, level: int) -> None:
        if names_reference_list(header) and not self._in_references():
            self._section_before_references = self._current_section_id
        self._reference_list = False
        while self._heading_stack and self._heading_stack[-1][0] >= level:
            self._heading_stack.pop()
        parent_id = self._heading_stack[-1][1] if self._heading_stack else 0
        self._section_counter += 1
        section_id = self._section_counter
        section_type = _map_heading(header)
        if (
            self._detected_title
            and header == self._detected_title
            and section_type == CanonicalSection.UNKNOWN
        ):
            # A page without a title of its own takes its first heading as
            # one; a "References" or "Abstract" heading still heads its section.
            section_type = CanonicalSection.TITLE
        self.sections.append(
            PaperSection(
                section_id=section_id,
                header=header,
                level=level,
                parent_section_id=parent_id,
                section_type=section_type,
                classification_score=1.0 if section_type != CanonicalSection.UNKNOWN else 0.0,
                classification_source=(
                    "title" if section_type == CanonicalSection.TITLE else "exact_alias"
                )
                if section_type != CanonicalSection.UNKNOWN
                else None,
            )
        )
        self._section_types[section_id] = section_type
        self._heading_stack.append((level, section_id))
        self._current_section_id = section_id

    def _handle_text_block(self, tag: Tag) -> None:
        text = _text(tag)
        if not text:
            return
        if _tag_name(tag) == "p":
            if reads_as_reference_heading(text):
                # A paragraph reading "References" heads the list whatever its
                # markup; under a references heading already, it repeats it.
                if not self._in_references():
                    self._open_section(text, self._top_heading_level())
                return
            if self._bold_headings and reads_as_bold_heading(text) and _bold_throughout(tag):
                self._open_section(text, self._top_heading_level())
                return
            if _BIBLIOGRAPHY_CLASSES & set(_attr_tokens(tag, "class").split()):
                if not self._in_references():
                    self._ensure_references_section()
                self._reference_list = True
                self._append_reference(text)
                return
        # A dated paragraph after a reference list can be one more entry
        # (typed in by hand after the ones Word marked); anything else follows
        # the list rather than belonging to it.
        if self._in_references() and self._reference_list and not YEARISH_RE.search(text):
            self._end_reference_list()
        if self._in_references():
            self._append_reference(text)
            return
        deferred_index = self.assembler.append(text, None, self._current_section_id, True, False)
        links: list[tuple[Tag, str, str]] = []
        for a in tag.find_all("a", href=True):
            url = clean_extracted_url(str(a.get("href") or ""))
            if url:
                links.append((a, url, _text(a)))
        offsets = _anchor_offsets(tag, [(a, display) for a, _, display in links])
        for (_, url, display), offset in zip(links, offsets, strict=True):
            self._pending_url_links.append(
                (url, display, self._current_section_id, deferred_index, offset)
            )

    def _process_list(self, tag: Tag) -> None:
        for li in tag.find_all("li", recursive=False):
            self._handle_text_block(li)

    def _handle_reference_items(self, tag: Tag) -> None:
        if not self._in_references():
            self._ensure_references_section()
        for li in tag.find_all("li", recursive=False):
            self._append_reference(_text(li))
        self._reference_list = True

    def _handle_reference_definitions(self, tag: Tag) -> None:
        """A reference list written as a definition list: each ``<dd>``
        entry, after the label its ``<dt>`` prints ("[1]")"""
        if not self._in_references():
            self._ensure_references_section()
        label = ""
        for child in tag.find_all(["dt", "dd"], recursive=False):
            if _tag_name(child) == "dt":
                label = _text(child)
                continue
            self._append_reference(f"{label} {_text(child)}")
            label = ""
        if label:
            self._append_reference(label)
        self._reference_list = True

    def _is_reference_container(self, tag: Tag) -> bool:
        """Whether *tag* holds a reference list and nothing else, by its
        class or id (``_REFERENCE_CONTAINER_NAMES``)."""
        if _tag_name(tag) in _RUN_TAGS:
            return False
        names = set(_attr_tokens(tag, "id", "class").split())
        return bool(names & _REFERENCE_CONTAINER_NAMES)

    def _handle_reference_container(self, tag: Tag) -> None:
        """Each entry of a reference-list container: its lists' items, and
        its other blocks (citeproc's ``<div class="csl-entry">``) one each,
        without the headings inside it."""
        if not self._in_references():
            self._ensure_references_section()
        stack: list[Iterator[Any]] = [iter(self._blocks(tag))]
        while stack:
            child = next(stack[-1], None)
            if child is None:
                stack.pop()
                continue
            name = _tag_name(child)
            if name in _HEADING_TAGS or name in _NON_CONTENT_TAGS:
                continue
            if name in {"ol", "ul"}:
                for li in child.find_all("li", recursive=False):
                    self._append_reference(_text(li))
            elif name in {"p", "li", "dd", "dt"} or self._holds_only_runs(child):
                self._append_reference(_text(child))
            else:
                stack.append(iter(self._blocks(child)))
        self._reference_list = True

    def _holds_only_runs(self, tag: Tag) -> bool:
        """Whether *tag* holds text and inline elements only, no block."""
        return all(
            not isinstance(child, Tag)
            or (_tag_name(child) in _RUN_TAGS and id(child) not in self._block_holders)
            for child in tag.children
        )

    def _end_reference_list(self) -> None:
        """Close the reference list: what follows it under the same heading
        (a note, a licence, the page's tables) goes back to the section open
        before the references heading, so it is not read as references."""
        self._reference_list = False
        self._current_section_id = self._section_before_references
        while len(self._heading_stack) > 1 and self._heading_stack[-1][1] != (
            self._current_section_id
        ):
            self._heading_stack.pop()

    def _top_heading_level(self) -> int:
        """The level of a heading no element marks: that of the shallowest
        section heading so far other than the title, else 1."""
        levels = [
            section.level
            for section in self.sections[1:]
            if section.section_type != CanonicalSection.TITLE
        ]
        return min(levels, default=1)

    def _ensure_references_section(self) -> None:
        self._section_before_references = self._current_section_id
        self._reference_list = False
        self._section_counter += 1
        section_id = self._section_counter
        self.sections.append(
            PaperSection(
                section_id=section_id,
                header=REFERENCES_HEADER,
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.REFERENCES,
                classification_score=1.0,
                classification_source="exact_alias",
            )
        )
        self._section_types[section_id] = CanonicalSection.REFERENCES
        self._heading_stack = [(0, 0), (1, section_id)]
        self._current_section_id = section_id

    def _append_reference(self, text: str) -> None:
        text = collapse_ws(text).strip()
        if not text:
            return
        if self._native_ref_strings is None:
            self._native_ref_strings = []
        self._native_ref_strings.append(text)
        self.assembler.append(text, None, self._current_section_id, False, False)

    def _in_references(self) -> bool:
        return self._section_types.get(self._current_section_id) == CanonicalSection.REFERENCES

    @staticmethod
    def _looks_like_reference_list(tag: Tag) -> bool:
        tokens = _attr_tokens(tag, "id", "class", "role", "aria-label")
        return any(word in tokens for word in ("reference", "bibliography", "citation"))

    def _handle_table(self, tag: Tag) -> None:
        if is_hidden_table(tag):
            # A display:none table (a print-only or responsive duplicate of a
            # visible one) is not part of the page as read.
            return
        caption_tag = tag.find("caption")
        caption = _text(caption_tag) or None
        label = caption_label(caption, "table")
        try:
            df = html_table_frame(tag, self._table_budget)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HTML table parse failed: %s", exc)
            df = None
        if df is None:
            # No cell grid (an image-only table, say, or one over the table
            # limits, which the budget's warning records). A table whose caption
            # prints a table label ("Table 3. ...") is still a table that
            # mentions resolve to, so it is kept with its markup and no
            # contents. Any other grid-less table is dropped: a spacer, or a
            # layout table holding a figure ("Figure 1. ...").
            if label is None:
                return
            df = pd.DataFrame()
        html = str(tag)
        self.tables.append(
            PaperTable(
                table_id=self._table_counter,
                df=df,
                tbl_html=html,
                section_id=self._current_section_id,
                caption=caption,
                page_number=None,
                parts=[
                    PaperTablePart(
                        page_number=None,
                        bbox=None,
                        tbl_html=html,
                        df=df,
                    )
                ],
                label=label,
            )
        )
        self._table_counter += 1

    def _handle_figure(self, tag: Tag) -> None:
        caption = _text(tag.find("figcaption"))
        if not caption:
            img = tag.find("img")
            caption = str(img.get("alt") or "").strip() if img is not None else ""
        self.figures.append(
            PaperFigure(
                figure_id=self._figure_counter,
                section_id=self._current_section_id,
                image_b64=None,
                caption=caption or None,
                page_number=None,
                parts=[
                    PaperFigurePart(
                        page_number=None,
                        bbox=None,
                        image_b64=None,
                    )
                ],
                label=caption_label(caption, "figure"),
            )
        )
        self._figure_counter += 1

    def _detect_urls(self, sent: PaperSentence, covered: set[tuple[str, int]]) -> None:
        for match in URL_RE.finditer(sent.text):
            url = clean_extracted_url(match.group(0))
            if (url, sent.text_id) in covered:
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


def _parse_soup(html_bytes: bytes) -> BeautifulSoup:
    """Parse *html_bytes* with html5lib, reading undeclared UTF-8 as UTF-8.

    Without a byte-order mark or a ``<meta charset>`` in the first 1024 bytes
    (where html5lib looks for one) html5lib decodes as windows-1252, so a
    UTF-8 page saved without a declaration (pandoc's fragments, an XHTML file
    declaring its encoding only in the XML declaration, which HTML parsing
    ignores) read "Café" as "CafÃ©". Bytes that decode as UTF-8 are
    UTF-8 text in practice; anything else keeps html5lib's detection.
    """
    from_encoding = None
    if (
        not html_bytes.startswith(_BOMS)
        and _META_CHARSET_RE.search(html_bytes[:_PRESCAN_BYTES]) is None
    ):
        try:
            html_bytes.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            from_encoding = "utf-8"
    return BeautifulSoup(html_bytes, "html5lib", from_encoding=from_encoding)


def _reads_as_markup(html_bytes: bytes) -> bool:
    """Whether the start of *html_bytes* holds an HTML element tag.

    The bytes are decoded by their BOM when they have one, as html5lib
    decodes them (UTF-8, or UTF-16 as Word's "Save as Unicode" writes it),
    and otherwise as windows-1252: every other encoding html5lib reads,
    whatever a ``<meta charset>`` names, spells the ASCII tag names in ASCII
    bytes. A NUL ahead of the first tag means binary data that merely
    contains ``<p``, such as a gzipped page; the wider window would otherwise
    let those through.
    """
    text, _ = webencodings.decode(html_bytes[:_SNIFF_BYTES], "windows-1252", errors="replace")
    match = _ELEMENT_RE.search(text)
    return match is not None and "\x00" not in text[: match.start()]


def inspect_html(html_bytes: bytes) -> tuple[bool, BeautifulSoup | None]:
    """Return whether HTML has article text plus its reusable parsed DOM.

    Raises :class:`~bibr.exceptions.InputValidationError` for input over
    ``_MAX_HTML_BYTES`` before html5lib parses it: the pipeline hands the DOM
    returned here to :class:`HtmlParser`, so this is where the cap applies.
    """
    if len(html_bytes) > _MAX_HTML_BYTES:
        from bibr.exceptions import InputValidationError

        raise InputValidationError(
            f"HTML input exceeds the {_MAX_HTML_BYTES}-byte parse limit ({len(html_bytes)} bytes)"
        )
    if not _reads_as_markup(html_bytes):
        return False, None
    try:
        soup = _parse_soup(html_bytes)
    except Exception:
        return False, None
    HtmlParser._remove_noise(soup)
    root = soup.find("article") or soup.find("main") or soup.body or soup
    has_content = bool(collapse_ws(root.get_text(" ", strip=True)).strip())
    return has_content, soup if has_content else None
