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
from collections.abc import Iterator
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup, CData, NavigableString, Tag
from webencodings import lookup as _lookup_label  # type: ignore[import-untyped]

from bibr.input.mathml_whitespace import FlatText, mspace_separates
from bibr.models import PaperAuthor, PaperMetadata, canonicalize_orcid
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
from bibr.structure.html_table import html_table_frame, is_hidden_table
from bibr.structure.xref_utils import URL_RE, detect_xrefs
from bibr.utils.text import clean_extracted_url, collapse_ws, normalize_doi

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
    # Form controls and buttons are page chrome, never article prose.
    "button",
    "datalist",
    "input",
    "option",
    "select",
    "textarea",
}
# Screen-reader-only classes (the visuallyhidden spans around viewer links
# and download descriptions) and hidden subtrees carry no visible prose.
_NOISE_CLASS_TOKENS = frozenset({"visuallyhidden", "visually-hidden", "sr-only"})
# Paragraph, list-item and heading text was read in full before the walker
# recursed, hidden spans included. Dropping them there shifted sentence
# boundaries in author-detail lists (the "This ORCID iD identifies the author"
# label beside each ORCID link) far enough that the lexical statement capture
# ran into the next author's name and affiliation, so hidden elements inside
# these blocks are still read; only loose container text drops them.
_HIDDEN_KEPT_INSIDE = frozenset(
    {"p", "blockquote", "pre", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
)
# Replaced/void elements that sit inside a running sentence (an inline
# image, a form value, an embedded object): they carry no block structure,
# so they buffer inline instead of splitting the sentence around them.
_VOID_INLINE_TAGS = frozenset(
    {"img", "input", "output", "object", "video", "audio", "canvas", "picture", "embed"}
)
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
# Tags whose presence inside a container means the container holds block
# structure and must be recursed into rather than read as one text block:
# headings, text blocks, lists, floats, rules, and nested sectioning
# containers (publisher HTML wraps each paragraph in its own div).
_BLOCK_STRUCTURE_TAGS = frozenset(
    {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "blockquote",
        "pre",
        "ol",
        "ul",
        "li",
        "dl",
        "table",
        "figure",
        "hr",
        "div",
        "section",
        "article",
        "main",
        "header",
        "footer",
        "center",
        "hgroup",
    }
)
_META_CHARSET_RE = re.compile(rb"<meta[^>]+charset\s*=\s*[\"']?\s*([A-Za-z0-9._-]+)", re.IGNORECASE)
_XML_ENCODING_RE = re.compile(rb"<\?xml[^>]+encoding\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
# A site suffix on <title> ("Title | Journal", "Title - Site"): the separator
# must be followed by more text, so "A-B" or "Methods: a study" never strip.
_TITLE_SUFFIX_RE = re.compile(r"^\s*[|\-–—·»]\s+\S")


def _decode_html_bytes(data: bytes) -> str:
    """Decode raw HTML/XHTML bytes to text, honouring their charset.

    html5lib sniffs only the BOM and a meta charset, ignoring the XML
    declaration, and without chardet falls back to windows-1252 — so UTF-8
    without a meta decodes as mojibake. Decode first: BOM, then strict
    UTF-8 (so an undeclared UTF-8 page reads correctly), then a meta or
    XML-declared charset resolved through the WHATWG label table (so
    ``iso-8859-1`` reads as windows-1252, as browsers do, and a stray
    ``utf-16`` label on 8-bit bytes reads as UTF-8 instead of CJK
    garbage), finally windows-1252 with replacement, the html5lib and
    browser default for an undeclared legacy page.
    """
    if data.startswith(codecs.BOM_UTF8):
        return data.decode("utf-8-sig")
    if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return data.decode("utf-32")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    head = bytes(data[:4096])
    match = _META_CHARSET_RE.search(head) or _XML_ENCODING_RE.search(head)
    if match:
        label = match.group(1).decode("ascii", errors="ignore").strip().lower()
        if label in {"utf-16", "utf-16le", "utf-16be", "utf16", "utf_16"}:
            # WHATWG: a utf-16 meta label without a BOM means UTF-8.
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                pass
        else:
            try:
                encoding = _lookup_label(label)
            except Exception:  # noqa: BLE001 — unknown label reads as legacy below
                encoding = None
            if encoding is not None:
                try:
                    return data.decode(encoding.name)
                except (LookupError, UnicodeDecodeError):
                    pass
    return data.decode("windows-1252", errors="replace")


# Upper bound on HTML fed to the pure-Python html5lib parser (audit L9). Well
# above any real article/JATS/EPUB spine document, below what makes parsing a
# DoS. Kept below the serve upload cap so it fails fast on the parse path.
_MAX_HTML_BYTES = 48 * 1024 * 1024
_DATE_RE = re.compile(r"(\d{4})(?:[-/](\d{1,2})(?:[-/](\d{1,2}))?)?")


def _tag_name(tag: Any) -> str:
    return (getattr(tag, "name", "") or "").lower()


def _in_text_block(tag: Tag) -> bool:
    """True when *tag* is, or sits inside, a paragraph, list item or heading."""
    if _tag_name(tag) in _HIDDEN_KEPT_INSIDE:
        return True
    return any(_tag_name(parent) in _HIDDEN_KEPT_INSIDE for parent in tag.parents)


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


def _split_generic_author(value: str) -> list[str]:
    """Split a generic ``<meta name=\"author\">`` value into individual names.

    Publishers write these as ``'Jane Smith; John Doe'`` or ``'Jane Smith,
    John Doe'``, while a single ``'Family, Given'`` name uses the comma the
    other way round. A comma splits two names only when both sides read as
    full names (two or more tokens each); otherwise ``_split_person_name``
    keeps ``'Family, Given'`` intact.
    """
    names: list[str] = []
    for piece in value.split(";"):
        piece = piece.strip()
        if not piece:
            continue
        if "," in piece:
            parts = [part.strip() for part in piece.split(",")]
            nonempty = [part for part in parts if part]
            if len(nonempty) > 1 and all(len(part.split()) > 1 for part in nonempty):
                names.extend(nonempty)
                continue
        names.append(piece)
    return names


def _has_trustworthy_front_matter(soup: BeautifulSoup, metadata: PaperMetadata) -> bool:
    """True when meta tags carry structured article identity, not just SEO.

    A ``citation_title`` or ``dc.title`` together with authors, a valid DOI,
    a PMID or an arXiv id covers Highwire, Dublin Core (eLife pages carry
    ``dc.title`` plus a DOI and no ``citation_*`` tags) and ePub OPF records
    (title plus DOI but no creator). Anything else — a bare ``<title>``/``og``
    title, a generic description, a byline alone — defers to the printed
    article instead of locking in site chrome.
    """
    seen_structured_title = False
    for tag in soup.find_all("meta"):
        key = _normal_meta_key(str(tag.get("name") or tag.get("property") or ""))
        content = str(tag.get("content") or "").strip()
        if not content:
            continue
        if key in ("citation.title", "dc.title"):
            seen_structured_title = True
            break
    if not seen_structured_title:
        return False
    return bool(metadata.authors or metadata.doi or metadata.pmid or metadata.arxiv)


def _has_non_link_text(text: str, pending_links: list[tuple[str, str]]) -> bool:
    """True when *text* holds words beyond the buffered hyperlink texts."""
    rest = text
    for _, link_text in pending_links:
        norm = collapse_ws(link_text).strip()
        if norm:
            rest = rest.replace(norm, "", 1)
    return bool(rest.strip())


def _has_direct_inline_content(tag: Tag) -> bool:
    """True when *tag* mixes direct prose with nested block markup.

    A reference ``<div>`` holding author/year spans next to title/source
    ``<div>`` parts is one reference to flatten, while a wrapper holding
    only nested ``<div>`` references recurses into one entry per child.
    Whitespace-only strings do not count as prose.
    """
    for child in tag.children:
        if type(child) in (NavigableString, CData):
            if str(child).strip():
                return True
        elif isinstance(child, Tag):
            child_name = _tag_name(child)
            if child_name in _INLINE_TAGS or child_name in _VOID_INLINE_TAGS:
                return True
    return False


_REFERENCE_SPLIT_TAGS = ("p", "li", "ol", "ul", "table", "figure", "blockquote", "pre", "hr")

# A float's label printed on its own, outside any caption: the header strip
# publisher pages put above a figure ("Figure 3", "Table 1", "Video 2",
# "Author response image 1", "Appendix 1—figure 2", "Key resources table"),
# optionally with a supplements toggle ("Figure 3 with 2 supplements see
# all"). As loose container text it is furniture, never prose.
_FLOAT_WORD = (
    r"(?:figure|fig\.?|table|video|movie|image|box|scheme|chart|audio|animation"
    r"|chemical\s+structure|key\s+resources?\s+table|source\s+(?:data|code)"
    r"|supplementary\s+file|figure\s+supplement)"
)
_FLOAT_LABEL_ONLY_RE = re.compile(
    rf"^(?:(?:appendix|supplementary|supplemental|author\s+response)\s*[A-Z]?\d*\s*[—–-]?\s*)?"
    rf"{_FLOAT_WORD}\.?(?:\s*[A-Z]?\d+(?:\.\d+)*[a-z]?)?"
    rf"(?:\s*[—–-]\s*{_FLOAT_WORD}(?:\s*\d+)?)*"
    r"[.:]?(?:\s+with\s+\d+\s+supplements?)?(?:\s+see\s+all)?$",
    re.IGNORECASE,
)
# A lone bracketed year: the date fragment of a formatted citation block.
_BARE_YEAR_RE = re.compile(r"^\(\s*\d{4}[a-z]?\s*\)[.,]?$")


def _is_page_furniture(text: str) -> bool:
    """True when loose container *text* is a float label or a lone year.

    Only text that base never read (direct container strings and inline-only
    divs/spans) passes through this filter; ``<p>``, list-item and heading
    text is untouched.
    """
    text = text.strip()
    return bool(_FLOAT_LABEL_ONLY_RE.match(text) or _BARE_YEAR_RE.match(text))


_DOWNLOAD_LINK_RE = re.compile(r"^download\b", re.IGNORECASE)


def _is_navigation_list(tag: Tag, heading_texts: frozenset[str]) -> bool:
    """True for a list of bare links that navigates the page, not prose.

    Every non-empty item must be link text only, and the links must be
    buttons or download links (``<a class="button">Download BibTeX</a>``,
    "Download .RIS"), or at least half of the items must jump to an anchor
    on the same page under the text of one of the page's headings (a table
    of contents such as "Abstract / Introduction / Methods"). A list of
    author-name links or external resource links is not navigation.
    """
    items = [li for li in tag.find_all("li", recursive=False) if _text(li)]
    if not items:
        return False
    jumps = 0
    buttons = 0
    for item in items:
        anchors = item.find_all("a", href=True)
        if not anchors:
            return False
        rest = _text(item)
        for anchor in anchors:
            rest = rest.replace(_text(anchor), "", 1)
        if re.sub(r"[\W_]+", "", rest):
            return False
        hrefs = [str(anchor.get("href") or "") for anchor in anchors]
        if (
            all(href.startswith("#") and len(href) > 1 for href in hrefs)
            and _text(item).lower() in heading_texts
        ):
            jumps += 1
        classes = [
            str(token).lower() for anchor in anchors for token in (anchor.get("class") or [])
        ]
        if "button" in classes or _DOWNLOAD_LINK_RE.match(_text(item)):
            buttons += 1
    return buttons == len(items) or jumps * 2 >= len(items)


_CITATION_BLOCK_TOKENS = frozenset({"reference", "citation"})
_CITATION_PROSE_TAGS = (
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "table",
    "figure",
    "blockquote",
    "pre",
)
_BASE_REFERENCE_LIST_RE = re.compile(r"reference|bibliography|citation")


def _is_citation_block(tag: Tag) -> bool:
    """True for a formatted "cite this article" block outside the references.

    Publisher pages print the article's own citation as a ``div.reference``
    (or ``.citation``) of author, year, title, source and DOI parts. Outside
    a references section it is page furniture: read as text it gave one
    fragment per part ("(2021)", "eLife 10:e55070."), and its author list
    used to become one reference string per author. It is skipped only when
    it holds nothing read as body text before — no paragraph, heading,
    table, figure or quote, and list items only in reference-named lists.
    """
    classes: Any = tag.get("class") or []
    tokens = {str(token).lower() for token in (classes if isinstance(classes, list) else [classes])}
    if not tokens & _CITATION_BLOCK_TOKENS:
        return False
    if tag.find(_CITATION_PROSE_TAGS) is not None:
        return False
    for item in tag.find_all("li"):
        holder = item.parent
        if not isinstance(holder, Tag) or not _BASE_REFERENCE_LIST_RE.search(
            _attr_tokens(holder, "id", "class", "role", "aria-label")
        ):
            return False
    return True


def _map_heading(header: str) -> CanonicalSection:
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
        self._heading_stack: list[tuple[int, int]] = [(0, 0)]
        self._detected_title: str | None = None
        self._metadata: PaperMetadata = PaperMetadata(doi="", title="")
        self._native_ref_strings: list[str] | None = None
        self._pending_url_links: list[tuple[str, str, int, int]] = []
        # Lower-cased text of every heading on the page, for telling a table
        # of contents from a list of links.
        self._heading_texts: frozenset[str] = frozenset()

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
        # when we parse here (a caller-supplied _parsed_soup is already bounded).
        if self._parsed_soup is None and len(self.html_bytes) > _MAX_HTML_BYTES:
            raise ProcessingError(
                f"HTML input exceeds the {_MAX_HTML_BYTES}-byte parse limit "
                f"({len(self.html_bytes)} bytes)"
            )

        try:
            soup = (
                self._parsed_soup
                if self._parsed_soup is not None
                else BeautifulSoup(_decode_html_bytes(self.html_bytes), "html5lib")
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
        self._heading_texts = frozenset(
            _text(heading).lower() for heading in soup.find_all(list(_HEADING_TAGS))
        )
        self._process_children(root)

        if not self.assembler.entries and not self.tables and not self.figures:
            raise ProcessingError("HTML input did not contain parseable article content")

        # Generic SEO meta is not front matter: without a structured title and
        # article identity the record would lock in the <title> suffix, the
        # site description and a garbled byline, and an LLM run would skip
        # the front-matter pass that reads the printed article. The record is
        # still returned, marked untrusted: a no-LLM run keeps it (its
        # language, keywords and licence are right), an LLM run extracts the
        # front matter and only fills fields left empty from it.
        trusted = _has_trustworthy_front_matter(soup, self._metadata)
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
            preparsed_metadata_trusted=trusted,
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
        self.sentences, self._sentence_counter, self._paragraph_counter = self.assembler.emit(
            all_segments,
            sentence_factory=self._make_sentence,
            sentence_counter=self._sentence_counter,
            paragraph_counter=self._paragraph_counter,
        )

        covered: set[tuple[str, int]] = set()
        for url, link_text, section_id, deferred_index in self._pending_url_links:
            if deferred_index >= len(self.assembler.last_text_id):
                continue
            text_id = self.assembler.last_text_id[deferred_index]
            if text_id is None:
                continue
            sentence = next((s for s in self.sentences if s.text_id == text_id), None)
            paragraph_id = sentence.paragraph_id if sentence is not None else 0
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
        # find_all returns descendants of a matched tag after the tag itself;
        # once the ancestor is decomposed they are dead (attrs None), so every
        # pass skips tags already removed with an ancestor.
        for tag in soup.find_all(_DROP_TAGS):
            if not tag.decomposed:
                tag.decompose()
        for tag in soup.find_all(role=True):
            if tag.decomposed:
                continue
            role = str(tag.get("role") or "").lower()
            if role in {"navigation", "complementary", "banner", "contentinfo", "search"}:
                tag.decompose()
        for tag in soup.find_all(True):
            if tag.decomposed:
                continue
            if str(tag.get("aria-hidden") or "").lower() == "true" and not _in_text_block(tag):
                tag.decompose()
        for tag in soup.find_all(class_=True):
            if tag.decomposed:
                continue
            classes: list[str] = [str(token) for token in (tag.get("class") or [])]
            tokens = {token.lower() for token in classes}
            if tokens & _NOISE_CLASS_TOKENS and not _in_text_block(tag):
                tag.decompose()
        # A spine chapter's <head><title> lands in the body of the combined
        # ePub document and would read as a paragraph; the <head> title the
        # metadata fallback uses is left alone.
        for tag in soup.find_all("title"):
            if not tag.decomposed and tag.find_parent("head") is None:
                tag.decompose()

    def _parse_metadata(self, soup: BeautifulSoup) -> PaperMetadata:
        meta = PaperMetadata(doi="", title="")
        lookup: dict[str, list[str]] = {}
        for tag in soup.find_all("meta"):
            key = _normal_meta_key(
                str(tag.get("name") or tag.get("property") or tag.get("itemprop") or "")
            )
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

        title = first("citation_title", "dc.title", "og.title")
        if not title:
            # The <title> fallback carries the site chrome ("Title | Journal
            # | Press"); drop the suffix when the first h1 is its prefix.
            title = _text(soup.find("title"))
            heading = _text(soup.find("h1"))
            if (
                title
                and heading
                and len(heading) < len(title)
                and title.startswith(heading)
                and _TITLE_SUFFIX_RE.match(title[len(heading) :])
            ):
                title = heading
        meta.title = title

        doi = first("citation_doi", "dc.identifier", "dc.identifier.doi")
        meta.doi = normalize_doi(doi) or ""

        # The generic SEO "description" is not the paper's abstract when the
        # page prints one: it is the site's blurb or an impact statement, and
        # it used to win over the printed ABSTRACT section. It stays the last
        # resort (below) only for structured front matter on a page with no
        # Abstract heading, where publisher pages put the article's own
        # standfirst there (an eLife editorial's JATS abstract is exactly its
        # description).
        meta.abstract = first("citation_abstract", "dc.description")
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

        # Authors come from the first source that has any — merging
        # citation_author with dc.creator doubled every author. Each
        # citation_author_institution, _email and _orcid follows its own
        # citation_author in document order (Highwire), and one author can
        # carry several institutions — so the tags are walked in order and
        # each one attaches to the most recent author, instead of zipping
        # the flattened lists by position.
        meta.authors = []
        citation_authors = values("citation_author")
        dc_authors = values("dc.creator")
        if citation_authors:
            pending: list[dict[str, Any]] = []
            for tag in soup.find_all("meta"):
                key = _normal_meta_key(
                    str(tag.get("name") or tag.get("property") or tag.get("itemprop") or "")
                )
                content = str(tag.get("content") or "").strip()
                if not content:
                    continue
                if key == "citation.author":
                    pending.append(
                        {"name": content, "affiliations": [], "email": None, "orcid": None}
                    )
                elif pending and key == "citation.author.institution":
                    pending[-1]["affiliations"].append(content)
                elif pending and key == "citation.author.email" and not pending[-1]["email"]:
                    pending[-1]["email"] = content
                elif pending and key == "citation.author.orcid" and not pending[-1]["orcid"]:
                    pending[-1]["orcid"] = canonicalize_orcid(content)
            for idx, entry in enumerate(pending, start=1):
                given, family = _split_person_name(entry["name"])
                if not (family or given):
                    continue
                meta.authors.append(
                    PaperAuthor(
                        author_id=idx,
                        given=given,
                        family=family,
                        affiliation="; ".join(entry["affiliations"]),
                        email=entry["email"],
                        orcid=entry["orcid"],
                    )
                )
        else:
            names: list[str] = []
            if dc_authors:
                names = [name for name in dc_authors if name.strip()]
            else:
                for value in values("author"):
                    names.extend(_split_generic_author(value))
            for idx, name in enumerate(names, start=1):
                given, family = _split_person_name(name)
                if family or given:
                    meta.authors.append(
                        PaperAuthor(
                            author_id=idx,
                            given=given,
                            family=family,
                            affiliation="",
                        )
                    )

        if (
            not meta.abstract
            and _has_trustworthy_front_matter(soup, meta)
            and not any(
                _map_heading(_text(heading)) == CanonicalSection.ABSTRACT
                for heading in soup.find_all(list(_HEADING_TAGS))
            )
        ):
            meta.abstract = first("description")

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

    @staticmethod
    def _has_block_structure(tag: Tag) -> bool:
        """True when *tag* holds block-level markup deeper down.

        A container with direct text or only inline children (publisher
        ``<div class=\"para\">`` paragraphs, ePub chapter ``<section>`` text)
        reads as one text block; a container holding headings, paragraphs,
        lists, floats or nested sectioning containers is recursed into.
        """
        return any(
            isinstance(desc, Tag) and _tag_name(desc) in _BLOCK_STRUCTURE_TAGS
            for desc in tag.descendants
        )

    def _flush_pending_text(self, parts: list[str], links: list[tuple[str, str]]) -> None:
        """Emit text buffered at the current level as one deferred entry."""
        text = re.sub(r"\s+([,.;:!?])", r"\1", collapse_ws("".join(parts))).strip()
        parts.clear()
        pending = links.copy()
        links.clear()
        if not text:
            return
        if pending and not _has_non_link_text(text, pending):
            # A container holding nothing but links (a section-header
            # protocol link, a bare DOI anchor) is navigation chrome, not
            # prose — base dropped it the same way.
            return
        if _is_page_furniture(text):
            return
        if self._in_references():
            if not any(char.isdigit() for char in text):
                # Loose text in a references section with no year, volume or
                # page is a lead-in ("The following previously published
                # data sets were used"), not a reference; base never read it.
                return
            self._append_reference(text)
            return
        deferred_index = self.assembler.append(text, None, self._current_section_id, True, False)
        for url, link_text in pending:
            self._pending_url_links.append(
                (url, link_text, self._current_section_id, deferred_index)
            )

    def _buffer_inline(self, tag: Tag, parts: list[str], links: list[tuple[str, str]]) -> None:
        """Buffer an inline element's text and hyperlinks at the current level."""
        parts.append(_flatten(tag))
        anchors = list(tag.find_all("a", href=True))
        if _tag_name(tag) == "a" and tag.get("href"):
            # find_all covers descendants only — the anchor itself comes first.
            anchors.insert(0, tag)
        for anchor in anchors:
            url = clean_extracted_url(str(anchor.get("href") or ""))
            if url:
                links.append((url, _text(anchor)))

    def _process_children(self, parent: Tag, *, _skip_chrome: bool = False) -> None:
        # Direct strings and inline elements at this level form their own
        # paragraph ("Bare section text…" between a heading and a <p>); a
        # block child flushes them first so document order is kept.
        # Inside <header>/<footer> only headings and block prose are read —
        # section-header links ("Request a detailed protocol") and date/DOI
        # furniture are chrome that base dropped the same way.
        parts: list[str] = []
        links: list[tuple[str, str]] = []
        for child in parent.children:
            if type(child) in (NavigableString, CData):
                if _skip_chrome:
                    continue
                if str(child).strip():
                    parts.append(str(child))
                elif str(child):
                    # A whitespace-only string between inline siblings is the
                    # word boundary ("Smith <i>J</i> Doe"); collapse_ws folds
                    # runs, and a lone boundary flushes to nothing.
                    parts.append(" ")
                continue
            if not isinstance(child, Tag):
                continue
            name = _tag_name(child)
            if name in _HEADING_TAGS:
                self._flush_pending_text(parts, links)
                self._handle_heading(child)
            elif name in _BLOCK_TEXT_TAGS:
                self._flush_pending_text(parts, links)
                self._handle_text_block(child)
            elif name in {"ol", "ul"}:
                self._flush_pending_text(parts, links)
                if self._in_references() or self._looks_like_reference_list(child):
                    self._handle_reference_items(child)
                elif not _is_navigation_list(child, self._heading_texts):
                    self._process_list(child)
            elif name == "li":
                self._flush_pending_text(parts, links)
                if self._in_references():
                    self._append_reference(_text(child))
                else:
                    self._handle_text_block(child)
            elif name == "table":
                self._flush_pending_text(parts, links)
                self._handle_table(child)
            elif name == "figure":
                self._flush_pending_text(parts, links)
                self._handle_figure(child)
            elif name == "hr":
                self._flush_pending_text(parts, links)
            elif name in {"header", "footer"}:
                self._flush_pending_text(parts, links)
                self._process_children(child, _skip_chrome=True)
            elif name in _INLINE_TAGS:
                if _skip_chrome and not self._has_block_structure(child):
                    continue
                if self._has_block_structure(child):
                    # Invalid nesting (an unclosed <b>/<font>/<span> higher up
                    # nests whole sections inside this inline element) — the
                    # block structure wins and is recursed into, not flattened.
                    self._flush_pending_text(parts, links)
                    self._process_children(child, _skip_chrome=_skip_chrome)
                else:
                    self._buffer_inline(child, parts, links)
            elif name == "br":
                if not _skip_chrome:
                    parts.append(" ")
            elif name in _VOID_INLINE_TAGS and not self._has_block_structure(child):
                # A replaced element inside a running sentence (an inline
                # image, an embedded object) stays inline instead of splitting
                # the sentence. An image is a word boundary only, as in <p>
                # text: its alt text is mostly icon chrome ("Is a
                # corresponding author", "ORCID icon").
                if not _skip_chrome:
                    if name == "img":
                        parts.append(" ")
                    else:
                        flat = _flatten(child)
                        if flat.strip():
                            parts.append(flat)
            elif not self._in_references() and _is_citation_block(child):
                # The article's own formatted citation: furniture.
                self._flush_pending_text(parts, links)
            elif self._has_block_structure(child):
                # A container holding block markup — recurse rather than
                # flatten it into one entry.
                self._flush_pending_text(parts, links)
                if (
                    self._in_references()
                    and name == "div"
                    and _has_direct_inline_content(child)
                    and child.find([*_REFERENCE_SPLIT_TAGS, *_HEADING_TAGS]) is None
                ):
                    # One structured reference (author/year spans next to
                    # title/source divs): flatten it into a single reference
                    # string instead of one per inner div.
                    self._append_reference(_text(child))
                else:
                    self._process_children(child, _skip_chrome=_skip_chrome)
            else:
                if _skip_chrome:
                    continue
                # A container with direct text or only inline children: a
                # div/section/span paragraph reads as one text block. It goes
                # through the same buffer as loose inline text, so a
                # link-only container (a download link) and a stand-alone
                # float label stay out of the body like any other furniture.
                self._flush_pending_text(parts, links)
                self._buffer_inline(child, parts, links)
                self._flush_pending_text(parts, links)
        self._flush_pending_text(parts, links)

    def _handle_heading(self, tag: Tag) -> None:
        header = _text(tag)
        if not header:
            return
        level = int(_tag_name(tag)[1])
        while self._heading_stack and self._heading_stack[-1][0] >= level:
            self._heading_stack.pop()
        parent_id = self._heading_stack[-1][1] if self._heading_stack else 0
        self._section_counter += 1
        section_id = self._section_counter
        section_type = _map_heading(header)
        if self._detected_title and header == self._detected_title:
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
        self._heading_stack.append((level, section_id))
        self._current_section_id = section_id

    def _handle_text_block(self, tag: Tag) -> None:
        text = _text(tag)
        if not text:
            return
        if self._in_references():
            self._append_reference(text)
            return
        deferred_index = self.assembler.append(text, None, self._current_section_id, True, False)
        for a in tag.find_all("a", href=True):
            url = clean_extracted_url(str(a.get("href") or ""))
            if not url:
                continue
            self._pending_url_links.append(
                (url, _text(a), self._current_section_id, deferred_index)
            )

    def _process_list(self, tag: Tag) -> None:
        for li in tag.find_all("li", recursive=False):
            self._handle_text_block(li)

    def _handle_reference_items(self, tag: Tag) -> None:
        if not self._in_references():
            self._ensure_references_section()
        for li in tag.find_all("li", recursive=False):
            self._append_reference(_text(li))

    def _ensure_references_section(self) -> None:
        self._section_counter += 1
        section_id = self._section_counter
        self.sections.append(
            PaperSection(
                section_id=section_id,
                header="References",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.REFERENCES,
                classification_score=1.0,
                classification_source="exact_alias",
            )
        )
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
        for section in self.sections:
            if section.section_id == self._current_section_id:
                return section.section_type == CanonicalSection.REFERENCES
        return False

    @staticmethod
    def _looks_like_reference_list(tag: Tag) -> bool:
        tokens = _attr_tokens(tag, "id", "class", "role", "aria-label")
        # BEM sub-elements (reference__authors_list, reference__abstracts)
        # name the parts of one formatted citation, not a bibliography —
        # matching them turned every "cite this article" block into a bogus
        # References section of author-name fragments.
        return bool(re.search(r"reference(?!__)|bibliography|citation(?!__)", tokens))

    def _handle_table(self, tag: Tag, caption_override: str | None = None) -> None:
        if is_hidden_table(tag):
            # A display:none table (a print-only or responsive duplicate of a
            # visible one) is not part of the page as read.
            return
        caption_tag = tag.find("caption")
        caption = _text(caption_tag) or caption_override or None
        label = caption_label(caption, "table")
        try:
            df = html_table_frame(tag)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HTML table parse failed: %s", exc)
            df = None
        if df is None:
            # No cell grid (an image-only table, say). A table whose caption
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
        # HTML5 wraps tables as <figure><figcaption>Table 1…</figcaption>
        # <table>…</table></figure>. Without an image the figure is a table
        # wrapper: route each nested table (with the figcaption as its
        # caption) instead of recording a labelless figure and losing the
        # table. A figure holding both stays a figure (a layout table).
        tables = tag.find_all("table")
        if tables and not tag.find(["img", "picture"]):
            caption = _text(tag.find("figcaption")) or None
            for position, table in enumerate(tables):
                # The figcaption names the figure's first table; repeating it
                # on every nested table duplicated the caption and the label
                # that xrefs resolve against.
                self._handle_table(table, caption_override=caption if position == 0 else None)
            return
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


def inspect_html(html_bytes: bytes) -> tuple[bool, BeautifulSoup | None]:
    """Return whether HTML has article text plus its reusable parsed DOM."""
    if not re.search(
        rb"<\s*(?:html|body|article|main|section|p|h[1-6]|table|figure|div|li|blockquote|ol|ul)\b",
        html_bytes[:4096].lower(),
    ):
        return False, None
    try:
        soup = BeautifulSoup(_decode_html_bytes(html_bytes), "html5lib")
    except Exception:
        return False, None
    HtmlParser._remove_noise(soup)
    root = soup.find("article") or soup.find("main") or soup.body or soup
    has_content = bool(collapse_ws(root.get_text(" ", strip=True)).strip())
    return has_content, soup if has_content else None
