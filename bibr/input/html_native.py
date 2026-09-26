"""Native HTML parser: article HTML/XHTML -> PaperContents.

This mirrors the native DOCX/JATS parser surface so the pipeline can parse
publisher HTML without OCR. It is intentionally structural rather than
browser-like: scripts/styles/navigation are discarded, no remote resources are
fetched, and body text is mapped into the same deferred sentence-segmentation
contract used by the other native inputs.
"""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Iterator
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup, CData, NavigableString, Tag

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
from bibr.structure.html_table import html_table_frame, is_hidden_table
from bibr.structure.xref_utils import URL_RE, detect_xrefs
from bibr.utils.text import clean_extracted_url, collapse_ws

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
# Upper bound on HTML fed to the pure-Python html5lib parser (audit L9). Well
# above any real article/JATS/EPUB spine document, below what makes parsing a
# DoS. Kept below the serve upload cap so it fails fast on the parse path.
_MAX_HTML_BYTES = 48 * 1024 * 1024
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


def _anchor_offset_in_block(block: Tag, anchor: Tag, display: str) -> int:
    """Character offset where an anchor's display text starts in the block text.

    The strings before the anchor in document order are the anchor's prefix;
    block text is that accumulation collapsed, so the collapsed prefix
    length is the anchor's start, plus one separator space when the source
    spells whitespace on either side of it.
    """
    parts: list[str] = []
    for node in block.descendants:
        if node is anchor:
            break
        if isinstance(node, (NavigableString, CData)):
            parts.append(str(node))
    pre_raw = "".join(parts)
    pre = collapse_ws(pre_raw)
    if not pre or not display:
        return len(pre)
    first_inside = next(
        (str(s) for s in anchor.descendants if isinstance(s, (NavigableString, CData))),
        "",
    )
    trail = pre_raw[-1:].isspace()
    lead = first_inside[:1].isspace()
    return len(pre) + (1 if trail or lead else 0)


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
        self._pending_url_links: list[tuple[str, str, int, int, int]] = []

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
                else BeautifulSoup(self.html_bytes, "html5lib")
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
        for pending in self._pending_url_links:
            if len(pending) == 5:
                url, link_text, section_id, deferred_index, anchor_offset = pending
            else:  # links recorded without an offset fall back to text search
                url, link_text, section_id, deferred_index = pending
                anchor_offset = None
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
        for child in parent.children:
            if not isinstance(child, Tag):
                continue
            name = _tag_name(child)
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
            elif name == "table":
                self._handle_table(child)
            elif name == "figure":
                self._handle_figure(child)
            else:
                # Every other element (known containers and unknown tags alike)
                # may hold text deeper down — recurse rather than drop it.
                self._process_children(child)

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
            display = _text(a)
            self._pending_url_links.append(
                (
                    url,
                    display,
                    self._current_section_id,
                    deferred_index,
                    _anchor_offset_in_block(tag, a, display),
                )
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
        soup = BeautifulSoup(html_bytes, "html5lib")
    except Exception:
        return False, None
    HtmlParser._remove_noise(soup)
    root = soup.find("article") or soup.find("main") or soup.body or soup
    has_content = bool(collapse_ws(root.get_text(" ", strip=True)).strip())
    return has_content, soup if has_content else None
