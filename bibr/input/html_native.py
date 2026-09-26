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
    UTF-8, then a meta or XML-declared charset, then charset_normalizer,
    finally UTF-8 with replacement.
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
        charset = match.group(1).decode("ascii", errors="ignore")
        try:
            return data.decode(charset)
        except (LookupError, UnicodeDecodeError):
            pass
    try:
        from charset_normalizer import from_bytes as _detect_encoding
    except ImportError:  # pragma: no cover — charset-normalizer is declared
        _detect_encoding = None  # type: ignore[assignment]
    if _detect_encoding is not None:
        try:
            detected = _detect_encoding(data).best()
        except Exception:  # noqa: BLE001 — fall through to the replacement decode
            detected = None
        if detected is not None and str(detected).strip():
            return str(detected)
    return data.decode("utf-8", errors="replace")


# Upper bound on HTML fed to the pure-Python html5lib parser (audit L9). Well
# above any real article/JATS/EPUB spine document, below what makes parsing a
# DoS. Kept below the serve upload cap so it fails fast on the parse path.
_MAX_HTML_BYTES = 48 * 1024 * 1024
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
            left, right = [part.strip() for part in piece.split(",", 1)]
            if len(left.split()) > 1 and len(right.split()) > 1:
                names.extend([left, right])
                continue
        names.append(piece)
    return names


def _has_citation_front_matter(soup: BeautifulSoup) -> bool:
    """True when ``citation_*`` tags supply at least a title and one author."""
    seen_title = False
    seen_author = False
    for tag in soup.find_all("meta"):
        key = _normal_meta_key(tag.get("name") or tag.get("property") or "")
        content = str(tag.get("content") or "").strip()
        if not content:
            continue
        if key == "citation.title":
            seen_title = True
        elif key == "citation.author":
            seen_author = True
        if seen_title and seen_author:
            return True
    return False


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
        self._process_children(root)

        if not self.assembler.entries and not self.tables and not self.figures:
            raise ProcessingError("HTML input did not contain parseable article content")

        # Generic SEO meta is not front matter: without citation_* title and
        # authors the preparsed record would lock in the <title> suffix, the
        # site description and a garbled byline, and post_parse would skip
        # the front-matter pass that reads the printed article instead.
        preparsed = self._metadata if _has_citation_front_matter(soup) else None
        return PaperContents(
            sentences=[],
            sections=self.sections,
            tables=self.tables,
            links=self.links,
            sections_text={},
            figures=self.figures,
            xrefs=[],
            detected_title=self._detected_title,
            preparsed_metadata=preparsed,
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

        # The generic SEO "description" is the site's blurb, not the paper's
        # abstract — it used to win over the printed ABSTRACT section.
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
        # citation_author with dc.creator doubled every author. Affiliations,
        # ORCIDs and emails following citation_author pair up by position.
        meta.authors = []
        citation_authors = values("citation_author")
        dc_authors = values("dc.creator")
        if citation_authors:
            institutions = values("citation_author_institution")
            orcids = values("citation_author_orcid")
            emails = values("citation_author_email")
            for idx, author in enumerate(citation_authors, start=1):
                given, family = _split_person_name(author)
                if not (family or given):
                    continue
                meta.authors.append(
                    PaperAuthor(
                        author_id=idx,
                        given=given,
                        family=family,
                        affiliation=institutions[idx - 1].strip()
                        if idx - 1 < len(institutions)
                        else "",
                        email=emails[idx - 1].strip() or None if idx - 1 < len(emails) else None,
                        orcid=canonicalize_orcid(orcids[idx - 1])
                        if idx - 1 < len(orcids)
                        else None,
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
        if self._in_references():
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

    def _process_children(self, parent: Tag) -> None:
        # Direct strings and inline elements at this level form their own
        # paragraph ("Bare section text…" between a heading and a <p>); a
        # block child flushes them first so document order is kept.
        parts: list[str] = []
        links: list[tuple[str, str]] = []
        for child in parent.children:
            if type(child) in (NavigableString, CData):
                if str(child).strip():
                    parts.append(str(child))
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
                else:
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
            elif name in _INLINE_TAGS:
                if self._has_block_structure(child):
                    # Invalid nesting (an unclosed <b>/<font>/<span> higher up
                    # nests whole sections inside this inline element) — the
                    # block structure wins and is recursed into, not flattened.
                    self._flush_pending_text(parts, links)
                    self._process_children(child)
                else:
                    self._buffer_inline(child, parts, links)
            elif name == "br":
                parts.append(" ")
            elif self._has_block_structure(child):
                # A container holding block markup — recurse rather than
                # flatten it into one entry.
                self._flush_pending_text(parts, links)
                self._process_children(child)
            else:
                # A container with direct text or only inline children: a
                # div/section/span paragraph reads as one text block.
                self._flush_pending_text(parts, links)
                self._handle_text_block(child)
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
        return any(word in tokens for word in ("reference", "bibliography", "citation"))

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
            for table in tables:
                self._handle_table(table, caption_override=caption)
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
