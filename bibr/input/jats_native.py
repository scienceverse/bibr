"""Native JATS-XML parser: ``<article>`` XML → PaperContents.

Accepts a JATS (Journal Article Tag Suite) XML document and produces the same
:class:`~bibr.paper_contents.PaperContents` shape the PDF and DOCX paths build,
so the pipeline can dispatch on file type without further branching. Because
JATS carries the full front matter and a structured reference list, this parser
also pre-populates a :class:`~bibr.models.PaperMetadata` (stashed on
``contents.preparsed_metadata``) and ingests the ref-list natively — so
post-parse can skip the core LLM extraction and, for structured
``element-citation`` refs, the reference extractor entirely.

Mirrors :class:`bibr.input.docx_native.DocxParser`'s public surface
(``parse``, ``_deferred_texts``, ``apply_segmentation``,
``create_content_sections``) and reuses the shared
:class:`~bibr.structure.assembler.DocumentAssembler`. Sentence segmentation is
deferred exactly like the DOCX path — every entry has ``page_number=None``.

JATS files may or may not declare namespaces (default JATS namespace, the
``xlink`` namespace for hrefs, ``mml`` for math). All element/attribute lookups
match on the local name so both namespaced and bare documents parse.
"""

from __future__ import annotations

import itertools
import logging
import re

import pandas as pd

from bibr.input.mathml_whitespace import FlatText, mspace_separates
from bibr.input.xml_entities import parse_xml
from bibr.models import (
    ORGANIZATION_ROLE,
    PaperAuthor,
    PaperMetadata,
    PaperReference,
    canonicalize_orcid,
    migrate_bib_type,
)
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
from bibr.structure.assembler import DocumentAssembler
from bibr.structure.float_labels import FloatKind, caption_label, label_element_label
from bibr.structure.xref_utils import URL_RE, detect_xrefs
from bibr.utils.text import clean_extracted_url, collapse_ws

logger = logging.getLogger(__name__)

# JATS ``@publication-type`` values → BibType strings. Falls back to
# ``migrate_bib_type`` (which knows the BibTeX/Crossref spellings) for anything
# not listed here, e.g. "book-chapter".
_JATS_PUB_TYPE: dict[str, str] = {
    "journal": "journal_article",
    "book": "book",
    "chapter": "book_chapter",
    "confproc": "conference_paper",
    "conf-proc": "conference_paper",
    "conference": "conference_paper",
    "data": "dataset",
    "database": "dataset",
    "software": "software",
    "preprint": "preprint",
    "report": "report",
    "web": "other",
    "webpage": "other",
    "other": "other",
}


# ----------------------------------------------------------------------------
# Namespace-agnostic element helpers
# ----------------------------------------------------------------------------


def _ln(el) -> str:
    """Local name of an element's tag (namespace stripped)."""
    tag = el.tag
    if not isinstance(tag, str):  # comments / PIs
        return ""
    return tag.rsplit("}", 1)[-1]


def _iter_children(el, name: str):
    """Yield direct children whose local name is *name*."""
    for child in el:
        if _ln(child) == name:
            yield child


def _first_child(el, name: str):
    """First direct child with local name *name*, or ``None``."""
    return next(_iter_children(el, name), None)


def _first_desc(el, name: str):
    """First descendant (any depth) with local name *name*, or ``None``."""
    for desc in el.iter():
        if desc is not el and _ln(desc) == name:
            return desc
    return None


def _attr(el, name: str) -> str | None:
    """Attribute value matched by local name (handles ``xlink:href`` etc.)."""
    for key, value in el.attrib.items():
        if key.rsplit("}", 1)[-1] == name:
            return value
    return None


def _topmost_ref_lists(scope) -> list:
    """``<ref-list>`` descendants of *scope* not nested in another ref-list."""
    found = []
    for el in scope.iter():
        if el is scope or _ln(el) != "ref-list":
            continue
        parent = el.getparent()
        nested = False
        while parent is not None and parent is not scope:
            if _ln(parent) == "ref-list":
                nested = True
                break
            parent = parent.getparent()
        if not nested:
            found.append(el)
    return found


# Elements that end the run of text they sit in. XML carries no whitespace of
# its own between adjacent children, so plain concatenation fuses the words on
# either side of these: "Line one<break/>Line two" collapses to "Line oneLine
# two", and a structured <aff> becomes "Dept of PsychologyUtrecht University".
# Inline markup (italic, sup, xref, ext-link...) is deliberately absent — a
# separator there would split "H<sub>2</sub>O" into "H 2 O".
_TEXT_BOUNDARY = frozenset(
    {
        # Explicit line break, and block-level containers.
        "break",
        "p",
        "sec",
        "title",
        "label",
        "abstract",
        "disp-quote",
        "list-item",
        "def",
        "term",
        "tr",
        "td",
        "th",
        # MathML matrix rows and cells, as in the HTML parser.
        "mtr",
        "mlabeledtr",
        "mtd",
        # Structured <aff>/<address> fields — sibling values, not a sentence.
        "institution",
        "institution-wrap",
        "institution-id",
        "addr-line",
        "city",
        "state",
        "country",
        "postal-code",
        "phone",
        "fax",
        "email",
    }
)


# Boundaries that also end the run they sit in. ``_TEXT_BOUNDARY`` inserts a
# space before the element; these additionally need one after, because the
# source spells no whitespace there either: a printed ``<label>`` fuses with
# whatever follows it ("(2)where", "1Smith"). Restricted to elements that
# never occur inside MathML, where a trailing space would corrupt a formula
# ("[ 0 1 10 20]" must not become "[ 0 1 10 20 ]").
_TEXT_BOUNDARY_AFTER = frozenset({"label", "title", "term"})

# Children of <alternatives> that carry no text of their own — never the
# fallback representative.
_ALT_GRAPHIC = frozenset({"graphic", "inline-graphic", "media", "inline-media"})

# Block-level JATS that may sit inside a <p> (PMC nests display equations in
# paragraphs; author manuscripts nest floats). A paragraph walker flushes the
# prose so far, dispatches the child to its handler, and continues with the
# child's tail.
_P_BLOCKS = frozenset(
    {
        "fig",
        "table-wrap",
        "disp-formula",
        "list",
        "list-item",
        "boxed-text",
        "disp-quote",
        "fig-group",
        "table-wrap-group",
        "def-list",
        "def-item",
        "fn-group",
        "ref-list",
        "statement",
        "supplementary-material",
        "preformat",
        "caption",
    }
)

_TEX_BEGIN_DOCUMENT = re.compile(r"\\begin\{document\}")
_TEX_END_DOCUMENT = re.compile(r"\\end\{document\}")

# Citation sub-elements that end the run of text they sit in. An
# element-citation has no whitespace of its own between fields, so plain
# flattening fuses "Smith" + "J" + "Sleep and memory". Inline markup (italic,
# sub, sup, xref...) stays absent so "H<sub>2</sub>O" keeps reading "H2O".
# Used only for reference rows — never for body prose.
_CITATION_BOUNDARY_EXTRA = frozenset(
    {
        "person-group",
        "name",
        "string-name",
        "surname",
        "given-names",
        "suffix",
        "prefix",
        "degrees",
        "etal",
        "collab",
        "article-title",
        "chapter-title",
        "trans-title",
        "trans-source",
        "source",
        "series",
        "year",
        "month",
        "day",
        "volume",
        "issue",
        "supplement",
        "fpage",
        "lpage",
        "elocation-id",
        "pub-id",
        "comment",
        "annotation",
        "publisher-name",
        "publisher-loc",
        "edition",
        "version",
        "conf-name",
        "conf-date",
        "conf-loc",
        "conf-sponsor",
        "size",
    }
)
_CITATION_BOUNDARY = _TEXT_BOUNDARY | _CITATION_BOUNDARY_EXTRA


def _citation_text(ec) -> str:
    """Flatten an ``<element-citation>`` with its fields kept apart.

    Unlike a string synthesized from the parsed reference, this keeps every
    sub-element the parser does not model — ``<comment>`` access notes and
    their URLs, consortium authors, name suffixes, conference details — so a
    row never loses printed text the old fused flattening kept.
    """
    return collapse_ws(_Walker(None, boundaries=_CITATION_BOUNDARY).run(ec)).strip()


# ``notes-type`` values with an unambiguous canonical section. Anything else
# (competing-interest wordings, funding statements, plain notes) keeps its
# text under an UNKNOWN section the classifier can still re-type by header.
_NOTES_TYPE_MAP = {
    "data-availability": CanonicalSection.OPEN_DATA,
    "coi-statement": CanonicalSection.COI,
    "financial-disclosure": CanonicalSection.FUNDING,
}


def _trim_tex_math(text: str) -> str:
    """Keep only a TeX preamble's document body, with ``$$`` delimiters stripped."""
    body = text
    m = _TEX_BEGIN_DOCUMENT.search(body)
    if m:
        body = body[m.end() :]
    m = _TEX_END_DOCUMENT.search(body)
    if m:
        body = body[: m.start()]
    return collapse_ws(body.replace("$$", "")).strip()


def _choose_alternative(alt) -> tuple[object | None, str | None]:
    """Pick the single representative child of an ``<alternatives>`` element.

    Returns ``(element, None)`` to walk, or ``(None, text)`` to emit — never
    both. MathML first (it is the rendered form), else the TeX body without
    its document preamble, else the first non-graphic child.
    """
    math = tex = fallback = first = None
    for child in alt:
        ln = _ln(child)
        if not ln:  # comments / processing instructions are not content
            continue
        if first is None:
            first = child
        if ln == "math" and math is None:
            math = child
        elif ln == "tex-math" and tex is None:
            tex = child
        elif ln not in _ALT_GRAPHIC and fallback is None:
            fallback = child
    if math is not None:
        return math, None
    if tex is not None:
        body = _trim_tex_math(_flatten(tex))
        if body:
            return None, body
    target = fallback if fallback is not None else first
    return target, None


class _Walker:
    """One text-flattening pass over an element.

    :meth:`run` matches the old ``_flatten`` exactly (plus the
    ``<alternatives>`` single-emission and the after-boundary spaces).
    :class:`JatsParser` reuses it to split a ``<p>`` at block children:
    *on_block* flushes the prose so far and dispatches the child, and
    *on_link* records each ``ext-link``/``uri`` target for the entry being
    accumulated.
    """

    def __init__(self, exclude, on_block=None, on_link=None, boundaries=None) -> None:
        self.flat = FlatText()
        self.serials = itertools.count()
        self.exclude = exclude
        self.on_block = on_block
        self.on_link = on_link
        self.boundaries = _TEXT_BOUNDARY if boundaries is None else boundaries

    def run(self, el) -> str:
        self.walk(el, False, next(self.serials), True)
        return self.flat.join()

    def reset(self) -> None:
        self.flat = FlatText()

    def _add(self, text: str, in_math: bool, owner: str | None, group: int) -> None:
        if in_math:
            self.flat.add_math(text, owner, group)
        else:
            self.flat.add(text)

    def _walk_alternative(self, alt, in_math: bool, serial: int) -> None:
        chosen, text = _choose_alternative(alt)
        if chosen is not None:
            self.walk(chosen, in_math, serial, False)
        elif text:
            self._add(text, False, "alternatives", serial)

    def walk(self, node, in_math: bool, parent: int, top: bool) -> None:
        ln = _ln(node)
        in_math = in_math or ln == "math"
        serial = next(self.serials)
        if node.text:
            self._add(node.text, in_math, ln, parent)
        for child in node:
            child_ln = _ln(child)
            if not child_ln:  # comments / processing instructions are not content
                if child.tail:
                    self._add(child.tail, in_math, None, serial)
                continue
            if top and self.on_block is not None and self.on_block(child):
                # Dispatched (and flushed) by the caller; the tail still
                # belongs to the running text that follows the block.
                pass
            elif self.exclude is None or child_ln not in self.exclude:
                if child_ln in self.boundaries or (
                    child_ln == "mspace" and mspace_separates(child.attrib)
                ):
                    self.flat.separate()
                if child_ln == "alternatives":
                    self._walk_alternative(child, in_math, serial)
                elif child_ln == "tex-math":
                    # A bare TeX formula (no <alternatives> around it) carries
                    # the same Springer preamble — keep only its body.
                    body = _trim_tex_math(_flatten(child))
                    if body:
                        self._add(body, False, "tex-math", serial)
                else:
                    self.walk(child, in_math, serial, False)
                if child_ln in _TEXT_BOUNDARY_AFTER:
                    self.flat.separate()
                if self.on_link is not None and child_ln in ("ext-link", "uri"):
                    href = _attr(child, "href")
                    if href:
                        url = clean_extracted_url(href)
                        if url:
                            self.on_link(url, _text(child))
            if child.tail:
                self._add(child.tail, in_math, None, serial)


def _flatten(el, exclude: set[str] | None = None) -> str:
    """Concatenate descendant text, skipping local names in *exclude*.

    Inserts a single space where :data:`_TEXT_BOUNDARY` markup implies a word
    boundary the source itself does not spell out — but never where the text so
    far already ends in whitespace, so an ``<aff>`` whose fields are separated
    by ", " in the source stays "X, Y" instead of becoming "X , Y". Whitespace
    between MathML elements is dropped as a renderer drops it, except where it
    keeps two words apart (:mod:`bibr.input.mathml_whitespace`).
    """
    return _Walker(exclude).run(el)


def _text(el) -> str:
    """Flatten all descendant text to a whitespace-collapsed string."""
    if el is None:
        return ""
    return collapse_ws(_flatten(el)).strip()


def _flatten_excluding(el, exclude: set[str]) -> str:
    """Concatenate text of *el* skipping subtrees whose local name is in *exclude*."""
    return _flatten(el, exclude)


def _unwrap_article_root(root):
    """Return the ``<article>`` element for a parsed JATS document.

    An NCBI E-utilities efetch (db=pmc) response wraps its single article in
    a ``<pmc-articleset>`` element; unwrap that single child so programmatic
    PMC downloads parse. Multi-article sets and any other root raise
    ``ValueError`` with a specific message.
    """
    if _ln(root) == "article":
        return root
    if _ln(root) == "pmc-articleset":
        articles = [
            child
            for child in root
            if isinstance(getattr(child, "tag", None), str) and _ln(child) == "article"
        ]
        if len(articles) == 1:
            return articles[0]
        raise ValueError(
            f"PMC articleset contains {len(articles)} <article> documents; "
            "only single-article JATS <article> documents are supported"
        )
    raise ValueError(f"Expected <article> root, got <{_ln(root)}>")


class JatsParser:
    """Parses JATS-XML bytes directly into a :class:`PaperContents`.

    Sentence segmentation is deferred — call :meth:`apply_segmentation` after
    :meth:`parse` with externally-produced segments, identical to the PDF and
    DOCX parsers.
    """

    def __init__(self, xml_bytes: bytes) -> None:
        self.xml_bytes = xml_bytes
        # page_number is None for XML — pages are a render-time concept.
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
        self._detected_title: str | None = None

        self._metadata: PaperMetadata = PaperMetadata(doi="", title="")
        self._native_references: list[PaperReference] | None = None
        self._native_ref_strings: list[str] | None = None
        # (text, printed <label>) of each back-matter <fn>.
        self._footnotes: list[tuple[str, str | None]] = []
        self._aff_map: dict[str, str] = {}
        self._body_ref_lists: list[tuple[object, int]] = []
        # Anchor targets seen while walking paragraphs: (url, display text,
        # section id, deferred entry index), resolved to text_ids in
        # apply_segmentation like the HTML parser's _pending_url_links.
        self._pending_url_links: list[tuple[str, str, int, int]] = []
        # Reference accumulation across every <ref-list> (fix: a later list
        # must extend the bibliography, not replace it). Rows and structured
        # refs collect here with one continuous bib_id counter; the
        # structured-vs-strings invariant is decided once, over the union, in
        # _finalize_references after the whole back matter is walked.
        self._ref_rows: list[str] = []
        self._ref_structured: list[PaperReference] = []
        self._ref_all_structured = True
        self._ref_next_id = 1

    # ------------------------------------------------------------------
    # Public API (mirrors DocxParser / PDFParser)
    # ------------------------------------------------------------------

    @property
    def _deferred_texts(self) -> list[tuple[str, int | None, int, bool, bool]]:
        """Legacy 5-tuple view of the deferred-text buffer (read-only)."""
        return [
            (e.text, e.page_number, e.section_id, e.needs_segmentation, e.is_formula)
            for e in self.assembler.entries
        ]

    def parse(self) -> PaperContents:
        """Parse the JATS document and populate sections/tables/deferred texts."""
        # Root section (section_id=0) — matches the DOCX/PDF contract.
        self.sections.append(
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None)
        )

        try:
            root = parse_xml(self.xml_bytes)
        except Exception as exc:
            from bibr.exceptions import ProcessingError

            raise ProcessingError(f"Failed to parse JATS XML: {exc}") from exc

        try:
            root = _unwrap_article_root(root)
        except ValueError as exc:
            from bibr.exceptions import ProcessingError

            raise ProcessingError(str(exc)) from exc

        front = _first_child(root, "front")
        body = _first_child(root, "body")
        back = _first_child(root, "back")

        if front is not None:
            self._metadata = self._parse_front(front)
            self._metadata.language = (
                root.get("{http://www.w3.org/XML/1998/namespace}lang") or root.get("lang") or None
            )
        if body is not None:
            self._process_container(body, section_id=0, depth=0)
        if back is not None:
            self._parse_back(back)
        self._recover_body_ref_list()
        self._finalize_references()

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
            native_references=self._native_references,
            native_ref_strings=self._native_ref_strings,
            native_ref_strings_authoritative=True,
        )

    def _make_sentence(self, entry, text: str, text_id: int, paragraph_id: int) -> PaperSentence:
        """Build a sentence — no provenance/region_meta side-channels, never OCR text
        (like DOCX)."""
        return PaperSentence(
            text_id=text_id,
            text=text,
            section_id=entry.section_id,
            paragraph_id=paragraph_id,
            page_number=entry.page_number,
            from_ocr=False,
        )

    def apply_segmentation(self, contents: PaperContents, all_segments: list[list[str]]) -> None:
        """Populate sentences from externally-segmented results (mirrors DOCX)."""
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
        """Create dedicated sections for figures, tables, and footnotes (mirrors DOCX)."""
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
                caption_sent = PaperSentence(
                    text_id=self._sentence_counter,
                    text=fig.caption,
                    section_id=self._section_counter,
                    paragraph_id=self._paragraph_counter,
                    page_number=None,
                    from_ocr=False,
                )
                contents.sentences.append(caption_sent)
                if getattr(fig, "_in_paragraph", False):
                    # Met inside a <p>: the caption merged into the paragraph
                    # sentence, so its URLs were linked — keep that regex pass.
                    self._detect_urls(caption_sent)
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
                caption_sent = PaperSentence(
                    text_id=self._sentence_counter,
                    text=tbl.caption,
                    section_id=self._section_counter,
                    paragraph_id=self._paragraph_counter,
                    page_number=None,
                    from_ocr=False,
                )
                contents.sentences.append(caption_sent)
                if getattr(tbl, "_in_paragraph", False):
                    # As for figures met inside a <p>.
                    self._detect_urls(caption_sent)
                self._sentence_counter += 1

        for footnote_num, (fn_text, fn_label) in enumerate(self._footnotes, start=1):
            self._section_counter += 1
            footnote_section_id = self._section_counter
            contents.sections.append(
                PaperSection(
                    section_id=footnote_section_id,
                    header=f"Footnote {footnote_num}",
                    level=1,
                    parent_section_id=0,
                    section_type=CanonicalSection.FOOTNOTE,
                    synthetic_kind="footnote",
                    footnote_label=fn_label,
                )
            )
            self._paragraph_counter += 1
            contents.sentences.append(
                PaperSentence(
                    text_id=self._sentence_counter,
                    text=fn_text,
                    section_id=footnote_section_id,
                    paragraph_id=self._paragraph_counter,
                    page_number=None,
                    from_ocr=False,
                )
            )
            self._sentence_counter += 1

    # ------------------------------------------------------------------
    # Front matter → PaperMetadata
    # ------------------------------------------------------------------

    def _parse_front(self, front) -> PaperMetadata:
        journal_meta = _first_desc(front, "journal-meta")
        article_meta = _first_desc(front, "article-meta")

        meta = PaperMetadata(doi="", title="")

        if journal_meta is not None:
            meta.journal = _text(_first_desc(journal_meta, "journal-title")) or None
            meta.issn = self._issn(journal_meta)
            pub = _first_desc(journal_meta, "publisher-name")
            meta.publisher = _text(pub) or None

        if article_meta is None:
            return meta

        # DOI and the other identifiers JATS declares
        for aid in _iter_children(article_meta, "article-id"):
            id_type = (_attr(aid, "pub-id-type") or "").lower()
            value = _text(aid) or None
            if id_type == "doi" and not meta.doi:
                meta.doi = value or ""
            elif id_type == "pmid" and meta.pmid is None:
                meta.pmid = value
            elif id_type in ("pmc", "pmcid") and meta.pmcid is None:
                meta.pmcid = (
                    value if value is None or value.upper().startswith("PMC") else f"PMC{value}"
                )
            elif id_type == "arxiv" and meta.arxiv is None:
                meta.arxiv = value

        # Title
        title_group = _first_desc(article_meta, "title-group")
        if title_group is not None:
            title_el = _first_desc(title_group, "article-title")
            title = _text(title_el)
            if title:
                meta.title = title
                self._detected_title = title

        # Abstract (first <abstract>, excluding its heading title)
        abstract_el = _first_desc(article_meta, "abstract")
        if abstract_el is not None:
            meta.abstract = collapse_ws(_flatten_excluding(abstract_el, {"title"})).strip()

        # Keywords
        keywords: list[str] = []
        for kwd_group in _iter_children(article_meta, "kwd-group"):
            for kwd in kwd_group.iter():
                if _ln(kwd) == "kwd":
                    kw = _text(kwd)
                    if kw:
                        keywords.append(kw)
        meta.keywords = keywords

        # Authors + affiliations
        self._aff_map = self._collect_affs(article_meta)
        meta.authors = self._parse_authors(article_meta)

        # Bibliographic self-identity
        meta.volume = _text(_first_desc(article_meta, "volume")) or None
        meta.issue = _text(_first_desc(article_meta, "issue")) or None
        meta.first_page = _text(_first_desc(article_meta, "fpage")) or None
        meta.last_page = _text(_first_desc(article_meta, "lpage")) or None
        meta.published = self._pub_date(article_meta)
        meta.license = self._license(article_meta)

        return meta

    def _issn(self, journal_meta) -> str | None:
        """Prefer the print ISSN, else the first electronic one."""
        electronic: str | None = None
        for issn in _iter_children(journal_meta, "issn"):
            fmt = (_attr(issn, "pub-type") or _attr(issn, "publication-format") or "").lower()
            value = _text(issn)
            if not value:
                continue
            if fmt in ("ppub", "print"):
                return value
            if electronic is None:
                electronic = value
        return electronic

    def _pub_date(self, article_meta) -> str | None:
        """Format the preferred publication date as YYYY[-MM[-DD]]."""
        dates = list(_iter_children(article_meta, "pub-date"))
        if not dates:
            return None
        chosen = None
        for d in dates:
            kind = (_attr(d, "pub-type") or _attr(d, "date-type") or "").lower()
            if kind in ("epub", "pub"):
                chosen = d
                break
        if chosen is None:
            chosen = dates[0]
        year = _text(_first_child(chosen, "year"))
        if not year:
            return None
        month = _text(_first_child(chosen, "month"))
        day = _text(_first_child(chosen, "day"))

        def _num(s: str) -> str | None:
            try:
                return f"{int(s):02d}"
            except (ValueError, TypeError):
                return None

        mm = _num(month) if month else None
        dd = _num(day) if day else None
        if mm and dd:
            return f"{year}-{mm}-{dd}"
        if mm:
            return f"{year}-{mm}"
        return year

    def _license(self, article_meta) -> str | None:
        """License href (preferred) or the license-p text."""
        permissions = _first_desc(article_meta, "permissions")
        scope = permissions if permissions is not None else article_meta
        lic = _first_desc(scope, "license")
        if lic is None:
            return None
        href = _attr(lic, "href")
        if href:
            return href
        lic_p = _first_desc(lic, "license-p")
        text = _text(lic_p) if lic_p is not None else _text(lic)
        return text or None

    def _collect_affs(self, scope) -> dict[str, str]:
        """Map every ``<aff>`` id to its flattened text (label excluded)."""
        affs: dict[str, str] = {}
        for aff in scope.iter():
            if _ln(aff) != "aff":
                continue
            aid = _attr(aff, "id")
            if aid:
                affs[aid] = collapse_ws(_flatten_excluding(aff, {"label"})).strip()
        return affs

    def _aff_text(self, aff) -> str:
        """Flattened ``<aff>`` text (label excluded), including id-less ones."""
        return collapse_ws(_flatten_excluding(aff, {"label"})).strip()

    def _parse_authors(self, article_meta) -> list[PaperAuthor]:
        authors: list[PaperAuthor] = []
        idx = 0
        # Article-meta-level <aff> elements: last-resort fallback when there
        # is exactly one (id-less ones count — _aff_map only holds ids).
        meta_affs = [self._aff_text(aff) for aff in _iter_children(article_meta, "aff")]
        meta_affs = [t for t in meta_affs if t]

        def handle_contrib(contrib, group_affs: list[str]) -> None:
            nonlocal idx
            if (_attr(contrib, "contrib-type") or "author") != "author":
                return
            # Direct children only: a consortium <collab> may nest member
            # <contrib>s whose <name> must not stand in for the group.
            name = _first_child(contrib, "name")
            if name is None:
                name = _first_child(contrib, "string-name")
            given = family = ""
            roles: list[str] = []
            is_group = False
            if name is not None:
                if _ln(name) == "string-name":
                    family = _text(name)
                else:
                    given = _text(_first_child(name, "given-names"))
                    family = _text(_first_child(name, "surname"))
            else:
                # A consortium/working-group byline carries <collab> instead of
                # a personal name: one unsplit name, marked as an organization
                # so the export writes it to ``author[].literal``. Nested
                # member lists are not part of the name.
                collab = _first_child(contrib, "collab")
                if collab is not None:
                    family = collapse_ws(_flatten_excluding(collab, {"contrib-group"})).strip()
                    roles = [ORGANIZATION_ROLE]
                    is_group = True

            if not given and not family:
                # No name of any kind — emitting the row would only produce a
                # blank author (VAL_AUTHOR_BLANK) and shift every later id.
                return
            idx += 1

            email = _text(_first_child(contrib, "email")) or None

            orcid = None
            for cid in _iter_children(contrib, "contrib-id"):
                if _attr(cid, "contrib-id-type") == "orcid":
                    orcid = canonicalize_orcid(_text(cid))
                    break

            corresponding = (_attr(contrib, "corresp") or "").lower() == "yes"
            aff_texts: list[str] = []
            for child in contrib:
                ln = _ln(child)
                if ln == "aff":
                    text = self._aff_text(child)
                    if text:
                        aff_texts.append(text)
                elif ln == "xref":
                    ref_type = (_attr(child, "ref-type") or "").lower()
                    if ref_type == "corresp":
                        corresponding = True
                    elif ref_type == "aff":
                        # @rid is IDREFS: one author may cite several affs.
                        for rid in (_attr(child, "rid") or "").split():
                            if rid in self._aff_map:
                                aff_texts.append(self._aff_map[rid])
            if not aff_texts and not is_group:
                # No xref or nested aff: the JATS convention for a shared
                # affiliation is an <aff> beside the contribs, else a single
                # article-meta-level one.
                aff_texts = [t for t in group_affs if t] or (
                    meta_affs if len(meta_affs) == 1 else []
                )

            affiliation = "; ".join(t for t in aff_texts if t)
            authors.append(
                PaperAuthor(
                    author_id=idx,
                    given=given,
                    family=family,
                    affiliation=affiliation,
                    email=email,
                    corresponding=corresponding,
                    orcid=orcid,
                    role=roles,
                )
            )
            if is_group:
                # Consortium members nested inside the <collab> are credited
                # authors in their own right — emit them once, right after
                # the group, instead of dropping them or folding their names
                # into the group row.
                for group in _iter_children(collab, "contrib-group"):
                    nested_affs = [self._aff_text(a) for a in _iter_children(group, "aff")]
                    for member in _iter_children(group, "contrib"):
                        handle_contrib(member, [t for t in nested_affs if t])

        for child in article_meta:
            ln = _ln(child)
            if ln == "contrib-group":
                # Member contribs nested in a <collab> are not authors of
                # their own; only the group's direct contribs are.
                group_affs = [self._aff_text(aff) for aff in _iter_children(child, "aff")]
                for contrib in _iter_children(child, "contrib"):
                    handle_contrib(contrib, group_affs)
            elif ln == "contrib":
                handle_contrib(child, [])
        return authors

    # ------------------------------------------------------------------
    # Body → sections / paragraphs / tables / figures
    # ------------------------------------------------------------------

    def _process_container(self, el, section_id: int, depth: int) -> None:
        """Walk a container's children in document order into the current section."""
        for child in el:
            self._handle_block(child, section_id, depth)

    def _handle_block(self, child, section_id: int, depth: int, in_paragraph: bool = False) -> None:
        """Dispatch one block-level child (shared by containers and <p> interiors).

        *in_paragraph* marks floats met inside a ``<p>`` — their captions
        used to merge into the paragraph sentence, so caption URLs were
        linked; the flag lets caption sentences keep that regex pass.
        """
        ln = _ln(child)
        if ln == "sec":
            self._handle_sec(child, depth + 1, section_id)
        elif ln == "p":
            self._handle_paragraph(child, section_id)
        elif ln == "disp-formula":
            self._handle_formula(child, section_id)
        elif ln == "table-wrap":
            self._handle_table_wrap(child, section_id, in_paragraph)
        elif ln == "fig":
            self._handle_fig(child, section_id, in_paragraph)
        elif ln in (
            "list",
            "list-item",
            "boxed-text",
            "disp-quote",
            "fig-group",
            "table-wrap-group",
            "def-list",
            "def-item",
            "statement",
            "supplementary-material",
            "caption",
            "fn",
        ):
            # Grouping wrappers and their items — recurse; list items carry
            # <p> children, a caption carries the <p> of its own text, and a
            # table-foot <fn> carries its <p> the same way.
            self._process_container(child, section_id, depth)
        elif ln in ("term", "def", "preformat"):
            txt = _text(child)
            if txt:
                self.assembler.append(txt, None, section_id, True, False)
        elif ln == "fn-group":
            # Notes printed under a heading of their own ("Footnotes")
            # are footnotes, like a back-matter <fn-group>.
            self._collect_footnotes(child)
        elif ln == "ref-list":
            # EuropePMC's fullTextXML puts the bibliography in <body> as a
            # <sec sec-type="ref-list"> instead of in <back>. Record it and
            # let parse() decide — <back> is walked after the body, and a
            # ref-list there is the authoritative one.
            self._body_ref_lists.append((child, section_id))
        elif ln in ("title", "label"):
            pass  # handled by the parent sec/fig/table-wrap
        else:
            logger.debug("JATS: ignoring <%s> in <%s>", ln, _ln(child.getparent()))

    def _collect_footnotes(self, fn_group) -> None:
        """Keep each <fn>'s text, and its printed <label> apart, for the footnote rows."""
        for fn in _iter_children(fn_group, "fn"):
            fn_text = _text(fn)
            if fn_text:
                label_el = next(_iter_children(fn, "label"), None)
                label = _text(label_el) if label_el is not None else ""
                self._footnotes.append((fn_text, label or None))

    def _handle_sec(self, sec, depth: int, parent_id: int) -> int:
        title_el = _first_child(sec, "title")
        header = _text(title_el) if title_el is not None else ""
        self._section_counter += 1
        sid = self._section_counter
        self.sections.append(
            PaperSection(
                section_id=sid,
                header=header,
                level=depth,
                parent_section_id=parent_id,
                section_type=self._map_sec_type(sec),
            )
        )
        self._process_container(sec, sid, depth)
        return sid

    @staticmethod
    def _map_sec_type(sec) -> CanonicalSection:
        """Map a slam-dunk ``@sec-type`` to a canonical section; else UNKNOWN.

        Only unambiguous IMRaD types are set — the shared section classifier
        (which keys on header text) handles everything else and will re-type
        these anyway; setting them is a defensive no-cost hint.
        """
        st = (_attr(sec, "sec-type") or "").lower().replace("|", " ")
        mapping = {
            "intro": CanonicalSection.INTRODUCTION,
            "introduction": CanonicalSection.INTRODUCTION,
            "methods": CanonicalSection.METHODS,
            "materials methods": CanonicalSection.METHODS,
            "materials and methods": CanonicalSection.METHODS,
            "results": CanonicalSection.RESULTS,
            "discussion": CanonicalSection.DISCUSSION,
        }
        return mapping.get(st, CanonicalSection.UNKNOWN)

    def _handle_paragraph(self, p, section_id: int) -> None:
        # A <p> may hold block children — PMC nests display equations in
        # paragraphs, manuscripts nest floats. Flattening the whole <p> would
        # merge captions and table cells into one body sentence and register
        # no figure, table or formula, so walk its children instead:
        # accumulate prose (with inline markup flattened, so detect_xrefs
        # still finds citation text), and on a block child flush the prose
        # so far as a paragraph entry, dispatch the child, and continue with
        # its tail. Anchor targets met along the way are attributed to the
        # entry being accumulated.
        links: list[tuple[str, str]] = []  # (url, display text) of this segment

        def flush() -> None:
            txt = collapse_ws(walker.flat.join()).strip()
            walker.reset()
            segment_links = list(links)
            links.clear()
            if not txt:
                return
            entry_idx = self.assembler.append(txt, None, section_id, True, False)
            for url, link_text in segment_links:
                self._pending_url_links.append((url, link_text, section_id, entry_idx))

        def on_block(child) -> bool:
            if _ln(child) not in _P_BLOCKS:
                return False
            flush()
            self._handle_block(child, section_id, 0, in_paragraph=True)
            return True

        def on_link(url: str, link_text: str) -> None:
            links.append((url, link_text))

        walker = _Walker(None, on_block=on_block, on_link=on_link)
        walker.run(p)
        flush()

    def _handle_formula(self, formula, section_id: int) -> None:
        math = _text(formula)
        if not math:
            return
        self.assembler.append(math, None, section_id, needs_segmentation=False, is_formula=True)

    def _handle_table_wrap(self, table_wrap, section_id: int, in_paragraph: bool = False) -> None:
        caption = self._caption_text(table_wrap)
        table_el = _first_desc(table_wrap, "table")
        df = self._table_to_df(table_el)
        if df is not None:
            try:
                html = df.to_html(index=False)
            except Exception:  # noqa: BLE001 — degrade gracefully, never crash
                html = ""
        elif caption:
            # No cell grid: a table printed as an image (<graphic> only). A
            # table-wrap is a table whatever it holds, so one with a label or
            # caption is kept with no contents and mentions resolve to it, as
            # the HTML parser keeps a captioned image-only <table>.
            df = pd.DataFrame()
            html = ""
        else:
            logger.warning("JATS table-wrap produced no parseable table; skipping")
            return
        record = PaperTable(
            table_id=self._table_counter,
            df=df,
            tbl_html=html,
            section_id=section_id,
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
            label=self._float_label(table_wrap, caption, "table"),
        )
        record._in_paragraph = in_paragraph
        self.tables.append(record)
        self._table_counter += 1
        # Table footnotes print under the table; keep their paragraphs rather
        # than dropping them with the grid.
        foot = _first_child(table_wrap, "table-wrap-foot")
        if foot is not None:
            self._process_container(foot, section_id, 0)

    @staticmethod
    def _table_to_df(table_el) -> pd.DataFrame | None:
        """Convert a JATS/XHTML ``<table>`` to a DataFrame; ``None`` on failure."""
        if table_el is None:
            return None
        try:
            trs = [tr for tr in table_el.iter() if _ln(tr) == "tr"]
            if not trs:
                return None

            def cells(tr) -> list[str]:
                return [_text(c) for c in tr if _ln(c) in ("td", "th")]

            thead = _first_desc(table_el, "thead")
            header_tr = None
            if thead is not None:
                header_tr = next((tr for tr in thead.iter() if _ln(tr) == "tr"), None)
            if header_tr is None:
                header_tr = trs[0]
                body_trs = trs[1:]
            else:
                body_trs = [tr for tr in trs if tr is not header_tr]

            header = cells(header_tr) if header_tr is not None else []
            data = [cells(tr) for tr in body_trs]
            width = max([len(header), *[len(r) for r in data]], default=0)
            if width == 0:
                return None
            header = header + [""] * (width - len(header))
            data = [r + [""] * (width - len(r)) for r in data]
            return pd.DataFrame(data, columns=header)
        except Exception:  # noqa: BLE001 — any malformed table degrades to skip
            return None

    def _handle_fig(self, fig, section_id: int, in_paragraph: bool = False) -> None:
        caption = self._caption_text(fig)
        # <graphic> hrefs are unresolvable in a bare XML file — image stays None.
        record = PaperFigure(
            figure_id=self._figure_counter,
            section_id=section_id,
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
            label=self._float_label(fig, caption, "figure"),
        )
        record._in_paragraph = in_paragraph
        self.figures.append(record)
        self._figure_counter += 1

    @staticmethod
    def _caption_text(el) -> str:
        """Label + caption text for a fig/table-wrap."""
        label = _text(_first_child(el, "label"))
        caption_el = _first_child(el, "caption")
        caption = _text(caption_el) if caption_el is not None else ""
        return " ".join(x for x in (label, caption) if x)

    @staticmethod
    def _float_label(el, caption: str, kind: FloatKind) -> str | None:
        """The printed label of a fig/table-wrap: its ``<label>`` without the
        word ("Table 2" → "2", a bare "S1" stays "S1"), else the label a
        caption opens with when there is no ``<label>``."""
        label_el = _first_child(el, "label")
        if label_el is not None:
            return label_element_label(_text(label_el), kind)
        return caption_label(caption, kind)

    # Back matter → acknowledgments / footnotes / references
    # ------------------------------------------------------------------

    def _parse_back(self, back) -> None:
        ref_lists = list(_iter_children(back, "ref-list"))
        ref_lists_handled = bool(ref_lists)
        for child in back:
            ln = _ln(child)
            if ln == "ack":
                self._section_counter += 1
                sid = self._section_counter
                self.sections.append(
                    PaperSection(
                        section_id=sid,
                        header="Acknowledgments",
                        level=1,
                        parent_section_id=0,
                        section_type=CanonicalSection.ACKNOWLEDGMENT,
                    )
                )
                self._process_container(child, sid, 1)
            elif ln == "sec":
                sid = self._handle_sec(child, 1, 0)
                # Some producers nest the bibliography in a back <sec>;
                # reuse its section rather than appending a second one.
                for nested_list in _topmost_ref_lists(child):
                    self._handle_ref_list(nested_list, section_id=sid)
                    ref_lists_handled = True
            elif ln == "ref-list":
                self._handle_ref_list(child)
            elif ln == "fn-group":
                self._collect_footnotes(child)
            elif ln == "notes":
                notes_type = (_attr(child, "notes-type") or "").lower()
                self._handle_back_section(
                    child,
                    _NOTES_TYPE_MAP.get(notes_type, CanonicalSection.UNKNOWN),
                    header_fallback=_attr(child, "notes-type") or "Notes",
                )
            elif ln == "app-group":
                apps = [app for app in child if _ln(app) == "app"]
                if apps:
                    for app in apps:
                        self._handle_back_section(
                            app,
                            CanonicalSection.APPENDIX,
                            header_fallback=_attr(app, "id") or "Appendix",
                        )
                else:
                    self._handle_back_section(
                        child, CanonicalSection.APPENDIX, header_fallback="Appendix"
                    )
            elif ln == "glossary":
                self._handle_back_section(
                    child, CanonicalSection.UNKNOWN, header_fallback="Glossary"
                )
            elif ln == "bio":
                self._handle_back_section(
                    child, CanonicalSection.UNKNOWN, header_fallback="Biography"
                )
            else:
                logger.debug("JATS: ignoring back-matter <%s>", ln)

        # A ref-list nested anywhere else in <back> (not a direct child, not
        # in a <sec>) is still the bibliography when nothing else was handled.
        if not ref_lists_handled:
            nested = _first_desc(back, "ref-list")
            if nested is not None:
                self._handle_ref_list(nested)

    def _handle_back_section(
        self, el, section_type: CanonicalSection, header_fallback: str
    ) -> None:
        """Emit one back-matter section (notes/app/glossary/bio) from its <title>."""
        title_el = _first_child(el, "title")
        header = _text(title_el) if title_el is not None else ""
        self._section_counter += 1
        sid = self._section_counter
        self.sections.append(
            PaperSection(
                section_id=sid,
                header=header or header_fallback,
                level=1,
                parent_section_id=0,
                section_type=section_type,
            )
        )
        self._process_container(el, sid, 1)

    def _retype_as_references(self, section_id: int, ref_list) -> None:
        """Mark an existing body section as the references section."""
        for section in self.sections:
            if section.section_id != section_id:
                continue
            section.section_type = CanonicalSection.REFERENCES
            if not section.header:
                section.header = _text(_first_child(ref_list, "title")) or "References"
            return

    def _recover_body_ref_list(self) -> None:
        """Ingest a ``<ref-list>`` found in ``<body>`` when ``<back>`` had none.

        Runs only when the document has otherwise yielded no references, so a
        well-formed ``<back>`` ref-list always wins and no document that parses
        correctly today changes. Without it a body-located bibliography is
        dropped in silence — ``_process_container`` has nowhere to put it.
        """
        if self._ref_rows or self._ref_structured:
            return
        if not self._body_ref_lists:
            return
        ref_list, section_id = self._body_ref_lists[0]
        # The enclosing <sec> is the references heading the producer already
        # emitted; reuse it rather than appending a second, competing one.
        self._handle_ref_list(ref_list, section_id=section_id or None)

    def _collect_refs(self, ref_list) -> list:
        """Direct ``<ref>`` children plus those of nested ``<ref-list>``s, in order."""
        refs = []
        for child in ref_list:
            ln = _ln(child)
            if ln == "ref":
                refs.append(child)
            elif ln == "ref-list":
                refs.extend(self._collect_refs(child))
        return refs

    def _finalize_references(self) -> None:
        """Decide the structured-vs-strings invariant once, over every ref-list."""
        if self._ref_all_structured and self._ref_structured:
            self._native_references = self._ref_structured
        elif self._ref_rows:
            self._native_ref_strings = self._ref_rows

    def _handle_ref_list(self, ref_list, section_id: int | None = None) -> None:
        if section_id is None:
            self._section_counter += 1
            ref_sid = self._section_counter
            header = _text(_first_child(ref_list, "title")) or "References"
            self.sections.append(
                PaperSection(
                    section_id=ref_sid,
                    header=header,
                    level=1,
                    parent_section_id=0,
                    section_type=CanonicalSection.REFERENCES,
                )
            )
        else:
            ref_sid = section_id
            self._retype_as_references(ref_sid, ref_list)

        # Refs accumulate across every ref-list with one continuous bib_id
        # counter (a later list extends the bibliography); nested groupings
        # contribute their refs and rows to the enclosing list's section.
        for ref in self._collect_refs(ref_list):
            pos = self._ref_next_id
            self._ref_next_id += 1
            element_citation = _first_desc(ref, "element-citation")
            mixed_citation = _first_desc(ref, "mixed-citation")
            mixed_text = _text(mixed_citation) if mixed_citation is not None else ""
            label_el = _first_child(ref, "label")
            label_text = _text(label_el) if label_el is not None else ""
            # A <ref> may carry a <note> beside its citation (ACS "Remarks",
            # access notes) — part of the printed reference text.
            note_text = " ".join(_text(note) for note in _iter_children(ref, "note")).strip()
            if element_citation is not None:
                structured = self._build_reference(element_citation, pos)
                row = mixed_text or _citation_text(element_citation)
                if label_text:
                    row = f"{label_text} {row}".strip() if row else label_text
                if note_text:
                    row = f"{row} {note_text}".strip() if row else note_text
                text = row or _text(ref)
                self._ref_structured.append(structured)
            else:
                # No element-citation: unstructured, whatever else is there.
                # The whole-ref fallback keeps the printed label, as before.
                text = mixed_text
                if label_text and text:
                    text = f"{label_text} {text}"
                if note_text and text:
                    text = f"{text} {note_text}"
                if not text:
                    text = _text(ref)
                self._ref_all_structured = False
            # One atomic sentence per ref (needs_segmentation=False) so each ref
            # stays a single row in the REFERENCES section for RefLocator.
            self.assembler.append(text, None, ref_sid, needs_segmentation=False, is_formula=False)
            self._ref_rows.append(text)

    def _build_reference(self, ec, pos: int) -> PaperReference:
        authors = self._person_names(ec, "author")
        editors = self._person_names(ec, "editor")

        year_text = _text(_first_desc(ec, "year"))
        year: int | None = None
        if year_text:
            digits = "".join(c for c in year_text if c.isdigit())
            if len(digits) >= 4:
                try:
                    year = int(digits[:4])
                except ValueError:
                    year = None

        title = _text(_first_desc(ec, "article-title")) or _text(_first_desc(ec, "chapter-title"))
        container = _text(_first_desc(ec, "source")) or None

        doi = None
        for pid in ec.iter():
            if _ln(pid) == "pub-id" and _attr(pid, "pub-id-type") == "doi":
                doi = _text(pid) or None
                break

        url = None
        ext = _first_desc(ec, "ext-link")
        if ext is not None:
            url = _attr(ext, "href") or _text(ext) or None
        if url is None:
            uri = _first_desc(ec, "uri")
            if uri is not None:
                url = _attr(uri, "href") or _text(uri) or None

        pub_type = _attr(ec, "publication-type")
        bib_type: str | None = None
        if pub_type:
            key = pub_type.lower().strip()
            bib_type = _JATS_PUB_TYPE.get(key) or migrate_bib_type(key)

        return PaperReference(
            bib_id=pos,
            title=title or "",
            first_page=_text(_first_desc(ec, "fpage")) or None,
            volume=_text(_first_desc(ec, "volume")) or None,
            authors=authors or None,
            year=year,
            container=container,
            doi=doi,
            bib_type=bib_type,
            last_page=_text(_first_desc(ec, "lpage")) or None,
            issue=_text(_first_desc(ec, "issue")) or None,
            editors=editors or None,
            publisher=_text(_first_desc(ec, "publisher-name")) or None,
            url=url,
        )

    @staticmethod
    def _person_names(ec, group_type: str) -> str:
        """Join a person-group's names as 'Family, G.; Family, G.'."""
        target = None
        for group in ec.iter():
            if _ln(group) != "person-group":
                continue
            gtype = (_attr(group, "person-group-type") or "author").lower()
            if gtype == group_type:
                target = group
                break
        # Authors may appear without an explicit person-group wrapper — but
        # only fall back to the whole citation when NO person-group exists at
        # all, else an editor group's names would be swept up as authors.
        if target is None and group_type == "author":
            has_any_group = any(_ln(el) == "person-group" for el in ec.iter())
            if not has_any_group:
                target = ec
        if target is None:
            return ""

        names: list[str] = []
        for name in target.iter():
            ln = _ln(name)
            if ln == "name":
                surname = _text(_first_child(name, "surname"))
                given = _text(_first_child(name, "given-names"))
                if surname:
                    names.append(f"{surname}, {given}" if given else surname)
            elif ln == "string-name":
                value = _text(name)
                if value:
                    names.append(value)
        return "; ".join(names)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_urls(
        self, sent: PaperSentence, covered: set[tuple[str, int]] | None = None
    ) -> None:
        """Match URL_RE on the sentence and append PaperURLLink entries.

        *(url, text_id)* pairs in *covered* (anchor targets collected while
        walking paragraphs) are skipped so one link is not recorded twice.
        """
        for m in URL_RE.finditer(sent.text):
            url = clean_extracted_url(m.group(0))
            if covered is not None and (url, sent.text_id) in covered:
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
