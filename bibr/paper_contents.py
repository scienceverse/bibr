"""Wrapper for the content (text) in a scientific paper. We should be able to retrieve it in various
formats - as a text_df, in a section:text dictionary, as a single string etc.

Is built by PDFParser from OCR output."""

import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING

import pandas as pd

from bibr.input.consolidate_text import clean_text_content_late
from bibr.processing_warnings import ProcessingWarning

if TYPE_CHECKING:
    from bibr.extract.front_matter import FrontMatterResolution
    from bibr.extract.front_role import FrontRolePredictions
    from bibr.models import PaperMetadata, PaperReference
    from bibr.validation import ValidationIssue


# section classification (IMRaD+)
class CanonicalSection(StrEnum):
    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "intro"
    METHODS = "method"
    RESULTS = "results"
    DISCUSSION = "discussion"
    REFERENCES = "references"
    ACKNOWLEDGMENT = "acknowledgment"
    FUNDING = "funding"
    KEYWORDS = "keywords"
    ENDNOTE = "endnote"
    APPENDIX = "appendix"
    OPEN_DATA = "open_data"
    AUTHOR_CONTRIBUTIONS = "author_contributions"
    COI = "coi"
    ETHICS = "ethics"
    FOOTNOTE = "footnote"
    TABLE = "table"
    FIGURE = "figure"
    UNKNOWN = "unknown"  # fallback


CANONICAL_SECTION_ALIASES = {
    # Heuristic aliases for lookup-based section classification.
    # All values must be in normalized form (lowercase, no leading numbers/dots)
    # to match the output of normalize_text().
    CanonicalSection.ABSTRACT: [
        "abstract",
        "summary",
        "precis",
        "executive summary",
        # Psych Science / Sage journal pattern: a brief policy-relevance
        # statement that appears alongside the abstract.
        "statement of relevance",
        "significance statement",
        "research highlights",
    ],
    CanonicalSection.INTRODUCTION: [
        "introduction",
        "background",
        "overview",
        "preamble",
        "motivation",
        "context",
        "background and related work",
        "background and motivation",
        "related work",
        "related works",
        "literature review",
        "state of the art",
        "prior work",
        "preliminaries",
    ],
    CanonicalSection.METHODS: [
        "method",
        "methods",
        "materials",
        "methodology",
        "experimental",
        "materials and methods",
        "materials & methods",
        "experimental section",
        "experimental design",
        "experimental setup",
        "study design",
        "procedures",
        "model",
        "data collection",
        "proposed method",
        "proposed approach",
        "implementation",
        "participants",
        "statistical analysis",
        "statistical methods",
        "measures",
        "instruments",
        "data sources",
        "study population",
        "sample",
        "research design",
        "research methodology",
    ],
    CanonicalSection.RESULTS: [
        "results",
        "findings",
        "experimental results",
        "computational results",
        "results and discussion",
        "results & discussion",
        "evaluation",
        "experiments",
        "experiments and results",
        "empirical results",
        "performance evaluation",
        "observations",
        "ablation study",
        "ablation studies",
    ],
    CanonicalSection.DISCUSSION: [
        "discussion",
        "interpretation",
        "limitations",
        "general discussion",
        "limitations and future work",
        "implications",
        "strengths and limitations",
        # Conclusions read more naturally as a flavour of discussion than as
        # an endnote (data-availability/appendix territory).
        "conclusion",
        "conclusions",
        "concluding remarks",
        "summary and conclusions",
        "broader impact",
        "broader impacts",
    ],
    CanonicalSection.REFERENCES: [
        "references",
        "bibliography",
        "works cited",
        "literature cited",
        "citations",
        "cited literature",
    ],
    CanonicalSection.ACKNOWLEDGMENT: [
        "acknowledgments",
        "acknowledgements",
        "declarations",  # umbrella container; specific subsections route to coi/ethics/etc.
        "transparency",
        "action editor",
        # Author-metadata blocks — currently surface as standalone headings
        # because PP-DocLayoutV3 emits them as `paragraph_title` regions.
        # Acknowledgment is the closest existing bucket; ideally these would
        # be parsed into PaperAuthor records and not become sections at all.
        "corresponding author",
        "corresponding authors",
        "orcid id",
        "orcid ids",
    ],
    CanonicalSection.AUTHOR_CONTRIBUTIONS: [
        "author contributions",
        "author contribution",
        "credit authorship contribution statement",
        "credit author statement",
        "contribution statement",
        "authors' contributions",
    ],
    CanonicalSection.COI: [
        "conflict of interest",
        "conflicts of interest",
        "competing interests",
        "competing interest",
        "declaration of interest",
        "declaration of conflicting interests",
        "declaration of competing interest",
    ],
    CanonicalSection.ETHICS: [
        "ethics",
        "ethics statement",
        "ethical approval",
        "irb approval",
        "informed consent",
        "consent to participate",
        "ethics declaration",
        "institutional review board",
    ],
    CanonicalSection.FUNDING: [
        "funding",
        "financial support",
    ],
    CanonicalSection.KEYWORDS: [
        "keywords",
        "key words",
    ],
    CanonicalSection.ENDNOTE: [
        "supplementary materials",
        "supplementary information",
        "supplemental data",
        "extended data",
        "online supplement",
        "annexes",
        "annex",
        "future work",
        "outlook",
        "conclusions and future work",
        "conclusion and outlook",
        "final remarks",
    ],
    CanonicalSection.APPENDIX: [
        "appendix",
        "appendices",
        "supplementary material",
        "supplemental material",
        "supporting information",
    ],
    CanonicalSection.OPEN_DATA: [
        "data availability",
        "data availability statement",
        "open practices",
        "open science",
        "open science statement",
        "code availability",
        "code availability statement",
        "materials availability",
        "data and code availability",
        "data sharing",
        "open data",
        "reproducibility",
        "reproducibility statement",
    ],
    CanonicalSection.FIGURE: [
        "figure",
        "figures",
    ],
    CanonicalSection.TABLE: [
        "table",
        "tables",
    ],
    CanonicalSection.FOOTNOTE: [
        "endnote",
        "endnotes",
        "footnote",
        "footnotes",
        "note",
        "notes",
    ],
}

# Recognize translated section headings so unclassified front-matter labels do not become
# spurious title sections.
_NON_ENGLISH_SECTION_ALIASES = {
    CanonicalSection.ABSTRACT: [
        "abstrak",
        "аннотация",
        "анотація",
        "özet",
        "resumen",
        "resumo",
        "résumé",
        "streszczenie",
        "zusammenfassung",
        "要旨",
        "摘要",
        "초록",
    ],
    CanonicalSection.KEYWORDS: [
        "kata kunci",
        "ключевые слова",
        "ключові слова",
        "anahtar kelimeler",
        "palabras clave",
        "palavras-chave",
        "palavras chave",
        "mots-clés",
        "mots clés",
        "schlagwörter",
        "słowa kluczowe",
        "キーワード",
        "关键词",
        "關鍵詞",
    ],
    CanonicalSection.REFERENCES: [
        "bibliografi",
        "bibliografia",
        "bibliografía",
        "bibliographie",
        "daftar pustaka",
        "literatur",
        "literatura",
        "literaturverzeichnis",
        "referencias",
        "referências",
        "références",
        "kaynakça",
        "литература",
        "список литературы",
        "список використаних джерел",
        "参考文献",
        "참고문헌",
    ],
    CanonicalSection.INTRODUCTION: [
        "pendahuluan",
        "introducción",
        "introdução",
        "einleitung",
        "giriş",
        "введение",
        "вступ",
        "wstęp",
        "はじめに",
        "引言",
    ],
    CanonicalSection.METHODS: [
        "metode penelitian",
        "metodología",
        "metodologia",
        "méthodologie",
        "methodik",
        "yöntem",
        "методы",
        "методика",
        "方法",
    ],
    CanonicalSection.RESULTS: [
        "hasil dan pembahasan",
        "hasil penelitian",
        "resultados",
        "résultats",
        "ergebnisse",
        "bulgular",
        "результаты",
        "результати",
        "结果",
    ],
    CanonicalSection.DISCUSSION: [
        "pembahasan",
        "discusión",
        "discussão",
        "diskussion",
        "tartışma",
        "обсуждение",
        "討論",
        "讨论",
    ],
}

for _section, _aliases in _NON_ENGLISH_SECTION_ALIASES.items():
    CANONICAL_SECTION_ALIASES.setdefault(_section, []).extend(
        alias for alias in _aliases if alias not in CANONICAL_SECTION_ALIASES.get(_section, ())
    )
del _section, _aliases


# Printed page furniture — article-type kickers, badges and info-box labels
# that are neither a section heading nor an article title. The section
# classifier has no vocabulary member for "journal furniture", so on these rows
# it emitted TITLE confidently and they seeded false front-matter records
# (10.1515_ffp-2017-0016 "SHORT COMMUNICATION" stole the DOI row from the real
# record). Lives here rather than in ``bibr.extract.front_matter`` because both
# the structure layer (section classification, running-header detection) and
# the extract layer need it, and structure must not import extract.
FRONT_MATTER_FURNITURE_LABELS = frozenset(
    {
        "article info",
        "article information",
        "check for updates",
        "copyright",
        "how to cite this article",
        "open access",
        "original article",
        "original research",
        "research article",
        "short communication",
        "you may also like",
    }
)


# Printed journal/proceedings mastheads. Like the furniture labels above, these
# are needed both in the structure layer (a masthead repeated as a page banner
# must stay demoted) and in the extract layer (it must never be selected as the
# article title).
FRONT_MATTER_MASTHEAD_RE = re.compile(
    r"^(?:program abstracts?|proceedings|session\s+\d+\b|table of contents|contents\b|"
    r"in\s+this\s+issue\b|journal of\b|annual (?:scientific )?meeting\b)",
    re.IGNORECASE,
)


def is_exact_front_matter_furniture(text: str) -> bool:
    """Anchored match: the whole row is a printed furniture label, not content.

    Anchored full-string matching only — a substring of a genuine title must
    never match.
    """

    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return normalized.strip(" :.") in FRONT_MATTER_FURNITURE_LABELS


@dataclass
class Provenance:
    """Spatial source of a parsed item on a source page.

    ``bbox`` follows glmocr's normalised 0–1000 coordinate system as
    ``(x1, y1, x2, y2)``.  Multiple instances per node are allowed when
    content spans pages or non-contiguous regions (e.g. column wraps).
    Internal-only: never exported in the JSON schema.
    """

    page_no: int
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class CaptionCandidate:
    """One source caption region considered for document-wide ownership."""

    caption_id: str
    text: str
    object_type: str
    page_number: int | None
    bbox: tuple[float, float, float, float] | None
    source_index: int


@dataclass(frozen=True)
class CaptionAssignment:
    """Deterministic caption ownership decision, including abstentions."""

    caption_id: str
    object_id: str | None
    score: float
    reasons: tuple[str, ...]
    ambiguous: bool = False


@dataclass(frozen=True)
class CaptionAssignmentReceipt:
    """Lossless caption candidates and their document-wide decisions."""

    candidates: tuple[CaptionCandidate, ...]
    assignments: tuple[CaptionAssignment, ...]


@dataclass
class PaperSection:
    """
    A flat representation of the Table of Contents.

    ``children`` is populated post-parse from ``parent_section_id`` to
    expose the tree without breaking the existing flat list contract.
    """

    section_id: int  # The unique ID (0=Root, then 1, 2, ...)
    header: str  # "Data Collection"
    level: int  # 1–6 (numbering-aware); 0 reserved for Root. Internal-only.
    parent_section_id: int | None  # Points to the containing section
    section_type: CanonicalSection = CanonicalSection.UNKNOWN
    classification_score: float = 0.0
    # Provenance of the classification decision — which tier produced
    # ``section_type``/``classification_score``: "exact_alias", "substring_alias",
    # "alias_prior", "model", "llm", "parent_context", "title", "implicit",
    # "positional", "appendix_repair", or "imrad_dedup" (type reset because a
    # higher-trust duplicate kept it). None when unclassified. Exported, with
    # the score, as ``extraction.diagnostics.section_classification``.
    classification_source: str | None = None
    # Optional signal from the trained section classifier — True/False when the
    # MiniLM head produced a prediction, None when no signal was available
    # (alias-lookup hit, LLM-only path, or no_llm). Consumed by
    # ``assign_hierarchy_from_top_level`` in ``section_tree.py``.
    is_top_level_predicted: bool | None = None
    # Transient flag: True when this section's ``level`` was assigned from a
    # confidently matched PDF-outline (bookmark) entry. Signals
    # ``assign_hierarchy_from_top_level`` to treat the level/parent as
    # authoritative (like a numbered heading) and not overwrite it. Never
    # serialized (json_export builds SectionExport from named fields).
    outline_level_authoritative: bool = False
    provenance: list[Provenance] = field(default_factory=list)
    children: list["PaperSection"] = field(default_factory=list, repr=False, compare=False)
    # Internal provenance: an inferred label is not text printed in the document.
    # Keep this independent of classification_source, which later tiers overwrite.
    header_is_synthetic: bool = False
    # "figure", "table" or "footnote" on the section ``create_content_sections``
    # makes to hold one caption or footnote. The export has no such sections:
    # their sentences become the caption and footnote rows that ``figure``,
    # ``table`` and ``footnote`` point at.
    synthetic_kind: str | None = None
    # On a footnote's synthetic section: the marker the note is printed with
    # ("1", "*", "†"); None when none is printed or detected.
    footnote_label: str | None = None


@dataclass
class PaperSentence:
    """
    Lightweight sentence object.
    """

    text_id: int
    text: str
    section_id: int  # originating section
    paragraph_id: int  # originating paragraph
    page_number: int | None = None  # source page (1-based, PDF only)
    is_display_formula: bool = False  # True when this sentence holds a display-math formula
    # Source bboxes for the underlying paragraph(s).  All sentences split from a
    # single paragraph share the same provenance list — sub-sentence bbox is
    # not derivable from region-level OCR output.  Empty for DOCX-native input.
    provenance: list["Provenance"] = field(default_factory=list)
    # Training-side region metadata sourced from the first contributing layout region.
    # Keys: font_size, font_bold, is_italic, bbox (0..1000 layout space),
    # region_type, plus region_page/region_index: that region's RegionSummary
    # ``(page, index)``.
    # None for DOCX-native input or when font metadata was not extracted.
    region_meta: dict | None = None
    # Whether any of the text may come from OCR. ``finalize_text`` repairs OCR
    # artifacts only there. Native parsers (DOCX, JATS, HTML, ePub) and PDF
    # paragraphs built only from the embedded text layer set False; unknown
    # provenance keeps the default, so it still gets the OCR repairs.
    from_ocr: bool = True


@dataclass
class RegionSummary:
    """Compact summary of a layout-detected region, preserved for analysis."""

    page: int
    index: int
    label: str
    bbox: tuple[float, float, float, float] | None
    font_size: float | None = None
    font_weight: int | None = None
    font_bold: bool | None = None
    section_id: int | None = None
    content: str | None = None
    canonical_ocr_content: str | None = None
    raw_ocr_content: str | None = None
    native_text_candidate: str | None = None
    native_text_rejection_reason: str | None = None
    bbox_height: float | None = None
    bbox_width: float | None = None
    char_density: float | None = None
    estimated_line_height: float | None = None


@dataclass
class PaperURLLink:
    url: str
    section_id: int  # originating section
    paragraph_id: int  # originating paragraph
    text_id: int  # originating sentence
    link_text: str | None = None  # text of the hyperlink (None for plain-text URLs)


@dataclass
class PaperTablePart:
    """One physical table region retained inside a logical table."""

    page_number: int | None
    bbox: tuple[float, float, float, float] | None
    tbl_html: str | None
    df: pd.DataFrame
    provenance: list[Provenance] = field(default_factory=list)

    @property
    def contents(self) -> list[list[str]]:
        if self.df.empty:
            return []
        return [[str(c) for c in self.df.columns.tolist()]] + [
            [str(value) for value in row] for row in self.df.values.tolist()
        ]


@dataclass(eq=False)
class PaperTable:
    table_id: int
    df: pd.DataFrame
    tbl_html: str
    section_id: int
    caption: str | None = None
    page_number: int | None = None  # source page (1-based, PDF only)
    provenance: list[Provenance] = field(default_factory=list)
    # Body section where the table was originally declared (preserved before
    # ``create_content_sections`` reassigns ``section_id`` to the table's own
    # synthetic section). Exported as the table's ``section_id``.
    _body_section_id: int | None = field(default=None)
    parts: list[PaperTablePart] = field(default_factory=list)
    # Printed label without the word ("3", "3.1", "S2", "IV"), from the
    # caption (``bibr.structure.float_labels``); in-text mentions resolve by it.
    label: str | None = None

    @property
    def contents(self) -> list[list[str]]:
        """Convert DataFrame to ``[headers_row, *data_rows]`` with all values stringified."""
        if self.df.empty:
            return []
        headers = [str(c) for c in self.df.columns.tolist()]
        data = [[str(v) for v in row] for row in self.df.values.tolist()]
        return [headers] + data


@dataclass
class PaperFigurePart:
    """One physical figure/chart region retained inside a logical figure."""

    page_number: int | None
    bbox: tuple[float, float, float, float] | None
    image_b64: str | None
    provenance: list[Provenance] = field(default_factory=list)


@dataclass
class PaperFigure:
    figure_id: int
    section_id: int  # section where figure appears
    image_b64: str | None  # base64-encoded JPEG of the cropped figure region
    caption: str | None  # alt text / caption
    page_number: int | None = None  # source page (1-based, PDF only)
    provenance: list[Provenance] = field(default_factory=list)
    # Body section where the figure was originally declared (preserved before
    # ``create_content_sections`` reassigns ``section_id`` to the figure's own
    # synthetic section). Exported as the figure's ``section_id``.
    _body_section_id: int | None = field(default=None)
    parts: list[PaperFigurePart] = field(default_factory=list)
    # Printed label without the word ("3", "3.1", "S2", "A1"), from the
    # caption (``bibr.structure.float_labels``); in-text mentions resolve by it.
    label: str | None = None


@dataclass
class PaperXref:
    """Cross-reference linking a sentence to a referenced item."""

    # ID of the referenced item: bib_id, table_id, figure_id, or the text_id of
    # the footnote's sentence (the export turns it into a footnote_id); 0 for a
    # table or figure reference that names no extracted float (or two). For
    # equation, section and supplementary references it is the number they
    # print (0 when none), which the export does not publish.
    xref_id: int
    xref_type: str  # "bib", "table", "figure", "foot", "supplementary", "equation", "section"
    contents: str  # The reference text as it appears (e.g., "[1]", "Table 2", "Figure 3")
    text_id: int  # The sentence containing this reference
    # How the xref was linked: for bib xrefs the detection tier ("numeric",
    # "paren-numeric", "flattened-superscript", "author-year", "llm"), for
    # table/figure xrefs "label" or "position" (``detect_xrefs``); None for
    # other types and until the citation linker records it. Exported as
    # ``extraction.diagnostics.xref_tier``.
    tier: str | None = None
    # Character span of the reference in the sentence text, when the detector
    # matched it there; the exporter verifies it (and locates ``contents``
    # itself when absent) before emitting ``xref[].start``/``end``.
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class CitationCandidate:
    """Evidence-bearing inline-citation span considered by the linker."""

    text_id: int
    start: int
    end: int
    raw: str
    style: str
    bib_ids: tuple[int, ...]
    evidence: tuple[str, ...]
    confidence: float
    accepted: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CitationLinkingReceipt:
    """Internal diagnostic record for one paper's citation-linking pass."""

    style_scores: dict[str, float]
    candidates: tuple[CitationCandidate, ...]
    resolved_candidate_fraction: float | None
    unique_linked_bib_fraction: float | None


@dataclass(frozen=True)
class ReferenceSegmentationAttempt:
    """One evidence-bearing tier considered during reference segmentation."""

    strategy: str
    spans: tuple[tuple[int, int], ...]
    credible_starts: int | None
    selected: bool
    reason_flags: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceYieldReceipt:
    """Internal diagnostic record for reference segmentation and parse yield."""

    credible_source_starts: int | None
    attempts: tuple[ReferenceSegmentationAttempt, ...]
    selected_spans: tuple[tuple[int, int], ...]
    source_character_coverage: float | None
    parsed_count: int
    valid_count: int
    duplicate_rate: float
    reason_flags: tuple[str, ...]


@dataclass
class PaperEquation:
    """A decomposed equation component extracted from a sentence."""

    text_id: int  # sentence containing this expression
    grp_id: int  # globally unique group ID (1-based)
    lhs: str  # left-hand side statistic name (e.g., "t", "p", "α")
    comp: str  # comparison operator ("=", "<", ">", "≤", "≥", "≈")
    rhs: str  # right-hand side (e.g., "3.42", ".003", "[2.0, 4.7]")
    df: str = ""  # degrees of freedom from the LHS parenthetical (e.g., "28" for t(28))


@dataclass
class PaperContents:
    sentences: list[PaperSentence]
    sections: list[PaperSection]
    tables: list[PaperTable]
    links: list[PaperURLLink]
    sections_text: dict[int, str]
    figures: list[PaperFigure] = field(default_factory=list)
    xrefs: list[PaperXref] = field(default_factory=list)
    citation_receipt: CitationLinkingReceipt | None = None
    equations: list[PaperEquation] = field(default_factory=list)
    detected_title: str | None = None
    detected_headers: list[str] = field(default_factory=list)
    detected_footers: list[str] = field(default_factory=list)
    layout_hints: list[tuple[str, int]] = field(default_factory=list)
    region_summaries: list[RegionSummary] = field(default_factory=list)
    # Page size in PDF points (as displayed) by 1-based page number; empty for
    # inputs without pages. The frame of the exported bounding boxes.
    page_sizes: dict[int, tuple[float, float]] = field(default_factory=dict)
    # Immutable record-boundary IR built after section classification and
    # before implicit-section normalization mutates the section structure.
    # Internal-only; metadata consumers opt into it incrementally.
    front_matter_resolution: "FrontMatterResolution | None" = None
    # Per-region role scores from the optional front-role classifier
    # (bibr/extract/front_role.py), keyed by (page_number, region index) —
    # the same key RegionSummary carries. None when the model is disabled or
    # the input had no OCR regions (native DOCX/JATS/HTML). Internal-only.
    front_role_predictions: "FrontRolePredictions | None" = None
    # Warnings recorded during content-level extraction (e.g. reference
    # segmentation falling back to CRF); surfaced onto
    # ``Paper.processing_warnings`` in post_parse.
    processing_warnings: list[ProcessingWarning] = field(default_factory=list)
    # Per-line reference-section geometry (serialized LineRecords) captured in
    # the OCR-stage native-text pass; consumed by the geom segmenter in extract.
    # None for DOCX / non-native / no-text-layer input (→ LLM cascade).
    ref_line_geometry: list[dict] | None = None
    # Front-matter metadata parsed natively from a structured input format
    # (JATS XML); when set, post-parse uses it as the PaperMetadata base and
    # skips the core LLM extraction. None for PDF/DOCX (→ LLM extraction).
    preparsed_metadata: "PaperMetadata | None" = None
    # References parsed directly from structured element-citations (JATS); when
    # set, post-parse assigns them to metadata.references and skips the
    # ReferenceExtractor entirely. None otherwise.
    native_references: "list[PaperReference] | None" = None
    # Pre-segmented reference strings from unstructured mixed-citations (JATS):
    # each entry is one full reference. Consumed by the ``native`` segmentation
    # branch so LLM/geom segmentation is skipped, then parsed as configured.
    native_ref_strings: list[str] | None = None
    # Internal diagnostics appended at the tail to preserve positional callers.
    reference_yield_receipt: ReferenceYieldReceipt | None = None
    reference_boundary_reason_flags: list[str] = field(default_factory=list)
    structure_validation_issues: list["ValidationIssue"] = field(default_factory=list)
    caption_assignment_receipt: CaptionAssignmentReceipt | None = None

    def invalidate_text_caches(self) -> None:
        """Drop cached DataFrames whose contents derive from sentence text or links.

        Call after any in-place mutation of ``self.sentences``, ``self.links``,
        or ``self.equations`` so subsequent ``*_df`` reads rebuild from current
        state.
        """
        self.__dict__.pop("sentences_df", None)
        self.__dict__.pop("text_df", None)
        self.__dict__.pop("links_df", None)
        self.__dict__.pop("equations_df", None)

    def finalize_text(self) -> None:
        """Run late-phase text cleaning on all sentences.

        Strips inline math delimiters (``$...$``) and flattens LaTeX
        commands (``^{}``, ``\\alpha``, etc.) that were intentionally
        preserved during parsing so that citation linking, equation
        extraction, and xref detection could operate on the raw patterns.
        Outside ``$...$`` spans only sentences with OCR text are cleaned
        (``PaperSentence.from_ocr``; see ``clean_text_content_late``).

        Display-formula sentences are skipped — their LaTeX content is
        the actual data and should not be cleaned.

        Must be called **after** all extraction stages (which access
        ``sentences_df``).  Sentence-derived caches are invalidated so
        ``text_df`` and ``sentences_df`` reflect the cleaned text on next
        access.
        """
        for sent in self.sentences:
            if not sent.is_display_formula:
                sent.text = clean_text_content_late(sent.text, from_ocr=sent.from_ocr)
        self.invalidate_text_caches()

    @cached_property
    def links_df(self) -> pd.DataFrame:
        """Populate a flattened DataFrame with all URL links."""
        rows = []
        for link in self.links:
            rows.append(
                {
                    "url": link.url,
                    "link_text": link.link_text,
                    "text_id": link.text_id,
                }
            )
        if not rows:
            return pd.DataFrame(columns=["url", "link_text", "text_id"])
        return pd.DataFrame(rows)

    @cached_property
    def text_df(self) -> pd.DataFrame:
        """Export view — all sections including bibliography."""
        rows = []
        for sent in self.sentences:
            text = "[equation]" if sent.is_display_formula else sent.text
            rows.append(
                {
                    "text_id": sent.text_id,
                    "section_id": sent.section_id,
                    "paragraph_id": sent.paragraph_id,
                    "text": text,
                    "page_number": sent.page_number,
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=["text_id", "section_id", "paragraph_id", "text", "page_number"]
            )
        return pd.DataFrame(rows)

    @cached_property
    def equations_df(self) -> pd.DataFrame:
        """DataFrame of decomposed equations extracted from sentences."""
        rows = []
        for eq in self.equations:
            rows.append(
                {
                    "text_id": eq.text_id,
                    "grp_id": eq.grp_id,
                    "lhs": eq.lhs,
                    "df": eq.df,
                    "comp": eq.comp,
                    "rhs": eq.rhs,
                }
            )
        return pd.DataFrame(rows)

    @cached_property
    def sentences_df(self) -> pd.DataFrame:
        """Internal view — includes ALL sections (needed by extractor for reference parsing).

        The extractor uses sentences_df with section_name lookups. This builds
        a denormalized view with section headers for that purpose.
        """
        section_map = {s.section_id: s for s in self.sections}
        rows = []
        for sent in self.sentences:
            sec = section_map.get(sent.section_id)
            rows.append(
                {
                    "text_id": sent.text_id,
                    "section_id": sent.section_id,
                    "paragraph_id": sent.paragraph_id,
                    "text": sent.text,
                    "section_name": sec.header if sec and sec.level > 0 else None,
                    "section_type": (
                        sec.section_type if sec is not None else CanonicalSection.UNKNOWN
                    ),
                    "page_number": sent.page_number,
                }
            )
        return pd.DataFrame(rows)
