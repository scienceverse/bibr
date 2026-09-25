"""PDF-native parser: converts glmocr json_result → PaperContents.

All file types (PDF and DOCX-converted-to-PDF) are processed through this
parser after OCR.

The glmocr SDK returns ``json_result`` as a list of pages, where each page is
a list of region dicts::

    [
        [  # page 1
            {"index": 0, "label": "doc_title", "content": "My Paper", "bbox_2d": [...]},
            {"index": 1, "label": "text", "content": "Some text.", "bbox_2d": [...]},
            ...
        ],
        [  # page 2
            ...
        ],
    ]

Labels are mapped to treatments via ``LABEL_TREATMENT`` which determines how
each region is processed (heading, body text, table, formula, figure, etc.).

The per-region treatment handlers live in cohesive mixins:
:class:`~bibr.structure.parse_headings.HeadingHandlersMixin`,
:class:`~bibr.structure.parse_media.MediaHandlersMixin`, and
:class:`~bibr.structure.parse_text.TextHandlersMixin`. ``PDFParser`` owns the
shared state, the ``_process_page`` dispatch, and running-header marking.
"""

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.input.pdf_outline import OutlineItem

from bibr.input.consolidate_text import fix_ocr_artifacts
from bibr.ocr.ref_patterns import alnum_key, alnum_text_covered
from bibr.ocr.types import OcrRegionResult
from bibr.paper_contents import (
    FRONT_MATTER_MASTHEAD_RE,
    CanonicalSection,
    PaperContents,
    PaperFigure,
    PaperSection,
    PaperSentence,
    PaperTable,
    PaperURLLink,
    PaperXref,
    RegionSummary,
    is_exact_front_matter_furniture,
)
from bibr.processing_warnings import ProcessingWarning, WarningCode
from bibr.structure.assembler import DocumentAssembler
from bibr.structure.carry_over_manager import CarryOverState
from bibr.structure.float_labels import LABEL, SUPPLEMENT_WORD
from bibr.structure.floats_normalize import (
    merge_figure_panels_with_remap,
    merge_table_continuations_with_remap,
    remap_caption_receipt,
)
from bibr.structure.footnote_buffer import FootnoteBuffer, printed_marker
from bibr.structure.parse_headings import HeadingHandlersMixin
from bibr.structure.parse_media import MediaHandlersMixin
from bibr.structure.parse_text import TextHandlersMixin
from bibr.structure.text_repair import (
    bbox_to_tuple,
    strip_markdown_emphasis,
)
from bibr.structure.xref_utils import detect_xrefs
from bibr.utils.text import OCR_CORRUPTION_MIN_CHARS, ocr_corruption_count

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label → treatment mapping
# ---------------------------------------------------------------------------
# Each glmocr region label is mapped to one of the treatment categories below.
# Treatment determines how the region content is turned into PaperContents.

LABEL_TREATMENT: dict[str, str] = {
    # headings → PaperSection
    "doc_title": "heading",
    "paragraph_title": "heading",
    # body text → PaperSentence
    "text": "content",
    "content": "content",
    "vertical_text": "content",
    "seal": "content",
    "algorithm": "content",
    # section hints → implicit section + body text
    "abstract": "section_hint",
    "reference": "section_hint",
    "reference_content": "section_hint",
    # footnotes → deferred footnote sections
    "footnote": "footnote",
    "vision_footnote": "footnote",
    # tables → PaperTable (markdown → DataFrame)
    "table": "table",
    # table captions (not produced by PP-DocLayoutV3, kept for compat)
    "table_title": "table_caption",
    # formulas → wrap in $$...$$ and treat as body text
    "display_formula": "formula",
    "inline_formula": "formula",
    "formula": "formula",
    "formula_number": "content",
    # figures → PaperFigure
    "chart": "figure",
    "image": "figure",
    # captions — figure_title is the only caption label PP-DocLayoutV3
    # produces; smart-routed to figure or table handler based on content.
    # chart_title kept for compat (not produced by PP-DocLayoutV3).
    "figure_title": "caption",
    "chart_title": "caption",
    # structural → metadata (headers/footers)
    "header": "structural",
    "footer": "structural",
    # abandoned → discard silently
    "number": "abandon",
    "header_image": "abandon",
    "footer_image": "abandon",
    "aside_text": "abandon",
}

# Body-text labels eligible for running-header de-duplication. A running
# header misclassified as body text repeats verbatim across pages; demoting it
# keeps page furniture out of section content (notably the References block).
_BODY_TEXT_LABELS: frozenset[str] = frozenset({"text", "content", "vertical_text"})

# Max normalized length of a body region treated as a candidate running
# header. Furniture (banners, footers, watermarks) is short; a real paragraph
# that happens to repeat across pages is longer and must be preserved.
_RUNNING_HEADER_MAX_LEN = 200
# Running heads are page furniture: they sit in the top or bottom margin band.
# ``bbox_2d`` is 0..1000 image space, matching the page-edge test in
# ``parse_media``. A heading in the middle of a column is never furniture,
# however often it repeats.
_RUNNING_HEADER_TOP_Y = 100.0
_RUNNING_HEADER_BOTTOM_Y = 900.0


def _bbox_containment_fraction(inner: list | tuple | None, outer: list | tuple | None) -> float:
    """Fraction of *inner* covered by *outer*; zero for malformed boxes."""
    inner_box = bbox_to_tuple(inner)
    outer_box = bbox_to_tuple(outer)
    if inner_box is None or outer_box is None:
        return 0.0
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    inner_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inner_area <= 0:
        return 0.0
    overlap = max(0.0, min(ix2, ox2) - max(ix1, ox1)) * max(0.0, min(iy2, oy2) - max(iy1, oy1))
    return overlap / inner_area


class PDFParser(HeadingHandlersMixin, MediaHandlersMixin, TextHandlersMixin):
    """Parses glmocr ``json_result`` into a :class:`PaperContents` object.

    Sentence segmentation is always deferred — call :meth:`apply_segmentation`
    after :meth:`parse` with externally-produced segments.

    Region-handler behaviour is split across
    :class:`~bibr.structure.parse_headings.HeadingHandlersMixin`,
    :class:`~bibr.structure.parse_media.MediaHandlersMixin`, and
    :class:`~bibr.structure.parse_text.TextHandlersMixin`; this class owns the
    shared parse state and the ``_process_page`` dispatch.

    Parameters
    ----------
    json_result : list[list[OcrRegionResult | dict]]
        Pages of OCR'd regions in reading order. The pipeline passes typed
        :class:`~bibr.ocr.types.OcrRegionResult` objects; wire-format dicts
        (frozen eval JSONs, fixtures) are normalized at entry via
        ``OcrRegionResult.from_dict``.
    """

    # Regex to detect table/figure captions embedded in *content* regions.
    # A separator (":", ".", dash, or "|") is REQUIRED here so prose like
    # "Table 7 shows the results." is not misrouted as a caption.  The pipe
    # covers backends that emit "Table 1 | Caption".  Case-insensitive.
    #
    # The id may be hierarchical ("Table 7.5: Descriptives"), so it consumes
    # its own dotted parts BEFORE the separator alternation, and a "." only
    # separates when it is not a decimal point.  Without that lookahead the
    # regex is satisfied by the decimal in ordinary prose — "Table 7.5 shows
    # the results" reads as id "7" + separator "." + caption "5 shows the
    # results" — and ``_handle_content`` then re-routes that sentence into
    # caption ownership, dropping it from the body text entirely.
    #
    # The id is any printed label (``bibr.structure.float_labels.LABEL``):
    # "Table S2:", "Figure A1.", "TABLE IV.", "Supplementary Table 4:".
    #
    # Shared by the heading gate (HeadingHandlersMixin._is_implausible_heading),
    # the content caption re-router (TextHandlersMixin._handle_content), and the
    # loose caption discriminator (MediaHandlersMixin._handle_caption), so it
    # stays on PDFParser and the mixins reach it via ``self``.
    _TABLE_CAPTION_RE = re.compile(
        rf"^({SUPPLEMENT_WORD}?Table\s+{LABEL}\s*(?:[:–\-—|]|\.(?!\d)).*)$",
        re.IGNORECASE | re.DOTALL,
    )
    _FIGURE_CAPTION_RE = re.compile(
        rf"^({SUPPLEMENT_WORD}?(?:Figure|Fig\.?)\s+{LABEL}\s*(?:[:–\-—|]|\.(?!\d)).*)$",
        re.IGNORECASE | re.DOTALL,
    )

    # Loose discriminator used ONLY on regions the layout model already labelled
    # as a caption (figure_title/chart_title).  No separator required — some
    # backends emit "Table 1 Overview" (space) or "Table 1 | Overview" (pipe).
    # Any printed label counts, so "Table S1" is routed to the tables too.
    _LOOSE_TABLE_CAPTION_RE = re.compile(
        rf"^{SUPPLEMENT_WORD}?Table\s+(?:{LABEL}|contin(?:ued|uation)\b)", re.IGNORECASE
    )

    def __init__(
        self,
        json_result: list[list[OcrRegionResult | dict]],
        outline: "list[OutlineItem] | None" = None,
        *,
        settings=None,
        first_page_index: int = 0,
    ) -> None:
        from bibr.config import snapshot_settings

        self._settings = settings if settings is not None else snapshot_settings()
        # Absolute 0-based index of the first *processed* page. OCR pads
        # ``json_result`` with empty pages so a list index stays the absolute
        # PDF page (export provenance depends on that), which means every
        # front-matter heuristic phrased as ``page_number == 1`` silently
        # never fires under page slicing (``--pages 5-12``, ``chew(pages=…)``,
        # serve ``start_page``). Those heuristics now ask
        # :meth:`_is_front_page` instead, so "front page" is threaded once
        # rather than re-derived from an absolute index in five places.
        self._first_page_index = max(0, first_page_index)
        # Normalize to the typed Region IR at entry: the pipeline hands over
        # OcrRegionResult objects; dict pages (frozen eval JSONs, hand-built
        # fixtures) go through the wire-format adapter.
        self.json_result: list[list[OcrRegionResult]] = [
            [r if isinstance(r, OcrRegionResult) else OcrRegionResult.from_dict(r) for r in page]
            for page in json_result
        ]
        # Deferred-text buffer + sentence-emission driver shared with DocxParser.
        # Each entry carries its own optional provenance (source-region bboxes,
        # inherited by every sentence split from it) and region_meta (font_size,
        # font_bold, bbox, region_type, is_italic from the
        # first contributing layout region — used by the v4 training features —
        # plus that region's region_page/region_index key)
        # side-channels, so they stay aligned with the text by construction.
        # When needs_segmentation is False the text is emitted as a single
        # sentence (formulas, reference entries) without the sentence segmenter.
        self.assembler = DocumentAssembler()

        # Output collections
        self.sections: list[PaperSection] = []
        self.sentences: list[PaperSentence] = []
        self.links: list[PaperURLLink] = []
        self.tables: list[PaperTable] = []
        # Pending footnotes: each record is (text, page_number,
        # body_section_id, deferred_text_index). Converted to sections +
        # sentences by create_content_sections().
        self._footnotes = FootnoteBuffer()
        self.figures: list[PaperFigure] = []
        self.detected_headers: list[str] = []
        self.detected_footers: list[str] = []
        self.layout_hints: list[tuple[str, int]] = []
        self.region_summaries: list[RegionSummary] = []
        # Page size in PDF points (as displayed) by 1-based page number, from
        # the native pass's ``_page_w``/``_page_h``: the frame the export
        # converts 0..1000 layout boxes into (bibr.export.geometry).
        self.page_sizes: dict[int, tuple[float, float]] = {}
        self._clean_region_content: dict[tuple[int, int], str] = {}
        # Regions whose raw text carried corrupted-OCR control chars (counted
        # pre-strip in _process_page); surfaced as one OCR_CONTROL_CHARS
        # processing warning by parse().
        self._corrupt_region_count = 0
        # Citation-critical regions recovered by OCR after rejecting embedded
        # private-use glyphs. Surfaced as one ordinary processing warning.
        self._native_text_pua_fallback_count = 0
        # Table regions whose content could not be parsed into a PaperTable
        # even after salvage; surfaced as one OCR_TABLE_DROPPED processing
        # warning by parse().
        self._dropped_table_count = 0

        # Counters (all 1-based except section_id=0 for Root)
        self._section_counter = 0
        self._sentence_counter = 1
        self._paragraph_counter = 0
        self._table_counter = 1
        self._figure_counter = 1
        self._current_section_id = 0
        # Once unmistakable publisher boilerplate closes References, later
        # layout rows may still be mislabeled reference_content. Keep their
        # ownership pinned to the new back-matter section until a real heading
        # establishes a subsequent section.
        self._terminal_reference_tail_section_id: int | None = None

        # State for cross-page continuity. The C6 invariant — that
        # section_id captured at append time survives across a heading
        # change before flush — is enforced by ``CarryOverState``.
        self._carry_over = CarryOverState()

        # Caption-matching state: pending captions awaiting a target
        # figure/table region, plus retro-attachment trackers for the last
        # captionless target.
        self._caption_candidates = []
        self._caption_display_text_by_id: dict[str, str] = {}
        # Every caption candidate id -> the id of the candidate that survived
        # de-duplication for its cluster. Ids recorded at parse time can be
        # de-duplicated away, so anything holding one must resolve it here.
        self._caption_canonical_by_id: dict[str, str] = {}
        self._table_caption_fragments: dict[str, str] = {}
        self._confirmed_table_caption_owners: dict[str, int] = {}
        self._non_caption_candidate_reasons: dict[str, tuple[str, ...]] = {}
        # caption_id → section the candidate was printed under, recorded at
        # capture time so an unowned candidate can be replayed as body text
        # under its own section rather than whatever section parsing ended in.
        self._caption_candidate_sections: dict[str, int] = {}
        self._figure_source_indices: dict[int, int] = {}
        self._table_source_indices: dict[int, int] = {}
        self._structure_validation_issues = []
        self._caption_barriers: list[int] = []
        self._source_region_index = -1
        # A bare ``Table N`` label may compose with exactly the next content
        # region, but that fragment stays provisional until the following
        # parseable table confirms ownership.
        self._pending_bare_table_label: tuple[str, list | None, int, str] | None = None
        self._pending_table_caption_fragment: tuple | None = None

        # Section hints already created (avoid duplicates)
        self._created_hint_sections: set[str] = set()
        # Lowercase header → section_id for fast hint section lookup
        self._hint_section_lookup: dict[str, int] = {}

        # Section IDs created by section hints (Abstract, References, etc.)
        # These should not act as parents for heading-created sections.
        self._hint_section_ids: set[int] = set()

        # First doc_title on page 1 (used as authoritative paper title)
        self._detected_title: str | None = None

        # Running-header de-dupe: set of (page_idx_0based, region_index_within_page)
        # tuples for heading regions whose normalized content appears on multiple
        # pages — those are running headers misclassified as ``doc_title`` /
        # ``paragraph_title`` and must not produce sections.
        self._running_header_regions: set[tuple[int, int]] = set()
        # Page-level ``reference`` envelopes and the ``reference_content``
        # entries inside them can carry the same text. Whichever side the other
        # already covers is shadowed (see ``_mark_reference_envelopes``).
        self._shadowed_reference_regions: set[tuple[int, int]] = set()

        # PDF outline (bookmarks) — the document's own declared heading
        # hierarchy. When present AND the feature is enabled, matched headings
        # take the bookmark level (see :meth:`_apply_outline_hierarchy`). None
        # (the default, and for DOCX/non-native input) keeps the feature inert.
        self._outline = outline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def _deferred_texts(self) -> list[tuple[str, int, int, bool, bool]]:
        """Legacy 5-tuple view of the deferred-text buffer (read-only).

        The source of truth is :attr:`assembler`; this projection keeps the
        historical ``(text, page, section_id, needs_seg, is_formula)`` contract
        that tests and analysis scripts unpack.
        """
        return [
            (e.text, e.page_number, e.section_id, e.needs_segmentation, e.is_formula)
            for e in self.assembler.entries
        ]

    @property
    def _deferred_last_text_id(self) -> list[int | None]:
        """Last text_id per deferred entry (footnote xref linking).

        Proxies :attr:`DocumentAssembler.last_text_id`; writable so tests can
        seed it before exercising the footnote path directly.
        """
        return self.assembler.last_text_id

    @_deferred_last_text_id.setter
    def _deferred_last_text_id(self, value: list[int | None]) -> None:
        self.assembler.last_text_id = value

    @property
    def _first_page_number(self) -> int:
        """1-based page number of the first processed page."""
        return self._first_page_index + 1

    def _is_front_page(self, page_number: int) -> bool:
        """Is ``page_number`` the first page this parse actually saw?

        Front-matter heuristics (title capture, byline affiliation-marker
        stripping, the page-1 decoration filter) mean "the first page we are
        looking at", not "absolute PDF page 1". Under page slicing those
        differ; unsliced they are identical, so the default is unchanged.
        """
        return page_number == self._first_page_number

    def parse(self) -> PaperContents:
        """Run the full parse and return :class:`PaperContents`."""
        # Create root section (section_id=0)
        self.sections.append(
            PaperSection(
                section_id=0,
                header="Root",
                level=0,
                parent_section_id=None,
            )
        )

        # Identify running headers across pages so they do not break section
        # structure when the layout model labels them as ``doc_title`` or
        # ``paragraph_title``.
        self._mark_running_headers()
        self._mark_reference_envelopes()

        for page_idx, page_regions in enumerate(self.json_result):
            page_number = page_idx + 1  # 1-based
            self._process_page(page_regions, page_number, page_idx=page_idx)

        # A provisional caption fragment with no following table remains body
        # content; replay it before flushing cross-region carry-over.
        self._rollback_pending_table_caption_fragment()
        self._expire_pending_table_label_fragment()

        # Caption ownership and multipart grouping require the complete
        # document. Finalize after provisional table-fragment rollback, but
        # before carry-over flush can mutate region ordering/text state.
        caption_assignment_receipt = self._finalize_media()

        # A caption candidate that found no owner is otherwise deleted — its
        # text reaches the export only through its figure/table. Put it back as
        # body content before the assembler closes.
        self.replay_unowned_captions(caption_assignment_receipt)

        # Flush any remaining carry-over text (routes through _flush_carry_over
        # so the originating section_id is honored — see C6).
        self._flush_carry_over()

        # Apply the PDF outline (bookmarks) as the authoritative heading
        # hierarchy when enabled — overrides numbering inference for matched
        # headings. Inert when there is no outline or the feature is off.
        self._apply_outline_hierarchy()

        # Collapse per-panel figures and per-page continuation tables before
        # detect_xrefs / create_content_sections consume the ids.
        self.figures, figure_remap = merge_figure_panels_with_remap(self.figures)
        self.tables, table_remap = merge_table_continuations_with_remap(self.tables)

        # Both mergers renumber survivors from 1, discarding the printed-id
        # reservation _finalize_media already froze into the receipt above —
        # an assignment naming "figure:12" would otherwise dangle in a
        # document whose highest figure_id is 1. The keys are namespaced
        # ("figure:" / "table:"), so the two maps cannot collide.
        caption_assignment_receipt = remap_caption_receipt(
            caption_assignment_receipt, {**figure_remap, **table_remap}
        )

        processing_warnings: list[ProcessingWarning] = []
        if self._corrupt_region_count:
            processing_warnings.append(
                ProcessingWarning(
                    WarningCode.OCR_CONTROL_CHARS,
                    f"{self._corrupt_region_count} region(s) contained control characters — "
                    "OCR output is corrupted; section headers and references may be unreliable",
                )
            )
        if self._native_text_pua_fallback_count:
            processing_warnings.append(
                ProcessingWarning(
                    WarningCode.OCR_NATIVE_TEXT_PUA_FALLBACK,
                    f"{self._native_text_pua_fallback_count} region(s) contained private-use "
                    "native text and were recovered with OCR",
                )
            )
        if self._dropped_table_count:
            processing_warnings.append(
                ProcessingWarning(
                    WarningCode.OCR_TABLE_DROPPED,
                    f"{self._dropped_table_count} table region(s) could not be parsed and were "
                    "dropped — table content is missing from the output",
                )
            )

        # Sentences not yet created — caller must invoke
        # apply_segmentation() to populate them.
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
            region_summaries=self.region_summaries,
            page_sizes=self.page_sizes,
            processing_warnings=processing_warnings,
            structure_validation_issues=self._structure_validation_issues,
            caption_assignment_receipt=caption_assignment_receipt,
        )

    def apply_segmentation(
        self,
        contents: PaperContents,
        all_segments: list[list[str]],
    ) -> None:
        """Populate sentences on *contents* from externally-segmented results.

        Must be called after :meth:`parse`.  *all_segments* must have the
        same length as the number of deferred texts that need segmentation
        (i.e. entries where ``needs_segmentation`` is ``True``).

        Deferred entries that do **not** need segmentation (formulas,
        reference entries) are emitted as single sentences in document
        order without consulting *all_segments*.
        """
        # The shared assembler drives emission (validation, seg/pass-through
        # split, text_id/paragraph_id bookkeeping, last_text_id trail). URLs are
        # detected inline per emitted sentence via the on_sentence hook.
        emitted, self._sentence_counter, self._paragraph_counter = self.assembler.emit(
            all_segments,
            sentence_factory=self._make_sentence,
            sentence_counter=self._sentence_counter,
            paragraph_counter=self._paragraph_counter,
            on_sentence=self._detect_urls,
        )
        self.sentences.extend(emitted)

        contents.sentences = self.sentences
        contents.links = self.links
        contents.sections_text = DocumentAssembler.build_sections_text(self.sentences)
        contents.xrefs = detect_xrefs(self.sentences, self.tables, self.figures)

        # Sentence/link state was rebuilt; drop any DataFrame caches built from
        # an earlier (empty) state so subsequent reads see the new sentences.
        contents.invalidate_text_caches()

    def create_content_sections(self, contents: PaperContents) -> None:
        """Create dedicated sections for figures, tables, and footnotes.

        Must be called after :meth:`apply_segmentation`.  Appends new
        sections (type ``fig``, ``table``, ``footnote``) after all body
        sections.  Captions and footnote text become sentences in the
        text table.  A footnote printed with a mark gets a xref from the
        sentence before it to its text.

        Preserves the original body ``section_id`` on each figure/table
        so that study-ID propagation can inherit from the correct section.
        """
        # --- Figure sections ---
        for fig in self.figures:
            # Remember the body section where the figure was declared so
            # study-ID propagation can inherit from the correct section.
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

            # Update figure to point to its own section
            fig.section_id = self._section_counter

            # Add caption as sentence if present
            if fig.caption:
                self._paragraph_counter += 1
                sent = PaperSentence(
                    text_id=self._sentence_counter,
                    text=fig.caption,
                    section_id=self._section_counter,
                    paragraph_id=self._paragraph_counter,
                    page_number=fig.page_number,
                )
                contents.sentences.append(sent)
                self._sentence_counter += 1

        # --- Table sections ---
        for tbl in self.tables:
            # Remember the body section where the table was declared.
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

            # Update table to point to its own section
            tbl.section_id = self._section_counter

            # Add caption as sentence if present
            if tbl.caption:
                self._paragraph_counter += 1
                sent = PaperSentence(
                    text_id=self._sentence_counter,
                    text=tbl.caption,
                    section_id=self._section_counter,
                    paragraph_id=self._paragraph_counter,
                    page_number=tbl.page_number,
                )
                contents.sentences.append(sent)
                self._sentence_counter += 1

        # --- Footnote sections + xrefs ---
        formula_text_ids = {s.text_id for s in self.sentences if s.is_display_formula}
        for footnote_num, (fn_text, fn_page, _fn_orig_section, fn_deferred_idx) in enumerate(
            self._footnotes, start=1
        ):
            self._section_counter += 1
            footnote_section_id = self._section_counter
            marker = printed_marker(fn_text)

            section = PaperSection(
                section_id=footnote_section_id,
                header=f"Footnote {footnote_num}",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.FOOTNOTE,
                synthetic_kind="footnote",
                footnote_label=marker,
            )
            contents.sections.append(section)

            # Add footnote text as sentence
            self._paragraph_counter += 1
            sent = PaperSentence(
                text_id=self._sentence_counter,
                text=fn_text,
                section_id=footnote_section_id,
                paragraph_id=self._paragraph_counter,
                page_number=fn_page,
            )
            contents.sentences.append(sent)
            self._sentence_counter += 1

            # A note printed without a mark (an author note, or text taken
            # for a note) is referenced from nowhere in the text, so it gets
            # no xref: one would carry an ordinal no reader can see.
            if marker is None:
                continue

            # Find nearest preceding text_id for the xref. Display-formula
            # sentences are skipped: they export as "[equation]" placeholders,
            # so an anchor there points consumers at text they never see
            # (mirrors the detect_xrefs guard).
            nearest_text_id = self._find_nearest_text_id(
                fn_deferred_idx, skip_text_ids=formula_text_ids
            )

            # Link the note to the sentence before it. bibr does not find the
            # mark in the text, so this is where the reference probably is,
            # and ``contents`` is the mark the note is printed with.
            contents.xrefs.append(
                PaperXref(
                    # The footnote's own text row: the xref's target.
                    xref_id=sent.text_id,
                    xref_type="foot",
                    contents=marker,
                    text_id=nearest_text_id,
                )
            )

        # New figure/table/footnote sections plus their caption/footnote
        # sentences were appended; drop any DataFrame caches so downstream
        # consumers rebuild against the post-mutation state.
        contents.invalidate_text_caches()
        # ``sections_text`` was built in ``apply_segmentation``, before any of
        # these sections existed, and nothing refreshed it — so it stayed
        # permanently stale, missing every caption and footnote section.
        # Consumers that noticed had to rebuild it themselves from
        # ``sentences`` (see ``research_integrity``); consumers that did not
        # silently read an incomplete document.
        contents.sections_text = DocumentAssembler.build_sections_text(contents.sentences)

    # ------------------------------------------------------------------
    # Page processing
    # ------------------------------------------------------------------

    def _is_in_margin_band(self, page_idx: int, region_idx: int) -> bool:
        """True if the region sits in a page's top or bottom margin band.

        A missing bbox is unknown rather than disqualifying — falling back to
        the pre-geometry behaviour keeps genuine furniture demoted on inputs
        whose layout carries no coordinates.
        """
        bbox = bbox_to_tuple(self.json_result[page_idx][region_idx].bbox_2d)
        if bbox is None:
            return True
        _, y1, _, y2 = bbox
        return y1 <= _RUNNING_HEADER_TOP_Y or y2 >= _RUNNING_HEADER_BOTTOM_Y

    def _mark_running_headers(self) -> None:
        """Detect heading regions that are actually per-page running headers.

        Two heuristics together catch the cases the layout model
        confuses:

        1. **Multi-page repeats** — a normalized heading content string
           appearing on ≥2 pages is a running header. Real section
           headers (Methods, References, Discussion) appear exactly once.

        2. **Extra ``doc_title`` regions** — academic papers have exactly
           one ``doc_title`` (the paper title, on page 1). Any subsequent
           ``doc_title`` region is the layout model misclassifying a
           running-header (page-2+ author line, banner, journal name).
           Demoting these prevents later author-line regions from splitting the body.
        """
        # Heuristic 1: multi-page repeats (any heading label).
        seen: dict[str, list[tuple[int, int]]] = {}
        # Heuristic 1b: multi-page repeats among *body-text* regions. A running
        # header GLM-OCR tags ``text`` (e.g. a wide-letter-spaced preprint
        # banner) bypasses the heading-only heuristics and leaks into whatever
        # section is active at the page break — corrupting the References
        # block. Page furniture is short; the length cap keeps a genuine
        # paragraph that happens to repeat from being demoted.
        seen_body: dict[str, list[tuple[int, int]]] = {}
        # Heuristic 2: track doc_title occurrences so all but the first
        # can be demoted regardless of repetition. Content is kept so
        # copyright blurbs can be excluded from anchoring.
        doc_title_occurrences: list[tuple[int, int, str]] = []

        for page_idx, regions in enumerate(self.json_result):
            for region_idx, region in enumerate(regions):
                native_label = region.native_label
                label = region.label
                effective = native_label if native_label in LABEL_TREATMENT else label
                clean_content = fix_ocr_artifacts(region.content)
                self._clean_region_content[(page_idx, region_idx)] = clean_content
                content = clean_content.strip()
                if not content:
                    continue
                if effective in _BODY_TEXT_LABELS:
                    normalized = re.sub(r"\s+", " ", content).lower().strip()
                    if normalized and len(normalized) <= _RUNNING_HEADER_MAX_LEN:
                        seen_body.setdefault(normalized, []).append((page_idx, region_idx))
                    continue
                if effective not in ("doc_title", "paragraph_title"):
                    continue
                # Strip markdown prefix + emphasis so identical text at
                # different layout levels still de-dupes.
                normalized = re.sub(r"^#{1,6}\s*", "", content)
                normalized = strip_markdown_emphasis(normalized).lower()
                normalized = re.sub(r"\s+", " ", normalized).strip()
                if not normalized:
                    continue
                seen.setdefault(normalized, []).append((page_idx, region_idx))
                if effective == "doc_title":
                    doc_title_occurrences.append((page_idx, region_idx, normalized))

        for normalized, occurrences in seen.items():
            if len({pi for pi, _ in occurrences}) >= 2:
                demoted = occurrences
                # Preserve the first printed title occurrence even when the same text is
                # repeated as a running header on later pages.
                first_page, _ = occurrences[0]
                if (
                    first_page == self._first_page_index
                    and not self._is_copyright_notice(normalized)
                    and not is_exact_front_matter_furniture(normalized)
                    and not FRONT_MATTER_MASTHEAD_RE.match(normalized)
                ):
                    demoted = occurrences[1:]
                # Repetition alone is not evidence: multi-study papers
                # legitimately repeat ``Method``/``Results``/``Participants``
                # per study, and demoting those deletes the heading and folds
                # its body into the preceding section. Require the geometry of
                # actual page furniture.
                demoted = [occ for occ in demoted if self._is_in_margin_band(*occ)]
                if not demoted:
                    continue
                self._running_header_regions.update(demoted)
                logger.debug(
                    "Demoting repeated heading %r as running header (%d of %d occurrences)",
                    normalized,
                    len(demoted),
                    len(occurrences),
                )
        for occurrences in seen_body.values():
            if len({pi for pi, _ in occurrences}) >= 2:
                self._running_header_regions.update(occurrences)

        # All ``doc_title`` regions after the *real* title are running
        # headers. The anchor is the first doc_title that is not a
        # copyright/permission blurb — publishers sometimes print one
        # above the actual title, and anchoring on it would demote (and
        # lose) the real title. Copyright lines before the anchor are
        # left alone; ``_handle_heading`` already skips them for title
        # detection.
        anchor = next(
            (
                i
                for i, (_, _, text) in enumerate(doc_title_occurrences)
                if not self._is_copyright_notice(text)
            ),
            0,
        )
        if len(doc_title_occurrences) > anchor + 1:
            # A long title is routinely split across two or three doc_title
            # regions on the same page — a main title and its subtitle, or a
            # line break the layout model kept as separate rows. Those are not
            # running headers: a running head is by definition printed on a
            # LATER page. Demoting same-page continuations truncated the title
            # to its first region.
            anchor_page = doc_title_occurrences[anchor][0]
            self._running_header_regions.update(
                (pi, ri) for pi, ri, _ in doc_title_occurrences[anchor + 1 :] if pi != anchor_page
            )

        if self._running_header_regions:
            logger.info(
                "Demoted %d heading regions detected as running headers",
                len(self._running_header_regions),
            )

    def _mark_reference_envelopes(self) -> None:
        """Shadow whichever of an aggregate reference box and its entries is redundant.

        The layout model can return one ``reference`` box over several entries
        plus ``reference_content`` boxes for some or all of them. The aggregate
        box is shadowed only when the entry boxes inside it already carry its
        text. Otherwise it stays and only the entry boxes whose text it holds
        are shadowed, so an entry without a box of its own is not lost and no
        entry is emitted twice. Shadowed regions keep their region summaries.

        Only entry boxes inside the aggregate box take part. They replace it at
        its place in reading order, where it keys the References section even
        when the OCR stage blanked its text, and a separate short entry
        elsewhere on the page ("PubMed") is never hidden because the aggregate
        box's text happens to contain it.
        """
        for page_idx, regions in enumerate(self.json_result):
            children = [
                (child_idx, child)
                for child_idx, child in enumerate(regions)
                if (child.native_label or child.label) == "reference_content"
            ]
            if not children:
                continue
            for region_idx, region in enumerate(regions):
                if (region.native_label or region.label) != "reference":
                    continue
                contained = [
                    (child_idx, child)
                    for child_idx, child in children
                    if _bbox_containment_fraction(child.bbox_2d, region.bbox_2d) >= 0.8
                ]
                if not contained:
                    continue
                envelope_text = alnum_key(region.content or "")
                children_text = alnum_key("".join(child.content or "" for _, child in contained))
                if alnum_text_covered(envelope_text, children_text):
                    self._shadowed_reference_regions.add((page_idx, region_idx))
                    continue
                self._shadowed_reference_regions.update(
                    (page_idx, child_idx)
                    for child_idx, child in contained
                    if alnum_text_covered(alnum_key(child.content or ""), envelope_text)
                )

        if self._shadowed_reference_regions:
            logger.info(
                "Shadowing %d duplicate reference region(s)",
                len(self._shadowed_reference_regions),
            )

    def _process_page(
        self, regions: list[OcrRegionResult], page_number: int, page_idx: int = -1
    ) -> None:
        """Process all regions on a single page."""
        # ``page_idx`` defaults to -1 for backwards compatibility with any
        # external caller; the running-header lookup uses the 0-based index.
        if page_idx < 0:
            page_idx = page_number - 1
        for region_idx, region in enumerate(regions):
            if region.page_w and region.page_h and page_number not in self.page_sizes:
                self.page_sizes[page_number] = (region.page_w, region.page_h)
            label = region.label
            native_label = region.native_label
            if (
                region.native_text_rejection_reason == "private_use"
                and region.native_text_candidate is not None
                and region.content.strip()
                and (native_label or label)
                in {"text", "content", "vertical_text", "reference", "reference_content"}
            ):
                self._native_text_pua_fallback_count += 1
            if ocr_corruption_count(region.content) >= OCR_CORRUPTION_MIN_CHARS:
                self._corrupt_region_count += 1
            key = (page_idx, region_idx)
            content = self._clean_region_content.get(key)
            if content is None:
                content = fix_ocr_artifacts(region.content)
                self._clean_region_content[key] = content
            bbox = region.bbox_2d

            bbox_tuple = bbox_to_tuple(bbox)
            bbox_h = bbox_w = char_dens = est_lh = None
            if bbox_tuple:
                x1, y1, x2, y2 = bbox_tuple
                bbox_h = y2 - y1
                bbox_w = x2 - x1
                area = bbox_h * bbox_w
                if area > 0 and content:
                    char_dens = round(len(content) / area, 4)
                if bbox_h > 0 and content:
                    n_lines = max(1, content.count("\n") + 1)
                    est_lh = round(bbox_h / n_lines, 2)

            region_summary = RegionSummary(
                page=page_number,
                index=region.index,
                label=native_label or label,
                bbox=bbox_tuple,
                font_size=region.font_size,
                font_weight=region.font_weight,
                font_bold=region.font_bold,
                content=content[:200] if content else None,
                canonical_ocr_content=(
                    content if region.native_text_rejection_reason is not None else None
                ),
                raw_ocr_content=region.raw_content,
                native_text_candidate=region.native_text_candidate,
                native_text_rejection_reason=region.native_text_rejection_reason,
                bbox_height=bbox_h,
                bbox_width=bbox_w,
                char_density=char_dens,
                estimated_line_height=est_lh,
            )
            self.region_summaries.append(region_summary)
            self._source_region_index = len(self.region_summaries) - 1

            if key in self._shadowed_reference_regions:
                region_summary.section_id = self._current_section_id or None
                continue

            # Prefer native_label for treatment dispatch — glmocr's _map_label()
            # collapses specific labels (e.g. "abstract" → "text"); native_label
            # preserves the original layout detector label.
            effective_label = native_label if native_label in LABEL_TREATMENT else label
            treatment = LABEL_TREATMENT.get(effective_label, "content")
            if effective_label == "vision_footnote" and self._BARE_TABLE_LABEL_RE.fullmatch(
                content.strip()
            ):
                # Some rotated PMC table labels are emitted as vision
                # footnotes. Only a bare TABLE label ("TABLE 2", "TABLE IV",
                # "Table S1") is safe to promote; ordinary statistical notes
                # remain footnotes.
                treatment = "table_caption"
            dispatch_treatment = (
                "structural"
                if (page_idx, region_idx) in self._running_header_regions
                else treatment
            )
            self._prepare_table_caption_state_for_region(dispatch_treatment, content)

            # Region metadata for v4 training feature export and source-region
            # provenance.  Built once per region and threaded through to
            # sentences via _handle_content / _handle_formula /
            # _handle_section_hint (References path).
            region_meta: dict | None = {
                "font_size": region.font_size,
                "font_bold": region.font_bold,
                "is_italic": region.is_italic,
                # The 0..1000 layout-space box; the export converts it to
                # points on the displayed page (bibr.export.geometry).
                "bbox": list(bbox_tuple) if bbox_tuple else None,
                "region_type": native_label or label or None,
                # This region's RegionSummary key, exported as the ``page`` and
                # ``index`` of its ``extraction.regions`` row. ``index`` is the
                # region's position on the page after OCR post-processing
                # renumbered it, which can differ from the layout detector's slot.
                "region_page": region_summary.page,
                "region_index": region_summary.index,
            }

            if treatment == "abandon":
                continue

            # Demote running headers detected by _mark_running_headers,
            # regardless of how the layout model labeled them (doc_title /
            # paragraph_title heading OR body ``text``). Routing to
            # ``_handle_structural`` keeps the furniture out of section
            # content (so it never pollutes the References block) while
            # preserving it via ``detected_headers``.
            if (page_idx, region_idx) in self._running_header_regions:
                self._handle_structural("header", content)
                continue

            if treatment == "structural":
                self._handle_structural(effective_label, content)
                continue

            if treatment == "heading":
                # Flush any carry-over before a heading
                self._flush_carry_over()
                self._handle_heading(effective_label, content, page_number, bbox)

            elif treatment == "section_hint":
                self._flush_carry_over()
                self._handle_section_hint(
                    effective_label, content, page_number, bbox, region_meta=region_meta
                )

            elif treatment == "content":
                self._handle_content(content, page_number, bbox, region_meta=region_meta)

            elif treatment == "formula":
                self._handle_formula(content, page_number, bbox, region_meta=region_meta)

            elif treatment == "table":
                # A full-width table may sit below a two-column sentence whose
                # continuation starts on the next page. Tables do not emit text
                # rows, so keep an unfinished prose carry-over alive across the
                # table; the normal heading/section and continuation guards
                # still prevent unrelated text from being joined.
                self._handle_table(content, page_number, bbox)

            elif treatment == "table_caption":
                self._handle_table_caption(content, bbox, page_number)

            elif treatment == "figure":
                self._flush_carry_over()
                self._handle_figure(
                    page_number,
                    bbox,
                    region.image_b64,
                    source_label=effective_label,
                )

            elif treatment == "caption":
                # Smart route: "Table …" → table caption, else → figure caption
                self._handle_caption(content, bbox, page_number, source_label=effective_label)

            elif treatment == "footnote":
                self._handle_footnote(content, page_number)

            region_summary.section_id = self._current_section_id or None

            # Record layout hints for section classification boosts
            if effective_label in (
                "abstract",
                "reference",
                "reference_content",
                "footnote",
                "vision_footnote",
            ) and treatment in {"footnote", "section_hint"}:
                self.layout_hints.append((effective_label, page_number))
