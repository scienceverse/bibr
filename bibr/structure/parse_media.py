"""Table / figure / caption region handlers for :class:`PDFParser`.

Extracted from :mod:`bibr.structure.pdf_parser` as :class:`MediaHandlersMixin`.
Methods move verbatim (only the ``text_repair.bbox_to_tuple`` helper is spelled
with its public name); shared state (``self.tables``, ``self.figures``, caption
trackers, counters) lives on :class:`PDFParser` and is reached through ``self``.
"""

import logging
import re

import pandas as pd

from bibr.paper_contents import (
    CaptionAssignment,
    CaptionAssignmentReceipt,
    CaptionCandidate,
    PaperFigure,
    PaperFigurePart,
    PaperTable,
    PaperTablePart,
    Provenance,
)
from bibr.structure.caption_matcher import CaptionTarget, assign_captions
from bibr.structure.float_images import composite_panel_image
from bibr.structure.float_labels import LABEL, SUPPLEMENT_WORD, caption_label, normalize_label
from bibr.structure.text_repair import bbox_to_tuple
from bibr.validation import IssueSeverity, ValidationIssue

logger = logging.getLogger(__name__)

# A markdown cell separator is a pipe that is not backslash-escaped, which is
# how GFM (and OCR following it) writes a literal "|" inside a cell.
_UNESCAPED_PIPE_RE = re.compile(r"(?<!\\)\|")


class MediaHandlersMixin:
    """Table, figure, and caption handlers for ``PDFParser``."""

    # Panel/sub-figure markers: "(a)", "a)", "(b) Without BN", "(c) With BN".
    # A single letter followed by ")" and an optional short description.  The
    # length cap in _is_panel_label prevents swallowing real captions that
    # happen to start with a lowercase letter.
    _PANEL_LABEL_RE = re.compile(r"^\(?[a-z]\)\s*(.*)$", re.DOTALL)

    _MARKDOWN_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*[-:]+[-|:\s]*$")

    # Salvage normalization for OCR-corrupted table markup: control-char
    # stripping can shred tag attributes (``<table\x00\x00">`` → ``<table">``),
    # making html5lib see an unknown tag. Rewriting table-family tags to their
    # bare form recovers the structure when rows/cells survived.
    _TABLE_TAG_RE = re.compile(r"<(/?)(table|thead|tbody|tr|t[dh])\b[^>]*>", re.IGNORECASE)
    # Captions that print a label (``bibr.structure.float_labels.LABEL``):
    # "Table 3", "Table 3.1", "Table S2", "Figure A1", "TABLE IV", and the
    # "Supplementary Table 4" spelling.
    _BARE_TABLE_LABEL_RE = re.compile(
        rf"^{SUPPLEMENT_WORD}?Table\s+{LABEL}\s*[.:]?\s*$", re.IGNORECASE
    )
    _EXPLICIT_FIGURE_RE = re.compile(
        rf"^(?P<supplement>{SUPPLEMENT_WORD})?(?:Figure|Fig\.?)\s+(?P<label>{LABEL})",
        re.IGNORECASE,
    )
    _TABLE_LABEL_RE = re.compile(
        rf"^(?P<supplement>{SUPPLEMENT_WORD})?Table\s+(?P<label>{LABEL})", re.IGNORECASE
    )
    _ROMAN_LABEL_RE = re.compile(r"[IVXLCDM]+")
    _DOTTED_NUMBER_RE = re.compile(r"(?P<major>\d+)(?:\.\d+)*")
    _DOI_URL_RE = re.compile(r"https?://doi\.org/[^\s]+", re.IGNORECASE)
    _TRAILING_DOI_URL_RE = re.compile(
        r"(?:\s*https?://doi\.org/[^\s]+)+\s*$",
        re.IGNORECASE,
    )

    # ---- Decoration suppression -------------------------------------------
    # ``bbox_to_tuple`` casts glmocr ``bbox_2d`` to float and nothing else, so
    # what reaches this mixin is PP-DocLayoutV3 image space normalized to
    # 0..1000 on BOTH axes, y-down from the top-left of the rendered CropBox
    # (``bibr/ocr/native_text.py`` module docstring and :445-447; the same
    # space ``parse_headings.py:522`` divides by 1000.0 to read a page-height
    # fraction, and ``_group_continuation_tables`` below reads as ``>= 800`` /
    # ``<= 200`` page edges). Every bound here is therefore a page fraction
    # times 1000 — not points, not pixels.
    _PAGE_SPAN = 1000.0
    # Bound publisher badges and licence stamps by page-relative height and area. Retain the
    # ownership and caption checks before suppressing any crop.
    _DECORATION_MAX_HEIGHT = 0.08 * _PAGE_SPAN  # 80
    _DECORATION_MAX_AREA = 0.012 * _PAGE_SPAN * _PAGE_SPAN  # 12_000
    # Off-body bands also cover repeated footer decorations beyond the first page. Apply the
    # remaining caption and ownership checks before suppressing a crop.
    _DECORATION_FOOTER_TOP = 0.88 * _PAGE_SPAN  # 880
    _DECORATION_MARGIN = 0.08 * _PAGE_SPAN  # 80
    # Both labels that can reach ``_handle_figure``: ``LABEL_TREATMENT`` maps
    # only "chart" and "image" to the "figure" treatment
    # (``pdf_parser.py:105-106``). The old filter accepted "image" alone, so a
    # badge the layout model happened to call a chart was never even eligible.
    _DECORATION_LABELS = frozenset({"chart", "image"})

    def _add_caption_candidate(
        self,
        text: str,
        object_type: str,
        bbox: list | tuple | None,
        page_number: int | None,
        *,
        source_index: int | None = None,
    ) -> str:
        caption_id = f"caption:{len(self._caption_candidates) + 1}"
        # Remember where the candidate was printed. ``CaptionCandidate`` is
        # part of the exported receipt, so the owning section rides in this
        # internal side map instead of widening that schema. Used to replay an
        # unowned candidate as body text (see ``replay_unowned_captions``).
        self._caption_candidate_sections[caption_id] = self._current_section_id
        self._caption_candidates.append(
            CaptionCandidate(
                caption_id=caption_id,
                text=text.strip(),
                object_type=object_type,
                page_number=page_number,
                bbox=bbox_to_tuple(bbox),
                source_index=(self._source_region_index if source_index is None else source_index),
            )
        )
        return caption_id

    def _handle_table(self, content: str, page_number: int, bbox: list | None = None) -> None:
        """Parse a table (HTML or markdown) into a PaperTable."""
        text = content.strip()
        if not text:
            self._rollback_pending_table_caption_fragment()
            self._expire_pending_table_label_fragment()
            return

        # Detect format and parse accordingly. For HTML input, keep the source
        # markup verbatim: the VLM's rowspan/colspan and multi-level headers
        # carry structure a flat DataFrame cannot, and round-tripping through
        # ``read_html`` -> ``to_html`` drops rowspan and injects
        # ``Unnamed: 0_level_0`` / ``class="dataframe"`` noise. The DataFrame is
        # still used for the flattened ``contents`` grid.
        source_html: str | None = None
        if text.lstrip().startswith("<table") or text.lstrip().startswith("<TABLE"):
            df = self._parse_html_table(text)
            source_html = text
            if df is None or df.empty:
                salvaged = self._TABLE_TAG_RE.sub(r"<\1\2>", text)
                if salvaged != text:
                    df = self._parse_html_table(salvaged)
                    source_html = salvaged
        else:
            df = self._parse_markdown_table(text)
        if df is None or df.empty:
            self._rollback_pending_table_caption_fragment()
            self._expire_pending_table_label_fragment()
            self._dropped_table_count += 1
            logger.warning(
                "Table region on page %d could not be parsed and was dropped", page_number
            )
            return

        # Markdown input has no source HTML; render it from the DataFrame.
        html = source_html if source_html is not None else df.to_html(index=False)

        # Caption-fragment composition stays provisional until a parseable,
        # nearby table confirms ownership.
        confirmed_caption_id = self._confirm_pending_table_caption_fragment(bbox, page_number)

        bbox_tuple = bbox_to_tuple(bbox)
        provenance = [Provenance(page_no=page_number, bbox=bbox_tuple)] if bbox_tuple else []
        part = PaperTablePart(
            page_number=page_number,
            bbox=bbox_tuple,
            tbl_html=html,
            df=df.copy(deep=True),
            provenance=list(provenance),
        )
        table = PaperTable(
            table_id=self._table_counter,
            df=df,
            tbl_html=html,
            section_id=self._current_section_id,
            caption=None,
            page_number=page_number,
            provenance=provenance,
            parts=[part],
        )
        self.tables.append(table)
        self._table_source_indices[id(table)] = self._source_region_index
        if confirmed_caption_id is not None:
            self._confirmed_table_caption_owners[confirmed_caption_id] = id(table)

        self._table_counter += 1

    def _handle_figure(
        self,
        page_number: int,
        bbox: list | None = None,
        image_b64: str | None = None,
        *,
        source_label: str | None = None,
    ) -> None:
        """Create a PaperFigure from an image/chart region.

        Decoration is no longer rejected here. Deleting a region at parse time
        means the object is gone before any caption can vouch for it, so one
        mis-set bound silently deletes a real float — which is how the bounds
        this replaced came to miss every badge in the campaign. Suppression now
        runs in :meth:`_suppress_unowned_decoration_figures` at finalize, on
        figures nothing owns.
        """
        self._expire_pending_table_label_fragment()
        bbox_tuple = bbox_to_tuple(bbox)
        provenance = [Provenance(page_no=page_number, bbox=bbox_tuple)] if bbox_tuple else []
        part = PaperFigurePart(
            page_number=page_number,
            bbox=bbox_tuple,
            image_b64=image_b64,
            provenance=list(provenance),
        )
        figure = PaperFigure(
            figure_id=self._figure_counter,
            section_id=self._current_section_id,
            image_b64=image_b64,
            caption=None,
            page_number=page_number,
            provenance=provenance,
            parts=[part],
        )
        self.figures.append(figure)
        self._figure_source_indices[id(figure)] = self._source_region_index
        # The region label is only in scope here, and the decoration post-pass
        # needs it after grouping has already rearranged ``self.figures``. The
        # mixin does not own ``PDFParser.__init__``, so the map is created on
        # first use — the same defensive access ``_pending_bare_table_label``
        # gets through ``getattr`` below.
        if not hasattr(self, "_figure_source_labels"):
            self._figure_source_labels: dict[int, str | None] = {}
        self._figure_source_labels[id(figure)] = source_label

        self._figure_counter += 1

    def _handle_table_caption(self, content: str, bbox: list | None, page_number: int) -> None:
        """Collect a table-caption candidate without mutating any table."""
        self._expire_pending_table_label_fragment()
        text = content.strip()
        if not text:
            return
        caption_id = self._add_caption_candidate(text, "table", bbox, page_number)
        pending = (text, bbox, page_number, caption_id)
        if self._BARE_TABLE_LABEL_RE.fullmatch(text):
            self._pending_bare_table_label = pending

    def _expire_pending_table_label_fragment(self) -> None:
        """End the one-region composition window for a bare table label."""
        self._pending_bare_table_label = None

    def _stage_pending_table_label_fragment(
        self,
        content: str,
        bbox: list | None,
        page_number: int,
        *,
        region_meta: dict | None = None,
        source_label: str = "content",
    ) -> bool:
        """Buffer the next adjacent fragment until a table confirms ownership."""
        pending = getattr(self, "_pending_bare_table_label", None)
        self._pending_bare_table_label = None
        if pending is None:
            return False

        label, label_bbox, label_page, _caption_id = pending
        if label_page != page_number or not self._is_bbox_nearby(
            label_bbox, label_page, bbox, page_number
        ):
            return False
        if self._TABLE_CAPTION_RE.match(content) or self._FIGURE_CAPTION_RE.match(content):
            return False

        self._pending_table_caption_fragment = (
            pending,
            content,
            bbox,
            page_number,
            region_meta,
            source_label,
            self._source_region_index,
        )
        return True

    def _confirm_pending_table_caption_fragment(
        self, table_bbox: list | None, page_number: int
    ) -> str | None:
        """Commit a staged label/fragment pair when the next table owns it."""
        staged = getattr(self, "_pending_table_caption_fragment", None)
        self._pending_table_caption_fragment = None
        self._pending_bare_table_label = None
        if staged is None:
            return None

        (
            pending,
            fragment,
            fragment_bbox,
            fragment_page,
            _region_meta,
            _source_label,
            fragment_source_index,
        ) = staged
        label, label_bbox, label_page, caption_id = pending
        composed_bbox = self._merge_bboxes(label_bbox, fragment_bbox)
        if fragment_page != page_number or not self._is_bbox_nearby(
            composed_bbox, label_page, table_bbox, page_number
        ):
            self._replay_table_caption_fragment(staged)
            return None

        fragment_id = self._add_caption_candidate(
            fragment,
            "table",
            fragment_bbox,
            fragment_page,
            source_index=fragment_source_index,
        )
        self._table_caption_fragments[caption_id] = fragment_id
        return caption_id

    def _rollback_pending_table_caption_fragment(self) -> None:
        """Replay an unowned provisional fragment through normal content handling."""
        staged = getattr(self, "_pending_table_caption_fragment", None)
        self._pending_table_caption_fragment = None
        self._pending_bare_table_label = None
        if staged is not None:
            self._replay_table_caption_fragment(staged)

    def _replay_table_caption_fragment(self, staged: tuple) -> None:
        _pending, content, bbox, page_number, region_meta, source_label, source_index = staged
        if source_label in {"figure_title", "chart_title"}:
            self._handle_figure_caption(
                content,
                bbox,
                page_number,
                source_index=source_index,
            )
        else:
            self._handle_content(content, page_number, bbox, region_meta=region_meta)

    def _prepare_table_caption_state_for_region(self, treatment: str, content: str) -> None:
        """Advance provisional caption state at every OCR region boundary."""
        if (
            getattr(self, "_pending_table_caption_fragment", None) is not None
            and treatment != "table"
        ):
            self._rollback_pending_table_caption_fragment()
        if not content or treatment not in {"caption", "content", "table"}:
            self._expire_pending_table_label_fragment()

    def _handle_caption(
        self,
        content: str,
        bbox: list | None,
        page_number: int,
        *,
        source_label: str | None = None,
    ) -> None:
        """Route a caption to figure or table handler based on content.

        The layout model (PP-DocLayoutV3) only produces ``figure_title`` for
        all captions — there is no dedicated table-caption label.  Because the
        region is ALREADY a caption, discriminate with a loose "Table N" prefix
        (no separator required — backends emit "Table 1 Overview" with a space
        or "Table 1 | Overview" with a pipe).  Everything else is routed to
        ``_handle_figure_caption`` (which also groups sub-panel labels).
        """
        text = content.strip()
        if not text:
            return
        if not self._LOOSE_TABLE_CAPTION_RE.match(
            text
        ) and self._stage_pending_table_label_fragment(
            text,
            bbox,
            page_number,
            source_label=source_label or "figure_title",
        ):
            return
        if self._LOOSE_TABLE_CAPTION_RE.match(text):
            self._handle_table_caption(content, bbox, page_number)
        else:
            self._handle_figure_caption(content, bbox, page_number)

    @classmethod
    def _is_panel_label(cls, text: str) -> bool:
        """True for short sub-figure markers like "(a)", "b)", "(c) With BN"."""
        text = text.strip()
        # Cap length so a real caption starting with a lowercase letter (e.g.
        # "a) The full description of …") is not mistaken for a panel marker.
        if not text or len(text) > 60:
            return False
        return bool(cls._PANEL_LABEL_RE.match(text))

    @classmethod
    def _panel_description(cls, text: str) -> str:
        """Return the descriptive tail of a panel label ("" for a bare marker)."""
        m = cls._PANEL_LABEL_RE.match(text.strip())
        return m.group(1).strip() if m else ""

    def _handle_figure_caption(
        self,
        content: str,
        bbox: list | None,
        page_number: int,
        *,
        source_index: int | None = None,
    ) -> None:
        """Collect every figure-title region for lossless ownership accounting."""
        self._expire_pending_table_label_fragment()
        text = content.strip()
        if not text:
            return
        caption_id = self._add_caption_candidate(
            text,
            "figure",
            bbox,
            page_number,
            source_index=source_index,
        )
        if text.casefold() == "author manuscript":
            self._non_caption_candidate_reasons[caption_id] = ("publisher_noise",)
        elif not self._caption_useful_text(text) and self._caption_doi_urls(text):
            self._non_caption_candidate_reasons[caption_id] = ("doi_only_evidence",)
        elif re.fullmatch(r"p\s*[<=>≤≥]\s*\.?\d+", text, re.IGNORECASE):
            self._non_caption_candidate_reasons[caption_id] = ("table_note",)

    def replay_unowned_captions(self, receipt: CaptionAssignmentReceipt) -> int:
        """Re-emit caption candidates that never found an owner as body text.

        A caption candidate lives only in ``_caption_candidates`` — nothing
        appends it to the assembler. Its text reaches ``contents.sentences``
        solely by being attached to a figure or table in
        ``create_content_sections``. So a candidate that ends with
        ``object_id is None`` was *deleted*: the parse raised
        ``VAL_CAPTION_OWNERSHIP`` and moved on, and the printed text appeared
        in no field of the export at all. That is the same silent body-text
        loss the caption re-router used to cause, one step further downstream.

        Replaying keeps the text in the document. Candidates deliberately
        rejected as non-captions (publisher noise, DOI-only fragments, table
        notes) stay dropped, and a candidate whose text was already assigned
        elsewhere is skipped so nothing is duplicated.

        Returns the number of replayed candidates.
        """
        assigned_text = {
            candidate.text.strip()
            for candidate, assignment in zip(receipt.candidates, receipt.assignments, strict=True)
            if assignment.object_id is not None
        }
        replayed = 0
        for candidate, assignment in zip(receipt.candidates, receipt.assignments, strict=True):
            text = candidate.text.strip()
            if (
                assignment.object_id is not None
                or not text
                or text in assigned_text
                or candidate.caption_id in self._non_caption_candidate_reasons
            ):
                continue
            self.assembler.append(
                text,
                candidate.page_number,
                self._caption_candidate_sections.get(
                    candidate.caption_id, self._current_section_id
                ),
                True,
                provenance=(
                    [Provenance(page_no=candidate.page_number, bbox=candidate.bbox)]
                    if candidate.bbox is not None
                    else None
                ),
            )
            replayed += 1
        if replayed:
            logger.debug("Replayed %d unowned caption candidate(s) as body text", replayed)
        return replayed

    @staticmethod
    def _caption_overlap(
        left: tuple[float, float, float, float] | None,
        right: tuple[float, float, float, float] | None,
    ) -> float:
        if left is None or right is None:
            return 0.0
        width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
        height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
        intersection = width * height
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        return intersection / max(1.0, min(left_area, right_area))

    def _deduplicate_caption_candidates(
        self,
    ) -> tuple[list[CaptionCandidate], list[CaptionAssignment]]:
        candidates = list(self._caption_candidates)
        semantic_keys = [self._caption_semantic_key(item.text) for item in candidates]
        doi_urls = [self._caption_doi_urls(item.text) for item in candidates]
        explicit_labels = [self._caption_printed_label(item) for item in candidates]
        parents = list(range(len(candidates)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for left_index, left in enumerate(candidates):
            for right_index in range(left_index + 1, len(candidates)):
                right = candidates[right_index]
                if (
                    left.object_type != right.object_type
                    or left.page_number != right.page_number
                    or self._caption_overlap(left.bbox, right.bbox) < 0.9
                ):
                    continue
                left_label = explicit_labels[left_index]
                right_label = explicit_labels[right_index]
                if left_label is not None and right_label is not None and left_label != right_label:
                    continue
                left_key = semantic_keys[left_index]
                right_key = semantic_keys[right_index]
                same_useful_caption = bool(left_key) and left_key == right_key
                same_number_prefix = bool(
                    left_key
                    and right_key
                    and left_label is not None
                    and left_label == right_label
                    and self._caption_key_prefix_overlap(left_key, right_key)
                )
                if same_useful_caption or same_number_prefix:
                    union(left_index, right_index)

        # DOI-only regions may corroborate one meaningful caption cluster, but
        # must never become a transitive bridge between otherwise distinct
        # captions that happen to carry the same article-level DOI.
        for doi_index, doi_candidate in enumerate(candidates):
            if semantic_keys[doi_index] or not doi_urls[doi_index]:
                continue
            owners = [
                owner_index
                for owner_index, owner in enumerate(candidates)
                if owner_index != doi_index
                and semantic_keys[owner_index]
                and doi_urls[doi_index] & doi_urls[owner_index]
                and doi_candidate.object_type == owner.object_type
                and doi_candidate.page_number == owner.page_number
                and self._caption_overlap(doi_candidate.bbox, owner.bbox) >= 0.9
            ]
            if owners:
                owner_index = max(
                    owners,
                    key=lambda index: (
                        self._caption_overlap(doi_candidate.bbox, candidates[index].bbox),
                        -abs(doi_candidate.source_index - candidates[index].source_index),
                        -candidates[index].source_index,
                        candidates[index].caption_id,
                    ),
                )
                union(doi_index, owner_index)

        clusters: dict[int, list[int]] = {}
        for index in range(len(candidates)):
            clusters.setdefault(find(index), []).append(index)

        canonical_indices: set[int] = set()
        duplicates: list[CaptionAssignment] = []
        self._caption_display_text_by_id = {}
        self._caption_canonical_by_id = {}
        for members in clusters.values():
            canonical_index = max(
                members,
                key=lambda index: (
                    len(self._caption_useful_text(candidates[index].text)),
                    not bool(self._caption_doi_urls(candidates[index].text)),
                    -candidates[index].source_index,
                    candidates[index].caption_id,
                ),
            )
            canonical = candidates[canonical_index]
            canonical_indices.add(canonical_index)
            for index in members:
                self._caption_canonical_by_id[candidates[index].caption_id] = canonical.caption_id
            if len(members) > 1:
                useful_text = self._caption_useful_text(canonical.text)
                if useful_text:
                    self._caption_display_text_by_id[canonical.caption_id] = useful_text
            for index in members:
                if index == canonical_index:
                    continue
                duplicates.append(
                    CaptionAssignment(
                        caption_id=candidates[index].caption_id,
                        object_id=None,
                        score=0.0,
                        reasons=(
                            f"duplicate_of:{canonical.caption_id}",
                            "semantic_caption_cluster",
                            "strong_overlap",
                        ),
                    )
                )
        active = [
            candidate for index, candidate in enumerate(candidates) if index in canonical_indices
        ]
        if duplicates:
            self._structure_validation_issues.append(
                ValidationIssue(
                    code="VAL_CAPTION_DUPLICATE",
                    severity=IssueSeverity.WARNING,
                    message=f"{len(duplicates)} duplicate caption region(s) retained as evidence",
                    origin_stage="structure",
                    evidence_ids=tuple(item.caption_id for item in duplicates),
                    count=len(duplicates),
                )
            )
        return active, duplicates

    @classmethod
    def _caption_useful_text(cls, text: str) -> str:
        return cls._TRAILING_DOI_URL_RE.sub("", text).strip()

    @classmethod
    def _caption_semantic_key(cls, text: str) -> str:
        useful = cls._caption_useful_text(text)
        useful = useful.replace(r"\(", "").replace(r"\)", "").replace("$", "")
        return " ".join(useful.casefold().split())

    @classmethod
    def _caption_printed_label(cls, candidate: CaptionCandidate) -> str | None:
        """The candidate's printed label, normalized for comparison ("3.1", "s2")."""
        label = caption_label(
            candidate.text, "table" if candidate.object_type == "table" else "figure"
        )
        return normalize_label(label) if label is not None else None

    @classmethod
    def _caption_explicit_label(cls, text: str) -> int | None:
        """The number an explicit caption prints, as the printed-id reservation
        in :meth:`_reconcile_object_ids` reads it: "Table 3.1" → 3, "TABLE IV"
        → 4. ``None`` for a label that is not a number ("S2", "A1", "2a") or a
        supplementary caption; mentions resolve by the whole printed label
        (``PaperTable.label``/``PaperFigure.label``) instead.
        """
        figure_match = cls._EXPLICIT_FIGURE_RE.match(text.strip())
        match = figure_match or cls._TABLE_LABEL_RE.match(text.strip())
        if match is None or match.group("supplement"):
            return None
        label = match.group("label")
        if number := cls._DOTTED_NUMBER_RE.fullmatch(label):
            return int(number.group("major"))
        # Only a table's roman numeral reserves an id, as before labels.
        if figure_match is not None or not cls._ROMAN_LABEL_RE.fullmatch(label):
            return None
        label = label.casefold()
        values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
        total = previous = 0
        for character in reversed(label):
            current = values[character]
            if current < previous:
                total -= current
            else:
                total += current
                previous = current
        return total or None

    @staticmethod
    def _caption_key_prefix_overlap(left: str, right: str) -> bool:
        shorter, longer = sorted((left, right), key=len)
        if shorter == longer or not longer.startswith(shorter):
            return False
        return longer[len(shorter)] in " \t\n:;,.–—-|()"

    def _reconcile_object_ids(
        self,
        objects: list,
        object_type: str,
        id_attribute: str,
        assignments: list[CaptionAssignment],
        candidate_by_id: dict[str, CaptionCandidate],
    ) -> tuple[dict[str, str], list[str]]:
        """Reserve unique printed IDs and place unclaimed objects above them."""
        old_object_ids = {f"{object_type}:{getattr(item, id_attribute)}": item for item in objects}
        claims_by_printed_id: dict[int, set[str]] = {}
        claims_by_object_id: dict[str, set[int]] = {}
        claim_records: list[tuple[str, str, int]] = []
        for assignment in assignments:
            if assignment.object_id not in old_object_ids or assignment.ambiguous:
                continue
            candidate = candidate_by_id[assignment.caption_id]
            if candidate.object_type != object_type:
                continue
            printed_id = self._caption_explicit_label(candidate.text)
            if printed_id is None:
                continue
            claims_by_printed_id.setdefault(printed_id, set()).add(assignment.object_id)
            claims_by_object_id.setdefault(assignment.object_id, set()).add(printed_id)
            claim_records.append((assignment.caption_id, assignment.object_id, printed_id))

        conflicting_printed_ids = {
            printed_id
            for printed_id, object_ids in claims_by_printed_id.items()
            if len(object_ids) > 1
        }
        conflicting_object_ids = {
            object_id
            for object_id, printed_ids in claims_by_object_id.items()
            if len(printed_ids) > 1
        }
        conflict_caption_ids = [
            caption_id
            for caption_id, object_id, printed_id in claim_records
            if printed_id in conflicting_printed_ids or object_id in conflicting_object_ids
        ]
        reserved_by_object_id = {
            next(iter(object_ids)): printed_id
            for printed_id, object_ids in claims_by_printed_id.items()
            if len(object_ids) == 1 and next(iter(object_ids)) not in conflicting_object_ids
        }

        used_ids = set(reserved_by_object_id.values())
        next_unclaimed_id = max(claims_by_printed_id, default=0) + 1
        remapped_ids: dict[str, str] = {}
        for old_object_id, item in old_object_ids.items():
            final_id = reserved_by_object_id.get(old_object_id)
            if final_id is None:
                while next_unclaimed_id in used_ids:
                    next_unclaimed_id += 1
                final_id = next_unclaimed_id
                next_unclaimed_id += 1
            used_ids.add(final_id)
            setattr(item, id_attribute, final_id)
            remapped_ids[old_object_id] = f"{object_type}:{final_id}"
        objects.sort(key=lambda item: getattr(item, id_attribute))
        return remapped_ids, conflict_caption_ids

    def _reconcile_media_ids(
        self,
        assignments: list[CaptionAssignment],
        candidate_by_id: dict[str, CaptionCandidate],
    ) -> list[CaptionAssignment]:
        figure_remap, figure_conflicts = self._reconcile_object_ids(
            self.figures,
            "figure",
            "figure_id",
            assignments,
            candidate_by_id,
        )
        table_remap, table_conflicts = self._reconcile_object_ids(
            self.tables,
            "table",
            "table_id",
            assignments,
            candidate_by_id,
        )
        conflicts = [*figure_conflicts, *table_conflicts]
        if conflicts:
            self._structure_validation_issues.append(
                ValidationIssue(
                    code="VAL_MEDIA_ID_CONFLICT",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"{len(conflicts)} explicit media caption(s) claim conflicting printed IDs"
                    ),
                    origin_stage="structure",
                    evidence_ids=tuple(conflicts),
                    count=len(conflicts),
                )
            )
        remapped_ids = {**figure_remap, **table_remap}
        return [
            CaptionAssignment(
                caption_id=item.caption_id,
                object_id=remapped_ids.get(item.object_id, item.object_id),
                score=item.score,
                reasons=item.reasons,
                ambiguous=item.ambiguous,
            )
            for item in assignments
        ]

    @classmethod
    def _caption_doi_urls(cls, text: str) -> set[str]:
        return {
            match.group(0).rstrip(".,;)").casefold() for match in cls._DOI_URL_RE.finditer(text)
        }

    def _figure_targets(self) -> list[CaptionTarget]:
        targets: list[CaptionTarget] = []
        for figure in self.figures:
            last_part = figure.parts[-1] if figure.parts else None
            targets.append(
                CaptionTarget(
                    object_id=f"figure:{figure.figure_id}",
                    object_type="figure",
                    page_number=last_part.page_number if last_part else figure.page_number,
                    bbox=last_part.bbox if last_part else None,
                    source_index=self._figure_source_indices[id(figure)],
                )
            )
        return targets

    def _table_targets(self) -> list[CaptionTarget]:
        targets: list[CaptionTarget] = []
        for table in self.tables:
            first_part = table.parts[0] if table.parts else None
            targets.append(
                CaptionTarget(
                    object_id=f"table:{table.table_id}",
                    object_type="table",
                    page_number=first_part.page_number if first_part else table.page_number,
                    bbox=first_part.bbox if first_part else None,
                    source_index=self._table_source_indices[id(table)],
                )
            )
        return targets

    def _group_panel_figures(
        self, candidates: list[CaptionCandidate]
    ) -> tuple[dict[str, list[str]], dict[str, int], dict[str, int], dict[str, int]]:
        descriptions: dict[str, list[str]] = {}
        panel_owners: dict[str, int] = {}
        supporting_owners: dict[str, int] = {}
        caption_owners: dict[str, int] = {}
        previous_caption_source = -1
        grouped_ids: set[int] = set()
        explicit_captions = sorted(
            [
                item
                for item in candidates
                if item.object_type == "figure" and self._EXPLICIT_FIGURE_RE.match(item.text)
            ],
            key=lambda item: item.source_index,
        )
        figure_sources = sorted(self._figure_source_indices.values())
        for caption_index, caption in enumerate(explicit_captions):
            boundary = max(
                [
                    previous_caption_source,
                    *(
                        barrier
                        for barrier in self._caption_barriers
                        if previous_caption_source < barrier < caption.source_index
                    ),
                ]
            )
            eligible = [
                figure
                for figure in self.figures
                if id(figure) not in grouped_ids
                and boundary < self._figure_source_indices[id(figure)] < caption.source_index
            ]
            previous_caption_source = caption.source_index
            if len(eligible) < 2:
                continue
            eligible.sort(key=lambda item: self._figure_source_indices[id(item)])
            trailing: list[PaperFigure] = [eligible[-1]]
            for figure in reversed(eligible[:-1]):
                figure_page = figure.parts[-1].page_number if figure.parts else figure.page_number
                next_figure = trailing[0]
                next_page = (
                    next_figure.parts[-1].page_number
                    if next_figure.parts
                    else next_figure.page_number
                )
                if (
                    figure_page is None
                    or next_page is None
                    or next_page - figure_page not in {0, 1}
                    or caption.page_number is None
                    or caption.page_number - figure_page > 3
                ):
                    break
                trailing.insert(0, figure)
            if len(trailing) < 2:
                continue

            next_figure_source = next(
                (source for source in figure_sources if source > caption.source_index),
                float("inf"),
            )
            current_label = self._caption_printed_label(caption)
            future_captions: list[CaptionCandidate] = []
            future_labels: set[str] = set()
            for later in explicit_captions[caption_index + 1 :]:
                label = self._caption_printed_label(later)
                if (
                    later.page_number is None
                    or caption.page_number is None
                    or not 0 <= later.page_number - caption.page_number <= 3
                    or later.source_index >= next_figure_source
                    or any(
                        caption.source_index < barrier < later.source_index
                        for barrier in self._caption_barriers
                    )
                    or label is None
                    or label == current_label
                    or label in future_labels
                ):
                    continue
                future_labels.add(label)
                future_captions.append(later)
            reserved_count = min(len(future_captions), len(trailing))
            if len(trailing) - reserved_count < 2:
                continue
            grouped_members = trailing[:-reserved_count] if reserved_count else trailing
            reserved_members = trailing[-reserved_count:] if reserved_count else []

            first_source = self._figure_source_indices[id(trailing[0])]
            evidence = [
                item
                for item in candidates
                if item.object_type == "figure"
                and first_source - 1 <= item.source_index < caption.source_index
                and self._is_short_panel_candidate(item)
            ]
            if len(evidence) < len(trailing):
                continue
            panel_targets = [
                target
                for target in self._figure_targets()
                if any(target.object_id == f"figure:{figure.figure_id}" for figure in trailing)
            ]
            evidence_assignments = assign_captions(
                evidence,
                panel_targets,
                ambiguity_margin=-1.0,
            )
            target_ids = {assignment.object_id for assignment in evidence_assignments}
            if None in target_ids or target_ids != {target.object_id for target in panel_targets}:
                continue
            primary = grouped_members[0]
            for member in grouped_members[1:]:
                primary.parts.extend(member.parts)
                primary.provenance.extend(member.provenance)
                grouped_ids.add(id(member))
            # The whole figure, not just its first panel's crop.
            primary.image_b64 = composite_panel_image(primary.parts) or primary.image_b64
            self._figure_source_indices[id(primary)] = max(
                self._figure_source_indices[id(item)] for item in grouped_members
            )
            evidence_by_id = {item.caption_id: item for item in evidence}
            grouped_target_ids = {f"figure:{figure.figure_id}" for figure in grouped_members}
            reserved_by_target = {
                f"figure:{figure.figure_id}": figure for figure in reserved_members
            }
            grouped_evidence = [
                evidence_by_id[item.caption_id]
                for item in evidence_assignments
                if item.object_id in grouped_target_ids
            ]
            descriptions[caption.caption_id] = [item.text for item in grouped_evidence]
            panel_owners.update({item.caption_id: id(primary) for item in grouped_evidence})
            caption_owners[caption.caption_id] = id(primary)
            caption_owners.update(
                {
                    future_caption.caption_id: id(member)
                    for future_caption, member in zip(
                        future_captions[:reserved_count], reserved_members, strict=True
                    )
                }
            )
            supporting_owners.update(
                {
                    item.caption_id: id(reserved_by_target[item.object_id])
                    for item in evidence_assignments
                    if item.object_id in reserved_by_target
                }
            )
        if grouped_ids:
            self.figures = [figure for figure in self.figures if id(figure) not in grouped_ids]
        return descriptions, panel_owners, supporting_owners, caption_owners

    def _is_short_panel_candidate(self, candidate: CaptionCandidate) -> bool:
        text = self._caption_useful_text(candidate.text)
        return bool(
            text
            and not self._EXPLICIT_FIGURE_RE.match(text)
            and len(text) <= 60
            and len(text.split()) <= 6
            and not text.endswith((".", ":", ";", "?", "!"))
        )

    @staticmethod
    def _table_label(caption: str | None) -> str | None:
        # The whole printed label, so "Table 3.1" and "Table 3.2" on adjacent
        # pages are two tables, not one continued.
        label = caption_label(caption, "table")
        return normalize_label(label) if label is not None else None

    def _group_continuation_tables(
        self, candidates: list[CaptionCandidate]
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        fragment_ids = set(self._table_caption_fragments.values())
        table_candidates = [
            item
            for item in candidates
            if item.object_type == "table" and item.caption_id not in fragment_ids
        ]
        table_targets = self._table_targets()
        preliminary: list[CaptionAssignment] = []
        for page_number in sorted(
            {
                item.page_number
                for item in [*table_candidates, *table_targets]
                if item.page_number is not None
            }
        ):
            preliminary.extend(
                assign_captions(
                    [item for item in table_candidates if item.page_number == page_number],
                    [item for item in table_targets if item.page_number == page_number],
                )
            )
        candidate_by_id = {item.caption_id: item for item in candidates}
        caption_by_object = {
            assignment.object_id: candidate_by_id[assignment.caption_id]
            for assignment in preliminary
            if assignment.object_id is not None
        }
        for table in self.tables:
            confirmed_caption_id = next(
                (
                    caption_id
                    for caption_id, owner_id in self._confirmed_table_caption_owners.items()
                    if owner_id == id(table)
                ),
                None,
            )
            if confirmed_caption_id is not None:
                # The id was recorded at parse time; de-duplication may since
                # have elected a different cluster member as canonical, and
                # only canonical members are in ``candidate_by_id``. Indexing
                # it directly raised KeyError, which ParseSegmentStage turned
                # into parse_failed — dropping the whole paper.
                canonical_id = self._caption_canonical_by_id.get(
                    confirmed_caption_id, confirmed_caption_id
                )
                confirmed_candidate = candidate_by_id.get(canonical_id)
                if confirmed_candidate is not None:
                    caption_by_object[f"table:{table.table_id}"] = confirmed_candidate
        for table in self.tables:
            candidate = caption_by_object.get(f"table:{table.table_id}")
            table.caption = candidate.text if candidate else None

        base_candidate_by_identity = {
            id(table): caption_by_object.get(f"table:{table.table_id}") for table in self.tables
        }
        continuation_owners: dict[str, int] = {}
        grouped: list[PaperTable] = []
        group_last_source: dict[int, int] = {}
        for table in self.tables:
            if not grouped:
                grouped.append(table)
                group_last_source[id(table)] = self._table_source_indices[id(table)]
                continue
            previous = grouped[-1]
            previous_candidate = base_candidate_by_identity.get(id(previous))
            current_candidate = base_candidate_by_identity.get(id(table))
            previous_label = self._table_label(previous.caption)
            current_label = self._table_label(table.caption)
            compatible_columns = [str(item) for item in previous.df.columns] == [
                str(item) for item in table.df.columns
            ]
            adjacent_page = (
                previous.parts
                and table.parts
                and previous.parts[-1].page_number is not None
                and table.parts[0].page_number == previous.parts[-1].page_number + 1
            )
            explicit_continuation = bool(
                table.caption and re.search(r"\bcontin(?:ued|uation)\b", table.caption, re.I)
            )
            same_explicit_label = previous_label is not None and previous_label == current_label
            continuation_of_previous = bool(
                explicit_continuation and previous_label is not None and current_label is None
            )
            previous_last_source = group_last_source[id(previous)]
            current_source = self._table_source_indices[id(table)]
            no_heading_barrier = not any(
                previous_last_source < barrier < current_source
                for barrier in self._caption_barriers
            )
            no_new_table_label = not any(
                item.object_type == "table"
                and item.caption_id not in fragment_ids
                and previous_last_source < item.source_index < current_source
                for item in candidates
            )
            previous_bbox = previous.parts[-1].bbox if previous.parts else None
            current_bbox = table.parts[0].bbox if table.parts else None
            page_edge_continuation = bool(
                previous_bbox is not None
                and current_bbox is not None
                and previous_bbox[3] >= 800
                and current_bbox[1] <= 200
            )
            implicit_repeated_header = bool(
                current_candidate is None
                and previous_label is not None
                and previous_candidate is not None
                and previous_candidate.caption_id in self._table_caption_fragments
                and page_edge_continuation
                and no_new_table_label
            )
            if not (
                adjacent_page
                and compatible_columns
                and no_heading_barrier
                and (same_explicit_label or continuation_of_previous or implicit_repeated_header)
            ):
                grouped.append(table)
                group_last_source[id(table)] = current_source
                continue
            previous.parts.extend(table.parts)
            previous.provenance.extend(table.provenance)
            previous.df = pd.concat([previous.df, table.df], ignore_index=True)
            # Keep each printed piece's source markup (rowspans, multi-level
            # headers) rather than re-rendering the merged frame, which is lossy
            # (see ``_handle_table``); the merged frame still feeds ``contents``.
            previous.tbl_html = "\n".join(
                part.tbl_html for part in previous.parts if part.tbl_html
            ) or previous.df.to_html(index=False)
            self._table_source_indices[id(previous)] = min(
                self._table_source_indices[id(previous)], self._table_source_indices[id(table)]
            )
            group_last_source[id(previous)] = current_source
            if current_candidate is not None and (explicit_continuation or same_explicit_label):
                continuation_owners[current_candidate.caption_id] = id(previous)
        self.tables = grouped
        fragment_owners = {
            self._table_caption_fragments[candidate.caption_id]: id(table)
            for table in grouped
            if (candidate := base_candidate_by_identity.get(id(table))) is not None
            and candidate.caption_id in self._table_caption_fragments
        }
        confirmed_owners = {
            candidate.caption_id: id(table)
            for table in grouped
            if (candidate := base_candidate_by_identity.get(id(table))) is not None
            and candidate.caption_id in self._confirmed_table_caption_owners
        }
        return continuation_owners, fragment_owners, confirmed_owners

    def _is_decoration_shaped(self, figure: PaperFigure) -> bool:
        """Does this figure have publisher-furniture geometry?

        Geometry and region label only — ownership is the caller's business
        (see :meth:`_suppress_unowned_decoration_figures`). A multipart figure
        is never decoration: parts are only ever joined by panel grouping,
        which is itself caption evidence.
        """
        if self._figure_source_labels_map().get(id(figure)) not in self._DECORATION_LABELS:
            return False
        if len(figure.parts) != 1:
            return False
        part = figure.parts[0]
        if part.bbox is None or part.page_number is None:
            return False
        x1, y1, x2, y2 = part.bbox
        height = y2 - y1
        if height > self._DECORATION_MAX_HEIGHT:
            return False
        if (x2 - x1) * height > self._DECORATION_MAX_AREA:
            return False
        if self._is_front_page(part.page_number):
            # Front-page badges sit anywhere: beside the byline, under the
            # masthead, against the DOI block. No band restriction applies.
            return True
        # Past the front page only page furniture qualifies — a badge-sized
        # region inside the body column there is a real (small) float.
        return (
            y1 >= self._DECORATION_FOOTER_TOP
            or x2 <= self._DECORATION_MARGIN
            or x1 >= self._PAGE_SPAN - self._DECORATION_MARGIN
        )

    def _figure_source_labels_map(self) -> dict[int, str | None]:
        """Region label per figure identity, empty when no figure was handled."""
        return getattr(self, "_figure_source_labels", {})

    def _assignment_veto_reason(
        self,
        candidate: CaptionCandidate,
        target: CaptionTarget | None,
        reasons: tuple[str, ...],
    ) -> str | None:
        """Why the parse refuses a matcher edge, or ``None`` when it stands.

        Called once, from the assignment loop in :meth:`_finalize_media`. A
        vetoed edge becomes an ``object_id=None`` assignment, so decoration
        suppression sees the veto without replaying it.
        """
        if target is None:
            return None
        bare_table_label_nearby = (
            candidate.object_type == "table"
            and bool(self._BARE_TABLE_LABEL_RE.fullmatch(candidate.text))
            and abs(candidate.source_index - target.source_index) <= 2
        )
        if not bare_table_label_nearby and any(
            min(candidate.source_index, target.source_index)
            < barrier
            < max(candidate.source_index, target.source_index)
            for barrier in self._caption_barriers
        ):
            return "heading_boundary"
        if "adjacent_page" in reasons and candidate.page_number != target.page_number:
            return "adjacent_page_not_authorized"
        return None

    def _suppress_unowned_decoration_figures(
        self,
        finalized: list[CaptionAssignment],
        figure_by_id: dict[str, PaperFigure],
    ) -> None:
        """Drop badge-shaped figures the finished assignment left unowned.

        *finalized* is the assignment list ``_finalize_media`` has already
        built: grouping owners, the matcher pass with its vetoes applied, the
        non-caption abstentions and the duplicate abstentions. Ownership is
        therefore read off the real run rather than modelled — same candidate
        set, same vetoes, same target ids, same matcher call — which is what
        two earlier preview-based versions of this method could not achieve.

        Ownership is the only permission this method has. Both code paths that
        set ``PaperFigure.caption`` in ``_finalize_media`` — the panel-caption
        loop and the assignment loop — put the matching assignment into
        *finalized*, so a figure absent from ``owned`` has ``caption is None``
        at this instant. Asserted, not assumed, by
        ``test_suppression_reads_captions_that_are_already_attached``: nothing
        dropped carried a caption, and the figures kept already had theirs.

        What this does NOT bound: a decoration-shaped region is a live matcher
        target, so it can win an explicit caption and survive — captioned and
        spurious. See ``test_a_badge_that_wins_a_caption_survives_captioned``.
        The geometry predicate is not consulted before matching and this method
        never overrides an assignment, so that case is out of reach here.
        """
        owned = {item.object_id for item in finalized if item.object_id is not None}
        dropped = [
            figure
            for figure in self.figures
            if f"figure:{figure.figure_id}" not in owned and self._is_decoration_shaped(figure)
        ]
        if not dropped:
            return
        logger.debug("Dropping %d unowned decoration figure(s)", len(dropped))
        dropped_identities = {id(figure) for figure in dropped}
        self.figures = [figure for figure in self.figures if id(figure) not in dropped_identities]
        # ``figure_by_id`` is not read again below today; kept in step so a
        # later statement cannot resolve a figure that is no longer exported.
        for figure in dropped:
            figure_by_id.pop(f"figure:{figure.figure_id}", None)

    def _finalize_media(self) -> CaptionAssignmentReceipt:
        """Group explicit multipart media, assign captions globally, and retain abstentions."""

        active_candidates, duplicate_assignments = self._deduplicate_caption_candidates()
        (
            panel_descriptions,
            panel_owner_ids,
            supporting_figure_owner_ids,
            panel_caption_owner_ids,
        ) = self._group_panel_figures(active_candidates)
        (
            continuation_owners,
            fragment_owners,
            confirmed_table_caption_owners,
        ) = self._group_continuation_tables(active_candidates)

        for index, figure in enumerate(self.figures, 1):
            figure.figure_id = index
            figure.caption = None
        for index, table in enumerate(self.tables, 1):
            table.table_id = index
            table.caption = None

        matching_candidates = [
            candidate
            for candidate in active_candidates
            if candidate.caption_id not in panel_owner_ids
            and candidate.caption_id not in supporting_figure_owner_ids
            and candidate.caption_id not in panel_caption_owner_ids
            and candidate.caption_id not in continuation_owners
            and candidate.caption_id not in fragment_owners
            and candidate.caption_id not in confirmed_table_caption_owners
            and candidate.caption_id not in self._non_caption_candidate_reasons
        ]
        assignments = list(
            assign_captions(
                matching_candidates,
                [*self._figure_targets(), *self._table_targets()],
            )
        )
        candidate_by_id = {item.caption_id: item for item in self._caption_candidates}
        figure_by_id = {f"figure:{item.figure_id}": item for item in self.figures}
        table_by_id = {f"table:{item.table_id}": item for item in self.tables}
        target_by_id = {
            target.object_id: target for target in [*self._figure_targets(), *self._table_targets()]
        }
        figure_id_by_identity = {
            id(figure): f"figure:{figure.figure_id}" for figure in self.figures
        }
        table_id_by_identity = {id(table): f"table:{table.table_id}" for table in self.tables}
        finalized: list[CaptionAssignment] = [
            CaptionAssignment(
                caption_id=caption_id,
                object_id=figure_id_by_identity[owner_id],
                score=0.0,
                reasons=("panel_evidence", "bounded_sequence"),
            )
            for caption_id, owner_id in panel_owner_ids.items()
        ]
        finalized.extend(
            CaptionAssignment(
                caption_id=caption_id,
                object_id=figure_id_by_identity[owner_id],
                score=0.0,
                reasons=("owner_evidence", "bounded_sequence"),
            )
            for caption_id, owner_id in supporting_figure_owner_ids.items()
        )
        finalized.extend(
            CaptionAssignment(
                caption_id=caption_id,
                object_id=figure_id_by_identity[owner_id],
                score=0.0,
                reasons=(
                    "panel_caption",
                    "bounded_sequence",
                    f"panel_evidence:{len(panel_descriptions.get(caption_id, []))}",
                ),
            )
            for caption_id, owner_id in panel_caption_owner_ids.items()
        )
        finalized.extend(
            CaptionAssignment(
                caption_id=caption_id,
                object_id=table_id_by_identity[owner_id],
                score=0.0,
                reasons=("confirmed_caption_sequence",),
            )
            for caption_id, owner_id in confirmed_table_caption_owners.items()
        )
        finalized.extend(
            CaptionAssignment(
                caption_id=caption_id,
                object_id=table_id_by_identity[owner_id],
                score=0.0,
                reasons=("continuation_evidence",),
            )
            for caption_id, owner_id in continuation_owners.items()
        )
        finalized.extend(
            CaptionAssignment(
                caption_id=caption_id,
                object_id=table_id_by_identity[owner_id],
                score=0.0,
                reasons=("caption_fragment", "confirmed_by_table"),
            )
            for caption_id, owner_id in fragment_owners.items()
        )
        active_ids = {candidate.caption_id for candidate in active_candidates}
        finalized.extend(
            CaptionAssignment(
                caption_id=caption_id,
                object_id=None,
                score=0.0,
                reasons=reasons,
            )
            for caption_id, reasons in self._non_caption_candidate_reasons.items()
            if caption_id in active_ids
        )
        for caption_id, owner_id in confirmed_table_caption_owners.items():
            text = self._caption_display_text_by_id.get(
                caption_id, candidate_by_id[caption_id].text
            )
            fragment_id = self._table_caption_fragments[caption_id]
            table_by_id[
                table_id_by_identity[owner_id]
            ].caption = f"{text} {candidate_by_id[fragment_id].text}".strip()
        for caption_id, owner_id in panel_caption_owner_ids.items():
            text = self._caption_display_text_by_id.get(
                caption_id, candidate_by_id[caption_id].text
            )
            evidence = panel_descriptions.get(caption_id, [])
            descriptions = [item for item in evidence if self._panel_description(item)]
            if descriptions:
                text = f"{text} | {'; '.join(descriptions)}"
            figure_by_id[figure_id_by_identity[owner_id]].caption = text
        for assignment in assignments:
            candidate = candidate_by_id[assignment.caption_id]
            target = target_by_id.get(assignment.object_id or "")
            veto_reason = self._assignment_veto_reason(candidate, target, assignment.reasons)
            if veto_reason is not None:
                assignment = CaptionAssignment(
                    assignment.caption_id,
                    None,
                    0.0,
                    (*assignment.reasons, veto_reason),
                )
            elif assignment.object_id in figure_by_id:
                text = self._caption_display_text_by_id.get(candidate.caption_id, candidate.text)
                evidence = panel_descriptions.get(candidate.caption_id, [])
                descriptions = [item for item in evidence if self._panel_description(item)]
                if descriptions:
                    text = f"{text} | {'; '.join(descriptions)}"
                figure_by_id[assignment.object_id].caption = text
                if evidence:
                    assignment = CaptionAssignment(
                        assignment.caption_id,
                        assignment.object_id,
                        assignment.score,
                        (*assignment.reasons, f"panel_evidence:{len(evidence)}"),
                        assignment.ambiguous,
                    )
            elif assignment.object_id in table_by_id:
                text = self._caption_display_text_by_id.get(candidate.caption_id, candidate.text)
                if fragment_id := self._table_caption_fragments.get(candidate.caption_id):
                    text = f"{text} {candidate_by_id[fragment_id].text}".strip()
                table_by_id[assignment.object_id].caption = text
            finalized.append(assignment)

        finalized.extend(duplicate_assignments)
        # Every caption is attached now. Its printed label is what in-text
        # mentions resolve by (``detect_xrefs``); the ids stay provisional.
        for figure in self.figures:
            figure.label = caption_label(figure.caption, "figure")
        for table in self.tables:
            table.label = caption_label(table.caption, "table")
        # ``finalized`` is complete here, so unowned means unowned. Delete
        # before ``_reconcile_media_ids`` renumbers, and it renumbers the
        # survivors only; no assignment can name a deleted figure, so the list
        # it rewrites needs no repair.
        self._suppress_unowned_decoration_figures(finalized, figure_by_id)
        finalized = self._reconcile_media_ids(finalized, candidate_by_id)
        duplicate_ids = {item.caption_id for item in duplicate_assignments}
        ownership_failures = [
            item
            for item in finalized
            if (
                item.object_id is None
                and item.caption_id not in duplicate_ids
                and item.caption_id not in self._non_caption_candidate_reasons
                and (
                    self._EXPLICIT_FIGURE_RE.match(candidate_by_id[item.caption_id].text)
                    or self._TABLE_LABEL_RE.match(candidate_by_id[item.caption_id].text)
                )
            )
        ]
        if ownership_failures:
            self._structure_validation_issues.append(
                ValidationIssue(
                    code="VAL_CAPTION_OWNERSHIP",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"{len(ownership_failures)} explicit caption(s) could not be assigned "
                        "unambiguously"
                    ),
                    origin_stage="structure",
                    evidence_ids=tuple(item.caption_id for item in ownership_failures),
                    count=len(ownership_failures),
                )
            )
        assignment_by_id = {item.caption_id: item for item in finalized}
        return CaptionAssignmentReceipt(
            candidates=tuple(self._caption_candidates),
            assignments=tuple(
                assignment_by_id[item.caption_id] for item in self._caption_candidates
            ),
        )

    @staticmethod
    def _parse_html_table(html: str) -> pd.DataFrame | None:
        """Parse an HTML ``<table>`` string into a DataFrame.

        Uses :func:`pandas.read_html` with the ``html5lib`` parser (stdlib
        fallback) to handle ``<tr>/<td>/<th>`` markup returned by the OCR
        engine.  Returns ``None`` on any parse failure.
        """
        try:
            from io import StringIO

            dfs = pd.read_html(StringIO(html), flavor="html5lib")
            if dfs:
                df = dfs[0].fillna("")
                # When OCR HTML lacks <th> tags, pandas assigns integer column
                # names (0, 1, 2, …).  Promote the first data row to headers.
                if len(df) > 0 and all(isinstance(c, (int, float)) for c in df.columns):
                    df.columns = [str(v) for v in df.iloc[0]]
                    df = df.iloc[1:].reset_index(drop=True)
                return df
        except Exception:
            logger.debug("HTML table parse failed, falling back to markdown parser")
        return None

    @classmethod
    def _parse_markdown_table(cls, markdown: str) -> pd.DataFrame | None:
        """Parse a markdown table string into a DataFrame.

        Handles the standard GFM format with ``|`` column delimiters and
        a ``---`` separator row between header and body.
        """
        lines = [s for line in markdown.strip().splitlines() if (s := line.strip())]
        if not lines:
            return None

        # Find the separator row
        separator_idx = None
        for i, line in enumerate(lines):
            if cls._MARKDOWN_TABLE_SEPARATOR.match(line):
                separator_idx = i
                break

        if separator_idx is not None and separator_idx > 0:
            header_lines = lines[:separator_idx]
            body_lines = lines[separator_idx + 1 :]
            # Every pre-separator line is header. Taking only the last one
            # deleted the spanner rows of a hierarchical header ("Group A" /
            # "Group B" above "Mean" / "SD"), so the surviving names were
            # ambiguous duplicates and the grouping printed in the paper was
            # gone from the export.
            columns = cls._merge_header_rows([cls._split_row(line) for line in header_lines])
            rows = [cls._split_row(line) for line in body_lines if line]
        else:
            # No separator found — treat first row as header
            columns = cls._split_row(lines[0])
            rows = [cls._split_row(line) for line in lines[1:] if line]

        if not columns:
            return None

        # Pad rows out to the widest row present. Truncating to the header
        # width silently deleted printed cells whenever a body row carried more
        # columns than the header — common with a stray/escaped pipe or a
        # header whose leading label cell is blank.
        n_cols = max([len(columns), *(len(row) for row in rows)]) if rows else len(columns)
        if n_cols > len(columns):
            logger.debug(
                "Markdown table body is wider than its header (%d > %d); widening",
                n_cols,
                len(columns),
            )
            columns = columns + [""] * (n_cols - len(columns))
        padded = [row + [""] * (n_cols - len(row)) for row in rows]

        df = pd.DataFrame(padded, columns=columns) if padded else pd.DataFrame(columns=columns)
        return df.fillna("")

    @staticmethod
    def _merge_header_rows(header_rows: list[list[str]]) -> list[str]:
        """Flatten a multi-row markdown header into one name per column.

        Cells are joined top-to-bottom with a space, skipping empties, so a
        spanner ("Group A" over "Mean"/"SD") becomes "Group A Mean" /
        "Group A SD" — the same flattening the HTML table path applies to
        rowspan/colspan headers. A single header row is returned unchanged.
        """
        rows = [row for row in header_rows if row]
        if not rows:
            return []
        if len(rows) == 1:
            return rows[0]
        width = max(len(row) for row in rows)
        merged = []
        for col in range(width):
            parts = [row[col].strip() for row in rows if col < len(row) and row[col].strip()]
            merged.append(" ".join(parts))
        return merged

    @staticmethod
    def _split_row(line: str) -> list[str]:
        """Split a markdown table row on unescaped ``|`` delimiters.

        A cell may contain a literal pipe as ``\\|`` (GFM's escape, and what
        OCR emits for a printed "|" inside a cell). Splitting on every pipe
        tore those cells in two and shifted every column after them.
        """
        # Strip leading/trailing pipes — but not an escaped trailing one.
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|") and not stripped.endswith("\\|"):
            stripped = stripped[:-1]
        return [cell.strip().replace("\\|", "|") for cell in _UNESCAPED_PIPE_RE.split(stripped)]
