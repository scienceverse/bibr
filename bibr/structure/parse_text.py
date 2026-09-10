"""Body-text / formula / footnote region handlers for :class:`PDFParser`.

Extracted from :mod:`bibr.structure.pdf_parser` as :class:`TextHandlersMixin`.
Methods move verbatim (only ``text_repair.bbox_to_tuple`` is spelled with its
public name, and the static ``_should_join`` now references
``TextHandlersMixin._has_terminal_punct`` rather than ``PDFParser`` to avoid a
circular import); shared state (``self.assembler``, carry-over, counters) lives
on :class:`PDFParser` and is reached through ``self``.
"""

import logging
from collections.abc import Set as AbstractSet

from bibr.input.consolidate_text import (
    DOI_URL_CONTEXT_RE,
    clean_formula_text,
    clean_text_content,
    strip_affiliation_markers,
)
from bibr.paper_contents import (
    CanonicalSection,
    PaperSection,
    PaperSentence,
    PaperURLLink,
    Provenance,
)
from bibr.structure.assembler import DeferredText
from bibr.structure.reference_boundaries import is_publisher_note_boundary
from bibr.structure.text_repair import bbox_to_tuple
from bibr.structure.xref_utils import URL_RE
from bibr.utils.text import clean_extracted_url, normalize_text

logger = logging.getLogger(__name__)


class TextHandlersMixin:
    """Content, formula, footnote, URL, and continuity handlers for ``PDFParser``."""

    def _handle_content(
        self,
        content: str,
        page_number: int,
        bbox: list | None = None,
        region_meta: dict | None = None,
    ) -> None:
        """Accumulate body text, handling cross-page continuity."""
        text = clean_text_content(content.strip())
        if not text:
            return

        # Strip affiliation markers (e.g. "Author ^{1,2}") from pre-section
        # content on page 1 (the author byline zone).  These are NOT citation
        # superscripts and would otherwise produce spurious bib xrefs.
        if self._is_front_page(page_number) and self._current_section_id == 0:
            text = strip_affiliation_markers(text)

        self._start_publisher_note_back_matter(text, page_number, bbox)

        # Detect table/figure captions that OCR labeled as plain content.
        # Route them through the caption handlers so they get attached to
        # their adjacent table/figure rather than appearing as body text.
        if self._TABLE_CAPTION_RE.match(text):
            self._handle_table_caption(text, bbox, page_number)
            return
        if self._FIGURE_CAPTION_RE.match(text):
            self._handle_figure_caption(text, bbox, page_number)
            return

        heading = self._promotable_content_heading(text, region_meta)
        if heading is not None:
            self._flush_carry_over()
            self._handle_heading("paragraph_title", heading, page_number, bbox)
            return

        if self._stage_pending_table_label_fragment(
            text, bbox, page_number, region_meta=region_meta
        ):
            return

        self._emit_content_without_promotion(text, page_number, bbox, region_meta=region_meta)

    def _start_publisher_note_back_matter(
        self,
        text: str,
        page_number: int,
        bbox: list | None,
    ) -> bool:
        """End an active References section at unmistakable publisher boilerplate."""
        if not is_publisher_note_boundary(text):
            return False

        current = next(
            (
                section
                for section in reversed(self.sections)
                if section.section_id == self._current_section_id
            ),
            None,
        )
        if current is None:
            return False

        from bibr.structure.section_classifier import _classify_lookup

        current_type, _score = _classify_lookup(normalize_text(current.header))
        if current_type != CanonicalSection.REFERENCES:
            return False

        self._flush_carry_over()
        self._section_counter += 1
        self.sections.append(
            PaperSection(
                section_id=self._section_counter,
                header="Publisher's Note",
                level=1,
                parent_section_id=0,
                provenance=[Provenance(page_no=page_number, bbox=bbox_to_tuple(bbox))],
            )
        )
        self._current_section_id = self._section_counter
        self._terminal_reference_tail_section_id = self._current_section_id
        logger.info("Opened Publisher's Note back matter after References on page %d", page_number)
        return True

    def _emit_content_without_promotion(
        self,
        content: str,
        page_number: int,
        bbox: list | None = None,
        *,
        region_meta: dict | None = None,
    ) -> None:
        """Emit body text without allowing it to re-enter heading promotion."""
        text = clean_text_content(content.strip())
        if not text:
            return

        if self._is_front_page(page_number) and self._current_section_id == 0:
            text = strip_affiliation_markers(text)

        prov = Provenance(page_no=page_number, bbox=bbox_to_tuple(bbox))

        if self._carry_over.text:
            # Check if this continues the previous text (no terminal punct + lowercase start).
            same_page = self._carry_over.last_page == page_number
            same_section = self._carry_over.section_id == self._current_section_id
            cross_page_continuation = (
                not same_page
                and same_section
                and self._starts_with_lowercase(text)
                and self._should_join(self._carry_over.text, text)
            )
            if (same_page and self._should_join(self._carry_over.text, text)) or (
                cross_page_continuation
            ):
                joiner = "" if self._is_url_wrap_join(self._carry_over.text, text) else " "
                if not same_page and isinstance(self._carry_over.page, int):
                    # Record where this page's text starts so the flushed
                    # sentences are attributed to the page they were printed on
                    # rather than to the page the paragraph started on.
                    if not self._carry_over.page_spans:
                        self._carry_over.page_spans.append((0, self._carry_over.page))
                    self._carry_over.page_spans.append(
                        (len(self._carry_over.text) + len(joiner), page_number)
                    )
                self._carry_over.text += joiner + text
                self._carry_over.provenance.append(prov)
                # Advance the cursor to the page just appended. Left stale, a
                # third region on the *new* page compares against the page the
                # paragraph started on, computes ``same_page=False``, and is
                # then judged by the stricter cross-page rule — so a
                # capitalised continuation of the same sentence is flushed
                # apart mid-sentence. ``page`` itself stays on the start page:
                # it stamps the flushed entry and seeds ``page_spans[0]``.
                self._carry_over.last_page = page_number
                return
            else:
                # Flush the carry-over as complete
                self._flush_carry_over()

        # Check if this text block ends without terminal punctuation
        if not self._has_terminal_punct(text):
            self._carry_over.text = text
            self._carry_over.page = page_number
            self._carry_over.last_page = page_number
            self._carry_over.provenance = [prov]
            self._carry_over.section_id = self._current_section_id
            self._carry_over.region_meta = region_meta
        else:
            self._emit_sentences(text, page_number, [prov], region_meta=region_meta)

    def _handle_formula(
        self,
        content: str,
        page_number: int,
        bbox: list | None = None,
        region_meta: dict | None = None,
    ) -> None:
        """Process formula content and emit as a display-formula sentence."""
        self._expire_pending_table_label_fragment()
        text = content.strip()
        if not text:
            return

        # Wrap in display math temporarily for clean_formula_text, which
        # expects $$ delimiters to detect \text{} wrappers and broken blocks.
        if not text.startswith("$"):
            text = f"$${text}$$"

        # Clean formulas that are actually garbled OCR or \text{} wrappers.
        # clean_formula_text returns None for broken blocks that should be
        # discarded, or unwrapped plain text for \text{} wrappers.
        cleaned = clean_formula_text(text)
        if cleaned is None:
            return

        # If the formula was unwrapped to plain text (no longer a formula),
        # route it through the normal content handler instead.
        if not cleaned.startswith("$"):
            self._handle_content(cleaned, page_number, bbox, region_meta=region_meta)
            return

        # Strip the $$ delimiters — the is_display_formula flag replaces them.
        inner = cleaned
        if inner.startswith("$$") and inner.endswith("$$"):
            inner = inner[2:-2].strip()

        # Flush carry-over before formula
        self._flush_carry_over()

        # Defer formula so it gets a text_id in document order.
        # needs_segmentation=False: emit as-is, skip the segmenter.
        self.assembler.append(
            inner,
            page_number,
            self._current_section_id,
            needs_segmentation=False,
            is_formula=True,
            provenance=[Provenance(page_no=page_number, bbox=bbox_to_tuple(bbox))],
            region_meta=region_meta,
        )

    def _handle_footnote(self, content: str, page_number: int) -> None:
        """Store a pending footnote for later conversion to section + sentences."""
        self._expire_pending_table_label_fragment()
        text = content.strip()
        if not text:
            return

        # Flush carry-over so deferred text position is accurate
        self._flush_carry_over()

        # Record (text, page, body_section_id, deferred_text_index).
        # The deferred_text_index lets us find the nearest preceding sentence
        # after segmentation populates real text_ids.
        self._footnotes.record(
            text=text,
            page_number=page_number,
            body_section_id=self._current_section_id,
            deferred_text_index=len(self.assembler),
        )

    def _emit_sentences(
        self,
        text: str,
        page_number: int,
        provenance: list[Provenance] | None = None,
        region_meta: dict | None = None,
        page_spans: list[tuple[int, int]] | None = None,
    ) -> None:
        """Store text for later batch segmentation via :meth:`apply_segmentation`."""
        self.assembler.append(
            text,
            page_number,
            self._current_section_id,
            needs_segmentation=True,
            is_formula=False,
            provenance=provenance,
            region_meta=region_meta,
            page_spans=page_spans,
        )

    def _detect_urls(self, sent: PaperSentence) -> None:
        """Find plain-text URLs in a sentence and add as PaperURLLink."""
        for match in URL_RE.finditer(sent.text):
            url = match.group(0)
            # Trim trailing prose delimiters the regex over-captures (sentence
            # punctuation, an unbalanced ')'); keeps balanced DOI parens.
            url = clean_extracted_url(url)
            if not url:
                continue
            self.links.append(
                PaperURLLink(
                    url=url,
                    section_id=sent.section_id,
                    paragraph_id=sent.paragraph_id,
                    text_id=sent.text_id,
                    link_text=None,  # plain-text URL, no anchor text
                )
            )

    def _flush_carry_over(self) -> None:
        """Emit any accumulated carry-over text as sentences.

        The carry-over may have been captured before a section change.  In
        that case temporarily restore the originating ``_current_section_id``
        around the emission so the deferred entry records the section the
        text actually belongs to (C6).
        """
        if self._carry_over.text.strip():
            origin_section = self._carry_over.section_id
            if origin_section is not None and origin_section != self._current_section_id:
                saved = self._current_section_id
                try:
                    self._current_section_id = origin_section
                    self._emit_sentences(
                        self._carry_over.text,
                        self._carry_over.page or 1,
                        self._carry_over.provenance,
                        region_meta=self._carry_over.region_meta,
                        page_spans=self._carry_over.page_spans,
                    )
                finally:
                    self._current_section_id = saved
            else:
                self._emit_sentences(
                    self._carry_over.text,
                    self._carry_over.page or 1,
                    self._carry_over.provenance,
                    region_meta=self._carry_over.region_meta,
                    page_spans=self._carry_over.page_spans,
                )
        self._carry_over.reset()

    @staticmethod
    def _has_terminal_punct(text: str) -> bool:
        """Return True if text ends with sentence-terminal punctuation."""
        stripped = text.rstrip()
        if not stripped:
            return True
        return stripped[-1] in ".!?:;)]"

    @staticmethod
    def _should_join(prev_text: str, next_text: str) -> bool:
        """Heuristic: join two text blocks when prev lacks terminal punct.

        Both lowercase continuations (hyphenation) and capitalized
        continuations (cross-column breaks where the next column starts a
        new clause within the same sentence) are valid joins. The caller
        The caller separately applies stricter guards to cross-page joins.
        """
        if not prev_text or not next_text:
            return False
        return not TextHandlersMixin._has_terminal_punct(prev_text)

    @staticmethod
    def _starts_with_lowercase(text: str) -> bool:
        """Return whether the first alphabetic character is lowercase.

        This is deliberately stricter than same-page carry-over joining. It
        repairs clear sentence continuations across page boundaries while
        avoiding accidental joins to a new heading or capitalized paragraph.
        """
        return next((char.islower() for char in text if char.isalpha()), False)

    @staticmethod
    def _is_url_wrap_join(prev_text: str, next_text: str) -> bool:
        """True when the region boundary falls inside a URL at a wrap hyphen.

        Mirrors the line-wrap bridge in ``consolidate_text`` (keep-hyphen
        policy): the previous block's last token is URL/DOI context and the
        hyphen sits at its end ("…/Lak-") or at the start of the next block
        ("-ens/…", APA breaks URLs before punctuation).  A hyphen between
        digits is always literal (bare ORCID iDs, year/page ranges).  Such
        joins must not insert the space joiner — it would truncate the URL
        at detection.
        """
        prev = prev_text.rstrip()
        nxt = next_text.lstrip()
        if not prev:
            return False
        hyphen_at_prev_end = prev.endswith("-")
        hyphen_at_next_start = nxt.startswith("-") and len(nxt) > 1 and not nxt[1].isspace()
        if not (hyphen_at_prev_end or hyphen_at_next_start):
            return False
        # Hyphen between digits across the boundary is always literal.
        if hyphen_at_prev_end and prev[-2:-1].isdigit() and nxt[:1].isdigit():
            return True
        if hyphen_at_next_start and prev[-1:].isdigit() and nxt[1:2].isdigit():
            return True
        token_start = max(prev.rfind(" "), prev.rfind("\n"), prev.rfind("\t")) + 1
        return bool(DOI_URL_CONTEXT_RE.search(prev[token_start:]))

    @staticmethod
    def _is_bbox_nearby(
        bbox_a: list | None,
        page_a: int | None,
        bbox_b: list | None,
        page_b: int | None,
        max_gap: int = 200,
    ) -> bool:
        """Check if two regions are close enough for caption–element matching.

        Uses ``bbox_2d`` coordinates (normalised 0–1000) from glmocr regions.
        Returns ``True`` when spatial info is missing so the parser falls back
        to sequential order (preserving legacy behaviour).
        """
        if page_a is None or page_b is None:
            return True  # missing page info → trust sequential order
        if page_a != page_b:
            return False  # different pages → not nearby
        if bbox_a is None or bbox_b is None:
            return True  # no bbox info → trust sequential order

        # Vertical gap between the two bboxes (bbox format: [x1, y1, x2, y2])
        a_top, a_bottom = bbox_a[1], bbox_a[3]
        b_top, b_bottom = bbox_b[1], bbox_b[3]

        if a_bottom <= b_top:
            vertical_gap = b_top - a_bottom
        elif b_bottom <= a_top:
            vertical_gap = a_top - b_bottom
        else:
            vertical_gap = 0  # overlapping vertically

        return bool(vertical_gap <= max_gap)

    @staticmethod
    def _merge_bboxes(a: list | None, b: list | None) -> list | None:
        """Return the bounding box that encloses both *a* and *b*."""
        if a is None:
            return b
        if b is None:
            return a
        return [
            min(a[0], b[0]),
            min(a[1], b[1]),
            max(a[2], b[2]),
            max(a[3], b[3]),
        ]

    @staticmethod
    def _is_copyright_notice(text: str) -> bool:
        """Return True if text looks like a copyright or permission notice.

        Publisher copyright blurbs sometimes appear as doc_title regions
        before the actual paper title on page 1.  This heuristic catches
        them so they don't get selected as the paper title.
        """
        lower = text.lower()
        # A real title is unlikely to contain these phrases
        _COPYRIGHT_KEYWORDS = (
            "permission to reproduce",
            "copyright",
            "all rights reserved",
            "licensed under",
            "creative commons",
            "hereby grants",
            "open access",
            "attribution is provided",
            "©",  # copyright symbol
        )
        return any(kw in lower for kw in _COPYRIGHT_KEYWORDS)

    _PUBLISHER_NOISE = frozenset(
        {
            "sage",
            "s sage",
            "sage publications",
            "elsevier",
            "springer",
            "wiley",
            "taylor & francis",
            "frontiers",
        }
    )

    @classmethod
    def _is_publisher_noise(cls, text: str) -> bool:
        """Return True if heading text is a known publisher running-head fragment."""
        return text.lower().strip(" .:") in cls._PUBLISHER_NOISE

    def _make_sentence(
        self,
        entry: DeferredText,
        text: str,
        text_id: int,
        paragraph_id: int,
    ) -> PaperSentence:
        """Build a PDF sentence, attaching source provenance + region_meta.

        ``is_display_formula`` reflects the entry flag — always ``False`` for
        segmentable body text, so it is safe to apply uniformly.
        """
        return PaperSentence(
            text_id=text_id,
            text=text,
            section_id=entry.section_id,
            paragraph_id=paragraph_id,
            page_number=entry.page_number,
            is_display_formula=entry.is_formula,
            provenance=list(entry.provenance),
            region_meta=entry.region_meta,
        )

    def _find_nearest_text_id(
        self, deferred_idx: int, skip_text_ids: AbstractSet[int] = frozenset()
    ) -> int:
        """Find the text_id of the last sentence before a given deferred text position.

        text_ids in *skip_text_ids* (display formulas) are passed over in
        favor of the nearest earlier sentence; the nearest skipped one is
        still returned when nothing else precedes the position — closer
        than the document-start fallback.
        """
        skipped_fallback: int | None = None
        for i in range(min(max(0, deferred_idx - 1), len(self._deferred_last_text_id) - 1), -1, -1):
            tid = self._deferred_last_text_id[i]
            if tid is None:
                continue
            if tid in skip_text_ids:
                if skipped_fallback is None:
                    skipped_fallback = tid
                continue
            return tid
        if skipped_fallback is not None:
            return skipped_fallback
        # Fallback: first sentence or 1
        fallback = self.sentences[0].text_id if self.sentences else 1
        logger.warning(
            "No preceding sentence found for footnote at deferred_idx=%d, using text_id=%d",
            deferred_idx,
            fallback,
        )
        return fallback
