"""Heading / section-hint region handlers for :class:`PDFParser`.

Extracted from :mod:`bibr.structure.pdf_parser` as :class:`HeadingHandlersMixin`.
Methods move verbatim (only ``text_repair`` helper calls are spelled with their
public names); shared state (``self.sections``, counters, caption/hint trackers)
lives on :class:`PDFParser` and is reached through ``self``.
"""

import logging
import re
from enum import StrEnum

from bibr.input.consolidate_text import strip_affiliation_markers
from bibr.paper_contents import CanonicalSection, PaperSection, Provenance
from bibr.structure.section_tree import STUDY_MARKER_RE, infer_level_from_numbering
from bibr.structure.text_repair import (
    bbox_to_tuple,
    collapse_numbered_prefix_spaces,
    repair_heading_artifacts,
    strip_markdown_emphasis,
)
from bibr.utils.text import normalize_text

logger = logging.getLogger(__name__)

_CONTENT_HEADING_MAX_CHARS = 80
_CONTENT_HEADING_MAX_WORDS = 8
# A body row that opens with a list-bullet glyph — never a heading, even bold.
_BULLET_ROW_RE = re.compile(r"^[-•*‣·]\s")
_BOLD_HEADING_MAX_WORDS = 6

# Heading label → level (H1 or H2)
_HEADING_LEVELS: dict[str, int] = {
    "doc_title": 1,
    "paragraph_title": 2,
}

# Section hints that create implicit sections with known canonical names
_SECTION_HINT_NAMES: dict[str, str] = {
    "abstract": "Abstract",
    "reference": "References",
    "reference_content": "References",
}

# A reference-list lead-in that OCR occasionally promotes to a heading, e.g.
# "Breiman [2001]:" or "Smith (1999)." — a capitalised surname immediately
# followed by a bracketed/parenthesised 4-digit year and a colon/period.
_REF_LEADIN_RE = re.compile(r"^[A-Z][\w'’\-]+\s*(?:\[\d{4}\]|\(\d{4}\))\s*[:.]")

# OCR badge-glyph artifacts that the layout model occasionally merges into the
# trailing edge of a page-1 doc_title region (e.g. the Open-Practices "TC"
# transparency badge in Psychological Science). An explicit allowlist keeps the
# strip from clipping legitimate trailing acronyms / Roman numerals.
_TITLE_BADGE_GLYPH_RE = re.compile(r"\s+(?:TC)\s*$")
_LOOSE_FIGURE_CAPTION_RE = re.compile(r"^(?:Figure|Fig\.?)\s+\d+\b", re.IGNORECASE)

# Prose that merely *opens* with a float reference. The loose caption
# discriminators require no separator, because a caption-labelled region may
# legitimately read "Table 1 Overview" — but applied to a ``paragraph_title``
# they also swallowed "Figure 3 illustrates the interaction between dose and
# latency", turning a body sentence into a caption candidate that usually
# found no owner and lost its text.
#
# The discriminator is what follows the number: a caption continues with its
# own capitalised noun phrase ("Overview of measures", "Mean latency"), prose
# continues with a lowercase finite verb ("illustrates", "shows", "reports").
# "Table 2 continued" is the one lowercase continuation that IS a caption, so
# it is excluded explicitly.
# Deliberately case-SENSITIVE: the lowercase continuation is the whole signal,
# so the label alternation spells its own case variants instead of using
# re.IGNORECASE.
_FLOAT_PROSE_CONTINUATION_RE = re.compile(
    r"^(?:[Tt]able|[Ff]igure|[Ff]ig\.?)\s+"
    r"(?:\d+(?:\.\d+)*|[IVXLCDM]+)\s+"
    r"(?!contin(?:ued|uation)\b)[a-z]"
)


class HeadingDisposition(StrEnum):
    """Terminal treatment for one OCR heading region."""

    SECTION_HEADING = "section_heading"
    TABLE_CAPTION = "table_caption"
    FIGURE_CAPTION = "figure_caption"
    BODY_TEXT = "body_text"
    DROP_PUBLISHER_NOISE = "drop_publisher_noise"


class HeadingHandlersMixin:
    """Heading, section-hint, and outline-hierarchy handlers for ``PDFParser``."""

    def _handle_structural(self, label: str, content: str) -> None:
        """Record headers/footers as metadata."""
        self._expire_pending_table_label_fragment()
        text = content.strip()
        if not text:
            return
        if label == "header":
            self.detected_headers.append(text)
        elif label == "footer":
            self.detected_footers.append(text)

    def _handle_heading(
        self,
        label: str,
        content: str,
        page_number: int,
        bbox: list | None = None,
    ) -> None:
        """Create a new PaperSection from a heading region."""
        self._expire_pending_table_label_fragment()

        # Strip markdown heading prefix (e.g. "## Methods" → "Methods")
        # glmocr ResultFormatter adds "# " for doc_title and "## " for paragraph_title
        text = re.sub(r"^#{1,6}\s*", "", content.strip()).strip()
        # Normalize Markdown-wrapped headings emitted by OCR (e.g.
        # "**References**" → "References") before hint-section deduplication.
        text = strip_markdown_emphasis(text)
        text = repair_heading_artifacts(text)
        if not text:
            return

        # Collapse stray spaces in numbered prefixes from GLM-OCR (e.g.
        # "3. 1 Encoder" → "3.1 Encoder"). Without this the heading both
        # displays oddly AND fails infer_level_from_numbering, which expects
        # tightly-spelled "N.M" for level inference.
        text = collapse_numbered_prefix_spaces(text)

        # Strip LaTeX affiliation markers (e.g. "Title $ ^{1} $" → "Title")
        text = strip_affiliation_markers(text)

        disposition = self._classify_heading_disposition(label, text)
        if disposition == HeadingDisposition.DROP_PUBLISHER_NOISE:
            logger.info("Dropping heading %r as publisher noise (page %d)", text, page_number)
            return
        if disposition == HeadingDisposition.TABLE_CAPTION:
            self._handle_table_caption(text, bbox, page_number)
            return
        if disposition == HeadingDisposition.FIGURE_CAPTION:
            self._handle_figure_caption(text, bbox, page_number)
            return
        if disposition == HeadingDisposition.BODY_TEXT:
            logger.debug("Heading gate: demoting implausible heading to content: %r", text[:80])
            self._emit_content_without_promotion(text, page_number, bbox)
            return

        # Only a real section boundary invalidates stale caption tracking.
        self._caption_barriers.append(self._source_region_index)
        self._terminal_reference_tail_section_id = None

        # Capture doc_title on page 1 as the detected paper title.
        # Skip text that looks like a copyright/permission/license notice
        # (these sometimes precede the actual title on page 1).
        if (
            label == "doc_title"
            and self._is_front_page(page_number)
            and self._detected_title is None
            and not self._is_copyright_notice(text)
        ):
            # OCR sometimes merges an Open-Practices badge glyph (rendered " TC")
            # from the icon row beneath the title into the doc_title region.
            # Strip it here so it pollutes neither _detected_title nor the
            # section header below (both derive from this one value). Scoped to
            # page-1 doc_title and an explicit allowlist so legitimate trailing
            # acronyms / Roman numerals (MRI, II, PTSD, DNA) are preserved.
            text = _TITLE_BADGE_GLYPH_RE.sub("", text)
            self._detected_title = text
            logger.debug("Detected doc_title on page 1: %s", self._detected_title[:80])

        # If a section hint already created a section with the same name,
        # reuse it instead of creating a duplicate (e.g., "reference" hint
        # creates "References" section, then a "References" heading appears).
        text_lower = text.lower().strip()
        hint_section_id = self._hint_section_lookup.get(text_lower)
        if hint_section_id is not None:
            self._current_section_id = hint_section_id
            for section in self.sections:
                if section.section_id == hint_section_id:
                    section.header_is_synthetic = False
                    # Earlier prose already belongs to this hinted section.
                    # A later heading does not relocate that span's boundary.
                    break
            logger.debug(
                "Heading '%s' matches existing hint section (id=%d), reusing",
                text,
                hint_section_id,
            )
            return

        # Default level from layout label; adjust paragraph_title level via
        # numbering patterns so multi-level nesting is preserved.  Numbering
        # depth maps to level as depth+1 (doc_title occupies level 1):
        #   "3 Methods"       → depth 1 → level 2
        #   "3.1 Encoder"     → depth 2 → level 3
        #   "3.2.1 Subsection"→ depth 3 → level 4
        level = _HEADING_LEVELS.get(label, 2)
        if label == "paragraph_title":
            inferred = infer_level_from_numbering(text)
            if inferred is not None:
                level = inferred + 1
        self._section_counter += 1

        # Determine parent section (closest section with lower level).
        # Skip sections created by section hints (Abstract, References) —
        # they are standalone and should not become parents of later headings.
        parent_id = 0
        for sec in reversed(self.sections):
            if sec.section_id in self._hint_section_ids:
                continue
            if sec.level < level and sec.level > 0:
                parent_id = sec.section_id
                break

        bbox_tuple = bbox_to_tuple(bbox)
        section = PaperSection(
            section_id=self._section_counter,
            header=text,
            level=level,
            parent_section_id=parent_id,
            provenance=[Provenance(page_no=page_number, bbox=bbox_tuple)],
        )
        self.sections.append(section)
        self._current_section_id = self._section_counter

    def _classify_heading_disposition(self, label: str, text: str) -> HeadingDisposition:
        """Classify a heading region exactly once into its terminal treatment."""
        if self._is_publisher_noise(text):
            return HeadingDisposition.DROP_PUBLISHER_NOISE
        if label != "paragraph_title":
            return HeadingDisposition.SECTION_HEADING
        # A ``paragraph_title`` is not a caption label. The loose
        # discriminators below exist for figure_title/chart_title regions,
        # where a missing separator is normal; here they stole ordinary prose.
        # BODY_TEXT keeps the sentence, which caption ownership would not.
        if _FLOAT_PROSE_CONTINUATION_RE.match(text):
            return HeadingDisposition.BODY_TEXT
        if self._LOOSE_TABLE_CAPTION_RE.match(text):
            return HeadingDisposition.TABLE_CAPTION
        if _LOOSE_FIGURE_CAPTION_RE.match(text):
            return HeadingDisposition.FIGURE_CAPTION
        if self._is_implausible_heading(text):
            return HeadingDisposition.BODY_TEXT
        return HeadingDisposition.SECTION_HEADING

    def _is_implausible_heading(self, text: str) -> bool:
        """Return True when a ``paragraph_title`` candidate should be demoted to
        content instead of opening a section.

        Conservative — each clause targets a distinct, verified false-positive
        class. The caller routes every rejection to ``_handle_content`` so the
        text is never dropped.
        """
        stripped = text.strip()
        if not stripped:
            return False

        # (a) Table/figure captions the layout model mislabelled as a heading.
        if (
            self._TABLE_CAPTION_RE.match(stripped)
            or self._FIGURE_CAPTION_RE.match(stripped)
            or self._LOOSE_TABLE_CAPTION_RE.match(stripped)
        ):
            return True

        # (b) Reference-list lead-in ("Breiman [2001]:").
        if _REF_LEADIN_RE.match(stripped):
            return True

        # (c) Alphabetic-char ratio < 0.4 — kills numeric OCR garbage
        # ("111115555557799991111") while real headers stay well above.
        non_space = sum(1 for c in stripped if not c.isspace())
        if non_space:
            alpha = sum(1 for c in stripped if c.isalpha())
            if alpha / non_space < 0.4:
                return True

        words = stripped.split()

        # (d) 1-3-word period-terminated fragment that is NOT a canonical alias
        # and has no numbering prefix — kills "Best case." / "Data access."
        # while keeping "Conclusion." (a DISCUSSION alias).
        if (
            stripped.endswith(".")
            and 1 <= len(words) <= 3
            and infer_level_from_numbering(stripped) is None
        ):
            from bibr.structure.section_classifier import _classify_lookup

            section, _score = _classify_lookup(normalize_text(stripped))
            if section == CanonicalSection.UNKNOWN:
                return True

        # (e) Long sentence-like candidate: >12 words ending in sentence
        # punctuation is prose, not a heading.
        return len(words) > 12 and stripped[-1] in ".!?"

    def _handle_section_hint(
        self,
        label: str,
        content: str,
        page_number: int,
        bbox: list | None = None,
        region_meta: dict | None = None,
    ) -> None:
        """Handle regions like 'abstract' or 'reference' that imply a section."""
        self._expire_pending_table_label_fragment()
        hint_name = _SECTION_HINT_NAMES.get(label, "")

        if hint_name == "References":
            text = content.strip()
            if self._start_publisher_note_back_matter(text, page_number, bbox):
                self._emit_content_without_promotion(
                    text,
                    page_number,
                    bbox,
                    region_meta=region_meta,
                )
                return
            if self._terminal_reference_tail_section_id is not None:
                self._current_section_id = self._terminal_reference_tail_section_id
                if text:
                    self._emit_content_without_promotion(
                        text,
                        page_number,
                        bbox,
                        region_meta=region_meta,
                    )
                return

        # Explicit Endnotes/Footnotes headings own their rows; layout hints must not create an
        # early References section that captures the later printed heading.
        if (
            hint_name == "References"
            and self._current_section_id not in self._hint_section_ids
            and self._current_section_id != 0
        ):
            current = next(
                (
                    section
                    for section in reversed(self.sections)
                    if section.section_id == self._current_section_id
                ),
                None,
            )
            if current is not None:
                from bibr.structure.section_classifier import _classify_lookup

                current_type, _score = _classify_lookup(normalize_text(current.header))
                if current_type == CanonicalSection.FOOTNOTE:
                    text = content.strip()
                    if text:
                        self._flush_carry_over()
                        self.assembler.append(
                            text,
                            page_number,
                            self._current_section_id,
                            needs_segmentation=False,
                            is_formula=False,
                            provenance=[Provenance(page_no=page_number, bbox=bbox_to_tuple(bbox))],
                            region_meta=region_meta,
                        )
                    return

        # Create an implicit section if we haven't already for this label group;
        # otherwise restore _current_section_id so subsequent regions stay in
        # the correct section (e.g. references spanning multiple pages).
        if hint_name and hint_name in self._created_hint_sections:
            stored_id = self._hint_section_lookup.get(hint_name.lower())
            if stored_id is not None:
                self._current_section_id = stored_id
        elif hint_name:
            # Check if a heading-created section with the same name already
            # exists (heading came before the hint region).  If so, reuse it
            # instead of creating a duplicate.
            hint_lower = hint_name.lower()
            existing = None
            for sec in self.sections:
                if sec.header.lower() == hint_lower and sec.level > 0:
                    existing = sec
                    break

            if existing is not None:
                self._current_section_id = existing.section_id
                self._hint_section_ids.add(existing.section_id)
                logger.debug(
                    "Section hint '%s' matches existing heading section (id=%d), reusing",
                    hint_name,
                    existing.section_id,
                )
            else:
                self._section_counter += 1
                section = PaperSection(
                    section_id=self._section_counter,
                    header=hint_name,
                    level=1,
                    parent_section_id=0,
                    header_is_synthetic=True,
                )
                self.sections.append(section)
                self._current_section_id = self._section_counter
                self._hint_section_ids.add(self._section_counter)

            self._created_hint_sections.add(hint_name)
            self._hint_section_lookup[hint_name.lower()] = self._current_section_id

        # Process the text content itself
        text = content.strip()
        if text:
            if hint_name == "References":
                # Reference text: emit directly without sentence segmentation.
                # The extractor's reference segmenter handles splitting.
                self._flush_carry_over()
                # Defer so it gets a text_id in document order.
                # needs_segmentation=False: emit as-is, skip the segmenter.
                self.assembler.append(
                    text,
                    page_number,
                    self._current_section_id,
                    needs_segmentation=False,
                    is_formula=False,
                    provenance=[Provenance(page_no=page_number, bbox=bbox_to_tuple(bbox))],
                    region_meta=region_meta,
                )
            else:
                self._handle_content(text, page_number, bbox, region_meta=region_meta)

    def _promotable_content_heading(self, text: str, region_meta: dict | None = None) -> str | None:
        """Return repaired heading text when a body row is a trusted section.

        Native PDF text can occasionally preserve heading glyphs as ordinary
        ``text`` regions. Promotion is deliberately narrow: short single-line
        rows that are study markers, trusted canonical aliases, or — when the
        region's font metadata says the whole row is bold — arbitrary heading
        text (the rescue for novel or OCR-corrupted headings no alias covers).
        """
        raw = text.strip()
        if not raw:
            return None
        if raw[-1] in ".!?":
            return None
        if "[" in raw or "]" in raw:
            return None

        candidate = repair_heading_artifacts(text)
        if not candidate or "\n" in candidate:
            return None
        if len(candidate) > _CONTENT_HEADING_MAX_CHARS:
            return None

        stripped = candidate.strip()
        words = stripped.split()
        if not words or len(words) > _CONTENT_HEADING_MAX_WORDS:
            return None
        non_space = sum(1 for c in stripped if not c.isspace())
        if non_space:
            alpha = sum(1 for c in stripped if c.isalpha())
            if alpha / non_space < 0.4:
                return None

        if STUDY_MARKER_RE.match(stripped):
            return stripped

        from bibr.structure.section_classifier import _classify_lookup_full

        section, _score, trusted = _classify_lookup_full(normalize_text(stripped))
        if section != CanonicalSection.UNKNOWN and trusted:
            return stripped

        # Font-signal rescue. Tighter word cap than the alias paths — without
        # an alias match, longer bold rows are more often sentence fragments
        # than headings. ``_is_implausible_heading`` vetoes anything the
        # plausibility gate would bounce straight back to content (captions,
        # reference lead-ins) — promotion and gate must never disagree, or a
        # promoted heading would oscillate between the two handlers.
        if (
            region_meta is not None
            and region_meta.get("font_bold")
            and len(stripped) >= 3
            and len(words) <= _BOLD_HEADING_MAX_WORDS
            and _BULLET_ROW_RE.match(stripped) is None
            and not self._is_implausible_heading(stripped)
        ):
            return stripped
        return None

    def _apply_outline_hierarchy(self) -> None:
        """Override heading levels from the PDF outline (bookmarks) when enabled.

        The outline is the document's own declared section tree — the most
        authoritative hierarchy signal. Bookmark titles are fuzzily matched
        (title + page) to the detected headings; a confidently matched heading
        takes the bookmark's compressed 1-based level, overriding the
        numbering-derived level from :meth:`_handle_heading`. Matched sections
        are flagged ``outline_level_authoritative`` so the later
        ``assign_hierarchy_from_top_level`` pass leaves their level/parent
        intact (the same protection numbered headings already enjoy). Unmatched
        headings keep their existing level/parent.

        Inert unless an outline was threaded in AND
        ``Settings.pipeline.outline_headings`` is on.
        """
        if not self._outline:
            return
        if not self._settings.pipeline.outline_headings:
            return

        from bibr.input.pdf_outline import HeadingRef, match_outline_to_headings

        # Candidate headings: real sections only — skip Root and standalone
        # hint sections (Abstract/References), which never carry outline levels
        # and must not become parents.
        candidates = [
            s for s in self.sections if s.level > 0 and s.section_id not in self._hint_section_ids
        ]
        if not candidates:
            return

        def _heading_ref(sec: PaperSection) -> HeadingRef:
            page_no: int | None = None
            y_top: float | None = None
            if sec.provenance:
                prov = sec.provenance[0]
                page_no = prov.page_no
                if prov.bbox is not None:
                    # bbox y1 is glmocr-normalized 0..1000, top-origin; scale to
                    # the 0..1 page-height fraction the matcher expects.
                    y_top = prov.bbox[1] / 1000.0
            return HeadingRef(text=sec.header, page_no=page_no, y_top=y_top)

        refs = [_heading_ref(s) for s in candidates]
        matched = match_outline_to_headings(self._outline, refs)
        if not matched:
            return

        for idx, level in matched.items():
            sec = candidates[idx]
            sec.level = level
            sec.outline_level_authoritative = True

        # Recompute parents for matched sections against the NEW levels, using
        # the same nearest-preceding-lower-level rule as _handle_heading. Other
        # sections keep their parents; the later hierarchy pass owns those.
        matched_ids = {candidates[idx].section_id for idx in matched}
        for i, sec in enumerate(self.sections):
            if sec.section_id not in matched_ids:
                continue
            parent_id = 0
            for prev in reversed(self.sections[:i]):
                if prev.section_id in self._hint_section_ids:
                    continue
                if 0 < prev.level < sec.level:
                    parent_id = prev.section_id
                    break
            sec.parent_section_id = parent_id
