"""Locating metadata and reference rows in the parsed sentence stream.

``RefLocator`` owns everything about WHERE the paper's front matter and
reference list live in ``sentences_df`` — canonical-section mapping, the
metadata cutoff, reference-section discovery (header-alias override →
canonical map → layout-hint → header-text fallbacks), and boundary-orphan
reclaim. It never calls an LLM and never parses reference contents; those
belong to ``bibr.extract.core_metadata`` and ``bibr.extract.ref_extractor``.
"""

import logging
import re
from dataclasses import dataclass

import pandas as pd

from bibr.config import GlobalSettings, snapshot_settings
from bibr.extract.author_email_harvester import _CORRESPONDING_MARKER_RE
from bibr.ocr.ref_patterns import _REF_HEADER_RE, _looks_like_author_date_start
from bibr.paper_contents import CanonicalSection, PaperContents
from bibr.structure.reference_boundaries import TERMINAL_REFERENCE_BOUNDARY_RE
from bibr.utils.text import YEARISH_RE

logger = logging.getLogger(__name__)

# Page-continuation decoration is not part of an article's reference heading.
# The heading itself uses the same multilingual grammar as OCR geometry.
_CONTINUED_SUFFIX_RE = re.compile(
    r"\s*(?:\(continued\)|[-–—:]\s*continued|\s+continued)\s*$",
    re.IGNORECASE,
)


def _is_reference_heading(header: str) -> bool:
    return bool(_REF_HEADER_RE.fullmatch(_CONTINUED_SUFFIX_RE.sub("", header).strip()))


@dataclass(frozen=True)
class _SectionRun:
    start: int
    stop: int
    header: str
    section_id: int | None


# Recognize parenthesized, dash-delimited, whitespace-separated, and period-delimited reference
# numbers, including non-Latin author text.
_ENTRY_NUMBERING_RE = re.compile(
    r"^\s*(?:"
    r"\[\d{1,3}\]\s*"
    r"|\(\d{1,3}\)\s*"
    r"|\d{1,3}\s*[-–—]\s+"
    r"|\d{1,3}[.)](?:\s+|(?=[^\d\s]))"
    r"|\d{1,3}\s+(?=[A-ZÀ-ÖØ-ÞЀ-ЯҊ-Ҿ])"
    r")"
)


def _printed_entry_number(text: str) -> int | None:
    """Return the printed list number of a reference row, when present."""
    match = _ENTRY_NUMBERING_RE.match(text)
    if match is None:
        return None
    digits = re.search(r"\d{1,3}", match.group(0))
    return int(digits.group(0)) if digits else None


# Boundary-only evidence that a row opens an unnumbered author/byline entry.
# Deliberately narrower than ``_looks_like_complete_entry_start``: journal and
# container continuations can be capitalized and carry a later online year.
_AUTHOR_BYLINE_ONSET_RE = re.compile(
    r"^\s*[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’`-]+"
    r"(?:,\s*(?:[A-Z]\.?\s*){1,4}|(?:\s+[A-Z]\.?){1,4})"
    r"(?:\s*[,;&]|\s+et al\b|\s*\()"
)

# A bare ORCID printed inline in a front-matter footnote (no orcid.org URL),
# e.g. "… University of Zagreb 0000-0002-5438-0665". De Gruyter prints ORCIDs
# this way, so the orcid.org text mask in collect_core_metadata_rows misses
# them; this inline form lets the page-1 footnote rescue pull those rows in.
_ORCID_BARE_INLINE_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")


def _looks_like_complete_entry_start(text: str) -> bool:
    s = _ENTRY_NUMBERING_RE.sub("", text.lstrip(), count=1)
    return bool(s) and s[0].isupper() and bool(YEARISH_RE.search(text))


def _looks_like_terminal_reference_start(text: str) -> bool:
    """High-precision onset evidence used only for terminal spill trimming."""
    if _ENTRY_NUMBERING_RE.match(text):
        return True
    return bool(
        (_AUTHOR_BYLINE_ONSET_RE.match(text) and YEARISH_RE.search(text))
        or _looks_like_author_date_start(text)
    )


class RefLocator:
    """Finds the metadata/reference row ranges in a paper's sentence stream."""

    # Priority order to find the "split" point in the paper
    CUTOFF_PRIORITY = (
        CanonicalSection.INTRODUCTION,
        CanonicalSection.METHODS,
        CanonicalSection.RESULTS,
        CanonicalSection.DISCUSSION,
    )

    def __init__(
        self,
        contents: PaperContents,
        *,
        settings: GlobalSettings | None = None,
    ):
        self.contents = contents
        self.sentences_df = contents.sentences_df
        self._settings = settings if settings is not None else snapshot_settings()

        # Cache for section mapping: Canonical Enum -> First matching Raw String
        self._canonical_map: dict[CanonicalSection, str] = {}
        self._is_mapped = False

    def map_canonical_sections(self) -> None:
        """
        Maps raw section names to canonical types using pre-classified sections.
        Only runs once per instance.
        """
        if self._is_mapped:
            return

        for section in self.contents.sections:
            if (
                section.section_type
                and section.section_type != CanonicalSection.UNKNOWN
                and section.section_type not in self._canonical_map
            ):
                self._canonical_map[section.section_type] = section.header
                logger.debug(f"Mapped '{section.header}' to {section.section_type}")

        self._is_mapped = True

    def get_cutoff_index(self) -> int:
        """
        Returns the integer index (iloc) where the metadata ends.
        """
        self.map_canonical_sections()

        cap = self._settings.llm.core_cutoff_max_sentences

        has_section_col = "section_name" in self.sentences_df.columns
        for section_type in self.CUTOFF_PRIORITY:
            if has_section_col and section_type in self._canonical_map:
                raw_name = self._canonical_map[section_type]

                # Create a boolean mask for where this section appears
                mask = self.sentences_df["section_name"] == raw_name

                if mask.any():
                    # argmax returns the index of the first True value
                    # This gives us the integer position for iloc
                    import numpy as np

                    cutoff = int(np.asarray(mask.values).argmax())
                    if cap > 0 and cutoff > cap:
                        # A late cutoff means the earlier IMRaD headers were
                        # misclassified: the first-matching type is deep in the
                        # body. Cap the front-matter slice and surface it — this
                        # is the observability signal for early-section
                        # classification failure.
                        logger.warning(
                            "Core-metadata cutoff for %s at iloc %d exceeds cap; "
                            "capping to %d (early section classification likely failed).",
                            section_type.name,
                            cutoff,
                            cap,
                        )
                        return cap
                    return cutoff

        # No IMRaD-style section break — common for short comments,
        # editorials, and brief reports. Use a sentence-count heuristic
        # for the metadata cutoff. INFO-level: this is a normal outcome,
        # not a failure.
        fallback = cap if cap > 0 else 250
        logger.info(
            "No IMRaD section header found; using leading %d sentences for metadata extraction.",
            fallback,
        )
        return fallback

    def collect_core_metadata_rows(
        self,
        cutoff_iloc: int,
        *,
        allowed_text_ids: frozenset[int] | None = None,
    ) -> pd.DataFrame:
        """
        Collects all rows considered part of the metadata.
        This includes all rows before the cutoff, plus any rows in the document
        that contain "ORCID".
        """
        # 1. Select all rows up to the cutoff
        metadata_df = self.sentences_df.iloc[:cutoff_iloc].copy()

        # 2. Find rows with "ORCID" throughout the ENTIRE document.
        # Only include the individual rows containing "orcid.org", not entire
        # sections — expanding to full sections can pull in reference text
        # whose DOIs then get mistakenly used as the paper's own DOI.
        text = self.sentences_df["text"]
        rescue_mask = text.str.contains("orcid.org", na=False)

        # 2b. Front-matter affiliation/ORCID footnotes that the page-flattened
        # stream pushes BELOW the cutoff (De Gruyter house style): page-1 rows
        # carrying a corresponding-author marker or a BARE ORCID (no orcid.org
        # URL). Without this the metadata LLM never sees them and emits null
        # affiliation/ORCID. Scoped to page 1 to avoid dragging body/reference
        # text into the metadata context. DOI selection is unaffected — it reads
        # the separate iloc[:cutoff] slice (see extract_core_metadata), so no
        # reference DOI leaks in here.
        if "page_number" in self.sentences_df.columns:
            page1 = self.sentences_df["page_number"] == 1
            footnote = page1 & (
                text.str.contains(_CORRESPONDING_MARKER_RE, na=False)
                | text.str.contains(_ORCID_BARE_INLINE_RE, na=False)
            )
            rescue_mask = rescue_mask | footnote

        orcid_df = self.sentences_df[rescue_mask].copy()

        # 3. Combine and drop duplicates (by index)
        # Using concat and checking indices is cleaner than manual numpy array manipulation
        combined_df = pd.concat([metadata_df, orcid_df])

        # Remove duplicate indices (rows that are both before cutoff and contain ORCID)
        combined_df = combined_df[~combined_df.index.duplicated(keep="first")]

        # 4. Sort by index to maintain document flow
        combined_df = combined_df.sort_index()

        if allowed_text_ids is not None:
            if "text_id" not in combined_df.columns:
                return combined_df.iloc[0:0]
            combined_df = combined_df[combined_df["text_id"].isin(allowed_text_ids)]

        return combined_df

    def _model_reference_header(self, threshold: float = 0.5) -> str | None:
        """Header of the last section whose heading region the front-role model
        calls ``ref_header`` with probability >= ``threshold``, else None."""
        from bibr.extract.front_role import FrontRolePredictions

        predictions = getattr(self.contents, "front_role_predictions", None)
        if not isinstance(predictions, FrontRolePredictions):
            return None
        summaries = getattr(self.contents, "region_summaries", None) or []
        by_section: dict[int, list] = {}
        for summary in summaries:
            if summary.section_id is not None and summary.label in ("paragraph_title", "doc_title"):
                by_section.setdefault(summary.section_id, []).append(summary)
        for section in reversed(self.contents.sections):
            header = (section.header or "").strip()
            if section.level <= 0 or not header:
                continue
            for summary in by_section.get(section.section_id, ()):
                scores = predictions.get(summary.page, summary.index)
                if scores is not None and scores.get("ref_header") >= threshold:
                    return section.header
        return None

    def collect_reference_rows(self) -> pd.DataFrame:
        """Collects all rows considered part of the reference list.

        A section whose printed header IS a references heading takes
        precedence over the classifier-assigned canonical map: the classifier
        occasionally types a body section as REFERENCES while the literal
        "References" header gets UNKNOWN (exp #2: 09567976241258149), and the
        canonical-map path would then short-circuit on the wrong section.
        Falls back to the canonical map, then to layout-hint and header-text
        heuristics.
        """
        self.contents.reference_boundary_reason_flags = []
        self.map_canonical_sections()

        has_section_col = "section_name" in self.sentences_df.columns

        # Step 0: trust what is printed — a literal references heading wins.
        if has_section_col:
            last_headers = self.sentences_df["section_name"].dropna().drop_duplicates(keep="last")
            for section_name in reversed(last_headers.tolist()):
                if not _is_reference_heading(str(section_name)):
                    continue
                mask = self.sentences_df["section_name"] == section_name
                if not mask.any():
                    continue
                mapped = self._canonical_map.get(CanonicalSection.REFERENCES)
                if mapped is not None and mapped != section_name:
                    logger.info(
                        f"Header-alias override: using section '{section_name}' over "
                        f"canonical-map REFERENCES section '{mapped}'"
                    )
                header_df = self._collect_bibliography_run(section_name)
                self._reclassify_as_references(header_df)
                reclaimed = self._reclaim_boundary_orphans(header_df, section_name)
                return self._trim_terminal_boundary(reclaimed)

        # Step 0b: model evidence can identify reference headings outside the
        # explicit multilingual grammar.
        if has_section_col:
            model_header = self._model_reference_header()
            if model_header is not None:
                mask = self.sentences_df["section_name"] == model_header
                if mask.any():
                    logger.info(
                        f"Front-role model header: using section '{model_header}' "
                        "as the reference section"
                    )
                    self.contents.reference_boundary_reason_flags.append("front_role_ref_header")
                    header_df = self._collect_bibliography_run(model_header)
                    self._reclassify_as_references(header_df)
                    reclaimed = self._reclaim_boundary_orphans(header_df, model_header)
                    return self._trim_terminal_boundary(reclaimed)

        if has_section_col and CanonicalSection.REFERENCES in self._canonical_map:
            ref_section_name = self._canonical_map[CanonicalSection.REFERENCES]

            # Create a boolean mask for where this section appears
            mask = self.sentences_df["section_name"] == ref_section_name

            if mask.any():
                ref_df = self._collect_bibliography_run(ref_section_name)
                self._reclassify_as_references(ref_df)
                ref_df = self._reclaim_boundary_orphans(ref_df, ref_section_name)
                return self._trim_terminal_boundary(ref_df)

        # Fallback 1: layout detection found reference regions but section
        # classification didn't identify a REFERENCES section (e.g., unusual
        # or non-English header that didn't match aliases or zero-shot NLI).
        if self.contents.layout_hints:
            hint_labels = {label for label, _ in self.contents.layout_hints}
            if "reference" in hint_labels or "reference_content" in hint_labels:
                ref_df = self._collect_last_unknown_section_rows()
                self._reclassify_as_references(ref_df)
                return self._trim_terminal_boundary(ref_df)

        # Fallback 2: scan section headers for reference-like text patterns.
        # Catches cases where the section wasn't classified as REFERENCES by
        # the NLI model but has a recognizable header.
        _ref_patterns = ("reference", "bibliography", "works cited", "literature cited")
        if not has_section_col:
            raise ValueError("No section_name column in sentences_df (OCR may have failed)")
        last_headers = self.sentences_df["section_name"].dropna().drop_duplicates(keep="last")
        for section_name in reversed(last_headers.tolist()):
            header_lower = section_name.lower().strip()
            if any(pat in header_lower for pat in _ref_patterns):
                mask = self.sentences_df["section_name"] == section_name
                if mask.any():
                    fallback_df = self._collect_bibliography_run(section_name)
                    logger.info(
                        f"Header-text fallback: using section '{section_name}' "
                        f"as reference section ({len(fallback_df)} rows)"
                    )
                    self._reclassify_as_references(fallback_df)
                    return self._trim_terminal_boundary(fallback_df)

        raise ValueError("No reference section found.")

    def _collect_bibliography_run(self, ref_section_name: str) -> pd.DataFrame:
        """Join adjacent sections belonging to the selected bibliography.

        A translated printed heading and a synthetic ``References`` section
        can split one list. Header equality alone also joins unrelated lists
        elsewhere in a document, so seed the last section identity and expand
        only across supported, adjacent bibliography sections. Deferred OCR
        rows can interrupt that section's text stream; those interruptions do
        not create a new owner or erase its earlier references.

        Subsections need explicit ancestry under an already selected section
        and reference-shaped text. A numbering restart is allowed within that
        ownership; numbering alone never establishes ownership.
        """
        df = self.sentences_df
        runs: list[_SectionRun] = []
        section_ids = df.get("section_id")
        for position, name in enumerate(df["section_name"]):
            header = str(name) if pd.notna(name) else ""
            value = section_ids.iloc[position] if section_ids is not None else None
            section_id = int(value) if value is not None and pd.notna(value) else None
            if runs and (runs[-1].section_id, runs[-1].header) == (section_id, header):
                previous = runs[-1]
                runs[-1] = _SectionRun(previous.start, position + 1, header, section_id)
            else:
                runs.append(_SectionRun(position, position + 1, header, section_id))

        seed = next(i for i in reversed(range(len(runs))) if runs[i].header == ref_section_name)
        selected_ids = {runs[seed].section_id} - {None}
        selected_runs = {
            index
            for index, run in enumerate(runs)
            if index == seed or run.section_id in selected_ids
        }
        if len(selected_runs) > 1:
            self.contents.reference_boundary_reason_flags.append("same_section_disjoint_runs")
        section_map = {section.section_id: section for section in self.contents.sections}

        def supported_neighbor(run: _SectionRun) -> bool:
            if TERMINAL_REFERENCE_BOUNDARY_RE.match(run.header):
                return False
            explicit_heading = _is_reference_heading(run.header)
            section = section_map.get(run.section_id)
            descendant = False
            seen: set[int] = set()
            while section is not None and section.parent_section_id not in seen:
                parent = section.parent_section_id
                if parent in selected_ids:
                    descendant = True
                    break
                if parent is None:
                    break
                seen.add(parent)
                section = section_map.get(parent)
            section = section_map.get(run.section_id)
            if not explicit_heading and not (
                descendant
                and section is not None
                and section.section_type in (CanonicalSection.UNKNOWN, CanonicalSection.REFERENCES)
            ):
                return False
            if explicit_heading and _CONTINUED_SUFFIX_RE.search(run.header):
                # A final continuation page may hold only a citation's tail,
                # with its author/year on the preceding page.
                return True
            texts = df.iloc[run.start : run.stop]["text"].astype(str)
            return any(
                _looks_like_terminal_reference_start(text) and YEARISH_RE.search(text)
                for text in texts
            )

        def adjacent(before: _SectionRun, after: _SectionRun) -> bool:
            # Do not step across a terminal row even if its section label was
            # never corrected by the parser.
            if any(
                TERMINAL_REFERENCE_BOUNDARY_RE.match(str(text))
                for text in df.iloc[before.start : before.stop]["text"]
            ) or TERMINAL_REFERENCE_BOUNDARY_RE.match(str(df.iloc[after.start]["text"])):
                return False
            if "page_number" in df.columns:
                end_page = df.iloc[before.stop - 1]["page_number"]
                start_page = df.iloc[after.start]["page_number"]
                if pd.notna(end_page) and pd.notna(start_page):
                    return 0 <= start_page - end_page <= 1
            return True

        expanded = False
        pending = list(selected_runs)
        while pending:
            current = pending.pop()
            for neighbor in (current - 1, current + 1):
                if not 0 <= neighbor < len(runs) or neighbor in selected_runs:
                    continue
                before, after = sorted((current, neighbor))
                if not adjacent(runs[before], runs[after]) or not supported_neighbor(
                    runs[neighbor]
                ):
                    continue
                expanded = True
                section_id = runs[neighbor].section_id
                if section_id is not None:
                    selected_ids.add(section_id)
                # A section ID identifies an owner, even when deferred rows
                # from another section interrupt its physical text stream.
                # Never include those intervening rows by taking a broad slice.
                added = {
                    index
                    for index, run in enumerate(runs)
                    if index == neighbor
                    or (section_id is not None and run.section_id == section_id)
                } - selected_runs
                selected_runs.update(added)
                pending.extend(added)
        if expanded:
            self.contents.reference_boundary_reason_flags.append("adjacent_bibliography_sections")
        positions = [
            position
            for index in sorted(selected_runs)
            for position in range(runs[index].start, runs[index].stop)
        ]
        return df.iloc[positions].copy()

    def _trim_terminal_boundary(self, ref_df: pd.DataFrame) -> pd.DataFrame:
        """Trim only a strong terminal transition after genuine ref rows.

        The cut is row-initial and heading-specific. It never searches for
        keywords inside citation text and requires at least two preceding rows,
        so a leading publisher note cannot erase the entire candidate section.
        """
        if ref_df.empty or "text" not in ref_df.columns:
            return ref_df
        credible_starts = 0
        for position, text in enumerate(ref_df["text"].astype(str)):
            if _looks_like_terminal_reference_start(text):
                credible_starts += 1
            if position < 2 or _ENTRY_NUMBERING_RE.match(text):
                continue
            if not TERMINAL_REFERENCE_BOUNDARY_RE.match(text):
                continue
            if credible_starts < 2:
                continue
            logger.info("Trimmed terminal reference spill at row %d: %s", position, text[:80])
            flags = getattr(self.contents, "reference_boundary_reason_flags", None)
            if not isinstance(flags, list):
                flags = []
                self.contents.reference_boundary_reason_flags = flags
            if "terminal_boundary_trimmed" not in flags:
                flags.append("terminal_boundary_trimmed")
            return ref_df.iloc[:position].copy()
        return ref_df

    def _reclaim_boundary_orphans(
        self, ref_df: pd.DataFrame, ref_section_name: str
    ) -> pd.DataFrame:
        """Include reference fragments orphaned at page boundaries.

        When the start of the first reference entry is attributed to the
        preceding section (the OCR region arrived before the reference section
        marker), the head must be reclaimed from the last non-reference
        sentence on the same page.

        Gated on the first reference row NOT already being a complete entry
        start: reclaiming unconditionally pulled whole acknowledgment
        sentences, numbered footnotes, and the paper's own byline into
        ref_text, where the segmenter emitted them as phantom bib stubs
        (exp #2 / B″ re-gate judged failures).
        """
        if ref_df.empty or "page_number" not in self.sentences_df.columns:
            return ref_df

        if _looks_like_complete_entry_start(str(ref_df.iloc[0]["text"])):
            return ref_df

        first_ref_page = ref_df.iloc[0]["page_number"]
        first_ref_text_id = ref_df.iloc[0]["text_id"]

        preceding = self.sentences_df[
            (self.sentences_df["text_id"] < first_ref_text_id)
            & (self.sentences_df["page_number"] == first_ref_page)
            & (self.sentences_df["section_name"] != ref_section_name)
        ]
        if preceding.empty:
            return ref_df

        # Take only the last row — the one immediately preceding references.
        orphan = preceding.iloc[[-1]]
        logger.info(
            "Reclaimed orphaned reference fragment: %s",
            orphan.iloc[0]["text"][:80],
        )
        return pd.concat([orphan, ref_df]).sort_values("text_id")

    def _reclassify_as_references(self, ref_df: pd.DataFrame) -> None:
        """Promote sections covered by *ref_df* to REFERENCES in contents.sections.

        This ensures downstream helpers like ``_map_bib_text_ids`` can locate
        reference sentences even when the section was found via a fallback path.
        """
        if "section_id" not in ref_df.columns:
            return
        fallback_ids = set(ref_df["section_id"].dropna().unique())
        for section in self.contents.sections:
            if section.section_id in fallback_ids:
                section.section_type = CanonicalSection.REFERENCES

    def _collect_last_unknown_section_rows(self) -> pd.DataFrame:
        """Collect rows from the last UNKNOWN-typed section as a reference fallback.

        When layout detection confirms references exist but no section was classified
        as REFERENCES, the last unclassified section is the most likely candidate
        (references are almost always the last major section in a paper).

        Returns:
            DataFrame of rows from the last UNKNOWN section.

        Raises:
            ValueError: If no suitable fallback section is found.
        """
        # Build a header → section_type lookup from pre-classified PaperSection objects
        header_to_type: dict[str, CanonicalSection] = {}
        for section in self.contents.sections:
            if section.header and section.header not in header_to_type:
                header_to_type[section.header] = section.section_type or CanonicalSection.UNKNOWN

        if "section_name" not in self.sentences_df.columns:
            raise ValueError("No section_name column in sentences_df")
        sections = self.sentences_df["section_name"].dropna().unique()

        for section_name in reversed(sections):
            classified_type = header_to_type.get(section_name, CanonicalSection.UNKNOWN)
            if classified_type == CanonicalSection.UNKNOWN:
                mask = self.sentences_df["section_name"] == section_name
                ref_df: pd.DataFrame = self.sentences_df[mask].copy()
                logger.info(
                    f"Layout-hint fallback: using last UNKNOWN section "
                    f"'{section_name}' as reference section ({len(ref_df)} rows)"
                )
                return ref_df

        raise ValueError(
            "No reference section found (layout hints indicate references "
            "but no UNKNOWN sections available for fallback)."
        )
