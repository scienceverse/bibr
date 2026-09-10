"""Equation extraction from paper sentences.

Extracts decomposed statistical expressions (e.g. ``t(28) = 3.42, p < .001``)
and LaTeX equations (e.g. ``$\\alpha = 0.05$``) from sentence text into
structured ``PaperEquation`` components with ``lhs/df/comp/rhs`` fields
(degrees of freedom shown parenthetically on the LHS, e.g. ``t(28)``, are
split into a separate ``df`` field: ``lhs="t", df="28"``).

Four-pass strategy:
1. **Parenthesized statistical groups** — find ``(...)`` containing statistical
   patterns, split on commas/semicolons, decompose each into lhs/comp/rhs.
2. **Bare statistical expressions** — scan the sentence for stat expressions
   outside parenthesized groups (e.g. comma-separated ``t(df)=…, p=…, d=…``
   after a closing paren).
3. **LaTeX-delimited equations** — parse ``$...$`` / ``$$...$$`` content for
   ``lhs = rhs`` structure.
4. **Broad equation detection** — a permissive regex (improved by Lisa
   DeBruine) catches remaining stat-like expressions (e.g. ``Cohen's d``,
   ``BF10``, scientific notation) that the structured passes missed.

Optional LLM fallback for sentences in methods/results sections that contain
parenthesized numeric groups but where regex extraction found nothing.
"""

import asyncio
import logging
import re

from bibr.exceptions import ProcessingError
from bibr.paper_contents import PaperEquation, PaperSection, PaperSentence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Comparison operators (order matters: longer patterns first)
# ---------------------------------------------------------------------------

_COMP_PATTERN = r"(?:≤|≥|≈|≠|≪|≫|<=|>=|<<|>>|<|>|=|~)"
_COMP_RE = re.compile(_COMP_PATTERN)

# ---------------------------------------------------------------------------
# Statistical LHS patterns
# ---------------------------------------------------------------------------

# Test statistics with parenthesized df: t(df), F(df1, df2), χ²(df), X²(df, N=n)
_STAT_WITH_DF = re.compile(r"(?:F|t|χ²|χ2|X²|X2)\s*\(\s*[\d.,\s=Nn]+\s*\)")

# Simple named statistics (no parenthesized args)
_STAT_SIMPLE_NAMES = (
    r"(?:"
    r"95\s*%?\s*CI|CI"  # confidence intervals (95% CI before CI)
    r"|R²|R2|adj\.\s*R²|adj\.\s*R2"  # R-squared variants
    r"|η[²p]?|η2p?|ηp²"  # eta-squared variants
    r"|ω²|ω2"  # omega-squared
    r"|OR|RR|HR"  # odds/risk/hazard ratios
    r"|AIC|BIC|VIF"  # model fit
    r"|Mdn|SD|SE"  # descriptives (before single-letter)
    r"|[MNndfzWUkrpdβBb]"  # single-letter stats
    r")"
)
_STAT_SIMPLE_RE = re.compile(_STAT_SIMPLE_NAMES)

# "chi2(1, N = 100)" is a statistic's own df argument, not a parenthesized stat
# group of its own. Treating it as one emitted a bare "N = 100" and recorded
# the span, which then vetoed the correct full match in both later passes.
_DF_ARGUMENT_INNER_RE = re.compile(r"^[\d.,\s=Nn]+$")
_DF_ARGUMENT_OWNER_RE = re.compile(r"(?:F|t|χ²|χ2|X²|X2)\s*$")

# Full LHS pattern: either stat-with-df or simple stat name
_LHS_PATTERN = r"(?:" + _STAT_WITH_DF.pattern + r"|" + _STAT_SIMPLE_NAMES + r")"

# ---------------------------------------------------------------------------
# RHS patterns: signed decimals, bracket ranges, negative numbers
# ---------------------------------------------------------------------------

_RHS_PATTERN = (
    r"(?:"
    r"\[[\d.,\s−–-]+\]"  # bracket range [a, b]
    # Grouped digits first: without it "1,204" matched only "1", so a sample
    # size was exported three orders of magnitude too small.
    r"|[−–-]?\s*(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r")"
)

# ---------------------------------------------------------------------------
# Complete component pattern: lhs comp rhs
# ---------------------------------------------------------------------------

_COMPONENT_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(" + _LHS_PATTERN + r")"  # group 1: lhs
    r"(?![A-Za-z0-9_])"
    r"\s*"
    r"(" + _COMP_PATTERN + r")"  # group 2: comp
    r"\s*"
    r"(" + _RHS_PATTERN + r")",  # group 3: rhs
    re.UNICODE,
)

# ---------------------------------------------------------------------------
# LaTeX equation detection
# ---------------------------------------------------------------------------

_LATEX_DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_LATEX_INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

# ---------------------------------------------------------------------------
# Broad equation detection — regex improved by Lisa DeBruine
# ---------------------------------------------------------------------------
# Catches a wider range of stat-like expressions (e.g. Cohen's d, BF10,
# scientific notation) that the structured passes above may miss.

_BROAD_OP_CHARS = "=<>~\u2248\u2260\u2264\u2265\u226a\u226b"
# Group 2 of _BROAD_EQUATION_RE is one or two of these, so a sentence without
# any cannot match — a necessary condition, not a heuristic. Measured on 86
# real exports (29,521 sentences) 5.3% carry one, and the broad pass is the
# most expensive of the four: ~25 ms/paper of the shared serve event loop.
_BROAD_OP_PRESCAN_RE = re.compile("[" + re.escape(_BROAD_OP_CHARS) + "]")

# LaTeX command in an LHS means the broad pass picked up a LaTeX fragment
# that should be handled by the LaTeX pass (pass 3) instead.
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]")

# "Trivial" LaTeX: bare superscript/subscript of digits and commas (citation
# markers like ^{6} or ^{18,28}), not real mathematical content.
_TRIVIAL_LATEX_RE = re.compile(r"^[\^_]\{[\d,\s]+\}$")

_NUMERIC_ONLY_RE = re.compile(r"^-?\d+$")
_SMALL_INTEGER_RE = re.compile(r"^-?\d$")

_BROAD_EQUATION_RE = re.compile(
    r"("  # group 1: lhs
    r"(?:(?:Cohen['\u2019]s|\d{1,2}%)\s+)?"  #   optional prefix
    r"[\u03B1-\u03C9a-zA-Z\-_\.0-9\{\}\^\\²]+"  #   statistic name
    r"(?:\s*\([^)]*\))?"  #   optional parenthesized args
    r")"
    r"\s*"
    r"([" + _BROAD_OP_CHARS + r"]{1,2})"  # group 2: comp
    r"\s*"
    r"("  # group 3: rhs
    r"[0-9.,+\-\u2212\u2013]*[0-9]"  #   number
    r"(?:\s*e\s*-\s*\d+)?"  #   scientific notation
    r"(?:\s*[x\*]\s*10\s*\^\s*-?\s*\d+)?"  #   power-of-10
    r"|\[[^\]]+\]"  #   or bracketed range
    r")",
    re.UNICODE,
)

# The broad pass exists for real statistic names that the structured patterns
# do not enumerate. Letting any ASCII word through turns prose such as
# ``dark green >60%`` into an equation. Keep the useful broad cases while
# requiring an LHS that actually resembles mathematical/statistical notation.
_PLAUSIBLE_BROAD_LHS_RE = re.compile(
    r"^(?:"
    r"Cohen['\u2019]s\s+d"
    r"|\d{1,2}%\s+CI"
    r"|[A-Z]{2,8}\d*"
    r"|(?:alpha|beta|gamma|delta|epsilon|theta|lambda|omega)"
    r"|[\u03B1-\u03C9]"
    r"|[A-Za-z]"
    r")(?:\s*\([^)]*\))?$"
)


def _llm_rhs_is_grounded(rhs: str | None, source_text: str) -> bool:
    """Return whether an LLM equation RHS occurs in its source sentence.

    The fallback may normalize whitespace, Unicode minus signs, or a leading
    zero (``0.05`` ↔ ``.05``), but it must never invent a numeric value.
    """
    if not rhs or not rhs.strip():
        return False

    def _compact(value: str) -> str:
        return re.sub(r"\s+", "", value.translate(str.maketrans({"−": "-", "–": "-"})))

    compact_rhs = _compact(rhs)
    compact_source = _compact(source_text)
    variants = {compact_rhs}
    if compact_rhs.startswith("."):
        variants.add("0" + compact_rhs)
    elif compact_rhs.startswith("-."):
        variants.add("-0" + compact_rhs[1:])
    elif compact_rhs.startswith("0."):
        variants.add(compact_rhs[1:])
    elif compact_rhs.startswith("-0."):
        variants.add("-" + compact_rhs[2:])
    return any(variant in compact_source for variant in variants)


# ---------------------------------------------------------------------------
# Equation Extractor
# ---------------------------------------------------------------------------


class EquationExtractor:
    """Extracts decomposed equations from paper sentences.

    Usage::

        extractor = EquationExtractor()
        equations = extractor.extract_from_sentences(sentences, sections)
    """

    def __init__(self) -> None:
        self._grp_counter = 0

    def _next_grp_id(self) -> int:
        self._grp_counter += 1
        return self._grp_counter

    def extract_from_sentences(
        self,
        sentences: list[PaperSentence],
        sections: list[PaperSection],  # noqa: ARG002
    ) -> list[PaperEquation]:
        """Extract equations from sentences using regex only (synchronous).

        Parameters
        ----------
        sentences : list[PaperSentence]
            All sentences from the paper.
        sections : list[PaperSection]
            Section metadata (kept for API consistency with LLM fallback).

        Returns
        -------
        list[PaperEquation]
            Extracted equation components.
        """
        self._grp_counter = 0
        equations: list[PaperEquation] = []

        for sent in sentences:
            # Pass 1: parenthesized statistical groups
            stat_eqs, paren_spans = self._extract_stat_groups(sent)
            equations.extend(stat_eqs)
            # Pass 2: bare statistical expressions outside parenthesized groups
            bare_eqs = self._extract_bare_stats(sent, paren_spans)
            equations.extend(bare_eqs)
            # Pass 3: LaTeX-delimited equations
            latex_eqs = self._extract_latex_equations(sent)
            equations.extend(latex_eqs)
            # Pass 4: broad equation detection for missed cases
            existing_keys = {
                (eq.text_id, eq.lhs, eq.comp, eq.rhs) for eq in stat_eqs + bare_eqs + latex_eqs
            }
            equations.extend(self._extract_broad_equations(sent, existing_keys, paren_spans))

        logger.info(
            "Regex equation extraction: %d components from %d sentences",
            len(equations),
            len(sentences),
        )
        return equations

    async def extract_with_llm_fallback(
        self,
        sentences: list[PaperSentence],
        sections: list[PaperSection],
        llm_client=None,
        min_regex_stats: int = 0,
        regex_equations: list[PaperEquation] | None = None,
    ) -> list[PaperEquation]:
        """Extract equations with optional LLM fallback for missed cases.

        First runs regex extraction, then identifies sentences in
        methods/results sections that have parenthesized numeric content
        but where regex found nothing, and sends those to the LLM.

        Parameters
        ----------
        sentences : list[PaperSentence]
            All sentences from the paper.
        sections : list[PaperSection]
            Section metadata.
        llm_client : LLMClient | None
            Optional LLM client for fallback extraction.
        regex_equations : list[PaperEquation] | None
            Results of an already-completed regex pass. Callers that bound
            this coroutine with a timeout run step 1 themselves so a slow LLM
            fan-out cannot discard results that are already in hand.
        min_regex_stats : int
            Opt-in cost gate (0 = disabled). When > 0, the LLM fallback runs
            only if the regex pass already found at least this many non-LaTeX
            statistical components — a paper-level proxy for "this paper
            reports statistics". Skips the fallback on papers (e.g. math/CS
            preprints) unlikely to carry the prose stats the fallback targets.

        Returns
        -------
        list[PaperEquation]
            Combined regex + LLM equation components.
        """
        from bibr.paper_contents import CanonicalSection

        # Step 1: regex extraction
        equations = (
            self.extract_from_sentences(sentences, sections)
            if regex_equations is None
            else regex_equations
        )

        if llm_client is None:
            return equations

        # Optional paper-level stats-density gate. A non-LaTeX regex component
        # (no backslash in lhs/rhs) is a t/F/p/d-style statistic; a paper with
        # too few is unlikely to reward the fallback's per-sentence LLM cost.
        if min_regex_stats > 0:
            n_stat = sum(
                1 for eq in equations if "\\" not in (eq.lhs or "") and "\\" not in (eq.rhs or "")
            )
            if n_stat < min_regex_stats:
                return equations

        # Step 2: identify candidates for LLM fallback
        extracted_text_ids = {eq.text_id for eq in equations}

        # Only consider sentences in methods/results sections
        target_section_types = {
            CanonicalSection.METHODS,
            CanonicalSection.RESULTS,
        }
        target_section_ids = {
            s.section_id for s in sections if s.section_type in target_section_types
        }

        # Detect parenthesized groups with numbers but no regex hits
        _paren_with_nums = re.compile(r"\([^)]*\d[^)]*\)")
        candidates: list[tuple[int, str]] = []
        for sent in sentences:
            if sent.text_id in extracted_text_ids:
                continue
            if sent.section_id not in target_section_ids:
                continue
            # Display formulas already carry their full LaTeX verbatim in the
            # exported ``formatted`` field; sending them to the LLM only
            # re-decomposes content the parser already captured losslessly.
            if sent.is_display_formula:
                continue
            if _paren_with_nums.search(sent.text):
                candidates.append((sent.text_id, sent.text))

        if not candidates:
            return equations

        # Step 3: LLM extraction in batches
        logger.info("LLM equation fallback: %d candidate sentences", len(candidates))
        batch_size = 10
        existing_keys = {(eq.text_id, eq.lhs, eq.comp, eq.rhs) for eq in equations}
        batches = [candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)]

        async def _extract_batch(index, batch):
            try:
                return await llm_client.extract_equations(batch, file_hash="equations")
            except ProcessingError:
                raise
            except Exception as e:
                logger.warning(
                    "LLM equation extraction failed for batch %d/%d (%d candidates): %s",
                    index + 1,
                    len(batches),
                    len(batch),
                    e,
                )
                return []

        batch_results = await asyncio.gather(
            *(_extract_batch(index, batch) for index, batch in enumerate(batches))
        )
        for batch, llm_equations in zip(batches, batch_results, strict=True):
            source_by_text_id = dict(batch)
            # Group components by sentence: the LLM emits flat components,
            # and a fallback sentence is one statistical statement in
            # practice. A unique grp_id per component would isolate exactly
            # the t/p/d trios Metacheck needs grouped (M7).
            grp_by_text_id: dict[int, int] = {}
            for eq in llm_equations:
                # Drop components where lhs, comp AND rhs are all
                # empty/whitespace — the LLM occasionally emits blank
                # placeholders that add nothing but noise.
                if (
                    not (eq.lhs or "").strip()
                    and not (eq.comp or "").strip()
                    and not (eq.rhs or "").strip()
                ):
                    continue
                source_text = source_by_text_id.get(eq.text_id)
                if source_text is None or not _llm_rhs_is_grounded(eq.rhs, source_text):
                    logger.warning(
                        "Dropping ungrounded LLM equation component "
                        "(text_id=%s, lhs=%r, comp=%r, rhs=%r)",
                        eq.text_id,
                        eq.lhs,
                        eq.comp,
                        eq.rhs,
                    )
                    continue
                # Dedupe on (text_id, lhs, comp, rhs), mirroring the regex
                # passes' existing_keys approach — both within this LLM
                # batch and against everything already extracted.
                key = (eq.text_id, eq.lhs, eq.comp, eq.rhs)
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                if eq.text_id not in grp_by_text_id:
                    grp_by_text_id[eq.text_id] = self._next_grp_id()
                eq.grp_id = grp_by_text_id[eq.text_id]
                equations.append(eq)

        return equations

    def _extract_stat_groups(
        self, sent: PaperSentence
    ) -> tuple[list[PaperEquation], list[tuple[int, int]]]:
        """Extract statistical equations from parenthesized groups in a sentence.

        Returns
        -------
        tuple[list[PaperEquation], list[tuple[int, int]]]
            (equations, extracted_spans) where extracted_spans are the
            (start, end) character ranges of parenthesized groups that
            produced equations, used to avoid double-extraction in the
            bare-stats pass.
        """
        results: list[PaperEquation] = []
        extracted_spans: list[tuple[int, int]] = []
        text = sent.text

        for inner, paren_start, paren_end in _iter_parenthesized_groups(text):
            # Validate: must contain a comparison operator and a digit
            if not _COMP_RE.search(inner) or not re.search(r"\d", inner):
                continue

            # Unwrapped APA form: "chi2(1, N = 100) = 3.84". The df parenthesis
            # is part of the statistic, not a group in its own right.
            if _DF_ARGUMENT_INNER_RE.match(inner) and _DF_ARGUMENT_OWNER_RE.search(
                text[:paren_start]
            ):
                continue

            # Split on commas/semicolons that are NOT inside () or []
            parts = _split_respecting_brackets(inner)

            group_equations: list[PaperEquation] = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                comp_match = _COMPONENT_RE.search(part)
                if comp_match:
                    lhs = comp_match.group(1).strip()
                    comp = comp_match.group(2).strip()
                    rhs = comp_match.group(3).strip()

                    # Normalize comparison operators
                    comp = _normalize_comp(comp)
                    # Split parenthesized degrees of freedom off the LHS:
                    # "t(28)" -> lhs="t", df="28".
                    lhs, df = _split_lhs_df(lhs)

                    group_equations.append(
                        PaperEquation(
                            text_id=sent.text_id,
                            grp_id=0,  # assigned below
                            lhs=lhs,
                            df=df,
                            comp=comp,
                            rhs=rhs,
                        )
                    )

            if group_equations:
                grp_id = self._next_grp_id()
                for eq in group_equations:
                    eq.grp_id = grp_id
                results.extend(group_equations)
                extracted_spans.append((paren_start, paren_end))

        return results, extracted_spans

    def _extract_bare_stats(
        self,
        sent: PaperSentence,
        extracted_spans: list[tuple[int, int]],
    ) -> list[PaperEquation]:
        """Extract stat expressions not inside already-extracted parenthesized groups.

        Finds comma-separated stat patterns like ``t(97.7)=2.9, p=0.005, d=0.59``
        that appear at the sentence level (outside wrapper parentheses).
        Consecutive matches separated only by commas/semicolons/whitespace share
        the same ``grp_id``.
        """
        results: list[PaperEquation] = []
        text = sent.text

        matches = list(_COMPONENT_RE.finditer(text))

        # Keep only matches whose spans do NOT overlap with extracted paren groups
        bare_matches = []
        for m in matches:
            overlaps = any(m.start() < end and m.end() > start for start, end in extracted_spans)
            if not overlaps:
                bare_matches.append(m)

        if not bare_matches:
            return results

        # Group consecutive matches separated only by commas/semicolons/whitespace
        groups: list[list[re.Match]] = []
        current_group: list[re.Match] = []

        for m in bare_matches:
            if current_group:
                between = text[current_group[-1].end() : m.start()]
                if re.fullmatch(r"[\s,;]+", between):
                    current_group.append(m)
                else:
                    groups.append(current_group)
                    current_group = [m]
            else:
                current_group.append(m)

        if current_group:
            groups.append(current_group)

        for group in groups:
            grp_id = self._next_grp_id()
            for m in group:
                lhs = m.group(1).strip()
                comp = _normalize_comp(m.group(2).strip())
                rhs = m.group(3).strip()
                if _NUMERIC_ONLY_RE.match(lhs):
                    continue
                if len(lhs) <= 2 and _SMALL_INTEGER_RE.match(rhs):
                    continue
                # Split df after the reject filters so a stat like "t(28)"
                # (len > 2) is not collapsed to "t" before the length check.
                lhs, df = _split_lhs_df(lhs)
                results.append(
                    PaperEquation(
                        text_id=sent.text_id,
                        grp_id=grp_id,
                        lhs=lhs,
                        df=df,
                        comp=comp,
                        rhs=rhs,
                    )
                )

        return results

    def _extract_latex_equations(self, sent: PaperSentence) -> list[PaperEquation]:
        """Extract equations from LaTeX-delimited content in a sentence."""
        results: list[PaperEquation] = []
        text = sent.text

        # Collect all LaTeX spans (display math first, then inline)
        latex_spans: list[tuple[str, str]] = []  # (content, format)
        seen_ranges: list[tuple[int, int]] = []

        if sent.is_display_formula:
            # The entire sentence is a display formula (no $$ delimiters).
            latex_spans.append((text, "display"))
        else:
            for m in _LATEX_DISPLAY_RE.finditer(text):
                latex_spans.append((m.group(1), "display"))
                seen_ranges.append((m.start(), m.end()))

            for m in _LATEX_INLINE_RE.finditer(text):
                # Skip if overlaps with display math
                overlaps = any(not (m.end() <= s or m.start() >= e) for s, e in seen_ranges)
                if not overlaps:
                    latex_spans.append((m.group(1), "inline"))

        for content, fmt in latex_spans:
            # Look for lhs = rhs pattern in LaTeX content.
            # Use top-level search to skip operators inside _{...} / ^{...}
            # subscripts/superscripts (e.g. \sum_{i=1} should not match).
            comp_match = _find_toplevel_comp(content)
            if comp_match:
                lhs = content[: comp_match.start()].strip()
                comp = _normalize_comp(comp_match.group())
                rhs = content[comp_match.end() :].strip()

                # Strip LaTeX commands for cleaner output but keep the substance
                if lhs and rhs:
                    results.append(
                        PaperEquation(
                            text_id=sent.text_id,
                            grp_id=self._next_grp_id(),
                            lhs=lhs,
                            comp=comp,
                            rhs=rhs,
                        )
                    )
            elif fmt == "display":
                # Pure display formula with no comparison operator (e.g. $$\Delta_{i}$$).
                # Still record it so all LaTeX formulas appear in the equations array.
                # Inline math without an operator is typically just notation, not an equation.
                stripped_content = content.strip()
                if stripped_content and not _TRIVIAL_LATEX_RE.match(stripped_content):
                    results.append(
                        PaperEquation(
                            text_id=sent.text_id,
                            grp_id=self._next_grp_id(),
                            lhs=stripped_content,
                            comp="",
                            rhs="",
                        )
                    )

        return results

    def _extract_broad_equations(
        self,
        sent: PaperSentence,
        existing_keys: set[tuple[int, str, str, str]],
        paren_spans: list[tuple[int, int]],
    ) -> list[PaperEquation]:
        """Broad regex pass to catch equations missed by the structured passes.

        Uses the permissive ``_BROAD_EQUATION_RE`` pattern.  Matches that
        overlap with already-extracted parenthesized groups or that duplicate
        an existing ``(text_id, lhs, comp, rhs)`` key are skipped.
        Consecutive matches separated only by commas/semicolons/whitespace
        share the same ``grp_id``.
        """
        results: list[PaperEquation] = []
        text = sent.text
        if not _BROAD_OP_PRESCAN_RE.search(text):
            return results

        kept: list[tuple[re.Match, str, str, str, str]] = []
        for m in _BROAD_EQUATION_RE.finditer(text):
            # Skip if overlapping with an already-extracted parenthesized group
            if any(m.start() < end and m.end() > start for start, end in paren_spans):
                continue

            raw_lhs = m.group(1).strip()
            comp = _normalize_comp(m.group(2).strip())
            rhs = m.group(3).strip()
            # Split parenthesized df off the LHS ("t(28)" -> "t", "28") so the
            # dedup key matches the structured passes, which also split.
            lhs, df = _split_lhs_df(raw_lhs)

            if (sent.text_id, lhs, comp, rhs) in existing_keys:
                continue

            # Skip if LHS contains a LaTeX command — these are LaTeX fragments
            # that should be handled by the LaTeX pass, not classified as stat.
            if _LATEX_CMD_RE.search(raw_lhs):
                continue

            if not _PLAUSIBLE_BROAD_LHS_RE.fullmatch(raw_lhs):
                continue

            # Skip trivial equations: purely numeric LHS (6 = 24), or
            # short LHS with small bare integer RHS (i = 1, b = 5).
            # Keeps real stats: d = 0.45, N = 120, p < .001.
            if _NUMERIC_ONLY_RE.match(raw_lhs):
                continue
            if len(raw_lhs) <= 2 and _SMALL_INTEGER_RE.match(rhs):
                continue

            kept.append((m, lhs, df, comp, rhs))

        if not kept:
            return results

        # Group consecutive matches separated only by commas/semicolons/whitespace
        groups: list[list[tuple[re.Match, str, str, str, str]]] = []
        current_group: list[tuple[re.Match, str, str, str, str]] = []

        for item in kept:
            m = item[0]
            if current_group:
                between = text[current_group[-1][0].end() : m.start()]
                if re.fullmatch(r"[\s,;]+", between):
                    current_group.append(item)
                else:
                    groups.append(current_group)
                    current_group = [item]
            else:
                current_group.append(item)

        if current_group:
            groups.append(current_group)

        for group in groups:
            grp_id = self._next_grp_id()
            for _, lhs, df, comp, rhs in group:
                results.append(
                    PaperEquation(
                        text_id=sent.text_id,
                        grp_id=grp_id,
                        lhs=lhs,
                        df=df,
                        comp=comp,
                        rhs=rhs,
                    )
                )

        return results


def _find_toplevel_comp(content: str) -> re.Match | None:
    """Find a comparison operator in LaTeX content that is not inside braces.

    Skips operators inside ``_{...}`` and ``^{...}`` subscript/superscript
    groups as well as any nested ``{...}`` braces.  This prevents
    ``\\sum_{i=1}`` from being split into ``lhs={i, comp==, rhs=1``.

    Returns the first :class:`re.Match` at brace depth 0, or ``None``.
    """
    depth = 0
    i = 0
    while i < len(content):
        ch = content[i]
        if ch == "{":
            depth += 1
            i += 1
            continue
        if ch == "}":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0:
            m = _COMP_RE.match(content, i)
            if m:
                return m
        i += 1
    return None


def _split_respecting_brackets(text: str) -> list[str]:
    """Split on commas/semicolons that are not inside parentheses or brackets."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for idx, ch in enumerate(text):
        if ch in "([":
            depth += 1
            current.append(ch)
        elif ch in ")]":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch in ",;" and depth == 0:
            # A comma between digits is a thousands separator, not a
            # component boundary: splitting "N = 1,204" gave "N = 1".
            if (
                ch == ","
                and idx > 0
                and text[idx - 1].isdigit()
                and idx + 1 < len(text)
                and text[idx + 1].isdigit()
            ):
                current.append(ch)
                continue
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _normalize_comp(comp: str) -> str:
    """Normalize comparison operator to Unicode form."""
    return {
        "<=": "≤",
        ">=": "≥",
        "<<": "≪",
        ">>": "≫",
    }.get(comp, comp)


# Statistic LHS with a trailing parenthesized degrees-of-freedom suffix:
# "t(28)", "F(2, 47)", "χ²(4, N=200)".  Captures (name, df).
_LHS_DF_RE = re.compile(r"^(?P<name>.+?)\s*\(\s*(?P<df>[^()]*?)\s*\)\s*$")


def _split_lhs_df(lhs: str) -> tuple[str, str]:
    """Split a statistic LHS into ``(name, degrees_of_freedom)``.

    ``"t(28)"`` -> ``("t", "28")``;  ``"F(2, 47)"`` -> ``("F", "2, 47")``.
    Returns ``(lhs, "")`` when there is no parenthesized df suffix on the LHS.
    """
    m = _LHS_DF_RE.match(lhs)
    if m and m.group("name").strip():
        return m.group("name").strip(), m.group("df").strip()
    return lhs, ""


def _iter_parenthesized_groups(text: str):
    """Yield balanced parenthesized groups as ``(inner, start, end)`` tuples.

    *inner*: text between the parentheses (not including them).
    *start*: index of the opening ``(``.
    *end*: index after the closing ``)``.

    This intentionally avoids complex nested-parentheses regex patterns,
    which can exhibit severe backtracking on malformed OCR text.
    """
    stack: list[int] = []
    for idx, char in enumerate(text):
        if char == "(":
            stack.append(idx)
            continue
        if char != ")":
            continue
        if not stack:
            continue

        start = stack.pop()
        # Only emit outermost completed groups to avoid duplicate extraction
        # from nested pieces like (t(28) = 2.1, p = .04).
        if not stack:
            inner = text[start + 1 : idx]
            if inner:
                yield inner, start, idx + 1
