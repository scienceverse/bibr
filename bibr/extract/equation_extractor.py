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
   DeBruine) catches remaining stat-like expressions (e.g. ``BF10``, ``ICC``,
   ``P``) that the structured passes missed.

A pass skips any match overlapping a component an earlier pass emitted, so
each printed expression is exported once. Values are captured whole, as
printed: scientific notation (``2.3 × 10−5``, ``1e-10``), decimal commas
(``0,05``) and ranges (``.85–.94``).

Optional LLM fallback for sentences in methods/results sections that contain
parenthesized numeric groups but where regex extraction found nothing.
"""

import asyncio
import logging
import re

from bibr.exceptions import ProcessingError
from bibr.paper_contents import PaperEquation, PaperSection, PaperSentence

logger = logging.getLogger(__name__)

# (start, end) character range in a sentence's text
Span = tuple[int, int]

# ---------------------------------------------------------------------------
# Comparison operators (order matters: longer patterns first)
# ---------------------------------------------------------------------------

_COMP_PATTERN = r"(?:≤|≥|≈|≠|≪|≫|<=|>=|<<|>>|<|>|=|~)"

# LaTeX relations, as OCR prints them inside $...$ before late clean-up, and
# the operator _normalize_comp maps each to.
_LATEX_COMPS = {
    r"\le": "≤",
    r"\leq": "≤",
    r"\ge": "≥",
    r"\geq": "≥",
    r"\ne": "≠",
    r"\neq": "≠",
    r"\approx": "≈",
    r"\sim": "~",
    r"\ll": "≪",
    r"\gg": "≫",
    r"\lt": "<",
    r"\gt": ">",
}
_LATEX_COMP_PATTERN = (
    r"(?:"
    + "|".join(re.escape(cmd) for cmd in sorted(_LATEX_COMPS, key=len, reverse=True))
    + r")(?![A-Za-z])"
)
_ANY_COMP_PATTERN = r"(?:" + _LATEX_COMP_PATTERN + r"|" + _COMP_PATTERN + r")"
_COMP_RE = re.compile(_ANY_COMP_PATTERN)

# ---------------------------------------------------------------------------
# Statistical LHS patterns
# ---------------------------------------------------------------------------

# Characters that continue a statistic's name, so no name starts right after
# one: ASCII word characters, Greek letters, and superscript/subscript digits
# and letters. "η²p" is partial eta squared, not a p-value, and "ΔR²" is not
# R². Unicode \w would also stop a name printed flush against CJK text.
_NAME_CHARS = r"A-Za-z0-9_\u0370-\u03FF\u00B2\u00B3\u00B9\u2070-\u209F"

# Test statistics with parenthesized df: t(df), F(df1, df2), χ²(df),
# X²(df, N=n), r(df) (APA correlation), H(df) (Kruskal-Wallis)
_DF_STAT_NAMES = r"(?:F|t|r|H|χ²|χ2|X²|X2)"
_STAT_WITH_DF = re.compile(_DF_STAT_NAMES + r"\s*\(\s*[\d.,\s=Nn]+\s*\)")

# Simple named statistics (no parenthesized args)
_STAT_SIMPLE_NAMES = (
    r"(?:"
    # Confidence intervals, with their level ("90% CI", "99.9% CI") before CI
    r"\d{1,2}(?:\.\d+)?\s*%\s*CI|95\s*%?\s*CI|CI"
    r"|R²|R2|adj\.\s*R²|adj\.\s*R2"  # R-squared variants
    # Eta and omega squared, partial or generalized, however the subscript is
    # printed: "η²p", "ηp²", "ηp2" and "η2 p" (JATS and PDF text layers
    # flatten the super- and subscript), "ηₚ²", "ηG²".
    r"|η(?:[²2]\s?[pPₚG]|[pPₚG]\s?[²2]|[²2pₚ])?"
    r"|ω(?:[²2]\s?[pPₚG]|[pPₚG]\s?[²2]|[²2])"
    r"|χ²|χ2"  # chi-square printed without its df
    r"|OR|RR|HR"  # odds/risk/hazard ratios
    r"|AIC|BIC|VIF"  # model fit
    r"|Mdn|SD|SE"  # descriptives (before single-letter)
    r"|Cohen['\u2019]s\s+d"  # before its bare "d"
    r"|[MNndfzZWUkrpdgβBb]"  # single-letter stats
    r")"
)
_STAT_SIMPLE_RE = re.compile(_STAT_SIMPLE_NAMES)

# "chi2(1, N = 100)" is a statistic's own df argument, not a parenthesized stat
# group of its own. Treating it as one emitted a bare "N = 100" and recorded
# the span, which then vetoed the correct full match in both later passes.
# The owner must be a whole name: the "r" ending "number (N = 100)" is not.
_DF_ARGUMENT_INNER_RE = re.compile(r"^[\d.,\s=Nn]+$")
_DF_ARGUMENT_OWNER_RE = re.compile(
    r"(?<![" + _NAME_CHARS + r"])(?:Δ\s?)?" + _DF_STAT_NAMES + r"\s*$"
)

# Full LHS pattern: either stat-with-df or simple stat name, optionally a
# change in it ("ΔR²", "Δχ²(1)", "ΔAIC")
_LHS_PATTERN = r"(?:Δ\s?)?(?:" + _STAT_WITH_DF.pattern + r"|" + _STAT_SIMPLE_NAMES + r")"

# ---------------------------------------------------------------------------
# RHS patterns: signed decimals, bracket ranges, negative numbers
# ---------------------------------------------------------------------------

# A power of ten written with its exponent: "10^-5", OCR LaTeX "10^{-5}",
# "10**-5", or Unicode superscripts "10⁻⁵". Appended to a literal "10".
_POWER = (
    r"(?:\s*(?:\^|\*\*)\s*"
    r"(?:\{\s*[+−–-]?\s*\d+\s*\}|\(\s*[+−–-]?\s*\d+\s*\)|[+−–-]?\s*\d+)"
    r"|[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
)

# Exponent of a number in scientific notation, in every spelling the
# extractor sees (sentence text before late clean-up): "2.3e-5", "1E−06",
# "3.2 e -5", "2.3 × 10⁻⁵", "2.1 x 10^-4", OCR LaTeX "2.3 \times 10^{-5}",
# and "2.3 × 10−5" or "2.5 × 103", where a JATS or PDF text layer flattened
# the superscript. Without it "p = 2.3 × 10−5" exported as "p = 2.3".
_EXPONENT = (
    r"(?:[eE][+−–-]?\d+"
    r"|\s*[eE]\s*[+−–-]\s*\d+"
    r"|\s*(?:[×x*·⋅]|\\times|\\cdot)\s*10(?:" + _POWER + r"|\s*[+−–-]\s*\d+|\d+))"
)

_NUMBER = (
    # Grouped digits first: without it "1,204" matched only "1", so a sample
    # size was exported three orders of magnitude too small.
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)"
    # Decimal comma ("0,05", "3,45", "0,0001"): a comma between digits that
    # does not group thousands separates the fraction, and cutting there made
    # "p = 0,05" "p = 0". Not when a comma continues the run ("i = 1,2,3",
    # "2,3,7,8-TCDD", "k = 1,2,..."): that is a list. Degrees of freedom are
    # safe, since they belong to the LHS ("F(1,23)"), and the value lists that
    # APA prints ("p = .03, .04") put a space after the comma.
    r"|\d+,\d+(?!\d|,[\d.…−–-])"
    r"|\d+(?:\.\d+)?"
    r"|\.\d+"
)

# One printed value: a power of ten, or a number with an optional exponent
_VALUE = r"(?:10" + _POWER + r"|(?:" + _NUMBER + r")" + _EXPONENT + r"?)"

# A range is one value: cut at its dash, "r = .85–.94" read "r = .85".
# "p < 10−8", which a text layer flattened from 10⁻⁸, reads as one too.
_RANGE_END = r"(?:\s*[−–-]\s*[−–-]?" + _VALUE + r")?"

_RHS_PATTERN = (
    r"(?:"
    r"\[[\d.,\s−–-]+\]"  # bracket range [a, b]
    r"|[−–-]?\s*" + _VALUE + _RANGE_END + r")"
)

# ---------------------------------------------------------------------------
# Complete component pattern: lhs comp rhs
# ---------------------------------------------------------------------------

_COMPONENT_RE = re.compile(
    r"(?<![" + _NAME_CHARS + r"])"
    r"(" + _LHS_PATTERN + r")"  # group 1: lhs
    r"(?![A-Za-z0-9_])"
    r"\s*"
    r"(" + _ANY_COMP_PATTERN + r")"  # group 2: comp
    r"\s*"
    r"(" + _RHS_PATTERN + r")",  # group 3: rhs
    re.UNICODE,
)

# ---------------------------------------------------------------------------
# LaTeX equation detection
# ---------------------------------------------------------------------------

_LATEX_DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
# A "$" before a digit is currency, as in late clean-up's strip_inline_math:
# "US$26.3 billion ... CD4 <200 to US$42.5" holds no formula.
_LATEX_INLINE_RE = re.compile(r"(?<!\$)\$(?![\d$])(.+?)(?<!\$)\$(?!\$)")

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

_BROAD_NAME_CHARS = r"\u03B1-\u03C9a-zA-Z\-_\.0-9\{\}\^\\²"

_BROAD_EQUATION_RE = re.compile(
    # A name starts at the start of its token. Starting inside one never adds
    # a match, and on a long token without an operator every start rescanned
    # the rest of it, which is quadratic.
    r"(?<![" + _BROAD_NAME_CHARS + r"])"
    r"("  # group 1: lhs
    r"(?:(?:Cohen['\u2019]s|\d{1,2}%)\s+)?"  #   optional prefix
    r"[" + _BROAD_NAME_CHARS + r"]+"  #   statistic name
    r"(?:\s*\([^)]*\))?"  #   optional parenthesized args
    r")"
    r"\s*"
    r"([" + _BROAD_OP_CHARS + r"]{1,2})"  # group 2: comp
    r"\s*"
    r"("  # group 3: rhs
    r"[−–-]?10" + _POWER + r"|"  #   power of ten: 10⁻⁵, 10^{-5}
    r"[0-9.,+\-\u2212\u2013]*[0-9]" + _EXPONENT + r"?"  #   number, e-5, × 10−5
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
            # Each pass records the character spans of the components it
            # emitted, and a later pass skips any match overlapping one. Only
            # those spans: a statistic the structured patterns do not know,
            # such as "BF10" in "(p < .001, BF10 = 12.3)", stays available to
            # the broad pass and joins the group of its parenthesis.
            # Pass 1: parenthesized statistical groups
            stat_eqs, taken, paren_groups = self._extract_stat_groups(sent)
            equations.extend(stat_eqs)
            # Pass 2: bare statistical expressions, and any a parenthesized
            # part holds after its first
            bare_eqs, bare_spans = self._extract_bare_stats(sent, taken, paren_groups)
            equations.extend(bare_eqs)
            taken += bare_spans
            # Pass 3: LaTeX-delimited equations the structured passes did not
            # already decompose ("$t(28) = 2.10$" is pass 1's t)
            latex_eqs, latex_spans = self._extract_latex_equations(sent, taken)
            equations.extend(latex_eqs)
            taken += latex_spans
            # Pass 4: broad equation detection for missed cases
            equations.extend(self._extract_broad_equations(sent, taken, paren_groups))

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
        # Results handed in may come from another extractor instance; new
        # groups must not reuse their grp_ids.
        self._grp_counter = max(self._grp_counter, max((eq.grp_id for eq in equations), default=0))

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
    ) -> tuple[list[PaperEquation], list[Span], list[tuple[int, int, int]]]:
        """Extract statistical equations from parenthesized groups in a sentence.

        Returns
        -------
        tuple[list[PaperEquation], list[Span], list[tuple[int, int, int]]]
            (equations, component_spans, paren_groups): the (start, end)
            character range of each emitted component, which later passes
            must not extract again, and (start, end, grp_id) of each
            parenthesized group that produced equations, whose group a later
            pass's match inside it joins.
        """
        results: list[PaperEquation] = []
        component_spans: list[Span] = []
        paren_groups: list[tuple[int, int, int]] = []
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
            group_equations: list[PaperEquation] = []
            part_start = paren_start + 1  # parts are split on one character
            for part in _split_respecting_brackets(inner):
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
                    component_spans.append(
                        (part_start + comp_match.start(), part_start + comp_match.end())
                    )
                part_start += len(part) + 1

            if group_equations:
                grp_id = self._next_grp_id()
                for eq in group_equations:
                    eq.grp_id = grp_id
                results.extend(group_equations)
                paren_groups.append((paren_start, paren_end, grp_id))

        return results, component_spans, paren_groups

    def _extract_bare_stats(
        self,
        sent: PaperSentence,
        taken: list[Span],
        paren_groups: list[tuple[int, int, int]],
    ) -> tuple[list[PaperEquation], list[Span]]:
        """Extract stat expressions the parenthesized-group pass did not.

        Finds comma-separated stat patterns like ``t(97.7)=2.9, p=0.005, d=0.59``
        that appear at the sentence level (outside wrapper parentheses), and
        any component a parenthesized part holds after its first.
        Consecutive matches separated only by commas/semicolons/whitespace
        share the same ``grp_id``; a match inside a pass-1 group joins it.
        Returns the equations and their character spans.
        """
        results: list[PaperEquation] = []
        spans: list[Span] = []
        text = sent.text

        # Keep only matches that do NOT overlap an already-extracted component
        bare_matches = [
            m for m in _COMPONENT_RE.finditer(text) if not _overlaps(m.start(), m.end(), taken)
        ]

        for group in _group_adjacent(text, bare_matches):
            grp_id = None
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
                if grp_id is None:
                    grp_id = self._group_id(m.start(), m.end(), paren_groups)
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
                spans.append(m.span())

        return results, spans

    def _extract_latex_equations(
        self, sent: PaperSentence, taken: list[Span]
    ) -> tuple[list[PaperEquation], list[Span]]:
        """Extract equations from LaTeX-delimited content in a sentence.

        A formula whose content the structured passes already decomposed
        (``$t(28) = 2.10$``) is skipped: it would repeat those components
        under a new group. Returns the equations and their formulas' spans.
        """
        results: list[PaperEquation] = []
        spans: list[Span] = []
        text = sent.text

        # Collect all LaTeX spans (display math first, then inline):
        # (content, format, formula span, content span)
        latex_spans: list[tuple[str, str, Span, Span]] = []
        seen_ranges: list[tuple[int, int]] = []

        if sent.is_display_formula:
            # The entire sentence is a display formula (no $$ delimiters).
            latex_spans.append((text, "display", (0, len(text)), (0, len(text))))
        else:
            for m in _LATEX_DISPLAY_RE.finditer(text):
                latex_spans.append((m.group(1), "display", m.span(), m.span(1)))
                seen_ranges.append((m.start(), m.end()))

            for m in _LATEX_INLINE_RE.finditer(text):
                # Skip if overlaps with display math
                overlaps = any(not (m.end() <= s or m.start() >= e) for s, e in seen_ranges)
                if not overlaps:
                    latex_spans.append((m.group(1), "inline", m.span(), m.span(1)))

        for content, fmt, span, content_span in latex_spans:
            if _covered(text, content_span, taken):
                continue
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
                    # "\chi^2(1)" -> "\chi^2", df "1", as the structured
                    # passes split "t(28)"; not "f(x)", which is no df.
                    name, df = _split_lhs_df(lhs)
                    if df and _DF_ARGUMENT_INNER_RE.match(df):
                        lhs = name
                    else:
                        df = ""
                    results.append(
                        PaperEquation(
                            text_id=sent.text_id,
                            grp_id=self._next_grp_id(),
                            lhs=lhs,
                            df=df,
                            comp=comp,
                            rhs=rhs,
                        )
                    )
                    spans.append(span)
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
                    spans.append(span)

        return results, spans

    def _extract_broad_equations(
        self,
        sent: PaperSentence,
        taken: list[Span],
        paren_groups: list[tuple[int, int, int]],
    ) -> list[PaperEquation]:
        """Broad regex pass to catch equations missed by the structured passes.

        Uses the permissive ``_BROAD_EQUATION_RE`` pattern. Matches that
        overlap a component an earlier pass extracted are skipped: the same
        statistic seen twice ("Cohen's d = 0.45" and its "d = 0.45") would
        otherwise be exported twice under different groups.
        Consecutive matches separated only by commas/semicolons/whitespace
        share the same ``grp_id``; a match inside a pass-1 group joins it.
        """
        results: list[PaperEquation] = []
        text = sent.text
        if not _BROAD_OP_PRESCAN_RE.search(text):
            return results

        kept: list[re.Match] = []
        for m in _BROAD_EQUATION_RE.finditer(text):
            if _overlaps(m.start(), m.end(), taken):
                continue
            # A match reaching into a pass-1 group from outside took its name
            # from the prose around it: "PPR (0.97 ± 0.05; t(d.f.16)=1.58".
            if any(
                m.start() < edge < m.end()
                for start, end, _ in paren_groups
                for edge in (start, end)
            ):
                continue

            raw_lhs = m.group(1).strip()
            rhs = m.group(3).strip()

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

            kept.append(m)

        for group in _group_adjacent(text, kept):
            grp_id = self._group_id(group[0].start(), group[0].end(), paren_groups)
            for m in group:
                # Split parenthesized df off the LHS ("t(28)" -> "t", "28"),
                # as the structured passes do.
                lhs, df = _split_lhs_df(m.group(1).strip())
                results.append(
                    PaperEquation(
                        text_id=sent.text_id,
                        grp_id=grp_id,
                        lhs=lhs,
                        df=df,
                        comp=_normalize_comp(m.group(2).strip()),
                        rhs=m.group(3).strip(),
                    )
                )

        return results

    def _group_id(self, start: int, end: int, paren_groups: list[tuple[int, int, int]]) -> int:
        """grp_id of the pass-1 group enclosing ``[start, end)``, else a new one."""
        for paren_start, paren_end, grp_id in paren_groups:
            if paren_start <= start and end <= paren_end:
                return grp_id
        return self._next_grp_id()


# What may separate the components of one reported result: "t(28) = 2.10, p =
# .04" and, in OCR text that wraps each statistic, "$t(28) = 2.10$, $p < .05$".
_GROUP_SEPARATOR_RE = re.compile(r"[\s,;$]+")


def _overlaps(start: int, end: int, spans: list[Span]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def _covered(text: str, span: Span, spans: list[Span]) -> bool:
    """Whether *spans* cover every character of ``text[span]`` but separators."""
    start, end = span
    chars = list(text[start:end])
    for span_start, span_end in spans:
        for index in range(max(span_start, start), min(span_end, end)):
            chars[index - start] = " "
    return not "".join(chars).strip(" \t\n,;.")


def _group_adjacent(text: str, matches: list[re.Match]) -> list[list[re.Match]]:
    """Group consecutive matches separated only by separators (see above)."""
    groups: list[list[re.Match]] = []
    for m in matches:
        if groups and _GROUP_SEPARATOR_RE.fullmatch(text[groups[-1][-1].end() : m.start()]):
            groups[-1].append(m)
        else:
            groups.append([m])
    return groups


def _find_toplevel_comp(content: str) -> re.Match | None:
    """Find a comparison operator in LaTeX content that is not inside braces.

    Skips operators inside ``_{...}`` and ``^{...}`` subscript/superscript
    groups as well as any nested ``{...}`` braces.  This prevents
    ``\\sum_{i=1}`` from being split into ``lhs={i, comp==, rhs=1``.
    A LaTeX relation inside parentheses is an argument, not the formula's
    relation: ``P(X \\geq x) = ...`` splits at ``=``.

    Returns the first :class:`re.Match` at brace depth 0, or ``None``.
    """
    depth = 0
    parens = 0
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
        if ch == "(":
            parens += 1
        elif ch == ")":
            parens = max(0, parens - 1)
        if depth == 0:
            m = _COMP_RE.match(content, i)
            if m and not (parens and m.group().startswith("\\")):
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
        **_LATEX_COMPS,
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
