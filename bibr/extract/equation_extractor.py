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

A pass skips any match overlapping a component an earlier pass emitted, so no
two passes export the same printed statistic. Pass 1 groups each parenthesis;
the components of the later passes are grouped by where they are printed:
adjacent ones share a group, and one inside a pass-1 parenthesis joins its
group. Values are captured whole, as printed: scientific notation
(``2.3 × 10−5``, ``1e-10``), decimal commas (``0,05``) and ranges
(``.85–.94``).

Optional LLM fallback for sentences in methods/results sections that contain
parenthesized numeric groups but where regex extraction found nothing.
"""

import asyncio
import logging
import re
from bisect import bisect_right
from collections.abc import Iterator

from bibr.exceptions import ProcessingError
from bibr.paper_contents import PaperEquation, PaperSection, PaperSentence

logger = logging.getLogger(__name__)

# (start, end) character range in a sentence's text
Span = tuple[int, int]

# ---------------------------------------------------------------------------
# Comparison operators (order matters: longer patterns first)
# ---------------------------------------------------------------------------

_COMP_PATTERN = r"(?:≤|≥|⩽|⩾|≈|≠|≪|≫|<=|>=|<<|>>|<|>|=|~)"

# LaTeX relations, as OCR prints them inside $...$ before late clean-up, and
# the operator _normalize_comp maps each to.
_LATEX_COMPS = {
    r"\le": "≤",
    r"\leq": "≤",
    r"\leqslant": "≤",
    r"\ge": "≥",
    r"\geq": "≥",
    r"\geqslant": "≥",
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
# one: ASCII word characters, Greek letters, superscript/subscript digits and
# letters, and the increment sign ∆ (U+2206) that PDF text layers and Word
# print for Δ. "η²p" is partial eta squared, not a p-value, and "ΔR²" is not
# R². Unicode \w would also stop a name printed flush against CJK text.
_NAME_CHARS = r"A-Za-z0-9_\u0370-\u03FF\u00B2\u00B3\u00B9\u2070-\u209F\u2206"

# A change in a statistic: "ΔR²", "Δχ²(1)", "∆AIC"
_DELTA = r"(?:[Δ∆]\s?)?"

# Test statistics with parenthesized df: t(df), F(df1, df2), χ²(df),
# X²(df, N=n), r(df) (APA correlation), H(df) (Kruskal-Wallis)
_DF_STAT_NAMES = r"(?:F|t|r|H|χ²|χ2|X²|X2)"
_STAT_WITH_DF = re.compile(_DF_STAT_NAMES + r"\s*\(\s*[\d.,\s=Nn]+\s*\)")
# χ² as OCR prints it inside $...$: "\chi^2", "\chi^{2}"
_LATEX_CHI_SQUARE = r"(?:\\chi|X)\s*\^\s*(?:2|\{\s*2\s*\})"

# Simple named statistics (no parenthesized args)
_STAT_SIMPLE_NAMES = (
    r"(?:"
    # Confidence intervals, with their level ("90% CI", "99.9% CI") before CI
    r"\d{1,2}(?:\.\d+)?\s*%\s*CI|95\s*%?\s*CI|CI"
    r"|R²|R2|adj\.\s*R²|adj\.\s*R2"  # R-squared variants
    # Eta and omega squared, partial or generalized, however the sub- and
    # superscript are printed: "η²p", "ηp²", "ηₚ²", "ηG²", and as JATS, HTML
    # and PDF text layers flatten them, "ηp2", "η2p", "η2 p", "η 2 p", "η p 2"
    # and "η 2". Read piecemeal, the p of "η 2 p = .05" was a p-value.
    r"|η(?:\s?[²2]\s?[pPₚG]|\s?[pPₚG]\s?[²2]|\s?[²2]|[pₚ])?"
    r"|ω(?:\s?[²2]\s?[pPₚG]|\s?[pPₚG]\s?[²2]|\s?[²2])"
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
_DF_ARGUMENT_INNER_RE = re.compile(r"[\d.,\s=Nn]+")
_DF_ARGUMENT_OWNER_RE = re.compile(
    r"(?<!["
    + _NAME_CHARS
    + r"])"
    + _DELTA
    + r"(?:"
    + _DF_STAT_NAMES
    + r"|"
    + _LATEX_CHI_SQUARE
    + r")\s*$"
)
# A LaTeX statistic whose parenthesis is its df: "\chi^2(1)" has df "1",
# while "y(n)" and "P''(1)" are function values
_LATEX_DF_NAME_RE = re.compile(
    r"(?:\\Delta\s*|[Δ∆]\s?)?(?:" + _DF_STAT_NAMES + r"|" + _LATEX_CHI_SQUARE + r")"
)

# Full LHS pattern: either stat-with-df or simple stat name, optionally a
# change in it
_LHS_PATTERN = _DELTA + r"(?:" + _STAT_WITH_DF.pattern + r"|" + _STAT_SIMPLE_NAMES + r")"

# ---------------------------------------------------------------------------
# RHS patterns: decimals, decimal commas, scientific notation, ranges
# ---------------------------------------------------------------------------

# A power of ten written with its exponent: "10^-5", OCR LaTeX "10^{-5}",
# "10**-5", or Unicode superscripts "10⁻⁵". Appended to a literal "10".
_POWER = (
    r"(?:\s*(?:\^|\*\*)\s*"
    r"(?:\{\s*[+−–-]?\s*\d+\s*\}|\(\s*[+−–-]?\s*\d+\s*\)|[+−–-]?\s*\d+)"
    r"|[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
)

# Exponent of a number in scientific notation, in every spelling the
# extractor sees (sentence text before late clean-up). Without it
# "p = 2.3 × 10−5" exported as "p = 2.3".
_EXPONENT = (
    # e notation without a sign, printed tight: "2.3e5", "8.0E04"
    r"(?:[eE]\d+"
    # with a sign, tight or spaced: "2.3e-5", "1E−06", "3.2 e -5"
    r"|\s*[eE]\s*[+−–-]\s*\d+"
    # times a power of ten: "× 10⁻⁵", "x 10^-4", "3.8∙10−35", OCR LaTeX
    # "\times 10^{-5}", and "× 10−5" or "× 103" where a JATS or PDF text
    # layer flattened the superscript
    r"|\s*(?:[×xX*·⋅∙]|\\times|\\cdot)\s*10(?:" + _POWER + r"|\s*[+−–-]\s*\d+|\d+))"
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
    # APA prints ("p = .03, .04") put a space after the comma. A count takes
    # no decimal comma at all (_count_value).
    r"|\d+,\d+(?!\d|,[\d.…−–-])"
    r"|\d+\{,\}\d+"  # decimal comma in OCR LaTeX: "0{,}05"
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
    # A name starts neither inside another nor as a superscript: "l^b \geq 0"
    # holds no statistic b. A caret after a space is a footnote mark, as in
    # "*p < .05, ^p < .01".
    r"(?<![" + _NAME_CHARS + r"])(?<![A-Za-z0-9})]\^)"
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
_LATEX_INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
# What makes a "$" before a digit open math rather than an amount (see
# _inline_math): a command ("$2 \times 2$", "$95 \% \mathrm{CI}$") or a lone
# number ("$0.5$")
_LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z%]")
_LONE_NUMBER_RE = re.compile(r"\s*\d[\d.,]*\s*")

# ---------------------------------------------------------------------------
# Broad equation detection — regex improved by Lisa DeBruine
# ---------------------------------------------------------------------------
# Catches stat-like expressions whose names the structured passes do not
# list (e.g. BF10, ICC, uppercase P, t without df).

_BROAD_OP_CHARS = "=<>~\u2248\u2260\u2264\u2265\u226a\u226b\u2a7d\u2a7e"
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
# "i = 1", and "z \le - 1" as a display formula spaces it
_SMALL_INTEGER_RE = re.compile(r"^-?\s*\d$")

# Greek letters (both cases, and the increment sign ∆), so "ΔCFI" and "Δt"
# are names of their own rather than "CFI" and a t statistic
_BROAD_NAME_CHARS = r"\u0391-\u03A9\u03B1-\u03C9\u2206a-zA-Z\-_\.0-9\{\}\^\\²"

_BROAD_EQUATION_RE = re.compile(
    # A name starts at the start of its token. A match starting inside one
    # read the end of a name as the statistic ("x = 120" of "4x = 120"), and
    # on a long token without an operator every start rescanned the rest of
    # it, which is quadratic.
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
    r"^[Δ∆]?(?:"
    r"Cohen['\u2019]s\s+d"
    r"|\d{1,2}%\s+CI"
    r"|[A-Z]{2,8}\d*"
    r"|(?:alpha|beta|gamma|delta|epsilon|theta|lambda|omega)"
    r"|[\u03B1-\u03C9]"
    r"|[A-Za-z]"
    r")(?:\s*\([^)]*\))?$"
)

# ---------------------------------------------------------------------------
# Matches that are not the statistic they look like
# ---------------------------------------------------------------------------

# A lowercase letter printed after a statistic's symbol and a space is that
# symbol's subscript, flattened by an HTML or PDF text layer: "p h = 0.74" (a
# Holm-adjusted p), "r s = 0.42" (Spearman's rho), "R m", "d z", "η p",
# "ε 2 p". Read alone it is another statistic: h is a test to Metacheck, p a
# p-value. The symbol stands alone: "CpxI-3R n = 49" is a group's n, and
# "versus P P < 0.05" a group P's p-value. Only a Greek symbol takes a
# subscript p: "Se d p<0.001" is a p-value.
_SUBSCRIPT_OWNER_RE = re.compile(
    r"(?<![A-Za-z0-9\u0370-\u03FF_.-])(?:[pPrRdt]|[\u0370-\u03FF]\s?[²2]?)\s$"
)
_GREEK_SUBSCRIPT_OWNER_RE = re.compile(
    r"(?<![A-Za-z0-9\u0370-\u03FF_.-])[\u0370-\u03FF]\s?[²2]?\s$"
)

# A value that runs on was cut: an integer before "/" or ":" is a fraction, a
# time or a ratio ("b \leq 1/3", "t = 12:30"), and a number flush against a
# Greek letter or product sign a coefficient ("M \leq 14\eta M"), though
# "63.0\pm 9.0" is whole. Not a count before "/": "n = 12/20 cells" and
# "n = 150/123 WT/KO" print it.
_INTEGER_RE = re.compile(r"[−–-]?\s*\d+")
_FRACTION_OR_TIME_RE = re.compile(r"[/:]\d")
_COEFFICIENT_RE = re.compile(
    r"\\(?:var)?(?:[Aa]lpha|[Bb]eta|[Gg]amma|[Dd]elta|epsilon|[Zz]eta|[Ee]ta|[Tt]heta|iota"
    r"|kappa|[Ll]ambda|mu|nu|[Xx]i|[Pp]i|rho|[Ss]igma|tau|[Uu]psilon|[Pp]hi|chi|[Pp]si"
    r"|[Oo]mega|cdot|times)(?![A-Za-z])"
)

# Counts are whole numbers, so a comma after a count's digits is no decimal
# comma: "n = 9,7 cells" lists two sample sizes, "(n=304,3% of ACSs)" runs
# into a percentage, and "n = 9380,37" into a citation number flattened from
# a superscript. The comma still groups thousands: "N = 1,204".
_COUNT_NAMES = frozenset({"n", "N", "k"})
_THOUSANDS_RE = re.compile(r"[−–-]?\s*\d{1,3}(?:,\d{3})+(?!\d)")


def _misread(text: str, lhs: str, start: int, rhs: str, end: int) -> bool:
    """Whether the match ``text[start:end]`` of *lhs* and *rhs* is not a statistic.

    See the patterns above: a spaced subscript read as a name, or a value
    that runs on past *end*.
    """
    if len(lhs) == 1 and lhs.isascii() and lhs.islower():
        owner = _GREEK_SUBSCRIPT_OWNER_RE if lhs == "p" else _SUBSCRIPT_OWNER_RE
        if owner.search(text, max(0, start - 4), start):
            return True
    if _COEFFICIENT_RE.match(text, end):
        return True
    if lhs in _COUNT_NAMES and text.startswith("/", end):
        return False
    return bool(_INTEGER_RE.fullmatch(rhs) and _FRACTION_OR_TIME_RE.match(text, end))


def _count_value(lhs: str, rhs: str) -> str:
    """*rhs* up to a comma after its digits when *lhs* names a count (see above).

    Not after a 0: "k = 0,75" is a coefficient with a decimal comma.
    """
    if lhs not in _COUNT_NAMES or _THOUSANDS_RE.match(rhs):
        return rhs
    whole = _INTEGER_RE.match(rhs)
    if whole and rhs.startswith((",", "{,}"), whole.end()) and whole.group().strip("−–- 0"):
        return whole.group()
    return rhs


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
            # Each pass marks the characters of the components it emitted,
            # and a later pass skips any match overlapping them. Only those
            # characters: a statistic the structured patterns do not know,
            # such as "BF10" in "(p < .001, BF10 = 12.3)", stays available to
            # the broad pass and joins the group of its parenthesis.
            taken = bytearray(len(sent.text))
            parens = _ParenGroups(sent.text)
            # Pass 1: parenthesized statistical groups
            equations.extend(self._extract_stat_groups(sent, taken, parens))
            # Pass 2: bare statistical expressions, and any a parenthesized
            # part holds after its first
            loose = self._extract_bare_stats(sent, taken, parens)
            # Pass 3: LaTeX-delimited equations the structured passes did not
            # already decompose ("$t(28) = 2.10$" is pass 1's t)
            loose += self._extract_latex_equations(sent, taken)
            # Pass 4: broad equation detection for missed cases
            loose += self._extract_broad_equations(sent, taken, parens)
            # Grouped together, by where they are printed: "Z = 2.84; P =
            # 0.035" is one result though pass 2 reads Z and pass 4 P.
            equations.extend(self._group_by_position(sent.text, loose, parens))

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
                # Dedupe on (text_id, lhs, comp, rhs), within this LLM batch
                # and against everything already extracted. The regex passes
                # dedupe by where a component is printed, which the LLM's
                # components do not record.
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
        self, sent: PaperSentence, taken: bytearray, parens: "_ParenGroups"
    ) -> list[PaperEquation]:
        """Extract statistical equations from parenthesized groups in a sentence.

        Marks each emitted component's characters in *taken*, which later
        passes must not extract again, and records in *parens* each group
        that produced equations, whose group a later pass's match inside it
        joins.
        """
        results: list[PaperEquation] = []
        text = sent.text

        for inner, paren_start, paren_end in _iter_parenthesized_groups(text):
            # Validate: must contain a comparison operator and a digit
            if not _COMP_RE.search(inner) or not re.search(r"\d", inner):
                continue

            # Unwrapped APA form: "chi2(1, N = 100) = 3.84". The df parenthesis
            # is part of the statistic, not a group in its own right.
            if parens.in_df_argument(paren_start, paren_end):
                continue

            # One component per comma/semicolon-separated part (not inside ()
            # or []); part i ends where part_ends[i] holds its separator.
            part_ends: list[int] = []
            position = paren_start + 1
            for part in _split_respecting_brackets(inner):
                position += len(part)
                part_ends.append(position)
                position += 1

            # Matched across the whole group rather than part by part, so a
            # value sees what follows its part: "(k = 1,2,…,K)" is a list.
            group_equations: list[PaperEquation] = []
            last_part = -1
            for m in _COMPONENT_RE.finditer(text, paren_start + 1, paren_end - 1):
                part_index = bisect_right(part_ends, m.start())
                if part_index == last_part:
                    continue  # a part's later components are pass 2's
                if parens.in_df_argument(m.start(), m.end()):
                    continue  # "N = 100" of "($\chi^{2}(1, N = 100) = 3.84$)"
                component = _structured_component(text, m)
                if component is None:
                    continue
                (start, end), lhs, df, comp, rhs = component
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
                _take(taken, start, end)
                last_part = part_index

            if group_equations:
                grp_id = self._next_grp_id()
                for eq in group_equations:
                    eq.grp_id = grp_id
                results.extend(group_equations)
                parens.add_group(paren_start, paren_end, grp_id)

        return results

    def _extract_bare_stats(
        self, sent: PaperSentence, taken: bytearray, parens: "_ParenGroups"
    ) -> list[tuple[Span, PaperEquation]]:
        """Extract stat expressions the parenthesized-group pass did not.

        Finds comma-separated stat patterns like ``t(97.7)=2.9, p=0.005, d=0.59``
        that appear at the sentence level (outside wrapper parentheses), and
        any component a parenthesized part holds after its first. Returns
        each component with its span, ungrouped, and marks it in *taken*.
        """
        results: list[tuple[Span, PaperEquation]] = []
        text = sent.text

        for m in _COMPONENT_RE.finditer(text):
            if _is_taken(taken, m.start(), m.end()) or parens.in_df_argument(m.start(), m.end()):
                continue
            component = _structured_component(text, m)
            if component is None:
                continue
            (start, end), lhs, df, comp, rhs = component
            raw_lhs = m.group(1).strip()
            if _NUMERIC_ONLY_RE.match(raw_lhs):
                continue
            # On the LHS as printed, so "t(28)" (len > 2) is not collapsed to
            # "t" before the length check.
            if len(raw_lhs) <= 2 and _SMALL_INTEGER_RE.match(rhs):
                continue
            results.append(
                (
                    (start, end),
                    PaperEquation(
                        text_id=sent.text_id, grp_id=0, lhs=lhs, df=df, comp=comp, rhs=rhs
                    ),
                )
            )
            _take(taken, start, end)

        return results

    def _extract_latex_equations(
        self, sent: PaperSentence, taken: bytearray
    ) -> list[tuple[Span, PaperEquation]]:
        """Extract equations from LaTeX-delimited content in a sentence.

        A formula whose content the structured passes already decomposed
        (``$t(28) = 2.10$``) is skipped: it would repeat those components.
        Returns each equation with its formula's span, ungrouped, and marks
        the formula in *taken*.
        """
        results: list[tuple[Span, PaperEquation]] = []
        text = sent.text

        # Collect all LaTeX spans (display math first, then inline):
        # (content, format, formula span, content span)
        latex_spans: list[tuple[str, str, Span, Span]] = []
        seen_ranges: list[Span] = []

        if sent.is_display_formula:
            # The entire sentence is a display formula (no $$ delimiters).
            latex_spans.append((text, "display", (0, len(text)), (0, len(text))))
        else:
            for m in _LATEX_DISPLAY_RE.finditer(text):
                latex_spans.append((m.group(1), "display", m.span(), m.span(1)))
                seen_ranges.append((m.start(), m.end()))

            for m in _inline_math(text):
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
            if (
                comp_match
                and fmt == "inline"
                and _is_taken(taken, content_span[0] + comp_match.start(), content_span[1])
            ):
                # The structured passes read this formula's relation or a
                # statistic after it: "$p < 0.001^{***}$" is their "p <
                # 0.001", and "$t = 2.1, p = .03$" is no t of "2.1, p = .03".
                continue
            if comp_match:
                lhs = content[: comp_match.start()].strip()
                comp = _normalize_comp(comp_match.group())
                rhs = content[comp_match.end() :].strip()

                # Strip LaTeX commands for cleaner output but keep the substance
                if lhs and rhs:
                    # "\chi^2(1)" -> "\chi^2", df "1", as the structured
                    # passes split "t(28)"; not "y(n)" or "P''(1)", which
                    # are function values.
                    name, df = _split_lhs_df(lhs)
                    if (
                        df
                        and _LATEX_DF_NAME_RE.fullmatch(name)
                        and _DF_ARGUMENT_INNER_RE.fullmatch(df)
                        and re.search(r"\d", df)
                    ):
                        lhs = name
                    else:
                        df = ""
                    results.append(
                        (
                            span,
                            PaperEquation(
                                text_id=sent.text_id, grp_id=0, lhs=lhs, df=df, comp=comp, rhs=rhs
                            ),
                        )
                    )
                    _take(taken, *span)
            elif fmt == "display":
                # Pure display formula with no comparison operator (e.g. $$\Delta_{i}$$).
                # Still record it so all LaTeX formulas appear in the equations array.
                # Inline math without an operator is typically just notation, not an equation.
                stripped_content = content.strip()
                if stripped_content and not _TRIVIAL_LATEX_RE.match(stripped_content):
                    results.append(
                        (
                            span,
                            PaperEquation(
                                text_id=sent.text_id,
                                grp_id=0,
                                lhs=stripped_content,
                                comp="",
                                rhs="",
                            ),
                        )
                    )
                    _take(taken, *span)

        return results

    def _extract_broad_equations(
        self, sent: PaperSentence, taken: bytearray, parens: "_ParenGroups"
    ) -> list[tuple[Span, PaperEquation]]:
        """Broad regex pass to catch equations missed by the structured passes.

        Uses the permissive ``_BROAD_EQUATION_RE`` pattern. Matches that
        overlap a component an earlier pass extracted are skipped: the
        broad pattern reads "t(28) = 2.10" too, which pass 2 already
        exported. Returns each equation with its span, ungrouped.
        """
        results: list[tuple[Span, PaperEquation]] = []
        text = sent.text
        if not _BROAD_OP_PRESCAN_RE.search(text):
            return results

        for m in _BROAD_EQUATION_RE.finditer(text):
            start, end = m.span()
            if _is_taken(taken, start, end):
                continue
            # A match reaching into a pass-1 group from outside took its name
            # from the prose around it: "PPR (0.97 ± 0.05; t(d.f.16)=1.58".
            if parens.crosses(start, end):
                continue

            raw_lhs = m.group(1).strip()
            rhs = m.group(3).strip()

            # Skip if LHS contains a LaTeX command — these are LaTeX fragments
            # that should be handled by the LaTeX pass, not classified as stat.
            if _LATEX_CMD_RE.search(raw_lhs):
                continue

            if not _PLAUSIBLE_BROAD_LHS_RE.fullmatch(raw_lhs):
                continue

            if _misread(text, raw_lhs, start, rhs, end):
                continue
            rhs = _count_value(raw_lhs, rhs)
            end = m.start(3) + len(rhs)

            # Skip trivial equations: purely numeric LHS (6 = 24), or
            # short LHS with small bare integer RHS (i = 1, b = 5).
            # Keeps real stats: d = 0.45, N = 120, p < .001.
            if _NUMERIC_ONLY_RE.match(raw_lhs):
                continue
            if len(raw_lhs) <= 2 and _SMALL_INTEGER_RE.match(rhs):
                continue

            # Split parenthesized df off the LHS ("t(28)" -> "t", "28"),
            # as the structured passes do.
            lhs, df = _split_lhs_df(raw_lhs)
            results.append(
                (
                    (start, end),
                    PaperEquation(
                        text_id=sent.text_id,
                        grp_id=0,
                        lhs=lhs,
                        df=df,
                        comp=_normalize_comp(m.group(2).strip()),
                        rhs=rhs,
                    ),
                )
            )

        return results

    def _group_by_position(
        self, text: str, components: list[tuple[Span, PaperEquation]], parens: "_ParenGroups"
    ) -> list[PaperEquation]:
        """Assign grp_ids to passes 2-4's components, in the order printed.

        Consecutive components separated only by commas, semicolons,
        whitespace or "$" share a group; one inside a pass-1 parenthesis
        joins that parenthesis's group.
        """
        components.sort(key=lambda component: component[0][0])
        grouped: list[PaperEquation] = []
        grp_id = 0
        previous_end = 0
        for (start, end), eq in components:
            if not grouped or not _GROUP_SEPARATOR_RE.fullmatch(text, previous_end, start):
                grp_id = parens.group_of(start, end) or self._next_grp_id()
            eq.grp_id = grp_id
            grouped.append(eq)
            previous_end = end
        return grouped


class _ParenGroups:
    """The parentheses of one sentence that the passes treat specially.

    The groups pass 1 emitted components for (outermost, so in text order
    and never nested), whose grp_id a later pass's match inside them joins;
    and statistics' own df arguments, at any depth: "(1, N = 100)" of
    "χ²(1, N = 100) = 3.84" or of OCR's "$\\chi^{2}(1, N = 100) = 3.84$" holds
    no statistic of its own, since the statistic's df holds it.
    """

    def __init__(self, text: str) -> None:
        self._starts: list[int] = []
        self._ends: list[int] = []
        self._grp_ids: list[int] = []
        self._edges: list[int] = []
        df_arguments: list[Span] = []
        stack: list[int] = []
        for index, char in enumerate(text):
            if char == "(":
                stack.append(index)
            elif char == ")" and stack:
                start = stack.pop()
                if _DF_ARGUMENT_INNER_RE.fullmatch(
                    text, start + 1, index
                ) and _DF_ARGUMENT_OWNER_RE.search(text, max(0, start - 24), start):
                    df_arguments.append((start, index + 1))
        # They hold no parenthesis, so they never nest either
        df_arguments.sort()
        self._df_starts = [start for start, _ in df_arguments]
        self._df_ends = [end for _, end in df_arguments]

    def add_group(self, start: int, end: int, grp_id: int) -> None:
        self._starts.append(start)
        self._ends.append(end)
        self._grp_ids.append(grp_id)
        self._edges += (start, end)

    def group_of(self, start: int, end: int) -> int | None:
        """grp_id of the group enclosing ``[start, end)``, if any."""
        index = bisect_right(self._starts, start) - 1
        if index >= 0 and end <= self._ends[index]:
            return self._grp_ids[index]
        return None

    def crosses(self, start: int, end: int) -> bool:
        """Whether a group's parenthesis lies strictly inside ``(start, end)``."""
        index = bisect_right(self._edges, start)
        return index < len(self._edges) and self._edges[index] < end

    def in_df_argument(self, start: int, end: int) -> bool:
        """Whether ``[start, end)`` lies in a statistic's df argument."""
        index = bisect_right(self._df_starts, start) - 1
        return index >= 0 and end <= self._df_ends[index]


# What may separate the components of one reported result: "t(28) = 2.10, p =
# .04" and, in OCR text that wraps each statistic, "$t(28) = 2.10$, $p < .05$".
_GROUP_SEPARATOR_RE = re.compile(r"[\s,;$]+")

# What may be left of a formula whose statistics the structured passes read
_FORMULA_SEPARATORS = frozenset(" \t\n,;.")


def _structured_component(text: str, m: re.Match[str]) -> tuple[Span, str, str, str, str] | None:
    """(span, lhs, df, comp, rhs) of a _COMPONENT_RE match, or None if misread."""
    lhs = m.group(1).strip()
    rhs = m.group(3).strip()
    if _misread(text, lhs, m.start(), rhs, m.end()):
        return None
    rhs = _count_value(lhs, rhs)
    name, df = _split_lhs_df(lhs)
    return (m.start(), m.start(3) + len(rhs)), name, df, _normalize_comp(m.group(2).strip()), rhs


def _take(taken: bytearray, start: int, end: int) -> None:
    taken[start:end] = b"\x01" * (end - start)


def _is_taken(taken: bytearray, start: int, end: int) -> bool:
    return taken.find(1, start, end) != -1


def _covered(text: str, span: Span, taken: bytearray) -> bool:
    """Whether *taken* covers every character of ``text[span]`` but separators."""
    start, end = span
    return all(taken[index] or text[index] in _FORMULA_SEPARATORS for index in range(start, end))


def _inline_math(text: str) -> Iterator[re.Match[str]]:
    """The inline ``$...$`` formulas of *text*, without currency amounts.

    A "$" before a digit opens an amount ("US$26.3 billion ... to US$42.5"),
    unless the span it would open reads as math: it holds a LaTeX command
    ("$2 \\times 2$") or a lone number ("$0.5$"), and its closing "$" does
    not itself start an amount. After an amount the scan resumes past its
    "$", as late clean-up's strip_inline_math does, so a formula later in
    the sentence still pairs up.
    """
    position = 0
    while (m := _LATEX_INLINE_RE.search(text, position)) is not None:
        content = m.group(1)
        if content[0].isdigit() and (
            text[m.end() : m.end() + 1].isdigit()
            or not (_LATEX_COMMAND_RE.search(content) or _LONE_NUMBER_RE.fullmatch(content))
        ):
            position = m.start() + 1
            continue
        yield m
        position = m.end()


def _find_toplevel_comp(content: str) -> re.Match | None:
    """Find a comparison operator in LaTeX content that is not inside braces.

    Skips operators inside ``_{...}`` and ``^{...}`` subscript/superscript
    groups as well as any nested ``{...}`` braces.  This prevents
    ``\\sum_{i=1}`` from being split into ``lhs={i, comp==, rhs=1``.
    An operator inside parentheses is an argument, not the formula's
    relation: ``P(X \\geq x) = ...`` and ``\\chi^{2}(1, N = 100) = 3.84``
    split at the ``=`` after the parenthesis. Only when there is none
    outside parentheses (an unclosed one, or ``P(X < x)`` alone) does a
    plain operator inside them count, as it did before.

    Returns the first :class:`re.Match` at brace depth 0, or ``None``.
    """
    depth = 0
    parens = 0
    fallback: re.Match | None = None
    for i, ch in enumerate(content):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        elif ch == "(":
            parens += 1
        elif ch == ")":
            parens = max(0, parens - 1)
        elif depth == 0:
            m = _COMP_RE.match(content, i)
            if m is None:
                continue
            if not parens:
                return m
            if fallback is None and not m.group().startswith("\\"):
                fallback = m
    return fallback


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
        "⩽": "≤",
        "⩾": "≥",
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
