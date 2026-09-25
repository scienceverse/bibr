"""Tests for equation extraction from paper sentences."""

import asyncio

import pytest

from bibr.extract.equation_extractor import EquationExtractor
from bibr.paper_contents import PaperSection, PaperSentence


def _make_sentence(text_id: int, text: str, section_id: int = 1) -> PaperSentence:
    return PaperSentence(text_id=text_id, text=text, section_id=section_id, paragraph_id=1)


def _make_sections() -> list[PaperSection]:
    return [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        PaperSection(section_id=1, header="Results", level=1, parent_section_id=0),
    ]


def _components(eqs) -> list[tuple[str, str, str, str]]:
    return [(eq.lhs, eq.df, eq.comp, eq.rhs) for eq in eqs]


def _extract(text: str):
    return EquationExtractor().extract_from_sentences([_make_sentence(1, text)], _make_sections())


class TestStatisticalExtraction:
    """Test extraction of statistical equations from parenthesized groups."""

    def test_t_test_in_parens(self):
        """Classic t-test result: (t(28) = 3.42, p = .003, d = 0.45)."""
        sent = _make_sentence(
            1,
            "The difference was significant (t(28) = 3.42, p = .003, d = 0.45).",
        )
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert len(eqs) == 3
        # All should share the same grp_id
        grp_ids = {eq.grp_id for eq in eqs}
        assert len(grp_ids) == 1
        assert all(eq.text_id == 1 for eq in eqs)

        # Check individual components — df is split off the LHS
        lhs_map = {eq.lhs: eq for eq in eqs}
        assert "t" in lhs_map
        assert lhs_map["t"].df == "28"
        assert lhs_map["t"].comp == "="
        assert lhs_map["t"].rhs == "3.42"
        assert "p" in lhs_map
        assert lhs_map["p"].df == ""
        assert lhs_map["p"].rhs == ".003"
        assert "d" in lhs_map
        assert lhs_map["d"].rhs == "0.45"

    def test_f_test(self):
        """F-test: (F(2, 47) = 5.13, p = .002)."""
        sent = _make_sentence(1, "The ANOVA showed (F(2, 47) = 5.13, p = .002).")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert len(eqs) == 2
        lhs_map = {eq.lhs: eq for eq in eqs}
        assert "F" in lhs_map
        assert lhs_map["F"].df == "2, 47"
        assert lhs_map["F"].rhs == "5.13"
        assert "p" in lhs_map

    def test_chi_square(self):
        """Chi-square: (χ²(4) = 12.3, p < .05)."""
        sent = _make_sentence(1, "The test was significant (χ²(4) = 12.3, p < .05).")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert len(eqs) == 2
        lhs_map = {eq.lhs: eq for eq in eqs}
        assert "χ²" in lhs_map
        assert lhs_map["χ²"].df == "4"
        assert lhs_map["χ²"].rhs == "12.3"
        assert "p" in lhs_map
        assert lhs_map["p"].comp == "<"

    def test_confidence_interval(self):
        """CI: (95% CI = [2.0, 4.7])."""
        sent = _make_sentence(1, "The mean difference (95% CI = [2.0, 4.7]) was large.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert _components(eqs) == [("95% CI", "", "=", "[2.0, 4.7]")]

    def test_effect_sizes(self):
        """Multiple effect sizes: (d = 0.45), (η² = .03), (R² = .42)."""
        sentences = [
            _make_sentence(1, "The effect was medium (d = 0.45)."),
            _make_sentence(2, "Partial eta-squared was small (η² = .03)."),
            _make_sentence(3, "The model explained variance (R² = .42)."),
        ]
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences(sentences, _make_sections())

        assert len(eqs) == 3
        lhs_values = {eq.lhs for eq in eqs}
        assert "d" in lhs_values
        assert "η²" in lhs_values
        assert "R²" in lhs_values

    def test_p_value_inequality(self):
        """p < .001 with inequality operator."""
        sent = _make_sentence(1, "The result was highly significant (p < .001).")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert len(eqs) == 1
        assert eqs[0].lhs == "p"
        assert eqs[0].comp == "<"
        assert eqs[0].rhs == ".001"

    def test_signed_leading_dot_correlation(self):
        sent = _make_sentence(1, "The measures were correlated (r = -.23, p = .04).")

        eqs = EquationExtractor().extract_from_sentences([sent], _make_sections())

        assert {eq.lhs: eq.rhs for eq in eqs} == {"r": "-.23", "p": ".04"}

    def test_no_false_positives(self):
        """Regular text without statistical content should not trigger extraction."""
        sentences = [
            _make_sentence(1, "The study was conducted at the University of Oxford."),
            _make_sentence(2, "Participants were recruited from local schools."),
            _make_sentence(3, "Data were collected over a period of six months."),
        ]
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences(sentences, _make_sections())

        assert len(eqs) == 0

    def test_multiple_groups_in_sentence(self):
        """Multiple parenthesized groups in one sentence get distinct grp_ids."""
        sent = _make_sentence(
            1,
            "Group A (M = 4.2, SD = 1.1) differed from Group B (M = 3.1, SD = 0.9).",
        )
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert len(eqs) == 4
        grp_ids = {eq.grp_id for eq in eqs}
        assert len(grp_ids) == 2  # two distinct groups

    def test_grp_id_globally_unique(self):
        """grp_ids should increment across sentences, not reset."""
        sentences = [
            _make_sentence(1, "First (t(10) = 2.1, p = .05)."),
            _make_sentence(2, "Second (F(1, 20) = 4.3, p = .04)."),
        ]
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences(sentences, _make_sections())

        grp_ids = sorted({eq.grp_id for eq in eqs})
        assert grp_ids == [1, 2]


class TestThousandsSeparators:
    """A grouped-digit number is one value, not a component boundary."""

    def test_sample_size_with_thousands_separator_survives(self):
        sent = _make_sentence(1, "The sample (N = 12,345) was large.")
        eqs = EquationExtractor().extract_from_sentences([sent], _make_sections())

        assert [(eq.lhs, eq.comp, eq.rhs) for eq in eqs] == [("N", "=", "12,345")]

    def test_separator_does_not_split_a_multi_component_group(self):
        sent = _make_sentence(1, "Results (N = 1,204, M = 3.4) held.")
        eqs = EquationExtractor().extract_from_sentences([sent], _make_sections())

        assert [(eq.lhs, eq.rhs) for eq in eqs] == [("N", "1,204"), ("M", "3.4")]
        assert len({eq.grp_id for eq in eqs}) == 1


class TestChiSquareDfArgument:
    """A statistic's own df parenthesis is not an independent stat group.

    Treating it as one emitted a bare ``N = 100``, recorded its span, and the
    span then vetoed the correct full match in both later passes — so the
    export carried N and p in different groups and no chi-square at all.
    """

    def test_unwrapped_chi_square_with_n_in_df(self):
        sent = _make_sentence(1, "We found \u03c7\u00b2(1, N = 100) = 3.84, p = .05.")
        eqs = EquationExtractor().extract_from_sentences([sent], _make_sections())

        by_lhs = {eq.lhs: eq for eq in eqs}
        assert set(by_lhs) == {"\u03c7\u00b2", "p"}
        assert by_lhs["\u03c7\u00b2"].df == "1, N = 100"
        assert by_lhs["\u03c7\u00b2"].rhs == "3.84"
        assert len({eq.grp_id for eq in eqs}) == 1

    def test_unwrapped_chi_square_with_grouped_n(self):
        sent = _make_sentence(1, "We found \u03c7\u00b2(2, N = 1,024) = 9.11, p = .01.")
        eqs = EquationExtractor().extract_from_sentences([sent], _make_sections())

        by_lhs = {eq.lhs: eq for eq in eqs}
        assert by_lhs["\u03c7\u00b2"].rhs == "9.11"
        assert "N" not in by_lhs

    def test_wrapped_chi_square_is_unchanged(self):
        sent = _make_sentence(1, "The model (\u03c7\u00b2(1, N = 100) = 3.84, p = .05) fit.")
        eqs = EquationExtractor().extract_from_sentences([sent], _make_sections())

        by_lhs = {eq.lhs: eq for eq in eqs}
        assert set(by_lhs) == {"\u03c7\u00b2", "p"}

    def test_latex_chi_square_df_holds_its_n(self):
        eqs = _extract("It held ($\\chi^{2}(1, N = 100) = 3.84$, $p = .05$).")

        assert _components(eqs) == [
            ("p", "", "=", ".05"),
            ("\\chi^{2}", "1, N = 100", "=", "3.84"),
        ]

    def test_n_of_a_statistic_without_a_value_is_kept(self):
        # Nothing else carries this N, so the broad pass reads it
        assert _components(_extract("The test χ²(1, N = 100) was significant.")) == [
            ("N", "", "=", "100")
        ]

    def test_f_test_df_pair_is_still_a_df(self):
        sent = _make_sentence(1, "F(2, 45) = 5.6, p = .007.")
        eqs = EquationExtractor().extract_from_sentences([sent], _make_sections())

        by_lhs = {eq.lhs: eq for eq in eqs}
        assert by_lhs["F"].df == "2, 45"
        assert by_lhs["F"].rhs == "5.6"


class TestBareStatisticalExtraction:
    """Test extraction of stat expressions outside parenthesized groups."""

    def test_bare_stats_after_means(self):
        """Stats like t(97.7)=2.9, p=0.005, d=0.59 outside parens."""
        sent = _make_sentence(
            1,
            "Group A (M=9.12) vs Group B (M=10.9), t(97.7)=2.9, p=0.005, d=0.59.",
        )
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert len(eqs) == 5

        # Parenthesized M values
        m_eqs = [eq for eq in eqs if eq.lhs == "M"]
        assert len(m_eqs) == 2

        # Bare stats
        t_eq = next(eq for eq in eqs if eq.lhs == "t")
        p_eq = next(eq for eq in eqs if eq.lhs == "p")
        d_eq = next(eq for eq in eqs if eq.lhs == "d")

        assert t_eq.df == "97.7"
        assert t_eq.rhs == "2.9"
        assert p_eq.rhs == "0.005"
        assert d_eq.rhs == "0.59"

        # Bare stats share the same grp_id
        assert t_eq.grp_id == p_eq.grp_id == d_eq.grp_id

        # Bare stats have different grp_id from parenthesized groups
        assert all(m.grp_id != t_eq.grp_id for m in m_eqs)

    def test_bare_stats_negative_rhs(self):
        """Bare t-test with negative value: t(97.2)=-1.96, p=0.152."""
        sent = _make_sentence(
            1,
            "The app (M=5.06) vs checklist (M=4.5), t(97.2)=-1.96, p=0.152.",
        )
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert len(eqs) == 4

        t_eq = next(eq for eq in eqs if eq.lhs == "t")
        assert t_eq.df == "97.2"
        assert t_eq.rhs == "-1.96"

        p_eq = next(eq for eq in eqs if eq.lhs == "p")
        assert p_eq.rhs == "0.152"
        assert t_eq.grp_id == p_eq.grp_id

    def test_no_bare_stats_when_all_in_parens(self):
        """When all stats are inside parens, bare pass should add nothing extra."""
        sent = _make_sentence(
            1,
            "The difference was significant (t(28) = 3.42, p = .003, d = 0.45).",
        )
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        # Should still be exactly 3, no duplicates from bare pass
        assert len(eqs) == 3

    def test_bare_stats_separated_from_non_stats(self):
        """Bare stats separated by non-stat text get different grp_ids."""
        sent = _make_sentence(
            1,
            "Result: t(10)=2.1, p=.05. Another result: F(1,20)=4.3, p=.04.",
        )
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        t_eq = next(eq for eq in eqs if eq.lhs == "t")
        f_eq = next(eq for eq in eqs if eq.lhs == "F")

        # t-test and F-test should be in different groups
        assert t_eq.grp_id != f_eq.grp_id


class TestLatexExtraction:
    """Test extraction of equations from LaTeX-delimited content."""

    def test_latex_equation(self):
        """$\\alpha = 0.05$ should be extracted as a LaTeX equation."""
        sent = _make_sentence(1, "We used $\\alpha = 0.05$ as the significance level.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert len(eqs) == 1
        eq = eqs[0]
        assert eq.lhs is not None
        assert eq.lhs == "\\alpha"
        assert eq.comp == "="
        assert eq.rhs == "0.05"

    def test_latex_no_comparison(self):
        """$x^2 + y^2$ with no comparison operator should not be extracted."""
        sent = _make_sentence(1, "The formula $x^2 + y^2$ describes the relationship.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert len(eqs) == 0

    def test_display_math_equation(self):
        """$$\\beta = 0.73$$ should be extracted as display math."""
        sent = _make_sentence(1, "The coefficient was $$\\beta = 0.73$$ in the model.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert len(eqs) == 1
        eq = eqs[0]
        assert eq.lhs is not None
        assert eq.lhs == "\\beta"
        assert eq.comp == "="
        assert eq.rhs == "0.73"


class TestBroadEquationDetection:
    """Test the broad regex pass (pass 4) that catches missed equations."""

    def test_cohens_d(self):
        """Cohen's d = 0.8 keeps its full name, once (not also as a bare d)."""
        sent = _make_sentence(1, "The effect was large, Cohen\u2019s d = 0.8.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert _components(eqs) == [("Cohen\u2019s d", "", "=", "0.8")]

    def test_bayes_factor(self):
        """BF10 = 12.4 should be caught by the broad pass."""
        sent = _make_sentence(1, "Evidence was strong, BF10 = 12.4.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert _components(eqs) == [("BF10", "", "=", "12.4")]

    def test_scientific_notation(self):
        """p = 3.2 e -5 is one component with the whole value, not also p = 3.2."""
        sent = _make_sentence(1, "The result was significant, p = 3.2 e -5.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert _components(eqs) == [("p", "", "=", "3.2 e -5")]

    def test_power_of_10_notation(self):
        """p = 2.1 x 10^-4 is one component with the power-of-10 suffix."""
        sent = _make_sentence(1, "We found p = 2.1 x 10^-4 in the analysis.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert _components(eqs) == [("p", "", "=", "2.1 x 10^-4")]

    def test_tilde_operator(self):
        """β ~ 0.45 with tilde as approximate operator."""
        sent = _make_sentence(1, "The estimate was approximately beta ~ 0.45.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert _components(eqs) == [("beta", "", "~", "0.45")]

    def test_icc_stat(self):
        """ICC = 0.85 should be caught by the broad pass."""
        sent = _make_sentence(1, "Reliability was high, ICC = 0.85.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert _components(eqs) == [("ICC", "", "=", "0.85")]

    def test_broad_no_duplicate_with_structured(self):
        """Broad pass should not duplicate equations already found by structured passes."""
        sent = _make_sentence(1, "The result (t(28) = 3.42, p < .001).")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        # Structured passes find t(28)=3.42 and p<.001; broad pass should add nothing
        assert len(eqs) == 2

    def test_percentage_prefix(self):
        """5% CI = [1.2, 3.4] with percentage prefix."""
        sent = _make_sentence(1, "The 5% CI = [1.2, 3.4] was computed.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert _components(eqs) == [("5% CI", "", "=", "[1.2, 3.4]")]

    def test_table_legend_color_threshold_is_not_an_equation(self):
        sent = _make_sentence(
            462,
            "Note: White is 0–20%; light green 21–40%; green 41–60%; and dark green >60%.",
        )
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert eqs == []

    def test_single_letter_stat_does_not_match_inside_word(self):
        sent = _make_sentence(1, "The dark green >60% category was retained.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert not any(eq.lhs == "n" for eq in eqs)


class TestTrivialLatexFiltering:
    """Test filtering of trivial LaTeX like citation superscripts."""

    def test_superscript_citation_filtered(self):
        """$^{6}$ is a citation marker, not a real equation."""
        sent = _make_sentence(1, "This was reported previously $^{6}$ in the literature.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())
        assert len(eqs) == 0

    def test_multi_citation_superscript_filtered(self):
        """$^{18,28}$ is a multi-citation marker, not a real equation."""
        sent = _make_sentence(1, "Several studies $^{18,28}$ confirmed this.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())
        assert len(eqs) == 0

    def test_real_latex_formula_kept(self):
        """Real LaTeX formulas like $\\Delta_{i}$ should still be extracted."""
        sent = _make_sentence(1, "The change $$\\Delta_{i}$$ was computed.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())
        assert len(eqs) == 1
        assert eqs[0].lhs is not None

    def test_latex_command_not_classified_as_stat(self):
        r"""\\mathrm{n = 3 should not be classified as stat by the broad pass."""
        sent = _make_sentence(1, r"The sample size was \mathrm{n = 3} participants.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())
        # Should not produce a stat equation with lhs containing \mathrm
        stat_eqs = [eq for eq in eqs if "mathrm" in eq.lhs]
        assert len(stat_eqs) == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_sentences(self):
        """Empty sentence list should return no equations."""
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([], _make_sections())
        assert eqs == []

    def test_negative_rhs(self):
        """Negative numbers in RHS: (β = -0.32)."""
        sent = _make_sentence(1, "The coefficient was negative (β = -0.32).")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert _components(eqs) == [("β", "", "=", "-0.32")]

    def test_less_than_or_equal(self):
        """p ≤ .05 with Unicode operator."""
        sent = _make_sentence(1, "Significance threshold (p ≤ .05) was met.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert len(eqs) == 1
        assert eqs[0].comp == "≤"

    def test_all_text_ids_correct(self):
        """Each equation should carry the correct text_id from its source sentence."""
        sentences = [
            _make_sentence(10, "First (p = .01)."),
            _make_sentence(20, "Second (p = .05)."),
        ]
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences(sentences, _make_sections())

        text_ids = {eq.text_id for eq in eqs}
        assert text_ids == {10, 20}

    def test_unbalanced_parentheses_do_not_hang(self):
        """Malformed OCR-style parenthesis noise should complete quickly and safely."""
        noisy = "Results " + "(" * 5000 + " p = .05 " + "x" * 2000
        sent = _make_sentence(1, noisy)
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        # The bare-stats pass may legitimately find "p = .05" even in noisy
        # text — the important thing is that extraction completes without hanging.
        assert len(eqs) <= 1


class TestCompleteValues:
    """A value is captured whole: cutting it gives a different number.

    ``p = 2.3 × 10−5`` exported as ``p = 2.3``, ``p < 1e-10`` as ``p < 1``
    (which passes any p <= 1 check) and ``p = 0,05`` as ``p = 0``.
    """

    @pytest.mark.parametrize(
        ("value", "note"),
        [
            ("2.3 × 10−5", "JATS/PDF text layer: superscript flattened, U+2212"),
            ("2.3 × 10⁻⁵", "Unicode superscripts"),
            ("2.3 x 10^-5", "ASCII caret"),
            ("2.3 × 10^(-5)", "parenthesized exponent"),
            ("2.3 x 10**-5", "ASCII double star"),
            ("2.3 X 10−5", "capital X"),
            ("3.4 · 10−6", "middle dot"),
            ("3.8∙10−35", "bullet operator (PLOS)"),
            ("2.3e-5", "e notation"),
            ("3.16E-20", "E notation"),
            ("3.43e–7", "e notation, en dash (native PDF text)"),
            ("1.45E − 09", "spaced e notation"),
            ("2.3 \\times 10^{-5}", "OCR LaTeX before late clean-up"),
            ("10⁻⁵", "bare power of ten"),
            ("10^{-10}", "bare power of ten, OCR LaTeX"),
            ("10−8", "bare power of ten, flattened superscript"),
        ],
    )
    def test_scientific_notation_in_parentheses(self, value, note):
        eqs = _extract(f"The effect was significant (t(28) = 5.10, p = {value}).")

        assert _components(eqs) == [("t", "28", "=", "5.10"), ("p", "", "=", value)], note
        assert len({eq.grp_id for eq in eqs}) == 1

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("It was decisive (BF10 = 2.3e5).", ("BF10", "", "=", "2.3e5")),
            ("Genome-wide (P < 5 × 10−8).", ("P", "", "<", "5 × 10−8")),
            ("Genome-wide (P < 10⁻⁸).", ("P", "", "<", "10⁻⁸")),
            ("Counts were high (N = 2.5 × 103).", ("N", "", "=", "2.5 × 103")),
            ("The slope was b = 8.0E04.", ("b", "", "=", "8.0E04")),
        ],
    )
    def test_scientific_notation_of_other_statistics(self, text, expected):
        # BF10 and P are the broad pass's; "× 103" is 10³ flattened.
        assert _components(_extract(text)) == [expected]

    def test_mantissa_of_one_is_not_a_valid_looking_p(self):
        eqs = _extract("The effect was significant (t(28) = 5.10, p < 1e-10).")

        assert ("p", "", "<", "1e-10") in _components(eqs)
        assert ("p", "", "<", "1") not in _components(eqs)

    def test_bare_value_is_one_component_not_a_truncated_pair(self):
        assert _components(_extract("The effect was significant, p = 2.3e-5.")) == [
            ("p", "", "=", "2.3e-5")
        ]

    def test_single_digit_mantissa_is_not_dropped_as_a_small_integer(self):
        # "p = 5" alone looks like an index and is filtered; the whole value is not.
        assert _components(_extract("Genome-wide, p = 5 × 10−8 was the threshold.")) == [
            ("p", "", "=", "5 × 10−8")
        ]

    def test_latex_value_is_extracted_once(self):
        eqs = _extract("It held ($p = 2.3 \\times 10^{-5}$).")

        assert _components(eqs) == [("p", "", "=", "2.3 \\times 10^{-5}")]

    def test_export_locates_the_whole_printed_value(self):
        from bibr.export.spans import equation_pattern

        text = "The effect was significant (t(28) = 5.10, p = 2.3 × 10−5)."
        p = next(eq for eq in _extract(text) if eq.lhs == "p")
        match = equation_pattern(p.lhs, p.df, p.comp, p.rhs).search(text)

        assert match is not None
        assert match.group() == "p = 2.3 × 10−5"

    def test_decimal_commas(self):
        eqs = _extract("Mittelwert (M = 3,45, SD = 1,20, p = 0,05).")

        assert _components(eqs) == [
            ("M", "", "=", "3,45"),
            ("SD", "", "=", "1,20"),
            ("p", "", "=", "0,05"),
        ]

    def test_decimal_commas_outside_parentheses(self):
        eqs = _extract("Es zeigte sich t(28) = 2,45, p = 0,02.")

        assert _components(eqs) == [("t", "28", "=", "2,45"), ("p", "", "=", "0,02")]
        assert len({eq.grp_id for eq in eqs}) == 1

    def test_decimal_comma_with_four_fraction_digits(self):
        # "0,000" matched as a thousands group, dropping the last digit.
        assert _components(_extract("It held (p = 0,0001).")) == [("p", "", "=", "0,0001")]

    def test_latex_decimal_comma(self):
        assert _components(_extract("Der Effekt war signifikant ($p = 0{,}05$).")) == [
            ("p", "", "=", "0{,}05")
        ]

    def test_commas_that_are_not_decimal(self):
        # df pairs belong to the LHS, and APA value lists put a space after
        # the comma.
        assert _components(_extract("The ANOVA (F(1,23) = 4,5, p = .04) held.")) == [
            ("F", "1,23", "=", "4,5"),
            ("p", "", "=", ".04"),
        ]
        assert _components(_extract("Values (p = .03, .04) were both small.")) == [
            ("p", "", "=", ".03")
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "Levels (M = 1,2,3) were used.",
            "Levels (M = 1,2,...) were used.",
            "Levels (M = 1,2,…,K) were used.",
        ],
    )
    def test_comma_run_is_a_list_not_a_decimal(self, text):
        # Seen part by part inside parentheses, "M = 1,2" of "M = 1,2,…,K"
        # read as the decimal 1.2.
        assert _components(_extract(text)) == [("M", "", "=", "1")]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "Only the relatively small cluster 8 (n=304,3% of ACSs) corresponds to it.",
                [("n", "", "=", "304")],
            ),
            (
                "Sample sizes ranged from n = 8246 to n = 9380,37 and men from 13.6% to 46.6%.",
                [("n", "", "=", "8246"), ("n", "", "=", "9380")],
            ),
            (
                "(stimulated vs. unstimulated: 2.97 ± 0.48; n = 9,7 cells; p<0.05)",
                [("n", "", "=", "9"), ("p", "", "<", "0.05")],
            ),
            ("Indices (n = 1,2,3) were used.", [("n", "", "=", "1")]),
            ("Indices (k = 1,2,…,K) were used.", [("k", "", "=", "1")]),
            ("The cohort (N = 1,204) was large.", [("N", "", "=", "1,204")]),
            # "n = 9" is too small to be told from an index, and so is "n = 9,7"
            ("Recordings came from n = 9,7 cells.", []),
            # A 0 before the comma makes it a decimal, not a count
            ("We take k = 0,75 for the losses.", [("k", "", "=", "0,75")]),
        ],
    )
    def test_count_takes_no_decimal_comma(self, text, expected):
        # A comma after a count's digits lists group sizes or runs into a
        # percentage or a flattened citation number; it still groups thousands.
        assert _components(_extract(text)) == expected

    def test_ranges(self):
        assert _components(_extract("The measures were intercorrelated with r = .85–.94.")) == [
            ("r", "", "=", ".85–.94")
        ]
        assert _components(_extract("Fits were good (RMSEA = 0.08, 90% CI = 0.06–0.09).")) == [
            ("90% CI", "", "=", "0.06–0.09"),
            ("RMSEA", "", "=", "0.08"),
        ]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("It held (r = .85 – .94).", ("r", "", "=", ".85 – .94")),
            ("It held (95% CI = −2.84–−0.19).", ("95% CI", "", "=", "−2.84–−0.19")),
            ("It held (99.9% CI = 0.2–0.5).", ("99.9% CI", "", "=", "0.2–0.5")),
        ],
    )
    def test_spaced_negative_and_levelled_ranges(self, text, expected):
        assert _components(_extract(text)) == [expected]

    @pytest.mark.parametrize(
        "text",
        [
            "Since \\( \\eta \\leq 1/50 \\), \\( b \\leq 1/3 \\) is always satisfied.",
            "It held for (c^J + 13\\eta)M \\leq 14\\eta M in all cases.",
            "The condition (T_{ab} l^a l^b \\geq 0) holds.",
        ],
    )
    def test_value_that_runs_on_is_not_cut(self, text):
        # b ≤ 1 of "b ≤ 1/3", M ≤ 14 of "M ≤ 14ηM", and the superscript b of
        # "l^b" are no statistics.
        assert _extract(text) == []

    def test_value_before_its_error_is_whole(self):
        assert _components(_extract("The mean was high (M = 63.0\\pm 9.0).")) == [
            ("M", "", "=", "63.0")
        ]

    def test_time_is_not_a_value(self):
        assert _components(_extract("It was measured (t = 12:30, p = .05).")) == [
            ("p", "", "=", ".05")
        ]

    def test_count_before_a_slash_is_kept(self):
        # "12 of 20 cells" and "150 WT and 123 KO": the count is kept, cut at
        # the slash, rather than dropped as a fraction
        assert _components(_extract("It fired (C; n = 12/20 cells).")) == [("n", "", "=", "12")]
        assert _components(_extract("It fell (p = .01; n = 150/123 WT/KO).")) == [
            ("p", "", "=", ".01"),
            ("n", "", "=", "150"),
        ]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("(OR 1·45, 95% CI 1·10–1·91; p=0·008)", [("p", "", "=", "0·008")]),
            ("Risk rose (OR = 3·4; CI95 1·4–8·2; P 0·007).", [("OR", "", "=", "3·4")]),
            (
                "It held (cotyledons: r = 0·86, P < 10−5; leaves: r = 0·78, P < 10−4).",
                [
                    ("r", "", "=", "0·86"),
                    ("r", "", "=", "0·78"),
                    ("P", "", "<", "10−5"),
                    ("P", "", "<", "10−4"),
                ],
            ),
            ("It held (r = 0·10–0·20).", [("r", "", "=", "0·10–0·20")]),
            ("It held, P < 0·001.", [("P", "", "<", "0·001")]),
            # The dot multiplies a power of ten
            ("It held (p = 6·10−6).", [("p", "", "=", "6·10−6")]),
            ("It held (p = 6·10⁻⁶).", [("p", "", "=", "6·10⁻⁶")]),
        ],
    )
    def test_mid_dot_decimals(self, text, expected):
        # Lancet and Oxford journals print "0·008": it was cut to "p = 0"
        assert _components(_extract(text)) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Age 0.21 −0.05 0.47 β = 0.21 −0.05 0.47 .12", ("β", "", "=", "0.21")),
            ("It held (r = .45 −.12 .60).", ("r", "", "=", ".45")),
            # An en dash is a range's, spaced or not
            (
                "Monkey RE: η = 5.1×10 −4 –9.2 × 10 −4 held.",
                ("η", "", "=", "5.1×10 −4 –9.2 × 10 −4"),
            ),
        ],
    )
    def test_signed_number_after_a_space_is_not_a_range_end(self, text, expected):
        # A table flattened to text prints an estimate and its negative bound
        assert _components(_extract(text)) == [expected]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("It held (p<10 −10).", ("p", "", "<", "10 −10")),
            ("It held (p = 3.10 −9).", ("p", "", "=", "3.10 −9")),
            ("It held (P < 10− 3 compared to control).", ("P", "", "<", "10− 3")),
            ("with expectation values of E = 10 −5.", ("E", "", "=", "10 −5")),
        ],
    )
    def test_flattened_exponent_after_a_space(self, text, expected):
        # eLife HTML and PDF text layers print 10⁻¹⁰ as "10 −10"
        assert _components(_extract(text)) == [expected]


class TestStatisticNames:
    """A statistic's name is not cut at a Greek letter, superscript or Δ.

    ``η²p`` (partial eta squared) exported as a p-value, and ``ΔR²`` as R².
    """

    @pytest.mark.parametrize(
        "name",
        [
            "η²p",
            "ηp²",
            "ηp2",
            "η2p",
            "η2 p",
            "η 2 p",
            "η p 2",
            "η 2",
            "ηₚ²",
            "ω²p",
            "ω 2 p",
            "ω p 2",
            "ηG²",
        ],
    )
    def test_partial_eta_squared_is_not_a_p_value(self, name):
        eqs = _extract(f"A main effect, F(1, 40) = 5.2, p = .03, {name} = .12.")

        assert _components(eqs) == [
            ("F", "1, 40", "=", "5.2"),
            ("p", "", "=", ".03"),
            (name, "", "=", ".12"),
        ]
        assert len({eq.grp_id for eq in eqs}) == 1

    def test_partial_eta_squared_in_parentheses(self):
        eqs = _extract("A main effect (F(1, 40) = 5.2, p = .03, η²p = .12).")

        assert [eq.lhs for eq in eqs] == ["F", "p", "η²p"]

    def test_spaced_partial_eta_squared_from_a_pdf_text_layer(self):
        # Verbatim from a bibr export and eLife HTML: "η 2 p" is η²p.
        assert _components(_extract("η 2 p = 0.05.")) == [("η 2 p", "", "=", "0.05")]
        eqs = _extract(
            "a main effect of SNR (F(3,54) = 36.22, p<0.001, Huynh-Feldt corrected, "
            "η 2 p = 0.67), and of visual context"
        )
        assert _components(eqs) == [
            ("F", "3,54", "=", "36.22"),
            ("p", "", "<", "0.001"),
            ("η 2 p", "", "=", "0.67"),
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "A main effect, F(1, 40) = 5.2, p = .03, ε²p = .12.",
            "A main effect, F(1, 40) = 5.2, p = .03, ε 2 p = .12.",
            "A main effect, F(1, 40) = 5.2, p = .03, η p = .12.",
        ],
    )
    def test_subscript_of_another_effect_size_is_not_a_p_value(self, text):
        assert _components(_extract(text)) == [("F", "1, 40", "=", "5.2"), ("p", "", "=", ".03")]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # A Holm-adjusted p and membrane resistance R_m
            (
                "(WT: 36.1 pA; n = 12; BACHD: 45.6 pA; n = 11; p h = 0.7399; Figure 2A)",
                [("n", "", "=", "12"), ("n", "", "=", "11")],
            ),
            ("(R m = 2.04 Ωm 2, R a = 1.02 Ωm; n = 5)", [("n", "", "=", "5")]),
        ],
    )
    def test_spaced_subscript_is_not_a_statistic(self, text, expected):
        # An HTML or PDF text layer prints p_h as "p h": h alone is a
        # Kruskal-Wallis H to a consumer.
        assert _components(_extract(f"It held {text}")) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Spearman's r_s, Cohen's d_z and d_av
            (
                "(Spearman’s Rho r s = 0.42; p=0.1)",
                [("r s", "", "=", "0.42"), ("p", "", "=", "0.1")],
            ),
            (
                "Spearman’s Rho r s = 0.42, p=0.1.",
                [("r s", "", "=", "0.42"), ("p", "", "=", "0.1")],
            ),
            ("(rs = 0.56, P < .0001)", [("rs", "", "=", "0.56"), ("P", "", "<", ".0001")]),
            ("(d z = 0.5, p = .03)", [("d z", "", "=", "0.5"), ("p", "", "=", ".03")]),
            ("the effect was d av = 0.41.", [("d av", "", "=", "0.41")]),
            ("the effect was Cohen’s d z = 0.41.", [("Cohen’s d z", "", "=", "0.41")]),
            # A tight "dz" is an integral's differential as often
            ("e^{-z^2} \\, dz = .29782.", []),
            ("allele (rs1467568 = 3) used", []),
        ],
    )
    def test_spaced_subscript_of_a_known_statistic_is_its_name(self, text, expected):
        assert _components(_extract(f"It held {text}")) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "Association was strong (OR = 1.2, P-value = 2.3 × 10−5).",
                [("OR", "", "=", "1.2"), ("P-value", "", "=", "2.3 × 10−5")],
            ),
            (
                "audit yield (82.2% vs. 33% on average, p-value = 3.8∙10−35) or",
                [("p-value", "", "=", "3.8∙10−35")],
            ),
            (
                "at the inpatient center (51% vs. 24%; p-value<0.001).",
                [("p-value", "", "<", "0.001")],
            ),
            ("The overall p value = 0.043 held.", [("p value", "", "=", "0.043")]),
            ("genes changing with a p -value <10 –5 were kept.", [("p -value", "", "<", "10 –5")]),
            ("Differences with P‐values < 0.05 were significant.", [("P‐values", "", "<", "0.05")]),
            ("The p-value-adjusted scores were used.", []),
        ],
    )
    def test_p_value_by_its_full_name(self, text, expected):
        eqs = _extract(text)

        assert _components(eqs) == expected
        assert len({eq.grp_id for eq in eqs}) <= 1

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "eggs (χ 2 = 148.9, df = 2, p < 0.001).",
                [("χ 2", "", "=", "148.9"), ("p", "", "<", "0.001")],
            ),
            (
                "by explanation condition (χ 2 (4) = 2.09, p = 0.72).",
                [("χ 2", "4", "=", "2.09"), ("p", "", "=", "0.72")],
            ),
            # Its df parenthesis is no group of its own with a bare N
            (
                "It held, χ 2 (2, N = 150) = 9.87, p = .007.",
                [("χ 2", "2, N = 150", "=", "9.87"), ("p", "", "=", ".007")],
            ),
            (
                "vs 7%, 7/101; X 2 = 4.73, p=0.03).",
                [("X 2", "", "=", "4.73"), ("p", "", "=", "0.03")],
            ),
            (
                "The fit improved, Δχ 2 (1) = 5.2, p = .02.",
                [("Δχ 2", "1", "=", "5.2"), ("p", "", "=", ".02")],
            ),
        ],
    )
    def test_spaced_chi_square(self, text, expected):
        # PDF text layers and eLife HTML print χ² as "χ 2": the χ² was lost and
        # its p exported alone.
        eqs = _extract(text)

        assert _components(eqs) == expected
        assert len({eq.grp_id for eq in eqs}) == 1

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # group labels and footnote marks before a statistic
            (
                "(T-KO n = 33, CpxI-3R n = 49, WT n = 50)",
                [("n", "", "=", "33"), ("n", "", "=", "49"), ("n", "", "=", "50")],
            ),
            (
                "MS versus P P < 0.05 and LS P < 0.01",
                [("P", "", "<", "0.05"), ("P", "", "<", "0.01")],
            ),
            ("Se m:Se d p<0.001", [("p", "", "<", "0.001")]),
            (
                "(n = 9–10 mice) *p < 0.001 versus DCS, ^p < 0.05 versus CDP",
                [("n", "", "=", "9–10"), ("p", "", "<", "0.001"), ("p", "", "<", "0.05")],
            ),
        ],
    )
    def test_letter_after_a_label_or_mark_is_a_statistic(self, text, expected):
        assert _components(_extract(text)) == expected

    def test_delta_prefix(self):
        assert _components(_extract("(ΔR² = .05, F(1, 96) = 6.2, p = .01)")) == [
            ("ΔR²", "", "=", ".05"),
            ("F", "1, 96", "=", "6.2"),
            ("p", "", "=", ".01"),
        ]
        assert _components(_extract("The fit improved, Δχ²(1) = 5.2, p = .02.")) == [
            ("Δχ²", "1", "=", "5.2"),
            ("p", "", "=", ".02"),
        ]
        assert _components(_extract("Model B was preferred (ΔAIC = 4.1).")) == [
            ("ΔAIC", "", "=", "4.1")
        ]
        assert _components(_extract("The fit improved, Δχ²(1, N = 100) = 3.84, p = .05.")) == [
            ("Δχ²", "1, N = 100", "=", "3.84"),
            ("p", "", "=", ".05"),
        ]

    def test_increment_sign_prefix(self):
        # ∆ (U+2206), as PDF text layers and Word print Δ
        assert _components(_extract("(∆R² = .05, F(1, 96) = 6.2, p = .01)")) == [
            ("∆R²", "", "=", ".05"),
            ("F", "1, 96", "=", "6.2"),
            ("p", "", "=", ".01"),
        ]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("pre-post pairing with Δt = −20 ms", [("Δt", "", "=", "−20")]),
            ("a displacement vector Δp = 0.35 was found", [("Δp", "", "=", "0.35")]),
            ("invariance holds when ΔCFI ≤ 0.01", [("ΔCFI", "", "≤", "0.01")]),
            ("the change ∆z = 0.65 held", [("∆z", "", "=", "0.65")]),
        ],
    )
    def test_delta_stays_on_the_name(self, text, expected):
        # "Δt = −20 ms" is a time difference, not a t statistic.
        assert _components(_extract(text)) == expected

    def test_greek_subscripted_name_is_not_a_bare_statistic(self):
        # τp is a time constant, not a p-value; τd is not Cohen's d.
        assert _extract("The barostat kept pressure at 1 bar with τp = 1.0 ps.") == []
        assert _extract("We set τg = 1 ms and τd = 20 ms.") == []

    def test_name_flush_against_cjk_text_still_matches(self):
        assert _components(_extract("结果显著t(28) = 2.1，p < .05。")) == [
            ("t", "28", "=", "2.1"),
            ("p", "", "<", ".05"),
        ]

    def test_df_owner_must_be_a_whole_name(self):
        # "accident (n = 1)" is not the df of a t: the group used to be skipped
        # and its small n then filtered.
        eqs = _extract("Seizures followed a motor vehicle accident (n = 1).")

        assert _components(eqs) == [("n", "", "=", "1")]


class TestTextLayerLineBreaks:
    """pdfium ends each line of a PDF text layer with "\\r\\n", also the lines
    of the stacked sub- and superscripts of partial eta squared.

    The sentences are from PDF text layers, before late clean-up.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Frontiers: the p of η²p was a p-value
            (
                "was significant, F(1,56) = 6.85, p = 0.011, η\r\n2\r\np = 0.11, the risk\r\nrate",
                [("F", "1,56", "=", "6.85"), ("p", "", "=", "0.011"), ("η 2 p", "", "=", "0.11")],
            ),
            # SAGE: η²p was not read at all
            (
                "audiovisual cues, F(1, \r\n19) = 29.57, p < .001, ηp\r\n2 = .61.",
                [("F", "1, 19", "=", "29.57"), ("p", "", "<", ".001"), ("ηp 2", "", "=", ".61")],
            ),
            (
                "direction, F(1, 19) < 0.01, p = .948, ηp\r\n2 <\r\n.01, and the",
                [("F", "1, 19", "<", "0.01"), ("p", "", "=", ".948"), ("ηp 2", "", "<", ".01")],
            ),
            # Not from a text layer: a lone carriage return is a line break too
            (
                "It held, p = .01, η\r2\rp = .11.",
                [("p", "", "=", ".01"), ("η 2 p", "", "=", ".11")],
            ),
        ],
    )
    def test_partial_eta_squared_printed_over_lines(self, text, expected):
        eqs = _extract(text)

        assert _components(eqs) == expected
        assert len({eq.grp_id for eq in eqs}) == 1

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "for angry stimuli, t(196) = 3.73, p < .001, Cohen’s \r\ndz = 0.27.",
                [
                    ("t", "196", "=", "3.73"),
                    ("p", "", "<", ".001"),
                    ("Cohen’s dz", "", "=", "0.27"),
                ],
            ),
            (
                "estimate = 0.19), 95% \r\nCI = [0.16, 0.21], d = 0.59, t(3160) = 13.0",
                [
                    ("95% CI", "", "=", "[0.16, 0.21]"),
                    ("d", "", "=", "0.59"),
                    ("t", "3160", "=", "13.0"),
                ],
            ),
            (
                "(Fig. 1a), slope = 0.36, 95% CI = [0.30, \r\n0.42], t(32.3) = 12.4",
                [("95% CI", "", "=", "[0.30, 0.42]"), ("t", "32.3", "=", "12.4")],
            ),
            (
                "failed to reach significance, F(1, \r\n19) = 0.34, p = .567, ηp\r\n2 = .02.",
                [("F", "1, 19", "=", "0.34"), ("p", "", "=", ".567"), ("ηp 2", "", "=", ".02")],
            ),
        ],
    )
    def test_names_and_values_are_exported_on_one_line(self, text, expected):
        # As late clean-up prints the sentence: a consumer folding "Cohen’s dz"
        # missed "Cohen’s \r\ndz".
        assert _components(_extract(text)) == expected

    def test_export_locates_them_after_late_clean_up(self):
        from bibr.export.spans import SpanLocator, equation_span
        from bibr.input.consolidate_text import clean_text_content_late

        raw = "F(1,56) = 6.85, p = 0.011, η\r\n2\r\np = 0.11, and Cohen’s \r\ndz = 0.27."
        text = clean_text_content_late(raw)
        locator = SpanLocator({1: text}, shared=False)
        spans = [equation_span(locator, eq) for eq in _extract(raw)]

        assert [text[a:b] for a, b in spans] == [
            "F(1,56) = 6.85",
            "p = 0.011",
            "η 2 p = 0.11",
            "Cohen’s dz = 0.27",
        ]

    async def test_llm_fallback_components_are_exported_on_one_line(self):
        from bibr.paper_contents import CanonicalSection, PaperEquation

        sections = [
            PaperSection(
                section_id=1,
                header="Results",
                level=1,
                parent_section_id=None,
                section_type=CanonicalSection.RESULTS,
            )
        ]
        sents = [_make_sentence(10, "An odd layout (slope: 0.36, 95% \r\nCI: [0.30, \r\n0.42]).")]

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                return [
                    PaperEquation(
                        text_id=10, grp_id=0, lhs="95% \r\nCI", comp="=", rhs="[0.30, \r\n0.42]"
                    )
                ]

        eqs = await EquationExtractor().extract_with_llm_fallback(
            sents, sections, llm_client=FakeLLM()
        )

        assert _components(eqs) == [("95% CI", "", "=", "[0.30, 0.42]")]


class TestSharedParentheses:
    """Statistics sharing parentheses with a recognised one are not dropped.

    The structured pass recorded the whole parenthesis as extracted, so the
    broad pass skipped the parts it had not recognised.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("(r(98) = .32, p = .001)", [("r", "98", "=", ".32"), ("p", "", "=", ".001")]),
            ("(Z = 2.31, p = .02)", [("Z", "", "=", "2.31"), ("p", "", "=", ".02")]),
            ("(H(2) = 8.10, p = .02)", [("H", "2", "=", "8.10"), ("p", "", "=", ".02")]),
            ("(χ² = 3.84, p = .05)", [("χ²", "", "=", "3.84"), ("p", "", "=", ".05")]),
            ("(p < .001, BF10 = 12.3)", [("p", "", "<", ".001"), ("BF10", "", "=", "12.3")]),
            ("(ICC = 0.85, p < .001)", [("p", "", "<", ".001"), ("ICC", "", "=", "0.85")]),
            ("(OR = 1.2, P < 0.001)", [("OR", "", "=", "1.2"), ("P", "", "<", "0.001")]),
            (
                "(t(28) = 2.1, p = .04, g = 0.45)",
                [("t", "28", "=", "2.1"), ("p", "", "=", ".04"), ("g", "", "=", "0.45")],
            ),
        ],
    )
    def test_every_statistic_is_kept_in_the_group(self, text, expected):
        eqs = _extract(f"It differed {text}.")

        assert _components(eqs) == expected
        assert len({eq.grp_id for eq in eqs}) == 1

    def test_correlation_with_df_outside_parentheses_groups_with_its_p(self):
        eqs = _extract("Correlation was positive, r(98) = .32, p = .001.")

        assert _components(eqs) == [("r", "98", "=", ".32"), ("p", "", "=", ".001")]
        assert len({eq.grp_id for eq in eqs}) == 1

    def test_second_component_of_one_part_is_kept(self):
        eqs = _extract("It held (p = 0.85 in 1 mM TEA and p = 0.95 in 5 mM TEA).")

        assert _components(eqs) == [("p", "", "=", "0.85"), ("p", "", "=", "0.95")]
        assert len({eq.grp_id for eq in eqs}) == 1

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "Error bars (n = 3 analytical replicates and n = 4 experimental replicates)",
                [("n", "", "=", "3"), ("n", "", "=", "4")],
            ),
            (
                "Summary data (n = 6 for control and n = 7 for βARK-CT, ***p=0.00032).",
                [("n", "", "=", "6"), ("n", "", "=", "7"), ("p", "", "=", "0.00032")],
            ),
        ],
    )
    def test_every_count_of_one_part_is_kept_in_order(self, text, expected):
        # Read as a bare statistic, a second small count looked like "i = 1"
        eqs = _extract(text)

        assert _components(eqs) == expected
        assert len({eq.grp_id for eq in eqs}) == 1

    def test_repeated_value_is_not_merged(self):
        eqs = _extract("Both rose (p < 0.001 and p < 0.001), respectively.")

        assert _components(eqs) == [("p", "", "<", "0.001"), ("p", "", "<", "0.001")]

    def test_broad_match_reaching_into_a_group_is_not_taken(self):
        eqs = _extract("It had minimal effect on the PPR (0.97 ± 0.05; t(d.f.16)=1.58, p=0.251).")

        assert all("PPR" not in eq.lhs for eq in eqs)


class TestGroupingAcrossPasses:
    """Components printed together share a group whichever pass read them."""

    def _groups(self, text: str) -> list[list[str]]:
        groups: dict[int, list[str]] = {}
        for eq in _extract(text):
            groups.setdefault(eq.grp_id, []).append(eq.lhs)
        return list(groups.values())

    def test_z_groups_with_its_broad_pass_p(self):
        # Pass 2 reads z, Z and k; pass 4 reads x and P (FWE-corrected).
        groups = self._groups(
            "(A) Common activation in the left amygdala, x = −20, y = 2, z = −16; Z = 2.84; "
            "P (FWE-corrected) = 0.035; k = 114 voxel."
        )

        assert groups == [["x"], ["z", "Z", "P", "k"]]

    def test_confidence_interval_groups_with_its_fit_indices(self):
        groups = self._groups(
            "χ2(96) = 179.24, p = .00, CFI = 0.91; RMSEA = 0.08, 90% CI = 0.06–0.09."
        )

        assert groups == [["χ2", "p", "CFI", "RMSEA", "90% CI"]]

    @pytest.mark.parametrize(
        ("text", "lhs"),
        [
            ("Groups differed ($\\chi^{2}(1) = 5.2$, $p = .02$).", ["p", "\\chi^{2}"]),
            (
                "A main effect ($F(1, 40) = 5.2$, $p = .03$, $\\eta_{p}^{2} = .12$).",
                ["F", "p", "\\eta_{p}^{2}"],
            ),
            ("It held (t(28) = 2.1, $\\chi^2(1) = 3.84$).", ["t", "\\chi^2"]),
            ("The fit differed, $\\chi^2(1) = 3.84$, $p = .05$.", ["\\chi^2", "p"]),
        ],
    )
    def test_latex_statistic_groups_with_the_statistics_printed_with_it(self, text, lhs):
        # Only the LaTeX pass reads \chi and \eta; OCR wraps each in $...$.
        assert self._groups(text) == [lhs]


class TestDuplicates:
    """Each printed expression is exported once, in one group."""

    def test_inline_latex_statistics(self):
        eqs = _extract("The difference was significant ($t(28) = 2.10$, $p < .05$).")

        assert _components(eqs) == [("t", "28", "=", "2.10"), ("p", "", "<", ".05")]
        assert len({eq.grp_id for eq in eqs}) == 1

    def test_inline_latex_statistics_outside_parentheses_share_a_group(self):
        eqs = _extract("Effects were reliable, $d = 0.45$, $p = .003$.")

        assert _components(eqs) == [("d", "", "=", "0.45"), ("p", "", "=", ".003")]
        assert len({eq.grp_id for eq in eqs}) == 1

    def test_cohens_d_in_a_group(self):
        eqs = _extract("The effect was large (t(28) = 2.1, p = .04, Cohen’s d = 0.45).")

        assert _components(eqs) == [
            ("t", "28", "=", "2.1"),
            ("p", "", "=", ".04"),
            ("Cohen’s d", "", "=", "0.45"),
        ]

    @pytest.mark.parametrize(
        ("latex", "comp"),
        [
            ("\\leq", "≤"),
            ("\\le", "≤"),
            ("\\leqslant", "≤"),
            ("\\geq", "≥"),
            ("\\geqslant", "≥"),
            ("\\neq", "≠"),
            ("\\approx", "≈"),
            ("\\sim", "~"),
        ],
    )
    def test_latex_comparators(self, latex, comp):
        assert _components(_extract(f"It held ($p {latex} 0.05$).")) == [("p", "", comp, "0.05")]

    def test_latex_comparator_printed_as_plain_text_in_a_group(self):
        eqs = _extract("It held (t(28) = 2.1, p \\leq .05).")

        assert _components(eqs) == [("t", "28", "=", "2.1"), ("p", "", "≤", ".05")]
        assert len({eq.grp_id for eq in eqs}) == 1

    def test_slanted_comparators(self):
        assert _components(_extract("The effect held (t(28) = 2.10, p ⩽ 0.05).")) == [
            ("t", "28", "=", "2.10"),
            ("p", "", "≤", "0.05"),
        ]
        # A name only the structured passes read
        assert _components(_extract("The fit held (Δχ²(1) ⩾ 3.84).")) == [("Δχ²", "1", "≥", "3.84")]

    def test_latex_command_starting_with_a_relation_is_not_one(self):
        # \left is not \le, nor \neg \ne
        display = PaperSentence(
            text_id=1,
            text="\\left( \\frac{a}{b} \\right) = 2",
            section_id=1,
            paragraph_id=1,
            is_display_formula=True,
        )
        eqs = EquationExtractor().extract_from_sentences([display], _make_sections())

        assert _components(eqs) == [("\\left( \\frac{a}{b} \\right)", "", "=", "2")]
        assert _components(_extract("It held ($\\neg A = 1$).")) == [("\\neg A", "", "=", "1")]

    def test_latex_statistic_splits_its_df(self):
        assert _components(_extract("The model fit ($\\chi^2(1) = 3.84$).")) == [
            ("\\chi^2", "1", "=", "3.84")
        ]
        # Its df argument holds no N of its own
        assert _components(_extract("It held ($\\chi^{2}(1, N = 100) = 3.84$).")) == [
            ("\\chi^{2}", "1, N = 100", "=", "3.84")
        ]

    def test_display_formula_the_structured_passes_decomposed_is_not_repeated(self):
        display = PaperSentence(
            text_id=1,
            text="t(28) = 2.10, p < .05.",
            section_id=1,
            paragraph_id=1,
            is_display_formula=True,
        )
        eqs = EquationExtractor().extract_from_sentences([display], _make_sections())

        assert _components(eqs) == [("t", "28", "=", "2.10"), ("p", "", "<", ".05")]

    def test_operator_in_an_unclosed_parenthesis_still_splits_a_formula(self):
        display = PaperSentence(
            text_id=1,
            text="f(1-\\exp(-x) = y",
            section_id=1,
            paragraph_id=1,
            is_display_formula=True,
        )
        eqs = EquationExtractor().extract_from_sentences([display], _make_sections())

        assert _components(eqs) == [("f(1-\\exp(-x)", "", "=", "y")]

    def test_small_integer_in_a_display_formula_is_not_a_statistic(self):
        # "z \le - 1" of a piecewise definition is a bound, as "z = -1" is
        text = "h(z) = \\left\\{ 0 \\quad {\\text{if }}z \\le - 1 \\right."
        display = PaperSentence(
            text_id=1, text=text, section_id=1, paragraph_id=1, is_display_formula=True
        )
        eqs = EquationExtractor().extract_from_sentences([display], _make_sections())

        assert [eq.lhs for eq in eqs] == ["h(z)"]

    @pytest.mark.parametrize(
        ("text", "lhs"),
        [
            ("y(n)=median(x(n-k:n+k),k=(l-1)/2)", "y(n)"),
            ("H(n)=-\\sum p(\\pi)\\log p(\\pi)", "H(n)"),
            ("P_R^{\\prime\\prime}(1) = 0.25", "P_R^{\\prime\\prime}(1)"),
        ],
    )
    def test_latex_function_argument_is_not_a_df(self, text, lhs):
        display = PaperSentence(
            text_id=1, text=text, section_id=1, paragraph_id=1, is_display_formula=True
        )
        eqs = EquationExtractor().extract_from_sentences([display], _make_sections())

        assert [(eq.lhs, eq.df) for eq in eqs] == [(lhs, "")]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("It held ($ICC = 0.85$).", [("ICC", "", "=", "0.85")]),
            ("It held ($BF10 = 12.3$).", [("BF10", "", "=", "12.3")]),
            (
                "It held ($t(28) = 2.10, p < .05$).",
                [("t", "28", "=", "2.10"), ("p", "", "<", ".05")],
            ),
            ("The effect was reliable ($p < 0.001^{***}$).", [("p", "", "<", "0.001")]),
            ("The effect was reliable, $p = 0.03^{a}$.", [("p", "", "=", "0.03")]),
            ("The effect was reliable ($p < .001\\,$).", [("p", "", "<", ".001")]),
            ("It held ($d = 0.45 \\pm 0.10$).", [("d", "", "=", "0.45")]),
            (
                "Effects were reliable ($t = 2.1, p = .03$).",
                [("p", "", "=", ".03"), ("t", "", "=", "2.1")],
            ),
        ],
    )
    def test_inline_latex_is_not_read_again(self, text, expected):
        # The LaTeX pass reads only what another pass did not ("^{***}" holds
        # no relation); the broad pass skips the LaTeX pass's.
        eqs = _extract(text)

        assert _components(eqs) == expected
        assert len({eq.grp_id for eq in eqs}) == 1

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "It held ($\\chi^2(1) = 3.84, p = .05$).",
                [("p", "", "=", ".05"), ("\\chi^2", "1", "=", "3.84")],
            ),
            (
                "The effect was large ($\\eta_p^2 = .12, p = .03$).",
                [("p", "", "=", ".03"), ("\\eta_p^2", "", "=", ".12")],
            ),
            (
                "Evidence favoured H1, $BF_{10} = 12.3, p < .001$.",
                [("BF_{10}", "", "=", "12.3"), ("p", "", "<", ".001")],
            ),
            (
                "A main effect was found, $F(1, 40) = 5.2, p = .03, \\eta_p^2 = .12$.",
                [("F", "1, 40", "=", "5.2"), ("p", "", "=", ".03"), ("\\eta_p^2", "", "=", ".12")],
            ),
            (
                "It held ($p = .03, \\chi^2 = 3.84$).",
                [("p", "", "=", ".03"), ("\\chi^2", "", "=", "3.84")],
            ),
        ],
    )
    def test_latex_statistic_sharing_a_formula_with_its_p(self, text, expected):
        # The structured passes read the p; the LaTeX pass reads the rest of
        # the formula rather than skipping it, and no pass reads \chi or \eta.
        eqs = _extract(text)

        assert _components(eqs) == expected
        assert len({eq.grp_id for eq in eqs}) == 1

    def test_latex_relation_inside_parentheses_is_an_argument(self):
        eqs = EquationExtractor().extract_from_sentences(
            [
                PaperSentence(
                    text_id=1,
                    text="P(X \\geq x) = 1 - F(x)",
                    section_id=1,
                    paragraph_id=1,
                    is_display_formula=True,
                )
            ],
            _make_sections(),
        )

        assert _components(eqs) == [("P(X \\geq x)", "", "=", "1 - F(x)")]

    def test_currency_is_not_inline_math(self):
        eqs = _extract(
            "Costs rose from US$26.3 billion for those with CD4 <200 to US$42.5 billion."
        )

        assert _components(eqs) == [("CD4", "", "<", "200")]
        # A command between the amounts does not make them math: the closing
        # "$" starts the second amount.
        eqs = _extract(
            "Costs rose from US$26.3 billion (a 12\\% rise) for those with CD4 <200 "
            "to US$42.5 billion."
        )
        assert _components(eqs) == [("CD4", "", "<", "200")]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "A $2 \\times 2$ design with $\\eta_{p}^{2} = 0.12$ overall.",
                ("\\eta_{p}^{2}", "", "=", "0.12"),
            ),
            ("Values $0.5$ and $R^{2} = 0.45$ were found.", ("R^{2}", "", "=", "0.45")),
            (
                "The difference was reliable, $95 \\% \\mathrm{CI} = [0.12, 0.45]$.",
                ("95 \\% \\mathrm{CI}", "", "=", "[0.12, 0.45]"),
            ),
            ("It cost $5 and yielded $\\eta^2 = .1$.", ("\\eta^2", "", "=", ".1")),
        ],
    )
    def test_math_starting_with_a_digit_is_not_currency(self, text, expected):
        # A command or a lone number makes "$2 ..." math, and its closing "$"
        # must not open the next formula.
        assert _components(_extract(text)) == [expected]


class TestGroupIds:
    def test_filtered_group_does_not_use_up_an_id(self):
        sentences = [
            _make_sentence(1, "Each group had n = 5 animals."),
            _make_sentence(2, "It held (t(10) = 2.1, p = .05)."),
        ]
        eqs = EquationExtractor().extract_from_sentences(sentences, _make_sections())

        assert {eq.grp_id for eq in eqs} == {1}

    def test_broad_match_never_starts_inside_a_token(self):
        # Starting at every character of a long token made the broad pass
        # quadratic in its length; such a start never adds a match.
        from bibr.extract.equation_extractor import _BROAD_EQUATION_RE

        assert _BROAD_EQUATION_RE.search("xyzd = 5", 1) is None
        assert _BROAD_EQUATION_RE.search("xyzd = 5").group(1) == "xyzd"


class TestLlmFallbackGrouping:
    """M7: LLM-fallback components from one sentence must share a grp_id.

    The regex path groups components of one statistical statement; the LLM
    fallback gave every component its own grp_id, isolating exactly the
    t/p/d trios Metacheck needs grouped.
    """

    async def test_components_of_one_sentence_share_grp_id(self):
        from bibr.paper_contents import CanonicalSection, PaperEquation

        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Results",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.RESULTS,
            ),
        ]
        # Parenthesized digits (LLM-fallback candidates) but no regex hits.
        sents = [
            _make_sentence(10, "Weird stat layout (t: 3.42, p: .003, d: 0.45) regex misses."),
            _make_sentence(11, "Another odd one (p: .05; 5 vs 7) the regex misses."),
        ]

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                return [
                    PaperEquation(text_id=10, grp_id=0, lhs="t", df="28", comp="=", rhs="3.42"),
                    PaperEquation(text_id=10, grp_id=0, lhs="p", df="", comp="=", rhs=".003"),
                    PaperEquation(text_id=10, grp_id=0, lhs="d", df="", comp="=", rhs="0.45"),
                    PaperEquation(text_id=11, grp_id=0, lhs="p", df="", comp="<", rhs=".05"),
                ]

        extractor = EquationExtractor()
        eqs = await extractor.extract_with_llm_fallback(sents, sections, llm_client=FakeLLM())

        groups_by_text: dict[int, set[int]] = {}
        for eq in eqs:
            groups_by_text.setdefault(eq.text_id, set()).add(eq.grp_id)
        # One group per sentence: the t/p/d trio stays together…
        assert len(groups_by_text[10]) == 1
        assert len(groups_by_text[11]) == 1
        # …and different sentences get different groups.
        assert groups_by_text[10] != groups_by_text[11]

    async def test_groups_continue_after_regex_results_handed_in(self):
        """Regex results from another extractor keep their grp_ids to themselves."""
        from bibr.paper_contents import CanonicalSection, PaperEquation

        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Results",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.RESULTS,
            ),
        ]
        sents = [
            _make_sentence(9, "The effect was reliable (t(28) = 3.42)."),
            _make_sentence(10, "Weird stat layout (t: 3.42, p: .003) regex misses."),
        ]
        regex_equations = EquationExtractor().extract_from_sentences(sents, sections)

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                return [PaperEquation(text_id=10, grp_id=0, lhs="p", df="", comp="=", rhs=".003")]

        eqs = await EquationExtractor().extract_with_llm_fallback(
            sents, sections, llm_client=FakeLLM(), regex_equations=regex_equations
        )

        assert [(eq.text_id, eq.grp_id) for eq in eqs] == [(9, 1), (10, 2)]


class TestLlmFallbackFilterDedupe:
    """LLM equation fallback must drop empty components and dedupe records."""

    async def test_drops_fully_empty_components(self):
        from bibr.paper_contents import CanonicalSection, PaperEquation

        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Results",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.RESULTS,
            ),
        ]
        sents = [_make_sentence(10, "Weird stat layout (t: 3.42, numbers 12 and 34) regex misses.")]

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                return [
                    PaperEquation(text_id=10, grp_id=0, lhs="", df="", comp="", rhs=""),
                    PaperEquation(text_id=10, grp_id=0, lhs=None, df="", comp=None, rhs="  "),
                    PaperEquation(text_id=10, grp_id=0, lhs="t", df="28", comp="=", rhs="3.42"),
                ]

        extractor = EquationExtractor()
        eqs = await extractor.extract_with_llm_fallback(sents, sections, llm_client=FakeLLM())

        assert len(eqs) == 1
        assert eqs[0].lhs == "t"

    async def test_dedupes_within_llm_batch(self):
        from bibr.paper_contents import CanonicalSection, PaperEquation

        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Results",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.RESULTS,
            ),
        ]
        sents = [_make_sentence(10, "Weird stat layout (t: 3.42, numbers 12 and 34) regex misses.")]

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                return [
                    PaperEquation(text_id=10, grp_id=0, lhs="t", df="28", comp="=", rhs="3.42"),
                    PaperEquation(text_id=10, grp_id=0, lhs="t", df="99", comp="=", rhs="3.42"),
                ]

        extractor = EquationExtractor()
        eqs = await extractor.extract_with_llm_fallback(sents, sections, llm_client=FakeLLM())

        assert len(eqs) == 1
        assert eqs[0].df == "28"

    async def test_dedupes_across_llm_batches(self):
        """Duplicate (text_id, lhs, comp, rhs) keys are dropped across
        separate LLM batches too — mirrors the regex passes' cumulative
        existing_keys approach rather than only deduping per-batch."""
        from bibr.paper_contents import CanonicalSection, PaperEquation

        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Results",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.RESULTS,
            ),
        ]
        # 11 candidate sentences force a second LLM batch (batch_size=10).
        sents = [
            _make_sentence(
                i, f"Weird stat layout (numbers {i} and {i + 1}; value 1.0) regex misses."
            )
            for i in range(1, 12)
        ]

        call_count = 0

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First batch: text_ids 1..10, one equation each.
                    return [
                        PaperEquation(text_id=tid, grp_id=0, lhs="t", df="", comp="=", rhs="1.0")
                        for tid, _text in batch
                    ]
                # Second batch (text_id 11): a legitimate new equation plus a
                # bogus repeat of the text_id=5 record from the first batch.
                return [
                    PaperEquation(text_id=11, grp_id=0, lhs="t", df="", comp="=", rhs="1.0"),
                    PaperEquation(text_id=5, grp_id=0, lhs="t", df="", comp="=", rhs="1.0"),
                ]

        extractor = EquationExtractor()
        eqs = await extractor.extract_with_llm_fallback(sents, sections, llm_client=FakeLLM())

        assert call_count == 2
        assert len(eqs) == 11
        assert sum(1 for eq in eqs if eq.text_id == 5) == 1

    async def test_batches_start_concurrently_and_merge_in_source_order(self):
        from bibr.paper_contents import CanonicalSection, PaperEquation

        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Results",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.RESULTS,
            ),
        ]
        sents = [
            _make_sentence(
                i, f"Weird stat layout (numbers {i} and {i + 1}; value 1.0) regex misses."
            )
            for i in range(1, 21)
        ]
        started = []
        both_started = asyncio.Event()
        release = asyncio.Event()

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                started.append(batch[0][0])
                if len(started) == 2:
                    both_started.set()
                await release.wait()
                text_id = batch[0][0]
                return [PaperEquation(text_id=text_id, grp_id=0, lhs="v", comp="=", rhs="1.0")]

        task = asyncio.create_task(
            EquationExtractor().extract_with_llm_fallback(sents, sections, llm_client=FakeLLM())
        )
        await asyncio.wait_for(both_started.wait(), timeout=0.2)
        release.set()
        equations = await task

        assert started == [1, 11]
        assert [equation.text_id for equation in equations] == [1, 11]

    async def test_display_formula_sentences_skip_llm_fallback(self):
        """Display-formula sentences are already captured verbatim in the
        exported ``formatted`` field. Re-decomposing their LaTeX via the LLM
        fallback only duplicates existing content, so they must not be sent."""
        from bibr.paper_contents import CanonicalSection, PaperEquation

        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Results",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.RESULTS,
            ),
        ]
        # A display formula (bare LaTeX, parenthesized digits, regex misses it)
        # and a genuine prose candidate the fallback SHOULD still receive.
        display = PaperSentence(
            text_id=10,
            text=r"J(y)\approx\left[E\left\{G(y)\right\}\right]^{2} \tag{3}",
            section_id=1,
            paragraph_id=1,
            is_display_formula=True,
        )
        prose = _make_sentence(11, "The KMO measure was 0.764 (>0.60) regex misses.")

        seen_text_ids: list[int] = []

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                seen_text_ids.extend(text_id for text_id, _ in batch)
                return [PaperEquation(text_id=11, grp_id=0, lhs="KMO", comp=">", rhs="0.60")]

        await EquationExtractor().extract_with_llm_fallback(
            [display, prose], sections, llm_client=FakeLLM()
        )

        # The display formula never reaches the LLM; the prose candidate does.
        assert 10 not in seen_text_ids
        assert seen_text_ids == [11]

    async def test_drops_rhs_hallucinated_from_participant_label(self):
        from bibr.paper_contents import CanonicalSection, PaperEquation

        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Results",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.RESULTS,
            ),
        ]
        sents = [
            _make_sentence(
                243,
                "One participant scored relatively high on both behavioral and cognitive "
                "engagement (both 43%, p. 5).",
            )
        ]

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                return [PaperEquation(text_id=243, grp_id=0, lhs="p", df="", comp="=", rhs=".003")]

        extractor = EquationExtractor()
        eqs = await extractor.extract_with_llm_fallback(sents, sections, llm_client=FakeLLM())

        assert eqs == []


class TestLlmFallbackStatsGate:
    """Opt-in ``min_regex_stats`` gate skips the fallback on low-stat papers."""

    def _sections(self):
        from bibr.paper_contents import CanonicalSection

        return [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Results",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.RESULTS,
            ),
        ]

    async def test_gate_skips_fallback_when_regex_finds_too_few_stats(self):
        from bibr.paper_contents import PaperEquation

        # A parenthesized-digit candidate the regex misses, in a paper whose
        # regex pass produced zero statistical components.
        sents = [_make_sentence(10, "Odd layout (KMO measure of 0.764, adequacy) regex misses.")]

        called = False

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                nonlocal called
                called = True
                return [PaperEquation(text_id=10, grp_id=0, lhs="KMO", comp=">", rhs="0.60")]

        eqs = await EquationExtractor().extract_with_llm_fallback(
            sents, self._sections(), llm_client=FakeLLM(), min_regex_stats=1
        )

        assert called is False
        assert eqs == []

    async def test_gate_runs_fallback_when_regex_meets_threshold(self):
        from bibr.paper_contents import PaperEquation

        # First sentence yields a regex stat (t-test); the threshold is met, so
        # the second sentence's missed candidate still reaches the LLM.
        sents = [
            _make_sentence(9, "The effect was reliable (t(28) = 3.42)."),
            _make_sentence(10, "Odd layout (KMO measure of 0.764, adequacy) regex misses."),
        ]

        seen: list[int] = []

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                seen.extend(text_id for text_id, _ in batch)
                return [PaperEquation(text_id=10, grp_id=0, lhs="KMO", comp=">", rhs="0.60")]

        await EquationExtractor().extract_with_llm_fallback(
            sents, self._sections(), llm_client=FakeLLM(), min_regex_stats=1
        )

        assert seen == [10]

    async def test_gate_disabled_by_default_always_runs(self):
        from bibr.paper_contents import PaperEquation

        sents = [_make_sentence(10, "Odd layout (KMO measure of 0.764, adequacy) regex misses.")]

        called = False

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                nonlocal called
                called = True
                return [PaperEquation(text_id=10, grp_id=0, lhs="KMO", comp=">", rhs="0.60")]

        await EquationExtractor().extract_with_llm_fallback(
            sents, self._sections(), llm_client=FakeLLM()
        )

        assert called is True

    async def test_latex_regex_hits_do_not_count_toward_gate(self):
        """Only non-LaTeX regex components satisfy the gate — a paper whose
        regex hits are all LaTeX fragments should still skip the fallback."""
        from bibr.paper_contents import PaperEquation

        display = PaperSentence(
            text_id=9,
            text=r"\alpha = \beta^{2} \tag{1}",
            section_id=1,
            paragraph_id=1,
            is_display_formula=True,
        )
        prose = _make_sentence(10, "Odd layout (KMO measure of 0.764, adequacy) regex misses.")

        called = False

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                nonlocal called
                called = True
                return [PaperEquation(text_id=10, grp_id=0, lhs="KMO", comp=">", rhs="0.60")]

        await EquationExtractor().extract_with_llm_fallback(
            [display, prose], self._sections(), llm_client=FakeLLM(), min_regex_stats=1
        )

        assert called is False
