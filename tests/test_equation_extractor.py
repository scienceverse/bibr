"""Tests for equation extraction from paper sentences."""

import asyncio

from bibr.extract.equation_extractor import EquationExtractor
from bibr.paper_contents import PaperSection, PaperSentence


def _make_sentence(text_id: int, text: str, section_id: int = 1) -> PaperSentence:
    return PaperSentence(text_id=text_id, text=text, section_id=section_id, paragraph_id=1)


def _make_sections() -> list[PaperSection]:
    return [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        PaperSection(section_id=1, header="Results", level=1, parent_section_id=0),
    ]


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

        assert len(eqs) == 1
        eq = eqs[0]
        assert "CI" in eq.lhs
        assert eq.comp == "="
        assert "[2.0, 4.7]" in eq.rhs

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
        """Cohen's d = 0.8 should be caught by the broad pass."""
        sent = _make_sentence(1, "The effect was large, Cohen\u2019s d = 0.8.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert any(eq.lhs == "Cohen\u2019s d" and eq.rhs == "0.8" for eq in eqs)

    def test_bayes_factor(self):
        """BF10 = 12.4 should be caught by the broad pass."""
        sent = _make_sentence(1, "Evidence was strong, BF10 = 12.4.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert any(eq.lhs == "BF10" and eq.rhs == "12.4" for eq in eqs)

    def test_scientific_notation(self):
        """p = 3.2 e -5 should capture the scientific notation suffix."""
        sent = _make_sentence(1, "The result was significant, p = 3.2 e -5.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        p_eqs = [eq for eq in eqs if eq.lhs == "p"]
        assert len(p_eqs) >= 1
        assert "3.2" in p_eqs[0].rhs

    def test_power_of_10_notation(self):
        """p = 2.1 x 10^-4 should capture the power-of-10 suffix."""
        sent = _make_sentence(1, "We found p = 2.1 x 10^-4 in the analysis.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        p_eqs = [eq for eq in eqs if eq.lhs == "p"]
        assert len(p_eqs) >= 1
        assert "2.1" in p_eqs[0].rhs

    def test_tilde_operator(self):
        """β ~ 0.45 with tilde as approximate operator."""
        sent = _make_sentence(1, "The estimate was approximately beta ~ 0.45.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert any(eq.comp == "~" and eq.rhs == "0.45" for eq in eqs)

    def test_icc_stat(self):
        """ICC = 0.85 should be caught by the broad pass."""
        sent = _make_sentence(1, "Reliability was high, ICC = 0.85.")
        extractor = EquationExtractor()
        eqs = extractor.extract_from_sentences([sent], _make_sections())

        assert any(eq.lhs == "ICC" and eq.rhs == "0.85" for eq in eqs)

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

        assert any("[1.2, 3.4]" in eq.rhs for eq in eqs)

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

        assert len(eqs) == 1
        assert "-0.32" in eqs[0].rhs or "−0.32" in eqs[0].rhs

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


class TestLlmFallbackCitationFilter:
    """x-performance-2: sentences whose digit-bearing parentheticals are only
    citations or figure/table references must not burn LLM calls."""

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

    async def test_citation_and_reference_only_sentences_skipped(self):
        sents = [
            _make_sentence(10, "This replicates prior findings (Teckchandani et al., 2014)."),
            _make_sentence(11, "Earlier work agrees (Smith, 2020; Jones et al., 2019)."),
            _make_sentence(12, "The layout follows (Figure 3c,d) with values in (Table 2)."),
            _make_sentence(13, "Derived in (Eq. 3) and reviewed in (Section 2.3)."),
            _make_sentence(14, "The cohort was recruited in (2020) with follow-up in (2021)."),
            _make_sentence(15, "Odd layout (KMO measure of 0.764, adequacy) regex misses."),
        ]
        seen: list[int] = []

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                seen.extend(text_id for text_id, _ in batch)
                return []

        await EquationExtractor().extract_with_llm_fallback(
            sents, self._sections(), llm_client=FakeLLM()
        )

        # Only the genuine prose candidate reaches the LLM.
        assert seen == [15]

    async def test_mixed_citation_and_stat_sentence_still_sent(self):
        sents = [
            _make_sentence(
                10,
                "Prior work disagrees (Smith, 2020) but odd layout (values 12 versus 17) persists.",
            ),
        ]
        seen: list[int] = []

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                seen.extend(text_id for text_id, _ in batch)
                return []

        await EquationExtractor().extract_with_llm_fallback(
            sents, self._sections(), llm_client=FakeLLM()
        )

        assert seen == [10]

    async def test_prose_stat_with_reference_paren_still_sent(self):
        """Guard from real gate192 exports: prose statistics ("grand mean of
        56.14", "PCC of 0.921") co-occur with reference parentheticals, and
        the regex pass misses them — the LLM must still see them."""
        sents = [
            _make_sentence(
                10, "The grand mean of 56.14 was observed for all 196 entries (Tab. 1)."
            ),
            _make_sentence(
                11,
                "Model fit was acceptable if CFI and TLI were above 0.90 (Byrne, 2012).",
            ),
            _make_sentence(12, "The compound achieved a PCC of 0.921 (Fig. 3a)."),
        ]
        seen: list[int] = []

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                seen.extend(text_id for text_id, _ in batch)
                return []

        await EquationExtractor().extract_with_llm_fallback(
            sents, self._sections(), llm_client=FakeLLM()
        )

        assert seen == [10, 11, 12]

    async def test_keywordless_digit_group_is_kept(self):
        """Guard against overreach: a digit group that matches neither the
        citation nor the reference pattern stays a candidate."""
        sents = [_make_sentence(10, "Responses (3c,d) were excluded from analysis.")]
        seen: list[int] = []

        class FakeLLM:
            async def extract_equations(self, batch, file_hash="x"):
                seen.extend(text_id for text_id, _ in batch)
                return []

        await EquationExtractor().extract_with_llm_fallback(
            sents, self._sections(), llm_client=FakeLLM()
        )

        assert seen == [10]

    def test_has_statistical_paren_unit_table(self):
        from bibr.extract.equation_extractor import _has_statistical_paren

        skipped = [
            "The effect replicated prior work (Teckchandani et al., 2014).",
            "As shown before (Smith, 2020; Jones et al., 2019).",
            "See the design (Figure 3c,d) for details.",
            "Values are in (Table 2) and (Supplementary Table S1).",
            "Derived in (Eq. 3) and discussed in (Section 2.3).",
            "Published in (2020) with follow-up (2021).",
            "Equal variances (Equal variances assumed, 2020) noted.",
        ]
        for text in skipped:
            assert _has_statistical_paren(text) is False, text

        kept = [
            "The difference was significant (t(28) = 3.42, p = .003).",
            "Sample size was set (n = 100) per group.",
            "Means differed (M = 4.2, SD = 1.1).",
            "KMO measure was 0.764 (>0.60) adequate.",
            "The model (n=50) showed improvement.",
            "Ambiguous (data 123) result.",
            "Mixed result (Smith, 2020) with (t(28) = 3.42).",
            "Kept safe (see Figure 3, t = 5.2) case.",
            "Tablet counts (Tablet 5mg, n = 30) recorded.",
            "No digits here at all.",  # no candidate either way
        ]
        for text in kept[:-1]:
            assert _has_statistical_paren(text) is True, text
        assert _has_statistical_paren(kept[-1]) is False

    def test_body_digits_rescue_reference_only_sentence(self):
        """A sentence whose digit groups are all references is still queued
        when the prose around them carries digits (real export cases)."""
        from bibr.extract.equation_extractor import _has_statistical_paren

        rescued = [
            "The grand mean of 56.14 was observed for all 196 entries (Tab. 1).",
            "Model fit was acceptable if CFI and TLI were above 0.90 (Byrne, 2012).",
            "The compound achieved a PCC of 0.921 (Fig. 3a).",
            "The mean ages were 49.6±13.6 for outpatients (Table 1).",
        ]
        for text in rescued:
            assert _has_statistical_paren(text) is True, text

    def test_statistic_hints_keep_mixed_reference_groups(self):
        """Statistics sharing a citation/reference paren stay candidates.

        A Figure/Table group may hold only reference tokens; a citation
        piece must read as names plus years. CI, "%", OR/HR/SD/SE and kin,
        "alpha"/"beta", a statistic letter with a number, and non-section
        decimals all mark a statistic.
        """
        from bibr.extract.equation_extractor import _has_statistical_paren

        kept = [
            "The effect was robust (Cohen's d 0.45; Smith et al., 2019).",
            "Accuracy improved (Table 2; M 3.45, SD 1.20).",
            "Responses were slower (Figure 3, 95% CI 1.2 to 3.4).",
            "Participants were recruited online (N 1850).",
            "Scores rose (SE 1.2, N 2013).",
            "Smoking was associated with higher risk (Table 2; OR 1.85, 95% CI 1.20-2.90).",
            "Effects were robust (see Supplementary Table S3; beta 0.34, SE 0.05).",
            "Scores were high overall (M 12.3, SD 2.1, N 2000).",
            "Reliability was good (Cronbach alpha .87; Figure 2).",
            "Accuracy improved markedly (Fig. 4b: 71.2% vs 64.5%).",
            "The indirect effect was significant (Model 4 from Hayes, 2013; 95% CI [0.12, 0.45]).",
        ]
        for text in kept:
            assert _has_statistical_paren(text) is True, text

    def test_single_letter_sample_size_is_not_a_citation(self):
        """ "N 1850" must not read as a one-letter author plus a year."""
        from bibr.extract.equation_extractor import _has_statistical_paren

        assert _has_statistical_paren("Participants were recruited online (N 1850).") is True
        assert _has_statistical_paren("Scores were high overall (M 12.3, SD 2.1, N 2000).") is True

    def test_tablet_without_operator_stays_a_candidate(self):
        """`Tablet` merely starts with the `table` keyword: the reference
        lookahead must keep it out, so the group stays a candidate. Same
        for a keyword prefix joined to reference tokens (`Tableand 2`):
        without the lookahead the tail alone would read as a reference.
        """
        from bibr.extract.equation_extractor import _has_statistical_paren

        assert _has_statistical_paren("Dose was fixed (Tablet 5mg) per protocol.") is True
        assert _has_statistical_paren("Results held (Tableand 2) across runs.") is True

    def test_see_lead_citation_is_filtered(self):
        """The citation "see" lead is part of the pattern; without it a
        lowercase "see Smith, 2020" would not match and would waste a call."""
        from bibr.extract.equation_extractor import _has_statistical_paren

        assert _has_statistical_paren("As shown before (see Smith, 2020).") is False

    def test_page_span_citation_is_filtered(self):
        """A trailing page span does not make a citation statistical."""
        from bibr.extract.equation_extractor import _has_statistical_paren

        assert _has_statistical_paren("As shown before (Smith, 2020, p. 5).") is False

    def test_long_citation_groups_finish_quickly(self):
        """Linear-time guard: a 30-citation group with an "in press" tail
        and a long comma-year run with a non-year tail each finish well
        under 50 ms (the old nested pattern hung exponentially)."""
        import time

        from bibr.extract.equation_extractor import _has_statistical_paren

        names = [
            "Smith",
            "Jones",
            "Brown",
            "Lee",
            "Kim",
            "Park",
            "Chen",
            "Wang",
            "Li",
            "Zhang",
            "Liu",
            "Garcia",
            "Miller",
            "Davis",
            "Wilson",
        ]
        cites = "; ".join(f"{names[i % len(names)]} et al., {2000 + i}" for i in range(30))
        long_press = f"Prior work agrees ({cites}; Martin, in press)."
        start = time.perf_counter()
        _has_statistical_paren(long_press)
        assert time.perf_counter() - start < 0.05, "30-citation group took too long"

        comma_run = "(A" + " 1999," * 8000 + " x)"
        start = time.perf_counter()
        _has_statistical_paren(comma_run)
        assert time.perf_counter() - start < 0.05, "comma-year run took too long"
