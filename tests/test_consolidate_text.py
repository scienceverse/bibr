"""Tests for bibr.input.consolidate_text module."""

from bibr.input.consolidate_text import (
    clean_formula_text,
    clean_text_content,
    clean_text_content_late,
    fix_ocr_artifacts,
    strip_affiliation_markers,
    strip_inline_math,
    strip_markdown_fences,
    unwrap_latex_text,
)


class TestFixOcrArtifacts:
    def test_normalizes_decomposed_unicode_to_nfc(self):
        assert fix_ocr_artifacts("Mu\u0308ller and Garci\u0301a") == "Müller and García"

    def test_fi_ligature(self):
        assert fix_ocr_artifacts("\ufb01nding") == "finding"

    def test_fl_ligature(self):
        assert fix_ocr_artifacts("\ufb02ow") == "flow"

    def test_ff_ligature(self):
        assert fix_ocr_artifacts("e\ufb00ect") == "effect"

    def test_ffi_ligature(self):
        assert fix_ocr_artifacts("e\ufb03cient") == "efficient"

    def test_ffl_ligature(self):
        assert fix_ocr_artifacts("ba\ufb04e") == "baffle"

    def test_soft_hyphen(self):
        assert fix_ocr_artifacts("pre\u00adprocessing") == "pre-processing"

    def test_no_artifacts(self):
        text = "This is clean text with no artifacts."
        assert fix_ocr_artifacts(text) == text

    def test_multiple_artifacts(self):
        assert fix_ocr_artifacts("\ufb01rst \ufb02oor") == "first floor"

    def test_full_width_latin_punctuation(self):
        assert fix_ocr_artifacts("（n=7）， participants") == "(n=7), participants"


class TestStxSoftHyphen:
    """GLM-OCR marks line-end soft-hyphen positions with STX (\\x02).

    Plain words re-join without the hyphen, but inside DOIs/URLs the printed
    hyphen is a literal part of the identifier and must be restored
    (exp #2 judged failure: ``annurev-psych-113011-143750`` losing a hyphen).
    """

    def test_stx_in_plain_word_joins_without_hyphen(self):
        assert fix_ocr_artifacts("off\x02line processing") == "offline processing"

    def test_stx_inside_doi_restores_hyphen(self):
        # Diamond (2013): PDF prints "annurev-psych-" at line end, "113011-143750" next line.
        assert (
            fix_ocr_artifacts("doi:10.1146/annurev-psych\x02113011-143750")
            == "doi:10.1146/annurev-psych-113011-143750"
        )

    def test_stx_inside_doi_url_restores_hyphen(self):
        assert (
            fix_ocr_artifacts("http://dx.doi.org/10.5018/economics\x02ejournal.ja.2018-26")
            == "http://dx.doi.org/10.5018/economics-ejournal.ja.2018-26"
        )

    def test_stx_inside_plain_url_restores_hyphen(self):
        assert (
            fix_ocr_artifacts("see https://osf.io/some\x02path/registrations for data")
            == "see https://osf.io/some-path/registrations for data"
        )

    def test_multiple_stx_in_one_doi(self):
        assert (
            fix_ocr_artifacts("10.1146/annurev\x02psych\x02113011")
            == "10.1146/annurev-psych-113011"
        )

    def test_stx_in_word_following_a_url_token_is_stripped(self):
        # The DOI/URL context must be the *same* token, not anywhere earlier.
        assert (
            fix_ocr_artifacts("at https://osf.io/abc the off\x02line condition")
            == "at https://osf.io/abc the offline condition"
        )

    def test_stx_between_digits_restores_hyphen(self):
        # Digits are never syllable-hyphenated: a line-end hyphen between
        # digits is literal (bare ORCID iD wrapped mid-group).
        assert (
            fix_ocr_artifacts("Daniel Lakens 0000-0002\x020247-239X and")
            == "Daniel Lakens 0000-0002-0247-239X and"
        )

    def test_stx_in_year_range_restores_hyphen(self):
        assert fix_ocr_artifacts("between 2010\x022014 the") == "between 2010-2014 the"


class TestStxLiteralCompoundHyphens:
    """Literal compound hyphens the OCR STX-marked at line wraps.

    Capitalised compounds ("Cross-National") and valid-word pairs whose joined
    form is not a word ("approach-related") print a literal hyphen — stripping
    it corrupts titles/abstracts as search keys (exp #2 residual: judged
    ``CrossNational`` / ``LifeSatisfaction`` / ``approachrelated`` fusions).
    True hyphenation ("under-standing", "psychophysio-logical") still joins.
    """

    # --- judged failures: literal hyphen must be restored

    def test_camel_compound_keeps_hyphen(self):
        # Rozer & Kraaykamp (2012) title, judged in ja.2018-43
        assert fix_ocr_artifacts("A Cross\x02National Study") == "A Cross-National Study"

    def test_camel_compound_keeps_hyphen_2(self):
        # Veenhoven (2005) title, judged in ja.2018-43
        assert (
            fix_ocr_artifacts("Dispersion of Life\x02Satisfaction Across Time")
            == "Dispersion of Life-Satisfaction Across Time"
        )

    def test_valid_word_pair_keeps_hyphen(self):
        # Carver & Harmon-Jones (2009) title, judged in 0956797617692000
        assert (
            fix_ocr_artifacts("Anger is an approach\x02related affect")
            == "Anger is an approach-related affect"
        )

    def test_self_compound_keeps_hyphen(self):
        assert fix_ocr_artifacts("low self\x02esteem scores") == "low self-esteem scores"

    def test_acronym_compound_keeps_hyphen(self):
        assert fix_ocr_artifacts("DNA\x02based methods") == "DNA-based methods"

    def test_capitalised_compound_pair_keeps_hyphen(self):
        # The docstring's former "LongTerm" wart, now resolved correctly.
        assert fix_ocr_artifacts("Long\x02Term Effects") == "Long-Term Effects"

    # --- true hyphenation still joins

    def test_joined_word_still_joins(self):
        assert fix_ocr_artifacts("off\x02line processing") == "offline processing"

    def test_hyphenated_common_word_joins(self):
        assert fix_ocr_artifacts("under\x02standing the effect") == "understanding the effect"

    def test_rare_word_hyphenation_joins(self):
        # Joined form is rare but the left fragment is not a word — hyphenation.
        assert (
            fix_ocr_artifacts("psychophysio\x02logical measures") == "psychophysiological measures"
        )

    def test_allcaps_hyphenation_joins(self):
        # All-caps headings hyphenate too; the camel rule must not fire on them.
        assert fix_ocr_artifacts("THE EXPER\x02IMENT BEGAN") == "THE EXPERIMENT BEGAN"


class TestUrlLineWrapBridge:
    """Literal line wraps inside URL/DOI tokens must re-join, keeping the hyphen.

    GLM-OCR usually marks wrap hyphens with STX, but sometimes emits them
    literally (``Lak-\\r\\nens`` or APA-style ``Lak\\n-ens``). Without bridging,
    per-sentence URL detection truncates the URL at the whitespace.
    """

    def test_hyphen_at_line_end_inside_url_bridges(self):
        assert (
            fix_ocr_artifacts("from https://github.com/Lak-\r\nens/to_err_is_human and")
            == "from https://github.com/Lak-ens/to_err_is_human and"
        )

    def test_hyphen_at_line_start_inside_url_bridges(self):
        # APA style breaks URLs *before* punctuation: hyphen leads the next line.
        assert (
            fix_ocr_artifacts("from https://github.com/Lak\n-ens/to_err_is_human and")
            == "from https://github.com/Lak-ens/to_err_is_human and"
        )

    def test_hyphen_linebreak_inside_doi_bridges(self):
        assert (
            fix_ocr_artifacts("doi:10.1146/annurev-psych-\n113011-143750")
            == "doi:10.1146/annurev-psych-113011-143750"
        )

    def test_chained_wraps_in_one_url_bridge(self):
        assert (
            fix_ocr_artifacts("at https://osf.io/long-\npath-\nname/files here")
            == "at https://osf.io/long-path-name/files here"
        )

    def test_hyphen_linebreak_in_plain_word_unchanged(self):
        # Ordinary word wraps are repaired after URL/DOI/numeric protection.
        text = "Data and analy-\nsis code is available"
        assert fix_ocr_artifacts(text) == "Data and analysis code is available"

    def test_linebreak_in_surname_joins_without_hyphen(self):
        assert fix_ocr_artifacts("Kahne-\nman (2011)") == "Kahneman (2011)"

    def test_real_compound_keeps_hyphen_but_loses_linebreak(self):
        assert fix_ocr_artifacts("a meta-\nanalysis") == "a meta-analysis"

    def test_compound_with_valid_joined_variant_keeps_hyphen(self):
        assert fix_ocr_artifacts("subjective well-\nbeing") == "subjective well-being"

    def test_email_token_keeps_literal_hyphen(self):
        assert fix_ocr_artifacts("foo-\nbar@example.org") == "foo-bar@example.org"

    def test_bare_orcid_literal_wrap_bridges(self):
        # Bare ORCID iD (no orcid.org prefix) wrapped mid-group: digits are
        # never syllable-hyphenated, so the hyphen is literal.
        assert (
            fix_ocr_artifacts("Daniel Lakens 0000-0002-\n0247-239X and")
            == "Daniel Lakens 0000-0002-0247-239X and"
        )

    def test_bare_orcid_wrap_at_first_group_bridges(self):
        assert (
            fix_ocr_artifacts("ORCID: 0000-\n0002-0247-239X here")
            == "ORCID: 0000-0002-0247-239X here"
        )

    def test_bare_orcid_apa_wrap_bridges(self):
        assert (
            fix_ocr_artifacts("ORCID: 0000-0002\n-0247-239X here")
            == "ORCID: 0000-0002-0247-239X here"
        )

    def test_scientific_alphanumeric_compounds_keep_hyphen(self):
        assert fix_ocr_artifacts("COVID-\n19 type-\n2 IL-\n6") == "COVID-19 type-2 IL-6"

    def test_numeric_prefix_compound_keeps_hyphen(self):
        assert fix_ocr_artifacts("a 5-\nyear follow-up") == "a 5-year follow-up"

    def test_year_range_wrap_bridges(self):
        assert fix_ocr_artifacts("from 2010-\n2014 we") == "from 2010-2014 we"

    def test_digit_hyphen_before_word_unchanged(self):
        # Digit before the hyphen but a word after: not a numeric token wrap.
        text = "in 2014-\nbased terms"
        assert fix_ocr_artifacts(text) == text


class TestStripMarkdownFences:
    def test_strips_markdown_fence(self):
        text = "```markdown\nHello world\n```"
        assert strip_markdown_fences(text) == "Hello world"

    def test_strips_plain_fence(self):
        text = "```\nSome text here\n```"
        assert strip_markdown_fences(text) == "Some text here"

    def test_no_fence(self):
        text = "Regular text"
        assert strip_markdown_fences(text) == "Regular text"


class TestUnwrapLatexText:
    def test_unwraps_single_text(self):
        assert unwrap_latex_text(r"$$\text{hello world}$$") == "hello world"

    def test_unwraps_multiple_text(self):
        result = unwrap_latex_text(r"$$\text{first} \text{second}$$")
        assert result == "first second"

    def test_no_match(self):
        text = "regular text"
        assert unwrap_latex_text(text) == text

    def test_display_math_not_text(self):
        text = r"$$x^2 + y^2$$"
        assert unwrap_latex_text(text) == text


class TestStripInlineMath:
    def test_strips_inline_stat(self):
        text = "result was $ t(97.7)=2.9 $ , $ p=0.005 $ , d=0.59."
        result = strip_inline_math(text)
        assert result == "result was t(97.7)=2.9 , p=0.005 , d=0.59."

    def test_strips_compact_inline(self):
        assert strip_inline_math("$M = 5.17$") == "M = 5.17"

    def test_does_not_match_display_math(self):
        text = r"$$\text{hello}$$"
        assert strip_inline_math(text) == text

    def test_does_not_match_lone_dollar(self):
        assert strip_inline_math("We donated $5 to charity") == "We donated $5 to charity"

    def test_strips_superscript_marker(self):
        assert strip_inline_math("text$^{1}$ more") == "text^{1} more"

    def test_two_currency_amounts_not_fused(self):
        """Two $N amounts in one sentence must not pair up as inline math."""
        text = "the price was $5 in May and $10 in June."
        assert strip_inline_math(text) == text

    def test_currency_pair_with_stat_inside_not_fused(self):
        """DG-Econ pattern: dollar amounts with statistics between them."""
        text = "donations rose from $50 (p < .001) and $200 overall."
        assert strip_inline_math(text) == text

    def test_mixed_currency_and_inline_math(self):
        """A currency $ must not consume the opening $ of a later math span."""
        text = "costs $5 and stat $ p=0.005 $ here"
        assert strip_inline_math(text) == "costs $5 and stat p=0.005 here"

    def test_no_dollars(self):
        text = "Plain text without math"
        assert strip_inline_math(text) == text


class TestStripAffiliationMarkers:
    def test_dollar_wrapped_superscript(self):
        assert strip_affiliation_markers("Daniel Lakens $ ^{1} $") == "Daniel Lakens"

    def test_bare_braced_superscript(self):
        assert strip_affiliation_markers("Lakens ^{1}") == "Lakens"

    def test_multi_affiliation(self):
        assert strip_affiliation_markers("Lakens ^{1,2}") == "Lakens"

    def test_multi_affiliation_spaced(self):
        assert strip_affiliation_markers("Lakens ^{1, 2, 3}") == "Lakens"

    def test_dagger_symbol(self):
        assert strip_affiliation_markers("Lakens ^{†}") == "Lakens"

    def test_asterisk_symbol(self):
        assert strip_affiliation_markers("Lakens ^{*}") == "Lakens"

    def test_bare_superscript_no_braces(self):
        assert strip_affiliation_markers("Lakens ^1") == "Lakens"

    def test_no_markers(self):
        assert strip_affiliation_markers("John Smith") == "John Smith"

    def test_preserves_particles(self):
        assert strip_affiliation_markers("van der Berg") == "van der Berg"

    def test_preserves_apostrophe(self):
        assert strip_affiliation_markers("O'Brien") == "O'Brien"


class TestCleanTextContent:
    def test_preserves_inline_math_for_extraction(self):
        """Early-phase cleaning preserves $...$ for equation extractor."""
        result = clean_text_content("result $ p=0.005 $ end.")
        assert result == "result $ p=0.005 $ end."

    def test_preserves_superscript_for_citation_linker(self):
        """Early-phase cleaning preserves ^{N} for citation linker."""
        result = clean_text_content("results^{3}.")
        assert result == "results^{3}."

    def test_discards_broken_latex(self):
        assert clean_text_content(r"$$\begin{array} broken stuff") is None

    def test_discards_trailing_latex(self):
        assert clean_text_content(r"\end{array}$$") is None

    def test_returns_none_for_empty(self):
        assert clean_text_content("") is None
        assert clean_text_content("   ") is None

    def test_passes_through_clean_text(self):
        text = "This is clean body text."
        assert clean_text_content(text) == text


class TestCleanTextContentLate:
    def test_strips_inline_math(self):
        result = clean_text_content_late("result $ p=0.005 $ end.")
        assert result == "result p=0.005 end."

    def test_strips_superscript(self):
        result = clean_text_content_late("results^{3}.")
        assert result == "results3."

    def test_strips_latex_commands(self):
        result = clean_text_content_late(r"the \alpha value")
        assert "\u03b1" in result

    def test_passes_through_clean_text(self):
        text = "This is clean body text."
        assert clean_text_content_late(text) == text

    def test_underscores_in_url_survive(self):
        # _LATEX_SUB_SINGLE_RE must not eat underscores inside URLs
        # ("to_err_is_human" → "toerrishuman").
        text = "code from https://github.com/Lakens/to_err_is_human now."
        assert clean_text_content_late(text) == text

    def test_latex_still_stripped_outside_url(self):
        result = clean_text_content_late("results^{3} see https://github.com/a_b.")
        assert result == "results3 see https://github.com/a_b."

    def test_caret_in_url_survives(self):
        text = "at https://example.com/q?x^2=1 end."
        assert clean_text_content_late(text) == text

    def test_discards_broken_array(self):
        assert clean_formula_text(r"$$\begin{array} stuff") is None

    def test_discards_trailing_array(self):
        assert clean_formula_text(r"\end{array}$$") is None

    def test_unwraps_text_formula(self):
        result = clean_formula_text(r"$$\text{plain text}$$")
        assert result == "plain text"

    def test_passes_through_real_formula(self):
        text = r"$$x^2 + y^2 = z^2$$"
        assert clean_formula_text(text) == text

    def test_discards_affiliation_line_single_dollar(self):
        assert clean_formula_text(r"$ ^1 \text{ Eindhoven University of Technology} $") is None

    def test_discards_affiliation_line_double_dollar(self):
        assert clean_formula_text(r"$$ ^{1,2} \text{MIT} $$") is None

    def test_collapses_spaced_text(self):
        result = clean_formula_text(r"$$\beta_{0} + \text{g e n d e r}_{i}$$")
        assert result == r"$$\beta_{0} + \text{gender}_{i}$$"

    def test_collapses_spaced_mathrm(self):
        result = clean_formula_text(r"\log(u(t)) = \mathrm{i n t e r c e p t} + s(t, 4)")
        assert result == r"\log(u(t)) = \mathrm{intercept} + s(t, 4)"

    def test_collapses_spaced_text_with_punctuation(self):
        result = clean_formula_text(r"\text{c h i l d h o o d s e l f - c o n t r o l}")
        assert result == r"\text{childhoodself-control}"

    def test_preserves_normal_text(self):
        """Multi-char tokens → not spaced-out OCR, leave alone."""
        result = clean_formula_text(r"\text{hello world}")
        assert result == r"\text{hello world}"

    def test_collapses_spaced_operatorname(self):
        result = clean_formula_text(r"\operatorname{A t t e n t i o n}(Q, K, V)")
        assert result == r"\operatorname{Attention}(Q, K, V)"


class TestLateCleanupScope:
    """Late cleanup flattens LaTeX inside ``$…$`` for every source; outside
    those spans it repairs OCR text only, and prose never loses characters."""

    PROSE = [
        "Participants completed a 2 x 2 x 3 mixed design.",
        "We used a 2 × 2 factorial design.",
        "Items 1 2 3 and 4 were reverse scored.",
        "The variable age_group was coded from one to five.",
        "Data are available from john_smith@uni.edu on request.",
        "We ran 10^6 bootstrap iterations with glmer_nb in analysis_script.R.",
        "The model is described in \\cite{smith} and uses a_{i} weights.",
    ]

    def test_text_read_from_the_document_keeps_its_prose(self):
        for text in self.PROSE:
            assert clean_text_content_late(text, from_ocr=False) == text

    def test_math_spans_are_flattened_for_every_source(self):
        text = "Fit was good, $R^2 = .45$, with $\\alpha_{i}$ free."
        expected = "Fit was good, R2 = .45, with αi free."
        assert clean_text_content_late(text, from_ocr=False) == expected
        assert clean_text_content_late(text) == expected

    def test_whitespace_is_collapsed_for_every_source(self):
        assert clean_text_content_late("one\r\ntwo  three", from_ocr=False) == "one two three"

    def test_ocr_text_keeps_identifiers_and_emails(self):
        text = "Contact john_smith@uni.edu; the age_group variable, 10^6 draws."
        assert clean_text_content_late(text) == text

    def test_ocr_text_still_loses_leaked_latex(self):
        text = "the \\alpha value of \\mathrm{SD} was 1.2^{3}, see $x_i$."
        assert clean_text_content_late(text) == "the α value of SD was 1.23, see xi."

    def test_ocr_design_notation_is_not_fused(self):
        for text in self.PROSE[:2]:
            assert clean_text_content_late(text) == text

    def test_ocr_character_spacing_is_still_collapsed(self):
        assert clean_text_content_late("F(1, 26) = 1 7. 9 0 6") == "F(1, 26) = 1 7. 906"
        assert clean_text_content_late("the mean was $ 1 7. 9 0 6 $") == "the mean was 1 7. 906"

    def test_only_an_x_between_digits_is_an_operator(self):
        assert clean_text_content_late("code x 1 2 3 end") == "code x123 end"
        assert clean_text_content_late("code 1 2 x a end") == "code 12xa end"

    LITERAL_DOLLARS = [
        "We recoded df$age_group and df$score_z before fitting.",
        "Costs ranged from US$ 60 to US$ 1,419 per episode.",
        "GDP was 54.9 US$ million, against 1414 US$ per unit_price.",
        "Reads were trimmed with cutadapt -o $SAMPLE_R1.fq -p $SAMPLE_R2.fq.",
        "adapt -a file:$ADAPTER -A file:$ADAPTER -o $SAMPLE.R1.fq",
        "Intensity was around 35 MJ/$, while it fell to only 10 MJ/$ in 2000.",
        "*P<0.05 between groups. $P<0.05, $$P<0.005 between isoforms.",
        r"the consensus (D\w\w[LIMV][LIMV]\w{0,3}$, and D\w\w[LI][LI]\w{0,20}$) we",
        'paste0("$", comma(mapdata$Cost_Off_Campus, digits = 0)),',
    ]

    def test_literal_dollars_in_document_text_are_not_math(self):
        """Only DOCX writes ``$…$`` math into document text; elsewhere a
        dollar is currency, R's ``df$col`` or a shell variable, and pairing
        two of them deleted the dollars and the underscores between."""
        for text in self.LITERAL_DOLLARS:
            assert clean_text_content_late(text, from_ocr=False) == text

    def test_document_math_spans_are_still_unwrapped(self):
        docx = "We fit a model where $β_i=0$ for age_group."
        assert clean_text_content_late(docx, from_ocr=False) == (
            "We fit a model where βi=0 for age_group."
        )
        tex = r"\begin{document}$\textbf{q} = \textbf{q}P$\end{document}"
        assert (
            clean_text_content_late(tex, from_ocr=False) == r"\begin{document}q = qP\end{document}"
        )

    def test_paddle_inline_math_is_flattened_in_full(self):
        """Paddle-OCR-VL writes inline math as ``\\(…\\)``; inside it ``_p``
        and ``^2`` are LaTeX, as they are inside ``$…$``."""
        text = r"The effect was large, \(\eta_p^2 = .12\), and \(d_z = 0.41\), see age_group."
        assert clean_text_content_late(text) == (
            r"The effect was large, \(ηp2 = .12\), and \(dz = 0.41\), see age_group."
        )
        assert clean_text_content_late(text, from_ocr=False) == text

    def test_email_inside_a_math_span_keeps_its_underscore(self):
        """A currency dollar can pair with a later one and put an address
        inside a math span, where the single-character rule applies."""
        text = "Pay US$ 20 by writing to john_smith@uni.edu, then $ back."
        assert "john_smith@uni.edu" in clean_text_content_late(text)
        assert clean_text_content_late(text, from_ocr=False) == text
