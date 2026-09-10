"""Tests for bibr.extract.segment_filter — dropping non-reference segments.

Junk strings below are verbatim offenders mined from the exp #2 judge
corrections (evaluation/results/exp2_verdict.md): front/back-matter that the
anchor-emit segmenter faithfully segmented as bib entries.
"""

from bibr.extract.segment_filter import drop_non_reference_segments, is_non_reference_segment

# Observed offenders (exp #2, arm B high-severity corrections).
JUNK_SEGMENTS = [
    "Additional supporting information can be found at http://journals.sagepub.com/doi/suppl/10.1177/0956797621995197",
    "Additional supporting information can be found at http://pss.sagepub.com/content/by/supplemental-data",
    "Supplemental Material",
    "More information about the Open Practices badges can be found at http://www.psychologicalscience.org/publications/badges",
    # MDPI back-matter disclaimer
    "Disclaimer/Publisher's Note: The statements, opinions and data contained in all publications are solely those of the individual author(s).",
    "Disclaimer/Publisher’s Note:",
    # ejournal DOI footer fragments
    "http://dx.doi.org/10.5018/economics-ejournal.ja.2013-9",
    "Please go to: http://dx.doi.org/10.5018/economics-ejournal.ja.2013-10",
    # byline of the paper itself captured as a "reference" (no year, no locator)
    "Delaram Farzanfar, Dirk B. Walther",
]

# Real references in the styles seen across the three corpora — must ALL be kept.
REAL_REFERENCES = [
    "Bor, A., & Petersen, M. B. (2022). The psychology of online political hostility: "
    "A comprehensive, cross-national test of the mismatch hypothesis. "
    "American Political Science Review, 116, 1–18.",
    "Brady, W. J., McLoughlin, K., Doan, T. N., & Crockett, M. J. (2021). How social "
    "learning amplifies moral outrage expression in online social networks. "
    "Science Advances, 7(33), Article eabe5641.",
    "Caballero, R. J., Hoshi, T., & Kashyap, A. K. (2008). Zombie lending and depressed "
    "restructuring in Japan. American Economic Review, 98(5), 1943–1977.",
    # numbered style
    "[12] Smith, J., & Jones, K. (2019). A study of things. Journal of Studies, 4, 1–10.",
    # n.d. web reference (no 4-digit year)
    "Pew Research Center. (n.d.). Internet/broadband fact sheet. https://www.pewresearch.org/internet/fact-sheet/",
    # in-press reference (no year)
    "Robertson, C. E., del Rosario, K., & Van Bavel, J. J. (in press). Inside the funhouse "
    "mirror factory: How social media distorts perceptions of norms.",
    # short organisational web ref with year + URL
    "WHO. (2020). Coronavirus disease. https://www.who.int/health-topics/coronavirus",
    # reference whose TITLE mentions supplemental material (must not be over-matched:
    # junk patterns are start-anchored, this starts with an author)
    "Kerr, N. L. (1998). HARKing: Hypothesizing after the results are known, with "
    "additional supporting information. Personality and Social Psychology Review, 2, 196–217.",
    # short real references that END just after their title/page range — the
    # in-text-citation filter must not over-match these (year is NOT terminal,
    # a title or page range follows it).
    "Jasper, H. H. (1948). Charting the sea of brain waves. Science, 108(2805), 343–347.",
    "Gelman, A., & Stern, H. (2006). The difference between significant and not "
    "significant. The American Statistician, 60(4), 328–331.",
    "Wang, X.-J. (2010). Cortical rhythms in cognition. Physiological Reviews, 90(3), 1195.",
    # Numbered styles END at the year, exactly like an in-text cite — only the
    # reference structure *before* the year separates them.
    "12. Rothman KJ. Modern epidemiology. Boston: Little, Brown; 1986.",
    "7. Doe J, Roe A. A trial of X. Lancet. 2019.",
    '[3] J. Smith, "Deep nets," IEEE Trans. Comput., vol. 5, no. 2, pp. 1-10, 1999.',
    "[8] A. Kumar and B. Lee, Handbook of Signals. New York, NY: Springer, 2004.",
]

# In-text parenthetical citations that leaked into ref_text (exp #2 paper
# 09567976211001317: an acknowledgment sentence reclaimed as a boundary
# orphan). The anchor-seg split the parenthetical into bare "Author, YEAR"
# fragments — year-bearing, so the year guard kept them; they must be dropped
# as citations (nothing but authors + a terminal year).
INTEXT_CITATIONS = [
    "Gelman & Stern, 2006;",
    "Nieuwenhuis et al., 2011).",
    "Tooby & Cosmides, 1992",
    "Smith et al., 2019)",
    "(Gelman & Stern, 2006; Nieuwenhuis et al., 2011).",
    "Baldauf & Desimone, 2014a;",
]


class TestIsNonReferenceSegment:
    def test_drops_each_observed_junk_segment(self):
        for seg in JUNK_SEGMENTS:
            assert is_non_reference_segment(seg), f"should drop: {seg!r}"

    def test_keeps_each_real_reference(self):
        for ref in REAL_REFERENCES:
            assert not is_non_reference_segment(ref), f"should keep: {ref!r}"

    def test_drops_each_intext_citation(self):
        for seg in INTEXT_CITATIONS:
            assert is_non_reference_segment(seg), f"should drop in-text cite: {seg!r}"

    def test_drops_empty_and_whitespace(self):
        assert is_non_reference_segment("")
        assert is_non_reference_segment("   \n ")


class TestDropNonReferenceSegments:
    def test_filters_junk_preserves_order(self):
        mixed = [JUNK_SEGMENTS[0], REAL_REFERENCES[0], JUNK_SEGMENTS[4], REAL_REFERENCES[1]]
        assert drop_non_reference_segments(mixed) == [REAL_REFERENCES[0], REAL_REFERENCES[1]]

    def test_all_real_references_survive(self):
        assert drop_non_reference_segments(REAL_REFERENCES) == REAL_REFERENCES

    def test_empty_input(self):
        assert drop_non_reference_segments([]) == []
