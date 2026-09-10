"""Tests for the per-reference author rescue (qual-grade30 M2).

A single batched parse call (REF_PARSE_BATCH_SIZE refs) intermittently returns
``authors=None`` for EVERY entry while parsing title/year/container correctly
(observed: 15 contiguous refs in 10.5018_economics-ejournal.ja.2018-7). The
authors are present verbatim at the head of each author-date segment, so the
loss is an LLM under-emission, not missing data.

``_rescue_authors_from_segment`` recovers the printed byline by cutting at the
first parenthesized publication year. It is scoped to author-date references:
Vancouver / numbered styles (no leading ``(YEAR)``) are declined so the cut
never over-captures title/container, and it never fabricates a byline. The
recovered string is normalized exactly like an LLM-parsed author (trailing
``.,`` stripped), so a rescued byline is indistinguishable from a parsed one.
"""

from bibr.extract.extractor import _rescue_authors_from_segment


class TestRescueAuthorsFromSegment:
    def test_recovers_two_author_byline(self):
        seg = (
            "Drehmann, M., and Tsatsaronis, K. (2014). The credit-to-GDP Gap and "
            "Counter-cyclical Capital Buffers: Questions and Answers. BIS Quarterly "
            "Review, 2014(March): 55-73."
        )
        assert _rescue_authors_from_segment(seg) == "Drehmann, M., and Tsatsaronis, K"

    def test_recovers_corporate_single_token_author(self):
        seg = "ECB (2010). Survey on Access to Finance of Enterprises. European Central Bank."
        assert _rescue_authors_from_segment(seg) == "ECB"

    def test_recovers_three_author_byline(self):
        seg = (
            "Fagiolo, G., Moneta, A., and Windrum, P. (2007). A Critical Guide to "
            "Empirical Validation of Agent-Based Models. Computational Economics, 30, 195-226."
        )
        assert _rescue_authors_from_segment(seg) == "Fagiolo, G., Moneta, A., and Windrum, P"

    def test_recovers_with_year_letter_suffix(self):
        seg = "Fagiolo, G., and Roventini, A. (2012a). On the Scientific Status of Economic Policy."
        assert _rescue_authors_from_segment(seg) == "Fagiolo, G., and Roventini, A"

    def test_recovers_with_month_in_year_parenthetical(self):
        seg = "Smith, J. (2014, March). A working paper. Some Institute."
        assert _rescue_authors_from_segment(seg) == "Smith, J"

    def test_strips_trailing_comma_before_year(self):
        seg = "Smith, J., (2020). A paper. Journal, 1, 2-3."
        assert _rescue_authors_from_segment(seg) == "Smith, J"

    def test_declines_vancouver_year_at_end(self):
        # Year is bare at the end; "(3)" is an issue number, not a 4-digit year.
        seg = "Smith J, Jones K. A randomized trial of the thing. Lancet. 2020;15(3):55-60."
        assert _rescue_authors_from_segment(seg) is None

    def test_declines_when_no_author_prefix(self):
        # Leading "(YEAR)" with no byline before it.
        seg = "(2020). Anonymous statistical report. Some Agency."
        assert _rescue_authors_from_segment(seg) is None

    def test_declines_when_no_parenthesized_year(self):
        seg = "Smith, J. A paper with no year in parentheses. Journal."
        assert _rescue_authors_from_segment(seg) is None

    def test_returns_none_on_empty_segment(self):
        assert _rescue_authors_from_segment("") is None
        assert _rescue_authors_from_segment(None) is None
