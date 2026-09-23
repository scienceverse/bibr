"""Tests for the low-reference-count safety net (residual #3).

When GLM-OCR drops reference regions, references are silently under-extracted
(processing_warnings stays empty). This conservative net counts distinct
author-year in-text citations in the BODY (excluding the reference list), or
the distinct printed numbers its numeric markers cite, and warns when the
extracted reference count is far below them. It is a general
gross-drop net — it does NOT catch a single dropped lead reference (the Aknin
off-by-one leaves no trace in the output); that needs the deferred text-layer
splice.
"""

from bibr.pipeline.stages.post_parse import (
    _count_distinct_intext_citations,
    _count_numbered_citations,
    _low_reference_count_warning,
)
from bibr.processing_warnings import WarningCode


class TestCountDistinctIntextCitations:
    def test_counts_distinct_author_year_forms(self):
        text = (
            "As shown by Smith (2020) and Jones et al. (2019), the effect holds "
            "(Brown, 2018; Lee & Park, 2021). Smith (2020) replicated it."
        )
        # Distinct: Smith 2020, Jones 2019, Brown 2018, Lee 2021 = 4
        # (Smith 2020 counted once despite two mentions)
        assert _count_distinct_intext_citations(text) == 4

    def test_numeric_citation_style_yields_zero(self):
        text = "Prior work [1, 2] showed the effect, replicated in [3] and [4-6]."
        assert _count_distinct_intext_citations(text) == 0

    def test_empty_text(self):
        assert _count_distinct_intext_citations("") == 0


def _body_with_n_distinct_cites(n: int) -> str:
    # Distinct letter-only surnames (real surnames carry no trailing digits).
    names = [f"Auth{a}{b}" for a in "ABCDEFGH" for b in "abcdefgh"][:n]
    return " ".join(f"As reported ({name}, {2000 + i})." for i, name in enumerate(names))


class TestLowReferenceCountWarning:
    def test_warns_on_gross_deficit(self):
        # 20 distinct cited works, only 5 references extracted.
        warn = _low_reference_count_warning(_body_with_n_distinct_cites(20), n_refs=5)
        assert warn is not None
        assert warn.code == WarningCode.REF_UNDER_EXTRACTION_SUSPECTED
        assert warn.message.startswith("5 references parsed vs 20 distinct in-text citations")

    def test_no_warning_when_counts_align(self):
        assert _low_reference_count_warning(_body_with_n_distinct_cites(20), n_refs=20) is None

    def test_no_warning_below_citation_floor(self):
        # Too few citations to judge — never warn (avoids noise on short papers).
        body = "A single citation (Smith, 2020)."
        assert _low_reference_count_warning(body, n_refs=0) is None

    def test_no_warning_for_offbyone(self):
        # Aknin-style off-by-one must NOT trip the conservative net, even well
        # above the citation floor.
        assert _low_reference_count_warning(_body_with_n_distinct_cites(30), n_refs=29) is None


class TestCountNumberedCitations:
    def test_counts_a_dense_run(self):
        assert _count_numbered_citations(set(range(1, 41))) == 40

    def test_tolerates_uncited_numbers_inside_the_run(self):
        # Every other number of 1..40 cited: still half of the run.
        assert _count_numbered_citations(set(range(2, 41, 2))) == 20

    def test_stray_numbers_far_above_the_run_do_not_count(self):
        # A 20-reference paper whose body also brackets 35, 88 and 120.
        assert _count_numbered_citations(set(range(1, 21)) | {35, 88, 120}) == 21

    def test_empty(self):
        assert _count_numbered_citations(set()) == 0


class TestNumberedLowReferenceCountWarning:
    def test_warns_when_numbered_citations_outrun_the_parsed_list(self):
        # A numeric-style body cites 1..40 but only 15 references were parsed.
        warn = _low_reference_count_warning("", n_refs=15, cited_numbers=set(range(1, 41)))
        assert warn is not None
        assert "15 references parsed vs 40 distinct numbered in-text citations" in warn.message

    def test_no_warning_when_counts_align(self):
        assert _low_reference_count_warning("", n_refs=40, cited_numbers=set(range(1, 41))) is None

    def test_no_warning_from_stray_bracketed_values(self):
        # 20 parsed references, all cited, plus 25 scattered bracketed values
        # above the list: none of them extend the dense run.
        stray = set(range(100, 1100, 40))
        cited = set(range(1, 21)) | stray
        assert _low_reference_count_warning("", n_refs=20, cited_numbers=cited) is None

    def test_no_warning_below_citation_floor(self):
        assert _low_reference_count_warning("", n_refs=0, cited_numbers=set(range(1, 15))) is None

    def test_author_year_wording_when_that_count_is_larger(self):
        warn = _low_reference_count_warning(
            _body_with_n_distinct_cites(30), n_refs=5, cited_numbers=set(range(1, 21))
        )
        assert warn is not None
        assert "vs 30 distinct in-text citations" in warn.message
