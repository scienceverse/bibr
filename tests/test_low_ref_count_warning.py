"""Tests for the low-reference-count safety net (residual #3).

When GLM-OCR drops reference regions, references are silently under-extracted
(processing_warnings stays empty). This conservative net counts distinct
author-year in-text citations in the BODY (excluding the reference list) and
warns when the extracted reference count is far below them. It is a general
gross-drop net — it does NOT catch a single dropped lead reference (the Aknin
off-by-one leaves no trace in the output); that needs the deferred text-layer
splice.
"""

from bibr.pipeline.stages.post_parse import (
    _count_distinct_intext_citations,
    _low_reference_count_warning,
)


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
        assert "5" in warn and "20" in warn

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
