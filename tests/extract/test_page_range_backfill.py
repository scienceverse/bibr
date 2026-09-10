"""Segment-anchored completion of half-emitted page ranges.

Both parse paths share ``_finalize_reference_fields``; the CRF drops the
start of a range (audit M9) and the LLM the end of a compact one. The fill
is anchored on the value the parser did emit and never overwrites a field.
"""

from bibr.extract.ref_extractor import _backfill_page_range, _finalize_reference_fields


def _finalize(segment, **fields):
    base = {"title": "T", "authors": "A", "doi": None}
    return _finalize_reference_fields({**base, **fields}, segment)


class TestAnchoredOnFirstPage:
    def test_llm_compact_range_gets_its_last_page(self):
        # NuExtract returned only first_page for a printed "339–42".
        seg = "Kopf, M. (1994). Impaired responses. Nature, 368(6469), 339–42."
        out = _finalize(seg, first_page="339", last_page=None)
        assert (out["first_page"], out["last_page"]) == ("339", "342")

    def test_letter_prefixed_pages(self):
        seg = "Smith J. Title. J Clin Oncol. 2019; 37 (suppl): S163-S170."
        out = _finalize(seg, first_page="S163", last_page=None)
        assert out["last_page"] == "S170"

    def test_anchor_inside_a_doi_is_not_a_range(self):
        seg = "Doe A. Title. BMC Cardiovasc Disord. 2021; 21: 1. doi:10.1186/s12872-021-02446-z"
        out = _finalize(seg, first_page="21", last_page=None)
        assert out["last_page"] is None

    def test_anchor_inside_a_volume_issue_span_is_not_a_range(self):
        seg = "Roe, B. (2022). Title. Journal of Items, 12(3-4), 100."
        out = _finalize(seg, first_page="3", last_page=None)
        assert out["last_page"] is None

    def test_two_candidate_ranges_fill_nothing(self):
        seg = "Roe, B. (2022). Title. Vol. 12-15, Journal of Items, 12-20."
        out = _finalize(seg, first_page="12", last_page=None)
        assert out["last_page"] is None

    def test_no_segment_fills_nothing(self):
        out = _finalize(None, first_page="339", last_page=None)
        assert out["last_page"] is None


class TestAnchoredOnLastPage:
    def test_crf_end_without_start_gets_its_first_page(self):
        # ajpa.23008-style row: the CRF tagged the end and left the start "O".
        seg = "Smith, J. (2014). Title. Virulence, 5(1), 20-26."
        out = _finalize(seg, first_page=None, last_page="26")
        assert (out["first_page"], out["last_page"]) == ("20", "26")

    def test_anchor_that_is_a_suffix_of_a_longer_number_is_ignored(self):
        seg = "Smith, J. (2014). Title. Virulence, 5(1), 1020-1026."
        out = _finalize(seg, first_page=None, last_page="26")
        assert out["first_page"] is None


class TestLumpedRanges:
    def test_range_lumped_into_last_page_is_split(self):
        fields = {"first_page": None, "last_page": "41–49"}
        _backfill_page_range(fields, None)
        assert (fields["first_page"], fields["last_page"]) == ("41", "49")

    def test_range_lumped_into_first_page_is_split(self):
        fields = {"first_page": "41-49", "last_page": None}
        _backfill_page_range(fields, "irrelevant")
        assert (fields["first_page"], fields["last_page"]) == ("41", "49")

    def test_last_page_repeating_the_first_keeps_its_own_end(self):
        # APA row where the parser put the printed "118–25" whole in last_page;
        # finalize then expands "25" → "125".
        seg = "Smith, J. (2014). Title. Am J Prev Med, 51, 118–25."
        out = _finalize(seg, first_page="118", last_page="118–25")
        assert (out["first_page"], out["last_page"]) == ("118", "125")

    def test_lumped_range_that_disagrees_with_first_page_is_left_alone(self):
        fields = {"first_page": "100", "last_page": "118–25"}
        _backfill_page_range(fields, None)
        assert (fields["first_page"], fields["last_page"]) == ("100", "118–25")


class TestFillOnly:
    def test_populated_pages_survive_a_different_printed_range(self):
        seg = "Smith, J. (2014). Title. Virulence, 5(1), 20-30."
        out = _finalize(seg, first_page="20", last_page="26")
        assert (out["first_page"], out["last_page"]) == ("20", "26")

    def test_single_printed_page_stays_single(self):
        seg = "Sprung CL. Title. N Engl J Med. 2015;373(9):880."
        out = _finalize(seg, first_page="880", last_page=None)
        assert out["last_page"] is None

    def test_article_number_stays_single(self):
        seg = "Doe, A. (2020). Title. PLoS ONE, 15(3), e0123456."
        out = _finalize(seg, first_page="e0123456", last_page=None)
        assert out["last_page"] is None
