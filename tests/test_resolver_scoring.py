from bibr.enrich.references import (
    _extract_year_from_item,
    _families_overlap,
    _score_candidate,
)


def test_families_overlap_true():
    assert _families_overlap("Smith, J. and Doe, A.", ["Smith"]) is True


def test_families_overlap_false():
    assert _families_overlap("Smith, J.", ["Nobody"]) is False


def test_families_overlap_skips_short_family():
    assert _families_overlap("S and others", ["S"]) is False  # len < 2 skipped


def test_score_exact_title_no_penalty():
    assert _score_candidate("Deep Learning", "Deep Learning", None, None, None, []) == 100.0


def test_score_year_penalty_halves():
    assert _score_candidate("Deep Learning", "Deep Learning", 2010, 2020, None, []) == 50.0


def test_score_year_within_two_no_penalty():
    assert _score_candidate("Deep Learning", "Deep Learning", 2019, 2020, None, []) == 100.0


def test_score_author_penalty_applies_when_no_overlap():
    # title >= 80, ref_authors present, no family overlap -> x0.7
    assert (
        _score_candidate("Deep Learning", "Deep Learning", None, None, "Smith", ["Jones"]) == 70.0
    )


def test_score_author_no_penalty_with_overlap():
    assert (
        _score_candidate("Deep Learning", "Deep Learning", None, None, "Smith", ["Smith"]) == 100.0
    )


def test_crossref_and_resolver_inputs_score_identically():
    """Regression guard: equivalent records in CrossRef-raw vs resolver-Candidate
    shape produce the same score through the shared scorer."""
    ref_title, ref_year, ref_authors = "Attention Is All You Need", 2017, "Vaswani"
    cr_item = {
        "title": ["Attention Is All You Need"],
        "issued": {"date-parts": [[2017]]},
        "author": [{"family": "Vaswani"}],
    }
    resolver_cand = {
        "title": "Attention Is All You Need",
        "year": 2017,
        "authors": [{"family": "Vaswani"}],
    }
    cr_score = _score_candidate(
        ref_title,
        cr_item["title"][0],
        ref_year,
        _extract_year_from_item(cr_item),
        ref_authors,
        [a.get("family", "") for a in cr_item["author"]],
    )
    rs_score = _score_candidate(
        ref_title,
        resolver_cand["title"],
        ref_year,
        resolver_cand["year"],
        ref_authors,
        [a.get("family", "") for a in resolver_cand["authors"]],
    )
    assert cr_score == rs_score == 100.0
