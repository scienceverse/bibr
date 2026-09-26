from bibr.enrich.references import (
    _families_overlap,
    _score_candidate,
    _TitleCandidate,
)
from bibr.enrich.schemas import CrossrefWorkItem


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
    shape reduce to the same candidate, so the one scorer sees one record."""
    ref_title, ref_year, ref_authors = "Attention Is All You Need", 2017, "Vaswani"
    cr_item = {
        "DOI": "10.1000/attention",
        "title": ["Attention Is <i>All</i> You Need"],
        "issued": {"date-parts": [[2017]]},
        "author": [{"family": "Vaswani", "given": "Ashish"}],
        "container-title": ["Advances in Neural Information Processing Systems"],
        "volume": "30",
        "page": "5998-6008",
        "type": "proceedings-article",
    }
    resolver_cand = {
        "doi": "10.1000/attention",
        "title": "Attention Is <i>All</i> You Need",
        "year": 2017,
        "authors": [{"family": "Vaswani", "given": "Ashish"}],
        "container": "Advances in Neural Information Processing Systems",
        "volume": "30",
        "first_page": "5998",
        "type": "proceedings-article",
    }
    from_crossref = _TitleCandidate.from_crossref(CrossrefWorkItem.from_raw(cr_item))
    from_resolver = _TitleCandidate.from_resolver(resolver_cand)
    assert from_crossref == from_resolver
    assert from_crossref.title == "Attention Is All You Need"
    score = _score_candidate(
        ref_title,
        from_crossref.title,
        ref_year,
        from_crossref.year,
        ref_authors,
        [a.family for a in from_crossref.authors],
    )
    assert score == 100.0
