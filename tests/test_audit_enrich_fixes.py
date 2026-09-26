"""Enrichment audit fixes: fingerprint gates, DOI coherence, completeness.

M18 / M19 / M20 plus L6 (dash-separated page ranges) and L7 (a resolver
response whose ``candidates`` is null).
"""

import pytest

from bibr.enrich.references import (
    _doi_agrees,
    _score_fingerprint,
    _split_page_range,
)
from bibr.schemas import PaperReference


def _ref(**kw):
    base = {
        "bib_id": 1,
        "title": "",
        "authors": "Smith, J., Doe, A.",
        "year": 2020,
        "container": "Nature",
        "volume": "12",
        "first_page": "100",
        "doi": None,
    }
    base.update(kw)
    return PaperReference(**base)


def _raw(**kw):
    base = {
        "DOI": "10.1000/x",
        "title": ["Some Work"],
        "author": [{"family": "Smith", "given": "J"}],
        "container-title": ["Nature"],
        "volume": "12",
        "page": "100-110",
        "issued": {"date-parts": [[2020]]},
    }
    base.update(kw)
    return base


class TestFingerprintGates:
    """M19: gates were weaker than their own docstring claimed."""

    def test_first_author_match_still_scores(self):
        assert _score_fingerprint(_ref(), _item(_raw())) is not None

    def test_a_mere_co_author_no_longer_clears_the_author_gate(self):
        """On a large collaboration, "any co-author" is nearly free."""
        collaboration = _raw(
            author=[{"family": f"Author{i}"} for i in range(200)] + [{"family": "Doe"}]
        )
        assert _score_fingerprint(_ref(), _item(collaboration)) is None

    def test_a_journal_is_not_its_own_family_member(self):
        """token_set_ratio scored "Nature" against "Nature Communications" 100."""
        assert (
            _score_fingerprint(
                _ref(), _item(_raw(**{"container-title": ["Nature Communications"]}))
            )
            is None
        )

    def test_the_exact_container_still_matches(self):
        assert (
            _score_fingerprint(_ref(), _item(_raw(**{"container-title": ["Nature"]}))) is not None
        )


class TestDoiCoherence:
    """M18: an unparseable printed DOI killed the fallback outright."""

    def test_an_agreeing_doi_keeps_only_the_agreeing_candidate(self):
        assert [
            doi for doi in ("10.1000/x", "10.1000/y", None) if _doi_agrees("10.1000/x", doi)
        ] == ["10.1000/x"]

    def test_an_unparseable_printed_doi_no_longer_discards_everything(self):
        """These are exactly the refs most in need of a fallback search."""
        assert [
            doi for doi in ("10.1000/x", "10.1000/y") if _doi_agrees("10 .1O00/garbled-by-ocr", doi)
        ] == ["10.1000/x", "10.1000/y"]

    def test_no_printed_doi_passes_everything_through(self):
        assert _doi_agrees(None, "10.1000/x") is True


class TestPageRangeSplitting:
    """L6: Crossref deposits whichever dash the publisher used."""

    @pytest.mark.parametrize(
        ("page", "expected"),
        [
            ("123-130", ("123", "130")),
            ("123–130", ("123", "130")),  # en dash
            ("123—130", ("123", "130")),  # em dash
            ("123 – 130", ("123", "130")),
            ("123--130", ("123", "130")),
            ("e12345", ("e12345", None)),
            ("", (None, None)),
            (None, (None, None)),
        ],
    )
    def test_split(self, page, expected):
        assert _split_page_range(page) == expected


def _item(raw):
    from bibr.enrich.references import CrossrefWorkItem

    return CrossrefWorkItem.from_raw(raw)
