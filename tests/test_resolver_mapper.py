from bibr.enrich.references import _build_match_from_candidate
from bibr.models import migrate_bib_type


def test_build_match_from_candidate_full():
    cand = {
        "title": "T",
        "source": "openalex",
        "doi": "10.1/x",
        "id": "W1",
        "authors": [{"given": "A", "family": "Smith"}, {"given": "", "family": ""}],
        "editors": [{"given": "E", "family": "Ed"}],
        "year": 2020,
        "container": "J",
        "volume": "5",
        "issue": "2",
        "first_page": "10",
        "last_page": "20",
        "publisher": "P",
        "type": "journal-article",
        "url": "http://x",
        "date": "2020-06",
    }
    m = _build_match_from_candidate(cand, 88.5)
    assert m.score == 88.5
    assert m.title == "T"
    assert m.doi == "10.1/x"
    assert m.id == "10.1/x"  # doi preferred for id
    assert [a.family for a in m.authors] == ["Smith"]  # empty-family entry filtered
    assert [e.family for e in m.editors] == ["Ed"]
    assert m.year == 2020
    assert m.container == "J"
    assert m.volume == "5" and m.issue == "2"
    assert m.first_page == "10" and m.last_page == "20"
    assert m.publisher == "P"
    assert m.url == "http://x" and m.date == "2020-06"
    assert m.bib_type == migrate_bib_type("journal-article")
    assert m.edition is None and m.version is None


def test_build_match_from_candidate_no_doi_uses_id():
    m = _build_match_from_candidate({"title": "T", "id": "W9"}, 90.0)
    assert m.id == "W9"
    assert m.doi is None


def test_build_match_type_to_bib_type_rename():
    m = _build_match_from_candidate({"title": "T", "type": "book"}, 90.0)
    assert m.bib_type == migrate_bib_type("book")


def test_build_match_no_type_leaves_bib_type_none():
    m = _build_match_from_candidate({"title": "T"}, 90.0)
    assert m.bib_type is None
    assert m.authors is None
