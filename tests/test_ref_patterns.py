"""The shared reference regexes live in a neutral module (relocated out of the
retired lead-reference-recovery band-aid)."""

from __future__ import annotations

from bibr.ocr.ref_patterns import _AUTHOR_DATE_START, _REF_HEADER_RE, _YEAR


def test_ref_header_matches_common_headers():
    assert _REF_HEADER_RE.match("References")
    assert _REF_HEADER_RE.match("BIBLIOGRAPHY")
    assert _REF_HEADER_RE.match("Literature Cited")
    assert not _REF_HEADER_RE.match("Reference list of things")


def test_ref_header_matches_multilingual_and_decorated_headers():
    for header in (
        "Referencias bibliográficas",
        "Références bibliographiques",
        "Literatur",
        "Список використаних джерел",
        "Daftar Pustaka",
        "Kaynakça",
        "引用文献",
        "Bibliografi",
        "5. REFERENCES",
        "IV. Bibliography",
        "References:",
        "■ References",
    ):
        assert _REF_HEADER_RE.match(header), header


def test_ref_header_still_rejects_prose_and_running_text():
    for text in (
        "Reference list of things",
        "References to the literature are given below",
        "See bibliography for details",
        "1. Introduction",
    ):
        assert not _REF_HEADER_RE.match(text), text


def test_author_date_start_matches_author_comma():
    assert _AUTHOR_DATE_START.match("Carstensen, L. L.")
    assert not _AUTHOR_DATE_START.match("the quick brown fox")


def test_author_date_start_matches_curly_apostrophe_surname():
    # U+2019 curly apostrophe in the surname must still match (verbatim-copy guard)
    assert _AUTHOR_DATE_START.match("D’Angelo, M.")


def test_year_matches_years_and_in_press():
    assert _YEAR.search("(2018)")
    assert _YEAR.search("published in 1999b")
    assert _YEAR.search("(in press)")
    assert not _YEAR.search("no year here")
