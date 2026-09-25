"""The shared reference regexes live in a neutral module (relocated out of the
retired lead-reference-recovery band-aid)."""

from __future__ import annotations

import pytest

from bibr.ocr.ref_patterns import (
    _AUTHOR_DATE_START,
    _REF_HEADER_RE,
    _YEAR,
    alnum_key,
    alnum_text_covered,
)


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


_ENTRIES = [
    "Kahneman, D., & Tversky, A. (1979). Prospect theory. Econometrica, 47, 263-291.",
    "Barnard, C. (1938). The functions of the executive. Harvard University Press.",
    "Modern, R. (2001). Governance in modern corporations. Journal of Finance, 12, 1-20.",
    "Dunn, D. (2004). Four short pieces. Journal of Things, 4, 1-9.",
]
_PAGE_TEXT = (
    "Running head: prospects. We report three studies of choice under risk and "
    "discuss how the findings extend earlier accounts of loss aversion. "
    + " ".join(_ENTRIES)
    + " Received 3 May 2021; accepted 9 June 2021."
)


def _noisy(text: str) -> str:
    """A second OCR read of *text*: "rn" for its first "m", one char longer."""
    return text.replace("m", "rn", 1)


@pytest.mark.parametrize(
    ("needle", "haystack", "covered"),
    [
        # the same text read twice, either read the longer one
        ("\n".join(_noisy(e) for e in _ENTRIES), "".join(_ENTRIES), True),
        ("\n".join(_ENTRIES), "".join(_noisy(e) for e in _ENTRIES), True),
        # a noisy read of one entry inside a page's worth of other text
        (_noisy(_ENTRIES[2]), _PAGE_TEXT, True),
        # an aggregate read holding entries its entry boxes do not
        ("\n".join(_ENTRIES), "".join(_ENTRIES[:2]), False),
        ("\n".join(_ENTRIES), "".join(_ENTRIES[:3]), False),
        # short needles must be contained exactly
        ("Image J", "lmage J PubMed", False),
        ("PubMed", "Image J PubMed", True),
    ],
    ids=[
        "needle-read-longer",
        "haystack-read-longer",
        "noisy-substring",
        "two-of-four",
        "three-of-four",
        "short-noisy",
        "short-exact",
    ],
)
def test_alnum_text_covered(needle, haystack, covered):
    assert alnum_text_covered(alnum_key(needle), alnum_key(haystack)) is covered
