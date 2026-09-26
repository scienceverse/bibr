"""The shared reference regexes live in a neutral module (relocated out of the
retired lead-reference-recovery band-aid)."""

from __future__ import annotations

import pytest

from bibr.ocr.ref_patterns import (
    _AUTHOR_DATE_START,
    _COVERED_MAX_MISSING,
    _REF_HEADER_RE,
    _YEAR,
    _missing_chars,
    alnum_key,
    alnum_text_covered,
)
from tests.reference_fixtures import BODY_TEXT, REFERENCE_LIST, read_again


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


_CONTINUATION = "Psychological Review, 94(2), 115-147."
_DOI = "https://doi.org/10.1037/0033-2909.117.3.497"
_NOISY_LIST = [read_again(entry) for entry in REFERENCE_LIST]


@pytest.mark.parametrize(
    ("needle", "haystack", "covered"),
    [
        # an aggregate box's read against the reads of its entry boxes
        ("\n".join(REFERENCE_LIST), "".join(REFERENCE_LIST[:2] + REFERENCE_LIST[3:]), False),
        (
            "\n".join([REFERENCE_LIST[0], f"{REFERENCE_LIST[1]} {_DOI}", *REFERENCE_LIST[2:]]),
            "".join(REFERENCE_LIST),
            False,
        ),
        ("\n".join(_NOISY_LIST), "".join(REFERENCE_LIST), True),
        ("\n".join(REFERENCE_LIST), "".join(_NOISY_LIST), True),
        # a reference region's read against a page of text regions
        (
            "\n".join([_CONTINUATION, *_NOISY_LIST]),
            BODY_TEXT + " ".join(REFERENCE_LIST) + BODY_TEXT,
            False,
        ),
        (
            "\n".join([_CONTINUATION, *_NOISY_LIST]),
            BODY_TEXT + _CONTINUATION + " ".join(REFERENCE_LIST) + BODY_TEXT,
            True,
        ),
    ],
    ids=[
        "one-entry-of-twelve-missing",
        "doi-missing",
        "needle-read-noisy",
        "haystack-read-noisy",
        "continued-reference-missing-on-page",
        "continued-reference-on-page",
    ],
)
def test_alnum_text_covered_keeps_a_long_needle_holding_text_the_haystack_lacks(
    needle, haystack, covered
):
    from rapidfuzz import fuzz

    needle, haystack = alnum_key(needle), alnum_key(haystack)
    # The fuzzy score alone would take every one of these as covered.
    assert max(fuzz.ratio(needle, haystack), fuzz.partial_ratio(needle, haystack)) >= 95
    assert alnum_text_covered(needle, haystack) is covered


def test_missing_chars_do_not_count_letters_matched_by_chance():
    # The middle of the needle differs from the haystack in every other
    # character. The single characters an alignment pairs up there are not a
    # copy of it.
    needle = "smithjohnson2020" + "abcdefghijklmnopqrstuvwxyz0123" + "journalofthings"
    haystack = "smithjohnson2020" + "aqcqeqgqiqkqmqoqqqsquqwqyq0q2q" + "journalofthings"

    assert _missing_chars(needle, haystack) >= _COVERED_MAX_MISSING


def test_missing_chars_align_a_noisy_edge_with_its_copy_not_the_text_after_it():
    entry = (
        "Meier, F., Fenner, D., Grassmann, T., Otto, M., & Scherer, D. (2017). "
        "Crowdsourcing air temperature from citizen weather stations for"
    )
    needle = alnum_key(entry.replace("weather", "vveather"))
    # Spread letter by letter over the body text after the copy, the noisy end
    # "vveatherstationsfor" costs the aligner less than matched to its copy.
    haystack = alnum_key(entry + " " + BODY_TEXT)

    assert _missing_chars(needle, haystack) < _COVERED_MAX_MISSING


def test_missing_chars_widen_a_window_that_cuts_off_a_longer_copy():
    text = " ".join(REFERENCE_LIST[7:9])
    needle = alnum_key(text)
    # OCR noise made the copy longer, and the window where the needle was
    # found ends before the copy does.
    copy = alnum_key(text.replace("m", "rn").replace("w", "vv"))
    body = alnum_key(BODY_TEXT)
    haystack = body + copy + body
    start = len(body)

    assert _missing_chars(needle, haystack, start, start + len(needle) - 12) < (
        _COVERED_MAX_MISSING
    )
