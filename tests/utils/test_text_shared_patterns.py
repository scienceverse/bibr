"""Unit tests for the shared whitespace/year patterns in bibr.utils.text."""

from bibr.utils.text import YEARISH_RE, collapse_ws


def test_collapse_ws_collapses_all_whitespace_runs():
    assert collapse_ws("a  b\tc\nd") == "a b c d"


def test_collapse_ws_collapses_nbsp_and_crlf_and_strips_ends():
    # \xa0 (NBSP) and \r\n must collapse like any other whitespace run.
    assert collapse_ws("  x\xa0\xa0y\r\nz  ") == "x y z"


def test_collapse_ws_empty_string():
    assert collapse_ws("") == ""


def test_collapse_ws_normalizes_unicode_to_nfc():
    assert collapse_ws("Mu\u0308ller") == "Müller"


def test_yearish_matches_year_with_optional_suffix():
    assert YEARISH_RE.search("Smith, J. 2020a. Title.") is not None
    assert YEARISH_RE.search("published in 1850") is not None  # 1600-1999 via 1[6-9]\d{2}


def test_yearish_matches_nd_inpress_forthcoming_case_insensitive():
    assert YEARISH_RE.search("(n.d.)") is not None
    assert YEARISH_RE.search("Jones (IN PRESS)") is not None
    assert YEARISH_RE.search("forthcoming") is not None


def test_yearish_rejects_text_without_year_token():
    assert YEARISH_RE.search("a short heading with no year") is None
