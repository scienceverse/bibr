"""Tests for _bridge_doi_url_crlf_wraps: rejoin CRLF wraps inside URL/DOI tokens
at positions the mid-word DOI bridge does not cover (registrant-internal,
after-digit, after-slash, host/scheme-split, www host), without over-joining a
URL into following prose."""

import pytest

from bibr.input.consolidate_text import _bridge_doi_url_crlf_wraps

# (input, expected substring that must be present after bridging)
MUST_JOIN = [
    # registrant-internal: wrap inside the "10" registrant
    ("https://doi.org/1\r\n0.1038/ncprheum0907", "https://doi.org/10.1038/ncprheum0907"),
    ("https://doi.org/1\r\n0.1186/s12891-016-1365-4", "https://doi.org/10.1186/s12891-016-1365-4"),
    # after-digit (mid-number suffix)
    ("https://doi.org/10.1002/acr.21\r\n979", "https://doi.org/10.1002/acr.21979"),
    ("https://doi.org/10.1177/02698811134\r\n95118", "https://doi.org/10.1177/0269881113495118"),
    # left ends mid-registrant then ".016/j…"
    (
        "https://doi.org/10.1\r\n016/j.biopsycho.2015.12.003",
        "https://doi.org/10.1016/j.biopsycho.2015.12.003",
    ),
    # after-slash
    (
        "https://doi.org/10.1093/rheumatology/\r\nket317",
        "https://doi.org/10.1093/rheumatology/ket317",
    ),
    ("https://doi.org/10.1521/\r\npedi_2012_26_049", "https://doi.org/10.1521/pedi_2012_26_049"),
    ("https://doi.org/10.1017/\r\nS0033291706008944", "https://doi.org/10.1017/S0033291706008944"),
    # structural-char continuation ('.', ')')
    (
        "https://doi.org/10.1016/j\r\n.copsyc.2019.07.028",
        "https://doi.org/10.1016/j.copsyc.2019.07.028",
    ),
    (
        "https://doi.org/10.1016/S0140-6736(16\r\n)00427-X",
        "https://doi.org/10.1016/S0140-6736(16)00427-X",
    ),
    (
        "https://doi.org/10.1016/0022-3999(94\r\n)90005-1",
        "https://doi.org/10.1016/0022-3999(94)90005-1",
    ),
    # host/scheme-split
    (
        "https:/\r\n/doi.org/10.1016/j.genhosppsych.2011.03.004",
        "https://doi.org/10.1016/j.genhosppsych.2011.03.004",
    ),
    (
        "http://www.psychologicalscience\r\n.org/publications/badges",
        "http://www.psychologicalscience.org/publications/badges",
    ),
    # www host (letter->letter, has a dot ahead)
    ("https://www.sea\r\nledenvelope.com/", "https://www.sealedenvelope.com/"),
    (
        "http://pss\r\n.sagepub.com/content/by/supplemental-data",
        "http://pss.sagepub.com/content/by/supplemental-data",
    ),
    # host-separator wrap: previous line ends with "." that is a domain-label
    # separator (still inside the host, no path "/" yet), next line starts a
    # domain-ish continuation.
    ("https://blog.\nopenai.com/better-language-models/", "https://blog.openai.com/"),
    ("https://blog.\r\nopenai.com/better-language-models/", "https://blog.openai.com/"),
    ("Visit www.example.\ncom/page for details.", "www.example.com/page"),
]

# (input, substring that must NOT appear after bridging)
MUST_NOT_JOIN = [
    # over-join: URL end -> wrapped author name / next word (capitalized)
    ("https://orcid.org/0000-0002-6139-5144\r\nSara Konrath", "5144Sara"),
    ("https://doi.org/10.1177/09567976231222288\r\nArticle reuse", "288Article"),
    # over-join: URL end -> lowercase prose (no URL structure in the continuation)
    ("https://osf.io/tnygr\r\nand the materials", "tnygrand"),
    # NOT this function's domain: DOI-body letter->letter (left -> handled by the
    # other bridge); must remain unjoined by THIS function.
    ("https://doi.org/10.1146/annurev\r\nclinpsy-032210-104544", "annurevclinpsy"),
    # complete DOI ending ".x" then a page range (left ends in a letter, not a digit)
    ("https://doi.org/10.1111/j.1469-8986.2007.00550.x\r\n1374-1385", "00550.x1374"),
    ("https://doi.org/10.1038/s41593-020-00742-z\r\nverbal report", "00742-zverbal"),
    # sentence-final period after a complete DOI, followed by the next
    # reference's number — must NOT be swallowed into the DOI as a
    # host-separator wrap (in_doi_body excludes it).
    (
        "https://doi.org/10.1177/09567976231222288.\r\n2. Article reuse guidelines",
        "288.2",
    ),
    # sentence-final period after a URL that already reached its path (a "/"
    # segment) — a trailing period there is prose punctuation, not a host
    # separator, so it must NOT join with the next citation number.
    ("https://blog.openai.com/foo.\r\n2. Next citation", "foo.2"),
]


@pytest.mark.parametrize("text,expected", MUST_JOIN)
def test_must_join(text, expected):
    assert expected in _bridge_doi_url_crlf_wraps(text)


@pytest.mark.parametrize("text,forbidden", MUST_NOT_JOIN)
def test_must_not_join(text, forbidden):
    assert forbidden not in _bridge_doi_url_crlf_wraps(text)


def test_crlf_bridge_alone_does_not_close_doi_body_wrap():
    # _bridge_doi_url_crlf_wraps joins the registrant wrap only; the j.bio\npsycho
    # DOI-body wrap is _bridge_doi_midword_wraps's domain (closed end-to-end by
    # fix_ocr_artifacts — see test_fix_ocr_artifacts_double_wrap_full_join).
    out = _bridge_doi_url_crlf_wraps("https://doi.org/10.1\r\n016/j.bio\r\npsycho.2015.12.003")
    assert "https://doi.org/10.1016/j.bio" in out
    assert "j.biopsycho" not in out


def test_non_url_text_untouched():
    # A plain prose line-wrap must not be joined.
    s = "the quick brown\r\nfox jumped over"
    assert _bridge_doi_url_crlf_wraps(s) == s


from bibr.input.consolidate_text import fix_ocr_artifacts


def test_fix_ocr_artifacts_bridges_registrant_wrap():
    out = fix_ocr_artifacts(
        "1. Grahame R. Nat Clin Pract Rheumatol. 2008. https://doi.org/1\r\n0.1038/ncprheum0907."
    )
    assert "https://doi.org/10.1038/ncprheum0907" in out


def test_fix_ocr_artifacts_preserves_existing_jneuron_bridge():
    # The pre-existing mid-word DOI bridge must still work end-to-end.
    out = fix_ocr_artifacts(
        "Neuron, 89(1), 221-234. https://doi.org/10.1016/j.neu\r\nron.2015.11.028"
    )
    assert "10.1016/j.neuron.2015.11.028" in out


def test_fix_ocr_artifacts_double_wrap_full_join():
    # Double-wrapped URL: registrant wrap (10.1->016) + DOI-body wrap (j.bio->psycho).
    # With the crlf bridge running before the midword bridge, the full join is now
    # reachable end-to-end through fix_ocr_artifacts.
    out = fix_ocr_artifacts("https://doi.org/10.1\r\n016/j.bio\r\npsycho.2015.12.003")
    assert "https://doi.org/10.1016/j.biopsycho.2015.12.003" in out
