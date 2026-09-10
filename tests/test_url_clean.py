"""clean_extracted_url: trim prose delimiters a URL regex over-captures while
keeping balanced DOI parens."""

import pytest

from bibr.utils.text import URL_RE, clean_extracted_url


@pytest.mark.parametrize(
    "raw,expected",
    [
        # balanced DOI parens kept
        (
            "https://doi.org/10.1016/S0140-6736(16)00427-X",
            "https://doi.org/10.1016/S0140-6736(16)00427-X",
        ),
        (
            "https://doi.org/10.1016/0022-3999(94)90005-1",
            "https://doi.org/10.1016/0022-3999(94)90005-1",
        ),
        # unbalanced trailing ) from prose stripped
        ("https://example.org/page)", "https://example.org/page"),
        (
            "https://en.wikipedia.org/wiki/Nan_(disambiguation))",
            "https://en.wikipedia.org/wiki/Nan_(disambiguation)",
        ),
        # trailing sentence punctuation stripped
        ("https://doi.org/10.1136/bmj.e7586.", "https://doi.org/10.1136/bmj.e7586"),
        ("https://example.org/a,", "https://example.org/a"),
        # period after a balanced paren DOI, then the paren kept
        (
            "https://doi.org/10.1016/S0140-6736(16)00427-X.",
            "https://doi.org/10.1016/S0140-6736(16)00427-X",
        ),
        # trailing slash preserved (not punctuation)
        ("https://www.sealedenvelope.com/", "https://www.sealedenvelope.com/"),
    ],
)
def test_clean_extracted_url(raw, expected):
    assert clean_extracted_url(raw) == expected


def test_url_re_now_captures_paren_doi():
    # End to end: the regex must capture through the ')', then the cleaner keeps it.
    text = "See https://doi.org/10.1016/S0140-6736(16)00427-X for details."
    m = URL_RE.search(text)
    assert clean_extracted_url(m.group(0)) == "https://doi.org/10.1016/S0140-6736(16)00427-X"


def test_prose_url_in_parens_end_to_end():
    text = "(available at https://example.org/data)"
    m = URL_RE.search(text)
    assert clean_extracted_url(m.group(0)) == "https://example.org/data"
