"""find_text_links: the links printed in running text — URLs with a scheme,
URLs printed without one, and bare DOIs."""

import pytest

from bibr.utils.text import find_text_links


def _found(text, **kwargs):
    return [(link.url, link.link_text) for link in find_text_links(text, **kwargs)]


@pytest.mark.parametrize(
    "text,expected",
    [
        # a repository host papers print without a scheme
        ("Materials are at osf.io/s6bjp.", [("https://osf.io/s6bjp", "osf.io/s6bjp")]),
        (
            "Code: github.com/org/repo_name (v1.2).",
            [("https://github.com/org/repo_name", "github.com/org/repo_name")],
        ),
        # a subdomain of one
        (
            "See gist.github.com/user/abc123, last updated in May.",
            [("https://gist.github.com/user/abc123", "gist.github.com/user/abc123")],
        ),
        # any host under a common top-level domain, with a path
        (
            "Browse amigo.geneontology.org/amigo for terms.",
            [("https://amigo.geneontology.org/amigo", "amigo.geneontology.org/amigo")],
        ),
        (
            "the survey (data.example.edu/files/wave2) was",
            [("https://data.example.edu/files/wave2", "data.example.edu/files/wave2")],
        ),
        # the query string belongs to it, the sentence's full stop does not
        (
            "see osf.io/abc12/files/x.csv?view_only=123abc.",
            [
                (
                    "https://osf.io/abc12/files/x.csv?view_only=123abc",
                    "osf.io/abc12/files/x.csv?view_only=123abc",
                )
            ],
        ),
        # a DOI behind the resolver's host stays one link
        (
            "doi.org/10.1234/abc.def",
            [("https://doi.org/10.1234/abc.def", "doi.org/10.1234/abc.def")],
        ),
    ],
)
def test_url_printed_without_a_scheme(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        # abbreviations and slashed alternatives
        "e.g., this and i.e./or that",
        "U.S./Canada and Ph.D./M.D.",
        "Smith.Jones/2020 and et al./2019",
        # file names and paths
        "stored in data.csv/raw and results.json/x",
        "run analysis.py/main, see README.md/usage",
        "the module at src/utils.com/x",
        # numbers
        "version 1.2.3/4 of R, and 3.10/2",
        "p = 0.05/2 and 10.5/3",
        # an email address
        "contact a.b@osf.io/x or jane@uni.edu/staff",
        # a top-level domain that is also a word or a statistic
        "with beta.se/sd ratios",
        # a host with no path
        "registered at aspredicted.org before the data were collected",
        "test.org/",
        # the middle of a URL the typesetter broke with a zero-width space
        "www​.example​.de/Path",
    ],
)
def test_not_a_link(text):
    assert _found(text) == []


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "deposited (doi:10.6084/m9.figshare.1234567).",
            [("https://doi.org/10.6084/m9.figshare.1234567", "10.6084/m9.figshare.1234567")],
        ),
        (
            "(DOI: 10.15154/z563-zd24)",
            [("https://doi.org/10.15154/z563-zd24", "10.15154/z563-zd24")],
        ),
        (
            "available at 10.5281/zenodo.7654321.",
            [("https://doi.org/10.5281/zenodo.7654321", "10.5281/zenodo.7654321")],
        ),
        # balanced parentheses belong to the DOI
        (
            "doi: 10.1016/S0140-6736(16)00427-X.",
            [("https://doi.org/10.1016/S0140-6736(16)00427-X", "10.1016/S0140-6736(16)00427-X")],
        ),
        # a typeset hyphen is read as "-"; link_text keeps what is printed
        (
            "see doi:10.15154/z563‐zd24 for the data",
            [("https://doi.org/10.15154/z563-zd24", "10.15154/z563‐zd24")],
        ),
        # a slash after the DOI separates it from the next word
        (
            "see 10.5281/zenodo.123/ and",
            [("https://doi.org/10.5281/zenodo.123", "10.5281/zenodo.123")],
        ),
        # a funder's DOI is a number and ends with it
        (
            "funded by MICIU/AEI/10.13039/501100011033/FEDER, UE",
            [("https://doi.org/10.13039/501100011033", "10.13039/501100011033")],
        ),
    ],
)
def test_bare_doi(text, expected):
    assert _found(text) == expected


def test_doi_url_is_one_link():
    # The DOI inside a doi.org URL is not found a second time.
    text = "available at https://doi.org/10.5281/zenodo.7654321 and"
    assert _found(text) == [("https://doi.org/10.5281/zenodo.7654321", None)]


@pytest.mark.parametrize(
    "text,expected",
    [
        # two URLs printed with no space between them
        (
            "see https://github.com/a/b):https://doi.org/10.5281/zenodo.1234567.",
            ["https://github.com/a/b", "https://doi.org/10.5281/zenodo.1234567"],
        ),
        # a scheme after "/" or "=" is part of the first URL
        (
            "https://web.archive.org/web/2020/https://example.org/a",
            ["https://web.archive.org/web/2020/https://example.org/a"],
        ),
        (
            "https://example.org/go?url=https://osf.io/abcde",
            ["https://example.org/go?url=https://osf.io/abcde"],
        ),
    ],
)
def test_glued_urls(text, expected):
    assert [link.url for link in find_text_links(text)] == expected
    assert [link.url for link in find_text_links(text, bare=False)] == expected


def test_tail_of_a_wrapped_url_is_not_a_second_link():
    # The line wrap left a space inside the URL; what follows is its remainder.
    text = "https://www.example.com/long- part.org/rest was cut"
    assert [link.url for link in find_text_links(text)] == ["https://www.example.com/long-"]


def test_links_come_in_reading_order_with_their_span():
    text = "Data: osf.io/s6bjp, code: https://github.com/a/b, archive: doi:10.5281/zenodo.99."
    links = find_text_links(text)
    assert [link.url for link in links] == [
        "https://osf.io/s6bjp",
        "https://github.com/a/b",
        "https://doi.org/10.5281/zenodo.99",
    ]
    assert [text[link.start : link.end] for link in links] == [
        "osf.io/s6bjp",
        "https://github.com/a/b",
        "10.5281/zenodo.99",
    ]


def test_bare_false_keeps_only_urls_written_in_full():
    text = "Smith J. A paper. doi:10.1234/abc.5678. github.com/a/b. https://example.org/x"
    assert _found(text, bare=False) == [("https://example.org/x", None)]
