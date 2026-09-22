"""Normalized companions for printed metadata values."""

from __future__ import annotations

import pytest

from bibr.export.normalize import CREDIT_BASE_URI, arxiv_id, credit_roles, iso_date, license_ids


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        ("2026-01-15", "2026-01-15"),
        ("2026-1-5T10:00:00Z", "2026-01-05"),
        ("2026/01/15", "2026-01-15"),
        ("15 January 2026", "2026-01-15"),
        ("Published: 3rd Feb. 2021", "2021-02-03"),
        ("January 15, 2026", "2026-01-15"),
        ("2026 Jan 15", "2026-01-15"),
        ("2020, May 3", "2020-05-03"),  # APA reference dates
        ("(2020, Jan.)", "2020-01"),
        ("2020, Spring", None),
        ("September 2019", "2019-09"),
        ("2019", "2019"),
        ("Received 2 March 2020; Accepted 9 May 2020; Published online 1 June 2020", "2020-06-01"),
        ("Received 2 March 2020; Accepted 9 May 2020", None),  # two dates, no cue
        ("in press", None),
        ("2026-13-01", None),
        (None, None),
    ],
)
def test_iso_date(printed, expected):
    assert iso_date(printed) == expected


@pytest.mark.parametrize(
    ("printed", "url", "spdx"),
    [
        ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/", "CC-BY-4.0"),
        (
            "CC-BY-NC-ND 4.0",
            "https://creativecommons.org/licenses/by-nc-nd/4.0/",
            "CC-BY-NC-ND-4.0",
        ),
        (
            "This article is licensed under a Creative Commons Attribution-NonCommercial 4.0 "
            "International License.",
            "https://creativecommons.org/licenses/by-nc/4.0/",
            "CC-BY-NC-4.0",
        ),
        (
            "Open access under http://creativecommons.org/licenses/by-sa/3.0/.",
            "http://creativecommons.org/licenses/by-sa/3.0/",
            "CC-BY-SA-3.0",
        ),
        ("Released under CC0.", "https://creativecommons.org/publicdomain/zero/1.0/", "CC0-1.0"),
        ("CC BY", None, None),  # version unknown: no SPDX id, no canonical URL
        ("© 2020 The Authors. All rights reserved.", None, None),
        ("See https://example.org/license", "https://example.org/license", None),
        (None, None, None),
    ],
)
def test_license_ids(printed, url, spdx):
    assert license_ids(printed) == (url, spdx)


def test_credit_roles_map_term_names_and_common_phrasings_in_credit_order():
    roles = ["Writing – review & editing", "wrote the manuscript", "Conceptualization", "tea"]
    assert credit_roles(roles) == [
        CREDIT_BASE_URI + "conceptualization/",
        CREDIT_BASE_URI + "writing-original-draft/",
        CREDIT_BASE_URI + "writing-review-editing/",
    ]
    assert credit_roles([]) == []


def test_arxiv_id_from_doi_or_the_page_one_stamp():
    assert arxiv_id("10.48550/arXiv.2101.12345", []) == "2101.12345"
    assert arxiv_id(None, ["arXiv:2101.12345v2 [cs.CL] 3 Feb 2021"]) == "2101.12345"
    # A cited arXiv ID without the category stamp is not the paper's own.
    assert arxiv_id("10.1234/x", ["see arXiv:1706.03762 for details"]) is None
