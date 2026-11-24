"""Unit tests for ISO 7064 ORCID checksum validation and AuthorOrcidHarvester."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bibr.extract.author_orcid_harvester import (
    AuthorOrcidHarvester,
    collect_printed_orcid_lines,
)
from bibr.models import canonicalize_orcid, is_valid_orcid_checksum
from bibr.paper import PaperAuthor
from bibr.paper_contents import CanonicalSection, PaperSection, PaperSentence


def _make_contents(
    sections_spec: list[tuple[CanonicalSection, str, list[str]]],
    *,
    pdf_uri_links: list[dict] | None = None,
    ref_page_lines: list[dict] | None = None,
):
    sections: list[PaperSection] = []
    sentences: list[PaperSentence] = []
    tid = 1
    for sid, (stype, header, texts) in enumerate(sections_spec, start=1):
        sections.append(
            PaperSection(
                section_id=sid,
                header=header,
                level=1,
                parent_section_id=None,
                section_type=stype,
            )
        )
        for txt in texts:
            sentences.append(
                PaperSentence(
                    text_id=tid,
                    text=txt,
                    section_id=sid,
                    paragraph_id=tid,
                    page_number=1,
                )
            )
            tid += 1
    return SimpleNamespace(
        sections=sections,
        sentences=sentences,
        pdf_uri_links=pdf_uri_links or [],
        ref_page_lines=ref_page_lines or [],
    )


def _author(aid: int, given: str, family: str, *, orcid: str | None = None) -> PaperAuthor:
    return PaperAuthor(author_id=aid, given=given, family=family, affiliation="", orcid=orcid)


# ---------------------------------------------------------------------------
# ISO 7064 MOD 11-2 checksum validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("orcid", "expected"),
    [
        ("0000-0002-1825-0097", True),  # canonical ORCID example
        ("0000-0001-5109-3700", True),
        ("0000-0003-3786-872X", True),  # valid X check digit
        ("https://orcid.org/0000-0003-3786-872x", True),  # lowercase x normalized
        ("0000-0001-2345-6780", False),  # invalid check digit
        ("0000-0002-1825-0098", False),  # wrong final digit
        ("not-an-orcid", False),
        (None, False),
    ],
)
def test_is_valid_orcid_checksum(orcid, expected):
    assert is_valid_orcid_checksum(orcid) is expected


def test_canonicalize_orcid_stays_format_only_for_synthetic_ids():
    # Existing tests rely on canonicalize_orcid accepting synthetic 16-digit IDs.
    assert (
        canonicalize_orcid("https://orcid.org/0000-0001-2345-6780")
        == "https://orcid.org/0000-0001-2345-6780"
    )
    assert is_valid_orcid_checksum("0000-0001-2345-6780") is False


# ---------------------------------------------------------------------------
# Printed-text harvesting
# ---------------------------------------------------------------------------


def test_harvests_wrapped_back_matter_orcids_and_disambiguates_shared_surnames():
    contents = _make_contents(
        [
            (CanonicalSection.TITLE, "Title", ["Study of Health"]),
            (
                CanonicalSection.FOOTNOTE,
                "ORCID",
                [
                    "Omid Dadras http://orcid.org/0000-0001-\n9385-2170",
                    "Fateme Dadras http://orcid.org/0000-0003-\n3786-872X",
                    "Takeo Nakayama https://orcid.org/0000-0002-7918-6252",
                ],
            ),
        ]
    )
    authors = [
        _author(1, "Omid", "Dadras"),
        _author(2, "Fateme", "Dadras"),
        _author(3, "Takeo", "Nakayama"),
    ]

    attached = AuthorOrcidHarvester(contents).harvest(authors)

    assert attached == 3
    assert [a.orcid for a in authors] == [
        "https://orcid.org/0000-0001-9385-2170",
        "https://orcid.org/0000-0003-3786-872X",
        "https://orcid.org/0000-0002-7918-6252",
    ]


def test_harvests_initials_prefixed_orcids_royal_society_layout():
    contents = _make_contents(
        [
            (
                CanonicalSection.TITLE,
                "Title",
                [
                    "Alisa Viitanen, Aada Ståhl, Milla K. Salonen, Océane Liehrmann",
                    "ORCID: MKS, 0000-0002-1825-0097; OL, 0000-0001-5109-3700",
                ],
            ),
        ]
    )
    authors = [
        _author(1, "Alisa", "Viitanen"),
        _author(2, "Aada", "Ståhl"),
        _author(3, "Milla K.", "Salonen"),
        _author(4, "Océane", "Liehrmann"),
    ]

    attached = AuthorOrcidHarvester(contents).harvest(authors)

    assert attached == 2
    assert [a.orcid for a in authors] == [
        None,
        None,
        "https://orcid.org/0000-0002-1825-0097",
        "https://orcid.org/0000-0001-5109-3700",
    ]


def test_harvests_orcid_when_author_name_is_on_preceding_line():
    contents = _make_contents(
        [
            (
                CanonicalSection.UNKNOWN,
                "Author ORCIDs",
                [
                    "Jane Doe",
                    "https://orcid.org/0000-0002-1825-0097",
                    "John Smith",
                    "https://orcid.org/0000-0001-5109-3700",
                ],
            )
        ]
    )
    authors = [_author(1, "Jane", "Doe"), _author(2, "John", "Smith")]

    assert AuthorOrcidHarvester(contents).harvest(authors) == 2
    assert [a.orcid for a in authors] == [
        "https://orcid.org/0000-0002-1825-0097",
        "https://orcid.org/0000-0001-5109-3700",
    ]


def test_skips_reference_sections_editorial_lines_and_invalid_checksums():
    contents = _make_contents(
        [
            (
                CanonicalSection.UNKNOWN,
                "Front",
                [
                    "Edited by: Jane Doe, https://orcid.org/0000-0002-1825-0097",
                    "Reviewed by: John Smith, https://orcid.org/0000-0001-5109-3700",
                    "Jane Doe https://orcid.org/0000-0001-2345-6780",  # invalid checksum
                ],
            ),
            (
                CanonicalSection.REFERENCES,
                "References",
                ["Jane Doe (2025). Dataset. https://orcid.org/0000-0003-3786-872X"],
            ),
        ]
    )
    authors = [_author(1, "Jane", "Doe"), _author(2, "John", "Smith")]

    assert AuthorOrcidHarvester(contents).harvest(authors) == 0
    assert [a.orcid for a in authors] == [None, None]


def test_never_overwrites_existing_orcid_or_duplicates_id():
    contents = _make_contents(
        [
            (
                CanonicalSection.UNKNOWN,
                "ORCID",
                [
                    "Jane Doe https://orcid.org/0000-0002-1825-0097",
                    "John Smith https://orcid.org/0000-0002-1825-0097",
                ],
            )
        ]
    )
    authors = [
        _author(1, "Jane", "Doe", orcid="https://orcid.org/0000-0003-3786-872X"),
        _author(2, "John", "Smith"),
    ]

    # Jane keeps her existing ORCID; John can receive the unclaimed printed ORCID.
    assert AuthorOrcidHarvester(contents).harvest(authors) == 1
    assert authors[0].orcid == "https://orcid.org/0000-0003-3786-872X"
    assert authors[1].orcid == "https://orcid.org/0000-0002-1825-0097"


# ---------------------------------------------------------------------------
# PDF URI-link annotation harvesting
# ---------------------------------------------------------------------------


def test_harvests_from_pdf_uri_links_across_multi_author_line_and_merges_split_rects():
    # Multi-author byline on page 1:
    # "Alice Vance [icon1], Bob Mercer [icon2]"
    # Nature Communications often splits a single ORCID icon into several tiny rects.
    page_lines = [
        {
            "page": 1,
            "bbox": [100.0, 200.0, 700.0, 220.0],
            "text": "Alice Vance , Bob Mercer ",
        }
    ]
    uri_links = [
        # Two split rects for Alice's icon around x=380
        {
            "page": 1,
            "bbox": [375.0, 200.0, 385.0, 218.0],
            "uri": "https://orcid.org/0000-0002-1825-0097",
        },
        {
            "page": 1,
            "bbox": [384.0, 200.0, 392.0, 218.0],
            "uri": "https://orcid.org/0000-0002-1825-0097",
        },
        # Bob's icon at right end of line
        {
            "page": 1,
            "bbox": [690.0, 200.0, 705.0, 218.0],
            "uri": "http://orcid.org/0000-0001-5109-3700",
        },
    ]
    contents = _make_contents([], pdf_uri_links=uri_links, ref_page_lines=page_lines)
    authors = [_author(1, "Alice", "Vance"), _author(2, "Bob", "Mercer")]

    assert AuthorOrcidHarvester(contents).harvest(authors) == 2
    assert [a.orcid for a in authors] == [
        "https://orcid.org/0000-0002-1825-0097",
        "https://orcid.org/0000-0001-5109-3700",
    ]


def test_uri_link_harvester_skips_editorial_sidebar_links():
    page_lines = [
        {"page": 1, "bbox": [50.0, 300.0, 250.0, 315.0], "text": "EDITED BY"},
        {"page": 1, "bbox": [50.0, 318.0, 250.0, 333.0], "text": "Alice Vance"},
    ]
    uri_links = [
        {
            "page": 1,
            "bbox": [252.0, 318.0, 265.0, 333.0],
            "uri": "https://orcid.org/0000-0002-1825-0097",
        }
    ]
    contents = _make_contents([], pdf_uri_links=uri_links, ref_page_lines=page_lines)
    authors = [_author(1, "Alice", "Vance")]

    assert AuthorOrcidHarvester(contents).harvest(authors) == 0
    assert authors[0].orcid is None


# ---------------------------------------------------------------------------
# collect_printed_orcid_lines
# ---------------------------------------------------------------------------


def test_collect_printed_orcid_lines_excludes_references_and_already_seen():
    contents = _make_contents(
        [
            (
                CanonicalSection.UNKNOWN,
                "ORCID",
                [
                    "Jane Doe",
                    "https://orcid.org/0000-0002-1825-0097",
                    "John Smith https://orcid.org/0000-0001-5109-3700",
                ],
            ),
            (
                CanonicalSection.REFERENCES,
                "References",
                ["Cited Author https://orcid.org/0000-0003-3786-872X"],
            ),
        ]
    )
    lines = collect_printed_orcid_lines(
        contents,
        already="John Smith 0000-0001-5109-3700",
    )

    assert lines == ["Jane Doe https://orcid.org/0000-0002-1825-0097"]


def test_unmatched_named_orcid_line_does_not_steal_previous_author():
    # If an author (Andrea M. Philipp) was omitted from the extracted author list,
    # her printed ORCID line must NOT walk up two lines and attach to Wolfgang Scharke.
    contents = _make_contents(
        [
            (
                CanonicalSection.UNKNOWN,
                "Author Affiliations",
                [
                    "Wolfgang Scharke Institute of Psychology",
                    "Iring Koch orcid.org/0000-0001-9664-8220 Institute of Psychology",
                    "Andrea M. Philipp orcid.org/0000-0003-1290-451X Institute of Psychology",
                ],
            )
        ]
    )
    authors = [
        _author(1, "Wolfgang", "Scharke"),
        _author(2, "Iring", "Koch"),
    ]

    assert AuthorOrcidHarvester(contents).harvest(authors) == 1
    assert authors[0].orcid is None
    assert authors[1].orcid == "https://orcid.org/0000-0001-9664-8220"
