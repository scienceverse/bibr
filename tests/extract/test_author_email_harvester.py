"""AuthorEmailHarvester — corresponding-author email enrichment tests."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from bibr.models import PaperAuthor


def _make_contents_with_text(text_lines: list[str]):
    """Build a stub PaperContents-like object exposing sections and sentences as the harvester needs."""
    sentences = [
        SimpleNamespace(text=t, section_id=0, text_id=i, page_number=1)
        for i, t in enumerate(text_lines)
    ]
    sections = [SimpleNamespace(section_id=0, section_type="abstract", header="")]
    df = pd.DataFrame(
        [
            {"text_id": i, "text": t, "page_number": 1, "section_id": 0}
            for i, t in enumerate(text_lines)
        ]
    )
    return SimpleNamespace(
        sentences=sentences,
        sections=sections,
        sentences_df=df,
        detected_headers=[],
        detected_footers=[],
    )


def test_harvest_corresponding_email_from_explicit_anchor():
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        [
            "Title here",
            "Jane Doe, John Smith",
            "Department of X, University of Y",
            "Corresponding author: jane.doe@university.edu",
        ]
    )
    authors = [
        PaperAuthor(author_id=1, given="Jane", family="Doe", affiliation="X"),
        PaperAuthor(author_id=2, given="John", family="Smith", affiliation="X"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    jane = next(a for a in authors if a.family == "Doe")
    assert jane.email == "jane.doe@university.edu"
    assert jane.corresponding is True
    john = next(a for a in authors if a.family == "Smith")
    assert john.corresponding is False


def test_demote_when_all_authors_marked_corresponding():
    """When LLM marks every author corresponding (3+), demote all so the harvester
    can re-promote the anchored one."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(["text"])
    authors = [
        PaperAuthor(author_id=1, given="A", family="One", affiliation="x", corresponding=True),
        PaperAuthor(author_id=2, given="B", family="Two", affiliation="x", corresponding=True),
        PaperAuthor(author_id=3, given="C", family="Three", affiliation="x", corresponding=True),
    ]
    AuthorEmailHarvester(contents).demote_implausible_flags(authors)
    assert all(a.corresponding is False for a in authors)


def test_demote_preserves_single_corresponding_flag():
    """A single corresponding author is plausible — leave it alone."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(["text"])
    authors = [
        PaperAuthor(author_id=1, given="A", family="One", affiliation="x", corresponding=True),
        PaperAuthor(author_id=2, given="B", family="Two", affiliation="x"),
    ]
    AuthorEmailHarvester(contents).demote_implausible_flags(authors)
    assert authors[0].corresponding is True


def test_no_anchor_no_email_assignment():
    """When no 'Corresponding author:' anchor is present, the email is still
    assignable to the matching author (closest-distance rule), but
    corresponding=True is NOT promoted without an anchor (multi-author paper —
    a sole author with an email IS promoted, see the single-author tests)."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        [
            "One sentence first.",
            "Author One discussed the topic.",
            "Random sentence containing one@bar.edu but no anchor.",
        ]
    )
    authors = [
        PaperAuthor(author_id=1, given="A", family="One", affiliation="x"),
        PaperAuthor(author_id=2, given="B", family="Two", affiliation="x"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)
    # Email may or may not be assigned depending on family-window match, but
    # corresponding flag must NOT be promoted without a marker.
    assert authors[0].corresponding is False
    assert authors[1].corresponding is False


def test_correspondence_line_email_excludes_byline_emails_in_window():
    """MDPI layout: every author's email sits in the affiliation block, with the
    '* Correspondence: <email>' line inside the same ±3-sentence window. Only the
    email named ON the marker sentence may be promoted — window proximity must
    not promote the rest (judged: 207607601421051739369458, 6 wrong flags)."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        [
            "Rehabilitation Program Outcomes.",
            "Maria K. Stone Smith, Rosa Carter, Mark Landis and Hanna Clark",
            "Department of Rehabilitation Studies, Example City, USA;"
            " mkstonesmith@uni-a.edu (M.K.S.S.); mark.landis@agency-b.gov (M.L.);"
            " hclark2@students.uni-c.edu (H.C.)",
            "* Correspondence: rcarter@agency-b.gov",
        ]
    )
    authors = [
        PaperAuthor(
            author_id=1,
            given="Maria K.",
            family="Stone Smith",
            affiliation="x",
            email="mkstonesmith@uni-a.edu",
        ),
        PaperAuthor(
            author_id=2,
            given="Rosa",
            family="Carter",
            affiliation="x",
            email="rcarter@agency-b.gov",
        ),
        PaperAuthor(
            author_id=3,
            given="Mark",
            family="Landis",
            affiliation="x",
            email="mark.landis@agency-b.gov",
        ),
        PaperAuthor(
            author_id=4,
            given="Hanna",
            family="Clark",
            affiliation="x",
            email="hclark2@students.uni-c.edu",
        ),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    flags = {a.family: a.corresponding for a in authors}
    assert flags == {
        "Stone Smith": False,
        "Carter": True,
        "Landis": False,
        "Clark": False,
    }


def test_correspondence_line_with_two_emails_promotes_both():
    """Co-corresponding authors listed on the marker sentence are both promoted."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        [
            "Byline emails: a.alpha@x.edu (A.A.); b.beta@y.edu (B.B.); c.gamma@z.edu (C.G.)",
            "* Correspondence: a.alpha@x.edu or b.beta@y.edu",
        ]
    )
    authors = [
        PaperAuthor(
            author_id=1, given="Ann", family="Alpha", affiliation="x", email="a.alpha@x.edu"
        ),
        PaperAuthor(author_id=2, given="Bo", family="Beta", affiliation="x", email="b.beta@y.edu"),
        PaperAuthor(
            author_id=3, given="Cy", family="Gamma", affiliation="x", email="c.gamma@z.edu"
        ),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    assert [a.corresponding for a in authors] == [True, True, False]


def test_marker_without_email_still_falls_back_to_window():
    """Psych-journal layout: 'Corresponding Author:' names the author, the email
    sits in an adjacent sentence. No email shares the marker sentence, so window
    proximity must keep working."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        [
            "Corresponding Author: Jane Doe, Department of X",
            "E-mail: jane.doe@university.edu",
        ]
    )
    authors = [
        PaperAuthor(author_id=1, given="Jane", family="Doe", affiliation="X"),
        PaperAuthor(author_id=2, given="John", family="Smith", affiliation="X"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    jane = next(a for a in authors if a.family == "Doe")
    assert jane.email == "jane.doe@university.edu"
    assert jane.corresponding is True
    assert next(a for a in authors if a.family == "Smith").corresponding is False


def test_single_author_with_email_is_corresponding():
    """A sole author with a contact email is the de-facto corresponding author,
    marker or not (judged: ja.2013-10 envelope glyph, ja.2018-43 byline email)."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        [
            "Trade Policy Reform.",
            "Arun Prasad — ap99@uni-d.edu",
        ]
    )
    authors = [
        PaperAuthor(
            author_id=1,
            given="Arun",
            family="Prasad",
            affiliation="Example University",
            email="ap99@uni-d.edu",
        )
    ]
    AuthorEmailHarvester(contents).harvest(authors)
    assert authors[0].corresponding is True


def test_single_author_without_email_stays_unflagged():
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(["A paper with no contact details at all."])
    authors = [PaperAuthor(author_id=1, given="A", family="One", affiliation="x")]
    AuthorEmailHarvester(contents).harvest(authors)
    assert authors[0].corresponding is False


def test_unrelated_earlier_email_does_not_block_real_one():
    """extract-metadata-8: an editorial-office address printed before the
    corresponding author's own marker-anchored address must not be handed to
    them (by surname-window proximity), blocking the real one."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        [
            "Jane Doe and John Roe",
            "Department of X",
            "Editorial office: editor@appliedthings.org",
            "Filler sentence one.",
            "Filler sentence two.",
            "Corresponding author: Jane Doe, jane.doe@uni.edu",
        ]
    )
    authors = [
        PaperAuthor(author_id=1, given="Jane", family="Doe", affiliation="X", corresponding=True),
        PaperAuthor(author_id=2, given="John", family="Roe", affiliation="X"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    jane = next(a for a in authors if a.family == "Doe")
    assert jane.email == "jane.doe@uni.edu"
    assert jane.corresponding is True
    assert next(a for a in authors if a.family == "Roe").email in (None, "")


def test_elimination_fallback_needs_marker_or_name_match():
    """extract-metadata-8: with no surname near any email (wide gap), the pure
    elimination fallback must not hand an unrelated address to the single
    corresponding author."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        [
            "Editorial office: editor@appliedthings.org",
            "Filler one.",
            "Filler two.",
            "Filler three.",
            "Filler four.",
            "Filler five.",
            "Filler six.",
            "Jane Doe and John Roe",
            "Filler seven.",
            "Filler eight.",
            "Filler nine.",
            "Filler ten.",
            "Corresponding author: Jane Doe, jane.doe@uni.edu",
        ]
    )
    authors = [
        PaperAuthor(author_id=1, given="Jane", family="Doe", affiliation="X", corresponding=True),
        PaperAuthor(author_id=2, given="John", family="Roe", affiliation="X"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    jane = next(a for a in authors if a.family == "Doe")
    assert jane.email == "jane.doe@uni.edu"
    assert next(a for a in authors if a.family == "Roe").email in (None, "")


def test_fallback_with_marker_assigns_real_correspondence_address():
    """The elimination fallback is for genuine correspondence footers: a
    'Corresponding author' line plus an 'E-mail:' line, far from every
    surname, still reaches the single corresponding author."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        [
            "Jane Doe and John Roe",
            "Filler one.",
            "Filler two.",
            "Filler three.",
            "Filler four.",
            "Corresponding author: Jane Doe.",
            "E-mail: jane.doe@uni.edu.",
        ]
    )
    authors = [
        PaperAuthor(author_id=1, given="Jane", family="Doe", affiliation="X", corresponding=True),
        PaperAuthor(author_id=2, given="John", family="Roe", affiliation="X"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    assert next(a for a in authors if a.family == "Doe").email == "jane.doe@uni.edu"
    assert next(a for a in authors if a.family == "Roe").email in (None, "")


def test_mdpi_correspondence_line_without_name_skips_affiliation_address():
    """MDPI layout: a co-author's affiliation address sits just above a
    '* Correspondence:' line that names nobody, all far below the byline.
    The strict pairing is exhaustive, so the affiliation address must not go
    to the corresponding author by elimination and block her real address."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        [
            "Article",
            "Jane Doe 1,* , Ann Kim 2 , Bo Park 3 and John Roe 4",
            "1 School of Public Health, Fudan University, Shanghai 200032, China",
            "2 Department of Economics, Seoul National University, Seoul 08826, Korea",
            "3 Department of Statistics, Yonsei University, Seoul 03722, Korea",
            "4 Institute of Health Policy, Tokyo 113-8654, Japan; jroe77@u-tokyo.ac.jp (J.R.)",
            "* Correspondence: jd2020@fudan.edu.cn; Tel.: +86-21-5423-7000",
            "Received: 23 May 2020; Accepted: 23 July 2020; Published: 28 July 2020",
        ]
    )
    authors = [
        PaperAuthor(author_id=1, given="Jane", family="Doe", affiliation="X", corresponding=True),
        PaperAuthor(author_id=2, given="Ann", family="Kim", affiliation="X"),
        PaperAuthor(author_id=3, given="Bo", family="Park", affiliation="X"),
        PaperAuthor(author_id=4, given="John", family="Roe", affiliation="X"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    jane = next(a for a in authors if a.family == "Doe")
    assert jane.email == "jd2020@fudan.edu.cn"
    assert jane.corresponding is True
    assert all(a.email in (None, "") for a in authors if a.family != "Doe")


def test_strict_pairing_blocks_unrelated_earlier_email_in_fallback():
    """Elimination fallback with a strict pairing elsewhere: an editorial
    address printed before the marker line must not go to the single
    corresponding author — even with a correspondence marker in its window —
    because the marker-line pairings are exhaustive."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        ["Jane Doe and John Roe"]
        + [f"Filler {n}." for n in range(6)]
        + ["Editorial office: editor@appliedthings.org", "* Correspondence: jd77@uni.edu"]
    )
    authors = [
        PaperAuthor(author_id=1, given="Jane", family="Doe", affiliation="X", corresponding=True),
        PaperAuthor(author_id=2, given="John", family="Roe", affiliation="X"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    jane = next(a for a in authors if a.family == "Doe")
    assert jane.email == "jd77@uni.edu"
    assert jane.corresponding is True
    assert next(a for a in authors if a.family == "Roe").email in (None, "")


def test_plos_star_email_line_assigns_in_fallback():
    """PLOS '* E-mail:' footnotes carry no correspondence marker, so the
    fallback recognises the starred line itself as one."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        ["Jane Doe and John Roe"] + [f"Filler {n}." for n in range(6)] + ["* E-mail: jd77@uni.edu"]
    )
    authors = [
        PaperAuthor(author_id=1, given="Jane", family="Doe", affiliation="X", corresponding=True),
        PaperAuthor(author_id=2, given="John", family="Roe", affiliation="X"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    assert next(a for a in authors if a.family == "Doe").email == "jd77@uni.edu"
    assert next(a for a in authors if a.family == "Roe").email in (None, "")


def test_lower_ranked_candidate_with_name_match_wins():
    """When the top-ranked candidate fails the gate, a lower-ranked
    candidate whose local part names them still gets the address — instead
    of the address being dropped (or going to the wrong author)."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        ["Ann Kim and Kouji Yamamoto", "Dept X; koujiy@yokohama-cu.ac.jp"]
    )
    authors = [
        PaperAuthor(author_id=1, given="Ann", family="Kim", affiliation="X"),
        PaperAuthor(author_id=2, given="Kouji", family="Yamamoto", affiliation="X"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    assert next(a for a in authors if a.family == "Yamamoto").email == "koujiy@yokohama-cu.ac.jp"
    assert next(a for a in authors if a.family == "Kim").email in (None, "")


def test_two_letter_family_name_prefix_licenses_address():
    """'lixh@' for Xiaohong Li: a 2-letter family name cannot match by
    containment, but as the address's leading letters it still names her."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        [
            "Mei Sun 2 , Xiaohong Li 2",
            "2 Center, Fudan; sunmei@f.edu (M.S.); lixh@fudan.edu.cn (X.L.)",
        ]
    )
    authors = [
        PaperAuthor(author_id=1, given="Mei", family="Sun", affiliation="X", email="sunmei@f.edu"),
        PaperAuthor(author_id=2, given="Xiaohong", family="Li", affiliation="X"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    assert next(a for a in authors if a.family == "Li").email == "lixh@fudan.edu.cn"


def test_two_letter_given_token_does_not_license_address():
    """'Yu' in Yu-Zhong Zhang must not hand him the address: only the family
    name gets the 2-letter prefix rule, so a merely nearby opaque address
    stays unassigned."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        [
            "Yu-Zhong Zhang and Bo Chen",
            "Dept X",
            "Filler one.",
            "Contact yuri@uni.edu for the dataset.",
        ]
    )
    authors = [
        PaperAuthor(author_id=1, given="Yu-Zhong", family="Zhang", affiliation="X"),
        PaperAuthor(author_id=2, given="Bo", family="Chen", affiliation="X"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    assert all(a.email in (None, "") for a in authors)


def test_short_local_part_does_not_license_address():
    """A 2-letter local part ('do@' for Doe) matches by containment, but the
    affinity gate needs 3+ letters — otherwise any initial would license any
    nearby opaque address."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        ["Jane Doe and John Roe", "Dept X", "Contact do@uni.edu for details."]
    )
    authors = [
        PaperAuthor(author_id=1, given="Jane", family="Doe", affiliation="X"),
        PaperAuthor(author_id=2, given="John", family="Roe", affiliation="X"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    assert all(a.email in (None, "") for a in authors)


def test_strict_paired_opaque_address_assigns_despite_distance():
    """An opaque address printed on a correspondence-marker line is assigned
    to the nearby author even at a distance with no name match — the pairing
    is explicit, not proximity."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(["Jane Doe", "Dept X", "Correspondence: contact7@uni.edu"])
    authors = [PaperAuthor(author_id=1, given="Jane", family="Doe", affiliation="X")]
    AuthorEmailHarvester(contents).harvest(authors)

    assert authors[0].email == "contact7@uni.edu"
    assert authors[0].corresponding is True


def test_given_name_affinity_assigns_without_marker():
    """A diminutive local part ('bathri' for Bathrinath) licenses the address
    even with no correspondence marker anywhere nearby."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        [
            "Bathrinath Sankaranarayanan and John Roe",
            "Department of X",
            "Filler sentence one.",
            "Dr. S. Bathrinath can be contacted at bathri@gmail.com for samples.",
        ]
    )
    authors = [
        PaperAuthor(author_id=1, given="Bathrinath", family="Sankaranarayanan", affiliation="X"),
        PaperAuthor(author_id=2, given="John", family="Roe", affiliation="X"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    assert next(a for a in authors if a.family == "Sankaranarayanan").email == "bathri@gmail.com"
    assert next(a for a in authors if a.family == "Roe").email in (None, "")


def test_same_sentence_pairing_still_assigns_opaque_address():
    """An address printed in the same sentence as the surname keeps the
    historical pairing behaviour (no marker, no name match)."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        [
            "Jane Doe and John Roe",
            "Department of X",
            "Jane Doe, contact@lab.org, provided the samples.",
        ]
    )
    authors = [
        PaperAuthor(author_id=1, given="Jane", family="Doe", affiliation="X"),
        PaperAuthor(author_id=2, given="John", family="Roe", affiliation="X"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    assert next(a for a in authors if a.family == "Doe").email == "contact@lab.org"
    assert next(a for a in authors if a.family == "Doe").corresponding is False


def test_distant_opaque_email_without_marker_is_dropped():
    """Window proximity alone (surname nearby, nothing else) no longer
    licenses an opaque address — it is left unassigned, not misassigned."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    contents = _make_contents_with_text(
        [
            "Jane Doe and John Roe",
            "Department of X",
            "The reagents came from contact@lab.org in a previous study.",
        ]
    )
    authors = [
        PaperAuthor(author_id=1, given="Jane", family="Doe", affiliation="X"),
        PaperAuthor(author_id=2, given="John", family="Roe", affiliation="X"),
    ]
    AuthorEmailHarvester(contents).harvest(authors)

    assert all(a.email in (None, "") for a in authors)


# --- name-paired footnote e-mails ----------------------------------------------------


def _contents_by_section(sections: list[tuple[str, list[str]]]):
    """Stub contents whose sentences are grouped into (section_type, texts) sections."""
    sentences, section_rows = [], []
    for section_id, (section_type, texts) in enumerate(sections):
        section_rows.append(
            SimpleNamespace(section_id=section_id, section_type=section_type, header="")
        )
        for text in texts:
            sentences.append(
                SimpleNamespace(
                    text=text, section_id=section_id, text_id=len(sentences), page_number=1
                )
            )
    return SimpleNamespace(sentences=sentences, sections=section_rows)


def _authors(*names: tuple[str, str]) -> list[PaperAuthor]:
    return [
        PaperAuthor(author_id=i, given=given, family=family, affiliation="")
        for i, (given, family) in enumerate(names, start=1)
    ]


def _harvest(sections, authors, scope: int = 1):
    """Harvest as the extractor does: scoped to the first *scope* sections, whole paper as document.

    The first section stands for the selected front-matter block; footnotes and body
    sections after it are outside the scope, as they are in a real paper.
    """
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    if sections[0][0] != "title":
        sections = [("title", ["A Study of Things"]), *sections]
    document = _contents_by_section(sections)
    scoped = _contents_by_section(sections[:scope])
    AuthorEmailHarvester(scoped, document=document).harvest(authors)
    return authors


def test_elsevier_footnote_single_email_is_attached_and_promoted():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"), ("Omar", "Reyes"))
    _harvest(
        [
            ("title", ["A Study of Things", "Alice Lee, Hui-Kai Tan, Omar Reyes"]),
            (
                "footnote",
                [
                    "* Corresponding author at: Department of Things, Example University.",
                    "E-mail address: hk.tan@example.edu (H.K. Tan).",
                ],
            ),
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, True, False]
    assert authors[1].email == "hk.tan@example.edu"
    assert authors[0].email is None and authors[2].email is None


def test_elsevier_footnote_in_the_same_sentence_as_the_marker():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"))
    _harvest(
        [
            ("title", ["Byline"]),
            (
                "footnote",
                [
                    "* Corresponding author. Example University.\r\nE-mail address: t@example.edu (H. Tan)."
                ],
            ),
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, True]


def test_footnote_label_after_a_telephone_number_on_the_same_line():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"))
    _harvest(
        [
            (
                "footnote",
                [
                    "* Corresponding author. Tel.: +1 555 0100. E-mail address: t@example.edu (H.K. Tan)."
                ],
            ),
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, True]


def test_footnote_with_several_pairs_separated_by_commas_and_semicolons():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"), ("Omar", "Reyes"), ("Mia", "Novak"))
    _harvest(
        [
            ("title", ["Byline"]),
            (
                "footnote",
                [
                    "* Corresponding authors.",
                    "E-mail addresses: lee@example.org (A. Lee), t@example.org (H.K. Tan); "
                    "mnovak@example.org (M. Novak).",
                ],
            ),
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [True, True, False, True]
    assert authors[3].email == "mnovak@example.org"
    assert authors[2].email is None


@pytest.mark.parametrize(
    "printed",
    ["H.-K. Tan", "H.K. Tan", "HK Tan", "H. K. Tan", "H. Tan", "Hui-Kai Tan", "Hui Kai Tan"],
)
def test_footnote_initials_variants_match_the_author(printed):
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"))
    _harvest(
        [
            (
                "footnote",
                ["* Corresponding author.", f"E-mail address: x@example.org ({printed})."],
            ),
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, True]


@pytest.mark.parametrize("printed", ["M. Tan", "H.K. Lim", "Tan"])
def test_footnote_with_conflicting_initials_or_another_surname_attaches_nothing(printed):
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"))
    _harvest(
        [
            ("title", ["A Study of Things"]),
            (
                "footnote",
                ["* Corresponding author.", f"E-mail address: x@example.org ({printed})."],
            ),
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, False]
    assert [a.email for a in authors] == [None, None]


def test_footnote_with_two_authors_named_tan_is_ambiguous_and_fails_closed():
    authors = _authors(("Hui-Kai", "Tan"), ("Hao-Kun", "Tan"), ("Alice", "Lee"))
    _harvest(
        [
            ("footnote", ["* Corresponding author.", "E-mail address: x@example.org (H.K. Tan)."]),
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, False, False]
    assert [a.email for a in authors] == [None, None, None]


def test_footnote_with_two_tans_is_resolved_by_conflicting_initials():
    authors = _authors(("Hui-Kai", "Tan"), ("Mei", "Tan"), ("Alice", "Lee"))
    _harvest(
        [
            ("footnote", ["* Corresponding author.", "E-mail address: x@example.org (H.K. Tan)."]),
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [True, False, False]


def test_footnote_pair_without_a_correspondence_phrase_attaches_but_does_not_flag():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"))
    _harvest(
        [
            (
                "footnote",
                ["1 Present address: Example University.", "E-mail: x@example.org (H.K. Tan)."],
            ),
            ("body", ["The estimate corresponds to the mean."]),
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, False]
    assert [a.email for a in authors] == [None, "x@example.org"]


def test_footnote_pair_does_not_overwrite_or_steal_an_address():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"))
    authors[1].email = "own@example.org"
    authors[0].email = "x@example.org"
    _harvest(
        [
            ("footnote", ["* Corresponding author.", "E-mail address: x@example.org (H.K. Tan)."]),
        ],
        authors,
    )
    assert authors[1].email == "own@example.org"
    assert authors[1].corresponding is False


def test_footnote_pair_marks_an_author_whose_address_the_llm_already_attached():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"))
    authors[1].email = "t@example.org"
    _harvest(
        [
            ("footnote", ["* Corresponding author.", "E-mail address: t@example.org (H.K. Tan)."]),
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, True]


def test_footnote_outside_the_front_matter_scope_is_still_read():
    """The extractor scopes the harvester to the front-matter block; the footnote lives elsewhere."""
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"))
    paper = [
        ("title", ["A Study of Things", "Alice Lee, Hui-Kai Tan"]),
        ("references", ["1. Doe J. A thing. 2020."]),
        ("footnote", ["* Corresponding author.", "E-mail address: t@example.org (H.K. Tan)."]),
    ]
    _harvest(paper, authors)
    assert [a.corresponding for a in authors] == [False, True]
    assert authors[1].email == "t@example.org"


def test_no_footnote_changes_nothing():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"))
    _harvest(
        [("title", ["A Study of Things", "Alice Lee, Hui-Kai Tan"]), ("body", ["Some text."])],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, False]
    assert [a.email for a in authors] == [None, None]


def test_singular_marker_with_several_pairs_attaches_all_and_flags_none():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"), ("Omar", "Reyes"))
    _harvest(
        [
            (
                "footnote",
                [
                    "* Corresponding author at: Department of Things, Example University.",
                    "E-mail addresses: alee@example.org (A. Lee), hktan@example.org (H.-K. Tan), "
                    "oreyes@example.org (O. Reyes).",
                ],
            ),
        ],
        authors,
    )
    assert [a.email for a in authors] == [
        "alee@example.org",
        "hktan@example.org",
        "oreyes@example.org",
    ]
    assert not any(a.corresponding for a in authors)


def test_singular_marker_with_one_pair_flags_that_author():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"), ("Omar", "Reyes"))
    _harvest(
        [
            (
                "footnote",
                ["* Corresponding author.", "E-mail address: hktan@example.org (H.-K. Tan)."],
            )
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, True, False]
    assert authors[1].email == "hktan@example.org"


def test_plural_marker_with_several_pairs_flags_each_paired_author():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"), ("Omar", "Reyes"))
    _harvest(
        [
            (
                "footnote",
                [
                    "* Corresponding authors.",
                    "E-mail addresses: alee@example.org (A. Lee), oreyes@example.org (O. Reyes).",
                ],
            ),
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [True, False, True]


def test_footnote_does_not_add_a_second_flag_next_to_an_llm_flagged_author():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"), ("Omar", "Reyes"))
    authors[0].corresponding = True
    _harvest(
        [
            (
                "footnote",
                ["* Corresponding author.", "E-mail address: hktan@example.org (H.-K. Tan)."],
            )
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [True, False, False]
    assert authors[1].email == "hktan@example.org"  # still attached


def test_footnote_keeps_the_flag_when_the_paired_author_is_the_flagged_one():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"))
    authors[1].corresponding = True
    _harvest(
        [
            (
                "footnote",
                ["* Corresponding author.", "E-mail address: hktan@example.org (H.-K. Tan)."],
            )
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, True]
    assert authors[1].email == "hktan@example.org"


def test_footnote_does_not_add_a_second_flag_next_to_a_marker_sentence_flag():
    """An in-scope "* Correspondence: x@y" line flags its author; the footnote adds nobody."""
    from bibr.extract.author_email_harvester import AuthorEmailHarvester

    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"), ("Omar", "Reyes"))
    authors[0].email = "alee@example.org"
    sections = [
        ("title", ["A Study of Things", "* Correspondence: alee@example.org"]),
        ("footnote", ["* Corresponding author.", "E-mail address: hktan@example.org (H.-K. Tan)."]),
    ]
    AuthorEmailHarvester(
        _contents_by_section(sections[:1]), document=_contents_by_section(sections)
    ).harvest(authors)
    assert [a.corresponding for a in authors] == [True, False, False]
    assert authors[1].email == "hktan@example.org"


def test_body_prose_with_a_mid_sentence_correspondence_phrase_flags_nobody():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"))
    _harvest(
        [
            (
                "body",
                [
                    "We ran a multiple correspondence analysis and report the first axis "
                    "(e-mail: hk@example.org (H.K. Tan)).",
                ],
            )
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, False]


def test_body_sentence_that_opens_with_correspondence_analysis_flags_nobody():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"))
    _harvest(
        [
            (
                "body",
                [
                    "Correspondence analysis was applied to the table.",
                    "E-mail address: hk@example.org (H.K. Tan).",
                ],
            )
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, False]


@pytest.mark.parametrize(
    ("given", "printed"),
    [
        ("Hendrik Jan", "H. de Vries"),  # lower-case particle is not an initial
        ("Hendrik van", "H.J. de Vries"),  # particle left in the given name
        ("Łukasz", "Ł. de Vries"),  # transliterated letters fold the same on both sides
        ("Søren", "S. de Vries"),
    ],
)
def test_footnote_particles_and_transliterated_letters(given, printed):
    authors = _authors(("Alice", "Lee"), (given, "de Vries"))
    _harvest(
        [("footnote", ["* Corresponding author.", f"E-mail address: x@example.org ({printed})."])],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, True]


def test_footnote_surname_with_a_stroked_letter_folds_to_ascii():
    authors = _authors(("Piotr", "Wałęsa"), ("Alice", "Lee"))
    _harvest(
        [("footnote", ["* Corresponding author.", "E-mail address: p@example.org (P. Walesa)."])],
        authors,
    )
    assert [a.corresponding for a in authors] == [True, False]


def test_footnote_pass_with_many_authors_is_fast():
    import time

    names = [(f"Given{i}", f"Family{i}") for i in range(800)]
    authors = _authors(*names)
    footnote = [
        "* Corresponding author.",
        "E-mail address: x@example.org (G. Family799).",
        *[f"E-mail: u{i}@example.org (G. Nobody{i})." for i in range(200)],
    ]
    started = time.perf_counter()
    _harvest([("title", ["Byline"]), ("footnote", footnote)], authors)
    assert time.perf_counter() - started < 1.0
    assert authors[799].email == "x@example.org"


def test_footnote_address_does_not_displace_a_marker_sentence_address():
    """A phrase-less footnote pair must not take the address an in-scope marker line gives."""
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"))
    _harvest(
        [
            ("title", ["Alice Lee, Hui-Kai Tan", "* Correspondence: hk.tan@uni.example.edu"]),
            (
                "footnote",
                [
                    "E-mail addresses: tan.hk@mail.example.com (H.K. Tan), alee@example.org (A. Lee)."
                ],
            ),
        ],
        authors,
    )
    assert authors[1].email == "hk.tan@uni.example.edu"
    assert [a.corresponding for a in authors] == [False, True]
    assert authors[0].email == "alee@example.org"  # still attached: Lee had no address


def test_in_scope_correspondence_address_that_names_no_one_is_flagged_via_the_footnote_pair():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"), ("Omar", "Reyes"))
    _harvest(
        [
            (
                "title",
                [
                    "Alice Lee, Hui-Kai Tan, Omar Reyes",
                    "First abstract sentence.",
                    "Second abstract sentence.",
                    "Third abstract sentence.",
                    "Fourth abstract sentence.",
                    "Correspondence to: x@example.org",
                ],
            ),
            ("footnote", ["E-mail address: x@example.org (H.K. Tan)."]),
        ],
        authors,
    )
    assert authors[1].email == "x@example.org"
    assert [a.corresponding for a in authors] == [False, True, False]


def test_body_sentence_that_opens_with_correspondence_between_flags_nobody():
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"))
    _harvest(
        [
            (
                "body",
                [
                    "Correspondence between the two raters was high.",
                    "Coding was checked by a second rater (e-mail: hk@example.org (H.K. Tan)).",
                ],
            )
        ],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, False]


@pytest.mark.parametrize(
    "opener",
    [
        "Correspondence: hk@example.org",
        "*Correspondence:",
        "Correspondence to Department of Things.",
        "Correspondence concerning this article should be addressed to the author.",
        "Correspondence should be addressed to Department of Things.",
    ],
)
def test_correspondence_used_as_a_label_still_counts(opener):
    authors = _authors(("Alice", "Lee"), ("Hui-Kai", "Tan"))
    _harvest(
        [("footnote", [opener, "E-mail address: t@example.org (H.K. Tan)."])],
        authors,
    )
    assert [a.corresponding for a in authors] == [False, True]
