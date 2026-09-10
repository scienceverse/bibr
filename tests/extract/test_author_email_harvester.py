"""AuthorEmailHarvester — corresponding-author email enrichment tests."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

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
