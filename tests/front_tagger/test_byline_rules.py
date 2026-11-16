"""Byline readings under ``rules``: names the tagger left out, and names that are no author's."""

from __future__ import annotations

from bibr.front_tagger.assemble import assemble

from .helpers import label, tagger_input

_TITLE = "Tidal patterns of the northern shelf"


def _names(inp, tags, *, rules=True):
    got = assemble(inp, tags, [1.0] * len(tags), rules=rules)
    return [(a.given, a.family) for a in got.byline.authors]


# The paper's own citation.


def test_names_in_the_papers_own_citation_are_no_byline():
    inp = tagger_input(
        ("doc_title", _TITLE),
        ("text", "Orla Venn1"),
        (
            "text",
            "Citation: Venn, O. (2021). Tidal patterns of the northern shelf. Shelf Letters, 4(2)",
        ),
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Orla"),
        ("FAMILY", "Venn"),
        ("AUTHOR_MARK", "1"),
        ("GIVEN", "O."),
    )
    assert _names(inp, tags, rules=False) == [("Orla", "Venn"), ("O.", "")]
    assert _names(inp, tags) == [("Orla", "Venn")]


def test_a_label_that_names_the_paper_drops_the_citations_names_alone():
    inp = tagger_input(
        ("doc_title", _TITLE),
        (
            "text",
            "To cite this article: Venn, O., & Brask, T. (2021). Tidal patterns. Shelf Letters",
        ),
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("FAMILY", "Venn"),
        ("GIVEN", "O."),
        ("FAMILY", "Brask"),
        ("GIVEN", "T."),
    )
    assert _names(inp, tags) == []


def test_a_label_alone_in_its_region_makes_the_next_region_the_citation():
    inp = tagger_input(
        ("doc_title", _TITLE),
        ("text", "Full citation:"),
        ("text", "Venn, O. (2021). Tidal patterns of the northern shelf. Shelf Letters"),
    )
    tags = label(inp, ("TITLE", _TITLE), ("FAMILY", "Venn"), ("GIVEN", "O."))
    assert _names(inp, tags) == []


def test_a_plain_citation_label_with_no_other_byline_keeps_its_names():
    # "Citation:" may introduce any work's citation; with no byline anywhere
    # else the names stay.
    inp = tagger_input(
        ("doc_title", _TITLE),
        ("text", "Citation: Venn, O. (2021). Tidal patterns of the northern shelf"),
    )
    tags = label(inp, ("TITLE", _TITLE), ("FAMILY", "Venn"), ("GIVEN", "O."))
    assert _names(inp, tags) == [("O.", "Venn")]


# Names inside an affiliation.


def test_a_name_an_institution_is_named_after_is_no_author():
    inp = tagger_input(
        ("doc_title", _TITLE),
        ("text", "Ada Kell"),
        ("text", "Faculty of Law, Ivo Marn Vela University, Norland"),
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("AFFILIATION", "Faculty of Law,"),
        ("GIVEN", "Ivo"),
        ("FAMILY", "Marn"),
        ("AFFILIATION", "Vela University, Norland"),
    )
    assert _names(inp, tags, rules=False) == [("Ada", "Kell"), ("Ivo", "Marn")]
    assert _names(inp, tags) == [("Ada", "Kell")]


def test_a_name_opening_its_line_before_a_post_and_a_department_stays():
    inp = tagger_input(
        ("doc_title", _TITLE),
        ("text", "Ada Kell Assistant Professor Department of Law"),
        ("text", "Ivo Marn Professor Department of History"),
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("AFFILIATION", "Department of Law"),
        ("GIVEN", "Ivo"),
        ("FAMILY", "Marn"),
        ("AFFILIATION", "Department of History"),
    )
    assert _names(inp, tags) == [("Ada", "Kell"), ("Ivo", "Marn")]


def test_a_name_after_its_key_or_title_at_the_line_start_stays():
    for lead, tagged in (("2 ", ("AUTHOR_MARK", "2")), ("Dr. ", None)):
        inp = tagger_input(
            ("doc_title", _TITLE),
            ("text", "1 Ada Kell Department of Law"),
            ("text", f"{lead}Ivo Marn Department of History"),
        )
        tags = label(
            inp,
            ("TITLE", _TITLE),
            ("AUTHOR_MARK", "1"),
            ("GIVEN", "Ada"),
            ("FAMILY", "Kell"),
            ("AFFILIATION", "Department of Law"),
            *([tagged] if tagged else []),
            ("GIVEN", "Ivo"),
            ("FAMILY", "Marn"),
            ("AFFILIATION", "Department of History"),
        )
        assert _names(inp, tags) == [("Ada", "Kell"), ("Ivo", "Marn")], lead


def test_a_place_between_an_institution_and_its_country_is_no_author():
    inp = tagger_input(
        ("doc_title", _TITLE),
        ("text", "Ada Kell Norland University, Vela Dora, Vietnam"),
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("AFFILIATION", "Norland University,"),
        ("GIVEN", "Vela"),
        ("FAMILY", "Dora"),
    )
    assert _names(inp, tags, rules=False) == [("Ada", "Kell"), ("Vela", "Dora")]
    assert _names(inp, tags) == [("Ada", "Kell")]


def test_names_in_a_list_are_never_read_as_part_of_an_affiliation():
    inp = tagger_input(
        ("doc_title", _TITLE),
        ("text", "Ada Kell, Ivo Marn Vela University, Norland"),
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("GIVEN", "Ivo"),
        ("FAMILY", "Marn"),
        ("AFFILIATION", "Vela University, Norland"),
    )
    assert _names(inp, tags) == [("Ada", "Kell"), ("Ivo", "Marn")]


# Names the tagger left out.


def test_the_last_name_of_a_keyed_list_is_read_whatever_its_case():
    inp = tagger_input(("doc_title", _TITLE), ("text", "Ada Kell1, Bea Moss2, and ivo marn3"))
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("AUTHOR_MARK", "1"),
        ("GIVEN", "Bea"),
        ("FAMILY", "Moss"),
        ("AUTHOR_MARK", "2"),
        ("AUTHOR_MARK", "3"),
    )
    assert _names(inp, tags, rules=False) == [("Ada", "Kell"), ("Bea", "Moss")]
    assert _names(inp, tags) == [("Ada", "Kell"), ("Bea", "Moss"), ("ivo", "marn")]


def test_no_last_name_is_read_when_an_author_of_the_list_shows_no_key():
    inp = tagger_input(("doc_title", _TITLE), ("text", "Ada Kell, Bea Moss2, and ivo3"))
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("GIVEN", "Bea"),
        ("FAMILY", "Moss"),
        ("AUTHOR_MARK", "2"),
        ("AUTHOR_MARK", "3"),
    )
    assert _names(inp, tags) == [("Ada", "Kell"), ("Bea", "Moss")]


def test_a_name_its_address_spells_is_read():
    inp = tagger_input(
        ("doc_title", _TITLE),
        (
            "text",
            "Ada Kell is at Norland University, Ivo Marn studies with Bea Moss. ivo.marn@vela.example",
        ),
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("GIVEN", "Bea"),
        ("FAMILY", "Moss"),
        ("EMAIL", "ivo.marn@vela.example"),
    )
    assert _names(inp, tags, rules=False) == [("Ada", "Kell"), ("Bea", "Moss")]
    assert _names(inp, tags) == [("Ada", "Kell"), ("Ivo", "Marn"), ("Bea", "Moss")]


def test_an_authors_name_printed_again_by_its_address_is_not_read_twice():
    inp = tagger_input(
        ("doc_title", _TITLE),
        ("text", "Ada Kell and Bea Moss"),
        ("text", "Correspondence: Ada Kell, ada.kell@norland.example"),
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("GIVEN", "Bea"),
        ("FAMILY", "Moss"),
        ("EMAIL", "ada.kell@norland.example"),
    )
    assert _names(inp, tags) == [("Ada", "Kell"), ("Bea", "Moss")]


def _role_address_names(note, address):
    inp = tagger_input(
        ("doc_title", _TITLE),
        ("text", "Ada Kell1, Bea Moss2"),
        ("text", "1 Norland University, Vela"),
        ("footnote", note),
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("AUTHOR_MARK", "1"),
        ("GIVEN", "Bea"),
        ("FAMILY", "Moss"),
        ("AUTHOR_MARK", "2"),
        ("EMAIL", address),
    )
    return _names(inp, tags)


def test_a_label_its_address_spells_is_no_name():
    # "Norland Lab:" labels the address it spells, as a name never is.
    assert _role_address_names(
        "Norland Lab: norland.lab@shelf.example", "norland.lab@shelf.example"
    ) == [("Ada", "Kell"), ("Bea", "Moss")]


def test_a_role_its_address_spells_is_no_name():
    assert _role_address_names(
        "Write to the Editorial Office at editorial.office@shelf.example",
        "editorial.office@shelf.example",
    ) == [("Ada", "Kell"), ("Bea", "Moss")]
    assert _role_address_names(
        "Data Availability data.availability@shelf.example", "data.availability@shelf.example"
    ) == [("Ada", "Kell"), ("Bea", "Moss")]


def test_a_given_name_cut_off_by_a_region_break_is_read():
    inp = tagger_input(
        ("doc_title", _TITLE),
        ("text", "Ada Kell1, Ivo"),
        ("text", "Marn2, and Bea Moss3"),
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("AUTHOR_MARK", "1"),
        ("FAMILY", "Marn"),
        ("AUTHOR_MARK", "2"),
        ("GIVEN", "Bea"),
        ("FAMILY", "Moss"),
        ("AUTHOR_MARK", "3"),
    )
    assert _names(inp, tags, rules=False) == [("Ada", "Kell"), ("", "Marn"), ("Bea", "Moss")]
    assert _names(inp, tags) == [("Ada", "Kell"), ("Ivo", "Marn"), ("Bea", "Moss")]


def test_a_surname_in_capitals_tagged_alone_stays_the_surname():
    inp = tagger_input(("doc_title", _TITLE), ("text", "KOVAR Tilen"), ("text", "Ada Kell"))
    tags = label(inp, ("TITLE", _TITLE), ("FAMILY", "KOVAR"), ("GIVEN", "Ada"), ("FAMILY", "Kell"))
    assert _names(inp, tags) == [("Tilen", "KOVAR"), ("Ada", "Kell")]


def test_a_name_all_in_capitals_reads_as_before():
    inp = tagger_input(("doc_title", _TITLE), ("text", "KOVAR TILEN"), ("text", "Ada Kell"))
    tags = label(inp, ("TITLE", _TITLE), ("FAMILY", "KOVAR"), ("GIVEN", "Ada"), ("FAMILY", "Kell"))
    assert _names(inp, tags) == [("KOVAR", "TILEN"), ("Ada", "Kell")]


def test_an_initial_tagged_alone_stays_the_given_name():
    # An initial is in capitals too, but it is no surname: the word after it is.
    for line, initial in (("J. Vossel", "J."), ("JV Vossel", "JV"), ("J.R. Vossel", "J.R.")):
        inp = tagger_input(("doc_title", _TITLE), ("text", line), ("text", "Norland University"))
        tags = label(
            inp, ("TITLE", _TITLE), ("FAMILY", initial), ("AFFILIATION", "Norland University")
        )
        assert _names(inp, tags) == [(initial, "Vossel")], line


# Grouping.


def test_a_family_comma_given_item_in_a_given_first_byline_is_one_author():
    inp = tagger_input(("doc_title", _TITLE), ("text", "Ada Kell & Moss, Bea T."))
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("FAMILY", "Moss"),
        ("GIVEN", "Bea T"),
    )
    assert _names(inp, tags, rules=False) == [("Ada", "Kell"), ("", "Moss"), ("Bea T", "")]
    assert _names(inp, tags) == [("Ada", "Kell"), ("Bea T", "Moss")]


def test_a_given_name_with_an_untagged_name_after_it_opens_its_own_item():
    inp = tagger_input(("doc_title", _TITLE), ("text", "Ada Kell, Moss, Bea Toller"))
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("FAMILY", "Moss"),
        ("GIVEN", "Bea"),
    )
    assert _names(inp, tags) == [("Ada", "Kell"), ("", "Moss"), ("Bea", "Toller")]


def _list_names(line, *spans):
    inp = tagger_input(("doc_title", _TITLE), ("text", line))
    return _names(inp, label(inp, ("TITLE", _TITLE), *spans))


_TWO_AUTHORS = (("GIVEN", "Ada"), ("FAMILY", "Kell"), ("GIVEN", "Bea"), ("FAMILY", "Moss"))


def test_two_one_name_authors_ending_the_list_stay_two():
    # "Orvan, Telis": a lone full word after a one-name author is the next
    # one-name author, not the given name of an inverted item.
    assert _list_names(
        "Ada Kell, Bea Moss, Orvan, Telis",
        *_TWO_AUTHORS,
        ("FAMILY", "Orvan"),
        ("GIVEN", "Telis"),
    ) == [("Ada", "Kell"), ("Bea", "Moss"), ("", "Orvan"), ("Telis", "")]


def test_an_inverted_item_with_initials_is_read_at_the_end_of_the_list():
    assert _list_names(
        "Ada Kell, Bea Moss, Orvan, T.", *_TWO_AUTHORS, ("FAMILY", "Orvan"), ("GIVEN", "T.")
    ) == [("Ada", "Kell"), ("Bea", "Moss"), ("T.", "Orvan")]


def test_an_inverted_item_with_a_full_given_name_is_read_beside_one_with_initials():
    # The byline prints "Roth, T." the other way round: "Orvan, Telis" may be too.
    assert _list_names(
        "Roth, T., Ada Kell, Bea Moss, Orvan, Telis",
        ("FAMILY", "Roth"),
        ("GIVEN", "T."),
        *_TWO_AUTHORS,
        ("FAMILY", "Orvan"),
        ("GIVEN", "Telis"),
    ) == [("T.", "Roth"), ("Ada", "Kell"), ("Bea", "Moss"), ("Telis", "Orvan")]


def test_a_name_of_a_list_tagged_an_affiliation_is_read_as_a_name():
    inp = tagger_input(
        ("doc_title", _TITLE), ("text", "Ada Kell, Bea Moss, Ivo Marn, and Tea Roth")
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("GIVEN", "Bea"),
        ("FAMILY", "Moss"),
        ("AFFILIATION", "Ivo Marn"),
        ("GIVEN", "Tea"),
        ("FAMILY", "Roth"),
    )
    assert _names(inp, tags, rules=False) == [("Ada", "Kell"), ("Bea", "Moss"), ("Tea", "Roth")]
    assert _names(inp, tags) == [
        ("Ada", "Kell"),
        ("Bea", "Moss"),
        ("Ivo", "Marn"),
        ("Tea", "Roth"),
    ]


def test_an_affiliation_between_authors_stays_when_the_list_prints_others():
    inp = tagger_input(
        ("doc_title", _TITLE), ("text", "Ada Kell Norland, Bea Moss, Ivo Marn, and Tea Roth")
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("GIVEN", "Bea"),
        ("FAMILY", "Moss"),
        ("AFFILIATION", "Ivo Marn"),
        ("GIVEN", "Tea"),
        ("FAMILY", "Roth"),
    )
    assert _names(inp, tags) == [("Ada", "Kell"), ("Bea", "Moss"), ("Tea", "Roth")]


def test_a_place_between_two_authors_stays_an_affiliation():
    # "Name, Place" items: a country is no name, and with two authors there is
    # no other gap to show the list prints no affiliation.
    for line, place, last in (
        ("Ada Kell, New Zealand, and Bea Moss, Norland", "New Zealand", "Norland"),
        ("Ada Kell, Vela Dora, Bea Moss, Norland", "Vela Dora", "Norland"),
    ):
        inp = tagger_input(("doc_title", _TITLE), ("text", line))
        tags = label(
            inp,
            ("TITLE", _TITLE),
            ("GIVEN", "Ada"),
            ("FAMILY", "Kell"),
            ("AFFILIATION", place),
            ("GIVEN", "Bea"),
            ("FAMILY", "Moss"),
            ("AFFILIATION", last),
        )
        assert _names(inp, tags) == [("Ada", "Kell"), ("Bea", "Moss")], line


def test_a_country_between_authors_of_a_list_stays_an_affiliation():
    inp = tagger_input(
        ("doc_title", _TITLE), ("text", "Ada Kell, Bea Moss, Sri Lanka, and Tea Roth")
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("GIVEN", "Bea"),
        ("FAMILY", "Moss"),
        ("AFFILIATION", "Sri Lanka"),
        ("GIVEN", "Tea"),
        ("FAMILY", "Roth"),
    )
    assert _names(inp, tags) == [("Ada", "Kell"), ("Bea", "Moss"), ("Tea", "Roth")]


def test_a_name_line_inside_an_affiliation_printed_around_it_is_an_author():
    inp = tagger_input(
        ("doc_title", _TITLE),
        (
            "text",
            "Ada Kell\nFaculty of Law, Norland University\nIvo Marn\nFaculty of Law, Norland University",
        ),
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        (
            "AFFILIATION",
            "Faculty of Law, Norland University Ivo Marn Faculty of Law, Norland University",
        ),
    )
    assert _names(inp, tags, rules=False) == [("Ada", "Kell")]
    assert _names(inp, tags) == [("Ada", "Kell"), ("Ivo", "Marn")]


def test_a_name_line_between_two_different_affiliations_stays_in_them():
    inp = tagger_input(
        ("doc_title", _TITLE),
        ("text", "Ada Kell\nFaculty of Law, Norland University\nIvo Marn\nVela College"),
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("AFFILIATION", "Faculty of Law, Norland University Ivo Marn Vela College"),
    )
    assert _names(inp, tags) == [("Ada", "Kell")]


_ABSTRACT = (
    "We measured tides at forty stations along the northern shelf over three winters and "
    "compared them with the model run for the same period. The model kept the phase of "
    "the main tide within ten minutes at most stations but missed the surge on the two "
    "nights of the largest storms, when the measured water stood half a metre higher."
)


def test_a_note_after_the_abstract_names_no_authors_beside_the_byline():
    inp = tagger_input(
        ("doc_title", _TITLE),
        ("text", "Ada Kell and Bea Moss"),
        ("abstract", _ABSTRACT),
        ("text", "Ada Kell is Professor of Law and holds the Ivo Marn Chair of Tides."),
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("GIVEN", "Bea"),
        ("FAMILY", "Moss"),
        ("ABSTRACT", _ABSTRACT),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("GIVEN", "Ivo"),
        ("FAMILY", "Marn"),
    )
    assert ("Ivo", "Marn") in _names(inp, tags, rules=False)
    assert _names(inp, tags) == [("Ada", "Kell"), ("Bea", "Moss"), ("Ada", "Kell")]


def test_names_after_the_abstract_stay_when_they_repeat_no_author():
    inp = tagger_input(
        ("doc_title", _TITLE),
        ("text", "Ada Kell and Bea Moss"),
        ("abstract", _ABSTRACT),
        ("text", "Ivo Marn and Tea Roth"),
    )
    tags = label(
        inp,
        ("TITLE", _TITLE),
        ("GIVEN", "Ada"),
        ("FAMILY", "Kell"),
        ("GIVEN", "Bea"),
        ("FAMILY", "Moss"),
        ("ABSTRACT", _ABSTRACT),
        ("GIVEN", "Ivo"),
        ("FAMILY", "Marn"),
        ("GIVEN", "Tea"),
        ("FAMILY", "Roth"),
    )
    assert _names(inp, tags) == [("Ada", "Kell"), ("Bea", "Moss"), ("Ivo", "Marn"), ("Tea", "Roth")]
