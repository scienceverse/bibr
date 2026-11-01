from __future__ import annotations

import pytest

from bibr.front_tagger.assemble import assemble
from bibr.front_tagger.contract import build_tagger_input

from .helpers import label, regions, tagger_input


def _ones(inp):
    return [1.0] * len(inp.words)


def _page():
    return tagger_input(
        ("header", "HANDEDNESS IN CHINESE SAMPLES 1"),
        ("doc_title", "Running Head: HANDEDNESS Prevalence and Heritability of Handedness"),
        ("text", "Thais de Oliveira1,2, Mary Jones2* and Li Wei3"),
        ("text", "1 Universidade de São Paulo, Brazil"),
        ("text", "2 University of Twente"),
        ("text", "3 Peking University"),
        ("text", "* Corresponding author: mary.jones@utwente.nl"),
        ("abstract", "Abstract"),
        (
            "abstract",
            "We studied handedness in 400 families from three countries over two decades. Left handedness was more common in sons than in daughters, ran in families, and was rarer among older participants. Twin pairs agreed more often than siblings. It is heritable.",
        ),
        ("text", "Keywords: handedness; heritability"),
        ("footer", "Laterality 2021, Vol. 26(3) 65-74 © The Author(s) 2021"),
    )


def _tags(inp):
    return label(
        inp,
        ("FURNITURE", "HANDEDNESS IN CHINESE SAMPLES 1"),
        ("FURNITURE", "Running Head: HANDEDNESS"),
        ("TITLE", "Prevalence and Heritability of Handedness"),
        ("GIVEN", "Thais"),
        ("FAMILY", "de Oliveira"),
        ("AUTHOR_MARK", "1,2"),
        ("GIVEN", "Mary"),
        ("FAMILY", "Jones"),
        ("AUTHOR_MARK", "2*"),
        ("GIVEN", "Li"),
        ("FAMILY", "Wei"),
        ("AUTHOR_MARK", "3"),
        ("AFF_MARK", "1"),
        ("AFFILIATION", "Universidade de São Paulo, Brazil"),
        ("AFF_MARK", "2"),
        ("AFFILIATION", "University of Twente"),
        ("AFF_MARK", "3"),
        ("AFFILIATION", "Peking University"),
        ("EMAIL", "mary.jones@utwente.nl"),
        (
            "ABSTRACT",
            "We studied handedness in 400 families from three countries over two decades. Left handedness was more common in sons than in daughters, ran in families, and was rarer among older participants. Twin pairs agreed more often than siblings. It is heritable.",
        ),
        ("KEYWORD", "handedness"),
        ("KEYWORD", "heritability"),
        ("JOURNAL", "Laterality"),
        ("VOLUME", "26"),
        ("ISSUE", "3"),
        ("PAGES", "65-74"),
    )


def test_full_page_reads_back_as_printed():
    inp = _page()
    got = assemble(inp, _tags(inp), _ones(inp))
    tk = got.title_keywords
    assert tk.title == "Prevalence and Heritability of Handedness"
    assert (
        tk.abstract
        == "We studied handedness in 400 families from three countries over two decades. Left handedness was more common in sons than in daughters, ran in families, and was rarer among older participants. Twin pairs agreed more often than siblings. It is heritable."
    )
    assert tk.keywords == ["handedness", "heritability"]
    assert (tk.journal, tk.volume, tk.issue) == ("Laterality", "26", "3")
    assert (tk.first_page, tk.last_page) == ("65", "74")
    names = [(a.given, a.family) for a in got.byline.authors]
    assert names == [("Thais", "de Oliveira"), ("Mary", "Jones"), ("Li", "Wei")]
    affs = [a.affiliation for a in got.byline.authors]
    assert affs == [
        "Universidade de São Paulo, Brazil; University of Twente",
        "University of Twente",
        "Peking University",
    ]
    jones = got.byline.authors[1]
    assert jones.email == "mary.jones@utwente.nl"
    assert jones.corresponding is True
    assert not got.byline.authors[0].corresponding
    assert got.problems == []
    assert got.accepted(0.9)


def test_family_first_bylines_pair_up():
    inp = tagger_input(("doc_title", "A title here"), ("text", "Smith J, Jones KL"))
    tags = label(
        inp,
        ("TITLE", "A title here"),
        ("FAMILY", "Smith"),
        ("GIVEN", "J"),
        ("FAMILY", "Jones"),
        ("GIVEN", "KL"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    assert [(a.given, a.family) for a in got.byline.authors] == [("J", "Smith"), ("KL", "Jones")]
    # One unmarked affiliation-free byline: nothing to link, nothing wrong.
    assert got.byline_reasons(0.9) == []


def test_low_confidence_edge_word_holds_the_title_back():
    inp = _page()
    tags = _tags(inp)
    conf = _ones(inp)
    # The word right before the title (the running head's last word) is unsure.
    title_start = next(i for i, t in enumerate(tags) if t == "B-TITLE")
    conf[title_start - 1] = 0.6
    got = assemble(inp, tags, conf)
    assert got.confidence["title"] == 0.6
    assert not got.accepted(0.9)
    assert "low_confidence_title" in got.rejection_reasons(0.9)


def test_unresolved_affiliation_mark_falls_back():
    inp = tagger_input(("doc_title", "Title words"), ("text", "Ann Lee1 and Bo Kim2"))
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ann"),
        ("FAMILY", "Lee"),
        ("AUTHOR_MARK", "1"),
        ("GIVEN", "Bo"),
        ("FAMILY", "Kim"),
        ("AUTHOR_MARK", "2"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    # No affiliation list was read at all: nothing to resolve the keys against.
    assert got.byline_reasons(0.9) == []
    inp2 = tagger_input(
        ("doc_title", "Title words"), ("text", "Ann Lee1 and Bo Kim2"), ("text", "1 Univ A")
    )
    tags2 = label(
        inp2,
        ("TITLE", "Title words"),
        ("GIVEN", "Ann"),
        ("FAMILY", "Lee"),
        ("AUTHOR_MARK", "1"),
        ("GIVEN", "Bo"),
        ("FAMILY", "Kim"),
        ("AUTHOR_MARK", "2"),
        ("AFF_MARK", "1"),
        ("AFFILIATION", "Univ A"),
    )
    got2 = assemble(inp2, tags2, [1.0] * len(tags2))
    assert "affiliation_mark_unresolved" in got2.problems
    assert not got2.accepted(0.5)


def test_mark_ranges_expand():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ann Lee1–3"),
        ("text", "1 Univ A"),
        ("text", "2 Univ B"),
        ("text", "3 Univ C"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ann"),
        ("FAMILY", "Lee"),
        ("AUTHOR_MARK", "1–3"),
        ("AFF_MARK", "1"),
        ("AFFILIATION", "Univ A"),
        ("AFF_MARK", "2"),
        ("AFFILIATION", "Univ B"),
        ("AFF_MARK", "3"),
        ("AFFILIATION", "Univ C"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    assert got.byline.authors[0].affiliation == "Univ A; Univ B; Univ C"


def test_unmarked_affiliations_link_by_position():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ann Lee"),
        ("text", "Univ A"),
        ("text", "Bo Kim"),
        ("text", "Univ B"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ann"),
        ("FAMILY", "Lee"),
        ("AFFILIATION", "Univ A"),
        ("GIVEN", "Bo"),
        ("FAMILY", "Kim"),
        ("AFFILIATION", "Univ B"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    assert [a.affiliation for a in got.byline.authors] == ["Univ A", "Univ B"]
    assert got.byline_reasons(0.9) == []


def test_unmarked_affiliations_after_the_byline_are_ambiguous():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ann Lee, Bo Kim"),
        ("text", "Univ A"),
        ("text", "Univ B"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ann"),
        ("FAMILY", "Lee"),
        ("GIVEN", "Bo"),
        ("FAMILY", "Kim"),
        ("AFFILIATION", "Univ A"),
        ("AFFILIATION", "Univ B"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    assert "affiliations_unlinked" in got.problems


def test_affiliations_no_author_carries_are_returned_in_print_order():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ann Lee, Bo Kim"),
        ("text", "Univ A"),
        ("text", "Univ B"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ann"),
        ("FAMILY", "Lee"),
        ("GIVEN", "Bo"),
        ("FAMILY", "Kim"),
        ("AFFILIATION", "Univ A"),
        ("AFFILIATION", "Univ B"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    assert [a.affiliation for a in got.byline.authors] == ["", ""]
    assert got.unattached_affiliations == ["Univ A", "Univ B"]
    rows = got.affiliation_rows()
    assert [(r.text, r.author_ids, r.institution) for r in rows] == [
        ("Univ A", [], None),
        ("Univ B", [], None),
    ]


def test_an_affiliation_whose_mark_no_author_prints_is_returned():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ann Lee1, Bo Kim1"),
        ("text", "1 Univ A"),
        ("text", "2 Univ B"),
        ("text", "Univ C"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ann"),
        ("FAMILY", "Lee"),
        ("AUTHOR_MARK", "1"),
        ("GIVEN", "Bo"),
        ("FAMILY", "Kim"),
        ("AUTHOR_MARK", "1"),
        ("AFF_MARK", "1"),
        ("AFFILIATION", "Univ A"),
        ("AFF_MARK", "2"),
        ("AFFILIATION", "Univ B"),
        ("AFFILIATION", "Univ C"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    assert [a.affiliation for a in got.byline.authors] == ["Univ A", "Univ A"]
    assert got.unattached_affiliations == ["Univ B", "Univ C"]


def test_a_second_unmarked_affiliation_after_an_author_is_returned():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ann Lee"),
        ("text", "Univ A"),
        ("text", "Univ B"),
        ("text", "Bo Kim"),
        ("text", "Univ C"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ann"),
        ("FAMILY", "Lee"),
        ("AFFILIATION", "Univ A"),
        ("AFFILIATION", "Univ B"),
        ("GIVEN", "Bo"),
        ("FAMILY", "Kim"),
        ("AFFILIATION", "Univ C"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    assert [a.affiliation for a in got.byline.authors] == ["Univ A", "Univ C"]
    assert got.unattached_affiliations == ["Univ B"]


def test_nothing_is_returned_when_every_affiliation_is_linked():
    inp = _page()
    got = assemble(inp, _tags(inp), _ones(inp))
    assert got.unattached_affiliations == []
    assert got.untagged_affiliations == []
    assert got.affiliation_rows() == []


def test_initials_tagged_as_an_affiliation_are_not_returned():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ann Lee, Bo Kim"),
        ("text", "Univ A"),
        ("text", "Univ B"),
        ("text", "(J.P.)"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ann"),
        ("FAMILY", "Lee"),
        ("GIVEN", "Bo"),
        ("FAMILY", "Kim"),
        ("AFFILIATION", "Univ A"),
        ("AFFILIATION", "Univ B"),
        ("AFFILIATION", "(J.P.)"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    assert got.unattached_affiliations == ["Univ A", "Univ B"]


def test_affiliation_rows_merge_variants_and_can_leave_untagged_text_out():
    from bibr.front_tagger.assemble import TaggedFrontMatter

    got = TaggedFrontMatter(
        title_keywords=None,
        byline=None,
        unattached_affiliations=["Université de Lyon, France", "Univ A"],
        untagged_affiliations=["UNIVERSITE DE LYON FRANCE.", "Univ B"],
    )
    assert [r.text for r in got.affiliation_rows()] == [
        "Université de Lyon, France",
        "Univ A",
        "Univ B",
    ]
    assert [r.text for r in got.affiliation_rows(untagged=False)] == [
        "Université de Lyon, France",
        "Univ A",
    ]


def test_an_institution_run_after_a_family_name_is_read_when_no_author_has_one():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ann Lee University of Michigan, USA; email: ann@umich.example"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ann"),
        ("FAMILY", "Lee"),
        ("EMAIL", "ann@umich.example"),
    )
    got = assemble(inp, tags, [1.0] * len(tags), rules=True)
    assert got.untagged_affiliations == ["University of Michigan, USA"]
    assert got.unattached_affiliations == []
    # Without rules the text stays unread.
    assert assemble(inp, tags, [1.0] * len(tags)).untagged_affiliations == []
    assert [r.text for r in got.affiliation_rows()] == ["University of Michigan, USA"]
    assert got.affiliation_rows(untagged=False) == []


def test_an_untagged_institution_line_is_read_before_the_abstract_only():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ann Lee, Bo Kim"),
        ("text", "Department of Physics, Univ of Oslo"),
        ("text", "Received 3 May 2020, accepted at University of Bergen"),
        ("text", "Correspondence to the Institute of Law"),
        ("abstract", "A short abstract."),
        ("text", "School of Law, Univ X"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ann"),
        ("FAMILY", "Lee"),
        ("GIVEN", "Bo"),
        ("FAMILY", "Kim"),
        ("ABSTRACT", "A short abstract."),
    )
    got = assemble(inp, tags, [1.0] * len(tags), rules=True)
    # The notes open as notes, and the line past the abstract is not front matter.
    assert got.untagged_affiliations == ["Department of Physics, Univ of Oslo"]
    assert assemble(inp, tags, [1.0] * len(tags)).untagged_affiliations == []


def test_untagged_institutions_are_not_read_where_an_author_has_an_affiliation():
    inp = _page()
    tags = _tags(inp)
    got = assemble(inp, tags, _ones(inp), rules=True)
    assert got.byline.authors[0].affiliation
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ann Lee1"),
        ("text", "1 Univ A"),
        ("text", "School of Law, Univ X"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ann"),
        ("FAMILY", "Lee"),
        ("AUTHOR_MARK", "1"),
        ("AFF_MARK", "1"),
        ("AFFILIATION", "Univ A"),
    )
    got = assemble(inp, tags, [1.0] * len(tags), rules=True)
    assert [a.affiliation for a in got.byline.authors] == ["Univ A"]
    assert got.untagged_affiliations == []


def test_parallel_abstracts_and_titles_keep_the_first_printed():
    inp = tagger_input(
        ("doc_title", "Título en español"),
        ("doc_title", "Title in English"),
        ("text", "Ana Ruiz"),
        ("abstract", "Resumen"),
        ("abstract", "Estudiamos la mano en familias."),
        ("abstract", "Abstract"),
        ("abstract", "We studied the hand in families."),
    )
    tags = label(
        inp,
        ("TITLE", "Título en español"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Ruiz"),
        ("ABSTRACT", "Estudiamos la mano en familias."),
        ("ABSTRACT", "We studied the hand in families."),
    )
    # A translated title is not the paper's title: the projection leaves it
    # untagged, so the model learns to tag the printed-first version only.
    got = assemble(inp, tags, [1.0] * len(tags))
    assert got.title_keywords.title == "Título en español"
    assert got.title_keywords.abstract == "Estudiamos la mano en familias."


def test_title_split_across_regions_and_subtitle():
    inp = tagger_input(
        ("doc_title", "Hand preference in"),
        ("doc_title", "Chinese families"),
        ("paragraph_title", "A twin study"),
        ("text", "Ann Lee"),
    )
    tags = label(
        inp,
        ("TITLE", "Hand preference in"),
        ("TITLE", "Chinese families"),
        ("SUBTITLE", "A twin study"),
        ("GIVEN", "Ann"),
        ("FAMILY", "Lee"),
    )
    tags[[w.text for w in inp.words].index("Chinese")] = "B-TITLE"
    got = assemble(inp, tags, [1.0] * len(tags))
    assert got.title_keywords.title == "Hand preference in Chinese families: A twin study"


def test_no_title_or_authors_is_a_problem():
    inp = tagger_input(("text", "Some body text only"))
    got = assemble(inp, ["O"] * len(inp.words), [1.0] * len(inp.words))
    assert got.problems == ["no_title", "no_abstract", "no_authors"]
    assert not got.title_keywords.title
    assert got.byline.authors == []


def test_orcid_and_license_normalised():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ann Lee https://orcid.org/0000-0002-1825-009x"),
        ("footer", "This is an open access article under the CC BY 4.0 license"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ann"),
        ("FAMILY", "Lee"),
        ("ORCID", "https://orcid.org/0000-0002-1825-009x"),
        ("LICENSE", "CC BY 4.0 license"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    assert got.byline.authors[0].orcid == "0000-0002-1825-009X"
    assert got.title_keywords.license == "CC BY 4.0"


def test_superscript_keys_symbol_runs_and_orcid_icons():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ana Neri-Suárez²,* Bo HeID**"),
        ("text", "2 Univ B"),
        ("text", "* Corresponding author"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Neri-Suárez"),
        ("AUTHOR_MARK", "²,*"),
        ("GIVEN", "Bo"),
        ("FAMILY", "HeID"),
        ("AUTHOR_MARK", "**"),
        ("AFF_MARK", "2"),
        ("AFFILIATION", "Univ B"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    first, second = got.byline.authors
    assert (first.family, first.affiliation, first.corresponding) == ("Neri-Suárez", "Univ B", True)
    # "**" is one key with no affiliation behind it: a symbol, not a lost list.
    assert (second.family, second.corresponding) == ("He", False)
    assert got.byline_reasons(0.9) == []


def test_letter_keys_glued_to_family_names():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Juan Vegaa, Ana Mallenb,c"),
        ("text", "a Univ A"),
        ("text", "b Univ B"),
        ("text", "c Univ C"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Juan"),
        ("FAMILY", "Vegaa"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Mallenb"),
        ("AUTHOR_MARK", "c"),
        ("AFF_MARK", "a"),
        ("AFFILIATION", "Univ A"),
        ("AFF_MARK", "b"),
        ("AFFILIATION", "Univ B"),
        ("AFF_MARK", "c"),
        ("AFFILIATION", "Univ C"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    rows = [(a.family, a.affiliation) for a in got.byline.authors]
    # "Mallenb" carries a mark of its own, so its last letter stays: the rule
    # only splits an unmarked name.
    assert rows == [("Vega", "Univ A"), ("Mallenb", "Univ C")]


@pytest.mark.parametrize(
    ("rules", "expected"),
    [
        (True, [("Rios", "Univ A; Univ B"), ("Costa", "Univ A"), ("Wangd", "")]),
        # Without rules only the last letter of a run goes.
        (False, [("Riosa", "Univ B"), ("Costa", "Univ A"), ("Wangd", "")]),
    ],
)
def test_a_run_of_letter_keys_glued_to_a_family_name_is_split_off(rules, expected):
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Juan Riosab, Ana Costaa, Bo Wangd"),
        ("text", "a Univ A"),
        ("text", "b Univ B"),
        ("text", "c Univ C"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Juan"),
        ("FAMILY", "Riosab"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Costaa"),
        ("GIVEN", "Bo"),
        ("FAMILY", "Wangd"),
        ("AFF_MARK", "a"),
        ("AFFILIATION", "Univ A"),
        ("AFF_MARK", "b"),
        ("AFFILIATION", "Univ B"),
        ("AFF_MARK", "c"),
        ("AFFILIATION", "Univ C"),
    )
    got = assemble(inp, tags, [1.0] * len(tags), rules=rules)
    rows = [(a.family, a.affiliation) for a in got.byline.authors]
    # Both keys of "ab" go and link; a doubled "aa" is the name's own "a" and
    # one key; "d" keys nothing printed, so it stays on the name.
    assert rows == expected


@pytest.mark.parametrize(
    ("rules", "families"),
    [(True, ["Vega", "Mallen"]), (False, ["Vegaa", "Mallen"])],
)
def test_letter_keys_the_tagger_did_not_mark_still_unglue_names(rules, families):
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Juan Vegaa, Ana Mallenb"),
        ("text", "a Univ A"),
        ("text", "b Univ B"),
    )
    # The "a" opening the first affiliation line is in no field.
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Juan"),
        ("FAMILY", "Vegaa"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Mallenb"),
        ("AFFILIATION", "Univ A"),
        ("AFF_MARK", "b"),
        ("AFFILIATION", "Univ B"),
    )
    got = assemble(inp, tags, [1.0] * len(tags), rules=rules)
    assert [a.family for a in got.byline.authors] == families


@pytest.mark.parametrize(
    ("rules", "families"),
    [(True, ["Vega", "Mallen"]), (False, ["Vegaa", "Mallenb"])],
)
def test_a_letter_key_glued_to_an_affiliation_line_unglues_names(rules, families):
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Juan Vegaa, Ana Mallenb"),
        ("text", "aComputational Science, Univ A"),
        ("text", "bPhysics, Univ B"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Juan"),
        ("FAMILY", "Vegaa"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Mallenb"),
        ("AFFILIATION", "aComputational Science, Univ A"),
        ("AFFILIATION", "bPhysics, Univ B"),
    )
    got = assemble(inp, tags, [1.0] * len(tags), rules=rules)
    # Each line opens with a key glued to its first word, and no mark was tagged.
    assert [a.family for a in got.byline.authors] == families


def test_digit_keyed_names_keep_their_last_letter():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Li Costa, Juan Vega1"),
        ("text", "1 Univ A"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Li"),
        ("FAMILY", "Costa"),
        ("GIVEN", "Juan"),
        ("FAMILY", "Vega"),
        ("AUTHOR_MARK", "1"),
        ("AFF_MARK", "1"),
        ("AFFILIATION", "Univ A"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    assert [a.family for a in got.byline.authors] == ["Costa", "Vega"]


def test_a_missing_or_clipped_abstract_holds_back_only_the_title_call():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ana Lima"),
        ("abstract", "We studied handedness."),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Lima"),
        ("ABSTRACT", "We studied handedness."),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    assert got.title_reasons(0.9) == ["short_abstract"]
    assert got.byline_reasons(0.9) == []
    untagged = [t if t != "B-ABSTRACT" and t != "I-ABSTRACT" else "O" for t in tags]
    assert assemble(inp, untagged, [1.0] * len(tags)).title_reasons(0.9) == ["no_abstract"]


def test_bylines_with_a_half_name_or_a_dropped_name_go_to_the_llm():
    inp = tagger_input(("doc_title", "Title words"), ("text", "Nour S., Ana Lima and Bo Chen"))
    half = label(
        inp, ("TITLE", "Title words"), ("GIVEN", "Nour S."), ("GIVEN", "Ana"), ("FAMILY", "Lima")
    )
    reasons = assemble(inp, half, [1.0] * len(half)).byline_reasons(0.9)
    assert "incomplete_author_name" in reasons
    # "Ana" is printed between named authors but tagged as nothing.
    dropped = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Nour S."),
        ("FAMILY", "Lima"),
        ("GIVEN", "Bo"),
        ("FAMILY", "Chen"),
    )
    assert "untagged_byline_word" in assemble(inp, dropped, [1.0] * len(dropped)).byline_reasons(
        0.9
    )


def test_a_group_author_needs_no_given_name():
    inp = tagger_input(
        ("doc_title", "Title words"), ("text", "Ana Lima and the ATLAS Collaboration")
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Lima"),
        ("FAMILY", "ATLAS Collaboration"),
    )
    assert assemble(inp, tags, [1.0] * len(tags)).byline_reasons(0.9) == []


def test_a_publisher_logo_tagged_as_a_name_is_no_author():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "DE GRUYTER OPEN"),
        ("text", "ANA LIMA, Univ A"),
        ("text", "Bo Springer, Elsevier"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "DE GRUYTER"),
        ("FAMILY", "OPEN"),
        ("GIVEN", "ANA"),
        ("FAMILY", "LIMA"),
        ("AFFILIATION", "Univ A"),
        ("GIVEN", "Bo"),
        ("FAMILY", "Springer"),
        ("FAMILY", "Elsevier"),
    )
    got = assemble(inp, tags, [1.0] * len(tags), rules=True)
    # The logo goes whole; a person named like a publisher stays.
    assert [(a.given, a.family) for a in got.byline.authors] == [
        ("ANA", "LIMA"),
        ("Bo", "Springer"),
    ]
    assert got.byline_reasons(0.9) == []


def test_without_rules_a_publisher_logo_tagged_as_a_name_stays_an_author():
    inp = tagger_input(("doc_title", "Title words"), ("text", "DE GRUYTER OPEN, ANA LIMA"))
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "DE GRUYTER"),
        ("FAMILY", "OPEN"),
        ("GIVEN", "ANA"),
        ("FAMILY", "LIMA"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    assert [(a.given, a.family) for a in got.byline.authors] == [
        ("DE GRUYTER", "OPEN"),
        ("ANA", "LIMA"),
    ]


def test_a_byline_of_only_a_publisher_logo_has_no_authors():
    inp = tagger_input(("doc_title", "Title words"), ("text", "Elsevier"))
    tags = label(inp, ("TITLE", "Title words"), ("FAMILY", "Elsevier"))
    got = assemble(inp, tags, [1.0] * len(tags), rules=True)
    assert got.byline.authors == []
    assert "no_authors" in got.problems


def test_roman_numeral_keys_glued_to_family_names():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "RITA LIMAII, JOSE GANDHI"),
        ("text", "I Univ A"),
        ("text", "II Univ B"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "RITA"),
        ("FAMILY", "LIMAII"),
        ("GIVEN", "JOSE"),
        ("FAMILY", "GANDHI"),
        ("AFF_MARK", "I"),
        ("AFFILIATION", "Univ A"),
        ("AFF_MARK", "II"),
        ("AFFILIATION", "Univ B"),
    )
    rows = [
        (a.family, a.affiliation) for a in assemble(inp, tags, [1.0] * len(tags)).byline.authors
    ]
    # A single capital "I" is no glued key on an all-capitals name.
    assert rows == [("LIMA", "Univ B"), ("GANDHI", "")]


def _title_reasons(title_region: str, title: str) -> list[str]:
    inp = tagger_input(("doc_title", title_region), ("text", "Ana Lima"))
    tags = label(inp, ("TITLE", title), ("GIVEN", "Ana"), ("FAMILY", "Lima"))
    got = assemble(inp, tags, [1.0] * len(tags))
    return [p for p in got.problems if p not in ("no_abstract", "short_abstract")]


def test_titles_a_text_layer_garbled_go_to_the_llm():
    glued = "Predictors of Social AnxietyinCollegeStudents"
    assert _title_reasons(glued, glued) == ["glued_title_words"]
    doubled = "Tools for Role-modeling and Teaching Critical Thinking modeling and Teaching Critical Thinking"
    assert _title_reasons(doubled, doubled) == ["repeated_title_words"]
    # A drop-capped first word left out of the title span.
    assert _title_reasons("Wine Tasting in the Lab", "Tasting in the Lab") == [
        "untagged_title_edge"
    ]
    # Case changes in names and acronyms are no glued words, and a rubric
    # left out of the title is no dropped word.
    for plain in ("McDonald Farms and CoViD-19", "A ConvNeXt-ViT for the Long Run"):
        assert _title_reasons(plain, plain) == []
    assert _title_reasons("Original Article Tasting in the Lab", "Tasting in the Lab") == []
    assert _title_reasons("ARTICLE Tasting in the Lab", "Tasting in the Lab") == []


def test_a_drop_cap_in_a_region_of_its_own_is_part_of_the_title():
    for lead, title in (("Wine", "Tasting in the Lab"), ("W", "ine Tasting in the Lab")):
        inp = tagger_input(("doc_title", lead), ("doc_title", title), ("text", "Ana Lima"))
        tags = label(inp, ("TITLE", title), ("GIVEN", "Ana"), ("FAMILY", "Lima"))
        assert "untagged_title_edge" in assemble(inp, tags, [1.0] * len(tags)).problems


def test_a_lone_furniture_word_before_the_title_is_a_cut_first_word():
    inp = tagger_input(("doc_title", "Wine Quality and Type Prediction"), ("text", "Ana Lima"))
    tags = label(
        inp,
        ("FURNITURE", "Wine"),
        ("TITLE", "Quality and Type Prediction"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Lima"),
    )
    assert "untagged_title_edge" in assemble(inp, tags, [1.0] * len(tags)).problems
    inp = tagger_input(("doc_title", "Original Article Quality and Type"), ("text", "Ana Lima"))
    tags = label(
        inp,
        ("FURNITURE", "Original Article"),
        ("TITLE", "Quality and Type"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Lima"),
    )
    assert "untagged_title_edge" not in assemble(inp, tags, [1.0] * len(tags)).problems
    inp = tagger_input(("header", "Cell"), ("doc_title", "Quality and Type"), ("text", "Ana Lima"))
    tags = label(
        inp,
        ("FURNITURE", "Cell"),
        ("TITLE", "Quality and Type"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Lima"),
    )
    assert "untagged_title_edge" not in assemble(inp, tags, [1.0] * len(tags)).problems


def test_parallel_keyword_lists_keep_the_first_printed():
    inp = tagger_input(
        ("doc_title", "Título en español"),
        ("text", "Ana Ruiz"),
        ("text", "Palabras clave: mano; familia"),
        ("text", "Keywords: hand; family"),
    )
    tags = label(
        inp,
        ("TITLE", "Título en español"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Ruiz"),
        ("KEYWORD", "mano"),
        ("KEYWORD", "familia"),
        ("KEYWORD", "hand"),
        ("KEYWORD", "family"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    assert got.title_keywords.keywords == ["mano", "familia"]


@pytest.mark.parametrize(
    ("rules", "keywords"),
    [
        (
            True,
            [
                "grade point average (GPA)",
                "Zimbabwe (code: Q1, Q2)",
                "model (A (B))",
                "pain",
            ],
        ),
        (
            False,
            ["grade point average (GPA", "Zimbabwe (code: Q1, Q2", "model (A (B", "pain"],
        ),
    ],
)
def test_a_keyword_is_closed_over_the_parenthesis_the_tagger_left_outside(rules, keywords):
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ana Lima"),
        (
            "text",
            "Keywords: grade point average (GPA); Zimbabwe (code: Q1, Q2); model (A (B)); pain",
        ),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Lima"),
        ("KEYWORD", "grade point average (GPA"),
        ("KEYWORD", "Zimbabwe (code: Q1, Q2"),
        ("KEYWORD", "model (A (B"),
        ("KEYWORD", "pain"),
    )
    got = assemble(inp, tags, [1.0] * len(tags), rules=rules)
    assert got.title_keywords.keywords == keywords


def test_a_keyword_is_not_closed_over_a_parenthesis_that_is_not_directly_after_it():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ana Lima"),
        ("text", "Keywords: stress (acute pain); cell (s"),
        ("text", ") note"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Lima"),
        ("KEYWORD", "stress (acute"),
        ("KEYWORD", "cell (s"),
    )
    got = assemble(inp, tags, [1.0] * len(tags), rules=True)
    # A word, or another region, lies between the span and the ")".
    assert got.title_keywords.keywords == ["stress (acute", "cell (s"]


def test_a_one_name_author_in_a_given_first_byline_keeps_alone():
    inp = tagger_input(
        ("doc_title", "Pengaruh pembelajaran daring"),
        ("text", "Suprihatin, Siti Nuraini Purnamawati, Budi Santoso"),
    )
    tags = label(
        inp,
        ("TITLE", "Pengaruh pembelajaran daring"),
        ("FAMILY", "Suprihatin"),
        ("GIVEN", "Siti Nuraini"),
        ("FAMILY", "Purnamawati"),
        ("GIVEN", "Budi"),
        ("FAMILY", "Santoso"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    names = [(a.given, a.family) for a in got.byline.authors]
    assert names == [("", "Suprihatin"), ("Siti Nuraini", "Purnamawati"), ("Budi", "Santoso")]


def test_a_family_first_byline_still_pairs_initials():
    inp = tagger_input(("doc_title", "A title"), ("text", "Smith J, Jones M"))
    tags = label(
        inp,
        ("TITLE", "A title"),
        ("FAMILY", "Smith"),
        ("GIVEN", "J"),
        ("FAMILY", "Jones"),
        ("GIVEN", "M"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    assert [(a.family, a.given) for a in got.byline.authors] == [("Smith", "J"), ("Jones", "M")]


def _keyed_page(title_row, byline, *rows, spans):
    inp = tagger_input(title_row, ("text", byline), *rows)
    return inp, label(inp, *spans)


_VEGA_MALLEN = [("GIVEN", "Juan"), ("FAMILY", "Vegaa"), ("GIVEN", "Ana"), ("FAMILY", "Mallenb")]


def test_a_title_word_with_a_letter_and_a_capital_keys_no_affiliation():
    inp, tags = _keyed_page(
        ("doc_title", "Effects of eHealth on Depression in Adults"),
        "Anna Moore, Ben Clarke",
        spans=[
            ("TITLE", "Effects of eHealth on Depression in Adults"),
            ("GIVEN", "Anna"),
            ("FAMILY", "Moore"),
            ("GIVEN", "Ben"),
            ("FAMILY", "Clarke"),
        ],
    )
    got = assemble(inp, tags, _ones(inp), rules=True)
    assert [a.family for a in got.byline.authors] == ["Moore", "Clarke"]


def test_a_running_head_with_a_letter_and_a_capital_keys_no_affiliation():
    inp = tagger_input(
        ("header", "eLife 2020;9:e12345"),
        ("doc_title", "Prevalence of handedness"),
        ("text", "Anna Moore, Ben Clarke"),
    )
    tags = label(
        inp,
        ("TITLE", "Prevalence of handedness"),
        ("GIVEN", "Anna"),
        ("FAMILY", "Moore"),
        ("GIVEN", "Ben"),
        ("FAMILY", "Clarke"),
    )
    got = assemble(inp, tags, _ones(inp), rules=True)
    assert [a.family for a in got.byline.authors] == ["Moore", "Clarke"]


def test_a_title_word_beside_a_marked_author_leaves_the_next_name_whole():
    inp = tagger_input(
        ("doc_title", "Reading an eBook in Class"),
        ("text", "Anna Moore1 and Lee Blake"),
        ("text", "1 University of Twente"),
    )
    tags = label(
        inp,
        ("TITLE", "Reading an eBook in Class"),
        ("GIVEN", "Anna"),
        ("FAMILY", "Moore"),
        ("AUTHOR_MARK", "1"),
        ("GIVEN", "Lee"),
        ("FAMILY", "Blake"),
        ("AFF_MARK", "1"),
        ("AFFILIATION", "University of Twente"),
    )
    got = assemble(inp, tags, _ones(inp), rules=True)
    assert [a.family for a in got.byline.authors] == ["Moore", "Blake"]
    assert "affiliation_mark_unresolved" not in got.problems


def test_a_word_inside_an_affiliation_keys_no_affiliation():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Anna Moore, Ben Clarke"),
        ("footnote", "1 Centre for eHealth Research, University of Twente"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Anna"),
        ("FAMILY", "Moore"),
        ("GIVEN", "Ben"),
        ("FAMILY", "Clarke"),
        ("AFF_MARK", "1"),
        ("AFFILIATION", "Centre for eHealth Research, University of Twente"),
    )
    got = assemble(inp, tags, _ones(inp), rules=True)
    assert [a.family for a in got.byline.authors] == ["Moore", "Clarke"]


def test_a_key_glued_to_one_line_that_reads_as_an_affiliation_unglues_names():
    inp, tags = _keyed_page(
        ("doc_title", "Title words"),
        "Juan Vegaa, Ana Mallen",
        ("text", "aDepartment of Physics, Univ A"),
        spans=[
            ("TITLE", "Title words"),
            ("GIVEN", "Juan"),
            ("FAMILY", "Vegaa"),
            ("GIVEN", "Ana"),
            ("FAMILY", "Mallen"),
        ],
    )
    got = assemble(inp, tags, _ones(inp), rules=True)
    assert [a.family for a in got.byline.authors] == ["Vega", "Mallen"]


def test_a_key_glued_to_the_head_of_an_affiliation_after_a_name_unglues_names():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Juan Vegaa aDepartment of Physics, Univ A"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Juan"),
        ("FAMILY", "Vegaa"),
    )
    got = assemble(inp, tags, _ones(inp), rules=True)
    # The word is untagged, just after the name, and the line goes on as an institution.
    assert [a.family for a in got.byline.authors] == ["Vega"]


def test_two_keyed_lines_with_distinct_letters_key_names_whatever_they_say():
    inp, tags = _keyed_page(
        ("doc_title", "Title words"),
        "Juan Vegaa, Ana Mallenb",
        ("text", "aAdvanced Modeling Group"),
        ("text", "bQuantum Systems Team"),
        spans=[("TITLE", "Title words"), *_VEGA_MALLEN],
    )
    got = assemble(inp, tags, _ones(inp), rules=True)
    assert [a.family for a in got.byline.authors] == ["Vega", "Mallen"]


def test_one_keyed_line_that_names_no_institution_keys_nothing():
    inp, tags = _keyed_page(
        ("doc_title", "Title words"),
        "Juan Vegaa, Ana Mallenb",
        ("text", "aAdvanced Modeling Group"),
        spans=[("TITLE", "Title words"), *_VEGA_MALLEN],
    )
    got = assemble(inp, tags, _ones(inp), rules=True)
    assert [a.family for a in got.byline.authors] == ["Vegaa", "Mallenb"]


_BYLINE = [("GIVEN", "Anna"), ("FAMILY", "Smith")]


def _untagged(*rows, spans=()):
    """The institution text read from the print, with the byline "Anna Smith" tagged."""
    inp = tagger_input(("doc_title", "Prevalence of handedness"), *rows)
    tags = label(inp, ("TITLE", "Prevalence of handedness"), *spans)
    return assemble(inp, tags, _ones(inp), rules=True).untagged_affiliations


@pytest.mark.parametrize(
    "line",
    [
        "Journal of Medical Internet Research",
        "Academy of Management Journal",
        "BMC Medical Education (2020) 20:123",
        "Downloaded from https://academic.oup.com by University of Liverpool on 3 March 2020",
        "Access provided by University of Michigan Library",
        "Funded by the National Science Foundation grant 12345",
        "Supported by the Wellcome Trust Foundation",
        "2020 Elsevier Ltd. All rights reserved.",
        "Wiley Periodicals, Inc.",
        "Send reprint requests to Jane Doe, Department of Psychology, Yale",
        "Jane Doe is a professor at the University of Michigan.",
        "Cite this article as: Smith A. Medical care in Spain. Hosp Med 2020.",
        "123 Hospital Road, Springfield, IL 62704, USA",
    ],
)
def test_a_line_that_is_no_affiliation_is_not_read_as_one(line):
    rows = [("text", "Anna Smith"), ("text", line), ("abstract", "Abstract")]
    assert _untagged(*rows, spans=[*_BYLINE]) == []


@pytest.mark.parametrize(
    "line",
    ["Department of Psychology, Yale University", "Professor of Medical Education, Univ of Oslo"],
)
def test_a_line_that_names_an_institution_is_read(line):
    rows = [("text", "Anna Smith"), ("text", line), ("abstract", "Abstract")]
    assert _untagged(*rows, spans=[*_BYLINE]) == [line]


@pytest.mark.parametrize(
    "run",
    [
        "Professor of Medical Education",
        "and the Medical Research Council Trial Group",
        "A Study of Medical Students in Spain",
        "Received 3 March from the College of Medicine",
    ],
)
def test_a_run_after_a_name_that_is_no_affiliation_is_not_read_as_one(run):
    rows = [("text", f"Anna Smith {run}"), ("abstract", "Abstract")]
    assert _untagged(*rows, spans=[*_BYLINE]) == []


def test_a_run_after_a_name_that_names_an_institution_is_read():
    rows = [("text", "Anna Smith University of Michigan"), ("abstract", "Abstract")]
    assert _untagged(*rows, spans=[*_BYLINE]) == ["University of Michigan"]


def test_text_after_the_first_heading_of_the_body_is_not_read():
    rows = [
        ("text", "Anna Smith"),
        ("paragraph_title", "1. Introduction"),
        ("text", "University of Michigan Medical School"),
    ]
    assert _untagged(*rows, spans=[*_BYLINE]) == []
    # The same text before any heading is read.
    rows = [("text", "Anna Smith"), ("text", "University of Michigan Medical School")]
    assert _untagged(*rows, spans=[*_BYLINE]) == ["University of Michigan Medical School"]


def test_a_sentence_of_the_paper_is_not_read_when_no_abstract_was_tagged():
    rows = [
        ("text", "Anna Smith"),
        ("text", "We recruited students at the University of Michigan and a medical school."),
    ]
    assert _untagged(*rows, spans=[*_BYLINE]) == []


def test_affiliations_on_the_second_page_after_a_cover_sheet_are_read():
    pages = regions(("doc_title", "Prevalence of handedness"), ("text", "Cover sheet"), page=0)
    pages += regions(
        ("text", "Anna Smith"), ("text", "Department of Psychology, Yale University"), page=1
    )
    inp = build_tagger_input(pages)
    tags = label(inp, ("TITLE", "Prevalence of handedness"), *_BYLINE)
    got = assemble(inp, tags, _ones(inp), rules=True)
    assert got.untagged_affiliations == ["Department of Psychology, Yale University"]


def test_a_run_of_keys_leaves_a_family_name_four_letters():
    inp = tagger_input(
        ("doc_title", "Title words"),
        ("text", "Ali Sarab, Juan Riosab"),
        ("text", "a Univ A"),
        ("text", "b Univ B"),
    )
    tags = label(
        inp,
        ("TITLE", "Title words"),
        ("GIVEN", "Ali"),
        ("FAMILY", "Sarab"),
        ("GIVEN", "Juan"),
        ("FAMILY", "Riosab"),
        ("AFF_MARK", "a"),
        ("AFFILIATION", "Univ A"),
        ("AFF_MARK", "b"),
        ("AFFILIATION", "Univ B"),
    )
    got = assemble(inp, tags, _ones(inp), rules=True)
    rows = [(a.family, a.affiliation) for a in got.byline.authors]
    # "Sarab" would be "Sar" keyed "a" and "b"; only the "b" goes, as for a single key.
    assert rows == [("Sara", "Univ B"), ("Rios", "Univ A; Univ B")]
