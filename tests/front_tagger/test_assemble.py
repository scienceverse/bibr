from __future__ import annotations

from bibr.front_tagger.assemble import assemble

from .helpers import label, tagger_input


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
        ("abstract", "We studied handedness in 400 families. It is heritable."),
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
        ("ABSTRACT", "We studied handedness in 400 families. It is heritable."),
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
    assert tk.abstract == "We studied handedness in 400 families. It is heritable."
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
    assert got.problems == []


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
    assert got.problems == []
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
    assert got.problems == []


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


def test_parallel_abstracts_prefer_english_and_titles_the_first_printed():
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
    assert got.title_keywords.abstract == "We studied the hand in families."


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
    assert got.problems == ["no_title", "no_authors"]
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
