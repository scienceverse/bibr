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
