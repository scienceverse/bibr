from __future__ import annotations

from bibr.front_tagger.assemble import _mark_keys, assemble

from .helpers import label, tagger_input


def _byline_page(note: str):
    inp = tagger_input(
        ("doc_title", "Severity levels of punishment"),
        ("text", "Ana Silva1*, Rui Costa2†"),
        ("text", "1 Univ A"),
        ("text", "2 Univ B"),
        ("text", note),
    )
    tags = label(
        inp,
        ("TITLE", "Severity levels of punishment"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Silva"),
        ("AUTHOR_MARK", "1*"),
        ("GIVEN", "Rui"),
        ("FAMILY", "Costa"),
        ("AUTHOR_MARK", "2†"),
        ("AFF_MARK", "1"),
        ("AFFILIATION", "Univ A"),
        ("AFF_MARK", "2"),
        ("AFFILIATION", "Univ B"),
    )
    return assemble(inp, tags, [1.0] * len(tags), rules=True)


def test_other_note_in_the_correspondence_footnote_flags_no_one():
    got = _byline_page("* Corresponding author: Univ A. † These authors contributed equally.")
    assert [a.corresponding for a in got.byline.authors] == [True, False]


def test_correspondence_note_after_another_note_keeps_its_own_symbol():
    got = _byline_page("† These authors contributed equally. * Correspondence to: Univ A.")
    assert [a.corresponding for a in got.byline.authors] == [True, False]


def test_symbol_printed_after_the_cue_still_flags():
    got = _byline_page("Corresponding author † : Univ B.")
    assert [a.corresponding for a in got.byline.authors] == [False, True]


def test_number_range_in_marks_is_not_expanded_past_a_key():
    # A year or a postcode read as a mark after a dash must not expand into
    # millions of keys (a ten-digit number would not finish).
    assert _mark_keys(["1", "–", "3000000"]) == ["1", "3000000"]
    assert _mark_keys(["1", "–", "4"]) == ["1", "2", "3", "4"]


def test_subtitle_of_a_later_title_is_not_joined_to_the_first():
    inp = tagger_input(
        ("doc_title", "Punishment of children"),
        ("text", "Ana Silva"),
        ("doc_title", "Punição de crianças"),
        ("paragraph_title", "uma análise de clusters"),
    )
    tags = label(
        inp,
        ("TITLE", "Punishment of children"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Silva"),
        ("TITLE", "Punição de crianças"),
        ("SUBTITLE", "uma análise de clusters"),
    )
    got = assemble(inp, tags, [1.0] * len(tags))
    assert got.title_keywords.title == "Punishment of children"


def _two_row_title(first: str, second: str):
    inp = tagger_input(("doc_title", first), ("text", second), ("text", "Ana Silva"))
    tags = label(inp, ("TITLE", first), ("GIVEN", "Ana"), ("FAMILY", "Silva"))
    # The second row continues the title's span, as the tagger reads it.
    for i, w in enumerate(inp.words):
        if w.region == 1:
            tags[i] = "I-TITLE"
    return inp, tags


def test_a_translation_the_title_tags_run_on_into_is_left_out():
    inp, tags = _two_row_title(
        "Levels of harsh discipline of children in the home: a cluster analysis",
        "Níveis de disciplina severa de crianças em casa: uma análise de clusters",
    )
    got = assemble(inp, tags, [1.0] * len(tags), rules=True)
    assert got.title_keywords.title == (
        "Levels of harsh discipline of children in the home: a cluster analysis"
    )


def test_a_title_wrapped_over_two_rows_in_one_language_stays_whole():
    inp, tags = _two_row_title(
        "Harsh parenting and fearfulness in toddlerhood interact",
        "to predict the amplitudes of an error-related negativity",
    )
    got = assemble(inp, tags, [1.0] * len(tags), rules=True)
    assert got.title_keywords.title == (
        "Harsh parenting and fearfulness in toddlerhood interact "
        "to predict the amplitudes of an error-related negativity"
    )
