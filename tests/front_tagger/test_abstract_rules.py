"""The fast-mode readings (``rules``) of the abstract and the title's first region."""

from __future__ import annotations

import pytest

from bibr.front_tagger.assemble import assemble
from bibr.front_tagger.contract import build_tagger_input

from .helpers import label, regions, tagger_input

TITLE = ("doc_title", "Quiet rivers of the plain")
BODY = (
    "We counted the reeds along the quiet rivers of the plain over nine summers and found "
    "that the reeds grew taller where the water ran slowly and the banks were soft and that "
    "they thinned where the water ran fast"
)
SECOND = "The herons followed the reeds and nested where the reeds were tallest"
SPANISH = (
    "Contamos los juncos de los ríos tranquilos de la llanura durante nueve veranos y "
    "encontramos que los juncos crecían más altos donde el agua corría despacio y las "
    "orillas eran blandas"
)


def _ones(inp):
    return [1.0] * len(inp.words)


def _read(inp, tags, conf=None, *, rules=True):
    return assemble(inp, tags, conf or _ones(inp), rules=rules)


def _abstract(inp, tags, conf=None, *, rules=True):
    return _read(inp, tags, conf, rules=rules).title_keywords.abstract


def _titled(inp, *spans):
    return label(inp, ("TITLE", TITLE[1]), *spans)


# --- the closing marks the tagger left out


def test_the_closing_full_stop_the_tagger_left_out_is_read():
    inp = tagger_input(TITLE, ("abstract", f"{BODY}."))
    tags = _titled(inp, ("ABSTRACT", BODY))
    assert _abstract(inp, tags) == f"{BODY}."
    assert _abstract(inp, tags, rules=False) == BODY


def test_a_full_stop_between_two_runs_is_read():
    inp = tagger_input(TITLE, ("abstract", f"{BODY}. {SECOND}?"))
    tags = _titled(inp, ("ABSTRACT", BODY), ("ABSTRACT", SECOND))
    assert _abstract(inp, tags) == f"{BODY}. {SECOND}?"


def test_a_mark_in_the_next_region_or_tagged_otherwise_is_not_read():
    inp = tagger_input(TITLE, ("abstract", BODY), ("text", ". Reed survey 4"))
    tags = _titled(inp, ("ABSTRACT", BODY))
    assert _abstract(inp, tags) == BODY
    inp = tagger_input(TITLE, ("abstract", f"{BODY}."))
    tags = _titled(inp, ("ABSTRACT", BODY), ("FURNITURE", "."))
    assert _abstract(inp, tags) == BODY


# --- companion summaries


def test_a_companion_summary_before_the_abstract_is_dropped():
    inp = tagger_input(
        TITLE,
        ("abstract", f"Simple Summary: {SECOND}."),
        ("abstract", f"Abstract: {BODY}."),
    )
    tags = _titled(
        inp, ("ABSTRACT", f"Simple Summary: {SECOND}."), ("ABSTRACT", f"Abstract: {BODY}.")
    )
    assert _abstract(inp, tags) == f"{BODY}."
    assert _abstract(inp, tags, rules=False).startswith("Simple Summary")


def test_a_companion_summary_after_the_abstract_is_dropped_up_to_the_next_abstract_label():
    inp = tagger_input(
        TITLE,
        ("abstract", f"{BODY}."),
        ("abstract", f"Plain language summary {SECOND}."),
        ("abstract", "The reeds are counted every summer."),
    )
    tags = _titled(
        inp,
        ("ABSTRACT", f"{BODY}."),
        ("ABSTRACT", f"Plain language summary {SECOND}."),
        ("ABSTRACT", "The reeds are counted every summer."),
    )
    assert _abstract(inp, tags) == f"{BODY}."


def test_the_heading_of_a_companion_summary_counts_when_untagged():
    inp = tagger_input(TITLE, ("abstract", f"{BODY}."), ("abstract", f"Highlights {SECOND}."))
    tags = _titled(inp, ("ABSTRACT", f"{BODY}."), ("ABSTRACT", f"{SECOND}."))
    assert _abstract(inp, tags) == f"{BODY}."


@pytest.mark.parametrize(("layout", "dropped"), [("paragraph_title", True), ("text", False)])
def test_a_box_heading_opens_a_companion_only_as_a_heading_region(layout, dropped):
    inp = tagger_input(
        TITLE, ("abstract", f"{BODY}."), (layout, "Key messages"), ("abstract", f"{SECOND}.")
    )
    tags = _titled(
        inp, ("ABSTRACT", f"{BODY}."), ("ABSTRACT", "Key messages"), ("ABSTRACT", f"{SECOND}.")
    )
    expected = f"{BODY}." if dropped else f"{BODY}. Key messages {SECOND}."
    assert _abstract(inp, tags) == expected


@pytest.mark.parametrize("box", [f"Highlights • {SECOND}.", f"HIGHLIGHTS: {SECOND}."])
def test_a_companion_heading_followed_by_a_mark_counts(box):
    inp = tagger_input(TITLE, ("abstract", f"{BODY}."), ("abstract", box))
    tags = _titled(inp, ("ABSTRACT", f"{BODY}."), ("ABSTRACT", box))
    assert _abstract(inp, tags) == f"{BODY}."


def test_a_region_going_on_with_the_verb_highlights_is_no_companion():
    # The abstract runs on into the next region in mid-sentence.
    first = "We counted the reeds along the quiet rivers of the plain. This study"
    rest = "highlights the need to keep the banks of slow rivers soft and wet."
    inp = tagger_input(TITLE, ("abstract", first), ("abstract", rest))
    tags = _titled(inp, ("ABSTRACT", first), ("ABSTRACT", rest))
    assert _abstract(inp, tags) == f"{first} {rest}"


def test_a_summary_alone_is_kept():
    inp = tagger_input(TITLE, ("abstract", f"Lay summary {BODY}."))
    tags = _titled(inp, ("ABSTRACT", f"Lay summary {BODY}."))
    assert _abstract(inp, tags) == f"Lay summary {BODY}."


# --- heading words between two runs


@pytest.mark.parametrize("heading", ["Sommario", "Streszczenie", "Özet", "Abstrak"])
def test_a_heading_word_the_rules_added_splits_runs_only_with_the_rules(heading):
    first = "Bu tarama ovanın sakin nehirleri boyunca sazları dokuz yaz saydı."
    inp = tagger_input(TITLE, ("abstract", first), ("text", heading), ("abstract", f"{BODY}."))
    tags = _titled(inp, ("ABSTRACT", first), ("ABSTRACT", f"{BODY}."))
    assert _abstract(inp, tags, rules=False) == f"{first} {BODY}."
    assert _abstract(inp, tags) == first


# --- the head of the region the tagger cut off


def _cut_head(layout="abstract", start_confidence=0.3, before=None):
    rows = [TITLE, *([("abstract", f"{before}.")] if before else []), (layout, f"Purpose {BODY}.")]
    inp = tagger_input(*rows)
    spans = [("ABSTRACT", f"{before}.")] if before else []
    tags = _titled(inp, *spans, ("ABSTRACT", BODY[len("We ") :]))
    conf = _ones(inp)
    conf[[i for i, t in enumerate(tags) if t == "B-ABSTRACT"][-1]] = start_confidence
    return inp, tags, conf


def test_an_untagged_head_of_an_abstract_region_is_read():
    inp, tags, conf = _cut_head()
    assert _abstract(inp, tags, conf) == f"Purpose {BODY}."
    assert _abstract(inp, tags, conf, rules=False) == BODY[len("We ") :]


@pytest.mark.parametrize(
    ("layout", "start_confidence"), [("text", 0.3), ("abstract", 0.8)], ids=["text", "sure"]
)
def test_a_head_is_not_read_outside_an_abstract_region_or_after_a_sure_start(
    layout, start_confidence
):
    inp, tags, conf = _cut_head(layout, start_confidence)
    assert _abstract(inp, tags, conf) == f"{BODY[len('We ') :]}."


def test_a_head_that_repeats_the_abstract_is_a_second_print_and_not_read():
    # A doubled text layer prints the end of the first region again.
    before = "The reeds bent low under the heavy rains of the last summer"
    inp = tagger_input(
        TITLE,
        ("abstract", f"{before}."),
        ("abstract", f"bent low under the heavy rains of the last summer. {SECOND}."),
    )
    tags = _titled(inp, ("ABSTRACT", f"{before}."), ("ABSTRACT", SECOND))
    conf = _ones(inp)
    conf[[i for i, t in enumerate(tags) if t == "B-ABSTRACT"][-1]] = 0.3
    assert _abstract(inp, tags, conf) == f"{before}. {SECOND}."


def test_a_head_holding_a_labelled_version_is_not_read():
    # One region prints both versions; the tagger read the English one only.
    inp = tagger_input(TITLE, ("abstract", f"Resumen: {SPANISH}. Abstract: {BODY}."))
    tags = _titled(inp, ("ABSTRACT", BODY))
    conf = _ones(inp)
    conf[tags.index("B-ABSTRACT")] = 0.3
    assert _abstract(inp, tags, conf) == f"{BODY}."


# --- a version in another language


def test_a_second_language_version_in_the_next_region_ends_the_abstract():
    inp = tagger_input(TITLE, ("abstract", f"{BODY}."), ("abstract", f"{SPANISH}."))
    tags = _titled(inp, ("ABSTRACT", f"{BODY}."), ("ABSTRACT", f"{SPANISH}."))
    assert _abstract(inp, tags) == f"{BODY}."
    assert _abstract(inp, tags, rules=False) == f"{BODY}. {SPANISH}."


def test_a_short_region_in_another_language_does_not_end_the_abstract():
    tail = "Los juncos de la llanura y los ríos."
    inp = tagger_input(TITLE, ("abstract", f"{BODY}."), ("abstract", tail))
    tags = _titled(inp, ("ABSTRACT", f"{BODY}."), ("ABSTRACT", tail))
    assert _abstract(inp, tags) == f"{BODY}. {tail}"


def test_a_passage_in_another_language_after_a_long_abstract_does_not_end_it():
    # The abstract's last region quotes a survey item; no second version starts.
    quote = (
        "Most answered the item «¿Con qué frecuencia visita usted los ríos de la llanura "
        "durante el verano y por qué?» with por la pesca."
    )
    rows = [("abstract", f"{BODY}. {SECOND}."), ("abstract", f"{BODY}."), ("abstract", quote)]
    inp = tagger_input(TITLE, *rows)
    tags = _titled(inp, *[("ABSTRACT", text) for _, text in rows])
    assert _abstract(inp, tags) == f"{BODY}. {SECOND}. {BODY}. {quote}"


def test_a_passage_quoted_in_another_language_does_not_end_an_abstract_of_its_length():
    first = (
        "We asked nine hundred people of the plain how often they visit its quiet rivers "
        "and why, and set their answers beside the reed counts of nine summers."
    )
    quote = (
        "Most answered the item «¿Con qué frecuencia visita usted los ríos de la llanura "
        "durante el verano y por qué?» with en el verano, por la pesca y por los juncos."
    )
    inp = tagger_input(TITLE, ("abstract", first), ("abstract", quote))
    tags = _titled(inp, ("ABSTRACT", first), ("ABSTRACT", quote))
    assert _abstract(inp, tags) == f"{first} {quote}"


def test_a_labelled_version_in_another_language_ends_the_abstract_whatever_its_length():
    short = SPANISH[: SPANISH.index(" donde")]  # a third of the English text
    rows = [
        ("abstract", f"{BODY}. {SECOND}."),
        ("abstract", f"{BODY}."),
        ("abstract", f"Resumen: {short}."),
    ]
    inp = tagger_input(TITLE, *rows)
    tags = _titled(inp, *[("ABSTRACT", text) for _, text in rows])
    assert _abstract(inp, tags) == f"{BODY}. {SECOND}. {BODY}."


def test_a_second_script_ends_the_abstract():
    cyrillic = (
        "Мы считали тростник вдоль тихих рек равнины девять лет и нашли что тростник рос "
        "выше там где вода текла медленно"
    )
    inp = tagger_input(TITLE, ("abstract", f"{BODY}."), ("abstract", f"{cyrillic}."))
    tags = _titled(inp, ("ABSTRACT", f"{BODY}."), ("ABSTRACT", f"{cyrillic}."))
    assert _abstract(inp, tags) == f"{BODY}."


# --- a leading abstract label


@pytest.mark.parametrize(
    ("printed", "read"),
    [
        (f"Abstract: {BODY}.", f"{BODY}."),
        (f"SUMMARY {BODY}.", f"{BODY}."),
        (f"Resumo. {BODY}.", f"{BODY}."),
        (f"Abstract—{BODY}.", f"{BODY}."),
        (f"Abstract Background {BODY}.", f"Background {BODY}."),
        (f"Summary statistics of {BODY}.", f"Summary statistics of {BODY}."),
    ],
)
def test_a_leading_abstract_label_is_dropped(printed, read):
    inp = tagger_input(TITLE, ("abstract", printed))
    tags = _titled(inp, ("ABSTRACT", printed))
    assert _abstract(inp, tags) == read
    assert _abstract(inp, tags, rules=False) == printed


def test_a_term_opening_the_abstract_under_its_heading_is_kept():
    # The heading is printed above; "Abstract Reed Graphs" is the abstract's subject.
    text = f"Abstract Reed Graphs (ARG) are a way to count reeds. {BODY}."
    inp = tagger_input(TITLE, ("paragraph_title", "Abstract"), ("abstract", text))
    tags = _titled(inp, ("ABSTRACT", text))
    assert _abstract(inp, tags) == text
    # Punctuated, it is the label.
    inp = tagger_input(TITLE, ("paragraph_title", "Abstract"), ("abstract", f"Abstract: {BODY}."))
    tags = _titled(inp, ("ABSTRACT", f"Abstract: {BODY}."))
    assert _abstract(inp, tags) == f"{BODY}."


def test_an_abstract_of_only_its_label_is_no_abstract():
    inp = tagger_input(TITLE, ("abstract", "Abstract:"))
    tags = _titled(inp, ("ABSTRACT", "Abstract:"))
    got = _read(inp, tags)
    assert got.title_keywords.abstract is None
    assert "no_abstract" in got.problems


# --- a region the layout calls an abstract, with no abstract tagged

LONG = f"{BODY} and {SECOND.lower()}."  # 40 words and more


@pytest.mark.parametrize(
    "rows",
    [
        [("abstract", f"Abstract {LONG}")],
        [("abstract", f"Özet: {LONG}")],
        [("paragraph_title", "Streszczenie"), ("abstract", LONG)],
        [("abstract", LONG), ("text", "Keywords: reeds; rivers")],
    ],
    ids=["headed", "headed-colon", "after-heading", "before-keywords"],
)
def test_an_untagged_layout_abstract_is_read_when_it_reads_as_one(rows):
    inp = tagger_input(TITLE, *rows)
    got = _read(inp, label(inp, ("TITLE", TITLE[1])))
    assert got.title_keywords.abstract == LONG
    assert "no_abstract" not in got.problems
    assert got.confidence["abstract"] == 0.0  # no tag vouches for it
    assert _read(inp, label(inp, ("TITLE", TITLE[1])), rules=False).title_keywords.abstract is None


@pytest.mark.parametrize(
    "rows",
    [
        [("abstract", LONG)],  # nothing says it is an abstract
        [("abstract", f"Abstract {SECOND}.")],  # too short for one
        [("text", f"Abstract {LONG}")],  # not an abstract region
    ],
    ids=["no-cue", "short", "text-region"],
)
def test_a_layout_abstract_needs_a_cue_and_an_abstracts_length(rows):
    inp = tagger_input(TITLE, *rows)
    got = _read(inp, label(inp, ("TITLE", TITLE[1])))
    assert got.title_keywords.abstract is None


def test_a_layout_abstract_carrying_another_field_is_not_read():
    inp = tagger_input(TITLE, ("abstract", f"Abstract {LONG}"))
    tags = label(inp, ("TITLE", TITLE[1]), ("AFFILIATION", "quiet rivers of the plain"))
    assert _read(inp, tags).title_keywords.abstract is None


# --- where the abstract runs off the input


def _two_pages(last: str, *after: tuple[str, str]):
    rows = regions(TITLE, ("text", "Opening words of the paper"), page=0)
    rows += regions(("abstract", last), *after, page=1)
    return build_tagger_input(rows)


def test_an_abstract_cut_by_the_end_of_the_input_names_its_page():
    inp = _two_pages(BODY, ("footer", "Reed Review 4"))
    tags = label(inp, ("TITLE", TITLE[1]), ("ABSTRACT", BODY))
    assert _read(inp, tags).abstract_runs_on == 1
    assert _read(inp, tags, rules=False).abstract_runs_on is None


@pytest.mark.parametrize(
    ("last", "after"),
    [
        (f"{BODY}.", ("footer", "Reed Review 4")),  # its last sentence is closed
        (BODY, ("text", "The next section of the paper")),  # the paper goes on
    ],
)
def test_an_abstract_that_ends_on_the_page_does_not_run_off(last, after):
    inp = _two_pages(last, after)
    tags = label(inp, ("TITLE", TITLE[1]), ("ABSTRACT", BODY))
    assert _read(inp, tags).abstract_runs_on is None


def test_an_abstract_before_the_last_page_does_not_run_off():
    rows = regions(TITLE, ("abstract", BODY), page=0)
    rows += regions(("text", "Opening words of the paper"), page=1)
    inp = build_tagger_input(rows)
    tags = label(inp, ("TITLE", TITLE[1]), ("ABSTRACT", BODY))
    assert _read(inp, tags).abstract_runs_on is None


# --- labels above the title


@pytest.mark.parametrize(
    "head",
    ["Running head: QUIET RIVERS", "Chapter 3", "Part II", "* * *"],
)
def test_a_label_region_run_into_the_title_is_dropped(head):
    inp = tagger_input(("text", head), TITLE)
    tags = label(inp, ("TITLE", head), ("TITLE", TITLE[1]))
    got = _read(inp, tags)
    assert got.title_keywords.title == TITLE[1]
    assert _read(inp, tags, rules=False).title_keywords.title == f"{head} {TITLE[1]}"


@pytest.mark.parametrize(
    ("rubric", "tagged"),
    [("Research article | Field notes", "Research"), ("DELTA NOTES | FIELD NOTES", "DELTA NOTES")],
    ids=["rubric-word", "capitals"],
)
def test_a_rubric_the_tagger_stopped_the_title_in_is_dropped(rubric, tagged):
    inp = tagger_input(("text", rubric), TITLE)
    tags = label(inp, ("TITLE", tagged), ("TITLE", TITLE[1]))
    assert _read(inp, tags).title_keywords.title == TITLE[1]


@pytest.mark.parametrize(
    ("first", "tagged"),
    [
        ("Growth of reeds in the quiet rivers of the plain", "Growth of reeds in the quiet rivers"),
        ("GROWTH OF REEDS IN THE QUIET RIVERS OF THE PLAIN", "GROWTH OF REEDS IN THE QUIET RIVERS"),
        ("A review of reed growth in the plain", "A review of reed growth"),
    ],
    ids=["plain", "capitals", "rubric-word"],
)
def test_a_title_line_whose_end_the_tagger_missed_is_no_rubric(first, tagged):
    inp = tagger_input(("doc_title", first), ("doc_title", "over nine summers"))
    tags = label(inp, ("TITLE", tagged), ("TITLE", "over nine summers"))
    assert _read(inp, tags).title_keywords.title == f"{tagged} over nine summers"


@pytest.mark.parametrize(
    ("first_row", "tagged"),
    [
        ("Field notes", "Field"),  # one untagged word: the title's own line may go on
        ("Quiet rivers of", "Quiet rivers of"),  # a title printed over two regions
    ],
)
def test_a_title_over_two_regions_keeps_its_first_line(first_row, tagged):
    inp = tagger_input(("doc_title", first_row), ("doc_title", "the plain"))
    tags = label(inp, ("TITLE", tagged), ("TITLE", "the plain"))
    assert _read(inp, tags).title_keywords.title == f"{tagged} the plain"


def test_a_title_continued_on_an_inside_tag_keeps_its_first_line():
    inp = tagger_input(("doc_title", "Quiet rivers of the"), ("doc_title", "wide plain"))
    tags = label(inp, ("TITLE", "Quiet rivers"), ("TITLE", "wide plain"))
    tags[tags.index("B-TITLE", 2)] = "I-TITLE"
    assert _read(inp, tags).title_keywords.title == "Quiet rivers wide plain"
