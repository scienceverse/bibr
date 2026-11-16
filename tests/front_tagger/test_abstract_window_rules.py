"""The later-window readings of ``tag_front_matter`` with ``rules``: a fragment of an
abstract, an abstract cut by the end of its window, and a title read in part."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bibr.front_tagger.runtime import tag_front_matter

from .helpers import label, regions

TITLE = "Quiet rivers of the plain"
LONG = (
    "We counted the reeds along the quiet rivers of the plain over nine summers and found "
    "that the reeds grew taller where the water ran slowly and the banks were soft and that "
    "they thinned where the water ran fast."
)
FRAGMENT = "The reeds grew taller."
CUT = "We counted the reeds along the quiet rivers of the plain over nine summers and found"
REST = "that the reeds grew taller where the water ran slowly."


class _WindowTagger:
    """Tags the phrases given for the page a window starts on, each at its confidence."""

    def __init__(self, by_page: dict[int, list[tuple[str, str, float]]]):
        self.by_page = by_page
        self.windows: list[int] = []

    def tag(self, inp):
        page = inp.regions[0].page
        self.windows.append(page)
        spans = self.by_page.get(page, [])
        tags = label(inp, *[(f, p) for f, p, _ in spans]) if spans else ["O"] * len(inp.words)
        conf = [1.0] * len(tags)
        cursor = 0
        for _, phrase, p in spans:
            start = inp.text.index(phrase, cursor)
            cursor = start + len(phrase)
            for i, word in enumerate(inp.words):
                if word.start >= start and word.end <= cursor:
                    conf[i] = p
        return tags, conf


def _front(*rows, page=0):
    return regions(*rows, page=page)


# --- a short abstract the tagger is unsure of


def _fragment_paper():
    rows = _front(("doc_title", TITLE), ("text", "Ana Ruiz"), ("abstract", FRAGMENT))
    rows += _front(("text", "Author note and contact details"), page=1)
    rows += _front(("text", "Abstract"), ("abstract", LONG), page=2)
    return rows


def _fragment_tagger(fragment_confidence: float, later: str = LONG):
    head = [("TITLE", TITLE, 1.0), ("GIVEN", "Ana", 1.0), ("FAMILY", "Ruiz", 1.0)]
    return _WindowTagger(
        {
            0: [*head, ("ABSTRACT", FRAGMENT, fragment_confidence)],
            1: [("ABSTRACT", later, 1.0)],
            2: [("ABSTRACT", later, 1.0)],
        }
    )


def test_an_unsure_short_abstract_is_looked_for_in_a_later_window():
    tagger = _fragment_tagger(0.5)
    got = tag_front_matter(tagger, _fragment_paper(), rules=True, min_confidence=0.9)
    assert got.title_keywords.abstract == LONG
    assert "short_abstract" not in got.problems
    assert got.confidence["abstract"] == 1.0
    assert got.title_keywords.title == TITLE
    assert tagger.windows == [0, 1]


@pytest.mark.parametrize(
    ("confidence", "rules", "min_confidence"),
    [(1.0, True, 0.9), (0.5, True, None), (0.5, False, 0.9)],
    ids=["sure", "no-bar", "no-rules"],
)
def test_a_short_abstract_stays_when_sure_or_without_the_rules(confidence, rules, min_confidence):
    tagger = _fragment_tagger(confidence)
    got = tag_front_matter(tagger, _fragment_paper(), rules=rules, min_confidence=min_confidence)
    assert got.title_keywords.abstract == FRAGMENT
    assert "short_abstract" in got.problems
    assert tagger.windows == [0]


def test_a_later_fragment_does_not_replace_the_first():
    tagger = _fragment_tagger(0.5, later="Reeds were counted.")
    rows = _fragment_paper()
    rows[-1] = replace(rows[-1], text="Reeds were counted.")
    got = tag_front_matter(tagger, rows, rules=True, min_confidence=0.9)
    assert got.title_keywords.abstract == FRAGMENT
    assert "short_abstract" in got.problems


# --- an abstract cut by the end of its window


def _cut_paper(*, next_page: bool = True):
    rows = _front(("doc_title", TITLE), ("text", "Ana Ruiz"), ("text", "Opening words"))
    rows += _front(("abstract", CUT), ("footer", "Reed Review 4"), page=1)
    if next_page:
        rows += _front(("abstract", REST), ("text", "The next section of the paper"), page=2)
    return rows


def _cut_tagger(later: list[tuple[str, str, float]]):
    head = [("TITLE", TITLE, 1.0), ("GIVEN", "Ana", 1.0), ("FAMILY", "Ruiz", 1.0)]
    return _WindowTagger({0: [*head, ("ABSTRACT", CUT, 1.0)], 1: later})


def test_an_abstract_cut_by_the_end_of_its_window_is_read_on():
    tagger = _cut_tagger([("ABSTRACT", CUT, 1.0), ("ABSTRACT", REST, 1.0)])
    got = tag_front_matter(tagger, _cut_paper(), rules=True)
    assert got.title_keywords.abstract == f"{CUT} {REST}"
    assert tagger.windows == [0, 1]
    assert got.abstract_runs_on is None


def test_a_label_in_front_of_the_abstract_read_on_does_not_hide_it():
    rows = _cut_paper()
    rows[3] = replace(rows[3], text=f"Background {CUT}")
    tagger = _cut_tagger([("ABSTRACT", f"Background {CUT}", 1.0), ("ABSTRACT", REST, 1.0)])
    got = tag_front_matter(tagger, rows, rules=True)
    assert got.title_keywords.abstract == f"Background {CUT} {REST}"


def test_another_abstract_in_the_next_window_does_not_replace_the_cut_one():
    tagger = _cut_tagger([("ABSTRACT", REST, 1.0)])
    got = tag_front_matter(tagger, _cut_paper(), rules=True)
    assert got.title_keywords.abstract == CUT
    assert tagger.windows == [0, 1]


def test_a_cut_abstract_is_not_read_on_without_a_next_page_or_without_the_rules():
    tagger = _cut_tagger([("ABSTRACT", CUT, 1.0), ("ABSTRACT", REST, 1.0)])
    got = tag_front_matter(tagger, _cut_paper(next_page=False), rules=True)
    assert got.title_keywords.abstract == CUT
    assert got.abstract_runs_on == 1
    got = tag_front_matter(tagger, _cut_paper(), rules=False)
    assert got.title_keywords.abstract == CUT
    assert tagger.windows == [0, 0]


@pytest.mark.parametrize("label", ["image", "table"])
def test_a_cut_abstract_is_not_tagged_again_when_the_next_page_holds_no_text(label):
    # The page after the abstract's is a figure: the window from the abstract's
    # page would read the first window's text again.
    rows = _front(("doc_title", TITLE), ("text", "Ana Ruiz"), ("abstract", CUT))
    rows += _front((label, "Figure 1"), page=1)
    rows += _front(("abstract", REST), page=2)
    tagger = _WindowTagger(
        {
            0: [
                ("TITLE", TITLE, 1.0),
                ("GIVEN", "Ana", 1.0),
                ("FAMILY", "Ruiz", 1.0),
                ("ABSTRACT", CUT, 1.0),
            ]
        }
    )
    got = tag_front_matter(tagger, rows, rules=True)
    assert got.title_keywords.abstract == CUT
    assert tagger.windows == [0]


# --- a title read in part on the first pages


def _reprinted_title_paper(reprint: str):
    rows = _front(("doc_title", "Field notes:"), ("doc_title", TITLE), ("text", "Ana Ruiz"))
    rows += _front(("text", "Author note and contact details"), page=1)
    rows += _front(("doc_title", reprint), ("abstract", LONG), page=2)
    return rows


def _title_tagger(reprint: str):
    head = [("FURNITURE", "Field notes:", 1.0), ("TITLE", TITLE, 1.0)]
    return _WindowTagger(
        {
            0: [*head, ("GIVEN", "Ana", 1.0), ("FAMILY", "Ruiz", 1.0)],
            1: [("TITLE", reprint, 1.0), ("ABSTRACT", LONG, 1.0)],
        }
    )


def test_a_later_window_extends_a_title_the_first_pages_print_whole():
    reprint = f"Field notes: {TITLE}"
    got = tag_front_matter(_title_tagger(reprint), _reprinted_title_paper(reprint), rules=True)
    assert got.title_keywords.abstract == LONG
    assert got.title_keywords.title == reprint
    got = tag_front_matter(_title_tagger(reprint), _reprinted_title_paper(reprint))
    assert got.title_keywords.title == TITLE


@pytest.mark.parametrize(
    "reprint",
    [f"Delta notes: {TITLE}", "Quiet rivers", f"{TITLE} revisited"],
    ids=["not-printed-first", "shorter", "not-next-to-it"],
)
def test_a_later_title_the_first_pages_do_not_print_whole_is_not_taken(reprint):
    got = tag_front_matter(_title_tagger(reprint), _reprinted_title_paper(reprint), rules=True)
    assert got.title_keywords.title == TITLE


def test_a_later_title_printed_elsewhere_on_the_shared_page_is_not_taken():
    # Page 1, read by both windows, prints a heading holding the title's words.
    heading = f"Comment on {TITLE} by A. Ruiz"
    rows = _front(("doc_title", TITLE), ("text", "Ana Ruiz"), ("text", "Cover note"))
    rows += _front(("paragraph_title", heading), ("text", "Funding and thanks"), page=1)
    rows += _front(("abstract", LONG), page=2)
    tagger = _WindowTagger(
        {
            0: [("TITLE", TITLE, 1.0), ("GIVEN", "Ana", 1.0), ("FAMILY", "Ruiz", 1.0)],
            1: [("TITLE", heading, 1.0), ("ABSTRACT", LONG, 1.0)],
        }
    )
    got = tag_front_matter(tagger, rows, rules=True)
    assert got.title_keywords.abstract == LONG
    assert got.title_keywords.title == TITLE
