from __future__ import annotations

from bibr.front_tagger.runtime import tag_front_matter

from .helpers import label, regions

ABSTRACT = (
    "We studied handedness in 400 families from three countries over two decades. Left "
    "handedness was more common in sons than in daughters, ran in families, and was rarer "
    "among older participants. Twin pairs agreed more often than siblings. It is heritable."
)


class _PhraseTagger:
    """Tags whichever of its phrases the window prints."""

    def __init__(self, *spans: tuple[str, str]):
        self.spans = spans
        self.windows: list[str] = []

    def tag(self, inp):
        self.windows.append(inp.text)
        present = [(f, p) for f, p in self.spans if p in inp.text]
        tags = label(inp, *present) if present else ["O"] * len(inp.words)
        return tags, [1.0] * len(tags)


def _manuscript(abstract_page: int):
    rows = regions(("doc_title", "Handedness in families"), ("text", "Ana Ruiz"), page=0)
    rows += regions(("text", "Author note and contact details"), page=1)
    rows += regions(("text", "Abstract"), ("abstract", ABSTRACT), page=abstract_page)
    return rows


def _tagger():
    return _PhraseTagger(
        ("TITLE", "Handedness in families"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Ruiz"),
        ("ABSTRACT", ABSTRACT),
    )


def test_an_abstract_on_page_three_comes_from_the_next_window():
    tagger = _tagger()
    got = tag_front_matter(tagger, _manuscript(abstract_page=2))
    assert got.title_keywords.title == "Handedness in families"
    assert got.title_keywords.abstract == ABSTRACT
    assert "no_abstract" not in got.problems
    assert [a.family for a in got.byline.authors] == ["Ruiz"]
    assert len(tagger.windows) == 2


def test_a_window_reads_at_most_two_pages_later():
    got = tag_front_matter(_tagger(), _manuscript(abstract_page=4))
    assert got.title_keywords.abstract is None
    assert "no_abstract" in got.problems


def test_an_abstract_on_the_first_pages_reads_one_window():
    tagger = _tagger()
    got = tag_front_matter(tagger, _manuscript(abstract_page=1))
    assert got.title_keywords.abstract == ABSTRACT
    assert len(tagger.windows) == 1


def test_a_blank_page_does_not_end_the_search():
    # The title page, a page with nothing on it, then the abstract.
    rows = regions(("doc_title", "Handedness in families"), ("text", "Ana Ruiz"), page=0)
    rows += regions(("text", "Abstract"), ("abstract", ABSTRACT), page=2)
    got = tag_front_matter(_tagger(), rows)
    assert got.title_keywords.abstract == ABSTRACT
    assert "no_abstract" not in got.problems


class _FailsOnWindow(_PhraseTagger):
    """``_PhraseTagger`` that raises on the given windows (counted from 1), like an out-of-memory GPU."""

    def __init__(self, *spans: tuple[str, str], fail_on: set[int]):
        super().__init__(*spans)
        self.fail_on = fail_on

    def tag(self, inp):
        if len(self.windows) + 1 in self.fail_on:
            self.windows.append(inp.text)
            raise RuntimeError("out of memory")
        return super().tag(inp)


def _failing_tagger(fail_on: set[int]):
    return _FailsOnWindow(
        ("TITLE", "Handedness in families"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Ruiz"),
        ("ABSTRACT", ABSTRACT),
        fail_on=fail_on,
    )


def test_later_windows_that_fail_keep_the_first_windows_answer():
    got = tag_front_matter(_failing_tagger({2, 3}), _manuscript(abstract_page=2))
    assert got.title_keywords.title == "Handedness in families"
    assert [a.family for a in got.byline.authors] == ["Ruiz"]
    assert "no_abstract" in got.problems


def test_a_failed_window_does_not_stop_the_next_one():
    tagger = _failing_tagger({2})
    got = tag_front_matter(tagger, _manuscript(abstract_page=2))
    assert len(tagger.windows) == 3  # the window from page 1 failed, the one from page 2 read it
    assert got.title_keywords.abstract == ABSTRACT
    assert "no_abstract" not in got.problems


def test_a_first_window_that_fails_still_raises():
    import pytest

    with pytest.raises(RuntimeError):
        tag_front_matter(_failing_tagger({1}), _manuscript(abstract_page=2))


def test_the_rules_flag_reaches_the_assembler():
    rows = regions(("doc_title", "Handedness in families"), ("text", "Ana Ruiz Elsevier"))
    tagger = _PhraseTagger(
        ("TITLE", "Handedness in families"),
        ("GIVEN", "Ana"),
        ("FAMILY", "Ruiz"),
        ("FAMILY", "Elsevier"),
    )
    assert [a.family for a in tag_front_matter(tagger, rows).byline.authors] == ["Ruiz", "Elsevier"]
    got = tag_front_matter(tagger, rows, rules=True)
    assert [a.family for a in got.byline.authors] == ["Ruiz"]
