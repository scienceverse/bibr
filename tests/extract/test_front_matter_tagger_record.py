"""Fast mode's tagger answer in place of an abstention. Every title and name is invented."""

from __future__ import annotations

from types import SimpleNamespace

from bibr.config import GlobalSettings
from bibr.extract.front_matter import resolve_front_matter
from bibr.paper_contents import PaperContents
from bibr.pipeline.identity import ExpectedIdentity
from bibr.run_mode import fast_settings

from .test_front_matter_record_agreement import _contents, _row, _two_abstracts

FIRST_TITLE = "Loneliness Among Older Immigrants in Urban Neighbourhoods"
SECOND_TITLE = "Social Technology Use and Loneliness During the Pandemic"
MASTHEAD = "JOURNAL OF INVENTED STUDIES"


def _fast():
    return fast_settings(GlobalSettings())


def _tagged(
    contents: PaperContents, title: str | None = FIRST_TITLE, authors: tuple[str, ...] = ("Lin",)
) -> PaperContents:
    contents.front_tagger_result = SimpleNamespace(
        title_keywords=SimpleNamespace(title=title),
        byline=SimpleNamespace(authors=[SimpleNamespace(family=family) for family in authors]),
    )
    return contents


def test_fast_mode_reads_the_record_the_tagger_answered_from():
    contents = _tagged(_two_abstracts(None))

    resolution, issues = resolve_front_matter(contents, target_required=True, settings=_fast())

    assert resolution.selection_method == "tagger_record"
    assert resolution.selected_block_id == "front-matter-block-1"
    assert "tagger_record:front-matter-block-1" in resolution.reason_flags
    assert "multiple_plausible_blocks" not in resolution.reason_flags
    assert issues == ()


def test_fast_mode_selects_the_block_printing_the_tagger_title():
    # A rubric set as a layout title ahead of the article's own title page.
    contents = _contents(
        [
            _row(1, "Reviews of Regional Practice", y=60.0, label="doc_title"),
            _row(2, "Notes on the year in review for this journal.", y=100.0, label="abstract"),
            _row(3, SECOND_TITLE, y=300.0, label="doc_title"),
            _row(4, "Mei Lin, Jordan Smith", y=340.0),
            _row(5, "We surveyed adults about social technology use.", y=380.0, label="abstract"),
        ]
    )
    _tagged(contents, title=SECOND_TITLE)

    resolution, issues = resolve_front_matter(contents, target_required=True, settings=_fast())

    assert resolution.selection_method == "tagger_record"
    assert resolution.selected_block_id == "front-matter-block-2"
    assert issues == ()


def test_two_blocks_with_bylines_keep_abstaining_in_fast_mode():
    contents = _tagged(_two_abstracts("Ana Reader, Bo Writer"))

    resolution, issues = resolve_front_matter(contents, target_required=True, settings=_fast())

    assert resolution.selected_block_id is None
    assert "several_bylines" in resolution.reason_flags
    assert "multiple_plausible_blocks" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_a_title_beside_another_blocks_byline_keeps_abstaining():
    # The tagger missed the first paper's title and read the second's: the one
    # byline and the first abstract belong to the first paper.
    contents = _contents(
        [
            _row(1, FIRST_TITLE, y=60.0, label="paragraph_title"),
            _row(2, "Mei Lin, Jordan Smith", y=100.0),
            _row(3, "We interviewed older immigrants about loneliness.", y=140.0, label="abstract"),
            _row(4, SECOND_TITLE, y=300.0, label="doc_title"),
            _row(5, "We surveyed adults about social technology use.", y=340.0, label="abstract"),
        ]
    )
    _tagged(contents, title=SECOND_TITLE, authors=("Lin", "Smith"))

    resolution, issues = resolve_front_matter(contents, target_required=True, settings=_fast())

    assert resolution.selected_block_id is None
    assert "tagger_title_outside_byline:front-matter-block-2" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_tagged_authors_the_record_does_not_print_keep_abstaining():
    # The tagger took in a second byline that no block shows as one.
    contents = _tagged(_two_abstracts(None), authors=("Lin", "Smith", "Reader", "Writer"))

    resolution, issues = resolve_front_matter(contents, target_required=True, settings=_fast())

    assert resolution.selected_block_id is None
    assert "tagger_authors_outside_record:front-matter-block-1" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_a_title_from_a_block_that_is_no_record_keeps_abstaining():
    # The end of the article before, its summary headed by its own title, ahead
    # of the page's article: neither block prints a byline or a layout title.
    contents = _contents(
        [
            _row(1, FIRST_TITLE, y=60.0, label="paragraph_title"),
            _row(2, "We interviewed older immigrants about loneliness.", y=100.0, label="abstract"),
            _row(3, SECOND_TITLE, y=300.0, label="paragraph_title"),
            _row(4, "We surveyed adults about social technology use.", y=340.0, label="abstract"),
        ]
    )
    _tagged(contents)

    resolution, issues = resolve_front_matter(contents, target_required=True, settings=_fast())

    assert resolution.selected_block_id is None
    assert "tagger_title_outside_record:front-matter-block-1" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_a_tagger_answer_missing_a_title_or_an_author_keeps_abstaining():
    for title, authors in ((None, ("Lin",)), (FIRST_TITLE, ())):
        contents = _tagged(_two_abstracts(None), title=title, authors=authors)

        resolution, issues = resolve_front_matter(contents, target_required=True, settings=_fast())

        assert resolution.selected_block_id is None
        assert "tagger_record_incomplete" in resolution.reason_flags
        assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_no_tagger_answer_keeps_abstaining():
    resolution, issues = resolve_front_matter(
        _two_abstracts(None), target_required=True, settings=_fast()
    )

    assert resolution.selected_block_id is None
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_a_tagger_title_no_block_prints_keeps_abstaining():
    contents = _tagged(_two_abstracts(None), title=MASTHEAD)

    resolution, _ = resolve_front_matter(contents, target_required=True, settings=_fast())

    assert resolution.selected_block_id is None
    assert "tagger_title_outside_blocks" in resolution.reason_flags


def test_the_llm_route_keeps_abstaining_with_a_tagger_answer():
    contents = _tagged(_two_abstracts(None))

    resolution, issues = resolve_front_matter(
        contents, target_required=True, settings=GlobalSettings()
    )

    assert resolution.selected_block_id is None
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_a_supplied_expected_identity_is_never_overridden():
    contents = _tagged(_two_abstracts(None))
    expected = ExpectedIdentity(queue_record_id="other", expected_title="A Different Paper")

    resolution, issues = resolve_front_matter(
        contents, expected_identity=expected, settings=_fast()
    )

    assert resolution.selected_block_id is None
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


def test_an_error_in_the_fallback_abstains_as_before(monkeypatch):
    import bibr.extract.front_matter as front_matter

    def boom(*args, **kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(front_matter, "_select_tagger_record", boom)

    resolution, issues = resolve_front_matter(
        _tagged(_two_abstracts(None)), target_required=True, settings=_fast()
    )

    assert resolution.selected_block_id is None
    assert "tagger_record_error" in resolution.reason_flags
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]
