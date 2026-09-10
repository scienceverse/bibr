"""Regression guard: the abstract must never be joined out of a shared section.

Background.  Under ownership scoping (``resolution is not None`` — PDF + LLM,
i.e. the default) ``select_abstract_span`` is total: every exit returns an
``AbstractSelection``, never None, so the document-wide ABSTRACT-section join in
``_finalize_abstract_and_keywords`` is unreachable and a paper whose span comes
back empty ships ``abstract: ""``.  Two attempts to make that join reachable and
bound it to the selected record were reverted, because on a multi-item front
page the bound does not exist:

``resolve_front_matter`` groups the page into blocks rooted at title-shaped
rows.  When the neighbouring article's title is NOT detected as a heading, the
page yields ONE block, the block's ``allowed_text_ids`` cover both records, and
every row of the shared ABSTRACT section carries the same ``abstract`` role —
``_candidate_roles`` suppresses ``title`` outright for rows in an
ABSTRACT-typed section (``abstract_owned`` at bibr/extract/front_matter.py:791),
and the neighbour's title line is neither byline- nor affiliation-shaped, so no
unsafe role marks it either.  The neighbour's rows are then indistinguishable
from the owned rows in the resolution, and the join ships both articles'
abstracts under this article's DOI.

``ungrounded``/``cross_boundary`` cannot report it: the grounding text is
computed from the same over-admitting ownership set, so the shipped text is a
substring of its own grounding.  On a realistically sized paper the size
heuristic ``non_reference_share_gt_20pct`` does not fire either, so the wrong
abstract ships with NO validation issue at all — strictly worse than the empty
abstract plus VAL_ABSTRACT_MISSING that this code emits today.

These tests use resolutions produced by ``resolve_front_matter`` itself.  A
hand-built ``FrontMatterResolution`` can express an ownership split the real
resolver never produces for these pages, which is how both earlier attempts
tested green while leaking in production.
"""

import pytest

from bibr.extract.front_matter import resolve_front_matter
from bibr.models import PaperMetadata
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
)
from bibr.pipeline.stages.post_parse import _finalize_abstract_and_keywords
from bibr.structure.implicit_sections import _selected_candidates

OWNED_TITLE = "Effects Of Sleep On Recall"
OWNED_ABSTRACT_ROWS = (
    "We tested whether sleep improves recall in adults.",
    "Recall improved by twelve percent after sleep.",
)
NEIGHBOUR_TITLE_ROW = "Kowalski J. A Second Article On The Same Page."
NEIGHBOUR_ABSTRACT_ROW = "This second study examined entirely different outcomes."


def _contents(sentences, sections, *, detected_title=OWNED_TITLE) -> PaperContents:
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={},
        detected_title=detected_title,
    )


def _two_records_neighbour_below() -> PaperContents:
    """One ABSTRACT section holding this record's abstract and the next one's.

    The neighbour's title row is prose, not a heading, so no section boundary
    and no block boundary separates the records.
    """
    sections = [
        PaperSection(0, "Root", 0, None),
        PaperSection(1, OWNED_TITLE, 1, 0, CanonicalSection.TITLE),
        PaperSection(2, "Abstract", 1, 0, CanonicalSection.ABSTRACT),
        PaperSection(3, "Introduction", 1, 0, CanonicalSection.INTRODUCTION),
    ]
    sentences = [
        PaperSentence(1, "Alice Author, Bob Builder", 1, 1, page_number=1),
        PaperSentence(2, "Department of Psychology, University of Somewhere", 1, 2, page_number=1),
        PaperSentence(10, OWNED_ABSTRACT_ROWS[0], 2, 10, page_number=1),
        PaperSentence(11, OWNED_ABSTRACT_ROWS[1], 2, 11, page_number=1),
        PaperSentence(12, NEIGHBOUR_TITLE_ROW, 2, 12, page_number=1),
        PaperSentence(13, NEIGHBOUR_ABSTRACT_ROW, 2, 13, page_number=1),
        PaperSentence(20, "Sleep research has a long history.", 3, 20, page_number=2),
    ]
    return _contents(sentences, sections)


def _two_records_neighbour_above() -> PaperContents:
    """Same page, neighbour printed FIRST — the run cannot simply start early."""
    sections = [
        PaperSection(0, "Root", 0, None),
        PaperSection(1, "Abstract", 1, 0, CanonicalSection.ABSTRACT),
        PaperSection(2, OWNED_TITLE, 1, 0, CanonicalSection.TITLE),
        PaperSection(3, "Introduction", 1, 0, CanonicalSection.INTRODUCTION),
    ]
    sentences = [
        PaperSentence(1, NEIGHBOUR_TITLE_ROW, 1, 1, page_number=1),
        PaperSentence(2, NEIGHBOUR_ABSTRACT_ROW, 1, 2, page_number=1),
        PaperSentence(10, "Alice Author, Bob Builder", 2, 10, page_number=1),
        PaperSentence(
            11, "Department of Psychology, University of Somewhere", 2, 11, page_number=1
        ),
        PaperSentence(12, OWNED_ABSTRACT_ROWS[0], 1, 12, page_number=1),
        PaperSentence(13, OWNED_ABSTRACT_ROWS[1], 1, 13, page_number=1),
        PaperSentence(20, "Sleep research has a long history.", 3, 20, page_number=2),
    ]
    return _contents(sentences, sections)


def _finalize(contents, *, sink=None):
    resolution, _ = resolve_front_matter(contents)
    metadata = PaperMetadata(doi="10.1/owned", title=OWNED_TITLE, abstract="")
    _finalize_abstract_and_keywords(
        contents, metadata, resolution=resolution, validation_issue_sink=sink
    )
    return resolution, metadata


class TestSharedAbstractSectionIsNotJoined:
    @pytest.mark.parametrize(
        "builder",
        [_two_records_neighbour_below, _two_records_neighbour_above],
        ids=["neighbour_below", "neighbour_above"],
    )
    def test_neighbouring_article_text_is_never_shipped(self, builder):
        _, metadata = _finalize(builder())

        assert NEIGHBOUR_TITLE_ROW not in (metadata.abstract or "")
        assert NEIGHBOUR_ABSTRACT_ROW not in (metadata.abstract or "")

    @pytest.mark.parametrize(
        "builder",
        [_two_records_neighbour_below, _two_records_neighbour_above],
        ids=["neighbour_below", "neighbour_above"],
    )
    def test_empty_abstract_is_the_shipped_answer(self, builder):
        # Not "some of the right rows": the recovery is refused entirely, which
        # is what leaves VAL_ABSTRACT_MISSING free to report the failure at
        # export (bibr/export/validation.py::_check_abstract_missing).
        _, metadata = _finalize(builder())

        assert metadata.abstract == ""

    def test_the_resolution_really_does_hand_both_records_to_one_block(self):
        """Why no ownership bound can separate them — asserted, not asserted-at.

        If this ever fails, ``resolve_front_matter`` learned to split the page
        and a scoped recovery becomes worth revisiting.
        """
        contents = _two_records_neighbour_below()
        resolution, _ = resolve_front_matter(contents)

        assert len(resolution.blocks) == 1
        assert resolution.selected_block_id is not None
        # Both records' rows are inside the selected record's own authorization.
        assert {10, 11, 12, 13}.issubset(resolution.allowed_text_ids)

        roles_by_text_id = {
            text_id: candidate.roles
            for candidate in _selected_candidates(resolution)
            for text_id in candidate.text_ids
        }
        # The neighbour's title row carries exactly the roles the owned abstract
        # rows carry, so no role-based seed-stop can fire on it.
        assert roles_by_text_id[12] == roles_by_text_id[10] == frozenset({"abstract"})


class TestOwnedRecordsAreUnaffected:
    def test_single_record_page_still_ships_the_empty_abstract_it_ships_today(self):
        # This pins the revert as TOTAL, and it is the cost of the revert stated
        # plainly rather than the defect: the reverted attempt did recover the
        # right two rows here, and this asserts that recovery is gone. A
        # single-record page whose span comes back empty ships "", and
        # VAL_ABSTRACT_MISSING reports it. Any future recovery has to beat that
        # WITHOUT reopening the cross-record class above; if a scoped recovery
        # lands, this expectation is the one it should change.
        sections = [
            PaperSection(0, "Root", 0, None),
            PaperSection(1, OWNED_TITLE, 1, 0, CanonicalSection.TITLE),
            PaperSection(2, "Abstract", 1, 0, CanonicalSection.ABSTRACT),
            PaperSection(3, "Introduction", 1, 0, CanonicalSection.INTRODUCTION),
        ]
        sentences = [
            PaperSentence(1, "Alice Author, Bob Builder", 1, 1, page_number=1),
            PaperSentence(
                2, "Department of Psychology, University of Somewhere", 1, 2, page_number=1
            ),
            PaperSentence(10, OWNED_ABSTRACT_ROWS[0], 2, 10, page_number=1),
            PaperSentence(11, OWNED_ABSTRACT_ROWS[1], 2, 11, page_number=1),
            PaperSentence(20, "Sleep research has a long history.", 3, 20, page_number=2),
        ]
        _, metadata = _finalize(_contents(sentences, sections))

        assert metadata.abstract == ""

    def test_llm_abstract_is_kept_and_the_neighbour_cannot_displace_it(self):
        contents = _two_records_neighbour_below()
        resolution, _ = resolve_front_matter(contents)
        metadata = PaperMetadata(
            doi="10.1/owned", title=OWNED_TITLE, abstract="LLM extracted abstract."
        )

        _finalize_abstract_and_keywords(contents, metadata, resolution=resolution)

        assert metadata.abstract == "LLM extracted abstract."

    @pytest.mark.parametrize("abstract_text", ["", "LLM extracted abstract."])
    def test_title_and_authors_are_never_touched(self, abstract_text):
        from bibr.models import PaperAuthor

        contents = _two_records_neighbour_below()
        resolution, _ = resolve_front_matter(contents)
        metadata = PaperMetadata(
            doi="10.1/owned",
            title="Owned Title",
            authors=[PaperAuthor(author_id=1, given="Alice", family="Author", affiliation="")],
            abstract=abstract_text,
        )

        _finalize_abstract_and_keywords(contents, metadata, resolution=resolution)

        assert metadata.title == "Owned Title"
        assert [(a.given, a.family) for a in metadata.authors] == [("Alice", "Author")]


class TestUnscopedJoinIsUnchanged:
    def test_native_docx_path_still_joins_the_abstract_section(self):
        # ``resolution is None`` is native/DOCX and --no-llm: one record by
        # construction, so the historic document-wide join is kept as-is. This
        # pins that the revert did not disturb it.
        sections = [
            PaperSection(0, "Root", 0, None),
            PaperSection(2, "Abstract", 1, 0, CanonicalSection.ABSTRACT),
        ]
        sentences = [
            PaperSentence(1, OWNED_ABSTRACT_ROWS[0], 2, 1),
            PaperSentence(2, OWNED_ABSTRACT_ROWS[1], 2, 2),
        ]
        contents = _contents(sentences, sections)
        metadata = PaperMetadata(doi="10.1/owned", title=OWNED_TITLE, abstract="")

        _finalize_abstract_and_keywords(contents, metadata, resolution=None)

        assert metadata.abstract == " ".join(OWNED_ABSTRACT_ROWS)
