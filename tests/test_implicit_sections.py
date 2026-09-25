"""Tests for implicit section boundary detection."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.clients.prompts import prompt_text
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence
from bibr.schemas import FrontMatterResult, FrontMatterSegment
from bibr.structure.implicit_sections import (
    _apply_boundaries,
    _apply_positional_abstract_fallback,
    _collect_front_matter,
    _detect_via_llm,
    _first_body_text_id,
    _owned_section_types,
    _rebuild_sections_text,
    _trim_bloated_abstract,
    detect_implicit_sections,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_contents(
    sections: list[PaperSection],
    sentences: list[PaperSentence],
    detected_title: str | None = "Paper Title",
) -> PaperContents:
    sections_text: dict[int, str] = {}
    for sec in sections:
        sec_sents = [s for s in sentences if s.section_id == sec.section_id]
        if sec_sents:
            sections_text[sec.section_id] = " ".join(s.text for s in sec_sents)
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text=sections_text,
        detected_title=detected_title,
    )


# ---------------------------------------------------------------------------
# TestFrontMatterSchemas
# ---------------------------------------------------------------------------


class TestFrontMatterSchemas:
    def test_valid_segment(self):
        seg = FrontMatterSegment(first_text_id=1, section_type="abstract")
        assert seg.first_text_id == 1
        assert seg.section_type == "abstract"

    def test_valid_result(self):
        result = FrontMatterResult(
            segments=[
                FrontMatterSegment(first_text_id=1, section_type="abstract"),
                FrontMatterSegment(first_text_id=5, section_type="intro"),
            ]
        )
        assert len(result.segments) == 2

    def test_empty_segments(self):
        result = FrontMatterResult(segments=[])
        assert result.segments == []


async def test_implicit_sections_llm_receives_task_output_cap():
    client = MagicMock()
    client._cap_input.side_effect = lambda text: text
    client.invoke_structured = AsyncMock(return_value=FrontMatterResult(segments=[]))
    front_matter = [
        PaperSentence(text_id=1, text="One.", section_id=1, paragraph_id=1),
        PaperSentence(text_id=2, text="Two.", section_id=1, paragraph_id=1),
    ]

    await _detect_via_llm(client, front_matter, "hash")

    assert client.invoke_structured.await_args.kwargs["max_tokens"] == 4096


async def test_implicit_sections_llm_prompt_has_explicit_native_roles():
    client = MagicMock()
    client._cap_input.side_effect = lambda text: text
    client.invoke_structured = AsyncMock(return_value=FrontMatterResult(segments=[]))
    front_matter = [
        PaperSentence(text_id=1, text="Abstract prose.", section_id=1, paragraph_id=1),
    ]

    await _detect_via_llm(client, front_matter, "hash")

    messages = client.invoke_structured.await_args.args[1]
    parts = messages[0]["content"]
    assert [item["nuextract_role"] for item in parts] == ["instructions", "document"]
    assert "Abstract prose." not in parts[0]["text"]
    assert "Abstract prose." in parts[1]["text"]
    assert "above" not in parts[0]["text"].casefold()
    assert "below" not in parts[0]["text"].casefold()
    assert prompt_text(parts).endswith("---")


# ---------------------------------------------------------------------------
# TestCollectFrontMatter
# ---------------------------------------------------------------------------


class TestCollectFrontMatter:
    def test_collects_unknown_sections_before_methods(self):
        """Sentences in UNKNOWN sections before METHODS are front-matter."""
        title_sec = PaperSection(
            section_id=1,
            header="Paper Title",
            level=0,
            parent_section_id=None,
            section_type=CanonicalSection.UNKNOWN,
        )
        methods_sec = PaperSection(
            section_id=2,
            header="Methods",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.METHODS,
        )
        sentences = [
            PaperSentence(text_id=0, text="Title text.", section_id=1, paragraph_id=0),
            PaperSentence(text_id=1, text="Abstract sentence 1.", section_id=1, paragraph_id=1),
            PaperSentence(text_id=2, text="Abstract sentence 2.", section_id=1, paragraph_id=1),
            PaperSentence(text_id=3, text="Methods paragraph.", section_id=2, paragraph_id=2),
        ]
        contents = _make_contents([title_sec, methods_sec], sentences)
        fm = _collect_front_matter(contents)
        fm_ids = {s.text_id for s in fm}
        # All sentences from title/unknown section before methods body section
        assert fm_ids == {0, 1, 2}
        assert 3 not in fm_ids

    def test_empty_when_all_sections_classified(self):
        """No front-matter when all sections have body types."""
        intro_sec = PaperSection(
            section_id=1,
            header="Introduction",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.INTRODUCTION,
        )
        methods_sec = PaperSection(
            section_id=2,
            header="Methods",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.METHODS,
        )
        sentences = [
            PaperSentence(text_id=0, text="Intro sentence.", section_id=1, paragraph_id=0),
            PaperSentence(text_id=1, text="Methods sentence.", section_id=2, paragraph_id=1),
        ]
        contents = _make_contents([intro_sec, methods_sec], sentences, detected_title=None)
        fm = _collect_front_matter(contents)
        assert fm == []

    def test_empty_when_no_body_section(self):
        """No front-matter when there's no classified body section to anchor against."""
        title_sec = PaperSection(
            section_id=1,
            header="Paper Title",
            level=0,
            parent_section_id=None,
            section_type=CanonicalSection.UNKNOWN,
        )
        sentences = [
            PaperSentence(text_id=0, text="Some text.", section_id=1, paragraph_id=0),
        ]
        contents = _make_contents([title_sec], sentences)
        fm = _collect_front_matter(contents)
        assert fm == []

    def test_collects_title_section_sentences(self):
        """Sentences in the title section (matched by header) are included."""
        title_sec = PaperSection(
            section_id=1,
            header="My Paper Title",
            level=0,
            parent_section_id=None,
            section_type=CanonicalSection.UNKNOWN,
        )
        results_sec = PaperSection(
            section_id=2,
            header="Results",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.RESULTS,
        )
        sentences = [
            PaperSentence(text_id=0, text="Title.", section_id=1, paragraph_id=0),
            PaperSentence(text_id=1, text="Body of abstract.", section_id=1, paragraph_id=1),
            PaperSentence(text_id=2, text="Results text.", section_id=2, paragraph_id=2),
        ]
        contents = _make_contents(
            [title_sec, results_sec], sentences, detected_title="My Paper Title"
        )
        fm = _collect_front_matter(contents)
        assert {s.text_id for s in fm} == {0, 1}

    def test_excludes_sentences_after_first_body(self):
        """Sentences with text_id >= first body sentence are excluded."""
        unknown_sec = PaperSection(
            section_id=1,
            header="Paper Title",
            level=0,
            parent_section_id=None,
            section_type=CanonicalSection.UNKNOWN,
        )
        intro_sec = PaperSection(
            section_id=2,
            header="Introduction",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.INTRODUCTION,
        )
        sentences = [
            PaperSentence(text_id=10, text="Front matter 1.", section_id=1, paragraph_id=0),
            PaperSentence(text_id=11, text="Front matter 2.", section_id=1, paragraph_id=0),
            PaperSentence(text_id=12, text="Intro sentence.", section_id=2, paragraph_id=1),
            PaperSentence(text_id=13, text="Also unknown.", section_id=1, paragraph_id=2),
        ]
        contents = _make_contents([unknown_sec, intro_sec], sentences)
        fm = _collect_front_matter(contents)
        # text_id=13 comes after the body anchor (text_id=12) so it is excluded
        assert {s.text_id for s in fm} == {10, 11}

    def test_selected_front_matter_uses_external_body_anchor(self):
        title = PaperSection(1, "Paper Title", 1, 0, CanonicalSection.UNKNOWN)
        methods = PaperSection(2, "Methods", 1, 0, CanonicalSection.METHODS)
        sentences = [
            PaperSentence(1, "Selected abstract one.", 1, 1, page_number=1),
            PaperSentence(2, "Selected abstract two.", 1, 2, page_number=1),
            PaperSentence(10, "Body anchor outside selected membership.", 2, 3, page_number=2),
        ]
        contents = _make_contents([title, methods], sentences)

        front_matter = _collect_front_matter(contents, frozenset({1, 2}))

        assert _first_body_text_id(contents, after_text_id=2) == 10
        assert [sentence.text_id for sentence in front_matter] == [1, 2]

    def test_body_before_selected_block_cannot_anchor_selected_front_matter(self):
        title = PaperSection(1, "Paper Title", 1, 0, CanonicalSection.UNKNOWN)
        methods = PaperSection(2, "Methods", 1, 0, CanonicalSection.METHODS)
        sentences = [
            PaperSentence(3, "Earlier adjacent body.", 2, 1, page_number=1),
            PaperSentence(5, "Selected front matter one.", 1, 2, page_number=1),
            PaperSentence(6, "Selected front matter two.", 1, 3, page_number=1),
        ]
        contents = _make_contents([title, methods], sentences)

        assert _collect_front_matter(contents, frozenset({5, 6})) == []


# ---------------------------------------------------------------------------
# TestApplyBoundaries
# ---------------------------------------------------------------------------


class TestApplyBoundaries:
    def _make_fm(self, text_ids: list[int], section_id: int = 1) -> list[PaperSentence]:
        return [
            PaperSentence(
                text_id=tid,
                text=f"Sentence {tid}.",
                section_id=section_id,
                paragraph_id=0,
            )
            for tid in text_ids
        ]

    def test_creates_abstract_and_intro_sections(self):
        """Boundaries create new sections and sentences are reassigned."""
        title_sec = PaperSection(
            section_id=1,
            header="Paper Title",
            level=0,
            parent_section_id=None,
            section_type=CanonicalSection.UNKNOWN,
        )
        methods_sec = PaperSection(
            section_id=2,
            header="Methods",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.METHODS,
        )
        sentences = [
            PaperSentence(text_id=0, text="Abstract line 1.", section_id=1, paragraph_id=0),
            PaperSentence(text_id=1, text="Abstract line 2.", section_id=1, paragraph_id=0),
            PaperSentence(text_id=2, text="Intro line.", section_id=1, paragraph_id=1),
            PaperSentence(text_id=3, text="Methods text.", section_id=2, paragraph_id=2),
        ]
        contents = _make_contents([title_sec, methods_sec], sentences)
        front_matter = [s for s in sentences if s.section_id == 1]

        result = FrontMatterResult(
            segments=[
                FrontMatterSegment(first_text_id=0, section_type="abstract"),
                FrontMatterSegment(first_text_id=2, section_type="intro"),
            ]
        )
        applied = _apply_boundaries(contents, result, front_matter)

        assert applied is True

        # Both new sections should be in contents
        section_types = {s.section_type for s in contents.sections}
        assert CanonicalSection.ABSTRACT in section_types
        assert CanonicalSection.INTRODUCTION in section_types

        # Sentences should be reassigned
        abs_sec = next(s for s in contents.sections if s.section_type == CanonicalSection.ABSTRACT)
        intro_sec = next(
            s for s in contents.sections if s.section_type == CanonicalSection.INTRODUCTION
        )
        assert sentences[0].section_id == abs_sec.section_id
        assert sentences[1].section_id == abs_sec.section_id
        assert sentences[2].section_id == intro_sec.section_id
        # Methods sentence unaffected
        assert sentences[3].section_id == 2
        # Implicitly created sections carry the "implicit" classification source.
        assert abs_sec.classification_source == "implicit"
        assert intro_sec.classification_source == "implicit"

    def test_explicit_intro_anchor_suppresses_only_duplicate_intro_segment(self):
        from types import SimpleNamespace

        unknown = PaperSection(1, "Paper Title", 1, 0, CanonicalSection.UNKNOWN)
        introduction = PaperSection(2, "Introduction", 1, 0, CanonicalSection.INTRODUCTION)
        sentences = [
            PaperSentence(1, "Selected abstract one.", 1, 1, page_number=1),
            PaperSentence(2, "Selected abstract two.", 1, 2, page_number=1),
            PaperSentence(3, "Ambiguous selected prose.", 1, 3, page_number=1),
            PaperSentence(10, "Explicit introduction body.", 2, 4, page_number=2),
        ]
        contents = _make_contents([unknown, introduction], sentences)
        contents.front_matter_resolution = SimpleNamespace(
            selected_block_id="selected",
            allowed_text_ids=frozenset({1, 2, 3}),
        )
        result = FrontMatterResult(
            segments=[
                FrontMatterSegment(first_text_id=1, section_type="abstract"),
                FrontMatterSegment(first_text_id=3, section_type="intro"),
            ]
        )

        assert _apply_boundaries(contents, result, sentences[:3])

        abstract = next(
            section
            for section in contents.sections
            if section.section_type == CanonicalSection.ABSTRACT
        )
        assert sentences[0].section_id == abstract.section_id
        assert sentences[1].section_id == abstract.section_id
        assert sentences[2].section_id == unknown.section_id
        assert [
            section
            for section in contents.sections
            if section.section_type == CanonicalSection.INTRODUCTION
        ] == [introduction]

    def test_skips_duplicate_abstract(self):
        """Does not create abstract if one already exists."""
        existing_abs = PaperSection(
            section_id=1,
            header="Abstract",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.ABSTRACT,
        )
        unknown_sec = PaperSection(
            section_id=2,
            header="Paper Title",
            level=0,
            parent_section_id=None,
            section_type=CanonicalSection.UNKNOWN,
        )
        sentences = [
            PaperSentence(text_id=0, text="Abstract text.", section_id=1, paragraph_id=0),
            PaperSentence(text_id=1, text="Front matter.", section_id=2, paragraph_id=1),
        ]
        contents = _make_contents([existing_abs, unknown_sec], sentences)
        front_matter = [sentences[1]]  # only the unknown sentence

        result = FrontMatterResult(
            segments=[FrontMatterSegment(first_text_id=1, section_type="abstract")]
        )
        applied = _apply_boundaries(contents, result, front_matter)

        # No new abstract created; existing one unchanged
        assert applied is False
        abstract_sections = [
            s for s in contents.sections if s.section_type == CanonicalSection.ABSTRACT
        ]
        assert len(abstract_sections) == 1
        assert abstract_sections[0].section_id == 1

    def test_rejects_invalid_text_ids(self):
        """Returns False when text_ids don't match front-matter."""
        title_sec = PaperSection(
            section_id=1,
            header="Paper Title",
            level=0,
            parent_section_id=None,
            section_type=CanonicalSection.UNKNOWN,
        )
        methods_sec = PaperSection(
            section_id=2,
            header="Methods",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.METHODS,
        )
        sentences = [
            PaperSentence(text_id=0, text="Front matter.", section_id=1, paragraph_id=0),
        ]
        contents = _make_contents([title_sec, methods_sec], sentences)
        front_matter = [sentences[0]]

        result = FrontMatterResult(
            segments=[
                FrontMatterSegment(first_text_id=99, section_type="abstract")  # invalid id
            ]
        )
        applied = _apply_boundaries(contents, result, front_matter)

        assert applied is False

    def test_rejects_non_ascending_text_ids(self):
        """Returns False when text_ids are not in ascending order."""
        title_sec = PaperSection(
            section_id=1,
            header="Paper Title",
            level=0,
            parent_section_id=None,
            section_type=CanonicalSection.UNKNOWN,
        )
        methods_sec = PaperSection(
            section_id=2,
            header="Methods",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.METHODS,
        )
        sentences = [
            PaperSentence(text_id=0, text="S0.", section_id=1, paragraph_id=0),
            PaperSentence(text_id=1, text="S1.", section_id=1, paragraph_id=0),
            PaperSentence(text_id=2, text="S2.", section_id=1, paragraph_id=0),
        ]
        contents = _make_contents([title_sec, methods_sec], sentences)
        front_matter = sentences[:]

        # first_text_ids are not ascending (5 then 3)
        result = FrontMatterResult(
            segments=[
                FrontMatterSegment(first_text_id=2, section_type="abstract"),
                FrontMatterSegment(first_text_id=0, section_type="intro"),
            ]
        )
        applied = _apply_boundaries(contents, result, front_matter)

        assert applied is False

    def test_rebuilds_sections_text(self):
        """sections_text is updated for old and new section IDs."""
        title_sec = PaperSection(
            section_id=1,
            header="Paper Title",
            level=0,
            parent_section_id=None,
            section_type=CanonicalSection.UNKNOWN,
        )
        methods_sec = PaperSection(
            section_id=2,
            header="Methods",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.METHODS,
        )
        sentences = [
            PaperSentence(text_id=0, text="Abstract.", section_id=1, paragraph_id=0),
            PaperSentence(text_id=1, text="Methods text.", section_id=2, paragraph_id=1),
        ]
        contents = _make_contents([title_sec, methods_sec], sentences)
        front_matter = [sentences[0]]

        result = FrontMatterResult(
            segments=[FrontMatterSegment(first_text_id=0, section_type="abstract")]
        )
        _apply_boundaries(contents, result, front_matter)

        abs_sec = next(s for s in contents.sections if s.section_type == CanonicalSection.ABSTRACT)

        # New abstract section should have text
        assert abs_sec.section_id in contents.sections_text
        assert "Abstract." in contents.sections_text[abs_sec.section_id]

        # Old section (id=1) lost all its sentences — should be empty or removed
        assert contents.sections_text.get(1, "") == ""

    def test_skips_keywords_and_metadata_segments(self):
        """keywords and metadata segments do not create new sections."""
        title_sec = PaperSection(
            section_id=1,
            header="Paper Title",
            level=0,
            parent_section_id=None,
            section_type=CanonicalSection.UNKNOWN,
        )
        methods_sec = PaperSection(
            section_id=2,
            header="Methods",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.METHODS,
        )
        sentences = [
            PaperSentence(text_id=0, text="Author affiliations.", section_id=1, paragraph_id=0),
            PaperSentence(text_id=1, text="Keywords: foo, bar.", section_id=1, paragraph_id=1),
        ]
        contents = _make_contents([title_sec, methods_sec], sentences)
        front_matter = sentences[:]

        result = FrontMatterResult(
            segments=[
                FrontMatterSegment(first_text_id=0, section_type="metadata"),
                FrontMatterSegment(first_text_id=1, section_type="keywords"),
            ]
        )
        applied = _apply_boundaries(contents, result, front_matter)

        assert applied is False
        # No new sections were created
        assert len(contents.sections) == 2

    def test_empty_segments_returns_false(self):
        """FrontMatterResult with no segments returns False."""
        title_sec = PaperSection(
            section_id=1,
            header="Paper Title",
            level=0,
            parent_section_id=None,
            section_type=CanonicalSection.UNKNOWN,
        )
        sentences = [
            PaperSentence(text_id=0, text="Some text.", section_id=1, paragraph_id=0),
        ]
        contents = _make_contents([title_sec], sentences)
        front_matter = sentences[:]

        result = FrontMatterResult(segments=[])
        applied = _apply_boundaries(contents, result, front_matter)
        assert applied is False

    def test_selected_ownership_rejects_boundary_from_adjacent_record(self):
        from types import SimpleNamespace

        title_sec = PaperSection(1, "Paper Title", 1, 0, CanonicalSection.UNKNOWN)
        body_sec = PaperSection(2, "Methods", 1, 0, CanonicalSection.METHODS)
        selected = PaperSentence(1, "Selected record title.", 1, 1)
        adjacent = PaperSentence(2, "Adjacent record abstract.", 1, 2)
        contents = _make_contents([title_sec, body_sec], [selected, adjacent])
        contents.front_matter_resolution = SimpleNamespace(
            selected_block_id="selected",
            allowed_text_ids=frozenset({1}),
        )
        result = FrontMatterResult(
            segments=[FrontMatterSegment(first_text_id=2, section_type="abstract")]
        )

        applied = _apply_boundaries(contents, result, [selected, adjacent])

        assert applied is False
        assert adjacent.section_id == title_sec.section_id

    def test_selected_segment_never_mutates_unallowed_interior_id(self):
        from types import SimpleNamespace

        title_sec = PaperSection(1, "Paper Title", 1, 0, CanonicalSection.UNKNOWN)
        body_sec = PaperSection(2, "Methods", 1, 0, CanonicalSection.METHODS)
        selected_start = PaperSentence(1, "Selected abstract.", 1, 1)
        adjacent_interior = PaperSentence(2, "Adjacent record evidence.", 1, 2)
        selected_intro = PaperSentence(3, "Selected introduction.", 1, 3)
        contents = _make_contents(
            [title_sec, body_sec], [selected_start, adjacent_interior, selected_intro]
        )
        contents.front_matter_resolution = SimpleNamespace(
            selected_block_id="selected",
            allowed_text_ids=frozenset({1, 3}),
            allowed_section_ids=frozenset({1}),
        )
        result = FrontMatterResult(
            segments=[
                FrontMatterSegment(first_text_id=1, section_type="abstract"),
                FrontMatterSegment(first_text_id=3, section_type="intro"),
            ]
        )

        assert _apply_boundaries(
            contents, result, [selected_start, adjacent_interior, selected_intro]
        )
        assert selected_start.section_id != title_sec.section_id
        assert adjacent_interior.section_id == title_sec.section_id
        assert selected_intro.section_id != title_sec.section_id

    def test_selected_segment_caps_unfiltered_input_at_first_real_body(self):
        from types import SimpleNamespace

        title_sec = PaperSection(1, "Paper Title", 1, 0, CanonicalSection.UNKNOWN)
        methods_sec = PaperSection(2, "Methods", 1, 0, CanonicalSection.METHODS)
        selected_start = PaperSentence(1, "Selected abstract prose.", 1, 1, page_number=1)
        first_body = PaperSentence(3, "First real body row.", 2, 2, page_number=1)
        selected_spill = PaperSentence(
            4, "Selected title-section spill after body.", 1, 3, page_number=1
        )
        later_body = PaperSentence(10, "Later body row.", 2, 4, page_number=2)
        contents = _make_contents(
            [title_sec, methods_sec],
            [selected_start, first_body, selected_spill, later_body],
        )
        contents.front_matter_resolution = SimpleNamespace(
            selected_block_id="selected",
            allowed_text_ids=frozenset({1, 4}),
        )
        result = FrontMatterResult(
            segments=[FrontMatterSegment(first_text_id=1, section_type="abstract")]
        )

        assert _apply_boundaries(contents, result, [selected_start, selected_spill])
        assert selected_start.section_id != title_sec.section_id
        assert selected_spill.section_id == title_sec.section_id


# ---------------------------------------------------------------------------
# TestRebuildSectionsText
# ---------------------------------------------------------------------------


class TestRebuildSectionsText:
    def test_rebuilds_text_from_sentences(self):
        """Joining all sentences for affected section IDs."""
        sec = PaperSection(
            section_id=1,
            header="Abstract",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.ABSTRACT,
        )
        sentences = [
            PaperSentence(text_id=0, text="First.", section_id=1, paragraph_id=0),
            PaperSentence(text_id=1, text="Second.", section_id=1, paragraph_id=0),
        ]
        contents = _make_contents([sec], sentences)
        # Corrupt the sections_text to verify it gets rebuilt
        contents.sections_text[1] = "stale data"

        _rebuild_sections_text(contents, {1})

        assert contents.sections_text[1] == "First. Second."

    def test_removes_entry_when_no_sentences(self):
        """An affected section with no sentences is removed from sections_text."""
        sec = PaperSection(
            section_id=1,
            header="Abstract",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.ABSTRACT,
        )
        sentences = [
            PaperSentence(text_id=0, text="Text in section 2.", section_id=2, paragraph_id=0),
        ]
        contents = _make_contents([sec], sentences)
        contents.sections_text[1] = "old text"

        _rebuild_sections_text(contents, {1})

        assert 1 not in contents.sections_text

    def test_invalidates_cached_dataframes(self):
        """sentences_df and text_df are cleared from __dict__."""
        sec = PaperSection(
            section_id=1,
            header="Intro",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.INTRODUCTION,
        )
        sentences = [
            PaperSentence(text_id=0, text="Hello.", section_id=1, paragraph_id=0),
        ]
        contents = _make_contents([sec], sentences)
        # Force the cached properties into __dict__
        _ = contents.sentences_df
        _ = contents.text_df
        assert "sentences_df" in contents.__dict__
        assert "text_df" in contents.__dict__

        _rebuild_sections_text(contents, {1})

        assert "sentences_df" not in contents.__dict__
        assert "text_df" not in contents.__dict__


# ---------------------------------------------------------------------------
# TestDetectImplicitSections
# ---------------------------------------------------------------------------


class TestDetectImplicitSections:
    """Orchestrator-level tests for detect_implicit_sections() with mocked LLM."""

    @staticmethod
    def _selected_resolution(*candidates):
        from bibr.extract.front_matter import FrontMatterBlock, FrontMatterResolution

        selected_ids = tuple(candidate.candidate_id for candidate in candidates)
        allowed_text_ids = frozenset(
            text_id for candidate in candidates for text_id in candidate.text_ids
        )
        allowed_section_ids = frozenset(
            candidate.section_id for candidate in candidates if candidate.section_id is not None
        )
        return FrontMatterResolution(
            candidates=candidates,
            blocks=(
                FrontMatterBlock(
                    block_id="selected",
                    candidate_ids=selected_ids,
                    title_candidate_ids=(),
                ),
            ),
            selected_block_id="selected",
            selection_method="test",
            reason_flags=(),
            allowed_text_ids=allowed_text_ids,
            allowed_section_ids=allowed_section_ids,
        )

    @staticmethod
    def _candidate(
        candidate_id,
        reading_order,
        section_id,
        text_ids,
        raw_text,
        *,
        roles=frozenset(),
        region_label="text",
    ):
        from bibr.extract.front_matter import FrontMatterCandidate

        return FrontMatterCandidate(
            candidate_id=candidate_id,
            source_kind="paragraph",
            reading_order=reading_order,
            page=1,
            bbox=None,
            region_label=region_label,
            font_size=None,
            font_bold=None,
            section_id=section_id,
            text_ids=tuple(text_ids),
            paragraph_id=reading_order,
            raw_text=raw_text,
            normalized_text=raw_text.casefold(),
            roles=frozenset(roles),
        )

    def _make_standard_contents(self) -> PaperContents:
        """Contents with a title section, a Methods body section, and front-matter sentences."""
        sections = [
            PaperSection(
                section_id=0,
                header="Root",
                level=0,
                parent_section_id=None,
                section_type=CanonicalSection.UNKNOWN,
            ),
            PaperSection(
                section_id=1,
                header="Paper Title",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.UNKNOWN,
            ),
            PaperSection(
                section_id=2,
                header="Method",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.METHODS,
            ),
        ]
        sentences = [
            PaperSentence(
                text_id=1, text="Abstract text.", section_id=1, paragraph_id=1, page_number=1
            ),
            PaperSentence(
                text_id=2, text="More abstract.", section_id=1, paragraph_id=2, page_number=1
            ),
            PaperSentence(
                text_id=3, text="Intro here.", section_id=1, paragraph_id=3, page_number=1
            ),
            PaperSentence(
                text_id=4, text="More intro.", section_id=1, paragraph_id=4, page_number=2
            ),
            PaperSentence(text_id=5, text="Method.", section_id=2, paragraph_id=5, page_number=2),
        ]
        return _make_contents(sections, sentences, detected_title="Paper Title")

    def test_skips_when_both_exist(self):
        """When both ABSTRACT and INTRODUCTION already exist, function is a no-op."""
        sections = [
            PaperSection(
                section_id=0,
                header="Root",
                level=0,
                parent_section_id=None,
                section_type=CanonicalSection.UNKNOWN,
            ),
            PaperSection(
                section_id=1,
                header="Abstract",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.ABSTRACT,
            ),
            PaperSection(
                section_id=2,
                header="Introduction",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.INTRODUCTION,
            ),
            PaperSection(
                section_id=3,
                header="Methods",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.METHODS,
            ),
        ]
        sentences = [
            PaperSentence(
                text_id=1, text="Abstract text.", section_id=1, paragraph_id=1, page_number=1
            ),
            PaperSentence(
                text_id=2, text="Intro text.", section_id=2, paragraph_id=2, page_number=1
            ),
            PaperSentence(
                text_id=3, text="Methods text.", section_id=3, paragraph_id=3, page_number=2
            ),
        ]
        contents = _make_contents(sections, sentences)

        initial_section_count = len(contents.sections)

        with patch(
            "bibr.structure.implicit_sections._detect_via_llm", new_callable=AsyncMock
        ) as mock_detect:
            asyncio.run(detect_implicit_sections(contents))

        # LLM should never be called
        mock_detect.assert_not_called()
        # No new sections created
        assert len(contents.sections) == initial_section_count

    def test_falls_back_when_too_few_sentences(self):
        """When front-matter has < 2 sentences, uses positional fallback (no LLM)."""
        sections = [
            PaperSection(
                section_id=0,
                header="Root",
                level=0,
                parent_section_id=None,
                section_type=CanonicalSection.UNKNOWN,
            ),
            PaperSection(
                section_id=1,
                header="Paper Title",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.UNKNOWN,
            ),
            PaperSection(
                section_id=2,
                header="Methods",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.METHODS,
            ),
        ]
        # Only one page-1 sentence in the unknown/title section — front-matter < 2
        sentences = [
            PaperSentence(
                text_id=1,
                text="Only abstract sentence.",
                section_id=1,
                paragraph_id=1,
                page_number=1,
            ),
            PaperSentence(
                text_id=2, text="Methods text.", section_id=2, paragraph_id=2, page_number=2
            ),
        ]
        contents = _make_contents(sections, sentences, detected_title="Paper Title")

        with patch(
            "bibr.structure.implicit_sections._detect_via_llm", new_callable=AsyncMock
        ) as mock_detect:
            asyncio.run(detect_implicit_sections(contents))

        # LLM must not be called since front-matter is too small
        mock_detect.assert_not_called()

        # Positional fallback should have created an Abstract section
        section_types = {s.section_type for s in contents.sections}
        assert CanonicalSection.ABSTRACT in section_types
        abs_sec = next(s for s in contents.sections if s.section_type == CanonicalSection.ABSTRACT)
        assert abs_sec.classification_source == "positional"

    def test_uses_llm_result(self):
        """LLM result is applied: both Abstract and Introduction sections are created."""
        contents = self._make_standard_contents()

        llm_result = FrontMatterResult(
            segments=[
                FrontMatterSegment(first_text_id=1, section_type="abstract"),
                FrontMatterSegment(first_text_id=3, section_type="intro"),
            ]
        )

        mock_client = MagicMock()
        mock_client.close = AsyncMock()

        with patch("bibr.clients.llm.LLMClient", return_value=mock_client):
            with patch(
                "bibr.structure.implicit_sections._detect_via_llm",
                new_callable=AsyncMock,
                return_value=llm_result,
            ) as mock_detect:
                asyncio.run(detect_implicit_sections(contents))

        # LLM was called once
        mock_detect.assert_called_once()

        # Both implicit sections must have been created
        section_types = {s.section_type for s in contents.sections}
        assert CanonicalSection.ABSTRACT in section_types
        assert CanonicalSection.INTRODUCTION in section_types

        # Sentences 1 and 2 → Abstract; sentences 3 and 4 → Introduction
        abs_sec = next(s for s in contents.sections if s.section_type == CanonicalSection.ABSTRACT)
        intro_sec = next(
            s for s in contents.sections if s.section_type == CanonicalSection.INTRODUCTION
        )

        sent_by_id = {s.text_id: s for s in contents.sentences}
        assert sent_by_id[1].section_id == abs_sec.section_id
        assert sent_by_id[2].section_id == abs_sec.section_id
        assert sent_by_id[3].section_id == intro_sec.section_id
        assert sent_by_id[4].section_id == intro_sec.section_id
        # Methods sentence unaffected
        assert sent_by_id[5].section_id == 2

    def test_falls_back_on_llm_failure(self):
        """When _detect_via_llm returns None (LLM failure), the positional abstract fallback runs.

        The real _detect_via_llm catches all exceptions internally and returns None on
        failure, so we mock it to return None to simulate that failure path.
        """
        contents = self._make_standard_contents()

        mock_client = MagicMock()
        mock_client.close = AsyncMock()

        with patch("bibr.clients.llm.LLMClient", return_value=mock_client):
            with patch(
                "bibr.structure.implicit_sections._detect_via_llm",
                new_callable=AsyncMock,
                return_value=None,
            ):
                asyncio.run(detect_implicit_sections(contents))

        # Positional fallback should have created an Abstract section
        section_types = {s.section_type for s in contents.sections}
        assert CanonicalSection.ABSTRACT in section_types

    def test_adjacent_existing_abstract_cannot_suppress_selected_unknown_rows(self):
        from types import SimpleNamespace

        sections = [
            PaperSection(0, "Root", 0, None, CanonicalSection.UNKNOWN),
            PaperSection(1, "Selected Paper", 1, 0, CanonicalSection.UNKNOWN),
            PaperSection(2, "Adjacent Abstract", 1, 0, CanonicalSection.ABSTRACT),
            PaperSection(3, "Introduction", 1, 0, CanonicalSection.INTRODUCTION),
        ]
        sentences = [
            PaperSentence(1, "Selected abstract one.", 1, 1, page_number=1),
            PaperSentence(2, "Selected abstract two.", 1, 2, page_number=1),
            PaperSentence(9, "Adjacent abstract.", 2, 9, page_number=1),
            PaperSentence(10, "Selected body anchor.", 3, 10, page_number=2),
        ]
        contents = _make_contents(sections, sentences, detected_title="Selected Paper")
        contents.front_matter_resolution = SimpleNamespace(
            selected_block_id="selected",
            allowed_text_ids=frozenset({1, 2}),
            allowed_section_ids=frozenset({1}),
        )
        result = FrontMatterResult(
            segments=[FrontMatterSegment(first_text_id=1, section_type="abstract")]
        )
        client = MagicMock()

        with patch(
            "bibr.structure.implicit_sections._detect_via_llm",
            new_callable=AsyncMock,
            return_value=result,
        ) as detect:
            asyncio.run(detect_implicit_sections(contents, llm_client=client))

        detect.assert_awaited_once()
        selected_abstract = next(
            section
            for section in contents.sections
            if section.section_type == CanonicalSection.ABSTRACT and section.section_id != 2
        )
        assert sentences[0].section_id == selected_abstract.section_id
        assert sentences[1].section_id == selected_abstract.section_id
        assert sentences[2].section_id == 2

    def test_unselected_evidence_never_reaches_implicit_llm(self):
        from types import SimpleNamespace

        sections = [
            PaperSection(1, "Selected Paper", 1, 0, CanonicalSection.UNKNOWN),
            PaperSection(2, "Methods", 1, 0, CanonicalSection.METHODS),
        ]
        sentences = [
            PaperSentence(1, "Selected front matter one.", 1, 1, page_number=1),
            PaperSentence(2, "Selected front matter two.", 1, 2, page_number=1),
            PaperSentence(3, "Adjacent record evidence.", 1, 3, page_number=1),
            PaperSentence(10, "Selected body anchor.", 2, 10, page_number=2),
        ]
        contents = _make_contents(sections, sentences, detected_title="Selected Paper")
        contents.front_matter_resolution = SimpleNamespace(
            selected_block_id="selected",
            allowed_text_ids=frozenset({1, 2}),
            allowed_section_ids=frozenset({1}),
        )
        client = MagicMock()

        with patch(
            "bibr.structure.implicit_sections._detect_via_llm",
            new_callable=AsyncMock,
            return_value=FrontMatterResult(segments=[]),
        ) as detect:
            asyncio.run(detect_implicit_sections(contents, llm_client=client))

        llm_front_matter = detect.await_args.args[1]
        assert {sentence.text_id for sentence in llm_front_matter} == {1, 2}

    def test_preserves_owned_abstract_before_explicit_introduction(self):
        sections = [
            PaperSection(0, "Root", 0, None, CanonicalSection.UNKNOWN),
            PaperSection(1, "The Llama 3 Herd of Models", 1, 0, CanonicalSection.UNKNOWN),
            PaperSection(
                2,
                "Abstract",
                1,
                0,
                CanonicalSection.ABSTRACT,
                classification_source="exact_alias",
            ),
            PaperSection(
                3,
                "1 Introduction",
                1,
                0,
                CanonicalSection.INTRODUCTION,
                classification_source="exact_alias",
            ),
        ]
        sentences = [
            PaperSentence(1, "The Llama 3 Herd of Models", 1, 1, page_number=1),
            PaperSentence(3, "Modern artificial intelligence systems are rapidly", 2, 3, 1),
            PaperSentence(4, "becoming a part of everyday life.", 2, 3, 1),
            PaperSentence(5, "Date: April 18, 2024", 2, 4, page_number=1),
            PaperSentence(6, "Website: https://llama.meta.com", 2, 5, page_number=1),
            PaperSentence(10, "In recent years, large language models have", 3, 6, 1),
        ]
        contents = _make_contents(
            sections,
            sentences,
            detected_title="The Llama 3 Herd of Models",
        )
        abstract = self._candidate(
            "abstract",
            1,
            2,
            (3, 4),
            f"{sentences[1].text} {sentences[2].text}",
            roles={"abstract"},
            region_label="abstract",
        )
        date = self._candidate(
            "date",
            2,
            2,
            (5,),
            sentences[3].text,
            roles={"metadata"},
        )
        website = self._candidate(
            "website",
            3,
            2,
            (6,),
            sentences[4].text,
            roles={"metadata"},
        )
        contents.front_matter_resolution = self._selected_resolution(abstract, date, website)
        client = MagicMock()

        with patch(
            "bibr.structure.implicit_sections._detect_via_llm",
            new_callable=AsyncMock,
        ) as detect:
            asyncio.run(detect_implicit_sections(contents, llm_client=client))

        detect.assert_not_awaited()
        assert sentences[1].section_id == 2
        assert sentences[2].section_id == 2
        assert not any(
            section.section_type == CanonicalSection.INTRODUCTION
            and section.classification_source == "implicit"
            for section in contents.sections
        )

    def test_owned_abstract_does_not_hide_unlabelled_introduction(self):
        sections = [
            PaperSection(0, "Root", 0, None, CanonicalSection.UNKNOWN),
            PaperSection(
                1,
                "Abstract",
                1,
                0,
                CanonicalSection.ABSTRACT,
                classification_source="exact_alias",
            ),
            PaperSection(2, "Methods", 1, 0, CanonicalSection.METHODS),
        ]
        sentences = [
            PaperSentence(1, "This paper presents a new model.", 1, 1, page_number=1),
            PaperSentence(2, "It improves several benchmarks.", 1, 1, page_number=1),
            PaperSentence(3, "Published: July 2026", 1, 2, page_number=1),
            PaperSentence(4, "Language models are widely used.", 1, 3, page_number=1),
            PaperSentence(5, "Prior work established this setting.", 1, 4, page_number=1),
            PaperSentence(10, "We trained the model for ten epochs.", 2, 5, page_number=2),
        ]
        contents = _make_contents(sections, sentences)
        abstract = self._candidate(
            "abstract",
            1,
            1,
            (1, 2),
            f"{sentences[0].text} {sentences[1].text}",
            roles={"abstract"},
            region_label="abstract",
        )
        metadata = self._candidate(
            "metadata",
            2,
            1,
            (3,),
            sentences[2].text,
            roles={"metadata"},
        )
        intro_one = self._candidate("intro-1", 3, 1, (4,), sentences[3].text)
        intro_two = self._candidate("intro-2", 4, 1, (5,), sentences[4].text)
        contents.front_matter_resolution = self._selected_resolution(
            abstract,
            metadata,
            intro_one,
            intro_two,
        )
        result = FrontMatterResult(
            segments=[
                FrontMatterSegment(first_text_id=3, section_type="metadata"),
                FrontMatterSegment(first_text_id=4, section_type="intro"),
            ]
        )
        client = MagicMock()

        with patch(
            "bibr.structure.implicit_sections._detect_via_llm",
            new_callable=AsyncMock,
            return_value=result,
        ) as detect:
            asyncio.run(detect_implicit_sections(contents, llm_client=client))

        llm_front_matter = detect.await_args.args[1]
        assert [sentence.text_id for sentence in llm_front_matter] == [3, 4, 5]
        assert sentences[0].section_id == 1
        assert sentences[1].section_id == 1
        intro_section = next(
            section
            for section in contents.sections
            if section.section_type == CanonicalSection.INTRODUCTION
        )
        assert sentences[3].section_id == intro_section.section_id
        assert sentences[4].section_id == intro_section.section_id

    def test_shared_abstract_section_does_not_create_selected_existing_type(self):
        from bibr.extract.front_matter import (
            FrontMatterBlock,
            FrontMatterCandidate,
            FrontMatterResolution,
        )

        sections = [
            PaperSection(0, "Root", 0, None, CanonicalSection.UNKNOWN),
            PaperSection(1, "Shared Abstract", 1, 0, CanonicalSection.ABSTRACT),
            PaperSection(2, "Methods", 1, 0, CanonicalSection.METHODS),
        ]
        sentences = [
            PaperSentence(1, "Selected Paper", 1, 1, page_number=1),
            PaperSentence(2, "Selected abstract prose.", 1, 2, page_number=1),
            PaperSentence(9, "Adjacent record abstract.", 1, 9, page_number=1),
            PaperSentence(10, "Selected body anchor.", 2, 10, page_number=2),
        ]
        contents = _make_contents(sections, sentences, detected_title="Selected Paper")

        def candidate(candidate_id, order, text_id, text, roles, region_label):
            return FrontMatterCandidate(
                candidate_id=candidate_id,
                source_kind="paragraph",
                reading_order=order,
                page=1,
                bbox=None,
                region_label=region_label,
                font_size=None,
                font_bold=None,
                section_id=1,
                text_ids=(text_id,),
                paragraph_id=order,
                raw_text=text,
                normalized_text=text.casefold(),
                roles=frozenset(roles),
            )

        title_candidate = candidate("title", 1, 1, "Selected Paper", {"title"}, "doc_title")
        selected_prose = candidate("selected-prose", 2, 2, sentences[1].text, set(), "text")
        adjacent_abstract = candidate(
            "adjacent-abstract", 3, 9, sentences[2].text, {"abstract"}, "abstract"
        )
        resolution = FrontMatterResolution(
            candidates=(title_candidate, selected_prose, adjacent_abstract),
            blocks=(
                FrontMatterBlock(
                    block_id="selected",
                    candidate_ids=(title_candidate.candidate_id, selected_prose.candidate_id),
                    title_candidate_ids=(title_candidate.candidate_id,),
                ),
            ),
            selected_block_id="selected",
            selection_method="test",
            reason_flags=(),
            allowed_text_ids=frozenset({1, 2}),
            allowed_section_ids=frozenset({1}),
        )
        contents.front_matter_resolution = resolution
        result = FrontMatterResult(
            segments=[FrontMatterSegment(first_text_id=2, section_type="abstract")]
        )
        client = MagicMock()

        assert CanonicalSection.ABSTRACT not in _owned_section_types(contents, resolution)
        with patch(
            "bibr.structure.implicit_sections._detect_via_llm",
            new_callable=AsyncMock,
            return_value=result,
        ) as detect:
            asyncio.run(detect_implicit_sections(contents, llm_client=client))

        detect.assert_awaited_once()
        selected_abstract = next(
            section
            for section in contents.sections
            if section.section_type == CanonicalSection.ABSTRACT and section.section_id != 1
        )
        assert sentences[1].section_id == selected_abstract.section_id
        assert sentences[2].section_id == 1
        from bibr.structure.implicit_sections import select_abstract_span

        assert select_abstract_span(contents, resolution).text_ids == (2,)


class TestBoundedPositionalAbstract:
    @pytest.mark.parametrize(
        "terminal_type", [CanonicalSection.REFERENCES, CanonicalSection.ENDNOTE]
    )
    def test_back_matter_alone_does_not_authorize_positional_fallback(self, terminal_type):
        title = PaperSection(1, "Paper Title", 1, 0, CanonicalSection.UNKNOWN)
        terminal = PaperSection(2, "Back matter", 1, 0, terminal_type)
        sentences = [
            PaperSentence(1, "Opening prose.", 1, 1, page_number=1),
            PaperSentence(20, "Back matter record.", 2, 2, page_number=5),
        ]
        contents = _make_contents([title, terminal], sentences)

        _apply_positional_abstract_fallback(contents)

        assert all(s.section_type != CanonicalSection.ABSTRACT for s in contents.sections)
        assert sentences[0].section_id == title.section_id

    @pytest.mark.parametrize(
        "body_type",
        [
            CanonicalSection.INTRODUCTION,
            CanonicalSection.METHODS,
            CanonicalSection.RESULTS,
            CanonicalSection.DISCUSSION,
        ],
    )
    def test_level_zero_imrad_does_not_authorize_positional_fallback(self, body_type):
        title = PaperSection(1, "Paper Title", 1, 0, CanonicalSection.UNKNOWN)
        container = PaperSection(2, "Container", 0, 0, body_type)
        sentences = [
            PaperSentence(1, "Opening prose.", 1, 1, page_number=1),
            PaperSentence(20, "Container prose.", 2, 2, page_number=2),
        ]
        contents = _make_contents([title, container], sentences)

        _apply_positional_abstract_fallback(contents)

        assert all(
            section.section_type != CanonicalSection.ABSTRACT for section in contents.sections
        )
        assert sentences[0].section_id == title.section_id

    def test_positional_body_anchor_and_mutation_are_selected_id_scoped(self):
        from bibr.extract.front_matter import (
            FrontMatterBlock,
            FrontMatterCandidate,
            FrontMatterResolution,
        )
        from bibr.structure.implicit_sections import select_abstract_span

        title = PaperSection(1, "Paper Title", 1, 0, CanonicalSection.UNKNOWN)
        methods = PaperSection(2, "Methods", 1, 0, CanonicalSection.METHODS)
        sentences = [
            PaperSentence(3, "Adjacent body anchor.", 2, 1, page_number=1),
            PaperSentence(5, "Selected abstract prose.", 1, 2, page_number=1),
            PaperSentence(6, "Adjacent title-section prose.", 1, 3, page_number=1),
            PaperSentence(10, "Selected body anchor.", 2, 4, page_number=2),
        ]
        contents = _make_contents([title, methods], sentences)
        candidate = FrontMatterCandidate(
            candidate_id="selected-prose",
            source_kind="paragraph",
            reading_order=1,
            page=1,
            bbox=None,
            region_label="text",
            font_size=None,
            font_bold=None,
            section_id=1,
            text_ids=(5,),
            paragraph_id=2,
            raw_text=sentences[1].text,
            normalized_text=sentences[1].text.casefold(),
            roles=frozenset(),
        )
        resolution = FrontMatterResolution(
            candidates=(candidate,),
            blocks=(
                FrontMatterBlock(
                    block_id="selected",
                    candidate_ids=(candidate.candidate_id,),
                    title_candidate_ids=(),
                ),
            ),
            selected_block_id="selected",
            selection_method="test",
            reason_flags=(),
            allowed_text_ids=frozenset({5}),
            allowed_section_ids=frozenset({1}),
        )
        contents.front_matter_resolution = resolution

        _apply_positional_abstract_fallback(contents)

        abstract = next(
            section
            for section in contents.sections
            if section.section_type == CanonicalSection.ABSTRACT
        )
        assert sentences[1].section_id == abstract.section_id
        assert sentences[2].section_id == title.section_id
        assert select_abstract_span(contents, resolution).text_ids == (5,)

    def test_straddled_selected_block_stops_before_first_body_across_all_consumers(self):
        from bibr.extract.front_matter import (
            FrontMatterBlock,
            FrontMatterCandidate,
            FrontMatterResolution,
        )
        from bibr.models import PaperMetadata
        from bibr.pipeline.stages.post_parse import _finalize_abstract_and_keywords
        from bibr.structure.implicit_sections import select_abstract_span

        def make_case():
            title = PaperSection(1, "Paper Title", 1, 0, CanonicalSection.UNKNOWN)
            methods = PaperSection(2, "Methods", 1, 0, CanonicalSection.METHODS)
            sentences = [
                PaperSentence(1, "Selected abstract prose.", 1, 1, page_number=1),
                PaperSentence(3, "First real body row.", 2, 2, page_number=1),
                PaperSentence(4, "Selected title-section spill after body.", 1, 3, page_number=1),
                PaperSentence(10, "Later body row.", 2, 4, page_number=2),
            ]
            contents = _make_contents([title, methods], sentences)

            def candidate(candidate_id, reading_order, text_id, *, roles=frozenset()):
                sentence = next(row for row in sentences if row.text_id == text_id)
                return FrontMatterCandidate(
                    candidate_id=candidate_id,
                    source_kind="paragraph",
                    reading_order=reading_order,
                    page=sentence.page_number,
                    bbox=None,
                    region_label="text",
                    font_size=None,
                    font_bold=None,
                    section_id=sentence.section_id,
                    text_ids=(text_id,),
                    paragraph_id=sentence.paragraph_id,
                    raw_text=sentence.text,
                    normalized_text=sentence.text.casefold(),
                    roles=frozenset(roles),
                )

            first = candidate("selected-first", 1, 1)
            spill = candidate("selected-spill", 2, 4, roles={"abstract"})
            resolution = FrontMatterResolution(
                candidates=(first, spill),
                blocks=(
                    FrontMatterBlock(
                        block_id="selected",
                        candidate_ids=(first.candidate_id, spill.candidate_id),
                        title_candidate_ids=(),
                    ),
                ),
                selected_block_id="selected",
                selection_method="test",
                reason_flags=(),
                allowed_text_ids=frozenset({1, 4}),
                allowed_section_ids=frozenset({1}),
            )
            contents.front_matter_resolution = resolution
            return contents, sentences, title, resolution

        llm_contents, llm_sentences, llm_title, llm_resolution = make_case()
        llm_front_matter = _collect_front_matter(llm_contents, llm_resolution.allowed_text_ids)

        assert _first_body_text_id(llm_contents, after_text_id=1) == 3
        assert [sentence.text_id for sentence in llm_front_matter] == [1]
        assert CanonicalSection.ABSTRACT not in _owned_section_types(llm_contents, llm_resolution)
        assert _apply_boundaries(
            llm_contents,
            FrontMatterResult(
                segments=[FrontMatterSegment(first_text_id=1, section_type="abstract")]
            ),
            llm_front_matter,
        )
        assert llm_sentences[0].section_id != llm_title.section_id
        assert llm_sentences[2].section_id == llm_title.section_id
        assert select_abstract_span(llm_contents, llm_resolution).text_ids == (1,)
        llm_metadata = PaperMetadata(doi="10.1/x", title="Paper Title", abstract="")
        _finalize_abstract_and_keywords(llm_contents, llm_metadata, resolution=llm_resolution)
        assert llm_metadata.abstract == "Selected abstract prose."

        positional_contents, positional_sentences, positional_title, positional_resolution = (
            make_case()
        )
        _apply_positional_abstract_fallback(positional_contents)

        assert positional_sentences[0].section_id != positional_title.section_id
        assert positional_sentences[2].section_id == positional_title.section_id
        assert select_abstract_span(positional_contents, positional_resolution).text_ids == (1,)
        positional_metadata = PaperMetadata(doi="10.1/x", title="Paper Title", abstract="")
        _finalize_abstract_and_keywords(
            positional_contents,
            positional_metadata,
            resolution=positional_resolution,
        )
        assert positional_metadata.abstract == "Selected abstract prose."

    def test_first_real_body_sentence_is_hard_upper_bound(self):
        title = PaperSection(1, "Paper Title", 1, 0, CanonicalSection.UNKNOWN)
        methods = PaperSection(2, "Methods", 1, 0, CanonicalSection.METHODS)
        sentences = [
            PaperSentence(1, "Bounded abstract prose.", 1, 1, page_number=1),
            PaperSentence(3, "The study sampled 50 people.", 2, 2, page_number=1),
            PaperSentence(4, "Title-section spill after body.", 1, 3, page_number=1),
        ]
        contents = _make_contents([title, methods], sentences)

        _apply_positional_abstract_fallback(contents)

        abstract = next(s for s in contents.sections if s.section_type == CanonicalSection.ABSTRACT)
        assert sentences[0].section_id == abstract.section_id
        assert sentences[1].section_id == methods.section_id
        assert sentences[2].section_id == title.section_id

    def test_page_less_native_input_does_not_raise_and_still_selects(self):
        """DOCX/JATS/HTML/ePub parses set page_number=None on every sentence:
        min() over those raised TypeError, and a page equality test would
        exclude every candidate."""
        title = PaperSection(1, "Paper Title", 1, 0, CanonicalSection.UNKNOWN)
        methods = PaperSection(2, "Methods", 1, 0, CanonicalSection.METHODS)
        sentences = [
            PaperSentence(1, "Bounded abstract prose.", 1, 1, page_number=None),
            PaperSentence(3, "The study sampled 50 people.", 2, 2, page_number=None),
            PaperSentence(4, "Title-section spill after body.", 1, 3, page_number=None),
        ]
        contents = _make_contents([title, methods], sentences)

        _apply_positional_abstract_fallback(contents)

        abstract = next(s for s in contents.sections if s.section_type == CanonicalSection.ABSTRACT)
        assert sentences[0].section_id == abstract.section_id
        assert sentences[2].section_id == title.section_id

    def test_bloated_abstract_is_retained_with_source_warning(self, caplog):
        abstract = PaperSection(1, "Abstract", 1, 0, CanonicalSection.ABSTRACT)
        intro = PaperSection(2, "Introduction", 1, 0, CanonicalSection.INTRODUCTION)
        sentences = [
            PaperSentence(1, "A" * 1800, 1, 1, page_number=1),
            PaperSentence(2, "B" * 1800, 1, 2, page_number=2),
            PaperSentence(3, "Body.", 2, 3, page_number=2),
        ]
        contents = _make_contents([abstract, intro], sentences, detected_title=None)

        _trim_bloated_abstract(contents)

        assert [sentence.section_id for sentence in sentences[:2]] == [1, 1]
        assert contents.sections_text[1] == f"{'A' * 1800} {'B' * 1800}"
        assert "section_id=1" in caplog.text
        assert "text_ids=(1, 2)" in caplog.text


# ---------------------------------------------------------------------------
# Implicit sections leave the list in document order for the sanity pass
# ---------------------------------------------------------------------------


def _normalize(contents: PaperContents) -> None:
    """Run post-parse normalization: implicit detection, then both enforce_* passes."""
    from bibr.config import GlobalSettings
    from bibr.pipeline.stages.post_parse import _normalize_section_structure

    class _Client:
        def _cap_input(self, text):
            return text

        async def close(self):
            pass

    asyncio.run(
        _normalize_section_structure(
            contents,
            False,
            _Client(),
            "hash",
            settings=GlobalSettings(),
        )
    )


class TestSectionsStayInDocumentOrder:
    def test_implicit_sections_leave_empty_parent_headings_in_place(self):
        """Headings without prose of their own (the root, the emptied title,
        numbered parents) used to sort behind References; the sanity pass then
        saw Methods/Results after References and reset it to UNKNOWN."""
        sections = [
            PaperSection(0, "", 0, None),
            PaperSection(1, "Paper Title", 1, 0, CanonicalSection.TITLE, 1.0, "title"),
            PaperSection(2, "2 Method", 1, 0, CanonicalSection.METHODS, 1.0, "exact_alias"),
            PaperSection(3, "2.1 Participants", 2, 2, CanonicalSection.METHODS, 1.0, "exact_alias"),
            PaperSection(4, "3 Results", 1, 0, CanonicalSection.RESULTS, 1.0, "exact_alias"),
            PaperSection(5, "3.1 Main effect", 2, 4, CanonicalSection.UNKNOWN),
            PaperSection(6, "4 Discussion", 1, 0, CanonicalSection.DISCUSSION, 1.0, "exact_alias"),
            PaperSection(7, "References", 1, 0, CanonicalSection.REFERENCES, 1.0, "exact_alias"),
            PaperSection(10, "Appendix", 1, 0, CanonicalSection.APPENDIX, 1.0, "exact_alias"),
        ]
        sentences = [
            PaperSentence(1, "Summary sentence one.", 1, 1, page_number=1),
            PaperSentence(2, "Summary sentence two.", 1, 1, page_number=1),
            PaperSentence(3, "Background prose one.", 1, 2, page_number=1),
            PaperSentence(4, "Background prose two.", 1, 2, page_number=1),
            PaperSentence(5, "We recruited 40 adults.", 3, 3, page_number=2),
            PaperSentence(6, "The effect was large.", 5, 4, page_number=3),
            PaperSentence(7, "We discuss it.", 6, 5, page_number=4),
            PaperSentence(8, "Smith, J. (2020).", 7, 6, page_number=5),
            PaperSentence(9, "Extra tables.", 10, 7, page_number=6),
        ]
        contents = _make_contents(sections, sentences)
        result = FrontMatterResult(
            segments=[
                FrontMatterSegment(first_text_id=1, section_type="abstract"),
                FrontMatterSegment(first_text_id=3, section_type="intro"),
            ]
        )

        with patch(
            "bibr.structure.implicit_sections._detect_via_llm",
            new=AsyncMock(return_value=result),
        ):
            _normalize(contents)

        assert [(s.section_id, s.header, s.section_type) for s in contents.sections] == [
            (0, "", CanonicalSection.UNKNOWN),
            (1, "Paper Title", CanonicalSection.TITLE),
            (11, "Abstract", CanonicalSection.ABSTRACT),
            (12, "Introduction", CanonicalSection.INTRODUCTION),
            (2, "2 Method", CanonicalSection.METHODS),
            (3, "2.1 Participants", CanonicalSection.METHODS),
            (4, "3 Results", CanonicalSection.RESULTS),
            (5, "3.1 Main effect", CanonicalSection.UNKNOWN),
            (6, "4 Discussion", CanonicalSection.DISCUSSION),
            (7, "References", CanonicalSection.REFERENCES),
            (10, "Appendix", CanonicalSection.APPENDIX),
        ]
        references = next(s for s in contents.sections if s.section_id == 7)
        assert references.classification_score == 1.0
