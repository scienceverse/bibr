"""Extraction-policy finalization in post-parse.

The abstract fallback, keyword recovery, and commentary abstract-fabrication
guard are extraction POLICY — they decide what the metadata is. They run once
at the end of post-parse (``_finalize_abstract_and_keywords``), mutating
``PaperMetadata`` in place, so the export layer serializes metadata verbatim
instead of re-deriving it (these cases moved here from the export-layer
tests when the policy moved out of ``json_export``).
"""

import pytest

from bibr.models import PaperMetadata
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
)
from bibr.pipeline.stages.post_parse import _finalize_abstract_and_keywords


def _contents(sentences, sections, sections_text=None) -> PaperContents:
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text=sections_text or {},
    )


def _abstract_contents(abstract_sentences: list[str]) -> PaperContents:
    sentences = [
        PaperSentence(text_id=i, text=t, section_id=2, paragraph_id=1)
        for i, t in enumerate(abstract_sentences, start=1)
    ]
    sections = [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        PaperSection(
            section_id=2,
            header="Abstract",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.ABSTRACT,
        ),
    ]
    return _contents(sentences, sections)


class TestAbstractFinalize:
    def test_explicit_null_is_not_overridden_by_synthetic_abstract(self):
        contents = _abstract_contents(["Opening editorial body."])
        contents.sections[1].header_is_synthetic = True
        meta = PaperMetadata(doi="", title="Editorial")
        meta._abstract_explicitly_absent = True
        _finalize_abstract_and_keywords(contents, meta)
        assert meta.abstract == ""
        assert "_abstract_explicitly_absent" not in meta.model_dump()

    def test_explicit_null_can_recover_a_printed_abstract(self):
        meta = PaperMetadata(doi="", title="Study")
        meta._abstract_explicitly_absent = True
        _finalize_abstract_and_keywords(_abstract_contents(["Printed summary."]), meta)
        assert meta.abstract == "Printed summary."

    def test_llm_abstract_kept_and_stripped(self):
        meta = PaperMetadata(doi="10.1/x", title="T", abstract="  The clean LLM abstract. ")
        _finalize_abstract_and_keywords(_abstract_contents(["Section noise."]), meta)
        assert meta.abstract == "The clean LLM abstract."

    def test_section_concat_fallback_when_abstract_empty(self):
        meta = PaperMetadata(doi="10.1/x", title="T", abstract="")
        _finalize_abstract_and_keywords(
            _abstract_contents(["Sentence one.", "Sentence two."]), meta
        )
        assert meta.abstract == "Sentence one. Sentence two."

    def test_section_concat_fallback_when_abstract_blank(self):
        meta = PaperMetadata(doi="10.1/x", title="T", abstract="   ")
        _finalize_abstract_and_keywords(_abstract_contents(["Body sentence."]), meta)
        assert meta.abstract == "Body sentence."

    def test_unknown_section_headed_abstract_counts(self):
        # An UNKNOWN-typed section whose header normalizes to "abstract" is
        # treated as the abstract section.
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=3,
                header="ABSTRACT",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.UNKNOWN,
            ),
        ]
        sentences = [PaperSentence(text_id=1, text="From unknown.", section_id=3, paragraph_id=1)]
        meta = PaperMetadata(doi="10.1/x", title="T", abstract="")
        _finalize_abstract_and_keywords(_contents(sentences, sections), meta)
        assert meta.abstract == "From unknown."

    def test_display_formula_sentences_excluded(self):
        contents = _abstract_contents(["Real sentence."])
        contents.sentences.append(
            PaperSentence(
                text_id=9, text="x = y", section_id=2, paragraph_id=1, is_display_formula=True
            )
        )
        meta = PaperMetadata(doi="10.1/x", title="T", abstract="")
        _finalize_abstract_and_keywords(contents, meta)
        assert meta.abstract == "Real sentence."

    def test_no_abstract_anywhere_stays_empty(self):
        meta = PaperMetadata(doi="10.1/x", title="T", abstract="")
        _finalize_abstract_and_keywords(_abstract_contents([]), meta)
        assert meta.abstract == ""


class TestKeywordFinalize:
    def _kw_contents(self, kw_text: str) -> PaperContents:
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=4,
                header="Keywords",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.KEYWORDS,
            ),
        ]
        sentences = [PaperSentence(text_id=1, text=kw_text, section_id=4, paragraph_id=1)]
        return _contents(sentences, sections)

    def test_recovers_keywords_from_section(self):
        meta = PaperMetadata(doi="10.1/x", title="T")
        _finalize_abstract_and_keywords(self._kw_contents("memory, aging, cognition."), meta)
        assert meta.keywords == ["memory", "aging", "cognition"]

    def test_llm_keywords_win(self):
        meta = PaperMetadata(doi="10.1/x", title="T", keywords=["from-llm"])
        _finalize_abstract_and_keywords(self._kw_contents("memory, aging"), meta)
        assert meta.keywords == ["from-llm"]

    def test_sentence_like_candidates_rejected(self):
        meta = PaperMetadata(doi="10.1/x", title="T")
        _finalize_abstract_and_keywords(
            self._kw_contents("This is not a keyword list. It has sentences, and periods."),
            meta,
        )
        assert meta.keywords == []


class TestCommentaryAbstractGuard:
    """Residual #1: the layout model mislabels a commentary's opening body as
    ABSTRACT; both the LLM string and the section fallback can carry it. The
    guard suppresses it at finalize time — using FINAL keywords, so a genuine
    commentary with a keywords block is preserved."""

    LONG = "In our current efforts to understand brain activity here. " * 50

    def _contents_with_abstract_body(self, body: str, kw_section: bool = False) -> PaperContents:
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Abstract",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.ABSTRACT,
            ),
        ]
        sentences = [PaperSentence(text_id=1, text=body, section_id=1, paragraph_id=1)]
        if kw_section:
            sections.append(
                PaperSection(
                    section_id=2,
                    header="Keywords",
                    level=1,
                    parent_section_id=0,
                    section_type=CanonicalSection.KEYWORDS,
                )
            )
            sentences.append(
                PaperSentence(text_id=2, text="memory, aging", section_id=2, paragraph_id=2)
            )
        return _contents(sentences, sections, sections_text={1: body})

    def test_long_section_fallback_is_preserved(self):
        meta = PaperMetadata(doi="10.1/x", title="T", abstract="", paper_type="commentary")
        _finalize_abstract_and_keywords(self._contents_with_abstract_body(self.LONG), meta)
        assert meta.abstract == self.LONG.strip()

    def test_long_metadata_abstract_is_preserved(self):
        meta = PaperMetadata(doi="10.1/x", title="T", abstract=self.LONG, paper_type="commentary")
        _finalize_abstract_and_keywords(self._contents_with_abstract_body(self.LONG), meta)
        assert meta.abstract == self.LONG.strip()

    def test_genuine_empirical_abstract_recovered(self):
        meta = PaperMetadata(doi="10.1/x", title="T", abstract="", paper_type="empirical")
        _finalize_abstract_and_keywords(self._contents_with_abstract_body(self.LONG), meta)
        assert meta.abstract and len(meta.abstract) > 1000

    def test_commentary_with_keyword_section_recovered(self):
        # The recovered keywords (finalized BEFORE the guard runs) mark the
        # abstract as genuine.
        meta = PaperMetadata(doi="10.1/x", title="T", abstract="", paper_type="commentary")
        _finalize_abstract_and_keywords(
            self._contents_with_abstract_body(self.LONG, kw_section=True), meta
        )
        assert meta.abstract


class TestPostParseWiring:
    async def test_no_llm_post_parse_finalizes_abstract(self):
        """End-to-end: a no-LLM post_parse run fills metadata.abstract from the
        ABSTRACT section, so export reads metadata verbatim."""
        from bibr.pipeline.stages.post_parse import post_parse

        contents = _abstract_contents(["Sentence one.", "Sentence two."])
        paper = await post_parse(contents, "f.pdf", "hash123", no_llm=True)
        assert paper.metadata.abstract == "Sentence one. Sentence two."

    async def test_preparsed_abstract_bypasses_selected_grounding_warning(self):
        from bibr.pipeline.stages.post_parse import post_parse

        contents = _abstract_contents(["Layout abstract text."])
        contents.preparsed_metadata = PaperMetadata(
            doi="10.1/native",
            title="Native title",
            abstract="Authoritative native abstract.",
        )

        paper = await post_parse(contents, "native.xml", "hash123", no_llm=True)

        assert paper.metadata.abstract == "Authoritative native abstract."
        assert not any(issue.code == "VAL_ABSTRACT_SUSPECT" for issue in paper.validation_issues)


def _selected_resolution(*candidates, allowed_text_ids):
    from bibr.extract.front_matter import FrontMatterBlock, FrontMatterResolution

    block = FrontMatterBlock(
        block_id="selected",
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        title_candidate_ids=tuple(
            candidate.candidate_id for candidate in candidates if "title" in candidate.roles
        ),
    )
    return FrontMatterResolution(
        candidates=tuple(candidates),
        blocks=(block,),
        selected_block_id="selected",
        selection_method="test",
        reason_flags=(),
        allowed_text_ids=frozenset(allowed_text_ids),
        allowed_section_ids=frozenset(
            candidate.section_id for candidate in candidates if candidate.section_id is not None
        ),
    )


def _candidate(
    candidate_id,
    reading_order,
    text_ids,
    raw_text,
    *,
    roles,
    page=1,
    section_id=1,
    region_label=None,
    source_kind="paragraph",
):
    from bibr.extract.front_matter import FrontMatterCandidate

    return FrontMatterCandidate(
        candidate_id=candidate_id,
        source_kind=source_kind,
        reading_order=reading_order,
        page=page,
        bbox=None,
        region_label=region_label or ("abstract" if "abstract" in roles else "text"),
        font_size=None,
        font_bold=None,
        section_id=section_id,
        text_ids=tuple(text_ids),
        paragraph_id=reading_order if source_kind == "paragraph" else None,
        raw_text=raw_text,
        normalized_text=" ".join(raw_text.casefold().split()),
        roles=frozenset(roles),
    )


class TestSelectedAbstractFinalize:
    @staticmethod
    def _bounded_contents():
        sections = [
            PaperSection(0, "Root", 0, None),
            PaperSection(1, "Paper Title", 1, 0, CanonicalSection.UNKNOWN),
            PaperSection(2, "Introduction", 1, 0, CanonicalSection.INTRODUCTION),
        ]
        sentences = [
            PaperSentence(1, "Paper Title", 1, 1, page_number=1),
            PaperSentence(2, "Selected abstract sentence one.", 1, 2, page_number=1),
            PaperSentence(3, "Selected continuation on page two.", 1, 3, page_number=2),
            PaperSentence(4, "Body sentence outside the selected block.", 2, 4, page_number=2),
            PaperSentence(9, "Other record abstract.", 1, 9, page_number=1),
        ]
        return _contents(sentences, sections)

    def test_explicit_abstract_accepts_contiguous_page_two_continuation(self):
        from bibr.structure.implicit_sections import select_abstract_span

        contents = self._bounded_contents()
        explicit = _candidate(
            "abstract-1",
            1,
            (2,),
            "Selected abstract sentence one.",
            roles={"abstract"},
        )
        continuation = _candidate(
            "abstract-2",
            2,
            (3,),
            "Selected continuation on page two.",
            roles=set(),
            page=2,
        )
        resolution = _selected_resolution(
            explicit,
            continuation,
            allowed_text_ids={1, 2, 3},
        )

        selected = select_abstract_span(contents, resolution)

        assert selected.text_ids == (2, 3)
        assert selected.text == (
            "Selected abstract sentence one. Selected continuation on page two."
        )

    def test_fallback_never_crosses_selected_block_boundary(self):
        contents = self._bounded_contents()
        explicit = _candidate(
            "abstract-1",
            1,
            (2,),
            "Selected abstract sentence one.",
            roles={"abstract"},
        )
        resolution = _selected_resolution(explicit, allowed_text_ids={1, 2})
        meta = PaperMetadata(doi="10.1/x", title="T", abstract="")

        _finalize_abstract_and_keywords(contents, meta, resolution=resolution)

        assert meta.abstract == "Selected abstract sentence one."
        assert "Other record abstract" not in meta.abstract

    def test_ungrounded_llm_abstract_is_preserved_and_warned(self):
        contents = self._bounded_contents()
        explicit = _candidate(
            "abstract-1",
            1,
            (2,),
            "Selected abstract sentence one.",
            roles={"abstract"},
        )
        resolution = _selected_resolution(explicit, allowed_text_ids={1, 2})
        meta = PaperMetadata(doi="10.1/x", title="T", abstract="Invented abstract text.")
        issues = []

        _finalize_abstract_and_keywords(
            contents,
            meta,
            resolution=resolution,
            validation_issue_sink=issues,
        )

        assert meta.abstract == "Invented abstract text."
        issue = next(issue for issue in issues if issue.code == "VAL_ABSTRACT_SUSPECT")
        assert issue.blocking is False
        assert "ungrounded" in issue.message

    def test_cross_boundary_llm_abstract_is_preserved_and_warned(self):
        contents = self._bounded_contents()
        explicit = _candidate(
            "abstract-1",
            1,
            (2,),
            "Selected abstract sentence one.",
            roles={"abstract"},
        )
        resolution = _selected_resolution(explicit, allowed_text_ids={1, 2})
        combined = "Selected abstract sentence one. Body sentence outside the selected block."
        meta = PaperMetadata(doi="10.1/x", title="T", abstract=combined)
        issues = []

        _finalize_abstract_and_keywords(
            contents,
            meta,
            resolution=resolution,
            validation_issue_sink=issues,
        )

        assert meta.abstract == combined
        issue = next(issue for issue in issues if issue.code == "VAL_ABSTRACT_SUSPECT")
        assert "cross_boundary" in issue.message

    def test_source_warning_canonically_combines_all_four_reasons_once(self):
        contents = self._bounded_contents()
        explicit = _candidate(
            "abstract-1",
            1,
            (2,),
            "Selected abstract sentence one.",
            roles={"abstract"},
        )
        resolution = _selected_resolution(explicit, allowed_text_ids={1, 2})
        abstract = (
            "Invented abstract text. " + ("A" * 2500) + " Body sentence outside the selected block."
        )
        meta = PaperMetadata(doi="10.1/x", title="T", abstract=abstract)
        issues = []

        _finalize_abstract_and_keywords(
            contents,
            meta,
            resolution=resolution,
            validation_issue_sink=issues,
        )

        suspect = [issue for issue in issues if issue.code == "VAL_ABSTRACT_SUSPECT"]
        assert len(suspect) == 1
        assert suspect[0].message == (
            "abstract suspicion: ungrounded, cross_boundary, length_gt_2500, "
            "non_reference_share_gt_20pct"
        )
        assert suspect[0].count == 1
        assert len(suspect[0].evidence_ids) <= 20
        assert meta.abstract == abstract

    def test_grounded_long_commentary_abstract_is_not_trimmed_in_shadow_mode(self):
        long_text = "Grounded source abstract. " * 130
        contents = _abstract_contents([long_text])
        explicit = _candidate(
            "abstract-1",
            1,
            (1,),
            long_text,
            roles={"abstract"},
            section_id=2,
        )
        resolution = _selected_resolution(explicit, allowed_text_ids={1})
        meta = PaperMetadata(
            doi="10.1/x",
            title="T",
            abstract=long_text,
            paper_type="commentary",
        )

        _finalize_abstract_and_keywords(contents, meta, resolution=resolution)

        assert meta.abstract == long_text.strip()

    def test_composite_role_without_explicit_boundary_fails_closed(self):
        from bibr.structure.implicit_sections import select_abstract_span

        contents = self._bounded_contents()
        composite = _candidate(
            "composite",
            1,
            (1, 2),
            "Paper Title Alice Author Selected abstract sentence one.",
            roles={"title", "byline", "abstract"},
            region_label="abstract",
        )
        resolution = _selected_resolution(composite, allowed_text_ids={1, 2})
        contents.sections[1].section_type = CanonicalSection.ABSTRACT

        assert select_abstract_span(contents, resolution).text_ids == ()

    def test_composite_abstract_boundary_stops_without_resuming_at_later_candidate(self):
        from bibr.structure.implicit_sections import select_abstract_span

        contents = self._bounded_contents()
        composite = _candidate(
            "composite",
            1,
            (1,),
            "Paper Title and abstract",
            roles={"title", "abstract"},
            region_label="abstract",
        )
        later = _candidate(
            "later-abstract",
            2,
            (2,),
            "Selected abstract sentence one.",
            roles={"abstract"},
        )
        resolution = _selected_resolution(composite, later, allowed_text_ids={1, 2})

        assert select_abstract_span(contents, resolution).text_ids == ()

    def test_overloaded_abstract_section_uses_candidate_ownership_not_section_bulk(self):
        from bibr.structure.implicit_sections import select_abstract_span

        contents = self._bounded_contents()
        contents.sections[1].section_type = CanonicalSection.ABSTRACT
        contents.sentences[3] = PaperSentence(
            4, "Page-three spill from the overloaded section.", 1, 4, page_number=3
        )
        candidates = (
            _candidate("title", 1, (1,), "Paper Title", roles={"title"}),
            _candidate("byline", 2, (9,), "Alice Author", roles={"byline"}),
            _candidate("abstract", 3, (2,), "Selected abstract sentence one.", roles={"abstract"}),
            _candidate(
                "spill",
                4,
                (4,),
                "Page-three spill from the overloaded section.",
                roles={"abstract"},
                page=3,
            ),
        )
        resolution = _selected_resolution(*candidates, allowed_text_ids={1, 2, 4, 9})

        assert select_abstract_span(contents, resolution).text_ids == (2,)

    def test_empty_abstract_heading_opens_only_immediately_following_safe_paragraph(self):
        from bibr.structure.implicit_sections import select_abstract_span

        contents = self._bounded_contents()
        heading = _candidate(
            "abstract-heading",
            1,
            (),
            "Abstract",
            roles={"abstract", "heading"},
            region_label="paragraph_title",
            source_kind="heading",
        )
        paragraph = _candidate(
            "abstract-paragraph",
            2,
            (2,),
            "Selected abstract sentence one.",
            roles=set(),
        )
        stop = _candidate("title-stop", 3, (3,), "Another Paper", roles={"title"}, page=2)
        after_stop = _candidate(
            "after-stop",
            4,
            (9,),
            "Other record abstract.",
            roles={"abstract"},
        )
        resolution = _selected_resolution(
            heading,
            paragraph,
            stop,
            after_stop,
            allowed_text_ids={2, 3, 9},
        )

        assert select_abstract_span(contents, resolution).text_ids == (2,)

    @pytest.mark.parametrize(
        "stop_candidate",
        [
            _candidate(
                "intro-heading",
                2,
                (),
                "Introduction",
                roles={"heading"},
                region_label="paragraph_title",
                source_kind="heading",
            ),
            _candidate("title", 2, (3,), "Another Paper", roles={"title"}, page=2),
            _candidate("byline", 2, (3,), "Alice Author", roles={"byline"}, page=2),
            _candidate(
                "affiliation",
                2,
                (3,),
                "Department of Physics",
                roles={"affiliation"},
                page=2,
            ),
            _candidate("doi", 2, (3,), "doi:10.1/x", roles={"doi"}, page=2),
            _candidate(
                "correspondence",
                2,
                (3,),
                "Correspondence: author@example.org",
                roles=set(),
                page=2,
            ),
            _candidate(
                "correspondence-to",
                2,
                (3,),
                "Correspondence to Alice Author, University of Example",
                roles=set(),
                page=2,
            ),
            _candidate(
                "address-correspondence-to",
                2,
                (3,),
                "Address correspondence to Alice Author, University of Example",
                roles=set(),
                page=2,
            ),
            _candidate(
                "correspondence-label",
                2,
                (3,),
                "author@example.org",
                roles=set(),
                region_label="correspondence",
                page=2,
            ),
            _candidate("keywords", 2, (3,), "Keywords: one, two", roles=set(), page=2),
            _candidate(
                "metadata",
                2,
                (3,),
                "Received 1 January 2025",
                roles=set(),
                region_label="metadata",
                page=2,
            ),
            _candidate(
                "footer",
                2,
                (3,),
                "Copyright 2025",
                roles=set(),
                region_label="footer",
                page=2,
            ),
            _candidate(
                "metadata-role",
                2,
                (3,),
                "Record metadata",
                roles={"metadata"},
                page=2,
            ),
            _candidate(
                "correspondence-role",
                2,
                (3,),
                "author@example.org",
                roles={"correspondence"},
                page=2,
            ),
            _candidate(
                "structural-role",
                2,
                (3,),
                "Running page furniture",
                roles={"structural"},
                page=2,
            ),
            _candidate(
                "keyword-role",
                2,
                (3,),
                "one, two",
                roles={"keywords"},
                page=2,
            ),
        ],
        ids=lambda candidate: candidate.candidate_id,
    )
    def test_open_abstract_span_stops_permanently_at_unsafe_candidate(self, stop_candidate):
        from bibr.structure.implicit_sections import select_abstract_span

        contents = self._bounded_contents()
        start = _candidate(
            "abstract",
            1,
            (2,),
            "Selected abstract sentence one.",
            roles={"abstract"},
        )
        after_stop = _candidate(
            "after-stop",
            3,
            (9,),
            "Other record abstract.",
            roles={"abstract"},
            page=2,
        )
        resolution = _selected_resolution(
            start,
            stop_candidate,
            after_stop,
            allowed_text_ids={2, 3, 9},
        )

        assert select_abstract_span(contents, resolution).text_ids == (2,)

    def test_open_span_stops_at_unselected_candidate_and_never_resumes(self):
        from bibr.structure.implicit_sections import select_abstract_span

        contents = self._bounded_contents()
        start = _candidate(
            "abstract", 1, (2,), "Selected abstract sentence one.", roles={"abstract"}
        )
        unselected = _candidate("unselected", 2, (3,), "Unselected boundary.", roles=set())
        after_stop = _candidate(
            "after-stop", 3, (9,), "Other record abstract.", roles={"abstract"}, page=2
        )
        resolution = _selected_resolution(start, unselected, after_stop, allowed_text_ids={2, 9})

        assert select_abstract_span(contents, resolution).text_ids == (2,)

    @pytest.mark.parametrize(
        ("candidate_text", "region_label"),
        [
            pytest.param("Correspondence:", None, id="correspondence-colon"),
            pytest.param("Correspondence to Alice", None, id="correspondence-to"),
            pytest.param("Correspondence to: Alice", None, id="correspondence-to-colon"),
            pytest.param("Address correspondence to Alice", None, id="address-correspondence-to"),
            pytest.param(
                "Address correspondence to: Alice", None, id="address-correspondence-to-colon"
            ),
            pytest.param(
                "Address for correspondence: Alice", None, id="address-for-correspondence"
            ),
            pytest.param("Corresponding author: Alice", None, id="corresponding-author"),
            pytest.param("Alice Author", "correspondence", id="correspondence-region"),
        ],
    )
    def test_generated_abstract_section_does_not_authorize_correspondence_candidate(
        self, candidate_text, region_label
    ):
        from bibr.structure.implicit_sections import select_abstract_span

        contents = self._bounded_contents()
        contents.sections[1].section_type = CanonicalSection.ABSTRACT
        contents.sections[1].classification_source = "positional"
        contents.sentences[1].text = candidate_text
        correspondence = _candidate(
            "correspondence",
            1,
            (2,),
            contents.sentences[1].text,
            roles=set(),
            region_label=region_label,
        )
        later_abstract = _candidate(
            "later-abstract",
            2,
            (3,),
            contents.sentences[2].text,
            roles={"abstract"},
            page=2,
        )
        resolution = _selected_resolution(correspondence, later_abstract, allowed_text_ids={2, 3})

        assert select_abstract_span(contents, resolution).text_ids == ()

    @pytest.mark.parametrize(
        "candidate_text",
        [
            "The correspondence to Alice was archived for analysis.",
            "Corresponding author responses were analyzed separately.",
            "Correspondence for the measured variables was assessed longitudinally.",
            "Correspondence to prior literature was assessed longitudinally.",
            "Address correspondence to methodological limitations in the discussion.",
        ],
    )
    def test_non_heading_body_mention_of_correspondence_remains_abstract_evidence(
        self, candidate_text
    ):
        from bibr.structure.implicit_sections import select_abstract_span

        contents = self._bounded_contents()
        contents.sentences[1].text = candidate_text
        abstract = _candidate(
            "abstract",
            1,
            (2,),
            contents.sentences[1].text,
            roles={"abstract"},
        )
        resolution = _selected_resolution(abstract, allowed_text_ids={2})

        assert select_abstract_span(contents, resolution).text_ids == (2,)

    def test_open_span_requires_contiguous_candidate_reading_order(self):
        from bibr.structure.implicit_sections import select_abstract_span

        contents = self._bounded_contents()
        start = _candidate(
            "abstract", 1, (2,), "Selected abstract sentence one.", roles={"abstract"}
        )
        after_gap = _candidate(
            "after-gap",
            3,
            (3,),
            "Selected continuation on page two.",
            roles=set(),
            page=2,
        )
        resolution = _selected_resolution(start, after_gap, allowed_text_ids={2, 3})

        assert select_abstract_span(contents, resolution).text_ids == (2,)

    def test_page_two_open_cannot_start_or_continue_to_page_three(self):
        from bibr.structure.implicit_sections import select_abstract_span

        contents = self._bounded_contents()
        page_two_start = _candidate(
            "page-two-abstract",
            1,
            (3,),
            "Selected continuation on page two.",
            roles={"abstract"},
            page=2,
        )
        page_three = _candidate(
            "page-three",
            2,
            (9,),
            "Other record abstract.",
            roles={"abstract"},
            page=3,
        )
        resolution = _selected_resolution(page_two_start, page_three, allowed_text_ids={3, 9})

        assert select_abstract_span(contents, resolution).text_ids == ()

    def test_page_one_span_allows_page_two_but_stops_before_page_three(self):
        from bibr.structure.implicit_sections import select_abstract_span

        contents = self._bounded_contents()
        candidates = (
            _candidate("page-one", 1, (2,), "Selected abstract sentence one.", roles={"abstract"}),
            _candidate(
                "page-two",
                2,
                (3,),
                "Selected continuation on page two.",
                roles=set(),
                page=2,
            ),
            _candidate(
                "page-three",
                3,
                (9,),
                "Other record abstract.",
                roles={"abstract"},
                page=3,
            ),
        )
        resolution = _selected_resolution(*candidates, allowed_text_ids={2, 3, 9})

        assert select_abstract_span(contents, resolution).text_ids == (2, 3)

    @pytest.mark.parametrize(
        ("candidate_kwargs", "allowed_ids"),
        [
            ({"roles": {"byline"}, "page": 2}, {1, 2, 3}),
            ({"roles": {"affiliation"}, "page": 2}, {1, 2, 3}),
            ({"roles": {"doi"}, "page": 2}, {1, 2, 3}),
            ({"roles": {"title"}, "page": 2}, {1, 2, 3}),
            ({"roles": set(), "page": 3}, {1, 2, 3}),
            ({"roles": set(), "page": None}, {1, 2, 3}),
            ({"roles": set(), "page": 2}, {1, 2}),
        ],
    )
    def test_page_two_continuation_stops_at_unsafe_boundary(self, candidate_kwargs, allowed_ids):
        from bibr.structure.implicit_sections import select_abstract_span

        contents = self._bounded_contents()
        explicit = _candidate(
            "abstract-1",
            1,
            (2,),
            "Selected abstract sentence one.",
            roles={"abstract"},
        )
        unsafe = _candidate(
            "unsafe",
            2,
            (3,),
            "Selected continuation on page two.",
            **candidate_kwargs,
        )
        resolution = _selected_resolution(explicit, unsafe, allowed_text_ids=allowed_ids)

        assert select_abstract_span(contents, resolution).text_ids == (2,)
