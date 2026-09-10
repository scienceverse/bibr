"""Tests for bibr.structure.citation_linker — inline citation detection & resolution."""

import pytest

from bibr.paper import PaperReference
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
)
from bibr.structure import citation_linker
from bibr.structure.citation_linker import (
    _expand_numeric_range,
    _get_body_sentences,
    _is_likely_citation_bracket,
    detect_bib_xrefs,
    strip_citation_superscripts,
)
from bibr.structure.citation_matcher import (
    detect_author_year_xrefs,
    extract_families,
    match_with_ambiguous,
    match_with_candidates,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ref(bib_id, author="", year=2020, title="", text_id=None):
    # authors is now a plain string (or None) on PaperReference
    authors = author if author else None
    return PaperReference(
        bib_id=bib_id,
        title=title,
        first_page=None,
        volume=None,
        authors=authors,
        year=year,
        container=None,
        text_id=text_id,
    )


def _sent(text_id, text, section_id=1):
    return PaperSentence(text_id=text_id, text=text, section_id=section_id, paragraph_id=1)


def _numbered_reference_source(count: int):
    refs = [_ref(i, text_id=10_000 + i) for i in range(1, count + 1)]
    sentences = [
        _sent(10_000 + i, f"[{i}] Smith AB. Printed reference {i}.", section_id=2)
        for i in range(1, count + 1)
    ]
    return refs, sentences


async def _detect_numbered(sentences, bib_ids, sections=None):
    """Exercise the production linker with explicit printed reference evidence."""
    sections = list(sections or _body_only_sections())
    ref_section_id = max((section.section_id for section in sections), default=0) + 1
    sections.append(
        PaperSection(
            section_id=ref_section_id,
            header="References",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.REFERENCES,
        )
    )
    refs = [_ref(i, text_id=10_000 + i) for i in sorted(bib_ids)]
    ref_sentences = [
        _sent(10_000 + i, f"[{i}] Smith AB. Printed reference {i}.", section_id=ref_section_id)
        for i in sorted(bib_ids)
    ]
    return await detect_bib_xrefs(list(sentences) + ref_sentences, sections, refs)


async def _detect_with_receipt(*args, **kwargs):
    return await citation_linker.detect_bib_xrefs_with_receipt(*args, **kwargs)


def _sections_with_refs():
    """Return a section list where section_id=1 is body and section_id=2 is REFERENCES."""
    return [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        PaperSection(
            section_id=1,
            header="Introduction",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.INTRODUCTION,
        ),
        PaperSection(
            section_id=2,
            header="References",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.REFERENCES,
        ),
    ]


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestExpandNumericRange:
    def test_single(self):
        assert _expand_numeric_range("1") == [1]

    def test_comma(self):
        assert _expand_numeric_range("1,3") == [1, 3]

    def test_dash(self):
        assert _expand_numeric_range("1-3") == [1, 2, 3]

    def test_mixed(self):
        assert _expand_numeric_range("1-3,5") == [1, 2, 3, 5]

    def test_en_dash(self):
        assert _expand_numeric_range("1–3") == [1, 2, 3]

    def test_semicolon(self):
        assert _expand_numeric_range("1;3") == [1, 3]


class TestInPressCitationDetection:
    """Integration tests for in-press citation detection (Tier 2 matcher)."""

    def test_parenthetical_in_press(self):
        """(Smith, in press) should link to the year=0 reference."""
        refs = [_ref(1, author="Smith, J.", year=0)]
        sents = [_sent(0, "Previous work (Smith, in press) shows this.")]
        xrefs = detect_author_year_xrefs(sents, refs)
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 1
        assert "in press" in xrefs[0].contents.lower()

    def test_narrative_in_press(self):
        """Smith (in press) should link to the year=0 reference."""
        refs = [_ref(1, author="Smith, J.", year=0)]
        sents = [_sent(0, "Smith (in press) demonstrated this effect.")]
        xrefs = detect_author_year_xrefs(sents, refs)
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 1

    def test_forthcoming_detected(self):
        """(Author, forthcoming) should also link."""
        refs = [_ref(1, author="Jones, A.", year=0)]
        sents = [_sent(0, "As noted by (Jones, forthcoming) recently.")]
        xrefs = detect_author_year_xrefs(sents, refs)
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 1

    def test_no_match_when_no_in_press_ref(self):
        """In-press citation text with no year=0 ref should produce no xref."""
        refs = [_ref(1, author="Smith, J.", year=2020)]
        sents = [_sent(0, "Smith (in press) said something.")]
        xrefs = detect_author_year_xrefs(sents, refs)
        in_press_xrefs = [x for x in xrefs if "in press" in (x.contents or "").lower()]
        assert len(in_press_xrefs) == 0


class TestVancouverFamilyExtraction:
    def test_family_initials_records(self):
        assert extract_families("Smith AB, Jones CD, Brown EF") == [
            "Smith",
            "Jones",
            "Brown",
        ]

    def test_spaced_initial_suffixes_do_not_leak_into_families(self):
        assert extract_families("Smith J. A., Jones C. D.") == ["Smith", "Jones"]
        assert extract_families("van der Waals J D, García Márquez G") == [
            "van der Waals",
            "García Márquez",
        ]

    def test_particles_and_diacritics(self):
        assert extract_families("van der Waals JD, García Márquez G, Della Sala S") == [
            "van der Waals",
            "García Márquez",
            "Della Sala",
        ]
        refs = [_ref(1, author="García Márquez G, van der Waals JD", year=2020)]
        xrefs = detect_author_year_xrefs(
            [_sent(1, "Prior work (Garcia Marquez & van der Waals, 2020) reported this.")],
            refs,
        )
        assert [xref.xref_id for xref in xrefs] == [1]

    def test_apostrophe_particle_links_in_narrative_and_parenthetical_forms(self):
        refs = [_ref(1, author="d’Ardenne P", year=2014)]
        sentences = [
            _sent(1, "d’Ardenne (2014) reported this."),
            _sent(2, "Prior work (d’Ardenne, 2014) reported this."),
        ]

        xrefs = detect_author_year_xrefs(sentences, refs)

        assert [(xref.text_id, xref.xref_id) for xref in xrefs] == [(1, 1), (2, 1)]
        assert [xref.contents for xref in xrefs] == [
            "d’Ardenne (2014)",
            "(d’Ardenne, 2014)",
        ]

    def test_multiword_particle_narrative_keeps_full_source_span(self):
        refs = [_ref(1, author="van der Waals JD", year=2020)]

        xrefs = detect_author_year_xrefs([_sent(1, "van der Waals (2020) reported this.")], refs)

        assert [(xref.xref_id, xref.contents) for xref in xrefs] == [(1, "van der Waals (2020)")]

    def test_source_shaped_connectors_and_terminal_et_al(self):
        assert extract_families("Alipanga B, and Kohrt BA") == ["Alipanga", "Kohrt"]
        assert extract_families("Dunne E, and Rawlins M") == ["Dunne", "Rawlins"]
        assert extract_families("Hall J, d’Ardenne P, Nsereko J, et al.") == [
            "Hall",
            "d’Ardenne",
            "Nsereko",
        ]
        assert extract_families("Jordans MJD, Tol WA, et al.") == ["Jordans", "Tol"]

    def test_same_surname_same_year_abstains_without_initial_or_coauthor_evidence(self):
        refs = [
            _ref(1, author="Smith AB, Jones CD", year=2020),
            _ref(2, author="Smith CD", year=2020),
        ]

        xrefs, ambiguous = match_with_ambiguous(
            [_sent(1, "Smith (2020) reported this result.")], refs
        )

        assert xrefs == []
        assert len(ambiguous) == 1
        assert set(ambiguous[0].candidate_bib_ids) == {1, 2}

    def test_same_surname_same_year_et_al_abstains_without_coauthor_evidence(self):
        refs = [
            _ref(1, author="Smith AB, Jones CD, Brown EF", year=2020),
            _ref(2, author="Smith CD, Foo GH, Bar IJ, Baz KL", year=2020),
        ]

        xrefs, ambiguous = match_with_ambiguous(
            [_sent(1, "Smith et al. (2020) reported this result.")], refs
        )

        assert xrefs == []
        assert len(ambiguous) == 1
        assert set(ambiguous[0].candidate_bib_ids) == {1, 2}

    def test_coauthor_evidence_resolves_same_surname_same_year(self):
        refs = [
            _ref(1, author="Smith AB, Jones CD", year=2020),
            _ref(2, author="Smith CD, Brown EF", year=2020),
        ]

        xrefs = detect_author_year_xrefs(
            [_sent(1, "Smith and Jones (2020) reported this result.")], refs
        )

        assert [xref.xref_id for xref in xrefs] == [1]


class TestGetBodySentences:
    def test_excludes_references(self):
        sections = _sections_with_refs()
        sents = [_sent(0, "Body text.", 1), _sent(1, "Ref text.", 2)]
        body = _get_body_sentences(sents, sections)
        assert len(body) == 1
        assert body[0].text_id == 0

    def test_no_ref_section(self):
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(section_id=1, header="Intro", level=1, parent_section_id=0),
        ]
        sents = [_sent(0, "Body text.", 1)]
        body = _get_body_sentences(sents, sections)
        assert len(body) == 1

    def test_excludes_every_title_typed_section_regardless_of_level(self):
        sections = [
            PaperSection(
                section_id=1,
                header="Article title",
                level=2,
                parent_section_id=0,
                section_type=CanonicalSection.TITLE,
            ),
            PaperSection(
                section_id=2,
                header="Introduction",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.INTRODUCTION,
            ),
        ]
        sents = [_sent(1, "Kelders1,2", 1), _sent(2, "Body [1].", 2)]

        assert [sent.text_id for sent in _get_body_sentences(sents, sections)] == [2]

    def test_includes_abstract_citation_sentences(self):
        sections = [
            PaperSection(
                section_id=1,
                header="Abstract",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.ABSTRACT,
            )
        ]
        sents = [_sent(1, "Prior work [1] established this result.", 1)]

        assert [sent.text_id for sent in _get_body_sentences(sents, sections)] == [1]


# ---------------------------------------------------------------------------
# Tier 1: Numeric bracket citations
# ---------------------------------------------------------------------------


class TestNumericCitations:
    async def test_single_bracket(self):
        refs = [_ref(1), _ref(2), _ref(3)]
        sents = [_sent(0, "As shown in [1], the results are clear.")]
        xrefs = await detect_bib_xrefs(sents, [], refs)
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 1
        assert xrefs[0].xref_type == "bib"
        assert xrefs[0].text_id == 0

    async def test_range(self):
        refs = [_ref(1), _ref(2), _ref(3)]
        sents = [_sent(0, "Several studies [1-3] support this.")]
        xrefs = await detect_bib_xrefs(sents, [], refs)
        assert len(xrefs) == 3
        assert {x.xref_id for x in xrefs} == {1, 2, 3}

    async def test_comma_list(self):
        refs = [_ref(1), _ref(2), _ref(3)]
        sents = [_sent(0, "Previous work [1,3] shows this.")]
        xrefs = await detect_bib_xrefs(sents, [], refs)
        assert len(xrefs) == 2
        assert {x.xref_id for x in xrefs} == {1, 3}

    async def test_no_match(self):
        refs = [_ref(1), _ref(2)]
        sents = [_sent(0, "As noted in [99], this is rare.")]
        xrefs = await detect_bib_xrefs(sents, [], refs)
        assert len(xrefs) == 0

    async def test_en_dash(self):
        refs = [_ref(1), _ref(2), _ref(3)]
        sents = [_sent(0, "Studies [1–3] confirm this.")]
        xrefs = await detect_bib_xrefs(sents, [], refs)
        assert len(xrefs) == 3
        assert {x.xref_id for x in xrefs} == {1, 2, 3}


class TestMathBracketRejection:
    """A bracket containing 0 (or below) is math notation, never a 1-indexed citation.

    Regression: the unit interval ``[0, 1]`` in a maths-heavy paper was tagged as a
    citation because its only in-range member (1) coincided with bib_id 1.
    """

    def test_unit_interval_not_citation_helper(self):
        # bib_ids are 1-indexed, so 0 disqualifies the whole bracket
        assert _is_likely_citation_bracket("0, 1", [0, 1], {1, 2, 3}) is False

    async def test_unit_interval_no_space(self):
        refs = [_ref(1), _ref(2), _ref(3)]
        sents = [_sent(0, "The probability lies in [0,1] for all cases.")]
        xrefs = await detect_bib_xrefs(sents, [], refs)
        assert len(xrefs) == 0

    async def test_unit_interval_with_space(self):
        refs = [_ref(1), _ref(2), _ref(3)]
        sents = [_sent(0, "We restrict the weight to the interval [0, 1].")]
        xrefs = await detect_bib_xrefs(sents, [], refs)
        assert len(xrefs) == 0

    def test_legit_citations_survive(self):
        # the guard must not break ordinary numbered citations (never 0/negative)
        assert _is_likely_citation_bracket("1", [1], {1, 2, 3}) is True
        assert _is_likely_citation_bracket("1, 3", [1, 3], {1, 2, 3}) is True


# ---------------------------------------------------------------------------
# Tier 1b: Superscript citations
# ---------------------------------------------------------------------------


class TestSuperscriptCitations:
    async def test_single_superscript(self):
        refs = [_ref(1), _ref(2), _ref(3)]
        sents = [_sent(0, "evidence for this ^{3} is strong.")]
        xrefs = await detect_bib_xrefs(sents, [], refs)
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 3
        assert xrefs[0].xref_type == "bib"
        assert xrefs[0].contents == "^{3}"

    async def test_comma_list_superscript(self):
        refs = [_ref(i) for i in range(1, 20)]
        sents = [_sent(0, "heritability of preferences ^{15,16} for some traits")]
        xrefs = await detect_bib_xrefs(sents, [], refs)
        assert len(xrefs) == 2
        assert {x.xref_id for x in xrefs} == {15, 16}

    async def test_multiple_superscripts_in_sentence(self):
        refs = [_ref(i) for i in range(1, 20)]
        sents = [_sent(0, "parent ^{17} and spouse ^{8,9} resemblance")]
        xrefs = await detect_bib_xrefs(sents, [], refs)
        assert len(xrefs) == 3
        assert {x.xref_id for x in xrefs} == {8, 9, 17}

    async def test_superscript_range(self):
        refs = [_ref(1), _ref(2), _ref(3)]
        sents = [_sent(0, "Studies ^{1-3} support this.")]
        xrefs = await detect_bib_xrefs(sents, [], refs)
        assert len(xrefs) == 3
        assert {x.xref_id for x in xrefs} == {1, 2, 3}

    async def test_no_match_invalid_bib_id(self):
        refs = [_ref(1), _ref(2)]
        sents = [_sent(0, "As noted ^{99} this is rare.")]
        xrefs = await detect_bib_xrefs(sents, [], refs)
        assert len(xrefs) == 0

    async def test_mixed_bracket_and_superscript(self):
        refs = [_ref(1), _ref(2), _ref(3)]
        sents = [_sent(0, "See [1] and also ^{2,3} for details.")]
        xrefs = await detect_bib_xrefs(sents, [], refs)
        assert len(xrefs) == 3
        assert {x.xref_id for x in xrefs} == {1, 2, 3}


# ---------------------------------------------------------------------------
# Tier 2: Author-year citations
# ---------------------------------------------------------------------------


class TestAuthorYearCitations:
    def test_parenthetical(self):
        refs = [_ref(1, author="Smith, J.", year=2020)]
        sents = [_sent(0, "This was shown (Smith, 2020) in their study.")]
        xrefs = detect_author_year_xrefs(sents, refs)
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 1

    def test_narrative(self):
        refs = [_ref(1, author="Smith, J.", year=2020)]
        sents = [_sent(0, "Smith (2020) demonstrated this effect.")]
        xrefs = detect_author_year_xrefs(sents, refs)
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 1

    def test_et_al(self):
        refs = [_ref(1, author="Williams, A., Chen, B., Park, C.", year=2021)]
        sents = [_sent(0, "Williams et al. (2021) reported similar findings.")]
        xrefs = detect_author_year_xrefs(sents, refs)
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 1

    def test_multi_cite(self):
        refs = [
            _ref(1, author="Smith, J.", year=2020),
            _ref(2, author="Jones, K.", year=2019),
        ]
        sents = [_sent(0, "Previous work (Smith, 2020; Jones, 2019) agrees.")]
        xrefs = detect_author_year_xrefs(sents, refs)
        assert len(xrefs) == 2
        assert {x.xref_id for x in xrefs} == {1, 2}

    def test_no_match(self):
        refs = [_ref(1, author="Smith, J.", year=2020)]
        sents = [_sent(0, "As shown (Unknown, 1999) this is rare.")]
        xrefs = detect_author_year_xrefs(sents, refs)
        assert len(xrefs) == 0


# ---------------------------------------------------------------------------
# Integration: detect_bib_xrefs
# ---------------------------------------------------------------------------


class TestDetectBibXrefs:
    async def test_empty_refs(self):
        result = await detect_bib_xrefs(
            sentences=[_sent(0, "Some text [1].")],
            sections=_sections_with_refs(),
            references=[],
        )
        assert result == []

    async def test_deduplication(self):
        """Same bib_id from Tier 1 numeric and Tier 2 author-year should deduplicate."""
        refs = [_ref(1, author="Smith, J.", year=2020)]
        sents = [_sent(0, "As shown in [1] by Smith (2020).")]

        result = await detect_bib_xrefs(
            sentences=sents,
            sections=_sections_with_refs(),
            references=refs,
        )
        # Should be deduplicated to a single xref (same text_id + bib_id)
        bib_xrefs = [x for x in result if x.xref_type == "bib"]
        assert len(bib_xrefs) == 1
        assert bib_xrefs[0].xref_id == 1

    async def test_fully_resolved_multicite_span_skips_llm(self):
        refs = [
            _ref(1, author="Smith, J.", year=2020),
            _ref(2, author="Jones, K.", year=2019),
        ]

        class FakeLLM:
            def __init__(self):
                self.calls = []

            async def resolve_citations(
                self, ambiguous_citations, reference_summary, file_hash="x"
            ):
                self.calls.append(list(ambiguous_citations))
                return []

        llm = FakeLLM()
        result = await detect_bib_xrefs(
            sentences=[_sent(1, "Prior work (Smith, 2020; Jones, 2019) agrees.")],
            sections=_sections_with_refs(),
            references=refs,
            llm_client=llm,
        )

        assert {xref.xref_id for xref in result} == {1, 2}
        assert llm.calls == []

    async def test_partially_resolved_multicite_span_reaches_llm(self):
        refs = [_ref(1, author="Smith, J.", year=2020)]
        captured = []

        class FakeLLM:
            async def resolve_citations(
                self, ambiguous_citations, reference_summary, file_hash="x"
            ):
                captured.extend(ambiguous_citations)
                return []

        await detect_bib_xrefs(
            sentences=[_sent(1, "Prior work (Smith, 2020; Unknown, 2019) agrees.")],
            sections=_sections_with_refs(),
            references=refs,
            llm_client=FakeLLM(),
        )

        assert captured == [(1, "(Smith, 2020; Unknown, 2019)")]

    async def test_excludes_reference_section(self):
        """Citations in the reference section should not be detected."""
        refs = [_ref(1), _ref(2)]
        sents = [
            _sent(0, "Body has [1].", section_id=1),
            _sent(1, "[1] Smith J. Title.", section_id=2),  # in references section
        ]

        result = await detect_bib_xrefs(
            sentences=sents,
            sections=_sections_with_refs(),
            references=refs,
        )
        assert len(result) == 1
        assert result[0].text_id == 0

    async def test_combined_tiers(self):
        """Tier 2 and Tier 3 can both contribute xrefs."""
        refs = [
            _ref(1, author="Smith, J.", year=2020),
            _ref(2, author="Jones, K.", year=2019),
        ]
        sents = [
            _sent(0, "As shown in [1], this is important."),
            _sent(1, "Jones (2019) agreed with these findings."),
        ]
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(section_id=1, header="Intro", level=1, parent_section_id=0),
        ]

        result = await detect_bib_xrefs(
            sentences=sents,
            sections=sections,
            references=refs,
        )
        assert len(result) == 2
        bib_ids = {x.xref_id for x in result}
        assert bib_ids == {1, 2}


class TestDetectBibXrefsTier3Widened:
    """Tests for Tier 3 widening to include unresolved author-year citations."""

    async def test_unresolved_author_year_sent_to_llm(self):
        """Author-year citation with no matching ref should go to Tier 3."""
        from unittest.mock import AsyncMock

        from bibr.schemas import CitationMatch, CitationResolutionResult

        refs = [_ref(1, author="Smith, J.", year=2020)]
        sents = [_sent(0, "As shown by Jones (2019), this is important.")]
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(section_id=1, header="Intro", level=1, parent_section_id=0),
        ]

        mock_llm = AsyncMock()
        mock_llm.resolve_citations = AsyncMock(
            return_value=CitationResolutionResult(
                matches=[CitationMatch(text_id=0, citation_text="Jones (2019)", bib_id=None)]
            )
        )

        await detect_bib_xrefs(
            sentences=sents,
            sections=sections,
            references=refs,
            llm_client=mock_llm,
        )

        # LLM should have been called with the unresolved author-year citation
        mock_llm.resolve_citations.assert_called_once()

    async def test_resolved_author_year_not_sent_to_llm(self):
        """Author-year citation that Tier 2 resolved should NOT go to Tier 3."""
        from unittest.mock import AsyncMock

        refs = [_ref(1, author="Smith, J.", year=2020)]
        sents = [_sent(0, "As shown by Smith (2020), this is important.")]
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(section_id=1, header="Intro", level=1, parent_section_id=0),
        ]

        mock_llm = AsyncMock()
        mock_llm.resolve_citations = AsyncMock(return_value=[])

        result = await detect_bib_xrefs(
            sentences=sents,
            sections=sections,
            references=refs,
            llm_client=mock_llm,
        )

        # Tier 2 resolves it, Tier 3 should not be called
        assert len(result) == 1
        assert result[0].xref_id == 1
        mock_llm.resolve_citations.assert_not_called()

    async def test_tier2_ambiguous_routed_to_llm(self):
        """When Tier 2 finds 2+ refs tied for the same author-year cite, the
        candidate span should be routed to Tier 3 for disambiguation."""
        from unittest.mock import AsyncMock

        from bibr.schemas import CitationMatch

        refs = [
            _ref(1, author="Smith, J., & Jones, K.", year=2020),
            _ref(2, author="Smith, J., & Williams, L.", year=2020),
        ]
        sents = [_sent(0, "As shown (Smith, 2020) this works.", section_id=1)]
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(section_id=1, header="Intro", level=1, parent_section_id=0),
        ]

        captured: dict = {}

        async def mock_resolve(ambiguous_citations, reference_summary, file_hash=None):
            captured["citations"] = ambiguous_citations
            return [CitationMatch(text_id=0, citation_text="(Smith, 2020)", bib_id=1)]

        mock_llm = AsyncMock()
        mock_llm.resolve_citations = mock_resolve

        result = await detect_bib_xrefs(
            sentences=sents,
            sections=sections,
            references=refs,
            llm_client=mock_llm,
        )

        # The ambiguous-tie span must have been forwarded to the LLM.
        cite_texts = [ct for _, ct in captured["citations"]]
        assert "(Smith, 2020)" in cite_texts
        # And the LLM's pick should appear in the final xref set.
        assert any(x.xref_id == 1 and x.text_id == 0 for x in result)

    async def test_hallucinated_bib_id_dropped(self):
        """Tier-3 matches whose bib_id doesn't exist in the references must be
        dropped — a hallucinated bib_id would otherwise flow unvalidated into
        the exported xrefs (Tier 1 gates on valid_bib_ids; Tier 3 must too)."""
        from unittest.mock import AsyncMock

        from bibr.schemas import CitationMatch

        refs = [_ref(1, author="WHO Collaborative Study Team", year=2019, title="Something")]
        sents = [_sent(0, "As shown [WHO, 2019], this is important.")]
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(section_id=1, header="Intro", level=1, parent_section_id=0),
        ]

        async def mock_resolve(ambiguous_citations, reference_summary, file_hash=None):
            # bib_id=7 does not exist (only bib_id=1 does) — e.g. copied from
            # the in-prompt worked example or hallucinated outright.
            return [CitationMatch(text_id=0, citation_text="[WHO, 2019]", bib_id=7)]

        mock_llm = AsyncMock()
        mock_llm.resolve_citations = mock_resolve

        result = await detect_bib_xrefs(
            sentences=sents,
            sections=sections,
            references=refs,
            llm_client=mock_llm,
        )

        assert not any(x.xref_id == 7 for x in result)

    async def test_duplicate_citations_deduplicated_for_llm(self):
        """Same citation text in multiple sentences should be sent to LLM only once."""
        from unittest.mock import AsyncMock

        from bibr.schemas import CitationMatch

        # Use a ref that Tier 2 cannot resolve (author name won't match pattern)
        refs = [_ref(1, author="WHO Collaborative Study Team", year=2019, title="Something")]
        # Non-numeric bracket citation — same text in 3 sentences, unresolvable by Tier 1/2
        sents = [
            _sent(0, "As shown [WHO, 2019], this is important."),
            _sent(1, "Evidence from [WHO, 2019] supports this."),
            _sent(2, "Further support from [WHO, 2019] was found."),
        ]
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(section_id=1, header="Intro", level=1, parent_section_id=0),
        ]

        captured_args = {}

        async def mock_resolve(ambiguous_citations, reference_summary, file_hash=None):
            captured_args["citations"] = ambiguous_citations
            # resolve_citations returns a list of CitationMatch (not CitationResolutionResult)
            return [
                CitationMatch(text_id=0, citation_text="[WHO, 2019]", bib_id=1),
            ]

        mock_llm = AsyncMock()
        mock_llm.resolve_citations = mock_resolve

        result = await detect_bib_xrefs(
            sentences=sents,
            sections=sections,
            references=refs,
            llm_client=mock_llm,
        )

        # LLM should receive deduplicated list (1 unique citation text, not 3)
        llm_citations = captured_args["citations"]
        cite_texts = [ct for _, ct in llm_citations]
        assert cite_texts.count("[WHO, 2019]") == 1

        # But xrefs should be created for all 3 sentences
        who_xrefs = [x for x in result if x.xref_id == 1]
        text_ids = {x.text_id for x in who_xrefs}
        assert text_ids == {0, 1, 2}


# ---------------------------------------------------------------------------
# Tier 3 bracket pre-filter
# ---------------------------------------------------------------------------


class TestNonCitationBracketPrefilter:
    def test_confidence_intervals_filtered(self):
        from bibr.structure.citation_linker import _looks_like_non_citation_bracket

        assert _looks_like_non_citation_bracket(".59, .72")
        assert _looks_like_non_citation_bracket("-.12, .04")
        assert _looks_like_non_citation_bracket("0.000, 0.114")

    def test_item_refs_filtered(self):
        from bibr.structure.citation_linker import _looks_like_non_citation_bracket

        assert _looks_like_non_citation_bracket("Item 5")
        assert _looks_like_non_citation_bracket("item 7")

    def test_uppercase_abbreviations_filtered(self):
        from bibr.structure.citation_linker import _looks_like_non_citation_bracket

        assert _looks_like_non_citation_bracket("IRR")
        assert _looks_like_non_citation_bracket("OR")
        assert _looks_like_non_citation_bracket("SE")

    def test_editorial_phrases_filtered(self):
        from bibr.structure.citation_linker import _looks_like_non_citation_bracket

        assert _looks_like_non_citation_bracket("missed opportunity")
        assert _looks_like_non_citation_bracket("sic")
        assert _looks_like_non_citation_bracket("emphasis added")
        assert _looks_like_non_citation_bracket("Emphasis Added")

    def test_bracketed_surname_survives(self):
        from bibr.structure.citation_linker import _looks_like_non_citation_bracket

        assert not _looks_like_non_citation_bracket("Smith")
        assert not _looks_like_non_citation_bracket("Jones-Brown")
        assert not _looks_like_non_citation_bracket("O'Brien")

    async def test_bracketed_surname_reaches_llm(self):
        """[Smith]-style citations must survive the pre-filter and reach Tier 3."""
        from unittest.mock import AsyncMock

        from bibr.schemas import CitationMatch

        refs = [_ref(1, author="Smith, J.", year=2020)]
        sents = [_sent(0, "As shown [Smith], this is important [sic].")]
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(section_id=1, header="Intro", level=1, parent_section_id=0),
        ]

        captured: dict = {}

        async def mock_resolve(ambiguous_citations, reference_summary, file_hash=None):
            captured["citations"] = ambiguous_citations
            return [CitationMatch(text_id=0, citation_text="[Smith]", bib_id=1)]

        mock_llm = AsyncMock()
        mock_llm.resolve_citations = mock_resolve

        result = await detect_bib_xrefs(
            sentences=sents,
            sections=sections,
            references=refs,
            llm_client=mock_llm,
        )

        cite_texts = [ct for _, ct in captured["citations"]]
        assert "[Smith]" in cite_texts
        assert "[sic]" not in cite_texts
        assert any(x.xref_id == 1 and x.text_id == 0 for x in result)


# ---------------------------------------------------------------------------
# strip_citation_superscripts
# ---------------------------------------------------------------------------


def test_level0_unknown_section_is_body():
    from bibr.paper_contents import CanonicalSection, PaperSection, PaperSentence
    from bibr.structure.citation_linker import _get_body_sentences

    sections = [
        PaperSection(
            section_id=1,
            header="Title",
            level=0,
            parent_section_id=None,
            section_type=CanonicalSection.TITLE,
        ),
        PaperSection(
            section_id=8,
            header="",
            level=0,
            parent_section_id=None,
            section_type=CanonicalSection.UNKNOWN,
        ),
    ]
    sentences = [
        PaperSentence(text_id=1, text="Byline (A & B, 2001)", section_id=1, paragraph_id=1),
        PaperSentence(text_id=25, text="…(Bond & Smith, 1996).", section_id=8, paragraph_id=1),
    ]
    body_ids = {s.text_id for s in _get_body_sentences(sentences, sections)}
    assert 25 in body_ids  # level-0 UNKNOWN body-continuation now included
    assert 1 not in body_ids  # level-0 TITLE still excluded


class TestSuperscriptCitationDetection:
    """A closing brace is neither ``\\w`` nor ``$``, so the two lookbehinds
    guarding SUPERSCRIPT_CITE_RE let a braced LaTeX base through and the
    exponent was detected as a bib citation."""

    def test_braced_latex_base_is_not_a_citation(self):
        from bibr.structure.citation_linker import SUPERSCRIPT_CITE_RE

        assert not SUPERSCRIPT_CITE_RE.search(r"measured $\mathrm{cm}^{2}$ in area")

    def test_word_carried_superscript_is_still_a_citation(self):
        from bibr.structure.citation_linker import SUPERSCRIPT_CITE_RE

        assert SUPERSCRIPT_CITE_RE.search("as reported. ^{12} in trials")


class TestStripCitationSuperscripts:
    def test_single(self):
        sents = [_sent(0, "effective^{9} in many cases.")]
        strip_citation_superscripts(sents, [])
        assert sents[0].text == "effective in many cases."

    def test_range(self):
        sents = [_sent(0, "DHIs.^{6-8,12} This marks")]
        strip_citation_superscripts(sents, [])
        assert sents[0].text == "DHIs. This marks"

    def test_multiple_per_sentence(self):
        sents = [_sent(0, "parent^{17} and spouse^{8,9} resemblance")]
        strip_citation_superscripts(sents, [])
        assert sents[0].text == "parent and spouse resemblance"

    def test_preserves_math_superscript(self):
        """Word-preceded superscripts like x^{2} are not citation markers."""
        sents = [_sent(0, "we computed x^{2} for each group.")]
        strip_citation_superscripts(sents, [])
        assert sents[0].text == "we computed x^{2} for each group."

    def test_preserves_numeric_scientific_exponent(self):
        sents = [_sent(0, "The sample size was 10^{4} observations.")]
        strip_citation_superscripts(sents, [])
        assert sents[0].text == "The sample size was 10^{4} observations."

    def test_preserves_dollar_superscript(self):
        """Dollar-preceded superscripts like $^{2}$ are math, not citations."""
        sents = [_sent(0, "partial $^{2}$ was reported.")]
        strip_citation_superscripts(sents, [])
        assert sents[0].text == "partial $^{2}$ was reported."

    def test_preserves_braced_latex_base(self):
        """Math was recognised only when the single preceding character was
        alphabetic and the one before it was not, so a LaTeX-braced base fell
        through to "citation" — the unit was deleted from the sentence."""
        sents = [_sent(0, r"Each plot measured $\mathrm{cm}^{2}$ in area.")]
        strip_citation_superscripts(sents, [])
        assert sents[0].text == r"Each plot measured $\mathrm{cm}^{2}$ in area."

    def test_preserves_multi_letter_greek_base(self):
        sents = [_sent(0, "The effect was large \u03b7p^{2} = .14.")]
        strip_citation_superscripts(sents, [])
        assert sents[0].text == "The effect was large \u03b7p^{2} = .14."

    def test_still_strips_a_word_carrying_a_marker_inside_a_math_free_sentence(self):
        sents = [_sent(0, "This was effective^{9} in trials.")]
        strip_citation_superscripts(sents, [])
        assert sents[0].text == "This was effective in trials."

    def test_skips_reference_section(self):
        sections = _sections_with_refs()
        sents = [
            _sent(0, "body^{1} text.", section_id=1),
            _sent(1, "ref^{1} text.", section_id=2),
        ]
        strip_citation_superscripts(sents, sections)
        assert sents[0].text == "body text."
        assert sents[1].text == "ref^{1} text."

    def test_space_before_superscript(self):
        sents = [_sent(0, "scarce. ^{4,5} Engagement")]
        strip_citation_superscripts(sents, [])
        assert sents[0].text == "scarce.  Engagement"

    def test_consumes_whitespace_padded_dollar_wrapper_atomically(self):
        sents = [
            _sent(
                0,
                "conditions, such as cardiac rehabilitation, $ ^{1} $ diabetes "
                "and obesity, and depression. $ ^{2} $",
            )
        ]
        strip_citation_superscripts(sents, [])
        assert "$" not in sents[0].text
        assert "rehabilitation,  diabetes" in sents[0].text

    def test_preserves_compact_dollar_math_superscript(self):
        sents = [_sent(0, "partial eta $^{2}$ was reported.")]
        strip_citation_superscripts(sents, [])
        assert sents[0].text == "partial eta $^{2}$ was reported."


# ---------------------------------------------------------------------------
# Tier 1c: Parenthetical-numeric citations  (3), (7, 8), (6-8)
# ---------------------------------------------------------------------------


class TestDetectParentheticalNumericStyleGate:
    async def test_single_sentence_below_gate(self):
        """One matched sentence must not clear the style-consistency gate."""

        sents = [_sent(0, "the result (3) shows this.")]
        assert await _detect_numbered(sents, set(range(1, 21))) == []

    async def test_list_enumeration_below_gate(self):
        """(1) ... (2) ... (3) in ONE sentence is a list, not citations."""

        sents = [_sent(0, "steps are (1) first, (2) second, and (3) third.")]
        assert await _detect_numbered(sents, set(range(1, 21))) == []

    async def test_three_sentences_pass_gate(self):

        sents = [
            _sent(0, "shown in (3) clearly."),
            _sent(1, "confirmed by (7, 8) later."),
            _sent(2, "and also (5) elsewhere."),
        ]
        xrefs = await _detect_numbered(sents, set(range(1, 21)))
        assert {x.xref_id for x in xrefs} == {3, 5, 7, 8}
        assert all(x.xref_type == "bib" for x in xrefs)

    async def test_multi_sentence_procedural_list_is_rejected_with_reasons(self):
        refs, ref_sents = _numbered_reference_source(8)
        sents = [
            _sent(1, "(1) Collect the source files."),
            _sent(2, "(2) Normalize each record."),
            _sent(3, "(3) Export the final table."),
        ] + ref_sents

        xrefs, receipt = await _detect_with_receipt(sents, _sections_with_refs(), refs)

        assert xrefs == []
        procedural = [
            candidate for candidate in receipt.candidates if candidate.style == "paren-numeric"
        ]
        assert len(procedural) == 3
        assert all("list_enumeration" in item.rejection_reasons for item in procedural)


class TestDetectFlattenedSuperscriptsStyleGate:
    async def test_three_sentences_pass_gate(self):

        sents = [
            _sent(0, "similar physical traits1, but"),
            _sent(1, "own characteristics2,3."),
            _sent(2, "still scarce.4,5"),
        ]
        xrefs = await _detect_numbered(sents, set(range(1, 43)), sections=[])
        assert {x.xref_id for x in xrefs} == {1, 2, 3, 4, 5}

    async def test_below_gate_returns_empty(self):

        sents = [_sent(0, "similar physical traits1, but")]
        assert await _detect_numbered(sents, set(range(1, 43)), sections=[]) == []

    async def test_title_section_excluded(self):
        """Author bylines carry affiliation superscripts (Kelders1,2) that must
        NOT be mistaken for citations, even in a doc that otherwise clears the
        gate."""

        sections = [
            PaperSection(
                section_id=1,
                header="Title",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.TITLE,
            ),
            PaperSection(
                section_id=2,
                header="Intro",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.INTRODUCTION,
            ),
        ]
        sents = [
            _sent(0, "Saskia M Kelders1,2 , Hanneke Kip1,3 , Nienke Jong4", section_id=1),
            _sent(1, "cardiac rehabilitation,1 diabetes", section_id=2),
            _sent(2, "own characteristics2,3.", section_id=2),
            _sent(3, "still scarce.4,5", section_id=2),
        ]
        xrefs = await _detect_numbered(sents, set(range(1, 43)), sections=sections)
        # byline superscripts on text_id 0 (title) must be absent
        assert 0 not in {x.text_id for x in xrefs}
        assert {x.xref_id for x in xrefs} == {1, 2, 3, 4, 5}


# ---------------------------------------------------------------------------
# Integration: fallback tiers via detect_bib_xrefs
# ---------------------------------------------------------------------------

# Invented prose preserves citation marker shapes without retaining paper excerpts.
_PAREN_STYLE_SENTS = [
    "The sample procedure follows an earlier comparison (3).",
    "Additional reports (6) describe related examples (7, 8).",
    "A separate analysis (9) agrees with the review (15).",
]
_WORD_ATTACHED_STYLE_SENTS = [
    "The first comparison concerns classroom tasks1, with several variations.",
    "Later evidence builds on those observations2,3.",
    "Further examples4,5 support the final comparison6,7.",
]
_PUNCTUATION_STYLE_SENTS = [
    "The examples include reading,1 drawing,2 and discussion.3",
    "Evidence about the classroom activities remains limited.4,5",
    "A related analysis considers different schedules.6–8",
    "Participation is associated with clearer explanations.7,10,11",
]


def _body_only_sections():
    return [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        PaperSection(
            section_id=1,
            header="Intro",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.INTRODUCTION,
        ),
    ]


class TestFallbackTiersIntegration:
    async def test_parenthetical_style_fixture(self):
        refs, ref_sents = _numbered_reference_source(20)
        sents = [_sent(i, t) for i, t in enumerate(_PAREN_STYLE_SENTS)] + ref_sents
        result = await detect_bib_xrefs(sents, _sections_with_refs(), refs)
        ids = {x.xref_id for x in result}
        assert ids == {3, 6, 7, 8, 9, 15}
        assert all(1 <= x.xref_id <= 20 for x in result)

    async def test_word_attached_style_fixture(self):
        refs, ref_sents = _numbered_reference_source(21)
        sents = [_sent(i, t) for i, t in enumerate(_WORD_ATTACHED_STYLE_SENTS)] + ref_sents
        result = await detect_bib_xrefs(sents, _sections_with_refs(), refs)
        assert {x.xref_id for x in result} == {1, 2, 3, 4, 5, 6, 7}

    async def test_punctuation_attached_style_fixture(self):
        refs, ref_sents = _numbered_reference_source(42)
        sents = [_sent(i, t) for i, t in enumerate(_PUNCTUATION_STYLE_SENTS)] + ref_sents
        result = await detect_bib_xrefs(sents, _sections_with_refs(), refs)
        assert len(result) > 0
        assert all(1 <= x.xref_id <= 42 for x in result)
        assert {x.xref_id for x in result} == {1, 2, 3, 4, 5, 6, 7, 8, 10, 11}

    async def test_one_incidental_bracket_does_not_veto_recurring_flattened_style(self):
        refs, ref_sents = _numbered_reference_source(42)
        sents = [
            _sent(0, "As shown in [1] and [2]."),
            _sent(1, "cardiac rehabilitation,3 diabetes"),
            _sent(2, "own characteristics4,5."),
            _sent(3, "still scarce.6,7"),
        ] + ref_sents

        result, receipt = await _detect_with_receipt(sents, _sections_with_refs(), refs)

        assert {x.xref_id for x in result} == {1, 2, 3, 4, 5, 6, 7}
        assert receipt.style_scores["numeric"] == 1 / 3
        assert any(x.tier == "flattened-superscript" for x in result)

    async def test_recurring_primary_numeric_and_flattened_styles_resolve_independently(self):
        refs, ref_sents = _numbered_reference_source(42)
        sents = [
            _sent(0, "As shown in [1]."),
            _sent(1, "Later work [2] agreed."),
            _sent(2, "A final study [3] confirmed it."),
            _sent(3, "cardiac rehabilitation,4 diabetes"),
            _sent(4, "own characteristics5,6."),
            _sent(5, "still scarce.7,8"),
        ] + ref_sents

        result, receipt = await _detect_with_receipt(sents, _sections_with_refs(), refs)

        assert {x.xref_id for x in result} == {1, 2, 3, 4, 5, 6, 7, 8}
        assert receipt.style_scores["numeric"] == 1.0
        assert {x.tier for x in result} == {"numeric", "flattened-superscript"}

    async def test_recurring_primary_numeric_and_parenthetical_styles_resolve_independently(self):
        refs, ref_sents = _numbered_reference_source(12)
        sents = [
            _sent(0, "As shown in [1]."),
            _sent(1, "Later work [2] agreed."),
            _sent(2, "A final study [3] confirmed it."),
            _sent(3, "The first comparison (4) held."),
            _sent(4, "A replication (5) agreed."),
            _sent(5, "The final comparison (6) held."),
        ] + ref_sents

        result, receipt = await _detect_with_receipt(sents, _sections_with_refs(), refs)

        assert {xref.xref_id for xref in result} == {1, 2, 3, 4, 5, 6}
        assert receipt.style_scores["numeric"] == 1.0
        assert {xref.tier for xref in result} == {"numeric", "paren-numeric"}

    async def test_stronger_flattened_style_does_not_suppress_real_parenthetical_style(self):
        refs, ref_sents = _numbered_reference_source(20)
        sents = [
            _sent(1, "Prior evidence.1,2 established this."),
            _sent(2, "Later reports3,4 confirmed it."),
            _sent(3, "A synthesis.5,6 agreed."),
            _sent(4, "Further findings7,8 remained stable."),
            _sent(5, "Other analyses.9,10 replicated it."),
            _sent(6, "Final reports11,12 supported it."),
            _sent(7, "The first external study (13) reported the same result."),
            _sent(8, "A separate replication (14) also agreed."),
            _sent(9, "The final comparison (15) reached the same conclusion."),
        ] + ref_sents

        result, receipt = await _detect_with_receipt(sents, _sections_with_refs(), refs)

        assert {xref.xref_id for xref in result} == set(range(1, 16))
        assert {xref.tier for xref in result} == {"flattened-superscript", "paren-numeric"}
        assert all(
            "weaker_competing_style" not in candidate.rejection_reasons
            for candidate in receipt.candidates
        )

    async def test_procedural_parenthetical_sequence_across_sentences_is_rejected(self):
        refs, ref_sents = _numbered_reference_source(12)
        sents = [
            _sent(1, "First, perform step (1) carefully."),
            _sent(2, "Second, repeat step (2) independently."),
            _sent(3, "Third, record step (3) in the form."),
        ] + ref_sents

        xrefs, receipt = await _detect_with_receipt(sents, _sections_with_refs(), refs)

        assert xrefs == []
        procedural = [
            candidate for candidate in receipt.candidates if candidate.style == "paren-numeric"
        ]
        assert {candidate.raw for candidate in procedural} == {"(1)", "(2)", "(3)"}
        assert all(
            not candidate.accepted and "list_enumeration" in candidate.rejection_reasons
            for candidate in procedural
        )

    async def test_discourse_ordinals_do_not_suppress_sentence_final_citations(self):
        refs, ref_sents = _numbered_reference_source(20)
        sents = [
            _sent(1, "First, the original cohort reported this association (13)."),
            _sent(2, "Second, an independent replication confirmed it (14)."),
            _sent(3, "Finally, the pooled analysis supported the result (15)."),
        ] + ref_sents

        xrefs, receipt = await _detect_with_receipt(sents, _sections_with_refs(), refs)

        assert {xref.xref_id for xref in xrefs} == {13, 14, 15}
        assert all(
            candidate.accepted
            for candidate in receipt.candidates
            if candidate.style == "paren-numeric"
        )

    async def test_aggregate_words_do_not_suppress_sentence_final_citations(self):
        refs, ref_sents = _numbered_reference_source(20)
        sents = [
            _sent(1, "Many participants showed the previously reported association (13)."),
            _sent(2, "The majority of respondents confirmed the published finding (14)."),
            _sent(3, "Half of the samples reproduced the earlier result (15)."),
        ] + ref_sents

        xrefs, receipt = await _detect_with_receipt(sents, _sections_with_refs(), refs)

        assert {xref.xref_id for xref in xrefs} == {13, 14, 15}
        assert all(
            candidate.accepted
            for candidate in receipt.candidates
            if candidate.style == "paren-numeric"
        )

    async def test_local_guards_reject_incidental_parenthetical_markers(self):
        """Synthetic mixed-marker prose separates citations from counts and labels."""
        refs, ref_sents = _numbered_reference_source(248)
        supported_titles = {
            3: "An Example Plan for Classroom Activities",
            5: "A Fictional Declaration on Shared Learning",
            6: "A Roadmap for Example Learning Groups",
            10: "Comparing Explanations Across Invented Classroom Tasks",
            63: "Shared Example Research Framework (FRAME)",
        }
        refs = [
            reference.model_copy(
                update={
                    "title": supported_titles.get(reference.bib_id, reference.title),
                    "text_id": None if reference.bib_id == 64 else reference.text_id,
                }
            )
            if reference.bib_id in supported_titles or reference.bib_id == 64
            else reference
            for reference in refs
        ]
        source_rows = {
            key: f"{key}. Example Research Group. {title}. Example Press, 2024."
            for key, title in supported_titles.items()
        }
        source_rows[64] = "64. Example Research Group. Example Watch. Example Press, 2024."
        ref_sents = [
            _sent(sentence.text_id, source_rows.get(index, sentence.text), section_id=2)
            for index, sentence in enumerate(ref_sents, start=1)
        ]
        sents = [
            _sent(
                140, "Many classroom indicators were grouped within the Discussion subdomain (89)."
            ),
            _sent(177, "The majority of classroom indicators (38) were from Example Watch64."),
            _sent(
                195,
                "Over half of the classroom indicators were from the Shared Example Research Framework (FRAME)63 (207) and INDEX 213 (111).",
            ),
            _sent(
                307,
                "The selected indicators must: (1) describe an activity; (2) be observable; (3) allow comparison; (4) concern learning.",
            ),
            _sent(17, "An example lesson introduces an unfamiliar topic to the learners1."),
            _sent(18, "The accompanying exercise asks each learner to explain a drawing2."),
            _sent(19, "Related classroom examples provide a useful comparison3,4."),
            _sent(20, "A fictional declaration describes shared learning goals5."),
            _sent(21, "A later roadmap outlines how groups can organize their activities6."),
            _sent(22, "The comparison includes a second set of invented tasks7."),
            _sent(24, "Additional examples build on these earlier descriptions5,8,9."),
            _sent(25, "The final analysis compares explanations across classroom tasks10."),
        ] + ref_sents

        result, receipt = await _detect_with_receipt(sents, _sections_with_refs(), refs)

        assert {xref.xref_id for xref in result} == set(range(1, 11)) | {63, 64}
        assert all(xref.tier == "flattened-superscript" for xref in result)
        incidental = [
            candidate for candidate in receipt.candidates if candidate.style == "paren-numeric"
        ]
        assert {candidate.raw for candidate in incidental} == {
            "(89)",
            "(38)",
            "(207)",
            "(111)",
            "(1)",
            "(2)",
            "(3)",
            "(4)",
        }
        assert all(
            not candidate.accepted
            and set(candidate.rejection_reasons)
            & {"parenthetical_count", "adjacent_numeric_label", "list_enumeration"}
            for candidate in incidental
        )

    async def test_ambiguous_flattened_tokens_abstain_without_document_proof(self):
        refs, ref_sents = _numbered_reference_source(70)
        sents = [
            _sent(1, "The BERT12 model was used."),
            _sent(2, "We tuned ResNet50 for images."),
            _sent(3, "Symptoms were assessed using PHQ9."),
            _sent(4, "The disease label was COVID19."),
            _sent(5, "The WHO system (GLASS)63 was used."),
            _sent(6, "Coverage was reported by UHC Watch64."),
        ] + ref_sents

        xrefs, receipt = await _detect_with_receipt(sents, _sections_with_refs(), refs)

        assert xrefs == []
        rejected = {
            candidate.raw: candidate.rejection_reasons
            for candidate in receipt.candidates
            if candidate.style == "flattened-superscript" and not candidate.accepted
        }
        assert rejected.keys() >= {"12", "50", "9", "19", "63", "64"}
        assert all(
            "ambiguous_alphanumeric_carrier" in rejected[raw]
            for raw in ("12", "50", "9", "19", "63", "64")
        )

    async def test_parenthesized_acronym_numbers_cannot_establish_flattened_style(self):
        refs, ref_sents = _numbered_reference_source(20)
        sents = [
            _sent(1, "We used the (BERT)12 model."),
            _sent(2, "Symptoms used the (PHQ)9 instrument."),
        ] + ref_sents

        xrefs, receipt = await _detect_with_receipt(sents, _sections_with_refs(), refs)

        assert xrefs == []
        acronym_candidates = [
            candidate
            for candidate in receipt.candidates
            if candidate.style == "flattened-superscript"
        ]
        assert {candidate.raw for candidate in acronym_candidates} == {"12", "9"}
        assert all(not candidate.accepted for candidate in acronym_candidates)
        assert all(
            "high_specificity_marker" not in candidate.evidence for candidate in acronym_candidates
        )

    async def test_adjacent_ambiguous_ids_cannot_corroborate_each_other(self):
        refs, ref_sents = _numbered_reference_source(20)
        sents = [
            _sent(1, "We used the (BERT)12 model."),
            _sent(2, "The Model13 was tuned."),
            _sent(3, "Symptoms used the (PHQ)9 instrument."),
            _sent(4, "The Scale10 was administered."),
        ] + ref_sents

        xrefs, receipt = await _detect_with_receipt(sents, _sections_with_refs(), refs)

        assert xrefs == []
        ambiguous = [
            candidate
            for candidate in receipt.candidates
            if candidate.style == "flattened-superscript"
        ]
        assert {candidate.raw for candidate in ambiguous} == {"9", "10", "12", "13"}
        assert all(not candidate.accepted for candidate in ambiguous)

    async def test_local_model_instrument_guards_survive_proven_flattened_style(self):
        refs, ref_sents = _numbered_reference_source(60)
        dangerous_titles = {
            9: "Patient Health Questionnaire PHQ instrument",
            12: "BERT language model",
            19: "COVID disease label surveillance",
            50: "ResNet image model architecture",
        }
        refs = [
            reference.model_copy(update={"title": dangerous_titles[reference.bib_id]})
            if reference.bib_id in dangerous_titles
            else reference
            for reference in refs
        ]
        sents = [
            _sent(1, "Prior work.1 established this."),
            _sent(2, "Evidence2,3 supports it."),
            _sent(3, "Reports4,5 agree."),
            _sent(4, "The BERT12 model was used."),
            _sent(5, "Symptoms were assessed using PHQ9."),
            _sent(6, "We tuned ResNet50 for images."),
            _sent(7, "The disease label was COVID19."),
        ] + ref_sents

        xrefs, receipt = await _detect_with_receipt(sents, _sections_with_refs(), refs)

        assert {xref.xref_id for xref in xrefs} == {1, 2, 3, 4, 5}
        rejected = [
            candidate
            for candidate in receipt.candidates
            if candidate.style == "flattened-superscript" and not candidate.accepted
        ]
        assert {candidate.raw for candidate in rejected} >= {"12", "9", "50", "19"}
        assert all(
            "model_or_instrument_context" in candidate.rejection_reasons
            for candidate in rejected
            if candidate.raw in {"12", "9", "50"}
        )
        covid = next(candidate for candidate in rejected if candidate.raw == "19")
        assert "entity_label_context" in covid.rejection_reasons

    async def test_grounded_entity_carriers_survive_proven_style_with_exact_spans(self):
        refs, ref_sents = _numbered_reference_source(70)
        supported_titles = {
            3: "National action plan on antimicrobial resistance (AMR)",
            5: "Political declaration on antimicrobial resistance (AMR)",
            6: "European roadmap for antimicrobial resistance (AMR)",
            63: "Global Antimicrobial Resistance Surveillance System (GLASS) report",
            64: "Universal Health Coverage UHC Watch report",
        }
        refs = [
            reference.model_copy(update={"title": supported_titles[reference.bib_id]})
            if reference.bib_id in supported_titles
            else reference
            for reference in refs
        ]
        body = [
            _sent(1, "Earlier reports.1,2 established the flattened style."),
            _sent(2, "Later evidence7,8 confirmed it."),
            _sent(3, "The first action plan addressed AMR3."),
            _sent(4, "The declaration renewed commitments on AMR5."),
            _sent(5, "The regional roadmap continued work on AMR6."),
            _sent(6, "The WHO system (GLASS)63 tracks resistance."),
            _sent(7, "Coverage was reported by UHC Watch64."),
            _sent(8, "The BERT12 model was used."),
            _sent(9, "Symptoms were assessed using PHQ9."),
        ]

        xrefs, receipt = await _detect_with_receipt(
            body + ref_sents,
            _sections_with_refs(),
            refs,
        )

        assert {xref.xref_id for xref in xrefs} == {1, 2, 3, 5, 6, 7, 8, 63, 64}
        flattened = [
            candidate
            for candidate in receipt.candidates
            if candidate.style == "flattened-superscript"
        ]
        text_by_id = {sentence.text_id: sentence.text for sentence in body}
        assert all(
            candidate.raw == text_by_id[candidate.text_id][candidate.start : candidate.end]
            for candidate in flattened
        )
        accepted = {candidate.raw for candidate in flattened if candidate.accepted}
        assert accepted >= {"3", "5", "6", "63", "64"}
        rejected = {
            candidate.raw: candidate.rejection_reasons
            for candidate in flattened
            if not candidate.accepted
        }
        assert "reference_carrier_mismatch" in rejected["12"]
        assert "reference_carrier_mismatch" in rejected["9"]

    async def test_printed_jats_reference_rows_establish_numeric_style_without_text_ids(self):
        refs = [_ref(index, text_id=None) for index in range(1, 6)]
        body = [
            _sent(1, "The first result (1) held."),
            _sent(2, "A replication (2) agreed."),
            _sent(3, "The final study (3) confirmed it."),
        ]
        reference_rows = [
            _sent(100 + index, f"{index}. Printed JATS reference {index}.", section_id=2)
            for index in range(1, 6)
        ]

        xrefs, receipt = await _detect_with_receipt(
            body + reference_rows,
            _sections_with_refs(),
            refs,
        )

        assert {xref.xref_id for xref in xrefs} == {1, 2, 3}
        assert receipt.style_scores["paren-numeric"] > 0

    async def test_dense_internal_bib_ids_do_not_establish_numeric_style(self):
        refs = [_ref(i) for i in range(1, 20)]
        sents = [
            _sent(1, "The first result (3) held."),
            _sent(2, "A replication (4) agreed."),
            _sent(3, "The final study (5) confirmed it."),
        ]

        xrefs, receipt = await _detect_with_receipt(sents, _body_only_sections(), refs)

        assert xrefs == []
        rejected = [
            candidate for candidate in receipt.candidates if candidate.style == "paren-numeric"
        ]
        assert len(rejected) == 3
        assert all(not candidate.accepted for candidate in rejected)
        assert all(
            "no_printed_numeric_bibliography_evidence" in candidate.rejection_reasons
            for candidate in rejected
        )

    async def test_weak_parenthetical_signal_does_not_veto_strong_flattened_detector(self):
        refs, ref_sents = _numbered_reference_source(8)
        formulas = [
            PaperSentence(
                text_id=index,
                text=rf"x_{number} = y, \quad ({number})",
                section_id=1,
                paragraph_id=index,
                is_display_formula=True,
            )
            for index, number in enumerate((1, 2, 3), start=20)
        ]
        body = [
            _sent(1, "This improves outcomes1 in adults."),
            _sent(2, "Related characteristics2,3 were stable."),
            _sent(3, "Overall effectiveness.4 remained high."),
            _sent(4, "Equation (1) defines the first constraint."),
            _sent(5, "Substituting (2) gives the next expression."),
            _sent(6, "The result follows from (3)."),
        ]

        xrefs, receipt = await _detect_with_receipt(
            formulas + body + ref_sents,
            _sections_with_refs(),
            refs,
        )

        assert {xref.xref_id for xref in xrefs} == {1, 2, 3, 4}
        assert all(xref.tier == "flattened-superscript" for xref in xrefs)
        assert any(
            candidate.style == "paren-numeric"
            and not candidate.accepted
            and "equation_tag" in candidate.rejection_reasons
            for candidate in receipt.candidates
        )

    async def test_author_year_style_no_flattened_links(self):
        """An author-year paper must produce zero flattened-tier links."""
        refs = [
            _ref(1, author="Smith, J.", year=2020),
            _ref(2, author="Jones, K.", year=2019),
            _ref(3, author="Brown, L.", year=2018),
        ]
        sents = [
            _sent(0, "Smith (2020) demonstrated the effect clearly."),
            _sent(1, "This was later confirmed by Jones (2019)."),
            _sent(2, "Brown (2018) extended these results further."),
        ]
        result = await detect_bib_xrefs(sents, _body_only_sections(), refs)
        # author-year matches resolve via Tier 2; none via the flattened tier
        assert all(x.xref_type == "bib" for x in result)
        # every link corresponds to a narrative author-year cite (year in text),
        # never a flattened "wordN" marker
        assert {x.xref_id for x in result} == {1, 2, 3}


# ---------------------------------------------------------------------------
# Detection-tier provenance
# ---------------------------------------------------------------------------


class TestTierStamping:
    async def test_numeric_bracket_tier(self):
        refs = [_ref(i) for i in range(1, 6)]
        xrefs = await detect_bib_xrefs([_sent(1, "Prior work [1] and [3-5] agrees.")], [], refs)
        assert xrefs
        assert all(x.tier == "numeric" for x in xrefs)

    async def test_superscript_tier(self):
        refs = [_ref(i) for i in range(1, 6)]
        xrefs = await detect_bib_xrefs([_sent(1, "as shown before ^{2,3} in trials.")], [], refs)
        assert xrefs
        assert all(x.tier == "numeric" for x in xrefs)

    async def test_paren_numeric_tier(self):

        refs_ids = set(range(1, 10))
        # style gate needs >= 3 distinct sentences
        sents = [
            _sent(1, "First finding (3)."),
            _sent(2, "Second finding (4, 5)."),
            _sent(3, "Third finding (6-8)."),
        ]
        xrefs = await _detect_numbered(sents, refs_ids)
        assert xrefs
        assert all(x.tier == "paren-numeric" for x in xrefs)

    async def test_flattened_superscript_tier(self):

        refs_ids = set(range(1, 10))
        sents = [
            _sent(1, "improves traits1 in adults."),
            _sent(2, "known characteristics2,3 of the sample."),
            _sent(3, "overall effectiveness.7 was high."),
        ]
        xrefs = await _detect_numbered(sents, refs_ids, sections=_sections_with_refs())
        assert xrefs
        assert all(x.tier == "flattened-superscript" for x in xrefs)

    def test_author_year_tier(self):
        refs = [_ref(1, author="Smith, J.", year=2020)]
        xrefs = detect_author_year_xrefs([_sent(1, "As shown (Smith, 2020).")], refs)
        assert xrefs
        assert all(x.tier == "author-year" for x in xrefs)

    async def test_llm_tier(self):
        class _FakeMatch:
            def __init__(self, text_id, bib_id, citation_text):
                self.text_id = text_id
                self.bib_id = bib_id
                self.citation_text = citation_text

        class _FakeLLM:
            async def resolve_citations(self, ambiguous_citations, reference_summary, file_hash):
                return [_FakeMatch(1, 1, "[Jones, 2019]")]

        refs = [_ref(1, author="Jones, A.", year=2019)]
        sents = [_sent(1, "Earlier work [Jones, 2019] found this.")]
        xrefs = await detect_bib_xrefs(sents, _sections_with_refs(), refs, llm_client=_FakeLLM())
        llm_links = [x for x in xrefs if x.tier == "llm"]
        assert llm_links and llm_links[0].xref_id == 1


class TestCitationReceipt:
    async def test_receipt_retains_accepted_and_rejected_candidates_with_honest_fractions(self):
        refs = [_ref(1), _ref(2)]
        sents = [
            _sent(1, "Prior work [1] established this."),
            _sent(2, "The interval was [.12, .44]."),
            _sent(3, "An unknown marker [9] was printed."),
        ]

        xrefs, receipt = await _detect_with_receipt(sents, _body_only_sections(), refs)

        assert type(receipt).__name__ == "CitationLinkingReceipt"
        assert [(xref.text_id, xref.xref_id) for xref in xrefs] == [(1, 1)]
        assert receipt.style_scores.keys() >= {
            "numeric",
            "paren-numeric",
            "flattened-superscript",
            "author-year",
        }
        assert receipt.resolved_candidate_fraction == 1 / 3
        assert receipt.unique_linked_bib_fraction == 0.5
        ci = next(candidate for candidate in receipt.candidates if candidate.raw == "[.12, .44]")
        assert ci.raw == sents[1].text[ci.start : ci.end]
        assert ci.accepted is False
        assert "non_citation_bracket" in ci.rejection_reasons
        assert any(
            candidate.raw == "[9]"
            and not candidate.accepted
            and "unknown_bib_id" in candidate.rejection_reasons
            for candidate in receipt.candidates
        )

    def test_author_year_candidate_raw_is_exact_multiline_source_slice(self):
        text = "Prior work (Smith,\n  2020) established this."

        _xrefs, _ambiguous, candidates = match_with_candidates(
            [_sent(1, text)],
            [_ref(1, author="Smith AB", year=2020)],
        )

        assert len(candidates) == 1
        candidate = candidates[0]
        assert (candidate.start, candidate.end) == (11, 26)
        assert candidate.raw == "(Smith,\n  2020)"
        assert candidate.raw == text[candidate.start : candidate.end]

    async def test_receipt_uses_none_for_empty_denominators(self):
        xrefs, receipt = await _detect_with_receipt(
            [_sent(1, "No citations here.")], _body_only_sections(), []
        )

        assert xrefs == []
        assert receipt.resolved_candidate_fraction is None
        assert receipt.unique_linked_bib_fraction is None

    async def test_strips_only_superscript_spans_accepted_by_receipt(self):
        refs = [_ref(1)]
        sents = [_sent(1, "Supported^{1}, but unresolved^{9} and x^{2} remain distinct.")]
        _xrefs, receipt = await _detect_with_receipt(sents, _body_only_sections(), refs)

        strip_citation_superscripts(sents, _body_only_sections(), receipt)

        assert sents[0].text == "Supported, but unresolved^{9} and x^{2} remain distinct."

    async def test_post_parse_link_seam_attaches_receipt_to_contents(self):
        from types import SimpleNamespace

        from bibr.pipeline.stages.post_parse import _link_citations

        refs = [_ref(1)]
        contents = PaperContents(
            sentences=[_sent(1, "Prior work [1] established this.")],
            sections=_body_only_sections(),
            tables=[],
            links=[],
            sections_text={},
        )

        await _link_citations(contents, SimpleNamespace(references=refs), "hash", None)

        assert type(contents.citation_receipt).__name__ == "CitationLinkingReceipt"
        assert contents.citation_receipt.unique_linked_bib_fraction == 1.0

    async def test_post_parse_link_seam_emits_typed_low_coverage_issue(self):
        from types import SimpleNamespace

        from bibr.pipeline.stages.post_parse import _link_citations

        refs = [_ref(i) for i in range(1, 11)]
        contents = PaperContents(
            sentences=[_sent(1, "Prior work [1] established this.")],
            sections=_body_only_sections(),
            tables=[],
            links=[],
            sections_text={},
        )
        issues = []

        await _link_citations(
            contents,
            SimpleNamespace(references=refs),
            "hash",
            None,
            validation_issue_sink=issues,
        )

        assert len(issues) == 1
        assert issues[0].code == "VAL_XREF_LOW_COVERAGE"
        assert issues[0].origin_stage == "post_parse"
        assert issues[0].evidence_ids == ("bib:1",)

    async def test_llm_resolution_updates_candidate_and_resolved_fraction(self):
        from bibr.schemas import CitationMatch

        refs = [
            _ref(1, author="Smith AB, Jones CD", year=2020),
            _ref(2, author="Smith CD, Brown EF", year=2020),
        ]

        class FakeLLM:
            async def resolve_citations(self, ambiguous_citations, reference_summary, file_hash):
                assert ambiguous_citations == [(1, "Smith (2020)")]
                return [CitationMatch(text_id=1, citation_text="Smith (2020)", bib_id=2)]

        xrefs, receipt = await _detect_with_receipt(
            [_sent(1, "Smith (2020) reported this result.")],
            _body_only_sections(),
            refs,
            llm_client=FakeLLM(),
        )

        assert [(xref.xref_id, xref.tier) for xref in xrefs] == [(2, "llm")]
        assert receipt.resolved_candidate_fraction == 1.0
        assert len(receipt.candidates) == 1
        resolved = receipt.candidates[0]
        assert resolved.accepted is True
        assert resolved.bib_ids == (2,)
        assert resolved.style == "author-year"
        assert "llm_resolution" in resolved.evidence
        assert resolved.rejection_reasons == ()

    async def test_repeated_llm_only_brackets_keep_distinct_receipt_spans(self):
        from bibr.schemas import CitationMatch

        class FakeLLM:
            async def resolve_citations(self, ambiguous_citations, reference_summary, file_hash):
                assert ambiguous_citations == [(1, "[Jones, 2019]")]
                return [
                    CitationMatch(
                        text_id=1,
                        citation_text="[Jones, 2019]",
                        bib_id=1,
                    )
                ]

        text = "Earlier [Jones, 2019] and again [Jones, 2019]."
        xrefs, receipt = await _detect_with_receipt(
            [_sent(1, text)],
            _body_only_sections(),
            [_ref(1, author="Jones AB", year=2019)],
            llm_client=FakeLLM(),
        )

        assert [(xref.text_id, xref.xref_id) for xref in xrefs] == [(1, 1)]
        accepted = [candidate for candidate in receipt.candidates if candidate.accepted]
        assert [(candidate.start, candidate.end) for candidate in accepted] == [(8, 21), (32, 45)]
        assert [text[candidate.start : candidate.end] for candidate in accepted] == [
            "[Jones, 2019]",
            "[Jones, 2019]",
        ]

    def test_repeated_identical_author_year_components_keep_distinct_receipt_spans(self):
        refs = [_ref(1, author="Smith AB", year=2020)]
        text = "Prior work (Smith, 2020; Smith, 2020) agrees."

        xrefs, _ambiguous, candidates = match_with_candidates([_sent(1, text)], refs)

        assert len(xrefs) == 1
        assert len(candidates) == 2
        assert [(candidate.start, candidate.end) for candidate in candidates] == [
            (12, 23),
            (25, 36),
        ]
        assert [text[candidate.start : candidate.end] for candidate in candidates] == [
            "Smith, 2020",
            "Smith, 2020",
        ]


@pytest.mark.parametrize("bib_ids", [set(range(1, 43)), set(), {5, 6, 7}, {1, 2, 40}])
async def test_internal_bib_ids_do_not_establish_printed_numeric_style(bib_ids):
    marker = min(bib_ids, default=1)
    sents = [_sent(i, f"Prior work ({marker}) supports this.") for i in range(3)]
    xrefs, receipt = await _detect_with_receipt(
        sents, _body_only_sections(), [_ref(i) for i in bib_ids]
    )
    assert xrefs == []
    reason = "no_printed_numeric_bibliography_evidence" if bib_ids else "unknown_bib_id"
    assert any(reason in candidate.rejection_reasons for candidate in receipt.candidates)


@pytest.mark.parametrize(
    ("style", "text", "expected"),
    [
        ("paren-numeric", "the result (3) shows", [("(3)", [3])]),
        ("paren-numeric", "both models (7, 8) agree", [("(7, 8)", [7, 8])]),
        ("paren-numeric", "studies (6-8) confirm", [("(6-8)", [6, 7, 8])]),
        ("paren-numeric", "published (2018) recently", []),
        ("paren-numeric", "the value (0.1) is small", []),
        ("paren-numeric", "range (5=3)", []),
        ("paren-numeric", "error (3)% here", []),
        ("paren-numeric", "computed 5(3) times", []),
        ("paren-numeric", "see (3, 99) refs", []),
        ("paren-numeric", "see equation (1)", [("(1)", [1])]),
        ("flattened-superscript", "similar physical traits1, but", [("1", [1])]),
        ("flattened-superscript", "own characteristics2,3.", [("2,3", [2, 3])]),
        ("flattened-superscript", "effectiveness.6–8 remain", [("6–8", [6, 7, 8])]),
        ("flattened-superscript", "cardiac rehabilitation,1 diabetes", [("1", [1])]),
        ("flattened-superscript", "more effectiveness.7,10,11 and", [("7,10,11", [7, 10, 11])]),
        ("flattened-superscript", "dataset has examples60,000 here", []),
        ("flattened-superscript", "a grid28x28 image", []),
        ("flattened-superscript", "accuracy5% overall", []),
        ("flattened-superscript", "a rate3.5 fold", []),
        ("flattened-superscript", "used L2 regularization", []),
        ("flattened-superscript", "postcode G12 here", []),
        ("flattened-superscript", "traits99 here", []),
        ("flattened-superscript", "we had 60,000 examples", []),
        ("flattened-superscript", "images are 28x28 pixels", []),
        ("flattened-superscript", "see Fig. 3 for details", []),
    ],
)
async def test_numeric_marker_cases_through_active_linker(style, text, expected):
    # Recurrence and printed references establish the style, so these cases
    # isolate each marker's local shape/context without using retired matchers.
    sents = [_sent(i, text) for i in range(3)]
    if style == "flattened-superscript":
        sents.extend(_sent(100 + i, "Additional evidence.2,3 supports this.") for i in range(3))
    xrefs = await _detect_numbered(sents, set(range(1, 43)))
    matches = {}
    for xref in xrefs:
        if xref.text_id == 0 and xref.tier == style:
            matches.setdefault(xref.contents, []).append(xref.xref_id)
    assert list(matches.items()) == expected
