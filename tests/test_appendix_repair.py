"""Lettered-appendix hierarchy repair (chunk 3).

Covers the numbered-prefix space collapse for letters, and
``repair_appendix_hierarchy`` / the TOC anchor guard inside
``assign_hierarchy_from_top_level``.
"""

from __future__ import annotations

from bibr.paper_contents import CanonicalSection, PaperSection
from bibr.structure.section_tree import (
    assign_hierarchy_from_top_level,
    repair_appendix_hierarchy,
)
from bibr.structure.text_repair import collapse_numbered_prefix_spaces


def _sec(
    sid: int,
    header: str,
    type_: CanonicalSection = CanonicalSection.UNKNOWN,
    level: int = 2,
    parent: int | None = 0,
) -> PaperSection:
    sec = PaperSection(section_id=sid, header=header, level=level, parent_section_id=parent)
    sec.section_type = type_
    return sec


# ---------------------------------------------------------------------------
# (i) collapse numbered/lettered prefix spaces
# ---------------------------------------------------------------------------


class TestCollapsePrefixSpaces:
    def test_digit_regressions(self):
        assert collapse_numbered_prefix_spaces("3. 1 Encoder") == "3.1 Encoder"
        assert collapse_numbered_prefix_spaces("3. 2. 1 Foo") == "3.2.1 Foo"
        assert collapse_numbered_prefix_spaces("3.1 Encoder") == "3.1 Encoder"
        assert collapse_numbered_prefix_spaces("3. Background") == "3. Background"

    def test_letter_prefix_collapse(self):
        assert collapse_numbered_prefix_spaces("A. 1 Proof") == "A.1 Proof"
        assert collapse_numbered_prefix_spaces("B. 2. 1 Lemma") == "B.2.1 Lemma"

    def test_letter_word_untouched(self):
        # A single letter followed by a word (no dotted-number run) is unchanged.
        assert collapse_numbered_prefix_spaces("A. Additional Results") == "A. Additional Results"


# ---------------------------------------------------------------------------
# (iii) repair pass — mis-nested appendices become top-level siblings
# ---------------------------------------------------------------------------


def _by_id(secs):
    return {s.section_id: s for s in secs}


class TestRepairAppendixHierarchy:
    def test_bert_letter_run_after_references_are_siblings(self):
        # 1..5 numbered body, References, then A/B/C appendices.
        secs = [
            _sec(1, "1 Introduction", CanonicalSection.INTRODUCTION),
            _sec(2, "2 Related Work", CanonicalSection.INTRODUCTION),
            _sec(3, "3 Method", CanonicalSection.METHODS),
            _sec(4, "4 Experiments", CanonicalSection.RESULTS),
            _sec(5, "5 Conclusion", CanonicalSection.DISCUSSION),
            _sec(6, "References", CanonicalSection.REFERENCES),
            _sec(7, "A Additional Experiments"),
            _sec(8, "B Hyperparameters"),
            _sec(9, "C Ablations"),
        ]
        assign_hierarchy_from_top_level(secs)
        by = _by_id(secs)
        for sid in (7, 8, 9):
            assert by[sid].level == 1, sid
            assert by[sid].parent_section_id == 0, sid

    def test_resnet_appendices_not_under_references(self):
        secs = [
            _sec(1, "1 Introduction", CanonicalSection.INTRODUCTION),
            _sec(2, "2 Method", CanonicalSection.METHODS),
            _sec(3, "3 Discussion", CanonicalSection.DISCUSSION),
            _sec(4, "References", CanonicalSection.REFERENCES),
            _sec(5, "A Network Details"),
            _sec(6, "B Extra Results"),
            _sec(7, "C Proofs"),
        ]
        ref_id = 4
        assign_hierarchy_from_top_level(secs)
        by = _by_id(secs)
        for sid in (5, 6, 7):
            assert by[sid].parent_section_id != ref_id, sid
            assert by[sid].parent_section_id == 0, sid

    def test_primes_dotted_children_nest_under_single_root(self):
        secs = [
            _sec(1, "1 Introduction", CanonicalSection.INTRODUCTION),
            _sec(2, "2 Results", CanonicalSection.RESULTS),
            _sec(3, "References", CanonicalSection.REFERENCES),
            _sec(4, "A Proofs"),
            _sec(5, "A.1 First Lemma"),
            _sec(6, "A.2 Second Lemma"),
        ]
        assign_hierarchy_from_top_level(secs)
        by = _by_id(secs)
        assert by[4].level == 1 and by[4].parent_section_id == 0
        # A.1 / A.2 are siblings, both nested under root A (section 4).
        assert by[5].parent_section_id == 4
        assert by[6].parent_section_id == 4
        assert by[5].level == 2 and by[6].level == 2

    def test_deepseek_dotted_child_and_letter_sibling(self):
        # Appendices starting at B (not A): B with a B.2 child, then C sibling.
        secs = [
            _sec(1, "1 Introduction", CanonicalSection.INTRODUCTION),
            _sec(2, "2 Method", CanonicalSection.METHODS),
            _sec(3, "References", CanonicalSection.REFERENCES),
            _sec(4, "B Model Details"),
            _sec(5, "B.2 Training Config"),
            _sec(6, "C Evaluation Protocol"),
        ]
        assign_hierarchy_from_top_level(secs)
        by = _by_id(secs)
        assert by[4].level == 1 and by[4].parent_section_id == 0
        assert by[5].parent_section_id == 4  # B.2 under B
        assert by[6].level == 1 and by[6].parent_section_id == 0  # C sibling of B

    def test_explicit_appendix_marker_anchors_single_letter(self):
        secs = [
            _sec(1, "1 Introduction", CanonicalSection.INTRODUCTION),
            _sec(2, "2 Method", CanonicalSection.METHODS),
            _sec(3, "Appendix A"),
            _sec(4, "A.1 Details"),
        ]
        handled = repair_appendix_hierarchy(secs)
        assert 3 in handled
        by = _by_id(secs)
        assert by[3].level == 1 and by[3].parent_section_id == 0
        assert by[4].parent_section_id == 3

    def test_early_a_framework_heading_untouched(self):
        # "A Framework for X" early in the doc with no sibling run, no anchor.
        secs = [
            _sec(1, "A Framework for Reasoning", level=1, parent=0),
            _sec(2, "Introduction", CanonicalSection.INTRODUCTION),
            _sec(3, "Method", CanonicalSection.METHODS),
            _sec(4, "Results", CanonicalSection.RESULTS),
            _sec(5, "Discussion", CanonicalSection.DISCUSSION),
            _sec(6, "References", CanonicalSection.REFERENCES),
        ]
        handled = repair_appendix_hierarchy(secs)
        assert 1 not in handled

    def test_no_appendices_is_noop(self):
        secs = [
            _sec(1, "Method", CanonicalSection.METHODS),
            _sec(2, "Results", CanonicalSection.RESULTS),
        ]
        assert repair_appendix_hierarchy(secs) == set()

    def test_unknown_roots_typed_appendix(self):
        secs = [
            _sec(1, "1 Introduction", CanonicalSection.INTRODUCTION),
            _sec(2, "2 Method", CanonicalSection.METHODS),
            _sec(3, "References", CanonicalSection.REFERENCES),
            _sec(4, "A Additional Results"),
            _sec(5, "B Proofs"),
        ]
        repair_appendix_hierarchy(secs)
        by = _by_id(secs)
        assert by[4].section_type == CanonicalSection.APPENDIX
        assert by[5].section_type == CanonicalSection.APPENDIX
        assert by[4].classification_source == "appendix_repair"
        assert by[5].classification_source == "appendix_repair"

    def test_exact_alias_root_keeps_specific_type(self):
        # A root carrying a type from an EXACT alias hit keeps it; a weak
        # substring/prior type (source != exact_alias) is overridden to APPENDIX.
        secs = [
            _sec(1, "1 Introduction", CanonicalSection.INTRODUCTION),
            _sec(2, "References", CanonicalSection.REFERENCES),
            _sec(3, "A Extra Data"),  # UNKNOWN → APPENDIX
            _sec(4, "B Conclusions", CanonicalSection.DISCUSSION),  # substring → overridden
        ]
        secs[3].classification_source = "substring_alias"
        # An exact-alias "Appendix" root keeps APPENDIX and stays exact-sourced.
        secs.append(_sec(5, "Appendix", CanonicalSection.APPENDIX))
        secs[4].classification_source = "exact_alias"
        repair_appendix_hierarchy(secs)
        by = _by_id(secs)
        assert by[3].section_type == CanonicalSection.APPENDIX
        assert by[4].section_type == CanonicalSection.APPENDIX  # DISCUSSION overridden
        assert by[5].section_type == CanonicalSection.APPENDIX
        assert by[5].classification_source == "exact_alias"  # untouched


class TestAppendixNeedsARealAnchor:
    """A lettered run is re-pinned and re-typed only after an "Appendix"
    marker or after the reference list that ends the body."""

    def test_ieee_lettered_subsections_and_roman_conclusion_are_not_appendices(self):
        import asyncio

        from bibr.paper_contents import PaperContents
        from bibr.pipeline.stages.post_parse import _classify_sections

        heads = [
            "Deep Nets for X",
            "Abstract",
            "I. INTRODUCTION",
            "II. RELATED WORK",
            "A. Object Detection",
            "B. Segmentation",
            "III. METHOD",
            "A. Architecture",
            "B. Loss Function",
            "IV. EXPERIMENTS",
            "A. Datasets",
            "B. Implementation Details",
            "C. Results",
            "D. Ablation Study",
            "V. CONCLUSION",
            "ACKNOWLEDGMENT",
            "REFERENCES",
        ]
        secs = [
            PaperSection(
                section_id=i,
                header=h,
                level=0 if i == 0 else 1,
                parent_section_id=None if i == 0 else 0,
            )
            for i, h in enumerate(heads)
        ]
        contents = PaperContents(
            sentences=[],
            sections=secs,
            tables=[],
            links=[],
            sections_text={},
            detected_title="Deep Nets for X",
        )

        asyncio.run(_classify_sections(contents, [], True, None))

        by = _by_id(secs)
        assert [s.section_id for s in secs if s.classification_source == "appendix_repair"] == []
        assert [s.section_id for s in secs if s.section_type == CanonicalSection.APPENDIX] == []
        assert (by[12].section_type, by[12].classification_source) == (
            CanonicalSection.RESULTS,
            "substring_alias",
        )
        assert [(by[sid].level, by[sid].parent_section_id) for sid in (10, 12, 13)] == [
            (2, 9),
            (2, 9),
            (2, 9),
        ]
        assert (by[14].section_type, by[14].level, by[14].parent_section_id) == (
            CanonicalSection.DISCUSSION,
            1,
            0,
        )
        # Known limitation of the positional hierarchy, not of this repair:
        # "B. Implementation Details" reads as METHODS by keyword and folds
        # under the first METHODS heading, "III. METHOD", not under
        # "IV. EXPERIMENTS" where it is printed.
        assert (by[11].section_type, by[11].level, by[11].parent_section_id) == (
            CanonicalSection.METHODS,
            2,
            6,
        )

    def test_lettered_subsections_of_results_and_discussion_untouched(self):
        secs = [
            _sec(1, "INTRODUCTION", CanonicalSection.INTRODUCTION, level=1),
            _sec(2, "METHOD", CanonicalSection.METHODS, level=1),
            _sec(3, "RESULTS AND DISCUSSION", CanonicalSection.RESULTS, level=1),
            _sec(4, "A. State Defense Regulation"),
            _sec(5, "B. The Ideal Concept of State Defense"),
            _sec(6, "CONCLUSION", CanonicalSection.DISCUSSION, level=1),
            _sec(7, "REFERENCES", CanonicalSection.REFERENCES, level=1),
        ]
        secs[3].classification_source = "llm"
        secs[4].classification_source = "llm"

        assert repair_appendix_hierarchy(secs) == set()
        by = _by_id(secs)
        assert [(by[sid].section_type, by[sid].level) for sid in (4, 5)] == [
            (CanonicalSection.UNKNOWN, 2),
            (CanonicalSection.UNKNOWN, 2),
        ]

    def test_appendix_marker_anchors_only_the_headings_after_it(self):
        secs = [
            _sec(1, "I. INTRODUCTION", CanonicalSection.INTRODUCTION, level=1),
            _sec(2, "IV. EXPERIMENTS", CanonicalSection.RESULTS, level=1),
            _sec(3, "A. Datasets"),
            _sec(4, "B. Results", CanonicalSection.RESULTS),
            _sec(5, "V. CONCLUSION", CanonicalSection.DISCUSSION, level=1),
            _sec(6, "APPENDIX A PROOF OF LEMMA 1"),
            _sec(7, "A.1 Preliminaries"),
            _sec(8, "REFERENCES", CanonicalSection.REFERENCES, level=1),
        ]
        for sec in secs:
            sec.classification_source = "substring_alias"

        assert repair_appendix_hierarchy(secs) == {6, 7}
        assert [(s.section_id, s.section_type) for s in secs] == [
            (1, CanonicalSection.INTRODUCTION),
            (2, CanonicalSection.RESULTS),
            (3, CanonicalSection.UNKNOWN),
            (4, CanonicalSection.RESULTS),
            (5, CanonicalSection.DISCUSSION),
            (6, CanonicalSection.APPENDIX),
            (7, CanonicalSection.UNKNOWN),
            (8, CanonicalSection.REFERENCES),
        ]
        assert _by_id(secs)[7].parent_section_id == 6

    def test_pre_body_citation_panel_is_not_a_references_anchor(self):
        # Frontiers prints a "Citation" box above the title; the classifier can
        # type it REFERENCES. The title that follows ("A protocol for ...")
        # reads as root letter A but is no appendix.
        secs = [
            _sec(1, "OPEN ACCESS"),
            _sec(2, "CITATION", CanonicalSection.REFERENCES, level=1),
            _sec(3, "COPYRIGHT"),
            _sec(4, "A protocol for mobilising novel finance models", CanonicalSection.TITLE),
            _sec(5, "Abstract", CanonicalSection.ABSTRACT, level=1),
            _sec(6, "Background", CanonicalSection.INTRODUCTION, level=1),
            _sec(7, "Methods", CanonicalSection.METHODS, level=1),
            _sec(8, "Discussion", CanonicalSection.DISCUSSION, level=1),
            _sec(9, "References", CanonicalSection.REFERENCES, level=1),
        ]
        secs[3].classification_source = "title"

        assert repair_appendix_hierarchy(secs) == set()
        assert (secs[3].section_type, secs[3].classification_source) == (
            CanonicalSection.TITLE,
            "title",
        )

    def test_references_anchor_without_typed_body_sections(self):
        # No heading typed as IMRaD body: the first reference list still anchors.
        secs = [
            _sec(1, "The Problem", level=1),
            _sec(2, "Our Argument", level=1),
            _sec(3, "References", CanonicalSection.REFERENCES, level=1),
            _sec(4, "A Proofs"),
            _sec(5, "B Extra Tables"),
        ]

        assert repair_appendix_hierarchy(secs) == {4, 5}
        assert [(s.section_type, s.level, s.parent_section_id) for s in secs[3:]] == [
            (CanonicalSection.APPENDIX, 1, 0),
            (CanonicalSection.APPENDIX, 1, 0),
        ]

    def test_references_anchor_when_body_words_appear_only_after_it(self):
        # An essay whose headings name no IMRaD part: the first heading typed
        # as body is an appendix after the references ("A. Methods of the
        # vignette survey"), and the reference list still anchors the run.
        secs = [
            _sec(1, "The Problem", level=1),
            _sec(2, "Three Replies", level=1),
            _sec(3, "References", CanonicalSection.REFERENCES, level=1),
            _sec(4, "A. Methods of the vignette survey", CanonicalSection.METHODS),
            _sec(5, "B. Robustness checks"),
        ]
        secs[3].classification_source = "substring_alias"

        assert repair_appendix_hierarchy(secs) == {4, 5}
        assert [(s.section_type, s.level, s.parent_section_id) for s in secs[3:]] == [
            (CanonicalSection.APPENDIX, 1, 0),
            (CanonicalSection.APPENDIX, 1, 0),
        ]

    def test_references_anchor_when_methods_follow_them(self):
        # Methods printed after the reference list, then lettered
        # supplementary notes: no body section comes before the references.
        secs = [
            _sec(1, "Main", level=1),
            _sec(2, "References", CanonicalSection.REFERENCES, level=1),
            _sec(3, "Methods", CanonicalSection.METHODS, level=1),
            _sec(4, "A. Supplementary Note"),
            _sec(5, "B. Supplementary Tables"),
        ]
        secs[2].classification_source = "exact_alias"

        assert repair_appendix_hierarchy(secs) == {4, 5}
        assert [s.section_type for s in secs] == [
            CanonicalSection.UNKNOWN,
            CanonicalSection.REFERENCES,
            CanonicalSection.METHODS,
            CanonicalSection.APPENDIX,
            CanonicalSection.APPENDIX,
        ]


class TestTocAnchorGuard:
    def test_contents_never_becomes_anchor(self):
        # A "CONTENTS" TOC heading must not capture the following headings.
        secs = [
            _sec(1, "Contents", level=1, parent=0),
            _sec(2, "Some Chapter", CanonicalSection.UNKNOWN),
            _sec(3, "Another Chapter", CanonicalSection.UNKNOWN),
        ]
        assign_hierarchy_from_top_level(secs)
        by = _by_id(secs)
        # CONTENTS stays a bare top-level orphan.
        assert by[1].level == 1 and by[1].parent_section_id == 0
        # Following headings do NOT fold under CONTENTS.
        assert by[2].parent_section_id != 1
        assert by[3].parent_section_id != 1
