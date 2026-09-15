"""Lettered-appendix hierarchy repair (chunk 3).

Covers the numbered-prefix space collapse for letters, and
``repair_appendix_hierarchy`` / the TOC anchor guard inside
``assign_hierarchy_from_top_level``.
"""

from __future__ import annotations

import pytest

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

    @pytest.mark.parametrize("title_type", [CanonicalSection.TITLE, CanonicalSection.UNKNOWN])
    @pytest.mark.parametrize("sidebar", ["CITATION", "How to cite this article", "Cite this paper"])
    def test_early_article_title_after_citation_sidebar_is_not_an_appendix(
        self, title_type, sidebar
    ):
        secs = [
            _sec(1, sidebar, CanonicalSection.REFERENCES),
            _sec(2, "A protocol for studying seasonal bird migration", title_type, level=1),
            _sec(3, "Abstract", CanonicalSection.ABSTRACT),
            _sec(4, "Introduction", CanonicalSection.INTRODUCTION),
            _sec(5, "Method", CanonicalSection.METHODS),
            _sec(6, "References", CanonicalSection.REFERENCES),
            _sec(7, "A Additional analyses"),
        ]

        handled = repair_appendix_hierarchy(secs)

        assert 2 not in handled
        assert secs[1].section_type == title_type
        assert secs[1].classification_source != "appendix_repair"
        assert secs[-1].section_type == CanonicalSection.APPENDIX

    def test_early_title_survives_a_nonstandard_false_reference_anchor(self):
        secs = [
            _sec(1, "Article information", CanonicalSection.REFERENCES),
            _sec(2, "A protocol for studying seasonal bird migration", CanonicalSection.TITLE),
            _sec(3, "Abstract", CanonicalSection.ABSTRACT),
            _sec(4, "Introduction", CanonicalSection.INTRODUCTION),
            _sec(5, "Method", CanonicalSection.METHODS),
        ]

        assert 2 not in repair_appendix_hierarchy(secs)
        assert secs[1].section_type == CanonicalSection.TITLE

    @pytest.mark.parametrize("heading", ["Appendix A", "A Additional analyses"])
    def test_late_appendix_mistyped_title_still_gets_repaired(self, heading):
        secs = [
            _sec(1, "Introduction", CanonicalSection.INTRODUCTION),
            _sec(2, "Method", CanonicalSection.METHODS),
            _sec(3, "References", CanonicalSection.REFERENCES),
            _sec(4, heading, CanonicalSection.TITLE),
            _sec(5, "A.1 Details"),
        ]

        handled = repair_appendix_hierarchy(secs)

        assert {4, 5}.issubset(handled)
        assert secs[3].section_type == CanonicalSection.APPENDIX
        assert secs[4].parent_section_id == 4

    def test_late_lettered_appendix_after_unclassified_body_still_repairs(self):
        secs = [
            _sec(1, "Study context"),
            _sec(2, "Sampling strategy"),
            _sec(3, "Statistical assessment"),
            _sec(4, "References", CanonicalSection.REFERENCES),
            _sec(5, "A Additional analyses", CanonicalSection.TITLE),
        ]

        assert 5 in repair_appendix_hierarchy(secs)
        assert secs[-1].section_type == CanonicalSection.APPENDIX

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
