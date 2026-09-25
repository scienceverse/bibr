"""Tests for enforce_section_sanity — positional demotion of implausible types."""

import pytest

from bibr.paper import enforce_section_sanity
from bibr.paper_contents import CanonicalSection, PaperSection


def _sec(i: int, header: str, stype: CanonicalSection) -> PaperSection:
    return PaperSection(
        section_id=i,
        header=header,
        level=1,
        parent_section_id=0,
        section_type=stype,
        classification_score=0.9,
    )


class TestAbstractSanity:
    def test_abstract_after_methods_demoted(self):
        sections = [
            _sec(1, "Introduction", CanonicalSection.INTRODUCTION),
            _sec(2, "Methods", CanonicalSection.METHODS),
            _sec(3, "Results", CanonicalSection.RESULTS),
            _sec(4, "Summary", CanonicalSection.ABSTRACT),
        ]
        enforce_section_sanity(sections)
        assert sections[3].section_type == CanonicalSection.UNKNOWN
        assert sections[3].classification_score == 0.0

    def test_front_matter_abstract_kept(self):
        sections = [
            _sec(1, "Abstract", CanonicalSection.ABSTRACT),
            _sec(2, "Introduction", CanonicalSection.INTRODUCTION),
            _sec(3, "Methods", CanonicalSection.METHODS),
        ]
        enforce_section_sanity(sections)
        assert sections[0].section_type == CanonicalSection.ABSTRACT

    def test_abstract_without_body_sections_kept(self):
        # No METHODS/RESULTS anchor → nothing to judge position against.
        sections = [
            _sec(1, "Introduction", CanonicalSection.INTRODUCTION),
            _sec(2, "Summary", CanonicalSection.ABSTRACT),
        ]
        enforce_section_sanity(sections)
        assert sections[1].section_type == CanonicalSection.ABSTRACT


class TestReferencesSanity:
    def test_early_references_with_core_body_after_demoted(self):
        sections = [
            _sec(1, "Introduction", CanonicalSection.INTRODUCTION),
            _sec(2, "Key References", CanonicalSection.REFERENCES),
            _sec(3, "Methods", CanonicalSection.METHODS),
            _sec(4, "Results", CanonicalSection.RESULTS),
            _sec(5, "Discussion", CanonicalSection.DISCUSSION),
            _sec(6, "References", CanonicalSection.UNKNOWN),
        ]
        enforce_section_sanity(sections)
        assert sections[1].section_type == CanonicalSection.UNKNOWN

    def test_terminal_references_kept(self):
        sections = [
            _sec(1, "Introduction", CanonicalSection.INTRODUCTION),
            _sec(2, "Methods", CanonicalSection.METHODS),
            _sec(3, "Discussion", CanonicalSection.DISCUSSION),
            _sec(4, "References", CanonicalSection.REFERENCES),
        ]
        enforce_section_sanity(sections)
        assert sections[3].section_type == CanonicalSection.REFERENCES

    def test_references_followed_only_by_appendix_kept(self):
        sections = [
            _sec(1, "Introduction", CanonicalSection.INTRODUCTION),
            _sec(2, "Discussion", CanonicalSection.DISCUSSION),
            _sec(3, "References", CanonicalSection.REFERENCES),
            _sec(4, "Appendix", CanonicalSection.ENDNOTE),
            _sec(5, "Supplementary Tables", CanonicalSection.TABLE),
        ]
        enforce_section_sanity(sections)
        assert sections[2].section_type == CanonicalSection.REFERENCES


class TestPositionsCountBodySectionsOnly:
    """The root and the tail figure/table/footnote sections take no position."""

    @staticmethod
    def _paper(n_floats: int) -> list[PaperSection]:
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            _sec(1, "Introduction", CanonicalSection.INTRODUCTION),
            _sec(2, "Methods", CanonicalSection.METHODS),
            _sec(3, "Results", CanonicalSection.RESULTS),
            _sec(4, "Discussion", CanonicalSection.DISCUSSION),
            _sec(5, "References", CanonicalSection.REFERENCES),
            _sec(6, "Supplementary Methods", CanonicalSection.METHODS),
        ]
        sections += [
            PaperSection(
                section_id=100 + k,
                header=f"Figure {k + 1}",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.FIGURE,
                classification_score=1.0,
                synthetic_kind="figure",
            )
            for k in range(n_floats)
        ]
        return sections

    @pytest.mark.parametrize("n_floats", [0, 10])
    def test_float_count_does_not_move_the_first_half(self, n_floats):
        sections = self._paper(n_floats)
        enforce_section_sanity(sections)
        references = sections[5]
        assert (references.header, references.section_type, references.classification_score) == (
            "References",
            CanonicalSection.REFERENCES,
            0.9,
        )

    @staticmethod
    def _nature_letter(*after: PaperSection) -> list[PaperSection]:
        # Methods printed after the reference list.
        return [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            _sec(1, "Paper Title", CanonicalSection.TITLE),
            _sec(2, "Results", CanonicalSection.RESULTS),
            _sec(3, "References", CanonicalSection.REFERENCES),
            *after,
        ]

    def test_references_with_as_many_body_sections_before_as_after_stay(self):
        sections = self._nature_letter(
            _sec(4, "Methods", CanonicalSection.METHODS),
            _sec(5, "Data availability", CanonicalSection.OPEN_DATA),
        )
        enforce_section_sanity(sections)
        assert (sections[3].section_type, sections[3].classification_score) == (
            CanonicalSection.REFERENCES,
            0.9,
        )

    def test_references_with_more_body_sections_after_are_reset(self):
        sections = self._nature_letter(
            _sec(4, "Methods", CanonicalSection.METHODS),
            _sec(5, "Discussion", CanonicalSection.DISCUSSION),
            _sec(6, "Data availability", CanonicalSection.OPEN_DATA),
        )
        enforce_section_sanity(sections)
        assert (sections[3].section_type, sections[3].classification_score) == (
            CanonicalSection.UNKNOWN,
            0.0,
        )


def test_empty_list_noop():
    enforce_section_sanity([])
