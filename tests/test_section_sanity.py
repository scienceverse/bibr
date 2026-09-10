"""Tests for enforce_section_sanity — positional demotion of implausible types."""

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


def test_empty_list_noop():
    enforce_section_sanity([])
