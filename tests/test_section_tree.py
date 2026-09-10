"""Tests for bibr.structure.section_tree."""

from bibr.paper_contents import PaperSection
from bibr.structure.section_tree import build_section_tree, infer_level_from_numbering


class TestInferLevelFromNumbering:
    def test_single_number(self):
        assert infer_level_from_numbering("1 Introduction") == 1
        assert infer_level_from_numbering("3. Methods") == 1

    def test_dotted_numbers(self):
        assert infer_level_from_numbering("1.1 Methods") == 2
        assert infer_level_from_numbering("2.3.1 Subsection") == 3
        assert infer_level_from_numbering("1.2.3.4 Deep") == 4

    def test_no_number(self):
        assert infer_level_from_numbering("Methods") is None
        # Bare single letter is ambiguous with the article "A" — stays None.
        assert infer_level_from_numbering("A Framework for Learning") is None
        assert infer_level_from_numbering("A") is None

    def test_empty_or_invalid(self):
        assert infer_level_from_numbering("") is None
        assert infer_level_from_numbering("   ") is None

    def test_caps_at_six(self):
        # 7-deep numbering capped at 6
        assert infer_level_from_numbering("1.2.3.4.5.6.7 Too deep") == 6

    def test_trailing_dot(self):
        assert infer_level_from_numbering("1.1.") == 2

    def test_colon_separator(self):
        assert infer_level_from_numbering("2.1: Background") == 2

    def test_appendix_headings_are_depth_one(self):
        assert infer_level_from_numbering("Appendix") == 1
        assert infer_level_from_numbering("Appendix A") == 1
        assert infer_level_from_numbering("APPENDIX C") == 1
        assert infer_level_from_numbering("Appendices") == 1

    def test_dotted_letter_subheadings(self):
        assert infer_level_from_numbering("A.1 Proof") == 2
        assert infer_level_from_numbering("B.2.1 Lemma") == 3

    def test_digit_regressions_still_hold(self):
        assert infer_level_from_numbering("1") == 1
        assert infer_level_from_numbering("1.1") == 2


class TestBuildSectionTree:
    def _section(self, sid: int, parent: int | None, level: int = 1) -> PaperSection:
        return PaperSection(
            section_id=sid,
            header=f"S{sid}",
            level=level,
            parent_section_id=parent,
        )

    def test_flat_list_under_root(self):
        sections = [
            self._section(0, None, level=0),  # Root
            self._section(1, 0, level=1),
            self._section(2, 0, level=1),
        ]
        build_section_tree(sections)
        assert len(sections[0].children) == 2
        assert sections[0].children[0].section_id == 1
        assert sections[0].children[1].section_id == 2
        assert sections[1].children == []

    def test_nested(self):
        sections = [
            self._section(0, None, level=0),
            self._section(1, 0, level=1),  # parent
            self._section(2, 1, level=2),  # child of 1
            self._section(3, 1, level=2),  # child of 1
            self._section(4, 2, level=3),  # grandchild
        ]
        build_section_tree(sections)
        assert len(sections[0].children) == 1
        assert sections[0].children[0].section_id == 1
        assert {c.section_id for c in sections[1].children} == {2, 3}
        assert sections[2].children[0].section_id == 4

    def test_idempotent(self):
        sections = [
            self._section(0, None, level=0),
            self._section(1, 0, level=1),
            self._section(2, 1, level=2),
        ]
        build_section_tree(sections)
        first_children = list(sections[0].children)
        build_section_tree(sections)
        assert sections[0].children == first_children
        assert len(sections[1].children) == 1

    def test_orphan_parent_id(self):
        # parent_section_id points to a missing section — silently ignored
        sections = [
            self._section(0, None, level=0),
            self._section(1, 99, level=1),  # 99 doesn't exist
        ]
        build_section_tree(sections)
        assert sections[0].children == []
        assert sections[1].children == []

    def test_self_reference_ignored(self):
        sections = [self._section(0, None, level=0), self._section(1, 1, level=1)]
        build_section_tree(sections)
        assert sections[1].children == []
