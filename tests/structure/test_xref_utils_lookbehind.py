"""Letter-lookbehind regression tests for table, figure, supplement, and section xrefs.

Author names, suffix words, URLs, and journal names must not yield substring matches. OCR-glued digit prefixes and genuine subsection references remain eligible."""

import pandas as pd

from bibr.paper_contents import PaperFigure, PaperSentence, PaperTable
from bibr.structure.xref_utils import detect_xrefs


def _sent(text: str) -> list[PaperSentence]:
    return [PaperSentence(text_id=1, text=text, section_id=1, paragraph_id=1)]


def _tables(*ids: int) -> list[PaperTable]:
    return [
        PaperTable(table_id=i, df=pd.DataFrame(), tbl_html="<table/>", section_id=1, label=str(i))
        for i in ids
    ]


def _figures(*ids: int) -> list[PaperFigure]:
    return [
        PaperFigure(figure_id=i, section_id=1, image_b64=None, caption=None, label=str(i))
        for i in ids
    ]


class TestTableXrefFalsePositives:
    def test_stable_with_glued_marker_not_matched(self):
        # "stable11" (word + glued footnote marker, econ 2007-11) contains
        # "table11" — must not become a table xref even when table 11 exists.
        xrefs = detect_xrefs(_sent("Input: P a stable11 distribution."), _tables(11), [])
        assert xrefs == []

    def test_author_name_suffix_not_matched(self):
        # "Constable 2014" — id validation would normally drop "table 2014",
        # but the regex itself must not match inside a word.
        xrefs = detect_xrefs(
            _sent("documented since the birth of a child (Constable 2014)."),
            _tables(2014),
            [],
        )
        assert xrefs == []

    def test_url_path_segment_not_matched(self):
        # "ditctab20121en.pdf" (UNCTAD URL, econ 2019-10) contains "tab20121"
        xrefs = detect_xrefs(
            _sent("See unctad.org/en/PublicationsLibrary/ditctab20121en.pdf for data."),
            _tables(20121),
            [],
        )
        assert xrefs == []


class TestFigureXrefFalsePositives:
    def test_config_not_matched(self):
        # "config 3" contains "fig 3"
        xrefs = detect_xrefs(_sent("We used config 3 for all runs."), [], _figures(3))
        assert xrefs == []


class TestSupplementaryXrefFalsePositives:
    def test_unstable_s_not_matched(self):
        # word ending in "table" + "S<digit>" — supp xrefs are unvalidated,
        # so the regex is the only line of defense.
        xrefs = detect_xrefs(_sent("The system stays unstable S1 excluded."), [], [])
        assert xrefs == []

    def test_config_s_not_matched(self):
        # "config S2" contains "fig S2"
        xrefs = detect_xrefs(_sent("Results for config S2 are shown."), [], [])
        assert xrefs == []


class TestSectionXrefFalsePositives:
    def test_journal_name_not_matched(self):
        # "Intersections 4: 26–50" (journal + volume in a reference line,
        # mdpi socsci) contains "sections 4"
        xrefs = detect_xrefs(_sent("Speech in Context. Intersections 4: 26–50."), [], [])
        assert xrefs == []


class TestSectionXrefSubsectionRecall:
    def test_subsection_still_matched(self):
        # "Subsection 2.4" is a genuine section reference — a bare letter
        # lookbehind would kill it, so "Sub" is allowed explicitly.
        xrefs = detect_xrefs(_sent("Recessions are analyzed in Subsection 2.4."), [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "section"
        assert xrefs[0].xref_id == 2
        assert xrefs[0].contents == "Subsection 2.4"

    def test_lowercase_subsection_still_matched(self):
        xrefs = detect_xrefs(_sent("as we mention below in subsection 3.4) use at t"), [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 3


class TestRecallPreserved:
    def test_plain_table(self):
        xrefs = detect_xrefs(_sent("Results are shown in Table 3."), _tables(3), [])
        assert [x.xref_id for x in xrefs] == [3]
        assert xrefs[0].xref_type == "table"

    def test_parenthesized_table(self):
        xrefs = detect_xrefs(_sent("Estimates rose sharply (Table 2)."), _tables(2), [])
        assert [x.xref_id for x in xrefs] == [2]

    def test_glued_digit_marker_table_still_matches(self):
        # OCR-flattened footnote marker glued before the word
        xrefs = detect_xrefs(_sent("9Table 1 lists the parameters."), _tables(1), [])
        assert [x.xref_id for x in xrefs] == [1]

    def test_figure_abbrev(self):
        xrefs = detect_xrefs(_sent("as shown in Fig. 4a."), [], _figures(4))
        assert [x.xref_id for x in xrefs] == [4]
        assert xrefs[0].xref_type == "figure"

    def test_label_glued_to_the_next_word(self):
        # Extracted text sometimes loses the space after the number.
        xrefs = detect_xrefs(_sent("Figure 3shows the effect."), [], _figures(3))
        assert [x.xref_id for x in xrefs] == [3]

    def test_a_year_is_not_a_table(self):
        xrefs = detect_xrefs(_sent("See the OECD Tables 2019 release."), _tables(1), [])
        assert xrefs == []

    def test_supp_table(self):
        xrefs = detect_xrefs(_sent("Full estimates appear in Table S1."), [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "supplementary"
        assert xrefs[0].xref_id == 1

    def test_supp_named(self):
        xrefs = detect_xrefs(_sent("See the Online Supplementary Materials for details."), [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "supplementary"

    def test_plain_section(self):
        xrefs = detect_xrefs(_sent("Robustness checks are in Section 5."), [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_type == "section"
        assert xrefs[0].xref_id == 5

    def test_section_symbol(self):
        xrefs = detect_xrefs(_sent("The proof follows §2.1 closely."), [], [])
        assert len(xrefs) == 1
        assert xrefs[0].xref_id == 2
