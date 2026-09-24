"""Structure-layer audit fixes: caption theft, markdown tables, footnotes.

Each test below fails on the pre-fix code — they are regression pins, not
characterisation tests.
"""

import pytest

from bibr.structure.footnote_buffer import FootnoteBuffer, printed_marker
from bibr.structure.parse_headings import HeadingDisposition
from bibr.structure.pdf_parser import PDFParser


class TestProseIsNotStolenAsACaption:
    """C1, second half: the LOOSE discriminators applied to a paragraph_title.

    The loose regexes require no separator — correct for a region the layout
    model already labelled a caption, wrong for a ``paragraph_title``, where
    they swallowed any sentence that merely opened with a float reference.
    A stolen sentence becomes a caption candidate, and one that finds no owner
    used to vanish from the document.
    """

    PROSE = [
        "Figure 3 illustrates the interaction between dose and latency.",
        "Fig. 3 illustrates the interaction between dose and latency.",
        "Table 7 shows the results of the regression.",
        "Table 7.5 reports the residuals for each cohort.",
        "Figure 2 depicts the experimental timeline.",
        "Table IV summarises the demographic characteristics.",
        "Table S1 lists the robustness checks.",
        "Supplementary Table 2 reports the items.",
        "Figure A1 plots the sensitivity analysis.",
    ]

    CAPTIONS = [
        "Table 1 Overview of measures",
        "Table 1: Descriptive statistics",
        "Figure 3. Effect of dose",
        "Figure 3 Mean latency by condition",
        "Table 2 continued",
        "Table 2 Continued",
        "Table S1 Robustness checks",
        "Supplementary Table 2: Items",
        "Figure A1. Sensitivity",
        "TABLE IV Characteristics",
    ]

    @pytest.mark.parametrize("text", PROSE)
    def test_prose_stays_body_text(self, text):
        parser = PDFParser(json_result=[])
        assert (
            parser._classify_heading_disposition("paragraph_title", text)
            is HeadingDisposition.BODY_TEXT
        )

    @pytest.mark.parametrize("text", CAPTIONS)
    def test_real_captions_are_still_routed_to_caption_handling(self, text):
        parser = PDFParser(json_result=[])
        disposition = parser._classify_heading_disposition("paragraph_title", text)
        assert disposition in {
            HeadingDisposition.TABLE_CAPTION,
            HeadingDisposition.FIGURE_CAPTION,
        }

    @pytest.mark.parametrize("text", PROSE)
    def test_the_stolen_sentence_survives_end_to_end(self, text):
        parser = PDFParser(json_result=[])
        parser._current_section_id = 1
        parser._handle_heading("paragraph_title", text, 4, bbox=[0, 0, 1, 1])
        parser._flush_carry_over()

        assert [e.text for e in parser.assembler.entries] == [text]

    def test_a_caption_labelled_region_still_matches_without_a_separator(self):
        """The loose discriminator must keep working where it belongs."""
        assert PDFParser._LOOSE_TABLE_CAPTION_RE.match("Table 1 Overview of measures")


class TestUnownedCaptionsKeepTheirText:
    """C1, second half: a caption candidate with no owner was deleted outright."""

    def test_an_orphan_caption_is_replayed_as_body_text(self):
        regions = [
            [
                {
                    "label": "figure_title",
                    "content": "Figure 9. A caption whose figure never arrives",
                    "bbox_2d": [0, 0, 100, 20],
                }
            ]
        ]
        parser = PDFParser(regions)
        contents = parser.parse()
        parser.apply_segmentation(
            contents, [[e.text] for e in parser.assembler.entries if e.needs_segmentation]
        )

        assert any("figure never arrives" in s.text for s in contents.sentences)

    def test_an_owned_caption_is_not_duplicated(self):
        regions = [
            [
                {
                    "label": "figure_title",
                    "content": "Figure 1. Mean latency by condition",
                    "bbox_2d": [0, 0, 100, 20],
                },
                {"label": "image", "content": "", "bbox_2d": [0, 25, 100, 200]},
            ]
        ]
        parser = PDFParser(regions)
        parser.parse()

        replayed = [e.text for e in parser.assembler.entries if "Mean latency" in e.text]
        assert replayed == []


class TestMarkdownTableParsing:
    """M9: multi-row headers, ragged rows, escaped pipes."""

    def test_multi_row_header_is_flattened_not_dropped(self):
        md = (
            "| | Group A | Group B |\n"
            "| Measure | Mean | Mean |\n"
            "| --- | --- | --- |\n"
            "| Latency | 1.0 | 2.0 |"
        )
        df = PDFParser._parse_markdown_table(md)

        assert list(df.columns) == ["Measure", "Group A Mean", "Group B Mean"]
        assert df.iloc[0].tolist() == ["Latency", "1.0", "2.0"]

    def test_a_ragged_row_is_widened_not_truncated(self):
        md = "| A | B |\n| --- | --- |\n| 1 | 2 | 3 |"
        df = PDFParser._parse_markdown_table(md)

        assert df.shape == (1, 3)
        assert df.iloc[0].tolist() == ["1", "2", "3"]

    def test_an_escaped_pipe_stays_inside_its_cell(self):
        md = "| Statistic | Value |\n| --- | --- |\n| \\|d\\| effect size | 0.42 |"
        df = PDFParser._parse_markdown_table(md)

        assert df.shape == (1, 2)
        assert df.iloc[0].tolist() == ["|d| effect size", "0.42"]

    def test_a_single_row_header_is_unchanged(self):
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        df = PDFParser._parse_markdown_table(md)

        assert list(df.columns) == ["A", "B"]
        assert df.iloc[0].tolist() == ["1", "2"]


class TestFootnoteBuffer:
    """L12: printed markers and duplicate regions."""

    def test_the_same_note_on_the_same_page_is_recorded_once(self):
        buf = FootnoteBuffer()
        for _ in range(2):
            buf.record(
                text="1 Data are available on OSF.",
                page_number=3,
                body_section_id=1,
                deferred_text_index=0,
            )

        assert len(list(buf)) == 1

    def test_whitespace_only_differences_still_count_as_duplicates(self):
        buf = FootnoteBuffer()
        buf.record(text="1 Data on OSF.", page_number=3, body_section_id=1, deferred_text_index=0)
        buf.record(
            text="1  Data   on OSF.", page_number=3, body_section_id=1, deferred_text_index=1
        )

        assert len(list(buf)) == 1

    def test_the_same_note_on_a_different_page_is_kept(self):
        """Repeating a note per page is a real convention."""
        buf = FootnoteBuffer()
        for page in (3, 4):
            buf.record(
                text="Reprinted with permission.",
                page_number=page,
                body_section_id=1,
                deferred_text_index=0,
            )

        assert len(list(buf)) == 2

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1 Data are available on OSF.", "1"),
            ("12. Data are available on OSF.", "12"),
            ("3) Data are available on OSF.", "3"),
            ("† Corresponding author.", "†"),
            ("* Equal contribution.", "*"),
            ("‡‡ Both senior authors.", "‡‡"),
            ("Data are available on OSF.", None),
            ("2024 was a good year for replication.", None),
        ],
    )
    def test_printed_marker_extraction(self, text, expected):
        assert printed_marker(text) == expected


class TestMultiRegionTitle:
    """L10: a title split across regions on one page is not a running header."""

    def _regions(self):
        return [
            [
                {"label": "doc_title", "content": "A Long Title", "bbox_2d": [0, 0, 100, 20]},
                {"label": "doc_title", "content": "And Its Subtitle", "bbox_2d": [0, 22, 100, 40]},
            ],
            [
                {"label": "doc_title", "content": "A Long Title", "bbox_2d": [0, 0, 100, 20]},
            ],
        ]

    def test_a_same_page_title_continuation_survives(self):
        parser = PDFParser(self._regions())
        parser._mark_running_headers()

        assert (0, 1) not in parser._running_header_regions

    def test_the_next_page_reprint_is_still_demoted(self):
        parser = PDFParser(self._regions())
        parser._mark_running_headers()

        assert (1, 0) in parser._running_header_regions
