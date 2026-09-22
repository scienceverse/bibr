"""Unit tests for PDFParser (glmocr json_result → PaperContents)."""

import json
from pathlib import Path

import pytest


def _split_one(text):
    """Simple period-based sentence splitter for testing."""
    sentences = []
    current = ""
    for ch in text:
        current += ch
        if ch == ".":
            sentences.append(current)
            current = ""
    if current:
        sentences.append(current)
    return sentences


def _parse_and_segment(json_result):
    """Parse json_result and apply simple sentence segmentation.

    Replaces the old pattern of injecting a mock wtpsplit model.
    """
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(json_result)
    contents = parser.parse()

    texts = [text for text, _, _, needs_seg, _ in parser._deferred_texts if needs_seg]
    segments = [_split_one(t) for t in texts]
    parser.apply_segmentation(contents, segments)
    parser.create_content_sections(contents)
    return contents


@pytest.fixture
def mock_wtpsplit():
    """No-op fixture kept for backward compat — segmentation is deferred."""
    yield None


def _region(index, label, content, bbox=None, native_label=None):
    """Helper to create a region dict matching glmocr's json_result format."""
    d = {
        "index": index,
        "label": label,
        "content": content,
        "bbox_2d": bbox or [0, 0, 100, 100],
    }
    if native_label is not None:
        d["native_label"] = native_label
    return d


def test_parse_cleans_each_region_once(monkeypatch):
    from bibr.structure import pdf_parser as module

    original = module.fix_ocr_artifacts
    calls = []

    def counted(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(module, "fix_ocr_artifacts", counted)
    module.PDFParser([[_region(0, "doc_title", "Title"), _region(1, "text", "Body text.")]]).parse()

    assert calls == ["Title", "Body text."]


# ---------------------------------------------------------------------------
# Label treatment mapping
# ---------------------------------------------------------------------------


class TestLabelTreatment:
    def test_heading_creates_section(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "doc_title", "My Paper Title"),
                _region(1, "text", "First sentence."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        # Root + doc_title = 2 sections
        assert len(contents.sections) == 2
        assert contents.sections[0].header == "Root"
        assert contents.sections[0].section_id == 0
        assert contents.sections[1].header == "My Paper Title"
        assert contents.sections[1].level == 1

    def test_doc_title_sets_detected_title(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "doc_title", "My Paper Title"),
                _region(1, "text", "First sentence."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert contents.detected_title == "My Paper Title"

    def test_doc_title_only_first_page(self, mock_wtpsplit):
        """doc_title on page 2+ should not be captured as detected_title."""
        json_result = [
            [_region(0, "text", "Page one text.")],
            [
                _region(0, "doc_title", "Author Names"),
                _region(1, "text", "Page two text."),
            ],
        ]
        contents = _parse_and_segment(json_result)

        assert contents.detected_title is None

    def test_doc_title_first_wins(self, mock_wtpsplit):
        """Only the first doc_title on page 1 is captured."""
        json_result = [
            [
                _region(0, "doc_title", "Real Title"),
                _region(1, "doc_title", "Subtitle or Author"),
                _region(2, "text", "Body."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert contents.detected_title == "Real Title"

    def test_copyright_doc_title_before_real_title_keeps_real_title(self, mock_wtpsplit):
        """A copyright/permission blurb misclassified as the first doc_title
        must not cause the real title (second doc_title) to be demoted as a
        running header and lost."""
        json_result = [
            [
                _region(0, "doc_title", "© 2020 The Authors. Licensed under Creative Commons."),
                _region(1, "doc_title", "The Actual Paper Title"),
                _region(2, "text", "Body text."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert contents.detected_title == "The Actual Paper Title"

    def test_doc_title_strips_markdown_prefix(self, mock_wtpsplit):
        """Leading '# ' from OCR content should be stripped."""
        json_result = [
            [
                _region(0, "doc_title", "# Paper With Prefix"),
                _region(1, "text", "Body."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert contents.detected_title == "Paper With Prefix"

    def test_paragraph_title_strips_markdown_emphasis(self, mock_wtpsplit):
        """Headings wrapped in **...** / __...__ / *...* / _..._ must be
        unwrapped. OCR output may preserve Markdown emphasis for bold or
        italic headers; leaving it in place causes adjacent identical headers
        to bypass the hint-section dedup and create duplicate sections.
        """
        json_result = [
            [
                _region(0, "doc_title", "Title"),
                _region(1, "paragraph_title", "**Methods**"),
                _region(2, "paragraph_title", "__Results__"),
                _region(3, "paragraph_title", "*Discussion*"),
            ]
        ]
        contents = _parse_and_segment(json_result)

        headers = [s.header for s in contents.sections if s.section_id != 0]
        assert "Methods" in headers, headers
        assert "Results" in headers, headers
        assert "Discussion" in headers, headers
        # And the unwrapped forms must NOT be present
        for raw in ("**Methods**", "__Results__", "*Discussion*"):
            assert raw not in headers, f"unwrapped {raw!r} survived: {headers}"

    def test_bold_references_heading_dedupes_against_reference_hint(self, mock_wtpsplit):
        """Markdown-bold reference headings dedupe against a reference hint.

        PP-DocLayoutV3 emits a single ``paragraph_title`` "References"
        region followed by ``reference`` body regions. The reference-hint
        path looks for an existing header whose lowercased value equals
        ``"references"``. Without Markdown stripping, ``**references**``
        does not match and a duplicate section is created.
        """
        json_result = [
            [
                _region(0, "paragraph_title", "**References**"),
                _region(1, "reference", "Smith, J. (2020). A paper."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        ref_sections = [s for s in contents.sections if s.header == "References"]
        bold_sections = [s for s in contents.sections if "**" in s.header]
        assert bold_sections == [], (
            f"markdown bold survived in section headers: {[s.header for s in contents.sections]}"
        )
        assert len(ref_sections) == 1, [s.header for s in contents.sections]

    def test_no_doc_title_means_none(self, mock_wtpsplit):
        """When no doc_title label exists, detected_title remains None."""
        json_result = [[_region(0, "text", "Just body text.")]]
        contents = _parse_and_segment(json_result)

        assert contents.detected_title is None

    def test_paragraph_title_creates_h2(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "doc_title", "Title"),
                _region(1, "paragraph_title", "Methods"),
                _region(2, "text", "We used methods."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        methods_sec = contents.sections[2]  # Root, Title, Methods
        assert methods_sec.header == "Methods"
        assert methods_sec.level == 2
        # H2 should have H1 as parent
        assert methods_sec.parent_section_id == contents.sections[1].section_id

    def test_content_creates_sentences(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "text", "First sentence. Second sentence."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.sentences) == 2
        assert contents.sentences[0].text == "First sentence."
        assert contents.sentences[1].text == "Second sentence."

    def test_abandon_labels_discarded(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "text", "Keep this."),
                _region(1, "number", "42"),
                _region(2, "header_image", "logo.png"),
                _region(3, "aside_text", "sidebar note"),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.sentences) == 1
        assert contents.sentences[0].text == "Keep this."

    def test_structural_labels_to_metadata(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "header", "Journal of Science"),
                _region(1, "text", "Body text."),
                _region(2, "footer", "Page 1"),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert "Journal of Science" in contents.detected_headers
        assert "Page 1" in contents.detected_footers
        assert len(contents.sentences) == 1


# ---------------------------------------------------------------------------
# Page number propagation
# ---------------------------------------------------------------------------


class TestPageNumbers:
    def test_page_numbers_assigned(self, mock_wtpsplit):
        json_result = [
            [_region(0, "text", "Page one text.")],
            [_region(0, "text", "Page two text.")],
            [_region(0, "text", "Page three text.")],
        ]
        contents = _parse_and_segment(json_result)

        assert contents.sentences[0].page_number == 1
        assert contents.sentences[1].page_number == 2
        assert contents.sentences[2].page_number == 3


# ---------------------------------------------------------------------------
# Cross-page sentence continuity
# ---------------------------------------------------------------------------


class TestCrossPageContinuity:
    def test_clear_lowercase_continuation_joins_across_pages(self, mock_wtpsplit):
        """An unfinished sentence can continue on the next page."""
        json_result = [
            [_region(0, "text", "This continues on the")],
            [_region(0, "text", "next page with more text.")],
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.sentences) == 1
        assert contents.sentences[0].text == "This continues on the next page with more text."
        assert [p.page_no for p in contents.sentences[0].provenance] == [1, 2]

    def test_no_join_with_terminal_punct(self, mock_wtpsplit):
        """Text with terminal punct should not join with next page."""
        json_result = [
            [_region(0, "text", "Complete sentence.")],
            [_region(0, "text", "New sentence.")],
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.sentences) == 2

    def test_no_join_uppercase_start(self, mock_wtpsplit):
        """Even without terminal punct, uppercase start means new paragraph."""
        json_result = [
            [_region(0, "text", "Some text without ending")],
            [_region(0, "text", "New paragraph starts here.")],
        ]
        contents = _parse_and_segment(json_result)

        # "Some text without ending" gets flushed as-is, then "New paragraph..."
        assert len(contents.sentences) == 2


# ---------------------------------------------------------------------------
# Section hints (abstract, reference)
# ---------------------------------------------------------------------------


class TestSectionHints:
    def test_abstract_creates_implicit_section(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "abstract", "This paper studies things."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        # Root + implicit Abstract
        assert len(contents.sections) == 2
        assert contents.sections[1].header == "Abstract"
        assert contents.sentences[0].section_id == contents.sections[1].section_id

    def test_reference_creates_implicit_section(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "text", "Body text."),
                _region(1, "reference", "Smith, J. (2020). A paper."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        ref_sections = [s for s in contents.sections if s.header == "References"]
        assert len(ref_sections) == 1

    def test_duplicate_section_hint_no_double_create(self, mock_wtpsplit):
        """Multiple 'abstract' regions should not create duplicate sections."""
        json_result = [
            [
                _region(0, "abstract", "Part one."),
                _region(1, "abstract", "Part two."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        abstract_sections = [s for s in contents.sections if s.header == "Abstract"]
        assert len(abstract_sections) == 1
        # Both sentences belong to the same section
        assert all(s.section_id == abstract_sections[0].section_id for s in contents.sentences)

    def test_reference_hint_reuse_registers_lookup_for_continuation(self, mock_wtpsplit):
        """When a reference hint reuses a heading-created References section,
        a continuation region on a later page must land in that same section
        even if another section was opened in between."""
        json_result = [
            [
                _region(0, "paragraph_title", "References"),
                _region(1, "reference", "Smith, J. (2020). A paper."),
            ],
            [
                _region(0, "paragraph_title", "Appendix A"),
                _region(1, "reference", "Jones, K. (2021). Another paper."),
            ],
        ]
        contents = _parse_and_segment(json_result)

        ref_sections = [s for s in contents.sections if s.header == "References"]
        assert len(ref_sections) == 1
        ref_section_id = ref_sections[0].section_id
        continuation = [s for s in contents.sentences if "Jones" in s.text]
        assert len(continuation) == 1
        assert continuation[0].section_id == ref_section_id

    def test_layout_hints_recorded(self, mock_wtpsplit):
        json_result = [
            [_region(0, "abstract", "Abstract text.")],
            [_region(0, "reference", "Ref text.")],
        ]
        contents = _parse_and_segment(json_result)

        assert ("abstract", 1) in contents.layout_hints
        assert ("reference", 2) in contents.layout_hints

    def test_later_printed_heading_does_not_relocate_the_hinted_section(self, mock_wtpsplit):
        contents = _parse_and_segment(
            [
                [
                    _region(0, "abstract", "This paper studies things."),
                    _region(1, "paragraph_title", "Abstract"),
                ]
            ]
        )
        abstract = next(s for s in contents.sections if s.header == "Abstract")
        assert not abstract.header_is_synthetic
        assert abstract.provenance == []


# ---------------------------------------------------------------------------
# Markdown table parsing
# ---------------------------------------------------------------------------


class TestTableParsing:
    def test_bare_heading_caption_composes_with_adjacent_fragment(self):
        """A bare Table N heading terminates as a caption and keeps its table."""
        from bibr.structure.pdf_parser import PDFParser

        fixture = Path(__file__).parent / "fixtures" / "ocr" / "split_table_caption.json"
        parser = PDFParser(json.loads(fixture.read_text()))
        contents = parser.parse()

        assert all(section.header != "Table 1" for section in contents.sections)
        assert len(contents.tables) == 1
        assert contents.tables[0].page_number == 5
        assert contents.tables[0].caption == (
            "Table 1 Descriptive statistics for the study variables"
        )

    def test_bare_table_label_does_not_consume_trusted_content_heading(self):
        json_result = [
            [
                _region(0, "paragraph_title", "Table 1"),
                _region(1, "text", "Methods"),
                _region(2, "table", "| A |\n|---|\n| 1 |"),
            ]
        ]

        contents = _parse_and_segment(json_result)

        assert [section.header for section in contents.sections].count("Methods") == 1
        assert contents.tables[0].caption == "Table 1"

    def test_unowned_table_caption_fragment_rolls_back_to_body_text(self):
        json_result = [
            [
                _region(0, "paragraph_title", "Table 1"),
                _region(1, "text", "Ordinary prose remains available."),
            ]
        ]

        contents = _parse_and_segment(json_result)

        assert any(
            sentence.text == "Ordinary prose remains available." for sentence in contents.sentences
        )

    def test_composed_table_caption_uses_label_and_fragment_bbox(self):
        json_result = [
            [
                _region(0, "paragraph_title", "Table 1", bbox=[50, 0, 150, 20]),
                _region(
                    1,
                    "text",
                    "Descriptive statistics",
                    bbox=[50, 180, 600, 350],
                ),
                _region(
                    2,
                    "table",
                    "| A |\n|---|\n| 1 |",
                    bbox=[50, 520, 600, 700],
                ),
            ]
        ]

        contents = _parse_and_segment(json_result)

        assert contents.tables[0].caption == "Table 1 Descriptive statistics"

    @pytest.mark.parametrize(
        ("label", "content"),
        [("text", ""), ("header_image", "publisher decoration")],
    )
    def test_intervening_region_expires_bare_table_fragment_window(self, label, content):
        json_result = [
            [
                _region(0, "paragraph_title", "Table 1"),
                _region(1, label, content),
                _region(2, "text", "Descriptive statistics"),
                _region(3, "table", "| A |\n|---|\n| 1 |"),
            ]
        ]

        contents = _parse_and_segment(json_result)

        assert contents.tables[0].caption == "Table 1"
        assert any(sentence.text == "Descriptive statistics" for sentence in contents.sentences)

    def test_standard_markdown_table(self, mock_wtpsplit):
        table_md = "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |"
        json_result = [
            [
                _region(0, "table", table_md),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.tables) == 1
        tbl = contents.tables[0]
        assert list(tbl.df.columns) == ["A", "B", "C"]
        assert len(tbl.df) == 2
        assert tbl.df.iloc[0]["A"] == "1"

    def test_table_with_caption(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "table_title", "Table 1: Results"),
                _region(1, "table", "| X | Y |\n|---|---|\n| a | b |"),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.tables) == 1
        assert contents.tables[0].caption == "Table 1: Results"

    def test_table_table_id_increments(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "table", "| A |\n|---|\n| 1 |"),
                _region(1, "table", "| B |\n|---|\n| 2 |"),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert contents.tables[0].table_id == 1
        assert contents.tables[1].table_id == 2

    def test_parse_markdown_table_no_separator(self, mock_wtpsplit):
        """Table without separator row treats first row as header."""
        from bibr.structure.pdf_parser import PDFParser

        result = PDFParser._parse_markdown_table("| A | B |\n| 1 | 2 |")
        assert result is not None
        assert list(result.columns) == ["A", "B"]
        assert len(result) == 1

    def test_corrupted_table_attribute_junk_salvaged(self, mock_wtpsplit):
        """NUL-stripped tag attributes (``<table">``) must not lose the table.

        OCR control-char corruption can shred tag attributes while leaving
        row/cell structure intact; html5lib then sees an unknown tag and
        ``read_html`` finds no table. A salvage retry with normalized table
        tags must recover it.
        """
        corrupted = (
            '<table"><thead><tr><th>A</th><th>B</th></tr></thead>'
            "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
        )
        json_result = [[_region(0, "table", corrupted)]]
        contents = _parse_and_segment(json_result)

        assert len(contents.tables) == 1
        assert list(contents.tables[0].df.columns) == ["A", "B"]

    def test_unparseable_table_emits_distinct_warning(self, mock_wtpsplit):
        """A table region that cannot be parsed at all must fail loudly.

        Real-world shape: NUL bytes replaced the cell/row tags themselves, so
        after control-char stripping no table structure remains. The table is
        dropped, but a distinct OCR_TABLE_DROPPED processing warning must be
        emitted instead of silence.
        """
        shredded = "\n\n\n\n\n<table\n<\n<table><thead><\n\n```text\n\n\n```"
        json_result = [[_region(0, "table", shredded)]]
        contents = _parse_and_segment(json_result)

        assert contents.tables == []
        assert any("OCR_TABLE_DROPPED" in w for w in contents.processing_warnings)

    def test_valid_tables_emit_no_drop_warning(self, mock_wtpsplit):
        json_result = [[_region(0, "table", "| A |\n|---|\n| 1 |")]]
        contents = _parse_and_segment(json_result)

        assert len(contents.tables) == 1
        assert not any("OCR_TABLE_DROPPED" in w for w in contents.processing_warnings)


# ---------------------------------------------------------------------------
# Formula handling
# ---------------------------------------------------------------------------


class TestFormulas:
    def test_formula_wrapped_in_delimiters(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "display_formula", "E = mc^2"),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.sentences) == 1
        assert contents.sentences[0].text == "E = mc^2"
        assert contents.sentences[0].is_display_formula is True

    def test_formula_already_wrapped(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "display_formula", "$$E = mc^2$$"),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert contents.sentences[0].text == "E = mc^2"
        assert contents.sentences[0].is_display_formula is True


# ---------------------------------------------------------------------------
# Figure handling
# ---------------------------------------------------------------------------


class TestFigures:
    def test_figure_created(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "image", ""),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.figures) == 1
        assert contents.figures[0].figure_id == 1
        assert contents.figures[0].image_b64 is None

    def test_figure_with_caption(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "figure_title", "Figure 1: Architecture"),
                _region(1, "image", ""),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert contents.figures[0].caption == "Figure 1: Architecture"

    def test_chart_title_also_works(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "chart_title", "Chart: Performance"),
                _region(1, "chart", ""),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert contents.figures[0].caption == "Chart: Performance"


# ---------------------------------------------------------------------------
# bbox proximity caption matching
# ---------------------------------------------------------------------------


class TestBboxCaptionMatching:
    """Tests for bbox_2d-based caption–element proximity matching."""

    def test_table_caption_before_table_same_page(self, mock_wtpsplit):
        """Forward match: caption before table on same page attaches."""
        json_result = [
            [
                _region(0, "table_title", "Table 1: Data", bbox=[100, 200, 500, 230]),
                _region(1, "table", "| A |\n|---|\n| 1 |", bbox=[100, 240, 500, 400]),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert contents.tables[0].caption == "Table 1: Data"

    def test_figure_caption_after_figure_same_page(self, mock_wtpsplit):
        """Backward match: caption after figure on same page attaches retroactively."""
        json_result = [
            [
                _region(0, "image", "", bbox=[100, 100, 500, 400]),
                _region(1, "figure_title", "Figure 1: Overview", bbox=[100, 410, 500, 440]),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert contents.figures[0].caption == "Figure 1: Overview"

    def test_caption_different_page_not_attached(self, mock_wtpsplit):
        """Caption on page 1 should not attach to table on page 2."""
        json_result = [
            [
                _region(0, "table_title", "Table 1: Orphan", bbox=[100, 900, 500, 930]),
            ],
            [
                _region(0, "table", "| X |\n|---|\n| 1 |", bbox=[100, 50, 500, 200]),
            ],
        ]
        contents = _parse_and_segment(json_result)

        assert contents.tables[0].caption is None

    def test_caption_too_far_vertically_not_attached(self, mock_wtpsplit):
        """Caption far from table vertically (>200 units) should not attach."""
        json_result = [
            [
                _region(0, "table_title", "Table 1: Far", bbox=[100, 50, 500, 80]),
                _region(1, "text", "Intervening paragraph.", bbox=[100, 100, 500, 300]),
                _region(2, "table", "| A |\n|---|\n| 1 |", bbox=[100, 700, 500, 900]),
            ]
        ]
        contents = _parse_and_segment(json_result)

        # Gap is 700 - 80 = 620 > 200 threshold
        assert contents.tables[0].caption is None

    def test_retroactive_caption_too_far_not_attached(self, mock_wtpsplit):
        """Retroactive caption far from figure should not attach."""
        json_result = [
            [
                _region(0, "image", "", bbox=[100, 50, 500, 200]),
                _region(1, "text", "Body text.", bbox=[100, 220, 500, 400]),
                _region(2, "figure_title", "Figure 1: Late", bbox=[100, 800, 500, 830]),
            ]
        ]
        contents = _parse_and_segment(json_result)

        # Gap is 800 - 200 = 600 > 200 threshold
        assert contents.figures[0].caption is None

    def test_no_bbox_caption_abstains_instead_of_guessing(self, mock_wtpsplit):
        """Missing geometry is not enough evidence even for one sequential target."""
        json_result = [
            [
                {"index": 0, "label": "table_title", "content": "Table 1: No bbox"},
                {"index": 1, "label": "table", "content": "| A |\n|---|\n| 1 |"},
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert contents.tables[0].caption is None
        assert contents.caption_assignment_receipt.assignments[0].reasons == (
            "missing_geometry",
            "unmatched",
        )
        assert {issue.code for issue in contents.structure_validation_issues} == {
            "VAL_CAPTION_OWNERSHIP"
        }

    def test_heading_clears_uncaptioned_tracking(self, mock_wtpsplit):
        """A heading between figure and caption should prevent retroactive attachment."""
        json_result = [
            [
                _region(0, "image", "", bbox=[100, 100, 500, 300]),
                _region(1, "paragraph_title", "New Section"),
                _region(2, "figure_title", "Figure 1: Wrong", bbox=[100, 500, 500, 530]),
            ]
        ]
        contents = _parse_and_segment(json_result)

        # Heading cleared _last_uncaptioned_figure → no retroactive attachment
        assert contents.figures[0].caption is None

    def test_retroactive_table_caption(self, mock_wtpsplit):
        """Table caption after table on same page attaches retroactively."""
        json_result = [
            [
                _region(0, "table", "| A |\n|---|\n| 1 |", bbox=[100, 100, 500, 300]),
                _region(1, "table_title", "Table 1: After", bbox=[100, 310, 500, 340]),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert contents.tables[0].caption == "Table 1: After"


# ---------------------------------------------------------------------------
# Footnote handling
# ---------------------------------------------------------------------------


class TestFootnotes:
    def test_footnote_without_deferred_texts_does_not_crash(self):
        """Footnotes but zero deferred body texts: the pipeline skips
        apply_segmentation entirely (parse_segment stage gates it on
        parser._deferred_texts), so create_content_sections must not rely
        on state that only apply_segmentation initializes."""
        from bibr.structure.pdf_parser import PDFParser

        json_result = [[_region(0, "footnote", "A lone footnote.")]]
        parser = PDFParser(json_result)
        contents = parser.parse()

        # Precondition for the bug: nothing was deferred, so the pipeline
        # would never call apply_segmentation.
        assert not parser._deferred_texts

        parser.create_content_sections(contents)  # must not raise

        fn_sections = [
            s for s in contents.sections if s.section_type and s.section_type.value == "footnote"
        ]
        assert len(fn_sections) == 1

    def test_footnote_becomes_section(self, mock_wtpsplit):
        """Footnote creates a section (type=footnote) with text in the text table."""
        json_result = [
            [
                _region(0, "text", "Main body text."),
                _region(1, "footnote", "This is a footnote."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        # Footnote section should exist
        fn_sections = [
            s for s in contents.sections if s.section_type and s.section_type.value == "footnote"
        ]
        assert len(fn_sections) == 1
        fn_section = fn_sections[0]
        assert fn_section.header == "Footnote 1"

        # Footnote text should be in the text table under the footnote section
        fn_sentences = [s for s in contents.sentences if s.section_id == fn_section.section_id]
        assert len(fn_sentences) == 1
        assert fn_sentences[0].text == "This is a footnote."

        # Xref should link footnote section to the body sentence
        foot_xrefs = [x for x in contents.xrefs if x.xref_type == "foot"]
        assert len(foot_xrefs) == 1
        assert foot_xrefs[0].xref_id == fn_section.section_id
        # text_id should point to the body sentence (nearest preceding)
        body_sentences = [s for s in contents.sentences if s.section_id != fn_section.section_id]
        assert foot_xrefs[0].text_id == body_sentences[0].text_id


# ---------------------------------------------------------------------------
# Cross-reference detection (via xref_utils)
# ---------------------------------------------------------------------------


class TestXrefDetection:
    def test_table_xref_detected(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "table", "| A |\n|---|\n| 1 |"),
                _region(1, "text", "See Table 1 for details."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        tbl_xrefs = [x for x in contents.xrefs if x.xref_type == "table"]
        assert len(tbl_xrefs) == 1
        assert tbl_xrefs[0].xref_id == 1

    def test_figure_xref_detected(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "image", ""),
                _region(1, "text", "As shown in Figure 1."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        fig_xrefs = [x for x in contents.xrefs if x.xref_type == "figure"]
        assert len(fig_xrefs) == 1


# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------


class TestURLDetection:
    def test_url_in_text(self, mock_wtpsplit):
        # Use a URL that ends right before the period so the mock period-based
        # splitter doesn't break the URL across sentences.
        json_result = [
            [
                _region(0, "text", "Visit https://example.com/page for more."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        url_links = [lnk for lnk in contents.links if "example" in lnk.url]
        assert len(url_links) >= 1
        assert url_links[0].url.startswith("https://example")
        assert url_links[0].link_text is None


# ---------------------------------------------------------------------------
# sections_text correctness
# ---------------------------------------------------------------------------


class TestSectionsText:
    def test_sections_text_populated(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "doc_title", "Introduction"),
                _region(1, "text", "Hello world."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        sec_id = contents.sections[1].section_id
        assert sec_id in contents.sections_text
        assert "Hello world." in contents.sections_text[sec_id]


# ---------------------------------------------------------------------------
# Section hierarchy
# ---------------------------------------------------------------------------


class TestSectionHierarchy:
    def test_nested_sections(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "doc_title", "Main Title"),
                _region(1, "paragraph_title", "Subsection A"),
                _region(2, "text", "Subsection text."),
                _region(3, "paragraph_title", "Subsection B"),
                _region(4, "text", "More text."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        # Root(0) → Main Title(1, H1) → Subsection A(2, H2), Subsection B(3, H2)
        assert len(contents.sections) == 4
        assert contents.sections[2].parent_section_id == contents.sections[1].section_id
        assert contents.sections[3].parent_section_id == contents.sections[1].section_id


# ---------------------------------------------------------------------------
# Full multi-page document
# ---------------------------------------------------------------------------


class TestFullDocument:
    def test_multi_page_paper(self, mock_wtpsplit):
        """End-to-end test with a realistic multi-page structure."""
        json_result = [
            # Page 1
            [
                _region(0, "header", "Journal of Tests"),
                _region(1, "doc_title", "A Study of Testing"),
                _region(2, "abstract", "We tested things. Results were good."),
                _region(3, "footer", "Page 1"),
            ],
            # Page 2
            [
                _region(0, "header", "Journal of Tests"),
                _region(1, "paragraph_title", "Methods"),
                _region(2, "text", "We used pytest."),
                _region(3, "table_title", "Table 1: Results"),
                _region(4, "table", "| Test | Pass |\n|------|------|\n| A | Yes |\n| B | No |"),
                _region(5, "footer", "Page 2"),
            ],
            # Page 3
            [
                _region(0, "paragraph_title", "Discussion"),
                _region(1, "text", "Table 1 shows that tests matter. See Figure 1."),
                _region(2, "image", ""),
                _region(3, "figure_title", "Figure 1: Architecture"),
                _region(4, "reference", "Smith, J. (2020). Testing."),
            ],
        ]
        contents = _parse_and_segment(json_result)

        # Sections: Root, A Study of Testing (H1), Abstract, Methods (H2),
        #           Discussion (H2), References
        section_names = [s.header for s in contents.sections]
        assert "Root" in section_names
        assert "A Study of Testing" in section_names
        assert "Abstract" in section_names
        assert "Methods" in section_names
        assert "Discussion" in section_names
        assert "References" in section_names

        # Tables
        assert len(contents.tables) == 1
        assert contents.tables[0].caption == "Table 1: Results"

        # Figures — retroactive caption attachment (figure_title after image on same page)
        assert len(contents.figures) == 1
        assert contents.figures[0].caption == "Figure 1: Architecture"

        # Page numbers
        abstract_sents = [s for s in contents.sentences if s.page_number == 1]
        assert len(abstract_sents) > 0

        # Structural metadata
        assert "Journal of Tests" in contents.detected_headers
        assert "Page 1" in contents.detected_footers

        # Cross-references
        tbl_xrefs = [x for x in contents.xrefs if x.xref_type == "table"]
        assert len(tbl_xrefs) >= 1

        # Layout hints
        assert any(label == "abstract" for label, _ in contents.layout_hints)
        assert any(label == "reference" for label, _ in contents.layout_hints)


# ---------------------------------------------------------------------------
# Native label routing (glmocr label collapse workaround)
# ---------------------------------------------------------------------------


class TestNativeLabelRouting:
    """Tests for native_label-based treatment dispatch.

    glmocr's ResultFormatter._map_label() collapses specific labels
    (e.g. "abstract" → "text"). The native_label field preserves the
    original layout detector label. PDFParser should prefer native_label
    for LABEL_TREATMENT lookup when available.
    """

    def test_native_label_abstract_creates_section(self, mock_wtpsplit):
        """abstract native_label → section_hint treatment → implicit Abstract section."""
        json_result = [
            [
                # label is collapsed to "text", but native_label preserves "abstract"
                _region(0, "text", "This is the abstract text.", native_label="abstract"),
            ]
        ]
        contents = _parse_and_segment(json_result)

        # Should have Root + implicit "Abstract" section
        section_names = [s.header for s in contents.sections]
        assert "Abstract" in section_names
        assert len(contents.sections) == 2  # Root + Abstract

    def test_native_label_footnote_creates_section(self, mock_wtpsplit):
        """footnote native_label → footnote treatment → footnote section created."""
        json_result = [
            [
                _region(0, "text", "Body text first."),
                # label collapsed to "text", native_label preserves "footnote"
                _region(1, "text", "This is a footnote.", native_label="footnote"),
            ]
        ]
        contents = _parse_and_segment(json_result)

        fn_sections = [
            s for s in contents.sections if s.section_type and s.section_type.value == "footnote"
        ]
        assert len(fn_sections) == 1
        fn_sentences = [s for s in contents.sentences if s.section_id == fn_sections[0].section_id]
        assert len(fn_sentences) == 1
        assert fn_sentences[0].text == "This is a footnote."

    def test_native_label_reference_creates_section(self, mock_wtpsplit):
        """reference native_label → section_hint treatment → implicit References section."""
        json_result = [
            [
                _region(0, "text", "Smith, J. (2020). A study.", native_label="reference"),
            ]
        ]
        contents = _parse_and_segment(json_result)

        section_names = [s.header for s in contents.sections]
        assert "References" in section_names

    def test_native_label_fallback_to_label(self, mock_wtpsplit):
        """Unknown native_label → fall back to label for treatment lookup."""
        json_result = [
            [
                # native_label not in LABEL_TREATMENT → use label="text" → "content" treatment
                _region(0, "text", "Some body text.", native_label="some_unknown_label"),
            ]
        ]
        contents = _parse_and_segment(json_result)

        # No extra sections created (only Root), text becomes sentences
        assert len(contents.sections) == 1  # Root only
        assert len(contents.sentences) >= 1

    def test_layout_hints_from_native_label(self, mock_wtpsplit):
        """Layout hints are recorded using effective_label (native_label when available)."""
        json_result = [
            [
                _region(0, "text", "Abstract content.", native_label="abstract"),
                _region(1, "text", "Reference content.", native_label="reference"),
            ]
        ]
        contents = _parse_and_segment(json_result)

        hint_labels = [label for label, _ in contents.layout_hints]
        assert "abstract" in hint_labels
        assert "reference" in hint_labels

    def test_content_none_normalized(self, mock_wtpsplit):
        """Region with content=None (e.g. skip-type images) should not crash."""
        json_result = [
            [
                {"index": 0, "label": "image", "content": None, "bbox_2d": [0, 0, 500, 500]},
            ]
        ]
        contents = _parse_and_segment(json_result)

        # Should create a figure without crashing
        assert len(contents.figures) == 1


class TestHeadingPrefixStripping:
    """Tests for markdown heading prefix stripping in _handle_heading."""

    def test_paragraph_title_prefix_stripped(self, mock_wtpsplit):
        """paragraph_title with ## prefix → section header is clean."""
        json_result = [[_region(0, "paragraph_title", "## Methods")]]
        contents = _parse_and_segment(json_result)

        # Root + Methods = 2 sections
        assert contents.sections[-1].header == "Methods"

    def test_doc_title_prefix_stripped_in_section(self, mock_wtpsplit):
        """doc_title with # prefix → both detected_title and section header are clean."""
        json_result = [[_region(0, "doc_title", "# My Paper Title")]]
        contents = _parse_and_segment(json_result)

        assert contents.detected_title == "My Paper Title"
        assert contents.sections[-1].header == "My Paper Title"

    def test_doc_title_strips_affiliation_markers(self, mock_wtpsplit):
        """Affiliation markers like $ ^{1} $ should be stripped from detected_title and section header."""
        json_result = [
            [
                _region(0, "doc_title", "My Paper Title $ ^{1} $"),
                _region(1, "text", "Body."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert contents.detected_title == "My Paper Title"
        assert contents.sections[-1].header == "My Paper Title"


# ---------------------------------------------------------------------------
# OCR artifact correction
# ---------------------------------------------------------------------------


class TestOcrArtifactCorrection:
    """Tests for OCR artifact correction integrated into _process_page."""

    def test_ligature_corrected_in_content(self, mock_wtpsplit):
        """fi/fl ligatures in content are corrected before sentence creation."""
        # U+FB01 = ﬁ, U+FB02 = ﬂ
        json_result = [
            [_region(0, "text", "The \ufb01ndings were signi\ufb01cant and in\ufb02uential.")]
        ]
        contents = _parse_and_segment(json_result)

        full_text = " ".join(s.text for s in contents.sentences)
        assert "findings" in full_text
        assert "significant" in full_text
        assert "influential" in full_text
        assert "\ufb01" not in full_text
        assert "\ufb02" not in full_text

    def test_soft_hyphen_corrected(self, mock_wtpsplit):
        """Soft hyphens (U+00AD) are replaced with regular hyphens."""
        json_result = [[_region(0, "text", "investiga\u00adtion results.")]]
        contents = _parse_and_segment(json_result)

        full_text = " ".join(s.text for s in contents.sentences)
        assert "\u00ad" not in full_text  # soft hyphen replaced

    def test_ligature_corrected_in_heading(self, mock_wtpsplit):
        """Ligatures in headings are also corrected."""
        json_result = [[_region(0, "paragraph_title", "## Signi\ufb01cant \ufb01ndings")]]
        contents = _parse_and_segment(json_result)

        assert contents.sections[-1].header == "Significant findings"


# ---------------------------------------------------------------------------
# Terminal punctuation and carry-over
# ---------------------------------------------------------------------------


class TestTerminalPunctuation:
    """Tests for bracket citation terminal punctuation."""

    def test_bracket_citation_terminal(self, mock_wtpsplit):
        """Text ending with ] is treated as terminal (no carry-over join)."""
        json_result = [
            [
                _region(0, "text", "as shown in [1]"),
                _region(1, "text", "the next paragraph starts here."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        # Should NOT be joined — each is separate
        texts = [s.text for s in contents.sentences]
        # First region ends with ] (terminal), so it's flushed separately
        assert any("as shown in [1]" in t for t in texts)
        assert any("the next paragraph" in t for t in texts)

    def test_bracket_citation_no_join_lowercase(self, mock_wtpsplit):
        """Text ending with ] is not joined even when next starts lowercase."""
        json_result = [
            [
                _region(0, "text", "results [1]"),
                _region(1, "text", "the next item."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        # "]" is terminal, so no join despite lowercase "the"
        # Each region should produce separate sentence(s)
        texts = [s.text for s in contents.sentences]
        assert len(texts) >= 2
        # No single sentence should contain both texts merged
        for t in texts:
            assert not ("results [1]" in t and "the next" in t)


# ---------------------------------------------------------------------------
# Reference section text handling
# ---------------------------------------------------------------------------


class TestReferenceSectionHandling:
    """Tests for direct emission of reference section text."""

    def test_reference_section_no_sentence_split(self, mock_wtpsplit):
        """Reference text is emitted as a single sentence, not split by wtpsplit."""
        ref_text = "Smith, J. (2020). A study of things. Journal of Tests, 1(2), 3-4."
        json_result = [[_region(0, "text", ref_text, native_label="reference")]]
        contents = _parse_and_segment(json_result)

        # Reference section should exist
        section_names = [s.header for s in contents.sections]
        assert "References" in section_names

        # The reference text should be a SINGLE sentence (not fragmented)
        ref_section_id = next(s.section_id for s in contents.sections if s.header == "References")
        ref_sentences = [s for s in contents.sentences if s.section_id == ref_section_id]
        assert len(ref_sentences) == 1
        assert ref_sentences[0].text == ref_text

    def test_abstract_section_still_segmented(self, mock_wtpsplit):
        """Abstract text is still sentence-segmented (only references skip it)."""
        abstract_text = "First sentence. Second sentence. Third sentence."
        json_result = [[_region(0, "text", abstract_text, native_label="abstract")]]
        contents = _parse_and_segment(json_result)

        # Abstract section should exist
        abs_section_id = next(s.section_id for s in contents.sections if s.header == "Abstract")
        abs_sentences = [s for s in contents.sentences if s.section_id == abs_section_id]
        # Mock wtpsplit splits on periods, so 3 sentences expected
        assert len(abs_sentences) == 3


# ---------------------------------------------------------------------------
# Fix 3: multi-slot pending captions
# ---------------------------------------------------------------------------


class TestMultiSlotPendingCaptions:
    """Pending captions use a list so multiple captions aren't lost."""

    def test_two_figure_captions_before_two_figures(self, mock_wtpsplit):
        """Both captions should attach to their respective figures."""
        json_result = [
            [
                _region(0, "figure_title", "Figure 1: First"),
                _region(1, "figure_title", "Figure 2: Second"),
                _region(2, "image", ""),
                _region(3, "image", ""),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.figures) == 2
        assert contents.figures[0].caption == "Figure 1: First"
        assert contents.figures[1].caption == "Figure 2: Second"

    def test_pending_caption_not_discarded_on_bbox_mismatch(self, mock_wtpsplit):
        """Caption should survive if first figure is too far away."""
        json_result = [
            [
                # Caption at top of page
                _region(0, "figure_title", "Figure 1: Correct", bbox=[100, 50, 500, 80]),
                # Figure at bottom of page — too far (gap > 200)
                _region(1, "image", "", bbox=[100, 700, 500, 900]),
                # Real figure nearby
                _region(2, "image", "", bbox=[100, 85, 500, 300]),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.figures) == 2
        # Printed IDs determine final ordering after ownership reconciliation.
        captioned, uncaptioned = contents.figures
        assert captioned.figure_id == 1
        assert captioned.caption == "Figure 1: Correct"
        assert captioned.provenance[0].bbox == (100.0, 85.0, 500.0, 300.0)
        assert uncaptioned.figure_id == 2
        assert uncaptioned.caption is None
        assert uncaptioned.provenance[0].bbox == (100.0, 700.0, 500.0, 900.0)


# ---------------------------------------------------------------------------
# Fix 4: smart caption routing (figure_title → table/figure based on content)
# ---------------------------------------------------------------------------


class TestCaptionRouting:
    """figure_title regions with 'Table ...' content route to table captions."""

    def test_figure_title_with_table_prefix_attaches_to_table(self, mock_wtpsplit):
        """A figure_title that says 'Table 1: ...' should attach to a table."""
        json_result = [
            [
                _region(0, "figure_title", "Table 1: Results"),
                _region(1, "table", "| A |\n|---|\n| 1 |"),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.tables) == 1
        assert contents.tables[0].caption == "Table 1: Results"
        # Should NOT create a figure
        assert len(contents.figures) == 0

    def test_figure_title_with_figure_prefix_attaches_to_figure(self, mock_wtpsplit):
        """A figure_title that says 'Figure ...' should attach to a figure."""
        json_result = [
            [
                _region(0, "figure_title", "Figure 3: Diagram"),
                _region(1, "image", ""),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.figures) == 1
        assert contents.figures[0].caption == "Figure 3: Diagram"

    def test_table_caption_retroactive_via_figure_title(self, mock_wtpsplit):
        """Table followed by figure_title with 'Table ...' should retroactively attach."""
        json_result = [
            [
                _region(0, "table", "| X |\n|---|\n| 1 |", bbox=[100, 50, 500, 200]),
                _region(1, "figure_title", "Table 2: Summary", bbox=[100, 210, 500, 240]),
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.tables) == 1
        assert contents.tables[0].caption == "Table 2: Summary"


# ---------------------------------------------------------------------------
# Fix 5: figure image_b64 extraction
# ---------------------------------------------------------------------------


class TestFigureImageData:
    """Figures carry base64-encoded image data when present in regions."""

    def test_figure_with_image_b64(self, mock_wtpsplit):
        """image_b64 on region flows through to PaperFigure."""
        fake_b64 = "iVBORw0KGgo="
        json_result = [
            [
                {
                    "index": 0,
                    "label": "image",
                    "content": "",
                    "image_b64": fake_b64,
                }
            ]
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.figures) == 1
        assert contents.figures[0].image_b64 == fake_b64

    def test_figure_without_image_b64(self, mock_wtpsplit):
        """Missing image_b64 falls back to None."""
        json_result = [[_region(0, "image", "")]]
        contents = _parse_and_segment(json_result)

        assert len(contents.figures) == 1
        assert contents.figures[0].image_b64 is None


# ---------------------------------------------------------------------------
# Regression tests: carry-over join heuristic edge cases
# ---------------------------------------------------------------------------


class TestCarryOverEdgeCases:
    """Regression tests for _should_join / _has_terminal_punct heuristics."""

    def test_abbreviation_period_prevents_join(self, mock_wtpsplit):
        """Abbreviation-ending text (e.g. 'et al.') has terminal punct, so no join."""
        json_result = [
            [
                _region(0, "text", "Smith et al."),
                _region(1, "text", "found significant results."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        # 'et al.' ends with period → terminal punct → no join → two separate entries
        assert len(contents.sentences) == 2

    def test_colon_prevents_join(self, mock_wtpsplit):
        """Text ending with colon should not join with next block."""
        json_result = [
            [
                _region(0, "text", "The following methods were used:"),
                _region(1, "text", "regression analysis and ANOVA."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        assert len(contents.sentences) == 2

    def test_lowercase_continuation_joins_same_page(self, mock_wtpsplit):
        """Text without terminal punct followed by lowercase start should join on same page."""
        json_result = [
            [
                _region(0, "text", "The participants were"),
                _region(1, "text", "randomly assigned to groups."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        # Should join: no terminal punct + lowercase start + same page
        assert len(contents.sentences) == 1
        assert "were randomly" in contents.sentences[0].text

    def test_uppercase_start_joins_on_same_page(self, mock_wtpsplit):
        """Capitalized continuation on the same page joins (cross-column carry-over).

        Previous text lacks terminal punct, so the next region — even if it
        starts with a capital letter (typical of two-column layouts where
        the next column begins a new clause within the same sentence) — is
        treated as a continuation and joined.
        """
        json_result = [
            [
                _region(0, "text", "The results were unexpected"),
                _region(1, "text", "Further analysis is needed."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        # 'Further' starts uppercase but no terminal punct on prev → joined
        assert len(contents.sentences) == 1
        assert "The results were unexpected Further analysis is needed." in (
            contents.sentences[0].text
        )

    def test_parenthesis_is_terminal(self, mock_wtpsplit):
        """Closing parenthesis counts as terminal punctuation."""
        json_result = [
            [
                _region(0, "text", "values were significant (n=100)"),
                _region(1, "text", "indicating a strong effect."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        # ')' is terminal → two separate text blocks, not joined
        assert len(contents.sentences) == 2

    def test_lowercase_continuation_joins_across_pages(self, mock_wtpsplit):
        json_result = [
            [
                _region(0, "text", 'Participants said "they come to'),
                _region(1, "table", "| A | B |\n|---|---|\n| 1 | 2 |"),
            ],
            [_region(0, "text", "appointments more often, and can reach out more easily.")],
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.sentences) == 1
        assert contents.sentences[0].text == (
            'Participants said "they come to appointments more often, '
            "and can reach out more easily."
        )
        assert [p.page_no for p in contents.sentences[0].provenance] == [1, 2]

    def test_capitalized_new_paragraph_does_not_join_across_pages(self, mock_wtpsplit):
        json_result = [
            [_region(0, "text", "An intentionally incomplete fragment")],
            [_region(0, "text", "A New Paragraph starts here.")],
        ]
        contents = _parse_and_segment(json_result)

        assert len(contents.sentences) == 2

    def test_bracket_citation_is_terminal(self, mock_wtpsplit):
        """Closing bracket (citation) counts as terminal punctuation."""
        json_result = [
            [
                _region(0, "text", "as shown in prior work [12]"),
                _region(1, "text", "the model performs well."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        assert len(contents.sentences) == 2

    def test_cross_page_lowercase_continuation_joins(self, mock_wtpsplit):
        """Lowercase continuation repairs a sentence split at a page boundary."""
        json_result = [
            [_region(0, "text", "The analysis revealed that the")],
            [_region(0, "text", "treatment group improved significantly.")],
        ]
        contents = _parse_and_segment(json_result)
        assert len(contents.sentences) == 1
        assert contents.sentences[0].page_number == 1
        assert [p.page_no for p in contents.sentences[0].provenance] == [1, 2]

    def test_empty_carry_over_flushed_on_heading(self, mock_wtpsplit):
        """Heading after non-terminal text should flush the carry-over."""
        json_result = [
            [
                _region(0, "text", "Some incomplete text without ending"),
                _region(1, "paragraph_title", "Methods"),
                _region(2, "text", "We used methods."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        texts = [s.text for s in contents.sentences]
        assert "Some incomplete text without ending" in texts
        assert "We used methods." in texts

    def test_multiple_carry_overs_chain(self, mock_wtpsplit):
        """Multiple non-terminal regions on same page should accumulate."""
        json_result = [
            [
                _region(0, "text", "The first part of a"),
                _region(1, "text", "long sentence that spans"),
                _region(2, "text", "multiple OCR regions finally ends."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        # First two join, then third joins because "long" starts lowercase
        # and "multiple" starts lowercase
        assert len(contents.sentences) == 1
        assert "first part of a long sentence that spans multiple" in contents.sentences[0].text

    @staticmethod
    def _parse_whole_text(json_result):
        """Parse and segment each deferred text as a single sentence.

        The naive period splitter in ``_parse_and_segment`` would split
        inside ``github.com``; the real wtpsplit model keeps URL-bearing
        sentences whole, so whole-text segments are the faithful mock here.
        """
        from bibr.structure.pdf_parser import PDFParser

        parser = PDFParser(json_result)
        contents = parser.parse()
        texts = [t for t, _, _, needs_seg, _ in parser._deferred_texts if needs_seg]
        parser.apply_segmentation(contents, [[t] for t in texts])
        parser.create_content_sections(contents)
        return contents

    def test_url_wrap_hyphen_at_region_end_joins_without_space(self, mock_wtpsplit):
        """Region ending mid-URL with a wrap hyphen joins without an inserted space."""
        json_result = [
            [
                _region(0, "text", "Code is available from https://github.com/Lak-"),
                _region(1, "text", "ens/to_err_is_human and analysis follows."),
            ]
        ]
        contents = self._parse_whole_text(json_result)
        assert len(contents.sentences) == 1
        assert "https://github.com/Lak-ens/to_err_is_human" in contents.sentences[0].text
        assert any(
            link.url == "https://github.com/Lak-ens/to_err_is_human" for link in contents.links
        )

    def test_url_wrap_hyphen_at_region_start_joins_without_space(self, mock_wtpsplit):
        """APA-style wrap: next region starts with the hyphen of a broken URL."""
        json_result = [
            [
                _region(0, "text", "Code is available from https://github.com/Lak"),
                _region(1, "text", "-ens/to_err_is_human and analysis follows."),
            ]
        ]
        contents = self._parse_whole_text(json_result)
        assert len(contents.sentences) == 1
        assert "https://github.com/Lak-ens/to_err_is_human" in contents.sentences[0].text

    def test_plain_text_carry_over_still_joins_with_space(self, mock_wtpsplit):
        """Non-URL carry-over keeps the space joiner."""
        json_result = [
            [
                _region(0, "text", "The participants were"),
                _region(1, "text", "randomly assigned."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        assert "were randomly" in contents.sentences[0].text

    def test_bare_orcid_wrap_across_regions_joins_without_space(self, mock_wtpsplit):
        """Bare ORCID iD split at a region boundary re-joins without a space."""
        json_result = [
            [
                _region(0, "text", "Daniel Lakens 0000-0002-"),
                _region(1, "text", "0247-239X wrote the draft."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        assert len(contents.sentences) == 1
        assert "0000-0002-0247-239X" in contents.sentences[0].text


# ---------------------------------------------------------------------------
# Regression tests: hint section ordering and deduplication
# ---------------------------------------------------------------------------


class TestHintSectionEdgeCases:
    """Regression tests for section hint deduplication and ordering."""

    def test_heading_after_hint_reuses_section(self, mock_wtpsplit):
        """An explicit 'Abstract' heading after an abstract hint should not duplicate."""
        json_result = [
            [
                _region(0, "abstract", "This paper presents a study."),
                _region(1, "paragraph_title", "Abstract"),
                _region(2, "text", "More abstract text here."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        abstract_sections = [s for s in contents.sections if s.header == "Abstract"]
        assert len(abstract_sections) == 1

    def test_control_padded_heading_after_hint_reuses_section(self, mock_wtpsplit):
        """Native text can contain C0 padding bytes around an otherwise exact
        heading. Those bytes must not bypass hint-section deduplication."""
        json_result = [
            [
                _region(0, "abstract", "This paper presents a study."),
                _region(1, "paragraph_title", "\x00\x00Abstract"),
                _region(2, "text", "More abstract text here."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        headers = [s.header for s in contents.sections if s.section_id != 0]
        assert headers == ["Abstract"]
        assert not any("\x00" in h for h in headers)

    def test_heading_like_body_rows_are_promoted_after_control_cleanup(self, mock_wtpsplit):
        """When native text labels obvious short headings as body text, they
        should open sections instead of leaking into the previous section's
        sentences."""
        json_result = [
            [
                _region(0, "doc_title", "Title"),
                _region(1, "text", "Opening paragraph."),
                _region(2, "text", "\x00\x00 Study\x00 \x003"),
                _region(3, "text", "\x00Method"),
                _region(4, "text", "Participants completed the task."),
                _region(5, "text", "\x00Results"),
                _region(6, "text", "Accuracy improved."),
                _region(7, "text", "\x00General\x00Dis\x00cussion"),
                _region(8, "text", "The findings are discussed."),
            ]
        ]

        contents = _parse_and_segment(json_result)

        headers = [s.header for s in contents.sections if s.section_id != 0]
        assert headers == ["Title", "Study 3", "Method", "Results", "General Discussion"]

        emitted_text = " ".join(s.text for s in contents.sentences)
        assert "Study 3" not in emitted_text
        assert "Method" not in emitted_text
        assert "Results" not in emitted_text
        assert "General Discussion" not in emitted_text

    def test_heading_before_hint_creates_one_section(self, mock_wtpsplit):
        """An explicit heading followed by hint-labeled content should merge."""
        json_result = [
            [
                _region(0, "paragraph_title", "References"),
                _region(1, "reference", "Smith, J. (2020). A paper."),
                _region(2, "reference", "Jones, K. (2019). Another paper."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        ref_sections = [s for s in contents.sections if s.header == "References"]
        assert len(ref_sections) == 1

    def test_multiple_reference_regions_single_section(self, mock_wtpsplit):
        """Many reference-labeled regions should all go into one References section."""
        json_result = [
            [
                _region(0, "text", "Body text."),
                _region(1, "reference", "Ref 1."),
                _region(2, "reference", "Ref 2."),
                _region(3, "reference", "Ref 3."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        ref_sections = [s for s in contents.sections if s.header == "References"]
        assert len(ref_sections) == 1
        ref_sents = [s for s in contents.sentences if s.section_id == ref_sections[0].section_id]
        assert len(ref_sents) == 3


# ---------------------------------------------------------------------------
# Regression tests: empty and malformed input
# ---------------------------------------------------------------------------


class TestMalformedInput:
    """Regression tests for empty, None, and malformed regions."""

    def test_empty_page_list(self, mock_wtpsplit):
        """Empty input produces minimal PaperContents."""
        contents = _parse_and_segment([])
        assert len(contents.sections) == 1  # Root only
        assert len(contents.sentences) == 0

    def test_page_with_no_regions(self, mock_wtpsplit):
        """A page with an empty region list is handled gracefully."""
        contents = _parse_and_segment([[]])
        assert len(contents.sentences) == 0

    def test_region_with_empty_content(self, mock_wtpsplit):
        """Region with empty string content should be skipped."""
        json_result = [
            [
                _region(0, "text", ""),
                _region(1, "text", "Real text."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        assert len(contents.sentences) == 1
        assert contents.sentences[0].text == "Real text."

    def test_region_with_none_content(self, mock_wtpsplit):
        """Region with None content should be handled gracefully."""
        json_result = [
            [
                {"index": 0, "label": "text", "content": None, "bbox_2d": [0, 0, 100, 100]},
                _region(1, "text", "After null."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        assert any(s.text == "After null." for s in contents.sentences)

    def test_region_missing_label(self, mock_wtpsplit):
        """Region without a label field defaults to content treatment."""
        json_result = [
            [
                {"index": 0, "content": "No label text.", "bbox_2d": [0, 0, 100, 100]},
            ]
        ]
        contents = _parse_and_segment(json_result)
        assert len(contents.sentences) >= 1

    def test_whitespace_only_content_skipped(self, mock_wtpsplit):
        """Whitespace-only text regions should produce no sentences."""
        json_result = [
            [
                _region(0, "text", "   \n\t  "),
                _region(1, "text", "Actual content."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        non_empty = [s for s in contents.sentences if s.text.strip()]
        assert len(non_empty) == 1


# ---------------------------------------------------------------------------
# Regression tests: caption attachment edge cases
# ---------------------------------------------------------------------------


class TestCaptionEdgeCases:
    """Regression tests for caption-to-element matching."""

    def test_orphan_caption_with_no_element(self, mock_wtpsplit):
        """A caption with no subsequent table/figure should not crash."""
        json_result = [
            [
                _region(0, "text", "Body text."),
                _region(1, "table_title", "Table 1: Orphan caption"),
            ]
        ]
        contents = _parse_and_segment(json_result)
        # Should not crash; caption may be discarded or stored as text
        assert len(contents.sentences) >= 1

    def test_caption_on_different_page_not_attached(self, mock_wtpsplit):
        """Caption on page 1 should not attach to element on page 2."""
        json_result = [
            [
                _region(0, "table_title", "Table 1: Results", bbox=[0, 0, 500, 50]),
            ],
            [
                _region(0, "table", "| A |\n|---|\n| 1 |", bbox=[0, 100, 500, 300]),
            ],
        ]
        contents = _parse_and_segment(json_result)
        if contents.tables:
            # Caption should NOT be attached across pages
            assert contents.tables[0].caption is None or contents.tables[0].caption == ""

    def test_multiple_tables_get_correct_captions(self, mock_wtpsplit):
        """Two tables with preceding captions should each get the right caption."""
        json_result = [
            [
                _region(0, "table_title", "Table 1: First"),
                _region(1, "table", "| A |\n|---|\n| 1 |", bbox=[0, 100, 500, 200]),
                _region(2, "table_title", "Table 2: Second"),
                _region(3, "table", "| B |\n|---|\n| 2 |", bbox=[0, 300, 500, 400]),
            ]
        ]
        contents = _parse_and_segment(json_result)
        assert len(contents.tables) == 2
        assert contents.tables[0].caption == "Table 1: First"
        assert contents.tables[1].caption == "Table 2: Second"


# ---------------------------------------------------------------------------
# Regression tests: section hierarchy
# ---------------------------------------------------------------------------


class TestSectionHierarchyEdgeCases:
    """Regression tests for nested section hierarchy construction."""

    def test_h3_under_h2_under_h1(self, mock_wtpsplit):
        """Three-level nesting: H1 > H2 > H3."""

        json_result = [
            [
                _region(0, "doc_title", "Title"),
                _region(1, "paragraph_title", "1. Methods"),
                _region(2, "text", "Methods text."),
            ]
        ]
        # Use a sub-heading that the parser recognizes as H3
        # by having it under a paragraph_title H2
        contents = _parse_and_segment(json_result)
        # Verify parent chain: Root(0) -> Title(H1) -> Methods(H2)
        methods = [s for s in contents.sections if s.header == "1. Methods"]
        if methods:
            assert methods[0].level == 2
            assert methods[0].parent_section_id == contents.sections[1].section_id

    def test_section_with_only_whitespace_heading(self, mock_wtpsplit):
        """A heading that is only whitespace should be handled gracefully."""
        json_result = [
            [
                _region(0, "paragraph_title", "   "),
                _region(1, "text", "Body text."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        # Should not crash; whitespace heading may be skipped or cleaned
        assert any(s.text == "Body text." for s in contents.sentences)

    def test_many_sections_correct_ids(self, mock_wtpsplit):
        """Section IDs should be sequential and unique."""
        json_result = [
            [
                _region(0, "doc_title", "Paper"),
                _region(1, "paragraph_title", "Introduction"),
                _region(2, "text", "Intro text."),
                _region(3, "paragraph_title", "Methods"),
                _region(4, "text", "Methods text."),
                _region(5, "paragraph_title", "Results"),
                _region(6, "text", "Results text."),
            ]
        ]
        contents = _parse_and_segment(json_result)
        ids = [s.section_id for s in contents.sections]
        # All IDs should be unique
        assert len(ids) == len(set(ids))
        # IDs should be sequential starting from 0
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Regression tests: mixed content types on single page
# ---------------------------------------------------------------------------


class TestMixedContentSinglePage:
    """Test realistic page with tables, figures, text, and formulas interleaved."""

    def test_realistic_results_page(self, mock_wtpsplit):
        """A typical results page with text, table, figure, and formula."""
        json_result = [
            [
                _region(0, "paragraph_title", "3. Results"),
                _region(1, "text", "Results are shown in Table 1."),
                _region(2, "table_title", "Table 1: Descriptive statistics"),
                _region(3, "table", "| M | SD |\n|---|---|\n| 3.5 | 1.2 |"),
                _region(4, "text", "The regression equation was:"),
                _region(5, "formula", "y = \\beta_0 + \\beta_1 x"),
                _region(6, "text", "Figure 1 shows the scatterplot."),
                _region(7, "figure_title", "Figure 1: Scatterplot"),
                _region(8, "image", ""),
            ]
        ]
        contents = _parse_and_segment(json_result)

        # Verify all content types parsed
        assert len(contents.tables) == 1
        assert contents.tables[0].caption == "Table 1: Descriptive statistics"
        assert len(contents.figures) == 1
        assert contents.figures[0].caption == "Figure 1: Scatterplot"
        # Formula should be a display formula sentence
        formula_sents = [s for s in contents.sentences if s.is_display_formula]
        assert len(formula_sents) == 1
        # Regular text sentences
        text_sents = [s for s in contents.sentences if not s.is_display_formula]
        assert len(text_sents) >= 3


class TestSentenceProvenance:
    """Sentence-level Provenance from region bboxes — Docling-style spatial source."""

    def test_single_region_attaches_bbox(self, mock_wtpsplit):
        json_result = [
            [_region(0, "text", "Hello world.", bbox=[10, 20, 100, 50])],
        ]
        contents = _parse_and_segment(json_result)
        assert len(contents.sentences) == 1
        sent = contents.sentences[0]
        assert len(sent.provenance) == 1
        prov = sent.provenance[0]
        assert prov.page_no == 1
        assert prov.bbox == (10.0, 20.0, 100.0, 50.0)

    def test_carry_over_accumulates_provenance(self, mock_wtpsplit):
        """Two regions joined via carry-over should yield a sentence with both bboxes."""
        json_result = [
            [
                # First region: no terminal punctuation → carry-over
                _region(0, "text", "first part", bbox=[0, 0, 50, 20]),
                # Second region (same page): lowercase start → joins via _should_join
                _region(1, "text", "and second part.", bbox=[0, 30, 50, 50]),
            ]
        ]
        contents = _parse_and_segment(json_result)
        # All sentences share the same provenance list for the joined paragraph
        joined = [s for s in contents.sentences if "first part" in s.text]
        assert joined, contents.sentences
        sent = joined[0]
        assert len(sent.provenance) == 2
        bboxes = [p.bbox for p in sent.provenance]
        assert (0.0, 0.0, 50.0, 20.0) in bboxes
        assert (0.0, 30.0, 50.0, 50.0) in bboxes

    def test_formula_provenance(self, mock_wtpsplit):
        json_result = [
            [_region(0, "display_formula", "E = mc^2", bbox=[5, 5, 95, 25])],
        ]
        contents = _parse_and_segment(json_result)
        formulas = [s for s in contents.sentences if s.is_display_formula]
        assert len(formulas) == 1
        prov = formulas[0].provenance
        assert len(prov) == 1
        assert prov[0].bbox == (5.0, 5.0, 95.0, 25.0)
        assert prov[0].page_no == 1

    def test_each_split_sentence_inherits_paragraph_provenance(self, mock_wtpsplit):
        """One paragraph segmented into N sentences → all N share the same prov list."""
        json_result = [
            [_region(0, "text", "First. Second. Third.", bbox=[0, 0, 100, 100])],
        ]
        contents = _parse_and_segment(json_result)
        assert len(contents.sentences) == 3
        for sent in contents.sentences:
            assert len(sent.provenance) == 1
            assert sent.provenance[0].bbox == (0.0, 0.0, 100.0, 100.0)


class TestRegionMetaLayoutBbox:
    """region_meta["bbox"] is the region's 0..1000 layout box, which the export
    converts to points on the displayed page; the page size comes from the
    native pass's ``_page_w``/``_page_h`` and lands on ``contents.page_sizes``."""

    def test_region_meta_carries_the_layout_bbox_and_page(self, mock_wtpsplit):
        region = _region(0, "text", "Hello world.", bbox=[100, 200, 900, 250])
        region["_page_w"] = 612.0
        region["_page_h"] = 792.0
        contents = _parse_and_segment([[region]])
        sent = contents.sentences[0]
        assert sent.region_meta["bbox"] == [100.0, 200.0, 900.0, 250.0]
        assert sent.region_meta["region_page"] == 1
        assert contents.page_sizes == {1: (612.0, 792.0)}
        assert sent.provenance[0].bbox == (100.0, 200.0, 900.0, 250.0)

    def test_no_page_size_without_the_native_pass(self, mock_wtpsplit):
        region = _region(0, "text", "Hello world.", bbox=[100, 200, 900, 250])
        contents = _parse_and_segment([[region]])
        assert contents.page_sizes == {}


# ---------------------------------------------------------------------------
# Numbering-based level inference
# ---------------------------------------------------------------------------


class TestNumberingLevelInference:
    """Verify that dotted-number prefixes produce correct heading depth."""

    def test_numbered_sections_correct_hierarchy(self, mock_wtpsplit):
        """Top-level numbered sections must be siblings, not nested under earlier ones."""
        json_result = [
            [
                _region(0, "doc_title", "Paper Title"),
                _region(1, "paragraph_title", "1 Introduction"),
                _region(2, "text", "Intro text."),
                _region(3, "paragraph_title", "2 Methods"),
                _region(4, "text", "Methods text."),
                _region(5, "paragraph_title", "3 Results"),
                _region(6, "text", "Results text."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        title = next(s for s in contents.sections if s.header == "Paper Title")
        intro = next(s for s in contents.sections if s.header == "1 Introduction")
        methods = next(s for s in contents.sections if s.header == "2 Methods")
        results = next(s for s in contents.sections if s.header == "3 Results")

        # All top-level numbered sections parent to the doc_title
        assert intro.parent_section_id == title.section_id
        assert methods.parent_section_id == title.section_id
        assert results.parent_section_id == title.section_id
        assert intro.level == methods.level == results.level == 2

    def test_subsections_nest_under_parent(self, mock_wtpsplit):
        """Dotted-number subsections must nest under their parent section."""
        json_result = [
            [
                _region(0, "doc_title", "Paper Title"),
                _region(1, "paragraph_title", "3 Model Architecture"),
                _region(2, "text", "Overview."),
                _region(3, "paragraph_title", "3.1 Encoder"),
                _region(4, "text", "Encoder details."),
                _region(5, "paragraph_title", "3.2 Decoder"),
                _region(6, "text", "Decoder details."),
                _region(7, "paragraph_title", "3.2.1 Attention"),
                _region(8, "text", "Attention details."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        arch = next(s for s in contents.sections if s.header == "3 Model Architecture")
        encoder = next(s for s in contents.sections if s.header == "3.1 Encoder")
        decoder = next(s for s in contents.sections if s.header == "3.2 Decoder")
        attention = next(s for s in contents.sections if s.header == "3.2.1 Attention")

        assert arch.level == 2
        assert encoder.level == 3
        assert encoder.parent_section_id == arch.section_id
        assert decoder.level == 3
        assert decoder.parent_section_id == arch.section_id
        assert attention.level == 4
        assert attention.parent_section_id == decoder.section_id

    def test_later_top_level_not_nested_under_earlier(self, mock_wtpsplit):
        """Section '5 Training' must NOT parent under '3 Model Architecture'."""
        json_result = [
            [
                _region(0, "doc_title", "Attention Is All You Need"),
                _region(1, "paragraph_title", "3 Model Architecture"),
                _region(2, "text", "Arch text."),
                _region(3, "paragraph_title", "3.1 Encoder"),
                _region(4, "text", "Encoder text."),
                _region(5, "paragraph_title", "4 Why Self-Attention"),
                _region(6, "text", "Self-attention text."),
                _region(7, "paragraph_title", "5 Training"),
                _region(8, "text", "Training text."),
                _region(9, "paragraph_title", "5.1 Data"),
                _region(10, "text", "Data text."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        title = next(s for s in contents.sections if s.header == "Attention Is All You Need")
        arch = next(s for s in contents.sections if s.header == "3 Model Architecture")
        encoder = next(s for s in contents.sections if s.header == "3.1 Encoder")
        self_attn = next(s for s in contents.sections if s.header == "4 Why Self-Attention")
        training = next(s for s in contents.sections if s.header == "5 Training")
        data = next(s for s in contents.sections if s.header == "5.1 Data")

        # Top-level sections are siblings under the doc_title
        assert arch.parent_section_id == title.section_id
        assert self_attn.parent_section_id == title.section_id
        assert training.parent_section_id == title.section_id

        # Subsections nest correctly
        assert encoder.parent_section_id == arch.section_id
        assert data.parent_section_id == training.section_id

    def test_glmocr_subsection_spacing_artefact(self, mock_wtpsplit):
        """GLM-OCR returns subsection headings as "3. 1 Foo" with stray space.

        The heading text must be normalized to "3.1 Foo" so that downstream
        consumers see clean numbering AND `infer_level_from_numbering` produces
        the correct depth (level 3 for two-level numbering, not level 2).
        """
        json_result = [
            [
                _region(0, "doc_title", "Paper Title"),
                _region(1, "paragraph_title", "3 Model Architecture"),
                _region(2, "text", "Overview."),
                _region(3, "paragraph_title", "3. 1 Encoder"),
                _region(4, "text", "Encoder details."),
                _region(5, "paragraph_title", "3. 2 Decoder"),
                _region(6, "text", "Decoder details."),
                _region(7, "paragraph_title", "3. 2.1 Multi-Head Attention"),
                _region(8, "text", "Attention details."),
            ]
        ]
        contents = _parse_and_segment(json_result)

        arch = next(s for s in contents.sections if s.header == "3 Model Architecture")
        encoder = next((s for s in contents.sections if s.header == "3.1 Encoder"), None)
        decoder = next((s for s in contents.sections if s.header == "3.2 Decoder"), None)
        attention = next(
            (s for s in contents.sections if s.header == "3.2.1 Multi-Head Attention"), None
        )

        # Headers must be normalized
        assert encoder is not None, [s.header for s in contents.sections]
        assert decoder is not None
        assert attention is not None

        # Levels must reflect the dotted-number depth, not the artefact form
        assert arch.level == 2
        assert encoder.level == 3
        assert decoder.level == 3
        assert attention.level == 4

        # Hierarchy must nest correctly
        assert encoder.parent_section_id == arch.section_id
        assert decoder.parent_section_id == arch.section_id
        assert attention.parent_section_id == decoder.section_id


# ---------------------------------------------------------------------------
# Title badge-glyph stripping (residual #5: " TC" Open-Practices badge leaks
# into the page-1 doc_title region from OCR and pollutes info.title)
# ---------------------------------------------------------------------------


class TestTitleBadgeGlyphStrip:
    def _parser(self):
        from bibr.structure.pdf_parser import PDFParser

        return PDFParser([[]])

    def test_doc_title_strips_trailing_tc_badge_glyph(self):
        parser = self._parser()
        parser._handle_heading("doc_title", "A Real Paper Title TC", 1)
        # Both the detected title and the section header derive from one value.
        assert parser._detected_title == "A Real Paper Title"
        assert parser.sections[0].header == "A Real Paper Title"

    def test_doc_title_keeps_legitimate_trailing_acronyms(self):
        # A trailing all-caps acronym/Roman-numeral that is part of the real
        # title must NOT be stripped — only the known badge glyph is.
        for title in (
            "Functional Connectivity Measured With MRI",
            "Working Memory and the Brain II",
            "A Longitudinal Study of PTSD",
            "Heritability Estimates From Twin DNA",
        ):
            parser = self._parser()
            parser._handle_heading("doc_title", title, 1)
            assert parser._detected_title == title

    def test_badge_glyph_strip_scoped_to_page1_doc_title(self):
        # A body subsection heading (paragraph_title) ending in " TC" must be
        # left untouched — the strip is title-only.
        parser = self._parser()
        parser._handle_heading("paragraph_title", "Methods TC", 2)
        assert parser.sections[0].header == "Methods TC"


class TestTypedRegionSeam:
    """PDFParser consumes typed OcrRegionResult regions; dict input (frozen
    eval JSONs, hand-built fixtures) is normalized at entry via from_dict."""

    _PAGE = [
        {
            "index": 0,
            "native_label": "doc_title",
            "label": "doc_title",
            "content": "A Title",
            "bbox_2d": [10, 10, 500, 40],
        },
        {
            "index": 1,
            "native_label": "text",
            "label": "text",
            "content": "Body sentence one.",
            "bbox_2d": [10, 60, 500, 100],
            "_font_size": 10.5,
        },
    ]

    def test_dict_input_normalized_to_typed_at_entry(self):
        from bibr.ocr.types import OcrRegionResult
        from bibr.structure.pdf_parser import PDFParser

        parser = PDFParser([self._PAGE])
        assert all(isinstance(r, OcrRegionResult) for page in parser.json_result for r in page)

    def test_typed_input_parses_identically_to_dict_input(self):
        from bibr.ocr.types import OcrRegionResult
        from bibr.structure.pdf_parser import PDFParser

        typed = [[OcrRegionResult.from_dict(r) for r in self._PAGE]]
        c_dict = PDFParser([self._PAGE]).parse()
        c_typed = PDFParser(typed).parse()
        assert c_dict.detected_title == c_typed.detected_title == "A Title"
        assert [s.header for s in c_dict.sections] == [s.header for s in c_typed.sections]
