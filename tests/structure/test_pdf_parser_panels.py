"""Panel-aware figure grouping and loose table-caption routing.

Root causes exercised here:

1. PP-DocLayoutV3 emits one ``image`` region per sub-panel PLUS one
   ``figure_title`` region per panel letter ("(a)", "(b) Without BN", …),
   interleaved in reading order. Panel letters must NOT split a single
   figure into N objects, and the real "Figure N: …" caption (arriving
   last) must attach to the merged figure.

2. Some OCR backends emit table captions as ``figure_title`` regions with a
   space ("Table 1 Overview") or pipe ("Table 1 | Overview") separator
   instead of a colon. Because the region is already a caption, the
   table-vs-figure discriminator must be loose (no separator required).

Caption composition choice (documented for the tests): bare panel markers
like "(a)"/"(b)" are DROPPED; descriptive panel labels like "(b) Without
BN" are PRESERVED and appended to the real caption as
``"<real caption> | <panel>; <panel>"``.
"""

from __future__ import annotations


def _region(index, label, content, bbox=None):
    return {
        "index": index,
        "label": label,
        "content": content,
        "bbox_2d": bbox or [0, 0, 100, 100],
    }


def _parse(json_result):
    from bibr.structure.pdf_parser import PDFParser

    return PDFParser(json_result).parse()


# ---------------------------------------------------------------------------
# A. Panel-aware figure grouping
# ---------------------------------------------------------------------------


class TestPanelGrouping:
    def test_bare_panels_collapse_to_one_figure(self):
        """4 images + interleaved "(a)".."(d)" + final real caption → 1 figure."""
        json_result = [
            [
                _region(0, "image", ""),
                _region(1, "figure_title", "(a)"),
                _region(2, "image", ""),
                _region(3, "figure_title", "(b)"),
                _region(4, "image", ""),
                _region(5, "figure_title", "(c)"),
                _region(6, "image", ""),
                _region(7, "figure_title", "(d)"),
                _region(8, "figure_title", "Figure 1: Full caption."),
            ]
        ]
        contents = _parse(json_result)

        assert len(contents.figures) == 1
        assert contents.figures[0].caption == "Figure 1: Full caption."
        assert "Figure 1" in contents.figures[0].caption

    def test_descriptive_panels_preserved_and_two_figures_split(self):
        """batch_norm shape: 3 panels + real caption, then a second figure."""
        json_result = [
            [
                _region(0, "image", ""),
                _region(1, "figure_title", "(a)"),
                _region(2, "image", ""),
                _region(3, "figure_title", "(b) Without BN"),
                _region(4, "image", ""),
                _region(5, "figure_title", "(c) With BN"),
                _region(6, "figure_title", "Figure 1: BN comparison."),
                _region(7, "image", ""),
                _region(8, "figure_title", "Figure 2: Second figure."),
            ]
        ]
        contents = _parse(json_result)

        assert len(contents.figures) == 2

        fig1 = contents.figures[0]
        assert fig1.caption.startswith("Figure 1: BN comparison.")
        assert "Without BN" in fig1.caption
        assert "With BN" in fig1.caption

        fig2 = contents.figures[1]
        assert fig2.caption == "Figure 2: Second figure."

    def test_lone_bare_panel_before_real_caption(self):
        """A single panel image + bare "(a)" + real caption still yields 1 figure."""
        json_result = [
            [
                _region(0, "image", ""),
                _region(1, "figure_title", "(a)"),
                _region(2, "figure_title", "Figure 5: Only one panel."),
            ]
        ]
        contents = _parse(json_result)

        assert len(contents.figures) == 1
        assert contents.figures[0].caption == "Figure 5: Only one panel."


# ---------------------------------------------------------------------------
# B. Loose table-caption routing (caption already labelled by the model)
# ---------------------------------------------------------------------------


class TestLooseTableCaptionRouting:
    def test_table_caption_space_separator_via_figure_title(self):
        json_result = [
            [
                _region(0, "figure_title", "Table 1 Overview of results", bbox=[0, 0, 100, 30]),
                _region(1, "table", "| A | B |\n|---|---|\n| 1 | 2 |", bbox=[0, 35, 100, 100]),
            ]
        ]
        contents = _parse(json_result)

        assert len(contents.figures) == 0
        assert len(contents.tables) == 1
        assert contents.tables[0].caption == "Table 1 Overview of results"

    def test_table_caption_pipe_separator_via_figure_title(self):
        json_result = [
            [
                _region(0, "figure_title", "Table 1 | Overview", bbox=[0, 0, 100, 30]),
                _region(1, "table", "| A | B |\n|---|---|\n| 1 | 2 |", bbox=[0, 35, 100, 100]),
            ]
        ]
        contents = _parse(json_result)

        assert len(contents.figures) == 0
        assert len(contents.tables) == 1
        assert contents.tables[0].caption == "Table 1 | Overview"


# ---------------------------------------------------------------------------
# Content-rescue: prose "Table N …" must stay body text
# ---------------------------------------------------------------------------


class TestContentRescueRegression:
    def test_prose_table_reference_stays_body(self):
        json_result = [
            [
                _region(0, "text", "Table 7 shows the results."),
            ]
        ]
        contents = _parse(json_result)

        assert len(contents.tables) == 0
        assert len(contents.figures) == 0

    def test_pipe_separated_table_caption_in_content_is_rescued(self):
        """Content region "Table 2 | Foo" (pipe) is rescued to a table caption."""
        json_result = [
            [
                _region(0, "table", "| A |\n|---|\n| 1 |", bbox=[0, 0, 100, 40]),
                _region(1, "text", "Table 2 | Summary stats", bbox=[0, 45, 100, 60]),
            ]
        ]
        contents = _parse(json_result)

        assert len(contents.tables) == 1
        assert contents.tables[0].caption == "Table 2 | Summary stats"


# ---------------------------------------------------------------------------
# Regression: ordinary single figure/caption path is unchanged
# ---------------------------------------------------------------------------


class TestSingleFigureRegression:
    def test_single_figure_normal_caption(self):
        json_result = [
            [
                _region(0, "image", ""),
                _region(1, "figure_title", "Figure 3: A caption"),
            ]
        ]
        contents = _parse(json_result)

        assert len(contents.figures) == 1
        assert contents.figures[0].caption == "Figure 3: A caption"

    def test_figure_xref_still_detected(self):
        """A "Figure 1" mention still produces a figure xref when the id exists."""
        from bibr.paper_contents import PaperFigure, PaperSentence
        from bibr.structure.xref_utils import detect_xrefs

        figures = [PaperFigure(figure_id=1, section_id=1, image_b64=None, caption="Figure 1: X")]
        sentences = [
            PaperSentence(
                text_id=1,
                text="As shown in Figure 1, results improved.",
                section_id=1,
                paragraph_id=1,
            )
        ]
        xrefs = detect_xrefs(sentences, [], figures)

        assert any(x.xref_type == "figure" and x.xref_id == 1 for x in xrefs)


# ---------------------------------------------------------------------------
# C. Post-assembly normalization (floats_normalize wired into parse())
# ---------------------------------------------------------------------------


class TestPostAssemblyFloatNormalization:
    def test_uppercase_letter_panels_collapse_after_parse(self):
        """Panels retro-captioned "A"/"B" (uppercase — invisible to the inline
        lowercase grouping) plus a final uncaptioned crop that receives the
        real caption must come out as ONE figure (10.1002_mco2.132 pattern)."""
        json_result = [
            [
                _region(0, "image", ""),
                _region(1, "figure_title", "A"),
                _region(2, "image", ""),
                _region(3, "figure_title", "B"),
                _region(4, "image", ""),
                _region(5, "figure_title", "FIGURE 1 Real caption text."),
            ]
        ]
        contents = _parse(json_result)

        assert len(contents.figures) == 1
        assert contents.figures[0].caption == "FIGURE 1 Real caption text."
        assert contents.figures[0].figure_id == 1

    def test_continued_table_pages_collapse_after_parse(self):
        """Per-page "(Continued)" tables merge into one
        (10.1186_s13002-018-0260-5 pattern)."""
        html = (
            "<table><tr><th>Species</th><th>Uses</th></tr><tr><td>{a}</td><td>{b}</td></tr></table>"
        )
        json_result = [
            [
                _region(0, "figure_title", "Table 1 Inventory of plants"),
                _region(1, "table", html.format(a="alpha", b="1")),
            ],
            [
                _region(0, "figure_title", "Table 1 Inventory of plants (Continued)"),
                _region(1, "table", html.format(a="beta", b="2")),
            ],
        ]
        contents = _parse(json_result)

        assert len(contents.tables) == 1
        assert contents.tables[0].caption == "Table 1 Inventory of plants"
        assert contents.tables[0].df["Species"].tolist() == ["alpha", "beta"]
