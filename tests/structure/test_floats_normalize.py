"""Post-assembly figure-panel merging and table continuation normalization.

Synthetic cases cover panels with uppercase, numeric, or empty captions and a separate numbered caption, plus repeated captions across table pages. Normalization must preserve physical float parts and their provenance."""

from __future__ import annotations

import pandas as pd

from bibr.paper_contents import PaperFigure, PaperTable
from bibr.structure.floats_normalize import merge_figure_panels, merge_table_continuations


def _fig(fid: int, page: int, caption: str | None, image: str | None = None) -> PaperFigure:
    return PaperFigure(
        figure_id=fid,
        section_id=0,
        image_b64=image,
        caption=caption,
        page_number=page,
    )


def _tbl(tid: int, page: int, caption: str | None, df: pd.DataFrame | None = None) -> PaperTable:
    if df is None:
        df = pd.DataFrame([["x", "y"]], columns=["c1", "c2"])
    return PaperTable(
        table_id=tid,
        df=df,
        tbl_html=df.to_html(index=False),
        section_id=0,
        caption=caption,
        page_number=page,
    )


class TestMergeFigurePanels:
    def test_uppercase_panel_letters_collapse_into_labeled_figure(self):
        """10.1002_mco2.132 page-4 pattern: A..F panels then 'FIGURE 1 …'."""
        figures = [
            *(_fig(i + 1, 4, letter, image=f"img{i}") for i, letter in enumerate("ABCDEF")),
            _fig(7, 4, "FIGURE 1 Identification and phenotype analysis"),
        ]
        out = merge_figure_panels(figures)
        assert len(out) == 1
        assert out[0].caption == "FIGURE 1 Identification and phenotype analysis"
        assert out[0].figure_id == 1
        # The labeled figure had no crop of its own — adopt one from a panel.
        assert out[0].image_b64 == "img0"

    def test_bare_digit_and_empty_captions_merge_too(self):
        """mco2.132 page-6 pattern: A..H panels + a bare '1' + empty caption."""
        figures = [
            _fig(1, 6, "A"),
            _fig(2, 6, None),
            _fig(3, 6, "1"),
            _fig(4, 6, "FIGURE 2 Preparation and selection"),
        ]
        out = merge_figure_panels(figures)
        assert len(out) == 1
        assert out[0].caption == "FIGURE 2 Preparation and selection"

    def test_two_groups_on_same_page_bind_to_nearest_following_label(self):
        figures = [
            _fig(1, 12, "A"),
            _fig(2, 12, "FIGURE 4 First group"),
            _fig(3, 12, "B"),
            _fig(4, 12, "FIGURE 5 Second group"),
        ]
        out = merge_figure_panels(figures)
        assert [f.caption for f in out] == ["FIGURE 4 First group", "FIGURE 5 Second group"]
        # The printed label is the id: renumbering positionally from 1 made a
        # body mention of "Figure 4" resolve to whatever landed in slot 4.
        assert [f.figure_id for f in out] == [4, 5]

    def test_panels_after_label_fall_back_to_preceding_target(self):
        figures = [
            _fig(1, 3, "Figure 2: Flow diagram"),
            _fig(2, 3, "(a)"),
            _fig(3, 3, "b"),
        ]
        out = merge_figure_panels(figures)
        assert len(out) == 1
        assert out[0].caption == "Figure 2: Flow diagram"

    def test_panels_do_not_merge_across_pages(self):
        figures = [
            _fig(1, 4, "A"),
            _fig(2, 5, "FIGURE 1 On the next page"),
        ]
        out = merge_figure_panels(figures)
        assert len(out) == 2

    def test_page1_uncaptioned_masthead_logo_is_kept(self):
        """A page-1 header-strip crop looks like a journal logo, but dropping
        it here would contradict the ownership layer's printed-id
        reconciliation (see test_pdf_parser_media_ownership). Merging never
        deletes; furniture suppression belongs upstream."""
        from bibr.paper_contents import Provenance

        logo = _fig(1, 1, None)
        logo.provenance = [Provenance(page_no=1, bbox=(700, 20, 950, 90))]  # header strip
        figures = [logo, _fig(2, 3, "Figure 1: Real figure")]
        out = merge_figure_panels(figures)
        assert [f.caption for f in out] == [None, "Figure 1: Real figure"]

    def test_page1_uncaptioned_body_image_kept(self):
        """A large mid-page uncaptioned image (e.g. graphical abstract) is
        content, not masthead furniture — only the top strip is dropped."""
        from bibr.paper_contents import Provenance

        art = _fig(1, 1, None)
        art.provenance = [Provenance(page_no=1, bbox=(100, 400, 900, 800))]
        figures = [art, _fig(2, 3, "Figure 1: Real figure")]
        out = merge_figure_panels(figures)
        assert len(out) == 2

    def test_page1_uncaptioned_without_bbox_kept(self):
        """No spatial info → cannot tell furniture from content; keep it."""
        figures = [_fig(1, 1, None), _fig(2, 3, "Figure 1: Real figure")]
        out = merge_figure_panels(figures)
        assert len(out) == 2

    def test_empty_caption_far_from_labeled_figure_not_absorbed(self):
        """Mirrors the parser contract: an uncaptioned figure far from the
        labeled figure on the same page is its own figure."""
        from bibr.paper_contents import Provenance

        far = _fig(1, 2, None)
        far.provenance = [Provenance(page_no=2, bbox=(100, 700, 500, 900))]
        target = _fig(2, 2, "Figure 1: Correct")
        target.provenance = [Provenance(page_no=2, bbox=(100, 85, 500, 300))]
        out = merge_figure_panels([far, target])
        assert len(out) == 2

    def test_uncaptioned_figure_beyond_page1_is_kept(self):
        figures = [_fig(1, 5, None)]
        out = merge_figure_panels(figures)
        assert len(out) == 1

    def test_page1_uncaptioned_kept_when_doc_has_no_labeled_figures(self):
        """An uncaptioned image may be the document's actual content (and is
        the contract of many existing parser tests); only treat it as journal
        furniture when the paper otherwise captions its figures."""
        figures = [_fig(1, 1, None)]
        out = merge_figure_panels(figures)
        assert len(out) == 1

    def test_panel_markers_without_target_are_kept(self):
        """No same-page labeled figure → nothing to merge into; keep as-is."""
        figures = [_fig(1, 7, "A"), _fig(2, 7, "B")]
        out = merge_figure_panels(figures)
        assert len(out) == 2

    def test_descriptive_captions_are_never_treated_as_panels(self):
        """'A randomized trial …' starts with a bare letter but is a caption."""
        figures = [
            _fig(1, 2, "A randomized trial of something"),
            _fig(2, 2, "Figure 1: The real one"),
        ]
        out = merge_figure_panels(figures)
        assert len(out) == 2

    def test_provenance_accumulates_on_merge(self):
        from bibr.paper_contents import Provenance

        panel = _fig(1, 4, "A")
        panel.provenance = [Provenance(page_no=4, bbox=(0, 0, 10, 10))]
        target = _fig(2, 4, "FIGURE 1 Cap")
        target.provenance = [Provenance(page_no=4, bbox=(0, 20, 10, 30))]
        out = merge_figure_panels([panel, target])
        assert len(out) == 1
        assert len(out[0].provenance) == 2

    def test_absorbed_panel_parts_are_conserved(self):
        """``parts`` records every physical crop that fed a logical figure.
        Absorbing a panel must carry its parts across, exactly as the inline
        ownership merge in ``parse_media`` does — otherwise the panel's
        page/bbox/image payload is destroyed."""
        from bibr.paper_contents import PaperFigurePart

        panel = _fig(1, 4, "A", image="img-a")
        panel.parts = [PaperFigurePart(page_number=4, bbox=(0, 0, 10, 10), image_b64="img-a")]
        target = _fig(2, 4, "FIGURE 1 Cap", image="img-main")
        target.parts = [PaperFigurePart(page_number=4, bbox=(0, 20, 10, 30), image_b64="img-main")]

        out = merge_figure_panels([panel, target])

        assert len(out) == 1
        assert len(out[0].parts) == 2
        assert [p.image_b64 for p in out[0].parts] == ["img-main", "img-a"]

    def test_absorbed_panels_composite_into_the_whole_figure_image(self):
        import base64
        import io

        from PIL import Image

        from bibr.paper_contents import PaperFigurePart

        def crop(color):
            out = io.BytesIO()
            Image.new("RGB", (10, 10), color).save(out, format="PNG")
            return base64.b64encode(out.getvalue()).decode("ascii")

        panel = _fig(1, 4, "A", image=crop("red"))
        panel.parts = [
            PaperFigurePart(page_number=4, bbox=(0, 0, 10, 10), image_b64=panel.image_b64)
        ]
        target = _fig(2, 4, "FIGURE 1 Cap", image=crop("blue"))
        target.parts = [
            PaperFigurePart(page_number=4, bbox=(0, 20, 10, 30), image_b64=target.image_b64)
        ]

        (figure,) = merge_figure_panels([panel, target])

        image = Image.open(io.BytesIO(base64.b64decode(figure.image_b64 or "")))
        assert image.size == (10, 30)


class TestMergeTableContinuations:
    def test_continued_pages_concatenate_into_first_table(self):
        """10.1186_s13002-018-0260-5 pattern: Table 1 once per page 4..6."""
        caption = "Table 1 Inventory of medicinal plants traded in the market"
        df1 = pd.DataFrame([["a", "1"]], columns=["Species", "Uses"])
        df2 = pd.DataFrame([["b", "2"]], columns=["Species", "Uses"])
        df3 = pd.DataFrame([["c", "3"]], columns=["Species", "Uses"])
        tables = [
            _tbl(1, 4, caption, df1),
            _tbl(2, 5, f"{caption} (Continued)", df2),
            _tbl(3, 6, f"{caption} (Continued)", df3),
            _tbl(4, 20, "Table 2 The used parts of medicinal plants"),
        ]
        out = merge_table_continuations(tables)
        assert len(out) == 2
        assert out[0].caption == caption
        assert out[0].df["Species"].tolist() == ["a", "b", "c"]
        assert out[0].page_number == 4
        assert [t.table_id for t in out] == [1, 2]

    def test_continued_marker_alone_suffices_with_matching_label(self):
        tables = [
            _tbl(1, 4, "Table 3 Full caption text"),
            _tbl(2, 5, "Table 3 (Continued)"),
        ]
        out = merge_table_continuations(tables)
        assert len(out) == 1
        assert len(out[0].df) == 2

    def test_a_different_dotted_label_is_not_a_continuation(self):
        """Table 3.2 shares the number 3 with Table 3.1, not the label."""
        tables = [
            _tbl(1, 4, "Table 3.1 Participants"),
            _tbl(2, 5, "Table 3.2 Participants (Continued)"),
        ]
        out = merge_table_continuations(tables)
        assert [t.caption for t in out] == [
            "Table 3.1 Participants",
            "Table 3.2 Participants (Continued)",
        ]

    def test_same_label_without_marker_not_merged(self):
        """Two genuinely different tables that happen to share a label."""
        tables = [
            _tbl(1, 4, "Table 1 First thing"),
            _tbl(2, 9, "Table 1 A completely different thing"),
        ]
        out = merge_table_continuations(tables)
        assert len(out) == 2

    def test_repeated_caption_without_marker_not_merged(self):
        """A verbatim-repeated caption is not sufficient evidence: the inline
        ownership layer refuses to merge across an intervening section
        heading, and this pass cannot see that."""
        tables = [
            _tbl(1, 4, "Table 7. Values"),
            _tbl(2, 5, "Table 7. Values"),
        ]
        out = merge_table_continuations(tables)
        assert len(out) == 2

    def test_column_count_mismatch_is_not_merged(self):
        df_wide = pd.DataFrame([["a", "b", "c"]], columns=["x", "y", "z"])
        tables = [
            _tbl(1, 4, "Table 2 Caption"),
            _tbl(2, 5, "Table 2 Caption (Continued)", df_wide),
        ]
        out = merge_table_continuations(tables)
        assert len(out) == 2

    def test_headerless_continuation_recovers_promoted_header_row(self):
        """A continuation page with no header row has its first data row
        promoted to column names by the HTML parser; positional concat must
        restore that row as data."""
        df1 = pd.DataFrame([["a", "1"]], columns=["Species", "Uses"])
        df2 = pd.DataFrame([["e", "5"]], columns=["d", "4"])  # promoted data row
        tables = [
            _tbl(1, 4, "Table 1 Inventory of plants", df1),
            _tbl(2, 5, "Table 1 Inventory of plants (Continued)", df2),
        ]
        # Preserve the actual headerless source. df.to_html() would invent <th>
        # cells and make this indistinguishable from a changed explicit header.
        tables[
            1
        ].tbl_html = "<table><tr><td>d</td><td>4</td></tr><tr><td>e</td><td>5</td></tr></table>"
        out = merge_table_continuations(tables)
        assert len(out) == 1
        assert out[0].df["Species"].tolist() == ["a", "d", "e"]
        assert out[0].df["Uses"].tolist() == ["1", "4", "5"]

    def test_unlabeled_tables_are_untouched(self):
        tables = [_tbl(1, 4, None), _tbl(2, 5, None)]
        out = merge_table_continuations(tables)
        assert len(out) == 2


class TestPrintedLabelsSurviveMerging:
    """``_reconcile_object_ids`` reserves the printed number as the object id
    and runs before these mergers, so a positional renumber from 1 silently
    repointed every body mention: with a caption-less panel absorbed into
    FIGURE 2, a mention of "Figure 2" resolved to the figure captioned
    FIGURE 3."""

    def test_absorbed_leading_panel_does_not_shift_the_survivors(self):
        figures = [
            _fig(1, 4, "A", image="panel"),
            _fig(2, 4, "FIGURE 2 Second"),
            _fig(3, 5, "FIGURE 3 Third"),
        ]

        out = merge_figure_panels(figures)

        assert [f.caption for f in out] == ["FIGURE 2 Second", "FIGURE 3 Third"]
        assert [f.figure_id for f in out] == [2, 3]

    def test_unlabelled_survivors_take_the_unclaimed_numbers(self):
        figures = [
            _fig(1, 4, None, image="orphan"),
            _fig(2, 9, "FIGURE 1 First"),
            _fig(3, 9, "A", image="panel"),
        ]

        out = merge_figure_panels(figures)

        assert [f.figure_id for f in out] == [2, 1]

    def test_continuation_merge_keeps_the_printed_table_numbers(self):
        caption = "Table 3 Inventory"
        df1 = pd.DataFrame([["a", "1"]], columns=["Species", "Uses"])
        df2 = pd.DataFrame([["b", "2"]], columns=["Species", "Uses"])
        tables = [
            _tbl(3, 4, caption, df1),
            _tbl(4, 5, f"{caption} (Continued)", df2),
            _tbl(5, 20, "Table 4 Other"),
        ]

        out = merge_table_continuations(tables)

        assert [t.table_id for t in out] == [3, 4]

    def test_continuation_parts_are_conserved(self):
        """Each continuation page is a physical table region of its own. The
        concat keeps its rows, so ``parts`` must keep its page/bbox/HTML too."""
        from bibr.paper_contents import PaperTablePart

        caption = "Table 1 Inventory"
        df1 = pd.DataFrame([["a", "1"]], columns=["Species", "Uses"])
        df2 = pd.DataFrame([["b", "2"]], columns=["Species", "Uses"])
        first = _tbl(1, 4, caption, df1)
        first.parts = [
            PaperTablePart(page_number=4, bbox=(0, 0, 10, 10), tbl_html=first.tbl_html, df=df1)
        ]
        second = _tbl(2, 5, f"{caption} (Continued)", df2)
        second.parts = [
            PaperTablePart(page_number=5, bbox=(0, 0, 10, 10), tbl_html=second.tbl_html, df=df2)
        ]

        out = merge_table_continuations([first, second])

        assert len(out) == 1
        assert [p.page_number for p in out[0].parts] == [4, 5]
