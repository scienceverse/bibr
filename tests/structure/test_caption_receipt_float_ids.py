"""Caption-receipt object ids must survive post-assembly float renumbering.

``_finalize_media`` freezes the caption-assignment receipt against the
printed-id reservation ("FIGURE 12" reserves ``figure:12``), and only *then*
does ``PDFParser.parse`` run the float mergers, which renumber survivors
positionally from 1. Measured at HEAD on the caption-before-image layout
below: the receipt named ``figure:12`` and ``figure:13`` in a document whose
only live figure was ``figure:1`` — every exported assignment dangled.

An absorbed panel is repointed at its survivor rather than dropped: the
receipt's whole job is to say where each printed caption ended up, and
"absorbed into figure 1" is that answer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bibr.paper_contents import (
    CaptionAssignment,
    CaptionAssignmentReceipt,
    CaptionCandidate,
    PaperFigure,
    PaperTable,
)
from bibr.structure.floats_normalize import (
    merge_figure_panels_with_remap,
    merge_table_continuations_with_remap,
    remap_caption_receipt,
)


def _region(index, label, content, bbox=None, image_b64=None):
    value = {
        "index": index,
        "label": label,
        "content": content,
        "bbox_2d": bbox or [0, 0, 100, 100],
    }
    if image_b64 is not None:
        value["image_b64"] = image_b64
    return value


def _parse(json_result):
    from bibr.structure.pdf_parser import PDFParser

    return PDFParser(json_result).parse()


def _panel_document():
    """Caption first, then its panels — the ordering the inline panel grouping
    misses (it only looks backwards from an explicit caption), so the panels
    survive to ``merge_figure_panels``. The printed "12" forces the reserved
    id away from the positional one so a stale receipt is visible."""
    return [
        [
            _region(0, "figure_title", "FIGURE 12 Panels of the thing.", bbox=[0, 0, 100, 20]),
            _region(1, "image", "", bbox=[0, 30, 100, 130], image_b64="p1"),
            _region(2, "figure_title", "A", bbox=[0, 130, 100, 140]),
            _region(3, "image", "", bbox=[0, 150, 100, 250], image_b64="p2"),
            _region(4, "figure_title", "B", bbox=[0, 250, 100, 260]),
        ]
    ]


def _live_object_ids(contents) -> set[str]:
    return {f"figure:{item.figure_id}" for item in contents.figures} | {
        f"table:{item.table_id}" for item in contents.tables
    }


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


class TestReceiptSurvivesFloatMerging:
    def test_every_receipt_object_id_resolves_to_a_live_float(self):
        contents = _parse(_panel_document())

        # The survivor keeps the printed label as its id — that is what
        # detect_xrefs resolves a body mention of "Figure 12" by.
        assert [figure.figure_id for figure in contents.figures] == [12]
        receipt = contents.caption_assignment_receipt
        assigned = [item.object_id for item in receipt.assignments if item.object_id is not None]
        assert assigned, "the fixture must produce at least one owned caption"
        assert set(assigned) <= _live_object_ids(contents)

    def test_absorbed_panel_points_at_the_surviving_figure(self):
        contents = _parse(_panel_document())
        receipt = contents.caption_assignment_receipt
        by_text = {
            candidate.text: assignment
            for candidate, assignment in zip(receipt.candidates, receipt.assignments, strict=True)
        }

        # "B" owned the panel figure that merge_figure_panels absorbed; its
        # entry must follow the survivor, not vanish and not keep the dead id.
        assert by_text["B"].object_id == "figure:12"
        assert by_text["FIGURE 12 Panels of the thing."].object_id == "figure:12"

    def test_export_emits_no_dangling_assignment_object_id(self):
        from bibr.export import export_paper_to_json
        from bibr.input.file import InputFile, InputFormat
        from bibr.models import PaperMetadata
        from bibr.paper import Paper

        contents = _parse(_panel_document())
        paper = Paper(
            input_file=InputFile(
                path=Path("/tmp/panels.pdf"),
                file_hash="hash",
                input_format=InputFormat(
                    file_extension=".pdf",
                    detected_mime_type="application/pdf",
                    file_type="pdf",
                ),
            ),
            metadata=PaperMetadata(title="Panels", doi=""),
            contents=contents,
        )
        from tests.export.conftest import extraction_block

        paper.extraction = extraction_block()
        output = export_paper_to_json(paper, validate=False)

        live = {f"figure:{item['figure_id']}" for item in output["figure"]} | {
            f"table:{item['table_id']}" for item in output["table"]
        }
        exported = [
            item["object_id"]
            for item in output["extraction"]["diagnostics"]["caption_assignment"]["assignments"]
            if item["object_id"] is not None
        ]
        assert exported
        assert set(exported) <= live


class TestMergeRemaps:
    def test_figure_remap_sends_absorbed_panels_to_their_survivor(self):
        figures = [
            _fig(12, 4, "FIGURE 12 Real caption"),
            _fig(13, 4, "A", image="img-a"),
            _fig(14, 4, "B", image="img-b"),
        ]
        merged, remap = merge_figure_panels_with_remap(figures)

        assert [figure.figure_id for figure in merged] == [12]
        assert remap == {
            "figure:12": "figure:12",
            "figure:13": "figure:12",
            "figure:14": "figure:12",
        }

    def test_figure_remap_is_empty_when_nothing_merged(self):
        figures = [_fig(3, 4, "FIGURE 1 One"), _fig(7, 5, "FIGURE 2 Two")]
        merged, remap = merge_figure_panels_with_remap(figures)

        assert [figure.figure_id for figure in merged] == [3, 7]
        assert remap == {}

    def test_table_remap_sends_continuation_pages_to_their_survivor(self):
        caption = "Table 1 Inventory"
        df1 = pd.DataFrame([["a", "1"]], columns=["Species", "Uses"])
        df2 = pd.DataFrame([["b", "2"]], columns=["Species", "Uses"])
        tables = [
            _tbl(4, 4, caption, df1),
            _tbl(5, 5, f"{caption} (Continued)", df2),
            _tbl(9, 20, "Table 2 Other"),
        ]
        merged, remap = merge_table_continuations_with_remap(tables)

        assert [table.table_id for table in merged] == [1, 2]
        assert remap == {"table:4": "table:1", "table:5": "table:1", "table:9": "table:2"}

    def test_table_remap_is_empty_when_nothing_merged(self):
        tables = [_tbl(4, 4, "Table 1 First"), _tbl(5, 9, "Table 1 Different thing")]
        merged, remap = merge_table_continuations_with_remap(tables)

        assert [table.table_id for table in merged] == [4, 5]
        assert remap == {}

    def test_remap_receipt_rewrites_only_known_ids(self):
        candidates = tuple(
            CaptionCandidate(f"caption:{i}", text, "figure", 4, None, i)
            for i, text in enumerate(("FIGURE 12 Real", "A", "orphan"))
        )
        receipt = CaptionAssignmentReceipt(
            candidates,
            (
                CaptionAssignment("caption:0", "figure:12", 8.5, ("same_page",)),
                CaptionAssignment("caption:1", "figure:13", 0.0, ("panel_evidence",)),
                CaptionAssignment("caption:2", None, 0.0, ("ambiguous",), ambiguous=True),
            ),
        )

        remapped = remap_caption_receipt(
            receipt, {"figure:12": "figure:1", "figure:13": "figure:1"}
        )

        assert [item.object_id for item in remapped.assignments] == ["figure:1", "figure:1", None]
        # Everything else is carried through untouched — the receipt is lossless.
        assert remapped.candidates == receipt.candidates
        assert remapped.assignments[0].score == 8.5
        assert remapped.assignments[0].reasons == ("same_page",)
        assert remapped.assignments[2].ambiguous is True
        assert remap_caption_receipt(receipt, {}) is receipt
