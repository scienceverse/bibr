"""Captions and footnotes are not sections, and every structural id is a position.

The parsers give each caption and footnote a synthetic section of their own;
the export drops those, keeps the sentences in ``text`` with no section, and
points ``figure``/``table``/``footnote`` rows at them. Section, figure and
table ids are renumbered 1..n in document order.
"""

from __future__ import annotations

import pandas as pd

from bibr.export.json_export import _export_paper_payload
from bibr.paper_contents import (
    CanonicalSection,
    PaperFigure,
    PaperFigurePart,
    PaperSection,
    PaperSentence,
    PaperTable,
    PaperXref,
)
from tests.export.conftest import as_parsed, extraction_block


def test_captions_and_footnotes_are_not_sections(demo_paper):
    as_parsed(demo_paper)
    payload = _export_paper_payload(demo_paper)

    assert [(s["section_id"], s["header"]) for s in payload["section"]] == [
        (1, "Introduction"),
        (2, "Results"),
    ]
    assert [(t["text_id"], t["section_id"]) for t in payload["text"]] == [
        (1, 1),
        (2, 2),
        (3, None),
        (4, None),
        (5, None),
    ]
    figure, table = payload["figure"][0], payload["table"][0]
    assert (figure["section_id"], figure["text_id"]) == (2, 3)
    assert (table["section_id"], table["text_id"]) == (2, 4)
    assert payload["footnote"] == [{"footnote_id": 1, "label": "*", "text_id": 5}]

    targets = {x["xref_type"]: x["target_id"] for x in payload["xref"]}
    assert targets == {"bib": 1, "table": 1, "figure": 1, "foot": 1}
    classified = payload["extraction"]["diagnostics"]["section_classification"]
    assert [row["section_id"] for row in classified] == [1, 2]
    codes = {issue["code"] for issue in payload["extraction"]["validation"]["issues"]}
    assert not {code for code in codes if "DANGLING" in code}


def test_a_float_without_a_caption_row_points_at_no_text(export_payload):
    """A figure or table the parsers never gave a caption section (or no
    caption) keeps its section and has no caption row."""
    assert payload_row(export_payload, "figure")["text_id"] is None
    assert payload_row(export_payload, "figure")["section_id"] == 2
    assert export_payload["footnote"] == []


def payload_row(payload: dict, table: str) -> dict:
    (row,) = payload[table]
    return row


def test_float_ids_are_positions_in_document_order(demo_paper):
    """PDF numbers floats by the printed number, leaving a gap where one was
    missed; the export numbers them 1..n by page, and everything that names a
    float follows."""
    contents = demo_paper.contents
    first = contents.figures[0]  # printed "Figure 1", page 2
    first.parts = [PaperFigurePart(page_number=2, bbox=(0, 0, 10, 10), image_b64=None)]
    third = PaperFigure(3, 2, None, "Figure 3. Later", 5, [])
    third.parts = [PaperFigurePart(page_number=5, bbox=(0, 0, 10, 10), image_b64=None)]
    early = PaperFigure(7, 1, None, "Figure 7. On page one", 1, [])
    contents.figures = [first, third, early]
    contents.xrefs.append(PaperXref(xref_id=3, xref_type="figure", contents="Figure 3", text_id=2))

    payload = _export_paper_payload(demo_paper)

    assert [(f["figure_id"], f["caption"]) for f in payload["figure"]] == [
        (1, "Figure 7. On page one"),
        (2, "Figure 1. Plot"),
        (3, "Figure 3. Later"),
    ]
    (xref,) = [x for x in payload["xref"] if x["xref_type"] == "figure"]
    assert xref["target_id"] == 3
    parts = payload["extraction"]["float_parts"]
    assert [(p["object_type"], p["object_id"], p["page_number"]) for p in parts] == [
        ("figure", 2, 2),
        ("figure", 3, 5),
    ]


def test_section_ids_are_positions_in_document_order(demo_paper):
    """A section appended late (an unheaded abstract found after the body) is
    numbered where its text is, and links to sections follow."""
    contents = demo_paper.contents
    contents.sections += [
        PaperSection(
            section_id=7,
            header="Details",
            level=2,
            parent_section_id=2,
            section_type=CanonicalSection.RESULTS,
        ),
        PaperSection(
            section_id=9,
            header=None,
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.ABSTRACT,
            header_is_synthetic=True,
        ),
    ]
    contents.sentences = [
        PaperSentence(text_id=1, text="We summarize.", section_id=9, paragraph_id=1),
        PaperSentence(text_id=2, text="We measured the thing.", section_id=1, paragraph_id=2),
        PaperSentence(
            text_id=3, text="It replicated prior work [1].", section_id=2, paragraph_id=3
        ),
        PaperSentence(text_id=4, text="More detail.", section_id=7, paragraph_id=4),
    ]
    contents.xrefs = []
    contents.links = []
    contents.equations = []

    payload = _export_paper_payload(demo_paper)

    assert [(s["section_id"], s["header"], s["parent_section_id"]) for s in payload["section"]] == [
        (1, None, None),
        (2, "Introduction", None),
        (3, "Results", None),
        (4, "Details", 3),
    ]
    assert [t["section_id"] for t in payload["text"]] == [1, 2, 3, 4]
    assert payload_row(payload, "table")["section_id"] == 3


def test_a_heading_without_text_keeps_its_place(demo_paper):
    """A heading with no text of its own is numbered where it is printed: a
    "Methods" heading whose paragraphs sit in flat subsections stays behind the
    Abstract and Introduction that ``implicit_sections`` cut from the title's
    text and inserted after it, though those carry the highest section ids."""
    contents = demo_paper.contents
    root, introduction, results = contents.sections
    introduction.section_id = 9
    introduction.header_is_synthetic = True
    results.section_id = 5
    contents.sections = [
        root,
        PaperSection(
            section_id=1,
            header="Paper Title",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.TITLE,
        ),
        PaperSection(
            section_id=8,
            header="Abstract",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.ABSTRACT,
            header_is_synthetic=True,
        ),
        introduction,
        PaperSection(
            section_id=3,
            header="Methods",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.METHODS,
        ),
        PaperSection(
            section_id=4,
            header="Participants",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.METHODS,
        ),
        results,
    ]
    contents.sentences = [
        PaperSentence(text_id=1, text="We summarize.", section_id=8, paragraph_id=1),
        PaperSentence(text_id=2, text="We ask.", section_id=9, paragraph_id=2),
        PaperSentence(text_id=3, text="Forty took part.", section_id=4, paragraph_id=3),
        PaperSentence(text_id=4, text="It replicated.", section_id=5, paragraph_id=4),
    ]
    contents.xrefs = []
    contents.links = []
    contents.equations = []

    payload = _export_paper_payload(demo_paper)

    assert [s["header"] for s in payload["section"]] == [
        "Paper Title",
        "Abstract",
        "Introduction",
        "Methods",
        "Participants",
        "Results",
    ]


def test_doi_candidates_name_export_sections(demo_paper):
    """A DOI read from a caption keeps its section_type but has no section; one
    read from the body names the body section's export id."""
    as_parsed(demo_paper)
    # A body section listed after the synthetic ones, as late stages add them.
    demo_paper.contents.sections.append(
        PaperSection(
            section_id=10,
            header="Acknowledgments",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.ACKNOWLEDGMENT,
        )
    )
    demo_paper.contents.sentences.append(
        PaperSentence(text_id=6, text="We thank them.", section_id=10, paragraph_id=6)
    )

    def candidate(section_id, section_type):
        return {
            "raw": "10.1234/x",
            "normalized": "10.1234/x",
            "source_kind": "sentence",
            "page": 1,
            "section_id": section_id,
            "section_type": section_type,
            "region_index": None,
            "region_type": None,
            "text_id": 2,
            "marker_kind": "article_doi",
            "repeated_header_footer_count": 0,
            "semantic_context": "body",
            "selection_tier": 1,
            "rejection_reason": None,
        }

    demo_paper.extraction = extraction_block(
        identity={
            "receipt": {
                "selected": candidate(10, "acknowledgment"),
                "candidates": [candidate(10, "acknowledgment"), candidate(3, "figure")],
                "issue_codes": [],
            }
        }
    )
    receipt = _export_paper_payload(demo_paper)["extraction"]["identity"]["receipt"]
    assert receipt["selected"]["section_id"] == 3
    assert [(c["section_id"], c["section_type"]) for c in receipt["candidates"]] == [
        (3, "acknowledgment"),
        (None, "figure"),
    ]
    # The paper object keeps the pipeline's ids.
    assert demo_paper.extraction["identity"]["receipt"]["selected"]["section_id"] == 10


def test_caption_receipt_names_export_floats(demo_paper):
    from bibr.paper_contents import (
        CaptionAssignment,
        CaptionAssignmentReceipt,
    )

    contents = demo_paper.contents
    contents.figures = [
        PaperFigure(4, 2, None, "Figure 4. Only", 2, []),
    ]
    contents.tables[0].table_id = 2
    contents.xrefs = []
    contents.caption_assignment_receipt = CaptionAssignmentReceipt(
        candidates=(),
        assignments=(
            CaptionAssignment(caption_id="c1", object_id="figure:4", score=1.0, reasons=()),
            CaptionAssignment(caption_id="c2", object_id="table:2", score=1.0, reasons=()),
            CaptionAssignment(caption_id="c3", object_id="figure:9", score=1.0, reasons=()),
            CaptionAssignment(caption_id="c4", object_id=None, score=0.0, reasons=()),
        ),
    )
    receipt = _export_paper_payload(demo_paper)["extraction"]["diagnostics"]["caption_assignment"]
    assert [a["object_id"] for a in receipt["assignments"]] == [
        "figure:1",
        "table:1",
        None,
        None,
    ]


def test_table_without_section_is_null(demo_paper):
    demo_paper.contents.tables = [
        PaperTable(1, pd.DataFrame([["1"]], columns=["A"]), "<table></table>", 0, None, None, [])
    ]
    assert payload_row(_export_paper_payload(demo_paper), "table")["section_id"] is None
