"""Frozen-layout regressions for caption ownership and multipart media."""

from __future__ import annotations

from dataclasses import fields

import pytest


def _region(index, label, content="", bbox=None, image_b64=None):
    value = {
        "index": index,
        "label": label,
        "content": content,
        "bbox_2d": bbox,
    }
    if image_b64 is not None:
        value["image_b64"] = image_b64
    return value


def _parse(pages):
    from bibr.structure.pdf_parser import PDFParser

    return PDFParser(pages).parse()


def test_part_models_are_trailing_and_preserve_physical_payloads():
    from bibr.paper_contents import PaperFigure, PaperFigurePart, PaperTable, PaperTablePart

    assert [item.name for item in fields(PaperFigurePart)] == [
        "page_number",
        "bbox",
        "image_b64",
        "provenance",
    ]
    assert [item.name for item in fields(PaperTablePart)] == [
        "page_number",
        "bbox",
        "tbl_html",
        "df",
        "provenance",
    ]
    assert fields(PaperFigure)[-1].name == "parts"
    assert fields(PaperTable)[-1].name == "parts"


def test_ord29_page_one_decoration_does_not_become_a_tenth_scholarly_figure():
    pages = [[] for _ in range(11)]
    pages[0] = [_region(2, "image", bbox=[82, 150, 205, 165])]
    pages[8] = [_region(0, "chart", bbox=[50, 100, 950, 800], image_b64="p9")]
    pages[9] = [
        _region(
            i,
            "chart",
            bbox=[
                50 + 450 * (i % 2),
                50 + 450 * (i // 2),
                480 + 450 * (i % 2),
                440 + 450 * (i // 2),
            ],
            image_b64=f"p10-{i}",
        )
        for i in range(4)
    ]
    pages[10] = [
        _region(
            i,
            "chart",
            bbox=[
                50 + 450 * (i % 2),
                50 + 450 * (i // 2),
                480 + 450 * (i % 2),
                440 + 450 * (i // 2),
            ],
            image_b64=f"p11-{i}",
        )
        for i in range(4)
    ]

    contents = _parse(pages)

    assert len(contents.figures) == 9
    assert [figure.figure_id for figure in contents.figures] == list(range(1, 10))
    assert [figure.parts[0].image_b64 for figure in contents.figures] == [
        "p9",
        "p10-0",
        "p10-1",
        "p10-2",
        "p10-3",
        "p11-0",
        "p11-1",
        "p11-2",
        "p11-3",
    ]


def test_ord121_containment_caption_variants_are_evidence_not_next_ownership():
    pages = [[] for _ in range(19)]
    pages[18] = [
        _region(1, "image", bbox=[325, 97, 835, 243], image_b64="figure-4"),
        _region(
            2,
            "figure_title",
            "Fig 4. Improved topology of full-bridge LLC converter. "
            "This is the improved equivalent main circuit.",
            bbox=[322, 253, 861, 266],
        ),
        _region(
            3,
            "figure_title",
            "Fig 4. Improved topology of full-bridge LLC converter. "
            "This is the improved equivalent main circuit.\n"
            "https://doi.org/10.1371/journal.pone.0205904.g004",
            bbox=[320, 252, 868, 286],
        ),
        _region(
            4,
            "figure_title",
            "https://doi.org/10.1371/journal.pone.0205904.g004",
            bbox=[322, 271, 579, 285],
        ),
        _region(7, "paragraph_title", "5. 1 Analysis on the steady state", [322, 405, 582, 422]),
        _region(10, "image", bbox=[323, 562, 835, 852], image_b64="figure-5"),
        _region(
            11,
            "figure_title",
            "Fig 5. Simplified equivalent model of MOSFET. This is the illustration of MOSFET.",
            bbox=[322, 864, 771, 878],
        ),
    ]

    contents = _parse(pages)

    assert len(contents.figures) == 2
    assert contents.figures[0].caption == (
        "Fig 4. Improved topology of full-bridge LLC converter. "
        "This is the improved equivalent main circuit."
    )
    assert contents.figures[1].caption == (
        "Fig 5. Simplified equivalent model of MOSFET. This is the illustration of MOSFET."
    )
    receipt = contents.caption_assignment_receipt
    assert receipt is not None
    assert [candidate.text for candidate in receipt.candidates] == [
        "Fig 4. Improved topology of full-bridge LLC converter. "
        "This is the improved equivalent main circuit.",
        "Fig 4. Improved topology of full-bridge LLC converter. "
        "This is the improved equivalent main circuit.\n"
        "https://doi.org/10.1371/journal.pone.0205904.g004",
        "https://doi.org/10.1371/journal.pone.0205904.g004",
        "Fig 5. Simplified equivalent model of MOSFET. This is the illustration of MOSFET.",
    ]
    duplicates = [item for item in receipt.assignments if item.object_id is None]
    assert len(duplicates) == 2
    assert all(
        any(reason.startswith("duplicate_of:") for reason in item.reasons) for item in duplicates
    )
    assert {issue.code for issue in contents.structure_validation_issues} == {
        "VAL_CAPTION_DUPLICATE"
    }
    duplicate_issue = contents.structure_validation_issues[0]
    assert duplicate_issue.message == "2 duplicate caption region(s) retained as evidence"
    assert duplicate_issue.count == 2
    assert len(duplicate_issue.evidence_ids) == 2

    from pathlib import Path

    from bibr.export import export_paper_to_json
    from bibr.input.file import InputFile, InputFormat
    from bibr.models import PaperMetadata
    from bibr.paper import Paper

    paper = Paper(
        input_file=InputFile(
            path=Path("/tmp/ord121.pdf"),
            file_hash="ord121",
            input_format=InputFormat(
                file_extension=".pdf",
                detected_mime_type="application/pdf",
                file_type="pdf",
            ),
        ),
        metadata=PaperMetadata(title="ord121", doi=""),
        contents=contents,
    )
    assert "VAL_CAPTION_DUPLICATE" in {issue.code for issue in paper.validation_issues}
    exported = export_paper_to_json(paper)
    assert "VAL_CAPTION_DUPLICATE" in {
        issue["code"] for issue in exported["extraction"]["validation"]["issues"]
    }


def test_doi_only_caption_does_not_bridge_different_explicit_figure_numbers():
    doi = "https://doi.org/10.1000/shared"
    contents = _parse(
        [
            [
                _region(0, "chart", bbox=[0, 100, 500, 200], image_b64="alpha"),
                _region(1, "chart", bbox=[0, 220, 500, 320], image_b64="beta"),
                _region(
                    2,
                    "figure_title",
                    f"Figure 1. Alpha\n{doi}",
                    bbox=[0, 325, 500, 365],
                ),
                _region(3, "figure_title", doi, bbox=[0, 325, 500, 365]),
                _region(
                    4,
                    "figure_title",
                    f"Figure 2. Beta\n{doi}",
                    bbox=[0, 325, 500, 365],
                ),
            ]
        ]
    )

    assert contents.figures[0].caption == "Figure 1. Alpha"
    assert contents.figures[1].caption.startswith("Figure 2. Beta")
    receipt = contents.caption_assignment_receipt
    assignments = list(zip(receipt.candidates, receipt.assignments, strict=True))
    alpha_assignment = next(
        assignment for candidate, assignment in assignments if candidate.text.startswith("Figure 1")
    )
    beta_assignment = next(
        assignment for candidate, assignment in assignments if candidate.text.startswith("Figure 2")
    )
    doi_assignment = next(
        assignment for candidate, assignment in assignments if candidate.text == doi
    )
    assert alpha_assignment.object_id == "figure:1"
    assert beta_assignment.object_id == "figure:2"
    assert not any(reason.startswith("duplicate_of:") for reason in beta_assignment.reasons)
    assert doi_assignment.object_id is None
    assert any(reason.startswith("duplicate_of:") for reason in doi_assignment.reasons)
    duplicate_issue = next(
        issue
        for issue in contents.structure_validation_issues
        if issue.code == "VAL_CAPTION_DUPLICATE"
    )
    assert duplicate_issue.count == 1


def test_same_number_overlapping_prefix_caption_is_duplicate_evidence():
    contents = _parse(
        [
            [
                _region(0, "chart", bbox=[0, 100, 500, 300], image_b64="figure"),
                _region(
                    1,
                    "figure_title",
                    "Figure 1. Base caption",
                    bbox=[0, 305, 500, 345],
                ),
                _region(
                    2,
                    "figure_title",
                    "Figure 1. Base caption with additional detail",
                    bbox=[0, 305, 500, 345],
                ),
            ]
        ]
    )

    assert contents.figures[0].caption == "Figure 1. Base caption with additional detail"
    receipt = contents.caption_assignment_receipt
    assignments = {
        candidate.text: assignment
        for candidate, assignment in zip(receipt.candidates, receipt.assignments, strict=True)
    }
    canonical = assignments["Figure 1. Base caption with additional detail"]
    duplicate = assignments["Figure 1. Base caption"]
    assert canonical.object_id == "figure:1"
    assert duplicate.object_id is None
    assert f"duplicate_of:{canonical.caption_id}" in duplicate.reasons
    assert "target_contention" not in duplicate.reasons


def test_ord237_rotated_continued_table_groups_before_caption_contention():
    pages = [[] for _ in range(5)]
    pages[3] = [
        _region(
            2,
            "figure_title",
            "Table 1 | Overview of subdomains showing the number of academic studies and "
            "grey literature sources reviewed",
            bbox=[70, 334, 90, 936],
        ),
        _region(
            3,
            "table",
            "| Domain | Sub-domain | Definition | Academic indicators |\n"
            "|---|---|---|---|\n| Monitoring | Governance | First | 10 |",
            bbox=[73, 56, 930, 932],
        ),
    ]
    pages[4] = [
        _region(
            2,
            "figure_title",
            "Table 1 (continued) | Overview of subdomains showing the number of academic "
            "studies and grey literature sources reviewed",
            bbox=[70, 273, 90, 937],
        ),
        _region(
            3,
            "table",
            "| Domain | Sub-domain | Definition | Academic indicators |\n"
            "|---|---|---|---|\n| Action | Equity | Second | 20 |",
            bbox=[73, 59, 720, 933],
        ),
    ]

    contents = _parse(pages)

    assert len(contents.tables) == 1
    table = contents.tables[0]
    assert table.caption == (
        "Table 1 | Overview of subdomains showing the number of academic studies and "
        "grey literature sources reviewed"
    )
    assert len(table.parts) == 2
    assert table.contents == [
        ["Domain", "Sub-domain", "Definition", "Academic indicators"],
        ["Monitoring", "Governance", "First", "10"],
        ["Action", "Equity", "Second", "20"],
    ]
    assert [part.page_number for part in table.parts] == [4, 5]
    continued = next(
        item
        for item in contents.caption_assignment_receipt.assignments
        if "(continued)"
        in next(
            candidate.text
            for candidate in contents.caption_assignment_receipt.candidates
            if candidate.caption_id == item.caption_id
        )
    )
    assert continued.object_id == "table:1"
    assert "continuation_evidence" in continued.reasons


def test_continued_html_table_keeps_each_pages_source_markup():
    """A table continued across pages exports the printed HTML of each piece,
    not a re-render of the merged frame, which drops rowspans and adds
    ``class="dataframe"`` noise."""
    first = (
        "<table><tr><th>Domain</th><th>Score</th></tr>"
        '<tr><td rowspan="2">Monitoring</td><td>10</td></tr><tr><td>11</td></tr></table>'
    )
    second = (
        "<table><tr><th>Domain</th><th>Score</th></tr><tr><td>Action</td><td>20</td></tr></table>"
    )
    pages = [[] for _ in range(2)]
    pages[0] = [
        _region(1, "figure_title", "Table 1 | Scores", bbox=[70, 60, 900, 80]),
        _region(2, "table", first, bbox=[70, 100, 930, 900]),
    ]
    pages[1] = [
        _region(1, "figure_title", "Table 1 (continued) | Scores", bbox=[70, 60, 900, 80]),
        _region(2, "table", second, bbox=[70, 100, 930, 600]),
    ]

    (table,) = _parse(pages).tables

    assert len(table.parts) == 2
    assert table.tbl_html == f"{first}\n{second}"
    assert "dataframe" not in table.tbl_html


def test_unlabelled_explicit_continuation_groups_only_with_compatible_previous_table():
    contents = _parse(
        [
            [
                _region(0, "figure_title", "Table 7. Values", bbox=[0, 10, 500, 40]),
                _region(1, "table", "| A | B |\n|---|---|\n| 1 | 2 |", bbox=[0, 50, 500, 900]),
            ],
            [
                _region(0, "figure_title", "Table continued", bbox=[0, 10, 500, 40]),
                _region(1, "table", "| A | B |\n|---|---|\n| 3 | 4 |", bbox=[0, 50, 500, 900]),
            ],
        ]
    )

    assert len(contents.tables) == 1
    assert len(contents.tables[0].parts) == 2
    assert contents.tables[0].caption == "Table 7. Values"


@pytest.mark.parametrize(
    "second_caption",
    ["Table 7. Values", "Table 7 (continued). Values", "Table continued"],
)
def test_table_continuation_evidence_does_not_cross_section_heading(second_caption):
    contents = _parse(
        [
            [
                _region(0, "figure_title", "Table 7. Values", bbox=[0, 10, 500, 40]),
                _region(1, "table", "| A | B |\n|---|---|\n| 1 | 2 |", bbox=[0, 50, 500, 900]),
            ],
            [
                _region(0, "paragraph_title", "Results", bbox=[0, 10, 500, 40]),
                _region(1, "figure_title", second_caption, bbox=[0, 50, 500, 80]),
                _region(2, "table", "| A | B |\n|---|---|\n| 3 | 4 |", bbox=[0, 90, 500, 900]),
            ],
        ]
    )

    assert len(contents.tables) == 2
    assert [[part.page_number for part in table.parts] for table in contents.tables] == [
        [1],
        [2],
    ]


def test_short_figure_title_near_one_figure_is_a_receipted_caption():
    contents = _parse(
        [
            [
                _region(0, "chart", bbox=[100, 100, 900, 600], image_b64="figure"),
                _region(
                    1,
                    "figure_title",
                    "Mean values by group",
                    bbox=[150, 620, 850, 650],
                ),
            ]
        ]
    )

    assert contents.figures[0].caption == "Mean values by group"
    receipt = contents.caption_assignment_receipt
    assert [candidate.text for candidate in receipt.candidates] == ["Mean values by group"]
    assert receipt.assignments[0].object_id == "figure:1"


def test_ord337_bounded_panel_sequence_groups_seven_parts_and_receipts_every_title():
    pages = [[] for _ in range(17)]
    pages[13] = [
        _region(1, "chart", bbox=[268, 82, 830, 349], image_b64="part-1"),
        _region(2, "figure_title", "Overall", bbox=[303, 366, 353, 380]),
        _region(3, "chart", bbox=[271, 406, 823, 678], image_b64="part-2"),
        _region(4, "figure_title", "All, by race", bbox=[303, 699, 377, 714]),
    ]
    pages[14] = [
        _region(1, "chart", bbox=[297, 83, 834, 346], image_b64="part-3"),
        _region(
            2,
            "figure_title",
            "All by College Degree",
            bbox=[292, 367, 434, 384],
        ),
        _region(3, "chart", bbox=[265, 400, 829, 683], image_b64="part-4"),
        _region(
            4,
            "figure_title",
            "Whites, by college graduation",
            bbox=[292, 697, 482, 714],
        ),
    ]
    pages[15] = [
        _region(1, "chart", bbox=[264, 79, 823, 355], image_b64="part-5"),
        _region(
            2,
            "figure_title",
            "Blacks, by college graduation",
            bbox=[283, 369, 471, 385],
        ),
        _region(3, "chart", bbox=[264, 402, 832, 683], image_b64="part-6"),
        _region(
            4,
            "figure_title",
            "No college degree, by race",
            bbox=[283, 697, 453, 713],
        ),
    ]
    pages[16] = [
        _region(
            1,
            "figure_title",
            "One Minus Survival Functions",
            bbox=[468, 77, 716, 96],
        ),
        _region(2, "chart", bbox=[156, 91, 949, 468], image_b64="part-7"),
        _region(
            4,
            "figure_title",
            "Figure 1.\nTime to Retire Over the Course of Follow Up (30 Years) by Race and "
            "College Graduation",
            bbox=[239, 539, 834, 573],
        ),
    ]

    contents = _parse(pages)

    assert len(contents.figures) == 1
    figure = contents.figures[0]
    assert figure.caption == (
        "Figure 1.\nTime to Retire Over the Course of Follow Up (30 Years) by Race and "
        "College Graduation"
    )
    assert len(figure.parts) == 7
    assert [part.image_b64 for part in figure.parts] == [f"part-{index}" for index in range(1, 8)]
    receipt = contents.caption_assignment_receipt
    assert [candidate.text for candidate in receipt.candidates] == [
        "Overall",
        "All, by race",
        "All by College Degree",
        "Whites, by college graduation",
        "Blacks, by college graduation",
        "No college degree, by race",
        "One Minus Survival Functions",
        "Figure 1.\nTime to Retire Over the Course of Follow Up (30 Years) by Race and "
        "College Graduation",
    ]
    panel_assignments = [item for item in receipt.assignments if "panel_evidence" in item.reasons]
    assert len(panel_assignments) == 7
    assert {item.object_id for item in panel_assignments} == {"figure:1"}
    assignment = next(
        item
        for item in receipt.assignments
        if item.object_id == "figure:1" and "panel_evidence" not in item.reasons
    )
    assert "panel_evidence:7" in assignment.reasons


def test_subsequent_explicit_figure_caption_fences_panel_group():
    contents = _parse(
        [
            [
                _region(0, "chart", bbox=[0, 100, 500, 200], image_b64="p1"),
                _region(1, "figure_title", "(a)", bbox=[0, 205, 500, 225]),
                _region(2, "chart", bbox=[0, 230, 500, 330], image_b64="p2"),
                _region(3, "figure_title", "(b)", bbox=[0, 335, 500, 355]),
                _region(4, "chart", bbox=[0, 360, 500, 460], image_b64="next"),
                _region(5, "figure_title", "Next plot", bbox=[0, 465, 500, 485]),
                _region(6, "figure_title", "Figure 1. Combined", bbox=[0, 490, 500, 510]),
                _region(7, "figure_title", "Figure 2. Separate", bbox=[0, 515, 500, 535]),
            ]
        ]
    )

    assert len(contents.figures) == 2
    assert [[part.image_b64 for part in figure.parts] for figure in contents.figures] == [
        ["p1", "p2"],
        ["next"],
    ]
    assert [figure.caption for figure in contents.figures] == [
        "Figure 1. Combined",
        "Figure 2. Separate",
    ]
    receipt = contents.caption_assignment_receipt
    assignments = {
        candidate.text: assignment
        for candidate, assignment in zip(receipt.candidates, receipt.assignments, strict=True)
    }
    assert assignments["(a)"].object_id == "figure:1"
    assert assignments["(b)"].object_id == "figure:1"
    assert "panel_evidence" in assignments["(a)"].reasons
    assert "panel_evidence" in assignments["(b)"].reasons
    assert assignments["Next plot"].object_id == "figure:2"
    assert "panel_evidence" not in assignments["Next plot"].reasons
    assert assignments["Figure 1. Combined"].object_id == "figure:1"
    assert assignments["Figure 2. Separate"].object_id == "figure:2"


def test_future_explicit_owner_skips_panel_group_when_only_one_member_would_remain():
    contents = _parse(
        [
            [
                _region(0, "chart", bbox=[0, 100, 500, 200], image_b64="p1"),
                _region(1, "figure_title", "First plot", bbox=[0, 205, 500, 225]),
                _region(2, "chart", bbox=[0, 230, 500, 330], image_b64="next"),
                _region(3, "figure_title", "Next plot", bbox=[0, 335, 500, 355]),
                _region(4, "figure_title", "Figure 1. First", bbox=[0, 360, 500, 380]),
                _region(5, "figure_title", "Figure 2. Separate", bbox=[0, 385, 500, 405]),
            ]
        ]
    )

    assert len(contents.figures) == 2
    assert [[part.image_b64 for part in figure.parts] for figure in contents.figures] == [
        ["p1"],
        ["next"],
    ]
    assert [figure.caption for figure in contents.figures] == [
        "Figure 1. First",
        "Figure 2. Separate",
    ]
    assignments = {
        candidate.text: assignment
        for candidate, assignment in zip(
            contents.caption_assignment_receipt.candidates,
            contents.caption_assignment_receipt.assignments,
            strict=True,
        )
    }
    assert assignments["Figure 1. First"].object_id == "figure:1"
    assert assignments["Figure 2. Separate"].object_id == "figure:2"


def test_future_explicit_owner_on_adjacent_page_fences_panel_group():
    contents = _parse(
        [
            [
                _region(0, "chart", bbox=[0, 100, 500, 300], image_b64="p1"),
                _region(1, "figure_title", "(a)", bbox=[0, 305, 500, 325]),
            ],
            [
                _region(0, "chart", bbox=[0, 100, 500, 300], image_b64="p2"),
                _region(1, "figure_title", "(b)", bbox=[0, 305, 500, 325]),
            ],
            [
                _region(0, "chart", bbox=[0, 100, 500, 300], image_b64="next"),
                _region(1, "figure_title", "Next plot", bbox=[0, 305, 500, 325]),
                _region(2, "figure_title", "Figure 1. Combined", bbox=[0, 330, 500, 350]),
            ],
            [
                _region(0, "figure_title", "Figure 2. Separate", bbox=[0, 10, 500, 30]),
            ],
        ]
    )

    assert len(contents.figures) == 2
    assert [[part.image_b64 for part in figure.parts] for figure in contents.figures] == [
        ["p1", "p2"],
        ["next"],
    ]
    assert [figure.caption for figure in contents.figures] == [
        "Figure 1. Combined",
        "Figure 2. Separate",
    ]


@pytest.mark.parametrize("decoration_count", [2, 4], ids=["ord121-shape", "ord343-shape"])
def test_printed_figure_id_reconciles_receipt_xref_and_content_section(decoration_count):
    from bibr.paper_contents import PaperSentence
    from bibr.structure.pdf_parser import PDFParser
    from bibr.structure.xref_utils import detect_xrefs

    decorations = [
        _region(
            index,
            "chart",
            bbox=[0, 50 + index * 100, 500, 130 + index * 100],
            image_b64=f"decoration-{index}",
        )
        for index in range(decoration_count)
    ]
    parser = PDFParser(
        [
            decorations,
            [
                _region(0, "chart", bbox=[0, 100, 500, 300], image_b64="printed-figure-1"),
                _region(
                    1,
                    "figure_title",
                    "Figure 1. Actual result",
                    bbox=[0, 305, 500, 345],
                ),
            ],
        ]
    )
    contents = parser.parse()

    assert [figure.figure_id for figure in contents.figures] == list(range(1, decoration_count + 2))
    printed_figure = contents.figures[0]
    assert printed_figure.figure_id == 1
    assert printed_figure.image_b64 == "printed-figure-1"
    assert printed_figure.parts[0].image_b64 == "printed-figure-1"
    assignment = next(
        item
        for item in contents.caption_assignment_receipt.assignments
        if item.caption_id == contents.caption_assignment_receipt.candidates[-1].caption_id
    )
    assert assignment.object_id == "figure:1"

    sentence = PaperSentence(
        text_id=1,
        text="See Figure 1.",
        section_id=0,
        paragraph_id=1,
    )
    assert [
        (xref.xref_type, xref.xref_id) for xref in detect_xrefs([sentence], [], contents.figures)
    ] == [("figure", 1)]

    parser.create_content_sections(contents)
    printed_section = next(
        section for section in contents.sections if section.section_id == printed_figure.section_id
    )
    assert printed_section.header == "Figure 1"


def test_printed_roman_table_id_reconciles_receipt_xref_and_content_section():
    from bibr.paper_contents import PaperSentence
    from bibr.structure.pdf_parser import PDFParser
    from bibr.structure.xref_utils import detect_xrefs

    table = "| A |\n|---|\n| 1 |"
    parser = PDFParser(
        [
            [
                _region(0, "table", table, bbox=[0, 100, 500, 300]),
                _region(1, "table", table, bbox=[0, 400, 500, 600]),
            ],
            [
                _region(0, "figure_title", "TABLE I. Values", bbox=[0, 10, 500, 40]),
                _region(1, "table", table, bbox=[0, 50, 500, 500]),
            ],
        ]
    )
    contents = parser.parse()

    assert [item.table_id for item in contents.tables] == [1, 2, 3]
    printed_table = contents.tables[0]
    assert printed_table.caption == "TABLE I. Values"
    assert printed_table.parts[0].page_number == 2
    assignment = next(
        item
        for item in contents.caption_assignment_receipt.assignments
        if item.caption_id == contents.caption_assignment_receipt.candidates[-1].caption_id
    )
    assert assignment.object_id == "table:1"

    sentence = PaperSentence(text_id=1, text="See Table 1.", section_id=0, paragraph_id=1)
    assert [
        (xref.xref_type, xref.xref_id) for xref in detect_xrefs([sentence], contents.tables, [])
    ] == [("table", 1)]

    parser.create_content_sections(contents)
    printed_section = next(
        section for section in contents.sections if section.section_id == printed_table.section_id
    )
    assert printed_section.header == "Table 1"


def test_duplicate_printed_figure_ids_warn_without_colliding():
    contents = _parse(
        [
            [
                _region(0, "chart", bbox=[0, 100, 500, 200], image_b64="alpha"),
                _region(1, "figure_title", "Figure 1. Alpha", bbox=[0, 205, 500, 225]),
                _region(2, "chart", bbox=[0, 300, 500, 400], image_b64="beta"),
                _region(3, "figure_title", "Figure 1. Beta", bbox=[0, 405, 500, 425]),
            ]
        ]
    )

    assert [figure.figure_id for figure in contents.figures] == [2, 3]
    assert len({figure.figure_id for figure in contents.figures}) == 2
    assert {
        assignment.object_id
        for assignment in contents.caption_assignment_receipt.assignments
        if assignment.object_id is not None
    } == {"figure:2", "figure:3"}
    conflict = next(
        issue
        for issue in contents.structure_validation_issues
        if issue.code == "VAL_MEDIA_ID_CONFLICT"
    )
    assert conflict.count == 2


def test_panel_grouping_does_not_merge_unrelated_distant_pages():
    pages = [[] for _ in range(30)]
    pages[0] = [
        _region(0, "chart", bbox=[100, 100, 900, 600], image_b64="unrelated"),
        _region(1, "figure_title", "Mean values by group", bbox=[150, 620, 850, 650]),
    ]
    pages[29] = [
        _region(0, "chart", bbox=[100, 100, 900, 600], image_b64="target"),
        _region(1, "figure_title", "Overall", bbox=[150, 620, 850, 650]),
        _region(2, "figure_title", "Figure 1. Target", bbox=[150, 670, 850, 700]),
    ]

    contents = _parse(pages)

    assert len(contents.figures) == 2
    assert [figure.figure_id for figure in contents.figures] == [1, 2]
    assert [part.image_b64 for part in contents.figures[0].parts] == ["target"]
    assert [part.image_b64 for part in contents.figures[1].parts] == ["unrelated"]
    assert contents.figures[0].caption == "Figure 1. Target"


def test_ord343_exact_split_labels_and_bare_continuations_form_three_tables():
    pages = [[] for _ in range(25)]
    pages[20] = [
        _region(1, "vision_footnote", "TABLE I.", bbox=[90, 440, 107, 491]),
        _region(
            2,
            "figure_title",
            "The Social Determinants of Health Factors Used in the Study",
            bbox=[119, 543, 139, 857],
        ),
        _region(
            3,
            "table",
            "| Variable | Frequency | Percentage |\n|---|---|---|\n| Sex | 29 | 100% |",
            bbox=[155, 245, 844, 857],
        ),
    ]
    pages[21] = [
        _region(
            1,
            "table",
            "| Variable | Frequency | Percentage |\n|---|---|---|\n| Internet | 11 | 37.9% |",
            bbox=[90, 74, 289, 683],
        ),
    ]
    pages[22] = [
        _region(1, "vision_footnote", "TABLE II.", bbox=[90, 439, 107, 492]),
        _region(
            2,
            "figure_title",
            "The Correlational Relationship of Social Determinants of Health on Psychological "
            "Distress of Students in Online Learning",
            bbox=[118, 233, 141, 858],
        ),
        _region(
            3,
            "table",
            "| Variable | Readiness for Online Learning | Psychological Distress |\n"
            "|---|---|---|\n| Sex | -.13 | .17 |",
            bbox=[155, 419, 456, 856],
        ),
        _region(4, "figure_title", "p < .05", bbox=[469, 820, 488, 854]),
    ]
    pages[23] = [
        _region(1, "figure_title", "Table III.", bbox=[91, 440, 107, 490]),
        _region(
            2,
            "figure_title",
            "Sample Curriculum Elements of Health Psychology",
            bbox=[119, 590, 139, 857],
        ),
        _region(
            3,
            "table",
            "| Subject Courses | Expected Student Outcomes | Teacher |\n"
            "|---|---|---|\n| A. Bio-Behavioral Science | Knowledge Base | Refresher |",
            bbox=[155, 76, 828, 856],
        ),
    ]
    pages[24] = [
        _region(1, "figure_title", "Author Manuscript", bbox=[31, 345, 52, 474]),
        _region(2, "figure_title", "Author Manuscript", bbox=[31, 559, 53, 687]),
        _region(
            3,
            "table",
            "| Subject Courses | Expected Student Outcomes | Teacher |\n"
            "|---|---|---|\n| D. Research | Scholarship | Thesis |",
            bbox=[87, 76, 267, 853],
        ),
    ]

    contents = _parse(pages)

    assert len(contents.tables) == 3
    assert [len(table.parts) for table in contents.tables] == [2, 1, 2]
    assert [[part.page_number for part in table.parts] for table in contents.tables] == [
        [21, 22],
        [23],
        [24, 25],
    ]
    assert [table.caption for table in contents.tables] == [
        "TABLE I. The Social Determinants of Health Factors Used in the Study",
        "TABLE II. The Correlational Relationship of Social Determinants of Health on "
        "Psychological Distress of Students in Online Learning",
        "Table III. Sample Curriculum Elements of Health Psychology",
    ]
    assert all("Author Manuscript" not in (table.caption or "") for table in contents.tables)
    receipt = contents.caption_assignment_receipt
    assert [candidate.text for candidate in receipt.candidates] == [
        "TABLE I.",
        "The Social Determinants of Health Factors Used in the Study",
        "TABLE II.",
        "The Correlational Relationship of Social Determinants of Health on Psychological "
        "Distress of Students in Online Learning",
        "p < .05",
        "Table III.",
        "Sample Curriculum Elements of Health Psychology",
        "Author Manuscript",
        "Author Manuscript",
    ]
    manuscript_assignments = [
        assignment
        for assignment, candidate in zip(receipt.assignments, receipt.candidates, strict=True)
        if candidate.text == "Author Manuscript"
    ]
    assert all(
        item.object_id is None and "publisher_noise" in item.reasons
        for item in manuscript_assignments
    )


def test_missing_bbox_caption_abstains_in_parser_and_emits_typed_issue():
    contents = _parse(
        [
            [
                _region(0, "image", bbox=None, image_b64="one"),
                _region(1, "image", bbox=None, image_b64="two"),
                _region(2, "figure_title", "Figure 1. Ambiguous", bbox=None),
            ]
        ]
    )

    assert len(contents.figures) == 2
    assert all(figure.caption is None for figure in contents.figures)
    assert {issue.code for issue in contents.structure_validation_issues} == {
        "VAL_CAPTION_OWNERSHIP"
    }


def test_paper_post_init_deduplicates_repeated_internal_structure_issues():
    from pathlib import Path

    from bibr.input.file import InputFile, InputFormat
    from bibr.models import PaperMetadata
    from bibr.paper import Paper

    contents = _parse(
        [[_region(0, "figure_title", "Figure 1. Missing target", bbox=[0, 0, 500, 30])]]
    )
    issue = contents.structure_validation_issues[0]
    contents.structure_validation_issues.append(issue)

    paper = Paper(
        input_file=InputFile(
            path=Path("/tmp/repeated-issue.pdf"),
            file_hash="repeated-issue",
            input_format=InputFormat(
                file_extension=".pdf",
                detected_mime_type="application/pdf",
                file_type="pdf",
            ),
        ),
        metadata=PaperMetadata(title="Repeated issue", doi=""),
        contents=contents,
    )

    assert paper.validation_issues == [issue]


def test_bare_table_label_exemption_does_not_cross_multiple_heading_boundaries():
    contents = _parse(
        [
            [
                _region(0, "paragraph_title", "Table 1", bbox=[0, 10, 500, 40]),
                _region(1, "paragraph_title", "Methods", bbox=[0, 50, 500, 80]),
                _region(2, "paragraph_title", "Results", bbox=[0, 90, 500, 120]),
                _region(3, "table", "| A |\n|---|\n| 1 |", bbox=[0, 130, 500, 800]),
            ]
        ]
    )

    assert contents.tables[0].caption is None


def test_confirmed_table_caption_deduplicated_away_does_not_raise():
    """The confirmed owner id is recorded at parse time.

    De-duplication elects a canonical purely by text length and source index,
    with no awareness of which cluster member holds the confirmed bare-table
    label, so the recorded id can be absent from the post-dedup candidates.
    Indexing the lookup with it raised KeyError, which ParseSegmentStage
    reported as parse_failed — dropping the whole paper.
    """
    import pandas as pd

    from bibr.paper_contents import PaperTable
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(json_result=[])
    table = PaperTable(table_id=1, df=pd.DataFrame({"a": [1]}), tbl_html="<table/>", section_id=1)
    parser.tables = [table]
    parser._table_source_indices[id(table)] = 0
    parser._confirmed_table_caption_owners["caption:2"] = id(table)

    # "caption:2" lost the canonical vote, so it is not among the actives.
    parser._group_continuation_tables([])

    assert table.caption is None
