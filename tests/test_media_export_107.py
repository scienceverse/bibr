"""Additive v10.7 media provenance and diagnostic receipt export."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from bibr.input.file import InputFile, InputFormat
from bibr.models import PaperMetadata
from bibr.paper import Paper
from bibr.paper_contents import (
    CaptionAssignment,
    CaptionAssignmentReceipt,
    CaptionCandidate,
    PaperContents,
    PaperFigure,
    PaperFigurePart,
    PaperTable,
    PaperTablePart,
    Provenance,
    ReferenceSegmentationAttempt,
    ReferenceYieldReceipt,
)


def _paper(*, receipts: bool = True) -> Paper:
    provenance = [Provenance(page_no=2, bbox=(10.0, 20.0, 30.0, 40.0))]
    figure_part = PaperFigurePart(2, (10.0, 20.0, 30.0, 40.0), "image", provenance)
    table_df = pd.DataFrame([["1"]], columns=["A"])
    table_part = PaperTablePart(
        3,
        (50.0, 60.0, 70.0, 80.0),
        "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>",
        table_df,
        [Provenance(page_no=3, bbox=(50.0, 60.0, 70.0, 80.0))],
    )
    caption_receipt = None
    reference_receipt = None
    if receipts:
        candidate = CaptionCandidate(
            "caption:1", "Figure 1. Plot", "figure", 2, (10.0, 42.0, 30.0, 48.0), 4
        )
        caption_receipt = CaptionAssignmentReceipt(
            (candidate,),
            (CaptionAssignment("caption:1", "figure:1", 8.5, ("same_page",)),),
        )
        reference_receipt = ReferenceYieldReceipt(
            credible_source_starts=2,
            attempts=(
                ReferenceSegmentationAttempt(
                    "geom", ((0, 10), (10, 20)), 2, True, ("credible_starts",)
                ),
            ),
            selected_spans=((0, 10), (10, 20)),
            source_character_coverage=1.0,
            parsed_count=2,
            valid_count=2,
            duplicate_rate=0.0,
            reason_flags=("complete_coverage",),
        )
    contents = PaperContents(
        sentences=[],
        sections=[],
        tables=[
            PaperTable(
                1,
                table_df,
                table_part.tbl_html or "",
                0,
                "Table 1. Values",
                3,
                list(table_part.provenance),
                parts=[table_part],
            )
        ],
        links=[],
        sections_text={},
        figures=[
            PaperFigure(
                1,
                0,
                "image",
                "Figure 1. Plot",
                2,
                provenance,
                parts=[figure_part],
            )
        ],
        reference_yield_receipt=reference_receipt,
        caption_assignment_receipt=caption_receipt,
    )
    return Paper(
        input_file=InputFile(
            path=Path("/tmp/paper.pdf"),
            file_hash="hash",
            input_format=InputFormat(
                file_extension=".pdf",
                detected_mime_type="application/pdf",
                file_type="pdf",
            ),
        ),
        metadata=PaperMetadata(title="Paper", doi=""),
        contents=contents,
    )


def test_v107_exports_parts_caption_assignment_and_reference_yield_losslessly():
    from bibr.export import PaperExport, export_paper_to_json

    output = export_paper_to_json(_paper(), validate=False)

    assert output["info"]["schema_version"] == "10.7"
    assert output["figure"][0]["parts"] == [
        {
            "part_index": 1,
            "image": "image",
            "page_number": 2,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "provenance": [{"page": 2, "bbox": [10.0, 20.0, 30.0, 40.0]}],
        }
    ]
    assert output["table"][0]["parts"][0]["part_index"] == 1
    assert output["table"][0]["parts"][0]["contents"] == [["A"], ["1"]]
    assert output["caption_assignment"]["candidates"][0]["bbox"] == [10.0, 42.0, 30.0, 48.0]
    assert output["caption_assignment"]["assignments"][0]["reasons"] == ["same_page"]
    assert output["reference_yield"]["attempts"][0]["spans"] == [[0, 10], [10, 20]]
    assert output["reference_yield"]["attempts"][0]["reason_flags"] == ["credible_starts"]
    assert output["reference_yield"]["selected_spans"] == [[0, 10], [10, 20]]
    assert output["reference_yield"]["reason_flags"] == ["complete_coverage"]
    assert (
        PaperExport.model_validate(output).model_dump(by_alias=True, exclude_unset=True) == output
    )


def test_v107_omits_unavailable_receipts_and_native_empty_parts_are_valid():
    from bibr.export import PaperExport, export_paper_to_json

    paper = _paper(receipts=False)
    paper.contents.figures[0].parts = []
    paper.contents.tables[0].parts = []
    output = export_paper_to_json(paper, validate=False)

    assert "caption_assignment" not in output
    assert "reference_yield" not in output
    assert output["figure"][0]["parts"] == []
    assert output["table"][0]["parts"] == []
    PaperExport.model_validate(output)


def test_typed_model_and_result_still_accept_v106_payload_without_additive_fields():
    from bibr.api import Result
    from bibr.export import PaperExport, export_paper_to_json

    legacy = deepcopy(export_paper_to_json(_paper(receipts=False), validate=False))
    legacy["info"]["schema_version"] = "10.6"
    for figure in legacy["figure"]:
        figure.pop("parts")
    for table in legacy["table"]:
        table.pop("parts")

    assert PaperExport.model_validate(legacy).info.schema_version == "10.6"
    assert Result(legacy).model.info.schema_version == "10.6"


def test_durable_replay_accepts_exact_bound_v106_and_v107_cores():
    from bibr.export import export_paper_to_json
    from bibr.pipeline.artifacts import (
        canonical_json_sha256,
        make_enrichment_sidecar,
        replay_enrichment_sidecar,
    )

    current = export_paper_to_json(_paper(receipts=False), validate=False)
    legacy = deepcopy(current)
    legacy["info"]["schema_version"] = "10.6"
    for core in (legacy, current):
        sidecar = make_enrichment_sidecar(
            core,
            core_sha256=canonical_json_sha256(core),
            settings_digest="settings",
            completeness="complete",
        )

        replayed = replay_enrichment_sidecar(core, sidecar, expected_settings_digest="settings")

        assert replayed["info"]["schema_version"] == core["info"]["schema_version"]
        assert canonical_json_sha256(core) == sidecar.core_sha256
