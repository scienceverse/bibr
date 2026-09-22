"""Media provenance and diagnostic receipt export (float parts since v12)."""

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
from tests.export.conftest import extraction_block as _extraction_block


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
        # Points per 0..1000 layout unit: 0.5 x 1.0 on page 2, 1.0 x 0.5 on page 3.
        page_sizes={2: (500.0, 1000.0), 3: (1000.0, 500.0)},
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


def test_float_parts_caption_assignment_and_reference_yield_export_losslessly():
    from bibr.export import PaperExport, export_paper_to_json

    paper = _paper()
    paper.extraction = _extraction_block()
    output = export_paper_to_json(paper, validate=False)
    diagnostics = output["extraction"]["diagnostics"]

    assert output["schema_version"] == "12.0"
    # v12: the figure/table rows are whole objects; each printed piece's page
    # and box ride extraction.float_parts.
    assert "parts" not in output["figure"][0] and "parts" not in output["table"][0]
    # Not image bytes, so the data URI cannot name an image type.
    assert output["figure"][0]["image"] == "data:application/octet-stream;base64,image"
    assert output["extraction"]["float_parts"] == [
        {
            "object_type": "figure",
            "object_id": 1,
            "part_index": 1,
            "page_number": 2,
            "bbox": [5.0, 20.0, 15.0, 40.0],
        },
        {
            "object_type": "table",
            "object_id": 1,
            "part_index": 1,
            "page_number": 3,
            "bbox": [50.0, 30.0, 70.0, 40.0],
        },
    ]
    assert diagnostics["caption_assignment"]["candidates"][0]["bbox"] == [5.0, 42.0, 15.0, 48.0]
    assert diagnostics["caption_assignment"]["assignments"][0]["reasons"] == ["same_page"]
    assert diagnostics["reference_yield"]["attempts"][0]["spans"] == [[0, 10], [10, 20]]
    assert diagnostics["reference_yield"]["attempts"][0]["reason_flags"] == ["credible_starts"]
    assert diagnostics["reference_yield"]["selected_spans"] == [[0, 10], [10, 20]]
    assert diagnostics["reference_yield"]["reason_flags"] == ["complete_coverage"]
    assert (
        PaperExport.model_validate(output).model_dump(by_alias=True, exclude_unset=True) == output
    )


def test_omits_unavailable_receipts_and_float_parts():
    from bibr.export import PaperExport, export_paper_to_json

    paper = _paper(receipts=False)
    paper.extraction = _extraction_block()
    paper.contents.figures[0].parts = []
    paper.contents.tables[0].parts = []
    output = export_paper_to_json(paper, validate=False)

    assert "caption_assignment" not in output["extraction"]["diagnostics"]
    assert "reference_yield" not in output["extraction"]["diagnostics"]
    assert "float_parts" not in output["extraction"]
    PaperExport.model_validate(output)


def test_typed_model_and_result_reject_a_v10_payload():
    """Each major is a clean break — there is no dual-read and no compatibility
    shim, so a v10-shaped payload must fail validation rather than half-load."""
    import pytest
    from pydantic import ValidationError

    from bibr.api import Result
    from bibr.export import PaperExport, export_paper_to_json

    legacy = deepcopy(export_paper_to_json(_paper(receipts=False), validate=False))
    legacy["info"] = {**legacy.pop("metadata"), **legacy.pop("source"), "schema_version": "10.6"}
    legacy.pop("schema_version")

    with pytest.raises(ValidationError):
        PaperExport.model_validate(legacy)
    with pytest.raises(ValidationError):
        Result(legacy)


def test_durable_replay_accepts_a_current_core_and_rejects_an_older_major():
    import pytest

    from bibr.export import export_paper_to_json
    from bibr.pipeline.artifacts import (
        ArtifactReplayError,
        canonical_json_sha256,
        make_enrichment_sidecar,
        replay_enrichment_sidecar,
    )

    paper = _paper(receipts=False)
    # Replay operates on pipeline-produced cores, which always carry
    # ``extraction`` (the enrichment receipt's home).
    paper.extraction = _extraction_block()
    current = export_paper_to_json(paper, validate=False)
    legacy = deepcopy(current)
    legacy["schema_version"] = "11.0"

    sidecar = make_enrichment_sidecar(
        current,
        core_sha256=canonical_json_sha256(current),
        settings_digest="settings",
        completeness="complete",
    )
    replayed = replay_enrichment_sidecar(current, sidecar, expected_settings_digest="settings")
    assert replayed["schema_version"] == "12.0"
    assert canonical_json_sha256(current) == sidecar.core_sha256

    legacy_sidecar = make_enrichment_sidecar(
        legacy,
        core_sha256=canonical_json_sha256(legacy),
        settings_digest="settings",
        completeness="complete",
    )
    with pytest.raises(ArtifactReplayError, match="schema does not match"):
        replay_enrichment_sidecar(legacy, legacy_sidecar, expected_settings_digest="settings")
