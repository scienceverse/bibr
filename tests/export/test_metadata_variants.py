"""Additive printed-version tables and inspectable record ownership."""

from dataclasses import replace

import pytest
from pydantic import ValidationError

from bibr.api import Result
from bibr.export.json_export import _export_paper_payload, validate_export
from bibr.export.models import _SCHEMA_VERSION, PaperExport
from bibr.export.schema_artifact import build_export_schema
from bibr.extract.metadata_variants import collect_metadata_variants
from bibr.extract.primary_presentation import select_printed_presentation
from tests.extract.test_metadata_variants import _fixture


def _attach_variants(paper):
    contents, resolution = _fixture()
    contents.front_matter_resolution = resolution
    contents.metadata_variants = collect_metadata_variants(contents, resolution)
    paper.contents = contents
    paper.metadata.title = "SHADE AND SEEDLING GROWTH"
    paper.metadata.abstract = "Shade improved growth."
    return resolution


def test_variants_preserve_text_sources_and_primary_scalar(demo_paper):
    resolution = _attach_variants(demo_paper)
    payload = _export_paper_payload(demo_paper)

    variants = payload["metadata_variant"]
    assert len(variants) == 4
    assert {row["record_id"] for row in variants} == {resolution.selected_block_id}
    assert all(row["source_section_ids"] and row["pages"] for row in variants)
    for field in ("title", "abstract"):
        primary = [row for row in variants if row["field"] == field and row["is_primary"]]
        assert len(primary) == 1
        assert primary[0]["text"] == payload["metadata"][field]
    assert all(row["language"] is None for row in variants)
    assert variants[0]["presentation_ids"] == variants[1]["presentation_ids"]
    assert variants[0]["presentation_ids"] != variants[2]["presentation_ids"]
    assert variants[1]["byline_source_text_ids"] == [2]
    receipt = payload["extraction"]["diagnostics"]["front_matter"]
    assert receipt["selected_block_id"] == resolution.selected_block_id
    assert receipt["blocks"][0]["candidate_ids"] == list(resolution.blocks[0].candidate_ids)
    assert {row["candidate_id"] for row in receipt["candidates"]} == set(
        resolution.blocks[0].candidate_ids
    )
    assert all("raw_text" not in row for row in receipt["candidates"])
    assert not validate_export(payload)
    assert Result(payload).metadata_variant.df.shape[0] == 4


def test_primary_flag_describes_final_scalar_after_guards(demo_paper):
    _attach_variants(demo_paper)
    demo_paper.metadata.abstract = ""
    payload = _export_paper_payload(demo_paper)
    assert not any(
        row["is_primary"] for row in payload["metadata_variant"] if row["field"] == "abstract"
    )


def test_abstention_retains_candidates_without_exporting_unowned_versions(demo_paper):
    resolution = _attach_variants(demo_paper)
    demo_paper.contents.front_matter_resolution = replace(
        resolution,
        selected_block_id=None,
        selection_method="abstained",
        reason_flags=("multiple_front_matter_blocks",),
    )
    payload = _export_paper_payload(demo_paper)
    assert payload["metadata_variant"] == []
    receipt = payload["extraction"]["diagnostics"]["front_matter"]
    assert receipt["selected_block_id"] is None
    assert receipt["blocks"] and receipt["candidates"]


def test_existing_v11_payload_without_additive_table_remains_readable(demo_paper):
    payload = _export_paper_payload(demo_paper)
    assert payload["metadata_variant"] == []
    payload.pop("metadata_variant")
    payload["schema_version"] = "11.0"
    assert not validate_export(payload)
    assert build_export_schema()["properties"]["schema_version"]["const"] == _SCHEMA_VERSION


def test_orphan_or_cross_record_presentation_is_rejected(demo_paper):
    _attach_variants(demo_paper)
    payload = _export_paper_payload(demo_paper)
    payload["metadata_variant"][0]["presentation_ids"] = []
    with pytest.raises(ValidationError, match="each presentation"):
        PaperExport.model_validate(payload)


def test_presentation_requires_compatible_byline_evidence_on_both_versions(demo_paper):
    _attach_variants(demo_paper)
    payload = _export_paper_payload(demo_paper)
    payload["metadata_variant"][1]["byline_source_text_ids"] = [999]
    with pytest.raises(ValidationError, match="share printed byline"):
        PaperExport.model_validate(payload)


@pytest.mark.parametrize("fault", [None, "selected_id", "variant", "byline", "record"])
def test_primary_selection_preserves_and_validates_source_links(demo_paper, fault):
    resolution = _attach_variants(demo_paper)
    demo_paper.contents.presentation_selection = select_printed_presentation(
        demo_paper.contents.metadata_variants, resolution
    )
    payload = _export_paper_payload(demo_paper)
    choice = payload["extraction"]["diagnostics"]["front_matter"]["presentation_selection"]
    assert choice["selected_presentation_id"] == choice["presentations"][0]["presentation_id"]
    assert choice["reason"] == "first_complete_printed"
    if fault is None:
        assert PaperExport.model_validate(payload)
        return
    row = choice["presentations"][0]
    if fault == "selected_id":
        choice["selected_presentation_id"] = "missing"
    elif fault == "variant":
        row["abstract_variant_id"] = "missing"
    elif fault == "byline":
        row["byline_source_text_ids"] = [999]
    else:
        row["record_id"] = "foreign-record"
    with pytest.raises(ValidationError, match="presentation"):
        PaperExport.model_validate(payload)
