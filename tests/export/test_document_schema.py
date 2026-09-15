"""Document output accounts for failed candidates without changing paper tables."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from bibr.export.document_models import (
    DocumentExport,
    DocumentRecordExport,
    build_document_schema,
)
from bibr.export.models import PaperExport

ARTIFACT = Path(__file__).resolve().parents[2] / "docs/schema/bibr-document-v1.schema.json"


@pytest.fixture
def document_paper(v11_payload):
    # The shared shape fixture deliberately removes nondeterministic timestamps.
    # Restore a fixed value because this suite validates complete real exports.
    payload = deepcopy(v11_payload)
    payload["extraction"]["completed_at"] = "2026-09-14T12:00:00Z"
    return payload


def _record(record_id, *, paper=None, status="extracted"):
    return {
        "record_id": record_id,
        "status": status,
        "paper": paper,
        "source_text_ids": [1, 2],
        "source_section_ids": [0, 1, 2],
        "pages": [1, 2],
        "reason_flags": [] if status == "extracted" else ["record_extraction_incomplete"],
        "error": "Extraction failed" if status == "failed" else None,
    }


def _document(paper, records, *, status="complete"):
    return {
        "document_schema_version": "1.0",
        "document_id": paper["source"]["file_hash"],
        "source": deepcopy(paper["source"]),
        "records": records,
        "status": status,
        "diagnostics": {
            "detected_record_ids": [record["record_id"] for record in records],
            "reason_flags": [],
        },
    }


def test_two_papers_keep_independent_tables_and_may_share_external_paper_id(document_paper):
    second = deepcopy(document_paper)
    second["metadata"]["title"] = "A different article with separately owned tables"
    second["author"][0]["family"] = "Other"
    payload = _document(
        document_paper,
        [_record("record-1", paper=document_paper), _record("record-2", paper=second)],
    )

    document = DocumentExport.model_validate(payload)
    exported = document.model_dump(mode="json")

    assert "schema_version" not in exported
    assert exported["document_schema_version"] == "1.0"
    assert exported["records"][0]["paper"]["paper_id"] == second["paper_id"]
    for row, original in zip(exported["records"], (document_paper, second), strict=True):
        expected = PaperExport.model_validate(original).model_dump(mode="json")
        assert row["paper"] == expected
        assert row["paper"]["schema_version"] == "11.1"
        assert isinstance(row["paper"]["metadata"]["abstract"], str)
    assert exported["records"][0]["paper"]["author"][0]["family"] != "Other"
    assert document.records[0].paper is not document.records[1].paper


def test_failed_and_unresolved_candidates_remain_in_partial_output(document_paper):
    rows = [
        _record("good", paper=document_paper),
        _record("failed", status="failed"),
        _record("unresolved", status="unresolved"),
    ]
    document = DocumentExport.model_validate(_document(document_paper, rows, status="partial"))

    assert document.status == "partial"
    assert len(document.records) == 3
    assert document.records[1].paper is None
    assert document.records[1].error == "Extraction failed"
    assert document.records[2].paper is None
    assert document.records[2].source_text_ids == [1, 2]


@pytest.mark.parametrize("statuses", [[], ["failed"], ["unresolved", "failed"]])
def test_no_extracted_papers_is_unresolved_not_complete(document_paper, statuses):
    rows = [_record(f"r{index}", status=status) for index, status in enumerate(statuses)]
    payload = _document(document_paper, rows, status="unresolved")
    assert DocumentExport.model_validate(payload).status == "unresolved"
    payload["status"] = "complete"
    with pytest.raises(ValidationError, match="document status"):
        DocumentExport.model_validate(payload)


@pytest.mark.parametrize("wrong_status", ["complete", "unresolved"])
def test_incomplete_document_cannot_claim_wrong_status(document_paper, wrong_status):
    rows = [_record("good", paper=document_paper), _record("failed", status="failed")]
    with pytest.raises(ValidationError, match="document status must be 'partial'"):
        DocumentExport.model_validate(_document(document_paper, rows, status=wrong_status))


@pytest.mark.parametrize("status", ["unresolved", "failed"])
def test_nonextracted_record_cannot_export_a_paper(document_paper, status):
    with pytest.raises(ValidationError, match="must have paper=null"):
        DocumentRecordExport.model_validate(_record("r1", paper=document_paper, status=status))


def test_extracted_record_cannot_omit_paper_or_supply_null():
    row = _record("r1")
    with pytest.raises(ValidationError, match="require a paper"):
        DocumentRecordExport.model_validate(row)
    row.pop("paper")
    with pytest.raises(ValidationError, match="Field required"):
        DocumentRecordExport.model_validate(row)


def test_failed_record_must_explicitly_emit_null_paper():
    row = _record("failed", status="failed")
    row.pop("paper")
    with pytest.raises(ValidationError, match="Field required"):
        DocumentRecordExport.model_validate(row)


def test_nested_paper_is_validated_against_existing_contract(document_paper):
    malformed = deepcopy(document_paper)
    malformed["metadata"]["abstract"] = ["Two abstracts cannot replace the scalar"]
    with pytest.raises(ValidationError, match="abstract"):
        DocumentRecordExport.model_validate(_record("r1", paper=malformed))


def test_printed_versions_cannot_claim_another_document_record(document_paper):
    document_paper["metadata_variant"] = [
        {
            "variant_id": "v1",
            "record_id": "foreign-record",
            "field": "abstract",
            "text": "An independently printed abstract.",
            "is_primary": False,
        }
    ]
    payload = _document(document_paper, [_record("r1", paper=document_paper)])
    with pytest.raises(ValidationError, match="enclosing record"):
        DocumentExport.model_validate(payload)


def test_duplicate_local_record_ids_are_rejected_even_for_different_papers(document_paper):
    rows = [_record("duplicate", paper=document_paper), _record("duplicate", status="failed")]
    with pytest.raises(ValidationError, match="record_id values must be unique"):
        DocumentExport.model_validate(_document(document_paper, rows, status="partial"))


def test_missing_failed_candidate_cannot_turn_partial_document_into_complete(document_paper):
    payload = _document(document_paper, [_record("good", paper=document_paper)])
    payload["diagnostics"]["detected_record_ids"] = ["good", "failed"]
    with pytest.raises(ValidationError, match="retain every detected_record_id"):
        DocumentExport.model_validate(payload)


def test_inventory_is_required_and_rejects_extra_or_duplicate_candidates(document_paper):
    payload = _document(document_paper, [_record("good", paper=document_paper)])
    payload.pop("diagnostics")
    with pytest.raises(ValidationError, match="diagnostics"):
        DocumentExport.model_validate(payload)
    payload["diagnostics"] = {"reason_flags": []}
    with pytest.raises(ValidationError, match="detected_record_ids"):
        DocumentExport.model_validate(payload)
    payload["diagnostics"]["detected_record_ids"] = []
    with pytest.raises(ValidationError, match="no additional IDs"):
        DocumentExport.model_validate(payload)
    payload["diagnostics"]["detected_record_ids"] = ["good", "good"]
    with pytest.raises(ValidationError, match="detected_record_ids must be unique"):
        DocumentExport.model_validate(payload)


def test_nested_source_must_identify_original_document(document_paper):
    payload = _document(document_paper, [_record("good", paper=deepcopy(document_paper))])
    payload["records"][0]["paper"]["source"]["file_hash"] = "another-document"
    with pytest.raises(ValidationError, match="source must match"):
        DocumentExport.model_validate(payload)


def test_complete_record_cannot_carry_failure_error(document_paper):
    row = _record("good", paper=document_paper)
    row["error"] = "Extraction failed"
    with pytest.raises(ValidationError, match="cannot carry a failure error"):
        DocumentRecordExport.model_validate(row)


@pytest.mark.parametrize("blocking_signal", ["count", "not_promotable", "issue"])
def test_complete_document_cannot_wrap_explicitly_blocking_paper(document_paper, blocking_signal):
    validation = document_paper["validation"]
    if blocking_signal == "count":
        validation["blocking"] = 1
    elif blocking_signal == "not_promotable":
        validation["promotable"] = False
    else:
        validation["issues"][0]["blocking"] = True
    payload = _document(document_paper, [_record("blocked", paper=document_paper)])

    with pytest.raises(ValidationError, match="blocking paper validation"):
        DocumentExport.model_validate(payload)
    jsonschema = pytest.importorskip("jsonschema")
    assert list(jsonschema.Draft202012Validator(build_document_schema()).iter_errors(payload))


def test_nonblocking_warnings_and_errors_do_not_change_extraction_status(document_paper):
    document_paper["validation"]["errors"] = 1
    document_paper["validation"]["issues"][0]["severity"] = "error"
    payload = _document(document_paper, [_record("good", paper=document_paper)])

    assert DocumentExport.model_validate(payload).status == "complete"


def test_schema_artifact_matches_model_and_preserves_paper_dispatch():
    schema = build_document_schema()
    assert json.loads(ARTIFACT.read_text()) == schema
    assert schema["properties"]["document_schema_version"]["const"] == "1.0"
    assert "schema_version" not in schema["properties"]
    assert schema["$defs"]["PaperExport"]["properties"]["schema_version"]["const"] == "11.1"
    assert {"records", "diagnostics", "status"} <= set(schema["required"])


def test_published_schema_validates_real_nested_exports_and_rejects_false_success(document_paper):
    jsonschema = pytest.importorskip("jsonschema")
    schema = build_document_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    payload = _document(
        document_paper,
        [_record("good", paper=document_paper), _record("failed", status="failed")],
        status="partial",
    )
    validator.validate(payload)
    bad_status = deepcopy(payload)
    bad_status["status"] = "complete"
    assert list(validator.iter_errors(bad_status))
    missing_paper = deepcopy(payload)
    missing_paper["records"][0]["paper"] = None
    assert list(validator.iter_errors(missing_paper))
    unowned_paper = deepcopy(payload)
    unowned_paper["records"][1]["paper"] = deepcopy(document_paper)
    assert list(validator.iter_errors(unowned_paper))


def _blocked_paper(paper):
    payload = deepcopy(paper)
    payload["validation"] = {
        "errors": 1,
        "warnings": 0,
        "blocking": 1,
        "promotable": False,
        "issues": [
            {
                "code": "VAL_METADATA_FIELD_FAILED",
                "severity": "error",
                "message": "A metadata field failed",
                "origin_stage": "extract",
                "evidence_ids": ["field:title"],
                "count": 1,
                "blocking": True,
            }
        ],
    }
    return payload


@pytest.mark.parametrize("status", ["unresolved", "failed"])
def test_blocked_partial_export_is_separate_from_success_in_runtime_and_schema(
    document_paper, status
):
    row = _record("blocked", status=status)
    row["partial_paper"] = _blocked_paper(document_paper)
    payload = _document(document_paper, [row], status="unresolved")
    parsed = DocumentExport.model_validate(payload)
    assert parsed.records[0].paper is None
    assert (
        parsed.records[0].partial_paper.author == PaperExport.model_validate(document_paper).author
    )
    assert parsed.records[0].partial_paper.validation.promotable is False
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator(build_document_schema()).validate(payload)


@pytest.mark.parametrize(
    "fault", ["successful-record", "no-validation", "promotable", "no-count", "no-blocking-issue"]
)
def test_partial_export_cannot_claim_success_or_omit_blocking_evidence(document_paper, fault):
    partial = _blocked_paper(document_paper)
    row = _record("blocked", status="unresolved")
    if fault == "successful-record":
        row = _record("blocked", paper=document_paper)
    elif fault == "no-validation":
        partial.pop("validation")
    elif fault == "promotable":
        partial["validation"]["promotable"] = True
    elif fault == "no-count":
        partial["validation"]["blocking"] = 0
    else:
        partial["validation"]["issues"][0]["blocking"] = False
    row["partial_paper"] = partial
    payload = _document(
        document_paper, [row], status="complete" if fault == "successful-record" else "unresolved"
    )
    with pytest.raises(ValidationError):
        DocumentExport.model_validate(payload)
    jsonschema = pytest.importorskip("jsonschema")
    assert list(jsonschema.Draft202012Validator(build_document_schema()).iter_errors(payload))


@pytest.mark.parametrize("fault", ["source", "record-ownership"])
def test_partial_export_has_the_same_document_and_record_ownership_guards(document_paper, fault):
    partial = _blocked_paper(document_paper)
    row = _record("blocked", status="unresolved")
    if fault == "source":
        partial["source"]["file_hash"] = "another-document"
    else:
        partial["metadata_variant"] = [
            {
                "variant_id": "v1",
                "record_id": "other-record",
                "field": "title",
                "text": "Another article",
                "is_primary": False,
            }
        ]
    row["partial_paper"] = partial
    with pytest.raises(ValidationError, match="source must match|enclosing record"):
        DocumentExport.model_validate(_document(document_paper, [row], status="unresolved"))


def test_old_document_records_without_partial_paper_remain_readable(document_paper):
    rows = [_record("good", paper=document_paper), _record("failed", status="failed")]
    parsed = DocumentExport.model_validate(_document(document_paper, rows, status="partial"))
    assert all(row.partial_paper is None for row in parsed.records)
