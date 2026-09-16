"""Document metrics keep inventory and ownership failures in their denominators."""

from copy import deepcopy

import pytest

from bibr.export.json_export import _export_paper_payload
from evaluation.document_metrics import evaluate_document, select_target_record
from tests.export.conftest import _demo_paper


@pytest.fixture
def sample():
    demo_paper = _demo_paper(with_refs=True)
    records, expected = [], []
    for number, start in enumerate((1, 4), 1):
        paper = _export_paper_payload(demo_paper)
        paper["metadata"].update(
            title=f"Study {number}", doi=f"10.9999/study-{number}", abstract=f"Finding {number}."
        )
        paper["metadata_variant"] = [
            {
                "variant_id": f"v{number}",
                "record_id": f"p{number}",
                "field": "abstract",
                "text": f"Finding {number}.",
                "is_primary": True,
                "language": "en",
                "source_text_ids": [start + 1],
            }
        ]
        records.append(
            {
                "record_id": f"p{number}",
                "status": "extracted",
                "paper": paper,
                "source_text_ids": [start, start + 1, start + 2],
            }
        )
        expected.append(
            {
                "record_id": f"gold-{number}",
                "source_text_ids": [start, start + 1, start + 2],
                "anchor_source_text_ids": [start],
                "abstracts": [
                    {
                        "text": f"Finding {number}.",
                        "language": "en",
                        "source_text_ids": [start + 1],
                        "is_primary": True,
                    }
                ],
            }
        )
    document = {
        "document_schema_version": "1.0",
        "document_id": "document",
        "source": deepcopy(records[0]["paper"]["source"]),
        "status": "complete",
        "records": records,
        "diagnostics": {"detected_record_ids": ["p1", "p2"]},
    }
    gold = {
        "source_file_hash": document["source"]["file_hash"],
        "source_namespace": "frozen-ocr-parser-receipt",
        "annotation_status": "development",
        "records": expected,
    }
    return document, gold


def _score(sample):
    document, gold = sample
    return evaluate_document(document, gold, source_namespace=gold["source_namespace"])


def test_complete_owned_inventory_and_abstracts(sample):
    result = _score(sample)
    assert result["expected_records"] == result["unambiguous_extracted_records"] == 2
    assert result["exact_owned_abstract_rate"] == result["language_accuracy"] == 1
    assert result["primary_abstract_rouge_l"] == 1
    assert result["extra_records"] == result["missing_records"] == 0


def test_missing_paper_stays_in_abstract_and_record_denominators(sample):
    document, _ = sample
    document["records"].pop()
    document["diagnostics"]["detected_record_ids"].pop()
    result = _score(sample)
    assert result["missing_records"] == 1
    assert result["expected_abstracts"] == 2
    assert result["record_detection_recall"] == result["exact_owned_abstract_rate"] == 0.5
    assert result["primary_abstract_rouge_l"] == 0.5
    assert result["unassigned_expected_source_text_ids"] == [4, 5, 6]


def test_missing_entire_export_counts_every_annotated_paper_and_abstract(sample):
    _, gold = sample
    result = evaluate_document(None, gold, source_namespace=gold["source_namespace"])
    assert result["prediction_missing"] is True
    assert result["missing_records"] == result["expected_records"] == 2
    assert result["missing_or_ambiguous_abstracts"] == result["expected_abstracts"] == 2
    assert result["exact_owned_abstract_rate"] == result["primary_abstract_rouge_l"] == 0
    assert select_target_record(None, doi="10.9999/study-1") is None


def test_whitespace_only_gold_cannot_award_a_missing_prediction_an_exact_match(sample):
    _, gold = sample
    gold["records"][0]["abstracts"][0]["text"] = " \t\n"
    with pytest.raises(ValueError, match="whitespace only"):
        evaluate_document(None, gold, source_namespace=gold["source_namespace"])


def test_correct_words_from_foreign_article_do_not_count_as_owned_abstract(sample):
    document, _ = sample
    document["records"][0]["paper"]["metadata_variant"][0]["source_text_ids"] = [5]
    result = _score(sample)
    assert result["primary_abstract_exact_rate"] == 1
    assert result["exact_owned_abstract_rate"] == 0.5
    assert result["abstracts_with_foreign_source"] == 1
    assert result["extra_or_ambiguous_abstracts"] == 1


def test_duplicate_output_is_a_split_and_cannot_get_best_record_scoring(sample):
    document, _ = sample
    duplicate = deepcopy(document["records"][0])
    duplicate["record_id"] = "duplicate"
    for variant in duplicate["paper"]["metadata_variant"]:
        variant["record_id"] = "duplicate"
    document["records"].append(duplicate)
    document["diagnostics"]["detected_record_ids"].append("duplicate")
    result = _score(sample)
    assert result["false_split_records"] == 1
    assert result["overlapping_source_text_ids"] == [1, 2, 3]
    assert result["exact_owned_abstract_rate"] == 0.5
    assert result["extra_or_ambiguous_abstracts"] == 2


def test_merged_paper_can_cover_all_sources_and_still_fail_inventory(sample):
    document, _ = sample
    document["records"][0]["source_text_ids"] = [1, 2, 3, 4, 5, 6]
    document["records"].pop()
    document["diagnostics"]["detected_record_ids"].pop()
    result = _score(sample)
    assert result["owned_source_coverage"] == 1
    assert result["false_merge_records"] == 1
    assert result["unambiguous_extracted_records"] == 0
    assert result["exact_owned_abstract_rate"] == 0


def test_failed_record_is_detected_but_not_an_extraction_success(sample):
    document, _ = sample
    document["records"][0].update(status="failed", paper=None)
    document["status"] = "partial"
    result = _score(sample)
    assert result["record_detection_recall"] == 1
    assert result["unambiguous_extracted_records"] == 1
    assert result["exact_owned_abstract_rate"] == 0.5


def test_retained_partial_fields_do_not_turn_incomplete_paper_into_scoring_success(sample):
    document, _ = sample
    record = document["records"][0]
    partial = record["paper"]
    partial["validation"] = {
        "errors": 1,
        "warnings": 0,
        "blocking": 1,
        "promotable": False,
        "issues": [
            {
                "code": "VAL_METADATA_FIELD_FAILED",
                "severity": "error",
                "message": "Incomplete metadata",
                "origin_stage": "extract",
                "blocking": True,
                "count": 1,
            }
        ],
    }
    record.update(status="unresolved", paper=None, partial_paper=partial)
    document["status"] = "partial"
    result = _score(sample)
    assert result["partial_metadata_records"] == 1
    assert result["unambiguous_extracted_records"] == 1
    assert result["exact_owned_abstract_rate"] == 0.5


def test_missing_language_label_is_counted_and_absence_requires_extracted_evidence(sample):
    document, gold = sample
    document["records"][0]["paper"]["metadata_variant"][0]["language"] = None
    gold["records"][1]["abstracts"] = []
    result = _score(sample)
    assert result["language_accuracy"] == 0
    assert result["correct_abstract_absence_rate"] == 0
    document["records"][1]["paper"]["metadata_variant"] = []
    document["records"][1]["paper"]["metadata"]["abstract"] = ""
    assert _score(sample)["correct_abstract_absence_rate"] == 1


@pytest.mark.parametrize("field", ["source_namespace", "source_file_hash"])
def test_incompatible_gold_is_rejected(sample, field):
    document, gold = sample
    changed = deepcopy(gold)
    changed[field] = "different"
    with pytest.raises(ValueError):
        evaluate_document(document, changed, source_namespace=gold["source_namespace"])


def test_target_selection_uses_declared_identity_without_field_score_search(sample):
    document, _ = sample
    selected = select_target_record(document, doi="10.9999/study-2", titles=("Study 2",))
    assert selected["metadata"]["title"] == "Study 2"
    assert select_target_record(document, doi="10.9999/study-2", titles=("Study 1",)) is None
    document["records"][0]["paper"]["metadata"]["doi"] = "10.9999/study-2"
    assert select_target_record(document, doi="10.9999/study-2") is None
    with pytest.raises(ValueError):
        select_target_record(document)
