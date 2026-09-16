"""Source-bound document inventory and abstract metrics, separate from paper floors.

Gold must be independently prepared for the same source-ID namespace (the frozen
OCR/parser receipt). PDF identity alone cannot establish that parser-local IDs
mean the same thing. No corpus or expected article identities are bundled here.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bibr.export.document_models import DocumentExport
from evaluation.validation_metrics import abstract_rouge_l, normalize_doi

DOCUMENT_METRICS_VERSION = 1


def _text(value: str | None) -> str:
    return " ".join(unicodedata.normalize("NFKC", value or "").split())


class AbstractGold(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    source_text_ids: list[int] = Field(min_length=1)
    language: str | None = None
    is_primary: bool = False

    @model_validator(mode="after")
    def _nonempty_printed_text(self):
        if not _text(self.text):
            raise ValueError("gold abstract text cannot be whitespace only")
        return self


class RecordGold(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str = Field(min_length=1)
    source_text_ids: list[int] = Field(min_length=1)
    anchor_source_text_ids: list[int] = Field(min_length=1)
    abstracts: list[AbstractGold]

    @model_validator(mode="after")
    def _owned_anchors_and_abstracts(self):
        owned = set(self.source_text_ids)
        if len(owned) != len(self.source_text_ids):
            raise ValueError("gold source_text_ids must be unique")
        if not set(self.anchor_source_text_ids).issubset(owned):
            raise ValueError("gold anchors must belong to their record")
        if sum(item.is_primary for item in self.abstracts) != bool(self.abstracts):
            raise ValueError("each abstract-positive record needs exactly one primary abstract")
        seen: set[int] = set()
        for item in self.abstracts:
            ids = set(item.source_text_ids)
            if not ids.issubset(owned) or ids & seen or len(ids) != len(item.source_text_ids):
                raise ValueError("gold abstract sources must be unique, disjoint and record-owned")
            seen.update(ids)
        return self


class DocumentGold(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_file_hash: str = Field(min_length=1)
    source_namespace: str = Field(min_length=1)
    annotation_status: Literal["development", "independently_reviewed"]
    records: list[RecordGold]

    @model_validator(mode="after")
    def _unique_records_and_sources(self):
        keys = [row.record_id for row in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError("gold record IDs must be unique")
        seen: set[int] = set()
        for row in self.records:
            if seen & set(row.source_text_ids):
                raise ValueError("gold records cannot share source text IDs")
            seen.update(row.source_text_ids)
        return self


def select_target_record(
    document: dict | DocumentExport | None, *, doi: str | None = None, titles: tuple[str, ...] = ()
) -> dict | None:
    """Select a unique target by caller-declared identity before any field scoring.

    Both supplied constraints must match. No fuzzy title score, abstract score,
    reference score or highest-scoring-record fallback participates in selection.
    Ambiguous and absent identities return None and remain scoring failures.
    """
    expected_doi = normalize_doi(doi) if doi else None
    expected_titles = {_text(title).casefold() for title in titles if _text(title)}
    if not expected_doi and not expected_titles:
        raise ValueError("target selection requires a DOI or declared printed title")
    if doi and not re.fullmatch(r"10\.\d{4,9}/\S+", expected_doi or ""):
        raise ValueError("invalid target DOI")
    if document is None:
        return None
    doc = DocumentExport.model_validate(document)
    matches = []
    for record in doc.records:
        paper = record.paper
        if paper is None:
            continue
        if expected_doi and normalize_doi(paper.metadata.doi or "") != expected_doi:
            continue
        printed_titles = {_text(paper.metadata.title).casefold()}
        printed_titles.update(
            _text(row.text).casefold() for row in paper.metadata_variant if row.field == "title"
        )
        if expected_titles and not expected_titles.intersection(printed_titles):
            continue
        matches.append(paper.model_dump(mode="json"))
    return matches[0] if len(matches) == 1 else None


def evaluate_document(
    document: dict | DocumentExport | None,
    gold: dict | DocumentGold,
    *,
    source_namespace: str,
) -> dict:
    """Score all annotated records and abstracts, including missing/failed output.

    Inventory associations use gold source anchors, not extraction field scores.
    A split/merge never gets a best-record choice for abstract scoring. A missing
    abstract contributes zero to the complete gold denominator. Exact abstract
    matching preserves case/punctuation after Unicode/whitespace normalization.
    """
    doc = DocumentExport.model_validate(document) if document is not None else None
    expected = DocumentGold.model_validate(gold)
    if doc is not None and doc.source.file_hash != expected.source_file_hash:
        raise ValueError("prediction and gold refer to different source files")
    if source_namespace != expected.source_namespace:
        raise ValueError("prediction and gold source-ID namespaces differ")
    records = doc.records if doc is not None else []
    links = [
        [
            index
            for index, prediction in enumerate(records)
            if set(prediction.source_text_ids) & set(record.anchor_source_text_ids)
        ]
        for record in expected.records
    ]
    reverse = [
        [index for index, linked in enumerate(links) if prediction in linked]
        for prediction in range(len(records))
    ]
    all_owned = {key for record in expected.records for key in record.source_text_ids}
    all_predicted = {key for record in records for key in record.source_text_ids}
    owners: dict[int, set[int]] = {}
    for index, predicted_record in enumerate(records):
        for key in set(predicted_record.source_text_ids):
            owners.setdefault(key, set()).add(index)
    abstract_total = sum(len(record.abstracts) for record in expected.records)
    language_total = sum(
        abstract.language is not None
        for record in expected.records
        for abstract in record.abstracts
    )
    abstract_exact = language_correct = primary_exact = positive_primary = 0
    primary_rouge_sum = 0.0
    absent_records = correct_absence = 0
    missing_abstracts = extra_abstracts = abstract_leaks = 0
    per_record = []
    scored_predictions: set[int] = set()
    for index, record in enumerate(expected.records):
        linked = links[index]
        unambiguous = len(linked) == 1 and len(reverse[linked[0]]) == 1
        prediction = records[linked[0]] if unambiguous else None
        paper = prediction.paper if prediction is not None else None
        variants = (
            [row for row in paper.metadata_variant if row.field == "abstract"] if paper else []
        )
        if prediction is not None:
            scored_predictions.add(linked[0])
        if record.abstracts:
            positive_primary += 1
            primary = next(row for row in record.abstracts if row.is_primary)
            scalar = paper.metadata.abstract if paper else None
            primary_exact += _text(scalar) == _text(primary.text)
            primary_rouge_sum += abstract_rouge_l(scalar or "", primary.text) or 0.0
        else:
            absent_records += 1
            correct_absence += (
                paper is not None and not _text(paper.metadata.abstract) and not variants
            )
        owned_ids = set(record.source_text_ids)
        abstract_leaks += sum(bool(set(row.source_text_ids) - owned_ids) for row in variants)
        consumed: set[int] = set()
        exact = 0
        for abstract in record.abstracts:
            matching = [
                offset
                for offset, row in enumerate(variants)
                if set(row.source_text_ids) & set(abstract.source_text_ids)
            ]
            if len(matching) != 1:
                missing_abstracts += 1
                continue
            offset = matching[0]
            candidate = variants[offset]
            # A merged pair cannot satisfy two expected abstracts, even if one
            # of their texts happens to equal the combined exported string.
            unique_owner = (
                sum(
                    bool(set(candidate.source_text_ids) & set(item.source_text_ids))
                    for item in record.abstracts
                )
                == 1
            )
            if offset in consumed or not unique_owner:
                missing_abstracts += 1
                continue
            consumed.add(offset)
            source_exact = set(candidate.source_text_ids) == set(abstract.source_text_ids)
            correct = source_exact and _text(candidate.text) == _text(abstract.text)
            exact += correct
            abstract_exact += correct
            language_correct += (
                abstract.language is not None
                and source_exact
                and candidate.language == abstract.language
            )
        extra_abstracts += len(variants) - len(consumed)
        per_record.append(
            {
                "record_id": record.record_id,
                "prediction_record_ids": [records[item].record_id for item in linked],
                "unambiguous": unambiguous,
                "extracted": paper is not None,
                "expected_abstracts": len(record.abstracts),
                "exact_owned_abstracts": exact,
                "missing_source_text_ids": sorted(owned_ids - all_predicted),
                "foreign_source_text_ids": sorted(set(prediction.source_text_ids) - owned_ids)
                if prediction is not None
                else [],
            }
        )
    # Abstracts from unmatched/merged/split predictions remain extra output;
    # none is silently chosen to improve completeness.
    extra_abstracts += sum(
        sum(row.field == "abstract" for row in record.paper.metadata_variant)
        for index, record in enumerate(records)
        if index not in scored_predictions and record.paper is not None
    )
    missing_records = sum(not linked for linked in links)
    return {
        "document_metrics_version": DOCUMENT_METRICS_VERSION,
        "source_file_hash": expected.source_file_hash,
        "source_namespace": source_namespace,
        "source_namespace_binding": "caller_asserted",
        "annotation_status": expected.annotation_status,
        "prediction_missing": doc is None,
        "expected_records": len(expected.records),
        "predicted_records": len(records),
        "partial_metadata_records": sum(record.partial_paper is not None for record in records),
        "missing_records": missing_records,
        "extra_records": sum(not linked for linked in reverse),
        "false_split_records": sum(len(linked) > 1 for linked in links),
        "false_merge_records": sum(len(linked) > 1 for linked in reverse),
        "record_detection_recall": (len(links) - missing_records) / len(links) if links else None,
        "unambiguous_extracted_records": sum(row["extracted"] for row in per_record),
        "owned_source_coverage": len(all_owned & all_predicted) / len(all_owned)
        if all_owned
        else None,
        "overlapping_source_text_ids": sorted(
            key for key, values in owners.items() if len(values) > 1
        ),
        "unassigned_expected_source_text_ids": sorted(all_owned - all_predicted),
        "expected_abstracts": abstract_total,
        "missing_or_ambiguous_abstracts": missing_abstracts,
        "extra_or_ambiguous_abstracts": extra_abstracts,
        "abstracts_with_foreign_source": abstract_leaks,
        "exact_owned_abstract_rate": abstract_exact / abstract_total if abstract_total else None,
        "primary_abstract_exact_rate": primary_exact / positive_primary
        if positive_primary
        else None,
        "primary_abstract_rouge_l": primary_rouge_sum / positive_primary
        if positive_primary
        else None,
        "abstract_absent_records": absent_records,
        "correct_abstract_absence_rate": correct_absence / absent_records
        if absent_records
        else None,
        "language_accuracy": language_correct / language_total if language_total else None,
        "language_scored_abstracts": language_total,
        "per_record": per_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--source-namespace", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate_document(
        json.loads(args.prediction.read_text()) if args.prediction.exists() else None,
        json.loads(args.gold.read_text()),
        source_namespace=args.source_namespace,
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
