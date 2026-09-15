# Scoring your own extractions

This directory contains generic metadata, section, and validation metrics.
It does not include reference corpora, predictions, or experiment runners.

From a development checkout, supply your own independently prepared gold JSON:

```sh
uv run python -m evaluation.evaluate \
  --results-dir /path/to/predictions \
  --gold-dirs /path/to/gold \
  --output /path/to/metrics.json
```

See the [evaluation guide](../docs/contributing/evaluation.md) for the input
format, cohort accounting, metric definitions, and regression gates.

## Documents containing several papers or printed abstracts

`document_metrics` scores the document envelope separately from the existing
paper floors. Supply a complete independently annotated article inventory; paper
gold for only one target cannot establish document completeness.

```sh
uv run python -m evaluation.document_metrics \
  --prediction /path/to/document.json \
  --gold /path/to/document-gold.json \
  --source-namespace OCR_AND_PARSER_RECEIPT_HASH \
  --output /path/to/document-metrics.json
```

Gold follows `DocumentGold` in `document_metrics.py`. It contains
`source_file_hash`, `source_namespace`, `annotation_status` (`development` or
`independently_reviewed`) and `records`. Each record declares a unique
`record_id`, all owned `source_text_ids`, a subset of
`anchor_source_text_ids` marking its identity, and every printed `abstracts`
entry. Each abstract declares `text`, owned `source_text_ids`, optional
source-supported `language` and `is_primary`; exactly one abstract is primary
when any exist. An empty list explicitly annotates abstract absence.

Source IDs must come from the same frozen OCR/parser namespace as the prediction.
Pass the prediction's receipt identity as `--source-namespace`; the evaluator
refuses a different gold namespace or PDF hash. It cannot independently verify a
caller-supplied receipt or that annotations have received human review.

Associations use source anchors, without consulting metadata accuracy. Missing,
failed, merged and split records remain in abstract denominators. Exact owned
abstracts require matching text and source IDs; scalar abstract accuracy is
reported separately. A merged record can cover all source text and still fail
the inventory measures. Unknown predicted languages count as incorrect when the
gold explicitly labels a language. No new metrics change existing paper floors.
An absent prediction file counts every gold record and abstract as missing;
`evaluate_document(None, gold, ...)` exposes the same behavior to Python callers.
Retained `partial_paper` payloads are counted separately and never turn an
incomplete record into an extraction or abstract-scoring success.

For a legacy target-paper comparison, use `select_target_record` with a declared
DOI and/or printed title aliases before invoking the paper scorer. Ambiguous or
missing targets return `None`; do not choose whichever returned paper scores best.
