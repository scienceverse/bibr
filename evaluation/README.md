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

## Scoring GROBID with the same evaluator

`grobid_run.py` sends a directory of PDFs to a GROBID server with fixed
parameters and records a manifest; `grobid_tei.py` writes GROBID's TEI as
bibr exports, so the unchanged evaluator scores both tools the same way:

```sh
uv run python -m evaluation.grobid_run --grobid-url http://localhost:8070 \
  --pdf-dir /path/to/pdfs --ids-file cohort-ids.txt --out grobid-tei/
uv run python -m evaluation.grobid_tei --tei-dir grobid-tei/ --out grobid-json/
uv run python -m evaluation.evaluate --results-dir grobid-json/ \
  --gold-dirs /path/to/gold --expected-ids grobid-tei/manifest.json \
  --output grobid-eval.json
```

`cohort-ids.txt` lists the paper ids bibr is scored on, one per line, so both
tools are scored against the same `--expected-ids` list. For the headline
comparison, read `pass_rate` in both runs, and compare bibr's
`mean_incl_abstained` with GROBID's `mean`: bibr's `mean` leaves out the
papers where it abstained, and GROBID never abstains. The
[evaluation guide](../docs/contributing/evaluation.md#scoring-grobid-output)
explains both. The converter's module docstring lists what it maps and what it
leaves out.
