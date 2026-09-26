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
  --pdf-dir /path/to/pdfs --out grobid-tei/
uv run python -m evaluation.grobid_tei --tei-dir grobid-tei/ --out grobid-json/
uv run python -m evaluation.evaluate --results-dir grobid-json/ \
  --gold-dirs /path/to/gold --expected-ids grobid-tei/manifest.json \
  --output grobid-eval.json
```

The converter's module docstring lists what it maps and what it leaves out.
