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
