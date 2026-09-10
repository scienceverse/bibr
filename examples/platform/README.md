# Scienceverse platform integration examples

These clients target the separately deployed Scienceverse platform's `/jobs`
API. They require a platform account and `PLATFORM_API_KEY`; that key is distinct
from the `AUTH_API_KEY` accepted by `bibr serve`. The platform is optional.

For local bibr usage, see the [Python REST notebook](../../notebooks/python_api_demo.ipynb),
[R REST notebook](../../notebooks/r_api_demo.qmd), or
[R library notebook](../../notebooks/r_library_demo.qmd).

Set `PLATFORM_API_URL` for your deployment and `PLATFORM_API_KEY` in the environment.
For notebooks, also set `PAPER_PATH` to your input file. All Python notebooks are
saved without execution output. Run an individual client from the repository root:

```bash
uv run python examples/platform/test_platform_api.py paper.pdf
Rscript examples/platform/test_platform_api.R paper.pdf
```

Open the Quarto notebook from this directory so its relative helper path resolves.
Bulk maintainer tools live under [evaluation/platform](../../evaluation/platform/README.md).
