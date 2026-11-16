# bibr examples

- [Python library](python_library_demo.ipynb): `await bibr.achew_file()` in
  Jupyter on the bundled sample paper, then the metadata, tables and references.
- [Python REST client](python_api_demo.ipynb): call your local `bibr serve` instance.
- [R REST client](r_api_demo.qmd): the same API through `httr2`.
- [R library and JSON](r_library_demo.qmd): call `bibr.chew_file()` through
  reticulate, or read an existing export with the shared R helper.

None of them needs `bibr setup`: with nothing configured, bibr runs in fast mode
(local models only, no API key) and reads each PDF's own text layer. The
notebooks are written for a clone of the bibr repository: they link to its
`docs/`, and the R library notebook reads `../scripts/read_json_response.R` and
uses `../.venv` (from `uv sync`) when it exists. Run them from this folder.

Set `PAPER_PATH` to an input paper (the Python library notebook falls back to
the bundled sample). REST clients default to localhost; set `BIBR_API_URL` for
another bibr server and `BIBR_API_KEY` to match its `AUTH_API_KEY` if
authentication is enabled. To start with an existing JSON export in the R
library notebook, skip its extraction section and set `BIBR_JSON_PATH`.

The separately deployed Scienceverse platform has its own `/jobs` API and
credentials. Its optional clients live under [examples/platform](../examples/platform/README.md).
