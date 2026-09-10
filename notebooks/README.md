# bibr examples

- [Python REST client](python_api_demo.ipynb): call your local `bibr serve` instance.
- [R REST client](r_api_demo.qmd): the same API through `httr2`.
- [R library and JSON](r_library_demo.qmd): call `bibr.chew_file()` through
  reticulate, or read an existing export with the shared R helper.

Set `PAPER_PATH` to an input paper. REST clients default to localhost; set
`BIBR_API_URL` for another bibr server and `BIBR_API_KEY` to match its
`AUTH_API_KEY` if authentication is enabled. Run Quarto notebooks from this
folder so relative helper and virtual environment paths resolve. To start with
an existing JSON export in the R library notebook, skip its extraction section
and set `BIBR_JSON_PATH`.

The separately deployed Scienceverse platform has its own `/jobs` API and
credentials. Its optional clients live under [examples/platform](../examples/platform/README.md).
