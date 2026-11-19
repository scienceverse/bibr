# R and Metacheck

bibr is a command-line program. You can use it from R without writing any
Python: run it in a terminal or from R, then read what it writes with
[jsonlite](https://cran.r-project.org/package=jsonlite) or
[arrow](https://cran.r-project.org/package=arrow), or hand it to
[Metacheck](https://www.scienceverse.org/metacheck/).

## Before you start

Install bibr once, as in the [quickstart](../getting-started/quickstart.md).
That creates a project folder (`paper-extraction` in the quickstart) where
`uv run bibr ...` works. With nothing configured, bibr runs in fast mode: local
models only, no LLM and no API key.

In R you need:

```r
install.packages(c("jsonlite", "arrow", "dplyr"))
```

To try everything below without a paper of your own, process bibr's bundled
synthetic sample paper:

```bash
uv run bibr chew --sample -o sample.json
```

## Run bibr

In a terminal, from the project folder:

```bash
uv run bibr chew paper.pdf -o paper.json
```

From R, call the same command with `system2()`. R's working directory must be
the project folder, so that `uv run` finds bibr:

```r
setwd("~/paper-extraction")
status <- system2("uv", c("run", "bibr", "chew", "paper.pdf", "-o", "paper.json"))
if (status != 0) stop("bibr chew failed with exit status ", status)
```

If `system2()` cannot find `uv`, `Sys.which("uv")` is empty: give the full
path to `uv`, or restart R after installing uv so it sees the new `PATH`.

A paper that fails validation badly is not written to `paper.json`: bibr
writes it under `_quarantine/` next to it, prints where it went, and still
exits with status 0. Check `file.exists("paper.json")` before you read it.

For many papers, use `bibr batch` ([below](#read-a-corpus-as-parquet-tables)).

## Read one result

A result is one JSON file in bibr's export schema (version 12.x). Paper-level
fields are under `metadata`; authors, sentences, sections and references are
arrays of records, which jsonlite turns into data frames:

```r
library(jsonlite)

paper <- read_json("paper.json", simplifyVector = TRUE)

paper$schema_version          # "12.5"
paper$metadata$title
paper$metadata$doi            # NULL when the paper prints no DOI
paper$author[, c("given", "family", "orcid")]
paper$bib[, c("bib_id", "authors", "year", "title", "doi")]
```

Each sentence is a row of `paper$text`, with the `section_id` of its section.
Join the two to get, say, the sentences of the methods:

```r
sentences <- merge(
  paper$text,
  paper$section[, c("section_id", "header", "section_type")],
  by = "section_id"
)
sentences$text[sentences$section_type == "method"]
```

`section_type` is bibr's classification of the section: `intro`, `method`,
`results`, `discussion`, `references` and others. Every table and field is
described in the [JSON schema reference](../reference/schema.md).

The bibr repository has a helper,
[`read_json_response.R`](https://github.com/scienceverse/bibr/blob/main/scripts/read_json_response.R),
whose `read_bibr_json()` turns every table into a data frame, empty tables
included.

## Read a corpus as Parquet tables

For a folder of papers, `bibr batch` writes one JSON file per paper and, at the
end of each run, the whole corpus as Parquet tables:

```bash
uv run bibr batch papers/ --out corpus/
```

Running the same command again resumes: papers that already finished are
skipped. The tables are in `corpus/tables/`, one file per table: `paper`,
`author`, `affiliation`, `funding`, `text`, `section`, `url`, `bib`, `xref`,
`figure`, `table`, `footnote`, `eq`, the `*_match` tables, and `extraction_*`
tables of processing details. Every row carries the `paper_id` of its paper,
and every file has the same columns however many papers it holds.

If you already have JSON results, `bibr tables` writes the same files from
them:

```bash
uv run bibr tables out/ --out tables/
```

In R, read each table with `arrow::read_parquet()`:

```r
library(arrow)
library(dplyr)

papers  <- read_parquet("corpus/tables/paper.parquet")
bib     <- read_parquet("corpus/tables/bib.parquet")
text    <- read_parquet("corpus/tables/text.parquet")
section <- read_parquet("corpus/tables/section.parquet")

papers |> select(paper_id, title, doi, paper_type)

bib |> count(paper_id, name = "references")

text |>
  inner_join(section, by = c("paper_id", "section_id")) |>
  filter(section_type == "method") |>
  select(paper_id, header, text)
```

Ids such as `section_id` and `bib_id` count from 1 within each paper, so join
on `paper_id` as well. `extraction_validation_issues.parquet` lists each
paper's validation warnings (`code`, `severity`, `message`), which helps you
find papers to check by hand.

## Reference lists for Zotero and BibTeX

`bibr references` writes the references of one result or a whole folder as one
file for a reference manager. The format follows the file name: `.json` writes
CSL-JSON, `.bib` BibTeX and `.ris` RIS.

```bash
uv run bibr references corpus/ -o references.json
uv run bibr references paper.json -o references.bib
```

Zotero imports all three formats. Values are as the paper printed them; see
`uv run bibr references --help` for filling gaps from Crossref matches.

## Call a bibr server from R

`bibr serve` runs bibr as an HTTP API, which R can call with
[httr2](https://cran.r-project.org/package=httr2). Start it in a terminal in
the project folder:

```bash
uv run bibr serve
```

It listens on `http://127.0.0.1:8000` and, with nothing configured, runs in fast
mode. Send a paper to `POST /papers/extract` as the multipart field `file`; the
response is the same JSON as `bibr chew` writes:

```r
library(httr2)

resp <- request("http://127.0.0.1:8000") |>
  req_url_path_append("papers", "extract") |>
  req_body_multipart(file = curl::form_file("paper.pdf")) |>
  req_timeout(300) |>
  req_perform()

paper <- jsonlite::fromJSON(resp_body_string(resp), simplifyVector = TRUE)
paper$metadata$title
```

A server started with `AUTH_API_KEY` set needs that key as a bearer token: add
`req_auth_bearer_token(key)` to the request. The
[REST API reference](../reference/rest-api.md) lists every endpoint, including
async jobs for long papers, and [production deployment](deployment.md) covers
running a shared server. The repository's
[`call_api.R`](https://github.com/scienceverse/bibr/blob/main/scripts/call_api.R)
and [`r_api_demo.qmd`](https://github.com/scienceverse/bibr/blob/main/notebooks/r_api_demo.qmd)
are longer examples.

## Metacheck

[Metacheck](https://www.scienceverse.org/metacheck/) is the R package bibr was
built for: it checks papers for research best practices. It reads bibr's JSON
directly.

Install it from GitHub, as its README describes:

```r
pak::pkg_install("scienceverse/metacheck")
```

Use Metacheck 0.3.2 or later: it is the first version that reads bibr's export
schema 12.x. Older versions expect the layout of bibr's exports before schema 11
and find no title, abstract or DOI.

Read one paper, or every paper of a `bibr batch` folder:

```r
library(metacheck)

paper <- read("paper.json")
paper$info$title

papers <- read("corpus")
```

Metacheck keeps its own names: the export's `metadata` becomes `paper$info`.
Reading a folder, `read()` takes every `.json` file in it. It skips files that
are not papers with a warning (`run_info.json` in a batch folder). Do not point
it at a folder that `bibr chew -o out/` wrote: each paper there also has a
`.core.json` file with the same `paper_id`, which `read()` would load as
well. `bibr batch` folders do not have these files.

To produce a report, see Metacheck's
[`report()` reference](https://www.scienceverse.org/metacheck/reference/report.html)
and the [Metacheck manual](https://www.scienceverse.org/metacheck_book/).
Metacheck can also send papers to a bibr server itself:
[`convert_bibr()`](https://www.scienceverse.org/metacheck/reference/convert_bibr.html)
with `backend = "selfhosted"` and a `save_path` folder uses `BIBR_API_URL` (default
`http://localhost:8000`) and, for a server with a key, `BIBR_API_KEY`.
