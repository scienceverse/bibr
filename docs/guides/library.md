# Python library

Use `chew()` to extract a paper or a batch, and `Chewer` to reuse a pipeline
across calls. `import bibr` stays light: pipeline stages and their ML
dependencies are loaded lazily.

## One file

```python
import bibr

result = bibr.chew("paper.pdf")

result.title            # metadata fields pass through as attributes
result.references        # list of dicts (alias for the schema's "bib" table)
result.save("paper.json")
```

`chew()` returns a `Result`, an attribute-based view over the export dict
with a validated v12 `PaperExport` model available as `result.model`.
Table-shaped keys (`author`, `affiliation`, `funding`, `text`, `section`,
`url`, `bib`, `xref`, `figure`, `table`, `eq`, `metadata_match`, `bib_match`,
`bib_check`)
come back as `Records`, a `list` subclass
with a `.df` convenience property for pandas:

```python
result.references.df    # bib rows as a pandas DataFrame
result.authors.df       # alias for the "author" table
result.data              # the raw export dict backing the Result
result.model.metadata.title  # typed Pydantic model access
```

`references`, `authors`, and `sections` are friendly aliases for the
schema's `bib`, `author`, and `section` tables; `metadata` fields (`title`,
`doi`, …), `source` fields (`file_name`, `sha256`, `input_format`), and
remaining top-level keys (`paper_id`, `extraction`, …) resolve as attributes
too.

`bibr.Result(data)` also wraps a saved export dict, such as
`json.loads(Path("paper.json").read_text())`. It accepts an export from any
12.x release, including a newer one that adds fields or bumps the minor
`schema_version` or adds enum values. Unknown keys are kept in `result.data`
and in the model's `model_extra`. A `schema_version` with a different major version (`11.x`,
`13.x`) raises a validation error.

## Batch

Directories and lists of paths run as a batch on a single pipeline,
reusing resources according to the selected memory mode:

```python
results = bibr.chew(["a.pdf", "b.pdf"])   # list of paths -> order-aligned list
results = bibr.chew("papers/")          # sorted supported files, non-recursive
```

A failure in one file doesn't abort the batch — its slot holds a
`ChewFailure` instead of a `Result`, in place, so the output list always
lines up with the input:

```python
good = [r for r in results if r.ok]   # ChewFailure.ok is False

for r in results:
    if isinstance(r, bibr.ChewFailure):
        print(r.path, r.error, r.failed_stage)
```

`ChewFailure.outage` is `True` when an OCR or LLM service was down or
unreachable, or a model or server could not start: the failure most likely
says nothing about the file, which may well succeed later. If the pipeline crashes on a
chunk of files, the files it left unfinished run again one by one, and only a
file that crashes it on its own fails, with `error_code="chunk_error"`. Files
that share a stem (`a/paper.pdf`, `b/paper.pdf`, `paper.xml`) get the ids
`bibr batch` gives them, `<stem>-<sha256[:8]>`, so a batch's results can be
joined and passed to `bibr.write_tables()`.

For a list input, results keep the input order. For a directory input,
files are processed in sorted order and anything that isn't
{{ supported_extensions }} is skipped. An empty list returns `[]`; a directory
with no supported files raises `ValueError`. Single-file failures
raise an exception instead of returning `ChewFailure`.

For a fixed return type, use `chew_file()` and `chew_many()`:

```python
result = bibr.chew_file("paper.pdf")          # Result
results = bibr.chew_many(["a.pdf", "b.pdf"])  # list[Result | ChewFailure]
```

Their async equivalents are `achew_file()` and `achew_many()`. The same
four methods are available on `Chewer`. File-specific methods reject
directories; batch-specific methods take an explicit sequence of paths.

## Corpus tables (Parquet)

`bibr.write_tables()` writes any number of results, export dicts or export
JSON files as one Parquet file per table: `paper.parquet` (one row per paper),
one file per record and match table (`author`, `bib`, `text`, `bib_match`, …)
and `extraction_*` files for the processing lists. Every row starts with
`paper_id`, and the column types come from the schema, so every file has the
same columns however many papers it holds. Failed slots of a batch are skipped,
and so are JSON files that are not 12.x exports (a receipt, an export of
another major version); the report lists them in `skipped`.

```python
report = bibr.write_tables(bibr.chew("papers/"), "tables/")

import pandas as pd
refs = pd.read_parquet("tables/bib.parquet")        # every paper's references
papers = pd.read_parquet("tables/paper.parquet")    # titles, DOIs, file hashes, …
```

In R: `arrow::read_parquet("tables/bib.parquet")`. The CLI equivalent is
`bibr tables results/ --out tables/`, which leaves out what `bibr chew -o`
writes beside each export `x.json` (`x.core.json`, `x.json.receipt.json`,
`x.json.enrichment.json`, `_quarantine/`), and `bibr batch` writes the same
files to `<out>/tables/` after every run. `paper_id` joins the tables, so it
must be unique across the corpus. It defaults to the input file's name without
its extension, so `a/paper.pdf` and `b/paper.pdf` clash: chew them in one batch
call (or with `bibr batch`), which gives such files `<stem>-<sha256[:8]>` ids,
or pass `paper_id=` to a single-file call.

## Options

`chew()` (and `Chewer`, below) accept keyword options that mirror the `bibr
chew` CLI flags:

| Option | CLI equivalent | Description |
|---|---|---|
| `ocr` | `--ocr` | OCR backend, e.g. `"paddle"` (default selector), `"paddle-http"`, `"paddle-vllm"`, or an explicit `"glm-*"` compatibility backend |
| `llm` | `--llm` | LLM backend: `"cloud"`, or a managed local server (`"local"`, `"vllm"`, `"vllm-mlx"`, `"rapid-mlx"`, `"llama-cpp"`, `"llmster"`) |
| `memory` | `--memory` | Memory mode: `"aggressive"`, `"balanced"`, `"keep_all"` (default: selected from hardware) |
| `refs` | `--refs` | Reference parser: `"ner"` (default), `"llm"` (batched LLM), `"llm-chunked"` (region-aligned chunk parsing), `"ref-llm"` (dedicated fine-tuned parser), or `"off"` / `False` to skip references |
| `ref_seg` | `--ref-seg` | Reference segmentation strategy: `"geom"` (default), `"region"`, `"llm"`, `"crf"` |
| `no_llm` | `--no-llm` | Skip metadata LLM calls, equation extraction, citation linking, and Crossref; native metadata survives. OCR still uses the selected backend |
| `device` | `--device` | Force compute device: `"cuda"`, `"mps"`, `"cpu"` |
| `crossref` | `--crossref` / `--no-crossref` | Tri-state: `True` runs Crossref/resolver reference enrichment for this call, `False` skips it, omitted/`None` follows `CROSSREF_ENRICH` (off by default) |
| `equations` | `--no-equations` (inverted) | Enable/disable equation extraction |
| `pages` | `--pages` | Page range to process, 1-based (e.g. `"1-5"`) |
| `figure_images` | `--figure-images` | Include figure images in the output, as `data:` URIs |
| `include_regions` | `--regions` | Include the `extraction.regions` debug payload (per-region bbox/font/content) |
| `include_region_meta` | `--region-meta` | Include `extraction.text_regions`: per-sentence layout features (`bbox`, `font_size`, `region_type`, …) keyed by `text_id` (opt-in v4-training metadata, distinct from `extraction.regions`) |
| `ocr_url` | `--ocr-url` | URL for an external OCR server: Paddle (`paddle-http`) unless `ocr` (or, without `ocr`, `OCR_BACKEND`) names a GLM backend, then `glm-http`; an explicit cloud vision `ocr` ignores it |
| `ocr_model` | `--ocr-model` | OCR model path or served model alias |
| `ocr_profile` | `--ocr-profile` | `"paddle"` or `"glm"`; required when a custom model alias does not identify its family |
| `start_page`, `end_page` | `--pages` | Lower-level zero-based, inclusive page indices; use these or `pages`, not both (both raise `TypeError`) |
| `paper_id` | `--paper-id` | Paper ID override (single-file calls only) |
| `batch_size` | `--batch-size` | Files per chunk in batch processing (batch calls only) |
| `consolidate` | `--consolidate` | Merge accepted Crossref matches into `bib` before export: `True` / `"fill"` fills only missing fields, `"replace"` also overwrites disagreeing ones, but only from a match carrying the reference's printed DOI (a match's catch-all `bib_type` `other` fills a missing type but never replaces a printed one); `False` forces it off |
| `settings` | `.env` / environment | A `GlobalSettings` instance copied into the pipeline at construction |

```python
result = bibr.chew("paper.pdf", ocr="paddle", refs="off", pages="1-5")
```

`refs="off"` is equivalent to `bibr chew paper.pdf --refs off`. It keeps
core metadata and equations while skipping reference segmentation, parsing,
bibliographic citation linking, and inline Crossref enrichment.

Backend and processing defaults also come from `.env` or environment
settings such as `LLM_PROVIDER`, `REF_SEG_STRATEGY`, and
`CROSSREF_CONSOLIDATE`. File-specific options such as `paper_id` and `pages`
are call arguments. See the [Settings reference](../reference/settings.md).

Use a settings snapshot to configure independent pipelines without mutating
the global `Settings` object:

```python
from bibr.config import GlobalSettings

settings = GlobalSettings()
settings.crossref.enrich = False
result = bibr.chew_file("paper.pdf", settings=settings)
```

`Chewer` snapshots settings when it is constructed, even though its pipeline
is built on the first call. Later changes to the source settings do not
change an existing session.

To merge Crossref matches after the fact instead of at extraction time, call
`.consolidate()` on a `Result` — it returns a new `Result` and leaves the
original extracted fields untouched. This only merges matches already
present in `bib_match`; it does not query Crossref:

```python
result = bibr.chew("paper.pdf")
enriched = result.consolidate()          # mode="fill" by default
enriched = result.consolidate("replace")
```

## Reference lists and checks

### Checking references

With Crossref enrichment on, every reference gets one `result.bib_check` row
saying whether it agrees with the record it was matched to:

| `status` | Meaning |
|---|---|
| `matched` | The reference agrees with its record: no major difference and at most one minor one (a different issue number is not counted). |
| `fields_disagree` | The title, first author, year or DOI differs (a major note), or two minor things do. A printed DOI that names another work lands here, and so does one whose record is dated more than a year from the printed year. |
| `doi_unregistered` | doi.org says the printed DOI does not exist. |
| `not_found` | Every lookup answered and Crossref has no record of the work. |
| `unchecked` | Nothing was compared: the DOI is registered with another agency (DataCite, mEDRA), Crossref has no record of a printed DOI that doi.org did not call unregistered, a lookup failed or ran out of time, or the reference gave nothing to look up. |

`not_found` means "not found in Crossref", not that the work does not exist:
books, reports, theses, preprints, datasets, software and many non-English
journals are missing from Crossref or hard to find there. Each row's `notes`
lists the differences as `{field, printed, record, severity}`, with the printed
and the record's values as they are; `retraction` holds the record's
retraction or withdrawal notice whatever the status:

```python
checks = result.bib_check.df
checks[checks.status == "fields_disagree"]
retracted = [row["bib_id"] for row in result.bib_check if row["retraction"]]
```

A title is compared after folding case, accents and punctuation. A subtitle
the reference or the record leaves out, a record title printed whole inside
the reference (when it has at least 20 letters and digits, or the DOI or the
volume and first page identify the work), an abbreviated journal name
("Psychol. Sci.", "BMJ") and a year one off (two when the reference is the
later, print-issue year) all agree. The record's first author must be named in
the reference (a rule such as "——" names the previous reference's first
author); a name spelled slightly differently is a minor note. The other authors may be cut short by "et al."
when the record has more than two. Volume, issue and pages come from
verification (`extraction.diagnostics.verification`); a page range is one
difference, and a page printed without its "S" or "e" prefix agrees.
`extraction.enrichment.lookups` says how each reference's lookup ended.

Enrichment does not take a printed DOI's record as the cited work when the
two are dated more than a year apart. The check still compares them: when the
titles agree, the year is the only note (minor for a book, whose record
carries the year of the edition registered, and for a record two years before
the printed year, an online-first date); when they do not, a DOI note says the
printed DOI names another work. A printed DOI Crossref has no record of is
not searched for by title, so it is `unchecked` unless doi.org says the DOI
does not exist.

### Reference-manager files

`result.to_csl_json()`, `result.to_bibtex()` and `result.to_ris()` give the
reference list in the formats reference managers import: CSL-JSON (Zotero,
Mendeley, Pandoc and other citeproc tools), BibTeX and RIS (EndNote, Zotero,
Mendeley). `bibr.write_references()` writes one file from one result or many,
with results, export dicts or export JSON paths as for `write_tables()`; it
takes the format from the file's extension (`.json`, `.bib`, `.ris`) unless
`format=` says otherwise, and cite keys are unique across the file:

```python
items = result.to_csl_json()                       # a list of dicts
bibr.write_references(result, "refs.bib")
bibr.write_references(bibr.chew("papers/"), "refs.ris")
bibr.write_references(results, "refs.txt", format="csl-json", prefer="record")
```

The export itself is never changed. Which values are written:

- **Printed first.** Each field is what the reference printed; a field it does
  not print (the DOI, the pages, the journal) is filled from its record.
  `prefer="record"` takes the record's values over the printed ones.
- **Only a matching record.** A record is used only when the reference's
  `bib_check` status is `matched`. A reference whose record disagrees with it,
  whose printed DOI does not exist, or that was not found or not checked is
  written as printed. An export without `bib_check` rows (one written before
  export schema 12.3) is checked first when it has `bib_match` rows; without
  enrichment, every value is the printed one.
- **Names.** A matched record lists its authors and editors as family and
  given names, and those are used. Otherwise the printed list is split: "Smith,
  J. A., & Doe, B.", "Smith JA, Doe B", "J. A. Smith and B. Doe", "John Smith,
  Jane Doe", Chicago's "Smith, John, and Jane Doe" and "Smith, J.; Doe, A." are
  read, with particles ("van der Berg") and suffixes ("Jr.") kept with their
  name. Two full names in a row are always two people. An organization ("World
  Health Organization") is one name: a CSL `literal`, a braced BibTeX name, one
  RIS `AU` line. A list that fits none of these styles is written as one such
  name rather than split wrongly. "et al." becomes "and others" in BibTeX and
  is left out of CSL-JSON and RIS. A rule ("——") repeats the previous
  reference's authors.
- **Pages.** An abbreviated last page is written in full ("1267–89" as
  1267–1289); an article number ("e0123456") is the first page only.
- **Notes.** A note says what the check found when it is not `matched` or
  `unchecked` (the fields that differ, an unregistered DOI, not found in
  Crossref) and names a retraction or withdrawal notice of the record, after
  any note the reference printed. `check_note=False` leaves it out. In
  CSL-JSON the note is `note`, which Zotero keeps in the Extra field.

| `bib_type` | CSL-JSON | BibTeX | RIS |
|---|---|---|---|
| `journal_article` | `article-journal` | `@article` | `JOUR` |
| `book` | `book` | `@book` | `BOOK` |
| `book_chapter` | `chapter` | `@incollection` | `CHAP` |
| `conference_paper` | `paper-conference` | `@inproceedings` | `CPAPER` |
| `report` | `report` | `@techreport` | `RPRT` |
| `thesis` | `thesis` | `@phdthesis` | `THES` |
| `dataset` | `dataset` | `@misc` | `DATA` |
| `software` | `software` | `@misc` | `COMP` |
| `preprint` | `article` | `@misc` | `UNPB` |
| `other` | `document` | `@misc` | `GEN` |

A matched reference takes its record's type, unless the record's type is
`other` and the reference printed one; a reference without a type that names a
journal with a volume, issue or page is a journal article. Dates, bare
lowercase DOIs, URLs, editors, edition, series, PMID, arXiv identifiers and
"in press" (CSL `status`, BibTeX `pubstate`) are written too. Cite keys are
the first author's family name and the year (`Smith2020`, then `Smith2020a`,
`Smith2020b`), folded to ASCII.

The CLI equivalent is `bibr references results/ -o refs.bib` (`--format`,
`--prefer record`, `--no-check-note`; without `-o` it writes to standard
output). bibr serve answers `GET /papers/jobs/{id}/references?format=` (see the
[REST API](../reference/rest-api.md#get-papersjobsidreferences)), and the MCP
servers have `export_references` and, over stdio, `save_references` (see
[MCP](mcp.md)).

## Warm sessions

Calling `chew()` repeatedly builds and tears down a pipeline (and its
models) every time. For repeated calls over time — notebooks, queue workers
— `Chewer` keeps a pipeline warm across calls instead:

```python
with bibr.Chewer(ocr="paddle") as chewer:
    r1 = chewer.chew("a.pdf")
    r2 = chewer.chew("b.pdf")   # reuses the pipeline and retained models
```

`Chewer` takes the same options as `chew()`, except `paper_id` and
`batch_size`, which move to `.chew()` / `.achew()` per call. The pipeline is
built lazily on the first call and released on `close()` / context exit; you
can also manage it explicitly:

```python
chewer = bibr.Chewer(ocr="paddle")
try:
    r1 = chewer.chew("a.pdf")
    r2 = chewer.chew("b.pdf")
finally:
    chewer.close()
```

Use one mode per instance: sync `.chew()` calls drive a private event loop
that async calls don't share. Inside a running event loop, use the async
form instead:

```python
async with bibr.Chewer() as chewer:
    r1 = await chewer.achew("a.pdf")
```

Run one call at a time on an instance that manages local servers (local OCR,
`llm="local"`): await each `achew()` before starting the next, because bibr may
stop one managed server to free GPU memory for the next stage.

## Async

Inside Jupyter or any other already-running event loop, `chew()` raises —
use the async twin instead:

```python
result = await bibr.achew("paper.pdf")
```

`achew()` mirrors `chew()`'s signature and options exactly, including batch
input (`await bibr.achew(["a.pdf", "b.pdf"])`) and per-call `paper_id` /
`batch_size`.

### OCR runtime selection and diagnostics

`ocr="paddle"` is the default automatic selector. It chooses one runtime at
startup. Linux x86_64 tries `paddle-vllm` when an NVIDIA GPU with at least
8 GB VRAM is detected, followed by `glm-llama`; other Linux machines go
directly to `glm-llama`. Apple Silicon tries `paddle-rapid-mlx`,
`paddle-mlx-vlm`, `glm-rapid-mlx`, then `glm-llama`.
Fallback is startup-only, so a ready Paddle runtime is not silently replaced
with GLM for a failed recognition call. Pass a concrete `ocr="glm-*"` value
when GLM is the deliberate choice.

The normal external-Paddle configuration uses the served alias and profile:

```bash
OCR_BACKEND=paddle-http
OCR_MODEL=paddle-ocr-vl-1.6
OCR_PROFILE=paddle
```

Custom `ocr_model` aliases must also set `OCR_PROFILE` (`paddle` or `glm`) so
the model prompt and normalizer are unambiguous. Paddle table output is decoded
from OTSL to HTML and formulas are normalized to their LaTeX body. To inspect
those transformations, request `include_regions=True`; `extraction.regions`
preserves the canonical content and `raw_ocr_content`. The export
`extraction.ocr` and OCR cache identity retain the selected
backend/model/profile.

The equivalent per-call configuration is:

```python
result = bibr.chew_file(
    "paper.pdf",
    ocr="paddle-http",
    ocr_url="http://localhost:8080/v1",
    ocr_model="my-paddle-model",
    ocr_profile="paddle",
)
```

An `ocr_url` follows the same rule as `bibr chew --ocr-url`: a bare URL
connects to a Paddle server (`paddle-http`), and `ocr="glm"` (or
`"glm-http"`, or any `"glm-*"` backend) connects to a GLM server
(`glm-http`) instead. Without `ocr`, the configured `OCR_BACKEND` decides
the same way, so `OCR_BACKEND=glm-http` keeps GLM. An explicit cloud vision
backend (`ocr="gemini"`, `"openai"` or `"anthropic"`) ignores `ocr_url`,
logs a warning and drops the URL; set `OCR_VISION_BASE_URL` to change its
endpoint. A cloud vision `OCR_BACKEND` in `.env` never overrides a URL you
pass: the run connects to that server as `paddle-http` and sends no page
image to the cloud provider.

## Escape hatch: `LocalPipeline`

`chew()` and `Chewer` cover single calls, batches, and warm sessions. For
full control over the pipeline — custom orchestration, holding a pipeline
across a larger application, or driving `process_chunk()` directly — build
`LocalPipeline` yourself:

```python
import asyncio
from bibr import LocalPipeline

async def main():
    pipeline = LocalPipeline(memory_mode="balanced")
    try:
        return await pipeline.process_file("paper.pdf")
    finally:
        await pipeline.aclose()

data = asyncio.run(main())
```

The returned dict matches the bibr v{{ schema_version }} JSON schema —
`chew()` wraps that same dict in a `Result` view. See the
[Architecture](architecture.md) guide for how the pipeline stages
(validate, input, structure, extract, enrich, export) fit together.

## Memory modes

The `memory` option (`--memory` on the CLI) controls how aggressively
models are loaded and unloaded to fit available GPU/RAM:

| Mode | Description | Use when |
|---|---|---|
| `balanced` | Keeps layout + segmenter loaded; OCR stays resident unless a local LLM needs the memory | Default when neither low-memory condition below applies |
| `aggressive` | Loads/unloads models between phases | Auto-selected for ≤8 GB system RAM or a CUDA GPU with ≤8 GB VRAM |
| `keep_all` | Keeps models loaded for the duration of the run | Explicit opt-in when all selected models fit in memory |

Leave `memory` unset to get the auto-detected default; pass it explicitly
(or set `PIPELINE_MEMORY_MODE`) to override.
