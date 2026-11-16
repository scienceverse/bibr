# Python library

Use `chew()` to extract a paper or a batch, and `Chewer` to reuse a pipeline
across calls. `import bibr` stays light: pipeline stages and their ML
dependencies are loaded lazily.

In Jupyter, or anywhere else an event loop is already running, call
`await bibr.achew(...)` instead of `bibr.chew(...)` (see [Async](#async)).

## One file

```python
import bibr

result = bibr.chew("paper.pdf")

result.title            # metadata fields pass through as attributes
result.references        # list of dicts (alias for the schema's "bib" table)
result.save("paper.json")
```

## Fast mode and the text layer

With nothing configured, `chew()` runs in fast mode: local models only, no LLM
call and no API key. A PDF is read from its own text layer, so no OCR model,
GPU or OCR server is needed (the export records this as
`extraction.ocr.backend == "text-layer"`). That works for born-digital PDFs.
A scanned PDF has no usable text layer and fails with the error code
`scan_needs_ocr` until you choose an OCR backend with `ocr=` or `ocr_url=`
(see [Errors](#errors)).

An LLM key already set in the environment (`GOOGLE_API_KEY`, or the `LLM_*`
settings that `bibr setup` writes) makes the default full mode, which also
asks the LLM. Pass `mode="fast"` to pin fast mode whatever is configured, or
`mode="full"` to require the LLM. `ocr="text-layer"` reads the text layer in
either mode. `bibr setup` is optional: it adds an LLM, an OCR backend or both.

```python
result = bibr.chew("paper.pdf", mode="fast")
result.extraction["ocr"]["backend"]    # "text-layer"
```

## What a Result holds

`chew()` returns a `Result`, an attribute-based view over the export dict
with a validated v12 `PaperExport` model available as `result.model`.
Table-shaped keys (`author`, `affiliation`, `funding`, `text`, `section`,
`url`, `bib`, `xref`, `figure`, `table`, `footnote`, `eq`, `metadata_match`,
`affiliation_match`, `funding_match`, `bib_match`) come back as `Records`, a
`list` of row dicts with a `.df` convenience property for pandas. Its
`repr()` is one line (the row count and the column names), and Jupyter shows
the rows as a table:

```python
result.references.df    # bib rows as a pandas DataFrame
result.authors.df       # alias for the "author" table
result.data              # the raw export dict backing the Result
result.model.metadata.title  # typed Pydantic model access
```

The friendly aliases are `references` (`bib`), `authors` (`author`),
`affiliations` (`affiliation`), `sections` (`section`), `footnotes`
(`footnote`), `figures` (`figure`), `tables` (`table`), `urls` (`url`) and
`equations` (`eq`). `result.text` is the sentence table, one row per sentence
of the body, not the paper as one string. `metadata` fields (`title`, `doi`,
…), `source` fields (`file_name`, `sha256`, `input_format`), and remaining
top-level keys (`paper_id`, `extraction`, …) resolve as attributes too.

`result.paper_id` is the input file's name without its extension
(`paper.pdf` gives `paper`), as in `bibr chew` and `bibr batch`, unless you
pass `paper_id=`. bibr serve gives a paper the first 16 characters of its
SHA-256 instead (see the [REST API](../reference/rest-api.md#post-papersextract)), so
join results from the two by `source.sha256`, not by `paper_id`.

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
raise an exception instead of returning `ChewFailure` (see [Errors](#errors)).

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
| `mode` | `--fast` / `--full` | `"fast"`: local models only, no LLM call and no key; `"full"`: the configured LLM too (raises when none is configured); `"auto"` / omitted: `PIPELINE_MODE`, which is fast when no LLM is configured |
| `ocr` | `--ocr` | OCR backend: `"text-layer"` (the PDF's own text, no OCR model; what a fast run uses when no OCR backend is set), `"paddle"` (picks a local PaddleOCR-VL or GLM-OCR runtime, see [OCR runtime selection](#ocr-runtime-selection-and-diagnostics)), `"paddle-http"`, `"paddle-vllm"`, an explicit `"glm-*"` backend, or a cloud vision backend (`"gemini"`, `"openai"`, `"anthropic"`). An unknown name raises `InvalidOptionError` at once |
| `llm` | `--llm` | LLM backend: `"cloud"`, or a managed local server (`"local"`, `"vllm"`, `"vllm-mlx"`, `"rapid-mlx"`, `"llama-cpp"`, `"llmster"`) |
| `memory` | `--memory` | Memory mode: `"aggressive"`, `"balanced"`, `"keep_all"` (default: selected from hardware) |
| `refs` | `--refs` | Reference parser: `"ner"` (default), `"llm"` (batched LLM), `"llm-chunked"` (region-aligned chunk parsing), `"ref-llm"` (dedicated fine-tuned parser), or `"off"` / `False` to skip references |
| `ref_seg` | `--ref-seg` | Reference segmentation strategy: `"geom"` (default), `"region"`, `"llm"`, `"crf"` |
| `no_llm` | `--no-llm` | Structure only: no LLM call, and no authors or references for a PDF or DOCX; equation extraction, citation linking and Crossref are off too. A PDF still needs an OCR backend. To avoid the LLM and keep the metadata, use `mode="fast"` |
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

`ocr_backend`, `llm_backend` and `memory_mode` are accepted as other names for
`ocr`, `llm` and `memory`. An option name `chew()` does not know raises
`TypeError` listing the valid ones; a value it does not accept (`refs="x"`,
`mode="fastest"`) raises `InvalidOptionError`, also a `ValueError`.

```python
result = bibr.chew("paper.pdf", refs="off", pages="1-5")
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

## Reference lists

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

- **Printed only.** Each field is what the reference printed. A field it does
  not print (the DOI, the pages, the journal) stays empty: bibr never adds a
  value the paper did not print. The registry record only feeds the check
  note.
- **`prefer="record"` (opt-in).** A matched record's values are taken over
  the printed ones, and the record fills what the reference does not print
  (a missing DOI, journal or page range). Each such entry gets a note naming
  the values that came from the record ("bibr: from the Crossref record, not
  the paper: pages, DOI"). This needs enrichment (`--crossref`); without it
  there is no record and the output is as printed.
- **Only a matching record.** With `prefer="record"`, a record is used only
  when it agrees with the reference: bibr compares the two when writing
  (title, first author, year and DOI after folding case, accents and
  punctuation, plus the volume, issue and pages from verification). A
  reference whose record disagrees with it, or that has no record, is written
  as printed; without enrichment, every value is the printed one.
- **Names.** The printed list is split into names (with `prefer="record"`, a
  matched record's family and given names are used instead). "Smith, J. A., &
  Doe, B.", "Smith JA, Doe B", "J. A. Smith and B. Doe", "John Smith,
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
- **Notes.** When a reference and its record disagree, a note names the
  fields that differ, after any note the reference printed. `check_note=False`
  leaves it out. In CSL-JSON the note is `note`, which Zotero keeps in the
  Extra field.

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

The printed type is used (with `prefer="record"`, a matched record's type
unless that type is the catch-all `other`), and a reference without a type that names a journal with a volume, issue or page is a
journal article. Dates, bare
lowercase DOIs, URLs, editors, edition, series, PMID, arXiv identifiers and
"in press" (CSL `status`, BibTeX `pubstate`) are written too. Cite keys are
the first author's family name and the year (`Smith2020`, then `Smith2020a`,
`Smith2020b`), folded to ASCII; keys that differ only in case count as the same
key, as they do in BibTeX.

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
with bibr.Chewer() as chewer:
    r1 = chewer.chew("a.pdf")
    r2 = chewer.chew("b.pdf")   # reuses the pipeline and retained models
```

`Chewer` takes the same options as `chew()`, but fixes them when you create it:
every call uses them, `pages` included (`chewer.chew("a.pdf", pages="1-2")`
raises `TypeError`). The only per-call arguments are `paper_id` and
`batch_size` (and `progress`). Create another `Chewer` for other options. The
pipeline is built lazily on the first call and released on `close()` /
context exit; you can also manage it explicitly:

```python
chewer = bibr.Chewer(refs="off")
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

## Errors

A single-file call raises; in a batch the same failures become `ChewFailure`
slots with the same `error_code`. Every bibr exception is a `bibr.BibrError`,
and the ones you are most likely to catch are importable from `bibr`:

| Exception | Raised when | Also a |
|---|---|---|
| `bibr.InputFileNotFoundError` | the path does not exist (`error_code` `"file_not_found"`), before any model loads | `InputValidationError`, `FileNotFoundError` |
| `bibr.InputValidationError` | the file is refused as input, such as a legacy `.doc` (`error_code` `"invalid_input"`) | |
| `bibr.ProcessingError` | the file could not be processed; `error_code` says why, `failed_stage` where | |
| `bibr.InvalidOptionError` | an option value is not accepted, such as an unknown `ocr=` backend, before any model loads | `ConfigurationError`, `ValueError` |
| `bibr.ConfigurationError` | a setting in `.env` or the environment is invalid, or a local LLM backend cannot start | |
| `bibr.UpstreamServiceError` | an OCR server, LLM or other service the run needs failed | |

`ProcessingError.error_code` values you can act on:

| `error_code` | Meaning |
|---|---|
| `unsupported_format` | Not a file type bibr reads ({{ supported_extensions }}), or content that does not match its extension |
| `corrupted_file` | The file could not be read |
| `encrypted_file` | The PDF is password-protected |
| `scan_needs_ocr` | A scanned PDF (no usable text layer) in a run without an OCR backend: pass `ocr=` or `ocr_url=` |
| `ocr_failed`, `ocr_mostly_failed` | The OCR backend failed, or returned too little text |
| `chunk_error` | The pipeline crashed on this file (batch slots only) |

Other codes name the stage that failed (`layout_failed`, `parse_failed`,
`export_failed`, …). `ProcessingError.outage` (and `ChewFailure.outage`) is
`True` when a service or model the run needs was down: the same file may well
succeed later.

```python
import bibr

try:
    result = bibr.chew("paper.pdf")
except FileNotFoundError as e:
    print("no such file:", e.filename)
except bibr.ProcessingError as e:
    if e.error_code == "scan_needs_ocr":
        print("a scanned PDF: pass an OCR backend, e.g. ocr='anthropic'")
    else:
        print(f"failed at {e.failed_stage}: {e}")
```

These expected errors log no traceback. bibr logs to the `bibr` logger and adds
no handler of its own beyond a `NullHandler`, so a script or notebook that
configures no logging prints nothing extra; the notes a run records are in
`result.extraction["warnings"]`. To see bibr's log, configure logging, for
example `logging.basicConfig(level=logging.INFO)`.

## OCR runtime selection and diagnostics

`ocr="paddle"` asks for a local OCR model and lets bibr pick the runtime. It
is what a full run uses when no OCR backend is set; a fast run reads the text
layer instead. It chooses one runtime at startup. Linux x86_64 tries `paddle-vllm` when an NVIDIA GPU with at least
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
| `aggressive` | Loads/unloads models between phases | Auto-selected for ≤8 GB system RAM, a CUDA GPU below 12 GB of VRAM that a managed local LLM server shares, or one of 8 GB or less (unless nothing in the run can load onto the card: a CPU-only install reading the text layer, or with cloud or remote OCR) |
| `keep_all` | Keeps models loaded for the duration of the run | Explicit opt-in when all selected models fit in memory |

Leave `memory` unset to get the auto-detected default; pass it explicitly
(or set `PIPELINE_MEMORY_MODE`) to override. `bibr setup` does not write
`PIPELINE_MEMORY_MODE`: the mode is chosen for each run.
