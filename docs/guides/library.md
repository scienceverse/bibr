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
with a validated v11 `PaperExport` model available as `result.model`.
Table-shaped keys (`bib`, `author`, `text`, `section`, `url`, `bib_match`,
`xref`, `figure`, `table`, `eq`) come back as `Records`, a `list` subclass
with a `.df` convenience property for pandas:

```python
result.references.df    # bib rows as a pandas DataFrame
result.authors.df       # alias for the "author" table
result.data              # the raw export dict backing the Result
result.model.metadata.title  # typed Pydantic model access
```

`references`, `authors`, and `sections` are friendly aliases for the
schema's `bib`, `author`, and `section` tables; `metadata` fields (`title`,
`doi`, …), `source` fields (`file_name`, `file_hash`, `input_format`), and
remaining top-level keys (`paper_id`, `extraction`, …) resolve as attributes
too.

## Several papers in one document

Use `bibr.chew_document("proceedings.pdf")` (or `await bibr.achew_document(...)`)
to return every detected article in a `DocumentResult`. Its `records` list retains
successful, unresolved and failed candidates; each successful `record.paper` is
a regular `Result`. `document.save("proceedings.json")` writes the document
envelope. The same operation is available as `LocalPipeline.process_document()`.
See [printed versions and multiple articles](printed-versions.md) for article
boundaries, alternate abstracts, status meanings and the document schema.

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

## Options

`chew()` (and `Chewer`, below) accept keyword options that mirror the `bibr
chew` CLI flags:

| Option | CLI equivalent | Description |
|---|---|---|
| `ocr` | `--ocr` | OCR backend, e.g. `"paddle"` (default selector), `"paddle-http"`, `"paddle-vllm"`, or an explicit `"glm-*"` compatibility backend |
| `llm` | `--llm` | LLM backend: `"cloud"`, or a managed local server (`"local"`, `"vllm"`, `"vllm-mlx"`, `"rapid-mlx"`, `"llama-cpp"`, `"llmster"`) |
| `memory` | `--memory` | Memory mode: `"aggressive"`, `"balanced"`, `"keep_all"` (default: selected from hardware) |
| `refs` | `--refs` | Reference parser: `"ner"` (default), `"llm"` (batched LLM), `"llm-chunked"` (region-aligned chunk parsing), or `"off"` / `False` to skip references |
| `ref_seg` | `--ref-seg` | Reference segmentation strategy: `"geom"` (default), `"region"`, `"llm"`, `"crf"` |
| `no_llm` | `--no-llm` | Skip metadata LLM calls, equation extraction, citation linking, and Crossref; native metadata survives. OCR still uses the selected backend |
| `device` | `--device` | Force compute device: `"cuda"`, `"mps"`, `"cpu"` |
| `crossref` | `--crossref` / `--no-crossref` | Tri-state: `True` runs Crossref/resolver reference enrichment for this call, `False` skips it, omitted/`None` follows `CROSSREF_ENRICH` (off by default) |
| `equations` | `--no-equations` (inverted) | Enable/disable equation extraction |
| `pages` | `--pages` | Page range to process, 1-based (e.g. `"1-5"`) |
| `figure_images` | `--figure-images` | Include base64-encoded figure images in the output |
| `include_regions` | `--regions` | Include the `extraction.regions` debug payload (per-region bbox/font/content) |
| `include_region_meta` | `--region-meta` | Include the per-text `_bbox_2d`/`_font_size`/`_region_type`/... underscore fields (opt-in v4-training metadata, distinct from `extraction.regions`) |
| `ocr_url` | `--ocr-url` | URL for an external OCR server |
| `ocr_model` | `--ocr-model` | OCR model path or served model alias |
| `ocr_profile` | `--ocr-profile` | `"paddle"` or `"glm"`; required when a custom model alias does not identify its family |
| `start_page`, `end_page` | `--pages` | Lower-level zero-based, inclusive page indices; use these or `pages`, not both |
| `paper_id` | `--paper-id` | Paper ID override (single-file calls only) |
| `batch_size` | `--batch-size` | Files per chunk in batch processing (batch calls only) |
| `consolidate` | `--consolidate` | Merge accepted Crossref matches into `bib` before export: `True` / `"fill"` fills only missing fields, `"replace"` also overwrites disagreeing ones; `False` forces it off |
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

A bare `ocr_url` selects the GLM HTTP compatibility path, so specify
`ocr="paddle-http"` when connecting to Paddle.

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

A bare `ocr_url` selects the GLM HTTP compatibility path, so specify
`ocr="paddle-http"` when connecting to Paddle.

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
