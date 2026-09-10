# Architecture

## Pipeline overview

bibr processes scientific papers through six logical phases. The shared
stage plan lives in `bibr/pipeline/plans.py`; local and served pipelines
use the same parsing, extraction, validation, and export stages.

```
File / bytes --> Validate
                   ├── DOCX / JATS / HTML / ePub --> Native parse ─────────┐
                   └── PDF --> Render --> Layout --> Native text / OCR ──┤
                                                                        v
                                                      Structure / sentence segmentation
                                                                        v
                                                      Extract --> Identity validation
                                                                        v
                                                      Core checkpoint --> Enrich --> Export
```

Two runtime options are available:

- **`bibr chew` / Python library** (`bibr/local/pipeline.py`) -- `LocalPipeline` manages model lifetimes on one machine. OCR uses the selected Paddle-first runtime, an explicit HTTP backend, or a cloud vision provider.
- **`bibr serve`** (`bibr/serve/app.py`) -- LitServe HTTP API with disk-backed
  multipart ingress and one spawned inference worker hosting the pipeline GPU
  models (layout, segmenter).

## Pipeline stages

### 1. Validate (`bibr/input/validate.py`)

Checks MIME type, file corruption, and encryption. Refuses unsupported formats (`.exe`, `.zip`, `.tex`, legacy `.doc`). Supports {{ supported_extensions }} — `.xml` as JATS.

### 2. Input (`bibr/ocr/`, `bibr/input/docx_native.py`)

- DOCX and JATS XML files are parsed natively (python-docx, and a JATS `<article>` parser respectively) to extract headings and text structure, skipping OCR entirely
- HTML (`.html`/`.htm`) and ePub (`.epub`) files are also parsed natively — `HtmlHandlingStage` (`bibr/pipeline/stages/html.py`), backed by `bibr/input/html_native.py` and `bibr/input/epub_native.py` — skipping OCR the same way as DOCX/JATS
- PP-DocLayoutV3 detects PDF regions (headings, body text, tables, formulas, figures, and more); regions then flow through native-text inspection and recognition
- `OCR_BACKEND=paddle` selects one Paddle-first OCR runtime transactionally at startup; explicit `paddle-*`, `glm-*`, and cloud backends remain available through `OcrOptions`

**Bounded PDF processing.** PDFs are rendered in windows of at most
`PIPELINE_PAGE_WINDOW_SIZE` pages (default 8). Each window completes layout,
native inspection, and OCR before its page images are released. The separate
`PIPELINE_MAX_PAGES` setting limits the processed page count (default 200).

**Native text bypass and recognition.** `bibr/ocr/pdf_inspection.py` inspects
embedded PDF text, metadata, outline headings, and reference-line geometry
under one PDFium walk. With `OCR_NATIVE_TEXT_ENABLED=true` (the default),
qualifying text regions are filled from the PDF text layer and bypass OCR.
Acceptance uses character-count and printable-text checks
(`OCR_NATIVE_TEXT_MIN_CHARS=20`, `OCR_NATIVE_TEXT_MIN_PRINTABLE_RATIO=0.85`)
plus guards for corrupt text. Remaining regions use the selected OCR runtime.
Disabling the bypass leaves metadata, outline, and geometry inspection available.

Experimental selective line and inline-formula repair is controlled by
`OCR_NATIVE_REPAIR_ENABLED` (default `false`). Native caption acceptance has
a separate `OCR_NATIVE_CAPTIONS_ENABLED` switch and requires repair to be
enabled. These options are distinct from the default native-text bypass.

### 3. Structure (`bibr/structure/pdf_parser.py`)

`PDFParser` converts OCR regions into `PaperContents`:

- Maps region labels to content handlers via `LABEL_TREATMENT` dispatch dict
- Runs sentence segmentation through the pipeline's wtpsplit-lite segmenter (local or served)
- Reconstructs heading hierarchy, incorporates PDF outline hints, and handles multi-study scopes
- Extracts tables, figures, footnotes, URL links, and inline figure/table references
- Assigns captions and preserves ordered physical parts when figures or tables span multiple regions/pages

### 4. Extract (`bibr/extract/extractor.py`, `bibr/paper.py`)

Post-parse pipeline runs after structure parsing:

- **Section classification** -- three-tier cascade maps headers to canonical IMRaD categories: alias lookup table, then a trained classifier model, then LLM fallback (`section_classifier.py`; see [Classifiers](classifiers.md))
- **Study hierarchy** -- regex markers such as Study 1 and Experiment A establish separate section scopes before classification (`section_tree.py`)
- **Metadata extraction** -- selected front-matter rows ground title, authors, abstract, DOI, and publication fields. JATS and HTML/ePub can supply preparsed metadata, avoiding the core metadata LLM call
- **Paper classification** -- the default SPECTER2 multitask model predicts paper type and OECD domains from title/abstract; confidence gates and LLM fallback are described in [Classifiers](classifiers.md)
- **Reference extraction** -- segmentation (default `geom`, a local geometry model, cascading through region anchors -> LLM -> CRF when geometry is absent or unconfident) locates each reference; parsing (default `ner`, a local ModernBERT-CRF model, with `llm` for opt-in batched LLM parsing) extracts structured fields
- **Citation linking** -- 3-tier hybrid approach: numeric bracket/superscript citations, author-year citations, then LLM fallback (`citation_linker.py`)
- **Equation extraction** -- regex + LLM fallback for statistical reporting decomposition (`equation_extractor.py`)
- **IMRaD enforcement** -- deduplication of Abstract and References by classification-source trust; repeated Methods/Results/Discussion remain valid
- **Source ownership** -- abstract spans, author grounding, and integrity-statement evidence are resolved against the selected article block. Ambiguous front matter can yield validation issues instead of metadata taken from another article in the file

Identity validation and a core checkpoint run before enrichment. Extraction
and enrichment have separate completion evidence, so a failed or delayed
external lookup need not force OCR and extraction to run again. The
integrity-statement resolver defaults to `PIPELINE_INTEGRITY_STATEMENT_MODE=shadow`:
it records typed evidence while preserving the compatibility scalar fields.

### 5. Enrich (`bibr/enrich/references.py`)

Optional enrichment of the paper's own identity and extracted references:

- DOI lookup for direct matches
- Bibliographic search and title-less bibliographic fingerprint matching where applicable
- Optional `bibr-resolver` integration for additional lookup sources
- Matches stay separate in `info_match` and `bib_match`; printed metadata is preserved
- `CROSSREF_ENRICH` controls inline enrichment. `CROSSREF_CONSOLIDATE=fill|replace` explicitly merges accepted matches into `bib` during export; the default is `off`
- `POST /papers/enrich` can backfill an existing 10.6/10.7 export without rerunning extraction; `Result.consolidate()` only merges matches already present

### 6. Export (`bibr/export/`)

**JSON** (`bibr/export/json_export.py`):

- JSON-serializable dict matching the bibr v{{ schema_version }} paper schema
- Top-level keys include: `paper_id`, `info`, `author`, `text`, `section`, `url`, `bib`, `bib_match`, `info_match`, `funding`, `affiliations`, `xref`, `citation_linking`, optional `caption_assignment`, optional `reference_yield`, `figure`, `table`, `eq`, `ocr_config`, `enrichment`, `extraction`, `processing_warnings`, `llm_usage`, `llm_usage_by_label`, `qualification_provenance`, `validation`. Figure/table rows retain legacy primary fields and add ordered physical `parts` with provenance.
- Schema version: `{{ schema_version }}`
- `info` is scalar-only (no nested objects or lists of objects) so R consumers can `as.data.frame(info)`. Pipeline metadata (`ocr_config`, `processing_warnings`) lives at the top level.
- All positional IDs are 1-based; `section_id=0` is the Root sentinel (excluded from export)
- Enrichment matches are in a separate top-level `bib_match` array (flat, keyed by `bib_id` + `service`)
- Optional `_regions` debug payload (per-region bbox/font/content) is opt-in via `include_regions=True` on `Paper.export_to_json()` / `export_paper_to_json()` / the `include_regions` form field on `POST /papers/extract` / the `--regions` CLI flag. It also preserves `_raw_ocr_content` when Paddle normalization changed a table or formula response, so diagnostics can compare the original model output with canonical content. The same option includes `_native_source`: detached PDF characters with stable source IDs, geometry, fonts, rotation, geometric ownership and raster coverage diagnostics. Region records link to that evidence through source IDs and retain typed native/repair spans. Geometric coverage and recognized spans do not establish fidelity.

### OCR selection and evidence

PDF OCR starts with PP-DocLayoutV3 regions. `OCR_BACKEND=paddle` then selects a
concrete runtime transactionally at startup: Linux x86_64 tries
`paddle-vllm` (PaddleOCR-VL-1.6) when an NVIDIA GPU with at least 8 GB VRAM
is detected, then `glm-llama`; without that GPU it uses `glm-llama` directly.
Apple Silicon tries
`paddle-rapid-mlx`, `paddle-mlx-vlm`, `glm-rapid-mlx`, then `glm-llama`.
The selected backend/model/profile becomes the OCR runtime identity used in
the OCR-cache key and export `ocr_config`, so cache entries and provenance
cannot be confused across recognizers or normalizers. There is no silent
per-request GLM fallback after a concrete runtime has passed startup.

Paddle table output uses OTSL markers (such as `<fcel>`, `<lcel>`, `<nl>`, and
`<ecel>`) that bibr decodes into canonical HTML. Paddle formula output has one
outer Markdown/LaTeX fence or balanced display delimiter removed; the LaTeX
body is otherwise preserved. The normalized value feeds parsing, while raw
Paddle output remains available in `_raw_ocr_content` through `_regions`.

## Key data structures

### PaperContents (`bibr/paper_contents.py`)

Holds parsed text content built by the native input parsers or `PDFParser`:

- `sections` -- list of `PaperSection` (header, section_type, classification_score)
- `sentences` -- list of `PaperSentence` (text, section_id, paragraph_id, page_number)
- `tables` -- list of `PaperTable` (caption, HTML markup, cells, physical parts)
- `links` -- list of `PaperURLLink`
- `figures` -- list of `PaperFigure` (caption, optional base64 image, physical parts)
- `xrefs` -- list of `PaperXref` (inline citations, table/figure refs)
- `equations` -- list of `PaperEquation` (lhs, comp, rhs)
- `text_df`, `links_df`, `sentences_df`, `equations_df` -- cached DataFrame properties

### Paper (`bibr/paper.py`)

Top-level dataclass wrapping `PaperContents` + `PaperMetadata`:

- `metadata` -- `PaperMetadata` (title, authors, DOI, keywords, references, matches, paper type, OECD domain)
- `contents` -- `PaperContents`
- `export_to_json()` -- serialize to dict

### PaperMetadata (`bibr/models.py`)

Pydantic model for the paper's own fields and reference list:

- `doi`, `title`, `keywords` -- core identifiers
- `paper_type`, `paper_type_confidence` -- paper type and confidence from the local classifier or LLM fallback
- `oecd_l1`, `oecd_l2`, `oecd_confidence` -- OECD Frascati domain taxonomy
- `authors` -- list of `PaperAuthor` (given, family, affiliation, email, orcid, role)
- `references` -- list of `PaperReference` (BibTeX fields + nested `match` dict with `ExternalMatch` per service)

### CanonicalSection (`bibr/paper_contents.py`)

IMRaD+ section classification enum:

| Value | Description |
|---|---|
| `title` | Paper title (root/level-0 heading) |
| `abstract` | Abstract / Summary |
| `intro` | Introduction / Background |
| `method` | Methods / Materials |
| `results` | Results / Findings |
| `discussion` | Discussion / Conclusion |
| `references` | References / Bibliography |
| `acknowledgment` | Acknowledgments |
| `funding` | Funding information |
| `keywords` | Keywords |
| `endnote` | Supplementary material, future work, outlook |
| `appendix` | Appendix / Supporting Information |
| `open_data` | Data availability / code availability |
| `author_contributions` | Author Contributions / CRediT statement |
| `coi` | Conflict of Interest / Competing Interests |
| `ethics` | Ethics statement / IRB approval / Informed consent |
| `footnote` | Footnotes |
| `table` | Table caption/label region |
| `figure` | Figure caption/label region |
| `unknown` | Unclassified (fallback) |

## External services

| Service | Purpose | Required |
|---|---|---|
| OCR runtime (`paddle-*` default; `glm-*` explicit/fallback) | PDF recognition after layout detection | `bibr chew` selects a local runtime; `bibr serve` proxies to an explicit external OCR service |
| LLM API (Google/OpenAI/Anthropic/Groq/Ollama or managed local server) | Metadata extraction and fallback tasks | Standard extraction path; `no_llm=True` disables these tasks |
| Crossref API | Reference enrichment (DOI lookup + search) | Optional |
| Redis | Response caching (`bibr serve` mode) | Optional |

## Deployment architecture

### `bibr` (single-machine)

Pipeline orchestrator (`bibr/local/pipeline.py`). Loads models sequentially to fit within limited GPU memory. Components:

| Module | Purpose |
|---|---|
| `bibr/local/cli/` | CLI entry point (`bibr chew` command) |
| `bibr/local/pipeline.py` | `LocalPipeline` orchestrator with memory management |
| `bibr/local/layout.py` | PP-DocLayoutV3 layout detector |
| `bibr/local/segmenter.py` | wtpsplit-lite sentence segmenter |
| `bibr/ocr/registry.py`, `bibr/pipeline/resources.py` | OCR backend selection and managed runtime resources |
| `bibr/pipeline/plans.py`, `bibr/pipeline/stages/` | Shared stage ordering and implementations |

Memory management modes control GPU VRAM usage: `aggressive` (load/unload per phase), `balanced` (keep layout + segmenter resident; OCR also stays resident across chunks unless a local LLM server needs the VRAM), `keep_all` (everything loaded).

Auto-selection uses `aggressive` with ≤8 GB system RAM or ≤8 GB CUDA VRAM;
otherwise it uses `balanced`. With a cloud LLM and non-aggressive memory
mode, `PIPELINE_STREAM_BACKHALF=true` allows completed files' parsing,
extraction, enrichment, and export to overlap subsequent OCR work. Managed
local LLMs retain the stage barrier for the OCR-to-LLM memory handoff.

### `bibr serve` (LitServe)

LitServe separates the HTTP API process from its spawned inference worker.
Public multipart bodies never cross that process boundary. The request flow is:

```
POST /papers/extract
  -> API admission + explicit multipart bounds
     (one file, seven unique options at most 64 bytes each)
  -> one multipart spool (at most 1 MiB in memory)
  -> owner-only disk file (50 MiB file limit; 51 MiB body envelope)
  -> leased UUID/size/SHA-256/options descriptor
  -> LitServe multiprocessing queue
  -> worker verifies one read, deletes file, and runs the pipeline
```

`/_bibr/inference` is the private LitServe descriptor route and returns `404`
when called over HTTP. It exists only so the API process can dispatch the
opaque descriptor through LitServe's in-process route adapter. A descriptor
contains no file bytes and no filesystem path.

| Component | Module | Purpose |
|---|---|---|
| `build_server()` | `bibr/serve/app.py` | Builds the `litserve.LitServer`, public multipart route, private descriptor dispatch, and owned upload lifecycle |
| `UploadStore` / `InferenceDispatchTracker` | `bibr/serve/ingress.py` | Persists bounded uploads, queues opaque descriptors, and cleans files on consumption, cancellation, failure, or shutdown |
| `BibrPipelineAPI` | `bibr/serve/deployments/pipeline.py` | Main processing orchestrator |
| `LayoutDetector` | `bibr/serve/deployments/layout.py` | PP-DocLayoutV3 layout detection (GPU/CPU) |
| `SentenceSegmenter` | `bibr/serve/deployments/segmenter.py` | wtpsplit-lite sentence segmentation (GPU/CPU) |

Exactly one inference worker is pinned in `build_server()`, with no setting to
change it: another worker means another model copy (~1.5 GB RSS), CUDA context,
and `GpuBatcher`, so batches shrink as workers rise. Async I/O and the GPU
batchers provide concurrency within that one worker, and the heavy CPU stages
already use every core from one process — see
[Production deployment](deployment.md#why-there-is-no-worker-count-setting).
Failed inference workers
fail-stop by default (`PIPELINE_RESTART_WORKERS=false`). LitServe 0.2.17 worker
replacement cannot reliably notify the API waiter owned by a dead worker, so
`true` remains an unsupported opt-in until the locked real-process death-path
gate proves completion notification.

Async jobs persist through the same upload store and submit the same descriptor
to LitServe. Their queue and result store remain in the HTTP API process, so
the server always pins exactly one API process, including with jobs disabled.
Upload leases keep queued and dispatched descriptors out of stale sweeping.
LitServe is pinned below 0.3 because this boundary relies on 0.2.x
manager/worker helpers; the spawned success and death-path compatibility tests
must be migrated before that version constraint is relaxed.

## Configuration

Settings are managed by `GlobalSettings` (`bibr/config.py`) using pydantic-settings.
Values are read from environment variables or a `.env` file. Each pipeline
owns a settings snapshot; per-run choices such as page range and reference
strategies travel through `RunConfig`, without changing process-global defaults.

Key settings:

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | LLM backend | `{{ default_llm_provider }}` |
| `LLM_MODEL` | LLM model name | `{{ default_llm_model }}` |
| `LLM_REASONING_EFFORT` | Default reasoning effort for OpenAI models | `minimal` |
| `LLM_REASONING_EFFORT_AUTHORS` | Reasoning effort override for author extraction | `low` |
| `LLM_REASONING_EFFORT_CITATIONS` | Reasoning effort override for citation resolution | `low` |
| `OCR_BACKEND` | OCR runtime selector/backend | `paddle` |
| `OCR_BASE_URL` | Base URL for an external HTTP OCR server | `http://localhost:8080` |
| `CROSSREF_ENRICH` | Enable reference enrichment | `true` |
| `EQUATION_EXTRACTION` | Enable equation extraction | `true` |
| `FIGURE_IMAGES` | Include base64-encoded figure images in output | `false` |
| `REF_SEG_STRATEGY` | Reference segmentation strategy (`geom`, `region`, `llm`, or `crf`) | `geom` |
| `REF_PARSE_STRATEGY` | Reference parsing strategy (`ner`, `llm`, `llm-chunked`, or `off`) | `ner` |
| `REF_TRAINING_DATA_DIR` | Save raw bib text + LLM extracts (parser data); LLM segmentation spans in `segmentation/` subdir (segmenter data) | disabled |
| `ENVIRONMENT` | Runtime mode (`development`/`production`) | `development` |

For the complete reference, run `bibr config example --full` or see the
[Settings reference](../reference/settings.md).
