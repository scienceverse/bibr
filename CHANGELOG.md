# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **The serve container image ships the `mcp` extra.** `Dockerfile.serve` now installs
  `bibr[mcp]`, so `MCP_ENABLED=true` on the Compose stack mounts the remote MCP endpoint
  without a custom build. The dependency is inert unless enabled.

### Fixed

- **Release tests collect on Windows again.** The process-group signal guard
  only installs where `os.killpg` exists; Unix runtime tests supply their own
  mock on other platforms.
- **MCP extraction shares REST admission limits.** JSON uploads are bounded
  before parsing, and `chew_paper` / `chew_url` acquire shared capacity before
  decoding or downloading. In-flight slots remain held until dispatch finishes,
  including when a caller cancels an extraction.
- **MCP page ranges follow the advertised 1-based contract.** Page 1 now
  selects the first physical page; zero and negative pages are rejected.
  REST page indices remain zero-based.
- **`bibr serve` no longer crashes at startup when the `mcp` extra is installed.** LitServe
  0.2.17 enables its own MCP connector whenever the official `mcp` package is importable
  but builds it from the third-party `fastmcp` package, so `server.run()` died with
  `NameError: name 'MCPServer' is not defined` on any install of `bibr[mcp]` (including the
  serve image above). bibr now switches LitServe's detection off — it mounts its own
  `/mcp` endpoint and never wanted LitServe's.
- **A plain Linux `bibr chew` no longer bootstraps vLLM behind your back.** The automatic
  `paddle` OCR chain only lists `paddle-vllm` on an NVIDIA GPU with at least 8 GB of VRAM;
  CPU-only and small-GPU Linux machines go straight to llama.cpp (`glm-llama`), as the
  tester guide always said. The managed vLLM launcher refuses to start without a suitable
  GPU (naming the alternatives), and when vLLM is not installed it now *warns* — with the
  `uv sync --extra vllm` remedy — before falling back to the isolated
  `uv tool run --from vllm==0.25.1` environment, which downloads several GB on first use.
  On Python 3.14, where `vllm==0.25.1` has no wheels, the bootstrap pins a managed 3.13
  interpreter instead of failing to resolve. `bibr chew` also checks before loading any
  model that at least one local OCR runtime can start for the PDFs it was given, and
  fails fast with the install hints otherwise.
- **`bibr setup` installs the runtime its Linux plan needs.** The "fully local" plan on
  Linux/CUDA selects the `vllm` extra; it used to select `local`, whose only member is
  Apple-Silicon-only, so nothing was installed and the first chew paid for the bootstrap
  above. The advanced picker no longer offers `local-cuda`, an extra that does not exist
  and made `uv sync` fail before `.env` was written.
- **`glm-mlx` is no longer offered anywhere.** The backend has been disabled since July
  (vllm-mlx produced corrupted OCR text and leaked memory), but the wizard, the `--ocr`
  choices, `bibr doctor` and the quickstart still presented it. `bibr doctor` now reports
  a config that still names it as a failed check pointing at `glm-rapid-mlx`.
- **`OCR_BASE_URL` may end in `/v1`.** bibr appends `/v1/models` and
  `/v1/chat/completions` itself, so the deployment guide's own example
  (`https://ocr.example.internal/v1`) was requested as `/v1/v1/...` and never became
  ready. A trailing `/v1` is now stripped with a warning in `bibr serve`, `bibr chew
  --ocr-url` and the readiness probe. `bibr serve` also honours
  `OCR_BACKEND=paddle-http` for its defaults (served alias `paddle-ocr-vl-1.6`, profile
  `paddle`) instead of silently assuming the GLM `glm-ocr` alias; the guide is corrected.
- **A stalled Redis can no longer wedge `bibr serve`.** The response cache, the Crossref
  response cache and the Redis rate limiter now carry connect, socket and health-check
  timeouts, and every cache touch on the request path (read, single-flight lease, release,
  write) is additionally bounded by the new `CACHE_OPERATION_TIMEOUT_SECONDS` (default
  5). A Redis that accepts connections but never answers now degrades to a cache miss
  instead of holding every request — and its admission slot — forever.

## [0.5.0] - 2026-09-01

### Added

- **`bibr mcp` — MCP server for agents** (new optional `mcp` extra, included in `all`).
  Exposes extraction as Model Context Protocol tools over stdio: `chew_paper` /
  `load_paper` register a paper and return a compact `bibr inspect`-style summary, then
  `get_metadata`, `get_sections`, `get_text`, `search_text`, `get_references`,
  `get_reference_citations`, `get_tables`, `get_figures`, and `save_paper` query the
  stored export in slices sized for an agent's context. One warm pipeline serves the
  whole session (models load once), extraction progress streams as MCP progress
  notifications, and pipeline options are fixed at server start via a subset of the
  `bibr chew` flags. Register with e.g. `claude mcp add bibr -- uv run bibr mcp`; see
  the new [MCP server guide](https://bibr.org/guides/mcp/).
- `Chewer.chew` / `achew` (and `chew_file` / `achew_file`) accept a `progress=` tracker
  (`bibr.pipeline.progress.ProgressTracker`, e.g. `RichProgress`) to observe stage
  transitions and per-region OCR progress from library code.
- **Remote MCP on `bibr serve`** (`MCP_ENABLED=true`, requires the `mcp` extra): mounts a
  streamable-HTTP Model Context Protocol endpoint at `/mcp` with the same chew-then-query
  tool surface as `bibr mcp`. Gated by the existing bearer auth; extraction rides the
  regular serve inference dispatch (resident worker models, admission control, size caps —
  no second pipeline). `chew_paper` takes base64 file content plus per-call
  `start_page`/`end_page`/`refs`/`consolidate` options; the filesystem tools
  (`load_paper`/`save_paper`) are not exposed remotely, and papers are held per MCP
  session, capped by `MCP_MAX_PAPERS_PER_SESSION` (default 16). See the
  [MCP server guide](https://bibr.org/guides/mcp/).
- **`chew_url` MCP tool** on both servers: extract a paper straight from a public
  `https://` URL. The download is SSRF-guarded by the new `bibr.utils.safe_fetch`
  (HTTPS/443 only, every DNS answer must be public unicast, the connection is pinned to
  the validated IP with TLS SNI/verification kept on the hostname to defeat DNS
  rebinding, redirects re-validated per hop, size-capped under a deadline). Capped at
  100MB on `bibr mcp`; on `bibr serve` it uses the upload size limit and rides the same
  inference dispatch, with `MCP_URL_ALLOWED_HOSTS` to pin hosts and
  `MCP_CHEW_URL_ENABLED=false` to remove the tool.

### Changed

- Removed ignored/no-op config names: `LLM_VLLM_MLX_CACHE_MB`, `LLM_BATCH_PROVIDER`;
  `OCR_API_HOST`, `OCR_API_PORT`, `OCR_CONFIG_PATH`, `OCR_ENABLE_LAYOUT`, `OCR_API_PATH`,
  `OCR_API_MODE`; and the reserved `ML_ENABLED`, `ML_SECTION`, `ML_REF_SEG`, `ML_REF_PARSE`,
  `ML_SECTION_ACCEPT_THRESHOLD`, `ML_SECTION_FLAG_THRESHOLD`, `ML_SECTION_REPO_ID`,
  `ML_REF_SEG_REPO_ID`, and `ML_REF_PARSE_REPO_ID`. These names are ignored if left in
  existing config and should be removed. External deployment/Compose and bundled-SGLang
  scope are unchanged.
- OCR disk cache format 7 removes the retired layout-toggle key; existing format-6 entries
  incur a one-time cache miss and rebuild.
- `OCR_MAX_CONCURRENT_REGIONS` now binds the single-machine OCR path, which previously
  ignored it and capped every run at `OCR_CONCURRENT_REGIONS_PER_FILE` (6). Against the
  managed `paddle-vllm` runtime — the Linux/CUDA default — that left its vLLM server
  (launched with `--max-num-seqs 12`) under-subscribed; it now runs at the server-wide cap
  (16 by default). Files still run one at a time, so page-image RAM is unchanged. Engines
  whose prefill serializes on the device (MLX, llama.cpp) keep the per-file cap, and the
  Apple Silicon auto-tune to 1 is unaffected.
- Crossref enrichment now prefetches every DOI-bearing reference in one
  `/works?filter=doi:...` query (up to 50 DOIs per request) before the per-reference
  fan-out, instead of spending one rate-limited request per DOI. The prefetch seeds the
  same response cache the per-reference path reads, so matching, consolidation and
  provenance are unchanged; only DOIs the bulk query returns are seeded, so a DOI Crossref
  does not know still takes its own lookup and still 404s rather than falling through to a
  bibliographic search. Disable with `CROSSREF_BULK_DOI_LOOKUP=false`.
- **Removed `PIPELINE_WORKERS_PER_DEVICE`.** `bibr serve` now pins exactly one inference
  worker in `build_server()`. There is no measured configuration where a second worker won:
  each worker gets its own `GpuBatcher` (so GPU batches shrink as workers rise) on top of
  duplicating the model weights and CUDA context, costs ~1.5 GB RSS (~570 MB of that in
  imports alone, before any model loads), and parallelizes only GIL-bound Python — the heavy
  CPU stages already use every core from one process, and the process-global pdfium lock it
  would have relieved is under 1% of a paper's wall clock (~10 ms/page render plus a
  comparable inspection pass). Left in an existing config the name is ignored, not rejected,
  but it should be deleted. Scale with `PIPELINE_MAX_INFLIGHT_REQUESTS` and the batch-timeout
  settings instead. `cap_inference_threads`, which existed only to divide cores among
  co-located workers, is removed with it; torch now uses its own default thread count, which
  on a hyperthreaded host is typically physical rather than logical cores.
- A managed local LLM server (`--llm local` on CUDA or Apple Silicon) auto-raises
  `LLM_RATE_LIMIT_RPM`, unless set explicitly. The 60 default guards a cloud provider's
  quota; against a server bibr owns it capped bulk runs near 8-12 papers/min regardless of
  hardware.

### Fixed

- `bibr doctor` now reports a missing system **libmagic** as its own named check, and
  `bibr.input.validate` imports the `python-magic` binding defensively instead of at
  module scope. libmagic is a system library a `pip install` cannot supply, so a fresh
  macOS/Linux setup died with a bare "failed to find libmagic" during `bibr setup`'s test
  extraction — and the import failure took down `import bibr` wholesale, so `doctor` could
  not run to diagnose it. The error now names the platform's install command
  (`brew install libmagic`, `apt install libmagic1`, `dnf install file-libs`). (#64)
- A managed local server whose port is held by an unrelated process now fails with a
  message naming the port, instead of spawning a subprocess that cannot bind it and dies
  with an unrelated-looking startup crash. The pre-spawn guard treated "listener with an
  unusable /v1/models" the same as "port free"; it now confirms the port is genuinely
  held with a TCP connect before reporting a conflict. (#82)

- LLM retries now acquire their own rate-limit slot. Only the first attempt of each logical
  call took one, so a retry storm spent budget it never acquired — precisely when the
  provider was already rate-limiting and, with Redis configured, when the shared limiter is
  meant to hold the whole fleet back.
- A missing `CROSSREF_API_EMAIL` now warns with the concrete rates: without it Crossref's
  anonymous pool caps the client at 60 RPM, so a configured `CROSSREF_RATE_LIMIT_RPM=200`
  was silently a third of that (~81s of an 80-reference paper's 120s enrichment budget).

## [0.4.0] - 2026-07-26

Consolidates roughly five weeks of work since 0.3.0: a new default OCR engine
(PaddleOCR-VL), mature local-LLM runtimes across CUDA / Apple Silicon / Windows,
native JATS/HTML/ePub input, a rebuilt reference pipeline, trained
section/paper-type classifiers on by default, a much richer extraction schema
(v10.7), resolver-based enrichment, a config-preset system, a full CLI/UX
overhaul, and a security-hardening pass.

### Added

**OCR**
- **PaddleOCR-VL is the new default OCR engine** (`OCR_BACKEND=paddle`), with PP-DocLayoutV3 layout detection. Backends: `paddle` (default), `paddle-vllm` (GPU/vLLM), `paddle-rapid-mlx` / `paddle-mlx-vlm` (Apple Silicon), and `paddle-http` (external). Includes OTSL table decoding, formula canonicalization, and incomplete-table recovery.
- GLM-OCR retained as an alternative family: `glm-mlx` / `glm-rapid-mlx` (Apple Silicon), `glm-llama` (Windows default; llama.cpp), `glm-http` (external).
- **Cloud vision-LLM OCR** backends — `gemini`, `openai`, `anthropic` (via Instructor).
- **Native PDF text bypass** (on by default) — regions backed by a good PDF text layer skip OCR, with a printable-ratio corruption gate that falls back to OCR.

**Local LLM runtime**
- **`--llm local`** auto-resolves a managed, self-hosted OpenAI-compatible server: managed **vLLM** on CUDA, **vllm-mlx** (continuous batching) on Apple Silicon, **`--llm rapid-mlx`**, and **`--llm llama-cpp`** for Windows / low-VRAM (6 GB+) GPUs. NuExtract3 is the default local extraction model, chosen from a curated hardware-detected model registry.

**Input formats**
- **Native JATS XML, HTML/`.htm`, and ePub ingestion** — parsed natively; skip OCR and core LLM extraction, like DOCX.

**References**
- **Reference segmentation rebuilt** around a local **geometry GBM (`geom`, now the default, ~free)** with a confidence-gated LLM-anchor cascade; CRF, region, and pure-LLM strategies remain selectable. Segmentation and parsing are decoupled via `REF_SEG_STRATEGY` / `REF_PARSE_STRATEGY` (CLI `--refs` / `--ref-seg`).
- **Local NER reference parser is the default** (`--refs ner`); `--refs llm` gives full-precision batched LLM parsing; **`--refs off`** skips reference extraction entirely.
- Deterministic **merged-reference splitter** (on by default), leading-reference salvage from truncated LLM batches, Vancouver year/container backfill, and under-extraction warnings (vs in-text citation count).

**Trained classifiers (on by default, LLM fallback only on low confidence)**
- **Context-aware section classifier (v3/v4)**, loaded from HF Hub, with a positional sanity pass.
- **Paper-type** and **OECD research-domain** classifiers.

**Extraction & schema (v10.7)**
- **Research-integrity mining** — data/code-availability & ethics statements, structured funding, and CRediT author-contribution roles.
- **Structured affiliations** and **paper self-identity** (journal, volume, issue, pages, ISSN, publisher, date, license, self-DOI match).
- Figure/table **captions**, an extraction **provenance** block, **per-label LLM token usage** (`llm_usage_by_label`), non-fatal **`processing_warnings`**, and an **output validation gate**.

**Enrichment**
- **Optional resolver-first enrichment** via **bibr-resolver** (`BIBR_RESOLVER_*`) — one `sources` query spanning OpenAlex + Crossref, short-circuiting on a clean resolver miss — layered on the default Crossref enrichment. Two-tier Crossref cache (in-process LRU → shared Redis). Optional **consolidation** of accepted matches into `bib` (`CROSSREF_CONSOLIDATE=off|fill|replace`, `--consolidate`). Title-less (Nature/Science-style) reference matching by fingerprint.

**Config presets**
- **`bibr preset`** subcommands + a **`--preset`** flag and `PresetManager` for named JSON config profiles; `~/.bibr/.env` fallback when the CWD has none.

**CLI / setup**
- **Unified terminal design system** across all commands; **`bibr config`** (show/path/set/example, always-redacted), **`bibr inspect`** for extraction results, **`chew --dry-run`** resolution preview, and **`--no-llm`** structural-only mode.
- **Setup wizard redesigned** — hardware-detected plan preview before installing, local-LLM onboarding, save-as-preset, and an end-to-end smoke extraction on a shipped synthetic sample.

**Serve**
- **Async job API** with per-request usage metering, **GPU micro-batching** for safe single-worker concurrency, **gzip** responses (~8× on paper JSON), bearer-token auth gating all non-probe routes, per-request `refs`/`ref_seg` overrides, and an opt-in OCR disk cache.

**Library API**
- **`bibr.chew()` / `achew()`** (single file, directory, or list) returning a `Result` with `.df` / `.records` views and per-file `ChewFailure`; a **`Chewer`** warm-pipeline session; a typed paper-export model; and isolated `Settings` on the library APIs.

### Changed
- **Default OCR engine switched from GLM-OCR to PaddleOCR-VL** (see Added).
- `OCR_SGLANG_GPUS` renamed to `OCR_LOCAL_GPUS` (old name still accepted as an alias). The unused `OCR_LOCAL_MEM_FRACTION` setting was removed — it only configured the in-process SGLang engine.
- Recommended cloud LLM updated to **Gemini 3.5 Flash-Lite**.
- Reference-parse batch size default raised **5 → 15**; layout-detection batch **4 → 8**.
- Apple Silicon throughput defaults unlocked (higher default MPS concurrency/batch).
- Section classification is now trained-model-first (LLM only on miss / low confidence), using document-context snippets.
- Core install stays torch-free; heavy ML deps remain behind the `ml` extra.
- OCR cache hardened (correct model-profile keys, safer concurrent writes).

### Fixed
- **Metadata regressions:** single-article title/byline is no longer blanked by front-matter multi-item abstention; native NuExtract author extraction no longer returns schema-valid empty author lists when byline/CRediT evidence exists; deterministic LLM invalid-output is no longer misclassified as a retryable upstream failure.
- **Compound figures:** panels are grouped as parts of their parent figure instead of exploding into independent top-level figures and sections.
- **Extraction quality:** OCR NUL/surrogate scrubbing before tokenization, mangled section-header repair, masthead/internal-heading title rejection, DOI line-wrap bridging with self-DOI selection over funder/reference/footnote candidates, full-name author scoring, footnote/xref positional anchoring, and filtering of parenthetical-numeric equation false positives (author-year veto + equation-tag guard).
- **References:** never drop the leading reference; drop bare in-text citations that leaked into `ref_text`; remove running-header bleed.
- **Windows:** OCR cache and section-classifier download handling, symlink-failure fallbacks, and `llama.cpp` PATH discovery; low-VRAM llama.cpp path hardened.
- **Security hardening** (audit 2026-07-23): redact secrets from `Settings` repr / `model_dump`, CLI & serve logs, and `bibr doctor`; exclude API keys from cache fingerprints; enforce real-byte zip caps and reject spoofed file types; reject path-traversal DOIs before resolver/Crossref lookup; cap HTML input and upload-filename length; gate `/ready` detail; guard wildcard-CORS credentials; rotate the metering log; gadget-restricted joblib load for the geom segmenter.

### Performance
- Serve concurrency reworked around a single async worker (event loop unblocked, pipeline reused, GPU work micro-batched); `workers_per_device` default 2.
- Inference offloaded off the event loop (NER/GBM parse, native-text pdfium work, `gc.collect`); CUDA TF32/cuDNN autotuner and MPS float16 autocast for layout; per-page pdfium locks plus next-file render prefetch enable parallel file processing.
- Crossref/resolver caching (in-process LRU + Redis tier-2, `select=` field trimming, higher enrich concurrency when the resolver is enabled).

### Removed
- **SGLang removed entirely** — both the managed SGLang _LLM_ backend and the in-process `glm-sglang` _OCR_ backend, along with the `sglang[all]` dependency and the now-empty `local-cuda` extra. The pinned 0.5.12 line carried three unpatched critical advisories (unauthenticated RCE, pickle deserialization on a `0.0.0.0` socket, path traversal) and transitively pulled `diffusers` (two high advisories). Local LLM serving is vLLM / vllm-mlx / rapid-mlx / llama.cpp; GPU OCR is `paddle-vllm`. **Migration:** run your own SGLang server and point `glm-http` at it (`OCR_BACKEND=glm-http`, `OCR_BASE_URL=...`) — the bundled Compose `bibr-ocr` service still does exactly this.
- **Falcon OCR backend.**
- Dropping `sglang[all]` shed ~82 locked packages, removed the last mutually exclusive extras (so **`--all-extras` resolves again**), and made the `pillow` override unnecessary (it existed only for `moviepy`, a transitive SGLang dep).
- Dead code and unused dependencies — `spacy`, `rpy2`, the `metacheck` extra, legacy CRF model files, and transitional flat-name config shims.

## [0.3.0] - 2026-06-15

First tagged release of the rebuilt pipeline. The intermediate `0.2.0` tag was never published, so its notes are folded in here.

### Added
- **One-call Python API** — `bibr.chew()` / `bibr.achew()` process a single file, a directory, or a list of paths in one call, returning a `Result` with `.df` / `.records` views and `.ok` / `ChewFailure` per-file error handling. `Chewer` is a warm-pipeline session context manager. `from bibr import LocalPipeline, Pipeline, Settings` remains the lower-level entry point (lazy-loaded; no heavy deps at import time).
- **Reference segmentation rebuilt around LLM anchor-emit** — references are segmented by an LLM anchor pass (CRF fallback) then parsed in batches, replacing the retired rule splitter. Strategies are decoupled and configurable via `REF_SEG_STRATEGY` / `REF_PARSE_STRATEGY`; NER parsing is opt-in (`--refs ner`). The pipeline warns in `processing_warnings` on CRF seg-fallback and on suspected reference under-extraction (vs in-text citation count).
- **In-text citation (xref) linking + evaluation** — improved narrative and parenthetical citation parsing, plus automated checks for citation-linking behavior.
- **URL extraction (`url[]`)** — printed DOI links and web URLs are extracted, including reconstruction of line-wrapped URLs (CRLF and mid-word wraps) and support for balanced-paren DOIs.
- **Local LLM serving** — managed SGLang LLM server with `--llm local` auto-resolution; `LLM_MAX_CONCURRENCY` gate for single-device servers; `LLM_VLLM_MLX_EXTRA_ARGS` passthrough; opt-in merged core-metadata call (`LLM_MERGED_CORE_METADATA`); schema-envelope unwrapping for small-model structured output.
- **Crossref consolidation** — optionally merge accepted Crossref matches into `bib` at export via `CROSSREF_CONSOLIDATE=off|fill|replace`, the `--consolidate` CLI flag, the `consolidate=` chew option, and a serve form field.
- **Scoped hierarchy (v5-lite)** for correct section nesting in multi-study papers.
- **Lead-reference recovery** from the PDF text layer for references the layout model drops.
- **Per-paper LLM token-usage export** (`llm_usage`).
- **MiniLM section classifier (v2)** on by default, with LLM fallback only on miss or low confidence.
- **`bibr_release`** stamped on every serve response `info`.
- **Saved-export evaluation** — scoring helpers compare extracted fields with independently prepared reference JSON.
- **CLI setup polish** — LLM credential preflight, ref-strategy knob in the setup wizard and doctor, `--refs` surfaced in help and `bibr demo`.
- **JSON v10 schema** — top-level shape change (`figure` replaces `fig`, drops `study`, adds `bib_match`). v10.1 moves `ocr_config` and `processing_warnings` to top-level so `info` is scalar-only (R consumers can `as.data.frame(info)`). Adds `ocr_config` block, `BibAuthorExport` author records in `bib_match`, section `level` field, and backfill of empty bib fields from high-confidence external matches.
- **`include_regions` toggle** (default off) for the large `_regions` layout debug payload — CLI `--regions`, the `include_regions` form field, `Paper.export_to_json(include_regions=...)`, `RunConfig.include_regions`. Reduces output size when diagnostics are not requested.

### Changed
- **Core install is now torch-free** — heavy ML dependencies moved to an optional `ml` extra, ML imports degrade gracefully, and OCR exports are lazy.
- **LLM client migrated from LangChain to Instructor**, with multi-provider support (Google, OpenAI, Anthropic, Groq, Ollama) through a single Instructor factory.
- **CLI flag renames**: `--ocr-backend` → `--ocr`, `--llm-backend` → `--llm`. All CLI unified under the `bibr` namespace.
- **Settings restructured** into sub-models: read via `Settings.ocr.backend`, `Settings.llm.provider`, etc. Env vars stay flat (`OCR_BACKEND`, `LLM_PROVIDER`).
- Reference parse batch size default raised 5 → 15.

### Fixed
- Extraction and structure: OCR wide-letter-spacing collapse before segmentation, repeated running-header demotion, mid-word DOI line-wrap bridging, DOI rescue from publisher `/doi/` URLs and clean `doi:` tokens, page-1 `TC` badge-glyph stripping, fabricated-abstract suppression on abstract-less commentaries, software/dataset title and book-edition handling, bracket-citation retention.
- xref parsing: nested group-cites, narrative colon-page and curly-apostrophe possessive cites, year-less back-references, et-al disambiguation, harvested-year constraints, parenthetical cap recovery.
- Robustness and security: pdfium lock in validation, DOCX zip-bomb ceilings, pdfium handle cleanup, CUDA gating by compute capability, per-file pipeline errors on resource-init failure, `401` responses carrying `WWW-Authenticate` + CORS, CVE-driven torch bump.
- CLI: batch output directories with dotted names are no longer misread as file suffixes.

### Performance
- Crossref works/search LRU cache; native-text pdfium work moved off the event loop; next-file page render prefetched during layout detection; serve `workers_per_device` default raised to 2.

### Removed
- **SSE streaming endpoint** `POST /papers/extract/stream`. The synchronous `POST /papers/extract` is the only paper-extraction endpoint.
- LibreOffice-based DOCX conversion path. `.docx` is parsed natively via `python-docx`; `.doc` (legacy Word) is no longer supported — convert to `.docx` first. Drops `DOCX_BACKEND`, `LIBREOFFICE_TIMEOUT_SECONDS`, the `WITH_OFFICE` build arg, and `bibr.clients.libreoffice.LibreOfficeClient`.
- `bibr-serve`, `bibr-setup`, `bibr-demo` console scripts (replaced by `bibr serve` / `setup` / `demo` subcommands).
- LangChain dependency; legacy rule reference segmenter (moved to `evaluation/`).
- Empty `bibr/_vendor/` package and orphan top-level `ocr/Dockerfile`.
- `debug_samples/`, `metacheck_integration/`, `prereg.json`, and tracked `notebooks/incest.json` artifact data.

## [0.1.3] - 2026-03

### Added
- **`bibr chew` CLI**: process PDF/DOCX files directly without an external OCR server or serve deployment. Includes in-process OCR via SGLang (NVIDIA CUDA + Apple Silicon MPS), sequential GPU model loading with configurable memory management (`aggressive`, `balanced`, `keep_all`), and support for external OCR servers via `--ocr-url`
- `[local]` optional extra: `uv sync --extra=local` installs SGLang for in-process OCR
- `BibType` enum with standard BibTeX entry types (article, book, inproceedings, incollection, etc.)
- `booktitle` field on `PaperReference` for book chapters and proceedings papers
- LLM extraction of 6 new reference fields: `last_page`, `issue`, `publisher`, `editor`, `booktitle`, `bibtype`
- Non-destructive Crossref backfill: enrichment now populates empty fields (DOI, volume, issue, pages, publisher, ISBN, ISSN, booktitle, bibtype)
- `BibTypeEnum` in Pydantic schemas for validated LLM bibtype output
- API key authentication middleware (`X-API-Key` header, backward compatible)
- In-memory per-IP rate limiting middleware with configurable window and request count
- Trivy vulnerability scanning for Docker images in CI (table + SARIF upload)
- SSE streaming endpoint (`POST /papers/extract/stream`) for real-time pipeline progress
- Study design classification: LLM-based (RCT, Retrospective Cohort, Case Report, Meta-Analysis, In Vitro)

### Changed
- **JSON is now the primary (and only) export format** -- JSON v8.0 schema with top-level keys: `paper_id`, `info`, `author`, `text`, `section`, `url`, `bib`, `xref`, `fig`, `table`, `eq`
- `bibtype` values normalized from custom capitalized strings (e.g. "Article", "BookChapter") to standard lowercase BibTeX types (e.g. "article", "incollection")
- Crossref `container-title` now routed to `booktitle` for book chapters and proceedings articles
- Demo reference tables now display "Book Title" column
- CORS origins automatically restricted from `["*"]` to `[]` in production mode (`ENVIRONMENT=production`)
- Health endpoints (`/health`, `/ready`) exempt from authentication and rate limiting
- Section classification switched from zero-shot NLI (bart-large-mnli) to lookup table + LLM fallback

### Removed
- Arrow IPC export format (v6.2 and earlier) -- replaced entirely by JSON v8.0

## [0.1.2] - 2026-02

### Added
- Evaluation harness with 9 per-field metrics (exact match, ROUGE-L, Jaccard, etc.)
- Multi-class paper type classifier (empirical, review, meta-analysis, case-study, commentary, unknown)
- OECD domain classifier using cascading zero-shot NLI (L1 + L2 taxonomy)
- Configurable reference deduplication thresholds (`DEDUP_TITLE_THRESHOLD`, `DEDUP_MIN_TITLE_LENGTH`)
- Ground truth loading from Parquet for evaluation

### Changed
- Paper type classification upgraded from binary stub to priority-ordered rule-based system
- Reference dedup thresholds now configurable via Settings (previously hardcoded)

## [0.1.1] - 2026-02

### Fixed
- OCR region label routing now uses `native_label` for correct treatment dispatch
- Markdown prefix stripping in section headers (prevents `#` leaking into classified text)
- Footnote crash on `content=None` regions
- Missing `layout_hints` attribute on OCR regions

### Added
- OCR artifact correction (ligature expansion, soft hyphen removal) at page processing level
- Reference section text preserved intact for downstream CRF/LLM segmenter
- Non-destructive IMRaD enforcement (repeatable section types preserved)
- Title fallback from first non-canonical heading when layout model and LLM both fail
- Graceful degradation on LLM failures (partial `PaperMetadata` returned instead of crash)

### Removed
- Dead AST-era code (Tier 1 citation linking, unused imports, stale type definitions)

## [0.1.0] - 2026-01

### Added
- Initial release
- PDF and DOCX input support (DOCX via LibreOffice conversion)
- OCR via glmocr SDK with Ollama, vLLM, and SGLang backends
- LLM-based metadata extraction (title, authors, DOI, keywords, references)
- Section classification using zero-shot NLI (facebook/bart-large-mnli)
- Sentence segmentation via wtpsplit (ONNX)
- Inline citation NER (DistilBERT) with citation linking
- Reference extraction (LLM and NER strategies)
- Optional Crossref reference enrichment
- Arrow IPC export (v5.5 schema) with manifest
- FastAPI REST API with Redis caching
- Gradio demo application
- CLI (`bibr-serve`, `bibr-setup`, `bibr-demo`)
- Docker deployment with GPU-accelerated OCR sidecar
