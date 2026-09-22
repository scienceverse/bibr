# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- The reference under-extraction warning in `processing_warnings` now also
  covers numeric citation styles. It previously counted only author-year
  citations, so a numbered paper whose reference region was lost to OCR was
  never flagged. It now also counts the distinct reference numbers cited by
  bracket and superscript markers, up to the highest number where at least
  half of 1..n are cited, and warns when fewer than half that many references
  were parsed (at least 15 cited).
- The OCR disk cache (`CACHE_OCR`, on by default in the local demo) is now keyed
  on the bibr version too, so an upgraded bibr no longer reuses rendering,
  layout, native-text and OCR bundles made by the previous release. The key
  still cannot see source changes between releases; the `CACHE_OCR` description
  now says to use a fresh `CACHE_OCR_DIR` per revision when comparing such
  changes, and to leave the cache off when timing runs.
- Reference-segmentation training capture (`REF_TRAINING_DATA_DIR`) no longer
  mixes geometry-segmenter predictions in with LLM segmentation labels; only
  LLM output is captured. Under the default `geom` strategy, that means only
  blocks the cascade sends to the LLM; set `REF_SEG_STRATEGY=llm` to label
  every block. Records from earlier versions lack `provenance` and may contain
  geometry output; discard them or capture into a fresh directory.
- Documentation and the `ML_PAPER_CLASSIFIER_MODEL_ID` setting description no
  longer call the default paper classifier SPECTER2-based; its model card
  documents an `all-MiniLM-L6-v2` encoder. The Classifiers guide also notes that
  the model has no `corrigendum` paper-type class and predicts 32 of the 36 OECD
  subdomains.
- Sentence DOI candidates in `extraction.identity.receipt` now record the layout
  region they were read from; `region_index` was previously always `null`. With
  `page`, it matches the `page` and `index` of an `extraction.regions` row: the
  region's position on that page after OCR post-processing renumbers merged
  regions. It stays `null` when no layout region is recorded for the sentence,
  or when the sentence is printed on a later page than the region that began its
  paragraph. The v11 export schema changes only by describing these fields.

### Added

- Captured reference training records carry a `provenance` object with the
  label source, LLM provider and model, prompt name and hash, and bibr version.

## [0.5.1] - 2026-09-12

### Fixed

- Cancelled OCR and local LLM startup reclaim servers that finish starting after
  cancellation. Shutdown waits for startup and completes resource cleanup even
  when interrupted repeatedly.
- Cancelled local sentence segmentation retains its inference lock until the
  worker finishes, preventing concurrent inference or premature model unloading.
- OCR cache hits preserve page-failure counts and warnings, and recheck the
  current minimum success ratio. Older entries without this evidence are rebuilt.
- Served requests can enable reference parsing and Crossref enrichment when the
  deployment defaults reference parsing to off.
- Layout initialization failures affect only PDFs in mixed batches; native
  documents continue through the pipeline, including streaming runs.
- Closing a pipeline releases resident layout and segmentation models and its
  cached front-matter classifier reference.

## [0.5.0] - 2026-09-11

### Fixed

- Release validation installs macOS's `libmagic` prerequisite and exercises
  Windows with platform-independent fixtures. Windows accepts sealed segmenter
  bundles, tolerates unavailable Unix memory metrics, and preserves upload
  identity and binary bytes. CLI output remains usable with legacy encodings,
  and disabling circuit-breaker deduplication counts failures even within one
  clock tick.
- Use patched vLLM 0.27.0 for the optional CUDA runtime and isolated LLM/OCR
  bootstraps, addressing GHSA-7m6h-x95x-82q5.
- Restore publication-date precision from an unambiguous printed publication
  date, and repair an empty author surname when the printed name and email
  establish a unique partition. Preserve ambiguous and already complete values.
- Preserve selected front-matter author evidence and distinguish explicit absent
  abstracts from inferred opening prose. Author-information tables supplement
  eligible single-record pages; generic literature-summary tables do not.
- Prefer a printed English abstract when parallel versions are available; otherwise
  retain the first complete printed version, without translating or concatenating.
- Stop requesting downstream author roles from metadata LLMs. Custom
  OpenAI-compatible endpoints can recover explicit decoder aborts through one
  validated JSON route, with token caps and chat-template options preserved.
- Structured-response caches distinguish generation schemas and chat-template
  options, and preserve explicit-null versus blank abstract intent.
- Recover a printed article DOI from complete retained OCR of a publisher box above
  its uniquely selected title. Explicit DOI, ISSN and publication labels establish
  identity; cited, ambiguous and truncated evidence remains excluded.

- LLM responses that echo a JSON Schema, including extracted values incorrectly
  nested under `properties`, now fail validation instead of being accepted as
  empty metadata with the schema name as the paper title.

- Numeric citations now follow reliable printed reference labels after dropped or spurious
  bibliography entries shift internal IDs. Citation diagnostics use the same corrected
  targets; duplicate and missing labels in that mapping cannot select a different entry.
- Reference-type inference reads the complete printed reference, recognizing thesis,
  preprint, conference and report labels outside the title when the parser omitted a type.
- GROBID benchmark runs accept both plain-text and JSON version responses without putting
  a JSON object into snapshot filenames or version columns.
- Dependabot CI keeps the coverage threshold and stores its report without attempting
  a Codecov upload that requires an unavailable Actions secret.

- **Release tests collect on Windows again.** The process-group signal guard
  only installs where `os.killpg` exists; Unix runtime tests supply their own
  mock on other platforms.
- **Cancelled REST and MCP extractions retain their admission slots.** The
  in-flight slot now belongs to the dispatch task until it finishes, so cancelling
  a caller cannot admit more work while its extraction is still running. Oversized
  MCP uploads are also rejected before allocating a decoded base64 copy.

- Async-job shutdown stops workers even if a dependency consumes cancellation while
  completing a request. Queued uploads are discarded instead of starting more work or
  waiting indefinitely for another job.
- **Protected docs smoke tests identify their HTTP client.** Both probes send
  `User-Agent: bibr-ci-smoke/1.0`, avoiding Cloudflare's error 1010 for Python's
  default agent. A browser-signature block is reported separately from an Access
  service-token rejection; anonymous protection and exact-revision checks remain enforced.
- **Docs deployments retain their revision marker.** The CI artifact now includes
  `.well-known/bibr-build`, so protected preview and production checks can verify
  the deployed commit after downloading the built site. Both jobs install Node/npm
  explicitly so Wrangler also runs on a freshly provisioned self-hosted runner.
- Redis-backed job workers preserve shutdown cancellation when a Redis reply arrives
  in the same event-loop turn. This prevents an intermittent Python 3.11 server shutdown
  hang while retaining the configured Redis operation timeout.
- Removed unsupported accuracy tables from the evaluation guide.
- The private-site CI smoke now distinguishes a rejected Cloudflare Access service token
  from a missing build marker, so deployment failures identify the required fix.

- **Exported URLs no longer carry PDF line-wrap artifacts.** A URL broken across a line in the
  source picked up the wrap whitespace when the text was re-joined, and a sentence-final period
  was absorbed into the href. `url[].href` and `bib[].url` are now collapsed and stripped of
  trailing dots at export, idempotently, so a downstream consumer can delete its own patch.
- **A JATS bibliography in `<body>` is no longer dropped.** EuropePMC's `fullTextXML`
  emits the reference list as a body `<sec sec-type="ref-list">` rather than inside
  `<back>`; the parser only looked in `<back>`, so every reference in such a document
  disappeared with no warning and the export shipped an empty `bib` — which is the whole
  contract for a reference-checking consumer. A body-located `<ref-list>` is now ingested
  into the section its producer already wrapped it in. A `<back>` ref-list still wins, so
  no document that parses correctly today changes.
- **JATS consortium authors survive.** A `<contrib>` carrying `<collab>` (a
  working-group or consortium byline) has no `<name>`, so it was emitted as an author row
  with an empty given *and* family name — a `VAL_AUTHOR_BLANK` validation error in place
  of the group's name. The collaboration name is now kept the way Crossref models a group
  author, and a `<contrib>` with no name of any kind is skipped instead of emitting a
  blank row.
- **JATS markup that means a line break no longer fuses words.** Flattening concatenated
  descendant text with nothing between the pieces, so `Cognitive load<break/>and recall`
  became `Cognitive loadand recall`, a structured `<aff>` became
  `Department of PsychologyUtrecht University`, and a two-paragraph abstract ran its
  sentences together. Block-level and structured-field elements now contribute a
  separator; inline markup still does not, so `H<sub>2</sub>O` stays `H2O`. XML comments
  are no longer flattened into the text either.
- **DOCX line breaks and tabs no longer fuse words together.** `<w:br/>`, `<w:tab/>` and
  `<w:cr/>` carry no text of their own and were dropped outright, so a title page laid out
  with Shift+Enter came out as `Cognitive load and recallJane SmithDepartment of
  Psychologyjane.smith@example.edu` — one unsplittable token where the title, author and
  affiliation should be. The same applied to footnote and endnote text, which is where a
  humanities bibliography lives. Those three elements now contribute a separator; adjacent
  `<w:t>` runs still concatenate untouched, because Word splits runs mid-word for
  formatting. Every DOCX fixture in the suite was built from python-docx plain strings,
  which never emit either element, so nothing caught this.
- **An unreadable DOCX is classified instead of crashing validation.** Validation only
  caught `BadZipFile`, but reading a *member* fails differently: `zipfile` raises
  `RuntimeError` for a password-protected entry and a truncated or damaged deflate stream
  surfaces as `zlib.error` from the real-size check. Both escaped
  `_check_docx_corruption` and took down the whole validation call rather than marking the
  file corrupt. A zip whose members are encrypted — what third-party tools produce, as
  opposed to the OLE container Word writes — is now reported as `encrypted_file` rather
  than as generic corruption.
- **The LLM rate limiter no longer freezes `bibr serve` while it probes Redis.** Deciding
  between the shared and the local limiter ran a *synchronous* `redis.Redis.ping()` from
  inside a coroutine. One LitServe worker with `enable_async=True` serves every concurrent
  request on a single event loop, so a Redis that accepts the connection but never answers
  stalled every in-flight paper, not just the caller — 5.1 s of total freeze, measured
  against a wedged-but-reachable Redis. The probe is async now, guarded by a
  loop-bound init lock so concurrent first callers build exactly one limiter, and the
  command round-trip is bounded (`socket_connect_timeout` only ever covered the connect).
  `CrossrefClient` was fixed this way already; `LLMClient` was missed.
- **A reference the LLM returns empty is re-parsed instead of deleted.** A batch item that
  came back with neither a title nor authors was counted as *covered*, so the NER recovery
  never ran for that slot — and then the completeness filter dropped it. One printed
  reference disappeared and every later `bib_id` shifted up by one, so an inline `[8]`
  resolved to what the paper printed as `[7]`, all the way down the list. Nothing warned:
  14 references returned for 15 entries clears the under-yield thresholds. Such a slot now
  counts as missing, goes through the same NER recovery as an entry the LLM skipped
  outright, and is reported if it cannot be recovered.
- **A repeated reference index no longer leaves segment-anchored backfills on.** When the
  LLM's reported indices are rejected the refs are re-numbered positionally, and the
  backfills that copy a printed DOI or issue number off the anchored segment are supposed
  to switch off whenever that mapping cannot be trusted. The check for that looked only at
  the *count*, so three rows labelled 1, 2, 2 for three entries passed it — while entry 3
  was missing and everything after the repeat sat one row off. The result was a
  neighbouring reference's DOI stamped onto the wrong row, which then enriched cleanly
  against Crossref and scored as a confident match. A repeated index now marks the batch
  untrusted too. A batch merely numbered from 1 instead of from `start_index` still stays
  trusted — positional re-indexing fixes that exactly.
- **The OCR disk cache now keys on the model pins.** A complete entry lets the pipeline
  skip layout detection and OCR inference outright, but the key recorded none of the
  settings that select those weights — `LAYOUT_MODEL_REVISION`, `OCR_PADDLE_REVISION` and
  `OCR_PADDLE_MODEL`. Re-pinning a model and re-running over cached papers silently
  replayed the *old* model's regions, so an A/B evaluation of the two pins reported no
  difference because it never ran the new one. `identity.model` did not cover this: for
  every served backend it is the alias (`paddle-ocr-vl-1.6`) that vLLM is launched with
  under `--served-model-name`, while `--revision` takes the pin — the alias is unchanged
  by a re-pin. The cache format version is bumped, so entries written without the pins are
  invalidated rather than trusted.
- **JATS keeps its Greek letters and accents.** The parser reads uploads with entity
  expansion disabled (the XXE and billion-laughs defense), which leaves every *named*
  character entity — `&alpha;`, `&uuml;`, `&deg;`, `&mdash;` — as an unresolved node whose
  text is the literal source string. Titles, author surnames and reference strings shipped
  markup like `M&uuml;ller` and `Effects of &alpha;-synuclein`. Because XML's five
  predefined entities and all numeric references resolve regardless, the output looked
  plausible rather than obviously broken. Named entities are now resolved after the parse
  against the HTML5 character table, which covers the ISO sets JATS DTDs pull in. Entities
  a document declares in its own internal DTD subset are still never expanded — the same
  applies to the ePub package document.
- **Non-ASCII ePub text is no longer mojibake.** The spine is re-emitted as one synthesized
  HTML document for the HTML parser, and that document declared no charset — so html5lib
  fell back to windows-1252 and decoded the UTF-8 bytes wrongly. Every non-ASCII character
  in an ePub's title, authors, publisher and body text was corrupted (`München` →
  `MÃ¼nchen`), affecting every non-English ePub.
- **Numbered bibliographies survive the in-text-citation filter.** Vancouver and IEEE entries
  terminate at the year exactly as a bare in-text cite does, so `"12. Rothman KJ. Modern
  epidemiology. Boston: Little, Brown; 1986."` was dropped as a citation — and because
  survivors are renumbered, one dropped entry shifted every later `bib_id` and repointed
  every numbered citation past the gap. Silent, with one `logger.info` line.
- **Multi-study `Method`/`Results` headings are no longer demoted as running headers.**
  Repetition alone was the test; page furniture's margin-band geometry is now required too,
  so a paper with per-study sections keeps them instead of exporting neither.
- **DOCX, JATS, HTML and ePub inputs no longer crash in the positional abstract fallback.**
  `min()` over page numbers that are all `None` raised `TypeError` for every native-format
  paper that reached it.
- **A crafted ePub can no longer exhaust server memory.** Every zip limit was per member, so
  a spine naming one member N times multiplied all of them: a 1 MB upload reached multi-GB
  RSS and OOM-killed the serve worker. Spine documents, total expanded bytes and repeats are
  bounded now, and percent-encoded hrefs resolve.
- **Statistics keep their sample size.** `(N = 1,204)` truncated to `N = 1` on the thousands
  separator, and a chi-square's own `df` parenthesis emitted a fabricated `N` that then
  vetoed the real match.
- **Figure and table numbers survive float merging.** Mergers renumbered survivors from 1, so
  a body mention of `Figure N` resolved to the wrong figure; renumbering now honours the
  printed label where a caption carries one.
- **Rotated pages map their text correctly.** `page.render()` applies `/Rotate` and the text
  layer does not, so on a rotated page every layout box sampled the wrong region of the PDF.
  Crop-relative coordinates are also emitted in the frame their page dimensions describe.
- **Plus 25 further defects** — a caption-dedup `KeyError` that surfaced as `parse_failed`
  and dropped the paper, an equation-extraction timeout that discarded the regex results it
  had already computed, OTSL row/column spans destroyed by a trailing newline, math exponents
  linked as citations and deleted from the sentence, an OCR backend that could never start,
  an OCR engine orphaned when an earlier stage failed, and a repeat scan whose cost grew
  superlinearly with region length (~11 s on a 50,000-character region, now under 15 ms).
- **`bibr serve` no longer crashes at startup when the `mcp` extra is installed.** LitServe
  0.2.17 enables its own MCP connector whenever the official `mcp` package is importable
  but builds it from the third-party `fastmcp` package, so `server.run()` died with
  `NameError: name 'MCPServer' is not defined` on any install of `bibr[mcp]` (including the
  serve image above). bibr now switches LitServe's detection off — it mounts its own
  `/mcp` endpoint and never wanted LitServe's.
- **The scorer no longer charges an elided page range against its expansion.** Gold keeps
  the printed ending ("486–92"); bibr and GROBID expand it to "492", and the exact
  string compare counted every such pair as a pages miss on both sides. `ref_pages_acc`
  now expands a compact ending against the first page on both sides before comparing.

- **Half-emitted page ranges are completed from the printed reference.** The CRF parser
  drops the start of a range and the LLM parser the end of a compact one ("339-42"); the
  shared finalize step now fills the missing end anchored on the value the parser did
  emit, only when the segment prints exactly one such range, and splits a range lumped
  into one field. Nothing populated is overwritten. This is the LLM-path repair the July
  analysis projected to lift `ref_pages_acc` from 0.389 to 0.745; measured by unit tests
  so far.
- **A plain Linux `bibr chew` no longer bootstraps vLLM behind your back.** The automatic
  `paddle` OCR chain only lists `paddle-vllm` on an NVIDIA GPU with at least 8 GB of VRAM;
  CPU-only and small-GPU Linux machines go straight to llama.cpp (`glm-llama`), as the
  tester guide always said. The managed vLLM launcher refuses to start without a suitable
  GPU (naming the alternatives), and when vLLM is not installed it now *warns* — with the
  `uv sync --extra vllm` remedy — before falling back to the isolated
  `uv tool run --from vllm==0.26.0` environment, which downloads several GB on first use.
  On Python 3.14, where `vllm==0.26.0` has no wheels, the bootstrap pins a managed 3.13
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
- **A bearer token containing non-ASCII bytes is rejected with 401, not 500.** The
  auth middleware and `/ready` compared the header as text, and `hmac.compare_digest`
  raises on non-ASCII strings; the comparison now runs on UTF-8 bytes.
- **502 bodies no longer name the internal OCR endpoint or the models it serves.** The
  readiness errors raised by `bibr serve`'s OCR backend (unreachable, not ready,
  cooling down) included `OCR_BASE_URL` and the server's model list; those details now
  go to the operator log only, and the client sees the expected served alias at most.
- **Idle MCP sessions now expire.** The serve MCP endpoint closes a client session after
  `MCP_SESSION_IDLE_TIMEOUT_SECONDS` (default 1800) of inactivity and drops its papers,
  as the MCP guide already promised. A client that reconnected without `DELETE` used to
  pin its session — and up to sixteen full exports — for the process lifetime.
- **Docker Compose publishes the API on loopback and passes the Redis password safely.**
  `bibr-serve` was published as `0.0.0.0:8000`, which Docker routes past host firewalls,
  with plaintext bearer tokens on the wire; it is now `127.0.0.1:8000` unless
  `BIBR_PUBLISH_HOST` says otherwise. `REDIS_PASSWORD` is no longer interpolated raw into
  `REDIS_URL` — a password with URL metacharacters silently disabled the cache — but
  handed to bibr, which URL-encodes it into the connection URL itself.
- **`LLM_LOCAL_MODEL` no longer defaults to the MLX weights on every platform.** The
  config default was `numind/NuExtract3-mlx-8bits`, and the CUDA vLLM and llama.cpp
  launchers read it too, so a hand-written `.env` with `LLM_BACKEND=vllm` or `llama-cpp`
  downloaded 4.8 GB of MLX weights and failed to load. Unset now resolves per backend
  (bf16 for vLLM, GGUF Q4_K_M for llama.cpp, 8-bit MLX on Apple Silicon); `bibr setup`
  keeps writing an explicit value.
- **`bibr setup` and `--llm local` no longer pick vLLM for a GPU that cannot hold the
  model.** The rule was "vLLM above 8 GB", but NuExtract 3's only vLLM variant (bf16)
  needs 11 GB, and the fit filter was dropped silently when nothing fit — a 9-10 GB
  card got a plan that OOMed after OCR. Both now choose vLLM only when a vLLM variant of
  the recommended model fits and llama.cpp otherwise; the wizard says which half (OCR,
  LLM, or both) needs `llama-server`.
- **The managed local LLM bootstrap matches the OCR one.** When vLLM is not installed the
  LLM launcher now warns (naming `uv sync --extra vllm`) before its `uv tool run`
  bootstrap and pins a managed Python 3.13 on 3.14, where `vllm==0.25.1` has no wheels
  and the `vllm` extra installs nothing; it used to fail there after OCR with "LLM
  server start failed" while `bibr doctor` reported vLLM as available. `bibr doctor`
  now says the runner is uv-managed, that the first run downloads several GB, and
  what 3.14 implies. Python 3.14 is listed in the package classifiers, matching CI.
- **`bibr.chew()` and `bibr.Chewer()` check the LLM before loading any model.** The
  library ran layout and OCR before discovering a missing API key or an unlaunchable
  local backend; the CLI already checked first. Both entry points now run the same
  preflight (skipped with `no_llm=True`), raising the provider's `ValueError` for
  credentials and `ConfigurationError` for a managed local backend.
- **A trained classifier that does not answer is now visible in the export.** When the
  section or paper classifier is configured but cannot load (core install without
  torch, failed download, a degraded serve resource) or errors during inference, the LLM
  classifies instead; the JSON was indistinguishable from a healthy run. The section path
  now records `section_classifier_degraded` in `processing_warnings` for every such
  case (previously only inference errors), and the paper path records
  `Metadata extraction WARNING: paper classifier degraded (<reason>)` — the exception
  type only, never document text.


- **Job results are bounded by size, not only by count.** `bibr serve` kept every completed
  result as a live dict and evicted only beyond `JOBS_MAX_RETAINED` (128), so a run of large
  exports could hold hundreds of megabytes for an hour. The result is now rendered once at
  completion (the bytes `/result` serves) and the store evicts oldest-first until both the
  count and the new `JOBS_MAX_RETAINED_BYTES` budget (default 256 MiB; `0` disables) fit;
  the newest result is always kept, so an export larger than the budget can still be
  fetched once.
- **The MCP chew tools are under upload admission, and a 50 MiB file fits.** The admission
  middleware only knew `/papers/extract` and `/papers/jobs`, so any number of `/mcp` calls
  could hold their bodies and decoded bytes in API memory and dispatch straight to the
  worker; and because `chew_paper` carries its file base64-encoded, LitServe's 51 MiB body
  cap refused a 40 MB PDF before the advertised 50 MB check. A large `/mcp` body now holds
  a `PIPELINE_MAX_ACTIVE_UPLOADS` slot while it is received, both chew tools take a spool
  slot through the persist and an inflight slot for the extraction itself — matching
  `POST /papers/extract`, so a running pipeline no longer refuses uploads the server has
  capacity to accept (a `server busy` tool error when none is free), the outer body cap grows to
  fit a full-size file in base64 when MCP is enabled, and an oversize body gets a `413` that
  explains the arithmetic before a byte is read.
- **`bibr serve` logs are configured — in both processes.** The CLI returned before its own
  logging setup, so `bibr.*` INFO records were dropped, warnings fell through
  `logging.lastResort` unformatted and unscrubbed, metering emitted nothing without
  `METER_LOG_PATH`, and the spawned inference worker never installed the metering handler,
  losing every per-extraction record with LLM token usage. Each process now installs one
  formatted, secret-scrubbed stderr sink (`SERVE_LOG_LEVEL`, default `info`), uvicorn and
  LitServe records ride it, and metering goes to stderr or, when configured, only to the
  JSONL file — from the worker too.
- **A born-digital window no longer starts an OCR engine it will not use.** OcrStage waited
  for the engine before counting the regions that would call it, and the automatic `paddle`
  chain was started before layout to key an OCR cache that is off by default. The count
  now comes first and, when native text covers every region, no engine is started or
  awaited; with `CACHE_OCR` off the automatic chain starts after native text is known.
  Captions, table titles and formula numbers still go through OCR by design.

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

### Changed

- Release preparation supports a manual rehearsal on `main` that tests and validates
  the distributions without publishing. PyPI uploads use Trusted Publishing on a
  GitHub-hosted runner and stay disabled until `PUBLISH_PYPI=true` is explicitly set.
- Shorten the README, keep the illustrated banner, and link to detailed guides.
  Add a draft LLM-use disclosure and clarify extraction accuracy limits and the
  current focus on English-language social science papers.
- The README opens with a paper-cream banner with square corners and no outer border.

- **The launch export uses schema 11.0 (breaking).** `schema_version` is at the root;
  `info` becomes `metadata`, `info_match` becomes `metadata_match`, and input file identity
  moves to `source`. Root `affiliations` becomes `affiliation`. Telemetry moves under
  `extraction`: engines, settings, timings (`stages` and `total_seconds`), usage (`totals`
  and per-label/provider/model `breakdown`), enrichment, diagnostics, identity receipts,
  warnings, and optional regions/trace. Validation findings live in `validation.issues`.
  `xref[].xref_id` becomes `target_id`; equations and funding gain explicit IDs;
  table cell contents are string grids. Match-table structured `authors`/`editors` become
  singular `author`/`editor`. The v10 output mode is retired; core checkpoints and enrichment
  sidecars reject older schema versions. The evaluation tools still read frozen v10 gold
  alongside v11 predictions without changing the scoring rules.
- **MCP Python SDK v2**, locked to 2.2.0. The server uses `MCPServer` and the public HTTP
  lifespan/idle-timeout API. Paper tools run on the event loop and keep stores isolated
  across initialized clients, including clients sharing a bearer key. HTTP clients negotiate
  the session-based 2025-11-25 protocol, which the chew/query workflow requires; automatic
  v2 clients fall back from sessionless discovery. Upload limits account for base64 overhead.

- Evaluation, aspect scoring and the benchmark harness now share `metrics_version=6`.
  The benchmark headline author score uses full printed names; family-name-only scores
  remain available as diagnostics. Re-score older benchmark records before comparing them.

- The LLM rate-limit slot is now acquired once inside `_invoke_structured`, below the cache
  check, instead of separately at each of the twelve call sites. A cache hit spends no
  provider quota, so it no longer waits on the budget that exists to protect that quota —
  previously a fully-cached corpus re-run was still paced at `LLM_RATE_LIMIT_RPM`. Live
  calls are unaffected: still one slot per dispatched request, plus one per retry.
- **The install extras are reorganised around that runtime.** `onnxruntime`, `tokenizers`,
  `huggingface-hub`, `scikit-learn` and `joblib` move into the core dependencies, so a
  plain `pip install bibr` runs the whole HTTP-service path — OCR and the LLM over HTTP,
  every bibr-owned model through ONNX Runtime — with no `torch`, `transformers` or OpenCV
  in the environment. The PyTorch stack is now the **`torch`** extra (training parity,
  Apple MPS, `torch.compile` on the serve layout model, transformers OCR, the CRF
  reference segmenter, and the fallback runtime); **`ml` is kept as an alias for it**, so
  existing installs, Dockerfiles and `bibr setup` plans are unaffected. `all` now bundles
  `batch,cache,demo,mcp,torch`. The two OpenCV calls in `bibr/ocr/image_processing.py` are
  Pillow/numpy.

- **The CLI stops treating a missing `torch` as a broken install.** `bibr doctor` reports
  the ONNX Runtime execution provider as the device instead of failing, and calls
  `seg=geom, parse=ner` healthy on a core install; only `REF_SEG_STRATEGY=crf` (torch-only,
  no ONNX export) and an explicit `ML_RUNTIME=torch` without torch still fail. `bibr chew`
  no longer refuses PDFs when OpenCV is absent — cv2 is reachable only through the torch
  layout path — and `--dry-run` names the ONNX provider it would use.

- **Enrichment's network wait overlaps the extract stage.** When enrichment is on, the
  enrich stage's up-front round-trips (resolver health probe and title searches, the
  Crossref bulk DOI lookup) start as soon as the references are parsed — while citation
  linking and structured-integrity LLM calls are still running — instead of strictly after
  extraction. `enrich_references` consumes the
  `EnrichmentPrefetch` when the pipeline hands it one and is unchanged otherwise; the core
  checkpoint still sees unenriched references, the enrichment stage's accounting is
  unchanged, and every path that does not enrich cancels the task. `extraction.timings`
  gains `enrich_prefetch` (its wall time; excluded from `total_seconds`).
- **Crossref reference enrichment is opt-in.** `CROSSREF_ENRICH` now defaults to `false`:
  a plain `bibr chew`, `bibr.chew()` or `POST /papers/extract` no longer calls Crossref or
  the resolver, `bib_match` stays empty, and `extraction.crossref_enrich` reports the
  effective per-run value. Enrichment was a network fan-out that added seconds of serial
  wall time per paper for every caller, including those that never read `bib_match`.
  Deployments that relied on the old default must set `CROSSREF_ENRICH=true` (or pass the
  per-run switch above); `bibr setup` now asks before writing it, and only offers
  consolidation once enrichment is on.

- **`bibr serve` keeps CPU-bound work off the shared event loop.** One LitServe worker runs
  with `enable_async=True`, so synchronous CPU inside a coroutine is head-of-line blocking
  for every co-resident request. Post-parse, citation linking and OCR post-processing now
  offload to a thread like their neighbouring stages (40.0 ms → 5.2 ms loop-tick latency for
  this class of work), four exact necessary-condition prefilters remove ~52 ms/paper of
  regex sweeps outright, and OCR crops moved inside the region semaphore (`Image.crop` is an
  eager copy; every crop of every page was held at once, ~1 GB at 8 in-flight requests).
- **The serve container image ships the `mcp` extra.** `Dockerfile.serve` now installs
  `bibr[mcp]`, so `MCP_ENABLED=true` on the Compose stack mounts the remote MCP endpoint
  without a custom build. The dependency is inert unless enabled.
- **CI runs for `main` only.** The retired February `dev` branch no longer triggers the
  suite on push, and pull requests can no longer target it.
- Removed unsupported comparative accuracy claims from the public documentation.
- **Every Hub-loaded model is pinned to a commit.** PP-DocLayoutV3, the section and paper
  classifiers and the default `sat-6l-sm` sentence segmenter loaded `main`, so a hub
  push could change extraction output between two runs of the same bibr version. Their
  audited commits are now the defaults (`LAYOUT_MODEL_REVISION`,
  `ML_SECTION_CLASSIFIER_REVISION`, `ML_PAPER_CLASSIFIER_REVISION`,
  `WTPSPLIT_MODEL_REVISION`; set any to `main` to track the head), `Dockerfile.serve`
  bakes the same revisions, and `scripts/prefetch_segmenter.py` accepts `--revision`.
- **The managed vLLM pin moves to 0.26.0** (`vllm` extra and the `uv tool run` bootstrap).
  It closes GHSA-87x5-vmc3-756j (completion prompt lists fanning out into unbounded engine
  requests) and drops `diskcache`, whose unfixed advisory bibr had been carrying as an audit
  exception; torch stays at 2.11.0. The lock resolves cleanly and the extra installs;
  serving with 0.26.0 has not yet been exercised on a GPU.
- **`LIMITATIONS.md` is current again** (native-format inputs, the `ner` default, the
  classifier-degraded warnings, single-tenant serve/MCP, and which benchmark numbers are
  held-out), and the local model registry's sizes were re-verified against the Hub.

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

### Security

- Remove the unused Accelerate dependency from the PyTorch extras and lockfile,
  eliminating CVE-2026-69112 from supported bibr installations. Existing environments
  need a locked sync or rebuild to remove the previously installed package.
- **A configuration error no longer prints your API keys.** `ConfigurationError` rendered
  pydantic's `input` payload; for a model-level validation failure that payload is the whole
  merged settings mapping, so one bad value printed every key in the environment to stderr
  and into any log collecting it. Model-level errors now omit the input, and a secret-named
  field's value is masked wherever it appears.
- **`MCP_URL_ALLOWED_HOSTS` accepts the form the docs give.** As a bare `list[str]`,
  pydantic-settings JSON-decoded it, so `MCP_URL_ALLOWED_HOSTS=arxiv.org,zenodo.org` failed
  startup outright — in practice no deployment had the `chew_url` SSRF allowlist on. The
  comma-separated and JSON forms both parse now, here and for the CORS lists.
- **No credential literals in the tree, and CI now scans for them.** Six tracked scripts
  and a notebook carried a metacheck platform API key as a string; they read it from the
  environment now (`PLATFORM_API_KEY`, `METACHECK_PLATFORM_API_KEY` for the `data/`
  scripts). A required gitleaks job scans the checked-out tree and the commits every pull
  request introduces, alongside Semgrep's tree-only secrets pack; `.gitleaks.toml` holds
  the allowlist of documented placeholders and test fixtures. The same scan runs as a
  pre-commit hook over the staged diff.

### Added

- **Structured reference names alongside the verbatim strings.**
  `bib[].authors` and `bib[].editors` stay exactly as printed; new `bib[].author` and
  `bib[].editor` carry a best-effort split into `{family, given, suffix}`, or a `{literal}`
  fallback for corporate and unsplittable names, and are `null` when there was nothing to
  split (never `[]`). Every emitted value is a substring of the verbatim string, so a consumer
  can always fall back to it. `author[]` gains an optional `suffix`. Included in schema 11.0.
- **A machine-readable JSON Schema of the export** is committed at
  `docs/schema/bibr-export-v11.schema.json`, generated from the pydantic export models by
  `scripts/generate_schema.py`. A test fails when the file drifts from the models, and its
  `required` list is derived from the exporter's own omit rules (`OMITTABLE_ROOT_KEYS`), so the
  artifact can never call an always-present table optional.
- **Opt-in LLM response cache** (`CACHE_LLM=true`, directory `CACHE_LLM_DIR`, default
  `$XDG_CACHE_HOME/bibr/llm`). Structured responses are cached on disk keyed by model,
  response schema, system prompt, user text, per-task `max_tokens`/`reasoning_effort`, and
  transport mode — so an entry can only serve a request that would have produced it. A hit
  costs no tokens; a miss, a stale entry, or an unwritable cache directory all fall through
  to a live call, so nothing about correctness depends on it. Re-running a corpus after a
  parser change (or an evaluation sweep over the same papers under different non-LLM
  settings) now pays for its LLM work once instead of every time. Off by default, like the
  OCR disk cache. Note the key canonicalises the per-call `uuid4` prompt-injection fence
  boundary, which 10 of the 13 call sites mint fresh each call — without that the same
  logical request would hash differently on every run and never hit.
- **`bib[]` carries the five reference fields the parser tagged and the decoder threw
  away (export schema 10.8).** The NER parser's 39-tag BIO scheme has covered `ARXIV`,
  `PMID`, `SERIES`, `ACCESS_DATE` and `NOTE` since v4, but `map_fields_to_paper_ref` had
  no target for any of them, so every predicted value was discarded at decode — `PMID`
  reaches 0.947 F1 on the JATS-supervised corpus and reached nothing else. They are now
  `PaperReference` fields (`arxiv`, `pmid`, `series`, `access_date`, `note`), exported
  verbatim as printed, and a test asserts no field type can be tagged and silently
  dropped again. Output from the shipped `bibr-parser-v4-5-gold` is unchanged in
  substance — its training corpus had no examples of any of the five, so it emits none —
  and the fields are explicit nulls. The LLM reference schema is deliberately *not*
  widened: the fields are removed from the JSON schema both LLM paths read, because the
  NuExtract template is qualified against a fixed shape and the LFM2.5 student was
  distilled on prompts embedding this exact schema.
- **A torch-free core: bibr's four local models now run on ONNX Runtime.** The layout
  detector, the section and paper classifiers and the ModernBERT+CRF reference parser each
  ship an `onnx/` bundle (graph, a `bibr_onnx.json` contract carrying preprocessing
  constants, label classes and CRF parameters, and the exact tokenizer) alongside the
  PyTorch weights at the same pinned revision. `ML_RUNTIME=auto|onnx|torch` chooses:
  `auto` prefers the ONNX bundle, falls back to PyTorch when the bundle is absent and
  `torch` is importable, and otherwise raises a `ConfigurationError` naming the model and
  the fix. `scripts/export_onnx_*.py` rebuild the bundles and check parity against the
  PyTorch classes; `bibr/ner/crf_numpy.py` is a numpy Viterbi decoder so the parser needs
  no `pytorch-crf`, and `bibr/utils/onnx_tokenizer.py` tokenizes through `tokenizers`
  alone. Layout's PyTorch weights live in a third-party repo, so its ONNX artifact has its
  own `LAYOUT_ONNX_MODEL_ID` / `LAYOUT_ONNX_REVISION`, published as
  `scienceverse/bibr-layout-onnx`. All four bundles are on the Hub and pinned, so a core
  install — 1.0 MB wheel, 677 MB venv, no `torch`, `transformers` or OpenCV — downloads
  them on first use with nothing to configure.

- **`JOBS_STORE=redis` shares async-job state between bibr-serve replicas.** Job status,
  results (zlib-compressed, under their own key) and the active-job cap move into Redis,
  so several `bibr serve` instances behind a load balancer answer status/result polls for
  each other's jobs, and `JOBS_MAX_ACTIVE` / `JOBS_MAX_RETAINED` /
  `JOBS_MAX_RETAINED_BYTES` bound the whole deployment. Uploads and execution stay on the
  replica that received the upload, and every job status now reports that `replica`.
  Admission is one Lua script (no cap race between replicas); every Redis call is bounded
  by the `REDIS_*_TIMEOUT_SECONDS` budgets; an unreachable store answers
  `503 {"detail": "job store unavailable"}` on the job routes and `jobs_store: error` on
  `/ready`; a replica lost mid-job frees its cap slots after a 24 h safety TTL. New
  settings: `JOBS_STORE`, `JOBS_REDIS_URL` (falls back to `REDIS_URL`), `JOBS_KEY_PREFIX`,
  `JOBS_REPLICA_ID`. The in-process store is unchanged and remains the default
  (`bibr.serve.jobs.JobStore` is now the protocol; the class is `MemoryJobStore`).
  Handing queued work to another replica (a shared queue) is documented as a follow-up.
- **Front-role classifier for front matter.** `bibr/extract/front_role.py` loads a small
  gradient-boosted bundle (`ML_FRONT_ROLE_MODEL_ID`, defaulting to the published
  `scienceverse/bibr-front-role-v1` at a pinned revision) that scores every OCR
  region as title / byline / affiliation / abstract / keywords / doi_line / masthead /
  heading / ref_header / body / other from page-relative geometry, relative font size and
  script-independent text shape. Front-matter ownership uses the scores as additive
  evidence (a model byline survives the English byline shape and the 45-word cap, a model
  title seeds non-Latin records, a confident masthead cannot root a record) and
  `RefLocator` accepts a model `ref_header` heading in any language. A title seed the model
  confidently types as something else keeps its title role and loses only the right to root a
  *second* record (`ML_FRONT_ROLE_RECORD_ROOT_CONFIDENCE`, default `0.9`) — boxed headers
  like `Correspondence` and `A R T I C L E I N F O` score `heading` at 1.00 and otherwise cut
  a page's real title away from its own abstract. The model is trained from publisher JATS projected onto cached OCR regions;
  see `docs/guides/classifiers.md`.

- **`bibr batch` — a first-class, resumable corpus runner.** Takes manifests (one path per
  line, `#` comments), directories (recursive) or files, writes `<out>/<paper_id>.json` per
  paper and an append-only `<out>/outcomes.jsonl` ledger — one line per attempt with
  status, error code and stage, timings, per-stage times, LLM tokens, reference and match
  counts, warning frequencies, bibr version and build sha. Re-running the same command
  resumes (`ok` skipped, `failed` skipped unless `--retry-failed`, `--force` for all;
  interrupted papers run again by default); `--limit`, `--shuffle`/`--seed` and
  `--deadline` shape a leg. Locally it feeds one warm pipeline in `--batch-size` chunks
  with every `bibr chew` option; with `--serve-url` it drives a `bibr serve` job API with
  adaptive concurrency (429 drops in-flight to `--min-concurrency`, 5xx/connection errors/
  upstream outages retry with backoff, successes grow back toward `--max-concurrency`) and
  a graceful Ctrl-C. `bibr batch report <out>` (or `--json`) summarises a ledger: ok/failed,
  throughput, latency percentiles, stage-time shares, tokens, match rate, failure and
  warning breakdowns; every run ends with the same table. `run_info.json` records the
  options, the serve build and a secret-redacted settings snapshot. `bibr chew` gains
  `--include-regions` as an alias of `--regions`. Guide: `docs/guides/batch.md`.
- **A per-run switch for reference enrichment.** `bibr chew --crossref` (mutually
  exclusive with `--no-crossref`), `bibr mcp --crossref`, `bibr.chew(..., crossref=True|False)`,
  the `crossref=true|false` multipart field on `POST /papers/extract`, and the `crossref`
  knob on the serve MCP `chew_paper`/`chew_url` tools all force enrichment on or off for
  that run, overriding `CROSSREF_ENRICH` either way. `RunConfig.crossref` is tri-state
  (`None` follows the setting) and resolves through `RunConfig.enrichment_enabled(settings)`;
  the serve response cache keys on the effective value, so an enriched and an unenriched
  result for the same file never collide. `bibr chew --dry-run` names why enrichment is
  off and how to turn it on.

- **`BIBR_DISABLE_DOTENV=1`** makes every settings model ignore `./.env` and `~/.bibr/.env`
  (the process environment still applies). `python -m benchmarks run --tool bibr` refuses
  to start while either file exists unless it is set, so a run's recorded configuration is
  the profile plus the environment and nothing a developer's `.env` slipped in.
- **`bibr.local-default` benchmark profile** (`geom` segmentation + `ner` parsing, what a
  fresh `bibr setup` runs) next to the LLM-parse `bibr.default`, so the install default
  can be promoted as its own row.

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
