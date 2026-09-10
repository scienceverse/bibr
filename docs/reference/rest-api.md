# REST API

`bibr serve` runs a [LitServe](https://lightning.ai/docs/litserve/home)
application exposing the bibr extraction pipeline over HTTP. See
[Production deployment](../guides/deployment.md) for running the server
(Docker, concurrency, hardware sizing); this page documents the REST
surface — endpoints, request/response shapes, caching, and error codes.

The server binds to `127.0.0.1:8000` by default. Interactive docs are at
`/docs` (Swagger UI) and `/redoc`; these and `/openapi.json` require the same
bearer token as extraction when authentication is enabled.

## Endpoints

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe -- LitServe's built-in endpoint; returns plain-text `ok` (status `200`) |
| `GET` | `/ready` | Readiness probe -- checks OCR service, configured classifier artifacts, and Redis when response caching is enabled; `200` when ready, `503` otherwise |

The probes are public. With authentication enabled, an anonymous `/ready`
response contains only `{"status": "ready"}` or `{"status": "not_ready"}`.
A valid bearer token also exposes `checks` and the deployment `build_sha`.

### Papers

#### `POST /papers/extract`

Extract metadata from a scientific paper synchronously. Returns JSON.

**Request:** `multipart/form-data`

This is the only public synchronous extraction ingress. The complete multipart
body is limited to 51 MiB by default, including boundaries, headers, and form
fields; the `file` bytes within it are limited to 50 MiB. At most 1 MiB of the
upload remains in API memory before the multipart spool rolls to disk.
Exactly one `file` part is accepted. The seven optional fields below must each
appear at most once and are capped at 64 bytes; duplicate/unknown parts or a
second file return `400`.

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | Paper file (PDF, DOCX, XML/JATS, HTML, or ePub) |
| `start_page` | int | No | Start page for PDFs (0-indexed, inclusive) |
| `end_page` | int | No | End page for PDFs (0-indexed, inclusive) |
| `include_figures` | bool | No | Emit base64-encoded figure images (default: `false`) |
| `include_regions` | bool | No | Emit `_regions` and `_native_source` diagnostics (default: `false`). Character geometry and repair receipts can substantially increase response size. |
| `consolidate` | `fill` \| `replace` | No | Merge accepted Crossref matches into `bib` before export (`fill` fills only missing fields, `replace` also overwrites disagreeing ones). Omit to defer to the server's `CROSSREF_CONSOLIDATE` setting. |
| `refs` | `ner` \| `llm` \| `llm-chunked` \| `off` | No | Per-request override of the reference-parsing strategy (`REF_PARSE_STRATEGY`). |
| `ref_seg` | `geom` \| `region` \| `llm` \| `crf` | No | Per-request override of the reference-segmentation strategy (`REF_SEG_STRATEGY`). |

Page indexes must be nonnegative, and `end_page` must not precede `start_page`.
The server caps the processed PDF range at `PIPELINE_MAX_PAGES` (default 200).
`refs=off` keeps core metadata extraction but produces empty bibliography,
reference matches, and citation links. Native DOCX, XML/JATS, HTML, and ePub
inputs skip PDF OCR; embedded PDF text is used for eligible regions.

**Response:** JSON conforming to the bibr v{{ schema_version }} schema. Top-level keys include: `paper_id`, `info`, `author`, `text`, `section`, `url`, `bib`, `bib_match`, `info_match`, `funding`, `affiliations`, `xref`, `citation_linking` (optional citation detector scores and accepted/rejected candidate receipt), `caption_assignment` (optional document-wide caption ownership receipt), `reference_yield` (optional reference segmentation/yield receipt), `figure`, `table`, `eq`, `ocr_config`, `enrichment`, `extraction`, `processing_warnings`, `llm_usage` (per-paper LLM token counts by model; `null` when no LLM ran), `llm_usage_by_label` (the same token counts attributed per extraction task; `null` when no LLM ran), `qualification_provenance` (deployment-qualification surface: identity SHAs, per-task protocol hashes, native-validity + fallback outcome, request counts; `null` when no LLM ran), and `validation` (output-validation-gate result: error/warning counts and issue list). Figure and table rows retain their legacy primary fields and add ordered `parts` with physical payload and source provenance. `_regions` is appended when `include_regions=true`.

`info` is scalar-only by design — pipeline metadata (`ocr_config`, `processing_warnings`) lives at the top level so R consumers can call `as.data.frame(info)` cleanly.
See the generated [JSON schema reference](schema.md) for current fields and
their definitions. `include_regions=true` also includes `_native_source`
when native PDF diagnostics are available.

**Example:**

```bash
curl -X POST http://localhost:8000/papers/extract \
  -F "file=@paper.pdf" \
  -F "include_regions=false"
```

LitServe's internal `POST /_bibr/inference` route accepts only the API
process's opaque disk descriptor and returns `404` to direct HTTP callers.
Only the upload UUID, filename, size, SHA-256, and extraction options cross the
worker queue; neither bytes nor a filesystem path do. Worker decode securely
reads and verifies the owned file once, then deletes it.

#### `POST /papers/enrich`

Backfill external enrichment from a **saved extraction**, without a PDF upload,
OCR, layout detection, reference parsing, or LLM calls. This endpoint runs even
when `CROSSREF_ENRICH=false` disabled enrichment during extraction.

**Request:** `application/json`, `{"paper": <saved bibr JSON>}`. Accepts export
schemas 10.6 and 10.7. The submitted object must be a valid full export; duplicate
bibliography IDs, dangling matches and inconsistent enrichment counts return
`422`. Request bodies are limited to 16 MiB and bibliographies to 5,000 entries.
At most four backfills run concurrently; additional requests receive `429` with
`Retry-After: 1`. An incomplete body upload times out after 30 seconds (`408`).

```bash
jq '{paper: .}' paper.json > backfill-request.json
curl http://localhost:8000/papers/enrich \
  -H "Authorization: Bearer your-secret-token" \
  -H "Content-Type: application/json" \
  --data-binary @backfill-request.json > backfill-response.json
jq '.paper' backfill-response.json > paper-enriched.json
jq '.enrichment' backfill-response.json > paper-enrichment.json
```

**Response:** an envelope with:

| Field | Meaning |
|---|---|
| `paper` | The saved extraction with `bib_match`, `info_match`, and enrichment completeness updated |
| `enrichment` | Replayable sidecar containing `core_sha256`, `settings_digest`, schema version, completeness, matches, and any warnings |
| `enrichment_version` | Backfill policy revision (`crossref-backfill-v1`) |
| `enrichment_key` | Hash of the exact submitted artifact and the enrichment settings digest |
| `status` | `complete`, `partial`, or `no_work` |

Existing matches are retained. A bibliography already marked `enrichment.complete`
is skipped, including references with a completed lookup that found no match.
Otherwise, only entries without a match are looked up. The paper's own DOI is
looked up when `info_match` is empty. Submit the returned `paper` to retry a
partial result; successful matches survive upstream failures and timeouts.
`complete` describes completion of the lookups, not a promise that every entry
has a match. A paper-DOI miss has no separate persistent receipt in the export,
so it may be looked up again on a subsequent submission.

The endpoint uses the configured resolver/Crossref routing, caches, rate limits,
and `CROSSREF_ENRICH_TIMEOUT`. `CROSSREF_CONSOLIDATE` does not apply: printed
`bib`, `info`, body text, and extraction provenance remain unchanged. The existing
enrichment-pending validation gate is cleared only on complete enrichment; other
validation issues remain. Partial results return `200` with `status: "partial"`
and diagnostic warnings, so callers can save the usable matches.

An empty bibliography remains empty: papers extracted with `refs=off` need
reference extraction before reference enrichment. The endpoint stores no papers
or sidecars itself. Keep the submitted JSON alongside its sidecar; replay rejects
a different core hash or settings digest. The key identifies the input and policy,
not an immutable snapshot of the external databases or a server-side job/cache.

### Async jobs

Holding an HTTP connection open for a full extraction (tens of seconds) is
fragile behind proxies and load balancers. These routes offer a
fire-and-poll alternative instead:

#### `POST /papers/jobs`

Accepts the same `multipart/form-data` fields as `/papers/extract`.
Returns `202` immediately with `{"job_id", "status": "queued",
"status_url"}` and runs the extraction in the background. It persists the
upload once and dispatches the same opaque descriptor through LitServe; it
does not rebuild or self-proxy a multipart request.

#### `GET /papers/jobs/{id}`

Returns the job's status (`queued`, `running`, `succeeded`, or `failed`)
plus timestamps — no result body. A `succeeded` status includes a
`result_url` pointing at the next endpoint.

#### `GET /papers/jobs/{id}/result`

Returns the extracted paper JSON once the job has `succeeded` (same shape
as `/papers/extract`'s response). Responds `409` while the job is still
queued/running, or the job's original error and status code if it failed.

Jobs are held in an in-process store. The server admits up to
`JOBS_MAX_ACTIVE` (default `32`) queued plus running jobs and dispatches
at most `JOBS_MAX_RUNNING` (default `2`) concurrently. Excess submissions
receive `429`. Completed job records expire after `JOBS_TTL_SECONDS`
(default `3600`), and `JOBS_MAX_RETAINED` (default `128`) also bounds retained
results by evicting the oldest completed records. Fetch and save results
before they expire or are evicted; later requests receive `404`.

The whole async API can be disabled with `JOBS_ENABLED=false`. Job queue,
status, and results are process-local. The service always pins one HTTP API process—even with jobs
disabled—because upload ownership and dispatch tracking are also process-local.
A complete server restart loses those records. `PIPELINE_RESTART_WORKERS=false`
fail-stops on worker death; `true` is an unsupported opt-in until the locked
LitServe compatibility gate proves reliable completion notification and does
not make jobs durable.

## Authentication

Set `AUTH_API_KEY` to require a bearer token on every route except
`/health` and `/ready`:

```bash
curl -X POST http://localhost:8000/papers/extract \
  -H "Authorization: Bearer your-secret-token" \
  -F "file=@paper.pdf"
```

A missing or wrong token gets a `401` with a `WWW-Authenticate: Bearer`
header. When `AUTH_API_KEY` is unset, the CLI permits loopback-only serving;
network-visible binds require a key at least 32 characters long.
See [Authentication](../guides/deployment.md#authentication) in the
deployment guide for the production-hardening checks (`ENVIRONMENT=production`)
that force it on.

## Caching

When `CACHE_ENABLED=true` (the default) and Redis is configured, the API caches
successful extraction responses. Keys distinguish file content, page range,
figure/region output, consolidation, and reference-strategy overrides. The
cache namespace also includes a settings fingerprint and code version.
Identical concurrent cache misses are coalesced; failed Redis operations are
bounded and extraction continues without the cache.

Configure caching:

| Variable | Description | Default |
|---|---|---|
| `CACHE_ENABLED` | Enable response caching when Redis is configured | `true` |
| `REDIS_URL` | Redis connection URL | auto-generated |
| `REDIS_PASSWORD` | Redis password | (none) |
| `CACHE_VERSION` | Cache key prefix version | auto-computed from source hash |
| `CACHE_TTL_SECONDS` | Cache TTL | `86400` (24h) |
| `CACHE_OPERATION_TIMEOUT_SECONDS` | Maximum wait for one cache operation | `5` |

## Request metering

With `METER_ENABLED=true` (the default), non-probe HTTP responses carry
`x-request-id` and `x-bibr-duration-ms`. A valid client-supplied `x-request-id`
is echoed; otherwise the server generates one. Request and extraction records
go to the `bibr.serve.metering` logger; `METER_LOG_PATH` optionally adds a
rotating JSONL file. Cache hits do not count the original extraction's LLM
tokens as new usage.

## Error responses

| Status | Meaning |
|---|---|
| `400` | Invalid input (missing filename, malformed/bounded option, duplicate or unknown multipart part) |
| `401` | Missing or invalid bearer token (`AUTH_API_KEY` set) |
| `404` | Unknown, expired, or evicted job; direct request to the private inference route |
| `408` | Saved-export enrichment body upload exceeded 30 seconds |
| `409` | Job result requested before the job finished |
| `413` | Upload limit exceeded (50 MiB file / 51 MiB multipart envelope), or enrichment exceeded 16 MiB / 5,000 references |
| `415` | `/papers/enrich` received a content type other than `application/json` |
| `422` | Extraction processing error, or invalid saved export for enrichment |
| `429` | Upload admission, async-job active cap, or enrichment concurrency limit reached |
| `500` | Unexpected internal error |
| `502` | Upstream service failed (OCR server, LLM API) |
| `503` | `/ready` reports an unavailable dependency or required classifier artifact |
| `504` | Pipeline processing timed out |
| `507` | Insufficient temporary storage for the disk-backed upload spool |
