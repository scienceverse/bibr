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
fields; the `file` bytes within it are limited to 50 MiB (with `MCP_ENABLED`
the body cap grows to fit a 50 MiB file in base64 form — see the MCP guide).
At most 1 MiB of the upload remains in API memory before the multipart spool
rolls to disk.
Exactly one `file` part is accepted. The eight optional fields below must each
appear at most once and are capped at 64 bytes; duplicate/unknown parts or a
second file return `400`.

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | Paper file (PDF, DOCX, XML/JATS, HTML, or ePub) |
| `start_page` | int | No | Start page for PDFs (0-indexed, inclusive) |
| `end_page` | int | No | End page for PDFs (0-indexed, inclusive) |
| `include_figures` | bool | No | Emit figure images as `data:` URIs (default: `false`) |
| `include_regions` | bool | No | Emit the `extraction.regions` layout debug payload (default: `false`). Contains per-region geometry and recognition content; response size depends on the document. |
| `crossref` | bool | No | Run Crossref/resolver reference enrichment for this request (`true`) or skip it (`false`). Omit to follow the server's `CROSSREF_ENRICH` setting, which is off by default. The response cache keys on the effective value. |
| `consolidate` | `fill` \| `replace` | No | Merge accepted Crossref matches into `bib` before export (`fill` fills only missing fields, `replace` also overwrites disagreeing ones, but only from a match carrying the reference's printed DOI). Omit to defer to the server's `CROSSREF_CONSOLIDATE` setting. |
| `refs` | `ner` \| `llm` \| `llm-chunked` \| `off` | No | Per-request override of the reference-parsing strategy (`REF_PARSE_STRATEGY`). |
| `ref_seg` | `geom` \| `region` \| `llm` \| `crf` | No | Per-request override of the reference-segmentation strategy (`REF_SEG_STRATEGY`). |

**Response:** JSON conforming to the bibr v{{ schema_version }} schema. Top-level keys, grouped by role: the paper and its input file — `paper_id`, `schema_version` (its presence at the root is how readers dispatch v11 and later from earlier versions), `source` (input-file identity: file name, SHA-256, format); what the paper says — `metadata` (scalar paper-level metadata), `author`, `affiliation`, `funding`, `text`, `section`, `url`, `bib`, `xref`, `figure`, `table`, `footnote`, `eq`; what external registries returned — `metadata_match` (matches for the paper's own identity), `affiliation_match` and `funding_match` (ROR organizations), `bib_match`; how the output was produced — `extraction` (engines, per-run settings, timings, LLM usage, enrichment completeness, identity receipts, qualification provenance, the output-validation result (`extraction.validation`: error/warning counts, promotion disposition and issue list), diagnostics receipts, figure/table piece locations and warnings; `regions` is added there when `include_regions=true`). Every key is always present, and content rows carry no processing fields. Figure and table rows describe the whole object.

`metadata` is scalar-only by design — pipeline telemetry lives under `extraction` and the input file's identity under `source` — so R consumers can call `as.data.frame(metadata)` cleanly.

Within major version 12 the schema is additive-only: new optional fields and
new enum values may appear in any `12.x` release, and clients should ignore
keys they don't recognize and accept enum values they don't know. Dispatch on the *presence* of the root `schema_version` key, never
on parsing its value — pre-v11 responses have no such key at all. See
`CHANGELOG.md` for the v12 break and forward-versioning policy.

**Example:**

```bash
curl -X POST http://localhost:8000/papers/extract \
  -F "file=@paper.pdf" \
  -F "include_regions=false" \
  -F "crossref=true"
```

LitServe's internal `POST /_bibr/inference` route accepts only the API
process's opaque disk descriptor and returns `404` to direct HTTP callers.
Only the upload UUID, filename, size, SHA-256, and extraction options cross the
worker queue; neither bytes nor a filesystem path do. Worker decode securely
reads and verifies the owned file once, then deletes it.

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

Jobs are held in an in-process store and purged after `JOBS_TTL_SECONDS`
(default `3600`); `JOBS_MAX_ACTIVE` (default `32`) caps concurrently
active jobs, returning `429` past the cap, and `JOBS_MAX_RUNNING` (default
`2`) caps how many run at once. Completed results are also evicted
oldest-first beyond `JOBS_MAX_RETAINED` results (default `128`) or
`JOBS_MAX_RETAINED_BYTES` of encoded JSON (default 256 MiB; `0` disables the
byte budget); the newest result is always kept, so a fetch of `/result` can
answer `404` once a result has been evicted. The whole async API can be
disabled with `JOBS_ENABLED=false`. By default (`JOBS_STORE=memory`) the job
queue, status, and results are process-local and a server restart loses them.
`JOBS_STORE=redis` keeps status and results in Redis instead, so any replica of
a load-balanced deployment answers the polls for a job another replica accepted
and the active-job cap spans all replicas — see
[Multiple bibr-serve replicas](../guides/deployment.md#multiple-bibr-serve-replicas).
Each status carries `replica`, the instance executing the job; with the Redis
store unreachable the job routes answer `503`. The service always pins one HTTP
API process per instance—even with jobs disabled—because upload ownership and
dispatch tracking are process-local. `PIPELINE_RESTART_WORKERS=false`
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
| `404` | Unknown job id (expired past `JOBS_TTL_SECONDS`, evicted by the retention limits, or never existed) |
| `409` | Job result requested before the job finished |
| `413` | Upload limit exceeded (50 MiB file / 51 MiB multipart envelope) |
| `422` | Extraction processing error |
| `429` | Upload admission or async-job active cap reached |
| `500` | Unexpected internal error |
| `502` | Upstream service failed (OCR server, LLM API) |
| `503` | `/ready` reports an unavailable dependency or required classifier artifact |
| `504` | Pipeline processing timed out |
| `503` | Job store unreachable (`JOBS_STORE=redis`): the upload was dropped and nothing queued — retry later |
| `507` | Insufficient temporary storage for the disk-backed upload spool |
