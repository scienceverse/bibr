# Production deployment

`bibr chew` processes files and directories locally. For production use, batch
processing, or team-shared infrastructure, `bibr serve` runs the same
pipeline behind an HTTP API — with an external OCR server and optional
Redis caching. This guide covers running it: the API itself, its
concurrency model, authentication, the OCR server it depends on, the
bundled Docker stack, reference extraction at scale, and hardware sizing.

## One-machine API

```bash
bibr serve
```

This starts a [LitServe](https://lightning.ai/docs/litserve/home)
application exposing the full extraction pipeline over HTTP, bound by
default to loopback at `127.0.0.1:8000`. Two flags control the bind address:

| Flag | Description | Default |
|---|---|---|
| `--host` | Host to bind to | `127.0.0.1` |
| `--port` | Port to bind to | `8000` |

```bash
bibr serve --host 127.0.0.1 --port 9000
```

Binding to a non-loopback address is refused unless `AUTH_API_KEY` is set
and is at least 32 characters long. Core installs run local layout and
classifiers through ONNX Runtime. The bundled serve image selects the PyTorch
runtime; see [local model runtime](configuration.md#local-model-runtime).
Add `cache` to use Redis and `mcp` for the optional agent endpoint.

The REST surface itself — `/papers/extract`, the async job API, health
and readiness probes, request/response shapes, caching, and error codes —
is documented in the [REST API reference](../reference/rest-api.md); this
page focuses on running the server. `/ready` checks the OCR endpoint,
configured classifier artifacts, and Redis when response caching is enabled.
Without a valid bearer token, it reports only the overall status; authenticated
callers also receive the individual checks and `BIBR_BUILD_SHA`.

## Upload ingress and worker handoff

`POST /papers/extract` is the public `multipart/form-data` ingress. LitServe's
underlying `POST /_bibr/inference` route is private descriptor dispatch, not a
second upload API; direct HTTP requests to it always receive `404`.

The API process accepts exactly one `file` part and at most the eight documented
option fields. Duplicate/unknown fields, a second file, or any option value over
64 bytes is rejected with `400` before descriptor creation. Starlette retains at
most 1 MiB of the one file in API memory by default before its multipart spool
rolls to disk, then bibr streams it into an owner-only temporary directory. The
default limits are separate:

| Variable | Default | Applies to |
|---|---:|---|
| `PIPELINE_MAX_FILE_SIZE` | 50 MiB | File bytes after multipart parsing |
| `PIPELINE_MULTIPART_OVERHEAD_BYTES` | 1 MiB | Extra headroom for boundaries, headers, and form fields |
| `PIPELINE_UPLOAD_SPOOL_MEMORY_BYTES` | 1 MiB | Upload bytes retained in API memory before disk spill |

Consequently, the default complete multipart-body envelope is 51 MiB while the
file itself remains capped at 50 MiB. Increasing envelope headroom does not
increase the accepted file size. With `MCP_ENABLED=true` the envelope grows to
fit a 50 MiB file in base64 form (about 68 MiB), because `chew_paper` carries
its upload inside a JSON-RPC body; the file limit is unchanged.

Only an opaque descriptor crosses LitServe's multiprocessing queue: the
canonical UUID, filename, byte size, SHA-256, and extraction options. Neither
file content nor a filesystem path is queued. The worker derives the owned
path from the UUID, opens only a regular non-symlink file, reads and verifies
it once, and deletes it after that read even when integrity validation fails.
The API-side store leases each persisted UUID through synchronous dispatch or
the complete queued-job lifetime. Stale sweeping skips leased entries and only
removes unowned orphans; remove, completion, failure, queued discard, and
shutdown release ownership. The original Starlette spool is closed immediately
after private persistence, before long-running inference begins. Graceful
application shutdown removes remaining entries and the temporary root.

`PIPELINE_RESTART_WORKERS=false` is the safe default. On LitServe 0.2.17, a
replacement worker cannot reliably notify the API process whose in-flight
request belonged to the dead worker, so transparent replacement can strand that
request. The default therefore fail-stops the API and inference processes; use
an external supervisor to restart the service. Setting the variable to `true`
is an unsupported opt-in until the locked real-process compatibility gate
proves reliable death-path completion notification. It does not provide durable
request replay: uploads and dispatch remain process-local. Redis-backed job
status/results can survive an instance restart, but do not replay a lost upload.

The private manager/worker startup and fail-stop death contracts are covered by
real spawned-process compatibility tests. Keep LitServe constrained to `<0.3`;
deliberately migrate those tests together with the private startup helpers
before allowing 0.3 or newer.

## Concurrency model

`BibrPipelineAPI` runs as a single async worker that serves many requests
concurrently on one event loop, overlapping the I/O-bound stages
(OCR/LLM/Crossref HTTP calls) instead of blocking one request at a time.
The GPU-bound stages — layout detection and sentence segmentation — don't
duplicate this way: they route through a shared `GpuBatcher` (one
collector coroutine + one worker thread per model) that serializes
forward passes and coalesces concurrent requests' pages/texts into fuller
batches, so peak VRAM stays bounded to one batch regardless of how many
requests are in flight.

Scale concurrency with these settings:

| Variable | Default | Meaning |
|---|---|---|
| `PIPELINE_MAX_ACTIVE_UPLOADS` | `8` | Fail-fast API-server admission cap applied before multipart parsing; excess uploads receive `429`. Also covers the MCP chew tools: a large `/mcp` body holds a slot while it is received, and the tool holds one during extraction. |
| `PIPELINE_MAX_INFLIGHT_REQUESTS` | `8` | Max requests running the pipeline concurrently per worker (`0` = unlimited). Bounds peak host RAM (page images) under an upload flood. |
| `PIPELINE_RESTART_WORKERS` | `false` | Fail-stop on worker death. `true` is an unsupported opt-in until the locked death-path gate proves reliable completion notification. |
| `PIPELINE_MAX_PAGES` | `200` | Hard per-file processing cap for compact many-page PDF protection. |
| `PIPELINE_PAGE_WINDOW_SIZE` | `8` | Maximum PDF pages rendered per file at once. Images are released after each window's layout, native-text reconstruction, and OCR work. This bounds memory without shortening the document. |
| `LAYOUT_MAX_RENDER_PIXELS` | `25000000` | Reject a PDF page before rasterization if its configured-DPI image would exceed this pixel count. |
| `LAYOUT_MAX_RENDER_DIMENSION` | `10000` | Reject a PDF page before rasterization if either image dimension would exceed this many pixels. |
| `LAYOUT_BATCH_TIMEOUT_MS` | `5` | Coalescing window for the layout `GpuBatcher` — how long it waits for more concurrent pages before flushing a partial batch. |
| `SEGMENTER_BATCH_TIMEOUT_MS` | `5` | Same coalescing window for the sentence segmenter's `GpuBatcher`. |
| `LLM_MAX_CONCURRENCY` | `6` in serve | Shared limit for concurrent LLM calls. An explicit setting overrides the server default. |
| `JOBS_MAX_RUNNING` | `2` | Maximum async jobs dispatched to inference at once. |
| `JOBS_MAX_ACTIVE` | `32` | Maximum queued plus running async jobs. |
| `JOBS_MAX_RETAINED` | `128` | Maximum completed job results held in memory; oldest completed records are evicted. |

`PIPELINE_MAX_PAGES` caps the requested processing range; raise it explicitly
for longer documents and size memory for the concurrent file count.
`PIPELINE_TIMEOUT` defaults to 300 seconds.

### Why there is no worker-count setting

`bibr serve` runs **exactly one inference worker**, pinned in `build_server()`
and not configurable. That is a deliberate design point, not a conservative
default, so scale with the async concurrency settings above instead.

Additional workers would duplicate model weights and runtime memory, and split
the requests that the shared batchers combine. The supported scaling controls
are async request concurrency, model batch sizes, and the coalescing windows.

`PIPELINE_WORKERS_PER_DEVICE` was removed. Left in an existing `.env` it is
ignored rather than rejected, but it no longer does anything and should be
deleted.

`bibr serve` always uses one HTTP API-server process, even when jobs are
disabled or the inference-worker count is raised. The upload root, leases,
dispatch tracker, admission accounting, readiness state, and the job queue are
process-local and share one destructive cleanup lifecycle. To add capacity
beyond one instance, run more *instances* of `bibr serve` — each with its own
API process, inference worker, and upload root — behind a load balancer, and
share job state between them through Redis as described next.

## Multiple bibr-serve replicas

One instance scales by async concurrency (above). Past that, run several
instances — *replicas* — behind a load balancer. `/papers/extract` needs
nothing extra: each request is served entirely by the replica that receives
it. The async job API does, because a status poll can land on a different
replica than the upload did. `JOBS_STORE=redis` moves job state into a Redis
that every replica shares:

| Shared through Redis | Stays on the replica that took the upload |
|---|---|
| Job status and timestamps (`GET /papers/jobs/{id}`) | The uploaded file, in that replica's owner-only temporary directory |
| Results (`GET /papers/jobs/{id}/result`; stored zlib-compressed under their own key) | Execution: the replica's own `JOBS_MAX_RUNNING` dispatcher runs the job through its inference worker |
| The active-job cap: `JOBS_MAX_ACTIVE` counts queued + running jobs across all replicas | Admission (`PIPELINE_MAX_ACTIVE_UPLOADS`) and the in-flight limits |
| Retention: `JOBS_TTL_SECONDS`, `JOBS_MAX_RETAINED`, `JOBS_MAX_RETAINED_BYTES` bound the whole namespace, oldest first | |

Every job status carries `replica`, the id of the instance executing it.

```bash
# .env on every replica
JOBS_STORE=redis
# Redis for job state. Unset it to reuse the cache's REDIS_URL / REDIS_PASSWORD
# (the usual choice); set it to keep job state on a separate Redis or database.
# REDIS_PASSWORD is added to it when it names the same host and port as REDIS_URL.
JOBS_REDIS_URL=redis://redis-host:6379/1
JOBS_KEY_PREFIX=bibr:jobs   # identical on every replica that shares a namespace
JOBS_REPLICA_ID=api-1       # optional; defaults to <hostname>:<pid>
```

Point the load balancer at every replica's `/ready` and use plain round-robin;
no session affinity is needed. A replica that cannot reach the job store reports
`"jobs_store": "error"` (HTTP `503`) there, and until Redis is back its job routes
answer `503 {"detail": "job store unavailable"}` — the upload is dropped and nothing
is queued, so the client can simply retry. Every job-store call is bounded by
`REDIS_CONNECT_TIMEOUT_SECONDS` + `REDIS_SOCKET_TIMEOUT_SECONDS`, the same budgets
the response cache uses, so a stalled Redis cannot wedge a request. TLS termination,
the bearer token, and `/papers/extract` are unchanged. Replicas that share a Redis
but must not see each other's jobs (staging next to production, say) get distinct
`JOBS_KEY_PREFIX` values.

Two consequences of keeping execution on the receiving replica:

- A replica that *crashes* mid-job leaves its queued/running jobs reporting their
  last status until a 24-hour safety TTL reaps them, and they hold cap slots that
  long. A clean shutdown is different: the replica marks the jobs it abandons
  `failed` with `503 replica shut down before the job finished`, so they free their
  slots at once and clients know to resubmit. Drain a replica before stopping it
  (stop routing new uploads to it, let its running jobs finish) to avoid even that.
- Work spreads by which replica receives the upload, not by queue depth.

**Follow-up (not implemented): a shared queue.** Letting an idle replica execute a
job another replica accepted would need the upload bytes on shared storage and a
Redis work queue with a visibility timeout, so a dead replica's jobs are re-run
instead of stranded. The job record already separates state from executor; the
queue and the upload hand-off are the missing pieces.

## Logging and metering

`bibr serve` configures one log sink per process: the API process and the
spawned inference worker each write formatted lines (`time level logger:
message`) to stderr, where Docker and systemd collect them. `SERVE_LOG_LEVEL`
(default `info`) sets the level for bibr's own loggers and is handed to uvicorn
and LitServe; HTTP client libraries are held at `warning` so request URLs are
not logged. Every sink carries the secret scrubber, so bearer tokens, URL
credentials and `?key=` query strings are masked before they are written,
tracebacks included.

Metering (`METER_ENABLED`, default on) emits one JSON line per HTTP request
from the API process and one per extraction — with LLM token usage — from the
worker. With `METER_LOG_PATH` unset they go to stderr with the other logs; set
it to route them to a size-rotated JSONL file instead (`METER_LOG_MAX_BYTES`,
`METER_LOG_BACKUP_COUNT`), which both processes append to. Metering does not
follow `SERVE_LOG_LEVEL`.

## Authentication

```bash
# .env
AUTH_API_KEY=<output of: openssl rand -hex 32>
```

When `AUTH_API_KEY` is set, every route except `/health` and `/ready`
requires an `Authorization: Bearer <token>` header — the probes stay open
so orchestrators (Docker, Kubernetes) can check liveness/readiness without
credentials.

```bash
curl -X POST http://localhost:8000/papers/extract \
  -H "Authorization: Bearer your-secret-token" \
  -F "file=@paper.pdf"
```

Without `AUTH_API_KEY` the server binds only `127.0.0.1`, `::1` or
`localhost`, and it refuses requests that a web page open in your browser
could make on your behalf: any request whose `Host` header is not one of
those names (`localhost`, `127.0.0.1`, `[::1]`, any port; a DNS-rebinding
page sends its own) gets a `421`, and a `POST` or other state-changing
request from another site's `Origin` (or with `Sec-Fetch-Site: cross-site`)
gets a `403`. REST routes and `/mcp` apply the same rule. `bibr batch
--serve-url`, MCP clients, curl, and your own browser on
`http://127.0.0.1:8000/docs` keep working; origins listed by name in
`CORS_ORIGINS` are accepted too, but `CORS_ORIGINS=*` admits none here. To
put a reverse proxy or tunnel in front of the server, set `AUTH_API_KEY`:
with a key the bearer token is the boundary and these checks are off.

When `ENVIRONMENT=production`, the server refuses to start at all unless
its production hardening checks pass:

- `AUTH_API_KEY` is set — an unauthenticated production deployment is
  refused outright, and the key must be at least 32 characters.
- `REDIS_PASSWORD` is set.
- `CORS_ORIGINS` does not contain `*` — set it to an explicit list of
  allowed origins instead.

## MCP endpoint (optional)

With the `mcp` extra installed (`Dockerfile.serve` bakes it in, so the
Compose image needs no custom build), `MCP_ENABLED=true` mounts a
[Model Context Protocol endpoint](mcp.md#remote-mcp-on-bibr-serve) at
`/mcp` (streamable HTTP), so remote agents can chew and query papers as
tools. It shares the bearer auth above and dispatches extraction through
the same worker path as `POST /papers/extract`.

## OCR server

Unlike `bibr chew`, `bibr serve` has no in-process OCR mode: PDF regions that
need recognition use the external OCR service at `OCR_BASE_URL` (default
`http://localhost:8080`; Compose overrides it with its private `bibr-ocr`
service name). Native document formats skip OCR, and eligible PDF regions use
embedded text; the server still checks the configured OCR service in `/ready`.

`OCR_BASE_URL` is the server **root**: bibr appends `/v1/models` and
`/v1/chat/completions` itself and probes `/health` at the root, so do not
include a `/v1` suffix (a trailing `/v1` is stripped with a warning). For a
PaddleOCR-VL-1.6 service behind an OpenAI-compatible endpoint, name the server
family with `OCR_BACKEND=paddle-http`; that alone selects the Paddle served-model
alias and profile, and `OCR_MODEL` / `OCR_PROFILE` override them:

```bash
OCR_BACKEND=paddle-http
OCR_BASE_URL=https://ocr.example.internal
OCR_MODEL=paddle-ocr-vl-1.6   # optional under paddle-http
OCR_PROFILE=paddle            # optional under paddle-http
```

The bundled `bibr-ocr` Compose service is an explicit GLM-OCR compatibility
deployment — an SGLang server, GPU required — covered in [Docker
deployment](#docker-deployment) below. To use it or another GLM service, opt
in explicitly:

```bash
python -m sglang.launch_server \
  --model zai-org/GLM-OCR \
  --revision ca5d8b3e287e52589e37c28385d9655ee4372f9d \
  --port 8080 --served-model-name glm-ocr

OCR_BACKEND=glm-http
OCR_MODEL=glm-ocr
OCR_PROFILE=glm
```

If your server serves a model under a different name, set `OCR_MODEL` to
match — a mismatch 404s. Custom aliases additionally require `OCR_PROFILE`
so bibr can choose the correct prompt and output normalizer.

Remote OCR URLs must use HTTPS by default. Set `OCR_API_KEY` to send a
bearer token. Plain HTTP to a non-loopback host requires the explicit
`OCR_ALLOW_INSECURE_HTTP=true` override and should only be used on a trusted
private network.

The local `paddle-*` and `glm-*` backends, and the cloud vision OCR providers
(`gemini`/`openai`/`anthropic`), are `bibr chew` options for single-machine
runs; `bibr serve` always expects an external HTTP OCR endpoint. Its runtime
selection is explicit, so serving GLM does not create a silent per-request GLM
fallback for a configured Paddle endpoint.

## Docker deployment

The 0.5.1 public release provides Dockerfiles and Compose configuration for
building locally. Prebuilt GHCR images are not part of this release.
Start from the public release source:

```bash
git clone --branch v0.5.1 --depth 1 https://github.com/scienceverse/bibr.git
cd bibr
```

`docker-compose.yml` defines three services:

| Service | Built from | Purpose | Enabled by |
|---|---|---|---|
| `redis` | `redis:7-alpine` | Response caching | Always (no profile) |
| `bibr-ocr` | `Dockerfile.ocr` | SGLang server for GLM-OCR | `--profile ocr` |
| `bibr-serve` | `Dockerfile.serve` | LitServe API + pipeline orchestrator + GPU models (layout, segmenter) | `--profile serve` |

Redis has no profile, so it always starts; `bibr-serve` and `bibr-ocr` are
opt-in via profile flags. Plain `docker compose up` therefore only starts
Redis — bring up the API and/or OCR sidecar explicitly:

```bash
# All-in-one on a single GPU host
cp .env.example .env      # set your LLM provider + API key
# Also set REDIS_PASSWORD and a 32+ character AUTH_API_KEY.
# Select the bundled GLM service explicitly in .env:
# OCR_BACKEND=glm-http
# OCR_MODEL=glm-ocr
# OCR_PROFILE=glm
# Remove a copied OCR_BASE_URL=http://localhost:8080 so Compose uses bibr-ocr.
docker compose --profile serve --profile ocr up -d --build

# Verify readiness (OCR, classifier artifacts, and enabled Redis cache)
curl http://localhost:8000/ready
```

Compose publishes the API on the host's loopback interface only
(`127.0.0.1:8000`): Docker bypasses host firewalls for ports it publishes on
`0.0.0.0`, and the API speaks plaintext HTTP with a bearer token. Put a
TLS-terminating reverse proxy in front for remote clients, or set
`BIBR_PUBLISH_HOST=0.0.0.0` when you deliberately want the port exposed.
`REDIS_PASSWORD` reaches `bibr-serve` as its own variable and is URL-encoded
into `REDIS_URL` at startup, so passwords containing `@`, `:`, `/`, `?`
or `#` work.

**Split deployment** (bibr-serve and the OCR server on different hosts):
the bundled OCR container has no published host port. Expose it through a
reverse proxy or an explicit Compose override before pointing another host at
it. For example, publish port 8080 only on the GPU host's private interface:

```yaml
# compose.ocr-port.yml — replace the address with your GPU host's private IP
services:
  bibr-ocr:
    ports:
      - "10.0.0.10:8080:8080"
```

```bash
# GPU host — OCR only
docker compose -f docker-compose.yml -f compose.ocr-port.yml --profile ocr up -d

# CPU (or smaller-GPU) host — API, pointed at the GPU host
# The bundled bibr-ocr server speaks plain HTTP; put a TLS-terminating
# reverse proxy in front of it if you need OCR_BASE_URL to be https://.
OCR_BASE_URL=http://10.0.0.10:8080 docker compose --profile serve up -d
```

This HTTP example relies on Compose's explicit
`OCR_ALLOW_INSECURE_HTTP=true` default for a trusted private network. For an
external endpoint, use HTTPS and set `OCR_ALLOW_INSECURE_HTTP=false`. Redis is
also private to the Compose network; plain `docker compose up` does not expose
a Redis port to a host-side `bibr chew` process.

`bibr-ocr` requests an NVIDIA GPU via Compose's device reservations
(`OCR_LOCAL_GPUS`, default `1`, also sets tensor parallelism) and needs
the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
on the host. `bibr-serve`'s own GPU models (layout detector, sentence
segmenter) are lighter; building with `--build-arg WITH_GPU=true` swaps in
`onnxruntime-gpu`, but the compose file only declares a GPU device
reservation for `bibr-ocr` — add one to `bibr-serve` too (or rely on
`nvidia` as the Docker default runtime) if you want its models on GPU as
well. Without a GPU, `bibr-serve` still runs — the layout detector and
segmenter fall back to CPU (see `LAYOUT_USE_GPU` / `SEGMENTER_USE_GPU`
below).

### Environment variables

Set these in `.env` (run `bibr config example --full` or see the
[settings reference](../reference/settings.md) for the full list):

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | LLM backend: `google`, `openai`, `anthropic`, `groq`, or `ollama` | `google` |
| `GOOGLE_API_KEY` | Google AI API key (`google` provider) | — |
| `LLM_API_KEY` | OpenAI-compatible API key (`openai` provider) | — |
| `ANTHROPIC_API_KEY` | Anthropic API key (`anthropic` provider) | — |
| `GROQ_API_KEY` | Groq API key (`groq` provider) | — |
| `OCR_BACKEND` | Which family of server sits at `OCR_BASE_URL`: `paddle-http` (Paddle alias + profile defaults) or `glm-http` (GLM defaults). `bibr serve` always proxies over HTTP; the local `paddle-*`/`glm-*` runtimes are `bibr chew` only | `paddle` (GLM defaults for `bibr serve`; set `paddle-http` for a Paddle endpoint) |
| `OCR_BASE_URL` | External OCR server root URL, without a `/v1` suffix | `http://localhost:8080` (Compose: private `bibr-ocr`) |
| `OCR_MODEL` | Served model name | `glm-ocr`; `paddle-ocr-vl-1.6` under `OCR_BACKEND=paddle-http` |
| `OCR_PROFILE` | `paddle` or `glm`; required for custom aliases | inferred for known names |
| `OCR_API_KEY` | Bearer credential sent to a protected OCR server | — |
| `OCR_ALLOW_INSECURE_HTTP` | Permit non-loopback plain HTTP (private networks only) | `false` (Compose: `true`) |
| `WTPSPLIT_MODEL` | Short wtpsplit name, full Hugging Face repo ID, or existing local bundle directory | `sat-6l-sm` |
| `WTPSPLIT_THRESHOLD` | Optional explicit sentence-boundary threshold in `[0, 1]` | wtpsplit model default |
| `WTPSPLIT_BLOCK_SIZE` | Optional explicit inference block size; set with `WTPSPLIT_STRIDE` | wtpsplit model default |
| `WTPSPLIT_STRIDE` | Optional explicit inference stride; set with `WTPSPLIT_BLOCK_SIZE` | wtpsplit model default |
| `AUTH_API_KEY` | API bearer token; at least 32 characters for network binds or production | — |
| `REDIS_PASSWORD` | Redis password (required when `ENVIRONMENT=production`) | — |
| `ENVIRONMENT` | `development` or `production` | `development` |

### Common commands

```bash
docker compose --profile serve --profile ocr logs -f bibr-serve  # tail logs
docker compose --profile serve --profile ocr up -d --build        # rebuild after code changes
docker compose down                                                # stop everything
docker compose down -v                                             # also clear the Redis cache volume
```

### Custom sentence segmenter

The multilingual `segment-any-text/sat-6l-sm` remains the default. A deployment can opt into
a complete Hugging Face repo ID or an existing local ONNX bundle:

```bash
# Hugging Face deployment bundle
WTPSPLIT_MODEL=scienceverse/bibr-sat-science-en
WTPSPLIT_THRESHOLD=0.47
WTPSPLIT_BLOCK_SIZE=256
WTPSPLIT_STRIDE=128

# Local bundle containing model_optimized.onnx, config.json, and tokenizer assets
WTPSPLIT_MODEL=/models/bibr-sat-science-en
```

Sealed training bundles include `segmenter_manifest.json` with a calibrated `threshold`.
bibr validates its schema, optimized-model identity, every declared checksum, and the exact
bundle file set before loading; unsealed, tampered, path-escaping, or symlinked bundles are
rejected. A legacy manifestless local directory can still be selected explicitly, but it
supplies no manifest settings. Threshold precedence is an explicit Python constructor value,
then `WTPSPLIT_THRESHOLD`, then a validated local manifest. Windowing precedence is the
paired `WTPSPLIT_BLOCK_SIZE`/`WTPSPLIT_STRIDE` settings, then a validated local manifest. If
absent, bibr preserves wtpsplit's model defaults. Remote model IDs do not supply implicit
manifest settings, so set all three values from their release metadata. Every explicit
threshold must be between 0 and 1 inclusive, and stride cannot exceed block size.

To bake a full Hugging Face bundle into the serve image with the same resolver used at
runtime:

```bash
docker build -f Dockerfile.serve \
  --build-arg WTPSPLIT_MODEL=scienceverse/bibr-sat-science-en \
  -t bibr-serve:science .

# Or copy a bundle directory from the Docker build context into the image
docker build -f Dockerfile.serve \
  --build-arg WTPSPLIT_LOCAL_BUNDLE=models/bibr-sat-science-en \
  --build-arg WTPSPLIT_MODEL=/app/segmenter_bundle \
  -t bibr-serve:science-local .
```

`WTPSPLIT_LOCAL_BUNDLE` is a directory relative to the Docker build context. Alternatively,
mount a local bundle read-only at runtime and set its in-container path as
`WTPSPLIT_MODEL`. The v1 scientific bundle is English-only and bibr does not auto-detect
language or switch models per request. Roll back by setting `WTPSPLIT_MODEL=sat-6l-sm` and
removing `WTPSPLIT_THRESHOLD`, `WTPSPLIT_BLOCK_SIZE`, and `WTPSPLIT_STRIDE`.

### Production considerations

- Compose forces `ENVIRONMENT=production`; set `REDIS_PASSWORD`, a strong
  `AUTH_API_KEY`, and non-wildcard `CORS_ORIGINS` (see
  [Authentication](#authentication) above).
- Put a reverse proxy (nginx, Caddy) in front for TLS.
- `/health` (liveness) and `/ready` (readiness) are standard Docker/Kubernetes
  health-check endpoints.

## Reference extraction at scale

Reference segmentation and parsing have two independent controls:

```bash
# .env
REF_SEG_STRATEGY=geom   # "geom" (default) | "region" | "llm" | "crf"
REF_PARSE_STRATEGY=ner  # "ner" (default) | "llm" | "llm-chunked" | "off"
```

By default, segmentation runs on a local geometry model (`geom`), which
cascades to LLM assistance only when its own confidence is low, and
parsing runs on a local ModernBERT-CRF model (`ner`) — zero per-reference
LLM cost in the common case. `REF_PARSE_STRATEGY=llm` trades that cost for
LLM-based field extraction by parsing the bibliography with the
configured LLM in batches (size controlled by
`REF_PARSE_BATCH_SIZE`, default `15`); `REF_PARSE_STRATEGY=off` skips
reference extraction entirely (empty `bib`/`bib_match`/`xref`) while
keeping core metadata. See [Architecture](architecture.md) for how the
segmentation cascade and parsing strategies fit together internally.

Crossref reference enrichment is off by default (`CROSSREF_ENRICH=false`):
requests get the extracted `bib` table with an empty `bib_match`. Set
`CROSSREF_ENRICH=true` to enrich every request, or let callers decide per
request with the `crossref=true|false` form field on `/papers/extract` (and
the `crossref` knob on the MCP `chew_paper`/`chew_url` tools), which
overrides the setting either way. The response cache keys on the effective
value, so an enriched and an unenriched result for the same file never
collide. Deployments that upgraded from a version where enrichment was on
by default must now set `CROSSREF_ENRICH=true` to keep that behaviour.

At volume, Crossref enrichment is rate-limited (`CROSSREF_RATE_LIMIT_RPM`,
default `200`; raise it once you've set `CROSSREF_API_EMAIL` or have an
API key) and can optionally be cached in Redis across requests
(`CROSSREF_REDIS_CACHE`, off by default — falls back to `REDIS_URL`).
Both cache tiers also remember a DOI lookup's 404 for
`CROSSREF_NOT_FOUND_TTL_SECONDS` (default one day; `0` disables), so a
re-run does not spend a request on each DOI Crossref has no record of.
`CROSSREF_CONSOLIDATE` (`off`/`fill`/`replace`, or the `consolidate` form
field on `/papers/extract`) controls whether accepted Crossref matches get
merged back into `bib`, or left only in `bib_match`.

### Optional external resolver (`BIBR_RESOLVER_URL`)

bibr can query the optional bibr-resolver service before falling back to
Crossref, to speed up reference resolution and reduce Crossref load. It's
off by default — leaving `BIBR_RESOLVER_URL` unset keeps bibr's enrichment
behavior unchanged.

| Env var | Default | Meaning |
|---|---|---|
| `BIBR_RESOLVER_URL` | _(unset)_ | Resolver base URL, e.g. `http://resolver-host:2010`. Unset = disabled. |
| `BIBR_RESOLVER_ENRICH` | `true` | Master gate; `false` disables even when a URL is set. |
| `BIBR_RESOLVER_TIMEOUT` | `10.0` | Per-request timeout (seconds). |
| `BIBR_RESOLVER_LIMIT` | `20` | Max candidates requested from `/search`. |
| `BIBR_RESOLVER_SEARCH_CONCURRENCY` | `8` | Max concurrent `/search` calls when prefetching a reference list's title searches. |

A resolver error falls through to Crossref. Misses also fall through by
default; `BIBR_RESOLVER_AUTHORITATIVE=true` makes a clean resolver miss
authoritative and skips the Crossref fallback.

## Hardware sizing

Size the API host separately from the OCR and LLM services. The serve image
includes PyTorch, the layout model, classifiers, and the sentence segmenter;
it is not a torch-free installation even when OCR and LLM calls are remote.
The API can run these local stages on CPU. The bundled SGLang OCR service
requires an NVIDIA GPU. Exact RAM and VRAM requirements depend on the model,
PDF page dimensions, batch sizes, and concurrently processed pages; measure a
representative workload before raising concurrency.

For local CLI runtime and platform choices, see
[Installation](../getting-started/install.md) and
[Quickstart](../getting-started/quickstart.md).

`bibr serve` keeps its GPU models resident for the life of the process —
there's no per-run memory mode like `bibr chew`'s `--memory` flag. On a
box that co-locates the OCR server and is tight on VRAM, force the layout
detector and/or sentence segmenter to CPU independently with
`LAYOUT_USE_GPU=false` / `SEGMENTER_USE_GPU=false` (both auto-detect CUDA
by default).
