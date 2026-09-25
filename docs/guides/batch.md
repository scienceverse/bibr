# Batch runs

`bibr batch` runs a corpus — hundreds to tens of thousands of papers — and
keeps an append-only **ledger** of what happened to each one, so a run can be
interrupted, resumed, audited and summarised. It replaces the ad-hoc campaign
scripts that used to sit next to the repository.

```bash
bibr batch papers/ --out results/                     # local: one warm pipeline
bibr batch manifest.txt --out results/ --refs llm     # a manifest, LLM reference parsing
bibr batch manifest.txt --out results/ \
    --serve-url http://gpu-box:8000 --concurrency 2   # remote: a bibr serve job API
bibr batch manifest.txt --out results/ --dry-run      # the plan, nothing processed
bibr batch report results/                            # summarise the ledger
```

Re-running the **same command** continues where the previous run stopped.

## Inputs

Any mix of:

| Input | Behaviour |
|---|---|
| a manifest text file (any extension bibr does not process: `.txt`, `.lst`, …) | one path per line; blank lines and lines starting with `#` are ignored; relative entries resolve against the manifest's directory; a directory entry is walked recursively |
| a directory | walked recursively for `.pdf`, `.docx`, `.xml`, `.html`/`.htm`, `.epub`, sorted |
| a file | processed as given |

Missing entries are reported (and counted in the dry run) but never abort the
run. Every file is identified by its **`paper_id`** — the file stem — and its
export is written to `<out>/<paper_id>.json`. Two inputs that share a stem
(case-insensitively) would overwrite each other, so colliding files get
`<stem>-<sha256[:8]>` instead, and so does a file named `run_info`, whose export
would replace the run's own `run_info.json`; the mapping is printed by
`--dry-run` and recorded under `collisions` in `run_info.json`, and every
ledger line carries the original `stem`. A file keeps the id the ledger
recorded for its path, so a corpus can grow: adding a second `paper.pdf` from
another directory gives only the newcomer a suffix and never renames, or
re-processes, the paper already done.

Selection flags: `--limit N` processes at most N papers this run (after
resume filtering), `--shuffle` randomises the order (`--seed` makes it
reproducible; the seed used is recorded in `run_info.json`), `--deadline`
(ISO-8601 or epoch seconds) stops *submitting* new papers after that time —
in-flight papers finish, and the next run picks up the rest.

## Resumability

The ledger's latest line per `paper_id` decides what a new run does with it:

| latest line | default | `--retry-failed` | `--force` |
|---|---|---|---|
| none | run | run | run |
| `ok` | skip | skip | run |
| `failed` | skip | run | run |
| `failed` before the paper got a verdict: `error_code` `interrupted`, `http_401` or `http_403` | run | run | run |
| `failed` on a crash or a service outage: `error_code` `chunk_error` or `upstream_unavailable`, or `transient_exhausted: true` | run until the paper has failed this way 3 times, then skip | run | run |

A paper the run was interrupted on (Ctrl-C), or whose upload the serve refused
because of the token, never got a verdict, so it always runs again. A crash or
a service outage — the OCR or LLM server unreachable or unable to start, a
serve that kept answering 502/503 until the retries ran out — is usually the
machine's or the service's, so it runs again too. It can also be the paper's
own: a bug its content triggers, a prompt that brings the LLM server down, a
model reply the serve reports as a 502. So a paper that has failed this way
three times since its last success waits for `--retry-failed`, and a batch
still finishes. A timeout is in neither group: a paper can be too slow on its
own, and re-running it by default would never finish.

Every attempt appends a new line with an incremented `attempt` counter —
nothing is ever rewritten, so `outcomes.jsonl` is a full history. A run killed
mid-write (out of memory, a full disk) can leave a torn last line; it is
skipped with a warning and the next run starts on a fresh line.

Ctrl-C is graceful in both executors: the local executor records the chunk
that was running as `interrupted`; the remote executor stops submitting,
waits up to 30 s for in-flight jobs and records the rest as `interrupted`.
A second Ctrl-C exits immediately. The process exits 130 after an interrupt,
1 when any paper failed, 0 otherwise.

## Local vs remote

**Local** (the default) builds one warm pipeline (`bibr.api.Chewer`) and feeds
it chunks of `--batch-size` files (auto-sized from the memory mode like
`bibr chew`). Inside a chunk the pipeline is stage-major — every file's
layout, then every file's OCR, and so on — so models load once and OCR batches
fill up. Ledger lines are written as each chunk finishes. All of `bibr chew`'s
pipeline flags apply: `--ocr`, `--ocr-url`, `--llm`, `--refs`, `--ref-seg`,
`--consolidate`, `--no-crossref`, `--no-llm`, `--pages`, `--memory`,
`--device`, `--figure-images`, `--include-regions`, `--preset`, …

**Remote** (`--serve-url URL`) submits papers to a
[`bibr serve`](deployment.md) async job API (`POST /papers/jobs`, poll
`GET /papers/jobs/{id}`, fetch `…/result`). The bearer token comes from
`--token`, else `AUTH_API_KEY` / `BIBR_SERVE_TOKEN` in the environment, else
`AUTH_API_KEY` in bibr's `.env`. The run waits for `GET /ready` first (up to
`--ready-timeout`) and records the serve's `build_sha` in every ledger line.
The options the job API accepts are passed through — `--refs`, `--ref-seg`,
`--consolidate`, `--pages` (as `start_page`/`end_page`), `--figure-images`,
`--include-regions` — plus arbitrary `--form k=v` fields; the pipeline flags
that configure a *local* pipeline are ignored with a warning (the serve's own
settings apply).

Concurrency adapts to the serve:

- `--concurrency` jobs start in flight (default 2).
- A **429** on submit is the serve's queue cap (`JOBS_MAX_ACTIVE`) — normal
  under load. In-flight drops to `--min-concurrency` and the submit waits
  `Retry-After`; it is never counted as a failure.
- **502/503/504** answers to a submit or poll, connection errors, and a job
  the serve failed because of an upstream OCR/LLM outage (circuit breaker open,
  OCR server unreachable, …) are *transient*: in-flight shrinks by one and the
  paper is retried with backoff up to `--retries` times (default 3). If it
  never recovers, the last transient code is recorded, and the next run picks
  the paper up again.
- Every success grows in-flight by one, back toward `--max-concurrency`.
- A job the serve failed with **504** ran out of the serve's
  `PIPELINE_TIMEOUT`: the serve is up and the paper was too slow. It is
  submitted once more (the serve's OCR and LLM are shared with other jobs)
  without shrinking in-flight, then recorded as `pipeline_timeout`.
- Other 4xx answers are the paper's own problem — recorded once, no retry. A
  401/403 stops the whole run; the papers it failed are picked up again once
  the token is fixed.
- `--poll-timeout` (default 2400 s) bounds one paper's wall clock; expiry is
  recorded as `poll_timeout` without a retry.

The serve dispatches at most `JOBS_MAX_RUNNING` jobs at once, so a client
in-flight much above that only lengthens the queue.

## Output layout

```
<out>/
  <paper_id>.json     the export, one per successful paper
  tables/*.parquet    every successful paper as one Parquet file per table
  outcomes.jsonl      the ledger — one JSON object per attempt
  run_info.json       the latest run: options, executor, redacted settings, counts
  runs.jsonl          run_info of every run, appended
```

Each export's `paper_id` is the batch's own id, the name of its JSON file, so it
is unique across the corpus even when papers share a DOI. `tables/` is rebuilt
from every paper whose latest attempt is `ok` at the end of each run
(`--no-tables` skips it); see the
[Python guide](library.md#corpus-tables-parquet) for its layout. An export of
another schema major, left in `<out>` by an older bibr, is left out of the
tables with a warning; `--force` re-runs every paper under the current schema.

`run_info.json` carries the `run_id` that stamps this run's ledger lines,
`started_at`/`finished_at`, the invocation (`options`, without the token),
`bibr_version`, `build_sha` (the serve's for remote runs, the checkout's git
head or `BIBR_BUILD_SHA` locally), the resume counts, the shuffle seed, the
collision map, and `settings` — every non-default bibr setting as
`bibr config show` renders it, secrets masked.

## Ledger schema

One JSON object per line of `outcomes.jsonl`:

| Field | Type | Meaning |
|---|---|---|
| `paper_id` | str | export name (`<out>/<paper_id>.json`); the file stem, sha-suffixed on collisions |
| `stem` | str | the original file stem |
| `path` | str | input path as given |
| `sha256`, `bytes` | str, int | identity and size of the input |
| `status` | `ok` / `failed` | |
| `error_code` | str / null | see below |
| `failed_stage` | str / null | pipeline stage that failed, when known |
| `error` | str / null | error text, clipped to 800 characters |
| `started_at`, `finished_at` | ISO-8601 UTC | for local runs the chunk's start/end |
| `duration_s` | float | remote: submit-to-result wall clock; local: the export's own pipeline time, else the chunk's |
| `pipeline_seconds` | float / null | `extraction.timings.total_seconds` from the export |
| `stage_times` | {stage: seconds} / null | `extraction.timings.stages` from the export |
| `llm_tokens`, `llm_input_tokens`, `llm_output_tokens` | int | from `extraction.usage.totals` (legacy: `llm_usage`) |
| `n_refs`, `n_matched` | int | `bib` rows and accepted `bib_match` rows (falls back to `extraction.enrichment.refs_enriched`) |
| `n_sentences` | int | `text` rows |
| `warnings` | `{count, first, codes}` | `extraction.warnings` (legacy: `processing_warnings`): total, the first three (as `CODE: message`), and a frequency map by warning code |
| `n_validation_errors`, `n_validation_warnings` | int | from the export's `validation` block |
| `bibr_version`, `build_sha` | str | producing bibr; remote runs record the serve's build |
| `executor` | `local` / `remote` | |
| `attempt` | int | 1 for the first line of this paper, +1 per further attempt |
| `run_id` | str | the run that wrote the line (matches `run_info.json`) |
| `job_id`, `retries`, `http_status`, `transient_exhausted` | remote only | serve job id, transient retries used, the failure's HTTP status |

`error_code` values: locally, the pipeline's own code (`ChewFailure.error_code`, e.g.
an OCR or reference-parse code) or `processing_error`, `upstream_unavailable`
(an OCR or LLM service was down or could not start: `ChewFailure.outage`),
`chunk_error` (the pipeline crashed on the paper: a chunk that crashes runs its
papers again one by one, so only a paper that crashes on its own gets it, or
the whole chunk when the pipeline could not run at all), `interrupted`;
remotely, the serve's `error_code` when it gave one, else `http_<status>`,
`connection_error`, `upstream_unavailable`, `job_lost`,
`submit_wait_exhausted`, `pipeline_timeout`, `poll_timeout`,
`bad_submit_response`, `bad_result_json`, `client_error`, `unreadable_input`,
`interrupted`.

## Reading the report

`bibr batch report <out>` (or `--report`, and `--json` for the same as JSON)
prints one compact table; every run ends with the same table for its own
lines:

```
bibr batch report · results/
  papers       412 ok · 9 failed · 421 total (430 attempts)
  window       2026-09-02T10:00:00+00:00 → 2026-09-03T04:12:31+00:00 (18.21 h)
  throughput   22.6 papers/h
  latency      p50 128.4s · p90 301.0s · max 812.7s (n=421)
  stage share  ocr 63% · extract 24% · enrich 8% · parse 3% · layout 2%
  llm tokens   41,220,118 total · 100,049 / paper
  references   19,870 refs · 17,102 matched (86%) · 48.2 / paper
  failures     poll_timeout ×5 · http_413 ×3 · OCR_EMPTY ×1 | stage: ocr ×1
  warnings     140 papers · OCR_REGION_FAILED ×212 · STATEMENT_LEXICAL_FALLBACK ×31
```

- **papers** counts the *latest* attempt per paper; **attempts** is the raw
  line count (a resumed run's earlier failures are history, not state).
- **throughput** is successful papers over the window from the earliest
  `started_at` to the latest `finished_at` — for a directory that holds
  several runs, that window spans all of them.
- **latency** percentiles are over successful attempts' `duration_s`.
- **stage share** is each stage's share of the *summed* stage time across
  successful attempts — where the pipeline spends its time, not a per-paper
  mean.
- **references** is the corpus match rate (`n_matched / n_refs`).
- **failures** groups the latest-failed papers by `error_code` and, when the
  export reported it, by `failed_stage`.
- **warnings** lists the ten most frequent warning codes
  (`extraction.warnings[].code`). An export older than 12.0 has prose warnings,
  counted by the text before the first colon (`VALIDATION:<severity>:<CODE>`
  for validation findings).

The ledger is plain JSONL, so anything else is one `pandas.read_json(...,
lines=True)` away.
