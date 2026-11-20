# Configuration

bibr works with nothing configured. This page shows the few settings most
people change, how to pick a mode, and the options for OCR, LLMs, references
and presets. The generated [Settings reference](../reference/settings.md) lists
every variable and its default.

## Start here

bibr reads its settings from environment variables and from `.env` files
(`./.env` and `~/.bibr/.env`); a command-line flag such as `--fast` or
`--ocr`, a saved preset (`--preset`) or a Python API argument overrides a
setting for one run. With nothing set, bibr runs in
**fast mode**: local models only, no LLM and no API key, and PDFs are read from
their text layer. `bibr setup` is optional: it writes a `.env` that adds an
LLM (full mode), OCR for scanned PDFs, or both.

These are the settings most people touch:

| Setting | Default | What it does |
|---|---|---|
| `PIPELINE_MODE` | `auto` | `fast` never calls an LLM; `full` uses the configured LLM; `auto` is full when an LLM is configured and fast otherwise. `--fast` / `--full` choose for one run. |
| `LLM_PROVIDER` | `{{ default_llm_provider }}` | The LLM provider full mode calls: `google`, `openai`, `anthropic`, `groq` or `ollama`. |
| API key | none | `google`: `GOOGLE_API_KEY` (also read as `GEMINI_API_KEY`). `anthropic`: `ANTHROPIC_API_KEY`. `groq`: `GROQ_API_KEY`. `openai`: `LLM_API_KEY`. `ollama`: no key. `LLM_API_KEY` works for every provider that needs a key. |
| `LLM_MODEL` | follows the provider | The model to call; see [Choosing an LLM](#choosing-an-llm). |
| `OCR_BACKEND` | not set | How PDF pages are read. Not set: the text layer in fast mode, the automatic `paddle` selector in full mode. `--ocr` chooses for one run; see [OCR backends](#ocr-backends). |
| `CROSSREF_ENRICH` | `false` | `true` looks each reference up in Crossref. `--crossref` / `--no-crossref` choose for one run. |
| `CROSSREF_API_EMAIL` | not set | Your e-mail address, sent to Crossref with enrichment lookups for its "polite pool". |

To see what is in effect, run `bibr config show` (settings that differ from
the defaults; `--all` for every one, `--sources` for where each value came
from). `bibr config set KEY VALUE` writes one setting to `.env`, and
`bibr chew paper.pdf --dry-run` shows the mode, OCR and LLM a run would use
without processing anything. In a uv project (the Quickstart install or a
source checkout), prefix the commands on this page with `uv run`.

## Pick a mode

A **mode** decides whether an LLM is called. Fast mode uses bibr's local models
alone. Full mode adds an LLM, which reads the parts of a paper the local
models are unsure of.

| You want | What to set | What it needs | What leaves your machine |
|---|---|---|---|
| **Fast** (the default) | Nothing, or `PIPELINE_MODE=fast` / `--fast` | Nothing more than the install. PDFs need a text layer; a scanned PDF needs an [OCR backend](#ocr-backends). | Nothing from the paper. The first run downloads bibr's models from Hugging Face. |
| **Full, cloud LLM** | `LLM_PROVIDER` and its API key, or run `bibr setup` | A provider account and key. With no OCR setting, full mode starts the automatic `paddle` OCR, which needs an NVIDIA GPU, llama.cpp or Apple Silicon; on other machines set `OCR_BACKEND=text-layer` or a cloud OCR backend. | The paper's text goes to the LLM provider; with cloud OCR, page images go to the OCR provider. |
| **Full, local LLM** | `bibr setup`, or `--llm local` | An NVIDIA GPU with 5 GB or more, or Apple Silicon. On Windows and on cards below 10 GB, also llama.cpp, which you install. See [Choosing an extraction model](extraction-models.md). | Nothing from the paper: the model servers run on this machine. |
| **Private server** | `LLM_PROVIDER=openai`, `LLM_BASE_URL` and `LLM_MODEL` for the LLM; `OCR_BACKEND=paddle-http` (or `glm-http` for a GLM-OCR server) and `OCR_BASE_URL`, or `--ocr-url`, for OCR | A GPU machine you run, such as the [Docker OCR/API stack](deployment.md#docker-deployment) | Page images and text, to your server only. |

`bibr setup` detects your hardware and asks which of these you want; every
run of it writes `PIPELINE_MODE` (`fast` or `full`). Run
`bibr setup --advanced` for exact control over the LLM provider, OCR backend,
Crossref settings, cache extras and model choices.

A `GOOGLE_API_KEY`, `GEMINI_API_KEY` or `LLM_API_KEY` already in your
environment counts as a configured LLM, and so does setting `LLM_PROVIDER`,
`LLM_MODEL` or `LLM_BASE_URL`: with `PIPELINE_MODE=auto`, such a run is in
full mode. `--fast` or `PIPELINE_MODE=fast` keeps every run in fast mode.

`--no-llm` is not a fast mode. It keeps sections and text but drops authors,
references, citation links and equations for PDF and DOCX input, and it still
runs OCR on PDFs, so it needs an OCR backend. To avoid every LLM call, use
`--fast`.

## What leaves your machine

In fast mode with no OCR setting, nothing from a paper leaves your machine.
A cloud LLM receives the paper's text, a cloud OCR backend receives page
images, and Crossref enrichment (off by default) sends each reference's DOI,
title and author names. [What leaves your machine](privacy.md) lists every
route, including model downloads and the Google-key caveat above.

## OCR backends

**OCR** (optical character recognition) reads text from page images. With no
OCR setting, a fast run reads each PDF from its embedded text layer
(`text-layer`) and starts no OCR model. A full run, or a run under
`--no-llm`, uses `paddle`, the automatic selector. Naming a route (`--ocr`,
`--ocr-url`, or `OCR_BACKEND`, `OCR_MODEL` or `OCR_PROFILE`) keeps that route
in either mode. `OCR_BASE_URL` alone names no route. `bibr doctor` shows what
this machine can run.

| Request or runtime | When it's used |
|---|---|
| `text-layer` | Default in fast mode with no OCR setting; `--ocr text-layer` or `OCR_BACKEND=text-layer` on any run. Runs no model and needs no GPU. Reads born-digital PDFs only: a scanned page stays unread, and a file with fewer than half of its text regions readable stops with `scan_needs_ocr`. See [Known limitations](../limitations.md#the-text-layer-route-reads-born-digital-pdfs-only). |
| `paddle` | Automatic selector, the default in full mode. Linux x86_64 with at least 8 GB detected NVIDIA VRAM: `paddle-vllm`, then `glm-llama`; otherwise `glm-llama`. Apple Silicon: `paddle-rapid-mlx`, `paddle-mlx-vlm`, `glm-rapid-mlx`, then `glm-llama`. Windows: `glm-llama`. |
| `paddle-vllm` | Explicit managed PaddleOCR-VL-1.6 vLLM runtime on Linux/CUDA. |
| `paddle-rapid-mlx` / `paddle-mlx-vlm` | Explicit PaddleOCR-VL-1.6 Apple-Silicon runtimes. Both use the quantized `olragon/PaddleOCR-VL-1.6-8bit` default. |
| `paddle-http` | Explicit external Paddle server, addressed by the served alias `paddle-ocr-vl-1.6`. |
| `glm-rapid-mlx`, `glm-llama`, `glm-http` | Explicit GLM-OCR compatibility choices. They are also the ordered startup fallback candidates for `paddle` where listed above. The older `glm-mlx` (vllm-mlx) backend is disabled and refuses to start; use `glm-rapid-mlx`. |
| `gemini` / `openai` / `anthropic` | Vision-LLM OCR via a cloud provider — no local GPU needed, useful for a laptop or a low-resource box. Page images leave your machine. |

`OCR_FALLBACK_TO_TEXT_LAYER` decides what happens when no runtime of the
automatic `paddle` selector can start on the machine (no NVIDIA GPU for vLLM,
no llama.cpp, no Apple Silicon for MLX; checked without starting one). `off`
(the default) stops with an error. `auto` warns once, reads born-digital PDFs
from their text layer as `text-layer` does, and marks each export
`OCR_BACKEND_UNAVAILABLE`; scanned PDFs then fail, since they need OCR. A
runtime that is installed but fails to start is still an error.

The `paddle` selector's fallback between candidates is **startup-only**: a
candidate must construct and pass its
readiness check before it is selected. Once one has started, bibr records its
identity and does not silently send failed individual OCR requests to GLM.
Choose a concrete backend with `--ocr <name>` (or `OCR_BACKEND` in `.env`).
`--ocr-url` (and the library's `ocr_url`) routes to an external Paddle
endpoint (`paddle-http`) by default; specify `--ocr glm-http` (or any `glm`
backend, or set `OCR_BACKEND` to one) for a GLM endpoint. An explicit
`--ocr gemini`, `openai` or `anthropic` ignores the URL with a warning. A
cloud vision `OCR_BACKEND` does not override a URL you pass: the run uses
that server as `paddle-http` and sends no page image to the cloud provider.
The cloud vision backends call the model in `OCR_VISION_MODEL`. Its default
(`gemini-3-flash-preview`) is a Gemini model. When the setting is unset,
`--ocr anthropic` uses `claude-haiku-5-5` instead; with `--ocr openai` set it to
one of OpenAI's vision models. A cloud OCR backend with no API key stops the run
before any model loads.
For example:

```bash
bibr chew paper.pdf --ocr paddle-http --ocr-url https://ocr.example.org
bibr chew paper.pdf --ocr glm-http --ocr-url https://ocr.example.org
```

Each cloud vision OCR call asks for at most `OCR_VISION_MAX_TOKENS` output
tokens (16384 by default). With `anthropic`, a region asks for one token per 32
pixels of its crop at 200 DPI (fewer pixels per token at a lower `LAYOUT_DPI`),
at least 2048 and never more than the setting, so a model that loops on a small
region stops sooner. Gemini and the OpenAI backend keep the full setting:
Gemini's thinking tokens and an OpenAI-compatible reasoning model's reasoning
tokens count against the same limit.

Remote OCR requires HTTPS by default. For a trusted private network without
TLS, explicitly set `OCR_ALLOW_INSECURE_HTTP=true`; loopback HTTP is allowed.

### Model and profile overrides

The automatic Paddle-first selector is:

```bash
# .env
OCR_BACKEND=paddle
```

Do not combine this automatic selector with global `OCR_MODEL` or
`OCR_PROFILE`: those overrides could pin the Paddle profile/model while the
Linux fallback starts GLM. `OCR_MODEL` is a model/served-name override for a
concrete backend. Known Paddle and GLM names infer their request prompt and
normalizer, but a custom alias is intentionally not guessed: set
`OCR_PROFILE=paddle` or `OCR_PROFILE=glm` with it. An explicit Paddle HTTP
endpoint uses all three settings:

```bash
OCR_BACKEND=paddle-http
OCR_MODEL=paddle-ocr-vl-1.6
OCR_PROFILE=paddle
OCR_BASE_URL=https://ocr.example.org
```

For example, an external Paddle server with a private served name must declare
both its model and profile:

```bash
OCR_BACKEND=paddle-http
OCR_MODEL=team-science-ocr
OCR_PROFILE=paddle
```

Profiles also preserve output semantics. The Paddle profile decodes OTSL table
markers (`<fcel>`, `<lcel>`, `<nl>`, `<ecel>`) into HTML and removes one outer
Markdown/LaTeX fence or balanced display wrapper from formulas. The GLM profile
leaves the output as it is. Both retain the unmodified model response as
`raw_ocr_content` in the opt-in `extraction.regions` diagnostic payload. The selected backend, model, and
profile are included in `extraction.ocr` and in the OCR-cache identity, so a
cache hit never crosses a Paddle/GLM or normalizer boundary.

## Choosing an LLM

An LLM is used only in full mode. With no LLM configured, a run is in fast mode
and needs no key.

### Cloud providers

`LLM_PROVIDER` selects the provider used for metadata extraction and LLM
fallbacks such as section classification and citation linking: `{{ default_llm_provider }}`
(the default), `openai`, `anthropic`, `groq`, or `ollama` (a local Ollama
server). `LLM_MODEL` picks the model for that provider. Left unset, it
follows the provider: `{{ default_llm_model }}` for `google`,
`claude-haiku-5-5` for `anthropic`, `gpt-5-nano` for `openai`,
`llama-3.3-70b-versatile` for `groq` and `gpt-oss:20b` for `ollama`, the same
models `bibr setup` writes. An `openai` provider pointed at your own server
with `LLM_BASE_URL` keeps `{{ default_llm_model }}`, so name the served model
there.

The key each provider reads is in [Start here](#start-here).

### Ollama and your own server

For Ollama, `LLM_OLLAMA_BASE_URL` names the server (default
`http://localhost:11434`). bibr talks to Ollama's OpenAI-compatible API under
`/v1`, so both `http://localhost:11434` and `http://localhost:11434/v1` work.

`LLM_BACKEND=cloud` is the default: bibr uses the configured provider/endpoint.
That endpoint can also be your own OpenAI-compatible server:

```bash
LLM_BACKEND=cloud
LLM_PROVIDER=openai
LLM_BASE_URL=http://gpu-host:8000/v1
LLM_MODEL=your-served-model
LLM_API_KEY=your-endpoint-key
```

Before the first prompt, bibr asks such a server, or the Ollama server, what
it is. An Ollama or llama.cpp server runs one fixed context and rejects or
cuts a longer prompt, so bibr reads its window (Ollama's `/api/ps` or the
model's `num_ctx`, llama.cpp's `/props`), cuts long inputs to fit it, and
applies the caps of the managed llama.cpp server, scaled to the window: at
16,384 tokens `LLM_MAX_INPUT_CHARS=24000`, `LLM_REF_SEG_WINDOW_CHARS=12000`
and `LLM_MAX_TOKENS=4096`. An Ollama model that is not loaded yet and sets no
`num_ctx` gets the 16,384-token caps, and bibr asks again on a later call
(after 30 seconds, then after twice as long each time, up to once every 10
minutes) until Ollama gives its window. When an Ollama or llama.cpp server
rejects a request as too long and names its window, the caps move to that
window too. A vLLM server is not sized in advance: bibr learns its window
when the server rejects a request as too long. A call whose window leaves
fewer than 256 tokens for the paper text, when that text would have to be
cut, is not sent: its fields are marked failed, or the step falls back where
it has one (the NER parser for references, the positional heuristic for the
abstract; headers keep their alias prior).

With `LLM_CONTEXT_TOKENS` you set the window yourself, and long inputs are
cut to fit it on any server. For an Ollama or llama.cpp server it also wins
over the window the server reports, and the caps above follow it; set it when
the server runs a smaller context than it reports. Caps you set yourself
(`LLM_MAX_INPUT_CHARS`, `LLM_MAX_TOKENS`, `LLM_REF_SEG_WINDOW_CHARS`) are
kept, with a warning when they exceed the cap for a known window.

An `LLM_BASE_URL` (or `OCR_VISION_BASE_URL`) that receives an API key may use
plain `http://` for loopback and private-network hosts — a local vLLM or
Ollama, a GPU box on the LAN or tailnet. For a public host bibr refuses to
send the key over plain HTTP unless `LLM_ALLOW_INSECURE_HTTP=true`; the
pipeline, `bibr setup` and `bibr doctor` apply the same rule. The vision
endpoint receives the LLM provider's key, so the LLM setting governs it too,
not `OCR_ALLOW_INSECURE_HTTP`.

### Local models

`--llm local` starts a managed server. It resolves to Rapid-MLX on Apple
Silicon when that executable is available, otherwise vllm-mlx; llama.cpp on
Windows, CUDA cards below 10 GB and CUDA cards older than Ampere (compute
capability below 8.0); and vLLM on other Linux/CUDA systems with at least
10 GB (or when VRAM detection is unavailable). With the same card it picks the
model and the OCR that `bibr setup` writes and the memory mode it shows, with
one exception on 10–11 GB cards (see
[choosing an extraction model](extraction-models.md#on-10-and-11-gb-cards)).
Explicit choices are `vllm`, `vllm-mlx`, `rapid-mlx`, `llama-cpp`, and `llmster`.
The OCR and LLM choices are independent, so local OCR with a cloud LLM is a
supported hybrid configuration.

Run `bibr setup` to choose the model as well as the runtime.
[Choosing an extraction model](extraction-models.md) lists the model for each
card, the memory it needs and how it scores, and the hosted model setup
recommends when no local model fits. `LLM_LOCAL_MODEL` selects managed local
weights; `LLM_MODEL` and `--llm-model` select the provider model.

[Local model runtime](model-runtime.md) covers how bibr shares GPU memory
between its servers (`LLM_LOCAL_MEM_FRACTION`, `LLM_LOCAL_MIN_VRAM_GB`),
pinned model revisions, speculative decoding (`LLM_VLLM_SPEC_DECODE`), the
managed llama.cpp flags (`LLM_LLAMA_CPP_EXTRA_ARGS`) and how managed servers
start and stop.

For `--llm llmster`, install LM Studio's `lms` CLI and download the desired model
first, then set `LLM_LLMSTER_MODEL` to its model key. Bibr can start its daemon
and API server and load that existing model. It does not install the runtime
or download a model, and cleans up only resources it started.

Local inference speed depends on the runtime, model, hardware, and document.
Validate the fields you need on representative papers before choosing a model
for a large run. Cloud LLMs or an external OpenAI-compatible server can be used
with local OCR when local LLM throughput is insufficient.

### Structured output and pointer answers

`LLM_STRUCTURED_BACKEND=auto` uses Instructor. The experimental
`nuextract-native` backend, for a NuExtract server you run yourself, is
explicit-only and needs `pip install 'bibr[nuextract]'`, because bibr no longer
installs numind by default.

`LLM_POINTER_OUTPUTS=true` is experimental and off by default. It shortens the
answers of the two front-matter calls; see
[choosing an extraction model](extraction-models.md#pointer-answers-llm_pointer_outputs)
for what it does and why it stays off.

### Rate limits, retries and caching

`LLM_RATE_LIMIT_RPM` defaults to 60 requests per minute to stay inside a cloud
provider's quota. For a server bibr starts itself, and for an `LLM_BASE_URL` on
loopback, a private network or a tailnet, bibr uses 600 instead, so the server's
own queue and `LLM_MAX_CONCURRENCY` set the pace. The `ollama` provider gets 10.
Setting `LLM_RATE_LIMIT_RPM` yourself overrides all of these.

`PIPELINE_MAX_CONCURRENT_POST_PARSE` sets how many papers of a chunk are in
extraction at once. It defaults to 4. For a vLLM server bibr starts itself
(the `vllm` runtime of `--llm local`) bibr uses 16, because that server
answers many requests in one batch. Setting the variable yourself overrides
both; lower it if a slow GPU runs into LLM request timeouts.

The provider SDK sends each LLM request once and gives it up after twice
`LLM_TIMEOUT_SECONDS`. bibr retries rate limits (429), server errors (5xx),
request timeouts and conflicts (408, 409) and dropped connections itself, up
to three attempts per call, so every retry waits for `LLM_RATE_LIMIT_RPM`; a
`Retry-After` hint is honoured up to 60 seconds. An attempt that fails with a
server error, a 408, a timeout or a dropped connection counts toward the
circuit breaker; a 429 or 409, like any other 4xx answer, does not, since the
provider is up. The `anthropic` provider does
not stream, so it caps the output of one call at 16,384 tokens (8,192 for
Claude Opus 4 and 4.1) even when `LLM_MAX_TOKENS` is higher. A
`LLM_THINKING_BUDGET` above that cap less 1,024 tokens leaves too little room
for the answer, so calls are then sent without thinking and a warning is
logged. Claude Opus 4.7 and 4.8, Haiku 5.5 and the Sonnet 5, Opus 5, Fable 5
and Mythos 5 families (5.5, 5.1 and 5.2 included) reject any `temperature`, so
calls to them carry none and the model samples at its own default; the same
paper can then come back slightly different between runs. They take adaptive
thinking instead of a token budget, which bibr does not send yet, so
`LLM_THINKING_BUDGET` is ignored for them and a warning is logged. Claude
Sonnet 5.5, Opus 5.5 and Fable and Mythos 5.1 and 5.2 also reject a forced
tool call, so bibr offers them its one answer tool without forcing it; an
answer that comes back without the tool call is retried like any other
invalid answer. The
`anthropic` provider needs the anthropic SDK below 1.0, which the `anthropic`
extra installs (`uv add 'bibr[anthropic]'`; `bibr setup` installs it when you pick
Anthropic). Without it, or with 1.0 or later, bibr stops before the first call
with an error that names the fix. The `google` provider likewise needs the
`google` extra (google-genai).

`CACHE_LLM=true` keeps structured LLM responses on disk (`CACHE_LLM_DIR`,
default `~/.cache/bibr/llm`), readable by your user only. An entry is reused
only for the same provider, endpoint, model, call parameters (temperature,
output cap, reasoning effort, thinking budget) and prompt, so changing any of
them calls the model again; for `--llm llmster`, so does changing
`LLM_LLMSTER_MODEL`. Another endpoint that serves different weights under the
same model name and URL cannot be told apart: clear the directory after
swapping the model behind it.

## References

### Crossref enrichment

Reference enrichment itself is opt-in: `CROSSREF_ENRICH` defaults to `false`,
so a plain `bibr chew` (or `POST /papers/extract`) never calls Crossref or the
resolver and the `bib_match` table stays empty. Set `CROSSREF_ENRICH=true` to
enrich by default, or switch it per run — `bibr chew --crossref` /
`--no-crossref`, `bibr.chew(..., crossref=True)`, or the `crossref=true|false`
form field on `/papers/extract` — which wins over the setting either way.
Matches stay in `bib_match` and do not become values of the references in
`bib`, unless you opt in with `CROSSREF_CONSOLIDATE=fill` or `replace` (off by
default). With enrichment on, `CROSSREF_VERIFY` (default `true`) checks each
reference's parsed volume, issue and pages against its matched record and
reports disagreements; it fills or corrects one of those values only when
the record's value is printed in the reference string, so a value the paper
did not print is never added. The optional resolver's
`BIBR_RESOLVER_RAW_SEARCH` (default `true`) only finds matches: it searches
again with the whole reference string for references the title searches
left unmatched; see
[Reference extraction at scale](deployment.md#reference-extraction-at-scale).

### Reference-extraction strategies

Reference parsing has its own strategy knob, `--refs` (or `REF_PARSE_STRATEGY`):

- **`ner`** (default) — the default `geom` segmentation locates each
  reference with a local geometry model, cascading through layout-region
  anchors, LLM segmentation, and CRF when earlier tiers cannot resolve it; a
  segmentation of the section's printed lines replaces a fallback,
  under-yielding or clearly weaker result. A local encoder-CRF model parses
  the resulting entries. The core install
  supports geometry and ONNX parsing. When a dedicated fine-tuned endpoint is
  configured, flagged entries receive bounded model review with NER retention
  on failed or weaker results. Without that endpoint, parsing is local.
  `REF_SEG_STRATEGY=crf` requires the `torch` extra.
- **`llm`** — parses references with the configured LLM in batches of up to
  `REF_PARSE_BATCH_SIZE` entries (default 15), with NER recovery for failed
  batches when available.
- **`llm-chunked`** (experimental) — lets the LLM find reference boundaries and
  fields inside region-aligned chunks instead of requiring one entry per input.
- **`ref-llm`** — sends one printed reference at a time to a
  dedicated fine-tuned Qwen3.5-0.8B server using the `ref-llm-sft@1` prompt
  and 14-field JSON contract. It retains the entry's language and year suffix.
  Invalid, truncated, unsupported or timed-out responses recover through NER
  in their original positions; successful model entries survive a recovery
  failure. This endpoint is configured separately from the metadata LLM.
- **`off`** — skips reference extraction entirely (empty bibliography/match/
  citation-link tables) while keeping everything else — titles, authors,
  sections, equations.

`--ref-seg` (or `REF_SEG_STRATEGY`) overrides just the segmentation step
independently of parsing. The full set of strategies and how they cascade is
covered in [Architecture](architecture.md).

With `ner`, the same parser also reads the citation a paper prints of itself
on its first page ("Citation: ...", "To cite this article: ...") and fills
the journal, volume, issue, pages and publication year the extraction left
empty, and the title and authors when it found none, once the line is shown
to be about this paper (same DOI, else same title, else a matching first
author, else a label that names this article). It never replaces a value;
`extraction.diagnostics.self_citation` says what it did. Set
`SELF_CITATION_METADATA=false` to turn it off. The other strategies, and
`--no-llm`, do not read the line.

### Dedicated fine-tuned reference parser

To enable automatic NER-to-Qwen review, configure both the dedicated endpoint
and the trained adapter alias:

```dotenv
REF_PARSE_STRATEGY=ner
REF_LLM_BASE_URL=http://127.0.0.1:30210/v1
REF_LLM_MODEL=bibr-ref-qwen35-08b-balanced
REF_NER_LLM_FALLBACK=true
```

Review is on by default when that endpoint and alias are configured. Failed or
weaker reviews retain NER results. Set `REF_NER_LLM_FALLBACK=false` for local NER
only, or select `--refs ref-llm` to parse all entries with the dedicated model and
aligned NER recovery. This server is separate from the metadata LLM.

See [Fine-tuned reference model](reference-model.md) for the pinned adapter
download, serving command, request limits and export diagnostics. The base
Qwen3.5-0.8B checkpoint alone is not the fine-tuned parser. Routing checks are
observable source signals, not calibrated confidence.

## Pages and memory

`--pages` (`pages=` in the Python API) processes only part of a PDF. Pages are
1-based: `--pages 1-5` reads the first five pages, `--pages 3` only the third.

`PIPELINE_MAX_PAGES` (default 200) caps how many pages of one PDF bibr
processes, counted from the first page it processes: with no `--pages`, pages
1 to 200; with `--pages 150-400`, pages 150 to 349. Pages past the cap are
dropped without a warning in the console or the export. To check a long
document, compare the length of the export's `extraction.pages` list with the
PDF's page count, and raise `PIPELINE_MAX_PAGES` if you need more pages; the
cap exists so that a long PDF cannot exhaust the memory used to render pages.

`--memory` (or `PIPELINE_MEMORY_MODE`) sets how many models stay loaded at
once: `aggressive` loads and unloads them stage by stage, `balanced` keeps the
layout and sentence models loaded, and `keep_all` keeps everything. Left
unset, bibr chooses from system RAM, CUDA VRAM and the LLM backend;
[choosing an extraction model](extraction-models.md#defaults-by-card) shows the
choice for each card.

## Presets

If you switch between setups often — cloud vs. local, different models for different
corpora — save each as a named preset instead of
hand-editing `.env` every time:

```bash
bibr preset save fast-gemini        # snapshot the current .env
bibr preset list                     # show all saved presets
bibr preset use fast-gemini          # apply a preset to .env
bibr preset show fast-gemini         # display a preset's contents
bibr preset diff fast-gemini         # compare a preset against the current .env
bibr preset rm fast-gemini --yes     # delete a preset (without --yes, it asks first)
bibr preset deactivate               # clear the active-preset marker (no other changes)
```

The "current `.env`" is the file whose values win: `./.env` when the working
directory has one, otherwise `~/.bibr/.env` (or the last existing file in
`BIBR_ENV_FILE`). `save`, `use` and `deactivate` name the file they read or
changed.

Presets are stored as JSON under `~/.bibr/presets/` (or `BIBR_PRESETS_DIR`),
readable only by you when bibr creates them. Secrets (API keys and similar,
and URLs such as `REDIS_URL` that carry a password or `?key=`) are excluded
when saving, and `save` names the URL settings it left out; endpoint URLs and
other private configuration may still be present. `show`, `diff` and `use`
mask secrets and URL credentials when they print values.

A preset can change where bibr sends requests while your API keys stay in
`.env`, or what it launches. When `bibr preset use` changes a `*_URL`,
`*_ARGS` or `*_EXECUTABLE` setting (`LLM_BASE_URL`, `OCR_BASE_URL`,
`LLM_LLAMA_CPP_EXTRA_ARGS`, ...), it names those settings; check them before
running bibr with a preset someone else gave you.

You can also apply a preset for a single run without touching `.env`:

```bash
bibr chew paper.pdf --preset fast-gemini
```

## Local model runtime

bibr runs all of its own models on your machine. Four of them (the
PP-DocLayoutV3 layout detector, the section classifier, the paper classifier
and the encoder+CRF reference parser) have both an ONNX Runtime and a PyTorch
implementation, and `ML_RUNTIME` picks which one loads; the default reference
parser ships as an ONNX bundle alone
([details](model-runtime.md#default-models-and-earlier-versions)). An ONNX bundle is `onnx/model.onnx` and
`onnx/bibr_onnx.json`, plus `onnx/tokenizer.json` for the text models, in the
model's Hub repo at its pinned revision:

| `ML_RUNTIME` | Behaviour |
|---|---|
| `auto` (default) | Use the ONNX bundle if it resolves; otherwise fall back to PyTorch if `torch` is importable; otherwise raise a `ConfigurationError` naming the model, `pip install 'bibr[torch]'`, and the setting that points at a local bundle. |
| `onnx` | The ONNX bundle must resolve, or `ConfigurationError`. |
| `torch` | `torch` must be importable, or `ConfigurationError`. |

This is what makes the [core install](../getting-started/install.md) able to run
the full HTTP-service path — OCR and the LLM over HTTP, every bibr-owned model
through ONNX Runtime — with no `torch`, `transformers` or OpenCV in the
environment. The `torch` extra remains the training-parity runtime, the Apple
MPS path, and the `torch.compile` path `bibr serve` uses for layout.

These bundles are published, so a core install needs no configuration (the front-matter tagger additionally stays off in a full run until `ML_FRONT_TAGGER_MODE=primary`; fast mode turns it on):

| Model | Repo | Pinned revision |
|---|---|---|
| Layout | `scienceverse/bibr-layout-onnx` | `2bcb16a6` |
| Section classifier | `scienceverse/bibr-section-classifier` | `cf7c2d85` |
| Paper classifier | `scienceverse/bibr-paper-classifier` | `600ef07f` |
| Reference parser | `scienceverse/bibr-parser-v7-mmbert-small` | `1045dc08` |
| Geometry line tagger | `scienceverse/bibr-ref-line-tagger-v1` | `b0effede` |
| Text reference segmenter | `scienceverse/bibr-ref-text-segmenter-v1` | `3c36a8c8` |
| Statement tagger | `scienceverse/bibr-statement-tagger-v1` | `b446f943` |
| Front-matter tagger | `scienceverse/bibr-front-tagger` | `13a0f5c3` |

`--device cpu` runs these models on the CPU. Loading rules, earlier model
versions, GPU and Mac execution, CPU threads and memory, and how to try a
retrained model are in [Local model runtime](model-runtime.md#bibrs-onnx-models).

## How settings resolve

For settings loaded from the environment, the order is:

1. **Environment variables** — always win.
2. **`.env` files** — both `~/.bibr/.env` and the current working directory's
   `.env` are read and merged key-by-key; where both set the same key, the
   CWD file wins. So a global config in `~/.bibr/.env` still applies when
   you run `bibr` from a directory that doesn't have its own `.env`, and a
   local `.env` can override individual keys from it.
3. **Built-in defaults** — used when a setting isn't in either of the above.

`BIBR_ENV_FILE` replaces layer 2 outright when you need to control it exactly:
set it to a specific path (or several, separated by `:` on Linux/macOS or `;`
on Windows, merged in the same last-wins order), or to an empty value to skip
`.env` loading altogether. The
empty form is useful in containers, CI, and test runs, where picking up
whatever `.env` happens to sit in the working directory is a surprise rather
than a convenience. `${NAME}` inside a `.env` value remains literal; bibr does
not interpolate it from another environment variable.

Explicit run flags such as `--ocr`, `--llm`, `--refs`, and `--memory` override
their corresponding settings for that invocation.

Set `BIBR_DISABLE_DOTENV=1` to skip both `.env` files entirely. Benchmark
harnesses and CI should do this so every setting a run records came from the
process environment, not from whatever `.env` happened to be in the checkout.

While `BIBR_DISABLE_DOTENV` or an empty `BIBR_ENV_FILE` turns `.env` loading
off, the configuration tools follow it: `bibr config show --sources` and
`bibr config path` report no `.env` file, and `bibr config set`, `bibr preset
save/use/deactivate/diff` and `bibr setup` refuse to run, naming the variable,
instead of writing a file bibr would not read.

### What `bibr setup` writes

`bibr setup` writes `./.env`, or the last file `BIBR_ENV_FILE` lists, since
that file's values win; it stops before asking anything when that file's
folder does not exist. Every run writes `PIPELINE_MODE` (`fast` or `full`), so
a key added or removed later cannot change the mode; a `.env` from an earlier
version has no entry and stays on `auto` until you run setup again. Fully local
runs can be slower, especially on weaker hardware and Apple Silicon, and first
runs may download several GB of model weights. If your machine is too weak but privacy still matters, use the
private-server path: run the Docker OCR/API stack on a GPU machine and point
your laptop's `.env` at that server.

Re-running `bibr setup` merges into an existing `.env` by default and keeps
values it does not ask about. When you choose an LLM provider, it also writes
`LLM_BACKEND=cloud`, and a blank `LLM_API_KEY` or `LLM_BASE_URL` where you
entered none, so that a key or server left by an earlier setup, in `./.env`
or `~/.bibr/.env`, cannot override the one you entered.

Environment variables still override it: when your
shell exports a setting the wizard writes (`LLM_API_KEY`, `LLM_BASE_URL`,
`LLM_PROVIDER`, ...), or another name bibr reads it under (`GEMINI_API_KEY`
for `GOOGLE_API_KEY`), with a different value, the wizard names it, so you
can unset or update it before running bibr.

### Namespaces

There's no global `BIBR_` prefix. Instead, each settings group has its own
short prefix — `LLM_` for the LLM client, `OCR_` for the OCR backend,
`CROSSREF_` for reference enrichment, `PIPELINE_` for orchestration, and so
on (the full list is in the [Settings reference](../reference/settings.md)).
