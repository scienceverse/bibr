# Configuration

bibr reads settings from environment variables and optional `.env` files.
CLI flags, presets, and Python API arguments also provide per-run overrides.
This page covers OCR, LLMs, references, and memory settings; the generated
[Settings reference](../reference/settings.md) lists every variable and default.

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
their corresponding settings for that invocation. In a uv project (the
Quickstart install or a source checkout), prefix the commands on this page with
`uv run`.

Set `BIBR_DISABLE_DOTENV=1` to skip both `.env` files entirely. Benchmark
harnesses and CI should do this so every setting a run records came from the
process environment, not from whatever `.env` happened to be in the checkout.

While `BIBR_DISABLE_DOTENV` or an empty `BIBR_ENV_FILE` turns `.env` loading
off, the configuration tools follow it: `bibr config show --sources` and
`bibr config path` report no `.env` file, and `bibr config set`, `bibr preset
save/use/deactivate/diff` and `bibr setup` refuse to run, naming the variable,
instead of writing a file bibr would not read.

```bash
bibr setup
```

`bibr setup` is the easiest way to get a working `.env`: it detects your
hardware, recommends a private/local setup where the machine can support it,
and asks only the follow-up questions needed to write the config. The wizard
is intentionally honest about trade-offs: fully local runs can be slower,
especially on weaker hardware and Apple Silicon, and first runs may download
several GB of model weights.

Run `bibr setup --advanced` when you want exact control over the LLM provider,
OCR backend, Crossref settings, cache extras, and model choices. If your local
machine is too weak but privacy still matters, use the private-server path:
run the Docker OCR/API stack on a GPU machine and point your laptop's `.env`
at that server.

Re-running `bibr setup` merges into an existing `.env` by default and keeps
values it does not ask about. When you choose an LLM provider, it also writes
`LLM_BACKEND=cloud`, and a blank `LLM_API_KEY` or `LLM_BASE_URL` where you
entered none, so that a key or server left by an earlier setup, in `./.env`
or `~/.bibr/.env`, cannot override the one you entered.

`bibr setup` writes `./.env`, or the last file `BIBR_ENV_FILE` lists, since
that file's values win; it stops before asking anything when that file's
folder does not exist. Environment variables still override it: when your
shell exports a setting the wizard writes (`LLM_API_KEY`, `LLM_BASE_URL`,
`LLM_PROVIDER`, ...), or another name bibr reads it under (`GEMINI_API_KEY`
for `GOOGLE_API_KEY`), with a different value, the wizard names it, so you
can unset or update it before running bibr.

## Namespaces

There's no global `BIBR_` prefix. Instead, each settings group has its own
short prefix — `LLM_` for the LLM client, `OCR_` for the OCR backend,
`CROSSREF_` for reference enrichment, `PIPELINE_` for orchestration, and so
on (the full list is in the [Settings reference](../reference/settings.md)).
Two examples:

```bash
# .env
LLM_PROVIDER=openai
CROSSREF_API_EMAIL=you@example.org
```

`LLM_PROVIDER` selects the LLM client's provider; `CROSSREF_API_EMAIL` is
passed to Crossref's "polite pool" for reference enrichment lookups.

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

## Choosing an OCR backend

`bibr chew` defaults to the `paddle` startup selector. It selects a concrete
runtime when OCR starts:

| Request or runtime | When it's used |
|---|---|
| `paddle` | Default automatic selector. Linux x86_64 with at least 8 GB detected NVIDIA VRAM: `paddle-vllm`, then `glm-llama`; otherwise `glm-llama`. Apple Silicon: `paddle-rapid-mlx`, `paddle-mlx-vlm`, `glm-rapid-mlx`, then `glm-llama`. Windows: `glm-llama`. |
| `paddle-vllm` | Explicit managed PaddleOCR-VL-1.6 vLLM runtime on Linux/CUDA. |
| `paddle-rapid-mlx` / `paddle-mlx-vlm` | Explicit PaddleOCR-VL-1.6 Apple-Silicon runtimes. Both use the quantized `olragon/PaddleOCR-VL-1.6-8bit` default. |
| `paddle-http` | Explicit external Paddle server, addressed by the served alias `paddle-ocr-vl-1.6`. |
| `glm-rapid-mlx`, `glm-llama`, `glm-http` | Explicit GLM-OCR compatibility choices. They are also the ordered startup fallback candidates for `paddle` where listed above. The older `glm-mlx` (vllm-mlx) backend is disabled and refuses to start; use `glm-rapid-mlx`. |
| `gemini` / `openai` / `anthropic` | Vision-LLM OCR via a cloud provider — no local GPU needed, useful for a laptop or a low-resource box. |

The fallback is **startup-only**: a candidate must construct and pass its
readiness check before it is selected. Once one has started, bibr records its
identity and does not silently send failed individual OCR requests to GLM.
Choose a concrete backend with `--ocr <name>` (or `OCR_BACKEND` in `.env`).
`--ocr-url` (and the library's `ocr_url`) routes to an external Paddle
endpoint (`paddle-http`) by default; specify `--ocr glm-http` (or any `glm`
backend, or set `OCR_BACKEND` to one) for a GLM endpoint. An explicit
`--ocr gemini`, `openai` or `anthropic` ignores the URL with a warning. A
cloud vision `OCR_BACKEND` does not override a URL you pass: the run uses
that server as `paddle-http` and sends no page image to the cloud provider.
The cloud vision backends all use `OCR_VISION_MODEL`, whose default
(`gemini-3-flash-preview`) is a Gemini model, so with `--ocr openai` or
`--ocr anthropic` set it to one of that provider's vision models, such as
`OCR_VISION_MODEL=claude-haiku-5-5`.
For example:

```bash
bibr chew paper.pdf --ocr paddle-http --ocr-url https://ocr.example.org
bibr chew paper.pdf --ocr glm-http --ocr-url https://ocr.example.org
```

Remote OCR requires HTTPS by default. For a trusted private network without
TLS, explicitly set `OCR_ALLOW_INSECURE_HTTP=true`; loopback HTTP is allowed.

An `LLM_BASE_URL` (or `OCR_VISION_BASE_URL`) that receives an API key may use
plain `http://` for loopback and private-network hosts — a local vLLM or
Ollama, a GPU box on the LAN or tailnet. For a public host bibr refuses to
send the key over plain HTTP unless `LLM_ALLOW_INSECURE_HTTP=true`; the
pipeline, `bibr setup` and `bibr doctor` apply the same rule. The vision
endpoint receives the LLM provider's key, so the LLM setting governs it too,
not `OCR_ALLOW_INSECURE_HTTP`.

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
Markdown/LaTeX fence or balanced display wrapper from formulas. It retains the
unmodified model response as `raw_ocr_content` in the opt-in
`extraction.regions` diagnostic payload. The selected backend, model, and
profile are included in `extraction.ocr` and in the OCR-cache identity, so a
cache hit never crosses a Paddle/GLM or normalizer boundary.

The managed llama.cpp servers default to full GPU offload, flash attention,
and `q8_0` KV-cache quantization. OCR uses one parallel slot. The LLM can use
two slots with a unified KV cache and n-gram speculative decoding when the
installed server supports those flags, otherwise it uses one slot. Override
individual flags with
`OCR_LLAMA_CPP_EXTRA_ARGS` / `LLM_LLAMA_CPP_EXTRA_ARGS` (user flags replace the
matching defaults). On older GPUs (Pascal / GTX 10-series) PyTorch may fall
back to CPU for layout/NER while llama.cpp still uses the GPU — that is
expected; install a CUDA build of `llama-server` and skip the `gpu` extra.

## Choosing an LLM

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

`--llm local` starts a managed server. It resolves to Rapid-MLX on Apple
Silicon when that executable is available, otherwise vllm-mlx; llama.cpp on
Windows, CUDA cards below 10 GB and CUDA cards older than Ampere (compute
capability below 8.0); and vLLM on other Linux/CUDA systems with at least
10 GB (or when VRAM detection is unavailable). With the same card it picks the
model, the memory mode and the OCR that `bibr setup` writes.
Explicit choices are `vllm`, `vllm-mlx`, `rapid-mlx`, `llama-cpp`, and `llmster`.
The OCR and LLM choices are independent, so local OCR with a cloud LLM is a
supported hybrid configuration. A managed vLLM LLM server and the managed
`paddle-vllm` OCR server each reserve a share of GPU memory. The LLM server
takes at most 0.85 of the card (see below); PaddleOCR-VL takes about 5 GB, so
0.16 of a 32 GB card, 0.21 of 24 GB and 0.32 of 16 GB. A vLLM server starts
only when its whole share is free, and the rest of the card holds bibr's
layout, sentence and reference models, which stay on the GPU between chunks
when `onnxruntime-gpu` is installed. Unless the two shares leave at least 6 GB
of the card for those models (with the LLM at 0.85, only on cards of about
80 GB), bibr stops the server that is running before starting the other: in a
batch, each chunk's OCR stops the LLM server and that chunk's LLM stage starts
it again. The shares are sized from the card that `CUDA_VISIBLE_DEVICES`
exposes; when bibr cannot tell which card that is (cards of different sizes
numbered in CUDA's default fastest-first order), PaddleOCR-VL keeps 0.92 and
the servers take turns. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` or select the card
by UUID to avoid that.

The LLM server works out its share each time it starts, from the GPU memory
free at that moment. It takes at most 0.85 of the card, and less when the card
is busier: the share leaves room for vLLM's brief extra use while it starts
(0.75 GiB for the 4B student's builds, 1 GiB for the 2B) and for bibr's
models. That room is 6 GB less what bibr's models already hold, and at least
1 GB: about 0.71 of a free 24 GB card and 0.78 of a free 32 GB card. When that
does not leave the model its minimum budget (11.0 GiB for the 4B student,
6.0 GiB for its W4A16 build, 6.25 GiB for the 2B; more with
[speculative decoding](#speculative-decoding-on-vllm)), a server that takes
turns with PaddleOCR-VL drops to the minimum while bibr's models keep at least
1 GB, since it starts again beside them in the next chunk. A server that stays
up for the run (with cloud OCR, an OCR server at `--ocr-url` or GLM-OCR) keeps
that room: it usually starts in the first chunk, before bibr's models have
grown to the 5 GB or so they reach in a batch. So on a 24 GB card that also
runs a PaddleOCR-VL server at `--ocr-url` (about 5.4 GB), the 4B student just
fits (it needs 17.25 GiB, and about 17.7 are free) with an 11.3 GiB budget,
below MTP's minimum; with a desktop on the card as well it does not, and its
W4A16 build does. When the share does not fit, the start fails at once with an
error that says how much memory was free and how much it needs, that chunk's
papers fail, and the next chunk tries again. Set `LLM_LOCAL_MEM_FRACTION`, or
`--gpu-memory-utilization` in `LLM_VLLM_EXTRA_ARGS`, to start the LLM server at
exactly that share instead.

The LLM server starts after a chunk's OCR, and OCR only adds to what bibr
holds. So while the LLM server is down, bibr checks before each chunk's OCR
whether the memory free then could start it. When it could not, the chunk's
papers fail with the same error before any OCR runs, and the check is made
again before the next chunk, which goes ahead once enough memory is free. A
server already listening on the LLM server's port (`LLM_VLLM_PORT`) is never
checked: the start reuses one that serves the model, such as one another
pipeline in the same process started, without asking for memory.

The minimum comes from bibr's model registry, which lists the published
builds with the smallest budget each was measured to start with: a first
start, before vLLM has filled its compile cache, that served one
32,768-token request on an RTX 3090, rounded up to a quarter GiB. The 4B
student's builds were measured with vLLM 0.32.0 at 64 sequences, the 2B with
vLLM 0.31.0 at 128. On vLLM 0.31 a later start reused the cache and needed
about 1 GiB less. The minimum is a budget, not a card size: on the swap path a
start also needs vLLM's extra use while it starts (0.75 GiB at 64 sequences,
1 GiB at 128) and 1 GiB for bibr's models free beside it, so the W4A16 build
needs 7.75 GiB free and the 4B student 12.75 GiB. A server that stays up needs
6 GiB less what bibr's models hold in place of that 1 GiB: 12.25 and 17.25 GiB
while they hold 0.5 GiB. A model the registry does not list, such as a local
path, has no minimum, so a start that cannot fit is launched and fails inside
vLLM. Set `LLM_LOCAL_MIN_VRAM_GB` to give it one. For a model the registry
lists, the setting can raise the minimum but not lower it: a lower value is
ignored and bibr logs a warning once, so a value left over from a local build
cannot start the 4B student below its 11.0 GiB. For a local copy of the 4B
student's W4A16 build, the minimum is 6.0 GiB with vLLM 0.32 and at most 64
sequences (more with 128, since vLLM keeps a cache block for each sequence):

```bash
LLM_LOCAL_MODEL=/path/to/bibr-extract-qwen35-4b-W4A16
LLM_LOCAL_MIN_VRAM_GB=6.0
LLM_VLLM_EXTRA_ARGS='--language-model-only --generation-config vllm --max-model-len 32768 --max-num-seqs 64'
LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
```

`bibr chew`, `bibr batch` and `bibr mcp` stop the managed servers they started
when the command ends, including on Ctrl-C, SIGTERM (`kill`, `docker stop`,
`timeout`, an MCP host stopping `bibr mcp`) and SIGHUP (a closed terminal or SSH
session). A `bibr chew` or `bibr mcp` stopped by a signal exits with 128 plus its
number (143 for SIGTERM) and `bibr batch` exits 130 as after Ctrl-C; in a systemd
unit, set `SuccessExitStatus=130 143` so that `systemctl stop` counts as a clean
stop.

Run `bibr setup` to choose the model as well as the runtime. It recommends a
model for your hardware ([choosing an extraction model](extraction-models.md)
has the measurements):

| Runtime | Setup model | Registry memory floor |
|---|---|---|
| vLLM, 20 GB or more | `scienceverse/bibr-extract-qwen35-4b` (Qwen3.5-4B fine-tuned for bibr, bf16) | an 11.0 GiB budget; 12.75 GiB free to start |
| vLLM, 10–19 GB | `scienceverse/bibr-extract-qwen35-4b-W4A16` (the same model with 4-bit weights) | a 6.0 GiB budget; 7.75 GiB free to start |
| llama.cpp (Windows, cards below 10 GB) | `scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0`, else `:Q4_K_M`, else the 2B's `scienceverse/bibr-extract-qwen35-2b-GGUF` builds | 7 GB for the 4B Q8_0, down to 3 GB for the 2B Q4_K_M |
| Rapid-MLX / vllm-mlx | `scienceverse/bibr-extract-qwen35-4b-mlx-8bit`, else the 2B's `-2b-mlx-8bit` | 6 GB for the 4B, 3 GB for the 2B |

When no local model fits, the recommended hosted model is Claude Haiku 5.5
(`LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-haiku-5-5`), which `bibr setup`
also uses for cloud OCR (`OCR_BACKEND=anthropic`,
`OCR_VISION_MODEL=claude-haiku-5-5`), so one Anthropic key covers both.

These are model-fit estimates, not total pipeline memory guarantees. Setup
matches a card that reports up to 0.25 GiB under a size (an RTX 5070 reports
12,227 MiB) to that size, for the model and for the memory mode; PaddleOCR-VL's
8 GB minimum has no such slack. On a Linux card below 10 GB that could run
PaddleOCR-VL, setup sets `OCR_BACKEND=glm-llama`, so OCR and the LLM both run
on llama.cpp and take turns on the card; without setup, the automatic OCR
chain does the same whenever the LLM runs on llama.cpp. Cards below 12 GB
whose LLM server bibr manages get the aggressive memory mode
(`PIPELINE_MEMORY_MODE=aggressive`, 3 papers a chunk), which unloads bibr's
own models before each start of the LLM; with a cloud or remote LLM, only
cards of 8 GB or less do. A managed llama.cpp LLM server stops before each
start of GLM-OCR on llama.cpp in the aggressive mode, and in the others only
when the memory free cannot also hold GLM-OCR (2.25 GiB) and 1 GiB for bibr's
models. In the balanced mode, when the memory free cannot
start the managed vLLM LLM server, bibr first releases the same models (and a
resident OCR server) and refuses the start only if it is still short.

Cards older than Ampere (compute capability below 8.0: Turing, Volta,
Pascal) run the LLM and the automatic OCR on llama.cpp at any size: bibr's
vLLM models are bf16, which vLLM serves from compute capability 8.0, and
on an older card it casts them to float16, which bibr has not tested. No
batch on a Turing or Pascal card has been recorded with either runtime; this
is a conservative routing rule. The advanced wizard
also offers the 2B student on vLLM from 12 GB
(`scienceverse/bibr-extract-qwen35-2b`, a 6.25 GiB budget at 128 sequences and
8.25 GiB free to start), Gemma 4 E4B on CUDA, and custom model IDs.
`LLM_LOCAL_MODEL` selects managed local weights; `LLM_MODEL` and `--llm-model`
select the provider model. When `LLM_LOCAL_MODEL` is unset, every managed
runtime uses the build `bibr setup` recommends for the card: on vLLM the 4B
bf16 from 20 GB and its W4A16 build below, on llama.cpp the 4B Q8_0 from 7 GB
and Q4_K_M below, and the 8-bit build on vllm-mlx. The managed llama.cpp server starts with thinking off
(`--reasoning off`, or `--reasoning-budget 0` on older builds, plus
`--chat-template-kwargs '{"enable_thinking": false}'`) when the installed
build supports it, and `bibr setup` also writes
`LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'` for the fine-tuned models. Rapid-MLX
has a separate `LLM_RAPID_MLX_MODEL` default (`qwen3.5-4b-4bit`) when no local
model was explicitly configured.

`LLM_STRUCTURED_BACKEND=auto` uses Instructor. The experimental
`nuextract-native` backend, for a NuExtract server you run yourself, is
explicit-only and needs `pip install 'bibr[nuextract]'`, because bibr no longer
installs numind by default.

`LLM_POINTER_OUTPUTS=true` is experimental and off by default. It shortens
the answers of the two front-matter calls: the title and keywords call names
the abstract's first and last words and bibr copies the abstract from the
front matter, and the authors call lists each affiliation once and gives each
author its numbers. When the abstract cannot be placed in the text, or the
answer is cut off or does not parse, bibr asks for the verbatim answer
instead. The merged call (`LLM_MERGED_CORE_METADATA`) and the
`nuextract-native` backend do not use it. The current revision of bibr's 4B
was trained on these answers and passes as many papers with the flag on as
off, but first-author accuracy is lower with it on and bibr's resolver can
take a wrong span as the abstract without a warning, so it stays off; see
[choosing an extraction model](extraction-models.md#pointer-answers-llm_pointer_outputs).

For `--llm llmster`, install LM Studio's `lms` CLI and download the desired model
first, then set `LLM_LLMSTER_MODEL` to its model key. Bibr can start its daemon
and API server and load that existing model. It does not install the runtime
or download a model, and cleans up only resources it started.

Local inference speed depends on the runtime, model, hardware, and document.
Validate the fields you need on representative papers before choosing a model
for a large run. Cloud LLMs or an external OpenAI-compatible server can be used
with local OCR when local LLM throughput is insufficient.

`LLM_RATE_LIMIT_RPM` defaults to 60 requests per minute to stay inside a cloud
provider's quota. For a server bibr starts itself, and for an `LLM_BASE_URL` on
loopback, a private network or a tailnet, bibr uses 600 instead, so the server's
own queue and `LLM_MAX_CONCURRENCY` set the pace. The `ollama` provider gets 10.
Setting `LLM_RATE_LIMIT_RPM` yourself overrides all of these.

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
logged. Claude Haiku 5.5, Sonnet 5.5, Opus 5.5 and Fable 5.1 reject any `temperature`, so
calls to them carry none and the model samples at its own default; the same
paper can then come back slightly different between runs. They take adaptive
thinking instead of a token budget, which bibr does not send yet, so
`LLM_THINKING_BUDGET` is ignored for them and a warning is logged. The
`anthropic` provider needs the anthropic SDK below 1.0, which every install
of bibr includes. With 1.0 or later, bibr stops before the first call with an error
that names the fix.

`CACHE_LLM=true` keeps structured LLM responses on disk (`CACHE_LLM_DIR`,
default `~/.cache/bibr/llm`), readable by your user only. An entry is reused
only for the same provider, endpoint, model, call parameters (temperature,
output cap, reasoning effort, thinking budget) and prompt, so changing any of
them calls the model again; for `--llm llmster`, so does changing
`LLM_LLMSTER_MODEL`. Another endpoint that serves different weights under the
same model name and URL cannot be told apart: clear the directory after
swapping the model behind it.

### Pinned model revisions

Each build of bibr's 4B student in the registry is pinned to a commit on the
Hugging Face Hub ([the list](extraction-models.md#builds-and-revisions)).
vLLM gets the commit as `--revision`. `llama-server -hf` and the MLX servers
(vllm-mlx, Rapid-MLX) always download a repository's `main`, so for the GGUF
and MLX builds bibr downloads the pinned files into the Hugging Face cache
itself (`HF_HUB_CACHE`, or `HF_HOME/hub`; for Rapid-MLX,
`RAPID_MLX_HF_HUB_CACHE` or `RAPID_MLX_HF_HOME/hub` when set) and starts the
server on the local copy, still under the model's Hub id. A build already in
the cache starts with no request to the Hub, so offline runs work once it has
been downloaded. Offline without it, the start stops with the `hf download`
command that fetches it. Another commit of the same repository in the cache,
such as its `main`, does not count: `bibr chew --dry-run` says
`will download revision 0aa228e8` until the pinned one is there.

Each repository's `main` still holds the builds of the 4B's previous revision
(`6ab47a4e`), but `main` will move on, so the commands below name each
build's commit. To run one of them, keep `LLM_LOCAL_MODEL` as setup wrote it
and override the revision, or point `LLM_LOCAL_MODEL` at a local copy:

```bash
# vLLM: the last --revision wins over the registry's.
LLM_VLLM_EXTRA_ARGS='--revision 6ab47a4e4e8b1b76c1ab234aec66f360683ff9fc'   # bf16
LLM_VLLM_EXTRA_ARGS='--revision a2af8f6e90e6481251af9eb07d02c034c30e3446'   # W4A16

# llama.cpp: a local .gguf file is loaded with -m
hf download scienceverse/bibr-extract-qwen35-4b-GGUF bibr-extract-qwen35-4b-Q8_0.gguf --revision 8f5cb6f80329f800f17f3b25dd40be131068f872
LLM_LOCAL_MODEL=/path/to/bibr-extract-qwen35-4b-Q8_0.gguf

# MLX: a local model directory
hf download scienceverse/bibr-extract-qwen35-4b-mlx-8bit --revision 16bf58c9e4f453176cdabf3cfe64665c92a006c3 --local-dir ./bibr-4b-mlx-6ab47a4e
LLM_LOCAL_MODEL=./bibr-4b-mlx-6ab47a4e
```

On vllm-mlx, bibr serves a local MLX directory as a text model, as it does the
pinned build: vllm-mlx alone would take the `image_token_id` in its
`config.json` for a vision model and fail to load it.

A local path is not a registry model, so bibr applies none of the registry's
memory floors or launch arguments to it; the vLLM `--revision` route keeps
them. The 4B's FP8-Dynamic build
(`scienceverse/bibr-extract-qwen35-4b-FP8-Dynamic`, 219/252 on an RTX 5090) is
not in the registry either, so setup does not offer it and it has no measured
memory floor. To try it on vLLM, set `LLM_LOCAL_MODEL` to that id and take
`LLM_VLLM_EXTRA_ARGS` and `LLM_CHAT_TEMPLATE_KWARGS` from the local W4A16
example above, with `--revision 101d1ee1d449195f6882019b0fd2aa4328ed8bef`
added to the extra args. That example's `LLM_LOCAL_MIN_VRAM_GB=6.0` was
measured for the W4A16 build, not this one.

### Speculative decoding on vLLM

`LLM_VLLM_SPEC_DECODE` sets speculative decoding for the managed vLLM LLM
server. It applies to the published builds of the 4B student, which bibr's
registry lists with the configs below; other models, local paths included,
start without it unless `LLM_VLLM_EXTRA_ARGS` passes a `--speculative-config`,
which always replaces bibr's choice.

| Value | What the server starts with |
|---|---|
| `auto` (default) | MTP (3 tokens drafted by the model's own multi-token-prediction layers) when the server stays up for the whole run and its budget is at least MTP's minimum: 12.0 GiB for the 4B student, 7.25 GiB for its W4A16 build. Otherwise none. Below the minimum the server starts without MTP; `auto` never refuses a start. |
| `none` | no speculative decoding |
| `mtp` | MTP at every start, including starts that take turns with PaddleOCR-VL. The start needs MTP's minimum: on the swap path 13.75 GiB free for the 4B student and 9 GiB for the W4A16 build; a server that stays up for the run needs 18.25 and 13.5 GiB while bibr's models hold 0.5 GiB. |
| `dflash` | DFlash (7 tokens drafted by [`z-lab/Qwen3.5-4B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash), a separate 1.27 GB model pinned to revision `9a1996cc`), for the 4B student's bf16 build only, with a 13.5 GiB minimum. Other models start without it, with a warning. |

The server stays up for the run when no managed PaddleOCR-VL server takes
turns with it: with cloud OCR, an OCR server at `OCR_BASE_URL`, or OCR on
llama.cpp. With the automatic OCR on a card of up to 64 GB, PaddleOCR-VL and
the LLM take turns, the LLM restarting every chunk, and `auto` leaves MTP off.
Both MTP and DFlash add `--no-async-scheduling`, which every measured run
used.

What was measured, with vLLM 0.32.0 at 64 sequences:

- **One LLM server for the run** (the 252 development papers, OCR read from a
  cache, the server at 0.85 of the card): with MTP the 4B student passed 218
  against 219 without it on an RTX 3090 and 218 against 218 on an RTX 5090;
  its LLM stage took 0.80x and 0.74x as long, and papers an hour rose 5.7%
  and 12.1%. With DFlash: 220 against 219 and 217 against 218, the LLM stage
  0.72x and 0.64x as long, papers an hour +8.1% and +15.8%, but about half the
  KV cache capacity (159,251 against 310,108 tokens at 0.85 of an RTX 3090).
  These runs served the bf16 build at the registry's revision (`9199424d`).
  The previous revision's W4A16 build (`a2af8f6e`) passed 218/252 with MTP,
  as without it on vLLM 0.31; the pinned W4A16 build has not been scored with
  MTP.
- **The default plan on 12 and 16 GB cards** (the previous revision's W4A16
  build taking turns with PaddleOCR-VL, 48 papers, emulated cards beside a
  0.5 GiB desktop): MTP passed 40 and 39 of 48 against 39 and 40 without it,
  but each LLM start took 51 s instead of 37 s, and papers an hour fell 5.0%
  and 8.6%. On the 12 GB card three of its six starts fitted only after bibr
  released its own models. This is why `auto` leaves MTP off there.
- Speculative decoding has not been measured on 24 GB cards in the swap plan,
  on the 2B student, or with DFlash on the W4A16 build.

### Managed local servers

The local OCR backends and `--llm local` start their model server as a child
process that listens on `127.0.0.1` at a fixed port. The vLLM and llama.cpp
servers get a new random API key at every launch. bibr passes the key through
the server's environment, not its command line, so other users on the machine
cannot call them. Rapid-MLX, vllm-mlx and MLX-VLM servers run without a key.
`OCR_API_KEY` goes only to an OCR endpoint you configure (`--ocr-url` or
`OCR_BASE_URL`), never to a server that bibr manages.

If the port already has a server that lists the requested model, bibr reuses
it and logs a warning that it did not start that server. A reused server
receives your documents, so stop any server there that you did not start
yourself. Two pipelines in one Python process, such as an open `bibr.Chewer`
and a `bibr.chew()` call, share the vLLM or llama.cpp server that the first one
started; it stops when the pipeline that started it closes. bibr does not share
a server that rejects its unauthenticated probe with HTTP 401 and was not
started by this process, which is usually a server that another bibr process
started with its own key. Stop that server or configure a free port.

The Rapid-MLX OCR backends (`glm-rapid-mlx`, `paddle-rapid-mlx`) restart their
server every `OCR_RAPID_MLX_RECYCLE_AFTER` regions (default 80). This releases
memory that Rapid-MLX's vision cache would otherwise leak. bibr can restart only
a server it started, so these backends refuse a server that is already running
on their port, including one that another pipeline in the same process started.
`paddle-mlx-vlm` shares its port and model with `paddle-rapid-mlx`, so it
refuses a Rapid-MLX or vllm-mlx server it finds there too; an MLX-VLM server is
still reused. To fix this, stop that server, move the port
(`OCR_RAPID_MLX_PORT`, or `OCR_PADDLE_MLX_PORT` for the Paddle backends), or set
`OCR_RAPID_MLX_RECYCLE_AFTER=0` so that the Rapid-MLX backends reuse the server
without restarts. Each refusal is logged as a warning that names the port. The
automatic `paddle` chain reports only which candidates failed, and then tries
the next one.

Each server writes its stderr to a private log file in the system temp
directory, and bibr logs the path when the server starts. A Rapid-MLX server
keeps a single log file for the whole run, across all of its restarts.

## Presets

If you switch between setups often — cloud vs. local, different models for different
corpora — save each as a named preset instead of
hand-editing `.env` every time:

```bash
bibr preset save fast-gemini   # snapshot the current .env
bibr preset list                # show all saved presets
bibr preset use fast-gemini     # apply a preset to .env
bibr preset show fast-gemini    # display a preset's contents
bibr preset diff fast-gemini    # compare a preset against the current .env
bibr preset rm fast-gemini      # delete a preset
bibr preset deactivate          # clear the active-preset marker (no other changes)
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

## Reference-extraction strategies

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

## Local model runtime

Four of bibr's models run locally: the PP-DocLayoutV3 layout detector, the
section classifier, the paper classifier and the encoder+CRF reference
parser. Each ships twice — as PyTorch weights, and as an ONNX bundle
(`onnx/model.onnx` + `onnx/bibr_onnx.json`, plus `onnx/tokenizer.json` for the
text models) in the same Hub repo at the same pinned revision. `ML_RUNTIME`
picks which one is loaded:

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

A bundle resolves from a local directory containing `onnx/` (point the model's
existing `*_MODEL_ID` / `NER_PARSER_CKPT` setting at it) or from the Hub at the
pinned revision, offline-tolerant through the Hub cache. A value shaped like a
Hub id (`org/name`) is always fetched from the Hub, even when a directory of
that name exists in the working directory; write a local directory as a path —
absolute, or starting with `./`, `../` or `~`. Layout is the one model
whose PyTorch weights live in a third-party repo, so its ONNX artifact has its
own pair of settings, `LAYOUT_ONNX_MODEL_ID` and `LAYOUT_ONNX_REVISION`.

All four bundles are published, so a core install needs no configuration:

| Model | Repo | Pinned revision |
|---|---|---|
| Layout | `scienceverse/bibr-layout-onnx` | `2bcb16a6` |
| Section classifier | `scienceverse/bibr-section-classifier` | `cf7c2d85` |
| Paper classifier | `scienceverse/bibr-paper-classifier` | `6046171b` |
| Reference parser | `scienceverse/bibr-parser-v5-1-ultrabert` | `97b06cc7` |

The paper-classifier pin is the commit that added `onnx/` to the previously
audited weights, so its PyTorch weights are unchanged. The reference-parser pin
is the v5.1 parser (UltraBERT-150M encoder); its repo also carries the encoder's
`config.json`, modeling code and tokenizer, so the PyTorch runtime loads them
from the same pinned commit with `trust_remote_code` and never runs unpinned
code, while the ONNX bundle needs no custom code at all. For that reason the
PyTorch runtime cannot load it at `NER_PARSER_REVISION=main`; pin a commit SHA. Set
`NER_PARSER_CKPT=scienceverse/bibr-parser-v4-5-gold` and
`NER_PARSER_REVISION=ff50a83e7f5b73dcf6f8f973a1a6e3847ec429e6` to keep the
previous ModernBERT-base parser. `NER_PARSER_ENCODER` (and
`NER_PARSER_ENCODER_REVISION`) override the encoder for a checkpoint that does
not ship one; an encoder with its own modeling code runs it only from a local
directory or a commit SHA. The section-classifier pin is the model retrained in 0.7.1; set
`ML_SECTION_CLASSIFIER_REVISION=ee1a83db01ce947e3a5eb87dfde7221328f4b207` to
keep the previous one.

Execution providers come from the same chain the sentence segmenter uses
(CUDA → CoreML → CPU), so `bibr[gpu]` accelerates all four models, not just
segmentation. `--device cpu` runs these models on the CPU provider alone (no
CoreML, which may compute in FP16), and `device="cuda:1"` in the Python API puts
them on that GPU.

`Dockerfile.serve` sets `ML_RUNTIME=torch` explicitly. That image exists for the
PyTorch stack — `torch.compile` on the layout model in particular — and since the
classifier revisions it bakes now also carry an `onnx/` bundle, leaving the
setting unset would let `auto` move serve onto ONNX Runtime the next time the
image is built. Switching it is a deliberate choice, not a build-time accident.

### Layout model generation

bibr ships with PP-DocLayoutV3 and can also run PP-DocLayoutV4, which keeps
V3's 25 region labels but predicts a quadrilateral per region (bibr uses the
rectangle enclosing it) and decodes reading order from a successor head as well
as V3's relative-order head. V4 is not the default: PaddlePaddle has not yet
published its weights (`PaddlePaddle/PP-DocLayoutV4_safetensors`), and the
downstream layout rules were tuned on V3, so switching goes through the
evaluation gate like any other model change.

Each runtime selects the generation on its own:

- **ONNX** (the default runtime): the bundle's `bibr_onnx.json` names the
  architecture it was exported from, and that picks the pre- and
  post-processing. Point `LAYOUT_ONNX_MODEL_ID` / `LAYOUT_ONNX_REVISION` at a V4
  export — `scripts/export_onnx_layout.py --model-id
  PaddlePaddle/PP-DocLayoutV4_safetensors --revision <sha>` writes one and checks
  it against transformers.
- **PyTorch**: set `LAYOUT_MODEL_ID` and `LAYOUT_MODEL_REVISION` to the V4
  checkpoint. This needs a transformers release that includes PP-DocLayoutV4.

Under the ONNX runtime `LAYOUT_MODEL_ID` and `LAYOUT_MODEL_REVISION` are not
used, and bibr logs a warning when they name a different checkpoint than the
bundle was exported from. bibr refuses a V4 bundle or checkpoint whose label
list differs from the one it maps, or a V4 bundle that declares none, since
every downstream rule keys on the label names. The OCR disk cache keys on all
four layout settings and on `ML_RUNTIME`, so switching either never replays
cached regions of the other model.
