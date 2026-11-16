# What leaves your machine

bibr sends nothing about a paper anywhere unless a setting below says so, and
it has no usage reporting of its own. This page lists every route that does
send something, so you can choose a setup that keeps your papers on your
machine. `bibr chew paper.pdf --dry-run` shows the mode, OCR and LLM a run
would use before it starts.

## Keep a paper on the machine

For papers that must not leave the machine:

- Run in fast mode: `PIPELINE_MODE=fast` in `.env`, or `--fast` on each run.
- Set no OCR, or set `--ocr text-layer`. A scanned PDF then needs a local OCR
  backend (`--ocr paddle`) or an OCR server you run (`--ocr-url`).
- Leave Crossref enrichment off (the default).
- Set `HF_HUB_OFFLINE=1` after the first run has downloaded the models.

## Every route

What a run contacts depends on how it is set up:

| Setup | What is sent | Where |
|---|---|---|
| Fast mode with no OCR setting (the default when no LLM is configured, or `--fast`) | Nothing from the paper | — |
| A local OCR or LLM server that bibr starts or that you run (`--ocr paddle`, `--llm local`, `--llm vllm`, `--ocr-url`, `LLM_BASE_URL` on your own host) | Page images and text from the paper | That server only |
| A cloud OCR model (`--ocr anthropic`, `gemini` or `openai`, also as `OCR_BACKEND` or `OCR_CONSENSUS_BACKEND`) | Images of the paper's pages, or parts of them | The OCR provider |
| A cloud LLM (full mode with `LLM_PROVIDER` `google`, `openai`, `anthropic` or `groq`, or an `LLM_BASE_URL` on someone else's host) | Text from the paper | The LLM provider |
| Enrichment (`--crossref`, or `CROSSREF_ENRICH=true`; off by default) | The references' DOIs, titles and author names; affiliation strings and funder names unless `ROR_ENRICH=false`; your `CROSSREF_API_EMAIL` if set | `api.crossref.org`, `api.ror.org`, and the resolver at `BIBR_RESOLVER_URL` if you set one |
| `bibr batch --serve-url` | The whole file | The bibr server you name |
| The MCP tool `chew_url` | A request for the URL it is given | The site that hosts it |

### LLM routes

An LLM is called only in full mode. Where the paper's text goes depends on the
provider:

| `LLM_PROVIDER` or `--llm` | Where the text goes |
|---|---|
| `google` (the default provider) | Google's Gemini API |
| `openai` | OpenAI, or the server at `LLM_BASE_URL` when you set one |
| `anthropic` | Anthropic |
| `groq` | Groq |
| `ollama` | The Ollama server at `LLM_OLLAMA_BASE_URL` (default `http://localhost:11434`) |
| `--llm local` (or `vllm`, `llama-cpp`, `vllm-mlx`, `rapid-mlx`) | A server bibr starts on this machine, listening on `127.0.0.1` |
| `--llm llmster` | LM Studio's server on this machine, run through its `lms` CLI |

bibr sends an API key over plain `http://` only to loopback and
private-network hosts, unless you set `LLM_ALLOW_INSECURE_HTTP=true`.

### Cloud OCR

The cloud OCR backends (`gemini`, `openai`, `anthropic`) send images of a
paper's pages, or of regions of them, to that provider. An OCR server URL wins
over a cloud `OCR_BACKEND`: with `--ocr-url`, the run uses that server as
`paddle-http` and sends no page image to the cloud provider. When `bibr setup`
chooses the hosted model Claude Haiku 5.5, it also sets Claude for OCR
(`OCR_BACKEND=anthropic`), so page images go to Anthropic as well as text.

### Crossref enrichment

Enrichment is off by default. With `--crossref` or `CROSSREF_ENRICH=true`,
bibr looks each reference up at Crossref and sends the references' DOIs,
titles and author names, and your `CROSSREF_API_EMAIL` if you set one.
Affiliation strings and funder names go to ROR (`api.ror.org`) unless
`ROR_ENRICH=false`. A resolver you set with `BIBR_RESOLVER_URL` also receives
the lookups.

## What happens without a setting of yours

- **Model downloads.** The first run downloads bibr's models from the Hugging
  Face Hub. Those requests name the models and the library versions, nothing
  from a paper. Once the models are in the cache, `HF_HUB_OFFLINE=1` keeps bibr
  from contacting the Hub at all.
- **A Google key in the environment.** With no mode chosen, bibr runs in full
  mode when an LLM is configured and in fast mode when none is. A
  `GOOGLE_API_KEY`, `GEMINI_API_KEY` or `LLM_API_KEY` in the shell or in `.env`
  counts as a configured LLM, even one exported for another tool, so a plain
  `bibr chew paper.pdf` then sends the paper's text to Google. `--fast` or
  `PIPELINE_MODE=fast` keeps a run on the machine whatever keys are set.

bibr turns off the usage statistics of the vLLM servers it starts
(`VLLM_NO_USAGE_STATS`, unless you set that variable yourself); a vLLM server
you run yourself keeps vLLM's default.
The web demo turns off Gradio's analytics (`GRADIO_ANALYTICS_ENABLED`) unless
you set that variable yourself.
