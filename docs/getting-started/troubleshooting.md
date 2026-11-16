# Troubleshooting and FAQ

Each entry quotes what you see, says why it happens and how to fix it. Two commands help
with almost everything here:

- `uv run bibr doctor` checks this machine and your configuration: the `.env` files read,
  the run mode, and the LLM and OCR a run would use.
- `uv run bibr chew paper.pdf --dry-run` shows the plan for one run (mode, OCR, LLM, what
  would download) and lists any **Blockers**, without processing anything.

If you are not using uv, drop the `uv run` prefix.

## A scanned PDF stops with "no usable text layer"

```text
  ✗ scan.pdf: no usable text layer: only 0/1 text regions could be read from the PDF text layer (0% < 50% threshold; 1 of 1 pages: no text layer (a scanned page)). This looks like a scanned PDF, or one whose fonts carry no usable text, and needs an OCR route: --ocr anthropic (a cloud model; the page images leave this machine), a GPU OCR server (--ocr paddle starts PaddleOCR-VL, --ocr-url points at one you run) or the hosted demo at try.bibr.org
         A scanned PDF needs an OCR backend; see which can run here with: bibr doctor
```

**Why.** With nothing configured, bibr runs in fast mode and reads a PDF from its text
layer, the text a born-digital PDF carries. A scan is only page images, so there is no
text to read. The command exits with status 1.

**Fix.** Give bibr an OCR backend, one of:

- Upload the paper at [try.bibr.org](https://try.bibr.org).
- On an NVIDIA GPU or Apple Silicon, run `uv run bibr setup`, answer **No** to fast mode,
  and accept a local OCR plan; or try `--ocr paddle` for one run. See
  [Choosing an extraction model](../guides/extraction-models.md) and
  [Windows and GPU setups](../guides/windows-gpu.md).
- Point `--ocr-url` at an OCR server you run ([Deployment](../guides/deployment.md)).
- Use a cloud vision model. The page images then go to that provider. For Claude, set
  `ANTHROPIC_API_KEY`, then run `uv run bibr chew scan.pdf --ocr anthropic` (it uses
  `claude-haiku-5-5` unless `OCR_VISION_MODEL` names another model).

Word, JATS, HTML and ePub files never need OCR.

## "No local OCR runtime can start on this machine"

```text
  ✗ No local OCR runtime can start on this machine for PDF input:
  - glm-llama: llama.cpp is not installed. Install a CUDA (or Vulkan) build of llama.cpp from https://github.com/ggml-org/llama.cpp/releases (extract so `llama-server` is on PATH), or build from source with GGML_CUDA=ON. CPU-only builds work but are very slow for OCR/LLM.
Alternatives: point --ocr-url at an external OCR server, use a cloud vision backend (--ocr gemini|openai|anthropic), or read born-digital PDFs from their text layer without OCR (--ocr text-layer, or OCR_FALLBACK_TO_TEXT_LAYER=auto).
```

**Why.** Only a plain fast run reads PDFs from their text layer by default. Anything that
makes the run a full run, or names an OCR setting, switches PDF reading to an OCR model,
and on a machine without a GPU or llama.cpp none can start. Common causes:

- `--no-llm`. It sounds like the light option but is a *structure-only* full run: no
  authors, references or equations for PDF and DOCX, and it needs an OCR backend.
- `--full`, or `--refs llm`, `--llm` and other LLM options.
- An LLM key or setting in your environment (see the [next entry](#a-key-in-your-environment-switches-the-run-to-full-mode)).
- `OCR_BACKEND`, `OCR_MODEL` or `OCR_PROFILE` set in a `.env` file.

**Fix.** To run without an LLM, use `--fast` (or nothing), not `--no-llm`. To keep a
full run but read a born-digital PDF from its text layer, add `--ocr text-layer`. To
find an OCR setting you did not expect, run `uv run bibr config show --sources`.

## A key in your environment switches the run to full mode

The mode line says `full` although you never ran `bibr setup`:

```text
  mode full · OCR glm-llama (automatic) · llm cloud · memory aggressive
```

and `--dry-run` shows `llm        google/gemini-3.5-flash-lite`.

**Why.** Unless `PIPELINE_MODE` says otherwise, bibr runs in full mode whenever an LLM
is configured, and an LLM key already in your shell counts: `GOOGLE_API_KEY` (or
`GEMINI_API_KEY`), `LLM_API_KEY`, or any of `LLM_PROVIDER`, `LLM_MODEL` and
`LLM_BASE_URL`. With a Google key, the paper's text then goes to Google Gemini. On a
machine without local OCR, the run may also stop with "No local OCR runtime can start".

**Fix.** Pass `--fast` to keep one run local whatever is set. To make fast mode the
default, run `uv run bibr setup` and keep fast mode (it writes `PIPELINE_MODE=fast` to
`.env`), or run `uv run bibr config set PIPELINE_MODE fast`. If you did not mean to set
the variable, remove it from your shell profile. What each mode sends where is in
[What leaves your machine](../guides/privacy.md).

## A run without a key stops with "Google API key required"

```text
  ✗ Google API key required. Set LLM_API_KEY or GOOGLE_API_KEY environment variable.
    Run bibr setup to configure an LLM provider, or pass --fast to run on the local models alone (--no-llm for structural output only).
```

**Why.** You asked for an LLM (`--full`, `--refs llm`, `--llm` …) but none is
configured, so bibr falls back to its default cloud provider, Google, which needs a key.

**Fix.** Drop the option, or pass `--fast`; or configure an LLM with `uv run bibr setup`.
In fast mode the same options are refused outright:

```text
  ✗ Invalid option: --refs llm calls an LLM, which fast mode (--fast) never does. Run in full mode to use the LLM, or drop the option.
```

## `bibr.chew()` fails in Jupyter

```text
RuntimeError: bibr.chew() cannot run inside an active event loop (e.g. Jupyter) — use 'await bibr.achew(...)' instead.
```

**Why.** Jupyter already runs an event loop, and `bibr.chew()` needs to start its own.

**Fix.** In a notebook cell, use the async form:

```python
result = await bibr.achew("paper.pdf")
```

The [Python guide](../guides/library.md#async) covers the async API and warm sessions.

## The first run is slow, or seems to hang

**Why.** The first run downloads about 1.5 GB of models from Hugging Face. bibr lists
them, with their sizes, before the first stage. Each model is fetched when its stage
first needs it, so a stage line (`[2/6] Analyzing layout...`) can sit for a while; in a
terminal a download bar shows under it, and piped output shows nothing until the stage
ends. In our test the whole first run took about a minute and a half; later runs use
the cache.

**Fix.** Wait for it; it happens once. To see what will download before you start, run
`uv run bibr chew paper.pdf --dry-run`: each model is listed as cached or
`will download`, with its size. A full run with a local LLM or OCR model downloads
several GB more, and a managed server can take minutes to start; bibr prints those
start-up lines.

## "You are sending unauthenticated requests to the HF Hub"

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN ...
```

**Why.** Hugging Face says this when models download without an account token. It is
advice, not an error. `bibr chew` and `bibr batch` leave it out unless you pass `-v`;
you can still meet it in Python, in `bibr serve` or in the web demo's log.

**Fix.** Nothing is required. A free Hugging Face token in `HF_TOKEN` gives higher
download rate limits and silences the notice.

## "The web demo needs Gradio" from `bibr demo`

```text
  ✗ The web demo needs Gradio, which is not installed.
    Add the demo extra: uv tool install 'bibr[demo]' if bibr is installed as a tool, or uv add 'bibr[demo]' in a uv project.
```

**Why.** The web demo is an optional extra.

**Fix.** In a uv project, run `uv add 'bibr[demo]'`, then `uv run bibr demo`. With pip:
`python -m pip install 'bibr[demo]'`. See [Web demo](../guides/demo.md).

## `uv` or `bibr`: command not found

**Why.** uv is not installed, or `bibr` is installed inside a project environment that
your shell does not see.

**Fix.** Install uv ([instructions](https://docs.astral.sh/uv/getting-started/installation/)).
In a uv project, run bibr as `uv run bibr ...` from the project folder. If `bibr` is
missing after a `pip install`, check the Python version: on Python 3.15 or 3.10, pip
installs an empty placeholder release (see [Installation](install.md#requirements)).

## Where do my settings come from?

bibr reads `~/.bibr/.env`, then the `.env` in the current folder; environment
variables override both. To see which files exist:

```bash
uv run bibr config path
```

```text
  /path/to/paper-extraction/.env — missing
  ~/.bibr/.env — missing
none found — `bibr config set` will create /path/to/paper-extraction/.env
```

To see every setting that differs from the default, and where it comes from:

```bash
uv run bibr config show --sources
```

On a fresh install this prints `No non-default settings found.` A safe starting
`.env`, with every setting commented out, comes from `uv run bibr config example`.
See [Configuration](../guides/configuration.md) and the
[settings reference](../reference/settings.md).

## How do I see more detail?

- `-v` shows bibr's log while it runs (stage decisions, model loads, fallbacks); `-vv`
  adds debug detail: `uv run bibr chew paper.pdf -o paper.json -v`.
- `uv run bibr inspect paper.json` summarizes a result, with its warnings.
- `uv run bibr doctor` checks the installation and configuration; each line is ok, a
  warning or a failure with the command that fixes it.

## Reporting a problem

Open an issue at [github.com/scienceverse/bibr/issues](https://github.com/scienceverse/bibr/issues/new/choose).
Pick **Wrong extraction** when bibr ran but a field is missing or wrong, and
**Crash or install problem** when it failed. Please include:

1. The exact command and its full output (add `-v` for more).
2. The output of `uv run bibr --version` and `uv run bibr doctor`.
3. Your operating system, and for GPU problems the output of `nvidia-smi`.
4. The paper, or its extracted JSON, if it may be shared.

Issues are public: remove API keys and paths you don't want to share, and attach only
papers that may be shared. Security problems go to the
[private advisory form](https://github.com/scienceverse/bibr/security/advisories/new)
instead.
