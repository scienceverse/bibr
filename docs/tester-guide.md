# bibr 🦫 tester guide

This guide is for colleagues testing bibr on their own computer. The recommended
first test is deliberately conservative: launch the local demo, prove the
pipeline with a short PDF, then add fully local inference once that works.

> **bibr is experimental.** It works, but parts of the pipeline are still being
> validated. Review the extracted JSON against the source paper and report
> missing or incorrect fields; a completed run is not an accuracy guarantee.
>
> Hit a bug or something confusing? Please
> [open an issue](https://github.com/scienceverse/bibr/issues) or email me
> directly at jakub@jakubwerner.com.

## Which setup should I use?

| Setup | Best for | What runs on the laptop? |
|---|---|---|
| **Windows hybrid (recommended first)** | A Windows laptop with a 6 GB NVIDIA GPU | Local OCR and reference ML; a cloud LLM handles metadata |
| **Windows fully local** | Local inference after the hybrid test passes | Quantized OCR and LLM models, loaded one at a time; disable Crossref for offline runs after model downloads |
| **Hosted Gradio demo** | The quickest evaluation, with no installation | Only the browser; the host processes the paper |

### Status of the 6 GB Windows path

bibr now has a native Windows path; WSL2 is not required. On Windows it uses
[llama.cpp](https://github.com/ggml-org/llama.cpp) for GLM-OCR and, optionally,
the local LLM. For a 6 GB GPU, setup selects:

- GLM-OCR Q8 (about 950 MB downloaded) for `glm-llama`;
- NuExtract 3 Q4 GGUF (about 2.7 GB downloaded) for a fully local LLM; and
- `aggressive` memory mode, which unloads one model before loading the next.

The code and automated tests explicitly cover the 6 GB selection, but this is a
new hardware path and has not yet been field-tested on every laptop/driver
combination. Start with **local OCR + a cloud LLM**. If fully local inference is
slow or runs out of memory, that hybrid setup is still a good local installation;
use the hosted demo only if installation itself is the problem.

Allow roughly 15 GB of free disk space for Python packages, model downloads,
and caches. A current NVIDIA driver and at least 16 GB of system RAM are
recommended.

## Windows: RTX A1000 6 GB setup

Use a normal 64-bit PowerShell terminal. Administrator rights may be requested
by `winget`, but bibr itself does not need to run as Administrator.

### 1. Install the tools

Install Git and the GitHub CLI if they are not already present:

```powershell
winget install --id Git.Git -e
winget install --id GitHub.cli -e
```

Install `uv`:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Now install the native llama.cpp runtime. For an NVIDIA GPU, use a CUDA build
from the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases).
`bibr doctor` and server startup checks warn when they detect a Vulkan build
running on NVIDIA hardware.

1. From the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)
   page, download **both** `llama-<ver>-bin-win-cuda-12.4-x64.zip` and
   `cudart-llama-bin-win-cuda-12.4-x64.zip` (`<ver>` is the release tag).
2. Extract both zips into the same folder.
3. Add that folder to `PATH` — **ahead of** any winget-installed llama.cpp
   location if one exists (for example from an earlier revision of this
   guide), so that `llama-server` resolves to the CUDA build.

GTX 10-series (Pascal) cards must use the `cuda-12.4` zips above — the
`cuda-13.x` builds dropped support for that generation.

Alternatively, install the package distributed through winget and inspect the
runtime it provides:

```powershell
winget install llama.cpp
```

Update the NVIDIA driver through NVIDIA or the laptop manufacturer, then close
and reopen PowerShell so the updated `PATH` takes effect. Check that the GPU
and llama.cpp are visible:

```powershell
nvidia-smi
Get-Command llama-server
llama-server --version
```

If `Get-Command` fails after reopening PowerShell, double-check that the
folder containing `llama-server.exe` — the one you extracted the CUDA zips
into, or the winget install location — was actually added to `PATH`. If the
command works but resolves to a leftover winget (Vulkan) install, move the
CUDA folder ahead of it in `PATH`, then re-run `llama-server --version` and
`uv run bibr doctor` — the Vulkan-build warning should disappear.

### 2. Get bibr and install its dependencies

Clone the source repository. If GitHub requests authentication, use the account
with repository access (`gh auth login`):

```powershell
git clone https://github.com/scienceverse/bibr.git
cd bibr
```

Install bibr's local-ML and demo dependencies. `--extra all` includes the
hardware-independent extras; serving runtimes are separate. The current extras
can be resolved together, but Windows does not need the Linux vLLM runtime.

```powershell
uv sync --extra all
```

The optional ONNX GPU package can accelerate sentence splitting. It is not
needed for the first run, and leaving that small stage on the CPU gives the OCR
model more VRAM. Add it later with:

```powershell
uv sync --extra all --extra gpu
uv pip install "onnxruntime-gpu[cuda,cudnn]"
```

### 3. Configure the recommended hybrid setup

Run the wizard from the repository directory:

```powershell
uv run bibr setup --advanced
```

For the first test:

1. Choose a cloud LLM provider, such as Google, and enter its API key. This
   sends document text to that provider for metadata extraction.
2. Choose `glm-llama` for OCR on this Windows machine and `aggressive` memory
   mode. Keep the default NER reference parser. Enter a Crossref email if
   desired; Crossref enrichment is a separate network service.
3. When the wizard offers to install extras, accept — `uv sync --extra all`
   already installed the relevant ones, and the install step uses
   `--inexact`, so accepting again won't remove anything.
4. Accept the offered smoke test on a bundled synthetic paper. It uses one
   page with LLM extraction and references disabled, so the complete test
   below is still needed to validate the cloud LLM.

The shorter `uv run bibr setup` flow recommends a complete configuration,
which can be fully local on a 6 GB GPU. Use it when you want the automatic
recommendation; the advanced steps above deliberately select the hybrid test.

The wizard writes `.env` in the current directory (merging with any existing
one by default). Keep running the commands below from that directory. Check
the resolved environment before processing a paper:

```powershell
uv run bibr doctor
uv run bibr chew --dry-run .\paper.pdf
```

### 4. Open the local demo first

The easiest first run is the drag-and-drop Gradio demo:

```powershell
uv run bibr demo
```

Upload a small paper and confirm that extraction starts. The demo uses the
`.env` settings written by `bibr setup`, including `PIPELINE_MEMORY_MODE` and
`LLM_BACKEND`. Stop the demo with `Ctrl+C` when you are done.

### 5. Run a command-line smoke test

Start with three pages and force the low-memory settings:

```powershell
uv run bibr chew .\paper.pdf --pages 1-3 --ocr glm-llama --memory aggressive -o .\result.json -v
```

The first run downloads model files and will be much slower than later runs.
While it is running, `nvidia-smi` should show `llama-server.exe` using the GPU.
If the three-page test succeeds, process the complete paper:

```powershell
uv run bibr chew .\paper.pdf --ocr glm-llama --memory aggressive -o .\result.json
```

### 6. Try fully local inference (optional)

Run setup again. If the plan preview already recommends a fully local setup
for this GPU, accept it. Otherwise decline and continue into the advanced
flow (or run `bibr setup --advanced` directly) to pick the **local** LLM
explicitly — on this GPU it should select the `llama-cpp` backend and
`numind/NuExtract3-GGUF:Q4_K_M` model. Keep `glm-llama` for OCR. When asked
about the existing `.env`, keep the default (merge) so earlier settings like
the Crossref email carry over.

```powershell
uv run bibr setup
uv run bibr doctor
uv run bibr chew .\paper.pdf --pages 1-3 --ocr glm-llama --llm local --memory aggressive -o .\result-local.json -v
```

OCR and the LLM run in separate llama.cpp processes and are not intended to
occupy VRAM simultaneously. Close other GPU-heavy applications while testing.
If this path fails or is uncomfortably slow, return to the hybrid setup;
OCR and default reference parsing remain local. For a run without Crossref
requests, also pass `--no-crossref`.

## Other platforms

Install [uv](https://docs.astral.sh/uv/) and the
[system prerequisites](getting-started/install.md#system-prerequisites), then
clone and install:

```bash
git clone https://github.com/scienceverse/bibr.git
cd bibr
uv sync --extra all
uv run bibr setup
uv run bibr doctor
uv run bibr demo
```

### Linux, NVIDIA GPU below 11 GB (e.g. GTX 1060)

Below 8 GB, the automatic selectors use **llama.cpp** for OCR and the local
LLM. From 8 GB to below 11 GB (an RTX 3080 10 GB, say), OCR tries vLLM but
the local LLM still uses llama.cpp, because NuExtract 3's vLLM build needs
11 GB of VRAM. Install a
**CUDA** (or Vulkan) build of llama.cpp separately:

1. Install or build llama.cpp with CUDA support (`GGML_CUDA=ON` for a source
   build) and put `llama-server` on your `PATH`. Check the
   [upstream build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
   for your GPU and operating system.
2. Install bibr with the compatible superset only:

```bash
uv sync --extra all
# No additional Python serving extra is needed for llama.cpp.
```

3. Run advanced setup for a cloud LLM plus `glm-llama`, then check the resolved
   configuration. Accept the extras-install offer if dependencies are missing;
   setup installs with `--inexact` to retain other installed extras.

```bash
uv run bibr setup --advanced
uv run bibr doctor
uv run bibr chew paper.pdf --pages 1-3 --ocr glm-llama --memory aggressive -v
```

**About “CUDA problems” on older cards (Pascal / GTX 10-series):** current
PyTorch wheels no longer ship kernels for compute capability 6.x. Layout and
NER then run on **CPU** automatically — that is expected and not fatal. GPU
work for OCR/LLM goes through llama.cpp. Do not try to “fix” this by
installing a different PyTorch inside the venv or enabling the `gpu` extra.

While a job runs, `nvidia-smi` should show `llama-server` using the GPU. If
doctor warns that llama.cpp is CPU-only, replace the binary with a CUDA build.

Optional overrides (already the managed defaults for low-VRAM) live in `.env`:

```bash
# Only needed if you want to change the defaults:
# LLM_LLAMA_CPP_EXTRA_ARGS=--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
# OCR_LLAMA_CPP_EXTRA_ARGS=--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
```

### Which backend, and how fast

`bibr setup` chooses a model and runtime; `--llm local` only resolves the runtime
from the platform and detected memory. The current defaults are:

| Platform | OCR backend | Local LLM backend |
|---|---|---|
| Windows | `glm-llama` (llama.cpp) | llama.cpp |
| Linux, NVIDIA GPU **≥ 11 GB** (e.g. RTX 3090) | `paddle-vllm` | vLLM |
| Linux, NVIDIA GPU **8–<11 GB** | `paddle-vllm` | llama.cpp |
| Linux, NVIDIA GPU **< 8 GB** | `glm-llama` (llama.cpp) | llama.cpp |
| Apple Silicon (M-series) | `paddle-rapid-mlx` (rapid-mlx) | rapid-mlx, else vllm-mlx |
| No suitable GPU | cloud vision OCR or an external OCR server | a cloud LLM |

The setup wizard defaults to [NuExtract 3](https://huggingface.co/numind/NuExtract3)
for local structured extraction. Windows and GPUs below 11 GB use llama.cpp;
the CUDA vLLM variant requires Linux and at least 11 GB. These thresholds are
model-fit estimates, not guarantees that every document fits available memory.

Throughput depends on the model, backend, hardware, and document. Scanned PDFs
require more OCR than PDFs with usable embedded text. Test a representative
sample before starting a large batch, and inspect both metadata and references.

If fully local inference does not run, or is too slow, walk down this ladder:

1. try the llama.cpp build (`--ocr glm-llama`, and `--llm local` on Windows or a
   small GPU already selects llama.cpp);
2. set an API key for a cloud LLM (Google, OpenAI, Anthropic, …) and keep OCR
   local — inexpensive and reliable; or
3. self-host an open-source LLM on a GPU box and point bibr at it — OSS models
   run much better on a real GPU than on a laptop.

## Using bibr

Process one PDF, DOCX, JATS XML, HTML, or ePub paper, or a directory of supported
files. Native non-PDF formats bypass rendering, layout detection, and OCR:

```bash
uv run bibr chew paper.pdf -o result.json
uv run bibr chew papers/ -o results/
```

Useful options include:

- `--pages 1-5` to make a quick, bounded test;
- `--refs ner` (default) parses references locally with no per-reference LLM
  cost. Segmentation defaults to `geom`, with region/LLM/CRF fallbacks;
- `--refs llm` parses reference fields with the configured LLM. `--ref-seg`
  selects segmentation independently; `--refs llm-chunked` is experimental;
- `--refs off` skips references and their Crossref enrichment while retaining
  title, authors, and other metadata extraction;
- `--no-equations` and `--no-crossref` disable those stages;
- `--no-llm` skips downstream LLM extraction and Crossref. OCR still uses the
  selected backend, and already structured native metadata may remain;
- `--ocr paddle-http --ocr-url https://ocr.example.org` uses an external Paddle OCR
  server; use `--ocr glm-http` for a GLM endpoint;
- `--regions` includes OCR/native-source diagnostics for inspecting a mismatch;
- `--figure-images` includes embedded figure images; and
- `--memory aggressive` for low-memory machines.

The Python API is available inside the same project environment:

```python
import bibr

result = bibr.chew("paper.pdf")
result.title
result.references.df
result.save("result.json")
```

In Jupyter, use `await bibr.achew(...)` instead.

## Gradio demo

Launch a local drag-and-drop interface with:

```bash
uv run bibr demo
```

The demo uses `PIPELINE_MEMORY_MODE` and `LLM_BACKEND` from the `.env` created
by `bibr setup`; pass `--memory aggressive` or `--llm llama-cpp` only when you
want to override that setup for this run.

Uploads are limited to `DEMO_MAX_FILE_SIZE_MB` (default 10). Uploaded papers
and JSON downloads are deleted an hour after they were made, and when the demo
stops; set `DEMO_CACHE_TTL_SECONDS` to change the hour, or to `0` to keep them.

For a remote demo, run it on the host that has bibr configured. Protect it with
a username and a strong password before creating a temporary public share link.
In PowerShell:

```powershell
$env:GRADIO_USERNAME="supervisor"
$env:GRADIO_PASSWORD="replace-with-a-long-random-password"
uv run bibr demo --share --memory aggressive
```

Send the printed HTTPS link and password separately. The host downloads the
models and processes uploaded papers; the supervisor only needs a browser.
Treat a Gradio share link as a temporary evaluation endpoint, stop it after the
session, and do not use it for confidential papers unless the hosting and data
handling arrangements have been approved.

## Updating

From the cloned repository:

```bash
git pull
uv sync --extra all
uv run bibr doctor
```

The `.env` file is not touched by an update. Repeat any hardware extras you
installed, for example `uv sync --extra all --extra vllm` on Linux/CUDA.

## Reporting issues

Found a bug, or something that looks wrong? Please
[open an issue](https://github.com/scienceverse/bibr/issues) or email me directly
at jakub@jakubwerner.com. To help me reproduce it, include:

1. The exact command and full error output.
2. `uv run bibr doctor` and `uv run bibr --version` output.
3. `nvidia-smi` output for GPU problems.
4. The input document, if it can be shared.
5. Whether the three-page hybrid test and fully local test succeeded.

## Integration with Metacheck

[Metacheck](https://github.com/scienceverse/metacheck) is the R package bibr feeds
into. Follow its [installation and usage guide](https://www.scienceverse.org/metacheck/articles/metacheck.html)
for the current package version and use a version supporting bibr's exported
schema. Compatibility depends on both projects' versions.

Extract a paper with bibr, then read the JSON in R:

```bash
uv run bibr chew paper.pdf -o paper.json
```

```r
paper <- metacheck::read("paper.json")
```

Use the [report reference](https://www.scienceverse.org/metacheck/reference/report.html)
for report formats and output options supported by your installed version.

## Uninstalling

Delete the cloned `bibr` directory. Model caches are stored separately by
Hugging Face/llama.cpp and can be removed independently if no other local-model
tools use them.
