# Windows and GPU setups

You do not need this page to get started. With nothing configured, bibr runs in
**fast mode**: local models on the CPU, no LLM, no API key, and PDFs read from
their own text layer. That works the same on Windows, macOS and Linux, with or
without a GPU (see the [quickstart](../getting-started/quickstart.md)).

Read on when you want one of these:

- **Scanned PDFs read on your own machine.** A scan has no text layer, so bibr
  needs an OCR model (optical character recognition: a model that reads the
  page image). Local OCR models need an NVIDIA GPU or an Apple Silicon Mac.
- **A local LLM.** Full mode adds an LLM that checks what the local models are
  unsure of. A local LLM keeps the paper on your machine and needs the same
  hardware.
- **Faster fast mode on Linux with an NVIDIA GPU.** The `gpu` extra runs
  bibr's own models on the card ([below](#nvidia-on-linux)).

If you only have a few scans, you can skip local OCR: `--ocr anthropic` sends
the page images to a cloud model, `--ocr-url` points at an OCR server you run,
and the hosted demo at [try.bibr.org](https://try.bibr.org) reads one paper at
a time. See [what leaves your machine](privacy.md) before you choose a cloud
route.

## What bibr runs on your hardware

`bibr setup` detects your GPU (on NVIDIA, through `nvidia-smi`) and, if you
answer no to its first question (fast mode), proposes a plan. For a fully
local plan it picks these runtimes:

| Machine | OCR for scans | Local LLM |
|---|---|---|
| Windows, NVIDIA GPU with 5 GB or more | GLM-OCR on llama.cpp (`glm-llama`) | bibr extract 4B, GGUF build, on llama.cpp |
| Linux, NVIDIA GPU (Ampere or newer), 12 GB or more | PaddleOCR-VL on vLLM (`paddle-vllm`) | bibr extract 4B on vLLM (bf16 from 20 GB, 4-bit W4A16 below) |
| Linux, NVIDIA GPU (Ampere or newer), 10–11 GB | PaddleOCR-VL on vLLM, or GLM-OCR on llama.cpp when setup finds a suitable CUDA `llama-server` | 4B W4A16 on vLLM, or the 4B GGUF build on llama.cpp in that case |
| Linux, NVIDIA GPU with 5–9 GB, or older than Ampere at any size | GLM-OCR on llama.cpp (`glm-llama`) | 4B GGUF build on llama.cpp |
| Apple Silicon Mac | PaddleOCR-VL on Rapid-MLX or MLX-VLM, with GLM-OCR fallbacks | 4B 8-bit MLX build on Rapid-MLX, else vllm-mlx (experimental) |
| No NVIDIA GPU, or one below 5 GB | none recommended: cloud OCR or an OCR server you run | a hosted model (Claude Haiku 5.5) or a server you run |

The GGUF build is Q8_0 on cards from 7 GB and Q4_K_M below. "Ampere or newer"
means compute capability 8.0 or higher (RTX 30-series and later); vLLM serves
bibr's models only from there, so older cards (RTX 20-series, GTX 16- and
10-series) use llama.cpp. No batch on such an older card has been recorded.

When bibr runs the LLM itself on a CUDA card, cards below 12 GB use the
`aggressive` memory mode, which loads one model at a time, and larger cards
use `balanced`. Without a local LLM, only cards of 8 GB or less use
`aggressive`. bibr chooses the mode for each run; `--memory` or
`PIPELINE_MEMORY_MODE` overrides it.

[Choosing an extraction model](extraction-models.md) is the authority on
which model fits which card, how accurate each build is, and how fast the
measured runs were. [Defaults by card](extraction-models.md#defaults-by-card)
has the full table.

## Windows

bibr runs natively on Windows; WSL2 is not needed. All local OCR and LLM work
goes through [llama.cpp](https://github.com/ggml-org/llama.cpp), a separate
program that bibr starts but does not install. vLLM does not run on Windows,
so `bibr setup` does not offer the `vllm` extra there.

### 1. Install bibr

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and bibr
as in the [quickstart](../getting-started/quickstart.md), in a PowerShell
terminal. bibr itself does not need Administrator rights.

### 2. Install llama.cpp

bibr needs `llama-server` (or the unified `llama` program) on your `PATH`.
You have two choices:

- **A CUDA build (faster on NVIDIA).** From the
  [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), download
  both `llama-<version>-bin-win-cuda-12.4-x64.zip` and
  `cudart-llama-bin-win-cuda-12.4-x64.zip`, extract them into one folder, and
  add that folder to your `PATH`. GTX 10-series cards need these CUDA 12.4
  builds; newer cards may use the CUDA 13 zips instead.
- **winget.** `winget install llama.cpp` is the command `bibr setup` prints.
  It installs the Vulkan build, which works on NVIDIA cards but is slower than a
  CUDA build.

If you have both, put the CUDA folder ahead of the winget one in `PATH`, so
`llama-server` resolves to the CUDA build.

Update your NVIDIA driver, open a new PowerShell window so the new `PATH`
applies, and check that the GPU and llama.cpp are visible:

```powershell
nvidia-smi
Get-Command llama-server
llama-server --version
```

bibr also reads the `PATH` saved in your Windows user and system settings, so
it finds a fresh install even in a terminal that was open before it.

Use a recent llama.cpp release. Every llama.cpp run behind bibr's model
defaults used build 11282 (`llama-server --version` prints the build number),
and the models need recent builds. bibr checks the build number only in one
case on Linux ([below](#nvidia-on-linux)).

### 3. Run setup

```powershell
uv run bibr setup
```

Answer no to the first question (fast mode) to get the local plan. On an NVIDIA
card with 5 GB or more it runs GLM-OCR and the bibr extract 4B GGUF build on
llama.cpp, taking turns on the card. The plan shows the model and the memory
mode, and if llama.cpp is not installed yet, setup says so and prints the
install command. The first run downloads
the OCR and LLM weights, several GB.

Use `uv run bibr setup --advanced` to choose each part yourself, for example
local OCR with a hosted LLM. A hosted LLM receives the paper's text.

Setup writes `.env` in the current folder. Run bibr from that folder, or the
run will not see the settings. Then [check your setup](#check-your-setup).

The `gpu` extra (ONNX Runtime for CUDA) also installs on Windows, but setup's
plan leaves it out there, because no run on Windows has been measured with it.
To try it, see [GPU ONNX Runtime](../getting-started/install.md#gpu-onnx-runtime).

## NVIDIA on Linux

On Linux, `bibr setup` chooses Python extras for your card and offers to
install them:

- **`gpu`**: ONNX Runtime's CUDA build, so bibr's own models (layout, sentence
  splitting, the reference parser, the classifiers) can run on the GPU. The plan installs it on
  x86_64 machines with an Ampere or newer card and a driver that supports
  CUDA 12, in fast mode too. It needs one
  extra reinstall step after every `uv sync`; see
  [GPU ONNX Runtime](../getting-started/install.md#gpu-onnx-runtime).
- **`vllm`**: the vLLM runtime for PaddleOCR-VL and the local LLM, on cards that
  run vLLM. See [choosing serving extras](../getting-started/install.md#choosing-serving-extras)
  for the Python versions it supports and what bibr does without it.

On cards below 10 GB, and on cards older than Ampere, OCR and the LLM run on
llama.cpp instead, which you install yourself:

1. Install a CUDA (or Vulkan) build of llama.cpp from the
   [releases](https://github.com/ggml-org/llama.cpp/releases), or build it from
   source with `GGML_CUDA=ON` (see the
   [upstream build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)).
   A CPU-only build works but is very slow.
2. Put `llama-server` on your `PATH`.
3. Run `uv run bibr setup`, answer no to fast mode, and accept the local plan.

On a 10 or 11 GB card (Ampere or newer), setup runs both models on llama.cpp
only when it finds a CUDA build of `llama-server` of build 11282 or newer on
your `PATH`, and says which one it found; otherwise it uses vLLM. After setup
picked llama.cpp there, run plain `bibr chew`, not `--llm local`: that would
start vLLM, which cannot serve the GGUF build. See
[on 10 and 11 GB cards](extraction-models.md#on-10-and-11-gb-cards).

## Apple Silicon

On an M-series Mac, `bibr setup`'s local plan installs the `local` extra
(vllm-mlx) and uses Rapid-MLX instead when the `rapid-mlx` command is on your
`PATH`. The MLX builds of bibr's models have only been smoke-tested, so setup
marks them experimental. A full local extraction takes minutes per paper with
Rapid-MLX and can take much longer with vllm-mlx; setup says which you will get.
Macs with 8 GB of memory or less use the `aggressive` memory mode.

## Check your setup

Run these from the folder that holds your `.env`.

`bibr doctor` checks Python, the `.env` it reads, the run mode, the LLM and
OCR routes that mode needs, the compute device and ONNX Runtime. It processes
no paper. With nothing configured it reports fast mode:

```text
$ uv run bibr doctor
Mode
  ✓ Mode: fast (auto: no model or API key is configured); ...
LLM
  ✓ LLM: not used in fast mode (no provider or key checked)
OCR
  ✓ OCR backend: text-layer — no OCR model; born-digital PDFs are read from
    their embedded text layer. A scanned PDF needs an OCR route: ...
Pipeline
  ✓ Device: cpu
  ✓ ONNX Runtime: CPU build 1.31.0
```

After a local setup, the OCR and LLM lines check llama.cpp, vLLM or MLX. For
llama.cpp, look for these:

- `llama.cpp not installed`: `llama-server` is not on your `PATH`.
- `llama.cpp looks CPU-only (no CUDA/Metal/Vulkan backend)`: replace it with
  a GPU build.
- `Vulkan build on an NVIDIA GPU — a CUDA build is faster`: the winget build;
  switch to the CUDA build as in [step 2](#2-install-llamacpp).

On Linux with the `gpu` extra, the ONNX Runtime line should say `GPU build`.

`bibr chew --dry-run` prints the plan for one run without processing
anything: the mode, the OCR and LLM routes, and the models still to download.
It ends with a **Blockers** section and exit status 1 when the real run would
fail. For example, asking for local OCR on a machine without llama.cpp:

```text
$ uv run bibr chew --dry-run paper.pdf --ocr glm-llama
Plan
  mode       fast: no LLM calls, ...
  ocr        glm-llama · ggml-org/GLM-OCR-GGUF:Q8_0 · cpu
  llm        disabled (fast mode)
  refs       seg geom · parse ner
...
Models
  ...
  ↓ OCR weights: ggml-org/GLM-OCR-GGUF:Q8_0 — will download (~1-2 GB)

Blockers (1)
  ✗ OCR backend cannot start here — glm-llama: llama.cpp is not installed. ...
The real run fails on these; fix them before processing.
```

Once both are clean, process the first three pages of a real paper, then the
whole paper:

```bash
uv run bibr chew paper.pdf --pages 1-3 -o test.json
uv run bibr inspect test.json
```

The first run downloads the model weights and is much slower than later runs.
While it runs, `nvidia-smi` should show `llama-server` (or vLLM) using the
GPU.

## If a local setup is too slow or does not fit

- Close other programs that use the GPU. On small cards a few hundred MB
  decide whether a model starts.
- `--memory aggressive` loads one model at a time, at the cost of speed.
- Keep OCR local and use a hosted LLM: choose it in `bibr setup --advanced`.
- Read scans without any LLM: add an OCR backend to a fast run, for example
  `bibr chew scan.pdf --fast --ocr paddle` (the automatic local OCR selector).
- Point bibr at a GPU server you run: `--ocr-url` for OCR; see
  [Ollama and your own server](configuration.md#ollama-and-your-own-server) for
  an LLM server.
- Go back to fast mode with `--fast`. It needs no GPU at all for PDFs that
  have a text layer.
