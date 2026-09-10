# Installation

bibr requires **Python >=3.11,<3.15** (3.11, 3.12, 3.13, or 3.14).

## System prerequisites

bibr identifies input files by content, through the `python-magic` binding to the system
**libmagic** library. `python-magic` is only the binding — on macOS and most Linux distros
libmagic itself has to be installed separately, or the first file bibr reads fails with
`failed to find libmagic`:

| Platform | Command |
|---|---|
| macOS | `brew install libmagic` |
| Debian / Ubuntu | `sudo apt install libmagic1` |
| Fedora / RHEL | `sudo dnf install file-libs` |
| Windows | nothing to do — the `python-magic-bin` wheel bundles it |

`bibr doctor` reports this as its own check, so run it if an extraction fails on a fresh
machine.

## Install the current source

```bash
git clone https://github.com/scienceverse/bibr.git
cd bibr
uv sync --extra all
uv run bibr setup
uv run bibr doctor
```

These docs follow the source repository. This installs the CLI, local ML models'
dependencies, demo, cache, batch, and MCP extras. Model weights download on first
use. Run source-checkout commands with `uv run` so they use this environment.

For a smaller environment, `uv sync --no-default-groups` installs only the core package: the CLI,
cloud-backed LLM extraction, and native DOCX, JATS XML, HTML, and ePub parsers.
Core has no PyTorch dependency. The default local reference parser needs `ml`;
use `--refs llm` or `--refs off` if that model is not installed.

PDF processing needs local layout detection, including when native PDF text can
replace some OCR calls or the selected OCR backend is cloud-hosted:

```bash
uv sync --extra ml
```

DOCX, JATS, HTML, and ePub bypass PDF rendering, layout detection, and OCR. Native
parsing does not imply that downstream LLM extraction is disabled. See the
[quickstart](quickstart.md) for the difference between `--refs off` and `--no-llm`.

Select extras with `uv sync --extra <extra>`; combine repeated `--extra` flags
to retain every extra you need. For an existing installed package, the equivalent
extra syntax is `uv add 'bibr[<extra>]'`.

## Extras

| Extra | Adds | When you need it |
|---|---|---|
| `ml` | `transformers`, `torch`, `torchvision`, `pytorch-crf`, `onnxruntime`, `opencv-python-headless`, `sentencepiece`, `accelerate`, `scikit-learn`, `joblib` | Local ML inference: layout detection, the NER reference segmenter/parser, the trained section classifier, transformers-based OCR, and OpenCV post-processing. |
| `local` | `vllm-mlx` on Apple Silicon | Local-runtime dependencies. The default `OCR_BACKEND=paddle` selector uses PaddleOCR-VL-1.6 first; Linux/CUDA can use `paddle-vllm`, while Apple Silicon can use the Paddle MLX candidates. Needs `ml` too for layout detection and other local pipeline models. On Linux/CUDA, GPU OCR comes from `paddle-vllm` (part of `vllm`) rather than this extra. |
| `local-mlx` | `vllm-mlx` (macOS, Apple Silicon only) | The vLLM-MLX local-LLM backend on Apple Silicon (`local` covers this too). Rapid-MLX uses a separately installed executable; these extras do not install it. |
| `vllm` | `vllm==0.25.1` (requires Python <3.14) | The *managed* local vLLM server bibr starts for `--llm local` on Linux/CUDA (the path `bibr setup`'s "fully local" flow configures). |
| `gpu` | `onnxruntime-gpu[cuda,cudnn]` (Linux/Windows) | GPU-accelerated ONNX runtime for faster sentence segmentation. `onnxruntime-gpu` shares an import name with the core `onnxruntime` package, so after syncing you also need to run `uv pip install 'onnxruntime-gpu[cuda,cudnn]'` to replace the CPU-only build. |
| `cache` | `redis>=5.0.0` | Redis-backed response caching — set `REDIS_URL` to enable it (the result cache is on by default, but only uses Redis once a URL is configured). |
| `demo` | `gradio>=6.15.0` | The interactive Gradio demo app. |
| `batch` | `anthropic>=0.40.0` | Offline processing via the Anthropic Message Batches API — bulk paper-type/OECD labeling and similar bulk LLM workflows. |
| `mcp` | `mcp>=1.28.1` | The `bibr mcp` server — extraction as [Model Context Protocol tools](../guides/mcp.md) for agents (Claude Code, Claude Desktop, any MCP client). |
| `all` | `batch` + `cache` + `demo` + `mcp` + `ml` | The cloud + ML set for a full local-dev setup. Does **not** include `vllm`, `local`, `local-mlx`, or `gpu` — add whichever match your hardware. |

## Choosing serving extras

There are no mutually exclusive extras in the current dependency configuration;
`--all-extras` can resolve them together. Hardware-specific runtimes are still
optional.

The serving extras are still hardware-specific, so most people want a subset:

- **`--extra all` is the superset to reach for.** It bundles `batch` + `cache` + `demo` + `mcp` + `ml` — everything that is useful regardless of hardware.
- Add **`vllm`** on Linux/CUDA (managed vLLM for `--llm local`, and the
  `paddle-vllm` GPU OCR backend), **`local`** or **`local-mlx`** on Apple
  Silicon, and **`gpu`** for GPU-accelerated ONNX sentence segmentation.

If `vllm` isn't installed, bibr's managed vLLM server falls back to launching it
via `uv tool run --from vllm==0.25.1 vllm serve ...` in an isolated environment,
instead of your project's virtual environment. That bootstrap downloads several
GB on first use. Use Python 3.11–3.13 when installing vLLM in the project; its
extra is excluded on Python 3.14. The automatic Paddle OCR selector requires
at least 8 GB of detected NVIDIA VRAM before trying vLLM. Smaller GPUs and
CPU-only Linux machines use `glm-llama`, which requires `llama-server` on `PATH`.
The local LLM has a separate threshold: Windows and CUDA cards with 8 GB or
less select llama.cpp. Run setup to choose a model that matches the runtime.

For example, a Linux/CUDA checkout using vLLM can be installed with:

```bash
uv sync --python 3.13 --extra all --extra vllm
```

## Installing from source (contributors)

```bash
git clone https://github.com/scienceverse/bibr
cd bibr
uv sync --extra all --all-groups
```

`--all-groups` adds the dev and docs dependency groups (tests, linting, mkdocs) on top of the `all` extras. See the [contributing guide](../contributing/evaluation.md) for running the test and evaluation suites.

## Verify your setup

```bash
uv run bibr setup
uv run bibr doctor
```

`bibr setup` starts with a short recommended setup flow. It prefers local/private
processing where your hardware can support it and will ask before installing
extras, using a cloud provider, or pointing at a private server. `bibr doctor`
then checks keys, dependencies, launchers, and configured endpoints. It does
not start managed OCR servers or prove model inference works; use setup's
smoke test and a paper extraction for that. Use `bibr setup --advanced` for
the detailed provider and backend picker.

The recommended local setup writes the automatic local OCR selector:

```bash
# .env
OCR_BACKEND=paddle
```

Do not set global `OCR_MODEL` or `OCR_PROFILE` with this selector: the runtime
needs to choose its ordered Paddle/GLM startup candidate and its matching
profile. For an explicit Paddle HTTP endpoint, set the concrete backend and
served alias together:

```bash
OCR_BACKEND=paddle-http
OCR_BASE_URL=https://ocr.example.org
OCR_MODEL=paddle-ocr-vl-1.6
OCR_PROFILE=paddle
```

For Apple Silicon, the managed Paddle MLX candidates default to
`olragon/PaddleOCR-VL-1.6-8bit`. Custom aliases require `OCR_PROFILE`; see the
configuration guide when overriding a model or backend.
