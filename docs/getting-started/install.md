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

## Install from PyPI

```bash
uv init --python 3.12 paper-extraction
cd paper-extraction
uv add bibr
uv run bibr setup
uv run bibr doctor
```

For an existing uv project, run `uv add bibr` in that project. In an existing
Python environment, `python -m pip install bibr` installs the same package; use
`bibr setup` and `bibr doctor` directly.

The commands above install core bibr. Add `bibr[all]` for the full cloud + ML
dependency set, or choose the extras below for your use case. Core includes the
`bibr` CLI, cloud-backed LLM extraction, native
DOCX, JATS, HTML, and ePub processing, and bibr's own trained models — layout
detection, the section and paper classifiers, the NER reference parser — served
through **ONNX Runtime**. It is deliberately light: no `torch`, no `transformers`,
no OpenCV, so it installs fast and stays small in containers and CI.

All four models are downloaded from the Hub on first use, as ONNX bundles at the
same pinned revisions the PyTorch weights use — nothing to configure. If a bundle
cannot be resolved, a core install fails with a `ConfigurationError` naming the
model and the fix, and an install that has the `torch` extra transparently falls
back to the PyTorch class instead — see
[`ML_RUNTIME`](../guides/configuration.md#local-model-runtime).

```bash
uv add 'bibr[torch]'       # installed project — PyTorch fallback + training parity
uv sync --extra torch      # source checkout
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
| `torch` | `transformers`, `torch`, `torchvision`, `pytorch-crf`, `opencv-python-headless`, `sentencepiece` | The PyTorch runtime for bibr's local models. Core already runs them through ONNX Runtime; add this for training parity, Apple MPS, `torch.compile` on the serve layout model, transformers-based OCR, OpenCV post-processing, the CRF reference segmenter (`REF_SEG_STRATEGY=crf`), and as the fallback runtime if an ONNX bundle cannot be resolved. |
| `ml` | `bibr[torch]` | Back-compat alias for `torch`. Every existing install, Dockerfile and `bibr setup` plan asks for `ml`, and it keeps meaning the same thing. |
| `local` | `vllm-mlx` on Apple Silicon | Local-runtime dependencies. The default `OCR_BACKEND=paddle` selector uses PaddleOCR-VL-1.6 first; Linux/CUDA can use `paddle-vllm`, while Apple Silicon can use the Paddle MLX candidates. On Linux/CUDA, GPU OCR comes from `paddle-vllm` (part of `vllm`) rather than this extra. |
| `local-mlx` | `vllm-mlx` (macOS, Apple Silicon only) | Pinning the vLLM-MLX local-LLM backend directly on Apple Silicon (`local` covers this automatically on macOS). |
| `vllm` | `vllm==0.27.0`, `openai>=2.54.0,<3` (requires Python <3.14) | The *managed* local vLLM server bibr starts for `--llm local` on Linux/CUDA (the path `bibr setup`'s "fully local" flow configures). |
| `gpu` | `onnxruntime-gpu[cuda,cudnn]` (Linux/Windows) | GPU execution providers for ONNX Runtime — sentence segmentation, and on a core install layout, the classifiers and the NER parser too. It needs one more command after syncing; see [GPU ONNX Runtime](#gpu-onnx-runtime). |
| `cache` | `redis>=5.0.0` | Redis-backed response caching — set `REDIS_URL` to enable it (the result cache is on by default, but only uses Redis once a URL is configured). |
| `demo` | `gradio>=6.15.0` | The interactive Gradio demo app. |
| `batch` | `anthropic>=0.40.0` | Offline processing via the Anthropic Message Batches API — bulk paper-type/OECD labeling and similar bulk LLM workflows. |
| `mcp` | `mcp>=2.2.0,<3` | The `bibr mcp` server — extraction as [Model Context Protocol tools](../guides/mcp.md) for agents (Claude Code, Claude Desktop, any MCP client). |
| `all` | `batch` + `cache` + `demo` + `mcp` + `torch` | The cloud + ML set for a full local-dev setup. Does **not** include `vllm`, `local`, `local-mlx`, or `gpu` — add whichever match your hardware. |

## Choosing serving extras

There are no mutually exclusive extras in the current dependency configuration;
`--all-extras` can resolve them together. Hardware-specific runtimes are still
optional.

The serving extras are still hardware-specific, so most people want a subset:

- **`--extra all` is the superset to reach for.** It bundles `batch` + `cache` + `demo` + `mcp` + `torch` — everything that is useful regardless of hardware.
- Add **`vllm`** on Linux/CUDA (managed vLLM for `--llm local`, and the
  `paddle-vllm` GPU OCR backend), **`local`** or **`local-mlx`** on Apple
  Silicon, and **`gpu`** for GPU-accelerated ONNX sentence segmentation
  (see [GPU ONNX Runtime](#gpu-onnx-runtime)).

If `vllm` isn't installed, bibr's managed vLLM server falls back to launching it
via `uv tool run --from vllm==0.27.0 --with 'openai>=2.54.0,<3' vllm serve ...` in an isolated environment,
instead of your project's virtual environment. That bootstrap downloads several
GB on first use, so it logs a warning naming `uv sync --extra vllm`, and
`bibr setup` adds `vllm` to its Linux/CUDA plan. It only runs on an NVIDIA GPU
with at least 8 GB of VRAM; smaller GPUs and CPU-only Linux machines take the
llama.cpp path (`glm-llama`) for OCR instead. The managed local LLM uses the
same bootstrap. On Python 3.14, where `vllm==0.27.0` has no wheels and the
`vllm` extra therefore installs nothing, both launchers run the bootstrap
inside a managed Python 3.13, and `bibr doctor` says so; a 3.11-3.13
interpreter for the project avoids the detour.

## GPU ONNX Runtime

The `gpu` extra installs `onnxruntime-gpu` next to the core `onnxruntime`
package. They are separate packages that write the same `onnxruntime/`
directory, and when one install writes both, as `uv sync --extra gpu` does,
either build can end up loaded. When the CPU build wins, every ONNX model runs
on the CPU. After syncing, reinstall the GPU build so its files are written
last:

```bash
uv sync --extra gpu
uv pip install --reinstall-package onnxruntime-gpu "onnxruntime-gpu[cuda,cudnn]==1.26.0"
```

`1.26.0` is the version `uv.lock` pins, the last `onnxruntime-gpu` release on
PyPI built for CUDA 12. In a project that installed `bibr[gpu]` from PyPI, use
the version that install chose (`uv pip show onnxruntime-gpu`). With pip, the
command is `python -m pip install --force-reinstall --no-deps
"onnxruntime-gpu==<version>"`. `bibr setup` runs this step itself when it
installs the `gpu` extra.

Leave `onnxruntime` installed too. `uv run` reinstalls it when it is missing,
and its files would then replace the GPU build's. For the same reason, repeat
the reinstall after any `uv sync` that changes either package's version, and
keep `--extra gpu` on every later `uv sync`: a sync without it removes
`onnxruntime-gpu` along with the files the two packages share, and `import
onnxruntime` fails until `uv sync --reinstall-package onnxruntime` restores the
CPU build.

When both packages are installed and the CPU build is the one loaded, bibr logs
a warning that gives the command for your environment.

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
then lists the `.env` files it read and checks keys, dependencies, launchers,
and configured endpoints. For an LLM provider (cloud, Ollama, or your own
OpenAI-compatible server) it sends one short request, built the way extraction
builds it. For a managed local LLM and for local OCR it runs the same launcher
and hardware checks `bibr chew` runs before it starts. It does not start
managed OCR servers or prove local model inference works; use setup's smoke
test and a paper extraction for that. Use `bibr setup --advanced` for the
detailed provider and backend picker.

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
