# Installation

## Requirements

| What | Needed? | Notes |
|---|---|---|
| Python 3.11, 3.12, 3.13 or 3.14 | Yes | With uv you don't need to install it yourself: `uv init --python 3.12` fetches Python 3.12 for the project. |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Recommended | The commands on this site use it. Without uv, `python -m pip install bibr` in a Python 3.11–3.14 environment installs the same package. |
| Disk space | Yes | About 0.7 GB for the Python environment, plus about 1.5 GB of models that bibr downloads from Hugging Face on first use. |
| An NVIDIA GPU or Apple Silicon | Optional | Only for a local LLM or local OCR. Fast mode, the default, runs on the CPU. |

!!! note "Python 3.15 is not supported yet"
    onnxruntime, which bibr's models run on, publishes no Python 3.15 wheels. On 3.15
    (or 3.10), `pip install bibr` cannot install a current release: it fails, or picks
    the empty early placeholder 0.0.2, after which `bibr` is not a command. The uv
    commands below avoid this, because they pin the project to Python 3.12.

## Install from PyPI

```bash
uv init --bare --pin-python --python 3.12 paper-extraction
cd paper-extraction
uv add bibr
uv run bibr doctor
```

For an existing uv project, run `uv add bibr` in that project. In an existing
Python environment, `python -m pip install bibr` installs the same package; then run
`bibr` directly instead of `uv run bibr`.

This installs core bibr: the `bibr` command, the Python library, native DOCX, JATS,
HTML and ePub parsing, LLM extraction through any OpenAI-compatible endpoint (OpenAI,
Groq, Ollama, a local model server), and bibr's own trained models (layout
detection, sentence splitting, the section and paper classifiers, the front-matter
and reference models). None of them needs PyTorch: the neural models run through
**ONNX Runtime**. Core has no `torch`, no `transformers` and no OpenCV, so it installs
quickly and stays small in containers.

The models download from Hugging Face the first time a run needs them, at pinned
revisions; there is nothing to configure. `bibr chew paper.pdf --dry-run` lists each
one, whether it is cached, and its size. [Local model runtime](../guides/model-runtime.md)
explains the ONNX and PyTorch runtimes.

The Google and Anthropic SDKs are extras: `google` for `LLM_PROVIDER=google`
(what a full run without a `.env` uses) and the `gemini` OCR backend, `anthropic`
for `LLM_PROVIDER=anthropic` and the `anthropic` OCR backend. `bibr setup` installs
the one the provider you choose needs. A run that needs a missing one stops before
any work and names the extra, and so do `bibr doctor` and `--dry-run`.

There is no setup step: with nothing configured, bibr runs in fast mode (local models,
no LLM, no API key). `bibr setup` is optional and adds an LLM or OCR; see the
[Quickstart](quickstart.md#5-optional-more-accuracy-with-an-llm).

## Extras

Extras add optional dependencies. In a uv project, add one with
`uv add 'bibr[<extra>]'` (several: `uv add 'bibr[demo,mcp]'`); with pip,
`python -m pip install 'bibr[<extra>]'`; in a source checkout, `uv sync --extra <extra>`.

| Extra | Adds | When you need it |
|---|---|---|
| `torch` | `transformers`, `torch`, `torchvision`, `pytorch-crf`, `opencv-python-headless`, `sentencepiece` | The PyTorch runtime for bibr's local models. Core already runs them through ONNX Runtime; add this for training parity, Apple MPS, `torch.compile` on the serve layout model, transformers-based OCR, OpenCV post-processing, the CRF reference segmenter (`REF_SEG_STRATEGY=crf`), and as the fallback runtime if an ONNX bundle cannot be resolved. |
| `ml` | `bibr[torch]` | Older name for `torch`; it means the same thing. |
| `local` | `vllm-mlx` on Apple Silicon | Local OCR and LLM serving on Apple Silicon. On Linux with an NVIDIA GPU, local serving comes from the `vllm` extra instead. |
| `local-mlx` | `vllm-mlx` (macOS, Apple Silicon only) | Pinning the vLLM-MLX local-LLM backend directly on Apple Silicon (`local` covers this automatically on macOS). |
| `vllm` | `vllm==0.32.0`, `openai>=2.54.0,<3` (Linux x86_64, Python <3.14) | The *managed* local vLLM server bibr starts for `--llm local` on Linux/CUDA, and the `paddle-vllm` GPU OCR backend. |
| `gpu` | `onnxruntime-gpu[cuda,cudnn]` (Linux/Windows) | Runs bibr's ONNX models (layout, sentence splitting, the classifiers, the reference parser) on an NVIDIA GPU. It needs one more command after installing; see [GPU ONNX Runtime](#gpu-onnx-runtime). |
| `serve` | `fastapi`, `uvicorn[standard]`, `python-multipart` | The `bibr serve` HTTP API. Without it, `bibr serve` exits with an install hint. |
| `cache` | `redis>=5.0.0` | Redis-backed response caching for `bibr serve`; set `REDIS_URL` to enable it. |
| `demo` | `gradio>=6.15.0` | The web demo, `bibr demo`. See [Web demo](../guides/demo.md). |
| `google` | `google-genai>=1.0.0`, `jsonref>=1.1.0` | Google Gemini: `LLM_PROVIDER=google` (the provider a full run without a `.env` uses) and `OCR_BACKEND=gemini`. |
| `anthropic` | `anthropic>=0.40.0,<1` | Anthropic Claude: `LLM_PROVIDER=anthropic` and `OCR_BACKEND=anthropic`, the hosted route `bibr setup` recommends. |
| `batch` | `bibr[anthropic]` | Offline processing via the Anthropic Message Batches API. The same SDK as `anthropic`, under the name existing installs ask for. |
| `nuextract` | `numind>=0.3,<0.3.1` | The experimental `LLM_STRUCTURED_BACKEND=nuextract-native` backend for NuExtract servers. bibr no longer installs numind otherwise, and that backend stops with an error naming this extra. |
| `mcp` | `mcp>=2.2.0,<3` | The `bibr mcp` server: extraction as [Model Context Protocol tools](../guides/mcp.md) for agents. |
| `all` | `anthropic` + `batch` + `cache` + `demo` + `google` + `mcp` + `serve` + `torch` | Everything useful regardless of hardware. Does **not** include `vllm`, `local`, `local-mlx` or `gpu`; add whichever match your hardware. |

## Choosing serving extras

Most people need no extra at all. For a local LLM or local OCR, `bibr setup` picks and
offers to install the extras your hardware needs. To choose by hand:

- **`all` is the superset to reach for.** It bundles `anthropic` + `batch` + `cache` + `demo` + `google` + `mcp` + `serve` + `torch` — everything that is useful regardless of hardware.
- Add **`vllm`** on Linux/CUDA, **`local`** or **`local-mlx`** on Apple Silicon, and
  **`gpu`** for ONNX Runtime's GPU build.

Without the `vllm` extra, bibr's managed vLLM server launches vLLM through
`uv tool run --from vllm==0.32.0 --with 'openai>=2.54.0,<3' vllm serve ...` in an isolated
environment instead. That downloads several GB on first use and logs a warning naming
the extra. On Python 3.14, where the `vllm` extra installs nothing, this is always the
route, inside a managed Python 3.13, and `bibr doctor` says so. Which GPU runs what, and
the llama.cpp route for smaller cards and Windows, are in
[Choosing an extraction model](../guides/extraction-models.md) and
[Windows and GPU setups](../guides/windows-gpu.md).

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
installs the `gpu` extra. Its recommended plan includes the extra on Linux x86_64
with an Ampere or newer NVIDIA card, and leaves it out elsewhere.

Leave `onnxruntime` installed too. `uv run` reinstalls it when it is missing,
and its files would then replace the GPU build's. For the same reason, repeat
the reinstall after any `uv sync` that changes either package's version, and
keep `--extra gpu` on every later `uv sync`: a sync without it removes
`onnxruntime-gpu` along with the files the two packages share, and `import
onnxruntime` fails until `uv sync --reinstall-package onnxruntime` restores the
CPU build.

When both packages are installed and the CPU build is the one loaded, bibr logs
a warning that gives the command for your environment. `bibr doctor` shows
which build loads, CPU or GPU, and its version on its ONNX Runtime line.

## Check your setup

```bash
uv run bibr doctor
```

`bibr doctor` lists the `.env` files it read, the run mode, and the LLM and OCR a run
would use, and checks keys, dependencies and configured endpoints. On a fresh install it
reports fast mode:

```text
Mode
  ✓ Mode: fast (auto: no model or API key is configured); makes no model calls, ...
```

For a configured LLM provider it sends one short request. For a managed local LLM and
local OCR it runs the same launcher and hardware checks `bibr chew` runs before it
starts, but it does not start them; `bibr setup`'s one-page test and a real extraction
show that inference works.

## Updating

In a uv project:

```bash
uv sync --upgrade-package bibr
uv run bibr --version
uv run bibr doctor
```

`uv sync --upgrade-package bibr` moves bibr to the newest release that your
`pyproject.toml` allows, and installs it. Extras you added with `uv add 'bibr[...]'` are
in `pyproject.toml`, so they stay. If you use the `gpu` extra, repeat the
`onnxruntime-gpu` reinstall from [GPU ONNX Runtime](#gpu-onnx-runtime) afterwards.

With pip:

```bash
python -m pip install --upgrade bibr
```

An update does not touch your `.env`, your presets or the downloaded models. A new
release can pin new model revisions, which then download on the next run.

In a source checkout: `git pull`, then the same `uv sync` line you installed with,
including every `--extra` you used (a sync without `--extra gpu` removes
`onnxruntime-gpu`).

## Uninstalling

Remove bibr from a uv project with `uv remove bibr`, or delete the project folder; with
pip, `python -m pip uninstall bibr`. That leaves three things behind, each safe to delete:

| What | Where | Remove with |
|---|---|---|
| Downloaded models | The Hugging Face cache (`~/.cache/huggingface/hub`, or `$HF_HOME/hub`). bibr's own models are the `scienceverse/*` repositories; `bibr chew paper.pdf --dry-run` lists every model it uses. | `uv run hf cache ls` lists the cache and `uv run hf cache rm model/<repo>` removes one repository (run them before removing bibr), or delete the folders by hand. Other tools may share this cache. |
| Your settings and presets | `~/.bibr/` (a user-wide `.env` and `presets/`), and any `.env` in your project folders | Delete them. |
| Optional caches | `~/.cache/bibr/` (or `$XDG_CACHE_HOME/bibr/`): the OCR and LLM response caches, which exist only if you turned on `CACHE_OCR` or `CACHE_LLM` | Delete the folder. |

Tools you installed separately for a local LLM or OCR, such as llama.cpp, are not part
of bibr; uninstall them the way you installed them.

## Installing from source (contributors)

```bash
git clone https://github.com/scienceverse/bibr
cd bibr
uv sync --extra all --all-groups
```

`--all-groups` adds the dev and docs dependency groups (tests, linting, mkdocs) on top of the `all` extras. See the [development setup](../contributing/setup.md) for running the tests and [evaluation](../contributing/evaluation.md) for the evaluation suites.
