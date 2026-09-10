# Quickstart

Install the current source, configure your backends, and extract a first paper.
The first run may download several GB of models.

## 1. Install

```bash
git clone https://github.com/scienceverse/bibr.git
cd bibr
uv sync --extra all
```

This includes the local layout/reference models' dependencies and the optional
demo. The setup wizard offers any additional runtime extras your hardware needs;
see [Installation](install.md) for prerequisites and smaller installs.

## 2. Configure once

```bash
uv run bibr setup
```

The wizard detects your hardware, then shows a plan — OCR backend, LLM, extras, memory mode, privacy, and expected speed — for one confirmation. Decline it and the wizard offers to switch into `bibr setup --advanced` for exact provider/backend control instead. Two common outcomes:

- **Capable hardware (Apple Silicon, or an NVIDIA GPU with enough VRAM):** the plan uses local OCR and a managed local LLM. On Apple Silicon, the OCR selector tries Paddle Rapid-MLX before Paddle MLX-VLM, with GLM startup fallbacks. Windows and small CUDA GPUs use GLM-OCR through llama.cpp. The preview shows the requested backends, model, and memory mode. Crossref enrichment remains a separate network service; use `--no-crossref` to disable it.
- **Weaker hardware:** the wizard asks whether you have a private OCR/LLM server; if not, it offers cloud processing via {{ default_llm_provider }}, with an explicit consent step (document content leaves your machine) and an API key prompt.

It writes `.env`, saves a preset, and offers to install the extras the plan
needs. Validation includes an optional connection/server check and a one-page
smoke test on a bundled synthetic paper. That smoke test disables downstream
LLM extraction and references; process your own paper next to check metadata.

The recommended local setup writes the Paddle-first automatic selector;
cloud and private-server plans write their chosen backend instead:

```bash
OCR_BACKEND=paddle
```

It intentionally leaves `OCR_MODEL` and `OCR_PROFILE` unset so the selected
Linux or Apple-Silicon candidate can retain its own model/profile. For an
explicit external Paddle endpoint, use `OCR_BACKEND=paddle-http` with
`OCR_MODEL=paddle-ocr-vl-1.6` and `OCR_PROFILE=paddle`; custom aliases also
require `OCR_PROFILE`.

Use `--ocr paddle-vllm` for explicit Paddle OCR on Linux/CUDA. Use
`--ocr glm-rapid-mlx` (Apple Silicon), `--ocr glm-llama`, or `--ocr glm-http`
when you deliberately want GLM-OCR. (The older `glm-mlx`
backend is disabled: vllm-mlx produced corrupted OCR text.) On Linux the
automatic selector only tries `paddle-vllm` on an NVIDIA GPU with at least 8 GB
of VRAM; smaller GPUs and CPU-only machines go straight to llama.cpp
(`glm-llama`), which needs `llama-server` on your `PATH`.
The automatic `paddle` selector falls back only while starting an OCR runtime;
it does not retry failed individual regions with GLM.

## 3. Chew a paper

```bash
uv run bibr chew paper.pdf -o paper.json
```

You get schema-versioned JSON (currently v{{ schema_version }}) with title,
authors, affiliations, DOI, sections, sentences, tables, figure metadata, and
parsed references. Add `--figure-images` to embed figure images.

The same command accepts DOCX, JATS XML (`.xml`), HTML (`.html`/`.htm`), and ePub
files. These formats are parsed natively without OCR. PDF text layers also supply
usable text directly; regions needing recognition still use the OCR backend.
Point the command at a directory to process supported files in that directory:

```bash
uv run bibr chew papers/ -o out/
```

## Common variations

| Goal | Command |
|---|---|
| Keep metadata, skip references and their enrichment | `uv run bibr chew paper.pdf --refs off` |
| Skip downstream LLM extraction and Crossref | `uv run bibr chew paper.pdf --no-llm` |
| Parse reference fields with the configured LLM | `uv run bibr chew paper.pdf --refs llm` |
| Try only the first three PDF pages | `uv run bibr chew paper.pdf --pages 1-3` |
| Preview resolved settings without processing | `uv run bibr chew paper.pdf --dry-run` |
| Check your setup | `uv run bibr doctor` |

`--no-llm` leaves the configured OCR route active, including cloud vision OCR if
selected. For PDFs and DOCX it primarily produces document structure; structured
metadata and references already parsed from native formats can still be retained.
It is different from `--refs off`, which preserves normal metadata extraction.

Every flag is documented in the generated [CLI reference](../reference/cli.md).

## Next steps

- [Installation matrix](install.md) — extras for local OCR, GPU, caching.
- [Python library](../guides/library.md) — `bibr.chew()` in your own code.
- [Production deployment](../guides/deployment.md) — `bibr serve`, Docker, sizing.
