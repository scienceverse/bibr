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
their corresponding settings for that invocation. Source-checkout users should
prefix the commands on this page with `uv run`.

Set `BIBR_DISABLE_DOTENV=1` to skip both `.env` files entirely. Benchmark
harnesses and CI should do this so every setting a run records came from the
process environment, not from whatever `.env` happened to be in the checkout.

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
`--ocr-url` routes to an external Paddle endpoint by default; specify
`--ocr glm-http` for a GLM endpoint. For example:

```bash
bibr chew paper.pdf --ocr paddle-http --ocr-url https://ocr.example.org
bibr chew paper.pdf --ocr glm-http --ocr-url https://ocr.example.org
```

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
server). `LLM_MODEL` picks the model for that provider.

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
Windows or CUDA cards below 11 GB; and vLLM on Linux/CUDA systems with at
least 11 GB (or when VRAM detection is unavailable).
Explicit choices are `vllm`, `vllm-mlx`, `rapid-mlx`, `llama-cpp`, and `llmster`.
The OCR and LLM choices are independent, so local OCR with a cloud LLM is a
supported hybrid configuration.

Run `bibr setup` to choose the model as well as the runtime. Its recommended
model is NuExtract 3, with runtime-specific weights:

| Runtime | Setup model | Registry memory floor |
|---|---|---|
| llama.cpp | `numind/NuExtract3-GGUF:Q4_K_M` | 5 GB VRAM |
| vLLM | `numind/NuExtract3` (bf16) | 11 GB VRAM |
| Rapid-MLX / vllm-mlx | `numind/NuExtract3-mlx-8bits`, with smaller quantizations available | 6 GB for the 8-bit variant |

These are model-fit estimates, not total pipeline memory guarantees. The
advanced wizard also offers Gemma 4 E4B on CUDA and custom model IDs.
`LLM_LOCAL_MODEL` selects managed local weights; `LLM_MODEL` and `--llm-model`
select the provider model. When `LLM_LOCAL_MODEL` is unset, vLLM, llama.cpp,
and vllm-mlx choose a compatible NuExtract 3 variant from the registry. Rapid-MLX
has a separate `LLM_RAPID_MLX_MODEL` default (`qwen3.5-4b-4bit`) when no local
model was explicitly configured.

`LLM_STRUCTURED_BACKEND=auto` uses Instructor, including with NuExtract 3.
The `nuextract-native` backend is experimental and explicit-only; it is not
automatically enabled by choosing a NuExtract model.

For `--llm llmster`, install LM Studio's `lms` CLI and download the desired model
first, then set `LLM_LLMSTER_MODEL` to its model key. Bibr can start its daemon
and API server and load that existing model. It does not install the runtime
or download a model, and cleans up only resources it started.

Local inference speed depends on the runtime, model, hardware, and document.
Validate the fields you need on representative papers before choosing a model
for a large run. Cloud LLMs or an external OpenAI-compatible server can be used
with local OCR when local LLM throughput is insufficient.

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

Presets are stored as JSON under `~/.bibr/presets/`. Secrets (API keys and
similar) are excluded by default when saving; endpoint URLs and other private
configuration may still be present. You can also apply a preset for a single run
without touching `.env`:

```bash
bibr chew paper.pdf --preset fast-gemini
```

## Reference-extraction strategies

Reference parsing has its own strategy knob, `--refs` (or `REF_PARSE_STRATEGY`):

- **`ner`** (default) — the default `geom` segmentation locates each
  reference with a local geometry model, cascading through layout-region
  anchors, LLM segmentation, and CRF when earlier tiers cannot resolve it.
  A local ModernBERT-CRF model parses the resulting entries. The core install
  supports geometry and ONNX parsing; parsing has no per-reference LLM cost.
  `REF_SEG_STRATEGY=crf` requires the `torch` extra.
- **`llm`** — parses references with the configured LLM in batches of up to
  `REF_PARSE_BATCH_SIZE` entries (default 15), with NER recovery for failed
  batches when available.
- **`llm-chunked`** (experimental) — lets the LLM find reference boundaries and
  fields inside region-aligned chunks instead of requiring one entry per input.
- **`off`** — skips reference extraction entirely (empty bibliography/match/
  citation-link tables) while keeping everything else — titles, authors,
  sections, equations.

`--ref-seg` (or `REF_SEG_STRATEGY`) overrides just the segmentation step
independently of parsing. The full set of strategies and how they cascade is
covered in [Architecture](architecture.md).

## Local model runtime

Four of bibr's models run locally: the PP-DocLayoutV3 layout detector, the
section classifier, the paper classifier and the ModernBERT+CRF reference
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
pinned revision, offline-tolerant through the Hub cache. Layout is the one model
whose PyTorch weights live in a third-party repo, so its ONNX artifact has its
own pair of settings, `LAYOUT_ONNX_MODEL_ID` and `LAYOUT_ONNX_REVISION`.

All four bundles are published, so a core install needs no configuration:

| Model | Repo | Pinned revision |
|---|---|---|
| Layout | `scienceverse/bibr-layout-onnx` | `2bcb16a6` |
| Section classifier | `scienceverse/bibr-section-classifier` | `ee1a83db` |
| Paper classifier | `scienceverse/bibr-paper-classifier` | `6046171b` |
| Reference parser | `scienceverse/bibr-parser-v4-5-gold` | `ff50a83e` |

The three text-model pins moved forward from the previously audited commits to
the commits that add `onnx/`. Those commits are purely additive — no existing
file changed — so the PyTorch path still loads byte-identical weights.

Execution providers come from the same chain the sentence segmenter uses
(CUDA → CoreML → CPU), so `bibr[gpu]` accelerates all four models, not just
segmentation.

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
