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

## Native input and PDF text

DOCX, JATS XML, HTML, and ePub are parsed natively and bypass PDF rendering,
layout detection, and OCR. Metadata embedded in JATS and supported HTML/ePub
documents can become the extraction result directly; unstructured metadata and
reference strings may still need the configured downstream models.

For PDFs, `OCR_NATIVE_TEXT_ENABLED=true` is the default. After local layout
detection, usable text-layer content fills eligible regions directly, with OCR
for the remaining regions. This is selective: it does not remove the local
layout dependency or guarantee that a text-layer PDF makes no OCR requests.
Set `OCR_NATIVE_TEXT_ENABLED=false` to force recognition for those text regions.

### Experimental native reconstruction

Native PDF reconstruction keeps stable character indices, bounding boxes, font
metadata and rotation in the inspection record. Set `OCR_NATIVE_REPAIR_ENABLED=true`
to repair damaged native lines and separately recognize owned inline formulas.
`OCR_NATIVE_CAPTIONS_ENABLED=true` additionally accepts unambiguous single-line
captions beginning with a figure/table/chart marker; it requires native repair.
Both options default to false and require `OCR_NATIVE_TEXT_ENABLED=true`.
Ambiguous ownership falls back to region OCR. Repair output is marked unverified;
failed or truncated repairs retain an unresolved marker and the original candidate.

Exports made with `include_regions=True` include `_native_source` character and
raster coverage diagnostics, plus region source IDs and typed native/repair spans.
Unassigned native characters and ink outside layout boxes remain visible even
when layout omits a block. Raster ink includes pictures and rules, so these counts
are review signals, not proof of missing text or source fidelity. Native line
grouping cannot reconstruct an entirely absent text layer. These diagnostics add
memory and cache size proportional to the PDF's native character count.

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
unmodified model response as `_raw_ocr_content` in the opt-in `_regions`
diagnostic payload. The selected backend, model, and profile are included in
`ocr_config` and in the OCR-cache identity, so a cache hit never crosses a
Paddle/GLM or normalizer boundary.

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
Windows or CUDA cards with at most 8 GB; and vLLM on larger Linux/CUDA systems.
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
select the provider model. The bare `LLM_LOCAL_MODEL` default is an MLX model,
so run setup or set a compatible model explicitly on CUDA/Windows. Rapid-MLX
has a separate `LLM_RAPID_MLX_MODEL` default (`qwen3.5-4b-4bit`) when no local
model was explicitly configured.

`LLM_STRUCTURED_BACKEND=auto` uses Instructor, including with NuExtract 3.
The `nuextract-native` backend is experimental and explicit-only; it is not
automatically enabled by choosing a NuExtract model.

For `--llm llmster`, install LM Studio's `lms` CLI and download the desired model
first, then set `LLM_LLMSTER_MODEL` to its model key. Bibr can start its daemon
and API server and load that existing model. It does not install the runtime
or download a model, and cleans up only resources it started.

For a custom `LLM_BASE_URL`, forward model-specific chat-template options with
`LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'` when the runtime supports
that option. Bibr sends this JSON object to the endpoint on every call, including
recovery calls. It is separate from `LLM_REASONING_EFFORT` and is ignored when
using the standard OpenAI endpoint.

If a custom OpenAI-compatible endpoint aborts JSON-constrained decoding, bibr
tries one recovery route with the schema in the prompt and validates the result
in Python. This avoids reusing a failed server grammar; malformed recovery
output still fails validation. Truncation and content-filter responses do not
trigger this fallback.

Reference parsing is configured separately from the metadata model. The default
`--refs ner` uses the local reference parser even when a large LLM is selected.
Use `--refs llm` to have the selected LLM parse reference fields; `--ref-seg`
independently controls how the bibliography is split into entries.

Local extraction speed and quality depend on the runtime, model, and paper.
Rapid-MLX and vllm-mlx default to one concurrent LLM call. Managed llama.cpp
matches its probed slot count (one or two); vLLM retains the configured
`LLM_MAX_CONCURRENCY`. Benchmark representative papers before raising that
setting; additional in-flight calls can increase memory pressure without
improving throughput.

## LLM input size

`LLM_PER_TASK_CONTEXT=true` (the default) gives author and classification tasks
separate context slices. `LLM_TITLE_CONTEXT=true` additionally lets the
title/abstract request omit identified author and affiliation rows between a
unique title and an abstract. This extra pruning is experimental and defaults to
false. Check completeness and response validity on your own papers before
enabling it. Unknown or mixed rows, publication evidence, and abstract text
remain in the experimental slice.
The author request and downstream grounding retain their own evidence. Ambiguous
boundaries and the experimental merged metadata call keep the full input.
Set `LLM_PER_TASK_CONTEXT=false` to send the full context to each metadata task.

`LLM_COMPACT_METADATA_PROMPT=true` selects shorter title/abstract/publication
instructions and response-field descriptions. It preserves the field types,
validation, and abstract-boundary rules. This option also defaults to false;
verify that it preserves the publication fields needed by your workflow. It
does not change the merged metadata contract.
Use the selected contract consistently when preparing fine-tuning data.

The statistical-equation fallback excludes clear citation, date, cross-reference,
and software-version parentheses when the sentence contains no other statistical
signal. Uncertain numeric expressions remain eligible. It packs whole sentences
into batches using an approximate input-token target:

```bash
LLM_EQUATION_BATCH_INPUT_TOKENS=1500
```

This estimates the sentence payload as UTF-8 bytes divided by three, with row
framing overhead; it excludes instructions and the response schema. It is a
packing target, not an exact tokenizer count or a hard context limit. A sentence
larger than the target stays intact in its own batch. Each batch contains at most
10 sentences to limit completion size. Lower the target for smaller context
windows. The existing `EQUATION_LLM_FALLBACK_MIN_REGEX_STATS` paper-level gate
remains optional and defaults to zero (disabled).

`LLM_CITATION_SHORTLIST=true` (the default) reduces the bibliography sent to the
citation-resolution fallback when every citation has usable author/year evidence.
It keeps every reference sharing a cited surname or year, including year-suffix
variants, and references with missing authors or years. Original bibliography IDs
and source order are preserved. Uncertain retrieval keeps the full bibliography;
numeric citation numbers are never treated as bibliography IDs.

Shortlisting requires at least 512 characters and 25% less bibliography text.
Unresolved, omitted, conflicting, or out-of-candidate model answers get one
full-bibliography retry for those citations. Valid first-pass answers are kept.
Both requests count toward LLM usage and rate limits. Existing input caps still
apply. Set `LLM_CITATION_SHORTLIST=false` to retain the previous full-bibliography
request. The shortlist is a retrieval heuristic, so a plausible incorrect answer
can still pass without triggering a retry.

## Presets

If you switch between setups often — cloud vs. local, a fast model for bulk
runs vs. a bigger one for precision — save each as a named preset instead of
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

## PDF memory and concurrency

`PIPELINE_MEMORY_MODE` (or `--memory`) controls model residency:

- `aggressive` unloads models between phases; auto-selected for CUDA cards with
  at most 8 GB VRAM or systems with at most 8 GB RAM.
- `balanced` is the other automatic default and retains layout and segmentation
  models while handing memory between OCR and a managed local LLM as needed.
- `keep_all` retains models for throughput when sufficient memory is available.

Local runs and `bibr serve` process PDFs in windows of eight pages by default.
Each window completes rendering, layout, native text extraction, and OCR before
its page images and temporary crops are released. Text and geometry accumulate
for whole-document parsing, including references that continue across windows.

```bash
PIPELINE_PAGE_WINDOW_SIZE=8
```

Lower this on machines with limited RAM. This controls resident page images per
active file; it does not truncate a document. `PIPELINE_MAX_PAGES` remains the
separate document-length limit. Concurrent files and requests each have their
own window, so also tune `OCR_MAX_CONCURRENT_FILES` and
`PIPELINE_MAX_INFLIGHT_REQUESTS` for available RAM. Requested figure images and
the final text output still grow with document size.

Serve retains shared GPU batching across requests. Balanced and keep-all modes
reuse models across page windows. Aggressive mode releases OCR between windows
so layout can reclaim memory; smaller windows can therefore increase model
reload overhead in that mode.

## Reference-extraction strategies

Reference parsing has its own strategy knob, `--refs` (or `REF_PARSE_STRATEGY`):

- **`ner`** (default) — the default `geom` segmentation locates each
  reference with a local geometry model, cascading through layout-region
  anchors, LLM segmentation, and CRF when earlier tiers cannot resolve it.
  A local ModernBERT-CRF model parses the resulting entries. Both local models
  require `ml`; parsing has no per-reference LLM cost.
- **`llm`** — parses references with the configured LLM in batches of up to
  `REF_PARSE_BATCH_SIZE` entries (default 15), with NER recovery for failed
  batches when available.
- **`llm-chunked`** (experimental) — lets the LLM find reference boundaries and
  fields inside region-aligned chunks instead of requiring one entry per input.
- **`off`** — skips reference extraction entirely (empty bibliography/match/
  citation-link tables) while keeping everything else — titles, authors,
  sections, equations.

`--ref-seg` (or `REF_SEG_STRATEGY`) independently selects `geom` (default),
`region`, `llm`, or `crf`. Set `REF_SEG_LLM_FALLBACK=false` to remove the LLM
tier from automatic geometry/region cascades; this does not disable an explicit
`--ref-seg llm`. Native reference boundaries and structured fields are reused
where available. See [Architecture](architecture.md) for the complete flow.

`--no-llm` disables downstream LLM extraction, equations, and Crossref. The
configured OCR backend still runs when needed. Native metadata and already
structured native references can survive this mode; it does not run NER to
recover unstructured references. `--refs off` also clears native references.

## Enrichment and optional figure analysis

`CROSSREF_ENRICH=true` enables reference matching by default. Matches are kept
separately from extracted bibliography fields. `CROSSREF_CONSOLIDATE=off`
(default) preserves those original fields; `fill` fills missing fields from
accepted matches, and `replace` replaces supplied fields. The CLI override is
`--consolidate` (equivalent to `fill`) or `--consolidate=replace`; set
`CROSSREF_CONSOLIDATE=off` to disable consolidation. Use `--no-crossref` for a run without this
network service.

Figure metadata is exported normally; embedded images require `--figure-images`.
The `FIG_*` settings reserve a future structured figure-analysis feature.
`FIG_EXTRACT=meta` is accepted as a setting but is not implemented: it logs a
warning and does not populate `figure[].analysis`. Keep its default, `off`,
for normal use.
