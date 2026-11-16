# Local model runtime

This page explains how bibr runs models on your own machine: the model servers
it starts, how they share a GPU, which model revisions it pins, speculative
decoding, its ONNX models, and how to try a retrained model. It is reference
material for tuning or debugging a local setup. For the settings most people
need, see [Configuration](configuration.md). For which model runs on which
card, see [Choosing an extraction model](extraction-models.md).

## Managed local servers

The local OCR backends and `--llm local` start their model server as a child
process that listens on `127.0.0.1` at a fixed port. The vLLM and llama.cpp
servers get a new random API key at every launch. bibr passes the key through
the server's environment, not its command line, so other users on the machine
cannot call them. Rapid-MLX, vllm-mlx and MLX-VLM servers run without a key.
`OCR_API_KEY` goes only to an OCR endpoint you configure (`--ocr-url` or
`OCR_BASE_URL`), never to a server that bibr manages.

If the port already has a server that lists the requested model, bibr reuses
it and logs a warning that it did not start that server. A reused server
receives your documents, so stop any server there that you did not start
yourself. Two pipelines in one Python process, such as an open `bibr.Chewer`
and a `bibr.chew()` call, share the vLLM or llama.cpp server that the first one
started; it stops when the pipeline that started it closes. bibr does not share
a server that rejects its unauthenticated probe with HTTP 401 and was not
started by this process, which is usually a server that another bibr process
started with its own key. Stop that server or configure a free port.

The Rapid-MLX OCR backends (`glm-rapid-mlx`, `paddle-rapid-mlx`) restart their
server every `OCR_RAPID_MLX_RECYCLE_AFTER` regions (default 80). This releases
memory that Rapid-MLX's vision cache would otherwise leak. bibr can restart only
a server it started, so these backends refuse a server that is already running
on their port, including one that another pipeline in the same process started.
`paddle-mlx-vlm` shares its port and model with `paddle-rapid-mlx`, so it
refuses a Rapid-MLX or vllm-mlx server it finds there too; an MLX-VLM server is
still reused. To fix this, stop that server, move the port
(`OCR_RAPID_MLX_PORT`, or `OCR_PADDLE_MLX_PORT` for the Paddle backends), or set
`OCR_RAPID_MLX_RECYCLE_AFTER=0` so that the Rapid-MLX backends reuse the server
without restarts. Each refusal is logged as a warning that names the port. The
automatic `paddle` chain reports only which candidates failed, and then tries
the next one.

Each server writes its stderr to a private log file in the system temp
directory, and bibr logs the path when the server starts. A Rapid-MLX server
keeps a single log file for the whole run, across all of its restarts.

The managed llama.cpp servers default to full GPU offload, flash attention,
and `q8_0` KV-cache quantization. OCR uses one parallel slot. The LLM can use
two slots with a unified KV cache and n-gram speculative decoding when the
installed server supports those flags, otherwise it uses one slot. Override
individual flags with
`OCR_LLAMA_CPP_EXTRA_ARGS` / `LLM_LLAMA_CPP_EXTRA_ARGS` (user flags replace the
matching defaults). On older GPUs (Pascal / GTX 10-series) PyTorch may fall
back to CPU for layout/NER while llama.cpp still uses the GPU — that is
expected; install a CUDA build of `llama-server` and skip the `gpu` extra.

The managed llama.cpp server starts the LLM with thinking off
(`--reasoning off`, or `--reasoning-budget 0` on older builds, plus
`--chat-template-kwargs '{"enable_thinking": false}'`) when the installed
build supports it, and `bibr setup` also writes
`LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'` for the fine-tuned models. Rapid-MLX
has a separate `LLM_RAPID_MLX_MODEL` default (`qwen3.5-4b-4bit`) when no local
model was explicitly configured.

`bibr chew`, `bibr batch` and `bibr mcp` stop the managed servers they started
when the command ends, including on Ctrl-C, SIGTERM (`kill`, `docker stop`,
`timeout`, an MCP host stopping `bibr mcp`) and SIGHUP (a closed terminal or SSH
session). A `bibr chew` or `bibr mcp` stopped by a signal exits with 128 plus its
number (143 for SIGTERM) and `bibr batch` exits 130 as after Ctrl-C; in a systemd
unit, set `SuccessExitStatus=130 143` so that `systemctl stop` counts as a clean
stop.

## GPU memory on a shared card

### How the shares are sized

A managed vLLM LLM server and the managed
`paddle-vllm` OCR server each reserve a share of GPU memory. The LLM server
takes at most 0.85 of the card (see below); PaddleOCR-VL takes about 5 GB, so
0.16 of a 32 GB card, 0.21 of 24 GB and 0.32 of 16 GB. A vLLM server starts
only when its whole share is free, and the rest of the card holds bibr's
layout, sentence and reference models, which stay on the GPU between chunks
when `onnxruntime-gpu` is installed. Unless the two shares leave at least 6 GB
of the card for those models (with the LLM at 0.85, only on cards of about
80 GB), bibr stops the server that is running before starting the other: in a
batch, each chunk's OCR stops the LLM server and that chunk's LLM stage starts
it again. The shares are sized from the card that `CUDA_VISIBLE_DEVICES`
exposes; when bibr cannot tell which card that is (cards of different sizes
numbered in CUDA's default fastest-first order), PaddleOCR-VL keeps 0.92 and
the servers take turns. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` or select the card
by UUID to avoid that.

The LLM server works out its share each time it starts, from the GPU memory
free at that moment. It takes at most 0.85 of the card, and less when the card
is busier: the share leaves room for vLLM's brief extra use while it starts
(0.75 GiB for the 4B student's builds, 1 GiB for the 2B) and for bibr's
models. That room is 6 GB less what bibr's models already hold, and at least
1 GB: about 0.71 of a free 24 GB card and 0.78 of a free 32 GB card. When that
does not leave the model its minimum budget (11.0 GiB for the 4B student,
6.0 GiB for its W4A16 build, 6.25 GiB for the 2B; more with
[speculative decoding](#speculative-decoding-on-vllm)), a server that takes
turns with PaddleOCR-VL drops to the minimum while bibr's models keep at least
1 GB, since it starts again beside them in the next chunk. A server that stays
up for the run (with cloud OCR, an OCR server at `--ocr-url` or GLM-OCR) keeps
that room: it usually starts in the first chunk, before bibr's models have
grown to the 5 GB or so they reach in a batch. So on a 24 GB card that also
runs a PaddleOCR-VL server at `--ocr-url` (about 5.4 GB), the 4B student just
fits (it needs 17.25 GiB, and about 17.7 are free) with an 11.3 GiB budget,
below MTP's minimum; with a desktop on the card as well it does not, and its
W4A16 build does. When the share does not fit, the start fails at once with an
error that says how much memory was free and how much it needs, that chunk's
papers fail, and the next chunk tries again. Set `LLM_LOCAL_MEM_FRACTION`, or
`--gpu-memory-utilization` in `LLM_VLLM_EXTRA_ARGS`, to start the LLM server at
exactly that share instead.

### Checks before each chunk

The LLM server starts after a chunk's OCR, and OCR only adds to what bibr
holds. So while the LLM server is down, bibr checks before each chunk's OCR
whether the memory free then could start it. When it could not, the chunk's
papers fail with the same error before any OCR runs, and the check is made
again before the next chunk, which goes ahead once enough memory is free. A
server already listening on the LLM server's port (`LLM_VLLM_PORT`) is never
checked: the start reuses one that serves the model, such as one another
pipeline in the same process started, without asking for memory.

### Minimum budgets

The minimum comes from bibr's model registry, which lists the published
builds with the smallest budget each was measured to start with: a first
start, before vLLM has filled its compile cache, that served one
32,768-token request on an RTX 3090, rounded up to a quarter GiB. The 4B
student's builds were measured with vLLM 0.32.0 at 64 sequences, the 2B with
vLLM 0.31.0 at 128. On vLLM 0.31 a later start reused the cache and needed
about 1 GiB less. The minimum is a budget, not a card size: on the swap path a
start also needs vLLM's extra use while it starts (0.75 GiB at 64 sequences,
1 GiB at 128) and 1 GiB for bibr's models free beside it, so the W4A16 build
needs 7.75 GiB free and the 4B student 12.75 GiB. A server that stays up needs
6 GiB less what bibr's models hold in place of that 1 GiB: 12.25 and 17.25 GiB
while they hold 0.5 GiB. A model the registry does not list, such as a local
path, has no minimum, so a start that cannot fit is launched and fails inside
vLLM. Set `LLM_LOCAL_MIN_VRAM_GB` to give it one. For a model the registry
lists, the setting can raise the minimum but not lower it: a lower value is
ignored and bibr logs a warning once, so a value left over from a local build
cannot start the 4B student below its 11.0 GiB. For a local copy of the 4B
student's W4A16 build, the minimum is 6.0 GiB with vLLM 0.32 and at most 64
sequences (more with 128, since vLLM keeps a cache block for each sequence):

```bash
LLM_LOCAL_MODEL=/path/to/bibr-extract-qwen35-4b-W4A16
LLM_LOCAL_MIN_VRAM_GB=6.0
LLM_VLLM_EXTRA_ARGS='--language-model-only --generation-config vllm --max-model-len 32768 --max-num-seqs 64'
LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
```

### Memory modes on small cards

The memory figures in [Choosing an extraction model](extraction-models.md) are
model-fit estimates, not total pipeline memory guarantees. Setup
matches a card that reports up to 0.25 GiB under a size (an RTX 5070 reports
12,227 MiB) to that size, for the model and for the memory mode; PaddleOCR-VL's
8 GB minimum has no such slack. On a Linux card below 10 GB that could run
PaddleOCR-VL, setup sets `OCR_BACKEND=glm-llama`, so OCR and the LLM both run
on llama.cpp and take turns on the card; without setup, the automatic OCR
chain does the same whenever the LLM runs on llama.cpp. Cards below 12 GB
whose LLM server bibr manages get the aggressive memory mode
(`PIPELINE_MEMORY_MODE=aggressive`, 3 papers a chunk), which unloads bibr's
own models before each start of the LLM; with a cloud or remote LLM, only
cards of 8 GB or less do, and not even those when nothing in the run can load
onto the card (no OCR engine to start, and an install without a CUDA
onnxruntime or torch). A managed llama.cpp LLM server stops before each
start of GLM-OCR on llama.cpp in the aggressive mode, and in the others only
when the memory free cannot also hold GLM-OCR (2.25 GiB) and 1 GiB for bibr's
models. In the balanced mode, when the memory free cannot
start the managed vLLM LLM server, bibr first releases the same models (and a
resident OCR server) and refuses the start only if it is still short.

## Pinned model revisions

Each build of bibr's 4B student in the registry is pinned to a commit on the
Hugging Face Hub ([the list](extraction-models.md#builds-and-revisions)).
vLLM gets the commit as `--revision`. `llama-server -hf` and the MLX servers
(vllm-mlx, Rapid-MLX) always download a repository's `main`, so for the GGUF
and MLX builds bibr downloads the pinned files into the Hugging Face cache
itself (`HF_HUB_CACHE`, or `HF_HOME/hub`; for Rapid-MLX,
`RAPID_MLX_HF_HUB_CACHE` or `RAPID_MLX_HF_HOME/hub` when set) and starts the
server on the local copy, still under the model's Hub id. A build already in
the cache starts with no request to the Hub, so offline runs work once it has
been downloaded. Offline without it, the start stops with the `hf download`
command that fetches it. Another commit of the same repository in the cache,
such as its `main`, does not count: `bibr chew --dry-run` says
`will download revision 0aa228e8` until the pinned one is there.

Each repository's `main` still holds the builds of the 4B's previous revision
(`6ab47a4e`), but `main` will move on, so the commands below name each
build's commit. To run one of them, keep `LLM_LOCAL_MODEL` as setup wrote it
and override the revision, or point `LLM_LOCAL_MODEL` at a local copy:

```bash
# vLLM: the last --revision wins over the registry's.
LLM_VLLM_EXTRA_ARGS='--revision 6ab47a4e4e8b1b76c1ab234aec66f360683ff9fc'   # bf16
LLM_VLLM_EXTRA_ARGS='--revision a2af8f6e90e6481251af9eb07d02c034c30e3446'   # W4A16

# llama.cpp: a local .gguf file is loaded with -m
hf download scienceverse/bibr-extract-qwen35-4b-GGUF bibr-extract-qwen35-4b-Q8_0.gguf --revision 8f5cb6f80329f800f17f3b25dd40be131068f872
LLM_LOCAL_MODEL=/path/to/bibr-extract-qwen35-4b-Q8_0.gguf

# MLX: a local model directory
hf download scienceverse/bibr-extract-qwen35-4b-mlx-8bit --revision 16bf58c9e4f453176cdabf3cfe64665c92a006c3 --local-dir ./bibr-4b-mlx-6ab47a4e
LLM_LOCAL_MODEL=./bibr-4b-mlx-6ab47a4e
```

On vllm-mlx, bibr serves a local MLX directory as a text model, as it does the
pinned build: vllm-mlx alone would take the `image_token_id` in its
`config.json` for a vision model and fail to load it.

A local path is not a registry model, so bibr applies none of the registry's
memory floors or launch arguments to it; the vLLM `--revision` route keeps
them. The 4B's FP8-Dynamic build
(`scienceverse/bibr-extract-qwen35-4b-FP8-Dynamic`, 219/252 on an RTX 5090) is
not in the registry either, so setup does not offer it and it has no measured
memory floor. To try it on vLLM, set `LLM_LOCAL_MODEL` to that id and take
`LLM_VLLM_EXTRA_ARGS` and `LLM_CHAT_TEMPLATE_KWARGS` from the local W4A16
example in [Minimum budgets](#minimum-budgets), with `--revision 101d1ee1d449195f6882019b0fd2aa4328ed8bef`
added to the extra args. That example's `LLM_LOCAL_MIN_VRAM_GB=6.0` was
measured for the W4A16 build, not this one.

## Speculative decoding on vLLM

`LLM_VLLM_SPEC_DECODE` sets speculative decoding for the managed vLLM LLM
server. It applies to the published builds of the 4B student, which bibr's
registry lists with the configs below; other models, local paths included,
start without it unless `LLM_VLLM_EXTRA_ARGS` passes a `--speculative-config`,
which always replaces bibr's choice.

| Value | What the server starts with |
|---|---|
| `auto` (default) | MTP (3 tokens drafted by the model's own multi-token-prediction layers) when the server stays up for the whole run and its budget is at least MTP's minimum: 12.0 GiB for the 4B student, 7.25 GiB for its W4A16 build. Otherwise none. Below the minimum the server starts without MTP; `auto` never refuses a start. |
| `none` | no speculative decoding |
| `mtp` | MTP at every start, including starts that take turns with PaddleOCR-VL. The start needs MTP's minimum: on the swap path 13.75 GiB free for the 4B student and 9 GiB for the W4A16 build; a server that stays up for the run needs 18.25 and 13.5 GiB while bibr's models hold 0.5 GiB. |
| `dflash` | DFlash (7 tokens drafted by [`z-lab/Qwen3.5-4B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash), a separate 1.27 GB model pinned to revision `9a1996cc`), for the 4B student's bf16 build only, with a 13.5 GiB minimum. Other models start without it, with a warning. |

The server stays up for the run when no managed PaddleOCR-VL server takes
turns with it: with cloud OCR, an OCR server at `OCR_BASE_URL`, or OCR on
llama.cpp. With the automatic OCR on a card of up to 64 GB, PaddleOCR-VL and
the LLM take turns, the LLM restarting every chunk, and `auto` leaves MTP off.
Both MTP and DFlash add `--no-async-scheduling`, which every measured run
used.

What was measured, with vLLM 0.32.0 at 64 sequences:

- **One LLM server for the run** (the 252 development papers, OCR read from a
  cache, the server at 0.85 of the card): with MTP the 4B student passed 218
  against 219 without it on an RTX 3090 and 218 against 218 on an RTX 5090;
  its LLM stage took 0.80x and 0.74x as long, and papers an hour rose 5.7%
  and 12.1%. With DFlash: 220 against 219 and 217 against 218, the LLM stage
  0.72x and 0.64x as long, papers an hour +8.1% and +15.8%, but about half the
  KV cache capacity (159,251 against 310,108 tokens at 0.85 of an RTX 3090).
  These runs served the bf16 build at the registry's revision (`9199424d`).
  The previous revision's W4A16 build (`a2af8f6e`) passed 218/252 with MTP,
  as without it on vLLM 0.31; the pinned W4A16 build has not been scored with
  MTP.
- **The default plan on 12 and 16 GB cards** (the previous revision's W4A16
  build taking turns with PaddleOCR-VL, 48 papers, emulated cards beside a
  0.5 GiB desktop): MTP passed 40 and 39 of 48 against 39 and 40 without it,
  but each LLM start took 51 s instead of 37 s, and papers an hour fell 5.0%
  and 8.6%. On the 12 GB card three of its six starts fitted only after bibr
  released its own models. This is why `auto` leaves MTP off there.
- Speculative decoding has not been measured on 24 GB cards in the swap plan,
  on the 2B student, or with DFlash on the W4A16 build.

## bibr's ONNX models

The [configuration guide](configuration.md#local-model-runtime) covers
`ML_RUNTIME` and lists the published bundles and their pinned revisions. This
section has the details.

### Loading a bundle

A bundle resolves from a local directory containing `onnx/`, or holding
`bibr_onnx.json` and `model.onnx` itself (point the model's existing
`*_MODEL_ID` / `NER_PARSER_CKPT` setting at it), or from the Hub at the pinned
revision, offline-tolerant through the Hub cache. A value shaped like a
Hub id (`org/name`) is always fetched from the Hub, even when a directory of
that name exists in the working directory; write a local directory as a path —
absolute, or starting with `./`, `../` or `~`. Layout is the one model
whose PyTorch weights live in a third-party repo, so its ONNX artifact has its
own pair of settings, `LAYOUT_ONNX_MODEL_ID` and `LAYOUT_ONNX_REVISION`.

### Default models and earlier versions

The paper-classifier pin is the 0.9 model, which replaced the PyTorch weights
and the ONNX bundle together; set
`ML_PAPER_CLASSIFIER_REVISION=6046171b3198a255acb1f07f81a586a32f399ac4` to keep
0.8.0's. The reference-parser pin
is the v7 parser (mmBERT-small encoder), published as an ONNX bundle alone: it
runs on ONNX Runtime whatever `ML_RUNTIME` says, as the front-matter tagger
does, and needs no custom code. Set
`NER_PARSER_CKPT=scienceverse/bibr-parser-v5-1-ultrabert` and
`NER_PARSER_REVISION=97b06cc7dd6511bdd53f8855c93a29020231d215` to keep the
0.8.0 parser (UltraBERT-150M), which also has a PyTorch form; its repo carries
the encoder's `config.json`, modeling code and tokenizer, so the PyTorch runtime
loads them from the same pinned commit with `trust_remote_code` and cannot load
it at `NER_PARSER_REVISION=main`.

The geometry line tagger, the text reference segmenter and the statement
tagger are defaults since 0.9. Each one turns off with its setting set to
null (`REF_GEOM_TAGGER_MODEL_ID=null`, `REF_TEXT_SEG_MODEL_ID=null`, which
brings back the CRF segmenter, and `ML_STATEMENT_TAGGER_MODEL_ID=null`), which
reads a paper as 0.8.0 did at that step. As shipped defaults, a model that
cannot be loaded (the Hub out of reach with nothing cached, for instance) does
not stop a run: the paper goes on without it, as described below, and the
log says so. `NER_PARSER_ENCODER` (and
`NER_PARSER_ENCODER_REVISION`) override the encoder for a checkpoint that does
not ship one; an encoder with its own modeling code runs it only from a local
directory or a commit SHA. The section-classifier pin is the model retrained in 0.7.1; set
`ML_SECTION_CLASSIFIER_REVISION=ee1a83db01ce947e3a5eb87dfde7221328f4b207` to
keep the previous one.

### Execution providers

Execution providers come from the same chain the sentence segmenter uses
(CUDA → CPU), so with `bibr[gpu]` these models can run on the GPU too, not
just segmentation. `--device cpu` runs these models on the CPU provider alone, and
`device="cuda:1"` in the Python API puts them on that GPU.

On a Mac these models run on the CPU provider. CoreML is opt-in, because it may
compute in FP16: set a model's own device to `mps`, such as `NER_DEVICE=mps` or
`ML_SECTION_CLASSIFIER_DEVICE=mps`. The layout model stays on the CPU provider
even then, because it fails on CoreML.

### The serve image

`Dockerfile.serve` sets `ML_RUNTIME=torch` explicitly. That image exists for the
PyTorch stack — `torch.compile` on the layout model in particular — and since the
classifier revisions it bakes now also carry an `onnx/` bundle, leaving the
setting unset would let `auto` move serve onto ONNX Runtime the next time the
image is built. Switching it is a deliberate choice, not a build-time accident.

## Running a candidate model

To compare a retrained model with the shipped one, point the shipped model's
own setting at the candidate and run the same papers twice, once without the
setting and once with it. No other setting changes:

| Model | Setting | Candidate format |
|---|---|---|
| Reference parser | `NER_PARSER_CKPT` | ONNX bundle directory (`model: ner_parser`) |
| Geometry reference segmenter | `REF_GEOM_SEG_MODEL_ID` | `geom_segmenter.json` manifest (with its `.npz`), or the directory holding them; `scripts/export_gbm_bundle.py` converts a `.joblib` bundle |
| Geometry line tagger | `REF_GEOM_TAGGER_MODEL_ID` | directory with `gate.json` and `tagger.onnx` + `tagger.json` (beside it or in `onnx/`), or a Hub repo id |
| CRF reference segmenter | `NER_SEG_CKPT` | checkpoint file (`torch` extra) |
| Text reference segmenter (replaces the CRF) | `REF_TEXT_SEG_MODEL_ID` | directory with `segmenter.json`, `model.onnx` and `tokenizer.json` (in it or in `onnx/`), or a Hub repo id |
| Front-role classifier | `ML_FRONT_ROLE_MODEL_ID` | `front_role.json` manifest (with its `.npz`), or the directory holding them; `scripts/export_gbm_bundle.py` converts a `.joblib` bundle |
| Front-matter tagger | `ML_FRONT_TAGGER_MODEL_ID` | ONNX bundle directory |
| Statement tagger (adds a source) | `ML_STATEMENT_TAGGER_MODEL_ID` | ONNX bundle directory (`model: statement_tagger`), or a Hub repo id |
| Section classifier | `ML_SECTION_CLASSIFIER_MODEL_ID` | ONNX bundle directory (`model: section_classifier`) |
| Paper classifier | `ML_PAPER_CLASSIFIER_MODEL_ID` | ONNX bundle directory (`model: paper_classifier`) |

A setting is an *override* when its value, or its revision, differs from the
shipped default. That includes a mirror of a shipped model, an offline copy
of it, and the shipped repo at another revision such as `main`: all of these
are recorded as overrides, and all of them load and run as before. Four rules
apply to overrides only. A run on the shipped models extracts what it did
before; its exports differ in two ways that have nothing to do with these
rules: they name the models the run was set to load in
`extraction.settings.models`, and the text reading joins more of the URLs and
DOIs that a line break split:

- **A model that cannot be loaded stops the run.** Whatever the reason (the
  path is missing; the Hub has no such repo, revision or file; the Hub cannot
  be reached and the files are not in the local Hub cache; a file is
  unreadable or of the wrong kind; the bundle fails its contract check or does
  not run on a sample input; the `torch` extra that `NER_SEG_CKPT` needs is
  not installed), the run stops with an error naming the setting and the
  reason, instead of going on without the model, so a comparison never
  measures the shipped path, or no model, under the candidate's name. `bibr
  chew` exits with status 1. `bibr batch` exits non-zero with `run_info.json`
  `reason` `model_override_failed`, and marks the papers it was running
  `model_override_failed` (a resumed run takes them again) rather than failed,
  so no paper is recorded as `extraction_failed` or `parse_failed` for it. A
  `bibr batch --serve-url` run stops the same way when the server answers
  `model_override_failed`.
- **An error at inference is recorded, and a repeated one stops the run.**
  When a loaded override raises on one paper, the paper takes that tier's
  error path (the path the code takes when that model raises: a failing text
  segmenter, for instance, sends the paper on to the split on printed
  reference markers, not to the CRF a run without the override would use),
  and its export carries a `MODEL_OVERRIDE_INFERENCE_FAILED` processing
  warning naming the setting.
  When the same overridden model raises on three papers in a row on which it
  ran (a paper it is not asked about neither counts nor ends the row; one it
  runs on without an error ends it), the run stops as for a model that cannot
  be loaded. The papers written with the
  fallback before the stop are named in the run's summary and in
  `run_info.json` under `override_fallbacks`, and a resumed `bibr batch` runs
  them again without `--retry-failed`, so once the model is repaired their
  exports hold its answers (a paper that keeps falling back waits for
  `--retry-failed` after three runs). The summary counts the papers the stop
  left unfinished as `not finished`, not as failed. `bibr serve` keeps one
  count per overridden model for the life of the process, across requests:
  the request of the third paper in a row the model raised on answers `500`
  with `error_code` `model_override_failed`, and so does every later one it
  raises on, until a paper it runs on without an error ends the row.
- **Checked first.** `bibr chew` and `bibr batch` load every override this run
  will use (given its mode, `--no-llm`, `--refs` and `--ref-seg`), local or
  Hub, before the first paper, whatever the input (PDF, DOCX, XML or HTML),
  `--dry-run` included. Every overridden ONNX model (the reference parser, the
  text segmenter, the geometry tagger, the front-matter tagger, the section
  and paper classifiers and the statement tagger) is also run at the preflight
  and whenever the run loads it on inputs built as a paper's are (its
  tokenizer, its feature layout, the tokenizer's last id), in the shapes the
  run sends it: the longest window and a short one, and, for the models that
  take a batch, a batch of several and a batch of one. An export fixed to one
  batch size or one length fails there. Each output shape is checked against
  the labels or tags its manifest lists (for the parser and the tagger, also
  the size of the CRF's scores); a model that fails the probe has not loaded.
  A shipped default is not probed. A problem stops the run with one line per
  setting. A setting for a model the run does not ask (the classifiers, the
  statement tagger and the reference models under `--no-llm`, the geometry
  models without `--ref-seg geom`, any reference model under `--refs off`) is
  neither checked nor loaded: the run warns once at the start, naming the
  setting and why, and records it as `not used in this run`.
  `bibr serve` checks the files of every local-path override at startup and
  loads a model when a request first needs it; an override that fails there
  answers `500` with `error_code` `model_override_failed`.
- **Recorded per run.** Each loaded override logs the sha256 of every file it
  read, and `bibr batch` writes one record per override to `run_info.json`
  under `model_overrides`: setting, value, revision (null for a local model),
  `loaded` (`true` with each file's path, size and sha256, or `false` with a
  short `reason` such as `not used in this run` or `torch extra not
  installed`). The key is absent
  for a run on the shipped models. Every export, from `bibr chew`, `bibr
  batch` or `bibr serve`, also names the models its run was set to load in
  `extraction.settings.models`: the setting, the Hub repo id and revision, and
  `override`; for a local model, instead of the path, a short name (the
  directory's own name and `@` the first 12 hex digits of a hash over its
  files, `local-model@...` when that name is a home directory or the user
  name) and the sha256 of each file, so an export shared with others does not
  show how the producing machine is laid out or who ran it.

`REF_GEOM_TAGGER_MODEL_ID` adds a tier to the geometry segmentation rather than
replacing one; the shipped tagger that cannot be loaded leaves the paper to the
shipped geometry segmenter. It takes a local directory or a Hub repo id (with
`REF_GEOM_TAGGER_REVISION`; `gate.json` at the top, `tagger.onnx` and
`tagger.json` in `onnx/`); a Hub-shaped id is never read from the working
directory, so write a local copy as a path (`./org/name`). With
`REF_SEG_STRATEGY=geom` the tagger is asked first, and its answer is taken when
its calibrated paper confidence reaches the threshold in its `gate.json` and
the geometry tier's own checks (alignment yield, segment count against the
layout's reference onsets) pass. An answer it takes is final: the reference
line stream does not replace it. Otherwise, including an error on one paper,
the shipped geometry segmenter and the rest of the cascade run exactly as
without the tagger: a declined answer has no effect on the references. It is
not invisible, though: the export's reference diagnostics list each attempt,
declined or not, in `reference_yield.attempts` as `geom_tagger`, with its
reason (`final`, `low_confidence_or_empty`, `tagger_error`, ...), confidence
and counts in `reason_flags`; a declined attempt carries no spans or credible
starts. An error (`tagger_error`) also adds a `MODEL_OVERRIDE_INFERENCE_FAILED`
warning to the paper, and errors on three papers in a row stop the run. The
tagger reads the same 29 line features as the shipped geometry segmenter, in
sorted name order, and needs only ONNX Runtime; loading it runs the graph on
the features of two short reference sections of different lengths and checks
that it returns two logits per line.

`REF_TEXT_SEG_MODEL_ID` replaces a tier: the CRF segmenter, the last resort of
every segmentation strategy (and the whole of `REF_SEG_STRATEGY=crf`). While it
is set the CRF is not loaded, unless the shipped segmenter cannot be loaded, in
which case the CRF runs in its place. Setting `NER_SEG_CKPT` away from its
default needs `REF_TEXT_SEG_MODEL_ID=null`; with both set the run stops. It takes a local directory or a Hub repo id (with
`REF_TEXT_SEG_REVISION`; `segmenter.json`, `model.onnx` and `tokenizer.json`
in `onnx/`) and needs only ONNX Runtime and `tokenizers`, not the `torch`
extra. The model tags each token of the reference-section text, the same text
the CRF reads, as outside a reference (`O`), opening one (`B-REF`) or inside
one (`I-REF`), in windows and with the stride its `segmenter.json` names; a
token read by two windows keeps the answer of the window it sits furthest
from the edge of. A reference opens at every `B-REF` token and at the first
character of the text, and runs to the next one. Its answer goes through the
same receipt as the CRF's. Unlike the CRF, it answers for a reference
list found by its heading whenever the list is not blank, and does not
decline to the next tier (the split on printed reference markers): when it
finds no reference start at all, the whole list comes back as one reference,
with `no_reference_start` in its `reason_flags` and a `REF_SEG_NOT_SPLIT`
warning on the paper, so one unsplit block is not read as a parsed list. A
list that was only guessed from the layout (`REF_SECTION_INFERRED`) and holds
no reference start is declined instead, with
`inferred_list_without_reference_start` in the receipt: the text is not
exported as a reference, the marker split runs, and the citations in the
paper's notes can stand in for the list. The export's reference
diagnostics list it as `text_tagger`, with its token and window counts, the
references it opened, an uncalibrated `expected_entry_precision` and the
lowest opening probability in `reason_flags`. When an override cannot be loaded, the Hub being out
of reach included, the run stops. An error on one paper is
a `segmentation_error` with a `REF_SEG_CRF_ERROR` warning, as for the CRF, and
a `MODEL_OVERRIDE_INFERENCE_FAILED` warning; errors on three papers in a row
stop the run. The preflight checks the files, the manifest and the tokenizer,
and runs the graph once on a full window of tokenised reference text,
checking that it returns three label scores per token.

`ML_STATEMENT_TAGGER_MODEL_ID` adds a source to statement resolution. It takes a local directory or a Hub repo id (with
`ML_STATEMENT_TAGGER_REVISION`; `bibr_onnx.json`, `model.onnx` and
`tokenizer.json` in `onnx/`) and needs only ONNX Runtime and `tokenizers`.
The model reads the paper's paragraphs in order, each section heading a block
of its own and the reference list left out, each block paired with the first
200 characters of the block before it, and gives each block a probability per
statement type; a block is of a type at or above that type's threshold in
`bibr_onnx.json`, and neighbouring blocks of a type join into one statement.
Where it finds a funding, competing interests, ethics or data availability
statement, that field takes the tagger's statements (all of them, in reading
order, each cut at the next statement label in its rows) instead of the
rules'; a field it finds nothing for keeps the rules' value, and so does a
field whose statements it reads exactly as the rules did (whitespace aside).
Otherwise the field's source in `extraction.fields` is `statement_tagger`.
When the paper has no acknowledgments or author contributions section and
the tagger finds one, its paragraphs become that section: a section all of
whose rows it tagged is retyped, unless its printed heading is one bibr's
heading lookup reads as a body section (an introduction, methods, results,
discussion or appendix heading; the section then stays as it is, and its
rows still give the statement fields their values), otherwise the rows move
to a new section with a synthetic heading right after the one they were in;
`classification_source` is `statement_tagger` either way. Each field or section it supplied adds a
`STATEMENT_TAGGER` warning. A `--no-llm` run does not use it. When the shipped
tagger cannot be loaded, the paper keeps the rules' statements; a bundle that
replaces it is an override, and when that cannot be loaded, the Hub being out
of reach included, the run stops. The preflight of an override
(and `bibr serve` at startup, for a local directory) opens the bundle and runs
the graph on a batch of three blocks that fills the manifest's `max_length`
and holds the tokenizer's last id, and on one short block, checking that it
returns one finite probability per type: a graph with fewer positions than
`max_length`, a vocabulary smaller than the tokenizer's, or an export fixed to
one batch size or one length is refused there. An error on one paper leaves that paper with the rules' statements
and, for an override, a `MODEL_OVERRIDE_INFERENCE_FAILED` warning; errors of an
override on three papers in a row stop the run.

## CPU threads

On the CPU provider every ONNX model (layout, sentence segmenter, reference
parser, line tagger, text segmenter, statement tagger, section and paper
classifiers, front-matter tagger) opens its own
onnxruntime session with its own thread pool. onnxruntime sizes each pool from
the machine's CPU count and ignores a CPU affinity mask (`taskset`, Slurm, a
cpuset container) and a container CPU quota (`docker --cpus`, Kubernetes
limits), so a process pinned to four CPUs on a 16-thread machine still runs
sixteen workers per session. Its idle workers also spin-wait for the next
operator, which keeps cores busy between calls. bibr sizes each CPU session to
the CPUs the process may use and lets idle workers block:

| Setting | Default | Meaning |
|---|---|---|
| `ML_ONNX_THREADS` | `0` | Intra-op threads per CPU session. `0` uses the CPUs this process may run on: the machine's CPU count cut by the affinity mask and by any cgroup CPU quota (rounded up). |
| `ML_ONNX_SPINNING` | `false` | `true` restores onnxruntime's spin-wait. |

Each session gets the whole CPU budget, and sessions can overlap: papers are
parsed and post-processed concurrently, and with a hosted LLM one window's back
half can run beside the next window's layout. Overlaps can briefly oversubscribe
the budget. To run several `bibr batch` processes side by side, pin each one
(`taskset -c 0-3 bibr batch …`) or set `ML_ONNX_THREADS` to its share of the
machine. Neither setting
changes a result: thread count and spinning only change how the same operators
are scheduled. CUDA sessions are not touched.

## CPU memory arenas

onnxruntime gives each CPU session a memory arena that keeps the largest
allocation the session has made for as long as the session lives. Every model
sees inputs of many sizes in a batch, so with arenas a long batch grew from
paper to paper: in a fast batch of 60 papers, the memory held between papers
went from 3.3 GB to 5.3 GB, and the peak reached 7.8 to 9.5 GB (4.2 to 4.8 GB
without arenas). bibr's CPU sessions therefore run without one:

| Setting | Default | Meaning |
|---|---|---|
| `ML_ONNX_CPU_ARENA` | `false` | `true` gives the CPU sessions onnxruntime's arena back. Layout and the sentence segmenter run without one either way: their arenas alone held gigabytes. |

The setting does not change a result, only where tensors are allocated. CUDA
sessions are not touched; they have their own arena, which bibr shrinks after
every run.

## Layout model generation

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
