# Choosing an extraction model

bibr's extraction LLM reads the front page and a few other parts of a paper and
returns titles, authors, affiliations, abstracts, funding and similar fields as
JSON. It does not parse references (see the
[fine-tuned reference model](reference-model.md)) or run OCR.

`bibr setup` picks one of three recommended models for you, based on what your
machine can run:

| Scenario | Recommended model | Needs | Passes, vs Qwen3.8-27B |
|---|---|---|---|
| Linux, NVIDIA GPU (Ampere or newer) with 20 GB or more | **bibr extract 4B**, Qwen3.5-4B fine-tuned for bibr (`scienceverse/bibr-extract-qwen35-4b`) | vLLM, 14.25 GiB free to start | 216/252, −2.8 points [−5.2, −0.8] |
| Linux, NVIDIA GPU (Ampere or newer) with 10–19 GB | **bibr extract 4B W4A16**, the same model with 4-bit weights (`scienceverse/bibr-extract-qwen35-4b-W4A16`) | vLLM, 8 GiB free to start | 218/252; vs the bf16 4B on the same machine (217/252) +0.4 points [−0.8, +1.6] |
| Linux, NVIDIA GPU with 5–9 GB or older than Ampere, or Windows | **bibr extract 4B GGUF**, the same model for llama.cpp (`scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0` from 7 GB, `:Q4_K_M` below) | llama.cpp, which you install; GLM-OCR runs on it too | Q8_0 218/252 on the same OCR as the rows above (a single run); with GLM-OCR, checked on 100 papers (Q8_0) and 48 (Q4_K_M) only |
| No suitable GPU, or you prefer a hosted model | **Claude Haiku 5.5** (`claude-haiku-5-5`) | an Anthropic API key, about $0.004 a paper | 221/252, −0.8 points [−2.4, +0.8] |

The memory is what must be free on the card each time bibr starts the model
(see [configuration](configuration.md) for how bibr sizes it): the model's own
budget (12.25 GiB for the 4B, 6.25 GiB for the W4A16 build, measured on an RTX
3090), plus what vLLM takes beyond it while it starts and 1 GiB for bibr's
other models.

### Defaults by card

`bibr setup` writes the default for your card to `.env`. `--llm local` without
setup resolves the same default from the card it detects: the model, the
runtime, the memory mode and the OCR (a test checks that the two agree on every
row). A card that reports up to 0.25 GiB under a size counts as that size (an
RTX 4060 reports 8,188 MiB, an RTX 5070 12,227 MiB).

| Card (Ampere or newer) | Default from `bibr setup` | Default from `--llm local` without setup | Memory mode | Batch run beside a 0.5 GiB desktop |
|---|---|---|---|---|
| 20 GB or more | 4B (bf16) on vLLM, PaddleOCR-VL on vLLM | the same | balanced, 8 papers a chunk | 20 GB: 48 of 48 papers, 138 an hour; 24 GB: 100 of 100, 161 an hour |
| 12–19 GB | 4B W4A16 on vLLM, PaddleOCR-VL on vLLM | the same | balanced, 8 | 12 GB: 100 of 100, 187 an hour; 12 GB beside a 1.5 GiB desktop: 48 of 48, 188 an hour |
| 10–11 GB | 4B W4A16 on vLLM, PaddleOCR-VL on vLLM | the same | aggressive, 3 | 10 GB: 48 of 48, 85 an hour |
| 7–9 GB | 4B GGUF Q8_0 on llama.cpp, GLM-OCR on llama.cpp | the same | aggressive, 3 | 8 GB: 100 of 100, 141 an hour |
| 5–6 GB | 4B GGUF Q4_K_M on llama.cpp, GLM-OCR on llama.cpp | the same | aggressive, 3 | 6 GB: 48 of 48, 151 an hour |

**These runs emulate the card's memory, not its speed.** Each one ran on an
RTX 3090 whose free memory a separate process cut to what the named card has,
with 0.5 GiB more taken for a desktop (on 24 GB, only the desktop), and with
bibr told the card's size. The papers an hour are that RTX 3090's speed with
that much memory; a real card of that size runs at its own speed, which has
not been measured. Each run used bibr's default memory mode and chunk size
for the card. On the same 100 papers the 8 GB default passed as many as the
24 GB one (86 each, by bibr's per-paper scorer); on 48 papers the 6 GB default
passed 38 to the bf16 4B's 41, a difference within chance at that size.

Cards older than Ampere (compute capability below 8.0: Turing, Volta, Pascal)
get the llama.cpp row for their size at any size, because bibr's vLLM models
are bf16, which vLLM serves from compute capability 8.0. No such card has been
tested; this is a conservative routing rule. A cloud or remote LLM with bibr's
local OCR keeps the balanced mode on cards above 8 GB, since no LLM server
shares the card.

The 4B needs a 20 GB card: on a 16 GB card, bibr's other models soon held more
than the 1.4 GiB it leaves them, and its later starts were refused (that run
predates the release described next and has not been repeated). Below 12 GB
bibr runs in its aggressive memory mode, which unloads its own models before
each start of the LLM. In the balanced mode, when a start would not fit, bibr
first releases the same models and a resident OCR server, and refuses the
start, with an error that says so, only if it is still short. On a 12 GB card
beside a 1.5 GiB desktop it released them before 3 of its 6 starts and finished
the batch. `PIPELINE_MEMORY_MODE=aggressive` makes room in every chunk, at the
cost of speed. **An 8 GB card has too little memory for vLLM**: with a desktop
on it about 7 GiB is free to CUDA, and bibr refused every start of the W4A16
build (it needs 8 GiB free). There setup runs both OCR (GLM-OCR) and the 4B's
GGUF build on llama.cpp, which take turns on the card. If nothing else uses an
8 GB card, you can still run the W4A16 build on vLLM by setting the LLM's share
yourself (`LLM_LOCAL_MEM_FRACTION=0.83`, about 6.3 GiB of 7.6, ran 48 papers
with many ONNX out-of-memory retries); with a desktop on the card the same
setting failed. The 2B student needs at least as much as the W4A16 build on
vLLM (6.25 GiB at 128 sequences). On an emulated 8 GB card with no desktop,
bibr refused every start of it at the default settings; with the LLM's share
set by hand it ran 48 papers. No run at the default settings beside a desktop
backs it, so it is no longer the default for any card.

Windows, NVIDIA cards below 10 GB and NVIDIA cards older than Ampere run the
models through llama.cpp (GGUF), with the largest build that fits. `bibr setup`
recommends a local model only from 5 GB of VRAM, so in practice that is the 4B;
the 2B builds are there for smaller cards through `bibr setup --advanced`.
Apple Silicon runs the 4B, or the 2B on a Mac with less memory, through MLX:

| Runtime | Builds, largest first | Memory |
|---|---|---|
| llama.cpp | 4B Q8_0, 4B Q4_K_M, 2B Q8_0, 2B Q4_K_M (`scienceverse/bibr-extract-qwen35-4b-GGUF`, `-2b-GGUF`) | from 7 GB down to 3 GB of VRAM |
| MLX (Apple Silicon) | 4B 8-bit, 2B 8-bit (`scienceverse/bibr-extract-qwen35-4b-mlx-8bit`, `-2b-mlx-8bit`) | from 6 GB down to 3 GB of memory |

The quantized builds are made from the same merged weights. Served by
llama.cpp on the same 252 papers, the 4B Q8_0 passes 218 (bf16 on vLLM in the
reference run: 216),
the 4B Q4_K_M 214 (216), the 2B Q8_0 209 (207) and the 2B Q4_K_M 204 (207).
These are single runs, and the differences from bf16 have not been tested.

Setup gives Linux cards with 10–19 GB the W4A16 build on vLLM rather than a
GGUF build: on those cards PaddleOCR-VL runs on vLLM too, and bibr can hand
VRAM between two vLLM servers, or between two llama.cpp servers, but not yet
between vLLM and llama.cpp. So an 8 or 9 GB card, whose LLM runs on llama.cpp,
gets GLM-OCR on llama.cpp too, even though PaddleOCR-VL alone fits from 8 GB;
without setup, the automatic OCR does the same whenever the LLM runs on
llama.cpp.
`bibr setup --advanced` offers the 4B Q8_0 build on llama.cpp on larger cards
as well, but it may run out of memory while PaddleOCR-VL and extraction share
the card. The W4A16
build's 218/252 and its paired difference are from one run on vLLM 0.31 on an
RTX 5090, against the published bf16 build on the same machine (217/252). The
4B's 216/252 and its difference from Qwen3.8-27B are from the reference run,
which served the fine-tune as an unmerged LoRA on another GPU. The
MLX builds have only been smoke-tested, so setup marks them experimental. There
are no 4-bit MLX builds: in their smoke tests the 4B's duplicated a funding
entry and the 2B's returned award IDs as numbers.

The hosted setup uses Claude for OCR as well (`OCR_BACKEND=anthropic`,
`OCR_VISION_MODEL=claude-haiku-5-5`), so it needs one Anthropic key. The 221/252
above was measured with bibr's local OCR; Claude's OCR has not been scored on
these papers yet, and the per-paper cost excludes it.

bibr used to recommend [NuExtract 3](https://huggingface.co/numind/NuExtract3)
for llama.cpp and MLX. On the same papers it passes 200/252, −9.1 points
[−12.7, −5.6], behind both fine-tuned models, so bibr no longer offers it.

Run `bibr setup --advanced` to choose another model, or set `LLM_LOCAL_MODEL`
(local) or `LLM_PROVIDER` and `LLM_MODEL` (hosted) yourself; see
[configuration](configuration.md).

## How close small models get to a large one

All scores in this section come from one evaluation: bibr's 0.7.0 prompts and
per-paper scorer on 252 development papers (gate192 plus ci60), with OCR
replayed from cache so only the LLM changes. A paper passes when every scored
field clears its threshold. Each model is paired paper by paper with
Qwen3.8-27B, which passes 223/252 (88.5%), and the brackets are 95% confidence
intervals of the difference. bibr counts a model as matching the 27B when the
lower end of that interval stays above −4 points.

| Model | Passes /252 | vs Qwen3.8-27B, points |
|---|---|---|
| Qwen3.8-27B (reference) | 223 | — |
| Claude Haiku 5.5 (hosted) | 221 | −0.8 [−2.4, +0.8] |
| bibr extract 4B (Qwen3.5-4B, fine-tuned) | 216 | −2.8 [−5.2, −0.8] |
| Qwen3.5-4B, not fine-tuned | 208 | −6.0 [−9.1, −3.2] |
| K2-Horizon-7B, not fine-tuned | 208 | −6.0 [−9.1, −3.2] |
| bibr extract 2B (Qwen3.5-2B, fine-tuned) | 207 | −6.3 [−9.9, −3.2] |
| NuExtract 3 (4.5B) | 200 | −9.1 [−12.7, −5.6] |
| Granite 4.2-8B, not fine-tuned | 195 | −11.1 [−15.1, −7.1] |
| Qwen3.5-2B, not fine-tuned | 190 | −13.1 [−17.5, −9.1] |
| Qwen3.5-0.8B, fine-tuned | 180 | −17.1 [−21.8, −12.7] |

What this shows:

- **Large models barely differ.** In an earlier run on the same 192 gate papers,
  Qwen3.8-27B, Gemini 3.5 Flash Lite, GLM 5.3 Flash, MiMo-V2.6-Pro, DeepSeek V4.1
  Flash and Gemini 3.8 Flash all passed between 79% and 82%. Haiku 5.5 is level
  with the 27B on both runs.
- **Below the 27B, size costs accuracy.** Off the shelf, models of 4 to 8 billion
  parameters trail it by 6 to 11 points and 2B models by about 13. Most of the
  loss is on author lists and non-English papers.
- **Fine-tuning on bibr's tasks closes about half of that gap,** but no small
  open model clears the −4 point rule yet. The fine-tuned 4B is 2.8 points
  behind, closer than any 7–8B model off the shelf. The fine-tuned 2B is 6.3
  points behind, ahead of NuExtract 3 at less than half its size. Below 2B the
  loss grows quickly.
- **Non-English papers are the weak spot.** On the 29 non-English papers, the
  fine-tuned 4B and Haiku 5.5 each trail the 27B by 6.9 points, the 2B by 13.8
  and NuExtract 3 by 20.7. These intervals are wide.

Limits of this evidence:

- These are development papers that were used to choose prompts and models, not
  a sealed test set. Expect smaller margins on your own papers.
- The gate papers' gold labels came from a Gemini model and only part of them
  was checked against the PDF, so they may favour Gemini-like answers.
- Each model ran once. Haiku 5.5 rejects a temperature setting, so its answers
  can vary between runs.
- Speed was measured on one RTX 3090, where the pipeline rather than the model
  sets the pace: the 4B processed 275 papers an hour and the 2B 291. The 2B
  saves memory, not time, on a card like that.

## How the fine-tuned models were made

Both models are Qwen3.5 bases (Apache-2.0) with a LoRA adapter merged into the
weights. Merging keeps the base model's multi-token prediction layers, so the
checkpoints can use speculative decoding like their base.

- **bibr extract 4B** was trained on 1,113 answers that Qwen3.8-27B gave to
  bibr's own prompts on 181 open-access PMC papers.
- **bibr extract 2B** was trained on the same 1,113 answers plus 4,186 author
  lists from JATS XML of 6,526 open-access papers, which made up for most of its
  weakness on authors.

The training papers were filtered by licence. For the 27B answers, the licence
statement in each PDF was read: 181 CC BY papers were kept and 11 with
non-commercial, no-derivatives or no licence statement were dropped. For the
JATS author lists, the licence recorded in bibr's training pool was used: CC BY
or CC0 papers were kept and 474 papers without a recorded licence were dropped.
None of the training papers overlaps the evaluation papers above, matched by
DOI, PMCID and title.
