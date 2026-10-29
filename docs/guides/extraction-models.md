# Choosing an extraction model

bibr's extraction LLM reads the front page and a few other parts of a paper and
returns titles, authors, affiliations, abstracts, funding and similar fields as
JSON. It does not parse references (see the
[fine-tuned reference model](reference-model.md)) or run OCR.

`bibr setup` picks one of three recommended models for you, based on what your
machine can run:

| Scenario | Recommended model | Needs | Passes /252 |
|---|---|---|---|
| Linux, NVIDIA GPU (Ampere or newer) with 20 GB or more | **bibr extract 4B**, Qwen3.5-4B fine-tuned for bibr (`scienceverse/bibr-extract-qwen35-4b`) | vLLM, 12.75 GiB free to start | 218; the previous revision 217 on the same machine, +0.4 points [−0.8, +2.0] |
| Linux, NVIDIA GPU (Ampere or newer) with 10–19 GB | **bibr extract 4B W4A16**, the same model with 4-bit weights (`scienceverse/bibr-extract-qwen35-4b-W4A16`) | vLLM, 7.75 GiB free to start | 218, level with the bf16 4B on the same machine: +0.0 points [−1.6, +1.6] |
| Linux, NVIDIA GPU with 5–9 GB or older than Ampere, or Windows; also 10–11 GB when setup finds a CUDA build of llama.cpp ([below](#on-10-and-11-gb-cards)) | **bibr extract 4B GGUF**, the same model for llama.cpp (`scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0` from 7 GB, `:Q4_K_M` below) | llama.cpp, which you install; GLM-OCR runs on it too | Q8_0 218, +0.0 points [−1.2, +1.2] against the bf16 4B; Q4_K_M 217, −0.4 [−2.0, +0.8]; both on the same OCR as the rows above. With GLM-OCR, the previous revision was checked on 100 papers (Q8_0) and 48 (Q4_K_M) only |
| No suitable GPU, or you prefer a hosted model | **Claude Haiku 5.5** (`claude-haiku-5-5`) | an Anthropic API key, about $0.004 a paper | 221, −0.8 points [−2.4, +0.8] against Qwen3.8-27B's 223 |

The 4B's scores are from bibr commit e491ed64 on an RTX 5090, with OCR replayed
from cache and pointer answers off (see
[below](#pointer-answers-llm_pointer_outputs)); each build is paired paper by
paper with the bf16 build of the same revision, and the bf16 build with the
previous revision of the 4B (`6ab47a4e`). Haiku's score is from the evaluation
in [the evaluation below](#how-close-small-models-get-to-a-large-one). The current
4B has not been run against Qwen3.8-27B; its previous revision passed 216,
−2.8 points [−5.2, −0.8], in that evaluation.

The memory is what must be free on the card each time bibr starts the model
(see [configuration](configuration.md) for how bibr sizes it): the model's own
budget (11.0 GiB for the 4B, 6.0 GiB for the W4A16 build, measured on an RTX
3090 with vLLM 0.32), plus what vLLM takes beyond it while it starts and 1 GiB
for bibr's other models. That is when PaddleOCR-VL takes turns with the model,
as in the defaults below. A model that stays up for the run beside cloud OCR or
an OCR server at `--ocr-url` leaves bibr's other models 6 GiB less what they
hold instead: about 17.25 GiB free for the 4B and 12.25 GiB for the W4A16
build. Speculative decoding needs more and is off in these default plans (see
[speculative decoding](configuration.md#speculative-decoding-on-vllm)).

### Defaults by card

`bibr setup` writes the default for your card to `.env`. `--llm local` without
setup resolves the same default from the card it detects: the model, the
runtime, the memory mode and the OCR (a test checks that the two agree on every
row). The one exception is a 10–11 GB card where setup finds a CUDA build of
llama.cpp ([below](#on-10-and-11-gb-cards)): setup looks on your PATH, the runtime
never does. A card that reports up to 0.25 GiB under a size counts as that size (an
RTX 4060 reports 8,188 MiB, an RTX 5070 12,227 MiB).

| Card (Ampere or newer) | Default from `bibr setup` | Default from `--llm local` without setup | Memory mode | Batch run beside a 0.5 GiB desktop |
|---|---|---|---|---|
| 20 GB or more | 4B (bf16) on vLLM, PaddleOCR-VL on vLLM | the same | balanced, 8 papers a chunk | 20 GB: 48 of 48 papers, 138 an hour; 24 GB: 100 of 100, 161 an hour |
| 12–19 GB | 4B W4A16 on vLLM, PaddleOCR-VL on vLLM | the same | balanced, 8 | 12 GB: 100 of 100, 187 an hour; 12 GB beside a 1.5 GiB desktop: 48 of 48, 188 an hour; 16 GB: 48 of 48, 161 an hour |
| 10–11 GB | 4B GGUF Q8_0 and GLM-OCR on llama.cpp when setup finds a CUDA `llama-server` of build 11282 or newer; otherwise 4B W4A16 on vLLM, PaddleOCR-VL on vLLM | 4B W4A16 on vLLM, PaddleOCR-VL on vLLM | aggressive, 3 | 10 GB, W4A16: 48 of 48, 85 an hour; the llama.cpp pair: see [below](#on-10-and-11-gb-cards) |
| 7–9 GB | 4B GGUF Q8_0 on llama.cpp, GLM-OCR on llama.cpp | the same | aggressive, 3 | 8 GB: 100 of 100, 141 an hour |
| 5–6 GB | 4B GGUF Q4_K_M on llama.cpp, GLM-OCR on llama.cpp | the same | aggressive, 3 | 6 GB: 48 of 48, 151 an hour; 6 GB beside a 1.0 GiB desktop: 48 of 48, 144 an hour |

**These runs emulate the card's memory, not its speed.** Each one ran on an
RTX 3090 whose free memory a separate process cut to what the named card has,
with 0.5 GiB more taken for a desktop (on 24 GB, only the desktop), and with
bibr told the card's size, and they served the previous revision of the 4B's
builds, whose files have the same sizes. The papers an hour are that RTX 3090's speed with
that much memory; a real card of that size runs at its own speed, which has
not been measured. Each run used bibr's default memory mode and chunk size
for the card, with the `gpu` extra installed: bibr's layout model and
sentence segmenter ran on the GPU, and so did the NER reference parser in
the balanced rows (the aggressive mode keeps it on the CPU). The classifiers
stayed on the CPU in every row. `bibr setup` installs the extra on Linux for
every row here. On the same 100 papers the 8 GB default passed as many as the
24 GB one (86 each, by bibr's per-paper scorer); on 48 papers the 6 GB
default passed 38 to the bf16 4B's 41, a difference within chance at that
size.

The 16 GB run (2026-10-26, on this release's code) had at least 4.74 GiB to
spare at every start of the LLM (free memory beyond the 8 GiB a start needs).
The 6 GB default has almost none: beside a 1.0 GiB desktop it finished, but the
card fell to 2 MiB free and the sentence segmenter fell back to the CPU 107
times. No run covers a 6 GB card beside a desktop above 1 GiB.

#### On 10 and 11 GB cards

On an RTX 3090 cut to 10 GB beside a 0.5 GiB desktop, the W4A16 build on vLLM
finished 48 papers at 85 an hour: every chunk of 3 papers restarts two vLLM
servers, PaddleOCR-VL and the LLM. The 4B GGUF Q8_0 with GLM-OCR, both on
llama.cpp, finished the same 48 papers at 148 an hour and passed 40 of them to
W4A16's 39, a difference within chance. That run kept the LLM up between
chunks, which left bibr's own models about 2 GiB while OCR ran. bibr now stops
it before each chunk's OCR in the aggressive mode, as the 8 GB runs of the same
pair did in every chunk after the first (150 papers an hour over 48 papers, and
141 over the 100 papers in the table above); no 10 GB run has used that rule
yet.

So on Linux and an Ampere or newer card of 10–11 GB, `bibr setup` runs both
models on llama.cpp when it finds a CUDA build of `llama-server` (or of the
unified `llama` binary) on your PATH that can start CUDA, with a `--version`
build of 11282 or newer, the build behind every llama.cpp batch in the table
above. It prints the server and its build, and `bibr setup --advanced` gives
you the W4A16 build instead. A server it finds but does not use (a Vulkan, HIP
or CPU build, a CUDA build whose CUDA cannot start on your driver, or an older
build) gets one line saying why, and setup keeps W4A16. If you later remove or
replace that llama.cpp build, rerun `bibr setup`. `--llm local` without setup
never looks at PATH and keeps W4A16. After setup picked the llama.cpp pair,
run plain `bibr chew` (or `--llm llama-cpp`), not `--llm local`: that would
start vLLM with the GGUF build setup wrote, which vLLM cannot serve.

Neither choice has been run on 10 GB beside a desktop larger than 0.5 GiB. By
arithmetic from the W4A16 run, whose starts found 8.76 GiB free, its starts
(7.75 GiB on vLLM 0.32, 8.00 on vLLM 0.31) would be refused beside a desktop
above about 1.5 GiB (1.26 GiB on vLLM 0.31). The llama.cpp pair, stopping the
LLM for OCR, would have as much memory free beside a 2.5 GiB desktop as the
8 GB runs had beside 0.5 GiB, with the same models and settings: an inference
from those runs, not a run.

Cards older than Ampere (compute capability below 8.0: Turing, Volta, Pascal)
get the llama.cpp row for their size at any size, because bibr's vLLM models
are bf16, which vLLM serves from compute capability 8.0. No batch on such a
card has been recorded; this is a conservative routing rule. A cloud or remote
LLM with bibr's local OCR keeps the balanced mode on cards above 8 GB, since no
LLM server shares the card.

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
build (it needed 8 GiB free on vLLM 0.31, and needs 7.75 on vLLM 0.32).
There setup runs both OCR (GLM-OCR) and the 4B's GGUF build on llama.cpp,
which take turns on the card. If nothing else uses an 8 GB card, you can
still run the W4A16 build on vLLM by setting the LLM's share yourself
(`LLM_LOCAL_MEM_FRACTION=0.83`, about 6.3 GiB of 7.6, ran 48 papers with many
ONNX out-of-memory retries); with a desktop on the card the same setting
failed. The 2B student needs more than the W4A16 build on vLLM (6.25 GiB at
128 sequences, measured on vLLM 0.31). On an emulated 8 GB card with no
desktop, bibr refused every start of it at the default settings; with the LLM's share
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

The quantized builds are made from the same merged weights. Each 4B build
was paired paper by paper with the 4B's bf16 build (218/252) on the same 252
papers, at bibr commit e491ed64 on an RTX 5090, one run each:

| 4B build | Runtime | Passes /252 | vs bf16, points |
|---|---|---|---|
| bf16 | vLLM 0.31 | 218 | — |
| W4A16 | vLLM 0.31 | 218 | +0.0 [−1.6, +1.6] |
| FP8-Dynamic (not offered by setup) | vLLM 0.31 | 219 | +0.4 [−0.8, +1.6] |
| GGUF Q8_0 | llama.cpp b11783 | 218 | +0.0 [−1.2, +1.2] |
| GGUF Q4_K_M | llama.cpp b11783 | 217 | −0.4 [−2.0, +0.8] |

Each build stays within the −4 point rule of the bf16 build. On the same
machine the previous revision's Q8_0 passed 217. The 2B's builds were scored
only in the evaluation below: served by llama.cpp, its Q8_0 passes 209 and its
Q4_K_M 204 (bf16 on vLLM: 207). These are single runs, and the differences
from bf16 have not been tested.

Setup gives Linux cards with 10–19 GB the W4A16 build on vLLM rather than a
GGUF build: on those cards PaddleOCR-VL runs on vLLM too, and bibr can hand
VRAM between two vLLM servers, or between two llama.cpp servers, but not yet
between vLLM and llama.cpp. So an 8 or 9 GB card, whose LLM runs on llama.cpp,
gets GLM-OCR on llama.cpp too, even though PaddleOCR-VL alone fits from 8 GB;
without setup, the automatic OCR does the same whenever the LLM runs on
llama.cpp.
`bibr setup --advanced` offers the 4B Q8_0 build on llama.cpp on larger cards
as well, but it may run out of memory while PaddleOCR-VL and extraction share
the card. The 4B's 216/252 and its
difference from Qwen3.8-27B belong to its previous revision and come from the
reference run, which served the fine-tune as an unmerged LoRA on another GPU.
The MLX builds have only been smoke-tested, so setup marks them experimental. There
are no 4-bit MLX builds: in their smoke tests the 4B's duplicated a funding
entry and the 2B's returned award IDs as numbers.

### Builds and revisions

bibr pins each 4B build to a commit on the Hugging Face Hub, so an update to a
repository does not change what a bibr version runs. The current builds are on
each repository's `pointer-v1` branch; `main` still holds the previous
revision's builds.

| 4B build | Revision bibr uses | How bibr gets it |
|---|---|---|
| bf16 | [`9199424d`](https://huggingface.co/scienceverse/bibr-extract-qwen35-4b/tree/9199424d4bd200feb247e26e4a860b63da73cbaf) | vLLM downloads it (`--revision`) |
| W4A16 | [`a32f2048`](https://huggingface.co/scienceverse/bibr-extract-qwen35-4b-W4A16/tree/a32f20485181574d849cfb32f8eac1f266c4143d) | vLLM downloads it (`--revision`) |
| GGUF Q8_0, Q4_K_M | [`0aa228e8`](https://huggingface.co/scienceverse/bibr-extract-qwen35-4b-GGUF/tree/0aa228e8c908512ad5e5e13fa55e7e7d3180c7d3) | bibr downloads the one file into the Hugging Face cache; llama.cpp loads it |
| MLX 8-bit | [`432b8cd8`](https://huggingface.co/scienceverse/bibr-extract-qwen35-4b-mlx-8bit/tree/432b8cd80ea920919ef26c48401473c6d8c8a3e1) | bibr downloads it into the Hugging Face cache; vllm-mlx or Rapid-MLX serves it |
| FP8-Dynamic | [`101d1ee1`](https://huggingface.co/scienceverse/bibr-extract-qwen35-4b-FP8-Dynamic/tree/101d1ee1d449195f6882019b0fd2aa4328ed8bef) | not in bibr's registry; see [configuration](configuration.md#pinned-model-revisions) |

llama.cpp (`-hf`) and the MLX servers always download a repository's `main`,
which is why bibr downloads those builds itself. Once a build is in the cache,
it starts without the Hub, so offline runs work. The
[configuration guide](configuration.md#pinned-model-revisions) shows how to run
the previous revision.

### Pointer answers (`LLM_POINTER_OUTPUTS`)

The current 4B was also trained on bibr's pointer answers for its two
front-matter calls: the title and keywords call names the abstract's first and
last words instead of writing the abstract out, and the authors call lists each
affiliation once. With `LLM_POINTER_OUTPUTS=true` it also passes 218/252
(+0.4 points [−0.8, +1.6] against the previous revision with the flag off), and
the longer of those two calls writes 300 output tokens a paper instead of 455.
The flag stays off by default, for two reasons. First-author accuracy, which
the pass rule does not include, is lower with the flag on: −0.015 [−0.031,
−0.003], 7 papers worse and none better. And on 3 papers bibr's pointer
resolver took a wrong span of the paper text as the abstract without any
warning, a resolver bug that the previous revision hits too.

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
| bibr extract 4B, previous revision `6ab47a4e` (Qwen3.5-4B, fine-tuned) | 216 | −2.8 [−5.2, −0.8] |
| Qwen3.5-4B, not fine-tuned | 208 | −6.0 [−9.1, −3.2] |
| K2-Horizon-7B, not fine-tuned | 208 | −6.0 [−9.1, −3.2] |
| bibr extract 2B (Qwen3.5-2B, fine-tuned) | 207 | −6.3 [−9.9, −3.2] |
| NuExtract 3 (4.5B) | 200 | −9.1 [−12.7, −5.6] |
| Granite 4.2-8B, not fine-tuned | 195 | −11.1 [−15.1, −7.1] |
| Qwen3.5-2B, not fine-tuned | 190 | −13.1 [−17.5, −9.1] |
| Qwen3.5-0.8B, fine-tuned | 180 | −17.1 [−21.8, −12.7] |

The current 4B revision was not part of this evaluation. On a later bibr
commit (e491ed64) it passed 218 to the previous revision's 217 on the same
machine, +0.4 points [−0.8, +2.0].

What this shows:

- **Large models barely differ.** In an earlier run on the same 192 gate papers,
  Qwen3.8-27B, Gemini 3.5 Flash Lite, GLM 5.3 Flash, MiMo-V2.6-Pro, DeepSeek V4.1
  Flash and Gemini 3.8 Flash all passed between 79% and 82%. Haiku 5.5 is level
  with the 27B on both runs.
- **Below the 27B, size costs accuracy.** Off the shelf, models of 4 to 8 billion
  parameters trail it by 6 to 11 points and 2B models by about 13. Most of the
  loss is on author lists and non-English papers.
- **Fine-tuning on bibr's tasks closes about half of that gap,** but no small
  open model clears the −4 point rule yet. The fine-tuned 4B (previous
  revision) is 2.8 points behind, closer than any 7–8B model off the shelf. The fine-tuned 2B is 6.3
  points behind, ahead of NuExtract 3 at less than half its size. Below 2B the
  loss grows quickly.
- **Non-English papers are the weak spot.** On the 29 non-English papers, the
  fine-tuned 4B (previous revision) and Haiku 5.5 each trail the 27B by 6.9 points, the 2B by 13.8
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
checkpoints can use speculative decoding like their base (see
[speculative decoding](configuration.md#speculative-decoding-on-vllm) for when
bibr turns it on).

- **bibr extract 4B** was trained on 1,444 rows from 162 open-access PMC
  papers: 1,113 answers that Qwen3.8-27B gave to bibr's own prompts, which its
  previous revision (`6ab47a4e`) was trained on, plus 331 of the title and
  keywords and authors answers restated as pointer answers. A pointer row was
  kept only when bibr's pointer resolver turned it back into the 27B's answer,
  ignoring whitespace (and, in 7 title/keywords rows, punctuation).
- **bibr extract 2B** was trained on the same 1,113 answers plus 4,186 author
  lists from JATS XML of 6,526 open-access papers, which made up for most of its
  weakness on authors.

The training papers were filtered by licence. For the 27B answers, the licence
statement in each PDF was read: CC BY and CC0 papers were kept and 11 with
non-commercial, no-derivatives or no licence statement were dropped. The 4B's
training rows come from 162 of the kept papers and its validation rows from 19
others. For the
JATS author lists, the licence recorded in bibr's training pool was used: CC BY
or CC0 papers were kept and 474 papers without a recorded licence were dropped.
None of the training papers overlaps the evaluation papers above, matched by
DOI, PMCID and title.
