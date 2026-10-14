# Choosing an extraction model

bibr's extraction LLM reads the front page and a few other parts of a paper and
returns titles, authors, affiliations, abstracts, funding and similar fields as
JSON. It does not parse references (see the
[fine-tuned reference model](reference-model.md)) or run OCR.

`bibr setup` picks one of three recommended models for you, based on what your
machine can run:

| Scenario | Recommended model | Needs | Passes, vs Qwen3.8-27B |
|---|---|---|---|
| Linux, NVIDIA GPU with 12 GB or more | **bibr extract 4B**, Qwen3.5-4B fine-tuned for bibr (`scienceverse/bibr-extract-qwen35-4b`) | vLLM, about 12 GB VRAM | 216/252, −2.8 points [−5.2, −0.8] |
| Linux, NVIDIA GPU with 8–11 GB | **bibr extract 2B**, Qwen3.5-2B fine-tuned for bibr (`scienceverse/bibr-extract-qwen35-2b`) | vLLM, about 8 GB VRAM | 207/252, −6.3 points [−9.9, −3.2] |
| No suitable GPU, or you prefer a hosted model | **Claude Haiku 5.5** (`claude-haiku-5-5`) | an Anthropic API key, about $0.004 a paper | 221/252, −0.8 points [−2.4, +0.8] |

Windows and NVIDIA cards below 8 GB run the same two models through llama.cpp
(GGUF), with the largest build that fits. Apple Silicon runs the 2B through
MLX:

| Runtime | Builds, largest first | Memory |
|---|---|---|
| llama.cpp | 4B Q8_0, 4B Q4_K_M, 2B Q8_0, 2B Q4_K_M (`scienceverse/bibr-extract-qwen35-4b-GGUF`, `-2b-GGUF`) | from 7 GB down to 3 GB of VRAM |
| MLX (Apple Silicon) | 2B 8-bit (`scienceverse/bibr-extract-qwen35-2b-mlx-8bit`) | 3 GB of memory |

The quantized builds are made from the same merged weights. Served by
llama.cpp on the same 252 papers, the 4B Q8_0 passes 218 (bf16 on vLLM: 216),
the 4B Q4_K_M 214 (216), the 2B Q8_0 209 (207) and the 2B Q4_K_M 204 (207).
These are single runs, and the differences from bf16 have not been tested. The
2B MLX build has only been smoke-tested, so setup marks it experimental. There
is no MLX build of the 4B yet: the 4-bit build duplicated a funding entry in its
smoke test.

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

All numbers on this page come from one evaluation: bibr's 0.7.0 prompts and
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
