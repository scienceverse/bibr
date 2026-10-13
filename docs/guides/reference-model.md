# Fine-tuned reference model

bibr 0.7.0 supports automatic review of local NER parses with a fine-tuned
**Qwen3.5-0.8B** LoRA adapter. NER parses each reference first. Missing titles or
printed years, unsupported primary fields, low word coverage, missing year
suffixes and unusually long entries trigger review. These checks are routing
signals, not a calibrated confidence score.

The model uses the training prompt, greedy decoding, thinking disabled and a
fixed 14-field JSON contract. A reviewed record replaces NER only when it improves
the source checks without losing supported title, author or year evidence.
Invalid, unsupported, weaker or timed-out responses retain the original NER
record. Entries keep their original order, bibliography IDs and note ownership.

## Download the trained adapter

```bash
bibr ref-model download --output ./models/bibr-ref-qwen35-08b-balanced-v1
```

This downloads an approximately 84 MB adapter bundle from the
[model release](https://github.com/scienceverse/bibr/releases/tag/model-ref-qwen35-08b-balanced-v1),
verifies its pinned SHA-256 and file manifest, and installs it into a new directory.
Re-running the command verifies an existing installation. It refuses to overwrite
other files. The core bibr package does not install a GPU runtime or bundle weights.

The adapter is `balanced-r32-v1`, trained for multilingual printed references.
Its base is `Qwen/Qwen3.5-0.8B`, pinned to revision
`2fc06364715b967f1860aea9cf38778875588b17`. The adapter is distributed under
AGPL-3.0-or-later; the base and tokenizer retain Apache-2.0. The bundle includes
both licenses. Its converted keys target vLLM's Qwen3.5 language-model loader;
no manual key conversion is needed. The base checkpoint alone is not the parser.

## Serve it separately

The validated serving setup uses vLLM 0.31.0 on Linux with an NVIDIA GPU. Install
it in its own Python environment; it is independent of bibr's metadata LLM and
its CPU/ONNX models. Allow several GB of GPU memory and a first-run base-model
download. Adjust memory allocation to fit your other running models.

```bash
uv venv --python 3.12 .ref-model-venv
uv pip install --python .ref-model-venv/bin/python 'vllm==0.31.0'
VLLM_USE_FLASHINFER_SAMPLER=0 .ref-model-venv/bin/vllm serve Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --host 127.0.0.1 --port 30210 \
  --served-model-name qwen35-08b-base \
  --enable-lora --max-lora-rank 32 \
  --lora-modules bibr-ref-qwen35-08b-balanced=./models/bibr-ref-qwen35-08b-balanced-v1 \
  --language-model-only --reasoning-parser qwen3 \
  --max-model-len 2048 --max-num-seqs 8 \
  --gpu-memory-utilization 0.12 --enforce-eager --attention-backend TRITON_ATTN
```

Model discovery at `http://127.0.0.1:30210/v1/models` must include
`bibr-ref-qwen35-08b-balanced`. Configure bibr to use that adapter alias.
For a server exposed beyond loopback, configure authentication and use a protected
connection. bibr sends only the selected printed references to this endpoint.

## Enable automatic routing

Add these settings to the environment used by the CLI, library, REST or MCP server:

```dotenv
REF_PARSE_STRATEGY=ner
REF_LLM_BASE_URL=http://127.0.0.1:30210/v1
REF_LLM_MODEL=bibr-ref-qwen35-08b-balanced
REF_NER_LLM_FALLBACK=true
# REF_LLM_API_KEY=your-server-key
```

Automatic review is enabled by default when both the endpoint and model are set.
Without them, NER continues locally. `REF_NER_LLM_FALLBACK=false` disables review.
`bibr doctor` reports whether the configuration enables review; it does not certify
server reachability or successful adapter loading.

Review defaults to at most 32 entries per parsing batch, four concurrent requests,
a 30-second per-call timeout and a 120-second overall deadline. Use
`REF_NER_LLM_MAX_REVIEWS`, `REF_LLM_MAX_CONCURRENCY`,
`REF_LLM_TIMEOUT_SECONDS` and `REF_LLM_TOTAL_TIMEOUT_SECONDS` to adjust those
bounds. Inputs longer than 4,000 characters are declined rather than truncated.
The coverage threshold defaults to 0.8 (`REF_NER_LLM_MIN_COVERAGE`); long-entry
review starts at 1,200 characters (`REF_NER_LLM_LONG_CHARS`).

An export's `extraction.diagnostics.reference_review` records the model alias,
configuration status, routing reasons and flagged, attempted, accepted, failed,
rejected and skipped counts. Model token usage is attributed separately under
`parse_reference_finetuned`. Check these receipts when assessing review rates and
latency. Reference parsing can still make mistakes; source grounding does not
prove that a printed value was assigned to the correct field.

To send every reference to the fine-tuned model with aligned NER recovery instead,
use `bibr chew paper.pdf --refs ref-llm`, `chew("paper.pdf", refs="ref-llm")`, or
the REST/MCP `refs="ref-llm"` option. Keep model weights immutable and change the
alias when changing adapters so cached results remain attributable.
