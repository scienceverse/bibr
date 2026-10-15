---
title: bibr
emoji: 🦫
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
license: agpl-3.0
short_description: Extract metadata and references from papers, then check them
suggested_hardware: zero-a10g
startup_duration_timeout: 1h
preload_from_hub:
  - PaddlePaddle/PaddleOCR-VL-1.6
  - scienceverse/bibr-extract-qwen35-4b
  - Qwen/Qwen3.5-0.8B
---

# bibr 🦫 + metacheck on ZeroGPU

Upload a paper (PDF). [bibr](https://bibr.org) extracts its metadata and
references, and [metacheck](https://github.com/scienceverse/pytacheck) runs a
few quick checks on the result.

## How it runs

bibr talks to its GPU models over an OpenAI-compatible API, normally served by
vLLM. ZeroGPU has no long-running GPU process, so `zerogpu_models.py` answers
those requests inside the Space: it mounts an in-process ASGI app on bibr's
HTTP clients and runs each batch of requests through a `@spaces.GPU` function
with `transformers`. bibr itself is unchanged.

| Stage | Model | Where |
|---|---|---|
| Layout, sentences, sections, reference NER | bibr's ONNX models | CPU |
| Tables, formulas, scanned pages | PaddleOCR-VL-1.6 | GPU batch |
| Front-page metadata | bibr's fine-tuned Qwen3.5-4B ([scienceverse/bibr-extract-qwen35-4b](https://huggingface.co/scienceverse/bibr-extract-qwen35-4b)) or a cloud API | GPU batch, or API |
| Review of uncertain references | bibr's Qwen3.5-0.8B LoRA (balanced-v1) | GPU batch |
| Checks | metacheck | CPU |

Text regions of born-digital PDFs come from the PDF's text layer and skip OCR.

## Settings

Space variables: `DEMO_LLM` (`local`, `anthropic`, `google`, `none`),
`DEMO_EXTRACTOR` (`4b`, `2b`),
`DEMO_REF_REVIEW` (`1`/`0`), `DEMO_MAX_PAGES` (30), `DEMO_MAX_FILE_SIZE_MB` (10),
`CROSSREF_API_EMAIL` (turns Crossref matching on).

Space secrets, only for the cloud LLM modes: `ANTHROPIC_API_KEY` or `GEMINI_API_KEY`.

## Licences

bibr and the reference adapter: AGPL-3.0-or-later. PaddleOCR-VL-1.6,
Qwen3.5-0.8B and bibr's extraction students: Apache-2.0.
