# [![bibr 🦫 — bibr chew paper.pdf | bibr.org](https://raw.githubusercontent.com/scienceverse/bibr/main/docs/assets/readme-banner.png)](https://bibr.org)

<!-- badges: start -->
[![PyPI version](https://img.shields.io/pypi/v/bibr.svg)](https://pypi.org/project/bibr/)
[![Docs](https://img.shields.io/badge/docs-bibr.org-blue)](https://bibr.org)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
![Made in Europe](https://img.shields.io/badge/Made_in_Europe-003399?logo=european-union&logoColor=FFCC00)
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![codecov](https://codecov.io/gh/scienceverse/bibr/graph/badge.svg?token=Mt0vQyE4qX)](https://codecov.io/gh/scienceverse/bibr)
<!-- badges: end -->

**bib**liography **r**odent 🦫 - a modern scientific extraction pipeline. Chews through papers, powered by open source and Metascience. Originally built for [Metacheck](https://github.com/scienceverse/metacheck) with accuracy as a priority.

- Reads PDF, DOCX, JATS XML, HTML, and ePub.
- Extracts metadata, references, full text, tables, figures, and equations into a
  [versioned JSON format](https://bibr.org/reference/schema/).
- Includes sentence and page references to help check extractions against the source.
- Works through the CLI, Python, an HTTP API, a web demo, or MCP.
- Lets you choose local or cloud models, limit page ranges, and skip extraction stages.

> **Alpha:** Expect bugs and uneven extraction quality. Current evaluation is strongest
> for English-language social science papers. See [known limitations](https://bibr.org/limitations/).

## Get started

Requires Python 3.11–3.14 and the
[system prerequisites](https://bibr.org/getting-started/install/#system-prerequisites).
Install from PyPI in a project managed by [uv](https://docs.astral.sh/uv/):

```bash
uv init --python 3.12 paper-extraction
cd paper-extraction
uv add bibr
uv run bibr setup
uv run bibr chew paper.pdf -o result.json
```

In an existing Python environment, you can also install with
`python -m pip install bibr` and run `bibr setup` / `bibr chew` directly.

The setup wizard detects your hardware, configures OCR and the LLM, and offers to
install any additional dependencies. Core installs run bibr's trained models through
ONNX Runtime; PyTorch, the demo, MCP, and hardware-specific serving runtimes are
[optional extras](https://bibr.org/getting-started/install/#extras). The first run may
download models and runtimes. See the [tester guide](https://bibr.org/tester-guide/)
for platform-specific instructions and the
[source installation guide](https://bibr.org/getting-started/install/#installing-from-source-contributors)
for development setup.

## Usage

### Command line

```bash
uv run bibr chew papers/ -o results/   # Process a directory
uv run bibr chew paper.pdf --dry-run   # Preview the processing plan
uv add 'bibr[demo]'                    # Add the optional web demo
uv run bibr demo                       # Open it locally
```

References are parsed locally by default. Use `--refs llm` to parse them with the
LLM, or `--refs off` to skip them. More options: [CLI reference](https://bibr.org/reference/cli/).

### Python

```python
import bibr

result = bibr.chew("paper.pdf")
print(result.title)
references = result.references.df  # pandas DataFrame
result.save("result.json")
```

See the [Python guide](https://bibr.org/guides/library/) for batch processing and
reusing loaded models with `bibr.Chewer`.

## LLM use

bibr uses LLMs selectively for tasks such as front-page metadata, with support
for small models tuned for extraction. You can disable downstream LLM extraction
with `--no-llm`, which returns structural output; PDF OCR may still use a
vision-language model. The [LLM use note](https://bibr.org/llm-use/) covers these choices
and how agentic LLMs helped develop bibr. It is a work in progress.

## Documentation

- [Configuration](https://bibr.org/guides/configuration/) — OCR, LLMs, reference parsing, and presets.
- [Deployment](https://bibr.org/guides/deployment/) — HTTP API (`bibr serve`), Docker, hardware, and authentication.
- [MCP server](https://bibr.org/guides/mcp/) — extraction tools for agents (`bibr mcp`).
- [JSON schema](https://bibr.org/reference/schema/) and [pipeline architecture](https://bibr.org/guides/architecture/).
- [Evaluating extraction quality](https://bibr.org/contributing/evaluation/) on papers from your workflow.

## Contributing

Bug reports, test papers, and contributions are welcome. See
[CONTRIBUTING.md](https://github.com/scienceverse/bibr/blob/main/CONTRIBUTING.md)
for development setup, tests, and pull requests.

Development began privately in December 2025. This public repository starts with
a clean source snapshot for the 0.5.0 launch; the earlier development history
remains private. Selected early design documents and their original contributions
are preserved in the [project history](https://github.com/scienceverse/bibr/tree/main/history).

---

## Acknowledgments

Special thanks to **Daniël Lakens** and **[Lisa DeBruine (@debruine)](https://github.com/debruine)**, for putting faith and patience in the project, and being generous with their time
 to help make bibr 🦫 better for everyone.

Lisa also contributed to the early paper-structure and metadata design documentation
preserved in the project history.

Also, to the whole [Metacheck](https://www.scienceverse.org/metacheck/) team, and **TU Eindhoven**.

We are grateful to the open-source projects that bibr builds on:

- [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) (PaddlePaddle) — default OCR recognizer
- [GLM-OCR](https://huggingface.co/THUDM/GLM-OCR) (THUDM, Tsinghua University) — explicit compatibility backend and fallback
- [GROBID](https://github.com/kermitt2/grobid) — a major source of inspiration for structured scientific document parsing
- [LitServe](https://lightning.ai/docs/litserve/home) (Lightning AI) — serving infrastructure
- [PP-DocLayoutV3](https://github.com/PaddlePaddle/PaddleOCR) (PaddlePaddle) — document layout analysis
- [wtpsplit](https://github.com/segment-any-text/wtpsplit) — sentence segmentation
- [Crossref](https://www.crossref.org/) — reference metadata enrichment

---

## License

[AGPL-3.0-or-later](https://github.com/scienceverse/bibr/blob/main/LICENSE.md).
