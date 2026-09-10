---
hide:
  - toc
---

# bibr 🦫

**bib**liography **r**odent chews through scientific papers and returns structured
JSON. Read PDF, DOCX, JATS XML, HTML, and ePub; extract metadata, references,
sections, sentences, tables, figures, and equations.

Install bibr from PyPI to use the features documented here. See
[installation](getting-started/install.md) for Python and system requirements,
optional extras, and contributor source installs.

```bash
uv init --python 3.12 paper-extraction
cd paper-extraction
uv add bibr
uv run bibr setup
uv run bibr chew paper.pdf -o paper.json
```

The setup wizard detects your hardware and configures OCR and metadata extraction.
The first run may download models and additional runtimes.

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[JSON schema v{{ schema_version }}](reference/schema.md){ .md-button }
[Explore Scienceverse ↗](https://scienceverse.org/){ .md-button }

!!! warning "Alpha: check important extractions against the source"
    Extraction quality varies by paper, language, layout, and model. Current
    evaluation is strongest for English social science papers. A valid JSON
    document can still contain mistakes. Read the [known limitations](limitations.md)
    and [evaluation guide](contributing/evaluation.md) before relying on a field.

<div class="grid cards" markdown>

-   **Native parsing and OCR**

    ---

    Structured formats are parsed natively. PDFs combine layout detection,
    usable native text, and OCR where needed.
    [Follow the pipeline](guides/architecture.md).

-   **Choose the work to run**

    ---

    References use local parsing by default. Choose `--refs llm` for LLM parsing
    or `--refs off` to skip them. `--no-llm` disables downstream LLM extraction;
    PDF OCR is configured separately. [Configure a run](guides/configuration.md).

-   **Inspect the evidence**

    ---

    Sentence IDs, page links, processing warnings, and extraction provenance
    help you review results. Coverage depends on the input and stage.
    [Explore the output](reference/schema.md).

-   **Local or hosted**

    ---

    Use local or cloud models from the CLI and Python, or deploy an HTTP API
    with `bibr serve`. Fully offline operation also requires local model assets
    and external enrichment to be disabled. [Deploy bibr](guides/deployment.md).

</div>

## Use it your way

| You want to… | Go to |
|---|---|
| Process your first paper | [Quickstart](getting-started/quickstart.md) |
| Choose dependencies for your hardware | [Installation](getting-started/install.md) / [Tester guide](tester-guide.md) |
| Work in Python or a notebook | [Python library](guides/library.md) |
| Connect an agent | [MCP server](guides/mcp.md) |
| Run an HTTP service | [Deployment](guides/deployment.md) / [REST API](reference/rest-api.md) |
| Look up a flag, setting, or JSON field | [CLI](reference/cli.md) / [Settings](reference/settings.md) / [Schema](reference/schema.md) |
| Understand model use and accuracy limits | [LLM use](llm-use.md) / [Known limitations](limitations.md) |
| Contribute or measure extraction quality | [Development](contributing/setup.md) / [Evaluation](contributing/evaluation.md) |

The CLI, settings, and schema references are generated from the code when this
site is built. Schema validation checks the output's structure; it does not
establish that an extracted fact is correct.

bibr was originally built as a preprocessing backend for
[Metacheck](https://github.com/scienceverse/metacheck) and can be used independently.
