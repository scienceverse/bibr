---
hide:
  - toc
---

# bibr 🦫

**bib**liography **r**odent turns scientific papers (PDF, Word DOCX, JATS XML, HTML and
ePub) into structured JSON: title, authors and affiliations, abstract, sections and
sentences, tables, figures, equations and a parsed reference list. Every sentence keeps
its page number and every row an ID, so you can check each field against the source.
It runs on a laptop with no API key.

To try one paper without installing anything, upload it at
[try.bibr.org](https://try.bibr.org).

To run it on your machine, install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and run:

```bash
uv init --bare --pin-python --python 3.12 paper-extraction
cd paper-extraction
uv add bibr
uv run bibr chew --sample -o sample.json   # the sample paper that ships with bibr
uv run bibr inspect sample.json
```

There is nothing to configure: with no settings, bibr runs in fast mode, with local
models only. The first run downloads about 1.5 GB of models once.

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[See example output](getting-started/output.md){ .md-button }

!!! warning "Alpha: check important extractions against the source"
    Extraction quality varies by paper, language, layout, and model. Current
    evaluation is strongest for English social science papers. A valid JSON
    document can still contain mistakes. See
    [where bibr is strong and where it isn't](limitations.md) and the
    [evaluation guide](contributing/evaluation.md) before relying on a field.

<div class="grid cards" markdown>

-   **Runs on your laptop, no key**

    ---

    With nothing configured, bibr reads a paper with its own local models and makes
    no LLM call. Born-digital PDFs are read from their text layer; Word, JATS, HTML
    and ePub files are parsed directly. A scanned PDF needs an OCR backend.
    [Install bibr](getting-started/install.md).

-   **See what you get**

    ---

    One JSON file per paper (schema v{{ schema_version }}): metadata, authors,
    affiliations, sections, sentences, tables, figures, equations and references,
    each a table of rows linked by IDs. Reference lists export to BibTeX, RIS or
    CSL-JSON, and whole corpora to Parquet.
    [Tour the output](getting-started/output.md).

-   **Check every field**

    ---

    Sentences carry page numbers, equations and citations point at the sentence they
    came from, and every result carries a validation block and coded warnings that
    say what to double-check. [Known limitations](limitations.md).

-   **Add an LLM for more accuracy**

    ---

    `bibr setup` adds bibr's fine-tuned local model on an NVIDIA GPU or Apple
    Silicon, a server you run, or Claude Haiku 5.5 in the cloud, and asks before
    anything leaves your machine. [Choose a model](guides/extraction-models.md) ·
    [What leaves your machine](guides/privacy.md).

</div>

## Use it your way

| You want to… | Go to |
|---|---|
| Process your first paper | [Quickstart](getting-started/quickstart.md) |
| Understand the JSON | [What you get](getting-started/output.md) / [JSON schema](reference/schema.md) |
| Fix an error or a slow first run | [Troubleshooting](getting-started/troubleshooting.md) |
| Pick extras, a GPU build, update or uninstall | [Installation](getting-started/install.md) / [Windows and GPU setups](guides/windows-gpu.md) |
| Work in Python or a notebook | [Python library](guides/library.md) |
| Work in R or with Metacheck | [R and Metacheck](guides/r.md) |
| Process a whole corpus | [Batch runs](guides/batch.md) |
| Connect an agent | [MCP server](guides/mcp.md) |
| Run an HTTP service | [Deployment](guides/deployment.md) / [REST API](reference/rest-api.md) |
| Look up a command, setting, or JSON field | [CLI](reference/cli.md) / [Settings](reference/settings.md) / [Schema](reference/schema.md) |
| Understand model use and accuracy limits | [LLM use](llm-use.md) / [Known limitations](limitations.md) |
| Contribute or measure extraction quality | [Development](contributing/setup.md) / [Evaluation](contributing/evaluation.md) |

The CLI, settings, and schema references are generated from the code when this
site is built. Schema validation checks the output's structure; it does not
establish that an extracted fact is correct.

bibr was originally built as a preprocessing backend for
[Metacheck](https://github.com/scienceverse/metacheck) and can be used independently.
