# LLM use

> [!NOTE]
> **Work in progress:** This section needs further work and will be revised.

bibr selectively uses different approaches to deal with the diverse formatting
of scientific papers and support richer extraction. We want to make accurate
extraction practical with small, open-source models, while giving researchers
control over how their papers are processed.

## Development

bibr 🦫 was developed together with AI agentic tools and workflows. Still,
it took me months to iteratively develop and manually validate as much of
the pipeline as possible. Without these tools, building something of this
scope on my own would probably have taken years—a real testament to the
GROBID team.

## Extraction

bibr combines native parsing, rules, specialized ML models, and selected LLM
calls. LLMs help with tasks such as front-page metadata, where publishers
arrange titles, authors, and affiliations differently. Small models fine-tuned
for extraction are useful candidates here; any accuracy advantage needs to be
measured for the task and corpus.

- OCR and extraction models can run locally or through cloud providers.
- `--no-llm` disables downstream LLM extraction, with reduced output. PDF OCR
  is configured separately and may still use vision-language models or remote
  services.

See [configuration](docs/guides/configuration.md) for options,
[evaluation guidance](docs/contributing/evaluation.md) for checking your own outputs, and
[known limitations](LIMITATIONS.md) for accuracy and language/domain coverage.
