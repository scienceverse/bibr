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
arrange titles, authors, and affiliations differently.

On bibr's development papers, large models barely differ, while smaller ones
lose accuracy: off the shelf, 4–8B models pass 6 to 11 points fewer papers than
Qwen3.8-27B. Fine-tuning on bibr's own tasks closes about half of that gap, so
bibr's local defaults are fine-tuned Qwen3.5 models: a 4B model 2.8 points
behind the 27B and a 2B model 6.3 points behind. The hosted default, Claude
Haiku 5.5, is within a point of the 27B. These are development-set results;
[choosing an extraction model](docs/guides/extraction-models.md) has the
numbers and their limits.

- OCR and extraction models can run locally or through cloud providers.
- `--no-llm` disables downstream LLM extraction, with reduced output. PDF OCR
  is configured separately and may still use vision-language models or remote
  services.

See [configuration](docs/guides/configuration.md) for options,
[evaluation guidance](docs/contributing/evaluation.md) for checking your own outputs, and
[known limitations](LIMITATIONS.md) for accuracy and language/domain coverage.
