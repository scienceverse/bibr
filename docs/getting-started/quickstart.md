# Quickstart

Install bibr, extract a first paper and look at the result. With uv already installed
this takes a few minutes, most of it one-time downloads. There is nothing to configure
first.

## 1. Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/), which also
installs a suitable Python for you. Create a project folder and add bibr to it:

```bash
uv init --bare --pin-python --python 3.12 paper-extraction
cd paper-extraction
uv add bibr
```

`--bare` creates only a `pyproject.toml`, and `--pin-python --python 3.12` pins the
project to Python 3.12, which bibr supports. In an existing Python 3.11–3.14
environment, `python -m pip install bibr` works too; then drop the `uv run` prefix from
the commands below. [Installation](install.md) covers extras, GPUs, updating and
uninstalling.

## 2. Chew a paper

bibr ships with a short synthetic sample paper, so you can start without a PDF of your
own:

```bash
uv run bibr chew --sample -o sample.json
```

```text
  mode fast · local models only, no LLM call · OCR text-layer (no OCR model; reads the PDF's own text)
  [1/1] sample_paper.pdf
  [1/6] Checking file...
  [2/6] Analyzing layout...
        2.4s
  [3/6] Reading the PDF text layer...
  [4/6] Structuring content...
        2.8s
  [5/6] Extracting metadata...
        2.0s
  [6/6] Writing output...
  ✓ Wrote sample.json (7.8s)
    4 warnings to review (abstract, authors, in-text citations) · 1 processing note · details: bibr inspect sample.json
```

The first line says what kind of run this is. With nothing configured, bibr runs in
**fast mode**: its own local models read the paper, and no LLM is called. A born-digital
PDF is read from its **text layer**, the text the PDF already carries, so no OCR model
is needed.

Then your own paper:

```bash
uv run bibr chew paper.pdf -o paper.json
```

The same command reads Word (`.docx`), JATS XML, HTML and ePub files. These are parsed
directly and never need OCR.

**The first run is slower.** bibr downloads about 1.5 GB of models from Hugging Face
once; in our test the whole first run took about a minute and a half. After that, a
full-length paper took about half a minute on a CPU, and the sample about 10 seconds.
To see what would download before anything does, run
`uv run bibr chew paper.pdf --dry-run`.

**A scanned PDF** has no text layer, so fast mode stops with "no usable text layer".
It needs an OCR backend; see [Troubleshooting](troubleshooting.md#a-scanned-pdf-stops-with-no-usable-text-layer).

## 3. Look at the result

`bibr inspect` prints a summary:

```bash
uv run bibr inspect sample.json
```

```text
bibr inspect — sample.json

Title: The Coefficient of Rodential Efficiency: A Synthetic Benchmark
Authors: 2 (Ada Vole, Boris Capybarov)
DOI: —
Paper type: empirical

Sections: 7
Sentences: 29
Tables: 1   Figures: 0   Footnotes: 0   Equations: 10

References
  Bibliography entries: 10
  In-text citations: not present
  Enrichment: not present

Validation
  4 warnings to review (abstract, authors, in-text citations) · 1 processing note
  ! abstract: abstract suspicion: non_reference_share_gt_20pct  [VAL_ABSTRACT_SUSPECT]
  ! authors: 2 extracted author(s) could not be checked: the selected front matter carries no byline row  [VAL_AUTHOR_UNCHECKED]
  ...
```

The warnings are not errors. Each one names a field bibr could not fully check, so look
at that field in the paper before you rely on it. `sample.json` itself is one JSON object
with a table per kind of thing (`metadata`, `author`, `section`, `text`, `bib`, …)
linked by IDs; [What you get](output.md) walks through it with real rows.

To put the reference list into a reference manager:

```bash
uv run bibr references sample.json -o refs.bib
```

```text
  ✓ 10 references from 1 paper → refs.bib (bibtex)
```

A `.ris` file name writes RIS, and a `.json` name CSL-JSON.

`-o` also writes `sample.core.json` and `sample.json.receipt.json` next to the result.
You can ignore both for now; [What you get](output.md#files-next-to-the-result) says what
they are, and [Batch runs](../guides/batch.md) covers them for large corpora.

## 4. Common variations

| Goal | Command |
|---|---|
| Every paper in a folder (one `results/<name>.json` each) | `uv run bibr chew papers/ -o results/` |
| Only pages 1 to 3 | `uv run bibr chew paper.pdf --pages 1-3` |
| Never call an LLM, even if a key is set | `uv run bibr chew paper.pdf --fast` |
| Skip the reference list | `uv run bibr chew paper.pdf --refs off` |
| Look references up in Crossref (sends them to Crossref) | `uv run bibr chew paper.pdf --crossref` |
| See the plan and what would download, process nothing | `uv run bibr chew paper.pdf --dry-run` |
| See more detail while it runs | `uv run bibr chew paper.pdf -v` |
| References of a folder of results for a reference manager | `uv run bibr references results/ -o refs.bib` |
| Results as Parquet tables for pandas, DuckDB or R | `uv run bibr tables results/ --out tables/` |
| A large corpus, resumable | `uv run bibr batch papers/ --out corpus/` |
| Check your setup | `uv run bibr doctor` |

A folder, a glob or several inputs always write one file per paper into the `-o`
directory; `-o` names a single file only when you give one input file. Every flag is in
the [CLI reference](../reference/cli.md).

## 5. Optional: more accuracy with an LLM

Fast mode reads titles, authors, abstracts and references with bibr's own local models.
An LLM helps with the hard cases. To add one, run:

```bash
uv run bibr setup
```

The wizard first detects your hardware and asks whether to run in fast mode; pressing
Enter keeps fast mode and finishes. Answer **No** and it proposes, for your machine:

- bibr's fine-tuned local model, on an NVIDIA GPU with 5 GB of memory or more, or on
  Apple Silicon;
- an OCR/LLM server you run yourself; or
- Claude Haiku 5.5 in the cloud, with your Anthropic API key. This sends page images and
  extracted text to Anthropic, so the wizard asks for your consent first (the default
  answer is No).

It writes `.env` (including `PIPELINE_MODE=fast` or `full`), can install the extras the
plan needs, and can run a one-page test on the sample paper. Runs then use full mode.
`--fast` still keeps any single run free of LLM calls.

For speeds, GPU memory tiers and OCR backends, see
[Choosing an extraction model](../guides/extraction-models.md),
[Configuration](../guides/configuration.md) and
[Windows and GPU setups](../guides/windows-gpu.md). What each choice sends where is in
[What leaves your machine](../guides/privacy.md).

## Where to go next

- [What you get](output.md): every table in the JSON, the warnings, the side files.
- [Troubleshooting](troubleshooting.md): the errors you are most likely to meet.
- [Python library](../guides/library.md): `bibr.chew()` in your own code.
- [R and Metacheck](../guides/r.md): the results in R.
- [Batch runs](../guides/batch.md): thousands of papers, resumable.
