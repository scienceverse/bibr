# What you get

bibr turns each paper into one JSON file. This page walks through that file for someone
who has never seen it, using the sample paper that ships with bibr:

```bash
uv run bibr chew --sample -o sample.json
```

The full contract, every field and its type, is in the [JSON schema reference](../reference/schema.md)
(schema v{{ schema_version }}).

!!! info "Five terms used on this page"
    - **Fast mode**: the default when no LLM is configured. bibr's own local models read
      the paper; no LLM is called and no API key is needed.
    - **Full mode**: a run with a configured LLM (`bibr setup`, or `--full`), which
      reads what the local models are unsure of.
    - **Text layer**: the text a born-digital PDF already carries. Fast mode reads it
      instead of running OCR; a scanned PDF has none.
    - **OCR backend**: what reads text from page images when there is no usable text
      layer: an OCR model on your machine, a server you run, or a cloud vision model.
    - **Enrichment**: looking references up in Crossref (and affiliations and funders in
      ROR) after extraction. Off unless you pass `--crossref`.

## The summary

`bibr inspect` prints what a result contains and what to double-check:

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
  ! in-text citations: 0/10 bibliography entries have a resolved in-text citation xref (0.0%)  [VAL_XREF_LOW_COVERAGE]
  ! in-text citations: 10 references but zero in-text citation xrefs resolve to a bib  [VAL_XREF_ZERO]

Processing notes
  · Fast mode made no LLM calls. Not produced: author roles and affiliation institutions and departments.  [LLM_SKIPPED]
  · 1 table(s) were rebuilt from text-layer glyph positions, not read by an OCR model  [TEXT_LAYER_TABLE_REBUILT]

LLM usage
  not present
```

The *Validation* and *Processing notes* parts are explained [below](#validation-and-warnings).

## The JSON at a glance

The file is one JSON object. Its top-level keys, in order:

| Key | What it holds |
|---|---|
| `paper_id`, `schema_version`, `source` | Which paper this is: an ID (by default the file name without its extension), the schema version, and the input file's name, SHA-256 and format. |
| `metadata` | One object with the paper-level fields: title, abstract, DOI, language, paper type, journal, dates, license, statements. |
| `author`, `affiliation`, `funding` | The byline and who funded the work. |
| `section`, `text` | The paper's structure and its text, one row per sentence. |
| `bib`, `xref`, `url` | The parsed reference list, in-text cross-references, and links. |
| `table`, `figure`, `footnote`, `eq` | Tables (with their cells), figures, footnotes, and statistics such as `d = 0.82`. |
| `metadata_match`, `affiliation_match`, `funding_match`, `bib_match` | What external registries returned. Empty unless you turn on enrichment. |
| `extraction` | How the result was produced: settings, timings, the OCR and LLM used, the validation result and warnings. |

Every table is a list of rows. Each row has an integer ID named after its table
(`author_id`, `text_id`, `bib_id`, …), numbered from 1. Fields ending in `_id` that
name another table point at that table's rows. A value bibr did not find is `null`,
never an empty string. Don't use list positions as IDs.

## One row from each table

The rows below come from `sample.json`, trimmed: `"...": "..."` stands for fields left
out. The [schema reference](../reference/schema.md#nested-record-fields) lists them all.

**`metadata`** is a single object, not a list:

```json
"metadata": {
  "title": "The Coefficient of Rodential Efficiency: A Synthetic Benchmark",
  "abstract": "We introduce the Coefficient of Rodential Efficiency (CRE), a fully synthetic benchmark statistic ...",
  "doi": null,
  "language": "en",
  "paper_type": "empirical",
  "oecd_l1": "Natural Sciences",
  "journal": null,
  "...": "..."
}
```

**`author`**: one row per author, in byline order.

```json
{"author_id": 1, "given": "Ada", "family": "Vole", "email": null, "corresponding": false,
 "orcid": null, "role": [], "credit_roles": [], "...": "..."}
```

**`affiliation`**: `author_ids` lists the authors at each affiliation.

```json
{"affiliation_id": 1,
 "text": "Department of Synthetic Studies, Institute for Fictional Research",
 "institution": null, "department": null, "city": null, "country": null,
 "author_ids": [1, 2]}
```

In fast mode the affiliation is kept as printed; splitting it into institution and
department needs an LLM (the `LLM_SKIPPED` note above says so).

**`section`**: headings, with a `section_type` such as `intro`, `method`, `results`
or `references`, and `parent_section_id` for subsections.

```json
{"section_id": 4, "header": "2. Methods", "level": 1, "parent_section_id": null,
 "section_type": "method"}
```

**`text`**: one row per sentence, with the section, paragraph and page it came from.

```json
{"text_id": 4, "section_id": 2, "paragraph_id": 2, "page_number": 1,
 "text": "CRE separates the groups with a reported effect size of d = 0.82.",
 "formatted": null}
```

**`bib`**: one row per entry of the reference list, parsed into fields. `text_id` is the
sentence row that holds the printed reference.

```json
{"bib_id": 1, "text_id": 19, "bib_type": "journal_article",
 "authors": "Vole, A., & Capybarov, B", "year": 2023,
 "title": "Synthetic coefficients for pipeline testing",
 "container": "Journal of Fabricated Metrics", "volume": "12", "issue": "3",
 "first_page": "145", "last_page": "158", "doi": "10.9999/jfm.2023.0145", "...": "..."}
```

**`xref`**: a place in the text that refers to something else. `xref_type` names the
table (`bib` for a citation, `table`, `figure`, …), `target_id` the row in it, and
`text_id` with `start` and `end` the characters of the sentence that hold the mention.

```json
{"xref_id": 1, "xref_type": "table", "target_id": 1, "contents": "Table 1",
 "text_id": 12, "start": 0, "end": 7}
```

Here sentence 12 ("Table 1 reports the simulated sample characteristics …") points at
table 1. The sample's citations were not linked to its references, which is what the
two `VAL_XREF_*` warnings report.

**`table`**: the cells as a list of rows (`contents`) and as HTML, with the caption.
`text_id` is the caption's sentence row.

```json
{"table_id": 1, "label": "1", "section_id": 4, "text_id": 29, "page_number": 1,
 "caption": "Table 1. Simulated sample characteristics by group.",
 "contents": [["Group", "N", "Mean CRE", "SD"], ["A", "12", "0.91", "0.14"], "..."],
 "html": "<table><tr><td>Group</td><td>N</td>...</table>"}
```

**`figure`**: the sample has none; this row is from a JATS XML file. `image` stays
`null` unless you pass `--figure-images`, and `page_number` is `null` because XML has no
pages.

```json
{"figure_id": 1, "label": "1", "section_id": 5, "text_id": 131, "image": null,
 "caption": "Figure 1. (a) Total scope of content (abstracts and articles) available in Europe PMC. ...",
 "page_number": null}
```

**`eq`**: statistics and equations found in sentences, split into their parts.

```json
{"eq_id": 1, "text_id": 4, "start": 56, "end": 64, "grp_id": 1, "verbatim": "d = 0.82",
 "lhs": "d", "df": null, "comp": "=", "rhs": "0.82"}
```

### Following the links

The IDs are what let you check a field against the paper. For example, to print each
statistic with the sentence and page it came from:

```python
import json

with open("sample.json", encoding="utf-8") as f:
    paper = json.load(f)

sentences = {t["text_id"]: t for t in paper["text"]}
for eq in paper["eq"][:2]:
    sentence = sentences[eq["text_id"]]
    print(eq["verbatim"], "| page", sentence["page_number"], "|", sentence["text"])
```

```text
d = 0.82 | page 1 | CRE separates the groups with a reported effect size of d = 0.82.
M = 0.91 | page 1 | Simulated group A showed the highest mean CRE (M = 0.91, SD = 0.14), followed by group B (M = 0.77, SD = 0.18) and group C (M = 0.69, SD = 0.21).
```

The same pattern joins `text.section_id` to `section`, `affiliation.author_ids` to
`author`, and `xref.target_id` to the table its `xref_type` names.

## Validation and warnings

Everything about how the result was produced lives under `extraction`. Two parts of it
tell you how far to trust the result.

**`extraction.validation`** is bibr checking its own output:

```json
"validation": {
  "errors": 0, "warnings": 4, "blocking": 0, "promotable": true,
  "issues": [
    {"code": "VAL_AUTHOR_UNCHECKED", "severity": "warning",
     "message": "2 extracted author(s) could not be checked: the selected front matter carries no byline row",
     "origin_stage": "extract", "evidence_ids": ["reason:no_printed_byline", "author:1", "author:2"],
     "count": 2, "blocking": false},
    "..."
  ]
}
```

- A **warning** means bibr could not confirm a field, or found something unusual about
  it. The result is still usable; check the named field against the paper before you
  rely on it. `evidence_ids` names the rows involved (`author:1` is `author_id` 1).
- An **error** is a more serious problem with the output. When an issue is
  **blocking**, the result is not *promotable*: `-o` then writes it to a quarantine
  folder instead (see [below](#files-next-to-the-result)).
- Each issue has a stable `code`; its `message` says what was found.

**`extraction.warnings`** are processing notes: things that happened during the run and
that affect what you got, such as a skipped stage.

```json
"warnings": [
  {"code": "LLM_SKIPPED",
   "message": "Fast mode made no LLM calls. Not produced: author roles and affiliation institutions and departments."},
  {"code": "TEXT_LAYER_TABLE_REBUILT",
   "message": "1 table(s) were rebuilt from text-layer glyph positions, not read by an OCR model"}
]
```

Every processing-warning code is listed, with its meaning, in the
[schema reference](../reference/schema.md#warning-codes).

**`extraction.fields`** records, for the main fields, what happened to each and which
step produced it. In the sample, `"title": {"state": "extracted", "source": "front_tagger", ...}`
and `"doi": {"state": "absent", ...}`. The states:

- `extracted`: a value was exported.
- `absent`: bibr looked and found none, so a `null` here is an answer.
- `abstained`: bibr found candidates but declined to choose between them.
- `failed`: the step that produces the field failed, so the value is missing.
- `not_attempted`: the run did not try, as with no LLM or with `--refs off`.

A valid file can still contain mistakes: validation checks consistency, not truth. See
[Known limitations](../limitations.md) for how accurate each field is.

## Files next to the result

With `-o sample.json`, bibr writes:

| File | What it is |
|---|---|
| `sample.json` | The result. |
| `sample.core.json` | The extraction before any enrichment. Without `--crossref` (the default) it has the same content as `sample.json`. |
| `sample.json.receipt.json` | The run's status: where the result went (`destination_path`), its `disposition` (`promotable` for a normal result) and the steps completed. |
| `sample.json.enrichment.json` | Only with `--crossref`: what enrichment returned. |

A result that is not promotable, for example one with a blocking validation error, goes
to `_quarantine/<disposition>/sample.json` beside the requested path instead (with its
`.core.json`). The receipt stays next to the requested path and records the
disposition, and the console says where the result went.

`bibr tables` and `bibr references` skip the side files when you point them at a folder.

## References and tables out of the JSON

**References** for a reference manager:

```bash
uv run bibr references sample.json -o refs.bib    # or refs.ris, or refs.json (CSL-JSON)
```

```text
  ✓ 10 references from 1 paper → refs.bib (bibtex)
```

```bibtex
@article{Vole2023,
  author = {Vole, A. and Capybarov, B},
  title = {Synthetic coefficients for pipeline testing},
  journal = {Journal of Fabricated Metrics},
  year = {2023},
  volume = {12},
  number = {3},
  pages = {145--158},
  doi = {10.9999/jfm.2023.0145},
  ...
}
```

Point it at a folder of results to get one file for all of them.

**Tables** for pandas, polars, DuckDB or R: `bibr tables` turns one or many results into
one Parquet file per table, each row keyed by `paper_id`:

```bash
uv run bibr tables sample.json --out tables/
```

This writes `tables/paper.parquet`, `tables/author.parquet`, `tables/text.parquet`,
`tables/bib.parquet` and so on, plus `extraction_*.parquet` files such as
`extraction_warnings.parquet` and `extraction_validation_issues.parquet`.

```python
import pandas as pd

bib = pd.read_parquet("tables/bib.parquet")
print(bib[["paper_id", "bib_id", "year", "title"]].head(3).to_string(index=False))
```

```text
    paper_id  bib_id  year                                                             title
sample_paper       1  2023                       Synthetic coefficients for pipeline testing
sample_paper       2  2022               Burrowing-agent simulations for software benchmarks
sample_paper       3  2024 Reproducible placeholder documents for text-extraction evaluation
```

For R, see [R and Metacheck](../guides/r.md); for corpora, [Batch runs](../guides/batch.md);
for doing all of this from Python, the [Python library](../guides/library.md).
