# Known limitations

Last updated: 2026-10-14

bibr is alpha software. It already exports rich paper structure, metadata,
references, cross-references, tables, figures, equations, and provenance fields,
but not every exported field has the same level of empirical validation.

This file is deliberately conservative. A capability listed here may work well
on many papers, but it should not yet be treated as fully verified across the
range of scholarly publishing.

## Where bibr is strong, and where it isn't

We rate each part of bibr on two things: **how well it works** on the papers we
tested, and **how we know**, meaning how strong that evidence is.

Most results come from development sets: papers we looked at while building
bibr. That flatters bibr. A sealed test on papers nobody has looked at comes
next. Every result has a date, because bibr changes every week.

**How well it works**

- 🟢 **Strong**: right on nearly every paper we tested (about 9 in 10 or better)
- 🟡 **Good**: right on most papers, with known weak spots
- 🟠 **Fair**: useful, but check what it gives you
- 🔴 **Weak**: often wrong or missing, so treat it as a first draft
- ⚪ **Not measured**: bibr exports it, but we have no answer key for it yet

**How we know**

- **Sealed test**: scored on papers that were kept out of development.
- **Answer key**: scored against a labelled answer key. Our answer keys were
  transcribed from the PDFs by language models (Claude and Gemini) and
  spot-checked by people, not checked line by line.
- **Spot checks**: a small sample checked by hand.
- **None yet**: no systematic measurement.

### Reading the paper

| Part | What it does | Rating | How we know | Latest result |
|---|---|---|---|---|
| Whole paper | Gets the title, DOI, author names and the reference list right, all at once | 🟢 Strong | Answer key | 96.0% of 200 social-science papers (GROBID 0.9.1: 52.5%). 89.1% of 192 papers sampled across fields (GROBID: 46.4%). With a 27B open model; see [Choosing an extraction model](#choosing-an-extraction-model). Oct 2026 |
| Title and DOI | The paper's own title and DOI | 🟢 Strong | Answer key | Title exactly right 98.4%, DOI 100%, on 192 papers. Oct 2026 |
| Author names | Every author's full name, in order | 🟢 Strong | Answer key | 0.99 F1 on 192 papers. Oct 2026 |
| Abstract | The abstract text | 🟢 Strong | Answer key | 0.94 text overlap (ROUGE-L) on 178 papers. Oct 2026 |
| Affiliations | Which institution each author belongs to | 🟡 Good | Answer key | Author–affiliation pairs: 0.90 F1 on 120 papers that weren't used to build this part, 0.96 on 99 that were. Oct 2026 |
| Corresponding author | Who to write to | 🟠 Fair | Answer key | 0.74 F1 on 300 biomedical papers; finds about half on social-science papers. Oct 2026 |
| Reference list | Finds and splits every printed reference | 🟢 Strong | Answer key | 0.96 F1 on 188 papers, 0.98 on 200. Oct 2026 |
| Reference details | Authors, title, year, DOI, journal and pages of each reference | 🟢 Strong for authors, titles and years · 🟡 Good for DOIs · 🟠 Fair for journal names and pages | Answer key; sealed test for footnote styles | Authors 0.98, titles 0.96, years 0.98, DOIs 0.93, journals 0.73 on 192 papers (Oct 2026); pages 0.65 (Sep 2026). Footnote styles and other languages are harder: every field right on 42% of 9,728 German, Italian and English footnote references (sealed test) and 56% of 10,000 RenoBench references |
| Reference reviewer (opt-in, experimental) | A small fine-tuned model (0.8B) double-checks the parsed references. Needs a GPU server | 🟡 Good, for the model on its own | Sealed test | Every field right on 82.6% of the sealed footnote references above (built-in parser: 42.4%), and on 68.0% of RenoBench references it never saw (55.7%). Not yet measured inside bibr. Oct 2026 |
| Scanned papers | Papers from before 2000, read with bibr's own OCR | 🟡 Good | Answer key | 82.5% of 80 scans pass (GROBID: 23.8%). On the 8 scans with no text layer at all, bibr passes all 8 and GROBID none. Slower than born-digital PDFs. Oct 2026 |
| Body text and headings | The running text, its headings and how they nest | 🟡 Good | Answer key | Sections recovered 0.81, headings found 0.85, nesting right 0.89. Oct 2026 |
| Section types | Labels sections as introduction, methods, results and so on | 🟡 Good in English · 🔴 Weak in other languages | Answer key | 88% of sections typed right on 119 papers (Oct 2026). The classifier retrained for 0.7.1 mostly helps unusual English headings; other languages still lag |
| Tables | Finds each table and its caption | 🟡 Good · 🟠 Fair for captions | Answer key, not yet checked by people | Tables found: 0.92 F1. Captions 0.56 similarity. Cell structure not measured. Oct 2026 |
| Figures | Finds each figure and its caption | 🟡 Good | Answer key, not yet checked by people | Figures found: 0.88 F1. Captions 0.73 similarity. Oct 2026 |
| Equations | Equation regions and their text | ⚪ Not measured | None yet | Exported, no answer key |
| In-text citations | Links "(Smith, 2020)" to the right reference | 🟡 Good | Answer key | 0.80 F1 on 89 papers. Links to tables 0.85, links to figures 0.51. Oct 2026 |
| URLs | Every URL printed in the paper | 🟡 Good | Answer key | 0.81 F1 on 92 papers. Oct 2026 |
| Statements | Funding, conflicts of interest, ethics approval, data availability | ⚪ Not measured | None yet | Exported, no answer key |
| Paper type | Research article, review and so on | 🟡 Good | Answer key (MEDLINE labels) | 0.94 macro F1. Jul 2026 |
| Research field | OECD field of science | 🟠 Fair | Rough labels only | About 0.68 macro F1 at the top level, against OpenAlex labels. Jul 2026 |
| Word, JATS, HTML and ePub files | Read natively, without OCR | ⚪ Not measured | None yet | All our benchmarks are PDFs |

Most of these papers are in English, and most are from the social sciences.
Other languages and fields are less tested. Of the 29 non-English papers in our
model comparison, Claude Haiku 5.5 passed 2 fewer than the 27B and NuExtract 3
passed 6 fewer.

### Checking references against the record

| Part | What it does | Rating | How we know | Latest result |
|---|---|---|---|---|
| Crossref enrichment (opt-in) | Looks each reference up and fills in missing DOIs, volumes and pages | 🟡 Good | Answer key, partial | On 630 papers it left the parsed reference list unchanged and raised correct volumes from 86% to 93% and pages from 81% to 84%. DOI precision stayed at about 96%. Most filled-in values can't be checked against our key, which only has the DOIs papers print. Sep 2026 |
| Whole-reference search (needs a bibr resolver) | Searches with the full reference text when a title search finds nothing | 🟡 Good | Spot checks | Found a record for 106 of 923 previously unmatched references on 119 papers; a hand check of the riskiest found one wrong version. Where a match already existed, it agreed on the DOI 98.6% of the time. Oct 2026 |

### Choosing an extraction model

A language model reads the title, authors and other front matter. We scored
three of them the same way on 252 development papers in October 2026. A paper
passes when its title, DOI, author names and reference list are all right.

| Model | Runs on | Papers passed | Compared with the 27B |
|---|---|---|---|
| Qwen3.8-27B | Your own model server on a 24 GB GPU | 223 of 252 (88.5%) | Reference |
| Claude Haiku 5.5 | Anthropic's API, with your key (bibr 0.7.2 or later). Text from your papers goes to Anthropic | 221 of 252 | −0.8 points |
| NuExtract 3 | bibr's default local model, on NVIDIA GPUs, Apple Silicon Macs or llama.cpp | 200 of 252 | −9.1 points |

`bibr setup` picks NuExtract 3 when you run locally. To use another model, point
bibr at your own OpenAI-compatible server
([configuration guide](docs/guides/configuration.md)).

On a computer with only a CPU, PDFs need llama.cpp (which is slow there), your
own OCR and model server, or cloud processing. Word, JATS, HTML and ePub files
work without any of them.

**Found a paper bibr gets wrong?** Please tell us with the
[wrong-extraction form](https://github.com/scienceverse/bibr/issues/new?template=wrong-extraction.yml).
Papers that break bibr are the most useful thing you can send us.

## Accuracy and scope

bibr will never be error-free across all papers. Its ML models can miss,
misinterpret, or invent details, and errors can propagate between pipeline
stages. These risks remain as the software matures: a larger model, more
training data, or a perfect score on one benchmark cannot guarantee correct
extraction from a new paper. Important fields still need checking against the
source.

bibr is currently optimized for social science papers in English. We are
actively working to improve support for other languages and disciplines, so
researchers beyond our original use case can benefit too. Working with a wider
range of papers also helps us find and fix problems that affect our own
research workflows. Support for a language or field should be distinguished
from evidence that extraction works reliably on it.

## Evaluation coverage is still uneven

The strongest evaluation coverage is around core bibliographic extraction:
titles, DOIs, abstracts, authors, and references. Coverage is strongest for
English social science papers and familiar publisher layouts.

Less is known about performance on:

- Non-English papers, including differences in scripts, terminology, and
  publisher conventions.
- Natural sciences, medicine, chemistry, physics, mathematics, engineering,
  computer science, humanities, law, and other fields outside the current
  social-science-heavy validation set.
- Papers with very dense equations, unusual table layouts, heavy appendices,
  multilingual content, non-standard reference styles, or unusually complex
  author/affiliation blocks.
- Low-quality scans, OCR-hostile PDFs, publisher proof PDFs, theses, books,
  proceedings, reports, and other non-article formats. Scanned PDFs deserve
  extra care; see the next section.

- Inputs that are not PDFs. DOCX, JATS XML, HTML, and ePub files are parsed
  natively and skip OCR and the core LLM extraction, which makes them fast and
  cheap, but the benchmark sets are PDF-only, so their field coverage is less
  measured.

If you use bibr in a new domain, treat the first batch as a validation run:
inspect representative JSON output manually, run `bibr inspect`, and keep an
eye on `extraction.warnings`.

## Scanned PDFs are slower and less verified

bibr can process scanned PDFs in practice, but scanned and OCR-heavy inputs are
not yet covered by the same validation work as born-digital PDFs with usable
native text.

Known caveats:

- A scanned PDF usually has to go through the full OCR path. Runtime can vary a
  lot by OCR backend, hardware, page count, image quality, and whether models
  are already warm.
- OCR output is the only text signal for scanned pages. When there is no native
  PDF text layer, downstream stages lose useful native-text cues that help with
  ordering, line boundaries, and reference segmentation.
- Reference segmentation accuracy is expected to be worse on OCR-only inputs.
  Adjacent references may be merged, long references may be split, and some
  references may be missed entirely.
- Scanned papers with skewed pages, low contrast, dense footnotes, two-column
  reference lists, or noisy page artifacts need manual spot-checking before the
  extracted `bib`, `xref`, and `bib_match` fields are trusted.

## `xref` is not yet a fully trusted citation graph

The top-level `xref` array records detected in-text references to bibliography
items and other document objects such as figures, tables, equations, sections,
and footnotes. This is useful for audit trails and navigation, but it is not yet
a fully verified citation graph.

Known caveats:

- Bibliography xrefs have automated checks, but their extraction-quality
  validation is less mature than that of core metadata and references.
- Non-bibliography xrefs, including figure/table/equation/section links, are
  pattern-based and have less empirical coverage than core metadata fields.
- Edge cases such as long multi-citation spans, flattened superscripts,
  footnote-heavy styles, unusual author-year punctuation, and OCR-damaged
  citation text can still be missed or linked incorrectly.
- `xref.target_id` values should be treated as best-effort links. Downstream systems
  should tolerate missing links and verify important links against the source
  text or PDF.

## Equations are exported, not deeply validated

bibr can emit equation/formula regions in `eq` and preserve formula-like text in
the surrounding text stream. This is currently a structural extraction feature,
not a mature mathematical OCR or semantic equation understanding system.

Known caveats:

- Equation detection and equation text are not yet covered by the same kind of
  broad gold evaluation as title/author/reference extraction.
- Complex multi-line equations, dense notation, alignment environments, mixed
  inline/display math, equation numbers, and formulas embedded inside tables may
  be incomplete, normalized oddly, or missed.
- Local OCR/model choices can affect equation quality. Smaller or heavily
  quantized local models may be faster or cheaper while losing table/equation
  fidelity.
- If equations matter for your workflow, verify `eq`, nearby `text`, and page
  provenance manually before relying on them.

## Tables and figures still need task-specific checking

Tables and figures are useful for locating content and preserving extracted
captions, but they are not yet a complete substitute for human review.

Known caveats:

- Table HTML can flatten hierarchy, merged cells, nested headers, footnotes, and
  multi-page tables.
- Figure exports preserve detected figure regions/captions, but bibr does not
  yet provide verified visual figure understanding or generated figure
  descriptions. We are looking into figure-evidence extraction, including
  high-level figure metadata and cautious chart-data reconstruction, but this is
  not available yet.
- Captions and table/figure links can be affected by OCR ordering and layout
  segmentation, especially in multi-column or publisher-specific layouts.

## Field correctness is stronger than full downstream readiness

The JSON schema validates output shape, not factual truth. A schema-valid export
can still contain a wrong DOI, a missed affiliation, a partial reference, a
misclassified paper type, or a low-confidence enrichment result.

Pay extra attention to:

- Author affiliations. They may be missed, merged, attached to the wrong author,
  or copied only partially when the source has dense, footnote-style, or
  publisher-specific affiliation blocks. We are actively looking into improving
  affiliation extraction.
- Crossref enrichment and consolidation. Enrichment is opt-in (off unless
  `CROSSREF_ENRICH=true`, `--crossref`, or the API's `crossref=true` turns it
  on), and when it runs the external metadata can be missing, rate-limited,
  stale, or matched to the wrong record.
- OECD/domain and paper-type classification outside familiar evaluation
  domains.
- Funding, conflicts of interest, author contribution roles, ethics statements,
  and other research-integrity fields, which are newer than the core
  bibliography metrics.
- Local/offline model configurations. The same paper can produce different
  results with different OCR, LLM, reference parsing, or quantization settings.
- References parsed with the default local parser (`--refs ner`), which trades
  some field precision for speed and zero token cost; `--refs llm` is more
  precise and sends every reference to the LLM.
- Silent fallbacks are no longer silent, but they still change results. When a
  trained section or paper-type classifier cannot load or fails, the LLM
  classifies instead and the export records it in `extraction.warnings`
  (codes `SECTION_CLASSIFIER_DEGRADED` and `PAPER_CLASSIFIER_DEGRADED`). Treat
  those exports as less validated than a clean run.

## Benchmark results depend on the evaluation protocol

Some development datasets overlap the training data of the default reference
models. Results on those datasets measure fit to familiar examples, not
performance on unseen papers. Even a set excluded from model training stops
being a blind test once its errors guide development. Use a fresh independent
sample for a new accuracy claim, and report the exact models, settings and
software version alongside any result.

## `bibr serve` and the MCP endpoint are single-tenant

The HTTP API and the remote MCP endpoint authenticate with one shared bearer
key: every caller is the same principal, job results and MCP paper stores are
not isolated per user, and unguessable job ids are the only thing keeping one
caller's results from another. By default, results live in the memory of a single API process; the optional
Redis store shares them across replicas. Both stores evict results by age,
count and size, so a client must fetch a result within `JOBS_TTL_SECONDS`. Put the service behind your own gateway when
you need per-user access control, quotas, or durable results.

## Recommended use

- Use bibr as a structured extraction and preprocessing tool, not as an
  unquestioned authority.
- Validate representative samples before using a field in a decision pipeline.
- Keep raw PDFs and bibr JSON together so any surprising field can be traced
  back to source regions.
- Prefer downstream logic that can handle missing, partial, or uncertain fields.
- When publishing metrics or comparisons, report the corpus, backend, model,
  settings, and whether references, xrefs, equations, and enrichment were
  enabled.
