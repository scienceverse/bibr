# Known limitations

Last updated: 2026-09-10

bibr is alpha software. It already exports rich paper structure, metadata,
references, cross-references, tables, figures, equations, and provenance fields,
but not every exported field has the same level of empirical validation.

This file is deliberately conservative. A capability listed here may work well
on many papers, but it should not yet be treated as fully verified across the
range of scholarly publishing.

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
  classifies instead and the export records it in `processing_warnings`
  (`section_classifier_degraded`, `paper classifier degraded`). Treat those
  exports as less validated than a clean run.

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
