# Classifiers

bibr combines alias rules, local classifiers, and LLM fallbacks for section
types, paper types, OECD domains, and citation linking. Multi-study heading
scopes keep repeated Methods and Results sections attached to the right study.

## Section classifier

**Module:** `bibr/structure/section_classifier.py`

The section classifier maps paper section headers to canonical IMRaD+ categories. It uses a three-tier cascade:

### Tier 1: Lookup-based matching

First, headers are matched against a dictionary of known aliases (`CANONICAL_SECTION_ALIASES` in `paper_contents.py`). This handles common headers like "Introduction", "Methods", "Results", etc.

- **Exact match** -- normalized header text matches an alias directly (confidence: 1.0), bypassing both later tiers
- **Substring match** -- a trusted alias appears as a word-boundary substring of the header (confidence: 0.95), also bypassing both later tiers
- A generic alias hit that isn't trusted is kept as a low-confidence prior, used only if both later tiers return `unknown`

### Tier 2: Trained classifier

For lookup misses, headers route through a trained MiniLM two-head model
(`ML_SECTION_CLASSIFIER_MODEL_ID`, default `scienceverse/bibr-section-classifier`).
It predicts the canonical type and whether the heading is top-level, using
relative position and neighboring headings. Predictions below
`ML_SECTION_CLASSIFIER_MIN_CONFIDENCE` (default `0.5`) collapse to `unknown`.
If the model is disabled or unavailable, the pipeline uses the LLM path.
Classifier load and inference failures are recorded as degraded operation;
`ML_CLASSIFIERS_REQUIRED=true` makes configured classifier availability a
requirement instead.

### Tier 3: LLM-based classification

Headers the trained model leaves as `unknown` (when `ML_SECTION_CLASSIFIER_LLM_ESCALATION` is on, the default) or that skip Tier 2 entirely are sent to the configured LLM provider in a batch. The LLM returns one of the canonical section values for each header (confidence: 0.85). Results are validated against the known section types. Ambiguous headings can include body-text snippets and study-scope context.

Exported section rows include `classification_source` alongside
`classification_score`, so consumers can distinguish model, lookup, LLM,
and later hierarchy decisions. With `no_llm=True`, heading classification
uses the lookup-only path.

### Canonical section types

| Type | Value | Examples |
|---|---|---|
| Title | `title` | Paper title (root/level-0 heading) |
| Abstract | `abstract` | Abstract, Summary, Executive Summary, Statement of Relevance |
| Introduction | `intro` | Introduction, Background, Related Work, Literature Review |
| Methods | `method` | Methods, Materials and Methods, Study Design, Participants |
| Results | `results` | Results, Findings, Experimental Results |
| Discussion | `discussion` | Discussion, General Discussion, Limitations, Conclusion |
| References | `references` | References, Bibliography, Works Cited |
| Acknowledgment | `acknowledgment` | Acknowledgments, Declarations, Corresponding Author |
| Author Contributions | `author_contributions` | Author Contributions, CRediT Authorship Contribution Statement |
| Conflict of Interest | `coi` | Conflict of Interest, Competing Interests, Declaration of Interest |
| Ethics | `ethics` | Ethics Statement, IRB Approval, Informed Consent |
| Funding | `funding` | Funding, Financial Support |
| Keywords | `keywords` | Keywords |
| Endnote | `endnote` | Supplementary Materials, Future Work, Outlook |
| Appendix | `appendix` | Appendix, Supplementary Material, Supporting Information |
| Data Availability | `open_data` | Data Availability, Code Availability, Reproducibility Statement |
| Footnote | `footnote` | Footnotes, Notes |
| Table | `table` | Table (caption/label region) |
| Figure | `figure` | Figure (caption/label region) |
| Unknown | `unknown` | (fallback for unrecognized headers) |

### IMRaD enforcement

After classification, `enforce_imrad_order()` deduplicates sections that should only appear once. Most section types are **repeatable** -- real papers often have multiple subsections with the same classification.

Only two types are **unique**:

- **Abstract** -- a paper has one abstract
- **References** -- a paper has one bibliography

For duplicate Abstract or References sections, the highest-trust
`classification_source` wins; document order breaks ties. Other candidates
are reset to `unknown` with `classification_source="imrad_dedup"`. A later
clean alias heading can therefore supersede an earlier weak classification.
Deduplication is global, including across study scopes. Every other canonical
type is allowed to repeat, and section order is not forced into IMRaD order.

### Layout hints

When the PP-DocLayoutV3 layout model tags a region with a semantic label (abstract, reference, footnote), `PDFParser._handle_section_hint` uses that label to create or reuse an implicit section for it directly — independent of, and prior to, header-based classification. This doesn't override an existing classification score; it's how sections without a text heading (e.g. a layout-detected reference block) get a section in the first place.

## Paper type classifier

**Module:** `bibr/structure/paper_classifier.py` (taxonomy constants), `bibr/extract/core_metadata.py` (LLM classification, via `CoreMetadataExtractor`; `bibr/extract/extractor.py` delegates to it)

The default `scienceverse/bibr-paper-classifier` model is a SPECTER2-based
multitask classifier. It reads the resolved title and abstract and predicts
paper type together with OECD L1/L2. `ML_PAPER_CLASSIFIER_MODEL_ID` selects
the model, and `ML_PAPER_CLASSIFIER_REVISION` pins its revision.

When the paper-type confidence falls below
`ML_PAPER_CLASSIFIER_MIN_CONFIDENCE` (default `0.5`), the pipeline can ask the
LLM paper-type labeler to resolve it. This escalation is controlled by
`ML_PAPER_CLASSIFIER_LLM_ESCALATION` (default `true`). If no model result is
available, core metadata extraction falls back to LLM classification of
paper type and OECD domains. The exported `paper_type_confidence` records
the model or escalation confidence when available.

The output supports these types; notice labels are also handled by title
guards during core metadata extraction:

| Type | Description |
|---|---|
| `empirical` | Original research with data collection/analysis |
| `review` | Systematic review, literature review, scoping review |
| `meta-analysis` | Quantitative synthesis of multiple studies |
| `case-study` | Case study, case report, case series |
| `commentary` | Commentary, editorial, opinion piece |
| `corrigendum` | Notice amending a previously published article |
| `erratum` | Notice amending a previously published article |
| `retraction` | Notice withdrawing a previously published article |

Unrecognized or uncertain values are coerced to `null` when the type cannot be determined, rather than an `unknown` sentinel.

## OECD domain classifier

**Module:** `bibr/structure/paper_classifier.py` (taxonomy constants), `bibr/extract/core_metadata.py` (LLM classification, via `CoreMetadataExtractor`; `bibr/extract/extractor.py` delegates to it)

The same multitask classifier predicts the broad domain and subdomain in
bibr's [OECD Frascati Manual](https://www.oecd.org/en/publications/frascati-manual-2015_9789264239012-en.html)
taxonomy. On the trained-model path, OECD labels are not sent to an LLM for
confidence escalation. A subdomain below
`ML_PAPER_CLASSIFIER_L2_MIN_CONFIDENCE` (default `0.5`) is exported as `null`;
setting the threshold to `0.0` disables that gate. `oecd_confidence` is the
L1 confidence.

### Level 1: Broad domain (6 categories)

| Domain |
|---|
| Natural Sciences |
| Engineering and Technology |
| Medical and Health Sciences |
| Agricultural and Veterinary Sciences |
| Social Sciences |
| Humanities and the Arts |

### Level 2: Subdomain

Each L1 domain has 4-10 subcategories. For example, "Social Sciences" includes: Psychology and Cognitive Sciences, Economics and Business, Education, Sociology, Law, Political Science, Social and Economic Geography, Media and Communications.

When the trained classifier is disabled or unavailable, the LLM fallback
returns both L1 and L2. Its labels are canonicalized against the taxonomy;
a recognized L2 also determines its parent L1. Native formats that already
provide structured metadata can bypass the core extraction/classification
path entirely.

## Multi-study hierarchy

**Module:** `bibr/structure/section_tree.py`

Headings such as `Study 1`, `Study II`, and `Experiment A` establish scopes
before classification. A combined heading such as `Study 2: Methods` can
supply both a study marker and a section label. Hierarchy reconstruction
uses numbering, classifier top-level predictions, and scoped IMRaD anchors
to keep one study's sections from becoming children of another study.

## Citation linking

**Module:** `bibr/structure/citation_linker.py`

The citation linker detects inline citations in text and resolves them to bibliography entries. It uses a 3-tier hybrid approach:

### Tier 1: Numeric citations
Bracket citations `[1]`, `[3-5]`, `[1, 2, 3]` and superscript citations
`^{3}`, `^{15,16}` are detected and checked against bibliography IDs.
Parenthesized numeric and flattened superscript styles have additional
document-style and local-context guards to avoid treating equations,
measurements, and ordinary digits as citations.

### Tier 2: Author-year citations
Parenthetical `(Smith, 2020)` and narrative `Smith (2020)` citation patterns are detected via regex and fuzzy-matched against reference author/year fields.

### Tier 3: LLM fallback
Remaining ambiguous or unresolved citation candidates are sent to the LLM for resolution against the reference list.

Resolved citations are stored as `PaperXref` objects with `xref_type="bib"`.
The exported `citation_linking` receipt records candidate spans, evidence,
accepted/rejected decisions, and coverage; this makes unresolved citations
visible alongside the successful `xref` rows. Rejected numeric candidates
do not automatically become LLM requests.

`refs="off"` skips bibliographic citation linking along with reference
extraction. `no_llm=True` skips citation linking as part of its reduced
post-processing path.
