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

The export records each section's `score` and `source` in
`extraction.diagnostics.section_classification`, keyed by `section_id`, so
consumers can distinguish model, lookup, LLM, and later hierarchy decisions. With `no_llm=True`, heading classification
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
| Data Availability | `data_availability` | Data Availability, Code Availability, Reproducibility Statement |
| Footnote | `footnote` | Footnotes, Notes |
| Table | `table` | Tables (a heading; captions are not sections) |
| Figure | `figure` | Figures (a heading; captions are not sections) |
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

## Front-role classifier

**Module:** `bibr/extract/front_role.py` (bundle loading, scoring), `bibr/extract/front_role_features.py` (the feature contract shared with the trainer), consumed by `bibr/extract/front_matter.py` and `bibr/extract/ref_locator.py`

An optional gradient-boosted model scores every OCR text region with a distribution over
eleven roles: `title`, `byline`, `affiliation`, `abstract`, `keywords`, `doi_line`,
`masthead`, `heading`, `ref_header`, `body`, `other`. Its features are the region's
geometry in a page-relative frame, its font relative to the largest type on the page, and
script-independent text shape (initial density, separator counts, affiliation and
correspondence cues). It is trained from publisher JATS projected onto
cached OCR regions, so its labels are verbatim ground truth rather than an LLM's opinion.

The scores are **evidence, not decisions**. Front-matter ownership (`resolve_front_matter`)
keeps its lexical heuristics and adds the model's roles on top:

- a region the model calls a byline (probability at or above
  `ML_FRONT_ROLE_MIN_CONFIDENCE`, default `0.5`) is admitted as a byline even when the
  English byline shape or the 45-word cap rejects it (consortium bylines, separator-rich
  house styles, non-Latin scripts), including page-1 rows the section classifier mistyped;
- a model title seeds a record for scripts the uppercase test cannot read;
- affiliation and abstract roles are added the same way;
- a confident masthead (`ML_FRONT_ROLE_MASTHEAD_CONFIDENCE`, default `0.8`) cannot root a
  record unless layout labelled the region `doc_title`;
- a row the heuristics seeded as a title but the model confidently types as something else
  (`ML_FRONT_ROLE_RECORD_ROOT_CONFIDENCE`, default `0.9`) keeps the title role and loses
  only the right to root a *second* record. `Correspondence`, `A R T I C L E I N F O`,
  `CITATION` and `Key Features` all score `heading` at 1.00 and all sit above an abstract,
  which is anatomy enough to develop a record — so admitting a correct byline above them
  used to cut the real title away from its own abstract. Layout's `doc_title` and a match
  against the reference parser's detected title both outrank the veto, and `1.0` disables
  it.

`RefLocator` accepts a heading the model scores as `ref_header` as the reference-section
header, which covers the non-English headings (`Literaturverzeichnis`, `Bibliografía`,
`Список литературы`, `参考文献`) the English regex misses.

`ML_FRONT_ROLE_MODEL_ID` defaults to `scienceverse/bibr-front-role-v1`, pinned to a commit
by `ML_FRONT_ROLE_REVISION` so a hub push cannot change front-matter output. Set the id to
null to fall back to the lexical heuristics alone, or `ML_FRONT_ROLE_ENABLED=false` to keep
a configured bundle unloaded. The bundle is a joblib **pickle**, read through the
gadget-restricted loader — only point the setting at a checkpoint you control. scikit-learn
and joblib are core dependencies, so the model runs on a torch-free install.
Loading is soft: an unavailable bundle logs one warning and the pipeline runs on the
heuristics alone. Resolutions influenced by the model carry the `front_role_model` reason
flag, and every candidate records the roles the model contributed (`model_roles`) with its
top scores (`model_scores`). Native DOCX/JATS/HTML inputs have no OCR regions and are never
scored.

## Paper type classifier

**Module:** `bibr/structure/paper_classifier.py` (taxonomy constants), `bibr/extract/core_metadata.py` (LLM classification, via `CoreMetadataExtractor`; `bibr/extract/extractor.py` delegates to it)

The default `scienceverse/bibr-paper-classifier` model is a multitask
classifier with a `sentence-transformers/all-MiniLM-L6-v2` encoder. It reads
the resolved title and abstract and predicts paper type together with OECD
L1/L2. `ML_PAPER_CLASSIFIER_MODEL_ID` selects the model, and
`ML_PAPER_CLASSIFIER_REVISION` pins its revision. According to the
[model card](https://huggingface.co/scienceverse/bibr-paper-classifier/blob/6046171b3198a255acb1f07f81a586a32f399ac4/README.md)
for the pinned revision, its training supervision is labels from a
DeepSeek-v4-Flash teacher, which changed 16,220 OECD L1 labels relative to a
matched OpenAlex baseline.

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
| `meta_analysis` | Quantitative synthesis of multiple studies |
| `case_study` | Case study, case report, case series |
| `commentary` | Commentary, editorial, opinion piece |
| `corrigendum` | Notice amending a previously published article |
| `erratum` | Notice amending a previously published article |
| `retraction` | Notice withdrawing a previously published article |

The default model's paper-type head has no `corrigendum` class. That label
comes from the title guard, which maps titles such as `Corrigendum: ...` and
`Correction to ...` to `corrigendum`, or from the LLM paper-type escalation and
fallback.

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

The default model's L2 head covers 32 of the 36 subdomains. It never predicts
Environmental Biotechnology, Industrial Biotechnology, Nano-technology, or
Agricultural Biotechnology; only the LLM fallback can return those.

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
The exported `extraction.diagnostics.citation_linking` receipt records candidate spans, evidence,
accepted/rejected decisions, and coverage; this makes unresolved citations
visible alongside the successful `xref` rows. Rejected numeric candidates
do not automatically become LLM requests.

`refs="off"` skips bibliographic citation linking along with reference
extraction. `no_llm=True` skips citation linking as part of its reduced
post-processing path.
