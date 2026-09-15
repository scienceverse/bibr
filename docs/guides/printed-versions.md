# Printed versions and multiple articles

A PDF can contain several presentations of one article: a cover sheet, repeated
front matter, or titles and abstracts in different languages. A proceedings PDF
can instead contain several independently authored articles. Extraction resolves
article ownership before choosing the main title and abstract.

The main-field preference is the explicitly identified original version, otherwise
the first complete printed version in reading order. English has no automatic
priority. Extracted text is not translated, and abstracts from different versions
are not concatenated.

## Current ownership rules

Adjacent independently developed front-matter blocks can merge when they have
the same normalized printed byline and either the same DOI in the title's source
section or a repeated title. DOI evidence from reference regions cannot establish
shared article identity.
Conflicting DOIs prevent merging. Matching authors alone, a matching title alone,
and a shared DOI with different bylines are insufficient. Merge decisions and
their source blocks appear in the extraction diagnostics.

Numbered descendant body headings do not create new article records just because
their section classification is unknown. A numbered heading with its own byline
or abstract remains a possible article, including when the parser mistakenly
nests it beneath another article.

Genuinely distinct records retain separate candidates. The existing single-paper
API requires a target when ownership is ambiguous. Local manifest selectors
(`expected_title`, `expected_doi`, and `target_block_hint`) can identify one record;
all supplied selectors must agree. The new document API returns an outcome for
every detected record instead.

## Returning all detected papers

```python
import bibr

document = bibr.chew_document("proceedings.pdf")
# In an existing event loop: await bibr.achew_document("proceedings.pdf")
for record in document.records:
    if record.paper is not None:
        print(record.record_id, record.paper.title)
        print(record.paper.metadata_variant)
    else:
        print(record.record_id, record.status, record.reason_flags)
        if record.partial_paper is not None:
            print(record.partial_paper.author)  # Retained fields; still incomplete.
document.save("proceedings.json")
```

`LocalPipeline.process_document()` exposes the same path for a reusable pipeline.
It parses/OCRs the input once, classifies and detects article candidates, and then
extracts each safe article scope independently. Each record gets its own body,
references, DOI evidence and linked objects. Two articles on the same page can
separate when their exact source anchors establish a boundary. Interleaved text,
ambiguous shared paragraphs or conflicting object ownership produce unresolved
records; they never trigger extraction over the whole document for each article.

The document detector can recover a byline printed together with affiliations
and abstract prose when an immediately preceding article heading, local layout,
person-name shapes and substantive prose corroborate the boundary. Recovered
copies/translations still require matching author evidence plus a repeated title
or one shared article DOI to merge; conflicting DOIs prevent it. This document
refinement leaves the existing single-paper target-selection policy unchanged.

When that complete local article evidence is present, it can also override a
learned heading label that would otherwise prevent the title from starting a
record. The original model scores and the override reason remain available in
diagnostics. Missing local evidence and other boundary vetoes still prevent
the override.

When the parser reuses an earlier Abstract section for later articles, exact OCR
region order takes precedence over section numbering. A paragraph spanning
columns uses its first supported source region to locate its start, and its
complete source span must fit inside the article boundary. Missing provenance
cannot override a known conflicting source order. Remaining unsupported title
boundaries are explicit unresolved outcomes.

Sentences can have different page numbers while sharing the source span of one
paragraph. Document ownership checks that complete paragraph span without
changing sentence pages or treating its repeated provenance as repeated text.
The paragraph must remain inside one article's boundaries.

The [document schema](../schema/bibr-document-v1.schema.json) uses the separate
root key `document_schema_version: "1.0"`. It contains:

| Field | Meaning |
| --- | --- |
| `document_id`, `source` | Content hash identity and original input file identity |
| `records` | Every detected article, including failed and unresolved candidates |
| `records[].record_id` | Unique document-local key; use this rather than a nested DOI or `paper_id` for joins |
| `records[].paper` | Existing 11.1 paper export when extracted; otherwise `null` |
| `records[].partial_paper` | Optional incomplete paper with blocking validation; never an extracted success |
| `records[].source_text_ids`, `source_section_ids`, `pages` | Article scope in the original parsed source, before per-article normalization |
| `records[].reason_flags`, `error` | Boundary/validation findings or a safe extraction failure summary |
| `diagnostics.detected_record_ids` | Complete candidate inventory, checked against the returned records |
| `diagnostics.unassigned_source_text_ids` | Parsed text outside all record scopes, such as a leading fragment or unresolved area |

Record status is `extracted`, `unresolved` or `failed`. Document status is
`complete` when all detected records extract, `partial` when some do, and
`unresolved` when none do (including no detected candidates). `document.ok` means
`complete`; `document.papers` contains extracted papers only, so inspect
`document.records` when accounting for missing results. Input, OCR and detection
failures that occur before an inventory exists still raise an exception.

A known field-validation failure can retain already extracted authors,
references and other fields in `record.partial_paper`. Its validation remains
blocking, and `record.paper` stays `null`. It never appears in `document.papers`
or makes the document complete. Unsafe scopes and failures before a paper is
built have no partial payload. Retaining partial data does not trigger later
enrichment or another extraction call.

“Complete” describes processing of detected candidates. It does not certify that
every article was detected or that every metadata field is correct. Page-range
requests cover only the requested range and are flagged in document diagnostics.
Unassigned source text is also flagged; it may belong to an incomplete article
whose title/byline is outside this input, and is not silently attached to a neighbor.
Nested paper timings cover per-record processing, excluding shared document
preprocessing. Nested paper IDs and author/reference IDs may repeat; their scope
is the containing record. The single-paper `chew()` API, CLI and REST response
shape remain compatible; this document envelope is an opt-in Python API.

## Preserving printed versions

Schema 11.1 adds a top-level `metadata_variant` table. Each row contains:

| Field | Meaning |
| --- | --- |
| `variant_id`, `record_id` | Version identifier and owning front-matter record |
| `field`, `text` | Printed `title` or `abstract`, with original wording |
| `language` | Source-supported language; currently `null` when not established |
| `is_primary` | Whether this text matches the final exported scalar field |
| `source_text_ids`, `source_section_ids`, `pages` | Source evidence captured before section normalization |
| `presentation_ids` | Explicit links between a locally owned title and abstract within the article |
| `byline_source_text_ids`, `byline_source_section_ids` | Printed byline evidence supporting those links |

Repeated identical versions are deduplicated while retaining their combined
provenance. The scalar `metadata` object keeps its existing shape. Older 11.0
exports remain readable and replayable; the published schema describes the
current writer output.

Two versions sharing a `presentation_id` belong to the same supported local
title/byline/abstract presentation. A repeated title may link to several
presentations after text deduplication. Empty links mean pairing is unestablished:
never pair the nth title with the nth abstract. Linking requires one supported
title, an intervening byline and one owned abstract before the next title.
It does not establish that two separate article records are translations.

Deterministic capture currently requires a complete, unambiguous field inventory:
titles with their own local article anatomy, or printed abstract headings with
fully owned paragraphs originally labelled as abstract. An OCR abstract region
with an explicit inline abstract label can also supply a version when its entire
text matches owned source sentences exactly. Capture happens before implicit
section normalization and never rebuilds abstracts from a later, possibly
enlarged tree.

This support declines ambiguous stacked title fragments,
partly owned sections, and text whose only abstract
evidence is a section classifier label. The table may therefore be empty or cover
only one field. If a later guard clears a scalar, the preserved versions remain
available with no matching primary row.

A complete linked presentation supplies the main title and abstract together,
and its physical byline supplies the author extraction context. Deduplicating
repeated text does not discard the individual byline ownership. An original
marker anchored to exactly one presentation selects that whole presentation;
otherwise the first complete presentation wins. Ambiguous original markers
leave the deterministic choice unresolved.

When alternatives exist without a complete pairing, the model receives the same
joint-selection rule, and independent first-title/first-abstract overrides are
disabled. `VAL_PRIMARY_PRESENTATION_UNRESOLVED` reports that the pairing is still
unverified. A model-selected pair is not evidence of a source-supported link.

`extraction.diagnostics.front_matter` retains selected record ID, selection method,
reason flags, block membership, merge evidence, and candidate source IDs and roles.
It is also present on unresolved pipeline exports, so a failed decision can be
inspected without rerunning model extraction.
Its `presentation_selection` receipt records the complete presentation inventory,
selected ID and decision reason, with variant, byline and original-marker source
links. The receipt records the choice before later validation guards; each
variant's `is_primary` flag still describes the final exported scalar.

## Remaining structural work

Translated bylines across scripts and cover sheets with the byline printed only
once need richer identity evidence before they can merge safely. The current
presentation links preserve proven local pairings; they do not guess cross-script
name correspondences. Author identifiers, explicit printed aliases and document
layout can supply corroboration. Shared contact details or a document-wide DOI
alone must not merge distinct abstracts in a proceedings volume.

Measure ownership, version completeness, and primary-field accuracy separately
from paper-level pass rate. Improving pass rate alone can hide a missing alternate
abstract or an incorrectly merged article.

The separate [document evaluator](../contributing/evaluation.md#document-inventory-and-abstracts) measures missing
and extra records, false merges/splits, source overlap, and abstract ownership,
completeness, absence and primary selection. It requires independently prepared
source-bound annotations and does not alter the existing paper pass floors.
