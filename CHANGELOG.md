# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed — export schema 12.0 (breaking)

The JSON export moves to schema `12.0`. It separates what the paper says from
how bibr produced it, and the generated JSON Schema is now a documented
contract. bibr writes and reads only 12.x; a v11 export or core checkpoint is
rejected, as v10 was by 11.0. The drafted 11.1 is retired and was never
released.

- `extraction` now holds everything about how the output was produced, and
  is always present; everything else is the paper. Root keys are emitted in the
  order `paper_id`, `schema_version`, `source`; `metadata`, `author`,
  `affiliation`, `funding`, `text`, `section`, `url`, `bib`, `xref`, `figure`,
  `table`, `footnote`, `eq`; `metadata_match`, `affiliation_match`,
  `funding_match`, `bib_match`; `extraction`. A Paper exported
  outside the pipeline gets a minimal `extraction` block (package version,
  export time, diagnostics, validation) with `settings` omitted.
- The root `validation` block moved to `extraction.validation`. Readers of
  saved files can use `bibr.validation.payload_validation(payload)`, which
  finds it in both 12.x and older exports.
- Figure and table `parts` left the content rows. The whole-object fields are
  now truly whole: a figure detected as several panel crops gets an `image`
  composited from them (it used to be the first panel), and a table continued
  across pages keeps each printed piece's HTML in `html` instead of a lossy
  re-render of the merged cells. Each piece's page and bounding box moved to
  `extraction.float_parts`. Merged figures and continued tables no longer lose
  pieces when a later merge step combines them. With images requested, each
  figure image is now serialized once, not twice.
- Processing fields left the content rows and now live under `extraction`,
  keyed by the rows' IDs: `section[].classification_score` and
  `classification_source` → `extraction.diagnostics.section_classification`
  (an unscored section's score is `null`, not `0.0`); `xref[].tier` →
  `extraction.diagnostics.xref_tier`; `metadata.paper_type_confidence` and
  `oecd_confidence` → `extraction.diagnostics.paper_classification`;
  `bib[].consolidated_fields` (a comma-joined string) →
  `extraction.diagnostics.consolidation` (a list of field names per `bib_id`);
  the opt-in per-sentence `text[]._bbox_2d`, `_font_size`, `_font_bold`,
  `_is_italic`, `_region_type`, `_page_w` and `_page_h` →
  `extraction.text_regions` (same `--region-meta` opt-in); and the root
  `qualification_provenance` → `extraction.qualification`, omitted rather than
  `null` when no LLM task ran.
- Duplicates are gone. `bib[].author` and `editor`, a split derived from the
  printed strings, are removed; `bib[].authors` and `editors` stay exactly as
  printed. `author[].affiliation` is removed; the `affiliation[]` table,
  linked by `author_ids`, is the one source and is now built on every run from
  the byline, with the parsed components `null` when no LLM ran (it used to be
  empty then).
- `xref[].xref_id` and `url[].url_id` are new 1-based primary keys, so every
  record table has one. The xref's target stays in `target_id`; up to v10,
  `xref_id` meant the target.
- Absent values are `null`, never `""`: `author[].given` and `family`,
  `section[].header` and `eq[].df`.
- Closed vocabularies are enums in the schema: `section[].section_type`,
  `bib_type` in `bib` and both match tables, `metadata.paper_type`,
  `metadata.oecd_l1` and `oecd_l2`, the match tables' `service`,
  `source.input_format`, `eq[].comp` and `validation.issues[].severity`. Export
  maps a foreign reference type (`journal-article`, `article`) into the enum and
  drops an off-vocabulary classifier label with a logged warning instead of
  failing the paper. Every token is snake_case: `paper_type` `meta_analysis` and
  `case_study`, `section_type` `data_availability` (was `open_data`), and
  `xref_tier` `paren_numeric`, `flattened_superscript`, `author_year`.
  `input_format` names the format rather than the file extension: `jats` for
  bibr's XML input (was `xml`), `html` for `.htm` too, and `tei` for GROBID
  TEI, which converters into this format write.
- `section[]` holds only the paper's sections. Captions and footnotes are no
  longer sections of their own (`section_type` `figure`, `table` or `footnote`,
  with a made-up header such as "Figure 2" or "Footnote 3"). Their text
  stays in `text[]`, after the body, one row per whole caption or note (not
  per sentence), with a `null` `section_id`, so text search
  still finds them and "the text of Results" is the running text of Results.
  `figure[]` and `table[]` gain `text_id`, the caption's row, and their
  `section_id` is now the section they are printed in (it was the caption's
  section). The new `footnote[]` table has one row per footnote or endnote:
  `footnote_id`, the printed marker as `label` ("1", "*", "†") and `text_id`.
  The `figure`, `table` and `footnote` section types remain for printed
  headings such as "Figures" or "Notes".
- Every id is a 1-based position in document order. Section ids have no gaps
  (a section added late, such as an unheaded abstract, is numbered where its
  text is), and figure and table ids count the paper's figures and tables in
  document order on every input; PDF used the printed number, which left gaps
  where a figure was missed. Ids stay stable for the same input and bibr
  version, not across versions: to match rows across versions, use a figure's
  or table's printed label, a reference's DOI, or the text.
- `xref[].target_id` names a real row or is `null`. A `foot` reference points
  at `footnote[].footnote_id` (it held the footnote's ordinal); `equation`,
  `section` and `supplementary` references are `null` (they held the number
  they print, or `0`, which named no row).
- A `foot` reference no longer claims more than bibr knows. A PDF note printed
  without a mark (an author note, or text taken for a note) gets no reference;
  it used to get one with its ordinal as `contents`, which is printed nowhere.
  A footnote reference has no `start`/`end`: its mark is not in the sentence
  text, and the search for the digit landed on any number there. Its
  `text_id` is approximate: the last sentence of the paragraph holding the
  mark (DOCX), or the sentence before the note (PDF).
- One scale and one spelling per concept. Every score and confidence is 0–1:
  `bib_match[]` and `metadata_match[]` `score` was 0–100. The match tables'
  ISO 8601 `date` is `published_date`, like `metadata.published_date`; `bib[]`
  gains `published_date` (the printed date or year in ISO form), and
  consolidation fills it from the match instead of overwriting the printed
  `bib[].date`. Every DOI is bare and lowercase.
- A group author (a consortium, a JATS `<collab>`) is `author[].literal`, with
  `given` and `family` null; it was in `family`.
- `figure[].image` is a `data:` URI that names its media type
  (`data:image/jpeg;base64,…`); the format used to vary unannounced (JPEG
  crops, PNG composites, whatever a DOCX embedded).
- `extraction.warnings` holds `{code, message}` objects instead of prose
  strings. `code` is a stable UPPER_SNAKE code like the validation issue codes
  (`OCR_PAGE_FAILED`, `REF_SEG_CRF_FALLBACK`, `CROSSREF_ENRICHMENT_TIMEOUT`, …)
  and `message` carries the details (page, counts, exception type). The schema
  pins the code's form, not a list, so another producer can add codes of its
  own; bibr's are listed in the JSON schema reference. An OCR page failure now
  numbers its page from 1, like the region warnings. `bibr batch` ledgers count
  warnings by code, `bibr tables` writes them to `extraction_warnings`, and the
  `*_WARNING_PREFIX` constants in `bibr.extract` are replaced by
  `bibr.processing_warnings.WarningCode`. The OCR disk cache (format 10) and the
  enrichment sidecar (schema 4) store the new shape, so entries written by an
  earlier build are not reused.
- Every field and model in `docs/schema/bibr-export-v12.schema.json` has a
  description, and a test keeps it that way; the documentation site's JSON
  schema page shows them. Both schema documents carry a stable `$id` under
  `https://bibr.org/schema/`. The v11 and v10 schema files stay published,
  frozen. In the strict document `required` means *present*: every key bibr
  always writes is required, nullable or not, so a producer that drops a column
  fails validation. Nullable fields are spelled `"type": [T, "null"]`, the form
  R and code generators read, and identifiers (DOI, ORCID, ROR, SHA-256, ISO
  dates, country codes, CRediT URIs), ids (1-based), offsets and scores carry
  patterns and bounds.
- `xref[]`, `url[]` and `eq[]` carry `start`/`end`: the span of the item
  within its sentence's `text` in Unicode code points (0-based, end exclusive),
  or `null` when it cannot be located unambiguously. `eq[].verbatim` is now
  filled from it.
- Normalized fields sit next to the printed ones: `metadata.published_date`
  (ISO 8601, as precise as printed), `license_url` and `license_spdx`
  (Creative Commons with a known version, CC0), `language`, `pmid`, `pmcid` and
  `arxiv` (declared by JATS/HTML inputs; arXiv also from an arXiv DOI or the
  page-1 arXiv stamp), and `author[].credit_roles` (CRediT term URIs matched
  from the printed roles).
- `source.file_hash` (the first 16 hex characters of the input's SHA-256) is
  replaced by `source.sha256`, the whole digest. `extraction.bibr_version` and
  `build_sha` are replaced by `extraction.producer` {`name`, `version`,
  `build_sha`}, so another tool writing this format can say so. `producer` is
  the software that extracted the content; the new `extraction.converter`, of
  the same shape, names a tool that wrote another extractor's output into this
  format, and is `null` in bibr's own exports. A file converted from GROBID TEI
  has producer `grobid` and the converter, and its `source` is the PDF GROBID
  read when the converter has it, else the TEI (`input_format` `tei`), so
  `source.sha256` joins it to a bibr export of the same PDF. A converter
  keeps the producer's `completed_at` (now the time the content was
  extracted) when it has it, takes `paper_id` from `source.file_name`, and
  starts its own warning codes with its name (`METACHECK_…`); a tool that
  rewrites an export keeps its `schema_version` and every key, including
  those of a later 12.x it does not know.
- `paper_id` is required and never `null`: `--paper-id`, else the input file's
  stem, as `bibr batch` and metacheck already name papers. It used to be the
  DOI, which changed whenever a later bibr read the DOI differently. `bibr
  batch` writes its own corpus-unique id (the name of the JSON file) into each
  export.
- One geometry convention for every bounding box: `[x0, y0, x1, y1]` in PDF
  points on the page as displayed, measured from the top-left corner. The new
  `extraction.pages` gives each page's width and height. `float_parts[].bbox`
  and the caption and region boxes were in the layout model's 0–1000 space;
  `text_regions[].bbox_2d` was in points from the bottom-left and is now `bbox`,
  with the row's `page_number` replacing the per-row `page_w`/`page_h`.
  `extraction.regions` drops `bbox_height`/`bbox_width` (derivable from
  `bbox`), and its `char_density` and `estimated_line_height` are now per point.
- The match tables carry the identifiers Crossref records hold: `author[]`
  entries gain `orcid` and `affiliation` (name and ROR ID), and each record
  gains `funder` (name, Open Funder Registry DOI, ROR ID, award numbers),
  `license_url` (the version-of-record license; text-mining licenses are
  skipped) and `license_spdx`.
- New `affiliation_match` and `funding_match` tables hold the ROR organization
  matched to each affiliation string and printed funder name (ROR ID, name,
  country code, and for funders the Open Funder Registry DOI). They are filled
  when enrichment runs (`--crossref`), with `ROR_ENRICH=false` to skip ROR. Only
  ROR's own recommended (`chosen`) match is kept; strings without one stay
  unmatched. `ROR_CLIENT_ID` raises ROR's rate limit from 50 to 2000 requests
  per 5 minutes; matching is capped per paper by `ROR_ENRICH_TIMEOUT` and never
  holds an export back.
- The schema documents in `docs/schema/` are dedicated to the public domain
  under CC0 1.0; the software stays AGPL. Example valid, invalid and
  newer-minor exports live in `tests/fixtures/schema_conformance/`, checked
  against both schema documents.
- `Result` exposes every root table as `Records`, now including
  `affiliation`, `funding`, `footnote` and the `metadata_match`,
  `affiliation_match` and `funding_match` tables.
- `figure[]` and `table[]` gain `label`: what the caption prints after the
  word, as printed without whitespace (`3`, `3.1`, `S2`, `A1`, `IV`, `C`); a
  "Supplementary Table 4" caption is labelled `S4`. It is read from PDF, DOCX
  and HTML captions and from the JATS `<label>`, and is `null` when none was
  printed or detected. Figure and table references now resolve by it, compared
  case-insensitively without whitespace, instead of taking the printed number
  as the id: "Table 3.1", "Table S2", "Figure A1" and "Table IV" link, and a
  float bibr missed no longer shifts every later link. When no float of a kind
  has a label, "Figure N" links the N-th figure by page and reading order. A
  reference whose label names no float, or two, is still exported with a
  `null` `target_id`; it used to be dropped. A piece captioned as a
  continuation ("Table 3 (continued)", "Figure 3. Cont.") that was not merged
  into its float does not count as a second float with that label. "Table S2" and "Supplementary
  Table 2" are `table` references when an extracted table carries that label
  and `supplementary` ones otherwise. `extraction.diagnostics.xref_tier`
  records `label` or `position` for every figure and table reference.
- 12.x is additive-only: new optional fields and new enum values may appear in
  any 12.x release, and the reader model and reader schema accept both. Any
  rename, move, removal, type change, new required key or dropped enum value
  needs 13.0.

### Fixed

- `table[].contents` keeps the cell text the paper printed. The OCR engines
  return a PDF's tables as HTML, and HTML and ePub input carries them as HTML
  too. That HTML was read with pandas type inference, so every column that
  looked numeric was rewritten: "2.50" became "2.5", "007" became "7",
  "1,234" became "1234", a decimal comma was read as a thousands separator
  ("1,5" became "15", "0,25" became "25"), an integer column with one empty
  cell came out as "12.0", and "TRUE" became "True". A cell printing "NA",
  "n/a" or "None" came out empty from a PDF and as "nan" from HTML or ePub,
  where every empty cell was "nan" too. A PDF table without a header row,
  whose first row becomes the header, could get headers such as "2019.0".
  Cells now keep their printed text and an empty cell is "", and tables keep
  the shape they had before. A table printed as an image is now kept
  with its caption and empty `contents`, so a mention of it resolves: in HTML
  and ePub input a `<table>` with no cell text whose caption prints a table
  label ("Table 3. ..."), in JATS a `<table-wrap>` with a label or caption but
  no `<table>` with rows. An HTML table with a span such as `colspan="2px"` is no longer
  dropped.
- A table continued across pages no longer gets its repeated header as a data
  row in the middle of `contents` when the later page prints the header with
  different spacing, case, dashes or punctuation ("Mean(SD)" under
  "Mean (SD)", "p value" under "p-value"), as per-page OCR often reads it.
- Text from DOCX, JATS, HTML, ePub and the PDF text layer no longer goes
  through the late clean-up meant for OCR output. That clean-up ran on every
  sentence of every input. It fused "a 2 x 2 x 3 design" into "a2x2x3" and
  "Items 1 2 3" into "Items 123". It deleted the underscore from identifiers,
  file names and email addresses (`age_group`, `NM_022770`, `RRID:SCR_003070`,
  `john_smith@uni.edu`), and it turned `10^6` into `106`. It also read two
  literal dollar signs as a math span and deleted them along with the
  underscores between (`df$age_group`, "US$ 60 to US$ 1,419", `$SAMPLE_R1`).
  Each sentence now records whether any of its text came from OCR, and only
  OCR text gets those repairs. DOCX inline equations are still unwrapped and
  flattened, one glued to a word included ("the $n$th" reads "the nth"),
  because each sentence also records the `$…$` spans the parser wrote.
  Elsewhere in document text only a tightly delimited `$…$` counts as math,
  as JATS tex-math writes it; two literal dollars that happen to fit that
  shape are still unwrapped. In OCR text, `_x` and `^x` are now flattened only
  inside `$…$` and `\(…\)` math, email addresses are protected like URLs, and
  "2 x 2" and "2 × 2" are no longer fused. OCR text keeps the spaced-run
  collapse, which repairs OCR's character spacing ("1 7. 9 0 6"), so an OCR'd
  "Items 1 2 3" still reads "Items 123"; narrowing it further waits on data
  from a GLM-OCR run. The PDF text layer sometimes extracts a superscript or
  subscript as a separate token ("R 2 ,", "r 2 ¼"). The old clean-up fused
  those by accident; they are now kept as extracted.
- The per-region OCR clean-up no longer touches the PDF text layer, and it no
  longer damages formulas and numbers. On text-layer regions it split "U.S."
  into "U. S." and "e.g." into "e. g.", and it broke a DOI that opens a line
  into "10. 1038/…". It cut a table of contents longer than 2,048 characters
  with spaced dot leaders to its first entry. It also turned a printed "* p <
  .05" into a bullet. Those regions are now only trimmed. In OCR output, a
  formula that starts with `\theta`, `\tau`, `\text`, `\tilde` or `\times` no
  longer loses its leading `\t`; on the default Paddle profile it exported as
  "heta_{t+1} = …". The list-marker spacing ("1.text" → "1. text") no longer
  applies to decimals, DOIs, abbreviations or formulas. The repeated-content
  trimmer keeps the text after a repeated run instead of dropping the rest of
  the region. A formula region that holds two formulas ("\(a\) + \(b\)") keeps
  its delimiters, and one wrapped in single dollars is no longer nested inside
  `$$`.
- HTML and ePub text keeps inline markup attached to its word. The parser put a
  space around every element, so `H<sub>2</sub>O` read "H 2 O",
  `m<sup>6</sup>A` "m 6 A" and a linked citation "( Figure 1 )". Now only
  block-level elements separate words, as in the JATS parser; like there, an
  exponent joins its number (`10<sup>6</sup>` reads "106"). eLife publishes
  each of its 984 test articles as both HTML and JATS. The share of HTML
  sentences that also appear word for word in the same article's JATS rose
  from about 24% to 41%.
- JATS and HTML text no longer splits inline MathML at the whitespace
  publishers put between its elements. PLOS and eLife pretty-print MathML
  (`<mi>t</mi> <mo>-</mo> <mn>1</mn>`), and the parsers kept that whitespace,
  so a formula read "( 0 , 2 . 5 )" or "y ¯ t - 1". The late clean-up's
  spaced-run collapse fused some of those runs back by accident, but it also
  fused prose, and it no longer touches document text. The parsers now drop
  whitespace between MathML elements as a renderer does, so "(0,2.5)" and
  "y¯t-1" read as they do from a publisher that writes none. They keep a
  space where it separates words: "ln dbh", "0.93 GeV", "direct effect"
  spelled one letter per element, a word after a comma, a function name
  before a bare argument ("sin x"), and the text around the formula. They
  also keep it between two numbers, so the parts of a fraction read "1 2" and
  not "12"; an index pair (x with 1 below and 2 above) reads "x1 2". An
  `<mspace>` (`\quad`, `\,`) and, in JATS as in HTML, a matrix row or cell
  now separate the text around them, whitespace or not: "E_{t-1} \quad
  0<λ≤1" reads "Et-1 0<λ≤1", and a matrix that read "(2112)" reads
  "( 2 1 1 2)". Other letters and digits still close up, across a fraction
  bar, a product or a script too, as they do from a publisher that writes no
  whitespace: a/b reads "ab", and a unit set as an upright `<mi>` after a
  number reads "5m" (in `<mtext>` it keeps its space).
  Measured against the parsers that kept every such space, on the 885 test
  articles that contain a MathML element (PMC_sample_1943, eLife_984 as JATS
  and as HTML, PLOS_1000): 3,329 of their 40,308 formulas read differently.
  Spaced decimals ("2 . 5") in them fell from 121 to none and spaced
  differences ("t - 1") from 346 to none. No space between two digits and
  none at an `<mspace>` is lost. Twenty closed spaces join two tokens of two
  letters or more, between terms of a product ("m3hgNa"), the parts of a
  fraction ("e-diλi") or a sum and its limits. Dropping every such space
  would have joined 187 ("lndbh", "directeffect").
- The Ollama provider sent every request to `/chat/completions` under
  `LLM_OLLAMA_BASE_URL`, and Ollama answers that with 404. With the default URL,
  `http://localhost:11434`, which is also what `bibr setup` writes, every paper
  failed at its first LLM call, and so did the setup wizard's connection test.
  Ollama serves its OpenAI-compatible API under `/v1`. bibr now adds `/v1`
  unless the URL already ends in it, so both forms work. The wizard no longer
  lists models from `/v1/v1/models` when the URL is typed with `/v1`.
- `bibr doctor` and the `bibr setup` connection test now send their test
  request through the provider adapter that extraction uses. Both built their
  own client, with a 64-token cap. For the default `gemini-3.5-flash-lite`
  they left out the thinking budget the Gemini adapter always sends. For
  OpenAI they sent `max_tokens`, which reasoning models such as the wizard's
  `gpt-5-nano` reject, where the adapter sends `max_completion_tokens`. The
  test could therefore fail a setup that `bibr chew` runs. Doctor also skipped
  Ollama; it now tests it like any other provider. When the wizard's test
  fails for Ollama, it offers to change the base URL. It used to ask for an
  API key and write the answer to `.env` as a line with no name, `=<key>`.
  Both tests now give up after twice `LLM_TIMEOUT_SECONDS`, the limit `bibr
  chew` puts on one LLM call. A server that accepted the request and never
  answered kept them waiting on the SDK's own timeout instead, which for the
  OpenAI SDK that Ollama and OpenAI-compatible servers go through is 600 s per
  attempt.
- `bibr setup` could leave an older `LLM_API_KEY`, `LLM_BASE_URL` or
  `LLM_BACKEND` in effect behind the provider just chosen. Merging into an
  existing `.env`, the default, keeps every key the wizard does not write, and
  `~/.bibr/.env` still applies under a new `./.env`. The Google, Anthropic and
  Groq adapters send `LLM_API_KEY` in place of their own key, so switching
  from OpenAI to Google sent the old OpenAI key to Gemini, while the
  connection test, which used the typed key, passed. Choosing a provider now
  also writes `LLM_BACKEND=cloud`, and a blank `LLM_API_KEY` or `LLM_BASE_URL`
  where none was entered, and the connection test uses those same values.
- `bibr doctor` checks the LLM the way `bibr chew` does. `LLM_BACKEND=local` is
  resolved to the backend chew would start on this machine; doctor used to
  check it as a cloud provider and ask for a key. The provider's credentials are
  checked by its adapter. `LLM_API_KEY` now counts for Google, Anthropic and
  Groq, and an OpenAI-compatible server set with `LLM_BASE_URL` needs no key.
  A managed local backend fails when chew's preflight would refuse it, for
  example vLLM on a machine with no NVIDIA GPU, which doctor passed. An unknown
  `LLM_BACKEND` value is reported instead of being checked as cloud.
- `bibr doctor`'s OCR check now fails where `bibr chew` refuses a PDF. With
  the default `OCR_BACKEND=paddle` on Windows, or on Linux without a GPU that
  fits paddle-vllm, the automatic chain is glm-llama alone. Doctor now looks
  for llama.cpp there, where it used to warn that availability was unverified.
  It fails when no runtime of the chain can start, when
  `OCR_BACKEND=paddle-vllm` has no GPU that fits it, and when a cloud vision
  backend (`gemini`, `openai`, `anthropic`) has no API key.
- `bibr doctor` no longer fails when the working directory has no `.env`.
  Settings come from `~/.bibr/.env` and `./.env`, or from `BIBR_ENV_FILE`.
  Doctor now names the files it read, and warns when there are none, because
  configuration from the environment alone is valid. `bibr preset` uses the
  file whose values are in effect, which is the last existing file of that
  chain. `preset save` and `preset use` therefore work when the configuration
  lives in `~/.bibr/.env`, and `save`, `use` and `deactivate` name the file.
  The demo's preset picker reads the active preset from the same file.
- `bibr doctor` reports a missing `uv` as a warning instead of a failure.
  `python -m pip install bibr` is a documented setup, and only the uv-managed
  vLLM and MLX-VLM runners need uv. Their own checks still fail without it.
- Reference enrichment (`--crossref`) rejected the correct Crossref or
  resolver record for references printed in most citation styles. The
  author-initials check took "and", "et" or "und" for a first name
  ("Smith, J. and Weber, E. U." read Weber as "A."), kept only the middle
  initial of "Jeffery M. Sobal", and vetoed a record whenever a reference cut
  short by "et al." printed one of two co-authors who share a surname. It now
  reads a given name on either side of the surname, treats conjunctions as
  separators, skips the second half of a hyphenated surname and pools
  co-authors who share one, and still rejects a different person with the same
  surname. Surnames now also match without their diacritics or with an umlaut
  spelled out ("González" and "Gonzalez", "Müller" and "Mueller"), and an
  organization author no longer counts as a surname that fails to match.
- A title search could accept a different work from the one the reference's
  own DOI names. When the DOI lookup failed with a timeout or server error,
  the Crossref title search took any record with a matching title, a preprint
  of the article for example, and consolidation filled the reference's gaps
  from it. A printed DOI that parses now admits only a record carrying that
  DOI, on the Crossref title and fingerprint searches as on the resolver
  fallback, which already had this rule. The Crossref search, the resolver
  search and the resolver fallback now share one matcher, so their rules
  cannot drift apart again.
- A generic title ("Introduction", "Emotion regulation") matched any record of
  that title by an author of the same surname in about the same year. A title
  of three words or fewer is now rejected when the printed volume disagrees
  with the record's, or when the printed container does, unless the printed
  volume and first page both agree with the record's: those identify the
  article even when the container is an abbreviation the matcher cannot
  expand, such as "PNAS". Any title is rejected when both the printed volume
  and the first page disagree with the record's. A value missing on either
  side counts for nothing.
- Crossref titles and journal names kept their inline markup
  (`CO<sub>2</sub>`, `<i>Drosophila</i>`, `&amp;`). The tags cost enough
  similarity to reject the correct record, and an accepted match carried them
  into `bib_match` and, through consolidation, into `bib`. Titles and
  container names from Crossref and the resolver are now plain text.
- On 2,312 printed references, each paired with the recorded Crossref record
  its printed DOI names, the title matcher with all of the changes above
  rejects 130 of those records, down from 798, mostly where the printed title
  differs from the deposited one.
- With `BIBR_RESOLVER_AUTHORITATIVE=true`, a failed resolver prefetch left
  every reference without a DOI unmatched and never asked Crossref, although
  the log said it was falling back to Crossref. Those references now go to
  Crossref. `BIBR_RESOLVER_SEARCH_CONCURRENCY` below 1, one way to make the
  prefetch fail, is now rejected when settings load.
- When the resolver fallback (`BIBR_RESOLVER_FALLBACK_SOURCES`) reached its
  deadline it discarded every search that had already answered. Each
  reference's result is now applied as it arrives, the deadline cancels only
  the searches still outstanding, and the warning says how many were left.
- A resolver that answered `/health` with JSON other than an object
  (`["ok"]`) failed the paper's whole reference enrichment, and one fallback
  candidate with `"authors": null` stopped the fallback for every remaining
  reference. Such a health answer now means unhealthy, a non-object `/works`
  answer is a miss, malformed authors count as none, and an error on one
  reference costs only that reference.
- A matched Crossref monograph, edited or reference book, book part, report
  component or database came back with `bib_type` `other`, because
  `migrate_bib_type` knew only the BibTeX names and a few Crossref ones.
  `consolidate="replace"` then overwrote a printed `book` with it. These
  Crossref types now map to `book`, `book_chapter`, `report` and `dataset`, a
  match's `other` still fills a missing type but never replaces a printed one,
  and a type that is not a string maps to `other` instead of raising.
- Resuming a `bibr batch` run now runs again the papers that failed because
  of the run or a service rather than the paper; they used to wait for
  `--retry-failed`, which also re-runs every genuine failure. A local OCR or
  LLM server that went down or could not start failed each later paper with
  the pipeline's own code (`ocr_failed`, `layout_failed`, …), and the next run
  skipped them all. Those papers are now recorded as `upstream_unavailable`,
  the code the remote executor uses for the same outage, and
  `ChewFailure.outage` tells library users the same. Papers failed with
  401/403 when the serve rejected the token are always picked up again. A
  paper that crashed, hit an outage, or ran out of remote transient retries
  is picked up again until it has failed that way three times, since the
  cause can still be the paper (a prompt that brings the LLM server down, a
  model reply the serve reports as a 502); after that it waits for
  `--retry-failed`, so a batch still finishes. Timeouts are not picked up: a
  paper can be too slow on its own.
- An OCR server that stopped answering mid-file left the regions it refused
  blank with a warning, so the file failed as `ocr_mostly_failed`, which
  resume skipped, or passed with text missing. A region or page whose OCR
  request was refused or dropped once the retries ran out now fails the file
  as an outage, as an `UpstreamServiceError` from the OCR service already did.
  A busy answer (429, 502, 503) still leaves only that region blank, with a
  warning.
- A crash in one chunk no longer fails every paper in it. `bibr.chew()` on a
  list or directory lost every result when any chunk raised, and `bibr batch`
  recorded the whole chunk as `chunk_error`, which resume then skipped. The
  papers a crashed chunk left unfinished now run again one by one, with the
  crashed chunk's pages released first, and only a paper that crashes on its
  own fails with `chunk_error`.
- A job the serve failed with 504 (`PIPELINE_TIMEOUT`) was treated as an OCR/LLM
  outage: it was resubmitted up to `--retries` times, each run cost a full
  timeout of a serve slot, in-flight shrank each time, and the paper was
  recorded as `upstream_unavailable`. It is now retried once without shrinking
  in-flight and then recorded as `pipeline_timeout`, and "timed out" in a
  failed job's text no longer marks it as an outage.
- Adding a file to a `bibr batch` input set could rename a paper already
  processed. Ids were disambiguated only within one run, so a second
  `paper.pdf` turned the existing `paper` into `paper-<sha8>`: both files were
  processed again, and the stale `paper.json` stayed in the ledger and the
  tables as a third paper. A file now keeps the id the ledger recorded for its
  path, and only the newcomer gets a suffix. An input named `run_info` gets
  one too; its export used to replace `run_info.json` mid-run and was then
  overwritten by it, while the ledger said `ok`. `bibr.chew()` on a list or
  directory gives stem collisions the same `<stem>-<sha8>` ids, where all of
  them got the bare stem and `bibr.write_tables()` refused the batch.
- A `bibr batch` run killed mid-write (out of memory, a full disk) left the
  ledger's last line without a newline, and the next run's first record was
  glued onto it and lost, so that paper ran again. The first record of a run
  now starts on a line of its own.
- `bibr batch` rebuilt no Parquet tables at all once its out dir held an export
  of an older schema major, as after resuming a 0.5.x run with 12.0: the whole
  rebuild failed on every later run. Such exports are now left out of the
  tables with a warning.
- Ctrl-C during `Chewer.chew()`, which `bibr batch` uses locally, left the call
  running on the Chewer's event loop, and `close()` then resumed it alongside
  the teardown, where it could start an OCR server after the OCR shutdown had
  run. The Chewer now drives its own loop through `asyncio.Runner`, so Ctrl-C
  cancels the call and waits for it to unwind before anything is closed; the
  thread's current event loop is left alone.
- `bibr.write_tables()` names the input files behind a duplicate `paper_id`,
  not the `paper_id` twice.
- llama.cpp server: an explicit `LLM_MAX_TOKENS`/`LLM_MAX_INPUT_CHARS`/
  `LLM_REF_SEG_WINDOW_CHARS` is kept with a warning, while unset values
  scale with `LLM_LLAMA_CPP_CONTEXT_SIZE`.
- llama.cpp server: `*_LLAMA_CPP_EXTRA_ARGS` documents the separate-token
  form (`--flag value`, not `--flag=value`) and names the setting on bad
  quoting, and `-hfr` joins the `--hf-repo` aliases for the
  identity-override check.
- llama.cpp server: startup failures report the END of the stderr tail, and
  the conservative-args retry fires only on argument-parse errors (unknown
  flags and rejected flag values) — load and OOM failures fail fast. A
  `/health` 200 is accepted only if the
  process is still alive after a short grace period, so a port-racing
  sibling's server is not mistaken for ours.
- llama.cpp server: Ctrl-C during startup shuts down the child instead of
  orphaning it; `--parallel N` is parsed for any N.
- llmster: an `lms` timeout names the subcommand that timed out, and the
  `load` hint suggests `LLM_LLMSTER_CONTEXT_LENGTH`. Reusing a pre-loaded
  identifier validates it still serves that model. Unset `LLM_TIMEOUT_SECONDS`,
  `LLM_MAX_CONCURRENCY` and `LLM_RATE_LIMIT_RPM` are raised to 300 s, 1 and
  600 for the loopback server, matching the other local backends.
- vLLM LLM server: readiness needs a second agreeing `/health` 200 after a
  grace period, and startup failures report the END of the stderr tail.
- vllm-mlx OCR server: Ctrl-C during startup shuts down the child, matching
  the other managed servers; startup failures report the END of the stderr
  tail, as does the paddle vLLM OCR server.
- Rapid-MLX LLM setup raises unset `LLM_RATE_LIMIT_RPM` for the loopback
  server like the other local backends.
- Rapid-MLX OCR: a failed engine restart no longer discards the region that
  triggered it — the caller gets its own transcription, and the restart
  error surfaces on the next request where the generation is retried.
  Callers parked on the recycle drain retry the failed restart once, and
  get an `UpstreamServiceError` naming the failure instead of a bare
  assert when the retry fails too, and replacing a dead generation
  shuts down its HTTP pool and handles first.
- `resolve_llm_backend` accepts the pipeline's settings when checking
  rapid-mlx availability instead of only the global snapshot.
- Cloud vision OCR: `settings.llm.api_key` is only forwarded to a vision
  provider when it is a real key for the same provider and the LLM is not
  pointed at another endpoint — cross-provider keys, managed-local
  placeholders and a set `llm.base_url` fall through to the provider's own
  key or SDK env fallback. The plain-HTTP refusal for a public vision
  endpoint now also covers that fallback: an `OPENAI_API_KEY` the OpenAI SDK
  would send is refused like a forwarded LLM key. Cloud vision backends
  (`gemini`, `openai`, `anthropic`) are no longer torn down between chunks:
  there are no local weights to reclaim.
- Rate limiter: the strict-interval Redis Lua script works in whole
  milliseconds (Redis truncates a Lua number reply to an integer, so a
  fractional-second wait became 0), and the key TTL spans the queued
  horizon so concurrent sleepers do not lose their place; the sliding
  window quantizes to the same. The strict-interval key is now
  `rate_limit:<resource>:next_allowed_ms`: it stores epoch milliseconds
  while older releases stored epoch seconds under `next_allowed`, so the
  two never read each other's values when old and new processes share one
  Redis.
- Circuit breaker: a waiter that times out no longer forces the breaker
  back to OPEN under a still-running probe — only the probe's completion
  owns the transition (a provably dead probe still fails fast).
- Batch manifests: `content_sha256` streams inputs in 1 MiB chunks instead
  of loading whole files. Validation still recomputes the hash for bytes
  entering the pipeline, so the up-front manifest pass remains for now.
- Layout preprocessing threads the resize column pass in row blocks; output
  is bit-identical to the serial loop. The OOM batch-halving retry restores
  torch.compile padding afterwards instead of leaving it disabled.

- `bibr batch` no longer refuses PDFs on a core install for lack of OpenCV. Its
  preflight required `cv2` for every PDF and suggested `uv sync --extra ml`,
  but only the torch layout path imports cv2. A core install runs layout
  through ONNX Runtime, and `bibr chew` already accepted the same PDFs. The two
  commands now share one check. Without torch, neither needs OpenCV. With
  torch, both refuse a missing or broken `cv2` before any model loads, and
  suggest `uv sync --extra torch`, or reinstalling `opencv-python-headless`
  for a broken one. `bibr chew` on a core install also runs the local OCR
  runtime check again, as `bibr batch` does. It stops before the layout model
  loads when no local OCR runtime can start, for example `--ocr paddle-vllm`
  without an NVIDIA GPU.
- A GPU install (`onnxruntime-gpu[cuda,cudnn]`, the `gpu` extra) ran bibr's
  ONNX models on the CPU. The CUDA and cuDNN libraries those wheels install
  are found only after `onnxruntime.preload_dlls()` loads them, and bibr never
  called it, so onnxruntime could not start its CUDA provider and fell back
  to CPU. bibr now calls it before it opens a CUDA session. The logs still
  said `cuda`, because they named the device bibr asked for. The layout
  detector and sentence segmenter now log the device their session got, and
  any ONNX model that loses CUDA this way logs a warning.
- On a GPU, bibr's ONNX models held on to all the GPU memory they had ever
  used. Page batches, reference lists and section headers come in different
  sizes, and each new size added memory, so a batch run filled a 24 GB card
  after 11 papers and every later paper failed. Each ONNX Runtime run on CUDA
  now ends by freeing the memory it no longer uses (onnxruntime's arena
  shrinkage), so GPU memory follows the model calls in flight.
- The OCR disk cache key now includes the layout checkpoint (`LAYOUT_MODEL_ID`),
  the ONNX layout bundle (`LAYOUT_ONNX_MODEL_ID`, `LAYOUT_ONNX_REVISION`) and
  `ML_RUNTIME`. It held only the torch revision, so moving the ONNX bundle,
  which is what the default runtime loads, replayed the previous model's cached
  regions.
- JATS footnotes printed under a heading of their own (an `<fn-group>` inside a
  `<sec>`, as Europe PMC writes them) were dropped; they are now footnotes like
  a back-matter `<fn-group>`. A JATS footnote keeps its printed `<label>`.
- A GPU install could run bibr's ONNX models on the CPU. The core `onnxruntime`
  package and the `gpu` extra's `onnxruntime-gpu` write the same `onnxruntime/`
  directory, and `uv sync --extra gpu` writes both at once, so either build
  could end up loaded. The documented remedy, `uv pip install
  'onnxruntime-gpu[cuda,cudnn]'` after the sync, did nothing, because
  `onnxruntime-gpu` was already installed. `bibr setup` now reinstalls the
  `onnxruntime-gpu` version the sync chose, which writes the GPU build's files
  last, and checks that the GPU build is the one that loads. The install guide
  and the tester guide give the same step: `uv pip install --reinstall-package
  onnxruntime-gpu "onnxruntime-gpu[cuda,cudnn]==1.26.0"`. `onnxruntime` stays
  installed, because `uv run` reinstalls a missing one and its files would
  replace the GPU build's. When both packages are installed and the CPU build is
  the one loaded, bibr logs a warning once, with the command that fixes it.
- The demo notebooks read each section's classification score from
  `extraction.diagnostics.section_classification`; since 12.0 moved it there,
  they showed 0% for every section.
- A title that opens with a parenthetical, such as "(Rural) Clinics as layered
  civic organizations" or "(Re)thinking …", keeps it. The metadata LLM can read
  the parenthetical as an annotation and return only the rest of the title. Title
  grounding accepted that because the rest is still printed verbatim, and once a
  front-matter record is selected the layout title is not consulted. Grounding
  now restores the parenthetical from the selected record's printed title row
  and adds a `VAL_TITLE_REGROUNDED` warning with evidence
  `reason:title_leading_parenthetical_dropped`. Numbering such as "(1)" or
  "(iv)" and article-type labels such as "(Review)" or "(Original Article)" are
  still left out.
- The reference under-extraction warning (`REF_UNDER_EXTRACTION_SUSPECTED` in
  `extraction.warnings`) now also covers numeric citation styles. It previously
  counted only author-year citations, so a numbered paper whose reference
  region was lost to OCR was never flagged. It now also counts the distinct
  reference numbers cited by bracket and superscript markers, up to the highest
  number where at least half of 1..n are cited, and warns when fewer than half
  that many references were parsed (at least 15 cited).
- The OCR disk cache (`CACHE_OCR`, on by default in the local demo) is now keyed
  on the bibr version too, so an upgraded bibr no longer reuses rendering,
  layout, native-text and OCR bundles made by the previous release. The key
  still cannot see source changes between releases; the `CACHE_OCR` description
  now says to use a fresh `CACHE_OCR_DIR` per revision when comparing such
  changes, and to leave the cache off when timing runs.
- `replace` consolidation (`CROSSREF_CONSOLIDATE=replace`, `--consolidate=replace`,
  `Result.consolidate("replace")`) no longer overwrites printed reference fields,
  the DOI included, from a match found by bibliographic search. That search
  accepts a title similarity of 80, so a near-miss hit could rewrite a correct
  printed volume, issue or page range. A printed value is now overwritten only by
  a match that carries the reference's own printed DOI, compared
  case-insensitively. Other matches still fill empty fields, as in `fill` mode.
- Reference-segmentation training capture (`REF_TRAINING_DATA_DIR`) no longer
  mixes geometry-segmenter predictions in with LLM segmentation labels; only
  LLM output is captured. Under the default `geom` strategy, that means only
  blocks the cascade sends to the LLM; set `REF_SEG_STRATEGY=llm` to label
  every block. Records from earlier versions lack `provenance` and may contain
  geometry output; discard them or capture into a fresh directory.
- Documentation and the `ML_PAPER_CLASSIFIER_MODEL_ID` setting description no
  longer call the default paper classifier SPECTER2-based; its model card
  documents an `all-MiniLM-L6-v2` encoder. The Classifiers guide also notes that
  the model has no `corrigendum` paper-type class and predicts 32 of the 36 OECD
  subdomains.
- Sentence DOI candidates in `extraction.identity.receipt` now record the layout
  region they were read from; `region_index` was previously always `null`. With
  `page`, it matches the `page` and `index` of an `extraction.regions` row: the
  region's position on that page after OCR post-processing renumbers merged
  regions. It stays `null` when no layout region is recorded for the sentence,
  or when the sentence is printed on a later page than the region that began its
  paragraph. The v11 export schema changes only by describing these fields.
- `bibr.Result(data)` loads exports written by newer releases of the same
  major version, as the additive-only policy promises. It previously rejected
  any unknown key and any `schema_version` other than the exact one it writes.
  Unknown keys at any nesting level are now kept in `result.data` and in the
  model's `model_extra`, and enum values it does not know yet are accepted. A
  different major `schema_version` (`11.x`, `13.x`) and known fields of the
  wrong type are still rejected. What bibr writes is still validated against
  the strict models.
- Tables captioned "Table 3.1" and "Table 3.2" on adjacent pages are no
  longer merged as one table continued across pages, which appended Table
  3.2's rows to Table 3.1 and lost its caption. Only the same whole label
  continues a table.
- "Supplementary Table 4" and "Supplementary Figure 4" give one
  `supplementary` xref; they also gave a `table` or `figure` xref to the
  paper's own Table 4 or Figure 4.
- The citation linker no longer turns test statistics, equation references,
  locators and unit exponents into bibliography links. In a paper with a
  numbered reference list, the degrees of freedom in "F(3, 84) = 4.49", "t(45)
  = 2.10" or "χ 2 (6) = 22.03" linked references 3, 84, 45 and 6, as did "Eq.
  (5)", "Equation (6)" and both labels of "Eqs. (7) and (8)". A
  parenthetical group glued to a one-letter or Greek statistic symbol, or
  followed by a comparison, is no longer a citation; a bracket group needs
  both ("F[2, 9] = 5.20"). "Fig.3", "Eq.5",
  "Tab.2", "Vol.12", "No.5", "pp.14-16" and "Exp.1" are no longer flattened
  superscript citations, and no longer switch that style on in a paper that
  cites with brackets. "25 cm^{2}" and "3 g cm $ ^{3} $" linked references 2
  and 3 and lost the exponent from the text; a length unit (cm, mm, km, µm,
  nm, ft) or "ms" before a superscript now keeps it as an exponent.
- Two distant numbers in one bracket, such as "[11, 33]" or "[91, 108]", are
  citations again in a paper that cites with brackets. They were taken for a
  confidence interval and dropped with no fallback; on 973 PLOS and 1,922 PMC
  JATS articles that cost 2,608 citation links. They stay intervals in a
  paper without bracket citations and after "CI", "IQR", "range" or
  "interval".
- A superscript citation group with a number the reference parser dropped
  links its other numbers, as a bracket group does. "form^{3-5}" with
  reference 4 missing linked nothing and was flattened into "form3-5"; it now
  links 3 and 5 and the marker is removed. A number past the last reference
  still rejects the group.
- The LLM citation step no longer re-offers citations the author-year
  matcher already linked. The matcher links "In Smith (2020)"; "Smith (2020)"
  went to the LLM again, and a different answer added a second, conflicting
  link. A group such as "(Smith, 2020; Jones, 2019; Brown, 2016)" is offered
  work by work, and every reference the LLM names for one citation is kept;
  before, the unresolved works were folded into the whole group and only one
  answer survived. An answer that names several references for a citation of
  one work, such as "(Smith, 2020)", links none of them; the last one was
  kept. The LLM may break a same-surname, same-year tie only with a reference
  that carries that surname and year: it could pick any reference, and the
  receipt kept the surname and year evidence for it.
- The LLM citation step sends at most 40 citations per request, run
  concurrently within the client's `LLM_MAX_CONCURRENCY` and rate limits.
  All candidates went into one request whose answer had to fit the
  8192-token `citation_max_tokens` cap; a paper with about a hundred
  candidate citations hit the cap and got no LLM link at all. A failed
  request now loses only its own citations.
- DOCX tables get their captions: a Caption-styled paragraph directly above
  or below a table is its caption and leaves the body text, as figure
  captions do; the table had none and the caption stayed in the body. A style
  based on Caption, such as pandoc's "Table Caption" and "Image Caption",
  counts as Caption-styled for tables and figures alike.
- A PDF caption the layout model tags as a figure title that opens with
  "Table S1", "Table A1" or "Supplementary Table 2" goes to the tables; it
  found no table and fell back into the body text.
- The 0.5.0 notes said evaluation, aspect scoring and the benchmark harness share
  `metrics_version=6`. That counter belongs to an aspect scorer and a benchmark
  harness that are not part of this repository. The evaluator here,
  `evaluation/evaluate.py`, records `metrics_version: 4`, as it did in 0.5.0 and
  0.5.1. No metric definition has changed since v4, so saved v4 evaluations need
  no re-scoring. Full printed names (`authors_fullname_f1`) were already its
  primary author metric in 0.5.0, with family-name-only `authors_f1` as a
  diagnostic.
- Statistics in `eq[]` keep their whole printed value: `p = 2.3 × 10−5`
  exported as `p = 2.3`, `p < 1e-10` as `p < 1` (which passes a p ≤ 1 check),
  `p = 0,05` and `p=0·008` as `p = 0`, and `r = .85–.94` as `r = .85`. `rhs`
  now holds, as printed, scientific notation (also as JATS, HTML and PDF text
  layers flatten the superscript, `× 10−5` and `10 −9`, and as OCR prints it,
  `\times 10^{-5}`), decimal commas, mid-dot decimals and ranges. A comma after
  a count is no decimal comma (`n = 9,7 cells` lists two groups), and a signed
  number after a space is the next column of a table flattened to text, not
  the end of a range (`β = 0.21 −0.05 0.47` gives `β = 0.21`). A value that
  runs on into a fraction or a time (`BF10<1/3`, `t = 12:30`) is no longer
  exported cut short; a count before a slash is kept, cut at the slash
  (`n = 12/20 cells` gives `n = 12`).
- Statistic names are read whole and exported as printed. A Greek letter,
  superscript or Δ (also ∆, U+2206) belongs to the name, so `η²p = .12` and
  `η 2 p` (partial eta squared) are no longer p-values, `ΔR²` is no longer
  `R²` and `τp` no longer a p. That holds where a PDF text layer puts the
  stacked scripts on lines of their own, `η\r\n2\r\np = 0.11` (a p-value
  before) and `ηp\r\n2 = .61` (not read at all before), as the extractor now
  reads each line break as one character. `p-value` and `P value`, `χ 2` (χ²
  as text layers print it), `r(df)`, `H(df)`, `Z`, `g`, `χ²` without df,
  `90% CI`, `r s` and `rs` (Spearman) and `d z` are recognised. A subscript
  printed after a space, such as the `h` of a Holm-adjusted `p h`, is no
  longer a statistic of its own. Consumers matching `p`, `χ²` or `ηp²` have to
  fold these spellings.
- `lhs`, `df` and `rhs` in `eq[]` are exported on one line: a run of
  whitespace, line breaks included, is one space, as in the sentence text.
  PDF text layers gave the lhs `Cohen’s \r\ndz`, the df `1, \r\n19` and the rhs
  `[0.30, \r\n0.42]`.
- Statistics that share parentheses or a formula with a recognised one are no
  longer dropped: `BF10` in `(p < .001, BF10 = 12.3)`, a second count in one
  clause, `\chi^2(1)` in `$\chi^2(1) = 3.84, p = .05$`. Statistics printed
  together share a group whichever pass reads them.
- No printed statistic is exported twice. Inline LaTeX (`$t(28) = 2.10$`) and
  `Cohen's d` were exported again under a new group, while two identical
  printed values were merged into one. LaTeX comparators (`\leq`, `\geqslant`,
  `\neq`, …) are read, a LaTeX statistic splits its df, and a `$` before a
  digit is currency unless it opens math. Group ids no longer skip numbers,
  the LLM fallback continues the regex results' ids, and the broad pass no
  longer rescans a long token from each of its characters.
- `eq[]` spans no longer place a value at a longer one that begins with it
  (`n = 1` at the `n = 16` earlier in the sentence), and locate a value that
  late clean-up respaced (`10 − 6` printed as `10 −6`) and a name it printed
  without its spaces (`η p 2`, read from MathML, printed as `ηp2`). Late
  clean-up prints `\leqslant` and `\geqslant` as `≤` and `≥`, not `≤slant`.
- A supplement's DOI no longer becomes the paper's DOI (`metadata.doi`, which
  also keys self-DOI enrichment) or makes its selection abstain. APA's
  "Supplemental materials: https://doi.org/….supp", Copernicus `-supplement`,
  MDPI `/s1`, PeerJ `/supp-1`, `/fig-1` and `/table-1`, and PLOS `.s001` DOIs
  are now components, like PLOS figure and table DOIs.
- BMJ articles from 2013 and 2014 (`10.1136/bmj.f1049`, `bmj.g2276`) and
  articles whose citation line names a supplement issue ("30 (Supplement 5)",
  "Volume 30, Supplement 5") were exported with no DOI, because their DOI was
  taken for a component's. A component suffix now needs PLOS's zero-padded
  number (`.g001`), and a supplement issue is not a component label.
- Preprints hosted on OSF (PsyArXiv `10.31234/osf.io/…`, SocArXiv, OSF
  Preprints) were exported with no DOI: "osf" inside the DOI marked it as a
  data deposit. Only the text around a DOI counts now. OSF project, Zenodo,
  Figshare and Dryad DOIs are still rejected by their registrant.
- A DOI that no label names the article's (a bare DOI or a doi.org link),
  printed outside the front matter and the running headers and footers, no
  longer becomes the paper's DOI unless it matches the manifest's expected DOI.
  Neither does a lone "Journal DOI". In a manuscript with no DOI of its own such
  a DOI was a cited work. When it is the only candidate, a manifest that
  requires a DOI now gets `VAL_EXPECTED_ID_MISSING`; before, the cited DOI was
  selected, and reported as `VAL_EXPECTED_ID_MISMATCH` when an expected DOI was
  given. The front matter is the title, abstract and keywords sections and
  pages 1 and 2. DOCX, ePub, HTML and JATS inputs have no pages, so for them it
  is the unclassified block before the first classified section (usually the
  Abstract), and a DOCX title page's doi.org link still names the paper.
- A DOI cited in a page-1 or page-2 footnote no longer replaces the paper's DOI
  when the running header repeats the paper's own. The conflict is reported and
  no DOI is selected.
- A JATS or HTML article's own DOI (its `article-id` or `citation_doi`) now
  wins over DOIs printed in its body text. eLife figure DOIs extend the article
  DOI with a number, so many eLife JATS and HTML files exported no DOI, and one
  a figure's.
- Wiley SICI DOIs (`10.1002/(SICI)1097-4679(199901)55:1<1::AID-JCLP1>3.0.CO;2-K`)
  were cut at the `<` when read as the paper's DOI or matched against a
  manifest's expected DOI. They are kept whole.
- Standard funding wording reached neither structured funding (`funding`, and
  so `funding_match`) in the default shadow integrity-statement mode nor
  `funding_statement` in active mode: "This project has received funding from
  the European Union's Horizon 2020 …", "The research leading to these results
  has received funding …", "We gratefully acknowledge funding from …",
  "Preparation of this article was supported by …" and "The first author was
  supported by …". Their subjects were checked as if they named authors, and
  failed. A named author who "has received funding" is now matched on the name.
- Active integrity-statement mode rejected or cut short standard declarations
  under a generic heading, and the default shadow mode raised
  `VAL_STATEMENT_SUSPECT` for each. "Available upon reasonable request to the
  corresponding author" was cut after "to the". "Data and analysis scripts are
  available at …", "The datasets can be obtained from the corresponding author
  …", "… will be made available by the authors", Frontiers' "conducted in the
  absence of any commercial or financial relationships …" conflict-of-interest
  statement and "Ethical approval was received …" were rejected. The licence
  the data are made available under ("… available … under a CC BY 4.0 license"
  or "under a Creative Commons Attribution 4.0 licence") and the date an
  approval was received ended the statement as if they were publisher
  boilerplate.
- A section whose heading merely contains the word "reference", such as
  "Revealed Preferences", "Reference standard" or "Reference values", is no
  longer taken for the bibliography when no other reference section is found.
  Its prose was parsed into references and the section was exported as
  `references`. The heading fallback now needs "references", "bibliography",
  "works cited", "literature cited" or "reference list" as whole words
  ("Selected References" and "Appendix B. References" still count), or a whole
  references heading in any language the reference-line capture recognises, so
  it also finds headings it used to miss, such as "Literaturverzeichnis" or
  "Daftar Pustaka".
- When a printed "References" heading overrides the sections the classifier
  had typed `references`, each of them now gets the type its heading looks up
  to ("General Discussion" becomes `discussion`, a heading with no known alias
  `unknown`). They kept `references`, so a bibliography entry could point its
  `text_id` at a body sentence, the citation linker read the section's numbers
  as reference numbers, and the export typed a body section as references. The
  reference receipt records `classifier_references_demoted`. A second list
  keeps its type when its heading names references or looks up to them, or
  when at least half of its rows open like a dated reference entry, as under
  "Studies Included in the Meta-Analysis".
- Reference strings from a JATS `<ref-list>` are parsed as given, one per
  `<ref>`. The filter for non-reference fragments dropped short entries without
  a year (a classic such as "Aristotle. Nicomachean Ethics.", or an entry whose
  year abuts its journal name), and the merged-reference splitter cut single
  entries in two, so every later entry shifted. HTML reference lists still go
  through both, because the HTML reader also collects other lists under a
  references heading, such as page navigation.
- The merged-reference splitter no longer cuts one reference in two at a
  citation inside its title when the title opens right after that reference's
  own date ("Brown, T. (2018). Beyond Kahneman and Tversky (1979): …"), or at
  an edition number that equals the next entry number in a numbered list
  ("1. Müller A. Lehrbuch. 2. Aufl. …", and likewise "udg.", "uppl.", "ed.",
  "wyd." and similar edition words).
- The geometry segmenter's segment-count check and the layout-region
  segmentation tier count only the reference onsets on the pages of the
  located reference section. Onsets from a second list elsewhere in the PDF,
  such as a transliterated copy of the bibliography or supplementary
  references, made the check decline a correct geometry result and the region
  tier decline as well, so the list went to the LLM segmenter.
- An aggregate `reference` layout box is dropped only when the entry boxes
  inside it hold all of its text. When the layout model returned entry boxes
  for only some of the entries in it, the entries without a box of their own
  were lost from the reference list. The aggregate box now stays and the entry
  boxes it repeats are hidden instead; they still count as layout onsets. The
  texts are compared with a tolerance for OCR noise in either direction, since
  a scanned page reads the aggregate box and each entry box separately and the
  reads differ by a character here and there. Text the entry boxes lack keeps
  the aggregate box however small a share of it that text is, such as one
  entry among twelve or more, a line or a DOI. A piece under 16 letters and
  digits long, such as a page range alone on the last line of an entry, still
  passes for OCR noise and is lost when the box is dropped or emptied. This
  applies both to the OCR stage's overlap cleanup and to the PDF parser. The
  OCR stage's second cleanup, which empties a `reference` region whose text
  the page's text regions already hold, uses the same comparison. It kept a
  region whose read ran a few characters longer than the text regions' reads,
  so its entries were emitted twice, and it emptied a region holding a line
  that no text region had, such as the end of a reference continued from the
  previous page, when that line was a small share of the region's text. The
  first is now emptied and the second kept.
- The page-furniture filter on the reference lines the geometry segmenter
  reads removes only lines at the top or bottom edge of a page, as it was
  documented to. It removed every line whose text, with digits masked, matched
  a line repeated at the edges of two pages, so reference text inside a page
  was dropped when the same short text also opened or closed two pages: a
  wrapped year or page range matching a page number, a wrapped "Polish)." line,
  or a reference label printed on its own line ("2." matching a "4." at the top
  of the next page).
- When implicit-section detection created an Abstract or Introduction, it
  moved every section without text of its own behind References: the root,
  the title once its text went to the new Abstract, a printed "Abstract"
  heading emptied the same way, and a numbered parent such as "2 Method"
  whose paragraphs sit in "2.1". The section sanity check reads list position
  as document position, so it then reset that printed Abstract heading to
  `unknown`, and reset References to `unknown` with score 0 whenever body
  sections now followed it in the first half of the list (reference location
  put the type back, not the score). Each new section is now inserted right
  after the section its text came from, usually the title, and no other
  section moves. Where a printed heading ("LITERATURE CITED") and the
  reference section bibr created for the rows both classify as references,
  the printed heading now comes first and keeps the type, as the tie-break by
  document order intends.
- The export placed a heading without text of its own right after the
  section with the next lower id. A "Method" heading whose subsections were
  not nested under it therefore came before the Abstract and Introduction
  that implicit-section detection cut from the title's text, since those are
  created last. It now follows the section listed before it.
- The section sanity check also counted the root and the figure, table and
  footnote sections added at the end of the list, so enough floats could
  place a terminal References section in the "first half" and reset it. Only
  body sections count now, and References is reset only when more body
  sections follow it than precede it, the threshold a paper without floats
  already had.
- The printed abstract span opened only on absolute page 1 and continued only
  onto page 2. A PDF processed with `--pages` or serve `start_page` keeps its
  absolute page numbers, and DOCX, HTML and ePub input has no pages, so none
  of them got a span: a DOCX with no model abstract exported an empty one, and
  a DOCX model abstract was flagged `VAL_ABSTRACT_SUSPECT` (`ungrounded`) even
  when it matched the printed abstract. The span now opens on the first page
  the parse saw, as the first-page abstract fallback already did, and skips
  the page test for input without pages. On the eLife HTML sample it now
  selects exactly the printed Abstract section in 841 of 984 articles, where it
  selected nothing.
- The lettered-appendix repair re-typed ordinary headings as top-level
  `appendix` sections. A run of "A. …", "B. …" headings in the last 40% of the
  section list qualified with no anchor at all, which caught the lettered
  subsections of IEEE-style papers ("IV. EXPERIMENTS", "A. Datasets", "C.
  Results") and of many regional journals ("Results and Discussion", "A. …",
  "B. …"), and a Roman "V. CONCLUSION", which reads as letter V. Any lettered
  heading after the first section typed references also qualified, so a
  Frontiers "Citation" panel above the title, or the navigation "References"
  at the top of an eLife HTML page, turned the title "A protocol for …" or
  subsections such as "A specific requirement for …" and "C. elegans strains"
  into appendices. A lettered run now needs a real anchor: an "Appendix"
  heading, which anchors only the headings after it, or the reference list
  that ends the body. That is the first section typed references after a
  body section, or the first one when no body section comes before any.
- A heading that is exactly the name of a part ("Materials and methods",
  "Experimental Section") folded under an earlier heading of the same type
  that only contains a keyword, such as the Results subsection "A neural
  implementation of oscillation" read as Methods. It became a subsection of
  that heading, and the subsections printed under it were attached to the
  part before it (Discussion, in eLife articles) and took that part's type.
  It now starts its own part and keeps its subsections. Exact names of
  subsections, such as "Study design" or "Limitations", still fold under a
  keyword part heading such as "Patients and methods" or "Discussion and
  conclusion". A part name printed as a subsection of such a heading, such as
  "Conclusions" inside "Discussion and conclusion", does start a part of its
  own. The earlier heading counts as a keyword hit only when the alias table
  typed it, which in runs with an LLM or the trained classifier happens only
  when the keyword covers most of the heading.
- On the default automatic OCR chain, a window covered entirely by native text
  stored the static fallback identity where later windows read the concrete
  runtime from. Those windows then ran a GLM engine with Paddle prompts, the
  Paddle profile and Paddle provenance, mangling HTML tables through OTSL
  decoding. The resolved identity now persists only after the engine starts,
  and a window that starts the engine adopts the runtime that actually
  started. `OcrStage` and the interleaved render/OCR stage share one identity
  state machine, and an engine-start failure fails only files still needing
  OCR instead of files already served from the OCR cache.
- Truncated Paddle tables are now retried once at a higher token budget on
  the Paddle HTTP transports (the local `paddle-*` clients and `bibr serve`)
  through one shared helper; previously only `bibr serve`
  retried. Only generations the provider cut short are retried
  (`finish_reason == "length"`, or structural truncation when no finish reason
  is reported) — a stop-terminated ragged or unterminated grid at temperature 0
  reproduces deterministically, as does a closed grid whose spans are
  malformed, so neither costs a second generation. The retry is skipped when
  the first request's table budget already meets or exceeds the recovery
  budget (for example under an `OCR_GENERATION_MAX_TOKENS` override at or
  above it), keeping the longer first output. The OCR cache key carries
  the recovery budget for the transports that run the retry, not just serve. Whitespace
  after OTSL continuation markers no longer destroys spans, stray text after
  a row terminator opens the next row, and blank table output decodes to empty
  content so the OCR success-rate gate still catches a silently degraded engine
  (blank tables are then reported only through OCR_TABLE_INCOMPLETE, no longer
  also as a parse-level OCR_TABLE_DROPPED).
- The test suite no longer depends on file order through global Settings,
  logging, network, or local-server-port state. Global Settings changes
  (including which fields count as user-set) and `bibr.*` logger levels
  are restored after every test; the section-classifier tests stub the LLM
  tier instead of calling the Gemini API; the doctor tests patch the
  helpers doctor actually calls; and the managed-server tests pass with
  bibr's default ports held. Opt-in `slow` tests under `tests/local/` skip
  those port-probe stubs and see the real server, so the vLLM integration
  test can poll a real `/health` endpoint. Tests that intentionally reach
  the network (live API and Hub downloads, including the slow extraction
  smoke test and the vLLM integration test) are marked `network` and opt
  out of the socket guard. Core CI now runs
  the default torch-free ONNX runtime (section/paper classifiers, NER parser,
  layout, and the HTTP-path no-torch check) against committed tiny bundles
  under `tests/fixtures/onnx`, regenerated by
  `scripts/generate_onnx_test_bundles.py`. The hermetic LitServe descriptor
  handoff test and the reference-geometry tests run in the default suite
  again, backed by committed synthetic fixtures
  (`scripts/generate_hermetic_test_fixtures.py`) instead of uncommitted
  corpus files.
- `bibr chew` writing JSON to stdout is now valid JSON under every console
  encoding. The startup stream setup replaced unencodable characters with
  Python escapes (`\U0001d465`), which no JSON parser accepts; the export is
  now written as UTF-8 bytes instead. File output already wrote UTF-8 and is
  unchanged.
- `bibr chew papers/ -o results/` with one paper in the directory no longer
  writes a file named `results`. A trailing slash or an existing directory in
  `-o` now means a directory even for a single resolved file, so the export
  lands as `results/<stem>.json` and a later run with the same `-o`
  writes into it again instead of crashing. A blocked `-o` is a clean exit 2
  before any model loads, and `-o` is resolved before the pipeline is
  constructed.
- The automatic-OCR dry-run line no longer reports the Paddle served-model
  alias as weights to download. For the default chain it cache-checks the
  weight repo the launcher loads, so a cached
  `PaddlePaddle/PaddleOCR-VL-1.6` renders `cached` instead of
  `will download (size unknown)`.
- A bad `--ocr-model`/`--ocr-profile` combination no longer blames `--pages`.
  Option errors from the run configuration print as `Invalid option: …` with
  exit 2 in both `chew` and `batch`.
- Per-file failure hints follow the pipeline's structured `error_code`
  instead of message substrings that never matched (or matched `rapid-mlx`
  for `api` and blamed API keys for local runtime failures).
- Building the CLI no longer needs installed package metadata. Running from
  a source tree via `PYTHONPATH` used to crash before argparse ran; the
  version now falls back to `?`, as the help screen already did.
- The batch report no longer shows a phantom `enrich_prefetch` stage share.
  That timer overlaps another stage's wall clock (the export stage already
  excludes it from `total_seconds`), but the report divided by a sum that
  included it, deflating the real stages. It now shares the export stage's
  exclusion list.
- Removed dead evaluation code with no in-repo callers: the unreferenced
  `keywords_fuzzy_f1`, `authors_count_ratio` and `authors_order_score`
  helpers, and the opposite-contract `validation_metrics.keywords_f1`
  duplicate (the harness's `evaluate.keywords_f1` is the one scored). The
  `title_soft_containment` docstring no longer promises a per-paper aggregate
  that was never emitted.
- Reference matching in the evaluator now runs once per paper instead of three
  times: one shared `match_references` pass feeds `ref_matching_f1`,
  `ref_field_scores` and `ref_field_counts`, and the gold-field predicates
  live in a single table that the per-pair loop gates on, instead of two
  copies that had to stay in lockstep. Scores are unchanged: re-scoring stored
  evaluation runs gives identical metrics, about three times faster.
- The abstract ROUGE-L length now comes from rapidfuzz's bit-parallel LCS
  instead of the pure-Python table — same value, roughly three orders of
  magnitude faster on long abstracts.
- Input validation no longer rejects readable files. A password-protected DOCX
  is reported as password-protected instead of corrupted: the CFB check
  searched for an ASCII `EncryptedPackage` stream name that real files store as
  UTF-16LE. A PDF that pypdfium2 opens is no longer rejected for bytes before
  `%PDF-` or data after `%%EOF`; the reader's open verdict is the corruption
  verdict, with the byte heuristics kept as a fallback. A PMC efetch download
  (`<pmc-articleset>` wrapping exactly one `<article>`) validates and parses;
  multi-article sets are still rejected, with a message naming the count. All
  3,927 PMC/eLife/PLOS corpus files validate exactly as before.
- One stale PDF bookmark no longer discards the whole outline; entries whose
  destination lies past the last page keep their title with no resolvable page.
  An ePub with one missing spine file now exports its readable chapters instead
  of failing, failing only when no spine member is readable, and each skipped
  chapter is recorded as an `EPUB_SPINE_MEMBER_SKIPPED` entry in
  `extraction.warnings` instead of vanishing silently. An over-cap or
  corrupt spine member still rejects the book. `dc:identifier` values in
  `doi:`, `urn:doi:` and `https://doi.org/` form are recognised as DOIs
  alongside bare `10.` strings.
- The export gate's `VAL_EMPTY_EQ` now fires on a blank `lhs` or a blank
  `rhs`, the shape a null equation side ships as, instead of only on an
  all-blank row the exporter cannot produce. A link dropped from the export as
  malformed is recorded as a `URL_MALFORMED_DROPPED` entry in
  `extraction.warnings` instead of vanishing silently. `VAL_DANGLING_REF` now
  covers `xref`/`url`/`eq` text ids, affiliation author ids, the three match
  tables and the `extraction` id lists, and a new `VAL_DUPLICATE_PK` flags
  repeated primary keys. The all-blank shape the old check fired on still
  fires; nothing previously flagged goes quiet.
- Impossible printed dates such as `31 April 2020` no longer export as
  `published_date: '2020-04-31'`; the value falls back to `YYYY-MM`. Valid
  dates, leap days included, are unchanged.
- A checkpointed (`-o`) export now consolidates like the unsinked export (so
  the `CONSOLIDATE_WITHOUT_ENRICHMENT` warning reaches the file, and the file
  is rewritten only when consolidation ran) and replays the enriched run's
  `extraction.timings`, so the `enrich` stage time is reported on both paths.
- Anthropic thinking budgets no longer produce requests the API rejects
  with a 400. With `LLM_THINKING_BUDGET` set, a task cap sends thinking only
  when it leaves at least 1024 tokens above the budget for the answer;
  smaller caps run without thinking at temperature 0. Budgets below the 1024
  minimum are raised to it and logged.
- `bibr mcp` chew tools report failure causes again: `chew_paper` and
  `chew_url` wrap non-`BibrError` failures in a `ToolError` carrying the
  exception type and a secret-scrubbed message with the file name only,
  never the full path or URL, instead of the detail-less "Error executing
  tool". `BibrError` messages are scrubbed the same way, and `chew_url`
  reports the downloaded file name rather than the URL.
- `bibr mcp` without cloud credentials prints the one-line missing-key
  message and exits 1 instead of dumping a traceback. Only the startup
  credential check is reported that way; a `ValueError` from the running
  server session keeps its traceback, and `bibr.chew()`/`Chewer` still raise
  the provider's `ValueError`.
- The offline batch docs no longer claim batch runs prefill the LLM
  response cache: nothing writes it, so `CACHE_LLM` serves live runs only.
- The Crossref bulk DOI prefetch sends no request and seeds nothing when
  both cache tiers are disabled, and leaves comma-bearing DOIs to their
  individual lookup instead of failing the whole chunk's filter.
- `CACHE_TTL_SECONDS=0` (or negative) now means "no expiry" instead of
  failing every Redis SET with "invalid expire time" and silently
  disabling the result cache.
- `dir(bibr)` lists the lazy public API (`chew`, `Chewer`, `Result`,
  `write_tables`, ...) without importing it, and `Result` answers the
  plural table aliases (`figures`, `tables`, `affiliations`, `urls`,
  `equations`).
- Constructing a pipeline with `FIG_EXTRACT=meta` logs the documented
  "not implemented" warning instead of passing silently.

### Added

- New `OCR_NATIVE_TEXT_HEADER_FOOTER` setting (default off): read header and
  footer regions from the PDF text layer instead of OCR on born-digital PDFs,
  under the same printable-ratio gate as body text. Default output is
  unchanged; enable it to compare.
- PP-DocLayoutV4 support, not yet the default. PaddlePaddle keeps
  `PaddlePaddle/PP-DocLayoutV4_safetensors` private until its release, so bibr
  still loads PP-DocLayoutV3; switching is a settings change behind an
  evaluation gate (see the Configuration guide, "Layout model generation").
  `scripts/export_onnx_layout.py` exports either generation, and the bundle
  manifest's `architecture` selects the pre- and post-processing of the ONNX
  runtime; the torch runtime loads V4 through `LAYOUT_MODEL_ID` once
  transformers ships it. V4 keeps V3's 25 region labels. bibr uses the
  rectangle enclosing each predicted quadrilateral and decodes V4's reading
  order (a successor graph made acyclic, sorted topologically, with
  relative-order votes breaking ties) in numpy, without scipy. It matches
  transformers' processor except where scores tie exactly or are NaN, which a
  trained head's output does not produce. A bundle or checkpoint whose label
  list differs from the one bibr maps, or a V4 bundle that declares none, is
  refused. Under the ONNX runtime a `LAYOUT_MODEL_ID` naming a different
  checkpoint than the bundle's source is logged as unused.
- `LAYOUT_MODEL_ID` names the torch layout checkpoint (default
  `PaddlePaddle/PP-DocLayoutV3_safetensors`). The serve image bakes it next to
  `LAYOUT_MODEL_REVISION`.
- Parquet corpus tables. `bibr tables <exports> --out DIR`, `bibr.write_tables()`
  and `bibr batch` (into `<out>/tables/` after every run; `--no-tables` skips
  it) write any number of exports as one Parquet file per table: `paper` (one
  row per paper), every record and match table, and `extraction_*` files for
  the processing lists. Rows start with `paper_id`; column types come from the
  schema, so every file has the same columns whatever papers it holds, with
  lists and nested records kept as Arrow lists and structs. `pyarrow` is now a
  core dependency.
- `CROSSREF_NOT_FOUND_TTL_SECONDS` (default 1 day, `0` disables): the Crossref
  response caches now remember a DOI lookup's 404 (no record, typically a
  malformed DOI or one registered elsewhere) for that long. Repeat lookups then
  skip the rate-limited request, and the DOI is left out of the bulk prefetch. A
  remembered 404 behaves exactly like a live one: the reference gets no Crossref
  match and no bibliographic search. With `CROSSREF_REDIS_CACHE` on it survives
  restarts, so re-running a batch no longer re-spends a request on every known
  missing DOI.
- Captured reference training records carry a `provenance` object with the
  label source, LLM provider and model, prompt name and hash, and bibr version.
- Evaluation artifacts now record a `bibr_dirty` flag (uncommitted tracked
  changes in the scoring checkout) and an `eval_code_sha256` digest over
  `evaluation/*.py`, so a score from a patched worktree no longer stamps
  the same provenance as unpatched code. No metric definition changed.
- `bibr.export.PaperExportReader`, a lenient reader model for any 12.x export,
  generated from the strict `PaperExport` models. `Result.model` is an instance
  of it when built from a dict, and it remains a `PaperExport` subclass.
  `docs/schema/bibr-export-v12-reader.schema.json` is its JSON Schema, published
  alongside the strict `bibr-export-v12.schema.json`.

### Changed

- `bibr chew --dry-run` now reports a Blockers section and exits 1 when the
  real run would fail immediately: missing inputs, missing LLM credentials
  (key lookup only, no client is built), an unstartable managed local LLM
  backend, and the PDF OCR/image runtime. `bibr batch --dry-run` runs the
  same local preflight (the PDF OCR/image runtime) the real run does and
  exits 1 with it. A clean preview still exits 0.
- Removed the shipped `bibr.metrics` package: the `PerformanceRecorder`
  (whose only consumer was never published, and whose per-request peaks were
  process-lifetime maxima) has no in-repo callers left, so `bibr.metrics`
  no longer imports.
- Enrichment looks up the paper's own DOI alongside the reference lookups
  instead of before them, so a DOI-bearing paper's references no longer wait
  one Crossref round-trip. If the self-DOI lookup fails, the reference lookups
  still finish before the enrichment is reported partial.
- On CPU-only machines the layout detector runs one page at a time unless
  `LAYOUT_BATCH_SIZE` is set explicitly. Batching pages on CPU only grows
  memory use without running faster.
- CPU sessions for the layout detector and the sentence segmenter no longer
  use the ONNX Runtime CPU arena, which otherwise holds its peak allocation
  for the life of the session. The smaller footprint costs about 10% more
  wall time on CPU.
- In aggressive memory mode the NER reference parser loads on CPU rather than
  the GPU and is released after post-parse. Balanced and keep-all modes keep
  the previous device choice (CUDA when available) and keep it loaded across
  files in one process; set `NER_DEVICE` to override.
- The equation fallback sends fewer methods/results sentences to the LLM:
  sentences whose digit-bearing parentheticals are only author-year citations,
  bare years, or figure, table, supplement, equation or section references are
  skipped — unless the surrounding prose carries digits of its own, in which
  case they are still sent, as is any sentence with statistic-like content.
  Measured over three public JATS corpora (3927 papers), about 18% of
  sentences with digit-bearing parentheticals are skipped, all
  citation/reference-only; no equation in the stored exports came from a
  skipped sentence.

### Security

- **A `bibr serve` without `AUTH_API_KEY` no longer takes orders from web pages.**
  Loopback was its only boundary, and a page open in the operator's browser could
  cross it: a cross-site form POST needs no CORS preflight, and a DNS-rebinding page
  reaches 127.0.0.1 under its own host name and reads the answers, `/mcp` included,
  whose rebinding protection was switched off. Without a key the server now answers
  only a `Host` of `127.0.0.1`, `localhost` or `[::1]` (`421` otherwise), refuses
  state-changing requests from another site's `Origin` or with
  `Sec-Fetch-Site: cross-site` (`403`; origins listed by name in `CORS_ORIGINS` are
  accepted, `*` admits none), and turns the MCP transport's rebinding check on with
  the same names. A keyless bind is accordingly limited to `127.0.0.1`, `::1` and
  `localhost`; another loopback address such as `127.0.0.2` needs a key. `bibr batch
  --serve-url`, MCP clients, curl and the operator's own browser on `/docs` are
  unaffected. With a key nothing changes; a proxy or tunnel in front of the server
  needs one.
- **Upstream errors no longer put the OCR or Crossref URL into exports and error
  bodies.** An httpx status error quotes the full request URL, user-info and query
  included, and region, page and file OCR failures and Crossref enrichment failures
  copied it into `extraction.warnings`, the `422` body and the job error. They now
  name the error and its status (`HTTPStatusError: HTTP 400 Bad Request`), other
  messages have URLs replaced by `<url>`, and the serve error body drops URLs too;
  the full text stays in the (scrubbed) log.
- **The log scrubber covers exceptions passed as arguments.** It rewrote only the
  `str` arguments of a record, so `logger.warning("...: %s", exc)` — the common
  form — logged `?key=…`, bearer tokens and URL passwords from the exception
  unmasked. It now masks every argument, keeping the argument tuple that formatters
  such as uvicorn's access-log formatter unpack, or freezes the masked message when
  that cannot be done argument by argument; and it masks a password-only URL
  (`redis://:password@redis:6379/0`) as well.
- **`bibr setup` writes a new `.env` readable by its owner only (0600)**, as
  `bibr config set` already did; it was created with the umask's mode (usually
  0644), readable by every account on a shared machine. An existing file is
  rewritten in place as before, keeping its mode, owner and links.
- **Bearer keys are not sent over plain HTTP to public hosts.** `bibr batch
  --serve-url` and an `LLM_BASE_URL` or `OCR_VISION_BASE_URL` receiving an API key
  had no scheme check. `http://` stays allowed for loopback and private-network
  hosts (LAN or tailnet addresses, single-label names,
  `.local`/`.internal`/`.lan`/`.ts.net`); a public host needs `https://`, or
  `--allow-insecure-http` for `bibr batch` and `LLM_ALLOW_INSECURE_HTTP=true` for the
  LLM key, which the vision endpoint also receives. The pipeline, `bibr setup` (model
  listing and connection test) and `bibr doctor` apply the same rule; the wizard asks
  again, or records the opt-in, instead of sending the key. `bibr batch` also warns
  when its token goes over plain `http://` to a LAN host, and its examples now use
  `https://`.
- **Script-capable links are dropped from exports.** `javascript:`, `vbscript:` and
  `data:` targets from HTML, JATS or DOCX links reached `url[].href`, `bib[].url`
  and the match rows' `url` and `license_url` (taken from the Crossref record as
  deposited), which readers render as anchors. Other schemes pass; none of the
  evaluation corpora carries a script-capable link.
- **`chew_url` in `bibr mcp` keeps a downloaded file inside its temporary
  directory on Windows.** A server-chosen name such as `D:evil.pdf` discarded the
  directory and wrote to drive D. The name is now reduced to one plain component:
  characters Windows forbids are replaced and device names such as `NUL` prefixed.
- **The serve response cache and the OCR cache are keyed on the full SHA-256.**
  The shared response cache used a 64-bit prefix, so two crafted files could share
  one cached result. It also ignored the file extension, which picks the parser:
  the same bytes uploaded as `.html` and then `.xml` got the HTML result back for
  the cache's lifetime. Both keys changed, so existing response and OCR cache
  entries are no longer read: the first request for each file extracts again, and
  old OCR cache files can be deleted.
- **Model bundles load through an allowlist.** The restricted joblib loader refused
  a list of dangerous modules, which any allowed module could hand back as an
  attribute, and it did not cover object-array payloads or joblib's pre-0.10
  format. It now resolves only the numpy, scikit-learn and joblib classes the
  front-role and geometry-segmenter bundles are built from, reads object arrays
  under the same allowlist, and refuses the old format. The shipped bundles load
  to identical models.
- **`REDIS_PASSWORD` no longer shows inside `REDIS_URL`.** `repr()`/`model_dump()`
  of the settings and `bibr config show` (and the batch ledger's settings snapshot)
  printed it in the URL they mask elsewhere; URL passwords are masked now.
- **`REDIS_PASSWORD` reaches the other Redis URLs on the same server.** It was
  added only to `REDIS_URL`, so a host-only `JOBS_REDIS_URL` or
  `CROSSREF_CACHE_REDIS_URL` on the compose Redis connected unauthenticated, and a
  user-only URL (`redis://default@redis:6379/0`) stayed without a password. A
  unix-socket URL gets it as before (`unix://:password@/path/redis.sock`), and so
  does a sibling URL on the same socket. A URL on a different server, or with its
  own password, is left alone.
- `ML_CLASSIFIERS_REQUIRED=true` is now enforced on the local pipeline, not
  just in serve. A required classifier that fails to load fails the file
  with code `classifier_required_failed` and a message naming the setting.
  The run used to continue silently without the classifiers; set
  `ML_CLASSIFIERS_REQUIRED=false` to allow the run to continue without them.
- `save_paper` writes only `.json` files and refuses to overwrite an
  existing file unless `overwrite=True` is passed explicitly.

### Removed

- Dead helpers with no callers anywhere (including tests):
  `snapshot_download_no_symlink`, `onnxruntime_available`,
  `ONNX_TOKENIZER`, `vllm_mlx_available`, `TAG_TO_IDX`, and
  `bibr.batch.runner._print`.

## [0.5.1] - 2026-09-12

### Fixed

- Cancelled OCR and local LLM startup reclaim servers that finish starting after
  cancellation. Shutdown waits for startup and completes resource cleanup even
  when interrupted repeatedly.
- Cancelled local sentence segmentation retains its inference lock until the
  worker finishes, preventing concurrent inference or premature model unloading.
- OCR cache hits preserve page-failure counts and warnings, and recheck the
  current minimum success ratio. Older entries without this evidence are rebuilt.
- Served requests can enable reference parsing and Crossref enrichment when the
  deployment defaults reference parsing to off.
- Layout initialization failures affect only PDFs in mixed batches; native
  documents continue through the pipeline, including streaming runs.
- Closing a pipeline releases resident layout and segmentation models and its
  cached front-matter classifier reference.

## [0.5.0] - 2026-09-11

### Fixed

- Release validation installs macOS's `libmagic` prerequisite and exercises
  Windows with platform-independent fixtures. Windows accepts sealed segmenter
  bundles, tolerates unavailable Unix memory metrics, and preserves upload
  identity and binary bytes. CLI output remains usable with legacy encodings,
  and disabling circuit-breaker deduplication counts failures even within one
  clock tick.
- Use patched vLLM 0.27.0 for the optional CUDA runtime and isolated LLM/OCR
  bootstraps, addressing GHSA-7m6h-x95x-82q5.
- Restore publication-date precision from an unambiguous printed publication
  date, and repair an empty author surname when the printed name and email
  establish a unique partition. Preserve ambiguous and already complete values.
- Preserve selected front-matter author evidence and distinguish explicit absent
  abstracts from inferred opening prose. Author-information tables supplement
  eligible single-record pages; generic literature-summary tables do not.
- Prefer a printed English abstract when parallel versions are available; otherwise
  retain the first complete printed version, without translating or concatenating.
- Stop requesting downstream author roles from metadata LLMs. Custom
  OpenAI-compatible endpoints can recover explicit decoder aborts through one
  validated JSON route, with token caps and chat-template options preserved.
- Structured-response caches distinguish generation schemas and chat-template
  options, and preserve explicit-null versus blank abstract intent.
- Recover a printed article DOI from complete retained OCR of a publisher box above
  its uniquely selected title. Explicit DOI, ISSN and publication labels establish
  identity; cited, ambiguous and truncated evidence remains excluded.

- LLM responses that echo a JSON Schema, including extracted values incorrectly
  nested under `properties`, now fail validation instead of being accepted as
  empty metadata with the schema name as the paper title.

- Numeric citations now follow reliable printed reference labels after dropped or spurious
  bibliography entries shift internal IDs. Citation diagnostics use the same corrected
  targets; duplicate and missing labels in that mapping cannot select a different entry.
- Reference-type inference reads the complete printed reference, recognizing thesis,
  preprint, conference and report labels outside the title when the parser omitted a type.
- GROBID benchmark runs accept both plain-text and JSON version responses without putting
  a JSON object into snapshot filenames or version columns.
- Dependabot CI keeps the coverage threshold and stores its report without attempting
  a Codecov upload that requires an unavailable Actions secret.

- **Release tests collect on Windows again.** The process-group signal guard
  only installs where `os.killpg` exists; Unix runtime tests supply their own
  mock on other platforms.
- **Cancelled REST and MCP extractions retain their admission slots.** The
  in-flight slot now belongs to the dispatch task until it finishes, so cancelling
  a caller cannot admit more work while its extraction is still running. Oversized
  MCP uploads are also rejected before allocating a decoded base64 copy.

- Async-job shutdown stops workers even if a dependency consumes cancellation while
  completing a request. Queued uploads are discarded instead of starting more work or
  waiting indefinitely for another job.
- **Protected docs smoke tests identify their HTTP client.** Both probes send
  `User-Agent: bibr-ci-smoke/1.0`, avoiding Cloudflare's error 1010 for Python's
  default agent. A browser-signature block is reported separately from an Access
  service-token rejection; anonymous protection and exact-revision checks remain enforced.
- **Docs deployments retain their revision marker.** The CI artifact now includes
  `.well-known/bibr-build`, so protected preview and production checks can verify
  the deployed commit after downloading the built site. Both jobs install Node/npm
  explicitly so Wrangler also runs on a freshly provisioned self-hosted runner.
- Redis-backed job workers preserve shutdown cancellation when a Redis reply arrives
  in the same event-loop turn. This prevents an intermittent Python 3.11 server shutdown
  hang while retaining the configured Redis operation timeout.
- Removed unsupported accuracy tables from the evaluation guide.
- The private-site CI smoke now distinguishes a rejected Cloudflare Access service token
  from a missing build marker, so deployment failures identify the required fix.

- **Exported URLs no longer carry PDF line-wrap artifacts.** A URL broken across a line in the
  source picked up the wrap whitespace when the text was re-joined, and a sentence-final period
  was absorbed into the href. `url[].href` and `bib[].url` are now collapsed and stripped of
  trailing dots at export, idempotently, so a downstream consumer can delete its own patch.
- **A JATS bibliography in `<body>` is no longer dropped.** EuropePMC's `fullTextXML`
  emits the reference list as a body `<sec sec-type="ref-list">` rather than inside
  `<back>`; the parser only looked in `<back>`, so every reference in such a document
  disappeared with no warning and the export shipped an empty `bib` — which is the whole
  contract for a reference-checking consumer. A body-located `<ref-list>` is now ingested
  into the section its producer already wrapped it in. A `<back>` ref-list still wins, so
  no document that parses correctly today changes.
- **JATS consortium authors survive.** A `<contrib>` carrying `<collab>` (a
  working-group or consortium byline) has no `<name>`, so it was emitted as an author row
  with an empty given *and* family name — a `VAL_AUTHOR_BLANK` validation error in place
  of the group's name. The collaboration name is now kept the way Crossref models a group
  author, and a `<contrib>` with no name of any kind is skipped instead of emitting a
  blank row.
- **JATS markup that means a line break no longer fuses words.** Flattening concatenated
  descendant text with nothing between the pieces, so `Cognitive load<break/>and recall`
  became `Cognitive loadand recall`, a structured `<aff>` became
  `Department of PsychologyUtrecht University`, and a two-paragraph abstract ran its
  sentences together. Block-level and structured-field elements now contribute a
  separator; inline markup still does not, so `H<sub>2</sub>O` stays `H2O`. XML comments
  are no longer flattened into the text either.
- **DOCX line breaks and tabs no longer fuse words together.** `<w:br/>`, `<w:tab/>` and
  `<w:cr/>` carry no text of their own and were dropped outright, so a title page laid out
  with Shift+Enter came out as `Cognitive load and recallJane SmithDepartment of
  Psychologyjane.smith@example.edu` — one unsplittable token where the title, author and
  affiliation should be. The same applied to footnote and endnote text, which is where a
  humanities bibliography lives. Those three elements now contribute a separator; adjacent
  `<w:t>` runs still concatenate untouched, because Word splits runs mid-word for
  formatting. Every DOCX fixture in the suite was built from python-docx plain strings,
  which never emit either element, so nothing caught this.
- **An unreadable DOCX is classified instead of crashing validation.** Validation only
  caught `BadZipFile`, but reading a *member* fails differently: `zipfile` raises
  `RuntimeError` for a password-protected entry and a truncated or damaged deflate stream
  surfaces as `zlib.error` from the real-size check. Both escaped
  `_check_docx_corruption` and took down the whole validation call rather than marking the
  file corrupt. A zip whose members are encrypted — what third-party tools produce, as
  opposed to the OLE container Word writes — is now reported as `encrypted_file` rather
  than as generic corruption.
- **The LLM rate limiter no longer freezes `bibr serve` while it probes Redis.** Deciding
  between the shared and the local limiter ran a *synchronous* `redis.Redis.ping()` from
  inside a coroutine. One LitServe worker with `enable_async=True` serves every concurrent
  request on a single event loop, so a Redis that accepts the connection but never answers
  stalled every in-flight paper, not just the caller — 5.1 s of total freeze, measured
  against a wedged-but-reachable Redis. The probe is async now, guarded by a
  loop-bound init lock so concurrent first callers build exactly one limiter, and the
  command round-trip is bounded (`socket_connect_timeout` only ever covered the connect).
  `CrossrefClient` was fixed this way already; `LLMClient` was missed.
- **A reference the LLM returns empty is re-parsed instead of deleted.** A batch item that
  came back with neither a title nor authors was counted as *covered*, so the NER recovery
  never ran for that slot — and then the completeness filter dropped it. One printed
  reference disappeared and every later `bib_id` shifted up by one, so an inline `[8]`
  resolved to what the paper printed as `[7]`, all the way down the list. Nothing warned:
  14 references returned for 15 entries clears the under-yield thresholds. Such a slot now
  counts as missing, goes through the same NER recovery as an entry the LLM skipped
  outright, and is reported if it cannot be recovered.
- **A repeated reference index no longer leaves segment-anchored backfills on.** When the
  LLM's reported indices are rejected the refs are re-numbered positionally, and the
  backfills that copy a printed DOI or issue number off the anchored segment are supposed
  to switch off whenever that mapping cannot be trusted. The check for that looked only at
  the *count*, so three rows labelled 1, 2, 2 for three entries passed it — while entry 3
  was missing and everything after the repeat sat one row off. The result was a
  neighbouring reference's DOI stamped onto the wrong row, which then enriched cleanly
  against Crossref and scored as a confident match. A repeated index now marks the batch
  untrusted too. A batch merely numbered from 1 instead of from `start_index` still stays
  trusted — positional re-indexing fixes that exactly.
- **The OCR disk cache now keys on the model pins.** A complete entry lets the pipeline
  skip layout detection and OCR inference outright, but the key recorded none of the
  settings that select those weights — `LAYOUT_MODEL_REVISION`, `OCR_PADDLE_REVISION` and
  `OCR_PADDLE_MODEL`. Re-pinning a model and re-running over cached papers silently
  replayed the *old* model's regions, so an A/B evaluation of the two pins reported no
  difference because it never ran the new one. `identity.model` did not cover this: for
  every served backend it is the alias (`paddle-ocr-vl-1.6`) that vLLM is launched with
  under `--served-model-name`, while `--revision` takes the pin — the alias is unchanged
  by a re-pin. The cache format version is bumped, so entries written without the pins are
  invalidated rather than trusted.
- **JATS keeps its Greek letters and accents.** The parser reads uploads with entity
  expansion disabled (the XXE and billion-laughs defense), which leaves every *named*
  character entity — `&alpha;`, `&uuml;`, `&deg;`, `&mdash;` — as an unresolved node whose
  text is the literal source string. Titles, author surnames and reference strings shipped
  markup like `M&uuml;ller` and `Effects of &alpha;-synuclein`. Because XML's five
  predefined entities and all numeric references resolve regardless, the output looked
  plausible rather than obviously broken. Named entities are now resolved after the parse
  against the HTML5 character table, which covers the ISO sets JATS DTDs pull in. Entities
  a document declares in its own internal DTD subset are still never expanded — the same
  applies to the ePub package document.
- **Non-ASCII ePub text is no longer mojibake.** The spine is re-emitted as one synthesized
  HTML document for the HTML parser, and that document declared no charset — so html5lib
  fell back to windows-1252 and decoded the UTF-8 bytes wrongly. Every non-ASCII character
  in an ePub's title, authors, publisher and body text was corrupted (`München` →
  `MÃ¼nchen`), affecting every non-English ePub.
- **Numbered bibliographies survive the in-text-citation filter.** Vancouver and IEEE entries
  terminate at the year exactly as a bare in-text cite does, so `"12. Rothman KJ. Modern
  epidemiology. Boston: Little, Brown; 1986."` was dropped as a citation — and because
  survivors are renumbered, one dropped entry shifted every later `bib_id` and repointed
  every numbered citation past the gap. Silent, with one `logger.info` line.
- **Multi-study `Method`/`Results` headings are no longer demoted as running headers.**
  Repetition alone was the test; page furniture's margin-band geometry is now required too,
  so a paper with per-study sections keeps them instead of exporting neither.
- **DOCX, JATS, HTML and ePub inputs no longer crash in the positional abstract fallback.**
  `min()` over page numbers that are all `None` raised `TypeError` for every native-format
  paper that reached it.
- **A crafted ePub can no longer exhaust server memory.** Every zip limit was per member, so
  a spine naming one member N times multiplied all of them: a 1 MB upload reached multi-GB
  RSS and OOM-killed the serve worker. Spine documents, total expanded bytes and repeats are
  bounded now, and percent-encoded hrefs resolve.
- **Statistics keep their sample size.** `(N = 1,204)` truncated to `N = 1` on the thousands
  separator, and a chi-square's own `df` parenthesis emitted a fabricated `N` that then
  vetoed the real match.
- **Figure and table numbers survive float merging.** Mergers renumbered survivors from 1, so
  a body mention of `Figure N` resolved to the wrong figure; renumbering now honours the
  printed label where a caption carries one.
- **Rotated pages map their text correctly.** `page.render()` applies `/Rotate` and the text
  layer does not, so on a rotated page every layout box sampled the wrong region of the PDF.
  Crop-relative coordinates are also emitted in the frame their page dimensions describe.
- **Plus 25 further defects** — a caption-dedup `KeyError` that surfaced as `parse_failed`
  and dropped the paper, an equation-extraction timeout that discarded the regex results it
  had already computed, OTSL row/column spans destroyed by a trailing newline, math exponents
  linked as citations and deleted from the sentence, an OCR backend that could never start,
  an OCR engine orphaned when an earlier stage failed, and a repeat scan whose cost grew
  superlinearly with region length (~11 s on a 50,000-character region, now under 15 ms).
- **`bibr serve` no longer crashes at startup when the `mcp` extra is installed.** LitServe
  0.2.17 enables its own MCP connector whenever the official `mcp` package is importable
  but builds it from the third-party `fastmcp` package, so `server.run()` died with
  `NameError: name 'MCPServer' is not defined` on any install of `bibr[mcp]` (including the
  serve image above). bibr now switches LitServe's detection off — it mounts its own
  `/mcp` endpoint and never wanted LitServe's.
- **The scorer no longer charges an elided page range against its expansion.** Gold keeps
  the printed ending ("486–92"); bibr and GROBID expand it to "492", and the exact
  string compare counted every such pair as a pages miss on both sides. `ref_pages_acc`
  now expands a compact ending against the first page on both sides before comparing.

- **Half-emitted page ranges are completed from the printed reference.** The CRF parser
  drops the start of a range and the LLM parser the end of a compact one ("339-42"); the
  shared finalize step now fills the missing end anchored on the value the parser did
  emit, only when the segment prints exactly one such range, and splits a range lumped
  into one field. Nothing populated is overwritten. This is the LLM-path repair the July
  analysis projected to lift `ref_pages_acc` from 0.389 to 0.745; measured by unit tests
  so far.
- **A plain Linux `bibr chew` no longer bootstraps vLLM behind your back.** The automatic
  `paddle` OCR chain only lists `paddle-vllm` on an NVIDIA GPU with at least 8 GB of VRAM;
  CPU-only and small-GPU Linux machines go straight to llama.cpp (`glm-llama`), as the
  tester guide always said. The managed vLLM launcher refuses to start without a suitable
  GPU (naming the alternatives), and when vLLM is not installed it now *warns* — with the
  `uv sync --extra vllm` remedy — before falling back to the isolated
  `uv tool run --from vllm==0.26.0` environment, which downloads several GB on first use.
  On Python 3.14, where `vllm==0.26.0` has no wheels, the bootstrap pins a managed 3.13
  interpreter instead of failing to resolve. `bibr chew` also checks before loading any
  model that at least one local OCR runtime can start for the PDFs it was given, and
  fails fast with the install hints otherwise.
- **`bibr setup` installs the runtime its Linux plan needs.** The "fully local" plan on
  Linux/CUDA selects the `vllm` extra; it used to select `local`, whose only member is
  Apple-Silicon-only, so nothing was installed and the first chew paid for the bootstrap
  above. The advanced picker no longer offers `local-cuda`, an extra that does not exist
  and made `uv sync` fail before `.env` was written.
- **`glm-mlx` is no longer offered anywhere.** The backend has been disabled since July
  (vllm-mlx produced corrupted OCR text and leaked memory), but the wizard, the `--ocr`
  choices, `bibr doctor` and the quickstart still presented it. `bibr doctor` now reports
  a config that still names it as a failed check pointing at `glm-rapid-mlx`.
- **`OCR_BASE_URL` may end in `/v1`.** bibr appends `/v1/models` and
  `/v1/chat/completions` itself, so the deployment guide's own example
  (`https://ocr.example.internal/v1`) was requested as `/v1/v1/...` and never became
  ready. A trailing `/v1` is now stripped with a warning in `bibr serve`, `bibr chew
  --ocr-url` and the readiness probe. `bibr serve` also honours
  `OCR_BACKEND=paddle-http` for its defaults (served alias `paddle-ocr-vl-1.6`, profile
  `paddle`) instead of silently assuming the GLM `glm-ocr` alias; the guide is corrected.
- **A stalled Redis can no longer wedge `bibr serve`.** The response cache, the Crossref
  response cache and the Redis rate limiter now carry connect, socket and health-check
  timeouts, and every cache touch on the request path (read, single-flight lease, release,
  write) is additionally bounded by the new `CACHE_OPERATION_TIMEOUT_SECONDS` (default
  5). A Redis that accepts connections but never answers now degrades to a cache miss
  instead of holding every request — and its admission slot — forever.
- **A bearer token containing non-ASCII bytes is rejected with 401, not 500.** The
  auth middleware and `/ready` compared the header as text, and `hmac.compare_digest`
  raises on non-ASCII strings; the comparison now runs on UTF-8 bytes.
- **502 bodies no longer name the internal OCR endpoint or the models it serves.** The
  readiness errors raised by `bibr serve`'s OCR backend (unreachable, not ready,
  cooling down) included `OCR_BASE_URL` and the server's model list; those details now
  go to the operator log only, and the client sees the expected served alias at most.
- **Idle MCP sessions now expire.** The serve MCP endpoint closes a client session after
  `MCP_SESSION_IDLE_TIMEOUT_SECONDS` (default 1800) of inactivity and drops its papers,
  as the MCP guide already promised. A client that reconnected without `DELETE` used to
  pin its session — and up to sixteen full exports — for the process lifetime.
- **Docker Compose publishes the API on loopback and passes the Redis password safely.**
  `bibr-serve` was published as `0.0.0.0:8000`, which Docker routes past host firewalls,
  with plaintext bearer tokens on the wire; it is now `127.0.0.1:8000` unless
  `BIBR_PUBLISH_HOST` says otherwise. `REDIS_PASSWORD` is no longer interpolated raw into
  `REDIS_URL` — a password with URL metacharacters silently disabled the cache — but
  handed to bibr, which URL-encodes it into the connection URL itself.
- **`LLM_LOCAL_MODEL` no longer defaults to the MLX weights on every platform.** The
  config default was `numind/NuExtract3-mlx-8bits`, and the CUDA vLLM and llama.cpp
  launchers read it too, so a hand-written `.env` with `LLM_BACKEND=vllm` or `llama-cpp`
  downloaded 4.8 GB of MLX weights and failed to load. Unset now resolves per backend
  (bf16 for vLLM, GGUF Q4_K_M for llama.cpp, 8-bit MLX on Apple Silicon); `bibr setup`
  keeps writing an explicit value.
- **`bibr setup` and `--llm local` no longer pick vLLM for a GPU that cannot hold the
  model.** The rule was "vLLM above 8 GB", but NuExtract 3's only vLLM variant (bf16)
  needs 11 GB, and the fit filter was dropped silently when nothing fit — a 9-10 GB
  card got a plan that OOMed after OCR. Both now choose vLLM only when a vLLM variant of
  the recommended model fits and llama.cpp otherwise; the wizard says which half (OCR,
  LLM, or both) needs `llama-server`.
- **The managed local LLM bootstrap matches the OCR one.** When vLLM is not installed the
  LLM launcher now warns (naming `uv sync --extra vllm`) before its `uv tool run`
  bootstrap and pins a managed Python 3.13 on 3.14, where `vllm==0.25.1` has no wheels
  and the `vllm` extra installs nothing; it used to fail there after OCR with "LLM
  server start failed" while `bibr doctor` reported vLLM as available. `bibr doctor`
  now says the runner is uv-managed, that the first run downloads several GB, and
  what 3.14 implies. Python 3.14 is listed in the package classifiers, matching CI.
- **`bibr.chew()` and `bibr.Chewer()` check the LLM before loading any model.** The
  library ran layout and OCR before discovering a missing API key or an unlaunchable
  local backend; the CLI already checked first. Both entry points now run the same
  preflight (skipped with `no_llm=True`), raising the provider's `ValueError` for
  credentials and `ConfigurationError` for a managed local backend.
- **A trained classifier that does not answer is now visible in the export.** When the
  section or paper classifier is configured but cannot load (core install without
  torch, failed download, a degraded serve resource) or errors during inference, the LLM
  classifies instead; the JSON was indistinguishable from a healthy run. The section path
  now records `section_classifier_degraded` in `processing_warnings` for every such
  case (previously only inference errors), and the paper path records
  `Metadata extraction WARNING: paper classifier degraded (<reason>)` — the exception
  type only, never document text.


- **Job results are bounded by size, not only by count.** `bibr serve` kept every completed
  result as a live dict and evicted only beyond `JOBS_MAX_RETAINED` (128), so a run of large
  exports could hold hundreds of megabytes for an hour. The result is now rendered once at
  completion (the bytes `/result` serves) and the store evicts oldest-first until both the
  count and the new `JOBS_MAX_RETAINED_BYTES` budget (default 256 MiB; `0` disables) fit;
  the newest result is always kept, so an export larger than the budget can still be
  fetched once.
- **The MCP chew tools are under upload admission, and a 50 MiB file fits.** The admission
  middleware only knew `/papers/extract` and `/papers/jobs`, so any number of `/mcp` calls
  could hold their bodies and decoded bytes in API memory and dispatch straight to the
  worker; and because `chew_paper` carries its file base64-encoded, LitServe's 51 MiB body
  cap refused a 40 MB PDF before the advertised 50 MB check. A large `/mcp` body now holds
  a `PIPELINE_MAX_ACTIVE_UPLOADS` slot while it is received, both chew tools take a spool
  slot through the persist and an inflight slot for the extraction itself — matching
  `POST /papers/extract`, so a running pipeline no longer refuses uploads the server has
  capacity to accept (a `server busy` tool error when none is free), the outer body cap grows to
  fit a full-size file in base64 when MCP is enabled, and an oversize body gets a `413` that
  explains the arithmetic before a byte is read.
- **`bibr serve` logs are configured — in both processes.** The CLI returned before its own
  logging setup, so `bibr.*` INFO records were dropped, warnings fell through
  `logging.lastResort` unformatted and unscrubbed, metering emitted nothing without
  `METER_LOG_PATH`, and the spawned inference worker never installed the metering handler,
  losing every per-extraction record with LLM token usage. Each process now installs one
  formatted, secret-scrubbed stderr sink (`SERVE_LOG_LEVEL`, default `info`), uvicorn and
  LitServe records ride it, and metering goes to stderr or, when configured, only to the
  JSONL file — from the worker too.
- **A born-digital window no longer starts an OCR engine it will not use.** OcrStage waited
  for the engine before counting the regions that would call it, and the automatic `paddle`
  chain was started before layout to key an OCR cache that is off by default. The count
  now comes first and, when native text covers every region, no engine is started or
  awaited; with `CACHE_OCR` off the automatic chain starts after native text is known.
  Captions, table titles and formula numbers still go through OCR by design.

- `bibr doctor` now reports a missing system **libmagic** as its own named check, and
  `bibr.input.validate` imports the `python-magic` binding defensively instead of at
  module scope. libmagic is a system library a `pip install` cannot supply, so a fresh
  macOS/Linux setup died with a bare "failed to find libmagic" during `bibr setup`'s test
  extraction — and the import failure took down `import bibr` wholesale, so `doctor` could
  not run to diagnose it. The error now names the platform's install command
  (`brew install libmagic`, `apt install libmagic1`, `dnf install file-libs`). (#64)
- A managed local server whose port is held by an unrelated process now fails with a
  message naming the port, instead of spawning a subprocess that cannot bind it and dies
  with an unrelated-looking startup crash. The pre-spawn guard treated "listener with an
  unusable /v1/models" the same as "port free"; it now confirms the port is genuinely
  held with a TCP connect before reporting a conflict. (#82)

- LLM retries now acquire their own rate-limit slot. Only the first attempt of each logical
  call took one, so a retry storm spent budget it never acquired — precisely when the
  provider was already rate-limiting and, with Redis configured, when the shared limiter is
  meant to hold the whole fleet back.
- A missing `CROSSREF_API_EMAIL` now warns with the concrete rates: without it Crossref's
  anonymous pool caps the client at 60 RPM, so a configured `CROSSREF_RATE_LIMIT_RPM=200`
  was silently a third of that (~81s of an 80-reference paper's 120s enrichment budget).

### Changed

- Release preparation supports a manual rehearsal on `main` that tests and validates
  the distributions without publishing. PyPI uploads use Trusted Publishing on a
  GitHub-hosted runner and stay disabled until `PUBLISH_PYPI=true` is explicitly set.
- Shorten the README, keep the illustrated banner, and link to detailed guides.
  Add a draft LLM-use disclosure and clarify extraction accuracy limits and the
  current focus on English-language social science papers.
- The README opens with a paper-cream banner with square corners and no outer border.

- **The launch export uses schema 11.0 (breaking).** `schema_version` is at the root;
  `info` becomes `metadata`, `info_match` becomes `metadata_match`, and input file identity
  moves to `source`. Root `affiliations` becomes `affiliation`. Telemetry moves under
  `extraction`: engines, settings, timings (`stages` and `total_seconds`), usage (`totals`
  and per-label/provider/model `breakdown`), enrichment, diagnostics, identity receipts,
  warnings, and optional regions/trace. Validation findings live in `validation.issues`.
  `xref[].xref_id` becomes `target_id`; equations and funding gain explicit IDs;
  table cell contents are string grids. Match-table structured `authors`/`editors` become
  singular `author`/`editor`. The v10 output mode is retired; core checkpoints and enrichment
  sidecars reject older schema versions. The evaluation tools still read frozen v10 gold
  alongside v11 predictions without changing the scoring rules.
- **MCP Python SDK v2**, locked to 2.2.0. The server uses `MCPServer` and the public HTTP
  lifespan/idle-timeout API. Paper tools run on the event loop and keep stores isolated
  across initialized clients, including clients sharing a bearer key. HTTP clients negotiate
  the session-based 2025-11-25 protocol, which the chew/query workflow requires; automatic
  v2 clients fall back from sessionless discovery. Upload limits account for base64 overhead.

- Evaluation, aspect scoring and the benchmark harness now share `metrics_version=6`.
  The benchmark headline author score uses full printed names; family-name-only scores
  remain available as diagnostics. Re-score older benchmark records before comparing them.

- The LLM rate-limit slot is now acquired once inside `_invoke_structured`, below the cache
  check, instead of separately at each of the twelve call sites. A cache hit spends no
  provider quota, so it no longer waits on the budget that exists to protect that quota —
  previously a fully-cached corpus re-run was still paced at `LLM_RATE_LIMIT_RPM`. Live
  calls are unaffected: still one slot per dispatched request, plus one per retry.
- **The install extras are reorganised around that runtime.** `onnxruntime`, `tokenizers`,
  `huggingface-hub`, `scikit-learn` and `joblib` move into the core dependencies, so a
  plain `pip install bibr` runs the whole HTTP-service path — OCR and the LLM over HTTP,
  every bibr-owned model through ONNX Runtime — with no `torch`, `transformers` or OpenCV
  in the environment. The PyTorch stack is now the **`torch`** extra (training parity,
  Apple MPS, `torch.compile` on the serve layout model, transformers OCR, the CRF
  reference segmenter, and the fallback runtime); **`ml` is kept as an alias for it**, so
  existing installs, Dockerfiles and `bibr setup` plans are unaffected. `all` now bundles
  `batch,cache,demo,mcp,torch`. The two OpenCV calls in `bibr/ocr/image_processing.py` are
  Pillow/numpy.

- **The CLI stops treating a missing `torch` as a broken install.** `bibr doctor` reports
  the ONNX Runtime execution provider as the device instead of failing, and calls
  `seg=geom, parse=ner` healthy on a core install; only `REF_SEG_STRATEGY=crf` (torch-only,
  no ONNX export) and an explicit `ML_RUNTIME=torch` without torch still fail. `bibr chew`
  no longer refuses PDFs when OpenCV is absent — cv2 is reachable only through the torch
  layout path — and `--dry-run` names the ONNX provider it would use.

- **Enrichment's network wait overlaps the extract stage.** When enrichment is on, the
  enrich stage's up-front round-trips (resolver health probe and title searches, the
  Crossref bulk DOI lookup) start as soon as the references are parsed — while citation
  linking and structured-integrity LLM calls are still running — instead of strictly after
  extraction. `enrich_references` consumes the
  `EnrichmentPrefetch` when the pipeline hands it one and is unchanged otherwise; the core
  checkpoint still sees unenriched references, the enrichment stage's accounting is
  unchanged, and every path that does not enrich cancels the task. `extraction.timings`
  gains `enrich_prefetch` (its wall time; excluded from `total_seconds`).
- **Crossref reference enrichment is opt-in.** `CROSSREF_ENRICH` now defaults to `false`:
  a plain `bibr chew`, `bibr.chew()` or `POST /papers/extract` no longer calls Crossref or
  the resolver, `bib_match` stays empty, and `extraction.crossref_enrich` reports the
  effective per-run value. Enrichment was a network fan-out that added seconds of serial
  wall time per paper for every caller, including those that never read `bib_match`.
  Deployments that relied on the old default must set `CROSSREF_ENRICH=true` (or pass the
  per-run switch above); `bibr setup` now asks before writing it, and only offers
  consolidation once enrichment is on.

- **`bibr serve` keeps CPU-bound work off the shared event loop.** One LitServe worker runs
  with `enable_async=True`, so synchronous CPU inside a coroutine is head-of-line blocking
  for every co-resident request. Post-parse, citation linking and OCR post-processing now
  offload to a thread like their neighbouring stages (40.0 ms → 5.2 ms loop-tick latency for
  this class of work), four exact necessary-condition prefilters remove ~52 ms/paper of
  regex sweeps outright, and OCR crops moved inside the region semaphore (`Image.crop` is an
  eager copy; every crop of every page was held at once, ~1 GB at 8 in-flight requests).
- **The serve container image ships the `mcp` extra.** `Dockerfile.serve` now installs
  `bibr[mcp]`, so `MCP_ENABLED=true` on the Compose stack mounts the remote MCP endpoint
  without a custom build. The dependency is inert unless enabled.
- **CI runs for `main` only.** The retired February `dev` branch no longer triggers the
  suite on push, and pull requests can no longer target it.
- Removed unsupported comparative accuracy claims from the public documentation.
- **Every Hub-loaded model is pinned to a commit.** PP-DocLayoutV3, the section and paper
  classifiers and the default `sat-6l-sm` sentence segmenter loaded `main`, so a hub
  push could change extraction output between two runs of the same bibr version. Their
  audited commits are now the defaults (`LAYOUT_MODEL_REVISION`,
  `ML_SECTION_CLASSIFIER_REVISION`, `ML_PAPER_CLASSIFIER_REVISION`,
  `WTPSPLIT_MODEL_REVISION`; set any to `main` to track the head), `Dockerfile.serve`
  bakes the same revisions, and `scripts/prefetch_segmenter.py` accepts `--revision`.
- **The managed vLLM pin moves to 0.26.0** (`vllm` extra and the `uv tool run` bootstrap).
  It closes GHSA-87x5-vmc3-756j (completion prompt lists fanning out into unbounded engine
  requests) and drops `diskcache`, whose unfixed advisory bibr had been carrying as an audit
  exception; torch stays at 2.11.0. The lock resolves cleanly and the extra installs;
  serving with 0.26.0 has not yet been exercised on a GPU.
- **`LIMITATIONS.md` is current again** (native-format inputs, the `ner` default, the
  classifier-degraded warnings, single-tenant serve/MCP, and which benchmark numbers are
  held-out), and the local model registry's sizes were re-verified against the Hub.

- Removed ignored/no-op config names: `LLM_VLLM_MLX_CACHE_MB`, `LLM_BATCH_PROVIDER`;
  `OCR_API_HOST`, `OCR_API_PORT`, `OCR_CONFIG_PATH`, `OCR_ENABLE_LAYOUT`, `OCR_API_PATH`,
  `OCR_API_MODE`; and the reserved `ML_ENABLED`, `ML_SECTION`, `ML_REF_SEG`, `ML_REF_PARSE`,
  `ML_SECTION_ACCEPT_THRESHOLD`, `ML_SECTION_FLAG_THRESHOLD`, `ML_SECTION_REPO_ID`,
  `ML_REF_SEG_REPO_ID`, and `ML_REF_PARSE_REPO_ID`. These names are ignored if left in
  existing config and should be removed. External deployment/Compose and bundled-SGLang
  scope are unchanged.
- OCR disk cache format 7 removes the retired layout-toggle key; existing format-6 entries
  incur a one-time cache miss and rebuild.
- `OCR_MAX_CONCURRENT_REGIONS` now binds the single-machine OCR path, which previously
  ignored it and capped every run at `OCR_CONCURRENT_REGIONS_PER_FILE` (6). Against the
  managed `paddle-vllm` runtime — the Linux/CUDA default — that left its vLLM server
  (launched with `--max-num-seqs 12`) under-subscribed; it now runs at the server-wide cap
  (16 by default). Files still run one at a time, so page-image RAM is unchanged. Engines
  whose prefill serializes on the device (MLX, llama.cpp) keep the per-file cap, and the
  Apple Silicon auto-tune to 1 is unaffected.
- Crossref enrichment now prefetches every DOI-bearing reference in one
  `/works?filter=doi:...` query (up to 50 DOIs per request) before the per-reference
  fan-out, instead of spending one rate-limited request per DOI. The prefetch seeds the
  same response cache the per-reference path reads, so matching, consolidation and
  provenance are unchanged; only DOIs the bulk query returns are seeded, so a DOI Crossref
  does not know still takes its own lookup and still 404s rather than falling through to a
  bibliographic search. Disable with `CROSSREF_BULK_DOI_LOOKUP=false`.
- **Removed `PIPELINE_WORKERS_PER_DEVICE`.** `bibr serve` now pins exactly one inference
  worker in `build_server()`. There is no measured configuration where a second worker won:
  each worker gets its own `GpuBatcher` (so GPU batches shrink as workers rise) on top of
  duplicating the model weights and CUDA context, costs ~1.5 GB RSS (~570 MB of that in
  imports alone, before any model loads), and parallelizes only GIL-bound Python — the heavy
  CPU stages already use every core from one process, and the process-global pdfium lock it
  would have relieved is under 1% of a paper's wall clock (~10 ms/page render plus a
  comparable inspection pass). Left in an existing config the name is ignored, not rejected,
  but it should be deleted. Scale with `PIPELINE_MAX_INFLIGHT_REQUESTS` and the batch-timeout
  settings instead. `cap_inference_threads`, which existed only to divide cores among
  co-located workers, is removed with it; torch now uses its own default thread count, which
  on a hyperthreaded host is typically physical rather than logical cores.
- A managed local LLM server (`--llm local` on CUDA or Apple Silicon) auto-raises
  `LLM_RATE_LIMIT_RPM`, unless set explicitly. The 60 default guards a cloud provider's
  quota; against a server bibr owns it capped bulk runs near 8-12 papers/min regardless of
  hardware.

### Security

- Remove the unused Accelerate dependency from the PyTorch extras and lockfile,
  eliminating CVE-2026-69112 from supported bibr installations. Existing environments
  need a locked sync or rebuild to remove the previously installed package.
- **A configuration error no longer prints your API keys.** `ConfigurationError` rendered
  pydantic's `input` payload; for a model-level validation failure that payload is the whole
  merged settings mapping, so one bad value printed every key in the environment to stderr
  and into any log collecting it. Model-level errors now omit the input, and a secret-named
  field's value is masked wherever it appears.
- **`MCP_URL_ALLOWED_HOSTS` accepts the form the docs give.** As a bare `list[str]`,
  pydantic-settings JSON-decoded it, so `MCP_URL_ALLOWED_HOSTS=arxiv.org,zenodo.org` failed
  startup outright — in practice no deployment had the `chew_url` SSRF allowlist on. The
  comma-separated and JSON forms both parse now, here and for the CORS lists.
- **No credential literals in the tree, and CI now scans for them.** Six tracked scripts
  and a notebook carried a metacheck platform API key as a string; they read it from the
  environment now (`PLATFORM_API_KEY`, `METACHECK_PLATFORM_API_KEY` for the `data/`
  scripts). A required gitleaks job scans the checked-out tree and the commits every pull
  request introduces, alongside Semgrep's tree-only secrets pack; `.gitleaks.toml` holds
  the allowlist of documented placeholders and test fixtures. The same scan runs as a
  pre-commit hook over the staged diff.

### Added

- **Structured reference names alongside the verbatim strings.**
  `bib[].authors` and `bib[].editors` stay exactly as printed; new `bib[].author` and
  `bib[].editor` carry a best-effort split into `{family, given, suffix}`, or a `{literal}`
  fallback for corporate and unsplittable names, and are `null` when there was nothing to
  split (never `[]`). Every emitted value is a substring of the verbatim string, so a consumer
  can always fall back to it. `author[]` gains an optional `suffix`. Included in schema 11.0.
- **A machine-readable JSON Schema of the export** is committed at
  `docs/schema/bibr-export-v11.schema.json`, generated from the pydantic export models by
  `scripts/generate_schema.py`. A test fails when the file drifts from the models, and its
  `required` list is derived from the exporter's own omit rules (`OMITTABLE_ROOT_KEYS`), so the
  artifact can never call an always-present table optional.
- **Opt-in LLM response cache** (`CACHE_LLM=true`, directory `CACHE_LLM_DIR`, default
  `$XDG_CACHE_HOME/bibr/llm`). Structured responses are cached on disk keyed by model,
  response schema, system prompt, user text, per-task `max_tokens`/`reasoning_effort`, and
  transport mode — so an entry can only serve a request that would have produced it. A hit
  costs no tokens; a miss, a stale entry, or an unwritable cache directory all fall through
  to a live call, so nothing about correctness depends on it. Re-running a corpus after a
  parser change (or an evaluation sweep over the same papers under different non-LLM
  settings) now pays for its LLM work once instead of every time. Off by default, like the
  OCR disk cache. Note the key canonicalises the per-call `uuid4` prompt-injection fence
  boundary, which 10 of the 13 call sites mint fresh each call — without that the same
  logical request would hash differently on every run and never hit.
- **`bib[]` carries the five reference fields the parser tagged and the decoder threw
  away (export schema 10.8).** The NER parser's 39-tag BIO scheme has covered `ARXIV`,
  `PMID`, `SERIES`, `ACCESS_DATE` and `NOTE` since v4, but `map_fields_to_paper_ref` had
  no target for any of them, so every predicted value was discarded at decode — `PMID`
  reaches 0.947 F1 on the JATS-supervised corpus and reached nothing else. They are now
  `PaperReference` fields (`arxiv`, `pmid`, `series`, `access_date`, `note`), exported
  verbatim as printed, and a test asserts no field type can be tagged and silently
  dropped again. Output from the shipped `bibr-parser-v4-5-gold` is unchanged in
  substance — its training corpus had no examples of any of the five, so it emits none —
  and the fields are explicit nulls. The LLM reference schema is deliberately *not*
  widened: the fields are removed from the JSON schema both LLM paths read, because the
  NuExtract template is qualified against a fixed shape and the LFM2.5 student was
  distilled on prompts embedding this exact schema.
- **A torch-free core: bibr's four local models now run on ONNX Runtime.** The layout
  detector, the section and paper classifiers and the ModernBERT+CRF reference parser each
  ship an `onnx/` bundle (graph, a `bibr_onnx.json` contract carrying preprocessing
  constants, label classes and CRF parameters, and the exact tokenizer) alongside the
  PyTorch weights at the same pinned revision. `ML_RUNTIME=auto|onnx|torch` chooses:
  `auto` prefers the ONNX bundle, falls back to PyTorch when the bundle is absent and
  `torch` is importable, and otherwise raises a `ConfigurationError` naming the model and
  the fix. `scripts/export_onnx_*.py` rebuild the bundles and check parity against the
  PyTorch classes; `bibr/ner/crf_numpy.py` is a numpy Viterbi decoder so the parser needs
  no `pytorch-crf`, and `bibr/utils/onnx_tokenizer.py` tokenizes through `tokenizers`
  alone. Layout's PyTorch weights live in a third-party repo, so its ONNX artifact has its
  own `LAYOUT_ONNX_MODEL_ID` / `LAYOUT_ONNX_REVISION`, published as
  `scienceverse/bibr-layout-onnx`. All four bundles are on the Hub and pinned, so a core
  install — 1.0 MB wheel, 677 MB venv, no `torch`, `transformers` or OpenCV — downloads
  them on first use with nothing to configure.

- **`JOBS_STORE=redis` shares async-job state between bibr-serve replicas.** Job status,
  results (zlib-compressed, under their own key) and the active-job cap move into Redis,
  so several `bibr serve` instances behind a load balancer answer status/result polls for
  each other's jobs, and `JOBS_MAX_ACTIVE` / `JOBS_MAX_RETAINED` /
  `JOBS_MAX_RETAINED_BYTES` bound the whole deployment. Uploads and execution stay on the
  replica that received the upload, and every job status now reports that `replica`.
  Admission is one Lua script (no cap race between replicas); every Redis call is bounded
  by the `REDIS_*_TIMEOUT_SECONDS` budgets; an unreachable store answers
  `503 {"detail": "job store unavailable"}` on the job routes and `jobs_store: error` on
  `/ready`; a replica lost mid-job frees its cap slots after a 24 h safety TTL. New
  settings: `JOBS_STORE`, `JOBS_REDIS_URL` (falls back to `REDIS_URL`), `JOBS_KEY_PREFIX`,
  `JOBS_REPLICA_ID`. The in-process store is unchanged and remains the default
  (`bibr.serve.jobs.JobStore` is now the protocol; the class is `MemoryJobStore`).
  Handing queued work to another replica (a shared queue) is documented as a follow-up.
- **Front-role classifier for front matter.** `bibr/extract/front_role.py` loads a small
  gradient-boosted bundle (`ML_FRONT_ROLE_MODEL_ID`, defaulting to the published
  `scienceverse/bibr-front-role-v1` at a pinned revision) that scores every OCR
  region as title / byline / affiliation / abstract / keywords / doi_line / masthead /
  heading / ref_header / body / other from page-relative geometry, relative font size and
  script-independent text shape. Front-matter ownership uses the scores as additive
  evidence (a model byline survives the English byline shape and the 45-word cap, a model
  title seeds non-Latin records, a confident masthead cannot root a record) and
  `RefLocator` accepts a model `ref_header` heading in any language. A title seed the model
  confidently types as something else keeps its title role and loses only the right to root a
  *second* record (`ML_FRONT_ROLE_RECORD_ROOT_CONFIDENCE`, default `0.9`) — boxed headers
  like `Correspondence` and `A R T I C L E I N F O` score `heading` at 1.00 and otherwise cut
  a page's real title away from its own abstract. The model is trained from publisher JATS projected onto cached OCR regions;
  see `docs/guides/classifiers.md`.

- **`bibr batch` — a first-class, resumable corpus runner.** Takes manifests (one path per
  line, `#` comments), directories (recursive) or files, writes `<out>/<paper_id>.json` per
  paper and an append-only `<out>/outcomes.jsonl` ledger — one line per attempt with
  status, error code and stage, timings, per-stage times, LLM tokens, reference and match
  counts, warning frequencies, bibr version and build sha. Re-running the same command
  resumes (`ok` skipped, `failed` skipped unless `--retry-failed`, `--force` for all;
  interrupted papers run again by default); `--limit`, `--shuffle`/`--seed` and
  `--deadline` shape a leg. Locally it feeds one warm pipeline in `--batch-size` chunks
  with every `bibr chew` option; with `--serve-url` it drives a `bibr serve` job API with
  adaptive concurrency (429 drops in-flight to `--min-concurrency`, 5xx/connection errors/
  upstream outages retry with backoff, successes grow back toward `--max-concurrency`) and
  a graceful Ctrl-C. `bibr batch report <out>` (or `--json`) summarises a ledger: ok/failed,
  throughput, latency percentiles, stage-time shares, tokens, match rate, failure and
  warning breakdowns; every run ends with the same table. `run_info.json` records the
  options, the serve build and a secret-redacted settings snapshot. `bibr chew` gains
  `--include-regions` as an alias of `--regions`. Guide: `docs/guides/batch.md`.
- **A per-run switch for reference enrichment.** `bibr chew --crossref` (mutually
  exclusive with `--no-crossref`), `bibr mcp --crossref`, `bibr.chew(..., crossref=True|False)`,
  the `crossref=true|false` multipart field on `POST /papers/extract`, and the `crossref`
  knob on the serve MCP `chew_paper`/`chew_url` tools all force enrichment on or off for
  that run, overriding `CROSSREF_ENRICH` either way. `RunConfig.crossref` is tri-state
  (`None` follows the setting) and resolves through `RunConfig.enrichment_enabled(settings)`;
  the serve response cache keys on the effective value, so an enriched and an unenriched
  result for the same file never collide. `bibr chew --dry-run` names why enrichment is
  off and how to turn it on.

- **`BIBR_DISABLE_DOTENV=1`** makes every settings model ignore `./.env` and `~/.bibr/.env`
  (the process environment still applies). `python -m benchmarks run --tool bibr` refuses
  to start while either file exists unless it is set, so a run's recorded configuration is
  the profile plus the environment and nothing a developer's `.env` slipped in.
- **`bibr.local-default` benchmark profile** (`geom` segmentation + `ner` parsing, what a
  fresh `bibr setup` runs) next to the LLM-parse `bibr.default`, so the install default
  can be promoted as its own row.

- **`bibr mcp` — MCP server for agents** (new optional `mcp` extra, included in `all`).
  Exposes extraction as Model Context Protocol tools over stdio: `chew_paper` /
  `load_paper` register a paper and return a compact `bibr inspect`-style summary, then
  `get_metadata`, `get_sections`, `get_text`, `search_text`, `get_references`,
  `get_reference_citations`, `get_tables`, `get_figures`, and `save_paper` query the
  stored export in slices sized for an agent's context. One warm pipeline serves the
  whole session (models load once), extraction progress streams as MCP progress
  notifications, and pipeline options are fixed at server start via a subset of the
  `bibr chew` flags. Register with e.g. `claude mcp add bibr -- uv run bibr mcp`; see
  the new [MCP server guide](https://bibr.org/guides/mcp/).
- `Chewer.chew` / `achew` (and `chew_file` / `achew_file`) accept a `progress=` tracker
  (`bibr.pipeline.progress.ProgressTracker`, e.g. `RichProgress`) to observe stage
  transitions and per-region OCR progress from library code.
- **Remote MCP on `bibr serve`** (`MCP_ENABLED=true`, requires the `mcp` extra): mounts a
  streamable-HTTP Model Context Protocol endpoint at `/mcp` with the same chew-then-query
  tool surface as `bibr mcp`. Gated by the existing bearer auth; extraction rides the
  regular serve inference dispatch (resident worker models, admission control, size caps —
  no second pipeline). `chew_paper` takes base64 file content plus per-call
  `start_page`/`end_page`/`refs`/`consolidate` options; the filesystem tools
  (`load_paper`/`save_paper`) are not exposed remotely, and papers are held per MCP
  session, capped by `MCP_MAX_PAPERS_PER_SESSION` (default 16). See the
  [MCP server guide](https://bibr.org/guides/mcp/).
- **`chew_url` MCP tool** on both servers: extract a paper straight from a public
  `https://` URL. The download is SSRF-guarded by the new `bibr.utils.safe_fetch`
  (HTTPS/443 only, every DNS answer must be public unicast, the connection is pinned to
  the validated IP with TLS SNI/verification kept on the hostname to defeat DNS
  rebinding, redirects re-validated per hop, size-capped under a deadline). Capped at
  100MB on `bibr mcp`; on `bibr serve` it uses the upload size limit and rides the same
  inference dispatch, with `MCP_URL_ALLOWED_HOSTS` to pin hosts and
  `MCP_CHEW_URL_ENABLED=false` to remove the tool.

## [0.4.0] - 2026-07-26

Consolidates roughly five weeks of work since 0.3.0: a new default OCR engine
(PaddleOCR-VL), mature local-LLM runtimes across CUDA / Apple Silicon / Windows,
native JATS/HTML/ePub input, a rebuilt reference pipeline, trained
section/paper-type classifiers on by default, a much richer extraction schema
(v10.7), resolver-based enrichment, a config-preset system, a full CLI/UX
overhaul, and a security-hardening pass.

### Added

**OCR**
- **PaddleOCR-VL is the new default OCR engine** (`OCR_BACKEND=paddle`), with PP-DocLayoutV3 layout detection. Backends: `paddle` (default), `paddle-vllm` (GPU/vLLM), `paddle-rapid-mlx` / `paddle-mlx-vlm` (Apple Silicon), and `paddle-http` (external). Includes OTSL table decoding, formula canonicalization, and incomplete-table recovery.
- GLM-OCR retained as an alternative family: `glm-mlx` / `glm-rapid-mlx` (Apple Silicon), `glm-llama` (Windows default; llama.cpp), `glm-http` (external).
- **Cloud vision-LLM OCR** backends — `gemini`, `openai`, `anthropic` (via Instructor).
- **Native PDF text bypass** (on by default) — regions backed by a good PDF text layer skip OCR, with a printable-ratio corruption gate that falls back to OCR.

**Local LLM runtime**
- **`--llm local`** auto-resolves a managed, self-hosted OpenAI-compatible server: managed **vLLM** on CUDA, **vllm-mlx** (continuous batching) on Apple Silicon, **`--llm rapid-mlx`**, and **`--llm llama-cpp`** for Windows / low-VRAM (6 GB+) GPUs. NuExtract3 is the default local extraction model, chosen from a curated hardware-detected model registry.

**Input formats**
- **Native JATS XML, HTML/`.htm`, and ePub ingestion** — parsed natively; skip OCR and core LLM extraction, like DOCX.

**References**
- **Reference segmentation rebuilt** around a local **geometry GBM (`geom`, now the default, ~free)** with a confidence-gated LLM-anchor cascade; CRF, region, and pure-LLM strategies remain selectable. Segmentation and parsing are decoupled via `REF_SEG_STRATEGY` / `REF_PARSE_STRATEGY` (CLI `--refs` / `--ref-seg`).
- **Local NER reference parser is the default** (`--refs ner`); `--refs llm` gives full-precision batched LLM parsing; **`--refs off`** skips reference extraction entirely.
- Deterministic **merged-reference splitter** (on by default), leading-reference salvage from truncated LLM batches, Vancouver year/container backfill, and under-extraction warnings (vs in-text citation count).

**Trained classifiers (on by default, LLM fallback only on low confidence)**
- **Context-aware section classifier (v3/v4)**, loaded from HF Hub, with a positional sanity pass.
- **Paper-type** and **OECD research-domain** classifiers.

**Extraction & schema (v10.7)**
- **Research-integrity mining** — data/code-availability & ethics statements, structured funding, and CRediT author-contribution roles.
- **Structured affiliations** and **paper self-identity** (journal, volume, issue, pages, ISSN, publisher, date, license, self-DOI match).
- Figure/table **captions**, an extraction **provenance** block, **per-label LLM token usage** (`llm_usage_by_label`), non-fatal **`processing_warnings`**, and an **output validation gate**.

**Enrichment**
- **Optional resolver-first enrichment** via **bibr-resolver** (`BIBR_RESOLVER_*`) — one `sources` query spanning OpenAlex + Crossref, short-circuiting on a clean resolver miss — layered on the default Crossref enrichment. Two-tier Crossref cache (in-process LRU → shared Redis). Optional **consolidation** of accepted matches into `bib` (`CROSSREF_CONSOLIDATE=off|fill|replace`, `--consolidate`). Title-less (Nature/Science-style) reference matching by fingerprint.

**Config presets**
- **`bibr preset`** subcommands + a **`--preset`** flag and `PresetManager` for named JSON config profiles; `~/.bibr/.env` fallback when the CWD has none.

**CLI / setup**
- **Unified terminal design system** across all commands; **`bibr config`** (show/path/set/example, always-redacted), **`bibr inspect`** for extraction results, **`chew --dry-run`** resolution preview, and **`--no-llm`** structural-only mode.
- **Setup wizard redesigned** — hardware-detected plan preview before installing, local-LLM onboarding, save-as-preset, and an end-to-end smoke extraction on a shipped synthetic sample.

**Serve**
- **Async job API** with per-request usage metering, **GPU micro-batching** for safe single-worker concurrency, **gzip** responses (~8× on paper JSON), bearer-token auth gating all non-probe routes, per-request `refs`/`ref_seg` overrides, and an opt-in OCR disk cache.

**Library API**
- **`bibr.chew()` / `achew()`** (single file, directory, or list) returning a `Result` with `.df` / `.records` views and per-file `ChewFailure`; a **`Chewer`** warm-pipeline session; a typed paper-export model; and isolated `Settings` on the library APIs.

### Changed
- **Default OCR engine switched from GLM-OCR to PaddleOCR-VL** (see Added).
- `OCR_SGLANG_GPUS` renamed to `OCR_LOCAL_GPUS` (old name still accepted as an alias). The unused `OCR_LOCAL_MEM_FRACTION` setting was removed — it only configured the in-process SGLang engine.
- Recommended cloud LLM updated to **Gemini 3.5 Flash-Lite**.
- Reference-parse batch size default raised **5 → 15**; layout-detection batch **4 → 8**.
- Apple Silicon throughput defaults unlocked (higher default MPS concurrency/batch).
- Section classification is now trained-model-first (LLM only on miss / low confidence), using document-context snippets.
- Core install stays torch-free; heavy ML deps remain behind the `ml` extra.
- OCR cache hardened (correct model-profile keys, safer concurrent writes).

### Fixed
- **Metadata regressions:** single-article title/byline is no longer blanked by front-matter multi-item abstention; native NuExtract author extraction no longer returns schema-valid empty author lists when byline/CRediT evidence exists; deterministic LLM invalid-output is no longer misclassified as a retryable upstream failure.
- **Compound figures:** panels are grouped as parts of their parent figure instead of exploding into independent top-level figures and sections.
- **Extraction quality:** OCR NUL/surrogate scrubbing before tokenization, mangled section-header repair, masthead/internal-heading title rejection, DOI line-wrap bridging with self-DOI selection over funder/reference/footnote candidates, full-name author scoring, footnote/xref positional anchoring, and filtering of parenthetical-numeric equation false positives (author-year veto + equation-tag guard).
- **References:** never drop the leading reference; drop bare in-text citations that leaked into `ref_text`; remove running-header bleed.
- **Windows:** OCR cache and section-classifier download handling, symlink-failure fallbacks, and `llama.cpp` PATH discovery; low-VRAM llama.cpp path hardened.
- **Security hardening** (audit 2026-07-23): redact secrets from `Settings` repr / `model_dump`, CLI & serve logs, and `bibr doctor`; exclude API keys from cache fingerprints; enforce real-byte zip caps and reject spoofed file types; reject path-traversal DOIs before resolver/Crossref lookup; cap HTML input and upload-filename length; gate `/ready` detail; guard wildcard-CORS credentials; rotate the metering log; gadget-restricted joblib load for the geom segmenter.

### Performance
- Serve concurrency reworked around a single async worker (event loop unblocked, pipeline reused, GPU work micro-batched); `workers_per_device` default 2.
- Inference offloaded off the event loop (NER/GBM parse, native-text pdfium work, `gc.collect`); CUDA TF32/cuDNN autotuner and MPS float16 autocast for layout; per-page pdfium locks plus next-file render prefetch enable parallel file processing.
- Crossref/resolver caching (in-process LRU + Redis tier-2, `select=` field trimming, higher enrich concurrency when the resolver is enabled).

### Removed
- **SGLang removed entirely** — both the managed SGLang _LLM_ backend and the in-process `glm-sglang` _OCR_ backend, along with the `sglang[all]` dependency and the now-empty `local-cuda` extra. The pinned 0.5.12 line carried three unpatched critical advisories (unauthenticated RCE, pickle deserialization on a `0.0.0.0` socket, path traversal) and transitively pulled `diffusers` (two high advisories). Local LLM serving is vLLM / vllm-mlx / rapid-mlx / llama.cpp; GPU OCR is `paddle-vllm`. **Migration:** run your own SGLang server and point `glm-http` at it (`OCR_BACKEND=glm-http`, `OCR_BASE_URL=...`) — the bundled Compose `bibr-ocr` service still does exactly this.
- **Falcon OCR backend.**
- Dropping `sglang[all]` shed ~82 locked packages, removed the last mutually exclusive extras (so **`--all-extras` resolves again**), and made the `pillow` override unnecessary (it existed only for `moviepy`, a transitive SGLang dep).
- Dead code and unused dependencies — `spacy`, `rpy2`, the `metacheck` extra, legacy CRF model files, and transitional flat-name config shims.

## [0.3.0] - 2026-06-15

First tagged release of the rebuilt pipeline. The intermediate `0.2.0` tag was never published, so its notes are folded in here.

### Added
- **One-call Python API** — `bibr.chew()` / `bibr.achew()` process a single file, a directory, or a list of paths in one call, returning a `Result` with `.df` / `.records` views and `.ok` / `ChewFailure` per-file error handling. `Chewer` is a warm-pipeline session context manager. `from bibr import LocalPipeline, Pipeline, Settings` remains the lower-level entry point (lazy-loaded; no heavy deps at import time).
- **Reference segmentation rebuilt around LLM anchor-emit** — references are segmented by an LLM anchor pass (CRF fallback) then parsed in batches, replacing the retired rule splitter. Strategies are decoupled and configurable via `REF_SEG_STRATEGY` / `REF_PARSE_STRATEGY`; NER parsing is opt-in (`--refs ner`). The pipeline warns in `processing_warnings` on CRF seg-fallback and on suspected reference under-extraction (vs in-text citation count).
- **In-text citation (xref) linking + evaluation** — improved narrative and parenthetical citation parsing, plus automated checks for citation-linking behavior.
- **URL extraction (`url[]`)** — printed DOI links and web URLs are extracted, including reconstruction of line-wrapped URLs (CRLF and mid-word wraps) and support for balanced-paren DOIs.
- **Local LLM serving** — managed SGLang LLM server with `--llm local` auto-resolution; `LLM_MAX_CONCURRENCY` gate for single-device servers; `LLM_VLLM_MLX_EXTRA_ARGS` passthrough; opt-in merged core-metadata call (`LLM_MERGED_CORE_METADATA`); schema-envelope unwrapping for small-model structured output.
- **Crossref consolidation** — optionally merge accepted Crossref matches into `bib` at export via `CROSSREF_CONSOLIDATE=off|fill|replace`, the `--consolidate` CLI flag, the `consolidate=` chew option, and a serve form field.
- **Scoped hierarchy (v5-lite)** for correct section nesting in multi-study papers.
- **Lead-reference recovery** from the PDF text layer for references the layout model drops.
- **Per-paper LLM token-usage export** (`llm_usage`).
- **MiniLM section classifier (v2)** on by default, with LLM fallback only on miss or low confidence.
- **`bibr_release`** stamped on every serve response `info`.
- **Saved-export evaluation** — scoring helpers compare extracted fields with independently prepared reference JSON.
- **CLI setup polish** — LLM credential preflight, ref-strategy knob in the setup wizard and doctor, `--refs` surfaced in help and `bibr demo`.
- **JSON v10 schema** — top-level shape change (`figure` replaces `fig`, drops `study`, adds `bib_match`). v10.1 moves `ocr_config` and `processing_warnings` to top-level so `info` is scalar-only (R consumers can `as.data.frame(info)`). Adds `ocr_config` block, `BibAuthorExport` author records in `bib_match`, section `level` field, and backfill of empty bib fields from high-confidence external matches.
- **`include_regions` toggle** (default off) for the large `_regions` layout debug payload — CLI `--regions`, the `include_regions` form field, `Paper.export_to_json(include_regions=...)`, `RunConfig.include_regions`. Reduces output size when diagnostics are not requested.

### Changed
- **Core install is now torch-free** — heavy ML dependencies moved to an optional `ml` extra, ML imports degrade gracefully, and OCR exports are lazy.
- **LLM client migrated from LangChain to Instructor**, with multi-provider support (Google, OpenAI, Anthropic, Groq, Ollama) through a single Instructor factory.
- **CLI flag renames**: `--ocr-backend` → `--ocr`, `--llm-backend` → `--llm`. All CLI unified under the `bibr` namespace.
- **Settings restructured** into sub-models: read via `Settings.ocr.backend`, `Settings.llm.provider`, etc. Env vars stay flat (`OCR_BACKEND`, `LLM_PROVIDER`).
- Reference parse batch size default raised 5 → 15.

### Fixed
- Extraction and structure: OCR wide-letter-spacing collapse before segmentation, repeated running-header demotion, mid-word DOI line-wrap bridging, DOI rescue from publisher `/doi/` URLs and clean `doi:` tokens, page-1 `TC` badge-glyph stripping, fabricated-abstract suppression on abstract-less commentaries, software/dataset title and book-edition handling, bracket-citation retention.
- xref parsing: nested group-cites, narrative colon-page and curly-apostrophe possessive cites, year-less back-references, et-al disambiguation, harvested-year constraints, parenthetical cap recovery.
- Robustness and security: pdfium lock in validation, DOCX zip-bomb ceilings, pdfium handle cleanup, CUDA gating by compute capability, per-file pipeline errors on resource-init failure, `401` responses carrying `WWW-Authenticate` + CORS, CVE-driven torch bump.
- CLI: batch output directories with dotted names are no longer misread as file suffixes.

### Performance
- Crossref works/search LRU cache; native-text pdfium work moved off the event loop; next-file page render prefetched during layout detection; serve `workers_per_device` default raised to 2.

### Removed
- **SSE streaming endpoint** `POST /papers/extract/stream`. The synchronous `POST /papers/extract` is the only paper-extraction endpoint.
- LibreOffice-based DOCX conversion path. `.docx` is parsed natively via `python-docx`; `.doc` (legacy Word) is no longer supported — convert to `.docx` first. Drops `DOCX_BACKEND`, `LIBREOFFICE_TIMEOUT_SECONDS`, the `WITH_OFFICE` build arg, and `bibr.clients.libreoffice.LibreOfficeClient`.
- `bibr-serve`, `bibr-setup`, `bibr-demo` console scripts (replaced by `bibr serve` / `setup` / `demo` subcommands).
- LangChain dependency; legacy rule reference segmenter (moved to `evaluation/`).
- Empty `bibr/_vendor/` package and orphan top-level `ocr/Dockerfile`.
- `debug_samples/`, `metacheck_integration/`, `prereg.json`, and tracked `notebooks/incest.json` artifact data.

## [0.1.3] - 2026-03

### Added
- **`bibr chew` CLI**: process PDF/DOCX files directly without an external OCR server or serve deployment. Includes in-process OCR via SGLang (NVIDIA CUDA + Apple Silicon MPS), sequential GPU model loading with configurable memory management (`aggressive`, `balanced`, `keep_all`), and support for external OCR servers via `--ocr-url`
- `[local]` optional extra: `uv sync --extra=local` installs SGLang for in-process OCR
- `BibType` enum with standard BibTeX entry types (article, book, inproceedings, incollection, etc.)
- `booktitle` field on `PaperReference` for book chapters and proceedings papers
- LLM extraction of 6 new reference fields: `last_page`, `issue`, `publisher`, `editor`, `booktitle`, `bibtype`
- Non-destructive Crossref backfill: enrichment now populates empty fields (DOI, volume, issue, pages, publisher, ISBN, ISSN, booktitle, bibtype)
- `BibTypeEnum` in Pydantic schemas for validated LLM bibtype output
- API key authentication middleware (`X-API-Key` header, backward compatible)
- In-memory per-IP rate limiting middleware with configurable window and request count
- Trivy vulnerability scanning for Docker images in CI (table + SARIF upload)
- SSE streaming endpoint (`POST /papers/extract/stream`) for real-time pipeline progress
- Study design classification: LLM-based (RCT, Retrospective Cohort, Case Report, Meta-Analysis, In Vitro)

### Changed
- **JSON is now the primary (and only) export format** -- JSON v8.0 schema with top-level keys: `paper_id`, `info`, `author`, `text`, `section`, `url`, `bib`, `xref`, `fig`, `table`, `eq`
- `bibtype` values normalized from custom capitalized strings (e.g. "Article", "BookChapter") to standard lowercase BibTeX types (e.g. "article", "incollection")
- Crossref `container-title` now routed to `booktitle` for book chapters and proceedings articles
- Demo reference tables now display "Book Title" column
- CORS origins automatically restricted from `["*"]` to `[]` in production mode (`ENVIRONMENT=production`)
- Health endpoints (`/health`, `/ready`) exempt from authentication and rate limiting
- Section classification switched from zero-shot NLI (bart-large-mnli) to lookup table + LLM fallback

### Removed
- Arrow IPC export format (v6.2 and earlier) -- replaced entirely by JSON v8.0

## [0.1.2] - 2026-02

### Added
- Evaluation harness with 9 per-field metrics (exact match, ROUGE-L, Jaccard, etc.)
- Multi-class paper type classifier (empirical, review, meta-analysis, case-study, commentary, unknown)
- OECD domain classifier using cascading zero-shot NLI (L1 + L2 taxonomy)
- Configurable reference deduplication thresholds (`DEDUP_TITLE_THRESHOLD`, `DEDUP_MIN_TITLE_LENGTH`)
- Ground truth loading from Parquet for evaluation

### Changed
- Paper type classification upgraded from binary stub to priority-ordered rule-based system
- Reference dedup thresholds now configurable via Settings (previously hardcoded)

## [0.1.1] - 2026-02

### Fixed
- OCR region label routing now uses `native_label` for correct treatment dispatch
- Markdown prefix stripping in section headers (prevents `#` leaking into classified text)
- Footnote crash on `content=None` regions
- Missing `layout_hints` attribute on OCR regions

### Added
- OCR artifact correction (ligature expansion, soft hyphen removal) at page processing level
- Reference section text preserved intact for downstream CRF/LLM segmenter
- Non-destructive IMRaD enforcement (repeatable section types preserved)
- Title fallback from first non-canonical heading when layout model and LLM both fail
- Graceful degradation on LLM failures (partial `PaperMetadata` returned instead of crash)

### Removed
- Dead AST-era code (Tier 1 citation linking, unused imports, stale type definitions)

## [0.1.0] - 2026-01

### Added
- Initial release
- PDF and DOCX input support (DOCX via LibreOffice conversion)
- OCR via glmocr SDK with Ollama, vLLM, and SGLang backends
- LLM-based metadata extraction (title, authors, DOI, keywords, references)
- Section classification using zero-shot NLI (facebook/bart-large-mnli)
- Sentence segmentation via wtpsplit (ONNX)
- Inline citation NER (DistilBERT) with citation linking
- Reference extraction (LLM and NER strategies)
- Optional Crossref reference enrichment
- Arrow IPC export (v5.5 schema) with manifest
- FastAPI REST API with Redis caching
- Gradio demo application
- CLI (`bibr-serve`, `bibr-setup`, `bibr-demo`)
- Docker deployment with GPU-accelerated OCR sidecar
