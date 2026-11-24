# Benchmark results: bibr 0.9.2 vs GROBID

bibr reads a scientific PDF and returns its title, authors, affiliations,
abstract, reference list, in-text citations and links as structured data. By
default it runs on your own machine with open models. This page compares bibr
0.9.2 with [GROBID](https://github.com/kermitt2/grobid) 0.9.1 on 300 recent
open-access papers, scored against the publishers' own XML.

!!! note "Not trained on these papers, but not a blind test either"
    All 300 papers were published between 2 July and 9 October 2026. The
    newest paper in bibr's training data is from 1 July 2026, and a leak check
    against every training source, by DOI, PMCID and title, found none of
    them. bibr's first run on this set, in October 2026, was blind. Since then
    we have fixed problems the set exposed, and the set has been public since
    17 October 2026, so the scores below are not held-out results. The blind
    first run is in the [public benchmark](https://github.com/scienceverse/bibr-jats-eval).

## How many papers come out fully right

Give bibr 10 recent papers and it gets about 9 completely right. GROBID gets
about 6.

| Papers | bibr default | bibr `--fast` (CPU, 200 DPI) | GROBID 0.9.1 |
| --- | --- | --- | --- |
| Behavioural science, 200 born-digital PDFs | **94.0%** | 90.5% | 61.5% |
| Other fields, 100 born-digital PDFs | 89.0% | **91.0%** | 62.0% |
| Simulated scans, 90 PDFs | **92.2%** | not supported | 0% (no OCR) |

**What "fully right" means.** For each paper we check four things against the
publisher's own record of that paper. The paper counts as fully right only if
all four are right at the same time:

1. **Title:** matches the printed title, ignoring capitals and punctuation.
2. **DOI:** exactly right.
3. **Authors:** the authors' full names score at least 0.9 on F1, a measure
   that counts both missed names and extra, wrong ones.
4. **References:** the reference list scores at least 0.8 on F1. Roughly, at
   least 80% of the references are found and matched to the right entry,
   without many made-up extras.

If any one of the four misses, the whole paper counts as wrong, and a paper a
tool returns nothing for also counts as wrong. So this is a strict,
all-or-nothing score.

On born-digital PDFs, bibr's lead is +32.5 points in behavioural science
(paired 95% CI +25.5 to +39.5) and +27.0 in other fields (+17.0 to +37.0).
GROBID cannot read image-only scans at all, because it has no text recognition
(OCR). `--fast` also needs the PDF's text layer, so it does not read scans
either.

The `--fast` scores were run with `LAYOUT_TEXT_LAYER_DPI=200`. At 0.9.2's
default of 100 DPI, `--fast` gets 88.5% of the behavioural-science papers fully
right instead of 90.5%, and still 91.0% in the other fields.

## Field by field

Each field also gets its own score per paper, from 0 (all wrong) to 1
(perfect), averaged over the 300 born-digital papers. For example, 0.988 for
author names means that, on a typical paper, practically every author comes
out with the correct full name.

### Highlights

bibr leads most on the front matter and on reference details, and it got the
DOI right on all 300 papers, where GROBID missed it on 5.

| Field (metric) | bibr default | bibr `--fast` | GROBID 0.9.1 |
| --- | --- | --- | --- |
| Affiliations (token-set similarity) | 0.932 | 0.932 | 0.859 |
| Abstract (ROUGE-L F1) | 0.970 | 0.951 | 0.859 |
| Title (soft match) | 0.997 | 0.993 | 0.910 |
| Author names (full-name F1) | 0.988 | 0.990 | 0.906 |
| Reference pages (accuracy) | 0.993 | 0.993 | 0.918 |
| In-text citations (F1) | 0.952 | 0.954 | 0.913 |
| Repository links found (recall) | 0.975 | 0.975 | 0.945 |
| Links to data and code (F1) | 0.930 | 0.934 | 0.908 |
| Reference journals (accuracy) | 0.986 | 0.986 | 0.962 |
| Reference list found (matching F1) | 0.964 | 0.965 | 0.946 |
| DOI (exact match) | 1.000 | 1.000 | 0.983 |

### Full list

Bold marks the highest score in each row, with ties all in bold. It does not
mean the gap is statistically significant.

| Field (metric) | bibr default | bibr `--fast` (CPU, 200 DPI) | GROBID 0.9.1 |
| --- | --- | --- | --- |
| Title (soft match: exact, ignoring case and punctuation) | **0.997** | 0.993 | 0.910 |
| DOI (exact match) | **1.000** | **1.000** | 0.983 |
| Author names (full-name F1) | 0.988 | **0.990** | 0.906 |
| Abstract (ROUGE-L F1) | **0.970** | 0.951 | 0.859 |
| Affiliations (token-set similarity) | **0.932** | **0.932** | 0.859 |
| Keywords (F1; 216 papers) | 0.941 | **0.958** | 0.922 |
| Corresponding author (exact-match accuracy; 289 papers) | 0.578 | **0.758** | 0.616 |
| ORCID iDs (F1; 36 papers) | 0.028 | not reported | **0.408** |
| Reference list found (reference-matching F1) | 0.964 | **0.965** | 0.946 |
| Reference titles (accuracy, fuzzy match ≥ 85) | **0.986** | **0.986** | **0.986** |
| Reference authors (accuracy, every surname found) | **0.994** | 0.993 | 0.978 |
| Reference years (accuracy) | **0.999** | **0.999** | 0.984 |
| Reference journals (accuracy, fuzzy match ≥ 85) | **0.986** | **0.986** | 0.962 |
| Reference volumes (accuracy, exact) | **0.994** | **0.994** | 0.979 |
| Reference pages (accuracy, exact pages) | **0.993** | **0.993** | 0.918 |
| Reference DOIs found (recall; 275 papers) | 0.986 | **0.987** | 0.983 |
| In-text citations tied to the right reference (F1; 28,090) | 0.952 | **0.954** | 0.913 |
| Links to data and code (F1; 378) | 0.930 | **0.934** | 0.908 |
| Repository links: OSF, GitHub, Zenodo (recall; 199) | **0.975** | **0.975** | 0.945 |

**How to read the metrics.** F1 balances what a tool missed against what it
added wrongly; 1.0 means nothing missed and nothing extra. ROUGE-L F1 compares
the abstract word by word with the printed one, in order; 1.0 means word for
word. Token-set similarity compares the words of each author's affiliation,
ignoring their order. Accuracy is the share of matched references where that
detail is right; recall is the share of the answer key's items that the tool
found (for reference DOIs, out of every reference with a DOI). `--fast` scores
come from a 20 November CPU run of the same code (corresponding author from 24
November). For reference volumes, authors and DOIs, GROBID's row uses its 0.9.0
values, which 0.9.1 matched to within ±0.002. The
[evaluation guide](contributing/evaluation.md) defines every metric.

## Speed

| How it runs | Papers per hour |
| --- | --- |
| bibr, one workstation GPU (RTX 5090), `--fast` | **1,565** |
| bibr, one workstation GPU, default local models | **1,005** |
| bibr, CPU only, no GPU (`--fast`, born-digital only) | 407 |
| GROBID 0.9.1, CPU, 6 workers | 4,650 |

All on the same 120 born-digital psychology PDFs (1,890 pages), from a cold
start, on one workstation. GROBID is 3 to 4.6 times faster than bibr's GPU
routes: that is the price of the accuracy gap. GROBID ran while the bibr GPU
run was also using the machine, so its figure is slightly low. The bibr figures
come from earlier timed runs of the same code (the CPU figure from 20 November
2026); a repeat timing of the 0.9.2 build is in progress.

## Tested on

- **300 recent research articles** (July to October 2026) from 18 open-access
  journals: 200 in behavioural science, 100 in other fields. All CC BY, from
  PubMed Central.
- **Answer key from the publishers' own XML** (JATS), checked against what each
  PDF prints. No language model was used to make it.
- **90 simulated scans:** the same papers turned into degraded page images with
  no text layer.
- **The same answer key and scorer for both tools.**

### Where the 300 papers come from

| Journal | Publisher | Papers |
| --- | --- | --- |
| **Behavioural science (200)** | | |
| Frontiers in Psychology | Frontiers | 34 |
| PLOS ONE (psychology) | PLOS | 34 |
| BMC Psychology | Springer Nature (BMC) | 33 |
| Psychological Medicine | Cambridge University Press | 26 |
| JMIR Mental Health | JMIR Publications | 22 |
| Open Mind | MIT Press | 19 |
| Developmental Science | Wiley | 18 |
| Journal of Cognition | Ubiquity Press | 12 |
| Royal Society Open Science; Cognitive Research: Principles and Implications | The Royal Society; Springer Nature | 1 each |
| **Other fields (100)** | | |
| PeerJ | PeerJ | 13 |
| Nature Communications | Springer Nature | 13 |
| PLOS ONE (economics) | PLOS | 13 |
| Ecology and Evolution | Wiley | 13 |
| PLOS Computational Biology | PLOS | 12 |
| eLife | eLife Sciences Publications | 12 |
| BMC Public Health | Springer Nature (BMC) | 12 |
| Global Health Action | Taylor & Francis | 12 |

All research articles (one brief report), published 2 July to 9 October 2026.
The list was frozen on 14 October, before either tool was run. Length: median
14 pages (6 to 40) in behavioural science, 16 pages (6 to 36) in other fields.
The 90 simulated scans are 60 behavioural-science and 30 other-field papers
from this list. Full list with DOIs:
[papers.csv](https://github.com/scienceverse/bibr-jats-eval/blob/main/papers.csv).

## Known limits

- **GROBID is faster:** about 4,650 papers an hour, 3 to 4.6 times bibr's GPU
  speed (see [Speed](#speed)).
- **One journal drives part of the gap:** GROBID fails every Frontiers in
  Psychology paper (34 of the 200 behavioural-science papers), because it reads
  the handling editor and reviewers as authors. Without that journal, the
  October comparison with GROBID 0.9.0 gave a gap of +19.9 points instead of
  +29.5 (public benchmark).
- **Not yet validated:** section structure, and data-availability, funding and
  conflict-of-interest statements. bibr extracts them, but they are not scored
  yet.
- **Weaker fields:** the corresponding author (0.58 by default vs GROBID's
  0.62, though `--fast` gets 0.76) and ORCID iDs (0.03 vs 0.41).
- **Sample:** recent English open-access papers, not a random sample of the
  literature. Real scans of old print are harder than simulated ones.

See also [Known limitations](limitations.md).

## Check it yourself

- [Public benchmark](https://github.com/scienceverse/bibr-jats-eval): the paper
  list, answer key, scorer and per-paper scores for both tools.
- [bibr Dev Hub](https://hub.bibr.org/validation) (sign-in needed): every
  module's evidence and known gaps.
- [Changelog](https://github.com/scienceverse/bibr/blob/main/CHANGELOG.md): what
  changed in each release.

## How these numbers were made

bibr 0.9.2 (commit `f18a242`) ran on 24 November 2026 on one workstation with
an RTX 5090, with the default local models (a fine-tuned Qwen3.5 4B extractor
and PaddleOCR-VL). GROBID 0.9.1 (Docker image `grobid/grobid:0.9.1-full`) ran
on 24 November 2026 on the same machine. Both were scored with the same answer
key and scorer (evaluate v5; links and in-text citations with the benchmark's
extension scorer, v2). GROBID 0.9.1 scores the same as 0.9.0 except on
affiliations, which it improved.
