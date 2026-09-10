# Evaluation

bibr's evaluation harness scores saved extraction JSON against printed-paper
ground truth. It reports title, DOI, abstract, author, and reference metrics,
plus paper-level pass rates and cohort coverage. **Primary** metrics participate
in the optional `--threshold` mean gate. **Diagnostic** metrics support analysis
and do not participate in that gate.

!!! note "Source checkout required"
    Run `uv run python -m evaluation.evaluate` from a source checkout with the
    development dependencies installed. The harness is not part of the
    published `bibr` package.

!!! note "Bring your own ground truth"
    Reference corpora and extraction captures are not distributed with bibr.
    Supply independently prepared gold exports with `--gold-dirs`. The harness
    fails when no gold directory exists or no predictions match ground truth.

## Quick start

```bash
# Use your own predictions and independently prepared gold exports
uv run python -m evaluation.evaluate \
    --results-dir outputs/local \
    --gold-dirs /path/to/gold \
    --output outputs/local-eval.json
```

Gold directories contain one bibr-shaped JSON export per paper. The harness
matches predictions by normalized DOI, falling back to filename, and prints
aggregate metrics. Keep gold independent of the predictions being evaluated;
copying an extraction into gold would preserve the same errors on both sides.

### Fixed cohorts and regression gates

```bash
uv run python -m evaluation.evaluate \
    --results-dir outputs/local \
    --gold-dirs /path/to/gold \
    --expected-ids /path/to/expected-paper-ids.json \
    --output outputs/local-eval.json \
    --threshold 0.80
```

`--expected-ids` accepts a JSON file containing a list of paper IDs or an
object with `members` or `ids`. Expected IDs without a prediction are added
with zero scores on the paper-level floor metrics. Otherwise, missing files
would disappear from the denominator. `--ids-file` separately restricts scoring
to prediction filenames listed one per line (without `.json`).

The `--threshold` flag exits with status 1 if any primary metric's non-null
mean is below the supplied value. It does not gate missing/unmatched counts,
abstention rate, or paper-level pass rate. Inspect those fields too when
comparing runs. Choose a threshold from a recorded baseline scored with the
same metric version and cohort; `0.80` above is only an example.

Public CI tests the evaluator using small synthetic examples. It checks scoring
behavior and software contracts; passing CI does not establish extraction
accuracy for your papers.

## Ground truth

Always provide `--gold-dirs` explicitly. Gold represents what is printed in
your papers, including reference fields; enrichment output is not extraction
ground truth. Keep the gold revision fixed when comparing saved runs and record
how each field was checked against its source.

## Running extraction for evaluation

Evaluate against exports produced by your own `bibr chew` run:

```bash
uv run bibr chew papers/ -o outputs/local/
uv run python -m evaluation.evaluate \
    --results-dir outputs/local/ \
    --gold-dirs /path/to/gold
```

## Metrics reference

Metric scores are in [0, 1], with higher being better. A metric can be `null`
when gold provides nothing to score; those entries are excluded from its mean.
The current definitions are **`metrics_version: 4`**, recorded in every saved
evaluation. Re-score predictions when definitions change rather than comparing
means from different versions.

### Title

| Metric | Tier | Description |
|---|---|---|
| `title_soft` | primary | Exact match after case, punctuation, and whitespace normalization, or a word-aligned prefix retaining at least 70% of the longer normalized title. The shorter side needs at least two words, or six characters in supported scripts without inter-word spaces. |

### DOI

| Metric | Tier | Description |
|---|---|---|
| `doi_match` | primary | 1.0 for equal normalized DOIs. An asserted DOI against empty gold scores 0; when both are empty there is no score. |

### Abstract

| Metric | Tier | Description |
|---|---|---|
| `abstract_rouge_l` | primary | ROUGE-L F1 score using whitespace tokenization and longest common subsequence |
| `abstract_ned` | diagnostic | Normalized Edit Distance similarity (1 - NED), following OmniDocBench methodology |

Predictions are scored from `info.abstract` only. The evaluator does not recover
an absent exported abstract from section text. Gold may use its abstract-typed
sections because some gold records store their abstract there.

### Authors

| Metric | Tier | Description |
|---|---|---|
| `authors_fullname_f1` | primary | F1 on full printed names ("given family") using fuzzy similarity — boundary-insensitive to the given/family split |
| `authors_f1` | diagnostic | F1 on normalized family names with fuzzy, globally greedy bipartite matching |
| `first_author` | diagnostic | Fuzzy similarity (token_sort_ratio) on the first author's family name |

Per-author front-matter fields (`affiliation_sim`, `email_f1`, `orcid_f1`, `corresponding_acc`) and `keywords_f1` are also emitted as diagnostics; they score only over papers whose gold carries the field.

### References

| Metric | Tier | Description |
|---|---|---|
| `ref_matching_f1` | primary | F1 for individual reference matching using a four-level cascade: DOI match (1.0) > fuzzy title match (0.9) > first-author + year (0.7) > unstructured citation string (0.6). Uses globally-greedy bipartite matching |
| `ref_title_acc` | primary | Among matched GT references that have titles, fraction with a correct title (fuzzy token_sort_ratio >= 85) |
| `ref_year_acc` | primary | Among matched GT references that have years, fraction with a correct year |
| `ref_doi_recall` | primary | Among **all** gold references with DOIs, fraction recovered with the correct DOI. Unmatched gold references with DOIs count as misses. |
| `ref_count_ratio` | diagnostic | min/max ratio of reference counts |
| `ref_author_acc` | diagnostic | Among matched gold references with authors, fraction where every gold surname is recalled |
| `ref_journal_acc` | diagnostic | Among matched GT references with a container/journal, fraction correct |
| `ref_volume_acc` | diagnostic | Among matched GT references with a volume, fraction correct |
| `ref_pages_acc` | diagnostic | Among matched GT references with page ranges, fraction correct |

!!! note "Sparse GT fields"
    The `ref_*_acc` metrics use matched pairs where gold has the field.
    `ref_doi_recall` instead uses all gold references with a DOI. Saved results
    include field coverage, counts, and micro-averaged scores to make these
    denominators visible.

## Output format

`--output` writes a JSON artifact containing:

| Fields | Meaning |
|---|---|
| `metrics_version`, `generated_at`, `bibr_commit` | Metric definitions, scoring timestamp, and commit of the evaluator checkout |
| `predictions_dir`, `predictions_tree_sha256` | Prediction source and content digest; the evaluator commit alone does not identify the code that produced these predictions |
| `papers_attempted`, `papers_evaluated`, `papers_missing`, `missing_ids` | Cohort accounting, including missing expected predictions |
| `papers_unmatched`, `unmatched_ids` | Predictions that matched no gold record and were excluded from aggregates |
| `metrics` | Per-metric mean, p10, median, min, max, and non-null count `n`, with additional abstention and reference-coverage fields where applicable |
| `pass_rate`, `pass_rate_n` | Fraction passing the paper-level floors and the exact denominator used |
| `abstention_rate`, `n_abstained`, `abstained_ids` | Front-matter abstentions and their share of evaluated predictions |
| `per_paper` | Individual scores, flags, and reference-field counts for investigation |

`pass_rate` uses these floors: `title_soft >= 0.9`, `doi_match >= 1.0`,
`authors_fullname_f1 >= 0.9`, and `ref_matching_f1 >= 0.8`. A null field is
excluded from that paper's floor checks. Missing expected predictions and
front-matter abstentions count as failures in the headline rate. The legacy
survivors-only rate is retained separately as `pass_rate_excl_abstained` with
its own denominator; it is not interchangeable with `pass_rate`.

Front-matter abstentions suppress affected field scores from ordinary metric
means. `mean_incl_abstained` reports the companion mean including their
pre-suppression scores. Always read mean accuracy alongside abstention and
cohort coverage.

## Section-text evaluation

The same entry point can score saved exports against `*.sectiongold.json`
artifacts:

```bash
uv run python -m evaluation.evaluate \
    --results-dir outputs/local \
    --sections \
    --section-gold-dirs /path/to/section-gold \
    --output outputs/local-sections.json
```

This mode emits section recall, coverage, and a drop report. It has a separate
output shape from metadata scoring, and returns before the metadata
`--threshold` gate. Per-type section recall is diagnostic.

## Interpreting results

Compare runs on the same papers, gold revision, metric version, and extraction
configuration. Record the extraction build and model identities separately from
the scoring commit. Review missing and unmatched papers before interpreting
accuracy changes, and use `per_paper` to separate OCR/native-text failures,
metadata errors, and reference segmentation or parsing errors. Historical
scores without this provenance are not a baseline for current `main`.
