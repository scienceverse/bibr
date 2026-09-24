"""Evaluation runner — compares bibr extraction results against ground truth.

Loads user-supplied gold exports, scores saved predictions or Paper objects,
and produces per-paper and aggregate metrics. No corpus is bundled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from bibr.validation import payload_validation
from evaluation.section_metrics import (
    BODY_SECTION_TYPES,
    SCORED_TYPES,
    aggregate_by_type,
    build_drop_report_rows,
    score_section,
    tokenize,
    unigram_recall,
)
from evaluation.validation_metrics import (
    FLOORS,
    # Reference matching/field internals, reused verbatim so the per-paper
    # denominators emitted here can never disagree with the accuracies
    # ref_field_scores() computes from the same pairs. Ideally ref_field_scores()
    # would return its own counts and this second matching pass would go away.
    _get_ref_container,
    _get_ref_pages,
    _get_ref_title,
    _get_ref_volume,
    _get_ref_year,
    _greedy_match_pairs,
    _ref_similarity,
    _ref_surname_tokens,
    abstract_ned,
    abstract_rouge_l,
    authors_family_f1,
    authors_fullname_f1,
    doi_match,
    first_author_match,
    normalize_doi,
    paper_passes_floors,
    pass_rate,
    ref_field_scores,
    ref_matching_f1,
    references_count_ratio,
    title_soft_match,
)

logger = logging.getLogger(__name__)

# Metric-definition version recorded in every scoring artifact. Re-score inputs
# before comparing results produced by different metric versions. Bump it only
# when a definition changes, together with docs/contributing/evaluation.md.
# v2: abstention-aware metrics and title containment for spaceless scripts.
# v3: bounded title containment, empty-gold DOI penalties, stricter author matching.
# v4: full-cohort pass rate; prediction abstracts come only from metadata.abstract
#     (info.abstract before schema 11, still read).
# Reading schema 11 and 12 exports changed no definition. The 0.5.0 changelog's
# metrics_version=6 was a counter shared with scorers that are not in this
# repository; it does not apply to this evaluator.
METRICS_VERSION = 4

# Suffix for the shadow columns holding a metric's value *before* an abstention
# suppressed it (see _apply_abstentions). Never emitted in ``per_paper`` — they
# exist only to compute the ``mean_incl_abstained`` companion.
_PREABSTENTION_SUFFIX = "__preabstention"

_BIBR_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Artifact provenance
# ---------------------------------------------------------------------------


def bibr_commit(repo_root: Path = _BIBR_REPO_ROOT) -> str | None:
    """HEAD of the bibr checkout doing the scoring, or None if unknowable.

    Without it an artifact cannot be tied to a build, and a re-score of an old
    prediction snapshot is indistinguishable from a fresh run.
    """
    try:
        proc = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = proc.stdout.strip()
    return commit if proc.returncode == 0 and commit else None


def _scored_prediction_paths(json_dir: Path, ids: set[str] | None = None) -> list[Path]:
    """The prediction files ``evaluate_json_exports`` would actually read."""
    return [
        p
        for p in sorted(json_dir.glob("*.json"))
        if p.name != "validation_report.json" and (ids is None or p.stem in ids)
    ]


def predictions_tree_sha256(json_dir: Path, ids: set[str] | None = None) -> str:
    """Stable digest over the prediction files actually scored.

    Hashes ``filename -> sha256(content)`` for the sorted file list, so two
    artifacts can be compared for input identity without shipping the inputs.
    """
    digest = hashlib.sha256()
    for path in _scored_prediction_paths(json_dir, ids):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_expected_ids(spec: str, registry_path: Path | None = None) -> set[str]:
    """Resolve ``--expected-ids`` to the set of paper ids the run should cover.

    ``spec`` is a JSON file (a list of ids, or an object with a ``members``/``ids``
    list). A named ``set_id`` also works when ``registry_path`` is supplied.
    """
    path = Path(spec).expanduser()
    if path.is_file():
        payload = json.loads(path.read_text())
        members = payload if isinstance(payload, list) else None
        if members is None:
            members = payload.get("members") or payload.get("ids")
        if not isinstance(members, list):
            raise ValueError(f"{path} does not contain a list of ids (or a 'members' list)")
        return {str(m) for m in members}

    if registry_path is None:
        raise ValueError(
            f"--expected-ids {spec!r} is not an existing JSON file. "
            "Provide a file of paper IDs or pass --evaluation-sets for a named set."
        )
    registry = Path(registry_path)
    if not registry.is_file():
        raise ValueError(
            f"--expected-ids {spec!r} is neither an existing file nor resolvable: "
            f"evaluation-set registry not found at {registry}"
        )
    sets = json.loads(registry.read_text()).get("sets") or []
    for entry in sets:
        if entry.get("set_id") == spec:
            return {str(m) for m in entry.get("members") or []}
    known = ", ".join(sorted(str(e.get("set_id")) for e in sets))
    raise ValueError(f"Unknown evaluation set {spec!r}; {registry} defines: {known}")


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


def load_ground_truth_crossref(
    parquet_path: Path,
    crossref_json_path: Path,
) -> pd.DataFrame:
    """Legacy Crossref ground truth: merges parquet + Crossref API JSON.

    Returns DataFrame with columns:
        doi, file_name, title, authors (list[dict] with family/given),
        abstract, reference_count (int), references (list[dict]).

    Prefer ``load_ground_truth_gold`` for new evaluations — gold has better
    abstract coverage, structured authors per reference, and matches the
    bibr v10 schema 1:1 so per-field metrics are apples-to-apples.
    """
    # Load parquet
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    # Normalize authors from crossref_authors column
    df["authors"] = df["crossref_authors"].apply(_normalize_authors)

    # Load Crossref JSON for reference counts
    with open(crossref_json_path) as f:
        crossref_data = json.load(f)

    df["reference_count"] = df["doi"].apply(
        lambda doi: len(crossref_data.get(doi, {}).get("reference", []))
    )

    # Extract individual reference records for ref-matching metrics
    df["references"] = df["doi"].apply(lambda doi: crossref_data.get(doi, {}).get("reference", []))

    # Clean abstract (may contain JATS XML tags)
    df["abstract"] = df["abstract"].apply(_clean_abstract)

    return df[
        [
            "doi",
            "file_name",
            "title",
            "authors",
            "abstract",
            "reference_count",
            "references",
        ]
    ]


def load_ground_truth_gold(
    gold_dirs: list[Path] | tuple[Path, ...],
) -> pd.DataFrame:
    """Load independently prepared gold in bibr v10 JSON format.

    Each ``gold_dir`` holds one paper per ``*.json`` file (with optional
    ``_manifest.json`` summaries that are skipped). Each JSON is treated as
    the source of truth for that paper; callers must prepare it independently
    of the predictions being scored.

    Returns a DataFrame with the same column shape as
    ``load_ground_truth_crossref`` (doi, file_name, title, authors,
    abstract, reference_count, references), plus ``keywords``.
    """
    gold_dirs = [Path(d) for d in gold_dirs]
    existing = [d for d in gold_dirs if d.exists()]
    if not existing:
        raise FileNotFoundError(
            "No gold ground-truth dirs found. Pass --gold-dirs with your independently "
            "prepared gold exports. See docs/contributing/evaluation.md. "
            f"Looked for: {[str(d) for d in gold_dirs]}"
        )

    rows: list[dict] = []
    for gold_dir in gold_dirs:
        gold_dir = Path(gold_dir)
        if not gold_dir.exists():
            logger.warning("Gold dir not found, skipping: %s", gold_dir)
            continue
        for json_path in sorted(gold_dir.glob("*.json")):
            if json_path.name.startswith("_"):
                continue
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse gold JSON %s: %s", json_path, e)
                continue

            extracted = extract_comparable_from_json(data, is_gold=True)
            info = data.get("metadata") or data.get("info") or {}
            # Use canonical DOI and PDF filename from the gold record itself.
            # `doi` (paper_id fallback) is the JOIN KEY; `doi_printed` is what
            # the page actually shows and is what doi_match scores against —
            # a paper that prints no DOI must be excluded, not penalized.
            doi = info.get("doi") or data.get("paper_id", "") or extracted.get("doi", "")
            file_name = (
                (data.get("source") or {}).get("file_name")
                or info.get("file_name")
                or json_path.stem
            )
            rows.append(
                {
                    "doi": doi,
                    "doi_printed": info.get("doi") or "",
                    "file_name": file_name,
                    "title": extracted.get("title", ""),
                    "authors": extracted.get("authors", []),
                    "keywords": extracted.get("keywords", []),
                    "abstract": extracted.get("abstract", ""),
                    "reference_count": extracted.get("reference_count", 0),
                    "references": extracted.get("references", []),
                }
            )

    if not rows:
        logger.warning("Gold loader produced 0 rows from dirs: %s", list(gold_dirs))

    return pd.DataFrame(rows)


def _author_affiliation(data: dict, author: dict) -> str:
    """One author's affiliation string, from any export version.

    Gold records and exports up to v11 carry ``author[].affiliation`` (the
    byline components joined by "; "). v12 dropped it in favour of the
    ``affiliation[]`` table; joining the author's rows in table order rebuilds
    the same string.
    """
    if "affiliation" in author:
        return author.get("affiliation") or ""
    author_id = author.get("author_id")
    return "; ".join(
        row.get("text") or ""
        for row in data.get("affiliation") or []
        if isinstance(row, dict) and author_id in (row.get("author_ids") or [])
    )


def _normalize_authors(authors_raw) -> list[dict]:
    """Normalize crossref_authors into list of {family, given} dicts."""
    if authors_raw is None:
        return []
    result = []
    for a in authors_raw:
        if isinstance(a, dict):
            result.append({"family": a.get("family", ""), "given": a.get("given", "")})
    return result


def _abstract_from_sections(data: dict) -> str:
    """Join the text rows of every ABSTRACT-typed section, in export order.

    The gold-only half of the D2 split (see extract_comparable_from_json). Kept
    byte-identical to the fallback that used to run for both sides so the gold
    abstract — and therefore every historical abstract_rouge_l — is unchanged.
    """
    abstract_section_ids = {
        s["section_id"]
        for s in data.get("section", [])
        if (s.get("section_type") or "").lower() == "abstract"
    }
    if not abstract_section_ids:
        return ""
    return " ".join(
        t["text"] for t in data.get("text", []) if t.get("section_id") in abstract_section_ids
    )


def _clean_abstract(text: str | None) -> str:
    """Strip JATS XML tags from abstract text."""
    if not text:
        return ""
    import re

    return re.sub(r"<[^>]+>", "", text).strip()


# ---------------------------------------------------------------------------
# Paper field extraction
# ---------------------------------------------------------------------------


def extract_comparable(paper) -> dict:
    """Extract comparable fields from a Paper object.

    Args:
        paper: A bibr.paper.Paper instance.

    Returns:
        Dict with keys: title, doi, authors (full dicts with family, given,
        affiliation, email, orcid, corresponding), keywords, abstract,
        reference_count, references.
    """
    from bibr.paper_contents import CanonicalSection

    metadata = paper.metadata
    contents = paper.contents

    # Extract abstract text from ABSTRACT sections
    abstract = ""
    if contents:
        abstract_section_ids = {
            s.section_id for s in contents.sections if s.section_type == CanonicalSection.ABSTRACT
        }
        if abstract_section_ids:
            abstract_sents = [
                sent.text for sent in contents.sentences if sent.section_id in abstract_section_ids
            ]
            abstract = " ".join(abstract_sents)

    # Extract individual reference details for ref-matching. Raw extraction
    # (r.doi) and enriched (r.match[...].doi) are kept side-by-side so the
    # evaluator can report both `ref_doi_recall` and `ref_doi_recall_enriched`.
    refs = []
    refs_enriched = []
    if metadata:
        for r in metadata.references:
            ref_dict: dict = {}
            if hasattr(r, "title") and r.title:
                ref_dict["title"] = r.title
            if hasattr(r, "doi") and r.doi:
                ref_dict["doi"] = r.doi
            if hasattr(r, "authors") and r.authors:
                ref_dict["author"] = r.authors if isinstance(r.authors, str) else ""
            if hasattr(r, "year") and r.year:
                ref_dict["year"] = str(r.year)
            refs.append(ref_dict)

            enriched = dict(ref_dict)
            if not enriched.get("doi") and hasattr(r, "match") and r.match:
                for source in ("crossref", "openalex", "openlibrary"):
                    sm = r.match.get(source) if isinstance(r.match, dict) else None
                    if sm and getattr(sm, "doi", None):
                        enriched["doi"] = sm.doi
                        break
            refs_enriched.append(enriched)

    return {
        "title": metadata.title if metadata else "",
        "doi": metadata.doi if metadata else "",
        "authors": [
            {
                "family": a.family,
                "given": a.given,
                "affiliation": a.affiliation or "",
                "email": a.email or "",
                "orcid": a.orcid or "",
                "corresponding": bool(a.corresponding),
            }
            for a in (metadata.authors if metadata else [])
        ],
        "keywords": list(metadata.keywords) if metadata else [],
        "abstract": abstract,
        "reference_count": len(metadata.references) if metadata else 0,
        "references": refs,
        "references_enriched": refs_enriched,
    }


def extract_sections_from_json(data: dict) -> dict[str, str]:
    """Concatenate a prediction's text[] by canonical section_type.

    Mirrors how the artifact groups the reference, so the two are comparable."""
    type_by_sid = {
        s.get("section_id"): (s.get("section_type") or "").lower() for s in data.get("section", [])
    }
    # Since 12.0 captions and footnotes have no section; group them by what
    # points at them, as their own sections were typed before.
    type_by_text_id = {
        row.get("text_id"): kind
        for kind in ("figure", "table", "footnote")
        for row in data.get(kind) or []
        if isinstance(row, dict) and row.get("text_id") is not None
    }
    by_type: dict[str, list[str]] = {}
    for t in data.get("text", []):
        st = type_by_sid.get(t.get("section_id")) or type_by_text_id.get(t.get("text_id"))
        if not st:
            continue
        txt = t.get("text") or ""
        if txt:
            by_type.setdefault(st, []).append(txt)
    return {st: "\n".join(chunks) for st, chunks in by_type.items()}


def extract_comparable_from_json(data: dict, *, is_gold: bool = False) -> dict:
    """Extract comparable fields from a bibr JSON export (current or legacy schema).

    Args:
        data: Parsed JSON export dict with top-level keys: metadata
              (``info`` before schema 11), author, text, section, bib, etc.
        is_gold: True when ``data`` is a gold record rather than a prediction.
            Only the abstract differs; see below.

    Returns:
        Dict with keys: title, doi, authors (full dicts with family, given,
        affiliation, email, orcid, corresponding), keywords, abstract,
        reference_count, references, file_name, abstained.
    """
    info = data.get("metadata") or data.get("info") or {}

    # Score the abstract consumers receive in metadata.abstract. Reconstructing an
    # empty prediction from sections would conceal a missing exported field.
    # Gold may instead store its independently prepared abstract as section text.
    if is_gold:
        abstract = info.get("abstract") or _abstract_from_sections(data)
    else:
        abstract = info.get("abstract") or ""

    # Build enriched-DOI lookup from bib_match (v10.0) and nested match dict (v9.0)
    enriched_doi_by_bib_id: dict = {}
    for bm in data.get("bib_match", []):
        if bm.get("bib_id") is not None and bm.get("doi"):
            enriched_doi_by_bib_id.setdefault(bm["bib_id"], bm["doi"])

    # Extract individual reference details. Build two parallel lists:
    #   refs           — raw bibr extraction (bib.doi only)
    #   refs_enriched  — same, but with bib_match/match fallback for DOI
    # This lets us report `ref_doi_recall` (extraction quality) alongside
    # `ref_doi_recall_enriched` (consumer-facing, bibr + Crossref).
    refs = []
    refs_enriched = []
    for bib in data.get("bib", []):
        ref_dict: dict = {}
        if bib.get("title"):
            ref_dict["title"] = bib["title"]
        if bib.get("doi"):
            ref_dict["doi"] = bib["doi"]
        if bib.get("authors"):
            authors_val = bib["authors"]
            if isinstance(authors_val, str):
                ref_dict["author"] = authors_val
                ref_dict["authors"] = authors_val
            elif isinstance(authors_val, list) and authors_val:
                first = authors_val[0]
                ref_dict["author"] = (
                    first.get("family", "") if isinstance(first, dict) else str(first)
                )
                ref_dict["authors_list"] = [
                    a.get("family", "") if isinstance(a, dict) else str(a) for a in authors_val
                ]
        if bib.get("year") or bib.get("publication_year"):
            ref_dict["year"] = str(bib.get("year") or bib["publication_year"])
        if bib.get("container"):
            ref_dict["container"] = bib["container"]
        if bib.get("volume"):
            ref_dict["volume"] = str(bib["volume"])
        if bib.get("first_page"):
            ref_dict["first_page"] = str(bib["first_page"])
        if bib.get("last_page"):
            ref_dict["last_page"] = str(bib["last_page"])
        refs.append(ref_dict)

        # Enriched variant: same dict, but fill in DOI from bib_match if missing
        enriched = dict(ref_dict)
        if not enriched.get("doi"):
            enr = enriched_doi_by_bib_id.get(bib.get("bib_id"))
            if not enr:
                match = bib.get("match") or {}
                for source in ("crossref", "openalex", "openlibrary"):
                    sm = match.get(source)
                    if isinstance(sm, dict) and sm.get("doi"):
                        enr = sm["doi"]
                        break
            if enr:
                enriched["doi"] = enr
        refs_enriched.append(enriched)

    return {
        "title": info.get("title", ""),
        "doi": info.get("doi", ""),
        "authors": [
            {
                # A group author's whole name is ``literal`` (12.0 and later).
                "family": a.get("family") or a.get("literal") or "",
                "given": a.get("given") or "",
                "affiliation": _author_affiliation(data, a),
                "email": a.get("email") or "",
                "orcid": a.get("orcid") or "",
                "corresponding": bool(a.get("corresponding")),
            }
            for a in data.get("author", [])
        ],
        "keywords": [k for k in (info.get("keywords") or []) if isinstance(k, str)],
        "abstract": abstract,
        "reference_count": len(data.get("bib", [])),
        "references": refs,
        "references_enriched": refs_enriched,
        # PDF basename — used as fallback match key when extraction's DOI is
        # hallucinated. paper_id is the DOI in v10 schema, so prefer info.file_name.
        "file_name": (data.get("source") or {}).get("file_name")
        or info.get("file_name")
        or data.get("paper_id", ""),
        "abstained": _front_matter_abstained(data),
    }


# Blocking validation codes meaning "the pipeline declined to assert front-matter
# metadata", not "the pipeline got it wrong" — resolve_front_matter() abstains
# rather than guess when it cannot select one record uniquely and safely. The
# per-metric means still exclude the fields an abstention suppresses (a refusal
# carries no evidence about title/author quality), with the penalized companion
# reported as `<metric>.mean_incl_abstained` — see _apply_abstentions.
#
# The PAPER-LEVEL pass_rate does NOT get that exemption any more. A consumer of an
# abstaining record receives a byte-empty PaperMetadata, which is a failed paper
# whatever the pipeline's reason for it was.
#
FRONT_MATTER_ABSTENTION_CODES = frozenset({"VAL_METADATA_MULTI_ITEM"})

# The fields resolve_front_matter() suppresses when it abstains. Deliberately
# EXCLUDES doi_match (info.doi survives abstention via the separate
# extract/doi_identity path, so it stays honestly scored) and every ref_* metric
# (references are extracted independently of front matter).
ABSTENTION_SUPPRESSED_COLS = [
    "title_soft",
    "abstract_rouge_l",
    "abstract_ned",
    "authors_fullname_f1",
    "authors_f1",
    "first_author",
    "keywords_f1",
    "affiliation_sim",
    "email_f1",
    "orcid_f1",
    "corresponding_acc",
]


def _front_matter_abstained(data: dict) -> bool:
    """True iff the export carries a blocking front-matter abstention issue."""
    issues = (payload_validation(data) or {}).get("issues") or []
    return any(
        issue.get("blocking") and issue.get("code") in FRONT_MATTER_ABSTENTION_CODES
        for issue in issues
        if isinstance(issue, dict)
    )


# ---------------------------------------------------------------------------
# Section-text recall (benchmark for layout-dropped section text)
# ---------------------------------------------------------------------------


def load_section_gold(section_gold_dirs: list[Path]) -> dict[str, dict]:
    """Load every ``*.sectiongold.json`` artifact, keyed by paper_id."""
    out: dict[str, dict] = {}
    for d in section_gold_dirs:
        d = Path(d)
        if not d.exists():
            logger.warning("Section-gold dir not found, skipping: %s", d)
            continue
        for p in sorted(d.glob("*.sectiongold.json")):
            try:
                data = json.loads(p.read_text())
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse section-gold %s: %s", p, e)
                continue
            pid = data.get("paper_id")
            if not pid:
                logger.warning("Section-gold missing paper_id, skipping: %s", p)
                continue
            out[pid] = data
    return out


def score_paper_sections(pred_sections: dict[str, str], artifact: dict) -> list[dict]:
    """Score one paper: a row per SCORED_TYPE present (and anchored) in the
    artifact, plus a type-agnostic '_total_body' row (the layout-drop alarm).

    Caveats for interpreting the output:
    - Per-type recall (intro/method/results/discussion/abstract) is DIAGNOSTIC: a
      low per-type value can mean text was MISFILED to another type, not lost. The
      `_total_body` row is the layout-drop alarm.
    - `_total_body` is type-agnostic only WITHIN BODY_SECTION_TYPES; prediction text
      bibr misclassified to a non-body type (unknown/title/references) is excluded
      from the prediction total and reads as a drop — confirm a low `_total_body`
      recall against the PDF before attributing it to the layout model.
    - References whose gold header is not printed in the PDF (common for the intro
      in psych journals) anchor as `unanchored`; their body is absorbed into the
      preceding (abstract) slice, depressing per-type abstract/intro recall while
      still being captured in `_total_body`.
    """
    paper_id = artifact["paper_id"]
    ref_sections = artifact.get("sections", {})
    rows: list[dict] = []

    for st in SCORED_TYPES:
        ref = ref_sections.get(st)
        if not ref or ref.get("provenance") == "unanchored":
            continue
        sc = score_section(pred_sections.get(st, ""), ref.get("text", ""))
        rows.append({"paper_id": paper_id, "section_type": st, "pages": ref.get("pages", []), **sc})

    ref_total = "\n".join(
        info.get("text", "")
        for st, info in ref_sections.items()
        if st in BODY_SECTION_TYPES and info.get("provenance") != "unanchored"
    )
    ref_total_tokens = tokenize(ref_total)
    if ref_total_tokens:
        pred_total = "\n".join(v for st, v in pred_sections.items() if st in BODY_SECTION_TYPES)
        sc = score_section(pred_total, ref_total)
        # layout_recall ignores bibr's section labels: recall of the reference body
        # against ALL of bibr's text. It isolates true layout-drop (text bibr never
        # emitted) from misclassification (text bibr emitted under another type),
        # which the body-only unigram_recall above cannot distinguish.
        all_pred = "\n".join(pred_sections.values())
        layout_recall = unigram_recall(tokenize(all_pred), ref_total_tokens)
        rows.append(
            {
                "paper_id": paper_id,
                "section_type": "_total_body",
                "pages": [],
                **sc,
                "layout_recall": layout_recall,
            }
        )

    return rows


def evaluate_sections(
    results_dir: Path,
    section_gold_dirs: list[Path],
    ids: set[str] | None = None,
) -> tuple[dict, list[dict], dict]:
    """Join predictions (``<results_dir>/<paper_id>.json``) to section-gold
    artifacts by paper_id; return (per_type_summary, drop_rows, coverage)."""
    gold = load_section_gold(section_gold_dirs)
    per_paper_rows: list[dict] = []
    n_scored = 0
    missing_pred: list[str] = []
    usable = sum(1 for a in gold.values() if a.get("text_layer_usable"))
    unanchored = sum(len(a.get("unanchored", [])) for a in gold.values())

    for paper_id, artifact in gold.items():
        if ids is not None and paper_id not in ids:
            continue
        if not artifact.get("text_layer_usable", True):
            continue
        pred_path = Path(results_dir) / f"{paper_id}.json"
        if not pred_path.exists():
            logger.warning("Missing prediction for %s: %s", paper_id, pred_path)
            missing_pred.append(paper_id)
            continue
        try:
            data = json.loads(pred_path.read_text())
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse prediction %s: %s", pred_path, e)
            missing_pred.append(paper_id)
            continue
        per_paper_rows.extend(score_paper_sections(extract_sections_from_json(data), artifact))
        n_scored += 1

    per_type = aggregate_by_type(per_paper_rows)
    drop_rows = build_drop_report_rows(per_paper_rows)
    coverage = {
        "papers_with_artifact": len(gold),
        "usable_text_layer": usable,
        "scored": n_scored,
        "missing_prediction": missing_pred,
        "unanchored_sections": unanchored,
    }
    return per_type, drop_rows, coverage


def print_section_summary(per_type: dict, coverage: dict) -> None:
    """Print per-type means (recall/shingle/precision/rouge/ned) + presence,
    with the type-agnostic _total_body row last (the layout-drop alarm)."""
    print(
        f"\nSection-Text Recall ({coverage['scored']} papers scored, "
        f"{coverage['usable_text_layer']} usable text layers, "
        f"{coverage['unanchored_sections']} unanchored ref sections, "
        f"{len(coverage['missing_prediction'])} missing predictions)"
    )
    print("=" * 92)
    header = (
        f"{'type':<14}{'uni_rec':>9}{'shingle':>9}{'precis':>9}"
        f"{'rouge_l':>9}{'ned':>9}{'layout':>9}{'present':>9}{'N':>6}"
    )
    print(header)
    print("-" * 92)
    print("(layout = recall vs ALL bibr text, ignoring section labels: pure layout-drop)")

    def row(stype: str) -> None:
        s = per_type[stype]

        def m(k: str) -> str:
            v = s[k]["mean"] if isinstance(s.get(k), dict) else s.get(k)
            return f"{v:>9.3f}" if v is not None else f"{'—':>9}"

        pr = s.get("presence_rate")
        pr_s = f"{pr:>9.3f}" if pr is not None else f"{'—':>9}"
        print(
            f"{stype:<14}{m('unigram_recall')}{m('shingle_recall')}{m('unigram_precision')}"
            f"{m('rouge_l')}{m('ned')}{m('layout_recall')}{pr_s}{s['n_papers']:>6}"
        )

    for stype in SCORED_TYPES:
        if stype in per_type:
            row(stype)
    if "_total_body" in per_type:
        print("-" * 92)
        row("_total_body")
    print("=" * 92)
    if coverage["missing_prediction"]:
        print(
            f"Missing predictions for {len(coverage['missing_prediction'])} papers "
            f"(first few): {coverage['missing_prediction'][:5]}"
        )


def save_section_results(
    per_type: dict, drop_rows: list[dict], coverage: dict, output_path: Path
) -> None:
    """Persist the section benchmark: per-type summary, full drop report,
    coverage stats. JSON for regression tracking + programmatic inspection."""
    notes = [
        "Per-type recall (intro/method/results/discussion/abstract) is DIAGNOSTIC: "
        "a low value can mean text was MISFILED to another type, not lost; "
        "_total_body is the layout-drop alarm.",
        "_total_body is type-agnostic only WITHIN BODY_SECTION_TYPES; text bibr "
        "misclassified to a non-body type (unknown/title/references) is excluded "
        "from the prediction total — confirm a low _total_body recall against the "
        "PDF before attributing it to the layout model.",
        "Sections whose gold header is not printed in the PDF anchor as 'unanchored' "
        "and are excluded from per-type scoring; their body text is absorbed into the "
        "preceding slice and still captured in _total_body.",
    ]
    output = {"per_type": per_type, "drop_report": drop_rows, "coverage": coverage, "notes": notes}
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info("Section results saved to %s", output_path)


# ---------------------------------------------------------------------------
# Diagnostic field metrics (keywords + author affiliation/email/orcid/corresponding)
# ---------------------------------------------------------------------------


def _set_f1(e_set: set, g_set: set) -> float | None:
    """Set F1; None when gold empty (no signal), 0.0 when gold present but extraction empty."""
    if not g_set:
        return None
    if not e_set:
        return 0.0
    tp = len(e_set & g_set)
    p = tp / len(e_set)
    r = tp / len(g_set)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def keywords_f1(e_kws, g_kws) -> float | None:
    """Set-F1 over casefolded, whitespace-collapsed keywords. Gold empty/missing -> None."""

    def norm(ks):
        return {" ".join(k.split()).casefold() for k in ks if k and k.strip()}

    return _set_f1(norm(e_kws or []), norm(g_kws or []))


def email_f1(e_authors, g_authors) -> float | None:
    """Set-F1 over lowercased author emails. Gold has none -> None."""

    def emails(authors):
        return {a.get("email", "").strip().lower() for a in authors if a.get("email", "").strip()}

    return _set_f1(emails(e_authors or []), emails(g_authors or []))


def _normalize_orcid(orcid: str) -> str:
    return orcid.strip().removeprefix("https://orcid.org/").removeprefix("http://orcid.org/")


def orcid_f1(e_authors, g_authors) -> float | None:
    """Set-F1 over normalized author ORCIDs. Gold has none -> None."""

    def orcids(authors):
        return {_normalize_orcid(a.get("orcid", "")) for a in authors if a.get("orcid", "").strip()}

    return _set_f1(orcids(e_authors or []), orcids(g_authors or []))


def corresponding_acc(e_authors, g_authors) -> float | None:
    """1.0 iff the casefolded family names flagged corresponding match gold exactly.

    Gold flags none -> None.
    """

    def flagged(authors):
        return {
            (a.get("family") or "").casefold()
            for a in authors
            if a.get("corresponding") and (a.get("family") or "").strip()
        }

    g_set = flagged(g_authors or [])
    if not g_set:
        return None
    return 1.0 if flagged(e_authors or []) == g_set else 0.0


def affiliation_sim(e_authors, g_authors) -> float | None:
    """Mean token_set_ratio/100 over authors paired greedily by casefolded family.

    Pairs each gold author to the first unmatched extracted author with the same
    casefolded family name (order-preserving). Averaged only over pairs where the
    GOLD affiliation is non-empty. No gold affiliations anywhere -> None.
    """
    from rapidfuzz.fuzz import token_set_ratio

    e_authors = e_authors or []
    g_authors = g_authors or []
    used: set[int] = set()
    sims: list[float] = []
    for g in g_authors:
        g_aff = (g.get("affiliation") or "").strip()
        g_family = (g.get("family") or "").casefold()
        match_idx = None
        for i, e in enumerate(e_authors):
            if i in used:
                continue
            if (e.get("family") or "").casefold() == g_family:
                match_idx = i
                break
        e_aff = ""
        if match_idx is not None:
            used.add(match_idx)
            e_aff = (e_authors[match_idx].get("affiliation") or "").strip()
        if not g_aff:
            # Consuming the match above (regardless of empty gold affiliation) is
            # required so a later same-family gold author doesn't steal this
            # extraction author's slot; scoring itself is still gated on g_aff.
            continue
        sims.append(token_set_ratio(e_aff, g_aff) / 100)
    if not sims:
        return None
    return sum(sims) / len(sims)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


# Gold-side "does this reference carry the field?" predicates, one per
# ref_field_scores() metric. Reused from validation_metrics so the counts below
# select exactly the references that metric's denominator selects.
_REF_GOLD_FIELD_PREDICATES = {
    "title": lambda r: bool(_get_ref_title(r)),
    "year": lambda r: bool(_get_ref_year(r)),
    "doi": lambda r: bool(normalize_doi(r.get("doi") or r.get("DOI") or "")),
    "author": lambda r: bool(_ref_surname_tokens(r)),
    "journal": lambda r: bool(_get_ref_container(r)),
    "volume": lambda r: bool(_get_ref_volume(r)),
    "pages": lambda r: bool(_get_ref_pages(r)[0]),
}

# Reference metric -> (count prefix, denominator the macro value divides by).
# ref_*_acc is correct/matched_pairs; ref_doi_recall is correct/gold-refs-with-a-DOI.
REF_FIELD_METRIC_COUNTS = {
    "ref_title_acc": ("title", "matched_pairs"),
    "ref_year_acc": ("year", "matched_pairs"),
    "ref_author_acc": ("author", "matched_pairs"),
    "ref_journal_acc": ("journal", "matched_pairs"),
    "ref_volume_acc": ("volume", "matched_pairs"),
    "ref_pages_acc": ("pages", "matched_pairs"),
    "ref_doi_recall": ("doi", "gold_refs_with_field"),
}


def ref_field_counts(extracted_refs: list[dict], ground_truth_refs: list[dict]) -> dict[str, int]:
    """Per-paper reference denominators behind the ``ref_*`` accuracies.

    Returns ``gold_refs`` plus, per field, ``<field>_gold`` (gold references
    carrying it) and ``<field>_matched`` (matched pairs whose gold side carries
    it). These are what the macro means hide: an ``ref_pages_acc`` of 0.0 over
    a single matched reference weighs the same as one over fifty. Pooling them
    across the corpus gives the micro-averaged companions, and
    ``<field>_gold / gold_refs`` gives the coverage fraction the metric was
    actually computed over.
    """
    counts: dict[str, int] = {"gold_refs": len(ground_truth_refs)}
    for field, has_field in _REF_GOLD_FIELD_PREDICATES.items():
        counts[f"{field}_gold"] = sum(1 for r in ground_truth_refs if has_field(r))
        counts[f"{field}_matched"] = 0

    if not (extracted_refs and ground_truth_refs):
        return counts

    pairs = []
    for gi, gt_ref in enumerate(ground_truth_refs):
        for ei, ext_ref in enumerate(extracted_refs):
            sim = _ref_similarity(ext_ref, gt_ref)
            if sim > 0:
                pairs.append((sim, gi, ei))

    for gi, _ei in _greedy_match_pairs(pairs):
        gt_ref = ground_truth_refs[gi]
        for field, has_field in _REF_GOLD_FIELD_PREDICATES.items():
            if has_field(gt_ref):
                counts[f"{field}_matched"] += 1
    return counts


def score_paper(extracted: dict, ground_truth: dict) -> dict[str, object]:
    """Compute all metrics for one paper.

    Args:
        extracted: Dict from extract_comparable() or extract_comparable_from_json().
        ground_truth: Dict with matching keys from ground truth DataFrame row.

    Returns:
        Dict with metric names as keys and float scores as values. A score is
        None when the ground truth lacks the field (excluded from aggregates).
    """
    e_title = extracted.get("title") or ""
    g_title = ground_truth.get("title") or ""
    e_doi = extracted.get("doi") or ""
    # Gold loader provides doi_printed (page authority); the plain doi column
    # doubles as join key with a paper_id fallback. Legacy Crossref GT only
    # has doi.
    if "doi_printed" in ground_truth:
        g_doi = ground_truth.get("doi_printed") or ""
    else:
        g_doi = ground_truth.get("doi") or ""
    e_abstract = extracted.get("abstract") or ""
    g_abstract = ground_truth.get("abstract") or ""
    e_authors = extracted.get("authors") or []
    g_authors = ground_truth.get("authors") or []
    e_refs = extracted.get("references") or []
    g_refs = ground_truth.get("references") or []
    e_ref_count = extracted.get("reference_count") or 0
    g_ref_count = ground_truth.get("reference_count") or 0

    scores = {
        "title_soft": title_soft_match(e_title, g_title),
        "doi_match": doi_match(e_doi, g_doi),
        "abstract_rouge_l": abstract_rouge_l(e_abstract, g_abstract),
        "abstract_ned": abstract_ned(e_abstract, g_abstract),
        "authors_f1": authors_family_f1(e_authors, g_authors),
        "authors_fullname_f1": authors_fullname_f1(e_authors, g_authors),
        "first_author": first_author_match(e_authors, g_authors),
        "ref_count_ratio": references_count_ratio(e_ref_count, g_ref_count),
        "ref_matching_f1": ref_matching_f1(e_refs, g_refs),
    }
    # Diagnostic field metrics: None when gold lacks the signal (excluded from aggregates).
    scores["keywords_f1"] = keywords_f1(extracted.get("keywords"), ground_truth.get("keywords"))
    scores["affiliation_sim"] = affiliation_sim(e_authors, g_authors)
    scores["email_f1"] = email_f1(e_authors, g_authors)
    scores["orcid_f1"] = orcid_f1(e_authors, g_authors)
    scores["corresponding_acc"] = corresponding_acc(e_authors, g_authors)
    # Field-level reference scores on raw extraction (primary).
    scores.update(ref_field_scores(e_refs, g_refs))
    # Denominators behind those accuracies — never a metric itself, but what the
    # micro-averaged companions and the gold-coverage annotations pool over.
    scores["ref_field_counts"] = ref_field_counts(e_refs, g_refs)
    return scores


def evaluate(papers: list, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """Match papers to ground truth by DOI and score each.

    Args:
        papers: List of bibr.paper.Paper objects.
        ground_truth_df: DataFrame from load_ground_truth_gold() or
            load_ground_truth_crossref().

    Returns:
        DataFrame with one row per matched paper, columns for DOI + all metrics.
    """
    return _evaluate_extracted(
        [extract_comparable(p) for p in papers],
        ground_truth_df,
    )


def evaluate_json_exports(
    json_dir: Path,
    ground_truth_df: pd.DataFrame,
    ids: set[str] | None = None,
    report: dict | None = None,
) -> pd.DataFrame:
    """Load JSON exports from a directory and evaluate against ground truth.

    Args:
        json_dir: Directory containing bibr JSON export files.
        ground_truth_df: DataFrame from load_ground_truth_gold() or
            load_ground_truth_crossref().
        ids: Optional set of file stems to restrict scoring to (e.g. a fixed
            subsample so cells stay comparable on the same papers).
        report: Optional dict filled in with run bookkeeping the DataFrame
            cannot carry — ``prediction_ids`` (stems actually loaded) and
            ``unmatched_ids`` (predictions no ground-truth row claimed).

    Returns:
        DataFrame with one row per matched paper, columns for DOI + all metrics.
    """
    extracted_list = []
    prediction_ids = []
    for json_path in _scored_prediction_paths(json_dir, ids):
        try:
            with open(json_path) as f:
                data = json.load(f)
            extracted_list.append(extract_comparable_from_json(data))
            prediction_ids.append(json_path.stem)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load %s: %s", json_path.name, e)

    if report is not None:
        report["prediction_ids"] = prediction_ids

    return _evaluate_extracted(extracted_list, ground_truth_df, report=report)


def _evaluate_extracted(
    extracted_list: list[dict],
    ground_truth_df: pd.DataFrame,
    report: dict | None = None,
) -> pd.DataFrame:
    """Score a list of extracted dicts against ground truth.

    Matches by normalized DOI first, then falls back to file_name.
    """
    # Build lookup indices
    gt_by_doi: dict[str, dict] = {}
    gt_by_filename: dict[str, dict] = {}
    for _, row in ground_truth_df.iterrows():
        row_dict = row.to_dict()
        doi_key = normalize_doi(row["doi"])
        if doi_key:
            gt_by_doi[doi_key] = row_dict
        if row.get("file_name"):
            gt_by_filename[row["file_name"].lower().strip()] = row_dict

    results = []
    unmatched_ids: list[str] = []
    matched = 0
    for extracted in extracted_list:
        # Try DOI match first
        doi_key = normalize_doi(extracted.get("doi", ""))
        gt = gt_by_doi.get(doi_key) if doi_key else None

        # Fall back to filename match
        if gt is None:
            fname = (extracted.get("file_name") or "").lower().strip()
            gt = gt_by_filename.get(fname)

        if gt is None:
            logger.warning(
                "No ground truth for DOI=%s file=%s",
                extracted.get("doi", ""),
                extracted.get("file_name", ""),
            )
            # An unmatched prediction is silently dropped from every aggregate,
            # so the count has to survive into the artifact to be gateable.
            unmatched_ids.append(
                Path(extracted.get("file_name") or "").stem or extracted.get("doi") or ""
            )
            continue

        matched += 1
        scores = score_paper(extracted, gt)
        scores["doi"] = extracted.get("doi", gt.get("doi", ""))
        scores["file_name"] = extracted.get("file_name", gt.get("file_name", ""))
        scores["abstained"] = bool(extracted.get("abstained"))
        results.append(scores)

    logger.info("Matched %d / %d papers to ground truth", matched, len(extracted_list))

    if report is not None:
        report["unmatched_ids"] = sorted(unmatched_ids)

    if not results:
        return pd.DataFrame()

    df = _apply_abstentions(pd.DataFrame(results))
    id_cols = [c for c in ["doi", "file_name", "abstained"] if c in df.columns]
    metric_cols = [c for c in df.columns if c not in id_cols]
    return df[id_cols + metric_cols]


def _apply_abstentions(df: pd.DataFrame) -> pd.DataFrame:
    """Null the metrics a front-matter abstention suppressed.

    An abstention is a refusal to assert, not a wrong answer, so scoring it 0.0
    would understate quality *and* let the pipeline be penalised for its own
    safety guard. None is already the established "nothing to score here" value:
    the summary table drops it rather than averaging it as 0, and
    ``paper_passes_floors`` never fails a paper on a None metric. The abstention
    stays visible via the ``abstained`` column and the abstention rate, so
    abstaining more cannot quietly buy a better score.

    The pre-suppression value of every suppressed field is kept in a shadow
    ``<col>__preabstention`` column so ``save_results`` can also report the
    abstention-penalized companion mean. Report both means so a run cannot
    appear better merely by abstaining on difficult inputs.
    """
    if "abstained" not in df.columns:
        return df
    mask = df["abstained"].fillna(False).astype(bool)
    if not mask.any():
        return df
    for col in ABSTENTION_SUPPRESSED_COLS:
        if col in df.columns:
            df[f"{col}{_PREABSTENTION_SUFFIX}"] = df[col]
            df.loc[mask, col] = None
    return df


def append_missing_papers(results: pd.DataFrame, missing_ids: Iterable[str]) -> pd.DataFrame:
    """Add a scored row for every attempted paper that produced no prediction.

    A paper that crashes upstream writes no prediction file, so it is absent
    from the artifact entirely and the means are computed over the survivors.
    Omitting failed papers can inflate the measured quality of the run.

    Each missing paper is therefore scored 0.0 on every ``FLOORS`` metric (a
    paper that produced nothing recovered nothing) and left None everywhere
    else, and flagged ``missing`` so it is greppable and never confused with an
    abstention (a refusal to assert) or with a real 0.0.
    """
    df = results.copy()
    if "missing" in df.columns:
        df["missing"] = df["missing"].fillna(False).astype(bool)
    else:
        df["missing"] = False

    ids = sorted({str(i) for i in missing_ids})
    if not ids:
        return df

    rows = []
    for paper_id in ids:
        row: dict = {"file_name": paper_id, "missing": True}
        if "doi" in df.columns:
            row["doi"] = ""
        if "abstained" in df.columns:
            row["abstained"] = False
        row.update(dict.fromkeys(FLOORS, 0.0))
        rows.append(row)

    out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    out["missing"] = out["missing"].fillna(False).astype(bool)
    if "abstained" in out.columns:
        out["abstained"] = out["abstained"].fillna(False).astype(bool)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# Primary metrics: the headline quality signal shown in the summary table and
# gated by --threshold. Each measures bibr's extraction quality against the
# print-verbatim gold on a field the gold can judge fairly.
PRIMARY_METRIC_COLS = [
    "title_soft",
    "doi_match",
    "abstract_rouge_l",
    "authors_fullname_f1",
    "ref_matching_f1",
    "ref_title_acc",
    "ref_year_acc",
    "ref_doi_recall",
]

# Diagnostic metrics: still computed per-paper and emitted in the --output JSON
# (eval_guard + the published GROBID head-to-head table depend on several of
# these), printed under a clearly-labeled diagnostics section, but NEVER part
# of the --threshold gate or the pass_rate floors. They are redundant with a
# primary metric, tail/field-coverage views, or (formerly ref_doi_recall_enriched)
# would contradict the print-verbatim gold principle.
DIAGNOSTIC_METRIC_COLS = [
    "abstract_ned",
    "authors_f1",
    "first_author",
    "ref_count_ratio",
    "ref_author_acc",
    "ref_journal_acc",
    "ref_volume_acc",
    "ref_pages_acc",
    "keywords_f1",
    "affiliation_sim",
    "email_f1",
    "orcid_f1",
    "corresponding_acc",
]

METRIC_COLS = PRIMARY_METRIC_COLS + DIAGNOSTIC_METRIC_COLS


def _print_metric_row(results: pd.DataFrame, col: str) -> None:
    # None/NaN = gold lacks the field for that paper — excluded, not averaged as 0.
    series = results[col].dropna()
    if series.empty:
        print(f"{col:<25} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {0:>5}")
        return
    print(
        f"{col:<25} {series.mean():>8.3f} {series.quantile(0.10):>8.3f} {series.median():>8.3f}"
        f" {series.min():>8.3f} {series.max():>8.3f} {len(series):>5}"
    )


def _paper_rows(results: pd.DataFrame) -> list[dict]:
    """DataFrame → per-paper dicts with NaN normalised to None.

    Shadow ``__preabstention`` columns are internal plumbing and stay out of
    the artifact.
    """
    cols = [c for c in results.columns if not c.endswith(_PREABSTENTION_SUFFIX)]
    frame = results[cols]
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def _split_abstained(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition per-paper rows into (scored, abstained)."""
    abstained = [r for r in rows if r.get("abstained")]
    scored = [r for r in rows if not r.get("abstained")]
    return scored, abstained


def _pass_rate_full_cohort(rows: list[dict], floors: dict[str, float] = FLOORS) -> float | None:
    """Pass rate over every attempted paper, with an abstention scored as a fail.

    ``pass_rate`` alone cannot express this: ``paper_passes_floors`` ignores None
    metrics by design, and ``_apply_abstentions`` nulls two of the four ``FLOORS``
    metrics (title_soft, authors_fullname_f1) while leaving the other two —
    **doi_match and ref_matching_f1**, only one of which is a ref_* metric —
    live. So an abstaining row handed to ``pass_rate`` is judged on doi_match and
    ref_matching_f1 alone and passes vacuously whenever those two are absent or
    already good. The previous fix was to drop those rows from the denominator
    instead, which merely hid the same papers.

    Neither is right. A paper that ships a byte-empty PaperMetadata is a failure
    the reviewer sees, so it counts, and it counts as a failure. Missing papers
    (crashed, no prediction) need no special case — append_missing_papers already
    scores them 0.0 on every floor.

    """
    if not rows:
        return None
    passing = sum(
        1 for row in rows if not row.get("abstained") and paper_passes_floors(row, floors)
    )
    return passing / len(rows)


def print_summary(results: pd.DataFrame) -> None:
    """Print aggregate metrics summary.

    Alongside the usual mean/median, prints ``p10`` (10th percentile) per
    metric and a paper-level ``pass_rate`` (fraction of papers whose primary
    metrics all clear ``validation_metrics.FLOORS``) — the mean is saturated
    near 1.0 and can't surface a handful of papers
    that regressed badly; p10/pass_rate are tail-sensitive.
    """
    if results.empty:
        print("No results to summarize.")
        return

    print(f"\nEvaluation Summary ({len(results)} papers)")
    print("=" * 75)
    print(f"{'Metric':<25} {'Mean':>8} {'p10':>8} {'Median':>8} {'Min':>8} {'Max':>8} {'N':>5}")
    print("-" * 75)

    for col in PRIMARY_METRIC_COLS:
        if col in results.columns:
            _print_metric_row(results, col)

    paper_rows = _paper_rows(results)
    scored, abstained = _split_abstained(paper_rows)
    pr = _pass_rate_full_cohort(paper_rows, FLOORS)
    print("-" * 75)
    print(
        f"paper-level pass_rate (floors: {', '.join(f'{k}>={v}' for k, v in FLOORS.items())})"
        f" = {'—' if pr is None else f'{pr:.3f}'}"
        f"  (n={len(paper_rows)}, abstentions count as failures)"
    )
    missing = [r for r in paper_rows if r.get("missing")]
    if missing:
        print(
            f"missing (attempted, no prediction — scored 0.0 on every floor): "
            f"{len(missing)} / {len(paper_rows)}"
        )
    if abstained:
        real = [r for r in paper_rows if not r.get("missing")] or paper_rows
        excl = pass_rate(scored, FLOORS)
        print(
            f"abstained (front-matter, scored as failures): {len(abstained)}"
            f" / {len(real)} = {len(abstained) / len(real):.3f}"
        )
        # Printed only so a pre-v4 artifact can be recognised, never as the
        # headline: this is the survivors-only number, and it rises every time
        # the pipeline abstains on a paper it would have failed.
        print(
            f"pass_rate_excl_abstained (pre-v4 definition, NOT comparable) ="
            f" {'—' if excl is None else f'{excl:.3f}'}  (n={len(scored)})"
        )
        print(
            "per-metric means above still exclude abstained papers; the penalized"
            " companion is <metric>.mean_incl_abstained in the --output JSON"
        )

    diagnostic_present = [c for c in DIAGNOSTIC_METRIC_COLS if c in results.columns]
    if diagnostic_present:
        print("-" * 75)
        print("Diagnostic (not extraction quality — excluded from pass_rate; see code)")
        for col in diagnostic_present:
            _print_metric_row(results, col)

    print("=" * 75)


def _abstention_penalized_mean(results: pd.DataFrame, col: str) -> float | None:
    """Mean of ``col`` with abstained papers restored to their raw value.

    None when nothing was suppressed (the companion would just repeat ``mean``).
    """
    shadow = f"{col}{_PREABSTENTION_SUFFIX}"
    if shadow not in results.columns:
        return None
    series = results[col].where(results[col].notna(), results[shadow]).dropna()
    return round(float(series.mean()), 4) if len(series) else None


def _ref_coverage_annotations(results: pd.DataFrame, col: str) -> dict:
    """Micro-averaged companion + gold coverage for a reference metric.

    Both answer questions the macro mean cannot. ``gold_field_coverage`` is the
    fraction of transcribed gold references that actually carry the field: gold
    prints reference DOIs only when the page does (~18%) and its page-range
    coverage is a known transcription gap (~44%), so neither metric may be read
    as full coverage. ``micro_*`` pools the per-paper numerators/denominators
    instead of averaging ratios, so a paper with three matched references stops
    weighing as much as one with fifty.

    The numerator is reconstructed as ``round(macro_value * denominator)`` from
    the value ``ref_field_scores`` already returned, so the micro companion can
    never disagree with the macro it accompanies.
    """
    if "ref_field_counts" not in results.columns or col not in results.columns:
        return {}
    field, denom_kind = REF_FIELD_METRIC_COUNTS[col]
    denom_key = f"{field}_gold" if denom_kind == "gold_refs_with_field" else f"{field}_matched"

    correct = denominator = gold_with_field = gold_refs = 0
    for value, counts in zip(results[col], results["ref_field_counts"], strict=False):
        if not isinstance(counts, dict):
            continue
        gold_refs += int(counts.get("gold_refs", 0))
        gold_with_field += int(counts.get(f"{field}_gold", 0))
        if value is None or pd.isna(value):
            continue
        n = int(counts.get(denom_key, 0))
        if n:
            correct += round(float(value) * n)
            denominator += n

    return {
        "micro_mean": round(correct / denominator, 4) if denominator else None,
        "micro_correct": correct,
        "micro_denominator": denominator,
        "micro_denominator_kind": denom_kind,
        "gold_field_coverage": round(gold_with_field / gold_refs, 4) if gold_refs else None,
    }


def save_results(
    results: pd.DataFrame,
    output_path: Path,
    *,
    predictions_dir: Path | None = None,
    ids: set[str] | None = None,
    unmatched_ids: Iterable[str] | None = None,
) -> None:
    """Save evaluation results to JSON for CI regression tracking.

    The artifact is self-describing: it records which bibr checkout scored it,
    which prediction files it scored (by content digest), and which version of
    the metric definitions produced the numbers, so a re-score of an older
    snapshot can never silently pass for a fresh run.
    """
    if results.empty:
        return

    summary = {}
    for col in METRIC_COLS:
        if col in results.columns:
            series = results[col].dropna()
            block = {
                "mean": round(float(series.mean()), 4) if len(series) else None,
                "p10": round(float(series.quantile(0.10)), 4) if len(series) else None,
                "median": round(float(series.median()), 4) if len(series) else None,
                "min": round(float(series.min()), 4) if len(series) else None,
                "max": round(float(series.max()), 4) if len(series) else None,
                "n": int(len(series)),
            }
            if col in ABSTENTION_SUPPRESSED_COLS:
                penalized = _abstention_penalized_mean(results, col)
                if penalized is not None:
                    block["mean_incl_abstained"] = penalized
            if col in REF_FIELD_METRIC_COUNTS:
                block.update(_ref_coverage_annotations(results, col))
            summary[col] = block

    # NaN → None so the JSON carries null (json.dump would emit bare NaN).
    per_paper = _paper_rows(results)
    scored, abstained = _split_abstained(per_paper)
    missing = [r for r in per_paper if r.get("missing")]
    # Denominator for the abstention rate stays the papers that actually ran, so
    # adding --expected-ids cannot dilute (and loosen) the abstention gate.
    evaluated = [r for r in per_paper if not r.get("missing")]
    unmatched = sorted(unmatched_ids or [])

    output = {
        # --- provenance: what produced this artifact ---
        "metrics_version": METRICS_VERSION,
        "generated_at": (
            datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        ),
        "bibr_commit": bibr_commit(),
        "predictions_dir": str(predictions_dir) if predictions_dir is not None else None,
        "predictions_tree_sha256": (
            predictions_tree_sha256(predictions_dir, ids) if predictions_dir is not None else None
        ),
        # --- denominators: what was attempted vs what survived ---
        # papers_attempted counts every row including attempted-but-missing ones;
        # papers_evaluated keeps its original meaning (papers that produced a
        # scored prediction), so `papers_attempted - papers_evaluated > 0` is the
        # attrition signal a min_papers floor alone cannot give.
        "papers_attempted": len(per_paper),
        "papers_evaluated": len(evaluated),
        "papers_missing": len(missing),
        "missing_ids": sorted(r.get("file_name") or r.get("doi") or "" for r in missing),
        # Predictions no ground-truth row claimed: dropped from every aggregate,
        # so the gate should require this to be 0.
        "papers_unmatched": len(unmatched),
        "unmatched_ids": unmatched,
        "metrics": summary,
        # Paper-level pass_rate: fraction of papers whose primary metrics all
        # clear validation_metrics.FLOORS. Tail-sensitive complement to the
        # (saturated) per-metric means above. Every attempted paper is in the
        # denominator — a crash scores 0.0 on every floor, and since v4 a
        # front-matter abstention counts as a failure too (an empty record is a
        # failure to its consumer whatever the pipeline's reason was).
        "pass_rate": _pass_rate_full_cohort(per_paper, FLOORS),
        # The full denominator includes missing predictions and abstentions;
        # papers_evaluated and papers_scored describe smaller subsets.
        "pass_rate_n": len(per_paper),
        # The pre-v4 survivor-only definition is retained under an explicit
        # name for compatibility. Its denominator is papers_scored.
        "pass_rate_excl_abstained": pass_rate(scored, FLOORS),
        "pass_rate_excl_abstained_n": len(scored),
        "papers_scored": len(scored),
        # `abstained` is the historical key; `n_abstained` is its self-describing
        # name. Both are emitted so a consumer written against either keeps working.
        "abstained": len(abstained),
        "n_abstained": len(abstained),
        "abstention_rate": round(len(abstained) / len(evaluated), 4) if evaluated else None,
        "abstained_ids": sorted(r.get("file_name") or r.get("doi") or "" for r in abstained),
        "per_paper": per_paper,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate bibr extraction accuracy")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory of pre-computed JSON exports to evaluate",
    )
    parser.add_argument(
        "--gold-dirs",
        "--gold-dir",
        type=Path,
        nargs="+",
        help=(
            "Directories of independently prepared gold in bibr v10 JSON format. "
            "Required for metadata scoring."
        ),
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        help=(
            "Optional file of result stems (one per line, '#' comments ok) to "
            "restrict scoring to a fixed subsample so cells stay comparable."
        ),
    )
    parser.add_argument(
        "--expected-ids",
        help=(
            "Papers the run was supposed to cover: either a JSON file of ids "
            "(list, or an object with a 'members' list) or a named set_id from an "
            "explicit --evaluation-sets registry. Every expected id with no prediction is "
            "scored 0.0 on each pass_rate floor metric instead of vanishing from "
            "the denominator."
        ),
    )
    parser.add_argument(
        "--evaluation-sets",
        type=Path,
        help="Optional JSON registry containing a 'sets' list for named --expected-ids.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save evaluation results JSON (for CI regression tracking)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Fail if any metric mean drops below this value (CI quality gate)",
    )
    parser.add_argument(
        "--sections",
        action="store_true",
        help="Run the section-text recall benchmark (vs *.sectiongold.json artifacts).",
    )
    parser.add_argument(
        "--section-gold-dirs",
        type=Path,
        nargs="+",
        help="Directories containing *.sectiongold.json artifacts; required with --sections.",
    )
    parser.add_argument(
        "--drop-report-top",
        type=int,
        default=40,
        help="How many worst (paper, section) rows to print from the drop report.",
    )
    args = parser.parse_args()
    if args.sections and not args.section_gold_dirs:
        parser.error("--sections requires --section-gold-dirs with your gold artifacts")
    if not args.sections and not args.gold_dirs:
        parser.error("metadata scoring requires --gold-dirs with your gold exports")

    ids = None
    if args.ids_file:
        ids = {
            ln.strip()
            for ln in args.ids_file.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        }
        print(f"Restricting scoring to {len(ids)} subsample ids from {args.ids_file}")

    if args.sections:
        section_gold_dirs = args.section_gold_dirs
        print(
            f"\nSection benchmark: predictions={args.results_dir}, "
            f"artifacts={[str(d) for d in section_gold_dirs]}"
        )
        per_type, drop_rows, coverage = evaluate_sections(
            args.results_dir, section_gold_dirs, ids=ids
        )
        if not per_type:
            print("No section scores produced (no artifact↔prediction matches).")
            raise SystemExit(1)
        print_section_summary(per_type, coverage)
        print(f"\nWorst {min(args.drop_report_top, len(drop_rows))} sections (drop report):")
        print(
            f"{'paper_id':<22}{'type':<14}{'ref_tok':>8}{'uni_rec':>9}{'shingle':>9}{'pages':>14}"
        )
        for r in drop_rows[: args.drop_report_top]:
            sh = f"{r['shingle_recall']:.3f}" if r["shingle_recall"] is not None else "—"
            print(
                f"{r['paper_id']:<22}{r['section_type']:<14}{r['ref_tokens']:>8}"
                f"{r['unigram_recall']:>9.3f}{sh:>9}{str(r['pages']):>14}"
            )
        if args.output:
            save_section_results(per_type, drop_rows, coverage, args.output)
            print(f"\nSection results saved to {args.output}")
        raise SystemExit(0)

    print(f"Loading gold ground truth from {len(args.gold_dirs)} dir(s)...")
    gt_df = load_ground_truth_gold(args.gold_dirs)
    print(f"Ground truth: {len(gt_df)} papers")

    print(f"\nEvaluating JSON exports from: {args.results_dir}")
    run_report: dict = {}
    results = evaluate_json_exports(args.results_dir, gt_df, ids=ids, report=run_report)

    if results.empty:
        print("No papers matched ground truth!")
        raise SystemExit(1)

    if args.expected_ids:
        expected = load_expected_ids(args.expected_ids, registry_path=args.evaluation_sets)
        if ids is not None:
            expected &= ids
        absent = sorted(expected - set(run_report.get("prediction_ids") or []))
        print(
            f"Expected {len(expected)} paper(s) from {args.expected_ids}; "
            f"{len(absent)} produced no prediction"
        )
        if absent:
            print("  scored 0.0 on every pass_rate floor metric: " + ", ".join(absent))
        results = append_missing_papers(results, absent)

    unmatched = run_report.get("unmatched_ids") or []
    if unmatched:
        print(
            f"\n{len(unmatched)} prediction(s) matched no ground-truth row: {', '.join(unmatched)}"
        )

    print_summary(results)

    if args.output:
        save_results(
            results,
            args.output,
            predictions_dir=args.results_dir,
            ids=ids,
            unmatched_ids=unmatched,
        )
        print(f"\nResults saved to {args.output}")

    # CI quality gate — primary metrics only; diagnostics never gate.
    if args.threshold is not None:
        failed = []
        for col in PRIMARY_METRIC_COLS:
            if col in results.columns:
                mean = results[col].mean()
                if mean < args.threshold:
                    failed.append(f"{col}: {mean:.3f} < {args.threshold:.3f}")
        if failed:
            print(f"\nQuality gate FAILED (threshold={args.threshold}):")
            for f in failed:
                print(f"  - {f}")
            raise SystemExit(1)
        else:
            print(f"\nQuality gate PASSED (all metrics >= {args.threshold})")
