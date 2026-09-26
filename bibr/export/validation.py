"""Output validation gate for the assembled export dict.

A quality audit found that catastrophic output defects (placeholder tokens,
exploded author lists, empty equations, dangling references, …) passed
silently because the processing warnings only ever carried
reference-segmentation events. This module runs a battery of cheap, defensive
checks over the finished export dict (schema v12.0 shape) and returns a list
of :class:`ValidationIssue`; the export wiring surfaces them in the structured
top-level ``validation`` block, and there only — findings are never mirrored
into ``extraction.warnings``.

Every check is defensive: it must never raise on malformed input, degrading to
skipping itself instead. :func:`validate_export` wraps each check so one broken
check cannot suppress the others (it emits a ``VAL_INTERNAL`` warning instead).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from urllib.parse import urlsplit

from bibr.utils.metadata import is_exact_generic_article_label
from bibr.validation import (
    IssueSeverity,
    ValidationIssue,
    abstract_suspicion_reasons,
    payload_validation,
)

log = logging.getLogger(__name__)

# NuExtract3 template-DSL type tokens occasionally echoed back as field values.
# Import the canonical frozenset; duplicate defensively if the import breaks.
try:
    from bibr.schemas import _PLACEHOLDER_TOKENS as _SENTINEL_TOKENS
except Exception:  # pragma: no cover - defensive fallback
    _SENTINEL_TOKENS = frozenset(
        {"verbatim-string", "string", "integer", "number", "boolean", "date-time", "date"}
    )

_GENERIC_TITLES = frozenset({"phd dissertation", "dissertation", "untitled", "article", "thesis"})

# Bare panel marker, e.g. "(a)" or "b)".
_PANEL_RE = re.compile(r"^\(?[a-z]\)$")
# Appendix-like headers: a single leading capital + separator, or "Appendix".
_APPENDIX_LETTER_RE = re.compile(r"^([A-Z])[\s.:]")
_APPENDIX_WORD_RE = re.compile(r"^Appendix", re.IGNORECASE)

# Statement anchors → the metadata field that should carry the parsed statement.
_STATEMENT_ANCHORS: tuple[tuple[str, str], ...] = (
    ("conflict of interest", "coi_statement"),
    ("ethical approval", "ethics_statement"),
    ("informed consent", "ethics_statement"),
    ("this work was supported by", "funding_statement"),
    ("data are available", "data_availability"),
)


# ── small defensive helpers ────────────────────────────────────────────


def _as_list(payload: dict, key: str) -> list:
    v = payload.get(key)
    return v if isinstance(v, list) else []


def _as_dict(payload: dict, key: str) -> dict:
    v = payload.get(key)
    return v if isinstance(v, dict) else {}


def _text(v: object) -> str:
    return v.strip() if isinstance(v, str) else ""


def _positive_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def _string_leaves(obj: object):
    """Yield every string scalar reachable in a nested dict/list structure."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _string_leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _string_leaves(v)


# ── ERROR-severity checks ──────────────────────────────────────────────


def _check_placeholder(payload: dict) -> list[ValidationIssue]:
    hits = 0
    for key in ("metadata", "funding", "affiliation", "author"):
        for s in _string_leaves(payload.get(key)):
            if s.strip().casefold() in _SENTINEL_TOKENS:
                hits += 1
    if hits:
        return [
            ValidationIssue(
                "VAL_PLACEHOLDER",
                IssueSeverity.ERROR,
                f"{hits} placeholder/sentinel token(s) leaked into metadata fields",
                count=hits,
            )
        ]
    return []


def _check_author_blank(payload: dict) -> list[ValidationIssue]:
    blank = sum(
        1
        for a in _as_list(payload, "author")
        if isinstance(a, dict)
        and not _text(a.get("given"))
        and not _text(a.get("family"))
        # A group author's whole name is ``literal``.
        and not _text(a.get("literal"))
    )
    if blank:
        return [
            ValidationIssue(
                "VAL_AUTHOR_BLANK",
                IssueSeverity.ERROR,
                f"{blank} author(s) with empty given and family name",
                count=blank,
            )
        ]
    return []


def _check_author_outlier(payload: dict) -> list[ValidationIssue]:
    authors = [a for a in _as_list(payload, "author") if isinstance(a, dict)]
    n = len(authors)
    if n > 64:
        return [
            ValidationIssue(
                "VAL_AUTHOR_OUTLIER",
                IssueSeverity.ERROR,
                f"{n} authors exceeds the plausible maximum (64)",
                count=n,
            )
        ]
    pairs = Counter(
        (
            _text(a.get("given")).casefold(),
            _text(a.get("family")).casefold(),
            _text(a.get("literal")).casefold(),
        )
        for a in authors
    )
    dupes = {p: c for p, c in pairs.items() if c > 2 and any(p)}
    if dupes:
        worst = max(dupes.values())
        return [
            ValidationIssue(
                "VAL_AUTHOR_OUTLIER",
                IssueSeverity.ERROR,
                f"{len(dupes)} author name(s) repeated more than twice (max {worst}x)",
                count=worst,
            )
        ]
    return []


def _check_bbox_space(payload: dict) -> list[ValidationIssue]:
    sample = []
    # Every box in ``extraction`` is in points on its page; ``extraction.pages``
    # gives the page sizes they must fall within.
    extraction = _as_dict(payload, "extraction")
    sizes = {
        p.get("page_number"): (p.get("width"), p.get("height"))
        for p in _as_list(extraction, "pages")
        if isinstance(p, dict)
    }
    for key in ("text_regions", "float_parts"):
        for t in _as_list(extraction, key):
            if not isinstance(t, dict):
                continue
            b = t.get("bbox")
            pw, ph = sizes.get(t.get("page_number"), (None, None))
            if (
                isinstance(b, (list, tuple))
                and len(b) == 4
                and _positive_number(pw)
                and _positive_number(ph)
            ):
                sample.append((b, float(pw), float(ph)))
                if len(sample) >= 500:
                    break
    if not sample:
        return []
    viol = 0
    for b, pw, ph in sample:
        try:
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        except (TypeError, ValueError):
            continue
        if not (0 <= x1 <= x2 <= pw) or not (0 <= y1 <= y2 <= ph):
            viol += 1
    if viol > 0.1 * len(sample):
        return [
            ValidationIssue(
                "VAL_BBOX_SPACE",
                IssueSeverity.ERROR,
                f"{viol}/{len(sample)} sampled bboxes violate page-coordinate bounds",
                count=viol,
            )
        ]
    return []


def _check_dangling_ref(payload: dict) -> list[ValidationIssue]:
    sections = _as_list(payload, "section")
    section_ids = {s.get("section_id") for s in sections if isinstance(s, dict)}
    texts = [t for t in _as_list(payload, "text") if isinstance(t, dict)]
    text_ids = {t.get("text_id") for t in texts}
    xrefs = [x for x in _as_list(payload, "xref") if isinstance(x, dict)]
    xref_ids = {x.get("xref_id") for x in xrefs}
    author_ids = {a.get("author_id") for a in _as_list(payload, "author") if isinstance(a, dict)}
    affiliation_ids = {
        a.get("affiliation_id") for a in _as_list(payload, "affiliation") if isinstance(a, dict)
    }
    funding_ids = {f.get("funding_id") for f in _as_list(payload, "funding") if isinstance(f, dict)}
    figure_ids = {f.get("figure_id") for f in _as_list(payload, "figure") if isinstance(f, dict)}
    table_ids = {t.get("table_id") for t in _as_list(payload, "table") if isinstance(t, dict)}
    targets = {
        "bib": {b.get("bib_id") for b in _as_list(payload, "bib") if isinstance(b, dict)},
        "figure": figure_ids,
        "table": table_ids,
        "foot": {
            f.get("footnote_id") for f in _as_list(payload, "footnote") if isinstance(f, dict)
        },
    }
    dangling = 0
    for x in xrefs:
        pool = targets.get(x.get("xref_type"))
        xid = x.get("target_id")
        if pool is not None and xid is not None and xid not in pool:
            dangling += 1
        tid = x.get("text_id")
        if tid is not None and tid not in text_ids:
            dangling += 1
    for s in sections:
        if not isinstance(s, dict):
            continue
        p = s.get("parent_section_id")
        if p is not None and p not in section_ids:
            dangling += 1
    for t in texts:
        sid = t.get("section_id")
        if sid is not None and sid not in section_ids:
            dangling += 1
    # Rows that point at their printed text: reference entries, captions,
    # footnotes, links and expressions.
    for key in ("bib", "figure", "table", "footnote", "url", "eq"):
        for row in _as_list(payload, key):
            if not isinstance(row, dict):
                continue
            tid = row.get("text_id")
            if tid is not None and tid not in text_ids:
                dangling += 1
    for key in ("figure", "table"):
        for row in _as_list(payload, key):
            if not isinstance(row, dict):
                continue
            sid = row.get("section_id")
            if sid is not None and sid not in section_ids:
                dangling += 1
    for row in _as_list(payload, "affiliation"):
        if not isinstance(row, dict):
            continue
        aids = row.get("author_ids")
        if isinstance(aids, list):
            dangling += sum(1 for aid in aids if aid not in author_ids)
    for row in _as_list(payload, "affiliation_match"):
        if not isinstance(row, dict):
            continue
        aid = row.get("affiliation_id")
        if aid is not None and aid not in affiliation_ids:
            dangling += 1
    for row in _as_list(payload, "funding_match"):
        if not isinstance(row, dict):
            continue
        fid = row.get("funding_id")
        if fid is not None and fid not in funding_ids:
            dangling += 1
    for row in _as_list(payload, "bib_match"):
        if not isinstance(row, dict):
            continue
        bid = row.get("bib_id")
        if bid is not None and bid not in targets["bib"]:
            dangling += 1
    extraction = _as_dict(payload, "extraction")
    for row in _as_list(extraction, "text_regions"):
        if not isinstance(row, dict):
            continue
        tid = row.get("text_id")
        if tid is not None and tid not in text_ids:
            dangling += 1
    for row in _as_list(extraction, "float_parts"):
        if not isinstance(row, dict):
            continue
        pool = {"figure": figure_ids, "table": table_ids}.get(row.get("object_type"))
        oid = row.get("object_id")
        if pool is not None and oid is not None and oid not in pool:
            dangling += 1
    diagnostics = extraction.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    for row in diagnostics.get("section_classification") or []:
        if not isinstance(row, dict):
            continue
        sid = row.get("section_id")
        if sid is not None and sid not in section_ids:
            dangling += 1
    for row in diagnostics.get("xref_tier") or []:
        if not isinstance(row, dict):
            continue
        xid = row.get("xref_id")
        if xid is not None and xid not in xref_ids:
            dangling += 1
    if dangling:
        return [
            ValidationIssue(
                "VAL_DANGLING_REF",
                IssueSeverity.ERROR,
                f"{dangling} reference id(s) point to nonexistent records",
                count=dangling,
            )
        ]
    return []


def _check_duplicate_pk(payload: dict) -> list[ValidationIssue]:
    dup_tables = 0
    dup_rows = 0
    for table, key in (
        ("author", "author_id"),
        ("affiliation", "affiliation_id"),
        ("funding", "funding_id"),
        ("text", "text_id"),
        ("section", "section_id"),
        ("bib", "bib_id"),
        ("figure", "figure_id"),
        ("table", "table_id"),
        ("footnote", "footnote_id"),
        ("url", "url_id"),
        ("xref", "xref_id"),
        ("eq", "eq_id"),
    ):
        seen: set[object] = set()
        dups = 0
        for row in _as_list(payload, table):
            if not isinstance(row, dict):
                continue
            value = row.get(key)
            if value is None or value in seen:
                if value is not None:
                    dups += 1
                continue
            seen.add(value)
        if dups:
            dup_tables += 1
            dup_rows += dups
    if dup_rows:
        return [
            ValidationIssue(
                "VAL_DUPLICATE_PK",
                IssueSeverity.ERROR,
                f"{dup_rows} duplicate primary key(s) across {dup_tables} table(s)",
                count=dup_rows,
            )
        ]
    return []


def _check_empty_eq(payload: dict) -> list[ValidationIssue]:
    # Every exported row carries a non-empty comparator (unknown comparators
    # are dropped at export), so an all-blank test never fires on real
    # exports — while a null lhs coerced to '' by the LLM schema ships
    # silently. Flag a blank side instead: an equation with no left- or
    # right-hand side has no usable content.
    empty = sum(
        1
        for e in _as_list(payload, "eq")
        if isinstance(e, dict) and (not _text(e.get("lhs")) or not _text(e.get("rhs")))
    )
    if empty:
        return [
            ValidationIssue(
                "VAL_EMPTY_EQ",
                IssueSeverity.ERROR,
                f"{empty} equation record(s) with empty lhs or rhs",
                count=empty,
            )
        ]
    return []


def _check_url_malformed(payload: dict) -> list[ValidationIssue]:
    bad = 0
    for u in _as_list(payload, "url"):
        if not isinstance(u, dict):
            continue
        href = u.get("href")
        if not isinstance(href, str):
            continue
        try:
            parts = urlsplit(href)
        except ValueError:
            continue
        if parts.scheme in ("http", "https") and "." not in parts.netloc:
            bad += 1
    if bad:
        return [
            ValidationIssue(
                "VAL_URL_MALFORMED",
                IssueSeverity.ERROR,
                f"{bad} URL(s) with an http(s) scheme but no dotted host",
                count=bad,
            )
        ]
    return []


# ── WARNING-severity checks ────────────────────────────────────────────


def _check_title_generic(payload: dict) -> list[ValidationIssue]:
    title = _as_dict(payload, "metadata").get("title")
    t = _text(title)
    if not t or t.casefold() in _GENERIC_TITLES or is_exact_generic_article_label(t):
        return [
            ValidationIssue(
                "VAL_TITLE_GENERIC",
                "warning",
                f"generic or empty title: {title!r}",
            )
        ]
    return []


_CANONICAL_METADATA_UNICODE_FIELDS = (
    "title",
    "doi",
    "journal",
    "volume",
    "issue",
    "pages",
    "first_page",
    "last_page",
    "issn",
    "publisher",
    "published",
)
_CANONICAL_AUTHOR_UNICODE_FIELDS = ("given", "family", "literal", "orcid")


def _private_use_count(value: object) -> int:
    if not isinstance(value, str):
        return 0
    return sum(unicodedata.category(ch) == "Co" for ch in value)


def _check_unicode_canonical(payload: dict) -> list[ValidationIssue]:
    """Warn only when final identity fields retain private-use glyphs.

    Body/reference text and all alternate-source diagnostics are deliberately
    outside this gate: they are evidence, not canonical bibliographic identity.
    """
    metadata = _as_dict(payload, "metadata")
    hits = sum(
        _private_use_count(metadata.get(field)) for field in _CANONICAL_METADATA_UNICODE_FIELDS
    )
    for author in _as_list(payload, "author"):
        if not isinstance(author, dict):
            continue
        hits += sum(
            _private_use_count(author.get(field)) for field in _CANONICAL_AUTHOR_UNICODE_FIELDS
        )
    if not hits:
        return []
    return [
        ValidationIssue(
            "VAL_UNICODE_CANONICAL",
            IssueSeverity.WARNING,
            f"{hits} private-use Unicode character(s) remain in canonical identity fields",
            count=hits,
            blocking=False,
        )
    ]


def _check_abstract_missing(payload: dict) -> list[ValidationIssue]:
    abstract_ids = {
        s.get("section_id")
        for s in _as_list(payload, "section")
        if isinstance(s, dict) and s.get("section_type") == "abstract"
    }
    if not abstract_ids:
        return []
    n = sum(
        1
        for t in _as_list(payload, "text")
        if isinstance(t, dict) and t.get("section_id") in abstract_ids
    )
    if n >= 2 and not _text(_as_dict(payload, "metadata").get("abstract")):
        return [
            ValidationIssue(
                "VAL_ABSTRACT_MISSING",
                "warning",
                "abstract section has content but metadata.abstract is empty",
            )
        ]
    return []


def _check_abstract_suspect(payload: dict) -> list[ValidationIssue]:
    # The gate block lives at extraction.validation on schema 12 and at the
    # root on older exports (both still circulate); read whichever is present.
    block = payload_validation(payload) or {}
    existing_issues = block.get("issues") if isinstance(block, dict) else []
    if isinstance(existing_issues, list) and any(
        isinstance(issue, dict) and issue.get("code") == "VAL_ABSTRACT_SUSPECT"
        for issue in existing_issues
    ):
        return []

    abstract = _text(_as_dict(payload, "metadata").get("abstract"))
    if not abstract:
        return []

    sections = [section for section in _as_list(payload, "section") if isinstance(section, dict)]
    abstract_ids = {
        section.get("section_id")
        for section in sections
        if section.get("section_type") == "abstract"
    }
    reference_ids = {
        section.get("section_id")
        for section in sections
        if section.get("section_type") in {"references", "endnote"}
    }
    text_rows = [row for row in _as_list(payload, "text") if isinstance(row, dict)]
    abstract_source = " ".join(
        _text(row.get("text")) for row in text_rows if row.get("section_id") in abstract_ids
    ).strip()
    non_reference_prose = " ".join(
        _text(row.get("text")) for row in text_rows if row.get("section_id") not in reference_ids
    ).strip()

    body_rows = [
        row for row in text_rows if row.get("section_id") not in abstract_ids | reference_ids
    ]
    reasons = abstract_suspicion_reasons(
        abstract,
        source_text=abstract_source if abstract_ids else None,
        outside_texts=(_text(row.get("text")) for row in body_rows),
        non_reference_prose=non_reference_prose,
    )
    if not reasons:
        return []
    evidence_ids = tuple(
        f"text:{row.get('text_id')}"
        for row in text_rows
        if row.get("section_id") in abstract_ids and row.get("text_id") is not None
    )[:20]
    return [
        ValidationIssue(
            "VAL_ABSTRACT_SUSPECT",
            IssueSeverity.WARNING,
            f"abstract suspicion: {', '.join(reasons)}",
            evidence_ids=evidence_ids,
            blocking=False,
        )
    ]


def _check_appendix_nesting(payload: dict) -> list[ValidationIssue]:
    sections = [s for s in _as_list(payload, "section") if isinstance(s, dict)]
    by_id = {s.get("section_id"): s for s in sections}
    ref_ids = {s.get("section_id") for s in sections if s.get("section_type") == "references"}

    def _is_appendix(header: object) -> bool:
        h = header if isinstance(header, str) else ""
        return bool(_APPENDIX_LETTER_RE.match(h) or _APPENDIX_WORD_RE.match(h))

    bad = 0
    for s in sections:
        if not _is_appendix(s.get("header")):
            continue
        p = s.get("parent_section_id")
        if p is None:
            continue
        if p in ref_ids:
            bad += 1
            continue
        parent = by_id.get(p)
        if parent is not None and _is_appendix(parent.get("header")):
            bad += 1
    if bad:
        return [
            ValidationIssue(
                "VAL_APPENDIX_NESTING",
                IssueSeverity.WARNING,
                f"{bad} appendix section(s) nested under another appendix or references",
                count=bad,
            )
        ]
    return []


def _check_caption_missing(payload: dict) -> list[ValidationIssue]:
    tables = [t for t in _as_list(payload, "table") if isinstance(t, dict)]
    if not tables:
        return []
    missing = sum(1 for t in tables if not _text(t.get("caption")))
    if missing / len(tables) > 0.5:
        return [
            ValidationIssue(
                "VAL_CAPTION_MISSING",
                IssueSeverity.WARNING,
                f"{missing}/{len(tables)} table(s) are missing a caption",
                count=missing,
            )
        ]
    return []


def _check_panel_caption(payload: dict) -> list[ValidationIssue]:
    n = sum(
        1
        for f in _as_list(payload, "figure")
        if isinstance(f, dict)
        and isinstance(f.get("caption"), str)
        and _PANEL_RE.match(f["caption"].strip())
    )
    if n:
        return [
            ValidationIssue(
                "VAL_PANEL_CAPTION",
                IssueSeverity.WARNING,
                f"{n} figure caption(s) are bare panel markers, e.g. '(a)'",
                count=n,
            )
        ]
    return []


def _check_xref_zero(payload: dict) -> list[ValidationIssue]:
    bib = _as_list(payload, "bib")
    if len(bib) < 10:
        return []
    valid_bib_ids = {
        row.get("bib_id") for row in bib if isinstance(row, dict) and row.get("bib_id") is not None
    }
    has_bib_target = any(
        isinstance(x, dict) and x.get("xref_type") == "bib" and x.get("target_id") in valid_bib_ids
        for x in _as_list(payload, "xref")
    )
    if not has_bib_target:
        return [
            ValidationIssue(
                "VAL_XREF_ZERO",
                "warning",
                f"{len(bib)} references but zero in-text citation xrefs resolve to a bib",
            )
        ]
    return []


def xref_low_coverage_issue(
    bib_ids: set[int],
    linked_bib_ids: set[int],
    *,
    bib_count: int | None = None,
    origin_stage: str = "export",
) -> ValidationIssue | None:
    """Return the conservative unique-target xref warning, when applicable."""

    denominator = len(bib_ids) if bib_count is None else bib_count
    if denominator < 10:
        return None
    valid_linked = bib_ids & linked_bib_ids
    fraction = len(valid_linked) / denominator
    if fraction >= 0.20:
        return None
    return ValidationIssue(
        "VAL_XREF_LOW_COVERAGE",
        IssueSeverity.WARNING,
        f"{len(valid_linked)}/{denominator} bibliography entries have a resolved "
        f"in-text citation xref ({fraction:.1%})",
        origin_stage=origin_stage,
        evidence_ids=tuple(f"bib:{bib_id}" for bib_id in sorted(valid_linked)),
        count=len(valid_linked),
        blocking=False,
    )


def _check_xref_low_coverage(payload: dict) -> list[ValidationIssue]:
    bib = _as_list(payload, "bib")
    bib_ids = {
        row.get("bib_id")
        for row in bib
        if isinstance(row, dict) and isinstance(row.get("bib_id"), int)
    }
    linked_bib_ids = {
        row.get("target_id")
        for row in _as_list(payload, "xref")
        if isinstance(row, dict)
        and row.get("xref_type") == "bib"
        and isinstance(row.get("target_id"), int)
    }
    issue = xref_low_coverage_issue(bib_ids, linked_bib_ids, bib_count=len(bib))
    return [issue] if issue is not None else []


def _check_ref_count_mismatch(payload: dict) -> list[ValidationIssue]:
    ref_ids = {
        s.get("section_id")
        for s in _as_list(payload, "section")
        if isinstance(s, dict) and s.get("section_type") == "references"
    }
    if not ref_ids:
        return []
    ref_text = sum(
        1
        for t in _as_list(payload, "text")
        if isinstance(t, dict) and t.get("section_id") in ref_ids
    )
    bib_n = len(_as_list(payload, "bib"))
    # A floor avoids noise on tiny reference lists where one stray line skews
    # the ratio; the audit targets sharp count divergence, not off-by-one.
    if ref_text >= 5 and ref_text > bib_n * 1.3:
        return [
            ValidationIssue(
                "VAL_REF_COUNT_MISMATCH",
                IssueSeverity.WARNING,
                f"{ref_text} reference-section text records vs {bib_n} parsed bib entries",
                count=ref_text,
            )
        ]
    return []


def _check_statement_orphan(payload: dict) -> list[ValidationIssue]:
    metadata = _as_dict(payload, "metadata")
    corpus = " ".join(
        t.get("text", "")
        for t in _as_list(payload, "text")
        if isinstance(t, dict) and isinstance(t.get("text"), str)
    ).casefold()
    if not corpus:
        return []
    orphans = [
        anchor
        for anchor, field_name in _STATEMENT_ANCHORS
        if anchor in corpus and not _text(metadata.get(field_name))
    ]
    if orphans:
        return [
            ValidationIssue(
                "VAL_STATEMENT_ORPHAN",
                IssueSeverity.WARNING,
                f"statement anchor(s) present in text but metadata field empty: {', '.join(orphans)}",
                count=len(orphans),
            )
        ]
    return []


_CHECKS = (
    _check_placeholder,
    _check_author_blank,
    _check_author_outlier,
    _check_bbox_space,
    _check_dangling_ref,
    _check_duplicate_pk,
    _check_empty_eq,
    _check_url_malformed,
    _check_title_generic,
    _check_unicode_canonical,
    _check_abstract_missing,
    _check_abstract_suspect,
    _check_appendix_nesting,
    _check_caption_missing,
    _check_panel_caption,
    _check_xref_zero,
    _check_xref_low_coverage,
    _check_ref_count_mismatch,
    _check_statement_orphan,
)


def validate_export(payload: dict) -> list[ValidationIssue]:
    """Run every output-validation check over the assembled export dict.

    Never raises: a malformed payload yields ``[]`` and a check that blows up
    is isolated (a ``VAL_INTERNAL`` warning is emitted in its place).
    """
    if not isinstance(payload, dict):
        return []
    issues: list[ValidationIssue] = []
    for check in _CHECKS:
        try:
            issues.extend(check(payload))
        except Exception:  # noqa: BLE001 - a broken check must not suppress the rest
            name = getattr(check, "__name__", "?")
            log.exception("output validation check %s raised", name)
            issues.append(
                ValidationIssue("VAL_INTERNAL", "warning", f"validation check {name} raised")
            )
    return issues
