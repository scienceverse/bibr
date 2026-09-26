"""Merge accepted ``bib_match`` rows into ``bib`` rows (consolidation).

Operates on the export-dict shape so the same function backs the
pipeline post-export hook, ``Result.consolidate()``, and the serve API.
``bib_match`` rows are already threshold-gated by enrichment (DOI lookups
score 1; bibliographic search is fuzzy-gated with author/year penalties),
so no further score filtering happens here. Overwriting a printed value is
held to a stricter bar than filling a gap: ``replace`` overwrites only from
a match that carries the reference's own printed DOI.
"""

from __future__ import annotations

from typing import Literal

from bibr.utils.text import normalize_doi

# bib columns fillable from a bib_match row. authors/editors are excluded:
# bib stores the printed string, bib_match stores structured dicts — there is
# no clean conversion that preserves the printed form. So is the printed
# ``date``: the registry's date is ISO 8601 and fills ``published_date``.
CONSOLIDATABLE_FIELDS = (
    "bib_type",
    "doi",
    "title",
    "publisher",
    "year",
    "published_date",
    "container",
    "volume",
    "issue",
    "first_page",
    "last_page",
    "edition",
    "version",
    "url",
)

# Highest-trust service first; unknown services sort after the known ones.
SERVICE_PRECEDENCE = ("crossref",)

# A registry's catch-all value says only that the record fits none of the
# others. It may fill a gap but never replaces what the paper printed: a
# Crossref "standard" or "journal-issue" maps to bib_type "other", and a
# printed "book" is more specific than that.
_CATCH_ALL = {"bib_type": "other"}


def _doi_key(value) -> str | None:
    """Case-folded DOI for identity checks; DOIs are case-insensitive."""
    if not isinstance(value, str) or not value.strip():
        return None
    return (normalize_doi(value) or value.strip()).casefold()


def _is_same_work(bib: dict, match: dict) -> bool:
    """Whether *match* is the work the reference's printed DOI names.

    A score cannot decide this: a fuzzy title match can also score 1, and
    ``bib_match`` rows do not record how they were found.
    """
    printed = _doi_key(bib.get("doi"))
    return printed is not None and printed == _doi_key(match.get("doi"))


def consolidate_bibs(data: dict, mode: Literal["fill", "replace"] = "fill") -> int:
    """Merge ``data["bib_match"]`` into ``data["bib"]`` in place.

    ``mode="fill"`` only fills empty/missing bib fields; ``mode="replace"``
    also overwrites fields that disagree with a match carrying the
    reference's printed DOI. A match found by bibliographic search (or any
    match for a reference printed without a DOI) only fills, in both modes,
    so a near-miss search hit never rewrites what the paper printed. The field
    names taken per row are recorded in ``extraction.diagnostics.consolidation``
    (see :func:`_record_consolidation`), never on the bib rows themselves.
    Returns the number of modified rows.
    """
    if mode not in ("fill", "replace"):
        raise ValueError(f"consolidate mode must be 'fill' or 'replace', got {mode!r}")

    matches_by_bib: dict[int, list[dict]] = {}
    for m in data.get("bib_match") or []:
        matches_by_bib.setdefault(m["bib_id"], []).append(m)

    rank = {s: i for i, s in enumerate(SERVICE_PRECEDENCE)}
    taken_by_bib: dict[int, list[str]] = {}
    for bib in data.get("bib") or []:
        matches = matches_by_bib.get(bib.get("bib_id"))
        if not matches:
            continue
        matches = sorted(matches, key=lambda m: rank.get(m.get("service"), len(rank)))
        same_work = [m for m in matches if _is_same_work(bib, m)] if mode == "replace" else []
        taken: list[str] = []
        for field in CONSOLIDATABLE_FIELDS:
            current = bib.get(field)
            empty = current in (None, "")
            sources = matches if empty else same_work
            value = next(
                (
                    m.get(field)
                    for m in sources
                    if m.get(field) not in (None, "")
                    and (empty or m.get(field) != _CATCH_ALL.get(field))
                ),
                None,
            )
            if value is None or value == current:
                continue
            bib[field] = value
            taken.append(field)
        # When `year` is taken from a match, its printed-entry companions no
        # longer describe the row: a matched work carries no disambiguating
        # suffix, and a concrete year means the ref is no longer "in press".
        # Reconcile them so consolidation never emits a self-contradictory row
        # (e.g. year=2007 with a stale "a", or a hard year with is_in_press).
        if "year" in taken:
            if bib.get("year_suffix") not in (None, ""):
                bib["year_suffix"] = None
                taken.append("year_suffix")
            if bib.get("is_in_press"):
                bib["is_in_press"] = False
                taken.append("is_in_press")
            # ``published_date`` restates the year. The loop above already took
            # a match's fuller date where the mode allows; when the record dates
            # the work only by year, the year stands in, filling a gap in either
            # mode and replacing a stale date only in ``replace``.
            year, published = bib.get("year"), bib.get("published_date")
            stale = not (isinstance(published, str) and published.startswith(f"{year:04d}"))
            if isinstance(year, int) and stale and (mode == "replace" or not published):
                bib["published_date"] = f"{year:04d}"
                if "published_date" not in taken:
                    taken.append("published_date")
        if taken:
            taken_by_bib[bib["bib_id"]] = taken
    _record_consolidation(data, taken_by_bib)
    return len(taken_by_bib)


def _record_consolidation(data: dict, taken_by_bib: dict[int, list[str]]) -> None:
    """Merge *taken_by_bib* into ``extraction.diagnostics.consolidation``.

    The receipt is processing provenance, so it lives under ``extraction``
    rather than on the content rows. A repeated consolidation (e.g. after an
    enrichment replay) unions field names per ``bib_id``. A payload without an
    ``extraction`` block (a Paper exported outside the pipeline) has nowhere to
    record it; the merged ``bib`` values are then indistinguishable from
    printed ones except by comparing against ``bib_match``.
    """
    extraction = data.get("extraction")
    if not isinstance(extraction, dict):
        return
    diagnostics = extraction.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = extraction["diagnostics"] = {}
    merged: dict[int, list[str]] = {
        row["bib_id"]: list(row["fields"]) for row in diagnostics.get("consolidation") or []
    }
    for bib_id, fields in taken_by_bib.items():
        existing = merged.setdefault(bib_id, [])
        existing.extend(f for f in fields if f not in existing)
    diagnostics["consolidation"] = [
        {"bib_id": bib_id, "fields": fields} for bib_id, fields in sorted(merged.items())
    ]
