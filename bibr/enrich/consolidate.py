"""Merge accepted ``bib_match`` rows into ``bib`` rows (consolidation).

Operates on the v10.3 export-dict shape so the same function backs the
pipeline post-export hook, ``Result.consolidate()``, and the serve API.
``bib_match`` rows are already threshold-gated by enrichment (DOI lookups
score 100; bibliographic search is fuzzy-gated with author/year penalties),
so no further score filtering happens here.
"""

from __future__ import annotations

from typing import Literal

# bib columns fillable from a bib_match row. authors/editors are excluded:
# bib stores the printed string, bib_match stores structured dicts — there is
# no clean conversion that preserves the printed form.
CONSOLIDATABLE_FIELDS = (
    "bib_type",
    "doi",
    "title",
    "publisher",
    "year",
    "date",
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


def consolidate_bibs(data: dict, mode: Literal["fill", "replace"] = "fill") -> int:
    """Merge ``data["bib_match"]`` into ``data["bib"]`` in place.

    ``mode="fill"`` only fills empty/missing bib fields; ``mode="replace"``
    also overwrites fields that disagree with the match. Modified rows get a
    ``consolidated_fields`` column — a comma-joined string of the field names
    taken from the match (scalar, so R consumers stay happy). Returns the
    number of modified rows.
    """
    if mode not in ("fill", "replace"):
        raise ValueError(f"consolidate mode must be 'fill' or 'replace', got {mode!r}")

    matches_by_bib: dict[int, list[dict]] = {}
    for m in data.get("bib_match") or []:
        matches_by_bib.setdefault(m["bib_id"], []).append(m)

    rank = {s: i for i, s in enumerate(SERVICE_PRECEDENCE)}
    modified = 0
    for bib in data.get("bib") or []:
        matches = matches_by_bib.get(bib.get("bib_id"))
        if not matches:
            continue
        matches = sorted(matches, key=lambda m: rank.get(m.get("service"), len(rank)))
        taken: list[str] = []
        for field in CONSOLIDATABLE_FIELDS:
            value = next(
                (m.get(field) for m in matches if m.get(field) not in (None, "")),
                None,
            )
            if value is None:
                continue
            current = bib.get(field)
            if current in (None, "") or (mode == "replace" and value != current):
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
        if taken:
            bib["consolidated_fields"] = ",".join(taken)
            modified += 1
    return modified
