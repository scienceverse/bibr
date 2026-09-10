"""Per-line features for reference-boundary classification (production).

GEOM ∪ TEXT only — the layout-region prior is omitted because it scored ≈0 in
the probe importance readout and the regions are freed before the extract stage.
These are the stable production base keys. ``GeomSegmenter`` asserts that
artifact ``feature_keys`` match either these keys for the v1 line-start bundle
or these keys plus adjacent-line features for the v2 adjacent-boundary bundle.
"""

from __future__ import annotations

import re

from bibr.ocr.ref_geometry import LineRecord
from bibr.ocr.ref_patterns import _AUTHOR_DATE_START, _YEAR

_GEOM_KEYS = frozenset({"x0_rel_page", "dx_prev", "dy_gap", "font_size", "dfont", "is_first"})
_TEXT_KEYS = frozenset(
    {
        "len_chars",
        "caps_ratio",
        "starts_author_date",
        "starts_numbered",
        "starts_year",
        "contains_year",
        "early_year_paren",
        "prev_ends_period",
        "prev_ends_year",
        "prev_ends_doi",
    }
)
PROD_FEATURE_KEYS = _GEOM_KEYS | _TEXT_KEYS

# group(1) captures the integer so _parse_ref_number can read it; bool(.match()) still works
_NUMBERED = re.compile(r"^\[?(\d{1,3})(?:\][\s.]|[.)])")
_STARTS_YEAR = re.compile(r"^\(?(?:19|20)\d\d")
_PREV_YEAR_END = re.compile(r"(?:19|20)\d\d[a-z]?\)?\.?$")
# A parenthesized year or "(n.d.)" within the first ~70 chars — the author-date
# cue at a reference start. Fires even when the hanging-indent geometry is flat
# (dx_prev≈0) and the author-date regex misses the lead (corporate authors with
# no comma, Unicode/diacritic surnames). Targets the geom psych merge failure.
_EARLY_YEAR_PAREN = re.compile(r"^.{0,70}?\((?:(?:19|20)\d\d[a-z]?|n\.?\s?d\.?)\)", re.IGNORECASE)
# Page ranges: "123-456", "123–456", "123—456" optionally followed by "."
_PAGE_RANGE_END = re.compile(r"\d+[-–—]\d+\.?$")


def _caps_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


def _parse_ref_number(text: str) -> int:
    m = _NUMBERED.match(text)
    return int(m.group(1)) if m else 0


def line_features(lines: list[LineRecord]) -> list[dict]:
    page_min_x: dict[int, float] = {}
    for ln in lines:
        page_min_x[ln.page] = min(page_min_x.get(ln.page, ln.x0), ln.x0)

    feats: list[dict] = []
    for i, ln in enumerate(lines):
        prev = lines[i - 1] if i > 0 else None
        t = ln.text
        prev_t = prev.text.rstrip() if prev else ""
        feats.append(
            {
                "x0_rel_page": ln.x0 - page_min_x[ln.page],
                "dx_prev": (ln.x0 - prev.x0) if prev else 0.0,
                "dy_gap": (prev.y_bottom - ln.y_top) if (prev and prev.page == ln.page) else 0.0,
                "font_size": ln.font_size,
                "dfont": (ln.font_size - prev.font_size) if prev else 0.0,
                "is_first": i == 0,
                "len_chars": len(t),
                "caps_ratio": _caps_ratio(t),
                "starts_author_date": bool(_AUTHOR_DATE_START.match(t)),
                "starts_numbered": bool(_NUMBERED.match(t)),
                "starts_year": bool(_STARTS_YEAR.match(t)),
                "contains_year": bool(_YEAR.search(t)),
                "early_year_paren": bool(_EARLY_YEAR_PAREN.search(t)),
                "prev_ends_period": prev_t.endswith("."),
                "prev_ends_year": bool(_PREV_YEAR_END.search(prev_t)),
                "prev_ends_doi": bool(prev_t and prev_t[-1].isdigit() and "10." in prev_t),
            }
        )
    return feats
