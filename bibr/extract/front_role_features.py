"""Per-region features for first-page role classification (production).

The single source of truth for the front-role feature contract: bibr-training's
trainer imports this module so the model sees identical features at fit time and
at inference, the same arrangement ``geom_features.line_features`` has with the
reference segmenter.

Callers hand in ``FrontRegion`` rows already normalized to a canonical page
frame — top-down, origin at the top-left, both axes scaled to 0..1. OCR emits
two incompatible boxes for the same region (``bbox_2d`` is 0..1000 top-down;
``bbox_pdf_pts`` is PDF points, bottom-up), so the conversion is the caller's
job and ``from_image_bbox`` / ``from_pdf_bbox`` exist to make it one line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_GEOM_KEYS = frozenset(
    {
        "x0_rel",
        "y0_rel",
        "width_rel",
        "height_rel",
        "center_offset",
        "dy_prev",
        "dx_prev",
        "font_size",
        "dfont_prev",
        "font_rel_max",
        "font_bold",
        "is_first_on_page",
        "index_rel",
        "page",
    }
)
_TEXT_KEYS = frozenset(
    {
        "len_chars",
        "n_tokens",
        "caps_ratio",
        "titlecase_ratio",
        "initial_ratio",
        "digit_ratio",
        "comma_per_token",
        "n_semicolons",
        "n_ands",
        "n_marks",
        "ends_period",
        "has_email",
        "has_year",
        "has_doi",
        "has_org_cue",
        "has_corresp_cue",
        "has_citation_cue",
        "has_keyword_cue",
        "has_abstract_cue",
        "has_history_cue",
        "has_licence_cue",
        "has_volume_cue",
        "has_contrib_cue",
    }
)
# ``region_label`` is a string; DictVectorizer expands it to one-hot columns, so
# it is deliberately absent from the numeric key sets above.
PROD_FEATURE_KEYS = _GEOM_KEYS | _TEXT_KEYS | {"region_label"}

# Layout labels that never carry front-matter text. They still take part in
# ``region_features`` (neighbour deltas, page counts) but are neither trained
# on nor scored. Shared with bibr-training's labeler so both sides agree.
NON_TEXT_LABELS = frozenset(
    {"image", "chart", "table", "seal", "formula", "display_formula", "inline_formula"}
)

_EMAIL = re.compile(r"[^\s@]+@[^\s@]+\.[A-Za-z]{2,}")
_YEAR = re.compile(r"\b(?:19|20)\d\d\b")
_DOI = re.compile(r"\b10\.\d{4,9}/", re.IGNORECASE)
_ORG_CUE = re.compile(
    r"\b(universit|department|departamento|dipartimento|institut|college|school|hospital|"
    r"laborator|facult|academy|centre|center|clinic|foundation|ministry)",
    re.IGNORECASE,
)
_CORRESP_CUE = re.compile(r"\b(corresponding|correspondence|e-?mail|reprint)", re.IGNORECASE)
_CITATION_CUE = re.compile(
    r"\b(citation|how to cite|cite this|para citar|como citar|doi:)", re.IGNORECASE
)
_KEYWORD_CUE = re.compile(
    r"\b(keywords?|key words|palabras clave|palavras[- ]chave|mots[- ]cl[eé]s|"
    r"schlagw[oö]rter)",
    re.IGNORECASE,
)
_ABSTRACT_CUE = re.compile(
    r"\b(abstract|resumen|resumo|zusammenfassung|r[eé]sum[eé])\b", re.IGNORECASE
)
_HISTORY_CUE = re.compile(
    r"\b(received|accepted|published|revised|submitted|available online)\b", re.IGNORECASE
)
_LICENCE_CUE = re.compile(
    r"(©|copyright|licensee|creative commons|open access|all rights reserved)", re.IGNORECASE
)
_VOLUME_CUE = re.compile(
    r"\bvol(?:ume)?\.?\s*\d|\be?-?issn\b|\bpp\.\s*\d|\bno\.\s*\d", re.IGNORECASE
)
_CONTRIB_CUE = re.compile(
    r"contributed equally|equal contribut|author contributions|joint (?:first|senior) author|"
    r"share[ds]? (?:first|last|senior) authorship",
    re.IGNORECASE,
)
# Superscript affiliation keys and corresponding-author daggers. OCR flattens
# them into the line, so their density is the cheapest byline signal there is.
_MARK = re.compile(r"[*†‡§¶#]")
_INITIAL = re.compile(r"^[A-Z]\.?$")


@dataclass(frozen=True)
class FrontRegion:
    """One layout region in the canonical top-down 0..1 page frame."""

    page: int
    index: int
    label: str | None
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float | None = None
    font_bold: bool | None = None

    @classmethod
    def from_image_bbox(cls, bbox, *, scale: float = 1000.0, **kw) -> FrontRegion:
        """Build from OCR ``bbox_2d`` — already top-down, scaled to 0..``scale``."""
        x0, y0, x1, y1 = (float(v) / scale for v in bbox)
        return cls(x0=x0, y0=y0, x1=x1, y1=y1, **kw)

    @classmethod
    def from_pdf_bbox(cls, bbox, *, page_w: float, page_h: float, **kw) -> FrontRegion:
        """Build from a PDF-point box (bottom-left origin, y-up); flips to top-down."""
        x0, y0, x1, y1 = (float(v) for v in bbox)
        return cls(
            x0=x0 / page_w,
            y0=(page_h - y1) / page_h,
            x1=x1 / page_w,
            y1=(page_h - y0) / page_h,
            **kw,
        )


def _ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def _text_features(text: str) -> dict:
    tokens = text.split()
    alpha = [c for c in text if c.isalpha()]
    alpha_tokens = [t for t in tokens if t[:1].isalpha()]
    return {
        "len_chars": len(text),
        "n_tokens": len(tokens),
        "caps_ratio": _ratio(sum(c.isupper() for c in alpha), len(alpha)),
        "titlecase_ratio": _ratio(sum(t[:1].isupper() for t in alpha_tokens), len(alpha_tokens)),
        # "A. B. Smith, C. D. Jones" — initials are dense in bylines and rare
        # everywhere else on the page except the self-citation that mimics one.
        "initial_ratio": _ratio(sum(bool(_INITIAL.match(t)) for t in tokens), len(tokens)),
        "digit_ratio": _ratio(sum(c.isdigit() for c in text), len(text)),
        "comma_per_token": _ratio(text.count(","), len(tokens)),
        "n_semicolons": text.count(";"),
        "n_ands": len(re.findall(r"\band\b|&", text, re.IGNORECASE)),
        "n_marks": len(_MARK.findall(text)),
        "ends_period": text.rstrip().endswith("."),
        "has_email": bool(_EMAIL.search(text)),
        "has_year": bool(_YEAR.search(text)),
        "has_doi": bool(_DOI.search(text)),
        "has_org_cue": bool(_ORG_CUE.search(text)),
        "has_corresp_cue": bool(_CORRESP_CUE.search(text)),
        "has_citation_cue": bool(_CITATION_CUE.search(text)),
        "has_keyword_cue": bool(_KEYWORD_CUE.search(text)),
        "has_abstract_cue": bool(_ABSTRACT_CUE.search(text)),
        "has_history_cue": bool(_HISTORY_CUE.search(text)),
        "has_licence_cue": bool(_LICENCE_CUE.search(text)),
        "has_volume_cue": bool(_VOLUME_CUE.search(text)),
        "has_contrib_cue": bool(_CONTRIB_CUE.search(text)),
    }


def region_features(regions: list[FrontRegion]) -> list[dict]:
    """Feature dicts for one paper's front-matter regions, in reading order."""

    page_max_font: dict[int, float] = {}
    page_count: dict[int, int] = {}
    for r in regions:
        page_count[r.page] = page_count.get(r.page, 0) + 1
        if r.font_size:
            page_max_font[r.page] = max(page_max_font.get(r.page, 0.0), float(r.font_size))

    seen: dict[int, int] = {}
    feats: list[dict] = []
    for i, r in enumerate(regions):
        prev = regions[i - 1] if i > 0 else None
        same_page = prev is not None and prev.page == r.page
        pos = seen.get(r.page, 0)
        seen[r.page] = pos + 1
        font = float(r.font_size) if r.font_size else 0.0
        prev_font = float(prev.font_size) if (prev and prev.font_size) else 0.0
        max_font = page_max_font.get(r.page, 0.0)
        feats.append(
            {
                "region_label": r.label or "",
                "x0_rel": r.x0,
                "y0_rel": r.y0,
                "width_rel": r.x1 - r.x0,
                "height_rel": r.y1 - r.y0,
                # Front matter is typeset centred far more often than body text.
                "center_offset": abs((r.x0 + r.x1) / 2 - 0.5),
                "dy_prev": (r.y0 - prev.y1) if same_page else 0.0,
                "dx_prev": (r.x0 - prev.x0) if same_page else 0.0,
                "font_size": font,
                "dfont_prev": (font - prev_font) if same_page else 0.0,
                # The title is the largest type on page 1; absolute sizes vary
                # per journal, the ratio does not.
                "font_rel_max": (font / max_font) if max_font else 0.0,
                "font_bold": bool(r.font_bold),
                "is_first_on_page": pos == 0,
                "index_rel": _ratio(pos, page_count.get(r.page, 0)),
                "page": r.page,
                **_text_features(r.text or ""),
            }
        )
    return feats
