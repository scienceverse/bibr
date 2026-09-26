"""Extract and match a PDF's outline (bookmarks / table-of-contents).

The outline, when a born-digital PDF carries one, is the most authoritative
heading-hierarchy signal available: it is the document's own declared section
tree. This module ports two pieces from Docling:

* :func:`extract_pdf_outline` / :func:`_walk_pdfium_outline` — the pypdfium2
  outline walk (``extract_outline_from_pdfium``): title, 0-based depth, 1-based
  target page and vertical position, converting destination view coordinates
  via the :data:`_VIEW_TOP_INDEX` table.
* :func:`match_outline_to_headings` — the *semantics* of Docling's bookmark
  cascade (``heading_hierarchy_model``): normalize titles (marker/number
  stripping), fuzzy-match to detected headings (threshold ~0.8), constrain by
  page, claim each heading at most once (greedy, document order), and compress
  the matched depths to contiguous 1-based levels.

Everything here is internal-only: the outline is a transient signal, never
part of the exported JSON. ``y_top`` diverges deliberately from Docling's
absolute-points representation — it is stored as a *fraction of page height
from the top* (0.0 = top edge, 1.0 = bottom edge) so it is directly comparable
to bibr's normalized 0–1000 layout bboxes (``bbox_y1 / 1000``) at match time.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rapidfuzz.fuzz import ratio as _fuzz_ratio

logger = logging.getLogger(__name__)

# Default fuzzy-similarity threshold for accepting a bookmark↔heading match.
_MATCH_THRESHOLD = 0.8
# Same-page tolerance (in pages) when a bookmark carries a target page.
_PAGE_TOLERANCE = 1

# Leading numbering marker stripped before fuzzy-matching a title, so a bookmark
# "Definitions" matches an on-page heading "1.1 Definitions" (and vice-versa).
# Ported from Docling's heading-hierarchy ``_LEADING_MARKER``.
_LEADING_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"(?:part|title|book|chapter|article|section|clause|schedule|annex|appendix|rule)"
    r"\b[\s.:]*[0-9ivxlcdm]*"
    r"|§+\s*[0-9.]+"
    r"|\(?[0-9]+(?:\.[0-9]+)*[).]?"
    r"|\(?[A-Za-z]{1,2}[).]"
    r")[\s.:)\-]*",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_OUTER_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$")


def _normalize_title(text: str) -> str:
    """Lower-case, collapse whitespace and trim outer punctuation for matching."""
    s = _WHITESPACE_RE.sub(" ", (text or "").lower()).strip()
    return _OUTER_PUNCT_RE.sub("", s)


def _strip_marker(text: str) -> str:
    return _LEADING_MARKER_RE.sub("", text or "", count=1)


@dataclass(frozen=True)
class OutlineItem:
    """A single PDF bookmark / table-of-contents entry (internal, document-ordered).

    The list is kept flat and in document order; each entry carries its own
    ``level`` so no tree structure is needed for matching.
    """

    title: str
    # 0-based depth as reported by the PDF outline; compressed to contiguous
    # 1-based levels by :func:`match_outline_to_headings`.
    level: int
    # 1-based target page; None when the entry has no resolvable page.
    page_no: int | None = None
    # Vertical position of the target as a fraction of page height from the top
    # (0.0 top … 1.0 bottom); None when the destination view encodes no top.
    y_top: float | None = None


@dataclass(frozen=True)
class HeadingRef:
    """A detected heading offered to the matcher (parser-side, internal).

    ``y_top`` is the heading's top edge as a page-height fraction (bbox y1 in
    the normalized 0–1000 space, divided by 1000), so it is comparable to
    :attr:`OutlineItem.y_top`.
    """

    text: str
    page_no: int | None = None
    y_top: float | None = None


# --------------------------------------------------------------------------- extractor


def _view_top_index() -> dict[int, int]:
    """Destination view modes → index of the top (vertical) coordinate.

    Coordinates are in PDF space (bottom-left origin). Modes not listed
    (FIT, FITV, FITB, FITBV, unknown) provide no usable top position. Built
    lazily so importing this module never forces the pypdfium2 raw bindings.
    """
    import pypdfium2.raw as pdfium_c

    return {
        pdfium_c.PDFDEST_VIEW_XYZ: 1,  # [x, y, zoom]
        pdfium_c.PDFDEST_VIEW_FITH: 0,  # [y]
        pdfium_c.PDFDEST_VIEW_FITBH: 0,  # [y]
        pdfium_c.PDFDEST_VIEW_FITR: 3,  # [left, bottom, right, top]
    }


def _dest_top_pdf(dest, view_top_index: dict[int, int]) -> tuple[int | None, float | None]:
    """Return ``(0-based page index, vertical top in PDF bottom-left coords)``.

    Either element may be ``None`` when the destination does not encode it.
    """
    page_index = dest.get_index()
    mode, pos = dest.get_view()
    idx = view_top_index.get(mode)
    y_pdf = pos[idx] if idx is not None and idx < len(pos) else None
    return page_index, y_pdf


def _walk_pdfium_outline(pdoc, page_heights: dict[int, float] | None = None) -> list[OutlineItem]:
    """Walk an open ``pypdfium2.PdfDocument``'s outline into flat ``OutlineItem``\\ s.

    The caller is responsible for holding the pdfium lock. Vertical positions
    are converted to a top-origin page-height fraction. Returns an empty list
    when the document has no outline or it cannot be read.
    """
    from pypdfium2 import PdfiumError

    view_top_index = _view_top_index()
    items: list[OutlineItem] = []
    page_heights = {} if page_heights is None else dict(page_heights)

    try:
        toc = list(pdoc.get_toc())
    except PdfiumError as exc:
        logger.debug("Could not read PDF outline: %s", exc)
        return []
    try:
        page_count = len(pdoc)
    except Exception:  # noqa: BLE001 — duck-typed documents (tests); bound checks apply when known
        page_count = None

    for bm in toc:
        title = (bm.get_title() or "").strip()
        if not title:
            continue

        page_no: int | None = None
        y_top: float | None = None
        try:
            dest = bm.get_dest()
        except PdfiumError:
            dest = None
        if dest is not None:
            try:
                page_index, y_pdf = _dest_top_pdf(dest, view_top_index)
            except Exception as exc:  # noqa: BLE001 — one bad destination must not lose the outline
                logger.debug("Could not read outline destination: %s", exc)
                page_index, y_pdf = None, None
            # A stale bookmark may point past the last page (common in excerpts
            # of proceedings volumes); keep the entry with no resolvable page
            # instead of discarding the whole outline (audit input-parsers-26).
            if page_index is not None and (page_count is None or 0 <= page_index < page_count):
                page_no = page_index + 1
                if y_pdf is not None:
                    try:
                        if page_index not in page_heights:
                            page = pdoc[page_index]
                            try:
                                page_heights[page_index] = page.get_height()
                            finally:
                                page.close()
                        height = page_heights[page_index]
                    except Exception as exc:  # noqa: BLE001 — one bad bookmark must not lose the outline
                        logger.debug("Could not read outline target page %d: %s", page_index, exc)
                        height = None
                    if height:
                        # Flip to top-origin and normalize to a 0..1 fraction.
                        y_top = min(1.0, max(0.0, (height - y_pdf) / height))

        items.append(OutlineItem(title=title, level=int(bm.level), page_no=page_no, y_top=y_top))

    return items


def extract_pdf_outline(pdf_bytes: bytes) -> list[OutlineItem]:
    """Open *pdf_bytes* under the shared pdfium lock and extract its outline.

    Best-effort: any pdfium failure (corruption, encryption, no outline)
    returns ``[]``. Callers run this off the event loop — it takes the shared
    :data:`bibr.ocr.utils.pdfium_lock`.
    """
    from bibr.ocr.utils import pdfium_lock

    try:
        import pypdfium2 as pdfium

        with pdfium_lock:
            doc = pdfium.PdfDocument(pdf_bytes)
            try:
                return _walk_pdfium_outline(doc)
            finally:
                doc.close()
    except Exception:  # noqa: BLE001 — best-effort; never fail the file
        logger.debug("PDF outline read failed", exc_info=True)
        return []


# --------------------------------------------------------------------------- matcher


def _title_similarity(cand_text: str, bm_title: str) -> float:
    """Fuzzy similarity in 0..1 between a detected heading and a bookmark title.

    Both strings are compared with and without their leading numbering marker
    (bookmarks often omit or carry a different marker), and containment of one
    normalized title in the other boosts the score (bookmarks are frequently
    truncated).
    """
    variants_a = {_normalize_title(cand_text), _normalize_title(_strip_marker(cand_text))} - {""}
    variants_b = {_normalize_title(bm_title), _normalize_title(_strip_marker(bm_title))} - {""}
    best = 0.0
    for a in variants_a:
        for b in variants_b:
            best = max(best, _fuzz_ratio(a, b) / 100.0)
            if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
                best = max(best, 0.92)
    return best


def match_outline_to_headings(
    outline,
    headings,
    *,
    threshold: float = _MATCH_THRESHOLD,
    page_tolerance: int = _PAGE_TOLERANCE,
) -> dict[int, int]:
    """Match outline entries to detected headings; return ``{heading_index: level}``.

    Ports Docling's bookmark cascade semantics:

    * Titles are normalized (marker/number stripping, casefold, whitespace
      collapse, outer-punctuation trim) and fuzzily compared; a match must
      reach *threshold*. A bookmark without a target page (``page_no is None``)
      demands a stronger match (threshold + 0.1) since the page cannot corroborate.
    * When both the bookmark and a heading carry a page, they must agree within
      *page_tolerance* pages.
    * Each outline entry claims at most one heading; iteration is in document
      order, greedy on best score (ties broken by page proximity first — a
      bookmark's exact target page wins — then by vertical proximity when both
      ``y_top`` values are known). Once claimed, a heading is unavailable.
    * The matched raw (0-based) depths are compressed to contiguous 1-based
      levels, so a document whose shallowest matched bookmark sits at depth 2
      is not forced to start at level 3.

    Levels are 1-based; unmatched headings are simply absent from the result.
    """
    inf = float("inf")
    claimed: set[int] = set()
    matches: list[tuple[int, int]] = []

    for item in outline:
        title = (item.title or "").strip()
        if not title:
            continue
        thr = threshold if item.page_no is not None else min(1.0, threshold + 0.1)

        best_idx: int | None = None
        best_score = 0.0
        best_dist = (inf, inf)
        for idx, head in enumerate(headings):
            if idx in claimed:
                continue
            if (
                item.page_no is not None
                and head.page_no is not None
                and abs(head.page_no - item.page_no) > page_tolerance
            ):
                continue
            score = _title_similarity(head.text, title)
            if score < thr:
                continue
            # Tie-break distance is a lexicographic (page, y) tuple: page wins,
            # then vertical proximity. y_top is PAGE-RELATIVE (fraction of page
            # height), so it is only meaningful WITHIN a page — a y-only tie-break
            # can let a duplicate heading on the wrong page out-tie the correct
            # one. Prefer the bookmark's target page first; unknown page/y sort
            # last (inf).
            page_distance = (
                abs(head.page_no - item.page_no)
                if (head.page_no is not None and item.page_no is not None)
                else inf
            )
            y_distance = (
                abs(head.y_top - item.y_top)
                if (head.y_top is not None and item.y_top is not None)
                else inf
            )
            dist = (page_distance, y_distance)
            if score > best_score + 1e-6 or (abs(score - best_score) <= 1e-6 and dist < best_dist):
                best_idx, best_score, best_dist = idx, score, dist
        if best_idx is not None:
            claimed.add(best_idx)
            matches.append((best_idx, item.level))

    if not matches:
        return {}

    used_levels = sorted({lvl for _, lvl in matches})
    level_map = {lvl: i + 1 for i, lvl in enumerate(used_levels)}
    return {idx: level_map[lvl] for idx, lvl in matches}
