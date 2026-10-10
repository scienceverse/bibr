"""Page triage: born-digital, born-digital with a broken text layer, or scan.

The class is read from what the native-text pass already decided, so it costs
nothing extra:

- ``scan``: the page has no text layer, or its text layer is the invisible OCR
  layer of a scanned image (``PdfPageInspection.invisible_text_layer``).
- ``broken_text_layer``: the page has a visible text layer, but the
  corruption gates rejected most of its text regions (a broken ToUnicode map,
  private-use glyphs), so OCR read them instead.
- ``born_digital``: everything else.

A page with no inspection (the native-text pass did not run or failed) has no
class: callers treat it as not a scan.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bibr.ocr.pdf_inspection import PdfInspection, PdfPageInspection


class PageKind(StrEnum):
    BORN_DIGITAL = "born_digital"
    BROKEN_TEXT_LAYER = "broken_text_layer"
    SCAN = "scan"


# Values of a layout region's ``_native_gate``: the native-text gate that
# rejected the region's text layer (set by bibr.ocr.native_text).
NATIVE_GATE_USABILITY = "usability"
NATIVE_GATE_PRIVATE_USE = "private_use"
_REJECTING_GATES = frozenset({NATIVE_GATE_USABILITY, NATIVE_GATE_PRIVATE_USE})

# A page is a broken text layer when at least this share of its text regions
# that had text-layer text were rejected by a corruption gate, and at least
# _BROKEN_MIN_REGIONS of them were. One bad region (a logo set in a symbol
# font) is not a broken page.
_BROKEN_MIN_SHARE = 0.5
_BROKEN_MIN_REGIONS = 2

# Labels whose text the layout model reads as running text; the share above
# is counted over these.
_TEXT_LABELS = frozenset(
    {
        "abstract",
        "content",
        "doc_title",
        "figure_title",
        "footnote",
        "paragraph_title",
        "reference",
        "reference_content",
        "text",
        "vision_footnote",
    }
)


def classify_page(
    inspection: PdfPageInspection | None, layout_regions: Iterable[dict[str, Any]] | None
) -> PageKind | None:
    """The class of one page, or None when the page was not inspected.

    ``layout_regions`` None means the layout was not kept (an OCR cache hit):
    the class then rests on the inspection alone.
    """
    if inspection is None:
        return None
    if inspection.invisible_text_layer:
        return PageKind.SCAN
    if layout_regions is None:
        return PageKind.SCAN if inspection.char_count == 0 else PageKind.BORN_DIGITAL
    regions = [r for r in layout_regions if r.get("label") in _TEXT_LABELS]
    if inspection.char_count == 0:
        # No text layer at all. A page with no text regions either (a blank
        # or full-page figure) has nothing to read, so it is not a scan.
        return PageKind.SCAN if regions else PageKind.BORN_DIGITAL
    read = [r for r in regions if r.get("_native_text_used") or r.get("_native_gate")]
    rejected = sum(1 for r in read if r.get("_native_gate") in _REJECTING_GATES)
    if rejected >= _BROKEN_MIN_REGIONS and rejected >= _BROKEN_MIN_SHARE * len(read):
        return PageKind.BROKEN_TEXT_LAYER
    return PageKind.BORN_DIGITAL


def classify_pages(
    inspection: PdfInspection | None,
    layout_results: Sequence[Sequence[dict[str, Any]]] | None,
    page_indices: Sequence[int] | None = None,
) -> dict[int, PageKind]:
    """Each inspected page's class, keyed by absolute 0-based page index.

    ``layout_results`` is window-relative (one entry per rendered page) and
    ``page_indices`` maps it to absolute pages; without it the two are taken to
    coincide. Pages the inspection does not cover are left out.
    """
    if inspection is None:
        return {}
    layout = list(layout_results or [])
    indices = list(page_indices) if page_indices is not None else list(range(len(layout)))
    regions_by_page = dict(zip(indices, layout, strict=False))
    return {
        page.index: kind
        for page in inspection.pages
        if (kind := classify_page(page, regions_by_page.get(page.index))) is not None
    }


def ensure_page_kinds(fs: Any) -> dict[int, PageKind]:
    """The file's page classes, computed once from its inspection and layout and kept.

    ``fs`` is a pipeline ``FileState``; the classes are stored on
    ``fs.page_kinds`` as plain strings.
    """
    if fs.page_kinds is None:
        kinds = classify_pages(
            getattr(fs, "pdf_inspection", None),
            getattr(fs, "layout_results", None),
            getattr(fs, "page_indices", None),
        )
        fs.page_kinds = {index: kind.value for index, kind in kinds.items()}
    return {index: PageKind(kind) for index, kind in fs.page_kinds.items()}


def summarize(kinds: dict[int, PageKind]) -> dict[str, int]:
    """How many pages fall in each class, for logs and diagnostics."""
    counts = {kind.value: 0 for kind in PageKind}
    for kind in kinds.values():
        counts[kind.value] += 1
    return counts
