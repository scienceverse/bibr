"""Page geometry of the exported bounding boxes.

Layout analysis reports region boxes in a normalized image space: 0..1000 on
both axes of the rendered page, origin top-left, y down. The render is the
page as displayed (its ``/Rotate`` applied, cropped to the CropBox), which is
also the frame ``pypdfium2``'s ``page.get_size()`` measures in PDF points.

The export uses one convention for every box: ``[x0, y0, x1, y1]`` in PDF
points (1/72 inch) on the page as displayed, measured from its top-left
corner with y increasing downward. :class:`PageGeometry` converts layout
boxes to it and lists the page sizes for ``extraction.pages``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from bibr.export.models import PageExport

LAYOUT_SPAN = 1000.0


class PageGeometry:
    """Page sizes of one document, keyed by 1-based page number."""

    def __init__(self, sizes: Mapping[int, tuple[float, float]]) -> None:
        self._sizes = {
            page: (float(w), float(h)) for page, (w, h) in sizes.items() if w > 0 and h > 0
        }

    def pages(self) -> list[PageExport] | None:
        """``extraction.pages`` rows, or ``None`` when no page size is known."""
        if not self._sizes:
            return None
        return [
            PageExport(page_number=page, width=round(w, 2), height=round(h, 2))
            for page, (w, h) in sorted(self._sizes.items())
        ]

    def scale(self, page_number: int | None) -> tuple[float, float] | None:
        """Points per layout unit on each axis of *page_number*."""
        size = self._sizes.get(page_number) if page_number is not None else None
        if size is None:
            return None
        return size[0] / LAYOUT_SPAN, size[1] / LAYOUT_SPAN

    def box(
        self, page_number: int | None, layout_bbox: Sequence[float] | None
    ) -> list[float] | None:
        """*layout_bbox* in exported points, or ``None`` when it cannot be placed."""
        scale = self.scale(page_number)
        if scale is None or not layout_bbox or len(layout_bbox) != 4:
            return None
        sx, sy = scale
        x0, y0, x1, y1 = (float(v) for v in layout_bbox)
        return [round(x0 * sx, 2), round(y0 * sy, 2), round(x1 * sx, 2), round(y1 * sy, 2)]
