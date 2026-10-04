"""One-open, detached inspection of born-digital PDF data."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from bibr.input.pdf_metadata import harvest_docinfo
from bibr.input.pdf_outline import OutlineItem, _walk_pdfium_outline
from bibr.ocr.native_text import (
    DEFAULT_ELIGIBLE_LABELS,
    _attach_bbox_pdf_pts,
    _attach_page_dimensions,
    _fill_page_regions_from_textpage,
    _is_invisible_text_layer_page,
    _page_crop_box,
    _page_rotation,
    _pdf_points_to_normalized_bbox,
    _sample_page_font_metadata,
    open_text_page,
)
from bibr.ocr.pdf_links import PdfUriLink, page_uri_links
from bibr.ocr.ref_geometry import (
    LineRecord,
    _extract_page_chars,
    group_chars_into_lines,
    record_to_dict,
    reference_lines_from_pages,
)
from bibr.ocr.ref_patterns import _REF_HEADER_RE
from bibr.ocr.utils import pdfium_lock

if TYPE_CHECKING:
    from bibr.document.harvest import LayerBuilder, RenderBudget
    from bibr.document.model import DocumentLayer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfPageInspection:
    index: int
    width: float
    height: float
    crop_box: tuple[float, float, float, float]
    char_count: int
    # A scan whose text layer is a hidden OCR layer (see
    # ``_is_invisible_text_layer_page``): no text-layer data is read from it,
    # apart from the regions' OCR fallback text (``_invisible_layer_text``).
    invisible_text_layer: bool = False
    # Diagonal watermark strings removed before the text layer was read
    # ("For Review Only", "RETRACTED"; see ``strip_furniture_objects``).
    watermarks: tuple[str, ...] = ()


@dataclass(frozen=True)
class PdfInspection:
    pages: tuple[PdfPageInspection, ...]
    layout_results: list[list[dict[str, Any]]]
    metadata: dict[str, Any]
    outline: list[OutlineItem]
    reference_lines: list[dict[str, Any]]
    component_errors: dict[str, str] = field(default_factory=dict)
    # Every text-layer line of every page, each with its box in the 0..1000
    # layout space as ``bbox``, ``page`` as the 1-based PDF page number (the
    # one region summaries carry, also under ``--start-page``) and the page's
    # ``rotation``, plus the PDF-point geometry the geom features read (in the
    # unrotated page frame). The reference line stream selects the lines
    # inside the located section's layout regions from these. Captured with
    # ``include_ref_geometry``.
    page_lines: list[dict[str, Any]] = field(default_factory=list)
    # URI link annotations (``page``, layout-space ``bbox``, ``uri``), same
    # frame and capture condition as ``page_lines``.
    uri_links: list[dict[str, Any]] = field(default_factory=list)
    # The PDF-native document layer (``bibr.document``), built with
    # ``include_doc_layer``. Never serialized with the inspection.
    document: DocumentLayer | None = field(default=None, compare=False, repr=False)


def inspect_pdf(
    pdf_bytes: bytes,
    layout_results: list[list[dict]],
    *,
    page_indices: list[int] | tuple[int, ...] | None = None,
    first_page_text: str = "",
    fill_native_text: bool,
    include_outline: bool,
    include_ref_geometry: bool,
    min_chars: int,
    min_printable_ratio: float,
    eligible_labels: frozenset[str] = DEFAULT_ELIGIBLE_LABELS,
    reject_invisible_text_layer: bool = False,
    include_doc_layer: bool = False,
    render_budget: RenderBudget | None = None,
) -> PdfInspection:
    """Inspect a PDF under one lock/open and return plain detached values.

    With *reject_invisible_text_layer*, a scanned page whose text layer is an
    invisible OCR layer is read like a page without one: no native fill, font
    metadata or text lines come from it, so OCR reads its regions. The text
    the fill would have taken stays on each region as the fallback OcrStage
    uses when OCR returns nothing for the page.

    With *include_doc_layer*, every page's text layer is also harvested into
    ``PdfInspection.document`` (see ``bibr.document``), with render recipes
    for *render_budget*. The harvest only reads: every other field is the
    same with it on or off, and its failures go to the layer's own
    ``component_errors``.
    """
    import pypdfium2

    layouts = deepcopy(layout_results)
    pages: list[PdfPageInspection] = []
    component_errors: dict[str, str] = {}
    page_lines = {}
    header_page: int | None = None
    stream_lines: list[dict[str, Any]] = []
    uri_links: list[dict[str, Any]] = []
    docinfo: dict[str, Any] = {}
    outline: list[OutlineItem] = []
    layer_builder = _layer_builder(pdf_bytes, render_budget) if include_doc_layer else None
    document: DocumentLayer | None = None

    with pdfium_lock:
        doc = pypdfium2.PdfDocument(pdf_bytes)
        try:
            if page_indices is None:
                page_pairs = list(enumerate(range(len(doc))))
            else:
                if len(page_indices) != len(layouts):
                    raise ValueError("page_indices must align one-to-one with layout_results")
                if len(set(page_indices)) != len(page_indices):
                    raise ValueError("page_indices must not contain duplicates")
                if any(page_index < 0 for page_index in page_indices):
                    raise ValueError("page_indices must not contain negative values")
                if any(page_index >= len(doc) for page_index in page_indices):
                    raise ValueError("page_indices contains an out-of-range PDF page")
                page_pairs = list(enumerate(page_indices))

            try:
                docinfo = {
                    key: doc.get_metadata_value(key) for key in ("Title", "Subject", "Keywords")
                }
            except Exception as exc:  # noqa: BLE001 - best-effort component
                component_errors["metadata"] = _error_text(exc)

            page_heights: dict[int, float] = {}
            for layout_slot, page_index in page_pairs:
                page = doc[page_index]
                try:
                    width, height = page.get_size()
                    page_heights[page_index] = float(height)
                    crop_box = _page_crop_box(page)
                    rotation = _page_rotation(page)
                    regions = layouts[layout_slot] if layout_slot < len(layouts) else []
                    _attach_page_dimensions(page, regions)
                    _attach_bbox_pdf_pts(crop_box, regions, rotation)

                    needs_text = fill_native_text or include_ref_geometry
                    char_count = 0
                    invisible_text_layer = False
                    watermarks: list[str] = []
                    # The document layer's furniture, and the char records it
                    # shares with the native fill.
                    furniture: list[tuple] = []
                    trace = records = None
                    if needs_text:
                        if layer_builder is None:
                            textpage = open_text_page(page, watermarks)
                        else:
                            textpage = open_text_page(page, watermarks, furniture)
                        try:
                            if layer_builder is not None:
                                trace, records = layer_builder.records(textpage, page_index)
                            char_count = textpage.count_chars()
                            if reject_invisible_text_layer and char_count:
                                try:
                                    invisible_text_layer = _is_invisible_text_layer_page(
                                        page, textpage, crop_box
                                    )
                                except Exception as exc:  # noqa: BLE001 - keeps the text layer
                                    component_errors[f"invisible_text_layer:{page_index}"] = (
                                        _error_text(exc)
                                    )
                            if fill_native_text and not invisible_text_layer:
                                try:
                                    _sample_page_font_metadata(
                                        textpage, crop_box, char_count, regions, rotation
                                    )
                                    _fill_page_regions_from_textpage(
                                        textpage,
                                        crop_box,
                                        regions,
                                        min_chars=min_chars,
                                        eligible_labels=eligible_labels,
                                        min_printable_ratio=min_printable_ratio,
                                        page_idx=page_index,
                                        rotation=rotation,
                                        records=records,
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    component_errors[f"native_text:{page_index}"] = _error_text(exc)
                            elif fill_native_text:
                                try:
                                    _keep_layer_text_as_ocr_fallback(
                                        textpage,
                                        crop_box,
                                        regions,
                                        min_chars=min_chars,
                                        eligible_labels=eligible_labels,
                                        min_printable_ratio=min_printable_ratio,
                                        page_idx=page_index,
                                        rotation=rotation,
                                        records=records,
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    component_errors[f"native_text:{page_index}"] = _error_text(exc)
                            if include_ref_geometry and not invisible_text_layer:
                                try:
                                    full_text = textpage.get_text_range() or ""
                                    if header_page is None and _REF_HEADER_RE.search(full_text):
                                        header_page = page_index
                                    chars = _extract_page_chars(textpage)
                                    if header_page is not None:
                                        page_lines[page_index] = group_chars_into_lines(
                                            chars, page_index
                                        )
                                    stream_lines.extend(
                                        _page_line_dicts(
                                            group_chars_into_lines(
                                                _break_wrapped_lines(chars), page_index
                                            ),
                                            page_index + 1,
                                            crop_box,
                                            rotation,
                                        )
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    component_errors[f"ref_geometry:{page_index}"] = _error_text(
                                        exc
                                    )
                            if layer_builder is not None:
                                _harvest_page(
                                    layer_builder,
                                    page,
                                    textpage,
                                    page_index,
                                    crop_box,
                                    rotation,
                                    trace,
                                    records,
                                    furniture,
                                )
                        finally:
                            textpage.close()
                    elif layer_builder is not None:
                        # Nothing else reads this page's text layer; the layer
                        # still does, through the same furniture strip.
                        _harvest_unread_page(layer_builder, page, page_index, crop_box, rotation)
                    if include_ref_geometry:
                        try:
                            uri_links.extend(
                                _uri_link_dicts(
                                    page_uri_links(doc, page, page_index),
                                    page_index + 1,
                                    crop_box,
                                    rotation,
                                )
                            )
                        except Exception as exc:  # noqa: BLE001 - best-effort component
                            component_errors[f"uri_links:{page_index}"] = _error_text(exc)
                    pages.append(
                        PdfPageInspection(
                            index=page_index,
                            width=float(width),
                            height=float(height),
                            crop_box=crop_box,
                            char_count=char_count,
                            invisible_text_layer=invisible_text_layer,
                            watermarks=tuple(watermarks),
                        )
                    )
                finally:
                    page.close()

            if include_outline:
                try:
                    outline = _walk_pdfium_outline(doc, page_heights=page_heights)
                except Exception as exc:  # noqa: BLE001
                    component_errors["outline"] = _error_text(exc)
        finally:
            doc.close()

    if layer_builder is not None:
        # Spans, lines and tags need no pdfium: built after the lock is released.
        try:
            document = layer_builder.finish()
        except Exception:  # noqa: BLE001 - the layer is optional
            logger.warning("Could not finish the document layer", exc_info=True)

    verification_text = first_page_text
    if not verification_text and layouts:
        verification_text = "\n".join(str(region.get("content") or "") for region in layouts[0])
    metadata = harvest_docinfo(docinfo, verification_text)
    reference_lines = [
        record_to_dict(record) for record in reference_lines_from_pages(page_lines, header_page)
    ]
    return PdfInspection(
        pages=tuple(pages),
        layout_results=layouts,
        metadata=metadata,
        outline=outline,
        reference_lines=reference_lines,
        component_errors=component_errors,
        page_lines=stream_lines,
        uri_links=uri_links,
        document=document,
    )


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


def _layer_builder(pdf_bytes: bytes, render_budget: RenderBudget | None) -> LayerBuilder | None:
    try:
        from bibr.document.harvest import LayerBuilder

        return LayerBuilder(pdf_bytes, render_budget)
    except Exception:  # noqa: BLE001 - the layer is optional
        logger.warning("Could not start the document layer", exc_info=True)
        return None


def _harvest_page(
    builder: LayerBuilder,
    page,
    textpage,
    page_index: int,
    crop_box: tuple[float, float, float, float],
    rotation: int,
    trace,
    records,
    furniture: list[tuple],
) -> None:
    """Add a page to the document layer; a failure stays in the layer's errors."""
    try:
        builder.add_page(
            page,
            textpage,
            page_index=page_index,
            crop_box=crop_box,
            rotation=rotation,
            trace=trace,
            records=records,
            furniture=furniture,
        )
    except Exception as exc:  # noqa: BLE001
        builder.error(f"harvest:{page_index}", exc)


def _harvest_unread_page(
    builder: LayerBuilder,
    page,
    page_index: int,
    crop_box: tuple[float, float, float, float],
    rotation: int,
) -> None:
    """Harvest a page the inspection itself does not read the text layer of."""
    try:
        furniture: list[tuple] = []
        textpage = open_text_page(page, [], furniture)
        try:
            trace, records = builder.records(textpage, page_index)
            _harvest_page(
                builder, page, textpage, page_index, crop_box, rotation, trace, records, furniture
            )
        finally:
            textpage.close()
    except Exception as exc:  # noqa: BLE001
        builder.error(f"harvest:{page_index}", exc)


def _keep_layer_text_as_ocr_fallback(textpage, crop_box, regions: list[dict], **fill) -> None:
    """Keep a flagged scan page's layer text as each region's OCR fallback.

    The page is read by OCR, but its invisible layer is what the native fill
    read before: the text the fill would take goes into
    ``_invisible_layer_text``, and OcrStage uses it only when OCR returns
    nothing for the page (an outage), so the page is never blanker than the
    layer. The fill runs on copies, so the regions stay unfilled.
    """
    candidates = [dict(region) for region in regions]
    _fill_page_regions_from_textpage(textpage, crop_box, candidates, **fill)
    for region, candidate in zip(regions, candidates, strict=True):
        if candidate.get("_native_text_used"):
            region["_invisible_layer_text"] = candidate["content"]


def _break_wrapped_lines(
    chars: list[tuple[str, tuple[float, float, float, float]]],
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Insert a line break where the text wraps to a new baseline without one.

    PDFium joins a line that ends in a hyphen (which it reads as U+FFFE) to
    the next, and on line-numbered manuscripts the next line's margin number
    sits between the two halves of the word. A glyph wholly below the
    previous glyph and to its left starts a new printed line; U+FFFE is read
    as the hyphen it prints.
    """
    out: list[tuple[str, tuple[float, float, float, float]]] = []
    previous: tuple[float, float, float, float] | None = None
    for char, box in chars:
        if char in ("\n", "\r"):
            previous = None
        else:
            if char == "\ufffe":
                char = "-"
            if not char.isspace():
                if previous is not None and box[3] <= previous[1] and box[0] < previous[0]:
                    out.append(("\n", box))
                previous = box
        out.append((char, box))
    return out


def _layout_box(
    box_pts: tuple[float, float, float, float],
    crop_box: tuple[float, float, float, float],
    rotation: int,
) -> list[float]:
    return [round(v, 2) for v in _pdf_points_to_normalized_bbox(box_pts, crop_box, rotation)]


def _page_line_dicts(
    records: list[LineRecord],
    page_number: int,
    crop_box: tuple[float, float, float, float],
    rotation: int,
) -> list[dict[str, Any]]:
    """Serialize one page's text-layer lines for the reference line stream."""
    lines: list[dict[str, Any]] = []
    for record in records:
        line = {
            key: round(value, 2) if isinstance(value, float) else value
            for key, value in record_to_dict(record).items()
        }
        line["page"] = page_number
        line["rotation"] = rotation
        line["bbox"] = _layout_box(
            (record.x0, record.y_bottom, record.x1, record.y_top), crop_box, rotation
        )
        lines.append(line)
    return lines


def _uri_link_dicts(
    links: list[PdfUriLink],
    page_number: int,
    crop_box: tuple[float, float, float, float],
    rotation: int,
) -> list[dict[str, Any]]:
    return [
        {"page": page_number, "bbox": _layout_box(link.rect, crop_box, rotation), "uri": link.uri}
        for link in links
    ]


def inspection_to_dict(inspection: PdfInspection | None) -> dict[str, Any] | None:
    """Serialize stable detached inspection fields without duplicating layouts."""
    if inspection is None:
        return None
    return {
        "pages": [asdict(page) for page in inspection.pages],
        "component_errors": dict(inspection.component_errors),
    }


def inspection_from_dict(
    data: dict[str, Any] | None,
    *,
    metadata: dict[str, Any] | None,
    outline: list[OutlineItem] | None,
    reference_lines: list[dict[str, Any]] | None,
) -> PdfInspection | None:
    """Restore a cache-safe inspection using the artifact bundle's data."""
    if data is None:
        return None
    return PdfInspection(
        pages=tuple(
            PdfPageInspection(
                **{
                    **page,
                    "crop_box": tuple(page["crop_box"]),
                    "watermarks": tuple(page.get("watermarks", ())),
                }
            )
            for page in data.get("pages", [])
        ),
        layout_results=[],
        metadata=metadata or {},
        outline=outline or [],
        reference_lines=reference_lines or [],
        component_errors=dict(data.get("component_errors", {})),
    )
