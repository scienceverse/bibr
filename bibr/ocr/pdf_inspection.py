"""One-open, detached inspection of born-digital PDF data."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from bibr.input.pdf_metadata import harvest_docinfo
from bibr.input.pdf_outline import OutlineItem, _walk_pdfium_outline
from bibr.ocr.native_source import account_characters, center_records, extract_characters
from bibr.ocr.native_text import (
    DEFAULT_ELIGIBLE_LABELS,
    _attach_bbox_pdf_pts,
    _attach_page_dimensions,
    _fill_page_regions_from_textpage,
    _page_crop_box,
    _sample_page_font_metadata,
)
from bibr.ocr.ref_geometry import (
    LineRecord,
    group_chars_into_lines,
    record_to_dict,
    reference_lines_from_pages,
)
from bibr.ocr.ref_patterns import _REF_HEADER_RE
from bibr.ocr.utils import pdfium_lock


@dataclass(frozen=True)
class PdfPageInspection:
    index: int
    width: float
    height: float
    crop_box: tuple[float, float, float, float]
    char_count: int
    source: dict[str, Any] | None = None


@dataclass(frozen=True)
class PdfInspection:
    pages: tuple[PdfPageInspection, ...]
    layout_results: list[list[dict[str, Any]]]
    metadata: dict[str, Any]
    outline: list[OutlineItem]
    reference_lines: list[dict[str, Any]]
    component_errors: dict[str, str] = field(default_factory=dict)


@dataclass
class PdfInspectionAccumulator:
    """Detached document context across page windows; never owns images or PDFium handles."""

    pages: list[PdfPageInspection] = field(default_factory=list)
    page_heights: dict[int, float] = field(default_factory=dict)
    page_lines: dict[int, list[LineRecord]] = field(default_factory=dict)
    header_page: int | None = None
    verification_text: str | None = None
    component_errors: dict[str, str] = field(default_factory=dict)

    def reference_lines(self) -> list[dict[str, Any]]:
        # Furniture detection must see the whole bibliography, including
        # continuations in a later window with no repeated References heading.
        return [
            record_to_dict(record)
            for record in reference_lines_from_pages(self.page_lines, self.header_page)
        ]


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
    accumulator: PdfInspectionAccumulator | None = None,
    source_accounting: bool = True,
    native_repair: bool = False,
    native_captions: bool = False,
) -> PdfInspection:
    """Inspect a PDF under one lock/open and return plain detached values."""
    import pypdfium2

    layouts = deepcopy(layout_results)
    state = accumulator if accumulator is not None else PdfInspectionAccumulator()
    pages = state.pages
    component_errors = state.component_errors
    page_lines = state.page_lines
    header_page = state.header_page
    docinfo: dict[str, Any] = {}
    outline: list[OutlineItem] = []

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

            page_heights = state.page_heights
            for layout_slot, page_index in page_pairs:
                page = doc[page_index]
                try:
                    width, height = page.get_size()
                    page_heights[page_index] = float(height)
                    crop_box = _page_crop_box(page)
                    regions = layouts[layout_slot] if layout_slot < len(layouts) else []
                    _attach_page_dimensions(page, regions)
                    _attach_bbox_pdf_pts(crop_box, regions)

                    source = None
                    needs_text = fill_native_text or include_ref_geometry or source_accounting
                    char_count = 0
                    if needs_text:
                        textpage = page.get_textpage()
                        try:
                            char_count = textpage.count_chars()
                            chars, fonts = extract_characters(textpage)
                            if source_accounting:
                                source = account_characters(
                                    chars, fonts, regions, crop_box, page_index, page.get_rotation()
                                )
                            if fill_native_text:
                                try:
                                    _sample_page_font_metadata(
                                        textpage, crop_box, char_count, regions
                                    )
                                    _fill_page_regions_from_textpage(
                                        textpage,
                                        crop_box,
                                        regions,
                                        min_chars=min_chars,
                                        eligible_labels=eligible_labels,
                                        min_printable_ratio=min_printable_ratio,
                                        page_idx=page_index,
                                        records=center_records(chars),
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    component_errors[f"native_text:{page_index}"] = _error_text(exc)
                            if fill_native_text and native_repair and source is not None:
                                from bibr.ocr.native_repair import plan_native_repairs

                                plan_native_repairs(
                                    regions,
                                    source,
                                    min_chars=min_chars,
                                    min_printable_ratio=min_printable_ratio,
                                    native_captions=native_captions,
                                )
                            if include_ref_geometry:
                                try:
                                    full_text = "".join(c.text for c in chars)
                                    if header_page is None and _REF_HEADER_RE.search(full_text):
                                        header_page = page_index
                                    if header_page is not None:
                                        page_lines[page_index] = group_chars_into_lines(
                                            [(c.text, c.bbox or (0, 0, 0, 0)) for c in chars],
                                            page_index,
                                        )
                                except Exception as exc:  # noqa: BLE001
                                    component_errors[f"ref_geometry:{page_index}"] = _error_text(
                                        exc
                                    )
                        finally:
                            textpage.close()
                    pages.append(
                        PdfPageInspection(
                            index=page_index,
                            width=float(width),
                            height=float(height),
                            crop_box=crop_box,
                            char_count=char_count,
                            source=source,
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

    verification_text = first_page_text
    if not verification_text and layouts:
        verification_text = "\n".join(str(region.get("content") or "") for region in layouts[0])
    if state.verification_text is None:
        state.verification_text = verification_text
    state.header_page = header_page
    metadata = harvest_docinfo(docinfo, state.verification_text)
    # The page-window owner finalizes geometry once, after all windows finish.
    reference_lines = state.reference_lines() if accumulator is None else []
    return PdfInspection(
        pages=tuple(pages),
        layout_results=layouts,
        metadata=metadata,
        outline=outline,
        reference_lines=reference_lines,
        component_errors=component_errors,
    )


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


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
            PdfPageInspection(**{**page, "crop_box": tuple(page["crop_box"])})
            for page in data.get("pages", [])
        ),
        layout_results=[],
        metadata=metadata or {},
        outline=outline or [],
        reference_lines=reference_lines or [],
        component_errors=dict(data.get("component_errors", {})),
    )
