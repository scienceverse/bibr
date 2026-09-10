"""Typed OCR region result.

Producers in ``bibr.pipeline.stages.ocr`` build region records that are
later consumed by ``bibr.structure.pdf_parser.PDFParser`` as plain dicts.
This module gives the producer side a typed seam: build an
``OcrRegionResult`` with explicit fields, then serialize via
``to_dict()`` to the legacy wire format the consumers expect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Layout-detector labels that should map to the merged "formula" label
# downstream — kept in sync with bibr.pipeline.stages.ocr._map_native_label.
_FORMULA_NATIVE_LABELS = frozenset({"display_formula", "inline_formula"})


def _map_native_label(native_label: str) -> str:
    if native_label in _FORMULA_NATIVE_LABELS:
        return "formula"
    return native_label


@dataclass
class OcrRegionResult:
    """A single OCR'd region in canonical reading order.

    Attributes mirror the legacy dict shape one-to-one. ``to_dict()`` emits
    the exact dict that downstream consumers (PDFParser, postprocessor)
    already understand; missing optional values are omitted from the dict
    rather than emitted as None, matching prior producer behaviour.
    """

    index: int
    native_label: str
    label: str
    content: str
    bbox_2d: list[float] | None
    image_b64: str | None = None
    font_size: float | None = None
    font_weight: str | None = None
    font_bold: bool | None = None
    is_italic: bool | None = None
    page_w: float | None = None
    page_h: float | None = None
    # Containment-correct region bbox in PDF points (bottom-left origin, y-up),
    # the native-text-space counterpart of the 0..1000 image-space ``bbox_2d``.
    # Set by native_text._attach_bbox_pdf_pts; None when no bbox / no native
    # pass ran. PDFParser exports it as the sentence's ``_bbox_2d``.
    bbox_pdf_pts: list[float] | None = None
    # Set by NativeTextStage's fill_regions_from_native_text; carried through
    # so the OCR success gate can exclude regions that bypassed OCR entirely.
    native_text_used: bool | None = None
    # Original model response retained for diagnostics when normalization altered
    # it. Canonical ``content`` remains the sole downstream contract.
    raw_content: str | None = None
    # OpenAI-compatible completion stop reason retained for diagnostics.
    finish_reason: str | None = None
    # Rejected embedded-PDF text and its bounded machine-readable reason.  The
    # candidate is intentionally full-length so canonical OCR can be compared
    # against the exact alternate source after parsing or a cache round trip.
    native_text_candidate: str | None = None
    native_text_rejection_reason: str | None = None
    source_region_id: str | None = None
    source_region_ids: list[str] | None = None
    native_spans: list[dict[str, Any]] | None = None
    formula_proposals: list[dict[str, Any]] | None = None

    @classmethod
    def from_layout_region(
        cls,
        layout: dict,
        *,
        slot_idx: int,
        content: str,
        image_b64: str | None = None,
        raw_content: str | None = None,
        finish_reason: str | None = None,
    ) -> OcrRegionResult:
        """Construct from a raw layout-detector region dict.

        Mirrors the existing producer pattern: pull ``label`` and
        ``bbox_2d`` from the source, derive merged ``label``, and copy
        font metadata keys.
        """
        native_label = layout.get("label", "text")
        return cls(
            index=slot_idx,
            native_label=native_label,
            label=_map_native_label(native_label),
            content=content,
            bbox_2d=layout.get("bbox_2d"),
            image_b64=image_b64,
            font_size=layout.get("_font_size"),
            font_weight=layout.get("_font_weight"),
            font_bold=layout.get("_font_bold"),
            is_italic=layout.get("_is_italic"),
            page_w=layout.get("_page_w"),
            page_h=layout.get("_page_h"),
            bbox_pdf_pts=layout.get("_bbox_pdf_pts"),
            native_text_used=layout.get("_native_text_used"),
            raw_content=raw_content,
            finish_reason=finish_reason,
            native_text_candidate=layout.get("_native_text_candidate"),
            native_text_rejection_reason=layout.get("_native_text_rejection_reason"),
            source_region_id=layout.get("_source_region_id"),
            source_region_ids=layout.get("_source_region_ids")
            or ([layout["_source_region_id"]] if layout.get("_source_region_id") else None),
            native_spans=layout.get("_native_spans"),
            formula_proposals=layout.get("_formula_proposals"),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OcrRegionResult:
        """Load from the legacy wire-format dict (inverse of :meth:`to_dict`).

        Tolerant the way the old dict consumers' ``.get()`` reads were:
        missing keys take the consumer defaults (``""``/``0``/None), unknown
        keys are ignored, and an explicit ``content: None`` normalizes to
        ``""``.
        """
        return cls(
            index=d.get("index", 0),
            native_label=d.get("native_label", ""),
            label=d.get("label", ""),
            content=d.get("content") or "",
            bbox_2d=d.get("bbox_2d"),
            image_b64=d.get("image_b64"),
            font_size=d.get("_font_size"),
            font_weight=d.get("_font_weight"),
            font_bold=d.get("_font_bold"),
            is_italic=d.get("_is_italic"),
            page_w=d.get("_page_w"),
            page_h=d.get("_page_h"),
            bbox_pdf_pts=d.get("_bbox_pdf_pts"),
            native_text_used=d.get("_native_text_used"),
            raw_content=d.get("_raw_ocr_content"),
            finish_reason=d.get("_ocr_finish_reason"),
            native_text_candidate=d.get("_native_text_candidate"),
            native_text_rejection_reason=d.get("_native_text_rejection_reason"),
            source_region_id=d.get("_source_region_id"),
            source_region_ids=d.get("_source_region_ids"),
            native_spans=d.get("_native_spans"),
            formula_proposals=d.get("_formula_proposals"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Emit the legacy wire-format dict consumed by PDFParser."""
        out: dict[str, Any] = {
            "index": self.index,
            "native_label": self.native_label,
            "label": self.label,
            "content": self.content,
            "bbox_2d": self.bbox_2d,
        }
        if self.image_b64 is not None:
            out["image_b64"] = self.image_b64
        if self.font_size is not None:
            out["_font_size"] = self.font_size
        if self.font_weight is not None:
            out["_font_weight"] = self.font_weight
        if self.font_bold is not None:
            out["_font_bold"] = self.font_bold
        if self.is_italic is not None:
            out["_is_italic"] = self.is_italic
        if self.page_w is not None:
            out["_page_w"] = self.page_w
        if self.page_h is not None:
            out["_page_h"] = self.page_h
        if self.bbox_pdf_pts is not None:
            out["_bbox_pdf_pts"] = self.bbox_pdf_pts
        if self.native_text_used is not None:
            out["_native_text_used"] = self.native_text_used
        if self.raw_content is not None:
            out["_raw_ocr_content"] = self.raw_content
        if self.finish_reason is not None:
            out["_ocr_finish_reason"] = self.finish_reason
        if self.native_text_candidate is not None:
            out["_native_text_candidate"] = self.native_text_candidate
        if self.native_text_rejection_reason is not None:
            out["_native_text_rejection_reason"] = self.native_text_rejection_reason
        for key, value in (
            ("_source_region_id", self.source_region_id),
            ("_source_region_ids", self.source_region_ids),
            ("_native_spans", self.native_spans),
            ("_formula_proposals", self.formula_proposals),
        ):
            if value is not None:
                out[key] = value
        return out
