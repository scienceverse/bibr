"""Capture the exact paragraph strings sent from PDFParser to wtpsplit."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from bibr.structure.pdf_parser import PDFParser


@dataclass(frozen=True)
class CapturedParagraph:
    """One segmentable deferred parser entry with source-region provenance."""

    text: str
    deferred_index: int
    section_id: int
    page_numbers: tuple[int, ...]
    source_region_indices: tuple[tuple[int, int], ...]


def _bbox_key(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    return tuple(float(part) for part in value)


def capture_segmenter_inputs(ocr_regions: list[list[dict]]) -> list[CapturedParagraph]:
    """Run the real PDF parser and capture its pre-segmentation input seam.

    Page numbers are one-based. Region indices are the source record's own
    ``index`` values, paired with their page so they remain unambiguous.
    """

    parser = PDFParser(ocr_regions)
    parser.parse()

    by_location: dict[
        tuple[int, tuple[float, float, float, float] | None], deque[tuple[int, str]]
    ] = defaultdict(deque)
    for page_number, page in enumerate(parser.json_result, start=1):
        for region in page:
            by_location[(page_number, _bbox_key(region.bbox_2d))].append(
                (region.index, region.content)
            )

    captured: list[CapturedParagraph] = []
    for deferred_index, entry in enumerate(parser.assembler.entries):
        if not entry.needs_segmentation:
            continue

        source_indices: list[tuple[int, int]] = []
        for provenance in entry.provenance:
            key = (provenance.page_no, _bbox_key(provenance.bbox))
            candidates = by_location[key]
            if not candidates:
                raise ValueError(
                    "Missing segmenter capture provenance for "
                    f"page={provenance.page_no}, bbox={provenance.bbox}"
                )

            matching = [
                position
                for position, (_, content) in enumerate(candidates)
                if content.strip() and content.strip() in entry.text
            ]
            if len(candidates) == 1:
                chosen = 0
            elif len(matching) == 1:
                chosen = matching[0]
            else:
                raise ValueError(
                    "Ambiguous segmenter capture provenance for "
                    f"page={provenance.page_no}, bbox={provenance.bbox}"
                )
            candidates.rotate(-chosen)
            region_index, _ = candidates.popleft()
            candidates.rotate(chosen)
            source_indices.append((provenance.page_no, region_index))

        page_numbers = tuple(dict.fromkeys(page for page, _ in source_indices))
        if not page_numbers and entry.page_number is not None:
            page_numbers = (entry.page_number,)
        captured.append(
            CapturedParagraph(
                text=entry.text,
                deferred_index=deferred_index,
                section_id=entry.section_id,
                page_numbers=page_numbers,
                source_region_indices=tuple(source_indices),
            )
        )

    return captured
