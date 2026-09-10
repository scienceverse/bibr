"""Experimental layout-only adapter. The optional MinerU runtime is caller-owned.

This adapter deliberately exposes no transcription methods. Use the normal bibr
native inspection and OCR stages after detection, with identical policies when
comparing providers. It is not registered as a production backend.
"""

from __future__ import annotations

import math
from copy import deepcopy

from bibr.layout_utils import _LABEL_TO_TASK

_LABELS = {
    "text": "text",
    "title": "paragraph_title",
    "doc_title": "doc_title",
    "paragraph_title": "paragraph_title",
    "ref_text": "reference_content",
    "equation": "display_formula",
    "equation_block": "display_formula",
    "formula_number": "formula_number",
    "table": "table",
    "image": "image",
    "image_block": "image",
    "chart": "chart",
    "code": "algorithm",
    "algorithm": "algorithm",
    "list": "text",
    "list_item": "text",
    "index": "text",
    "phonetic": "text",
    "table_caption": "table_title",
    "image_caption": "figure_title",
    "code_caption": "figure_title",
    "caption": "figure_title",
    "table_footnote": "footnote",
    "image_footnote": "footnote",
    "footnote": "footnote",
    "page_footnote": "footnote",
    "header": "header",
    "footer": "footer",
    "page_number": "number",
    "aside_text": "aside_text",
}


def adapt_layout(blocks):
    """Validate normalized geometry, retaining provider labels and raw proposals."""
    regions = []
    for index, block in enumerate(blocks):
        kind = block["type"]
        if kind not in _LABELS:
            raise ValueError(f"Unsupported MinerU layout type: {kind!r}")
        box = block["bbox"]
        if (
            len(box) != 4
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in box)
            or box[0] >= box[2]
            or box[1] >= box[3]
        ):
            raise ValueError(f"Invalid MinerU normalized box: {box!r}")
        label = _LABELS[kind]
        regions.append(
            {
                "index": index,
                "label": label,
                "bbox_2d": [v * 1000 for v in box],
                "task_type": _LABEL_TO_TASK.get(label, "text"),
                "content": "",
                "_layout_provider": "mineru",
                "_layout_proposal": deepcopy(dict(block)),
            }
        )
    return regions


class MinerULayoutDetector:
    """Implement bibr's batch layout contract with an injected MinerUClient."""

    def __init__(self, client, *, model_revision: str):
        if not model_revision:
            raise ValueError("A model revision is required for reproducible layout evaluation")
        self.client = client
        self.model_revision = model_revision
        self.raw_results = []

    async def detect_batch(self, images):
        if not images:
            return []
        results = await self.client.aio_batch_layout_detect(images)
        if len(results) != len(images):
            raise ValueError("MinerU returned a different number of pages")
        self.raw_results = deepcopy(results)
        return [adapt_layout(page) for page in results]
