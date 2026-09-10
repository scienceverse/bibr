"""Tests for OCR page region helpers (formerly bibr.pipeline_helpers)."""

import asyncio

from PIL import Image

from bibr.pipeline.stages.ocr import ocr_page_regions


def test_ocr_page_regions_skips_prefilled_regions():
    """A region with _native_text_used=True must not invoke ocr_fn."""
    page_img = Image.new("RGB", (1000, 1000), color="white")
    regions = [
        {
            "label": "text",
            "task_type": "text",
            "bbox_2d": [0, 0, 500, 100],
            "content": "pre-filled body text",
            "_native_text_used": True,
        },
        {
            "label": "text",
            "task_type": "text",
            "bbox_2d": [0, 100, 500, 200],
            "content": "",
        },
    ]
    ocr_calls = []

    async def fake_ocr_fn(img, prompt):
        ocr_calls.append(prompt)
        return "ocr result"

    result = asyncio.run(
        ocr_page_regions(page_img, regions, page_idx=0, filename="x.pdf", ocr_fn=fake_ocr_fn)
    )

    # Pre-filled region preserved
    assert result[0]["content"] == "pre-filled body text"
    assert result[0]["native_label"] == "text"
    assert result[0]["label"] == "text"
    assert result[0]["bbox_2d"] == [0, 0, 500, 100]
    # Second region went through OCR
    assert result[1]["content"] == "ocr result"
    # Only one OCR call made
    assert len(ocr_calls) == 1
