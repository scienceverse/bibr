"""Integration test: Settings.ocr.native_text_enabled causes the local pipeline
to pre-fill text regions from the PDF's native text layer, and the OCR stage
skips those regions.
"""

from pathlib import Path

import pytest

from bibr.config import Settings
from bibr.pipeline.stages.ocr import ocr_page_regions

FIXTURE_NATIVE = Path(__file__).parent / "fixtures" / "native_text_sample.pdf"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_local_pipeline_hook_populates_native_text(monkeypatch):
    """
    This is NOT a full LocalPipeline integration test (that requires OCR
    backend state). Instead, it verifies the *hook behavior*: with the flag
    enabled, fill_regions_from_native_text runs against the FileState and
    subsequent ocr_page_regions calls skip the pre-filled regions.
    """
    from bibr.ocr.native_text import fill_regions_from_native_text

    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)
    pdf_bytes = FIXTURE_NATIVE.read_bytes()

    # Synthesize a layout result that the pipeline would normally produce.
    # One full-page text region — should get pre-filled because the fixture
    # has native text.
    layout_results = [
        [
            {
                "label": "text",
                "task_type": "text",
                "bbox_2d": [0, 0, 1000, 1000],
                "content": "",
            },
        ],
    ]

    # Run the fill pass (this is what Stage 4b does).
    fill_regions_from_native_text(
        pdf_bytes,
        layout_results,
        min_chars=Settings.ocr.native_text_min_chars,
    )

    assert layout_results[0][0]["_native_text_used"] is True
    assert "The quick brown fox" in layout_results[0][0]["content"]

    # Confirm that ocr_page_regions now skips the OCR call for this region.
    from PIL import Image

    page_img = Image.new("RGB", (1000, 1000), color="white")
    ocr_calls = []

    async def fake_ocr_fn(img, prompt):
        ocr_calls.append(prompt)
        return "should not be called"

    result = await ocr_page_regions(
        page_img,
        layout_results[0],
        page_idx=0,
        filename=FIXTURE_NATIVE.name,
        ocr_fn=fake_ocr_fn,
    )

    assert len(ocr_calls) == 0
    assert "The quick brown fox" in result[0]["content"]


def test_local_pipeline_feature_enabled_by_default():
    """Feature flag is on by default after the 2026-04-17 eval (81% median bypass
    on born-digital corpus).  Opt out by setting OCR_NATIVE_TEXT_ENABLED=False."""
    assert Settings.ocr.native_text_enabled is True
