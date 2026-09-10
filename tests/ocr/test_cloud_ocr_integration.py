"""Integration test — real API call to vision OCR backend.

Gated on env var and marked slow so it never runs in CI.
"""

import os

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("BIBR_RUN_VISION_OCR_INTEGRATION"),
        reason="Set BIBR_RUN_VISION_OCR_INTEGRATION=1 to run",
    ),
]


@pytest.mark.asyncio
async def test_gemini_recognizes_text():
    """Gemini can OCR a simple synthetic image."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 30), "Hello World 2026", fill="black", font=font)

    from bibr.local.ocr_cloud import CloudOcrClient

    client = CloudOcrClient(provider="gemini")
    result = await client.recognize(img, "Text Recognition:")
    await client.shutdown()

    assert "hello" in result.lower() or "2026" in result
