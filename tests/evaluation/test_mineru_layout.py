from unittest.mock import AsyncMock

import pytest

from bibr.layout.mineru import MinerULayoutDetector, adapt_layout


@pytest.mark.asyncio
async def test_layout_only_adapter_preserves_order_and_does_not_use_prose():
    blocks = [
        {"type": "table", "bbox": [0.1, 0.2, 0.5, 0.6], "content": "untrusted prefill"},
        {"type": "equation", "bbox": [0.2, 0.7, 0.4, 0.8], "angle": 90},
    ]

    class LayoutOnlyClient:
        aio_batch_layout_detect = AsyncMock(return_value=[blocks])

    detector = MinerULayoutDetector(LayoutOnlyClient(), model_revision="pinned")
    pages = await detector.detect_batch([object()])
    assert [r["task_type"] for r in pages[0]] == ["table", "formula"]
    assert pages[0][0]["bbox_2d"] == [100, 200, 500, 600]
    assert all(r["content"] == "" for r in pages[0])
    assert pages[0][1]["_layout_proposal"]["angle"] == 90
    assert detector.raw_results == [blocks]


@pytest.mark.parametrize(
    "block",
    [
        {"type": "unknown", "bbox": [0, 0, 1, 1]},
        {"type": "text", "bbox": [0, 0, 1000, 1000]},
        {"type": "text", "bbox": [0, 0, float("nan"), 1]},
    ],
)
def test_unsupported_proposals_are_not_silently_dropped(block):
    with pytest.raises(ValueError):
        adapt_layout([block])


@pytest.mark.asyncio
async def test_page_cardinality_is_checked():
    class Client:
        aio_batch_layout_detect = AsyncMock(return_value=[])

    with pytest.raises(ValueError, match="number of pages"):
        await MinerULayoutDetector(Client(), model_revision="pinned").detect_batch([object()])
