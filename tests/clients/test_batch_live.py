import pytest

from bibr.clients.batch import AnthropicBatchAdapter, BatchRequest
from bibr.config import Settings
from bibr.schemas import PaperTypeLabel

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not Settings.ANTHROPIC_API_KEY, reason="no ANTHROPIC_API_KEY")
def test_live_two_item_batch():
    reqs = [
        BatchRequest.from_spec(
            "empirical-1",
            "paper_type_label",
            title="Effect of sleep on memory: a randomized controlled trial",
            abstract="We collected data from 120 participants across two conditions...",
        ),
        BatchRequest.from_spec(
            "review-1",
            "paper_type_label",
            title="A narrative review of sleep and cognition",
            abstract="We synthesize prior literature on sleep and cognition without pooled statistics...",
        ),
    ]
    adapter = AnthropicBatchAdapter(model=Settings.llm.batch_model)
    out = adapter.run(reqs, poll_interval=10)

    assert set(out) == {"empirical-1", "review-1"}
    for res in out.values():
        assert isinstance(res, PaperTypeLabel), res
        assert res.paper_type is not None
