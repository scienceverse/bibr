import hashlib
import json
from types import SimpleNamespace

from bibr.clients.batch import AnthropicBatchAdapter, BatchRequest
from bibr.schemas import PaperTypeLabel


def _succeeded(custom_id, payload):
    block = SimpleNamespace(type="tool_use", input=payload)
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="succeeded", message=SimpleNamespace(content=[block])),
    )


class _FakeBatches:
    def __init__(self, rows):
        self._rows = rows
        self.create_calls = 0

    def create(self, requests):
        self.create_calls += 1
        return SimpleNamespace(id="batch_abc")

    def retrieve(self, batch_id):
        return SimpleNamespace(
            id=batch_id,
            processing_status="ended",
            request_counts=SimpleNamespace(
                processing=0, succeeded=1, errored=0, canceled=0, expired=0
            ),
        )

    def results(self, batch_id):
        return iter(self._rows)


class _FakeClient:
    def __init__(self, rows):
        self.batches = _FakeBatches(rows)
        self.messages = SimpleNamespace(batches=self.batches)


def _reqs():
    return [BatchRequest.from_spec("p1", "paper_type_label", title="t", abstract="a")]


def test_run_submits_writes_state_and_returns_results(tmp_path):
    rows = [_succeeded("p1", {"paper_type": "review", "confidence": 0.8})]
    client = _FakeClient(rows)
    adapter = AnthropicBatchAdapter(client=client)
    state = tmp_path / "state.json"

    out = adapter.run(_reqs(), state_path=state, poll_interval=0)

    assert isinstance(out["p1"], PaperTypeLabel)
    assert client.batches.create_calls == 1
    saved = json.loads(state.read_text())
    assert saved["batch_id"] == "batch_abc"
    assert "p1" in saved["manifest"]


def _manifest(requests):
    """The same digest ``run`` writes at submit time.

    Resume now verifies this before reconnecting, so a placeholder value no
    longer stands in for it.
    """
    return {r.custom_id: hashlib.sha256(r.user_text.encode()).hexdigest()[:16] for r in requests}


def test_run_reconnects_to_existing_batch_without_resubmitting(tmp_path):
    rows = [_succeeded("p1", {"paper_type": "review", "confidence": 0.8})]
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "provider": "anthropic",
                "model": "claude-haiku-4-5",
                "batch_id": "batch_abc",
                "manifest": _manifest(_reqs()),
            }
        )
    )

    client = _FakeClient(rows)
    adapter = AnthropicBatchAdapter(client=client)
    out = adapter.run(_reqs(), state_path=state, poll_interval=0)

    assert client.batches.create_calls == 0  # reconnected, did not resubmit
    assert isinstance(out["p1"], PaperTypeLabel)
