from types import SimpleNamespace

from bibr.clients.batch import AnthropicBatchAdapter, BatchError, BatchRequest
from bibr.schemas import PaperTypeLabel


def _succeeded(custom_id, payload):
    block = SimpleNamespace(type="tool_use", input=payload)
    message = SimpleNamespace(content=[block])
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="succeeded", message=message),
    )


def _errored(custom_id, err_type, msg):
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="errored", error=SimpleNamespace(type=err_type, message=msg)),
    )


class _FakeBatches:
    def __init__(self, rows):
        self._rows = rows

    def results(self, batch_id):
        return iter(self._rows)


class _FakeClient:
    def __init__(self, rows):
        self.messages = SimpleNamespace(batches=_FakeBatches(rows))


def _reqs():
    return [
        BatchRequest.from_spec("p1", "paper_type_label", title="t", abstract="a"),
        BatchRequest.from_spec("p2", "paper_type_label", title="t", abstract="a"),
    ]


def test_parses_succeeded_and_errored_keyed_by_custom_id_any_order():
    rows = [
        _errored("p2", "invalid_request", "bad"),
        _succeeded("p1", {"paper_type": "review", "confidence": 0.8}),
    ]
    adapter = AnthropicBatchAdapter(client=_FakeClient(rows))
    out = adapter.retrieve("batch_x", _reqs())

    assert isinstance(out["p1"], PaperTypeLabel)
    assert out["p1"].paper_type == "review"
    assert isinstance(out["p2"], BatchError)
    assert out["p2"].error_type == "invalid_request"


def test_ignores_custom_ids_not_in_requests():
    rows = [
        _succeeded("p1", {"paper_type": "review", "confidence": 0.8}),
        _succeeded("stale", {"paper_type": "empirical", "confidence": 0.9}),
    ]
    adapter = AnthropicBatchAdapter(client=_FakeClient(rows))
    out = adapter.retrieve("batch_x", _reqs()[:1])  # only p1 requested
    assert set(out) == {"p1"}
