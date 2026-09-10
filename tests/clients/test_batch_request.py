from bibr.clients.batch import AnthropicBatchAdapter, BatchRequest, _tool_name
from bibr.schemas import PaperTypeLabel


def test_from_spec_resolves_prompt_and_schema():
    req = BatchRequest.from_spec(
        "paper-1", "paper_type_label", title="A Title", abstract="An abstract"
    )
    assert req.custom_id == "paper-1"
    assert req.schema is PaperTypeLabel
    assert req.system == "You are a scientific paper classifier."
    assert "A Title" in req.user_text
    assert "An abstract" in req.user_text


def test_build_request_forces_the_tool():
    req = BatchRequest.from_spec("paper-1", "paper_type_label", title="T", abstract="A")
    adapter = AnthropicBatchAdapter(model="claude-haiku-4-5", max_tokens=512)
    out = adapter._build_request(req)

    assert out["custom_id"] == "paper-1"
    params = out["params"]
    assert params["model"] == "claude-haiku-4-5"
    assert params["max_tokens"] == 512
    assert params["system"] == req.system
    assert params["messages"] == [{"role": "user", "content": req.user_text}]

    tool_name = _tool_name(PaperTypeLabel)
    assert tool_name == "PaperTypeLabel"
    assert params["tool_choice"] == {"type": "tool", "name": tool_name}
    assert len(params["tools"]) == 1
    tool = params["tools"][0]
    assert tool["name"] == tool_name
    # input_schema is sanitized strict-legal
    assert tool["input_schema"]["additionalProperties"] is False
    assert set(tool["input_schema"]["required"]) == {"paper_type", "confidence"}
    assert "title" not in tool["input_schema"]
