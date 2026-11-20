"""Claude models that reject a forced tool call (Sonnet 5.5, Opus 5.5, Fable and Mythos 5.1/5.2).

Instructor forces its response tool (``tool_choice`` of type ``tool``) unless
the call names a ``tool_choice`` itself. These models answer that with a 400
("tool_choice: type "tool" and "any" are not supported for this model."), so
every call bibr made to them failed. Calls to them now ask for
``{"type": "auto"}``; calls to every other model still force the tool.
"""

import asyncio
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import BaseModel

from bibr.clients.batch import AnthropicBatchAdapter, BatchRequest
from bibr.clients.providers.anthropic import AnthropicProvider, rejects_forced_tool_choice
from bibr.config import GlobalSettings

# The Anthropic SDK ships in the batch extra, not in a core install.
needs_anthropic = pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is None, reason="needs the anthropic SDK"
)

NO_FORCED_TOOL = [
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "claude-fable-5-1",
    "claude-fable-5-2",
    "claude-mythos-5-1",
    "claude-mythos-5-2",
    "claude-sonnet-5-5-20261101",  # dated snapshot ids match by prefix
]

# Sonnet 5, Opus 5, Fable 5 and Haiku 5.5 reject a temperature but accept a forced tool.
FORCED_TOOL = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-haiku-5-5",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-fable-5",
]


class _Reply(BaseModel):
    reply: str


def _provider(model, **llm):
    settings = GlobalSettings(
        llm={"provider": "anthropic", "model": model, "api_key": "sk-test-key-0000", **llm}
    )
    return AnthropicProvider(settings=settings)


@pytest.mark.parametrize("model", NO_FORCED_TOOL)
def test_calls_leave_the_tool_choice_to_the_model(model):
    assert rejects_forced_tool_choice(model)
    provider = _provider(model)
    assert provider.call_kwargs(None, max_tokens=512)["tool_choice"] == {"type": "auto"}
    assert provider.call_kwargs("low")["tool_choice"] == {"type": "auto"}
    # A thinking budget is dropped for these models; the tool choice stays.
    provider = _provider(model, thinking_budget=4096)
    assert provider.call_kwargs(None, max_tokens=8192)["tool_choice"] == {"type": "auto"}


@pytest.mark.parametrize("model", FORCED_TOOL)
def test_other_models_keep_the_forced_tool(model):
    assert not rejects_forced_tool_choice(model)
    provider = _provider(model)
    # No tool_choice of bibr's own, so Instructor forces its response tool.
    assert "tool_choice" not in provider.call_kwargs(None, max_tokens=512)
    assert "tool_choice" not in provider.call_kwargs("low")


@pytest.fixture
def anthropic_api(monkeypatch):
    """A loopback Messages API that records each request and calls the offered tool."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            data = json.dumps(
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": body["tools"][0]["name"],
                            "input": {"reply": "OK"},
                        }
                    ],
                    "stop_reason": "tool_use",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):  # noqa: A002, ARG002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        yield requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _send(model):
    provider = _provider(model)
    client = provider.build_client()

    async def call():
        return await client.create(
            response_model=_Reply,
            messages=[{"role": "user", "content": "Reply OK"}],
            max_retries=1,
            **provider.call_kwargs(None, max_tokens=64),
        )

    return asyncio.run(call())


@needs_anthropic
def test_sonnet_5_5_request_carries_no_forced_tool(anthropic_api):
    reply = _send("claude-sonnet-5-5")

    assert reply.reply == "OK"
    (body,) = anthropic_api
    assert body["tool_choice"] == {"type": "auto"}
    assert "temperature" not in body


@needs_anthropic
def test_sonnet_5_request_carries_no_temperature(anthropic_api):
    reply = _send("claude-sonnet-5")

    assert reply.reply == "OK"
    (body,) = anthropic_api
    assert "temperature" not in body
    assert "thinking" not in body
    assert body["tool_choice"]["type"] == "tool"  # Sonnet 5 still accepts the forced tool


@needs_anthropic
def test_haiku_4_5_request_is_unchanged(anthropic_api):
    _send("claude-haiku-4-5")

    (body,) = anthropic_api
    assert body["temperature"] == 0.0
    assert body["tool_choice"]["type"] == "tool"


@pytest.mark.parametrize("model", NO_FORCED_TOOL)
def test_batch_request_leaves_the_tool_choice_to_the_model(model):
    req = BatchRequest.from_spec("paper-1", "paper_type_label", title="T", abstract="A")
    params = AnthropicBatchAdapter(model=model, max_tokens=512)._build_request(req)["params"]

    assert params["tool_choice"] == {"type": "auto"}
    # Nothing forces the call any more, so the system prompt asks for it by name.
    assert params["system"].startswith(req.system)
    assert "PaperTypeLabel tool" in params["system"][len(req.system) :]
    (tool,) = params["tools"]
    assert tool["name"] == "PaperTypeLabel"
    assert tool["strict"] is True  # the schema guarantee does not depend on forcing


@pytest.mark.parametrize("model", FORCED_TOOL)
def test_batch_request_still_forces_the_tool_elsewhere(model):
    req = BatchRequest.from_spec("paper-1", "paper_type_label", title="T", abstract="A")
    params = AnthropicBatchAdapter(model=model, max_tokens=512)._build_request(req)["params"]

    assert params["tool_choice"] == {"type": "tool", "name": "PaperTypeLabel"}
    assert params["system"] == req.system
