"""The Ollama adapter talks to Ollama's OpenAI-compatible API under ``/v1``.

``LLM_OLLAMA_BASE_URL`` defaults to the bare server URL, and ``bibr setup``
writes the same form. The adapter passed it to Instructor unchanged, so every
request went to ``/chat/completions``, which Ollama answers with 404, and every
paper failed at its first LLM call.
"""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import BaseModel


class _Reply(BaseModel):
    reply: str


@pytest.fixture
def ollama_like_server():
    """A server that, like Ollama, serves chat completions only at ``/v1``."""
    paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            paths.append(self.path)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path != "/v1/chat/completions":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"404 page not found")
                return
            tools = body.get("tools") or []
            arguments = json.dumps({"reply": "OK"})
            message: dict = {"role": "assistant", "content": arguments}
            if tools:
                name = tools[0]["function"]["name"]
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            payload = json.dumps(
                {
                    "id": "chat-1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):  # noqa: A002, ARG002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", paths
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("suffix", ["", "/", "/v1", "/v1/"])
def test_ollama_adapter_posts_to_the_v1_api(ollama_like_server, suffix):
    from bibr.clients.providers.ollama import OllamaProvider
    from bibr.config import snapshot_settings

    host, paths = ollama_like_server
    settings = snapshot_settings()
    settings.llm.provider = "ollama"
    settings.llm.model = "gpt-oss:20b"
    settings.llm.ollama_base_url = host + suffix
    provider = OllamaProvider(settings=settings)
    client = provider.build_client()

    async def call():
        return await client.create(
            response_model=_Reply,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_retries=1,
            **provider.call_kwargs(None),
        )

    result = asyncio.run(call())

    assert result.reply == "OK"
    assert paths == ["/v1/chat/completions"]


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("http://localhost:11434", "http://localhost:11434/v1"),
        ("http://localhost:11434/", "http://localhost:11434/v1"),
        ("http://localhost:11434/v1", "http://localhost:11434/v1"),
        ("http://localhost:11434/v1/", "http://localhost:11434/v1"),
        (" http://gpu-box/ollama ", "http://gpu-box/ollama/v1"),
    ],
)
def test_ollama_openai_base_url(configured, expected):
    from bibr.clients.providers.ollama import ollama_openai_base_url

    assert ollama_openai_base_url(configured) == expected


def _ollama_settings(base_url):
    from bibr.config import snapshot_settings

    settings = snapshot_settings()
    settings.llm.provider = "ollama"
    settings.llm.model = "gpt-oss:20b"
    settings.llm.ollama_base_url = base_url
    return settings


def test_ping_llm_reaches_ollama_through_the_adapter(ollama_like_server):
    """``bibr doctor`` and the ``bibr setup`` connection test send this request."""
    from bibr.clients.llm import ping_llm

    host, paths = ollama_like_server

    assert ping_llm(_ollama_settings(host)) == "OK"
    assert paths == ["/v1/chat/completions"]


@pytest.fixture
def silent_server():
    """Accepts a request and never answers, like a wedged server or a model still loading."""
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            release.wait(30)

        def log_message(self, format, *args):  # noqa: A002, ARG002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_ping_llm_gives_up_where_chew_would(silent_server):
    """The SDK waits minutes for a reply; the ping stops at chew's limit for one call."""
    from bibr.clients.llm import ping_llm

    settings = _ollama_settings(silent_server)
    settings.llm.timeout_seconds = 1

    started = time.monotonic()
    with pytest.raises(TimeoutError) as excinfo:
        ping_llm(settings)

    assert time.monotonic() - started < 10
    assert str(excinfo.value) == (
        "No reply within 2 s, the limit bibr chew sets for one LLM call "
        "(twice LLM_TIMEOUT_SECONDS=1). Ollama loads the model on its first request, "
        "which can take longer; try again once it has loaded."
    )
