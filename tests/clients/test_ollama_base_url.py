"""The Ollama adapter talks to Ollama's OpenAI-compatible API under ``/v1``.

``LLM_OLLAMA_BASE_URL`` defaults to the bare server URL, and ``bibr setup``
writes the same form. The adapter passed it to Instructor unchanged, so every
request went to ``/chat/completions``, which Ollama answers with 404, and every
paper failed at its first LLM call.
"""

import asyncio
import json
import threading
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
