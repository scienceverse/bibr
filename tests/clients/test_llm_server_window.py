"""A context window bibr did not know, read from the server when an answer stops at it.

vLLM 0.31 answered a prompt whose completion cap ran past the window with HTTP
400 and named the window, and bibr asked again with the input cut to fit. vLLM
0.32 lowers the cap to what is left instead: the answer stops at the window
with ``finish_reason`` ``length``, and the title/keywords call failed as
truncated (E29r's stub). bibr now reads the window from ``/v1/models`` then.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from bibr.clients import server_window
from bibr.clients.llm import LLMClient
from bibr.config import GlobalSettings
from bibr.exceptions import LlmInvalidOutputError, LlmTruncatedError

WINDOW = 8_192
ANSWER = json.dumps(
    {
        "title": "A study of reading rooms",
        "abstract": " ".join(["We describe the reading rooms of public libraries in detail."] * 50),
        "keywords": ["libraries", "reading rooms"],
    }
)
WORD = "library "  # 2 tokens to the stub


def _tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


class _Stub:
    """An OpenAI-compatible server that answers like vLLM 0.31 or 0.32.

    A token is 4 characters of the messages and the response format. A prompt
    past the window alone is rejected by both versions. When prompt plus the
    completion cap runs past it, 0.31 rejects the request and 0.32 lowers the
    cap to what is left. ``server_cap`` is a limit of the server's own, such as
    a generation config's ``max_new_tokens``.
    """

    def __init__(
        self,
        version: str,
        *,
        lists_window: bool = True,
        server_cap: int | None = None,
        usage: bool = True,
        answer: str = ANSWER,
        second_answer_delay: float = 0.0,
    ):
        self.version = version
        self.lists_window = lists_window
        self.server_cap = server_cap
        self.usage = usage
        self.answer = answer
        self.second_answer_delay = second_answer_delay
        self.log: list[dict] = []

    def models(self) -> tuple[int, dict]:
        self.log.append({"route": "models"})
        entry: dict = {"id": "stub", "object": "model"}
        if self.lists_window:
            entry["max_model_len"] = WINDOW
        return 200, {"object": "list", "data": [entry]}

    def chat(self, body: dict) -> tuple[int, dict]:
        text = "".join(
            m["content"] if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
            for m in body.get("messages", [])
        )
        prompt = _tokens(text) + _tokens(json.dumps(body.get("response_format") or ""))
        cap = body.get("max_tokens") or body.get("max_completion_tokens") or WINDOW - prompt
        record = {"route": "chat", "prompt_tokens": prompt, "max_tokens": cap}
        self.log.append(record)
        if self.second_answer_delay and self.routes().count("chat") == 2:
            time.sleep(self.second_answer_delay)
        if prompt > WINDOW or (prompt + cap > WINDOW and self.version == "0.31"):
            record["status"] = 400
            return 400, {
                "error": {
                    "message": f"This model's maximum context length is {WINDOW} tokens. "
                    f"However, you requested {cap} output tokens and your prompt contains at "
                    f"least {prompt} input tokens.",
                    "type": "BadRequestError",
                    "code": 400,
                }
            }
        room = min(cap, WINDOW - prompt, self.server_cap or WINDOW)
        if room < _tokens(self.answer):
            content, finish, done = self.answer[: room * 4], "length", room
        else:
            content, finish, done = self.answer, "stop", _tokens(self.answer)
        record.update(status=200, finish_reason=finish, completion_tokens=done)
        usage = {"prompt_tokens": prompt, "completion_tokens": done, "total_tokens": prompt + done}
        return 200, {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "stub",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish,
                }
            ],
            **({"usage": usage} if self.usage else {}),
        }

    def routes(self) -> list[str]:
        return [r["route"] for r in self.log]


@pytest.fixture(autouse=True)
def _forget_servers():
    server_window.clear()
    yield
    server_window.clear()


@pytest.fixture
def serve():
    servers = []

    def start(stub: _Stub) -> str:
        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, payload: dict) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                self._send(*stub.models())

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(*stub.chat(body))

            def log_message(self, format, *args):  # noqa: A002, ARG002
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}/v1"

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _client(base_url: str, **llm) -> LLMClient:
    settings = GlobalSettings()
    settings.llm.provider = "openai"
    settings.llm.base_url = base_url
    settings.llm.model = "stub"
    settings.llm.api_key = "stub-key"
    settings.llm.context_tokens = None
    settings.llm.pointer_outputs = False
    for key, value in llm.items():
        setattr(settings.llm, key, value)
    return LLMClient(settings=settings)


def _room_300_text(serve) -> str:
    """Front matter whose prompt fits the window but leaves 300 tokens, fewer
    than the answer needs. The prompt around the text is measured first."""
    probe = _Stub("0.32")
    url = serve(probe)
    asyncio.run(_client(url).extract_title_keywords(WORD * 200))
    around = probe.log[0]["prompt_tokens"] - _tokens(WORD * 200)
    return WORD * ((WINDOW - 300 - around) // 2)


def test_vllm_031_names_the_window_in_a_400_and_the_call_is_asked_again(serve):
    text = _room_300_text(serve)
    stub = _Stub("0.31")
    client = _client(serve(stub))

    result = asyncio.run(client.extract_title_keywords(text))

    assert result.title == "A study of reading rooms"
    assert client._settings.llm.context_tokens == WINDOW
    # Learned from the rejection: the model list is not read.
    assert stub.routes() == ["chat", "chat"]
    assert [r["status"] for r in stub.log] == [400, 200]


def test_vllm_032_stops_at_the_window_and_the_window_is_read_from_the_model_list(serve):
    text = _room_300_text(serve)
    stub = _Stub("0.32")
    client = _client(serve(stub))

    result = asyncio.run(client.extract_title_keywords(text))

    assert result.title == "A study of reading rooms"
    assert client._settings.llm.context_tokens == WINDOW
    assert stub.routes() == ["chat", "models", "chat"]
    first, retry = stub.log[0], stub.log[2]
    assert first["finish_reason"] == "length"
    assert first["prompt_tokens"] + first["completion_tokens"] == WINDOW
    # Asked again with the input cut so the answer's cap fits beside it.
    assert retry["prompt_tokens"] + retry["max_tokens"] <= WINDOW
    assert retry["finish_reason"] == "stop"


def test_a_server_that_lists_no_window_fails_the_call_as_before_and_is_not_asked_again(serve):
    text = _room_300_text(serve)
    stub = _Stub("0.32", lists_window=False)
    client = _client(serve(stub))

    for _ in range(2):
        with pytest.raises(LlmTruncatedError):
            asyncio.run(client.extract_title_keywords(text))

    assert client._settings.llm.context_tokens is None
    assert stub.routes() == ["chat", "models", "chat"]


def test_an_answer_stopped_short_of_the_window_teaches_no_window(serve):
    """A limit of the server's own also stops an answer short of the cap. The
    window it lists is not reached, so it is not the cause and is not taken."""
    text = _room_300_text(serve)
    stub = _Stub("0.32", server_cap=100)
    client = _client(serve(stub))

    with pytest.raises(LlmTruncatedError):
        asyncio.run(client.extract_title_keywords(text))

    assert client._settings.llm.context_tokens is None
    assert stub.routes() == ["chat", "models"]


def test_an_answer_without_usage_cannot_be_measured_against_the_window(serve):
    text = _room_300_text(serve)
    stub = _Stub("0.32", usage=False)
    client = _client(serve(stub))

    with pytest.raises(LlmTruncatedError):
        asyncio.run(client.extract_title_keywords(text))

    assert client._settings.llm.context_tokens is None
    assert stub.routes() == ["chat"]


def test_an_answer_that_does_not_parse_is_not_put_down_to_the_window(serve):
    stub = _Stub("0.32", answer="not json at all")
    client = _client(serve(stub))

    with pytest.raises(LlmInvalidOutputError):
        asyncio.run(client.extract_title_keywords(WORD * 1_000))

    assert stub.routes() == ["chat"]


def test_a_call_cut_to_a_known_window_does_not_read_the_model_list(serve):
    stub = _Stub("0.32", server_cap=100)
    client = _client(serve(stub), context_tokens=WINDOW)

    with pytest.raises(LlmTruncatedError):
        asyncio.run(client.extract_title_keywords(WORD * 1_000))

    assert client._settings.llm.context_tokens == WINDOW
    assert stub.routes() == ["chat"]


def test_the_window_is_read_once_per_server(serve):
    text = _room_300_text(serve)
    stub = _Stub("0.32")
    url = serve(stub)

    for _ in range(2):
        # A new client with its own settings, as each pipeline has.
        assert asyncio.run(_client(url).extract_title_keywords(text)).title

    assert stub.routes() == ["chat", "models", "chat", "chat", "chat"]


def test_each_concurrent_call_that_stopped_at_the_window_is_asked_again(serve):
    """Also the one whose answer comes after the other call learned the
    window: it was sent without one, so its input was not cut."""
    text = _room_300_text(serve)
    stub = _Stub("0.32", second_answer_delay=0.5)
    client = _client(serve(stub))

    async def both():
        return await asyncio.gather(
            client.extract_title_keywords(text), client.extract_title_keywords(text)
        )

    results = asyncio.run(both())

    assert [r.title for r in results] == ["A study of reading rooms"] * 2
    assert client._settings.llm.context_tokens == WINDOW
    assert stub.routes().count("chat") == 4


def _listing(*entries: dict) -> dict:
    return {"object": "list", "data": list(entries)}


def _settings(base_url: str = "http://10.0.0.5:8000/v1", **llm) -> GlobalSettings:
    settings = GlobalSettings()
    settings.llm.provider = "openai"
    settings.llm.base_url = base_url
    settings.llm.model = "org/model"
    settings.llm.api_key = None
    for key, value in llm.items():
        setattr(settings.llm, key, value)
    return settings


def _transport(*answers: tuple[int, object]):
    requests: list[httpx.Request] = []
    pending = list(answers)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        status, body = pending.pop(0) if len(pending) > 1 else pending[0]
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler), requests


def _listed(settings: GlobalSettings, transport) -> int | None:
    return asyncio.run(server_window.listed_window(settings, transport=transport))


@pytest.mark.parametrize(
    ("listing", "window"),
    [
        (_listing({"id": "org/model", "max_model_len": 32_768}), 32_768),
        # One listed model is the one served, whatever it is called.
        (_listing({"id": "/models/local-copy", "max_model_len": 16_384}), 16_384),
        # A router listing several lends none of them its window.
        (
            _listing(
                {"id": "org/other", "max_model_len": 32_768},
                {"id": "org/model", "max_model_len": 8_192},
            ),
            8_192,
        ),
        (
            _listing(
                {"id": "org/other", "max_model_len": 32_768},
                {"id": "org/third", "max_model_len": 8_192},
            ),
            None,
        ),
        (_listing({"id": "org/model"}), None),
        (_listing({"id": "org/model", "max_model_len": 0}), None),
        (_listing({"id": "org/model", "max_model_len": True}), None),
        (_listing({"id": "org/model", "max_model_len": "32768"}), None),
        ({"data": "none"}, None),
        ([1, 2], None),
    ],
)
def test_the_window_listed_for_the_model(listing, window):
    transport, requests = _transport((200, listing))

    assert _listed(_settings(), transport) == window
    assert [str(r.url) for r in requests] == ["http://10.0.0.5:8000/v1/models"]


def test_a_server_that_does_not_answer_is_asked_again():
    transport, requests = _transport(
        (503, {"error": "loading"}), (200, _listing({"id": "org/model", "max_model_len": 4_096}))
    )
    settings = _settings()

    assert _listed(settings, transport) is None
    assert _listed(settings, transport) == 4_096
    assert _listed(settings, transport) == 4_096
    assert len(requests) == 2


@pytest.mark.parametrize("answer", [(404, {"detail": "Not Found"}), (200, "not json")])
def test_a_server_that_answers_without_a_window_is_not_asked_again(answer):
    transport, requests = _transport(answer)
    settings = _settings()

    assert _listed(settings, transport) is None
    assert _listed(settings, transport) is None
    assert len(requests) == 1


def test_each_model_on_a_server_has_its_own_window():
    transport, requests = _transport(
        (
            200,
            _listing(
                {"id": "org/model", "max_model_len": 8_192},
                {"id": "org/other", "max_model_len": 32_768},
            ),
        )
    )

    assert _listed(_settings(), transport) == 8_192
    assert _listed(_settings(model="org/other"), transport) == 32_768
    assert len(requests) == 2


def test_a_cloud_provider_is_not_asked():
    transport, requests = _transport((200, _listing({"id": "org/model", "max_model_len": 8_192})))

    assert _listed(_settings(provider="anthropic"), transport) is None
    assert _listed(_settings(base_url=None), transport) is None
    assert requests == []


def test_the_key_goes_only_where_the_chat_call_may_send_it():
    transport, requests = _transport((200, _listing({"id": "org/model", "max_model_len": 8_192})))

    public = _settings(base_url="http://example.org/v1", api_key="secret-key")
    assert _listed(public, transport) is None
    assert requests == []

    private = _settings(api_key="secret-key")
    assert _listed(private, transport) == 8_192
    assert requests[0].headers["Authorization"] == "Bearer secret-key"
