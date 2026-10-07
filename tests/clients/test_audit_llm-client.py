"""LLM client audit: cache key, SDK retries and timeouts, usage, JSON recovery.

Each test fails on the code before the fix. The loopback API below stands in
for a provider; it asks any SDK that retries to come back at once, so a
hidden SDK retry shows up as an extra request instead of a slow test.
"""

import asyncio
import json
import logging
import os
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest
from pydantic import BaseModel

from bibr.clients.llm import LLMClient, _extract_retry_after_seconds
from bibr.clients.llm_cache import LlmResponseCache, endpoint_identity
from bibr.clients.prompts import PROMPTS
from bibr.clients.structured_json import recover_structured_object
from bibr.config import GlobalSettings
from bibr.schemas import TitleKeywordsLLM


class _Reply(BaseModel):
    reply: str


@pytest.fixture
def fake_api():
    """A loopback HTTP API answering each POST with the next queued reply."""
    state = SimpleNamespace(replies=[], requests=[], url="")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append((self.path, body))
            reply = state.replies.pop(0) if state.replies else _status(500)
            status, payload = reply(body)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("retry-after-ms", "1")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):  # noqa: A002, ARG002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _status(code):
    return lambda _body: (code, {"type": "error", "error": {"type": "overloaded_error"}})


def _anthropic_tool_reply(body):
    """An Anthropic Messages reply calling the requested tool; usage as Anthropic sends it."""
    return 200, {
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
        "usage": {
            "input_tokens": 50,
            "output_tokens": 100,
            "cache_read_input_tokens": 9000,
            "cache_creation_input_tokens": 200,
        },
    }


def _mock_limiter():
    limiter = mock.MagicMock()
    limiter.acquire = mock.AsyncMock(return_value=None)
    limiter.close = mock.AsyncMock(return_value=None)
    return limiter


# --- 1. the response cache key names the provider configuration -------------


def _cache_key(tmp_path, **llm):
    """Key of one title call with no cap or effort of its own; *llm* are settings."""
    settings = GlobalSettings(
        cache={"llm": True, "llm_dir": str(tmp_path)}, llm={"model": "default", **llm}
    )
    spec = PROMPTS["title_keywords"]
    messages = [{"role": "user", "content": spec.build_user(boundary="a" * 32, text="front")}]
    _, key = LLMClient(settings=settings)._cache_lookup_key(
        spec.response_model,
        messages,
        spec.system,
        protocol="instructor",
        reasoning_effort=None,
        client_override=None,
        max_tokens=None,
    )
    assert key is not None
    return key


_GPU_A = {"provider": "openai", "base_url": "http://gpu-a:8000/v1"}


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (_GPU_A, {"provider": "anthropic"}),
        (_GPU_A, {**_GPU_A, "base_url": "http://gpu-b:8000/v1"}),
        (_GPU_A, {**_GPU_A, "temperature": 0.7}),
        (_GPU_A, {**_GPU_A, "instructor_mode": "json"}),
        # A call without its own cap or effort inherits the global one.
        ({"provider": "openai"}, {"provider": "openai", "max_tokens": 8192}),
        ({"provider": "openai"}, {"provider": "openai", "reasoning_effort": "high"}),
        ({"provider": "anthropic"}, {"provider": "anthropic", "thinking_budget": 8000}),
        ({"provider": "google"}, {"provider": "google", "thinking_budget": 2048}),
        (
            {"provider": "ollama", "ollama_base_url": "http://gpu-a:11434"},
            {"provider": "ollama", "ollama_base_url": "http://gpu-b:11434"},
        ),
        # Managed llmster serves any LLM_LLMSTER_MODEL as "bibr-local".
        (
            {**_GPU_A, "model": "bibr-local", "llmster_model": "qwen-a"},
            {**_GPU_A, "model": "bibr-local", "llmster_model": "qwen-b"},
        ),
    ],
)
def test_a_different_provider_configuration_gets_a_different_key(tmp_path, before, after):
    assert _cache_key(tmp_path, **before) != _cache_key(tmp_path, **after)


def test_the_same_configuration_still_hits(tmp_path):
    assert _cache_key(tmp_path, **_GPU_A) == _cache_key(tmp_path, **_GPU_A)


def test_endpoint_credentials_stay_out_of_the_key(tmp_path):
    secret = {**_GPU_A, "base_url": "http://user:hunter2@GPU-A:8000/v1/?key=sk-123"}
    assert _cache_key(tmp_path, **secret) == _cache_key(tmp_path, **_GPU_A)
    assert endpoint_identity("https://u:p@Host:8443/v1/?api_key=x#f") == "https://host:8443/v1"
    assert endpoint_identity(None) == endpoint_identity("  ") == ""


class _CountingBackend:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):  # noqa: ARG002
        self.calls += 1
        return TitleKeywordsLLM(title=f"answer {self.calls}"), None


async def test_another_server_behind_the_same_model_name_is_not_served_from_cache(tmp_path):
    """vLLM's ``--served-model-name default`` on two hosts: one model name."""
    backend = _CountingBackend()

    def client(base_url):
        settings = GlobalSettings(
            cache={"llm": True, "llm_dir": str(tmp_path)},
            llm={"provider": "openai", "model": "default", "base_url": base_url},
        )
        llm = LLMClient(settings=settings, backend=backend)
        llm._limiter = _mock_limiter()
        return llm

    first = await client("http://gpu-a:8000/v1").extract_title_keywords("body")
    second = await client("http://gpu-b:8000/v1").extract_title_keywords("body")
    again = await client("http://gpu-b:8000/v1").extract_title_keywords("body")

    assert backend.calls == 2
    assert (first.title, second.title, again.title) == ("answer 1", "answer 2", "answer 2")


# --- 2. SDK clients send each request once, bounded by bibr's timeout -------


def _settings(provider, **llm):
    return GlobalSettings(
        llm={"provider": provider, "model": "m", "api_key": "sk-test-key-0000", **llm}
    )


@pytest.mark.parametrize("provider", ["openai", "anthropic", "groq", "ollama"])
def test_provider_clients_disable_sdk_retries(monkeypatch, provider):
    from bibr.clients import providers

    raw = SimpleNamespace(max_retries=2, timeout=600.0)
    monkeypatch.setattr("instructor.from_provider", lambda *a, **k: SimpleNamespace(client=raw))

    providers.get(provider, settings=_settings(provider, timeout_seconds=45)).build_client()

    assert raw.max_retries == 0
    assert raw.timeout.read == 90.0  # the 2 x LLM_TIMEOUT_SECONDS limit of one call
    assert raw.timeout.connect == 5.0  # the SDKs' own: a host that is down fails fast


@pytest.mark.parametrize("provider", ["openai", "anthropic", "ollama"])
def test_real_sdk_clients_are_bounded(provider):
    from bibr.clients import providers

    client = providers.get(provider, settings=_settings(provider)).build_client()

    assert client.client.max_retries == 0
    assert client.client.timeout == httpx.Timeout(60.0, connect=5.0)


def test_google_client_gets_a_request_timeout(monkeypatch):
    from bibr.clients import providers

    built = []
    monkeypatch.setattr("instructor.from_provider", lambda *a, **k: built.append(k))

    providers.get("google", settings=_settings("google", timeout_seconds=45)).build_client()

    assert built[0]["http_options"] == {"timeout": 90_000}  # milliseconds


def test_nuextract_native_client_disables_sdk_retries():
    from bibr.clients.nuextract import NuExtractNativeBackend

    settings = _settings("openai", base_url="http://127.0.0.1:8000/v1")
    client = NuExtractNativeBackend(settings=settings)._get_client()

    assert client.max_retries == 0


def test_openai_server_error_is_one_request(fake_api):
    from bibr.clients.providers.openai import OpenAIProvider

    fake_api.replies = [_status(503)] * 3
    provider = OpenAIProvider(settings=_settings("openai", base_url=fake_api.url + "/v1"))
    client = provider.build_client()

    async def call():
        return await client.create(
            response_model=_Reply,
            messages=[{"role": "user", "content": "Reply OK"}],
            max_retries=1,
            **provider.call_kwargs(None, max_tokens=64),
        )

    with pytest.raises(Exception):  # noqa: B017, PT011 — any provider error
        asyncio.run(call())
    # bibr's own retry loop, which the rate limiter and breaker see, retries.
    assert len(fake_api.requests) == 1


def test_anthropic_overload_is_one_request(fake_api, monkeypatch):
    from bibr.clients.providers.anthropic import AnthropicProvider

    monkeypatch.setenv("ANTHROPIC_BASE_URL", fake_api.url)
    fake_api.replies = [_status(529)] * 3
    provider = AnthropicProvider(settings=_settings("anthropic"))
    client = provider.build_client()

    async def call():
        return await client.create(
            response_model=_Reply,
            messages=[{"role": "user", "content": "Reply OK"}],
            max_retries=1,
            **provider.call_kwargs(None, max_tokens=64),
        )

    with pytest.raises(Exception):  # noqa: B017, PT011 — any provider error
        asyncio.run(call())
    assert len(fake_api.requests) == 1


def _chat_completion(body):
    return 200, {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": body["model"],
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps({"reply": "OK"})},
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


@pytest.fixture
def backoffs(monkeypatch):
    """Record the retry loop's backoff waits instead of sleeping them."""
    real_sleep = asyncio.sleep
    waits: list[float] = []

    async def fake_sleep(delay, *args, **kwargs):
        if delay:
            waits.append(delay)
        await real_sleep(0)

    monkeypatch.setattr("bibr.clients.llm.asyncio.sleep", fake_sleep)
    return waits


async def _ask(client):
    client._limiter = _mock_limiter()
    try:
        return await client._invoke_structured(_Reply, [{"role": "user", "content": "hi"}], "s")
    finally:
        await client.close()


# Instructor wraps the SDK error and the status is on its cause. The retry
# loop read only the wrapper, so with SDK retries off a 5xx failed the call.
@pytest.mark.parametrize("code", [500, 503, 408, 409])
async def test_openai_server_error_is_retried_by_bibr(fake_api, backoffs, code):
    fake_api.replies = [_status(code), _chat_completion]
    settings = _settings("openai", base_url=fake_api.url + "/v1", instructor_mode="json")

    result = await _ask(LLMClient(settings=settings))

    assert result.reply == "OK"
    assert len(fake_api.requests) == 2
    assert len(backoffs) == 1


@pytest.mark.parametrize("code", [500, 529])
async def test_anthropic_overload_is_retried_by_bibr(fake_api, backoffs, monkeypatch, code):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", fake_api.url)
    fake_api.replies = [_status(code), _anthropic_tool_reply]

    result = await _ask(LLMClient(settings=_settings("anthropic", model="claude-haiku-4-5")))

    assert result.reply == "OK"
    assert len(fake_api.requests) == 2
    assert len(backoffs) == 1


async def test_a_rejected_request_is_not_retried(fake_api, backoffs):
    fake_api.replies = [_status(400), _chat_completion]
    settings = _settings("openai", base_url=fake_api.url + "/v1", instructor_mode="json")

    with pytest.raises(Exception):  # noqa: B017, PT011 — any provider error
        await _ask(LLMClient(settings=settings))
    assert len(fake_api.requests) == 1
    assert backoffs == []


# --- 3. Retry-After is honoured within a ceiling ----------------------------


def _rate_limited(retry_after: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example/v1")
    response = httpx.Response(429, headers={"retry-after": retry_after}, request=request)
    return httpx.HTTPStatusError("rate limited", request=request, response=response)


@pytest.mark.parametrize("value", ["inf", "Infinity", "nan", "-5"])
def test_unusable_retry_after_is_ignored(value):
    assert _extract_retry_after_seconds(_rate_limited(value)) is None


@pytest.mark.parametrize("value", ["3600", "1e300", "inf"])
async def test_retry_after_wait_is_capped(monkeypatch, value):
    client = LLMClient(settings=GlobalSettings(llm={"track_usage": False}))
    client._limiter = _mock_limiter()
    fake = mock.MagicMock()
    fake.create = mock.AsyncMock(side_effect=[_rate_limited(value), _Reply(reply="OK")])
    client._client = fake
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("bibr.clients.llm.asyncio.sleep", fake_sleep)
    result = await client._invoke_structured(_Reply, [{"role": "user", "content": "hi"}], "sys")

    assert result.reply == "OK"
    assert len(sleeps) == 1
    assert 0 < sleeps[0] <= LLMClient._RETRY_AFTER_MAX + 0.5


# --- 4 and 5. Anthropic: usage includes cached input; uncapped calls fit ----


def test_anthropic_usage_counts_cache_reads_and_writes_as_input():
    from anthropic.types import Usage

    client = LLMClient(settings=GlobalSettings(llm={"model": "claude"}))
    usage = Usage(
        input_tokens=50,
        output_tokens=100,
        cache_read_input_tokens=9000,
        cache_creation_input_tokens=200,
    )
    client._record_usage(SimpleNamespace(usage=usage))

    assert client.usage["claude"] == {
        "input_tokens": 9250,
        "output_tokens": 100,
        "total_tokens": 9350,
        "cached_input_tokens": 9000,
    }


def test_openai_shaped_usage_with_cache_fields_is_not_counted_twice():
    """A proxy translating Anthropic usage keeps its cache fields, but its
    ``prompt_tokens`` already includes them."""
    client = LLMClient(settings=GlobalSettings(llm={"model": "m"}))
    usage = SimpleNamespace(
        prompt_tokens=9050,
        completion_tokens=100,
        total_tokens=9150,
        cache_read_input_tokens=9000,
    )
    client._record_usage(SimpleNamespace(usage=usage))

    assert client.usage["m"]["input_tokens"] == 9050
    assert client.usage["m"]["total_tokens"] == 9150


@pytest.mark.parametrize(
    ("model", "requested", "sent"),
    [
        ("claude-haiku-4-5", None, 16384),
        ("claude-haiku-4-5", 65536, 16384),
        ("claude-haiku-4-5", 4096, 4096),
        # The SDK's own non-streaming limit for this model is lower.
        ("claude-opus-4-1-20250805", None, 8192),
    ],
)
def test_anthropic_caps_non_streaming_output(model, requested, sent):
    from bibr.clients.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(settings=_settings("anthropic", model=model, max_tokens=65536))
    assert provider.call_kwargs(None, max_tokens=requested)["max_tokens"] == sent


def test_a_thinking_budget_that_never_fits_warns_once(caplog):
    from bibr.clients.providers import anthropic

    anthropic._warn_budget_never_fits.cache_clear()
    provider = anthropic.AnthropicProvider(settings=_settings("anthropic", thinking_budget=16000))
    with caplog.at_level(logging.WARNING, logger="bibr.clients.providers.anthropic"):
        bodies = [provider.call_kwargs(None) for _ in range(3)]

    assert not any("thinking" in body for body in bodies)
    [record] = caplog.records
    assert "LLM_THINKING_BUDGET 16000" in record.getMessage()


async def test_uncapped_anthropic_call_goes_through_and_records_cached_input(fake_api, monkeypatch):
    """Reference segmentation and merged core metadata pass no task cap; the
    Anthropic SDK refused their 65536-token non-streaming request."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", fake_api.url)
    fake_api.replies = [_anthropic_tool_reply]
    settings = _settings("anthropic", model="claude-haiku-4-5", track_usage=True)
    client = LLMClient(settings=settings)
    client._limiter = _mock_limiter()

    result = await client._invoke_structured(_Reply, [{"role": "user", "content": "hi"}], "sys")

    assert result.reply == "OK"
    [(path, body)] = fake_api.requests
    assert path == "/v1/messages"
    assert body["max_tokens"] == 16384
    assert client.usage["claude-haiku-4-5"]["input_tokens"] == 50 + 9000 + 200
    assert client.usage["claude-haiku-4-5"]["cached_input_tokens"] == 9000
    await client.close()


# --- 6. JSON recovery keeps LaTeX commands --------------------------------


def test_recovery_keeps_latex_commands_that_look_like_json_escapes():
    raw = (
        r'{"title": "Effect of \alpha and \theta on \beta decay with \frac{1}{2},'
        r' \nu and \rho", "keywords": []}'
    )
    recovered = recover_structured_object(raw, TitleKeywordsLLM)

    assert recovered.value.title == (
        "Effect of \\alpha and \\theta on \\beta decay with \\frac{1}{2}, \\nu and \\rho"
    )
    assert recovered.repaired_backslashes == 6


def test_recovery_keeps_a_newline_that_starts_a_sentence():
    raw = r'{"title": "Spin \(1/2\) chains\nThe second line", "keywords": []}'
    assert recover_structured_object(raw, TitleKeywordsLLM).value.title == (
        "Spin \\(1/2\\) chains\nThe second line"
    )


def test_valid_json_escapes_are_untouched_without_raw_latex():
    raw = '{"title": "line one\\nline two\\tand\\bthree", "keywords": []}'
    recovered = recover_structured_object(raw, TitleKeywordsLLM)

    assert recovered.value.title == "line one\nline two\tand\bthree"
    assert recovered.repaired_backslashes == 0


# --- 7. cache files are private -----------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_cache_directories_and_entries_are_owner_only(tmp_path):
    root = tmp_path / "bibr" / "llm"
    cache = LlmResponseCache(root)
    key = "ab" + "0" * 30
    previous = os.umask(0o022)
    try:
        assert cache.put(key, {"title": "unpublished"}, model="m", schema_name="S")
    finally:
        os.umask(previous)

    path = cache.path_for(key)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert cache.get(key) == {"title": "unpublished"}
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no tmp left


# --- 8. close() and per-call usage logs ----------------------------------------


async def test_close_closes_the_sdk_clients_and_rebuilds_on_next_use():
    client = LLMClient(settings=_settings("openai"))
    raw = client._get_client().client
    json_raw = client._get_json_mode_client().client

    await client.close()

    assert raw.is_closed() and json_raw.is_closed()
    assert client._get_client().client is not raw


async def test_close_closes_the_native_backend_client():
    settings = _settings(
        "openai", base_url="http://127.0.0.1:8000/v1", structured_backend="nuextract-native"
    )
    client = LLMClient(settings=settings)
    raw = client._backend._get_client()

    await client.close()

    assert raw.is_closed()


async def test_close_leaves_an_injected_backend_alone():
    backend = SimpleNamespace(_client=object())
    client = LLMClient(settings=_settings("openai"), backend=backend)

    await client.close()

    assert backend._client is not None


async def test_concurrent_calls_each_log_their_own_usage(caplog):
    client = LLMClient(settings=GlobalSettings(llm={"track_usage": True, "model": "m"}))
    other_finished = asyncio.Event()

    def completion(prompt, output):
        usage = SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=output, total_tokens=prompt + output
        )
        return SimpleNamespace(usage=usage)

    async def slow():
        client._record_usage(completion(100, 10))
        await other_finished.wait()

    async def fast():
        client._record_usage(completion(7, 3))
        other_finished.set()

    with caplog.at_level(logging.INFO, logger="bibr.clients.llm"):
        await asyncio.gather(
            client._run_labeled_call("slow", slow), client._run_labeled_call("fast", fast)
        )

    logged = sorted(r.getMessage() for r in caplog.records if "Token usage" in r.getMessage())
    assert logged == [
        "Token usage [fast] model=m: input=7, output=3, total=10",
        "Token usage [slow] model=m: input=100, output=10, total=110",
    ]
    assert client.usage["m"]["total_tokens"] == 120


async def test_rate_limiter_is_keyed_by_provider_and_model():
    settings = _settings("openai", model="gpt-x")
    settings.redis.url = None
    client = LLMClient(settings=settings)

    await client._ensure_limiter()

    assert client.limiter.resource_id == "llm:openai:gpt-x"
