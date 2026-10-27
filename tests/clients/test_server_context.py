"""Inputs sized to a self-hosted server's context window, read from the server.

An earlier run replayed a hosted deployment's Ollama setup (16,384-token
context, bibr's 300,000-character input cap). The Ollama library model cut
over-long prompts from the front, the GGUF-template model answered HTTP 400
"request (N tokens) exceeds the available context size (16384 tokens)", and
three papers lost their export because the title/keywords call failed on it.
bibr now asks Ollama and llama.cpp for their window before the first prompt and
caps inputs the way the managed llama.cpp server does. External vLLM is left as
it was.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from unittest.mock import AsyncMock

import httpx
import instructor
import pytest
from openai import AsyncOpenAI

from bibr.clients import llm, server_context
from bibr.clients.llm import (
    LLMClient,
    _context_budget,
    _NoRoomForInputError,
    _share_by_need,
    _task_max_tokens,
)
from bibr.clients.server_context import (
    ServerContext,
    apply_learned_window,
    apply_server_context,
    fixed_context_caps,
)
from bibr.clients.server_context import probe_server_context as real_probe
from bibr.config import GlobalSettings
from bibr.exceptions import LlmRejectedError
from bibr.paper_contents import CanonicalSection
from bibr.processing_warnings import WarningCode
from bibr.schemas import (
    AuthorLLM,
    AuthorsLLM,
    PaperClassificationLLM,
    PaperTypeLabel,
    RefAnchors,
    TitleKeywordsLLM,
)
from bibr.structure.section_classifier import _classify_llm_batch, _fitting_spans

HOSTED_URL = "http://172.17.0.1:11434/v1"
FT_MODEL = "hf.co/scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0"
PAPER_TEXT = "A printed study of word growth."
FRONT_MATTER = f"{PAPER_TEXT} " * 9_000


def _hosted_settings(**llm_fields) -> GlobalSettings:
    """A hosted deployment's bibr.env LLM block: the openai provider pointed at Ollama's /v1."""
    return GlobalSettings(
        llm={"provider": "openai", "base_url": HOSTED_URL, "model": FT_MODEL, **llm_fields}
    )


class FakeOllama:
    """Ollama 0.40 as bibr sees it: its API routes and a fixed context.

    ``loaded`` puts the model in ``/api/ps`` with ``num_ctx`` as its
    ``context_length``; ``down`` refuses every connection. Chat prompts are
    counted at ``chars_per_token`` characters a token and rejected past
    ``num_ctx`` the way Ollama rejects them for a model that uses its GGUF
    chat template.
    """

    def __init__(self, *, num_ctx=16_384, loaded=True, parameters="", chars_per_token=3):
        self.num_ctx = num_ctx
        self.loaded = loaded
        self.down = False
        self.parameters = parameters
        self.chars_per_token = chars_per_token
        self.requests: list[tuple[str, str]] = []
        self.prompts: list[str] = []
        self.prompt_tokens: list[int] = []
        self.rejected: list[int] = []

    def respond(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused")
        path = request.url.path
        self.requests.append((request.method, path))
        if path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": FT_MODEL}]})
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.11"})
        if path == "/api/ps":
            models = [{"name": FT_MODEL, "model": FT_MODEL, "context_length": self.num_ctx}]
            return httpx.Response(200, json={"models": models if self.loaded else []})
        if path == "/api/show":
            # The trained maximum, which is not the context Ollama runs.
            return httpx.Response(
                200,
                json={
                    "model_info": {
                        "general.architecture": "qwen35",
                        "qwen35.context_length": 262144,
                    },
                    **({"parameters": self.parameters} if self.parameters else {}),
                },
            )
        if path == "/v1/chat/completions":
            return self._chat(json.loads(request.content))
        return httpx.Response(404, text="404 page not found")

    def probes(self) -> int:
        return self.requests.count(("GET", "/api/version"))

    def _chat(self, body: dict) -> httpx.Response:
        prompt = "".join(_text(m.get("content")) for m in body["messages"])
        tokens = len(prompt) // self.chars_per_token
        self.prompts.append(prompt)
        self.prompt_tokens.append(tokens)
        if tokens > self.num_ctx:
            self.rejected.append(tokens)
            message = (
                f"request ({tokens} tokens) exceeds the available context size "
                f"({self.num_ctx} tokens), try increasing it"
            )
            return httpx.Response(
                400, json={"error": {"message": message, "type": "invalid_request_error"}}
            )
        name = ((body.get("response_format") or {}).get("json_schema") or {}).get("name", "")
        content = {
            "TitleKeywordsLLM": {"title": "A printed title"},
            "AuthorsLLM": {"authors": [{"given": "Mara", "family": "Quill"}]},
            "PaperClassificationLLM": {"paper_type": "empirical"},
            "CitationResolutionResult": {"matches": []},
            "ResearchIntegrityLLM": {"funding": [], "contributions": [], "affiliations": []},
        }.get(name, {})
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 1,
                "model": FT_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json.dumps(content)},
                        "finish_reason": "stop",
                    }
                ],
            },
        )


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


@pytest.fixture
def clock(monkeypatch):
    """The clock ``ensure_server_context`` rate-limits asking again with."""
    now = [1_000.0]
    monkeypatch.setattr(llm, "_server_context_clock", lambda: now[0])
    return now


@pytest.fixture
def ollama_client(monkeypatch):
    """An ``LLMClient`` whose probe and chat calls both reach a ``FakeOllama``."""

    def make(server: FakeOllama, settings: GlobalSettings) -> LLMClient:
        transport = httpx.MockTransport(server.respond)
        http = httpx.AsyncClient(transport=transport)

        def create_client(mode_override=None, **_kwargs):
            sdk = AsyncOpenAI(api_key="x", base_url=HOSTED_URL, http_client=http, max_retries=0)
            return instructor.from_openai(
                sdk, mode=mode_override or instructor.Mode.JSON_SCHEMA, model=FT_MODEL
            )

        async def probe(settings, *, transport=None):
            return await real_probe(settings, transport=httpx.MockTransport(server.respond))

        monkeypatch.setattr(llm, "_create_client", create_client)
        monkeypatch.setattr(server_context, "probe_server_context", probe)
        client = LLMClient(settings=settings)
        monkeypatch.setattr(client, "_acquire_rate_limit", AsyncMock())
        return client

    return make


def _probe(settings: GlobalSettings, respond):
    return real_probe(settings, transport=httpx.MockTransport(respond))


# --- The probe -------------------------------------------------------------


async def test_ollama_behind_the_openai_provider_reports_its_running_context():
    server = FakeOllama(num_ctx=16_384)
    found = await _probe(_hosted_settings(), server.respond)

    assert found == ServerContext("ollama", 16_384)
    assert server.requests[:3] == [
        ("GET", "/v1/models"),
        ("GET", "/props"),
        ("GET", "/api/version"),
    ]


async def test_an_ollama_model_that_is_not_loaded_falls_back_to_its_num_ctx():
    server = FakeOllama(loaded=False, parameters="temperature 1\nnum_ctx 8192\ntop_k 20")
    assert await _probe(_hosted_settings(), server.respond) == ServerContext("ollama", 8_192)


async def test_the_trained_context_length_is_not_taken_for_the_running_one():
    server = FakeOllama(loaded=False)
    assert await _probe(_hosted_settings(), server.respond) == ServerContext("ollama", None)


async def test_the_ollama_provider_asks_ollama_routes_only():
    server = FakeOllama()
    # Ollama matches model names without regard to case.
    settings = GlobalSettings(
        llm={"provider": "ollama", "ollama_base_url": "http://localhost:11434", "model": "Llama3"}
    )

    def respond(request):
        if request.url.path == "/api/ps":
            server.requests.append((request.method, request.url.path))
            return httpx.Response(
                200, json={"models": [{"name": "llama3:latest", "context_length": 4096}]}
            )
        return server.respond(request)

    assert await _probe(settings, respond) == ServerContext("ollama", 4_096)
    assert ("GET", "/v1/models") not in server.requests
    assert ("GET", "/props") not in server.requests


async def test_vllm_is_recognised_by_its_max_model_len():
    def respond(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "other", "max_model_len": 131_072},
                    {"id": "served", "max_model_len": 32_768},
                ]
            },
        )

    settings = GlobalSettings(
        llm={"provider": "openai", "base_url": "http://gpu:8000/v1", "model": "served"}
    )
    assert await _probe(settings, respond) == ServerContext("vllm", 32_768)


async def test_a_server_listing_one_model_is_read_whatever_its_id():
    def respond(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "/models/q35", "max_model_len": 8_192}]})

    settings = GlobalSettings(
        llm={"provider": "openai", "base_url": "http://gpu:8000/v1", "model": "served"}
    )
    assert await _probe(settings, respond) == ServerContext("vllm", 8_192)


async def test_a_router_listing_several_models_lends_none_its_window():
    # A proxy in front of several servers: the first listed model's window
    # says nothing about the one bibr asks for.
    seen = []

    def respond(request):
        seen.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "big", "max_model_len": 131_072},
                        {"id": "small", "max_model_len": 8_192},
                    ]
                },
            )
        return httpx.Response(404, text="not found")

    settings = GlobalSettings(
        llm={"provider": "openai", "base_url": "http://router:4000/v1", "model": "served"}
    )
    assert await _probe(settings, respond) == ServerContext("other", None)
    assert seen == ["/v1/models", "/props", "/api/version"]


async def test_llama_cpp_reports_its_slot_context():
    def respond(request):
        if request.url.path == "/props":
            return httpx.Response(
                200, json={"default_generation_settings": {"n_ctx": 16_384}, "total_slots": 1}
            )
        return httpx.Response(200, json={"data": [{"id": "model"}]})

    settings = GlobalSettings(llm={"provider": "openai", "base_url": "http://box:8080/v1"})
    assert await _probe(settings, respond) == ServerContext("llama.cpp", 16_384)


async def test_the_api_key_goes_to_a_private_server():
    seen = []

    def respond(request):
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"data": [{"id": "m", "max_model_len": 8_192}]})

    settings = GlobalSettings(
        llm={"provider": "openai", "base_url": "http://10.0.0.5:8000/v1", "api_key": "k"}
    )
    assert await _probe(settings, respond) == ServerContext("vllm", 8_192)
    assert seen == ["Bearer k"]


async def test_no_key_goes_over_plain_http_to_a_public_host():
    def respond(request):
        raise AssertionError("no request expected")

    settings = GlobalSettings(
        llm={"provider": "openai", "base_url": "http://llm.example.com/v1", "api_key": "k"}
    )
    assert await _probe(settings, respond) is None


@pytest.mark.parametrize(
    "llm_fields", [{"provider": "google"}, {"provider": "anthropic"}, {"provider": "openai"}]
)
async def test_cloud_providers_are_not_asked(llm_fields):
    def respond(request):
        raise AssertionError("no request expected")

    assert await _probe(GlobalSettings(llm=llm_fields), respond) is None


async def test_a_server_that_is_down_or_starting_gives_none():
    def down(request):
        raise httpx.ConnectError("refused")

    def starting(request):
        return httpx.Response(503, json={"error": {"message": "Loading model"}})

    assert await _probe(_hosted_settings(), down) is None
    assert await _probe(_hosted_settings(), starting) is None


async def test_a_server_that_answers_on_no_known_route_is_other():
    def silent(request):
        return httpx.Response(404, text="not found")

    assert await _probe(_hosted_settings(), silent) == ServerContext("other", None)


@pytest.mark.parametrize(
    "llm_fields",
    [
        {"provider": "openai", "base_url": "http://host:port/v1"},
        {"provider": "openai", "base_url": "http://[::1/v1"},
        {"provider": "openai", "base_url": "http://ho\x00st:8080/v1"},
        {"provider": "openai", "base_url": "http://host:port/v1", "api_key": "k"},
        {"provider": "ollama", "ollama_base_url": "http://[::1"},
    ],
)
async def test_a_malformed_url_gives_none_and_never_raises(llm_fields):
    sent = []

    def respond(request):
        sent.append(request.url)
        return httpx.Response(404)

    assert await _probe(GlobalSettings(llm=llm_fields), respond) is None
    assert sent == []


async def test_an_unexpected_error_in_the_probe_is_not_raised():
    def broken(request):
        raise RuntimeError("a transport bug")

    assert await _probe(_hosted_settings(), broken) is None


# --- Applying what the server said ------------------------------------------


def test_a_known_fixed_context_sets_the_window_and_the_llama_cpp_caps():
    llm_options = GlobalSettings().llm
    apply_server_context(llm_options, ServerContext("ollama", 16_384))

    assert llm_options.context_tokens == 16_384
    assert llm_options.max_tokens == 4_096
    assert llm_options.max_input_chars == 24_000
    assert llm_options.ref_seg_window_chars == 12_000


def test_caps_scale_with_a_larger_context():
    llm_options = GlobalSettings().llm
    apply_server_context(llm_options, ServerContext("llama.cpp", 32_768))

    assert dict(fixed_context_caps(32_768)) == {
        "max_tokens": 8_192,
        "max_input_chars": 48_000,
        "ref_seg_window_chars": 24_000,
    }
    assert llm_options.max_input_chars == 48_000
    assert llm_options.context_tokens == 32_768


def test_an_unknown_ollama_context_gets_the_16k_caps_and_no_window():
    llm_options = GlobalSettings().llm
    apply_server_context(llm_options, ServerContext("ollama", None))

    assert llm_options.context_tokens is None
    assert (llm_options.max_tokens, llm_options.max_input_chars) == (4_096, 24_000)


@pytest.mark.parametrize("found", [ServerContext("vllm", 32_768), ServerContext("other", None)])
def test_vllm_and_other_servers_are_left_as_they_were(found):
    llm_options = GlobalSettings().llm
    apply_server_context(llm_options, found)

    assert llm_options.context_tokens is None
    assert llm_options.max_input_chars == 300_000
    assert llm_options.max_tokens == 65_536
    assert llm_options.ref_seg_window_chars == 16_000


def test_explicit_settings_always_win(caplog):
    llm_options = GlobalSettings(llm={"max_input_chars": 100_000, "max_tokens": 2_048}).llm
    with caplog.at_level(logging.WARNING, logger="bibr.clients.server_context"):
        apply_server_context(llm_options, ServerContext("ollama", 16_384))

    assert llm_options.max_input_chars == 100_000
    assert llm_options.max_tokens == 2_048
    assert llm_options.ref_seg_window_chars == 12_000
    assert "explicit llm.max_input_chars=100000 exceeds" in caplog.text


def test_caps_bibr_set_are_not_taken_for_the_users(caplog):
    # Assigned fields land in pydantic's model_fields_set. Read from there,
    # the 16k caps looked user-set when a 4096-token window came in later:
    # max_tokens stayed 4096 and the input budget of a title call was 0.
    settings = _hosted_settings()
    apply_server_context(settings.llm, ServerContext("ollama", 16_384))
    with caplog.at_level(logging.WARNING, logger="bibr.clients.server_context"):
        apply_server_context(settings.llm, ServerContext("ollama", 4_096))

    assert settings.llm.context_tokens == 4_096
    assert settings.llm.max_tokens == 1_024
    assert settings.llm.max_input_chars == 6_000
    assert settings.llm.ref_seg_window_chars == 3_000
    title_cap = _task_max_tokens(settings, settings.llm.title_max_tokens)
    assert _context_budget(4_096, title_cap) == 1_024
    assert "explicit" not in caplog.text


def test_a_larger_window_found_later_raises_the_caps_bibr_set_up_to_the_defaults():
    llm_options = GlobalSettings().llm
    apply_server_context(llm_options, ServerContext("ollama", None))
    apply_server_context(llm_options, ServerContext("ollama", 32_768))
    assert (llm_options.max_tokens, llm_options.max_input_chars) == (8_192, 48_000)

    apply_server_context(llm_options, ServerContext("ollama", 1_048_576))
    assert (llm_options.max_tokens, llm_options.max_input_chars) == (65_536, 300_000)


def test_a_learned_window_moves_the_caps_bibr_owns_on_fixed_context_servers(caplog):
    settings = _hosted_settings(max_input_chars=50_000)
    apply_server_context(settings.llm, ServerContext("ollama", None))
    settings.llm.context_tokens = 4_096
    with caplog.at_level(logging.WARNING, logger="bibr.clients.server_context"):
        apply_learned_window(settings.llm, "ollama", 4_096)

    assert settings.llm.max_tokens == 1_024
    assert settings.llm.ref_seg_window_chars == 3_000
    assert settings.llm.max_input_chars == 50_000, "the user's cap is kept"
    assert "explicit llm.max_input_chars=50000 exceeds the 4096-token" in caplog.text

    # vLLM, other servers and a server that never answered keep their caps.
    for server in ("vllm", "other", None):
        llm_options = GlobalSettings().llm
        apply_learned_window(llm_options, server, 4_096)
        assert (llm_options.max_tokens, llm_options.max_input_chars) == (65_536, 300_000)


async def test_a_users_window_sizes_the_caps_and_wins_over_the_servers(ollama_client):
    # LLM_CONTEXT_TOKENS used to skip the probe, and with it every cap:
    # max_tokens went back to 65536 and the segmentation window to 16000.
    server = FakeOllama(num_ctx=16_384)
    client = ollama_client(server, _hosted_settings(context_tokens=8_192))

    await client.ensure_server_context()

    assert server.probes() == 1
    llm_options = client._settings.llm
    assert llm_options.context_tokens == 8_192
    assert (llm_options.max_tokens, llm_options.max_input_chars) == (2_048, 12_000)
    assert llm_options.ref_seg_window_chars == 6_000


async def test_explicit_caps_still_win_over_a_users_window(ollama_client):
    server = FakeOllama(num_ctx=16_384)
    client = ollama_client(server, _hosted_settings(context_tokens=8_192, max_input_chars=50_000))

    await client.ensure_server_context()

    assert client._settings.llm.max_input_chars == 50_000
    assert client._settings.llm.max_tokens == 2_048


async def test_the_ollama_provider_with_a_users_window_is_not_asked(monkeypatch):
    probe = AsyncMock()
    monkeypatch.setattr(server_context, "probe_server_context", probe)
    settings = GlobalSettings(
        llm={"provider": "ollama", "model": "llama3", "context_tokens": 8_192}
    )
    client = LLMClient(settings=settings)

    await client.ensure_server_context()

    probe.assert_not_awaited()
    assert client._settings.llm.context_tokens == 8_192
    assert client._settings.llm.max_input_chars == 12_000


async def test_a_server_bibr_started_is_not_asked(monkeypatch):
    probe = AsyncMock()
    monkeypatch.setattr(server_context, "probe_server_context", probe)
    settings = _hosted_settings()
    # What the managed llama.cpp and vLLM servers do when they start.
    settings.llm.context_tokens = 16_384
    client = LLMClient(settings=settings)

    await client.ensure_server_context()

    probe.assert_not_awaited()


# --- Asking again ------------------------------------------------------------


async def test_a_server_down_at_the_first_call_is_asked_again_later(ollama_client, clock):
    # One client serves a run or a serve process for its whole life: an
    # answer missed at the first call must not stick until a restart.
    server = FakeOllama(num_ctx=16_384)
    server.down = True
    client = ollama_client(server, _hosted_settings())

    await client.ensure_server_context()
    assert client._settings.llm.context_tokens is None
    assert client._settings.llm.max_input_chars == 300_000

    server.down = False
    clock[0] += 5
    await client.ensure_server_context()
    assert server.probes() == 0, "asked again too soon"

    clock[0] += 30
    await client.ensure_server_context()
    assert server.probes() == 1
    assert client._settings.llm.context_tokens == 16_384
    assert client._settings.llm.max_input_chars == 24_000

    clock[0] += 300
    await client.ensure_server_context()
    assert server.probes() == 1, "a complete answer is not asked again"


async def test_an_unknown_ollama_window_is_asked_again_and_fits_4096(ollama_client, clock):
    # The model was not loaded at the first call, so the 16k caps stood in.
    # It then loads with a 4,096-token context.
    server = FakeOllama(num_ctx=4_096, loaded=False)
    client = ollama_client(server, _hosted_settings())

    await client.ensure_server_context()
    llm_options = client._settings.llm
    assert llm_options.context_tokens is None
    assert (llm_options.max_tokens, llm_options.max_input_chars) == (4_096, 24_000)

    server.loaded = True
    clock[0] += 31
    combined = await client.extract_core_metadata(FRONT_MATTER)

    assert server.probes() == 2
    assert llm_options.context_tokens == 4_096
    assert (llm_options.max_tokens, llm_options.max_input_chars) == (1_024, 6_000)
    assert server.rejected == []
    assert combined.title == "A printed title"
    # Each prompt still carries the paper's text, not only the instructions.
    assert all("A printed study of word growth." in prompt for prompt in server.prompts)


async def test_concurrent_calls_ask_the_server_once(monkeypatch, clock):
    calls = 0
    release = asyncio.Event()

    async def slow_probe(settings, *, transport=None):
        nonlocal calls
        calls += 1
        await release.wait()
        return ServerContext("ollama", None)

    monkeypatch.setattr(server_context, "probe_server_context", slow_probe)
    client = LLMClient(settings=_hosted_settings())

    # First calls wait for the one answer, so their prompts are sized.
    first = [asyncio.create_task(client.ensure_server_context()) for _ in range(8)]
    await asyncio.sleep(0)
    assert calls == 1 and not any(task.done() for task in first)
    release.set()
    await asyncio.gather(*first)
    assert calls == 1
    assert client._settings.llm.max_input_chars == 24_000

    # Asking again: one call asks, the others keep the caps and go on.
    release.clear()
    clock[0] += 31
    asking = asyncio.create_task(client.ensure_server_context())
    await asyncio.sleep(0)
    assert calls == 2
    await asyncio.wait_for(asyncio.gather(*(client.ensure_server_context() for _ in range(5))), 1)
    assert calls == 2 and not asking.done()
    release.set()
    await asking


async def test_a_probe_that_raises_does_not_fail_the_call(monkeypatch, clock):
    probe = AsyncMock(side_effect=RuntimeError("a probe bug"))
    monkeypatch.setattr(server_context, "probe_server_context", probe)
    client = LLMClient(settings=_hosted_settings())

    await client.ensure_server_context()
    clock[0] += 31
    await client.ensure_server_context()

    assert probe.await_count == 2


async def test_an_endpoint_that_never_says_is_asked_less_and_less_often(monkeypatch, clock):
    # A proxy that answers 5xx on the routes it does not serve gave no answer
    # and was asked again every 30 s for as long as the process ran.
    asked: list[float] = []
    up = False

    def proxy(request):
        path = request.url.path
        if path == "/v1/models":
            asked.append(clock[0])
        if up:
            if path == "/props":
                return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 8_192}})
            return httpx.Response(404, text="not found")
        return httpx.Response(502, text="Bad Gateway")

    async def probe(settings, *, transport=None):
        return await real_probe(settings, transport=httpx.MockTransport(proxy))

    monkeypatch.setattr(server_context, "probe_server_context", probe)
    client = LLMClient(
        settings=GlobalSettings(
            llm={"provider": "openai", "base_url": "http://10.0.0.7:8080/v1", "model": "m"}
        )
    )
    start = clock[0]
    for _ in range(360):  # a call every 10 s for an hour
        await client.ensure_server_context()
        clock[0] += 10

    gaps = [later - earlier for earlier, later in zip(asked, asked[1:], strict=False)]
    assert asked[0] == start
    assert gaps == [30, 60, 120, 240, 480, 600, 600, 600, 600]
    assert client._settings.llm.context_tokens is None

    # It still gets sized once it answers, within the longest wait.
    up = True
    clock[0] += 600
    await client.ensure_server_context()
    assert len(asked) == 11
    llm_options = client._settings.llm
    assert llm_options.context_tokens == 8_192
    assert (llm_options.max_tokens, llm_options.max_input_chars) == (2_048, 12_000)
    clock[0] += 3_600
    await client.ensure_server_context()
    assert len(asked) == 11, "a complete answer is not asked again"


async def test_a_long_outage_keeps_the_longest_wait_without_overflowing(monkeypatch, clock):
    # 1,025 misses in a row (about a week of calls) overflowed 2 ** misses as a
    # float, and from then on every LLM call raised before it was sent.
    probe = AsyncMock(return_value=None)
    monkeypatch.setattr(server_context, "probe_server_context", probe)
    client = LLMClient(
        settings=GlobalSettings(
            llm={"provider": "openai", "base_url": "http://10.0.0.7:8080/v1", "model": "m"}
        )
    )
    await client.ensure_server_context()
    client._server_context_misses = 5_000

    clock[0] += 600
    await client.ensure_server_context()

    assert probe.await_count == 2
    assert client._server_context_retry_at == clock[0] + 600


async def test_a_new_endpoint_starts_with_the_shortest_wait_and_no_server_kind(monkeypatch, clock):
    answers = [ServerContext("ollama", None)] * 3 + [None] * 3
    probe = AsyncMock(side_effect=answers)
    monkeypatch.setattr(server_context, "probe_server_context", probe)
    client = LLMClient(settings=_hosted_settings())
    for _ in range(3):  # an Ollama model that is not loaded: asked again
        await client.ensure_server_context()
        clock[0] += 600
    assert client._server_kind == "ollama"
    assert client._server_context_misses == 3

    client._settings.llm.base_url = "http://10.0.0.8:8080/v1"
    await client.ensure_server_context()

    assert probe.await_count == 4
    assert client._server_kind is None
    assert client._server_context_misses == 1
    assert client._server_context_retry_at == clock[0] + 30


async def test_the_ollama_provider_down_at_the_first_calls_is_sized_once_it_is_up(
    ollama_client, clock
):
    server = FakeOllama(num_ctx=8_192)
    server.down = True
    settings = GlobalSettings(
        llm={
            "provider": "ollama",
            "ollama_base_url": "http://172.17.0.1:11434",
            "model": FT_MODEL,
        }
    )
    client = ollama_client(server, settings)
    llm_options = client._settings.llm

    await client.ensure_server_context()  # down: asked again after 30 s
    clock[0] += 30
    await client.ensure_server_context()  # still down: after 60 s
    assert llm_options.context_tokens is None
    assert (llm_options.max_tokens, llm_options.max_input_chars) == (4_096, 24_000)

    server.down = False
    clock[0] += 30
    await client.ensure_server_context()
    assert server.probes() == 0, "asked again before the wait was over"
    clock[0] += 30
    await client.ensure_server_context()

    assert server.probes() == 1
    assert llm_options.context_tokens == 8_192
    assert (llm_options.max_tokens, llm_options.max_input_chars) == (2_048, 12_000)


async def test_an_external_vllm_gets_no_caps_before_the_first_call(monkeypatch, clock):
    # No vLLM arm was measured, and bibr's eval harness points LLM_BASE_URL
    # at vLLM: it keeps learning its window from a context-length 400.
    seen = []

    def vllm(request):
        seen.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "served", "max_model_len": 32_768}]})
        return httpx.Response(404, json={"detail": "Not Found"})

    async def probe(settings, *, transport=None):
        return await real_probe(settings, transport=httpx.MockTransport(vllm))

    monkeypatch.setattr(server_context, "probe_server_context", probe)
    settings = GlobalSettings(
        llm={"provider": "openai", "base_url": "http://10.0.0.5:8000/v1", "model": "served"}
    )
    client = LLMClient(settings=settings)

    await client.ensure_server_context()
    clock[0] += 300
    await client.ensure_server_context()

    assert seen == ["/v1/models"]
    llm_options = client._settings.llm
    assert llm_options.context_tokens is None
    assert (llm_options.max_tokens, llm_options.max_input_chars) == (65_536, 300_000)
    assert llm_options.ref_seg_window_chars == 16_000


# --- End to end against a fake Ollama ----------------------------------------


async def test_hosted_defaults_keep_every_front_matter_call_inside_16k(ollama_client):
    server = FakeOllama(num_ctx=16_384)
    client = ollama_client(server, _hosted_settings())

    combined = await client.extract_core_metadata(FRONT_MATTER)

    assert combined.title == "A printed title"
    assert [(a.given, a.family) for a in combined.authors] == [("Mara", "Quill")]
    assert server.rejected == []
    assert len(server.prompt_tokens) == 3 and max(server.prompt_tokens) < 16_384
    # One probe for the three concurrent calls.
    assert server.probes() == 1
    assert client._settings.llm.context_tokens == 16_384


async def test_an_overflow_at_an_unknown_window_is_asked_again_cut_to_it(ollama_client):
    # Ollama's model is not loaded and sets no num_ctx, so bibr assumes 16k
    # caps, but the server runs a smaller context: every call overflows
    # once, learns the window from the 400 and asks again with its input cut
    # to fit. Before, only the first of the concurrent calls asked again.
    server = FakeOllama(num_ctx=6_000, loaded=False)
    client = ollama_client(server, _hosted_settings())

    combined = await client.extract_core_metadata(FRONT_MATTER)

    assert combined.title == "A printed title"
    assert [(a.given, a.family) for a in combined.authors] == [("Mara", "Quill")]
    assert combined.paper_type == "empirical"
    assert len(server.rejected) == 3
    assert client._settings.llm.context_tokens == 6_000
    accepted = [t for t in server.prompt_tokens if t not in server.rejected]
    assert len(accepted) == 3 and max(accepted) <= 6_000


def _accepted(server: FakeOllama) -> list[str]:
    return [
        p for p, t in zip(server.prompts, server.prompt_tokens, strict=True) if t <= server.num_ctx
    ]


async def test_a_window_learned_from_a_rejection_leaves_room_for_the_paper(ollama_client, clock):
    # Ollama's model is not loaded at the probe, so the 16k caps stand in
    # (max_tokens 4,096), and the server runs a 4,096-token context. The
    # window came from the 400 alone: the caps stayed at 4,096 and each
    # retry, and every call until the next probe, had an input budget of
    # _context_budget(4096, 4096) == 0. The title and author prompts went
    # out with no paper text, were accepted, and no field was marked failed.
    server = FakeOllama(num_ctx=4_096, loaded=False)
    client = ollama_client(server, _hosted_settings())

    combined = await client.extract_core_metadata(FRONT_MATTER)

    llm_options = client._settings.llm
    assert server.probes() == 1, "the clock did not move: no second probe"
    assert llm_options.context_tokens == 4_096
    assert (llm_options.max_tokens, llm_options.max_input_chars) == (1_024, 6_000)
    for task_cap in (llm_options.title_max_tokens, llm_options.authors_max_tokens):
        assert _context_budget(4_096, _task_max_tokens(client._settings, task_cap)) == 1_024
    assert len(server.rejected) == 3
    assert len(_accepted(server)) == 3
    assert all(PAPER_TEXT in prompt for prompt in _accepted(server))
    assert combined.title == "A printed title"
    assert [(a.given, a.family) for a in combined.authors] == [("Mara", "Quill")]
    assert combined._field_failures == {}

    # The next paper, still before the server is asked again.
    sent = len(server.prompts)
    again = await client.extract_core_metadata(FRONT_MATTER)

    assert server.probes() == 1
    assert len(server.rejected) == 3
    assert len(server.prompts) - sent == 3
    assert all(PAPER_TEXT in prompt for prompt in server.prompts[sent:])
    assert again.title == "A printed title" and again._field_failures == {}


async def test_no_prompt_goes_out_without_paper_text(ollama_client):
    # A server bibr cannot place (it answers none of the probe routes) gets
    # no caps. It names a 4,096-token window in a 400, and the default title
    # and author completion caps leave no room beside it: those prompts are
    # not sent, and their fields are marked failed as an overflow is.

    class UnplacedServer(FakeOllama):
        def respond(self, request: httpx.Request) -> httpx.Response:
            if request.url.path in {"/v1/models", "/props", "/api/version"}:
                return httpx.Response(404, text="not found")
            return super().respond(request)

    server = UnplacedServer(num_ctx=4_096)
    client = ollama_client(server, _hosted_settings())

    combined = await client.extract_core_metadata(FRONT_MATTER)

    assert client._settings.llm.context_tokens == 4_096
    assert client._settings.llm.max_tokens == 65_536
    assert not combined.title and combined.authors == []
    assert combined._field_failures["title"] == "llm_failed"
    assert combined._field_failures["authors"] == "llm_failed"
    # The classification cap (1,024) leaves room: it is sent and kept.
    assert combined.paper_type == "empirical"
    assert len(server.rejected) == 3
    assert len(_accepted(server)) == 1
    assert all(PAPER_TEXT in prompt for prompt in _accepted(server))


async def test_an_overflow_past_the_known_window_keeps_the_record(ollama_client):
    # The server named the window bibr already cut to (the estimate was too
    # low): asking again cannot help, and the paper is kept without the
    # fields of the calls that overflowed.
    server = FakeOllama(num_ctx=16_384, chars_per_token=1)
    client = ollama_client(server, _hosted_settings())

    combined = await client.extract_core_metadata(FRONT_MATTER)

    assert not combined.title and combined.authors == []
    assert combined._field_failures["title"] == "llm_failed"
    assert combined._field_failures["authors"] == "llm_failed"
    # Title, authors, classification and the empty-author classification
    # re-roll, each sent once.
    assert server.rejected == server.prompt_tokens and len(server.rejected) == 4


async def test_a_context_overflow_on_the_title_call_does_not_fail_the_paper(monkeypatch):
    overflow = RuntimeError(
        "Error code: 400 - {'error': {'message': 'request (20666 tokens) exceeds the "
        "available context size (16384 tokens), try increasing it'}}"
    )
    client = LLMClient(settings=GlobalSettings())
    monkeypatch.setattr(
        client,
        "extract_title_keywords",
        AsyncMock(side_effect=LlmRejectedError("Failed to extract title/keywords", overflow)),
    )
    monkeypatch.setattr(
        client,
        "extract_authors",
        AsyncMock(return_value=AuthorsLLM(authors=[AuthorLLM(given="Mara", family="Quill")])),
    )
    monkeypatch.setattr(
        client,
        "extract_paper_classification",
        AsyncMock(return_value=PaperClassificationLLM(paper_type="empirical")),
    )

    combined = await client.extract_core_metadata("printed content")

    assert not combined.title
    assert [(a.given, a.family) for a in combined.authors] == [("Mara", "Quill")]
    assert combined._field_failures == dict.fromkeys(TitleKeywordsLLM.model_fields, "llm_failed")


# --- Several texts in one prompt ---------------------------------------------


def test_share_by_need_keeps_small_inputs_whole_and_gives_the_rest_to_large_ones():
    assert _share_by_need([10, 100, 1_000], 300) == [10, 100, 190]
    assert _share_by_need([1_000, 10], 300) == [290, 10]
    assert _share_by_need([500, 500], 300) == [150, 150]
    assert _share_by_need([5, 7], 300) == [5, 7]


def _fenced(prompt: str, name: str) -> str:
    match = re.search(rf"--- (\w+) BEGIN {name} ---\n(.*?)\n--- \1 END {name} ---", prompt, re.S)
    assert match is not None
    return match.group(2)


def _refs(count: int) -> list[dict]:
    return [
        {"bib_id": i, "author": "Quill, Mara", "year": "2019", "title": "Word growth " * 4}
        for i in range(count)
    ]


async def test_two_long_citation_lists_together_stay_inside_the_window(ollama_client):
    # A long reference list and a long citation list were each cut to the
    # whole window, so together they could still run past it.
    server = FakeOllama(num_ctx=16_384)
    client = ollama_client(server, _hosted_settings())
    cites = [(i, "(Quill and Vexler, in press; Quill, in press)") for i in range(800)]

    await client.resolve_citations(cites, _refs(500))

    assert server.rejected == [] and len(server.prompt_tokens) == 1
    assert server.prompt_tokens[0] < 16_384 - 4_096


async def test_a_short_citation_list_leaves_its_share_to_the_references(ollama_client):
    # osf_bpkce: a 54,940-character reference list next to a short citation
    # list. An even split cut the references to about 28%.
    server = FakeOllama(num_ctx=16_384)
    client = ollama_client(server, _hosted_settings())
    cites = [(i, "(Quill, 2019)") for i in range(20)]

    await client.resolve_citations(cites, _refs(500))

    prompt = server.prompts[0]
    sent_refs, sent_cites = _fenced(prompt, "REFERENCES"), _fenced(prompt, "CITATIONS")
    assert sent_cites.count("text_id=") == 20
    # The two lists share LLM_MAX_INPUT_CHARS (24,000 here): the references
    # get all the citations leave, not half.
    assert len(sent_refs) + len(sent_cites) > 0.95 * 24_000
    assert server.rejected == []


async def test_the_research_integrity_texts_share_the_window(ollama_client):
    server = FakeOllama(num_ctx=16_384)
    client = ollama_client(server, _hosted_settings())
    authors = [("Mara", f"Quill{i}") for i in range(3_000)]

    await client.extract_research_integrity(
        "Funded by the Word Growth Council. " * 900,
        "M.Q. designed the study and wrote the paper. " * 700,
        authors,
        affiliation_list=[f"Department of Words {i}, University of Growth" for i in range(700)],
    )

    assert server.rejected == [] and len(server.prompt_tokens) == 1
    assert server.prompt_tokens[0] < 16_384 - 4_096


async def test_research_integrity_shares_the_char_cap_when_the_window_is_unknown(ollama_client):
    # The model is not loaded: the 16k caps apply but the window is unknown,
    # so only LLM_MAX_INPUT_CHARS bounds the prompt. Each text was cut to
    # all of it, up to 3 x 24,000 characters plus an uncapped author list.
    server = FakeOllama(num_ctx=16_384, loaded=False)
    client = ollama_client(server, _hosted_settings())

    await client.extract_research_integrity(
        "Funded by the Word Growth Council. " * 900,
        "M.Q. designed the study and wrote the paper. " * 700,
        [("Mara", f"Quill{i}") for i in range(3_000)],
        affiliation_list=[f"Department of Words {i}, University of Growth" for i in range(700)],
    )

    assert server.rejected == [] and len(server.prompt_tokens) == 1
    assert client._settings.llm.context_tokens is None


async def test_the_paper_type_label_input_is_cut_to_the_window():
    settings = GlobalSettings(llm={"context_tokens": 4_096})
    client = LLMClient(settings=settings)
    sent = []

    async def invoke(model, messages, system, **kwargs):
        sent.append(_text(messages[0]["content"]))
        return PaperTypeLabel(paper_type="empirical")

    client._invoke_structured = invoke
    await client.label_paper_type("A study of word growth", "Words grow. " * 30_000)

    assert "A study of word growth" in sent[0]
    # 4,096 - 512 completion - 2,048 prompt = 1,536 tokens of title and
    # abstract, well under 8,000 characters.
    assert len(sent[0]) < 8_000


# --- Learning a window from a rejection --------------------------------------


async def test_the_no_room_guard_holds_for_every_provider():
    # A cloud provider with a user-set window: the default 4,096-token title
    # cap leaves 0 tokens for input in a 4,096-token window.
    settings = GlobalSettings(llm={"context_tokens": 4_096})
    client = LLMClient(settings=settings)
    client._invoke_structured = AsyncMock(return_value=TitleKeywordsLLM(title="x"))

    with pytest.raises(LlmRejectedError) as raised:
        await client.extract_title_keywords(FRONT_MATTER)

    assert isinstance(raised.value, _NoRoomForInputError)
    assert raised.value.error_code == "llm_failed"
    assert "leaves 0 tokens" in str(raised.value)
    client._invoke_structured.assert_not_awaited()


def test_the_no_room_guard_fires_only_below_the_floor_and_only_when_text_is_cut():
    settings = GlobalSettings(llm={"context_tokens": 4_096})
    # Budget 0, nothing to cut.
    assert LLMClient._cap_input("", settings, max_tokens=4_096) == ""
    # Budget 148 tokens: a short text fits whole and is sent.
    assert _context_budget(4_096, 1_900) == 148
    assert LLMClient._cap_input(PAPER_TEXT, settings, max_tokens=1_900) == PAPER_TEXT
    with pytest.raises(_NoRoomForInputError):
        LLMClient._cap_input(FRONT_MATTER, settings, max_tokens=1_900)
    # Several texts share the budget, and the guard.
    with pytest.raises(_NoRoomForInputError):
        LLMClient._cap_inputs([FRONT_MATTER, PAPER_TEXT], settings, max_tokens=1_900)
    # At the floor (256 tokens) the text is cut, not refused.
    assert _context_budget(4_096, 1_792) == 256
    capped = LLMClient._cap_input(FRONT_MATTER, settings, max_tokens=1_792)
    assert PAPER_TEXT in capped and len(capped) < 1_000


async def test_invoke_structured_learns_the_window_a_server_names():
    client = LLMClient(settings=GlobalSettings())
    client._invoke_structured = AsyncMock(
        side_effect=RuntimeError(
            "Error code: 400 - This model's maximum context length is 8192 tokens. "
            "However, you requested 9000 tokens."
        )
    )

    with pytest.raises(RuntimeError):
        await client.invoke_structured(
            TitleKeywordsLLM, [{"role": "user", "content": "x"}], "s", label="section_classifier"
        )

    assert client._settings.llm.context_tokens == 8_192


async def test_a_segmentation_window_rejected_as_too_long_is_split_when_asked_again():
    # A context 400 re-ran every segmentation window unchanged, so the
    # second attempt failed the same way.
    client = LLMClient(
        settings=GlobalSettings(
            llm={"provider": "openai", "base_url": "http://10.0.0.5:8000/v1", "model": "m"}
        )
    )
    windows = []

    async def invoke(model, messages, system, **kwargs):
        window = _text(messages[0]["content"])
        windows.append(len(window))
        if len(window) > 6_000:
            raise RuntimeError(
                "Error code: 400 - This model's maximum context length is 4096 tokens. "
                "However, your request has 5200 input tokens."
            )
        return RefAnchors(anchors=["Quill, M."])

    client._invoke_structured = invoke
    text = "Quill, M. (2019). Word growth in printed studies. J. Words 12, 1-10.\n" * 400

    anchors = await client.segment_references(text)

    assert client._settings.llm.context_tokens == 4_096
    rejected = [n for n in windows if n > 6_000]
    assert rejected and len(windows) > len(rejected)
    assert anchors and all(a == "Quill, M." for a in anchors)


# --- Section classifier and implicit sections --------------------------------


def test_fitting_spans_split_where_the_next_line_would_not_fit():
    lines = ["a" * 40, "b" * 40, "c" * 40, "d" * 200, "e" * 10]

    def limit_of(text):
        return min(len(text), 100)

    assert _fitting_spans(lines, limit_of) == [(0, 2), (2, 3), (3, 4), (4, 5)]
    assert _fitting_spans(["x"], limit_of) == [(0, 1)]


def _classifier_reply(model, messages, system, *, label, max_tokens=None):
    document = messages[0]["content"][1]["text"]
    count = len(re.findall(r"^\d+\. ", document, re.MULTILINE))
    return model(classifications=[{"header": "", "section_type": "method"}] * count)


async def test_a_long_heading_list_is_classified_in_calls_that_fit():
    # One paper in the Ollama replay sent 80 headings with their opening text in
    # one 17,332-token prompt; at the 16k caps it now goes out in parts.
    settings = GlobalSettings(llm={"max_input_chars": 24_000, "context_tokens": 16_384})
    client = LLMClient(settings=settings)
    documents = []

    async def invoke(model, messages, system, **kwargs):
        documents.append(messages[0]["content"][1]["text"])
        return _classifier_reply(model, messages, system, **kwargs)

    client.invoke_structured = invoke
    headers = [f"Heading {i}" for i in range(80)]
    snippets = [
        f"Opening text of part {i}. " + "Lorem ipsum dolor sit amet. " * 30 for i in range(80)
    ]

    results = await _classify_llm_batch(headers, snippets, llm_client=client, settings=settings)

    assert len(documents) > 1
    assert all(len(doc) < 24_000 + 200 for doc in documents)
    assert sum(len(re.findall(r"^\d+\. ", d, re.MULTILINE)) for d in documents) == 80
    assert (
        results == [(CanonicalSection.METHODS, settings.layout.section_classification_score)] * 80
    )


async def test_a_short_heading_list_is_one_call_as_before():
    client = LLMClient(settings=GlobalSettings(llm={"max_input_chars": 24_000}))
    calls = []

    async def invoke(model, messages, system, **kwargs):
        calls.append(messages)
        return _classifier_reply(model, messages, system, **kwargs)

    client.invoke_structured = invoke
    results = await _classify_llm_batch(["Mystery one", "Mystery two"], llm_client=client)

    assert len(calls) == 1 and len(results) == 2


async def test_the_classifier_asks_the_server_before_splitting(ollama_client, monkeypatch):
    server = FakeOllama(num_ctx=16_384)
    client = ollama_client(server, _hosted_settings())
    documents = []

    async def invoke(model, messages, system, **kwargs):
        documents.append(messages[0]["content"][1]["text"])
        return _classifier_reply(model, messages, system, **kwargs)

    monkeypatch.setattr(client, "invoke_structured", invoke)
    headers = [f"Heading {i}" for i in range(80)]
    snippets = ["Lorem ipsum dolor sit amet. " * 20] * 80

    await _classify_llm_batch(headers, snippets, llm_client=client)

    assert client._settings.llm.max_input_chars == 24_000
    assert len(documents) > 1 and max(len(d) for d in documents) < 24_200


def _front_matter():
    from bibr.paper_contents import PaperSentence

    return [
        PaperSentence(
            text_id=i, text="A sentence of front matter. " * 5, section_id=1, paragraph_id=i
        )
        for i in range(200)
    ]


async def test_implicit_sections_cut_with_the_clients_settings():
    from bibr.structure.implicit_sections import _detect_via_llm

    settings = GlobalSettings(llm={"max_input_chars": 2_000})
    client = LLMClient(settings=settings)
    documents = []

    async def invoke(model, messages, system, **kwargs):
        documents.append(messages[0]["content"][1]["text"])
        return model(segments=[])

    client.invoke_structured = invoke

    await _detect_via_llm(client, _front_matter(), "h")

    assert len(documents) == 1 and len(documents[0]) < 2_000 + 200


async def test_a_failure_sizing_implicit_sections_takes_the_fallback(monkeypatch):
    from bibr.structure.implicit_sections import _detect_via_llm

    client = LLMClient(settings=GlobalSettings())
    monkeypatch.setattr(client, "ensure_server_context", AsyncMock(side_effect=RuntimeError("x")))
    warnings = []

    assert await _detect_via_llm(client, _front_matter(), "h", warnings=warnings) is None
    assert [w.code for w in warnings] == [WarningCode.IMPLICIT_SECTIONS_LLM_FAILED]
