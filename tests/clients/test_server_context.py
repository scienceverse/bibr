"""Inputs sized to a self-hosted server's context window, read from the server.

E10 replayed SURF's Ollama setup (16,384-token context, bibr's 300,000-character
input cap). The Ollama library model cut over-long prompts from the front, the
GGUF-template model answered HTTP 400 "request (N tokens) exceeds the available
context size (16384 tokens)", and three papers lost their export because the
title/keywords call failed on it. bibr now asks Ollama, llama.cpp and vLLM for
their window before the first prompt and caps inputs the way the managed
llama.cpp server does.
"""

from __future__ import annotations

import json
import logging
import re
from unittest.mock import AsyncMock

import httpx
import instructor
import pytest
from openai import AsyncOpenAI

from bibr.clients import llm, server_context
from bibr.clients.llm import LLMClient
from bibr.clients.server_context import (
    ServerContext,
    apply_server_context,
    fixed_context_caps,
)
from bibr.clients.server_context import probe_server_context as real_probe
from bibr.config import GlobalSettings
from bibr.exceptions import LlmRejectedError
from bibr.paper_contents import CanonicalSection
from bibr.schemas import AuthorLLM, AuthorsLLM, PaperClassificationLLM, TitleKeywordsLLM
from bibr.structure.section_classifier import _classify_llm_batch, _fitting_spans

SURF_URL = "http://172.17.0.1:11434/v1"
FT_MODEL = "hf.co/scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0"


def _surf_settings(**llm_fields) -> GlobalSettings:
    """SURF's bibr.env LLM block: the openai provider pointed at Ollama's /v1."""
    return GlobalSettings(
        llm={"provider": "openai", "base_url": SURF_URL, "model": FT_MODEL, **llm_fields}
    )


class FakeOllama:
    """Ollama 0.40 as bibr sees it: its API routes and a fixed context.

    ``loaded`` puts the model in ``/api/ps`` with ``num_ctx`` as its
    ``context_length``. Chat prompts are counted at ``chars_per_token``
    characters a token and rejected past ``num_ctx`` the way Ollama rejects
    them for a model that uses its GGUF chat template.
    """

    def __init__(self, *, num_ctx=16_384, loaded=True, parameters="", chars_per_token=3):
        self.num_ctx = num_ctx
        self.loaded = loaded
        self.parameters = parameters
        self.chars_per_token = chars_per_token
        self.requests: list[tuple[str, str]] = []
        self.prompt_tokens: list[int] = []
        self.rejected: list[int] = []

    def respond(self, request: httpx.Request) -> httpx.Response:
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

    def _chat(self, body: dict) -> httpx.Response:
        chars = sum(len(_text(m.get("content"))) for m in body["messages"])
        tokens = chars // self.chars_per_token
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
def ollama_client(monkeypatch):
    """An ``LLMClient`` whose probe and chat calls both reach a ``FakeOllama``."""

    def make(server: FakeOllama, settings: GlobalSettings) -> LLMClient:
        transport = httpx.MockTransport(server.respond)
        http = httpx.AsyncClient(transport=transport)

        def create_client(mode_override=None, **_kwargs):
            sdk = AsyncOpenAI(api_key="x", base_url=SURF_URL, http_client=http, max_retries=0)
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


def _probe(server: FakeOllama | None, settings: GlobalSettings, respond=None):
    handler = respond or server.respond
    return real_probe(settings, transport=httpx.MockTransport(handler))


# --- The probe -------------------------------------------------------------


async def test_ollama_behind_the_openai_provider_reports_its_running_context():
    server = FakeOllama(num_ctx=16_384)
    found = await _probe(server, _surf_settings())

    assert found == ServerContext("ollama", 16_384)
    assert server.requests[:3] == [
        ("GET", "/v1/models"),
        ("GET", "/props"),
        ("GET", "/api/version"),
    ]


async def test_an_ollama_model_that_is_not_loaded_falls_back_to_its_num_ctx():
    server = FakeOllama(loaded=False, parameters="temperature 1\nnum_ctx 8192\ntop_k 20")
    assert await _probe(server, _surf_settings()) == ServerContext("ollama", 8_192)


async def test_the_trained_context_length_is_not_taken_for_the_running_one():
    server = FakeOllama(loaded=False)
    assert await _probe(server, _surf_settings()) == ServerContext("ollama", None)


async def test_the_ollama_provider_asks_ollama_routes_only():
    server = FakeOllama()
    settings = GlobalSettings(
        llm={"provider": "ollama", "ollama_base_url": "http://localhost:11434", "model": "llama3"}
    )
    server.respond_model = "llama3:latest"

    def respond(request):
        if request.url.path == "/api/ps":
            server.requests.append((request.method, request.url.path))
            return httpx.Response(
                200, json={"models": [{"name": "llama3:latest", "context_length": 4096}]}
            )
        return server.respond(request)

    assert await _probe(None, settings, respond) == ServerContext("ollama", 4_096)
    assert ("GET", "/v1/models") not in server.requests
    assert ("GET", "/props") not in server.requests


async def test_vllm_reports_max_model_len():
    def respond(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200, json={"data": [{"id": "other"}, {"id": "served", "max_model_len": 32_768}]}
        )

    settings = GlobalSettings(
        llm={"provider": "openai", "base_url": "http://gpu:8000/v1", "model": "served"}
    )
    assert await _probe(None, settings, respond) == ServerContext("vllm", 32_768)


async def test_llama_cpp_reports_its_slot_context():
    def respond(request):
        if request.url.path == "/props":
            return httpx.Response(
                200, json={"default_generation_settings": {"n_ctx": 16_384}, "total_slots": 1}
            )
        return httpx.Response(200, json={"data": [{"id": "model"}]})

    settings = GlobalSettings(llm={"provider": "openai", "base_url": "http://box:8080/v1"})
    assert await _probe(None, settings, respond) == ServerContext("llama.cpp", 16_384)


async def test_the_api_key_goes_to_a_private_server():
    seen = []

    def respond(request):
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"data": [{"id": "m", "max_model_len": 8_192}]})

    settings = GlobalSettings(
        llm={"provider": "openai", "base_url": "http://10.0.0.5:8000/v1", "api_key": "k"}
    )
    assert await _probe(None, settings, respond) == ServerContext("vllm", 8_192)
    assert seen == ["Bearer k"]


async def test_no_key_goes_over_plain_http_to_a_public_host():
    def respond(request):
        raise AssertionError("no request expected")

    settings = GlobalSettings(
        llm={"provider": "openai", "base_url": "http://llm.example.com/v1", "api_key": "k"}
    )
    assert await _probe(None, settings, respond) is None


@pytest.mark.parametrize(
    "llm_fields", [{"provider": "google"}, {"provider": "anthropic"}, {"provider": "openai"}]
)
async def test_cloud_providers_are_not_asked(llm_fields):
    def respond(request):
        raise AssertionError("no request expected")

    assert await _probe(None, GlobalSettings(llm=llm_fields), respond) is None


async def test_a_server_that_is_down_or_says_nothing_gives_none():
    def down(request):
        raise httpx.ConnectError("refused")

    def silent(request):
        return httpx.Response(404, text="not found")

    assert await _probe(None, _surf_settings(), down) is None
    assert await _probe(None, _surf_settings(), silent) is None


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


def test_vllm_gets_the_window_only():
    llm_options = GlobalSettings().llm
    apply_server_context(llm_options, ServerContext("vllm", 32_768))

    assert llm_options.context_tokens == 32_768
    assert llm_options.max_input_chars == 300_000
    assert llm_options.max_tokens == 65_536


def test_explicit_settings_always_win(caplog):
    llm_options = GlobalSettings(llm={"max_input_chars": 100_000, "max_tokens": 2_048}).llm
    with caplog.at_level(logging.WARNING, logger="bibr.clients.server_context"):
        apply_server_context(llm_options, ServerContext("ollama", 16_384))

    assert llm_options.max_input_chars == 100_000
    assert llm_options.max_tokens == 2_048
    assert llm_options.ref_seg_window_chars == 12_000
    assert "explicit llm.max_input_chars=100000 exceeds" in caplog.text


async def test_a_set_context_window_is_not_probed(monkeypatch):
    probe = AsyncMock()
    monkeypatch.setattr(server_context, "probe_server_context", probe)
    client = LLMClient(settings=_surf_settings(context_tokens=8_192))

    await client.ensure_server_context()

    probe.assert_not_awaited()
    assert client._settings.llm.context_tokens == 8_192


# --- End to end against a fake Ollama ----------------------------------------


async def test_surf_defaults_keep_every_front_matter_call_inside_16k(ollama_client):
    server = FakeOllama(num_ctx=16_384)
    client = ollama_client(server, _surf_settings())

    combined = await client.extract_core_metadata("A printed study of word growth. " * 9_000)

    assert combined.title == "A printed title"
    assert [(a.given, a.family) for a in combined.authors] == [("Mara", "Quill")]
    assert server.rejected == []
    assert len(server.prompt_tokens) == 3 and max(server.prompt_tokens) < 16_384
    # One probe for the three concurrent calls.
    assert server.requests.count(("GET", "/api/version")) == 1
    assert client._settings.llm.context_tokens == 16_384


async def test_an_overflow_at_an_unknown_window_is_asked_again_cut_to_it(ollama_client):
    # Ollama's model is not loaded and sets no num_ctx, so bibr assumes 16k
    # caps, but the server runs a smaller context: every call overflows
    # once, learns the window from the 400 and asks again with its input cut
    # to fit. Before, only the first of the concurrent calls asked again.
    server = FakeOllama(num_ctx=6_000, loaded=False)
    client = ollama_client(server, _surf_settings())

    combined = await client.extract_core_metadata("A printed study of word growth. " * 9_000)

    assert combined.title == "A printed title"
    assert [(a.given, a.family) for a in combined.authors] == [("Mara", "Quill")]
    assert combined.paper_type == "empirical"
    assert len(server.rejected) == 3
    assert client._settings.llm.context_tokens == 6_000
    accepted = [t for t in server.prompt_tokens if t not in server.rejected]
    assert len(accepted) == 3 and max(accepted) <= 6_000


async def test_an_overflow_past_the_known_window_keeps_the_record(ollama_client):
    # The server named the window bibr already cut to (the estimate was too
    # low): asking again cannot help, and the paper is kept without the
    # fields of the calls that overflowed.
    server = FakeOllama(num_ctx=16_384, chars_per_token=1)
    client = ollama_client(server, _surf_settings())

    combined = await client.extract_core_metadata("A printed study of word growth. " * 9_000)

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


async def test_the_citation_blocks_share_the_window(ollama_client):
    # A long reference list and a long citation list were each cut to the
    # whole window, so together they could still run past it.
    server = FakeOllama(num_ctx=16_384)
    client = ollama_client(server, _surf_settings())
    refs = [
        {"bib_id": i, "author": "Quill, Mara", "year": "2019", "title": "Word growth " * 4}
        for i in range(500)
    ]
    cites = [(i, "(Quill and Vexler, in press; Quill, in press)") for i in range(800)]

    await client.resolve_citations(cites, refs)

    assert server.rejected == [] and len(server.prompt_tokens) == 1
    assert server.prompt_tokens[0] < 16_384 - 4_096


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
    # W4406122303 sent 80 headings with their opening text in one 17,332-token
    # prompt; at the 16k caps it now goes out in parts.
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
    client = ollama_client(server, _surf_settings())
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


async def test_implicit_sections_cut_with_the_clients_settings():
    from bibr.paper_contents import PaperSentence
    from bibr.structure.implicit_sections import _detect_via_llm

    settings = GlobalSettings(llm={"max_input_chars": 2_000})
    client = LLMClient(settings=settings)
    documents = []

    async def invoke(model, messages, system, **kwargs):
        documents.append(messages[0]["content"][1]["text"])
        return model(segments=[])

    client.invoke_structured = invoke
    front_matter = [
        PaperSentence(
            text_id=i, text="A sentence of front matter. " * 5, section_id=1, paragraph_id=i
        )
        for i in range(200)
    ]

    await _detect_via_llm(client, front_matter, "h")

    assert len(documents) == 1 and len(documents[0]) < 2_000 + 200
