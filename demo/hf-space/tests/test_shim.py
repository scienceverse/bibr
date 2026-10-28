"""The shim routes bibr's real HTTP clients to in-process fake models.

Runs without torch, model weights or network: the backends are fakes, and the
clients are bibr's own (paddle-http OCR client, ref-llm parser, OpenAI SDK).
The runner records a context variable, standing in for the Gradio request
that ``@spaces.GPU`` reads to bill the visitor.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zerogpu_models as zm  # noqa: E402

VISITOR = contextvars.ContextVar("visitor", default=None)
seen_visitors: list = []


class FakeBackend(zm.Backend):
    def __init__(self, alias, reply, max_batch=8):
        self.alias = alias
        self.reply = reply
        self.max_batch = max_batch
        self.batches: list[int] = []

    def generate(self, batch):
        self.batches.append(len(batch))
        return [zm.Completion(self.reply(r), "stop", 3, 5) for r in batch]


class SlowBackend(FakeBackend):
    def generate(self, batch):
        import time

        time.sleep(1.0)
        return super().generate(batch)


def runner(alias, batch):
    seen_visitors.append(VISITOR.get())
    return zm.BACKENDS[alias].generate(batch)


REF_RECORD = {"title": "A study of things", "year": "2020"}


@pytest.fixture(scope="module", autouse=True)
def shim():
    zm.BACKENDS.clear()
    zm.BACKENDS.update(
        {
            "paddle-ocr-vl-1.6": FakeBackend("paddle-ocr-vl-1.6", lambda r: "recognized text", 16),
            "llm": FakeBackend("llm", lambda r: json.dumps({"title": "T"})),
            "ref": FakeBackend("ref", lambda r: json.dumps(REF_RECORD), 32),
            "slow": SlowBackend("slow", lambda r: "late"),
        }
    )
    zm.install(zm.build_app(zm.BACKENDS, runner=runner))


def test_models_listing_and_unknown_alias():
    async def go():
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{zm.BASE_URL}/v1/models")
            assert {m["id"] for m in r.json()["data"]} == set(zm.BACKENDS)
            r = await c.post(f"{zm.BASE_URL}/v1/chat/completions", json={"model": "nope", "messages": []})
            assert r.status_code == 404

    asyncio.run(go())


def test_paddle_http_client_batches_regions_in_visitor_context():
    from PIL import Image

    from bibr.local.ocr import PaddleHttpOcrClient

    async def run():
        VISITOR.set("visitor-A")
        client = PaddleHttpOcrClient(base_url=zm.BASE_URL)
        if hasattr(client, "load"):
            loaded = client.load()
            if asyncio.iscoroutine(loaded):
                await loaded
        await client.wait_for_server()
        img = Image.new("RGB", (64, 32), "white")
        return await asyncio.gather(*(client.recognize(img, "OCR:") for _ in range(10)))

    texts = asyncio.run(run())
    assert [str(t) for t in texts] == ["recognized text"] * 10
    batches = zm.BACKENDS["paddle-ocr-vl-1.6"].batches
    assert sum(batches) == 10 and len(batches) <= 2, batches
    assert seen_visitors[-1] == "visitor-A"


def test_ref_llm_client_sends_one_batch():
    from bibr.clients import ref_llm
    from bibr.config import snapshot_settings

    settings = snapshot_settings()
    settings.REF_LLM_BASE_URL = f"{zm.BASE_URL}/v1"
    settings.REF_LLM_MODEL = "ref"
    settings.REF_LLM_MAX_CONCURRENCY = 32
    refs = [f"Doe, J. (2020). A study of things {i}. Journal, 1, 1-2." for i in range(12)]

    async def run():
        VISITOR.set("visitor-B")
        return await ref_llm.parse_batch(refs, settings)

    out = asyncio.run(run())
    assert len(out) == 12
    assert zm.BACKENDS["ref"].batches == [12]
    assert seen_visitors[-1] == "visitor-B"


def test_openai_sdk_structured_request():
    from openai import AsyncOpenAI

    async def run():
        VISITOR.set("visitor-C")
        client = AsyncOpenAI(base_url=f"{zm.BASE_URL}/v1", api_key="not-needed")
        return await client.chat.completions.create(
            model="llm",
            messages=[{"role": "user", "content": "extract"}],
            response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}},
            max_tokens=64,
        )

    resp = asyncio.run(run())
    assert json.loads(resp.choices[0].message.content) == {"title": "T"}
    assert resp.choices[0].finish_reason == "stop"
    assert seen_visitors[-1] == "visitor-C"


def test_chat_request_parsing():
    req = zm.ChatRequest.from_body(
        {
            "messages": [],
            "max_tokens": 10,
            "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
            "chat_template_kwargs": {"enable_thinking": False},
        },
        default_max_tokens=99,
    )
    assert req.max_tokens == 10 and req.json_schema == {"type": "object"}
    assert req.chat_template_kwargs == {"enable_thinking": False}
    assert zm.ChatRequest.from_body({"messages": []}, 99).max_tokens == 99


def test_client_read_timeout_fires_through_the_shim():
    async def run():
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=0.2)) as c:
            await c.post(
                f"{zm.BASE_URL}/v1/chat/completions",
                json={"model": "slow", "messages": [{"role": "user", "content": "x"}]},
            )

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(run())


def test_structured_and_plain_requests_batch_separately():
    backend = zm.BACKENDS["llm"]
    backend.batches.clear()
    schema = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}

    async def one(c, structured):
        body = {"model": "llm", "messages": [{"role": "user", "content": "x"}], "max_tokens": 8}
        if structured:
            body["response_format"] = schema
        return await c.post(f"{zm.BASE_URL}/v1/chat/completions", json=body)

    async def run():
        async with httpx.AsyncClient() as c:
            return await asyncio.gather(*(one(c, i % 2 == 0) for i in range(6)))

    responses = asyncio.run(run())
    assert all(r.status_code == 200 for r in responses)
    assert sorted(backend.batches) == [3, 3]


def test_paper_mode_runner_calls_the_models_directly():
    """In paper mode the shim runs inside the paper's GPU window: no GPU call per batch."""
    zm.install(zm.build_app(zm.BACKENDS, runner=zm.generate_batch))
    try:
        backend = zm.BACKENDS["llm"]
        backend.batches.clear()

        async def run():
            async with httpx.AsyncClient() as c:
                body = {"model": "llm", "messages": [{"role": "user", "content": "x"}], "max_tokens": 8}
                return await asyncio.gather(
                    *(c.post(f"{zm.BASE_URL}/v1/chat/completions", json=body) for _ in range(3))
                )

        responses = asyncio.run(run())
        assert all(r.status_code == 200 for r in responses)
        assert backend.batches == [3]
    finally:
        zm.install(zm.build_app(zm.BACKENDS, runner=runner))
