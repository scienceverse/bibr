"""LLMClient._record_trace / InstructorBackend capture-site tests (Task 8).

Complements tests/export/test_llm_trace.py (model shape, export wiring) with
tests against the actual capture site: LLMClient._record_trace and its
integration into InstructorBackend.create().
"""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

import pytest

from bibr.clients.llm import (
    LLMClient,
    _sanitize_trace_value,
    _trace_message_text,
    _usage_label,
    usage_file_context,
)
from bibr.config import GlobalSettings
from bibr.export.models import LlmTraceExport
from bibr.schemas import TitleKeywordsLLM


def _settings(**llm_overrides) -> GlobalSettings:
    settings = GlobalSettings()
    for key, value in llm_overrides.items():
        setattr(settings.llm, key, value)
    return settings


def _fake_completion(content: str, finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class _FakeInstructor:
    def __init__(self, result, completion):
        self._result = result
        self._completion = completion

    async def create_with_completion(self, *, response_model, messages, max_retries, **kw):  # noqa: ARG002
        return self._result, self._completion


def test_capture_off_by_default_records_nothing(monkeypatch):
    client = LLMClient(settings=_settings())  # capture_trace defaults False
    assert client._settings.llm.capture_trace is False
    completion = _fake_completion('{"title": "x"}')
    monkeypatch.setattr(
        client, "_get_client", lambda: _FakeInstructor(TitleKeywordsLLM(title="x"), completion)
    )

    async def _run():
        with usage_file_context("nokey#1"):
            await client._backend.create(
                response_model=TitleKeywordsLLM,
                system="SYS",
                messages=[{"role": "user", "content": "hi"}],
                want_completion=True,
            )

    import asyncio

    asyncio.run(_run())
    assert client.traces_pop_file("nokey#1") == []


def test_capture_records_scrubbed_row_when_enabled(monkeypatch):
    client = LLMClient(settings=_settings(capture_trace=True, track_usage=True))
    secret = "sk-" + "test-value"
    completion = _fake_completion(f'{{"title": "x", "leak": "{secret}"}}')
    monkeypatch.setattr(
        client, "_get_client", lambda: _FakeInstructor(TitleKeywordsLLM(title="x"), completion)
    )

    async def _run():
        with usage_file_context("capkey#1"):
            token = _usage_label.set("extract_title_keywords")
            try:
                await client._backend.create(
                    response_model=TitleKeywordsLLM,
                    system="SYS",
                    messages=[{"role": "user", "content": f"use key {secret} to auth"}],
                    want_completion=True,
                )
            finally:
                _usage_label.reset(token)

    import asyncio

    asyncio.run(_run())
    rows = client.traces_pop_file("capkey#1")
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "extract_title_keywords"
    assert row["provider"] == client._settings.llm.provider
    assert row["model"] == client._settings.llm.model
    assert row["parsed_ok"] is True
    assert row["attempt"] == 1
    assert row["error"] is None
    assert row["finish_reason"] == "stop"
    # Secret must not survive into either the prompt or the completion.
    assert secret not in str(row["messages"])
    assert secret not in row["raw_completion"]
    assert "[REDACTED]" in str(row["messages"])
    assert "[REDACTED]" in row["raw_completion"]
    # Resolved sampling params were captured, not dropped.
    assert row["params"]

    # Popped bucket is gone.
    assert client.traces_pop_file("capkey#1") == []


def test_capture_noop_without_active_file_context(monkeypatch):
    client = LLMClient(settings=_settings(capture_trace=True, track_usage=True))
    completion = _fake_completion('{"title": "x"}')
    monkeypatch.setattr(
        client, "_get_client", lambda: _FakeInstructor(TitleKeywordsLLM(title="x"), completion)
    )

    async def _run():
        # No usage_file_context active — _usage_file_hash.get() is None.
        await client._backend.create(
            response_model=TitleKeywordsLLM,
            system="SYS",
            messages=[{"role": "user", "content": "hi"}],
            want_completion=True,
        )

    import asyncio

    asyncio.run(_run())
    assert client._traces_by_file == {}


async def test_close_clears_traces_by_file():
    client = LLMClient(settings=_settings(capture_trace=True))
    client._traces_by_file["leftover#1"] = [{"label": "x"}]
    await client.close()
    assert client._traces_by_file == {}


# ── fix round 1, item 1: params must be JSON-safe at capture time ───────


class _Mode(Enum):
    JSON = "json"


class _Sentinel:
    def __repr__(self) -> str:
        return "<NotGiven>"


def test_unsanitized_params_construct_but_fail_json_dump_documenting_the_gap():
    """params: dict is untyped — pydantic doesn't validate its values. A raw
    non-JSON-primitive value (e.g. an SDK sentinel like instructor's
    NotGiven, modeled here by _Sentinel — a plain object pydantic has no
    special-case for, unlike Enum, which it can already serialize via
    .value) constructs a valid row and only blows up later at
    model_dump(mode="json"). This is exactly the gap _sanitize_trace_value
    closes at capture time — documented here so nobody "fixes" it by
    loosening the model instead of sanitizing at the source."""
    row = LlmTraceExport(label="x", parsed_ok=True, params={"sentinel": _Sentinel()})
    with pytest.raises(Exception):  # noqa: PT011, B017 — pydantic-core PydanticSerializationError
        row.model_dump(mode="json")


def test_sanitize_trace_value_coerces_non_primitives_to_repr_strings():
    params = {
        "temperature": 0.0,
        "mode": _Mode.JSON,
        "sentinel": _Sentinel(),
        "nested": {"inner": _Sentinel()},
        "list_of_things": [_Sentinel(), 1, "ok"],
    }
    sanitized = _sanitize_trace_value(params)

    assert sanitized["temperature"] == 0.0
    assert isinstance(sanitized["mode"], str)
    assert sanitized["sentinel"] == "<NotGiven>"
    assert sanitized["nested"]["inner"] == "<NotGiven>"
    assert sanitized["list_of_things"] == ["<NotGiven>", 1, "ok"]

    # Now safe to construct and JSON-dump.
    row = LlmTraceExport(label="x", parsed_ok=True, params=sanitized)
    dumped = row.model_dump(mode="json")
    assert dumped["params"]["sentinel"] == "<NotGiven>"


# ── fix round 1, item 2: params must be scrubbed of credentials too ─────


def test_sanitize_trace_value_scrubs_credentials_in_nested_strings():
    params = {
        "base_url": "https://example.org",
        "leaked": "token " + ("sk-" + "test-value") + " here",
        "nested": {"key": "AIzaSyFAKEKEY"},
        "list": ["sv_1a2b3c4d5e6f7g8h9i0j", "harmless"],
    }
    sanitized = _sanitize_trace_value(params)

    assert ("sk-" + "test-value") not in sanitized["leaked"]
    assert "[REDACTED]" in sanitized["leaked"]
    assert "AIzaSyFAKEKEY" not in sanitized["nested"]["key"]
    assert "sv_1a2b3c4d5e6f7g8h9i0j" not in sanitized["list"][0]
    assert sanitized["list"][1] == "harmless"


def test_record_trace_sanitizes_and_scrubs_params_end_to_end():
    """Exercises the real capture path (_record_trace, not the helper
    directly): a secret-bearing, non-JSON-primitive params dict must come out
    scrubbed, JSON-safe, and round-trip through the pydantic model."""
    client = LLMClient(settings=_settings(capture_trace=True, track_usage=True))
    completion = _fake_completion('{"title": "x"}')

    with usage_file_context("paramkey#1"):
        client._record_trace(
            messages=[{"role": "user", "content": "hi"}],
            completion=completion,
            params={
                "temperature": 0.0,
                "leak": "use " + ("sk-" + "test-value") + " now",
                "mode": _Mode.JSON,
            },
            parsed_ok=True,
        )

    row = client.traces_pop_file("paramkey#1")[0]
    assert row["params"]["temperature"] == 0.0
    assert ("sk-" + "test-value") not in row["params"]["leak"]
    assert "[REDACTED]" in row["params"]["leak"]
    assert isinstance(row["params"]["mode"], str)

    validated = LlmTraceExport(**row)
    validated.model_dump(mode="json")  # must not raise


# ── fix round 1, item 3: nuextract-native + capture_trace warns loudly ──


def test_warns_when_capture_trace_on_and_backend_resolves_to_nuextract_native(caplog):
    settings = _settings(
        provider="openai",
        base_url="http://127.0.0.1:8767/v1",
        model="numind/NuExtract3-mlx-8bits",
        structured_backend="nuextract-native",
        capture_trace=True,
    )
    with caplog.at_level("WARNING", logger="bibr.clients.llm"):
        LLMClient(settings=settings)

    assert "not yet instrumented" in caplog.text
    assert "nuextract-native" in caplog.text


def test_no_warning_when_capture_trace_off_with_nuextract_native(caplog):
    settings = _settings(
        provider="openai",
        base_url="http://127.0.0.1:8767/v1",
        model="numind/NuExtract3-mlx-8bits",
        structured_backend="nuextract-native",
        capture_trace=False,
    )
    with caplog.at_level("WARNING", logger="bibr.clients.llm"):
        LLMClient(settings=settings)

    assert "not yet instrumented" not in caplog.text


def test_no_warning_when_capture_trace_on_with_instructor_backend(caplog):
    with caplog.at_level("WARNING", logger="bibr.clients.llm"):
        LLMClient(settings=_settings(capture_trace=True))

    assert "not yet instrumented" not in caplog.text


# ── fix round 1, item 4: anthropic cache-marker parts render as text ────


def test_trace_message_text_joins_content_block_list():
    content = [
        {"type": "text", "text": "Document prefix", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "\nRemainder"},
    ]
    assert _trace_message_text(content) == "Document prefix\nRemainder"


def test_trace_message_text_handles_plain_string_and_none():
    assert _trace_message_text("hello") == "hello"
    assert _trace_message_text(None) == ""


def test_record_trace_renders_cache_marker_parts_as_text_not_repr():
    """Content shaped like Anthropic's post-transform_messages blocks must
    render as prompt text in the trace row, not Python repr noise — while
    still being scrubbed of any embedded credential."""
    client = LLMClient(settings=_settings(capture_trace=True, track_usage=True))
    completion = _fake_completion('{"title": "x"}')
    content = [
        {
            "type": "text",
            "text": "Document body with key sk-livekey123",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    with usage_file_context("anthropickey#1"):
        client._record_trace(
            messages=[{"role": "user", "content": content}],
            completion=completion,
            params={},
            parsed_ok=True,
        )

    rendered = client.traces_pop_file("anthropickey#1")[0]["messages"][0]["content"]
    assert "cache_control" not in rendered
    assert "{'type': 'text'" not in rendered
    assert ("sk-" + "test-value") not in rendered
    assert "[REDACTED]" in rendered
    assert "Document body with key" in rendered
