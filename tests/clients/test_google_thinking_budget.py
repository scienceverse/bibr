"""Gemini thinking-budget resolution.

An illegal ``thinkingBudget`` is a 400 INVALID_ARGUMENT that kills every LLM
call in the pipeline, so the cases below are pinned to what the live API
actually accepted on 2026-07-25 rather than to what the docs imply.
"""

import json

import httpx
import instructor
import pytest
from pydantic import BaseModel

from bibr.clients.providers.google import (
    GoogleProvider,
    _resolve_thinking_budget,
    _resolve_thinking_level,
    _uses_thinking_level,
)


class _Llm:
    def __init__(self, model, thinking_budget, max_tokens=4096):
        self.model = model
        self.thinking_budget = thinking_budget
        self.max_tokens = max_tokens
        self.api_key = None


class _Settings:
    def __init__(self, model, thinking_budget):
        self.llm = _Llm(model, thinking_budget)
        self.GOOGLE_API_KEY = "test-key"


def test_default_model_cannot_disable_thinking_so_zero_becomes_legal():
    # gemini-3.5-flash-lite is bibr's default model and rejects budget 0.
    assert _resolve_thinking_budget("gemini-3.5-flash-lite", 0) == 1


def test_models_that_support_disabling_keep_zero():
    for model in ("gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite"):
        assert _resolve_thinking_budget(model, 0) == 0


def test_zero_is_honoured_on_a_model_that_can_disable_despite_a_high_minimum():
    # 2.5-flash-lite accepts 0 (disabled) even though its minimum *budget* is 512.
    assert _resolve_thinking_budget("gemini-2.5-flash-lite", 0) == 0


def test_positive_budget_is_raised_to_the_model_minimum():
    # A blanket clamp-to-1 would 400 here; 2.5-flash-lite rejects everything <512.
    assert _resolve_thinking_budget("gemini-2.5-flash-lite", 1) == 512
    assert _resolve_thinking_budget("gemini-2.5-flash-lite", 511) == 512
    assert _resolve_thinking_budget("gemini-2.5-flash-lite", 1024) == 1024


def test_negative_budget_means_dynamic_and_is_no_longer_clamped_away():
    # The previous max(0, ...) made dynamic thinking unreachable.
    for model in ("gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "gemini-3.5-flash"):
        assert _resolve_thinking_budget(model, -1) == -1
        assert _resolve_thinking_budget(model, -999) == -1


def test_floating_latest_aliases_are_treated_as_undisableable():
    for model in ("gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.6-flash"):
        assert _resolve_thinking_budget(model, 0) == 1


def test_pinned_point_releases_inherit_family_rules():
    assert _resolve_thinking_budget("gemini-3.5-flash-lite-001", 0) == 1
    assert _resolve_thinking_budget("gemini-2.5-flash-lite-001", 1) == 512


def test_unknown_models_keep_the_configured_budget():
    assert _resolve_thinking_budget("some-future-model", 0) == 0
    assert _resolve_thinking_budget("some-future-model", 256) == 256
    assert _resolve_thinking_budget("", 0) == 0


def test_call_kwargs_emits_a_legal_budget_for_the_default_model():
    kwargs = GoogleProvider(_Settings("gemini-3.5-flash-lite", 0)).call_kwargs(None)
    assert kwargs["thinking_config"] == {"thinking_budget": 1}
    assert kwargs["generation_config"]["temperature"] == 0.0


def test_call_kwargs_still_disables_thinking_where_the_model_allows_it():
    kwargs = GoogleProvider(_Settings("gemini-3.1-flash-lite", 0)).call_kwargs(None)
    assert kwargs["thinking_config"] == {"thinking_budget": 0}


# --- models that take a thinking level ---------------------------------------
#
# From the vendor's model documentation (2026-09-21), not the live API: these
# models take ``thinking_level`` (low/medium/high) instead of a budget, and the
# sampling parameters are stripped from their requests.


def test_thinking_level_models_are_matched_by_family_prefix():
    for model in ("gemini-3.8-flash", "gemini-3.8-flash-001", " gemini-3.8-flash "):
        assert _uses_thinking_level(model)
    for model in (
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3-flash-preview",
        "gemini-flash-latest",
        "gemini-2.5-flash-lite",
        "",
    ):
        assert not _uses_thinking_level(model)


def test_budget_maps_to_the_documented_thinking_levels():
    # 0 ("disabled") becomes the lowest level these models accept.
    assert _resolve_thinking_level(0) == "low"
    assert _resolve_thinking_level(1) == "low"
    assert _resolve_thinking_level(1024) == "low"
    assert _resolve_thinking_level(1025) == "medium"
    assert _resolve_thinking_level(8192) == "medium"
    assert _resolve_thinking_level(8193) == "high"
    assert _resolve_thinking_level(32768) == "high"
    # Negative ("let the model decide") sends no level: the model default.
    assert _resolve_thinking_level(-1) is None


def test_request_for_a_budget_model_is_unchanged():
    kwargs = GoogleProvider(_Settings("gemini-3.5-flash-lite", 0)).call_kwargs(None)
    assert kwargs == {
        "generation_config": {"temperature": 0.0, "max_tokens": 4096},
        "thinking_config": {"thinking_budget": 1},
    }
    kwargs = GoogleProvider(_Settings("gemini-3-flash-preview", 0)).call_kwargs(None, 512)
    assert kwargs == {
        "generation_config": {"temperature": 0.0, "max_tokens": 512},
        "thinking_config": {"thinking_budget": 0},
    }


def test_request_for_a_thinking_level_model_sends_a_level_and_no_sampling_params():
    kwargs = GoogleProvider(_Settings("gemini-3.8-flash", 0)).call_kwargs(None)
    assert kwargs == {
        "generation_config": {"max_tokens": 4096},
        "thinking_config": {"thinking_level": "low"},
    }
    kwargs = GoogleProvider(_Settings("gemini-3.8-flash", 16384)).call_kwargs(None, 512)
    assert kwargs == {
        "generation_config": {"max_tokens": 512},
        "thinking_config": {"thinking_level": "high"},
    }


def test_dynamic_budget_leaves_a_thinking_level_model_on_its_default():
    kwargs = GoogleProvider(_Settings("gemini-3.8-flash", -1)).call_kwargs(None)
    assert kwargs == {"generation_config": {"max_tokens": 4096}}


class _Ping(BaseModel):
    reply: str


def _function_call_response() -> httpx.Response:
    call = {"name": _Ping.__name__, "args": {"reply": "OK"}}
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"functionCall": call}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
        },
    )


@pytest.mark.parametrize(
    ("model", "expected_config", "expected_thinking"),
    [
        (
            "gemini-3.5-flash-lite",
            {"temperature": 0.0, "maxOutputTokens": 4096},
            {"thinkingbudget": 1},
        ),
        ("gemini-3.8-flash", {"maxOutputTokens": 4096}, {"thinkinglevel": "LOW"}),
    ],
)
async def test_wire_request_through_instructor_and_the_sdk(
    monkeypatch, model, expected_config, expected_thinking
):
    """What the installed instructor + google-genai put on the wire (mocked transport)."""
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return _function_call_response()

    transport = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    real_from_provider = instructor.from_provider

    def from_provider(*args, **kwargs):
        return real_from_provider(*args, http_options={"httpx_async_client": transport}, **kwargs)

    monkeypatch.setattr(instructor, "from_provider", from_provider)
    provider = GoogleProvider(_Settings(model, 0))
    try:
        result = await provider.build_client().create(
            response_model=_Ping,
            messages=[{"role": "user", "content": "Reply with OK"}],
            **provider.call_kwargs(None),
        )
    finally:
        await transport.aclose()

    assert result.reply == "OK"
    assert len(bodies) == 1
    config = bodies[0]["generationConfig"]
    thinking = config.pop("thinkingConfig")
    assert config == expected_config
    # The locked SDK serialises this nested config's keys in snake_case, as
    # it always has for the budget; compare spelling-insensitively so an SDK
    # switch to camelCase is not mistaken for a request change.
    assert {k.replace("_", "").lower(): v for k, v in thinking.items()} == expected_thinking
