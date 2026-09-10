"""`LlmClient` Protocol + LLMClient conformance."""

import inspect
import sys


def test_llm_client_protocol_exists():
    from bibr.clients.llm_protocol import LlmClient

    assert getattr(LlmClient, "_is_runtime_protocol", False) is True


def test_llm_client_protocol_declares_public_surface():
    """Protocol must declare exactly the methods callers use."""
    from bibr.clients.llm_protocol import LlmClient

    for method in (
        "extract_core_metadata",
        "extract_references",
        "extract_equations",
        "resolve_citations",
        "invoke_structured",
        "close",
    ):
        assert hasattr(LlmClient, method), f"Protocol missing method: {method}"
    # limiter is a property descriptor on the protocol class
    # (__protocol_attrs__ would be the direct check but only exists on 3.12+)
    assert isinstance(inspect.getattr_static(LlmClient, "limiter"), property)


def test_llm_client_protocol_methods_are_coroutines():
    from bibr.clients.llm_protocol import LlmClient

    for method in (
        "extract_core_metadata",
        "extract_references",
        "extract_equations",
        "resolve_citations",
        "invoke_structured",
        "close",
    ):
        fn = getattr(LlmClient, method)
        assert inspect.iscoroutinefunction(fn), f"{method} must be a coroutinefunction"


def test_llm_client_protocol_does_not_require_create():
    """Regression pin: ``create`` was an instructor-internal method that the
    real ``LLMClient`` does not expose. The Protocol must not re-introduce it."""
    from bibr.clients.llm_protocol import LlmClient

    proto_methods = {m for m in dir(LlmClient) if not m.startswith("_")}
    assert "create" not in proto_methods, proto_methods


def test_real_llm_client_satisfies_protocol():
    """Structural conformance: a real ``LLMClient`` instance must satisfy the
    ``runtime_checkable`` Protocol via ``isinstance``."""
    from bibr.clients.llm import LLMClient
    from bibr.clients.llm_protocol import LlmClient

    # Use __new__ to avoid requiring API keys at construction time.
    instance = LLMClient.__new__(LLMClient)

    # Static surface check — version-independent.
    for attr in (
        "extract_core_metadata",
        "extract_references",
        "segment_references",
        "extract_equations",
        "resolve_citations",
        "invoke_structured",
        "close",
    ):
        assert callable(inspect.getattr_static(LLMClient, attr)), attr
    assert isinstance(inspect.getattr_static(LLMClient, "limiter"), property)

    if sys.version_info >= (3, 12):
        # 3.11's isinstance() invokes descriptors during protocol checks
        # (getattr_static lookup landed in 3.12), so the ``limiter`` property
        # explodes on the half-built instance there. 3.12+ checks it safely.
        assert isinstance(instance, LlmClient)


def test_extract_core_metadata_protocol_signature_matches_production():
    from bibr.clients.llm import LLMClient
    from bibr.clients.llm_protocol import LlmClient

    def parameter_contract(method):
        return [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in inspect.signature(method).parameters.values()
        ]

    assert parameter_contract(LlmClient.extract_core_metadata) == parameter_contract(
        LLMClient.extract_core_metadata
    )


def test_llm_provider_protocol_exists():
    from bibr.clients.providers.base import LlmProvider

    assert getattr(LlmProvider, "_is_runtime_protocol", False) is True
    # Check methods directly; ClassVar 'name' is in __annotations__
    assert hasattr(LlmProvider, "build_client"), "missing build_client"
    assert hasattr(LlmProvider, "call_kwargs"), "missing call_kwargs"
    assert "name" in LlmProvider.__annotations__, "missing name annotation"


def test_llm_provider_registry_register_and_get():
    import pytest

    from bibr.clients import providers

    class _FakeProvider:
        name = "fake-provider"

        def build_client(self):
            return object()

        def call_kwargs(self, reasoning_effort, max_tokens=None):
            return {}

    providers.register(_FakeProvider)
    got = providers.get("fake-provider")
    assert isinstance(got, _FakeProvider)

    with pytest.raises(ValueError):
        providers.get("not-registered")


def test_llm_provider_registry_duplicate_name_raises():
    import pytest

    from bibr.clients import providers

    class _Dup:
        name = "dup-provider"

        def build_client(self):
            return object()

        def call_kwargs(self, reasoning_effort, max_tokens=None):
            return {}

    class _DupConflict:
        name = "dup-provider"

        def build_client(self):
            return object()

        def call_kwargs(self, reasoning_effort, max_tokens=None):
            return {}

    providers.register(_Dup)
    with pytest.raises(ValueError, match="already registered"):
        providers.register(_DupConflict)


def test_llm_provider_registry_missing_name_raises():
    import pytest

    from bibr.clients import providers

    class _NoName:
        def build_client(self):
            return object()

        def call_kwargs(self, reasoning_effort, max_tokens=None):
            return {}

    with pytest.raises(ValueError, match="class-level `name` attribute"):
        providers.register(_NoName)


def test_all_bundled_providers_registered():
    from bibr.clients import providers

    for name in ("google", "openai", "anthropic", "groq", "ollama"):
        assert name in providers.known_providers(), f"{name} not registered"


def test_google_provider_call_kwargs_uses_legal_default_thinking_budget():
    """gemini-3 family models default to thinking-on, which makes the model
    emit reasoning tokens alongside the structured tool call. Instructor
    rejects that with 'Instructor does not support multiple function calls'.
    The provider must always pass an explicit ``thinking_config`` so model
    behavior is deterministic regardless of model version."""
    import os

    os.environ["LLM_PROVIDER"] = "google"
    os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")

    from bibr.clients import providers
    from bibr.config import Settings

    provider = providers.get("google")
    kwargs = provider.call_kwargs(reasoning_effort=None)
    assert kwargs == {
        "generation_config": {
            # Deterministic extraction: temperature 0 (matches the OpenAI adapter).
            "temperature": 0.0,
            "max_tokens": Settings.llm.max_tokens,
        },
        "thinking_config": {"thinking_budget": 1},
    }


def test_google_provider_thinking_budget_passthrough(monkeypatch):
    """``Settings.llm.thinking_budget > 0`` must surface as the actual
    thinking_budget passed through to the Gemini API (mirroring the
    Anthropic provider)."""
    from bibr.clients import providers
    from bibr.config import Settings

    monkeypatch.setattr(Settings.llm, "thinking_budget", 2048)
    provider = providers.get("google")
    kwargs = provider.call_kwargs(reasoning_effort=None)
    assert kwargs["thinking_config"] == {"thinking_budget": 2048}


def test_openai_provider_honours_reasoning_effort(monkeypatch):
    from bibr.clients import providers
    from bibr.config import Settings

    monkeypatch.setattr(Settings.llm, "base_url", None)
    monkeypatch.setattr(Settings.llm, "temperature", 0.2)

    provider = providers.get("openai")
    kwargs = provider.call_kwargs(reasoning_effort="high")
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_completion_tokens"] == Settings.llm.max_tokens
    assert kwargs["reasoning_effort"] == "high"


def test_openai_provider_per_call_max_tokens_override_without_base_url(monkeypatch):
    """No base_url (real OpenAI): a per-call ``max_tokens`` override still wins
    over the configured default, unchanged from current behavior."""
    from bibr.clients import providers
    from bibr.config import Settings

    monkeypatch.setattr(Settings.llm, "base_url", None)

    provider = providers.get("openai")
    kwargs = provider.call_kwargs(reasoning_effort=None, max_tokens=8192)
    assert kwargs["max_completion_tokens"] == 8192


def test_openai_provider_omits_cap_for_self_hosted_without_explicit_setting(monkeypatch):
    """Self-hosted servers (base_url set) reject caps above their unknown
    max_model_len. When the user hasn't explicitly set LLM_MAX_TOKENS, the
    global cap must be omitted."""
    from bibr.clients import providers
    from bibr.config import Settings

    monkeypatch.setattr(Settings.llm, "base_url", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(
        Settings.llm,
        "__pydantic_fields_set__",
        Settings.llm.model_fields_set - {"max_tokens"},
    )

    provider = providers.get("openai")

    kwargs = provider.call_kwargs(reasoning_effort=None)
    assert "max_completion_tokens" not in kwargs


def test_openai_provider_keeps_per_call_cap_for_self_hosted(monkeypatch):
    """A task-specific cap is intentional and must survive the custom-base-URL
    compatibility path even when LLM_MAX_TOKENS itself was not configured."""
    from bibr.clients import providers
    from bibr.config import Settings

    monkeypatch.setattr(Settings.llm, "base_url", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(
        Settings.llm,
        "__pydantic_fields_set__",
        Settings.llm.model_fields_set - {"max_tokens"},
    )

    provider = providers.get("openai")
    kwargs_override = provider.call_kwargs(reasoning_effort=None, max_tokens=8192)
    assert kwargs_override["max_tokens"] == 8192
    assert "max_completion_tokens" not in kwargs_override


def test_openai_provider_keeps_cap_for_self_hosted_with_explicit_setting(monkeypatch):
    """When the user explicitly sets LLM_MAX_TOKENS, the cap is kept even with
    base_url set — that's the opt-in for users who want a hard cap. Self-hosted
    OpenAI-compatible servers such as vllm-mlx commonly expose this as
    ``max_tokens`` rather than OpenAI's newer ``max_completion_tokens``. A
    per-call override still wins over the configured setting."""
    from bibr.clients import providers
    from bibr.config import Settings

    monkeypatch.setattr(Settings.llm, "base_url", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(Settings.llm, "max_tokens", 4096)
    monkeypatch.setattr(
        Settings.llm,
        "__pydantic_fields_set__",
        Settings.llm.model_fields_set | {"max_tokens"},
    )

    provider = providers.get("openai")

    kwargs = provider.call_kwargs(reasoning_effort=None)
    assert kwargs["max_tokens"] == 4096
    assert "max_completion_tokens" not in kwargs

    kwargs_override = provider.call_kwargs(reasoning_effort=None, max_tokens=8192)
    assert kwargs_override["max_tokens"] == 8192


def test_openai_provider_uses_json_schema_for_custom_base_url(monkeypatch):
    """Local OpenAI-compatible servers (LM Studio, vLLM, etc.) often reject
    instructor's default TOOLS mode (object-form ``tool_choice``). When a
    custom ``base_url`` is configured we must build the client in
    ``JSON_SCHEMA`` mode, which those servers do accept."""
    import instructor

    from bibr.clients import providers
    from bibr.config import Settings

    monkeypatch.setattr(Settings.llm, "api_key", "test-key")
    monkeypatch.setattr(Settings.llm, "base_url", "http://127.0.0.1:1234/v1")
    monkeypatch.setattr(Settings.llm, "model", "google/gemma-4-12b-qat")

    client = providers.get("openai").build_client()
    assert client.mode == instructor.Mode.JSON_SCHEMA


def test_openai_provider_keeps_default_mode_without_base_url(monkeypatch):
    """Against real OpenAI (no custom base_url) the provider must keep
    instructor's default TOOLS mode — JSON_SCHEMA is opt-in for local servers."""
    import instructor

    from bibr.clients import providers
    from bibr.config import Settings

    monkeypatch.setattr(Settings.llm, "api_key", "test-key")
    monkeypatch.setattr(Settings.llm, "base_url", None)
    monkeypatch.setattr(Settings.llm, "model", "gpt-4o")

    client = providers.get("openai").build_client()
    assert client.mode == instructor.Mode.TOOLS


def test_anthropic_provider_thinking_budget(monkeypatch):
    from bibr.clients import providers
    from bibr.config import Settings

    # Temporarily override Settings.llm.thinking_budget to test the behavior
    monkeypatch.setattr(Settings.llm, "thinking_budget", 2048)
    provider = providers.get("anthropic")
    kwargs = provider.call_kwargs(reasoning_effort=None)
    assert kwargs["max_tokens"] == Settings.llm.max_tokens
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_anthropic_provider_marks_cacheable_parts_for_prompt_cache():
    """Anthropic prompt-cache markers must land on the content parts the
    builders flagged cacheable (the fenced document / static instruction),
    so the large shared prefix is reused across calls. Tiny system prompts
    are below Anthropic's minimum cacheable length and get no marker."""
    from bibr.clients import providers
    from bibr.clients.prompts import part

    provider = providers.get("anthropic")
    assert hasattr(provider, "transform_messages"), "anthropic must transform messages"

    out = provider.transform_messages(
        [
            {"role": "system", "content": "You are an extractor."},
            {"role": "user", "content": [part("fenced doc", cache=True), part("instruction")]},
            {"role": "user", "content": "plain string"},
        ]
    )
    # System stays a plain string (no wasted cache breakpoint).
    assert out[0] == {"role": "system", "content": "You are an extractor."}
    blocks = out[1]["content"]
    assert blocks[0] == {
        "type": "text",
        "text": "fenced doc",
        "cache_control": {"type": "ephemeral"},
    }
    assert blocks[1] == {"type": "text", "text": "instruction"}
    # Non-part content is untouched.
    assert out[2] == {"role": "user", "content": "plain string"}


def test_other_providers_do_not_advertise_transform_messages():
    """Only the Anthropic adapter applies prompt-cache markers — the rest must
    leave messages alone (no incidental dependency on a default passthrough)."""
    from bibr.clients import providers

    for name in ("google", "openai", "groq", "ollama"):
        assert not hasattr(providers.get(name), "transform_messages"), name


def test_groq_provider_call_kwargs():
    from bibr.clients import providers
    from bibr.config import Settings

    provider = providers.get("groq")
    kwargs = provider.call_kwargs(reasoning_effort=None)
    assert kwargs == {"temperature": 0.0, "max_tokens": Settings.llm.max_tokens}


def test_ollama_provider_call_kwargs():
    from bibr.clients import providers
    from bibr.config import Settings

    provider = providers.get("ollama")
    kwargs = provider.call_kwargs(reasoning_effort=None)
    assert kwargs == {
        "temperature": 0.0,
        "extra_body": {"options": {"num_predict": Settings.llm.max_tokens}},
    }


def test_llm_client_dispatches_via_provider_registry(monkeypatch):
    """``LLMClient._get_client`` resolves the provider adapter, not the legacy `if/elif`."""
    from bibr.clients import llm as llm_mod
    from bibr.clients import providers
    from bibr.config import GlobalSettings

    captured: dict = {}

    class _StubProvider:
        name = "stub"

        def build_client(self):
            captured["built"] = True
            return object()

        def call_kwargs(self, reasoning_effort, max_tokens=None):
            return {"stub": True, "reasoning_effort": reasoning_effort, "max_tokens": max_tokens}

    providers.register(_StubProvider)

    settings = GlobalSettings()
    settings.llm.provider = "stub"

    client = llm_mod.LLMClient(settings=settings)
    client._get_client()
    assert captured["built"] is True

    kwargs = llm_mod._build_call_kwargs(reasoning_effort="medium", settings=settings)
    assert kwargs == {"stub": True, "reasoning_effort": "medium", "max_tokens": None}
    # The per-call max_tokens override is threaded through to the provider.
    capped = llm_mod._build_call_kwargs(
        reasoning_effort="medium", max_tokens=4096, settings=settings
    )
    assert capped["max_tokens"] == 4096
