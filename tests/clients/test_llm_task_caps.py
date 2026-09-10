from types import SimpleNamespace
from unittest import mock

from bibr.clients.llm import LLMClient, _task_max_tokens
from bibr.config import GlobalSettings
from bibr.schemas import AuthorsLLM, PaperClassificationLLM, PaperTypeLabel, TitleKeywordsLLM


def _settings(**llm):
    return GlobalSettings(llm=llm)


def test_task_cap_is_bounded_by_global_and_zero_none_disable():
    settings = _settings(max_tokens=2000)
    assert _task_max_tokens(settings, 4096) == 2000
    assert _task_max_tokens(settings, 1000) == 1000
    assert _task_max_tokens(settings, 0) is None
    assert _task_max_tokens(settings, None) is None


def test_ollama_provider_uses_per_call_task_cap():
    from bibr.clients import providers

    settings = _settings(provider="ollama", max_tokens=2000)
    provider = providers.get("ollama", settings=settings)

    assert provider.call_kwargs(None, max_tokens=1200) == {
        "temperature": 0.0,
        "extra_body": {"options": {"num_predict": 1200}},
    }


def test_ollama_provider_uses_lower_resolved_global_ceiling():
    from bibr.clients import providers

    settings = _settings(provider="ollama", max_tokens=700)
    resolved = _task_max_tokens(settings, 1200)
    provider = providers.get("ollama", settings=settings)

    assert provider.call_kwargs(None, max_tokens=resolved) == {
        "temperature": 0.0,
        "extra_body": {"options": {"num_predict": 700}},
    }


def test_ollama_provider_disabled_task_cap_uses_configured_global():
    from bibr.clients import providers

    settings = _settings(provider="ollama", max_tokens=700)
    resolved = _task_max_tokens(settings, 0)
    provider = providers.get("ollama", settings=settings)

    assert resolved is None
    assert provider.call_kwargs(None, max_tokens=resolved) == {
        "temperature": 0.0,
        "extra_body": {"options": {"num_predict": 700}},
    }


async def test_core_task_caps_reach_structured_backend():
    settings = _settings(
        max_tokens=65536,
        title_max_tokens=4001,
        authors_max_tokens=8001,
        paper_classification_max_tokens=1001,
        paper_type_max_tokens=501,
    )
    client = LLMClient(settings=settings)
    client._limiter = mock.MagicMock()
    client._limiter.acquire = mock.AsyncMock(return_value=None)
    seen = []

    async def fake_invoke(response_model, messages, system_prompt, **kwargs):
        seen.append(kwargs["max_tokens"])
        if response_model is TitleKeywordsLLM:
            return TitleKeywordsLLM(title="T", keywords=[])
        if response_model is AuthorsLLM:
            return AuthorsLLM(authors=[])
        if response_model is PaperClassificationLLM:
            return PaperClassificationLLM()
        if response_model is PaperTypeLabel:
            return PaperTypeLabel(paper_type="empirical", confidence=1.0)
        raise AssertionError(response_model)

    client._invoke_structured = fake_invoke
    await client.extract_title_keywords("x")
    await client.extract_authors("x")
    await client.extract_paper_classification("x")
    await client.label_paper_type("T", "A")

    assert seen == [4001, 8001, 1001, 501]


async def test_post_core_task_caps_reach_structured_backend():
    settings = _settings(
        integrity_max_tokens=4002,
        citation_max_tokens=8002,
        equation_max_tokens=4003,
    )
    client = LLMClient(settings=settings)
    client._limiter = mock.MagicMock()
    client._limiter.acquire = mock.AsyncMock(return_value=None)
    seen = []

    async def fake_invoke(response_model, messages, system_prompt, **kwargs):
        seen.append(kwargs["max_tokens"])
        name = response_model.__name__
        if name == "ResearchIntegrityLLM":
            return SimpleNamespace(funding=[], contributions=[], affiliations=[])
        if name == "CitationResolutionResult":
            return SimpleNamespace(matches=[])
        if name == "EquationExtractionResult":
            return SimpleNamespace(equations=[])
        raise AssertionError(name)

    client._invoke_structured = fake_invoke
    await client.extract_research_integrity("funding", "", [], affiliation_list=[])
    await client.resolve_citations([], [])
    await client.extract_equations([])

    assert seen == [4002, 8002, 4003]
