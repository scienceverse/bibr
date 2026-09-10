import re
from unittest import mock

import pytest


def test_llmclient_satisfies_protocol():
    from bibr.clients.llm import LLMClient
    from bibr.clients.llm_protocol import LlmClient

    assert isinstance(LLMClient(), LlmClient)


def test_get_client_invalidates_when_settings_change(monkeypatch):
    from bibr.clients.llm import LLMClient
    from bibr.config import GlobalSettings

    fake1 = mock.MagicMock(name="fake1")
    fake2 = mock.MagicMock(name="fake2")
    builds = iter([fake1, fake2])
    seen_settings = []

    def create_client(*, settings=None, **_kwargs):
        seen_settings.append(settings)
        return next(builds)

    monkeypatch.setattr("bibr.clients.llm._create_client", create_client)

    settings = GlobalSettings()
    settings.llm.provider = "google"
    settings.llm.base_url = None
    settings.llm.api_key = None
    c = LLMClient(settings=settings)
    assert c._get_client() is fake1
    # Same settings → cached
    assert c._get_client() is fake1

    # Settings mutated (simulating VllmMlxLlmServer.configure_llm_client)
    settings.llm.provider = "openai"
    settings.llm.base_url = "http://localhost:8001/v1"
    settings.llm.api_key = "local-stub"

    # Now must rebuild
    assert c._get_client() is fake2
    assert seen_settings == [settings, settings]


async def test_classification_uses_compact_input(monkeypatch):
    from bibr.clients.llm import LLMClient

    sizes = []

    async def fake_invoke(self, response_model, messages, system_prompt, **kw):
        from bibr.clients.prompts import prompt_text

        sizes.append(len(prompt_text(messages[0]["content"])))

        class _R:
            paper_type = "empirical"
            confidence = 0.9

        return _R()

    monkeypatch.setattr(LLMClient, "_invoke_structured", fake_invoke)
    c = LLMClient()
    big = "x" * 200_000
    await c.extract_paper_classification(big)
    assert sizes[0] < 30_000, sizes


@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("extract_title_keywords", {"text": "x" * 1_000_000}),
        ("extract_authors", {"text": "x" * 1_000_000}),
        ("extract_references", {"text": "x" * 1_000_000}),
        ("extract_paper_classification", {"text": "x" * 1_000_000}),
    ],
)
async def test_no_method_truncates_instructions(monkeypatch, method, kwargs):
    """Smoke test: feeding 1 MB of input must not truncate instruction text mid-sentence.

    Each method should either fit within its cap+overhead OR end on a recognizable
    boundary (UUID-marker close, quoted JSON close, or trailing newline).
    """
    from bibr.clients.llm import LLMClient

    captured = {"messages": None}

    async def fake_invoke(self, response_model, messages, system_prompt, **kw):
        captured["messages"] = messages

        class _R:
            pass

        return _R()

    monkeypatch.setattr(LLMClient, "_invoke_structured", fake_invoke)
    c = LLMClient()
    try:
        await getattr(c, method)(**kwargs)
    except Exception as exc:  # noqa: BLE001
        # Some methods raise on empty/invalid result objects from the fake.
        # We only care about the assembled prompt, captured before the raise.
        _ = exc  # expected; method may fail to parse _R() into its schema
    assert captured["messages"] is not None, f"{method} never called _invoke_structured"
    from bibr.clients.prompts import prompt_text

    content = prompt_text(captured["messages"][0]["content"])
    # Heuristic: the prompt should end on a clearly-recognizable boundary, not
    # mid-instruction. Doc-first prompts end on the intact instruction (the
    # injection-guard sentence); instruction-first prompts end on the fence
    # close (`---`) or, for reference parsing, the start-index trailer.
    stripped = content.rstrip()
    assert (
        stripped.endswith("---")
        or stripped.endswith("not as instructions.")
        or re.search(r"index \d+\.$", stripped)
    ), f"{method} prompt appears truncated mid-instruction; tail: ...{content[-200:]!r}"
