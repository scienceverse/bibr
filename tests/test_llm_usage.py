"""Tests for LLM token usage tracking in LLMClient."""

import logging
from types import SimpleNamespace
from unittest import mock

import pytest

from bibr.config import Settings


@pytest.fixture()
def _enable_tracking(monkeypatch):
    """Enable LLM usage tracking for the duration of a test."""
    monkeypatch.setattr(Settings.llm, "track_usage", True)


@pytest.fixture()
def _disable_tracking(monkeypatch):
    """Disable LLM usage tracking for the duration of a test."""
    monkeypatch.setattr(Settings.llm, "track_usage", False)


def _mock_limiter():
    """Create a mock rate limiter that can be set on client._limiter."""
    limiter = mock.AsyncMock()
    limiter.acquire = mock.AsyncMock()
    return limiter


def _make_fake_completion(input_tokens=100, output_tokens=50, total_tokens=150):
    """Create a fake completion object with usage metadata.

    Uses ``SimpleNamespace`` (not ``MagicMock``) so attribute probes behave
    like real provider response objects: absent fields raise AttributeError
    instead of auto-creating truthy mocks.
    """
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        # Some providers use prompt_tokens/completion_tokens instead
        prompt_tokens=0,
        completion_tokens=0,
    )
    return SimpleNamespace(usage=usage)


@pytest.mark.usefixtures("_enable_tracking")
class TestUsageTracking:
    """Tests for token usage tracking when enabled."""

    async def test_usage_aggregated_after_call(self):
        """Token usage should be recorded in client.usage after a call."""
        from bibr.clients.llm import LLMClient
        from bibr.schemas import TitleKeywordsLLM

        fake_response = TitleKeywordsLLM(title="Test Paper", keywords=["test"])
        fake_completion = _make_fake_completion()

        client = LLMClient()
        client._limiter = _mock_limiter()

        fake_instructor = mock.AsyncMock()
        fake_instructor.create_with_completion = mock.AsyncMock(
            return_value=(fake_response, fake_completion)
        )
        client._client = fake_instructor

        await client.extract_title_keywords("Some paper text")

        usage = client.usage
        model = Settings.llm.model
        assert model in usage
        assert usage[model]["input_tokens"] == 100
        assert usage[model]["output_tokens"] == 50
        assert usage[model]["total_tokens"] == 150

    async def test_usage_accumulates_across_calls(self):
        """Multiple calls should accumulate token counts."""
        from bibr.clients.llm import LLMClient
        from bibr.schemas import TitleKeywordsLLM

        fake_response = TitleKeywordsLLM(title="Test Paper", keywords=["test"])
        fake_completion = _make_fake_completion()

        client = LLMClient()
        client._limiter = _mock_limiter()

        fake_instructor = mock.AsyncMock()
        fake_instructor.create_with_completion = mock.AsyncMock(
            return_value=(fake_response, fake_completion)
        )
        client._client = fake_instructor

        await client.extract_title_keywords("Paper 1")
        await client.extract_title_keywords("Paper 2")

        usage = client.usage
        model = Settings.llm.model
        assert usage[model]["total_tokens"] == 300

    async def test_usage_property_returns_empty_when_no_calls(self):
        """Usage should be empty dict before any calls are made."""
        from bibr.clients.llm import LLMClient

        client = LLMClient()
        assert len(client.usage) == 0


@pytest.mark.usefixtures("_enable_tracking")
class TestProviderUsageShapes:
    """_record_usage must understand each provider's raw completion shape."""

    def test_genai_usage_metadata_recorded(self):
        """google-genai completions carry usage_metadata, not .usage.

        This is the default provider — a completion with only
        ``usage_metadata`` must still be recorded.
        """
        from bibr.clients.llm import LLMClient

        completion = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=200,
                candidates_token_count=40,
                total_token_count=240,
                cached_content_token_count=120,
            )
        )

        client = LLMClient()
        client._record_usage(completion)

        counts = client.usage[Settings.llm.model]
        assert counts["input_tokens"] == 200
        assert counts["output_tokens"] == 40
        assert counts["total_tokens"] == 240
        assert counts["cached_input_tokens"] == 120

    def test_openai_cached_tokens_recorded(self):
        """OpenAI reports prefix-cache hits in prompt_tokens_details.cached_tokens."""
        from bibr.clients.llm import LLMClient

        completion = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=300,
                completion_tokens=60,
                total_tokens=360,
                prompt_tokens_details=SimpleNamespace(cached_tokens=256),
            )
        )

        client = LLMClient()
        client._record_usage(completion)

        counts = client.usage[Settings.llm.model]
        assert counts["input_tokens"] == 300
        assert counts["cached_input_tokens"] == 256

    def test_anthropic_cache_read_tokens_recorded(self):
        """Anthropic reports prompt-cache hits as cache_read_input_tokens."""
        from bibr.clients.llm import LLMClient

        completion = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=150,
                output_tokens=30,
                total_tokens=180,
                cache_read_input_tokens=90,
            )
        )

        client = LLMClient()
        client._record_usage(completion)

        counts = client.usage[Settings.llm.model]
        assert counts["input_tokens"] == 150
        assert counts["cached_input_tokens"] == 90

    def test_no_cache_info_defaults_to_zero(self):
        """Completions without any cache fields record cached_input_tokens=0."""
        from bibr.clients.llm import LLMClient

        client = LLMClient()
        client._record_usage(_make_fake_completion())

        counts = client.usage[Settings.llm.model]
        assert counts["cached_input_tokens"] == 0

    def test_cached_tokens_accumulate(self):
        """Cached-token counts accumulate across calls like the other counters."""
        from bibr.clients.llm import LLMClient

        completion = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=10,
                total_token_count=110,
                cached_content_token_count=64,
            )
        )

        client = LLMClient()
        client._record_usage(completion)
        client._record_usage(completion)

        counts = client.usage[Settings.llm.model]
        assert counts["cached_input_tokens"] == 128


@pytest.mark.usefixtures("_disable_tracking")
class TestUsageTrackingDisabled:
    """Tests for when token tracking is disabled."""

    async def test_no_tracking_when_disabled(self):
        """Tracking flag should be False when tracking is disabled."""
        from bibr.clients.llm import LLMClient

        client = LLMClient()
        assert client._track_usage is False

    async def test_usage_returns_empty_dict(self):
        """Usage property should return empty dict when disabled."""
        from bibr.clients.llm import LLMClient

        client = LLMClient()
        assert client.usage == {}

    async def test_invoke_works_without_tracking(self):
        """Extraction should work normally even with tracking disabled."""
        from bibr.clients.llm import LLMClient
        from bibr.schemas import TitleKeywordsLLM

        fake_response = TitleKeywordsLLM(title="Test Paper", keywords=["test"])

        client = LLMClient()
        client._limiter = _mock_limiter()

        fake_instructor = mock.AsyncMock()
        fake_instructor.create = mock.AsyncMock(return_value=fake_response)
        client._client = fake_instructor

        result = await client.extract_title_keywords("Some paper text")

        assert result.title == "Test Paper"
        # When tracking is disabled, create() is called (not create_with_completion)
        fake_instructor.create.assert_called()
        fake_instructor.create_with_completion.assert_not_called()


@pytest.mark.usefixtures("_enable_tracking")
class TestInvokeStructuredLabeled:
    """Tests for the public, per-call-site-labeled invoke_structured wrapper."""

    async def test_logs_usage_with_call_site_label(self, caplog):
        """invoke_structured(..., label=...) logs a Token usage line tagged
        with the caller-supplied label, not the wrapper's own function name.
        """
        from bibr.clients.llm import LLMClient
        from bibr.schemas import TitleKeywordsLLM

        fake_response = TitleKeywordsLLM(title="Test Paper", keywords=["test"])
        fake_completion = _make_fake_completion()

        client = LLMClient()
        client._limiter = _mock_limiter()

        fake_instructor = mock.AsyncMock()
        fake_instructor.create_with_completion = mock.AsyncMock(
            return_value=(fake_response, fake_completion)
        )
        client._client = fake_instructor

        with caplog.at_level(logging.INFO, logger="bibr.clients.llm"):
            result = await client.invoke_structured(
                TitleKeywordsLLM,
                [{"role": "user", "content": "classify these sections"}],
                "You are a scientific paper section classifier.",
                label="section_classifier",
            )

        assert result.title == "Test Paper"
        assert any("Token usage [section_classifier]" in r.getMessage() for r in caplog.records)

        usage = client.usage
        model = Settings.llm.model
        assert usage[model]["total_tokens"] == 150

    async def test_no_usage_log_when_tracking_disabled(self, caplog, monkeypatch):
        """With tracking disabled, invoke_structured should skip the usage
        snapshot/delta machinery entirely (matching @track_llm_usage's own
        early-return behavior) and still return the underlying result.
        """
        from bibr.clients.llm import LLMClient
        from bibr.schemas import TitleKeywordsLLM

        monkeypatch.setattr(Settings.llm, "track_usage", False)

        fake_response = TitleKeywordsLLM(title="Test Paper", keywords=["test"])

        client = LLMClient()
        client._limiter = _mock_limiter()

        fake_instructor = mock.AsyncMock()
        fake_instructor.create = mock.AsyncMock(return_value=fake_response)
        client._client = fake_instructor

        with caplog.at_level(logging.INFO, logger="bibr.clients.llm"):
            result = await client.invoke_structured(
                TitleKeywordsLLM,
                [{"role": "user", "content": "classify these sections"}],
                "You are a scientific paper section classifier.",
                label="section_classifier",
            )

        assert result.title == "Test Paper"
        assert not any("Token usage" in r.getMessage() for r in caplog.records)
        fake_instructor.create.assert_called()
        fake_instructor.create_with_completion.assert_not_called()
