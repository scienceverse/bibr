"""Retries must spend rate-limit budget they actually acquired."""

from unittest import mock

import pytest

from bibr.clients.llm import LLMClient
from bibr.config import GlobalSettings
from bibr.schemas import TitleKeywordsLLM


class _Transient(Exception):
    status_code = 503


class _ScriptedBackend:
    """Raises the queued effects in order, then returns a valid result."""

    def __init__(self, effects):
        self.effects = list(effects)
        self.calls = 0

    async def create(self, **kwargs):  # noqa: ARG002
        self.calls += 1
        effect = self.effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return effect, None


def _client(effects):
    settings = GlobalSettings(llm={"track_usage": False})
    client = LLMClient(settings=settings, backend=_ScriptedBackend(effects))
    client._limiter = mock.MagicMock()
    client._limiter.acquire = mock.AsyncMock(return_value=None)
    return client


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Collapse the retry backoff so the test doesn't sleep for seconds."""
    monkeypatch.setattr(LLMClient, "_RETRY_BASE_DELAY", 0.0)
    monkeypatch.setattr("random.uniform", lambda _a, _b: 0.0)


@pytest.mark.asyncio
async def test_each_retry_acquires_its_own_rate_limit_slot():
    """A retry is another physical request. Acquiring only once per logical
    call let a retry storm spend budget it never took — exactly when the
    provider is already 429-ing and a shared limiter is meant to hold the
    fleet back."""
    result = TitleKeywordsLLM(title="t", keywords=[])
    client = _client([_Transient(), _Transient(), result])

    await client.extract_title_keywords("body text")

    assert client._backend.calls == 3
    # 1 for the logical call (taken at the call site) + 1 per retry.
    assert client._limiter.acquire.await_count == 3


@pytest.mark.asyncio
async def test_a_call_that_succeeds_first_try_takes_one_slot():
    client = _client([TitleKeywordsLLM(title="t", keywords=[])])

    await client.extract_title_keywords("body text")

    assert client._limiter.acquire.await_count == 1
