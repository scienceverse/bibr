"""Tests for LLM client retry logic."""

from unittest import mock
from unittest.mock import AsyncMock

import pytest

from bibr.clients.llm import LLMClient
from bibr.utils.circuit_breaker import AsyncCircuitBreaker, CircuitOpenError, CircuitState


@pytest.fixture()
def _no_circuit_breaker(monkeypatch):
    """Replace the circuit breaker with a pass-through context manager."""

    async def noop_aenter(self):
        return self

    async def noop_aexit(self, *args):
        return False

    monkeypatch.setattr(AsyncCircuitBreaker, "__aenter__", noop_aenter)
    monkeypatch.setattr(AsyncCircuitBreaker, "__aexit__", noop_aexit)


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch):
    """Eliminate retry delays in tests."""
    monkeypatch.setattr(LLMClient, "_RETRY_BASE_DELAY", 0.0)


def _make_client_with_mock(side_effect=None, return_value=None):
    """Create an LLMClient with a mocked instructor client."""
    from pydantic import BaseModel

    class FakeModel(BaseModel):
        result: str = "ok"

    client = LLMClient()
    client._track_usage = False

    fake_instructor = AsyncMock()
    if side_effect is not None:
        fake_instructor.create = AsyncMock(side_effect=side_effect)
    elif return_value is not None:
        fake_instructor.create = AsyncMock(return_value=return_value)
    else:
        fake_instructor.create = AsyncMock(return_value=FakeModel())

    client._client = fake_instructor
    return client, fake_instructor, FakeModel


@pytest.mark.usefixtures("_no_circuit_breaker")
class TestInvokeStructuredRetry:
    async def test_succeeds_on_first_try(self):
        client, fake_instructor, FakeModel = _make_client_with_mock()

        result = await client._invoke_structured(
            FakeModel, [{"role": "user", "content": "hi"}], "system"
        )
        assert result.result == "ok"
        assert fake_instructor.create.call_count == 1

    async def test_retries_on_timeout_then_succeeds(self):
        from pydantic import BaseModel

        class FakeModel(BaseModel):
            result: str = "ok"

        client, fake_instructor, _ = _make_client_with_mock(
            side_effect=[TimeoutError(), FakeModel()]
        )

        result = await client._invoke_structured(
            FakeModel, [{"role": "user", "content": "hi"}], "system"
        )
        assert result.result == "ok"
        assert fake_instructor.create.call_count == 2

    async def test_retries_on_upstream_error_then_succeeds(self):
        from pydantic import BaseModel

        from bibr.exceptions import UpstreamServiceError

        class FakeModel(BaseModel):
            result: str = "ok"

        client, fake_instructor, _ = _make_client_with_mock(
            side_effect=[UpstreamServiceError("LLM", "503", None), FakeModel()]
        )

        result = await client._invoke_structured(
            FakeModel, [{"role": "user", "content": "hi"}], "system"
        )
        assert result.result == "ok"
        assert fake_instructor.create.call_count == 2

    async def test_raises_after_max_retries(self):
        client, fake_instructor, FakeModel = _make_client_with_mock(side_effect=TimeoutError())

        with pytest.raises(TimeoutError):
            await client._invoke_structured(
                FakeModel, [{"role": "user", "content": "hi"}], "system"
            )
        assert fake_instructor.create.call_count == 3

    async def test_no_retry_on_value_error(self):
        """Non-transient errors should not be retried."""
        client, fake_instructor, FakeModel = _make_client_with_mock(
            side_effect=ValueError("bad input")
        )

        with pytest.raises(ValueError, match="bad input"):
            await client._invoke_structured(
                FakeModel, [{"role": "user", "content": "hi"}], "system"
            )
        assert fake_instructor.create.call_count == 1


class TestRetryCircuitBreakerInteraction:
    """Test that retry failures propagate through the real circuit breaker."""

    async def test_final_timeout_trips_breaker(self):
        cb = AsyncCircuitBreaker(
            failure_threshold=1, reset_timeout=60.0, name="test", failure_dedup_window=0.0
        )
        client, fake_instructor, FakeModel = _make_client_with_mock(side_effect=TimeoutError())
        client._breaker = cb

        # With failure_threshold=1 and the fix in place, the breaker opens after
        # the first timeout, so the second attempt raises CircuitOpenError (not TimeoutError).
        with pytest.raises((TimeoutError, CircuitOpenError)):
            await client._invoke_structured(
                FakeModel, [{"role": "user", "content": "hi"}], "system"
            )

        assert cb.state == CircuitState.OPEN


@pytest.mark.usefixtures("_no_circuit_breaker")
async def test_invoke_enables_validation_retries(monkeypatch):
    """Schema/validation failures must be re-asked to the model (Instructor's
    self-correction loop), so ``max_retries`` is a validation-aware
    ``AsyncRetrying`` rather than 0. Provider 429/5xx retries stay in
    ``_invoke_structured``'s own loop and must not be double-counted here."""
    from tenacity import AsyncRetrying

    from bibr.clients.llm import LLMClient

    captured = {}

    async def fake_create(**kw):
        captured.update(kw)

        class _R:
            pass

        return _R()

    fake = mock.MagicMock()
    fake.create = fake_create
    fake.create_with_completion = fake_create
    client = LLMClient()
    monkeypatch.setattr(client, "_get_client", lambda: fake)

    await client._invoke_structured(mock.MagicMock(), [{"role": "user", "content": "x"}], "sys")
    assert isinstance(captured["max_retries"], AsyncRetrying)


def test_validation_attempts_auto_selects_local_vs_cloud():
    from bibr.clients.llm import _validation_attempts
    from bibr.config import GlobalSettings

    cloud = GlobalSettings(
        llm={"provider": "google", "base_url": None, "validation_attempts": None}
    )
    local = GlobalSettings(
        llm={
            "provider": "openai",
            "base_url": "http://127.0.0.1:8767/v1",
            "validation_attempts": None,
        }
    )
    explicit = GlobalSettings(
        llm={
            "provider": "openai",
            "base_url": "http://127.0.0.1:8767/v1",
            "validation_attempts": 2,
        }
    )

    assert _validation_attempts(cloud) == 3
    assert _validation_attempts(local) == 1
    assert _validation_attempts(explicit) == 2


@pytest.mark.usefixtures("_no_circuit_breaker")
async def test_retries_on_google_resource_exhausted(monkeypatch):
    """Google's ResourceExhausted (429) lacks ``response.status_code`` — the
    retry path must detect it via the exception type name."""
    from pydantic import BaseModel

    class FakeModel(BaseModel):
        result: str = "ok"

    class ResourceExhausted(Exception):
        """Stand-in for google.api_core.exceptions.ResourceExhausted."""

    client, fake_instructor, _ = _make_client_with_mock(
        side_effect=[ResourceExhausted("rate exceeded"), FakeModel()]
    )
    result = await client._invoke_structured(
        FakeModel, [{"role": "user", "content": "hi"}], "system"
    )
    assert result.result == "ok"
    assert fake_instructor.create.call_count == 2


@pytest.mark.usefixtures("_no_circuit_breaker")
async def test_retries_on_google_503_server_error():
    """Google's genai ``ServerError`` (503 UNAVAILABLE) on the async/aiohttp
    path carries its status in ``.code`` (int) and ``.response.status`` (the
    aiohttp ``ClientResponse``), never ``.status_code``. The 5xx retry path
    must classify it as transient and retry instead of failing on attempt 1."""
    from pydantic import BaseModel

    class FakeModel(BaseModel):
        result: str = "ok"

    class _AiohttpResp:
        """aiohttp.ClientResponse shape: code is ``.status``, no ``.status_code``."""

        status = 503
        headers: dict = {}  # noqa: RUF012

    class ServerError(Exception):
        """Stand-in for google.genai.errors.ServerError on the aiohttp path."""

        def __init__(self):
            super().__init__("503 UNAVAILABLE")
            self.code = 503  # genai APIError.code holds the int status
            self.status = "UNAVAILABLE"  # APIError.status is the string label, not the int
            self.response = _AiohttpResp()

    client, fake_instructor, _ = _make_client_with_mock(side_effect=[ServerError(), FakeModel()])
    result = await client._invoke_structured(
        FakeModel, [{"role": "user", "content": "hi"}], "system"
    )
    assert result.result == "ok"
    assert fake_instructor.create.call_count == 2


@pytest.mark.usefixtures("_no_circuit_breaker")
async def test_retry_honors_retry_after_header(monkeypatch):
    """When the provider includes ``Retry-After``, sleep at least that long."""
    import httpx
    from pydantic import BaseModel

    class FakeModel(BaseModel):
        result: str = "ok"

    request = httpx.Request("POST", "https://api.example/v1")
    response = httpx.Response(429, headers={"retry-after": "7"}, request=request)
    err = httpx.HTTPStatusError("rate limit", request=request, response=response)

    client, fake_instructor, _ = _make_client_with_mock(side_effect=[err, FakeModel()])

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("bibr.clients.llm.asyncio.sleep", fake_sleep)
    await client._invoke_structured(FakeModel, [{"role": "user", "content": "hi"}], "system")
    assert sleeps and sleeps[0] >= 7.0, sleeps


async def test_timeouts_count_against_breaker(monkeypatch):
    """Three sequential timeouts must each trip the circuit breaker failure counter."""
    client = LLMClient()
    # Use failure_dedup_window=0.0 so every failure increments the counter independently.
    client._breaker = AsyncCircuitBreaker(
        failure_threshold=99,
        reset_timeout=60.0,
        name="test-timeout",
        failure_dedup_window=0.0,
    )
    fake = mock.MagicMock()
    fake.create = mock.AsyncMock(side_effect=TimeoutError())
    fake.create_with_completion = mock.AsyncMock(side_effect=TimeoutError())
    monkeypatch.setattr(client, "_get_client", lambda: fake)

    # Set per_request very low so the wait_for fires fast.
    from bibr.config import Settings

    monkeypatch.setattr(Settings.llm, "timeout_seconds", 0.01)

    # Drive 3 attempts (one full _invoke_structured call) and expect breaker
    # failure counter == 3 after (one per attempt, not just for the final raise).
    with pytest.raises(TimeoutError):
        await client._invoke_structured(mock.MagicMock(), [{"role": "user", "content": "x"}], "sys")
    assert client._breaker._failure_count >= 3, (
        f"Breaker must record all {LLMClient._RETRY_MAX_ATTEMPTS} timeout failures, "
        f"got {client._breaker._failure_count}"
    )
