"""Unit tests for CloudOcrClient (vision-LLM OCR backends).

All tests are fully mocked — no network calls, no API keys required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.local.ocr_cloud import (
    _PROMPT_MAP,
    _VISION_PROMPTS,
    CloudOcrClient,
    OcrResult,
    _make_provider_backend,
)

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_cloud_ocr_satisfies_protocol():
    """CloudOcrClient instances pass the OcrBackend runtime_checkable check."""
    from bibr.ocr.backend import OcrBackend

    client = CloudOcrClient.__new__(CloudOcrClient)
    # Minimal attribute setup so runtime_checkable can inspect
    client._loaded = True
    assert isinstance(client, OcrBackend)


def test_cloud_ocr_has_required_attributes():
    """CloudOcrClient declares all OcrBackend attributes and methods."""
    import inspect

    assert hasattr(CloudOcrClient, "name")
    assert hasattr(CloudOcrClient, "loaded")
    for method_name in ("recognize", "wait_for_server", "shutdown"):
        method = getattr(CloudOcrClient, method_name)
        assert inspect.iscoroutinefunction(method), f"{method_name} must be async"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_backends_registered():
    """The three cloud providers are registered in the OCR registry."""
    from bibr.ocr import registry

    known = registry.known_backends()
    for name in ("gemini", "openai", "anthropic"):
        assert name in known, f"{name} not in registry"


def test_make_provider_backend_creates_subclass():
    """_make_provider_backend produces a subclass with the correct name."""
    cls = _make_provider_backend("test-provider")
    assert issubclass(cls, CloudOcrClient)
    assert cls.name == "test-provider"
    assert cls.__name__ == "CloudOcrClient_test-provider"


def test_provider_subclass_init_passes_provider():
    """Provider subclasses pass the correct provider string to __init__."""
    cls = _make_provider_backend("test-init")
    with patch.object(CloudOcrClient, "__init__", return_value=None) as mock_init:
        cls()
        mock_init.assert_called_once_with(provider="test-init")


# ---------------------------------------------------------------------------
# Prompt mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "glm_prompt,expected_key",
    [
        ("Text Recognition:", "text"),
        ("Table Recognition:", "table"),
        ("Formula Recognition:", "formula"),
    ],
)
def test_prompt_mapping(glm_prompt, expected_key):
    """GLM-OCR task prompts map to the correct vision prompt key."""
    assert _PROMPT_MAP[glm_prompt] == expected_key


def test_prompt_mapping_unknown_falls_back_to_text():
    """Unknown GLM prompts fall back to the 'text' vision prompt."""
    client = CloudOcrClient.__new__(CloudOcrClient)
    resolved = client._resolve_prompt("Unknown Prompt:")
    assert resolved == _VISION_PROMPTS["text"]


def test_all_vision_prompt_keys_covered():
    """Every mapped key exists in _VISION_PROMPTS."""
    for key in _PROMPT_MAP.values():
        assert key in _VISION_PROMPTS


# ---------------------------------------------------------------------------
# recognize() — mock instructor client
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cloud_client():
    """Return a CloudOcrClient with mocked internals."""
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.ocr_vision.rate_limit_rpm = 60
    settings.ocr_vision.timeout_seconds = 30
    settings.ocr_vision.max_tokens = 4096
    settings.ocr_vision.model = "test-model"
    settings.ocr_vision.base_url = None
    settings.cb.failure_threshold = 5
    settings.cb.reset_timeout_seconds = 30.0
    settings.llm.api_key = "test-key"
    settings.GOOGLE_API_KEY = "test-google-key"

    client = CloudOcrClient(provider="google", settings=settings)
    yield client


@pytest.fixture
def _mock_image_utils():
    """Patch PIL-to-base64 and Instructor Image to avoid real image processing."""
    fake_image = MagicMock(name="InstructorImage")
    with (
        patch(
            "bibr.ocr.image_utils.pil_to_base64_glmocr",
            return_value="AAAA",
        ),
        patch(
            "instructor.processing.multimodal.Image.from_raw_base64",
            return_value=fake_image,
        ),
    ):
        yield


async def test_recognize_returns_text(mock_cloud_client, _mock_image_utils):
    """recognize() returns the text from the OcrResult."""
    mock_instructor = AsyncMock()
    mock_instructor.create = AsyncMock(return_value=OcrResult(text="Hello world"))
    mock_cloud_client._client = mock_instructor

    result = await mock_cloud_client.recognize(MagicMock(), "Text Recognition:")
    assert result == "Hello world"


async def test_recognize_uses_vision_prompt(mock_cloud_client, _mock_image_utils):
    """recognize() sends the expanded vision prompt, not the raw GLM prompt."""
    mock_instructor = AsyncMock()
    mock_instructor.create = AsyncMock(return_value=OcrResult(text="ok"))
    mock_cloud_client._client = mock_instructor

    await mock_cloud_client.recognize(MagicMock(), "Table Recognition:")

    call_kwargs = mock_instructor.create.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
    user_content = messages[0]["content"]
    # The first element should be the verbose vision prompt
    assert user_content[0] == _VISION_PROMPTS["table"]


async def test_recognize_calls_rate_limiter(mock_cloud_client, _mock_image_utils):
    """recognize() calls the rate limiter before making the API call."""
    mock_instructor = AsyncMock()
    mock_instructor.create = AsyncMock(return_value=OcrResult(text="ok"))
    mock_cloud_client._client = mock_instructor
    mock_cloud_client._rate_limiter = AsyncMock()
    mock_cloud_client._rate_limiter.acquire = AsyncMock()
    mock_cloud_client._rate_limiter.close = AsyncMock()

    await mock_cloud_client.recognize(MagicMock(), "Text Recognition:")

    mock_cloud_client._rate_limiter.acquire.assert_awaited_once()


async def test_recognize_retries_reacquire_rate_limiter(mock_cloud_client, _mock_image_utils):
    """Each retry attempt must re-acquire a rate-limit slot — otherwise
    retries bypass the limiter exactly when the API is telling us to slow down."""
    exc = Exception("rate limited")
    exc.status_code = 429

    mock_instructor = AsyncMock()
    mock_instructor.create = AsyncMock(side_effect=[exc, exc, OcrResult(text="ok")])
    mock_cloud_client._client = mock_instructor
    mock_cloud_client._rate_limiter = AsyncMock()

    with patch("bibr.local.ocr_cloud.asyncio.sleep", new_callable=AsyncMock):
        result = await mock_cloud_client.recognize(MagicMock(), "Text Recognition:")

    assert result == "ok"
    assert mock_cloud_client._rate_limiter.acquire.await_count == 3


async def test_aget_client_uses_instance_settings(monkeypatch):
    """Instructor construction reads the settings object passed to the OCR client."""
    import instructor

    from bibr.config import GlobalSettings, Settings

    custom = GlobalSettings()
    custom.ocr_vision.model = "custom-vision-model"
    custom.ocr_vision.base_url = "https://custom-vision.example/v1"
    custom.ocr_vision.rate_limit_rpm = 120
    custom.llm.api_key = "custom-google-key"
    custom.GOOGLE_API_KEY = "custom-google-fallback"

    monkeypatch.setattr(Settings.ocr_vision, "model", "global-vision-model")
    monkeypatch.setattr(Settings.ocr_vision, "base_url", None)
    monkeypatch.setattr(Settings.llm, "api_key", "global-google-key")
    monkeypatch.setattr(Settings, "GOOGLE_API_KEY", "global-google-fallback")

    captured: dict[str, object] = {}

    def fake_from_provider(model_string: str, **kwargs: object):
        captured["model_string"] = model_string
        captured["kwargs"] = kwargs
        return "fake-client"

    monkeypatch.setattr(instructor, "from_provider", fake_from_provider)

    client = CloudOcrClient(provider="google", settings=custom)
    result = await client._aget_client()

    assert result == "fake-client"
    assert captured["model_string"] == "google/custom-vision-model"
    assert captured["kwargs"] == {
        "async_client": True,
        "api_key": "custom-google-key",
        "base_url": "https://custom-vision.example/v1",
    }


# ---------------------------------------------------------------------------
# local-runtimes sweep: vision key resolution without cross-contamination (2)
# ---------------------------------------------------------------------------


def _vision_settings(**overrides):
    """GlobalSettings with vision/LLM key fields set directly."""
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    for dotted, value in overrides.items():
        if "." in dotted:
            section, field = dotted.split(".", 1)
            setattr(getattr(settings, section), field, value)
        else:
            setattr(settings, dotted, value)
    return settings


def test_api_key_ignores_cross_provider_llm_key():
    """LLM on OpenAI + OCR on Gemini must use GOOGLE_API_KEY, not the OpenAI key (2)."""
    from bibr.local.ocr_cloud import _api_key_for_provider

    settings = _vision_settings(
        **{
            "llm.provider": "openai",
            "llm.api_key": "sk-test-openai-key-placeholder",
            "GOOGLE_API_KEY": "sk-test-google-key-placeholder",
        }
    )
    assert _api_key_for_provider("google", settings) == "sk-test-google-key-placeholder"
    assert _api_key_for_provider("gemini", settings) == "sk-test-google-key-placeholder"


def test_api_key_ignores_managed_local_placeholders():
    """Managed-local placeholder LLM keys are never forwarded as vision keys (2)."""
    from bibr.local.ocr_cloud import _api_key_for_provider

    for placeholder in ("not-needed", "lm-studio"):
        settings = _vision_settings(
            **{
                "llm.provider": "openai",
                "llm.base_url": "http://127.0.0.1:8770/v1",
                "llm.api_key": placeholder,
                "GOOGLE_API_KEY": "sk-test-google-key-placeholder",
            }
        )
        assert _api_key_for_provider("google", settings) == "sk-test-google-key-placeholder"
        # OpenAI vision with no matching LLM key: leave unset for SDK env fallback.
        assert _api_key_for_provider("openai", settings) is None


def test_api_key_uses_llm_key_for_same_provider():
    """LLM and vision on the same cloud provider share the LLM key (2 guard)."""
    from bibr.local.ocr_cloud import _api_key_for_provider

    settings = _vision_settings(
        **{
            "llm.provider": "google",
            "llm.api_key": "sk-test-shared-key-placeholder",
            "GOOGLE_API_KEY": "sk-test-google-key-placeholder",
        }
    )
    assert _api_key_for_provider("google", settings) == "sk-test-shared-key-placeholder"


def test_api_key_ignores_llm_key_pointing_at_another_endpoint():
    """Same provider string but `llm.base_url` set: the key may belong to a proxy (2).

    With LLM_PROVIDER=openai and LLM_BASE_URL pointing at OpenRouter/a fleet
    proxy, the LLM key must not be sent to api.openai.com for `--ocr openai`.
    """
    from bibr.local.ocr_cloud import _api_key_for_provider

    settings = _vision_settings(
        **{
            "llm.provider": "openai",
            "llm.base_url": "https://openrouter.ai/api/v1",
            "llm.api_key": "sk-test-third-party-key-placeholder",
        }
    )
    # OpenAI vision with no matching LLM key: leave unset for SDK env fallback.
    assert _api_key_for_provider("openai", settings) is None

    google_settings = _vision_settings(
        **{
            "llm.provider": "google",
            "llm.base_url": "https://fleet-proxy.example/v1",
            "llm.api_key": "sk-test-proxy-key-placeholder",
            "GOOGLE_API_KEY": "sk-test-google-key-placeholder",
        }
    )
    assert _api_key_for_provider("google", google_settings) == "sk-test-google-key-placeholder"


async def test_aget_client_drops_placeholder_llm_key(monkeypatch):
    """End to end: a managed-local snapshot never sends its placeholder to Google (2)."""
    import instructor

    from bibr.config import GlobalSettings

    custom = GlobalSettings()
    custom.llm.provider = "openai"
    custom.llm.base_url = "http://127.0.0.1:8770/v1"
    custom.llm.api_key = "not-needed"
    custom.GOOGLE_API_KEY = "sk-test-google-key-placeholder"

    captured: dict[str, object] = {}

    def fake_from_provider(model_string: str, **kwargs: object):
        captured["kwargs"] = kwargs
        return "fake-client"

    monkeypatch.setattr(instructor, "from_provider", fake_from_provider)

    client = CloudOcrClient(provider="google", settings=custom)
    assert await client._aget_client() == "fake-client"
    assert captured["kwargs"]["api_key"] == "sk-test-google-key-placeholder"


# ---------------------------------------------------------------------------
# Error handling — graceful degradation
# ---------------------------------------------------------------------------


async def test_recognize_returns_empty_on_failure(mock_cloud_client, _mock_image_utils):
    """recognize() returns '' on unrecoverable failure instead of raising."""
    mock_instructor = AsyncMock()
    mock_instructor.create = AsyncMock(side_effect=ValueError("bad request"))
    mock_cloud_client._client = mock_instructor

    result = await mock_cloud_client.recognize(MagicMock(), "Text Recognition:")
    assert result == ""


async def test_recognize_retries_on_transient_error(mock_cloud_client, _mock_image_utils):
    """recognize() retries on transient errors (e.g. 429, 5xx)."""
    # Create an exception with a status_code attribute
    exc = Exception("rate limited")
    exc.status_code = 429

    mock_instructor = AsyncMock()
    mock_instructor.create = AsyncMock(side_effect=[exc, exc, OcrResult(text="recovered")])
    mock_cloud_client._client = mock_instructor
    mock_cloud_client._rate_limiter = AsyncMock()

    with patch("bibr.local.ocr_cloud.asyncio.sleep", new_callable=AsyncMock):
        result = await mock_cloud_client.recognize(MagicMock(), "Text Recognition:")

    assert result == "recovered"
    assert mock_instructor.create.await_count == 3


async def test_recognize_returns_empty_after_max_retries(mock_cloud_client, _mock_image_utils):
    """recognize() returns '' after exhausting retries on transient errors."""
    exc = Exception("server error")
    exc.status_code = 500

    mock_instructor = AsyncMock()
    mock_instructor.create = AsyncMock(side_effect=exc)
    mock_cloud_client._client = mock_instructor
    mock_cloud_client._rate_limiter = AsyncMock()

    with patch("bibr.local.ocr_cloud.asyncio.sleep", new_callable=AsyncMock):
        result = await mock_cloud_client.recognize(MagicMock(), "Text Recognition:")

    assert result == ""
    assert mock_instructor.create.await_count == 3  # _MAX_RETRIES


def _rate_limited() -> Exception:
    exc = Exception("429 RESOURCE_EXHAUSTED")
    exc.status_code = 429
    return exc


async def test_recognize_fails_fast_once_quota_is_used_up(mock_cloud_client, _mock_image_utils):
    """A key out of quota answers 429 to every retry; after a run of such regions
    the file fails with a clear error instead of grinding through every region."""
    from bibr.exceptions import UpstreamServiceError

    mock_instructor = AsyncMock()
    mock_instructor.create = AsyncMock(side_effect=_rate_limited())
    mock_cloud_client._client = mock_instructor
    mock_cloud_client._rate_limiter = AsyncMock()
    trip = CloudOcrClient._RATE_LIMIT_TRIP_REGIONS

    with patch("bibr.local.ocr_cloud.asyncio.sleep", new_callable=AsyncMock):
        for _ in range(trip - 1):
            assert await mock_cloud_client.recognize(MagicMock(), "Text Recognition:") == ""
        with pytest.raises(UpstreamServiceError, match="quota"):
            await mock_cloud_client.recognize(MagicMock(), "Text Recognition:")
        calls = mock_instructor.create.await_count
        assert calls == trip * CloudOcrClient._MAX_RETRIES

        # Later regions fail at once, without another API call.
        with pytest.raises(UpstreamServiceError, match="quota"):
            await mock_cloud_client.recognize(MagicMock(), "Text Recognition:")
        assert mock_instructor.create.await_count == calls


async def test_recognize_rate_limit_run_resets_on_success(mock_cloud_client, _mock_image_utils):
    """Occasional rate-limited regions between successes are a busy API, not a used-up quota."""
    mock_instructor = AsyncMock()
    mock_cloud_client._client = mock_instructor
    mock_cloud_client._rate_limiter = AsyncMock()
    exhausted = [_rate_limited()] * CloudOcrClient._MAX_RETRIES
    trip = CloudOcrClient._RATE_LIMIT_TRIP_REGIONS
    side_effect = []
    for _ in range(3):
        side_effect += exhausted * (trip - 1) + [OcrResult(text="ok")]
    mock_instructor.create = AsyncMock(side_effect=side_effect)

    with patch("bibr.local.ocr_cloud.asyncio.sleep", new_callable=AsyncMock):
        results = [
            await mock_cloud_client.recognize(MagicMock(), "Text Recognition:")
            for _ in range(3 * trip)
        ]
    assert results.count("ok") == 3
    assert results.count("") == 3 * (trip - 1)


@pytest.mark.parametrize("status", [401, 403, 404])
async def test_recognize_raises_on_fatal_auth_error(mock_cloud_client, _mock_image_utils, status):
    """recognize() raises UpstreamServiceError on auth/config status codes
    instead of silently returning blank content (which would corrupt the
    whole document for a misconfigured backend)."""
    from bibr.exceptions import UpstreamServiceError

    exc = Exception(f"status {status}")
    exc.status_code = status

    mock_instructor = AsyncMock()
    mock_instructor.create = AsyncMock(side_effect=exc)
    mock_cloud_client._client = mock_instructor

    with pytest.raises(UpstreamServiceError):
        await mock_cloud_client.recognize(MagicMock(), "Text Recognition:")
    # Fail fast — no retries on fatal status codes.
    assert mock_instructor.create.await_count == 1


def _instructor_wrapped(status: int) -> Exception:
    """The shape Instructor raises: its own exception around the SDK's status error."""
    from instructor.core.exceptions import InstructorRetryException

    sdk_error = Exception(f"Error code: {status}")
    sdk_error.status_code = status
    try:
        raise InstructorRetryException(str(sdk_error), n_attempts=1, total_usage=0) from sdk_error
    except InstructorRetryException as wrapped:
        return wrapped


async def test_recognize_reads_the_status_under_instructors_wrapper(
    mock_cloud_client, _mock_image_utils
):
    """local-runtimes-1: the wrapper has no status, so a bad key returned blank
    regions and a 503 was never retried."""
    from bibr.exceptions import UpstreamServiceError

    mock_instructor = AsyncMock()
    mock_instructor.create = AsyncMock(side_effect=_instructor_wrapped(401))
    mock_cloud_client._client = mock_instructor
    with pytest.raises(UpstreamServiceError):
        await mock_cloud_client.recognize(MagicMock(), "Text Recognition:")

    mock_instructor.create = AsyncMock(
        side_effect=[_instructor_wrapped(503), OcrResult(text="recovered")]
    )
    mock_cloud_client._rate_limiter = AsyncMock()
    with patch("bibr.local.ocr_cloud.asyncio.sleep", new_callable=AsyncMock):
        assert await mock_cloud_client.recognize(MagicMock(), "Text Recognition:") == "recovered"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_wait_for_server_is_noop(mock_cloud_client):
    """wait_for_server() completes immediately."""
    await mock_cloud_client.wait_for_server()  # should not raise


async def test_shutdown_marks_unloaded(mock_cloud_client):
    """shutdown() marks the client as not loaded."""
    assert mock_cloud_client.loaded is True
    await mock_cloud_client.shutdown()
    assert mock_cloud_client.loaded is False


# ---------------------------------------------------------------------------
# OcrResult schema
# ---------------------------------------------------------------------------


def test_ocr_result_schema():
    """OcrResult has a single text field."""
    r = OcrResult(text="hello")
    assert r.text == "hello"
    assert set(OcrResult.model_fields.keys()) == {"text"}


# ---------------------------------------------------------------------------
# Per-region output budget (Anthropic)
# ---------------------------------------------------------------------------


def _client_for(provider, max_tokens=16384):
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.ocr_vision.max_tokens = max_tokens
    return CloudOcrClient(provider=provider, settings=settings)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((708, 28), 2048),  # one footer line: the floor
        ((1000, 200), 6250),  # 200,000 px / 32
        ((1654, 2339), 16384),  # a whole A4 page at 200 DPI: the setting
    ],
)
def test_anthropic_region_budget_follows_crop_area(size, expected):
    from PIL import Image

    assert _client_for("anthropic")._max_tokens_for(Image.new("RGB", size)) == expected


def test_anthropic_region_budget_never_exceeds_the_setting():
    from PIL import Image

    client = _client_for("anthropic", max_tokens=1024)
    assert client._max_tokens_for(Image.new("RGB", (708, 28))) == 1024


@pytest.mark.parametrize("provider", ["openai", "google", "gemini"])
def test_other_providers_keep_the_full_setting(provider):
    from PIL import Image

    client = _client_for(provider)
    assert client._max_tokens_for(Image.new("RGB", (708, 28))) == 16384


async def test_recognize_sends_the_region_budget_to_anthropic(_mock_image_utils):
    from PIL import Image

    client = _client_for("anthropic")
    client._client = AsyncMock()
    client._client.create = AsyncMock(return_value=OcrResult(text="ok"))

    await client.recognize(Image.new("RGB", (708, 28)), "Text Recognition:")

    assert client._client.create.call_args.kwargs["max_tokens"] == 2048


async def test_recognize_keeps_gemini_on_the_setting(mock_cloud_client, _mock_image_utils):
    from PIL import Image

    mock_cloud_client._client = AsyncMock()
    mock_cloud_client._client.create = AsyncMock(return_value=OcrResult(text="ok"))

    await mock_cloud_client.recognize(Image.new("RGB", (708, 28)), "Text Recognition:")

    kwargs = mock_cloud_client._client.create.call_args.kwargs
    assert kwargs["generation_config"]["max_tokens"] == 4096
    assert "max_tokens" not in kwargs
