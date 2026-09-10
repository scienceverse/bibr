"""Contract tests for the shared BaseHttpOcrClient transport.

These exercise the machinery used by ``HttpOcrClient``: connection setup,
health polling, retry/backoff, response parsing, and teardown. A minimal
subclass supplies the payload builder so the shared paths can be tested in
isolation.
"""

from unittest import mock

import pytest

httpx = pytest.importorskip("httpx")

from bibr.exceptions import UpstreamServiceError  # noqa: E402
from bibr.local.ocr_transport import BaseHttpOcrClient  # noqa: E402
from bibr.ocr.profiles import GLM_PROFILE  # noqa: E402


class _StubClient(BaseHttpOcrClient):
    name = "stub-http"
    _DEFAULT_MODEL = "stub-model"
    _SERVICE_LABEL = "Stub-OCR"

    def __init__(self, **kw):
        kw.setdefault("profile", GLM_PROFILE)
        super().__init__(**kw)

    def _build_payload(self, image_b64: str, prompt: str) -> dict:
        return {"model": self._model, "image": image_b64, "prompt": prompt}


class _CleaningClient(_StubClient):
    def _clean_output(self, text: str) -> str:
        return text.replace("<junk>", "").strip()


def _make(**kw):
    return _StubClient(base_url="http://localhost:9999", **kw)


@pytest.fixture
def no_sleep(monkeypatch):
    async def fake_sleep(delay):
        pass

    monkeypatch.setattr("asyncio.sleep", fake_sleep)


class TestConstruction:
    def test_default_model_and_loaded(self):
        client = _make()
        assert client._model == "stub-model"
        assert client.loaded is True

    def test_model_override(self):
        client = _make(model="custom/model")
        assert client._model == "custom/model"

    def test_base_url_trailing_slash_stripped(self):
        client = _StubClient(base_url="http://localhost:1/", model="custom/model")
        assert client._base_url == "http://localhost:1"

    def test_configured_bearer_token_attached(self, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.ocr, "api_key", "ocr-secret")
        client = _StubClient(base_url="https://ocr.example.com")
        assert client._client.headers["authorization"] == "Bearer ocr-secret"


class TestSendRequest:
    async def test_success_returns_content(self):
        client = _make()

        async def fake_post(url, *, json=None, **_kw):
            assert json == {"model": "stub-model", "image": "B64", "prompt": "P"}

            class _Resp:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "hello"}}]}

            return _Resp()

        client._client.post = fake_post
        assert await client._send_request("B64", "P") == "hello"

    async def test_success_preserves_finish_reason(self):
        client = _make()

        async def fake_post(url, *, json=None, **_kw):  # noqa: ARG001
            class _Resp:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {
                        "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]
                    }

            return _Resp()

        client._client.post = fake_post
        result = await client._send_request("B64", "P")

        assert result == "partial"
        assert result.finish_reason == "length"

    async def test_clean_output_hook_applied(self):
        client = _CleaningClient(base_url="http://localhost:9999")

        async def fake_post(url, *, json=None, **_kw):
            class _Resp:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "hi<junk> "}}]}

            return _Resp()

        client._client.post = fake_post
        assert await client._send_request("B64", "P") == "hi"

    async def test_bad_structure_raises_valueerror(self):
        client = _make()

        async def fake_post(url, *, json=None, **_kw):
            class _Resp:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"unexpected": True}

            return _Resp()

        client._client.post = fake_post
        with pytest.raises(ValueError, match="Stub-OCR response structure"):
            await client._send_request("B64", "P")

    async def test_retries_on_retryable_status(self, no_sleep):
        client = _make()
        calls = {"n": 0}

        async def fake_post(url, *, json=None, **_kw):
            calls["n"] += 1

            class _Resp:
                status_code = 503 if calls["n"] == 1 else 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "ok"}}]}

            return _Resp()

        client._client.post = fake_post
        assert await client._send_request("B64", "P") == "ok"
        assert calls["n"] == 2


class TestWaitForServer:
    async def test_returns_when_ready(self, no_sleep):
        client = _make()
        resp = httpx.Response(
            200,
            json={"data": [{"id": "stub-model"}]},
            request=httpx.Request("GET", "http://localhost:9999/v1/models"),
        )
        client._client.get = mock.AsyncMock(return_value=resp)
        await client.wait_for_server()
        assert client._client.get.await_count == 1

    async def test_retries_model_listing_until_expected_alias_appears(self, no_sleep):
        client = _make()
        client._client.get = mock.AsyncMock(
            side_effect=[
                httpx.Response(200, json={"data": [{"id": "other-model"}]}),
                httpx.Response(200, json={"data": [{"id": "stub-model"}]}),
            ]
        )

        await client.wait_for_server()

        assert client._client.get.await_count == 2

    async def test_wrong_model_alias_times_out_with_observed_ids(self, monkeypatch, no_sleep):
        client = _make()
        monkeypatch.setattr(client._settings.pipeline, "deployment_ready_timeout", 0.5)
        monotonic_values = iter((0.0, 0.0, 1.0))
        real_monotonic = __import__("time").monotonic
        monkeypatch.setattr("time.monotonic", lambda: next(monotonic_values, real_monotonic()))
        client._client.get = mock.AsyncMock(
            return_value=httpx.Response(200, json={"data": [{"id": "other-model"}]})
        )

        with pytest.raises(UpstreamServiceError, match="stub-model.*other-model"):
            await client.wait_for_server()

    async def test_malformed_model_listing_times_out(self, monkeypatch, no_sleep):
        client = _make()
        monkeypatch.setattr(client._settings.pipeline, "deployment_ready_timeout", 0.5)
        monotonic_values = iter((0.0, 0.0, 1.0))
        real_monotonic = __import__("time").monotonic
        monkeypatch.setattr("time.monotonic", lambda: next(monotonic_values, real_monotonic()))
        client._client.get = mock.AsyncMock(return_value=httpx.Response(200, content=b"not-json"))

        with pytest.raises(UpstreamServiceError, match="did not become ready"):
            await client.wait_for_server()

    async def test_fast_fails_after_three_refusals(self, no_sleep):
        client = _make()
        client._client.get = mock.AsyncMock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(UpstreamServiceError) as exc:
            await client.wait_for_server()
        assert client._client.get.await_count == 3
        assert "Stub-OCR server" in str(exc.value)
        assert "unreachable" in str(exc.value)


class TestShutdown:
    async def test_shutdown_sets_loaded_false(self):
        client = _make()
        await client.shutdown()
        assert client.loaded is False

    async def test_shutdown_idempotent(self):
        client = _make()
        await client.shutdown()
        await client.shutdown()
        assert client.loaded is False


def test_base_url_v1_suffix_is_stripped():
    """The deployment guide's own example carried ``/v1``; requests then hit ``/v1/v1``."""
    client = _StubClient(base_url="http://localhost:1/v1", model="custom/model")
    assert client._base_url == "http://localhost:1"
