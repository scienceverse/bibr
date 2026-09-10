"""``BibrServeOcrBackend`` — HTTPX-backed OCR backend used by ``ServePipeline``."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from bibr.config import Settings
from bibr.serve.ocr_backend import BibrServeOcrBackend
from bibr.utils.circuit_breaker import AsyncCircuitBreaker


def _ok_transport():
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "choices": [{"message": {"content": "hello world"}}],
        }
        return httpx.Response(200, content=json.dumps(payload).encode())

    return httpx.MockTransport(handler)


def _status_transport(status_code: int, body: dict | None = None):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, content=json.dumps(body or {}).encode())

    return httpx.MockTransport(handler)


def _capture_transport(captured: dict):
    """Transport that records the outgoing request JSON body and returns OK."""

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        body = {"choices": [{"message": {"content": "<fcel>ok<nl>"}}]}
        return httpx.Response(200, content=json.dumps(body).encode())

    return httpx.MockTransport(handler)


def _sequence_transport(responses: list[tuple[str, str]], payloads: list[dict]):
    """Return successive OCR choices while recording every request payload."""
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        payloads.append(json.loads(request.content))
        content, finish_reason = responses[calls]
        calls += 1
        body = {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
        return httpx.Response(200, content=json.dumps(body).encode())

    return httpx.MockTransport(handler)


def _make_backend(
    transport: httpx.MockTransport,
    breaker=None,
    sem_global=None,
    sem_per=None,
    model=None,
    profile=None,
):
    client = httpx.AsyncClient(transport=transport, base_url="http://ocr.local")
    kwargs = {
        "base_url": "http://ocr.local",
        "http_client": client,
        "sem_global": sem_global or asyncio.Semaphore(4),
        "sem_per_file": sem_per or asyncio.Semaphore(2),
        "breaker": breaker or AsyncCircuitBreaker(failure_threshold=3, reset_timeout=60, name="t"),
    }
    if model is not None:
        kwargs["model"] = model
    if profile is not None:
        kwargs["profile"] = profile
    return BibrServeOcrBackend(**kwargs)


def _tiny_image() -> Image.Image:
    return Image.new("RGB", (16, 16), color="white")


class TestContract:
    def test_satisfies_ocr_backend_protocol(self):
        from bibr.ocr.backend import OcrBackend

        backend = _make_backend(_ok_transport())
        assert isinstance(backend, OcrBackend)

    def test_registered_under_name_serve_http(self):
        # create() is how ResourceManager instantiates; must at minimum find the class
        assert "serve-http" in _registered_names()

    def test_loaded_is_true_after_init(self):
        backend = _make_backend(_ok_transport())
        assert backend.loaded is True


def _registered_names():
    # Import for side-effect registration.
    from bibr.ocr.registry import _BACKENDS
    from bibr.serve import ocr_backend as _  # noqa: F401

    return set(_BACKENDS)


class TestRecognize:
    @pytest.mark.asyncio
    async def test_happy_path_returns_content(self):
        pytest.importorskip("cv2")
        backend = _make_backend(_ok_transport())
        out = await backend.recognize(_tiny_image(), "describe")
        assert out == "hello world"

    @pytest.mark.asyncio
    async def test_preserves_finish_reason_on_string_compatible_result(self):
        pytest.importorskip("cv2")

        async def handler(request):  # noqa: ARG001
            body = {
                "choices": [{"message": {"content": "partial table"}, "finish_reason": "length"}]
            }
            return httpx.Response(200, content=json.dumps(body).encode())

        backend = _make_backend(httpx.MockTransport(handler))
        out = await backend.recognize(_tiny_image(), "Table Recognition:")

        assert out == "partial table"
        assert out.finish_reason == "length"

    @pytest.mark.asyncio
    async def test_retries_length_truncated_paddle_table_once_at_8192(self):
        pytest.importorskip("cv2")
        from bibr.ocr.profiles import PADDLE_PROFILE

        payloads: list[dict] = []
        backend = _make_backend(
            _sequence_transport(
                [
                    ("<fcel>A<fcel>B", "length"),
                    ("<fcel>A<fcel>B<nl>", "stop"),
                ],
                payloads,
            ),
            model="paddle-ocr-vl-1.6",
            profile=PADDLE_PROFILE,
        )

        out = await backend.recognize(_tiny_image(), "Table Recognition:")

        assert out == "<fcel>A<fcel>B<nl>"
        assert out.finish_reason == "stop"
        assert [payload["max_tokens"] for payload in payloads] == [4096, 8192]

    @pytest.mark.asyncio
    async def test_retries_structurally_incomplete_stopped_paddle_table(self):
        pytest.importorskip("cv2")
        from bibr.ocr.profiles import PADDLE_PROFILE

        payloads: list[dict] = []
        backend = _make_backend(
            _sequence_transport(
                [
                    ("<fcel>A<fcel>B<nl><fcel>C<nl>", "stop"),
                    ("<fcel>A<fcel>B<nl><fcel>C<fcel>D<nl>", "stop"),
                ],
                payloads,
            ),
            profile=PADDLE_PROFILE,
        )

        out = await backend.recognize(_tiny_image(), "Table Recognition:")

        assert out == "<fcel>A<fcel>B<nl><fcel>C<fcel>D<nl>"
        assert [payload["max_tokens"] for payload in payloads] == [4096, 8192]

    @pytest.mark.asyncio
    async def test_does_not_retry_complete_paddle_table(self):
        pytest.importorskip("cv2")
        from bibr.ocr.profiles import PADDLE_PROFILE

        payloads: list[dict] = []
        backend = _make_backend(
            _sequence_transport([("<fcel>A<nl>", "stop")], payloads),
            profile=PADDLE_PROFILE,
        )

        out = await backend.recognize(_tiny_image(), "Table Recognition:")

        assert out == "<fcel>A<nl>"
        assert len(payloads) == 1
        assert payloads[0]["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_does_not_retry_paddle_formula_or_glm_table(self):
        pytest.importorskip("cv2")
        from bibr.ocr.profiles import GLM_PROFILE, PADDLE_PROFILE

        paddle_payloads: list[dict] = []
        paddle = _make_backend(
            _sequence_transport([("partial formula", "length")], paddle_payloads),
            profile=PADDLE_PROFILE,
        )
        glm_payloads: list[dict] = []
        glm = _make_backend(
            _sequence_transport([("partial table", "length")], glm_payloads),
            profile=GLM_PROFILE,
        )

        await paddle.recognize(_tiny_image(), "Formula Recognition:")
        await glm.recognize(_tiny_image(), "Table Recognition:")

        assert len(paddle_payloads) == 1
        assert len(glm_payloads) == 1

    @pytest.mark.asyncio
    async def test_empty_recovery_falls_back_to_initial_table(self):
        pytest.importorskip("cv2")
        from bibr.ocr.profiles import PADDLE_PROFILE

        payloads: list[dict] = []
        backend = _make_backend(
            _sequence_transport(
                [("<fcel>A", "length"), ("", "stop")],
                payloads,
            ),
            profile=PADDLE_PROFILE,
        )

        out = await backend.recognize(_tiny_image(), "Table Recognition:")

        assert out == "<fcel>A"
        assert out.finish_reason == "length"
        assert [payload["max_tokens"] for payload in payloads] == [4096, 8192]

    @pytest.mark.asyncio
    async def test_recovery_error_falls_back_to_initial_table(self, monkeypatch):
        pytest.importorskip("cv2")
        from bibr.ocr.backend import OcrText
        from bibr.ocr.profiles import PADDLE_PROFILE

        backend = _make_backend(_ok_transport(), profile=PADDLE_PROFILE)
        post = AsyncMock(
            side_effect=[
                OcrText("<fcel>A", finish_reason="length"),
                RuntimeError("recovery failed"),
            ]
        )
        monkeypatch.setattr(backend, "_post_with_retry", post)

        out = await backend.recognize(_tiny_image(), "Table Recognition:")

        assert out == "<fcel>A"
        assert out.finish_reason == "length"
        assert post.await_count == 2

    @pytest.mark.asyncio
    async def test_recovery_cancellation_propagates(self, monkeypatch):
        pytest.importorskip("cv2")
        from bibr.ocr.backend import OcrText
        from bibr.ocr.profiles import PADDLE_PROFILE

        backend = _make_backend(_ok_transport(), profile=PADDLE_PROFILE)
        post = AsyncMock(
            side_effect=[
                OcrText("<fcel>A", finish_reason="length"),
                asyncio.CancelledError(),
            ]
        )
        monkeypatch.setattr(backend, "_post_with_retry", post)

        with pytest.raises(asyncio.CancelledError):
            await backend.recognize(_tiny_image(), "Table Recognition:")

    @pytest.mark.asyncio
    async def test_acquires_both_semaphores(self):
        pytest.importorskip("cv2")
        sem_g = asyncio.Semaphore(1)
        sem_f = asyncio.Semaphore(1)
        backend = _make_backend(_ok_transport(), sem_global=sem_g, sem_per=sem_f)

        async def one_call():
            await backend.recognize(_tiny_image(), "p")

        await asyncio.gather(one_call(), one_call(), one_call())
        # Both semaphores must be fully released after completion.
        assert sem_g._value == 1
        assert sem_f._value == 1

    @pytest.mark.asyncio
    async def test_retries_on_retryable_status(self, monkeypatch):
        pytest.importorskip("cv2")
        monkeypatch.setattr("bibr.serve.ocr_backend._OCR_RETRY_BACKOFF_BASE", 0)
        attempts = {"n": 0}

        async def handler(request):
            attempts["n"] += 1
            if attempts["n"] < 2:
                return httpx.Response(503, content=b"{}")
            return httpx.Response(
                200, content=json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
            )

        backend = _make_backend(httpx.MockTransport(handler))
        out = await backend.recognize(_tiny_image(), "p")
        assert out == "ok"
        assert attempts["n"] == 2

    @pytest.mark.asyncio
    async def test_open_breaker_raises_upstream_service_error(self, monkeypatch):
        """A tripped breaker is a SYSTEMIC OCR outage. ``recognize`` must surface
        it as ``UpstreamServiceError`` (a ``BibrError`` → fails the page/file and
        maps to HTTP 502) — never leak ``CircuitOpenError``, which OcrStage would
        silently convert to blank region content."""
        pytest.importorskip("cv2")
        monkeypatch.setattr("bibr.serve.ocr_backend._OCR_RETRY_BACKOFF_BASE", 0)
        from bibr.exceptions import UpstreamServiceError
        from bibr.utils.circuit_breaker import CircuitOpenError

        breaker = AsyncCircuitBreaker(
            failure_threshold=2, reset_timeout=60, failure_dedup_window=0.0, name="t"
        )
        backend = _make_backend(_status_transport(500), breaker=breaker)
        # Two real failures trip the breaker.
        for _ in range(2):
            with pytest.raises(httpx.HTTPStatusError):
                await backend.recognize(_tiny_image(), "p")
        # Third call: breaker is OPEN → translated to an upstream error.
        with pytest.raises(UpstreamServiceError) as ei:
            await backend.recognize(_tiny_image(), "p")
        assert ei.value.service_name == "ocr"
        assert isinstance(ei.value.original_error, CircuitOpenError)


class TestModelName:
    @pytest.mark.asyncio
    async def test_default_model_in_payload_is_glm_ocr(self):
        """No model configured → the documented ``glm-ocr`` served-name alias."""
        pytest.importorskip("cv2")
        captured: dict = {}
        backend = _make_backend(_capture_transport(captured))
        await backend.recognize(_tiny_image(), "p")
        assert captured["payload"]["model"] == "glm-ocr"

    @pytest.mark.asyncio
    async def test_model_override_flows_to_payload(self):
        """An explicit GLM-profile alias reaches the chat-completions model field."""
        pytest.importorskip("cv2")
        from bibr.ocr.profiles import GLM_PROFILE

        captured: dict = {}
        backend = _make_backend(
            _capture_transport(captured),
            model="numind/NuExtract-2.0-2B",
            profile=GLM_PROFILE,
        )
        await backend.recognize(_tiny_image(), "p")
        assert captured["payload"]["model"] == "numind/NuExtract-2.0-2B"

    @pytest.mark.asyncio
    async def test_paddle_profile_flows_to_payload_without_glm_only_knobs(self):
        pytest.importorskip("cv2")
        from bibr.ocr.profiles import PADDLE_PROFILE

        captured: dict = {}
        backend = _make_backend(
            _capture_transport(captured),
            model="paddle-ocr-vl-1.6",
            profile=PADDLE_PROFILE,
        )
        await backend.recognize(_tiny_image(), "OCR:")

        assert {
            key: captured["payload"][key] for key in ("model", "max_tokens", "temperature")
        } == {
            "model": "paddle-ocr-vl-1.6",
            "max_tokens": 1024,
            "temperature": 0.0,
        }
        assert "top_k" not in captured["payload"]
        assert "repetition_penalty" not in captured["payload"]

    @pytest.mark.asyncio
    async def test_paddle_table_profile_uses_larger_output_budget(self):
        pytest.importorskip("cv2")
        from bibr.ocr.profiles import PADDLE_PROFILE

        captured: dict = {}
        backend = _make_backend(
            _capture_transport(captured),
            model="paddle-ocr-vl-1.6",
            profile=PADDLE_PROFILE,
        )
        await backend.recognize(_tiny_image(), "Table Recognition:")

        assert captured["payload"]["max_tokens"] == 4096


class TestWaitForServer:
    def test_default_ready_timeout_uses_pipeline_setting(self, monkeypatch):
        monkeypatch.setattr(Settings.pipeline, "deployment_ready_timeout", 7)
        backend = _make_backend(_ok_transport())
        assert backend._ready_timeout == 7

    @pytest.mark.asyncio
    async def test_polls_models_until_configured_alias_appears(self):
        attempts = {"n": 0}

        async def handler(request):
            attempts["n"] += 1
            if attempts["n"] < 2:
                return httpx.Response(200, json={"data": [{"id": "other-model"}]})
            return httpx.Response(200, json={"data": [{"id": "glm-ocr"}]})

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://ocr.local"
        )
        backend = BibrServeOcrBackend(
            base_url="http://ocr.local",
            http_client=client,
            sem_global=asyncio.Semaphore(1),
            sem_per_file=asyncio.Semaphore(1),
            breaker=AsyncCircuitBreaker(failure_threshold=3, reset_timeout=60, name="t"),
            ready_poll_interval=0.01,
            ready_timeout=5.0,
        )
        await backend.wait_for_server()
        assert attempts["n"] >= 2

    @pytest.mark.asyncio
    async def test_readiness_failure_keeps_the_endpoint_and_model_list_out_of_the_error(
        self, caplog
    ):
        """The error text becomes a 502 body; the internal URL and what the server
        serves belong in the operator log only."""
        from bibr.exceptions import UpstreamServiceError

        async def handler(request):
            return httpx.Response(200, json={"data": [{"id": "secret-internal-model"}]})

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://ocr.internal.example"
        )
        backend = BibrServeOcrBackend(
            base_url="http://ocr.internal.example:8080",
            http_client=client,
            sem_global=asyncio.Semaphore(1),
            sem_per_file=asyncio.Semaphore(1),
            breaker=AsyncCircuitBreaker(failure_threshold=3, reset_timeout=60, name="t"),
            ready_poll_interval=0.01,
            ready_timeout=0.05,
        )
        with caplog.at_level("ERROR", logger="bibr.serve.ocr_backend"):
            with pytest.raises(UpstreamServiceError) as exc_info:
                await backend.wait_for_server()
        message = str(exc_info.value)
        assert "ocr.internal.example" not in message
        assert "secret-internal-model" not in message
        assert "glm-ocr" in message  # the expected alias is deployment-visible config
        logged = " ".join(r.getMessage() for r in caplog.records)
        assert "ocr.internal.example:8080" in logged
        assert "secret-internal-model" in logged

        # The cooldown path is client-facing too.
        with pytest.raises(UpstreamServiceError) as cooled:
            await backend.wait_for_server()
        assert "ocr.internal.example" not in str(cooled.value)
        assert "unavailable" in str(cooled.value)

    @pytest.mark.asyncio
    async def test_unreachable_server_error_names_no_endpoint(self):
        from bibr.exceptions import UpstreamServiceError

        async def handler(request):
            raise httpx.ConnectError("refused", request=request)

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://ocr.internal.example"
        )
        backend = BibrServeOcrBackend(
            base_url="http://ocr.internal.example:8080",
            http_client=client,
            sem_global=asyncio.Semaphore(1),
            sem_per_file=asyncio.Semaphore(1),
            breaker=AsyncCircuitBreaker(failure_threshold=3, reset_timeout=60, name="t"),
            ready_poll_interval=0.01,
            ready_timeout=0.05,
        )
        with pytest.raises(UpstreamServiceError, match="unreachable") as exc_info:
            await backend.wait_for_server()
        assert "ocr.internal.example" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_keeps_polling_refusals_until_ready_within_deadline(self):
        attempts = {"n": 0}

        async def handler(request):
            attempts["n"] += 1
            if attempts["n"] <= 4:
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(200, json={"data": [{"id": "glm-ocr"}]})

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://ocr.local"
        )
        backend = BibrServeOcrBackend(
            base_url="http://ocr.local",
            http_client=client,
            sem_global=asyncio.Semaphore(1),
            sem_per_file=asyncio.Semaphore(1),
            breaker=AsyncCircuitBreaker(failure_threshold=3, reset_timeout=60, name="t"),
            ready_poll_interval=0.01,
            ready_timeout=1.0,
        )

        await asyncio.wait_for(backend.wait_for_server(), timeout=0.5)
        assert attempts["n"] == 5
