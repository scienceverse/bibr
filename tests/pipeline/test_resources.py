"""ResourceManager lifecycle — layout + segmenter."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.pipeline.resources import ResourceManager


def _candidate(backend, model, profile):
    from bibr.ocr.registry import OcrBackendCandidate

    return OcrBackendCandidate(backend=backend, model=model, profile=profile)


def test_resource_manager_accepts_serve_resources():
    layout = MagicMock(loaded=True)
    segmenter = MagicMock(loaded=True)

    manager = ResourceManager(layout=layout, segmenter=segmenter)

    assert manager.layout is layout
    assert manager.segmenter is segmenter


def test_resource_manager_accepts_resource_factories():
    layout = MagicMock(loaded=True)
    segmenter = MagicMock(loaded=True)
    layout_factory = MagicMock(return_value=layout)
    segmenter_factory = MagicMock(return_value=segmenter)
    manager = ResourceManager(
        layout_factory=layout_factory,
        segmenter_factory=segmenter_factory,
        device="cpu",
    )

    manager.ensure_layout()
    manager.ensure_segmenter()

    assert manager.layout is layout
    assert manager.segmenter is segmenter
    layout_factory.assert_called_once()
    segmenter_factory.assert_called_once()


def test_ensure_layout_loads_once():
    fake = MagicMock(loaded=True)
    factory = MagicMock(return_value=fake)
    rm = ResourceManager(layout_factory=factory)
    rm.ensure_layout()
    rm.ensure_layout()  # second call must not re-instantiate
    assert factory.call_count == 1
    assert rm.layout is fake


def test_unload_layout_calls_unload_when_loaded():
    layout = MagicMock(loaded=True)
    rm = ResourceManager(layout=layout)
    rm.unload_layout()
    layout.unload.assert_called_once()


def test_unload_layout_is_noop_when_missing():
    rm = ResourceManager()
    rm.unload_layout()  # must not raise
    assert rm.layout is None


def test_ensure_segmenter_loads_once():
    fake = MagicMock(loaded=True)
    factory = MagicMock(return_value=fake)
    rm = ResourceManager(segmenter_factory=factory)
    rm.ensure_segmenter()
    rm.ensure_segmenter()
    assert factory.call_count == 1
    assert rm.segmenter is fake


def test_unload_segmenter_calls_unload_when_loaded():
    segmenter = MagicMock(loaded=True)
    rm = ResourceManager(segmenter=segmenter)
    rm.unload_segmenter()
    segmenter.unload.assert_called_once()


def test_create_ocr_client_http_backend():
    rm = ResourceManager(
        ocr_backend="glm-http",
        ocr_url="http://host:8000",
        ocr_model="custom/glm-ocr",
    )
    with patch("bibr.ocr.registry.create") as mk:
        mk.return_value = MagicMock(loaded=True)
        rm._create_ocr_client()
    mk.assert_called_once()
    call_args = mk.call_args
    assert call_args[0][0] == "glm-http"  # first positional arg is backend name
    assert call_args[1]["base_url"] == "http://host:8000"
    assert call_args[1]["model"] == "custom/glm-ocr"


def test_create_ocr_client_paddle_http_url_keeps_concrete_backend():
    rm = ResourceManager(
        ocr_backend="paddle-http",
        ocr_url="http://host:8000",
    )

    with patch("bibr.ocr.registry.create") as mk:
        mk.return_value = MagicMock(loaded=True)
        rm._create_ocr_client()

    assert mk.call_args[0][0] == "paddle-http"
    assert mk.call_args[1]["model"] == "paddle-ocr-vl-1.6"
    assert mk.call_args[1]["profile"].name == "paddle"


def test_create_ocr_client_paddle_vllm_uses_served_alias_and_paddle_profile():
    rm = ResourceManager(ocr_backend="paddle-vllm", ocr_model="some/source-model")

    with patch("bibr.ocr.registry.create") as mk:
        mk.return_value = MagicMock(loaded=True)
        rm._create_ocr_client()

    assert mk.call_args[0][0] == "paddle-vllm"
    assert mk.call_args[1]["model"] == "paddle-ocr-vl-1.6"
    assert mk.call_args[1]["model_path"] == "some/source-model"
    assert mk.call_args[1]["profile"].name == "paddle"


def test_create_ocr_client_ocr_url_compatibility_passes_model_to_http_backend():
    rm = ResourceManager(
        ocr_backend="glm-llama",
        ocr_url="http://host:8000",
        ocr_model="custom/glm-ocr",
    )
    with patch("bibr.ocr.registry.create") as mk:
        mk.return_value = MagicMock(loaded=True)
        rm._create_ocr_client()
    mk.assert_called_once()
    call_args = mk.call_args
    assert call_args[0][0] == "glm-http"
    assert call_args[1]["model"] == "custom/glm-ocr"


def test_create_ocr_client_rejects_removed_backend_with_ocr_url():
    removed_name = "fal" + "con"
    rm = ResourceManager(
        ocr_backend=removed_name,
        ocr_url="http://host:8000",
    )

    with pytest.raises(ValueError, match=f"Unknown OCR backend: '{removed_name}'"):
        rm._create_ocr_client()


def test_create_ocr_client_serve_http_passes_model():
    import asyncio

    rm = ResourceManager(
        ocr_backend="serve-http",
        ocr_url="http://ocr.local",
        ocr_model="numind/NuExtract-2.0-2B",
        http_client="fake",
        ocr_sem_global=asyncio.Semaphore(4),
        ocr_breaker="breaker",
    )
    with patch("bibr.ocr.registry.create") as mk:
        mk.return_value = MagicMock(loaded=True)
        rm._create_ocr_client()
    call_kwargs = mk.call_args[1]
    assert call_kwargs["model"] == "numind/NuExtract-2.0-2B"


def test_create_ocr_client_glm_default():
    rm = ResourceManager(ocr_backend="glm-llama", ocr_model="model-x", device="cuda")
    with patch("bibr.ocr.registry.create") as mk:
        mk.return_value = MagicMock(loaded=True)
        rm._create_ocr_client()
    mk.assert_called_once()
    call_args = mk.call_args
    assert call_args[0][0] == "glm-llama"  # first positional arg is backend name
    assert call_args[1]["model_path"] == "model-x"
    assert call_args[1]["device"] == "cuda"


def test_start_ocr_preload_submits_once():
    rm = ResourceManager()
    with patch.object(rm, "_create_ocr_client", return_value=MagicMock(loaded=True)):
        rm.start_ocr_preload()
        first = rm._ocr_future
        rm.start_ocr_preload()
    assert rm._ocr_future is first  # idempotent


def test_start_ocr_preload_noop_if_loaded():
    rm = ResourceManager()
    rm._ocr = MagicMock(loaded=True)
    rm.start_ocr_preload()
    assert rm._ocr_future is None


@pytest.mark.asyncio
async def test_await_ocr_no_preload_loads_sync():
    rm = ResourceManager()
    fake = MagicMock(loaded=True)
    with patch.object(rm, "_create_ocr_client", return_value=fake):
        await rm.await_ocr()
    assert rm._ocr is fake


@pytest.mark.asyncio
async def test_readiness_failure_without_cooldown_discards_client():
    """Guard: only a backend that owns a cross-request readiness cooldown is
    kept after a readiness failure. A plain HTTP client also defines
    ``wait_for_server``, so the gate must not use that — it must still be
    discarded and shut down, never reused."""
    from bibr.local.ocr_transport import BaseHttpOcrClient

    class _PlainHttpClient(BaseHttpOcrClient):
        name = "glm-http"

        def __init__(self):
            self._loaded = True
            self.shutdown_calls = 0

        async def wait_for_server(self):
            raise RuntimeError("not ready")

        async def shutdown(self):
            self.shutdown_calls += 1
            self._loaded = False

    rm = ResourceManager(ocr_backend="glm-http")
    broken = _PlainHttpClient()
    with patch.object(rm, "_create_ocr_client", return_value=broken):
        ready = AsyncMock(side_effect=RuntimeError("not ready"))
        with patch.object(rm, "_await_ocr_client_ready", new=ready):
            with pytest.raises(RuntimeError, match="not ready"):
                await rm.await_ocr()
    assert rm._ocr is None
    assert broken.shutdown_calls == 1
    assert broken.loaded is False


@pytest.mark.asyncio
async def test_readiness_failure_with_cooldown_publishes_without_shutdown():
    """A backend with ``keeps_readiness_cooldown`` stays published and live
    after a readiness failure, so later requests fail fast on its cooldown."""
    from bibr.local.ocr_transport import BaseHttpOcrClient

    class _CooldownHttpClient(BaseHttpOcrClient):
        name = "glm-http"
        keeps_readiness_cooldown = True

        def __init__(self):
            self._loaded = True
            self.shutdown_calls = 0

        async def wait_for_server(self):
            raise RuntimeError("not ready")

        async def shutdown(self):
            self.shutdown_calls += 1
            self._loaded = False

    rm = ResourceManager(ocr_backend="glm-http")
    failed = _CooldownHttpClient()
    with patch.object(rm, "_create_ocr_client", return_value=failed):
        ready = AsyncMock(side_effect=RuntimeError("not ready"))
        with patch.object(rm, "_await_ocr_client_ready", new=ready):
            with pytest.raises(RuntimeError, match="not ready"):
                await rm.await_ocr()
    assert rm._ocr is failed
    assert failed.shutdown_calls == 0
    assert failed.loaded is True


@pytest.mark.asyncio
async def test_serve_http_readiness_failure_cools_down_across_requests():
    """x-concurrency-2: after one full readiness poll fails, the failed
    serve-http backend is kept so its own cooldown fail-fasts the next
    request instead of re-polling for another full timeout."""
    import httpx

    from bibr.config import GlobalSettings
    from bibr.exceptions import UpstreamServiceError
    from bibr.serve import ocr_backend as _serve_ocr_backend  # noqa: F401 — registers serve-http
    from bibr.utils.circuit_breaker import AsyncCircuitBreaker

    polls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        polls["n"] += 1
        return httpx.Response(503, json={"error": {"message": "model loading"}})

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ocr.local"
    )
    settings = GlobalSettings(pipeline={"deployment_ready_timeout": 1})
    rm = ResourceManager(
        ocr_backend="serve-http",
        ocr_url="http://ocr.local",
        http_client=http_client,
        ocr_sem_global=asyncio.Semaphore(4),
        ocr_breaker=AsyncCircuitBreaker(failure_threshold=3, reset_timeout=60, name="t"),
        settings=settings,
    )

    started = time.monotonic()
    with pytest.raises(UpstreamServiceError):
        await rm.await_ocr()
    assert time.monotonic() - started >= 1.0
    polls_after_first = polls["n"]
    assert polls_after_first >= 1

    # The next request reuses the failed backend and fails fast on its
    # cooldown — no new /v1/models polls (OcrStage then surfaces the
    # cooldown via rm.ocr.wait_for_server()).
    started = time.monotonic()
    await rm.await_ocr()
    assert time.monotonic() - started < 1.0
    assert polls["n"] == polls_after_first
    assert rm.ocr is not None
    with pytest.raises(UpstreamServiceError, match="retried"):
        await rm.ocr.wait_for_server()


@pytest.mark.asyncio
async def test_automatic_paddle_falls_back_only_during_startup_and_records_identity(monkeypatch):
    """Constructor/readiness failures release owned clients before trying the next runtime."""
    rm = ResourceManager(ocr_backend="paddle")
    candidates = (
        _candidate("paddle-rapid-mlx", "paddle/rapid", "paddle"),
        _candidate("paddle-mlx-vlm", "paddle/mlx", "paddle"),
        _candidate("glm-rapid-mlx", "glm/rapid", "glm"),
    )
    rejected = MagicMock(loaded=True)
    rejected.keeps_readiness_cooldown = False  # local engines own no cooldown
    rejected.wait_for_server = AsyncMock(side_effect=RuntimeError("smoke rejected"))
    rejected.shutdown = AsyncMock()
    accepted = MagicMock(loaded=True)
    accepted.wait_for_server = AsyncMock()

    monkeypatch.setattr(
        "bibr.ocr.registry.resolve_backend_candidates", lambda name, settings: candidates
    )
    with patch.object(
        rm,
        "_create_ocr_client_for",
        side_effect=[FileNotFoundError("rapid-mlx absent"), rejected, accepted],
    ) as create:
        await rm.await_ocr()

    assert [call.args[0].backend for call in create.call_args_list] == [
        "paddle-rapid-mlx",
        "paddle-mlx-vlm",
        "glm-rapid-mlx",
    ]
    rejected.shutdown.assert_awaited_once()
    assert rm.ocr is accepted
    assert rm.ocr_runtime_identity.backend == "glm-rapid-mlx"
    assert rm.ocr_fallback_reason == (
        "paddle-rapid-mlx: install rapid-mlx and make its executable available; "
        "paddle-mlx-vlm: install MLX-VLM dependencies and the Paddle model"
    )


@pytest.mark.asyncio
async def test_automatic_paddle_aggregates_all_startup_failures(monkeypatch):
    from bibr.exceptions import UpstreamServiceError

    rm = ResourceManager(ocr_backend="paddle")
    candidates = (
        _candidate("paddle-rapid-mlx", "paddle/rapid", "paddle"),
        _candidate("glm-llama", "glm/llama", "glm"),
    )
    monkeypatch.setattr(
        "bibr.ocr.registry.resolve_backend_candidates", lambda name, settings: candidates
    )
    with (
        patch.object(
            rm,
            "_create_ocr_client_for",
            side_effect=[RuntimeError("unavailable"), RuntimeError("unavailable")],
        ) as create,
        pytest.raises(UpstreamServiceError, match="paddle-rapid-mlx.*glm-llama"),
    ):
        await rm.await_ocr()

    assert create.call_count == 2


@pytest.mark.asyncio
async def test_selected_automatic_candidate_does_not_fallback_after_recognize_failure(monkeypatch):
    rm = ResourceManager(ocr_backend="paddle")
    selected = MagicMock(loaded=True)
    selected.wait_for_server = AsyncMock()
    selected.recognize = AsyncMock(side_effect=RuntimeError("inference failed"))
    candidates = (
        _candidate("paddle-vllm", "paddle-served", "paddle"),
        _candidate("glm-llama", "glm/llama", "glm"),
    )
    monkeypatch.setattr(
        "bibr.ocr.registry.resolve_backend_candidates", lambda name, settings: candidates
    )
    with patch.object(rm, "_create_ocr_client_for", return_value=selected) as create:
        await rm.await_ocr()
        with pytest.raises(RuntimeError, match="inference failed"):
            await rm.ocr.recognize(None, "OCR:")

    assert create.call_count == 1


@pytest.mark.asyncio
async def test_explicit_paddle_rapid_preserves_request_model_and_profile(monkeypatch):
    rm = ResourceManager(
        ocr_backend="paddle-rapid-mlx", ocr_model="paddle/custom", ocr_profile="glm"
    )
    client = MagicMock(loaded=True)
    client.wait_for_server = AsyncMock()
    monkeypatch.setattr(
        "bibr.ocr.registry.resolve_backend_candidates",
        lambda name, settings: (_candidate("paddle-rapid-mlx", "paddle/default", "paddle"),),
    )
    with patch.object(rm, "_create_ocr_client_for", return_value=client) as create:
        await rm.await_ocr()

    assert create.call_args.args[0].model == "paddle/custom"
    assert create.call_args.args[0].profile == "glm"
    assert rm.ocr_runtime_identity.backend == "paddle-rapid-mlx"
    assert rm.ocr_runtime_identity.model == "paddle/custom"
    assert rm.ocr_runtime_identity.profile == "glm"


@pytest.mark.asyncio
async def test_explicit_paddle_vllm_keeps_source_model_but_records_served_alias(monkeypatch):
    rm = ResourceManager(
        ocr_backend="paddle-vllm", ocr_model="paddle/source-custom", ocr_profile="glm"
    )
    client = MagicMock(loaded=True)
    client.wait_for_server = AsyncMock()
    monkeypatch.setattr(
        "bibr.ocr.registry.resolve_backend_candidates",
        lambda name, settings: (_candidate("paddle-vllm", "paddle-ocr-vl-1.6", "paddle"),),
    )
    with patch.object(rm, "_create_ocr_client_for", return_value=client) as create:
        await rm.await_ocr()

    assert create.call_args.args[0].model == "paddle-ocr-vl-1.6"
    assert create.call_args.args[0].profile == "glm"
    assert rm.ocr_runtime_identity.model == "paddle-ocr-vl-1.6"
    assert rm.ocr_runtime_identity.profile == "glm"


@pytest.mark.asyncio
async def test_explicit_backend_uses_settings_profile_for_runtime_and_cache_identity(monkeypatch):
    from bibr.ocr.profiles import resolve_ocr_runtime_identity
    from bibr.pipeline.context import RunConfig

    rm = ResourceManager(ocr_backend="paddle-rapid-mlx")
    monkeypatch.setattr(rm._settings.ocr, "profile", "glm")
    client = MagicMock(loaded=True)
    client.wait_for_server = AsyncMock()
    monkeypatch.setattr(
        "bibr.ocr.registry.resolve_backend_candidates",
        lambda name, settings: (_candidate("paddle-rapid-mlx", "paddle/default", "paddle"),),
    )
    with patch.object(rm, "_create_ocr_client_for", return_value=client) as create:
        await rm.await_ocr()

    cache_export_identity = resolve_ocr_runtime_identity(
        RunConfig(ocr_backend="paddle-rapid-mlx"), rm._settings
    )
    assert create.call_args.args[0].profile == "glm"
    assert rm.ocr_runtime_identity.profile == "glm"
    assert cache_export_identity.profile == "glm"


@pytest.mark.asyncio
async def test_automatic_fallback_reason_is_bounded_actionable_and_redacted(monkeypatch):
    from bibr.exceptions import UpstreamServiceError

    rm = ResourceManager(ocr_backend="paddle")
    candidates = (
        _candidate("paddle-rapid-mlx", "paddle/rapid", "paddle"),
        _candidate("glm-llama", "glm/llama", "glm"),
    )
    monkeypatch.setattr(
        "bibr.ocr.registry.resolve_backend_candidates", lambda name, settings: candidates
    )
    with (
        patch.object(
            rm,
            "_create_ocr_client_for",
            side_effect=[
                RuntimeError("https://alice:secret-token@host/failed"),
                RuntimeError("secret"),
            ],
        ),
        pytest.raises(UpstreamServiceError, match="paddle-rapid-mlx.*glm-llama"),
    ):
        await rm.await_ocr()

    assert "secret" not in rm.ocr_fallback_reason
    assert "https" not in rm.ocr_fallback_reason
    assert "install" in rm.ocr_fallback_reason
    assert len(rm.ocr_fallback_reason) <= 500


@pytest.mark.asyncio
async def test_concurrent_await_ocr_constructs_one_client():
    rm = ResourceManager()
    constructed = []

    def create_client():
        time.sleep(0.05)
        client = MagicMock(loaded=True)
        constructed.append(client)
        return client

    with patch.object(rm, "_create_ocr_client", side_effect=create_client):
        await asyncio.gather(rm.await_ocr(), rm.await_ocr())

    assert len(constructed) == 1
    assert rm.ocr is constructed[0]


@pytest.mark.asyncio
async def test_await_ocr_raises_when_factory_returns_none():
    rm = ResourceManager(ocr_backend="glm-llama")
    with (
        patch.object(rm, "_create_ocr_client", return_value=None),
        pytest.raises(RuntimeError, match="glm-llama"),
    ):
        await rm.await_ocr()
    assert rm._ocr is None


@pytest.mark.asyncio
async def test_await_ocr_preload_raises_when_factory_returns_none():
    rm = ResourceManager(ocr_backend="glm-llama")
    with patch.object(rm, "_create_ocr_client", return_value=None):
        rm.start_ocr_preload()
        with pytest.raises(RuntimeError, match="glm-llama"):
            await rm.await_ocr()
    assert rm._ocr is None


@pytest.mark.asyncio
async def test_shutdown_ocr_calls_sync_shutdown():
    rm = ResourceManager()
    client = MagicMock()
    client.shutdown = MagicMock(return_value=None)
    rm._ocr = client
    await rm.shutdown_ocr()
    client.shutdown.assert_called_once()
    assert rm.ocr is None


@pytest.mark.asyncio
async def test_start_llm_server_vllm_mlx():
    rm = ResourceManager()
    fake = MagicMock()
    with patch("bibr.local.llm.VllmMlxLlmServer", return_value=fake) as constructor:
        await rm.start_llm_server(backend="vllm-mlx")
    constructor.assert_called_once_with(settings=rm._settings)
    fake.configure_llm_client.assert_called_once()
    assert rm._llm_server is fake


@pytest.mark.asyncio
async def test_start_llm_server_cloud_is_noop():
    rm = ResourceManager()
    await rm.start_llm_server(backend="cloud")
    assert rm._llm_server is None


def test_shutdown_llm_server():
    rm = ResourceManager()
    fake = MagicMock()
    rm._llm_server = fake
    rm.shutdown_llm_server()
    fake.shutdown.assert_called_once()
    assert rm._llm_server is None


def test_resource_manager_defaults_serve_fields_to_none():
    from bibr.pipeline.resources import ResourceManager

    rm = ResourceManager()
    assert rm.http_client is None
    assert rm.ocr_sem_global is None
    assert rm.ocr_breaker is None


def test_resource_manager_accepts_serve_fields():
    import asyncio

    from bibr.pipeline.resources import ResourceManager

    sem_g = asyncio.Semaphore(4)
    rm = ResourceManager(
        http_client="fake",
        ocr_sem_global=sem_g,
        ocr_breaker="breaker",
    )
    assert rm.http_client == "fake"
    assert rm.ocr_sem_global is sem_g
    assert rm.ocr_breaker == "breaker"


# --- Preloaded-but-unclaimed OCR engine (audit [21]) -------------------------


def test_shutdown_drains_an_ocr_preload_nobody_claimed():
    """Only await_ocr, reached from OcrStage, transfers the preload result.

    When an earlier stage raises — classically the layout model OOMing on the
    GPU the preload just claimed — the future was never drained and the
    managed inference subprocess survived the run.
    """
    import concurrent.futures

    client = MagicMock()
    client.shutdown = MagicMock(return_value=None)
    executor = concurrent.futures.ThreadPoolExecutor(1)
    rm = ResourceManager()
    rm._ocr_executor = executor
    rm._ocr_future = executor.submit(lambda: client)

    asyncio.run(rm.shutdown_ocr())

    client.shutdown.assert_called_once()
    assert rm._ocr_future is None
    assert rm._ocr_executor is None


def test_shutdown_does_not_double_shutdown_a_claimed_engine():
    import concurrent.futures

    client = MagicMock()
    client.shutdown = MagicMock(return_value=None)
    executor = concurrent.futures.ThreadPoolExecutor(1)
    rm = ResourceManager()
    rm._ocr = client
    rm._ocr_executor = executor
    rm._ocr_future = executor.submit(lambda: client)

    asyncio.run(rm.shutdown_ocr())

    client.shutdown.assert_called_once()


def test_shutdown_survives_a_preload_that_raised():
    import concurrent.futures

    def _boom():
        raise RuntimeError("OCR backend failed to start")

    executor = concurrent.futures.ThreadPoolExecutor(1)
    rm = ResourceManager()
    rm._ocr_executor = executor
    rm._ocr_future = executor.submit(_boom)

    asyncio.run(rm.shutdown_ocr())

    assert rm._ocr_future is None
