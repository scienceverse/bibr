"""``bibr serve`` on the text-layer route: no OCR server, PDFs read from their own text."""

from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import MagicMock

import pytest

pytest.importorskip("litserve")

from bibr.config import GlobalSettings
from bibr.serve.deployments.pipeline import BibrPipelineAPI
from bibr.serve.pipeline import ServePipeline, serve_ocr_route

_LLM_ENV = (
    "GOOGLE_API_KEY",
    "LLM_API_KEY",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "LLM_BACKEND",
    "PIPELINE_MODE",
)
_OCR_ENV = ("OCR_BACKEND", "OCR_MODEL", "OCR_PROFILE", "OCR_BASE_URL")


@pytest.fixture
def bare(monkeypatch):
    """An environment that configures neither an LLM nor an OCR route."""
    for var in _LLM_ENV + _OCR_ENV:
        monkeypatch.delenv(var, raising=False)


def _settings(monkeypatch, **env) -> GlobalSettings:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    settings = GlobalSettings()
    settings.ror.enrich = False
    return settings


# --- the rule -------------------------------------------------------------------


def test_a_fast_server_with_no_ocr_route_reads_the_text_layer(bare, monkeypatch):
    assert serve_ocr_route(_settings(monkeypatch)) == "text-layer"


@pytest.mark.parametrize(
    "env",
    [
        {"OCR_BASE_URL": "http://ocr.local:8080"},
        {"OCR_BACKEND": "paddle-http"},
        {"OCR_MODEL": "paddle-ocr-vl-1.6"},
        {"OCR_PROFILE": "glm"},
    ],
)
def test_a_fast_server_keeps_the_ocr_server_somebody_chose(bare, monkeypatch, env):
    assert serve_ocr_route(_settings(monkeypatch, **env)) == "serve-http"


def test_a_blank_ocr_setting_chooses_nothing(bare, monkeypatch):
    assert serve_ocr_route(_settings(monkeypatch, OCR_MODEL="")) == "text-layer"


def test_a_full_server_keeps_its_ocr_server(bare, monkeypatch):
    settings = _settings(monkeypatch, GOOGLE_API_KEY="test-key")
    assert serve_ocr_route(settings) == "serve-http"
    # The deployment's mode decides, whichever way it was given.
    assert serve_ocr_route(settings, "fast") == "text-layer"
    assert serve_ocr_route(_settings(monkeypatch, PIPELINE_MODE="fast")) == "text-layer"


def test_the_text_layer_backend_can_be_asked_for_in_any_mode(bare, monkeypatch):
    settings = _settings(
        monkeypatch,
        GOOGLE_API_KEY="test-key",
        OCR_BACKEND="text-layer",
        OCR_BASE_URL="http://ocr.local:8080",
    )
    assert serve_ocr_route(settings) == "text-layer"
    assert serve_ocr_route(settings, "full") == "text-layer"


# --- the pipeline -----------------------------------------------------------------


def _pipeline(settings, **kwargs) -> ServePipeline:
    return ServePipeline(
        layout=MagicMock(),
        segmenter=MagicMock(),
        http_client=MagicMock(),
        ocr_base_url="http://ocr.local",
        ocr_sem_global=asyncio.Semaphore(16),
        ocr_breaker=MagicMock(),
        settings=settings,
        **kwargs,
    )


def test_a_text_layer_pipeline_names_no_server_and_no_model(bare, monkeypatch):
    pipeline = _pipeline(_settings(monkeypatch, OCR_BACKEND="text-layer", OCR_PROFILE="paddle"))

    assert pipeline.ocr_route == "text-layer"
    config = pipeline._config
    assert (config.ocr_backend, config.ocr_url, config.ocr_model, config.ocr_profile) == (
        "text-layer",
        None,
        None,
        None,
    )
    resources = pipeline._resources
    assert resources.ocr_url is None
    (candidate,) = resources._resolve_ocr_candidates()
    assert candidate.backend == "text-layer"


def test_a_text_layer_pipeline_is_in_text_layer_mode_for_every_request(bare, monkeypatch):
    import dataclasses
    from types import SimpleNamespace

    from bibr.ocr.text_layer import is_text_layer_mode

    pipeline = _pipeline(_settings(monkeypatch))
    # A request may ask for full mode; the route stays the deployment's.
    for mode in ("fast", "full"):
        config = dataclasses.replace(pipeline._config, mode=mode)
        ctx = SimpleNamespace(
            scratch={}, resources=None, config=config, settings=pipeline._settings
        )
        assert is_text_layer_mode(ctx)


def test_a_pipeline_with_an_ocr_server_is_as_it_was(bare, monkeypatch):
    pipeline = _pipeline(_settings(monkeypatch, OCR_BASE_URL="http://ocr.local"))

    assert pipeline.ocr_route == "serve-http"
    assert pipeline._config.mode == "fast"
    assert pipeline._config.ocr_backend == "serve-http"
    assert pipeline._config.ocr_url == "http://ocr.local"
    assert pipeline._resources.http_client is not None


# --- readiness --------------------------------------------------------------------


def _ready_client(monkeypatch, settings, requests: list):
    import httpx
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import bibr.serve.app as app_mod

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        raise httpx.ConnectError("no OCR server on this machine")

    class _Server:
        def __init__(self):
            self.app = FastAPI()

    class _PinnedClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), timeout=5.0)

    monkeypatch.setattr(httpx, "AsyncClient", _PinnedClient)
    server = _Server()
    app_mod._register_readiness_route(server, settings)
    return TestClient(server.app)


def test_a_text_layer_server_is_ready_with_no_ocr_server(bare, monkeypatch):
    settings = _settings(monkeypatch, ML_FRONT_TAGGER_MODEL_ID="")
    requests: list[str] = []

    resp = _ready_client(monkeypatch, settings, requests).get("/ready")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["checks"]["ocr"] == "not_used (text layer)"
    assert body["mode"] == "fast"
    assert requests == []  # nothing was sent to OCR_BASE_URL


def test_a_fast_server_with_a_chosen_ocr_server_still_needs_it(bare, monkeypatch):
    settings = _settings(
        monkeypatch, ML_FRONT_TAGGER_MODEL_ID="", OCR_BASE_URL="http://127.0.0.1:8080"
    )
    requests: list[str] = []

    resp = _ready_client(monkeypatch, settings, requests).get("/ready")

    assert resp.status_code == 503
    assert resp.json()["checks"]["ocr"] == "unreachable"
    assert requests == ["/health"]


def test_a_text_layer_server_does_not_judge_an_ocr_url_it_never_uses(bare, monkeypatch):
    # Docker Compose always sets OCR_BASE_URL; plain http to another host is
    # refused for an OCR server, and is nobody's business on this route.
    settings = _settings(
        monkeypatch,
        ML_FRONT_TAGGER_MODEL_ID="",
        OCR_BACKEND="text-layer",
        OCR_BASE_URL="http://bibr-ocr:8080",
    )
    requests: list[str] = []

    resp = _ready_client(monkeypatch, settings, requests).get("/ready")

    assert resp.status_code == 200
    assert requests == []


# --- the response cache -------------------------------------------------------------


def _key(**kwargs) -> str:
    digest = hashlib.sha256(b"same-pdf").hexdigest()
    return BibrPipelineAPI._cache_key(digest, None, None, False, False, None, **kwargs)


def test_the_cache_key_names_the_text_layer_route():
    served = _key(mode="fast", input_format=".pdf")

    assert _key(mode="fast", input_format=".pdf", ocr_route="serve-http") == served
    assert _key(mode="fast", input_format=".pdf", ocr_route=None) == served
    text_layer = _key(mode="fast", input_format=".pdf", ocr_route="text-layer")
    assert text_layer != served
    assert text_layer.endswith(":fast:textlayer:fmt:pdf")


class _MemoryCache:
    def __init__(self):
        self.values: dict[str, bytes] = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value):
        self.values[key] = value


async def test_predict_keys_the_cache_by_the_deployments_route(bare, monkeypatch, tmp_path):
    settings = _settings(monkeypatch)
    api = BibrPipelineAPI(upload_root=tmp_path, settings=settings)

    class _Pipeline:
        ocr_route = "text-layer"
        _config = _pipeline(settings)._config

        async def process_file(self, filename, **kwargs):  # noqa: ARG002
            return {"metadata": {}, "provenance": {}}

    api._pipeline = _Pipeline()
    api._cache = _MemoryCache()
    api._cache_inited = True
    api._inflight_sem = None

    await api.predict(
        {
            "filename": "paper.pdf",
            "content": b"same-pdf",
            "start_page": None,
            "end_page": None,
            "include_figures": False,
            "include_regions": False,
            "consolidate": None,
        }
    )

    assert all(":textlayer" in key for key in api._cache.values)


# --- the command line ---------------------------------------------------------------


def test_serve_ocr_text_layer_reaches_the_workers_through_the_environment(bare, monkeypatch):
    import bibr.config as config_mod
    import bibr.serve.app as app_mod

    settings = _settings(monkeypatch, GOOGLE_API_KEY="test-key")
    monkeypatch.setattr(config_mod, "Settings", settings)
    monkeypatch.setattr("sys.argv", ["bibr serve", "--ocr", "text-layer"])
    # main() writes OCR_BACKEND for the worker processes; undo it after the test.
    monkeypatch.setenv("OCR_BACKEND", "")

    seen: dict[str, object] = {}

    class _Server:
        def run(self, **kwargs):
            import os

            seen["env"] = os.environ.get("OCR_BACKEND")
            seen["route"] = serve_ocr_route(settings)

    monkeypatch.setattr(app_mod, "build_server", lambda: _Server())
    monkeypatch.setattr("bibr.serve.logsetup.configure_serve_logging", lambda *_a, **_k: None)
    monkeypatch.setattr("bibr.serve.logsetup.install_litserve_logging_hook", lambda: None)

    app_mod.main()

    assert seen == {"env": "text-layer", "route": "text-layer"}


# --- the worker ---------------------------------------------------------------------


def test_the_worker_opens_no_ocr_client_on_the_text_layer_route(bare, monkeypatch, tmp_path):
    from unittest import mock

    settings = _settings(monkeypatch, OCR_BACKEND="text-layer", OCR_BASE_URL="http://bibr-ocr:8080")
    api = BibrPipelineAPI(upload_root=tmp_path, settings=settings)
    with (
        mock.patch("bibr.serve.deployments.layout.LayoutDetector"),
        mock.patch("bibr.serve.deployments.segmenter.SentenceSegmenter"),
        mock.patch("bibr.serve.pipeline.ServePipeline") as pipeline_cls,
    ):
        api.setup(device="cpu")

    assert pipeline_cls.call_args.kwargs["http_client"] is None


def test_the_worker_still_refuses_plain_http_to_a_remote_ocr_server(bare, monkeypatch, tmp_path):
    from unittest import mock

    settings = _settings(monkeypatch, OCR_BASE_URL="http://bibr-ocr:8080")
    api = BibrPipelineAPI(upload_root=tmp_path, settings=settings)
    with (
        mock.patch("bibr.serve.deployments.layout.LayoutDetector"),
        mock.patch("bibr.serve.deployments.segmenter.SentenceSegmenter"),
        mock.patch("bibr.serve.pipeline.ServePipeline"),
        pytest.raises(ValueError, match="Refusing insecure HTTP"),
    ):
        api.setup(device="cpu")
