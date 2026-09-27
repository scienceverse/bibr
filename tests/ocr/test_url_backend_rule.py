"""One ``ocr_url`` rule for the CLI, the library and the OCR runtime identity.

An OCR URL means Paddle, bibr's default OCR: a GLM request (``glm``,
``glm-*``) becomes ``glm-http``; ``paddle-http``, ``serve-http`` and an
explicitly requested cloud vision backend are kept; everything else, including
the ``paddle`` selector and no backend at all, becomes ``paddle-http``. The
request is the explicit backend, else a configured ``OCR_BACKEND`` unless it
names a cloud vision provider. Without a URL nothing changes.

``bibr chew --ocr-url`` used to pick ``paddle-http`` while
``LocalPipeline(ocr_url=...)`` (and so ``bibr.chew``) picked ``glm-http`` for
the same URL, and ``ResourceManager`` turned a vision backend plus a URL into
``glm-http`` while the runtime identity kept the vision backend. These tests
hold every entry point to the one table below, and the OCR cache identity to
the backend that actually starts.
"""

from types import SimpleNamespace

import pytest

from bibr.config import Settings, snapshot_settings
from bibr.local.cli import _build_parser, resolve_run_config
from bibr.ocr import registry
from bibr.ocr.profiles import resolve_ocr_runtime_identity
from bibr.pipeline.context import RunConfig

URL = "http://ocr.example:8000"

# (requested backend, backend with an OCR URL). ``None`` is no backend at all.
URL_RULE = [
    (None, "paddle-http"),
    ("paddle", "paddle-http"),
    ("paddle-vllm", "paddle-http"),
    ("paddle-rapid-mlx", "paddle-http"),
    ("paddle-mlx-vlm", "paddle-http"),
    ("paddle-http", "paddle-http"),
    ("glm", "glm-http"),
    ("glm-llama", "glm-http"),
    ("glm-rapid-mlx", "glm-http"),
    ("glm-http", "glm-http"),
    ("gemini", "gemini"),
    ("openai", "openai"),
    ("anthropic", "anthropic"),
]

# The same requests without a URL: the rule leaves them alone.
NO_URL = [
    (None, "paddle"),
    ("paddle", "paddle"),
    ("paddle-vllm", "paddle-vllm"),
    ("paddle-rapid-mlx", "paddle-rapid-mlx"),
    ("paddle-mlx-vlm", "paddle-mlx-vlm"),
    ("paddle-http", "paddle-http"),
    ("glm", "glm-llama"),
    ("glm-llama", "glm-llama"),
    ("glm-rapid-mlx", "glm-rapid-mlx"),
    ("glm-http", "glm-http"),
    ("gemini", "gemini"),
    ("openai", "openai"),
    ("anthropic", "anthropic"),
]

TABLE = [(requested, URL, expected) for requested, expected in URL_RULE] + [
    (requested, None, expected) for requested, expected in NO_URL
]


def _table_id(row) -> str:
    requested, url, _expected = row
    return f"{requested or 'unset'}-{'url' if url else 'no-url'}"


@pytest.fixture(autouse=True)
def _linux_without_gpu(monkeypatch):
    """Pin the platform so ``glm`` and the ``paddle`` chain resolve the same everywhere."""
    monkeypatch.setattr(registry, "_cuda_vram_gb", lambda: None)
    monkeypatch.setattr(registry.sys, "platform", "linux")
    monkeypatch.setattr(registry.platform, "machine", lambda: "x86_64")


def _cli_backend(requested: str | None, url: str | None) -> str:
    argv = ["chew", "paper.pdf", "--memory", "balanced"]
    if requested is not None:
        argv += ["--ocr", requested]
    if url is not None:
        argv += ["--ocr-url", url]
    return resolve_run_config(_build_parser().parse_args(argv)).ocr_backend


def _pipeline(requested: str | None, url: str | None):
    from bibr.local.pipeline import LocalPipeline

    return LocalPipeline(ocr_backend=requested, ocr_url=url)


class _Captured:
    """Stand-in for ``registry.create``: records what would start."""

    def __init__(self):
        self.name = None
        self.kwargs = None

    def __call__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
        return SimpleNamespace(loaded=True)


async def _start(rm, monkeypatch) -> _Captured:
    captured = _Captured()
    monkeypatch.setattr(registry, "create", captured)
    await rm.await_ocr()
    return captured


def _assert_identity_is_what_started(identity, rm, captured) -> None:
    # The cache key and the export provenance describe the client that runs.
    assert identity == rm.ocr_runtime_identity
    assert captured.name == identity.backend
    if "profile" in captured.kwargs:  # HTTP clients receive the resolved request profile
        assert captured.kwargs["model"] == identity.model
        assert captured.kwargs["profile"].name == identity.profile


def test_resolve_url_backend_is_the_table():
    for requested, expected in URL_RULE:
        assert registry.resolve_url_backend(requested, URL) == expected, requested
    for requested, _expected in NO_URL:
        assert registry.resolve_url_backend(requested, None) == requested


def test_the_rule_is_idempotent():
    # The CLI hands its resolved backend to LocalPipeline, which hands it to
    # ResourceManager: applying the rule again must not move it.
    for _requested, expected in URL_RULE:
        assert registry.resolve_url_backend(expected, URL) == expected
    assert registry.resolve_url_backend("serve-http", URL) == "serve-http"


@pytest.mark.parametrize(("requested", "url", "expected"), TABLE, ids=map(_table_id, TABLE))
def test_cli_library_and_runtime_identity_agree(requested, url, expected):
    pipeline = _pipeline(requested, url)

    assert _cli_backend(requested, url) == expected
    assert pipeline._config.ocr_backend == expected
    assert pipeline._resources.ocr_backend == expected
    if url is not None:
        # The identity resolved from the raw request (an embedder's RunConfig)
        # names the same backend as the resolved pipeline.
        raw = resolve_ocr_runtime_identity(
            RunConfig(ocr_backend=requested, ocr_url=url), snapshot_settings()
        )
        assert raw.backend == expected


@pytest.mark.parametrize(
    ("requested", "expected"), URL_RULE, ids=[r or "unset" for r, _ in URL_RULE]
)
async def test_library_cache_identity_is_the_backend_that_starts(requested, expected, monkeypatch):
    pipeline = _pipeline(requested, URL)
    identity = resolve_ocr_runtime_identity(pipeline._config, pipeline._settings)
    assert identity.backend == expected

    rm = pipeline._resources
    captured = await _start(rm, monkeypatch)

    _assert_identity_is_what_started(identity, rm, captured)
    # A cloud vision backend never reads the URL, so the pipeline drops it.
    assert captured.kwargs["base_url"] == (
        None if expected in registry.CLOUD_VISION_OCR_BACKENDS else URL
    )


@pytest.mark.parametrize(
    ("requested", "expected"), URL_RULE, ids=[r or "unset" for r, _ in URL_RULE]
)
async def test_resource_manager_starts_the_rule_backend_for_a_raw_request(
    requested, expected, monkeypatch
):
    """Embedders building ``ResourceManager`` and ``RunConfig`` directly get the same answer."""
    from bibr.pipeline.resources import ResourceManager

    rm = ResourceManager(ocr_backend=requested, ocr_url=URL)
    assert [c.backend for c in rm._resolve_ocr_candidates()] == [expected]
    captured = await _start(rm, monkeypatch)

    identity = resolve_ocr_runtime_identity(
        RunConfig(ocr_backend=requested, ocr_url=URL), rm._settings
    )
    _assert_identity_is_what_started(identity, rm, captured)


def test_a_configured_backend_is_the_request_for_a_bare_url(monkeypatch):
    # OCR_BACKEND=glm-http in .env plus --ocr-url: every entry point keeps GLM.
    monkeypatch.setattr(Settings.ocr, "backend", "glm-http")
    pipeline = _pipeline(None, URL)

    assert _cli_backend(None, URL) == "glm-http"
    assert pipeline._config.ocr_backend == "glm-http"
    assert resolve_ocr_runtime_identity(pipeline._config, pipeline._settings).backend == "glm-http"
    raw = RunConfig(ocr_backend=None, ocr_url=URL)
    assert resolve_ocr_runtime_identity(raw, snapshot_settings()).backend == "glm-http"


def test_url_request_prefers_the_explicit_backend_over_a_configured_cloud_default():
    assert registry.url_request("glm", "gemini") == "glm"
    assert registry.url_request("gemini", "paddle") == "gemini"
    assert registry.url_request(None, "glm-http") == "glm-http"
    assert registry.url_request(None, None) is None
    for cloud in sorted(registry.CLOUD_VISION_OCR_BACKENDS):
        assert registry.url_request(None, cloud) is None
        assert registry.url_request(cloud, cloud) == cloud


CLOUD = sorted(registry.CLOUD_VISION_OCR_BACKENDS)


@pytest.mark.parametrize("cloud", CLOUD)
async def test_a_configured_cloud_backend_never_replaces_an_explicit_url(cloud, monkeypatch):
    # OCR_BACKEND=gemini (the cloud setup tier and preset) plus --ocr-url: the
    # named server is used, and no page image goes to the cloud provider.
    from bibr.pipeline.resources import ResourceManager

    monkeypatch.setattr(Settings.ocr, "backend", cloud)
    pipeline = _pipeline(None, URL)

    assert _cli_backend(None, URL) == "paddle-http"
    assert pipeline._config.ocr_backend == "paddle-http"
    assert pipeline._config.ocr_url == URL
    identity = resolve_ocr_runtime_identity(pipeline._config, pipeline._settings)
    assert identity.backend == "paddle-http"
    # ``ocr_backend=None`` is how an embedder's RunConfig or ResourceManager
    # asks for the configured OCR_BACKEND (both default to ``paddle``).
    raw = resolve_ocr_runtime_identity(
        RunConfig(ocr_backend=None, ocr_url=URL), snapshot_settings()
    )
    assert raw.backend == "paddle-http"
    assert raw.profile == "paddle"

    captured = await _start(pipeline._resources, monkeypatch)
    _assert_identity_is_what_started(identity, pipeline._resources, captured)
    assert captured.kwargs["base_url"] == URL

    rm = ResourceManager(ocr_backend=None, ocr_url=URL)
    assert [c.backend for c in rm._resolve_ocr_candidates()] == ["paddle-http"]
    captured = await _start(rm, monkeypatch)
    _assert_identity_is_what_started(raw, rm, captured)


@pytest.mark.parametrize("cloud", CLOUD)
def test_an_explicit_cloud_backend_keeps_it_despite_the_url(cloud, monkeypatch):
    from bibr.pipeline.resources import ResourceManager

    monkeypatch.setattr(Settings.ocr, "backend", cloud)

    assert _cli_backend(cloud, URL) == cloud
    assert _pipeline(cloud, URL)._config.ocr_backend == cloud
    raw = RunConfig(ocr_backend=cloud, ocr_url=URL)
    assert resolve_ocr_runtime_identity(raw, snapshot_settings()).backend == cloud
    rm = ResourceManager(ocr_backend=cloud, ocr_url=URL)
    assert [c.backend for c in rm._resolve_ocr_candidates()] == [cloud]


def test_a_vision_backend_warns_that_it_ignores_the_url(caplog):
    with caplog.at_level("WARNING", logger="bibr.local.pipeline"):
        pipeline = _pipeline("gemini", URL)

    assert pipeline._config.ocr_backend == "gemini"
    assert "ignored" in caplog.text
    assert "OCR_VISION_BASE_URL" in caplog.text
    # The unused URL is dropped, so it neither keys the OCR cache
    # (``cfg.ocr_url`` is part of the key) nor reaches the client.
    assert pipeline._config.ocr_url is None
    assert pipeline._resources.ocr_url is None
    assert pipeline.ocr_url is None
