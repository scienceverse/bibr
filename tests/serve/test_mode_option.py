"""Per-request ``mode`` on the serve API: fast (no LLM call) or full.

``mode`` is validated at ingress, before an upload is stored or a job exists, so
``mode=full`` on a server with no LLM is a 400 and not a failed job. The decode →
predict → RunConfig chain carries it to the pipeline, and the response cache must
never hand a full request a cached fast result (or vice versa) — while a full-mode
key stays exactly what it was before the option existed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("litserve")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from bibr.config import GlobalSettings  # noqa: E402
from bibr.extract.ref_extractor import _resolve_ref_strategies  # noqa: E402
from bibr.pipeline.context import RunConfig  # noqa: E402
from bibr.serve.deployments.pipeline import BibrPipelineAPI  # noqa: E402
from bibr.serve.ingress import InvalidUploadOptionError, _validate_upload_options  # noqa: E402

_LLM_ENV = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_BACKEND",
    "PIPELINE_MODE",
)


@pytest.fixture
def keyless(monkeypatch):
    """Settings for a server with no LLM configured, whatever the test environment holds."""
    for var in _LLM_ENV:
        monkeypatch.delenv(var, raising=False)
    settings = GlobalSettings()
    monkeypatch.setattr("bibr.config.Settings", settings)
    return settings


@pytest.fixture
def keyed(keyless):
    keyless.GOOGLE_API_KEY = "test-key"
    return keyless


# --- ingress validation --------------------------------------------------------


@pytest.mark.parametrize("value", ["auto", "fast", "full", "FAST", "Full"])
def test_validate_upload_options_accepts_each_mode(keyed, value):
    assert _validate_upload_options({"mode": value}) == {"mode": value}


def test_validate_upload_options_rejects_an_unknown_mode(keyed):
    with pytest.raises(InvalidUploadOptionError, match="mode must be one of: auto, fast, full"):
        _validate_upload_options({"mode": "turbo"})


def test_validate_upload_options_drops_an_empty_mode(keyed):
    assert "mode" not in _validate_upload_options({"mode": ""})


def test_mode_full_with_no_llm_is_refused_and_names_the_way_out(keyless):
    with pytest.raises(InvalidUploadOptionError) as excinfo:
        _validate_upload_options({"mode": "full"})
    message = str(excinfo.value)
    assert "mode=full needs an LLM" in message
    assert "mode=fast" in message


def test_mode_full_with_an_llm_is_accepted(keyed):
    assert _validate_upload_options({"mode": "full", "refs": "llm"}) == {
        "mode": "full",
        "refs": "llm",
    }


@pytest.mark.parametrize("mode", ["fast", "auto", ""])
def test_mode_fast_or_auto_with_no_llm_is_accepted(keyless, mode):
    assert _validate_upload_options({"mode": mode, "refs": "ner"}).get("refs") == "ner"


@pytest.mark.parametrize("option", [{"refs": "llm"}, {"refs": "ref-llm"}, {"ref_seg": "llm"}])
def test_an_llm_option_beside_mode_fast_is_refused(keyed, option):
    with pytest.raises(InvalidUploadOptionError, match="fast mode never does"):
        _validate_upload_options({"mode": "fast", **option})


def test_an_llm_option_on_a_fast_server_is_refused_unless_the_request_says_full(keyless):
    """With no LLM configured the server's own mode is fast, so the option has nothing to call."""
    with pytest.raises(InvalidUploadOptionError, match="refs=llm calls an LLM"):
        _validate_upload_options({"refs": "llm"})
    with pytest.raises(InvalidUploadOptionError, match="refs=llm calls an LLM"):
        _validate_upload_options({"mode": "auto", "refs": "llm"})


def test_an_llm_option_on_a_server_pinned_to_fast_is_refused_even_with_a_key(keyed):
    keyed.pipeline.mode = "fast"
    with pytest.raises(InvalidUploadOptionError, match="ref_seg=llm calls an LLM"):
        _validate_upload_options({"ref_seg": "llm"})
    # The request can still ask for the full pipeline, which has the LLM.
    assert _validate_upload_options({"mode": "full", "ref_seg": "llm"})["mode"] == "full"


def test_an_llm_option_on_a_full_server_is_unchanged(keyed):
    assert _validate_upload_options({"refs": "llm", "ref_seg": "llm"}) == {
        "refs": "llm",
        "ref_seg": "llm",
    }


# --- the HTTP routes refuse before anything is stored ---------------------------


class _CountingTracker:
    def __init__(self):
        self.submitted: list[dict] = []

    async def submit(self, descriptor, request_state=None, **_kwargs):  # noqa: ARG002
        self.submitted.append(descriptor)
        return {"ok": True}

    async def discard(self, descriptor):
        pass


def _extract_client(tracker):
    from bibr.serve.ingress import UploadStore, register_extract_route

    app = FastAPI()
    store = UploadStore.create(max_size=1000, spool_memory_bytes=4, stale_after_seconds=120)
    register_extract_route(app, store, tracker)
    return TestClient(app), store


def _post_paper(client, path, **data):
    return client.post(
        path,
        files={"file": ("paper.pdf", b"%PDF", "application/pdf")},
        data=data,
    )


def test_extract_refuses_mode_full_with_no_llm_before_dispatch(keyless):
    tracker = _CountingTracker()
    client, store = _extract_client(tracker)
    try:
        response = _post_paper(client, "/papers/extract", mode="full")
        assert response.status_code == 400, response.text
        assert "mode=full needs an LLM" in response.text
        assert tracker.submitted == []
        assert list(store.root.iterdir()) == []
    finally:
        asyncio.run(store.close())


def test_extract_passes_mode_to_the_descriptor(keyless):
    tracker = _CountingTracker()
    client, store = _extract_client(tracker)
    try:
        response = _post_paper(client, "/papers/extract", mode="fast")
        assert response.status_code == 200, response.text
        (descriptor,) = tracker.submitted
        assert descriptor["mode"] == "fast"
    finally:
        asyncio.run(store.close())


def test_extract_rejects_an_unknown_mode_with_400(keyed):
    tracker = _CountingTracker()
    client, store = _extract_client(tracker)
    try:
        response = _post_paper(client, "/papers/extract", mode="turbo")
        assert response.status_code == 400
        assert tracker.submitted == []
    finally:
        asyncio.run(store.close())


def test_job_route_refuses_mode_full_with_no_llm_before_a_job_exists(keyless):
    from bibr.serve.ingress import UploadStore
    from bibr.serve.jobs import MemoryJobStore, register_job_routes

    class _Store(MemoryJobStore):
        created = 0

        async def create(self, *, filename):
            type(self).created += 1
            return await super().create(filename=filename)

    tracker = _CountingTracker()
    app = FastAPI()
    upload_store = UploadStore.create(max_size=1000, spool_memory_bytes=4, stale_after_seconds=120)
    job_store = _Store()
    register_job_routes(app, store=job_store, upload_store=upload_store, tracker=tracker)
    try:
        response = _post_paper(TestClient(app), "/papers/jobs", mode="full")
        assert response.status_code == 400, response.text
        assert "mode=full needs an LLM" in response.text
        assert _Store.created == 0
        assert tracker.submitted == []
    finally:
        asyncio.run(upload_store.close())


def test_job_route_accepts_mode_fast_with_no_llm(keyless):
    from bibr.serve.ingress import UploadStore
    from bibr.serve.jobs import MemoryJobStore, register_job_routes

    tracker = _CountingTracker()
    app = FastAPI()
    upload_store = UploadStore.create(max_size=1000, spool_memory_bytes=4, stale_after_seconds=120)
    register_job_routes(app, store=MemoryJobStore(), upload_store=upload_store, tracker=tracker)
    try:
        response = _post_paper(TestClient(app), "/papers/jobs", mode="fast")
        assert response.status_code == 202, response.text
    finally:
        asyncio.run(upload_store.close())


def test_openapi_documents_the_mode_field():
    from bibr.serve.ingress import MULTIPART_OPENAPI_EXTRA

    schema = MULTIPART_OPENAPI_EXTRA["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert schema["properties"]["mode"]["enum"] == ["auto", "fast", "full"]


# --- the deployment: decode, predict, RunConfig ---------------------------------


class _MemoryCache:
    def __init__(self):
        self.values: dict[str, bytes] = {}
        self.set_calls = 0

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def set(self, key: str, value: bytes) -> None:
        self.set_calls += 1
        self.values[key] = value

    async def delete(self, key: str) -> None:
        self.values.pop(key, None)


class _RecordingPipeline:
    def __init__(self, deployment_mode: str | None = None):
        self._config = RunConfig(mode=deployment_mode)
        self.configs: list[RunConfig] = []

    async def process_file(self, filename, paper_id, content, config):  # noqa: ARG002
        self.configs.append(config)
        await asyncio.sleep(0)
        return {"paper_id": paper_id, "info": {"file_name": filename}}


def _api(tmp_path, *, deployment_mode: str | None = "full", llm: bool = True):
    settings = GlobalSettings()
    if llm:
        settings.GOOGLE_API_KEY = "test-key"
    api = BibrPipelineAPI(upload_root=tmp_path, settings=settings)
    pipeline = _RecordingPipeline(deployment_mode)
    cache = _MemoryCache()
    api._pipeline = pipeline
    api._cache = cache
    api._cache_inited = True
    api._inflight_sem = None
    return api, pipeline, cache


def _inputs(mode: str | None = None, content: bytes = b"same-pdf") -> dict:
    inputs = {
        "filename": "paper.pdf",
        "content": content,
        "start_page": None,
        "end_page": None,
        "include_figures": False,
        "include_regions": False,
        "consolidate": None,
    }
    if mode is not None:
        inputs["mode"] = mode
    return inputs


def _descriptor(**extra) -> dict:
    return {"upload_id": "u1", "filename": "paper.pdf", "size": 13, "sha256": "ab" * 32, **extra}


@pytest.fixture
def decoding(tmp_path, monkeypatch):
    import bibr.serve.deployments.pipeline as mod

    monkeypatch.setattr(
        mod, "consume_upload_descriptor", lambda *_a, **_k: (b"%PDF-1.4 stub", "ab" * 32)
    )
    return BibrPipelineAPI(upload_root=tmp_path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("fast", "fast"),
        ("FULL", "full"),
        ("auto", None),
        ("", None),
        (None, None),
    ],
)
async def test_decode_request_parses_mode(decoding, raw, expected):
    descriptor = _descriptor() if raw is None else _descriptor(mode=raw)

    decoded = await decoding.decode_request(descriptor)

    assert decoded["mode"] == expected


async def test_decode_request_rejects_an_unknown_mode(decoding):
    with pytest.raises(HTTPException) as excinfo:
        await decoding.decode_request(_descriptor(mode="turbo"))
    assert excinfo.value.status_code == 400
    assert "mode must be one of: auto, fast, full" in excinfo.value.detail


async def test_request_mode_overrides_the_deployment_run_config(tmp_path):
    api, pipeline, _ = _api(tmp_path, deployment_mode="full")

    await api.predict(_inputs(mode="fast"))
    await api.predict(_inputs(content=b"other"))

    forced_fast, absent = pipeline.configs
    assert forced_fast.mode == "fast"
    # Absent keeps the deployment's mode.
    assert absent.mode == "full"


async def test_a_full_request_overrides_a_fast_deployment(tmp_path):
    api, pipeline, _ = _api(tmp_path, deployment_mode="fast")

    await api.predict(_inputs(mode="full"))

    (config,) = pipeline.configs
    assert config.mode == "full"


def test_effective_mode_resolution_order(tmp_path, monkeypatch):
    for var in _LLM_ENV:
        monkeypatch.delenv(var, raising=False)
    api, _, _ = _api(tmp_path, deployment_mode=None, llm=False)
    # No request value, no deployment mode: what the settings resolve to.
    assert api._effective_mode(None) == "fast"
    api._settings.GOOGLE_API_KEY = "test-key"
    assert api._effective_mode(None) == "full"
    # The deployment's mode beats the settings.
    api._pipeline._config = dataclasses.replace(api._pipeline._config, mode="fast")
    assert api._effective_mode(None) == "fast"
    # The request beats both.
    assert api._effective_mode("full") == "full"
    api._pipeline._config = dataclasses.replace(api._pipeline._config, mode="full")
    assert api._effective_mode("fast") == "fast"


# --- cache key separation -----------------------------------------------------


def _key(api: BibrPipelineAPI, inputs: dict, **kwargs) -> str:
    digest = hashlib.sha256(inputs["content"]).hexdigest()
    ref_seg, refs = _resolve_ref_strategies(None, None)
    return api._cache_key(
        digest,
        None,
        None,
        False,
        False,
        None,
        refs=refs,
        ref_seg=ref_seg,
        input_format=".pdf",
        **kwargs,
    )


def test_cache_key_adds_fast_only_for_a_fast_run(tmp_path):
    api, _, _ = _api(tmp_path)
    inputs = _inputs()

    full_key = _key(api, inputs, mode="full")
    assert full_key == _key(api, inputs)  # full is what a key was before the option
    assert full_key == _key(api, inputs, mode=None)
    assert ":fast" not in full_key
    assert _key(api, inputs, mode="fast") != full_key
    # The suffix sits with the other options, before the input format.
    assert _key(api, inputs, mode="fast").endswith(":fast:fmt:pdf")
    assert _key(api, inputs, mode="fast", crossref=True).endswith(":enrich:fast:fmt:pdf")


async def test_a_fast_result_never_answers_a_full_request(tmp_path):
    api, pipeline, cache = _api(tmp_path, deployment_mode="full")

    await api.predict(_inputs(mode="fast"))
    await api.predict(_inputs())  # full: must miss the cache and run again

    assert [c.mode for c in pipeline.configs] == ["fast", "full"]
    assert set(cache.values) == {
        _key(api, _inputs(), mode="fast"),
        _key(api, _inputs(), mode="full"),
    }


async def test_a_full_result_never_answers_a_fast_request(tmp_path):
    api, pipeline, cache = _api(tmp_path, deployment_mode="full")

    await api.predict(_inputs())
    await api.predict(_inputs(mode="fast"))

    assert [c.mode for c in pipeline.configs] == ["full", "fast"]
    assert cache.set_calls == 2


async def test_naming_the_deployments_own_mode_shares_the_cache_entry(tmp_path):
    api, pipeline, cache = _api(tmp_path, deployment_mode="fast")

    await api.predict(_inputs())
    await api.predict(_inputs(mode="fast"))

    assert len(pipeline.configs) == 1
    assert cache.set_calls == 1
    assert set(cache.values) == {_key(api, _inputs(), mode="fast")}


async def test_a_full_deployment_keeps_the_key_it_always_had(tmp_path):
    api, _, cache = _api(tmp_path, deployment_mode="full")

    await api.predict(_inputs())

    (key,) = cache.values
    assert key.startswith("json:") and ":fast" not in key
    assert key.endswith(":fmt:pdf")


# --- the deployment's own mode: ServePipeline ---------------------------------------


def _serve_pipeline(settings, **kwargs):
    from unittest.mock import MagicMock

    from bibr.serve.pipeline import ServePipeline

    settings.ror.enrich = False
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


def test_a_serve_pipeline_with_no_llm_runs_fast(keyless):
    assert _serve_pipeline(keyless)._config.mode == "fast"


def test_a_serve_pipeline_with_a_key_runs_full(keyed):
    assert _serve_pipeline(keyed)._config.mode == "full"


def test_a_serve_pipeline_follows_the_pipeline_mode_setting(keyed):
    keyed.pipeline.mode = "fast"
    assert _serve_pipeline(keyed)._config.mode == "fast"


def test_an_explicit_serve_pipeline_mode_beats_the_setting(keyed):
    assert _serve_pipeline(keyed, mode="fast")._config.mode == "fast"


# --- readiness: the mode, and the tagger a fast deployment needs ----------------


def test_readiness_payload_reports_the_mode_only_in_the_detail(keyed):
    from bibr.serve.app import readiness_payload

    checks = {"ocr": "ok"}
    detailed = readiness_payload("ready", checks, keyed, include_detail=True, mode="fast")
    assert detailed["mode"] == "fast"
    # An anonymous probe learns the status and nothing else.
    assert readiness_payload("ready", checks, keyed, include_detail=False, mode="fast") == {
        "status": "ready"
    }
    # Callers that name no mode (older embedders) get the body they always did.
    assert "mode" not in readiness_payload("ready", checks, keyed)


def _ready_response(monkeypatch, settings, *, authenticated: bool):
    from fastapi import Response

    import bibr.serve.app as app_mod

    seen: list[bool] = []

    def _artifacts(_settings, *, fast=False, **_kwargs):
        seen.append(fast)
        return ("ok", True)

    monkeypatch.setattr(app_mod, "classifier_artifact_readiness", _artifacts)
    monkeypatch.setattr(
        "bibr.serve.auth.check_bearer",
        lambda _header: None if authenticated else Response(status_code=401),
    )
    from tests.serve.test_readiness import _ocr_transport, _ready_client

    response = _ready_client(monkeypatch, settings, _ocr_transport()).get("/ready")
    return response, seen


@pytest.mark.parametrize(("fixture", "mode"), [("keyless", "fast"), ("keyed", "full")])
def test_ready_names_the_mode_to_an_authenticated_caller(request, monkeypatch, fixture, mode):
    settings = request.getfixturevalue(fixture)

    response, _ = _ready_response(monkeypatch, settings, authenticated=True)

    assert response.status_code == 200
    assert response.json()["mode"] == mode


def test_ready_hides_the_mode_from_an_anonymous_caller(monkeypatch, keyless):
    response, _ = _ready_response(monkeypatch, keyless, authenticated=False)

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@pytest.mark.parametrize(("fixture", "fast"), [("keyless", True), ("keyed", False)])
def test_ready_asks_for_the_tagger_only_of_a_fast_deployment(request, monkeypatch, fixture, fast):
    settings = request.getfixturevalue(fixture)

    _, seen = _ready_response(monkeypatch, settings, authenticated=True)

    assert seen == [fast]


def _ml_settings(**overrides):
    from types import SimpleNamespace

    ml = SimpleNamespace(
        paper_classifier_model_id="paper/model",
        paper_classifier_revision="paper-rev",
        section_classifier_model_id="section/model",
        section_classifier_revision="section-rev",
        front_tagger_model_id="tagger/model",
        front_tagger_revision="tagger-rev",
        classifiers_required=False,
    )
    for name, value in overrides.items():
        setattr(ml, name, value)
    return SimpleNamespace(ml=ml)


def test_a_fast_deployment_checks_the_tagger_in_the_local_cache():
    from bibr.serve.app import classifier_artifact_readiness

    calls = []

    def cached(model_id, *, revision, local_files_only):
        calls.append((model_id, revision, local_files_only))
        return "/cache/model"

    assert classifier_artifact_readiness(_ml_settings(), snapshot_download=cached, fast=True) == (
        "ok",
        True,
    )
    assert calls == [
        ("paper/model", "paper-rev", True),
        ("section/model", "section-rev", True),
        ("tagger/model", "tagger-rev", True),
    ]


def test_a_full_deployment_never_asks_for_the_tagger():
    from bibr.serve.app import classifier_artifact_readiness

    calls = []

    def cached(model_id, *, revision, local_files_only):  # noqa: ARG001
        calls.append(model_id)
        return "/cache/model"

    classifier_artifact_readiness(_ml_settings(), snapshot_download=cached)
    assert "tagger/model" not in calls


@pytest.mark.parametrize(("required", "expected"), [(False, "degraded"), (True, "failed_required")])
def test_a_missing_tagger_is_degraded_or_a_required_failure(required, expected):
    from bibr.serve.app import classifier_artifact_readiness

    def cached_but_the_tagger(model_id, *, revision, local_files_only):  # noqa: ARG001
        if model_id == "tagger/model":
            raise FileNotFoundError("not cached")
        return "/cache/model"

    settings = _ml_settings(classifiers_required=required)
    status, ok = classifier_artifact_readiness(
        settings, snapshot_download=cached_but_the_tagger, fast=True
    )
    assert status == expected
    assert ok is (not required)
    # The same cache reads ready to a full deployment, which never loads the tagger.
    assert classifier_artifact_readiness(
        settings, snapshot_download=cached_but_the_tagger, fast=False
    ) == ("ok", True)


def _onnx_bundle(root, *, files=("model.onnx", "bibr_onnx.json")):
    bundle = root / "onnx"
    bundle.mkdir(parents=True)
    for name in files:
        (bundle / name).write_bytes(b"x")
    return root


def _local_only_settings(tagger_dir):
    """A deployment whose classifiers are all absent except a local tagger directory."""
    return _ml_settings(
        paper_classifier_model_id=None,
        section_classifier_model_id=None,
        front_tagger_model_id=str(tagger_dir),
        front_tagger_revision=None,
    )


def test_a_local_tagger_directory_needs_an_onnx_bundle(tmp_path):
    from bibr.serve.app import classifier_artifact_readiness

    with_bundle = _onnx_bundle(tmp_path / "with")
    without = tmp_path / "without"
    without.mkdir()
    (without / "config.json").write_text("{}")  # a torch checkpoint is no tagger
    partial = _onnx_bundle(tmp_path / "partial", files=("model.onnx",))

    def never(*_a, **_k):
        raise AssertionError("a local path is never looked up on the Hub")

    def check(path):
        return classifier_artifact_readiness(
            _local_only_settings(path), snapshot_download=never, fast=True
        )

    assert check(with_bundle) == ("ok", True)
    assert check(without) == ("degraded", True)
    assert check(partial) == ("degraded", True)


# --- starting the server in a mode ---------------------------------------------------


def _run_main(monkeypatch, settings, argv):
    """Run ``bibr serve``'s entry point with the server itself stubbed out."""
    import sys

    import bibr.serve.app as app_mod

    ran: list[dict] = []

    class _Server:
        def run(self, **kwargs):
            ran.append(kwargs)

    monkeypatch.setattr(sys, "argv", ["bibr serve", *argv])
    monkeypatch.setattr("bibr.config.Settings", settings)
    monkeypatch.setattr(app_mod, "build_server", lambda: _Server())
    monkeypatch.setattr("bibr.serve.auth.validate_bind_auth", lambda *_a, **_k: None)
    monkeypatch.setattr("bibr.serve.logsetup.configure_serve_logging", lambda *_a, **_k: None)
    monkeypatch.setattr("bibr.serve.logsetup.install_litserve_logging_hook", lambda: None)
    app_mod.main()
    return ran


@pytest.mark.parametrize("flag", ["fast", "full"])
def test_serve_flags_set_the_mode_for_this_process_and_its_workers(monkeypatch, keyed, flag):
    import os

    monkeypatch.setenv("PIPELINE_MODE", "auto")

    ran = _run_main(monkeypatch, keyed, [f"--{flag}"])

    assert len(ran) == 1
    # The workers are fresh interpreters: they read the mode from the environment.
    assert os.environ["PIPELINE_MODE"] == flag
    assert keyed.pipeline.mode == flag


def test_serve_without_a_flag_leaves_the_mode_alone(monkeypatch, keyed):
    import os

    monkeypatch.setenv("PIPELINE_MODE", "auto")

    _run_main(monkeypatch, keyed, [])

    assert os.environ["PIPELINE_MODE"] == "auto"
    assert keyed.pipeline.mode == "auto"


def test_serve_fast_and_full_are_mutually_exclusive(monkeypatch, keyed, capsys):
    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, keyed, ["--fast", "--full"])

    assert excinfo.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_serve_refuses_to_start_in_full_mode_with_no_llm(monkeypatch, keyless):
    from bibr.exceptions import ConfigurationError
    from bibr.serve.app import build_server

    keyless.pipeline.mode = "full"
    monkeypatch.setattr("bibr.config.validate_production_settings", lambda _settings: None)

    with pytest.raises(ConfigurationError) as excinfo:
        build_server()

    message = str(excinfo.value)
    assert "full mode" in message
    assert "bibr serve --fast" in message
