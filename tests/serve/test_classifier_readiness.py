from pathlib import Path


def test_serve_image_prefetches_both_classifier_revisions():
    dockerfile = Path("Dockerfile.serve").read_text()
    assert "PAPER_CLASSIFIER_MODEL" in dockerfile
    assert "PAPER_CLASSIFIER_REVISION" in dockerfile
    assert "SECTION_CLASSIFIER_MODEL" in dockerfile
    assert "SECTION_CLASSIFIER_REVISION" in dockerfile
    assert "snapshot_download" in dockerfile


def test_classifier_readiness_distinguishes_degraded_and_required_failure():
    from bibr.pipeline.classifier_resources import ClassifierState, ClassifierStatus
    from bibr.serve.app import classifier_readiness

    degraded = {
        "paper": ClassifierStatus(ClassifierState.DEGRADED, "cpu", "missing"),
        "section": ClassifierStatus(ClassifierState.READY, "cpu"),
    }
    required = {
        **degraded,
        "paper": ClassifierStatus(ClassifierState.FAILED_REQUIRED, "cpu", "missing"),
    }
    assert classifier_readiness(degraded) == ("degraded", True)
    assert classifier_readiness(required) == ("failed_required", False)


def test_artifact_readiness_checks_local_cache_only():
    from types import SimpleNamespace

    from bibr.serve.app import classifier_artifact_readiness

    calls = []
    settings = SimpleNamespace(
        ml=SimpleNamespace(
            paper_classifier_model_id="paper/model",
            paper_classifier_revision="paper-rev",
            section_classifier_model_id="section/model",
            section_classifier_revision="section-rev",
            classifiers_required=False,
        )
    )

    def cached(model_id, *, revision, local_files_only):
        calls.append((model_id, revision, local_files_only))
        return "/cache/model"

    assert classifier_artifact_readiness(settings, snapshot_download=cached) == ("ok", True)
    assert calls == [
        ("paper/model", "paper-rev", True),
        ("section/model", "section-rev", True),
    ]


def test_missing_artifact_is_degraded_or_required_failure():
    from types import SimpleNamespace

    from bibr.serve.app import classifier_artifact_readiness

    def missing(*args, **kwargs):
        raise FileNotFoundError("not cached")

    ml = SimpleNamespace(
        paper_classifier_model_id="paper/model",
        paper_classifier_revision="main",
        section_classifier_model_id=None,
        section_classifier_revision="main",
        classifiers_required=False,
    )
    assert classifier_artifact_readiness(SimpleNamespace(ml=ml), snapshot_download=missing) == (
        "degraded",
        True,
    )
    ml.classifiers_required = True
    assert classifier_artifact_readiness(SimpleNamespace(ml=ml), snapshot_download=missing) == (
        "failed_required",
        False,
    )


def test_required_classifier_failure_is_fatal_to_serve_setup():
    import pytest

    from bibr.pipeline.classifier_resources import ClassifierState, ClassifierStatus
    from bibr.serve.deployments.pipeline import _raise_for_required_classifier_failure

    statuses = {
        "paper": ClassifierStatus(ClassifierState.FAILED_REQUIRED, "cpu", "missing"),
        "section": ClassifierStatus(ClassifierState.READY, "cpu"),
    }
    with pytest.raises(RuntimeError, match="paper.*missing"):
        _raise_for_required_classifier_failure(statuses)


def test_readiness_payload_includes_exact_build_sha():
    from types import SimpleNamespace

    from bibr.serve.app import readiness_payload

    payload = readiness_payload(
        "ready",
        {"ocr": "ok", "classifiers": "ok"},
        SimpleNamespace(BIBR_BUILD_SHA="a" * 40),
    )

    assert payload["status"] == "ready"
    assert payload["build_sha"] == "a" * 40


def test_serve_image_preserves_build_sha_from_build_argument():
    dockerfile = Path("Dockerfile.serve").read_text()

    assert "ARG BIBR_BUILD_SHA" in dockerfile
    assert "ENV BIBR_BUILD_SHA=${BIBR_BUILD_SHA}" in dockerfile


def _local_classifier_dir(tmp_path):
    bundle = tmp_path / "paper-cls" / "onnx"
    bundle.mkdir(parents=True)
    (bundle / "model.onnx").write_bytes(b"fake")
    (bundle / "bibr_onnx.json").write_text('{"schema_version": 1, "model_file": "model.onnx"}')
    return tmp_path / "paper-cls"


def test_local_classifier_dir_counts_as_present(tmp_path):
    """A baked-in local-path classifier resolves as the loaders resolve it.

    snapshot_download rejects filesystem paths, so the pre-fix check reported
    a correctly loaded classifier as degraded / failed_required.
    """
    import os
    from types import SimpleNamespace

    from bibr.serve.app import classifier_artifact_readiness

    local = _local_classifier_dir(tmp_path)
    ml = SimpleNamespace(
        paper_classifier_model_id=str(local),
        paper_classifier_revision=None,
        section_classifier_model_id=None,
        section_classifier_revision=None,
        classifiers_required=True,
    )

    def _must_not_download(*args, **kwargs):
        raise AssertionError("local paths must not reach snapshot_download")

    old_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        assert classifier_artifact_readiness(
            SimpleNamespace(ml=ml), snapshot_download=_must_not_download
        ) == ("ok", True)
    finally:
        if old_offline is None:
            del os.environ["HF_HUB_OFFLINE"]
        else:
            os.environ["HF_HUB_OFFLINE"] = old_offline


def _torch_classifier_dir(tmp_path):
    """A torch-format baked-in classifier: safetensors weights, no onnx/ bundle."""
    local = tmp_path / "paper-cls-torch"
    local.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "label_maps.json", "tokenizer.json"):
        (local / name).write_text("{}")
    return local


def test_torch_format_local_dir_counts_as_present_when_torch_allowed(tmp_path, monkeypatch):
    """A baked-in torch-format directory loads via the torch runtime, so
    /ready must not demand an ONNX bundle unless ML_RUNTIME=onnx."""
    from types import SimpleNamespace

    import bibr.utils.ml_runtime as ml_runtime
    from bibr.serve.app import classifier_artifact_readiness

    monkeypatch.setattr(ml_runtime, "torch_available", lambda: True)
    local = _torch_classifier_dir(tmp_path)
    ml = SimpleNamespace(
        paper_classifier_model_id=str(local),
        paper_classifier_revision=None,
        section_classifier_model_id=None,
        section_classifier_revision=None,
        classifiers_required=True,
        runtime="torch",
    )

    def _must_not_download(*args, **kwargs):
        raise AssertionError("local paths must not reach snapshot_download")

    assert classifier_artifact_readiness(
        SimpleNamespace(ml=ml), snapshot_download=_must_not_download
    ) == ("ok", True)


def test_torch_format_local_dir_still_missing_when_onnx_required(tmp_path, monkeypatch):
    """ML_RUNTIME=onnx keeps requiring the ONNX bundle for a local directory."""
    from types import SimpleNamespace

    import bibr.utils.ml_runtime as ml_runtime
    from bibr.serve.app import classifier_artifact_readiness

    monkeypatch.setattr(ml_runtime, "torch_available", lambda: True)
    local = _torch_classifier_dir(tmp_path)
    ml = SimpleNamespace(
        paper_classifier_model_id=str(local),
        paper_classifier_revision=None,
        section_classifier_model_id=None,
        section_classifier_revision=None,
        classifiers_required=False,
        runtime="onnx",
    )

    def _must_not_download(*args, **kwargs):
        raise AssertionError("local paths must not reach snapshot_download")

    assert classifier_artifact_readiness(
        SimpleNamespace(ml=ml), snapshot_download=_must_not_download
    ) == ("degraded", True)


def _ready_client(monkeypatch, settings, snapshot_download):
    """Mount the real /ready route with a stubbed OCR server + Hub probe."""
    import httpx
    import huggingface_hub
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import bibr.serve.app as app_mod

    def _handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "glm-ocr"}]})
        return httpx.Response(404, json={})

    real_client = httpx.AsyncClient

    class _PinnedClient(real_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(_handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _PinnedClient)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    server = FastAPI()

    class _Server:
        app = server

    app_mod._register_readiness_route(_Server(), settings)
    return TestClient(server, raise_server_exceptions=False)


def test_ready_rechecks_a_failed_classifier_verdict(monkeypatch, tmp_path):
    """A probe that raced worker startup recovers without a restart."""
    from bibr.config import GlobalSettings
    from bibr.serve import app as app_mod

    calls = {"n": 0}

    def miss_then_hit(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise FileNotFoundError("cache cold (worker still downloading)")
        return "/fake/snapshot"

    monkeypatch.setattr(app_mod, "_CLASSIFIER_CHECK_RETRY_SECONDS", 0, raising=False)
    settings = GlobalSettings()
    settings.ml.paper_classifier_model_id = "scienceverse/bibr-paper-classifier"
    settings.ml.section_classifier_model_id = "scienceverse/bibr-section-classifier"
    settings.ml.classifiers_required = True
    client = _ready_client(monkeypatch, settings, miss_then_hit)

    first = client.get("/ready")
    assert first.status_code == 503
    assert first.json()["checks"]["classifiers"] == "failed_required"
    second = client.get("/ready")
    assert second.status_code == 200
    assert second.json()["checks"]["classifiers"] == "ok"
    assert calls["n"] == 3  # paper miss, then paper + section hits


def test_ready_caches_an_ok_classifier_verdict(monkeypatch):
    """A cached ok is never re-probed: no model load on the request path."""
    from bibr.config import GlobalSettings

    calls = {"n": 0}

    def hit(*args, **kwargs):
        calls["n"] += 1
        return "/fake/snapshot"

    settings = GlobalSettings()
    settings.ml.paper_classifier_model_id = "scienceverse/bibr-paper-classifier"
    settings.ml.section_classifier_model_id = "scienceverse/bibr-section-classifier"
    client = _ready_client(monkeypatch, settings, hit)

    assert client.get("/ready").status_code == 200
    assert client.get("/ready").status_code == 200
    assert calls["n"] == 2  # one evaluation for both classifiers, then cached


def test_ready_does_not_reprobe_a_failure_before_its_ttl(monkeypatch):
    """Failures are retried after a bounded interval, not on every probe."""
    from bibr.config import GlobalSettings

    calls = {"n": 0}

    def miss_then_hit(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise FileNotFoundError("cache cold (worker still downloading)")
        return "/fake/snapshot"

    settings = GlobalSettings()
    settings.ml.paper_classifier_model_id = "scienceverse/bibr-paper-classifier"
    settings.ml.section_classifier_model_id = "scienceverse/bibr-section-classifier"
    settings.ml.classifiers_required = True
    client = _ready_client(monkeypatch, settings, miss_then_hit)

    assert client.get("/ready").status_code == 503
    assert client.get("/ready").status_code == 503
    assert calls["n"] == 1
