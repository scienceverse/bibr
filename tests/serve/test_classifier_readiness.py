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
