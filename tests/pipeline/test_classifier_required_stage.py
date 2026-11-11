"""Local enforcement of ML_CLASSIFIERS_REQUIRED (pipeline-core-7).

A required classifier that fails to load must fail the chunk's files with a
dedicated code and a message naming the setting and its opt-out — not fall
back to the LLM tier silently.
"""

import asyncio
from types import SimpleNamespace

from bibr.pipeline.classifier_resources import ClassifierResources
from bibr.pipeline.stages.classifiers import ClassifierStage


def _shipped(name):
    from bibr.config import MlOptions

    return MlOptions.model_fields[name].default


def _settings(*, required):
    # The shipped models: an overridden one is required whatever the setting says.
    ml = SimpleNamespace(
        paper_classifier_model_id=_shipped("paper_classifier_model_id"),
        paper_classifier_revision=_shipped("paper_classifier_revision"),
        paper_classifier_device=None,
        paper_classifier_batch_size=8,
        paper_classifier_batch_timeout_ms=2.0,
        paper_classifier_estimated_peak_mb=1,
        section_classifier_model_id=_shipped("section_classifier_model_id"),
        section_classifier_revision=_shipped("section_classifier_revision"),
        section_classifier_device=None,
        section_classifier_batch_size=8,
        section_classifier_batch_timeout_ms=2.0,
        section_classifier_estimated_peak_mb=1,
        classifier_vram_safety_reserve_mb=1,
        classifiers_required=required,
    )
    return SimpleNamespace(ml=ml)


class _File:
    def __init__(self):
        self.error = None
        self.error_code = None
        self.failed_stage = None
        self.error_outage = False

    def set_error(self, message, *, code=None, stage=None, exc=None, outage=False):
        self.error = message
        self.error_code = code
        self.failed_stage = stage
        self.error_outage = outage


def _run(*, required, loader):
    resources = ClassifierResources(
        _settings(required=required),
        memory_mode="balanced",
        managed_vllm_fraction=0.0,
        cuda_available=False,
        paper_loader=loader,
        section_loader=loader,
    )
    files = [_File()]
    ctx = SimpleNamespace(
        resources=SimpleNamespace(start_classifiers=resources.start, classifiers=resources),
        file_states=files,
    )
    asyncio.run(ClassifierStage().run(ctx))
    return resources, files


def _boom(model_id, revision, device):
    raise OSError("classifier repo unreachable")


def test_required_failure_fails_files_with_dedicated_code():
    resources, files = _run(required=True, loader=_boom)
    states = resources.status()
    assert states["paper"].state.value == "failed_required"
    assert states["section"].state.value == "failed_required"
    assert files[0].error_code == "classifier_required_failed"
    assert "ML_CLASSIFIERS_REQUIRED=true" in files[0].error
    assert "ML_CLASSIFIERS_REQUIRED=false" in files[0].error


def test_optional_failure_stays_degraded_and_silent():
    resources, files = _run(required=False, loader=_boom)
    assert resources.status()["paper"].state.value == "degraded"
    assert files[0].error is None


def test_required_success_leaves_files_alone():
    resources, files = _run(
        required=True, loader=lambda *args: SimpleNamespace(classify_batch=list)
    )
    assert resources.status()["paper"].state.value == "ready"
    assert files[0].error is None


def _run_override(settings, loader):
    resources = ClassifierResources(
        settings,
        memory_mode="balanced",
        managed_vllm_fraction=0.0,
        cuda_available=False,
        paper_loader=loader,
        section_loader=loader,
    )
    files = [_File()]
    ctx = SimpleNamespace(
        resources=SimpleNamespace(start_classifiers=resources.start, classifiers=resources),
        file_states=files,
    )
    return resources, files, ctx


def test_an_overridden_classifier_that_does_not_exist_stops_the_run(tmp_path):
    import pytest

    from bibr.exceptions import ModelOverrideError

    settings = _settings(required=False)
    settings.ml.paper_classifier_model_id = str(tmp_path / "paper-candidate")
    resources, files, ctx = _run_override(settings, _boom)
    with pytest.raises(ModelOverrideError, match="never falls back"):
        asyncio.run(ClassifierStage().run(ctx))
    states = resources.status()
    assert states["paper"].state.value == "failed_required"
    assert states["paper"].override
    # The shipped section model may still degrade.
    assert states["section"].state.value == "degraded"
    # Not an outage on the papers: a resumed batch would fail the same way.
    assert files[0].error is None


def _blip(model_id, revision, device):
    import httpx

    raise httpx.ConnectError("temporary failure in name resolution")


def test_a_network_blip_on_an_overridden_classifier_stops_the_run_too():
    import pytest

    from bibr.exceptions import ModelOverrideError

    for required in (False, True):
        settings = _settings(required=required)
        settings.ml.paper_classifier_model_id = "myorg/paper-classifier-mirror"
        resources, files, ctx = _run_override(settings, _blip)
        with pytest.raises(ModelOverrideError, match="the Hub could not be reached"):
            asyncio.run(ClassifierStage().run(ctx))
        status = resources.status()["paper"]
        assert status.state.value == "failed_required" and status.override
        # No verdict on the paper: a resumed batch runs it again.
        assert files[0].error is None


def test_a_network_blip_on_the_shipped_classifier_degrades_as_before():
    for required, state in ((False, "degraded"), (True, "failed_required")):
        resources, files, ctx = _run_override(_settings(required=required), _blip)
        asyncio.run(ClassifierStage().run(ctx))
        status = resources.status()["paper"]
        assert status.state.value == state and not status.override
        if required:
            # The shipped outage: bibr batch runs the paper again on resume.
            assert files[0].error_code == "classifier_required_failed"
            assert files[0].error_outage is True
        else:
            assert files[0].error is None


def test_a_no_llm_run_starts_no_classifier():
    """--no-llm types sections by the heading lookup alone and classifies no paper."""
    loaded = []

    def loader(model_id, revision, device):
        loaded.append(model_id)
        return SimpleNamespace(classify_batch=list)

    resources, files, ctx = _run_override(_settings(required=True), loader)
    ctx.config = SimpleNamespace(no_llm=True)
    asyncio.run(ClassifierStage().run(ctx))
    assert loaded == [] and not resources.started
    assert files[0].error is None
    ctx.config = SimpleNamespace(no_llm=False)
    asyncio.run(ClassifierStage().run(ctx))
    assert len(loaded) == 2 and resources.started
