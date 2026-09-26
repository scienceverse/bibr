"""Local enforcement of ML_CLASSIFIERS_REQUIRED (pipeline-core-7).

A required classifier that fails to load must fail the chunk's files with a
dedicated code and a message naming the setting and its opt-out — not fall
back to the LLM tier silently.
"""

import asyncio
from types import SimpleNamespace

from bibr.pipeline.classifier_resources import ClassifierResources
from bibr.pipeline.stages.classifiers import ClassifierStage


def _settings(*, required):
    ml = SimpleNamespace(
        paper_classifier_model_id="paper/model",
        paper_classifier_revision="rev",
        paper_classifier_device=None,
        paper_classifier_batch_size=8,
        paper_classifier_batch_timeout_ms=2.0,
        paper_classifier_estimated_peak_mb=1,
        section_classifier_model_id="section/model",
        section_classifier_revision="rev",
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

    def set_error(self, message, *, code=None, stage=None, exc=None):
        self.error = message
        self.error_code = code
        self.failed_stage = stage


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
