"""Audit fix: a required classifier that fails to load is an outage, and once
it is known to have failed, later chunks fail before render/OCR."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def _classifier_resources():
    from bibr.pipeline.classifier_resources import ClassifierResources

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
        classifiers_required=True,
    )

    def _boom(model_id, revision, device):
        raise OSError("classifier repo unreachable")

    classifiers = ClassifierResources(
        SimpleNamespace(ml=ml),
        memory_mode="balanced",
        managed_vllm_fraction=0.0,
        cuda_available=False,
        paper_loader=_boom,
        section_loader=_boom,
    )
    return SimpleNamespace(start_classifiers=classifiers.start, classifiers=classifiers)


async def test_required_classifier_failure_is_an_outage_resume_reruns():
    from bibr.api import ChewFailure
    from bibr.batch.ledger import UPSTREAM_UNAVAILABLE, reruns_by_default
    from bibr.batch.runner import _local_outcome
    from bibr.pipeline.context import PipelineContext, RunConfig
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.stages.classifiers import ClassifierStage
    from bibr.pipeline.state import FileState

    fs = FileState(path=Path("a.pdf"))
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=_classifier_resources(),
        config=RunConfig(),
    )

    await ClassifierStage().run(ctx)

    assert fs.error_code == "classifier_required_failed"
    assert fs.error_outage is True
    failure = ChewFailure(fs.path, fs.error, fs.error_code, fs.failed_stage, outage=fs.error_outage)
    outcome = _local_outcome(failure, "t0", "t1", 1.0)
    assert outcome.error_code == UPSTREAM_UNAVAILABLE
    assert reruns_by_default({"error_code": outcome.error_code})


def test_local_barrier_plan_gates_render_ocr_on_known_classifier_failure():
    from bibr.pipeline.plans import build_stage_plan

    names = [s.name for s in build_stage_plan(mode="local", stream_backhalf=False, enrichers=[])]

    assert names.index("classifier_gate") + 1 == names.index("render_ocr")
    assert names.index("render_ocr") < names.index("classifiers")


async def test_later_chunks_fail_before_render_ocr_once_the_classifier_failed():
    from bibr.config import snapshot_settings
    from bibr.pipeline.context import RunConfig
    from bibr.pipeline.pipeline import Pipeline
    from bibr.pipeline.stages.classifiers import ClassifierStage, RequiredClassifierGate
    from bibr.pipeline.state import FileState

    rendered: list[str] = []

    class _RenderOcr:
        name = "render_ocr"
        requires = ()
        produces = ()

        async def run(self, ctx):
            rendered.extend(fs.path.name for fs in ctx.alive())

    pipeline = Pipeline(
        stages=[RequiredClassifierGate(), _RenderOcr(), ClassifierStage()],
        resources=_classifier_resources(),
        config=RunConfig(),
        settings=snapshot_settings(),
    )
    first, second = FileState(path=Path("a.pdf")), FileState(path=Path("b.pdf"))

    await pipeline.process_chunk([first])
    await pipeline.process_chunk([second])

    assert rendered == ["a.pdf"]
    for fs in (first, second):
        assert fs.error_code == "classifier_required_failed"
        assert fs.failed_stage == "classifiers"
        assert fs.error_outage is True
