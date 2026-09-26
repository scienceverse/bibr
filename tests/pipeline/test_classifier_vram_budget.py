"""Classifier VRAM budgeting across both models (pipeline-core-6).

``ClassifierResources.start()`` must deduct the paper model's estimated peak
from the free-VRAM budget before choosing the section model's device, and
must leave room for a managed PaddleOCR vLLM server that starts later.
Fake VRAM numbers throughout — no CUDA required.
"""

import asyncio
from types import SimpleNamespace

_MIB = 1024 * 1024


def _settings(
    *,
    paper_peak_mb=1536,
    section_peak_mb=512,
    reserve_mb=2048,
    required=False,
    ocr_backend="paddle",
):
    ml = SimpleNamespace(
        paper_classifier_model_id="paper/model",
        paper_classifier_revision="rev",
        paper_classifier_device=None,
        paper_classifier_batch_size=8,
        paper_classifier_batch_timeout_ms=2.0,
        paper_classifier_estimated_peak_mb=paper_peak_mb,
        section_classifier_model_id="section/model",
        section_classifier_revision="rev",
        section_classifier_device=None,
        section_classifier_batch_size=8,
        section_classifier_batch_timeout_ms=2.0,
        section_classifier_estimated_peak_mb=section_peak_mb,
        classifier_vram_safety_reserve_mb=reserve_mb,
        classifiers_required=required,
    )
    return SimpleNamespace(ml=ml, ocr=SimpleNamespace(backend=ocr_backend))


class _Model:
    def classify_batch(self, items):
        return list(items)


def _start(settings, *, free_gib, total_gib, fraction):
    from bibr.pipeline.classifier_resources import ClassifierResources

    seen = {}

    def loader(name):
        def _load(model_id, revision, device):
            seen[name] = device
            return _Model()

        return _load

    resources = ClassifierResources(
        settings,
        memory_mode="balanced",
        managed_vllm_fraction=fraction,
        cuda_available=True,
        free_vram_bytes=int(free_gib * 1024**3),
        total_vram_bytes=int(total_gib * 1024**3),
        paper_loader=loader("paper"),
        section_loader=loader("section"),
    )
    asyncio.run(resources.start())
    return seen


def test_second_model_sees_budget_minus_first_peak():
    # 24 GiB card, managed LLM at 0.85: usable after reserve is 1638 MiB —
    # the paper model (1536 MiB) fits, but both together (2048 MiB) do not.
    seen = _start(_settings(), free_gib=20, total_gib=24, fraction=0.85)
    assert seen["paper"] == "cuda"
    assert seen["section"] == "cpu"


def test_managed_ocr_vllm_reservation_applies():
    # Cloud LLM (fraction 0.0) with a managed paddle-vllm OCR backend: the
    # OCR server's 92% claim leaves no room for CUDA classifiers.
    seen = _start(_settings(ocr_backend="paddle-vllm"), free_gib=20, total_gib=24, fraction=0.0)
    assert seen["paper"] == "cpu"
    assert seen["section"] == "cpu"


def test_ample_vram_keeps_both_on_cuda():
    seen = _start(_settings(), free_gib=20, total_gib=24, fraction=0.0)
    assert seen == {"paper": "cuda", "section": "cuda"}
