from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"explicit_device": "cuda:1"}, "cuda:1"),
        ({"memory_mode": "aggressive"}, "cpu"),
        ({"cuda_available": False}, "cpu"),
        ({"free_vram_bytes": 3_000, "estimated_peak_bytes": 2_000}, "cpu"),
        ({"free_vram_bytes": 9_000, "estimated_peak_bytes": 2_000}, "cuda"),
    ],
)
def test_choose_classifier_device(kwargs, expected):
    from bibr.pipeline.classifier_resources import choose_classifier_device

    defaults = {
        "explicit_device": None,
        "memory_mode": "balanced",
        "cuda_available": True,
        "free_vram_bytes": 9_000,
        "total_vram_bytes": 10_000,
        "managed_vllm_fraction": 0.4,
        "estimated_peak_bytes": 2_000,
        "safety_reserve_bytes": 1_000,
    }
    defaults.update(kwargs)
    assert choose_classifier_device(**defaults) == expected


def _settings(*, required=False, paper_id="paper/model", section_id="section/model"):
    ml = SimpleNamespace(
        paper_classifier_model_id=paper_id,
        paper_classifier_revision="paper-rev",
        paper_classifier_device=None,
        paper_classifier_batch_size=8,
        paper_classifier_batch_timeout_ms=2.0,
        paper_classifier_estimated_peak_mb=1,
        section_classifier_model_id=section_id,
        section_classifier_revision="section-rev",
        section_classifier_device=None,
        section_classifier_batch_size=8,
        section_classifier_batch_timeout_ms=2.0,
        section_classifier_estimated_peak_mb=1,
        classifier_vram_safety_reserve_mb=1,
        classifiers_required=required,
    )
    return SimpleNamespace(ml=ml)


class _Model:
    def __init__(self, prefix):
        self.prefix = prefix
        self.calls = []

    def classify_batch(self, items):
        self.calls.append(list(items))
        return [f"{self.prefix}:{item}" for item in items]


def test_models_loaded_on_setup_loop_classify_on_request_loop():
    import asyncio

    from bibr.pipeline.classifier_resources import ClassifierResources

    model = _Model("paper")
    resources = ClassifierResources(
        _settings(section_id=None),
        memory_mode="balanced",
        managed_vllm_fraction=0.0,
        cuda_available=False,
        paper_loader=lambda *args: model,
    )
    asyncio.run(resources.start())
    assert resources._paper.batcher._collector is None

    async def classify_and_close():
        result = await resources.classify_paper("x")
        await resources.close()
        return result

    assert asyncio.run(classify_and_close()) == "paper:x"


async def test_start_loads_once_and_batches_both_models():
    from bibr.pipeline.classifier_resources import ClassifierResources, ClassifierState

    loads = []
    paper = _Model("paper")
    section = _Model("section")

    def paper_loader(model_id, revision, device):
        loads.append(("paper", model_id, revision, device))
        return paper

    def section_loader(model_id, revision, device):
        loads.append(("section", model_id, revision, device))
        return section

    resources = ClassifierResources(
        _settings(),
        memory_mode="balanced",
        managed_vllm_fraction=0.0,
        cuda_available=False,
        paper_loader=paper_loader,
        section_loader=section_loader,
    )
    try:
        await resources.start()
        await resources.start()
        paper_results = await __import__("asyncio").gather(
            resources.classify_paper("a"), resources.classify_paper("b")
        )
        section_results = await __import__("asyncio").gather(
            resources.classify_section("x"), resources.classify_section("y")
        )
        assert paper_results == ["paper:a", "paper:b"]
        assert section_results == ["section:x", "section:y"]
        assert len(paper.calls) == 1
        assert len(section.calls) == 1
        assert resources.status()["paper"].state is ClassifierState.READY
        assert resources.status()["section"].state is ClassifierState.READY
        assert [load[0] for load in loads] == ["paper", "section"]
    finally:
        await resources.close()


async def test_load_failure_is_sticky_and_degraded_when_fallback_allowed():
    from bibr.pipeline.classifier_resources import ClassifierResources, ClassifierState

    attempts = 0

    def fail(*args):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("secret path /tmp/model exploded")

    resources = ClassifierResources(
        _settings(section_id=None),
        memory_mode="balanced",
        managed_vllm_fraction=0.0,
        cuda_available=False,
        paper_loader=fail,
    )
    await resources.start()
    await resources.start()
    assert attempts == 1
    status = resources.status()["paper"]
    assert status.state is ClassifierState.DEGRADED
    assert "exploded" in status.error
    assert await resources.classify_paper("x") is None
    await resources.close()


async def test_required_load_failure_is_failed_required():
    from bibr.pipeline.classifier_resources import ClassifierResources, ClassifierState

    resources = ClassifierResources(
        _settings(required=True, section_id=None),
        memory_mode="balanced",
        managed_vllm_fraction=0.0,
        cuda_available=False,
        paper_loader=lambda *args: (_ for _ in ()).throw(RuntimeError("no model")),
    )
    await resources.start()
    assert resources.status()["paper"].state is ClassifierState.FAILED_REQUIRED
    await resources.close()


async def test_unconfigured_resource_returns_none_without_loading():
    from bibr.pipeline.classifier_resources import ClassifierResources, ClassifierState

    resources = ClassifierResources(
        _settings(paper_id=None, section_id=None),
        memory_mode="balanced",
        managed_vllm_fraction=0.0,
        cuda_available=False,
    )
    await resources.start()
    assert resources.status()["paper"].state is ClassifierState.UNCONFIGURED
    assert await resources.classify_paper("x") is None
    await resources.close()
    assert resources.status()["paper"].state is ClassifierState.CLOSED
