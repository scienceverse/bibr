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
        # A classifier already placed comes off the headroom of the next one.
        ({"placed_vram_bytes": 3_000}, "cpu"),
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


# --- VRAM budget across both classifiers and managed servers (pipeline-core-6) ---

_GIB = 1024**3


def _recorder(devices, name):
    """A loader that records the device it was given instead of loading a model."""

    def load(model_id, revision, device, settings=None):
        devices[name] = device
        return _Model(name)

    return load


def _recording_loaders(monkeypatch, devices):
    import bibr.pipeline.classifier_resources as cr

    monkeypatch.setattr(cr, "_load_paper_model", _recorder(devices, "paper"))
    monkeypatch.setattr(cr, "_load_section_model", _recorder(devices, "section"))
    monkeypatch.setattr(cr, "_peak_vram_bytes", lambda device: None)


def _default_peaks_settings(*, paper_device=None, section_device=None):
    """The shipped peak estimates and safety reserve (1536 / 512 / 2048 MiB)."""
    from bibr.config import MlOptions

    fields = MlOptions.model_fields
    settings = _settings()
    settings.ml.paper_classifier_device = paper_device
    settings.ml.section_classifier_device = section_device
    for name in (
        "paper_classifier_estimated_peak_mb",
        "section_classifier_estimated_peak_mb",
        "classifier_vram_safety_reserve_mb",
    ):
        setattr(settings.ml, name, fields[name].default)
    return settings


async def _placed(monkeypatch, settings, *, fraction, free_gib, total_gib):
    from bibr.pipeline.classifier_resources import ClassifierResources

    monkeypatch.setattr("bibr.pipeline.classifier_resources._peak_vram_bytes", lambda device: None)
    devices = {}
    resources = ClassifierResources(
        settings,
        memory_mode="balanced",
        managed_vllm_fraction=fraction,
        cuda_available=True,
        free_vram_bytes=int(free_gib * _GIB),
        total_vram_bytes=int(total_gib * _GIB),
        paper_loader=_recorder(devices, "paper"),
        section_loader=_recorder(devices, "section"),
    )
    try:
        await resources.start()
    finally:
        await resources.close()
    return devices


async def test_second_classifier_budgets_for_the_first(monkeypatch):
    """24 GB GPU, local vLLM LLM at 0.85: 1638 MiB usable after the reserve.

    Each model fit alone, so both went to CUDA (2048 MiB together) and ate
    into the reserve kept for vLLM. The section model now sees what the paper
    model already took and goes to CPU.
    """
    devices = await _placed(
        monkeypatch, _default_peaks_settings(), fraction=0.85, free_gib=20, total_gib=24
    )
    assert devices == {"paper": "cuda", "section": "cpu"}


async def test_explicit_device_wins_and_counts_toward_the_budget(monkeypatch):
    devices = await _placed(
        monkeypatch,
        _default_peaks_settings(paper_device="cuda"),
        fraction=0.85,
        free_gib=20,
        total_gib=24,
    )
    assert devices == {"paper": "cuda", "section": "cpu"}


async def test_explicit_section_device_counts_toward_the_paper_budget(monkeypatch):
    """Section pinned to CUDA, paper on auto: the paper pick sees the section model.

    The paper model used to be picked first with nothing placed, so both went
    to CUDA (2048 MiB against 1638 MiB usable). The explicitly placed section
    model now loads first, and the paper model goes to CPU.
    """
    devices = await _placed(
        monkeypatch,
        _default_peaks_settings(section_device="cuda"),
        fraction=0.85,
        free_gib=20,
        total_gib=24,
    )
    assert devices == {"paper": "cpu", "section": "cuda"}


def _pipeline_devices(monkeypatch, *, free_gib, total_gib, **pipeline_kwargs):
    """Build a LocalPipeline and start its classifiers against a fake GPU."""
    import asyncio

    from bibr.config import GlobalSettings
    from bibr.local.pipeline import LocalPipeline

    devices = {}
    _recording_loaders(monkeypatch, devices)
    monkeypatch.setattr(
        "bibr.pipeline.classifier_resources._cuda_memory_info",
        lambda: (True, int(free_gib * _GIB), int(total_gib * _GIB)),
    )
    settings = GlobalSettings()
    # The suite blanks the model ids so nothing downloads; the loaders are fakes.
    settings.ml.paper_classifier_model_id = "paper/model"
    settings.ml.section_classifier_model_id = "section/model"
    pipe = LocalPipeline(memory_mode="balanced", settings=settings, **pipeline_kwargs)

    async def start():
        try:
            await pipe._resources.start_classifiers()
        finally:
            await pipe._resources.close_classifiers()

    asyncio.run(start())
    return pipe, devices


def _pin_linux_x86(monkeypatch, *, vram_gb):
    import platform

    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr("bibr.ocr.registry._cuda_vram_gb", lambda: vram_gb)


@pytest.mark.parametrize("ocr_backend", ["paddle-vllm", "paddle"])
def test_classifiers_leave_room_for_managed_paddle_vllm_ocr(monkeypatch, ocr_backend):
    """Cloud LLM + managed Paddle OCR vLLM on a 24 GB GPU.

    The streaming plan loads the classifiers before OCR starts; with a cloud
    LLM nothing was reserved, so both went to CUDA and vLLM's 92% claim then
    found the memory gone. Both now go to CPU (same results, slower).
    """
    _pin_linux_x86(monkeypatch, vram_gb=24.0)
    _, devices = _pipeline_devices(
        monkeypatch, free_gib=22, total_gib=24, llm_backend="cloud", ocr_backend=ocr_backend
    )
    assert devices == {"paper": "cpu", "section": "cpu"}


def test_local_vllm_llm_keeps_its_own_fraction_beside_paddle_vllm(monkeypatch):
    """The OCR claim only widens the classifier reserve, not the LLM launch."""
    _pin_linux_x86(monkeypatch, vram_gb=80.0)
    pipe, devices = _pipeline_devices(
        monkeypatch, free_gib=78, total_gib=80, llm_backend="vllm", ocr_backend="paddle-vllm"
    )
    assert pipe._resources._managed_vllm_fraction == pipe._settings.llm.local_mem_fraction
    assert pipe._resources.classifiers._managed_vllm_fraction == 0.92
    # 8% of 80 GiB minus the 2 GiB reserve still holds both models.
    assert devices == {"paper": "cuda", "section": "cuda"}


@pytest.mark.parametrize(
    ("pipeline_kwargs", "vram_gb"),
    [
        # Remote OCR, cloud LLM: no managed GPU server at all.
        ({"ocr_url": "http://127.0.0.1:9/v1"}, 24.0),
        # A small GPU: the automatic chain skips paddle-vllm (llama.cpp OCR).
        ({"ocr_backend": "paddle"}, 6.0),
    ],
)
def test_free_gpu_without_managed_vllm_keeps_both_on_cuda(monkeypatch, pipeline_kwargs, vram_gb):
    _pin_linux_x86(monkeypatch, vram_gb=vram_gb)
    _, devices = _pipeline_devices(
        monkeypatch,
        free_gib=vram_gb - 1,
        total_gib=vram_gb,
        llm_backend="cloud",
        **pipeline_kwargs,
    )
    assert devices == {"paper": "cuda", "section": "cuda"}


@pytest.mark.parametrize(
    ("llm_backend", "llm_fraction", "ocr_backend", "expected"),
    [
        ("cloud", 0.85, "paddle-vllm", 0.92),
        ("cloud", 0.85, "glm-http", 0.0),
        ("cloud", 0.85, "glm-llama", 0.0),
        ("vllm", 0.85, "glm-llama", 0.85),
        # 0.85 + 0.92 cannot both be up on one GPU: whichever is up is the bound.
        ("vllm", 0.85, "paddle-vllm", 0.92),
        # Shares that fit together can be up at once (keep_all, later chunks).
        ("vllm", 0.05, "paddle-vllm", 0.97),
    ],
)
def test_managed_gpu_fraction(llm_backend, llm_fraction, ocr_backend, expected):
    from bibr.config import GlobalSettings
    from bibr.local.pipeline import _managed_gpu_fraction

    settings = GlobalSettings()
    settings.llm.local_mem_fraction = llm_fraction
    assert _managed_gpu_fraction(llm_backend, ocr_backend, settings) == pytest.approx(expected)
