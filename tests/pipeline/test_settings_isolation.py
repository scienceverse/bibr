"""Cross-pipeline runtime settings and resource isolation contracts."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from bibr.config import GlobalSettings
from bibr.paper import PaperReference
from bibr.paper_contents import CanonicalSection
from bibr.pipeline import ocr_cache
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.resources import ResourceManager
from bibr.pipeline.stages.render_ocr import InterleavedRenderOcrStage
from bibr.pipeline.state import FileState
from bibr.structure.section_classifier import classify_headers_batch_async

RUNTIME_STAGE_MODULES = (
    "bibr/pipeline/context.py",
    "bibr/pipeline/ocr_cache.py",
    "bibr/pipeline/stages/export.py",
    "bibr/pipeline/stages/layout.py",
    "bibr/pipeline/stages/native_text.py",
    "bibr/pipeline/stages/ocr.py",
    "bibr/pipeline/stages/post_parse.py",
    "bibr/pipeline/stages/render_ocr.py",
)

RUNTIME_HELPER_MODULES = (
    "bibr/extract/core_metadata.py",
    "bibr/extract/ref_extractor.py",
    "bibr/extract/ref_locator.py",
    "bibr/extract/training_capture.py",
    "bibr/enrich/references.py",
    "bibr/structure/implicit_sections.py",
    "bibr/structure/paper_classifier.py",
    "bibr/structure/section_classifier.py",
)

PROCESS_BOUNDARY_SETTINGS_MODULES = {
    "bibr/__init__.py",
    "bibr/config.py",
    "bibr/demo/local_app.py",
    "bibr/local/cli/doctor.py",
    "bibr/local/cli/dry_run.py",
    "bibr/local/cli/run_config.py",
    "bibr/serve/app.py",
    "bibr/serve/auth.py",
    "bibr/serve/jobs.py",
    "bibr/serve/jobs_redis.py",  # same API-process boundary as jobs.py (Settings.jobs)
    "bibr/setup_wizard.py",
}


@pytest.mark.parametrize("path", RUNTIME_STAGE_MODULES)
def test_runtime_stages_do_not_read_global_settings(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "Settings"
    ]
    assert reads == []


@pytest.mark.parametrize("path", RUNTIME_HELPER_MODULES)
def test_runtime_helpers_do_not_read_global_settings(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "Settings"
    ]
    assert reads == []


def test_only_process_boundaries_import_global_settings():
    violations: list[str] = []
    for path in sorted(Path("bibr").rglob("*.py")):
        relative = path.as_posix()
        if relative in PROCESS_BOUNDARY_SETTINGS_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "bibr.config"
                and any(alias.name == "Settings" for alias in node.names)
            ):
                violations.append(f"{relative}:{node.lineno}")
    assert violations == []


def test_reference_strategy_uses_supplied_settings_snapshot():
    from bibr.extract.ref_extractor import _resolve_ref_strategies

    first = GlobalSettings()
    second = GlobalSettings()
    first.REF_SEG_STRATEGY = "llm"
    first.REF_PARSE_STRATEGY = "ner"
    second.REF_SEG_STRATEGY = "region"
    second.REF_PARSE_STRATEGY = "llm"

    assert _resolve_ref_strategies(settings=first) == ("llm", "ner")
    assert _resolve_ref_strategies(settings=second) == ("region", "llm")


def test_training_capture_uses_supplied_settings_snapshot(tmp_path):
    from bibr.extract.training_capture import save_seg_training_data

    first = GlobalSettings()
    second = GlobalSettings()
    first.REF_TRAINING_DATA_DIR = str(tmp_path / "first")
    second.REF_TRAINING_DATA_DIR = str(tmp_path / "second")

    save_seg_training_data("first block", ["first block"], settings=first)
    save_seg_training_data("second block", ["second block"], settings=second)

    assert len(list((tmp_path / "first" / "segmentation").glob("*.json"))) == 1
    assert len(list((tmp_path / "second" / "segmentation").glob("*.json"))) == 1


def test_ref_locator_uses_supplied_settings_snapshot():
    import pandas as pd

    from bibr.extract.ref_locator import RefLocator

    contents = MagicMock()
    contents.sentences_df = pd.DataFrame({"text": ["front matter"]})
    contents.sections = []
    first = GlobalSettings()
    second = GlobalSettings()
    first.llm.core_cutoff_max_sentences = 7
    second.llm.core_cutoff_max_sentences = 19

    assert RefLocator(contents, settings=first).get_cutoff_index() == 7
    assert RefLocator(contents, settings=second).get_cutoff_index() == 19


async def test_resolver_construction_uses_supplied_settings_snapshot():
    from bibr.enrich.references import enrich_references

    class Resolver:
        healthy = AsyncMock(return_value=False)
        close = AsyncMock(return_value=None)

    crossref = SimpleNamespace(enrich_semaphore=asyncio.Semaphore(1))
    reference = PaperReference(
        bib_id=1,
        title="",
        first_page=None,
        volume=None,
        authors=None,
        year=None,
        container=None,
    )
    first = GlobalSettings()
    second = GlobalSettings()
    first.resolver.url = "http://first"
    first.resolver.enrich = True
    first.resolver.timeout = 1.5
    second.resolver.url = "http://second"
    second.resolver.enrich = True
    second.resolver.timeout = 9.0

    with patch("bibr.clients.resolver.ResolverClient", side_effect=[Resolver(), Resolver()]) as cls:
        await enrich_references([reference], crossref_client=crossref, settings=first)
        await enrich_references([reference], crossref_client=crossref, settings=second)

    assert cls.call_args_list == [
        call("http://first", timeout=1.5),
        call("http://second", timeout=9.0),
    ]


def test_ocr_cache_key_uses_supplied_settings_snapshot():
    file_state = FileState(path=Path("paper.pdf"))
    file_state.file_hash = "same-paper"
    config = RunConfig(ocr_backend="glm-llama")
    first = GlobalSettings()
    second = GlobalSettings()
    first.layout.dpi = 100
    second.layout.dpi = 240

    from bibr.ocr.profiles import resolve_ocr_runtime_identity

    assert ocr_cache._key(
        file_state, config, resolve_ocr_runtime_identity(config, first), first
    ) != ocr_cache._key(file_state, config, resolve_ocr_runtime_identity(config, second), second)


async def test_render_ocr_window_uses_context_settings_snapshot():
    class RecordingLayout:
        def __init__(self):
            self.windows = []

        async def run(self, ctx):
            self.windows.append(len(ctx.file_states))

    class NoopStage:
        async def run(self, ctx):  # noqa: ARG002
            return None

    async def run_with(max_files):
        settings = GlobalSettings()
        settings.ocr.max_concurrent_files = max_files
        settings.cache.ocr = False
        layout = RecordingLayout()
        stage = InterleavedRenderOcrStage(
            layout=layout,
            native_text=NoopStage(),
            ocr=NoopStage(),
        )
        files = [FileState(path=Path(f"{index}.pdf"), pdf_bytes=b"pdf") for index in range(3)]
        resources = MagicMock()
        resources.shutdown_ocr = AsyncMock(return_value=None)
        context = PipelineContext(
            file_states=files,
            progress=NullProgress(),
            resources=resources,
            config=RunConfig(ocr_backend="glm-http", llm_backend="cloud"),
            settings=settings,
        )
        await stage.run(context)
        return layout.windows

    first, second = await asyncio.gather(run_with(1), run_with(2))

    assert first == [1, 1, 1]
    assert second == [2, 1]


class _FakeClassifierResources:
    def __init__(self, canonical_type, score=0.99):
        self.canonical_type = canonical_type
        self.score = score
        self.calls = []
        self.closed = False

    async def start(self):
        return None

    async def classify_sections(self, items):
        assert not self.closed
        self.calls.extend(item.heading for item in items)
        return [
            SimpleNamespace(
                canonical_type=self.canonical_type,
                score=self.score,
                is_top_level=False,
            )
            for _ in items
        ]

    async def close(self):
        self.closed = True


async def test_concurrent_classifier_dispatch_is_resource_manager_owned():
    settings = GlobalSettings()
    first = _FakeClassifierResources(CanonicalSection.METHODS)
    second = _FakeClassifierResources(CanonicalSection.RESULTS)
    first_manager = ResourceManager(settings=settings, classifier_resources=first)
    second_manager = ResourceManager(settings=settings, classifier_resources=second)

    first_result, second_result = await asyncio.gather(
        classify_headers_batch_async(
            ["enigmatic alpha"],
            classifier_resources=first_manager.classifiers,
            settings=settings,
        ),
        classify_headers_batch_async(
            ["enigmatic beta"],
            classifier_resources=second_manager.classifiers,
            settings=settings,
        ),
    )

    assert first.calls == ["enigmatic alpha"]
    assert second.calls == ["enigmatic beta"]
    assert first_result[0][0] is CanonicalSection.METHODS
    assert second_result[0][0] is CanonicalSection.RESULTS


async def test_closing_one_manager_does_not_clear_another_classifier():
    settings = GlobalSettings()
    first = _FakeClassifierResources(CanonicalSection.METHODS)
    second = _FakeClassifierResources(CanonicalSection.RESULTS)
    first_manager = ResourceManager(settings=settings, classifier_resources=first)
    second_manager = ResourceManager(settings=settings, classifier_resources=second)

    await first_manager.close_classifiers()
    result = await classify_headers_batch_async(
        ["enigmatic survivor"],
        classifier_resources=second_manager.classifiers,
        settings=settings,
    )

    assert first.closed is True
    assert second.closed is False
    assert second.calls == ["enigmatic survivor"]
    assert result[0][0] is CanonicalSection.RESULTS


async def test_classifier_threshold_uses_supplied_settings_snapshot():
    first_settings = GlobalSettings()
    second_settings = GlobalSettings()
    first_settings.ml.section_classifier_min_confidence = 0.8
    second_settings.ml.section_classifier_min_confidence = 0.5
    first = _FakeClassifierResources(CanonicalSection.METHODS, score=0.7)
    second = _FakeClassifierResources(CanonicalSection.METHODS, score=0.7)

    first_result, second_result = await asyncio.gather(
        classify_headers_batch_async(
            ["enigmatic alpha"],
            classifier_resources=first,
            settings=first_settings,
        ),
        classify_headers_batch_async(
            ["enigmatic beta"],
            classifier_resources=second,
            settings=second_settings,
        ),
    )

    assert first_result[0][0] is CanonicalSection.UNKNOWN
    assert second_result[0][0] is CanonicalSection.METHODS
