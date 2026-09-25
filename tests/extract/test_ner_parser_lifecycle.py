"""Lifecycle of the NER reference-parser singleton (audit x-performance-5).

The parser (~1 GB) lived in a module-global singleton that aggressive mode
never unloaded and it picked CUDA regardless of memory mode. These tests pin
the unload hook, the ResourceManager wiring, and the aggressive-mode CPU
device routing — all with fake parsers (no model weights needed).
"""

import pytest

from bibr.config import GlobalSettings
from bibr.extract import ref_extractor as ex


@pytest.fixture(autouse=True)
def _empty_singleton(monkeypatch):
    monkeypatch.setattr(ex, "_NER_PARSER", None)
    monkeypatch.setattr(ex, "_NER_PARSER_KEY", None)


class _FakeParser:
    def __init__(self):
        self.closed = False
        self.device = None

    def parse_batch(self, texts, batch_size=32):
        return [{} for _ in texts]

    def close(self):
        self.closed = True


def _load_with(monkeypatch, parser):
    seen = {}

    def fake_load(ckpt, device=None, revision=None, settings=None):
        seen["device"] = device
        parser.device = device
        return parser

    monkeypatch.setattr("bibr.ner.runtime.load_ref_parser", fake_load)
    return seen


def test_unload_clears_singleton_and_closes_parser(monkeypatch):
    parser = _FakeParser()
    _load_with(monkeypatch, parser)
    assert ex._get_ner_parser(GlobalSettings()) is parser

    ex.unload_ner_parser()

    assert ex._NER_PARSER is None
    assert ex._NER_PARSER_KEY is None
    assert parser.closed is True


def test_unload_without_parser_is_noop():
    ex.unload_ner_parser()

    assert ex._NER_PARSER is None


def test_unload_survives_failing_close(monkeypatch):
    class _BadClose(_FakeParser):
        def close(self):
            raise RuntimeError("cannot release")

    parser = _BadClose()
    _load_with(monkeypatch, parser)
    ex._get_ner_parser(GlobalSettings())

    ex.unload_ner_parser()  # never raises

    assert ex._NER_PARSER is None


def test_unload_reloads_lazily(monkeypatch):
    parser = _FakeParser()
    _load_with(monkeypatch, parser)
    ex._get_ner_parser(GlobalSettings())
    ex.unload_ner_parser()

    assert ex._get_ner_parser(GlobalSettings()) is parser


def test_aggressive_mode_loads_parser_on_cpu(monkeypatch):
    seen = _load_with(monkeypatch, _FakeParser())

    ex._get_ner_parser(GlobalSettings(), memory_mode="aggressive")

    assert seen["device"] == "cpu"


def test_balanced_mode_keeps_auto_device(monkeypatch):
    seen = _load_with(monkeypatch, _FakeParser())

    ex._get_ner_parser(GlobalSettings(), memory_mode="balanced")

    assert seen["device"] is None


def test_explicit_ner_device_wins_over_aggressive():
    settings = GlobalSettings(NER_DEVICE="cuda")

    assert ex.resolve_ner_device(settings, "aggressive") == "cuda"


def test_settings_memory_mode_aggressive_forces_cpu(monkeypatch):
    monkeypatch.setenv("PIPELINE_MEMORY_MODE", "aggressive")
    settings = GlobalSettings()

    assert ex.resolve_ner_device(settings) == "cpu"


def test_parser_cache_key_includes_device(monkeypatch):
    parser = _FakeParser()
    loads = []

    def fake_load(ckpt, device=None, revision=None, settings=None):
        loads.append(device)
        return parser

    monkeypatch.setattr("bibr.ner.runtime.load_ref_parser", fake_load)
    settings = GlobalSettings()

    ex._get_ner_parser(settings, memory_mode="balanced")
    ex._get_ner_parser(settings, memory_mode="aggressive")

    # Different resolved devices must not share one session.
    assert loads == [None, "cpu"]


def test_resource_manager_unloads_ner_parser(monkeypatch):
    from bibr.pipeline.resources import ResourceManager

    parser = _FakeParser()
    _load_with(monkeypatch, parser)
    ex._get_ner_parser(GlobalSettings())
    manager = ResourceManager(memory_mode="aggressive", settings=GlobalSettings())

    manager.unload_ner_parser()

    assert ex._NER_PARSER is None
    assert parser.closed is True


async def test_resource_manager_close_models_releases_ner_parser(monkeypatch):
    from bibr.pipeline.resources import ResourceManager

    parser = _FakeParser()
    _load_with(monkeypatch, parser)
    ex._get_ner_parser(GlobalSettings())
    manager = ResourceManager(memory_mode="balanced", settings=GlobalSettings())

    await manager.close_models()

    assert ex._NER_PARSER is None
    assert parser.closed is True


async def test_resource_manager_close_models_without_ownership_keeps_parser(monkeypatch):
    from bibr.pipeline.resources import ResourceManager

    parser = _FakeParser()
    _load_with(monkeypatch, parser)
    ex._get_ner_parser(GlobalSettings())
    manager = ResourceManager(memory_mode="balanced", settings=GlobalSettings(), owns_models=False)

    await manager.close_models()

    assert ex._NER_PARSER is parser
    assert parser.closed is False
