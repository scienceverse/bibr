"""Lifecycle of the NER reference-parser singleton (audit x-performance-5).

The parser (~1 GB) lived in a module-global singleton that aggressive mode
never unloaded and it picked CUDA regardless of memory mode. These tests pin
the unload hook, the ResourceManager wiring, and the aggressive-mode CPU
device routing — all with fake parsers (no model weights needed).
"""

import threading

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
        self.session = object()

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


def test_unload_drops_reference_without_closing_parser(monkeypatch):
    """Unload clears the singleton but never closes the shared object.

    In-flight holders keep their own reference and stay usable; the object
    is freed once they finish.
    """
    parser = _FakeParser()
    _load_with(monkeypatch, parser)
    assert ex._get_ner_parser(GlobalSettings()) is parser

    ex.unload_ner_parser()

    assert ex._NER_PARSER is None
    assert ex._NER_PARSER_KEY is None
    assert parser.closed is False
    assert parser.session is not None
    assert parser.parse_batch(["a", "b"]) == [{}, {}]


def test_unload_without_parser_is_noop():
    ex.unload_ner_parser()

    assert ex._NER_PARSER is None


def test_unload_keeps_held_parser_usable(monkeypatch):
    """A holder that fetched the parser before unload keeps parsing."""
    parser = _FakeParser()
    _load_with(monkeypatch, parser)
    held = ex._get_ner_parser(GlobalSettings())

    ex.unload_ner_parser()

    assert ex._NER_PARSER is None
    assert held is parser
    assert held.parse_batch(["x"]) == [{}]
    assert parser.closed is False


def test_unload_during_parse_does_not_break_in_flight_caller(monkeypatch):
    """Concurrent unload (another pipeline closing) never breaks a parse.

    Thread A holds the parser and blocks inside the session; the main
    thread drops the singleton reference; thread A then finishes cleanly
    on the object it holds.
    """
    import time

    started = threading.Event()
    release = threading.Event()

    class _BlockingSession:
        def run(self, names, feeds):
            started.set()
            assert release.wait(5)
            n = feeds["input_ids"].shape[0]
            import numpy as np

            t = feeds["input_ids"].shape[1]
            return [np.zeros((n, t, 3), dtype="float32")]

    import numpy as np

    import bibr.ner.parser_onnx as po

    parser = object.__new__(po.OnnxRefParser)
    parser.session = _BlockingSession()
    parser._input_names = ["input_ids", "attention_mask"]
    parser.tokenizer = None
    parser.max_seq_len = 16
    parser.tags = ["O", "B-title", "I-title"]
    z = np.zeros(3, np.float32)
    parser.start_transitions = z
    parser.end_transitions = z
    parser.transitions = np.zeros((3, 3), np.float32)

    monkeypatch.setattr(ex, "_NER_PARSER", parser)
    monkeypatch.setattr(ex, "_NER_PARSER_KEY", ("ckpt", None, "rev"))
    monkeypatch.setattr(
        po,
        "encode_batch",
        lambda tok, texts, max_length, add_special_tokens: type(
            "B",
            (),
            {
                "input_ids": np.zeros((len(texts), 4), np.int64),
                "attention_mask": np.ones((len(texts), 4), np.int64),
                "offsets": [[(0, 1)] * 4 for _ in texts],
            },
        )(),
    )
    monkeypatch.setattr(
        po, "viterbi_decode", lambda em, mask, **kw: [[0, 0, 0, 0] for _ in range(em.shape[0])]
    )

    out = {}

    def pipeline_a():
        try:
            held = ex._NER_PARSER
            assert held is parser
            held.parse_batch(["ref one", "ref two"], batch_size=1)
            out["err"] = None
        except Exception as e:  # noqa: BLE001
            out["err"] = repr(e)

    thread = threading.Thread(target=pipeline_a)
    thread.start()
    assert started.wait(5)
    time.sleep(0.05)
    ex.unload_ner_parser()
    assert ex._NER_PARSER is None
    release.set()
    thread.join(5)
    assert out.get("err") is None, out


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


def test_reference_extractor_threads_memory_mode_to_parser(monkeypatch):
    """ReferenceExtractor reaches the parser with its memory mode, so
    aggressive mode loads the parser on CPU end to end."""
    seen = {}

    def fake_get_parser(settings, memory_mode=None):
        seen["memory_mode"] = memory_mode
        parser = _FakeParser()
        parser.parse_batch = lambda texts, batch_size=32: [{} for _ in texts]
        return parser

    monkeypatch.setattr(ex, "_get_ner_parser", fake_get_parser)
    settings = GlobalSettings()
    extractor = ex.ReferenceExtractor(
        contents=None, file_hash="x", settings=settings, memory_mode="aggressive"
    )
    monkeypatch.setattr("bibr.extract.ref_extractor._strip_enum_markers", lambda refs: refs)
    extractor._parse_references_ner_aligned(["Doe, J. (2020). Title. Journal, 1, 1-2."])

    assert seen["memory_mode"] == "aggressive"


def test_metadata_extractor_threads_memory_mode_to_reference_extractor(monkeypatch):
    """MetadataExtractor forwards memory_mode to ReferenceExtractor."""
    from bibr.extract import extractor as extractor_mod

    seen = {}
    real_reference_extractor = extractor_mod.ReferenceExtractor

    def fake_reference_extractor(*args, **kwargs):
        seen["memory_mode"] = kwargs.get("memory_mode")
        return real_reference_extractor(*args, **kwargs)

    monkeypatch.setattr(extractor_mod, "ReferenceExtractor", fake_reference_extractor)
    from unittest.mock import MagicMock

    contents = MagicMock()
    contents.sentences_df = MagicMock()
    extractor_mod.MetadataExtractor(
        contents, file_hash="x", settings=GlobalSettings(), memory_mode="aggressive"
    )

    assert seen["memory_mode"] == "aggressive"


def test_aggressive_metadata_extractor_loads_parser_on_cpu(monkeypatch):
    """End to end: MetadataExtractor in aggressive mode loads the NER
    parser on CPU (memory_mode threads through ReferenceExtractor)."""
    from unittest.mock import MagicMock

    from bibr.extract import extractor as extractor_mod

    parser = _FakeParser()
    seen = _load_with(monkeypatch, parser)
    contents = MagicMock()
    contents.sentences_df = MagicMock()
    extractor = extractor_mod.MetadataExtractor(
        contents, file_hash="x", settings=GlobalSettings(), memory_mode="aggressive"
    )
    monkeypatch.setattr("bibr.extract.ref_extractor._strip_enum_markers", lambda refs: refs)
    extractor.refs._parse_references_ner_aligned(
        ["Doe, J. (2020). A study of things. Journal, 1, 1-2."]
    )

    assert seen["device"] == "cpu"


async def test_extract_metadata_and_equations_threads_memory_mode(monkeypatch):
    """Phase 1 forwards memory_mode to both the preparsed path and the
    MetadataExtractor branch."""
    from unittest.mock import AsyncMock, MagicMock

    from bibr.pipeline.stages import post_parse as post_parse_mod

    seen_preparsed = {}
    seen_extractor = {}

    async def fake_preparsed(*args, **kwargs):
        seen_preparsed["memory_mode"] = kwargs.get("memory_mode")
        from bibr.models import PaperMetadata

        return PaperMetadata(doi="", title="")

    def fake_metadata_extractor(*args, **kwargs):
        seen_extractor["memory_mode"] = kwargs.get("memory_mode")
        mock = MagicMock()
        mock.extract_all_metadata = AsyncMock(
            return_value=__import__("bibr.models", fromlist=["PaperMetadata"]).PaperMetadata(
                doi="", title=""
            )
        )
        mock.validation_issues = []
        return mock

    monkeypatch.setattr(post_parse_mod, "_resolve_preparsed_references", fake_preparsed)
    monkeypatch.setattr("bibr.extract.extractor.MetadataExtractor", fake_metadata_extractor)
    settings = GlobalSettings()
    contents = MagicMock()
    contents.preparsed_metadata = MagicMock()
    contents.sentences = []
    contents.sections = []

    await post_parse_mod._extract_metadata_and_equations(
        contents,
        "hash",
        False,
        llm_client=MagicMock(),
        extract_equations=False,
        settings=settings,
        memory_mode="aggressive",
    )
    assert seen_preparsed["memory_mode"] == "aggressive"

    contents2 = MagicMock()
    contents2.preparsed_metadata = None
    contents2.sentences = []
    contents2.sections = []
    await post_parse_mod._extract_metadata_and_equations(
        contents2,
        "hash",
        False,
        llm_client=MagicMock(),
        extract_equations=False,
        settings=settings,
        memory_mode="aggressive",
    )
    assert seen_extractor["memory_mode"] == "aggressive"


async def test_resolve_preparsed_references_threads_memory_mode(monkeypatch):
    """The JATS preparsed path forwards memory_mode to MetadataExtractor."""
    from unittest.mock import MagicMock

    from bibr.pipeline.stages import post_parse as post_parse_mod

    seen = {}

    def fake_metadata_extractor(*args, **kwargs):
        seen["memory_mode"] = kwargs.get("memory_mode")
        mock = MagicMock()
        mock._collect_reference_rows.return_value = None
        return mock

    monkeypatch.setattr("bibr.extract.extractor.MetadataExtractor", fake_metadata_extractor)
    contents = MagicMock()
    contents.native_references = None
    paper_metadata = MagicMock()

    await post_parse_mod._resolve_preparsed_references(
        contents,
        paper_metadata,
        "hash",
        llm_client=MagicMock(),
        ref_seg_strategy=None,
        ref_parse_strategy="ner",
        settings=GlobalSettings(),
        memory_mode="aggressive",
    )

    assert seen["memory_mode"] == "aggressive"


def test_resource_manager_unloads_ner_parser(monkeypatch):
    from bibr.pipeline.resources import ResourceManager

    parser = _FakeParser()
    _load_with(monkeypatch, parser)
    ex._get_ner_parser(GlobalSettings())
    manager = ResourceManager(memory_mode="aggressive", settings=GlobalSettings())

    manager.unload_ner_parser()

    assert ex._NER_PARSER is None
    assert parser.closed is False


async def test_resource_manager_close_models_releases_ner_parser_in_aggressive(
    monkeypatch,
):
    from bibr.pipeline.resources import ResourceManager

    parser = _FakeParser()
    _load_with(monkeypatch, parser)
    ex._get_ner_parser(GlobalSettings())
    manager = ResourceManager(memory_mode="aggressive", settings=GlobalSettings())

    await manager.close_models()

    assert ex._NER_PARSER is None
    assert parser.closed is False


async def test_resource_manager_close_models_keeps_ner_parser_in_balanced(monkeypatch):
    """Balanced one-shot chew() calls share the process: closing must not
    unload the parser there, or every file would reload its ~1 GB."""
    from bibr.pipeline.resources import ResourceManager

    parser = _FakeParser()
    _load_with(monkeypatch, parser)
    ex._get_ner_parser(GlobalSettings())
    manager = ResourceManager(memory_mode="balanced", settings=GlobalSettings())

    await manager.close_models()

    assert ex._NER_PARSER is parser
    assert parser.closed is False


async def test_resource_manager_close_models_without_ownership_keeps_parser(monkeypatch):
    from bibr.pipeline.resources import ResourceManager

    parser = _FakeParser()
    _load_with(monkeypatch, parser)
    ex._get_ner_parser(GlobalSettings())
    manager = ResourceManager(
        memory_mode="aggressive", settings=GlobalSettings(), owns_models=False
    )

    await manager.close_models()

    assert ex._NER_PARSER is parser
    assert parser.closed is False


def test_post_parse_forwards_memory_mode_to_phase1():
    """post_parse passes memory_mode into _extract_metadata_and_equations.

    Driving post_parse end to end needs model weights, so this pins the
    call-site wiring textually: the mutant that drops the kwarg removes
    exactly this line.
    """
    import inspect

    from bibr.pipeline.stages import post_parse as post_parse_mod

    src = inspect.getsource(post_parse_mod.post_parse)
    start = src.index("_extract_metadata_and_equations(") + len("_extract_metadata_and_equations")
    depth = 0
    for i, ch in enumerate(src[start:], start):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                block = src[start : i + 1]
                break
    else:
        raise AssertionError("unbalanced _extract_metadata_and_equations call")

    assert "memory_mode=memory_mode" in block
