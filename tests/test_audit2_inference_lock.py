"""The shared inference lock is held only by runtimes that need it.

An ONNX Runtime session on the CPU provider alone runs concurrently; torch
models and every other ORT provider keep ``LOCAL_INFERENCE_LOCK``.
"""

import threading
from unittest import mock

import pytest

from bibr.utils.locks import LOCAL_INFERENCE_LOCK, inference_lock_for


def _lock_free_from_other_thread() -> bool:
    result = {}

    def probe():
        acquired = LOCAL_INFERENCE_LOCK.acquire(blocking=False)
        result["acquired"] = acquired
        if acquired:
            LOCAL_INFERENCE_LOCK.release()

    t = threading.Thread(target=probe)
    t.start()
    t.join()
    return result["acquired"]


class _Session:
    def __init__(self, *providers):
        self._providers = list(providers)

    def get_providers(self):
        return self._providers


@pytest.mark.parametrize(
    ("providers", "locked"),
    [
        (("CPUExecutionProvider",), False),
        (("CUDAExecutionProvider", "CPUExecutionProvider"), True),
        (("CoreMLExecutionProvider", "CPUExecutionProvider"), True),
        (("DmlExecutionProvider", "CPUExecutionProvider"), True),
        ((), True),
    ],
)
def test_lock_by_provider(providers, locked):
    assert (inference_lock_for(_Session(*providers)) is LOCAL_INFERENCE_LOCK) is locked


@pytest.mark.parametrize("obj", [None, object(), mock.Mock(), mock.MagicMock()])
def test_anything_but_an_ort_session_keeps_the_lock(obj):
    assert inference_lock_for(obj) is LOCAL_INFERENCE_LOCK


def _extractor(re_mod):
    from bibr.paper_contents import PaperContents

    contents = mock.Mock(spec=PaperContents)
    contents.region_summaries = []
    contents.processing_warnings = []
    return re_mod.ReferenceExtractor(contents, llm_client=mock.Mock())


def test_cpu_onnx_parse_runs_without_the_lock(monkeypatch):
    from bibr.extract import ref_extractor as re_mod

    held = {}

    class CpuParser:
        session = _Session("CPUExecutionProvider")

        def parse_batch(self, ref_strings):
            held["free"] = _lock_free_from_other_thread()
            return [{} for _ in ref_strings]

    monkeypatch.setattr(re_mod, "_get_ner_parser", lambda *_a: CpuParser())
    _extractor(re_mod)._parse_references_ner(["Smith, J. (2020). A title. Journal, 1, 1-2."])
    assert held["free"] is True


def test_cuda_onnx_parse_holds_the_lock(monkeypatch):
    from bibr.extract import ref_extractor as re_mod

    held = {}

    class CudaParser:
        session = _Session("CUDAExecutionProvider", "CPUExecutionProvider")

        def parse_batch(self, ref_strings):
            held["free"] = _lock_free_from_other_thread()
            return [{} for _ in ref_strings]

    monkeypatch.setattr(re_mod, "_get_ner_parser", lambda *_a: CudaParser())
    _extractor(re_mod)._parse_references_ner(["Smith, J. (2020). A title. Journal, 1, 1-2."])
    assert held["free"] is False


def test_concurrent_papers_keep_their_own_truncation_count(monkeypatch):
    """Both papers' parses write the shared parser's count before either reads it."""
    from bibr.extract import ref_extractor as re_mod
    from bibr.processing_warnings import WarningCode

    both_parsed = threading.Barrier(2, timeout=10)

    class CpuParser:
        session = _Session("CPUExecutionProvider")
        last_truncated_count = 0

        def parse_batch(self, ref_strings):
            self.last_truncated_count = len(ref_strings) if "LONG" in ref_strings[0] else 0
            both_parsed.wait()
            return [{} for _ in ref_strings]

    shared = CpuParser()
    monkeypatch.setattr(re_mod, "_get_ner_parser", lambda *_a: shared)
    long_paper, short_paper = _extractor(re_mod), _extractor(re_mod)
    threads = [
        threading.Thread(target=long_paper._parse_references_ner, args=(["LONG a", "LONG b"],)),
        threading.Thread(target=short_paper._parse_references_ner, args=(["short"],)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    def truncated(ext):
        return [
            w.message
            for w in ext.contents.processing_warnings
            if w.code == WarningCode.REF_PARSE_TRUNCATED
        ]

    assert truncated(short_paper) == []
    assert len(truncated(long_paper)) == 1 and truncated(long_paper)[0].startswith("2 ")


def test_parser_load_holds_the_lock(monkeypatch):
    from bibr.config import snapshot_settings
    from bibr.extract import ref_extractor as re_mod

    held = {}

    def fake_load(*_a, **_k):
        held["free"] = _lock_free_from_other_thread()
        return object()

    monkeypatch.setattr(re_mod, "_NER_PARSER", None)
    monkeypatch.setattr(re_mod, "_NER_PARSER_KEY", None)
    monkeypatch.setattr("bibr.ner.runtime.load_ref_parser", fake_load)
    parser = re_mod._get_ner_parser(snapshot_settings())
    assert held["free"] is False
    # The cached parser comes back without taking the lock.
    LOCAL_INFERENCE_LOCK.acquire()
    try:
        result = {}
        t = threading.Thread(
            target=lambda: result.setdefault("p", re_mod._get_ner_parser(snapshot_settings()))
        )
        t.start()
        t.join(timeout=5)
        assert result.get("p") is parser
    finally:
        LOCAL_INFERENCE_LOCK.release()
