"""Local native-model calls must hold the shared inference lock.

Concurrent native model work (torch-MPS loads/forwards, LightGBM predicts)
from sibling post-parse threads intermittently segfaulted `bibr chew`
(SIGSEGV/SIGABRT). The fix serializes every
local-model load + forward behind ``bibr.utils.locks.LOCAL_INFERENCE_LOCK``;
these tests pin that the wrappers actually take it (a probe thread must fail
a non-blocking acquire while the fake model is executing).
"""

import threading
from unittest import mock

from bibr.utils.locks import LOCAL_INFERENCE_LOCK


def _probe_lock_from_other_thread() -> bool:
    """True if LOCAL_INFERENCE_LOCK could be acquired from a fresh thread."""
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


def test_ner_parse_holds_inference_lock(monkeypatch):
    from bibr.extract import ref_extractor as re_mod
    from bibr.paper_contents import PaperContents

    held = {}

    class FakeParser:
        def parse_batch(self, ref_strings):
            held["locked_during_parse"] = not _probe_lock_from_other_thread()
            return [{} for _ in ref_strings]

    monkeypatch.setattr(re_mod, "_get_ner_parser", lambda *_a: FakeParser())
    contents = mock.Mock(spec=PaperContents)
    contents.region_summaries = []
    contents.processing_warnings = []
    ext = re_mod.ReferenceExtractor(contents, llm_client=mock.Mock())

    ext._parse_references_ner(["Smith, J. (2020). A title. Journal, 1, 1-2."])
    assert held["locked_during_parse"] is True
    # And released afterwards.
    assert _probe_lock_from_other_thread() is True


async def test_classifier_batch_holds_inference_lock(monkeypatch):
    from bibr.structure import section_classifier as sc_mod

    held = {}

    class FakeModel:
        def classify_batch(self, items):
            held["locked_during_classify"] = not _probe_lock_from_other_thread()
            return []

    monkeypatch.setattr(sc_mod, "_model_cache", FakeModel())
    out = await sc_mod._classify_trained_batch([object()])
    assert out == []
    assert held["locked_during_classify"] is True
    assert _probe_lock_from_other_thread() is True


async def test_classifier_load_holds_inference_lock(monkeypatch):
    from bibr.structure import section_classifier as sc_mod

    held = {}

    def fake_get_trained_model():
        held["locked_during_load"] = not _probe_lock_from_other_thread()
        return None

    monkeypatch.setattr(sc_mod, "_get_trained_model", fake_get_trained_model)
    model = await sc_mod._get_trained_model_async()
    assert model is None
    assert held["locked_during_load"] is True


def test_geom_segment_holds_inference_lock(monkeypatch):
    from bibr.extract import ref_extractor as re_mod
    from bibr.paper_contents import PaperContents

    held = {}

    class FakeGeom:
        def segment_spans(self, ref_text, lines):
            held["locked_during_segment"] = not _probe_lock_from_other_thread()
            return [(0, len(ref_text))], 1.0, 1, 1

    monkeypatch.setattr(re_mod, "_get_geom_segmenter", lambda *_a: FakeGeom())
    contents = mock.Mock(spec=PaperContents)
    contents.region_summaries = []
    contents.processing_warnings = []
    contents.ref_line_geometry = [object()]
    ext = re_mod.ReferenceExtractor(contents, llm_client=mock.Mock())

    ext._segment_geom("Smith, J. (2020). A title. Journal, 1, 1-2.")
    assert held["locked_during_segment"] is True
