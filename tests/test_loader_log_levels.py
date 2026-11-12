"""Model loaders keep the long cache paths out of the default (INFO) output.

A first run printed one INFO line per ONNX bundle with its full cache path; the
path now goes to DEBUG and the INFO line says only what loaded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

_BUNDLE = Path("/very/long/hub/cache/models--org--bundle/snapshots/" + "a" * 40 + "/onnx")


class _Loaded:
    """Stands in for any ONNX model class: construction takes the bundle and a device."""

    def __init__(self, bundle, device=None):
        self.bundle = bundle

    def check_graph(self):
        """The load probe an overridden model runs: this stand-in passes it."""


def _onnx_runtime(label, *, settings, bundle, bundle_hint):  # noqa: ARG001
    return "onnx", _BUNDLE


def _records(caplog):
    return [(r.levelno, r.getMessage()) for r in caplog.records]


def _assert_path_only_at_debug(caplog, name_fragment):
    records = _records(caplog)
    named = [(level, msg) for level, msg in records if name_fragment in msg]
    assert named, records
    for level, message in named:
        if str(_BUNDLE) in message:
            assert level == logging.DEBUG, message
    info = [msg for level, msg in named if level == logging.INFO]
    assert info, "the load is still announced at INFO"
    assert not any("/very/long" in msg for msg in info)
    assert any(str(_BUNDLE) in msg for level, msg in named if level == logging.DEBUG)


@pytest.fixture
def at_debug(caplog):
    caplog.set_level(logging.DEBUG)
    return caplog


def test_the_paper_classifier_load_logs_its_path_at_debug(monkeypatch, at_debug):
    from bibr.structure import paper_classifier_common, paper_classifier_onnx

    monkeypatch.setattr("bibr.utils.ml_runtime.resolve_runtime", _onnx_runtime)
    monkeypatch.setattr(paper_classifier_onnx, "OnnxPaperClassifierModel", _Loaded)

    paper_classifier_common.load_paper_classifier("org/paper", "a" * 40, settings=object())

    _assert_path_only_at_debug(at_debug, "Paper classifier")


def test_the_section_classifier_load_logs_its_path_at_debug(monkeypatch, at_debug):
    from bibr.structure import section_classifier_common, section_classifier_onnx

    monkeypatch.setattr("bibr.utils.ml_runtime.resolve_runtime", _onnx_runtime)
    monkeypatch.setattr(section_classifier_onnx, "OnnxSectionClassifierModel", _Loaded)

    section_classifier_common.load_section_classifier("org/section", "a" * 40, settings=object())

    _assert_path_only_at_debug(at_debug, "Section classifier")


def test_the_reference_parser_load_logs_its_path_at_debug(monkeypatch, at_debug):
    from bibr.ner import parser_onnx, runtime

    monkeypatch.setattr("bibr.utils.ml_runtime.resolve_runtime", _onnx_runtime)
    monkeypatch.setattr(parser_onnx, "OnnxRefParser", _Loaded)

    runtime.load_ref_parser("org/parser", revision="a" * 40, settings=object())

    _assert_path_only_at_debug(at_debug, "NER parser")


def test_the_front_matter_tagger_load_logs_its_path_at_debug(monkeypatch, at_debug):
    from bibr.config import GlobalSettings
    from bibr.front_tagger import runtime

    settings = GlobalSettings()
    settings.ml.front_tagger_mode = "primary"
    settings.ml.front_tagger_model_id = "org/tagger"
    monkeypatch.setattr("bibr.utils.ml_runtime.find_onnx_bundle", lambda *a, **k: _BUNDLE)
    monkeypatch.setattr(runtime, "OnnxFrontTagger", _Loaded)
    monkeypatch.setattr(runtime, "_CACHE", {})

    assert runtime.load_front_tagger(settings) is not None

    _assert_path_only_at_debug(at_debug, "Front-matter tagger")
