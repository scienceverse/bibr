"""The load probes of the ONNX reference parser, front-matter tagger and classifiers.

An overridden model runs once at load on the input a paper sends, and its
output shape is checked against its manifest; a graph that opens but fails
that probe stops the run at the preflight or at its first load. A shipped
default is not probed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from bibr.config import snapshot_settings
from bibr.exceptions import ConfigurationError, ModelOverrideError
from bibr.utils.model_overrides import check_model_overrides, reset_loaded_model_overrides

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

BUNDLES = Path(__file__).parent / "fixtures" / "onnx"


@pytest.fixture(autouse=True)
def _fresh():
    from bibr.extract.ref_extractor import unload_ner_parser
    from bibr.front_tagger.runtime import reset_front_tagger_cache

    reset_loaded_model_overrides()
    unload_ner_parser()
    reset_front_tagger_cache()
    yield
    reset_loaded_model_overrides()
    unload_ner_parser()
    reset_front_tagger_cache()


def _copy(name: str, dest: Path) -> Path:
    shutil.copytree(BUNDLES / name / "onnx", dest / "onnx")
    return dest


def _edit_manifest(bundle: Path, **changes) -> None:
    path = bundle / "onnx" / "bibr_onnx.json"
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    path.write_text(json.dumps(manifest))


def _manifest(bundle: Path) -> dict:
    return json.loads((bundle / "onnx" / "bibr_onnx.json").read_text())


def _grow_vocabulary(bundle: Path, extra: int = 40) -> None:
    """A tokenizer with more ids than the graph's embedding table has rows."""
    path = bundle / "onnx" / "tokenizer.json"
    data = json.loads(path.read_text())
    vocab = data["model"]["vocab"]
    for i in range(extra):
        vocab[f"extra{i}"] = len(vocab)
    path.write_text(json.dumps(data))


def _settings(**ml):
    settings = snapshot_settings()
    settings.ml.runtime = "onnx"
    for name, value in ml.items():
        setattr(settings.ml, name, value)
    return settings


# -- the reference parser -----------------------------------------------------------------


def _parser_settings(bundle: Path):
    settings = _settings()
    settings.NER_PARSER_CKPT = str(bundle)
    settings.NER_PARSER_REVISION = None
    return settings


def test_an_overridden_parser_that_runs_on_a_full_window_loads(tmp_path):
    from bibr.extract.ref_extractor import _get_ner_parser

    bundle = _copy("ner", tmp_path / "parser-candidate")
    parser = _get_ner_parser(_parser_settings(bundle))
    assert parser.parse_batch(["Smith, J. (2019). A title. Journal, 1, 2-3."])


def test_a_parser_whose_emissions_miss_a_manifest_tag_stops_at_load(tmp_path):
    from bibr.extract.ref_extractor import _get_ner_parser

    bundle = _copy("ner", tmp_path / "parser-candidate")
    manifest = _manifest(bundle)
    n = len(manifest["bio_tags"]) + 1
    _edit_manifest(
        bundle,
        bio_tags=[*manifest["bio_tags"], "B-EXTRA"],
        crf={
            "start_transitions": [0.0] * n,
            "end_transitions": [0.0] * n,
            "transitions": [[0.0] * n for _ in range(n)],
        },
    )
    settings = _parser_settings(bundle)
    with pytest.raises(ModelOverrideError, match=rf"emissions of shape .* want \(4, \d+, {n}\)"):
        _get_ner_parser(settings)
    (problem,) = check_model_overrides(settings)[0]
    assert problem.startswith("NER_PARSER_CKPT=") and "bio_tags" in problem


def test_a_parser_crf_of_the_wrong_size_stops_at_load(tmp_path):
    from bibr.extract.ref_extractor import _get_ner_parser

    bundle = _copy("ner", tmp_path / "parser-candidate")
    crf = _manifest(bundle)["crf"]
    _edit_manifest(bundle, crf={**crf, "end_transitions": crf["end_transitions"][:-1]})
    with pytest.raises(ModelOverrideError, match="crf.end_transitions"):
        _get_ner_parser(_parser_settings(bundle))


def test_a_parser_graph_smaller_than_its_tokenizer_stops_at_load(tmp_path):
    from bibr.extract.ref_extractor import _get_ner_parser

    bundle = _copy("ner", tmp_path / "parser-candidate")
    _grow_vocabulary(bundle)
    with pytest.raises(ModelOverrideError, match="the graph does not run on"):
        _get_ner_parser(_parser_settings(bundle))


def test_the_shipped_parser_is_not_probed(tmp_path, monkeypatch):
    from bibr.ner.parser_onnx import OnnxRefParser
    from bibr.ner.runtime import load_ref_parser

    def probe(self):
        raise AssertionError("a default model is not probed")

    monkeypatch.setattr(OnnxRefParser, "check_graph", probe)
    monkeypatch.setattr("bibr.utils.model_overrides.is_model_override", lambda s, env: False)
    bundle = _copy("ner", tmp_path / "parser")
    assert load_ref_parser(str(bundle), settings=_parser_settings(bundle)) is not None


# -- the section and paper classifiers -------------------------------------------------------


def _load_section(bundle: Path):
    from bibr.structure.section_classifier_common import load_section_classifier

    settings = _settings(section_classifier_model_id=str(bundle))
    return load_section_classifier(str(bundle), revision="main", device="cpu", settings=settings)


def _load_paper(bundle: Path):
    from bibr.structure.paper_classifier_common import load_paper_classifier

    settings = _settings(paper_classifier_model_id=str(bundle))
    return load_paper_classifier(str(bundle), revision="main", device="cpu", settings=settings)


def test_overridden_classifiers_that_run_load(tmp_path):
    assert _load_section(_copy("section", tmp_path / "section")).label_classes
    assert _load_paper(_copy("paper", tmp_path / "paper")).l1_classes


def test_a_section_classifier_with_another_label_count_is_refused(tmp_path):
    bundle = _copy("section", tmp_path / "section")
    labels = _manifest(bundle)["label_classes"]
    _edit_manifest(bundle, label_classes=[*labels, "references"])
    with pytest.raises(ConfigurationError, match=r"type_logits of shape \(3, 5\).*want \(3, 6\)"):
        _load_section(bundle)


def test_a_section_classifier_naming_an_unknown_section_type_is_refused(tmp_path):
    bundle = _copy("section", tmp_path / "section")
    labels = _manifest(bundle)["label_classes"]
    _edit_manifest(bundle, label_classes=[*labels[:-1], "not-a-section"])
    with pytest.raises(ConfigurationError, match="section types bibr does not know"):
        _load_section(bundle)


def test_a_paper_classifier_with_another_label_count_is_refused(tmp_path):
    bundle = _copy("paper", tmp_path / "paper")
    _edit_manifest(bundle, l2_classes=[*_manifest(bundle)["l2_classes"], "Law"])
    with pytest.raises(ConfigurationError, match=r"l2_logits of shape \(2, 3\).*want \(2, 4\)"):
        _load_paper(bundle)


def test_a_paper_classifier_with_a_zero_temperature_is_refused(tmp_path):
    bundle = _copy("paper", tmp_path / "paper")
    _edit_manifest(bundle, paper_type_temperature=0.0)
    with pytest.raises(ConfigurationError, match="paper_type_temperature"):
        _load_paper(bundle)


def test_the_shipped_classifiers_are_not_probed(tmp_path, monkeypatch):
    from bibr.structure.paper_classifier_onnx import OnnxPaperClassifierModel
    from bibr.structure.section_classifier_onnx import OnnxSectionClassifierModel

    def probe(self):
        raise AssertionError("a default model is not probed")

    monkeypatch.setattr(OnnxSectionClassifierModel, "check_graph", probe)
    monkeypatch.setattr(OnnxPaperClassifierModel, "check_graph", probe)
    monkeypatch.setattr("bibr.utils.model_overrides.is_model_override", lambda s, env: False)
    assert _load_section(_copy("section", tmp_path / "section")) is not None
    assert _load_paper(_copy("paper", tmp_path / "paper")) is not None


def test_a_failed_classifier_probe_stops_the_preflight_and_the_lazy_load(tmp_path, monkeypatch):
    from bibr.structure import paper_classifier, section_classifier

    section = _copy("section", tmp_path / "section")
    _edit_manifest(section, label_classes=[*_manifest(section)["label_classes"], "references"])
    paper = _copy("paper", tmp_path / "paper")
    _edit_manifest(paper, l1_classes=["Natural Sciences"])
    settings = _settings(
        section_classifier_model_id=str(section), paper_classifier_model_id=str(paper)
    )
    problems, _warnings = check_model_overrides(settings)
    assert [line.split("=")[0] for line in problems] == [
        "ML_SECTION_CLASSIFIER_MODEL_ID",
        "ML_PAPER_CLASSIFIER_MODEL_ID",
    ]
    # The library path that loads without the pipeline's classifier resources.
    monkeypatch.setattr(section_classifier, "_model_cache", None)
    monkeypatch.setattr(paper_classifier, "_paper_model_cache", None)
    with pytest.raises(ModelOverrideError, match="ML_SECTION_CLASSIFIER_MODEL_ID"):
        section_classifier._get_trained_model(settings)
    with pytest.raises(ModelOverrideError, match="ML_PAPER_CLASSIFIER_MODEL_ID"):
        paper_classifier._get_paper_model(settings)
    assert section_classifier._model_cache is None  # nothing cached for a failure
    assert paper_classifier._paper_model_cache is None


# -- the front-matter tagger -----------------------------------------------------------------


def _tagger_settings(bundle: Path):
    return _settings(
        front_tagger_mode="primary", front_tagger_model_id=str(bundle), front_tagger_revision=None
    )


def _positional_tagger(tmp_path: Path, *, positions: int, max_length: int) -> Path:
    """The test tagger plus a per-position bias that exists for ``positions`` tokens only."""
    from onnx import TensorProto, helper, numpy_helper

    from bibr.front_tagger.contract import BIO_TAGS, FEATURE_KEYS
    from tests.front_tagger.test_runtime import _bundle

    bundle = _bundle(tmp_path, max_length=max_length)
    n_tags, n_feats = len(BIO_TAGS), len(FEATURE_KEYS)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["token_features", "W"], ["scores"]),
            helper.make_node("Add", ["scores", "P"], ["emissions"]),
        ],
        "front_tagger",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["B", "L"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["B", "L"]),
            helper.make_tensor_value_info("token_features", TensorProto.FLOAT, ["B", "L", n_feats]),
        ],
        [helper.make_tensor_value_info("emissions", TensorProto.FLOAT, ["B", "L", n_tags])],
        [
            numpy_helper.from_array(np.zeros((n_feats, n_tags), np.float32), "W"),
            numpy_helper.from_array(np.zeros((1, positions, n_tags), np.float32), "P"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, bundle / "model.onnx")
    return tmp_path


def test_an_overridden_tagger_that_runs_on_a_full_window_loads(tmp_path):
    from bibr.front_tagger.runtime import load_front_tagger

    bundle = _positional_tagger(tmp_path, positions=256, max_length=256)
    assert load_front_tagger(_tagger_settings(bundle)) is not None


def test_a_tagger_that_fails_on_its_longest_window_stops_at_load(tmp_path):
    from bibr.front_tagger.runtime import load_front_tagger

    # It would run on a short front page, and fail on the first paper whose
    # first two pages fill a window.
    bundle = _positional_tagger(tmp_path, positions=256, max_length=1024)
    settings = _tagger_settings(bundle)
    with pytest.raises(ModelOverrideError, match=r"does not run on a \(1, 1024\) window"):
        load_front_tagger(settings)
    (problem,) = check_model_overrides(settings)[0]
    assert problem.startswith("ML_FRONT_TAGGER_MODEL_ID=")


def test_a_tagger_crf_of_the_wrong_size_stops_at_load(tmp_path):
    from bibr.front_tagger.runtime import load_front_tagger
    from tests.front_tagger.test_runtime import _bundle

    bundle = _bundle(tmp_path)
    manifest = json.loads((bundle / "bibr_onnx.json").read_text())
    manifest["crf"]["start_transitions"] = manifest["crf"]["start_transitions"][:-1]
    (bundle / "bibr_onnx.json").write_text(json.dumps(manifest))
    with pytest.raises(ModelOverrideError, match="crf.start_transitions"):
        load_front_tagger(_tagger_settings(tmp_path))


def test_the_shipped_tagger_is_not_probed(tmp_path, monkeypatch):
    from bibr.front_tagger.runtime import OnnxFrontTagger, load_front_tagger

    def probe(self):
        raise AssertionError("a default model is not probed")

    monkeypatch.setattr(OnnxFrontTagger, "check_graph", probe)
    monkeypatch.setattr("bibr.utils.model_overrides.is_model_override", lambda s, env: False)
    bundle = _positional_tagger(tmp_path, positions=256, max_length=1024)
    assert load_front_tagger(_tagger_settings(bundle)) is not None
