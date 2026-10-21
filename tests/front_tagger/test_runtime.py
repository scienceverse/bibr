"""The ONNX runtime against a hand-built graph whose emissions come from the features."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bibr.front_tagger.contract import BIO_TAGS, CONTRACT_VERSION, FEATURE_KEYS

from .helpers import tagger_input

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def _bio_transitions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(BIO_TAGS)
    trans = np.zeros((n, n), dtype=np.float32)
    start = np.zeros(n, dtype=np.float32)
    for j, to_tag in enumerate(BIO_TAGS):
        if not to_tag.startswith("I-"):
            continue
        start[j] = -10000.0
        for i, from_tag in enumerate(BIO_TAGS):
            if from_tag[2:] != to_tag[2:]:
                trans[i, j] = -10000.0
    return start, np.zeros(n, dtype=np.float32), trans


def _bundle(tmp_path, *, max_length: int = 512):
    from onnx import TensorProto, helper, numpy_helper

    from tests.onnx_fixtures import tiny_tokenizer

    n_tags, n_feats = len(BIO_TAGS), len(FEATURE_KEYS)
    weights = np.zeros((n_feats, n_tags), dtype=np.float32)
    title_col = FEATURE_KEYS.index("label_doc_title")
    text_col = FEATURE_KEYS.index("label_text")
    weights[title_col, BIO_TAGS.index("B-TITLE")] = 5.0
    weights[title_col, BIO_TAGS.index("I-TITLE")] = 6.0
    weights[text_col, BIO_TAGS.index("O")] = 5.0
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["token_features", "W"], ["emissions"])],
        "front_tagger",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["B", "L"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["B", "L"]),
            helper.make_tensor_value_info("token_features", TensorProto.FLOAT, ["B", "L", n_feats]),
        ],
        [helper.make_tensor_value_info("emissions", TensorProto.FLOAT, ["B", "L", n_tags])],
        [numpy_helper.from_array(weights, "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    bundle = tmp_path / "onnx"
    bundle.mkdir(parents=True)
    onnx.save(model, bundle / "model.onnx")
    tiny_tokenizer().save(str(bundle / "tokenizer.json"))
    start, end, trans = _bio_transitions()
    manifest = {
        "schema_version": 1,
        "model_file": "model.onnx",
        "tokenizer_file": "tokenizer.json",
        "pad_token": "[PAD]",
        "pad_token_id": 0,
        "max_length": max_length,
        "bio_tags": BIO_TAGS,
        "feature_keys": list(FEATURE_KEYS),
        "contract_version": CONTRACT_VERSION,
        "crf": {
            "start_transitions": start.tolist(),
            "end_transitions": end.tolist(),
            "transitions": trans.tolist(),
        },
        "files": ["model.onnx", "tokenizer.json"],
    }
    (bundle / "bibr_onnx.json").write_text(json.dumps(manifest))
    return bundle


def test_tags_follow_the_emissions_and_bio_constraints(tmp_path):
    from bibr.front_tagger.runtime import OnnxFrontTagger

    tagger = OnnxFrontTagger(_bundle(tmp_path))
    inp = tagger_input(("doc_title", "memory and consciousness"), ("text", "journal of science"))
    tags, conf = tagger.tag(inp)
    assert tags == ["B-TITLE", "I-TITLE", "I-TITLE", "O", "O", "O"]
    assert all(c > 0.5 for c in conf)


def test_windows_decode_like_one_pass(tmp_path):
    from bibr.front_tagger.runtime import OnnxFrontTagger

    inp = tagger_input(
        ("doc_title", "memory and consciousness of science"),
        ("text", "the participants were recruited early and late"),
    )
    whole = OnnxFrontTagger(_bundle(tmp_path / "a", max_length=512)).tag(inp)
    windowed = OnnxFrontTagger(_bundle(tmp_path / "b", max_length=4)).tag(inp)
    assert whole[0] == windowed[0]
    np.testing.assert_allclose(whole[1], windowed[1], rtol=1e-5)


def test_a_bundle_from_another_contract_is_refused(tmp_path):
    from bibr.exceptions import ConfigurationError
    from bibr.front_tagger.runtime import OnnxFrontTagger

    bundle = _bundle(tmp_path)
    manifest = json.loads((bundle / "bibr_onnx.json").read_text())
    manifest["feature_keys"] = manifest["feature_keys"][:-1]
    (bundle / "bibr_onnx.json").write_text(json.dumps(manifest))
    with pytest.raises(ConfigurationError, match="input contract"):
        OnnxFrontTagger(bundle)


def test_loader_is_off_by_default():
    from types import SimpleNamespace

    from bibr.front_tagger.runtime import load_front_tagger, reset_front_tagger_cache

    reset_front_tagger_cache()
    settings = SimpleNamespace(
        ml=SimpleNamespace(front_tagger_mode="off", front_tagger_model_id="x/y")
    )
    assert load_front_tagger(settings) is None


def test_loader_reads_a_local_bundle(tmp_path):
    from types import SimpleNamespace

    from bibr.front_tagger.runtime import (
        OnnxFrontTagger,
        load_front_tagger,
        reset_front_tagger_cache,
    )

    reset_front_tagger_cache()
    _bundle(tmp_path)
    settings = SimpleNamespace(
        ml=SimpleNamespace(
            front_tagger_mode="primary",
            front_tagger_model_id=str(tmp_path),
            front_tagger_revision=None,
        )
    )
    assert isinstance(load_front_tagger(settings), OnnxFrontTagger)
    reset_front_tagger_cache()


def test_words_inside_a_merged_token_continue_its_tag():
    from bibr.front_tagger.runtime import _word_tags, _words_of_tokens

    # "Smith," as one token, then CJK "山田太郎" as two tokens ("山田", "太郎").
    inp = tagger_input(("text", "Smith, 山田太郎"))
    assert [w.text for w in inp.words] == ["Smith", ",", "山", "田", "太", "郎"]
    words_of = _words_of_tokens([(0, 6), (7, 9), (9, 11)], inp)
    assert words_of == [[0, 1], [2, 3], [4, 5]]
    tags, conf = _word_tags(
        words_of, ["B-FAMILY", "B-FAMILY", "B-GIVEN"], [0.9, 0.8, 0.7], len(inp.words)
    )
    assert tags == ["B-FAMILY", "I-FAMILY", "B-FAMILY", "I-FAMILY", "B-GIVEN", "I-GIVEN"]
    assert conf == [0.9, 0.9, 0.8, 0.8, 0.7, 0.7]


def test_a_word_a_token_starts_in_keeps_that_tokens_tag():
    from bibr.front_tagger.runtime import _word_tags, _words_of_tokens

    # Tokens "ab c", "d", "ef": "cd" is covered by "ab c" but its own token is
    # "d", the first token whose first word it is (training labels "d" with it).
    inp = tagger_input(("text", "ab cd ef"))
    words_of = _words_of_tokens([(0, 4), (4, 5), (6, 8)], inp)
    assert words_of == [[0, 1], [1], [2]]
    tags, _ = _word_tags(words_of, ["O", "O", "B-TITLE"], [1.0, 1.0, 0.6], 3)
    assert tags == ["O", "O", "B-TITLE"]
    tags, conf = _word_tags(words_of, ["B-TITLE", "B-ABSTRACT", "I-ABSTRACT"], [0.9, 0.5, 0.6], 3)
    assert tags == ["B-TITLE", "B-ABSTRACT", "I-ABSTRACT"]
    assert conf == [0.9, 0.5, 0.6]
