"""The ONNX runtime against a hand-built graph whose emissions come from the features."""

from __future__ import annotations

import json
from pathlib import Path

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


def _recording_tagger(monkeypatch):
    """``OnnxFrontTagger`` swapped for a recording stand-in, so no session (or GPU) is made."""
    from bibr.front_tagger import runtime

    built: list[tuple] = []

    class _Tagger:
        def __init__(self, bundle, device=None):
            built.append((bundle, device))

    monkeypatch.setattr(runtime, "OnnxFrontTagger", _Tagger)
    monkeypatch.setattr("sys.platform", "linux")  # the device tests are not macOS ones
    runtime.reset_front_tagger_cache()
    return runtime, built


def _fake_loader(monkeypatch, tmp_path):
    """``load_front_tagger`` over a recording stand-in that finds ``tmp_path`` as the bundle."""
    from bibr.utils import ml_runtime

    runtime, built = _recording_tagger(monkeypatch)
    monkeypatch.setattr(ml_runtime, "find_onnx_bundle", lambda *a, **k: tmp_path)
    return runtime, built


def _tagger_settings(mode, model_id=None):
    """Settings naming the shipped tagger (another id is an override, which never falls back)."""
    from types import SimpleNamespace

    from bibr.config import MlOptions

    fields = MlOptions.model_fields
    revision = None if model_id else fields["front_tagger_revision"].default
    model_id = model_id or fields["front_tagger_model_id"].default

    return SimpleNamespace(
        ml=SimpleNamespace(
            front_tagger_mode=mode, front_tagger_model_id=model_id, front_tagger_revision=revision
        )
    )


@pytest.mark.parametrize("mode", ["primary", "only"])
def test_loader_reads_the_bundle_in_every_mode_that_uses_it(monkeypatch, tmp_path, mode):
    runtime, built = _fake_loader(monkeypatch, tmp_path)
    assert runtime.load_front_tagger(_tagger_settings(mode)) is not None
    assert built == [(tmp_path, None)]
    runtime.reset_front_tagger_cache()


def test_loader_gives_the_run_device_to_the_session(monkeypatch, tmp_path):
    runtime, built = _fake_loader(monkeypatch, tmp_path)
    runtime.load_front_tagger(_tagger_settings("only"), device="cpu")
    assert built == [(tmp_path, "cpu")]
    runtime.reset_front_tagger_cache()


def test_a_tagger_is_cached_per_device(monkeypatch, tmp_path):
    runtime, built = _fake_loader(monkeypatch, tmp_path)
    settings = _tagger_settings("only")
    on_cpu = runtime.load_front_tagger(settings, device="cpu")
    assert runtime.load_front_tagger(settings, device="cpu") is on_cpu
    assert runtime.load_front_tagger(settings, device="cuda") is not on_cpu
    assert [device for _bundle, device in built] == ["cpu", "cuda"]
    runtime.reset_front_tagger_cache()


def test_releasing_a_tagger_drops_only_the_cache_entries_that_hold_it(monkeypatch, tmp_path):
    runtime, built = _fake_loader(monkeypatch, tmp_path)
    settings = _tagger_settings("only")
    on_cpu = runtime.load_front_tagger(settings, device="cpu")
    on_gpu = runtime.load_front_tagger(settings, device="cuda")

    runtime.release_front_tagger(on_cpu)

    assert runtime.load_front_tagger(settings, device="cuda") is on_gpu
    assert len(built) == 2
    # The released one is read again, not handed back from the cache.
    assert runtime.load_front_tagger(settings, device="cpu") is not on_cpu
    assert len(built) == 3
    runtime.reset_front_tagger_cache()


def test_releasing_a_tagger_the_cache_never_held_is_harmless(monkeypatch, tmp_path):
    runtime, _built = _fake_loader(monkeypatch, tmp_path)
    runtime.release_front_tagger(object())


def test_macos_runs_the_tagger_on_the_cpu_provider(monkeypatch, tmp_path):
    runtime, built = _fake_loader(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.platform", "darwin")
    settings = _tagger_settings("only")
    tagger = runtime.load_front_tagger(settings)
    # Whatever device the run asked for, and one cache entry for all of them.
    assert runtime.load_front_tagger(settings, device="cuda") is tagger
    assert runtime.load_front_tagger(settings, device="auto") is tagger
    assert built == [(tmp_path, "cpu")]
    runtime.reset_front_tagger_cache()


def _hub_bundle(tmp_path):
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "bibr_onnx.json").write_text(json.dumps({"schema_version": 1}))
    (onnx_dir / "model.onnx").write_bytes(b"")
    return onnx_dir


def _hub_that_fails_with(monkeypatch, tmp_path, failure):
    """A Hub that raises ``failure`` while ``failure`` is set, then serves a bundle."""
    onnx_dir = _hub_bundle(tmp_path)
    calls: list[str] = []

    def download(repo_id, filename, revision=None):
        calls.append(filename)
        if failure:
            raise failure[0]
        return str(onnx_dir / filename.split("/")[-1])

    monkeypatch.setattr("bibr.utils.hf_cache.hf_download_or_cached", download)
    return calls


def _network_blips():
    from huggingface_hub.errors import LocalEntryNotFoundError

    return [ConnectionError("network is unreachable"), LocalEntryNotFoundError("offline")]


@pytest.mark.parametrize("blip", _network_blips(), ids=lambda exc: type(exc).__name__)
def test_a_network_blip_does_not_disable_the_tagger_for_the_process(monkeypatch, tmp_path, blip):
    runtime, built = _recording_tagger(monkeypatch)
    failure = [blip]
    calls = _hub_that_fails_with(monkeypatch, tmp_path, failure)
    settings = _tagger_settings("only")

    assert runtime.load_front_tagger(settings) is None
    failure.clear()  # the network is back
    assert runtime.load_front_tagger(settings) is not None
    assert len(built) == 1
    assert calls[0] == calls[1] == "onnx/bibr_onnx.json"  # the second paper looked again
    runtime.reset_front_tagger_cache()


def test_a_bundle_the_hub_does_not_have_is_looked_for_once(monkeypatch, tmp_path):
    runtime, built = _recording_tagger(monkeypatch)
    calls = _hub_that_fails_with(monkeypatch, tmp_path, [FileNotFoundError("no such file")])
    settings = _tagger_settings("only")

    assert runtime.load_front_tagger(settings) is None
    assert runtime.load_front_tagger(settings) is None
    assert calls == ["onnx/bibr_onnx.json"]
    assert built == []
    runtime.reset_front_tagger_cache()


def test_chunks_that_ask_together_read_the_bundle_once(monkeypatch, tmp_path):
    import threading
    import time

    runtime, built = _fake_loader(monkeypatch, tmp_path)
    reads = []

    def slow_find(*args, **kwargs):
        reads.append(args)
        time.sleep(0.2)
        return tmp_path

    monkeypatch.setattr("bibr.utils.ml_runtime.find_onnx_bundle", slow_find)
    settings = _tagger_settings("only")
    got = []
    threads = [
        threading.Thread(target=lambda: got.append(runtime.load_front_tagger(settings)))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(reads) == 1
    assert len(built) == 1
    assert len({id(tagger) for tagger in got}) == 1
    runtime.reset_front_tagger_cache()


def test_an_overridden_tagger_without_a_bundle_stops_the_run(tmp_path):
    from bibr.exceptions import ModelOverrideError
    from bibr.front_tagger import runtime

    runtime.reset_front_tagger_cache()
    with pytest.raises(ModelOverrideError, match="never falls back") as raised:
        runtime.load_front_tagger(_tagger_settings("only", model_id=str(tmp_path / "missing")))
    # Said once, not wrapped in itself.
    assert str(raised.value).count("names a front-matter tagger") == 1
    assert "ModelOverrideError" not in str(raised.value)
    runtime.reset_front_tagger_cache()


def test_a_network_blip_on_a_hub_override_stops_too(monkeypatch):
    import httpx

    from bibr.exceptions import ModelOverrideError
    from bibr.front_tagger import runtime

    runtime.reset_front_tagger_cache()

    def blip(*args, **kwargs):
        raise httpx.ConnectError("temporary failure in name resolution")

    monkeypatch.setattr("bibr.utils.ml_runtime.find_onnx_bundle", blip)
    settings = _tagger_settings("only", model_id="myorg/front-tagger-mirror")
    with pytest.raises(ModelOverrideError, match="the Hub could not be reached"):
        runtime.load_front_tagger(settings)
    runtime.reset_front_tagger_cache()


def test_an_overridden_tagger_is_recorded_by_checksum(tmp_path):
    import hashlib

    from bibr.front_tagger import runtime
    from bibr.utils.model_overrides import loaded_model_overrides, reset_loaded_model_overrides

    runtime.reset_front_tagger_cache()
    reset_loaded_model_overrides()
    _bundle(tmp_path)
    assert runtime.load_front_tagger(_tagger_settings("primary", model_id=str(tmp_path)))
    (record,) = loaded_model_overrides()
    assert record["setting"] == "ML_FRONT_TAGGER_MODEL_ID"
    model = next(f for f in record["files"] if f["path"].endswith("model.onnx"))
    assert model["sha256"] == hashlib.sha256(Path(model["path"]).read_bytes()).hexdigest()
    runtime.reset_front_tagger_cache()
    reset_loaded_model_overrides()
