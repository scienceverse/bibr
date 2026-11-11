"""The opt-in text reference segmenter (``REF_TEXT_SEG_MODEL_ID``) and its place in the cascade.

The bundles here are built in the test: a word-level ``tokenizer.json`` and a
small ONNX graph whose logits come from a per-token table plus a term that grows
with the token's position in its window, so overlapping windows disagree and
the merge decides.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from bibr.config import snapshot_settings
from bibr.exceptions import ConfigurationError, ModelOverrideError
from bibr.extract import ref_extractor, text_segmenter
from bibr.extract.ref_extractor import ReferenceExtractor
from bibr.extract.text_segmenter import (
    TAGS,
    OnnxTextSegmenter,
    merge_windows,
    read_segmenter_manifest,
    windows,
)
from bibr.paper_contents import PaperContents
from bibr.processing_warnings import WarningCode
from bibr.utils.model_overrides import (
    begin_run,
    check_model_overrides,
    loaded_model_overrides,
    models_in_use,
    reset_loaded_model_overrides,
    run_records,
)

onnx = pytest.importorskip("onnx")

# "@" opens a reference, "~" is outside one, every other word is inside one.
_VOCAB = {"[UNK]": 0, "@": 1, "~": 2}
_REFS = (
    "@ Aknin, L. B. (2013). Prosocial spending. Journal of X, 1, 2-3.",
    "@ Carstensen, L. L. (2009). Another title here. Journal of Y, 4, 5-6.",
    "@ Smith, J. (2010). A third title. Journal of Z, 7, 8-9.",
)
_REF_TEXT = "\n".join(_REFS)


def _write_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel(vocab=dict(_VOCAB), unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(path))


def _table(b_logit=8.0) -> np.ndarray:
    """Per-token logits: "@" -> B-REF, "~" -> O, anything else -> I-REF."""
    table = np.zeros((len(_VOCAB), len(TAGS)), np.float32)
    table[0] = (0.0, 0.0, 4.0)
    table[1] = (0.0, b_logit, 0.0)
    table[2] = (4.0, 0.0, 0.0)
    return table


def _write_graph(path: Path, *, table=None, slope=(0.0, 0.0, 0.0), n_out=3) -> None:
    """``logits = table[input_ids] + position_in_window * slope`` (positions count from 1)."""
    from onnx import TensorProto, helper, numpy_helper

    table = _table() if table is None else np.asarray(table, np.float32)
    nodes = [
        helper.make_node("Gather", ["table", "input_ids"], ["emb"], axis=0),
        helper.make_node("CumSum", ["attention_mask", "axis1"], ["pos_i"]),
        helper.make_node("Cast", ["pos_i"], ["pos"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["pos", "axis_last"], ["pos3"]),
        helper.make_node("Mul", ["pos3", "slope"], ["ramp"]),
        helper.make_node("Add", ["emb", "ramp"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_text_tagger",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "t"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "t"]),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", "t", n_out])],
        initializer=[
            numpy_helper.from_array(table, "table"),
            numpy_helper.from_array(np.asarray(1, np.int64), "axis1"),
            numpy_helper.from_array(np.asarray([-1], np.int64), "axis_last"),
            numpy_helper.from_array(np.asarray(slope, np.float32), "slope"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def _bundle(
    root: Path, *, window=1024, stride=768, in_onnx_dir=False, manifest=None, **graph
) -> Path:
    bundle = root / "onnx" if in_onnx_dir else root
    bundle.mkdir(parents=True, exist_ok=True)
    _write_graph(bundle / "model.onnx", **graph)
    _write_tokenizer(bundle / "tokenizer.json")
    meta = {
        "tags": list(TAGS),
        "window": window,
        "stride": stride,
        "add_special_tokens": False,
        "decode": "argmax per token; trust merge across windows",
    }
    (bundle / "segmenter.json").write_text(json.dumps({**meta, **(manifest or {})}))
    return root


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    reset_loaded_model_overrides()
    monkeypatch.setattr(ref_extractor, "_TEXT_SEGMENTER", None)
    monkeypatch.setattr(ref_extractor, "_TEXT_SEGMENTER_KEY", None)
    yield
    reset_loaded_model_overrides()


# -- the training-side procedure, written out as plainly as it was specified ----------


def _reference_segments(text, ids, offsets, logits_fn, window, stride):
    """Windows, argmax, trust merge and decode, one loop at a time."""
    n = len(ids)
    tags, trust = [0] * n, [-1] * n
    start = 0
    while start < n:
        end = min(start + window, n)
        pred = logits_fn(ids[start:end]).argmax(-1)
        for i, t in enumerate(pred.tolist()):
            tr = min(i, end - start - 1 - i)
            if tr > trust[start + i]:
                tags[start + i], trust[start + i] = t, tr
        if end >= n:
            break
        start += stride
    starts = []
    for (a, b), t in zip(offsets, tags, strict=True):
        if t != 1 or a == b:
            continue
        s = a
        while s < b and text[s].isspace():
            s += 1
        if not starts or s > starts[-1]:
            starts.append(s)
    first = len(text) - len(text.lstrip())
    if first < len(text) and (not starts or starts[0] > first):
        starts = [first, *starts]
    out = []
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else len(text)
        while e > s and text[e - 1].isspace():
            e -= 1
        if text[s:e].strip():
            out.append(text[s:e].strip())
    return tags, out


# -- windows and merge ----------------------------------------------------------------


def test_windows_step_by_stride_until_one_reaches_the_end():
    assert windows(0, 4, 3) == []
    assert windows(4, 4, 3) == [(0, 4)]
    assert windows(10, 4, 3) == [(0, 4), (3, 7), (6, 10)]
    assert windows(11, 4, 3) == [(0, 4), (3, 7), (6, 10), (9, 11)]
    assert windows(5, 1024, 768) == [(0, 5)]


def test_merge_keeps_each_token_from_its_most_central_window():
    def onehot(tags):
        out = np.full((len(tags), 3), -5.0)
        out[np.arange(len(tags)), tags] = 5.0
        return out

    # Window A covers tokens 0-3, window B tokens 2-5. Token 2 has trust 1 in A
    # and 0 in B (A wins); token 3 has trust 0 in A and 1 in B (B wins).
    tags, probs = merge_windows(6, [(0, 4, onehot([1, 2, 2, 2])), (2, 6, onehot([0, 0, 0, 1]))])
    assert tags.tolist() == [1, 2, 2, 0, 0, 1]
    assert probs.shape == (6, 3) and np.allclose(probs.sum(axis=1), 1.0)
    # A tie in trust keeps the first window: tokens 0 and 1 have trust 0 and 1 in
    # both; token 2 has trust 0 in the first and 1 in the second.
    tags, _ = merge_windows(4, [(0, 3, onehot([2, 2, 2])), (0, 4, onehot([0, 0, 0, 0]))])
    assert tags.tolist() == [2, 2, 0, 0]


def test_a_text_longer_than_one_window_matches_the_training_procedure(tmp_path):
    # Window 5, stride 3: the B-REF logit grows with the position in the window,
    # so a word read late in one window and early in the next gets two answers.
    bundle = _bundle(
        tmp_path / "b",
        window=5,
        stride=3,
        table=_table(b_logit=0.0),
        slope=(0.0, 1.0, 0.0),
    )
    seg = OnnxTextSegmenter(bundle)
    text = _REF_TEXT + "\n~ trailing words outside any reference ~"
    ids, offsets = seg.encode(text)
    assert len(ids) > 5
    expected_tags, expected = _reference_segments(
        text, ids, offsets, lambda w: seg._run(w)[0], 5, 3
    )
    result = seg.segment_text(text)
    assert result.windows == len(windows(len(ids), 5, 3)) > 1
    assert result.tags.tolist() == expected_tags
    assert result.segments == expected
    # The merge matters here: a single pass over the whole text tags differently.
    whole = seg._run(ids)[0].argmax(-1).tolist()
    assert whole != expected_tags


def test_references_open_at_b_ref_and_run_to_the_next(tmp_path):
    seg = OnnxTextSegmenter(_bundle(tmp_path / "b"))
    result = seg.segment_text(_REF_TEXT)
    assert result.segments == list(_REFS)
    assert result.starts == [_REF_TEXT.index(r) for r in _REFS]
    flags = result.receipt_flags()
    assert "starts=3" in flags and "windows=1" in flags
    assert any(f.startswith("expected_entry_precision=") for f in flags)
    assert any(f.startswith("start_prob_min=") for f in flags)


def test_o_tags_never_end_a_reference(tmp_path):
    # "~" is tagged O; the decode only opens references, so it stays in the one before.
    seg = OnnxTextSegmenter(_bundle(tmp_path / "b"))
    text = "@ First ref ~ ~ still first\n@ Second ref"
    assert seg.segment(text) == ["@ First ref ~ ~ still first", "@ Second ref"]


def test_an_empty_text_gives_nothing(tmp_path):
    seg = OnnxTextSegmenter(_bundle(tmp_path / "b"))
    for text in ("", "   \n\t "):
        result = seg.segment_text(text)
        assert result.segments == [] and result.tokens == 0 and result.windows == 0


def test_a_text_with_no_reference_opening_is_one_reference(tmp_path):
    # No B-REF anywhere: the first non-space character is still a start, as in training.
    seg = OnnxTextSegmenter(_bundle(tmp_path / "b"))
    text = "  ~ no opening marker anywhere here ~  "
    result = seg.segment_text(text)
    assert result.segments == [text.strip()]
    assert result.starts == [2]
    assert len(result.start_probs) == 0


def test_lone_surrogates_are_read_as_replacement_characters(tmp_path):
    seg = OnnxTextSegmenter(_bundle(tmp_path / "b"))
    out = seg.segment("@ Bad \ud800 char ref\n@ Next ref")
    assert out == ["@ Bad � char ref", "@ Next ref"]


# -- the bundle -----------------------------------------------------------------------


@pytest.mark.parametrize("in_onnx_dir", [False, True])
def test_bundle_layouts(tmp_path, in_onnx_dir):
    seg = OnnxTextSegmenter(_bundle(tmp_path / "b", in_onnx_dir=in_onnx_dir))
    assert [p.name for p in seg.files] == ["model.onnx", "segmenter.json", "tokenizer.json"]
    assert (seg.window, seg.stride) == (1024, 768)


def test_bundle_problems_are_named(tmp_path):
    with pytest.raises(ConfigurationError, match="not a directory"):
        OnnxTextSegmenter(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigurationError, match="lacks model.onnx, segmenter.json, tokenizer"):
        OnnxTextSegmenter(tmp_path / "empty")
    partial = _bundle(tmp_path / "partial")
    (partial / "tokenizer.json").unlink()
    with pytest.raises(ConfigurationError, match="lacks tokenizer.json"):
        OnnxTextSegmenter(partial)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"tags": ["O", "B", "I"]}, "tags"),
        ({"tags": ["B-REF", "I-REF", "O"]}, "tags"),
        ({"window": 0}, "window"),
        ({"stride": 2000}, "stride"),
        ({"stride": 1.5}, "stride"),
        ({"add_special_tokens": True}, "add_special_tokens"),
    ],
)
def test_unusable_manifests_are_refused(tmp_path, manifest, message):
    bundle = _bundle(tmp_path / "b", manifest=manifest)
    with pytest.raises(ConfigurationError, match=message):
        read_segmenter_manifest(bundle / "segmenter.json")
    with pytest.raises(ConfigurationError, match=message):
        OnnxTextSegmenter(bundle)


def test_a_graph_of_the_wrong_shape_is_refused_at_load(tmp_path):
    bundle = _bundle(tmp_path / "wide")
    # Two logits per token instead of three.
    _write_graph(bundle / "model.onnx", table=np.zeros((3, 2), np.float32), slope=(0, 0), n_out=2)
    with pytest.raises(ConfigurationError, match=r"shape \(1, 1024, 2\)"):
        OnnxTextSegmenter(bundle)
    (bundle / "model.onnx").write_bytes(b"not a graph")
    with pytest.raises(Exception):  # noqa: B017 - onnxruntime's own error
        OnnxTextSegmenter(bundle)


def test_a_hub_id_is_downloaded_at_its_revision(tmp_path, monkeypatch):
    cached = _bundle(tmp_path / "cache", in_onnx_dir=True)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "org" / "seg").mkdir(parents=True)  # never read: a Hub id is a Hub id
    calls = []

    def download(repo_id, filename, revision=None):
        calls.append((repo_id, filename, revision))
        return str(cached / filename)

    monkeypatch.setattr("bibr.utils.hf_cache.hf_download_or_cached", download)
    seg = OnnxTextSegmenter("org/seg", "0123abcd")
    assert calls == [
        ("org/seg", "onnx/segmenter.json", "0123abcd"),
        ("org/seg", "onnx/tokenizer.json", "0123abcd"),
        ("org/seg", "onnx/model.onnx", "0123abcd"),
    ]
    assert seg.onnx_path == cached / "onnx" / "model.onnx"


# -- the tier ---------------------------------------------------------------------------


def _extractor(monkeypatch, model: str | None, *, seg="crf"):
    contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={})
    ex = ReferenceExtractor(
        contents, file_hash="h", llm_client=MagicMock(), seg_strategy=seg, parse_strategy="ner"
    )
    ex.llm_client.segment_references = AsyncMock(return_value=[])
    monkeypatch.setattr(ex._settings, "REF_TEXT_SEG_MODEL_ID", model)
    return ex


def _attempts(ex):
    return [(a.strategy, a.selected, a.reason_flags[:1]) for a in ex._segmentation_attempts]


def _no_crf(*args, **kwargs):
    raise AssertionError("the CRF segmenter was loaded")


async def test_the_tier_takes_the_crf_place_and_the_crf_is_not_loaded(tmp_path, monkeypatch):
    ex = _extractor(monkeypatch, str(_bundle(tmp_path / "b")))
    with patch("bibr.extract.ref_extractor._get_ner_segmenter", side_effect=_no_crf):
        out = await ex._segment_references(_REF_TEXT, "crf")
    assert out == list(_REFS)
    assert _attempts(ex) == [("text_tagger", True, (f"tokens={len(_REF_TEXT.split())}",))]
    (attempt,) = ex._segmentation_attempts
    assert attempt.spans == tuple((_REF_TEXT.index(r), _REF_TEXT.index(r) + len(r)) for r in _REFS)
    assert "starts=3" in attempt.reason_flags
    assert not ex.contents.processing_warnings


async def test_the_tier_is_the_last_resort_of_the_geom_cascade(tmp_path, monkeypatch):
    ex = _extractor(monkeypatch, str(_bundle(tmp_path / "b")), seg="geom")
    monkeypatch.setattr(ex._settings, "REF_SEG_LLM_FALLBACK", False)
    with patch("bibr.extract.ref_extractor._get_ner_segmenter", side_effect=_no_crf):
        out = await ex._segment_references(_REF_TEXT, "geom")
    assert out == list(_REFS)
    assert [a.strategy for a in ex._segmentation_attempts][-1] == "text_tagger"
    assert "geom" in [a.strategy for a in ex._segmentation_attempts]


async def test_without_the_setting_the_crf_runs_as_before(monkeypatch):
    ex = _extractor(monkeypatch, None)
    crf = MagicMock()
    crf.segment.return_value = list(_REFS)
    with (
        patch.object(text_segmenter, "OnnxTextSegmenter", side_effect=AssertionError("built")),
        patch("bibr.extract.ref_extractor._get_ner_segmenter", return_value=crf),
    ):
        out = await ex._segment_references(_REF_TEXT, "crf")
    assert out == list(_REFS)
    assert [(a.strategy, a.selected, a.reason_flags) for a in ex._segmentation_attempts] == [
        ("crf", True, ())
    ]
    assert loaded_model_overrides() == []


async def test_an_empty_answer_goes_on_to_the_marker_split(tmp_path, monkeypatch):
    ex = _extractor(monkeypatch, str(_bundle(tmp_path / "b")))
    assert await ex._segment_references("   ", "crf") == []
    assert _attempts(ex) == [("text_tagger", False, ("no_segments",))]


async def test_an_error_on_one_paper_declines_to_the_next_tier(tmp_path, monkeypatch, caplog):
    text = "\n".join(f"[{i}] Author {i}. A title number {i}. Journal, 2010." for i in range(1, 8))
    ex = _extractor(monkeypatch, str(_bundle(tmp_path / "b")))
    with (
        patch.object(OnnxTextSegmenter, "segment_text", side_effect=RuntimeError("bad input")),
        caplog.at_level("WARNING"),
    ):
        out = await ex._segment_references(text, "crf")
    assert out and all(s.startswith("[") for s in out)
    assert [(a.strategy, a.selected, a.reason_flags) for a in ex._segmentation_attempts] == [
        ("text_tagger", False, ("segmentation_error", "error=RuntimeError")),
        ("marker_split", True, ()),
    ]
    codes = [w.code for w in ex.contents.processing_warnings]
    assert WarningCode.REF_SEG_CRF_ERROR in codes
    assert "text reference segmenter failed on this paper" in caplog.text


async def test_a_network_failure_on_a_hub_id_stops(monkeypatch):
    import httpx

    def offline(repo_id, filename, revision=None):
        raise httpx.ConnectError("name resolution")

    monkeypatch.setattr("bibr.utils.hf_cache.hf_download_or_cached", offline)
    ex = _extractor(monkeypatch, "org/unreachable-seg")
    begin_run(ex._settings)
    with (
        patch("bibr.extract.ref_extractor._get_ner_segmenter", side_effect=_no_crf),
        pytest.raises(ModelOverrideError, match="the Hub could not be reached"),
    ):
        await ex._segment_references(_REF_TEXT, "crf")
    assert not ex._segmentation_attempts  # no tier answered in its place
    (record,) = run_records()
    assert record["loaded"] is False
    assert record["reason"].startswith("the Hub could not be reached")
    assert ref_extractor._TEXT_SEGMENTER is None


async def test_a_missing_hub_segmenter_stops(monkeypatch):
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError

    def missing(repo_id, filename, revision=None):
        response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/x"))
        raise RepositoryNotFoundError("no such repo", response=response)

    monkeypatch.setattr("bibr.utils.hf_cache.hf_download_or_cached", missing)
    ex = _extractor(monkeypatch, "org/no-such-seg")
    with pytest.raises(ModelOverrideError, match="REF_TEXT_SEG_MODEL_ID='org/no-such-seg'"):
        await ex._segment_references(_REF_TEXT, "crf")


def test_the_loaded_segmenter_is_recorded_by_checksum(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path / "b")
    ex = _extractor(monkeypatch, str(bundle))
    begin_run(ex._settings)
    ex._crf_segment_or_recover(_REF_TEXT)
    (record,) = loaded_model_overrides()
    assert record["setting"] == "REF_TEXT_SEG_MODEL_ID" and record["loaded"] is True
    assert record["value"] == str(bundle)
    by_name = {Path(f["path"]).name: f for f in record["files"]}
    assert set(by_name) == {"model.onnx", "segmenter.json", "tokenizer.json"}
    graph = bundle / "model.onnx"
    assert by_name["model.onnx"]["sha256"] == hashlib.sha256(graph.read_bytes()).hexdigest()
    assert by_name["model.onnx"]["bytes"] == graph.stat().st_size


async def test_a_missing_segmenter_stops_extract_all_metadata(tmp_path):
    import pandas as pd

    from bibr.extract.extractor import MetadataExtractor
    from bibr.paper import PaperMetadata

    settings = snapshot_settings()
    settings.ml.runtime = "onnx"
    settings.REF_TEXT_SEG_MODEL_ID = str(tmp_path / "nowhere")
    contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={})
    llm = MagicMock()
    llm.segment_references = AsyncMock(return_value=[])
    ex = MetadataExtractor(
        contents,
        file_hash="h",
        llm_client=llm,
        ref_seg_strategy="crf",
        ref_parse_strategy="ner",
        settings=settings,
    )
    ex.sentences_df = pd.DataFrame({"text": ["A sentence."]})

    async def core():
        ex.metadata = PaperMetadata(doi="", title="T", keywords=[], authors=[])

    ex.extract_core_metadata = core
    ex._collect_reference_rows = lambda: pd.DataFrame(
        {"text": [r[2:] for r in _REFS], "page_number": [5, 5, 5]}
    )
    with pytest.raises(ModelOverrideError, match="REF_TEXT_SEG_MODEL_ID"):
        await ex.extract_all_metadata()


# -- configuration ------------------------------------------------------------------------


def test_both_the_segmenter_and_an_overridden_crf_is_an_error(tmp_path, monkeypatch):
    settings = snapshot_settings()
    settings.REF_TEXT_SEG_MODEL_ID = str(_bundle(tmp_path / "b"))
    assert check_model_overrides(settings) == ([], [])
    settings.NER_SEG_CKPT = "/elsewhere/seg.pt"
    problems, _ = check_model_overrides(settings)
    assert len(problems) == 1 and "are both set" in problems[0]
    ex = _extractor(monkeypatch, settings.REF_TEXT_SEG_MODEL_ID)
    monkeypatch.setattr(ex._settings, "NER_SEG_CKPT", "/elsewhere/seg.pt")
    with pytest.raises(ModelOverrideError, match="are both set"):
        ex._crf_segment_or_recover(_REF_TEXT)


def test_the_preflight_checks_the_segmenter_files(tmp_path):
    settings = snapshot_settings()
    settings.REF_TEXT_SEG_MODEL_ID = str(tmp_path / "missing")
    (problem,), _ = check_model_overrides(settings)
    assert problem.startswith("REF_TEXT_SEG_MODEL_ID=") and "no such file" in problem
    bad = _bundle(tmp_path / "bad", manifest={"tags": ["O", "B"]})
    settings.REF_TEXT_SEG_MODEL_ID = str(bad)
    (problem,), _ = check_model_overrides(settings)
    assert "unusable text reference segmenter manifest" in problem


def test_the_segmenter_replaces_the_crf_in_the_models_a_run_loads():
    settings = snapshot_settings()
    assert "NER_SEG_CKPT" in models_in_use(settings)
    assert "REF_TEXT_SEG_MODEL_ID" not in models_in_use(settings)
    settings.REF_TEXT_SEG_MODEL_ID = "org/seg"
    used = models_in_use(settings)
    assert "REF_TEXT_SEG_MODEL_ID" in used and "NER_SEG_CKPT" not in used
    settings.REF_PARSE_STRATEGY = "off"
    assert "REF_TEXT_SEG_MODEL_ID" not in models_in_use(settings)


def test_the_setting_is_unset_by_default():
    settings = snapshot_settings()
    assert settings.REF_TEXT_SEG_MODEL_ID is None and settings.REF_TEXT_SEG_REVISION is None
