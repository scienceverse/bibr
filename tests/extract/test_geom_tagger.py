"""The opt-in geometry line tagger (``REF_GEOM_TAGGER_MODEL_ID``) and its place in the geom tier.

The bundles here are built in the test: a one-layer ONNX graph over the 29 line
features, with a constant or feature-weighted logit, and a gate.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from bibr.exceptions import ConfigurationError
from bibr.extract import geom_tagger
from bibr.extract.geom_segmenter import MODEL_KIND_ADJACENT_BOUNDARY_V2, _feature_rows_for_kind
from bibr.extract.geom_tagger import (
    FEATURE_KEYS,
    GeomLineTagger,
    expected_entry_precision,
    feature_matrix,
)
from bibr.extract.ref_extractor import ReferenceExtractor
from bibr.ocr.ref_geometry import records_from_dicts
from bibr.paper_contents import PaperContents
from bibr.utils.model_overrides import loaded_model_overrides, reset_loaded_model_overrides

onnx = pytest.importorskip("onnx")

_REFS = (
    "Aknin, L. B. (2013). Prosocial spending and well-being. Journal of X, 1, 2-3.",
    "Carstensen, L. L. (2009). Another title here. Journal of Y, 4, 5-6.",
    "Smith, J. (2010). A third title. Journal of Z, 7, 8-9.",
)
_REF_TEXT = "\n".join(_REFS)


def _lines(texts=_REFS):
    return [
        {
            "text": text,
            "page": 5,
            "x0": 72.0,
            "y_top": 700.0 - 14.0 * i,
            "x1": 520.0,
            "y_bottom": 690.0 - 14.0 * i,
            "font_size": 10.0,
        }
        for i, text in enumerate(texts)
    ]


def _write_graph(path: Path, *, bias, weights=None) -> None:
    from onnx import TensorProto, helper, numpy_helper

    w = np.zeros((len(FEATURE_KEYS), 2), np.float32) if weights is None else weights
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["features", "W"], ["mm"]),
            helper.make_node("Add", ["mm", "B"], ["logits"]),
        ],
        "tiny_line_tagger",
        [
            helper.make_tensor_value_info(
                "features", TensorProto.FLOAT, ["lines", len(FEATURE_KEYS)]
            )
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["lines", 2])],
        initializer=[
            numpy_helper.from_array(np.asarray(w, np.float32), "W"),
            numpy_helper.from_array(np.asarray(bias, np.float32), "B"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def _bundle(
    root: Path,
    *,
    bias=(0.0, 8.0),
    weights=None,
    threshold=0.9,
    temperature=1.0,
    in_onnx_dir=True,
    feature_keys=None,
) -> Path:
    graph_dir = root / "onnx" if in_onnx_dir else root
    graph_dir.mkdir(parents=True, exist_ok=True)
    _write_graph(graph_dir / "tagger.onnx", bias=bias, weights=weights)
    (graph_dir / "tagger.json").write_text(
        json.dumps(
            {
                "input": "features",
                "feature_keys": list(feature_keys or FEATURE_KEYS),
                "outputs": {"logits": ["lines", 2]},
            }
        )
    )
    (root / "gate.json").write_text(
        json.dumps(
            {
                "temperature": temperature,
                "threshold": threshold,
                "score": "expected_entry_precision",
                "isotonic_x": [0.0, 1.0],
                "isotonic_y": [0.0, 1.0],
            }
        )
    )
    return root


def _extractor(lines, tagger_path, monkeypatch):
    contents = PaperContents(
        sentences=[],
        sections=[],
        tables=[],
        links=[],
        sections_text={},
        ref_line_geometry=lines,
    )
    ex = ReferenceExtractor(contents, file_hash="h", llm_client=MagicMock())
    ex.llm_client.segment_references = AsyncMock(return_value=[])
    monkeypatch.setattr(ex._settings, "REF_GEOM_TAGGER_MODEL_ID", str(tagger_path))
    return ex


@pytest.fixture(autouse=True)
def _fresh_records():
    reset_loaded_model_overrides()
    yield
    reset_loaded_model_overrides()


# -- features ------------------------------------------------------------------


def test_feature_keys_are_the_gbm_adjacent_contract_in_sorted_order():
    assert len(FEATURE_KEYS) == 29
    assert list(FEATURE_KEYS) == sorted(FEATURE_KEYS)


def test_tagger_features_equal_the_gbm_features_on_the_same_lines():
    """Both paths read the same lines: the tagger's matrix is the GBM's rows, column for column."""
    lines = records_from_dicts(
        _lines(
            (
                "References",
                *_REFS,
                "  continued line of the third reference, 10.1000/xyz",
                "[4] Numbered, A. 2011. Title. Venue.",
            )
        )
    )
    rows = _feature_rows_for_kind(lines, MODEL_KIND_ADJACENT_BOUNDARY_V2)
    expected = np.asarray([[float(r[k]) for k in FEATURE_KEYS] for r in rows], dtype=np.float32)
    got = feature_matrix(lines)
    assert got.shape == (len(lines), 29)
    assert np.array_equal(got, expected)


# -- gate ------------------------------------------------------------------------


def test_expected_entry_precision_counts_the_forced_first_line_as_certain():
    # The first line is an opening whatever its probability; the rest add their doubt.
    assert expected_entry_precision(np.array([0.1, 0.9, 0.2])) == pytest.approx(
        1 - 2 * (0.1 + 0.2) / 2
    )
    assert expected_entry_precision(np.array([])) == 0.0
    # Four openings at p = 0.5, three of them in doubt by 0.5 each.
    assert expected_entry_precision(np.array([0.5] * 4)) == pytest.approx(1 - 2 * 1.5 / 4)


def test_tagger_reads_temperature_and_isotonic_gate(tmp_path):
    tagger = GeomLineTagger(_bundle(tmp_path, bias=(0.0, 2.0), temperature=2.0))
    probs = tagger.line_probabilities(records_from_dicts(_lines()))
    assert probs == pytest.approx([1 / (1 + np.exp(-1.0))] * 3, abs=1e-6)
    assert tagger.confidence(probs) == pytest.approx(expected_entry_precision(probs))


@pytest.mark.parametrize("in_onnx_dir", [True, False])
def test_bundle_layouts(tmp_path, in_onnx_dir):
    tagger = GeomLineTagger(_bundle(tmp_path, in_onnx_dir=in_onnx_dir))
    assert [p.name for p in tagger.files] == ["tagger.onnx", "tagger.json", "gate.json"]


def test_bundle_problems_are_named(tmp_path):
    with pytest.raises(ConfigurationError, match="not a directory"):
        GeomLineTagger(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigurationError, match="gate.json, tagger.onnx"):
        GeomLineTagger(tmp_path / "empty")
    other = _bundle(tmp_path / "other", feature_keys=sorted(FEATURE_KEYS)[:-1] + ["new_feature"])
    with pytest.raises(ConfigurationError, match="features differ"):
        GeomLineTagger(other)
    gate = _bundle(tmp_path / "gate")
    (gate / "gate.json").write_text(json.dumps({"temperature": 1.0, "threshold": 0.9}))
    with pytest.raises(ConfigurationError, match="isotonic_x"):
        GeomLineTagger(gate)


# -- the geom tier -----------------------------------------------------------------


def _attempts(ex):
    return [(a.strategy, a.selected, a.reason_flags) for a in ex._segmentation_attempts]


def test_a_confident_tagger_answers_and_the_gbm_is_not_asked(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path / "tagger")
    ex = _extractor(_lines(), bundle, monkeypatch)
    gbm = MagicMock()
    with patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=gbm) as get_gbm:
        out = ex._segment_geom(_REF_TEXT)
    get_gbm.assert_not_called()
    assert [s.strip() for s in out] == list(_REFS)
    assert _attempts(ex) == [("geom_tagger", True, ())]
    # No cascade warning: the tier answered.
    assert not ex.contents.processing_warnings


def test_a_declining_gate_leaves_the_gbm_tier_unchanged(tmp_path, monkeypatch):
    # p = sigmoid(0.5) on every line: a paper score far under the threshold.
    bundle = _bundle(tmp_path / "tagger", bias=(0.0, 0.5))
    ex = _extractor(_lines(), bundle, monkeypatch)
    gbm = MagicMock()
    spans = [(0, len(_REFS[0])), (len(_REFS[0]) + 1, len(_REF_TEXT))]
    gbm.segment_spans.return_value = (spans, 0.99, 2, 2)
    with patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=gbm):
        out = ex._segment_geom(_REF_TEXT)
    assert out == [_REF_TEXT[s:e] for s, e in spans]
    assert _attempts(ex) == [
        ("geom_tagger", False, ("low_confidence_or_empty",)),
        ("geom", True, ()),
    ]
    gbm.segment_spans.assert_called_once_with(_REF_TEXT, _lines())


def test_a_tagger_under_the_region_floor_declines(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path / "tagger")
    ex = _extractor(_lines(), bundle, monkeypatch)
    monkeypatch.setattr(ex, "_geom_region_floor", lambda ref_text: (10, 10, 0.9))
    gbm = MagicMock()
    gbm.segment_spans.return_value = ([], 0.0, 0, 0)
    with patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=gbm):
        assert ex._segment_geom(_REF_TEXT) is None
    assert _attempts(ex)[0] == ("geom_tagger", False, ("low_segment_count",))
    assert _attempts(ex)[1][0] == "geom"


def test_without_the_setting_the_tagger_is_never_built(monkeypatch):
    contents = PaperContents(
        sentences=[], sections=[], tables=[], links=[], sections_text={}, ref_line_geometry=_lines()
    )
    ex = ReferenceExtractor(contents, file_hash="h", llm_client=MagicMock())
    assert ex._settings.REF_GEOM_TAGGER_MODEL_ID is None
    gbm = MagicMock()
    gbm.segment_spans.return_value = ([(0, len(_REF_TEXT))], 0.99, 1, 1)
    with (
        patch.object(geom_tagger, "GeomLineTagger", side_effect=AssertionError("built")),
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=gbm),
    ):
        ex._segment_geom(_REF_TEXT)
    assert [a.strategy for a in ex._segmentation_attempts] == ["geom"]
    assert loaded_model_overrides() == []


def test_a_missing_tagger_stops_instead_of_falling_back(tmp_path, monkeypatch):
    ex = _extractor(_lines(), tmp_path / "nowhere", monkeypatch)
    gbm = MagicMock()
    with (
        patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=gbm),
        pytest.raises(ConfigurationError, match="never falls back"),
    ):
        ex._segment_geom(_REF_TEXT)
    gbm.segment_spans.assert_not_called()


def test_the_loaded_tagger_is_recorded_by_checksum(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path / "tagger")
    ex = _extractor(_lines(), bundle, monkeypatch)
    with patch("bibr.extract.ref_extractor._get_geom_segmenter", return_value=MagicMock()):
        ex._segment_geom(_REF_TEXT)
    (record,) = loaded_model_overrides()
    assert record["setting"] == "REF_GEOM_TAGGER_MODEL_ID"
    assert record["value"] == str(bundle)
    by_name = {Path(f["path"]).name: f for f in record["files"]}
    onnx_file = bundle / "onnx" / "tagger.onnx"
    assert by_name["tagger.onnx"]["sha256"] == hashlib.sha256(onnx_file.read_bytes()).hexdigest()
    assert by_name["tagger.onnx"]["bytes"] == onnx_file.stat().st_size
    assert set(by_name) == {"tagger.onnx", "tagger.json", "gate.json"}
