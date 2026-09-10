import joblib
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer

from bibr.extract.geom_features import PROD_FEATURE_KEYS, line_features
from bibr.extract.geom_segmenter import (
    ADJACENT_FEATURE_KEYS,
    MODEL_KIND_ADJACENT_BOUNDARY_V2,
    GeomSegmenter,
)
from bibr.ocr.ref_geometry import LineRecord, record_to_dict


def _ln(text, x0, page=1, y=700.0):
    return LineRecord(text, page, x0, y, x0 + 400.0, y - 10.0, 10.0)


def _with_adjacent_rows(rows, lines):
    return [
        {
            **row,
            **{key: float(i + 1) for key in ADJACENT_FEATURE_KEYS},
        }
        for i, row in enumerate(rows)
    ]


def _train_tiny_bundle(path, *, feature_keys=None, model_kind=None):
    """Train a tiny GBM that learns 'indented line == not a boundary'."""
    lines = [
        _ln("Aknin, L. B. (2013). A title.", 72.0, y=700),
        _ln("Journal of X, 1, 2-3.", 108.0, y=688),
        _ln("Carstensen, L. (2009). Another.", 72.0, y=676),
        _ln("Journal of Y, 2, 4-5.", 108.0, y=664),
    ]
    rows = line_features(lines)
    if model_kind == MODEL_KIND_ADJACENT_BOUNDARY_V2:
        rows = _with_adjacent_rows(rows, lines)
    keys = sorted(feature_keys if feature_keys is not None else PROD_FEATURE_KEYS)
    ys = [1, 0, 1, 0]
    dv = DictVectorizer(sparse=False)
    x = dv.fit_transform([{k: r[k] for k in keys} for r in rows])
    # min_samples_leaf=1: sklearn >= 1.9.0 defaults to 20, which prevents any
    # split on this 4-sample fixture (all predict_proba collapse to 0.5).
    model = HistGradientBoostingClassifier(max_iter=50, random_state=0, min_samples_leaf=1).fit(
        x, ys
    )
    joblib.dump(
        {
            "model": model,
            "vectorizer": dv,
            "feature_keys": keys,
            "cascade_threshold": 0.0,
            "boundary_f1_val": 0.0,
            "version": "test",
            **({"model_kind": model_kind} if model_kind else {}),
        },
        path,
    )


def test_segment_round_trips_to_ref_strings(tmp_path):
    bundle = tmp_path / "geom_segmenter.joblib"
    _train_tiny_bundle(bundle)
    seg = GeomSegmenter(str(bundle))
    lines = [
        _ln("Aknin, L. B. (2013). A title.", 72.0, y=700),
        _ln("Journal of X, 1, 2-3.", 108.0, y=688),
        _ln("Carstensen, L. (2009). Another.", 72.0, y=676),
    ]
    ref_text = "Aknin, L. B. (2013). A title. Journal of X, 1, 2-3. Carstensen, L. (2009). Another."
    refs, conf = seg.segment(ref_text, [record_to_dict(x) for x in lines])
    assert len(refs) == 2  # two boundaries (indented middle line is not one)
    assert 0.0 <= conf <= 1.0


def test_empty_lines_returns_empty_and_zero_conf(tmp_path):
    bundle = tmp_path / "geom_segmenter.joblib"
    _train_tiny_bundle(bundle)
    seg = GeomSegmenter(str(bundle))
    assert seg.segment("whatever", []) == ([], 0.0)


def test_feature_key_mismatch_raises_at_load(tmp_path):
    bundle = tmp_path / "geom_segmenter.joblib"
    _train_tiny_bundle(bundle, feature_keys=["dx_prev", "x0_rel_page"])  # wrong subset
    with pytest.raises(ValueError, match="feature_keys"):
        GeomSegmenter(str(bundle))


def test_adjacent_boundary_bundle_loads_and_predicts_with_augmented_features(tmp_path):
    bundle = tmp_path / "geom_segmenter.joblib"
    keys = sorted({*PROD_FEATURE_KEYS, *ADJACENT_FEATURE_KEYS})
    _train_tiny_bundle(bundle, feature_keys=keys, model_kind=MODEL_KIND_ADJACENT_BOUNDARY_V2)
    seg = GeomSegmenter(str(bundle))

    labels, probs = seg._predict(_LINES)

    assert seg.model_kind == MODEL_KIND_ADJACENT_BOUNDARY_V2
    assert labels[0] == "B-REF"
    assert len(probs) == len(_LINES)


class _FakeModel:
    """Stub classifier returning fixed B-REF probabilities (column 1)."""

    def __init__(self, b_ref_probs):
        self._p = b_ref_probs

    def predict_proba(self, x):
        import numpy as np

        return np.array([[1.0 - p, p] for p in self._p])


_LINES = [
    _ln("Aknin, L. B. (2013). A title.", 72.0, y=700),
    _ln("Journal of X, 1, 2-3.", 108.0, y=688),
    _ln("Carstensen, L. (2009). Another.", 72.0, y=676),
]


def test_predict_forces_first_line_b_ref_even_with_later_b_ref(tmp_path):
    """D2: the first region line always opens the first reference.

    The model can mislabel a dense / firstname-first / numbered lead line as
    I-REF while confidently labeling a *later* line B-REF. Anchor snapping would
    then drop the genuine first reference. labels[0] must be forced B-REF
    unconditionally — not only when no B-REF exists anywhere.
    """
    bundle = tmp_path / "geom_segmenter.joblib"
    _train_tiny_bundle(bundle)
    seg = GeomSegmenter(str(bundle))
    seg.model = _FakeModel([0.2, 0.1, 0.9])  # line 0 I-REF, line 2 B-REF

    labels, probs = seg._predict(_LINES)

    assert labels[0] == "B-REF"
    assert probs == [0.2, 0.1, 0.9]  # forcing the label must not alter confidence


def test_predict_forces_first_line_b_ref_when_none_present(tmp_path):
    bundle = tmp_path / "geom_segmenter.joblib"
    _train_tiny_bundle(bundle)
    seg = GeomSegmenter(str(bundle))
    seg.model = _FakeModel([0.1, 0.2, 0.3])  # all I-REF

    labels, _ = seg._predict(_LINES)

    assert labels[0] == "B-REF"


# --- Span contract (roadmap: span-based segmenter output) ---

_REF_TEXT = "Aknin, L. B. (2013). A title. Journal of X, 1, 2-3. Carstensen, L. (2009). Another."


def test_segment_spans_returns_char_spans_and_confidence(tmp_path):
    bundle = tmp_path / "geom_segmenter.joblib"
    _train_tiny_bundle(bundle)
    seg = GeomSegmenter(str(bundle))
    lines = [record_to_dict(x) for x in _LINES]

    spans, conf, labeled, aligned = seg.segment_spans(_REF_TEXT, lines)

    assert spans == [
        (0, _REF_TEXT.index(" Carstensen")),
        (_REF_TEXT.index("Carstensen"), len(_REF_TEXT)),
    ]
    assert 0.0 <= conf <= 1.0
    assert labeled == 2  # two B-REF lines (indices 0 and 2)
    assert aligned == 2  # both boundary lines locate cleanly in this fixture


def test_segment_strings_are_span_slices(tmp_path):
    bundle = tmp_path / "geom_segmenter.joblib"
    _train_tiny_bundle(bundle)
    seg = GeomSegmenter(str(bundle))
    lines = [record_to_dict(x) for x in _LINES]

    spans, _, _, _ = seg.segment_spans(_REF_TEXT, lines)
    refs, _ = seg.segment(_REF_TEXT, lines)

    assert refs == [_REF_TEXT[s:e] for s, e in spans]


def test_segment_spans_empty_lines(tmp_path):
    bundle = tmp_path / "geom_segmenter.joblib"
    _train_tiny_bundle(bundle)
    seg = GeomSegmenter(str(bundle))
    assert seg.segment_spans("whatever", []) == ([], 0.0, 0, 0)


def test_segment_spans_labeled_count_survives_collapsed_alignment(tmp_path):
    """``labeled`` must reflect the GBM's full count even when ``aligned`` (and
    therefore ``spans``) collapses -- this is the signal the yield gate in
    ref_extractor consumes (see REF_GEOM_MIN_ALIGN_YIELD in config.py for the
    OOD rationale)."""
    bundle = tmp_path / "geom_segmenter.joblib"
    _train_tiny_bundle(bundle)
    seg = GeomSegmenter(str(bundle))
    lines = [record_to_dict(x) for x in _LINES]
    # ref_text only contains the first line's text -- the other two boundary
    # lines' text is entirely absent, simulating dropped OCR markers.
    starved_ref_text = "Aknin, L. B. (2013). A title."

    spans, _conf, labeled, aligned = seg.segment_spans(starved_ref_text, lines)

    assert labeled == 2  # GBM still labels both boundary lines B-REF
    assert aligned == 1  # only the first boundary line locates in the text
    assert len(spans) == 1
