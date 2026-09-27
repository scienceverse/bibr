"""Front-role classifier: bundle contract, region keying, and soft loading."""

from __future__ import annotations

import joblib
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer

from bibr.extract import front_role
from bibr.extract.front_role import (
    FrontRoleClassifier,
    build_front_regions,
    load_front_role_classifier,
    reset_front_role_cache,
)
from bibr.extract.front_role_features import PROD_FEATURE_KEYS, FrontRegion, region_features
from bibr.ocr.types import OcrRegionResult

ROLES = (
    "title",
    "doi_line",
    "byline",
    "affiliation",
    "keywords",
    "abstract",
    "ref_header",
    "heading",
    "masthead",
    "body",
    "other",
)


def _synthetic_rows() -> tuple[list[dict], list[int]]:
    """A tiny separable dataset: big centred type = title, initials = byline."""
    rows: list[FrontRegion] = []
    labels: list[int] = []
    for i in range(30):
        rows.append(
            FrontRegion(
                page=0,
                index=3 * i,
                label="doc_title",
                text=f"A Study Of Things Number {i}",
                x0=0.2,
                y0=0.05,
                x1=0.8,
                y1=0.1,
                font_size=18.0,
                font_bold=True,
            )
        )
        labels.append(ROLES.index("title"))
        rows.append(
            FrontRegion(
                page=0,
                index=3 * i + 1,
                label="text",
                text=f"A. B. Smith{i},1 C. D. Jones,2 and E. F. Wu3*",
                x0=0.2,
                y0=0.12,
                x1=0.8,
                y1=0.14,
                font_size=9.0,
                font_bold=False,
            )
        )
        labels.append(ROLES.index("byline"))
        rows.append(
            FrontRegion(
                page=0,
                index=3 * i + 2,
                label="text",
                text="We present a load side management scheme for networks in detail here.",
                x0=0.1,
                y0=0.3,
                x1=0.9,
                y1=0.5,
                font_size=9.0,
                font_bold=False,
            )
        )
        labels.append(ROLES.index("body"))
    return region_features(rows), labels


@pytest.fixture
def bundle_path(tmp_path):
    feats, labels = _synthetic_rows()
    keys = sorted(PROD_FEATURE_KEYS)
    dv = DictVectorizer(sparse=False)
    x = dv.fit_transform([{k: f[k] for k in keys} for f in feats])
    model = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(x, labels)
    bundle = {
        "model": model,
        "vectorizer": dv,
        "feature_keys": keys,
        "roles": list(ROLES),
        "model_kind": "front_role_v1",
        "version": "test",
    }
    path = tmp_path / "front_role.joblib"
    joblib.dump(bundle, path)
    return path


def _typed_region(index: int, label: str, content: str, bbox, *, font_size=9.0, bold=False):
    return OcrRegionResult(
        index=index,
        native_label=label,
        label=label,
        content=content,
        bbox_2d=bbox,
        font_size=font_size,
        font_bold=bold,
    )


def test_bundle_loads_and_scores_typed_regions(bundle_path):
    clf = FrontRoleClassifier(str(bundle_path))
    pages = [
        [
            _typed_region(
                0, "doc_title", "A Study Of Things", [200, 50, 800, 100], font_size=18.0, bold=True
            ),
            _typed_region(
                1, "text", "A. B. Smith,1 C. D. Jones,2 and E. F. Wu3*", [200, 120, 800, 140]
            ),
            _typed_region(2, "image", "", [0, 0, 10, 10]),
            _typed_region(
                3,
                "text",
                "We present a load side management scheme for networks in detail here.",
                [100, 300, 900, 500],
            ),
        ]
    ]
    predictions = clf.predict_pages(pages)
    # Non-text regions are not scored; keys are (1-based page, region index).
    assert len(predictions) == 3
    assert predictions.get(1, 2) is None
    assert predictions.get(1, 0).top == "title"
    assert predictions.get(1, 1).top == "byline"
    assert predictions.get(1, 3).top == "body"
    scores = predictions.get(1, 0)
    assert abs(sum(scores.probs.values()) - 1.0) < 1e-6
    assert scores.confidence == scores.get("title")


def test_dict_regions_use_underscored_font_keys_like_the_ocr_cache():
    pages = [
        [
            {
                "index": 4,
                "label": "text",
                "content": "x",
                "bbox_2d": [0, 0, 500, 1000],
                "_font_size": 7.5,
                "_font_bold": True,
            }
        ]
    ]
    (region,) = build_front_regions(pages, first_page_index=2)
    assert region.page == 2 and region.index == 4
    assert region.font_size == 7.5 and region.font_bold is True
    assert region.x1 == 0.5 and region.y1 == 1.0


def test_start_page_offset_keeps_page_numbers_relative_to_processed_pages(bundle_path):
    clf = FrontRoleClassifier(str(bundle_path))
    pages = [
        [_typed_region(0, "doc_title", "A Study Of Things", [200, 50, 800, 100], font_size=18.0)]
    ]
    predictions = clf.predict_pages(pages, first_page_index=3)
    assert predictions.get(1, 0) is not None


def test_bundle_with_foreign_feature_keys_is_rejected(tmp_path):
    joblib.dump(
        {"model": None, "vectorizer": None, "feature_keys": ["nope"], "roles": list(ROLES)},
        tmp_path / "front_role.joblib",
    )
    with pytest.raises(ValueError, match="feature_keys"):
        FrontRoleClassifier(str(tmp_path / "front_role.joblib"))


def test_loader_is_soft_and_caches_failures(monkeypatch, caplog):
    from bibr.config import snapshot_settings

    reset_front_role_cache()
    settings = snapshot_settings()
    settings.ml.front_role_model_id = "/nonexistent/front_role.joblib"
    calls = []

    def _boom(*a, **k):
        calls.append(1)
        raise OSError("no such bundle")

    monkeypatch.setattr(front_role, "FrontRoleClassifier", _boom)
    with caplog.at_level("WARNING"):
        assert load_front_role_classifier(settings) is None
        assert load_front_role_classifier(settings) is None
    assert len(calls) == 1
    assert "falling back to heuristics" in caplog.text
    reset_front_role_cache()


def test_loader_retries_after_a_network_failure(monkeypatch):
    """One Hub blip must not disable the default prior for the process."""
    import httpx

    from bibr.config import snapshot_settings

    reset_front_role_cache()
    settings = snapshot_settings()
    settings.ml.front_role_model_id = "org/front-role"
    loaded = object()
    attempts = []

    def _flaky(*a, **k):
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectTimeout("hub unreachable")
        return loaded

    monkeypatch.setattr(front_role, "FrontRoleClassifier", _flaky)
    assert load_front_role_classifier(settings) is None
    assert load_front_role_classifier(settings) is loaded
    assert load_front_role_classifier(settings) is loaded
    assert len(attempts) == 2
    reset_front_role_cache()


def test_resources_pin_only_a_loaded_front_role_model(monkeypatch):
    from bibr.config import snapshot_settings
    from bibr.pipeline.resources import ResourceManager

    results = [None, "model"]
    monkeypatch.setattr(
        "bibr.extract.front_role.load_front_role_classifier", lambda settings: results.pop(0)
    )
    rm = ResourceManager(settings=snapshot_settings())

    assert rm.ensure_front_role() is None
    assert rm.ensure_front_role() == "model"
    assert rm.ensure_front_role() == "model"  # pinned: no third load
    assert results == []


def test_loader_respects_disabled_and_unset(monkeypatch):
    from bibr.config import snapshot_settings

    reset_front_role_cache()
    settings = snapshot_settings()
    settings.ml.front_role_model_id = None
    assert load_front_role_classifier(settings) is None
    settings.ml.front_role_model_id = "org/repo"
    settings.ml.front_role_enabled = False
    assert load_front_role_classifier(settings) is None
