"""First-page region-role classifier (``ML_FRONT_ROLE_MODEL_ID``).

A small gradient-boosted model over the production feature contract in
:mod:`bibr.extract.front_role_features` scores every text region of a paper
with a distribution over roles (title, byline, affiliation, abstract,
keywords, doi_line, masthead, heading, ref_header, body, other). Trained in
bibr-training from publisher JATS projected onto cached OCR regions, so the
labels are verbatim ground truth rather than an LLM's opinion.

The scores are *evidence*, consumed by :mod:`bibr.extract.front_matter` next
to its lexical heuristics: a region the model calls a byline is admitted as
one even when the English byline shape or the 45-word cap would have rejected
it, a model title is a title seed for scripts the uppercase test cannot read,
and a confident masthead cannot root a record. ``RefLocator`` uses the
``ref_header`` role for non-English reference headings.

Loading is soft: a missing/unloadable bundle logs once and disables the model
(``load_front_role_classifier`` returns ``None``), so extraction never fails
because an optional prior is absent.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bibr.extract.front_role_features import (
    NON_TEXT_LABELS,
    PROD_FEATURE_KEYS,
    FrontRegion,
    region_features,
)

if TYPE_CHECKING:
    from bibr.config import GlobalSettings

logger = logging.getLogger(__name__)

_BUNDLE_FILENAME = "front_role.joblib"
MODEL_KIND_FRONT_ROLE_V1 = "front_role_v1"
ROLE_TITLE = "title"
ROLE_BYLINE = "byline"
ROLE_AFFILIATION = "affiliation"
ROLE_ABSTRACT = "abstract"
ROLE_KEYWORDS = "keywords"
ROLE_MASTHEAD = "masthead"
ROLE_REF_HEADER = "ref_header"


@dataclass(frozen=True)
class RoleScores:
    """Role distribution for one region."""

    probs: Mapping[str, float]
    top: str
    confidence: float

    def get(self, role: str) -> float:
        return float(self.probs.get(role, 0.0))


class FrontRolePredictions:
    """``(page_number, region_index) -> RoleScores`` for one paper.

    ``page_number`` follows ``RegionSummary.page`` (1-based within the
    processed pages) so front-matter candidates can look themselves up by
    their ``region_order``.
    """

    def __init__(self, scores: Mapping[tuple[int, int], RoleScores], *, model_version: str):
        self._scores = dict(scores)
        self.model_version = model_version

    def get(self, page_number: int | None, index: int | None) -> RoleScores | None:
        if page_number is None or index is None:
            return None
        return self._scores.get((int(page_number), int(index)))

    def __len__(self) -> int:
        return len(self._scores)

    def __iter__(self):
        return iter(self._scores.items())


def _resolve_spec(model_spec: str) -> str:
    if Path(model_spec).exists() or ":" in model_spec:
        return model_spec
    return f"{model_spec}:{_BUNDLE_FILENAME}"


def _region_attr(region: Any, name: str, default=None):
    if isinstance(region, Mapping):
        return region.get(name, region.get(f"_{name}", default))
    return getattr(region, name, default)


def build_front_regions(
    pages: Sequence[Sequence[Any]], *, first_page_index: int = 0
) -> list[FrontRegion]:
    """Reading-order ``FrontRegion`` rows from typed or dict OCR regions.

    Every region participates (non-text ones included) so neighbour features
    match training, where the labeler saw the full page too.
    """
    rows: list[FrontRegion] = []
    for page_idx, regions in enumerate(pages):
        for pos, region in enumerate(regions):
            bbox = _region_attr(region, "bbox_2d")
            if not bbox or len(bbox) != 4:
                bbox = [0, 0, 0, 0]
            index = _region_attr(region, "index", pos)
            font_size = _region_attr(region, "font_size")
            font_bold = _region_attr(region, "font_bold")
            label = _region_attr(region, "label") or _region_attr(region, "native_label")
            rows.append(
                FrontRegion.from_image_bbox(
                    list(bbox),
                    page=first_page_index + page_idx,
                    index=int(index if index is not None else pos),
                    label=str(label) if label else None,
                    text=str(_region_attr(region, "content") or ""),
                    font_size=float(font_size) if isinstance(font_size, (int, float)) else None,
                    font_bold=font_bold if isinstance(font_bold, bool) else None,
                )
            )
    return rows


class FrontRoleClassifier:
    """Loads a bundle and scores regions; safe to share across threads."""

    def __init__(self, model_spec: str, revision: str | None = None) -> None:
        from bibr.ner.checkpoint import resolve_checkpoint
        from bibr.utils.safe_pickle import safe_joblib_load

        path = resolve_checkpoint(_resolve_spec(model_spec), revision=revision)
        # Same trust boundary as the geom segmenter: joblib executes code on
        # load, so only the gadget-restricted loader touches the file.
        bundle = safe_joblib_load(path)
        self.model_kind = str(bundle.get("model_kind", MODEL_KIND_FRONT_ROLE_V1))
        if self.model_kind != MODEL_KIND_FRONT_ROLE_V1:
            raise ValueError(f"unknown front-role model_kind: {self.model_kind!r}")
        feature_keys = list(bundle["feature_keys"])
        if set(feature_keys) != set(PROD_FEATURE_KEYS):
            raise ValueError(
                "front-role bundle feature_keys do not match production feature keys "
                f"(bundle={sorted(feature_keys)}, prod={sorted(PROD_FEATURE_KEYS)})"
            )
        self.model = bundle["model"]
        self.vectorizer = bundle["vectorizer"]
        self.feature_keys = feature_keys
        self.roles: list[str] = [str(r) for r in bundle["roles"]]
        self.version = str(bundle.get("version", MODEL_KIND_FRONT_ROLE_V1))
        self._lock = threading.Lock()

    def _score_rows(self, rows: list[dict]) -> list[RoleScores]:
        if not rows:
            return []
        vectors = self.vectorizer.transform([{k: r[k] for k in self.feature_keys} for r in rows])
        with self._lock:
            proba = self.model.predict_proba(vectors)
        classes = [int(c) for c in getattr(self.model, "classes_", range(len(self.roles)))]
        out: list[RoleScores] = []
        for row in proba:
            probs = dict.fromkeys(self.roles, 0.0)
            for col, cls in enumerate(classes):
                probs[self.roles[cls]] = float(row[col])
            top = max(probs, key=probs.get)
            out.append(RoleScores(probs=probs, top=top, confidence=probs[top]))
        return out

    def predict_pages(
        self, pages: Sequence[Sequence[Any]], *, first_page_index: int = 0
    ) -> FrontRolePredictions:
        """Score every text region; keys are ``(page_number, index)``."""
        regions = build_front_regions(pages, first_page_index=first_page_index)
        feats = region_features(regions)
        keep = [
            (region, feat)
            for region, feat in zip(regions, feats, strict=True)
            if (region.label or "") not in NON_TEXT_LABELS
        ]
        scores = self._score_rows([feat for _, feat in keep])
        keyed = {
            (region.page - first_page_index + 1, region.index): score
            for (region, _), score in zip(keep, scores, strict=True)
        }
        return FrontRolePredictions(keyed, model_version=self.version)


_CACHE: dict[tuple[str, str | None], FrontRoleClassifier | None] = {}
_CACHE_LOCK = threading.Lock()


def load_front_role_classifier(settings: GlobalSettings) -> FrontRoleClassifier | None:
    """Return the configured classifier, or ``None`` when disabled/unavailable.

    Failures are logged once per (model, revision) and cached as ``None`` so a
    missing optional bundle never costs more than one attempt per process.
    """
    ml = settings.ml
    model_id = getattr(ml, "front_role_model_id", None)
    if not getattr(ml, "front_role_enabled", True) or not model_id:
        return None
    revision = getattr(ml, "front_role_revision", None) or None
    key = (str(model_id), revision)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
    try:
        clf: FrontRoleClassifier | None = FrontRoleClassifier(str(model_id), revision=revision)
    except Exception as exc:  # noqa: BLE001 - optional prior, never fatal
        logger.warning(
            "Front-role classifier %s unavailable (%s); falling back to heuristics",
            model_id,
            exc,
        )
        clf = None
    with _CACHE_LOCK:
        _CACHE[key] = clf
    return clf


def release_front_role_classifier(classifier: FrontRoleClassifier) -> None:
    """Drop the cache's ownership without unloading another pipeline's model."""
    with _CACHE_LOCK:
        for key in [key for key, value in _CACHE.items() if value is classifier]:
            del _CACHE[key]


def reset_front_role_cache() -> None:
    """Test hook: forget cached classifiers (and cached failures)."""
    with _CACHE_LOCK:
        _CACHE.clear()


__all__ = [
    "FrontRoleClassifier",
    "FrontRolePredictions",
    "RoleScores",
    "build_front_regions",
    "load_front_role_classifier",
    "reset_front_role_cache",
]
