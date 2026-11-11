"""Model overrides: a model setting pointed at something other than its shipped default.

Each model bibr loads is named by one setting (``NER_PARSER_CKPT``,
``ML_FRONT_ROLE_MODEL_ID``, ...; :data:`MODEL_SETTINGS`), which takes a Hub id
or a local path. Set to anything but its default (or with a revision other than
its pinned one), that setting is an *override*. Overrides exist to run a
candidate model in place of the shipped one, so three rules hold for them that
the defaults do not need:

1. **Checked before the run.** ``bibr chew`` and ``bibr batch`` check every
   override that names a local path before the first paper
   (:func:`check_model_overrides`): the path must exist and hold the files that
   model's loader reads, in a format this bibr can run. A problem stops the run.
   A Hub id is checked when it loads.
2. **No fallback.** Where the shipped model's load failure lets the pipeline go
   on without it (the front-role prior, the geometry segmenter, the classifiers,
   the front-matter tagger, the CRF segmenter), an override's failure raises
   :class:`~bibr.exceptions.ConfigurationError`. A comparison run must never
   measure the shipped model under a candidate's name.
3. **Identity recorded.** The loader records the sha256 of every file it read
   (:func:`record_model_load`), logs it, and ``bibr batch`` writes the records to
   ``run_info.json`` under ``model_overrides``.

With every model setting at its default nothing here runs: no check, no record,
no change to the exports.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bibr.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSetting:
    """One model setting: its env name, where it sits in the settings, and its format."""

    env: str
    label: str
    attr: tuple[str, ...]
    revision_attr: tuple[str, ...] | None
    # "onnx" (an onnx/ bundle; ``onnx_model`` is the manifest's ``model`` field, or None for a
    # bundle checked otherwise), "joblib" (one bundle file), "geom_tagger", "checkpoint" (a file).
    kind: str
    onnx_model: str | None = None


MODEL_SETTINGS: tuple[ModelSetting, ...] = (
    ModelSetting(
        "NER_PARSER_CKPT",
        "reference parser",
        ("NER_PARSER_CKPT",),
        ("NER_PARSER_REVISION",),
        "onnx",
        "ner_parser",
    ),
    ModelSetting(
        "REF_GEOM_SEG_MODEL_ID",
        "geometry reference segmenter",
        ("REF_GEOM_SEG_MODEL_ID",),
        ("REF_GEOM_SEG_REVISION",),
        "joblib",
    ),
    ModelSetting(
        "REF_GEOM_TAGGER_MODEL_ID",
        "geometry line tagger",
        ("REF_GEOM_TAGGER_MODEL_ID",),
        None,
        "geom_tagger",
    ),
    ModelSetting(
        "NER_SEG_CKPT",
        "CRF reference segmenter",
        ("NER_SEG_CKPT",),
        ("NER_SEG_REVISION",),
        "checkpoint",
    ),
    ModelSetting(
        "ML_FRONT_ROLE_MODEL_ID",
        "front-role classifier",
        ("ml", "front_role_model_id"),
        ("ml", "front_role_revision"),
        "joblib",
    ),
    ModelSetting(
        "ML_FRONT_TAGGER_MODEL_ID",
        "front-matter tagger",
        ("ml", "front_tagger_model_id"),
        ("ml", "front_tagger_revision"),
        "onnx",
    ),
    ModelSetting(
        "ML_SECTION_CLASSIFIER_MODEL_ID",
        "section classifier",
        ("ml", "section_classifier_model_id"),
        ("ml", "section_classifier_revision"),
        "onnx",
        "section_classifier",
    ),
    ModelSetting(
        "ML_PAPER_CLASSIFIER_MODEL_ID",
        "paper classifier",
        ("ml", "paper_classifier_model_id"),
        ("ml", "paper_classifier_revision"),
        "onnx",
        "paper_classifier",
    ),
)
_BY_ENV = {spec.env: spec for spec in MODEL_SETTINGS}

_LOCK = threading.Lock()
_LOADED: dict[str, dict[str, Any]] = {}
_HASH_CHUNK = 1 << 20


def _default(attr: tuple[str, ...]) -> Any:
    from bibr.config import GlobalSettings, MlOptions

    cls, name = (MlOptions, attr[1]) if attr[0] == "ml" else (GlobalSettings, attr[0])
    return cls.model_fields[name].default


def _read(settings: Any, attr: tuple[str, ...]) -> Any:
    obj = settings
    for part in attr:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _text(value: Any) -> str | None:
    """A setting value as text; anything else (a test double, None, "") is no value."""
    if isinstance(value, Path):
        value = str(value)
    return value if isinstance(value, str) and value else None


def setting_value(settings: Any, env: str) -> tuple[str | None, str | None]:
    """``(value, revision)`` of a model setting as configured."""
    spec = _BY_ENV[env]
    value = _text(_read(settings, spec.attr))
    revision = _text(_read(settings, spec.revision_attr)) if spec.revision_attr else None
    return value, revision


def is_model_override(settings: Any, env: str) -> bool:
    """Whether the model setting ``env`` names something other than its shipped default.

    An unset (null) model is not an override: it turns the model off, it loads nothing.
    """
    spec = _BY_ENV[env]
    value, revision = setting_value(settings, env)
    if value is None:
        return False
    if value != _default(spec.attr):
        return True
    if spec.revision_attr is None:
        return False
    return revision != (_text(_default(spec.revision_attr)))


def model_overrides(settings: Any) -> list[ModelSetting]:
    """The model settings that are overrides, in :data:`MODEL_SETTINGS` order."""
    return [spec for spec in MODEL_SETTINGS if is_model_override(settings, spec.env)]


def _is_local(value: str) -> bool:
    from bibr.utils.ml_runtime import is_hub_repo_id

    repo, _, filename = value.partition(":")
    return not is_hub_repo_id(repo, filename=filename or None, warn=False)


def onnx_bundle_files(bundle_dir: str | Path) -> list[Path]:
    """The manifest and every file it lists, as the ONNX loaders read them."""
    from bibr.utils.ml_runtime import ONNX_MANIFEST, ONNX_MODEL, read_onnx_manifest

    bundle_dir = Path(bundle_dir)
    manifest = read_onnx_manifest(bundle_dir)
    names = list(manifest.get("files") or [])
    for key, fallback in (("model_file", ONNX_MODEL), ("tokenizer_file", None)):
        name = manifest.get(key, fallback)
        if name and name not in names:
            names.append(name)
    return [bundle_dir / ONNX_MANIFEST, *(bundle_dir / name for name in names)]


def _check_onnx(spec: ModelSetting, value: str) -> None:
    from bibr.utils.ml_runtime import local_onnx_bundle, read_onnx_manifest, torch_available

    path = Path(value).expanduser()
    bundle = local_onnx_bundle(path)
    if bundle is None:
        if spec.env == "NER_PARSER_CKPT" and path.is_file() and torch_available():
            return  # a torch checkpoint, loaded by the torch parser
        raise ConfigurationError(
            f"{spec.env}={value!r}: no ONNX bundle there (want bibr_onnx.json and model.onnx "
            "in the directory itself or in its onnx/ subdirectory)"
        )
    manifest = read_onnx_manifest(bundle)
    if spec.onnx_model is not None and manifest.get("model") != spec.onnx_model:
        raise ConfigurationError(
            f"{spec.env}={value!r}: the bundle is a {manifest.get('model')!r} model, not a "
            f"{spec.onnx_model!r} one"
        )
    if spec.env == "ML_FRONT_TAGGER_MODEL_ID":
        from bibr.front_tagger.contract import CONTRACT_VERSION, FEATURE_KEYS

        keys = tuple(manifest.get("feature_keys") or ())
        if keys != FEATURE_KEYS or manifest.get("contract_version") != CONTRACT_VERSION:
            raise ConfigurationError(
                f"{spec.env}={value!r}: the bundle uses tagger contract "
                f"{manifest.get('contract_version')!r}; this bibr reads contract {CONTRACT_VERSION}"
            )
    missing = [p.name for p in onnx_bundle_files(bundle) if not p.is_file()]
    if missing:
        raise ConfigurationError(f"{spec.env}={value!r}: the bundle lacks {', '.join(missing)}")


def _check_local(spec: ModelSetting, value: str) -> None:
    path = Path(value).expanduser()
    if not path.exists():
        raise ConfigurationError(f"{spec.env}={value!r}: no such file or directory")
    if spec.kind == "onnx":
        _check_onnx(spec, value)
    elif spec.kind == "geom_tagger":
        from bibr.extract.geom_tagger import GeomLineTagger

        GeomLineTagger(path)
    elif spec.kind == "joblib":
        if not path.is_file():
            raise ConfigurationError(f"{spec.env}={value!r}: name the .joblib bundle file itself")
        # The loaders check the bundle's kind and feature contract; both bundles are small.
        try:
            if spec.env == "REF_GEOM_SEG_MODEL_ID":
                from bibr.extract.geom_segmenter import GeomSegmenter

                GeomSegmenter(str(path))
            else:
                from bibr.extract.front_role import FrontRoleClassifier

                FrontRoleClassifier(str(path))
        except ConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported as the override's problem
            raise ConfigurationError(f"{spec.env}={value!r}: cannot be loaded: {exc}") from exc
    elif not path.is_file():
        raise ConfigurationError(f"{spec.env}={value!r}: name the checkpoint file itself")


def check_model_overrides(settings: Any, *, fast: bool = False) -> list[str]:
    """One line per override with a local path that this run cannot load; empty when fine.

    ``fast`` is a fast run, which turns the front-matter tagger on for itself. A
    model the run leaves off (the tagger outside primary/only/fast mode, the
    front-role classifier with ``ML_FRONT_ROLE_ENABLED=false``) is not checked.
    """
    problems: list[str] = []
    ml = getattr(settings, "ml", None)
    for spec in model_overrides(settings):
        tagger_off = getattr(ml, "front_tagger_mode", "off") == "off"
        if spec.env == "ML_FRONT_TAGGER_MODEL_ID" and not fast and tagger_off:
            continue
        if (
            spec.env == "ML_FRONT_ROLE_MODEL_ID"
            and getattr(ml, "front_role_enabled", True) is False
        ):
            continue
        value, _revision = setting_value(settings, spec.env)
        if value is None or not _is_local(value):
            continue
        try:
            _check_local(spec, value)
        except ConfigurationError as exc:
            problems.append(str(exc))
    return problems


def override_load_error(settings: Any, env: str, exc: BaseException) -> ConfigurationError | None:
    """The error to raise when an override's model failed to load, or None for a default.

    Callers whose default model may be missing (and then go on without it) raise
    this instead when the setting is an override.
    """
    if not is_model_override(settings, env):
        return None
    spec = _BY_ENV[env]
    value, revision = setting_value(settings, env)
    where = f"{value}@{revision}" if revision and not _is_local(value or "") else value
    return ConfigurationError(
        f"{spec.env}={where!r} names a {spec.label} that could not be loaded "
        f"({type(exc).__name__}: {exc}). An overridden model never falls back to the shipped "
        f"one; fix the path, or unset {spec.env} to use the default."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def record_model_load(settings: Any, env: str, files: Iterable[str | Path]) -> dict | None:
    """Record which files an override loaded (no-op for a default); returns the record."""
    if not is_model_override(settings, env):
        return None
    spec = _BY_ENV[env]
    value, revision = setting_value(settings, env)
    rows = []
    for name in files:
        path = Path(name)
        if path.is_file():
            rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
        else:  # a directory a loader read as a whole (a torch checkpoint dir)
            rows.append({"path": str(path), "bytes": None, "sha256": None})
    record = {
        "setting": spec.env,
        "model": spec.label,
        "value": value,
        "revision": revision,
        "files": rows,
    }
    with _LOCK:
        _LOADED[spec.env] = record
    logger.info(
        "Model override %s=%s: %s loaded from %s",
        spec.env,
        value,
        spec.label,
        ", ".join(f"{Path(r['path']).name} sha256 {(r['sha256'] or '-')[:16]}" for r in rows),
    )
    return record


def loaded_model_overrides() -> list[dict]:
    """The records of every override loaded in this process, in :data:`MODEL_SETTINGS` order."""
    with _LOCK:
        return [dict(_LOADED[spec.env]) for spec in MODEL_SETTINGS if spec.env in _LOADED]


def reset_loaded_model_overrides() -> None:
    """Test hook: forget the load records."""
    with _LOCK:
        _LOADED.clear()


__all__ = [
    "MODEL_SETTINGS",
    "ModelSetting",
    "check_model_overrides",
    "is_model_override",
    "loaded_model_overrides",
    "model_overrides",
    "onnx_bundle_files",
    "override_load_error",
    "record_model_load",
    "reset_loaded_model_overrides",
    "setting_value",
]
