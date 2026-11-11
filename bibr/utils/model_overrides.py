"""Model overrides: a model setting pointed at something other than its shipped default.

Each model bibr loads is named by one setting (``NER_PARSER_CKPT``,
``ML_FRONT_ROLE_MODEL_ID``, ...; :data:`MODEL_SETTINGS`), which takes a Hub id
or a local path. Set to anything but its default (or with a revision other than
its pinned one), that setting is an *override*: a candidate model, a mirror, an
offline copy or another revision of the shipped one. Overrides exist so a run
can be compared with the shipped models, so these rules hold for them:

1. **Recorded.** Every override is listed in the run's record
   (:func:`begin_run`, :func:`run_records`; ``bibr batch`` writes it to
   ``run_info.json`` under ``model_overrides``) with ``loaded`` true and the
   sha256 of every file it read, or ``loaded`` false and why.
2. **A model that does not exist stops the run.** When the named model cannot
   be loaded for a permanent reason (a missing path, a repo, revision or file
   the Hub does not have, an unreadable or wrong-kind file), the loader raises
   :class:`~bibr.exceptions.ModelOverrideError` instead of going on without
   the model, and the extraction code passes it to the caller. A comparison
   must never report the shipped path, or no model, under the candidate's name.
3. **Everything else behaves as for the shipped model.** A network failure
   while fetching from the Hub, and a missing optional dependency (``torch``
   for the CRF segmenter), are handled exactly as for a default setting: the
   loaders retry or leave the tier out as they always did. The record says so.
4. **Checked before the run.** ``bibr chew`` and ``bibr batch`` check every
   local-path override the run will load (:func:`check_model_overrides`) before
   the first paper. A Hub id is checked when it loads.

With every model setting at its default nothing here changes what bibr does.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from bibr.exceptions import ConfigurationError, ModelOverrideError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSetting:
    """One model setting: its env name, where it sits in the settings, and its format."""

    env: str
    label: str
    attr: tuple[str, ...]
    revision_attr: tuple[str, ...] | None
    # "onnx" (an onnx/ bundle; ``onnx_model`` is the manifest's ``model`` field, or None for a
    # bundle checked otherwise), "joblib" (one bundle file), "geom_tagger", "text_segmenter",
    # "checkpoint" (a file).
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
        ("REF_GEOM_TAGGER_REVISION",),
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
        "REF_TEXT_SEG_MODEL_ID",
        "text reference segmenter",
        ("REF_TEXT_SEG_MODEL_ID",),
        ("REF_TEXT_SEG_REVISION",),
        "text_segmenter",
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

_HASH_CHUNK = 1 << 20
NOT_USED = "not used in this run"


# -- what is an override ---------------------------------------------------------------


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


def is_local_model(value: str) -> bool:
    """Whether a model value is a local path (anything not shaped like a Hub repo id)."""
    from bibr.utils.ml_runtime import is_hub_repo_id

    repo, _, filename = value.partition(":")
    return not is_hub_repo_id(repo, filename=filename or None, warn=False)


# -- why a load failed -----------------------------------------------------------------

LoadFailure = Literal["permanent", "transient", "missing_dependency"]

# huggingface_hub's answers that the repo, revision or file does not exist (or may
# not be read): no later attempt can fix them. LocalEntryNotFoundError subclasses
# EntryNotFoundError but means "Hub unreachable and not cached", so it is checked first.
_HUB_TRANSIENT = frozenset({"LocalEntryNotFoundError"})
_HUB_PERMANENT = frozenset(
    {
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "EntryNotFoundError",
        "RemoteEntryNotFoundError",
        "GatedRepoError",
        "DisabledRepoError",
        "HFValidationError",
    }
)
_MAX_CHAIN = 6
# Packages of the optional torch extra. An ImportError naming one means "this
# install has no such tier", as for the default; any other ImportError (a pickled
# bundle that names a module bibr does not have) is the model's problem.
_OPTIONAL_PACKAGES = frozenset({"torch", "torchvision", "transformers", "safetensors"})


def _missing_optional(link: BaseException) -> bool:
    if not isinstance(link, ImportError):
        return False
    from bibr.utils.ml_extra import TORCH_EXTRA

    name = (getattr(link, "name", None) or "").split(".")[0]
    return name in _OPTIONAL_PACKAGES or f"'{TORCH_EXTRA}' extra" in str(link)


def _chain(exc: BaseException) -> list[BaseException]:
    out: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(out) < _MAX_CHAIN and all(current is not e for e in out):
        out.append(current)
        current = current.__cause__ or current.__context__
    return out


def load_failure_kind(exc: BaseException) -> LoadFailure:
    """Whether a model load failed for good, on the network, or for a missing dependency."""
    from bibr.utils.transient import is_transient_network_error

    for link in _chain(exc):
        names = {klass.__name__ for klass in type(link).__mro__}
        if names & _HUB_TRANSIENT:
            # Unless the cache recorded the Hub's 404 at a pinned commit (hf_cache).
            return "permanent" if "recorded by an earlier request" in str(link) else "transient"
        if names & _HUB_PERMANENT:
            return "permanent"
    if is_transient_network_error(exc):
        return "transient"
    if any(_missing_optional(link) for link in _chain(exc)):
        return "missing_dependency"
    return "permanent"


def override_load_error(settings: Any, env: str, exc: BaseException) -> ModelOverrideError | None:
    """The error to raise when a model failed to load, or ``None`` to behave as for a default.

    ``None`` for a default setting, and for an override whose failure is a
    network error or a missing optional dependency (rule 3); that failure is
    written to the run's record instead.
    """
    if not is_model_override(settings, env):
        return None
    if isinstance(exc, ModelOverrideError):
        return exc
    kind = load_failure_kind(exc)
    if kind != "permanent":
        reason = "torch extra not installed" if kind == "missing_dependency" else "network error"
        record_model_unavailable(settings, env, f"{reason}: {type(exc).__name__}: {exc}")
        return None
    spec = _BY_ENV[env]
    value, revision = setting_value(settings, env)
    where = f"{value}@{revision}" if revision and not is_local_model(value or "") else value
    return ModelOverrideError(
        f"{spec.env}={where!r} names a {spec.label} that cannot be loaded "
        f"({type(exc).__name__}: {exc}). A model setting changed from its default never "
        f"falls back to the shipped model; fix the setting, or unset {spec.env}."
    )


# -- bundle files ------------------------------------------------------------------------


def bundle_member(bundle_dir: str | Path, name: Any) -> Path:
    """``bundle_dir / name`` for a file name read from a manifest, kept inside the bundle."""
    if not isinstance(name, str) or not name:
        raise ConfigurationError(f"{bundle_dir}: manifest file name {name!r} is not a file name")
    pure = PurePosixPath(name.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or Path(name).is_absolute():
        raise ConfigurationError(
            f"{bundle_dir}: manifest file name {name!r} points outside the bundle directory"
        )
    return Path(bundle_dir) / Path(*pure.parts)


def onnx_bundle_files(bundle_dir: str | Path) -> list[Path]:
    """The manifest and every file it lists, as the ONNX loaders read them."""
    from bibr.utils.ml_runtime import ONNX_MANIFEST, ONNX_MODEL, read_onnx_manifest

    bundle_dir = Path(bundle_dir)
    manifest = read_onnx_manifest(bundle_dir)
    listed = manifest.get("files") or []
    if not isinstance(listed, list):
        raise ConfigurationError(f"{bundle_dir}: manifest 'files' must be a list of file names")
    names = list(listed)
    for key, fallback in (("model_file", ONNX_MODEL), ("tokenizer_file", None)):
        name = manifest.get(key, fallback)
        if name and name not in names:
            names.append(name)
    return [bundle_dir / ONNX_MANIFEST, *(bundle_member(bundle_dir, name) for name in names)]


# -- preflight -----------------------------------------------------------------------------


def models_in_use(settings: Any) -> set[str]:
    """The model settings a run with these (effective) settings can load.

    ``settings`` are the run's: for a fast run, :func:`bibr.run_mode.fast_settings`
    of them, with the run's ``--refs`` / ``--ref-seg`` applied. Reference models
    only when references are parsed (``REF_PARSE_STRATEGY`` other than ``off``):
    the reference parser and the last-resort text segmenter of every
    segmentation strategy (the CRF, or the token tagger that replaces it when
    ``REF_TEXT_SEG_MODEL_ID`` is set), and the geometry segmenter and the line
    tagger only for ``REF_SEG_STRATEGY=geom``.
    The front-matter tagger when its mode is on, the front-role classifier
    unless ``ML_FRONT_ROLE_ENABLED=false``, the classifiers when configured.
    """
    ml = getattr(settings, "ml", None)
    used = {"ML_SECTION_CLASSIFIER_MODEL_ID", "ML_PAPER_CLASSIFIER_MODEL_ID"}
    if getattr(ml, "front_role_enabled", True) is not False:
        used.add("ML_FRONT_ROLE_MODEL_ID")
    if getattr(ml, "front_tagger_mode", "off") != "off":
        used.add("ML_FRONT_TAGGER_MODEL_ID")
    parse = str(getattr(settings, "REF_PARSE_STRATEGY", None) or "ner").lower()
    if parse != "off":
        used.add("NER_PARSER_CKPT")
        text_seg = _text(getattr(settings, "REF_TEXT_SEG_MODEL_ID", None))
        used.add("REF_TEXT_SEG_MODEL_ID" if text_seg else "NER_SEG_CKPT")
        if str(getattr(settings, "REF_SEG_STRATEGY", None) or "geom").lower() == "geom":
            used |= {"REF_GEOM_SEG_MODEL_ID", "REF_GEOM_TAGGER_MODEL_ID"}
    return used


def _check_onnx(spec: ModelSetting, value: str) -> None:
    from bibr.utils.ml_runtime import local_onnx_bundle, read_onnx_manifest, torch_available

    path = Path(value).expanduser()
    bundle = local_onnx_bundle(path)
    if bundle is None:
        if spec.env == "NER_PARSER_CKPT" and path.is_file() and torch_available():
            return  # a torch checkpoint, loaded by the torch parser
        raise ConfigurationError(
            "no ONNX bundle there (want bibr_onnx.json and model.onnx in the directory "
            "itself or in its onnx/ subdirectory)"
        )
    manifest = read_onnx_manifest(bundle)
    if spec.onnx_model is not None and manifest.get("model") != spec.onnx_model:
        raise ConfigurationError(
            f"the bundle is a {manifest.get('model')!r} model, not a {spec.onnx_model!r} one"
        )
    if spec.env == "ML_FRONT_TAGGER_MODEL_ID":
        from bibr.front_tagger.contract import CONTRACT_VERSION, FEATURE_KEYS

        keys = tuple(manifest.get("feature_keys") or ())
        if keys != FEATURE_KEYS or manifest.get("contract_version") != CONTRACT_VERSION:
            raise ConfigurationError(
                f"the bundle uses tagger contract {manifest.get('contract_version')!r}; this "
                f"bibr reads contract {CONTRACT_VERSION}"
            )
    missing = [p.name for p in onnx_bundle_files(bundle) if not p.is_file()]
    if missing:
        raise ConfigurationError(f"the bundle lacks {', '.join(missing)}")


def _check_local(spec: ModelSetting, value: str) -> None:
    path = Path(value).expanduser()
    if not path.exists():
        raise ConfigurationError("no such file or directory")
    if spec.kind == "onnx":
        _check_onnx(spec, value)
    elif spec.kind == "geom_tagger":
        from bibr.extract.geom_tagger import GeomLineTagger

        GeomLineTagger(path)  # reads the bundle, opens the graph and runs it once
    elif spec.kind == "text_segmenter":
        # The files, the manifest and the tokenizer; the graph (hundreds of MB)
        # is opened and run once when the tier first loads.
        from tokenizers import Tokenizer

        from bibr.extract.text_segmenter import read_segmenter_manifest, segmenter_bundle_files

        _graph, manifest, tokenizer = segmenter_bundle_files(path)
        read_segmenter_manifest(manifest)
        Tokenizer.from_file(str(tokenizer))
    elif spec.kind == "joblib":
        if not path.is_file():
            raise ConfigurationError("name the .joblib bundle file itself")
        # The loaders check the bundle's kind and feature contract; both bundles are small.
        if spec.env == "REF_GEOM_SEG_MODEL_ID":
            from bibr.extract.geom_segmenter import GeomSegmenter

            GeomSegmenter(str(path))
        else:
            from bibr.extract.front_role import FrontRoleClassifier

            FrontRoleClassifier(str(path))
    elif not path.is_file():
        raise ConfigurationError("name the checkpoint file itself")


def text_segmenter_conflict(settings: Any) -> str | None:
    """The problem when both the text segmenter and an overridden CRF are set, else ``None``.

    The text segmenter replaces the CRF, so a run with both would measure one of
    them under the other's name.
    """
    text_seg, _revision = setting_value(settings, "REF_TEXT_SEG_MODEL_ID")
    if text_seg is None or not is_model_override(settings, "NER_SEG_CKPT"):
        return None
    crf, _revision = setting_value(settings, "NER_SEG_CKPT")
    return (
        f"REF_TEXT_SEG_MODEL_ID={text_seg!r} and NER_SEG_CKPT={crf!r} are both set: the text "
        "segmenter replaces the CRF segmenter, so set one of them (unset REF_TEXT_SEG_MODEL_ID "
        "to run the CRF, or leave NER_SEG_CKPT at its default)"
    )


def check_model_overrides(settings: Any) -> tuple[list[str], list[str]]:
    """``(problems, warnings)`` for the local-path overrides this run will load.

    One line per problem: the run cannot load that model. A model the run does
    not load (:func:`models_in_use`) is not checked, and a Hub id is checked
    when it loads. A warning is a model the run will go without, as it would
    with the default (the CRF segmenter without ``torch``). Never raises.
    """
    from bibr.utils.ml_runtime import torch_available

    problems: list[str] = []
    warnings: list[str] = []
    try:
        used = models_in_use(settings)
        specs = [spec for spec in model_overrides(settings) if spec.env in used]
    except Exception as exc:  # noqa: BLE001 - the preflight reports, it never crashes
        return [f"model settings could not be read ({type(exc).__name__}: {exc})"], []
    conflict = text_segmenter_conflict(settings)
    if conflict and "REF_TEXT_SEG_MODEL_ID" in used:
        problems.append(conflict)
    for spec in specs:
        value, _revision = setting_value(settings, spec.env)
        if value is None or not is_local_model(value):
            continue
        if spec.env == "NER_SEG_CKPT" and not torch_available():
            warnings.append(
                f"{spec.env}={value!r}: the CRF segmenter needs the torch extra, which is not "
                "installed; this run goes without it, as it would with the default"
            )
            continue
        try:
            _check_local(spec, value)
        except Exception as exc:  # noqa: BLE001 - any failure is the override's problem
            detail = (
                str(exc) if isinstance(exc, ConfigurationError) else f"{type(exc).__name__}: {exc}"
            )
            problems.append(f"{spec.env}={value!r}: {detail}".splitlines()[0][:400])
    return problems, warnings


# -- the run's record ----------------------------------------------------------------------

_LOCK = threading.Lock()
_RUN: dict[str, dict[str, Any]] = {}
# sha256 rows per (setting, value, revision), so a model used by every paper is hashed once.
_IDENTITY: dict[tuple[str, str | None, str | None], list[dict[str, Any]]] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _file_rows(files: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for name in files:
        path = Path(name)
        if path.is_file():
            rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
        else:  # a directory a loader read as a whole (a torch checkpoint dir)
            rows.append({"path": str(path), "bytes": None, "sha256": None})
    return rows


def _entry(settings: Any, spec: ModelSetting) -> dict[str, Any]:
    value, revision = setting_value(settings, spec.env)
    if isinstance(value, str) and is_local_model(value):
        revision = None  # a local model is read as it is; the (default) revision is not used
    return {
        "setting": spec.env,
        "model": spec.label,
        "value": value,
        "revision": revision,
        "loaded": False,
        "reason": NOT_USED,
        "files": [],
    }


def begin_run(settings: Any) -> None:
    """Start a run's record: every override in ``settings``, none loaded yet."""
    with _LOCK:
        _RUN.clear()
        for spec in model_overrides(settings):
            _RUN[spec.env] = _entry(settings, spec)


def record_model_load(
    settings: Any, env: str, files: Iterable[str | Path] | Callable[[], Iterable[str | Path]]
) -> dict | None:
    """Note that the override ``env`` was loaded (or used again); no-op for a default.

    ``files`` (or a callable returning them) are the files the loader read; they
    are hashed once per setting value and revision in a process.
    """
    if not is_model_override(settings, env):
        return None
    spec = _BY_ENV[env]
    value, revision = setting_value(settings, env)
    key = (env, value, revision)
    with _LOCK:
        rows = _IDENTITY.get(key)
    if rows is None:
        rows = _file_rows(files() if callable(files) else files)
        with _LOCK:
            _IDENTITY[key] = rows
        logger.info(
            "Model override %s=%s: %s loaded from %s",
            spec.env,
            value,
            spec.label,
            ", ".join(f"{Path(r['path']).name} sha256 {(r['sha256'] or '-')[:16]}" for r in rows),
        )
    with _LOCK:
        entry = _RUN.get(env) or _entry(settings, spec)
        if entry["loaded"] and entry["files"] == rows:
            return dict(entry)
        entry.update(loaded=True, reason=None, files=[dict(r) for r in rows])
        _RUN[env] = entry
        return dict(entry)


def record_model_unavailable(settings: Any, env: str, reason: str) -> None:
    """Note that the override ``env`` was not loaded, and why; no-op for a default."""
    if not is_model_override(settings, env):
        return
    with _LOCK:
        entry = _RUN.get(env) or _entry(settings, _BY_ENV[env])
        if not entry["loaded"]:
            entry["reason"] = reason[:300]
            _RUN[env] = entry


def run_records() -> list[dict]:
    """The run's record, one entry per override, in :data:`MODEL_SETTINGS` order."""
    with _LOCK:
        return [dict(_RUN[spec.env]) for spec in MODEL_SETTINGS if spec.env in _RUN]


def loaded_model_overrides() -> list[dict]:
    """The entries of :func:`run_records` whose model was loaded."""
    return [entry for entry in run_records() if entry["loaded"]]


def reset_loaded_model_overrides() -> None:
    """Test hook: forget the run's record and the cached file hashes."""
    with _LOCK:
        _RUN.clear()
        _IDENTITY.clear()


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "MODEL_SETTINGS",
    "NOT_USED",
    "ModelSetting",
    "begin_run",
    "bundle_member",
    "check_model_overrides",
    "is_local_model",
    "is_model_override",
    "load_failure_kind",
    "loaded_model_overrides",
    "model_overrides",
    "models_in_use",
    "onnx_bundle_files",
    "override_load_error",
    "record_model_load",
    "record_model_unavailable",
    "reset_loaded_model_overrides",
    "run_records",
    "setting_value",
    "text_segmenter_conflict",
]
