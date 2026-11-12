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
2. **A model that does not load stops the run.** When the named model cannot
   be loaded, whatever the reason (a missing path, a repo, revision or file the
   Hub does not have, a Hub that cannot be reached with the files not in the
   local cache, an unreadable or wrong-kind file, a failed load probe, a
   missing optional dependency), the loader raises
   :class:`~bibr.exceptions.ModelOverrideError` instead of going on without
   the model, and the pipeline stops the run with it
   (:func:`find_model_override_error`): no paper gets a verdict for it, so a
   resumed ``bibr batch`` runs them all again once the setting is fixed. A
   comparison must never report the shipped path, or no model, under the
   candidate's name.
3. **An inference error is recorded per paper, and repeated ones stop the run.**
   A loaded override that raises on one paper takes that tier's error path for
   that paper (the path a model error takes, which is not always what a run
   without the override would do), and the paper's export says so with a
   ``MODEL_OVERRIDE_INFERENCE_FAILED`` warning. When it raised on
   :data:`INFERENCE_FAILURE_LIMIT` papers in a row, the run stops with
   :class:`~bibr.exceptions.ModelOverrideError` (:func:`note_inference_failure`).
   A resumed ``bibr batch`` runs the papers written with the warning again
   (:func:`bibr.batch.ledger.fell_back_on_override`).
4. **Checked before the run.** ``bibr chew`` and ``bibr batch`` load every
   override the run will use, a local path or a Hub id, before the first paper
   (:func:`check_model_overrides`), so a model that does not load stops the run
   before any paper is processed. An overridden ONNX model runs at every load
   on inputs in the shapes a paper sends (the full length and a short one, two
   batch sizes where the runtime batches), and its output shapes are checked
   against its manifest (:mod:`bibr.utils.onnx_probe`).

With every model setting at its default nothing here changes what bibr does:
the shipped models keep their own fallbacks (a network failure retried, a
missing ``torch`` leaving the CRF tier out).
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import itertools
import logging
import math
import threading
from collections.abc import Callable, Iterable, Iterator
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
        "ML_STATEMENT_TAGGER_MODEL_ID",
        "statement tagger",
        ("ml", "statement_tagger_model_id"),
        ("ml", "statement_tagger_revision"),
        "onnx",
        "statement_tagger",
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


_KIND_REASON = {
    "transient": "the Hub could not be reached and the files are not in the local cache",
    "missing_dependency": "the torch extra is not installed",
}


def override_load_error(settings: Any, env: str, exc: BaseException) -> ModelOverrideError | None:
    """The error to raise when a model failed to load, or ``None`` to behave as for a default.

    ``None`` only for a default setting. For an override every failure is the
    run's (rule 2), a network failure and a missing optional dependency
    included: the shipped model's retry or fallback would run the comparison on
    something other than the candidate. The failure is written to the run's
    record and to the run's pending failures (:func:`override_failure_scope`).
    """
    if not is_model_override(settings, env):
        return None
    if isinstance(exc, ModelOverrideError):
        _note_pending(exc)
        return exc
    kind = load_failure_kind(exc)
    detail = f"{type(exc).__name__}: {exc}"
    if kind in _KIND_REASON:
        detail = f"{_KIND_REASON[kind]}; {detail}"
    record_model_unavailable(settings, env, detail)
    spec = _BY_ENV[env]
    value, revision = setting_value(settings, env)
    where = f"{value}@{revision}" if revision and not is_local_model(value or "") else value
    error = ModelOverrideError(
        f"{spec.env}={where!r} names a {spec.label} that cannot be loaded "
        f"({detail}). A model setting changed from its default never "
        f"falls back to the shipped model; fix the setting, or unset {spec.env}."
    )
    _note_pending(error)
    return error


def find_model_override_error(exc: BaseException | None) -> ModelOverrideError | None:
    """The :class:`~bibr.exceptions.ModelOverrideError` ``exc`` is or was raised from, if any."""
    if exc is None:
        return None
    for link in _chain(exc):
        if isinstance(link, ModelOverrideError):
            return link
    return None


# -- the run's pending failures -----------------------------------------------------------
#
# A model override that fails stops the run. The loaders raise, and the stages
# pass the error on or record it on the paper (``FileState.original_error``),
# which :meth:`bibr.pipeline.pipeline.Pipeline.process_chunk` turns back into
# a raise. Code that catches every exception around a model call would still
# swallow it; for that, each failure is also noted here, in a list owned by the
# chunk that is running (a context variable, so concurrent chunks of a server
# do not see each other's), and the chunk raises it after the stage.

_PENDING: contextvars.ContextVar[list[ModelOverrideError] | None] = contextvars.ContextVar(
    "bibr_model_override_failures", default=None
)


def _note_pending(error: ModelOverrideError) -> None:
    pending = _PENDING.get()
    if pending is not None and all(error is not e for e in pending):
        pending.append(error)


@contextlib.contextmanager
def override_failure_scope() -> Iterator[list[ModelOverrideError]]:
    """Collect the model override failures of the code run inside (one chunk).

    Tasks and :func:`asyncio.to_thread` workers started inside inherit the
    scope; a scope already open (a chunk inside a chunk) is kept.
    """
    if _PENDING.get() is not None:
        yield _PENDING.get()  # type: ignore[misc]
        return
    pending: list[ModelOverrideError] = []
    token = _PENDING.set(pending)
    try:
        yield pending
    finally:
        _PENDING.reset(token)


def pending_override_failure(
    pending: list[ModelOverrideError], file_states: Iterable[Any] = ()
) -> ModelOverrideError | None:
    """The first override failure noted in ``pending`` or recorded on one of ``file_states``."""
    if pending:
        return pending[0]
    for fs in file_states:
        if (found := find_model_override_error(getattr(fs, "original_error", None))) is not None:
            return found
    return None


# -- inference failures ---------------------------------------------------------------------

# Papers in a row on which one overridden model raised at inference before the run stops.
INFERENCE_FAILURE_LIMIT = 3

_PAPER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "bibr_model_override_paper", default=None
)
_ANONYMOUS = itertools.count()
_STREAK_LOCK = threading.Lock()
# (setting, value, revision) -> the papers, in order, of the current run of failures.
_STREAKS: dict[tuple[str, str | None, str | None], list[str]] = {}


@contextlib.contextmanager
def paper_scope(paper: Any) -> Iterator[None]:
    """Name the paper the code run inside works on, for :func:`note_inference_failure`."""
    token = _PAPER.set(None if paper is None else str(paper))
    try:
        yield
    finally:
        _PAPER.reset(token)


def _paper_key(paper: Any) -> str:
    if paper is not None and str(paper):
        return str(paper)
    current = _PAPER.get()
    # A call outside any paper's scope counts as a paper of its own.
    return current if current else f"#{next(_ANONYMOUS)}"


def note_inference_failure(
    settings: Any, env: str, exc: BaseException, *, paper: Any = None
) -> Any:
    """Note that the model of ``env`` raised ``exc`` at inference on one paper.

    ``None`` for a default setting, which falls back as it always did. For an
    override, the :class:`~bibr.processing_warnings.ProcessingWarning` the
    caller records on the paper before it falls back for that paper; when the
    override has raised on :data:`INFERENCE_FAILURE_LIMIT` papers in a row
    (``paper``, else the paper of the enclosing :func:`paper_scope`), raises
    :class:`~bibr.exceptions.ModelOverrideError` instead.
    """
    if isinstance(exc, ModelOverrideError) or not is_model_override(settings, env):
        return None
    from bibr.processing_warnings import ProcessingWarning, WarningCode

    spec = _BY_ENV[env]
    value, revision = setting_value(settings, env)
    key = (env, value, revision)
    paper_key = _paper_key(paper)
    with _STREAK_LOCK:
        streak = _STREAKS.setdefault(key, [])
        if paper_key not in streak:
            streak.append(paper_key)
        papers = len(streak)
    if papers >= INFERENCE_FAILURE_LIMIT:
        error = ModelOverrideError(
            f"{spec.env}={value!r} names a {spec.label} that loaded but raised at inference "
            f"on {papers} papers in a row (last: {type(exc).__name__}: {str(exc)[:200]}). A "
            "model setting changed from its default never falls back to the shipped model "
            f"run-wide; fix the model, or unset {spec.env}."
        )
        _note_pending(error)
        raise error from exc
    logger.warning(
        "Model override %s: the %s raised on this paper (%s); this paper falls back",
        spec.env,
        spec.label,
        type(exc).__name__,
    )
    return ProcessingWarning(
        WarningCode.MODEL_OVERRIDE_INFERENCE_FAILED,
        f"{spec.env} ({spec.label}) raised {type(exc).__name__} on this paper; the paper "
        "took that tier's error path, which is not always what a run without the override "
        "would do",
    )


def note_inference_success(settings: Any, env: str, *, paper: Any = None) -> None:
    """Note that the model of ``env`` ran on a paper: a paper without a failure ends the streak."""
    if not _STREAKS or not is_model_override(settings, env):
        return
    value, revision = setting_value(settings, env)
    paper_key = _paper_key(paper)
    with _STREAK_LOCK:
        streak = _STREAKS.get((env, value, revision))
        if streak and paper_key not in streak:
            streak.clear()


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


_REFERENCE_MODELS = (
    "NER_PARSER_CKPT",
    "REF_GEOM_SEG_MODEL_ID",
    "REF_GEOM_TAGGER_MODEL_ID",
    "NER_SEG_CKPT",
    "REF_TEXT_SEG_MODEL_ID",
)


def unused_models(settings: Any, *, no_llm: bool = False) -> dict[str, str]:
    """``{setting: why}`` for each model setting a run with these settings never asks.

    ``settings`` are the run's: for a fast run, :func:`bibr.run_mode.fast_settings`
    of them, with the run's ``--refs`` / ``--ref-seg`` applied; ``no_llm`` is
    ``--no-llm``, which wins over fast. A ``--no-llm`` run extracts no metadata
    or references, types sections by the heading lookup alone and runs no
    statement tagger (``bibr.pipeline.stages.post_parse``), so it asks the
    front-role classifier and, when its mode is on, the front-matter tagger,
    and nothing else. Otherwise the reference models only when references are
    parsed (``REF_PARSE_STRATEGY`` other than ``off``): the reference parser and
    the last-resort text segmenter of every segmentation strategy (the CRF, or
    the token tagger that replaces it unless ``REF_TEXT_SEG_MODEL_ID`` is null),
    and the geometry segmenter and the line tagger only for
    ``REF_SEG_STRATEGY=geom``. The front-matter tagger when its mode is on,
    the front-role classifier unless ``ML_FRONT_ROLE_ENABLED=false``, the
    classifiers always, the statement tagger unless it is null.
    """
    ml = getattr(settings, "ml", None)
    unused: dict[str, str] = {}
    if getattr(ml, "front_role_enabled", True) is False:
        unused["ML_FRONT_ROLE_MODEL_ID"] = "ML_FRONT_ROLE_ENABLED=false turns it off"
    if getattr(ml, "front_tagger_mode", "off") == "off":
        unused["ML_FRONT_TAGGER_MODEL_ID"] = (
            "ML_FRONT_TAGGER_MODE=off; a fast run, or the mode primary or only, turns it on"
        )
    if not getattr(ml, "statement_tagger_model_id", None):
        unused["ML_STATEMENT_TAGGER_MODEL_ID"] = "it is set to null"
    if no_llm:
        unused["ML_SECTION_CLASSIFIER_MODEL_ID"] = (
            "--no-llm types sections by the heading lookup alone"
        )
        unused["ML_PAPER_CLASSIFIER_MODEL_ID"] = "--no-llm does not classify the paper"
        unused.setdefault("ML_STATEMENT_TAGGER_MODEL_ID", "--no-llm runs no statement tagger")
        for env in _REFERENCE_MODELS:
            unused[env] = "--no-llm extracts no references"
        return unused
    parse = str(getattr(settings, "REF_PARSE_STRATEGY", None) or "ner").lower()
    if parse == "off":
        for env in _REFERENCE_MODELS:
            unused[env] = "REF_PARSE_STRATEGY=off (--refs off) extracts no references"
        return unused
    if _text(getattr(settings, "REF_TEXT_SEG_MODEL_ID", None)):
        unused["NER_SEG_CKPT"] = "REF_TEXT_SEG_MODEL_ID is set and replaces the CRF segmenter"
    else:
        unused["REF_TEXT_SEG_MODEL_ID"] = "it is set to null"
    seg = str(getattr(settings, "REF_SEG_STRATEGY", None) or "geom").lower()
    if seg != "geom":
        for env in ("REF_GEOM_SEG_MODEL_ID", "REF_GEOM_TAGGER_MODEL_ID"):
            unused[env] = f"REF_SEG_STRATEGY={seg} (--ref-seg {seg}) skips the geometry tier"
    return unused


def models_in_use(settings: Any, *, no_llm: bool = False) -> set[str]:
    """The model settings a run with these (effective) settings asks (:func:`unused_models`)."""
    unused = unused_models(settings, no_llm=no_llm)
    return {spec.env for spec in MODEL_SETTINGS if spec.env not in unused}


# The classifiers' torch runtime reads a transformers directory (``from_pretrained``).
_TORCH_CLASSIFIERS = frozenset({"ML_SECTION_CLASSIFIER_MODEL_ID", "ML_PAPER_CLASSIFIER_MODEL_ID"})
_TORCH_WEIGHTS = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")


def _torch_runtime(settings: Any) -> bool:
    """Whether the classifiers would load with torch when no ONNX bundle is found."""
    from bibr.utils.ml_runtime import torch_available

    mode = getattr(getattr(settings, "ml", None), "runtime", "auto")
    return mode != "onnx" and torch_available()


def _check_torch_classifier(path: Path) -> None:
    """A transformers directory: its config and its weights (the load reads them)."""
    if not path.is_dir():
        raise ConfigurationError("name the model directory (config.json and its weights)")
    missing = [] if (path / "config.json").is_file() else ["config.json"]
    if not any((path / name).is_file() for name in _TORCH_WEIGHTS):
        missing.append(" or ".join(_TORCH_WEIGHTS))
    if missing:
        raise ConfigurationError(
            f"neither an ONNX bundle (bibr_onnx.json and model.onnx) nor a torch model "
            f"directory: it lacks {', '.join(missing)}"
        )


def _check_onnx(spec: ModelSetting, value: str, settings: Any = None) -> None:
    from bibr.utils.ml_runtime import local_onnx_bundle, read_onnx_manifest, torch_available

    path = Path(value).expanduser()
    bundle = local_onnx_bundle(path)
    if bundle is None:
        if spec.env == "NER_PARSER_CKPT" and path.is_file() and torch_available():
            return  # a torch checkpoint, loaded by the torch parser
        if spec.env in _TORCH_CLASSIFIERS and _torch_runtime(settings):
            _check_torch_classifier(path)  # loaded with from_pretrained
            return
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
    if spec.env == "ML_STATEMENT_TAGGER_MODEL_ID":
        from bibr.statement_tagger.runtime import StatementTagger, StatementTaggerError

        try:
            StatementTagger(bundle, threads=1)  # reads the manifest, opens the graph, runs it once
        except StatementTaggerError as exc:
            raise ConfigurationError(str(exc)) from exc


def _check_local(spec: ModelSetting, value: str, settings: Any = None) -> None:
    path = Path(value).expanduser()
    if not path.exists():
        raise ConfigurationError("no such file or directory")
    if spec.kind == "onnx":
        _check_onnx(spec, value, settings)
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
        "segmenter replaces the CRF segmenter, so set one of them (REF_TEXT_SEG_MODEL_ID=null "
        "runs the CRF, or leave NER_SEG_CKPT at its default)"
    )


def load_model_override(settings: Any, env: str, *, memory_mode: str | None = None) -> None:
    """Load the model of ``env`` as the run loads it, probe included.

    The reference models, the front-role classifier, the front-matter tagger
    and the statement tagger stay in their loaders' process caches, so the run that follows does not
    load them again; a classifier is loaded on the CPU and released (the
    pipeline places it). Raises what the loader raises: for an override, a
    :class:`~bibr.exceptions.ModelOverrideError` for any failure.
    """
    if env in _TORCH_CLASSIFIERS:
        import importlib

        module, loader = {
            "ML_SECTION_CLASSIFIER_MODEL_ID": (
                "bibr.structure.section_classifier_common",
                "load_section_classifier",
            ),
            "ML_PAPER_CLASSIFIER_MODEL_ID": (
                "bibr.structure.paper_classifier_common",
                "load_paper_classifier",
            ),
        }[env]
        value, revision = setting_value(settings, env)
        try:
            model = getattr(importlib.import_module(module), loader)(
                value, revision=revision or "main", device="cpu", settings=settings
            )
        except Exception as exc:
            if (strict := override_load_error(settings, env, exc)) is not None:
                raise strict from exc
            raise
        del model
        return
    if env == "ML_FRONT_ROLE_MODEL_ID":
        from bibr.extract.front_role import load_front_role_classifier

        load_front_role_classifier(settings)
        return
    if env == "ML_FRONT_TAGGER_MODEL_ID":
        from bibr.front_tagger.runtime import load_front_tagger

        load_front_tagger(settings)
        return
    if env == "ML_STATEMENT_TAGGER_MODEL_ID":
        from bibr.statement_tagger.pipeline import get_statement_tagger

        get_statement_tagger(settings)
        return
    from bibr.extract import ref_extractor

    if env == "NER_PARSER_CKPT":
        ref_extractor._get_ner_parser(settings, memory_mode)
    elif env == "NER_SEG_CKPT":
        ref_extractor._get_ner_segmenter(settings)
    elif env == "REF_GEOM_SEG_MODEL_ID":
        ref_extractor._get_geom_segmenter(settings)
    elif env == "REF_GEOM_TAGGER_MODEL_ID":
        ref_extractor._get_geom_tagger(settings)
    elif env == "REF_TEXT_SEG_MODEL_ID":
        ref_extractor._get_text_segmenter(settings)


def _problem(spec: ModelSetting, value: str, exc: BaseException) -> str:
    if isinstance(exc, ModelOverrideError):
        detail = str(exc)
    elif isinstance(exc, ConfigurationError):
        detail = f"{value!r}: {exc}"
    else:
        detail = f"{value!r}: {type(exc).__name__}: {exc}"
    if not detail.startswith(spec.env):
        detail = f"{spec.env}={detail}"
    return " ".join(detail.split())[:600]


def unused_override_warning(spec: ModelSetting, value: str | None, reason: str) -> str:
    """The start-of-run line for an override this run never asks."""
    return (
        f"{spec.env}={value!r} is set, but this run does not use the {spec.label} ({reason}); "
        f"the setting is not checked or loaded, and the run records it as {NOT_USED!r}"
    )


def check_model_overrides(
    settings: Any, *, load: bool = True, memory_mode: str | None = None, no_llm: bool = False
) -> tuple[list[str], list[str]]:
    """``(problems, warnings)`` for the overrides this run will load.

    One line per problem: the run cannot load that model. A model the run does
    not ask (:func:`unused_models`; ``no_llm`` for ``--no-llm``) is not checked
    or loaded: it gets one line in ``warnings`` that names the setting and why.
    A local path's files are checked first; then, with ``load`` (``bibr chew``
    and ``bibr batch``), every override, a local path or a Hub id, is loaded as
    the run loads it (:func:`load_model_override`): the Hub files are fetched or
    found in the cache, the graph is opened and probed. Without ``load`` only
    the local paths' files are checked. Never raises.
    """
    problems: list[str] = []
    warnings: list[str] = []
    try:
        unused = unused_models(settings, no_llm=no_llm)
        used = models_in_use(settings, no_llm=no_llm)
        overrides = model_overrides(settings)
        specs = [spec for spec in overrides if spec.env in used]
    except Exception as exc:  # noqa: BLE001 - the preflight reports, it never crashes
        return [f"model settings could not be read ({type(exc).__name__}: {exc})"], []
    conflict = text_segmenter_conflict(settings)
    if conflict and "REF_TEXT_SEG_MODEL_ID" in used:
        problems.append(conflict)
    for spec in overrides:
        if spec.env in unused and not (conflict and spec.env == "NER_SEG_CKPT"):
            value, _revision = setting_value(settings, spec.env)
            warnings.append(unused_override_warning(spec, value, unused[spec.env]))
    for spec in specs:
        value, _revision = setting_value(settings, spec.env)
        if value is None:
            continue
        try:
            if is_local_model(value):
                _check_local(spec, value, settings)
            elif not load:
                continue
            if load and not (conflict and spec.env == "REF_TEXT_SEG_MODEL_ID"):
                load_model_override(settings, spec.env, memory_mode=memory_mode)
        except Exception as exc:  # noqa: BLE001 - any failure is the override's problem
            problems.append(_problem(spec, value, exc))
    return problems, warnings


# -- the run's record ----------------------------------------------------------------------

_LOCK = threading.Lock()
_RUN: dict[str, dict[str, Any]] = {}
# sha256 rows per (setting, value, revision), so a model used by every paper is hashed once.
_IDENTITY: dict[tuple[str, str | None, str | None], list[dict[str, Any]]] = {}


_SHA_LOCK = threading.Lock()
# sha256 per (resolved path, size, mtime): a model file is hashed once per process,
# for the run's record and the export's provenance alike, and again only when it changes.
_SHA: dict[tuple[str, int, int], str] = {}


def _sha256(path: Path) -> str:
    """The file's sha256, from the process cache while its size and mtime are unchanged."""
    resolved = Path(path).resolve()
    st = resolved.stat()
    key = (str(resolved), st.st_size, st.st_mtime_ns)
    with _SHA_LOCK:
        cached = _SHA.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while block := handle.read(_HASH_CHUNK):
            digest.update(block)
    value = digest.hexdigest()
    with _SHA_LOCK:
        _SHA[key] = value
    return value


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
    with _STREAK_LOCK:
        _STREAKS.clear()


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
    """Test hook: forget the run's record, the cached file hashes and the inference streaks."""
    with _LOCK:
        _RUN.clear()
        _IDENTITY.clear()
    with _STREAK_LOCK:
        _STREAKS.clear()
    with _SHA_LOCK:
        _SHA.clear()


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "MODEL_SETTINGS",
    "NOT_USED",
    "INFERENCE_FAILURE_LIMIT",
    "ModelSetting",
    "begin_run",
    "bundle_member",
    "check_model_overrides",
    "find_model_override_error",
    "is_local_model",
    "is_model_override",
    "load_failure_kind",
    "load_model_override",
    "loaded_model_overrides",
    "model_overrides",
    "models_in_use",
    "note_inference_failure",
    "note_inference_success",
    "onnx_bundle_files",
    "override_failure_scope",
    "override_load_error",
    "paper_scope",
    "pending_override_failure",
    "record_model_load",
    "record_model_unavailable",
    "reset_loaded_model_overrides",
    "run_records",
    "setting_value",
    "text_segmenter_conflict",
    "unused_models",
    "unused_override_warning",
]
