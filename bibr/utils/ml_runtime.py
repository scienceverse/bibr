"""Runtime selection for bibr's local models: ONNX Runtime first, torch as fallback.

Every model that bibr serves locally (layout detector, section and paper
classifiers, NER reference parser) has two implementations with the same
interface — an ONNX Runtime class (core install) and the original torch class
(``torch`` extra). This module owns the one rule that picks between them,
driven by ``ML_RUNTIME``:

- ``auto`` (default): the ONNX bundle when it resolves, else torch when it is
  importable, else a :class:`~bibr.exceptions.ConfigurationError` that names
  the extra and the setting that points at a bundle.
- ``onnx``: the bundle must resolve.
- ``torch``: torch must be importable.

An ONNX bundle is an ``onnx/`` directory holding ``bibr_onnx.json`` (the
manifest), ``model.onnx`` and, for text models, ``tokenizer.json``. It lives
either next to the torch artifacts in the model's Hub repo at the pinned
revision, or in a local directory named by the model setting.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from bibr.exceptions import ConfigurationError
from bibr.utils.ml_extra import TORCH_EXTRA_HINT

logger = logging.getLogger(__name__)

Runtime = Literal["onnx", "torch"]

ONNX_DIRNAME = "onnx"
ONNX_MANIFEST = "bibr_onnx.json"
ONNX_MODEL = "model.onnx"
ONNX_TOKENIZER = "tokenizer.json"
MANIFEST_SCHEMA_VERSION = 1


def torch_available() -> bool:
    """True when torch is installed (checked without importing it)."""
    return importlib.util.find_spec("torch") is not None


def onnxruntime_available() -> bool:
    return importlib.util.find_spec("onnxruntime") is not None


def read_onnx_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    """Load and sanity-check ``bibr_onnx.json`` from a bundle directory."""
    bundle_dir = Path(bundle_dir)
    path = bundle_dir / ONNX_MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"ONNX bundle manifest {path} is unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ConfigurationError(f"ONNX bundle manifest {path} must contain a JSON object")
    version = manifest.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ConfigurationError(
            f"ONNX bundle manifest {path} has schema_version {version!r}; this bibr "
            f"understands {MANIFEST_SCHEMA_VERSION}"
        )
    model_file = bundle_dir / manifest.get("model_file", ONNX_MODEL)
    if not model_file.is_file():
        raise ConfigurationError(f"ONNX bundle {bundle_dir} is missing {model_file.name}")
    return manifest


def _local_bundle(path: Path) -> Path | None:
    if path.is_dir():
        bundle = path if path.name == ONNX_DIRNAME else path / ONNX_DIRNAME
    elif path.is_file():
        bundle = path.parent / ONNX_DIRNAME
    else:
        return None
    return (
        bundle if (bundle / ONNX_MODEL).is_file() and (bundle / ONNX_MANIFEST).is_file() else None
    )


def find_onnx_bundle(
    model_id: str | None, revision: str | None = None, *, label: str = "model"
) -> Path | None:
    """Locate the ``onnx/`` bundle for ``model_id``, or ``None``.

    ``model_id`` may be a local directory (the bundle is ``<dir>/onnx``, or
    the directory itself when it is already named ``onnx``), a local file (a
    sibling ``onnx/`` directory, for ``NER_PARSER_CKPT`` style checkpoint
    paths), or a Hub repo id — optionally ``org/repo:filename`` — resolved at
    ``revision``. Hub lookups download the manifest first and then every file
    it lists, so a fully cached bundle resolves offline; a repo, revision or
    file that does not exist, or an unreachable Hub with nothing cached, yields
    ``None`` (logged at INFO — this is the normal state until the artifacts
    are published).
    """
    if not model_id:
        return None
    local = Path(str(model_id)).expanduser()
    if local.exists():
        return _local_bundle(local)

    repo_id = str(model_id).split(":", 1)[0]
    if repo_id.count("/") != 1:
        return None

    from bibr.utils.hf_cache import hf_download_or_cached

    try:
        manifest_path = Path(
            hf_download_or_cached(repo_id, f"{ONNX_DIRNAME}/{ONNX_MANIFEST}", revision)
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files") or [ONNX_MODEL]
        for name in files:
            hf_download_or_cached(repo_id, f"{ONNX_DIRNAME}/{name}", revision)
    except Exception as exc:  # noqa: BLE001 — absence is the expected outcome
        logger.info(
            "No ONNX bundle for %s (%s@%s): %s: %s",
            label,
            repo_id,
            revision or "main",
            type(exc).__name__,
            str(exc).splitlines()[0][:200] if str(exc) else "",
        )
        return None
    return manifest_path.parent


def resolve_runtime(
    label: str,
    *,
    settings,
    bundle: Callable[[], Path | None],
    bundle_hint: str,
) -> tuple[Runtime, Path | None]:
    """Apply the ``ML_RUNTIME`` rule for one model.

    ``bundle`` is called at most once (only when ONNX is a candidate) and
    returns the bundle directory or ``None``. ``bundle_hint`` completes the
    sentence "… or <hint>" in the error raised when neither runtime is usable.
    """
    mode = getattr(getattr(settings, "ml", None), "runtime", "auto")
    if mode == "torch":
        if torch_available():
            return "torch", None
        raise ConfigurationError(
            f"{label}: ML_RUNTIME=torch but torch is not installed. Install it with "
            f"{TORCH_EXTRA_HINT}, or set ML_RUNTIME=auto to use an ONNX bundle."
        )

    bundle_dir = bundle()
    if bundle_dir is not None:
        return "onnx", bundle_dir
    if mode == "onnx":
        raise ConfigurationError(
            f"{label}: ML_RUNTIME=onnx but no ONNX bundle was found; {bundle_hint}."
        )
    if torch_available():
        return "torch", None
    raise ConfigurationError(
        f"{label}: no ONNX bundle was found and torch is not installed. Either install the "
        f"torch extra ({TORCH_EXTRA_HINT}) or {bundle_hint}."
    )


def hub_bundle_hint(setting: str, model_id: str | None, revision: str | None) -> str:
    """Standard ``bundle_hint`` wording for a Hub-hosted model."""
    where = f"{model_id}@{revision or 'main'}" if model_id else "the configured repo"
    return (
        f"publish an onnx/ bundle to {where} or point {setting} at a local directory "
        "containing onnx/model.onnx and onnx/bibr_onnx.json"
    )
