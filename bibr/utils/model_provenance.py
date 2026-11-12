"""Which models a run was set to load, as the export records them (``extraction.settings.models``).

One row per model the run's settings load: the setting that names it, the
model's role, and its identity. A Hub model is its repo id and revision, the
revision resolved to a commit through the local Hub cache when it is a branch
or tag. A local model is a short name and the sha256 of each file of it, named
relative to the model's directory. The short name is the directory's own name
(its parent's for a bundle directory named ``onnx``) and ``@`` the first 12
hex digits of a hash over its files, so two different models of the same name
differ. It is never the path, and never a part of the home directory or the
user name: exports are shared, and a path tells the reader how the producing
machine is laid out and who ran it. ``override`` is true when the setting
differs from the shipped default (:mod:`bibr.utils.model_overrides`).

The rows are what the run was set up with, not a trace of one paper: a paper
may not reach every tier (the reference receipts in ``extraction.diagnostics``
say which ran). A model the install cannot run at all is left out: the CRF
segmenter without ``torch``. Building the rows never raises. A local model's
file list is read once per process and each file is hashed once per process
(keyed by path, size and mtime), so a run does not walk a model directory or
re-read its files per paper.
"""

from __future__ import annotations

import getpass
import hashlib
import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_COMMIT = re.compile(r"[0-9a-f]{40}")
_LOCK = threading.Lock()
# (setting, value, revision, local file identity) -> row; a row is built once per process.
_ROWS: dict[tuple, dict[str, Any]] = {}
# (setting, value) -> (root, files) of a local model; listed once per process.
_FILES: dict[tuple[str, str], tuple[Path, list[Path]]] = {}
# The short name of a local model whose own name is empty or private.
_NEUTRAL_NAME = "local-model"
# The directories that hold one directory per user (compared without case:
# Windows and macOS paths are case-insensitive).
_HOME_ROOTS = ("home", "users", "documents and settings")
# A Windows drive (``C:``) or a UNC share root, which is no part of the model's name.
_DRIVE = re.compile(r"[A-Za-z]:(?=/|$)|//[^/]+/[^/]+")


def _resolved_revision(repo_id: str, revision: str | None) -> str | None:
    """``revision`` as a commit: as given when it is one, else the cache's ref, else as given."""
    if revision and _COMMIT.fullmatch(revision):
        return revision
    ref = revision or "main"
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        refs = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "refs" / ref
        commit = refs.read_text(encoding="utf-8").strip()
    except Exception:  # noqa: BLE001 — no cache entry: keep the configured revision
        return revision
    return commit if _COMMIT.fullmatch(commit) else revision


def _private_names() -> set[str]:
    names: set[str] = set()
    try:
        names.update(p for p in Path.home().parts if p not in ("", "/"))
    except (KeyError, RuntimeError, OSError):  # no home directory: nothing to leave out
        pass
    try:
        names.add(getpass.getuser())
    except (KeyError, OSError):  # no user name: nothing to leave out
        pass
    return names


def _path_parts(path: Path) -> list[str]:
    """The names along ``path``, read the same on every system.

    A Windows path is split on either slash and its drive or share is left out,
    so ``C:\\Users\\<user>\\onnx`` reads as ``Users``, ``<user>``, ``onnx`` here too.
    """
    text = str(path.expanduser()).replace("\\", "/")
    drive = _DRIVE.match(text)
    if drive is not None:
        text = text[drive.end() :]
    return [p for p in text.split("/") if p not in ("", ".", "..")]


def _short_name(path: Path, rows: list[dict[str, Any]]) -> str:
    """``name@<12 hex>``: the model's own name and a hash over its files (``rows``)."""
    parts = _path_parts(path)
    at = len(parts) - 1
    if at >= 1 and parts[at].lower() == "onnx":
        at -= 1
    name = parts[at] if at >= 0 else ""
    # Any user's home directory (/home/<user>, /Users/<user>, C:\Users\<user>, /root)
    # names its user.
    a_home = (at == 1 and parts[0].lower() in _HOME_ROOTS) or (at == 0 and name == "root")
    if not name or a_home or name in _private_names() or name.startswith("~"):
        name = _NEUTRAL_NAME
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r["name"]):
        digest.update(f"{row['name']}\0{row['sha256'] or ''}\n".encode())
    return f"{name}@{digest.hexdigest()[:12]}"


def _local_files(env: str, value: str) -> tuple[Path, list[Path]]:
    """``(root, files)`` a local model of setting ``env`` is read from, listed once per process.

    An empty listing (the path is missing) is not kept, so a model that
    appears later in the run is listed then.
    """
    key = (env, value)
    with _LOCK:
        listed = _FILES.get(key)
    if listed is None:
        listed = _list_local_files(env, value)
        if listed[1]:
            with _LOCK:
                _FILES[key] = listed
    return listed[0], list(listed[1])


def _list_local_files(env: str, value: str) -> tuple[Path, list[Path]]:
    from bibr.utils.ml_runtime import local_onnx_bundle

    path = Path(value).expanduser()
    if env == "REF_GEOM_TAGGER_MODEL_ID":
        from bibr.extract.geom_tagger import tagger_bundle_files

        return path, list(tagger_bundle_files(path))
    if env == "REF_TEXT_SEG_MODEL_ID":
        from bibr.extract.text_segmenter import segmenter_bundle_files

        return path, list(segmenter_bundle_files(path))
    if path.is_file():
        return path.parent, [path]
    bundle = local_onnx_bundle(path)
    if bundle is not None and env not in ("WTPSPLIT_MODEL", "LAYOUT_MODEL_ID"):
        from bibr.utils.model_overrides import onnx_bundle_files

        return path, onnx_bundle_files(bundle)
    if path.is_dir():
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(path).parts)
        )
        return path, files
    return path, []


def _file_identity(files: list[Path]) -> tuple:
    out = []
    for f in files:
        try:
            st = f.stat()
            out.append((str(f), st.st_size, st.st_mtime_ns))
        except OSError:
            out.append((str(f), None, None))
    return tuple(out)


def _model_row(
    env: str,
    label: str,
    value: str,
    revision: str | None,
    override: bool,
    local: bool | None = None,
) -> dict[str, Any]:
    from bibr.utils.model_overrides import _sha256, is_local_model

    if not isinstance(value, str) or not value:
        raise TypeError(f"{env}: model value {value!r} is not a name")
    if local is None:
        local = is_local_model(value)
    row: dict[str, Any] = {"setting": env, "model": label}
    if not local:
        repo_id = value.partition(":")[0]
        key = (env, value, revision, None)
        with _LOCK:
            cached = _ROWS.get(key)
        if cached is None:
            cached = {
                **row,
                "id": value,
                "revision": _resolved_revision(repo_id, revision),
                "files": None,
                "override": override,
            }
            with _LOCK:
                _ROWS[key] = cached
        return dict(cached)
    root, files = _local_files(env, value)
    key = (env, value, revision, _file_identity(files))
    with _LOCK:
        cached = _ROWS.get(key)
    if cached is None:
        rows = []
        for f in files:
            try:
                name = f.relative_to(root).as_posix()
            except ValueError:
                name = f.name
            rows.append({"name": name, "sha256": _sha256(f) if f.is_file() else None})
        cached = {
            **row,
            "id": _short_name(Path(value), rows),
            "revision": None,
            "files": rows,
            "override": override,
        }
        with _LOCK:
            _ROWS[key] = cached
    return dict(cached)


def _layout_spec(settings: Any) -> tuple[str, str, str | None, str | None, bool]:
    """``(setting, value, revision, default value, default revision)`` of the layout model."""
    layout = settings.layout
    from bibr.config import LayoutOptions

    fields = LayoutOptions.model_fields
    runtime = str(getattr(getattr(settings, "ml", None), "runtime", "auto") or "auto")
    if runtime != "torch" and layout.onnx_model_id:
        value, revision = layout.onnx_model_id, layout.onnx_revision
        default = (fields["onnx_model_id"].default, fields["onnx_revision"].default)
        return "LAYOUT_ONNX_MODEL_ID", value, revision, *default
    value, revision = layout.model_id, layout.model_revision
    default = (fields["model_id"].default, fields["model_revision"].default)
    return "LAYOUT_MODEL_ID", value, revision, *default


def _rows(settings: Any, *, layout: bool, extraction: bool) -> list[Callable[[], dict[str, Any]]]:
    from bibr.utils.ml_runtime import torch_available
    from bibr.utils.model_overrides import (
        MODEL_SETTINGS,
        is_model_override,
        models_in_use,
        setting_value,
    )

    builders: list[Callable[[], dict[str, Any]]] = []
    if layout:
        env, value, revision, default_value, default_revision = _layout_spec(settings)
        override = (value, revision) != (default_value, default_revision)
        builders.append(
            lambda env=env, value=value, revision=revision, override=override: _model_row(
                env, "layout model", value, revision, override
            )
        )
    from bibr.config import GlobalSettings
    from bibr.segmenter_base import resolve_wtpsplit_model

    sat = resolve_wtpsplit_model(settings.WTPSPLIT_MODEL, revision=settings.WTPSPLIT_MODEL_REVISION)
    if sat.is_local:
        sat_value, sat_revision = sat.model_name, None
    else:
        prefix = f"{sat.hub_prefix}/" if sat.hub_prefix else ""
        sat_value, sat_revision = f"{prefix}{sat.model_name}", sat.revision
    sat_default = GlobalSettings.model_fields["WTPSPLIT_MODEL"].default
    sat_override = sat_default != settings.WTPSPLIT_MODEL or bool(settings.WTPSPLIT_MODEL_REVISION)
    builders.append(
        lambda: _model_row(
            "WTPSPLIT_MODEL",
            "sentence splitter",
            sat_value,
            sat_revision,
            sat_override,
            local=sat.is_local,
        )
    )
    used = models_in_use(settings) if extraction else set()
    if not torch_available():
        used.discard("NER_SEG_CKPT")
    for spec in MODEL_SETTINGS:
        if spec.env not in used:
            continue
        value, revision = setting_value(settings, spec.env)
        if value is None:
            continue
        override = is_model_override(settings, spec.env)
        builders.append(
            lambda spec=spec, value=value, revision=revision, override=override: _model_row(
                spec.env, spec.label, value, revision, override
            )
        )
    return builders


def export_models(
    settings: Any, *, layout: bool = True, extraction: bool = True
) -> list[dict[str, Any]] | None:
    """The ``extraction.settings.models`` rows for a run with these (effective) settings.

    ``layout`` is false for an input that is not laid out (DOCX, XML, HTML);
    ``extraction`` is false for a run that extracts nothing with a model
    (``--no-llm``), which lists only layout and the sentence splitter. ``None``
    when the rows cannot be built (logged; an export is never stopped by its
    provenance).
    """
    try:
        out = []
        for build in _rows(settings, layout=layout, extraction=extraction):
            try:
                out.append(build())
            except Exception as exc:  # noqa: BLE001 — one model's identity, not the export
                logger.warning("model provenance: a row could not be built (%r)", exc)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("model provenance could not be built (%r)", exc)
        return None


def reset_model_provenance() -> None:
    """Test hook: forget the cached rows and file listings."""
    with _LOCK:
        _ROWS.clear()
        _FILES.clear()


__all__ = ["export_models", "reset_model_provenance"]
