"""Pinned Hub builds for the managed runtimes that cannot take a revision.

vLLM takes ``--revision``, so its registry variants pin a commit in their
server args. ``llama-server -hf <repo>:<quant>`` and the MLX servers
(``vllm-mlx --model <repo>``, ``rapid-mlx serve <repo>``) download whatever a
repository's ``main`` holds, so a registry build pinned with ``revision`` and
``hub_files`` (``bibr.local.llm_models.LocalLLMVariant``) is downloaded here,
at that commit, into the Hugging Face cache, and the runtime gets the local
path. A cached copy is served with no network round-trip (``hf_hub_download``
returns a file already in ``snapshots/<commit>/`` without asking the Hub), so
offline runs keep working once the build has been downloaded.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from bibr.exceptions import UpstreamServiceError
from bibr.local.http_runtime import check_startup_stop

logger = logging.getLogger(__name__)

#: How often a download that the owner may stop checks its stop event, seconds.
_STOP_POLL_S = 0.5


def _repo_id(model: str) -> str:
    """The repository of a registry id (``repo`` or ``repo:QUANT``)."""
    return model.split(":", 1)[0]


def _cached_file(repo_id: str, filename: str, revision: str, cache_dir: str | None) -> str | None:
    """The cached path of *filename* at *revision*, or None (no network)."""
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir, revision=revision)
    return found if isinstance(found, str) else None


def pinned_build_cached(model: str, *, cache_dir: str | None = None) -> bool | None:
    """Whether every pinned file of registry build *model* is in the HF cache.

    None when *model* is not a build bibr downloads itself
    (``pinned_download_variant``). Reads the cache only.
    """
    from bibr.local.llm_models import pinned_download_variant

    variant = pinned_download_variant(model)
    if variant is None or variant.revision is None:
        return None
    repo_id = _repo_id(model)
    return all(
        _cached_file(repo_id, name, variant.revision, cache_dir) is not None
        for name in variant.hub_files
    )


def _download_files(
    repo_id: str, files: tuple[str, ...], revision: str, cache_dir: str | None
) -> list[str]:
    """Download *files* of *repo_id* at *revision* (cached ones are returned as they are)."""
    from bibr.utils.hf_cache import hf_hub_download_no_symlink

    return [
        hf_hub_download_no_symlink(
            repo_id=repo_id, filename=name, revision=revision, cache_dir=cache_dir
        )
        for name in files
    ]


def _run_stoppable(fn, stop_event: threading.Event | None, server_label: str):
    """Run *fn* and return its result; stop waiting once *stop_event* is set.

    The download runs in a daemon thread so a stopped startup (Ctrl-C, a
    cancelled task) does not wait for several GB to finish; the abandoned
    thread leaves an ``.incomplete`` file that the next download resumes.
    """
    if stop_event is None:
        return fn()
    outcome: dict[str, object] = {}

    def _target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            outcome["error"] = exc

    thread = threading.Thread(target=_target, name=f"bibr-hub-download-{server_label}", daemon=True)
    thread.start()
    while thread.is_alive():
        check_startup_stop(stop_event, server_label)
        thread.join(_STOP_POLL_S)
    check_startup_stop(stop_event, server_label)
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome["value"]


def fetch_pinned_build(
    model: str,
    *,
    cache_dir: str | None = None,
    stop_event: threading.Event | None = None,
    server_label: str = "local inference",
) -> str | None:
    """The local path of registry build *model* at its pinned commit, downloading it if needed.

    Returns None when *model* is not a build bibr downloads itself (any id the
    runtime should get unchanged). For a llama.cpp build the path is the GGUF
    file; for an MLX build, the snapshot directory holding its files.
    ``cache_dir`` None is the Hugging Face default (``HF_HUB_CACHE`` /
    ``HF_HOME``). Raises ``UpstreamServiceError`` when a file can be neither
    found in the cache nor downloaded, and ``StartupCancelled`` once
    *stop_event* is set.
    """
    from bibr.local.llm_models import pinned_download_variant

    variant = pinned_download_variant(model)
    if variant is None or variant.revision is None:
        return None
    revision = variant.revision
    repo_id = _repo_id(model)
    files = variant.hub_files
    missing = [name for name in files if _cached_file(repo_id, name, revision, cache_dir) is None]
    if missing:
        logger.info(
            "Downloading %s at its pinned revision %s (~%.1f GB) into the Hugging Face cache%s",
            model,
            revision[:8],
            variant.approx_download_gb,
            f" ({cache_dir})" if cache_dir else "",
        )

    try:
        paths = _run_stoppable(
            lambda: _download_files(repo_id, files, revision, cache_dir), stop_event, server_label
        )
    except Exception as exc:
        from bibr.local.http_runtime import StartupCancelled

        if isinstance(exc, StartupCancelled):
            raise
        raise UpstreamServiceError(
            "local inference",
            _fetch_failure_message(model, repo_id, revision, files, cache_dir, exc),
            original_error=exc,
        ) from exc
    if variant.runtime == "llama-cpp":
        return paths[0]
    return str(Path(paths[0]).parent)


def _fetch_failure_message(
    model: str,
    repo_id: str,
    revision: str,
    files: tuple[str, ...],
    cache_dir: str | None,
    exc: BaseException,
) -> str:
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:  # pragma: no cover - huggingface_hub < 0.23
        offline = False
    else:
        offline = isinstance(exc, LocalEntryNotFoundError)
    # One GGUF: name it. An MLX build: the whole revision (its README too).
    parts = ["hf", "download", repo_id, *(files if len(files) == 1 else ()), "--revision", revision]
    if cache_dir:
        parts += ["--cache-dir", cache_dir]
    command = " ".join(parts)
    cause = (
        "it is not in the Hugging Face cache and the Hub could not be reached "
        "(no network, or HF_HUB_OFFLINE is set)"
        if offline
        else f"the download failed ({type(exc).__name__}: {exc})"
    )
    return (
        f"Could not get {model} at its pinned revision {revision[:8]}: {cause}. bibr downloads "
        "this build itself because its runtime would fetch the repository's main branch, which "
        "holds a different build. Connect to the Hub once, or download it yourself with "
        f"`{command}`."
    )


def local_gguf_file(model: str) -> str | None:
    """*model* when it is the path of an existing ``.gguf`` file, else None.

    ``llama-server`` loads a local file with ``-m``; ``-hf`` takes only Hub ids.
    """
    path = Path(model).expanduser()
    return str(path) if path.suffix.lower() == ".gguf" and path.is_file() else None
