"""Hugging Face cache helpers for Windows first-run downloads and offline loads."""

from __future__ import annotations

import functools
import inspect
import logging
import os
import re
import sys
from typing import Any

logger = logging.getLogger(__name__)

#: A full commit hash names one immutable tree, so a file cached under it is current.
_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")

#: Whether Hub downloads draw progress bars named after the model they fetch.
#: Off for a library caller (and for a piped run, which draws no bars at all);
#: ``bibr chew`` turns it on once it has announced what it is about to download.
_name_download_bars = False

_WINDOWS_SYMLINK_PRIVILEGE_ERROR = 1314


def disable_hf_cache_symlinks_on_windows() -> None:
    """Force copy-based HF Hub cache entries on Windows.

    Some Windows installs lack the privilege required for cache symlinks. The
    Hub reads this value early, so patch both the environment and the already
    imported constants module before any first-run model download.
    """
    if sys.platform != "win32":
        return
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    try:
        from huggingface_hub import constants as hf_constants
    except ImportError:
        return
    hf_constants.HF_HUB_DISABLE_SYMLINKS = True


def is_windows_symlink_privilege_error(error: OSError) -> bool:
    return sys.platform == "win32" and (
        getattr(error, "winerror", None) == _WINDOWS_SYMLINK_PRIVILEGE_ERROR
        or f"[WinError {_WINDOWS_SYMLINK_PRIVILEGE_ERROR}]" in str(error)
    )


def hf_hub_download_no_symlink(**kwargs: Any) -> str:
    """Run ``hf_hub_download`` after disabling Windows cache symlinks."""
    disable_hf_cache_symlinks_on_windows()
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(**kwargs)
    except OSError as exc:
        if not is_windows_symlink_privilege_error(exc):
            raise
        disable_hf_cache_symlinks_on_windows()
        return hf_hub_download(**kwargs)


def name_download_bars(enabled: bool = True) -> None:
    """Make Hub downloads from here on draw progress bars named after their model.

    Only on a terminal: a run with stderr piped has its bars switched off
    (``bibr.local.cli._suppress_progress_bars_if_not_tty``) and keeps them off.
    """
    global _name_download_bars
    _name_download_bars = enabled and sys.stderr.isatty()


@functools.cache
def _accepts_tqdm_class(func: Any) -> bool:
    try:
        return "tqdm_class" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def named_bar_kwargs(func: Any, repo_id: str, filename: str | None = None) -> dict[str, Any]:
    """``tqdm_class`` kwargs that title *func*'s bar ``repo: file``, or ``{}``.

    Empty when naming is off or the installed ``huggingface_hub`` takes no
    ``tqdm_class`` (it arrived in 1.x; bibr supports older releases too), so
    every caller can splat the result into the download call.
    """
    if not _name_download_bars or not _accepts_tqdm_class(func):
        return {}
    try:
        from tqdm.auto import tqdm as base
    except ImportError:
        return {}
    title = repo_id if filename is None else f"{repo_id.rsplit('/', 1)[-1]}: {filename}"

    class _NamedBar(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["desc"] = title
            super().__init__(*args, **kwargs)

    return {"tqdm_class": _NamedBar}


def cached_at_pinned_revision(repo_id: str, filename: str, revision: str | None) -> str | None:
    """Path of ``filename`` when the cache holds it at the pinned commit *revision*, else ``None``.

    Never touches the network. Only a full commit hash qualifies: a branch or
    tag can move, so those still go to the Hub. Raises
    :class:`~huggingface_hub.errors.LocalEntryNotFoundError` when the cache
    records that the file does not exist at that commit (the Hub answered 404
    once, and a commit never changes), so a repo that ships no ``tokenizer.json``
    costs one request ever, not one per run.
    """
    if not revision or not _COMMIT_HASH.fullmatch(revision):
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        found = try_to_load_from_cache(repo_id=repo_id, filename=filename, revision=revision)
    except Exception as exc:  # noqa: BLE001 — a cache we cannot read is a cache miss
        logger.debug("HF cache lookup for %s/%s failed: %s", repo_id, filename, exc)
        return None
    if isinstance(found, str):
        return found
    if found is None:
        return None
    from huggingface_hub.errors import LocalEntryNotFoundError

    raise LocalEntryNotFoundError(
        f"{filename} does not exist in {repo_id} at {revision[:8]} (recorded by an earlier request)"
    )


def hf_download_or_cached(repo_id: str, filename: str, revision: str | None = None) -> str:
    """Return ``filename`` from ``repo_id``: the cache first at a pinned commit, else the Hub.

    A file already cached at a full commit hash is returned with no request, so
    a warm run starts with the network untouched; a file the cache records as
    absent at that commit raises ``LocalEntryNotFoundError`` the same way.
    Anything else is downloaded, and the Hub being unreachable (offline mode,
    DNS failure, 5xx, …) must not fail a load whose weights are already fully
    cached from a prior run. If nothing is cached, the original error surfaces.
    """
    cached = cached_at_pinned_revision(repo_id, filename, revision)
    if cached is not None:
        return cached
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download_no_symlink(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            **named_bar_kwargs(hf_hub_download, repo_id, filename),
        )
    except Exception:
        try:
            return hf_hub_download_no_symlink(
                repo_id=repo_id, filename=filename, revision=revision, local_files_only=True
            )
        except Exception as cache_exc:  # noqa: BLE001 — surface the original error below
            logger.debug("%s/%s not in the local HF cache: %s", repo_id, filename, cache_exc)
        raise


def hub_offline() -> bool:
    """Whether the Hub library is in offline mode (``HF_HUB_OFFLINE``): nothing can be downloaded."""
    try:
        from huggingface_hub import constants
    except ImportError:
        return False
    return bool(getattr(constants, "HF_HUB_OFFLINE", False))
