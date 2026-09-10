"""Hugging Face cache helpers for Windows first-run downloads and offline loads."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

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


def hf_download_or_cached(repo_id: str, filename: str, revision: str | None = None) -> str:
    """Download ``filename`` from ``repo_id``, falling back to the cache when offline.

    The Hub being unreachable (offline mode, DNS failure, 5xx, …) must not
    fail a load whose weights are already fully cached from a prior run. If
    nothing is cached, the original error surfaces.
    """
    try:
        return hf_hub_download_no_symlink(repo_id=repo_id, filename=filename, revision=revision)
    except Exception:
        try:
            return hf_hub_download_no_symlink(
                repo_id=repo_id, filename=filename, revision=revision, local_files_only=True
            )
        except Exception as cache_exc:  # noqa: BLE001 — surface the original error below
            logger.debug("%s/%s not in the local HF cache: %s", repo_id, filename, cache_exc)
        raise


def snapshot_download_no_symlink(*args: Any, **kwargs: Any) -> str:
    """Run ``snapshot_download`` after disabling Windows cache symlinks."""
    disable_hf_cache_symlinks_on_windows()
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(*args, **kwargs)
    except OSError as exc:
        if not is_windows_symlink_privilege_error(exc):
            raise
        disable_hf_cache_symlinks_on_windows()
        retry_kwargs = dict(kwargs)
        retry_kwargs.setdefault("max_workers", 1)
        return snapshot_download(*args, **retry_kwargs)
