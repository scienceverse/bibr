"""Resolve NER checkpoint identifiers to a local file path.

Accepts either:
  - a local filesystem path (returned unchanged if it exists)
  - an HF Hub repo id, with optional filename: ``org/repo`` or
    ``org/repo:filename.pt``. When no filename is given, the resolver
    expects exactly one ``*.pt`` (or ``*.bin``/``*.safetensors``) file in the
    repo and picks it. Downloads via ``huggingface_hub.hf_hub_download``.

Offline-resilient: a fully-cached checkpoint resolves without any network
round-trip, and the Hub being unreachable falls back to the local cache rather
than crashing. Pin ``revision`` (a commit SHA) for reproducible loads and to let
``hf_hub_download`` short-circuit the etag check entirely.
"""

from __future__ import annotations

from pathlib import Path

_CKPT_SUFFIXES = (".pt", ".bin", ".safetensors")


def resolve_checkpoint(ckpt: str | Path, revision: str | None = None) -> str:
    """Return a local path to the checkpoint file.

    If ``ckpt`` already points at an existing local file, return it as a
    string. Otherwise treat it as ``repo_id[:filename]`` and download via the
    HF Hub at ``revision`` (defaulting to the Hub's default branch).
    """
    ckpt_str = str(ckpt)
    if Path(ckpt_str).exists():
        return ckpt_str

    if ":" in ckpt_str:
        repo_id, filename = ckpt_str.split(":", 1)
    else:
        repo_id, filename = ckpt_str, None

    if repo_id.count("/") != 1:
        raise ValueError(
            f"Checkpoint {ckpt_str!r} is not a local file and not a valid "
            "HF Hub identifier (expected 'org/repo' or 'org/repo:filename')."
        )

    if filename is None:
        filename = _discover_checkpoint_filename(repo_id, revision)

    return _download(repo_id, filename, revision)


def _pick_checkpoint(files) -> list[str]:
    return [f for f in files if f.endswith(_CKPT_SUFFIXES)]


def _discover_checkpoint_filename(repo_id: str, revision: str | None) -> str:
    """Find the single checkpoint filename in ``repo_id``.

    Prefers the local cache: when the snapshot is already present the filename
    is found without any network round-trip (the recurring cost the old resolver
    paid on every load, and which broke entirely when the Hub was unreachable).
    Falls back to the Hub only when the snapshot is not cached.
    """
    snap = _local_snapshot_dir(repo_id, revision)
    if snap is not None:
        cached = _pick_checkpoint(p.name for p in snap.iterdir() if p.is_file())
        if len(cached) == 1:
            return cached[0]

    from huggingface_hub import list_repo_files

    candidates = _pick_checkpoint(list_repo_files(repo_id, revision=revision))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one checkpoint file in {repo_id!r}, "
            f"found {candidates}. Specify one with 'repo:filename'."
        )
    return candidates[0]


def _local_snapshot_dir(repo_id: str, revision: str | None) -> Path | None:
    """Return the cached snapshot dir for ``repo_id``@``revision``, or None.

    Resolves a commit-SHA revision directly, a branch/tag name (or ``main`` when
    ``revision`` is None) through ``refs/``, and finally a lone cached snapshot.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    snap_root = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snap_root.is_dir():
        return None

    if revision:
        direct = snap_root / revision
        if direct.is_dir():
            return direct

    ref_file = snap_root.parent / "refs" / (revision or "main")
    if ref_file.is_file():
        resolved = snap_root / ref_file.read_text().strip()
        if resolved.is_dir():
            return resolved

    snaps = [p for p in snap_root.iterdir() if p.is_dir()]
    if len(snaps) == 1:
        return snaps[0]
    return None


def _download(repo_id: str, filename: str, revision: str | None) -> str:
    """Download ``filename`` from ``repo_id``, falling back to cache when offline."""
    from bibr.utils.hf_cache import hf_download_or_cached

    return hf_download_or_cached(repo_id, filename, revision)
