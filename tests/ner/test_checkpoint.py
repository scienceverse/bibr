"""Resolver tests for bibr.ner.checkpoint.resolve_checkpoint.

The resolver must (a) serve a fully-cached checkpoint without any network
round-trip, (b) survive the Hub being unreachable when the weights are cached,
and (c) thread a pinned ``revision`` through to the Hub calls. Hub functions are
monkeypatched so no test touches the network.
"""

from __future__ import annotations

import os
import sys

import huggingface_hub
import huggingface_hub.constants
import pytest

from bibr.ner import checkpoint as ckpt_mod
from bibr.ner.checkpoint import resolve_checkpoint


class _Offline(Exception):
    """Stand-in for any 'Hub unreachable' error (offline, DNS, 5xx)."""


def _make_cache(tmp_path, repo_id, sha, files, ref="main"):
    """Build a minimal HF-hub-style cache dir and return its root."""
    repo_dir = tmp_path / f"models--{repo_id.replace('/', '--')}"
    snap = repo_dir / "snapshots" / sha
    snap.mkdir(parents=True)
    for f in files:
        (snap / f).write_text("weights")
    if ref:
        refs = repo_dir / "refs"
        refs.mkdir(parents=True)
        (refs / ref).write_text(sha)
    return tmp_path


def test_local_path_returned_unchanged(tmp_path):
    f = tmp_path / "local.pt"
    f.write_text("weights")
    assert resolve_checkpoint(f) == str(f)


def test_invalid_repo_id_raises_valueerror():
    with pytest.raises(ValueError, match="HF Hub identifier"):
        resolve_checkpoint("not-a-repo")


def test_pinned_filename_passes_revision_and_skips_listing(monkeypatch):
    seen = {}

    def fake_download(*, repo_id, filename, revision, **kw):
        seen.update(repo_id=repo_id, filename=filename, revision=revision)
        return "/cache/best.pt"

    def fail_listing(*a, **k):  # must not be called when filename is pinned
        raise AssertionError("list_repo_files should not be called")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", fail_listing)

    out = resolve_checkpoint("org/repo:best.pt", revision="abc123")

    assert out == "/cache/best.pt"
    assert seen == {"repo_id": "org/repo", "filename": "best.pt", "revision": "abc123"}


def test_offline_download_falls_back_to_cache(monkeypatch):
    def fake_download(*, repo_id, filename, revision, local_files_only=False, **kw):
        if not local_files_only:
            raise _Offline("hub down")
        return "/cache/best.pt"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert resolve_checkpoint("org/repo:best.pt", revision="abc") == "/cache/best.pt"


def test_download_disables_hf_symlinks_on_windows(monkeypatch):
    calls = []
    _ = huggingface_hub.hf_hub_download

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_DISABLE_SYMLINKS", False)

    def fake_download(*, repo_id, filename, revision, **kw):
        calls.append(
            (
                repo_id,
                filename,
                revision,
                os.environ.get("HF_HUB_DISABLE_SYMLINKS"),
                huggingface_hub.constants.HF_HUB_DISABLE_SYMLINKS,
            )
        )
        return "/cache/best.pt"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert resolve_checkpoint("org/repo:best.pt", revision="abc") == "/cache/best.pt"
    assert calls == [("org/repo", "best.pt", "abc", "1", True)]


def test_offline_download_without_cache_reraises_original(monkeypatch):
    sentinel = _Offline("hub down")

    def fake_download(*, local_files_only=False, **kw):
        if not local_files_only:
            raise sentinel
        raise FileNotFoundError("not in cache")  # noqa: TRY003

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    with pytest.raises(_Offline) as exc:
        resolve_checkpoint("org/repo:best.pt")
    assert exc.value is sentinel


def test_filename_discovery_prefers_cache_without_network(monkeypatch, tmp_path):
    root = _make_cache(tmp_path, "org/repo", "deadbeef", ["best.pt", "config.json"])
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(root))

    def fail_listing(*a, **k):  # cached → never hit the Hub to list files
        raise AssertionError("list_repo_files should not be called when cached")

    captured = {}

    def fake_download(*, repo_id, filename, revision, **kw):
        captured["filename"] = filename
        return "/cache/best.pt"

    monkeypatch.setattr(huggingface_hub, "list_repo_files", fail_listing)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    out = resolve_checkpoint("org/repo", revision="deadbeef")

    assert out == "/cache/best.pt"
    assert captured["filename"] == "best.pt"


def test_filename_discovery_resolves_revision_via_refs(monkeypatch, tmp_path):
    # revision is a branch name ("main"), not a SHA → resolve through refs/main.
    root = _make_cache(tmp_path, "org/repo", "sha999", ["best.pt"], ref="main")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(root))
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *, filename, **kw: f"/cache/{filename}"
    )

    assert resolve_checkpoint("org/repo", revision="main") == "/cache/best.pt"


def test_filename_discovery_falls_back_to_hub_when_uncached(monkeypatch, tmp_path):
    # Empty cache → must consult the Hub to discover the single checkpoint file.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(
        huggingface_hub, "list_repo_files", lambda *a, **k: ["config.json", "best.pt"]
    )
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *, filename, **kw: f"/cache/{filename}"
    )

    assert resolve_checkpoint("org/repo") == "/cache/best.pt"


def test_multiple_checkpoints_in_uncached_repo_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda *a, **k: ["a.pt", "b.pt"])

    with pytest.raises(ValueError, match="exactly one checkpoint"):
        resolve_checkpoint("org/repo")


def test_revision_threaded_to_listing_when_uncached(monkeypatch, tmp_path):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    seen = {}

    def fake_listing(repo_id, revision=None, **k):
        seen["revision"] = revision
        return ["best.pt"]

    monkeypatch.setattr(huggingface_hub, "list_repo_files", fake_listing)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *, filename, **kw: "/c/best.pt")

    resolve_checkpoint("org/repo", revision="v4-sha")
    assert seen["revision"] == "v4-sha"


def test_resolve_checkpoint_accepts_no_revision(monkeypatch):
    # Back-compat: revision is optional and defaults to None (→ Hub default).
    seen = {}
    monkeypatch.setattr(
        ckpt_mod, "_discover_checkpoint_filename", lambda repo_id, revision: "best.pt"
    )

    def fake_download(*, revision, **kw):
        seen["revision"] = revision
        return "/cache/best.pt"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert resolve_checkpoint("org/repo") == "/cache/best.pt"
    assert seen["revision"] is None
