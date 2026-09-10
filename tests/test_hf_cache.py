"""Windows Hugging Face cache helpers."""

from __future__ import annotations

import os
import sys

import huggingface_hub
import huggingface_hub.constants as hf_constants


def test_disable_hf_cache_symlinks_on_windows_sets_env_and_loaded_constant(monkeypatch):
    from bibr.utils.hf_cache import disable_hf_cache_symlinks_on_windows

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising=False)
    monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_SYMLINKS", False)

    disable_hf_cache_symlinks_on_windows()

    assert os.environ["HF_HUB_DISABLE_SYMLINKS"] == "1"
    assert hf_constants.HF_HUB_DISABLE_SYMLINKS is True


def test_hf_hub_download_retries_windows_symlink_privilege_error(monkeypatch):
    from bibr.utils.hf_cache import hf_hub_download_no_symlink

    calls = []
    _ = huggingface_hub.hf_hub_download

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising=False)
    monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_SYMLINKS", False)

    def fake_hf_hub_download(**kwargs):
        calls.append(
            (
                kwargs,
                os.environ.get("HF_HUB_DISABLE_SYMLINKS"),
                hf_constants.HF_HUB_DISABLE_SYMLINKS,
            )
        )
        if len(calls) == 1:
            error = OSError("[WinError 1314] A required privilege is not held by the client")
            error.winerror = 1314
            raise error
        return "/cache/best.pt"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    assert (
        hf_hub_download_no_symlink(repo_id="org/repo", filename="best.pt", revision="abc")
        == "/cache/best.pt"
    )
    assert calls == [
        (
            {"repo_id": "org/repo", "filename": "best.pt", "revision": "abc"},
            "1",
            True,
        ),
        (
            {"repo_id": "org/repo", "filename": "best.pt", "revision": "abc"},
            "1",
            True,
        ),
    ]
