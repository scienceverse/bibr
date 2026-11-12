"""Windows Hugging Face cache helpers."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import huggingface_hub
import huggingface_hub.constants as hf_constants
import pytest


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


# --- pinned-cache-first: a warm run makes no request -----------------------------

from bibr.utils import hf_cache  # noqa: E402

# The suite stubs ``cached_at_pinned_revision`` (tests/conftest.py) so unit tests see
# the downloads they patch; these tests need the real lookup, kept from import time.
_REAL_LOOKUP = hf_cache.cached_at_pinned_revision

_SHA = "a" * 40
_OTHER_SHA = "b" * 40


@pytest.fixture
def hub_cache(tmp_path, monkeypatch):
    """An empty Hugging Face cache the real lookup reads, with downloads forbidden."""
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(hf_cache, "cached_at_pinned_revision", _REAL_LOOKUP)
    requests: list[dict] = []

    def no_request(**kwargs):
        requests.append(kwargs)
        raise AssertionError(f"unexpected Hub request: {kwargs}")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_request)
    cache = type("Cache", (), {})()
    cache.root = tmp_path
    cache.requests = requests

    def put(repo_id: str, filename: str, sha: str = _SHA) -> str:
        path = tmp_path / f"models--{repo_id.replace('/', '--')}" / "snapshots" / sha / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        return str(path)

    def record_missing(repo_id: str, filename: str, sha: str = _SHA) -> None:
        path = tmp_path / f"models--{repo_id.replace('/', '--')}" / ".no_exist" / sha / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    cache.put = put
    cache.record_missing = record_missing
    return cache


def test_a_file_cached_at_a_pinned_commit_is_returned_without_a_request(hub_cache):
    cached = hub_cache.put("org/repo", "model.onnx")

    assert hf_cache.hf_download_or_cached("org/repo", "model.onnx", _SHA) == cached
    assert hub_cache.requests == []


def test_a_nested_filename_is_found_in_the_pinned_snapshot(hub_cache):
    cached = hub_cache.put("org/repo", "tokenizer/config.json")

    assert hf_cache.hf_download_or_cached("org/repo", "tokenizer/config.json", _SHA) == cached
    assert hub_cache.requests == []


@pytest.mark.parametrize("revision", [None, "main", "v1.0", "abc1234", "A" * 40, "a" * 39])
def test_only_a_full_commit_hash_is_trusted_from_the_cache(hub_cache, revision):
    """A branch, a tag or a short hash can move, so the Hub is still asked about those."""
    hub_cache.put("org/repo", "model.onnx", _SHA)
    hub_cache.put("org/repo", "model.onnx", "main")
    assert hf_cache.cached_at_pinned_revision("org/repo", "model.onnx", revision) is None


def test_an_unpinned_download_still_goes_to_the_hub(hub_cache, monkeypatch):
    hub_cache.put("org/repo", "model.onnx", "main")
    seen: list[dict] = []

    def fake_download(**kwargs):
        seen.append(kwargs)
        return "/downloaded"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert hf_cache.hf_download_or_cached("org/repo", "model.onnx", None) == "/downloaded"
    assert [call["revision"] for call in seen] == [None]


def test_a_pinned_file_missing_from_the_cache_is_downloaded(hub_cache, monkeypatch):
    hub_cache.put("org/repo", "config.json", _SHA)  # a different file of the same commit
    hub_cache.put("org/repo", "model.onnx", _OTHER_SHA)  # the same file of another commit
    seen: list[dict] = []

    def fake_download(**kwargs):
        seen.append(kwargs)
        return "/downloaded"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert hf_cache.hf_download_or_cached("org/repo", "model.onnx", _SHA) == "/downloaded"
    assert len(seen) == 1
    assert seen[0]["revision"] == _SHA


def test_a_file_the_cache_records_as_absent_raises_without_a_request(hub_cache):
    """A repo with no tokenizer.json at the pinned commit costs one request ever, not one a run."""
    from huggingface_hub.errors import EntryNotFoundError

    hub_cache.record_missing("org/repo", "tokenizer.json")

    with pytest.raises(EntryNotFoundError):
        hf_cache.hf_download_or_cached("org/repo", "tokenizer.json", _SHA)
    assert hub_cache.requests == []


def test_an_unreadable_cache_is_a_miss_not_an_error(hub_cache, monkeypatch):
    def broken(**_kwargs):
        raise PermissionError("cache is read-protected")

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", broken)

    assert hf_cache.cached_at_pinned_revision("org/repo", "model.onnx", _SHA) is None


def test_the_hub_being_down_still_falls_back_to_the_cache_for_an_unpinned_load(
    hub_cache, monkeypatch
):
    calls: list[dict] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        if not kwargs.get("local_files_only"):
            raise ConnectionError("hub unreachable")
        return "/from-cache"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert hf_cache.hf_download_or_cached("org/repo", "model.onnx", "main") == "/from-cache"
    assert [bool(call.get("local_files_only")) for call in calls] == [False, True]


def test_a_warm_segmenter_load_makes_no_hub_request(hub_cache):
    """The SaT model and its base tokenizer come from the cache; a repo that never had a
    ``tokenizer.json`` at the pinned commit is remembered as such, so it is not asked again."""
    from bibr.segmenter_base import (
        _SAT_BASE_TOKENIZER,
        _SAT_BASE_TOKENIZER_REVISION,
        base_tokenizer_dir,
        materialize_hub_snapshot,
    )

    onnx = hub_cache.put("org/sat", "model_optimized.onnx")
    hub_cache.put("org/sat", "config.json")
    hub_cache.record_missing("org/sat", "tokenizer.json")
    tokenizer = hub_cache.put(_SAT_BASE_TOKENIZER, "tokenizer.json", _SAT_BASE_TOKENIZER_REVISION)

    assert materialize_hub_snapshot("org/sat", _SHA) == (str(Path(onnx).parent), None)
    assert base_tokenizer_dir() == str(Path(tokenizer).parent)
    assert hub_cache.requests == []


def test_the_sat_base_tokenizer_is_pinned_to_a_full_commit():
    from bibr.segmenter_base import _SAT_BASE_TOKENIZER_REVISION

    assert len(_SAT_BASE_TOKENIZER_REVISION) == 40
    assert set(_SAT_BASE_TOKENIZER_REVISION) <= set("0123456789abcdef")


# --- named download bars ------------------------------------------------------


def _download_with_bars(repo_id, filename, *, tqdm_class=None):  # noqa: ARG001
    raise AssertionError("only the signature is read")


def _download_without_bars(repo_id, filename):  # noqa: ARG001
    raise AssertionError("only the signature is read")


@pytest.fixture
def bars_on(monkeypatch):
    monkeypatch.setattr(hf_cache, "_name_download_bars", False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    yield
    hf_cache.name_download_bars(False)


def test_download_bars_are_not_named_until_the_cli_asks(monkeypatch):
    monkeypatch.setattr(hf_cache, "_name_download_bars", False)
    assert hf_cache.named_bar_kwargs(_download_with_bars, "org/repo", "model.onnx") == {}


def test_download_bars_are_named_after_the_model_on_a_terminal(bars_on):
    hf_cache.name_download_bars(True)
    kwargs = hf_cache.named_bar_kwargs(_download_with_bars, "org/repo", "model.onnx")

    out = io.StringIO()
    bar = kwargs["tqdm_class"](total=1, file=out)
    bar.close()
    assert bar.desc == "repo: model.onnx"
    assert "repo: model.onnx" in out.getvalue()


def test_download_bars_stay_off_when_stderr_is_piped(monkeypatch):
    monkeypatch.setattr(hf_cache, "_name_download_bars", False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    hf_cache.name_download_bars(True)

    assert hf_cache.named_bar_kwargs(_download_with_bars, "org/repo", "model.onnx") == {}


def test_an_older_hub_without_tqdm_class_gets_no_bar_kwargs(bars_on):
    hf_cache.name_download_bars(True)
    assert hf_cache.named_bar_kwargs(_download_without_bars, "org/repo", "model.onnx") == {}
