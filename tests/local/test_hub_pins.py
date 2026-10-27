"""Pinned registry builds that bibr downloads itself (``bibr.local.hub_pins``).

``llama-server -hf`` and the MLX servers download a repository's ``main``, so
the 4B student's GGUF and MLX builds are fetched at their pinned commit into
the Hugging Face cache and launched from there. The Hub is mocked throughout;
the cache tests use a real cache layout under ``tmp_path``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from bibr.local import hub_pins

# Captured at import, before tests/conftest.py stubs them for every test.
_REAL_CACHED_FILE = hub_pins._cached_file
_REAL_DOWNLOAD_FILES = hub_pins._download_files

GGUF_REPO = "scienceverse/bibr-extract-qwen35-4b-GGUF"
GGUF_REVISION = "0aa228e8c908512ad5e5e13fa55e7e7d3180c7d3"
Q8_0 = f"{GGUF_REPO}:Q8_0"
Q4_K_M = f"{GGUF_REPO}:Q4_K_M"
MLX_REPO = "scienceverse/bibr-extract-qwen35-4b-mlx-8bit"
MLX_REVISION = "432b8cd80ea920919ef26c48401473c6d8c8a3e1"
MLX_FILES = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "model.safetensors",
)


@pytest.fixture
def real_hub_pins(monkeypatch):
    """Undo the suite-wide stub: the real cache reads and downloads."""
    monkeypatch.setattr(hub_pins, "_cached_file", _REAL_CACHED_FILE)
    monkeypatch.setattr(hub_pins, "_download_files", _REAL_DOWNLOAD_FILES)


@pytest.fixture
def hub_download(monkeypatch, real_hub_pins, tmp_path):
    """Mock ``huggingface_hub.hf_hub_download``; record each call's kwargs."""
    calls: list[dict] = []

    def fake(**kwargs):
        calls.append(kwargs)
        path = tmp_path / "snapshots" / kwargs["revision"] / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
    return calls


def _cache_file(cache: Path, repo_id: str, revision: str, filename: str) -> Path:
    """Write *filename* into a Hugging Face cache layout at *revision*."""
    path = cache / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"cached")
    return path


@pytest.mark.parametrize(
    "model",
    [
        "org/model:Q4",
        "ggml-org/GLM-OCR-GGUF:Q8_0",
        # vLLM takes --revision itself.
        "scienceverse/bibr-extract-qwen35-4b",
        "scienceverse/bibr-extract-qwen35-4b-W4A16",
        # The 2B's GGUF and MLX builds are not pinned.
        "scienceverse/bibr-extract-qwen35-2b-GGUF:Q8_0",
        "scienceverse/bibr-extract-qwen35-2b-mlx-8bit",
        # A repository without its quant is not a registry build.
        GGUF_REPO,
        "/models/bibr-extract-qwen35-4b-Q8_0.gguf",
    ],
)
def test_ids_that_are_not_pinned_builds_are_left_to_the_runtime(model, monkeypatch):
    monkeypatch.setattr(
        hub_pins, "_download_files", lambda *a: pytest.fail("nothing should be downloaded")
    )
    assert hub_pins.fetch_pinned_build(model) is None
    assert hub_pins.pinned_build_cached(model) is None


@pytest.mark.parametrize(
    ("model", "filename"),
    [(Q8_0, "bibr-extract-qwen35-4b-Q8_0.gguf"), (Q4_K_M, "bibr-extract-qwen35-4b-Q4_K_M.gguf")],
)
def test_a_gguf_build_is_downloaded_at_its_pinned_commit(model, filename, hub_download):
    path = hub_pins.fetch_pinned_build(model)

    assert hub_download == [
        {"repo_id": GGUF_REPO, "filename": filename, "revision": GGUF_REVISION, "cache_dir": None}
    ]
    assert Path(path).name == filename
    assert Path(path).parent.name == GGUF_REVISION


def test_an_mlx_build_returns_the_snapshot_directory_of_its_files(hub_download):
    path = hub_pins.fetch_pinned_build(MLX_REPO, cache_dir="/Volumes/hf/hub")

    assert [call["filename"] for call in hub_download] == list(MLX_FILES)
    assert {call["repo_id"] for call in hub_download} == {MLX_REPO}
    assert {call["revision"] for call in hub_download} == {MLX_REVISION}
    assert {call["cache_dir"] for call in hub_download} == {"/Volumes/hf/hub"}
    assert Path(path).name == MLX_REVISION
    assert sorted(p.name for p in Path(path).iterdir()) == sorted(MLX_FILES)


def test_a_cached_build_is_served_from_the_cache_without_the_hub(
    real_hub_pins, monkeypatch, tmp_path, caplog
):
    """A commit-pinned file already in ``snapshots/<commit>/`` needs no network."""
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    cached = _cache_file(tmp_path, GGUF_REPO, GGUF_REVISION, "bibr-extract-qwen35-4b-Q4_K_M.gguf")

    assert hub_pins.pinned_build_cached(Q4_K_M, cache_dir=str(tmp_path)) is True
    with caplog.at_level("INFO", logger="bibr.local.hub_pins"):
        path = hub_pins.fetch_pinned_build(Q4_K_M, cache_dir=str(tmp_path))

    assert Path(path) == cached
    assert not any("Downloading" in r.getMessage() for r in caplog.records)


def test_main_in_the_cache_does_not_count_as_the_pinned_build(real_hub_pins, monkeypatch, tmp_path):
    """The previous build on ``main`` is another commit: offline, the run stops with the fix."""
    import huggingface_hub.constants

    from bibr.exceptions import UpstreamServiceError

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    main_commit = "8f5cb6f8" + "0" * 32
    _cache_file(tmp_path, GGUF_REPO, main_commit, "bibr-extract-qwen35-4b-Q4_K_M.gguf")
    refs = tmp_path / f"models--{GGUF_REPO.replace('/', '--')}" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(main_commit)

    assert hub_pins.pinned_build_cached(Q4_K_M, cache_dir=str(tmp_path)) is False
    with pytest.raises(UpstreamServiceError) as excinfo:
        hub_pins.fetch_pinned_build(Q4_K_M, cache_dir=str(tmp_path))

    message = str(excinfo.value)
    assert "pinned revision 0aa228e8" in message
    assert "not in the Hugging Face cache" in message
    assert (
        f"`hf download {GGUF_REPO} bibr-extract-qwen35-4b-Q4_K_M.gguf "
        f"--revision {GGUF_REVISION} --cache-dir {tmp_path}`" in message
    )


def test_an_uncached_mlx_build_offline_names_the_whole_revision(real_hub_pins, monkeypatch):
    import huggingface_hub.constants

    from bibr.exceptions import UpstreamServiceError

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", "/nonexistent-hf-cache-e30")

    with pytest.raises(UpstreamServiceError) as excinfo:
        hub_pins.fetch_pinned_build(MLX_REPO)

    assert f"`hf download {MLX_REPO} --revision {MLX_REVISION}`" in str(excinfo.value)


def test_a_failed_download_names_its_cause(monkeypatch, real_hub_pins):
    from bibr.exceptions import UpstreamServiceError

    def broken(**_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", broken)

    with pytest.raises(UpstreamServiceError) as excinfo:
        hub_pins.fetch_pinned_build(Q8_0)

    message = str(excinfo.value)
    assert "the download failed (OSError: No space left on device)" in message
    assert isinstance(excinfo.value.original_error, OSError)


def test_a_stopped_startup_stops_waiting_for_the_download(monkeypatch):
    from bibr.local.http_runtime import StartupCancelled

    started = threading.Event()
    release = threading.Event()

    def slow_download(*_args):
        started.set()
        release.wait(30)
        return ["/never/used.gguf"]

    monkeypatch.setattr(hub_pins, "_download_files", slow_download)
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()

    begun = time.monotonic()
    try:
        with pytest.raises(StartupCancelled):
            hub_pins.fetch_pinned_build(Q8_0, stop_event=stop, server_label="llama.cpp")
        assert started.is_set()
        assert time.monotonic() - begun < 3
    finally:
        release.set()


def test_a_download_error_surfaces_through_the_stoppable_wait(monkeypatch):
    from bibr.exceptions import UpstreamServiceError

    def broken(*_args):
        raise ConnectionError("reset by peer")

    monkeypatch.setattr(hub_pins, "_download_files", broken)

    with pytest.raises(UpstreamServiceError, match="reset by peer"):
        hub_pins.fetch_pinned_build(Q8_0, stop_event=threading.Event())


def test_the_stoppable_wait_returns_the_download(monkeypatch):
    monkeypatch.setattr(hub_pins, "_download_files", lambda *a: ["/hf/snap/rev/config.json"])

    assert hub_pins.fetch_pinned_build(MLX_REPO, stop_event=threading.Event()) == "/hf/snap/rev"


def test_local_gguf_file(tmp_path):
    gguf = tmp_path / "bibr-extract-qwen35-4b-Q8_0.gguf"
    gguf.write_bytes(b"GGUF")
    (tmp_path / "dir.gguf").mkdir()

    assert hub_pins.local_gguf_file(str(gguf)) == str(gguf)
    assert hub_pins.local_gguf_file(str(tmp_path / "missing.gguf")) is None
    assert hub_pins.local_gguf_file(str(tmp_path / "dir.gguf")) is None
    assert hub_pins.local_gguf_file(Q8_0) is None
