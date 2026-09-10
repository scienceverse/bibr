"""The ML_RUNTIME selection rule and ONNX bundle discovery (no models, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bibr.exceptions import ConfigurationError
from bibr.utils import ml_runtime
from bibr.utils.ml_runtime import find_onnx_bundle, read_onnx_manifest, resolve_runtime


def _settings(mode: str):
    return SimpleNamespace(ml=SimpleNamespace(runtime=mode))


def _bundle(tmp_path: Path, *, manifest: dict | None = None) -> Path:
    bundle = tmp_path / "onnx"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"\x08\x07")
    manifest = {"schema_version": 1, "model": "x"} if manifest is None else manifest
    (bundle / "bibr_onnx.json").write_text(json.dumps(manifest))
    return bundle


# -- selection rule ----------------------------------------------------------


def test_auto_prefers_onnx_bundle(monkeypatch, tmp_path):
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(ml_runtime, "torch_available", lambda: True)
    runtime, path = resolve_runtime(
        "m", settings=_settings("auto"), bundle=lambda: bundle, bundle_hint="h"
    )
    assert (runtime, path) == ("onnx", bundle)


def test_auto_falls_back_to_torch_without_bundle(monkeypatch):
    monkeypatch.setattr(ml_runtime, "torch_available", lambda: True)
    runtime, path = resolve_runtime(
        "m", settings=_settings("auto"), bundle=lambda: None, bundle_hint="h"
    )
    assert (runtime, path) == ("torch", None)


def test_auto_without_either_names_extra_and_setting(monkeypatch):
    monkeypatch.setattr(ml_runtime, "torch_available", lambda: False)
    with pytest.raises(ConfigurationError) as exc:
        resolve_runtime(
            "section classifier",
            settings=_settings("auto"),
            bundle=lambda: None,
            bundle_hint="point ML_SECTION_CLASSIFIER_MODEL_ID at a bundle",
        )
    text = str(exc.value)
    assert "section classifier" in text
    assert "pip install 'bibr[torch]'" in text
    assert "ML_SECTION_CLASSIFIER_MODEL_ID" in text


def test_onnx_mode_requires_bundle(monkeypatch):
    monkeypatch.setattr(ml_runtime, "torch_available", lambda: True)
    with pytest.raises(ConfigurationError, match="ML_RUNTIME=onnx"):
        resolve_runtime("m", settings=_settings("onnx"), bundle=lambda: None, bundle_hint="h")


def test_torch_mode_never_probes_bundle(monkeypatch):
    monkeypatch.setattr(ml_runtime, "torch_available", lambda: True)
    calls = []

    def probe():
        calls.append(1)
        return None

    assert resolve_runtime("m", settings=_settings("torch"), bundle=probe, bundle_hint="h") == (
        "torch",
        None,
    )
    assert calls == []


def test_torch_mode_without_torch_fails(monkeypatch):
    monkeypatch.setattr(ml_runtime, "torch_available", lambda: False)
    with pytest.raises(ConfigurationError, match="ML_RUNTIME=torch"):
        resolve_runtime("m", settings=_settings("torch"), bundle=lambda: None, bundle_hint="h")


def test_default_ml_runtime_is_auto(monkeypatch):
    monkeypatch.delenv("ML_RUNTIME", raising=False)
    from bibr.config import GlobalSettings

    assert GlobalSettings().ml.runtime == "auto"


# -- bundle discovery --------------------------------------------------------


def test_find_bundle_in_local_directory(tmp_path):
    bundle = _bundle(tmp_path)
    assert find_onnx_bundle(str(tmp_path)) == bundle
    assert find_onnx_bundle(str(bundle)) == bundle  # the onnx/ dir itself


def test_find_bundle_next_to_local_checkpoint_file(tmp_path):
    bundle = _bundle(tmp_path)
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"")
    assert find_onnx_bundle(str(ckpt)) == bundle


def test_find_bundle_requires_manifest_and_model(tmp_path):
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"")
    assert find_onnx_bundle(str(tmp_path)) is None  # no manifest
    assert find_onnx_bundle(None) is None
    assert find_onnx_bundle("not-a-repo") is None


def test_find_bundle_on_hub_downloads_manifest_then_files(monkeypatch, tmp_path):
    bundle = _bundle(
        tmp_path, manifest={"schema_version": 1, "files": ["model.onnx", "tokenizer.json"]}
    )
    (bundle / "tokenizer.json").write_text("{}")
    fetched = []

    def fake_download(repo_id, filename, revision=None):
        fetched.append((repo_id, filename, revision))
        return str(bundle / Path(filename).name)

    monkeypatch.setattr("bibr.utils.hf_cache.hf_download_or_cached", fake_download)
    assert find_onnx_bundle("org/repo", "abc123") == bundle
    assert fetched == [
        ("org/repo", "onnx/bibr_onnx.json", "abc123"),
        ("org/repo", "onnx/model.onnx", "abc123"),
        ("org/repo", "onnx/tokenizer.json", "abc123"),
    ]


def test_find_bundle_on_hub_missing_returns_none(monkeypatch):
    def fake_download(repo_id, filename, revision=None):
        raise FileNotFoundError(f"{repo_id}/{filename} not found")

    monkeypatch.setattr("bibr.utils.hf_cache.hf_download_or_cached", fake_download)
    assert find_onnx_bundle("org/repo:best.pt", "main") is None


# -- manifest --------------------------------------------------------------


def test_read_manifest_rejects_wrong_schema(tmp_path):
    bundle = _bundle(tmp_path, manifest={"schema_version": 99})
    with pytest.raises(ConfigurationError, match="schema_version"):
        read_onnx_manifest(bundle)


def test_read_manifest_requires_model_file(tmp_path):
    bundle = _bundle(tmp_path)
    (bundle / "model.onnx").unlink()
    with pytest.raises(ConfigurationError, match="missing model.onnx"):
        read_onnx_manifest(bundle)
