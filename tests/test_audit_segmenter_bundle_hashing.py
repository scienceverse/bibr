"""A sealed sentence-segmenter bundle hashes each of its files once per resolve."""

import json
from collections import Counter

import pytest

import bibr.segmenter_base as segmenter_base
from tests.test_segmenter_base import _write_sealed_bundle


def test_sealed_bundle_hashes_every_file_once(tmp_path, monkeypatch):
    _write_sealed_bundle(tmp_path)
    hashed = Counter()
    real = segmenter_base._sha256

    def counting(path):
        hashed[path.relative_to(tmp_path).as_posix()] += 1
        return real(path)

    monkeypatch.setattr(segmenter_base, "_sha256", counting)

    segmenter_base.resolve_wtpsplit_model(str(tmp_path))

    assert hashed
    assert set(hashed.values()) == {1}


def test_model_revision_must_still_name_the_optimized_graph(tmp_path):
    _write_sealed_bundle(tmp_path)
    manifest_path = tmp_path / "segmenter_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_revision"] = manifest["checksums"]["model.onnx"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="model_revision does not match"):
        segmenter_base.resolve_wtpsplit_model(str(tmp_path))


def _bundle_resolves(monkeypatch) -> list[str]:
    resolved: list[str] = []
    real = segmenter_base.resolve_wtpsplit_model

    def counting(model_name, revision=None):
        resolved.append(model_name)
        return real(model_name, revision=revision)

    monkeypatch.setattr(segmenter_base, "resolve_wtpsplit_model", counting)
    return resolved


def test_the_export_resolves_a_sealed_bundle_once_per_process(tmp_path, monkeypatch):
    """Every export lists the sentence splitter, and resolving a sealed bundle
    hashes all of it (about a second of the default model's two graphs)."""
    from bibr.config import snapshot_settings
    from bibr.utils.model_provenance import export_models, reset_model_provenance

    _write_sealed_bundle(tmp_path)
    settings = snapshot_settings()
    settings.WTPSPLIT_MODEL = str(tmp_path)
    resolved = _bundle_resolves(monkeypatch)
    reset_model_provenance()
    try:
        first = export_models(settings, layout=False, extraction=False)
        for _paper in range(4):
            assert export_models(settings, layout=False, extraction=False) == first
        assert resolved == [str(tmp_path)]
        assert first[0]["files"]
    finally:
        reset_model_provenance()


def test_a_changed_bundle_is_resolved_again(tmp_path, monkeypatch):
    import os

    from bibr.config import snapshot_settings
    from bibr.utils.model_provenance import export_models, reset_model_provenance

    _write_sealed_bundle(tmp_path)
    settings = snapshot_settings()
    settings.WTPSPLIT_MODEL = str(tmp_path)
    resolved = _bundle_resolves(monkeypatch)
    reset_model_provenance()
    try:
        assert export_models(settings, layout=False, extraction=False)
        config = tmp_path / "config.json"
        config.write_text(config.read_text(encoding="utf-8") + " ", encoding="utf-8")
        stat = config.stat()
        os.utime(config, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        # The tampered bundle no longer validates, so its row is not built.
        assert export_models(settings, layout=False, extraction=False) is None
        assert len(resolved) == 2
    finally:
        reset_model_provenance()
