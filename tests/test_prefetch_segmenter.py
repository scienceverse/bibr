"""Shared sentence-segmenter prefetch behavior and Docker wiring."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("model", "hub_prefix"),
    [
        ("sat-6l-sm", "segment-any-text"),
        ("scienceverse/bibr-sat-science-en", None),
    ],
)
def test_prefetch_uses_runtime_hub_resolution(monkeypatch, model, hub_prefix):
    from bibr.segmenter_base import resolve_wtpsplit_model
    from scripts.prefetch_segmenter import prefetch_segmenter

    captured = {}

    def fake_sat(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs

    monkeypatch.setattr("wtpsplit_lite.SaT", fake_sat)
    monkeypatch.setattr(
        "scripts.prefetch_segmenter.materialize_hub_snapshot",
        lambda repo_id, revision: (f"/pinned/{repo_id}@{revision}", None),
    )
    prefetch_segmenter(model, cache_dir=None)
    assert captured["kwargs"]["ort_providers"] == ["CPUExecutionProvider"]
    assert "from_pretrained_kwargs" not in captured["kwargs"]
    revision = resolve_wtpsplit_model(model).revision
    if revision is not None:
        # The default short name is pinned: SaT gets the audited snapshot directory.
        assert captured["name"] == f"/pinned/segment-any-text/{model}@{revision}"
        assert captured["kwargs"]["hub_prefix"] is None
        assert "tokenizer_name_or_path" not in captured["kwargs"]
    else:
        assert captured["name"] == model
        assert captured["kwargs"]["hub_prefix"] == hub_prefix
        if "/" in model:
            assert captured["kwargs"]["tokenizer_name_or_path"] == model
        else:
            assert "tokenizer_name_or_path" not in captured["kwargs"]


def test_prefetch_preserves_existing_local_bundle(monkeypatch, tmp_path):
    from scripts.prefetch_segmenter import prefetch_segmenter

    captured = {}

    def fake_sat(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs

    monkeypatch.setattr("wtpsplit_lite.SaT", fake_sat)

    prefetch_segmenter(str(tmp_path), cache_dir=None)

    assert captured["name"] == str(tmp_path)
    assert captured["kwargs"]["hub_prefix"] is None


def test_prefetch_uses_tokenizer_from_complete_local_bundle(monkeypatch, tmp_path):
    from scripts.prefetch_segmenter import prefetch_segmenter

    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    captured = {}

    def fake_sat(name, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr("wtpsplit_lite.SaT", fake_sat)

    prefetch_segmenter(str(tmp_path), cache_dir=None)

    assert captured["kwargs"]["tokenizer_name_or_path"] == str(tmp_path)


def test_prefetch_passes_explicit_cache_directory(monkeypatch, tmp_path):
    import httpx
    from huggingface_hub.errors import RemoteEntryNotFoundError

    from scripts.prefetch_segmenter import prefetch_segmenter

    captured = {}
    cache_dir = tmp_path / "cache"
    downloads = []

    def fake_sat(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs

    def fake_hf_hub_download(repo_id, filename, *, cache_dir, revision=None):  # noqa: ARG001
        downloads.append((repo_id, filename, cache_dir))
        if filename == "tokenizer.json" and repo_id == "segment-any-text/sat-6l-sm":
            response = httpx.Response(
                404, request=httpx.Request("GET", "https://huggingface.co/missing")
            )
            raise RemoteEntryNotFoundError("missing", response=response)
        target = Path(cache_dir) / repo_id.replace("/", "--") / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
        return str(target)

    monkeypatch.setattr("wtpsplit_lite.SaT", fake_sat)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    prefetch_segmenter("sat-6l-sm", cache_dir=cache_dir)

    assert cache_dir.is_dir()
    assert downloads == [
        ("segment-any-text/sat-6l-sm", "model_optimized.onnx", str(cache_dir)),
        ("segment-any-text/sat-6l-sm", "config.json", str(cache_dir)),
        ("segment-any-text/sat-6l-sm", "tokenizer.json", str(cache_dir)),
        ("FacebookAI/xlm-roberta-base", "tokenizer.json", str(cache_dir)),
    ]
    assert captured["name"] == str(cache_dir / "segment-any-text--sat-6l-sm")
    assert captured["kwargs"]["hub_prefix"] is None
    assert captured["kwargs"]["tokenizer_name_or_path"] == str(
        cache_dir / "FacebookAI--xlm-roberta-base"
    )
    assert "from_pretrained_kwargs" not in captured["kwargs"]


def test_serve_dockerfile_uses_shared_configurable_prefetch_script():
    dockerfile = Path("Dockerfile.serve").read_text(encoding="utf-8")

    assert "ARG WTPSPLIT_MODEL=sat-6l-sm" in dockerfile
    assert 'python scripts/prefetch_segmenter.py "$WTPSPLIT_MODEL"' in dockerfile
    assert "SaT('sat-6l-sm'" not in dockerfile


def test_serve_dockerfile_can_copy_local_bundle_from_build_context():
    dockerfile = Path("Dockerfile.serve").read_text(encoding="utf-8")

    assert "ARG WTPSPLIT_LOCAL_BUNDLE" in dockerfile
    assert "source=.,target=/build-context,ro" in dockerfile
    assert '"/build-context/$WTPSPLIT_LOCAL_BUNDLE" /app/segmenter_bundle' in dockerfile


def test_docker_context_includes_shared_prefetch_script():
    patterns = Path(".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "scripts/" not in patterns
    assert "scripts/*" in patterns
    assert "!scripts/prefetch_segmenter.py" in patterns
    assert "!**/MODEL_CARD.md" in patterns
