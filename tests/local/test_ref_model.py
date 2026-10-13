import hashlib
import io
import json
import zipfile

import pytest

from bibr.exceptions import BibrError
from bibr.local import ref_model


@pytest.fixture
def bundle(monkeypatch):
    files = dict.fromkeys(ref_model._FILES - {"manifest.json"}, b"fixture")
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    manifest = json.dumps(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "base_revision": ref_model.BASE_REVISION,
            "adapter": "balanced-r32-v1",
            "schema": "ref-llm-sft@1",
            "files": hashes,
        }
    ).encode()
    files["manifest.json"] = manifest
    monkeypatch.setattr(ref_model, "MANIFEST_SHA256", hashlib.sha256(manifest).hexdigest())
    monkeypatch.setattr(ref_model, "ADAPTER_SHA256", hashes["adapter_model.safetensors"])
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    data = stream.getvalue()
    monkeypatch.setattr(ref_model, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(ref_model.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(data))
    return data


def test_download_verifies_existing_install_without_network(bundle, tmp_path, monkeypatch):
    destination = tmp_path / "adapter"
    assert ref_model.download(destination) == destination
    monkeypatch.setattr(
        ref_model.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("network")
    )
    assert ref_model.download(destination) == destination
    (destination / "tokenizer.json").write_text("tampered")
    with pytest.raises(BibrError, match="checksum mismatch"):
        ref_model.download(destination)
    assert (destination / "tokenizer.json").read_text() == "tampered"


def test_untrusted_manifest_cannot_certify_modified_tokenizer(bundle, tmp_path):
    destination = ref_model.download(tmp_path / "adapter")
    tokenizer = destination / "tokenizer.json"
    tokenizer.write_bytes(b"different")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["tokenizer.json"] = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(BibrError, match="manifest checksum"):
        ref_model.download(destination)


def test_bad_download_is_not_installed(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(ref_model.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"bad"))
    with pytest.raises(BibrError, match="download checksum"):
        ref_model.download(tmp_path / "adapter")
    assert not (tmp_path / "adapter").exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("extra", ["../escape", "/absolute", "nested/adapter", "manifest.json"])
def test_archive_paths_and_duplicate_members_are_refused(bundle, tmp_path, extra):
    archive = tmp_path / "model.zip"
    archive.write_bytes(bundle)
    with zipfile.ZipFile(archive, "a") as stream:
        stream.writestr(extra, b"bad")
    output = tmp_path / "adapter"
    output.mkdir()
    with pytest.raises(ValueError, match="archive contents"):
        ref_model._unpack(archive, output)
    assert list(output.iterdir()) == []


def test_existing_unrelated_directory_is_preserved(tmp_path):
    destination = tmp_path / "adapter"
    destination.mkdir()
    (destination / "keep").write_text("keep")
    with pytest.raises(BibrError, match="unexpected or missing"):
        ref_model.download(destination)
    assert (destination / "keep").read_text() == "keep"


def test_download_limit_leaves_no_partial_install(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(ref_model, "_MAX_BYTES", len(bundle) - 1)
    with pytest.raises(BibrError, match="size limit"):
        ref_model.download(tmp_path / "adapter")
    assert list(tmp_path.iterdir()) == []


def test_ref_model_cli_requires_destination():
    from bibr.local.cli import _build_parser

    args = _build_parser().parse_args(["ref-model", "download", "--output", "models/reference"])
    assert args.output == "models/reference"
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["ref-model", "download"])
