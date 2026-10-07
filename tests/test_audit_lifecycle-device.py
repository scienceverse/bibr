"""ONNX device requests are honoured, and Hub ids are never read from the cwd."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bibr.utils import onnx_providers

# -- ONNX device selection ---------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_cuda_preload(monkeypatch):
    monkeypatch.setattr(onnx_providers, "_cuda_libraries_preloaded", False)


def _session_providers(available: list[str], device: str | None) -> list:
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = available
    mock_ort.InferenceSession.return_value.get_providers.return_value = ["CPUExecutionProvider"]
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        onnx_providers.create_session("model.onnx", device=device, model_name="layout")
    return mock_ort.InferenceSession.call_args.kwargs["providers"]


def test_cpu_device_keeps_coreml_out_of_the_session():
    """``--device cpu`` asks for the CPU provider; CoreML may compute in FP16."""
    available = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    assert _session_providers(available, "cpu") == ["CPUExecutionProvider"]
    assert _session_providers(available, None) == available
    assert _session_providers(available, "mps") == available


def test_cuda_index_selects_the_cuda_provider_device():
    available = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    name, options = _session_providers(available, "cuda:1")[0]
    assert name == "CUDAExecutionProvider"
    assert options["device_id"] == 1
    _, options = _session_providers(available, "cuda")[0]
    assert "device_id" not in options  # ORT's default GPU


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cuda:2", 2), ("CUDA:0", 0), ("cuda", None), ("cuda:x", None), ("cpu", None), (None, None)],
)
def test_cuda_device_id_for(device, expected):
    assert onnx_providers.cuda_device_id_for(device) == expected


# -- Hub ids shadowed by the working directory ------------------------------


def _onnx_bundle(root: Path) -> Path:
    bundle = root / "onnx"
    bundle.mkdir(parents=True)
    (bundle / "model.onnx").write_bytes(b"planted")
    (bundle / "bibr_onnx.json").write_text(json.dumps({"schema_version": 1}))
    return bundle


def test_default_segmenter_id_ignores_a_cwd_directory(tmp_path, monkeypatch):
    from bibr.segmenter_base import resolve_wtpsplit_model

    monkeypatch.chdir(tmp_path)
    (tmp_path / "sat-6l-sm").mkdir()  # manifestless "legacy" bundle

    resolved = resolve_wtpsplit_model("sat-6l-sm")
    assert not resolved.is_local
    assert resolved.repo_id == "segment-any-text/sat-6l-sm"
    assert resolved.revision == "d85d2b6ddfb19036c4c8e8b3b7ca45da684b0905"

    (tmp_path / "org" / "sat").mkdir(parents=True)
    assert not resolve_wtpsplit_model("org/sat").is_local
    # Written as a path, the same directory still loads.
    assert resolve_wtpsplit_model("./sat-6l-sm").is_local
    assert resolve_wtpsplit_model(str(tmp_path / "sat-6l-sm")).is_local


@pytest.fixture
def sat_load(tmp_path, monkeypatch):
    """Build a segmenter in a cwd full of planted Hub-id paths; return what SaT got.

    The Hub is an ``hf_cache/`` directory beside the planted cwd, whose model
    repos ship no ``tokenizer.json`` (like the ``sat-*`` repos).
    """
    from huggingface_hub.errors import LocalEntryNotFoundError

    from bibr import segmenter_base

    cwd = tmp_path / "cwd"
    for planted in ("sat-3l-sm", "sat-6l-sm", "org/sat", "facebookAI/xlm-roberta-base"):
        (cwd / planted).mkdir(parents=True)
        for name in ("model_optimized.onnx", "config.json", "tokenizer.json"):
            (cwd / planted / name).write_text("planted")
    (cwd / "linked").symlink_to(cwd / "sat-3l-sm")
    monkeypatch.chdir(cwd)
    hub = tmp_path / "hf_cache"
    downloads = []

    def fake_hf_hub_download(repo_id, filename, revision=None):
        downloads.append((repo_id, filename, revision))
        if filename == "tokenizer.json" and "xlm-roberta" not in repo_id:
            raise LocalEntryNotFoundError(f"{repo_id} has no {filename}")
        target = hub / repo_id / str(revision) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("cached")
        return str(target)

    class _Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    seen = []

    def fake_sat(name, **kwargs):
        seen.append((name, kwargs))
        return SimpleNamespace(model=SimpleNamespace(ort_session=_Session()))

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr("wtpsplit_lite.SaT", fake_sat)
    monkeypatch.setattr(
        "bibr.utils.onnx_providers.get_ort_providers", lambda **kwargs: ["CPUExecutionProvider"]
    )
    monkeypatch.setattr(segmenter_base.BaseSentenceSegmenter, "_run_warmup", lambda self: None)
    segmenter_base.materialize_hub_snapshot.cache_clear()
    segmenter_base.base_tokenizer_dir.cache_clear()

    def load(model_name: str) -> tuple[str, str]:
        segmenter_base.BaseSentenceSegmenter(model_name=model_name, use_gpu=False)
        name, kwargs = seen[-1]
        assert kwargs["hub_prefix"] is None
        return name, kwargs["tokenizer_name_or_path"]

    load.hub = hub
    load.downloads = downloads
    yield load
    segmenter_base.materialize_hub_snapshot.cache_clear()
    segmenter_base.base_tokenizer_dir.cache_clear()


@pytest.mark.parametrize(
    ("model_name", "repo_id", "revision"),
    [
        ("sat-6l-sm", "segment-any-text/sat-6l-sm", "d85d2b6ddfb19036c4c8e8b3b7ca45da684b0905"),
        ("sat-3l-sm", "segment-any-text/sat-3l-sm", None),
        ("org/sat", "org/sat", None),
    ],
)
def test_segmenter_never_hands_sat_a_hub_id(sat_load, model_name, repo_id, revision):
    """wtpsplit-lite tries any model or tokenizer name as a cwd path before the
    Hub, so SaT only ever gets absolute HF cache directories for Hub models,
    including for its XLM-R base tokenizer when the repo ships none."""
    model, tokenizer = sat_load(model_name)

    assert model == str(sat_load.hub / repo_id / str(revision))
    assert tokenizer == str(sat_load.hub / "facebookAI/xlm-roberta-base/None")


def test_segmenter_hub_lookups_are_cached_per_process(sat_load):
    """Aggressive mode reloads the segmenter per phase; that asks the Hub once."""
    sat_load("sat-3l-sm")
    asked = len(sat_load.downloads)
    sat_load("sat-3l-sm")

    assert asked and len(sat_load.downloads) == asked


def test_explicit_local_segmenter_paths_still_load(sat_load, tmp_path):
    """Written as a path, a cwd bundle loads; one without a tokenizer gets the
    cached base tokenizer, not a cwd ``facebookAI/xlm-roberta-base``."""
    (tmp_path / "cwd" / "bare").mkdir()

    assert sat_load("./sat-3l-sm") == ("sat-3l-sm", "sat-3l-sm")
    assert sat_load("./bare") == (
        "bare",
        str(sat_load.hub / "facebookAI/xlm-roberta-base/None"),
    )
    with pytest.raises(ValueError, match="symlink"):
        sat_load("./linked")


def test_hub_bundle_id_ignores_a_cwd_bundle(tmp_path, monkeypatch, caplog):
    from bibr.utils.ml_runtime import find_onnx_bundle

    monkeypatch.chdir(tmp_path)
    planted = _onnx_bundle(tmp_path / "scienceverse" / "bibr-parser-v4-5-gold")
    fetched = []

    def offline(repo_id, filename, revision=None):
        fetched.append((repo_id, revision))
        raise FileNotFoundError("not cached")

    monkeypatch.setattr("bibr.utils.hf_cache.hf_download_or_cached", offline)

    assert find_onnx_bundle("scienceverse/bibr-parser-v4-5-gold", "ff50a83e") is None
    assert find_onnx_bundle("scienceverse/bibr-parser-v4-5-gold:best.pt", "ff50a83e") is None
    assert fetched == [("scienceverse/bibr-parser-v4-5-gold", "ff50a83e")] * 2
    assert "write ./scienceverse/bibr-parser-v4-5-gold" in caplog.text

    assert find_onnx_bundle("./scienceverse/bibr-parser-v4-5-gold") == Path(
        "scienceverse/bibr-parser-v4-5-gold/onnx"
    )
    assert find_onnx_bundle(str(planted.parent)) == planted


def test_bare_local_bundle_names_still_resolve(tmp_path, monkeypatch):
    """A one-part name can never be a bundle repo id, so it stays a path."""
    from bibr.utils.ml_runtime import find_onnx_bundle

    monkeypatch.chdir(tmp_path)
    bundle = _onnx_bundle(tmp_path / "parser")

    assert find_onnx_bundle("parser") == Path("parser/onnx")
    assert find_onnx_bundle(str(bundle)) == bundle


def test_hub_checkpoint_id_ignores_a_cwd_path(tmp_path, monkeypatch):
    from bibr.ner import checkpoint as ckpt_mod
    from bibr.ner.checkpoint import resolve_checkpoint

    monkeypatch.chdir(tmp_path)
    (tmp_path / "org" / "repo").mkdir(parents=True)
    (tmp_path / "org" / "repo" / "best.pt").write_bytes(b"planted")
    monkeypatch.setattr(ckpt_mod, "_discover_checkpoint_filename", lambda repo, rev: "best.pt")
    monkeypatch.setattr(
        ckpt_mod, "_download", lambda repo, filename, rev: f"/cache/{repo}@{rev}/{filename}"
    )

    assert resolve_checkpoint("org/repo", revision="abc") == "/cache/org/repo@abc/best.pt"
    # Written as a path (or passed as a Path), the local file still loads.
    assert resolve_checkpoint("./org/repo/best.pt") == "org/repo/best.pt"
    assert resolve_checkpoint(Path("org/repo/best.pt")) == "org/repo/best.pt"
    (tmp_path / "best.pt").write_bytes(b"local")
    assert resolve_checkpoint("best.pt") == "best.pt"


def test_cwd_shadow_warning_names_the_local_file(tmp_path, monkeypatch, caplog):
    from bibr.ner import checkpoint as ckpt_mod
    from bibr.ner.checkpoint import resolve_checkpoint

    monkeypatch.chdir(tmp_path)
    (tmp_path / "org" / "repo").mkdir(parents=True)
    monkeypatch.setattr(ckpt_mod, "_download", lambda repo, filename, rev: "/cache/best.pt")

    resolve_checkpoint("org/repo:best.pt")
    assert "write ./org/repo/best.pt to load the local copy" in caplog.text


@pytest.mark.parametrize("module", ["bibr.extract.front_role", "bibr.extract.geom_segmenter"])
def test_bundle_spec_ignores_a_cwd_directory(tmp_path, monkeypatch, caplog, module):
    """A planted ./org/repo used to strip the bundle filename from the Hub id,
    so the optional model failed to resolve and dropped out."""
    import importlib

    mod = importlib.import_module(module)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "org" / "repo").mkdir(parents=True)

    assert mod._resolve_spec("org/repo") == f"org/repo:{mod._BUNDLE_FILENAME}"
    assert not caplog.records  # resolve_checkpoint warns, naming the file
    assert mod._resolve_spec("org/repo:other.joblib") == "org/repo:other.joblib"
    assert mod._resolve_spec("./org/repo/b.joblib") == "./org/repo/b.joblib"
    assert mod._resolve_spec(str(tmp_path / "b.joblib")) == str(tmp_path / "b.joblib")


def test_tilde_checkpoint_path_expands(tmp_path, monkeypatch):
    from bibr.ner.checkpoint import resolve_checkpoint

    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "best.pt").write_bytes(b"weights")

    assert resolve_checkpoint("~/best.pt") == str(tmp_path / "best.pt")


@pytest.mark.parametrize(
    ("repo_id", "bare_name", "expected"),
    [
        ("org/repo", False, True),
        ("org/Qwen2.5-7B_x", False, True),
        ("sat-6l-sm", True, True),
        ("sat-6l-sm", False, False),
        ("./org/repo", False, False),
        ("../org/repo", False, False),
        ("~/repo", False, False),
        ("/abs/repo", False, False),
        ("a/b/c", False, False),
        ("C:\\models\\sat", True, False),
    ],
)
def test_is_hub_repo_id(repo_id, bare_name, expected):
    from bibr.utils.ml_runtime import is_hub_repo_id

    assert is_hub_repo_id(repo_id, bare_name=bare_name) is expected
