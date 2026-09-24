"""Unit tests for shared sentence-segmenter model resolution and inference."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from bibr.config import Settings
from bibr.segmenter_base import BaseSentenceSegmenter, resolve_wtpsplit_model


class _EchoSplitModel:
    def __init__(self) -> None:
        self.seen: list[str] | None = None
        self.kwargs: dict[str, object] | None = None

    def split(self, batch: list[str], **kwargs) -> list[list[str]]:
        self.seen = list(batch)
        self.kwargs = kwargs
        return [[item] for item in batch]


class _FailsOnBadModel:
    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def split(self, batch: list[str], **kwargs) -> list[list[str]]:
        self.seen.append(list(batch))
        if any("bad" in item for item in batch):
            raise TypeError(
                "TextEncodeInput must be Union[TextInputSequence, "
                "Tuple[InputSequence, InputSequence]]"
            )
        return [[f"model:{item}"] for item in batch]


class _FakeOrtSession:
    def __init__(self, providers: list[str]) -> None:
        self._providers = providers
        self.runs: list[tuple] = []

    def get_providers(self) -> list[str]:
        return list(self._providers)

    def get_provider_options(self) -> dict[str, dict[str, str]]:
        return {"CUDAExecutionProvider": {"device_id": "0"}} if "CUDA" in self._providers[0] else {}

    def run(self, output_names, input_feed, run_options=None):
        self.runs.append((output_names, input_feed, run_options))
        return [None]


class _RecordingSplitModel:
    def __init__(self, session_providers: list[str] | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        # wtpsplit-lite's SaT keeps the ORT session it opened at ``.model.ort_session``.
        self.opened_session = _FakeOrtSession(session_providers or ["CPUExecutionProvider"])
        self.model = SimpleNamespace(ort_session=self.opened_session)

    def split(self, batch: list[str], **kwargs) -> list[list[str]]:
        self.calls.append((list(batch), kwargs))
        return [[item] for item in batch]


def _build_segmenter(
    monkeypatch,
    *,
    model_name: str,
    threshold: float | None = None,
    requested_providers: list | None = None,
    session_providers: list[str] | None = None,
):
    # Constructing a segmenter probes the torch device, which lives in the
    # optional ``ml`` extra. The pure-resolution tests above need none of it.
    # exc_type: the module re-raises a plain ImportError (not the
    # ModuleNotFoundError importorskip defaults to) when torch is absent.
    pytest.importorskip(
        "bibr.utils.device",
        reason="segmenter startup needs the 'ml' extra",
        exc_type=ImportError,
    )

    model = _RecordingSplitModel(session_providers)
    init: dict[str, object] = {}

    def fake_sat(name, **kwargs):
        init["name"] = name
        init["kwargs"] = kwargs
        return model

    monkeypatch.setattr("wtpsplit_lite.SaT", fake_sat)
    # A pinned Hub model is materialised through the HF cache; never touch it here.
    monkeypatch.setattr(
        "bibr.segmenter_base.materialize_hub_snapshot",
        lambda repo_id, revision: (f"/pinned/{repo_id}@{revision}", None),
    )
    monkeypatch.setattr(
        "bibr.utils.onnx_providers.get_ort_providers",
        lambda **kwargs: requested_providers or ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        "bibr.utils.device.report_device",
        lambda component, device, **kwargs: init.__setitem__("reported_device", device),
    )
    segmenter = BaseSentenceSegmenter(
        model_name=model_name,
        use_gpu=False,
        threshold=threshold,
    )
    return segmenter, model, init


def _write_sealed_bundle(path: Path, *, threshold: float = 0.25) -> None:
    files = {
        "model.onnx": b"raw-model",
        "model_optimized.onnx": b"optimized-model",
        "config.json": b'{"model_type":"subword-xlmr"}',
        "tokenizer/config.json": b'{"model_type":"xlm-roberta"}',
        "tokenizer/tokenizer.json": b'{"version":"1.0","model":{}}',
        "tokenizer/tokenizer_config.json": b'{"model_max_length":512}',
        "tokenizer/sentencepiece.bpe.model": b"sentencepiece-fixture",
        "MODEL_CARD.md": b"# Scientific sentence segmenter\n",
        "segmenter_staging.json": b'{"status":"unsealed"}',
        "evaluation/metrics.json": b"{}",
        "evaluation/calibration.json": b"{}",
        "evaluation/report.md": b"# Evaluation\n",
        "evaluation/gate.json": b"{}",
        "evaluation/performance.json": b"{}",
        "provenance/release_evidence.json": b"{}",
        "provenance/source_manifest.yaml": b"sources: []\n",
        "provenance/build_manifest.json": b"{}",
        "provenance/training_run.json": b"{}",
        "provenance/uv.lock": b"version = 1\n",
        "provenance/pyproject.toml": b"[project]\nname = 'segmenter-build'\n",
        "provenance/compat_probe.py": b"print('compatible')\n",
        "parity/report.json": b"{}",
    }
    for relative, content in files.items():
        destination = path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    checksums = {
        relative: hashlib.sha256((path / relative).read_bytes()).hexdigest() for relative in files
    }
    manifest = {
        "schema_version": 1,
        "threshold": threshold,
        "model_revision": checksums["model_optimized.onnx"],
        "tokenizer_model": "FacebookAI/xlm-roberta-base",
        "tokenizer_revision": "0123456789abcdef0123456789abcdef01234567",
        "tokenizer_checksums": {
            relative: checksums[relative]
            for relative in (
                "tokenizer/config.json",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/sentencepiece.bpe.model",
            )
        },
        "language_scope": "en",
        "windowing": {"block_size": 256, "eval_stride": 128},
        "metrics_path": "evaluation/metrics.json",
        "calibration_path": "evaluation/calibration.json",
        "evaluation_report_path": "evaluation/report.md",
        "gate_path": "evaluation/gate.json",
        "performance_path": "evaluation/performance.json",
        "release_evidence_path": "provenance/release_evidence.json",
        "source_manifest_path": "provenance/source_manifest.yaml",
        "build_manifest_path": "provenance/build_manifest.json",
        "training_run_path": "provenance/training_run.json",
        "environment_lock_path": "provenance/uv.lock",
        "environment_project_path": "provenance/pyproject.toml",
        "compat_probe_path": "provenance/compat_probe.py",
        "parity_report_path": "parity/report.json",
        "model_card_path": "MODEL_CARD.md",
        "staging_manifest_path": "segmenter_staging.json",
        "checksums": checksums,
    }
    (path / "segmenter_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("value", "hub_prefix"),
    [("sat-6l-sm", "segment-any-text"), ("scienceverse/bibr-sat-science-en", None)],
)
def test_resolver_distinguishes_short_and_full_hf_ids(value, hub_prefix):
    resolved = resolve_wtpsplit_model(value)

    assert resolved.model_name == value
    assert resolved.hub_prefix == hub_prefix
    assert not resolved.is_local


def test_default_hub_model_loads_its_pinned_snapshot(monkeypatch):
    """The default short name resolves to the audited commit's snapshot directory —
    wtpsplit-lite's own config loader cannot take a revision."""
    _segmenter, _model, init = _build_segmenter(monkeypatch, model_name="sat-6l-sm")
    revision = resolve_wtpsplit_model("sat-6l-sm").revision
    assert init["name"] == f"/pinned/segment-any-text/sat-6l-sm@{revision}"
    assert init["kwargs"]["hub_prefix"] is None
    assert "from_pretrained_kwargs" not in init["kwargs"]


_CUDA_CHAIN = [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]


def test_segmenter_reports_cpu_when_its_session_dropped_cuda(monkeypatch, caplog):
    """ORT runs the session on CPU when the CUDA provider fails to start; the
    segmenter reports that, not the CUDA it asked for."""
    with caplog.at_level(logging.WARNING, logger="bibr.utils.onnx_providers"):
        _, _, init = _build_segmenter(
            monkeypatch,
            model_name="sat-6l-sm",
            requested_providers=_CUDA_CHAIN,
            session_providers=["CPUExecutionProvider"],
        )

    assert init["reported_device"] == "cpu"
    assert "wtpsplit-sat requested CUDA" in caplog.text


def test_segmenter_reports_cuda_when_its_session_has_it(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="bibr.utils.onnx_providers"):
        _, _, init = _build_segmenter(
            monkeypatch,
            model_name="sat-6l-sm",
            requested_providers=_CUDA_CHAIN,
            session_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    assert init["reported_device"] == "cuda"
    assert "requested CUDA" not in caplog.text


def test_segmenter_cuda_session_frees_unused_arena_memory_after_each_run(monkeypatch):
    """SaT runs its ORT session itself; on CUDA the segmenter wraps that session
    so every run ends with arena shrinkage."""
    _, model, _ = _build_segmenter(
        monkeypatch,
        model_name="sat-6l-sm",
        requested_providers=_CUDA_CHAIN,
        session_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    model.model.ort_session.run(["logits"], {"input_ids": 1})

    [(names, feed, run_options)] = model.opened_session.runs
    assert (names, feed) == (["logits"], {"input_ids": 1})
    assert run_options.get_run_config_entry("memory.enable_memory_arena_shrinkage") == "gpu:0"
    assert model.model.ort_session.get_providers()[0] == "CUDAExecutionProvider"


def test_segmenter_cpu_session_is_left_as_opened(monkeypatch):
    _, model, _ = _build_segmenter(monkeypatch, model_name="sat-6l-sm")

    assert model.model.ort_session is model.opened_session


def test_resolver_preserves_existing_local_directory(tmp_path):
    resolved = resolve_wtpsplit_model(str(tmp_path))

    assert resolved.model_name == str(tmp_path)
    assert resolved.hub_prefix is None
    assert resolved.is_local


def test_resolver_rejects_unsealed_staging_export(tmp_path):
    (tmp_path / "segmenter_staging.json").write_text(
        json.dumps({"status": "unsealed"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unsealed"):
        resolve_wtpsplit_model(str(tmp_path))


def test_resolver_accepts_checksum_valid_sealed_bundle(tmp_path):
    _write_sealed_bundle(tmp_path)

    resolved = resolve_wtpsplit_model(str(tmp_path))

    assert resolved.manifest_threshold == pytest.approx(0.25)
    assert resolved.manifest_revision == hashlib.sha256(b"optimized-model").hexdigest()
    assert resolved.manifest_block_size == 256
    assert resolved.manifest_eval_stride == 128


def test_resolver_rejects_empty_manifest_bypass_on_staging_export(tmp_path):
    (tmp_path / "segmenter_staging.json").write_text('{"status":"unsealed"}', encoding="utf-8")
    (tmp_path / "segmenter_manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="missing"):
        resolve_wtpsplit_model(str(tmp_path))


@pytest.mark.parametrize(
    "field",
    [
        "performance_path",
        "release_evidence_path",
        "training_run_path",
        "environment_lock_path",
        "environment_project_path",
        "compat_probe_path",
    ],
)
def test_resolver_requires_complete_release_artifact_contract(tmp_path, field):
    _write_sealed_bundle(tmp_path)
    manifest_path = tmp_path / "segmenter_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop(field)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        resolve_wtpsplit_model(str(tmp_path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tokenizer_model", "segment-any-text/xlm-roberta-base"),
        ("tokenizer_revision", "0123456789abcdef0123456789abcdef0123456"),
        ("tokenizer_revision", "0123456789ABCDEF0123456789ABCDEF01234567"),
    ],
)
def test_resolver_requires_exact_tokenizer_provenance(tmp_path, field, value):
    _write_sealed_bundle(tmp_path)
    manifest_path = tmp_path / "segmenter_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        resolve_wtpsplit_model(str(tmp_path))


def test_resolver_requires_exact_tokenizer_checksum_inventory(tmp_path):
    _write_sealed_bundle(tmp_path)
    manifest_path = tmp_path / "segmenter_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer_checksums"]["tokenizer/added_tokens.json"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer_checksums"):
        resolve_wtpsplit_model(str(tmp_path))


def test_resolver_rejects_tokenizer_checksum_mismatch(tmp_path):
    _write_sealed_bundle(tmp_path)
    manifest_path = tmp_path / "segmenter_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer_checksums"]["tokenizer/tokenizer.json"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer checksum mismatch"):
        resolve_wtpsplit_model(str(tmp_path))


def test_resolver_rejects_tampered_sealed_bundle(tmp_path):
    _write_sealed_bundle(tmp_path)
    (tmp_path / "model_optimized.onnx").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        resolve_wtpsplit_model(str(tmp_path))


@pytest.mark.parametrize(
    "windowing",
    [
        None,
        {"block_size": 256},
        {"block_size": 0, "eval_stride": 128},
        {"block_size": 64, "eval_stride": 128},
    ],
)
def test_resolver_rejects_invalid_manifest_windowing(tmp_path, windowing):
    _write_sealed_bundle(tmp_path)
    manifest_path = tmp_path / "segmenter_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if windowing is None:
        manifest.pop("windowing")
    else:
        manifest["windowing"] = windowing
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="windowing"):
        resolve_wtpsplit_model(str(tmp_path))


def test_resolver_rejects_noncanonical_evidence_path(tmp_path):
    _write_sealed_bundle(tmp_path)
    manifest_path = tmp_path / "segmenter_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metrics_path"] = "evaluation/report.md"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical path"):
        resolve_wtpsplit_model(str(tmp_path))


def test_resolver_rejects_bundle_file_symlink_escape(tmp_path):
    _write_sealed_bundle(tmp_path)
    outside = tmp_path.parent / "outside-metrics.json"
    outside.write_text("{}", encoding="utf-8")
    metrics = tmp_path / "evaluation/metrics.json"
    metrics.unlink()
    metrics.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink|escapes"):
        resolve_wtpsplit_model(str(tmp_path))


@pytest.mark.parametrize(
    "relative",
    ["evaluation/predictions.jsonl", "provenance/audit.parquet", "debug.log"],
)
def test_resolver_rejects_unexpected_or_privacy_unsafe_bundle_file(tmp_path, relative):
    _write_sealed_bundle(tmp_path)
    extra = tmp_path / relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"private-or-unexpected")
    manifest_path = tmp_path / "segmenter_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checksums"][relative] = hashlib.sha256(extra.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected"):
        resolve_wtpsplit_model(str(tmp_path))


def test_resolver_rejects_unexpected_symlink_even_when_broken(tmp_path):
    _write_sealed_bundle(tmp_path)
    (tmp_path / "unexpected-link").symlink_to(tmp_path / "does-not-exist")

    with pytest.raises(ValueError, match="symlink"):
        resolve_wtpsplit_model(str(tmp_path))


def test_complete_local_bundle_uses_its_tokenizer_assets(tmp_path, monkeypatch):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Settings, "WTPSPLIT_THRESHOLD", None)

    _, _, init = _build_segmenter(monkeypatch, model_name=str(tmp_path))

    assert init["kwargs"]["tokenizer_name_or_path"] == str(tokenizer_dir)


def test_manifestless_legacy_root_tokenizer_remains_explicitly_loadable(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    resolved = resolve_wtpsplit_model(str(tmp_path))

    assert resolved.tokenizer_name_or_path == str(tmp_path)


def test_local_bundle_without_tokenizer_json_keeps_default_tokenizer(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sentencepiece.bpe.model").write_bytes(b"fixture")

    resolved = resolve_wtpsplit_model(str(tmp_path))

    assert resolved.tokenizer_name_or_path is None


def test_constructor_threshold_wins_over_setting_and_manifest(tmp_path, monkeypatch):
    _write_sealed_bundle(tmp_path, threshold=0.25)
    monkeypatch.setattr(Settings, "WTPSPLIT_THRESHOLD", 0.5)

    segmenter, _, _ = _build_segmenter(monkeypatch, model_name=str(tmp_path), threshold=0.75)

    assert segmenter._threshold == pytest.approx(0.75)
    assert segmenter._threshold_source == "constructor"


def test_setting_threshold_wins_over_local_manifest(tmp_path, monkeypatch):
    _write_sealed_bundle(tmp_path, threshold=0.25)
    monkeypatch.setattr(Settings, "WTPSPLIT_THRESHOLD", 0.5)

    segmenter, _, _ = _build_segmenter(monkeypatch, model_name=str(tmp_path))

    assert segmenter._threshold == pytest.approx(0.5)
    assert segmenter._threshold_source == "WTPSPLIT_THRESHOLD"


def test_local_manifest_threshold_is_used_when_no_explicit_value(tmp_path, monkeypatch):
    _write_sealed_bundle(tmp_path, threshold=0.25)
    monkeypatch.setattr(Settings, "WTPSPLIT_THRESHOLD", None)

    segmenter, model, _ = _build_segmenter(monkeypatch, model_name=str(tmp_path))

    assert segmenter._threshold == pytest.approx(0.25)
    assert segmenter._threshold_source == "segmenter_manifest.json"
    assert model.calls == [
        (
            ["This is a warmup sentence. It has two parts."],
            {"threshold": 0.25, "block_size": 256, "stride": 128},
        )
    ]
    assert (
        segmenter._resolved_model.manifest_revision
        == hashlib.sha256(b"optimized-model").hexdigest()
    )


def test_missing_threshold_preserves_wtpsplit_default(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "WTPSPLIT_THRESHOLD", None)

    segmenter, model, _ = _build_segmenter(monkeypatch, model_name=str(tmp_path))

    assert segmenter._threshold is None
    assert segmenter._threshold_source == "wtpsplit default"
    assert model.calls == [(["This is a warmup sentence. It has two parts."], {})]


@pytest.mark.parametrize("threshold", [-0.01, 1.01, math.nan])
def test_constructor_rejects_threshold_outside_unit_interval(tmp_path, monkeypatch, threshold):
    monkeypatch.setattr(Settings, "WTPSPLIT_THRESHOLD", None)

    with pytest.raises(ValueError, match="threshold"):
        _build_segmenter(monkeypatch, model_name=str(tmp_path), threshold=threshold)


def test_local_manifest_rejects_threshold_outside_unit_interval(tmp_path, monkeypatch):
    _write_sealed_bundle(tmp_path, threshold=1.01)
    monkeypatch.setattr(Settings, "WTPSPLIT_THRESHOLD", None)

    with pytest.raises(ValueError, match="threshold"):
        _build_segmenter(monkeypatch, model_name=str(tmp_path))


def test_resolved_hub_prefix_is_passed_to_wtpsplit(monkeypatch):
    monkeypatch.setattr(Settings, "WTPSPLIT_THRESHOLD", None)

    _, _, init = _build_segmenter(monkeypatch, model_name="scienceverse/bibr-sat-science-en")

    assert init["name"] == "scienceverse/bibr-sat-science-en"
    assert init["kwargs"]["hub_prefix"] is None
    assert init["kwargs"]["tokenizer_name_or_path"] == "scienceverse/bibr-sat-science-en"


def test_remote_model_uses_explicit_setting_windowing(monkeypatch):
    monkeypatch.setattr(Settings, "WTPSPLIT_THRESHOLD", 0.42)
    monkeypatch.setattr(Settings, "WTPSPLIT_BLOCK_SIZE", 256, raising=False)
    monkeypatch.setattr(Settings, "WTPSPLIT_STRIDE", 128, raising=False)

    _, model, _ = _build_segmenter(
        monkeypatch,
        model_name="scienceverse/bibr-sat-science-en",
    )

    assert model.calls == [
        (
            ["This is a warmup sentence. It has two parts."],
            {"threshold": 0.42, "block_size": 256, "stride": 128},
        )
    ]


def test_remote_model_rejects_partial_setting_windowing(monkeypatch):
    monkeypatch.setattr(Settings, "WTPSPLIT_THRESHOLD", None)
    monkeypatch.setattr(Settings, "WTPSPLIT_BLOCK_SIZE", 256, raising=False)
    monkeypatch.setattr(Settings, "WTPSPLIT_STRIDE", None, raising=False)

    with pytest.raises(ValueError, match="block size and stride"):
        _build_segmenter(
            monkeypatch,
            model_name="scienceverse/bibr-sat-science-en",
        )


def test_threshold_is_passed_to_warmup_and_normal_split_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "WTPSPLIT_THRESHOLD", None)
    segmenter, model, _ = _build_segmenter(monkeypatch, model_name=str(tmp_path), threshold=0.42)

    assert segmenter._collect_split_result(["First. Second."]) == [["First. Second."]]
    assert model.calls == [
        (["This is a warmup sentence. It has two parts."], {"threshold": 0.42}),
        (["First. Second."], {"threshold": 0.42}),
    ]


def test_collect_split_result_normalizes_non_string_batch_items():
    seg = object.__new__(BaseSentenceSegmenter)
    model = _EchoSplitModel()
    seg.model = model
    seg._threshold = None

    out = seg._collect_split_result([None, b"byte text", 123])  # type: ignore[list-item]

    assert model.seen == ["", "byte text", "123"]
    assert out == [[""], ["byte text"], ["123"]]


def test_collect_split_result_falls_back_only_for_tokenizer_rejected_text():
    seg = object.__new__(BaseSentenceSegmenter)
    model = _FailsOnBadModel()
    seg.model = model
    seg._threshold = None

    out = seg._collect_split_result(["first ok", "bad text. More bad.", "last ok"])

    assert out == [["model:first ok"], ["bad text.", "More bad."], ["model:last ok"]]
    assert model.seen == [
        ["first ok", "bad text. More bad.", "last ok"],
        ["first ok"],
        ["bad text. More bad.", "last ok"],
        ["bad text. More bad."],
        ["last ok"],
    ]
