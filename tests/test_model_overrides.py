"""Model overrides: what counts as one, the preflight check, strict loading and identity records."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bibr.config import snapshot_settings
from bibr.exceptions import ConfigurationError
from bibr.utils.ml_runtime import local_onnx_bundle
from bibr.utils.model_overrides import (
    check_model_overrides,
    is_model_override,
    loaded_model_overrides,
    model_overrides,
    override_load_error,
    record_model_load,
    reset_loaded_model_overrides,
)

BUNDLES = Path(__file__).parent / "fixtures" / "onnx"


@pytest.fixture(autouse=True)
def _fresh_records():
    reset_loaded_model_overrides()
    yield
    reset_loaded_model_overrides()


def _flat_bundle(src: str, dest: Path) -> Path:
    """A copy of a committed bundle with its files in ``dest`` itself (an export's layout)."""
    shutil.copytree(BUNDLES / src / "onnx", dest)
    return dest


# -- what is an override ------------------------------------------------------------


def test_the_shipped_settings_have_no_overrides():
    assert model_overrides(snapshot_settings()) == []


def test_a_value_or_a_revision_other_than_the_default_is_an_override():
    settings = snapshot_settings()
    settings.NER_PARSER_REVISION = "0" * 40
    assert is_model_override(settings, "NER_PARSER_CKPT")
    settings = snapshot_settings()
    settings.ml.front_role_model_id = "./front_role.joblib"
    assert [s.env for s in model_overrides(settings)] == ["ML_FRONT_ROLE_MODEL_ID"]


def test_unset_models_and_test_doubles_are_not_overrides():
    settings = snapshot_settings()
    settings.ml.section_classifier_model_id = None
    assert not is_model_override(settings, "ML_SECTION_CLASSIFIER_MODEL_ID")
    assert model_overrides(MagicMock()) == []


def test_nothing_is_recorded_for_a_default():
    assert record_model_load(snapshot_settings(), "NER_PARSER_CKPT", ["/nonexistent"]) is None
    assert loaded_model_overrides() == []
    assert override_load_error(snapshot_settings(), "NER_PARSER_CKPT", OSError("x")) is None


# -- bundle resolution ---------------------------------------------------------------


def test_a_local_bundle_may_sit_in_the_named_directory_itself(tmp_path):
    flat = _flat_bundle("ner", tmp_path / "parser-int8")
    assert local_onnx_bundle(flat) == flat
    assert local_onnx_bundle(BUNDLES / "ner") == BUNDLES / "ner" / "onnx"
    assert local_onnx_bundle(tmp_path / "missing") is None


# -- preflight -------------------------------------------------------------------------


def test_preflight_names_a_missing_path(tmp_path):
    settings = snapshot_settings()
    settings.NER_PARSER_CKPT = str(tmp_path / "missing")
    (problem,) = check_model_overrides(settings)
    assert "NER_PARSER_CKPT" in problem and "no such file" in problem


def test_preflight_names_a_bundle_of_the_wrong_model(tmp_path):
    settings = snapshot_settings()
    settings.NER_PARSER_CKPT = str(_flat_bundle("paper", tmp_path / "paper"))
    (problem,) = check_model_overrides(settings)
    assert "'paper_classifier' model, not a 'ner_parser'" in problem


def test_preflight_passes_a_loadable_bundle_and_skips_hub_ids(tmp_path):
    settings = snapshot_settings()
    settings.NER_PARSER_CKPT = str(_flat_bundle("ner", tmp_path / "ner"))
    settings.ml.section_classifier_model_id = "org/another-section-model"
    assert check_model_overrides(settings) == []


def test_preflight_skips_a_front_tagger_the_run_leaves_off(tmp_path):
    settings = snapshot_settings()
    settings.ml.front_tagger_mode = "off"
    settings.ml.front_tagger_model_id = str(tmp_path / "missing")
    assert check_model_overrides(settings) == []
    (problem,) = check_model_overrides(settings, fast=True)
    assert "ML_FRONT_TAGGER_MODEL_ID" in problem


def test_cli_preflight_reports_the_problem(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from bibr.local.cli.run_config import _preflight_model_overrides

    settings = snapshot_settings()
    monkeypatch.setattr("bibr.config.Settings", settings)
    assert _preflight_model_overrides(SimpleNamespace(mode="full")) is None
    settings.ml.front_role_model_id = str(tmp_path / "front_role.joblib")
    problem = _preflight_model_overrides(SimpleNamespace(mode="full"))
    assert problem.startswith("Model override problem: ML_FRONT_ROLE_MODEL_ID=")


# -- loading and identity ------------------------------------------------------------------


def test_an_overridden_parser_is_loaded_and_recorded_by_checksum(tmp_path):
    from bibr.extract.ref_extractor import _get_ner_parser, unload_ner_parser

    bundle = _flat_bundle("ner", tmp_path / "parser-int8")
    settings = snapshot_settings()
    settings.NER_PARSER_CKPT = str(bundle)
    settings.NER_PARSER_REVISION = None
    settings.ml.runtime = "onnx"
    unload_ner_parser()
    try:
        parser = _get_ner_parser(settings)
        assert Path(parser.bundle_dir) == bundle
    finally:
        unload_ner_parser()
    (record,) = loaded_model_overrides()
    assert record["setting"] == "NER_PARSER_CKPT"
    assert record["value"] == str(bundle)
    by_name = {Path(f["path"]).name: f["sha256"] for f in record["files"]}
    assert set(by_name) == {"bibr_onnx.json", "model.onnx", "tokenizer.json"}
    for name, digest in by_name.items():
        assert digest == hashlib.sha256((bundle / name).read_bytes()).hexdigest()


def test_the_load_error_says_there_is_no_fallback():
    settings = snapshot_settings()
    settings.REF_GEOM_SEG_MODEL_ID = "./candidate.joblib"
    err = override_load_error(settings, "REF_GEOM_SEG_MODEL_ID", OSError("gone"))
    assert isinstance(err, ConfigurationError)
    assert "REF_GEOM_SEG_MODEL_ID='./candidate.joblib'" in str(err)
    assert "never falls back" in str(err)


def test_an_overridden_geom_segmenter_that_fails_stops(tmp_path):
    from bibr.extract import ref_extractor

    settings = snapshot_settings()
    settings.REF_GEOM_SEG_MODEL_ID = str(tmp_path / "missing.joblib")
    with pytest.raises(ConfigurationError, match="never falls back"):
        ref_extractor._get_geom_segmenter(settings)
