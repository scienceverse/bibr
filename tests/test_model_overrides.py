"""Model overrides: what counts as one, why a load failed, the preflight, and the run's record."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from bibr.config import snapshot_settings
from bibr.exceptions import ConfigurationError, ModelOverrideError
from bibr.utils.ml_runtime import local_onnx_bundle
from bibr.utils.model_overrides import (
    NOT_USED,
    begin_run,
    bundle_member,
    check_model_overrides,
    is_model_override,
    load_failure_kind,
    loaded_model_overrides,
    model_overrides,
    models_in_use,
    onnx_bundle_files,
    override_load_error,
    record_model_load,
    reset_loaded_model_overrides,
    run_records,
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


def _hub_error(cls_name: str):
    from huggingface_hub import errors

    response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/x"))
    return getattr(errors, cls_name)(f"{cls_name} (test)", response=response)


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


# -- why a load failed -----------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["RepositoryNotFoundError", "RevisionNotFoundError", "RemoteEntryNotFoundError"]
)
def test_a_hub_404_is_permanent(name):
    exc = _hub_error(name)
    assert load_failure_kind(exc) == "permanent"
    # Also when a loader wrapped it.
    try:
        raise OSError("could not load") from exc
    except OSError as wrapped:
        assert load_failure_kind(wrapped) == "permanent"


def test_network_failures_are_transient():
    from huggingface_hub.errors import LocalEntryNotFoundError

    assert load_failure_kind(httpx.ConnectError("name resolution")) == "transient"
    assert load_failure_kind(TimeoutError("read timed out")) == "transient"
    # "Offline and not cached", although it subclasses EntryNotFoundError.
    assert load_failure_kind(LocalEntryNotFoundError("offline, not cached")) == "transient"
    # The cache's record of a 404 at a pinned commit is the Hub's answer, not the network.
    recorded = LocalEntryNotFoundError(
        "x does not exist in r at 0123abcd (recorded by an earlier request)"
    )
    assert load_failure_kind(recorded) == "permanent"


def test_only_the_torch_extra_is_a_missing_dependency():
    from bibr.utils.ml_extra import ml_import_error

    assert load_failure_kind(ml_import_error("The CRF segmenter")) == "missing_dependency"
    assert load_failure_kind(ModuleNotFoundError("no torch", name="torch")) == "missing_dependency"
    # A pickled bundle naming a module bibr does not have is the model's problem.
    assert load_failure_kind(ModuleNotFoundError("x", name="candidate_pkg")) == "permanent"
    assert load_failure_kind(FileNotFoundError("gone")) == "permanent"
    assert load_failure_kind(ValueError("bad bundle")) == "permanent"


def test_the_load_error_says_there_is_no_fallback():
    settings = snapshot_settings()
    settings.REF_GEOM_SEG_MODEL_ID = "./candidate.joblib"
    err = override_load_error(settings, "REF_GEOM_SEG_MODEL_ID", OSError("gone"))
    assert isinstance(err, ModelOverrideError) and isinstance(err, ConfigurationError)
    assert "REF_GEOM_SEG_MODEL_ID='./candidate.joblib'" in str(err)
    assert "never falls back" in str(err)
    # An error already raised for the override is passed on as it is, not wrapped again.
    assert override_load_error(settings, "REF_GEOM_SEG_MODEL_ID", err) is err


def test_a_transient_failure_of_an_override_behaves_as_for_the_default():
    settings = snapshot_settings()
    settings.ml.front_role_model_id = "myorg/front-role-mirror"
    begin_run(settings)
    assert override_load_error(settings, "ML_FRONT_ROLE_MODEL_ID", httpx.ConnectError("x")) is None
    (record,) = run_records()
    assert record["loaded"] is False and record["reason"].startswith("network error")


def test_an_overridden_geom_segmenter_that_does_not_exist_stops(tmp_path):
    from bibr.extract import ref_extractor

    settings = snapshot_settings()
    settings.REF_GEOM_SEG_MODEL_ID = str(tmp_path / "missing.joblib")
    with pytest.raises(ModelOverrideError, match="never falls back"):
        ref_extractor._get_geom_segmenter(settings)


# -- bundle files ------------------------------------------------------------------------


def test_a_local_bundle_may_sit_in_the_named_directory_itself(tmp_path):
    flat = _flat_bundle("ner", tmp_path / "parser-int8")
    assert local_onnx_bundle(flat) == flat
    assert local_onnx_bundle(BUNDLES / "ner") == BUNDLES / "ner" / "onnx"
    assert local_onnx_bundle(tmp_path / "missing") is None


@pytest.mark.parametrize("name", ["../outside.bin", "/etc/hostname", "a/../../b", "", 7, None])
def test_manifest_file_names_stay_inside_the_bundle(tmp_path, name):
    with pytest.raises(ConfigurationError):
        bundle_member(tmp_path, name)


def test_a_manifest_listing_a_file_outside_the_bundle_is_refused(tmp_path):
    bundle = _flat_bundle("ner", tmp_path / "ner")
    manifest = json.loads((bundle / "bibr_onnx.json").read_text())
    manifest["files"] = [*manifest.get("files", []), "../secret.txt"]
    (bundle / "bibr_onnx.json").write_text(json.dumps(manifest))
    with pytest.raises(ConfigurationError, match="outside the bundle"):
        onnx_bundle_files(bundle)
    settings = snapshot_settings()
    settings.NER_PARSER_CKPT = str(bundle)
    (problem,), _ = check_model_overrides(settings)
    assert "outside the bundle" in problem


# -- preflight -------------------------------------------------------------------------


def test_preflight_names_a_missing_path(tmp_path):
    settings = snapshot_settings()
    settings.NER_PARSER_CKPT = str(tmp_path / "missing")
    (problem,), warnings = check_model_overrides(settings)
    assert "NER_PARSER_CKPT" in problem and "no such file" in problem
    assert warnings == []


def test_preflight_names_a_bundle_of_the_wrong_model(tmp_path):
    settings = snapshot_settings()
    settings.NER_PARSER_CKPT = str(_flat_bundle("paper", tmp_path / "paper"))
    (problem,), _ = check_model_overrides(settings)
    assert "'paper_classifier' model, not a 'ner_parser'" in problem


def test_preflight_passes_a_loadable_bundle_and_skips_hub_ids(tmp_path):
    settings = snapshot_settings()
    settings.NER_PARSER_CKPT = str(_flat_bundle("ner", tmp_path / "ner"))
    settings.ml.section_classifier_model_id = "org/another-section-model"
    assert check_model_overrides(settings) == ([], [])


def test_preflight_never_raises(tmp_path):
    # A malformed manifest: "files" is not a list.
    bundle = _flat_bundle("ner", tmp_path / "ner")
    manifest = json.loads((bundle / "bibr_onnx.json").read_text())
    manifest["files"] = 7
    (bundle / "bibr_onnx.json").write_text(json.dumps(manifest))
    settings = snapshot_settings()
    settings.NER_PARSER_CKPT = str(bundle)
    (problem,), _ = check_model_overrides(settings)
    assert problem.startswith("NER_PARSER_CKPT=")


def test_preflight_checks_only_the_models_the_run_loads(tmp_path):
    stale = str(tmp_path / "stale")
    settings = snapshot_settings()
    settings.REF_PARSE_STRATEGY = "ner"
    settings.REF_SEG_STRATEGY = "llm"
    settings.REF_GEOM_SEG_MODEL_ID = stale + ".joblib"
    settings.REF_GEOM_TAGGER_MODEL_ID = stale
    # The geometry models belong to REF_SEG_STRATEGY=geom only.
    assert check_model_overrides(settings) == ([], [])
    settings.REF_SEG_STRATEGY = "geom"
    problems, _ = check_model_overrides(settings)
    assert [p.split("=")[0] for p in problems] == [
        "REF_GEOM_SEG_MODEL_ID",
        "REF_GEOM_TAGGER_MODEL_ID",
    ]
    # No reference model is loaded when references are off.
    settings.REF_PARSE_STRATEGY = "off"
    settings.NER_PARSER_CKPT = stale
    assert check_model_overrides(settings) == ([], [])


def test_models_in_use_follow_the_run():
    settings = snapshot_settings()
    settings.ml.front_tagger_mode = "off"
    settings.ml.front_role_enabled = False
    used = models_in_use(settings)
    assert "ML_FRONT_TAGGER_MODEL_ID" not in used and "ML_FRONT_ROLE_MODEL_ID" not in used
    from bibr.run_mode import fast_settings

    assert "ML_FRONT_TAGGER_MODEL_ID" in models_in_use(fast_settings(settings))


def test_preflight_skips_a_front_tagger_the_run_leaves_off(tmp_path):
    from bibr.run_mode import fast_settings

    settings = snapshot_settings()
    settings.ml.front_tagger_mode = "off"
    settings.ml.front_tagger_model_id = str(tmp_path / "missing")
    assert check_model_overrides(settings) == ([], [])
    (problem,), _ = check_model_overrides(fast_settings(settings))
    assert "ML_FRONT_TAGGER_MODEL_ID" in problem


def test_a_crf_override_without_torch_is_a_warning_not_a_problem(tmp_path, monkeypatch):
    monkeypatch.setattr("bibr.utils.ml_runtime.torch_available", lambda: False)
    settings = snapshot_settings()
    settings.NER_SEG_CKPT = str(tmp_path / "seg.pt")
    problems, (warning,) = check_model_overrides(settings)
    assert problems == []
    assert warning.startswith("NER_SEG_CKPT=") and "torch extra" in warning


# -- the CLI preflight -------------------------------------------------------------------


def _cli_config(**kw):
    return SimpleNamespace(**{"mode": "full", "no_llm": False, "refs": None, "ref_seg": None, **kw})


def test_cli_preflight_reports_the_problem(tmp_path, monkeypatch):
    from bibr.local.cli.run_config import _preflight_model_overrides

    settings = snapshot_settings()
    monkeypatch.setattr("bibr.config.Settings", settings)
    assert _preflight_model_overrides(_cli_config()) is None
    settings.ml.front_role_model_id = str(tmp_path / "front_role.joblib")
    problem = _preflight_model_overrides(_cli_config())
    assert problem.startswith("Model override problem: ML_FRONT_ROLE_MODEL_ID=")


def test_cli_preflight_follows_the_run_options(tmp_path, monkeypatch):
    from bibr.local.cli.run_config import _preflight_model_overrides

    settings = snapshot_settings()
    settings.REF_SEG_STRATEGY = "geom"
    settings.REF_PARSE_STRATEGY = "ner"
    settings.REF_GEOM_TAGGER_MODEL_ID = str(tmp_path / "stale")
    monkeypatch.setattr("bibr.config.Settings", settings)
    assert _preflight_model_overrides(_cli_config()) is not None
    # --ref-seg llm and --refs off leave the tagger unused: a stale setting does not block.
    assert _preflight_model_overrides(_cli_config(ref_seg="llm")) is None
    assert _preflight_model_overrides(_cli_config(refs="off")) is None


def test_cli_preflight_passes_warnings_on(tmp_path, monkeypatch):
    from bibr.local.cli.run_config import _preflight_model_overrides

    monkeypatch.setattr("bibr.utils.ml_runtime.torch_available", lambda: False)
    settings = snapshot_settings()
    settings.NER_SEG_CKPT = str(tmp_path / "seg.pt")
    monkeypatch.setattr("bibr.config.Settings", settings)
    seen: list[str] = []
    assert _preflight_model_overrides(_cli_config(), warn=seen.append) is None
    assert len(seen) == 1 and seen[0].startswith("Model override: NER_SEG_CKPT=")


# -- the run's record ----------------------------------------------------------------------


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
    assert record["loaded"] is True and record["reason"] is None
    by_name = {Path(f["path"]).name: f["sha256"] for f in record["files"]}
    assert set(by_name) == {"bibr_onnx.json", "model.onnx", "tokenizer.json"}
    for name, digest in by_name.items():
        assert digest == hashlib.sha256((bundle / name).read_bytes()).hexdigest()


def test_the_record_is_per_run(tmp_path):
    settings = snapshot_settings()
    settings.ml.front_role_model_id = str(tmp_path / "front_role.joblib")
    settings.NER_PARSER_REVISION = "0" * 40
    (tmp_path / "front_role.joblib").write_bytes(b"x")
    begin_run(settings)
    assert [(r["setting"], r["loaded"], r["reason"]) for r in run_records()] == [
        ("NER_PARSER_CKPT", False, NOT_USED),
        ("ML_FRONT_ROLE_MODEL_ID", False, NOT_USED),
    ]
    record_model_load(settings, "ML_FRONT_ROLE_MODEL_ID", [tmp_path / "front_role.joblib"])
    loaded = {r["setting"]: r for r in run_records()}
    assert loaded["ML_FRONT_ROLE_MODEL_ID"]["loaded"] is True
    assert (
        loaded["ML_FRONT_ROLE_MODEL_ID"]["files"][0]["sha256"] == hashlib.sha256(b"x").hexdigest()
    )
    # A second run in the same process starts from nothing.
    other = snapshot_settings()
    other.NER_SEG_CKPT = str(tmp_path / "seg.pt")
    begin_run(other)
    assert [(r["setting"], r["loaded"]) for r in run_records()] == [("NER_SEG_CKPT", False)]


# -- an override that does not exist reaches the caller ----------------------------------

_REFS = [
    "Aknin, L. B. (2013). Prosocial spending and well-being. Journal of X, 1, 2-3.",
    "Carstensen, L. L. (2009). Another title here. Journal of Y, 4, 5-6.",
]
_LINES = [
    {
        "text": t,
        "page": 5,
        "x0": 72.0,
        "y_top": 700.0 - 14 * i,
        "x1": 520.0,
        "y_bottom": 690.0 - 14 * i,
        "font_size": 10.0,
    }
    for i, t in enumerate(_REFS)
]


def _metadata_extractor(settings, seg="geom", parse="ner"):
    from unittest.mock import AsyncMock

    import pandas as pd

    from bibr.extract.extractor import MetadataExtractor
    from bibr.paper import PaperMetadata
    from bibr.paper_contents import PaperContents

    contents = PaperContents(
        sentences=[], sections=[], tables=[], links=[], sections_text={}, ref_line_geometry=_LINES
    )
    llm = MagicMock()
    llm.segment_references = AsyncMock(return_value=[])
    ex = MetadataExtractor(
        contents,
        file_hash="h",
        llm_client=llm,
        ref_seg_strategy=seg,
        ref_parse_strategy=parse,
        settings=settings,
    )
    ex.sentences_df = pd.DataFrame({"text": ["A sentence."]})

    async def core():
        ex.metadata = PaperMetadata(doi="", title="T", keywords=[], authors=[])

    ex.extract_core_metadata = core
    ex._collect_reference_rows = lambda: pd.DataFrame({"text": _REFS, "page_number": [5, 5]})
    return ex


@pytest.mark.parametrize(
    ("env", "value"),
    [
        ("REF_GEOM_TAGGER_MODEL_ID", "/nonexistent/tagger"),
        ("REF_GEOM_SEG_MODEL_ID", "/nonexistent/geom.joblib"),
        ("NER_PARSER_CKPT", "/nonexistent/parser"),
    ],
)
async def test_an_override_that_does_not_exist_stops_extract_all_metadata(env, value):
    settings = snapshot_settings()
    settings.ml.runtime = "onnx"
    setattr(settings, env, value)
    ex = _metadata_extractor(settings)
    with pytest.raises(ModelOverrideError, match=env):
        await ex.extract_all_metadata()


async def test_an_overridden_crf_that_does_not_exist_stops(monkeypatch):
    import sys

    def missing(*args, **kwargs):
        raise FileNotFoundError("no checkpoint there")

    from bibr.extract import ref_extractor

    monkeypatch.setitem(sys.modules, "bibr.ner.segmenter", SimpleNamespace(RefSegmenter=missing))
    monkeypatch.setattr(ref_extractor, "_NER_SEGMENTER", None)
    settings = snapshot_settings()
    settings.NER_SEG_CKPT = "/nonexistent/seg.pt"
    ex = _metadata_extractor(settings, seg="crf", parse="off")
    ex.refs._ref_parse_strategy = "ner"
    with pytest.raises(ModelOverrideError, match="NER_SEG_CKPT"):
        await ex.refs._segment_references("\n".join(_REFS), "crf")


async def test_an_overridden_crf_without_torch_behaves_as_the_default(monkeypatch):
    import sys

    from bibr.extract import ref_extractor
    from bibr.utils.ml_extra import ml_import_error

    def no_torch(*args, **kwargs):
        raise ml_import_error("The CRF segmenter")

    monkeypatch.setitem(sys.modules, "bibr.ner.segmenter", SimpleNamespace(RefSegmenter=no_torch))
    monkeypatch.setattr(ref_extractor, "_NER_SEGMENTER", None)
    settings = snapshot_settings()
    settings.NER_SEG_CKPT = "/elsewhere/seg.pt"
    begin_run(settings)
    ex = _metadata_extractor(settings, seg="crf")
    out = await ex.refs._segment_references("\n".join(_REFS), "crf")
    assert isinstance(out, list)  # the marker split or nothing, as for the default
    (record,) = run_records()
    assert record["loaded"] is False and record["reason"].startswith("torch extra not installed")


async def test_a_crf_inference_error_is_not_a_load_failure(monkeypatch):
    from bibr.extract import ref_extractor

    segmenter = MagicMock()
    segmenter.segment.side_effect = RuntimeError("one paper's bad input")
    monkeypatch.setattr(ref_extractor, "_get_ner_segmenter", lambda settings: segmenter)
    settings = snapshot_settings()
    settings.NER_SEG_CKPT = "/elsewhere/seg.pt"
    ex = _metadata_extractor(settings, seg="crf")
    ex.refs._crf_segment_or_recover("\n".join(_REFS))  # no raise, as at the default
    assert ex.refs._segmentation_attempts[0].reason_flags == ("segmentation_error",)


async def test_an_override_that_does_not_exist_stops_the_native_path():
    from unittest.mock import patch

    import pandas as pd

    from bibr.paper import PaperMetadata
    from bibr.paper_contents import PaperContents
    from bibr.pipeline.stages.post_parse import _resolve_preparsed_references

    settings = snapshot_settings()
    settings.ml.runtime = "onnx"
    settings.NER_PARSER_CKPT = "/nonexistent/parser"
    contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={})
    contents.native_ref_strings = list(_REFS)
    metadata = PaperMetadata(doi="", title="T", keywords=[], authors=[])
    with (
        patch(
            "bibr.extract.extractor.MetadataExtractor._collect_reference_rows",
            return_value=pd.DataFrame({"text": _REFS, "page_number": [5, 5]}),
        ),
        pytest.raises(ModelOverrideError, match="NER_PARSER_CKPT"),
    ):
        await _resolve_preparsed_references(
            contents, metadata, "h", MagicMock(), "geom", "ner", settings=settings
        )
    assert not metadata._references_incomplete_diagnostic


@pytest.mark.parametrize(
    ("module", "loader", "attr"),
    [
        (
            "bibr.structure.paper_classifier_common",
            "load_paper_classifier",
            "paper_classifier_model_id",
        ),
        (
            "bibr.structure.section_classifier_common",
            "load_section_classifier",
            "section_classifier_model_id",
        ),
    ],
)
def test_an_overridden_classifier_tells_the_network_from_a_missing_bundle(
    monkeypatch, module, loader, attr
):
    import importlib

    asked = []

    def find(model_id, revision=None, *, label="model", raise_transient=False):
        asked.append(raise_transient)
        if raise_transient:
            raise httpx.ConnectError("name resolution")
        return None

    monkeypatch.setattr("bibr.utils.ml_runtime.find_onnx_bundle", find)
    settings = snapshot_settings()
    settings.ml.runtime = "onnx"
    setattr(settings.ml, attr, "myorg/classifier-mirror")
    load = getattr(importlib.import_module(module), loader)
    with pytest.raises(httpx.ConnectError) as raised:
        load("myorg/classifier-mirror", revision="main", settings=settings)
    assert load_failure_kind(raised.value) == "transient"
    # The shipped id keeps the shipped lookup, which reads any failure as "no bundle".
    from bibr.config import MlOptions

    setattr(settings.ml, attr, MlOptions.model_fields[attr].default)
    with pytest.raises(ConfigurationError):
        load(MlOptions.model_fields[attr].default, revision="main", settings=settings)
    assert asked == [True, False]
