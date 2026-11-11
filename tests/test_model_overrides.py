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


@pytest.mark.parametrize(
    ("exc", "reason"),
    [
        (httpx.ConnectError("x"), "the Hub could not be reached"),
        (ModuleNotFoundError("no torch", name="torch"), "the torch extra is not installed"),
    ],
)
def test_every_load_failure_of_an_override_stops(exc, reason):
    settings = snapshot_settings()
    settings.ml.front_role_model_id = "myorg/front-role-mirror"
    begin_run(settings)
    err = override_load_error(settings, "ML_FRONT_ROLE_MODEL_ID", exc)
    assert isinstance(err, ModelOverrideError) and reason in str(err)
    (record,) = run_records()
    assert record["loaded"] is False and record["reason"].startswith(reason)
    # The shipped setting keeps its own fallback.
    assert override_load_error(snapshot_settings(), "ML_FRONT_ROLE_MODEL_ID", exc) is None


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


def test_preflight_loads_a_bundle_and_a_hub_id(tmp_path, monkeypatch):
    from bibr.extract.ref_extractor import unload_ner_parser

    settings = snapshot_settings()
    settings.ml.runtime = "onnx"
    settings.NER_PARSER_CKPT = str(_flat_bundle("ner", tmp_path / "ner"))
    settings.ml.section_classifier_model_id = "org/another-section-model"
    # Without load (bibr serve's startup), only the local files are checked.
    assert check_model_overrides(settings, load=False) == ([], [])
    loaded = []

    def section(model_id, revision="main", device=None, settings=None):
        loaded.append((model_id, device))
        return object()

    monkeypatch.setattr("bibr.structure.section_classifier_common.load_section_classifier", section)
    try:
        assert check_model_overrides(settings) == ([], [])
    finally:
        unload_ner_parser()
    assert loaded == [("org/another-section-model", "cpu")]

    def unreachable(model_id, revision="main", device=None, settings=None):
        raise httpx.ConnectError("name resolution")

    monkeypatch.setattr(
        "bibr.structure.section_classifier_common.load_section_classifier", unreachable
    )
    settings.NER_PARSER_CKPT = snapshot_settings().NER_PARSER_CKPT
    (problem,), _ = check_model_overrides(settings)
    assert problem.startswith("ML_SECTION_CLASSIFIER_MODEL_ID='org/another-section-model")
    assert "the Hub could not be reached" in problem


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


def test_a_crf_override_without_torch_is_a_problem(tmp_path, monkeypatch):
    import sys

    from bibr.extract import ref_extractor
    from bibr.utils.ml_extra import ml_import_error

    def no_torch(*args, **kwargs):
        raise ml_import_error("The CRF segmenter")

    monkeypatch.setitem(sys.modules, "bibr.ner.segmenter", SimpleNamespace(RefSegmenter=no_torch))
    monkeypatch.setattr(ref_extractor, "_NER_SEGMENTER", None)
    checkpoint = tmp_path / "seg.pt"
    checkpoint.write_bytes(b"weights")
    settings = snapshot_settings()
    settings.NER_SEG_CKPT = str(checkpoint)
    (problem,), warnings = check_model_overrides(settings)
    assert warnings == []
    assert problem.startswith("NER_SEG_CKPT=") and "the torch extra is not installed" in problem


def test_preflight_accepts_a_local_torch_classifier_and_refuses_a_broken_one(tmp_path, monkeypatch):
    monkeypatch.setattr("bibr.utils.ml_runtime.torch_available", lambda: True)
    model = tmp_path / "section-torch"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"weights")
    settings = snapshot_settings()
    settings.ml.runtime = "torch"
    settings.ml.section_classifier_model_id = str(model)
    assert check_model_overrides(settings, load=False) == ([], [])
    # The load reads it with from_pretrained, as the run does.
    loaded = []
    monkeypatch.setattr(
        "bibr.structure.section_classifier_common.load_section_classifier",
        lambda model_id, revision="main", device=None, settings=None: loaded.append(model_id),
    )
    assert check_model_overrides(settings) == ([], [])
    assert loaded == [str(model)]
    # A directory without weights is neither a bundle nor a torch model.
    (model / "model.safetensors").unlink()
    (problem,), _ = check_model_overrides(settings, load=False)
    assert "nor a torch model directory" in problem and "model.safetensors" in problem
    # Weights that do not load are refused by the load.
    (model / "model.safetensors").write_bytes(b"not weights")

    def broken(model_id, revision="main", device=None, settings=None):
        raise OSError("unable to load weights")

    monkeypatch.setattr("bibr.structure.section_classifier_common.load_section_classifier", broken)
    (problem,), _ = check_model_overrides(settings)
    assert "unable to load weights" in problem
    # An ONNX-only run still wants a bundle.
    settings.ml.runtime = "onnx"
    (problem,), _ = check_model_overrides(settings, load=False)
    assert "no ONNX bundle there" in problem


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


def test_cli_preflight_loads_a_hub_override(monkeypatch):
    from bibr.local.cli.run_config import _preflight_model_overrides

    settings = snapshot_settings()
    settings.ml.front_role_model_id = "no-such-org/front-role"
    monkeypatch.setattr("bibr.config.Settings", settings)

    def unreachable(*args, **kwargs):
        raise httpx.ConnectError("name resolution")

    monkeypatch.setattr("bibr.extract.front_role.FrontRoleClassifier", unreachable)
    seen: list[str] = []
    problem = _preflight_model_overrides(_cli_config(), warn=seen.append)
    assert problem.startswith("Model override problem: ML_FRONT_ROLE_MODEL_ID=")
    assert "the Hub could not be reached" in problem and seen == []


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
    # A local model has no revision, whatever the (default) revision setting says.
    assert loaded["ML_FRONT_ROLE_MODEL_ID"]["revision"] is None
    assert loaded["NER_PARSER_CKPT"]["revision"] == "0" * 40
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


async def test_an_overridden_crf_without_torch_stops(monkeypatch):
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
    with pytest.raises(ModelOverrideError, match="the torch extra is not installed"):
        await ex.refs._segment_references("\n".join(_REFS), "crf")
    (record,) = run_records()
    assert record["loaded"] is False
    assert record["reason"].startswith("the torch extra is not installed")


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


# -- an override that loads but raises at inference ---------------------------------


def _crf_extractor(settings, monkeypatch, segmenter, file_hash):
    from bibr.extract import ref_extractor
    from bibr.extract.ref_extractor import ReferenceExtractor
    from bibr.paper_contents import PaperContents

    monkeypatch.setattr(ref_extractor, "_get_ner_segmenter", lambda settings: segmenter)
    contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={})
    ex = ReferenceExtractor(
        contents,
        file_hash=file_hash,
        llm_client=MagicMock(),
        seg_strategy="crf",
        parse_strategy="off",
        settings=settings,
    )
    return ex


def _codes(ex):
    return [str(w.code) for w in ex.contents.processing_warnings]


def _raising_segmenter():
    segmenter = MagicMock()
    segmenter.segment.side_effect = RuntimeError("bad input")
    return segmenter


def test_one_inference_failure_falls_back_and_is_recorded_on_the_paper(monkeypatch):
    from bibr.processing_warnings import WarningCode

    settings = snapshot_settings()
    settings.NER_SEG_CKPT = "/elsewhere/seg.pt"
    begin_run(settings)
    ex = _crf_extractor(settings, monkeypatch, _raising_segmenter(), "paper-1")
    ex._crf_segment_or_recover("\n".join(_REFS))
    assert ex._segmentation_attempts[0].reason_flags == ("segmentation_error",)
    assert _codes(ex).count(WarningCode.MODEL_OVERRIDE_INFERENCE_FAILED) == 1
    (warning,) = [
        w
        for w in ex.contents.processing_warnings
        if w.code == WarningCode.MODEL_OVERRIDE_INFERENCE_FAILED
    ]
    assert "NER_SEG_CKPT" in warning.message and "RuntimeError" in warning.message
    assert "/elsewhere" not in warning.message


def test_a_default_model_that_raises_falls_back_as_before(monkeypatch):
    from bibr.processing_warnings import WarningCode
    from bibr.utils.model_overrides import INFERENCE_FAILURE_LIMIT

    settings = snapshot_settings()
    begin_run(settings)
    for paper in range(INFERENCE_FAILURE_LIMIT + 1):
        ex = _crf_extractor(settings, monkeypatch, _raising_segmenter(), f"paper-{paper}")
        ex._crf_segment_or_recover("\n".join(_REFS))
        assert WarningCode.MODEL_OVERRIDE_INFERENCE_FAILED not in _codes(ex)


def test_failures_on_three_papers_in_a_row_stop_the_run(monkeypatch):
    from bibr.utils.model_overrides import INFERENCE_FAILURE_LIMIT

    assert INFERENCE_FAILURE_LIMIT == 3
    settings = snapshot_settings()
    settings.NER_SEG_CKPT = "/elsewhere/seg.pt"
    begin_run(settings)
    for paper in ("paper-1", "paper-2"):
        ex = _crf_extractor(settings, monkeypatch, _raising_segmenter(), paper)
        ex._crf_segment_or_recover("\n".join(_REFS))
        # The same paper failing again is still one paper.
        ex._crf_segment_or_recover("\n".join(_REFS))
    ex = _crf_extractor(settings, monkeypatch, _raising_segmenter(), "paper-3")
    with pytest.raises(ModelOverrideError, match=r"NER_SEG_CKPT.*raised at inference on 3 papers"):
        ex._crf_segment_or_recover("\n".join(_REFS))


def test_a_paper_the_override_ran_on_ends_the_streak(monkeypatch):
    settings = snapshot_settings()
    settings.NER_SEG_CKPT = "/elsewhere/seg.pt"
    begin_run(settings)
    working = MagicMock()
    working.segment.return_value = list(_REFS)
    for paper, segmenter in (
        ("paper-1", _raising_segmenter()),
        ("paper-2", _raising_segmenter()),
        ("paper-3", working),
        ("paper-4", _raising_segmenter()),
        ("paper-5", _raising_segmenter()),
    ):
        ex = _crf_extractor(settings, monkeypatch, segmenter, paper)
        ex._crf_segment_or_recover("\n".join(_REFS))  # never three in a row: no raise


def test_a_new_run_starts_a_new_streak(monkeypatch):
    settings = snapshot_settings()
    settings.NER_SEG_CKPT = "/elsewhere/seg.pt"
    for run in range(2):
        begin_run(settings)
        for paper in ("paper-1", "paper-2"):
            ex = _crf_extractor(settings, monkeypatch, _raising_segmenter(), f"{run}-{paper}")
            ex._crf_segment_or_recover("\n".join(_REFS))


# -- a Hub id that is not cached, offline -------------------------------------------


def test_an_uncached_hub_override_offline_stops_the_preflight_and_the_run(tmp_path, monkeypatch):
    from huggingface_hub import constants

    from bibr.extract import ref_extractor

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub-cache"))
    monkeypatch.setattr(ref_extractor, "_TEXT_SEGMENTER", None)
    monkeypatch.setattr(ref_extractor, "_TEXT_SEGMENTER_KEY", None)
    settings = snapshot_settings()
    settings.REF_TEXT_SEG_MODEL_ID = "no-such-org-for-tests/never-cached-text-segmenter"
    settings.REF_SEG_STRATEGY = "crf"

    problems, warnings = check_model_overrides(settings)
    assert warnings == []
    (problem,) = [p for p in problems if "REF_TEXT_SEG_MODEL_ID" in p]
    assert "the Hub could not be reached and the files are not in the local cache" in problem

    begin_run(settings)
    with pytest.raises(ModelOverrideError, match="REF_TEXT_SEG_MODEL_ID"):
        ref_extractor._get_text_segmenter(settings)
    (record,) = [r for r in run_records() if r["setting"] == "REF_TEXT_SEG_MODEL_ID"]
    assert record["loaded"] is False


# -- the error reaches the run, whatever caught it ----------------------------------


def _override_error(settings=None):
    settings = settings or snapshot_settings()
    settings.REF_GEOM_SEG_MODEL_ID = "/nonexistent/geom.joblib"
    error = override_load_error(settings, "REF_GEOM_SEG_MODEL_ID", FileNotFoundError("gone"))
    assert isinstance(error, ModelOverrideError)
    return error


class _Stage:
    requires = None
    produces = None

    def __init__(self, name, run):
        self.name = name
        self._run = run
        self.ran = 0

    async def run(self, ctx):
        self.ran += 1
        await self._run(ctx)


def _pipeline(*stages):
    from bibr.pipeline.context import RunConfig
    from bibr.pipeline.pipeline import Pipeline

    return Pipeline(
        stages=stages, resources=MagicMock(), config=RunConfig(), settings=snapshot_settings()
    )


async def _swallowed(ctx):
    try:
        raise _override_error()
    except ModelOverrideError:
        pass  # a stage that keeps going on any error


async def _on_the_paper(ctx):
    for fs in ctx.file_states:
        error = _override_error()
        fs.set_error(f"Post-parse failed: {error}", code="extraction_failed", exc=error)


async def _in_a_thread(ctx):
    import asyncio

    def load():
        try:
            raise _override_error()
        except ModelOverrideError:
            return None

    await asyncio.to_thread(load)


async def _nothing(ctx):
    return None


@pytest.mark.parametrize("stage", [_swallowed, _on_the_paper, _in_a_thread])
async def test_a_chunk_raises_an_override_failure_whatever_the_stage_did_with_it(tmp_path, stage):
    from bibr.pipeline.state import FileState

    first, after = _Stage("parse_segment", stage), _Stage("export", _nothing)
    with pytest.raises(ModelOverrideError, match="REF_GEOM_SEG_MODEL_ID"):
        await _pipeline(first, after).process_chunk([FileState(path=tmp_path / "a.pdf")])
    assert (first.ran, after.ran) == (1, 0)


async def test_a_chunk_with_an_ordinary_paper_error_does_not_raise(tmp_path):
    from bibr.pipeline.state import FileState

    async def paper_error(ctx):
        for fs in ctx.file_states:
            fs.set_error("bad paper", code="extraction_failed", exc=ValueError("bad"))

    fs = FileState(path=tmp_path / "a.pdf")
    await _pipeline(_Stage("parse_segment", paper_error), _Stage("export", _nothing)).process_chunk(
        [fs]
    )
    assert fs.error_code == "extraction_failed"


def test_find_model_override_error_follows_the_cause_chain():
    from bibr.exceptions import ProcessingError
    from bibr.utils.model_overrides import find_model_override_error

    error = _override_error()
    try:
        try:
            raise error
        except ModelOverrideError as inner:
            raise ProcessingError("wrapped") from inner
    except ProcessingError as outer:
        assert find_model_override_error(outer) is error
    assert find_model_override_error(ValueError("x")) is None
    assert find_model_override_error(None) is None


# -- serve ------------------------------------------------------------------------


def test_serve_answers_an_override_failure_as_model_override_failed_without_paths():
    from bibr.batch.ledger import MODEL_OVERRIDE_FAILED
    from bibr.exceptions import ProcessingError
    from bibr.serve.deployments.pipeline import _ERROR_KIND_TO_HTTP, BibrPipelineAPI

    error = _override_error()
    wrapped = ProcessingError("Post-parse failed")
    wrapped.__cause__ = error
    for exc in (error, wrapped):
        failure = BibrPipelineAPI._translate_error("x.pdf", exc)
        assert failure["error_kind"] == "model_override"
        assert failure["error_code"] == MODEL_OVERRIDE_FAILED
        assert "/nonexistent" not in failure["error"]
        assert failure["outage"] is False
    assert _ERROR_KIND_TO_HTTP["model_override"] == 500


def test_serve_refuses_to_start_on_a_missing_local_override(tmp_path):
    from bibr.serve.app import _check_model_overrides

    settings = snapshot_settings()
    _check_model_overrides(settings)  # the shipped models: no problem, and nothing loads
    settings.REF_GEOM_SEG_MODEL_ID = str(tmp_path / "missing.joblib")
    with pytest.raises(ConfigurationError, match="REF_GEOM_SEG_MODEL_ID"):
        _check_model_overrides(settings)
