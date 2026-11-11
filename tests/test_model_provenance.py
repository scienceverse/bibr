"""``extraction.settings.models``: which models a run was set to load, in the export itself."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import types
from pathlib import Path

import pytest

from bibr.config import snapshot_settings
from bibr.export.models import ExtractionSettingsExport, ModelProvenanceExport
from bibr.run_mode import fast_settings
from bibr.utils import model_provenance
from bibr.utils.model_provenance import export_models, reset_model_provenance

BUNDLES = Path(__file__).parent / "fixtures" / "onnx"


@pytest.fixture(autouse=True)
def _fresh():
    reset_model_provenance()
    yield
    reset_model_provenance()


def _shipped():
    """The settings as shipped: the test session turns some models off and pins torch."""
    from bibr.config import MlOptions

    settings = snapshot_settings()
    for name in (
        "runtime",
        "section_classifier_model_id",
        "paper_classifier_model_id",
        "front_role_model_id",
        "front_tagger_model_id",
    ):
        setattr(settings.ml, name, MlOptions.model_fields[name].default)
    return settings


def _by_setting(rows):
    return {row["setting"]: row for row in rows}


def test_a_default_fast_run_lists_every_model_it_loads_by_hub_id_and_revision():
    rows = export_models(fast_settings(_shipped()))
    assert [row["setting"] for row in rows] == [
        "LAYOUT_ONNX_MODEL_ID",
        "WTPSPLIT_MODEL",
        "NER_PARSER_CKPT",
        "REF_GEOM_SEG_MODEL_ID",
        "ML_FRONT_ROLE_MODEL_ID",
        "ML_FRONT_TAGGER_MODEL_ID",
        "ML_SECTION_CLASSIFIER_MODEL_ID",
        "ML_PAPER_CLASSIFIER_MODEL_ID",
    ]
    settings = _shipped()
    by = _by_setting(rows)
    assert by["NER_PARSER_CKPT"]["id"] == settings.NER_PARSER_CKPT
    assert by["NER_PARSER_CKPT"]["revision"] == settings.NER_PARSER_REVISION
    assert by["LAYOUT_ONNX_MODEL_ID"]["revision"] == settings.layout.onnx_revision
    assert by["WTPSPLIT_MODEL"]["id"] == "segment-any-text/sat-6l-sm"
    assert all(row["override"] is False and row["files"] is None for row in rows)
    for row in rows:
        ModelProvenanceExport(**row)


def test_a_full_run_without_the_front_tagger_does_not_list_it():
    settings = _shipped()
    settings.ml.front_tagger_mode = "off"
    assert "ML_FRONT_TAGGER_MODEL_ID" not in _by_setting(export_models(settings))


def test_a_local_model_is_a_short_name_and_file_checksums_never_a_path(tmp_path):
    bundle = tmp_path / "machine" / "layout" / "section-candidate"
    shutil.copytree(BUNDLES / "section" / "onnx", bundle / "onnx")
    settings = _shipped()
    settings.ml.section_classifier_model_id = str(bundle)
    row = _by_setting(export_models(settings))["ML_SECTION_CLASSIFIER_MODEL_ID"]
    assert re.fullmatch(r"section-candidate@[0-9a-f]{12}", row["id"])
    assert row["revision"] is None and row["override"] is True
    names = {f["name"]: f["sha256"] for f in row["files"]}
    assert "onnx/bibr_onnx.json" in names and "onnx/model.onnx" in names
    graph = bundle / "onnx" / "model.onnx"
    assert names["onnx/model.onnx"] == hashlib.sha256(graph.read_bytes()).hexdigest()
    assert str(tmp_path) not in json.dumps(row)
    ModelProvenanceExport(**row)


def test_a_bundle_directory_named_onnx_takes_its_parent_name(tmp_path):
    bundle = tmp_path / "paper-candidate" / "onnx"
    shutil.copytree(BUNDLES / "paper" / "onnx", bundle)
    settings = _shipped()
    settings.ml.paper_classifier_model_id = str(bundle)
    row = _by_setting(export_models(settings))["ML_PAPER_CLASSIFIER_MODEL_ID"]
    assert row["id"].startswith("paper-candidate@")
    assert {f["name"] for f in row["files"]} >= {"bibr_onnx.json", "model.onnx"}


@pytest.mark.parametrize("layout", ["home", "home/onnx", "user/onnx"])
def test_a_local_model_never_names_the_user_or_the_home_directory(tmp_path, monkeypatch, layout):
    home = tmp_path / "people" / "someone"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(model_provenance.getpass, "getuser", lambda: "someone")
    target = {
        "home": home,
        "home/onnx": home / "onnx",
        "user/onnx": tmp_path / "elsewhere" / "someone" / "onnx",
    }[layout]
    shutil.copytree(BUNDLES / "paper" / "onnx", target)
    settings = _shipped()
    settings.ml.paper_classifier_model_id = str(target)
    row = _by_setting(export_models(settings))["ML_PAPER_CLASSIFIER_MODEL_ID"]
    assert re.fullmatch(r"local-model@[0-9a-f]{12}", row["id"])
    text = json.dumps(row)
    assert "someone" not in text and "people" not in text and str(tmp_path) not in text


@pytest.mark.parametrize(
    ("value", "name"),
    [
        ("/home/other/onnx", "local-model"),
        ("/Users/other", "local-model"),
        ("/root/onnx", "local-model"),
        ("/home/other/models/section/onnx", "section"),
        ("/srv/models/section", "section"),
        # Windows paths, read on any system: the drive is no part of the name.
        ("C:/Users/other/onnx", "local-model"),
        ("C:\\Users\\other\\onnx", "local-model"),
        ("c:\\users\\other", "local-model"),
        ("C:\\Documents and Settings\\other\\onnx", "local-model"),
        ("\\\\server\\share\\Users\\other\\onnx", "local-model"),
        ("/Users/other/onnx", "local-model"),
        ("C:\\Users\\other\\models\\section\\onnx", "section"),
        ("D:/models/section", "section"),
    ],
)
def test_the_home_directory_of_any_user_is_never_a_short_name(value, name):
    assert model_provenance._short_name(Path(value), []).partition("@")[0] == name


def test_two_local_models_of_one_name_differ_by_their_files(tmp_path):
    first, second = tmp_path / "a" / "candidate", tmp_path / "b" / "candidate"
    shutil.copytree(BUNDLES / "paper" / "onnx", first / "onnx")
    shutil.copytree(BUNDLES / "paper" / "onnx", second / "onnx")
    (second / "onnx" / "model.onnx").write_bytes(b"other weights")
    ids = []
    for bundle in (first, first, second):
        reset_model_provenance()
        settings = _shipped()
        settings.ml.paper_classifier_model_id = str(bundle)
        ids.append(_by_setting(export_models(settings))["ML_PAPER_CLASSIFIER_MODEL_ID"]["id"])
    assert ids[0] == ids[1] != ids[2]
    assert all(i.startswith("candidate@") for i in ids)


def test_a_model_directory_is_listed_once_per_process(tmp_path, monkeypatch):
    bundle = tmp_path / "tagger"
    shutil.copytree(BUNDLES / "paper" / "onnx", bundle / "onnx")
    settings = _shipped()
    settings.ml.paper_classifier_model_id = str(bundle)
    listed = []
    real = model_provenance._list_local_files
    monkeypatch.setattr(
        model_provenance,
        "_list_local_files",
        lambda env, value: listed.append(env) or real(env, value),
    )
    for _paper in range(5):
        export_models(settings)
    assert listed.count("ML_PAPER_CLASSIFIER_MODEL_ID") == 1


def test_a_changed_model_file_is_hashed_again(tmp_path):
    import os

    from bibr.utils import model_overrides

    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"one")
    first = model_overrides._sha256(weights)
    assert model_overrides._sha256(weights) == first
    weights.write_bytes(b"two")
    st = weights.stat()
    os.utime(weights, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert model_overrides._sha256(weights) == hashlib.sha256(b"two").hexdigest() != first


def test_files_are_hashed_once_per_process(tmp_path, monkeypatch):
    bundle = tmp_path / "sec"
    shutil.copytree(BUNDLES / "section" / "onnx", bundle / "onnx")
    settings = _shipped()
    settings.ml.section_classifier_model_id = str(bundle)
    calls = []
    from bibr.utils import model_overrides

    real = model_overrides._sha256
    monkeypatch.setattr(model_overrides, "_sha256", lambda p: calls.append(p) or real(p))
    first = export_models(settings)
    hashed = len(calls)
    assert hashed > 0
    assert export_models(settings) == first
    assert len(calls) == hashed


def test_a_branch_revision_is_resolved_through_the_hub_cache(tmp_path, monkeypatch):
    commit = "a" * 40
    refs = tmp_path / "models--org--section" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(commit)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    settings = _shipped()
    settings.ml.section_classifier_model_id = "org/section"
    settings.ml.section_classifier_revision = "main"
    row = _by_setting(export_models(settings))["ML_SECTION_CLASSIFIER_MODEL_ID"]
    assert row["revision"] == commit
    # Not in the cache: the configured revision as given.
    settings.ml.section_classifier_revision = "v2"
    reset_model_provenance()
    row = _by_setting(export_models(settings))["ML_SECTION_CLASSIFIER_MODEL_ID"]
    assert row["revision"] == "v2"


def test_the_crf_is_listed_only_where_torch_can_run_it(monkeypatch):
    settings = _shipped()
    monkeypatch.setattr("bibr.utils.ml_runtime.torch_available", lambda: False)
    assert "NER_SEG_CKPT" not in _by_setting(export_models(settings))
    monkeypatch.setattr("bibr.utils.ml_runtime.torch_available", lambda: True)
    assert "NER_SEG_CKPT" in _by_setting(export_models(settings))


def test_the_text_segmenter_replaces_the_crf_row(monkeypatch):
    monkeypatch.setattr("bibr.utils.ml_runtime.torch_available", lambda: True)
    settings = _shipped()
    settings.REF_TEXT_SEG_MODEL_ID = "org/text-seg"
    settings.REF_TEXT_SEG_REVISION = "b" * 40
    by = _by_setting(export_models(settings))
    assert "NER_SEG_CKPT" not in by
    assert by["REF_TEXT_SEG_MODEL_ID"] == {
        "setting": "REF_TEXT_SEG_MODEL_ID",
        "model": "text reference segmenter",
        "id": "org/text-seg",
        "revision": "b" * 40,
        "files": None,
        "override": True,
    }


def test_an_input_without_layout_and_a_no_llm_run_list_less():
    settings = _shipped()
    assert "LAYOUT_ONNX_MODEL_ID" not in _by_setting(export_models(settings, layout=False))
    rows = export_models(settings, extraction=False)
    assert [row["setting"] for row in rows] == ["LAYOUT_ONNX_MODEL_ID", "WTPSPLIT_MODEL"]


def test_the_torch_runtime_lists_the_torch_layout_checkpoint():
    settings = _shipped()
    settings.ml.runtime = "torch"
    row = export_models(settings)[0]
    assert row["setting"] == "LAYOUT_MODEL_ID"
    assert (row["id"], row["revision"]) == (
        settings.layout.model_id,
        settings.layout.model_revision,
    )


def test_building_the_rows_never_raises(monkeypatch, caplog):
    monkeypatch.setattr(
        model_provenance, "_rows", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    )
    with caplog.at_level("WARNING"):
        assert export_models(_shipped()) is None
    assert "model provenance could not be built" in caplog.text


def test_a_missing_local_model_drops_only_its_row(tmp_path, caplog):
    settings = _shipped()
    settings.REF_GEOM_TAGGER_MODEL_ID = str(tmp_path / "gone")
    with caplog.at_level("WARNING"):
        by = _by_setting(export_models(settings))
    assert "REF_GEOM_TAGGER_MODEL_ID" not in by and "NER_PARSER_CKPT" in by


# -- in the export ------------------------------------------------------------------------


def _ctx(settings, *, no_llm=False, mode="fast"):
    from bibr.ocr.profiles import OcrRuntimeIdentity
    from bibr.pipeline.context import RunConfig

    return types.SimpleNamespace(
        config=RunConfig(no_llm=no_llm, mode=mode),
        settings=settings,
        scratch={
            "ocr_runtime_identity": OcrRuntimeIdentity(
                backend="text-layer", model="pdf-text-layer", profile=None, normalizer_version="x"
            )
        },
    )


def _paper(name="paper.pdf", file_type="pdf", ext=".pdf"):
    from bibr.input.file import InputFile, InputFormat
    from bibr.paper import Paper
    from bibr.paper_contents import PaperContents

    return Paper(
        input_file=InputFile(
            path=Path(name),
            file_hash="abc",
            input_format=InputFormat(
                file_extension=ext, detected_mime_type="application/pdf", file_type=file_type
            ),
        ),
        contents=PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={}),
    )


def test_the_export_carries_the_rows_under_extraction_settings():
    from bibr.pipeline.stages.export import _build_extraction

    settings = fast_settings(_shipped())
    extraction = _build_extraction(_ctx(settings), _paper())
    block = ExtractionSettingsExport(**extraction["settings"])
    dumped = block.model_dump()
    assert [m["setting"] for m in dumped["models"]][:2] == [
        "LAYOUT_ONNX_MODEL_ID",
        "WTPSPLIT_MODEL",
    ]
    assert all("files" not in m for m in dumped["models"])  # omitted for Hub models


def test_the_export_follows_the_run_strategies():
    from bibr.pipeline.stages.export import _build_extraction

    settings = _shipped()
    ctx = _ctx(settings, mode="full")
    ctx.config.ref_parse_strategy = "off"
    extraction = _build_extraction(ctx, _paper())
    settings_rows = {m["setting"] for m in extraction["settings"]["models"]}
    assert "NER_PARSER_CKPT" not in settings_rows
    assert "ML_SECTION_CLASSIFIER_MODEL_ID" in settings_rows
