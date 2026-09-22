"""Tests for the one-call library API (bibr.chew / bibr.achew / Result)."""

import asyncio
import inspect
import json
from unittest.mock import patch

import pytest

import bibr
from bibr.api import Records, Result, _pipeline_kwargs


def _export_fixture() -> dict:
    return {
        "paper_id": "10.1234/example",
        "schema_version": "12.0",
        "source": {
            "file_name": "paper.pdf",
            "file_hash": "abc123",
            "input_format": "pdf",
        },
        "metadata": {
            "title": "A Paper",
            "keywords": [],
            "doi": "10.1234/example",
        },
        "author": [
            {
                "author_id": 1,
                "given": "Jane",
                "family": "Doe",
                "corresponding": False,
            }
        ],
        "text": [],
        "section": [
            {
                "section_id": 1,
                "header": "Intro",
                "level": 1,
                "parent_section_id": 0,
                "section_type": "intro",
            }
        ],
        "url": [],
        "bib": [
            {"bib_id": 1, "title": "Ref One", "doi": "10.1/1"},
            {"bib_id": 2, "title": "Ref Two", "doi": "10.1/2"},
        ],
        "bib_match": [],
        "xref": [],
        "figure": [],
        "table": [],
        "eq": [],
        "extraction": {
            "bibr_version": "0.3.0",
            "completed_at": "2026-07-24T10:00:00Z",
            "ocr": {"backend": "glm-mlx"},
            "llm": {"provider": "google", "model": "some-model"},
            "settings": {
                "ref_seg": "geom",
                "ref_parse": "ner",
                "crossref_enrich": False,
                "consolidate": "off",
            },
            "usage": {
                "totals": {
                    "calls": 1,
                    "input_tokens": 10,
                    "cached_input_tokens": 0,
                    "output_tokens": 2,
                    "total_tokens": 12,
                },
                "breakdown": [
                    {
                        "label": "extract_authors",
                        "provider": "google",
                        "model": "some-model",
                        "calls": 1,
                        "input_tokens": 10,
                        "cached_input_tokens": 0,
                        "output_tokens": 2,
                        "total_tokens": 12,
                    }
                ],
            },
            "warnings": [],
        },
    }


class _StubPipeline:
    """Captures constructor kwargs; fakes process_file/process_chunk."""

    instances: list = []
    memory_mode = "balanced"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.settings = kwargs.get("settings")
        self.closed = False
        self.chunks: list[list] = []
        _StubPipeline.instances.append(self)

    async def process_file(self, path, paper_id=None, progress=None):
        self.path = path
        self.paper_id = paper_id
        self.progress = progress
        return _export_fixture()

    async def process_chunk(self, file_states, progress=None, config=None):
        self.chunks.append([fs.path for fs in file_states])
        for fs in file_states:
            if "bad" in fs.path.name:
                fs.error = "OCR exploded"
                fs.error_code = "ocr_failed"
                fs.failed_stage = "ocr"
            else:
                data = _export_fixture()
                data["paper_id"] = fs.path.stem
                fs.result_json = data

    async def aclose(self):
        self.closed = True


@pytest.fixture
def stub_pipeline():
    _StubPipeline.instances = []
    with patch("bibr.local.pipeline.LocalPipeline", _StubPipeline):
        yield _StubPipeline


# --- LLM preflight ----------------------------------------------------------


def _fail_preflight(monkeypatch):
    from bibr.clients import llm as llm_client_mod

    def boom(settings=None):  # noqa: ARG001
        raise ValueError("GOOGLE_API_KEY is not set")

    monkeypatch.setattr(llm_client_mod, "preflight_credentials", boom)


def test_chew_fails_on_missing_credentials_before_any_model_loads(stub_pipeline, monkeypatch):
    """The CLI checks the key before OCR; the library used to find out after it."""
    import bibr

    _fail_preflight(monkeypatch)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        bibr.chew("paper.pdf")
    assert stub_pipeline.instances == []


def test_chewer_fails_on_missing_credentials_at_construction(stub_pipeline, monkeypatch):
    import bibr

    _fail_preflight(monkeypatch)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        bibr.Chewer()
    assert stub_pipeline.instances == []


def test_chew_skips_the_llm_preflight_with_no_llm(stub_pipeline, monkeypatch):
    import bibr

    _fail_preflight(monkeypatch)
    result = bibr.chew("paper.pdf", no_llm=True)
    assert result.ok
    assert len(stub_pipeline.instances) == 1


def test_chew_preflights_a_managed_local_backend(stub_pipeline, monkeypatch):
    import bibr
    from bibr.exceptions import ConfigurationError
    from bibr.local.cli import run_config

    monkeypatch.setattr(
        run_config,
        "_preflight_local_backend",
        lambda backend: f"{backend} backend selected but no server executable was found.",
    )
    with pytest.raises(ConfigurationError, match="llama-cpp backend selected"):
        bibr.chew("paper.pdf", llm="llama-cpp")
    assert stub_pipeline.instances == []


# --- Records / Result views -------------------------------------------------


def test_records_df():
    records = Records([{"a": 1}, {"a": 2}])
    df = records.df
    assert list(df["a"]) == [1, 2]
    assert len(df) == 2


def test_result_table_keys_and_aliases():
    result = Result(_export_fixture())
    assert isinstance(result.bib, Records)
    assert result.references == result.bib
    assert result.authors == result.author
    assert result.sections == result.section
    assert [r["title"] for r in result.references] == ["Ref One", "Ref Two"]
    assert list(result.references.df["doi"]) == ["10.1/1", "10.1/2"]


def test_result_metadata_source_and_toplevel_passthrough():
    result = Result(_export_fixture())
    assert result.title == "A Paper"
    assert result.doi == "10.1234/example"
    assert result.paper_id == "10.1234/example"
    assert result.extraction["usage"]["totals"]["input_tokens"] == 10
    assert result["metadata"]["title"] == "A Paper"
    assert result.file_hash == "abc123"
    assert result.data is not None


def test_result_exposes_validated_export_model_without_replacing_raw_data():
    from bibr.export import PaperExport

    payload = _export_fixture()
    result = Result(payload)

    assert isinstance(result.model, PaperExport)
    assert result.model.metadata.title == "A Paper"
    assert result.data is payload


def test_result_missing_attribute_raises():
    result = Result(_export_fixture())
    with pytest.raises(AttributeError):
        _ = result.nonexistent_field


def test_result_empty_table_is_empty_records():
    result = Result(_export_fixture())
    assert result.figure == []
    assert isinstance(result.figure, Records)


def test_result_repr_and_dir():
    result = Result(_export_fixture())
    text = repr(result)
    assert "10.1234/example" in text
    assert "2 refs" in text
    listing = dir(result)
    assert "references" in listing
    assert "title" in listing


def test_result_save(tmp_path):
    result = Result(_export_fixture())
    out = result.save(tmp_path / "out.json")
    assert json.loads(out.read_text())["paper_id"] == "10.1234/example"
    compact = result.save(tmp_path / "compact.json", compact=True)
    assert "\n" not in compact.read_text()


# --- forward compatibility within 12.x -----------------------------------------


def _next_minor_export_fixture() -> dict:
    """A valid export as a later 12.x writer might emit it: the next minor
    ``schema_version`` plus fields this bibr does not know, at several depths."""
    data = _export_fixture()
    data["schema_version"] = "12.1"
    data["future_block"] = {"enabled": True, "items": [1, 2]}
    data["metadata"]["subtitle"] = "A sequel"
    data["section"][0]["numbering"] = "1."
    data["bib"][0]["raw"] = "Doe J. Ref One. 2020."
    data["bib_match"] = [
        {
            "bib_id": 1,
            "service": "crossref",
            "author": [{"family": "Doe", "given": "J.", "particle": "van"}],
        }
    ]
    data["text"] = [
        {
            "text": "Some sentence.",
            "text_id": 1,
            "paragraph_id": 1,
            "section_id": 1,
            "page_number": 1,
            "language": "en",
        }
    ]
    data["table"] = [
        {
            "table_id": 1,
            "contents": [["a", "b"], ["1", "2"]],
            "page_number": 2,
            "cell_count": 4,
        }
    ]
    data["extraction"]["float_parts"] = [
        {
            "object_type": "table",
            "object_id": 1,
            "part_index": 1,
            "page_number": 2,
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "rotation": 90,
        }
    ]
    data["extraction"]["settings"]["future_knob"] = "on"
    data["extraction"]["usage"]["totals"]["reasoning_tokens"] = 0
    return data


def test_result_loads_a_newer_minor_export_with_unknown_fields():
    from bibr.export import PaperExport

    result = Result(_next_minor_export_fixture())

    assert result.title == "A Paper"
    assert result.doi == "10.1234/example"
    assert result.file_hash == "abc123"
    assert [r["title"] for r in result.references] == ["Ref One", "Ref Two"]
    assert result.sections[0]["header"] == "Intro"

    model = result.model
    assert isinstance(model, PaperExport)
    assert model.schema_version == "12.1"
    assert model.metadata.title == "A Paper"
    assert model.section[0].section_type == "intro"
    assert model.bib[0].title == "Ref One"
    assert model.bib_match[0].author is not None
    assert model.bib_match[0].author[0].family == "Doe"
    assert model.text[0].text == "Some sentence."
    assert model.table[0].contents == [["a", "b"], ["1", "2"]]
    assert model.extraction is not None
    assert model.extraction.float_parts is not None
    assert model.extraction.float_parts[0].page_number == 2
    assert model.extraction.settings.ref_seg == "geom"
    assert model.extraction.usage is not None
    assert model.extraction.usage.totals.input_tokens == 10


def test_result_keeps_unknown_fields_from_a_newer_minor_export():
    payload = _next_minor_export_fixture()
    result = Result(payload)

    # The raw dict is untouched, and unknown top-level/metadata keys resolve
    # as attributes like known ones.
    assert result.data is payload
    assert result.future_block == {"enabled": True, "items": [1, 2]}
    assert result.subtitle == "A sequel"
    assert result.references[0]["raw"] == "Doe J. Ref One. 2020."

    model = result.model
    assert model.model_extra == {"future_block": {"enabled": True, "items": [1, 2]}}
    assert model.metadata.model_extra == {"subtitle": "A sequel"}
    assert model.section[0].model_extra == {"numbering": "1."}
    assert model.bib[0].model_extra == {"raw": "Doe J. Ref One. 2020."}
    assert model.bib_match[0].author is not None
    assert model.bib_match[0].author[0].model_extra == {"particle": "van"}
    assert model.table[0].model_extra == {"cell_count": 4}
    assert model.extraction is not None
    assert model.extraction.float_parts is not None
    assert model.extraction.float_parts[0].model_extra == {"rotation": 90}
    assert model.extraction.settings.model_extra == {"future_knob": "on"}

    # Re-wrapping the model keeps the unknown fields in the dumped dict.
    rewrapped = Result(model)
    assert rewrapped.future_block == {"enabled": True, "items": [1, 2]}
    assert rewrapped.data["extraction"]["float_parts"][0]["rotation"] == 90


@pytest.mark.parametrize("version", ["13.0", "13.1", "11.0", "12", "12.1.0", "v12.1"])
def test_result_rejects_a_schema_version_outside_12x(version):
    from pydantic import ValidationError

    payload = _next_minor_export_fixture()
    payload["schema_version"] = version
    with pytest.raises(ValidationError, match="schema_version"):
        Result(payload)


def test_result_rejects_an_export_without_schema_version():
    from pydantic import ValidationError

    payload = _next_minor_export_fixture()
    del payload["schema_version"]
    with pytest.raises(ValidationError, match="schema_version"):
        Result(payload)


def test_result_still_rejects_a_known_field_of_the_wrong_type():
    from pydantic import ValidationError

    payload = _next_minor_export_fixture()
    payload["metadata"]["keywords"] = "not a list"
    with pytest.raises(ValidationError, match="keywords"):
        Result(payload)


@pytest.mark.parametrize(
    ("location", "mutate"),
    [
        ("root", lambda d: d.update(future_block={})),
        ("metadata", lambda d: d["metadata"].update(subtitle="A sequel")),
        ("section", lambda d: d["section"][0].update(numbering="1.")),
        ("bib", lambda d: d["bib"][0].update(raw="Doe J.")),
        ("extraction", lambda d: d["extraction"]["settings"].update(future_knob="on")),
        ("schema_version", lambda d: d.update(schema_version="12.1")),
    ],
)
def test_producer_models_still_reject_fields_they_do_not_define(location, mutate):
    """Leniency is for reading only: what bibr writes must match its own schema."""
    from pydantic import ValidationError

    from bibr.export import PaperExport, validate_export

    payload = _export_fixture()
    PaperExport.model_validate(payload)
    mutate(payload)

    with pytest.raises(ValidationError):
        PaperExport.model_validate(payload)
    assert validate_export(payload), location


# --- option mapping ----------------------------------------------------------


def test_pipeline_kwargs_friendly_names():
    kwargs = _pipeline_kwargs({"ocr": "glm-llama", "llm": "cloud", "memory": "aggressive"})
    assert kwargs == {
        "ocr_backend": "glm-llama",
        "llm_backend": "cloud",
        "memory_mode": "aggressive",
    }


def test_chew_ocr_profile_is_forwarded_to_local_pipeline(stub_pipeline, tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4")

    bibr.chew(path, ocr_profile="glm")

    assert stub_pipeline.instances[0].kwargs["ocr_profile"] == "glm"


def test_pipeline_kwargs_crossref_is_tri_state():
    """``crossref`` passes through untouched: True/False force, None defers."""
    assert _pipeline_kwargs({"crossref": True}) == {"crossref": True}
    assert _pipeline_kwargs({"crossref": False}) == {"crossref": False}
    assert _pipeline_kwargs({"crossref": None}) == {"crossref": None}
    assert "crossref" not in _pipeline_kwargs({})


def test_pipeline_kwargs_pages():
    kwargs = _pipeline_kwargs({"pages": "2-5"})
    assert kwargs == {"start_page": 1, "end_page": 4}


def test_pipeline_kwargs_unknown_option_raises():
    with pytest.raises(TypeError, match="unknown chew\\(\\) option 'banana'"):
        _pipeline_kwargs({"banana": True})


# --- chew / achew ------------------------------------------------------------


async def test_achew_returns_result_and_closes(stub_pipeline):
    result = await bibr.achew("paper.pdf", ocr="glm-llama", paper_id="p1")
    assert isinstance(result, Result)
    assert result.title == "A Paper"
    (pipeline,) = stub_pipeline.instances
    assert pipeline.kwargs["ocr_backend"] == "glm-llama"
    assert pipeline.paper_id == "p1"
    assert pipeline.closed


def test_all_public_processing_boundaries_expose_settings():
    for function in (
        bibr.chew,
        bibr.achew,
        bibr.chew_file,
        bibr.achew_file,
        bibr.chew_many,
        bibr.achew_many,
        bibr.Chewer,
    ):
        assert "settings" in inspect.signature(function).parameters


def test_chew_passes_explicit_settings_to_pipeline(stub_pipeline):
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.ocr.model = "public-api-model"

    bibr.chew("paper.pdf", settings=settings)

    (pipeline,) = stub_pipeline.instances
    assert pipeline.settings is settings


async def test_achew_closes_pipeline_on_error(stub_pipeline):
    async def boom(self, path, paper_id=None, progress=None):  # noqa: ARG002
        raise RuntimeError("ocr exploded")

    with patch.object(_StubPipeline, "process_file", boom):
        with pytest.raises(RuntimeError, match="ocr exploded"):
            await bibr.achew("paper.pdf")
    (pipeline,) = stub_pipeline.instances
    assert pipeline.closed


async def test_chew_refuses_running_loop():
    with pytest.raises(RuntimeError, match="achew"):
        bibr.chew("paper.pdf")


def test_chew_sync_end_to_end(stub_pipeline):
    result = bibr.chew("paper.pdf", memory="balanced")
    assert result.references.df.shape[0] == 2
    (pipeline,) = stub_pipeline.instances
    assert pipeline.kwargs["memory_mode"] == "balanced"
    assert pipeline.closed


def test_chew_refs_is_per_run_not_global(stub_pipeline, monkeypatch):
    # refs= must ride the pipeline's RunConfig; mutating the process-global
    # Settings races concurrent Chewers and leaks into later calls.
    import bibr.config

    monkeypatch.setattr(bibr.config.Settings, "REF_PARSE_STRATEGY", "llm")
    bibr.chew("paper.pdf", refs="ner")
    assert bibr.config.Settings.REF_PARSE_STRATEGY == "llm"
    (pipeline,) = stub_pipeline.instances
    assert pipeline.kwargs["ref_parse_strategy"] == "ner"


@pytest.mark.parametrize("refs_value", ["ner", "llm", "llm-chunked"])
def test_chew_refs_accepts_whitelisted_values(stub_pipeline, refs_value):
    bibr.chew("paper.pdf", refs=refs_value)
    (pipeline,) = stub_pipeline.instances
    assert pipeline.kwargs["ref_parse_strategy"] == refs_value


def test_chew_refs_invalid_raises():
    with pytest.raises(ValueError, match="refs must be 'ner', 'llm', 'llm-chunked', or 'off'"):
        bibr.chew("paper.pdf", refs="bogus")


def test_lazy_exports():
    import importlib

    mod = importlib.import_module("bibr")
    for name in ("chew", "achew", "Result", "Records", "ChewFailure", "Chewer"):
        assert callable(getattr(mod, name)) or isinstance(getattr(mod, name), type)


# --- ChewFailure / Result.ok --------------------------------------------------


def test_result_ok_is_true():
    assert Result(_export_fixture()).ok is True


def test_chew_failure_attributes_and_ok():
    from bibr.api import ChewFailure

    f = ChewFailure(path="bad.pdf", error="OCR failed", error_code="ocr_failed", failed_stage="ocr")
    assert f.ok is False
    assert f.path.name == "bad.pdf"
    assert f.error == "OCR failed"
    assert f.error_code == "ocr_failed"
    assert f.failed_stage == "ocr"
    assert "bad.pdf" in repr(f)
    assert "OCR failed" in repr(f)


# --- batch chew() -------------------------------------------------------------


def _touch_pdfs(tmp_path, *names):
    paths = []
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"%PDF-1.4 fake")
        paths.append(p)
    return paths


def test_chew_list_returns_ordered_results(stub_pipeline, tmp_path):
    paths = _touch_pdfs(tmp_path, "b.pdf", "a.pdf")
    results = bibr.chew(paths)
    assert isinstance(results, list)
    assert [r.paper_id for r in results] == ["b", "a"]  # input order, not sorted
    assert len(stub_pipeline.instances) == 1  # models load once
    assert stub_pipeline.instances[0].closed


def test_chew_directory_sorted_supported_only(stub_pipeline, tmp_path):
    _touch_pdfs(tmp_path, "z.pdf", "a.pdf")
    (tmp_path / "notes.txt").write_text("skip me")
    (tmp_path / "sub").mkdir()
    results = bibr.chew(tmp_path)
    assert [r.paper_id for r in results] == ["a", "z"]  # sorted, .txt and subdir skipped


def test_chew_single_file_still_returns_result(stub_pipeline, tmp_path):
    (p,) = _touch_pdfs(tmp_path, "one.pdf")
    result = bibr.chew(p)
    assert isinstance(result, Result)


def test_chew_file_has_unambiguous_result_type(stub_pipeline, tmp_path):
    (path,) = _touch_pdfs(tmp_path, "one.pdf")

    result = bibr.chew_file(path)

    assert isinstance(result, Result)


def test_chew_file_rejects_directory_before_constructing_pipeline(stub_pipeline, tmp_path):
    with pytest.raises(IsADirectoryError, match="requires a file"):
        bibr.chew_file(tmp_path)
    assert stub_pipeline.instances == []


def test_chew_many_has_unambiguous_batch_type(stub_pipeline, tmp_path):
    paths = _touch_pdfs(tmp_path, "a.pdf", "bad.pdf")

    results = bibr.chew_many(paths, batch_size=1)

    assert [result.ok for result in results] == [True, False]


def test_chew_batch_failure_yields_chewfailure_in_place(stub_pipeline, tmp_path):
    from bibr.api import ChewFailure

    paths = _touch_pdfs(tmp_path, "good.pdf", "bad.pdf", "fine.pdf")
    results = bibr.chew(paths)
    assert [type(r) for r in results] == [Result, ChewFailure, Result]
    assert results[1].error == "OCR exploded"
    assert [r.paper_id for r in results if r.ok] == ["good", "fine"]


def test_chew_batch_respects_batch_size(stub_pipeline, tmp_path):
    paths = _touch_pdfs(tmp_path, "a.pdf", "b.pdf", "c.pdf")
    bibr.chew(paths, batch_size=2)
    assert [len(c) for c in stub_pipeline.instances[0].chunks] == [2, 1]


def test_chew_empty_dir_raises(stub_pipeline, tmp_path):
    with pytest.raises(ValueError, match="no supported files"):
        bibr.chew(tmp_path)


def test_chew_empty_list_returns_empty(stub_pipeline):
    assert bibr.chew([]) == []
    assert stub_pipeline.instances == []  # no pipeline constructed


def test_chew_paper_id_with_batch_raises(stub_pipeline, tmp_path):
    paths = _touch_pdfs(tmp_path, "a.pdf")
    with pytest.raises(TypeError, match="paper_id"):
        bibr.chew(paths, paper_id="x")


def test_chewfailure_importable_from_bibr():
    from bibr import ChewFailure  # noqa: F401


# --- consolidate --------------------------------------------------------------


def test_chew_consolidate_true_maps_to_fill(stub_pipeline, tmp_path):
    p = tmp_path / "one.pdf"
    p.write_bytes(b"%PDF-1.4")
    bibr.chew(p, consolidate=True)
    assert stub_pipeline.instances[0].kwargs["consolidate"] == "fill"


def test_chew_consolidate_replace_passthrough(stub_pipeline, tmp_path):
    p = tmp_path / "one.pdf"
    p.write_bytes(b"%PDF-1.4")
    bibr.chew(p, consolidate="replace")
    assert stub_pipeline.instances[0].kwargs["consolidate"] == "replace"


def test_chew_consolidate_invalid_raises():
    with pytest.raises(ValueError, match="consolidate"):
        bibr.chew("paper.pdf", consolidate="merge")


def test_result_consolidate_returns_new_result():
    data = _export_fixture()
    data["bib_match"] = [{"bib_id": 1, "service": "crossref", "container": "J. Test"}]
    result = Result(data)
    enriched = result.consolidate()
    assert enriched is not result
    assert enriched.bib[0]["container"] == "J. Test"
    assert enriched.extraction["diagnostics"]["consolidation"] == [
        {"bib_id": 1, "fields": ["container"]}
    ]
    assert "container" not in result.bib[0]  # original untouched
    assert "consolidation" not in (result.extraction.get("diagnostics") or {})


def test_chew_consolidate_false_forces_off(stub_pipeline, tmp_path):
    p = tmp_path / "one.pdf"
    p.write_bytes(b"%PDF-1.4")
    bibr.chew(p, consolidate=False)
    assert stub_pipeline.instances[0].kwargs["consolidate"] == "off"


def test_chew_consolidate_none_defers_to_settings(stub_pipeline, tmp_path):
    p = tmp_path / "one.pdf"
    p.write_bytes(b"%PDF-1.4")
    bibr.chew(p, consolidate=None)
    assert "consolidate" not in stub_pipeline.instances[0].kwargs


# --- Chewer (warm-pipeline session) --------------------------------------------


def test_chewer_reuses_one_pipeline_across_calls(stub_pipeline, tmp_path):
    a, b = _touch_pdfs(tmp_path, "a.pdf", "b.pdf")
    with bibr.Chewer(ocr="glm-llama") as chewer:
        r1 = chewer.chew(a)
        r2 = chewer.chew(b)
    assert isinstance(r1, Result)
    assert isinstance(r2, Result)
    (pipeline,) = stub_pipeline.instances  # models load once across calls
    assert pipeline.kwargs["ocr_backend"] == "glm-llama"
    assert pipeline.closed


def test_chewer_single_and_batch_semantics_match_chew(stub_pipeline, tmp_path):
    from bibr.api import ChewFailure

    good, bad = _touch_pdfs(tmp_path, "good.pdf", "bad.pdf")
    with bibr.Chewer() as chewer:
        single = chewer.chew(good)
        batch = chewer.chew([good, bad])
    assert isinstance(single, Result)
    assert [type(r) for r in batch] == [Result, ChewFailure]
    assert len(stub_pipeline.instances) == 1


def test_chewer_explicit_single_and_batch_methods(stub_pipeline, tmp_path):
    good, bad = _touch_pdfs(tmp_path, "good.pdf", "bad.pdf")
    with bibr.Chewer() as chewer:
        single = chewer.chew_file(good)
        batch = chewer.chew_many([good, bad])

    assert isinstance(single, Result)
    assert [result.ok for result in batch] == [True, False]


def test_chewer_closed_raises(stub_pipeline, tmp_path):
    (p,) = _touch_pdfs(tmp_path, "one.pdf")
    chewer = bibr.Chewer()
    with chewer:
        chewer.chew(p)
    with pytest.raises(RuntimeError, match="closed"):
        chewer.chew(p)


def test_chewer_close_idempotent_and_lazy(stub_pipeline):
    chewer = bibr.Chewer()
    chewer.close()
    chewer.close()
    assert stub_pipeline.instances == []  # never chewed → no pipeline constructed


def test_chewer_invalid_option_raises_eagerly(stub_pipeline):
    with pytest.raises(TypeError, match="banana"):
        bibr.Chewer(banana=True)
    assert stub_pipeline.instances == []


async def test_chewer_async_context(stub_pipeline, tmp_path):
    a, b = _touch_pdfs(tmp_path, "a.pdf", "b.pdf")
    async with bibr.Chewer() as chewer:
        r1 = await chewer.achew(a)
        r2 = await chewer.achew(b)
    assert r1.ok
    assert r2.ok
    (pipeline,) = stub_pipeline.instances
    assert pipeline.closed


async def test_two_chewers_keep_distinct_settings_snapshots(stub_pipeline, monkeypatch):
    from bibr.config import GlobalSettings, Settings

    first_source = GlobalSettings()
    second_source = GlobalSettings()
    first_source.ocr.model = "first-model"
    first_source.resolver.url = "http://first"
    second_source.ocr.model = "second-model"
    second_source.resolver.url = "http://second"

    first = bibr.Chewer(settings=first_source, refs="off")
    second = bibr.Chewer(settings=second_source, refs="off")

    first_source.ocr.model = "mutated-source"
    second_source.resolver.url = "http://mutated-source"
    monkeypatch.setattr(Settings.ocr, "model", "mutated-global")
    monkeypatch.setattr(Settings.resolver, "url", "http://mutated-global")

    try:
        await asyncio.gather(first.achew("first.pdf"), second.achew("second.pdf"))
        pipelines = {pipeline.settings.ocr.model: pipeline for pipeline in stub_pipeline.instances}
        assert pipelines["first-model"].settings.resolver.url == "http://first"
        assert pipelines["second-model"].settings.resolver.url == "http://second"
        assert pipelines["first-model"].settings is not first_source
        assert pipelines["second-model"].settings is not second_source
    finally:
        await asyncio.gather(first.aclose(), second.aclose())


async def test_chewer_sync_chew_refuses_running_loop(stub_pipeline):
    chewer = bibr.Chewer()
    with pytest.raises(RuntimeError, match="achew"):
        chewer.chew("paper.pdf")
    await chewer.aclose()


def test_chewer_paper_id_with_batch_raises(stub_pipeline, tmp_path):
    paths = _touch_pdfs(tmp_path, "a.pdf")
    with bibr.Chewer() as chewer, pytest.raises(TypeError, match="paper_id"):
        chewer.chew(paths, paper_id="x")


def test_chewer_refs_is_per_run_not_global(stub_pipeline, monkeypatch, tmp_path):
    # Two Chewers with different refs must not clobber each other through
    # process-global Settings; each carries its strategy in pipeline kwargs.
    import bibr.config

    monkeypatch.setattr(bibr.config.Settings, "REF_PARSE_STRATEGY", "llm")
    (p,) = _touch_pdfs(tmp_path, "one.pdf")
    chewer = bibr.Chewer(refs="ner")
    chewer.chew(p)
    chewer.close()
    assert bibr.config.Settings.REF_PARSE_STRATEGY == "llm"
    (pipeline,) = stub_pipeline.instances
    assert pipeline.kwargs["ref_parse_strategy"] == "ner"


def test_chewer_repr_states(stub_pipeline, tmp_path):
    (p,) = _touch_pdfs(tmp_path, "one.pdf")
    chewer = bibr.Chewer()
    assert "cold" in repr(chewer)
    chewer.chew(p)
    assert "warm" in repr(chewer)
    chewer.close()
    assert "closed" in repr(chewer)


def test_chew_refs_off_rides_run_config(stub_pipeline):
    bibr.chew("paper.pdf", refs="off")
    (pipeline,) = stub_pipeline.instances
    assert pipeline.kwargs["ref_parse_strategy"] == "off"


def test_chew_refs_false_means_off(stub_pipeline):
    bibr.chew("paper.pdf", refs=False)
    (pipeline,) = stub_pipeline.instances
    assert pipeline.kwargs["ref_parse_strategy"] == "off"


def test_chew_ref_seg_rides_run_config(stub_pipeline, monkeypatch):
    # ref_seg= must ride RunConfig like refs= — never the process-global Settings.
    import bibr.config

    monkeypatch.setattr(bibr.config.Settings, "REF_SEG_STRATEGY", "geom")
    bibr.chew("paper.pdf", ref_seg="region")
    assert bibr.config.Settings.REF_SEG_STRATEGY == "geom"
    (pipeline,) = stub_pipeline.instances
    assert pipeline.kwargs["ref_seg_strategy"] == "region"


def test_chew_ref_seg_invalid_raises():
    with pytest.raises(ValueError, match="ref_seg must be"):
        bibr.chew("paper.pdf", ref_seg="bogus")
