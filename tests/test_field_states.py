"""``extraction.fields`` (schema 12.1): one state per tracked field.

The builder turns facts the pipeline already records (final values, each
field's decision, the run's scope, and the codes of the issues and warnings
that explain a missing value) into ``extracted``, ``absent``, ``abstained``,
``failed`` or ``not_attempted``.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bibr.field_states import TRACKED_FIELDS, FieldScope, FieldState, build_field_states
from bibr.processing_warnings import ProcessingWarning, WarningCode
from bibr.validation import IssueSeverity, ValidationIssue

CONFORMANCE = Path(__file__).parent / "fixtures" / "schema_conformance"


def _states(present=(), *, sources=None, scope=FieldScope(), issues=(), warnings=(), doi=False):
    records = build_field_states(
        present={field: field in present for field in TRACKED_FIELDS},
        sources=sources or {},
        scope=scope,
        issues=issues,
        warnings=warnings,
        doi_selected=doi,
    )
    return {field: (str(r.state), r.source, list(r.issues)) for field, r in records.items()}


def _issue(code, *evidence, blocking=False):
    return ValidationIssue(
        code=code,
        severity=IssueSeverity.ERROR if blocking else IssueSeverity.WARNING,
        message="m",
        evidence_ids=tuple(evidence),
        blocking=blocking,
    )


def test_every_state_is_a_schema_value():
    from typing import get_args

    from bibr.export.models import FieldStateLiteral

    assert set(get_args(FieldStateLiteral)) == {str(state) for state in FieldState}


def test_present_values_are_extracted_with_their_source():
    states = _states(
        {"title", "author", "doi", "bib"},
        sources={"title": "layout_title", "author": "llm", "bib": "ner"},
        doi=True,
    )
    assert states["title"] == ("extracted", "layout_title", [])
    assert states["author"] == ("extracted", "llm", [])
    assert states["doi"] == ("extracted", "identity", [])
    assert states["bib"] == ("extracted", "ner", [])
    assert states["abstract"] == ("absent", None, [])


def test_failed_title_call_marks_its_fields_failed():
    failure = _issue(
        "VAL_METADATA_FIELD_FAILED",
        "reason:llm_truncated",
        "field:title",
        "field:abstract",
        "field:keywords",
        blocking=True,
    )
    states = _states({"abstract"}, sources={"abstract": "abstract_section"}, issues=[failure])

    assert states["title"] == ("failed", None, ["VAL_METADATA_FIELD_FAILED"])
    assert states["keywords"] == ("failed", None, ["VAL_METADATA_FIELD_FAILED"])
    # A fallback filled the abstract; the failure stays visible on it.
    assert states["abstract"] == ("extracted", "abstract_section", ["VAL_METADATA_FIELD_FAILED"])
    assert states["author"] == ("absent", None, [])


def test_front_matter_abstention_is_not_a_miss_or_a_failure():
    abstention = _issue("VAL_METADATA_MULTI_ITEM", blocking=True)
    states = _states({"doi", "bib"}, issues=[abstention], doi=True)

    for field in ("title", "author", "abstract", "keywords", "published", "paper_type"):
        assert states[field] == ("abstained", None, ["VAL_METADATA_MULTI_ITEM"]), field
    # DOI and references do not come from the abstained record.
    assert states["doi"][0] == states["bib"][0] == "extracted"
    # A non-blocking copy of the code is not an abstention.
    assert _states(issues=[_issue("VAL_METADATA_MULTI_ITEM")])["title"][0] == "absent"


def test_no_llm_run_did_not_attempt_the_llm_fields():
    states = _states({"title"}, sources={"title": "layout_title"}, scope=FieldScope(no_llm=True))

    assert states["title"] == ("extracted", "layout_title", [])
    for field in ("author", "abstract", "keywords", "published", "journal", "paper_type"):
        assert states[field][0] == "not_attempted", field
    assert states["funding"][0] == "not_attempted"
    # Reference parsing needs the LLM or NER parse of an LLM run.
    assert states["bib"][0] == "not_attempted"
    # Found without an LLM.
    assert states["doi"][0] == states["funding_statement"][0] == "absent"


def test_native_front_matter_was_attempted_without_an_llm():
    states = _states(scope=FieldScope(no_llm=True, native_metadata=True))
    assert states["title"][0] == "absent"
    assert states["funding"][0] == "not_attempted"
    # Reference strings still need a parser; structured citations do not.
    assert states["bib"][0] == "not_attempted"
    native = _states(
        {"bib"}, sources={"bib": "native"}, scope=FieldScope(no_llm=True, native_metadata=True)
    )
    assert native["bib"] == ("extracted", "native", [])


def test_references_off_did_not_attempt_the_bib():
    assert _states(scope=FieldScope(references_off=True))["bib"][0] == "not_attempted"


def test_reference_states():
    incomplete = _issue("VAL_REFERENCES_INCOMPLETE", blocking=True)
    assert _states(issues=[incomplete])["bib"] == ("failed", None, ["VAL_REFERENCES_INCOMPLETE"])
    missing = ProcessingWarning(WarningCode.REF_SECTION_NOT_FOUND, "x")
    assert _states(warnings=[missing])["bib"] == ("absent", None, ["REF_SECTION_NOT_FOUND"])
    inferred = ProcessingWarning(WarningCode.REF_SECTION_INFERRED, "x")
    assert _states({"bib"}, sources={"bib": "llm"}, warnings=[inferred])["bib"] == (
        "extracted",
        "llm",
        ["REF_SECTION_INFERRED"],
    )


def test_ambiguous_doi_is_an_abstention_only_when_unresolved():
    ambiguous = _issue("VAL_DOI_AMBIGUOUS", "text:3", "text:9")
    assert _states(issues=[ambiguous])["doi"] == ("abstained", None, ["VAL_DOI_AMBIGUOUS"])
    assert _states({"doi"}, issues=[ambiguous], doi=True)["doi"] == (
        "extracted",
        "identity",
        ["VAL_DOI_AMBIGUOUS"],
    )


def test_integrity_failure_fails_funding_only_when_there_was_a_statement():
    failed = ProcessingWarning(WarningCode.RESEARCH_INTEGRITY_LLM_FAILED, "llm_timeout: x")
    with_statement = _states(
        {"funding_statement"},
        sources={"funding_statement": "integrity_statement"},
        warnings=[failed],
    )
    assert with_statement["funding"] == ("failed", None, ["RESEARCH_INTEGRITY_LLM_FAILED"])
    assert with_statement["funding_statement"] == ("extracted", "integrity_statement", [])
    assert _states(warnings=[failed])["funding"] == ("absent", None, [])


def test_sources_and_rules_come_from_the_decisions():
    records = build_field_states(
        present={field: field == "funding_statement" for field in TRACKED_FIELDS},
        sources={"funding_statement": "lexical_anchor"},
        scope=FieldScope(references_source="ner"),
        issues=(),
        warnings=(),
        rules={"funding_statement": "integrity_resolution", "title": "none"},
    )
    assert records["funding_statement"].source == "lexical_anchor"
    assert records["funding_statement"].rule == "integrity_resolution"
    assert records["title"].rule == "none"
    # The reference list's source is part of the run's scope; no decision names it.
    assert records["bib"].rule is None
    assert records["doi"].rule is None


def test_author_states():
    failed = ProcessingWarning(WarningCode.AUTHORS_LLM_FAILED, "llm_timeout: x")
    assert _states(warnings=[failed])["author"] == ("failed", None, ["AUTHORS_LLM_FAILED"])
    recovered = _states({"author"}, sources={"author": "llm_recovery"}, warnings=[failed])
    assert recovered["author"] == ("extracted", "llm_recovery", ["AUTHORS_LLM_FAILED"])
    truncated = ProcessingWarning(WarningCode.AUTHORS_TRUNCATED, "llm_truncated: x")
    assert _states({"author"}, sources={"author": "llm"}, warnings=[truncated])["author"] == (
        "extracted",
        "llm",
        ["AUTHORS_TRUNCATED"],
    )


def test_degraded_core_call_fails_the_front_matter():
    degraded = _issue("VAL_CORE_METADATA_DEGRADED", "reason:core_metadata_call_failed")
    states = _states(issues=[degraded])
    assert states["title"] == ("failed", None, ["VAL_CORE_METADATA_DEGRADED"])
    assert states["doi"][0] == "absent"


# ---------------------------------------------------------------------------
# Export and readers
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_paper():
    from tests.export.conftest import _demo_paper, as_parsed

    paper = _demo_paper(with_refs=True)
    as_parsed(paper)
    return paper


def test_export_outside_the_pipeline_has_no_fields(demo_paper):
    from bibr.export.json_export import export_paper_to_json

    payload = export_paper_to_json(demo_paper)
    assert payload["schema_version"] == "12.1"
    assert "fields" not in payload["extraction"]


def test_pipeline_export_carries_valid_fields(demo_paper):
    from bibr.export import PaperExport
    from bibr.export.json_export import export_paper_to_json
    from bibr.extract.field_decisions import FieldCandidate, FieldDecision, FieldDecisions

    demo_paper.field_scope = FieldScope()
    decisions = FieldDecisions()
    title = FieldCandidate("title", "llm", demo_paper.metadata.title)
    decisions.record(FieldDecision("title", title.value, title, "extracted"))
    demo_paper.field_decisions = decisions
    payload = export_paper_to_json(demo_paper)

    fields = payload["extraction"]["fields"]
    assert list(fields) == list(TRACKED_FIELDS)
    assert fields["title"] == {
        "state": "extracted",
        "source": "llm",
        "issues": [],
        "rule": "extracted",
    }
    # A field without a decision carries no rule.
    assert "rule" not in fields["abstract"]
    PaperExport.model_validate(payload)


@pytest.mark.parametrize("version", ["12.0", "12.1"])
def test_readers_accept_old_and_new_minor_exports(version):
    """A 12.0 export has no fields; the reader, the evaluator and
    payload_validation read it exactly as before."""
    from bibr.api import Result
    from bibr.export import PaperExportReader
    from bibr.validation import payload_validation
    from evaluation.evaluate import _front_matter_abstained, extract_comparable_from_json

    new = json.loads((CONFORMANCE / "valid" / "full.json").read_text())
    payload = copy.deepcopy(new)
    if version == "12.0":
        payload["schema_version"] = "12.0"
        del payload["extraction"]["fields"]

    model = PaperExportReader.model_validate(payload)
    assert model.schema_version == version
    assert (model.extraction.fields is None) is (version == "12.0")
    assert Result(payload).title == new["metadata"]["title"]
    assert payload_validation(payload) == new["extraction"]["validation"]
    assert extract_comparable_from_json(payload) == extract_comparable_from_json(new)
    assert _front_matter_abstained(payload) is False


def test_core_checkpoint_replay_accepts_the_current_minor():
    from bibr.export.models import _SCHEMA_VERSION
    from bibr.pipeline.artifacts import CORE_SCHEMA_VERSION

    assert CORE_SCHEMA_VERSION == _SCHEMA_VERSION == "12.1"


# ---------------------------------------------------------------------------
# End to end through the pipeline (smoke fakes)
# ---------------------------------------------------------------------------


async def _smoke_export(tmp_path, monkeypatch, llm=None, **pipeline_kwargs):
    from bibr.config import Settings
    from bibr.local.pipeline import LocalPipeline
    from bibr.pipeline.resources import ResourceManager
    from tests import test_pipeline_smoke as smoke

    monkeypatch.setattr(Settings.ocr, "native_text_min_chars", 4)
    monkeypatch.setattr(Settings.ml, "section_classifier_model_id", None)
    pdf = tmp_path / "smoke.pdf"
    pdf.write_bytes(smoke._build_pdf(smoke._LINES))
    pipeline = LocalPipeline(crossref=False, **pipeline_kwargs)
    rm = ResourceManager(
        memory_mode=pipeline.memory_mode,
        ocr_backend=pipeline.ocr_backend,
        ocr_url=pipeline.ocr_url,
        ocr_model=pipeline.ocr_model,
        device=pipeline.device,
        settings=pipeline.settings,
        layout=smoke.FakeLayout(),
        segmenter=smoke.FakeSegmenter(),
    )
    pipeline._resources = rm
    rm._ocr = smoke.FakeOcr()
    rm._llm_client = llm or smoke._fake_llm_client()
    try:
        return await pipeline.process_file(pdf)
    finally:
        await pipeline.aclose()


def _pairs(result):
    return {field: (r["state"], r["source"]) for field, r in result["extraction"]["fields"].items()}


async def test_llm_run_field_states(tmp_path, monkeypatch):
    result = await _smoke_export(tmp_path, monkeypatch)

    pairs = _pairs(result)
    assert pairs["title"] == ("extracted", "llm")
    assert pairs["author"] == ("extracted", "llm")
    assert pairs["abstract"] == ("extracted", "llm")
    assert pairs["keywords"] == ("extracted", "llm")
    assert pairs["paper_type"] == ("extracted", "llm")
    assert pairs["bib"] == ("extracted", "llm")
    # The smoke paper prints no journal, date, own DOI or funding statement.
    for field in ("journal", "published", "doi", "funding_statement", "funding"):
        assert pairs[field] == ("absent", None), field


async def test_no_llm_run_field_states(tmp_path, monkeypatch):
    result = await _smoke_export(tmp_path, monkeypatch, no_llm=True)

    pairs = _pairs(result)
    assert pairs["title"] == ("extracted", "layout_title")
    for field in ("author", "keywords", "published", "journal", "paper_type", "funding", "bib"):
        assert pairs[field] == ("not_attempted", None), field


async def test_references_off_field_state(tmp_path, monkeypatch):
    result = await _smoke_export(tmp_path, monkeypatch, ref_parse_strategy="off")
    assert _pairs(result)["bib"] == ("not_attempted", None)


async def test_failed_title_call_field_states(tmp_path, monkeypatch):
    from bibr.exceptions import LlmInvalidOutputError
    from tests.pipeline.test_title_failure_containment import _llm_failing_title

    llm = _llm_failing_title(LlmInvalidOutputError("x", cause="1 validation error"))
    result = await _smoke_export(tmp_path, monkeypatch, llm=llm)

    fields = result["extraction"]["fields"]
    # The selected record's title row and the abstract section refilled title
    # and abstract; the failure stays visible on them.
    assert fields["title"] == {
        "state": "extracted",
        "source": "front_matter_candidate",
        "issues": ["VAL_METADATA_FIELD_FAILED"],
        "rule": "selected_record_title",
    }
    assert fields["abstract"]["state"] == "extracted"
    assert fields["abstract"]["source"] == "abstract_section"
    assert fields["abstract"]["rule"] == "abstract_section_fallback"
    # A failed field names the step that failed.
    assert fields["keywords"] == {
        "state": "failed",
        "source": "llm",
        "issues": ["VAL_METADATA_FIELD_FAILED"],
        "rule": "none",
    }
    assert fields["author"]["state"] == fields["bib"]["state"] == "extracted"


def test_field_scope_is_only_set_by_the_pipeline():
    from bibr.paper import Paper

    assert Paper(input_file=SimpleNamespace()).field_scope is None
