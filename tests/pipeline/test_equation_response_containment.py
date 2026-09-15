"""Optional equation response failures retain independent, source-backed output."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bibr.clients.structured import StructuredResponseError
from bibr.config import GlobalSettings
from bibr.exceptions import ProcessingError
from bibr.export.models import PaperExport
from bibr.extract.equation_extractor import EquationExtractor
from bibr.models import PaperAuthor, PaperMetadata, PaperReference
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
)
from bibr.pipeline.context import RunConfig
from bibr.pipeline.stages.export import build_result_payload
from bibr.pipeline.stages.post_parse import _extract_metadata_and_equations, post_parse
from bibr.pipeline.state import FileState
from bibr.validation import ValidationIssue


def equation_fixture():
    sections = [
        PaperSection(1, "Abstract", 1, None, CanonicalSection.ABSTRACT),
        PaperSection(2, "Results", 1, None, CanonicalSection.RESULTS),
        PaperSection(3, "References", 1, None, CanonicalSection.REFERENCES),
    ]
    abstract = "Plant growth was measured under two controlled conditions."
    sentences = [
        PaperSentence(1, abstract, 1, 1, page_number=1),
        PaperSentence(2, "The contrast was significant (t(28) = 3.42, p < .001).", 2, 2),
        PaperSentence(3, "A different measure (12 versus 17) was reported.", 2, 3),
        PaperSentence(
            4,
            "$$x = 2$$",
            2,
            4,
            page_number=2,
            is_display_formula=True,
            provenance=[Provenance(page_no=2, bbox=(100, 200, 400, 250))],
        ),
    ]
    refs = []
    for number in range(1, 3):
        sentences.append(
            PaperSentence(100 + number, f"Quill M. Printed study {number}. 2020.", 3, 100 + number)
        )
        refs.append(
            PaperReference(
                bib_id=number,
                title=f"Printed study {number}",
                authors="Quill M",
                year=2020,
                text_id=100 + number,
                first_page=None,
                volume=None,
                container=None,
            )
        )
    metadata = PaperMetadata(
        doi="10.9999/printed",
        title="A printed study of plant growth",
        abstract=abstract,
        authors=[PaperAuthor(author_id=1, given="Mara", family="Quill", affiliation="")],
        references=refs,
    )
    contents = PaperContents(
        sentences, sections, [], [], {}, preparsed_metadata=metadata, native_references=refs
    )
    settings = GlobalSettings()
    settings.EQUATION_EXTRACTION = True
    settings.EQUATION_LLM_FALLBACK_MIN_REGEX_STATS = 0
    return contents, metadata, settings


@pytest.mark.parametrize(
    "category",
    ["empty", "non_json", "non_object", "truncated", "trailing_content", "schema_invalid"],
)
async def test_invalid_optional_response_keeps_metadata_and_source_equations(
    monkeypatch, category, caplog
):
    contents, metadata, settings = equation_fixture()
    # Exercise the non-native MetadataExtractor branch with an already validated
    # independent result. The real regex extractor and LLM fallback still run.
    contents.preparsed_metadata = None
    existing_issue = ValidationIssue("VAL_EXISTING", "warning", "Existing metadata diagnostic")
    extractor = MagicMock(
        validation_issues=[existing_issue],
        extract_all_metadata=AsyncMock(return_value=metadata),
    )
    monkeypatch.setattr("bibr.extract.extractor.MetadataExtractor", lambda *a, **kw: extractor)
    expected = EquationExtractor().extract_from_sentences(contents.sentences, contents.sections)
    formula = contents.sentences[3]
    provenance = list(formula.provenance)
    error = StructuredResponseError(category, last_completion="RAW-PROVIDER-SECRET")
    error.category = "MUTABLE-ATTRIBUTE-SECRET"
    client = SimpleNamespace(extract_equations=AsyncMock(side_effect=error))
    issues = []

    result = await _extract_metadata_and_equations(
        contents, "hash", False, client, settings=settings, validation_issue_sink=issues
    )

    assert result is metadata
    assert len(result.references) == 2
    assert contents.equations == expected
    assert {equation.text_id for equation in contents.equations} == {2, 4}
    assert formula.text == "$$x = 2$$"
    assert formula.provenance == provenance
    client.extract_equations.assert_awaited_once()
    assert client.extract_equations.await_args.args[0] == [(3, contents.sentences[2].text)]
    assert existing_issue in issues
    issue = next(row for row in issues if row.code == "VAL_EQUATION_LLM_RESPONSE_INVALID")
    assert issue.severity == "warning" and not issue.blocking
    assert issue.evidence_ids == ("field:equations", f"reason:llm_response_invalid:{category}")
    assert issue.origin_stage == "post_parse" and issue.count == 1
    assert contents.processing_warnings == [
        f"EQUATION_LLM_RESPONSE_INVALID:{category}: kept regex-only equation extraction"
    ]
    diagnostics = repr(issues) + repr(contents.processing_warnings) + caplog.text
    assert "RAW-PROVIDER-SECRET" not in diagnostics
    assert "MUTABLE-ATTRIBUTE-SECRET" not in diagnostics


async def test_real_post_parse_export_retains_native_metadata_refs_and_formula(monkeypatch):
    contents, metadata, settings = equation_fixture()
    client = SimpleNamespace(
        extract_equations=AsyncMock(
            side_effect=StructuredResponseError("non_json", last_completion="RAW-PROVIDER-SECRET")
        ),
        resolve_citations=AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        "bibr.pipeline.stages.post_parse._classify_sections", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "bibr.structure.implicit_sections.detect_implicit_sections", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "bibr.extract.research_integrity.extract_structured_integrity", AsyncMock(return_value=None)
    )
    paper = await post_parse(
        contents=contents,
        file_name="printed.xml",
        file_hash="a" * 64,
        llm_client=client,
        settings=settings,
    )
    assert paper.metadata is metadata
    state = FileState(path=Path("printed.xml"))
    state.paper = paper
    context = SimpleNamespace(
        config=RunConfig(crossref=False, consolidate="off"), settings=settings, scratch={}
    )
    exported = PaperExport.model_validate(await build_result_payload(context, state))

    assert exported.metadata.title == metadata.title
    assert exported.metadata.doi == metadata.doi
    assert [(row.given, row.family) for row in exported.author] == [("Mara", "Quill")]
    assert [row.title for row in exported.bib] == ["Printed study 1", "Printed study 2"]
    assert {equation.text_id for equation in exported.eq} == {2, 4}
    assert not any(equation.text_id == 3 for equation in exported.eq)
    formula = next(row for row in exported.text if row.text_id == 4)
    assert formula.formatted == "$$x = 2$$" and formula.page_number == 2
    assert contents.sentences[3].provenance == [Provenance(page_no=2, bbox=(100, 200, 400, 250))]
    assert exported.validation is not None and exported.validation.promotable
    issue = next(
        row for row in exported.validation.issues if row.code == "VAL_EQUATION_LLM_RESPONSE_INVALID"
    )
    assert issue.severity == "warning" and not issue.blocking
    assert "RAW-PROVIDER-SECRET" not in exported.model_dump_json()


@pytest.mark.parametrize("error_code", [None, "llm_invalid_output", "upstream_timeout"])
async def test_generic_typed_equation_failures_still_propagate(error_code):
    contents, _, settings = equation_fixture()
    error = ProcessingError("Typed operational failure", error_code=error_code)
    client = SimpleNamespace(extract_equations=AsyncMock(side_effect=error))
    issues = []
    with pytest.raises(ProcessingError) as raised:
        await _extract_metadata_and_equations(
            contents, "hash", False, client, settings=settings, validation_issue_sink=issues
        )
    assert raised.value is error
    assert issues == [] and contents.processing_warnings == []


async def test_response_error_without_safe_diagnostics_is_not_contained():
    contents, _, settings = equation_fixture()
    error = StructuredResponseError("non_json")
    error.safe_diagnostics = None
    client = SimpleNamespace(extract_equations=AsyncMock(side_effect=error))
    with pytest.raises(StructuredResponseError) as raised:
        await _extract_metadata_and_equations(contents, "hash", False, client, settings=settings)
    assert raised.value is error


@pytest.mark.parametrize("kind", ["ordinary", "typed", "structured", "cancelled"])
async def test_metadata_failure_is_not_hidden_by_optional_invalid_response(monkeypatch, kind):
    contents, _, settings = equation_fixture()
    errors = {
        "ordinary": RuntimeError("Core metadata failed"),
        "typed": ProcessingError("Core metadata failed"),
        "structured": StructuredResponseError("schema_invalid"),
        "cancelled": asyncio.CancelledError(),
    }
    error = errors[kind]
    contents.preparsed_metadata = None
    extractor = MagicMock(validation_issues=[], extract_all_metadata=AsyncMock(side_effect=error))
    monkeypatch.setattr("bibr.extract.extractor.MetadataExtractor", lambda *a, **kw: extractor)
    client = SimpleNamespace(
        extract_equations=AsyncMock(side_effect=StructuredResponseError("non_json"))
    )
    with pytest.raises(type(error)):
        await _extract_metadata_and_equations(contents, "hash", False, client, settings=settings)
    assert contents.processing_warnings == []


async def test_child_equation_cancellation_propagates_without_warning():
    contents, _, settings = equation_fixture()
    client = SimpleNamespace(extract_equations=AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await _extract_metadata_and_equations(contents, "hash", False, client, settings=settings)
    assert contents.processing_warnings == []


async def test_parent_cancellation_reaches_equation_task():
    contents, _, settings = equation_fixture()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def extract(*args, **kwargs):
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    client = SimpleNamespace(extract_equations=extract)
    task = asyncio.create_task(
        _extract_metadata_and_equations(contents, "hash", False, client, settings=settings)
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert contents.processing_warnings == []


async def test_successful_empty_optional_response_does_not_warn():
    contents, metadata, settings = equation_fixture()
    expected = EquationExtractor().extract_from_sentences(contents.sentences, contents.sections)
    client = SimpleNamespace(extract_equations=AsyncMock(return_value=[]))
    issues = []
    result = await _extract_metadata_and_equations(
        contents, "hash", False, client, settings=settings, validation_issue_sink=issues
    )
    assert result is metadata and contents.equations == expected
    assert issues == [] and contents.processing_warnings == []
