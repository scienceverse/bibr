"""Typed invalid-output failures must never enter optional degradation paths."""

from types import SimpleNamespace
from unittest import mock

import pytest

from bibr.config import GlobalSettings
from bibr.exceptions import ProcessingError
from bibr.extract.equation_extractor import EquationExtractor
from bibr.extract.ref_extractor import ReferenceExtractor
from bibr.extract.research_integrity import extract_structured_integrity
from bibr.models import PaperAuthor, PaperMetadata
from bibr.paper import PaperReference
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
)
from bibr.pipeline.stages.post_parse import (
    _extract_metadata_and_equations,
    _resolve_preparsed_references,
)
from bibr.structure.citation_linker import _resolve_with_llm
from bibr.structure.implicit_sections import _detect_via_llm
from bibr.structure.section_classifier import _classify_llm_batch


@pytest.fixture
def invalid_output_error() -> ProcessingError:
    return ProcessingError(
        "LLM returned invalid structured output",
        error_code="llm_invalid_output",
    )


async def test_section_classifier_preserves_processing_error_identity(invalid_output_error):
    client = mock.MagicMock()
    client.invoke_structured = mock.AsyncMock(side_effect=invalid_output_error)
    client.close = mock.AsyncMock()

    with pytest.raises(ProcessingError) as raised:
        await _classify_llm_batch(
            ["unfamiliar section"],
            llm_client=client,
            settings=GlobalSettings(),
        )

    assert raised.value is invalid_output_error


async def test_implicit_section_detector_preserves_processing_error_identity(
    invalid_output_error,
):
    client = mock.MagicMock()
    client._cap_input = lambda text: text
    client.invoke_structured = mock.AsyncMock(side_effect=invalid_output_error)
    front_matter = [
        PaperSentence(text_id=1, text="Summary.", section_id=1, paragraph_id=1),
        PaperSentence(text_id=2, text="Background.", section_id=1, paragraph_id=2),
    ]

    with pytest.raises(ProcessingError) as raised:
        await _detect_via_llm(
            client,
            front_matter,
            "hash",
            settings=GlobalSettings(),
        )

    assert raised.value is invalid_output_error


async def test_research_integrity_preserves_processing_error_identity(
    invalid_output_error,
):
    contents = PaperContents(
        sentences=[
            PaperSentence(
                text_id=1,
                text="Funded by Example Council.",
                section_id=1,
                paragraph_id=1,
            )
        ],
        sections=[
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Funding",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.FUNDING,
            ),
        ],
        tables=[],
        links=[],
        sections_text={},
    )
    metadata = PaperMetadata(
        doi="",
        title="Title",
        funding_statement="Funded by Example Council.",
        authors=[
            PaperAuthor(
                author_id=1,
                given="Ada",
                family="Lovelace",
                affiliation="",
            )
        ],
    )
    client = mock.MagicMock()
    client.extract_research_integrity = mock.AsyncMock(side_effect=invalid_output_error)

    with pytest.raises(ProcessingError) as raised:
        await extract_structured_integrity(contents, metadata, client, "hash")

    assert raised.value is invalid_output_error


async def test_citation_resolution_preserves_processing_error_identity(
    invalid_output_error,
):
    client = mock.MagicMock()
    client.resolve_citations = mock.AsyncMock(side_effect=invalid_output_error)
    references = [
        PaperReference(
            bib_id=1,
            title="Reference",
            authors="Lovelace, A.",
            year=1843,
            first_page=None,
            volume=None,
            container=None,
        )
    ]

    with pytest.raises(ProcessingError) as raised:
        await _resolve_with_llm([(1, "Lovelace (1843)")], references, client, "hash")

    assert raised.value is invalid_output_error


async def test_equation_fallback_preserves_processing_error_identity(
    invalid_output_error,
):
    sections = [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        PaperSection(
            section_id=1,
            header="Results",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.RESULTS,
        ),
    ]
    sentences = [
        PaperSentence(
            text_id=1,
            text="An unusual statistic (values 12 versus 17) was observed.",
            section_id=1,
            paragraph_id=1,
        )
    ]
    client = mock.MagicMock()
    client.extract_equations = mock.AsyncMock(side_effect=invalid_output_error)

    with pytest.raises(ProcessingError) as raised:
        await EquationExtractor().extract_with_llm_fallback(
            sentences,
            sections,
            client,
            min_regex_stats=0,
        )

    assert raised.value is invalid_output_error


def _reference_extractor(client) -> ReferenceExtractor:
    contents = SimpleNamespace(
        native_ref_strings=None,
        region_summaries=[],
        ref_line_geometry=[],
        processing_warnings=[],
    )
    return ReferenceExtractor(
        contents,
        file_hash="hash",
        llm_client=client,
        settings=GlobalSettings(),
    )


async def test_reference_segmentation_does_not_fall_back_on_processing_error(
    invalid_output_error,
):
    client = mock.MagicMock()
    client.segment_references = mock.AsyncMock(side_effect=invalid_output_error)
    extractor = _reference_extractor(client)
    extractor._crf_segment_or_recover = mock.Mock(
        side_effect=AssertionError("typed failure must not reach CRF")
    )

    with pytest.raises(ProcessingError) as raised:
        await extractor._segment_llm_then_crf("Reference text", try_region=False)

    assert raised.value is invalid_output_error
    extractor._crf_segment_or_recover.assert_not_called()


@pytest.mark.parametrize(
    "method_name",
    ["_parse_references_llm", "_parse_references_llm_chunked"],
)
async def test_reference_gathered_parse_does_not_fall_back_to_ner(
    invalid_output_error,
    method_name,
):
    client = mock.MagicMock()
    client.extract_references = mock.AsyncMock(side_effect=invalid_output_error)
    client.extract_references_chunk = mock.AsyncMock(side_effect=invalid_output_error)
    extractor = _reference_extractor(client)
    extractor._parse_references_ner = mock.Mock(
        side_effect=AssertionError("typed failure must not reach NER")
    )

    with pytest.raises(ProcessingError) as raised:
        await getattr(extractor, method_name)("Reference text", ["Reference text"])

    assert raised.value is invalid_output_error
    extractor._parse_references_ner.assert_not_called()


async def test_reference_split_retry_preserves_processing_error_identity(
    invalid_output_error,
):
    client = mock.MagicMock()
    client.extract_references = mock.AsyncMock(side_effect=invalid_output_error)
    extractor = _reference_extractor(client)

    with pytest.raises(ProcessingError) as raised:
        await extractor._reparse_split(["Reference text"], 1, depth=1)

    assert raised.value is invalid_output_error


async def test_preparsed_reference_path_does_not_mark_typed_failure_incomplete(
    invalid_output_error,
):
    import pandas as pd

    contents = SimpleNamespace(
        native_references=None,
        native_ref_strings=["Reference text"],
    )
    metadata = PaperMetadata(doi="", title="Title")
    extractor = mock.MagicMock()
    extractor._collect_reference_rows.return_value = pd.DataFrame({"text": ["Reference text"]})
    extractor._extract_references = mock.AsyncMock(side_effect=invalid_output_error)

    with mock.patch(
        "bibr.extract.extractor.MetadataExtractor",
        return_value=extractor,
    ):
        with pytest.raises(ProcessingError) as raised:
            await _resolve_preparsed_references(
                contents,
                metadata,
                "hash",
                mock.MagicMock(),
                "native",
                "llm",
                settings=GlobalSettings(),
            )

    assert raised.value is invalid_output_error
    assert metadata.references_incomplete is not True


async def test_metadata_equation_gather_preserves_typed_equation_failure(
    invalid_output_error,
):
    metadata = PaperMetadata(doi="", title="Title")
    contents = SimpleNamespace(
        preparsed_metadata=metadata,
        native_references=[],
        native_ref_strings=None,
        equations=[],
        sentences=[
            PaperSentence(
                text_id=1,
                text="An unusual statistic (12 versus 17).",
                section_id=1,
                paragraph_id=1,
            )
        ],
        sections=[],
    )
    settings = GlobalSettings()
    settings.EQUATION_EXTRACTION = True

    with mock.patch(
        "bibr.extract.equation_extractor.EquationExtractor.extract_with_llm_fallback",
        mock.AsyncMock(side_effect=invalid_output_error),
    ):
        with pytest.raises(ProcessingError) as raised:
            await _extract_metadata_and_equations(
                contents,
                "hash",
                False,
                mock.MagicMock(),
                settings=settings,
            )

    assert raised.value is invalid_output_error


async def test_typed_equation_failure_wins_over_ordinary_metadata_failure(
    invalid_output_error,
):
    contents = SimpleNamespace(
        preparsed_metadata=None,
        equations=[],
        sentences=[],
        sections=[],
    )
    extractor = mock.MagicMock()
    extractor.validation_issues = []
    extractor.extract_all_metadata = mock.AsyncMock(
        side_effect=RuntimeError("ordinary metadata failure")
    )
    settings = GlobalSettings()
    settings.EQUATION_EXTRACTION = True

    with (
        mock.patch(
            "bibr.extract.extractor.MetadataExtractor",
            return_value=extractor,
        ),
        mock.patch(
            "bibr.extract.equation_extractor.EquationExtractor.extract_with_llm_fallback",
            mock.AsyncMock(side_effect=invalid_output_error),
        ),
    ):
        with pytest.raises(ProcessingError) as raised:
            await _extract_metadata_and_equations(
                contents,
                "hash",
                False,
                mock.MagicMock(),
                settings=settings,
            )

    assert raised.value is invalid_output_error
