"""EQUATION_LLM_FALLBACK=false keeps the regex equations and sends nothing to the LLM."""

from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from bibr.config import GlobalSettings
from bibr.models import PaperMetadata
from bibr.paper_contents import CanonicalSection, PaperSection, PaperSentence


def _contents():
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
        # The regex pass reads this one.
        PaperSentence(
            text_id=10,
            text="Accuracy was higher after training, t(28) = 2.10, p = .045.",
            section_id=1,
            paragraph_id=1,
            page_number=None,
        ),
        # The regex pass misses this one, so only the LLM fallback would see it.
        PaperSentence(
            text_id=11,
            text="Weird stat layout (t: 3.42, p: .003) regex misses.",
            section_id=1,
            paragraph_id=2,
            page_number=None,
        ),
    ]
    return SimpleNamespace(
        sentences=sentences,
        sections=sections,
        preparsed_metadata=None,
        processing_warnings=[],
        equations=None,
    )


@pytest.mark.parametrize("fallback", [True, False])
async def test_equation_llm_fallback_setting(fallback):
    from bibr.pipeline.stages.post_parse import _extract_metadata_and_equations

    contents = _contents()
    client = SimpleNamespace(extract_equations=AsyncMock(return_value=[]))
    metadata = PaperMetadata(doi="", title="T")
    extractor = SimpleNamespace(
        extract_all_metadata=AsyncMock(return_value=metadata), validation_issues=[]
    )
    settings = GlobalSettings()
    settings.EQUATION_LLM_FALLBACK = fallback

    with mock.patch("bibr.extract.extractor.MetadataExtractor", return_value=extractor):
        result = await _extract_metadata_and_equations(
            contents, "h", False, client, settings=settings
        )

    assert result is metadata
    assert client.extract_equations.await_count == (1 if fallback else 0)
    # The regex equations stay either way.
    assert {eq.text_id for eq in contents.equations} == {10}
    assert contents.processing_warnings == []
