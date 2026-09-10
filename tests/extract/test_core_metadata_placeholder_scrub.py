"""Integration: omitted/placeholder title and author fields must
degrade-not-crash through CoreMetadataExtractor.

The schema scrub maps explicitly emitted placeholders to "". An omitted LLM
title remains ``None`` because layout detection supplies the authoritative
title later. These tests exercise the full extraction path with a mocked LLM
client to prove both cases end to end.
"""

from unittest import mock

import pandas as pd

from bibr.extract.extractor import MetadataExtractor
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection
from bibr.schemas import AuthorLLM, CoreMetadataLLM


def _build_extractor(llm_metadata: CoreMetadataLLM) -> MetadataExtractor:
    df = pd.DataFrame(
        {
            "section_name": ["Abstract", "1. Introduction"],
            "text": ["Some abstract text.", "Some intro text."],
            "page_number": [1, 1],
        }
    )
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = df
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = [
        PaperSection(0, "Abstract", 2, None, CanonicalSection.ABSTRACT, 1.0),
        PaperSection(1, "1. Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
    ]
    contents.sentences = []
    llm_client = mock.MagicMock()
    llm_client.extract_core_metadata = mock.AsyncMock(return_value=llm_metadata)
    return MetadataExtractor(contents, llm_client=llm_client)


class TestPlaceholderDegradeNotCrash:
    async def test_omitted_title_completes_with_empty_title(self):
        ext = _build_extractor(
            CoreMetadataLLM(
                authors=[AuthorLLM(given="A", family="B")],
                keywords=["emotion"],
            )
        )
        await ext.extract_core_metadata()

        m = ext.metadata
        assert m is not None
        assert m.title == ""

    async def test_placeholder_title_completes_with_empty_title(self):
        # Placeholder title is scrubbed to "" at schema validation, so
        # strip_affiliation_markers("") does not raise and extraction completes.
        ext = _build_extractor(
            CoreMetadataLLM(
                title="verbatim-string",
                authors=[AuthorLLM(given="A", family="B")],
                keywords=["emotion"],
            )
        )
        await ext.extract_core_metadata()

        m = ext.metadata
        assert m is not None
        assert m.title == ""

    async def test_placeholder_author_name_parts_survive_as_empty(self):
        # given/family placeholders scrub to "" (non-Optional PaperAuthor fields);
        # the author survives — matching the existing empty-name handling, which
        # does NOT drop authors with empty name parts.
        ext = _build_extractor(
            CoreMetadataLLM(
                title="A Study",
                authors=[
                    AuthorLLM(given="verbatim-string", family="Doe", affiliation="Integer"),
                    AuthorLLM(given="Jane", family="string"),
                ],
                keywords=[],
            )
        )
        await ext.extract_core_metadata()

        m = ext.metadata
        assert m is not None
        assert len(m.authors) == 2
        assert m.authors[0].given == ""
        assert m.authors[0].family == "Doe"
        assert m.authors[0].affiliation == ""
        assert m.authors[1].given == "Jane"
        assert m.authors[1].family == ""
