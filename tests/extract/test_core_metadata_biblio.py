"""The paper's own bibliographic self-identity flows CoreMetadataLLM → PaperMetadata."""

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


def _llm_result(**biblio) -> CoreMetadataLLM:
    return CoreMetadataLLM(
        title="Emotional Vocalizations Are Recognized Across Cultures",
        abstract="We tested recognition across cultures.",
        keywords=["emotion"],
        authors=[AuthorLLM(given="A", family="B")],
        oecd_domain="Social Sciences",
        oecd_subdomain="Psychology",
        paper_type="empirical",
        **biblio,
    )


class TestBiblioMapping:
    async def test_biblio_fields_mapped(self):
        ext = _build_extractor(
            _llm_result(
                journal="Psychological Science",
                volume="31",
                issue="1",
                first_page="65",
                last_page="74",
                issn="0956-7976",
                publisher="SAGE Publications",
                published="2020-01-01",
                license="CC BY 4.0",
            )
        )
        await ext.extract_core_metadata()

        m = ext.metadata
        assert m is not None
        assert m.journal == "Psychological Science"
        assert m.volume == "31"
        assert m.issue == "1"
        assert m.first_page == "65"
        assert m.last_page == "74"
        assert m.issn == "0956-7976"
        assert m.publisher == "SAGE Publications"
        assert m.published == "2020-01-01"
        assert m.license == "CC BY 4.0"

    async def test_absent_biblio_is_none(self):
        ext = _build_extractor(_llm_result())
        await ext.extract_core_metadata()

        m = ext.metadata
        assert m is not None
        assert m.journal is None
        assert m.volume is None
        assert m.publisher is None
        assert m.published is None
        assert m.license is None
