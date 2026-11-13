"""A long byline the front-matter tagger read is kept whole.

The 64-author cap stands against a generative model looping on its own output.
The tagger labels printed tokens and cannot loop, so a 100-author many-labs
byline it read must not lose its last 36 authors to the cap.
"""

from unittest import mock

import pandas as pd

from bibr.clients.skipped import SkippedLlmClient
from bibr.config import GlobalSettings
from bibr.extract.core_metadata import _MAX_LLM_AUTHORS
from bibr.extract.extractor import MetadataExtractor
from bibr.front_tagger.assemble import TaggedFrontMatter
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, RegionSummary
from bibr.processing_warnings import WarningCode
from bibr.run_mode import fast_settings
from bibr.schemas import AuthorLLM, AuthorsLLM, CoreMetadataLLM, TitleKeywordsLLM

_COUNT = 100


def _names() -> list[tuple[str, str]]:
    return [
        (f"Given{chr(65 + i // 26)}{chr(65 + i % 26)}", f"Family{i:03d}") for i in range(_COUNT)
    ]


def _contents() -> PaperContents:
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = pd.DataFrame(
        {
            "section_name": ["Title", "Abstract"],
            "text": ["A Many Labs Replication", "Some abstract text."],
            "page_number": [1, 1],
        }
    )
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = [
        PaperSection(0, "Title", 1, None, CanonicalSection.TITLE, 1.0),
        PaperSection(1, "Abstract", 2, None, CanonicalSection.ABSTRACT, 1.0),
    ]
    contents.sentences = []
    contents.processing_warnings = []
    contents.region_summaries = [RegionSummary(page=1, index=0, label="text", bbox=None)]
    return contents


def _authors() -> AuthorsLLM:
    return AuthorsLLM(authors=[AuthorLLM(given=g, family=f) for g, f in _names()])


async def test_the_tagger_byline_is_not_cut_at_the_model_cap():
    assert _COUNT > _MAX_LLM_AUTHORS
    contents = _contents()
    contents.front_tagger_result = TaggedFrontMatter(
        title_keywords=TitleKeywordsLLM(title="A Many Labs Replication"),
        byline=_authors(),
        confidence={"title": 0.99, "authors": 0.99, "abstract": 0.99},
        problems=[],
    )
    ext = MetadataExtractor(
        contents, llm_client=SkippedLlmClient(), settings=fast_settings(GlobalSettings())
    )

    await ext.extract_core_metadata()

    assert [(a.given, a.family) for a in ext.metadata.authors] == _names()
    assert [a.author_id for a in ext.metadata.authors] == list(range(1, _COUNT + 1))
    assert WarningCode.AUTHORS_ANOMALY not in {w.code for w in contents.processing_warnings}


async def test_a_model_list_over_the_cap_is_still_trimmed():
    from unittest.mock import AsyncMock

    contents = _contents()
    ext = MetadataExtractor(contents, llm_client=SkippedLlmClient())
    ext.core._call_core_llm = AsyncMock(
        return_value=CoreMetadataLLM(title="A Many Labs Replication", authors=_authors().authors)
    )

    await ext.extract_core_metadata()

    assert len(ext.metadata.authors) == _MAX_LLM_AUTHORS
    assert WarningCode.AUTHORS_ANOMALY in {w.code for w in contents.processing_warnings}
