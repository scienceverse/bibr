"""Per-task context slices for the core-metadata LLM fan-out (LLM_PER_TASK_CONTEXT).

When ``Settings.llm.per_task_context`` is on, ``CoreMetadataExtractor.extract``
sends the authors call a page-1 + ORCID/correspondence slice and the
classification call a front-matter-through-abstract slice, while the
title/keywords call still receives the full front-matter blob. With the setting
off, behavior is byte-identical to today (all three get the full text / None).
"""

from unittest import mock

import pandas as pd

from bibr.config import Settings
from bibr.extract.core_metadata import CoreMetadataExtractor
from bibr.extract.extractor import MetadataExtractor
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection
from bibr.schemas import AuthorLLM, CoreMetadataLLM


def _make_core_extractor(paper_sections=None):
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = pd.DataFrame({"section_name": [], "text": []})
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = paper_sections or []
    contents.sentences = []
    return CoreMetadataExtractor(contents)


class TestBuildAuthorsText:
    def test_keeps_page1_orcid_and_corresponding_drops_body(self):
        meta_df = pd.DataFrame(
            {
                "section_name": ["Title", "Body", "Body", "Footnote"],
                "text": [
                    "Jane Byline, John Coauthor",
                    "Page-2 body sentence that should be excluded.",
                    "Extra ORCID footnote https://orcid.org/0000-0001-2345-6789",
                    "* Corresponding author: Jane Byline",
                ],
                "page_number": [1, 2, 5, 1],
            }
        )
        ext = _make_core_extractor()
        out = ext._build_authors_text(meta_df, full_text="FULL")
        assert "Jane Byline, John Coauthor" in out
        assert "orcid.org/0000-0001-2345-6789" in out
        assert "Corresponding author: Jane Byline" in out
        assert "Page-2 body sentence" not in out

    def test_keeps_bare_orcid_row(self):
        meta_df = pd.DataFrame(
            {
                "section_name": ["Title", "Body"],
                "text": ["Author One", "Bare orcid 0000-0002-5438-0665 on page 3"],
                "page_number": [1, 3],
            }
        )
        ext = _make_core_extractor()
        out = ext._build_authors_text(meta_df, full_text="FULL")
        assert "0000-0002-5438-0665" in out

    def test_fallback_to_full_text_when_no_page_column(self):
        meta_df = pd.DataFrame({"section_name": ["Title"], "text": ["Author One"]})
        ext = _make_core_extractor()
        out = ext._build_authors_text(meta_df, full_text="FULL BLOB")
        assert out == "FULL BLOB"

    def test_fallback_to_full_text_when_slice_empty(self):
        # No usable page numbers and no rescue matches -> empty slice -> full text.
        meta_df = pd.DataFrame(
            {
                "section_name": ["Body"],
                "text": ["Body sentence with no byline signal."],
                "page_number": [None],
            }
        )
        ext = _make_core_extractor()
        out = ext._build_authors_text(meta_df, full_text="FULL BLOB")
        assert out == "FULL BLOB"

    def test_byline_slice_anchors_on_the_lowest_page_present(self):
        """ "Page 1" means the first page in the frame, not the literal page 1.

        Page numbers are absolute, so under page slicing (``--pages 5-12``,
        serve ``start_page``) — and for a paper whose front matter starts after
        a textless cover page — nothing carries page 1. A literal comparison
        emptied the slice and dumped the whole document into the authors call,
        which is exactly what the per-task slices exist to avoid.
        """
        meta_df = pd.DataFrame(
            {
                "section_name": ["Title", "Body"],
                "text": ["Jane Doe, John Roe", "Body sentence on the next page."],
                "page_number": [4, 5],
            }
        )
        ext = _make_core_extractor()
        out = ext._build_authors_text(meta_df, full_text="FULL BLOB")

        assert "Jane Doe, John Roe" in out
        assert "Body sentence on the next page." not in out


class TestBuildClassificationText:
    def test_ends_at_last_abstract_row(self):
        meta_df = pd.DataFrame(
            {
                "section_name": ["Title", "Abstract", "Abstract", "Introduction"],
                "text": [
                    "A Title",
                    "Abstract first sentence.",
                    "Abstract last sentence.",
                    "Introduction body that must be excluded.",
                ],
            }
        )
        sections = [
            PaperSection(0, "Title", 1, None, CanonicalSection.UNKNOWN, 0.0),
            PaperSection(1, "Abstract", 1, None, CanonicalSection.ABSTRACT, 1.0),
            PaperSection(2, "Introduction", 1, None, CanonicalSection.INTRODUCTION, 1.0),
        ]
        ext = _make_core_extractor(paper_sections=sections)
        out = ext._build_classification_text(meta_df)
        assert "Abstract last sentence." in out
        assert "Introduction body that must be excluded." not in out

    def test_fallback_first_30_rows_when_no_abstract(self):
        meta_df = pd.DataFrame(
            {
                "section_name": ["Body"] * 40,
                "text": [f"Sentence {i}" for i in range(40)],
            }
        )
        sections = [PaperSection(0, "Body", 1, None, CanonicalSection.UNKNOWN, 0.0)]
        ext = _make_core_extractor(paper_sections=sections)
        out = ext._build_classification_text(meta_df)
        assert "Sentence 29" in out
        assert "Sentence 30" not in out


def _build_full_extractor(paper_sections, llm_metadata):
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
    contents.sections = paper_sections
    contents.sentences = []
    llm_client = mock.MagicMock()
    llm_client.extract_core_metadata = mock.AsyncMock(return_value=llm_metadata)
    return MetadataExtractor(contents, llm_client=llm_client), llm_client


_SECTIONS = [
    PaperSection(0, "Abstract", 1, None, CanonicalSection.ABSTRACT, 1.0),
    PaperSection(1, "1. Introduction", 1, None, CanonicalSection.INTRODUCTION, 1.0),
]


def _core_meta():
    return CoreMetadataLLM(
        title="A Study", authors=[AuthorLLM(given="A", family="B")], keywords=["k"]
    )


class TestSettingRouting:
    async def test_setting_off_passes_none(self, monkeypatch):
        monkeypatch.setattr(Settings.llm, "per_task_context", False)
        ext, llm_client = _build_full_extractor(_SECTIONS, _core_meta())
        await ext.extract_core_metadata()
        kwargs = llm_client.extract_core_metadata.await_args.kwargs
        assert kwargs.get("authors_text") is None
        assert kwargs.get("classification_text") is None

    async def test_setting_on_passes_slices(self, monkeypatch):
        monkeypatch.setattr(Settings.llm, "per_task_context", True)
        ext, llm_client = _build_full_extractor(_SECTIONS, _core_meta())
        await ext.extract_core_metadata()
        kwargs = llm_client.extract_core_metadata.await_args.kwargs
        assert kwargs.get("authors_text") is not None
        assert kwargs.get("classification_text") is not None
        # classification slice stops at the abstract, never reaches the intro.
        assert "Some intro text." not in kwargs["classification_text"]
