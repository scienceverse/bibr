"""Regression tests for Fix 1B reference-extraction short-circuits and indexing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from bibr.extract.ref_extractor import ReferenceExtractor, _distinct_count, _map_bib_text_ids
from bibr.extract.ref_line_stream import LineStream, StreamLine
from bibr.paper import PaperReference
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    ReferenceSegmentationAttempt,
)


def _dummy_contents(pdf_uri_links=None) -> PaperContents:
    return PaperContents(
        sections=[
            PaperSection(
                section_id=1,
                header="References",
                level=1,
                parent_section_id=None,
                section_type=CanonicalSection.REFERENCES,
            )
        ],
        sentences=[],
        tables=[],
        links=[],
        sections_text="",
        pdf_uri_links=pdf_uri_links or [],
    )


def test_geom_tagger_short_circuits_line_stream_without_doi_links() -> None:
    contents = _dummy_contents(
        pdf_uri_links=[{"page": 1, "uri": "https://example.org", "bbox": [0, 0, 10, 10]}]
    )
    ex = ReferenceExtractor(contents, llm_client=MagicMock())
    ex._segmentation_attempts.append(
        ReferenceSegmentationAttempt("geom_tagger", ((0, 20),), 1, True, ("final",))
    )
    ref = "Smith, J. (2020). A study title. Journal, 1, 1-2."
    with (
        patch("bibr.extract.ref_extractor.build_line_stream") as m_build,
        patch("bibr.extract.ref_extractor.stream_probabilities") as m_probs,
        patch("bibr.extract.ref_extractor.segment_line_stream") as m_seg,
        patch("bibr.extract.ref_extractor.segmentation_quality") as m_qual,
    ):
        choice = ex._line_stream_choice(pd.DataFrame(), ref, [ref])

    assert choice is not None
    assert choice.selected is False
    assert choice.ref_strings == [ref]
    assert choice.link_dois == [None]
    assert ex._segmentation_attempts[-1].reason_flags == ("tagger_final",)
    m_build.assert_not_called()
    m_probs.assert_not_called()
    m_seg.assert_not_called()
    m_qual.assert_not_called()


def test_geom_tagger_still_borrows_link_dois_without_running_gbm_or_quality() -> None:
    doi_links = [{"page": 1, "uri": "https://doi.org/10.1234/abc", "bbox": [10, 10, 90, 20]}]
    contents = _dummy_contents(pdf_uri_links=doi_links)
    ex = ReferenceExtractor(contents, llm_client=MagicMock())
    ex._segmentation_attempts.append(
        ReferenceSegmentationAttempt("geom_tagger", ((0, 20),), 1, True, ("final",))
    )
    ref = "Smith, J. (2020). A study title. Journal, 1, 1-2."
    stream = LineStream(
        lines=[
            StreamLine(
                text=ref,
                page=1,
                bbox=(10.0, 10.0, 90.0, 20.0),
                region=0,
                region_label="text",
                region_first=True,
                link_dois=["10.1234/abc"],
            )
        ]
    )
    with (
        patch("bibr.extract.ref_extractor.build_line_stream", return_value=stream) as m_build,
        patch("bibr.extract.ref_extractor.stream_probabilities") as m_probs,
        patch("bibr.extract.ref_extractor.segment_line_stream") as m_seg,
        patch("bibr.extract.ref_extractor.segmentation_quality") as m_qual,
    ):
        choice = ex._line_stream_choice(pd.DataFrame(), ref, [ref])

    assert choice is not None
    assert choice.selected is False
    assert choice.link_dois == ["10.1234/abc"]
    m_build.assert_called_once()
    m_probs.assert_not_called()
    m_seg.assert_not_called()
    m_qual.assert_not_called()


def test_distinct_count_buckets_by_year_set_exact() -> None:
    segs = [
        "Smith, J. (2020). Deep learning for science. Nature, 10, 1-10.",
        "Smith, J. (2020). Deep learning for science. Nature, 10, 1-10!",  # near-dup same year
        "Smith, J. (2021). Deep learning for science. Nature, 10, 1-10.",  # different year -> distinct
        "Jones, K. (2019). Another unrelated paper. Science, 5, 20-30.",
        "   ",  # empty key ignored
    ]
    assert _distinct_count(segs) == 3


def test_map_bib_text_ids_handles_duplicates_and_taken_rows() -> None:
    contents = PaperContents(
        sections=[
            PaperSection(
                section_id=1,
                header="References",
                level=1,
                parent_section_id=None,
                section_type=CanonicalSection.REFERENCES,
            )
        ],
        sentences=[
            PaperSentence(
                text_id=10,
                text="Smith (2020). Shared Title Here. J1.",
                section_id=1,
                paragraph_id=1,
            ),
            PaperSentence(
                text_id=20,
                text="Jones (2021). Shared Title Here. J2.",
                section_id=1,
                paragraph_id=2,
            ),
            PaperSentence(
                text_id=30,
                text="Brown (2019). Unique Paper Title. J3.",
                section_id=1,
                paragraph_id=3,
            ),
        ],
        tables=[],
        links=[],
        sections_text="",
    )
    refs = [
        PaperReference(
            bib_id=1,
            title="Shared Title Here",
            authors="Smith",
            year=2020,
            container="J1",
            volume="1",
            first_page="1",
        ),
        PaperReference(
            bib_id=2,
            title="Shared Title Here",
            authors="Jones",
            year=2021,
            container="J2",
            volume="2",
            first_page="2",
        ),
        PaperReference(
            bib_id=3,
            title="Unique Paper Title",
            authors="Brown",
            year=2019,
            container="J3",
            volume="3",
            first_page="3",
        ),
    ]
    _map_bib_text_ids(refs, contents)
    assert [r.text_id for r in refs] == [10, 20, 30]
