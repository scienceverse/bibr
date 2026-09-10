"""Observability: a 0-author extraction on a non-notice paper must surface a
machine-greppable processing_warning instead of failing silently."""

from unittest import mock

import pandas as pd

from bibr.extract.extractor import EMPTY_AUTHORS_WARNING_PREFIX, MetadataExtractor
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection
from bibr.schemas import AuthorLLM, AuthorsLLM, CoreMetadataLLM


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
    contents.processing_warnings = []
    llm_client = mock.MagicMock()
    llm_client.extract_core_metadata = mock.AsyncMock(return_value=llm_metadata)
    # The extractor-owned empty-author recovery issues one free-JSON
    # extract_authors call; keep it empty so the warning path stays exercised.
    llm_client.extract_authors = mock.AsyncMock(return_value=AuthorsLLM(authors=[]))
    return MetadataExtractor(contents, llm_client=llm_client)


async def test_empty_authors_emits_processing_warning():
    llm_result = CoreMetadataLLM(
        title="A Normal Research Title",
        abstract="Some real abstract.",
        keywords=["k"],
        authors=[],
        paper_type="empirical",
    )
    ext = _build_extractor(llm_result)
    await ext.extract_core_metadata()

    assert ext.metadata.authors == []
    assert any(
        w.startswith(EMPTY_AUTHORS_WARNING_PREFIX) for w in ext.contents.processing_warnings
    ), ext.contents.processing_warnings


async def test_populated_authors_emits_no_warning():
    llm_result = CoreMetadataLLM(
        title="A Normal Research Title",
        abstract="Some real abstract.",
        keywords=["k"],
        authors=[AuthorLLM(given="Charles", family="Hulme")],
        paper_type="empirical",
    )
    ext = _build_extractor(llm_result)
    await ext.extract_core_metadata()

    assert len(ext.metadata.authors) == 1
    assert not any(
        w.startswith(EMPTY_AUTHORS_WARNING_PREFIX) for w in ext.contents.processing_warnings
    )


async def test_correction_notice_empty_authors_emits_no_warning():
    # The corrigendum guard legitimately empties authors — that is expected,
    # not the silent-drop defect, so it must NOT raise the empty-author warning.
    llm_result = CoreMetadataLLM(
        title="Corrigendum: Causal Inference About Good and Bad Outcomes",
        abstract="Because of a data-entry error...",
        keywords=["causal inference"],
        authors=[AuthorLLM(given="H. M.", family="Dorfman")],
        paper_type="commentary",
    )
    ext = _build_extractor(llm_result)
    await ext.extract_core_metadata()

    assert ext.metadata.authors == []  # guard cleared them
    assert not any(
        w.startswith(EMPTY_AUTHORS_WARNING_PREFIX) for w in ext.contents.processing_warnings
    )


def _resolution_with_byline(byline_text: str):
    """Selected front-matter block holding one title and one byline candidate."""
    from bibr.extract.front_matter import (
        FrontMatterBlock,
        FrontMatterCandidate,
        FrontMatterResolution,
    )

    def candidate(cid: str, text: str, roles: frozenset[str], order: int):
        return FrontMatterCandidate(
            candidate_id=cid,
            source_kind="paragraph",
            reading_order=order,
            page=1,
            bbox=None,
            region_label="text",
            font_size=None,
            font_bold=None,
            section_id=1,
            text_ids=(),
            paragraph_id=None,
            raw_text=text,
            normalized_text=" ".join(text.casefold().split()),
            roles=roles,
        )

    # The affiliation row matters: render_author_context spans title..byline and
    # keeps trailing non-title/abstract rows, so authors_text carries the
    # affiliation too. Without it the author zone collapses to exactly the byline
    # and the test could not tell the two context sources apart.
    candidates = (
        candidate("c1", "A Normal Research Title", frozenset({"title"}), 1),
        candidate("c2", byline_text, frozenset({"byline"}), 2),
        candidate("c3", "Department of Examples, Utrecht", frozenset({"affiliation"}), 3),
    )
    return FrontMatterResolution(
        candidates=candidates,
        blocks=(
            FrontMatterBlock(
                block_id="selected",
                candidate_ids=("c1", "c2", "c3"),
                title_candidate_ids=("c1",),
            ),
        ),
        selected_block_id="selected",
        selection_method="unique_block",
        reason_flags=(),
        allowed_text_ids=frozenset(),
        allowed_section_ids=frozenset({1}),
    )


async def test_empty_author_recovery_is_regrounded_on_the_selected_byline():
    """The retry must not re-ask with the same wide context that just failed."""
    llm_result = CoreMetadataLLM(
        title="A Normal Research Title",
        abstract="Some real abstract.",
        keywords=["k"],
        authors=[],
        paper_type="empirical",
    )
    ext = _build_extractor(llm_result)
    ext.core._front_matter_resolution = _resolution_with_byline("Fabian Vexler and Anika Q Mboro")
    ext.core.llm_client.extract_authors = mock.AsyncMock(
        return_value=AuthorsLLM(
            authors=[
                AuthorLLM(given="Fabian", family="Vexler"),
                AuthorLLM(given="Anika Q", family="Mboro"),
            ]
        )
    )

    await ext.extract_core_metadata()

    context = ext.core.llm_client.extract_authors.await_args.args[0]
    # Byline alone — NOT the wider author zone, which also holds the affiliation.
    assert context == "Fabian Vexler and Anika Q Mboro"
    assert "Utrecht" not in context
    assert [(a.given, a.family) for a in ext.metadata.authors] == [
        ("Fabian", "Vexler"),
        ("Anika Q", "Mboro"),
    ]


async def test_nameless_llm_entries_still_trigger_recovery():
    """A truthy raw list that sanitises to zero must not suppress the retry."""
    llm_result = CoreMetadataLLM(
        title="A Normal Research Title",
        abstract="Some real abstract.",
        keywords=["k"],
        # Sanitises to zero: no name at all, and an organization fragment.
        authors=[
            AuthorLLM(given="", family=""),
            AuthorLLM(given="", family="Department of Examples", role=["organization"]),
        ],
        paper_type="empirical",
    )
    ext = _build_extractor(llm_result)
    ext.core._front_matter_resolution = _resolution_with_byline("Fabian Vexler")
    ext.core.llm_client.extract_authors = mock.AsyncMock(
        return_value=AuthorsLLM(authors=[AuthorLLM(given="Fabian", family="Vexler")])
    )

    await ext.extract_core_metadata()

    ext.core.llm_client.extract_authors.assert_awaited_once()
    assert [(a.given, a.family) for a in ext.metadata.authors] == [("Fabian", "Vexler")]


async def test_sanitizer_wipeout_is_reported_with_the_clause_that_fired():
    """A total wipeout warns even below the drop threshold, naming the clause."""
    llm_result = CoreMetadataLLM(
        title="A Normal Research Title",
        abstract="Some real abstract.",
        keywords=["k"],
        authors=[AuthorLLM(given="", family="Some Institute", role=["organization"])],
        paper_type="empirical",
    )
    ext = _build_extractor(llm_result)

    await ext.extract_core_metadata()

    wipeout = [
        w
        for w in ext.contents.processing_warnings
        if "sanitizer dropped all" in w and "organization_fragment=1" in w
    ]
    assert wipeout, ext.contents.processing_warnings


async def test_degraded_recovery_is_distinguishable_from_a_real_abstention():
    """An upstream failure inside the recovery must not look like an abstention.

    bibr fails closed, so the empty list is the right *value*; the defect is
    that a consumer cannot tell a refusal from a swallowed error."""
    from bibr.exceptions import UpstreamServiceError

    llm_result = CoreMetadataLLM(
        title="A Normal Research Title",
        abstract="Some real abstract.",
        keywords=["k"],
        authors=[],
        paper_type="empirical",
    )
    ext = _build_extractor(llm_result)
    ext.core._front_matter_resolution = _resolution_with_byline("Fabian Vexler")
    ext.core.llm_client.extract_authors = mock.AsyncMock(
        side_effect=UpstreamServiceError("LLM", "Failed to extract authors")
    )

    await ext.extract_core_metadata()

    assert ext.metadata.authors == []  # fail closed: no fabricated byline
    assert "VAL_AUTHOR_RECOVERY_DEGRADED" in {issue.code for issue in ext.core.validation_issues}, (
        ext.core.validation_issues
    )


async def test_degraded_core_metadata_call_is_distinguishable_from_a_refusal():
    """``_call_core_llm``'s broad-except empties the whole record silently."""
    ext = _build_extractor(
        CoreMetadataLLM(
            title="A Normal Research Title",
            abstract="Some real abstract.",
            keywords=["k"],
            authors=[AuthorLLM(given="Charles", family="Hulme")],
            paper_type="empirical",
        )
    )
    # A heterogeneous, non-typed LLM failure: swallowed by design, value-wise.
    ext.core.llm_client.extract_core_metadata = mock.AsyncMock(
        side_effect=RuntimeError("provider returned a malformed envelope")
    )

    await ext.extract_core_metadata()

    assert ext.metadata.authors == []
    assert not ext.metadata.title
    assert "VAL_CORE_METADATA_DEGRADED" in {issue.code for issue in ext.core.validation_issues}, (
        ext.core.validation_issues
    )


async def test_clean_recovery_abstention_is_not_marked_degraded():
    """A recovery that ran and legitimately found nothing stays unmarked."""
    llm_result = CoreMetadataLLM(
        title="A Normal Research Title",
        abstract="Some real abstract.",
        keywords=["k"],
        authors=[],
        paper_type="empirical",
    )
    ext = _build_extractor(llm_result)
    ext.core._front_matter_resolution = _resolution_with_byline("Fabian Vexler")
    ext.core.llm_client.extract_authors = mock.AsyncMock(return_value=AuthorsLLM(authors=[]))

    await ext.extract_core_metadata()

    assert ext.metadata.authors == []
    assert "VAL_AUTHOR_RECOVERY_DEGRADED" not in {
        issue.code for issue in ext.core.validation_issues
    }


async def test_empty_author_recovery_keeps_wide_context_without_a_byline():
    """No name-like byline entry → fall back to the previous behaviour."""
    llm_result = CoreMetadataLLM(
        title="A Normal Research Title",
        abstract="Some real abstract.",
        keywords=["k"],
        authors=[],
        paper_type="empirical",
    )
    ext = _build_extractor(llm_result)
    await ext.extract_core_metadata()

    context = ext.core.llm_client.extract_authors.await_args.args[0]
    assert context != ""
    assert "Fabian" not in context
