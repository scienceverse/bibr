"""Integration tests for FrontMatterResolution + CoreMetadataExtractor wiring of ORCID and correspondence."""

from __future__ import annotations

from unittest import mock

from bibr.config import snapshot_settings
from bibr.extract.core_metadata import (
    CoreMetadataExtractor,
    render_author_context,
    render_block_context,
)
from bibr.extract.front_matter import (
    FrontMatterBlock,
    FrontMatterCandidate,
    FrontMatterResolution,
    resolve_front_matter,
)
from bibr.front_tagger.assemble import TaggedFrontMatter
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence
from bibr.schemas import AuthorLLM, AuthorsLLM, CoreMetadataLLM, TitleKeywordsLLM


def _section(
    section_id: int,
    header: str,
    section_type: CanonicalSection,
    *,
    synthetic_kind: str | None = None,
    classification_source: str | None = None,
) -> PaperSection:
    return PaperSection(
        section_id=section_id,
        header=header,
        level=1,
        parent_section_id=None,
        section_type=section_type,
        synthetic_kind=synthetic_kind,
        classification_source=classification_source,
    )


def _contents(sections: list[PaperSection], sentences: list[PaperSentence]) -> PaperContents:
    return PaperContents(
        sections=sections,
        sentences=sentences,
        tables=[],
        links=[],
        sections_text="\n".join(s.text for s in sentences),
    )


def _candidate(
    cid: str,
    text: str,
    *,
    roles: frozenset[str] = frozenset(),
    page: int = 1,
    section_id: int | None = 1,
    text_ids: tuple[int, ...] = (),
) -> FrontMatterCandidate:
    return FrontMatterCandidate(
        candidate_id=cid,
        source_kind="paragraph",
        reading_order=int(cid.removeprefix("c")),
        page=page,
        bbox=None,
        region_label="text",
        font_size=None,
        font_bold=None,
        section_id=section_id,
        text_ids=text_ids,
        paragraph_id=None,
        raw_text=text,
        normalized_text=" ".join(text.casefold().split()),
        roles=roles,
    )


def _resolution(
    *candidates: FrontMatterCandidate,
    selected_ids: tuple[str, ...] | None = None,
    extra_blocks: tuple[FrontMatterBlock, ...] = (),
) -> FrontMatterResolution:
    sel_ids = selected_ids or tuple(c.candidate_id for c in candidates)
    sel_block = FrontMatterBlock(
        block_id="b1",
        candidate_ids=sel_ids,
        title_candidate_ids=tuple(
            c.candidate_id for c in candidates if c.candidate_id in sel_ids and "title" in c.roles
        ),
    )
    allowed = frozenset(tid for c in candidates if c.candidate_id in sel_ids for tid in c.text_ids)
    allowed_sections = frozenset(
        c.section_id for c in candidates if c.candidate_id in sel_ids and c.section_id is not None
    )
    return FrontMatterResolution(
        candidates=tuple(candidates),
        blocks=(sel_block, *extra_blocks),
        selected_block_id="b1",
        selection_method="unique_block",
        reason_flags=(),
        allowed_text_ids=allowed,
        allowed_section_ids=allowed_sections,
    )


# ---------------------------------------------------------------------------
# FrontMatterResolution: page-1 author-metadata heading regression (54cad3c3)
# ---------------------------------------------------------------------------


def test_front_matter_admits_page1_corresponding_author_heading_without_false_title():
    contents = _contents(
        sections=[
            _section(1, "Title", CanonicalSection.TITLE),
            _section(2, "Abstract", CanonicalSection.ABSTRACT),
            _section(
                3,
                "CORRESPONDING AUTHOR: Emie Lefebvre",
                CanonicalSection.FOOTNOTE,
                classification_source="exact_alias",
            ),
            _section(4, "Introduction", CanonicalSection.INTRODUCTION),
        ],
        sentences=[
            PaperSentence(
                text_id=1,
                text="Cognitive flexibility across the lifespan",
                section_id=1,
                paragraph_id=1,
                page_number=1,
            ),
            PaperSentence(
                text_id=2,
                text="Emie Lefebvre1, Mark Novak2",
                section_id=1,
                paragraph_id=2,
                page_number=1,
            ),
            PaperSentence(
                text_id=3,
                text="1 Department of Psychology, University of Example, Montreal, Canada",
                section_id=1,
                paragraph_id=3,
                page_number=1,
            ),
            PaperSentence(
                text_id=4,
                text="We tested cognitive flexibility in 200 participants.",
                section_id=2,
                paragraph_id=4,
                page_number=1,
            ),
            PaperSentence(
                text_id=5,
                text="CORRESPONDING AUTHOR: Emie Lefebvre, emie.lefebvre@example.edu",
                section_id=3,
                paragraph_id=5,
                page_number=1,
            ),
            PaperSentence(
                text_id=6,
                text="Introduction body text.",
                section_id=4,
                paragraph_id=6,
                page_number=2,
            ),
        ],
    )

    resolution, _ = resolve_front_matter(contents)

    assert resolution.selected_block_id is not None
    assert 5 in resolution.allowed_text_ids
    # Must remain a single front-matter block (the heading must not seed a second title record).
    assert len(resolution.blocks) == 1
    assert "Emie Lefebvre" in render_block_context(resolution)


# ---------------------------------------------------------------------------
# render_author_context: pre-title sidebar & post-abstract correspondence
# ---------------------------------------------------------------------------


def test_render_author_context_preserves_pre_title_sidebar_and_post_abstract_correspondence():
    sidebar = _candidate(
        "c0",
        "*CORRESPONDENCE\nJane Doe\njane.doe@example.edu",
        section_id=10,
        text_ids=(10,),
    )
    title = _candidate(
        "c1", "A Study of Perception", roles=frozenset({"title"}), section_id=1, text_ids=(1,)
    )
    byline = _candidate(
        "c2", "Jane Doe, Bob Smith", roles=frozenset({"byline"}), section_id=1, text_ids=(2,)
    )
    abstract = _candidate(
        "c3",
        "Abstract prose goes here.",
        roles=frozenset({"abstract"}),
        section_id=2,
        text_ids=(3,),
    )
    post_corr = _candidate(
        "c4",
        "Corresponding authors: Bob Smith, Department of Psych",
        section_id=11,
        text_ids=(11,),
    )
    resolution = _resolution(sidebar, title, byline, abstract, post_corr)

    author_ctx = render_author_context(resolution, full_text=render_block_context(resolution))

    assert "Jane Doe, Bob Smith" in author_ctx
    assert "*CORRESPONDENCE" in author_ctx
    assert "jane.doe@example.edu" in author_ctx
    assert "Corresponding authors: Bob Smith" in author_ctx
    assert "Abstract prose goes here." not in author_ctx


# ---------------------------------------------------------------------------
# CoreMetadataExtractor wiring under FrontMatterResolution != None
# ---------------------------------------------------------------------------


async def test_core_metadata_extractor_passes_footnotes_and_backmatter_orcids_and_harvests():
    contents = _contents(
        sections=[
            _section(1, "Title", CanonicalSection.TITLE),
            _section(2, "", CanonicalSection.FOOTNOTE, synthetic_kind="footnote"),
            _section(3, "ORCID", CanonicalSection.FOOTNOTE),
        ],
        sentences=[
            PaperSentence(
                text_id=1,
                text="Population Study of Health",
                section_id=1,
                paragraph_id=1,
                page_number=1,
            ),
            PaperSentence(
                text_id=2,
                text="Alice Vance*, Bob Mercer",
                section_id=1,
                paragraph_id=2,
                page_number=1,
            ),
            PaperSentence(
                text_id=3,
                text="*Correspondence: Alice Vance, opaque99@example.edu",
                section_id=2,
                paragraph_id=3,
                page_number=1,
            ),
            PaperSentence(
                text_id=4,
                text="Bob Mercer https://orcid.org/0000-0002-1825-0097",
                section_id=3,
                paragraph_id=4,
                page_number=9,
            ),
        ],
    )
    resolution = _resolution(
        _candidate(
            "c1",
            "Population Study of Health",
            roles=frozenset({"title"}),
            section_id=1,
            text_ids=(1,),
        ),
        _candidate(
            "c2",
            "Alice Vance*, Bob Mercer",
            roles=frozenset({"byline"}),
            section_id=1,
            text_ids=(2,),
        ),
    )
    # Footnote (text_id=3) and back-matter ORCID (text_id=4) are outside allowed_text_ids.
    assert 3 not in resolution.allowed_text_ids
    assert 4 not in resolution.allowed_text_ids

    captured: dict[str, str | None] = {}

    async def fake_extract(text, *, authors_text=None, classification_text=None, **_kwargs):
        captured["authors_text"] = authors_text
        return CoreMetadataLLM(
            title="Population Study of Health",
            authors=[
                AuthorLLM(given="Alice", family="Vance", affiliation="Example Univ"),
                AuthorLLM(given="Bob", family="Mercer", affiliation="Example Univ"),
            ],
            keywords=[],
        )

    llm_client = mock.MagicMock()
    llm_client.extract_core_metadata = mock.AsyncMock(side_effect=fake_extract)

    extractor = CoreMetadataExtractor(
        contents, llm_client=llm_client, front_matter_resolution=resolution
    )
    metadata = await extractor.extract()

    # LLM authors_text received both the page-1 correspondence footnote and back-matter ORCIDs.
    assert captured["authors_text"] is not None
    assert "*Correspondence: Alice Vance, opaque99@example.edu" in captured["authors_text"]
    assert "0000-0002-1825-0097" in captured["authors_text"]

    # Harvesters backfilled email, corresponding flag, and ORCID even though the LLM omitted them.
    alice, bob = metadata.authors
    assert alice.corresponding is True
    assert alice.email == "opaque99@example.edu"
    assert bob.corresponding is False
    assert bob.orcid == "https://orcid.org/0000-0002-1825-0097"
    authors_decision = metadata._field_decisions.get("author")
    assert authors_decision is not None and authors_decision.selected is not None
    assert "emails_harvested" in authors_decision.selected.transforms
    assert "orcids_harvested" in authors_decision.selected.transforms


async def test_multi_record_front_page_does_not_leak_page1_correspondence_footnotes():
    contents = _contents(
        sections=[
            _section(1, "Title", CanonicalSection.TITLE),
            _section(2, "", CanonicalSection.FOOTNOTE, synthetic_kind="footnote"),
        ],
        sentences=[
            PaperSentence(
                text_id=1, text="First Paper", section_id=1, paragraph_id=1, page_number=1
            ),
            PaperSentence(
                text_id=2,
                text="Alice Vance, Bob Mercer",
                section_id=1,
                paragraph_id=2,
                page_number=1,
            ),
            PaperSentence(
                text_id=3,
                text="*Correspondence: Alice Vance, alice@example.edu",
                section_id=2,
                paragraph_id=3,
                page_number=1,
            ),
        ],
    )
    c1 = _candidate(
        "c1", "First Paper", roles=frozenset({"title"}), page=1, section_id=1, text_ids=(1,)
    )
    c2 = _candidate(
        "c2",
        "Alice Vance, Bob Mercer",
        roles=frozenset({"byline"}),
        page=1,
        section_id=1,
        text_ids=(2,),
    )
    c_other = _candidate(
        "c3",
        "Second Paper on Same Page",
        roles=frozenset({"title"}),
        page=1,
        section_id=9,
        text_ids=(9,),
    )
    resolution = _resolution(
        c1,
        c2,
        c_other,
        selected_ids=("c1", "c2"),
        extra_blocks=(
            FrontMatterBlock(block_id="b2", candidate_ids=("c3",), title_candidate_ids=("c3",)),
        ),
    )

    llm_client = mock.MagicMock()
    llm_client.extract_core_metadata = mock.AsyncMock(
        return_value=CoreMetadataLLM(
            title="First Paper",
            authors=[
                AuthorLLM(given="Alice", family="Vance", affiliation=""),
                AuthorLLM(given="Bob", family="Mercer", affiliation=""),
            ],
            keywords=[],
        )
    )
    extractor = CoreMetadataExtractor(
        contents, llm_client=llm_client, front_matter_resolution=resolution
    )
    metadata = await extractor.extract()

    # On a multi-record page, page-1 footnotes cannot be safely assigned to one record.
    assert extractor._front_page_correspondence_footnotes("") == []
    assert [a.corresponding for a in metadata.authors] == [False, False]


async def test_fast_mode_front_tagger_only_still_runs_orcid_harvester():
    contents = _contents(
        sections=[
            _section(1, "Title", CanonicalSection.TITLE),
            _section(2, "Author ORCIDs", CanonicalSection.FOOTNOTE),
        ],
        sentences=[
            PaperSentence(
                text_id=1, text="Fast Mode Study", section_id=1, paragraph_id=1, page_number=1
            ),
            PaperSentence(
                text_id=2, text="Alice Vance", section_id=1, paragraph_id=2, page_number=1
            ),
            PaperSentence(
                text_id=3,
                text="Alice Vance https://orcid.org/0000-0002-1825-0097",
                section_id=2,
                paragraph_id=3,
                page_number=8,
            ),
        ],
    )
    contents.front_tagger_result = TaggedFrontMatter(
        title_keywords=TitleKeywordsLLM(title="Fast Mode Study", keywords=[]),
        byline=AuthorsLLM(
            authors=[AuthorLLM(given="Alice", family="Vance", affiliation="Example Univ")]
        ),
        confidence={"title": 0.99, "authors": 0.99},
    )
    settings = snapshot_settings()
    settings.ml.front_tagger_mode = "only"
    resolution = _resolution(
        _candidate(
            "c1", "Fast Mode Study", roles=frozenset({"title"}), section_id=1, text_ids=(1,)
        ),
        _candidate("c2", "Alice Vance", roles=frozenset({"byline"}), section_id=1, text_ids=(2,)),
    )

    extractor = CoreMetadataExtractor(
        contents,
        llm_client=mock.MagicMock(),
        settings=settings,
        front_matter_resolution=resolution,
    )
    metadata = await extractor.extract()

    assert len(metadata.authors) == 1
    assert metadata.authors[0].orcid == "https://orcid.org/0000-0002-1825-0097"
    authors_decision = metadata._field_decisions.get("author")
    assert authors_decision is not None and authors_decision.selected is not None
    assert "orcids_harvested" in authors_decision.selected.transforms
