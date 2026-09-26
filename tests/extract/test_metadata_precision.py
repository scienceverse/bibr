"""Source-backed repairs use synthetic papers, not benchmark-specific names/dates."""

from unittest import mock

import pandas as pd
import pytest

from bibr.extract.core_metadata import CoreMetadataExtractor
from bibr.extract.front_matter import (
    FrontMatterBlock,
    FrontMatterCandidate,
    FrontMatterResolution,
)
from bibr.models import PaperAuthor
from bibr.paper_contents import PaperContents
from bibr.schemas import AuthorLLM, CoreMetadataLLM


def _extractor(*, published="2020", given="Mira Ellison", family="", outside="", date_text=None):
    texts = [
        "Measuring tides with acoustic sensors",
        "Mira Ellison 1; Correspondence: m.ellison@example.org",
        date_text
        if date_text is not None
        else "Received: 8 January 2020 / Accepted: 2 April 2020 / Published online: 11 April 2020",
    ]
    candidates = tuple(
        FrontMatterCandidate(
            candidate_id=f"c{i}",
            source_kind="paragraph",
            reading_order=i,
            page=1,
            bbox=None,
            region_label="text",
            font_size=None,
            font_bold=None,
            section_id=0,
            text_ids=(i,),
            paragraph_id=i,
            raw_text=text,
            normalized_text=text.casefold(),
            roles=frozenset(roles),
        )
        for i, (text, roles) in enumerate(
            zip(texts + [outside], ({"title"}, {"byline"}, set(), set()), strict=True), 1
        )
    )
    resolution = FrontMatterResolution(
        candidates=candidates,
        blocks=(FrontMatterBlock("selected", ("c1", "c2", "c3"), ("c1",)),),
        selected_block_id="selected",
        selection_method="unique_block",
        reason_flags=(),
        allowed_text_ids=frozenset({1, 2, 3}),
        allowed_section_ids=frozenset({0}),
    )
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = pd.DataFrame(
        {
            "text_id": [1, 2, 3, 4],
            "text": texts + [outside],
            "page_number": [1] * 4,
            "section_name": ["Front matter"] * 4,
        }
    )
    contents.sentences = []
    contents.sections = []
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.processing_warnings = []
    llm_result = CoreMetadataLLM(
        title=texts[0],
        keywords=[],
        published=published,
        authors=[AuthorLLM(given=given, family=family, email="m.ellison@example.org")],
    )
    client = mock.MagicMock()
    client.extract_core_metadata = mock.AsyncMock(return_value=llm_result)
    extractor = CoreMetadataExtractor(
        contents,
        llm_client=client,
        front_matter_resolution=resolution,
    )
    return extractor, llm_result


async def test_extract_preserves_printed_publication_date_precision():
    extractor, raw = _extractor()
    metadata = await extractor.extract()
    assert metadata.published == "2020-04-11"
    assert raw.published == "2020"
    assert any(
        issue.code == "VAL_PUBLICATION_DATE_REFINED" for issue in extractor.validation_issues
    )


async def test_extract_repairs_whole_given_from_printed_name_and_email():
    extractor, raw = _extractor()
    metadata = await extractor.extract()
    assert (metadata.authors[0].given, metadata.authors[0].family) == ("Mira", "Ellison")
    assert (raw.authors[0].given, raw.authors[0].family) == ("Mira Ellison", "")
    assert any(
        issue.code == "VAL_AUTHOR_PARTITION_REPAIRED" for issue in extractor.validation_issues
    )


async def test_extract_ignores_conflicting_date_outside_selected_record():
    extractor, _ = _extractor(outside="Published online: 19 August 2020")
    metadata = await extractor.extract()
    assert metadata.published == "2020-04-11"


async def test_extract_cannot_borrow_date_from_other_record():
    extractor, _ = _extractor(date_text="Copyright 2020", outside="Published: 19 August 2020")
    metadata = await extractor.extract()
    assert metadata.published == "2020"


@pytest.mark.parametrize(
    "printed, expected",
    [
        ("Published online: 11 April 2020", "2020-04-11"),
        ("Online Published: April 11, 2020", "2020-04-11"),
        ("Publication date: 2020-04-11", "2020-04-11"),
        ("First published on 11 Apr. 2020", "2020-04-11"),
        ("Published online:\n11 April 2020", "2020-04-11"),
        (
            "Received: April 1, 2020 Accepted: April 8, 2020 Online Published: April 11, 2020",
            "2020-04-11",
        ),
        ("Accepted: 2 April 2020 / Published: 11 April 2020", "2020-04-11"),
        ("Published: 29 February 2020", "2020-02-29"),
        ("Published: 11 April 2020\nPublished online: 11 April 2020", "2020-04-11"),
        ("Published: 11 April 2020 / Published online: 11 April 2020", "2020-04-11"),
        ("Received: 11 April 2020 / Accepted: 12 April 2020", "2020"),
        ("Revised: April 11, 2020", "2020"),
        ("Copyright 2020, Publisher. 11 April 2020", "2020"),
        ("Earlier research was published online: 11 April 2020", "2020"),
        ("Published: 31 April 2020", "2020"),
        ("Published: 2020-13-01", "2020"),
        ("Published: 11 April 2020s", "2020"),
        ("Published: 11/04/2020", "2020"),
        ("Published: 11 April 2019", "2020"),
        ("Published: 11 April 2020\nPublished online: 10 April 2020", "2020"),
        ("Published: 11 April 2020 / Published online: 10 April 2020", "2020"),
        ("Published: 11 April 2020\nPublished online: 31 April 2020", "2020"),
    ],
)
def test_date_evidence_and_ambiguity(printed, expected):
    from bibr.extract.metadata_precision import refine_publication_date

    assert refine_publication_date("2020", printed) == expected


@pytest.mark.parametrize(
    "boilerplate",
    [
        "Published by Elsevier Ltd.",
        "Published under a CC BY 4.0 license.",
        "Published in partnership with the Society.",
        "Published as part of the proceedings.",
    ],
)
@pytest.mark.parametrize("position", ["before", "after"])
def test_publisher_boilerplate_does_not_block_refinement(boilerplate, position):
    """extract-metadata-15: 'Published by/under/...' carries no date, so it
    must not veto refinement from a labelled date elsewhere in the block."""
    from bibr.extract.metadata_precision import refine_publication_date

    history = "Received: 2 January 2021 / Accepted: 1 March 2021 / Published online: 12 March 2021"
    source = f"{boilerplate}\n{history}" if position == "before" else f"{history}\n{boilerplate}"
    assert refine_publication_date("2021-03", source) == "2021-03-12"


def test_boilerplate_alone_leaves_date_unchanged():
    """Boilerplate with no labelled date anywhere refines nothing."""
    from bibr.extract.metadata_precision import refine_publication_date

    assert refine_publication_date("2021-03", "Published by Elsevier Ltd.") == "2021-03"


def test_unparseable_date_like_tail_still_blocks_refinement():
    """A date-like tail that fails to parse ('31 April') stays ambiguous even
    when a valid labelled date follows — only non-date boilerplate is skipped."""
    from bibr.extract.metadata_precision import refine_publication_date

    source = "Published: 31 April 2020\nPublished online: 11 April 2020"
    assert refine_publication_date("2020", source) == "2020"


def test_month_first_unparseable_tail_still_blocks_refinement():
    """A month-first tail with no day ('March 2021') is date-like: it fails
    to parse and keeps the record ambiguous even with a valid labelled date
    next to it. A digits-only date check would misread it as boilerplate."""
    from bibr.extract.metadata_precision import refine_publication_date

    source = "Published: March 2021\nPublished online: 11 April 2020"
    assert refine_publication_date("2020", source) == "2020"


@pytest.mark.parametrize(
    "published, expected",
    [
        (None, None),
        ("", ""),
        ("2020-05", "2020-05"),
        ("2020-04", "2020-04-11"),
        ("2020-04-01", "2020-04-01"),
        ("2019", "2019"),
        ("2020-00", "2020-00"),
    ],
)
def test_only_compatible_partial_dates_are_refined(published, expected):
    from bibr.extract.metadata_precision import refine_publication_date

    assert refine_publication_date(published, "Published: 11 April 2020") == expected


def _person(given, family="", email="m.ellison@example.org", **kwargs):
    return PaperAuthor(
        author_id=1, given=given, family=family, email=email, affiliation="Lab", **kwargs
    )


@pytest.mark.parametrize(
    "name, email, expected",
    [
        ("Mira Ellison", "m.ellison@example.org", ("Mira", "Ellison")),
        ("Ana Silva Costa", "a.silva.costa@example.org", ("Ana", "Silva Costa")),
        ("Ana Maria Silva Costa", "am.silvacosta@example.org", ("Ana Maria", "Silva Costa")),
        ("Lena van der Meer", "l.vandermeer@example.org", ("Lena", "van der Meer")),
        ("Jean-Pierre O’Neill", "j.oneill@example.org", ("Jean-Pierre", "O’Neill")),
        ("Éva Kovács", "e.kovacs@example.org", ("Éva", "Kovács")),
        ("Chen Mei", "m.chen@example.org", ("Mei", "Chen")),
        ("Ellison Mira", "ellison.m@example.org", ("Mira", "Ellison")),
    ],
)
def test_unique_email_partition_conserves_names_and_other_fields(name, email, expected):
    from bibr.extract.metadata_precision import repair_author_partitions

    author = _person(name, email=email, corresponding=True, orcid="0000-0002-1825-0097")
    before = author.model_dump(exclude={"given", "family"})
    issues = repair_author_partitions([author], f"{name} 1\nCorrespondence: {email}")
    assert (author.given, author.family) == expected
    assert author.model_dump(exclude={"given", "family"}) == before
    assert len(issues) == 1
    assert "author:1" in issues[0].evidence_ids


@pytest.mark.parametrize(
    "author, source",
    [
        (_person("Mira", "Ellison"), "Mira Ellison m.ellison@example.org"),
        (_person("Mira Ellison", "Wong"), "Mira Ellison Wong m.ellison@example.org"),
        (_person("Mira Ellison"), "Someone Else m.ellison@example.org"),
        (_person("Mira Ellison"), "Mira Ellison"),
        (
            _person("Mira Ellison", email="mira.ellison@example.org"),
            "Mira Ellison mira.ellison@example.org",
        ),
        (
            _person("Mira Ellison", email="mellison@example.org"),
            "Mira Ellison mellison@example.org",
        ),
        (
            _person("Mira Ellison", email="m.ellison7@example.org"),
            "Mira Ellison m.ellison7@example.org",
        ),
        (_person("Mira Ellison", email=None), "Mira Ellison m.ellison@example.org"),
        (_person("Sukarno"), "Sukarno m.ellison@example.org"),
        (_person("M. E.", email="m.e@example.org"), "M. E. m.e@example.org"),
        (_person("Mira Ellison", role=["organization"]), "Mira Ellison m.ellison@example.org"),
        (_person("Research Team", email="r.team@example.org"), "Research Team r.team@example.org"),
        (
            _person("Mira Ellison", email="m.ellison@example.org"),
            "Mira Ellison xm.ellison@example.org",
        ),
    ],
)
def test_incomplete_or_ambiguous_evidence_never_guesses(author, source):
    from bibr.extract.metadata_precision import repair_author_partitions

    before = author.model_dump()
    assert repair_author_partitions([author], source) == ()
    assert author.model_dump() == before


def test_shared_initial_and_surname_is_ambiguous():
    from bibr.extract.metadata_precision import repair_author_partitions

    authors = [_person("Mira Ellison"), _person("Mark", "Ellison", email=None)]
    before = [author.model_dump() for author in authors]
    assert (
        repair_author_partitions(authors, "Mira Ellison; Mark Ellison; m.ellison@example.org") == ()
    )
    assert [author.model_dump() for author in authors] == before


@pytest.mark.parametrize(
    "name,email",
    [
        ("Ana Silva Costa", "a.costa@example.org"),
        ("Lena van der Meer", "l.meer@example.org"),
        ("Ana Maria Silva Costa", "a.silvacosta@example.org"),
    ],
)
def test_email_cannot_move_omitted_name_words_into_given(name, email):
    from bibr.extract.metadata_precision import repair_author_partitions

    author = _person(name, email=email)
    before = author.model_dump()
    assert repair_author_partitions([author], f"{name} {email}") == ()
    assert author.model_dump() == before
